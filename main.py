import argparse
from model import LinearAttentionTransformerModel, diffusion_SA
from imnoise import *
import os, shutil
import json
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from utils import *
from data import get_dataloader
import time
from tqdm import tqdm

def train(local_rank:int, args):
    dist_url='tcp://localhost:13457'
    dist.init_process_group(backend='nccl', init_method=dist_url, world_size=args.ngpu, rank=local_rank)
    # assign log writer and checkpoint dir
    if local_rank==0:
        log_path = os.path.join(args.exp_dir, 'log')
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.mkdir(log_path)
        tb_writer = SummaryWriter(log_dir=log_path)
    if not os.path.exists(os.path.join(args.exp_dir, 'checkpoint')):
        os.mkdir(os.path.join(args.exp_dir, 'checkpoint'))
    # load data
    train_dataloader, dev_dataloader, test_dataloader = get_dataloader(args)
    # initialize model
    cfg = json.load(open(args.cfg_path, 'r'))
    model_cfg = cfg['model']
    denosing_model = LinearAttentionTransformerModel(**model_cfg)
    denosing_model.to(f'cuda:{local_rank}')
    denosing_model = DDP(denosing_model, device_ids=[local_rank], output_device=local_rank)
    imnoise_func = imnoise_multinomial if args.imnoise_method=='multinomial' else imnoise_bigram
    beta_schedule = linear_beta_schedule if args.beta_schedule=='linear' else cosine_beta_schedule
    diffusinSA = diffusion_SA(imnoise_func, denosing_model, timesteps=args.timesteps, beta_schedule=beta_schedule,
        use_cache=args.use_cache, dataset=train_dataloader.dataset)
    # save config
    if local_rank==0:
        cfg['others'] = vars(args)
        json.dump(cfg, open(os.path.join(args.exp_dir, 'config.json'), 'w'), indent=2)
    # initialize optimizer and scheduler
    if args.optimizer=='adam':
        optimizer = optim.Adam(denosing_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    total_steps = args.epochs*len(train_dataloader.dataset)//args.batch_size
    if local_rank==0:
        print('Total sentences:{}, batch size:{}, batch num for one gpu:{}, epochs:{}, total steps:{}'.format(
            len(train_dataloader.dataset), args.batch_size, len(train_dataloader), args.epochs, total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_ratio*total_steps, num_training_steps=total_steps)
    
    # init_eval
    if args.init_eval and local_rank==0:
        print('****Init Evaluation****')
        denosing_model.eval()
        st = time.time()
        count=0
        with torch.no_grad():
            eval_loss = 0
            for batch, batch_ids in tqdm(dev_dataloader):
                loss = diffusinSA.cal_loss(batch, mode='test', batch_ids=batch_ids)
                eval_loss += loss
                count +=1
                if count>=10:
                    break
        if local_rank==0:
            print('Init_eval_loss:', eval_loss)
        total_time = (time.time()-st)/60
        print('Init eval time:{:.2f}, dev loss:{:.3f}'.format(total_time, eval_loss))
        checkpoint_path = os.path.join(args.exp_dir, 'checkpoint', 'checkpoint_init.pt')
        # if resuming is needed, then optimizer and scheduler must be saved too
        torch.save(denosing_model.state_dict(), checkpoint_path)

    # training
    step = 0
    for epoch in range(args.epochs):
        training_loss = 0
        denosing_model.train()
        st = time.time()
        for batch, batch_ids in tqdm(train_dataloader):
            loss = diffusinSA.cal_loss(batch, mode='train', batch_ids=batch_ids) # backward
            # loss = diffusinSA.cal_loss(batch)
            # loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            if local_rank==0:
                tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
                tb_writer.add_scalar('train_loss', loss, step)
            training_loss += loss
        total_time = (time.time()-st)/60
        # validation
        if local_rank==0:
            print('Epoch:{}, train time:{:.2f} min, training loss:{:.3f}'.format(epoch, total_time, training_loss))
            denosing_model.eval()
            st = time.time()
            with torch.no_grad():
                eval_loss = 0
                for batch, batch_ids in tqdm(dev_dataloader):
                    loss = diffusinSA.cal_loss(batch, mode='test', batch_ids=batch_ids)
                    eval_loss += loss
            if local_rank==0:
                tb_writer.add_scalar('eval_loss', eval_loss, epoch)
            total_time = (time.time()-st)/60
            print('Epoch:{}, eval time:{:.2f}, dev loss:{:.3f}'.format(epoch, total_time, eval_loss))
            checkpoint_path = os.path.join(args.exp_dir, 'checkpoint', f'checkpoint_{epoch}.pt')
            # if resuming is needed, then optimizer and scheduler must be saved too
            torch.save(denosing_model.state_dict(), checkpoint_path)
            
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--cfg_path', type=str, help='config file path')
    # dataset
    parser.add_argument('--dataset', type=str, default='text8')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=256)
    # training
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--warmup_ratio', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--init_eval', type=bool, default=False)
    # diffusion SA
    parser.add_argument('--imnoise_method', type=str, choices=['multinomial', 'bigram'], default='multinomial')
    parser.add_argument('--timesteps', type=int, default=5)
    parser.add_argument('--beta_schedule', type=str, default='linear')
    parser.add_argument('--use_cache', type=bool, default=False)
    args = parser.parse_args()
    if args.cfg_path is None:
        args.cfg_path = os.path.join(args.exp_dir, 'config.json')
        assert os.path.exists(args.cfg_path), "No configuration file specified"
    set_seeds(args.seed)
    mp.spawn(train, nprocs=args.ngpu, args=(args,))
