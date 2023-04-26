import argparse
from model import LinearAttentionTransformerModel, diffusion_SA
from imnoise import *
import os, shutil
import json
import math
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
        if not os.path.exists(args.exp_dir):
            os.mkdir(args.exp_dir)
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
    cfg['model']['input_dim'] = 27 if args.character_level else args.vocab_size
    cfg['model']['output_dim'] = 27 if args.character_level else args.vocab_size
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
            total_tokens = 0
            total_log_probs = 0
            for batch, batch_ids in tqdm(dev_dataloader):
                log_probs, _, _ = diffusinSA.cal_loss(batch, mode='test') # avg log probs per sentence
                total_tokens += torch.numel(batch)
                total_log_probs += log_probs*batch.size(0)
                count += 1
                if count>=10:
                    break
        avg_log_prob = total_log_probs/total_tokens
        bpc = -avg_log_prob/math.log(2)
        total_time = (time.time()-st)/60
        print('Init eval time:{:.2f}, dev tokens:{}, dev bpc:{:.3f}'.format(total_time, total_tokens, bpc))
        checkpoint_path = os.path.join(args.exp_dir, 'checkpoint', 'checkpoint_init.pt')
        save_checkpoint(checkpoint_path, denosing_model, optimizer, scheduler)

    # training
    step = 0
    for epoch in range(args.epochs):
        training_loss = 0
        denosing_model.train()
        st = time.time()
        for batch, batch_ids in tqdm(train_dataloader):
            loss, accept_rate = diffusinSA.cal_loss(batch, mode='train', batch_ids=batch_ids) # backward
            # loss = diffusinSA.cal_loss(batch)
            # loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            if args.debugging and step>=100: # only run several training steps in debugging mode
                break
            if local_rank==0:
                tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
                tb_writer.add_scalar('train_loss', loss, step)
                tb_writer.add_scalar('training_accept_rate', accept_rate, step)
            training_loss += loss
        total_time = (time.time()-st)/60
        # validation
        if local_rank==0:
            print('Epoch:{}, train time:{:.2f} min, training loss:{:.3f}'.format(epoch, total_time, training_loss))
            denosing_model.eval()
            st = time.time()
            with torch.no_grad():
                total_tokens = 0
                total_log_probs = 0
                for batch, batch_ids in tqdm(dev_dataloader):
                    log_probs, _, _ = diffusinSA.cal_loss(batch, mode='test') # avg log probs per sentence
                    total_tokens += torch.numel(batch)
                    total_log_probs += log_probs*batch.size(0)
            avg_log_prob = total_log_probs/total_tokens
            bpc = -avg_log_prob/math.log(2)
            tb_writer.add_scalar('eval_bpc', bpc, epoch)
            total_time = (time.time()-st)/60
            print('Epoch:{}, eval time:{:.2f}, dev tokens:{}, dev bpc:{:.3f}'.format(epoch, total_time, total_tokens, bpc))
            checkpoint_path = os.path.join(args.exp_dir, 'checkpoint', f'checkpoint_{epoch}.pt')
            save_checkpoint(checkpoint_path, denosing_model, optimizer, scheduler)
            
def test(local_rank:int, args):
    dist_url='tcp://localhost:13458'
    dist.init_process_group(backend='nccl', init_method=dist_url, world_size=args.ngpu, rank=local_rank)
    # load data
    train_dataloader, dev_dataloader, test_dataloader = get_dataloader(args)
    # initialize model
    cfg = json.load(open(args.cfg_path, 'r'))
    model_cfg = cfg['model']
    denosing_model = LinearAttentionTransformerModel(**model_cfg)
    denosing_model.to(f'cuda:{local_rank}')
    denosing_model = DDP(denosing_model, device_ids=[local_rank], output_device=local_rank)
    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    denosing_model.load_state_dict(checkpoint['model'])
    # for storing samples
    store_batch_num = 1
    store_sample_path = args.checkpoint[:-3]+'.json'
    samples = {
        'proposals':[],
        'predictions':[],
        'samples':[]
    }
    imnoise_func = imnoise_multinomial if args.imnoise_method=='multinomial' else imnoise_bigram
    beta_schedule = linear_beta_schedule if args.beta_schedule=='linear' else cosine_beta_schedule
    diffusinSA = diffusion_SA(imnoise_func, denosing_model, timesteps=args.timesteps, beta_schedule=beta_schedule,
        use_cache=args.use_cache, dataset=train_dataloader.dataset)
    # validation
    if local_rank==0:
        denosing_model.eval()
        st = time.time()
        with torch.no_grad():
            total_tokens = 0
            total_log_probs = 0
            batch_num = 0
            for batch, _ in tqdm(dev_dataloader):
                log_probs, x_series_proposed, x_series_pred = diffusinSA.cal_loss(batch, mode='test') # avg log probs per sentence
                total_tokens += torch.numel(batch)
                total_log_probs += log_probs*batch.size(0)
                if batch_num<store_batch_num:
                    samples['proposals'].append(x_series_proposed.cpu().tolist())
                    samples['predictions'].append(x_series_pred.cpu().tolist())
                    denoised_x_series_samples=diffusinSA.sample_x(batch.shape, greedy=args.greedy_sampling)
                    samples['samples'].append(denoised_x_series_samples.cpu().tolist())
                    batch_num +=1
                    if args.debugging:
                        break
        avg_log_prob = total_log_probs/total_tokens
        bpc = -avg_log_prob/math.log(2)
        total_time = (time.time()-st)/60
        print('Eval time:{:.2f}, test tokens:{}, test bpc:{:.3f}'.format(total_time, total_tokens, bpc))
        json.dump(samples, open(store_sample_path, 'w'), indent=2)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--cfg_path', type=str, default="model_config.json", help='config file path')
    # dataset
    parser.add_argument('--dataset', type=str, default='text8')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--character_level', action="store_true")
    parser.add_argument('--vocab_size', type=int, default=500)
    # training
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--warmup_ratio', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--init_eval', action="store_true")
    parser.add_argument('--debugging', action="store_true", help="debugging mode")
    # testing
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--checkpoint', type=str, help="checkpoint path for testing")
    parser.add_argument('--greedy_sampling', action="store_true")
    # diffusion SA
    parser.add_argument('--imnoise_method', type=str, choices=['multinomial', 'bigram'], default='multinomial')
    parser.add_argument('--timesteps', type=int, default=5)
    parser.add_argument('--beta_schedule', type=str, default='linear')
    parser.add_argument('--use_cache', action="store_true")
    args = parser.parse_args()
    if args.cfg_path is None:
        args.cfg_path = os.path.join(args.exp_dir, 'config.json')
        assert os.path.exists(args.cfg_path), "No model configuration file specified"
    set_seeds(args.seed)
    print(args)
    if args.test:
        mp.spawn(test, nprocs=1, args=(args,))
    else:
        mp.spawn(train, nprocs=args.ngpu, args=(args,))
