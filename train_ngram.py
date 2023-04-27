import argparse
from model import NeuralNGram
import os, shutil
import json
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
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
    num_classes = 27 if args.character_level else args.vocab_size
    ngram_model = NeuralNGram(num_classes, emb_dim=args.embed_dim, conv_dim=args.conv_dim, kernel_size=args.ngram-1)
    ngram_model.to(f'cuda:{local_rank}')
    ngram_model = DDP(ngram_model, device_ids=[local_rank], output_device=local_rank)

    # save config
    if local_rank==0:
        json.dump(vars(args), open(os.path.join(args.exp_dir, 'config.json'), 'w'), indent=2)
    # initialize optimizer, no need for scheduler
    optimizer = optim.Adam(ngram_model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # training
    step = 0
    for epoch in range(args.epochs):
        training_loss = 0
        ngram_model.train()
        st = time.time()
        for batch, batch_ids in tqdm(train_dataloader):
            # batch: (B, L)
            _, loss = ngram_model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if local_rank==0:
                tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
                tb_writer.add_scalar('train_loss', loss.item(), step)
            training_loss += loss
        total_time = (time.time()-st)/60
        # validation
        if local_rank==0:
            print('Epoch:{}, train time:{:.2f} min, training loss:{:.3f}'.format(epoch, total_time, training_loss))
            ngram_model.eval()
            st = time.time()
            with torch.no_grad():
                total_loss = 0
                for batch, batch_ids in tqdm(dev_dataloader):
                    _, loss = ngram_model(batch)
                    total_loss += loss.item()
            avg_loss = total_loss/len(dev_dataloader)
            tb_writer.add_scalar('eval_loss', avg_loss, epoch)
            total_time = (time.time()-st)/60
            print('Epoch:{}, eval time:{:.2f}, dev loss:{:.3f}'.format(epoch, total_time, loss))
            checkpoint_path = os.path.join(args.exp_dir, 'checkpoint', f'checkpoint_{epoch}.pt')
            save_checkpoint(checkpoint_path, ngram_model, optimizer, None)
     
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str)
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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    # architecture
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--conv_dim', type=int, default=512)
    parser.add_argument('--ngram', type=int, default=3)
    args = parser.parse_args()
    set_seeds(args.seed)
    print(args)
    mp.spawn(train, nprocs=args.ngpu, args=(args,))