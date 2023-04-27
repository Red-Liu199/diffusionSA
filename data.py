import torch
import os
import json
import zipfile
import urllib.request
from torch.utils.data import Dataset, DataLoader
from utils import Vocab
import sentencepiece as spm

class Text8Dataset(Dataset):
    """
    The text8 dataset consisting of 100M characters (with vocab size 27).
    We here split the dataset into (90M, 5M, 5M) characters for
    (train, val, test) as in [1,2,3].

    The sets are then split into chunks of equal length as specified by `seq_len`.
    The default is 256, corresponding to what was used in [1]. Other choices
    include 180, as [2] reports using.

    [1] Discrete Flows: Invertible Generative Models of Discrete Data
        Tran et al., 2019, https://arxiv.org/abs/1905.10347
    [2] Architectural Complexity Measures of Recurrent Neural Networks
        Zhang et al., 2016, https://arxiv.org/abs/1602.08210
    [3] Subword Language Modeling with Neural Networks
        Mikolov et al., 2013, http://www.fit.vutbr.cz/~imikolov/rnnlm/char.pdf
    """

    def __init__(self, root='data', seq_len=256, split='train', download=False, timesteps=None, character_level=True, vocab_size=500):
        assert split in {'train', 'valid', 'test'}
        self.root = os.path.join(root, 'text8')
        self.seq_len = seq_len
        self.split = split
        self.timesteps = timesteps

        if not os.path.exists(self.raw_file):
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it.')
        
        if character_level:
        # Get vocabulary
            self.vocab = Vocab()
            vocab_file = os.path.join(self.root, 'vocab.json')
            if os.path.exists(vocab_file):
                self.vocab.load_json(self.root)
            else:
                stoi = self._create_stoi()
                self.vocab.fill(stoi)
                self.vocab.save_json(self.root)
        else:
            tokenizer_path = os.path.join(os.path.dirname(self.raw_file), f'spm_model{vocab_size}.model')
            if not os.path.exists(tokenizer_path):
                self.train_sentencepiece(vocab_size=vocab_size)
            # load tokenizer model
            self.spm = spm.SentencePieceProcessor(model_file=tokenizer_path)
        
        # Preprocess data
        if not os.path.exists(self.processed_file(split, character_level)):
            self._preprocess_data(split, character_level)


        # Load data
        self.data = torch.load(self.processed_file(split, character_level))

    def __getitem__(self, index):
        return self.data[index], index

    def __len__(self):
        return len(self.data)

    def _create_stoi(self):
        rawdata = zipfile.ZipFile(self.raw_file).read('text8').decode('utf-8')
        s = sorted(list(set(rawdata)))
        stoi = {s[i]: i for i in range(len(s))}
        return stoi

    def _preprocess_data(self, split, character_level=True):
        # Read raw data
        rawdata = zipfile.ZipFile(self.raw_file).read('text8').decode('utf-8')

        # Extract subset
        if split == 'train':
            rawdata = rawdata[:90000000]
        elif split == 'valid':
            rawdata = rawdata[90000000:95000000]
        elif split == 'test':
            rawdata = rawdata[95000000:]

        # Encode characters
        if character_level:
            data = torch.tensor([self.vocab.stoi[s] for s in rawdata])
        else:
            data = torch.tensor(self.spm.encode(rawdata))

        # Split into chunks
        data = data[:self.seq_len*(len(data)//self.seq_len)]
        data = data.reshape(-1, self.seq_len) # N, L
        if self.timesteps is not None:
            data = data.unsqueeze(1) # N, 1, L
            cache_data = (-1*torch.ones((data.size(0), self.timesteps, data.size(2)))).long() # N, T, L
            data = torch.cat((data, cache_data), dim=1) # N, T+1, L
        # Save processed data
        torch.save(data, self.processed_file(split, character_level))


    def train_sentencepiece(self, vocab_size):
        # train sentencepiece tokenizer. unzip the text.zip first
        data_path = os.path.dirname(self.raw_file)
        text_file = os.path.join(data_path, 'text8')
        multi_row_text_file = os.path.join(data_path, 'multi_row_text8.txt')
        if not os.path.exists(multi_row_text_file):
            # convert data into multi-row text
            long_text_str = open(text_file, 'r').readlines()[0]
            data = long_text_str.split(' ')
            new_data = []
            start_idx = 0
            step_size = 200 # any length (words) less than 4192 tokens
            while(start_idx<len(data)):
                new_data.append(' '.join(data[start_idx:start_idx+step_size])+'\n')
                start_idx += step_size
            with open(multi_row_text_file, 'w') as fp:
                fp.writelines(new_data)
        # train sentencepiece
        model_prefix = os.path.join(data_path, f'spm_model{vocab_size}')
        spm.SentencePieceTrainer.train(
            input = multi_row_text_file,
            model_prefix = model_prefix,
            vocab_size = vocab_size,
            character_coverage = 1
        )

    @property
    def raw_file(self):
        return os.path.join(self.root, 'text8.zip')

    def processed_file(self, split, character_level=True):
        spm_field = '' if character_level else '_spm'
        if self.timesteps is not None:
            return os.path.join(self.root, 'processed{}_{}_cache.pt'.format(spm_field, split))
        else:
            return os.path.join(self.root, 'processed{}_{}.pt'.format(spm_field, split))

    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading text8...')
        url = 'http://mattmahoney.net/dc/text8.zip'
        print('Downloading from {}...'.format(url))
        urllib.request.urlretrieve(url, self.raw_file)
        print('Saved to {}'.format(self.raw_file))

def get_dataloader(args):
    if args.dataset=='text8':
        time_steps = args.timesteps if hasattr(args, 'use_cache') and args.use_cache else None
        train = Text8Dataset(seq_len=args.seq_len, split='train', download=True, timesteps=time_steps, character_level=args.character_level, vocab_size=args.vocab_size)
        valid = Text8Dataset(seq_len=args.seq_len, split='valid', character_level=args.character_level, vocab_size=args.vocab_size)
        test = Text8Dataset(seq_len=args.seq_len, split='test', character_level=args.character_level, vocab_size=args.vocab_size)
        sampler = torch.utils.data.distributed.DistributedSampler(train)
        train_loader = DataLoader(train, batch_size=args.batch_size//args.ngpu, num_workers=args.num_workers, sampler=sampler)
        valid_loader = DataLoader(valid, batch_size=args.batch_size//args.ngpu, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test, batch_size=args.batch_size//args.ngpu, shuffle=False, num_workers=args.num_workers)
    else:
        pass
    return train_loader, valid_loader, test_loader

if __name__=='__main__':
    train = Text8Dataset(seq_len=256, split='train', download=True, character_level=False, vocab_size=500)
    # print(train.raw_file)
    # train_loader = DataLoader(train, batch_size=64, num_workers=4, shuffle=True)
    # print(len(train_loader), len(train_loader.dataset))