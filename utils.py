import random
import numpy as np
import torch
import json
import os
import torch.nn as nn
from collections import OrderedDict
def set_seeds(seed, cuda_deterministic=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

def save_checkpoint(path, model, optimizer, scheduler):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None
        }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.load_state_dict(checkpoint['model'])
    else:
        state_dict = checkpoint['model']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # remove the 'module.'
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

class Vocab():
    
    def __init__(self, stoi={}):
        self.fill(stoi)

    def fill(self, stoi):
        self.stoi = stoi
        self.itos = {i:s for s,i in stoi.items()}

    def save_json(self, path):
        if not os.path.exists(path): os.makedirs(path)
        vocab_file = os.path.join(path, 'vocab.json')
        with open(vocab_file, 'w') as f:
            json.dump(self.stoi, f, indent=4)

    def load_json(self, path):
        vocab_file = os.path.join(path, 'vocab.json')
        with open(vocab_file, 'r') as f:
            stoi = json.load(f)
        self.fill(stoi)

    def string_to_idx(self, string):
        assert isinstance(string, str)
        return [self.stoi[s] for s in string]

    def idx_to_string(self, idx):
        assert isinstance(idx, list)
        count_err = np.sum([1 for i in idx if i not in self.itos])
        if count_err > 0:
            print(f'Warning, {count_err} decodings were not in vocab.')
            print(set([i for i in idx if i not in self.itos]))
        return ''.join([self.itos[i] if i in self.itos else '?' for i in idx])

    def encode(self, text, padding_value=0):
        assert isinstance(text, list)
        length = torch.tensor([len(string) for string in text])
        tensor_list = [torch.tensor(self.string_to_idx(string)) for string in text]
        tensor = nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)
        return tensor, length

    def decode(self, tensor, length):
        assert torch.is_tensor(tensor)
        assert tensor.dim() == 2, 'Tensor should have shape (batch_size, seq_len)'
        text = [self.idx_to_string(tensor[b][:length[b]].tolist()) for b in range(tensor.shape[0])]
        return text