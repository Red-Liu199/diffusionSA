import torch
import torch.nn.functional as F
import numpy as np
import math
# q(x_t|x_t-1)
def linear_beta_schedule(timesteps):
    alphas_cumprod = np.linspace(1, 0, timesteps+1)
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = np.clip(alphas, a_min=0.001, a_max=1.)
    return 1-alphas

def cosine_beta_schedule(timesteps, s = 0.2):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return 1-alphas

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    
    x_onehot = F.one_hot(x, num_classes) # (B, L, C)
    # permute_order = (0, -1) + tuple(range(1, len(x.size()))) # (0, -1, 1) if x is two-dimensional
    # x_onehot = x_onehot.permute(permute_order) # (B, C, L)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30)) # -30 or 0

    return log_x

def imnoise_multinomial(x: torch.tensor, beta, num_classes):
    # x: (B, L)
    log_x_onehot = index_to_log_onehot(x, num_classes).float() # (B,L,C)
    log_prob = torch.logaddexp(np.log(1-beta)+log_x_onehot, torch.tensor(np.log(beta)-np.log(num_classes), device=x.device))
    return log_prob
    

def imnoise_bigram(x: torch.tensor, bigram_model: dict, beta, vocab_size: int):
    # bigram_model: bigram_model(w1, w2) = Prob(w2|w1)
    # x: (B,L)
    x_length = x.size[-1]
    