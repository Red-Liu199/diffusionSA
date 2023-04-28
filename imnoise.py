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
    log_x = torch.log(x_onehot.float().clamp(min=1e-30)) # -30 or 0

    return log_x

def imnoise_multinomial(x: torch.tensor, beta, num_classes, target_samples=None):
    # x: (B, L)
    # return log_probs (B,) of each sentence and noisy samples
    # target_samples (B, L): if it is not None, then the function is to score the transition: x --> target_samples
    log_x_onehot = index_to_log_onehot(x, num_classes).float() # (B,L,C)
    log_prob = torch.logaddexp(np.log(1-beta)+log_x_onehot, torch.tensor(np.log(beta)-np.log(num_classes), device=x.device)) # (B,L,C)
    if target_samples is None:
        prob = log_prob.exp().view(-1, log_prob.size(-1)) #(B*L, C)
        x_noise = torch.multinomial(prob, 1).view(log_prob.shape[:-1]) # (B, L)
        log_sent_prob = log_prob.gather(index=x_noise.unsqueeze(-1), dim=-1).squeeze(-1).sum(-1) # (B,)
        return log_sent_prob, x_noise
    else:
        log_sent_prob = log_prob.gather(index=target_samples.unsqueeze(-1), dim=-1).squeeze(-1).sum(-1)
        return log_sent_prob
    

def imnoise_ngram(x: torch.tensor, ngram: int, model, beta, vocab_size: int, target_samples=None):
    # x: (B,L)
    # ngram: 2,3,4
    # model: the ngram model, model(x_{i-ngram+1,...,i-1}) = prob(x_i|x_{i-ngram+1,...,i-1})
    if target_samples is None:
        seq_len = x.size(1)
        x_noise = x.clone()
        log_probs = torch.zeros((x.size(0)), dtype=torch.float, device=x.device) # (B,)
        for i in range(seq_len):
            # autoregressive
            if i<ngram-1:
                # the noise distribution of the first few tokens is uniform distribution
                log_onehot = index_to_log_onehot(x_noise[:, i], vocab_size) #  B,V
                log_prob = torch.logaddexp(np.log(1-beta)+log_onehot, torch.tensor(np.log(beta)-np.log(vocab_size), device=x.device)) # (B,V)
                prob = log_prob.exp()
            else:
                # the noise distribution is the ngram distribution given previous N tokens
                model_input = x_noise[:, i-ngram+1:i] # B,N
                prob_model = model.forward(model_input) # B, N, V
                prob_model = prob_model[:, -1, :] # B,V
                prob = (1-beta)*F.one_hot(x_noise[:, i], vocab_size).float() + beta*prob_model

            x_i_noise = torch.multinomial(prob, 1).squeeze() # (B,)
            log_probs += torch.log(prob.clamp(min=1e-30)).gather(index=x_i_noise.unsqueeze(1), dim=1).squeeze(1) # (B,)
            x_noise[:, i] = x_i_noise
        return log_probs, x_noise
    else:
        next_token_log_probs = torch.log(model.forward(target_samples).clamp(min=1e-30)) # (B, L, V)
        log_onehot = index_to_log_onehot(x, vocab_size) #  B, L, V
        log_probs = torch.logaddexp(np.log(1-beta)+log_onehot, np.log(beta)+next_token_log_probs) # (B, L, V)
        log_sent_probs = log_probs.gather(index=target_samples.unsqueeze(-1), dim=-1).squeeze(-1) # (B, L)
        return log_sent_probs.sum(-1)
        

    # x_onehot = F.one_hot(x, vocab_size) # B,L,V
    # p_x_given_xm1 = torch.matmul(x_onehot.float(), bigram_matrix) # p(·|x_{i-1}) shape:(B,L,V)
    # p_xp1_given_x = torch.matmul(x_onehot.float(), bigram_matrix.T) # p(x_{i+1}|·) shape:(B,L,V)
    # paddings = torch.ones((x.size(0), 1, vocab_size), dtype=torch.float) # B,1,V
    # p_x_given_xm1 = torch.cat((paddings, p_x_given_xm1[:, :-1, :]), dim=1)
    # p_xp1_given_x = torch.cat((p_xp1_given_x[:, 1:, :], paddings), dim=1)
    # p_x = p_x_given_xm1*p_xp1_given_x 
