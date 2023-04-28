from aifc import Error
from distutils.log import log
import math
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from axial_positional_embedding import AxialPositionalEmbedding
from linear_attention_transformer import LinearAttentionTransformer
from imnoise import *
from utils import load_checkpoint
from typing import Callable, Literal
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LinearAttentionTransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, dim, depth, n_blocks, max_seq_len, num_timesteps, heads = 8, dim_head = None, causal = False, one_kv_head = False, reversible = False, ff_chunks = 1, ff_glu = False, ff_dropout = 0., attn_layer_dropout = 0., attn_dropout = 0., blindspot_size = 1, n_local_attn_heads = 0, local_attn_window_size = 128, return_embeddings = False, receives_context = False, pkm_layers = tuple(), pkm_num_keys = 128, attend_axially = False, linformer_settings = None, context_linformer_settings = None):
        assert (max_seq_len % local_attn_window_size) == 0, 'max sequence length must be divisible by the window size, to calculate number of kmeans cluster'
        super().__init__()
        # emb_dim = default(emb_dim, dim)
        self.num_classes = input_dim
        self.max_seq_len = max_seq_len

        self.depth = depth
        emb_dim = dim
        self.emb_dim = emb_dim

        self.depth = depth
        self.n_blocks = n_blocks

        self.first = nn.Embedding(input_dim, emb_dim)

        self.time_pos_emb = SinusoidalPosEmb(emb_dim, num_timesteps)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.Softplus(),
            nn.Linear(emb_dim * 4, emb_dim * n_blocks * depth)
        )

        # self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.axial_pos_emb = AxialPositionalEmbedding(emb_dim, axial_shape=(max_seq_len // local_attn_window_size, local_attn_window_size))

        self.transformer_blocks = torch.nn.ModuleList()
        for i in range(n_blocks):
            self.transformer_blocks.append(torch.nn.ModuleList())
            for j in range(depth):
                self.transformer_blocks[-1].append(
                    LinearAttentionTransformer(
                        dim, 1, max_seq_len, heads = heads, dim_head = dim_head,
                        causal = causal, # one_kv_head = one_kv_head,
                        ff_chunks = ff_chunks, ff_glu = ff_glu,
                        ff_dropout = ff_dropout,
                        attn_layer_dropout = attn_layer_dropout,
                        attn_dropout = attn_dropout, reversible = reversible,
                        blindspot_size = blindspot_size,
                        n_local_attn_heads = n_local_attn_heads,
                        local_attn_window_size = local_attn_window_size,
                        receives_context = receives_context,
                        pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys,
                        attend_axially = attend_axially,
                        linformer_settings = linformer_settings,
                        context_linformer_settings = context_linformer_settings))

        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(emb_dim, output_dim) if not return_embeddings else nn.Identity()

    def forward(self, x, t, **kwargs):
        t = self.time_pos_emb(t)
        t = self.mlp(t)
        time_embed = t.view(x.size(0), 1, self.emb_dim, self.n_blocks, self.depth)
        x = self.first(x)
        x_embed_axial = x + self.axial_pos_emb(x).type(x.type())
        # x_embed_axial_time = x_embed_axial + time_embed
        h = torch.zeros_like(x_embed_axial)

        for i, block in enumerate(self.transformer_blocks):
            h = h + x_embed_axial
            for j, transformer in enumerate(block):
                h = transformer(h + time_embed[..., i, j])

        h = self.norm(h)
        return self.out(h)

class NeuralNGram(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int, conv_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.act = nn.Tanh()
        self.conv = nn.Sequential(
            nn.ConstantPad1d((kernel_size-1, 0), 0),
            nn.Conv1d(emb_dim, conv_dim, kernel_size=kernel_size)
        )

    def forward(self, x: torch.Tensor):
        """
        x: (N, T)
        -> embedded (N, T, E)
        -> transpose (N, E, T)
        -> act & conv (N, H, T)
        -> transpose (N, T, H)
        -> linear (N, T, V)
        """
        target = x[:, 1:]
        x = self.embedding(x)
        x = self.conv(self.act(x).transpose(1, 2)).transpose(1, 2)
        logits = self.classifier(x)
        loss = F.cross_entropy(logits[:, :-1, :].transpose(1, 2), target)
        return logits, loss

class WrappedNGram(object):
    def __init__(self, ngram_model) -> None:
        self.ngram_model = ngram_model
    
    def forward(self, x):
        # x:(B,L)
        if isinstance(self.ngram_model, nn.Module):
            with torch.no_grad():
                logits, _ = self.ngram_model(x) # B,L,V
                probs = F.softmax(logits, dim=-1)
        else:
            pass
        return probs

class noise_model(object):
    def __init__(self, method, num_classes, device) -> None:
        self.num_classes = num_classes
        print('Imnoise method:', method)
        if method=='multinomial':
            self.imnoise_func = imnoise_multinomial
        else:
            ngram_map = {'bigram':2, 'trigram':3, 'fourgram':4}
            self.imnoise_func = imnoise_ngram
            self.ngram = ngram_map[method]
            # here we load the neural ngram model trained before, you can load your own ngram model here
            # ngram_model: an autoregressive language model which only consider N tokens to predict next token
            model_dir = f'exp/{self.ngram}gram/'
            model_cfg = json.load(open(model_dir+'config.json', 'r'))
            ngram_model = NeuralNGram(model_cfg['vocab_size'], model_cfg['embed_dim'], model_cfg['conv_dim'], model_cfg['ngram']-1)
            load_checkpoint(model_dir+'checkpoint/checkpoint.pt', ngram_model, None, None)
            ngram_model.to(device)
            self.ngram_model = WrappedNGram(ngram_model)

    def imnoise(self, x: torch.tensor, beta, num_classes, target_samples):
        if self.imnoise_func==imnoise_multinomial:
            return self.imnoise_func(x, beta, num_classes=num_classes, target_samples=target_samples)
        elif self.imnoise_func==imnoise_ngram:
            return self.imnoise_func(x, ngram=self.ngram, model=self.ngram_model, beta=beta, vocab_size=num_classes, target_samples=target_samples)
        else:
            raise ValueError

    def score_noise(self, x):
        # x: (B, L)
        if self.imnoise_func == imnoise_multinomial:
            # uniform distribution for every token
            log_probs = -np.log(self.num_classes)*torch.ones(x.shape)
            log_probs = log_probs.sum(-1) # (B,)
        elif self.imnoise_func == imnoise_ngram:
            probs = self.ngram_model.forward(x) # B,L,V
            log_probs = torch.log(probs.clamp(min=1e-30))
            log_probs = log_probs[:, :-1, :].gather(index=x[:, 1:].unsqueeze(-1), dim=-1).squeeze(-1) # B,L
            log_probs = log_probs.sum(-1)
        else:
            raise ValueError
        return log_probs

    def sample_noise(self, sample_size):
        # sample_size=(B,L)
        if self.imnoise_func == imnoise_multinomial:
            x = torch.randint(0, self.num_classes, sample_size)
        elif self.imnoise_func == imnoise_ngram:
            batch_size = sample_size[0]
            start_tokens = torch.randint(0, self.num_classes, (batch_size, self.ngram-1)) # B, N-1
            x = torch.zeros(sample_size, dtype=torch.long) # B,L
            for k in range(sample_size[1]):
                probs = self.ngram_model.forward(start_tokens) # B, N-1, V
                next_token_probs = probs[:, -1, :] # B,V
                x_k_noise = torch.multinomial(next_token_probs, 1).squeeze() # (B,)
                x[:, k] = x_k_noise
                start_tokens = torch.cat((start_tokens[:, 1:], x_k_noise.unsqueeze(-1)), dim=-1)
        return x

class diffusion_SA(object):
    def __init__(self, imnoise_func: Callable, denoise_func: nn.Module, timesteps: int, beta_schedule: Callable, use_cache: bool, dataset) -> None:
        
        self.imnoise_func = imnoise_func
        self.denoise_func = denoise_func
        self.timesteps = timesteps
        self.betas = beta_schedule(self.timesteps)
        self.use_cache = use_cache
        self.dataset = dataset
        self.num_classes = self.denoise_func.module.num_classes
        
    
    def proposal(self, x_0, mode:Literal['single', 'all']='all', time_idx= None):
        # x_0: a batch of data, size (B, L)
        if mode=='single':
            raise ValueError("No single proposal currently")
        else:
            x = x_0
            x_series=[x_0]
            log_q_probs = 0
            for t in range(self.timesteps):
                log_x_prob, x = self.imnoise_func.imnoise(x.to(self.denoise_func.device), self.betas[t], num_classes=self.num_classes, target_samples=None) # (B,), (B,L)
                x_series.append(x)
                log_q_probs += log_x_prob # (B,)
        # x_series: [(B, L), (B, L), ...] --> (T+1, B, L)
        x_series = torch.stack(x_series, dim=0)
        return x_series, log_q_probs
    
    def imnoising_score(self, x_series):
        # x_series: (T+1, B, L)
        log_q_probs = 0
        for t in range(1, len(x_series)):
            x_t = x_series[t]
            x_tm1 = x_series[t-1]
            log_x_prob = self.imnoise_func.imnoise(x_tm1, self.betas[t-1], num_classes=self.num_classes, target_samples=x_t) # (B,)
            log_q_probs += log_x_prob # (B,)
        return log_q_probs
    
    def sample_x(self, sample_size, greedy=False):
        # sample x from x_T from denoising model
        x_T = self.imnoise_func.sample_noise(sample_size).to(self.denoise_func.device) # B, L
        x_series = torch.zeros((self.timesteps+1,)+tuple(sample_size), dtype=torch.long, device=x_T.device) # T+1, B, L
        x_series[self.timesteps, :, :] = x_T
        for t in range(self.timesteps, 0, -1):
            t_batch = (t*torch.ones(x_T.size(0), device=x_T.device)).long()
            logits = self.denoise_func(x_series[t, :, :], t_batch) # B, L, V
            if not greedy:
                probs = F.softmax(logits, dim=-1).view(-1, logits.size(-1)) # B*L, V
                x_tm1_pred = torch.multinomial(probs, 1).view(logits.shape[:-1]) # B, L
            else:
                x_tm1_pred = logits.argmax(-1)
            x_series[t-1, :, :] = x_tm1_pred
        return x_series

    def denoising_score(self, x_series, return_samples=False):
        # x_series: (T+1, B, L)
        # return_samples: return imnoising samples and denosing samples at each step
        with torch.no_grad():
            log_p_probs = 0
            if return_samples:
                x_series_preds = x_series.clone()
            for t in range(1, len(x_series)):
                x_t = x_series[t]
                x_tm1 = x_series[t-1] # B, L
                t_batch = (t*torch.ones(x_t.size(0), device=x_t.device)).long()
                logits = self.denoise_func(x_t, t_batch) # B, L, V
                log_prob_pred = F.log_softmax(logits, dim=-1)
                log_p_probs += log_prob_pred.gather(index=x_tm1.unsqueeze(-1), dim=-1).squeeze(-1).sum(-1) # B,L
                if return_samples:
                    x_tm1_pred = logits.argmax(-1) # B, L
                    x_series_preds[t-1, :, :] = x_tm1_pred
            # we assume that logp(x_T)=logq(x_T|x_0)
            log_p_probs += self.imnoise_func.score_noise(x_series[-1]).to(log_p_probs.device)
            if return_samples:
                return log_p_probs, x_series_preds
            else:
                return log_p_probs
        

    def MIS(self, x_0, steps=2, batch_ids=None):
        total_poposal_num, accept_num = 0, 0
        if self.use_cache:
            # x_0: (B,T+1,L)
            for i in range(steps):
                x_proposals, log_q = self.proposal(x_0[:, 0, :].to(self.denoise_func.device)) # (T+1, B, L)
                total_poposal_num += x_0.size(0)
                log_p = self.denoising_score(x_proposals) # (B,)
                if x_0[0, 1, 0]==-1: # the first step, directly accept all
                    x_series = x_proposals
                    pv_log_p = log_p
                    pv_log_q = log_q
                    # update the sample cache
                    self.dataset.data[batch_ids] = x_proposals.permute(1, 0, 2).clone()
                    accept_num += x_0.size(0)
                else:
                    # x_0 is the previous x series
                    x_series = x_0.permute(1, 0, 2).to(self.denoise_func.device) #(T+1, B, L)
                    pv_log_p = self.denoising_score(x_series)
                    pv_log_q = self.imnoising_score(x_series)
                    log_accept_prob = log_p - pv_log_p + pv_log_q.to(log_p.device) - log_q.to(log_p.deivce) # (B, )
                    accept_prob = torch.clamp(log_accept_prob.exp(), max=1) # (B,)
                    change_flag = False
                    for b in range(x_0.size(0)):
                        randnum = random.random()
                        if accept_prob[b]>randnum:
                            accept_num += 1
                            change_flag = True
                            x_series[:, b, :] = x_proposals[:, b, :]
                            pv_log_p[b] = log_p[b]
                            pv_log_q[b] = log_q[b]
                    if change_flag:
                        self.dataset.data[batch_ids] = x_series.permute(1, 0, 2).clone()

        else:
            # x_0: (B,L)
            x_series = None
            for i in range(steps):
                x_proposals, log_q = self.proposal(x_0.to(self.denoise_func.device)) # (T+1, B, L)
                total_poposal_num += x_0.size(0)
                log_p = self.denoising_score(x_proposals) # (B,)
                if x_series is None: # the first step, directly accept all
                    x_series = x_proposals
                    pv_log_p = log_p
                    pv_log_q = log_q
                    accept_num += x_0.size(0)
                else:
                    log_accept_prob = log_p - pv_log_p + pv_log_q.to(log_p.device) - log_q.to(log_p.device) # (B, )
                    accept_prob = torch.clamp(log_accept_prob.exp(), max=1) # (B,)
                    for b in range(x_0.size(0)):
                        randnum = random.random()
                        if accept_prob[b]>randnum:
                            accept_num += 1
                            x_series[:, b, :] = x_proposals[:, b, :]
                            pv_log_p[b] = log_p[b]
                            pv_log_q[b] = log_q[b]
        accept_rate = accept_num/total_poposal_num
        return x_series, pv_log_p, pv_log_q, accept_rate

    def log_prob_std(self, input_data):
        # calculate the log prob of one sentence in a standard way
        pass 

    def cal_loss(self, input_data, mode='train', batch_ids=None):
        MIS_steps = 1 if self.use_cache else 2
        if mode=='train':
            x_series, _, _, accept_rate = self.MIS(input_data, steps=MIS_steps, batch_ids=batch_ids)
            x_series = x_series.to(self.denoise_func.device)
            total_loss = 0
            for t in range(1, len(x_series)):
                x_t = x_series[t]
                x_tm1 = x_series[t-1] # B, L
                t_batch = (t*torch.ones(x_t.size(0), device=x_t.device)).long()
                logits = self.denoise_func(x_t, t_batch) # B, L, V
                log_prob_pred = F.log_softmax(logits, dim=-1)
                loss = -log_prob_pred.gather(index=x_tm1.unsqueeze(-1), dim=-1).squeeze(-1) # B,L
                loss = torch.mean(loss.sum(-1))
                loss.backward()
                total_loss += loss.item()
            return total_loss, accept_rate
        else:
            # return log_prob(input_data)
            x_proposals, log_q = self.proposal(input_data.to(self.denoise_func.device)) # (T+1, B, L)
            log_p, x_preds = self.denoising_score(x_proposals, return_samples=True) # (B,)
            return torch.mean(log_p - log_q.to(log_p.device)).item(), x_proposals, x_preds

