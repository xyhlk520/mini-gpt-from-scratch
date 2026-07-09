import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class selfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.Wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wk = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self,x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        qkt = q @ k.transpose(-2,-1)
        qkt_scaled = qkt / math.sqrt(q.size(-1))
        attn = F.softmax(qkt_scaled, dim=-1)
        y = attn @ v

        return y

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wk = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.seq_len = config.seq_len

    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        qkt = q @ k.transpose(-2,-1)
        qkt_scaled = qkt / math.sqrt(q.size(-1))
        causal_mask = torch.tril(torch.ones(self.seq_len, self.seq_len, device=x.device)).bool()
        qkt_scaled = qkt_scaled.masked_fill(~causal_mask, float("-inf"))
        attn = F.softmax(qkt_scaled, dim=-1)
        y = attn @ v

        return y