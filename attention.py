import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class selfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.Wq = nn.Linear(config.n_embd, config.n_embd, bias=False).to(device)
        self.Wk = nn.Linear(config.n_embd, config.n_embd, bias=False).to(device)
        self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=False).to(device)

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
        self.Wq = nn.Linear(config.n_embd, config.n_embd, bias=False).to(device)
        self.Wk = nn.Linear(config.n_embd, config.n_embd, bias=False).to(device)
        self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=False).to(device)
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

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Wqkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False).to(device)
        self.Proj = nn.Linear(config.n_embd, config.n_embd, bias=False).to(device)
        self.n_head = config.n_head
        self.seq_len = config.seq_len
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
            
    def forward(self, x):
        B = x.shape[0]
        qkv = self.Wqkv(x) #initially shape (B, seq_len, 3*n_embd)
        q_mat, k_mat, v_mat = qkv.split(self.n_embd, dim=-1)
        q = q_mat.view(q_mat.shape[0], self.seq_len, self.n_head, self.head_dim).transpose(1,2)
        k = k_mat.view(k_mat.shape[0], self.seq_len, self.n_head, self.head_dim).transpose(1,2)
        v = v_mat.view(v_mat.shape[0], self.seq_len, self.n_head, self.head_dim).transpose(1,2)
        

        qkt = q @ k.transpose(-2,-1)
        qkt_scaled = qkt / math.sqrt(self.head_dim)#scale by head_dim
        causal_mask = torch.tril(torch.ones(self.seq_len, self.seq_len, device=x.device)).bool()
        qkt_scaled = qkt_scaled.masked_fill(~causal_mask, float("-inf"))
        attn = F.softmax(qkt_scaled, dim=-1)

        y = attn @ v
        y = y.transpose(1,2).contiguous().view(B, self.seq_len, self.n_embd)

        attn_out = self.Proj(y)
        return attn_out
        
