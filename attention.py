import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Wqkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.Proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
            
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.Wqkv(x) #initially shape (B, T, 3*C)
        q_mat, k_mat, v_mat = qkv.split(self.n_embd, dim=-1)
        q = q_mat.view(q_mat.shape[0], T, self.n_head, self.head_dim).transpose(1,2)
        k = k_mat.view(k_mat.shape[0], T, self.n_head, self.head_dim).transpose(1,2)
        v = v_mat.view(v_mat.shape[0], T, self.n_head, self.head_dim).transpose(1,2)
        

        qkt = q @ k.transpose(-2,-1)
        qkt_scaled = qkt / math.sqrt(self.head_dim)#scale by head_dim
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        qkt_scaled = qkt_scaled.masked_fill(~causal_mask, float("-inf"))
        attn = F.softmax(qkt_scaled, dim=-1)

        y = attn @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)

        attn_out = self.Proj(y)
        return attn_out
        
