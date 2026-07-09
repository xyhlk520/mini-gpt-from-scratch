import attention as attn
import torch.nn as nn
import utils

class GPTblock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = attn.MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ffl = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4*config.n_embd, config.n_embd, bias=False),
        )
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffl(self.ln2(x))

        return x

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.Sequential(*[GPTblock(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)
    
        self.register_buffer(
            "pos_encode",
            utils.get_position_encoding(config.seq_len, config.n_embd),
            persistent=False,
            )

    def forward(self, x):
        embd_pos = self.embedding(x) + self.pos_encode[:, :x.size(1), :]
        x = self.blocks(embd_pos)
        logits = self.head(self.ln(x))

        return logits