class GPTConfig:
    def __init__(self,
                 vocab_size,
                 seq_len,
                 n_embd,
                 n_head,
                 n_layer
                ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

