import torch.nn as nn
import torch
from train_conf import *

class TestModel(nn.Module):
    def __init__(self, src_embed, encoder):
        super().__init__()

        self.src_embed = src_embed
        self.encoder = encoder
        
    def make_pad_mask(self, query, key, pad_idx=5852):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask
    
    def make_enc_mask(self, src): # src.shape = 개단

        enc_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2) # 개11단
        enc_mask = enc_mask.repeat(1, self.n_heads, src.shape[1], 1) # 개헤단단
        """ src pad mask (문장 마다 다르게 생김. 이건 한 문장에 대한 pad 행렬)
        F F T T
        F F T T
        F F T T
        F F T T
        """
        return enc_mask
    
    def forward(self, src, trg):

        enc_mask = self.make_enc_mask(src)

        enc_out, atten_encs = self.encoder(src, enc_mask)

        return enc_out, atten_encs