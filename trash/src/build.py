import torch
import torch.nn as nn

from Embedding import ModelEmbedding, TokenEmbedding, PositionalEncoding
from EncoderBlock import Encoder, EncoderLayer
from MultiHeadAttention import MHA, FeedForward
import torch.nn.functional as F
# from test_model import TestModel
from train_conf import *
from memory_usage import memory_usage

import sys

class TestModel(nn.Module):
    def __init__(self, src_embed, encoder, generator, n_heads, pad_idx):
        super().__init__()

        self.src_embed = src_embed
        self.encoder = encoder
        self.generator = generator        
        self.n_heads = n_heads
        self.pad_idx = pad_idx
        
    def make_pad_mask(self, query, key):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(self.pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = query.ne(self.pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask
    
    def make_enc_mask(self, src): # 배치, 길이

        enc_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2) # 배치 길이 차원 -> 배치 1 1 길이 차원
        enc_mask = enc_mask.repeat(1, self.n_heads, src.shape[1], 1) # 배치 헤드 길이 길이
        """ src pad mask (문장 마다 다르게 생김. 이건 한 문장에 대한 pad 행렬)
        F F T T
        F F T T
        F F T T
        F F T T
        """
        return enc_mask
    
    def discretize_outputs(self, out, threshold=0.5):
        """
        모델 출력을 이산화하는 함수
        
        Parameters:
        outputs (torch.Tensor): 모델의 출력값.
        threshold (float): 이산화를 위한 임계값. 기본값은 0.5
        
        Returns:
        torch.Tensor: 이산화된 출력값. 값은 0 또는 1
        """
        # outputs 텐서에 대해 threshold 이상이면 1, 미만이면 0으로 변환
        
        flattened = out.view(out.size(0), -1)
        result = self.fc(flattened)
        result = result.view(out.size(0), out.size(1), -1)
        return result.mean(dim=2)
    
    def forward(self, src):
        
        enc_mask = self.make_enc_mask(src)

        enc_out, atten_encs = self.encoder(src, enc_mask)
        aggregated_features = torch.mean(enc_out, dim=1)  # [배치 크기, 특징 차원]
        out_prob = self.generator(aggregated_features)
        
        return out_prob, atten_encs

def build_model(device=torch.device("cuda"), d_embed=1, n_layer=4, d_model=128, n_heads=8, d_ff=512, pad_idx=999, drop_p=0.1):
    import copy
    copy = copy.deepcopy

    src_token_embed = TokenEmbedding(
                                     d_embed = d_embed,
                                     map_size = map_size,
                                     pad_idx=pad_idx)
    
    pos_embed = PositionalEncoding(
                                   d_embed = d_embed)

    src_embed = ModelEmbedding(
                                token_embed = src_token_embed, # 어차피 사용안되니 나중에 삭제
                                pos_embed = pos_embed)

    attention = MHA(
                    d_embed = d_embed,
                    d_model = d_model,
                    n_heads = n_heads)
    
    ff = FeedForward(d_model=d_model, d_ff=d_ff, drop_p=drop_p)

    encoder_block = EncoderLayer(
                                 self_attention = attention,
                                 ff = ff,
                                 d_model=d_model,
                                 drop_p=drop_p)
    
    encoder = Encoder(input_embedding=src_embed,
                      encoder_layer=encoder_block,
                      n_layers=n_layer,
                      d_model=d_model,
                      drop_p=drop_p)
    
    generator = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    model = TestModel(
        src_embed=src_embed,
        encoder=encoder,
        generator=generator,
        n_heads=n_heads,
        pad_idx = pad_idx
        )
    
    model.to(device)
    model.to(torch.float16)
    
    return model