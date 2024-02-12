import torch
import torch.nn as nn
import copy
from src.models.layers.Attention import MHA, FeedForward

# from memory_usage import memory_usage
import sys

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, ff, d_model, drop_p):
        super().__init__()

        self.self_atten = self_attention
        self.self_atten_LN = nn.LayerNorm(d_model)

        self.FF = ff
        self.FF_LN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, enc_mask):

        residual, atten_enc = self.self_atten(x, x, x, enc_mask)
        residual = self.dropout(residual)
        x = self.self_atten_LN(x + residual)

        residual = self.FF(x)
        residual = self.dropout(residual)
        x = self.FF_LN(x + residual)

        return x, atten_enc

class EncoderBlock(nn.Module):
    def __init__(self, input_embedding, encoder_layer, n_layers, d_model, drop_p):  
        super().__init__()

        self.scale = torch.sqrt(torch.tensor(d_model))
        self.input_embedding = input_embedding

        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.ModuleList([encoder_layer for _ in range(n_layers)])
        
    def forward(self, src, mask, atten_map_save = False): # src : 배치 길이 mask : 배치 헤드 길이 길이

        # x = self.input_embedding(src) # 5.3696 MB, 140392170968016
        # self.scale 을 곱해주면 position 보다 token 정보를 더 보게 된다 (gradient에 self.scale 만큼이 더 곱해짐)
        x = self.dropout(src)

        atten_encs = torch.tensor([])
        
        for layer in self.layers:
            x, atten_enc = layer(x, mask)
            if atten_map_save is True:
                atten_encs = torch.cat([atten_encs , atten_enc[0].unsqueeze(0)], dim=0) # 레이어 헤드 길이 길이
        # # 시퀀스 차원 전체에 걸쳐 평균 풀링
        # x = x.mean(dim=2)
        # x = self.logistic_regression(x)
        return x, atten_encs