import math
import torch
from torch import nn

# from memory_usage import memory_usage

class ModelEmbedding(nn.Module):
    def __init__(self, token_embed, pos_embed):
        super().__init__()
        self.embedding = pos_embed # nn.Sequential(token_embed, pos_embed)

    def forward(self, x, t_diffs):
        out = self.embedding(x, t_diffs)
        return out

class TokenEmbedding(nn.Module):
    def __init__(self, d_embed, map_size, pad_idx):
        super().__init__() 
        self.embedding = nn.Embedding(num_embeddings=map_size, embedding_dim=d_embed, padding_idx=pad_idx)
        self.d_embed = d_embed
    
    def forward(self, x):
        out = self.embedding(x.int()) * math.sqrt(self.d_embed) # 레이어 정규화
        return out
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_embed):
        super().__init__()
        self.d_embed = d_embed

    def forward(self, x, t_diffs):
        # 1. 시간 차이 정보는 이미 입력받음
        # 2. 미리 정해진 범위가 아니라 각각 다른 값이니 반복문으로 하나씩 처리
        # 3. 각각 처리하고 stack -> x + pos_table
        sinusoid_table = torch.zeros(x.shape[1], x.shape[2]-1).to(x.device)
        even = t_diffs % 2 == 0
        odd = t_diffs % 2 != 0
        for t_diff in t_diffs:
            encoding = torch.zeros(t_diff, self.d_embed)
            encoding.requires_grad = False
            position = t_diff.unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_embed, 2) * -(math.log(10000.0) / self.d_embed))
            encoding[:, 0::2] = torch.sin(position * div_term)
            encoding[:, 1::2] = torch.cos(position * div_term)
            encoding.unsqueeze(0)
            sinusoid_table = torch.stack(encoding, dim=0).to(x.device)
        
        
        # sinusoid_table_list = [self.get_posi_angle_vec(t_diff, self.d_embed) for t_diff in t_diffs]
        # sinusoid_table = torch.stack(sinusoid_table_list, dim=0)
        # sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # even index sin 
        # sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # odd index cos
        out = x[:, : , 0] + sinusoid_table
        return out
    
