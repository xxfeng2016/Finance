from torch import nn
import torch

from einops import rearrange
# from memory_usage import memory_usage

class MHA(nn.Module):
    def __init__(self, d_embed, d_model, n_heads):
        super().__init__()

        self.n_heads = n_heads

        self.fc_q = nn.Linear(d_embed, d_model) # 배치, 길이, d_embed -> 배치, 길이, d_model
        self.fc_k = nn.Linear(d_embed, d_model) #
        self.fc_v = nn.Linear(d_embed, d_model) #
        self.fc_o = nn.Linear(d_model, d_model) #

        self.scale = torch.sqrt(torch.tensor(d_model / n_heads))

    def forward(self, Q, K, V, mask = None): 

        Q = self.fc_q(Q.reshape(1, -1, 1))
        K = self.fc_k(K.reshape(1, -1, 1))
        V = self.fc_v(V.reshape(1, -1, 1))

        Q = rearrange(Q, '배치 길이 (헤드 d_model) -> 배치 헤드 길이 d_model', 헤드 = self.n_heads) 
        K = rearrange(K, '배치 길이 (헤드 d_model) -> 배치 헤드 길이 d_model', 헤드 = self.n_heads)
        V = rearrange(V, '배치 길이 (헤드 d_model) -> 배치 헤드 길이 d_model', 헤드 = self.n_heads)

        attention_score = Q @ K.transpose(-2,-1)/self.scale # 배치 헤드 길이 길이

        if mask is not None:
            attention_score = attention_score.type(torch.float32)
            attention_score[mask] = -1e10
        attention_weights = torch.softmax(attention_score, dim=-1) # 배치 헤드 길이 길이

        attention = attention_weights @ V # 배치 헤드 길이 차원

        x = rearrange(attention, '배치 헤드 길이 d_model -> 배치 길이 (헤드 d_model)') # 배치 헤드 길이 차원 -> 배치 길이 차원
        x = self.fc_o(x) 

        return x, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(d_model, d_ff),
                                    nn.ReLU(),
                                    nn.Dropout(drop_p), 
                                    nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = self.linear(x)
        return x