import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

class NoamScheduler:
    def __init__(self, optimizer, model_size, warmup, scale_factor=1.0):
        self.optimizer = optimizer
        self.model_size = model_size
        self.scale_factor = scale_factor
        self.warmup = warmup
        self._step = 0

    def step(self):
        """학습률을 업데이트하고 옵티마이저 스텝을 수행하는 메소드"""
        self._step += 1
        lr = self.scale_factor * (self.model_size ** (-0.5) * 
                            min(self._step ** (-0.5), self._step * self.warmup ** (-1.5)))
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()