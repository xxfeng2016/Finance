import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import sys

from base import BaseModel

class FinCNN(BaseModel):
    def __init__(self, num_classes, d_k=3, d_model=16, kernel_size=3):
        super(FinCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_k, out_channels=d_model, kernel_size=kernel_size, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        # self.conv1 = nn.Sequential(
        #         nn.Conv1d(in_channels=d_k, out_channels=d_model, kernel_size=kernel_size, stride=1),
        #         nn.ReLU(),
        #         nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        #     )
        
        self.conv2 = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=kernel_size, stride=1),
                nn.AdaptiveAvgPool1d(1),  # Global Average Pooling
                nn.Flatten()
            )

        self.fc1 = nn.Linear(d_model*2, 128)  # GAP으로 인해 Linear의 input dimention == conv2의 out_channels
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        ####################
        x = nn.ReLU(x)
        x = self.pool(x)
        ####################
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x