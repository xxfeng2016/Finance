import numpy as np
import pandas as pd
import torch

# First Party

from torch.utils.data import DataLoader
from custom_dataset import DS, collate_func
from glob import glob
from tqdm import tqdm
import os
# from base import BaseDataLoader
from sklearn.model_selection import train_test_split
from datasets import DatasetDict
from scipy.special import inv_boxcox

dataset = DS(data_paths=sorted(glob("data/raw/*.parquet")))
# dataset
saved_data = []

# progress = tqdm(DataLoader(dataset, batch_size=2, shuffle=False))

for x, y in dataset:
    print(x)
    # print("inverse 거래량 :", inv_boxcox(volume[0], volume[1]))
    # print("원본 거래량 :", x["CNTG_VOL"].values + 1e-9)
    
    # print("Batch Idx", idx)




# class FinDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         # self.data_dir = data_dir
#         self.dataset = data_dir
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
#     def _split_sampler(self, split):
#         train_sampler, valid_sampler = train_test_split(x, test_size=0.2, random_state=32, shuffle=True)

#         # sampler의 Random Shuffle 옵션 끄기.
#         self.shuffle = False
#         self.n_samples = len(train_sampler)

#         return train_sampler, valid_sampler

#     def split_validation(self):
#         if self.valid_sampler is None:
#             return None
#         else:
#             return DataLoader(dataset=self.valid_sampler, **self.init_kwargs)