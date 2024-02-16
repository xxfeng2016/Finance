from base import BaseDataLoader
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .custom_dataset import FinDataset, FinCollate

import os
from glob import glob
from copy import deepcopy
from torch.utils.data import DataLoader

class FinDataLoader():
    """
    
    """
    def __init__(self, data_root, batch_size=2, shuffle=False, validation_split=0.0, num_workers=1, training=True, window_size=(3, 0), sliding_size=(0, 30), target_size=(1, 0)):
        self.data_dir = glob(os.path.join(os.path.abspath(data_root), "*.parquet"))
        self.train_dataset, self.valid_dataset = train_test_split(self.data_dir, shuffle=True, test_size=validation_split)
        
        # self.init_kwargs = {
        #     'dataset': FinDataset(self.train_dataset),
        #     'batch_size': batch_size,
        #     'shuffle': shuffle,
        #     'collate_fn': FinCollate(),
        #     'num_workers': num_workers
        # }
        
        self.train_loader = DataLoader(dataset=FinDataset(self.train_dataset, window_size, sliding_size, target_size), batch_size=batch_size, shuffle=shuffle, collate_fn=FinCollate(), num_workers=num_workers)
        self.valid_loader = DataLoader(dataset=FinDataset(self.train_dataset, window_size, sliding_size, target_size), batch_size=batch_size, shuffle=shuffle, collate_fn=FinCollate(), num_workers=num_workers)