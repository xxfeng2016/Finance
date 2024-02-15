from base import BaseDataLoader
from .custom_dataset import FinDataset
import os
from glob import glob

class FinDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = glob(os.path.join(os.path.abspath(data_dir), "*.parquet"))
        self.dataset = FinDataset(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)