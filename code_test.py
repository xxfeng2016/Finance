from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from loader.custom_dataset import FinDataset, collate_func
from model.model import FinCNN
import os
import torch

dataset = FinDataset(data_paths=glob(os.path.join("data", "raw", "*.parquet")))

progress = tqdm(DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_func(True)), desc="Loading Data")
model = FinCNN(num_classes=2)

for batch_idx, (frame_idx, inv_lambda, x, y) in enumerate(progress):
    y_prob = model(x)
    y_pred = torch.argmax(y_prob, dim=1).reshape(-1, 1)
