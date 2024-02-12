import torch
import torch.nn as nn
import os
from glob import glob
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count()

lr = 1e-4
map_size = 5853 # pad 인덱스 포함
max_len = 10997
pad_idx = 5852

criterion = nn.BCELoss(weight=None, reduction="none") # nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss(ignore_index = pad_idx)

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
batch = 1
epoch = 100

n_layers = 4
d_model = 128
d_ff = 512
n_heads = 4

LAMBDA = 0
drop_p = 0.1

LR_scale = 0.5
LR_init = 5e-4
T0 = 1500
T_mult = 2

save_model_path = f"test{os.path.sep}results{os.path.sep}Encoder.pt"
save_history_path = f"test{os.path.sep}results{os.path.sep}Encoder_history.json"

new_model_train = True
scheduler_name = 'Noam'

with open(f"data{os.path.sep}encoding_map.json", "r") as f:
    encoding_map = json.load(f)