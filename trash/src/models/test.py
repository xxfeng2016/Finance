import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from torch.utils.data import DataLoader, random_split
from data_loader.custom_dataset import DS, collate_func
from train_conf import *
from Scheduler import NoamScheduler
import math
import matplotlib.pyplot as plt
from build import build_model
import numpy as np

import time
import sys
# from memory_usage import memory_usage

dataset = DS(data_paths=sorted(glob("data/ByContinuous_pandas/*.parquet")))
collate_fn = collate_func

# 데이터 분할
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn(True))

# for x, y in loader:
#     print(x)
#     print(y)

def loss_epoch(model, DL, criterion, optimizer=None, scheduler=None, scaler=None, is_train=True):
    hist = {"y_hat":[], "label":[]}

    if is_train:
        model.train()

    else:
        model.eval()
    
    N = 0
    rloss = 0
    correct_prediction_total = 0
    
    # 파일 단위
    for batch_idx, (x, y) in enumerate(start=1, iterable=loader):
        N += batch # 데이터를 시간 단위로 잘라 사용하므로 최대 길이를 모르니 batch씩 더해주기
        x, y = x.cuda(), y.cuda()
        
        start = time.time()
        y_hat, _ = model(x)
        # prde = y_hat.item() >= 0.5
        loss = criterion(y_hat.type(torch.float16), y)
        end = time.time()
            
        # Backward pass and optimization only if in training mode
        if is_train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

        # Loss accumulation
        loss_b = loss.item() * input.shape[0]
        rloss += loss_b
        loss_a = rloss / batch_idx

        # Calculate accuracy
        prediction = y_hat >= 0.7  # Using threshold of 0.5
        correct_prediction = prediction.float() == y
        correct_prediction_total += correct_prediction.sum().item()
        
        print(f'Batch {batch_idx}/{len(DL)}\tloss average : {loss_a:.6f}\tloss: {loss.item():.6f}\tAccuracy : {correct_prediction_total / ((batch_idx + 1) * input.shape[0]):.3f}\ttime : {(end-start)*1000:.3f}ms')

    loss_e = rloss / N
    overall_accuracy = correct_prediction_total / N
    
    return loss_e, overall_accuracy

def train_and_eval(model, train_DL, validation_DL, criterion, optimizer, scheduler=None, num_epochs=10):
    scaler = GradScaler()
    loss_history = {"train": [], "val": []}
    best_loss = 9999
    
    for ep in tqdm(range(num_epochs), desc="Epoch"):
        train_loss, train_accuracy = loss_epoch(model, train_DL, criterion, optimizer, scheduler, scaler, is_train=True)
                
        loss_history["train"] += [train_loss]
        model.eval()
        
        with torch.no_grad():
            val_loss, val_accuracy = loss_epoch(model, validation_DL, criterion, is_train=False)
            loss_history["val"] += [val_loss]
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"model": model,
                            "ep": ep,
                            "optimizer": optimizer,
                            "scheduler": scheduler}, save_model_path)
        
        # print loss
        print(f"Epoch {ep+1}: Train Loss: {train_loss:.5f}, Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Epoch {ep+1}: Val Loss: {val_loss:.5f}, Val Accuracy: {val_accuracy * 100:.2f}%")
        print("-" * 20)

    torch.save({"loss_history": loss_history,
                "EPOCH": num_epochs,
                "BATCH_SIZE": batch}, save_history_path)
    
def Test(model, test_DL, criterion):
    model.eval() # test mode로 전환
    
    with torch.no_grad():
        test_loss = loss_epoch(model, test_DL, criterion)
        
    print(f"Test loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num

def plot_scheduler(optimizer, scheduler, total_steps): # LR curve 보기
    lr_history = []
    steps = range(1, total_steps)

    for _ in steps: # step마다 lr 기록
        lr_history += [optimizer.param_groups[0]['lr']]
        scheduler.step()

    plt.figure()
    plt.plot(steps, (512 ** -0.5) * torch.tensor(steps) ** -0.5, 'g--', linewidth=1, label=r"$d_{\mathrm{model}}^{-0.5} \cdot \mathrm{step}^{-0.5}$")
    plt.plot(steps, (512 ** -0.5) * torch.tensor(steps) * 4000 ** -1.5, 'r--', linewidth=1, label=r"$d_{\mathrm{model}}^{-0.5} \cdot \mathrm{step} \cdot \mathrm{warmup\_steps}^{-1.5}$")
    plt.plot(steps, lr_history, 'b', linewidth=2, alpha=0.8, label="Learning Rate")
    plt.ylim([-0.1*max(lr_history), 1.2*max(lr_history)])
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.grid()
    plt.legend()
    plt.show()

# optimizer = Adam(nn.Linear(1, 1).parameters(), lr=0)
# scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, LR_scale=LR_scale)
# plot_scheduler(scheduler_name = 'Noam', optimizer = optimizer, scheduler = scheduler, total_steps = int(len(train_dataset) * epoch / batch))

model = build_model()

# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = NoamScheduler(optimizer=optimizer, model_size=d_model)
# plot_scheduler(optimizer, scheduler, total_steps=int(len(train_DL)/batch * epoch))
if new_model_train:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=0,
                            betas=(0.9, 0.98), eps=1e-9,
                            weight_decay=LAMBDA) # 논문에서 제시한 beta와 eps 사용, l2-Regularization은 한번 써봄 & 맨 처음 step 의 LR=0으로 출발 (warm-up)
    scheduler = NoamScheduler(optimizer=optimizer, model_size=d_model, warmup=4000, scale_factor=LR_scale)
    train_and_eval(model, loader, loader, criterion, optimizer, scheduler=None, num_epochs=10)