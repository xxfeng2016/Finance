import torch

from tqdm import tqdm
from train_conf import *
import math
import matplotlib.pyplot as plt

def loss_epoch(model, DL, criterion, optimizer = None, scheduler = None):
    N = len(DL.dataset)
    rloss=0
    
    for input, label in tqdm(DL, desc=f"{model.device} : Train", position=1):
        
        # inference
        y_hat, atten_encs = model(input)
        pred = torch.max(y_hat, 1).indices.reshape(-1,1) # 임계값 0.5 의 경우
        
        # loss
        loss = criterion(pred.float(), label.float().to(pred.device)) #
        
        # update
        if optimizer is not None:
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # loss accumulation
        loss_b = loss.item() * input.shape[0]
        rloss += loss_b
        
    loss_e = rloss/N
    return loss_e

def Train(model, train_DL, val_DL, criterion, optimizer, scheduler = None):
    loss_history = {"train": [], "val": []}
    best_loss = 9999

    for ep in tqdm(range(epoch), desc="Epoch", position=0):
        model.train() # train mode로 전환
        train_loss = loss_epoch(model, train_DL, criterion, optimizer = optimizer, scheduler = scheduler)
        loss_history["train"] += [train_loss]
    
        model.eval() # validation mode로 전환
        
        with torch.no_grad():
            val_loss = loss_epoch(model, val_DL, criterion)
            loss_history["val"] += [val_loss]
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"model": model,
                            "ep": ep,
                            "optimizer": optimizer,
                            "scheduler": scheduler,}, save_model_path)
        # print loss
        print(f"Epoch {ep+1}: train loss: {train_loss:.5f}   val loss: {val_loss:.5f}   current_LR: {optimizer.param_groups[0]['lr']:.8f}")
        print("-" * 20)

    torch.save({"loss_history": loss_history,
                "EPOCH": epoch,
                "BATCH_SIZE": batch}, save_history_path)
    
def Test(model, test_DL, criterion):
    model.eval() # test mode로 전환
    
    with torch.no_grad():
        test_loss = loss_epoch(model, test_DL, criterion)
        
    print(f"Test loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, LR_scale = 1.0):
        self.optimizer = optimizer
        self.current_step = 0
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.LR_scale = LR_scale

    def step(self):
        self.current_step += 1
        lrate = self.LR_scale * (self.d_model ** -0.5) * min(self.current_step ** -0.5, self.current_step * self.warmup_steps ** -1.5)
        self.optimizer.param_groups[0]['lr'] = lrate

def plot_scheduler(scheduler_name, optimizer, scheduler, total_steps): # LR curve 보기
    lr_history = []
    steps = range(1, total_steps)

    for _ in steps: # step마다 lr 기록
        lr_history += [optimizer.param_groups[0]['lr']]
        scheduler.step()

    plt.figure()
    if scheduler_name == 'Noam':
        if total_steps == 100000:
            plt.plot(steps, (512 ** -0.5) * torch.tensor(steps) ** -0.5, 'g--', linewidth=1, label=r"$d_{\mathrm{model}}^{-0.5} \cdot \mathrm{step}^{-0.5}$")
            plt.plot(steps, (512 ** -0.5) * torch.tensor(steps) * 4000 ** -1.5, 'r--', linewidth=1, label=r"$d_{\mathrm{model}}^{-0.5} \cdot \mathrm{step} \cdot \mathrm{warmup\_steps}^{-1.5}$")
        plt.plot(steps, lr_history, 'b', linewidth=2, alpha=0.8, label="Learning Rate")
    elif scheduler_name == 'Cos':
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


# model = Model(max_len, n_layers, d_model, d_ff, n_heads, drop_p).to(device)
# model.eval()

# with torch.no_grad():
#     for x, y in train_DL:
        
#         # 1단계 임베딩, 포지션 인코딩
#         token_embed = TokenEmbedding(512, map_size, pad_idx)
#         pos_embed = PositionalEncoding(512)
#         # tk_embedding = TokenEmbedding(512)
#         # pos_encodding = PositionalEncoding(512, max_len, device)
#         # tm = nn.Embedding(num_embeddings=map_size, embedding_dim=512, padding_idx=pad_idx, device=device)
        
#         embed = Embedding(token_embed, pos_embed)
#         print(embed(x))
#         # print(tk_embedding(x).to(device))
        
        
#         # Example input: batch of indices
#         # Get embeddings for these indices
        
#         # print(embedded)

#         break