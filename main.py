# from glob import glob
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
# from loader.custom_dataset import FinDataset, FinCollate
# from model.model import FinCNN

# import torch
# import torch.nn as nn
# from torch.optim import Adam

# import os
# from glob import glob

# BATCH_SIZE = 1
# learning_rate = 0.001
# num_epochs = 1
# num_classes= 2

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = FinCNN(num_classes=num_classes)
# model = model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train_set, valid_set = train_test_split(glob(os.path.join(os.path.abspath("data/raw"), "*.parquet")),shuffle=True, test_size=0.2, random_state=42)

# train_loader = enumerate(tqdm(DataLoader(FinDataset(data_paths=train_set), batch_size=BATCH_SIZE, collate_fn=FinCollate())))
# valid_loader = enumerate(tqdm(DataLoader(FinDataset(data_paths=valid_set), batch_size=BATCH_SIZE, collate_fn=FinCollate())))

# def metric(logit, target, batch_size):
#     corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
#     accuracy = 100.0 * corrects/batch_size
#     return accuracy.item()

# for epoch in range(num_epochs):
#     train_running_loss = 0.0
#     train_acc = 0.0

#     model = model.train()

#     ## training step
#     for i, (x, y) in train_loader:
        
#         x = x.to(device)
#         y = y.to(device)

#         ## forward + backprop + loss
#         logits = model(x)
#         loss = criterion(logits, y)
        
#         optimizer.zero_grad()
#         loss.backward()

#         ## update model params
#         optimizer.step()

#         train_running_loss += loss.detach().item()
#         train_acc += metric(logits, y, BATCH_SIZE)
    
#     model.eval()
#     print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
#           %(epoch, train_running_loss / i, train_acc/i))     
    
# test_acc = 0.0
# for i, (images, labels) in enumerate(valid_loader, 0):
#     images = images.to(device)
#     labels = labels.to(device)
#     outputs = model(images)
#     test_acc += metric(outputs, labels, BATCH_SIZE)
        
# print('Test Accuracy: %.2f'%( test_acc/i))