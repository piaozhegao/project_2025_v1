import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from UNet_class_model import UNet3DClassifier

pkl_dir = r"D:\data_new_0521\data_train_SeoulSM"
ckpt_path = "./unet3d_0904_300_NFdata.pth"
batch_size = 8
lr = 1e-4
num_epochs = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_files = sorted([
    os.path.join(pkl_dir, f)
    for f in os.listdir(pkl_dir)
    if f.lower().endswith(".pkl")
])

y_files = []
n_files = []
for fp in all_files:
    with open(fp, 'rb') as f:
        d = pickle.load(f)
    if d.get('label') == 'F':
        y_files.append(fp)
    else:
        n_files.append(fp)

half = len(y_files) // 2
y_train = y_files[:half]
y_val = y_files[half:]

split_idx = int(len(n_files) * 0.2)
n_val = n_files[:split_idx]
n_train = n_files[split_idx:]

train_files = y_train + n_train
val_files = y_val   + n_val

print(f"Train samples: {len(train_files)} (F={len(y_train)}, N={len(n_train)})")
print(f"Val samples: {len(val_files)}   (F={len(y_val)}, N={len(n_val)})")

class PickleVolumeDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        with open(fp, 'rb') as f:
            d = pickle.load(f)

        vol = d['data'].astype(np.float32)
        vol = vol / (vol.max() if vol.max() > 0 else 1.0)

        if vol.ndim == 3:
            vol = np.expand_dims(vol, 0)

        lbl = d['label']
        label = 1 if lbl == 'F' else 0

        return torch.from_numpy(vol), label

train_ds = PickleVolumeDataset(train_files)
val_ds = PickleVolumeDataset(val_files)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)

model = UNet3DClassifier(in_channels=1, hidden_dim=256, num_classes=2).to(device)

state = torch.load(ckpt_path, map_location="cpu")
enc_state = {
    k.replace("conv_blocks_context", "context_blocks"): v
    for k, v in state.items()
    if k.startswith("conv_blocks_context")
}
model.encoder.load_state_dict(enc_state, strict=False)

for p in model.encoder.parameters():
    p.requires_grad = False
for p in model.encoder.context_blocks[-1].parameters():
    p.requires_grad = True
for p in model.encoder.context_blocks[-2].parameters():
    p.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    epoch_loss = correct = total = 0
    loop = tqdm(loader, desc="Train" if train else " Val ", leave=False)
    for vols, labels in loop:
        vols, labels = vols.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(vols)
            loss   = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epoch_loss += loss.item() * vols.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += vols.size(0)
    return epoch_loss/total, correct/total

best_val_acc  = 0.0
best_val_loss = float('inf')

for epoch in range(1, num_epochs+1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)

    print(f"[Epoch {epoch:02d}] "
          f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
          f" Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_unet3d_cls_acc_0904_allSM.pth")
        print("Saved best ACC model.")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_unet3d_cls_loss_0904_allSM.pth")
        print("Saved best LOSS model.")