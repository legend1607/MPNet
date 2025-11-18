import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# -------------------------------
# Dataset with mask
# -------------------------------
class PathDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file, allow_pickle=True)
        self.grids = data['grids']       # (N,H,W)
        self.starts = data['start']      # (N,2)
        self.goals = data['goal']        # (N,2)
        self.paths = data['paths']       # (N,max_len,2)
        self.masks = data['masks']       # (N,max_len)

        self.samples = []
        for i in range(len(self.paths)):
            path = self.paths[i]         # (max_len,2)
            mask = self.masks[i]         # (max_len,)
            start = self.starts[i]
            goal = self.goals[i]
            grid = self.grids[i]

            for t in range(len(path)-1):
                if mask[t+1] == 0:
                    continue
                sample = {
                    'grid': grid.astype(np.float32),
                    'current_pos': path[t].astype(np.float32),
                    'next_pos': path[t+1].astype(np.float32),
                    'start_pos': start.astype(np.float32),
                    'goal_pos': goal.astype(np.float32),
                    'mask': float(mask[t+1])
                }
                sample['prev_pos'] = path[t-1].astype(np.float32) if t>0 else path[t].astype(np.float32)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'grid': torch.from_numpy(s['grid']).float(),
            'current_pos': torch.from_numpy(s['current_pos']).float(),
            'next_pos': torch.from_numpy(s['next_pos']).float(),
            'start_pos': torch.from_numpy(s['start_pos']).float(),
            'goal_pos': torch.from_numpy(s['goal_pos']).float(),
            'prev_pos': torch.from_numpy(s['prev_pos']).float(),
            'mask': torch.tensor(s['mask'], dtype=torch.float32)
        }

# -------------------------------
# 导入模型
# -------------------------------
from MPNet.AE.cae_2d import Encoder_CNN_2D
from MPNet.model import MLP

# -------------------------------
# 超参数
# -------------------------------
train_npz = 'data.random_2d.train.npz'
val_npz   = 'data.random_2d.val.npz'
latent_dim = 28
batch_size = 64
num_epochs = 200
learning_rate = 1e-3
dt = 0.1

v_max = 0.1
a_max = 0.5
lambda_v = 0.1
lambda_a = 0.1
alpha_nll = 0.3
beta_goal = 0.5

save_dir = './models'
os.makedirs(save_dir, exist_ok=True)

# -------------------------------
# Dataset & DataLoader
# -------------------------------
train_ds = PathDataset(train_npz)
val_ds   = PathDataset(val_npz)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# -------------------------------
# Encoder CAE
# -------------------------------
encoder = Encoder_CNN_2D(mask_size=train_ds.grids.shape[1], latent_dim=latent_dim).cuda()
encoder_ckpt = 'encoder_ckpt.pth'
encoder.load_state_dict(torch.load(encoder_ckpt))
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

# -------------------------------
# MLP_with_uncertainty
# -------------------------------
mlp_input_dim = latent_dim + 4
mlp = MLP(input_size=mlp_input_dim).cuda()
optimizer = optim.Adagrad(mlp.parameters(), lr=learning_rate)

# -------------------------------
# TensorBoard
# -------------------------------
writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))

# -------------------------------
# 训练循环
# -------------------------------
for epoch in range(num_epochs):
    mlp.train()
    epoch_loss = 0.0

    for batch in train_loader:
        grids = batch['grid'].unsqueeze(1).cuda()
        current_pos = batch['current_pos'].cuda()
        goal_pos    = batch['goal_pos'].cuda()
        next_pos    = batch['next_pos'].cuda()
        prev_pos    = batch['prev_pos'].cuda()
        mask        = batch['mask'].unsqueeze(1).cuda()  # (B,1)

        with torch.no_grad():
            latent, _ = encoder(grids)

        mlp_input = torch.cat([latent, current_pos, goal_pos], dim=1)
        mu, sigma = mlp(mlp_input)

        # --- 弱监督 NLL ---
        nll_loss = 0.5 * ((next_pos - mu)/sigma)**2 + torch.log(sigma[:,0]*sigma[:,1])
        nll_loss = (nll_loss * mask).sum() / mask.sum()

        # --- 速度/加速度约束 ---
        vel = (mu - current_pos) / dt
        acc = (mu - 2*current_pos + prev_pos) / (dt**2)
        vel_loss = torch.mean(torch.relu(torch.norm(vel, dim=1) - v_max)**2 * mask.squeeze())
        acc_loss = torch.mean(torch.relu(torch.norm(acc, dim=1) - a_max)**2 * mask.squeeze())

        # --- 终点约束 ---
        goal_mask = mask.squeeze()
        goal_loss = torch.mean(torch.norm(goal_pos - mu, dim=1)**2 * goal_mask)

        # --- 总损失 ---
        loss = alpha_nll * nll_loss + lambda_v * vel_loss + lambda_a * acc_loss + beta_goal * goal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_epoch_loss:.6f}")
    writer.add_scalar('Loss/train', avg_epoch_loss, epoch+1)

    # -------------------------------
    # 验证
    # -------------------------------
    mlp.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            grids = batch['grid'].unsqueeze(1).cuda()
            current_pos = batch['current_pos'].cuda()
            goal_pos    = batch['goal_pos'].cuda()
            next_pos    = batch['next_pos'].cuda()
            prev_pos    = batch['prev_pos'].cuda()
            mask        = batch['mask'].unsqueeze(1).cuda()

            latent, _ = encoder(grids)
            mlp_input = torch.cat([latent, current_pos, goal_pos], dim=1)
            mu, sigma = mlp(mlp_input)

            nll_loss = 0.5 * ((next_pos - mu)/sigma)**2 + torch.log(sigma[:,0]*sigma[:,1])
            nll_loss = (nll_loss * mask).sum() / mask.sum()

            vel = (mu - current_pos) / dt
            acc = (mu - 2*current_pos + prev_pos) / (dt**2)
            vel_loss = torch.mean(torch.relu(torch.norm(vel, dim=1) - v_max)**2 * mask.squeeze())
            acc_loss = torch.mean(torch.relu(torch.norm(acc, dim=1) - a_max)**2 * mask.squeeze())

            goal_mask = mask.squeeze()
            goal_loss = torch.mean(torch.norm(goal_pos - mu, dim=1)**2 * goal_mask)

            loss = alpha_nll * nll_loss + lambda_v * vel_loss + lambda_a * acc_loss + beta_goal * goal_loss
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar('Loss/val', avg_val_loss, epoch+1)
    print(f"[Epoch {epoch+1}/{num_epochs}] Val Loss: {avg_val_loss:.6f}")

    # -------------------------------
    # 保存模型
    # -------------------------------
    if (epoch+1) % 50 == 0:
        torch.save(mlp.state_dict(), os.path.join(save_dir, f'mlp_epoch_{epoch+1}.pth'))

torch.save(mlp.state_dict(), os.path.join(save_dir, 'mlp_final.pth'))
writer.close()
