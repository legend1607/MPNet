#!/usr/bin/env python3
"""
train_mlp_with_encoder.py

训练 MLP 来预测下一步点（步长 + theta/phi 方向），使用预训练 CAE encoder 提供环境 embedding。
输入 npz 必须包含字段：env_grids / path / start / goal / sample_envid
"""
import os, argparse, logging, numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from MPNet.AE.cae_2d import Encoder_CNN_2D
from MPNet.model import MLPPathPredictor

# ----------------------------
# Dataset
# ----------------------------
class PathDatasetFromNPZ(Dataset):
    def __init__(self, npz_file, num_sector_theta=8, num_sector_phi=16):
        self.num_sector_theta = num_sector_theta
        self.num_sector_phi = num_sector_phi

        data = np.load(npz_file, allow_pickle=True)
        self.env_grids = data['env_grids']
        self.paths = data['path']
        self.start = data['start']
        self.goal = data['goal']
        self.sample_envid = data['sample_envid']

        self.max_len = max([len(p) for p in self.paths])

        self.dataset, self.norms, self.theta_idx, self.phi_idx, self.masks = [], [], [], [], []

        for path in self.paths:
            L = len(path)
            inp = np.zeros((self.max_len, path.shape[1]), dtype=np.float32)
            inp[:L] = path
            if L < self.max_len:
                inp[L:] = path[-1]

            mask = np.zeros(self.max_len, dtype=np.float32)
            mask[:L-1] = 1.0

            theta = np.zeros(self.max_len, dtype=np.int64)
            phi = np.zeros(self.max_len, dtype=np.int64)
            norm = np.zeros(self.max_len, dtype=np.float32)

            for k in range(L-1):
                diff = path[k+1]-path[k]
                r = np.linalg.norm(diff)
                norm[k] = r

                # 防止 r=0
                t = 0.0 if r == 0 else np.arccos(np.clip(diff[-1]/r, -1.0, 1.0))
                p = np.arctan2(diff[1], diff[0])

                # 离散化并 clip
                theta_idx = int(np.degrees(t)//(180/self.num_sector_theta))
                phi_idx = int(np.degrees(p)//(360/self.num_sector_phi) + self.num_sector_phi//2)

                theta[k] = np.clip(theta_idx, 0, self.num_sector_theta-1)
                phi[k] = np.clip(phi_idx, 0, self.num_sector_phi-1)

            # 填充末尾，保持最后有效值
            if L < self.max_len:
                theta[L:] = theta[L-1]
                phi[L:] = phi[L-1]
                norm[L:] = norm[L-1]

            self.dataset.append(inp)
            self.masks.append(mask)
            self.norms.append(norm)
            self.theta_idx.append(theta)
            self.phi_idx.append(phi)

        self.dataset = np.array(self.dataset, dtype=np.float32)
        self.norms = np.array(self.norms, dtype=np.float32)
        self.theta_idx = np.array(self.theta_idx, dtype=np.int64)
        self.phi_idx = np.array(self.phi_idx, dtype=np.int64)
        self.masks = np.array(self.masks, dtype=np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'input': torch.from_numpy(self.dataset[idx]),
            'mask': torch.from_numpy(self.masks[idx]),
            'norm': torch.from_numpy(self.norms[idx]),
            'orient_theta': torch.from_numpy(self.theta_idx[idx]),
            'orient_phi': torch.from_numpy(self.phi_idx[idx]),
            'start': torch.from_numpy(self.start[idx]),
            'goal': torch.from_numpy(self.goal[idx]),
            'env_grid': torch.from_numpy(self.env_grids[self.sample_envid[idx]])
        }

# ----------------------------
# Training
# ----------------------------
def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.save_dir,'train.log'),
                        level=logging.INFO, format='%(asctime)s %(message)s')
    log = lambda s: (print(s), logging.info(s))
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir,"tensorboard"))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    log(f"Using device: {device}")

    # Dataset
    train_ds = PathDatasetFromNPZ(args.train_npz, args.num_sector_theta, args.num_sector_phi)
    val_ds = PathDatasetFromNPZ(args.val_npz, args.num_sector_theta, args.num_sector_phi)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    log(f"Loaded train={len(train_ds)}, val={len(val_ds)}")

    # CAE Encoder
    encoder = Encoder_CNN_2D(input_size=train_ds.env_grids.shape[1], latent_dim=args.latent_dim)
    state = torch.load(args.encoder_ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
    encoder.load_state_dict(state)
    encoder.to(device).eval()
    for p in encoder.parameters(): p.requires_grad = False
    log("Loaded CAE encoder (frozen)")

    # MLP
    mlp_input_dim = args.latent_dim + train_ds.dataset.shape[2]*2
    mlp = MLPPathPredictor(input_size=mlp_input_dim,
                           num_sector_theta=args.num_sector_theta,
                           num_sector_phi=args.num_sector_phi,
                           dropout_p=args.dropout_p).to(device)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, args.epochs+1):
        mlp.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            mask = batch['mask'].to(device)
            norm_target = batch['norm'].to(device)
            theta_target = batch['orient_theta'].to(device)
            phi_target = batch['orient_phi'].to(device)
            start = batch['start'].to(device)
            goal = batch['goal'].to(device)
            env_grid = batch['env_grid'].to(device).unsqueeze(1)

            with torch.no_grad():
                obs_latent, _ = encoder(env_grid)

            B,L,_ = batch['input'].shape
            start_goal = torch.cat([start, goal], dim=-1).unsqueeze(1).repeat(1,L,1)
            mlp_input = torch.cat([obs_latent.unsqueeze(1).repeat(1,L,1), start_goal], dim=-1)

            optimizer.zero_grad()
            out_norm, out_theta_raw, out_phi_raw = mlp(mlp_input)

            loss_norm = ((out_norm.squeeze(-1)-norm_target)**2 * mask).sum() / mask.sum()
            loss_theta = nn.CrossEntropyLoss(reduction='none')(out_theta_raw.view(-1, args.num_sector_theta), theta_target.view(-1))
            loss_theta = (loss_theta.view_as(mask)*mask).sum()/mask.sum()
            loss_phi = nn.CrossEntropyLoss(reduction='none')(out_phi_raw.view(-1, args.num_sector_phi), phi_target.view(-1))
            loss_phi = (loss_phi.view_as(mask)*mask).sum()/mask.sum()
            loss = loss_norm + args.k2_theta*loss_theta + args.k2_phi*loss_phi

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss/len(train_loader)
        log(f"[Epoch {epoch}] train_loss={avg_train_loss:.6f}")
        writer.add_scalar("train_loss", avg_train_loss, epoch)

        # Validation
        mlp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                mask = batch['mask'].to(device)
                norm_target = batch['norm'].to(device)
                theta_target = batch['orient_theta'].to(device)
                phi_target = batch['orient_phi'].to(device)
                start = batch['start'].to(device)
                goal = batch['goal'].to(device)
                env_grid = batch['env_grid'].to(device).unsqueeze(1)

                obs_latent, _ = encoder(env_grid)
                B,L,_ = batch['input'].shape
                start_goal = torch.cat([start, goal], dim=-1).unsqueeze(1).repeat(1,L,1)
                mlp_input = torch.cat([obs_latent.unsqueeze(1).repeat(1,L,1), start_goal], dim=-1)

                out_norm, out_theta_raw, out_phi_raw = mlp(mlp_input)
                loss_norm = ((out_norm.squeeze(-1)-norm_target)**2 * mask).sum()/mask.sum()
                loss_theta = nn.CrossEntropyLoss(reduction='none')(out_theta_raw.view(-1, args.num_sector_theta), theta_target.view(-1))
                loss_theta = (loss_theta.view_as(mask)*mask).sum()/mask.sum()
                loss_phi = nn.CrossEntropyLoss(reduction='none')(out_phi_raw.view(-1, args.num_sector_phi), phi_target.view(-1))
                loss_phi = (loss_phi.view_as(mask)*mask).sum()/mask.sum()
                val_loss += (loss_norm + args.k2_theta*loss_theta + args.k2_phi*loss_phi).item()

        avg_val_loss = val_loss/len(val_loader)
        log(f"[Epoch {epoch}] val_loss={avg_val_loss:.6f}")
        writer.add_scalar("val_loss", avg_val_loss, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(mlp.state_dict(), os.path.join(args.save_dir,"mlp_best.pt"))
            log(f"Saved best model at epoch {epoch} with val_loss={best_val_loss:.6f}")

    writer.close()
    log("Training finished.")

# ----------------------------
# Argparse
# ----------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_npz', type=str, default='data/random_2d/train.npz')
    parser.add_argument('--val_npz', type=str, default='data/random_2d/val.npz')
    parser.add_argument('--encoder_ckpt', type=str, default='results/cae/encoder_best.pth')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_sector_theta', type=int, default=8)
    parser.add_argument('--num_sector_phi', type=int, default=16)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--k2_theta', type=float, default=4.0)
    parser.add_argument('--k2_phi', type=float, default=4.0)
    parser.add_argument('--force_cpu', action='store_true')
    args = parser.parse_args()
    train(args)
