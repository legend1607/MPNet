import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

# -------------------------------
# Dataset for single-step training
# -------------------------------
class PathDataset(Dataset):
    def __init__(self, npz_path, normalize=True, grid_size=160):
        data = np.load(npz_path, allow_pickle=True)
        self.grids = data['grids']          # [num_env, H, W]
        self.sample_envid = data['sample_envid']
        self.paths = data['paths']          # [num_samples, max_len, 2]
        self.masks = data['masks']          # [num_samples, max_len]
        self.goal = data['goal']            # [num_samples, 2]
        self.normalize = normalize
        self.grid_size = grid_size          # 栅格尺寸，用于归一化

        # Flatten each path into single-step samples
        self.samples = []
        for idx in range(len(self.sample_envid)):
            env_idx = int(self.sample_envid[idx])
            path = self.paths[idx]
            mask = self.masks[idx]
            goal = self.goal[idx]
            for t in range(len(path)-1):
                if mask[t] > 0:
                    self.samples.append({
                        'grid_idx': env_idx,
                        'current_pos': path[t],
                        'next_pos': path[t+1],
                        'goal_pos': goal
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        grid = self.grids[s['grid_idx']].astype(np.float32)
        current_pos = s['current_pos'].astype(np.float32)
        next_pos = s['next_pos'].astype(np.float32)
        goal_pos = s['goal_pos'].astype(np.float32)

        # 归一化到 [0,1]
        if self.normalize:
            current_pos = current_pos / self.grid_size
            next_pos = next_pos / self.grid_size
            goal_pos = goal_pos / self.grid_size

        return {
            'grid': torch.from_numpy(grid),          # H, W
            'current_pos': torch.from_numpy(current_pos),  # 2
            'next_pos': torch.from_numpy(next_pos),       # 2
            'goal_pos': torch.from_numpy(goal_pos)        # 2
        }

# -------------------------------
# Training function
# -------------------------------
def train_single_step(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # -------------------------------
    # Load Dataset
    # -------------------------------
    train_ds = PathDataset(args.train_npz)
    val_ds = PathDataset(args.val_npz)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # -------------------------------
    # Load Encoder
    # -------------------------------
    from MPNet.AE.cae_2d import Encoder_CNN_2D  # 自己的 CNN Encoder
    from MPNet.model import MLP_original        # MLP_original

    # 获取 Encoder latent_dim
    grids_data = np.load(args.train_npz)['grids']
    mask_size = grids_data.shape[1]
    encoder = Encoder_CNN_2D(mask_size=mask_size).to(device)
    encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    with torch.no_grad():
        dummy_grid = torch.from_numpy(grids_data[0:1]).unsqueeze(1).float().to(device)  # 1,1,H,W
        latent_dummy, _ = encoder(dummy_grid)
        latent_dim = latent_dummy.shape[1]

    # -------------------------------
    # Build MLP
    # -------------------------------
    mlp_input_dim = latent_dim + 4  # latent + current_pos(2) + goal_pos(2)
    mlp = MLP_original(input_size=mlp_input_dim, output_size=2).to(device)
    optimizer = Adam(mlp.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard"))
    best_val_loss = float('inf')

    # -------------------------------
    # Training Loop
    # -------------------------------
    for epoch in range(args.epochs):
        mlp.train()
        epoch_loss = 0.0
        total_samples = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training", ncols=120)

        for batch in train_bar:
            grids = batch['grid'].unsqueeze(1).float().to(device)       # B,1,H,W
            current_pos = batch['current_pos'].float().to(device)       # B,2
            next_pos = batch['next_pos'].float().to(device)             # B,2
            goal_pos = batch['goal_pos'].float().to(device)             # B,2

            # 编码环境
            with torch.no_grad():
                latent, _ = encoder(grids)                             # B, latent_dim

            # 构造 MLP 输入
            mlp_input = torch.cat([latent, current_pos, goal_pos], dim=-1)  # B, latent+4

            # 前向 + 反向传播
            pred = mlp(mlp_input)
            loss = criterion(pred, next_pos)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * grids.size(0)
            total_samples += grids.size(0)

            train_bar.set_postfix({"loss": f"{loss.item():.6f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        avg_train_loss = epoch_loss / (total_samples + 1e-12)
        writer.add_scalar('Loss/train', avg_train_loss, epoch+1)

        # -------------------------------
        # Validation
        # -------------------------------
        mlp.eval()
        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                grids = batch['grid'].unsqueeze(1).float().to(device)
                current_pos = batch['current_pos'].float().to(device)
                next_pos = batch['next_pos'].float().to(device)
                goal_pos = batch['goal_pos'].float().to(device)

                latent, _ = encoder(grids)
                mlp_input = torch.cat([latent, current_pos, goal_pos], dim=-1)
                pred = mlp(mlp_input)

                val_loss_sum += criterion(pred, next_pos).item() * grids.size(0)
                val_samples += grids.size(0)

        avg_val_loss = val_loss_sum / (val_samples + 1e-12)
        writer.add_scalar('Loss/val', avg_val_loss, epoch+1)

        print(f"[Epoch {epoch+1}/{args.epochs}] Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")

        # -------------------------------
        # Checkpoint
        # -------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(mlp.state_dict(), os.path.join(args.save_dir, 'mlp_best.pth'))
            print(">>> Saved new BEST model!")

        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save(mlp.state_dict(), os.path.join(args.save_dir, f'mlp_epoch_{epoch+1}.pth'))

    torch.save(mlp.state_dict(), os.path.join(args.save_dir, 'mlp_final.pth'))
    writer.close()
    print("Training finished. Final model saved.")

# -------------------------------
# Args
# -------------------------------
class Args:
    train_npz = "data/random_2d/train.npz"
    val_npz = "data/random_2d/val.npz"
    encoder_ckpt = "models/cae/encoder_best.pth"
    save_dir = "results/pointgen_single_step"
    batch_size = 64
    epochs = 100
    lr = 1e-4
    workers = 4
    checkpoint_every = 10

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()
    args = Args()
    train_single_step(args)
