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
    """
    使用 Encoder_CNN_2D 直接将 grid -> latent
    返回 CPU tensor，训练循环再转 GPU
    """
    def __init__(self, npz_path, encoder, device="cpu"):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        self.grids = data["grids"]
        self.start = data["start"]
        self.goal = data["goal"]
        self.paths = data["paths"]

        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.device = device

        # 展开所有 path segment：每个样本是 (env_id, t)
        self.index = []
        for env_id, path in enumerate(self.paths):
            L = len(path)
            for t in range(L - 1):
                self.index.append((env_id, t))

    def __len__(self):
        return len(self.index)

    @torch.no_grad()
    def __getitem__(self, idx):
        env_id, t = self.index[idx]

        # --------------------
        # CNN Encoder 对 grid 编码 (CPU tensor)
        # --------------------
        grid = self.grids[env_id]  # numpy (H, W)
        grid = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # CPU tensor

        latent, _ = self.encoder(grid.to(self.device))  # 转 GPU 编码
        latent = latent.squeeze(0).cpu()  # 返回 CPU tensor

        # 当前点 & 下一个点
        s = torch.tensor(self.start[env_id], dtype=torch.float32)
        g = torch.tensor(self.goal[env_id], dtype=torch.float32)
        path = self.paths[env_id]
        curr = torch.tensor(path[t], dtype=torch.float32)
        nxt = torch.tensor(path[t+1], dtype=torch.float32)

        return latent, s, g, curr, nxt


def get_dataloader_with_encoder(base_dir, encoder, batch_size=64, device="cpu", workers=4):
    train_ds = PathDataset(f"{base_dir}/train.npz", encoder, device=device)
    val_ds   = PathDataset(f"{base_dir}/val.npz",   encoder, device=device)
    test_ds  = PathDataset(f"{base_dir}/test.npz",  encoder, device=device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)
    return train_loader, val_loader, test_loader
# -------------------------------
# Gaussian NLL loss
# -------------------------------
def gaussian_nll(mu, sigma, target):
    """
    mu:     (B, K, D)
    sigma:  (B, K, D)
    target: (B, D)
    """
    target = target.unsqueeze(1)  # -> (B, 1, D)
    ll = -0.5 * (
        torch.log(2 * torch.pi * sigma**2) +
        (target - mu)**2 / (sigma**2)
    ).sum(dim=-1)  # (B, K)
    best_ll = ll.max(dim=1)[0]
    return -best_ll.mean()


def train_single_step(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    from MPNet.AE.cae_2d import Encoder_CNN_2D
    from MPNet.model import MLP_fusion_gaussian

    # -------------------------------
    # Load encoder
    # -------------------------------
    encoder = Encoder_CNN_2D(latent_dim=128)
    encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device))
    encoder.to(device)
    encoder.eval()

    # -------------------------------
    # DataLoader
    # -------------------------------
    train_loader, val_loader, test_loader = get_dataloader_with_encoder(
        base_dir=os.path.dirname(args.train_npz),
        encoder=encoder,
        batch_size=args.batch_size,
        device=device,
        workers=args.workers
    )

    # -------------------------------
    # Build MLP
    # -------------------------------
    joint_dim = 2
    num_candidates = 5
    model = MLP_fusion_gaussian(
        cae_input_size=128,
        joint_input_size=joint_dim*2,
        joint_dim=joint_dim,
        num_candidates=num_candidates
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard"))
    best_val_loss = float("inf")

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0

        for latent, s, g, curr, nxt in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            # 转 GPU
            latent = latent.to(device)
            s = s.to(device)
            g = g.to(device)
            curr = curr.to(device)
            nxt = nxt.to(device)

            joint_input = torch.cat([curr, g], dim=-1)
            mu, sigma = model(latent, joint_input)
            loss = gaussian_nll(mu, sigma, nxt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for latent, s, g, curr, nxt in val_loader:
                latent = latent.to(device)
                s = s.to(device)
                g = g.to(device)
                curr = curr.to(device)
                nxt = nxt.to(device)

                joint_input = torch.cat([curr, g], dim=-1)
                mu, sigma = model(latent, joint_input)
                val_loss += gaussian_nll(mu, sigma, nxt).item()

        avg_val_loss = val_loss / len(val_loader)

        # -------------------------------
        # Logging
        # -------------------------------
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # -------------------------------
        # Checkpoint
        # -------------------------------
        if epoch % args.checkpoint_every == 0 or avg_val_loss < best_val_loss:
            ckpt_path = os.path.join(args.save_dir, f"mlp_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, "mlp_best.pth"))
                print("Updated best model!")

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
