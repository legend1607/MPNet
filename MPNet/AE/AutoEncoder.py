import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime

# ---------------- 使用可变尺寸 Encoder/Decoder ----------------
from cae_2d import Encoder_CNN_2D, Decoder_CNN_2D
from CNN_data_loader import load_dataset_mask

def main(args):
    os.makedirs(args.model_path, exist_ok=True)
    log_file = os.path.join(args.model_path, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # ---------------- 数据 ----------------
    train_masks = load_dataset_mask(split="train", N=400).astype(np.float32)
    val_masks   = load_dataset_mask(split="val", N=50).astype(np.float32)
    print(f"Train masks shape: {train_masks.shape}, Val masks shape: {val_masks.shape}")

    train_tensor = torch.from_numpy(train_masks).unsqueeze(1)  # (N,1,H,W)
    val_tensor   = torch.from_numpy(val_masks).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_tensor), batch_size=args.batch_size, shuffle=False)

    # ---------------- 模型 ----------------
    latent_dim = 128
    encoder = Encoder_CNN_2D(latent_dim=latent_dim)
    decoder = Decoder_CNN_2D(latent_dim=latent_dim)

    if args.preload_encoder_decoder:
        encoder.load_state_dict(torch.load(args.preload_encoder_path))
        decoder.load_state_dict(torch.load(args.preload_decoder_path))
        print("Loaded pre-trained encoder & decoder")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device)
    decoder.to(device)

    # ---------------- 损失和优化器 ----------------
    # Decoder 已输出 sigmoid(out) → 必须使用 BCELoss
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.learning_rate
    )

    # ---------------- 训练 ----------------
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 50

    for epoch in range(1, args.num_epochs + 1):

        encoder.train()
        decoder.train()
        train_loss = 0

        for batch, in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            latent, enc_feats = encoder(batch)
            out = decoder(latent, enc_feats)

            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # ---------------- 验证 ----------------
        encoder.eval()
        decoder.eval()
        val_loss = 0

        with torch.no_grad():
            for batch, in val_loader:
                batch = batch.to(device)
                latent, enc_feats = encoder(batch)
                out = decoder(latent, enc_feats)
                vloss = criterion(out, batch)
                val_loss += vloss.item() * batch.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        log_str = f"[Epoch {epoch}/{args.num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        print(log_str)
        with open(log_file, "a") as f:
            f.write(log_str + "\n")

        # ---------------- 保存最优模型 ----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            torch.save(encoder.state_dict(), os.path.join(args.model_path, "encoder_best.pth"))
            torch.save(decoder.state_dict(), os.path.join(args.model_path, "decoder_best.pth"))
            print(f"Saved new best model with Val Loss: {best_val_loss:.6f}")

        else:
            epochs_no_improve += 1

        # ---------------- Early stop ----------------
        if epochs_no_improve >= early_stop_patience:
            print(f"No improvement for {early_stop_patience} epochs, stopping early.")
            break

        # ---------------- 周期保存 ----------------
        if epoch % args.save_every == 0 or epoch == args.num_epochs:
            torch.save(encoder.state_dict(), os.path.join(args.model_path, f"encoder_epoch_{epoch}.pth"))
            torch.save(decoder.state_dict(), os.path.join(args.model_path, f"decoder_epoch_{epoch}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/cae')
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--preload_encoder_decoder', type=int, default=0)
    parser.add_argument('--preload_encoder_path', type=str, default='')
    parser.add_argument('--preload_decoder_path', type=str, default='')

    args = parser.parse_args()
    main(args)
