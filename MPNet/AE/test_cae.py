import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cae_2d as AE
import cv2

# ---------------- 配置 ----------------
NPZ_FILE = "data/random_2d/test.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ORIG_SIZE = 224          # 原始环境尺寸
MODEL_PATH = "models/cae"

# ---------------- 绘图函数 ----------------
def draw_env(ax, obstacles, env_dims=(ORIG_SIZE, ORIG_SIZE)):
    """绘制原始尺寸环境障碍"""
    ax.set_xlim(0, env_dims[0])
    ax.set_ylim(0, env_dims[1])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    for rect in obstacles:
        if len(rect) == 4:
            x, y, w, h = rect
            ax.add_patch(plt.Rectangle((x, y), w, h, color='black'))
    ax.axis('off')

def draw_paths(ax, path, env_dims=(ORIG_SIZE, ORIG_SIZE)):
    """绘制路径（原始尺寸），paths 为 list of [L,2]"""

    path = np.array(path)
    xs, ys = path[:,0], path[:,1]
    ax.plot(xs, ys, color='red', linewidth=2)

# ---------------- 读取 NPZ ----------------
data = np.load(NPZ_FILE, allow_pickle=True)
grids = data['grids']                # [num_env, H, W]
obstacles_list = data['obstacles']   # 每个环境的障碍信息
paths_all = data['paths']            # 每个样本的路径（list of [L,2]）

num_envs = len(grids)
idx = np.random.randint(0, num_envs)

mask = grids[idx]                     # (H, W)
obstacles = obstacles_list[idx]       # list of rects
mask_paths = paths_all[idx] if len(paths_all)>idx else []

# ---------------- 加载 CAE ----------------
encoder = AE.Encoder_CNN_2D(latent_dim=128)
decoder = AE.Decoder_CNN_2D(latent_dim=128)

encoder.load_state_dict(torch.load(os.path.join(MODEL_PATH, "encoder_best.pth"), map_location=DEVICE))
decoder.load_state_dict(torch.load(os.path.join(MODEL_PATH, "decoder_best.pth"), map_location=DEVICE))
encoder.to(DEVICE).eval()
decoder.to(DEVICE).eval()

# ---------------- 推理 ----------------
mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    latent, enc_feats = encoder(mask_tensor)
    recon = decoder(latent, enc_feats)

recon_np = recon.squeeze().cpu().numpy()

# 将 mask / recon 放大到原始尺寸
mask_up = cv2.resize(mask, (ORIG_SIZE, ORIG_SIZE), interpolation=cv2.INTER_NEAREST)
recon_up = cv2.resize(recon_np, (ORIG_SIZE, ORIG_SIZE), interpolation=cv2.INTER_LINEAR)

# ---------------- 可视化 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15,5))

axes[0].set_title("Original Environment")
draw_env(axes[0], obstacles, env_dims=(ORIG_SIZE, ORIG_SIZE))
draw_paths(axes[0], mask_paths, env_dims=(ORIG_SIZE, ORIG_SIZE))

axes[1].set_title("Mask Map (Upscaled)")
axes[1].imshow(mask_up, cmap="gray", vmin=0, vmax=1, extent=[0, ORIG_SIZE, ORIG_SIZE, 0])
draw_paths(axes[1], mask_paths, env_dims=(ORIG_SIZE, ORIG_SIZE))
axes[1].axis('on')

axes[2].set_title("CAE Reconstructed (Upscaled)")
axes[2].imshow(recon_up, cmap="gray", vmin=0, vmax=1, extent=[0, ORIG_SIZE, ORIG_SIZE, 0])
draw_paths(axes[2], mask_paths, env_dims=(ORIG_SIZE, ORIG_SIZE))
axes[2].axis('on')

plt.tight_layout()
plt.show()
