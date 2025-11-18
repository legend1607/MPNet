import numpy as np
import os
from os.path import join

def load_dataset_mask(N=120, split="train"):
    """
    从 data/random_2d/<split>.npz 中加载 grid 作为 mask.
    返回形状 (N, H, W) 的 numpy 数组。
    """

    dataset_dir = "data/random_2d"
    npz_file = join(dataset_dir, f"{split}.npz")

    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"找不到数据文件: {npz_file}")

    # 读取 npz 文件
    data = np.load(npz_file, allow_pickle=True)

    if "grids" not in data:
        raise KeyError(f"{npz_file} 中未找到 'grids'")

    env_masks_all = data["grids"].astype(np.float32)   # (M, H, W)

    # 只取前 N 个
    N = min(N, len(env_masks_all))
    enviro_mask_set = env_masks_all[:N]

    return enviro_mask_set
