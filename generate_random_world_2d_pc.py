
import os
import json
import time
import numpy as np
from os.path import join
from tqdm import tqdm
import cv2


TARGET_GRID_SIZE = 160    # ← 固定 CNN 输入尺寸 160×160


def env_to_grid(env_dict, resolution=1.0, out_size=TARGET_GRID_SIZE):
    """将环境转成固定大小 160×160 二值栅格（兼容旧模型 CNN）"""
    width, height = env_dict["env_dims"]
    w_cells, h_cells = int(width / resolution), int(height / resolution)

    # 原始分辨率栅格
    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)

    # 障碍填充
    for rx, ry, rw, rh in env_dict.get("rectangle_obstacles", []):
        x1, y1 = int(rx / resolution), int(ry / resolution)
        x2, y2 = int((rx + rw) / resolution), int((ry + rh) / resolution)
        grid[y1:y2, x1:x2] = 1

    # ⭐ 最关键：固定 resize → 160×160 ⭐
    grid_resized = cv2.resize(
        grid, 
        (out_size, out_size), 
        interpolation=cv2.INTER_NEAREST
    )

    return grid_resized.astype(np.float32)


def convert_json_to_npz_with_grids_safe(env_type="random_2d", resolution=1.0):
    dataset_dir = join("data", env_type)
    os.makedirs(dataset_dir, exist_ok=True)

    for mode in ["train", "val"]:
        env_json_path = join(dataset_dir, mode, "envs.json")
        if not os.path.exists(env_json_path):
            print(f"⚠️ 跳过 {mode} (未找到 {env_json_path})")
            continue

        with open(env_json_path, "r") as f:
            env_list = json.load(f)

        print(f"📦 开始转换 {mode} 数据集，共 {len(env_list)} 个环境...")
        t0 = time.time()

        env_grids = []
        env_start = []
        env_goal = []
        env_paths = []
        env_masks = []
        sample_envid = []

        # 找最长路径长度
        max_len = 0
        for env_dict in env_list:
            for path in env_dict["paths"]:
                max_len = max(max_len, len(path))

        for env_dict in tqdm(env_list):
            env_idx = env_dict["env_idx"]
            grid = env_to_grid(env_dict, resolution)
            env_grids.append(grid)

            S = env_dict["start"]
            G = env_dict["goal"]
            P = env_dict["paths"]

            # 每条路径一条样本
            for s, g, path in zip(S, G, P):
                path_np = np.array(path, dtype=np.float32)
                L = len(path_np)

                # mask
                mask = np.zeros(max_len, dtype=np.float32)
                mask[:L] = 1.0

                # padding 用最后一个有效点
                path_padded = np.zeros((max_len, 2), dtype=np.float32)
                path_padded[:L] = path_np
                if L < max_len:
                    path_padded[L:] = path_np[-1]

                env_start.append(np.array(s, dtype=np.float32))
                env_goal.append(np.array(g, dtype=np.float32))
                env_paths.append(path_padded)
                env_masks.append(mask)
                sample_envid.append(env_idx)

        # 保存 npz
        np.savez_compressed(
            join(dataset_dir, f"{mode}.npz"),
            grids=np.array(env_grids),          # [num_env, H, W]
            sample_envid=np.array(sample_envid),    # [num_samples]
            start=np.array(env_start),
            goal=np.array(env_goal),
            paths=np.array(env_paths),              # [num_samples, max_len, 2]
            masks=np.array(env_masks)               # [num_samples, max_len]
        )

        print(f"✅ 完成 {mode}, 用时 {time.time()-t0:.1f}s, 最大路径长度 {max_len}")


if __name__ == "__main__":
    convert_json_to_npz_with_grids_safe(env_type="random_2d", resolution=1.0)