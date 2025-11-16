"""
优化版 convert_json_to_npz_with_grids_safe.py
-----------------------------------------
"""

import os
import json
import time
import numpy as np
from os.path import join
from tqdm import tqdm


def env_to_grid(env_dict, resolution=1.0):
    """将 JSON 环境转成二值栅格"""
    width, height = env_dict["env_dims"]
    w_cells, h_cells = int(width / resolution), int(height / resolution)
    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)

    # 矩形障碍
    for rx, ry, rw, rh in env_dict.get("rectangle_obstacles", []):
        x1, y1 = int(rx / resolution), int(ry / resolution)
        x2, y2 = int((rx + rw) / resolution), int((ry + rh) / resolution)
        grid[y1:y2, x1:x2] = 1

    return grid.astype(np.float32)


def convert_json_to_npz_with_grids(env_type="random_2d", resolution=1.0):
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

        # ------- 环境级别数据 -------
        env_grids = []
        env_start = []
        env_goal = []
        env_paths = []
        sample_envid = []

        for env_dict in tqdm(env_list):
            env_idx = env_dict["env_idx"]
            grid = env_to_grid(env_dict, resolution)
            env_grids.append(grid)

            # 每个环境有多组样本
            S = env_dict["start"]
            G = env_dict["goal"]
            P = env_dict["paths"]

            for s, g, path in zip(S, G, P):
                env_start.append(np.array(s, dtype=np.float32))
                env_goal.append(np.array(g, dtype=np.float32))
                env_paths.append(np.array(path, dtype=np.float32))  # ⚠️ 保持 list
                sample_envid.append(env_idx)

        # ---------------- 保存 ----------------
        np.savez_compressed(
            join(dataset_dir, f"{mode}.npz"),
            env_grids=np.array(env_grids),          # [num_env, H, W]
            sample_envid=np.array(sample_envid),    # [num_samples]
            start=np.array(env_start),
            goal=np.array(env_goal),
    path=np.array(env_paths, dtype=object)         # list，长度可不一致
        )



if __name__ == "__main__":
    convert_json_to_npz_with_grids(env_type="random_2d", resolution=1.0)
