import os
import json
import time
import numpy as np
from os.path import join
from tqdm import tqdm
import cv2

def env_to_grid(env_dict, resolution=1.0):
    """将环境转成二值栅格"""
    width, height = env_dict["env_dims"]
    w_cells, h_cells = int(width / resolution), int(height / resolution)

    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)

    # 障碍填充
    for rx, ry, rw, rh in env_dict.get("rectangle_obstacles", []):
        x1, y1 = int(rx / resolution), int(ry / resolution)
        x2, y2 = int((rx + rw) / resolution), int((ry + rh) / resolution)
        grid[y1:y2, x1:x2] = 1

    return grid.astype(np.float32)


def convert_json_to_npz_no_padding(env_type="random_2d", resolution=1.0):
    dataset_dir = join("data", env_type)
    os.makedirs(dataset_dir, exist_ok=True)

    for mode in ["train", "val", "test"]:
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
        env_obstacles = []

        for env_dict in tqdm(env_list):
            grid = env_to_grid(env_dict, resolution)
            obstacles = np.array(env_dict.get("rectangle_obstacles", []), dtype=np.float32)

            for s, g, path in zip(env_dict["start"], env_dict["goal"], env_dict["paths"]):

                env_grids.append(grid.copy())
                env_obstacles.append(obstacles)
                env_start.append(np.array(s, dtype=np.float32))
                env_goal.append(np.array(g, dtype=np.float32))
                env_paths.append(np.array(path, dtype=np.float32))  

        # 保存 npz（paths, obstacles 使用 object，以支持变长）
        np.savez_compressed(
            join(dataset_dir, f"{mode}.npz"),
            grids=np.array(env_grids),
            obstacles=np.array(env_obstacles, dtype=object),
            start=np.array(env_start),
            goal=np.array(env_goal),
            paths=np.array(env_paths, dtype=object)
        )

        print(f"✅ 完成 {mode}, 用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    convert_json_to_npz_no_padding(env_type="random_2d", resolution=1.0)
