import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from os.path import join

def visualize_env(npz_path, sample_idx=0):
    """
    可视化环境
    npz_path: 数据集路径，如 "data/random_2d/train.npz"
    sample_idx: 样本索引
    """
    data = np.load(npz_path, allow_pickle=True)
    
    grids = data['grids']        # 栅格图 (object)
    obstacles = data['obstacles']  # 原始障碍 (object)
    start = data['start']        # 起点
    goal = data['goal']          # 终点
    paths = data['paths']        # 路径
    masks = data['masks']        # mask

    grid = grids[sample_idx]
    obs = obstacles[sample_idx]
    s = start[sample_idx]
    g = goal[sample_idx]
    path = paths[sample_idx]
    mask = masks[sample_idx]

    # 原始路径长度
    path_len = int(mask.sum())
    path = path[:path_len]

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 栅格背景
    ax.imshow(grid, cmap='Greys', origin='lower')

    # 障碍
    for rect in obs:
        rx, ry, rw, rh = rect
        patch = patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='r', facecolor='r', alpha=0.5)
        ax.add_patch(patch)

    # 路径
    ax.plot(path[:,0], path[:,1], 'b-o', label='Path', markersize=4)

    # 起点和终点
    ax.plot(s[0], s[1], 'go', markersize=8, label='Start')
    ax.plot(g[0], g[1], 'ro', markersize=8, label='Goal')

    ax.set_title(f"Sample {sample_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    npz_path = join("data", "random_2d", "train.npz")
    visualize_env(npz_path, sample_idx=np.random.randint(0, 500))
