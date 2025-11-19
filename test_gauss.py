# mpnet_infer.py
import math
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import cv2  # 用于 bresenham style line rasterization on grid

# ---------------------------
# 辅助函数：栅格/几何
# ---------------------------
def point_to_cell(pt, resolution=1.0):
    # pt: (x, y) in same coordinates used by env->grid conversion
    # assume grid indexing: grid[y_cell, x_cell]
    x, y = pt
    return int(round(x / resolution)), int(round(y / resolution))

def line_collision(grid, p0, p1, resolution=1.0):
    """
    在栅格上检查从 p0 到 p1 的线段是否碰撞（任何中间像素为 1 表示障碍）
    p0, p1: (x, y) in world coords where grid cells were computed as int(x/res)
    grid: numpy (H, W) with 1 = obstacle, 0 = free
    """
    x0, y0 = point_to_cell(p0, resolution)
    x1, y1 = point_to_cell(p1, resolution)

    H, W = grid.shape
    # 使用 cv2.line 在 mask 上画线并检查是否与障碍相交
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.line(mask, (x0, y0), (x1, y1), color=1, thickness=1)
    # 如果任何画线像素在 grid 上为障碍 -> collision
    return np.any(np.logical_and(mask, grid.astype(np.uint8)))

def point_collision(grid, p, resolution=1.0):
    x, y = point_to_cell(p, resolution)
    H, W = grid.shape
    if x < 0 or x >= W or y < 0 or y >= H:
        return True
    return bool(grid[y, x] != 0)

def dist(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.linalg.norm(a - b))

# ---------------------------
# 采样与选择
# ---------------------------
def sample_from_gaussians(mu, sigma, method='gaussian'):
    """
    mu, sigma: torch tensors shape (K, D) or (1, K, D)
    返回 numpy (K, D)
    """
    if isinstance(mu, torch.Tensor):
        mu = mu.detach().cpu().numpy()
        sigma = sigma.detach().cpu().numpy()
    if mu.ndim == 3:
        mu = mu[0]
        sigma = sigma[0]
    if method == 'gaussian':
        eps = np.random.randn(*mu.shape)
        return mu + sigma * eps
    elif method == 'mu':
        return mu
    else:
        raise ValueError("method must be 'gaussian' or 'mu'")

# ---------------------------
# MPNetPlanner 类
# ---------------------------
class MPNetPlanner:
    def __init__(self, encoder, model, device='cuda', resolution=1.0,
                 num_candidates=5, sample_method='gaussian', max_it=500,
                 goal_threshold=1.0):
        """
        encoder: Encoder_CNN_2D instance (should be on device)
        model: MLP_fusion_gaussian instance (on device)
        device: 'cuda' or 'cpu'
        resolution: grid resolution used to convert world coords -> cells
        num_candidates: K
        sample_method: 'gaussian' or 'mu'
        max_it: max planning iterations
        goal_threshold: distance to goal threshold (world units)
        """
        self.encoder = encoder.to(device)
        self.model = model.to(device)
        self.device = device
        self.resolution = resolution
        self.num_candidates = num_candidates
        self.sample_method = sample_method
        self.max_it = max_it
        self.goal_threshold = goal_threshold

        self.encoder.eval()
        self.model.eval()

    def encode_env(self, grid):
        """
        grid: numpy (H, W) float32 (0 free, 1 obstacle)
        returns latent: torch tensor on device shape (1, latent_dim)
        """
        with torch.no_grad():
            g = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            if hasattr(self.encoder, "encode"):
                latent = self.encoder.encode(g)
            else:
                latent = self.encoder(g)[0] if isinstance(self.encoder(g), tuple) else self.encoder(g)
            # ensure shape [1, latent_dim]
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
            return latent

    def plan(self, grid, start, goal, ax=None, visualize=False):
        """
        grid: numpy (H, W) with 0/1
        start, goal: (x, y) world coords (same scale as env_to_grid)
        returns: path: list of (x,y) from start to goal (or None)
        """
        latent = self.encode_env(grid)  # (1, latent_dim)
        path = [tuple(start)]
        curr = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)

        for it in range(self.max_it):
            # if close to goal -> try to connect directly
            if dist(curr, goal) <= self.goal_threshold:
                if not line_collision(grid, curr, goal, self.resolution):
                    path.append(tuple(goal))
                    return path
                # else continue and let model propose bridge

            # prepare inputs
            curr_t = torch.tensor(curr, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, D)
            goal_t = torch.tensor(goal, dtype=torch.float32, device=self.device).unsqueeze(0)
            joint_input = torch.cat([curr_t, goal_t], dim=-1)  # (1, 2*D)

            # model predict mu,sigma for K candidates
            with torch.no_grad():
                mu, sigma = self.model(latent, joint_input)  # mu,sigma: (1,K,D)

            samples = sample_from_gaussians(mu, sigma, method=self.sample_method)  # (K, D)

            # Evaluate candidates: filter collision-free and sort by distance-to-goal (ascending)
            feasible = []
            for k, s in enumerate(samples):
                p = np.array(s, dtype=float)
                # optional: clamp p within grid/world bounds? skip for now
                if not point_collision(grid, p, self.resolution) and not line_collision(grid, curr, p, self.resolution):
                    feasible.append((k, p, dist(p, goal)))

            if len(feasible) == 0:
                # Fallback: sample along direction to goal (small step)
                fallback = self._fallback_sample(curr, goal, grid)
                if fallback is None:
                    # if fallback fails, try random perturbation around curr
                    fallback = self._random_perturb(curr, grid)
                if fallback is None:
                    # totally stuck
                    return None
                next_pt = fallback
            else:
                # choose candidate nearest to goal
                feasible.sort(key=lambda x: x[2])
                next_pt = feasible[0][1]

            path.append(tuple(next_pt))
            if ax is not None:
                ax.plot([path[-2][0], path[-1][0]], [path[-2][1], path[-1][1]], 'r-')
                ax.plot(next_pt[0], next_pt[1], 'ro')
                plt.pause(0.02)
            curr = np.array(next_pt, dtype=float)

        # exceeded max iterations
        return None

    def _fallback_sample(self, curr, goal, grid):
        """
        插值向目标方向，尝试连接（small step）
        """
        dir_vec = np.array(goal) - np.array(curr)
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-6:
            return None
        step = min(self.goal_threshold, norm * 0.5)
        candidate = np.array(curr) + (dir_vec / norm) * step
        if (not point_collision(grid, candidate, self.resolution)) and (not line_collision(grid, curr, candidate, self.resolution)):
            return candidate
        return None

    def _random_perturb(self, curr, grid, radius=2.0, tries=16):
        for _ in range(tries):
            ang = random.random() * 2 * math.pi
            r = random.random() * radius
            cand = np.array(curr) + np.array([math.cos(ang), math.sin(ang)]) * r
            if not point_collision(grid, cand, self.resolution) and not line_collision(grid, curr, cand, self.resolution):
                return cand
        return None

    def smooth_path(self, path):
        """
        简单的路径平滑：尝试直接连通远端点，删除中间冗余点
        path: list of (x,y)
        """
        if path is None or len(path) < 3:
            return path
        grid = None  # user should provide grid if needed; here we assume not colliding for smoothing
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if not line_collision(self._grid_for_smooth, path[i], path[j], self.resolution):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

def draw_env(ax, obstacles, env_dims=(224,224)):
    ax.set_xlim(0, env_dims[0])
    ax.set_ylim(0, env_dims[1])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    for rect in obstacles:
        if len(rect) == 4:
            x, y, w, h = rect
            ax.add_patch(plt.Rectangle((x, y), w, h, color='black'))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
def draw_path(ax, path):
    if path is not None and len(path) > 1:
        ax.plot(path[:,0], path[:,1], 'r-o', label="Planned Path")
# ---------------------------
# 使用示例（单次规划）
# ---------------------------
if __name__ == "__main__":
    # 假设你有：
    # - grid: numpy (H, W) float32 (0 free,1 obstacle)
    # - encoder_ckpt, mlp_ckpt 保存的模型权重路径
    from MPNet.AE.cae_2d import Encoder_CNN_2D
    from MPNet.model import MLP_fusion_gaussian

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load encoder
    encoder = Encoder_CNN_2D(latent_dim=128)
    encoder.load_state_dict(torch.load("models/cae/encoder_best.pth", map_location=device))
    encoder.to(device).eval()

    # load mlp
    model = MLP_fusion_gaussian(
        cae_input_size=128,
        joint_input_size=4,   # curr(2)+goal(2)
        joint_dim=2,
        num_candidates=5
    )
    model.load_state_dict(torch.load("results/pointgen_single_step/mlp_best.pth", map_location=device))
    model.to(device).eval()

    # load environment grid (example)
    # grid should be float32 with 1.0 obstacles, 0.0 free, shape (H, W)

    test_npz = "data/random_2d/train.npz"
    data = np.load(test_npz, allow_pickle=True)
    grids = data['grids']
    starts = data['start']
    goals = data['goal']
    obstacles_all = data['obstacles']
    index = np.random.randint(len(grids))
    planner = MPNetPlanner(encoder, model, device=device, resolution=1.0,
                           num_candidates=8, sample_method='gaussian',
                           max_it=500, goal_threshold=1.0)
    fig, ax = plt.subplots(figsize=(6,6))
    draw_env(ax, obstacles_all[index], env_dims=(224, 224))
    ax.scatter(starts[index][0], starts[index][1], c='g', s=100, marker='o', label='Start')
    ax.scatter(goals[index][0], goals[index][1], c='b', s=100, marker='*', label='Goal')
    plt.legend()

    path = planner.plan(grids[index], starts[index], goals[index], ax=ax)
    
    draw_path(ax, path)
    plt.ioff()
    plt.show()
