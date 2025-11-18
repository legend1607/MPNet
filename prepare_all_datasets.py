import os
import json
import numpy as np
from tqdm import tqdm
from skimage.draw import polygon
import torch
import torch.nn.functional as F

# ---------------- 参数配置 ----------------
BASE_DIR = "data/random_2d"  # 数据集总目录
SAVE_DIR = os.path.join(BASE_DIR, "processed_all")
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = (160, 160)  # CNN输入掩码大小
NUM_SECTORS = 8         # 方向分类数

DATASETS = ["train", "val", "test"]

# ---------------- 辅助函数 ----------------
def create_env_mask(env_dict, img_size=(160,160)):
    H, W = img_size
    mask = np.zeros((H, W), dtype=np.float32)
    for rect in env_dict.get("rectangle_obstacles", []):
        x, y, w, h = rect
        xs = np.array([x, x + w, x + w, x])
        ys = np.array([y, y, y + h, y + h])
        xs = np.round(xs / env_dict["env_dims"][0] * (W-1)).astype(np.int32)
        ys = np.round(ys / env_dict["env_dims"][1] * (H-1)).astype(np.int32)
        xs = np.clip(xs, 0, W-1)
        ys = np.clip(ys, 0, H-1)
        rr, cc = polygon(ys, xs, shape=mask.shape)
        mask[rr, cc] = 1.0
    return mask

def sample_inner_surface_in_pixel(image):
    a = F.max_pool2d(-image[None,None].float(), kernel_size=(3,1), stride=1, padding=(1,0))[0]
    b = F.max_pool2d(-image[None,None].float(), kernel_size=(1,3), stride=1, padding=(0,1))[0]
    border, _ = torch.max(torch.cat([a,b], dim=0), dim=0)
    surface = border + image.float()
    return surface.long()

def SamplesFunc(img, value=1, type='array'):
    samples_points = np.where(img == value)
    samples = list(zip(list(samples_points[0]), list(samples_points[1])))
    if type=='list':
        return samples
    elif type=='array':
        return np.array(samples)

def surface_to_real(surface_points, bias, scale_para):
    real_points = surface_points / scale_para
    real_points = np.subtract(real_points, bias)
    return real_points

def direction_to_class(current, next_point, num_sectors=NUM_SECTORS):
    vec = np.array(next_point) - np.array(current)
    angle = np.arctan2(vec[1], vec[0])
    sector = int(((angle + np.pi) / (2 * np.pi)) * num_sectors)
    return sector % num_sectors
def interpolate_path(path, distance_per_point=0.1):
    """
    对一条路径进行线性插值，使得相邻点之间的距离大约为 distance_per_point
    """
    interpolated = []
    for i in range(len(path)-1):
        p1 = path[i]
        p2 = path[i+1]
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        n_points = max(int(dist / distance_per_point), 1)
        segment = np.linspace(p1, p2, n_points + 1, endpoint=False)
        interpolated.extend(segment)
    interpolated.append(path[-1])  # 保证最后一个点在末尾
    return np.array(interpolated, dtype=np.float32)

# ---------------- 主处理函数 ----------------
for dataset in DATASETS:
    DATA_DIR = os.path.join(BASE_DIR, dataset)
    JSON_FILE = os.path.join(DATA_DIR, "envs.json")
    PROCESSED_DIR = os.path.join(SAVE_DIR, dataset)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"\n🔹 处理数据集 [{dataset}]...")

    env_masks_list = []
    boundary_points_list = []
    boundary_lengths_list = []

    dataset_list = []
    targets_list = []
    orient_dataset = []
    classification_orient_targets = []
    classification_norm_targets = []
    padded_targets_future_all = []
    samples_length = []
    env_indices = []

    max_future_length = 0
    max_boundary_length = 0
    bias = 0.0
    scale_para = 1.0

    env_list = json.load(open(JSON_FILE, "r"))

    # ---------------- 逐环境处理 ----------------
    for env_idx, env in enumerate(tqdm(env_list)):
        # mask
        env_mask = create_env_mask(env, img_size=IMG_SIZE)
        env_masks_list.append(env_mask)

        # 内表面采样 + 坐标转换
        surface = sample_inner_surface_in_pixel(torch.from_numpy(env_mask))
        surface = surface.cpu().numpy()
        surface_points = SamplesFunc(surface)
        boundary_points = surface_to_real(surface_points, bias, scale_para)
        boundary_points_list.append(boundary_points)
        boundary_lengths_list.append(len(boundary_points))
        max_boundary_length = max(max_boundary_length, len(boundary_points))

        # 路径处理
        paths = env.get("paths", [])
        for path in paths:
            path = np.array(path, dtype=np.float32)
            # 插值
            path_interp = interpolate_path(path, distance_per_point=0.1)
            T = len(path_interp)
            samples_length.append(T)
            padded_targets_future_all.append(path_interp)
            max_future_length = max(max_future_length, T)

            for i in range(T - 1):
                current = path_interp[i]
                next_point = path_interp[i + 1]

                dataset_list.append(current)
                targets_list.append(next_point)
                orient_dataset.append(np.concatenate([current, next_point]))
                classification_orient_targets.append(direction_to_class(current, next_point))
                classification_norm_targets.append(np.linalg.norm(next_point - current))
                env_indices.append(env_idx)


    # ---------------- pad boundary_points_set ----------------
    N = len(boundary_points_list)
    boundary_points_set = np.full((N, max_boundary_length, 2), 100.0, dtype=np.float32)
    for i, pts in enumerate(boundary_points_list):
        boundary_points_set[i, :len(pts), :] = pts

    # ---------------- pad env_mask_set ----------------
    enviro_mask_set = np.zeros((N, IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)
    for i, mask in enumerate(env_masks_list):
        enviro_mask_set[i, :, :] = mask

    # ---------------- pad future paths ----------------
    padded_future = np.zeros((len(padded_targets_future_all), max_future_length, 2), dtype=np.float32)
    for i, path in enumerate(padded_targets_future_all):
        padded_future[i, :path.shape[0], :] = path
    padded_targets_future_all = padded_future

    # ---------------- 保存 npy ----------------
    np.save(os.path.join(PROCESSED_DIR, "env_masks.npy"), enviro_mask_set)
    np.save(os.path.join(PROCESSED_DIR, "boundary_points_set.npy"), boundary_points_set)
    np.save(os.path.join(PROCESSED_DIR, "boundary_lengths.npy"), np.array(boundary_lengths_list, dtype=np.int32))

    np.save(os.path.join(PROCESSED_DIR, "dataset.npy"), np.array(dataset_list, dtype=np.float32))
    np.save(os.path.join(PROCESSED_DIR, "targets.npy"), np.array(targets_list, dtype=np.float32))
    np.save(os.path.join(PROCESSED_DIR, "orient_dataset.npy"), np.array(orient_dataset, dtype=np.float32))
    np.save(os.path.join(PROCESSED_DIR, "classification_orient_targets.npy"), np.array(classification_orient_targets, dtype=np.int64))
    np.save(os.path.join(PROCESSED_DIR, "classification_norm_targets.npy"), np.array(classification_norm_targets, dtype=np.float32))
    np.save(os.path.join(PROCESSED_DIR, "padded_targets_future_all.npy"), padded_targets_future_all)
    np.save(os.path.join(PROCESSED_DIR, "samples_length.npy"), np.array(samples_length, dtype=np.int32))
    np.save(os.path.join(PROCESSED_DIR, "env_indices.npy"), np.array(env_indices, dtype=np.int32))

    print(f"✅ [{dataset}] 数据集处理完成，保存至 {PROCESSED_DIR}")
