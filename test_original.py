import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from MPNet.AE.cae_2d import Encoder_CNN_2D
from MPNet.model import MLP_original
from environment.random_2d_env import Random2DEnv

# ------------------- 基础函数 -------------------
def steer_to(env, start, end):
    return env._edge_fp(start, end, step_size=0.5)

def is_in_collision(env, point):
    return not env._point_in_free_space(point)

def is_reaching_target(p1, p2, thresh=1.0):
    return np.linalg.norm(p1 - p2) < thresh

def path_feasible(env, path):
    for i in range(len(path)-1):
        if not steer_to(env, path[i], path[i+1]):
            return False
    return True

def lvc(env, path):
    """懒惰顶点压缩"""
    n = len(path)
    i = 0
    while i < n-1:
        j = n-1
        while j > i+1:
            if steer_to(env, path[i], path[j]):
                path = path[:i+1] + path[j:]
                n = len(path)
                break
            j -= 1
        i += 1
    return path

# ------------------- 局部重规划 -------------------
def replan_path(path, latent, mlp, device, env, ax=None, max_local_steps=50, reach_thresh=1.0):
    new_path = [path[0]]
    for i in range(len(path)-1):
        start = new_path[-1]
        goal = path[i+1]
        if steer_to(env, start, goal):
            new_path.append(goal)
            continue

        temp_start, temp_goal = start.copy(), goal.copy()
        segment = [temp_start]
        tree, step, reached = 0, 0, False

        while not reached and step < max_local_steps:
            step += 1
            if tree == 0:
                mlp_input = torch.cat([
                    latent,
                    torch.from_numpy(temp_start).float().to(device),
                    torch.from_numpy(temp_goal).float().to(device)
                ], dim=0).unsqueeze(0)
                with torch.no_grad():
                    temp_start = mlp(mlp_input).cpu().numpy().squeeze()
                if not is_in_collision(env, temp_start):
                    segment.append(temp_start)
                    if ax is not None:
                        ax.plot([segment[-2][0], temp_start[0]], [segment[-2][1], temp_start[1]], 'r-')
                        ax.plot(temp_start[0], temp_start[1], 'ro')
                        plt.pause(0.02)
                tree = 1
            else:
                mlp_input = torch.cat([
                    latent,
                    torch.from_numpy(temp_goal).float().to(device),
                    torch.from_numpy(temp_start).float().to(device)
                ], dim=0).unsqueeze(0)
                with torch.no_grad():
                    temp_goal = mlp(mlp_input).cpu().numpy().squeeze()
                if not is_in_collision(env, temp_goal):
                    segment.append(temp_goal)
                    if ax is not None:
                        ax.plot([segment[-2][0], temp_goal[0]], [segment[-2][1], temp_goal[1]], 'r-')
                        ax.plot(temp_goal[0], temp_goal[1], 'ro')
                        plt.pause(0.02)
                tree = 0

            if is_reaching_target(temp_start, temp_goal, reach_thresh):
                reached = True

        if not reached:
            print("Warning: local replanning failed")
            return None

        new_path.extend(segment[1:])
    return new_path

# ------------------- 单向扩展路径规划 -------------------
def single_direction_planning(encoder, mlp, grid, start, goal, device, env, ax=None,
                              max_steps=80, reach_thresh=1.0):
    encoder.eval()
    mlp.eval()
    grid_tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        latent_tensor, _ = encoder(grid_tensor)
    latent = latent_tensor.squeeze(0)

    path = [start.copy()]
    step, reached = 0, False

    while not reached and step < max_steps:
        step += 1
        mlp_input = torch.cat([
            latent,
            torch.from_numpy(path[-1]).float().to(device),
            torch.from_numpy(goal).float().to(device)
        ], dim=0).unsqueeze(0)

        with torch.no_grad():
            next_point = mlp(mlp_input).cpu().numpy().squeeze()

        if not is_in_collision(env, next_point) and steer_to(env, path[-1], next_point):
            path.append(next_point)
            if ax is not None:
                ax.plot([path[-2][0], path[-1][0]], [path[-2][1], path[-1][1]], 'r-')
                ax.plot(next_point[0], next_point[1], 'ro')
                plt.pause(0.02)

        if is_reaching_target(path[-1], goal, reach_thresh):
            reached = True

    path = lvc(env, path)
    if path_feasible(env, path):
        return np.array(path)
    else:
        replanned = replan_path(path, latent, mlp, device, env, ax=ax)
        if replanned is not None:
            replanned = lvc(env, replanned)
            if path_feasible(env, replanned):
                return np.array(replanned)
        print("Warning: final path still contains collisions")
        return None

# ------------------- 双向路径规划 -------------------
def bidirectional_planning_full(encoder, mlp, grid, start, goal, device, env, ax=None,
                                max_steps=80, reach_thresh=1.0):
    encoder.eval()
    mlp.eval()
    grid_tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        latent_tensor, _ = encoder(grid_tensor)
    latent = latent_tensor.squeeze(0)

    path_start = [start.copy()]
    path_goal = [goal.copy()]
    tree, step, reached = 0, 0, False

    while not reached and step < max_steps:
        step += 1
        if tree == 0:
            mlp_input = torch.cat([
                latent,
                torch.from_numpy(path_start[-1]).float().to(device),
                torch.from_numpy(path_goal[-1]).float().to(device)
            ], dim=0).unsqueeze(0)
            with torch.no_grad():
                next_point = mlp(mlp_input).cpu().numpy().squeeze()
            if not is_in_collision(env, next_point) and steer_to(env, path_start[-1], next_point):
                path_start.append(next_point)
                if ax is not None:
                    ax.plot([path_start[-2][0], next_point[0]], [path_start[-2][1], next_point[1]], 'r-')
                    ax.plot(next_point[0], next_point[1], 'ro')
                    plt.pause(0.02)
            tree = 1
        else:
            mlp_input = torch.cat([
                latent,
                torch.from_numpy(path_goal[-1]).float().to(device),
                torch.from_numpy(path_start[-1]).float().to(device)
            ], dim=0).unsqueeze(0)
            with torch.no_grad():
                next_point = mlp(mlp_input).cpu().numpy().squeeze()
            if not is_in_collision(env, next_point) and steer_to(env, path_goal[-1], next_point):
                path_goal.append(next_point)
                if ax is not None:
                    ax.plot([path_goal[-2][0], next_point[0]], [path_goal[-2][1], next_point[1]], 'r-')
                    ax.plot(next_point[0], next_point[1], 'ro')
                    plt.pause(0.02)
            tree = 0

        if is_reaching_target(path_start[-1], path_goal[-1], reach_thresh):
            reached = True

    full_path = path_start + path_goal[::-1]
    full_path = lvc(env, full_path)
    if path_feasible(env, full_path):
        return np.array(full_path)
    else:
        replanned = replan_path(full_path, latent, mlp, device, env, ax=ax)
        if replanned is not None:
            replanned = lvc(env, replanned)
            if path_feasible(env, replanned):
                return np.array(replanned)
        print("Warning: final path still contains collisions")
        return None

# ------------------- 绘图函数 -------------------
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

# ------------------- 主函数 -------------------
def main():
    plt.ion()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_npz = "data/random_2d/train.npz"
    data = np.load(test_npz, allow_pickle=True)
    grids = data['grids']
    sample_envid = data['sample_envid']
    starts = data['start']
    goals = data['goal']
    obstacles_all = data['obstacles']

    mask_size = grids.shape[1]
    encoder = Encoder_CNN_2D(mask_size=mask_size).to(device)
    encoder.load_state_dict(torch.load("models/cae/encoder_best.pth", map_location=device))
    encoder.eval()

    with torch.no_grad():
        dummy = torch.from_numpy(grids[0:1]).unsqueeze(1).float().to(device)
        latent_dummy, _ = encoder(dummy)
        latent_dim = latent_dummy.shape[1]

    mlp_input_dim = latent_dim + 4
    mlp = MLP_original(input_size=mlp_input_dim, output_size=2).to(device)
    mlp.load_state_dict(torch.load("results/pointgen_single_step/mlp_best.pth", map_location=device))
    mlp.eval()

    ORIG_SIZE = 224
    for idx in range(len(sample_envid)):
        env_idx = int(sample_envid[idx])
        grid = grids[env_idx]
        obstacles_list = obstacles_all[env_idx]
        env_dict = {
            "env_dims": [224, 224],
            "start": [starts[idx]],
            "goal": [goals[idx]],
            "rectangle_obstacles": obstacles_list,
            "circle_obstacles": []
        }
        env = Random2DEnv(env_dict)
        start, goal = starts[idx], goals[idx]

        fig, ax = plt.subplots(figsize=(6,6))
        draw_env(ax, obstacles_list, env_dims=(ORIG_SIZE, ORIG_SIZE))
        ax.scatter(start[0], start[1], c='g', s=100, marker='o', label='Start')
        ax.scatter(goal[0], goal[1], c='b', s=100, marker='*', label='Goal')
        plt.legend()

        # 选择单向还是双向规划
        planned_path = single_direction_planning(encoder, mlp, grid, start, goal, device, env, ax=ax)
        # planned_path = bidirectional_planning_full(encoder, mlp, grid, start, goal, device, env, ax=ax)

        draw_path(ax, planned_path)
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()
