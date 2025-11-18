import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import load_test_dataset
from model import MLP
import math
import time

size = 5.0  # 碰撞尺寸

# -------------------- 加载模型 --------------------
mlp = MLP(32, 2)  # simple @D
mlp.load_state_dict(torch.load('models/mlp_best.pkl'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mlp.to(device)
mlp.eval()

# -------------------- 加载数据 --------------------
obc, obstacles, paths, path_lengths = load_test_dataset()

# -------------------- 碰撞检测 --------------------
def IsInCollision(x, idx):
    s = np.array([x[0], x[1]], dtype=np.float32)
    for i in range(7):
        cf = True
        for j in range(2):
            if abs(obc[idx][i][j] - s[j]) > size/2.0:
                cf = False
                break
        if cf:
            return True
    return False

# -------------------- 距离判定 --------------------
def is_reaching_target(start1, start2, thresh=1.0):
    return np.linalg.norm(np.array(start1) - np.array(start2)) <= thresh

# -------------------- 路径可行性 --------------------
def steerTo(start, end, idx):
    DISCRETIZATION_STEP = 0.01
    dists = np.array(end) - np.array(start)
    distTotal = np.linalg.norm(dists)
    if distTotal == 0:
        return 1
    incrementTotal = distTotal / DISCRETIZATION_STEP
    dists /= incrementTotal
    stateCurr = np.array(start, dtype=np.float32)
    for _ in range(int(math.floor(incrementTotal))):
        if IsInCollision(stateCurr, idx):
            return 0
        stateCurr += dists
    if IsInCollision(end, idx):
        return 0
    return 1

def feasibility_check(path, idx):
    for i in range(len(path)-1):
        if steerTo(path[i], path[i+1], idx) == 0:
            return 0
    return 1

# -------------------- Lazy Vertex Contraction --------------------
def lvc(path, idx):
    for i in range(len(path)-1):
        for j in range(len(path)-1, i, -1):
            if steerTo(path[i], path[j], idx):
                pc = path[:i+1] + path[j:]
                return lvc(pc, idx)
    return path

# -------------------- MLP delta forward --------------------
def mlp_delta_step(model, obs, curr, goal):
    """MLP delta forward step"""
    with torch.no_grad():
        inp = torch.cat([torch.tensor(obs, dtype=torch.float32).flatten(),
                         torch.tensor(curr, dtype=torch.float32),
                         torch.tensor(goal, dtype=torch.float32)]).unsqueeze(0).unsqueeze(0).to(device)
        delta = model(inp)[0].squeeze(0).squeeze(0)
        next_pos = curr + delta.cpu().numpy()
    return next_pos

# -------------------- 双向生成路径 --------------------
def bidirectional_mlp_path(model, obs, start, goal, max_steps=80, connect_thresh=1.0):
    start1 = np.array(start, dtype=np.float32)
    start2 = np.array(goal, dtype=np.float32)
    path1 = [start1]
    path2 = [start2]

    for step in range(max_steps):
        # 交替推进
        if step % 2 == 0:
            next1 = mlp_delta_step(model, obs, start1, start2)
            if not IsInCollision(next1, 0):
                path1.append(next1)
                start1 = next1
        else:
            next2 = mlp_delta_step(model, obs, start2, start1)
            if not IsInCollision(next2, 0):
                path2.append(next2)
                start2 = next2

        # 检查是否连接
        if is_reaching_target(start1, start2, connect_thresh):
            break

    full_path = path1 + path2[::-1]
    return full_path

# -------------------- Main --------------------
def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    tp = 0  # 总路径
    fp = 0  # 可行路径
    tot = []

    for i in range(len(obstacles)):
        et = []
        for j in range(len(paths[i])):
            if path_lengths[i][j] == 0:
                continue

            start = paths[i][j][0]
            goal = paths[i][j][path_lengths[i][j]-1]
            obs_env = obstacles[i]

            tic = time.time()
            path = bidirectional_mlp_path(mlp, obs_env, start, goal, max_steps=80, connect_thresh=1.0)
            path = lvc(path, i)
            indicator = feasibility_check(path, i)
            toc = time.time()

            tp += 1
            if indicator:
                fp += 1
                et.append(toc - tic)
                print(f"Path feasible: {i}-{j}, steps={len(path)}, time={toc-tic:.3f}s")
            else:
                print(f"Path infeasible: {i}-{j}")

        tot.append(et)

    pickle.dump(tot, open("time_s2D_unseen_mlp_delta.p", "wb"))
    print("Total paths:", tp)
    print("Feasible paths:", fp)

# -------------------- Entry --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/', help='path for saving trained models')
    args = parser.parse_args()
    print(args)
    main(args)
