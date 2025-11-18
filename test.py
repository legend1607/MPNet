import numpy as np
import matplotlib.pyplot as plt
from environment.random_2d_env import Random2DEnv

# -------------------- 配置环境 --------------------
env_dict = {
    'env_dims': [50, 50],  # 2D环境宽高
    'start': [[5, 5]],
    'goal': [[45, 45]],
    'rectangle_obstacles': [[10, 10, 10, 5], [30, 20, 5, 10]],  # [x, y, width, height]
}

env = Random2DEnv(env_dict)

# -------------------- 可视化函数 --------------------
def plot_env(env, path=None):
    fig, ax = plt.subplots()
    # 绘制矩形障碍
    for rx, ry, rw, rh in env.rect_obstacles:
        ax.add_patch(plt.Rectangle((rx, ry), rw, rh, color='gray'))
    # 绘制圆形障碍
    for cx, cy, r in env.circle_obstacles:
        circle = plt.Circle((cx, cy), r, color='gray')
        ax.add_patch(circle)
    # 绘制起点和目标
    ax.plot(env.start[0], env.start[1], 'go', markersize=10, label='Start')
    ax.plot(env.goal[0], env.goal[1], 'ro', markersize=10, label='Goal')
    # 绘制路径
    if path is not None:
        path = np.array(path)
        ax.plot(path[:,0], path[:,1], 'b-', linewidth=2, label='Path')
        ax.scatter(path[:,0], path[:,1], color='blue')
    ax.set_xlim(env.bound[0][0], env.bound[1][0])
    ax.set_ylim(env.bound[0][1], env.bound[1][1])
    ax.set_aspect('equal')
    ax.legend()
    plt.show()

# -------------------- 简单随机路径测试 --------------------
# 使用直线连接 start->goal，并检查碰撞
def simple_path_test(env):
    steps = 100
    path = [env.start]
    for i in range(1, steps+1):
        ratio = i / steps
        point = env.interpolate(env.start, env.goal, ratio)
        if not env._point_in_free_space(point):
            print(f"Collision at step {i}, stopping path growth.")
            break
        path.append(point)
    path.append(env.goal)
    return path

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    print("环境信息:", env.get_problem())
    
    # 采样自由点
    sample = env.sample_empty_points()
    print("随机自由点:", sample)

    # 检查起点是否在自由空间
    print("Start in free space?", env._point_in_free_space(env.start))

    # 生成并检查简单路径
    path = simple_path_test(env)
    print("生成路径长度:", len(path))

    # 可视化
    plot_env(env, path)
