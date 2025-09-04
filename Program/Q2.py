import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

'''
参数设置
'''
missile_pos = np.array([20000, 0, 2000])                # 导弹初始位置
fake_target_pos = np.array([0, 0, 0])    # 假目标
missile_speed = 300
missile_dir = (target_pos - missile_pos) / np.linalg.norm(target_pos - missile_pos)

drone_pos = np.array([17800, 0, 1800])    # 无人机 FY1 初始位置

smoke_radius = 10.0
smoke_duration = 20.0
smoke_sink_speed = 3.0

# -----------------------------
# 搜索空间
# -----------------------------
space = [
    Real(70, 140, name="v"),              # 无人机速度
    Real(0, 2*np.pi, name="theta"),       # 飞行方向
    Real(0, 60, name="t_drop"),           # 投放时刻
    Real(1, 10, name="t_delay")           # 起爆延时
]

def missile_position(t):
    return missile_pos + missile_speed * t * missile_dir

@use_named_args(space)
def objective(v, theta, t_drop, t_delay):
    # 无人机投放点
    drone_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
    drop_point = drone_pos + v * t_drop * drone_dir

    # 起爆点
    explode_point = drop_point
    explode_time = t_drop + t_delay

    # 云团随时间下降
    t_samples = np.linspace(explode_time, explode_time + smoke_duration, 200)
    smoke_centers = np.array([explode_point + np.array([0,0,-smoke_sink_speed*(t-explode_time)]) for t in t_samples])

    # 遮蔽时间计算
    cover_time = 0.0
    dt = (t_samples[1] - t_samples[0])
    for i, t in enumerate(t_samples):
        missile_pos_t = missile_position(t)
        dist = np.linalg.norm(missile_pos_t - smoke_centers[i])
        if dist <= smoke_radius:
            cover_time += dt

    return -cover_time  # BO 要求最小化

# -----------------------------
# 贝叶斯优化
# -----------------------------
res = gp_minimize(objective, space, n_calls=40, random_state=42)

print("最优参数:")
print("速度 v =", res.x[0])
print("方向 theta =", res.x[1])
print("投放时刻 t_drop =", res.x[2])
print("起爆延时 t_delay =", res.x[3])
print("最大遮蔽时间 =", -res.fun)

# -----------------------------
# 可视化
# -----------------------------
# 计算最优解下的轨迹
v, theta, t_drop, t_delay = res.x
drone_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
drop_point = drone_pos + v * t_drop * drone_dir
explode_point = drop_point
explode_time = t_drop + t_delay

t_samples = np.linspace(0, 80, 400)  # 仿真到80秒
missile_traj = np.array([missile_position(t) for t in t_samples])

t_smoke = np.linspace(explode_time, explode_time + smoke_duration, 100)
smoke_traj = np.array([explode_point + np.array([0,0,-smoke_sink_speed*(t-explode_time)]) for t in t_smoke])

# 遮蔽区间
cover_mask = []
for i, t in enumerate(t_smoke):
    missile_pos_t = missile_position(t)
    if np.linalg.norm(missile_pos_t - smoke_traj[i]) <= smoke_radius:
        cover_mask.append(t)
if cover_mask:
    cover_start, cover_end = min(cover_mask), max(cover_mask)
else:
    cover_start, cover_end = None, None

# 绘制3D轨迹
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 导弹轨迹
ax.plot(missile_traj[:,0], missile_traj[:,1], missile_traj[:,2], 'r-', label="Missile Trajectory")

# 无人机投放点 & 起爆点
ax.scatter(drone_pos[0], drone_pos[1], drone_pos[2], c='b', marker='^', s=80, label="Drone Start")
ax.scatter(drop_point[0], drop_point[1], drop_point[2], c='g', marker='o', s=80, label="Drop Point")
ax.scatter(explode_point[0], explode_point[1], explode_point[2], c='y', marker='*', s=120, label="Explode Point")

# 烟幕轨迹
ax.plot(smoke_traj[:,0], smoke_traj[:,1], smoke_traj[:,2], 'c--', label="Smoke Cloud Center")

# 标签与设置
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Missile & Smoke Trajectory (BO Optimal)")
ax.legend()

plt.show()

# 输出遮蔽区间
if cover_start:
    print(f"导弹被遮蔽的时间区间: {cover_start:.2f} s - {cover_end:.2f} s")
else:
    print("没有有效遮蔽。")