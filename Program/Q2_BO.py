# -*- coding: utf-8 -*-
"""
问题2: 使用贝叶斯优化自动搜索 FY1 的飞行方向、速度、投放时刻、起爆延时，
最大化对导弹 M1 的有效遮蔽时间（视线遮挡判定）并修正遮蔽区间绘图的 bug。
"""
import time
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 配置matplot
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

'''
设置参数
'''
# 坐标系：假目标在原点 O=(0,0,0)，xy 为水平面，z 竖直向上
O = np.array([0.0, 0.0, 0.0])

# 导弹 M1 初始信息
M0 = np.array([20000.0, 0.0, 2000.0])   # 初始位置
vm = 300.0                              # 速度 m/s，指向假目标
um = (O - M0) / np.linalg.norm(O - M0)  # 单位方向向量
T_hit = np.linalg.norm(O - M0) / vm     # 导弹到达假目标时刻

# 真目标，近似用中心点表示视线目标点
T_true = np.array([0.0, 200.0, 5.0])

# 无人机 FY1 初始信息
F0 = np.array([17800.0, 0.0, 1800.0])   # 初始位置

# 烟幕参数
R_eff = 10.0            # 有效半径 10m
smoke_duration = 20.0   # 起爆后有效时长 20s
sink_speed = 3.0        # 云团下沉速度 3m/s
g = 9.8                 # 假设重力加速度为 9.8m/s^2

# 评估时间步长（越小越精确，计算越慢）
dt_eval = 0.02


# =============================
# 几何与物理工具函数
# =============================

def missile_pos(t: float) -> np.ndarray:
    """导弹在时间 t 的位置"""
    return M0 + vm * t * um

def distance_point_to_segment(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """
    点 P 到线段 AB 的最小距离。
    若 AB 为零长，则退化为点距。
    """
    AB = B - A
    AB2 = np.dot(AB, AB)
    if AB2 <= 1e-12:
        return np.linalg.norm(P - A)
    u = np.dot(P - A, AB) / AB2
    u = max(0.0, min(1.0, u))
    closest = A + u * AB
    return np.linalg.norm(P - closest)

def simulate_coverage(v_uav: float, theta: float, t_drop: float, tau: float):
    """
    给定 UAV 速度/航向/投放时刻/引信延时，返回：
    - cover_time: 遮蔽总时长（秒）
    - intervals: 遮蔽时间区间列表 [(t1_start, t1_end), ...]
    - drop_point: 投放点坐标
    - explode_point: 起爆点坐标
    - explode_time: 起爆时刻
    物理模型：
      * 投放后抛体：初速度 = UAV 速度向量 (vx, vy, vz=0)，z 方向仅受重力
      * 起爆后云团中心以 (0,0,-sink_speed) 匀速下沉
      * 遮蔽判定：云团中心到“导弹位置—真目标点”的线段距离 ≤ R_eff
    """
    # UAV 航向单位向量（仅 xy 平面）
    uf = np.array([np.cos(theta), np.sin(theta), 0.0])
    # UAV 速度向量（等高度）
    vf = v_uav * uf

    # 投放点（UAV 等高度直线飞到 t_drop 时刻的位置）
    drop_point = F0 + vf * t_drop
    drop_point[2] = F0[2]  # 等高度飞行，确保 z=1800

    # 起爆时刻
    t_exp = t_drop + tau

    # 抛体运动得到起爆点：
    # 初速度 = vf，位移 = vf * tau + 0.5 * a * tau^2，其中 a = (0,0,-g)
    explode_point = drop_point + vf * tau + np.array([0.0, 0.0, -0.5 * g * tau * tau])

    # 物理可行性检查：起爆点必须在地面以上
    if explode_point[2] <= 0.0:
        return 0.0, [], drop_point, explode_point, t_exp

    # 遮蔽评估：在 [t_exp, min(t_exp+20, T_hit)] 内做时间采样
    t0 = t_exp
    t1 = min(t_exp + smoke_duration, T_hit)
    if t1 <= t0:
        return 0.0, [], drop_point, explode_point, t_exp

    times = np.arange(t0, t1 + dt_eval/2, dt_eval)
    covered_mask = np.zeros_like(times, dtype=bool)

    # 云团中心：C(t) = explode_point + (0,0,-sink_speed*(t - t_exp))
    # 逐时刻判定：dist( C(t), segment [M(t), T_true] ) ≤ R_eff
    for idx, t in enumerate(times):
        Cm = explode_point + np.array([0.0, 0.0, -sink_speed * (t - t_exp)])
        Mt = missile_pos(t)
        dist = distance_point_to_segment(Cm, Mt, T_true)
        covered_mask[idx] = (dist <= R_eff)

    # 统计遮蔽总时长与区间
    cover_time = float(np.sum(covered_mask) * dt_eval)

    # 提取连续区间
    intervals = []
    if np.any(covered_mask):
        edges = np.diff(covered_mask.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0] + 1
        if covered_mask[0]:
            starts = np.r_[0, starts]
        if covered_mask[-1]:
            ends = np.r_[ends, covered_mask.size]
        for s, e in zip(starts, ends):
            intervals.append((times[s], times[e - 1] + dt_eval))

    return cover_time, intervals, drop_point, explode_point, t_exp


# =============================
# 贝叶斯优化（BO）设置
# =============================

# 决策变量：
t_drop_max = min(60.0, max(1.0, T_hit - 0.1))  # 避免完全晚于撞击
space = [
    Real(70.0, 140.0, name="v_uav"),
    Real(0.0, 2.0 * np.pi, name="theta"),
    Real(0.0, t_drop_max, name="t_drop"),
    Real(0.2, 12.0, name="tau"),
]

@use_named_args(space)
def objective(v_uav, theta, t_drop, tau):
    """
    BO 的目标：最小化 -遮蔽时长（即最大化遮蔽时长）。
    对不可行（起爆点落地以下）自动返回较差值。
    """
    cover_time, intervals, drop_point, explode_point, t_exp = simulate_coverage(
        v_uav, theta, t_drop, tau
    )
    # 若起爆点在地面以下，给惩罚（返回大正值）
    if explode_point[2] <= 0.0:
        return 1e3
    # 目标：-遮蔽时长
    return -cover_time


# =============================
# 运行 BO
# =============================
print("开始贝叶斯优化（可能需要几十秒，取决于 n_calls）...")
res = gp_minimize(
    objective,
    space,
    n_calls=60,          # 评估次数，可根据性能调整
    n_initial_points=12, # 初始随机采样
    random_state=42,
    acq_func="EI"
)

best_v, best_theta, best_t_drop, best_tau = res.x
best_cover, best_intervals, best_drop, best_explode, best_t_exp = simulate_coverage(
    best_v, best_theta, best_t_drop, best_tau
)

print("\n======== 最优解（BO） ========")
print(f"无人机速度 v_uav：{best_v:.3f} m/s")
print(f"航向角 theta：{best_theta:.6f} rad（相对 x 轴）")
print(f"投放时刻 t_drop：{best_t_drop:.3f} s")
print(f"起爆延时 tau：{best_tau:.3f} s")
print(f"起爆时刻 t_exp：{best_t_exp:.3f} s")
print(f"投放点 drop_point：({best_drop[0]:.2f}, {best_drop[1]:.2f}, {best_drop[2]:.2f}) m")
print(f"起爆点 explode_point：({best_explode[0]:.2f}, {best_explode[1]:.2f}, {best_explode[2]:.2f}) m")
print(f"最大遮蔽总时长：{best_cover:.3f} s")
if best_intervals:
    print("遮蔽时间区间：")
    for (a, b) in best_intervals:
        print(f"  [{a:.3f} s, {b:.3f} s]")
else:
    print("无有效遮蔽区间。")


# =============================
# 可视化（中文）
# =============================

# 3D 轨迹数据
t_traj = np.linspace(0.0, T_hit, 400)
M_traj = np.array([missile_pos(t) for t in t_traj])

t_smoke = np.arange(best_t_exp, min(best_t_exp + smoke_duration, T_hit) + dt_eval/2, dt_eval)
if t_smoke.size > 0:
    C_traj = np.array([best_explode + np.array([0.0, 0.0, -sink_speed*(t - best_t_exp)]) for t in t_smoke])
else:
    C_traj = np.empty((0, 3))

# 2D 距离曲线：dist( 云团中心, 线段[ M(t), T_true ] )
dist_curve = np.array([distance_point_to_segment(C_traj[i], missile_pos(t_smoke[i]), T_true)
                       for i in range(len(t_smoke))]) if t_smoke.size > 0 else np.array([])

# ---- 绘图开始 ----
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(1, 2, 1, projection='3d')

# 导弹轨迹
ax1.plot(M_traj[:, 0], M_traj[:, 1], M_traj[:, 2], label="导弹轨迹", color='red')

# 无人机点位
ax1.scatter(F0[0], F0[1], F0[2], marker='^', s=60, label="无人机起点", color='tab:blue')
ax1.scatter(best_drop[0], best_drop[1], best_drop[2], marker='o', s=50, label="投放点", color='tab:green')
ax1.scatter(best_explode[0], best_explode[1], best_explode[2], marker='*', s=80, label="起爆点", color='gold')

# 云团中心轨迹（若存在）
if C_traj.size > 0:
    ax1.plot(C_traj[:, 0], C_traj[:, 1], C_traj[:, 2], linestyle='--', label="云团中心轨迹", color='c')

# 真目标点
ax1.scatter(T_true[0], T_true[1], T_true[2], marker='s', s=50, label="真目标（中心点）", color='magenta')

ax1.set_title("三维轨迹示意")
ax1.set_xlabel("X (米)")
ax1.set_ylabel("Y (米)")
ax1.set_zlabel("Z (米)")
ax1.legend(loc='upper right')

# 2D 距离-时间图
ax2 = plt.subplot(1, 2, 2)
if t_smoke.size > 0:
    ax2.plot(t_smoke, dist_curve, label="距离：云团中心→视线（导弹-真目标）", color='tab:blue')
ax2.axhline(y=R_eff, linestyle='--', color='r', label="烟幕有效半径")

# 遮蔽区间高亮（稳健实现：只给第一个 patch 加 label，避免重复图例条目）
if best_intervals and t_smoke.size > 0:
    shaded_plotted = False
    for (a, b) in best_intervals:
        a_clip = max(a, t_smoke[0])
        b_clip = min(b, t_smoke[-1])
        if b_clip > a_clip:
            label = "遮蔽区间" if not shaded_plotted else None
            ax2.axvspan(a_clip, b_clip, alpha=0.25, label=label, color='yellow')
            shaded_plotted = True

ax2.set_title("时间-距离 曲线")
ax2.set_xlabel("时间 (秒)")
ax2.set_ylabel("最小距离 (米)")
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
