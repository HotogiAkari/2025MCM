# -*- coding: utf-8 -*-
"""
问题2: 使用贝叶斯优化自动搜索 FY1 的飞行方向、速度、投放时刻、起爆延时，
最大化对导弹 M1 的有效遮蔽时间(视线遮挡判定)并修正遮蔽区间绘图的 bug.
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
常量与参数设置
'''

# 坐标系: 假目标在原点 O=(0,0,0)，xy 为水平面，z 竖直向上
O = np.array([0.0, 0.0, 0.0])
# 导弹 M1 初始信息
M0 = np.array([20000.0, 0.0, 2000.0])  # 初始位置
vm = 300.0  # 速度 m/s，指向假目标
um = (O - M0) / np.linalg.norm(O - M0)  # 单位方向向量
T_hit = np.linalg.norm(O - M0) / vm  # 导弹到达假目标时刻
# 真目标，近似用中心点表示视线目标点
T_true = np.array([0.0, 200.0, 5.0])
# 无人机 FY1 初始信息
F0 = np.array([17800.0, 0.0, 1800.0])  # 初始位置
# 烟幕参数
R_eff = 10.0  # 有效半径 10m
smoke_duration = 20.0  # 起爆后有效时长 20s
sink_speed = 3.0  # 云团下沉速度 3m/s
g = 9.8  # 假设重力加速度为 9.8m/s^2
# 评估时间步长(越小越精确，计算越慢)
dt_eval = 0.02


def missile_pos(t: float) -> np.ndarray:
    """导弹在时间 t 的位置"""
    return M0 + vm * t * um

def distance_point_to_segment(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """点 P 到线段 AB 的最小距离."""
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
    给定 UAV 速度/航向/投放时刻/引信延时，返回: 
    - cover_time: 遮蔽总时长(秒)
    - intervals: 遮蔽时间区间列表 [(t1_start, t1_end), ...]
    - drop_point: 投放点坐标
    - explode_point: 起爆点坐标
    - explode_time: 起爆时刻
    """
    uf = np.array([np.cos(theta), np.sin(theta), 0.0])
    vf = v_uav * uf
    drop_point = F0 + vf * t_drop
    drop_point[2] = F0[2]
    t_exp = t_drop + tau
    explode_point = drop_point + vf * tau + np.array([0.0, 0.0, -0.5 * g * tau * tau])
    
    if explode_point[2] <= 0.0:
        return 0.0, [], drop_point, explode_point, t_exp
    
    t0 = t_exp
    t1 = min(t_exp + smoke_duration, T_hit)
    if t1 <= t0:
        return 0.0, [], drop_point, explode_point, t_exp
    
    times = np.arange(t0, t1 + dt_eval/2, dt_eval)
    covered_mask = np.zeros_like(times, dtype=bool)
    
    for idx, t in enumerate(times):
        Cm = explode_point + np.array([0.0, 0.0, -sink_speed * (t - t_exp)])
        Mt = missile_pos(t)
        dist = distance_point_to_segment(Cm, Mt, T_true)
        covered_mask[idx] = (dist <= R_eff)
    
    cover_time = float(np.sum(covered_mask) * dt_eval)
    
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

'''
贝叶斯优化目标函数'''
# 决策变量的搜索空间
t_drop_max = min(60.0, max(1.0, T_hit - 0.1))
space = [
    Real(70.0, 140.0, name="v_uav"),
    Real(0.0, 2.0 * np.pi, name="theta"),
    Real(0.0, t_drop_max, name="t_drop"),
    Real(0.2, 12.0, name="tau"),
]

@use_named_args(space)
def objective(v_uav, theta, t_drop, tau):
    """
    BO 的目标: 最小化 -遮蔽时长(即最大化遮蔽时长).
    对不可行(起爆点落地以下)自动返回较差值.
    """
    cover_time, _, _, explode_point, _ = simulate_coverage(v_uav, theta, t_drop, tau)
    if explode_point[2] <= 0.0:
        return 1e3
    return -cover_time

'''
封装主函数
'''
def run_optimization_and_plot(n_calls=60, n_initial_points=12):
    """
    执行贝叶斯优化并绘制结果图表.
    :n_calls (int): 贝叶斯优化的总评估次数.
    :n_initial_points (int): 初始随机采样的点数.
    """
    print("开始贝叶斯优化...")
    start_time = time.time()

    # 运行BO
    res = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=42,
        acq_func="EI"
    )

    end_time = time.time()
    
    # 获取最优解并重新计算结果
    best_v, best_theta, best_t_drop, best_tau = res.x
    best_cover, best_intervals, best_drop, best_explode, best_t_exp = simulate_coverage(
        best_v, best_theta, best_t_drop, best_tau
    )

    # 打印最优解
    print("\n======== 最优解(BO) ========")
    print(f"无人机速度 v_uav: {best_v:.3f} m/s")
    print(f"航向角 theta: {best_theta:.6f} rad(相对 x 轴)")
    print(f"投放时刻 t_drop: {best_t_drop:.3f} s")
    print(f"起爆延时 tau: {best_tau:.3f} s")
    print(f"起爆时刻 t_exp: {best_t_exp:.3f} s")
    print(f"投放点 drop_point: ({best_drop[0]:.2f}, {best_drop[1]:.2f}, {best_drop[2]:.2f}) m")
    print(f"起爆点 explode_point: ({best_explode[0]:.2f}, {best_explode[1]:.2f}, {best_explode[2]:.2f}) m")
    print(f"最大遮蔽总时长: {best_cover:.3f} s")
    if best_intervals:
        print("遮蔽时间区间: ")
        for (a, b) in best_intervals:
            print(f"  [{a:.3f} s, {b:.3f} s]")
    else:
        print("无有效遮蔽区间.")
        
    print(f"总耗时: {end_time - start_time:.2f} 秒")

    # 3D 轨迹数据
    t_traj = np.linspace(0.0, T_hit, 400)
    M_traj = np.array([missile_pos(t) for t in t_traj])
    t_smoke = np.arange(best_t_exp, min(best_t_exp + smoke_duration, T_hit) + dt_eval/2, dt_eval)
    if t_smoke.size > 0:
        C_traj = np.array([best_explode + np.array([0.0, 0.0, -sink_speed*(t - best_t_exp)]) for t in t_smoke])
    else:
        C_traj = np.empty((0, 3))
    dist_curve = np.array([distance_point_to_segment(C_traj[i], missile_pos(t_smoke[i]), T_true)
                           for i in range(len(t_smoke))]) if t_smoke.size > 0 else np.array([])
    
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1, projection='3d')
    ax1.plot(M_traj[:, 0], M_traj[:, 1], M_traj[:, 2], label="导弹轨迹", color='red')
    ax1.scatter(F0[0], F0[1], F0[2], marker='^', s=60, label="无人机起点", color='tab:blue')
    ax1.scatter(best_drop[0], best_drop[1], best_drop[2], marker='o', s=50, label="投放点", color='tab:green')
    ax1.scatter(best_explode[0], best_explode[1], best_explode[2], marker='*', s=80, label="起爆点", color='gold')
    if C_traj.size > 0:
        ax1.plot(C_traj[:, 0], C_traj[:, 1], C_traj[:, 2], linestyle='--', label="云团中心轨迹", color='c')
    ax1.scatter(T_true[0], T_true[1], T_true[2], marker='s', s=50, label="真目标(中心点)", color='magenta')
    ax1.set_title("三维轨迹示意")
    ax1.set_xlabel("X (米)")
    ax1.set_ylabel("Y (米)")
    ax1.set_zlabel("Z (米)")
    ax1.legend(loc='upper right')

    ax2 = plt.subplot(1, 2, 2)
    if t_smoke.size > 0:
        ax2.plot(t_smoke, dist_curve, label="距离: 云团中心→视线(导弹-真目标)", color='tab:blue')
    ax2.axhline(y=R_eff, linestyle='--', color='r', label="烟幕有效半径")
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

if __name__ == "__main__":
    # 修改n_call改变评估次数,修改n_initial_points改变初始采样点
    run_optimization_and_plot(n_calls=60, n_initial_points=12)