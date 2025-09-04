import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random
from tqdm import tqdm
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool

# 尝试导入 CuPy，如果失败则回退到 NumPy
try:
    import cupy as cp
    _np = cp
    CUPY_ENABLED = True
except ImportError:
    _np = np
    CUPY_ENABLED = False

# 常数定义
g = 9.8
v_drone_min, v_drone_max = 70, 140
smoke_radius = 10
smoke_duration = 20
smoke_sink_speed = 3
target_pos_np = np.array([0, 200, 0], dtype=np.float64)
drone_pos0_np = np.array([17800, 0, 1800], dtype=np.float64)
missile_pos0_np = np.array([20000, 0, 2000], dtype=np.float64)
missile_speed = 300.0
missile_target_np = np.array([0, 0, 0], dtype=np.float64)

# 计算命中时间与导弹方向向量
T_hit = np.linalg.norm(missile_target_np - missile_pos0_np) / missile_speed
missile_dir_np = (missile_target_np - missile_pos0_np) / np.linalg.norm(missile_target_np - missile_pos0_np)

def missile_pos_func(t, m_pos0, m_speed, m_dir):
    return m_pos0 + t.reshape(-1, 1) * m_speed * m_dir

def smoke_position(release_pos, release_time, burst_time, t, v, direction):
    fall_time = burst_time - release_time
    horizontal_disp = fall_time * v * direction
    vertical_disp = -0.5 * g * fall_time**2
    burst_pos = release_pos + horizontal_disp + _np.array([0.0, 0.0, vertical_disp])
    
    dt = t - burst_time
    # 使用 .stack 替代 .array，避免类型错误
    displacement = _np.stack([_np.zeros_like(dt), _np.zeros_like(dt), -smoke_sink_speed * dt], axis=-1)
    return burst_pos + displacement

def calc_coverage_time(individual, dt=0.1):
    v, dx, dy, t_release, t_burst = individual
    direction = _np.array([dx, dy, 0], dtype=_np.float64)
    direction /= _np.linalg.norm(direction)
    
    v_arr = _np.array(v, dtype=_np.float64)
    release_pos_arr = _np.array(drone_pos0_np, dtype=_np.float64) + t_release * v_arr * direction
    
    missile_dir_arr = _np.array(missile_dir_np, dtype=_np.float64)
    missile_pos0_arr = _np.array(missile_pos0_np, dtype=_np.float64)
    target_pos_arr = _np.array(target_pos_np, dtype=_np.float64)
    
    t_global_arr = _np.arange(t_burst, min(t_burst + smoke_duration, T_hit) + dt/2, dt, dtype=_np.float64)
    if t_global_arr.size == 0:
        return 0.0
    
    smoke_pos_arr = smoke_position(release_pos_arr, t_release, t_burst, t_global_arr, v_arr, direction)
    m_pos_arr = missile_pos_func(t_global_arr, missile_pos0_arr, missile_speed, missile_dir_arr)

    ab = target_pos_arr - m_pos_arr
    ap = smoke_pos_arr - m_pos_arr
    
    proj_numerator = _np.einsum('ij,ij->i', ap, ab)
    proj_denominator = _np.einsum('ij,ij->i', ab, ab)
    safe_denominator = _np.where(_np.isclose(proj_denominator, 0), 1e-10, proj_denominator)
    proj = _np.clip(proj_numerator / safe_denominator, 0, 1)
    
    closest = m_pos_arr + proj.reshape(-1, 1) * ab
    dist_arr = _np.linalg.norm(smoke_pos_arr - closest, axis=1)

    coverage_mask = dist_arr <= smoke_radius
    total_time = _np.sum(coverage_mask) * dt

    if CUPY_ENABLED:
        total_time = cp.asnumpy(total_time)
        
    return float(total_time)

def evaluate(individual):
    v, dx, dy, t1, t2, t3, dt1, dt2, dt3 = individual
    norm = np.sqrt(dx**2 + dy**2 + 1e-10)
    dx, dy = dx / norm, dy / norm
    
    total_coverage = 0.0
    penalty = 0.0
    min_drop_interval = 2.0
    
    if t2 < t1 + min_drop_interval:
        penalty -= 1000 * (min_drop_interval + t1 - t2)
    if t3 < t2 + min_drop_interval:
        penalty -= 1000 * (min_drop_interval + t2 - t3)
        
    for ti, dti in [(t1, dt1), (t2, dt2), (t3, dt3)]:
        burst_time = ti + dti
        if burst_time > T_hit + 1:
            penalty -= 500
            continue
        total_coverage += calc_coverage_time((v, dx, dy, ti, burst_time))
    
    return total_coverage + penalty,

def optimize_smoke_strategy(pop_size=400, gen_count=500):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_v", random.uniform, v_drone_min, v_drone_max)
    toolbox.register("attr_dir", random.uniform, -1, 1)
    toolbox.register("attr_t", random.uniform, 0, T_hit)
    toolbox.register("attr_dt", random.uniform, 0, 10)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_v, toolbox.attr_dir, toolbox.attr_dir,
                      toolbox.attr_t, toolbox.attr_t, toolbox.attr_t,
                      toolbox.attr_dt, toolbox.attr_dt, toolbox.attr_dt), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=pop_size)
    
    if CUPY_ENABLED:
        toolbox.register("map", map)
        pbar = tqdm(total=gen_count, desc="Optimizing Strategy (CuPy)")
    else:
        num_processes = os.cpu_count() - 1 or 1
        pool = Pool(processes=num_processes)
        toolbox.register("map", pool.map)
        pbar = tqdm(total=gen_count, desc="Optimizing Strategy (NumPy/MP)")

    for gen in range(gen_count):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.3)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=len(pop))
        pbar.update(1)

    pbar.close()
    if not CUPY_ENABLED:
        pool.close()
        pool.join()

    best_ind = tools.selBest(pop, 1)[0]
    
    v, dx, dy, t1, t2, t3, dt1, dt2, dt3 = best_ind
    norm = np.sqrt(dx**2 + dy**2 + 1e-10)
    dx, dy = dx / norm, dy / norm
    direction_angle = math.degrees(math.atan2(dy, dx))
    if direction_angle < 0: direction_angle += 360
    direction = np.array([dx, dy, 0])

    data = []
    release_points, explode_points, coverage_times = [], [], []
    for i, (ti, dti) in enumerate([(t1, dt1), (t2, dt2), (t3, dt3)], 1):
        burst_time = ti + dti
        coverage_time = calc_coverage_time((v, dx, dy, ti, burst_time))
        coverage_times.append(coverage_time)
        release_pos = drone_pos0_np + ti * v * direction
        fall_time = dti
        horizontal_disp = fall_time * v * direction
        vertical_disp = -0.5 * g * fall_time**2
        burst_pos = release_pos + horizontal_disp + np.array([0, 0, vertical_disp])
        data.append([direction_angle, v, i, *release_pos, *burst_pos, coverage_time])
        release_points.append(release_pos)
        explode_points.append(burst_pos)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'Data')
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(data, columns=['无人机运动方向 (度)', '无人机运动速度 (m/s)', '烟幕干扰弹编号',
                                     '烟幕干扰弹投放点x坐标 (m)', '烟幕干扰弹投放点y坐标 (m)', 
                                     '烟幕干扰弹投放点z坐标 (m)', '烟幕干扰弹起爆点x坐标 (m)', 
                                     '烟幕干扰弹起爆点y坐标 (m)', '烟幕干扰弹起爆点z坐标 (m)', 
                                     '有效干扰时长 (s)'])
    df.to_excel(os.path.join(output_dir, 'result1.xlsx'), index=False)
    
    # 绘图部分 - 整合为一张三维图
    dt_eval = 0.1
    t_traj = np.linspace(0.0, T_hit, 400)
    M_traj = missile_pos_func(t_traj, missile_pos0_np, missile_speed, missile_dir_np)

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(M_traj[:, 0], M_traj[:, 1], M_traj[:, 2], label="导弹轨迹", color='red')
    ax1.scatter(drone_pos0_np[0], drone_pos0_np[1], drone_pos0_np[2], marker='^', s=60, label="无人机起点", color='tab:blue')
    ax1.scatter(target_pos_np[0], target_pos_np[1], target_pos_np[2], marker='s', s=50, label="真目标", color='magenta')
    
    # 绘制无人机轨迹
    drone_path_t = np.linspace(0, T_hit, 400)
    drone_path = drone_pos0_np + drone_path_t[:, np.newaxis] * v * direction
    ax1.plot(drone_path[:, 0], drone_path[:, 1], drone_path[:, 2], label="无人机轨迹", color='tab:blue', linestyle='--')
    
    # 绘制每个烟雾弹的轨迹
    colors = ['tab:green', 'purple', 'orange']
    labels = ['烟幕弹1', '烟幕弹2', '烟幕弹3']
    
    for i in range(3):
        best_drop = release_points[i]
        best_explode = explode_points[i]
        best_t_exp = t1 + dt1 if i == 0 else (t2 + dt2 if i == 1 else t3 + dt3)

        t_smoke_np = np.arange(best_t_exp, min(best_t_exp + smoke_duration, T_hit) + dt_eval/2, dt_eval)
        if t_smoke_np.size == 0:
            continue
        
        t_smoke_arr = _np.array(t_smoke_np)
        release_pos_arr = _np.array(release_points[i])
        direction_arr = _np.array(direction)
        
        # 修正：确保所有计算使用正确的数组类型
        C_traj_arr = smoke_position(release_pos_arr, best_t_exp - (dt1 if i==0 else (dt2 if i==1 else dt3)), best_t_exp, t_smoke_arr, v, direction_arr)
        
        # 修正：直接在主循环外部计算的M_traj上索引
        M_traj_smoke_np = M_traj[np.searchsorted(t_traj, t_smoke_np)]
        
        # 修正：统一使用CuPy/NumPy数组进行绘图
        ax1.scatter(best_drop[0], best_drop[1], best_drop[2], marker='o', s=50, color=colors[i])
        ax1.scatter(best_explode[0], best_explode[1], best_explode[2], marker='*', s=80, color=colors[i], label=labels[i])
        ax1.plot(C_traj_arr[:, 0], C_traj_arr[:, 1], C_traj_arr[:, 2], linestyle='--', color=colors[i])

        # 时间-距离曲线图
        ax2 = fig.add_subplot(1, 2, 2)
        ab = target_pos_np - M_traj_smoke_np
        ap = C_traj_arr - _np.array(M_traj_smoke_np)

        proj_numerator = _np.einsum('ij,ij->i', ap, ab)
        proj_denominator = _np.einsum('ij,ij->i', ab, ab)
        safe_denominator = _np.where(_np.isclose(proj_denominator, 0), 1e-10, proj_denominator)
        proj = _np.clip(proj_numerator / safe_denominator, 0, 1)
        
        closest = _np.array(M_traj_smoke_np) + proj.reshape(-1, 1) * ab
        dist_arr = _np.linalg.norm(C_traj_arr - closest, axis=1)
        
        dist_curve = cp.asnumpy(dist_arr) if CUPY_ENABLED else dist_arr
        
        ax2.plot(t_smoke_np, dist_curve, label=labels[i], color=colors[i])
        
        best_intervals = []
        masking = dist_curve <= smoke_radius
        start = None
        for k in range(len(masking)):
            if masking[k]:
                if start is None: start = t_smoke_np[k]
            elif start is not None:
                best_intervals.append((start, t_smoke_np[k-1]))
                start = None
        if start is not None: best_intervals.append((start, t_smoke_np[-1]))
        
        shaded_plotted = False
        for a, b in best_intervals:
            a_clip = max(a, t_smoke_np[0])
            b_clip = min(b, t_smoke_np[-1])
            if b_clip > a_clip:
                ax2.axvspan(a_clip, b_clip, alpha=0.15, color=colors[i])
                
    ax1.set_title("三维轨迹示意图")
    ax1.set_xlabel("X (米)")
    ax1.set_ylabel("Y (米)")
    ax1.set_zlabel("Z (米)")
    ax1.legend()
    
    ax2.axhline(y=smoke_radius, linestyle='--', color='r', label="烟幕有效半径")
    ax2.set_title("时间-距离 曲线")
    ax2.set_xlabel("时间 (秒)")
    ax2.set_ylabel("最小距离 (米)")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

    print(f"优化完成，结果已保存到 {os.path.join(output_dir, 'result1.xlsx')}。")
    print(f"最佳总有效干扰时长: {sum(coverage_times):.2f} 秒")
    print("最佳策略参数:")
    print(f"  无人机速度: {v:.2f} m/s")
    print(f"  运动方向 (角度): {direction_angle:.2f} 度")
    print("  烟幕弹投放时间 (相对无人机起点):")
    for i, t_val in enumerate([t1, t2, t3], 1):
        print(f"    烟幕弹{i}: {t_val:.2f} s")
    print("  烟幕弹自由落体时间:")
    for i, dt_val in enumerate([dt1, dt2, dt3], 1):
        print(f"    烟幕弹{i}: {dt_val:.2f} s")
    print(f"  有效干扰时长: {[f'{ct:.2f}' for ct in coverage_times]}")

if __name__ == "__main__":
    if CUPY_ENABLED:
        print("CUDA/GPU 加速已启用，将不使用多进程。")
    else:
        print("未检测到 CuPy 或 CUDA 环境，使用 NumPy 并启用多进程。")
    
    # 更改pop_size 改变种群大小(默认400), 更变gen_count 改变迭代次数(默认500)
    optimize_smoke_strategy(pop_size=500, gen_count=500)