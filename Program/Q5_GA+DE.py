# -*- coding: utf-8 -*-
"""
问题5：利用5架无人机，每架无人机至多投放3枚烟幕干扰弹，
实施对M1、M2、M3等3枚来袭导弹的干扰。
优化方法：三阶段混合优化，结合混合GA-DE、约束修复和迭代局部精炼。
"""

import numpy as np
import pandas as pd
import os
import pickle
import random
from scipy.optimize import minimize
from numba import jit, cuda
from tqdm import tqdm
import multiprocessing

# ======================
# 参数设置
# ======================

# 导弹与烟幕参数
MISSILE_SPEED = 300.0   # 导弹速度 (m/s)，用于计算导弹飞行路径
SMOKE_RADIUS = 10.0      # 烟幕的有效干扰半径（m），判断导弹是否被干扰的关键参数
SMOKE_LIFE = 20.0        # 烟幕的有效时长 (s)，超过此时间烟幕失效
SMOKE_SINK = 3.0         # 烟幕的下沉速度 (m/s)，影响烟幕的实时高度
DT = 0.1                 # 模拟时间步长 (s)，用于计算导弹和烟幕在每个瞬间的位置
TARGET_CENTER = np.array([0.0, 200.0, 5.0], dtype=np.float64) # 目标（圆柱，近似中心点），导弹的最终攻击点

# 无人机与烟幕弹数量
NUM_UAVS = 5            # 无人机数量
MAX_BOMBS_PER_UAV = 3   # 每架无人机最多可投放的烟幕弹数量
NUM_BOMBS = NUM_UAVS * MAX_BOMBS_PER_UAV # 烟幕弹总数
NUM_PARAMS_PER_UAV = 2 + MAX_BOMBS_PER_UAV # 每架无人机对应的优化参数数量 (方向、速度、3个投放时间)
NUM_PARAMS = NUM_UAVS * NUM_PARAMS_PER_UAV # 优化参数总数

# 导弹初始位置
MISSILE_INITS = {
    'M1': np.array([20000.0, 0.0, 2000.0]),
    'M2': np.array([19000.0, 600.0, 2100.0]),
    'M3': np.array([18000.0, -600.0, 1900.0])
}

# 无人机 FY1-FY5 初始位置
UAV_INITS = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0, -2000.0, 1300.0])
}
UAV_INITS_LIST = np.array(list(UAV_INITS.values()))

# 最大仿真时间，以最远导弹M1的命中时间为准
T_MAX = np.linalg.norm(TARGET_CENTER - MISSILE_INITS['M1']) / MISSILE_SPEED
T_MAX = round(T_MAX, 1) + 1.0 # 确保涵盖整个仿真过程

# ======================
# 工具函数（使用numba加速）
# ======================

@jit(nopython=True)
def line_segment_point_dist(p1, p2, q):
    """
    计算点 q 到线段 p1-p2 的最短距离。
    用于判断烟幕中心是否在导弹的攻击路径上。
    
    参数:
    p1 (np.array): 线段的起点，例如导弹在t时刻的位置。
    p2 (np.array): 线段的终点，例如目标中心。
    q (np.array): 要计算距离的点，例如烟幕的实时中心。
    """
    v = p2 - p1
    w = q - p1
    # 计算点q在线段向量v上的投影比例t
    t = np.dot(w, v) / (np.dot(v, v) + 1e-9)
    # 将t限制在[0, 1]范围内，确保投影点在线段上
    t = max(0.0, min(t, 1.0))
    # 找到线段上离q最近的点
    closest = p1 + t * v
    # 返回欧氏距离
    return np.linalg.norm(closest - q)

# ======================
# 优化目标函数 (GPU加速)
# ======================

@cuda.jit
def final_evaluate_kernel(pop, missile_paths_cache_gpu, time_steps_gpu, uav_inits_gpu, target_center_gpu, num_missiles, num_time_steps, results_out):
    """
    GPU内核，并行计算整个种群的适应度。每个线程负责计算种群中的一个个体。
    
    参数:
    pop (np.array): 种群，每个个体是优化参数数组。
    missile_paths_cache_gpu (cuda.array): 预计算的导弹路径（GPU内存）。
    time_steps_gpu (cuda.array): 时间步长数组（GPU内存）。
    uav_inits_gpu (cuda.array): 无人机初始位置（GPU内存）。
    target_center_gpu (cuda.array): 目标中心位置（GPU内存）。
    num_missiles (int): 导弹数量。
    num_time_steps (int): 时间步数。
    results_out (cuda.array): 存储计算结果的数组（GPU内存）。
    """
    idx = cuda.grid(1)
    
    if idx < pop.shape[0]:
        params = pop[idx]
        total_coverage = 0.0
        
        # 使用本地数组存储烟幕弹信息，提高访问效率
        all_bombs_t_drop = cuda.local.array(NUM_BOMBS, dtype=np.float64)
        all_bombs_pos = cuda.local.array((NUM_BOMBS, 3), dtype=np.float64)

        bomb_counter = 0
        for uav_i in range(NUM_UAVS):
            start_idx = uav_i * NUM_PARAMS_PER_UAV
            theta_deg = params[start_idx]
            v = params[start_idx + 1]
            theta_rad = theta_deg * np.pi / 180.0
            
            direction = cuda.local.array(3, dtype=np.float64)
            direction[0] = np.cos(theta_rad)
            direction[1] = np.sin(theta_rad)
            direction[2] = 0.0
            
            uav_init_pos = uav_inits_gpu[uav_i]

            for j in range(MAX_BOMBS_PER_UAV):
                t_drop = params[start_idx + 2 + j]
                # 只有当投放时间有效（>=0）时才计算
                if t_drop >= 0.0:
                    all_bombs_t_drop[bomb_counter] = t_drop
                    # 计算烟幕弹投放时的位置
                    all_bombs_pos[bomb_counter, 0] = uav_init_pos[0] + direction[0] * v * t_drop
                    all_bombs_pos[bomb_counter, 1] = uav_init_pos[1] + direction[1] * v * t_drop
                    all_bombs_pos[bomb_counter, 2] = uav_init_pos[2] + direction[2] * v * t_drop
                    bomb_counter += 1
        
        # 将全局常量复制到局部变量以进行计算，这是Numba CUDA的最佳实践
        target_center_local = cuda.local.array(3, dtype=np.float64)
        target_center_local[0] = target_center_gpu[0]
        target_center_local[1] = target_center_gpu[1]
        target_center_local[2] = target_center_gpu[2]
        
        for missile_i in range(num_missiles):
            covered_count = 0
            for i in range(num_time_steps):
                t = time_steps_gpu[i]
                m_pos = missile_paths_cache_gpu[missile_i, i]
                blocked = False
                
                for bomb_i in range(bomb_counter):
                    t_b = all_bombs_t_drop[bomb_i]
                    pos_b = all_bombs_pos[bomb_i]
                    
                    # 检查烟幕是否在有效时间内
                    if t < t_b or t > t_b + SMOKE_LIFE:
                        continue

                    # 计算烟幕的实时中心位置（考虑下沉）
                    smoke_center = cuda.local.array(3, dtype=np.float64)
                    smoke_center[0] = pos_b[0]
                    smoke_center[1] = pos_b[1]
                    smoke_center[2] = pos_b[2] - SMOKE_SINK * (t - t_b)
                    
                    # 关键修复：将向量减法转换为逐元素操作，以便Numba CUDA编译
                    v_vec = cuda.local.array(3, dtype=np.float64)
                    w_vec = cuda.local.array(3, dtype=np.float64)
                    
                    for k in range(3):
                        v_vec[k] = target_center_local[k] - m_pos[k]
                        w_vec[k] = smoke_center[k] - m_pos[k]
                        
                    # 计算点积
                    dot_w_v = w_vec[0] * v_vec[0] + w_vec[1] * v_vec[1] + w_vec[2] * v_vec[2]
                    dot_v_v = v_vec[0] * v_vec[0] + v_vec[1] * v_vec[1] + v_vec[2] * v_vec[2]

                    t_val = dot_w_v / (dot_v_v + 1e-9)
                    t_val = max(0.0, min(t_val, 1.0))

                    closest = cuda.local.array(3, dtype=np.float64)
                    for k in range(3):
                        closest[k] = m_pos[k] + t_val * v_vec[k]
                    
                    # 计算距离
                    dist_sq = 0.0
                    for k in range(3):
                        dist_sq += (closest[k] - smoke_center[k])**2
                    
                    dist = dist_sq ** 0.5

                    # 如果距离小于等于烟幕半径，则认为导弹被干扰
                    if dist <= SMOKE_RADIUS:
                        blocked = True
                        break
                
                # 如果导弹在当前时间步被干扰，则累加时间
                if blocked:
                    covered_count += 1
            # 累加每个导弹的被干扰总时长
            total_coverage += covered_count * DT

        # 将总遮蔽时间取负值作为适应度，因为优化器通常寻找最小值
        results_out[idx] = -total_coverage

def final_evaluate(params, missile_paths_cache, time_steps):
    """
    CPU精细优化目标函数：最大化所有烟幕弹对所有导弹的实际总遮蔽时间。
    这是GPU版本在CPU上的实现，用于局部优化和结果验证。
    """
    strategy = _parse_params(params)
    total_coverage = 0.0
    
    all_bombs = []
    for uav_name, data in strategy.items():
        uav_init_pos = UAV_INITS[uav_name]
        theta_rad = np.deg2rad(data['theta'])
        direction = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
        for t_drop in data['drop_times']:
            # 计算烟幕弹投放位置
            pos_b = uav_init_pos + direction * data['v'] * t_drop
            all_bombs.append({'t_drop': t_drop, 'pos': pos_b})

    bombs_tuple = [(bomb['t_drop'], bomb['pos']) for bomb in all_bombs]

    missile_names = list(MISSILE_INITS.keys())
    
    for missile_name, path in zip(missile_names, missile_paths_cache):
        covered_count = 0
        for i, t in enumerate(time_steps):
            m_pos = path[i]
            blocked = False
            for t_b, pos_b in bombs_tuple:
                # 检查烟幕是否在有效时间内
                if t < t_b or t > t_b + SMOKE_LIFE:
                    continue
                # 计算烟幕实时位置（考虑下沉）
                smoke_center = np.array([pos_b[0], pos_b[1], pos_b[2] - SMOKE_SINK * (t - t_b)])
                # 计算烟幕与导弹路径的距离
                d = line_segment_point_dist(m_pos, TARGET_CENTER, smoke_center)
                if d <= SMOKE_RADIUS:
                    blocked = True
                    break
            if blocked:
                covered_count += 1
        total_coverage += covered_count * DT

    return -total_coverage

def _parse_params(params):
    """
    解析一维参数列表为可读的投放策略字典。
    
    参数列表的结构：
    [theta_uav1, v_uav1, t_drop1_uav1, t_drop2_uav1, t_drop3_uav1, 
     theta_uav2, v_uav2, t_drop1_uav2, t_drop2_uav2, t_drop3_uav2,
     ... ]
    """
    uav_names = list(UAV_INITS.keys())
    strategy = {}
    for i in range(NUM_UAVS):
        start_idx = i * NUM_PARAMS_PER_UAV
        theta_deg = params[start_idx]
        v = params[start_idx + 1]
        
        bombs = []
        for j in range(MAX_BOMBS_PER_UAV):
            t_drop = params[start_idx + 2 + j]
            # 忽略无效的投放时间（<0）
            if t_drop < 0:
                continue
            bombs.append(t_drop)
            
        strategy[uav_names[i]] = {
            'theta': theta_deg,
            'v': v,
            'drop_times': bombs
        }
    return strategy

# ======================
# 种群生成与约束修复
# ======================

def _generate_collaborative_strategy():
    """
    生成一个人工设计的、具有协同效果的初始策略。
    该策略基于无人机与导弹之间的初始距离，优先分配给最近的导弹。
    这有助于为优化提供一个更好的起点，提高收敛速度和质量。
    """
    uav_names = list(UAV_INITS.keys())
    missile_names = list(MISSILE_INITS.keys())
    
    assigned_uavs = set()
    assigned_missiles = set()
    uav_assignments = {}

    # 贪心匹配：将未分配的无人机和导弹中距离最近的一对进行匹配
    while len(assigned_missiles) < len(missile_names) and len(assigned_uavs) < len(uav_names):
        min_dist = float('inf')
        best_uav, best_missile = None, None

        for uav_name in uav_names:
            if uav_name in assigned_uavs:
                continue
            for missile_name in missile_names:
                if missile_name in assigned_missiles:
                    continue
                
                dist = np.linalg.norm(UAV_INITS[uav_name] - MISSILE_INITS[missile_name])
                if dist < min_dist:
                    min_dist = dist
                    best_uav = uav_name
                    best_missile = missile_name
        
        if best_uav and best_missile:
            uav_assignments[best_uav] = best_missile
            assigned_uavs.add(best_uav)
            assigned_missiles.add(best_missile)

    # 将剩余无人机分配给最近的导弹
    remaining_uavs = [uav for uav in uav_names if uav not in assigned_uavs]
    for uav_name in remaining_uavs:
        closest_missile = min(missile_names, key=lambda m: np.linalg.norm(UAV_INITS[uav_name] - MISSILE_INITS[m]))
        uav_assignments[uav_name] = closest_missile
            
    # 基于分配生成初始参数
    params = []
    for uav_name in uav_names:
        uav_init_pos = UAV_INITS[uav_name]
        assigned_missile_name = uav_assignments.get(uav_name)
        
        if assigned_missile_name:
            assigned_missile_pos = MISSILE_INITS[assigned_missile_name]
            direction_vector = assigned_missile_pos - uav_init_pos
            # 计算无人机朝向导弹的航向角
            theta_rad = np.arctan2(direction_vector[1], direction_vector[0])
            theta_deg = np.rad2deg(theta_rad)
            if theta_deg < 0:
                theta_deg += 360
            v = 105.0 # 设定一个初始速度
            
            # 估计到达导弹位置的时间，并以此为基准设定投放时间
            travel_time_to_missile = np.linalg.norm(direction_vector) / v
            t_drops = sorted([travel_time_to_missile, travel_time_to_missile + 10.0, travel_time_to_missile + 20.0])
            
            params.extend([theta_deg, v] + t_drops)
        else:
            # 如果没有找到分配，则随机生成参数
            params.extend([np.random.uniform(0, 360), np.random.uniform(70, 140)] + 
                          [np.random.uniform(0, T_MAX) for _ in range(MAX_BOMBS_PER_UAV)])
    
    return np.array(params)

def _generate_hybrid_initial_population(popsize, bounds):
    """
    生成一个混合初始种群，结合协同策略和随机生成。
    大约20%的个体来自协同策略，剩余80%随机生成。
    """
    initial_pop = []
    for _ in range(int(popsize * 0.2)):
        initial_pop.append(_generate_collaborative_strategy())
    
    for _ in range(popsize - len(initial_pop)):
        initial_pop.append(np.random.uniform(bounds[:, 0], bounds[:, 1]))
        
    return initial_pop

def _repair_constraints(params, min_interval=1.0):
    """
    约束修复函数：直接调整投放时间，使其满足最小时间间隔。
    防止烟幕弹在同一时间或非常接近的时间投放。
    """
    repaired_params = params.copy()
    for i in range(NUM_UAVS):
        start_idx = i * NUM_PARAMS_PER_UAV + 2
        # 获取并排序投放时间
        t_drops_indices = np.argsort(repaired_params[start_idx : start_idx + MAX_BOMBS_PER_UAV])
        t_drops = repaired_params[start_idx : start_idx + MAX_BOMBS_PER_UAV][t_drops_indices]

        for j in range(len(t_drops) - 1):
            # 如果时间间隔小于最小间隔，则进行修复
            if t_drops[j+1] - t_drops[j] < min_interval:
                t_drops[j+1] = t_drops[j] + min_interval

        # 将修复后的时间重新放回原位置
        repaired_params[start_idx : start_idx + MAX_BOMBS_PER_UAV][t_drops_indices] = t_drops
    return repaired_params

# ======================
# 混合GA-DE优化算法
# ======================

def ga_de_hybrid_optimization(func_gpu, bounds, popsize, max_iter, F_start, F_end, CR_start, CR_end, gpu_data):
    """
    自定义的混合GA-DE优化算法，包含约束修复。
    结合了遗传算法（GA）的交叉和变异，以及差分进化（DE）的变异策略。
    使用GPU加速的评估函数来提高计算效率。
    """
    num_params = len(bounds)
    pop = np.array(_generate_hybrid_initial_population(popsize, bounds), dtype=np.float64)
    
    # 第一次评估：使用GPU
    fitness = func_gpu(pop, gpu_data)
    
    best_idx = np.argmin(fitness)
    best_solution = pop[best_idx]
    
    for i in tqdm(range(max_iter), desc="混合GA-DE优化中"):
        # 线性递减的F和CR参数，平衡全局探索和局部开发
        F = F_start - (F_start - F_end) * (i / max_iter)
        CR = CR_start - (CR_start - CR_end) * (i / max_iter)
        
        new_pop = np.zeros_like(pop)
        
        for j in range(popsize):
            if random.random() < 0.5:
                # 随机选择GA或DE策略
                # GA变异与交叉
                parent1, parent2 = random.choices(pop, k=2)
                child = np.copy(pop[j])
                for k in range(num_params):
                    if random.random() < CR:
                        child[k] = (parent1[k] + parent2[k]) / 2.0
                
                mutation_rate = 0.1
                for k in range(num_params):
                    if random.random() < mutation_rate:
                        child[k] += np.random.normal(0, 0.1) * (bounds[k][1] - bounds[k][0])
                
                new_pop[j] = np.clip(child, bounds[:, 0], bounds[:, 1])

            else:
                # DE变异
                idxs = [idx for idx in range(popsize) if idx != j]
                a, b, c = pop[random.choice(idxs)], pop[random.choice(idxs)], pop[random.choice(idxs)]
                mutant = a + F * (b - c)
                
                trial = np.copy(pop[j])
                cross_points = np.random.rand(num_params) < CR
                trial[cross_points] = mutant[cross_points]
                
                new_pop[j] = np.clip(trial, bounds[:, 0], bounds[:, 1])
                
        # 约束修复
        for j in range(popsize):
            new_pop[j] = _repair_constraints(new_pop[j])
            
        new_fitness = func_gpu(new_pop, gpu_data)
        
        # 精英保留，如果新个体更好则替换旧个体
        for j in range(popsize):
            if new_fitness[j] < fitness[j]:
                pop[j] = new_pop[j]
                fitness[j] = new_fitness[j]
        
        current_best_idx = np.argmin(fitness)
        # 确保追踪到全局最佳解
        # 注意：此处需要调用CPU版本的final_evaluate来比较，因为GPU结果是设备上的数据
        if fitness[current_best_idx] < final_evaluate(best_solution, gpu_data['missile_paths_cache'], gpu_data['time_steps']):
            best_solution = pop[current_best_idx]
            
    return best_solution, -final_evaluate(best_solution, gpu_data['missile_paths_cache'], gpu_data['time_steps'])

# ======================
# 结果处理与保存
# ======================

def _generate_excel_data(best_params, excel_file_path, missile_paths_cache, time_steps):
    """
    生成并保存最终优化结果到Excel文件，包括详细的投放策略和总得分。
    """
    strategy = _parse_params(best_params)
    data_rows = []
    
    missile_names = list(MISSILE_INITS.keys())
    
    for uav_name, uav_data in strategy.items():
        uav_init_pos = UAV_INITS[uav_name]
        theta_rad = np.deg2rad(uav_data['theta'])
        direction = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])

        for bomb_idx, t_drop in enumerate(uav_data['drop_times']):
            if t_drop < 0:
                continue
            
            pos_b = uav_init_pos + direction * uav_data['v'] * t_drop
            total_bomb_coverage = 0.0
            interfered_missiles = []
            
            # 计算该烟幕弹对所有导弹的总有效遮蔽时间
            for missile_name, path in zip(missile_names, missile_paths_cache):
                covered_count = 0
                for i, t in enumerate(time_steps):
                    m_pos = path[i]
                    if t < t_drop or t > t_drop + SMOKE_LIFE:
                        continue
                    
                    smoke_center = np.array([pos_b[0], pos_b[1], pos_b[2] - SMOKE_SINK * (t - t_drop)])
                    d = line_segment_point_dist(m_pos, TARGET_CENTER, smoke_center)
                    if d <= SMOKE_RADIUS:
                        covered_count += 1
                
                coverage = covered_count * DT
                if coverage > 0:
                    total_bomb_coverage += coverage
                    interfered_missiles.append(f"{missile_name} ({coverage:.2f}s)")
            
            # 构造单行数据，包含总干扰时长
            data_row = {
                '无人机编号': uav_name,
                '无人机运动方向 (度)': uav_data['theta'],
                '无人机运动速度 (m/s)': uav_data['v'],
                '烟幕干扰弹编号': bomb_idx + 1,
                '烟幕干扰弹投放点的x坐标 (m)': pos_b[0],
                '烟幕干扰弹投放点的y坐标 (m)': pos_b[1],
                '烟幕干扰弹投放点的z坐标 (m)': pos_b[2],
                '烟幕干扰弹起爆点的x坐标 (m)': pos_b[0],
                '烟幕干扰弹起爆点的y坐标 (m)': pos_b[1],
                '烟幕干扰弹起爆点的z坐标 (m)': pos_b[2],
                '总有效干扰时长 (s)': total_bomb_coverage,
                '干扰的导弹编号': ', '.join(interfered_missiles) if interfered_missiles else '无'
            }
            data_rows.append(data_row)
            
    final_score = -final_evaluate(best_params, missile_paths_cache, time_steps)
    summary_data = [{'总有效遮蔽时间 (s)': final_score}]

    writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')
    
    columns_order = [
        '无人机编号', '无人机运动方向 (度)', '无人机运动速度 (m/s)', '烟幕干扰弹编号',
        '烟幕干扰弹投放点的x坐标 (m)', '烟幕干扰弹投放点的y坐标 (m)', '烟幕干扰弹投放点的z坐标 (m)',
        '烟幕干扰弹起爆点的x坐标 (m)', '烟幕干扰弹起爆点的y坐标 (m)', '烟幕干扰弹起爆点的z坐标 (m)',
        '总有效干扰时长 (s)', '干扰的导弹编号'
    ]
    df_result = pd.DataFrame(data_rows, columns=columns_order)
    df_result.to_excel(writer, sheet_name='详细投放策略', index=False)
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_excel(writer, sheet_name='总结', index=False)
    
    writer.close()
    print(f"\n最终结果已保存到 {excel_file_path}")

def _load_optimization_data(filepath):
    """从文件中加载历史优化数据，用于多轮次运行的断点续传。"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        print(f"未找到历史优化文件 {filepath}，将创建新文件。")
        return {'best_seed': None, 'best_coverage': -1.0, 'best_params': None, 'last_tried_seed': -1}

def _save_optimization_data(data, filepath):
    """将优化数据保存到文件。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
        print(f"历史优化数据已成功保存到 {filepath}。")
        
def _get_detailed_coverage(params, missile_paths_cache, time_steps):
    """
    计算每个烟幕弹对所有导弹的总有效遮蔽时间，并返回详细列表。
    用于第三阶段的迭代局部优化，以识别低效的烟幕弹。
    """
    strategy = _parse_params(params)
    detailed_coverage = []
    
    missile_names = list(MISSILE_INITS.keys())
    
    for uav_name, uav_data in strategy.items():
        uav_init_pos = UAV_INITS[uav_name]
        theta_rad = np.deg2rad(uav_data['theta'])
        direction = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
        
        for bomb_idx, t_drop in enumerate(uav_data['drop_times']):
            if t_drop < 0:
                continue
                
            pos_b = uav_init_pos + direction * uav_data['v'] * t_drop
            bomb_total_coverage = 0
            
            for missile_name, path in zip(missile_names, missile_paths_cache):
                covered_count = 0
                for i, t in enumerate(time_steps):
                    m_pos = path[i]
                    if t < t_drop or t > t_drop + SMOKE_LIFE:
                        blocked = False
                    else:
                        smoke_center = np.array([pos_b[0], pos_b[1], pos_b[2] - SMOKE_SINK * (t - t_drop)])
                        d = line_segment_point_dist(m_pos, TARGET_CENTER, smoke_center)
                        blocked = d <= SMOKE_RADIUS
                    if blocked:
                        covered_count += 1
                bomb_total_coverage += covered_count * DT
                
            detailed_coverage.append({
                'uav': uav_name,
                'bomb_index': bomb_idx,
                'coverage': bomb_total_coverage
            })
            
    return detailed_coverage
    
def _localized_evaluate(localized_params, full_params_template, localized_indices, missile_paths_cache, time_steps):
    """
    一个用于局部优化的包装器，只优化特定参数。
    SciPy的`minimize`函数需要一个只接受待优化参数的函数。
    """
    full_params = full_params_template.copy()
    full_params[localized_indices] = localized_params
    return final_evaluate(full_params, missile_paths_cache, time_steps)


# ======================
# 主程序
# ======================

def run_optimization(NUM_RUNS, POP_SIZE, MAX_ITER, F_START, F_END, CR_START, CR_END, seed_file_path):
    """
    运行多轮优化以寻找最佳烟幕弹投放方案。
    """
    
    # 参数边界
    bounds = []
    for _ in range(NUM_UAVS):
        # 优化参数依次为：航向角（度）、速度（m/s）、投放时间1、投放时间2、投放时间3
        bounds.extend([(0, 360), (70, 140)] + [(0, T_MAX)] * MAX_BOMBS_PER_UAV)
    
    bounds = np.array(bounds, dtype=np.float64)

    # 预先计算导弹路径和时间步长，并传入所有需要调用的函数
    time_steps = np.arange(0, T_MAX, DT, dtype=np.float64)
    missile_names = list(MISSILE_INITS.keys())
    missile_paths_cache = np.array([
        np.array([
            MISSILE_INITS[m] + (TARGET_CENTER - MISSILE_INITS[m]) / np.linalg.norm(TARGET_CENTER - MISSILE_INITS[m]) * MISSILE_SPEED * t
            for t in time_steps
        ], dtype=np.float64) for m in missile_names
    ], dtype=np.float64)

    # 将数据传输到GPU，以便在GPU内核中使用
    missile_paths_cache_gpu = cuda.to_device(missile_paths_cache)
    time_steps_gpu = cuda.to_device(time_steps)
    uav_inits_gpu = cuda.to_device(UAV_INITS_LIST)
    target_center_gpu = cuda.to_device(TARGET_CENTER)

    gpu_data = {
        'missile_paths_cache_gpu': missile_paths_cache_gpu,
        'time_steps_gpu': time_steps_gpu,
        'uav_inits_gpu': uav_inits_gpu,
        'target_center_gpu': target_center_gpu,
        'missile_paths_cache': missile_paths_cache,
        'time_steps': time_steps
    }
    
    def gpu_evaluate_wrapper(pop, gpu_data):
        """包装函数，用于在主程序中调用GPU内核。"""
        pop_gpu = cuda.to_device(pop)
        results_out = cuda.device_array(pop.shape[0], dtype=np.float64)
        
        # 定义GPU网格和块大小
        threads_per_block = 256
        blocks_per_grid = (pop.shape[0] + (threads_per_block - 1)) // threads_per_block
        
        final_evaluate_kernel[blocks_per_grid, threads_per_block](
            pop_gpu,
            gpu_data['missile_paths_cache_gpu'],
            gpu_data['time_steps_gpu'],
            gpu_data['uav_inits_gpu'],
            gpu_data['target_center_gpu'],
            len(MISSILE_INITS),
            len(time_steps),
            results_out
        )
        return results_out.copy_to_host()

    for run_num in range(1, NUM_RUNS + 1):
        print("="*40)
        print(f"开始第 {run_num}/{NUM_RUNS} 次混合GA-DE优化运行 (GPU加速)")
        print("="*40)

        history = _load_optimization_data(seed_file_path)
        historical_best_coverage = history.get('best_coverage', -1.0)
        last_tried_seed = history.get('last_tried_seed', -1)
        
        current_seed = last_tried_seed + 1
        np.random.seed(current_seed)
        random.seed(current_seed)
        
        print(f"开始使用新随机种子: {current_seed} 进行优化...")
        
        # --- 阶段1: 混合GA-DE全局搜索 ---
        print("=== 阶段1: 开始混合GA-DE全局搜索 ===")
        best_params, best_score = ga_de_hybrid_optimization(
            gpu_evaluate_wrapper,
            bounds,
            popsize=POP_SIZE,
            max_iter=MAX_ITER,
            F_start=F_START,
            F_end=F_END,
            CR_start=CR_START,
            CR_end=CR_END,
            gpu_data=gpu_data
        )
        
        rough_best_params = best_params

        # --- 阶段2: 精细优化 (混合策略) ---
        print("\n=== 阶段2: 开始精细优化 (多点局部精确抛光) ===")
        
        fine_polish_result = minimize(
            final_evaluate,
            x0=rough_best_params,
            args=(gpu_data['missile_paths_cache'], gpu_data['time_steps']),
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True}
        )
            
        final_best_params = fine_polish_result.x
        final_total_coverage = -fine_polish_result.fun

        print("\n=== 混合自适应优化后的最终结果 ===")
        print(f"最终总遮蔽时间: {final_total_coverage:.2f} s")

        # --- 比较并更新历史最佳结果 ---
        if final_total_coverage > historical_best_coverage:
            print("\n本次优化找到比历史最佳更好的结果！正在更新历史记录。")
            history['best_coverage'] = final_total_coverage
            history['best_seed'] = current_seed
            history['best_params'] = final_best_params
        else:
            print(f"\n本次结果 ({final_total_coverage:.2f} s) 不如历史最佳 ({historical_best_coverage:.2f} s)，沿用历史最佳参数。")
        
        history['last_tried_seed'] = current_seed
        _save_optimization_data(history, seed_file_path)

    return history['best_params'], history['best_coverage'], gpu_data['missile_paths_cache'], gpu_data['time_steps']

def run_full_pipeline(NUM_RUNS, POP_SIZE, MAX_ITER, F_START, F_END, CR_START, CR_END, NUM_LOCAL_ITERS, seed_file_path, excel_file_path):
    """
    运行完整的优化流程，包括全局优化和迭代局部优化。
    """
    
    best_params, best_coverage, missile_paths_cache, time_steps = run_optimization(
        NUM_RUNS, POP_SIZE, MAX_ITER, F_START, F_END, CR_START, CR_END, seed_file_path
    )

    if best_params is None:
        print("未找到任何最佳参数，无法进行迭代局部优化。")
        return
        
    print("\n\n" + "="*40)
    print("开始第三阶段：基于贡献度的迭代局部优化")
    print("="*40)
    
    current_best_params = best_params.copy()
    current_best_coverage = best_coverage

    for i in range(NUM_LOCAL_ITERS):
        print(f"\n--- 迭代局部优化 第 {i+1}/{NUM_LOCAL_ITERS} 轮 ---")
        
        detailed_coverage = _get_detailed_coverage(current_best_params, missile_paths_cache, time_steps)
        
        # 找出总有效遮蔽时间低于阈值的烟幕弹
        low_score_bombs = [
            d for d in detailed_coverage if d['coverage'] < 5.0
        ]
        
        if not low_score_bombs:
            print("没有发现低得分的烟幕弹，提前结束迭代。")
            break

        print(f"发现 {len(low_score_bombs)} 枚低得分烟幕弹，正在进行局部优化...")
        
        for bomb in low_score_bombs:
            uav_index = list(UAV_INITS.keys()).index(bomb['uav'])
            bomb_index = bomb['bomb_index']
            
            # 确定需要局部优化的参数索引（方向、速度和单个投放时间）
            start_idx = uav_index * NUM_PARAMS_PER_UAV
            localized_indices = np.array([start_idx, start_idx + 1, start_idx + 2 + bomb_index])
            localized_params_initial = current_best_params[localized_indices]
            
            localized_bounds = np.array(list(zip([0, 70, 0], [360, 140, T_MAX])))
            
            # 使用L-BFGS-B进行局部精确优化
            localized_result = minimize(
                fun=_localized_evaluate,
                x0=localized_params_initial,
                args=(current_best_params, localized_indices, missile_paths_cache, time_steps),
                method='L-BFGS-B',
                bounds=localized_bounds,
                options={'disp': False}
            )

            if localized_result.success:
                current_best_params[localized_indices] = localized_result.x
            
        new_total_coverage = -final_evaluate(current_best_params, missile_paths_cache, time_steps)
        
        if new_total_coverage > current_best_coverage:
            print(f"本轮迭代使总得分从 {current_best_coverage:.2f} 提升至 {new_total_coverage:.2f}。")
            current_best_coverage = new_total_coverage
            history = _load_optimization_data(seed_file_path)
            history['best_coverage'] = current_best_coverage
            history['best_params'] = current_best_params
            _save_optimization_data(history, seed_file_path)
        else:
            print(f"本轮迭代未能提升总得分，保持当前最佳。")
            
    print("\n\n" + "="*40)
    print("所有优化完成，以下为历史最佳结果")
    print("="*40)
    print(f"历史最佳总遮蔽时间: {current_best_coverage:.2f} s")
    _generate_excel_data(current_best_params, excel_file_path, missile_paths_cache, time_steps)


if __name__ == "__main__":
    
    # 优化超参数
    NUM_RUNS = 5        # 运行轮次
    POP_SIZE = 100      # 种群规模
    MAX_ITER = 1000     # 最大迭代次数
    F_START = 0.8       # DE的缩放因子初始值
    F_END = 0.2         # DE的缩放因子结束值
    CR_START = 0.9      # GA和DE的交叉概率初始值
    CR_END = 0.5        # GA和DE的交叉概率结束值
    NUM_LOCAL_ITERS = 3 # 局部迭代优化次数
    EXCEL_FILE_PATH = "Data/result3.xlsx"
    SEED_FILE_PATH = "Data/Q5/optimization_seed.pkl"

    run_full_pipeline(
        NUM_RUNS=NUM_RUNS,
        POP_SIZE=POP_SIZE,
        MAX_ITER=MAX_ITER,
        F_START=F_START,
        F_END=F_END,
        CR_START=CR_START,
        CR_END=CR_END,
        NUM_LOCAL_ITERS=NUM_LOCAL_ITERS,
        seed_file_path=SEED_FILE_PATH,
        excel_file_path=EXCEL_FILE_PATH
    )
