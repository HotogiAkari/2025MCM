# -*- coding: utf-8 -*-
"""
单文件：问题5 三阶段混合优化（GPU 批量评估 + 自动分批 + CUDA 内核优化）
适配 GPU: RTX 4060 (8GB)
保持原始算法结构，仅提升性能与稳定性。
"""

import os
import math
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from numba import cuda, jit, float32, float64

# ======================
# 参数设置（与原程序一致）
# ======================

MISSILE_SPEED = 300.0   # m/s
SMOKE_RADIUS = 10.0     # m
SMOKE_LIFE = 20.0       # s
SMOKE_SINK = 3.0        # m/s
DT = 0.1                # s
TARGET_CENTER = np.array([0.0, 200.0, 5.0], dtype=np.float64)

NUM_UAVS = 5
MAX_BOMBS_PER_UAV = 3
NUM_BOMBS = NUM_UAVS * MAX_BOMBS_PER_UAV        # 15
NUM_PARAMS_PER_UAV = 2 + MAX_BOMBS_PER_UAV      # theta, v, t1,t2,t3
NUM_PARAMS = NUM_UAVS * NUM_PARAMS_PER_UAV      # 25

MISSILE_INITS = {
    'M1': np.array([20000.0, 0.0, 2000.0], dtype=np.float64),
    'M2': np.array([19000.0, 600.0, 2100.0], dtype=np.float64),
    'M3': np.array([18000.0, -600.0, 1900.0], dtype=np.float64)
}

UAV_INITS = {
    'FY1': np.array([17800.0, 0.0, 1800.0], dtype=np.float64),
    'FY2': np.array([12000.0, 1400.0, 1400.0], dtype=np.float64),
    'FY3': np.array([6000.0, -3000.0, 700.0], dtype=np.float64),
    'FY4': np.array([11000.0, 2000.0, 1800.0], dtype=np.float64),
    'FY5': np.array([13000.0, -2000.0, 1300.0], dtype=np.float64)
}
UAV_INITS_LIST = np.array(list(UAV_INITS.values()), dtype=np.float64)

T_MAX = np.linalg.norm(TARGET_CENTER - MISSILE_INITS['M1']) / MISSILE_SPEED
T_MAX = round(T_MAX, 1) + 1.0

# GPU 批量评估参数
THREADS_PER_BLOCK = 128       # 可调整为 256 做性能测试
MAX_BATCH_SIZE = 256          # 自动分批，避免显存峰值（4060 8GB 建议 128-512 之间调试）

# For CUDA local arrays: use literal constants (compilation-time constants)
# These literal ints must match the above constants
_NUM_PARAMS_LITERAL = NUM_PARAMS   # 25
_NUM_BOMBS_LITERAL = NUM_BOMBS     # 15
_POS_SIZE_LITERAL = NUM_BOMBS * 3  # 45

# ======================
# CPU 辅助函数（保留 double 精度）
# ======================

@jit(nopython=True)
def line_segment_point_dist(p1, p2, q):
    """
    numba 加速的点到线段最短距离（用于 CPU final_evaluate）
    """
    v0 = p2[0] - p1[0]
    v1 = p2[1] - p1[1]
    v2 = p2[2] - p1[2]
    w0 = q[0] - p1[0]
    w1 = q[1] - p1[1]
    w2 = q[2] - p1[2]
    dot = w0*v0 + w1*v1 + w2*v2
    vv = v0*v0 + v1*v1 + v2*v2 + 1e-9
    t = dot / vv
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = p1[0] + t * v0
    cy = p1[1] + t * v1
    cz = p1[2] + t * v2
    dx = cx - q[0]
    dy = cy - q[1]
    dz = cz - q[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def _parse_params(params):
    """
    将一维参数向量解析为策略字典（与原逻辑一致）
    """
    uav_names = list(UAV_INITS.keys())
    strategy = {}
    for i in range(NUM_UAVS):
        start_idx = i * NUM_PARAMS_PER_UAV
        theta_deg = float(params[start_idx])
        v = float(params[start_idx + 1])
        bombs = []
        for j in range(MAX_BOMBS_PER_UAV):
            t_drop = params[start_idx + 2 + j]
            if t_drop < 0:
                continue
            bombs.append(float(t_drop))
        strategy[uav_names[i]] = {'theta': theta_deg, 'v': v, 'drop_times': bombs}
    return strategy

def final_evaluate(params, missile_paths_cache, time_steps):
    """
    CPU 版精细评估（double 精度）——用于 SciPy minimize 与结果输出
    """
    strategy = _parse_params(params)
    total_coverage = 0.0

    # 预计算所有炸弹位置
    all_bombs = []
    for uav_name, data in strategy.items():
        uav_init_pos = UAV_INITS[uav_name]
        theta_rad = math.radians(data['theta'])
        direction = np.array([math.cos(theta_rad), math.sin(theta_rad), 0.0], dtype=np.float64)
        for t_drop in data['drop_times']:
            pos_b = uav_init_pos + direction * data['v'] * t_drop
            all_bombs.append((t_drop, pos_b))

    for path in missile_paths_cache:
        covered_count = 0
        for i, t in enumerate(time_steps):
            m_pos = path[i]
            blocked = False
            for t_b, pos_b in all_bombs:
                if t < t_b or t > t_b + SMOKE_LIFE:
                    continue
                smoke_center = np.array([pos_b[0], pos_b[1], pos_b[2] - SMOKE_SINK*(t - t_b)], dtype=np.float64)
                d = line_segment_point_dist(m_pos, TARGET_CENTER, smoke_center)
                if d <= SMOKE_RADIUS:
                    blocked = True
                    break
            if blocked:
                covered_count += 1
        total_coverage += covered_count * DT

    return -total_coverage

# ======================
# CUDA 内核（float32） - 每线程负责评估一个个体
# 使用 cuda.local.array（编译期常量大小）
# ======================

@cuda.jit
def final_evaluate_kernel(pop, missile_paths, time_steps, uav_inits, target_center, num_missiles, num_time_steps, results):
    """
    GPU 内核：每个线程处理一个个体（pop 的一行）。
    pop: shape (popsize, NUM_PARAMS) float32
    missile_paths: (num_missiles, num_time_steps, 3) float32
    time_steps: (num_time_steps,) float32
    uav_inits: (NUM_UAVS, 3) float32
    target_center: (3,) float32
    results: (popsize,) float32 (输出 -total_coverage)
    """
    idx = cuda.grid(1)
    popsize = pop.shape[0]
    if idx >= popsize:
        return

    # 将参数拷贝到局部数组（每线程）
    local_params = cuda.local.array(_NUM_PARAMS_LITERAL, dtype=float32)
    for i in range(_NUM_PARAMS_LITERAL):
        local_params[i] = pop[idx, i]

    # 局部数组保存每枚烟幕的 t_drop 和 position（展开为一维）
    t_drops = cuda.local.array(_NUM_BOMBS_LITERAL, dtype=float32)
    pos_flat = cuda.local.array(_POS_SIZE_LITERAL, dtype=float32)  # 3 * NUM_BOMBS

    bomb_counter = 0
    # 解析 UAV 参数并计算每个炸弹的初始投放位置（在投放时刻的位置）
    for u in range(NUM_UAVS):
        base = u * NUM_PARAMS_PER_UAV
        theta_deg = local_params[base + 0]
        v = local_params[base + 1]
        theta_rad = theta_deg * 0.017453292519943295  # pi/180
        dir_x = math.cos(theta_rad)
        dir_y = math.sin(theta_rad)
        ux = uav_inits[u, 0]
        uy = uav_inits[u, 1]
        uz = uav_inits[u, 2]
        # 三个投放时间（固定3个）
        for j in range(MAX_BOMBS_PER_UAV):
            tdrop = local_params[base + 2 + j]
            if tdrop >= 0.0 and bomb_counter < _NUM_BOMBS_LITERAL:
                t_drops[bomb_counter] = tdrop
                bbase = bomb_counter * 3
                pos_flat[bbase + 0] = ux + dir_x * v * tdrop
                pos_flat[bbase + 1] = uy + dir_y * v * tdrop
                pos_flat[bbase + 2] = uz + 0.0  # dir_z == 0, so stays uz
                bomb_counter += 1

    # target center
    tx = target_center[0]
    ty = target_center[1]
    tz = target_center[2]

    total_coverage = 0.0

    # 对每个导弹
    for mi in range(num_missiles):
        covered_count = 0
        # 时间步循环
        for ti in range(num_time_steps):
            t = time_steps[ti]  # float32
            # 导弹位置
            mx = missile_paths[mi, ti, 0]
            my = missile_paths[mi, ti, 1]
            mz = missile_paths[mi, ti, 2]

            blocked = False
            # 遍历每枚烟幕弹
            for b in range(bomb_counter):
                t_b = t_drops[b]
                # 判断时间有效性
                if t < t_b or t > t_b + SMOKE_LIFE:
                    continue
                bbase = b * 3
                sx = pos_flat[bbase + 0]
                sy = pos_flat[bbase + 1]
                sz0 = pos_flat[bbase + 2]
                # 实时下沉
                sz = sz0 - SMOKE_SINK * (t - t_b)

                # 计算点到导弹-目标线段距离（逐元素）
                v0 = tx - mx
                v1 = ty - my
                v2 = tz - mz
                w0 = sx - mx
                w1 = sy - my
                w2 = sz - mz

                dot_w_v = w0 * v0 + w1 * v1 + w2 * v2
                dot_v_v = v0 * v0 + v1 * v1 + v2 * v2 + 1e-9
                tval = dot_w_v / dot_v_v
                if tval < 0.0:
                    tval = 0.0
                elif tval > 1.0:
                    tval = 1.0

                cx = mx + tval * v0
                cy = my + tval * v1
                cz = mz + tval * v2

                dx = cx - sx
                dy = cy - sy
                dz = cz - sz
                if dx*dx + dy*dy + dz*dz <= SMOKE_RADIUS * SMOKE_RADIUS:
                    blocked = True
                    break

            if blocked:
                covered_count += 1

        total_coverage += covered_count * DT

    # 取负作为优化目标（最小化）
    results[idx] = -total_coverage

# ======================
# GPUEvaluator：管理 device arrays，自动分批执行 kernel
# ======================

class GPUEvaluator:
    def __init__(self, missile_paths_cache_f64, time_steps_f64, uav_inits_f64, target_center_f64):
        # 转为 float32 并拷贝到 device
        self.missile_paths = np.array(missile_paths_cache_f64, dtype=np.float32)
        self.time_steps = np.array(time_steps_f64, dtype=np.float32)
        self.uav_inits = np.array(uav_inits_f64, dtype=np.float32)
        self.target = np.array(target_center_f64, dtype=np.float32)

        # device copies
        self.dp_missile_paths = cuda.to_device(self.missile_paths)
        self.dp_time_steps = cuda.to_device(self.time_steps)
        self.dp_uav_inits = cuda.to_device(self.uav_inits)
        self.dp_target = cuda.to_device(self.target)

    def evaluate_batch(self, pop_f64):
        """
        自动分批评估：将 pop 分块传到 GPU，运行 kernel，汇总结果
        返回 double 精度的 fitness（与原程序接口保持一致）
        """
        pop_f32 = np.array(pop_f64, dtype=np.float32)
        popsize = pop_f32.shape[0]
        results = np.zeros(popsize, dtype=np.float64)

        # 按 MAX_BATCH_SIZE 分批
        for start in range(0, popsize, MAX_BATCH_SIZE):
            end = min(start + MAX_BATCH_SIZE, popsize)
            chunk = pop_f32[start:end].copy()
            dp_pop = cuda.to_device(chunk)
            dp_res = cuda.device_array(end - start, dtype=np.float32)

            blocks = (chunk.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            final_evaluate_kernel[blocks, THREADS_PER_BLOCK](
                dp_pop,
                self.dp_missile_paths,
                self.dp_time_steps,
                self.dp_uav_inits,
                self.dp_target,
                self.missile_paths.shape[0],
                self.time_steps.shape[0],
                dp_res
            )

            host_res = dp_res.copy_to_host()
            results[start:end] = host_res.astype(np.float64)

            # free device memory references for GC (helpful for long runs)
            del dp_pop, dp_res

        return results

# ======================
# 生成初始种群、约束修复等（保持原逻辑）
# ======================

def _generate_collaborative_strategy():
    uav_names = list(UAV_INITS.keys())
    missile_names = list(MISSILE_INITS.keys())

    assigned_uavs = set()
    assigned_missiles = set()
    uav_assignments = {}

    # 贪心分配
    while len(assigned_missiles) < len(missile_names) and len(assigned_uavs) < len(uav_names):
        min_dist = float('inf')
        best_uav, best_missile = None, None
        for uav in uav_names:
            if uav in assigned_uavs:
                continue
            for m in missile_names:
                if m in assigned_missiles:
                    continue
                dist = np.linalg.norm(UAV_INITS[uav] - MISSILE_INITS[m])
                if dist < min_dist:
                    min_dist = dist
                    best_uav = uav
                    best_missile = m
        if best_uav and best_missile:
            uav_assignments[best_uav] = best_missile
            assigned_uavs.add(best_uav)
            assigned_missiles.add(best_missile)

    remaining = [u for u in uav_names if u not in assigned_uavs]
    for u in remaining:
        closest = min(missile_names, key=lambda m: np.linalg.norm(UAV_INITS[u] - MISSILE_INITS[m]))
        uav_assignments[u] = closest

    params = []
    for u in uav_names:
        uav_init_pos = UAV_INITS[u]
        assigned_missile = uav_assignments.get(u)
        if assigned_missile:
            assigned_pos = MISSILE_INITS[assigned_missile]
            dir_vec = assigned_pos - uav_init_pos
            theta_rad = math.atan2(dir_vec[1], dir_vec[0])
            theta_deg = math.degrees(theta_rad)
            if theta_deg < 0:
                theta_deg += 360.0
            v = 105.0
            travel = np.linalg.norm(dir_vec) / v
            t_drops = sorted([travel, travel + 10.0, travel + 20.0])
            params.extend([theta_deg, v] + t_drops)
        else:
            params.extend([random.uniform(0,360), random.uniform(70,140)] + [random.uniform(0, T_MAX) for _ in range(MAX_BOMBS_PER_UAV)])
    return np.array(params, dtype=np.float64)

def _generate_hybrid_initial_population(popsize, bounds):
    initial = []
    for _ in range(int(popsize * 0.2)):
        initial.append(_generate_collaborative_strategy())
    for _ in range(popsize - len(initial)):
        initial.append(np.random.uniform(bounds[:,0], bounds[:,1]))
    return np.array(initial, dtype=np.float64)

def _repair_constraints(params, min_interval=1.0):
    rep = params.copy()
    for i in range(NUM_UAVS):
        start = i * NUM_PARAMS_PER_UAV + 2
        seg = rep[start:start + MAX_BOMBS_PER_UAV]
        idxs = np.argsort(seg)
        sorted_vals = seg[idxs].copy()
        for j in range(len(sorted_vals)-1):
            if sorted_vals[j+1] - sorted_vals[j] < min_interval:
                sorted_vals[j+1] = sorted_vals[j] + min_interval
        rep[start:start + MAX_BOMBS_PER_UAV][idxs] = sorted_vals
    return rep

# ======================
# 混合 GA-DE 优化（保留原算法结构），调用 GPU 评估
# ======================

def ga_de_hybrid_optimization(gpu_eval_func, bounds, popsize, max_iter, F_start, F_end, CR_start, CR_end, gpu_data, rng_seed=None):
    num_params = len(bounds)
    rng = np.random.RandomState(rng_seed)
    pop = _generate_hybrid_initial_population(popsize, bounds)
    pop = np.array(pop, dtype=np.float64)

    fitness = gpu_eval_func(pop)
    best_idx = np.argmin(fitness)
    best_solution = pop[best_idx].copy()

    for it in tqdm(range(max_iter), desc="混合GA-DE优化中"):
        F = F_start - (F_start - F_end) * (it / max_iter)
        CR = CR_start - (CR_start - CR_end) * (it / max_iter)

        new_pop = np.zeros_like(pop)
        for j in range(popsize):
            if random.random() < 0.5:
                # GA 风格（交叉 + 轻微变异）
                p1 = pop[rng.randint(popsize)]
                p2 = pop[rng.randint(popsize)]
                child = pop[j].copy()
                for k in range(num_params):
                    if random.random() < CR:
                        child[k] = 0.5*(p1[k] + p2[k])
                mutation_rate = 0.1
                for k in range(num_params):
                    if random.random() < mutation_rate:
                        child[k] += rng.normal(0,0.1) * (bounds[k][1] - bounds[k][0])
                new_pop[j] = np.clip(child, bounds[:,0], bounds[:,1])
            else:
                # DE 变异
                idxs = [idx for idx in range(popsize) if idx != j]
                a = pop[rng.choice(idxs)]
                b = pop[rng.choice(idxs)]
                c = pop[rng.choice(idxs)]
                mutant = a + F * (b - c)
                trial = pop[j].copy()
                cross = rng.rand(num_params) < CR
                trial[cross] = mutant[cross]
                new_pop[j] = np.clip(trial, bounds[:,0], bounds[:,1])

        # 约束修复
        for j in range(popsize):
            new_pop[j] = _repair_constraints(new_pop[j])

        new_fitness = gpu_eval_func(new_pop)

        # 精英保留
        for j in range(popsize):
            if new_fitness[j] < fitness[j]:
                pop[j] = new_pop[j]
                fitness[j] = new_fitness[j]

        current_best_idx = np.argmin(fitness)
        # 用 CPU final_evaluate 比较历史 best（保持原代码思路）
        if fitness[current_best_idx] < final_evaluate(best_solution, gpu_data['missile_paths_cache'], gpu_data['time_steps']):
            best_solution = pop[current_best_idx].copy()

    return best_solution, -final_evaluate(best_solution, gpu_data['missile_paths_cache'], gpu_data['time_steps'])

# ======================
# 结果输出 / 详细覆盖 / 本地化评估（保持原逻辑）
# ======================

def _generate_excel_data(best_params, excel_file_path, missile_paths_cache, time_steps):
    strategy = _parse_params(best_params)
    data_rows = []
    missile_names = list(MISSILE_INITS.keys())
    for uav_name, uav_data in strategy.items():
        uav_init_pos = UAV_INITS[uav_name]
        theta_rad = math.radians(uav_data['theta'])
        direction = np.array([math.cos(theta_rad), math.sin(theta_rad), 0.0])
        for bomb_idx, t_drop in enumerate(uav_data['drop_times']):
            if t_drop < 0:
                continue
            pos_b = uav_init_pos + direction * uav_data['v'] * t_drop
            total_bomb_coverage = 0.0
            interfered_missiles = []
            for missile_name, path in zip(missile_names, missile_paths_cache):
                covered_count = 0
                for i, t in enumerate(time_steps):
                    m_pos = path[i]
                    if t < t_drop or t > t_drop + SMOKE_LIFE:
                        continue
                    smoke_center = np.array([pos_b[0], pos_b[1], pos_b[2] - SMOKE_SINK*(t - t_drop)])
                    d = line_segment_point_dist(m_pos, TARGET_CENTER, smoke_center)
                    if d <= SMOKE_RADIUS:
                        covered_count += 1
                coverage = covered_count * DT
                if coverage > 0:
                    total_bomb_coverage += coverage
                    interfered_missiles.append(f"{missile_name}")
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
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='总结', index=False)
    writer.close()
    print(f"最终结果已保存到 {excel_file_path}")

def _load_optimization_data(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        print(f"未找到历史优化文件 {filepath}，将创建新文件。")
        return {'best_seed': None, 'best_coverage': -1.0, 'best_params': None, 'last_tried_seed': -1}

def _save_optimization_data(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"历史优化数据已成功保存到 {filepath}")

def _get_detailed_coverage(params, missile_paths_cache, time_steps):
    # CPU 计算每枚炸弹贡献（用于局部化优化决策，计算量较小）
    strategy = _parse_params(params)
    detailed = []
    missile_names = list(MISSILE_INITS.keys())
    for uav_name, uav_data in strategy.items():
        uav_init_pos = UAV_INITS[uav_name]
        theta_rad = math.radians(uav_data['theta'])
        direction = np.array([math.cos(theta_rad), math.sin(theta_rad), 0.0])
        for bomb_idx, t_drop in enumerate(uav_data['drop_times']):
            if t_drop < 0:
                continue
            pos_b = uav_init_pos + direction * uav_data['v'] * t_drop
            bomb_total = 0.0
            for path in missile_paths_cache:
                covered_count = 0
                for i, t in enumerate(time_steps):
                    m_pos = path[i]
                    if t < t_drop or t > t_drop + SMOKE_LIFE:
                        continue
                    smoke_center = np.array([pos_b[0], pos_b[1], pos_b[2] - SMOKE_SINK*(t - t_drop)])
                    d = line_segment_point_dist(m_pos, TARGET_CENTER, smoke_center)
                    if d <= SMOKE_RADIUS:
                        covered_count += 1
                bomb_total += covered_count * DT
            detailed.append({'uav': uav_name, 'bomb_index': bomb_idx, 'coverage': bomb_total})
    return detailed

def _localized_evaluate(localized_params, full_params_template, localized_indices, missile_paths_cache, time_steps):
    full = full_params_template.copy()
    full[localized_indices] = localized_params
    return final_evaluate(full, missile_paths_cache, time_steps)

# ======================
# 主流程：run_optimization、run_full_pipeline（保持原结构与行为）
# ======================

def run_optimization(NUM_RUNS, POP_SIZE, MAX_ITER, F_START, F_END, CR_START, CR_END, seed_file_path):
    # bounds
    bounds = []
    for _ in range(NUM_UAVS):
        bounds.extend([(0,360), (70,140)] + [(0, T_MAX)] * MAX_BOMBS_PER_UAV)
    bounds = np.array(bounds, dtype=np.float64)

    # 预计算导弹路径与时间步
    time_steps = np.arange(0.0, T_MAX, DT, dtype=np.float64)
    missile_names = list(MISSILE_INITS.keys())
    missile_paths_cache = np.array([
        np.array([
            MISSILE_INITS[m] + (TARGET_CENTER - MISSILE_INITS[m]) / np.linalg.norm(TARGET_CENTER - MISSILE_INITS[m]) * MISSILE_SPEED * t
            for t in time_steps
        ], dtype=np.float64) for m in missile_names
    ], dtype=np.float64)

    # 初始化 GPU evaluator（会把数据复制到 device）
    try:
        gpu_eval = GPUEvaluator(missile_paths_cache, time_steps, UAV_INITS_LIST, TARGET_CENTER)
        print("GPU evaluator 初始化成功（CUDA 可用）。")
    except Exception as e:
        raise RuntimeError("GPU evaluator 初始化失败，请检查 CUDA/Numba/驱动 环境；错误：" + str(e))

    def gpu_eval_wrapper(pop, gpu_data=None):
        return gpu_eval.evaluate_batch(pop)

    history = _load_optimization_data(seed_file_path)

    for run_num in range(1, NUM_RUNS + 1):
        print("="*40)
        print(f"开始第 {run_num}/{NUM_RUNS} 次混合GA-DE优化运行 (GPU加速)")
        print("="*40)

        historical_best_coverage = history.get('best_coverage', -1.0)
        last_tried_seed = history.get('last_tried_seed', -1)
        current_seed = last_tried_seed + 1
        np.random.seed(current_seed)
        random.seed(current_seed)

        print(f"开始使用新随机种子: {current_seed} 进行优化...")

        best_params, best_score = ga_de_hybrid_optimization(
            gpu_eval_wrapper,
            bounds,
            popsize=POP_SIZE,
            max_iter=MAX_ITER,
            F_start=F_START,
            F_end=F_END,
            CR_start=CR_START,
            CR_end=CR_END,
            gpu_data={'missile_paths_cache': missile_paths_cache, 'time_steps': time_steps},
            rng_seed=current_seed
        )

        rough_best_params = best_params

        # 阶段2：L-BFGS-B 精细优化（使用 CPU double 精度 final_evaluate）
        print("\n=== 阶段2: 开始精细优化 (L-BFGS-B) ===")
        fine_res = minimize(
            final_evaluate,
            x0=rough_best_params,
            args=(missile_paths_cache, time_steps),
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True}
        )

        final_best_params = fine_res.x
        final_total_coverage = -fine_res.fun
        print(f"\n=== 混合自适应优化后的最终结果 ===\n最终总遮蔽时间: {final_total_coverage:.2f} s")

        if final_total_coverage > historical_best_coverage:
            print("本次优化优于历史最佳，更新历史记录。")
            history['best_coverage'] = final_total_coverage
            history['best_seed'] = current_seed
            history['best_params'] = final_best_params
        else:
            print(f"本次结果 ({final_total_coverage:.2f}) 未超过历史最佳 ({historical_best_coverage:.2f})")

        history['last_tried_seed'] = current_seed
        _save_optimization_data(history, seed_file_path)

    return history['best_params'], history['best_coverage'], missile_paths_cache, time_steps

def run_full_pipeline(NUM_RUNS, POP_SIZE, MAX_ITER, F_START, F_END, CR_START, CR_END, NUM_LOCAL_ITERS, seed_file_path, excel_file_path):
    best_params, best_coverage, missile_paths_cache, time_steps = run_optimization(
        NUM_RUNS, POP_SIZE, MAX_ITER, F_START, F_END, CR_START, CR_END, seed_file_path
    )

    if best_params is None:
        print("未找到任何最佳参数，无法进行后续局部优化。")
        return

    print("\n\n" + "="*40)
    print("开始第三阶段：基于贡献度的迭代局部优化")
    print("="*40)

    current_best_params = best_params.copy()
    current_best_coverage = best_coverage

    for i in range(NUM_LOCAL_ITERS):
        print(f"\n--- 迭代局部优化 第 {i+1}/{NUM_LOCAL_ITERS} 轮 ---")
        detailed_coverage = _get_detailed_coverage(current_best_params, missile_paths_cache, time_steps)

        low_score_bombs = [d for d in detailed_coverage if d['coverage'] < 5.0]
        if not low_score_bombs:
            print("没有发现低得分烟幕弹，提前结束迭代。")
            break

        print(f"发现 {len(low_score_bombs)} 枚低得分烟幕弹，进行局部优化...")
        for bomb in low_score_bombs:
            uav_index = list(UAV_INITS.keys()).index(bomb['uav'])
            bomb_index = bomb['bomb_index']
            start_idx = uav_index * NUM_PARAMS_PER_UAV
            localized_indices = np.array([start_idx, start_idx + 1, start_idx + 2 + bomb_index], dtype=np.int64)
            localized_params_initial = current_best_params[localized_indices]
            localized_bounds = np.array(list(zip([0,70,0],[360,140,T_MAX])))
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
            print(f"本轮迭代提升: {current_best_coverage:.2f} -> {new_total_coverage:.2f}")
            current_best_coverage = new_total_coverage
            history = _load_optimization_data(seed_file_path)
            history['best_coverage'] = current_best_coverage
            history['best_params'] = current_best_params
            _save_optimization_data(history, seed_file_path)
        else:
            print("本轮迭代未能提升，保持当前最佳。")

    print("\n所有优化完成，写入 Excel ...")
    _generate_excel_data(current_best_params, excel_file_path, missile_paths_cache, time_steps)

# ======================
# 脚本入口（默认参数）
# ======================

if __name__ == "__main__":
    NUM_RUNS = 5            # 运行轮次
    POP_SIZE = 150          # 种群规模
    MAX_ITER = 1000         # 最大迭代次数
    F_START = 0.8           # DE的缩放因子初始值
    F_END = 0.2             # DE的缩放因子结束值
    CR_START = 0.9          # GA和DE的交叉概率初始值
    CR_END = 0.5            # GA和DE的交叉概率结束值
    NUM_LOCAL_ITERS = 20    # 局部迭代优化次数
    EXCEL_FILE_PATH = "Data/result3.xlsx"
    SEED_FILE_PATH = "Data/Q5/optimization_seed.pkl"

    # 保证 Data/ 目录存在
    os.makedirs(os.path.dirname(EXCEL_FILE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SEED_FILE_PATH), exist_ok=True)

    # 检查 CUDA 是否可用（若不可用，抛出错误）
    if not cuda.is_available():
        raise RuntimeError("CUDA 不可用：请检查 GPU 驱动和 Numba/ CUDA 配置。")

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