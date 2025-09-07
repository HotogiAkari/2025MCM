# =========================================================================
# 烟幕干扰弹投放策略优化 (混合逆向求解模型)
#
# 核心改变:
# 1. 优化变量重构为 (目标导弹, 方向, 投放时间)，维度更低，策略性更强。
# 2. 评估函数是一个复杂的混合过程，严格遵守“单一飞行策略”约束。
# 3. 使用 scipy.differential_evolution 并正确处理混合整数问题。
# 4. 采用 CUDA 进行大规模并行评估，显著提升优化速度。
# =========================================================================

import os
import numpy as np
import pandas as pd
from numba import cuda, jit
from tqdm import tqdm
from scipy.optimize import differential_evolution, OptimizeResult
from typing import Dict, List, Tuple

# =========================================================================
# 1. 常量定义
# =========================================================================
# (此部分无变化)
G, DT, MISSILE_SPEED = 9.8, 0.02, 300.0
SMOKE_RADIUS, SMOKE_RADIUS_SQ, SMOKE_LIFE, SMOKE_SINK = 10.0, 10.0**2, 20.0, 3.0
V_UAV_MIN, V_UAV_MAX, TAU_MIN, MIN_TIME_BETWEEN_DROPS = 70.0, 140.0, 0.2, 1.0
TGT_CYL_CENTER, TGT_CYL_RADIUS, TGT_CYL_HEIGHT = np.array([0.0, 200.0, 0.0]), 7.0, 10.0
UAV_INITS = {'FY1': np.array([17800.0, 0.0, 1800.0]), 'FY2': np.array([12000.0, 1400.0, 1400.0]), 'FY3': np.array([6000.0, -3000.0, 700.0]), 'FY4': np.array([11000.0, 2000.0, 1800.0]), 'FY5': np.array([13000.0, -2000.0, 1300.0])}
MISSILE_INITS = {'M1': np.array([20000.0, 0.0, 2000.0]), 'M2': np.array([19000.0, 600.0, 2100.0]), 'M3': np.array([18000.0, -600.0, 1900.0])}
NUM_UAVS, MAX_BOMBS_PER_UAV = len(UAV_INITS), 3
UAV_NAMES, MISSILE_NAMES = list(UAV_INITS.keys()), list(MISSILE_INITS.keys())
PARAMS_PER_UAV_HYBRID = 1 + 1 + MAX_BOMBS_PER_UAV

# =========================================================================
# 2. 核心计算模块 (GPU & CPU)
# (此部分无变化)
# =========================================================================
@cuda.jit(device=True)
def is_fully_covered_gpu(missile_pos: Tuple, smoke_center: Tuple, target_rep_points: np.ndarray) -> bool:
    for i in range(target_rep_points.shape[0]):
        q_x, q_y, q_z = target_rep_points[i, 0], target_rep_points[i, 1], target_rep_points[i, 2]
        mq_x, mq_y, mq_z = q_x - missile_pos[0], q_y - missile_pos[1], q_z - missile_pos[2]
        mq_norm_sq = mq_x**2 + mq_y**2 + mq_z**2
        if mq_norm_sq < 1e-9: return False
        ms_x, ms_y, ms_z = smoke_center[0] - missile_pos[0], smoke_center[1] - missile_pos[1], smoke_center[2] - missile_pos[2]
        dot_product = ms_x * mq_x + ms_y * mq_y + ms_z * mq_z
        t = dot_product / mq_norm_sq
        if not (0.0 <= t <= 1.0): return False
        closest_x, closest_y, closest_z = missile_pos[0] + t * mq_x, missile_pos[1] + t * mq_y, missile_pos[2] + t * mq_z
        dist_sq = (smoke_center[0] - closest_x)**2 + (smoke_center[1] - closest_y)**2 + (smoke_center[2] - closest_z)**2
        if dist_sq > SMOKE_RADIUS_SQ: return False
    return True

@cuda.jit
def calculate_coverage_kernel(all_bomb_strategies, start_indices, num_bombs_per_individual,
                              d_missile_paths, d_time_steps, d_target_rep_points, results):
    idx = cuda.grid(1)
    if idx >= results.shape[0]: return
    start, num_bombs = start_indices[idx], num_bombs_per_individual[idx]
    if num_bombs == 0:
        results[idx] = 0.0
        return
    individual_bombs = all_bomb_strategies[start : start + num_bombs]
    total_coverage = 0.0
    num_missiles = d_missile_paths.shape[0]
    num_time_steps = d_time_steps.shape[0]
    for m_idx in range(num_missiles):
        covered_steps = 0
        for t_idx in range(num_time_steps):
            t = d_time_steps[t_idx]
            missile_pos_tuple = (d_missile_paths[m_idx, t_idx, 0], d_missile_paths[m_idx, t_idx, 1], d_missile_paths[m_idx, t_idx, 2])
            is_blocked = False
            for i in range(num_bombs):
                t_exp, exp_x, exp_y, exp_z = individual_bombs[i, 0], individual_bombs[i, 1], individual_bombs[i, 2], individual_bombs[i, 3]
                if not (t_exp <= t <= t_exp + SMOKE_LIFE): continue
                smoke_center_z = exp_z - SMOKE_SINK * (t - t_exp)
                smoke_center_tuple = (exp_x, exp_y, smoke_center_z)
                if is_fully_covered_gpu(missile_pos_tuple, smoke_center_tuple, d_target_rep_points):
                    is_blocked = True
                    break
            if is_blocked: covered_steps += 1
        total_coverage += covered_steps * DT
    results[idx] = total_coverage

@jit(nopython=True)
def is_fully_covered_cpu(missile_pos: np.ndarray, smoke_center: np.ndarray, target_rep_points: np.ndarray) -> bool:
    for i in range(target_rep_points.shape[0]):
        q = target_rep_points[i]
        mq_vec = q - missile_pos
        mq_norm_sq = np.sum(mq_vec**2)
        if mq_norm_sq < 1e-9: return False
        ms_vec = smoke_center - missile_pos
        t = np.dot(ms_vec, mq_vec) / mq_norm_sq
        if not (0.0 <= t <= 1.0): return False
        closest_point = missile_pos + t * mq_vec
        dist_sq = np.sum((smoke_center - closest_point)**2)
        if dist_sq > SMOKE_RADIUS_SQ: return False
    return True

@jit(nopython=True)
def calculate_total_coverage_cpu(bomb_strategies: np.ndarray, missile_paths: np.ndarray, 
                                 time_steps: np.ndarray, target_rep_points: np.ndarray) -> float:
    total_coverage = 0.0
    if len(bomb_strategies) == 0: return 0.0
    num_missiles = missile_paths.shape[0]
    for m_idx in range(num_missiles):
        covered_steps = 0
        for t_idx in range(len(time_steps)):
            t = time_steps[t_idx]
            missile_pos = missile_paths[m_idx, t_idx]
            is_blocked = False
            for i in range(bomb_strategies.shape[0]):
                t_exp, exp_x, exp_y, exp_z = bomb_strategies[i]
                if not (t_exp <= t <= t_exp + SMOKE_LIFE): continue
                smoke_center = np.array([exp_x, exp_y, exp_z - SMOKE_SINK * (t - t_exp)])
                if is_fully_covered_cpu(missile_pos, smoke_center, target_rep_points):
                    is_blocked = True
                    break
            if is_blocked: covered_steps += 1
        total_coverage += covered_steps * DT
    return total_coverage

@jit(nopython=True)
def solve_interception(uav_pos: np.ndarray, uav_dir_vec: np.ndarray, t_drop: float, 
                       missile_start: np.ndarray, missile_dir_vec: np.ndarray, tgt_center: np.ndarray) -> Tuple:
    best_v, best_tau, best_t_exp = -1.0, -1.0, -1.0
    best_p_exp = np.zeros(3)
    min_v_diff = 1e9
    for t_exp_candidate in np.arange(t_drop + TAU_MIN, t_drop + SMOKE_LIFE, 0.1):
        tau = t_exp_candidate - t_drop
        if tau < TAU_MIN: continue
        missile_pos = missile_start + missile_dir_vec * MISSILE_SPEED * t_exp_candidate
        a, b, c_solve = uav_dir_vec, tgt_center[:2] - missile_pos[:2], missile_pos[:2] - uav_pos[:2]
        m11, m12, m21, m22 = t_exp_candidate * a[0], -b[0], t_exp_candidate * a[1], -b[1]
        det = m11 * m22 - m12 * m21
        if abs(det) < 1e-6: continue
        v_uav, lambda_val = (c_solve[0] * m22 - m12 * c_solve[1]) / det, (m11 * c_solve[1] - c_solve[0] * m21) / det
        if not (V_UAV_MIN <= v_uav <= V_UAV_MAX and 0.01 <= lambda_val <= 0.99): continue
        v_diff = abs(v_uav - (V_UAV_MIN + V_UAV_MAX) / 2)
        if v_diff < min_v_diff:
            min_v_diff = v_diff
            best_v, best_tau, best_t_exp = v_uav, tau, t_exp_candidate
            p_exp_z = uav_pos[2] - 0.5 * G * tau**2
            p_exp_xy = uav_pos[:2] + uav_dir_vec * best_v * t_exp_candidate
            best_p_exp = np.array([p_exp_xy[0], p_exp_xy[1], max(0.0, p_exp_z)])
    return best_v > 0, best_v, best_tau, best_p_exp, best_t_exp

# =========================================================================
# 3. 数据准备辅助函数
# (此部分无变化)
# =========================================================================
def generate_target_rep_points() -> np.ndarray:
    points = []
    thetas = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    heights = np.array([0, TGT_CYL_HEIGHT / 2, TGT_CYL_HEIGHT])
    for h in heights:
        for theta in thetas:
            points.append([TGT_CYL_CENTER[0] + TGT_CYL_RADIUS * np.cos(theta), TGT_CYL_CENTER[1] + TGT_CYL_RADIUS * np.sin(theta), TGT_CYL_CENTER[2] + h])
    return np.array(points, dtype=np.float32)

def _setup_optimizer_bounds() -> Tuple[List[Tuple], np.ndarray]:
    bounds, integrality_mask = [], []
    t_max_for_bounds = np.max([np.linalg.norm(TGT_CYL_CENTER - pos) for pos in MISSILE_INITS.values()]) / MISSILE_SPEED + 5.0
    for _ in range(NUM_UAVS):
        bounds.append((0, len(MISSILE_NAMES) - 0.01))
        integrality_mask.append(True)
        bounds.append((0, 360))
        integrality_mask.append(False)
        for _ in range(MAX_BOMBS_PER_UAV):
            bounds.append((0.1, t_max_for_bounds - 5.0))
            integrality_mask.append(False)
    return bounds, np.array(integrality_mask)

# =========================================================================
# 4. 优化器模块
# =========================================================================
BOUNDS, INTEGRALITY_MASK = _setup_optimizer_bounds()

def objective_wrapper(population: np.ndarray, context: Dict) -> np.ndarray:
    if population.ndim == 1:
        population = population.reshape(1, -1)
    population[:, INTEGRALITY_MASK] = np.round(population[:, INTEGRALITY_MASK])
    coverage_times = evaluate_population_hybrid(population, context)
    fitness_values = -coverage_times
    return fitness_values[0] if population.shape[0] == 1 else fitness_values

def evaluate_population_hybrid(population: np.ndarray, context: Dict) -> np.ndarray:
    D_MISSILE_PATHS = cuda.to_device(context['MISSILE_PATHS'])
    D_TIME_STEPS = cuda.to_device(context['TIME_STEPS'])
    D_TARGET_REP_POINTS = cuda.to_device(context['TARGET_REP_POINTS'])
    popsize = population.shape[0]
    uav_pos_list = context['uav_pos_list']
    missile_starts, missile_dirs = context['missile_starts'], context['missile_dirs']
    num_missiles = len(missile_starts)

    all_bomb_strategies_list, start_indices, num_bombs_per_individual = [], np.zeros(popsize, dtype=np.int32), np.zeros(popsize, dtype=np.int32)
    current_start_idx = 0

    for i in range(popsize):
        individual = population[i]
        bombs_for_this_individual = []
        for uav_idx in range(NUM_UAVS):
            base_idx = uav_idx * PARAMS_PER_UAV_HYBRID
            
            # [FIX] 使用 np.clip 确保索引安全
            raw_idx = individual[base_idx]
            target_missile_idx = int(np.clip(raw_idx, 0, num_missiles - 1))
            
            theta_deg = individual[base_idx + 1]
            raw_drop_times = sorted([individual[base_idx + 2 + j] for j in range(MAX_BOMBS_PER_UAV)])
            solved_velocities, solved_bomb_params = [], []
            last_t_drop = -1.0
            for t_drop in raw_drop_times:
                if t_drop < last_t_drop + MIN_TIME_BETWEEN_DROPS: t_drop = last_t_drop + MIN_TIME_BETWEEN_DROPS
                uav_dir_vec = np.array([np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))])
                is_feasible, v, tau, _, t_exp = solve_interception(
                    uav_pos_list[uav_idx], uav_dir_vec, t_drop, 
                    missile_starts[target_missile_idx], missile_dirs[target_missile_idx], TGT_CYL_CENTER
                )
                if is_feasible:
                    solved_velocities.append(v)
                    solved_bomb_params.append({'t_drop': t_drop, 'tau': tau, 't_exp': t_exp, 'theta': theta_deg})
                last_t_drop = t_drop
            if not solved_velocities: continue
            avg_v = np.mean(solved_velocities)
            if not (V_UAV_MIN <= avg_v <= V_UAV_MAX): continue
            for bomb in solved_bomb_params:
                dir_vec = np.array([np.cos(np.radians(bomb['theta'])), np.sin(np.radians(bomb['theta']))])
                uav_pos = uav_pos_list[uav_idx]
                p_exp_xy = uav_pos[:2] + dir_vec * avg_v * bomb['t_exp']
                p_exp_z = uav_pos[2] - 0.5 * G * bomb['tau']**2
                bombs_for_this_individual.append([bomb['t_exp'], p_exp_xy[0], p_exp_xy[1], max(0.0, p_exp_z)])

        start_indices[i], num_bombs_per_individual[i] = current_start_idx, len(bombs_for_this_individual)
        if num_bombs_per_individual[i] > 0:
            all_bomb_strategies_list.extend(bombs_for_this_individual)
            current_start_idx += num_bombs_per_individual[i]
            
    if not all_bomb_strategies_list:
        return np.zeros(popsize)

    d_all_bomb_strategies = cuda.to_device(np.array(all_bomb_strategies_list, dtype=np.float32))
    d_start_indices = cuda.to_device(start_indices)
    d_num_bombs = cuda.to_device(num_bombs_per_individual)
    d_results = cuda.device_array(popsize, dtype=np.float32)
    threads_per_block = 256
    blocks_per_grid = (popsize + threads_per_block - 1) // threads_per_block
    calculate_coverage_kernel[blocks_per_grid, threads_per_block](d_all_bomb_strategies, d_start_indices, d_num_bombs, D_MISSILE_PATHS, D_TIME_STEPS, D_TARGET_REP_POINTS, d_results)
    return d_results.copy_to_host()

# =========================================================================
# 5. 报告生成模块
# (此部分无变化)
# =========================================================================
def generate_excel_report(best_solution: np.ndarray, final_coverage: float, context: Dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print("\n--- 1. 解码最优解并重新计算详细参数 ---")
    final_bombs_info = []
    num_missiles = len(context['missile_starts'])
    for uav_idx in range(NUM_UAVS):
        base_idx = uav_idx * PARAMS_PER_UAV_HYBRID
        raw_idx = best_solution[base_idx]
        target_missile_idx = int(np.clip(raw_idx, 0, num_missiles - 1))
        theta_deg = best_solution[base_idx + 1]
        raw_drop_times = sorted([best_solution[base_idx + 2 + j] for j in range(MAX_BOMBS_PER_UAV)])
        solved_velocities, solved_bombs = [], []
        last_t_drop = -1.0
        for t_drop in raw_drop_times:
            if t_drop < last_t_drop + MIN_TIME_BETWEEN_DROPS: t_drop = last_t_drop + MIN_TIME_BETWEEN_DROPS
            uav_dir_vec = np.array([np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))])
            is_feasible, v, tau, _, t_exp = solve_interception(
                context['uav_pos_list'][uav_idx], uav_dir_vec, t_drop, 
                context['missile_starts'][target_missile_idx], context['missile_dirs'][target_missile_idx], TGT_CYL_CENTER
            )
            if is_feasible:
                solved_velocities.append(v)
                solved_bombs.append({'theta': theta_deg, 't_drop': t_drop, 'tau': tau, 't_exp': t_exp, 'missile_idx': target_missile_idx})
            last_t_drop = t_drop
        if not solved_velocities: continue
        avg_v = np.mean(solved_velocities)
        if not (V_UAV_MIN <= avg_v <= V_UAV_MAX): continue
        for i, bomb in enumerate(solved_bombs):
            bomb['v'], bomb['uav_idx'], bomb['bomb_id'] = avg_v, uav_idx, f"{UAV_NAMES[uav_idx]}-{i+1}"
            final_bombs_info.append(bomb)
    if not final_bombs_info:
        print("优化找到了一个解，但在最终解析后没有发现可行的炸弹投放策略。")
        return

    print("--- 2. 构建详细策略DataFrame ---")
    report_data = []
    for row in final_bombs_info:
        uav_pos, dir_vec = context['uav_pos_list'][row['uav_idx']], np.array([np.cos(np.radians(row['theta'])), np.sin(np.radians(row['theta']))])
        p_drop_xy, p_exp_xy = uav_pos[:2] + dir_vec * row['v'] * row['t_drop'], uav_pos[:2] + dir_vec * row['v'] * row['t_exp']
        p_exp_z = uav_pos[2] - 0.5 * G * row['tau']**2
        report_data.append({'无人机编号': context['UAV_NAMES'][row['uav_idx']], '无人机运动方向 (度)': row['theta'], '无人机运动速度 (m/s)': row['v'], '烟幕干扰弹编号': row['bomb_id'], '主要拦截目标': context['MISSILE_NAMES'][row['missile_idx']], '投放时间(s)': row['t_drop'], '引信延迟(s)': row['tau'], '投放点_x (m)': p_drop_xy[0], '投放点_y (m)': p_drop_xy[1], '投放点_z (m)': uav_pos[2], '起爆点_x (m)': p_exp_xy[0], '起爆点_y (m)': p_exp_xy[1], '起爆点_z (m)': max(0.0, p_exp_z)})
    df_report = pd.DataFrame(report_data)

    print("--- 3. 计算每个烟幕弹的独立贡献 (CPU) ---")
    contributions, interfered_missiles_list = [], []
    for _, row in tqdm(df_report.iterrows(), total=df_report.shape[0], desc="分析炸弹贡献"):
        t_exp = row['投放时间(s)'] + row['引信延迟(s)']
        bomb_strategy = np.array([[t_exp, row['起爆点_x (m)'], row['起爆点_y (m)'], row['起爆点_z (m)']]], dtype=np.float32)
        bomb_total_contrib, interfered = 0, []
        for m_idx, m_name in enumerate(context['MISSILE_NAMES']):
            single_missile_path = np.expand_dims(context['MISSILE_PATHS'][m_idx], axis=0)
            coverage = calculate_total_coverage_cpu(bomb_strategy, single_missile_path, context['TIME_STEPS'], context['TARGET_REP_POINTS'])
            if coverage > 0.01: interfered.append(m_name)
            bomb_total_contrib += coverage
        contributions.append(bomb_total_contrib)
        interfered_missiles_list.append(', '.join(interfered) if interfered else '无')
    df_report['单弹总有效干扰时长 (s)'], df_report['干扰的导弹编号'] = contributions, interfered_missiles_list
    
    print("--- 4. 格式化并保存Excel报告 ---")
    df_report.drop(columns=['投放时间(s)', '引信延迟(s)'], inplace=True)
    column_order = ['无人机编号', '无人机运动方向 (度)', '无人机运动速度 (m/s)', '烟幕干扰弹编号', '主要拦截目标', '投放点_x (m)', '投放点_y (m)', '投放点_z (m)', '起爆点_x (m)', '起爆点_y (m)', '起爆点_z (m)', '单弹总有效干扰时长 (s)', '干扰的导弹编号']
    df_report = df_report[column_order]
    summary = pd.DataFrame([{'总有效遮蔽时间(s)': final_coverage}])
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df_report.to_excel(writer, sheet_name='详细投放策略', index=False)
        summary.to_excel(writer, sheet_name='总结', index=False)
    print(f"详细报告已成功保存到: {filepath}")

# =========================================================================
# 6. 主程序入口
# =========================================================================
def main():
    """主执行函数"""
    
    ### ================================= ###
    ###    优化器超参数 (可在此处调整)    ###
    ### ================================= ###
    POPSIZE = 150 
    MAX_GENERATIONS = 200
    F_FACTOR = 0.8
    CR_RATE = 0.9
    EXCEL_FILEPATH = os.path.join("Data", "result_hybrid_model.xlsx")

    # 1. 初始化
    print("1. 初始化...")
    os.makedirs(os.path.dirname(EXCEL_FILEPATH), exist_ok=True)
    
    # 2. 准备上下文
    print("2. 准备全局上下文 (预计算导弹路径等)...")
    T_MAX = np.max([np.linalg.norm(TGT_CYL_CENTER - pos) for pos in MISSILE_INITS.values()]) / MISSILE_SPEED + 5.0
    TIME_STEPS = np.arange(0.0, T_MAX, DT, dtype=np.float32)
    MISSILE_PATHS = np.zeros((len(MISSILE_INITS), len(TIME_STEPS), 3), dtype=np.float32)
    missile_starts, missile_dirs = [], []
    for i, pos in enumerate(MISSILE_INITS.values()):
        direction = (TGT_CYL_CENTER - pos) / np.linalg.norm(TGT_CYL_CENTER - pos)
        MISSILE_PATHS[i, :, :] = pos + direction * MISSILE_SPEED * TIME_STEPS[:, np.newaxis]
        missile_starts.append(pos)
        missile_dirs.append(direction)
    TARGET_REP_POINTS = generate_target_rep_points()
    CONTEXT = {
        'uav_pos_list': np.array(list(UAV_INITS.values())),
        'missile_starts': missile_starts, 'missile_dirs': missile_dirs,
        'MISSILE_PATHS': MISSILE_PATHS, 'TIME_STEPS': TIME_STEPS, 'TARGET_REP_POINTS': TARGET_REP_POINTS,
        'TGT_CYL_CENTER': TGT_CYL_CENTER,
        'UAV_NAMES': UAV_NAMES, 'MISSILE_NAMES': MISSILE_NAMES
    }

    # 3. 运行差分进化算法
    print("\n--- 开始混合逆向全局优化 (使用多进程并行) ---")
    
    # [NEW] 使用tqdm创建进度条
    with tqdm(total=MAX_GENERATIONS, desc="优化进度") as pbar:
        def callback(xk, convergence):
            # xk 是当前最优的参数向量
            # convergence 是种群的收敛性度量
            pbar.update(1)
            # 在进度条后显示当前最优适应度（总遮蔽时间）
            # 注意：这里需要重新计算最优值，因为callback中没有直接提供
            best_fitness = -objective_wrapper(xk, CONTEXT)
            pbar.set_postfix(best_coverage=f"{best_fitness:.2f} s")

        result = differential_evolution(
            func=objective_wrapper,
            bounds=BOUNDS,
            args=(CONTEXT,),
            strategy='best1bin',
            maxiter=MAX_GENERATIONS,
            popsize=POPSIZE,
            mutation=(F_FACTOR, 1.0),
            recombination=CR_RATE,
            disp=False, # 关闭scipy自带的输出，使用我们的tqdm
            workers=-1,
            callback=callback # 传入回调函数
        )
    
    # 4. 处理并报告结果
    print(f"\n--- 优化完成！最优总遮蔽时间: {-result.fun:.2f} s ---")
    if result.x is not None:
        final_solution = np.copy(result.x)
        final_solution[INTEGRALITY_MASK] = np.round(final_solution[INTEGRALITY_MASK])
        generate_excel_report(final_solution, -result.fun, CONTEXT, filepath=EXCEL_FILEPATH)
    else:
        print("优化未找到任何有效的解决方案。")

if __name__ == '__main__':
    main()