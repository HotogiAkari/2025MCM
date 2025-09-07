# =========================================================================
# 最终版 V11: 严格遵守无人机飞行策略不变约束的优化器 (单文件)
#
# 核心改变:
# 1. 优化变量直接包含每架无人机的 (v, θ) 和所有炸弹的 (t_drop, τ)。
# 2. 从根本上保证了每架无人机在所有投弹过程中都保持唯一的 (v, θ) 策略。
# 3. 采用单一的全局优化阶段，辅以简化的热力图引导增强。
# =========================================================================

import numpy as np
import pandas as pd
from numba import cuda, jit, float32
import math
from tqdm import tqdm
import pickle
import os
from scipy.optimize import differential_evolution

# =========================================================================
# 1. 常量定义 (与V10版相同)
# =========================================================================
print("1. 初始化常量...")
G = 9.8; DT = 0.02; MISSILE_SPEED = 300.0
SMOKE_RADIUS = 10.0; SMOKE_RADIUS_SQ = SMOKE_RADIUS**2; SMOKE_LIFE = 20.0; SMOKE_SINK = 3.0
V_UAV_MIN = 70.0; V_UAV_MAX = 140.0; TAU_MIN = 0.2
TGT_CYL_CENTER = np.array([0.0, 200.0, 0.0]); TGT_CYL_RADIUS = 7.0; TGT_CYL_HEIGHT = 10.0
FAKE_TARGET = np.array([0.0, 0.0, 0.0])
UAV_INITS = {'FY1': np.array([17800.0, 0.0, 1800.0]), 'FY2': np.array([12000.0, 1400.0, 1400.0]), 'FY3': np.array([6000.0, -3000.0, 700.0]), 'FY4': np.array([11000.0, 2000.0, 1800.0]), 'FY5': np.array([13000.0, -2000.0, 1300.0])}
MISSILE_INITS = {'M1': np.array([20000.0, 0.0, 2000.0]), 'M2': np.array([19000.0, 600.0, 2100.0]), 'M3': np.array([18000.0, -600.0, 1900.0])}
NUM_UAVS = 5; MAX_BOMBS_PER_UAV = 3
UAV_NAMES = list(UAV_INITS.keys()); MISSILE_NAMES = list(MISSILE_INITS.keys())
# 新增：定义每个无人机的参数数量
PARAMS_PER_UAV_FORWARD = 2 + MAX_BOMBS_PER_UAV * 2
TOTAL_PARAMS_FORWARD = NUM_UAVS * PARAMS_PER_UAV_FORWARD # 总参数数量 = 5 * (2 + 3*2) = 40

# =========================================================================
# 2. 核心模块: 物理反算与性能评估 (与V10版相同)
# =========================================================================
@jit(nopython=True)
def calculate_uav_params(uav_initial_pos, P_exp, t_exp):
    z_uav, z_exp = uav_initial_pos[2], P_exp[2]
    delta_z = z_uav - z_exp
    if delta_z < 0: return False, -1.0, -1.0, -1.0, -1.0
    sqrt_arg = 2 * delta_z / G
    if sqrt_arg < 0: return False, -1.0, -1.0, -1.0, -1.0
    tau = np.sqrt(sqrt_arg)
    if tau < TAU_MIN: return False, -1.0, -1.0, -1.0, -1.0
    t_drop = t_exp - tau
    if t_drop < 0: return False, -1.0, -1.0, -1.0, -1.0
    delta_p_horizontal = P_exp[:2] - uav_initial_pos[:2]
    dist_horizontal = np.linalg.norm(delta_p_horizontal)
    if t_exp < 1e-6: return False, -1.0, -1.0, -1.0, -1.0
    v = dist_horizontal / t_exp
    if not (V_UAV_MIN <= v <= V_UAV_MAX): return False, -1.0, -1.0, -1.0, -1.0
    theta_rad = np.arctan2(delta_p_horizontal[1], delta_p_horizontal[0])
    theta_deg = np.degrees(theta_rad)
    if theta_deg < 0: theta_deg += 360.0
    return True, v, theta_deg, t_drop, tau

@cuda.jit(device=True)
def is_fully_covered_gpu(missile_pos, smoke_center, target_rep_points):
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
def calculate_coverage_kernel(all_bomb_strategies, start_indices, num_bombs_per_individual, d_missile_paths, d_time_steps, d_target_rep_points, results):
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
            missile_pos_x, missile_pos_y, missile_pos_z = d_missile_paths[m_idx, t_idx, 0], d_missile_paths[m_idx, t_idx, 1], d_missile_paths[m_idx, t_idx, 2]
            is_blocked = False
            for i in range(num_bombs):
                t_exp, exp_x, exp_y, exp_z = individual_bombs[i, 0], individual_bombs[i, 1], individual_bombs[i, 2], individual_bombs[i, 3]
                if not (t_exp <= t <= t_exp + SMOKE_LIFE): continue
                smoke_center_z = exp_z - SMOKE_SINK * (t - t_exp)
                missile_pos = (missile_pos_x, missile_pos_y, missile_pos_z)
                smoke_center = (exp_x, exp_y, smoke_center_z)
                if is_fully_covered_gpu(missile_pos, smoke_center, d_target_rep_points):
                    is_blocked = True
                    break
            if is_blocked: covered_steps += 1
        total_coverage += covered_steps * DT
    results[idx] = total_coverage

@jit(nopython=True)
def is_fully_covered_cpu(missile_pos, smoke_center, target_rep_points):
    for i in range(target_rep_points.shape[0]):
        q = target_rep_points[i]
        mq_vec = q - missile_pos
        mq_norm_sq = np.sum(mq_vec**2)
        if mq_norm_sq < 1e-9: return False
        ms_vec = smoke_center - missile_pos
        t = np.dot(ms_vec.astype(np.float32), mq_vec.astype(np.float32)) / mq_norm_sq
        if not (0.0 <= t <= 1.0): return False
        closest_point = missile_pos + t * mq_vec
        dist_sq = np.sum((smoke_center - closest_point)**2)
        if dist_sq > SMOKE_RADIUS_SQ: return False
    return True

@jit(nopython=True)
def calculate_total_coverage_cpu(bomb_strategies, missile_paths, time_steps, target_rep_points):
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
                smoke_center = np.array([exp_x, exp_y, exp_z - SMOKE_SINK * (t - t_exp)], dtype=np.float32)
                if is_fully_covered_cpu(missile_pos, smoke_center, target_rep_points):
                    is_blocked = True
                    break
            if is_blocked: covered_steps += 1
        total_coverage += covered_steps * DT
    return total_coverage

# =========================================================================
# 3. 数据准备辅助函数 (与V10版相同)
# =========================================================================
def generate_target_rep_points():
    points = []
    thetas = np.linspace(0, 2 * np.pi, 6, endpoint=False, dtype=np.float32)
    heights = np.array([0, TGT_CYL_HEIGHT / 2, TGT_CYL_HEIGHT], dtype=np.float32)
    for h in heights:
        for theta in thetas:
            points.append([TGT_CYL_CENTER[0] + TGT_CYL_RADIUS * np.cos(theta), TGT_CYL_CENTER[1] + TGT_CYL_RADIUS * np.sin(theta), TGT_CYL_CENTER[2] + h])
    return np.array(points, dtype=np.float32)

# =========================================================================
# 4. 优化器模块 (新的正向模拟优化器)
# =========================================================================
def evaluate_population_forward(pop, context):
    """
    新的批量评估函数，基于严格约束的正向模拟。
    每个个体(X)包含所有无人机的 (v, θ, t_drop1, τ1, t_drop2, τ2, t_drop3, τ3)。
    """
    popsize = pop.shape[0]
    uav_pos_list = context['uav_pos_list']
    D_MISSILE_PATHS, D_TIME_STEPS, D_TARGET_REP_POINTS = \
        context['D_MISSILE_PATHS'], context['D_TIME_STEPS'], context['D_TARGET_REP_POINTS']
    
    all_bomb_strategies_list = []
    start_indices = np.zeros(popsize, dtype=np.int32)
    num_bombs_per_individual = np.zeros(popsize, dtype=np.int32)
    current_start_idx = 0

    for i in range(popsize):
        X_individual = pop[i]
        bombs_for_this_individual = []
        
        for uav_idx in range(NUM_UAVS):
            base_idx = uav_idx * PARAMS_PER_UAV_FORWARD
            
            # --- 1. 获取无人机固定飞行策略 ---
            v = X_individual[base_idx]
            theta_deg = X_individual[base_idx + 1]
            theta_rad = np.radians(theta_deg)
            dir_vec = np.array([np.cos(theta_rad), np.sin(theta_rad)])
            uav_initial_pos = uav_pos_list[uav_idx]

            # --- 2. 获取并修正炸弹投放参数 ---
            raw_drop_params = [] # (t_drop, tau, original_bomb_num)
            for j in range(MAX_BOMBS_PER_UAV):
                t_drop = X_individual[base_idx + 2 + 2*j]
                tau = X_individual[base_idx + 3 + 2*j]
                raw_drop_params.append((t_drop, tau, j)) # 记录原始炸弹编号
            
            # 排序以便于修正投放间隔
            sorted_drop_params = sorted(raw_drop_params, key=lambda item: item[0])
            
            last_t_drop = -1.0
            for t_drop, tau, original_bomb_num in sorted_drop_params:
                # 强制修正1秒间隔约束
                if t_drop < last_t_drop + 1.0:
                    t_drop = last_t_drop + 1.0
                
                # --- 3. 正向计算爆炸点和时间 ---
                # P_drop_horizontal = uav_initial_pos[:2] + dir_vec * v * t_drop
                # P_exp_horizontal = P_drop_horizontal + dir_vec * v * tau
                # P_exp_z = uav_initial_pos[2] - 0.5 * G * tau**2
                
                # 简化计算 P_exp
                P_exp_x = uav_initial_pos[0] + dir_vec[0] * v * (t_drop + tau)
                P_exp_y = uav_initial_pos[1] + dir_vec[1] * v * (t_drop + tau)
                P_exp_z = uav_initial_pos[2] - 0.5 * G * tau**2
                if P_exp_z < 0: P_exp_z = 0.0 # 确保不低于地面

                t_exp = t_drop + tau
                
                bombs_for_this_individual.append({
                    'uav_idx': uav_idx, 
                    'v': v, 'theta': theta_deg, 
                    't_drop': t_drop, 'tau': tau, 
                    'P_exp': np.array([P_exp_x, P_exp_y, P_exp_z], dtype=np.float32), 
                    't_exp': t_exp,
                    'original_bomb_num': original_bomb_num # 记录原始炸弹编号，方便后续报告
                })
                last_t_drop = t_drop
        
        # 将所有炸弹信息转换为GPU可接受的格式
        gpu_bomb_list = []
        for bomb_info in bombs_for_this_individual:
            gpu_bomb_list.append([
                bomb_info['t_exp'], 
                bomb_info['P_exp'][0], bomb_info['P_exp'][1], bomb_info['P_exp'][2]
            ])

        start_indices[i] = current_start_idx
        num_bombs = len(gpu_bomb_list)
        num_bombs_per_individual[i] = num_bombs
        if num_bombs > 0:
            all_bomb_strategies_list.extend(gpu_bomb_list)
            current_start_idx += num_bombs
            
    if not all_bomb_strategies_list: return np.zeros(popsize)

    d_all_bomb_strategies = cuda.to_device(np.array(all_bomb_strategies_list, dtype=np.float32))
    d_start_indices = cuda.to_device(start_indices)
    d_num_bombs = cuda.to_device(num_bombs_per_individual)
    d_results = cuda.device_array(popsize, dtype=np.float32)
    threads_per_block = 256
    blocks_per_grid = (popsize + threads_per_block - 1) // threads_per_block
    calculate_coverage_kernel[blocks_per_grid, threads_per_block](d_all_bomb_strategies, d_start_indices, d_num_bombs, D_MISSILE_PATHS, D_TIME_STEPS, D_TARGET_REP_POINTS, d_results)
    return d_results.copy_to_host()


def run_de_optimizer_forward(bounds, context, popsize=50, max_generations=100, F=0.8, CR=0.9, history_file='de_history.pkl', desc="DE优化"):
    """自定义的差分进化全局优化器 (使用正向模拟评估)"""
    dims = len(bounds)
    lower_bounds, upper_bounds = np.array(bounds)[:, 0], np.array(bounds)[:, 1]
    historical_best_vector = None
    historical_best_fitness = np.inf 
    if os.path.exists(history_file):
        try:
            with open(history_file, 'rb') as f:
                data = pickle.load(f)
                if data['vector'].shape[0] == dims:
                    historical_best_vector, historical_best_fitness = data['vector'], data['fitness']
        except Exception: pass
    pop = lower_bounds + np.random.rand(popsize, dims) * (upper_bounds - lower_bounds)
    if historical_best_vector is not None: pop[0] = historical_best_vector
    
    fitness = -evaluate_population_forward(pop, context)
    
    best_idx = np.argmin(fitness)
    best_vector, best_fitness = pop[best_idx].copy(), fitness[best_idx]
    if historical_best_vector is not None and historical_best_fitness < best_fitness:
        best_fitness, best_vector = historical_best_fitness, historical_best_vector.copy()

    for gen in tqdm(range(max_generations), desc=desc):
        trial_pop = np.zeros_like(pop)
        for i in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), lower_bounds, upper_bounds)
            cross_points = np.random.rand(dims) < CR
            if not np.any(cross_points): cross_points[np.random.randint(0, dims)] = True
            trial = np.where(cross_points, mutant, pop[i])
            trial_pop[i] = trial
        
        trial_fitness = -evaluate_population_forward(trial_pop, context)
        
        improvement_mask = trial_fitness < fitness
        pop[improvement_mask], fitness[improvement_mask] = trial_pop[improvement_mask], trial_fitness[improvement_mask]
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness, best_vector = fitness[current_best_idx], pop[current_best_idx].copy()
            tqdm.write(f"第 {gen+1} 代: {desc} 新最优覆盖率 = {-best_fitness:.2f} s")
            with open(history_file, 'wb') as f: pickle.dump({'vector': best_vector, 'fitness': best_fitness}, f)
    return best_vector, -best_fitness

# --- 辅助函数：从最优参数向量中解析出可读的炸弹信息列表 ---
def parse_solution_to_bombs_info(solution_vector, context):
    bombs_info_list = []
    uav_pos_list = context['uav_pos_list']
    
    PARAMS_PER_UAV = 2 + MAX_BOMBS_PER_UAV * 2
    
    for uav_idx in range(NUM_UAVS):
        base_idx = uav_idx * PARAMS_PER_UAV
        v, theta_deg = solution_vector[base_idx], solution_vector[base_idx + 1]
        theta_rad = np.radians(theta_deg)
        dir_vec = np.array([np.cos(theta_rad), np.sin(theta_rad)])
        uav_initial_pos = uav_pos_list[uav_idx]

        drop_times_raw = []
        for j in range(MAX_BOMBS_PER_UAV):
            t_drop = solution_vector[base_idx + 2 + 2*j]
            tau = solution_vector[base_idx + 3 + 2*j]
            drop_times_raw.append((t_drop, tau, j)) # 包含原始炸弹编号

        sorted_drop_params = sorted(drop_times_raw, key=lambda item: item[0])
        
        last_t_drop = -1.0
        for t_drop, tau, original_bomb_num in sorted_drop_params:
            if t_drop < last_t_drop + 1.0: t_drop = last_t_drop + 1.0
            
            P_exp_x = uav_initial_pos[0] + dir_vec[0] * v * (t_drop + tau)
            P_exp_y = uav_initial_pos[1] + dir_vec[1] * v * (t_drop + tau)
            P_exp_z = uav_initial_pos[2] - 0.5 * G * tau**2
            if P_exp_z < 0: P_exp_z = 0.0
            
            bombs_info_list.append({
                'uav_idx': uav_idx, 
                'v': v, 'theta': theta_deg, 
                't_drop': t_drop, 'tau': tau, 
                'P_exp': np.array([P_exp_x, P_exp_y, P_exp_z], dtype=np.float32), 
                'bomb_id': f"{UAV_NAMES[uav_idx]}-{original_bomb_num+1}", # 恢复原始bomb_id
                'missile_idx': -1 # 阶段一和二的炸弹没有明确分配目标，报告时设为-1
            })
            last_t_drop = t_drop
            
    return bombs_info_list

# =========================================================================
# 5. 报告生成模块
# =========================================================================
def generate_excel_report(final_bombs_info, final_coverage, context, filepath="Data/result3.xlsx"):
    """生成符合最终要求的、包含详细贡献度的Excel报告"""
    if not final_bombs_info:
        print("无法生成报告，因为没有找到有效的解决方案。")
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    report_data = []
    final_bombs_info.sort(key=lambda x: (x['uav_idx'], x['t_drop'])) # 按无人机和投放时间排序

    for info in final_bombs_info:
        uav_pos = context['uav_pos_list'][info['uav_idx']]
        dir_vec = np.array([np.cos(np.radians(info['theta'])), np.sin(np.radians(info['theta']))])
        p_drop_xy = uav_pos[:2] + dir_vec * info['v'] * info['t_drop']
        p_exp = info['P_exp']
        
        report_data.append({
            '无人机编号': context['UAV_NAMES'][info['uav_idx']],
            '无人机运动方向 (度)': info['theta'],
            '无人机运动速度 (m/s)': info['v'],
            '烟幕干扰弹编号': info['bomb_id'],
            '主要拦截目标': context['MISSILE_NAMES'][info['missile_idx']] if info['missile_idx'] != -1 else '增强投放', # 修复 '增强投放' 标签
            '投放时间(s)': info['t_drop'], # 临时列，用于计算贡献度
            '引信延迟(s)': info['tau'], # 临时列，用于计算贡献度
            '烟幕干扰弹投放点的x坐标 (m)': p_drop_xy[0],
            '烟幕干扰弹投放点的y坐标 (m)': p_drop_xy[1],
            '烟幕干扰弹投放点的z坐标 (m)': uav_pos[2],
            '烟幕干扰弹起爆点的x坐标 (m)': p_exp[0],
            '烟幕干扰弹起爆点的y坐标 (m)': p_exp[1],
            '烟幕干扰弹起爆点的z坐标 (m)': p_exp[2],
        })
    df = pd.DataFrame(report_data)
    
    if df.empty: # 如果没有有效炸弹，则生成空DataFrame
        print("警告：没有有效的烟幕弹策略可以报告。")
        df = pd.DataFrame(columns=['无人机编号', '无人机运动方向 (度)', '无人机运动速度 (m/s)', '烟幕干扰弹编号', '烟幕干扰弹投放点的x坐标 (m)', '烟幕干扰弹投放点的y坐标 (m)', '烟幕干扰弹投放点的z坐标 (m)', '烟幕干扰弹起爆点的x坐标 (m)', '烟幕干扰弹起爆点的y坐标 (m)', '烟幕干扰弹起爆点的z坐标 (m)', '总有效干扰时长 (s)', '干扰的导弹编号'])
    else:
        print("正在计算每个烟幕弹的详细贡献...")
        missile_paths, time_steps, target_rep_points = context['MISSILE_PATHS'], context['TIME_STEPS'], context['TARGET_REP_POINTS']
        total_contributions, interfered_missiles_list = [], []
        
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="分析炸弹贡献"):
            t_exp = row['投放时间(s)'] + row['引信延迟(s)']
            bomb_strategy = np.array([[t_exp, row['烟幕干扰弹起爆点的x坐标 (m)'], row['烟幕干扰弹起爆点的y坐标 (m)'], row['烟幕干扰弹起爆点的z坐标 (m)']]], dtype=np.float32)
            
            bomb_total_contrib, interfered = 0, []
            for m_idx, m_name in enumerate(MISSILE_NAMES):
                coverage = calculate_total_coverage_cpu(bomb_strategy, np.expand_dims(missile_paths[m_idx], axis=0), time_steps, target_rep_points)
                if coverage > 0.01: interfered.append(m_name)
                bomb_total_contrib += coverage
            
            total_contributions.append(bomb_total_contrib)
            interfered_missiles_list.append(', '.join(interfered) if interfered else '无')

        df['总有效干扰时长 (s)'] = total_contributions
        df['干扰的导弹编号'] = interfered_missiles_list
        df = df.drop(columns=['投放时间(s)', '引信延迟(s)']) # 移除临时列
    
    column_order = ['无人机编号', '无人机运动方向 (度)', '无人机运动速度 (m/s)', '烟幕干扰弹编号', '烟幕干扰弹投放点的x坐标 (m)', '烟幕干扰弹投放点的y坐标 (m)', '烟幕干扰弹投放点的z坐标 (m)', '烟幕干扰弹起爆点的x坐标 (m)', '烟幕干扰弹起爆点的y坐标 (m)', '烟幕干扰弹起爆点的z坐标 (m)', '总有效干扰时长 (s)', '干扰的导弹编号']
    df = df[column_order]

    summary = pd.DataFrame([{'总有效遮蔽时间(s)': final_coverage}])
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='详细投放策略', index=False)
        summary.to_excel(writer, sheet_name='总结', index=False)
    print(f"详细报告已保存到 {filepath}")
    
# =========================================================================
# 6. 主程序入口
# =========================================================================
if __name__ == '__main__':
    # --- 优化参数配置 ---
    POPSIZE = 400             # 种群大小 (广度)
    MAX_GENERATIONS = 400     # 迭代次数 (深度)
    F_FACTOR = 0.7            # DE 缩放因子
    CR_RATE = 0.9             # DE 交叉概率
    
    # --- 文件路径配置 ---
    DATA_DIR = "Data"; Q5_DIR = os.path.join(DATA_DIR, "Q5")
    EXCEL_FILEPATH = os.path.join(DATA_DIR, "result3.xlsx")
    HISTORY_FILE = os.path.join(Q5_DIR, "de_history_forward.pkl")
    # 自动创建目录
    os.makedirs(Q5_DIR, exist_ok=True)

    print("2. 准备全局上下文...")
    T_MAX = np.max([np.linalg.norm(FAKE_TARGET - pos) for pos in MISSILE_INITS.values()]) / MISSILE_SPEED + 5.0
    TIME_STEPS = np.arange(0.0, T_MAX, DT, dtype=np.float32)
    MISSILE_PATHS = np.zeros((len(MISSILE_INITS), len(TIME_STEPS), 3), dtype=np.float32)
    for i, pos in enumerate(MISSILE_INITS.values()):
        direction = (FAKE_TARGET - pos) / np.linalg.norm(FAKE_TARGET - pos)
        MISSILE_PATHS[i, :, :] = pos + direction * MISSILE_SPEED * TIME_STEPS[:, np.newaxis]
    TARGET_REP_POINTS = generate_target_rep_points()
    D_MISSILE_PATHS, D_TIME_STEPS, D_TARGET_REP_POINTS = cuda.to_device(MISSILE_PATHS), cuda.to_device(TIME_STEPS), cuda.to_device(TARGET_REP_POINTS)
    uav_pos_list = np.array(list(UAV_INITS.values()))
    
    CONTEXT = {'UAV_INITS': UAV_INITS, 'MISSILE_INITS': MISSILE_INITS, 'TGT_CYL_CENTER': TGT_CYL_CENTER, 'UAV_NAMES': UAV_NAMES, 'MISSILE_NAMES': MISSILE_NAMES, 'G': G, 'DT': DT, 'T_MAX': T_MAX, 'TIME_STEPS': TIME_STEPS, 'MISSILE_PATHS': MISSILE_PATHS, 'TARGET_REP_POINTS': TARGET_REP_POINTS, 'D_MISSILE_PATHS': D_MISSILE_PATHS, 'D_TIME_STEPS': D_TIME_STEPS, 'D_TARGET_REP_POINTS': D_TARGET_REP_POINTS, 'uav_pos_list': uav_pos_list, 'V_UAV_MIN': V_UAV_MIN, 'V_UAV_MAX': V_UAV_MAX, 'TAU_MIN': TAU_MIN}

    # --- 构建优化边界 ---
    bounds = []
    # 每架无人机的参数：v, theta, (t_drop1, tau1), (t_drop2, tau2), (t_drop3, tau3)
    for uav_idx in range(NUM_UAVS):
        bounds.extend([
            (V_UAV_MIN, V_UAV_MAX),      # v
            (0, 360)                   # theta
        ])
        for _ in range(MAX_BOMBS_PER_UAV):
            bounds.extend([
                (0.1, T_MAX - 5.0),     # t_drop (投放时间)
                (TAU_MIN, 15.0)        # tau (引信延迟)
            ])

    # --- 运行全局优化 ---
    print("\n开始全局优化")
    
    best_solution, best_coverage = run_de_optimizer_forward(
        bounds=bounds, context=CONTEXT,
        popsize=POPSIZE, max_generations=MAX_GENERATIONS,
        F=F_FACTOR, CR=CR_RATE, history_file=HISTORY_FILE, desc="全局优化"
    )
    
    print(f"\n--- 优化完成！最优覆盖率: {best_coverage:.2f} s ---")
    
    if best_solution is not None:
        # 这里，我们直接从 best_solution 解析出最终的炸弹信息列表
        # 因为正向模拟已经严格遵守了约束，不再需要额外的增强阶段
        final_bombs_info = parse_solution_to_bombs_info(best_solution, CONTEXT)
        
        print("\n--- 生成最终报告 ---")
        generate_excel_report(final_bombs_info, best_coverage, CONTEXT, filepath=EXCEL_FILEPATH)
    else:
        print("优化未找到任何有效的解决方案。")