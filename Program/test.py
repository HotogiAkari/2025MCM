# -*- coding: utf-8 -*-
"""
问题5：利用5架无人机，每架无人机至多投放3枚烟幕干扰弹，
实施对M1、M2、M3等3枚来袭导弹的干扰。
优化方法：多阶段混合优化，结合全局搜索与局部精炼。
"""

import numpy as np
import pandas as pd
import os
import pickle
from scipy.optimize import differential_evolution, minimize
from numba import jit
from tqdm import tqdm

# ======================
# 参数设置
# ======================

MISSILE_SPEED = 300.0   # 导弹速度 (m/s)
SMOKE_RADIUS = 10.0     # 烟幕有效半径
SMOKE_LIFE = 20.0       # 有效时长 (s)
SMOKE_SINK = 3.0        # 烟幕下沉速度 (m/s)
DT = 0.1                # 模拟时间步长
NUM_UAVS = 5            # 无人机数量
MAX_BOMBS_PER_UAV = 3   # 每架无人机最多投放的烟幕弹数量

# 目标（圆柱，近似中心点）
TARGET_CENTER = np.array([0.0, 200.0, 5.0])

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

# 最大仿真时间，以最远导弹M1的命中时间为准
T_MAX = np.linalg.norm(TARGET_CENTER - MISSILE_INITS['M1']) / MISSILE_SPEED
T_MAX = round(T_MAX, 1) + 1.0

# ======================
# 工具函数（使用numba加速）
# ======================

@jit(nopython=True)
def line_segment_point_dist(p1, p2, q):
    """点 q 到线段 p1-p2 的最小距离"""
    v = p2 - p1
    w = q - p1
    t = np.dot(w, v) / (np.dot(v, v) + 1e-9)
    t = max(0.0, min(t, 1.0))
    closest = p1 + t * v
    return np.linalg.norm(closest - q)

@jit(nopython=True)
def is_blocked(missile_pos, target_pos, smoke_center, R):
    """判断烟幕是否阻断导弹与目标之间的视线"""
    d = line_segment_point_dist(missile_pos, target_pos, smoke_center)
    return d <= R

@jit(nopython=True)
def missile_pos(t, M_init, target_pos=TARGET_CENTER):
    """导弹在时刻 t 的位置"""
    direction = (target_pos - M_init)
    direction = direction / np.linalg.norm(direction)
    return M_init + direction * MISSILE_SPEED * t

@jit(nopython=True)
def simulate_single_coverage(t_b, pos_b, missile_name, t_max=T_MAX, dt=DT):
    """计算单个烟幕弹对指定导弹的遮蔽时间"""
    covered_count = 0
    missile_init_pos = MISSILE_INITS[missile_name]
    for t in np.arange(0, t_max, dt):
        m_pos = missile_pos(t, missile_init_pos, TARGET_CENTER)
        if t < t_b or t > t_b + SMOKE_LIFE:
            blocked = False
        else:
            smoke_center = np.array([pos_b[0], pos_b[1], pos_b[2] - SMOKE_SINK * (t - t_b)])
            blocked = is_blocked(m_pos, TARGET_CENTER, smoke_center, SMOKE_RADIUS)
        if blocked:
            covered_count += 1
    return covered_count * dt

# ======================
# 优化目标函数
# ======================

def _parse_params(params):
    """解析一维参数列表为可读的投放策略字典"""
    uav_names = list(UAV_INITS.keys())
    strategy = {}
    for i in range(NUM_UAVS):
        start_idx = i * (2 + MAX_BOMBS_PER_UAV)
        theta_deg = params[start_idx]
        v = params[start_idx + 1]
        
        bombs = []
        for j in range(MAX_BOMBS_PER_UAV):
            t_drop = params[start_idx + 2 + j]
            if t_drop < 0:
                continue
            bombs.append(t_drop)
            
        strategy[uav_names[i]] = {
            'theta': theta_deg,
            'v': v,
            'drop_times': bombs
        }
    return strategy

def _check_constraints(params):
    """检查投放时间间隔约束"""
    strategy = _parse_params(params)
    for _, data in strategy.items():
        drop_times = sorted(data['drop_times'])
        for i in range(len(drop_times) - 1):
            if drop_times[i+1] - drop_times[i] < 1.0:
                return False
    return True

def rough_evaluate(params):
    """
    粗略优化目标函数：最大化所有烟幕弹的独立遮蔽时间总和。
    此阶段忽略烟幕弹之间的重叠。
    """
    strategy = _parse_params(params)
    
    if not _check_constraints(params):
        return 1e9 # 巨大的惩罚

    total_individual_coverage = 0
    missile_names = list(MISSILE_INITS.keys())
    
    for uav_name, data in strategy.items():
        uav_init_pos = UAV_INITS[uav_name]
        theta_rad = np.deg2rad(data['theta'])
        direction = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
        
        for t_drop in data['drop_times']:
            pos_b = uav_init_pos + direction * data['v'] * t_drop
            for missile_name in missile_names:
                coverage = simulate_single_coverage(t_drop, pos_b, missile_name)
                total_individual_coverage += coverage
                
    return -total_individual_coverage

def final_evaluate(params):
    """
    精细优化目标函数：最大化所有烟幕弹对所有导弹的实际总遮蔽时间。
    此阶段考虑所有烟幕弹对所有导弹的重叠效果。
    """
    if not _check_constraints(params):
        return 1e9
    
    strategy = _parse_params(params)

    total_coverage = 0
    missile_names = list(MISSILE_INITS.keys())
    
    for missile_name in missile_names:
        missile_init_pos = MISSILE_INITS[missile_name]
        
        all_bombs = []
        for uav_name, data in strategy.items():
            uav_init_pos = UAV_INITS[uav_name]
            theta_rad = np.deg2rad(data['theta'])
            direction = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
            for t_drop in data['drop_times']:
                pos_b = uav_init_pos + direction * data['v'] * t_drop
                all_bombs.append({'t_drop': t_drop, 'pos': pos_b})

        bombs_tuple = [(bomb['t_drop'], bomb['pos']) for bomb in all_bombs]
        
        covered_count = 0
        for t in np.arange(0, T_MAX, DT):
            m_pos = missile_pos(t, missile_init_pos, TARGET_CENTER)
            blocked = False
            for t_b, pos_b in bombs_tuple:
                if t < t_b or t > t_b + SMOKE_LIFE:
                    continue
                smoke_center = np.array([pos_b[0], pos_b[1], pos_b[2] - SMOKE_SINK * (t - t_b)])
                if is_blocked(m_pos, TARGET_CENTER, smoke_center, SMOKE_RADIUS):
                    blocked = True
                    break
            if blocked:
                covered_count += 1
        total_coverage += covered_count * DT

    return -total_coverage

# ======================
# 结果处理与保存
# ======================

def _generate_excel_data(best_params, excel_file_path):
    """生成用于Excel导出的数据列表和总结数据"""
    strategy = _parse_params(best_params)
    data_rows = []
    
    total_effective_coverage = 0
    summary_data = []

    missile_names = list(MISSILE_INITS.keys())
    
    for uav_name, uav_data in strategy.items():
        uav_init_pos = UAV_INITS[uav_name]
        theta_rad = np.deg2rad(uav_data['theta'])
        direction = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])

        for bomb_idx, t_drop in enumerate(uav_data['drop_times']):
            pos_b = uav_init_pos + direction * uav_data['v'] * t_drop
            
            bomb_total_coverage = 0
            for missile_name in missile_names:
                coverage = simulate_single_coverage(t_drop, pos_b, missile_name)
                
                data_rows.append({
                    '无人机名称': uav_name,
                    '导弹名称': missile_name,
                    '无人机运动方向 (度)': uav_data['theta'],
                    '无人机运动速度 (m/s)': uav_data['v'],
                    '烟幕干扰弹名称': f'Bomb-{bomb_idx + 1}',
                    '烟幕干扰弹投放点的x坐标 (m)': pos_b[0],
                    '烟幕干扰弹投放点的y坐标 (m)': pos_b[1],
                    '烟幕干扰弹投放点的z坐标 (m)': pos_b[2],
                    '有效遮蔽时间 (s)': coverage
                })
                bomb_total_coverage += coverage
            
    final_score = -final_evaluate(best_params)
    summary_data.append({'总有效遮蔽时间 (s)': final_score})

    writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')
    
    df_result = pd.DataFrame(data_rows)
    df_result.to_excel(writer, sheet_name='详细投放策略', index=False)
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_excel(writer, sheet_name='总结', index=False)
    
    writer.close()
    print(f"\n最终结果已保存到 {excel_file_path}")

def _load_optimization_data(filepath):
    """从文件中加载历史优化数据。"""
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

# ======================
# 主程序
# ======================

def run_optimization(num_runs=5, seed_file_path="Data/Q5/optimization_seed.pkl"):
    """
    运行多轮优化以寻找最佳烟幕弹投放方案。
    """
    
    # 参数边界
    bounds = []
    for _ in range(NUM_UAVS):
        bounds.extend([(0, 360), (70, 140)] + [(0, T_MAX)] * MAX_BOMBS_PER_UAV)
    
    for run_num in range(1, num_runs + 1):
        print("="*40)
        print(f"开始第 {run_num}/{num_runs} 次混合优化运行")
        print("="*40)

        history = _load_optimization_data(seed_file_path)
        historical_best_coverage = history.get('best_coverage', -1.0)
        last_tried_seed = history.get('last_tried_seed', -1)
        
        current_seed = last_tried_seed + 1
        
        print(f"开始使用新随机种子: {current_seed} 进行优化...")
        
        # --- 阶段1: 粗略优化 (最大化所有烟幕弹的独立贡献) ---
        print("=== 阶段1: 开始粗略优化 (全局搜索) ===")
        
        rough_result = differential_evolution(
            rough_evaluate,
            bounds,
            maxiter=1000,    
            popsize=100,
            polish=False,
            disp=True,
            seed=current_seed
        )
        
        rough_best_params = rough_result.x
        
        # --- 阶段2: 精细优化 (混合策略) ---
        print("\n=== 阶段2: 开始精细优化 (全局-局部混合策略) ===")
        
        # 2a: 使用差分进化进行全局精细搜索
        print("--- 子阶段 2a: 差分进化 (全局精细搜索) ---")
        fine_de_result = differential_evolution(
            final_evaluate,
            bounds,
            maxiter=500, 
            popsize=50,
            x0=rough_best_params,
            polish=False, # 关闭DE的内部polish
            disp=True,
            seed=current_seed
        )
        
        fine_de_params = fine_de_result.x
        
        # 2b: 从 DE 的结果出发，使用局部优化器进行精确抛光
        print("\n--- 子阶段 2b: L-BFGS-B (局部精确抛光) ---")
        fine_polish_result = minimize(
            final_evaluate,
            x0=fine_de_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True}
        )
        
        final_best_params = fine_polish_result.x
        final_total_coverage = -fine_polish_result.fun

        print("\n=== 混合优化后的最终结果 ===")
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

    return history['best_params'], history['best_coverage']

if __name__ == "__main__":
    NUM_RUNS_TO_RUN = 5 # 更改运行的次数
    SEED_FILE_PATH = "Data/Q5/optimization_seed.pkl"
    EXCEL_FILE_PATH = "result3.xlsx"

    best_params, best_coverage = run_optimization(num_runs=NUM_RUNS_TO_RUN, seed_file_path=SEED_FILE_PATH)
    
    print("\n\n" + "="*40)
    print("所有运行完成，以下为历史最佳结果")
    print("="*40)
    
    if best_params is not None:
        print(f"历史最佳总遮蔽时间: {best_coverage:.2f} s")
        _generate_excel_data(best_params, EXCEL_FILE_PATH)
    else:
        print("未找到历史最佳参数，请重新运行。")