# -*- coding: utf-8 -*-
"""
问题3：使用FY1投放3枚烟幕弹干扰M1
优化方法：差分进化搜索最佳航向、速度和投放时机
新增功能：
1. 保存产生最优解的随机种子，以便复现。
2. 采用“先粗略后精细”的三阶段优化策略。
3. 调整Excel输出格式，使其更清晰。
"""

import numpy as np
import pandas as pd
import os
import pickle
from scipy.optimize import differential_evolution
from tqdm import tqdm
import sys

# ======================
# 参数设置
# ======================

MISSILE_SPEED = 300.0   # 导弹速度 (m/s)
SMOKE_RADIUS = 10.0     # 烟幕有效半径
SMOKE_LIFE = 20.0       # 有效时长 (s)
SMOKE_SINK = 3.0        # 烟幕下沉速度 (m/s)
DT = 0.1                # 模拟时间步长
NUM_BOMBS = 3           # 烟幕弹数量

# 目标（圆柱，近似中心点）
TARGET_CENTER = np.array([0.0, 200.0, 5.0])

# 导弹初始位置 (M1)
MISSILE_INIT = np.array([20000.0, 0.0, 2000.0])

# 无人机 FY1 初始位置
UAV_INIT = np.array([17800.0, 0.0, 1800.0])

# 最大仿真时间，根据导弹命中目标时间确定
T_MAX = np.linalg.norm(TARGET_CENTER - MISSILE_INIT) / MISSILE_SPEED
T_MAX = round(T_MAX, 1) + 1.0 # 确保模拟时间足够

# ======================
# 工具函数
# ======================

def line_segment_point_dist(p1, p2, q):
    """点 q 到线段 p1-p2 的最小距离"""
    v = p2 - p1
    w = q - p1
    t = np.dot(w, v) / (np.dot(v, v) + 1e-9)
    t = np.clip(t, 0.0, 1.0)
    closest = p1 + t * v
    return np.linalg.norm(closest - q)


def is_blocked(missile_pos, target_pos, smoke_center, R):
    """判断烟幕是否阻断导弹与目标之间的视线"""
    d = line_segment_point_dist(missile_pos, target_pos, smoke_center)
    return d <= R


def missile_pos(t, M0=MISSILE_INIT, target=TARGET_CENTER):
    """导弹在时刻 t 的位置"""
    direction = (target - M0)
    direction = direction / np.linalg.norm(direction)
    return M0 + direction * MISSILE_SPEED * t


def simulate_coverage(bombs, t_max=T_MAX, dt=DT):
    """
    计算给定爆炸点序列的总遮蔽时间
    bombs: [(t_b, pos), ...]
    """
    covered = []
    for t in np.arange(0, t_max, dt):
        m_pos = missile_pos(t)
        blocked = False
        for (t_b, pos_b) in bombs:
            if t < t_b or t > t_b + SMOKE_LIFE:
                continue
            smoke_center = np.array([pos_b[0], pos_b[1], pos_b[2] - SMOKE_SINK * (t - t_b)])
            if is_blocked(m_pos, TARGET_CENTER, smoke_center, SMOKE_RADIUS):
                blocked = True
                break
        covered.append(blocked)
    return np.sum(covered) * dt


def simulate_single_coverage(t_b, pos_b, t_max=T_MAX, dt=DT):
    """计算单个烟幕弹的遮蔽时间"""
    covered = []
    for t in np.arange(0, t_max, dt):
        m_pos = missile_pos(t)
        if t < t_b or t > t_b + SMOKE_LIFE:
            blocked = False
        else:
            smoke_center = np.array([pos_b[0], pos_b[1], pos_b[2] - SMOKE_SINK * (t - t_b)])
            blocked = is_blocked(m_pos, TARGET_CENTER, smoke_center, SMOKE_RADIUS)
        covered.append(blocked)
    return np.sum(covered) * dt


def get_coverage_from_params(params, num_bombs=NUM_BOMBS):
    """计算给定参数的总遮蔽时间和每个烟幕弹的贡献。"""
    theta_deg, v = params[:2]
    
    # 将角度从度转换为弧度进行计算
    theta_rad = np.deg2rad(theta_deg)
    
    direction = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
    direction = direction / np.linalg.norm(direction)
    
    per_drop_coverage = {}
    bombs_list = []
    
    for i in range(num_bombs):
        t_b = params[2 + i]
        pos = UAV_INIT + direction * v * t_b
        
        bombs_list.append((t_b, pos))
    
    # 计算每个烟幕弹的独立贡献
    for i, (t_b, pos) in enumerate(bombs_list):
        coverage = simulate_single_coverage(t_b, pos)
        per_drop_coverage[f'Bomb-{i+1}'] = coverage

    # 计算总遮蔽时间
    total_coverage = simulate_coverage(bombs_list)

    return total_coverage, per_drop_coverage

# ======================
# 优化目标函数
# ======================

def global_evaluate(params):
    """
    粗略优化目标函数：最小化 -所有烟幕弹贡献时间总和
    此阶段旨在确保每个烟幕弹都尽可能地有效。
    参数: [theta, v, t1, t2, t3]
    """
    theta_deg, v = params[:2]
    t_drops = params[2:]
    
    # 投放间隔至少1s，如果间隔太小，则施加惩罚
    t_drops.sort()
    for i in range(len(t_drops) - 1):
        if t_drops[i+1] - t_drops[i] < 1.0:
            return 1e6

    theta_rad = np.deg2rad(theta_deg)
    direction = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
    bombs = []
    for t_b in t_drops:
        pos = UAV_INIT + direction * v * t_b
        bombs.append((t_b, pos))

    # 计算每个烟幕弹的贡献总和作为优化目标
    total_individual_coverage = 0
    for t_b, pos in bombs:
        total_individual_coverage += simulate_single_coverage(t_b, pos)

    return -total_individual_coverage


def final_evaluate(params):
    """
    精细优化目标函数：最小化 -最终总遮蔽时间
    此阶段旨在基于粗略优化结果进行微调，以最大化总遮蔽时间。
    参数: [theta, v, t1, t2, t3]
    """
    theta_deg, v = params[:2]
    t_drops = params[2:]
    
    # 投放间隔至少1s，如果间隔太小，则施加惩罚
    t_drops.sort()
    for i in range(len(t_drops) - 1):
        if t_drops[i+1] - t_drops[i] < 1.0:
            return 1e6

    theta_rad = np.deg2rad(theta_deg)
    direction = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
    bombs = []
    for t_b in t_drops:
        pos = UAV_INIT + direction * v * t_b
        bombs.append((t_b, pos))

    # 计算总遮蔽时间作为最终优化目标
    score = simulate_coverage(bombs)
    return -score


def _generate_excel_data(best_params, per_drop_coverage):
    """生成用于Excel导出的数据列表。"""
    if best_params is None:
        return []

    theta_deg, v = best_params[:2]
    t_drops = best_params[2:]
    
    drop_data = []
    
    # 确保烟幕弹按投放时间顺序排序
    sorted_drops = sorted(zip(t_drops, per_drop_coverage.keys()), key=lambda x: x[0])

    theta_rad = np.deg2rad(theta_deg)
    direction_vec = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
    
    for t_drop, bomb_name in sorted_drops:
        pos = UAV_INIT + direction_vec * v * t_drop
        cover_time = per_drop_coverage.get(bomb_name, 0.0)
            
        drop_data.append({
            '无人机运动方向 (度)': theta_deg,
            '无人机运动速度 (m/s)': v,
            '烟幕干扰弹名称': bomb_name,
            '烟幕干扰弹投放点的x坐标 (m)': pos[0],
            '烟幕干扰弹投放点的y坐标 (m)': pos[1],
            '烟幕干扰弹投放点的z坐标 (m)': pos[2],
            '有效干扰时长 (s)': cover_time
        })
    return drop_data


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

def run_optimization(num_runs=5, seed_file_path="Data/Q4/optimization_seed.pkl"):
    """
    运行多轮优化以寻找最佳烟幕弹投放方案。

    Args:
        num_runs (int): 运行优化的总次数。
        seed_file_path (str): 保存历史最佳结果的文件路径。
    """
    for run_num in range(1, num_runs + 1):
        print("="*40)
        print(f"开始第 {run_num}/{num_runs} 次优化运行")
        print("="*40)

        history = _load_optimization_data(seed_file_path)
        historical_best_coverage = history.get('best_coverage', -1.0)
        last_tried_seed = history.get('last_tried_seed', -1)
        
        current_seed = last_tried_seed + 1
        
        print(f"开始使用新随机种子: {current_seed} 进行优化...")
        
        # --- 阶段1: 粗略优化 (最大化所有烟幕弹的独立贡献) ---
        print("=== 阶段1: 开始粗略优化 (最大化独立贡献总和) ===")
        
        bounds = [
            (0, 360),     # 航向角 (度)
            (70, 140),    # 无人机速度
        ] + [(0, T_MAX)] * NUM_BOMBS
        
        rough_result = differential_evolution(
            global_evaluate,
            bounds,
            maxiter=400,    
            popsize=50,
            polish=True,
            disp=True,
            seed=current_seed
        )
        
        rough_best_params = rough_result.x
        total_coverage_rough, per_drop_coverage_rough = get_coverage_from_params(rough_best_params, NUM_BOMBS)
        
        print("\n=== 粗略优化结果 ===")
        print(f"无人机航向角: {rough_best_params[0]:.2f}° , 速度: {rough_best_params[1]:.2f} m/s")
        print(f"总独立贡献时间: {sum(per_drop_coverage_rough.values()):.2f} s")
        print(f"实际总遮蔽时间 (粗略优化): {total_coverage_rough:.2f} s")
        print("每个烟幕弹贡献:", {k: f"{v:.2f}" for k, v in per_drop_coverage_rough.items()})

        # --- 阶段2: 精细优化 (最大化最终总遮蔽时间) ---
        print("\n=== 阶段2: 开始精细优化 (基于粗略结果微调) ===")
        
        fine_result = differential_evolution(
            final_evaluate,
            bounds,
            maxiter=100, 
            popsize=20,
            x0=rough_best_params,
            polish=True,
            disp=True,
            seed=current_seed
        )
        
        final_best_params = fine_result.x
        final_total_coverage, final_per_drop_coverage = get_coverage_from_params(final_best_params, NUM_BOMBS)
        
        print("\n=== 精细优化后的最终结果 ===")
        print(f"无人机航向角: {final_best_params[0]:.2f}° , 速度: {final_best_params[1]:.2f} m/s")
        print(f"最终总遮蔽时间: {final_total_coverage:.2f} s")
        print("每个烟幕弹最终贡献:", {k: f"{v:.2f}" for k, v in final_per_drop_coverage.items()})

        # --- 阶段3: 针对性优化 (优化无效烟幕弹) ---
        ineffective_bombs = [i for i, (k, v) in enumerate(final_per_drop_coverage.items()) if v < 0.1]
        
        if ineffective_bombs:
            print("\n=== 阶段3: 开始针对性优化 (优化无效烟幕弹) ===")

            # 创建一个局部优化函数，只改变单个烟幕弹的投放时间
            def targeted_evaluate(t_drop_new, params_fixed, bomb_index_to_optimize):
                params_temp = np.array(params_fixed)
                params_temp[2 + bomb_index_to_optimize] = t_drop_new[0]
                # 这里调用final_evaluate是因为我们想在总遮蔽时间的基础上优化这个投放点
                return final_evaluate(params_temp)

            for idx in ineffective_bombs:
                bomb_name = f'Bomb-{idx + 1}'
                initial_t = final_best_params[2 + idx]
                
                print(f"对 {bomb_name} 进行局部优化，初始投放时间: {initial_t:.2f} s")
                
                # 设定一个窄的搜索范围，在初始投放时间附近进行搜索
                local_bounds = [(max(0, initial_t - 5), min(T_MAX, initial_t + 5))]
                
                local_result = differential_evolution(
                    lambda t_drop_new: targeted_evaluate(t_drop_new, final_best_params, idx),
                    local_bounds,
                    maxiter=50, 
                    popsize=10,
                    polish=True,
                    disp=False,
                    seed=current_seed
                )
                
                # 更新最佳参数
                final_best_params[2 + idx] = local_result.x[0]

            # 重新计算最终结果
            final_total_coverage, final_per_drop_coverage = get_coverage_from_params(final_best_params, NUM_BOMBS)
            
            print("\n=== 针对性优化后的最终结果 ===")
            print(f"最终总遮蔽时间: {final_total_coverage:.2f} s")
            print("每个烟幕弹最终贡献:", {k: f"{v:.2f}" for k, v in final_per_drop_coverage.items()})

        # --- 比较并更新历史最佳结果 ---
        if final_total_coverage > historical_best_coverage:
            print("\n本次优化找到比历史最佳更好的结果！正在更新历史记录。")
            history['best_coverage'] = final_total_coverage
            history['best_seed'] = current_seed
            history['best_params'] = final_best_params
        else:
            print(f"\n本次结果 ({final_total_coverage:.2f} s) 不如历史最佳 ({historical_best_coverage:.2f} s)，沿用历史最佳参数。")
        
        # 无论是否找到更好的结果，都更新上次尝试的种子
        history['last_tried_seed'] = current_seed
        _save_optimization_data(history, seed_file_path)

    return history['best_params'], history['best_coverage']

if __name__ == "__main__":
    NUM_RUNS_TO_RUN = 5 # 更改运行的次数
    SEED_FILE_PATH = "Data/Q3/optimization_seed.pkl"
    EXCEL_FILE_PATH = "Data/result1.xlsx"

    best_params, best_coverage = run_optimization(num_runs=NUM_RUNS_TO_RUN, seed_file_path=SEED_FILE_PATH)
    
    print("\n\n" + "="*40)
    print("所有运行完成，以下为历史最佳结果")
    print("="*40)
    
    if best_params is not None:
        final_total_coverage, final_per_drop_coverage = get_coverage_from_params(best_params, NUM_BOMBS)
        print(f"历史最佳总遮蔽时间: {final_total_coverage:.2f} s")
        print("历史最佳参数:")
        print(f"无人机航向角: {best_params[0]:.2f}° , 速度: {best_params[1]:.2f} m/s")
        
        # 为了美观，对烟幕弹贡献进行格式化输出
        sorted_per_drop = sorted(final_per_drop_coverage.items(), key=lambda item: float(item[0].split('-')[1]))
        print("每个烟幕弹贡献:", {k: f"{v:.2f}" for k, v in dict(sorted_per_drop).items()})

        # --- 结果保存 ---
        os.makedirs(os.path.dirname(EXCEL_FILE_PATH), exist_ok=True)
        excel_data = _generate_excel_data(best_params, final_per_drop_coverage)
        
        if excel_data:
            df = pd.DataFrame(excel_data)
            df.to_excel(EXCEL_FILE_PATH, index=False)
            print(f"\n最终结果已保存到 {EXCEL_FILE_PATH}")
        else:
            print("\n没有找到有效的遮蔽方案，未生成 Excel 文件。")
    else:
        print("未找到历史最佳参数，请重新运行。")
