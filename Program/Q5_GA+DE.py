# -*- coding: utf-8 -*-
"""
无人机蜂群烟雾干扰优化 (第五题) - 优化版
优化目标：最大化3枚导弹被烟雾遮蔽的总时长
优化方法：
1. 贪心分配：将每架无人机分配给它能产生最大干扰效果的导弹。
2. 局部优化：针对每个无人机及其分配的主要任务，使用贝士优化（BO）寻找最佳参数。

此版本已修改为使用CuPy进行GPU加速。
"""

import time
# 尝试导入CuPy，如果失败则回退到NumPy
try:
    import cupy as np
    from cupy import asnumpy
except ImportError:
    import numpy as np
    def asnumpy(arr):
        return arr
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from tqdm import tqdm
import pickle
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# ----------------- 常量和全局变量 -----------------
# 物理常量
O = np.array([0.0, 0.0, 0.0]) # 假目标位置
T_true = np.array([0.0, 200.0, 0.0]) # 真目标位置（圆柱体底面圆心）
vm = 300.0 # 导弹速度

# 导弹初始位置
MISSILES_LIST = {
    'M1': {'M0': np.array([20000.0, 0.0, 2000.0])},
    'M2': {'M0': np.array([19000.0, 600.0, 2100.0])},
    'M3': {'M0': np.array([18000.0, -600.0, 1900.0])}
}

# 无人机初始位置
F0_list = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0, -2000.0, 1300.0])
}

# 烟雾和干扰常量
R_eff = 10.0 # 烟雾有效半径
smoke_duration = 20.0 # 烟雾持续时间
sink_speed = 3.0 # 烟雾下沉速度
g = 9.8 # 重力加速度

# 模拟评估常量
dt_eval = 0.1 # 模拟时间步长

# ----------------- 预计算导弹轨迹 -----------------
def pre_calculate_missile_trajectories(missiles):
    """预先计算所有导弹的轨迹并缓存，如果可能，将数据保存在GPU上。"""
    trajectories = {}
    for name, data in missiles.items():
        um = (O - data['M0']) / np.linalg.norm(O - data['M0']) # 导弹方向单位向量
        T_hit = np.linalg.norm(O - data['M0']) / vm # 导弹理论命中假目标时间
        times = np.arange(0, T_hit + dt_eval / 2, dt_eval)
        positions = data['M0'] + vm * times[:, np.newaxis] * um
        trajectories[name] = {
            'times': times,
            'positions': positions,
            'T_hit': T_hit
        }
    return trajectories

# ----------------- 核心模拟函数 (已向量化) -----------------
def simulate_coverage(v_uav, theta, t_drop, tau, F0, missile_trajectory):
    """
    模拟单架无人机投放的一枚烟雾干扰弹，计算对特定导弹的遮蔽时间。
    
    参数:
    v_uav (float): 无人机速度
    theta (float): 无人机飞行角度 (弧度)
    t_drop (float): 无人机投放烟雾时间
    tau (float): 烟雾从投放至爆炸的时间
    F0 (np.array): 无人机初始位置
    missile_trajectory (dict): 特定导弹的轨迹数据
    
    返回:
    float: 总遮蔽时间
    """
    # 显式地将 CuPy 标量转换为 Python 浮点数，以避免类型错误
    uf = np.array([np.cos(theta).item(), np.sin(theta).item(), 0.0])
    vf = v_uav * uf

    drop_point = F0 + vf * t_drop
    drop_point[2] = F0[2]
    t_exp = t_drop + tau
    explode_point = drop_point + vf * tau + np.array([0.0, 0.0, -0.5 * g * tau * tau])
    
    if explode_point[2] <= 0.0:
        return 0.0

    t_start_smoke = t_exp
    t_end_smoke = min(t_exp + smoke_duration, missile_trajectory['T_hit'])
    
    start_idx = int(np.floor(t_start_smoke / dt_eval))
    end_idx = int(np.floor(t_end_smoke / dt_eval))
    
    if start_idx >= end_idx:
        return 0.0

    valid_times = missile_trajectory['times'][start_idx:end_idx]
    
    delta_z = -sink_speed * (valid_times - t_exp)
    Cm_positions = explode_point + np.array([0, 0, 1]) * delta_z[:, np.newaxis]
    
    Mt_positions = missile_trajectory['positions'][start_idx:end_idx]

    line_of_sight_vec = T_true - Mt_positions
    line_of_sight_len_sq = np.sum(line_of_sight_vec**2, axis=1)
    
    v_mt_to_cm = Cm_positions - Mt_positions

    t = np.sum(v_mt_to_cm * line_of_sight_vec, axis=1) / line_of_sight_len_sq

    Q_positions = Mt_positions + t[:, np.newaxis] * line_of_sight_vec

    distances_to_line = np.linalg.norm(Cm_positions - Q_positions, axis=1)

    covered_mask = (distances_to_line <= R_eff) & (t >= 0.0) & (t <= 1.0)
    
    cover_time = np.sum(covered_mask) * dt_eval
    
    return float(asnumpy(cover_time))

# ----------------- 优化器类 (重构) -----------------
class DroneObscurationOptimizer:
    def __init__(self, drones, missiles, processes=None):
        self.drones = drones
        self.drone_names = list(drones.keys())
        self.n_drone = len(drones)
        self.missiles = missiles
        self.missile_names = list(missiles.keys())
        self.processes = processes if processes is not None else cpu_count()
        print(f"检测到 {self.processes} 个 CPU 核心，将用于并行优化。")
        self.best_params = {}
        self.assignments = {}
        self.detailed_results = {}

    @staticmethod
    def _precompute_best_per_missile(drone_name, missile_name, space, n_calls, n_initial_points, drones_data, missile_trajectories):
        """
        为单个无人机-导弹对寻找最佳干扰策略，并支持结果持久化。
        此方法被 ProcessPoolExecutor 调用，必须能够独立运行。
        因此，所有必要的全局数据（如无人机和导弹轨迹）都作为参数传递。
        """
        @use_named_args(space)
        def objective(v_uav, theta, t_drop1, tau1, t_drop2, tau2, t_drop3, tau3):
            t_drops = [t_drop1, t_drop2, t_drop3]
            
            # 确保投放时间有序，否则返回极高值作为无效解
            sorted_drop_times = sorted(t_drops)
            if any(sorted_drop_times[i] < sorted_drop_times[i-1] + 1.0 for i in range(1, 3)):
                return 1e9

            # 重新整理参数，按投放时间升序
            t_drops_and_taus = list(zip(t_drops, [tau1, tau2, tau3]))
            t_drops_and_taus.sort(key=lambda x: x[0])

            coverage = 0.0
            for t_drop, tau in t_drops_and_taus:
                coverage += simulate_coverage(v_uav, theta, t_drop, tau, drones_data[drone_name], missile_trajectories[missile_name])
            return -coverage

        # 定义持久化文件路径
        data_dir = "Data/Q5/OptimizationData"
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f"{drone_name}_{missile_name}_results.pkl")
        
        # 尝试从文件中加载之前的优化结果
        x0, y0 = None, None
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                x0 = loaded_data['x_iters']
                y0 = loaded_data['func_vals']
                # 排除无效解，以确保初始点有效
                valid_indices = [i for i, val in enumerate(y0) if val < 1e8]
                x0 = [x0[i] for i in valid_indices]
                y0 = [y0[i] for i in valid_indices]
                print(f"为 {drone_name} -> {missile_name} 加载了 {len(x0)} 个历史点。")
            except Exception as e:
                print(f"加载历史数据失败: {e}")
                x0, y0 = None, None

        # 运行贝叶斯优化，传入历史数据作为初始点
        res = gp_minimize(
            objective, 
            space, 
            n_calls=n_calls, 
            n_initial_points=n_initial_points, 
            x0=x0, 
            y0=y0,
            random_state=42
        )
        
        # 保存最新的优化结果的关键数据，避免序列化问题
        with open(file_path, 'wb') as f:
            pickle.dump({'x_iters': res.x_iters, 'func_vals': res.func_vals}, f)
        
        best_coverage = -float(asnumpy(res.fun))
        best_params = res.x
        
        # 获取每次投放的具体贡献
        t_drops_unsorted = [best_params[2], best_params[4], best_params[6]]
        taus_unsorted = [best_params[3], best_params[5], best_params[7]]
        sorted_pairs = sorted(zip(t_drops_unsorted, taus_unsorted))
        
        individual_bomb_contributions = []
        for t_drop, tau in sorted_pairs:
            individual_bomb_contributions.append(simulate_coverage(best_params[0], best_params[1], t_drop, tau, drones_data[drone_name], missile_trajectories[missile_name]))
        
        return drone_name, missile_name, best_coverage, best_params, individual_bomb_contributions

    def run_optimization_pipeline(self, initial_opt_params, refinement_opt_params):
        if 'cupy' in globals() and isinstance(np, type(cupy)):
            print("成功导入 CuPy，将使用 GPU 进行计算。")
        else:
            print("未检测到 CuPy 或 GPU，将回退到使用 NumPy 的 CPU 计算。")
            
        print("第一步：预计算每架无人机对每枚导弹的最大干扰效果...")
        best_per_pair = {}
        with ProcessPoolExecutor(max_workers=self.processes) as executor:
            # 传递 drones_data 和 missile_trajectories 给子进程
            futures = {executor.submit(DroneObscurationOptimizer._precompute_best_per_missile, d, m, initial_opt_params['space'], initial_opt_params['n_calls'], initial_opt_params['n_initial_points'], self.drones, self.missiles): (d, m) 
                       for d in self.drone_names for m in self.missile_names}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="预计算进度"):
                drone_name, missile_name, coverage, params, individual_bomb_contributions = future.result()
                best_per_pair[(drone_name, missile_name)] = {
                    'coverage': coverage, 
                    'params': params,
                    'individual_bomb_contributions': individual_bomb_contributions
                }
        
        print("\n第二步：基于贪心算法进行无人机-导弹任务分配...")
        remaining_missiles = list(self.missile_names)
        remaining_drones = list(self.drone_names)
        
        assignments = {}
        for _ in range(len(remaining_drones)):
            if not remaining_missiles or not remaining_drones:
                break
            
            best_drone, best_missile, max_coverage = None, None, -1.0
            for d in remaining_drones:
                for m in remaining_missiles:
                    current_coverage = best_per_pair[(d, m)]['coverage']
                    if current_coverage > max_coverage:
                        max_coverage = current_coverage
                        best_drone = d
                        best_missile = m
            
            if best_drone and best_missile:
                assignments[best_drone] = best_missile
                self.best_params[best_drone] = best_per_pair[(best_drone, best_missile)]['params']
                self.detailed_results[best_drone] = best_per_pair[(best_drone, best_missile)]
                remaining_drones.remove(best_drone)
                remaining_missiles.remove(best_missile)
                print(f"分配: 无人机 {best_drone} -> 导弹 {best_missile} (预期贡献: {max_coverage:.2f} s)")
        
        for d in remaining_drones:
            closest_missile = min(self.missile_names, key=lambda m: np.linalg.norm(self.drones[d] - MISSILES_LIST[m]['M0']))
            assignments[d] = closest_missile
            self.best_params[d] = best_per_pair[(d, closest_missile)]['params']
            self.detailed_results[d] = best_per_pair[(d, closest_missile)]
            print(f"分配: 无人机 {d} -> 剩余导弹 {closest_missile}")
            
        self.assignments = assignments
        
        final_total_coverage = sum(self.detailed_results[drone_name]['coverage'] for drone_name in self.assignments)
        final_per_drone = {drone: self.detailed_results[drone]['coverage'] for drone in self.assignments}
        
        self.per_drone_coverage = final_per_drone
        
        print("\n=== 阶段一：贪心分配结果 ===")
        print(f"最终总遮蔽时长: {final_total_coverage:.2f} s")
        print("每架无人机贡献:", {k: f"{v:.2f}" for k, v in self.per_drone_coverage.items()})

        # --- 第三步：二次优化 ---
        print("\n第三步：二次优化低贡献无人机...")
        initial_contributions = self.per_drone_coverage
        avg_contribution = sum(initial_contributions.values()) / len(initial_contributions)
        low_contributing_drones = [d for d, c in initial_contributions.items() if c < avg_contribution]
        
        if not low_contributing_drones:
            print("没有发现低贡献无人机，无需二次优化。")
        else:
            with ProcessPoolExecutor(max_workers=self.processes) as executor:
                # 传递 drones_data 和 missile_trajectories 给子进程
                futures = {executor.submit(DroneObscurationOptimizer._precompute_best_per_missile, d, self.assignments[d], refinement_opt_params['space'], refinement_opt_params['n_calls'], refinement_opt_params['n_initial_points'], self.drones, self.missiles): d
                           for d in low_contributing_drones}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="二次优化进度"):
                    drone_name, missile_name, refined_coverage, refined_params, refined_individual_bomb_contributions = future.result()
                    
                    original_coverage = initial_contributions[drone_name]
                    if refined_coverage > original_coverage:
                        print(f"无人机 {drone_name} 优化成功！贡献从 {original_coverage:.2f}s 提升到 {refined_coverage:.2f}s。")
                        self.detailed_results[drone_name] = {
                            'coverage': refined_coverage,
                            'params': refined_params,
                            'individual_bomb_contributions': refined_individual_bomb_contributions
                        }
                        self.per_drone_coverage[drone_name] = refined_coverage

        final_total_coverage = sum(self.per_drone_coverage.values())
        
        print("\n=== 最终结果（经过二次优化） ===")
        print(f"最终总遮蔽时长: {final_total_coverage:.2f} s")
        print("每架无人机贡献:", {k: f"{v:.2f}" for k, v in self.per_drone_coverage.items()})


    def _generate_excel_data(self):
        """生成用于Excel导出的数据列表，直接使用存储的详细结果，避免重复计算。"""
        drone_data_list = []
        for name in self.drone_names:
            if name not in self.detailed_results:
                drone_data_list.append({'无人机编号': name, '烟幕干扰弹编号': '无', '无人机运动方向 (度)': '无',
                                        '无人机运动速度 (m/s)': '无', '烟幕干扰弹投放时间 (s)': '无', '烟幕干扰弹起爆时间 (s)': '无',
                                        '对所有导弹的有效干扰时长 (s)': 0})
                continue
            
            details = self.detailed_results[name]
            params = details['params']
            v_uav = params[0]
            theta = params[1]
            F0 = F0_list[name]
            individual_bomb_contributions = details['individual_bomb_contributions']
            
            t_drops_unsorted = [params[2], params[4], params[6]]
            taus_unsorted = [params[3], params[5], params[7]]
            sorted_pairs = sorted(zip(t_drops_unsorted, taus_unsorted))
            
            for j, (t_drop, tau) in enumerate(sorted_pairs):
                bomb_coverage_on_all_missiles = individual_bomb_contributions[j]
                
                # 确保将 CuPy/NumPy 标量转换为 Python 浮点数
                uf = np.array([np.cos(theta).item(), np.sin(theta).item(), 0.0])
                drop_point = F0 + v_uav * uf * t_drop
                drop_point[2] = F0[2]
                explode_point = drop_point + v_uav * uf * tau + np.array([0.0, 0.0, -0.5*g*tau*tau])
                
                direction_deg = (np.degrees(asnumpy(theta).item()) % 360 + 360) % 360
                
                drone_data_list.append({
                    '无人机编号': name,
                    '烟幕干扰弹编号': j+1,
                    '无人机运动方向 (度)': direction_deg,
                    '无人机运动速度 (m/s)': float(asnumpy(v_uav).item()),
                    '烟幕干扰弹投放时间 (s)': float(asnumpy(t_drop).item()),
                    '烟幕干扰弹起爆时间 (s)': float(asnumpy(tau).item()),
                    '烟幕干扰弹投放点的x坐标 (m)': float(asnumpy(drop_point[0]).item()),
                    '烟幕干扰弹投放点的y坐标 (m)': float(asnumpy(drop_point[1]).item()),
                    '烟幕干扰弹投放点的z坐标 (m)': float(asnumpy(drop_point[2]).item()),
                    '烟幕干扰弹起爆点的x坐标 (m)': float(asnumpy(explode_point[0]).item()),
                    '烟幕干扰弹起爆点的y坐标 (m)': float(asnumpy(explode_point[1]).item()),
                    '烟幕干扰弹起爆点的z坐标 (m)': float(asnumpy(explode_point[2]).item()),
                    '对所有导弹的有效干扰时长 (s)': float(asnumpy(bomb_coverage_on_all_missiles).item())
                })
        
        results_df = pd.DataFrame(drone_data_list)
        
        summary_df = results_df.groupby('无人机编号').agg(
            {'对所有导弹的有效干扰时长 (s)': 'sum'}
        ).rename(columns={'对所有导弹的有效干扰时长 (s)': '总有效干扰时长 (s)'})
        summary_df.reset_index(inplace=True)
        
        total_row = pd.DataFrame([['总计', summary_df['总有效干扰时长 (s)'].sum()]], columns=summary_df.columns)
        summary_df = pd.concat([summary_df, total_row], ignore_index=True)
        
        return results_df, summary_df

if __name__=="__main__":
    
    # ----------------- 可调参数 -----------------
    # 优化参数
    OPTIMIZATION_SPACE = [
        Real(70.0, 140.0, name="v_uav"),
        Real(0.0, 2*np.pi, name="theta"),
        Real(0.0, 60.0, name="t_drop1"),
        Real(0.2, 12.0, name="tau1"),
        Real(0.0, 60.0, name="t_drop2"),
        Real(0.2, 12.0, name="tau2"),
        Real(0.0, 60.0, name="t_drop3"),
        Real(0.2, 12.0, name="tau3")
    ]
    
    # 初始贪心分配阶段的BO参数
    INITIAL_OPT_PARAMS = {
        'space': OPTIMIZATION_SPACE,
        'n_calls': 100,
        'n_initial_points': 20
    }

    # 二次优化阶段的BO参数
    REFINEMENT_OPT_PARAMS = {
        'space': OPTIMIZATION_SPACE,
        'n_calls': 200,
        'n_initial_points': 50
    }
    
    # ----------------- 主程序流程 -----------------
    start_time = time.time()
    
    # 预计算导弹轨迹，此操作只需进行一次
    MISSILE_TRAJECTORIES = pre_calculate_missile_trajectories(MISSILES_LIST)
    
    optimizer = DroneObscurationOptimizer(F0_list, MISSILE_TRAJECTORIES)
    optimizer.run_optimization_pipeline(INITIAL_OPT_PARAMS, REFINEMENT_OPT_PARAMS)
    
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")
    
    excel_file_path = "Data/result3.xlsx"
    os.makedirs(os.path.dirname(excel_file_path), exist_ok=True)
    print("\n正在将结果保存到 result3.xlsx...")
    
    results_df, summary_df = optimizer._generate_excel_data()
    
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='详细投放策略', index=False)
        summary_df.to_excel(writer, sheet_name='无人机总贡献', index=False)
    
    print(f"结果已成功保存到 {excel_file_path}")
