# -*- coding: utf-8 -*-
"""
无人机蜂群烟雾干扰优化
优化目标：最大化导弹被烟雾遮蔽的总时长
优化方法：
1. 全局差分进化（DE）优化，使用多进程并行加速。
2. 针对在DE阶段贡献为零的无人机，进行局部贝叶斯优化（BO）精细调整。
"""

import time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from tqdm import tqdm
import pickle
from multiprocessing import cpu_count
import sys
import os

# ----------------- 常量和全局变量 -----------------
# 物理常量
O = np.array([0.0, 0.0, 0.0]) # 假目标位置
T_true = np.array([0.0, 200.0, 0.0]) # 真目标位置（圆柱体底面圆心）
M0 = np.array([20000.0, 0.0, 2000.0]) # 导弹初始位置 M1
vm = 300.0 # 导弹速度
um = (O - M0) / np.linalg.norm(O - M0) # 导弹方向单位向量，直指假目标
T_hit = np.linalg.norm(O - M0) / vm # 导弹理论命中假目标时间

# 无人机初始位置
F0_list = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0])
}

# 烟雾和干扰常量
R_eff = 10.0 # 烟雾有效半径
smoke_duration = 20.0 # 烟雾持续时间
sink_speed = 3.0 # 烟雾下沉速度
g = 9.8 # 重力加速度

# 模拟评估常量
dt_eval = 0.1 # 模拟时间步长

# 全局变量：用于缓存导弹轨迹，避免重复计算
# 轨迹从t=0到T_hit
MISSILE_TRAJECTORY_TIMES = np.arange(0, T_hit + dt_eval/2, dt_eval)
MISSILE_TRAJECTORY_POS = M0 + vm * MISSILE_TRAJECTORY_TIMES[:, np.newaxis] * um

# ----------------- 核心模拟函数 (已向量化) -----------------
def simulate_coverage(v_uav, theta, t_drop, tau, F0):
    """
    模拟单架无人机的烟雾干扰，计算遮蔽时间。
    该函数已完全向量化，不含任何循环。

    参数:
    v_uav (float): 无人机速度
    theta (float): 无人机飞行角度 (弧度)
    t_drop (float): 无人机投放烟雾时间
    tau (float): 烟雾从投放至爆炸的时间
    F0 (np.array): 无人机初始位置
    
    返回:
    tuple: (总遮蔽时间, 被遮蔽的时间点数组)
    """
    uf = np.array([np.cos(theta), np.sin(theta), 0.0])
    vf = v_uav * uf

    drop_point = F0 + vf * t_drop
    drop_point[2] = F0[2]
    t_exp = t_drop + tau
    explode_point = drop_point + vf * tau + np.array([0.0,0.0,-0.5*g*tau*tau])
    
    if explode_point[2] <= 0.0:
        return 0.0, []

    t_start_smoke = t_exp
    t_end_smoke = min(t_exp + smoke_duration, T_hit)
    
    start_idx = int(np.floor(t_start_smoke / dt_eval))
    end_idx = int(np.floor(t_end_smoke / dt_eval))
    
    if start_idx >= end_idx:
        return 0.0, []

    valid_times = MISSILE_TRAJECTORY_TIMES[start_idx:end_idx]
    
    delta_z = -sink_speed * (valid_times - t_exp)
    Cm_positions = explode_point + np.array([0, 0, 1]) * delta_z[:, np.newaxis]
    
    Mt_positions = MISSILE_TRAJECTORY_POS[start_idx:end_idx]

    line_of_sight_vec = T_true - Mt_positions
    line_of_sight_len_sq = np.sum(line_of_sight_vec**2, axis=1)
    
    v_mt_to_cm = Cm_positions - Mt_positions

    t = np.sum(v_mt_to_cm * line_of_sight_vec, axis=1) / line_of_sight_len_sq

    Q_positions = Mt_positions + t[:, np.newaxis] * line_of_sight_vec

    distances_to_line = np.linalg.norm(Cm_positions - Q_positions, axis=1)

    covered_mask = (distances_to_line <= R_eff) & (t >= 0.0) & (t <= 1.0)
    
    cover_time = np.sum(covered_mask) * dt_eval
    
    if np.any(covered_mask):
        covered_times = valid_times[covered_mask]
        return cover_time, covered_times
    else:
        return 0.0, []

# ----------------- 优化器类 -----------------
class DroneObscurationOptimizer:
    def __init__(self, drones, processes=None):
        self.drones = drones
        self.drone_names = list(drones.keys())
        self.n_drone = len(drones)
        self.best_params = None
        self.best_coverage = None
        self.per_drone = {k:0.0 for k in self.drone_names}
        self.processes = processes if processes is not None else cpu_count()
        print(f"检测到 {self.processes} 个 CPU 核心，将用于并行优化。")

    def get_initial_guesses(self):
        """基于启发式方法为每架无人机生成一个初始策略。"""
        initial_guesses = []
        for name in self.drone_names:
            F0 = self.drones[name]
            t_proj = max(0.0, np.dot(F0 - M0, um) / vm)
            closest_point_xy = (M0 + um * vm * t_proj)[:2]
            drone_to_point_vec = closest_point_xy - F0[:2]
            theta_guess = np.arctan2(drone_to_point_vec[1], drone_to_point_vec[0])
            v_uav_guess = 105.0
            t_drop_guess = np.linalg.norm(drone_to_point_vec) / v_uav_guess
            tau_guess = max(0.2, t_proj - t_drop_guess)
            initial_guesses.extend([v_uav_guess, theta_guess, t_drop_guess, tau_guess])
        return np.array(initial_guesses)

    def _get_coverage_from_params(self, params):
        """计算给定参数的总遮蔽时间和每架无人机的贡献。"""
        coverages = [simulate_coverage(*params[i*4:(i+1)*4], F0=self.drones[name])[0] for i, name in enumerate(self.drone_names)]
        return sum(coverages), {name: coverages[i] for i, name in enumerate(self.drone_names)}

    def de_objective(self, x):
        return -self._get_coverage_from_params(x)[0]

    def solve_de(self, pop_size=50, max_iter=100, seed=None):
        bounds = [(70.0, 140.0), (np.pi / 2, 3 * np.pi / 2), (0.0, min(60.0, T_hit - 0.1)), (0.2, 12.0)] * self.n_drone
        initial_guess = self.get_initial_guesses()
        initial_population = np.zeros((pop_size, len(initial_guess)))
        initial_population[0] = initial_guess
        
        rng = np.random.default_rng(seed)
        for i, (lower, upper) in enumerate(bounds):
            initial_population[1:, i] = rng.uniform(lower, upper, pop_size - 1)

        print(f"\n开始全局 DE 优化 ({self.n_drone} 架无人机)...")
        pbar = tqdm(total=max_iter, desc="DE 优化进度")
        result = differential_evolution(self.de_objective, bounds, popsize=pop_size, maxiter=max_iter, workers=self.processes, updating='deferred', callback=lambda xk, conv: pbar.update(1), init=initial_population, seed=seed)
        pbar.close()
        
        best_params = result.x
        best_coverage, per_drone_temp = self._get_coverage_from_params(best_params)
        return best_params, best_coverage, per_drone_temp

    def single_drone_bo(self, name, fixed_params=None, n_calls=50, n_initial_points=10):
        space = [
            Real(70.0, 140.0, name="v_uav"),
            Real(0.0, 2*np.pi, name="theta"),
            Real(0.0, min(60.0, T_hit - 0.1), name="t_drop"),
            Real(0.2, 12.0, name="tau")
        ]

        @use_named_args(space)
        def objective(v_uav, theta, t_drop, tau):
            params = np.array(self.best_params)
            idx = self.drone_names.index(name)
            params[idx*4:(idx+1)*4] = [v_uav, theta, t_drop, tau]
            if fixed_params:
                for fixed_name, vals in fixed_params.items():
                    if fixed_name != name:
                        fixed_idx = self.drone_names.index(fixed_name)
                        params[fixed_idx*4:(fixed_idx+1)*4] = vals
            return -self._get_coverage_from_params(params)[0]

        print(f"\n=== 开始对无人机 {name} 进行单无人机 BO 优化 ===")
        pbar_bo = tqdm(total=n_calls, desc=f"BO 优化进度 ({name})")
        res = gp_minimize(objective, space, n_calls=n_calls, n_initial_points=n_initial_points, random_state=42, callback=lambda res: pbar_bo.update(1))
        pbar_bo.close()
        
        idx = self.drone_names.index(name)
        self.best_params[idx*4:(idx+1)*4] = res.x
        self.best_coverage, self.per_drone = self._get_coverage_from_params(self.best_params)
        os.makedirs("Data/Q4", exist_ok=True)
        np.save("Data/Q4/best_params_final.npy", self.best_params)
        print(f"无人机 {name} 的 BO 优化完成。其贡献变为: {self.per_drone[name]:.2f}")
        return res

    def run_optimization_pipeline(self, de_pop_size, de_max_iter, bo_calls, bo_initial_points, n_random_seeds, seed_file_path):
        history = self._load_optimization_data(seed_file_path)
        
        self.best_coverage = history.get('best_coverage', -1.0)
        self.best_params = history.get('best_params')
        best_seed = history.get('best_seed')
        failed_seeds = set(history.get('failed_seeds', []))
        
        if self.best_params is not None:
            self.best_coverage, self.per_drone = self._get_coverage_from_params(self.best_params)
            print(f"从历史记录中读取的最佳结果总遮蔽时长为: {self.best_coverage:.2f} s")

        new_seeds_to_try = [s for s in range(n_random_seeds) if s not in failed_seeds]
        print(f"将尝试 {len(new_seeds_to_try)} 个新种子: {new_seeds_to_try}")

        current_run_best_coverage = -1.0
        
        for current_seed in new_seeds_to_try:
            print(f"\n--- 正在运行 DE 优化（随机种子: {current_seed}） ---")
            current_params, current_coverage, current_per_drone = self.solve_de(
                pop_size=de_pop_size, max_iter=de_max_iter, seed=current_seed
            )
            print(f"本次运行结果：总遮蔽时长为 {current_coverage:.2f} s")
            
            if current_coverage > current_run_best_coverage:
                current_run_best_coverage = current_coverage
                self.best_params = current_params
                self.per_drone = current_per_drone
                best_seed = current_seed

            if current_coverage == 0.0:
                print(f"种子 {current_seed} 产生的遮蔽时长为0，已标记为无效。")
                failed_seeds.add(current_seed)
        
        if current_run_best_coverage > self.best_coverage:
            print("\n本次运行找到比历史最佳更好的结果！正在更新历史记录。")
            self.best_coverage = current_run_best_coverage
            history.update({
                'best_seed': best_seed,
                'best_coverage': self.best_coverage,
                'best_params': self.best_params,
                'failed_seeds': list(failed_seeds)
            })
            self._save_optimization_data(history, seed_file_path)
        else:
            print("\n本次运行没有找到比历史最佳更好的结果，将沿用历史最佳参数。")
        
        print("\n=== 最终全局 DE 优化结果 ===")
        print(f"总遮蔽时长: {self.best_coverage:.2f} s")
        print("每架无人机贡献:", {k: f"{v:.2f}" for k, v in self.per_drone.items()})

        zero_drones = [name for name, val in self.per_drone.items() if val == 0.0]
        if zero_drones and self.best_coverage > 0:
            print("\n检测到以下无人机贡献为零，将进行BO精细优化:", zero_drones)
            # 使用字典推导式优化重复部分
            fixed_params = {name: self.best_params[i*4:(i+1)*4].tolist() for i, name in enumerate(self.drone_names) if name not in zero_drones}
            for name in zero_drones:
                self.single_drone_bo(name, fixed_params, n_calls=bo_calls, n_initial_points=bo_initial_points)

        print("\n=== 最终优化结果 ===")
        print(f"最终总遮蔽时长: {self.best_coverage:.2f} s")
        print("最终每架无人机贡献:", {k: f"{v:.2f}" for k, v in self.per_drone.items()})
        print("最终最优参数:", self.best_params)

    def _load_optimization_data(self, filepath):
        """从文件中加载历史优化数据。"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            print(f"未找到历史优化文件 {filepath}，将创建新文件。")
            return {'best_seed': None, 'best_coverage': -1.0, 'best_params': None, 'failed_seeds': []}

    def _save_optimization_data(self, data, filepath):
        """将优化数据保存到文件。"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            print(f"历史优化数据已成功保存到 {filepath}。")
            
    def _generate_excel_data(self):
        """生成用于Excel导出的数据列表。"""
        drone_data = []
        for i, name in enumerate(self.drone_names):
            v_uav, theta, t_drop, tau = self.best_params[i*4:(i+1)*4]
            uf = np.array([np.cos(theta), np.sin(theta), 0.0])
            drop_point = F0_list[name] + v_uav * uf * t_drop
            drop_point[2] = F0_list[name][2]
            explode_point = drop_point + v_uav * uf * tau + np.array([0.0, 0.0, -0.5*g*tau*tau])
            
            direction_deg = (np.degrees(theta) % 360 + 360) % 360
            cover_time = self.per_drone[name]
            
            drone_data.append({
                '无人机编号': name,
                '无人机运动方向 (度)': direction_deg,
                '无人机运动速度 (m/s)': v_uav,
                '烟幕干扰弹投放点的x坐标 (m)': drop_point[0],
                '烟幕干扰弹投放点的y坐标 (m)': drop_point[1],
                '烟幕干扰弹投放点的z坐标 (m)': drop_point[2],
                '烟幕干扰弹起爆点的x坐标 (m)': explode_point[0],
                '烟幕干扰弹起爆点的y坐标 (m)': explode_point[1],
                '烟幕干扰弹起爆点的z坐标 (m)': explode_point[2],
                '有效干扰时长 (s)': cover_time
            })
        return drone_data

if __name__=="__main__":
    start_time = time.time()
    optimizer = DroneObscurationOptimizer(F0_list)

    # 可调整的优化参数
    # --- DE) 参数 ---
    DE_POP_SIZE = 500   # 种群大小，影响搜索广度
    DE_MAX_ITER = 500   # 最大迭代次数，影响搜索深度
    
    # --- BO 参数 ---
    BO_CALLS = 200      # BO 的评估总次数
    BO_INITIAL_POINTS = 200 # BO 的初始随机评估点数

    # --- 随机种子尝试次数 ---
    N_RANDOM_SEEDS = 15
    SEED_FILE_PATH = "Data/Q4/optimization_seeds.pkl"

    optimizer.run_optimization_pipeline(DE_POP_SIZE, DE_MAX_ITER, BO_CALLS, BO_INITIAL_POINTS, N_RANDOM_SEEDS, SEED_FILE_PATH)
    
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")
    
    excel_file_path = "Data/result2.xlsx"
    os.makedirs(os.path.dirname(excel_file_path), exist_ok=True)
    print("\n正在将结果保存到 result2.xlsx...")
    
    results_df = pd.DataFrame(optimizer._generate_excel_data())
    results_df.to_excel(excel_file_path, index=False)
    
    print(f"结果已成功保存到 {excel_file_path}")
