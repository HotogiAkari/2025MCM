# -*- coding: utf-8 -*-
"""
一架无人机多枚烟幕弹干扰优化
优化目标：最大化导弹被烟雾遮蔽的总时长
优化方法：
1. 全局差分进化（DE）优化，使用多进程并行加速。
2. 针对在DE阶段贡献为零的烟幕弹，进行局部贝叶斯优化（BO）精细调整。
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
T_TRUE_BASE = np.array([0.0, 200.0, 0.0]) # 真目标圆柱体底面圆心
vm = 300.0 # 导弹速度
um = (O - np.array([20000.0, 0.0, 2000.0])) / np.linalg.norm(O - np.array([20000.0, 0.0, 2000.0])) # 导弹方向单位向量，直指假目标
M0 = np.array([20000.0, 0.0, 2000.0]) # 导弹初始位置 M1
T_hit = np.linalg.norm(O - M0) / vm # 导弹理论命中假目标时间

# 无人机初始位置
F0 = np.array([17800.0, 0.0, 1800.0])

# 三枚烟幕弹的虚拟标识，用于优化和结果记录
BOMB_NAMES = ['Bomb-1', 'Bomb-2', 'Bomb-3']

# 烟雾和干扰常量
R_eff = 10.0 # 烟雾有效半径
smoke_duration = 20.0 # 烟雾持续时间
sink_speed = 3.0 # 烟雾下沉速度
g = 9.8 # 重力加速度

# 新增：真目标圆柱体参数
T_CYLINDER_RADIUS = 7.0 # 圆柱体半径
T_CYLINDER_HEIGHT = 10.0 # 圆柱体高度

# 模拟评估常量
dt_eval = 0.1 # 模拟时间步长

# 全局变量：用于缓存导弹轨迹，避免重复计算
# 轨迹从t=0到T_hit
MISSILE_TRAJECTORY_TIMES = np.arange(0, T_hit + dt_eval/2, dt_eval)
MISSILE_TRAJECTORY_POS = M0 + vm * MISSILE_TRAJECTORY_TIMES[:, np.newaxis] * um

# ----------------- 核心模拟函数 (已向量化) -----------------
def simulate_coverage(v_uav, theta, t_drop, tau):
    """
    模拟单枚烟幕弹的干扰，计算遮蔽时间。
    该函数已完全向量化，不含任何循环。

    参数:
    v_uav (float): 无人机速度
    theta (float): 无人机飞行角度 (弧度)
    t_drop (float): 无人机投放烟雾时间
    tau (float): 烟雾从投放至爆炸的时间
    
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

    # 计算导弹与真目标基准点（圆心）的视线向量
    line_of_sight_vec = T_TRUE_BASE - Mt_positions
    line_of_sight_len_sq = np.sum(line_of_sight_vec**2, axis=1)
    
    v_mt_to_cm = Cm_positions - Mt_positions

    # 在导弹-目标基准点直线上找到离烟幕中心最近的点
    t = np.sum(v_mt_to_cm * line_of_sight_vec, axis=1) / line_of_sight_len_sq
    Q_positions = Mt_positions + t[:, np.newaxis] * line_of_sight_vec

    # 计算烟幕中心到该直线的距离
    distances_to_line = np.linalg.norm(Cm_positions - Q_positions, axis=1)

    # 确定烟幕是否有效遮蔽
    # 遮蔽条件：
    # 1. 烟幕中心到视线的距离 <= 烟雾半径 + 圆柱体半径
    # 2. 投影点位于导弹与目标基准点之间
    # 3. 烟幕的Z坐标在圆柱体高度范围内（考虑烟雾半径）
    covered_mask = (distances_to_line <= R_eff + T_CYLINDER_RADIUS) & \
                   (t >= 0.0) & (t <= 1.0) & \
                   (Cm_positions[:, 2] >= T_TRUE_BASE[2]) & \
                   (Cm_positions[:, 2] <= T_TRUE_BASE[2] + T_CYLINDER_HEIGHT)
    
    cover_time = np.sum(covered_mask) * dt_eval
    
    if np.any(covered_mask):
        covered_times = valid_times[covered_mask]
        return cover_time, covered_times
    else:
        return 0.0, []

# ----------------- 优化器类 -----------------
class DroneObscurationOptimizer:
    def __init__(self, processes=None):
        self.n_drops = len(BOMB_NAMES)
        self.best_params = None
        self.best_coverage = None
        self.per_drop = {k:0.0 for k in BOMB_NAMES}
        self.processes = processes if processes is not None else cpu_count()
        print(f"检测到 {self.processes} 个 CPU 核心，将用于并行优化。")

    def get_initial_guesses(self):
        """基于启发式方法为每个烟幕弹投放事件生成一个初始策略。"""
        # 共享的无人机参数
        t_proj = max(0.0, np.dot(F0 - M0, um) / vm)
        closest_point_xy = (M0 + um * vm * t_proj)[:2]
        drone_to_point_vec = closest_point_xy - F0[:2]
        theta_guess = np.arctan2(drone_to_point_vec[1], drone_to_point_vec[0])
        v_uav_guess = 105.0

        initial_guesses = [v_uav_guess, theta_guess]

        # 针对每个烟幕弹的独立参数
        for i in range(self.n_drops):
            t_drop_guess = np.linalg.norm(drone_to_point_vec) / v_uav_guess
            tau_guess = max(0.2, t_proj - t_drop_guess)
            initial_guesses.extend([t_drop_guess + i * 5, tau_guess]) # 初始时间错开

        return np.array(initial_guesses)

    def _get_coverage_from_params(self, params):
        """计算给定参数的总遮蔽时间和每个烟幕弹的贡献。"""
        v_uav = params[0]
        theta = params[1]
        
        coverages = []
        for i in range(self.n_drops):
            t_drop = params[2 + i*2]
            tau = params[3 + i*2]
            coverages.append(simulate_coverage(v_uav, theta, t_drop, tau)[0])

        return sum(coverages), {name: coverages[i] for i, name in enumerate(BOMB_NAMES)}

    def de_objective(self, x):
        """
        优化目标函数，包含时间间隔惩罚。
        惩罚项：如果任何两枚烟幕弹的投放时间间隔小于1秒，则施加惩罚。
        """
        # 提取所有烟幕弹的投放时间
        t_drop_values = [x[2 + i*2] for i in range(self.n_drops)]
        
        penalty = 0.0
        # 检查每对烟幕弹的时间间隔
        for i in range(self.n_drops):
            for j in range(i + 1, self.n_drops):
                if abs(t_drop_values[i] - t_drop_values[j]) < 1.0:
                    # 施加一个较大的惩罚，以避免这种情况
                    penalty += 1e6
        
        # 正常的目标函数值（最大化覆盖时间，所以取负）
        base_objective = -self._get_coverage_from_params(x)[0]
        
        # 返回带惩罚的总目标值
        return base_objective + penalty

    def solve_de(self, pop_size=50, max_iter=100, seed=None):
        # 优化参数顺序：v_uav, theta, t_drop_1, tau_1, t_drop_2, tau_2, ...
        # 方向约束为 (pi/2, 3*pi/2)，这确保无人机总是向x轴负方向前进，
        # 从而朝着目标区域飞行，避免了远离目标的无效飞行。
        bounds = [(70.0, 140.0), (np.pi / 2, 3 * np.pi / 2)] + \
                 [(0.0, min(60.0, T_hit - 0.1)), (0.2, 12.0)] * self.n_drops

        initial_guess = self.get_initial_guesses()
        initial_population = np.zeros((pop_size, len(initial_guess)))
        initial_population[0] = initial_guess
        
        rng = np.random.default_rng(seed)
        for i, (lower, upper) in enumerate(bounds):
            initial_population[1:, i] = rng.uniform(lower, upper, pop_size - 1)

        print(f"\n开始全局 DE 优化 ({self.n_drops} 个烟幕弹投放事件)...")
        pbar = tqdm(total=max_iter, desc="DE 优化进度")
        result = differential_evolution(self.de_objective, bounds, popsize=pop_size, maxiter=max_iter, workers=self.processes, updating='deferred', callback=lambda xk, conv: pbar.update(1), init=initial_population, seed=seed)
        pbar.close()
        
        best_params = result.x
        best_coverage, per_drop_temp = self._get_coverage_from_params(best_params)
        return best_params, best_coverage, per_drop_temp

    def single_drop_bo(self, name, fixed_params=None, n_calls=50, n_initial_points=10):
        # 此时只优化投放时间和起爆时间，因为无人机速度和方向是共享的
        space = [
            Real(0.0, min(60.0, T_hit - 0.1), name="t_drop"),
            Real(0.2, 12.0, name="tau")
        ]

        @use_named_args(space)
        def objective(t_drop, tau):
            # 构建完整的参数数组
            params = np.array(self.best_params)
            
            # 提取共享参数
            v_uav = params[0]
            theta = params[1]

            # 替换需要优化的烟幕弹参数
            idx = BOMB_NAMES.index(name)
            params[2 + idx*2] = t_drop
            params[3 + idx*2] = tau

            return -self._get_coverage_from_params(params)[0]

        print(f"\n=== 开始对烟幕弹 {name} 进行单烟幕弹 BO 优化 ===")
        pbar_bo = tqdm(total=n_calls, desc=f"BO 优化进度 ({name})")
        res = gp_minimize(objective, space, n_calls=n_calls, n_initial_points=n_initial_points, random_state=42, callback=lambda res: pbar_bo.update(1))
        pbar_bo.close()
        
        idx = BOMB_NAMES.index(name)
        self.best_params[2 + idx*2] = res.x[0]
        self.best_params[3 + idx*2] = res.x[1]
        
        self.best_coverage, self.per_drop = self._get_coverage_from_params(self.best_params)
        os.makedirs("Data/Q3", exist_ok=True)
        np.save("Data/Q3/best_params_final.npy", self.best_params)
        print(f"烟幕弹 {name} 的 BO 优化完成。其贡献变为: {self.per_drop[name]:.2f}")
        return res

    def run_optimization_pipeline(self, de_pop_size, de_max_iter, bo_calls, bo_initial_points, n_random_seeds, seed_file_path):
        history = self._load_optimization_data(seed_file_path)
        
        self.best_coverage = history.get('best_coverage', -1.0)
        self.best_params = history.get('best_params')
        best_seed = history.get('best_seed')
        failed_seeds = set(history.get('failed_seeds', []))
        
        if self.best_params is not None:
            self.best_coverage, self.per_drop = self._get_coverage_from_params(self.best_params)
            print(f"从历史记录中读取的最佳结果总遮蔽时长为: {self.best_coverage:.2f} s")

        new_seeds_to_try = [s for s in range(n_random_seeds) if s not in failed_seeds]
        print(f"将尝试 {len(new_seeds_to_try)} 个新种子: {new_seeds_to_try}")

        current_run_best_coverage = -1.0
        
        for current_seed in new_seeds_to_try:
            print(f"\n--- 正在运行 DE 优化（随机种子: {current_seed}） ---")
            current_params, current_coverage, current_per_drop = self.solve_de(
                pop_size=de_pop_size, max_iter=de_max_iter, seed=current_seed
            )
            print(f"本次运行结果：总遮蔽时长为 {current_coverage:.2f} s")
            
            if current_coverage > current_run_best_coverage:
                current_run_best_coverage = current_coverage
                self.best_params = current_params
                self.per_drop = current_per_drop
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
        print("每个烟幕弹贡献:", {k: f"{v:.2f}" for k, v in self.per_drop.items()})

        zero_drops = [name for name, val in self.per_drop.items() if val == 0.0]
        if zero_drops and self.best_coverage > 0:
            print("\n检测到以下烟幕弹贡献为零，将进行BO精细优化:", zero_drops)
            for name in zero_drops:
                self.single_drop_bo(name, n_calls=bo_calls, n_initial_points=bo_initial_points)

        print("\n=== 最终优化结果 ===")
        print(f"最终总遮蔽时长: {self.best_coverage:.2f} s")
        print("最终每个烟幕弹贡献:", {k: f"{v:.2f}" for k, v in self.per_drop.items()})
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
        drop_data = []
        v_uav = self.best_params[0]
        theta = self.best_params[1]
        uf = np.array([np.cos(theta), np.sin(theta), 0.0])
        
        direction_deg = (np.degrees(theta) % 360 + 360) % 360

        for i, name in enumerate(BOMB_NAMES):
            t_drop = self.best_params[2 + i*2]
            tau = self.best_params[3 + i*2]
            
            drop_point = F0 + v_uav * uf * t_drop
            drop_point[2] = F0[2]
            explode_point = drop_point + v_uav * uf * tau + np.array([0.0, 0.0, -0.5*g*tau*tau])
            
            cover_time = self.per_drop[name]
            
            drop_data.append({
                '无人机运动方向 (度)': direction_deg,
                '无人机运动速度 (m/s)': v_uav,
                '烟幕干扰弹编号': i + 1,
                '烟幕干扰弹投放点的x坐标 (m)': drop_point[0],
                '烟幕干扰弹投放点的y坐标 (m)': drop_point[1],
                '烟幕干扰弹投放点的z坐标 (m)': drop_point[2],
                '烟幕干扰弹起爆点的x坐标 (m)': explode_point[0],
                '烟幕干扰弹起爆点的y坐标 (m)': explode_point[1],
                '烟幕干扰弹起爆点的z坐标 (m)': explode_point[2],
                '有效干扰时长 (s)': cover_time
            })
        return drop_data

if __name__=="__main__":
    start_time = time.time()
    optimizer = DroneObscurationOptimizer()

    DE_POP_SIZE = 500
    DE_MAX_ITER = 500
    BO_CALLS = 200
    BO_INITIAL_POINTS = 200
    N_RANDOM_SEEDS = 50
    SEED_FILE_PATH = "Data/Q3/optimization_seeds.pkl"

    optimizer.run_optimization_pipeline(DE_POP_SIZE, DE_MAX_ITER, BO_CALLS, BO_INITIAL_POINTS, N_RANDOM_SEEDS, SEED_FILE_PATH)
    
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")
    
    excel_file_path = "Data/result1.xlsx"
    os.makedirs(os.path.dirname(excel_file_path), exist_ok=True)
    print("\n正在将结果保存到 result1.xlsx...")
    
    results_df = pd.DataFrame(optimizer._generate_excel_data())
    results_df.to_excel(excel_file_path, index=False)
    
    print(f"结果已成功保存到 {excel_file_path}")
