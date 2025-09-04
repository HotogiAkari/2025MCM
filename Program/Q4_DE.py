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
    # 无人机飞行向量
    uf = np.array([np.cos(theta), np.sin(theta), 0.0])
    vf = v_uav * uf

    # 烟雾投放点和爆炸点
    drop_point = F0 + vf * t_drop
    drop_point[2] = F0[2] # 确保投放高度为无人机初始高度
    t_exp = t_drop + tau
    explode_point = drop_point + vf * tau + np.array([0.0,0.0,-0.5*g*tau*tau])
    
    # 如果爆炸点在地面以下，则无干扰
    if explode_point[2] <= 0.0:
        return 0.0, []

    # 确定烟雾有效时间段
    t_start_smoke = t_exp
    t_end_smoke = min(t_exp + smoke_duration, T_hit)
    
    # 查找对应的时间步索引
    start_idx = int(np.floor(t_start_smoke / dt_eval))
    end_idx = int(np.floor(t_end_smoke / dt_eval))
    
    if start_idx >= end_idx:
        return 0.0, []

    # 烟雾云团在有效时间段内的位置 (向量化)
    valid_times = MISSILE_TRAJECTORY_TIMES[start_idx:end_idx]
    
    delta_z = -sink_speed * (valid_times - t_exp)
    Cm_positions = np.zeros((valid_times.shape[0], 3))
    Cm_positions[:, 0] = explode_point[0]
    Cm_positions[:, 1] = explode_point[1]
    Cm_positions[:, 2] = explode_point[2] + delta_z
    
    # 导弹在同一时间段内的位置 (从缓存中获取)
    Mt_positions = MISSILE_TRAJECTORY_POS[start_idx:end_idx]

    # --- 核心逻辑变更：判断烟雾云团是否遮蔽导弹与真目标的视线 ---
    # 定义导弹与真目标之间的视线向量
    line_of_sight_vec = T_true - Mt_positions
    line_of_sight_len_sq = np.sum(line_of_sight_vec**2, axis=1)
    
    # 无人机到视线向量的向量
    v_mt_to_cm = Cm_positions - Mt_positions

    # 向量化计算投影 t
    t = np.sum(v_mt_to_cm * line_of_sight_vec, axis=1) / line_of_sight_len_sq

    # 投影点 Q 的位置
    Q_positions = Mt_positions + t[:, np.newaxis] * line_of_sight_vec

    # 计算云团中心到视线的距离
    distances_to_line = np.linalg.norm(Cm_positions - Q_positions, axis=1)

    # 遮蔽条件：
    # 1. 云团中心到视线的距离小于有效半径
    # 2. 投影点 Q 位于视线段上 (即 0 <= t <= 1)
    covered_mask = (distances_to_line <= R_eff) & (t >= 0.0) & (t <= 1.0)
    
    # 计算遮蔽总时间
    cover_time = np.sum(covered_mask) * dt_eval
    
    # 返回结果
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
        """
        基于启发式方法为每架无人机生成一个初始策略。
        此策略旨在让无人机飞向导弹轨迹的最近点，并在该点附近投放烟雾。
        """
        initial_guesses = []
        for name in self.drone_names:
            F0 = self.drones[name]
            
            # 计算导弹轨迹上离无人机初始位置最近的点
            v_m_f = F0 - M0
            # 投影到导弹方向，计算最近点所需的时间
            t_proj = np.dot(v_m_f, um) / vm
            
            # 确保时间在合理范围内
            t_proj = max(0.0, t_proj)
            
            # 计算该点在xy平面上的位置
            closest_point_xy = (M0 + um * vm * t_proj)[:2]
            
            # 计算无人机到该点的方向向量和角度
            drone_to_point_vec = closest_point_xy - F0[:2]
            theta_guess = np.arctan2(drone_to_point_vec[1], drone_to_point_vec[0])
            
            # 无人机速度取中值
            v_uav_guess = 105.0
            
            # 估计投放时间：飞到最近点所需的时间
            dist_to_point = np.linalg.norm(drone_to_point_vec)
            t_drop_guess = dist_to_point / v_uav_guess
            
            # 估计起爆时间：确保烟雾在t_proj时爆炸
            tau_guess = max(0.2, t_proj - t_drop_guess)

            initial_guesses.extend([v_uav_guess, theta_guess, t_drop_guess, tau_guess])
        
        return np.array(initial_guesses)

    # DE 目标函数：将所有无人机的参数作为单个向量输入
    def de_objective(self, x):
        cover_total = 0.0
        # 内部不保存per_drone，因为多进程环境下不安全，在主进程中重新计算
        for i, name in enumerate(self.drone_names):
            v, theta, t_drop, tau = x[i*4:(i+1)*4]
            cover, _ = simulate_coverage(v, theta, t_drop, tau, F0=self.drones[name])
            cover_total += cover
        return -cover_total # DE 最小化

    def solve_de(self, pop_size=50, max_iter=100, seed=None):
        bounds = []
        # 定义飞行方向的约束
        # 导弹从 M0(20000, 0, 2000) 沿 x 轴负方向飞向 O(0,0,0)
        # 无人机需要朝 x 轴负方向飞行以拦截导弹轨迹
        # 这意味着无人机飞行角度 theta (np.cos(theta) = x 方向) 必须在 (pi/2, 3pi/2) 范围内
        THETA_MIN_RAD = np.pi / 2
        THETA_MAX_RAD = 3 * np.pi / 2
        
        for _ in range(self.n_drone):
            # v, theta, t_drop, tau
            bounds += [(70.0, 140.0), (THETA_MIN_RAD, THETA_MAX_RAD), (0.0, min(60.0, T_hit - 0.1)), (0.2, 12.0)]
        
        # 获取启发式初始猜测
        initial_guess = self.get_initial_guesses()
        num_params = len(initial_guess)
        
        # 创建包含启发式猜测的初始种群
        initial_population = np.zeros((pop_size, num_params))
        initial_population[0] = initial_guess
        
        # 使用随机值填充剩余种群成员
        rng = np.random.default_rng(seed)
        for i, (lower, upper) in enumerate(bounds):
            initial_population[1:, i] = rng.uniform(lower, upper, pop_size - 1)

        print(f"\n开始全局 DE 优化 ({self.n_drone} 架无人机)...")
        # 使用tqdm包装进度条
        pbar = tqdm(total=max_iter, desc="DE 优化进度")
        def callback(xk, convergence=0):
            pbar.update(1)

        # 传入正确形状的初始种群，并设置随机种子
        result = differential_evolution(self.de_objective, bounds, popsize=pop_size, maxiter=max_iter, workers=self.processes, updating='deferred', callback=callback, init=initial_population, seed=seed)
        pbar.close()
        
        best_params = result.x
        best_coverage = -result.fun
        
        # 重新计算每架无人机的贡献
        per_drone_temp = {}
        for i, name in enumerate(self.drone_names):
            v, theta, t_drop, tau = best_params[i*4:(i+1)*4]
            cover, _ = simulate_coverage(v, theta, t_drop, tau, F0=self.drones[name])
            per_drone_temp[name] = cover

        return best_params, best_coverage, per_drone_temp

    # 用于 BO 内部计算总遮蔽
    def coverage_for_params(self, params):
        cover_total = 0.0
        per_drone = {}
        for i, name in enumerate(self.drone_names):
            v, theta, t_drop, tau = params[i*4:(i+1)*4]
            cover, _ = simulate_coverage(v, theta, t_drop, tau, F0=self.drones[name])
            cover_total += cover
            per_drone[name] = cover
        return cover_total, per_drone

    # 单无人机 BO
    def single_drone_bo(self, name, fixed_params=None, n_calls=50, n_initial_points=10):
        F0 = self.drones[name]
        space = [
            Real(70.0, 140.0, name="v_uav"),
            Real(0.0, 2*np.pi, name="theta"),
            Real(0.0, min(60.0, T_hit - 0.1), name="t_drop"),
            Real(0.2, 12.0, name="tau")
        ]

        @use_named_args(space)
        def objective(v_uav, theta, t_drop, tau):
            params = np.array(self.best_params) # 创建副本
            idx = self.drone_names.index(name)
            params[idx*4:(idx+1)*4] = [v_uav, theta, t_drop, tau]
            
            # 使用 fixed_params 修正其他无人机的参数
            if fixed_params:
                for fixed_name, vals in fixed_params.items():
                    if fixed_name != name:
                        fixed_idx = self.drone_names.index(fixed_name)
                        params[fixed_idx*4:(fixed_idx+1)*4] = vals
            
            cover_total, _ = self.coverage_for_params(params)
            return -cover_total

        print(f"\n=== 开始对无人机 {name} 进行单无人机 BO 优化 ===")
        
        # 添加BO优化的进度条
        pbar_bo = tqdm(total=n_calls, desc=f"BO 优化进度 ({name})")
        def on_step(res):
            pbar_bo.update(1)

        res = gp_minimize(objective, space, n_calls=n_calls, n_initial_points=n_initial_points, random_state=42, callback=on_step)
        pbar_bo.close()
        
        idx = self.drone_names.index(name)
        self.best_params[idx*4:(idx+1)*4] = res.x
        
        # 重新计算并更新总遮蔽和每架无人机的贡献
        self.best_coverage, self.per_drone = self.coverage_for_params(self.best_params)

        # 确保目录存在，然后保存np文件
        os.makedirs("Data/Q4", exist_ok=True)
        np.save("Data/Q4/best_params_final.npy", self.best_params)
        print(f"无人机 {name} 的 BO 优化完成。其贡献变为: {self.per_drone[name]:.2f}")
        return res

# ----------------- 主流程 -----------------
def load_optimization_data(filepath="Data/Q4/optimization_seeds.pkl"):
    """从文件中加载优化数据。"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            print(f"成功从 {filepath} 加载历史优化数据。")
            return data
    except (FileNotFoundError, EOFError):
        print(f"未找到历史优化文件 {filepath}，将创建新文件。")
        return {
            'best_seed': None,
            'best_coverage': -1.0,
            'best_params': None,
            'failed_seeds': []
        }

def save_optimization_data(data, filepath="Data/Q4/optimization_seeds.pkl"):
    """将优化数据保存到文件。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
        print(f"历史优化数据已成功保存到 {filepath}。")

if __name__=="__main__":
    start_time = time.time()
    
    optimizer = DroneObscurationOptimizer(F0_list)
    
    # 可调整的优化参数
    # --- 全局差分进化 (DE) 参数 ---
    DE_POP_SIZE = 500   # 种群大小，影响搜索广度
    DE_MAX_ITER = 500   # 最大迭代次数，影响搜索深度
    
    # --- 局部贝叶斯优化 (BO) 参数 ---
    BO_CALLS = 200      # BO 的评估总次数
    BO_INITIAL_POINTS = 200 # BO 的初始随机评估点数

    # --- 新增参数：随机种子尝试次数 ---
    N_RANDOM_SEEDS = 15
    SEED_FILE_PATH = "Data/Q4/optimization_seeds.pkl"

    # --- 1. 加载历史数据 ---
    history = load_optimization_data(SEED_FILE_PATH)
    
    best_overall_coverage = history['best_coverage']
    best_overall_params = history['best_params']
    best_seed = history['best_seed']
    failed_seeds = set(history['failed_seeds'])
    
    # 如果历史记录存在最佳结果，先评估它
    if best_overall_params is not None:
        coverage_from_history, per_drone_from_history = optimizer.coverage_for_params(best_overall_params)
        print(f"从历史记录中读取的最佳结果总遮蔽时长为: {coverage_from_history:.2f} s")
        # 确保历史记录的覆盖率是准确的
        best_overall_coverage = coverage_from_history
        optimizer.best_params = best_overall_params
        optimizer.best_coverage = best_overall_coverage
        optimizer.per_drone = per_drone_from_history

    # --- 2. 尝试新的种子 ---
    new_seeds_to_try = [s for s in range(N_RANDOM_SEEDS) if s not in failed_seeds]
    print(f"将尝试 {len(new_seeds_to_try)} 个新种子: {new_seeds_to_try}")

    current_run_best_coverage = -1.0
    current_run_best_params = None
    current_run_best_per_drone = None
    current_run_best_seed = None
    
    for current_seed in new_seeds_to_try:
        print(f"\n--- 正在运行 DE 优化（随机种子: {current_seed}） ---")
        current_params, current_coverage, current_per_drone = optimizer.solve_de(
            pop_size=DE_POP_SIZE, 
            max_iter=DE_MAX_ITER, 
            seed=current_seed
        )
        
        print(f"本次运行结果：总遮蔽时长为 {current_coverage:.2f} s")
        
        # 记录本次运行的最佳结果
        if current_coverage > current_run_best_coverage:
            current_run_best_coverage = current_coverage
            current_run_best_params = current_params
            current_run_best_per_drone = current_per_drone
            current_run_best_seed = current_seed

        # 检查是否为无效种子 (所有无人机贡献都为0)
        if current_coverage == 0.0:
            print(f"种子 {current_seed} 产生的遮蔽时长为0，已标记为无效。")
            failed_seeds.add(current_seed)
    
    # --- 3. 比较并更新历史最佳结果 ---
    if current_run_best_coverage > best_overall_coverage:
        print("\n本次运行找到比历史最佳更好的结果！正在更新历史记录。")
        best_overall_coverage = current_run_best_coverage
        best_overall_params = current_run_best_params
        best_seed = current_run_best_seed
        optimizer.best_params = best_overall_params
        optimizer.best_coverage = best_overall_coverage
        optimizer.per_drone = current_run_best_per_drone
        
        # 保存新的历史最佳结果
        history['best_seed'] = best_seed
        history['best_coverage'] = best_overall_coverage
        history['best_params'] = best_overall_params
        history['failed_seeds'] = list(failed_seeds)
        save_optimization_data(history, SEED_FILE_PATH)
    else:
        print("\n本次运行没有找到比历史最佳更好的结果，将沿用历史最佳参数。")
        # 确保优化器使用历史最佳参数进行后续BO
        optimizer.best_params = best_overall_params
        optimizer.best_coverage = best_overall_coverage
        
    print("\n=== 最终全局 DE 优化结果 ===")
    print(f"总遮蔽时长: {optimizer.best_coverage:.2f} s")
    print("每架无人机贡献:", {k: f"{v:.2f}" for k, v in optimizer.per_drone.items()})

    # --- 4. 局部贝叶斯优化 (BO) ---
    zero_drones = [name for name, val in optimizer.per_drone.items() if val == 0.0]
    
    if zero_drones and optimizer.best_coverage > 0:
        print("\n检测到以下无人机贡献为零，将进行BO精细优化:", zero_drones)
        
        fixed_params = {}
        for i, name in enumerate(optimizer.drone_names):
            if name not in zero_drones:
                v, theta, t_drop, tau = optimizer.best_params[i*4:(i+1)*4]
                fixed_params[name] = [v, theta, t_drop, tau]
                
        for name in zero_drones:
            optimizer.single_drone_bo(name, fixed_params=fixed_params, n_calls=BO_CALLS, n_initial_points=BO_INITIAL_POINTS)

        print("\n=== 最终优化结果 ===")
        print(f"最终总遮蔽时长: {optimizer.best_coverage:.2f} s")
        print("最终每架无人机贡献:", {k: f"{v:.2f}" for k, v in optimizer.per_drone.items()})
        print("最终最优参数:", optimizer.best_params)
    else:
        print("\n所有无人机在DE阶段均有贡献，无需进行BO精细优化。")
        print("最终最优参数:", optimizer.best_params)
        
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")
    
    # --- 5. 保存结果到 result2.xlsx 文件 ---
    excel_file_path = "Data/result2.xlsx"
    os.makedirs(os.path.dirname(excel_file_path), exist_ok=True)
    print("\n正在将结果保存到 result2.xlsx...")
    
    # 重新组织数据以匹配新的Excel格式
    drone_data = []
    for i, name in enumerate(optimizer.drone_names):
        F0 = F0_list[name]
        v_uav, theta, t_drop, tau = optimizer.best_params[i*4:(i+1)*4]
        
        # 计算投放点和起爆点
        uf = np.array([np.cos(theta), np.sin(theta), 0.0])
        vf = v_uav * uf
        drop_point = F0 + vf * t_drop
        drop_point[2] = F0[2] # 确保投放高度为无人机初始高度
        t_exp = t_drop + tau
        explode_point = drop_point + vf * tau + np.array([0.0,0.0,-0.5*g*tau*tau])
        
        # 计算运动方向（转换为度）
        direction_deg = np.degrees(theta) % 360.0
        if direction_deg < 0:
            direction_deg += 360
        
        # 获取有效干扰时长
        cover_time = optimizer.per_drone[name]
        
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

    results_df = pd.DataFrame(drone_data)
    results_df.to_excel(excel_file_path, index=False)
    
    print(f"结果已成功保存到 {excel_file_path}")