import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from tqdm import tqdm
import os
import time
import math

class DroneObscurationOptimizer:
    """
    使用差分进化算法解决无人机协同烟幕干扰问题的优化器。
    该类包含定义物理和场景常量的初始化、计算烟幕干扰效果的私有方法，
    以及使用差分进化算法寻找最优无人机策略的求解器。
    """

    def __init__(self):
        # --- 基本物理和场景常量 ---
        self.G = 9.8  # 重力加速度
        self.MISSILE_V = 300.0  # 导弹速度
        self.CLOUD_SINK_V = 3.0  # 烟云下沉速度
        self.CLOUD_RADIUS = 10.0  # 烟云半径
        self.CLOUD_LIFETIME = 20.0  # 烟云持续时间
        
        # --- 坐标定义 ---
        self.M1_INITIAL_POS = np.array([20000.0, 0.0, 2000.0])  # 导弹初始位置
        self.FALSE_TARGET_POS = np.array([0.0, 0.0, 0.0])  # 假目标位置
        self.TRUE_TARGET_CENTER_POS = np.array([0.0, 200.0, 5.0]) # 真目标中心位置 

        self.DRONES_INITIAL_POS = {
            'FY1': np.array([17800.0, 0.0, 1800.0]),
            'FY2': np.array([12000.0, 1400.0, 1400.0]),
            'FY3': np.array([6000.0, -3000.0, 700.0]),
        }
        self.DRONE_NAMES = ['FY1', 'FY2', 'FY3']

        # --- 导弹路径预计算 ---
        missile_path_vector = self.FALSE_TARGET_POS - self.M1_INITIAL_POS
        self.missile_path_dist = np.linalg.norm(missile_path_vector)
        self.missile_unit_vector = missile_path_vector / self.missile_path_dist
        self.missile_total_flight_time = self.missile_path_dist / self.MISSILE_V
        
        # FIX: 删除 self.pbar 实例变量。
        # 进度条应该在 solve() 方法中创建和销毁，而不是作为类的持久状态。

    def _get_missile_pos(self, t):
        """根据时间 t 计算导弹位置。"""
        if t > self.missile_total_flight_time:
            return self.FALSE_TARGET_POS
        return self.M1_INITIAL_POS + self.missile_unit_vector * self.MISSILE_V * t

    def _dist_point_to_line_segment(self, p, a, b):
        """计算点 p 到线段 (a, b) 的最短距离。"""
        ap = p - a
        ab = b - a
        ab_squared = np.dot(ab, ab)
        if ab_squared == 0.0:
            return np.linalg.norm(ap)
        t = np.dot(ap, ab) / ab_squared
        if t < 0.0:
            return np.linalg.norm(p - a)
        elif t > 1.0:
            return np.linalg.norm(p - b)
        projection = a + t * ab
        return np.linalg.norm(p - projection)

    def _calculate_obscuration_intervals(self, params_12d):
        """
        计算在给定的12维参数（3架无人机，每架4个参数）下，总的有效遮蔽时间。
        
        参数:
            params_12d (list or np.array): 包含所有无人机策略参数的12个元素。
        
        返回:
            float: 总的有效遮蔽时间（单位：秒）。
        """
        obscuration_times = set()
        drone_params = np.array(params_12d).reshape(3, 4)
        for i, drone_name in enumerate(self.DRONE_NAMES):
            v, theta, t_fly, t_fall = drone_params[i]
            drone_initial_pos = self.DRONES_INITIAL_POS[drone_name]
            
            # 无人机飞行过程
            drone_v_vector = np.array([v * np.cos(theta), v * np.sin(theta), 0])
            drop_pos = drone_initial_pos + drone_v_vector * t_fly
            
            # 烟幕弹下落过程
            # 考虑到烟幕弹的重力下落
            detonation_pos_z = drop_pos[2] - 0.5 * self.G * t_fall**2
            detonation_pos_xy = drop_pos[:2] + drone_v_vector[:2] * t_fall
            detonation_pos = np.array([detonation_pos_xy[0], detonation_pos_xy[1], detonation_pos_z])
            
            # 检查起爆点是否在地面以上
            if detonation_pos[2] < 0:
                continue
            
            detonation_time = t_fly + t_fall
            start_time = detonation_time
            end_time = detonation_time + self.CLOUD_LIFETIME
            
            # 遍历烟云生命周期内的每 0.2 秒，判断是否遮蔽
            for t in np.arange(start_time, end_time, 0.2):
                if t > self.missile_total_flight_time:
                    break
                
                missile_pos = self._get_missile_pos(t)
                # 烟云中心位置随时间下沉
                cloud_center = detonation_pos - np.array([0, 0, self.CLOUD_SINK_V * (t - detonation_time)])
                
                # 计算导弹轨迹与烟云中心的距离
                distance = self._dist_point_to_line_segment(p=cloud_center, a=missile_pos, b=self.TRUE_TARGET_CENTER_POS)
                
                if distance <= self.CLOUD_RADIUS:
                    obscuration_times.add(round(t, 1))
        
        return len(obscuration_times) * 0.1

    def _objective_function(self, x):
        """目标函数，用于最大化遮蔽时间，因此需要对结果取负。"""
        return -self._calculate_obscuration_intervals(x)

    def solve(self, population_size=40, max_iterations=100):
        """
        执行差分进化优化，并返回最优策略。
        
        参数:
            population_size (int): 种群大小。
            max_iterations (int): 最大迭代次数。
        """
        print("开始进行无人机协同策略优化...")
        print(f"优化参数: 种群大小={population_size}, 最大迭代次数={max_iterations}")
        start_time = time.time()
        
        # 定义每个无人机的参数边界 [速度, 角度, 飞行时间, 下落时间]
        drone_bounds = [
            (70, 140),  # 速度 (m/s)
            (0, 2 * np.pi),  # 角度 (弧度)
            (1, self.missile_total_flight_time - 5), # 飞行时间 (s)
            (1, 20)  # 下落时间 (s)
        ]
        # 三架无人机共12个参数
        bounds = drone_bounds * 3

        # FIX: 在 solve() 方法内部创建 tqdm 进度条。
        pbar = tqdm(total=max_iterations, desc="优化进度")

        # FIX: 定义一个嵌套的回调函数，使其可以访问外部的 pbar 变量。
        def callback(xk, convergence):
            """
            嵌套回调函数，用于在每次迭代后更新进度条。
            xk: 当前最佳解
            convergence: 收敛性指标
            """
            pbar.update(1)

        result = differential_evolution(
            func=self._objective_function,
            bounds=bounds,
            strategy='best1bin',
            maxiter=max_iterations,
            popsize=population_size,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            workers=-1,
            # FIX: 传入新定义的嵌套回调函数。
            callback=callback,
            updating='deferred'
        )
        
        # FIX: 优化完成后，关闭进度条。
        pbar.close()
        
        end_time = time.time()
        print(f"\n优化完成。耗时: {end_time - start_time:.2f} 秒。")
        
        best_params = result.x
        max_obscuration_time = -result.fun
        print(f"找到的最优策略可实现最长总遮蔽时间: {max_obscuration_time:.2f} 秒。")
        
        return self._format_and_save_output(best_params)

    def _format_and_save_output(self, best_params):
        """格式化输出结果并保存到Excel文件。"""
        drone_params = np.array(best_params).reshape(3, 4)
        output_data = []
        for i, drone_name in enumerate(self.DRONE_NAMES):
            v, theta, t_fly, t_fall = drone_params[i]
            drone_initial_pos = self.DRONES_INITIAL_POS[drone_name]
            
            drone_v_vector = np.array([v * np.cos(theta), v * np.sin(theta), 0])
            drop_pos = drone_initial_pos + drone_v_vector * t_fly
            
            detonation_pos_z = drop_pos[2] - 0.5 * self.G * t_fall**2
            detonation_pos_xy = drop_pos[:2] + drone_v_vector[:2] * t_fall
            detonation_pos = np.array([detonation_pos_xy[0], detonation_pos_xy[1], detonation_pos_z])
            
            single_drone_params = np.zeros(12)
            single_drone_params[i*4:(i+1)*4] = drone_params[i]
            effective_duration = self._calculate_obscuration_intervals(single_drone_params)
            
            row = {
                '无人机编号': drone_name,
                '无人机运动方向 (度)': math.degrees(theta) % 360,
                '无人机运动速度 (m/s)': v,
                '烟幕干扰弹投放点的x坐标 (m)': drop_pos[0],
                '烟幕干扰弹投放点的y坐标 (m)': drop_pos[1],
                '烟幕干扰弹投放点的z坐标 (m)': drop_pos[2],
                '烟幕干扰弹起爆点的x坐标 (m)': detonation_pos[0],
                '烟幕干扰弹起爆点的y坐标 (m)': detonation_pos[1],
                '烟幕干扰弹起爆点的z坐标 (m)': detonation_pos[2],
                '有效干扰时长 (s)': effective_duration,
            }
            output_data.append(row)
            
        df = pd.DataFrame(output_data)
        output_dir = 'Data'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, 'result2.xlsx')
        try:
            df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"结果已成功保存到: {output_path}")
        except Exception as e:
            print(f"保存Excel文件失败: {e}")
        return df

if __name__ == '__main__':
    optimizer = DroneObscurationOptimizer()
    final_results_df = optimizer.solve(population_size=50, max_iterations=150)
    print("\n--- 最优策略详情 ---")
    print(final_results_df.to_string())
