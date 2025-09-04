import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from tqdm import tqdm
import os
import time
import math
import datetime

class DroneObscurationOptimizer:
    """
    使用差分进化算法解决无人机协同烟幕干扰问题的优化器，
    支持断点续传和分步保存。
    """

    def __init__(self):
        # --- 基本物理和场景常量 ---
        self.G = 9.8  # 重力加速度
        self.MISSILE_V = 300.0  # 导弹速度
        self.CLOUD_SINK_V = 3.0  # 烟云下沉速度
        self.CLOUD_RADIUS = 10.0  # 烟云半径
        self.CLOUD_LIFETIME = 20.0  # 烟云持续时间
        self.DRONE_BOUNDS = [
            (70, 140),
            (0, 2 * np.pi),
            (1, 1000 - 5),
            (1, 20)
        ]
        
        # --- 坐标定义 ---
        self.M1_INITIAL_POS = np.array([20000.0, 0.0, 2000.0])
        self.FALSE_TARGET_POS = np.array([0.0, 0.0, 0.0])
        self.TRUE_TARGET_CENTER_POS = np.array([0.0, 200.0, 5.0]) 

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
        
        self.DRONE_BOUNDS[2] = (1, self.missile_total_flight_time - 5)

    def _get_missile_pos(self, t):
        if t > self.missile_total_flight_time:
            return self.FALSE_TARGET_POS
        return self.M1_INITIAL_POS + self.missile_unit_vector * self.MISSILE_V * t

    def _dist_point_to_line_segment(self, p, a, b):
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
        obscuration_times = set()
        drone_params = np.array(params_12d).reshape(3, 4)
        for i, drone_name in enumerate(self.DRONE_NAMES):
            v, theta, t_fly, t_fall = drone_params[i]
            drone_initial_pos = self.DRONES_INITIAL_POS[drone_name]
            
            drone_v_vector = np.array([v * np.cos(theta), v * np.sin(theta), 0])
            drop_pos = drone_initial_pos + drone_v_vector * t_fly
            
            detonation_pos_z = drop_pos[2] - 0.5 * self.G * t_fall**2
            detonation_pos_xy = drop_pos[:2] + drone_v_vector[:2] * t_fall
            detonation_pos = np.array([detonation_pos_xy[0], detonation_pos_xy[1], detonation_pos_z])
            
            if detonation_pos[2] < 0:
                continue
            
            detonation_time = t_fly + t_fall
            start_time = detonation_time
            end_time = detonation_time + self.CLOUD_LIFETIME
            
            for t in np.arange(start_time, end_time, 0.2):
                if t > self.missile_total_flight_time:
                    break
                
                missile_pos = self._get_missile_pos(t)
                cloud_center = detonation_pos - np.array([0, 0, self.CLOUD_SINK_V * (t - detonation_time)])
                
                distance = self._dist_point_to_line_segment(p=cloud_center, a=missile_pos, b=self.TRUE_TARGET_CENTER_POS)
                
                if distance <= self.CLOUD_RADIUS:
                    obscuration_times.add(round(t, 1))
        
        return len(obscuration_times) * 0.2

    def _objective_function(self, x):
        return -self._calculate_obscuration_intervals(x)

    def _format_and_save_output(self, best_params, prefix=""):
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
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'{prefix}result_{timestamp}.xlsx')
        try:
            df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"结果已成功保存到: {output_path}")
        except Exception as e:
            print(f"保存Excel文件失败: {e}")
        return df

    def solve_and_save_checkpoints(self, population_size=40, max_iterations=100, 
                                   checkpoint_interval=10, init_population=None):
        """
        执行差分进化优化，并定期保存检查点。
        
        参数:
            population_size (int): 种群大小。
            max_iterations (int): 最大迭代次数。
            checkpoint_interval (int): 每隔多少次迭代保存一次检查点。
            init_population (np.ndarray): 用于热启动的初始种群，通常为上一次的最佳解。
        """
        print(f"开始进行无人机协同策略优化...")
        print(f"优化参数: 种群大小={population_size}, 最大迭代次数={max_iterations}")
        
        global_iteration_counter = 0
        pbar = tqdm(total=max_iterations, desc="优化进度")

        def callback(xk, convergence):
            nonlocal global_iteration_counter
            global_iteration_counter += 1
            pbar.update(1)

            if global_iteration_counter % checkpoint_interval == 0:
                print(f"\n--- 已完成 {global_iteration_counter} 次迭代，正在保存检查点 ---")
                
                # 保存最佳参数到npy文件
                checkpoint_data = {'iterations': global_iteration_counter, 'best_params': xk}
                checkpoint_dir = 'Checkpoints'
                os.makedirs(checkpoint_dir, exist_ok=True)
                np.save(os.path.join(checkpoint_dir, 'checkpoint.npy'), checkpoint_data)
                print(f"检查点已保存到: Checkpoints/checkpoint.npy")
                
                # 保存Excel结果文件
                self._format_and_save_output(xk, prefix=f'iter_{global_iteration_counter}_')
                pbar.set_description(f"优化进度 (检查点已保存)")

        bounds = self.DRONE_BOUNDS * 3

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
            callback=callback,
            updating='deferred',
            init=init_population if init_population is not None else 'random'
        )
        
        pbar.close()
        
        best_params = result.x
        max_obscuration_time = -result.fun
        print(f"\n优化完成。")
        print(f"找到的最优策略可实现最长总遮蔽时间: {max_obscuration_time:.2f} 秒。")
        
        return self._format_and_save_output(best_params, prefix='final_')

    def resume_from_checkpoint(self, population_size=40, total_iterations=100, 
                               checkpoint_interval=10, checkpoint_path='Checkpoints/checkpoint.npy'):
        """
        从检查点文件加载模型并继续优化。
        
        参数:
            population_size (int): 种群大小。
            total_iterations (int): 总共要运行的迭代次数。
            checkpoint_interval (int): 每隔多少次迭代保存一次检查点。
            checkpoint_path (str): 检查点文件的路径。
        """
        print(f"正在从检查点文件 {checkpoint_path} 读取模型...")
        if not os.path.exists(checkpoint_path):
            print("错误：检查点文件不存在，无法恢复。")
            return

        checkpoint_data = np.load(checkpoint_path, allow_pickle=True).item()
        start_iterations = checkpoint_data['iterations']
        initial_params = checkpoint_data['best_params']
        
        print(f"已从第 {start_iterations} 次迭代恢复。")
        
        # 构建一个热启动的初始种群，将上次的最佳解作为一部分
        dim = len(initial_params)
        init_population = np.random.rand(population_size, dim)
        init_population[0] = initial_params
        
        # 接下来调用主优化函数，传入热启动种群，并设置剩余迭代次数
        self.solve_and_save_checkpoints(
            population_size=population_size,
            max_iterations=total_iterations - start_iterations,
            checkpoint_interval=checkpoint_interval,
            init_population=init_population
        )

# --- 使用示例 ---
if __name__ == '__main__':
    optimizer = DroneObscurationOptimizer()
    
    # 定义优化参数
    POP_SIZE = 50
    MAX_ITER = 200
    CHECKPOINT_INTERVAL = 50
    CHECKPOINT_PATH = 'Checkpoints/checkpoint.npy'
    
    # 自动检测检查点并决定运行模式
    if os.path.exists(CHECKPOINT_PATH):
        print("检测到检查点文件，将从上次中断处继续优化...")
        optimizer.resume_from_checkpoint(
            population_size=POP_SIZE,
            total_iterations=MAX_ITER,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            checkpoint_path=CHECKPOINT_PATH
        )
    else:
        print("未检测到检查点文件，将开始新的优化...")
        final_results_df = optimizer.solve_and_save_checkpoints(
            population_size=POP_SIZE, 
            max_iterations=MAX_ITER, 
            checkpoint_interval=CHECKPOINT_INTERVAL
        )
        print("\n--- 最终最优策略详情 ---")
        print(final_results_df.to_string())