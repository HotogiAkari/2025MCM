# Q5_GA_DE_optimized_v3.py
# -*- coding: utf-8 -*-
"""
混合差分进化优化：解决烟幕干扰投放问题（高性能版）
- 修复 missile_pos 形状不匹配问题
- GPU 加速（CuPy 可选）
- Numba 优化核心计算
- 向量化循环，减少重复计算
- 全局缓存导弹轨迹
- 改进初始种群和并行效率
"""

import os
import math
import time
import pickle
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional CMA-ES
try:
    import cma
    CMA_OK = True
except ImportError:
    CMA_OK = False

# Optional Numba
try:
    from numba import jit
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False

# Optional CuPy for GPU
try:
    import cupy as cp
    CUPY_OK = True
except ImportError:
    CUPY_OK = False
    cp = np  # Fallback to NumPy

# ---------------- Problem constants ----------------
g = 9.8
cloud_radius = 10.0            # m
cloud_duration = 20.0          # s
cloud_sink_speed = 3.0         # m/s
missile_speed = 300.0          # m/s
drone_vmin, drone_vmax = 70.0, 140.0

# Target center
target_center = np.array([0.0, 200.0, 5.0])

# Missiles initial pos
missiles_init = {
    "M1": np.array([20000.0, 0.0, 2000.0]),
    "M2": np.array([19000.0, 600.0, 2100.0]),
    "M3": np.array([18000.0, -600.0, 1900.0]),
}
missile_names = list(missiles_init.keys())

# Drones initial pos
drones_init = {
    "FY1": np.array([17800.0, 0.0, 1800.0]),
    "FY2": np.array([12000.0, 1400.0, 1400.0]),
    "FY3": np.array([6000.0, -3000.0, 700.0]),
    "FY4": np.array([11000.0, 2000.0, 1800.0]),
    "FY5": np.array([13000.0, -2000.0, 1300.0]),
}
drone_names = list(drones_init.keys())

# Simulation defaults
DEFAULT_DT = 0.5
TIME_HORIZON = 200.0
MAX_BOMBS = 3

# Global cache
missile_traj_cache = {}
NUM_VARS = None
VAR_LOW = None
VAR_HIGH = None
guided_headings = None


# ---------------- Geometry helpers ----------------
if NUMBA_OK:
    @jit(nopython=True)
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
else:
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


def missile_dir(name):
    return unit(np.array([0.0, 0.0, 0.0]) - missiles_init[name])


def missile_pos(name, t, use_cupy=False):
    xp = cp if use_cupy and CUPY_OK else np
    s0 = xp.array(missiles_init[name], dtype=xp.float64)
    v = xp.array(missile_dir(name), dtype=xp.float64)
    t = xp.asarray(t, dtype=xp.float64)
    # Ensure broadcasting: s0 (3,) + v (3,) * t (n,) -> (n, 3)
    return s0 + v * missile_speed * t[:, None]


def missile_time_to_origin(name):
    return np.linalg.norm(missiles_init[name]) / missile_speed


if NUMBA_OK:
    @jit(nopython=True)
    def line_point_distance_segment(a, b, p):
        ab = b - a
        ab2 = np.dot(ab, ab)
        if ab2 == 0:
            return np.linalg.norm(p - a), 0.0
        u = float(np.dot(p - a, ab) / ab2)
        u_clamped = max(0.0, min(1.0, u))
        closest = a + u_clamped * ab
        return float(np.linalg.norm(p - closest)), u_clamped
else:
    def line_point_distance_segment(a, b, p):
        ab = b - a
        ab2 = np.dot(ab, ab)
        if ab2 == 0:
            return np.linalg.norm(p - a), 0.0
        u = float(np.dot(p - a, ab) / ab2)
        u_clamped = max(0.0, min(1.0, u))
        closest = a + u_clamped * ab
        return float(np.linalg.norm(p - closest)), u_clamped


# ---------------- Drone / Bomb geometry ----------------
if NUMBA_OK:
    @jit(nopython=True)
    def drone_pos_at(p0, heading_deg, speed, t):
        theta = math.radians(heading_deg)
        vxy = np.array([math.cos(theta), math.sin(theta), 0.0]) * speed
        pos = p0 + vxy * t
        pos[2] = p0[2]
        return pos
else:
    def drone_pos_at(p0, heading_deg, speed, t):
        theta = math.radians(heading_deg)
        vxy = np.array([math.cos(theta), math.sin(theta), 0.0]) * speed
        pos = p0 + vxy * t
        pos[2] = p0[2]
        return pos


if NUMBA_OK:
    @jit(nopython=True)
    def bomb_drop_det_points(p0, heading_deg, speed, t_drop, tau):
        drop = drone_pos_at(p0, heading_deg, speed, t_drop)
        theta = math.radians(heading_deg)
        vxy = np.array([math.cos(theta), math.sin(theta), 0.0]) * speed
        det = drop + vxy * tau
        det_z = p0[2] - 0.5 * g * (tau ** 2)
        det[2] = max(0.0, det_z)
        return drop, det
else:
    def bomb_drop_det_points(p0, heading_deg, speed, t_drop, tau):
        drop = drone_pos_at(p0, heading_deg, speed, t_drop)
        theta = math.radians(heading_deg)
        vxy = np.array([math.cos(theta), math.sin(theta), 0.0]) * speed
        det = drop + vxy * tau
        det_z = p0[2] - 0.5 * g * (tau ** 2)
        det[2] = max(0.0, det_z)
        return drop, det


def cloud_center_at(det_point, t_det, t, use_cupy=False):
    xp = cp if use_cupy and CUPY_OK else np
    if isinstance(t, (int, float)):
        if t < t_det or t > t_det + cloud_duration:
            return None
        center = det_point.copy()
        center[2] = max(0.0, det_point[2] - cloud_sink_speed * (t - t_det))
        return center
    # Vectorized for arrays
    t = xp.asarray(t, dtype=xp.float64)
    mask = (t >= t_det) & (t <= t_det + cloud_duration)
    centers = xp.zeros_like(t, dtype=xp.float64)
    centers[:] = det_point[2] - cloud_sink_speed * (t - t_det)
    centers = xp.maximum(0.0, centers)
    centers[~mask] = xp.nan
    return xp.vstack([det_point[0] * xp.ones_like(t), det_point[1] * xp.ones_like(t), centers]).T


if NUMBA_OK:
    @jit(nopython=True)
    def is_blocking_cloud(center, missile_pos_t, target_center_point, tol_extra=0.0):
        dist, u = line_point_distance_segment(missile_pos_t, target_center_point, center)
        return u > 0.0 and u < 1.0 and dist <= (cloud_radius + tol_extra)
else:
    def is_blocking_cloud(center, missile_pos_t, target_center_point, tol_extra=0.0):
        dist, u = line_point_distance_segment(missile_pos_t, target_center_point, center)
        return u > 0.0 and u < 1.0 and dist <= (cloud_radius + tol_extra)


# ---------------- Assignment: nearest missile per drone ----------------
def assign_drones_to_missiles():
    assign = {}
    for dname, p0 in drones_init.items():
        best_m = None
        best_dist = float('inf')
        for m in missile_names:
            a = missiles_init[m]
            b = np.array([0.0, 0.0, 0.0])
            dist, _ = line_point_distance_segment(a, b, p0)
            if dist < best_dist:
                best_dist = dist
                best_m = m
        assign[dname] = best_m
    return assign


# ---------------- Fitness / evaluation ----------------
if NUMBA_OK:
    @jit(nopython=True)
    def compute_blocking_score(missile_pos, target_center, cloud_center, t, t_det, tol_extra, angular_weight):
        dist, u = line_point_distance_segment(missile_pos, target_center, cloud_center)
        if u <= 0.0 or u >= 1.0 or dist > (cloud_radius + tol_extra):
            return 0.0
        vm = target_center - missile_pos
        vc = target_center - cloud_center
        nm = np.linalg.norm(vm)
        nc = np.linalg.norm(vc)
        if nm < 1e-9 or nc < 1e-9:
            cosang = 1.0
        else:
            cosang = min(1.0, max(-1.0, np.dot(vm, vc) / (nm * nc)))
        reward = (cosang + 1.0) / 2.0
        return (1.0 - angular_weight) + angular_weight * reward
else:
    def compute_blocking_score(missile_pos, target_center, cloud_center, t, t_det, tol_extra, angular_weight):
        dist, u = line_point_distance_segment(missile_pos, target_center, cloud_center)
        if u <= 0.0 or u >= 1.0 or dist > (cloud_radius + tol_extra):
            return 0.0
        vm = target_center - missile_pos
        vc = target_center - cloud_center
        nm = np.linalg.norm(vm)
        nc = np.linalg.norm(vc)
        if nm < 1e-9 or nc < 1e-9:
            cosang = 1.0
        else:
            cosang = min(1.0, max(-1.0, np.dot(vm, vc) / (nm * nc)))
        reward = (cosang + 1.0) / 2.0
        return (1.0 - angular_weight) + angular_weight * reward


def evaluate_chromosome(chrom, drone_assignments, dt=DEFAULT_DT, tol_extra=6.0, angular_weight=0.6, spacing_penalty_coeff=10.0, use_cupy=False):
    xp = cp if use_cupy and CUPY_OK else np
    vals = xp.array(chrom, dtype=xp.float64)
    num_drones = len(drone_names)

    # Build events
    events = []
    for di, dname in enumerate(drone_names):
        base = di * (2 + 2 * MAX_BOMBS)
        heading = float(vals[base + 0]) % 360.0
        speed = float(xp.clip(vals[base + 1], drone_vmin, drone_vmax))
        p0 = drones_init[dname]
        for bi in range(MAX_BOMBS):
            t_drop = float(vals[base + 2 + 2 * bi])
            tau = float(vals[base + 2 + 2 * bi + 1])
            t_drop = max(0.0, t_drop)
            tau = max(0.0, tau)
            drop, det = bomb_drop_det_points(p0, heading, speed, t_drop, tau)
            events.append({
                "drone": dname,
                "bomb_idx": bi + 1,
                "heading": heading,
                "speed": speed,
                "t_drop": t_drop,
                "tau": tau,
                "drop": drop,
                "det": det,
                "t_det": t_drop + tau,
                "assigned_missile": drone_assignments[dname]
            })

    # Spacing penalty
    spacing_penalty = 0.0
    for dname in drone_names:
        tlist = sorted([ev["t_drop"] for ev in events if ev["drone"] == dname])
        for i in range(1, len(tlist)):
            delta = tlist[i] - tlist[i - 1]
            if delta < 1.0:
                spacing_penalty += (1.0 - delta)
    spacing_penalty *= spacing_penalty_coeff

    # Dynamic time grid
    max_missile_time = max(missile_time_to_origin(m) for m in missile_names)
    times = xp.arange(0.0, min(max_missile_time, TIME_HORIZON) + 1e-9, dt)

    # Use cached missile positions
    missile_positions = {}
    for m in missile_names:
        cache_key = (m, dt, float(max_missile_time))  # Unique key based on missile, dt, and time horizon
        if cache_key not in missile_traj_cache:
            missile_traj_cache[cache_key] = missile_pos(m, times, use_cupy=use_cupy)
        missile_positions[m] = missile_traj_cache[cache_key]

    per_missile_block = {m: 0.0 for m in missile_names}
    for m in missile_names:
        mpos = missile_positions[m]
        if use_cupy and CUPY_OK:
            mpos = cp.asarray(mpos)
        missile_end_time = missile_time_to_origin(m)
        valid_times = times[times <= missile_end_time]
        if len(valid_times) == 0:
            continue
        valid_mpos = mpos[:len(valid_times)]
        for ev in [e for e in events if e["assigned_missile"] == m]:
            t_det = ev["t_det"]
            centers = cloud_center_at(ev["det"], t_det, valid_times, use_cupy=use_cupy)
            if use_cupy and CUPY_OK:
                valid_centers = cp.isfinite(centers[:, 2])
                centers = centers[valid_centers]
                valid_mpos_t = valid_mpos[valid_centers]
                valid_times_t = valid_times[valid_centers]
                if len(centers) == 0:
                    continue
                ab = target_center - valid_mpos_t
                ab2 = cp.sum(ab * ab, axis=1)
                nonzero = ab2 > 0
                ab = ab[nonzero]
                ab2 = ab2[nonzero]
                centers = centers[nonzero]
                valid_mpos_t = valid_mpos_t[nonzero]
                valid_times_t = valid_times_t[nonzero]
                if len(ab) == 0:
                    continue
                u = cp.sum((centers - valid_mpos_t) * ab, axis=1) / ab2
                u_clamped = cp.clip(u, 0.0, 1.0)
                closest = valid_mpos_t + u_clamped[:, None] * ab
                dist = cp.linalg.norm(centers - closest, axis=1)
                blocking = (u_clamped > 0.0) & (u_clamped < 1.0) & (dist <= cloud_radius + tol_extra)
                if not cp.any(blocking):
                    continue
                vm = target_center - valid_mpos_t[blocking]
                vc = target_center - centers[blocking]
                nm = cp.linalg.norm(vm, axis=1)
                nc = cp.linalg.norm(vc, axis=1)
                valid = (nm >= 1e-9) & (nc >= 1e-9)
                cosang = cp.ones_like(nm)
                cosang[valid] = cp.sum(vm[valid] * vc[valid], axis=1) / (nm[valid] * nc[valid])
                cosang = cp.clip(cosang, -1.0, 1.0)
                reward = (cosang + 1.0) / 2.0
                weighted = (1.0 - angular_weight) + angular_weight * reward
                per_missile_block[m] += float(dt * cp.max(weighted))
            else:
                for ti, t in enumerate(valid_times):
                    center = cloud_center_at(ev["det"], t_det, t, use_cupy=False)
                    if center is None:
                        continue
                    reward = compute_blocking_score(valid_mpos[ti], target_center, center, t, t_det, tol_extra, angular_weight)
                    per_missile_block[m] = max(per_missile_block[m], dt * reward)

    score = sum(per_missile_block.values()) - spacing_penalty
    return max(0.0, float(score))


# ---------------- Guided initialization ----------------
def compute_guided_headings(offsets_deg=None):
    if offsets_deg is None:
        offsets_deg = [-8.0, 6.0, -10.0, 8.0, -6.0]
    headings = {}
    for i, d in enumerate(drone_names):
        p0 = drones_init[d]
        base = math.degrees(math.atan2(target_center[1] - p0[1], target_center[0] - p0[0])) % 360.0
        headings[d] = (base + offsets_deg[i]) % 360.0
    return headings


def greedy_initial_pop(popsize=48):
    num_drones = len(drone_names)
    num_vars = num_drones * (2 + 2 * MAX_BOMBS)
    pop = []
    for _ in range(popsize):
        chrom = np.zeros(num_vars)
        for di, dname in enumerate(drone_names):
            idx = di * (2 + 2 * MAX_BOMBS)
            dist_to_target = np.linalg.norm(drones_init[dname] - target_center)
            t_max = dist_to_target / drone_vmin
            t_min = dist_to_target / drone_vmax
            t_window = (max(5.0, t_min * 0.8), min(55.0, t_max * 1.2))
            chrom[idx + 0] = (guided_headings[dname] + np.random.normal(0.0, 4.0)) % 360.0
            chrom[idx + 1] = np.clip(110.0 + np.random.normal(0.0, 8.0), drone_vmin, drone_vmax)
            pts = np.linspace(t_window[0], t_window[1], MAX_BOMBS + 2)[1:-1]
            for j in range(MAX_BOMBS):
                chrom[idx + 2 + 2 * j] = np.clip(pts[j] + np.random.normal(0.0, 2.0), 0.0, t_window[1])
                chrom[idx + 2 + 2 * j + 1] = np.clip(4.5 + j + np.random.normal(0.0, 1.0), 0.1, 40.0)
        pop.append(chrom)
    return np.array(pop)


# ---------------- Differential Evolution ----------------
def differential_evolution(pop_init, fitness_fn, iters=600, F=0.8, CR=0.7, workers=4,
                           checkpoint_dir="Data/Q5", checkpoint_every=50):
    os.makedirs(checkpoint_dir, exist_ok=True)
    popsize = pop_init.shape[0]
    pop = pop_init.copy()
    chunksize = max(1, popsize // max(workers, 1) // 4)  # Finer chunksize
    if workers > 1:
        with Pool(workers) as pool:
            scores = np.array(pool.map(fitness_fn, list(pop), chunksize=chunksize))
    else:
        scores = np.array([fitness_fn(ind) for ind in pop])
    best_idx = int(np.argmax(scores))
    best = pop[best_idx].copy()
    best_score = float(scores[best_idx])

    pbar = tqdm(range(iters), desc="DE iters")
    for it in pbar:
        trials = []
        for i in range(popsize):
            idxs = [j for j in range(popsize) if j != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            a, b, c = pop[a_idx], pop[b_idx], pop[c_idx]
            mutant = a + F * (b - c)
            mutant = np.clip(mutant, VAR_LOW, VAR_HIGH)
            trial = pop[i].copy()
            jrand = np.random.randint(0, NUM_VARS)
            for j in range(NUM_VARS):
                if np.random.rand() < CR or j == jrand:
                    trial[j] = mutant[j]
            trials.append(trial)
        if workers > 1:
            with Pool(workers) as pool:
                trial_scores = np.array(pool.map(fitness_fn, trials, chunksize=chunksize))
        else:
            trial_scores = np.array([fitness_fn(tr) for tr in trials])
        for i in range(popsize):
            if trial_scores[i] >= scores[i]:
                pop[i] = trials[i]
                scores[i] = trial_scores[i]
        cur_best_idx = int(np.argmax(scores))
        if scores[cur_best_idx] > best_score:
            best_score = float(scores[cur_best_idx])
            best = pop[cur_best_idx].copy()
        if (it + 1) % checkpoint_every == 0:
            fname = os.path.join(checkpoint_dir, f"de_ckpt_it{it+1}.pkl")
            with open(fname, "wb") as f:
                pickle.dump({"pop": pop, "scores": scores, "best": best, "best_score": best_score, "iter": it+1}, f)
        pbar.set_postfix({"best": round(best_score, 3)})
    return best, best_score


# ---------------- Optional refine (CMA-ES) ----------------
def refine_with_cmaes(base_chrom, fitness_fn, maxiter=200, workers=4):
    if not CMA_OK:
        return base_chrom, fitness_fn(base_chrom)
    x0 = base_chrom.copy()
    sigma0 = 2.0
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, {'popsize': 12})
    chunksize = max(1, 12 // max(workers, 1) // 4)
    while not es.stop() and es.countiter < maxiter:
        X = es.ask()
        with Pool(workers) as pool:
            vals = pool.map(lambda x: -fitness_fn(np.array(x)), X, chunksize=chunksize)
        es.tell(X, vals)
    xbest = np.array(es.result.xbest)
    xbest = np.clip(xbest, VAR_LOW, VAR_HIGH)
    return xbest, fitness_fn(xbest)


# ---------------- Save results to Excel ----------------
def save_result_excel(chrom, out_path="Data/result3.xlsx", dt=DEFAULT_DT, drone_assignments=None):
    rows = []
    for di, dname in enumerate(drone_names):
        base = di * (2 + 2 * MAX_BOMBS)
        heading = float(chrom[base + 0]) % 360.0
        speed = float(np.clip(chrom[base + 1], drone_vmin, drone_vmax))
        p0 = drones_init[dname]
        for bi in range(MAX_BOMBS):
            t_drop = float(chrom[base + 2 + 2 * bi])
            tau = float(chrom[base + 2 + 2 * bi + 1])
            drop, det = bomb_drop_det_points(p0, heading, speed, t_drop, tau)
            assigned_m = drone_assignments[dname]
            block_dur = 0.0
            t_det = t_drop + tau
            times = np.arange(t_det, t_det + cloud_duration + 1e-9, dt)
            for t in times:
                mpos = missile_pos(assigned_m, t)
                center = cloud_center_at(det, t_det, t)
                if center is None:
                    continue
                if is_blocking_cloud(center, mpos, target_center, tol_extra=0.0):
                    block_dur += dt
            rows.append({
                "无人机编号": dname,
                "无人机运动方向（度）": round(heading, 3),
                "无人机运动速度 (m/s)": round(speed, 3),
                "烟幕干扰弹编号": f"{dname}-B{bi+1}",
                "烟幕干扰弹投放点x (m)": round(float(drop[0]), 3),
                "烟幕干扰弹投放点y (m)": round(float(drop[1]), 3),
                "烟幕干扰弹投放点z (m)": round(float(drop[2]), 3),
                "烟幕干扰弹起爆点x (m)": round(float(det[0]), 3),
                "烟幕干扰弹起爆点y (m)": round(float(det[1]), 3),
                "烟幕干扰弹起爆点z (m)": round(float(det[2]), 3),
                "有效干扰时长 (s)": round(block_dur, 4),
                "干扰的导弹编号": assigned_m
            })
    df = pd.DataFrame(rows, columns=[
        "无人机编号", "无人机运动方向（度）", "无人机运动速度 (m/s)", "烟幕干扰弹编号",
        "烟幕干扰弹投放点x (m)", "烟幕干扰弹投放点y (m)", "烟幕干扰弹投放点z (m)",
        "烟幕干扰弹起爆点x (m)", "烟幕干扰弹起爆点y (m)", "烟幕干扰弹起爆点z (m)",
        "有效干扰时长 (s)", "干扰的导弹编号"
    ])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_excel(out_path, index=False)
    return df


# ---------------- Interface function ----------------
def optimize_strategy(iters=600, popsize=60, F=0.8, CR=0.7, workers=None, dt=DEFAULT_DT,
                     tol_extra=6.0, angular_w=0.6, checkpoint_dir="Data/Q5", refine=True, use_cupy=False):
    global NUM_VARS, VAR_LOW, VAR_HIGH, guided_headings, DEFAULT_DT, missile_traj_cache
    DEFAULT_DT = dt
    if workers is None:
        workers = max(1, min(cpu_count() - 1, 8))

    # Precompute missile trajectories
    missile_traj_cache = {}
    max_missile_time = max(missile_time_to_origin(m) for m in missile_names)
    times = np.arange(0.0, min(max_missile_time, TIME_HORIZON) + 1e-9, dt)
    for m in missile_names:
        cache_key = (m, dt, float(max_missile_time))
        missile_traj_cache[cache_key] = missile_pos(m, times, use_cupy=use_cupy)

    # Assignments
    drone_assignments = assign_drones_to_missiles()
    guided_headings = compute_guided_headings()

    # Variable bounds
    num_drones = len(drone_names)
    NUM_VARS = num_drones * (2 + 2 * MAX_BOMBS)
    VAR_LOW = np.zeros(NUM_VARS)
    VAR_HIGH = np.zeros(NUM_VARS)
    for di, dname in enumerate(drone_names):
        base = di * (2 + 2 * MAX_BOMBS)
        dist_to_target = np.linalg.norm(drones_init[dname] - target_center)
        t_max = dist_to_target / drone_vmin
        VAR_LOW[base + 0], VAR_HIGH[base + 0] = 0.0, 360.0
        VAR_LOW[base + 1], VAR_HIGH[base + 1] = drone_vmin, drone_vmax
        for bi in range(MAX_BOMBS):
            VAR_LOW[base + 2 + 2 * bi] = 5.0
            VAR_HIGH[base + 2 + 2 * bi] = min(80.0, t_max)
            VAR_LOW[base + 2 + 2 * bi + 1] = 0.2
            VAR_HIGH[base + 2 + 2 * bi + 1] = 40.0

    pop_init = greedy_initial_pop(popsize=popsize)
    pop_init = np.clip(pop_init, VAR_LOW, VAR_HIGH)

    fitness_fn = partial(evaluate_chromosome, drone_assignments=drone_assignments, dt=dt, tol_extra=tol_extra,
                         angular_weight=angular_w, use_cupy=use_cupy)

    best, best_score = differential_evolution(pop_init, fitness_fn, iters=iters, F=F, CR=CR,
                                             workers=workers, checkpoint_dir=checkpoint_dir,
                                             checkpoint_every=max(1, iters // 10))

    if refine and CMA_OK:
        refined, refined_score = refine_with_cmaes(best, fitness_fn, maxiter=200, workers=workers)
        if refined_score > best_score:
            best, best_score = refined, refined_score

    df = save_result_excel(best, out_path="Data/result3.xlsx", dt=dt, drone_assignments=drone_assignments)
    return best, best_score, df


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--pop", type=int, default=48)
    parser.add_argument("--F", type=float, default=0.8)
    parser.add_argument("--CR", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=6.0)
    parser.add_argument("--angw", type=float, default=0.6)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--use-cupy", action="store_true", help="Enable CuPy for GPU acceleration")
    args = parser.parse_args()

    np.random.seed(12345)
    start = time.time()
    best_chrom, best_score, df = optimize_strategy(
        iters=args.iters, popsize=args.pop, F=args.F, CR=args.CR,
        workers=args.workers, dt=args.dt, tol_extra=args.tol, angular_w=args.angw,
        checkpoint_dir="Data/Q5", refine=not args.no_refine, use_cupy=args.use_cupy
    )
    end = time.time()
    print(f"Done. best_score={best_score:.3f}, time={(end-start):.1f}s")
    print(df.head(20))