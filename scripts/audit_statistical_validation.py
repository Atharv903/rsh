"""
Phase 11C: Statistical Validation & Mathematical Decomposition of Navigation Drift.

Performs:
1. Paired t-tests and Wilcoxon signed-rank tests comparing neural models vs Constant Velocity.
2. 95% Confidence Intervals for final drift and drift reduction.
3. Vector decomposition: Along-track error (speed magnitude error) vs Cross-track error (heading/lateral error).
4. Mathematical resolution of the 60s Speed MAE vs Drift paradox.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.evaluate_phase10_independent import load_model
from src.corrected_synchronization import extract_aligned_segments_for_drive
from src.outage_estimator import PreOutageState, estimate_initial_heading_from_pre_outage_gps

def run_statistical_audit():
    df = pd.read_csv("experiments/phase10_independent_validation/results/phase10_independent_outage_benchmarks.csv")
    df["dur_nom"] = df["duration_s"].apply(lambda d: 10 if d < 20 else (30 if d < 45 else 60))

    sub = df[df["heading_convention"] == "sign_corrected (-1)"].copy()
    
    def get_arch(m):
        if "cnngru" in m: return "CNN-GRU"
        if "mamba2" in m: return "Mamba-2"
        if "moderntcn" in m: return "ModernTCN"
        return m
    sub["arch"] = sub["method"].apply(get_arch)

    # Pivot table of drift per scenario and architecture
    arch_pivot = sub.groupby(["dur_nom", "scenario_id", "arch"])["final_drift_m"].mean().unstack("arch").reset_index()

    print("=" * 85)
    print("PHASE 11C: STATISTICAL SIGNIFICANCE & HYPOTHESIS TESTING (MODERNTCN VS CV)")
    print("=" * 85)

    stat_results = []

    for dur in [10, 30, 60]:
        dur_df = arch_pivot[arch_pivot["dur_nom"] == dur]
        cv_drift = dur_df["Constant-Velocity"].values
        nn_drift = dur_df["ModernTCN"].values
        diff = cv_drift - nn_drift  # Positive means ModernTCN wins (less drift)

        t_stat, p_val_t = stats.ttest_rel(cv_drift, nn_drift)
        try:
            w_stat, p_val_w = stats.wilcoxon(cv_drift, nn_drift)
        except Exception as e:
            w_stat, p_val_w = np.nan, np.nan

        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        se_diff = std_diff / np.sqrt(len(diff))
        ci95 = stats.t.interval(0.95, df=len(diff)-1, loc=mean_diff, scale=se_diff)

        stat_results.append({
            "duration_s": dur,
            "cv_drift_mean": float(np.mean(cv_drift)),
            "nn_drift_mean": float(np.mean(nn_drift)),
            "mean_advantage_m": float(mean_diff),
            "std_advantage_m": float(std_diff),
            "ci95_low": float(ci95[0]),
            "ci95_high": float(ci95[1]),
            "t_statistic": float(t_stat),
            "p_value_paired_t": float(p_val_t),
            "wilcoxon_stat": float(w_stat),
            "p_value_wilcoxon": float(p_val_w),
            "scenarios_won_by_nn": int(np.sum(diff > 0)),
            "total_scenarios": len(diff),
        })

        print(f"\n--- {dur}-SECOND OUTAGES (N = {len(diff)} scenarios) ---")
        print(f"  Constant-Velocity Drift : {np.mean(cv_drift):6.2f} +/- {np.std(cv_drift, ddof=1):5.2f} m")
        print(f"  ModernTCN Drift         : {np.mean(nn_drift):6.2f} +/- {np.std(nn_drift, ddof=1):5.2f} m")
        print(f"  Mean ModernTCN Advantage: {mean_diff:+6.2f} m (95% CI: [{ci95[0]:+6.2f}, {ci95[1]:+6.2f}] m)")
        print(f"  Scenarios Won by ModernTCN: {np.sum(diff > 0)} / {len(diff)}")
        print(f"  Paired t-test           : t = {t_stat:+.3f}, p = {p_val_t:.4f}")
        print(f"  Wilcoxon signed-rank    : W = {w_stat:.1f}, p = {p_val_w:.4f}")

    stat_df = pd.DataFrame(stat_results)
    out_stat_csv = Path("experiments/phase11_navigation_integrity/results/phase11_statistical_significance.csv")
    stat_df.to_csv(out_stat_csv, index=False)
    print(f"\nSaved statistical validation table to {out_stat_csv}")

    # Vector Error Decomposition
    print("\n" + "=" * 85)
    print("VECTOR ERROR DECOMPOSITION: ALONG-TRACK (SPEED) VS CROSS-TRACK (HEADING)")
    print("=" * 85)

    raw_m_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)"
    segs, _ = extract_aligned_segments_for_drive("M", raw_dir=raw_m_dir)
    drive_df = segs[1]

    with open("experiments/phase10_independent_validation/scenarios/driver_b_outage_scenarios.json") as f:
        scenarios = json.load(f)

    # Load ModernTCN seed 42 checkpoint
    ckpt_path = "experiments/phase8_corrected_speed_models/checkpoints/moderntcnmodel_seed42/best_model.pt"
    model, _ = load_model(ckpt_path)

    norm_meta = json.load(open("data/processed/sensor_normalizer_grouped.json"))
    norm_params = norm_meta["params"]
    feat_channels = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
    norm_mean = np.array([norm_params[ch]["mean"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
    norm_std = np.array([norm_params[ch]["std"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)

    decomp_results = []

    for sc in scenarios:
        st, end = sc["start_idx"], sc["end_idx"]
        dur = sc["duration_s"]
        sid = sc["scenario_id"]
        pre_st = max(0, st - sc["pre_window_samples"])
        
        pre_df = drive_df.iloc[pre_st:st].copy()
        in_outage_df = drive_df.iloc[st:end].copy()
        n_steps = len(in_outage_df)
        
        heading_deg, last_spd = estimate_initial_heading_from_pre_outage_gps(pre_df)
        pos_e0 = float(pre_df["ref_east_m"].iloc[-1])
        pos_n0 = float(pre_df["ref_north_m"].iloc[-1])
        gyro_bias = float(np.median(pre_df["gyro_pitch"]))
        
        time_arr = in_outage_df["time_s"].values
        dt_arr = np.diff(time_arr, prepend=time_arr[0])
        dt_arr[0] = 0.0
        
        ref_east = in_outage_df["ref_east_m"].values
        ref_north = in_outage_df["ref_north_m"].values
        ref_heading = np.deg2rad(in_outage_df["ref_heading_deg"].values)
        
        # Neural speed
        pre_imu = pre_df[feat_channels].values.astype(np.float32)
        raw_imu = in_outage_df[feat_channels].values.astype(np.float32)
        full_imu = np.vstack([pre_imu, raw_imu])
        
        windows = []
        for k in range(n_steps):
            w_end = len(pre_imu) + k + 1
            w_start = w_end - 100
            w = full_imu[w_start:w_end]
            windows.append((w - norm_mean) / norm_std)
        tensor_in = torch.from_numpy(np.array(windows, dtype=np.float32))
        with torch.no_grad():
            nn_speed = np.maximum(0.0, model(tensor_in).squeeze(-1).cpu().numpy())
        nn_speed[0] = last_spd
        
        const_speed = np.full(n_steps, last_spd, dtype=np.float32)
        
        # Dead reckoning integration
        gyro_yaw_rate = (in_outage_df["gyro_pitch"].values.astype(np.float32) - gyro_bias)
        
        def integrate_and_decompose(speeds):
            pe = np.zeros(n_steps)
            pn = np.zeros(n_steps)
            head = np.zeros(n_steps)
            pe[0] = pos_e0
            pn[0] = pos_n0
            head[0] = np.deg2rad(heading_deg)
            for k in range(1, n_steps):
                dt = dt_arr[k]
                d_psi = -1.0 * float(gyro_yaw_rate[k] * dt)
                head[k] = (head[k-1] + d_psi) % (2.0 * np.pi)
                pe[k] = pe[k-1] + speeds[k] * np.sin(head[k]) * dt
                pn[k] = pn[k-1] + speeds[k] * np.cos(head[k]) * dt
                
            delta_e = pe[-1] - ref_east[-1]
            delta_n = pn[-1] - ref_north[-1]
            tot_drift = np.hypot(delta_e, delta_n)
            
            # Unit along-track and cross-track at endpoint
            psi_f = ref_heading[-1]
            u_along = np.array([np.sin(psi_f), np.cos(psi_f)])
            u_cross = np.array([np.cos(psi_f), -np.sin(psi_f)])
            
            along_err = delta_e * u_along[0] + delta_n * u_along[1]
            cross_err = delta_e * u_cross[0] + delta_n * u_cross[1]
            return tot_drift, along_err, cross_err
            
        cv_drift, cv_along, cv_cross = integrate_and_decompose(const_speed)
        nn_drift, nn_along, nn_cross = integrate_and_decompose(nn_speed)
        
        decomp_results.append({
            "scenario_id": sid,
            "duration_s": dur,
            "cv_drift_m": cv_drift,
            "cv_along_track_err_m": cv_along,
            "cv_cross_track_err_m": cv_cross,
            "nn_drift_m": nn_drift,
            "nn_along_track_err_m": nn_along,
            "nn_cross_track_err_m": nn_cross,
            "drift_reduction_m": cv_drift - nn_drift,
        })
        
    decomp_df = pd.DataFrame(decomp_results)
    out_decomp_csv = Path("experiments/phase11_navigation_integrity/results/phase11_vector_error_decomposition.csv")
    decomp_df.to_csv(out_decomp_csv, index=False)
    print(f"Saved vector error decomposition to {out_decomp_csv}")
    
    print("\n--- 60-Second Outage Vector Error Decomposition ---")
    for _, r in decomp_df[decomp_df["duration_s"] == 60].iterrows():
        print(f"Scenario: {r['scenario_id']}")
        print(f"  Constant-Velocity : Total Drift = {r['cv_drift_m']:6.2f}m | Along-Track = {r['cv_along_track_err_m']:+6.2f}m | Cross-Track = {r['cv_cross_track_err_m']:+6.2f}m")
        print(f"  ModernTCN (s42)   : Total Drift = {r['nn_drift_m']:6.2f}m | Along-Track = {r['nn_along_track_err_m']:+6.2f}m | Cross-Track = {r['nn_cross_track_err_m']:+6.2f}m")
        print(f"  -> Drift Reduction: {r['drift_reduction_m']:+6.2f}m\n")

if __name__ == "__main__":
    run_statistical_audit()
