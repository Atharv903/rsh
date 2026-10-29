"""
Phase 10C: Independent-Drive Validation Benchmark for IO-VNBD.

Evaluates:
- Baselines: Zero-Speed, Constant-Velocity (Last-Known-Speed)
- 9 Frozen Phase 8 GPU Checkpoints:
  - CNN-GRU (seeds 42, 123, 456)
  - Mamba-2 / SSD (seeds 42, 123, 456)
  - ModernTCN (seeds 42, 123, 456)
On 15 matched GNSS outage scenarios from Driver B (M), a genuinely independent drive.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.cnn_gru import CNNGRUModel
from src.models.modern_tcn import ModernTCNModel
from src.models.mamba2 import Mamba2Model
from src.corrected_synchronization import extract_aligned_segments_for_drive
from src.outage_estimator import (
    PreOutageState,
    estimate_initial_heading_from_pre_outage_gps,
)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    mtype = cfg.get("model_type")
    margs = cfg.get("model", {})
    
    # In some checkpoints, mtype might be in checkpoint directly
    if not mtype:
        p_str = str(ckpt_path).lower()
        if "cnngru" in p_str:
            mtype = "CNNGRUModel"
            margs = {"in_channels": 6, "hidden_dim": 128, "num_layers": 2, "out_dim": 1, "dropout": 0.1}
        elif "moderntcn" in p_str:
            mtype = "ModernTCNModel"
            margs = {"in_channels": 6, "num_classes": 1, "stem_dim": 64, "depths": [2, 2], "dims": [64, 128], "kernel_size": 7, "dropout": 0.1}
        elif "mamba2" in p_str:
            mtype = "Mamba2Model"
            margs = {"in_channels": 6, "d_model": 128, "n_layers": 4, "d_state": 16, "out_dim": 1, "dropout": 0.1}
            
    if mtype == "CNNGRUModel":
        model = CNNGRUModel(**margs)
    elif mtype == "ModernTCNModel":
        model = ModernTCNModel(**margs)
    elif mtype == "Mamba2Model":
        model = Mamba2Model(**margs)
    else:
        raise ValueError(f"Unknown model_type: {mtype}")
        
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, mtype


def run_fast_causal_dead_reckoning(
    pre_state: PreOutageState,
    in_outage_df: pd.DataFrame,
    pred_speed: np.ndarray,
    method_name: str,
    scenario_id: str,
    drive_id: str,
    heading_sign: float = +1.0,  # +1.0 for original repo convention, -1.0 for right-hand navigation
):
    n_steps = len(in_outage_df)
    time_arr = in_outage_df["time_s"].values
    dt_arr = np.diff(time_arr, prepend=time_arr[0])
    dt_arr[0] = 0.0
    
    ref_east = in_outage_df["ref_east_m"].values
    ref_north = in_outage_df["ref_north_m"].values
    ref_speed = in_outage_df["ref_speed_mps"].values
    
    seg_dists = np.sqrt(np.diff(ref_east)**2 + np.diff(ref_north)**2)
    total_length_m = float(np.sum(seg_dists))
    
    gyro_yaw_rate = (in_outage_df["gyro_pitch"].values.astype(np.float32) - pre_state.gyro_bias_rads)
    
    pred_east = np.zeros(n_steps, dtype=np.float64)
    pred_north = np.zeros(n_steps, dtype=np.float64)
    pred_heading = np.zeros(n_steps, dtype=np.float64)
    
    pred_east[0] = pre_state.pos_east_m
    pred_north[0] = pre_state.pos_north_m
    pred_heading[0] = np.deg2rad(pre_state.heading_deg)
    
    for k in range(1, n_steps):
        dt = dt_arr[k]
        # Heading propagation
        d_psi = heading_sign * float(gyro_yaw_rate[k] * dt)
        pred_heading[k] = (pred_heading[k-1] + d_psi) % (2.0 * np.pi)
        
        # Dead-reckoning integration
        v_e = pred_speed[k] * np.sin(pred_heading[k])
        v_n = pred_speed[k] * np.cos(pred_heading[k])
        
        pred_east[k] = pred_east[k-1] + v_e * dt
        pred_north[k] = pred_north[k-1] + v_n * dt
        
    speed_err = pred_speed - ref_speed
    speed_mae = float(np.mean(np.abs(speed_err)))
    speed_rmse = float(np.sqrt(np.mean(speed_err**2)))
    speed_bias = float(np.mean(speed_err))
    
    pos_err = np.sqrt((pred_east - ref_east)**2 + (pred_north - ref_north)**2)
    ate_rmse = float(np.sqrt(np.mean(pos_err**2)))
    final_drift_m = float(pos_err[-1])
    drift_pct = float((final_drift_m / max(total_length_m, 1.0)) * 100.0)
    
    return {
        "scenario_id": scenario_id,
        "drive_id": drive_id,
        "duration_s": float(time_arr[-1] - time_arr[0]),
        "method": method_name,
        "heading_convention": "original_repo (+1)" if heading_sign > 0 else "sign_corrected (-1)",
        "total_length_m": total_length_m,
        "speed_mae_mps": speed_mae,
        "speed_rmse_mps": speed_rmse,
        "speed_bias_mps": speed_bias,
        "ate_rmse_m": ate_rmse,
        "final_drift_m": final_drift_m,
        "drift_pct": drift_pct,
    }


def main():
    print("=" * 80)
    print("PHASE 10C: INDEPENDENT-DRIVE VALIDATION BENCHMARK (DRIVER B - M)")
    print("=" * 80)
    
    raw_m_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)"
    segs, _ = extract_aligned_segments_for_drive("M", raw_dir=raw_m_dir)
    drive_df = segs[1]  # M_seg1
    print(f"Loaded Driver B Segment 1: {len(drive_df)} samples ({len(drive_df)*0.1/60:.1f} mins).")
    
    scenarios_path = "experiments/phase10_independent_validation/scenarios/driver_b_outage_scenarios.json"
    with open(scenarios_path) as f:
        scenarios = json.load(f)
    print(f"Loaded {len(scenarios)} matched outage scenarios.")
    
    # Frozen Sensor Normalizer from Train Set (S1+S2)
    norm_meta = json.load(open("data/processed/sensor_normalizer_grouped.json"))
    norm_params = norm_meta["params"]
    feat_channels = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
    norm_mean = np.array([norm_params[ch]["mean"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
    norm_std = np.array([norm_params[ch]["std"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
    
    # Load 9 Frozen Phase 8 GPU Checkpoints
    ckpt_dir = Path("experiments/phase8_corrected_speed_models/checkpoints")
    models = {}
    for d in sorted(ckpt_dir.iterdir()):
        pt_path = d / "best_model.pt"
        if pt_path.exists():
            m, mtype = load_model(pt_path)
            models[d.name] = (m, mtype)
    print(f"Loaded {len(models)} frozen GPU checkpoints.")
    
    results = []
    t_start = time.time()
    
    for sc_idx, sc in enumerate(scenarios):
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
        
        pre_state = PreOutageState(
            t0_s=float(pre_df["time_s"].iloc[-1]),
            pos_east_m=pos_e0,
            pos_north_m=pos_n0,
            speed_mps=last_spd,
            heading_deg=heading_deg,
            gyro_bias_rads=gyro_bias,
        )
        
        # 1. Baseline: Zero-Speed
        zero_speed_arr = np.zeros(n_steps, dtype=np.float32)
        zero_speed_arr[0] = last_spd
        
        # 2. Baseline: Constant-Velocity
        const_speed_arr = np.full(n_steps, last_spd, dtype=np.float32)
        
        # Build batched causal sliding windows for the outage steps
        raw_imu = in_outage_df[feat_channels].values.astype(np.float32)
        window_size = 100
        
        # Construct pre-padded IMU stream including pre_df history
        pre_imu = pre_df[feat_channels].values.astype(np.float32)
        full_imu = np.vstack([pre_imu, raw_imu])
        # In full_imu, outage step k corresponds to index len(pre_imu) + k
        # Window ending at step k is full_imu[len(pre_imu) + k - window_size + 1 : len(pre_imu) + k + 1]
        
        windows = []
        for k in range(n_steps):
            w_end = len(pre_imu) + k + 1
            w_start = w_end - window_size
            w = full_imu[w_start:w_end]
            w_norm = (w - norm_mean) / norm_std
            windows.append(w_norm)
            
        windows_tensor = torch.from_numpy(np.array(windows, dtype=np.float32))
        
        # Precompute neural model predictions in batched forward pass
        model_speeds = {}
        for m_name, (m, _) in models.items():
            with torch.no_grad():
                preds = m(windows_tensor).squeeze(-1).cpu().numpy()
            preds = np.maximum(0.0, preds)
            preds[0] = last_spd  # At t0, speed is exactly known from pre-outage GNSS
            model_speeds[m_name] = preds
            
        # Run evaluations for both heading conventions
        for h_sign in [+1.0, -1.0]:
            # Zero-speed
            r_z = run_fast_causal_dead_reckoning(pre_state, in_outage_df, zero_speed_arr, "Zero-Speed", sid, "DriverB_M", h_sign)
            results.append(r_z)
            
            # Constant-velocity
            r_c = run_fast_causal_dead_reckoning(pre_state, in_outage_df, const_speed_arr, "Constant-Velocity", sid, "DriverB_M", h_sign)
            results.append(r_c)
            
            # Neural models
            for m_name, preds in model_speeds.items():
                r_m = run_fast_causal_dead_reckoning(pre_state, in_outage_df, preds, m_name, sid, "DriverB_M", h_sign)
                results.append(r_m)
                
        print(f"[{sc_idx+1:2d}/{len(scenarios)}] Completed scenario {sid} ({dur}s).")
        
    elapsed = time.time() - t_start
    print(f"All evaluations completed in {elapsed:.2f} seconds.")
    
    # Save raw detailed CSV
    df_results = pd.DataFrame(results)
    out_csv = Path("experiments/phase10_independent_validation/results/phase10_independent_outage_benchmarks.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Saved detailed results to {out_csv}")
    
    # Grouped Summary
    summary = df_results.groupby(["heading_convention", "duration_s", "method"])[
        ["speed_mae_mps", "speed_rmse_mps", "speed_bias_mps", "ate_rmse_m", "final_drift_m", "drift_pct"]
    ].mean().reset_index()
    
    sum_csv = Path("experiments/phase10_independent_validation/results/phase10_aggregated_summary.csv")
    summary.to_csv(sum_csv, index=False)
    print(f"Saved aggregated summary to {sum_csv}")
    
    # Display Original Repo Protocol Benchmark Table
    print("\n" + "=" * 95)
    print("INDEPENDENT-DRIVE BENCHMARK (DRIVER B - M) — ORIGINAL PROTOCOL (+1)")
    print("=" * 95)
    orig_sum = summary[summary["heading_convention"] == "original_repo (+1)"].sort_values(["duration_s", "final_drift_m"])
    print(f"{'Duration':10s} | {'Method':25s} | {'Speed MAE':10s} | {'ATE RMSE':10s} | {'Final Drift':12s} | {'Drift %':8s}")
    print("-" * 95)
    for _, row in orig_sum.iterrows():
        print(f"{int(row['duration_s']):2d} seconds  | {row['method']:25s} | {row['speed_mae_mps']:8.2f} m/s | {row['ate_rmse_m']:8.2f} m | {row['final_drift_m']:10.2f} m | {row['drift_pct']:6.1f} %")


if __name__ == "__main__":
    main()
