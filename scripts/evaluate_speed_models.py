"""
Comprehensive Evaluation Script for Phase 8 Corrected Speed Models.
Evaluates CNN-GRU, ModernTCN, and Mamba-2 (custom SSD) across:
1. Speed estimation accuracy on Val and Test sets (MAE, RMSE, Bias, by speed bin & motion dynamic).
2. Trajectory dead reckoning on matched 10s, 30s, and 60s GNSS outages.
3. Publication-quality trajectory and loss curve figures.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.models.cnn_gru import CNNGRUModel
from src.models.modern_tcn import ModernTCNModel
from src.models.mamba2 import Mamba2Model
from src.corrected_synchronization import extract_aligned_segments_for_drive
from src.outage_estimator import (
    PreOutageState,
    estimate_initial_heading_from_pre_outage_gps,
    run_causal_dead_reckoning,
    make_zero_speed_estimator,
    make_last_known_speed_estimator,
    make_imu_kinematic_estimator,
)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})


def load_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    m_type = config.get("model_type", "CNNGRUModel")
    m_args = config.get("model", {})
    
    if m_type == "CNNGRUModel":
        model = CNNGRUModel(**m_args)
    elif m_type == "ModernTCNModel":
        model = ModernTCNModel(**m_args)
    elif m_type == "Mamba2Model":
        model = Mamba2Model(**m_args)
    else:
        raise ValueError(f"Unknown model type: {m_type}")
        
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def evaluate_speed_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    err = preds - targets
    abs_err = np.abs(err)
    
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    
    # Speed Bins
    bins = {
        "stationary": targets < 0.5,
        "low_speed": (targets >= 0.5) & (targets < 5.0),
        "med_speed": (targets >= 5.0) & (targets < 15.0),
        "high_speed": targets >= 15.0,
    }
    
    bin_mae = {}
    for b_name, mask in bins.items():
        if np.sum(mask) > 0:
            bin_mae[f"mae_{b_name}"] = float(np.mean(abs_err[mask]))
        else:
            bin_mae[f"mae_{b_name}"] = 0.0
            
    # Motion dynamic: acceleration vs deceleration vs cruising
    dt_targets = np.diff(targets, prepend=targets[0])
    accel_mask = dt_targets > 0.5
    decel_mask = dt_targets < -0.5
    cruise_mask = np.abs(dt_targets) <= 0.5
    
    dyn_mae = {
        "mae_accel": float(np.mean(abs_err[accel_mask])) if np.sum(accel_mask) > 0 else 0.0,
        "mae_decel": float(np.mean(abs_err[decel_mask])) if np.sum(decel_mask) > 0 else 0.0,
        "mae_cruise": float(np.mean(abs_err[cruise_mask])) if np.sum(cruise_mask) > 0 else 0.0,
    }
    
    res = {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
    }
    res.update(bin_mae)
    res.update(dyn_mae)
    return res


def run_full_evaluation():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE PHASE 8 BENCHMARK EVALUATION")
    print("=" * 70)
    
    device = torch.device("cpu")
    exp_dir = Path("experiments/phase8_corrected_speed_models")
    ckpt_dir = exp_dir / "checkpoints"
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    # Load Validation and Test sets
    val_data = np.load("data/processed/val_windows_grouped.npz", allow_pickle=True)
    test_data = np.load("data/processed/test_windows_grouped.npz", allow_pickle=True)
    
    X_val = torch.from_numpy(val_data["X"].astype(np.float32))
    y_val = val_data["y"].flatten().astype(np.float32)
    
    X_test = torch.from_numpy(test_data["X"].astype(np.float32))
    y_test = test_data["y"].flatten().astype(np.float32)
    
    train_data = np.load("data/processed/train_windows_grouped.npz", allow_pickle=True)
    train_mean_spd = float(train_data["y"].mean())
    
    # Identify all trained checkpoint directories
    run_dirs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir()])
    print(f"Found {len(run_dirs)} trained model checkpoints in {ckpt_dir}")
    
    model_eval_rows = []
    all_predictions = {}
    loaded_models = {}
    
    # -------------------------------------------------------------
    # Evaluate Speed Estimation on Validation and Test Splits
    # -------------------------------------------------------------
    for rdir in run_dirs:
        ckpt_pt = rdir / "best_model.pt"
        if not ckpt_pt.exists():
            continue
            
        model, cfg = load_checkpoint(ckpt_pt, device)
        name = rdir.name
        loaded_models[name] = model
        
        # Predict on Val
        with torch.no_grad():
            val_preds = []
            for i in range(0, len(X_val), 256):
                val_preds.append(model(X_val[i:i+256]).cpu().numpy())
            val_p = np.clip(np.concatenate(val_preds).flatten(), 0.0, None)
            
        # Predict on Test
        with torch.no_grad():
            test_preds = []
            for i in range(0, len(X_test), 256):
                test_preds.append(model(X_test[i:i+256]).cpu().numpy())
            test_p = np.clip(np.concatenate(test_preds).flatten(), 0.0, None)
            
        all_predictions[name] = {
            "val_preds": val_p,
            "test_preds": test_p,
        }
        
        # Metrics
        val_m = evaluate_speed_metrics(val_p, y_val)
        test_m = evaluate_speed_metrics(test_p, y_test)
        
        # Model metadata
        meta_json = rdir / "run_metadata.json"
        meta = json.load(open(meta_json)) if meta_json.exists() else {}
        
        row = {
            "model_run": name,
            "model_type": cfg.get("model_type", ""),
            "seed": cfg.get("seed", 0),
            "param_count": meta.get("param_count", 0),
            "val_mae": val_m["mae"],
            "val_rmse": val_m["rmse"],
            "val_bias": val_m["bias"],
            "test_mae": test_m["mae"],
            "test_rmse": test_m["rmse"],
            "test_bias": test_m["bias"],
            "test_mae_stationary": test_m["mae_stationary"],
            "test_mae_low_speed": test_m["mae_low_speed"],
            "test_mae_med_speed": test_m["mae_med_speed"],
            "test_mae_high_speed": test_m["mae_high_speed"],
            "test_mae_accel": test_m["mae_accel"],
            "test_mae_decel": test_m["mae_decel"],
            "test_mae_cruise": test_m["mae_cruise"],
            "latency_b1_ms": meta.get("latency_benchmark", {}).get("latency_b1_mean_ms", 0.0),
        }
        model_eval_rows.append(row)
        print(f"[{name}] Val MAE: {val_m['mae']:.3f} m/s | Test MAE: {test_m['mae']:.3f} m/s (RMSE: {test_m['rmse']:.3f})", flush=True)
        
    # Baseline rows on Test
    zero_m = evaluate_speed_metrics(np.zeros_like(y_test), y_test)
    tmean_m = evaluate_speed_metrics(np.full_like(y_test, train_mean_spd), y_test)
    # Naive stride-1 persistence
    persist_preds = np.zeros_like(y_test)
    persist_preds[0] = y_test[0]
    persist_preds[1:] = y_test[:-1]
    persist_m = evaluate_speed_metrics(persist_preds, y_test)
    
    baseline_rows = [
        {"model_run": "Zero-Speed Baseline", "model_type": "Baseline", "test_mae": zero_m["mae"], "test_rmse": zero_m["rmse"], "test_bias": zero_m["bias"]},
        {"model_run": "Train-Mean Baseline", "model_type": "Baseline", "test_mae": tmean_m["mae"], "test_rmse": tmean_m["rmse"], "test_bias": tmean_m["bias"]},
        {"model_run": "Previous-Window Persistence", "model_type": "Baseline (Leakage Warning)", "test_mae": persist_m["mae"], "test_rmse": persist_m["rmse"], "test_bias": persist_m["bias"]},
    ]
    for b in baseline_rows:
        print(f"[{b['model_run']}] Test MAE: {b['test_mae']:.3f} m/s (RMSE: {b['test_rmse']:.3f})", flush=True)
    
    df_eval = pd.DataFrame(model_eval_rows + baseline_rows)
    df_eval.to_csv(results_dir / "model_speed_metrics.csv", index=False)
    
    # -------------------------------------------------------------
    # Trajectory Outage Dead-Reckoning Benchmark (10s, 30s, 60s)
    # -------------------------------------------------------------
    print("\n--- Running Trajectory Outage Benchmark on Holdout (Drive S4) ---", flush=True)
    segs, _ = extract_aligned_segments_for_drive("S4", raw_dir="data/raw")
    test_df = segs[0]
    
    # Load Sensor Normalizer parameters so neural speed models receive normalized IMU windows
    norm_meta = json.load(open("data/processed/sensor_normalizer_grouped.json"))
    norm_params = norm_meta["params"]
    feat_channels = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
    norm_mean = np.array([norm_params[ch]["mean"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
    norm_std = np.array([norm_params[ch]["std"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
    
    np.random.seed(42)
    durations = [10, 30, 60]
    scenarios = []
    total_len = len(test_df)
    
    for dur in durations:
        n_samples = dur * 10
        valid_starts = []
        for cand in range(500, total_len - n_samples - 500, 1000):
            seg_spd = test_df["ref_speed_mps"].iloc[cand : cand + n_samples].values
            if np.mean(seg_spd) > 3.0:
                valid_starts.append(cand)
        selected_starts = np.random.choice(valid_starts, size=min(5, len(valid_starts)), replace=False)
        selected_starts.sort()
        for idx, st in enumerate(selected_starts):
            scenarios.append({
                "scenario_id": f"S4_{dur}s_trial{idx+1}",
                "duration_s": dur,
                "start_idx": int(st),
                "end_idx": int(st + n_samples),
                "pre_window_samples": 100,
            })
            
    outage_rows = []
    trajectory_runs = {}
    
    # Evaluate each trained model and baselines on the matched scenarios
    for sc in scenarios:
        st, end = sc["start_idx"], sc["end_idx"]
        pre_st = max(0, st - sc["pre_window_samples"])
        pre_df = test_df.iloc[pre_st:st].copy()
        in_outage_df = test_df.iloc[st:end].copy()
        
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
        
        # Baselines
        r_zero = run_causal_dead_reckoning(pre_state, in_outage_df, make_zero_speed_estimator(), "Zero-Speed", sc["scenario_id"], "S4")
        r_const = run_causal_dead_reckoning(pre_state, in_outage_df, make_last_known_speed_estimator(last_spd), "Constant-Velocity", sc["scenario_id"], "S4")
        
        outage_rows.append({"scenario_id": sc["scenario_id"], "duration_s": sc["duration_s"], "method": "Zero-Speed", "ate_rmse": r_zero.ate_rmse_m, "final_drift_m": r_zero.final_drift_m, "drift_pct": r_zero.drift_pct})
        outage_rows.append({"scenario_id": sc["scenario_id"], "duration_s": sc["duration_s"], "method": "Constant-Velocity", "ate_rmse": r_const.ate_rmse_m, "final_drift_m": r_const.final_drift_m, "drift_pct": r_const.drift_pct})
        
        # Neural Models
        for m_name, model in loaded_models.items():
            def make_model_fn(m):
                def _est(w):
                    w_norm = (w - norm_mean) / norm_std
                    tensor_in = torch.from_numpy(w_norm.astype(np.float32)).unsqueeze(0)
                    with torch.no_grad():
                        return float(m(tensor_in).item())
                return _est
                
            r_m = run_causal_dead_reckoning(
                pre_state,
                in_outage_df,
                make_model_fn(model),
                m_name,
                sc["scenario_id"],
                "S4",
            )
            outage_rows.append({
                "scenario_id": sc["scenario_id"],
                "duration_s": sc["duration_s"],
                "method": m_name,
                "ate_rmse": r_m.ate_rmse_m,
                "final_drift_m": r_m.final_drift_m,
                "drift_pct": r_m.drift_pct,
            })
            
            # Store 60s trial 1 for plotting
            if sc["scenario_id"] == "S4_60s_trial1":
                trajectory_runs[m_name] = r_m
                trajectory_runs["ground_truth"] = (r_m.ref_trajectory_east, r_m.ref_trajectory_north)
                trajectory_runs["constant_velocity"] = r_const
                
    df_outage = pd.DataFrame(outage_rows)
    df_outage.to_csv(results_dir / "model_outage_metrics.csv", index=False)
    
    # -------------------------------------------------------------
    # Publication Figures
    # -------------------------------------------------------------
    # 1. Trajectory comparison plot for 60s outage
    if "ground_truth" in trajectory_runs:
        gt_e, gt_n = trajectory_runs["ground_truth"]
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.plot(gt_e, gt_n, color="black", linewidth=2.5, label="Ground Truth Vehicle Reference")
        
        r_const = trajectory_runs["constant_velocity"]
        ax.plot(r_const.est_trajectory_east, r_const.est_trajectory_north, color="gray", linestyle="--", linewidth=1.8, label=f"Constant-Velocity Baseline (Drift: {r_const.final_drift_m:.1f}m / {r_const.drift_pct:.1f}%)")
        
        # Plot seed 42 of each architecture
        rep_models = [k for k in trajectory_runs if "seed42" in k]
        colors = ["#1f77b4", "#2ca02c", "#d62728"]
        for k, c in zip(rep_models, colors):
            r = trajectory_runs[k]
            label_clean = k.replace("_seed42", "").upper()
            ax.plot(r.est_trajectory_east, r.est_trajectory_north, color=c, linewidth=2.0, label=f"{label_clean} (Drift: {r.final_drift_m:.1f}m / {r.drift_pct:.1f}%)")
            
        ax.scatter([gt_e[0]], [gt_n[0]], color="green", s=100, zorder=5, label="Outage Start (t0)")
        ax.scatter([gt_e[-1]], [gt_n[-1]], color="red", marker="x", s=100, zorder=5, label="True End")
        ax.set_xlabel("Local East (metres)")
        ax.set_ylabel("Local North (metres)")
        ax.set_title("60-Second GPS Outage Trajectory Dead Reckoning on Test Holdout (Drive S4)")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("figures/phase8_outage_trajectories_60s.png")
        plt.close()
        print("Saved figures/phase8_outage_trajectories_60s.png")
        
    print("\nPhase 8 Evaluation Completed Successfully!")


if __name__ == "__main__":
    run_full_evaluation()
