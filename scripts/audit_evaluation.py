"""
Comprehensive Audit Script for GPS-Denied Navigation Benchmark.
Audits code, recomputes metrics, evaluates naive baselines, analyzes error by speed bin,
audits trajectory dead reckoning, checks leakage, and profiles hardware latency.
"""

import os
import sys
import json
import time
import platform
from pathlib import Path

# Add project root
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import builtins
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.dataset import IMUNavigationDataset
from src.evaluate import load_model, evaluate_test_set, evaluate_outage_trajectories


def run_comprehensive_audit():
    print("=" * 70)
    print("STARTING INDEPENDENT RIGOROUS AUDIT")
    print("=" * 70)
    
    device = torch.device("cpu")
    
    # -------------------------------------------------------------------------
    # AUDIT ITEM 1 & 2: Recompute Test Metrics from Raw Arrays & Verify Units
    # -------------------------------------------------------------------------
    print("\n--- ITEM 1 & 2: RECOMPUTING TEST METRICS FROM CHECKPOINTS ---")
    test_npz_path = "data/processed/test_windows.npz"
    test_ds = IMUNavigationDataset(test_npz_path, filter_valid_only=True)
    
    # Raw target inspection
    data = np.load(test_npz_path, allow_pickle=True)
    raw_Y = data["Y"]
    # Verify dataset conversion
    target_vel = test_ds.y  # Should be raw_Y / 10.0
    diff = np.abs(target_vel - (raw_Y / 10.0))
    print(f"Target verification: Max diff between dataset y and raw_Y/10.0 = {np.max(diff):.6e}")
    assert np.max(diff) < 1e-5, "Target conversion discrepancy detected!"
    
    models = {
        "CNN-GRU (Seed 42)": "checkpoints/cnn_gru_seed42/best_model.pt",
        "CNN-GRU (Seed 123)": "checkpoints/cnn_gru_seed123/best_model.pt",
        "ModernTCN (Seed 42)": "checkpoints/moderntcn_seed42/best_model.pt",
        "ModernTCN (Seed 123)": "checkpoints/moderntcn_seed123/best_model.pt",
        "Mamba-2 SSD (Seed 42)": "checkpoints/mamba2_seed42/best_model.pt",
        "Mamba-2 SSD (Seed 123)": "checkpoints/mamba2_seed123/best_model.pt",
    }
    
    predictions = {}
    test_metrics = {}
    
    for name, ckpt_path in models.items():
        m, cfg = load_model(Path(ckpt_path), device)
        res = evaluate_test_set(m, test_npz_path, device)
        preds = res["preds"]
        predictions[name] = preds
        
        # Raw recomputation of MAE, RMSE, East MAE, North MAE
        vel_err = np.linalg.norm(preds - target_vel, axis=1)
        mae = float(np.mean(vel_err))
        rmse = float(np.sqrt(np.mean(vel_err ** 2)))
        e_mae = float(np.mean(np.abs(preds[:, 0] - target_vel[:, 0])))
        n_mae = float(np.mean(np.abs(preds[:, 1] - target_vel[:, 1])))
        
        test_metrics[name] = {
            "mae": mae,
            "rmse": rmse,
            "e_mae": e_mae,
            "n_mae": n_mae,
        }
        print(f"  {name:25s} | MAE: {mae:.3f} m/s | RMSE: {rmse:.3f} m/s | E-MAE: {e_mae:.3f} | N-MAE: {n_mae:.3f}")

    # -------------------------------------------------------------------------
    # AUDIT ITEM 3: Naive Baselines
    # -------------------------------------------------------------------------
    print("\n--- ITEM 3: NAIVE BASELINE COMPARISONS ---")
    # Baseline A: Zero Velocity Predictor [0, 0]
    zero_preds = np.zeros_like(target_vel)
    zero_err = np.linalg.norm(zero_preds - target_vel, axis=1)
    zero_mae = float(np.mean(zero_err))
    zero_rmse = float(np.sqrt(np.mean(zero_err ** 2)))
    
    # Baseline B: Train Mean Velocity Predictor
    train_data = np.load("data/processed/train_windows.npz", allow_pickle=True)
    train_mean = (train_data["Y"] / 10.0).mean(axis=0)
    train_mean_preds = np.tile(train_mean, (len(target_vel), 1))
    tmean_err = np.linalg.norm(train_mean_preds - target_vel, axis=1)
    tmean_mae = float(np.mean(tmean_err))
    tmean_rmse = float(np.sqrt(np.mean(tmean_err ** 2)))
    
    # Baseline C: Previous-Window Persistence Baseline (v_t = v_{t-1})
    # For sequential windows from the same drive, use target_{k-1}
    drive_ids = test_ds.drive_id
    persist_preds = np.zeros_like(target_vel)
    persist_preds[0] = target_vel[0]
    for k in range(1, len(target_vel)):
        if drive_ids[k] == drive_ids[k-1]:
            persist_preds[k] = target_vel[k-1]
        else:
            persist_preds[k] = target_vel[k]  # First window of new drive
            
    persist_err = np.linalg.norm(persist_preds - target_vel, axis=1)
    persist_mae = float(np.mean(persist_err))
    persist_rmse = float(np.sqrt(np.mean(persist_err ** 2)))
    
    print(f"  {'Zero-Velocity [0, 0]':25s} | MAE: {zero_mae:.3f} m/s | RMSE: {zero_rmse:.3f} m/s")
    print(f"  {'Train Mean Vector':25s} | MAE: {tmean_mae:.3f} m/s | RMSE: {tmean_rmse:.3f} m/s (Vector: {train_mean})")
    print(f"  {'Previous-Window (1s Stride)':25s} | MAE: {persist_mae:.3f} m/s | RMSE: {persist_rmse:.3f} m/s")

    # -------------------------------------------------------------------------
    # AUDIT ITEM 4: Analysis of Test MAE (~9.8 m/s) and Speed Range Breakdown
    # -------------------------------------------------------------------------
    print("\n--- ITEM 4: SPEED RANGE BREAKDOWN & ERROR ANALYSIS ---")
    true_speeds = np.linalg.norm(target_vel, axis=1)
    print(f"Test Target Speed Distribution: Mean={true_speeds.mean():.2f} m/s, Median={np.median(true_speeds):.2f} m/s, Std={true_speeds.std():.2f} m/s, Max={true_speeds.max():.2f} m/s")
    
    bins = [
        ("Stationary (speed < 1 m/s)", true_speeds < 1.0),
        ("Low Speed (1 <= speed < 5 m/s)", (true_speeds >= 1.0) & (true_speeds < 5.0)),
        ("Medium Speed (5 <= speed < 15 m/s)", (true_speeds >= 5.0) & (true_speeds < 15.0)),
        ("High Speed (speed >= 15 m/s)", true_speeds >= 15.0),
    ]
    
    print(f"\n{'Speed Bin':35s} | {'Count':7s} | {'% Total':7s} | {'Mean Speed':10s} | {'CNN-GRU':10s} | {'ModernTCN':10s} | {'Mamba-2':10s} | {'Zero Pred':10s}")
    for bin_name, mask in bins:
        count = int(np.sum(mask))
        pct = (count / len(true_speeds)) * 100.0
        m_speed = float(np.mean(true_speeds[mask])) if count > 0 else 0.0
        
        # Compute MAE for each model
        cnn_b = float(np.mean(np.linalg.norm(predictions["CNN-GRU (Seed 42)"][mask] - target_vel[mask], axis=1))) if count > 0 else 0.0
        tcn_b = float(np.mean(np.linalg.norm(predictions["ModernTCN (Seed 42)"][mask] - target_vel[mask], axis=1))) if count > 0 else 0.0
        mam_b = float(np.mean(np.linalg.norm(predictions["Mamba-2 SSD (Seed 42)"][mask] - target_vel[mask], axis=1))) if count > 0 else 0.0
        zer_b = float(np.mean(np.linalg.norm(zero_preds[mask] - target_vel[mask], axis=1))) if count > 0 else 0.0
        
        print(f"{bin_name:35s} | {count:7d} | {pct:6.1f}% | {m_speed:8.2f} m/s | {cnn_b:8.3f} m/s | {tcn_b:8.3f} m/s | {mam_b:8.3f} m/s | {zer_b:8.3f} m/s")
        
    # Analyze Prediction Norms vs Target Norms
    for name, p in predictions.items():
        pred_speeds = np.linalg.norm(p, axis=1)
        print(f"  {name:25s} Prediction Speeds: Mean={pred_speeds.mean():.2f} m/s, Std={pred_speeds.std():.2f} m/s, Max={pred_speeds.max():.2f} m/s")

    # -------------------------------------------------------------------------
    # AUDIT ITEM 5: Trajectory Integration & Initial Velocity/Heading Verification
    # -------------------------------------------------------------------------
    print("\n--- ITEM 5: TRAJECTORY INTEGRATION & OUTAGE AUDIT ---")
    with open("reports/outage_evaluation_results.json", "r") as f:
        outage_logs = json.load(f)
        
    print(f"Loaded outage evaluation logs for models: {list(outage_logs.keys())}")
    for m_key in outage_logs.keys():
        for s_key in outage_logs[m_key].keys():
            scs = outage_logs[m_key][s_key]["scenarios"]
            print(f"  {m_key} / {s_key}: {len(scs)} scenarios evaluated")
            for sc in scs[:3]:
                print(f"    Scenario {sc['scenario_id']} ({sc['duration_s']}s, Drive {sc['drive']}): Path Length = {sc['total_length_m']:.1f}m | ATE = {sc['ate_rmse_m']:.2f}m | Final Drift = {sc['final_drift_m']:.2f}m ({sc['drift_pct']:.1f}%)")
                # Verify formula: drift_pct = (final_drift / path_length) * 100
                expected_pct = (sc['final_drift_m'] / max(sc['total_length_m'], 1.0)) * 100.0
                assert abs(sc['drift_pct'] - expected_pct) < 1e-3, "Drift percentage formula discrepancy!"

    # -------------------------------------------------------------------------
    # AUDIT ITEM 6: Leakage Audit (Drives, Normalizer, Early Stopping)
    # -------------------------------------------------------------------------
    print("\n--- ITEM 6: DATA LEAKAGE AUDIT ---")
    with open("data/splits/drive_split_manifest.json", "r") as f:
        split_manifest = json.load(f)
    train_drives = set(split_manifest["splits"]["train"]["drives"])
    val_drives = set(split_manifest["splits"]["val"]["drives"])
    test_drives = set(split_manifest["splits"]["test"]["drives"])
    print(f"Train drives: {train_drives}")
    print(f"Val drives:   {val_drives}")
    print(f"Test drives:  {test_drives}")
    assert len(train_drives.intersection(test_drives)) == 0, "CRITICAL: Train and Test share drives!"
    assert len(val_drives.intersection(test_drives)) == 0, "CRITICAL: Val and Test share drives!"
    print("[PASS] Drive splits are completely disjoint.")
    
    with open("data/processed/sensor_normalizer.json", "r") as f:
        norm_info = json.load(f)
    print(f"Normalizer fitted on: {norm_info.get('fit_split', 'train')} (Fitted flag: {norm_info['is_fitted']})")

    # -------------------------------------------------------------------------
    # AUDIT ITEM 7: Random Seeds & Weight Independence
    # -------------------------------------------------------------------------
    print("\n--- ITEM 7: RANDOM SEED & WEIGHT INDEPENDENCE AUDIT ---")
    for m_type in ["cnn_gru", "moderntcn", "mamba2"]:
        ckpt42 = torch.load(f"checkpoints/{m_type}_seed42/best_model.pt", map_location="cpu")
        ckpt123 = torch.load(f"checkpoints/{m_type}_seed123/best_model.pt", map_location="cpu")
        
        # Check first layer weights
        key = list(ckpt42["model_state_dict"].keys())[0]
        w42 = ckpt42["model_state_dict"][key].numpy()
        w123 = ckpt123["model_state_dict"][key].numpy()
        weight_diff = np.max(np.abs(w42 - w123))
        print(f"  {m_type:10s} | Seed 42 vs Seed 123 weight diff ({key}): {weight_diff:.6f}")
        assert weight_diff > 1e-4, f"CRITICAL: {m_type} weights are identical between seeds!"
    print("[PASS] Seeds 42 and 123 produced verifiably independent model parameters.")

    # -------------------------------------------------------------------------
    # AUDIT ITEM 8: Recalculate Claimed Percentage Improvements
    # -------------------------------------------------------------------------
    print("\n--- ITEM 8: RECALCULATING ALL CLAIMED PERCENTAGES FROM RAW DATA ---")
    df_metrics = pd.read_csv("reports/metrics_table.csv")
    print(df_metrics.to_string(index=False))
    
    # Extract numerical values
    def parse_mean(val_str):
        if pd.isna(val_str) or "N/A" in str(val_str):
            return 0.0
        clean = str(val_str).replace("%", "").split("±")[0].strip()
        return float(clean)
        
    models_df = df_metrics.set_index("Model")
    cnn_10_ate = parse_mean(models_df.loc["CNN_GRU", "10s Outage ATE (m)"])
    tcn_10_ate = parse_mean(models_df.loc["MODERNTCN", "10s Outage ATE (m)"])
    mam_10_ate = parse_mean(models_df.loc["MAMBA2", "10s Outage ATE (m)"])
    
    cnn_30_ate = parse_mean(models_df.loc["CNN_GRU", "30s Outage ATE (m)"])
    tcn_30_ate = parse_mean(models_df.loc["MODERNTCN", "30s Outage ATE (m)"])
    mam_30_ate = parse_mean(models_df.loc["MAMBA2", "30s Outage ATE (m)"])
    
    cnn_60_ate = parse_mean(models_df.loc["CNN_GRU", "60s Outage ATE (m)"])
    tcn_60_ate = parse_mean(models_df.loc["MODERNTCN", "60s Outage ATE (m)"])
    mam_60_ate = parse_mean(models_df.loc["MAMBA2", "60s Outage ATE (m)"])
    
    tcn_10_imp = ((cnn_10_ate - tcn_10_ate) / cnn_10_ate) * 100.0
    mam_10_imp = ((cnn_10_ate - mam_10_ate) / cnn_10_ate) * 100.0
    tcn_30_imp = ((cnn_30_ate - tcn_30_ate) / cnn_30_ate) * 100.0
    mam_30_imp = ((cnn_30_ate - mam_30_ate) / cnn_30_ate) * 100.0
    tcn_60_imp = ((cnn_60_ate - tcn_60_ate) / cnn_60_ate) * 100.0
    mam_60_imp = ((cnn_60_ate - mam_60_ate) / cnn_60_ate) * 100.0
    
    print(f"Recalculated Improvements over CNN-GRU:")
    print(f"  10s Outage ATE: ModernTCN = {tcn_10_imp:.2f}% | Mamba-2 = {mam_10_imp:.2f}%")
    print(f"  30s Outage ATE: ModernTCN = {tcn_30_imp:.2f}% | Mamba-2 = {mam_30_imp:.2f}%")
    print(f"  60s Outage ATE: ModernTCN = {tcn_60_imp:.2f}% | Mamba-2 = {mam_60_imp:.2f}%")

    # -------------------------------------------------------------------------
    # AUDIT ITEM 9: Latency Methodology Verification
    # -------------------------------------------------------------------------
    print("\n--- ITEM 9: LATENCY METHODOLOGY & HARDWARE AUDIT ---")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
    print(f"CPU Thread Count: {torch.get_num_threads()}")
    
    # Rigorous benchmark for batch_size = 1 and batch_size = 64
    for m_type, ckpt_name in [("CNN-GRU", "cnn_gru_seed42"), ("ModernTCN", "moderntcn_seed42"), ("Mamba-2 SSD", "mamba2_seed42")]:
        m, _ = load_model(Path(f"checkpoints/{ckpt_name}/best_model.pt"), device)
        
        for bs in [1, 64]:
            x_sample = torch.randn(bs, 100, 6, device=device)
            # Warmup: 20 passes
            for _ in range(20):
                _ = m(x_sample)
            # Timed: 50 passes
            times = []
            for _ in range(50):
                t0 = time.perf_counter()
                _ = m(x_sample)
                times.append((time.perf_counter() - t0) * 1000.0)
            avg_t = np.mean(times)
            per_sample_t = avg_t / bs
            std_t = np.std(times)
            print(f"  {m_type:12s} (Batch {bs:2d}) | Total: {avg_t:.2f} ± {std_t:.2f} ms | Per Window: {per_sample_t:.2f} ms")


if __name__ == "__main__":
    run_comprehensive_audit()
