"""
Evaluation script for GPS-denied navigation.
Evaluates velocity estimation and dead-reckoning trajectory integration during simulated outages.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add repository root to python search path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import torch
import torch.nn as nn

from src.dataset import IMUNavigationDataset
from src.models.cnn_gru import CNNGRUModel
from src.models.modern_tcn import ModernTCNModel
from src.models.mamba2 import Mamba2Model


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
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


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_npz: str,
    device: torch.device,
) -> Dict[str, float]:
    ds = IMUNavigationDataset(test_npz, filter_valid_only=True)
    batch_x = torch.from_numpy(ds.X).to(device)
    targets = ds.y
    
    # Predict in chunks to avoid memory spikes
    chunk_size = 512
    preds = []
    for i in range(0, len(batch_x), chunk_size):
        chunk = batch_x[i:i + chunk_size]
        preds.append(model(chunk).cpu().numpy())
    preds_arr = np.concatenate(preds, axis=0)
    
    vel_err = np.linalg.norm(preds_arr - targets, axis=1)
    mae = float(np.mean(vel_err))
    rmse = float(np.sqrt(np.mean(vel_err ** 2)))
    e_mae = float(np.mean(np.abs(preds_arr[:, 0] - targets[:, 0])))
    n_mae = float(np.mean(np.abs(preds_arr[:, 1] - targets[:, 1])))
    
    return {
        "vel_mae": mae,
        "vel_rmse": rmse,
        "v_east_mae": e_mae,
        "v_north_mae": n_mae,
        "preds": preds_arr,
        "targets": targets,
    }


def evaluate_outage_trajectories(
    model: nn.Module,
    outage_scenarios_path: str,
    processed_dir: str,
    normalizer_path: str,
    device: torch.device,
    window_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Performs dead-reckoning trajectory integration for each outage scenario.
    """
    with open(outage_scenarios_path, "r") as f:
        scenarios = json.load(f)
        
    with open(normalizer_path, "r") as f:
        norm_params = json.load(f)
        
    channels = [
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_yaw",
        "gyro_pitch",
        "gyro_roll",
    ]
    
    # Parse normalizer params
    if "params" in norm_params:
        norm_mean = np.array([norm_params["params"][ch]["mean"] for ch in channels], dtype=np.float32)
        norm_std = np.array([norm_params["params"][ch]["std"] for ch in channels], dtype=np.float32)
    else:
        norm_mean = np.array(norm_params["mean"], dtype=np.float32)
        norm_std = np.array(norm_params["std"], dtype=np.float32)
    
    results = []
    loaded_dfs = {}
    
    for sc in scenarios:
        drive_csv = sc["drive_csv"]
        if drive_csv not in loaded_dfs:
            proc_csv_path = Path(processed_dir) / drive_csv.replace(".csv", "_processed_aligned.csv")
            if not proc_csv_path.exists():
                continue
            loaded_dfs[drive_csv] = pd.read_csv(proc_csv_path)
            
        df = loaded_dfs[drive_csv]
        start_idx = sc.get("start_idx", sc.get("start_row"))
        end_idx = sc.get("end_idx", sc.get("end_row"))
        
        # We need a 100-sample window ending at each step t in [start_idx, end_idx]
        if start_idx < window_size:
            continue
            
        # Extract sensor data for the full interval plus preceding window
        sub_df = df.iloc[start_idx - window_size : end_idx].copy().reset_index(drop=True)
        
        # Ground truth trajectory in Local ENU metres
        east_col = "ref_east_m" if "ref_east_m" in df.columns else "local_east_m"
        north_col = "ref_north_m" if "ref_north_m" in df.columns else "local_north_m"
        time_col = "time_s" if "time_s" in df.columns else "time_seconds"
        
        gt_east = df.loc[start_idx:end_idx, east_col].to_numpy()
        gt_north = df.loc[start_idx:end_idx, north_col].to_numpy()
        time_arr = df.loc[start_idx:end_idx, time_col].to_numpy()
        
        if len(gt_east) < 2:
            continue
            
        # Ground truth path length
        seg_dists = np.sqrt(np.diff(gt_east)**2 + np.diff(gt_north)**2)
        total_length_m = float(np.sum(seg_dists))
        
        # Build sliding windows for the outage steps
        # Window ending at step k corresponds to sub_df[k : k + window_size]
        outage_steps = len(gt_east)
        windows = []
        raw_sensor_vals = sub_df[channels].to_numpy(dtype=np.float32)
        # Normalize with fitted train normalizer
        norm_sensor_vals = (raw_sensor_vals - norm_mean) / norm_std
        
        for k in range(outage_steps):
            w = norm_sensor_vals[k : k + window_size]
            windows.append(w)
            
        windows_arr = np.array(windows, dtype=np.float32)
        batch_tensor = torch.from_numpy(windows_arr).to(device)
        
        with torch.no_grad():
            vel_preds = model(batch_tensor).cpu().numpy()  # (outage_steps, 2)
            
        # Dead-reckoning integration: p_k = p_0 + sum_{i=1}^k v_i * dt
        dt = np.diff(time_arr, prepend=time_arr[0])
        dt[0] = 0.0  # At step 0, displacement is 0
        
        pred_east = np.zeros(outage_steps)
        pred_north = np.zeros(outage_steps)
        pred_east[0] = gt_east[0]
        pred_north[0] = gt_north[0]
        
        for k in range(1, outage_steps):
            pred_east[k] = pred_east[k-1] + vel_preds[k, 0] * dt[k]
            pred_north[k] = pred_north[k-1] + vel_preds[k, 1] * dt[k]
            
        # Compute errors
        pos_err = np.sqrt((pred_east - gt_east)**2 + (pred_north - gt_north)**2)
        ate_rmse = float(np.sqrt(np.mean(pos_err**2)))
        final_drift_m = float(pos_err[-1])
        drift_pct = float((final_drift_m / max(total_length_m, 1.0)) * 100.0)
        
        results.append({
            "scenario_id": sc["scenario_id"],
            "duration_s": sc["duration_s"],
            "drive": sc["drive_csv"],
            "total_length_m": total_length_m,
            "ate_rmse_m": ate_rmse,
            "final_drift_m": final_drift_m,
            "drift_pct": drift_pct,
            "gt_east": gt_east.tolist(),
            "gt_north": gt_north.tolist(),
            "pred_east": pred_east.tolist(),
            "pred_north": pred_north.tolist(),
        })
        
    return results
