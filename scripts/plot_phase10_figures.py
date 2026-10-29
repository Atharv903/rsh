"""
Plots publication-quality figures for Phase 10 Independent-Drive Validation:
1. figures/phase10_drift_comparison_barplot.png
2. figures/phase10_sample_trajectories.png
"""

import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.modern_tcn import ModernTCNModel
from src.corrected_synchronization import extract_aligned_segments_for_drive
from src.outage_estimator import PreOutageState, estimate_initial_heading_from_pre_outage_gps

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# 1. Bar Plot of Final Drift across Methods
df = pd.read_csv("experiments/phase10_independent_validation/results/phase10_independent_outage_benchmarks.csv")
df["duration_nominal"] = df["duration_s"].apply(lambda d: 10 if d < 20 else (30 if d < 45 else 60))

def get_arch(m):
    if "cnngru" in m: return "CNN-GRU"
    if "mamba2" in m: return "Mamba-2"
    if "moderntcn" in m: return "ModernTCN"
    return m

df["arch"] = df["method"].apply(get_arch)

sub = df[df["heading_convention"] == "sign_corrected (-1)"]
arch_sum = sub.groupby(["duration_nominal", "arch"])[["final_drift_m", "ate_rmse_m", "speed_mae_mps"]].mean().unstack(level=1)

methods_order = ["Zero-Speed", "Constant-Velocity", "CNN-GRU", "Mamba-2", "ModernTCN"]
colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
x = np.arange(3)
width = 0.16

for i, (m, col) in enumerate(zip(methods_order, colors)):
    vals = [arch_sum["final_drift_m"][m].loc[10], arch_sum["final_drift_m"][m].loc[30], arch_sum["final_drift_m"][m].loc[60]]
    bars = ax.bar(x + (i - 2) * width, vals, width, label=m, color=col, edgecolor="black", alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}m", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5, rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(["10-Second Outage", "30-Second Outage", "60-Second Outage"], fontsize=12, fontweight="bold")
ax.set_ylabel("Final Position Drift (meters)", fontsize=12, fontweight="bold")
ax.set_title("Independent-Drive Validation: GPS-Denied Dead-Reckoning Drift (Driver B - M)", fontsize=14, fontweight="bold")
ax.legend(title="Navigation Method", fontsize=10, title_fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.set_ylim(0, 850)

plt.tight_layout()
barplot_path = figures_dir / "phase10_drift_comparison_barplot.png"
plt.savefig(barplot_path)
plt.close()
print(f"Saved barplot to {barplot_path}")

# 2. Sample Outage Trajectories Plot
raw_m_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)"
segs, _ = extract_aligned_segments_for_drive("M", raw_dir=raw_m_dir)
drive_df = segs[1]

scenarios_path = "experiments/phase10_independent_validation/scenarios/driver_b_outage_scenarios.json"
with open(scenarios_path) as f:
    scenarios = json.load(f)

# Load ModernTCN seed 42 checkpoint
from scripts.evaluate_phase10_independent import load_model
ckpt_path = "experiments/phase8_corrected_speed_models/checkpoints/moderntcnmodel_seed42/best_model.pt"
model, _ = load_model(ckpt_path)

norm_meta = json.load(open("data/processed/sensor_normalizer_grouped.json"))
norm_params = norm_meta["params"]
feat_channels = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
norm_mean = np.array([norm_params[ch]["mean"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
norm_std = np.array([norm_params[ch]["std"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)

# Select 3 sample scenarios: one 10s, one 30s, one 60s
sample_indices = [0, 5, 10]
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

for ax_idx, s_idx in enumerate(sample_indices):
    sc = scenarios[s_idx]
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
    
    # Neural model speed predictions
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
    
    # Dead reckoning integration (sign corrected)
    gyro_yaw_rate = (in_outage_df["gyro_pitch"].values.astype(np.float32) - gyro_bias)
    
    def integrate(speeds):
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
        return pe, pn

    nn_e, nn_n = integrate(nn_speed)
    cv_e, cv_n = integrate(const_speed)
    
    ax = axes[ax_idx]
    ax.plot(ref_east - pos_e0, ref_north - pos_n0, "k-", linewidth=2.5, label="Ground Truth (GNSS/INS)")
    ax.plot(cv_e - pos_e0, cv_n - pos_n0, "b--", linewidth=1.8, label="Constant-Velocity")
    ax.plot(nn_e - pos_e0, nn_n - pos_n0, "r-.", linewidth=2.0, label="ModernTCN (Seed 42)")
    ax.plot(0, 0, "go", markersize=8, label="Outage Start (GPS Lost)")
    ax.plot(ref_east[-1] - pos_e0, ref_north[-1] - pos_n0, "k*", markersize=10, label="GT End")
    
    drift_cv = np.hypot(cv_e[-1] - ref_east[-1], cv_n[-1] - ref_north[-1])
    drift_nn = np.hypot(nn_e[-1] - ref_east[-1], nn_n[-1] - ref_north[-1])
    
    ax.set_title(f"{dur}s Outage ({sid})\nCV Drift: {drift_cv:.1f}m | ModernTCN: {drift_nn:.1f}m", fontsize=11, fontweight="bold")
    ax.set_xlabel("Relative East (m)", fontsize=10)
    ax.set_ylabel("Relative North (m)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    if ax_idx == 0:
        ax.legend(fontsize=8.5, loc="best")

plt.suptitle("Independent-Drive (Driver B) GPS-Denied Outage Navigation Tracks", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
traj_path = figures_dir / "phase10_sample_trajectories.png"
plt.savefig(traj_path, bbox_inches="tight")
plt.close()
print(f"Saved trajectory plots to {traj_path}")
