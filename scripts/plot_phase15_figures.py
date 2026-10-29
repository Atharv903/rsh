"""
Plot publication-quality figures for Phase 15.
Generates:
- figures/phase15_drift_by_duration_and_protocol.png
- figures/phase15_speed_mae_comparison.png
- figures/phase15_trajectory_samples.png
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
p15_dir = repo_root / "experiments/phase15_validation"
fig_dir = repo_root / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

df_sum = pd.read_csv(p15_dir / "summary_metrics.csv")
df_scen = pd.read_csv(p15_dir / "per_scenario_metrics.csv")

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
})

# -------------------------------------------------------------------------
# Figure 1: Positional Drift by Duration & Protocol
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
protocols = [
    ("Protocol_A_ZeroBias", "Protocol A: Zero Bias (b=0)"),
    ("Protocol_B_PriorCalib", "Protocol B: Prior Calib (Trip Start)"),
    ("Protocol_C_CausalOnline", "Protocol C: Causal Online Calib"),
]

estimators = [
    ("Constant-Velocity", "CV Baseline", "#4e79a7"),
    ("ModernTCN", "ModernTCN", "#f28e2b"),
    ("Phase14-Causal-Hybrid", "Causal Hybrid", "#59a14f"),
    ("Constrained-Causal-Hybrid", "Constrained Hybrid", "#b07aa1"),
    ("Complementary-Filter", "Complementary", "#edc949"),
    ("Acceleration-Integration", "Pure Accel Int", "#e15759"),
]

for ax_idx, (proto_key, proto_title) in enumerate(protocols):
    ax = axes[ax_idx]
    x = np.arange(3)  # 10s, 30s, 60s
    width = 0.13
    
    for est_idx, (est_key, est_label, color) in enumerate(estimators):
        means = []
        for dur in [10, 30, 60]:
            sub = df_sum[
                (df_sum["session"] == "POOLED") &
                (df_sum["calibration_protocol"] == proto_key) &
                (df_sum["duration_s"] == dur) &
                (df_sum["estimator"] == est_key)
            ]
            val = sub["mean_final_drift_m"].values[0] if len(sub) > 0 else 0
            means.append(val)
            
        offset = (est_idx - len(estimators) / 2 + 0.5) * width
        rects = ax.bar(x + offset, means, width, label=est_label if ax_idx == 0 else "", color=color, alpha=0.9)
        
    ax.set_xticks(x)
    ax.set_xticklabels(["10s Outage", "30s Outage", "60s Outage"], fontweight="bold")
    ax.set_title(proto_title, fontweight="bold")
    ax.set_ylabel("Mean Positional Drift (m)" if ax_idx == 0 else "")
    ax.grid(True, linestyle="--", alpha=0.5)
    if ax_idx == 0:
        ax.legend(loc="upper left", framealpha=0.9)

plt.tight_layout()
fig1_path = fig_dir / "phase15_drift_by_duration_and_protocol.png"
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Saved {fig1_path}")

# -------------------------------------------------------------------------
# Figure 2: Speed MAE Comparison across Estimators and Regimes
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
sub_spd = df_sum[
    (df_sum["session"] == "POOLED") &
    (df_sum["calibration_protocol"] == "Protocol_C_CausalOnline")
]

x = np.arange(3)
width = 0.14
for est_idx, (est_key, est_label, color) in enumerate(estimators):
    vals = []
    for dur in [10, 30, 60]:
        row = sub_spd[(sub_spd["duration_s"] == dur) & (sub_spd["estimator"] == est_key)]
        vals.append(row["mean_speed_mae_mps"].values[0] if len(row) > 0 else 0)
    offset = (est_idx - len(estimators) / 2 + 0.5) * width
    ax.bar(x + offset, vals, width, label=est_label, color=color, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(["10s Outage", "30s Outage", "60s Outage"], fontweight="bold")
ax.set_ylabel("Speed Mean Absolute Error (m/s)", fontweight="bold")
ax.set_title("Forward Speed Estimation Accuracy (Protocol C Causal Online)", fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left")

plt.tight_layout()
fig2_path = fig_dir / "phase15_speed_mae_comparison.png"
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"Saved {fig2_path}")

# -------------------------------------------------------------------------
# Figure 3: Sample 2D Dead Reckoning Trajectories (60s Outages)
# -------------------------------------------------------------------------
# Pick two representative 60s scenarios: one from Driver B, one from Driver D
manifest_path = repo_root / "experiments/phase12_validation/phase12_scenarios.json"
with open(manifest_path) as f:
    scenarios = json.load(f)

from src.corrected_synchronization import extract_aligned_segments_for_drive
from src.outage_estimator import PreOutageState, estimate_initial_heading_from_pre_outage_gps
from experiments.phase14_validation.evaluate_phase14_causal import get_causal_online_bias, compute_causal_hybrid_speed

raw_b_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)"
segs_b, _ = extract_aligned_segments_for_drive("M", raw_dir=raw_b_dir)
df_b = segs_b[1]

raw_d_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/Y (Driver D)/Y1"
segs_d, _ = extract_aligned_segments_for_drive("Y1", raw_dir=raw_d_dir)
df_d = segs_d[2]
drive_dfs = {"M": df_b, "Y1": df_d}

# Pick scenario 14 (Driver B 60s) and scenario 29 (Driver D 60s)
sc_b = [s for s in scenarios if s["driver_id"] == "Driver B" and s["duration_s"] == 60][2]
sc_d = [s for s in scenarios if s["driver_id"] == "Driver D" and s["duration_s"] == 60][2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for ax, sc, title in [(ax1, sc_b, "Driver B (60s Outage)"), (ax2, sc_d, "Driver D (60s Outage)")]:
    d_df = drive_dfs[sc["drive_id"]]
    st, end = sc["start_idx"], sc["end_idx"]
    pre_st = max(0, st - sc["pre_window_samples"])
    pre_df = d_df.iloc[pre_st:st]
    in_df = d_df.iloc[st:end]
    n = len(in_df)
    
    t_arr = in_df["time_s"].values
    dt = np.diff(t_arr, prepend=t_arr[0])
    dt[0] = 0.0
    
    head_deg, last_spd = estimate_initial_heading_from_pre_outage_gps(pre_df)
    e0, n0 = float(pre_df["ref_east_m"].iloc[-1]), float(pre_df["ref_north_m"].iloc[-1])
    
    ref_e = in_df["ref_east_m"].values - e0
    ref_n = in_df["ref_north_m"].values - n0
    
    b_c, _, _ = get_causal_online_bias(d_df, st)
    gyro_yaw = in_df["gyro_pitch"].values.astype(np.float32) - b_c
    
    # Run CV, Hybrid, Pure Accel
    def integrate_traj(spd_vec):
        pe = np.zeros(n)
        pn = np.zeros(n)
        h = np.deg2rad(head_deg)
        for k in range(1, n):
            h = (h - float(gyro_yaw[k] * dt[k])) % (2.0 * np.pi)
            pe[k] = pe[k-1] + spd_vec[k] * np.sin(h) * dt[k]
            pn[k] = pn[k-1] + spd_vec[k] * np.cos(h) * dt[k]
        return pe, pn
        
    spd_cv = np.full(n, last_spd)
    pe_cv, pn_cv = integrate_traj(spd_cv)
    
    # Load MTCN
    norm_meta = json.load(open("data/processed/sensor_normalizer_grouped.json"))
    norm_params = norm_meta["params"]
    feat_channels = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
    norm_mean = np.array([norm_params[ch]["mean"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
    norm_std = np.array([norm_params[ch]["std"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
    
    import torch
    from scripts.evaluate_phase10_independent import load_model
    ckpt_dir = repo_root / "experiments/phase8_corrected_speed_models/checkpoints"
    mtcn_models = [load_model(ckpt_dir / f"moderntcnmodel_seed{s}/best_model.pt")[0] for s in [42, 123, 456]]
    
    spd_mtcn = np.zeros(n)
    spd_mtcn[0] = last_spd
    for k in range(1, n):
        w_raw = d_df.iloc[(st+k)-100 : st+k][feat_channels].values.astype(np.float32)
        w_norm = (w_raw - norm_mean) / norm_std
        inp_t = torch.from_numpy(w_norm).unsqueeze(0)
        with torch.no_grad():
            preds = [m(inp_t).item() for m in mtcn_models]
        spd_mtcn[k] = max(0.0, float(np.mean(preds)))
        
    spd_hyb = compute_causal_hybrid_speed(t_arr, last_spd, spd_mtcn, in_df["accel_x"].values, 10.0, 2.0, 5, -1.5)
    pe_hyb, pn_hyb = integrate_traj(spd_hyb)
    pe_mtcn, pn_mtcn = integrate_traj(spd_mtcn)
    
    ax.plot(ref_e, ref_n, "k-", linewidth=2.5, label="Ground Truth (GNSS)")
    ax.plot(pe_cv, pn_cv, "--", color="#4e79a7", linewidth=2.0, label=f"Constant Velocity (Drift {np.sqrt((pe_cv[-1]-ref_e[-1])**2 + (pn_cv[-1]-ref_n[-1])**2):.1f}m)")
    ax.plot(pe_mtcn, pn_mtcn, ":", color="#f28e2b", linewidth=2.0, label=f"ModernTCN (Drift {np.sqrt((pe_mtcn[-1]-ref_e[-1])**2 + (pn_mtcn[-1]-ref_n[-1])**2):.1f}m)")
    ax.plot(pe_hyb, pn_hyb, "-", color="#59a14f", linewidth=2.2, label=f"Causal Hybrid (Drift {np.sqrt((pe_hyb[-1]-ref_e[-1])**2 + (pn_hyb[-1]-ref_n[-1])**2):.1f}m)")
    
    ax.scatter([0], [0], color="green", s=80, marker="o", label="Outage Start", zorder=5)
    ax.scatter([ref_e[-1]], [ref_n[-1]], color="red", s=80, marker="X", label="Ground Truth End", zorder=5)
    
    ax.set_xlabel("East Displacement (m)", fontweight="bold")
    ax.set_ylabel("North Displacement (m)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", "datalim")

plt.tight_layout()
fig3_path = fig_dir / "phase15_trajectory_samples.png"
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"Saved {fig3_path}")
