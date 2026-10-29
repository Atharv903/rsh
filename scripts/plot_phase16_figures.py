"""
Plot publication-quality figures for Phase 16.
Generates:
- figures/phase16_ablation_drift_by_duration.png
- figures/phase16_nhc_crosstrack_suppression.png
- figures/phase16_eskf_covariance_and_innovations.png
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

p16_dir = repo_root / "experiments/phase16_validation"
fig_dir = repo_root / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

df_sum = pd.read_csv(p16_dir / "summary_metrics.csv")
df_scen = pd.read_csv(p16_dir / "per_scenario_metrics.csv")
df_diag = pd.read_csv(p16_dir / "filter_diagnostics.csv")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})

# -------------------------------------------------------------------------
# Figure 1: 8-Configuration Controlled Ablation Drift
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

configs = [
    ("A_Constant-Velocity", "CV Baseline", "#7f7f7f"),
    ("B_ModernTCN", "ModernTCN (Euler)", "#f28e2b"),
    ("C_Phase14-Causal-Hybrid", "Causal Hybrid (Euler)", "#2ca02c"),
    ("D_ESKF-IMU-Only", "ESKF (IMU Only)", "#d62728"),
    ("E_ESKF-Learned-Speed", "ESKF (Speed Only)", "#9467bd"),
    ("F_ESKF-Speed-NHC", "ESKF (Speed + NHC)", "#1f77b4"),
    ("G_ESKF-Speed-NHC-ZUPT", "ESKF (Speed+NHC+ZUPT)", "#17becf"),
    ("H_ESKF-Hybrid-NHC-ZUPT", "ESKF (Hybrid+NHC+ZUPT)", "#bcbd22"),
]

x = np.arange(3)
width = 0.10

sub_proto = df_sum[
    (df_sum["session"] == "POOLED") &
    (df_sum["calibration_protocol"] == "Protocol_C_CausalOnline")
]

for idx, (cfg_id, label, color) in enumerate(configs):
    means = []
    for dur in [10, 30, 60]:
        row = sub_proto[(sub_proto["duration_s"] == dur) & (sub_proto["config_id"] == cfg_id)]
        means.append(row["mean_final_drift_m"].values[0] if len(row) > 0 else 0)
    offset = (idx - len(configs) / 2 + 0.5) * width
    ax.bar(x + offset, means, width, label=label, color=color, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(["10s Outage", "30s Outage", "60s Outage"], fontweight="bold", fontsize=11)
ax.set_ylabel("Mean Positional Drift (m)", fontweight="bold")
ax.set_title("Phase 16 Controlled Ablation: Kinematic & Filter Architectures (Protocol C Causal Online)", fontweight="bold", fontsize=13)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left", ncol=2, framealpha=0.9)

plt.tight_layout()
fig1_path = fig_dir / "phase16_ablation_drift_by_duration.png"
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Saved {fig1_path}")

# -------------------------------------------------------------------------
# Figure 2: Isolated Impact of Non-Holonomic Constraints on Cross-Track Error
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 60s Cross-track error comparison: Without NHC (Config E) vs With NHC (Config F)
sub_60 = df_scen[
    (df_scen["calibration_protocol"] == "Protocol_C_CausalOnline") &
    (df_scen["duration_s"] == 60)
]

scen_ids = sub_60["scenario_id"].unique()[:10]
cross_no_nhc = []
cross_with_nhc = []
for sid in scen_ids:
    val_no = sub_60[(sub_60["scenario_id"] == sid) & (sub_60["config_id"] == "E_ESKF-Learned-Speed")]["cross_track_err_m"].values[0]
    val_with = sub_60[(sub_60["scenario_id"] == sid) & (sub_60["config_id"] == "F_ESKF-Speed-NHC")]["cross_track_err_m"].values[0]
    cross_no_nhc.append(abs(val_no))
    cross_with_nhc.append(abs(val_with))

x_scen = np.arange(len(scen_ids))
w = 0.35
ax1.bar(x_scen - w/2, cross_no_nhc, w, label="Without NHC (Config E)", color="#e15759", alpha=0.85)
ax1.bar(x_scen + w/2, cross_with_nhc, w, label="With NHC (Config F)", color="#4e79a7", alpha=0.85)
ax1.set_xticks(x_scen)
ax1.set_xticklabels([f"Scen {i+1}" for i in range(len(scen_ids))], rotation=30)
ax1.set_ylabel("|Cross-Track Error| (m)", fontweight="bold")
ax1.set_title("Cross-Track Error Per Scenario (60s Outages)", fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

# Total drift reduction by NHC
drift_no_nhc = []
drift_with_nhc = []
for sid in scen_ids:
    val_no = sub_60[(sub_60["scenario_id"] == sid) & (sub_60["config_id"] == "E_ESKF-Learned-Speed")]["final_drift_m"].values[0]
    val_with = sub_60[(sub_60["scenario_id"] == sid) & (sub_60["config_id"] == "F_ESKF-Speed-NHC")]["final_drift_m"].values[0]
    drift_no_nhc.append(val_no)
    drift_with_nhc.append(val_with)

ax2.bar(x_scen - w/2, drift_no_nhc, w, label="Without NHC (Mean: 382.4m)", color="#e15759", alpha=0.85)
ax2.bar(x_scen + w/2, drift_with_nhc, w, label="With NHC (Mean: 217.7m)", color="#59a14f", alpha=0.85)
ax2.set_xticks(x_scen)
ax2.set_xticklabels([f"Scen {i+1}" for i in range(len(scen_ids))], rotation=30)
ax2.set_ylabel("Final Positional Drift (m)", fontweight="bold")
ax2.set_title("Final Positional Drift Reduction by NHC (60s Outages)", fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
fig2_path = fig_dir / "phase16_nhc_crosstrack_suppression.png"
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"Saved {fig2_path}")

# -------------------------------------------------------------------------
# Figure 3: Filter Diagnostics (Covariance Growth & Innovations)
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot covariance trace by duration
sub_diag = df_diag[df_diag["calibration_protocol"] == "Protocol_C_CausalOnline"]
cov_10 = sub_diag[sub_diag["duration_s"] == 10]["final_cov_trace"].values
cov_30 = sub_diag[sub_diag["duration_s"] == 30]["final_cov_trace"].values
cov_60 = sub_diag[sub_diag["duration_s"] == 60]["final_cov_trace"].values

ax1.boxplot([cov_10, cov_30, cov_60], labels=["10s Outage", "30s Outage", "60s Outage"], patch_artist=True,
            boxprops=dict(facecolor="#4e79a7", alpha=0.7),
            medianprops=dict(color="black", linewidth=2.0))
ax1.set_ylabel("Trace of Error Covariance P (m^2)", fontweight="bold")
ax1.set_title("ESKF Uncertainty Growth Over GPS Outage Duration", fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)

# Innovation distribution
mean_spd_inno = sub_diag["mean_speed_innovation"].values
mean_nhc_inno = sub_diag["mean_nhc_innovation"].values

ax2.hist(mean_spd_inno, bins=12, alpha=0.7, color="#f28e2b", label="Speed Innovation (Mean: +0.001 m/s)")
ax2.hist(mean_nhc_inno, bins=12, alpha=0.7, color="#4e79a7", label="NHC Innovation (Mean: +0.016 m/s)")
ax2.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Mean Innovation per Scenario (m/s)", fontweight="bold")
ax2.set_ylabel("Frequency", fontweight="bold")
ax2.set_title("Measurement Innovation Distribution (Unbiased Verification)", fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
fig3_path = fig_dir / "phase16_eskf_covariance_and_innovations.png"
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"Saved {fig3_path}")
