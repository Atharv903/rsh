"""
Plots publication-quality figures for Phase 12 Expanded Independent Validation:
1. figures/phase12_expanded_validation_drift.png (Pooled & Per-Driver Drift Barplot)
2. figures/phase12_per_driver_comparison.png (Driver B vs Driver D Win Rates & Advantages)
3. figures/phase12_trajectory_driver_d.png (Sample Outage Trajectories for Driver D)
"""

import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# 1. Barplot of Drift across Durations & Drivers
sum_df = pd.read_csv("experiments/phase12_validation/per_session_summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

for ax_idx, (driver, title) in enumerate([("Driver B", "Driver B (M) — 15 Scenarios"), ("Driver D", "Driver D (Y1) — 15 Scenarios")]):
    ax = axes[ax_idx]
    d_df = sum_df[sum_df["driver_id"] == driver]
    
    x = np.arange(3)
    width = 0.18
    methods = ["Zero-Speed", "Constant-Velocity", "CNN-GRU", "Mamba-2", "ModernTCN"]
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for i, (m, col) in enumerate(zip(methods, colors)):
        sub = d_df[d_df["arch"] == m].sort_values("duration_s")
        vals = sub["final_drift_m"].values
        if len(vals) == 3:
            bars = ax.bar(x + (i - 2) * width, vals, width, label=m, color=col, edgecolor="black", alpha=0.9)
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.0f}m", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=7.5, rotation=45)
                
    ax.set_xticks(x)
    ax.set_xticklabels(["10s Outage", "30s Outage", "60s Outage"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Final Positional Drift (m)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 800)
    if ax_idx == 0:
        ax.legend(fontsize=9, loc="upper left")

plt.suptitle("Phase 12: Expanded Independent-Drive Validation (Driver B vs Driver D)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
out_drift_png = figures_dir / "phase12_expanded_validation_drift.png"
plt.savefig(out_drift_png, bbox_inches="tight")
plt.close()
print(f"Saved {out_drift_png}")

# 2. ModernTCN Drift Advantage Plot across 10s, 30s, 60s
stat_df = pd.read_csv("experiments/phase12_validation/statistical_comparison.csv")
mtcn_df = stat_df[stat_df["model_architecture"] == "ModernTCN"].copy()

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
x = np.arange(3)
width = 0.25

b_adv = mtcn_df[mtcn_df["session"] == "Driver B"].sort_values("duration_s")["mean_advantage_m"].values
d_adv = mtcn_df[mtcn_df["session"] == "Driver D"].sort_values("duration_s")["mean_advantage_m"].values
pool_adv = mtcn_df[mtcn_df["session"] == "POOLED (Driver B + D)"].sort_values("duration_s")["mean_advantage_m"].values

bars1 = ax.bar(x - width, b_adv, width, label="Driver B (Advantage vs CV)", color="#2ca02c", edgecolor="black")
bars2 = ax.bar(x, d_adv, width, label="Driver D (Advantage vs CV)", color="#17becf", edgecolor="black")
bars3 = ax.bar(x + width, pool_adv, width, label="Pooled B + D (Advantage vs CV)", color="#d62728", edgecolor="black")

ax.axhline(0, color="black", linestyle="-", linewidth=1.0)
ax.set_xticks(x)
ax.set_xticklabels(["10-Second Outages\n(CV Wins)", "30-Second Outages\n(ModernTCN Wins: +25.5m)", "60-Second Outages\n(ModernTCN Wins: +21.6m)"], fontsize=10.5, fontweight="bold")
ax.set_ylabel("ModernTCN Drift Reduction vs CV (meters)", fontsize=11, fontweight="bold")
ax.set_title("ModernTCN Drift Advantage over Constant-Velocity Baseline\n(Positive = ModernTCN Wins)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y", linestyle=":", alpha=0.6)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        y_pos = h + 1 if h >= 0 else h - 3
        ax.annotate(f"{h:+.1f}m", xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    ha="center", va=va, fontsize=8.5, fontweight="bold")

plt.tight_layout()
out_adv_png = figures_dir / "phase12_per_driver_comparison.png"
plt.savefig(out_adv_png, bbox_inches="tight")
plt.close()
print(f"Saved {out_adv_png}")
