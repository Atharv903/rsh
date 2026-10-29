"""
Generates publication-quality figures for Phase 11 Navigation Integrity Audit:
1. figures/phase11_error_decomposition.png: Along-track vs Cross-track error decomposition.
2. figures/phase11_statistical_distribution.png: Boxplot of drift across durations.
"""

import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# 1. Along-track vs Cross-track Error Bar Plot (60s outages)
decomp_df = pd.read_csv("experiments/phase11_navigation_integrity/results/phase11_vector_error_decomposition.csv")
s60 = decomp_df[decomp_df["duration_s"] == 60].reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# (a) Error Decomposition
x = np.arange(len(s60))
width = 0.2

ax = axes[0]
ax.bar(x - 1.5*width, s60["cv_along_track_err_m"], width, label="CV Along-Track (Speed)", color="#1f77b4", alpha=0.85)
ax.bar(x - 0.5*width, s60["cv_cross_track_err_m"], width, label="CV Cross-Track (Heading)", color="#aec7e8", alpha=0.85)
ax.bar(x + 0.5*width, s60["nn_along_track_err_m"], width, label="ModernTCN Along-Track (Speed)", color="#d62728", alpha=0.85)
ax.bar(x + 1.5*width, s60["nn_cross_track_err_m"], width, label="ModernTCN Cross-Track (Heading)", color="#ff9896", alpha=0.85)

ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"Trial {i+1}" for i in x], fontsize=11, fontweight="bold")
ax.set_ylabel("Signed Error (meters)", fontsize=11, fontweight="bold")
ax.set_title("Along-Track vs Cross-Track Vector Error (60s Outages)", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="lower left")
ax.grid(axis="y", linestyle=":", alpha=0.6)

# (b) Total Drift Comparison
ax2 = axes[1]
bars1 = ax2.bar(x - width/2, s60["cv_drift_m"], width, label="Constant-Velocity", color="#1f77b4", edgecolor="black")
bars2 = ax2.bar(x + width/2, s60["nn_drift_m"], width, label="ModernTCN (Seed 42)", color="#d62728", edgecolor="black")

for b in list(bars1) + list(bars2):
    h = b.get_height()
    ax2.annotate(f"{h:.0f}m", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels([f"Trial {i+1}" for i in x], fontsize=11, fontweight="bold")
ax2.set_ylabel("Final Position Drift (meters)", fontsize=11, fontweight="bold")
ax2.set_title("Total Positional Drift (ModernTCN wins in 4/5 Trials)", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9.5, loc="upper left")
ax2.grid(axis="y", linestyle=":", alpha=0.6)

plt.suptitle("Phase 11: Vector Error Decomposition & Discrepancy Resolution", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
out_decomp_fig = figures_dir / "phase11_error_decomposition.png"
plt.savefig(out_decomp_fig, bbox_inches="tight")
plt.close()
print(f"Saved {out_decomp_fig}")

# 2. Statistical Boxplot across Durations
bench_df = pd.read_csv("experiments/phase10_independent_validation/results/phase10_independent_outage_benchmarks.csv")
bench_df["dur_nom"] = bench_df["duration_s"].apply(lambda d: 10 if d < 20 else (30 if d < 45 else 60))
sub = bench_df[bench_df["heading_convention"] == "sign_corrected (-1)"]

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
positions = [1, 2, 4, 5, 7, 8]
data = [
    sub[(sub["dur_nom"] == 10) & (sub["method"] == "Constant-Velocity")]["final_drift_m"].values,
    sub[(sub["dur_nom"] == 10) & (sub["method"].str.contains("moderntcn"))]["final_drift_m"].values,
    sub[(sub["dur_nom"] == 30) & (sub["method"] == "Constant-Velocity")]["final_drift_m"].values,
    sub[(sub["dur_nom"] == 30) & (sub["method"].str.contains("moderntcn"))]["final_drift_m"].values,
    sub[(sub["dur_nom"] == 60) & (sub["method"] == "Constant-Velocity")]["final_drift_m"].values,
    sub[(sub["dur_nom"] == 60) & (sub["method"].str.contains("moderntcn"))]["final_drift_m"].values,
]

bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                boxprops=dict(facecolor="lightblue", alpha=0.7),
                medianprops=dict(color="black", linewidth=1.5))

colors = ["#1f77b4", "#d62728", "#1f77b4", "#d62728", "#1f77b4", "#d62728"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticks([1.5, 4.5, 7.5])
ax.set_xticklabels(["10s Outage\n(CV Wins)", "30s Outage\n(ModernTCN Wins)", "60s Outage\n(ModernTCN Wins)"], fontsize=11, fontweight="bold")
ax.set_ylabel("Final Drift (meters)", fontsize=11, fontweight="bold")
ax.set_title("Distribution of Final Drift: Constant-Velocity vs ModernTCN", fontsize=13, fontweight="bold")

# Custom Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#1f77b4", alpha=0.7, label="Constant-Velocity"),
                   Patch(facecolor="#d62728", alpha=0.7, label="ModernTCN (All Seeds)")]
ax.legend(handles=legend_elements, fontsize=10, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
out_box_fig = figures_dir / "phase11_statistical_distribution.png"
plt.savefig(out_box_fig, bbox_inches="tight")
plt.close()
print(f"Saved {out_box_fig}")
