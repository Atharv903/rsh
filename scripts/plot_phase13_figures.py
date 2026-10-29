"""
Publication-Quality Visualization for Phase 13 Research Audit.

Generates:
1. figures/phase13_gyro_bias_impact.png (Drift, Cross-Track Error, Heading Error across bias strategies)
2. figures/phase13_hybrid_vs_baselines.png (Constant Velocity vs ModernTCN vs Adaptive-Hybrid)
3. figures/phase13_forensic_heteroscedasticity.png (Driver B vs Driver D paired differences & outliers)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent

def plot_gyro_bias_impact():
    summary_path = repo_root / "experiments/phase13/gyro_bias_summary.csv"
    if not summary_path.exists():
        print(f"File {summary_path} does not exist.")
        return
    df = pd.read_csv(summary_path)
    
    # Filter to ModernTCN across durations
    m_df = df[df["method"] == "ModernTCN"]
    
    strategies = ["legacy_pre_median", "zero_bias", "global_stationary", "causal_pre_stationary"]
    strat_labels = ["Phase 12 (Moving Median)", "Zero Bias (b=0)", "Global Stationary", "Causal Pre-Stationary"]
    colors = ["#d95f02", "#7570b3", "#1b9e77", "#2b83ba"]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    durations = [10, 30, 60]
    x = np.arange(len(durations))
    width = 0.2
    
    # 1. Final Drift
    for i, strat in enumerate(strategies):
        vals = [m_df[(m_df["duration_s"] == d) & (m_df["bias_strategy"] == strat)]["final_drift_m"].values[0] for d in durations]
        axes[0].bar(x + (i - 1.5) * width, vals, width, label=strat_labels[i], color=colors[i], alpha=0.9, edgecolor="black", linewidth=0.8)
    axes[0].set_title("Mean Positional Drift by Bias Strategy", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Outage Duration (s)", fontsize=11)
    axes[0].set_ylabel("Final Drift (m)", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["10s", "30s", "60s"], fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9)
    
    # 2. Cross-Track Error
    for i, strat in enumerate(strategies):
        vals = [abs(m_df[(m_df["duration_s"] == d) & (m_df["bias_strategy"] == strat)]["cross_track_err_m"].values[0]) for d in durations]
        axes[1].bar(x + (i - 1.5) * width, vals, width, label=strat_labels[i], color=colors[i], alpha=0.9, edgecolor="black", linewidth=0.8)
    axes[1].set_title("Absolute Cross-Track Error", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Outage Duration (s)", fontsize=11)
    axes[1].set_ylabel("Cross-Track Error (m)", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["10s", "30s", "60s"], fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    
    # 3. Final Heading Error
    for i, strat in enumerate(strategies):
        vals = [m_df[(m_df["duration_s"] == d) & (m_df["bias_strategy"] == strat)]["final_heading_err_deg"].values[0] for d in durations]
        axes[2].bar(x + (i - 1.5) * width, vals, width, label=strat_labels[i], color=colors[i], alpha=0.9, edgecolor="black", linewidth=0.8)
    axes[2].set_title("Final Heading Error", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Outage Duration (s)", fontsize=11)
    axes[2].set_ylabel("Heading Error (deg)", fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["10s", "30s", "60s"], fontsize=11)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    out_file = repo_root / "figures/phase13_gyro_bias_impact.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")

def plot_hybrid_performance():
    summary_path = repo_root / "experiments/phase13/hybrid_evaluation_summary.csv"
    if not summary_path.exists():
        print(f"File {summary_path} does not exist.")
        return
    df = pd.read_csv(summary_path)
    
    # Filter to causal stationary bias
    c_df = df[df["bias_strategy"] == "causal_pre_stationary"]
    
    methods = ["Constant-Velocity", "ModernTCN", "Adaptive-Hybrid"]
    labels = ["Constant Velocity (Baseline)", "ModernTCN (Neural)", "Adaptive-Hybrid (Ours)"]
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    durations = [10, 30, 60]
    x = np.arange(len(durations))
    width = 0.25
    
    # 1. Final Drift
    for i, m in enumerate(methods):
        vals = [c_df[(c_df["duration_s"] == d) & (c_df["method"] == m)]["final_drift_m"].values[0] for d in durations]
        rects = axes[0].bar(x + (i - 1) * width, vals, width, label=labels[i], color=colors[i], alpha=0.9, edgecolor="black", linewidth=0.8)
        for rect in rects:
            height = rect.get_height()
            axes[0].annotate(f"{height:.1f}m",
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=9, fontweight='bold')
                             
    axes[0].set_title("Final Positional Drift: Baseline vs Neural vs Adaptive-Hybrid", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Outage Duration", fontsize=11)
    axes[0].set_ylabel("Mean Final Drift (m)", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["10 Seconds", "30 Seconds", "60 Seconds"], fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9.5)
    
    # 2. Speed MAE
    for i, m in enumerate(methods):
        vals = [c_df[(c_df["duration_s"] == d) & (c_df["method"] == m)]["speed_mae_mps"].values[0] for d in durations]
        rects = axes[1].bar(x + (i - 1) * width, vals, width, label=labels[i], color=colors[i], alpha=0.9, edgecolor="black", linewidth=0.8)
        for rect in rects:
            height = rect.get_height()
            axes[1].annotate(f"{height:.2f} m/s",
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=9, fontweight='bold')
                             
    axes[1].set_title("Speed MAE Across Outage Durations", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Outage Duration", fontsize=11)
    axes[1].set_ylabel("Speed MAE (m/s)", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["10 Seconds", "30 Seconds", "60 Seconds"], fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    out_file = repo_root / "figures/phase13_hybrid_performance.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")

def plot_heteroscedasticity():
    disc_path = repo_root / "experiments/phase13/statistical_discrepancy_analysis.json"
    if not disc_path.exists():
        print(f"File {disc_path} does not exist.")
        return
    import json
    with open(disc_path) as f:
        data = json.load(f)
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 30s Paired Differences
    db_30 = next(r for r in data if r["session"] == "Driver B" and r["duration_s"] == 30 and r["arch"] == "ModernTCN")
    dd_30 = next(r for r in data if r["session"] == "Driver D" and r["duration_s"] == 30 and r["arch"] == "ModernTCN")
    
    axes[0].axhline(0, color="black", linestyle="--", alpha=0.7)
    axes[0].scatter(np.repeat(1, len(db_30["diff_values"])), db_30["diff_values"], color="#1f78b4", s=90, label=f"Driver B (std={db_30['std_diff']:.1f}m)", zorder=3)
    axes[0].scatter(np.repeat(2, len(dd_30["diff_values"])), dd_30["diff_values"], color="#e31a1c", s=90, label=f"Driver D (std={dd_30['std_diff']:.1f}m)", zorder=3)
    
    # Mean and CIs
    axes[0].errorbar(1.15, db_30["mean_diff"], yerr=[[db_30["mean_diff"] - db_30["t_interval"][0]], [db_30["t_interval"][1] - db_30["mean_diff"]]], fmt="o", color="#1f78b4", capsize=5, capthick=1.5, elinewidth=1.5, label="Driver B 95% t-CI")
    axes[0].errorbar(2.15, dd_30["mean_diff"], yerr=[[dd_30["mean_diff"] - dd_30["t_interval"][0]], [dd_30["t_interval"][1] - dd_30["mean_diff"]]], fmt="o", color="#e31a1c", capsize=5, capthick=1.5, elinewidth=1.5, label="Driver D 95% t-CI")
    
    axes[0].set_title("30-Second Paired Difference Distribution by Driver", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("ModernTCN Advantage over CV (m)", fontsize=11)
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Driver B (M)", "Driver D (Y1)"], fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9)
    
    # 60s Paired Differences
    db_60 = next(r for r in data if r["session"] == "Driver B" and r["duration_s"] == 60 and r["arch"] == "ModernTCN")
    dd_60 = next(r for r in data if r["session"] == "Driver D" and r["duration_s"] == 60 and r["arch"] == "ModernTCN")
    
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.7)
    axes[1].scatter(np.repeat(1, len(db_60["diff_values"])), db_60["diff_values"], color="#1f78b4", s=90, label=f"Driver B (std={db_60['std_diff']:.1f}m)", zorder=3)
    axes[1].scatter(np.repeat(2, len(dd_60["diff_values"])), dd_60["diff_values"], color="#e31a1c", s=90, label=f"Driver D (std={dd_60['std_diff']:.1f}m)", zorder=3)
    
    axes[1].errorbar(1.15, db_60["mean_diff"], yerr=[[db_60["mean_diff"] - db_60["t_interval"][0]], [db_60["t_interval"][1] - db_60["mean_diff"]]], fmt="o", color="#1f78b4", capsize=5, capthick=1.5, elinewidth=1.5)
    axes[1].errorbar(2.15, dd_60["mean_diff"], yerr=[[dd_60["mean_diff"] - dd_60["t_interval"][0]], [dd_60["t_interval"][1] - dd_60["mean_diff"]]], fmt="o", color="#e31a1c", capsize=5, capthick=1.5, elinewidth=1.5)
    
    axes[1].set_title("60-Second Paired Difference Distribution by Driver", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("ModernTCN Advantage over CV (m)", fontsize=11)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["Driver B (M)", "Driver D (Y1)"], fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    out_file = repo_root / "figures/phase13_forensic_heteroscedasticity.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    plot_gyro_bias_impact()
    plot_heteroscedasticity()
    plot_hybrid_performance()
