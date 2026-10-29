"""
Phase 14 Publication Figures:
1. figures/phase14_calibration_protocols.png
2. figures/phase14_hybrid_validation.png
3. figures/phase14_reconciliation_bridge.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent

def plot_protocols():
    summary_path = repo_root / "experiments/phase14_validation/summary_metrics.csv"
    if not summary_path.exists():
        print("Run evaluate_phase14_causal.py first.")
        return
    df = pd.read_csv(summary_path)
    
    # Filter to POOLED across protocols for ModernTCN
    sub = df[(df["session"] == "POOLED") & (df["estimator"] == "ModernTCN")]
    
    protocols = ["Protocol_A_ZeroBias", "Protocol_B_PriorCalib", "Protocol_C_CausalOnline"]
    labels = ["Protocol A: Zero Bias (b=0)", "Protocol B: Prior Calib (Trip Start)", "Protocol C: Causal Online Calib"]
    colors = ["#4575b4", "#fee090", "#d73027"]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    durations = [10, 30, 60]
    x = np.arange(len(durations))
    width = 0.25
    
    # 1. Final Drift
    for i, proto in enumerate(protocols):
        vals = [sub[(sub["duration_s"] == d) & (sub["calibration_protocol"] == proto)]["mean_final_drift_m"].values[0] for d in durations]
        rects = axes[0].bar(x + (i - 1) * width, vals, width, label=labels[i], color=colors[i], alpha=0.9, edgecolor="black", linewidth=0.8)
        for rect in rects:
            h = rect.get_height()
            axes[0].annotate(f"{h:.1f}m", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    axes[0].set_title("ModernTCN Drift Under 3 Causal Calibration Protocols", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Outage Duration", fontsize=11)
    axes[0].set_ylabel("Final Drift (m)", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["10 Seconds", "30 Seconds", "60 Seconds"], fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9)
    
    # 2. Heading Error
    for i, proto in enumerate(protocols):
        vals = [sub[(sub["duration_s"] == d) & (sub["calibration_protocol"] == proto)]["mean_heading_err_deg"].values[0] for d in durations]
        rects = axes[1].bar(x + (i - 1) * width, vals, width, label=labels[i], color=colors[i], alpha=0.9, edgecolor="black", linewidth=0.8)
        for rect in rects:
            h = rect.get_height()
            axes[1].annotate(f"{h:.1f}°", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    axes[1].set_title("Heading Error Under 3 Causal Calibration Protocols", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Outage Duration", fontsize=11)
    axes[1].set_ylabel("Mean Heading Error (deg)", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["10 Seconds", "30 Seconds", "60 Seconds"], fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    out_file = repo_root / "figures/phase14_calibration_protocols.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")

def plot_hybrid_validation():
    summary_path = repo_root / "experiments/phase14_validation/summary_metrics.csv"
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    
    # Under Protocol C (Causal Online Calibration)
    sub = df[(df["session"] == "POOLED") & (df["calibration_protocol"] == "Protocol_C_CausalOnline")]
    
    estimators = ["Constant-Velocity", "ModernTCN", "Existing-Hybrid", "Corrected-Causal-Hybrid"]
    labels = ["Constant-Velocity", "ModernTCN", "Existing Hybrid (Phase 13)", "Corrected Causal Hybrid (Phase 14)"]
    colors = ["#e41a1c", "#377eb8", "#ff7f00", "#4daf4a"]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    durations = [10, 30, 60]
    x = np.arange(len(durations))
    width = 0.2
    
    for i, est in enumerate(estimators):
        vals = [sub[(sub["duration_s"] == d) & (sub["estimator"] == est)]["mean_final_drift_m"].values[0] for d in durations]
        rects = ax.bar(x + (i - 1.5) * width, vals, width, label=labels[i], color=colors[i], alpha=0.9, edgecolor="black", linewidth=0.8)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}m", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
            
    ax.set_title("Estimator Final Drift Under Protocol C (Causal Online Calibration)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Outage Duration", fontsize=11)
    ax.set_ylabel("Mean Final Positional Drift (m)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(["10 Seconds", "30 Seconds", "60 Seconds"], fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9.5)
    
    plt.tight_layout()
    out_file = repo_root / "figures/phase14_hybrid_validation.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    plot_protocols()
    plot_hybrid_validation()
