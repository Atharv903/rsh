"""
Master Experiment Runner.
Executes training and evaluation for CNN-GRU, ModernTCN, and Mamba-2 across multiple seeds,
computes statistical summaries, and generates comparison artifacts.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add repo root to python path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import torch

from src.train import train_single_run
from src.evaluate import load_model, evaluate_test_set, evaluate_outage_trajectories


def run_experiments():
    print("=" * 60, flush=True)
    print("STARTING BENCHMARK: CNN-GRU vs. ModernTCN vs. Mamba-2", flush=True)
    print("=" * 60, flush=True)
    
    models_to_run = [
        ("configs/cnn_gru.yaml", "CNN-GRU Baseline"),
        ("configs/moderntcn.yaml", "ModernTCN"),
        ("configs/mamba2.yaml", "Mamba-2 SSD"),
    ]
    
    output_dir = Path("checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if training results already exist
    training_res_file = Path("reports/training_benchmark_results.json")
    if training_res_file.exists():
        with open(training_res_file, "r") as f:
            all_results = json.load(f)
        print("Loaded existing training benchmark results for all models.", flush=True)
    else:
        all_results = {}
        # 1. Training loop across models and seeds
        for cfg_path, display_name in models_to_run:
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
                
            m_name = cfg["model_name"]
            seeds = cfg["training"].get("seeds", [42, 123, 999])
            all_results[m_name] = []
            
            print(f"\n>>> Model: {display_name} ({m_name}) | Seeds: {seeds} <<<", flush=True)
            for s in seeds:
                print(f"\n--- Training {m_name} (Seed {s}) ---", flush=True)
                res = train_single_run(cfg, s, output_dir, smoke_test=False)
                all_results[m_name].append(res)
                
        # Save training summary
        with open("reports/training_benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
    print("\n" + "=" * 60, flush=True)
    print("TRAINING COMPLETE. EVALUATING ON TEST SCENARIOS...", flush=True)
    print("=" * 60, flush=True)
    
    # 2. Outage Trajectory Evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outage_scenarios_path = "data/processed/test_outage_scenarios.json"
    if not Path(outage_scenarios_path).exists():
        outage_scenarios_path = "data/processed/outage_scenarios.json"
        
    processed_dir = "data/processed"
    normalizer_path = "data/processed/sensor_normalizer.json"
    
    outage_evals = {}
    
    for cfg_path, _ in models_to_run:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        m_name = cfg["model_name"]
        seeds = cfg["training"].get("seeds", [42, 123, 999])
        outage_evals[m_name] = {}
        
        for s in seeds:
            ckpt_path = output_dir / f"{m_name}_seed{s}" / "best_model.pt"
            if not ckpt_path.exists():
                continue
            model, _ = load_model(ckpt_path, device)
            
            # Evaluate test set
            test_res = evaluate_test_set(model, cfg["data"]["test_npz"], device)
            
            # Evaluate outages
            sc_res = evaluate_outage_trajectories(
                model=model,
                outage_scenarios_path=outage_scenarios_path,
                processed_dir=processed_dir,
                normalizer_path=normalizer_path,
                device=device,
            )
            
            outage_evals[m_name][f"seed_{s}"] = {
                "test_vel_mae": test_res["vel_mae"],
                "test_vel_rmse": test_res["vel_rmse"],
                "scenarios": sc_res,
            }
            
    with open("reports/outage_evaluation_results.json", "w") as f:
        json.dump(outage_evals, f, indent=2)
        
    # 3. Compile Metric Summary Tables
    compile_summary_tables(all_results, outage_evals)
    
    # 4. Generate Figures
    generate_comparison_figures(all_results, outage_evals)
    
    print("\n" + "=" * 60, flush=True)
    print("BENCHMARK EXECUTION SUCCESSFULLY FINISHED!", flush=True)
    print("=" * 60, flush=True)


def compile_summary_tables(all_results: Dict[str, Any], outage_evals: Dict[str, Any]):
    summary_rows = []
    
    for m_name, runs in all_results.items():
        params = runs[0]["parameters"]
        latencies = [r["latency_ms"] for r in runs]
        train_times = [r["total_train_time_s"] for r in runs]
        test_maes = [r["test_vel_mae"] for r in runs]
        test_rmses = [r["test_vel_rmse"] for r in runs]
        
        # Outage metrics for 10s, 30s, 60s
        outage_10s_ates = []
        outage_10s_drifts = []
        outage_30s_ates = []
        outage_30s_drifts = []
        outage_60s_ates = []
        outage_60s_drifts = []
        
        if m_name in outage_evals:
            for seed_k, s_data in outage_evals[m_name].items():
                for sc in s_data.get("scenarios", []):
                    dur = sc["duration_s"]
                    if dur == 10:
                        outage_10s_ates.append(sc["ate_rmse_m"])
                        outage_10s_drifts.append(sc["drift_pct"])
                    elif dur == 30:
                        outage_30s_ates.append(sc["ate_rmse_m"])
                        outage_30s_drifts.append(sc["drift_pct"])
                    elif dur == 60:
                        outage_60s_ates.append(sc["ate_rmse_m"])
                        outage_60s_drifts.append(sc["drift_pct"])
                        
        summary_rows.append({
            "Model": m_name.upper(),
            "Params": params,
            "Latency (ms)": f"{np.mean(latencies):.2f} ± {np.std(latencies):.2f}",
            "Train Time (s)": f"{np.mean(train_times):.1f}",
            "Test Vel MAE (m/s)": f"{np.mean(test_maes):.3f} ± {np.std(test_maes):.3f}",
            "Test Vel RMSE (m/s)": f"{np.mean(test_rmses):.3f} ± {np.std(test_rmses):.3f}",
            "10s Outage ATE (m)": f"{np.mean(outage_10s_ates):.2f} ± {np.std(outage_10s_ates):.2f}" if outage_10s_ates else "N/A",
            "10s Drift (%)": f"{np.mean(outage_10s_drifts):.2f}% ± {np.std(outage_10s_drifts):.2f}%" if outage_10s_drifts else "N/A",
            "30s Outage ATE (m)": f"{np.mean(outage_30s_ates):.2f} ± {np.std(outage_30s_ates):.2f}" if outage_30s_ates else "N/A",
            "30s Drift (%)": f"{np.mean(outage_30s_drifts):.2f}% ± {np.std(outage_30s_drifts):.2f}%" if outage_30s_drifts else "N/A",
            "60s Outage ATE (m)": f"{np.mean(outage_60s_ates):.2f} ± {np.std(outage_60s_ates):.2f}" if outage_60s_ates else "N/A",
            "60s Drift (%)": f"{np.mean(outage_60s_drifts):.2f}% ± {np.std(outage_60s_drifts):.2f}%" if outage_60s_drifts else "N/A",
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("reports/metrics_table.csv", index=False)
    print("\n=== BENCHMARK METRIC TABLE ===")
    print(df_summary.to_string(index=False), flush=True)


def generate_comparison_figures(all_results: Dict[str, Any], outage_evals: Dict[str, Any]):
    figs_dir = Path("figures")
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Training Loss Curves
    plt.figure(figsize=(10, 6), dpi=300)
    colors = {"cnn_gru": "#1f77b4", "moderntcn": "#2ca02c", "mamba2": "#d62728"}
    labels = {"cnn_gru": "CNN-GRU Baseline", "moderntcn": "ModernTCN", "mamba2": "Mamba-2 SSD"}
    
    for m_name, runs in all_results.items():
        # Plot mean train & val loss
        epochs = len(runs[0]["history"])
        val_losses = np.array([[h["val_loss"] for h in r["history"][:epochs]] for r in runs])
        mean_val = np.mean(val_losses, axis=0)
        std_val = np.std(val_losses, axis=0)
        ep_arr = np.arange(1, len(mean_val) + 1)
        
        plt.plot(ep_arr, mean_val, label=f"{labels.get(m_name, m_name)} (Val)", color=colors.get(m_name, "black"), linewidth=2)
        plt.fill_between(ep_arr, mean_val - std_val, mean_val + std_val, alpha=0.15, color=colors.get(m_name, "black"))
        
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Huber Loss", fontsize=12)
    plt.title("Validation Loss Progression Across Random Seeds (Mean ± 1 SD)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(figs_dir / "training_loss_curves.png")
    plt.close()
    
    # 2. Outage Trajectory Plots for 10s, 30s, 60s
    for target_dur in [10, 30, 60]:
        plt.figure(figsize=(9, 8), dpi=300)
        plotted_gt = False
        
        for m_name in ["cnn_gru", "moderntcn", "mamba2"]:
            if m_name not in outage_evals:
                continue
            # Pick seed 42
            s_data = outage_evals[m_name].get("seed_42", {})
            scenarios = [s for s in s_data.get("scenarios", []) if s["duration_s"] == target_dur]
            if not scenarios:
                continue
            sample_sc = scenarios[0]
            
            if not plotted_gt:
                plt.plot(sample_sc["gt_east"], sample_sc["gt_north"], "k-", linewidth=2.5, label="Ground Truth Reference")
                plt.plot(sample_sc["gt_east"][0], sample_sc["gt_north"][0], "go", markersize=8, label="Outage Start Fix")
                plt.plot(sample_sc["gt_east"][-1], sample_sc["gt_north"][-1], "rs", markersize=8, label="Ground Truth End")
                plotted_gt = True
                
            plt.plot(
                sample_sc["pred_east"],
                sample_sc["pred_north"],
                linestyle="--",
                linewidth=2,
                color=colors.get(m_name, "blue"),
                label=f"{labels.get(m_name, m_name)} (ATE: {sample_sc['ate_rmse_m']:.1f}m, Drift: {sample_sc['drift_pct']:.1f}%)"
            )
            
        plt.xlabel("Local East Displacement (m)", fontsize=12)
        plt.ylabel("Local North Displacement (m)", fontsize=12)
        plt.title(f"Simulated {target_dur}s GNSS Outage: Dead-Reckoning Trajectory Integration", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="best", fontsize=10)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(figs_dir / f"trajectory_comparison_{target_dur}s.png")
        plt.close()
        
    print(f"Generated comparison figures in {figs_dir}/", flush=True)


if __name__ == "__main__":
    run_experiments()
