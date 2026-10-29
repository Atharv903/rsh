"""
Summarizes Phase 10 Independent-Drive Benchmark Results.
Outputs clean tables for both original repo heading convention and sign-corrected convention.
"""

import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("experiments/phase10_independent_validation/results/phase10_independent_outage_benchmarks.csv")
    df["duration_nominal"] = df["duration_s"].apply(lambda d: 10 if d < 20 else (30 if d < 45 else 60))

    def get_arch(m):
        if "cnngru" in m: return "CNN-GRU"
        if "mamba2" in m: return "Mamba-2"
        if "moderntcn" in m: return "ModernTCN"
        return m

    df["arch"] = df["method"].apply(get_arch)

    for h_conv in ["original_repo (+1)", "sign_corrected (-1)"]:
        print("=" * 105)
        print(f"HEADING CONVENTION: {h_conv}")
        print("=" * 105)
        sub = df[df["heading_convention"] == h_conv]
        
        # 1. By Architecture (averaged over 5 trials x 3 seeds)
        arch_sum = sub.groupby(["duration_nominal", "arch"])[["speed_mae_mps", "ate_rmse_m", "final_drift_m", "drift_pct"]].mean().reset_index()
        arch_sum = arch_sum.sort_values(["duration_nominal", "final_drift_m"])
        print("--- Architecture Averages (across 5 trials x 3 seeds = 15 evaluations per duration) ---")
        print(f"{'Duration':10s} | {'Architecture':18s} | {'Speed MAE':12s} | {'ATE RMSE':12s} | {'Final Drift':14s} | {'Drift %':10s}")
        print("-" * 95)
        for _, r in arch_sum.iterrows():
            d = int(r['duration_nominal'])
            arch = r['arch']
            mae = r['speed_mae_mps']
            ate = r['ate_rmse_m']
            drift = r['final_drift_m']
            pct = r['drift_pct']
            print(f"{d:2d} seconds  | {arch:18s} | {mae:9.2f} m/s | {ate:9.2f} m | {drift:10.2f} m | {pct:8.1f} %")
            
        print("\n--- Per-Model & Seed Breakdown ---")
        seed_sum = sub.groupby(["duration_nominal", "method"])[["speed_mae_mps", "ate_rmse_m", "final_drift_m", "drift_pct"]].mean().reset_index()
        seed_sum = seed_sum.sort_values(["duration_nominal", "final_drift_m"])
        for dur in [10, 30, 60]:
            print(f">> {dur}-Second Outages:")
            for _, r in seed_sum[seed_sum["duration_nominal"] == dur].iterrows():
                m = r['method']
                mae = r['speed_mae_mps']
                ate = r['ate_rmse_m']
                drift = r['final_drift_m']
                pct = r['drift_pct']
                print(f"   {m:24s} | Speed MAE: {mae:5.2f} m/s | ATE: {ate:6.2f} m | Drift: {drift:6.2f} m ({pct:5.1f}%)")

if __name__ == "__main__":
    main()
