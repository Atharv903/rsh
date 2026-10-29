import pandas as pd
import numpy as np

df_abl = pd.read_csv("experiments/phase15_validation/error_ablation_summary.csv")
df_reg = pd.read_csv("experiments/phase15_validation/error_regime_summary.csv")

print("--- ABLATION SUMMARY (Mean Final Drift in Metres) ---")
for dur in [10, 30, 60]:
    sub = df_abl[df_abl["duration_s"] == dur]
    print(f"\nDuration: {dur}s (N = {len(sub)})")
    print(f"  Actual Hybrid Drift:       mean = {sub['drift_hybrid_actual_m'].mean():.2f} m, median = {sub['drift_hybrid_actual_m'].median():.2f} m")
    print(f"  Perfect Speed (Noisy Gyro): mean = {sub['drift_perfect_speed_m'].mean():.2f} m, median = {sub['drift_perfect_speed_m'].median():.2f} m")
    print(f"  Perfect Gyro (Noisy Speed): mean = {sub['drift_perfect_heading_m'].mean():.2f} m, median = {sub['drift_perfect_heading_m'].median():.2f} m")

print("\n--- REGIME-BASED SPEED MAE (m/s) ---")
for dur in [10, 30, 60]:
    sub = df_reg[df_reg["duration_s"] == dur]
    print(f"\nDuration: {dur}s:")
    print(f"  Overall Speed MAE:  {sub['speed_mae_overall'].mean():.2f} m/s")
    print(f"  Cruise Speed MAE:   {sub['speed_mae_cruise'].mean():.2f} m/s")
    print(f"  Accel Speed MAE:    {sub['speed_mae_accel'].mean():.2f} m/s")
    print(f"  Braking Speed MAE:  {sub['speed_mae_brake'].mean():.2f} m/s")
    print(f"  Init Speed Error:   {sub['init_speed_error_mps'].mean():.2f} m/s")
    print(f"  Init Heading Error: {sub['init_heading_error_deg'].mean():.2f} deg")
