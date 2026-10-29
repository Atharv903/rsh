"""
Detailed breakdown of the 60-second outages to investigate why Constant Velocity
has lower speed MAE than ModernTCN despite having higher final position drift.
"""

import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("experiments/phase10_independent_validation/results/phase10_independent_outage_benchmarks.csv")
    df["dur_nom"] = df["duration_s"].apply(lambda d: 10 if d < 20 else (30 if d < 45 else 60))

    sub = df[(df["dur_nom"] == 60) & (df["heading_convention"] == "sign_corrected (-1)")]
    cols = ["scenario_id", "method", "speed_mae_mps", "speed_bias_mps", "ate_rmse_m", "final_drift_m", "drift_pct"]

    cv_df = sub[sub["method"] == "Constant-Velocity"][cols]
    nn_df = sub[sub["method"] == "moderntcnmodel_seed42"][cols]

    print("=" * 80)
    print("60-SECOND OUTAGES: SCENARIO-BY-SCENARIO DRIFT & SPEED ERROR AUDIT")
    print("=" * 80)
    merged = pd.merge(cv_df, nn_df, on="scenario_id", suffixes=("_CV", "_ModernTCN42"))
    for _, r in merged.iterrows():
        sid = r["scenario_id"]
        mae_cv, bias_cv, drift_cv = r["speed_mae_mps_CV"], r["speed_bias_mps_CV"], r["final_drift_m_CV"]
        mae_nn, bias_nn, drift_nn = r["speed_mae_mps_ModernTCN42"], r["speed_bias_mps_ModernTCN42"], r["final_drift_m_ModernTCN42"]
        diff = drift_cv - drift_nn
        print(f"Scenario: {sid}")
        print(f"  Constant-Velocity: Speed MAE = {mae_cv:5.2f} m/s | Bias = {bias_cv:+.2f} m/s | Drift = {drift_cv:6.2f} m ({r['drift_pct_CV']:.1f}%)")
        print(f"  ModernTCN (s42)  : Speed MAE = {mae_nn:5.2f} m/s | Bias = {bias_nn:+.2f} m/s | Drift = {drift_nn:6.2f} m ({r['drift_pct_ModernTCN42']:.1f}%)")
        print(f"  -> Drift Advantage of ModernTCN: {diff:+.2f} m\n")

if __name__ == "__main__":
    main()
