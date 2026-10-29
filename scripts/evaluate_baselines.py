"""
Script to evaluate fair navigation baselines across 10s, 30s, and 60s simulated outages.
Evaluates on the independent holdout test set (Drive S4) using strictly legitimate pre-outage
state and causal IMU observations.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Add project root
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.corrected_synchronization import extract_aligned_segments_for_drive
from src.outage_estimator import (
    PreOutageState,
    estimate_initial_heading_from_pre_outage_gps,
    run_causal_dead_reckoning,
    make_zero_speed_estimator,
    make_last_known_speed_estimator,
    make_imu_kinematic_estimator,
)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})


def generate_test_outage_scenarios(test_df: pd.DataFrame, seed: int = 42) -> List[Dict[str, Any]]:
    np.random.seed(seed)
    durations = [10, 30, 60]  # seconds
    scenarios = []
    
    # Use Segment 0 of Drive S4 (rows 0 to 35,185, length 3,518s)
    # Available range: [200, 34000]
    total_len = len(test_df)
    
    for dur in durations:
        n_samples = dur * 10
        # Sample 5 diverse outage start points per duration
        # Pick intervals with active driving (speed > 3 m/s)
        valid_starts = []
        for cand in range(500, total_len - n_samples - 500, 1000):
            seg_spd = test_df["ref_speed_mps"].iloc[cand : cand + n_samples].values
            if np.mean(seg_spd) > 3.0:
                valid_starts.append(cand)
                
        selected_starts = np.random.choice(valid_starts, size=min(5, len(valid_starts)), replace=False)
        selected_starts.sort()
        
        for idx, st in enumerate(selected_starts):
            scenarios.append({
                "scenario_id": f"S4_{dur}s_trial{idx+1}",
                "drive_id": "S4",
                "duration_s": dur,
                "start_idx": int(st),
                "end_idx": int(st + n_samples),
                "pre_window_samples": 100,
            })
            
    return scenarios


def main():
    print("=" * 70)
    print("EVALUATING FAIR NAVIGATION BASELINES ON TEST HOLDOUT (S4)")
    print("=" * 70)
    
    os.makedirs("figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Load aligned Drive S4 Segment 0
    segs, _ = extract_aligned_segments_for_drive("S4", raw_dir="data/raw")
    test_df = segs[0]  # 35,186 continuous samples
    
    scenarios = generate_test_outage_scenarios(test_df, seed=42)
    print(f"Generated {len(scenarios)} matched simulated outage scenarios across 10s, 30s, 60s durations.")
    
    baseline_results = []
    
    for sc in scenarios:
        st = sc["start_idx"]
        end = sc["end_idx"]
        pre_st = max(0, st - sc["pre_window_samples"])
        
        pre_df = test_df.iloc[pre_st:st].copy()
        in_outage_df = test_df.iloc[st:end].copy()
        
        # 1. Legitimate Pre-outage State Extraction
        heading_deg, last_spd = estimate_initial_heading_from_pre_outage_gps(pre_df)
        pos_e0 = float(pre_df["ref_east_m"].iloc[-1])
        pos_n0 = float(pre_df["ref_north_m"].iloc[-1])
        
        # Gyro bias from stationary pre-outage or pre-outage median
        gyro_bias = float(np.median(pre_df["gyro_pitch"]))
        
        pre_state = PreOutageState(
            t0_s=float(pre_df["time_s"].iloc[-1]),
            pos_east_m=pos_e0,
            pos_north_m=pos_n0,
            speed_mps=last_spd,
            heading_deg=heading_deg,
            gyro_bias_rads=gyro_bias,
        )
        
        # -------------------------------------------------------------
        # Baseline 1: Zero-Speed
        # -------------------------------------------------------------
        res_zero = run_causal_dead_reckoning(
            pre_state=pre_state,
            in_outage_df=in_outage_df,
            speed_estimator_fn=make_zero_speed_estimator(),
            method_name="Zero-Speed",
            scenario_id=sc["scenario_id"],
            drive_id="S4",
        )
        baseline_results.append(res_zero)
        
        # -------------------------------------------------------------
        # Baseline 2: Last-Known-Speed Persistence
        # -------------------------------------------------------------
        res_persist = run_causal_dead_reckoning(
            pre_state=pre_state,
            in_outage_df=in_outage_df,
            speed_estimator_fn=make_last_known_speed_estimator(last_spd),
            method_name="Last-Known-Speed",
            scenario_id=sc["scenario_id"],
            drive_id="S4",
        )
        baseline_results.append(res_persist)
        
        # -------------------------------------------------------------
        # Baseline 3: Constant-Velocity (Speed + Straight Line Heading)
        # -------------------------------------------------------------
        # In straight line heading, gyro_pitch is zeroed
        in_outage_straight = in_outage_df.copy()
        in_outage_straight["gyro_pitch"] = 0.0
        pre_state_straight = PreOutageState(
            t0_s=pre_state.t0_s,
            pos_east_m=pos_e0,
            pos_north_m=pos_n0,
            speed_mps=last_spd,
            heading_deg=heading_deg,
            gyro_bias_rads=0.0,
        )
        res_const = run_causal_dead_reckoning(
            pre_state=pre_state_straight,
            in_outage_df=in_outage_straight,
            speed_estimator_fn=make_last_known_speed_estimator(last_spd),
            method_name="Constant-Velocity",
            scenario_id=sc["scenario_id"],
            drive_id="S4",
        )
        baseline_results.append(res_const)
        
        # -------------------------------------------------------------
        # Baseline 4: IMU Kinematic Integration
        # -------------------------------------------------------------
        res_imu = run_causal_dead_reckoning(
            pre_state=pre_state,
            in_outage_df=in_outage_df,
            speed_estimator_fn=make_imu_kinematic_estimator(last_spd, fwd_axis_idx=0),
            method_name="IMU-Kinematic",
            scenario_id=sc["scenario_id"],
            drive_id="S4",
        )
        baseline_results.append(res_imu)
        
    # Aggregate and Tabulate Results
    df_res = pd.DataFrame([{
        "scenario_id": r.scenario_id,
        "duration_s": r.duration_s,
        "method": r.method_name,
        "length_m": r.total_length_m,
        "speed_mae": r.speed_mae_mps,
        "speed_rmse": r.speed_rmse_mps,
        "ate_rmse": r.ate_rmse_m,
        "final_drift_m": r.final_drift_m,
        "drift_pct": r.drift_pct,
    } for r in baseline_results])
    
    # Save CSV
    df_res.to_csv("reports/phase7_baseline_results.csv", index=False)
    
    # Group by Duration and Method
    summary = df_res.groupby(["duration_s", "method"])[["speed_mae", "ate_rmse", "final_drift_m", "drift_pct"]].mean().reset_index()
    print("\n--- Summary Performance Across Outage Durations ---")
    print(summary.to_string(index=False))
    
    # Save JSON summary
    summary_dict = summary.to_dict(orient="records")
    with open("reports/phase7_baseline_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)
    print("\nSaved reports/phase7_baseline_summary.json and reports/phase7_baseline_results.csv")
    
    # -------------------------------------------------------------------------
    # Publication Trajectory Comparison Plot (60s Outage Trial)
    # -------------------------------------------------------------------------
    # Find a 60s trial
    t60_runs = [r for r in baseline_results if r.duration_s >= 59.0 and r.scenario_id.endswith("trial1")]
    if t60_runs:
        ref_e = t60_runs[0].ref_trajectory_east
        ref_n = t60_runs[0].ref_trajectory_north
        
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.plot(ref_e, ref_n, color="black", linewidth=2.5, label="Ground Truth Vehicle Reference")
        
        styles = {
            "Zero-Speed": ("gray", ":"),
            "Last-Known-Speed": ("#1f77b4", "--"),
            "Constant-Velocity": ("#ff7f0e", "-."),
            "IMU-Kinematic": ("#2ca02c", "-"),
        }
        
        for r in t60_runs:
            c, ls = styles.get(r.method_name, ("purple", "-"))
            ax.plot(
                r.est_trajectory_east, r.est_trajectory_north,
                color=c, linestyle=ls, linewidth=1.8,
                label=f"{r.method_name} (Drift: {r.final_drift_m:.1f}m / {r.drift_pct:.1f}%)"
            )
            
        ax.scatter([ref_e[0]], [ref_n[0]], color="green", s=100, zorder=5, label="Outage Start (t0)")
        ax.scatter([ref_e[-1]], [ref_n[-1]], color="red", marker="x", s=100, zorder=5, label="True Outage End")
        ax.set_xlabel("Local East (metres)")
        ax.set_ylabel("Local North (metres)")
        ax.set_title("60-Second GNSS Outage: Fair Baseline Trajectory Comparison (Drive S4)")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig("figures/phase7_baseline_trajectory_comparison_60s.png")
        plt.close()
        print("Saved figures/phase7_baseline_trajectory_comparison_60s.png")


if __name__ == "__main__":
    main()
