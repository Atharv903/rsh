import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.corrected_synchronization import extract_aligned_segments_for_drive
from src.outage_estimator import (
    PreOutageState,
    estimate_initial_heading_from_pre_outage_gps,
    run_causal_dead_reckoning,
    make_zero_speed_estimator,
    make_last_known_speed_estimator,
)

segs, _ = extract_aligned_segments_for_drive("S4", raw_dir="data/raw")
test_df = segs[0]
total_len = len(test_df)

np.random.seed(42)
durations = [10, 30, 60]
scenarios = []

for dur in durations:
    n_samples = dur * 10
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
            "duration_s": dur,
            "start_idx": int(st),
            "end_idx": int(st + n_samples),
            "pre_window_samples": 100,
        })

print(f"Constructed {len(scenarios)} matched scenarios on real Drive S4.")

rows = []
for sc in scenarios:
    st, end = sc["start_idx"], sc["end_idx"]
    pre_st = max(0, st - sc["pre_window_samples"])
    pre_df = test_df.iloc[pre_st:st].copy()
    in_outage_df = test_df.iloc[st:end].copy()
    
    heading_deg, last_spd = estimate_initial_heading_from_pre_outage_gps(pre_df)
    pos_e0 = float(pre_df["ref_east_m"].iloc[-1])
    pos_n0 = float(pre_df["ref_north_m"].iloc[-1])
    gyro_bias = float(np.median(pre_df["gyro_pitch"]))
    
    pre_state = PreOutageState(
        t0_s=float(pre_df["time_s"].iloc[-1]),
        pos_east_m=pos_e0,
        pos_north_m=pos_n0,
        speed_mps=last_spd,
        heading_deg=heading_deg,
        gyro_bias_rads=gyro_bias,
    )
    
    r_const = run_causal_dead_reckoning(pre_state, in_outage_df, make_last_known_speed_estimator(last_spd), "Constant-Velocity", sc["scenario_id"], "S4")
    rows.append({
        "scenario_id": sc["scenario_id"],
        "duration_s": sc["duration_s"],
        "final_drift_m": r_const.final_drift_m,
        "ate_rmse": r_const.ate_rmse_m,
        "drift_pct": r_const.drift_pct,
    })

df = pd.DataFrame(rows)
summary = df.groupby("duration_s")[["ate_rmse", "final_drift_m", "drift_pct"]].mean().round(2)
print("\n--- Constant-Velocity Baseline across 5 trials on Real Drive S4 ---")
print(summary)
