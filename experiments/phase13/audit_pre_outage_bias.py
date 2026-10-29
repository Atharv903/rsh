import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.corrected_synchronization import extract_aligned_segments_for_drive

manifest_path = "experiments/phase12_validation/phase12_scenarios.json"
with open(manifest_path) as f:
    scenarios = json.load(f)

raw_b_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)"
segs_b, _ = extract_aligned_segments_for_drive("M", raw_dir=raw_b_dir)
df_b = segs_b[1]

raw_d_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/Y (Driver D)/Y1"
segs_d, _ = extract_aligned_segments_for_drive("Y1", raw_dir=raw_d_dir)
df_d = segs_d[2]

drive_dfs = {"M": df_b, "Y1": df_d}

records = []
for sc in scenarios:
    drive_key = sc["drive_id"]
    df = drive_dfs[drive_key]
    st = sc["start_idx"]
    pre_st = max(0, st - sc["pre_window_samples"])
    pre_df = df.iloc[pre_st:st]
    
    speed_mean = float(pre_df["ref_speed_mps"].mean())
    speed_min = float(pre_df["ref_speed_mps"].min())
    speed_max = float(pre_df["ref_speed_mps"].max())
    gyro_median = float(np.median(pre_df["gyro_pitch"]))
    gyro_mean = float(pre_df["gyro_pitch"].mean())
    gyro_std = float(pre_df["gyro_pitch"].std())
    ref_yaw_mean = float(pre_df["ref_yaw_rate_rads"].mean()) if "ref_yaw_rate_rads" in pre_df.columns else float(np.nan)
    
    records.append({
        "scenario_id": sc["scenario_id"],
        "driver_id": sc["driver_id"],
        "duration_s": sc["duration_s"],
        "speed_mean_mps": speed_mean,
        "speed_min_mps": speed_min,
        "speed_max_mps": speed_max,
        "is_pre_stationary": speed_max < 0.5,
        "gyro_median_rads": gyro_median,
        "gyro_median_deg_s": np.rad2deg(gyro_median),
        "ref_yaw_mean_rads": ref_yaw_mean,
    })

res = pd.DataFrame(records)
print(res.to_string())
res.to_csv("experiments/phase13/pre_outage_bias_audit.csv", index=False)
