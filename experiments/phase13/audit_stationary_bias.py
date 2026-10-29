import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.corrected_synchronization import extract_aligned_segments_for_drive

def analyze_stationary_intervals(df, drive_name, min_stationary_sec=5.0):
    # Identify stationary periods: ref_speed_mps < 0.1
    is_stat = df["ref_speed_mps"] < 0.1
    # Find contiguous blocks
    stat_blocks = []
    curr_start = None
    
    for i, val in enumerate(is_stat):
        if val and curr_start is None:
            curr_start = i
        elif not val and curr_start is not None:
            if (i - curr_start) >= int(min_stationary_sec * 10):
                stat_blocks.append((curr_start, i))
            curr_start = None
    if curr_start is not None and (len(df) - curr_start) >= int(min_stationary_sec * 10):
        stat_blocks.append((curr_start, len(df)))
        
    print(f"\n=======================================================")
    print(f"STATIONARY PERIODS AUDIT FOR {drive_name}")
    print(f"Total session duration: {len(df)*0.1:.1f} s ({len(df)*0.1/60:.1f} mins)")
    print(f"Found {len(stat_blocks)} stationary blocks (>= {min_stationary_sec}s).")
    print(f"=======================================================")
    
    block_records = []
    all_stat_gyro = []
    
    for idx, (st, end) in enumerate(stat_blocks):
        block_df = df.iloc[st:end]
        dur_s = (end - st) * 0.1
        t_mid = block_df["time_s"].mean()
        
        g_pitch = block_df["gyro_pitch"]
        g_yaw = block_df["gyro_yaw"]
        g_roll = block_df["gyro_roll"]
        
        acc_x = block_df["accel_x"]
        acc_y = block_df["accel_y"]
        acc_z = block_df["accel_z"]
        
        all_stat_gyro.extend(g_pitch.values)
        
        block_records.append({
            "block_idx": idx,
            "start_idx": st,
            "end_idx": end,
            "duration_s": dur_s,
            "time_mid_s": t_mid,
            "gyro_pitch_mean_rads": float(g_pitch.mean()),
            "gyro_pitch_std_rads": float(g_pitch.std()),
            "gyro_pitch_mean_deg_s": float(np.rad2deg(g_pitch.mean())),
            "accel_z_mean": float(acc_z.mean()),
            "accel_z_std": float(acc_z.std()),
        })
        print(f"Block {idx:2d} (t={t_mid:6.1f}s, dur={dur_s:4.1f}s): "
              f"gyro_pitch = {g_pitch.mean():+.6f} rad/s ({np.rad2deg(g_pitch.mean()):+.3f} deg/s), "
              f"std = {g_pitch.std():.6f} rad/s, "
              f"acc_z = {acc_z.mean():.3f} m/s²")
              
    all_stat_gyro = np.array(all_stat_gyro)
    if len(all_stat_gyro) > 0:
        overall_bias = float(np.mean(all_stat_gyro))
        overall_std = float(np.std(all_stat_gyro))
        print(f"\n--> OVERALL STATIONARY BIAS FOR {drive_name}:")
        print(f"    Mean bias b_0 = {overall_bias:+.6f} rad/s ({np.rad2deg(overall_bias):+.3f} deg/s)")
        print(f"    White noise sigma = {overall_std:.6f} rad/s ({np.rad2deg(overall_std):.3f} deg/s)")
    else:
        overall_bias, overall_std = 0.0, 0.0
        print("    NO STATIONARY INTERVALS FOUND!")
        
    return pd.DataFrame(block_records), overall_bias, overall_std

def main():
    raw_b_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)"
    segs_b, _ = extract_aligned_segments_for_drive("M", raw_dir=raw_b_dir)
    df_b = segs_b[1]

    raw_d_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/Y (Driver D)/Y1"
    segs_d, _ = extract_aligned_segments_for_drive("Y1", raw_dir=raw_d_dir)
    df_d = segs_d[2]
    
    b_df, b_bias, b_std = analyze_stationary_intervals(df_b, "Driver B (M_seg1)")
    d_df, d_bias, d_std = analyze_stationary_intervals(df_d, "Driver D (Y1_seg3)")
    
    b_df.to_csv("experiments/phase13/stationary_blocks_driver_b.csv", index=False)
    d_df.to_csv("experiments/phase13/stationary_blocks_driver_d.csv", index=False)

if __name__ == "__main__":
    main()
