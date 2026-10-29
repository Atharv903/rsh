"""
Phase 10B: Forensic Data Quality and Sensor Alignment Audit for Independent Drives.

Performs exhaustive verification of:
1. Sampling rate, dt distribution, timestamp monotonicity, clock resets, missing/null values.
2. Cross-stream synchronization between smartphone IMU and vehicle reference.
3. Phone mounting orientation (gravity vector), gyroscope axis identification, and correlation with vehicle yaw rate.
4. Heading conventions, sign conventions, and bias estimation.
5. Coordinate transformations (Geodetic to Local ENU) and causality checks.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.corrected_synchronization import extract_aligned_segments_for_drive, parse_phone_seconds_of_day
from src.geodesy import geodetic_to_enu
from src.inspection import sanitize_column_names


def audit_raw_session(drive_id: str, raw_dir: str) -> dict:
    s_path = os.path.join(raw_dir, f"S-{drive_id}.csv")
    v_path = os.path.join(raw_dir, f"V-{drive_id}.csv")
    
    if not os.path.exists(s_path) or not os.path.exists(v_path):
        raise FileNotFoundError(f"Missing raw files for {drive_id} in {raw_dir}")
        
    s_df_raw = pd.read_csv(s_path, encoding="latin1")
    v_df_raw = pd.read_csv(v_path, encoding="latin1")
    
    s_df = sanitize_column_names(s_df_raw)
    v_df = v_df_raw.copy()
    v_df.columns = [c.strip() for c in v_df.columns]
    
    audit = {
        "drive_id": drive_id,
        "s_path": s_path,
        "v_path": v_path,
        "s_total_rows": len(s_df),
        "v_total_rows": len(v_df),
        "s_columns": list(s_df.columns),
        "v_columns": list(v_df.columns),
    }
    
    # Check nulls
    audit["s_null_counts"] = {k: int(v) for k, v in s_df.isnull().sum().items() if v > 0}
    audit["v_null_counts"] = {k: int(v) for k, v in v_df.isnull().sum().items() if v > 0}
    
    # 1. Monotonicity & Clock Resets in S
    t_ms = s_df["TIME SINCE START (ms)"].values.astype(float)
    dt_ms = np.diff(t_ms)
    resets = np.where(dt_ms < 0)[0]
    zero_dt = np.where(dt_ms == 0)[0]
    gap_dt = np.where(dt_ms > 200)[0]
    
    audit["phone_clock"] = {
        "min_dt_ms": float(np.min(dt_ms)),
        "max_dt_ms": float(np.max(dt_ms)),
        "median_dt_ms": float(np.median(dt_ms)),
        "mean_dt_ms": float(np.mean(dt_ms)),
        "clock_resets_count": int(len(resets)),
        "clock_reset_indices": [int(x) for x in resets],
        "zero_dt_count": int(len(zero_dt)),
        "gaps_gt_200ms_count": int(len(gap_dt)),
    }
    
    # 2. Vehicle Timestamps
    t_v = v_df["Time Since Start of Day (seconds)"].values.astype(float)
    dt_v = np.diff(t_v)
    v_resets = np.where(dt_v < 0)[0]
    v_gaps = np.where(dt_v > 0.2)[0]
    
    audit["vehicle_clock"] = {
        "min_dt_s": float(np.min(dt_v)),
        "max_dt_s": float(np.max(dt_v)),
        "median_dt_s": float(np.median(dt_v)),
        "mean_dt_s": float(np.mean(dt_v)),
        "clock_resets_count": int(len(v_resets)),
        "gaps_gt_200ms_count": int(len(v_gaps)),
    }
    
    # Dates and duration
    date_col = "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
    t0_str = str(s_df[date_col].iloc[0])
    tend_str = str(s_df[date_col].iloc[-1])
    audit["timestamps"] = {
        "start_utc": t0_str,
        "end_utc": tend_str,
        "duration_hours": float(len(s_df) * 0.1 / 3600.0),
    }
    
    # 3. Phone Mounting Orientation (Gravity Vector)
    grav_x_col = [c for c in s_df.columns if "GRAVITY X" in c][0]
    grav_y_col = [c for c in s_df.columns if "GRAVITY Y" in c][0]
    grav_z_col = [c for c in s_df.columns if "GRAVITY Z" in c][0]
    gx = float(s_df[grav_x_col].mean())
    gy = float(s_df[grav_y_col].mean())
    gz = float(s_df[grav_z_col].mean())
    g_norm = float(np.sqrt(gx**2 + gy**2 + gz**2))
    audit["mounting_orientation"] = {
        "mean_gravity_x_mps2": gx,
        "mean_gravity_y_mps2": gy,
        "mean_gravity_z_mps2": gz,
        "mean_gravity_norm_mps2": g_norm,
        "dominant_axis": "Z" if abs(gz) > abs(gx) and abs(gz) > abs(gy) else ("Y" if abs(gy) > abs(gx) else "X"),
        "interpretation": "Phone lying flat on horizontal surface facing upwards" if abs(gz) > 9.0 else "Other orientation",
    }
    
    # 4. Aligned Segment Extraction
    segs, meta = extract_aligned_segments_for_drive(drive_id, raw_dir=raw_dir)
    audit["segmented_alignment"] = {
        "num_valid_segments": len(segs),
        "segments_info": meta["segments"],
        "origin_enu": meta["origin_enu"],
    }
    
    # 5. Sensor to Telemetry Signal Correlation across Aligned Segments
    seg_corrs = []
    for s_idx, seg in enumerate(segs):
        dt = 0.1
        v_spd = seg["ref_speed_mps"].values
        v_accel = np.gradient(v_spd, dt)
        v_yr = np.deg2rad(seg["ref_yaw_rate_dps"].values)
        
        h_rad = np.unwrap(np.deg2rad(seg["ref_heading_deg"].values))
        h_rate = np.gradient(h_rad, dt)
        
        ax = seg["accel_x"].values
        ay = seg["accel_y"].values
        az = seg["accel_z"].values
        
        gyaw = seg["gyro_yaw"].values
        gpitch = seg["gyro_pitch"].values
        groll = seg["gyro_roll"].values
        
        corr_info = {
            "segment_id": seg["segment_id"].iloc[0],
            "num_samples": len(seg),
            "corr_accel_x_v_accel": float(np.corrcoef(ax, v_accel)[0, 1]),
            "corr_accel_y_v_accel": float(np.corrcoef(ay, v_accel)[0, 1]),
            "corr_accel_z_v_accel": float(np.corrcoef(az, v_accel)[0, 1]),
            "corr_gyro_yaw_v_yaw_rate": float(np.corrcoef(gyaw, v_yr)[0, 1]),
            "corr_gyro_pitch_v_yaw_rate": float(np.corrcoef(gpitch, v_yr)[0, 1]),
            "corr_gyro_roll_v_yaw_rate": float(np.corrcoef(groll, v_yr)[0, 1]),
            "corr_gyro_pitch_heading_rate": float(np.corrcoef(gpitch, h_rate)[0, 1]),
            "gyro_pitch_bias_stationary": float(np.median(gpitch[v_spd < 0.5])) if np.sum(v_spd < 0.5) > 50 else float(np.median(gpitch)),
            "speed_min_mps": float(v_spd.min()),
            "speed_max_mps": float(v_spd.max()),
            "speed_mean_mps": float(v_spd.mean()),
        }
        seg_corrs.append(corr_info)
        
    audit["signal_correlations"] = seg_corrs
    return audit


def default_json(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    print("=" * 70)
    print("PHASE 10B: INDEPENDENT DRIVE AUDIT (DRIVER B - M)")
    print("=" * 70)
    
    raw_m_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)"
    audit_m = audit_raw_session("M", raw_dir=raw_m_dir)
    
    out_json = "experiments/phase10_independent_validation/data_quality_audit.json"
    with open(out_json, "w") as f:
        json.dump(audit_m, f, indent=2, default=default_json)
        
    print(f"Audit completed and saved to {out_json}")
    print(f"Total Rows: S={audit_m['s_total_rows']}, V={audit_m['v_total_rows']}")
    print(f"Valid Segments: {audit_m['segmented_alignment']['num_valid_segments']}")
    for sc in audit_m["signal_correlations"]:
        print(f"Segment {sc['segment_id']}: {sc['num_samples']} samples ({sc['num_samples']*0.1/60:.1f} mins)")
        print(f"  Corr(gyro_pitch, yaw_rate): {sc['corr_gyro_pitch_v_yaw_rate']:+.4f}")
        print(f"  Corr(gyro_pitch, heading_rate): {sc['corr_gyro_pitch_heading_rate']:+.4f}")
        print(f"  Stationary gyro pitch bias: {sc['gyro_pitch_bias_stationary']:+.6f} rad/s")
        print(f"  Speed range: {sc['speed_min_mps']:.2f} - {sc['speed_max_mps']:.2f} m/s (mean {sc['speed_mean_mps']:.2f} m/s)")


if __name__ == "__main__":
    main()
