"""
Corrected S/V Synchronization and Window Generation Module.

Synchronizes smartphone IMU and vehicle telemetry using verified UTC timestamps,
handles clock resets via segment-aware partitioning, excludes misaligned fragments,
and builds leakage-free sliding window datasets for speed and velocity modeling.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from src.geodesy import geodetic_to_enu
from src.inspection import sanitize_column_names


def parse_phone_seconds_of_day(date_series: pd.Series) -> np.ndarray:
    """Parse 'YYYY-MO-DD HH-MI-SS_SSS' into seconds since start of day (UTC).
    Smartphone timestamps are logged in British Summer Time (BST = UTC+1 = +3600s).
    """
    times = date_series.str.strip().str.split(' ').str[1].str.replace(':', '-').str.replace('_', '-').str.split('-')
    h = times.str[0].astype(float).values
    m = times.str[1].astype(float).values
    s = times.str[2].astype(float).values
    ms = times.str[3].astype(float).values
    return (h * 3600.0 + m * 60.0 + s + ms / 1000.0) - 3600.0


def extract_aligned_segments_for_drive(
    drive_id: str,
    raw_dir: str = "data/raw",
    origin: Optional[Tuple[float, float, float]] = None,
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """Extract continuous, sample-accurate synchronized segments for a drive pair.
    
    Returns:
        list of aligned segment DataFrames, sync_metadata
    """
    s_path = os.path.join(raw_dir, f"S-{drive_id}.csv")
    v_path = os.path.join(raw_dir, f"V-{drive_id}.csv")
    
    if not os.path.exists(s_path) or not os.path.exists(v_path):
        raise FileNotFoundError(f"Missing raw files for drive {drive_id}")
        
    s_df = sanitize_column_names(pd.read_csv(s_path, encoding="latin1"))
    v_df = pd.read_csv(v_path, encoding="latin1")
    
    s_cols = {c.strip(): c for c in s_df.columns}
    v_cols = {c.strip(): c for c in v_df.columns}
    
    n_rows = len(s_df)
    
    # 1. Parse UTC timelines
    phone_utc = parse_phone_seconds_of_day(s_df[s_cols["DATE (YYYY-MO-DD HH-MI-SS_SSS)"]])
    veh_utc = v_df[v_cols["Time Since Start of Day (seconds)"]].values.astype(float)
    
    # 2. Identify phone clock resets
    t_ms = s_df[s_cols["TIME SINCE START (ms)"]].values.astype(float)
    resets = np.where(np.diff(t_ms) < 0)[0]
    
    # Define segment row boundaries in phone stream
    seg_bounds = [0] + list(resets + 1) + [n_rows]
    
    # Local ENU Origin
    if origin is None:
        lat0 = float(v_df[v_cols["Latitude (degrees)"]].iloc[0])
        lon0 = float(v_df[v_cols["Longitude (degrees)"]].iloc[0])
        alt0 = float(v_df[v_cols["Height (km)"]].iloc[0])  # Metres above sea level
    else:
        lat0, lon0, alt0 = origin
        
    aligned_segments = []
    seg_metadata = []
    
    for seg_idx in range(len(seg_bounds) - 1):
        st = seg_bounds[seg_idx]
        end = seg_bounds[seg_idx + 1]
        
        # Calculate median offset for this segment
        seg_phone_utc = phone_utc[st:end]
        seg_veh_utc = veh_utc[st:end]
        offset_s = float(np.median(seg_phone_utc - seg_veh_utc))
        shift_smp = int(np.round(offset_s / 0.1))
        
        # Check vehicle bounds with shift: phone row k matches vehicle row k + shift_smp
        v_st = st + shift_smp
        v_end = end + shift_smp
        
        # Clamp to valid vehicle indices
        v_len = len(v_df)
        valid_p_st = st
        valid_p_end = end
        
        if v_st < 0:
            valid_p_st = st + (-v_st)
            v_st = 0
        if v_end > v_len:
            valid_p_end = end - (v_end - v_len)
            v_end = v_len
            
        seg_len = valid_p_end - valid_p_st
        # If segment is too short for a single 10s window (100 samples), exclude
        if seg_len < 120:
            seg_metadata.append({
                "segment_idx": seg_idx,
                "drive_id": drive_id,
                "status": "EXCLUDED_TOO_SHORT",
                "phone_rows": [st, end],
                "reason": f"Length {seg_len} < 120 samples",
            })
            continue
            
        # Extract phone IMU slice
        phone_slice = s_df.iloc[valid_p_st:valid_p_end]
        veh_slice = v_df.iloc[v_st:v_end]
        
        # Construct continuous timeline for this segment
        seg_time_s = np.arange(seg_len) * 0.1
        
        # Input IMU channels (Strictly smartphone)
        accel_x = phone_slice["ACCELEROMETER X (m/s²)"].values.astype(np.float32)
        accel_y = phone_slice["ACCELEROMETER Y (m/s²)"].values.astype(np.float32)
        accel_z = phone_slice["ACCELEROMETER Z (m/s²)"].values.astype(np.float32)
        gyro_yaw = phone_slice["GYROSCOPE Yaw (rad/s)"].values.astype(np.float32)
        gyro_pitch = phone_slice["GYROSCOPE Pitch (rad/s)"].values.astype(np.float32)
        gyro_roll = phone_slice["GYROSCOPE Roll (rad/s)"].values.astype(np.float32)
        
        # Vehicle Ground Truth Reference (Strictly supervision)
        v_spd_mps = veh_slice[v_cols["Velocity (km/hr)"]].values.astype(np.float32) / 3.6
        v_heading_deg = veh_slice[v_cols["Heading (degrees)"]].values.astype(np.float32)
        v_yaw_rate_dps = veh_slice[v_cols["Yaw Rate (deg/sec)"]].values.astype(np.float32)
        v_lats = veh_slice[v_cols["Latitude (degrees)"]].values.astype(np.float64)
        v_lons = veh_slice[v_cols["Longitude (degrees)"]].values.astype(np.float64)
        v_alts = veh_slice[v_cols["Height (km)"]].values.astype(np.float64)
        
        # Convert vehicle positions to Local ENU
        v_east, v_north, v_up = geodetic_to_enu(v_lats, v_lons, v_alts, lat0, lon0, alt0)
        
        # Earth frame velocity components
        heading_rad = np.deg2rad(v_heading_deg)
        v_vel_east = v_spd_mps * np.sin(heading_rad)
        v_vel_north = v_spd_mps * np.cos(heading_rad)
        
        # Validity mask
        valid_mask = (v_lats >= -90.0) & (v_lats <= 90.0) & ~((v_lats == 0.0) & (v_lons == 0.0))
        
        seg_df = pd.DataFrame({
            "segment_id": f"{drive_id}_seg{seg_idx}",
            "drive_id": drive_id,
            "time_s": seg_time_s,
            "phone_utc_s": phone_utc[valid_p_st:valid_p_end],
            "veh_utc_s": veh_utc[v_st:v_end],
            # 6 Model Input Channels
            "accel_x": accel_x,
            "accel_y": accel_y,
            "accel_z": accel_z,
            "gyro_yaw": gyro_yaw,
            "gyro_pitch": gyro_pitch,
            "gyro_roll": gyro_roll,
            # Supervision Labels
            "ref_speed_mps": v_spd_mps,
            "ref_heading_deg": v_heading_deg,
            "ref_yaw_rate_dps": v_yaw_rate_dps,
            "ref_east_m": v_east,
            "ref_north_m": v_north,
            "ref_up_m": v_up,
            "ref_vel_east_mps": v_vel_east,
            "ref_vel_north_mps": v_vel_north,
            "valid_reference": valid_mask,
        })
        
        aligned_segments.append(seg_df)
        seg_metadata.append({
            "segment_id": f"{drive_id}_seg{seg_idx}",
            "drive_id": drive_id,
            "status": "VALID_SYNCHRONIZED",
            "phone_rows": [valid_p_st, valid_p_end],
            "veh_rows": [v_st, v_end],
            "num_samples": seg_len,
            "offset_s": offset_s,
            "shift_samples": shift_smp,
            "mean_speed_mps": float(np.mean(v_spd_mps)),
            "max_speed_mps": float(np.max(v_spd_mps)),
        })
        
    return aligned_segments, {
        "drive_id": drive_id,
        "origin_enu": {"lat0": lat0, "lon0": lon0, "alt0": alt0},
        "segments": seg_metadata,
    }


def generate_causal_sliding_windows(
    segment_df: pd.DataFrame,
    window_size: int = 100,  # 10.0 seconds at 10 Hz
    stride: int = 10,        # 1.0 second stride
    feature_channels: List[str] = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Generates strictly causal windows and labels within a single continuous segment.
    
    Returns:
        X: Shape (N, window_size, 6) feature tensor.
        y_speed: Shape (N, 1) window-average forward speed in m/s.
        y_vel_enu: Shape (N, 2) window-average [v_east, v_north] in m/s.
        metadata: List of window provenance metadata dictionaries.
    """
    total_len = len(segment_df)
    features = segment_df[feature_channels].values.astype(np.float32)
    time_s = segment_df["time_s"].values.astype(np.float64)
    speed = segment_df["ref_speed_mps"].values.astype(np.float32)
    vel_e = segment_df["ref_vel_east_mps"].values.astype(np.float32)
    vel_n = segment_df["ref_vel_north_mps"].values.astype(np.float32)
    valid = segment_df["valid_reference"].values
    
    seg_id = segment_df["segment_id"].iloc[0]
    drive_id = segment_df["drive_id"].iloc[0]
    
    x_list, y_speed_list, y_vel_list, meta_list = [], [], [], []
    
    for start_idx in range(0, total_len - window_size + 1, stride):
        end_idx = start_idx + window_size
        
        # Check validity across window
        if not np.all(valid[start_idx:end_idx]):
            continue
            
        x_window = features[start_idx:end_idx]
        
        # Window-average speed (scalar distance / window duration)
        avg_speed = float(np.mean(speed[start_idx:end_idx]))
        
        # Window-average ENU velocity components
        avg_ve = float(np.mean(vel_e[start_idx:end_idx]))
        avg_vn = float(np.mean(vel_n[start_idx:end_idx]))
        
        x_list.append(x_window)
        y_speed_list.append(np.array([avg_speed], dtype=np.float32))
        y_vel_list.append(np.array([avg_ve, avg_vn], dtype=np.float32))
        meta_list.append({
            "segment_id": seg_id,
            "drive_id": drive_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time_s": float(time_s[start_idx]),
            "end_time_s": float(time_s[end_idx - 1]),
            "avg_speed_mps": avg_speed,
            "instant_speed_mps": float(speed[end_idx - 1]),
        })
        
    if not x_list:
        return (
            np.empty((0, window_size, len(feature_channels)), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            [],
        )
        
    return np.stack(x_list, axis=0), np.stack(y_speed_list, axis=0), np.stack(y_vel_list, axis=0), meta_list
