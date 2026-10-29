"""
Timestamp Synchronization and Reference Trajectory Alignment Module.

Aligns asynchronous GNSS fixes with high-rate IMU timestamps using rigorous,
gap-aware interpolation, producing a strict validity mask to prevent corrupt labels.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.geodesy import geodetic_to_enu


def align_sensor_and_reference(
    df: pd.DataFrame,
    v_df: Optional[pd.DataFrame] = None,
    origin: Optional[Tuple[float, float, float]] = None,
    max_gps_gap_s: float = 15.0,
    max_gps_accuracy_m: float = 20.0,
    interpolation_method: str = "linear",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Synchronize raw phone recordings to uniform IMU timeline and generate reference labels.
    
    Args:
        df: Sanitized DataFrame containing raw smartphone sensor columns.
        v_df: Optional paired Vehicle dataset (V-*.csv) providing continuous 10 Hz high-rate
              ground truth GNSS coordinates and speed. (Strictly for evaluation; never model input).
        origin: Optional (lat0, lon0, alt0) for Local ENU conversion. If None, uses first valid fix.
        max_gps_gap_s: Maximum allowable gap between GPS fixes for interpolation.
        max_gps_accuracy_m: Threshold beyond which GPS fixes are considered untrustworthy.
        interpolation_method: 'linear' or 'cubic' (default: 'linear' to prevent overshoots).
        
    Returns:
        (aligned_df, sync_metadata)
    """
    total_samples = len(df)
    
    # 1. Establish common timebase
    t_ms = df["TIME SINCE START (ms)"].values.astype(float)
    t_zero = t_ms[0]
    time_s = (t_ms - t_zero) / 1000.0
    dt_imu = np.concatenate([[0.1], np.diff(time_s)])
    
    # 2. Extract IMU input channels
    accel_x = df["ACCELEROMETER X (m/s²)"].values.astype(float)
    accel_y = df["ACCELEROMETER Y (m/s²)"].values.astype(float)
    accel_z = df["ACCELEROMETER Z (m/s²)"].values.astype(float)
    gyro_yaw = df["GYROSCOPE Yaw (rad/s)"].values.astype(float)
    gyro_pitch = df["GYROSCOPE Pitch (rad/s)"].values.astype(float)
    gyro_roll = df["GYROSCOPE Roll (rad/s)"].values.astype(float)
    
    # 3. Identify genuine discrete GPS updates from phone
    raw_lat = df["GPS LATITUDE (degrees)"].values.astype(float)
    raw_lon = df["GPS LONGITUDE (degrees)"].values.astype(float)
    raw_alt = df["GPS ALTITUDE (m)"].values.astype(float)
    raw_speed_val = df["GPS SPEED (Kmh)"].values.astype(float)
    raw_acc = df["GPS ACCURACY (m)"].values.astype(float)
    
    # Discrete change detection (detect transition edges)
    gps_change_mask = np.concatenate([[True], (np.diff(raw_lat) != 0) | (np.diff(raw_lon) != 0)])
    fix_indices = np.where(gps_change_mask)[0]
    
    fix_times = time_s[fix_indices]
    fix_lats = raw_lat[fix_indices]
    fix_lons = raw_lon[fix_indices]
    fix_alts = raw_alt[fix_indices]
    fix_accs = raw_acc[fix_indices]
    
    # 4. Filter trustworthy fixes
    fix_valid_coords = (
        (fix_lats >= -90.0) & (fix_lats <= 90.0) &
        (fix_lons >= -180.0) & (fix_lons <= 180.0) &
        ~((fix_lats == 0.0) & (fix_lons == 0.0)) &
        (fix_accs <= max_gps_accuracy_m)
    )
    
    valid_fix_indices = fix_indices[fix_valid_coords]
    valid_fix_times = fix_times[fix_valid_coords]
    valid_fix_lats = fix_lats[fix_valid_coords]
    valid_fix_lons = fix_lons[fix_valid_coords]
    valid_fix_alts = fix_alts[fix_valid_coords]
    
    if len(valid_fix_indices) < 2 and v_df is None:
        raise ValueError("Insufficient valid GPS fixes found to establish reference trajectory.")
        
    # Determine Local ENU origin
    if origin is None:
        if v_df is not None:
            v_cols = {c.strip(): c for c in v_df.columns}
            v_lat0 = float(v_df[v_cols["Latitude (degrees)"]].iloc[0])
            v_lon0 = float(v_df[v_cols["Longitude (degrees)"]].iloc[0])
            v_alt0 = float(v_df[v_cols["Height (km)"]].iloc[0] * 1000.0) if "Height (km)" in v_cols else 100.0
            lat0, lon0, alt0 = v_lat0, v_lon0, v_alt0
        else:
            lat0, lon0, alt0 = float(valid_fix_lats[0]), float(valid_fix_lons[0]), float(valid_fix_alts[0])
    else:
        lat0, lon0, alt0 = origin
        
    # 5. Compute reference trajectory
    gaps_skipped = []
    
    if v_df is not None and len(v_df) == total_samples:
        # High-Rate 10 Hz Vehicle Reference Mode (Gold Standard ground truth)
        v_cols = {c.strip(): c for c in v_df.columns}
        v_lats = v_df[v_cols["Latitude (degrees)"]].values.astype(float)
        v_lons = v_df[v_cols["Longitude (degrees)"]].values.astype(float)
        v_alts = v_df[v_cols["Height (km)"]].values.astype(float) * 1000.0 if "Height (km)" in v_cols else np.zeros(total_samples)
        v_spds_kmh = v_df[v_cols["Velocity (km/hr)"]].values.astype(float) if "Velocity (km/hr)" in v_cols else np.zeros(total_samples)
        
        # Convert 10 Hz vehicle fixes directly to Local ENU
        ref_east, ref_north, ref_up = geodetic_to_enu(v_lats, v_lons, v_alts, lat0, lon0, alt0)
        ref_speed = v_spds_kmh / 3.6
        
        # Velocity vectors
        de = np.concatenate([[0.0], np.diff(ref_east)])
        dn = np.concatenate([[0.0], np.diff(ref_north)])
        dt_safe = np.where(dt_imu <= 0, 0.1, dt_imu)
        ref_vel_east = de / dt_safe
        ref_vel_north = dn / dt_safe
        
        valid_reference_mask = np.ones(total_samples, dtype=bool)
        # Invalidate any extreme vehicle GPS dropout if lat=0
        bad_v = (v_lats == 0.0) | np.isnan(v_lats)
        valid_reference_mask[bad_v] = False
        reference_source = "vehicle_high_rate_10hz"
    else:
        # Phone Interpolated Reference Mode (Linear piecewise with gap gating)
        fix_east, fix_north, fix_up = geodetic_to_enu(
            valid_fix_lats, valid_fix_lons, valid_fix_alts, lat0, lon0, alt0
        )
        
        ref_east = np.full(total_samples, np.nan, dtype=np.float64)
        ref_north = np.full(total_samples, np.nan, dtype=np.float64)
        ref_up = np.full(total_samples, np.nan, dtype=np.float64)
        ref_vel_east = np.full(total_samples, np.nan, dtype=np.float64)
        ref_vel_north = np.full(total_samples, np.nan, dtype=np.float64)
        ref_speed = np.full(total_samples, np.nan, dtype=np.float64)
        valid_reference_mask = np.zeros(total_samples, dtype=bool)
        
        for k in range(len(valid_fix_times) - 1):
            t_start = valid_fix_times[k]
            t_end = valid_fix_times[k + 1]
            gap_duration = t_end - t_start
            
            idx_in_segment = np.where((time_s >= t_start) & (time_s <= t_end))[0]
            if len(idx_in_segment) == 0:
                continue
                
            if gap_duration > max_gps_gap_s:
                gaps_skipped.append({
                    "start_time_s": float(t_start),
                    "end_time_s": float(t_end),
                    "duration_s": float(gap_duration),
                    "samples_affected": int(len(idx_in_segment)),
                })
                continue
                
            alpha = (time_s[idx_in_segment] - t_start) / gap_duration
            ref_east[idx_in_segment] = fix_east[k] + alpha * (fix_east[k + 1] - fix_east[k])
            ref_north[idx_in_segment] = fix_north[k] + alpha * (fix_north[k + 1] - fix_north[k])
            ref_up[idx_in_segment] = fix_up[k] + alpha * (fix_up[k + 1] - fix_up[k])
            
            vx = (fix_east[k + 1] - fix_east[k]) / gap_duration
            vy = (fix_north[k + 1] - fix_north[k]) / gap_duration
            ref_vel_east[idx_in_segment] = vx
            ref_vel_north[idx_in_segment] = vy
            ref_speed[idx_in_segment] = np.sqrt(vx**2 + vy**2)
            valid_reference_mask[idx_in_segment] = True
            
        reference_source = "phone_interpolated_gap_gated"
        
    # Build aligned DataFrame
    aligned = pd.DataFrame({
        # Timeline
        "time_s": time_s,
        "time_ms": t_ms,
        "dt_imu_s": dt_imu,
        
        # 6 Model Inputs (Accelerations and Angular Rates)
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "gyro_yaw": gyro_yaw,
        "gyro_pitch": gyro_pitch,
        "gyro_roll": gyro_roll,
        
        # Reference Labels (STRICTLY FOR EVALUATION / LOSS COMPUTATION)
        "ref_east_m": ref_east,
        "ref_north_m": ref_north,
        "ref_up_m": ref_up,
        "ref_vel_east_mps": ref_vel_east,
        "ref_vel_north_mps": ref_vel_north,
        "ref_speed_mps": ref_speed,
        "valid_reference": valid_reference_mask,
        
        # Raw GNSS channels (Retained for observation masking / diagnostic check)
        "raw_lat": raw_lat,
        "raw_lon": raw_lon,
        "raw_alt": raw_alt,
        "raw_speed_mps": raw_speed_val,  # Note: phone speed logged in m/s
        "raw_accuracy_m": raw_acc,
    })
    
    valid_count = int(np.sum(valid_reference_mask))
    valid_ratio = float(valid_count / total_samples)
    
    sync_metadata = {
        "origin_enu": {"lat0": lat0, "lon0": lon0, "alt0": alt0},
        "reference_source": reference_source,
        "total_samples": total_samples,
        "valid_reference_samples": valid_count,
        "valid_reference_ratio": valid_ratio,
        "total_discrete_gps_fixes": len(fix_indices),
        "valid_discrete_gps_fixes": len(valid_fix_indices),
        "max_gps_gap_s": max_gps_gap_s,
        "skipped_large_gaps_count": len(gaps_skipped),
        "skipped_large_gaps": gaps_skipped,
        "interpolation_method": interpolation_method,
    }
    
    return aligned, sync_metadata
        
    # Build aligned DataFrame
    aligned = pd.DataFrame({
        # Timeline
        "time_s": time_s,
        "time_ms": t_ms,
        "dt_imu_s": dt_imu,
        
        # 6 Model Inputs (Accelerations and Angular Rates)
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "gyro_yaw": gyro_yaw,
        "gyro_pitch": gyro_pitch,
        "gyro_roll": gyro_roll,
        
        # Reference Labels (STRICTLY FOR EVALUATION / LOSS COMPUTATION)
        "ref_east_m": ref_east,
        "ref_north_m": ref_north,
        "ref_up_m": ref_up,
        "ref_vel_east_mps": ref_vel_east,
        "ref_vel_north_mps": ref_vel_north,
        "ref_speed_mps": ref_speed,
        "valid_reference": valid_reference_mask,
        
        # Raw GNSS channels (Retained for observation masking / diagnostic check)
        "raw_lat": raw_lat,
        "raw_lon": raw_lon,
        "raw_alt": raw_alt,
        "raw_speed_mps": raw_speed_kmh / 3.6,
        "raw_accuracy_m": raw_acc,
    })
    
    valid_count = int(np.sum(valid_reference_mask))
    valid_ratio = float(valid_count / total_samples)
    
    sync_metadata = {
        "origin_enu": {"lat0": lat0, "lon0": lon0, "alt0": alt0},
        "total_samples": total_samples,
        "valid_reference_samples": valid_count,
        "valid_reference_ratio": valid_ratio,
        "total_discrete_gps_fixes": len(fix_indices),
        "valid_discrete_gps_fixes": len(valid_fix_indices),
        "max_gps_gap_s": max_gps_gap_s,
        "skipped_large_gaps_count": len(gaps_skipped),
        "skipped_large_gaps": gaps_skipped,
        "interpolation_method": interpolation_method,
    }
    
    return aligned, sync_metadata
