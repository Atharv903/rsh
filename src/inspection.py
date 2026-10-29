"""
Dataset Inspection and Data Quality Assessment Module.

Performs thorough verification of schema, sampling rates, gaps, GPS validity,
and produces structured metrics and comprehensive quality reports.
"""

import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


# Canonical column mappings and roles
COLUMN_ROLES = {
    # 6 Input IMU channels
    "ACCELEROMETER X (m/s²)": {"role": "input", "standard_name": "accel_x", "unit": "m/s²", "type": "IMU accelerometer"},
    "ACCELEROMETER Y (m/s²)": {"role": "input", "standard_name": "accel_y", "unit": "m/s²", "type": "IMU accelerometer"},
    "ACCELEROMETER Z (m/s²)": {"role": "input", "standard_name": "accel_z", "unit": "m/s²", "type": "IMU accelerometer"},
    "GYROSCOPE Yaw (rad/s)": {"role": "input", "standard_name": "gyro_yaw", "unit": "rad/s", "type": "IMU gyroscope"},
    "GYROSCOPE Pitch (rad/s)": {"role": "input", "standard_name": "gyro_pitch", "unit": "rad/s", "type": "IMU gyroscope"},
    "GYROSCOPE Roll (rad/s)": {"role": "input", "standard_name": "gyro_roll", "unit": "rad/s", "type": "IMU gyroscope"},
    
    # Timestamps
    "TIME SINCE START (ms)": {"role": "timestamp", "standard_name": "time_ms", "unit": "ms", "type": "Monotonic millisecond clock"},
    "DATE (YYYY-MO-DD HH-MI-SS_SSS)": {"role": "metadata", "standard_name": "date_str", "unit": "ISO text", "type": "Wall-clock datetime"},
    
    # GPS reference labels (Strictly evaluation only; NEVER model inputs)
    "GPS LATITUDE (degrees)": {"role": "reference_label", "standard_name": "gps_lat", "unit": "degrees", "type": "GNSS coordinate"},
    "GPS LONGITUDE (degrees)": {"role": "reference_label", "standard_name": "gps_lon", "unit": "degrees", "type": "GNSS coordinate"},
    "GPS ALTITUDE (m)": {"role": "reference_label", "standard_name": "gps_alt", "unit": "meters", "type": "GNSS altitude"},
    "GPS SPEED (Kmh)": {"role": "reference_label", "standard_name": "gps_speed_kmh", "unit": "km/h", "type": "GNSS speed"},
    "GPS ACCURACY (m)": {"role": "reference_quality", "standard_name": "gps_accuracy_m", "unit": "meters", "type": "GNSS horizontal accuracy"},
    "GPS ORIENTATION (°)": {"role": "reference_label", "standard_name": "gps_heading_deg", "unit": "degrees", "type": "GNSS course over ground"},
    "GPS SATELLITES IN RANGE": {"role": "reference_quality", "standard_name": "gps_satellites", "unit": "count", "type": "GNSS satellite count"},
    
    # Auxiliary smartphone channels (Excluded from baseline model inputs to prevent leakage / over-specification)
    "GRAVITY X (m/s²)": {"role": "excluded_auxiliary", "standard_name": "gravity_x", "unit": "m/s²", "type": "Device fused gravity vector"},
    "GRAVITY Y (m/s²)": {"role": "excluded_auxiliary", "standard_name": "gravity_y", "unit": "m/s²", "type": "Device fused gravity vector"},
    "GRAVITY Z (m/s²)": {"role": "excluded_auxiliary", "standard_name": "gravity_z", "unit": "m/s²", "type": "Device fused gravity vector"},
    "MAGNETIC FIELD X (μT)": {"role": "excluded_auxiliary", "standard_name": "mag_x", "unit": "μT", "type": "Magnetometer"},
    "MAGNETIC FIELD Y (μT)": {"role": "excluded_auxiliary", "standard_name": "mag_y", "unit": "μT", "type": "Magnetometer"},
    "MAGNETIC FIELD Z (μT)": {"role": "excluded_auxiliary", "standard_name": "mag_z", "unit": "μT", "type": "Magnetometer"},
    "ORIENTATION (Yaw) (°)": {"role": "excluded_auxiliary", "standard_name": "orient_yaw", "unit": "degrees", "type": "Device fused orientation"},
    "ORIENTATION (Pitch) (°)": {"role": "excluded_auxiliary", "standard_name": "orient_pitch", "unit": "degrees", "type": "Device fused orientation"},
    "ORIENTATION (Roll ) (°)": {"role": "excluded_auxiliary", "standard_name": "orient_roll", "unit": "degrees", "type": "Device fused orientation"},
}


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and standardize known encoding glitches in column names."""
    clean_cols = {}
    for col in df.columns:
        c = col.strip()
        c_upper = c.upper()
        # Accel
        if "ACCELEROMETER" in c_upper:
            if " X " in c_upper or c_upper.startswith("ACCELEROMETER X"):
                canonical = "ACCELEROMETER X (m/s²)"
            elif " Y " in c_upper or c_upper.startswith("ACCELEROMETER Y"):
                canonical = "ACCELEROMETER Y (m/s²)"
            elif " Z " in c_upper or c_upper.startswith("ACCELEROMETER Z"):
                canonical = "ACCELEROMETER Z (m/s²)"
            else:
                canonical = c
        # Gravity
        elif "GRAVITY" in c_upper:
            if " X " in c_upper or c_upper.startswith("GRAVITY X"):
                canonical = "GRAVITY X (m/s²)"
            elif " Y " in c_upper or c_upper.startswith("GRAVITY Y"):
                canonical = "GRAVITY Y (m/s²)"
            elif " Z " in c_upper or c_upper.startswith("GRAVITY Z"):
                canonical = "GRAVITY Z (m/s²)"
            else:
                canonical = c
        # Gyro
        elif "GYROSCOPE" in c_upper:
            if "YAW" in c_upper:
                canonical = "GYROSCOPE Yaw (rad/s)"
            elif "PITCH" in c_upper:
                canonical = "GYROSCOPE Pitch (rad/s)"
            elif "ROLL" in c_upper:
                canonical = "GYROSCOPE Roll (rad/s)"
            else:
                canonical = c
        # Mag
        elif "MAGNETIC FIELD" in c_upper:
            if " X " in c_upper or c_upper.startswith("MAGNETIC FIELD X"):
                canonical = "MAGNETIC FIELD X (μT)"
            elif " Y " in c_upper or c_upper.startswith("MAGNETIC FIELD Y"):
                canonical = "MAGNETIC FIELD Y (μT)"
            elif " Z " in c_upper or c_upper.startswith("MAGNETIC FIELD Z"):
                canonical = "MAGNETIC FIELD Z (μT)"
            else:
                canonical = c
        # GPS Orientation vs Device Orientation
        elif "GPS" in c_upper and "ORIENTATION" in c_upper:
            canonical = "GPS ORIENTATION (°)"
        elif "ORIENTATION" in c_upper:
            if "YAW" in c_upper:
                canonical = "ORIENTATION (Yaw) (°)"
            elif "PITCH" in c_upper:
                canonical = "ORIENTATION (Pitch) (°)"
            elif "ROLL" in c_upper:
                canonical = "ORIENTATION (Roll ) (°)"
            else:
                canonical = c
        elif "TIME SINCE START" in c_upper:
            canonical = "TIME SINCE START (ms)"
        elif "DATE" in c_upper:
            canonical = "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
        elif "LATITUDE" in c_upper:
            canonical = "GPS LATITUDE (degrees)"
        elif "LONGITUDE" in c_upper:
            canonical = "GPS LONGITUDE (degrees)"
        elif "ALTITUDE" in c_upper:
            canonical = "GPS ALTITUDE (m)"
        elif "SPEED" in c_upper:
            canonical = "GPS SPEED (Kmh)"
        elif "ACCURACY" in c_upper:
            canonical = "GPS ACCURACY (m)"
        elif "SATELLITES" in c_upper:
            canonical = "GPS SATELLITES IN RANGE"
        else:
            canonical = c
        clean_cols[col] = canonical
    return df.rename(columns=clean_cols)


def inspect_recording(csv_path: str) -> Dict[str, Any]:
    """Perform comprehensive data inspection on a raw recording.
    
    Args:
        csv_path: Path to the raw CSV file.
        
    Returns:
        Structured dictionary of schema, sampling rates, statistics, and quality warnings.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
        
    file_size_bytes = os.path.getsize(csv_path)
    
    # Read raw CSV using latin1 to handle special unit characters safely
    df_raw = pd.read_csv(csv_path, encoding="latin1")
    df = sanitize_column_names(df_raw)
    
    total_rows, total_cols = df.shape
    
    # Timestamp inspection
    time_col = "TIME SINCE START (ms)"
    has_time = time_col in df.columns
    if not has_time:
        raise ValueError(f"Missing mandatory timestamp column: {time_col}")
        
    t_ms = df[time_col].astype(float)
    t_sec = t_ms / 1000.0
    dt_ms = t_ms.diff()
    
    is_monotonic = bool(t_ms.is_monotonic_increasing)
    is_strictly_monotonic = bool((dt_ms.iloc[1:] > 0).all()) if len(dt_ms) > 1 else True
    duplicate_timestamps = int(t_ms.duplicated().sum())
    
    dt_stats = dt_ms.dropna().describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    mean_dt_ms = float(dt_stats.get("mean", 100.0))
    nominal_rate_hz = 1000.0 / mean_dt_ms if mean_dt_ms > 0 else 0.0
    duration_s = float(t_sec.iloc[-1] - t_sec.iloc[0]) if len(t_sec) > 1 else 0.0
    
    # Timestamp gaps check (> 2x nominal dt)
    gap_threshold_ms = 2.0 * mean_dt_ms
    gaps_mask = dt_ms > gap_threshold_ms
    num_imu_gaps = int(gaps_mask.sum())
    
    # GPS column analysis
    lat_col = "GPS LATITUDE (degrees)"
    lon_col = "GPS LONGITUDE (degrees)"
    alt_col = "GPS ALTITUDE (m)"
    speed_col = "GPS SPEED (Kmh)"
    acc_col = "GPS ACCURACY (m)"
    
    has_gps = all(c in df.columns for c in [lat_col, lon_col])
    gps_info = {}
    
    if has_gps:
        lat = df[lat_col].astype(float)
        lon = df[lon_col].astype(float)
        acc = df[acc_col].astype(float) if acc_col in df.columns else pd.Series(np.zeros(total_rows))
        speed = df[speed_col].astype(float) if speed_col in df.columns else pd.Series(np.zeros(total_rows))
        
        # Valid GPS coordinate criteria: lat in [-90, 90], lon in [-180, 180], not exactly (0,0)
        invalid_coords = ((lat < -90) | (lat > 90) | (lon < -180) | (lon > 180) | ((lat == 0.0) & (lon == 0.0)))
        num_invalid_coords = int(invalid_coords.sum())
        
        # Identify discrete updates (where lat or lon changes)
        gps_changes = (lat.diff() != 0) | (lon.diff() != 0)
        gps_changes.iloc[0] = True  # initial fix
        unique_gps_fixes = int(gps_changes.sum())
        
        gps_change_times = t_sec[gps_changes]
        gps_dt_s = gps_change_times.diff().dropna()
        gps_dt_stats = gps_dt_s.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict() if len(gps_dt_s) > 0 else {}
        
        gps_info = {
            "lat_min": float(lat.min()),
            "lat_max": float(lat.max()),
            "lon_min": float(lon.min()),
            "lon_max": float(lon.max()),
            "unique_fixes": unique_gps_fixes,
            "invalid_coords_count": num_invalid_coords,
            "accuracy_mean_m": float(acc.mean()),
            "accuracy_median_m": float(acc.median()),
            "accuracy_max_m": float(acc.max()),
            "speed_kmh_max": float(speed.max()),
            "speed_kmh_mean": float(speed.mean()),
            "gps_dt_stats": gps_dt_stats,
        }
    
    # Missing values and data types
    missing_counts = df.isnull().sum().to_dict()
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    
    # Sensor channel summaries
    sensor_stats = {}
    candidate_sensors = [
        "ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)",
        "GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)",
    ]
    for sc in candidate_sensors:
        if sc in df.columns:
            s = df[sc].astype(float)
            sensor_stats[sc] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "median": float(s.median()),
                "max": float(s.max()),
            }
            
    warnings = []
    if duplicate_timestamps > 0:
        warnings.append(f"Detected {duplicate_timestamps} duplicate timestamps.")
    if not is_strictly_monotonic:
        warnings.append("Timestamps are not strictly monotonic.")
    if num_imu_gaps > 0:
        warnings.append(f"Detected {num_imu_gaps} IMU sampling intervals greater than 2x nominal dt.")
    if has_gps and gps_info.get("accuracy_max_m", 0) > 25.0:
        warnings.append(f"GPS accuracy degraded up to {gps_info['accuracy_max_m']:.1f} meters.")
    if has_gps and gps_info.get("invalid_coords_count", 0) > 0:
        warnings.append(f"Found {gps_info['invalid_coords_count']} invalid GPS coordinates.")
        
    return {
        "file_path": csv_path,
        "file_size_bytes": file_size_bytes,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "duration_seconds": duration_s,
        "duration_minutes": duration_s / 60.0,
        "timestamps": {
            "monotonic": is_monotonic,
            "strictly_monotonic": is_strictly_monotonic,
            "duplicates": duplicate_timestamps,
            "mean_dt_ms": mean_dt_ms,
            "nominal_rate_hz": nominal_rate_hz,
            "dt_stats_ms": dt_stats,
            "gap_count": num_imu_gaps,
        },
        "gps": gps_info,
        "sensors": sensor_stats,
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "warnings": warnings,
    }


def generate_data_quality_report(report_data: Dict[str, Any], output_path: str = "reports/data_quality_report.md") -> str:
    """Generate a publication-grade markdown data quality report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    t_info = report_data["timestamps"]
    gps_info = report_data.get("gps", {})
    sensors = report_data.get("sensors", {})
    warnings = report_data.get("warnings", [])
    
    md = []
    md.append("# Data Quality & Sensor Inspection Report")
    md.append(f"**Target Dataset File:** `{report_data['file_path']}`  ")
    md.append(f"**Total Samples:** {report_data['total_rows']:,} rows | **Total Columns:** {report_data['total_columns']}  ")
    md.append(f"**Total Duration:** {report_data['duration_seconds']:.2f} s ({report_data['duration_minutes']:.2f} min)  ")
    md.append(f"**File Size:** {report_data['file_size_bytes'] / (1024*1024):.2f} MB\n")
    
    md.append("## 1. Schema & Column Categorization\n")
    md.append("| Column Name | Assigned Role | Standard Identifier | Unit | Description |")
    md.append("|---|---|---|---|---|")
    for col in report_data["columns"]:
        role_meta = COLUMN_ROLES.get(col, {"role": "unknown", "standard_name": col, "unit": "N/A", "type": "Unmapped"})
        md.append(f"| `{col}` | **{role_meta['role']}** | `{role_meta['standard_name']}` | {role_meta['unit']} | {role_meta['type']} |")
        
    md.append("\n## 2. Temporal & Sampling Analysis\n")
    md.append(f"- **Monotonic Increasing:** {'Yes (Strict)' if t_info['strictly_monotonic'] else 'No'}")
    md.append(f"- **Duplicate Timestamps:** {t_info['duplicates']}")
    md.append(f"- **Nominal IMU Sampling Rate:** {t_info['nominal_rate_hz']:.2f} Hz (Mean dt: {t_info['mean_dt_ms']:.2f} ms, std: {t_info['dt_stats_ms'].get('std', 0):.4f} ms)")
    md.append(f"- **IMU Sampling Interval Percentiles (ms):**")
    md.append(f"  - 1%: {t_info['dt_stats_ms'].get('1%', 0):.1f} ms | Median (50%): {t_info['dt_stats_ms'].get('50%', 0):.1f} ms | 99%: {t_info['dt_stats_ms'].get('99%', 0):.1f} ms | Max: {t_info['dt_stats_ms'].get('max', 0):.1f} ms")
    md.append(f"- **IMU Timestamp Gaps (> 200 ms):** {t_info['gap_count']}")
    
    md.append("\n## 3. GNSS / GPS Characteristics & Ground Truth Reference\n")
    if gps_info:
        md.append(f"- **Geographic Bounds:** Latitude [{gps_info['lat_min']:.6f}°, {gps_info['lat_max']:.6f}°], Longitude [{gps_info['lon_min']:.6f}°, {gps_info['lon_max']:.6f}°]")
        md.append(f"- **Total Discrete GPS Fix Events:** {gps_info['unique_fixes']}")
        gps_dt = gps_info.get("gps_dt_stats", {})
        md.append(f"- **Effective GPS Update Interval:** Median: {gps_dt.get('50%', 0):.2f} s | Mean: {gps_dt.get('mean', 0):.2f} s | Max: {gps_dt.get('max', 0):.2f} s")
        md.append(f"- **GPS Horizontal Accuracy:** Mean: {gps_info['accuracy_mean_m']:.2f} m | Median: {gps_info['accuracy_median_m']:.2f} m | Max: {gps_info['accuracy_max_m']:.2f} m")
        md.append(f"- **GPS Speed:** Max: {gps_info['speed_kmh_max']:.2f} km/h ({gps_info['speed_kmh_max']/3.6:.2f} m/s) | Mean: {gps_info['speed_kmh_mean']:.2f} km/h")
        md.append(f"- **Invalid Coordinate Fixes (e.g. 0,0 or out-of-range):** {gps_info['invalid_coords_count']}")
    else:
        md.append("- No GPS data present.")
        
    md.append("\n## 4. IMU Sensor Inputs (Initial 6 Channels)\n")
    md.append("| Sensor Channel | Mean | Std | Min | Median (50%) | Max | Unit |")
    md.append("|---|---|---|---|---|---|---|")
    for sc, s in sensors.items():
        unit = "m/s²" if "ACCELEROMETER" in sc else "rad/s"
        md.append(f"| `{sc}` | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['median']:.4f} | {s['max']:.4f} | {unit} |")
        
    md.append("\n## 5. Leakage Prevention Protocol (Mandatory Rules)\n")
    md.append("1. **Strict Input Channel Isolation:** Only 6 channels (`accel_x`, `accel_y`, `accel_z`, `gyro_yaw`, `gyro_pitch`, `gyro_roll`) enter the model feature space.")
    md.append("2. **Zero GNSS Leakage:** GPS latitude, longitude, speed, altitude, accuracy, and heading are strictly isolated as evaluation-only reference labels during simulated outages.")
    md.append("3. **No External CAN/OBD Signals:** External vehicle signals (wheel speeds, steering angle, OBD throttle) are completely excluded.")
    md.append("4. **Coordinate Frame Distinction:** Smartphone body frame $(x_b, y_b, z_b)$ is distinct from vehicle frame $(x_v, y_v, z_v)$. The pipeline maintains explicit transformation documentation.")

    md.append("\n## 6. Data Quality Warnings & Flags\n")
    if warnings:
        for w in warnings:
            md.append(f"- ⚠️ **WARNING:** {w}")
    else:
        md.append("- ✅ **PASSED:** No data quality anomalies or integrity violations detected.")
        
    content = "\n".join(md)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    return content
