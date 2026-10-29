"""Unit tests for Timestamp Synchronization and Interpolation."""

import numpy as np
import pandas as pd
import pytest
from src.synchronization import align_sensor_and_reference


def test_linear_interpolation_between_fixes():
    """Reference coordinates must interpolate linearly between valid fixes."""
    # Synthetic recording: 100 IMU samples at 10 Hz (t = 0 to 9.9s)
    # Fix 1 at t=0s: (lat=52.40, lon=-1.50, alt=100)
    # Fix 2 at t=5s (index 50): (lat=52.4001, lon=-1.50, alt=100)
    n = 100
    t_ms = np.arange(n) * 100
    lat = np.full(n, 52.40)
    lat[50:] = 52.4001
    lon = np.full(n, -1.50)
    alt = np.full(n, 100.0)
    
    df = pd.DataFrame({
        "TIME SINCE START (ms)": t_ms,
        "ACCELEROMETER X (m/s²)": np.zeros(n),
        "ACCELEROMETER Y (m/s²)": np.zeros(n),
        "ACCELEROMETER Z (m/s²)": np.full(n, 9.81),
        "GYROSCOPE Yaw (rad/s)": np.zeros(n),
        "GYROSCOPE Pitch (rad/s)": np.zeros(n),
        "GYROSCOPE Roll (rad/s)": np.zeros(n),
        "GPS LATITUDE (degrees)": lat,
        "GPS LONGITUDE (degrees)": lon,
        "GPS ALTITUDE (m)": alt,
        "GPS SPEED (Kmh)": np.zeros(n),
        "GPS ACCURACY (m)": np.full(n, 3.0),
    })
    
    aligned, meta = align_sensor_and_reference(df, max_gps_gap_s=15.0)
    
    # At t=2.5s (index 25, midway), north position should be exactly halfway
    n_start = aligned.loc[0, "ref_north_m"]
    n_mid = aligned.loc[25, "ref_north_m"]
    n_end = aligned.loc[50, "ref_north_m"]
    
    assert np.isclose(n_start, 0.0, atol=1e-3)
    assert np.isclose(n_mid, 0.5 * n_end, atol=1e-3)
    assert aligned.loc[25, "valid_reference"] == True


def test_large_gap_rejection():
    """Gaps exceeding max_gps_gap_s must NOT be interpolated and marked invalid."""
    n = 300
    t_ms = np.arange(n) * 100  # 30 seconds
    lat = np.full(n, 52.40)
    lat[250:] = 52.4005  # Fix at t=0s and t=25s -> Gap is 25s > 15s threshold!
    lon = np.full(n, -1.50)
    alt = np.full(n, 100.0)
    
    df = pd.DataFrame({
        "TIME SINCE START (ms)": t_ms,
        "ACCELEROMETER X (m/s²)": np.zeros(n),
        "ACCELEROMETER Y (m/s²)": np.zeros(n),
        "ACCELEROMETER Z (m/s²)": np.full(n, 9.81),
        "GYROSCOPE Yaw (rad/s)": np.zeros(n),
        "GYROSCOPE Pitch (rad/s)": np.zeros(n),
        "GYROSCOPE Roll (rad/s)": np.zeros(n),
        "GPS LATITUDE (degrees)": lat,
        "GPS LONGITUDE (degrees)": lon,
        "GPS ALTITUDE (m)": alt,
        "GPS SPEED (Kmh)": np.zeros(n),
        "GPS ACCURACY (m)": np.full(n, 3.0),
    })
    
    aligned, meta = align_sensor_and_reference(df, max_gps_gap_s=15.0)
    assert meta["skipped_large_gaps_count"] == 1
    # Samples in gap must have valid_reference == False and NaN coordinates
    assert aligned.loc[100, "valid_reference"] == False
    assert np.isnan(aligned.loc[100, "ref_east_m"])
