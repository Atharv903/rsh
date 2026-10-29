"""
Unit tests for Phase 3 Corrected Synchronization and Label Generation.
Tests UTC timestamp alignment, segment boundary handling, leakage isolation,
and label physical bounds.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.corrected_synchronization import (
    parse_phone_seconds_of_day,
    extract_aligned_segments_for_drive,
    generate_causal_sliding_windows,
)


RAW_S1_EXISTS = os.path.exists("data/raw/S-S1.csv") or os.path.exists("/content/drive/MyDrive/GPS_Denied_Navigation_Research/data/raw/S-S1.csv")
RAW_S2_EXISTS = os.path.exists("data/raw/S-S2.csv") or os.path.exists("/content/drive/MyDrive/GPS_Denied_Navigation_Research/data/raw/S-S2.csv")


def test_parse_phone_seconds_of_day():
    dates = pd.Series([
        "2019-09-08 10:07:49:546",  # 10:07:49.546 BST -> 09:07:49.546 UTC
        "2019-09-08 12:00:00_000",  # 12:00:00.000 BST -> 11:00:00.000 UTC
    ])
    utc_s = parse_phone_seconds_of_day(dates)
    expected_0 = 9 * 3600 + 7 * 60 + 49 + 0.546
    expected_1 = 11 * 3600
    assert np.isclose(utc_s[0], expected_0, atol=1e-3)
    assert np.isclose(utc_s[1], expected_1, atol=1e-3)


@pytest.mark.skipif(not RAW_S1_EXISTS, reason="Raw CSV data/raw/S-S1.csv not present locally")
def test_drive_s1_segment_alignment():
    raw_dir = "data/raw" if os.path.exists("data/raw/S-S1.csv") else "/content/drive/MyDrive/GPS_Denied_Navigation_Research/data/raw"
    segs, meta = extract_aligned_segments_for_drive("S1", raw_dir=raw_dir)
    assert len(segs) == 1, "Drive S1 should have exactly 1 continuous segment"
    df = segs[0]
    
    # Assert required columns
    expected_inputs = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
    expected_labels = ["ref_speed_mps", "ref_vel_east_mps", "ref_vel_north_mps", "valid_reference"]
    for col in expected_inputs + expected_labels:
        assert col in df.columns
        
    # Assert physical speed bounds
    speeds = df["ref_speed_mps"].values
    assert np.all(speeds >= 0.0), "Speed cannot be negative"
    assert np.max(speeds) < 40.0, "Speed in S1 should be within normal vehicle range (< 40 m/s)"


@pytest.mark.skipif(not RAW_S2_EXISTS, reason="Raw CSV data/raw/S-S2.csv not present locally")
def test_clock_reset_segment_isolation():
    # Drive S2 has a clock reset at row 1863
    raw_dir = "data/raw" if os.path.exists("data/raw/S-S2.csv") else "/content/drive/MyDrive/GPS_Denied_Navigation_Research/data/raw"
    segs, meta = extract_aligned_segments_for_drive("S2", raw_dir=raw_dir)
    assert len(segs) == 2, "Drive S2 should be split into 2 independent segments at clock reset"
    
    # Assert that time within each segment is strictly monotonic
    for seg in segs:
        t = seg["time_s"].values
        dt = np.diff(t)
        assert np.all(dt > 0), "Segment timebase must be strictly monotonic"
        assert np.allclose(dt, 0.1, atol=1e-4), "Segment sampling rate must be uniform 10 Hz"


@pytest.mark.skipif(not RAW_S1_EXISTS, reason="Raw CSV data/raw/S-S1.csv not present locally")
def test_sliding_window_causality_and_bounds():
    raw_dir = "data/raw" if os.path.exists("data/raw/S-S1.csv") else "/content/drive/MyDrive/GPS_Denied_Navigation_Research/data/raw"
    segs, _ = extract_aligned_segments_for_drive("S1", raw_dir=raw_dir)
    df = segs[0].iloc[:500]  # Take 500 samples
    
    X, y_spd, y_vel, meta = generate_causal_sliding_windows(df, window_size=100, stride=10)
    
    # 500 samples with W=100, stride=10 -> (500 - 100) / 10 + 1 = 41 windows
    assert X.shape == (41, 100, 6), f"Expected shape (41, 100, 6), got {X.shape}"
    assert y_spd.shape == (41, 1), f"Expected y_spd shape (41, 1), got {y_spd.shape}"
    assert y_vel.shape == (41, 2), f"Expected y_vel shape (41, 2), got {y_vel.shape}"
    assert len(meta) == 41
    
    # Verify causality: each window's end time matches its metadata
    for i, m in enumerate(meta):
        assert m["end_idx"] - m["start_idx"] == 100
        # Endpoint speed must match speed at window final timestamp
        inst_speed = df["ref_speed_mps"].iloc[m["end_idx"] - 1]
        win_speed = df["ref_speed_mps"].iloc[m["start_idx"]:m["end_idx"]].mean()
        assert np.isclose(y_spd[i, 0], inst_speed, atol=1e-4)
        assert np.isclose(m["avg_speed_mps"], win_speed, atol=1e-4)


def test_corrected_dataset_npz_files_exist_and_valid():
    for split in ["train", "val", "test"]:
        p_grp = f"data/processed/{split}_windows_grouped.npz"
        p_cor = f"data/processed/{split}_windows_corrected.npz"
        p = p_grp if os.path.exists(p_grp) else p_cor
        if not os.path.exists(p):
            pytest.skip(f"Processed tensor {p} not found locally")
            
        data = np.load(p, allow_pickle=True)
        assert "X" in data
        assert "y" in data
        assert "y_vel_enu" in data
        assert "meta" in data
        
        X = data["X"]
        y = data["y"]
        assert X.ndim == 3 and X.shape[1] == 100 and X.shape[2] == 6
        assert y.ndim in (1, 2)
        assert len(X) == len(y)
        # Ensure finite values
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
