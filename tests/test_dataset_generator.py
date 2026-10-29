"""Unit tests for Sliding Window Generation and Drive-Level Splitting."""

import numpy as np
import pandas as pd
import pytest
from src.dataset_generator import generate_sliding_windows, create_split_manifests


def test_sliding_window_shapes():
    """Window generation must produce correct tensor dimensions (N, window_size, 6) and (N, 2)."""
    n = 250
    df = pd.DataFrame({
        "time_s": np.arange(n) * 0.1,
        "accel_x": np.random.randn(n),
        "accel_y": np.random.randn(n),
        "accel_z": np.random.randn(n),
        "gyro_yaw": np.random.randn(n),
        "gyro_pitch": np.random.randn(n),
        "gyro_roll": np.random.randn(n),
        "ref_east_m": np.arange(n) * 0.5,
        "ref_north_m": np.arange(n) * 0.2,
        "valid_reference": np.ones(n, dtype=bool),
    })
    
    X, Y, meta = generate_sliding_windows(
        df,
        drive_id="test_drive",
        window_size=100,
        stride=10,
        min_valid_ratio=1.0,
    )
    
    # Total windows: (250 - 100) // 10 + 1 = 16
    assert X.shape == (16, 100, 6)
    assert Y.shape == (16, 2)
    assert len(meta) == 16
    assert meta[0]["drive_id"] == "test_drive"


def test_split_manifest_smoke_test(tmp_path):
    """Single-drive manifest must be clearly marked as pipeline smoke test."""
    manifest = create_split_manifests(
        all_drives=["S-S1.csv"],
        drive_stats={"S-S1.csv": {"total_samples": 5000, "total_windows": 400}},
        output_dir=str(tmp_path),
        primary_drive_only=True,
    )
    assert manifest["is_generalization_benchmark"] == False
    assert manifest["status"] == "PIPELINE_SMOKE_TEST_ONLY"
