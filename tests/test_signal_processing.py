"""Unit tests for Signal Conditioning and Normalization Leakage Prevention."""

import os
import numpy as np
import pandas as pd
import pytest
from src.signal_processing import (
    SensorNormalizer,
    clip_outliers,
    handle_missing_values,
)


def test_normalizer_train_only_fit():
    """Normalizer parameters must remain unchanged when transforming test data."""
    train_df = pd.DataFrame({
        "accel_x": [0.0, 2.0, 4.0],  # mean = 2.0, std = 1.63299
        "accel_y": [1.0, 3.0, 5.0],
    })
    test_df = pd.DataFrame({
        "accel_x": [100.0, 200.0],  # Out-of-distribution values
        "accel_y": [10.0, 20.0],
    })
    
    norm = SensorNormalizer(method="standard", channels=["accel_x", "accel_y"])
    norm.fit(train_df)
    
    mean_before = norm.params["accel_x"]["mean"]
    std_before = norm.params["accel_x"]["std"]
    
    # Transform test
    norm.transform(test_df)
    
    # Assert parameters did not mutate
    assert norm.params["accel_x"]["mean"] == mean_before
    assert norm.params["accel_x"]["std"] == std_before


def test_normalizer_json_serialization(tmp_path):
    """Normalizer save and load must produce identical transformations."""
    df = pd.DataFrame({
        "accel_x": np.random.randn(50),
        "accel_y": np.random.randn(50),
    })
    norm = SensorNormalizer(method="standard", channels=["accel_x", "accel_y"]).fit(df)
    
    json_path = os.path.join(tmp_path, "normalizer.json")
    norm.save(json_path)
    
    norm_loaded = SensorNormalizer.load(json_path)
    
    out1 = norm.transform(df)
    out2 = norm_loaded.transform(df)
    pd.testing.assert_frame_equal(out1, out2)


def test_outlier_clipping():
    """Extreme spikes beyond limits must be clipped."""
    df = pd.DataFrame({
        "accel_x": [-50.0, 0.0, 60.0],
        "gyro_yaw": [-20.0, 0.0, 25.0],
    })
    clipped, counts = clip_outliers(
        df,
        accel_channels=["accel_x"],
        gyro_channels=["gyro_yaw"],
        accel_limit_mps2=30.0,
        gyro_limit_rads=10.0,
    )
    assert clipped["accel_x"].max() == 30.0
    assert clipped["accel_x"].min() == -30.0
    assert clipped["gyro_yaw"].max() == 10.0
    assert clipped["gyro_yaw"].min() == -10.0
    assert counts["accel_x"] == 2
    assert counts["gyro_yaw"] == 2
