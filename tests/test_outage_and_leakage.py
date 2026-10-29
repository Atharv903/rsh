"""Unit tests for Simulated GPS Outages and Zero Leakage Verification."""

import numpy as np
import pandas as pd
import pytest
from src.outage_simulation import (
    find_valid_outage_intervals,
    apply_outage_mask,
    verify_zero_leakage,
)


def test_outage_duration_and_masking():
    """Outage interval must strictly match requested duration and mask GPS inputs."""
    # Synthetic continuous trajectory: 1000 samples at 10 Hz (100 seconds)
    n = 1000
    df = pd.DataFrame({
        "time_s": np.arange(n) * 0.1,
        "time_ms": np.arange(n) * 100,
        "dt_imu_s": np.full(n, 0.1),
        "accel_x": np.random.randn(n),
        "accel_y": np.random.randn(n),
        "accel_z": np.full(n, 9.81),
        "gyro_yaw": np.random.randn(n) * 0.05,
        "gyro_pitch": np.zeros(n),
        "gyro_roll": np.zeros(n),
        "ref_east_m": np.arange(n) * 0.5,
        "ref_north_m": np.arange(n) * 0.2,
        "ref_up_m": np.zeros(n),
        "ref_vel_east_mps": np.full(n, 5.0),
        "ref_vel_north_mps": np.full(n, 2.0),
        "ref_speed_mps": np.full(n, 5.38),
        "valid_reference": np.ones(n, dtype=bool),
        "raw_lat": np.full(n, 52.40),
        "raw_lon": np.full(n, -1.50),
        "raw_alt": np.full(n, 100.0),
        "raw_speed_mps": np.full(n, 5.38),
        "raw_accuracy_m": np.full(n, 3.0),
    })
    
    for dur in [10, 30]:
        scenarios = find_valid_outage_intervals(df, duration_s=dur, max_scenarios=2)
        assert len(scenarios) > 0
        sc = scenarios[0]
        assert sc["duration_s"] == dur
        assert sc["num_samples"] == int(dur * 10)
        
        model_in, eval_ref = apply_outage_mask(df, sc)
        checks = verify_zero_leakage(model_in, eval_ref, sc)
        
        # All leakage checks must pass
        assert checks["all_leakage_checks_passed"] == True
        assert checks["no_reference_columns_in_input"] == True
        assert checks["gps_correctly_masked_during_outage"] == True
        
        # Verify GPS is unavailable during outage
        s_idx, e_idx = sc["start_idx"], sc["end_idx"]
        assert not model_in.loc[s_idx:e_idx, "gps_available"].any()
        
        # Verify model_in contains ONLY allowed columns
        allowed_cols = {
            "time_s", "time_ms", "dt_imu_s",
            "accel_x", "accel_y", "accel_z",
            "gyro_yaw", "gyro_pitch", "gyro_roll",
            "gps_available"
        }
        assert set(model_in.columns) == allowed_cols
