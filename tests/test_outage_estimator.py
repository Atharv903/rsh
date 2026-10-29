"""
Unit tests for Phase 6 Causal Outage Estimator, Baselines, and Phase 5 Grouped Split.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.outage_estimator import (
    PreOutageState,
    estimate_initial_heading_from_pre_outage_gps,
    run_causal_dead_reckoning,
    make_zero_speed_estimator,
    make_last_known_speed_estimator,
    make_imu_kinematic_estimator,
)


def test_initial_heading_estimation():
    # Synthetic pre-outage trajectory moving strictly East (heading = 90 deg)
    t = np.arange(50) * 0.1
    east = 100.0 + t * 10.0  # moving East at 10 m/s
    north = np.full(50, 200.0)
    speed = np.full(50, 10.0)
    heading = np.full(50, 90.0)
    
    df = pd.DataFrame({
        "time_s": t,
        "ref_east_m": east,
        "ref_north_m": north,
        "ref_speed_mps": speed,
        "ref_heading_deg": heading,
    })
    
    est_heading, last_spd = estimate_initial_heading_from_pre_outage_gps(df)
    assert np.isclose(est_heading, 90.0, atol=1.0), f"Expected heading ~90 deg, got {est_heading}"
    assert np.isclose(last_spd, 10.0, atol=1e-3)


def test_causal_dead_reckoning_zero_speed_baseline():
    # If speed is predicted as 0, vehicle should remain at initial position
    t = np.arange(100) * 0.1
    ref_east = 50.0 + t * 5.0
    ref_north = 100.0 + t * 0.0
    ref_speed = np.full(100, 5.0)
    
    df = pd.DataFrame({
        "time_s": t,
        "ref_east_m": ref_east,
        "ref_north_m": ref_north,
        "ref_speed_mps": ref_speed,
        "accel_x": np.zeros(100),
        "accel_y": np.zeros(100),
        "accel_z": np.full(100, 9.81),
        "gyro_yaw": np.zeros(100),
        "gyro_pitch": np.zeros(100),
        "gyro_roll": np.zeros(100),
    })
    
    pre_state = PreOutageState(
        t0_s=0.0,
        pos_east_m=50.0,
        pos_north_m=100.0,
        speed_mps=5.0,
        heading_deg=90.0,
        gyro_bias_rads=0.0,
    )
    
    res = run_causal_dead_reckoning(
        pre_state=pre_state,
        in_outage_df=df,
        speed_estimator_fn=make_zero_speed_estimator(),
        method_name="Zero-Speed Baseline",
        scenario_id="test_zero",
        drive_id="test_drive",
    )
    
    # Position should not move from initial position
    assert np.allclose(res.est_trajectory_east, 50.0)
    assert np.allclose(res.est_trajectory_north, 100.0)
    assert res.speed_mae_mps > 4.5, "Zero speed error against 5 m/s truth should be ~5 m/s"
    assert res.final_drift_m > 40.0, "Vehicle drifted ~50m away while model stayed still"


def test_causal_dead_reckoning_last_known_speed_baseline():
    t = np.arange(100) * 0.1
    # Constant moving East at 10 m/s
    ref_east = t * 10.0
    ref_north = np.zeros(100)
    ref_speed = np.full(100, 10.0)
    
    df = pd.DataFrame({
        "time_s": t,
        "ref_east_m": ref_east,
        "ref_north_m": ref_north,
        "ref_speed_mps": ref_speed,
        "accel_x": np.zeros(100),
        "accel_y": np.zeros(100),
        "accel_z": np.full(100, 9.81),
        "gyro_yaw": np.zeros(100),
        "gyro_pitch": np.zeros(100),  # zero yaw rate -> straight line
        "gyro_roll": np.zeros(100),
    })
    
    pre_state = PreOutageState(
        t0_s=0.0,
        pos_east_m=0.0,
        pos_north_m=0.0,
        speed_mps=10.0,
        heading_deg=90.0,  # East
        gyro_bias_rads=0.0,
    )
    
    res = run_causal_dead_reckoning(
        pre_state=pre_state,
        in_outage_df=df,
        speed_estimator_fn=make_last_known_speed_estimator(10.0),
        method_name="Last-Known-Speed Baseline",
        scenario_id="test_constant",
        drive_id="test_drive",
    )
    
    # Since vehicle was truly at constant speed and straight line, error should be near zero!
    assert np.isclose(res.speed_mae_mps, 0.0, atol=1e-3)
    assert np.isclose(res.final_drift_m, 0.0, atol=1e-2)
    assert np.isclose(res.drift_pct, 0.0, atol=1e-2)


def test_grouped_split_manifest_and_leakage_isolation():
    manifest_p = "data/splits/drive_split_manifest_grouped.json"
    assert os.path.exists(manifest_p)
    with open(manifest_p, "r") as f:
        manifest = json.load(f)
        
    train_drives = set(manifest["splits"]["train"]["drives"])
    val_drives = set(manifest["splits"]["val"]["drives"])
    test_drives = set(manifest["splits"]["test"]["drives"])
    
    # Assert zero drive overlap
    assert len(train_drives.intersection(val_drives)) == 0, "Train and Val must not share drives"
    assert len(train_drives.intersection(test_drives)) == 0, "Train and Test must not share drives"
    assert len(val_drives.intersection(test_drives)) == 0, "Val and Test must not share drives"
    
    # Assert all fragments of S3 are exclusively in Val
    assert val_drives == {"S3a", "S3b", "S3c"}, "All fragments of S3 must be grouped in Val"
    assert test_drives == {"S4"}, "Test must be exclusively independent Drive S4"
    assert train_drives == {"S1", "S2"}, "Train must be Session S1 & S2"
