"""
Unit tests for Navigation Heading Integration and Dead Reckoning.
Phase 12A: Verification of heading mathematics, coordinate frames, and sign conventions.

Covers:
1. Straight-line motion (zero yaw rate).
2. Right turn (negative body yaw rate, increasing clockwise heading).
3. Left turn (positive body yaw rate, decreasing heading).
4. 90-degree right turn from North (heading transitions from 0 to pi/2, moving East).
5. Complete 360-degree rotation (heading wraps around 2*pi, returns to original heading).
6. Constant-speed straight-line position integration (d = v * t).
7. Legacy vs Corrected implementation divergence on turns.
"""

import pytest
import numpy as np
import pandas as pd
from src.outage_estimator import (
    PreOutageState,
    run_causal_dead_reckoning,
    make_last_known_speed_estimator,
    make_zero_speed_estimator,
)


def create_synthetic_outage_df(
    duration_s: float = 10.0,
    dt: float = 0.1,
    speed_mps: float = 10.0,
    gyro_pitch_rads: float = 0.0,
    initial_heading_deg: float = 0.0,
):
    """Creates a deterministic synthetic telemetry stream for testing."""
    n_steps = int(duration_s / dt)
    time_s = np.arange(n_steps) * dt
    
    # Ground truth heading: right-hand body rate omega = gyro_pitch
    # In navigation frame, dpsi/dt = -gyro_pitch
    true_heading_rad = (np.deg2rad(initial_heading_deg) - gyro_pitch_rads * time_s) % (2.0 * np.pi)
    true_heading_deg = np.rad2deg(true_heading_rad)
    
    # Ground truth positions (integrated with dt)
    gt_east = np.zeros(n_steps)
    gt_north = np.zeros(n_steps)
    for k in range(1, n_steps):
        v_e = speed_mps * np.sin(true_heading_rad[k])
        v_n = speed_mps * np.cos(true_heading_rad[k])
        gt_east[k] = gt_east[k-1] + v_e * dt
        gt_north[k] = gt_north[k-1] + v_n * dt
        
    df = pd.DataFrame({
        "time_s": time_s,
        "accel_x": np.zeros(n_steps, dtype=np.float32),
        "accel_y": np.zeros(n_steps, dtype=np.float32),
        "accel_z": np.full(n_steps, 9.81, dtype=np.float32),
        "gyro_yaw": np.zeros(n_steps, dtype=np.float32),
        "gyro_pitch": np.full(n_steps, gyro_pitch_rads, dtype=np.float32),
        "gyro_roll": np.zeros(n_steps, dtype=np.float32),
        "ref_east_m": gt_east,
        "ref_north_m": gt_north,
        "ref_speed_mps": np.full(n_steps, speed_mps, dtype=np.float32),
        "ref_heading_deg": true_heading_deg,
    })
    return df, PreOutageState(
        t0_s=0.0,
        pos_east_m=0.0,
        pos_north_m=0.0,
        speed_mps=speed_mps,
        heading_deg=initial_heading_deg,
        gyro_bias_rads=0.0,
    )


def test_straight_line_motion():
    """Case 1: Straight-line motion with zero yaw rate."""
    speed = 15.0
    dur = 10.0
    df, pre_state = create_synthetic_outage_df(duration_s=dur, speed_mps=speed, gyro_pitch_rads=0.0, initial_heading_deg=0.0)
    
    res = run_causal_dead_reckoning(
        pre_state, df, make_last_known_speed_estimator(speed), "CV", "test_straight", "synth", heading_convention="corrected"
    )
    
    # After 10s at 15 m/s Heading North (0 deg), North position should be ~ 150m, East ~ 0m
    expected_north = speed * (dur - 0.1)
    assert np.isclose(res.est_trajectory_north[-1], expected_north, atol=1e-3)
    assert np.isclose(res.est_trajectory_east[-1], 0.0, atol=1e-3)
    assert np.isclose(res.final_drift_m, 0.0, atol=1e-4)


def test_constant_speed_straight_line_east():
    """Case 2: Constant-speed straight line heading East (90 deg)."""
    speed = 20.0
    dur = 10.0
    df, pre_state = create_synthetic_outage_df(duration_s=dur, speed_mps=speed, gyro_pitch_rads=0.0, initial_heading_deg=90.0)
    
    res = run_causal_dead_reckoning(
        pre_state, df, make_last_known_speed_estimator(speed), "CV", "test_east", "synth", heading_convention="corrected"
    )
    
    expected_east = speed * (dur - 0.1)
    assert np.isclose(res.est_trajectory_east[-1], expected_east, atol=1e-3)
    assert np.isclose(res.est_trajectory_north[-1], 0.0, atol=1e-3)
    assert np.isclose(res.final_drift_m, 0.0, atol=1e-4)


def test_right_turn():
    """Case 3: Right turn from North.
    In vehicle body frame, turning right produces negative yaw rate (omega < 0).
    In clockwise navigation azimuth, heading must increase from 0 towards positive East.
    """
    speed = 10.0
    dur = 5.0
    turn_rate_deg_s = -10.0  # -10 deg/s body yaw rate -> +10 deg/s azimuth rate
    omega = np.deg2rad(turn_rate_deg_s)
    
    df, pre_state = create_synthetic_outage_df(duration_s=dur, speed_mps=speed, gyro_pitch_rads=omega, initial_heading_deg=0.0)
    
    # Corrected convention
    res_corr = run_causal_dead_reckoning(
        pre_state, df, make_last_known_speed_estimator(speed), "CV", "test_right", "synth", heading_convention="corrected"
    )
    # Legacy convention
    res_leg = run_causal_dead_reckoning(
        pre_state, df, make_last_known_speed_estimator(speed), "CV", "test_right", "synth", heading_convention="legacy"
    )
    
    # Corrected should have zero drift against true right turn
    assert np.isclose(res_corr.final_drift_m, 0.0, atol=1e-4)
    # Legacy should have large drift (it turned left instead of right)
    assert res_leg.final_drift_m > 30.0


def test_left_turn():
    """Case 4: Left turn from North.
    In vehicle body frame, turning left produces positive yaw rate (omega > 0).
    In clockwise navigation azimuth, heading must decrease from 360 towards West (270).
    """
    speed = 10.0
    dur = 5.0
    turn_rate_deg_s = +15.0  # +15 deg/s body yaw rate -> -15 deg/s azimuth rate
    omega = np.deg2rad(turn_rate_deg_s)
    
    df, pre_state = create_synthetic_outage_df(duration_s=dur, speed_mps=speed, gyro_pitch_rads=omega, initial_heading_deg=0.0)
    
    res_corr = run_causal_dead_reckoning(
        pre_state, df, make_last_known_speed_estimator(speed), "CV", "test_left", "synth", heading_convention="corrected"
    )
    res_leg = run_causal_dead_reckoning(
        pre_state, df, make_last_known_speed_estimator(speed), "CV", "test_left", "synth", heading_convention="legacy"
    )
    
    assert np.isclose(res_corr.final_drift_m, 0.0, atol=1e-4)
    assert res_leg.final_drift_m > 40.0


def test_90_degree_heading_change():
    """Case 5: Exact 90-degree right turn from North (0 deg) to East (90 deg).
    Over 10 seconds at -9 deg/s = -90 deg body rate -> +90 deg azimuth change.
    """
    speed = 10.0
    dur = 10.0
    turn_rate_deg_s = -9.0
    omega = np.deg2rad(turn_rate_deg_s)
    
    df, pre_state = create_synthetic_outage_df(duration_s=dur, speed_mps=speed, gyro_pitch_rads=omega, initial_heading_deg=0.0)
    
    res_corr = run_causal_dead_reckoning(
        pre_state, df, make_last_known_speed_estimator(speed), "CV", "test_90deg", "synth", heading_convention="corrected"
    )
    
    assert np.isclose(res_corr.final_drift_m, 0.0, atol=1e-4)


def test_complete_360_degree_rotation():
    """Case 6: Complete 360-degree rotation (full circle).
    Vehicle performs full loop and returns to initial heading (0 deg).
    """
    speed = 10.0
    dur = 36.0  # 36 seconds at -10 deg/s = -360 deg body rate
    turn_rate_deg_s = -10.0
    omega = np.deg2rad(turn_rate_deg_s)
    
    df, pre_state = create_synthetic_outage_df(duration_s=dur, speed_mps=speed, gyro_pitch_rads=omega, initial_heading_deg=0.0)
    
    res_corr = run_causal_dead_reckoning(
        pre_state, df, make_last_known_speed_estimator(speed), "CV", "test_360deg", "synth", heading_convention="corrected"
    )
    
    assert np.isclose(res_corr.final_drift_m, 0.0, atol=1e-3)
