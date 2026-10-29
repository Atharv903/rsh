"""
Unit tests for the Phase 16 Error-State Kalman Filter (ESKF).
Verifies:
- State propagation kinematics
- Covariance symmetry and positive definiteness
- Joseph update numerical stability
- Speed measurement updates
- Non-Holonomic Constraint (NHC) updates
- Zero-Velocity Updates (ZUPT)
"""

import pytest
import numpy as np
from src.eskf import VehicleESKF

def test_eskf_initialization():
    filter_ = VehicleESKF(
        init_p_e=10.0,
        init_p_n=20.0,
        init_speed_mps=15.0,
        init_heading_rad=np.deg2rad(90.0), # Heading East (pi/2)
        init_gyro_bias_rads=0.002,
    )
    s = filter_.state
    assert np.isclose(s.p_e, 10.0)
    assert np.isclose(s.p_n, 20.0)
    assert np.isclose(s.psi, np.pi / 2.0)
    assert np.isclose(s.v_e, 15.0, atol=1e-5) # Moving East
    assert np.isclose(s.v_n, 0.0, atol=1e-5)
    assert np.isclose(s.b_g, 0.002)
    # Check covariance positive definiteness
    eigvals = np.linalg.eigvals(s.P)
    assert (eigvals > 0).all()
    assert np.allclose(s.P, s.P.T)

def test_eskf_straight_line_propagation():
    # Vehicle moving North at 10 m/s for 5 seconds (50 steps of 0.1s)
    filter_ = VehicleESKF(
        init_p_e=0.0,
        init_p_n=0.0,
        init_speed_mps=10.0,
        init_heading_rad=0.0, # North
        init_gyro_bias_rads=0.0,
    )
    dt = 0.1
    for _ in range(50):
        filter_.predict(dt, gyro_pitch_rads=0.0)
        
    s = filter_.state
    assert np.isclose(s.p_e, 0.0, atol=1e-5)
    assert np.isclose(s.p_n, 50.0, atol=1e-3)
    assert np.isclose(s.psi, 0.0)
    # Check covariance symmetry and positivity
    assert np.allclose(s.P, s.P.T)
    assert (np.linalg.eigvals(s.P) > 0).all()

def test_eskf_turn_propagation():
    # Vehicle turning right 90 degrees over 10 seconds: omega = pi / (2 * 10) = 0.157 rad/s
    filter_ = VehicleESKF(
        init_p_e=0.0,
        init_p_n=0.0,
        init_speed_mps=0.0,
        init_heading_rad=0.0, # North
        init_gyro_bias_rads=0.0,
        h_sign=-1.0,
    )
    dt = 0.1
    # Recall dpsi/dt = h_sign * (gyro_pitch - b_g) = -1.0 * gyro_pitch
    # To increase psi by 90 deg (pi/2) clockwise, gyro_pitch must be -0.1570796 rad/s
    turn_rate_gyro = -np.pi / 20.0
    for _ in range(100): # 10 seconds
        filter_.predict(dt, gyro_pitch_rads=turn_rate_gyro)
        
    assert np.isclose(filter_.state.psi, np.pi / 2.0, atol=1e-4) # Heading East

def test_eskf_speed_update_reduces_variance():
    filter_ = VehicleESKF(
        init_p_e=0.0,
        init_p_n=0.0,
        init_speed_mps=10.0,
        init_heading_rad=np.deg2rad(90.0), # East
        sigma_vel_init=2.0,
    )
    initial_vel_var = filter_.state.P[2, 2] + filter_.state.P[3, 3]
    # Update with accurate speed measurement
    accepted = filter_.update_speed(v_pred_mps=12.0, var_v=0.25)
    assert accepted
    updated_vel_var = filter_.state.P[2, 2] + filter_.state.P[3, 3]
    assert updated_vel_var < initial_vel_var
    # Forward velocity moved towards 12.0
    fwd_speed = filter_.state.v_e * np.sin(filter_.state.psi) + filter_.state.v_n * np.cos(filter_.state.psi)
    assert fwd_speed > 10.0
    assert np.allclose(filter_.state.P, filter_.state.P.T)
    assert (np.linalg.eigvals(filter_.state.P) > 0).all()

def test_eskf_nhc_suppresses_lateral_velocity():
    # Vehicle heading North (psi = 0), but with artificial lateral velocity v_e = 3.0 m/s
    filter_ = VehicleESKF(
        init_p_e=0.0,
        init_p_n=0.0,
        init_speed_mps=10.0,
        init_heading_rad=0.0, # North: v_n = 10, v_e = 0
    )
    # Inject spurious lateral velocity v_e = 3.0 m/s
    filter_.state.v_e = 3.0
    v_lat_before = filter_.state.v_e * np.cos(filter_.state.psi) - filter_.state.v_n * np.sin(filter_.state.psi)
    assert np.isclose(v_lat_before, 3.0)
    
    # Apply NHC constraint (virtual observation v_lat = 0)
    accepted = filter_.update_nhc(var_nhc=0.05, gate_threshold=100.0)
    assert accepted
    
    v_lat_after = filter_.state.v_e * np.cos(filter_.state.psi) - filter_.state.v_n * np.sin(filter_.state.psi)
    # Lateral velocity should be suppressed substantially towards zero
    assert abs(v_lat_after) < abs(v_lat_before)
    assert abs(v_lat_after) < 1.0
    assert np.allclose(filter_.state.P, filter_.state.P.T)

def test_eskf_zupt():
    filter_ = VehicleESKF(
        init_p_e=0.0,
        init_p_n=0.0,
        init_speed_mps=5.0,
        init_heading_rad=0.0,
    )
    # Apply ZUPT
    accepted = filter_.update_zupt(var_zupt=0.001)
    assert accepted
    speed = np.sqrt(filter_.state.v_e**2 + filter_.state.v_n**2)
    assert speed < 0.5
    assert np.allclose(filter_.state.P, filter_.state.P.T)
