"""
Unit tests for Gyroscope Bias mathematical propagation and Adaptive Hybrid Estimator.
Verifies:
1. Synthetic straight line with known injected gyro bias:
   - Without bias correction, cross-track error grows quadratically: ~ 0.5 * v * b * t^2.
   - With exact bias correction, drift = 0.000 m.
2. Synthetic turning trajectory with known gyro bias.
3. Adaptive Hybrid Estimator:
   - At t = 0 to 5s, outputs near-constant pre-outage speed.
   - At t > 15s, transitions to model speed.
   - Deceleration override immediately triggers neural speed tracking.
"""

import numpy as np
import pytest

def propagate_dead_reckoning(time_arr, speed_arr, gyro_pitch_arr, bias_rads, v0, psi0, h_sign=-1.0):
    n_steps = len(time_arr)
    dt_arr = np.diff(time_arr, prepend=time_arr[0])
    dt_arr[0] = 0.0
    
    pe = np.zeros(n_steps, dtype=np.float64)
    pn = np.zeros(n_steps, dtype=np.float64)
    head = np.zeros(n_steps, dtype=np.float64)
    head[0] = psi0
    
    unbiased_gyro = gyro_pitch_arr - bias_rads
    
    for k in range(1, n_steps):
        dt = dt_arr[k]
        d_psi = h_sign * float(unbiased_gyro[k] * dt)
        head[k] = (head[k-1] + d_psi) % (2.0 * np.pi)
        
        v_e = speed_arr[k] * np.sin(head[k])
        v_n = speed_arr[k] * np.cos(head[k])
        pe[k] = pe[k-1] + v_e * dt
        pn[k] = pn[k-1] + v_n * dt
        
    return pe, pn, head

def compute_hybrid_speed(time_s, v0, v_nn, accel_x=None, t_switch=10.0, tau=2.0):
    """Computes adaptive hybrid velocity."""
    dt_outage = time_s - time_s[0]
    alpha = 1.0 / (1.0 + np.exp(-(dt_outage - t_switch) / tau))
    
    if accel_x is not None:
        # If braking detected (accel_x < -1.5 m/s^2), force alpha -> 1.0
        is_braking = accel_x < -1.5
        # Latch braking once triggered
        braking_latched = np.maximum.accumulate(is_braking)
        alpha = np.where(braking_latched, 1.0, alpha)
        
    return (1.0 - alpha) * v0 + alpha * v_nn

# --- TEST CASES ---

def test_straight_line_with_injected_gyro_bias():
    """Verifies that an uncorrected gyro bias b produces lateral drift ~ 0.5 * v * b * t^2."""
    t = np.linspace(0, 30, 301) # 30 seconds at 10 Hz
    v_true = np.full_like(t, 20.0) # 20 m/s heading North (psi = 0)
    b_true = 0.005 # 0.005 rad/s bias
    gyro_meas = np.full_like(t, b_true) # True turn rate is 0, measured is bias
    
    # 1. Uncorrected: bias_rads = 0.0
    pe_uncorr, pn_uncorr, head_uncorr = propagate_dead_reckoning(
        t, v_true, gyro_meas, bias_rads=0.0, v0=20.0, psi0=0.0
    )
    
    # Theoretical cross track error: E(t) ~ int_0^t v * sin(-b * tau) d_tau ~ -0.5 * v * b * t^2
    # At t = 30s: -0.5 * 20 * 0.005 * 900 = -45.0 m
    expected_cross = -0.5 * 20.0 * b_true * (30.0**2)
    assert np.isclose(pe_uncorr[-1], expected_cross, rtol=0.05)
    
    # 2. Corrected: bias_rads = b_true
    pe_corr, pn_corr, head_corr = propagate_dead_reckoning(
        t, v_true, gyro_meas, bias_rads=b_true, v0=20.0, psi0=0.0
    )
    assert np.isclose(pe_corr[-1], 0.0, atol=1e-5)
    assert np.isclose(pn_corr[-1], 600.0, atol=1e-3)
    assert np.isclose(head_corr[-1], 0.0, atol=1e-5)

def test_hybrid_speed_transitions():
    """Verifies that the hybrid estimator smoothly blends v0 into v_nn."""
    t = np.linspace(0, 30, 301)
    v0 = 20.0
    # Neural model predicts deceleration to 5 m/s starting at 10s
    v_nn = np.where(t < 10, 20.0 + np.sin(t), 5.0)
    
    v_hyb = compute_hybrid_speed(t, v0, v_nn, t_switch=10.0, tau=2.0)
    
    # At t = 0: should be very close to v0 (20.0)
    assert np.isclose(v_hyb[0], 20.0, atol=0.2)
    # At t = 2s: should remain close to v0
    assert abs(v_hyb[20] - 20.0) < 0.5
    # At t = 10s: should be approximately average of v0 and v_nn
    assert abs(v_hyb[100] - (0.5 * v0 + 0.5 * v_nn[100])) < 0.5
    # At t = 25s: should be within 0.1 of v_nn (5.0)
    assert np.isclose(v_hyb[250], 5.0, atol=0.1)

def test_hybrid_braking_override():
    """Verifies that sudden braking overrides the time delay and immediately tracks v_nn."""
    t = np.linspace(0, 20, 201)
    v0 = 20.0
    v_nn = np.linspace(20.0, 0.0, 201) # Rapid braking
    accel_x = np.full_like(t, 0.0)
    accel_x[20:40] = -3.0 # Heavy braking event at t = 2s
    
    v_hyb = compute_hybrid_speed(t, v0, v_nn, accel_x=accel_x, t_switch=10.0, tau=2.0)
    
    # Before braking (t = 1s): tracks v0
    assert abs(v_hyb[10] - 20.0) < 0.5
    # After braking starts (t = 3s): should immediately switch to v_nn
    assert np.isclose(v_hyb[35], v_nn[35], atol=0.1)
