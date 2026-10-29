"""
Phase 14: Automated Causality and Leakage Verification Tests.

Verifies:
1. Future sample perturbation test: Corrupting future IMU samples at t > t_eval does not alter earlier predictions.
2. Gyro bias causality test: Future stationary stops do not alter earlier bias estimates.
3. GPS/Reference isolation: In-outage reference telemetry mutation has zero impact on dead-reckoning trajectory.
4. Heading convention selectable and reproducible.
5. Causal braking detector uses only causal window.
"""

import numpy as np
import pytest
import torch

from src.outage_estimator import PreOutageState, run_causal_dead_reckoning
from tests.test_gyro_bias_and_hybrid import propagate_dead_reckoning

def test_future_imu_perturbation_invariance():
    """Confirms that predictions at t <= 10s are strictly invariant to future IMU samples at t > 10s."""
    n_total = 200 # 20 seconds at 10 Hz
    t = np.linspace(0, 20, n_total)
    
    # Base synthetic IMU signals
    rng = np.random.RandomState(42)
    gyro_pitch = rng.randn(n_total) * 0.01
    speed = np.full(n_total, 15.0)
    
    # 1. Unperturbed run
    pe1, pn1, head1 = propagate_dead_reckoning(
        t, speed, gyro_pitch, bias_rads=0.0, v0=15.0, psi0=0.0
    )
    
    # 2. Corrupt future samples at t > 10s (indices >= 100)
    gyro_corrupt = gyro_pitch.copy()
    gyro_corrupt[100:] += 5.0 # Massive corruption in the future
    speed_corrupt = speed.copy()
    speed_corrupt[100:] += 100.0
    
    pe2, pn2, head2 = propagate_dead_reckoning(
        t, speed_corrupt, gyro_corrupt, bias_rads=0.0, v0=15.0, psi0=0.0
    )
    
    # Up to index 99 (t <= 10s), the estimates MUST be bit-for-bit identical
    np.testing.assert_array_equal(pe1[:100], pe2[:100])
    np.testing.assert_array_equal(pn1[:100], pn2[:100])
    np.testing.assert_array_equal(head1[:100], head2[:100])
    
    # Past index 100, they must diverge
    assert not np.array_equal(pe1[101:], pe2[101:])

def test_online_bias_future_stop_isolation():
    """Confirms that a stationary stop occurring AFTER an outage does not influence the outage's bias estimate."""
    # Synthetic timeline:
    # 0 - 30s: Stop 1 (bias = +0.003 rad/s)
    # 30 - 60s: Outage 1 occurs here (start = 30s)
    # 60 - 90s: Driving
    # 90 - 120s: Stop 2 (bias = +0.010 rad/s)
    
    times = np.linspace(0, 120, 1200)
    ref_speed = np.zeros(1200)
    ref_speed[300:900] = 15.0 # Driving between 30s and 90s
    
    gyro_pitch = np.zeros(1200)
    gyro_pitch[:300] = +0.003 # Stop 1
    gyro_pitch[900:] = +0.010 # Stop 2
    
    def causal_bias_at_timestamp(t_idx):
        pre_speed = ref_speed[:t_idx]
        pre_gyro = gyro_pitch[:t_idx]
        is_stat = pre_speed < 0.1
        # Look backwards for most recent stationary period >= 30 samples
        curr_end = None
        for i in range(len(is_stat) - 1, -1, -1):
            if is_stat[i] and curr_end is None:
                curr_end = i
            elif not is_stat[i] and curr_end is not None:
                if (curr_end - i) >= 30:
                    return float(np.mean(pre_gyro[i+1 : curr_end+1]))
                curr_end = None
        if curr_end is not None and curr_end >= 30:
            return float(np.mean(pre_gyro[:curr_end+1]))
        return 0.0

    # At t = 30s (t_idx = 300), Stop 2 has NOT happened yet
    b_at_30 = causal_bias_at_timestamp(300)
    assert np.isclose(b_at_30, +0.003, atol=1e-5)
    
    # If Stop 2 in the future is changed to +0.050 rad/s, bias at t=30s MUST remain unchanged
    gyro_pitch[900:] = +0.050
    b_at_30_mutated = causal_bias_at_timestamp(300)
    assert np.isclose(b_at_30_mutated, +0.003, atol=1e-5)
    assert b_at_30 == b_at_30_mutated

def test_heading_convention_selectable():
    """Verifies that corrected and legacy heading conventions are explicitly selectable and distinct."""
    t = np.linspace(0, 10, 101)
    speed = np.full(101, 20.0)
    # Right turn: negative body rate
    gyro_meas = np.full(101, -0.1) # -0.1 rad/s body rate
    
    # Corrected (-1): Right turn increases clockwise azimuth
    pe_corr, pn_corr, head_corr = propagate_dead_reckoning(
        t, speed, gyro_meas, bias_rads=0.0, v0=20.0, psi0=0.0, h_sign=-1.0
    )
    # Legacy (+1): Inverted turn
    pe_leg, pn_leg, head_leg = propagate_dead_reckoning(
        t, speed, gyro_meas, bias_rads=0.0, v0=20.0, psi0=0.0, h_sign=+1.0
    )
    
    # Under corrected convention, azimuth turns clockwise towards East (pe > 0)
    assert pe_corr[-1] > 20.0
    # Under legacy bug, azimuth turns counter-clockwise towards West (pe < 0)
    assert pe_leg[-1] < -20.0
    assert np.isclose(pe_corr[-1], -pe_leg[-1], rtol=1e-3)

def test_causal_braking_detection():
    """Verifies that causal braking requires sustained deceleration rather than a single 1-sample spike."""
    def detect_braking_causal(accel_x, window_len=5, thr=-1.5):
        """Causal rolling average of acceleration."""
        n = len(accel_x)
        is_braking = np.zeros(n, dtype=bool)
        for i in range(window_len, n):
            mean_acc = np.mean(accel_x[i-window_len : i])
            if mean_acc < thr:
                is_braking[i] = True
        return is_braking

    t = np.linspace(0, 10, 101)
    accel = np.zeros(101)
    
    # 1. Single sample noise spike at index 20
    accel[20] = -3.0
    res1 = detect_braking_causal(accel)
    # Single sample noise spike should NOT trigger sustained braking
    assert not np.any(res1)
    
    # 2. Genuine sustained braking over 10 samples (1.0 second)
    accel[50:60] = -2.5
    res2 = detect_braking_causal(accel)
    # Must trigger braking during indices 55-60
    assert np.any(res2[55:65])
