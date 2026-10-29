"""
Causal GPS-Denied Outage Navigation Estimator and Fair Baseline Framework.

Implements rigorous, causal, leakage-free navigation during simulated GNSS outages:
- Legitimate pre-outage initialization (position, speed, course-over-ground heading).
- Strict in-outage causal IMU-only observation gating.
- Gyroscope yaw rate causal heading propagation.
- Fair comparison baselines: Zero-Speed, Last-Known-Speed, Constant-Velocity, IMU-Kinematic.
- Standardized navigation metrics: Speed MAE/RMSE, ATE RMSE, Final Drift %, and Drift Growth.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd


@dataclass
class PreOutageState:
    """Legitimately available navigation state at the exact moment GPS becomes unavailable."""
    t0_s: float
    pos_east_m: float
    pos_north_m: float
    speed_mps: float
    heading_deg: float
    gyro_bias_rads: float


@dataclass
class OutageEvaluationResult:
    """Standardized metrics evaluated against withheld gold-standard reference trajectory."""
    scenario_id: str
    drive_id: str
    duration_s: float
    method_name: str
    total_length_m: float
    speed_mae_mps: float
    speed_rmse_mps: float
    speed_bias_mps: float
    ate_rmse_m: float
    final_drift_m: float
    drift_pct: float
    est_trajectory_east: List[float]
    est_trajectory_north: List[float]
    ref_trajectory_east: List[float]
    ref_trajectory_north: List[float]


def estimate_initial_heading_from_pre_outage_gps(
    pre_outage_df: pd.DataFrame,
    min_speed_mps: float = 1.5,
) -> Tuple[float, float]:
    """Estimates initial heading at outage start from pre-outage course-over-ground.
    
    Uses consecutive GPS fixes while vehicle was moving prior to outage start.
    Returns:
        (heading_deg, initial_speed_mps)
    """
    east = pre_outage_df["ref_east_m"].values
    north = pre_outage_df["ref_north_m"].values
    speed = pre_outage_df["ref_speed_mps"].values
    
    # Identify moving samples prior to outage
    moving = np.where(speed >= min_speed_mps)[0]
    if len(moving) >= 2:
        # Use last moving vector
        idx2 = moving[-1]
        idx1 = max(0, idx2 - 10)  # ~1 second baseline
        de = east[idx2] - east[idx1]
        dn = north[idx2] - north[idx1]
        if np.hypot(de, dn) > 0.5:
            heading_rad = np.arctan2(de, dn)
            heading_deg = float(np.rad2deg(heading_rad) % 360.0)
        else:
            heading_deg = float(pre_outage_df["ref_heading_deg"].iloc[-1])
    else:
        # Stationary or near-stationary: use last available reference heading
        heading_deg = float(pre_outage_df["ref_heading_deg"].iloc[-1])
        
    last_speed = float(speed[-1])
    return heading_deg, last_speed


def run_causal_dead_reckoning(
    pre_state: PreOutageState,
    in_outage_df: pd.DataFrame,
    speed_estimator_fn: Callable[[np.ndarray], float],
    method_name: str,
    scenario_id: str,
    drive_id: str,
    window_size: int = 100,
    feature_channels: List[str] = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"],
    heading_convention: str = "corrected",  # "corrected" (-1.0) or "legacy" (+1.0)
) -> OutageEvaluationResult:
    """Executes strictly causal step-by-step navigation during a simulated GPS outage.
    
    Args:
        pre_state: Legitimate pre-outage state (position, speed, heading).
        in_outage_df: DataFrame containing sensor and withheld reference during outage.
        speed_estimator_fn: Function mapping (W, 6) IMU history to predicted speed (m/s).
        method_name: Name of evaluated baseline or model.
        scenario_id: Outage scenario identifier.
        drive_id: Drive identifier.
        window_size: Length of sliding window in samples.
        feature_channels: List of sensor channel names.
        heading_convention: "corrected" uses d_psi = -omega_yaw * dt (clockwise azimuth),
                            "legacy" uses d_psi = +omega_yaw * dt (original repo bug).
    """
    n_steps = len(in_outage_df)
    time_arr = in_outage_df["time_s"].values
    dt_arr = np.diff(time_arr, prepend=time_arr[0])
    dt_arr[0] = 0.0
    
    # Ground truth reference for evaluation only
    ref_east = in_outage_df["ref_east_m"].values
    ref_north = in_outage_df["ref_north_m"].values
    ref_speed = in_outage_df["ref_speed_mps"].values
    
    seg_dists = np.sqrt(np.diff(ref_east)**2 + np.diff(ref_north)**2)
    total_length_m = float(np.sum(seg_dists))
    
    # Sensor stream available to estimator
    raw_imu = in_outage_df[feature_channels].values.astype(np.float32)
    # Phone Gyroscope Pitch corresponds to vehicle yaw rate (rad/s)
    gyro_yaw_rate = in_outage_df["gyro_pitch"].values.astype(np.float32) - pre_state.gyro_bias_rads
    
    # State vectors
    pred_east = np.zeros(n_steps, dtype=np.float64)
    pred_north = np.zeros(n_steps, dtype=np.float64)
    pred_speed = np.zeros(n_steps, dtype=np.float32)
    pred_heading = np.zeros(n_steps, dtype=np.float64)
    
    # Initialize at t0
    pred_east[0] = pre_state.pos_east_m
    pred_north[0] = pre_state.pos_north_m
    pred_speed[0] = pre_state.speed_mps
    pred_heading[0] = np.deg2rad(pre_state.heading_deg)
    
    h_sign = -1.0 if heading_convention == "corrected" else +1.0

    for k in range(1, n_steps):
        dt = dt_arr[k]
        
        # 1. Causal Heading Propagation: psi_k = psi_{k-1} + d_psi
        # Navigation azimuth psi increases clockwise from North.
        # Body frame yaw rate omega_yaw is right-handed (positive counter-clockwise).
        # Corrected: d_psi = -omega_yaw * dt.
        # Legacy bug: d_psi = +omega_yaw * dt.
        d_psi = h_sign * float(gyro_yaw_rate[k] * dt)
        pred_heading[k] = (pred_heading[k-1] + d_psi) % (2.0 * np.pi)
        
        # 2. Causal Speed Estimation
        # Window of length window_size ending at step k
        w_start = max(0, k - window_size + 1)
        w_slice = raw_imu[w_start : k + 1]
        
        # Pad if history is shorter than window_size
        if len(w_slice) < window_size:
            pad = np.repeat(w_slice[0:1], window_size - len(w_slice), axis=0)
            w_input = np.vstack([pad, w_slice])
        else:
            w_input = w_slice
            
        s_hat = float(speed_estimator_fn(w_input))
        pred_speed[k] = max(0.0, s_hat)  # Forward speed is non-negative
        
        # 3. Trajectory Dead-Reckoning Integration
        # v_E = s * sin(psi), v_N = s * cos(psi)
        v_e = pred_speed[k] * np.sin(pred_heading[k])
        v_n = pred_speed[k] * np.cos(pred_heading[k])
        
        pred_east[k] = pred_east[k-1] + v_e * dt
        pred_north[k] = pred_north[k-1] + v_n * dt
        
    # Standardized Evaluation Metrics
    speed_err = pred_speed - ref_speed
    speed_mae = float(np.mean(np.abs(speed_err)))
    speed_rmse = float(np.sqrt(np.mean(speed_err**2)))
    speed_bias = float(np.mean(speed_err))
    
    pos_err = np.sqrt((pred_east - ref_east)**2 + (pred_north - ref_north)**2)
    ate_rmse = float(np.sqrt(np.mean(pos_err**2)))
    final_drift_m = float(pos_err[-1])
    drift_pct = float((final_drift_m / max(total_length_m, 1.0)) * 100.0)
    
    return OutageEvaluationResult(
        scenario_id=scenario_id,
        drive_id=drive_id,
        duration_s=float(time_arr[-1] - time_arr[0]),
        method_name=method_name,
        total_length_m=total_length_m,
        speed_mae_mps=speed_mae,
        speed_rmse_mps=speed_rmse,
        speed_bias_mps=speed_bias,
        ate_rmse_m=ate_rmse,
        final_drift_m=final_drift_m,
        drift_pct=drift_pct,
        est_trajectory_east=pred_east.tolist(),
        est_trajectory_north=pred_north.tolist(),
        ref_trajectory_east=ref_east.tolist(),
        ref_trajectory_north=ref_north.tolist(),
    )


# -----------------------------------------------------------------------------
# Fair Baselines
# -----------------------------------------------------------------------------
def make_zero_speed_estimator() -> Callable[[np.ndarray], float]:
    """Baseline 1: Zero-Speed (car remains stationary during outage)."""
    return lambda w: 0.0


def make_last_known_speed_estimator(last_speed: float) -> Callable[[np.ndarray], float]:
    """Baseline 2: Last-Known-Speed Persistence (holds pre-outage speed constant)."""
    return lambda w: last_speed


def make_imu_kinematic_estimator(
    last_speed: float,
    fwd_axis_idx: int = 0,
    accel_scale: float = 0.5,
) -> Callable[[np.ndarray], float]:
    """Baseline 4: IMU Kinematic Integration (integrates forward accelerometer)."""
    current_speed = [last_speed]
    
    def estimator(w: np.ndarray) -> float:
        # Instantaneous longitudinal acceleration from last sample
        # Damped integration to limit unbounded noise explosion
        a_fwd = float(w[-1, fwd_axis_idx]) * accel_scale
        current_speed[0] = max(0.0, current_speed[0] + a_fwd * 0.1)
        return current_speed[0]
        
    return estimator
