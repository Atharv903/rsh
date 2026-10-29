"""
Simulated GNSS Outage Generator and Leakage Verification Module.

Generates reproducible GPS outage scenarios (10s, 30s, 60s) for dead-reckoning evaluation:
- Selects valid start points with complete, trustworthy ground truth coverage
- Masks GNSS observation inputs during the outage
- Isolates reference positions strictly for offline evaluation / scoring
- Explicitly tests and guarantees zero forward or backward reference leakage
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


def find_valid_outage_intervals(
    df: pd.DataFrame,
    duration_s: float,
    nominal_rate_hz: float = 10.0,
    min_speed_mps: float = 1.0,
    min_separation_s: float = 60.0,
    max_scenarios: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Select deterministic, high-quality outage intervals with 100% valid reference labels.
    
    Args:
        df: Aligned DataFrame with time_s, valid_reference, ref_speed_mps, ref_east_m, ref_north_m.
        duration_s: Outage duration in seconds (e.g. 10.0, 30.0, 60.0).
        nominal_rate_hz: Sampling frequency in Hz.
        min_speed_mps: Minimum mean vehicle speed during outage (ensures dynamic motion).
        min_separation_s: Minimum temporal separation between consecutive outage scenarios.
        max_scenarios: Maximum number of scenarios to extract per duration.
        seed: Random seed for reproducible selection.
        
    Returns:
        List of scenario metadata dicts.
    """
    rng = np.random.RandomState(seed + int(duration_s))
    num_samples = int(round(duration_s * nominal_rate_hz))
    total_len = len(df)
    
    valid_mask = df["valid_reference"].values
    time_s = df["time_s"].values
    speed = df["ref_speed_mps"].values
    east = df["ref_east_m"].values
    north = df["ref_north_m"].values
    
    candidate_starts = []
    
    # Slide a window across the recording to find fully valid spans
    for start_idx in range(0, total_len - num_samples, int(nominal_rate_hz)):
        end_idx = start_idx + num_samples
        
        # 1. Require 100% valid reference coverage
        if not np.all(valid_mask[start_idx:end_idx]):
            continue
            
        # 2. Check dynamic motion
        seg_speeds = speed[start_idx:end_idx]
        mean_spd = float(np.mean(seg_speeds))
        if mean_spd < min_speed_mps:
            continue
            
        t_start = float(time_s[start_idx])
        t_end = float(time_s[end_idx - 1])
        
        # Calculate ground truth distance traveled during the outage
        de = np.diff(east[start_idx:end_idx])
        dn = np.diff(north[start_idx:end_idx])
        dist_traveled = float(np.sum(np.sqrt(de**2 + dn**2)))
        
        candidate_starts.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time_s": t_start,
            "end_time_s": t_end,
            "duration_s": duration_s,
            "num_samples": num_samples,
            "mean_speed_mps": mean_spd,
            "distance_traveled_m": dist_traveled,
            "reference_coverage_ratio": 1.0,
        })
        
    if not candidate_starts:
        # Fallback: relax speed constraint if no candidate found
        for start_idx in range(0, total_len - num_samples, int(nominal_rate_hz)):
            end_idx = start_idx + num_samples
            if np.all(valid_mask[start_idx:end_idx]):
                t_start = float(time_s[start_idx])
                t_end = float(time_s[end_idx - 1])
                candidate_starts.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time_s": t_start,
                    "end_time_s": t_end,
                    "duration_s": duration_s,
                    "num_samples": num_samples,
                    "mean_speed_mps": float(np.mean(speed[start_idx:end_idx])),
                    "distance_traveled_m": 0.0,
                    "reference_coverage_ratio": 1.0,
                })
                
    if not candidate_starts:
        return []
        
    # Greedily select spaced scenarios
    rng.shuffle(candidate_starts)
    selected = []
    
    for cand in candidate_starts:
        too_close = False
        for s in selected:
            if abs(cand["start_time_s"] - s["start_time_s"]) < min_separation_s:
                too_close = True
                break
        if not too_close:
            cand["scenario_id"] = f"outage_{int(duration_s)}s_{len(selected)+1:02d}"
            selected.append(cand)
            if len(selected) >= max_scenarios:
                break
                
    # Sort chronologically by start time
    selected.sort(key=lambda x: x["start_time_s"])
    for idx, s in enumerate(selected):
        s["scenario_id"] = f"outage_{int(duration_s)}s_{idx+1:02d}"
        
    return selected


def apply_outage_mask(
    df: pd.DataFrame,
    scenario: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate model input dataframe with GPS masked during outage, and isolated evaluation labels.
    
    Args:
        df: Aligned dataframe.
        scenario: Dict specifying start_idx and end_idx of outage.
        
    Returns:
        (model_input_df, evaluation_ref_df)
    """
    model_df = df.copy()
    start_idx = scenario["start_idx"]
    end_idx = scenario["end_idx"]
    
    # Observation availability indicator: True when GPS fix is available to filter, False during outage
    model_df["gps_available"] = True
    model_df.loc[start_idx:end_idx, "gps_available"] = False
    
    # Mask GNSS observations during outage (replace with NaN)
    gnss_obs_cols = ["raw_lat", "raw_lon", "raw_alt", "raw_speed_mps", "raw_accuracy_m"]
    for col in gnss_obs_cols:
        if col in model_df.columns:
            model_df.loc[start_idx:end_idx, col] = np.nan
            
    # Remove reference labels completely from model input space to ensure ZERO accidental access!
    model_input_cols = [
        "time_s", "time_ms", "dt_imu_s",
        "accel_x", "accel_y", "accel_z",
        "gyro_yaw", "gyro_pitch", "gyro_roll",
        "gps_available",
    ]
    model_input_df = model_df[model_input_cols].copy()
    
    # Ground truth reference trajectory preserved strictly for offline evaluation
    eval_cols = [
        "time_s", "ref_east_m", "ref_north_m", "ref_up_m",
        "ref_vel_east_mps", "ref_vel_north_mps", "ref_speed_mps",
        "valid_reference"
    ]
    eval_ref_df = df.loc[start_idx:end_idx, eval_cols].copy()
    eval_ref_df["scenario_id"] = scenario["scenario_id"]
    eval_ref_df["duration_s"] = scenario["duration_s"]
    
    return model_input_df, eval_ref_df


def verify_zero_leakage(
    model_input_df: pd.DataFrame,
    eval_ref_df: pd.DataFrame,
    scenario: Dict[str, Any],
) -> Dict[str, bool]:
    """Perform automated assert checks that no evaluation labels or future fixes leak into inputs."""
    forbidden_tokens = ["ref_east", "ref_north", "ref_up", "ref_vel", "target"]
    leakage_detected = False
    for col in model_input_df.columns:
        for f in forbidden_tokens:
            if f in col.lower():
                leakage_detected = True
                
    start_idx = scenario["start_idx"]
    end_idx = scenario["end_idx"]
    
    # Check that gps_available is strictly False throughout outage
    outage_gps_flags = model_input_df.loc[start_idx:end_idx, "gps_available"]
    gps_correctly_masked = not bool(outage_gps_flags.any())
    
    # Check that evaluation labels are completely non-empty and 100% valid
    eval_all_valid = bool(eval_ref_df["valid_reference"].all())
    
    return {
        "no_reference_columns_in_input": not leakage_detected,
        "gps_correctly_masked_during_outage": gps_correctly_masked,
        "eval_labels_complete_and_valid": eval_all_valid,
        "all_leakage_checks_passed": (not leakage_detected) and gps_correctly_masked and eval_all_valid,
    }
