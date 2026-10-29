"""
Publication-Quality Visualization Suite for Inertial Navigation Preprocessing.

Generates all required figures:
1. Raw accelerometer and gyroscope time-series
2. Cleaned vs raw sensor signals
3. Local ENU metric GPS trajectory
4. IMU and GPS sampling interval distributions
5. Missing-data, gap, and invalid-label timelines
6. Example 10s, 30s, and 60s simulated GPS outage scenarios
7. Visual architecture distinguishing model inputs, masked observations, and evaluation references
8. High-rate 10 Hz vehicle ground truth vs 9s phone linear interpolation comparison
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set clean aesthetic styling for research publication
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "lines.linewidth": 1.2,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})


def plot_raw_imu_timeseries(
    df: pd.DataFrame,
    output_path: str = "figures/fig1_raw_imu_timeseries.png",
    time_window_s: Optional[Tuple[float, float]] = (0, 300),
) -> str:
    """Plot 1: Raw Accelerometer and Gyroscope Time-Series."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    t = df["time_s"].values if "time_s" in df.columns else (df["TIME SINCE START (ms)"].values - df["TIME SINCE START (ms)"].values[0]) / 1000.0
    
    if time_window_s is not None:
        mask = (t >= time_window_s[0]) & (t <= time_window_s[1])
        t_plot = t[mask]
        idx = mask
    else:
        t_plot = t
        idx = slice(None)
        
    ax = df["accel_x"].values[idx] if "accel_x" in df.columns else df["ACCELEROMETER X (m/s²)"].values[idx]
    ay = df["accel_y"].values[idx] if "accel_y" in df.columns else df["ACCELEROMETER Y (m/s²)"].values[idx]
    az = df["accel_z"].values[idx] if "accel_z" in df.columns else df["ACCELEROMETER Z (m/s²)"].values[idx]
    
    gx = df["gyro_yaw"].values[idx] if "gyro_yaw" in df.columns else df["GYROSCOPE Yaw (rad/s)"].values[idx]
    gy = df["gyro_pitch"].values[idx] if "gyro_pitch" in df.columns else df["GYROSCOPE Pitch (rad/s)"].values[idx]
    gz = df["gyro_roll"].values[idx] if "gyro_roll" in df.columns else df["GYROSCOPE Roll (rad/s)"].values[idx]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    
    ax1.plot(t_plot, ax, label="Accel X", color="#1f77b4", alpha=0.85)
    ax1.plot(t_plot, ay, label="Accel Y", color="#ff7f0e", alpha=0.85)
    ax1.plot(t_plot, az, label="Accel Z (incl. gravity ~9.8 m/s²)", color="#2ca02c", alpha=0.85)
    ax1.set_ylabel("Specific Force (m/s²)")
    ax1.set_title("Raw Tri-Axial Accelerometer Signals (Phone Body Frame)")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True)
    
    ax2.plot(t_plot, gx, label="Gyro Yaw", color="#d62728", alpha=0.85)
    ax2.plot(t_plot, gy, label="Gyro Pitch", color="#9467bd", alpha=0.85)
    ax2.plot(t_plot, gz, label="Gyro Roll", color="#8c564b", alpha=0.85)
    ax2.set_xlabel("Time Elapsed (seconds)")
    ax2.set_ylabel("Angular Rate (rad/s)")
    ax2.set_title("Raw Tri-Axial Gyroscope Signals (Phone Body Frame)")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_cleaned_vs_raw(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    output_path: str = "figures/fig2_cleaned_vs_raw.png",
    zoom_window_s: Tuple[float, float] = (100.0, 160.0),
) -> str:
    """Plot 2: Cleaned versus Raw Sensor Signals Comparison."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    t = df_clean["time_s"].values
    mask = (t >= zoom_window_s[0]) & (t <= zoom_window_s[1])
    t_z = t[mask]
    
    raw_ax = df_raw["accel_x"].values[mask] if "accel_x" in df_raw.columns else df_raw["ACCELEROMETER X (m/s²)"].values[mask]
    clean_ax = df_clean["accel_x"].values[mask]
    
    raw_gz = df_raw["gyro_yaw"].values[mask] if "gyro_yaw" in df_raw.columns else df_raw["GYROSCOPE Yaw (rad/s)"].values[mask]
    clean_gz = df_clean["gyro_yaw"].values[mask]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    
    ax1.plot(t_z, raw_ax, label="Raw Accel X", color="#aec7e8", linestyle="--", linewidth=1.0)
    ax1.plot(t_z, clean_ax, label="Conditioned Accel X (Outlier Clipped / Filtered)", color="#1f77b4", linewidth=1.5)
    ax1.set_ylabel("Accel X (m/s²)")
    ax1.set_title(f"Sensor Conditioning Verification: Raw vs Cleaned Signals [{zoom_window_s[0]:.0f}s - {zoom_window_s[1]:.0f}s]")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True)
    
    ax2.plot(t_z, raw_gz, label="Raw Gyro Yaw", color="#ffbb78", linestyle="--", linewidth=1.0)
    ax2.plot(t_z, clean_gz, label="Conditioned Gyro Yaw", color="#d62728", linewidth=1.5)
    ax2.set_xlabel("Time Elapsed (seconds)")
    ax2.set_ylabel("Gyro Yaw (rad/s)")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_gps_trajectory_local_enu(
    df_aligned: pd.DataFrame,
    origin: Dict[str, float],
    output_path: str = "figures/fig3_gps_trajectory_local_metres.png",
) -> str:
    """Plot 3: Ground Truth Trajectory in Local Metric ENU Tangent Plane."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    east = df_aligned["ref_east_m"].values
    north = df_aligned["ref_north_m"].values
    valid = df_aligned["valid_reference"].values
    time_s = df_aligned["time_s"].values
    
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
    
    ax_traj = fig.add_subplot(gs[0, :])
    ax_alt = fig.add_subplot(gs[1, :])
    
    # 2D Trajectory
    ax_traj.plot(east[valid], north[valid], color="#2ca02c", label="Valid Reference Trajectory (ENU)", linewidth=1.5)
    if not np.all(valid):
        ax_traj.scatter(east[~valid], north[~valid], color="#d62728", s=4, alpha=0.4, label="Gapped / Invalid Reference Segments")
        
    first_valid = np.where(valid)[0][0]
    last_valid = np.where(valid)[0][-1]
    ax_traj.plot(east[first_valid], north[first_valid], "o", markersize=9, color="#1f77b4", label=f"Start (0, 0) [Lat0={origin['lat0']:.4f}°]")
    ax_traj.plot(east[last_valid], north[last_valid], "s", markersize=9, color="#9467bd", label=f"End ({east[last_valid]:.0f}m, {north[last_valid]:.0f}m)")
    
    ax_traj.set_xlabel("Local East Position (meters)")
    ax_traj.set_ylabel("Local North Position (meters)")
    ax_traj.set_title(f"Drive Trajectory in Local Metric ENU Coordinates (Origin: {origin['lat0']:.4f}°N, {origin['lon0']:.4f}°E)")
    ax_traj.axis("equal")
    ax_traj.legend(loc="best", framealpha=0.9)
    ax_traj.grid(True)
    
    up = df_aligned["ref_up_m"].values
    ax_alt.plot(time_s[valid], up[valid], color="#7f7f7f", label="Relative Vertical Altitude (Up m)")
    ax_alt.set_xlabel("Time Elapsed (seconds)")
    ax_alt.set_ylabel("Local Up (m)")
    ax_alt.legend(loc="upper right", framealpha=0.9)
    ax_alt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_sampling_interval_distributions(
    df_raw: pd.DataFrame,
    output_path: str = "figures/fig4_sampling_interval_distributions.png",
) -> str:
    """Plot 4: Empirical Sampling Interval Distributions for IMU and GNSS."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    time_ms = df_raw["TIME SINCE START (ms)"].values.astype(float)
    dt_imu_ms = np.diff(time_ms)
    
    lat = df_raw["GPS LATITUDE (degrees)"].values.astype(float)
    lon = df_raw["GPS LONGITUDE (degrees)"].values.astype(float)
    gps_change = np.concatenate([[True], (np.diff(lat) != 0) | (np.diff(lon) != 0)])
    gps_times_s = (time_ms[gps_change] - time_ms[0]) / 1000.0
    dt_gps_s = np.diff(gps_times_s)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    ax1.hist(dt_imu_ms, bins=np.arange(85, 116, 1), color="#1f77b4", edgecolor="black", alpha=0.75, density=True)
    ax1.axvline(np.mean(dt_imu_ms), color="red", linestyle="--", label=f"Mean: {np.mean(dt_imu_ms):.2f} ms (~10.0 Hz)")
    ax1.axvline(np.median(dt_imu_ms), color="green", linestyle=":", label=f"Median: {np.median(dt_imu_ms):.1f} ms")
    ax1.set_xlabel("IMU Sampling Interval dt (ms)")
    ax1.set_ylabel("Empirical Probability Density")
    ax1.set_title("Smartphone IMU Sampling Interval Distribution")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    
    ax2.hist(dt_gps_s, bins=30, color="#ff7f0e", edgecolor="black", alpha=0.75)
    ax2.axvline(np.median(dt_gps_s), color="green", linestyle=":", label=f"Median: {np.median(dt_gps_s):.2f} s (~0.11 Hz)")
    ax2.axvline(15.0, color="red", linestyle="--", label="Max Interpolation Threshold (15s)")
    ax2.set_xlabel("GNSS Discrete Fix Update Interval (seconds)")
    ax2.set_ylabel("Occurrence Count")
    ax2.set_title("Smartphone GNSS Fix Update Distribution")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_missing_and_invalid_data(
    df_aligned: pd.DataFrame,
    output_path: str = "figures/fig5_missing_and_invalid_labels.png",
) -> str:
    """Plot 5: Quality, Masked Segments, and Label Trustworthiness Over Time."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    t = df_aligned["time_s"].values
    valid = df_aligned["valid_reference"].values.astype(int)
    acc = df_aligned["raw_accuracy_m"].values
    speed = df_aligned["ref_speed_mps"].values
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    
    ax1.fill_between(t, 0, valid, step="pre", color="#2ca02c", alpha=0.7, label="Valid Reference (Trustworthy)")
    ax1.fill_between(t, valid, 1, step="pre", color="#d62728", alpha=0.5, label="Invalid / Large Gap (Untrustworthy)")
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Invalid", "Valid"])
    ax1.set_title("Ground Truth Reference Validity Mask Across Recording Timeline")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True)
    
    ax2.plot(t, acc, color="#1f77b4", label="GPS Horizontal Accuracy (m)")
    ax2.axhline(20.0, color="red", linestyle="--", label="Gating Threshold (20 m)")
    ax2.set_ylabel("Accuracy (m)")
    ax2.set_title("GNSS Reported Horizontal Accuracy (m)")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    
    ax3.plot(t, speed, color="#ff7f0e", label="Ground Truth Speed (m/s)")
    ax3.set_xlabel("Time Elapsed (seconds)")
    ax3.set_ylabel("Speed (m/s)")
    ax3.set_title("Vehicle Ground Truth Forward Speed Profile")
    ax3.legend(loc="upper right")
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_outage_examples(
    df_aligned: pd.DataFrame,
    scenarios: List[Dict[str, Any]],
    output_path: str = "figures/fig6_outage_examples.png",
) -> str:
    """Plot 6: Example 10-, 30-, and 60-second Simulated Outage Intervals."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sc_by_dur = {}
    for s in scenarios:
        dur = int(s["duration_s"])
        if dur not in sc_by_dur:
            sc_by_dur[dur] = s
            
    durs = sorted(sc_by_dur.keys())
    fig, axes = plt.subplots(len(durs), 1, figsize=(11, 2.8 * len(durs)), sharex=False)
    if len(durs) == 1:
        axes = [axes]
        
    for ax, dur in zip(axes, durs):
        sc = sc_by_dur[dur]
        s_idx, e_idx = sc["start_idx"], sc["end_idx"]
        pad = int(10 * 10)
        p_start = max(0, s_idx - pad)
        p_end = min(len(df_aligned), e_idx + pad)
        
        t_sub = df_aligned["time_s"].values[p_start:p_end]
        spd_sub = df_aligned["ref_speed_mps"].values[p_start:p_end]
        
        ax.plot(t_sub, spd_sub, color="#1f77b4", label="Reference Speed (m/s)")
        ax.axvspan(sc["start_time_s"], sc["end_time_s"], color="#d62728", alpha=0.25, label=f"Simulated GNSS Outage ({dur}s)")
        ax.axvline(sc["start_time_s"], color="red", linestyle="--")
        ax.axvline(sc["end_time_s"], color="red", linestyle="--")
        
        ax.set_title(f"{dur}-Second GNSS Outage Scenario [{sc['scenario_id']}] — Traveled: {sc['distance_traveled_m']:.1f} m, Mean Speed: {sc['mean_speed_mps']:.2f} m/s")
        ax.set_ylabel("Speed (m/s)")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True)
        
    axes[-1].set_xlabel("Time Elapsed (seconds)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_input_observation_reference_distinction(
    df_aligned: pd.DataFrame,
    scenario: Dict[str, Any],
    output_path: str = "figures/fig7_input_observation_reference_distinction.png",
) -> str:
    """Plot 7: Architectural Distinction Between Model Inputs, Masked GPS, and Reference Labels."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    s_idx = scenario["start_idx"]
    e_idx = scenario["end_idx"]
    pad = int(15 * 10)
    p_start = max(0, s_idx - pad)
    p_end = min(len(df_aligned), e_idx + pad)
    
    t_seg = df_aligned["time_s"].values[p_start:p_end]
    ax_seg = df_aligned["accel_x"].values[p_start:p_end]
    gz_seg = df_aligned["gyro_yaw"].values[p_start:p_end]
    
    raw_spd = df_aligned["raw_speed_mps"].values[p_start:p_end].copy()
    outage_mask_local = (t_seg >= scenario["start_time_s"]) & (t_seg <= scenario["end_time_s"])
    masked_gps_spd = raw_spd.copy()
    masked_gps_spd[outage_mask_local] = np.nan
    
    ref_east = df_aligned["ref_east_m"].values[p_start:p_end]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    
    ax1.plot(t_seg, ax_seg, label="Input: Accel X (m/s²)", color="#1f77b4", linewidth=1.2)
    ax1.plot(t_seg, gz_seg, label="Input: Gyro Yaw (rad/s)", color="#9467bd", linewidth=1.2)
    ax1.axvspan(scenario["start_time_s"], scenario["end_time_s"], color="gray", alpha=0.15, label="Outage Window (Inputs Intact)")
    ax1.set_ylabel("IMU Input Signals")
    ax1.set_title("PANEL 1: Model Continuous Inputs (6 IMU Channels Only — Active Throughout Outage)")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True)
    
    ax2.plot(t_seg, masked_gps_spd, "o-", color="#ff7f0e", markersize=4, label="Available GNSS Observations (Speed m/s)")
    ax2.axvspan(scenario["start_time_s"], scenario["end_time_s"], color="#d62728", alpha=0.25, label="OUTAGE: GPS Masked / Unavailable to Model")
    ax2.set_ylabel("GNSS Observation (m/s)")
    ax2.set_title("PANEL 2: GNSS Observation Stream (Zero Observation Supplied to Model During Outage)")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True)
    
    ax3.plot(t_seg, ref_east, color="#2ca02c", linewidth=2.0, label="Ground Truth Position Reference (ref_east_m)")
    ax3.axvspan(scenario["start_time_s"], scenario["end_time_s"], color="#2ca02c", alpha=0.12, label="Evaluation Span (Used Exclusively for Metric Computation)")
    ax3.set_xlabel("Time Elapsed (seconds)")
    ax3.set_ylabel("Reference East (m)")
    ax3.set_title("PANEL 3: Ground Truth Reference Labels (Strictly Isolated for Offline Evaluation)")
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_highrate_vs_interpolated(
    v_df: pd.DataFrame,
    s_df: pd.DataFrame,
    origin: Dict[str, float],
    output_path: str = "figures/fig8_highrate_groundtruth_vs_phone_interpolation.png",
    zoom_time_s: Tuple[float, float] = (140.0, 200.0),
) -> str:
    """Plot 8: High-Rate 10 Hz Vehicle Ground Truth vs 9s Phone Linear Interpolation."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    from src.geodesy import geodetic_to_enu
    
    v_cols = {c.strip(): c for c in v_df.columns}
    v_lat = v_df[v_cols["Latitude (degrees)"]].values.astype(float)
    v_lon = v_df[v_cols["Longitude (degrees)"]].values.astype(float)
    v_alt = v_df[v_cols["Height (km)"]].values.astype(float) * 1000.0
    
    ve, vn, vu = geodetic_to_enu(v_lat, v_lon, v_alt, origin["lat0"], origin["lon0"], origin["alt0"])
    
    time_s = s_df["time_s"].values
    se = s_df["ref_east_m"].values
    sn = s_df["ref_north_m"].values
    
    mask = (time_s >= zoom_time_s[0]) & (time_s <= zoom_time_s[1])
    t_z = time_s[mask]
    ve_z, vn_z = ve[mask], vn[mask]
    se_z, sn_z = se[mask], sn[mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # 2D Trajectory Zoom
    ax1.plot(ve_z, vn_z, "g-", linewidth=2.0, label="High-Rate 10 Hz Vehicle GPS (True Continuous Path)")
    ax1.plot(se_z, sn_z, "r--", linewidth=1.5, marker="x", markevery=10, label="Smartphone ~9s Interpolated GPS (Chord Cuts)")
    ax1.set_xlabel("Local East Position (meters)")
    ax1.set_ylabel("Local North Position (meters)")
    ax1.set_title(f"2D Cornering Trajectory [{zoom_time_s[0]:.0f}s - {zoom_time_s[1]:.0f}s]\n(Illustrating Linear Chord Truncation on Curves)")
    ax1.axis("equal")
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(True)
    
    # Position Error vs Time
    err_2d = np.sqrt((se_z - ve_z)**2 + (sn_z - vn_z)**2)
    ax2.plot(t_z, err_2d, color="#d62728", linewidth=1.5, label="2D Position Discrepancy (m)")
    ax2.set_xlabel("Time Elapsed (seconds)")
    ax2.set_ylabel("Discrepancy (meters)")
    ax2.set_title("Instantaneous Discrepancy (Phone Interp vs Vehicle Ground Truth)")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
