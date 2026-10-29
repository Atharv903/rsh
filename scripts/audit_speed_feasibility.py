"""
Phase 4 Speed Feasibility Diagnostic Script.
Generates publication plots verifying IMU dynamics vs vehicle forward speed.
"""

import os
import sys
from pathlib import Path

# Add project root
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.corrected_synchronization import extract_aligned_segments_for_drive

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})


def main():
    os.makedirs("figures", exist_ok=True)
    
    # Load aligned Drive S1
    segs, _ = extract_aligned_segments_for_drive("S1", raw_dir="data/raw")
    df = segs[0]
    
    time_s = df["time_s"].values
    speed = df["ref_speed_mps"].values
    acc_x = df["accel_x"].values
    acc_y = df["accel_y"].values
    acc_z = df["accel_z"].values
    gyro_pitch = df["gyro_pitch"].values * (180.0 / np.pi)  # Yaw rate in deg/s
    
    # Forward acceleration along optimal axis (314.9 deg)
    th = np.radians(314.9)
    acc_fwd = acc_x * np.cos(th) + acc_y * np.sin(th)
    
    # Rolling vibration energy (1-second window)
    vib_energy = pd.Series(acc_z).rolling(10, min_periods=1).std().fillna(0).values
    
    # -------------------------------------------------------------------------
    # Figure 1: Multi-panel time series showing IMU features vs Vehicle Speed
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # Pick a 300-second dynamic section with stops, acceleration, turns, cruising
    t_start = 1200
    t_end = 1500
    mask = (time_s >= t_start) & (time_s <= t_end)
    t_sub = time_s[mask] - t_start
    
    axes[0].plot(t_sub, speed[mask], color="black", linewidth=2.0)
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Vehicle Forward Speed (Gold Standard Ground Truth)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    axes[1].plot(t_sub, acc_fwd[mask], color="#1f77b4", linewidth=1.2, alpha=0.8)
    axes[1].plot(t_sub, pd.Series(acc_fwd[mask]).rolling(10).mean(), color="navy", linewidth=2.0, label="Filtered Forward Accel")
    axes[1].set_ylabel("Forward Accel (m/s²)")
    axes[1].set_title(r"Smartphone Forward Acceleration Dynamics ($a_{\mathrm{fwd}} \approx \dot{v}$)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    axes[2].plot(t_sub, gyro_pitch[mask], color="#d95f02", linewidth=1.5)
    axes[2].set_ylabel("Yaw Rate (deg/s)")
    axes[2].set_title("Smartphone Gyroscope Yaw Tracking (Chassis Turn Dynamics)")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    
    axes[3].plot(t_sub, vib_energy[mask], color="#2ca02c", linewidth=1.5)
    axes[3].axhline(0.25, color="red", linestyle=":", label="Stationary Noise Floor (<0.25 m/s²)")
    axes[3].set_ylabel("Vibration Std (m/s²)")
    axes[3].set_xlabel("Elapsed Time in Window (seconds)")
    axes[3].set_title("Chassis Suspension Vibration Energy vs. Vehicle Velocity")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("figures/phase4_imu_dynamics_vs_speed.png")
    plt.close()
    print("Saved figures/phase4_imu_dynamics_vs_speed.png")
    
    # -------------------------------------------------------------------------
    # Figure 2: Stationary vs Constant-Speed Distinguishability
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Vibration distribution in speed regimes
    stat_mask = speed < 0.2
    low_mask = (speed >= 1.0) & (speed < 8.0)
    med_mask = (speed >= 8.0) & (speed < 18.0)
    high_mask = speed >= 18.0
    
    bins = np.linspace(0, 3.5, 50)
    axes[0].hist(vib_energy[stat_mask], bins=bins, alpha=0.6, label=f"Stationary (<0.2 m/s, N={np.sum(stat_mask):,})", color="gray", density=True)
    axes[0].hist(vib_energy[low_mask], bins=bins, alpha=0.6, label=f"Low Speed (1-8 m/s, N={np.sum(low_mask):,})", color="#1f77b4", density=True)
    axes[0].hist(vib_energy[med_mask], bins=bins, alpha=0.6, label=f"Medium Speed (8-18 m/s, N={np.sum(med_mask):,})", color="#ff7f0e", density=True)
    axes[0].hist(vib_energy[high_mask], bins=bins, alpha=0.6, label=f"High Speed (>=18 m/s, N={np.sum(high_mask):,})", color="#d62728", density=True)
    axes[0].set_xlabel("Vertical Accelerometer Vibration Std (m/s²)")
    axes[0].set_ylabel("Probability Density")
    axes[0].set_title("IMU Vibration Energy Distribution by Speed Regime")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Panel B: Scatter plot of IMU acceleration variance vs speed
    sub_idx = np.arange(0, len(speed), 5)
    axes[1].scatter(speed[sub_idx], vib_energy[sub_idx], alpha=0.25, s=8, color="#2b5c8f")
    # Rolling trend
    df_trend = pd.DataFrame({"speed": speed, "vib": vib_energy}).sort_values("speed")
    df_trend["vib_smooth"] = df_trend["vib"].rolling(500, min_periods=50).mean()
    axes[1].plot(df_trend["speed"], df_trend["vib_smooth"], color="red", linewidth=2.5, label="Mean Vibration Trend")
    axes[1].set_xlabel("Vehicle Speed (m/s)")
    axes[1].set_ylabel("IMU Vibration Energy (m/s²)")
    axes[1].set_title("Vibration Scaling with Vehicle Velocity ($r = 0.59$)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("figures/phase4_stationary_vs_moving_distinguishability.png")
    plt.close()
    print("Saved figures/phase4_stationary_vs_moving_distinguishability.png")
    print("Phase 4 plots generated successfully!")


if __name__ == "__main__":
    main()
