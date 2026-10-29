"""
Phase 2 Timing Audit Script.
Analyzes timestamps, sampling cadences, GPS fix intervals, and vehicle ground truth
across all 6 recordings in the IO-VNBD dataset.
"""

import os
import sys
import json
from pathlib import Path

# Add project root
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set publication style
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


def analyze_phone_recording(csv_path: str):
    print(f"Analyzing Phone: {Path(csv_path).name}...", flush=True)
    df = pd.read_csv(csv_path, encoding="latin1")
    cols = {c.strip(): c for c in df.columns}
    
    n_rows = len(df)
    
    # 1. IMU Timestamps
    t_ms = df[cols["TIME SINCE START (ms)"]].values.astype(float)
    dt_ms = np.diff(t_ms)
    dt_s = dt_ms / 1000.0
    
    resets = np.where(dt_ms < 0)[0]
    zero_dts = np.where(dt_ms == 0)[0]
    
    # 2. Wall clock timestamps if available
    date_col = cols.get("DATE (YYYY-MO-DD HH-MI-SS_SSS)", None)
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col].str.strip(), format="%Y-%m-%d %H-%M-%S_%f", errors="coerce")
            wall_dt_s = dates.diff().dt.total_seconds().values[1:]
        except Exception:
            wall_dt_s = None
    else:
        wall_dt_s = None
        
    # 3. Phone GPS Fixes
    raw_lat = df[cols["GPS LATITUDE (degrees)"]].values.astype(float)
    raw_lon = df[cols["GPS LONGITUDE (degrees)"]].values.astype(float)
    raw_acc = df[cols["GPS ACCURACY (m)"]].values.astype(float)
    raw_spd = df[cols["GPS SPEED (Kmh)"]].values.astype(float)
    
    # Genuine coordinate change detection
    gps_change = np.concatenate([[True], (np.diff(raw_lat) != 0) | (np.diff(raw_lon) != 0)])
    fix_indices = np.where(gps_change)[0]
    
    # Times of fixes based on row index * nominal 0.1s and based on t_ms
    # To handle t_ms resets, compute continuous time
    dt_s_clipped = np.where(dt_s <= 0, 0.1, dt_s)
    cont_time_s = np.concatenate([[0.0], np.cumsum(dt_s_clipped)])
    
    fix_times_s = cont_time_s[fix_indices]
    inter_fix_dt_s = np.diff(fix_times_s)
    
    # Valid fixes
    valid_mask = (raw_lat >= -90) & (raw_lat <= 90) & (raw_lon >= -180) & (raw_lon <= 180) & ~((raw_lat == 0) & (raw_lon == 0)) & (raw_acc <= 20.0)
    valid_fix_indices = fix_indices[valid_mask[fix_indices]]
    valid_fix_times = cont_time_s[valid_fix_indices]
    valid_inter_fix_dt_s = np.diff(valid_fix_times) if len(valid_fix_times) > 1 else np.array([])
    
    return {
        "drive": Path(csv_path).name,
        "total_rows": n_rows,
        "duration_s": float(cont_time_s[-1]),
        "imu_dt_s": dt_s,
        "imu_dt_stats": {
            "mean": float(np.mean(dt_s)),
            "median": float(np.median(dt_s)),
            "std": float(np.std(dt_s)),
            "min": float(np.min(dt_s)),
            "max": float(np.max(dt_s)),
            "resets_count": int(len(resets)),
            "resets_indices": resets.tolist(),
            "zero_dt_count": int(len(zero_dts)),
        },
        "gps_fixes": {
            "total_changes": int(len(fix_indices)),
            "valid_fixes": int(len(valid_fix_indices)),
            "inter_fix_dt_s": inter_fix_dt_s,
            "valid_inter_fix_dt_s": valid_inter_fix_dt_s,
            "mean_inter_fix_s": float(np.mean(inter_fix_dt_s)) if len(inter_fix_dt_s) > 0 else 0.0,
            "median_inter_fix_s": float(np.median(inter_fix_dt_s)) if len(inter_fix_dt_s) > 0 else 0.0,
            "min_inter_fix_s": float(np.min(inter_fix_dt_s)) if len(inter_fix_dt_s) > 0 else 0.0,
            "max_inter_fix_s": float(np.max(inter_fix_dt_s)) if len(inter_fix_dt_s) > 0 else 0.0,
            "effective_rate_hz": float(len(fix_indices) / max(cont_time_s[-1], 1.0)),
        }
    }


def analyze_vehicle_recording(csv_path: str):
    print(f"Analyzing Vehicle: {Path(csv_path).name}...", flush=True)
    df = pd.read_csv(csv_path, encoding="latin1")
    cols = {c.strip(): c for c in df.columns}
    
    n_rows = len(df)
    
    t_sec = df[cols["Time Since Start of Day (seconds)"]].values.astype(float)
    dt_v = np.diff(t_sec)
    
    sp_col = cols.get("Sample period (seconds)", None)
    sp_vals = df[sp_col].values.astype(float) if sp_col else np.full(n_rows, 0.1)
    
    spd_kmh = df[cols["Velocity (km/hr)"]].values.astype(float) if "Velocity (km/hr)" in cols else np.zeros(n_rows)
    heading = df[cols["Heading (degrees)"]].values.astype(float) if "Heading (degrees)" in cols else np.zeros(n_rows)
    lat = df[cols["Latitude (degrees)"]].values.astype(float)
    lon = df[cols["Longitude (degrees)"]].values.astype(float)
    
    # Coordinate change detection in vehicle data
    v_change = np.concatenate([[True], (np.diff(lat) != 0) | (np.diff(lon) != 0)])
    
    return {
        "drive": Path(csv_path).name,
        "total_rows": n_rows,
        "duration_s": float(t_sec[-1] - t_sec[0]),
        "dt_v": dt_v,
        "dt_v_stats": {
            "mean": float(np.mean(dt_v)),
            "median": float(np.median(dt_v)),
            "std": float(np.std(dt_v)),
            "min": float(np.min(dt_v)),
            "max": float(np.max(dt_v)),
            "monotonic": bool(np.all(dt_v > 0)),
        },
        "sample_period_stats": {
            "mean": float(np.mean(sp_vals)),
            "median": float(np.median(sp_vals)),
            "min": float(np.min(sp_vals)),
            "max": float(np.max(sp_vals)),
        },
        "gps_update_frequency_hz": float(np.sum(v_change) / max((t_sec[-1] - t_sec[0]), 1.0)),
        "speed_kmh_stats": {
            "mean": float(np.mean(spd_kmh)),
            "max": float(np.max(spd_kmh)),
            "std": float(np.std(spd_kmh)),
        }
    }


def main():
    raw_dir = Path("data/raw")
    drives = ["S1", "S2", "S3a", "S3b", "S3c", "S4"]
    
    phone_results = {}
    vehicle_results = {}
    summary_rows = []
    
    for d in drives:
        s_path = raw_dir / f"S-{d}.csv"
        v_path = raw_dir / f"V-{d}.csv"
        
        pr = analyze_phone_recording(str(s_path))
        vr = analyze_vehicle_recording(str(v_path))
        
        phone_results[f"S-{d}"] = pr
        vehicle_results[f"V-{d}"] = vr
        
        summary_rows.append({
            "Drive": d,
            "Phone Rows": pr["total_rows"],
            "Vehicle Rows": vr["total_rows"],
            "Rows Match": pr["total_rows"] == vr["total_rows"],
            "Duration (s)": round(pr["duration_s"], 1),
            "Phone IMU Mean dt (ms)": round(pr["imu_dt_stats"]["mean"] * 1000, 2),
            "Phone IMU Median dt (ms)": round(pr["imu_dt_stats"]["median"] * 1000, 2),
            "Phone IMU Resets": pr["imu_dt_stats"]["resets_count"],
            "Phone GPS Fixes": pr["gps_fixes"]["total_changes"],
            "Phone GPS Mean Inter-Fix (s)": round(pr["gps_fixes"]["mean_inter_fix_s"], 2),
            "Phone GPS Median Inter-Fix (s)": round(pr["gps_fixes"]["median_inter_fix_s"], 2),
            "Phone GPS Effective Rate (Hz)": round(pr["gps_fixes"]["effective_rate_hz"], 3),
            "Vehicle dt Mean (s)": round(vr["dt_v_stats"]["mean"], 3),
            "Vehicle Strictly Monotonic": vr["dt_v_stats"]["monotonic"],
            "Vehicle Max Speed (km/h)": round(vr["speed_kmh_stats"]["max"], 1),
            "Vehicle Mean Speed (km/h)": round(vr["speed_kmh_stats"]["mean"], 1),
        })
        
    # Save CSV and JSON summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("reports/timing_statistics.csv", index=False)
    print("\nSaved reports/timing_statistics.csv")
    
    clean_json = {
        "summary": summary_rows,
        "phone_details": {k: {
            "drive": v["drive"],
            "total_rows": v["total_rows"],
            "duration_s": v["duration_s"],
            "imu_dt_stats": v["imu_dt_stats"],
            "gps_fixes_summary": {
                "total_changes": v["gps_fixes"]["total_changes"],
                "valid_fixes": v["gps_fixes"]["valid_fixes"],
                "mean_inter_fix_s": v["gps_fixes"]["mean_inter_fix_s"],
                "median_inter_fix_s": v["gps_fixes"]["median_inter_fix_s"],
                "min_inter_fix_s": v["gps_fixes"]["min_inter_fix_s"],
                "max_inter_fix_s": v["gps_fixes"]["max_inter_fix_s"],
                "effective_rate_hz": v["gps_fixes"]["effective_rate_hz"],
            }
        } for k, v in phone_results.items()},
        "vehicle_details": {k: {
            "drive": v["drive"],
            "total_rows": v["total_rows"],
            "duration_s": v["duration_s"],
            "dt_v_stats": v["dt_v_stats"],
            "sample_period_stats": v["sample_period_stats"],
            "gps_update_frequency_hz": v["gps_update_frequency_hz"],
            "speed_kmh_stats": v["speed_kmh_stats"],
        } for k, v in vehicle_results.items()}
    }
    
    with open("reports/timing_statistics.json", "w") as f:
        json.dump(clean_json, f, indent=2)
    print("Saved reports/timing_statistics.json")
    
    # -------------------------------------------------------------------------
    # GENERATE PUBLICATION FIGURES
    # -------------------------------------------------------------------------
    os.makedirs("figures", exist_ok=True)
    
    # Figure 1: GPS Inter-Fix Distributions across drives
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = axes.flatten()
    
    for i, d in enumerate(drives):
        pr = phone_results[f"S-{d}"]
        dt_arr = pr["gps_fixes"]["inter_fix_dt_s"]
        
        # Filter extreme outliers for visualization
        dt_filtered = dt_arr[dt_arr <= 30.0]
        
        ax = axes[i]
        ax.hist(dt_filtered, bins=30, color="#1f77b4", edgecolor="black", alpha=0.7)
        ax.set_title(f"Drive {d} (N={len(dt_arr)} fixes)")
        ax.set_xlabel("Inter-Fix Interval $\Delta t_{GPS}$ (seconds)")
        ax.set_ylabel("Count" if i % 3 == 0 else "")
        ax.axvline(np.median(dt_arr), color="red", linestyle="--", linewidth=1.5, label=f"Median: {np.median(dt_arr):.2f}s")
        ax.axvline(np.mean(dt_arr), color="green", linestyle=":", linewidth=1.5, label=f"Mean: {np.mean(dt_arr):.2f}s")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.suptitle("Empirical Phone GPS Inter-Fix Interval Distributions Across IO-VNBD Drives", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/gps_inter_fix_distributions.png")
    plt.close()
    print("Saved figures/gps_inter_fix_distributions.png")
    
    # Figure 2: IMU Sampling Interval Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, d in enumerate(drives):
        pr = phone_results[f"S-{d}"]
        dt_ms = pr["imu_dt_s"] * 1000.0
        # Exclude resets and zoom into [0, 200 ms]
        dt_normal = dt_ms[(dt_ms > 0) & (dt_ms < 250)]
        
        ax = axes[i]
        ax.hist(dt_normal, bins=40, color="#2ca02c", edgecolor="black", alpha=0.7)
        ax.set_title(f"Drive {d} IMU $\Delta t$")
        ax.set_xlabel("IMU Sampling Interval (ms)")
        ax.set_ylabel("Frequency" if i % 3 == 0 else "")
        ax.axvline(100.0, color="darkorange", linestyle="-", linewidth=1.5, label="Nominal 100 ms (10 Hz)")
        ax.axvline(np.median(dt_normal), color="red", linestyle="--", linewidth=1.5, label=f"Median: {np.median(dt_normal):.1f} ms")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.suptitle("Phone IMU Sampling Interval Distributions ($\Delta t$) Across IO-VNBD Drives", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/imu_sampling_interval_distributions.png")
    plt.close()
    print("Saved figures/imu_sampling_interval_distributions.png")
    
    # Figure 3: Comparison of Phone vs Vehicle Sampling Cadence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot A: Phone GPS vs Vehicle GPS update frequency
    drive_names = [f"Drive {d}" for d in drives]
    phone_rates = [phone_results[f"S-{d}"]["gps_fixes"]["effective_rate_hz"] for d in drives]
    vehicle_rates = [vehicle_results[f"V-{d}"]["gps_update_frequency_hz"] for d in drives]
    
    x = np.arange(len(drives))
    width = 0.35
    
    axes[0].bar(x - width/2, phone_rates, width, label="Phone GPS Effective Rate", color="#e74c3c")
    axes[0].bar(x + width/2, vehicle_rates, width, label="Vehicle Ground Truth Rate", color="#2ecc71")
    axes[0].set_ylabel("Sampling / Fix Rate (Hz)")
    axes[0].set_title("GPS Update Rate: Phone vs. Vehicle Ground Truth")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(drive_names)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot B: Median Phone Inter-fix duration
    phone_medians = [phone_results[f"S-{d}"]["gps_fixes"]["median_inter_fix_s"] for d in drives]
    axes[1].bar(drive_names, phone_medians, color="#3498db", edgecolor="black")
    axes[1].axhline(9.0, color="red", linestyle="--", linewidth=1.5, label="Nominal ~9.0s Line")
    axes[1].set_ylabel("Median Inter-Fix Interval (seconds)")
    axes[1].set_title("Phone GPS Median Inter-Fix Interval (s)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("Comparison of Phone Internal GPS vs Vehicle Ground Truth Cadence", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/phone_vs_vehicle_timing_comparison.png")
    plt.close()
    print("Saved figures/phone_vs_vehicle_timing_comparison.png")
    print("All Phase 2 timing audits completed successfully!")


if __name__ == "__main__":
    main()
