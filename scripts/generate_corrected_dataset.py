"""
Script to build and validate the corrected IO-VNBD dataset.
Uses segment-aware UTC synchronization, fits normalizer on training split only,
generates causal windows, and produces diagnostic plots and audit reports.
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

from src.corrected_synchronization import (
    extract_aligned_segments_for_drive,
    generate_causal_sliding_windows,
)
from src.signal_processing import SensorNormalizer

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
    print("=" * 70)
    print("BUILDING AND VALIDATING CORRECTED IO-VNBD DATASET")
    print("=" * 70)
    
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Drive Partitions
    # Train: S1, S2, S4
    # Val: S3a
    # Test: S3b, S3c
    partitions = {
        "train": ["S1", "S2", "S4"],
        "val": ["S3a"],
        "test": ["S3b", "S3c"],
    }
    
    all_drives = ["S1", "S2", "S3a", "S3b", "S3c", "S4"]
    drive_segments = {}
    drive_sync_meta = {}
    
    for d in all_drives:
        segs, meta = extract_aligned_segments_for_drive(d, raw_dir=raw_dir)
        drive_segments[d] = segs
        drive_sync_meta[d] = meta
        print(f"Drive {d:4s}: Extracted {len(segs)} valid continuous segment(s). Total samples: {sum(len(s) for s in segs):,}")
        
    # 2. Fit Normalizer STRICTLY on Train Segments
    print("\n--- Fitting SensorNormalizer on Training Split Only ---")
    train_dfs = []
    for d in partitions["train"]:
        train_dfs.extend(drive_segments[d])
        
    combined_train = pd.concat(train_dfs, ignore_index=True)
    feature_channels = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
    
    normalizer = SensorNormalizer(method="standard", channels=feature_channels)
    normalizer.fit(combined_train)
    
    norm_path = os.path.join(processed_dir, "sensor_normalizer_corrected.json")
    normalizer.save(norm_path)
    print(f"Normalizer fitted on {len(combined_train):,} training samples. Saved to: {norm_path}")
    
    # 3. Apply Normalization to all segments
    for d in all_drives:
        for i, seg in enumerate(drive_segments[d]):
            drive_segments[d][i] = normalizer.transform(seg)
            
    # 4. Generate Causal Sliding Windows (W=100, Stride=10)
    print("\n--- Generating Causal Sliding Windows (W=100, Stride=10) ---")
    partition_datasets = {}
    partition_counts = {}
    
    for split_name, drives in partitions.items():
        x_list, y_spd_list, y_vel_list, meta_list = [], [], [], []
        
        for d in drives:
            for seg in drive_segments[d]:
                X_seg, y_spd_seg, y_vel_seg, meta_seg = generate_causal_sliding_windows(
                    seg,
                    window_size=100,
                    stride=10,
                    feature_channels=feature_channels,
                )
                if len(X_seg) > 0:
                    x_list.append(X_seg)
                    y_spd_list.append(y_spd_seg)
                    y_vel_list.append(y_vel_seg)
                    meta_list.extend(meta_seg)
                    
        X_split = np.concatenate(x_list, axis=0) if x_list else np.empty((0, 100, 6), dtype=np.float32)
        y_spd_split = np.concatenate(y_spd_list, axis=0) if y_spd_list else np.empty((0, 1), dtype=np.float32)
        y_vel_split = np.concatenate(y_vel_list, axis=0) if y_vel_list else np.empty((0, 2), dtype=np.float32)
        
        partition_datasets[split_name] = {
            "X": X_split,
            "y_speed": y_spd_split,
            "y_vel_enu": y_vel_split,
            "meta": meta_list,
        }
        partition_counts[split_name] = len(X_split)
        
        # Save NPZ archive (corrected, non-destructive to legacy files)
        out_npz = os.path.join(processed_dir, f"{split_name}_windows_corrected.npz")
        np.savez_compressed(
            out_npz,
            X=X_split,
            y=y_spd_split,       # Primary target: scalar forward speed (m/s)
            y_vel_enu=y_vel_split, # Secondary target: [v_east, v_north] (m/s)
            meta=np.array(meta_list, dtype=object),
        )
        print(f"  Split '{split_name:5s}': {len(X_split):,d} windows saved to {out_npz}")
        print(f"    Speed stats (m/s): min={y_spd_split.min():.2f}, mean={y_spd_split.mean():.2f}, max={y_spd_split.max():.2f}")
        
    # 5. Save Corrected Split Manifest
    manifest = {
        "split_type": "corrected_drive_level_session_partition",
        "description": "Segment-aware UTC synchronized dataset with verified continuous vehicle telemetry reference.",
        "primary_target": "scalar_forward_speed_mps",
        "secondary_target": "horizontal_velocity_enu_mps",
        "window_size_samples": 100,
        "stride_samples": 10,
        "sample_rate_hz": 10.0,
        "splits": {
            k: {
                "drives": partitions[k],
                "total_windows": partition_counts[k],
            } for k in partitions
        },
        "drive_sync_metadata": drive_sync_meta,
    }
    
    # Convert numpy types in manifest for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    manifest_path = "data/splits/drive_split_manifest_corrected.json"
    with open(manifest_path, "w") as f:
        json.dump(convert_types(manifest), f, indent=2)
    print(f"\nSaved corrected split manifest to: {manifest_path}")
    
    # 6. Generate Diagnostic and Comparison Plots
    print("\n--- Generating Publication Diagnostic Figures ---")
    
    # Figure 1: Speed Distribution Across Splits
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for idx, (s_name, c) in enumerate(zip(["train", "val", "test"], colors)):
        speeds = partition_datasets[s_name]["y_speed"].flatten()
        ax = axes[idx]
        ax.hist(speeds, bins=35, color=c, edgecolor="black", alpha=0.7)
        ax.set_title(f"{s_name.capitalize()} Split (N={len(speeds):,})")
        ax.set_xlabel("Window-Average Speed (m/s)")
        if idx == 0:
            ax.set_ylabel("Frequency (Windows)")
        ax.axvline(np.mean(speeds), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(speeds):.1f} m/s")
        ax.axvline(np.median(speeds), color="darkgreen", linestyle=":", linewidth=1.5, label=f"Median: {np.median(speeds):.1f} m/s")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.suptitle("Target Speed Distribution Across Verified Train / Val / Test Partitions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/phase3_speed_distribution_by_split.png")
    plt.close()
    print("Saved figures/phase3_speed_distribution_by_split.png")
    
    # Figure 2: Phone GPS Speed vs. Vehicle Telemetry Speed Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot on Drive S1
    s1_seg = drive_segments["S1"][0]
    # Find discrete phone GPS fixes
    phone_raw = pd.read_csv("data/raw/S-S1.csv", encoding="latin1")
    s_cols = {col.strip(): col for col in phone_raw.columns}
    raw_p_spd = phone_raw[s_cols["GPS SPEED (Kmh)"]].values / 3.6
    
    # Aligned vehicle speed
    v_spd = s1_seg["ref_speed_mps"].values
    
    axes[0].plot(s1_seg["time_s"].values[:1000], v_spd[:1000], label="Vehicle Telemetry (10 Hz Continuous)", color="black", linewidth=2.0)
    axes[0].plot(s1_seg["time_s"].values[:1000], raw_p_spd[5:1005], label="Phone GPS Speed (Discrete 9s Fixes)", color="#e74c3c", linestyle="--", alpha=0.8)
    axes[0].set_title("Drive S1: 10 Hz Continuous Telemetry vs. Phone GPS Speed")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Correlation scatter
    axes[1].scatter(v_spd[::10], raw_p_spd[5:len(v_spd)+5:10], alpha=0.2, s=8, color="#2b5c8f")
    axes[1].plot([0, 30], [0, 30], color="red", linestyle="--", label="Ideal 1:1 Line")
    axes[1].set_title("Phone GPS Speed vs. Vehicle Telemetry Speed")
    axes[1].set_xlabel("Vehicle Telemetry Speed (m/s)")
    axes[1].set_ylabel("Phone GPS Speed (m/s)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("Validation of Vehicle Telemetry Reference vs Phone GNSS", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/phase3_vehicle_vs_phone_speed_validation.png")
    plt.close()
    print("Saved figures/phase3_vehicle_vs_phone_speed_validation.png")
    
    print("\nAll corrected dataset generation tasks completed successfully!")


if __name__ == "__main__":
    main()
