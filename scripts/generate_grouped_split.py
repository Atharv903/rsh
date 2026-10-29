"""
Script to construct the Grouped Session-Isolated IO-VNBD Dataset.
Completely eliminates route and temporal leakage by grouping all fragments
of the same underlying trip into the same partition:
- Train: Session 3 (Drives S1 & S2, Sept 8, 2019)
- Val:   Session 1 (Drives S3a, S3b, S3c, Sept 4, 2019)
- Test:  Session 2 (Drive S4, Sept 6, 2019)
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


def main():
    print("=" * 70)
    print("GENERATING LEAKAGE-FREE GROUPED SESSION PARTITIONS")
    print("=" * 70)
    
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Session Grouping:
    # Train: S1, S2 (Sept 8, 2019 trip)
    # Val:   S3a, S3b, S3c (Sept 4, 2019 trip)
    # Test:  S4 (Sept 6, 2019 trip)
    grouped_partitions = {
        "train": ["S1", "S2"],
        "val": ["S3a", "S3b", "S3c"],
        "test": ["S4"],
    }
    
    all_drives = ["S1", "S2", "S3a", "S3b", "S3c", "S4"]
    drive_segments = {}
    
    for d in all_drives:
        segs, _ = extract_aligned_segments_for_drive(d, raw_dir=raw_dir)
        drive_segments[d] = segs
        
    # 1. Fit SensorNormalizer STRICTLY on Train Drives (S1 & S2)
    print("\n--- Fitting SensorNormalizer strictly on Train Session (S1 & S2) ---")
    train_dfs = []
    for d in grouped_partitions["train"]:
        train_dfs.extend(drive_segments[d])
        
    combined_train = pd.concat(train_dfs, ignore_index=True)
    feature_channels = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
    
    normalizer = SensorNormalizer(method="standard", channels=feature_channels)
    normalizer.fit(combined_train)
    
    norm_path = os.path.join(processed_dir, "sensor_normalizer_grouped.json")
    normalizer.save(norm_path)
    print(f"Normalizer fitted on {len(combined_train):,} training samples. Saved to: {norm_path}")
    
    # 2. Apply Normalization to all segments
    for d in all_drives:
        for i, seg in enumerate(drive_segments[d]):
            drive_segments[d][i] = normalizer.transform(seg)
            
    # 3. Generate Causal Sliding Windows
    print("\n--- Generating Causal Sliding Windows for Grouped Partitions ---")
    partition_counts = {}
    
    for split_name, drives in grouped_partitions.items():
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
                    
        X_split = np.concatenate(x_list, axis=0)
        y_spd_split = np.concatenate(y_spd_list, axis=0)
        y_vel_split = np.concatenate(y_vel_list, axis=0)
        
        partition_counts[split_name] = len(X_split)
        
        out_npz = os.path.join(processed_dir, f"{split_name}_windows_grouped.npz")
        np.savez_compressed(
            out_npz,
            X=X_split,
            y=y_spd_split,
            y_vel_enu=y_vel_split,
            meta=np.array(meta_list, dtype=object),
        )
        print(f"  Split '{split_name:5s}': {len(X_split):,d} windows -> {out_npz}")
        print(f"    Speed stats (m/s): min={y_spd_split.min():.2f}, mean={y_spd_split.mean():.2f}, max={y_spd_split.max():.2f}")
        
    # 4. Save Grouped Split Manifest
    manifest = {
        "split_type": "grouped_session_isolated_partition",
        "description": "Zero-leakage partition grouping all fragments of a single journey into the same split.",
        "primary_target": "scalar_forward_speed_mps",
        "splits": {
            "train": {
                "session": "Sept 8, 2019 (Coventry-Warwickshire)",
                "drives": grouped_partitions["train"],
                "total_windows": int(partition_counts["train"]),
            },
            "val": {
                "session": "Sept 4, 2019 (Coventry-Rugby Round Trip)",
                "drives": grouped_partitions["val"],
                "total_windows": int(partition_counts["val"]),
            },
            "test": {
                "session": "Sept 6, 2019 (Coventry Urban-Highway Loop)",
                "drives": grouped_partitions["test"],
                "total_windows": int(partition_counts["test"]),
            },
        },
    }
    
    manifest_path = "data/splits/drive_split_manifest_grouped.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved grouped split manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
