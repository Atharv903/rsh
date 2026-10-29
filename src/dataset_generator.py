"""
Sliding Window Generation, Dataset Partitioning, and Manifest Management Module.

Prepares tensor-ready time-series windows for deep learning models (CNN-GRU, ModernTCN, Mamba),
enforces drive-level train/val/test splitting to prevent temporal autocorrelation leakage,
and logs complete experiment metadata.
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


def generate_sliding_windows(
    df: pd.DataFrame,
    drive_id: str,
    feature_channels: List[str] = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"],
    window_size: int = 100,  # 10.0 seconds at 10 Hz
    stride: int = 10,        # 1.0 second stride
    min_valid_ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Slice continuous recording into model-ready time-series windows.
    
    Args:
        df: Aligned and normalized recording DataFrame.
        drive_id: Identifier for the drive recording.
        feature_channels: 6 IMU input channel names.
        window_size: Window length in samples.
        stride: Step size between consecutive windows.
        min_valid_ratio: Required proportion of valid reference samples in window.
        
    Returns:
        X: Shape (N, window_size, 6) feature tensor.
        Y: Shape (N, 2) 2D horizontal displacement target [delta_east, delta_north] in meters.
        metadata: List of window provenance metadata dictionaries.
    """
    total_len = len(df)
    features = df[feature_channels].values.astype(np.float32)
    time_s = df["time_s"].values.astype(np.float64)
    east = df["ref_east_m"].values.astype(np.float64)
    north = df["ref_north_m"].values.astype(np.float64)
    valid_mask = df["valid_reference"].values
    
    x_list = []
    y_list = []
    meta_list = []
    
    for start_idx in range(0, total_len - window_size + 1, stride):
        end_idx = start_idx + window_size
        
        # Check reference validity across window
        seg_valid = valid_mask[start_idx:end_idx]
        val_ratio = float(np.mean(seg_valid))
        if val_ratio < min_valid_ratio:
            continue
            
        x_window = features[start_idx:end_idx]
        # Target: total 2D displacement over the window [delta_east, delta_north]
        d_east = east[end_idx - 1] - east[start_idx]
        d_north = north[end_idx - 1] - north[start_idx]
        y_target = np.array([d_east, d_north], dtype=np.float32)
        
        x_list.append(x_window)
        y_list.append(y_target)
        meta_list.append({
            "drive_id": drive_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time_s": float(time_s[start_idx]),
            "end_time_s": float(time_s[end_idx - 1]),
            "valid_ratio": val_ratio,
        })
        
    if not x_list:
        return np.empty((0, window_size, len(feature_channels)), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []
        
    X = np.stack(x_list, axis=0)
    Y = np.stack(y_list, axis=0)
    return X, Y, meta_list


def create_split_manifests(
    all_drives: List[str],
    drive_stats: Dict[str, Dict[str, Any]],
    output_dir: str = "data/splits",
    primary_drive_only: bool = False,
    multi_drive_config: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Create drive-level split manifests avoiding window boundary leakage.
    
    Args:
        all_drives: List of drive file names.
        drive_stats: Dict containing window and sample counts for each drive.
        output_dir: Directory where manifests are saved.
        primary_drive_only: Flag indicating whether only S-S1 is processed.
        multi_drive_config: Config specifying train, val, test drive assignments.
        
    Returns:
        Dict summarizing split allocations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if primary_drive_only or len(all_drives) == 1:
        # Single-drive smoke test manifest
        smoke_manifest = {
            "split_type": "single_drive_smoke_test",
            "is_generalization_benchmark": False,
            "status": "PIPELINE_SMOKE_TEST_ONLY",
            "warning": (
                "Row-level or window-level random splitting of a single inertial recording "
                "causes massive temporal autocorrelation leakage. This manifest is strictly "
                "for software pipeline verification and smoke testing."
            ),
            "drives": {
                "train": all_drives[:1],
                "val": all_drives[:1],
                "test": all_drives[:1],
            },
            "drive_statistics": drive_stats,
        }
        manifest_path = os.path.join(output_dir, "smoke_test_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(smoke_manifest, f, indent=2)
        return smoke_manifest
        
    # Multi-drive rigorous partitioning
    if multi_drive_config is not None:
        train_drives = multi_drive_config.get("train", [])
        val_drives = multi_drive_config.get("val", [])
        test_drives = multi_drive_config.get("test", [])
    else:
        # Default partition
        train_drives = [d for d in all_drives if "S1" in d or "S2" in d or "S4" in d]
        val_drives = [d for d in all_drives if "S3a" in d]
        test_drives = [d for d in all_drives if "S3b" in d or "S3c" in d]
        
    def aggregate_counts(drive_list: List[str]) -> Dict[str, Any]:
        total_samples = sum(drive_stats.get(d, {}).get("total_samples", 0) for d in drive_list)
        total_windows = sum(drive_stats.get(d, {}).get("total_windows", 0) for d in drive_list)
        return {
            "drives": drive_list,
            "drive_count": len(drive_list),
            "total_samples": total_samples,
            "total_windows": total_windows,
        }
        
    full_manifest = {
        "split_type": "drive_level_partitioning",
        "is_generalization_benchmark": True,
        "leakage_prevention": "Windows are strictly contained within complete separate drives.",
        "splits": {
            "train": aggregate_counts(train_drives),
            "val": aggregate_counts(val_drives),
            "test": aggregate_counts(test_drives),
        },
        "per_drive_details": drive_stats,
    }
    
    manifest_path = os.path.join(output_dir, "drive_split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(full_manifest, f, indent=2)
        
    return full_manifest
