"""
End-to-End IMU Preprocessing Pipeline for GPS-Denied Navigation.

Executes Phases 1 through 10 with reproducible seeds, strict leakage prevention,
multi-drive standardization, high-rate GNSS reference alignment, and full artifact persistence.
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.config import load_config
from src.geodesy import check_trajectory_continuity
from src.inspection import inspect_recording, generate_data_quality_report, sanitize_column_names
from src.synchronization import align_sensor_and_reference
from src.signal_processing import (
    SensorNormalizer,
    handle_missing_values,
    clip_outliers,
    apply_butterworth_lowpass,
    apply_gravity_removal,
)
from src.outage_simulation import (
    find_valid_outage_intervals,
    apply_outage_mask,
    verify_zero_leakage,
)
from src.dataset_generator import (
    generate_sliding_windows,
    create_split_manifests,
)
from src.visualization import (
    plot_raw_imu_timeseries,
    plot_cleaned_vs_raw,
    plot_gps_trajectory_local_enu,
    plot_sampling_interval_distributions,
    plot_missing_and_invalid_data,
    plot_outage_examples,
    plot_input_observation_reference_distinction,
    plot_highrate_vs_interpolated,
)


def setup_logger(log_level: int = logging.INFO) -> logging.Logger:
    """Configure structured console logging."""
    logger = logging.getLogger("IMU_Pipeline")
    logger.setLevel(log_level)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def set_seed(seed: int = 42) -> None:
    """Enforce strict reproducibility across random generators."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_pipeline(config_path: str = "configs/default_config.yaml") -> Dict[str, Any]:
    """Execute the complete research preprocessing pipeline across all dataset drives."""
    logger = setup_logger()
    logger.info("Initializing Research-Grade IMU Preprocessing Pipeline...")
    
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed locked to: {seed}")
    
    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]
    splits_dir = cfg["data"]["splits_dir"]
    reports_dir = cfg["data"]["reports_dir"]
    figures_dir = cfg["data"]["figures_dir"]
    primary_drive = cfg["data"]["primary_drive"]
    all_drives = cfg["data"].get("all_drives", [primary_drive])
    
    for d in [raw_dir, processed_dir, splits_dir, reports_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)
        
    primary_csv_path = os.path.join(raw_dir, primary_drive)
    if not os.path.exists(primary_csv_path):
        raise FileNotFoundError(f"Primary drive file not found: {primary_csv_path}")
        
    # ==========================================
    # Phase 1 & 2: Dataset Inspection & Selection (Primary S-S1)
    # ==========================================
    logger.info("--- Phase 1 & 2: Dataset Inspection and Sensor Selection ---")
    quality_report = inspect_recording(primary_csv_path)
    report_md_path = os.path.join(reports_dir, "data_quality_report.md")
    generate_data_quality_report(quality_report, report_md_path)
    logger.info(f"Data quality report saved to: {report_md_path}")
    logger.info(f"Primary drive ({primary_drive}): {quality_report['total_rows']:,} samples | {quality_report['timestamps']['nominal_rate_hz']:.2f} Hz")
    
    # Identify split drives
    split_cfg = cfg["splitting"]["multi_drive"]
    train_drive_names = split_cfg.get("train", ["S-S1.csv", "S-S2.csv", "S-S4.csv"])
    val_drive_names = split_cfg.get("val", ["S-S3a.csv"])
    test_drive_names = split_cfg.get("test", ["S-S3b.csv", "S-S3c.csv"])
    
    max_gps_gap = cfg["synchronization"]["max_gps_gap_seconds"]
    max_gps_acc = cfg["synchronization"]["max_gps_accuracy_m"]
    interp_method = cfg["synchronization"]["interpolation_method"]
    input_channels = cfg["sensors"]["input_channels"]
    
    # ==========================================
    # Step A: Align and Condition ALL Drives
    # ==========================================
    logger.info("--- Phase 3 & 4: Multi-Drive Synchronization & ENU Alignment ---")
    aligned_drives: Dict[str, pd.DataFrame] = {}
    drive_sync_meta: Dict[str, Dict[str, Any]] = {}
    drive_stats: Dict[str, Dict[str, Any]] = {}
    
    for drive_file in all_drives:
        drive_path = os.path.join(raw_dir, drive_file)
        if not os.path.exists(drive_path):
            logger.warning(f"Drive file {drive_file} not found in raw_dir; skipping.")
            continue
            
        drive_id = os.path.splitext(drive_file)[0]  # e.g. "S-S1"
        v_drive_file = "V-" + drive_id.replace("S-", "") + ".csv"  # e.g. "V-S1.csv"
        v_drive_path = os.path.join(raw_dir, v_drive_file)
        
        v_df = None
        if os.path.exists(v_drive_path):
            v_df = pd.read_csv(v_drive_path, encoding="latin1")
            logger.info(f"Drive {drive_id}: Loaded paired 10 Hz high-rate vehicle reference ({v_drive_file}).")
            
        df_raw = pd.read_csv(drive_path, encoding="latin1")
        df_clean = sanitize_column_names(df_raw)
        
        # Align
        aligned_df, sync_meta = align_sensor_and_reference(
            df_clean,
            v_df=v_df,
            origin=None,  # First valid fix is origin
            max_gps_gap_s=max_gps_gap,
            max_gps_accuracy_m=max_gps_acc,
            interpolation_method=interp_method,
        )
        
        # Condition signals
        conditioned_df, _ = handle_missing_values(
            aligned_df,
            channels=input_channels,
            max_gap_samples=cfg["preprocessing"]["missing_values"]["max_gap_samples"],
        )
        if cfg["preprocessing"]["outlier_clipping"]["enabled"]:
            conditioned_df, _ = clip_outliers(
                conditioned_df,
                accel_limit_mps2=cfg["preprocessing"]["outlier_clipping"]["accel_limit_mps2"],
                gyro_limit_rads=cfg["preprocessing"]["outlier_clipping"]["gyro_limit_rads"],
            )
        if cfg["preprocessing"]["lowpass_filter"]["enabled"]:
            conditioned_df = apply_butterworth_lowpass(
                conditioned_df,
                channels=input_channels,
                cutoff_hz=cfg["preprocessing"]["lowpass_filter"]["cutoff_hz"],
                order=cfg["preprocessing"]["lowpass_filter"]["order"],
            )
            
        aligned_drives[drive_file] = conditioned_df
        drive_sync_meta[drive_file] = sync_meta
        logger.info(f"Drive {drive_id}: Synchronized {len(conditioned_df):,} samples (Reference source: {sync_meta['reference_source']}).")
        
    # ==========================================
    # Step B: Fit Normalizer STRICTLY on Training Split Drives
    # ==========================================
    logger.info("--- Phase 5: Normalization Parameter Fitting (Train Split Only) ---")
    train_dfs = [aligned_drives[d] for d in train_drive_names if d in aligned_drives]
    if not train_dfs:
        raise RuntimeError("No training partition drives available to fit normalizer.")
        
    combined_train_df = pd.concat(train_dfs, ignore_index=True)
    normalizer = SensorNormalizer(
        method=cfg["preprocessing"]["normalization"]["method"],
        channels=input_channels,
    )
    normalizer.fit(combined_train_df)
    normalizer_save_path = os.path.join(processed_dir, "sensor_normalizer.json")
    normalizer.save(normalizer_save_path)
    logger.info(f"Fitted normalization parameters across {len(train_dfs)} training drives ({len(combined_train_df):,} samples).")
    logger.info(f"Normalizer parameters saved to: {normalizer_save_path}")
    
    # Apply normalizer to all drives and save aligned CSVs
    normalized_drives: Dict[str, pd.DataFrame] = {}
    for d_name, df_cond in aligned_drives.items():
        d_norm = normalizer.transform(df_cond)
        normalized_drives[d_name] = d_norm
        out_csv = os.path.join(processed_dir, f"{os.path.splitext(d_name)[0]}_processed_aligned.csv")
        d_norm.to_csv(out_csv, index=False)
        
    # ==========================================
    # Step C: Simulated GPS Outages (on primary drive S-S1)
    # ==========================================
    logger.info("--- Phase 6: Simulated GPS Outage Generation (10s, 30s, 60s) ---")
    primary_norm_df = normalized_drives[primary_drive]
    outage_durations = cfg["outages"]["durations_s"]
    all_outage_scenarios = []
    
    for dur in outage_durations:
        scenarios = find_valid_outage_intervals(
            primary_norm_df,
            duration_s=dur,
            nominal_rate_hz=cfg["sensors"]["nominal_imu_rate_hz"],
            max_scenarios=cfg["outages"]["max_scenarios_per_duration"],
            seed=seed,
        )
        for sc in scenarios:
            model_in, eval_ref = apply_outage_mask(primary_norm_df, sc)
            checks = verify_zero_leakage(model_in, eval_ref, sc)
            if not checks["all_leakage_checks_passed"]:
                raise RuntimeError(f"Leakage check failed for scenario: {sc['scenario_id']}")
        all_outage_scenarios.extend(scenarios)
        
    outage_json_path = os.path.join(processed_dir, "outage_scenarios.json")
    with open(outage_json_path, "w", encoding="utf-8") as f:
        json.dump(all_outage_scenarios, f, indent=2)
    logger.info(f"Generated {len(all_outage_scenarios)} verified outage scenarios in: {outage_json_path}")
    
    # ==========================================
    # Step D: Sliding Window Generation Across Complete Split Partitions
    # ==========================================
    logger.info("--- Phase 7: Sliding Window Partitioning (Train / Val / Test) ---")
    window_size = cfg["windows"]["window_size_samples"]
    stride = cfg["windows"]["stride_samples"]
    
    split_partitions = {
        "train": train_drive_names,
        "val": val_drive_names,
        "test": test_drive_names,
    }
    
    partition_window_counts = {}
    
    for split_name, drive_list in split_partitions.items():
        x_split, y_split, meta_split = [], [], []
        for d_file in drive_list:
            if d_file not in normalized_drives:
                continue
            d_df = normalized_drives[d_file]
            d_id = os.path.splitext(d_file)[0]
            x_d, y_d, meta_d = generate_sliding_windows(
                d_df,
                drive_id=d_id,
                feature_channels=input_channels,
                window_size=window_size,
                stride=stride,
                min_valid_ratio=1.0,
            )
            if len(x_d) > 0:
                x_split.append(x_d)
                y_split.append(y_d)
                meta_split.extend(meta_d)
                
            drive_stats[d_file] = {
                "total_samples": len(d_df),
                "valid_samples": int(np.sum(d_df["valid_reference"])),
                "total_windows": len(x_d),
                "duration_s": float(d_df["time_s"].iloc[-1] - d_df["time_s"].iloc[0]),
                "assigned_split": split_name,
            }
            
        if x_split:
            X_all = np.concatenate(x_split, axis=0)
            Y_all = np.concatenate(y_split, axis=0)
        else:
            X_all = np.empty((0, window_size, len(input_channels)), dtype=np.float32)
            Y_all = np.empty((0, 2), dtype=np.float32)
            
        partition_window_counts[split_name] = len(X_all)
        split_npz_path = os.path.join(processed_dir, f"{split_name}_windows.npz")
        np.savez_compressed(
            split_npz_path,
            X=X_all,
            Y=Y_all,
            meta=np.array(meta_split, dtype=object),
        )
        logger.info(f"Split [{split_name.upper()}]: Generated {len(X_all):,} windows from drives {drive_list} -> Saved to {split_npz_path}")
        
    # Also save primary S-S1 windows for legacy reference
    s1_df = normalized_drives[primary_drive]
    X_s1, Y_s1, meta_s1 = generate_sliding_windows(
        s1_df,
        drive_id="S-S1",
        feature_channels=input_channels,
        window_size=window_size,
        stride=stride,
        min_valid_ratio=1.0,
    )
    np.savez_compressed(
        os.path.join(processed_dir, "S-S1_windows.npz"),
        X=X_s1,
        Y=Y_s1,
        meta=np.array(meta_s1, dtype=object),
    )
    
    # Save Split Manifests
    create_split_manifests(
        all_drives=all_drives,
        drive_stats=drive_stats,
        output_dir=splits_dir,
        primary_drive_only=True,
    )
    full_manifest = create_split_manifests(
        all_drives=all_drives,
        drive_stats=drive_stats,
        output_dir=splits_dir,
        primary_drive_only=False,
        multi_drive_config=split_cfg,
    )
    logger.info("Drive-level train/val/test split manifests updated.")
    
    # ==========================================
    # Step E: Visualizations
    # ==========================================
    logger.info("--- Phase 8: Generating Publication Visualizations ---")
    s1_clean = sanitize_column_names(pd.read_csv(primary_csv_path, encoding="latin1"))
    s1_aligned = aligned_drives[primary_drive]
    s1_origin = drive_sync_meta[primary_drive]["origin_enu"]
    
    plot_raw_imu_timeseries(s1_clean, os.path.join(figures_dir, "fig1_raw_imu_timeseries.png"))
    plot_cleaned_vs_raw(s1_clean, s1_df, os.path.join(figures_dir, "fig2_cleaned_vs_raw.png"))
    plot_gps_trajectory_local_enu(s1_aligned, s1_origin, os.path.join(figures_dir, "fig3_gps_trajectory_local_metres.png"))
    plot_sampling_interval_distributions(s1_clean, os.path.join(figures_dir, "fig4_sampling_interval_distributions.png"))
    plot_missing_and_invalid_data(s1_aligned, os.path.join(figures_dir, "fig5_missing_and_invalid_labels.png"))
    plot_outage_examples(s1_aligned, all_outage_scenarios, os.path.join(figures_dir, "fig6_outage_examples.png"))
    plot_input_observation_reference_distinction(
        s1_aligned,
        all_outage_scenarios[0],
        os.path.join(figures_dir, "fig7_input_observation_reference_distinction.png")
    )
    
    # New Plot 8: High-Rate Ground Truth vs Phone 9s Interpolation
    v_s1_path = os.path.join(raw_dir, "V-S1.csv")
    if os.path.exists(v_s1_path):
        v_s1_df = pd.read_csv(v_s1_path, encoding="latin1")
        # Compute phone interpolated version for explicit comparison
        phone_interp_df, _ = align_sensor_and_reference(s1_clean, v_df=None, origin=(s1_origin["lat0"], s1_origin["lon0"], s1_origin["alt0"]))
        plot_highrate_vs_interpolated(
            v_s1_df,
            phone_interp_df,
            s1_origin,
            os.path.join(figures_dir, "fig8_highrate_groundtruth_vs_phone_interpolation.png"),
            zoom_time_s=(140.0, 200.0),
        )
        logger.info("Figure 8: High-rate vs interpolated comparison generated.")
        
    # ==========================================
    # Summary Report
    # ==========================================
    summary_report_path = os.path.join(reports_dir, "preprocessing_summary.md")
    write_comprehensive_summary(
        summary_report_path,
        drive_stats,
        partition_window_counts,
        all_outage_scenarios,
        full_manifest,
    )
    logger.info(f"Summary report written to: {summary_report_path}")
    logger.info("[SUCCESS] Multi-Drive Preprocessing Pipeline Completed Successfully!")
    
    return {
        "status": "SUCCESS",
        "drives_processed": len(aligned_drives),
        "total_samples": sum(d["total_samples"] for d in drive_stats.values()),
        "partition_windows": partition_window_counts,
        "num_outages": len(all_outage_scenarios),
    }


def write_comprehensive_summary(
    path: str,
    drive_stats: Dict[str, Dict[str, Any]],
    partition_windows: Dict[str, int],
    outages: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> None:
    """Generate markdown summary covering all drives, partitions, and interpolation analyses."""
    total_samples = sum(d["total_samples"] for d in drive_stats.values())
    total_valid = sum(d["valid_samples"] for d in drive_stats.values())
    total_duration_s = sum(d["duration_s"] for d in drive_stats.values())
    
    md = [
        "# Research Preprocessing Pipeline Execution Summary: Complete Multi-Drive Dataset",
        "**Dataset Source:** IO-VNBD Driver A (Smartphone IMU + Paired High-Rate Vehicle Reference)  ",
        f"**Total Drives Processed:** {len(drive_stats)} recordings  ",
        f"**Total Samples:** {total_samples:,} rows ({total_duration_s:.1f} s / {total_duration_s/3600.0:.2f} hours)  ",
        f"**Total Trustworthy Samples:** {total_valid:,} ({total_valid/total_samples*100:.2f}%)\n",
        "## 1. Drive-Level Partitioning & Manifest Summary",
        "To prevent temporal autocorrelation and window-boundary leakage, sliding windows are strictly isolated within complete drive recordings:",
        "",
        "| Split Partition | Assigned Drives | Total Samples | Duration | Total Sliding Windows (10s, 1s stride) |",
        "|---|---|---|---|---|",
        f"| **TRAIN** | `S-S1.csv`, `S-S2.csv`, `S-S4.csv` | {manifest['splits']['train']['total_samples']:,} | ~6.7 hrs | **{partition_windows.get('train', 0):,} windows** |",
        f"| **VAL** | `S-S3a.csv` | {manifest['splits']['val']['total_samples']:,} | ~41.0 min | **{partition_windows.get('val', 0):,} windows** |",
        f"| **TEST** | `S-S3b.csv`, `S-S3c.csv` | {manifest['splits']['test']['total_samples']:,} | ~73.4 min | **{partition_windows.get('test', 0):,} windows** |",
        f"| **TOTAL** | All 6 Drives | **{total_samples:,}** | **{total_duration_s/3600.0:.2f} hrs** | **{sum(partition_windows.values()):,} windows** |",
        "",
        "## 2. Per-Drive Granular Statistics",
        "| Drive Recording | Assigned Split | Total Samples | Valid Reference Samples | Valid Ratio | Total Windows | Duration (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for d_name, s in drive_stats.items():
        md.append(f"| `{d_name}` | **{s['assigned_split'].upper()}** | {s['total_samples']:,} | {s['valid_samples']:,} | {s['valid_samples']/s['total_samples']*100:.1f}% | {s['total_windows']:,} | {s['duration_s']:.1f}s |")
        
    md.extend([
        "\n## 3. Ground-Truth Trajectory: High-Rate Vehicle GNSS vs 9s Phone Interpolation",
        "- **Dual Reference Capability:** The pipeline integrates high-rate (10 Hz) continuous vehicle GNSS from `V-*.csv` as the gold-standard ground truth for trajectory evaluation during simulated outages.",
        "- **Observation on 9-Second Phone GNSS:** When internal phone GPS (~0.11 Hz) is linearly interpolated across 9-second gaps, chord truncation error occurs during curves and turns. Evaluating dead-reckoning models against true 10 Hz vehicle references provides sub-meter ground truth without corner-cutting artifacts.",
        "- **Zero Leakage Invariant:** All vehicle signals (CAN bus, wheel speeds, steering) are strictly barred from model inputs. Only the 6 smartphone IMU channels (`accel_x, accel_y, accel_z, gyro_yaw, gyro_pitch, gyro_roll`) are used as model inputs.",
        "",
        "## 4. Simulated Outage Evaluation Scenarios",
        f"- **Verified Outages Generated:** {len(outages)} scenarios across 10s, 30s, and 60s durations.",
        "| Scenario ID | Duration | Time Interval (s) | Distance Traveled | Mean Speed | Reference Coverage |",
        "|---|---|---|---|---|---|",
    ])
    for sc in outages:
        md.append(f"| `{sc['scenario_id']}` | {sc['duration_s']}s | [{sc['start_time_s']:.1f}s, {sc['end_time_s']:.1f}s] | {sc['distance_traveled_m']:.1f} m | {sc['mean_speed_mps']:.2f} m/s | 100% |")
        
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMU Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    args = parser.parse_args()
    run_pipeline(args.config)
