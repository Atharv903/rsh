"""
Phase 12D & 12E: Expanded Independent Validation Benchmark and Statistical Analysis.

Evaluates 11 methods across 30 scenarios from 2 independent sessions:
- Driver B (M_seg1, Sept 7, 2019)
- Driver D (Y1_seg3, Aug 30, 2019)

Generates:
1. experiments/phase12_validation/per_scenario_metrics.csv
2. experiments/phase12_validation/per_session_summary.csv
3. experiments/phase12_validation/statistical_comparison.csv
4. experiments/phase12_validation/data_quality_audit.json
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.evaluate_phase10_independent import load_model
from src.corrected_synchronization import extract_aligned_segments_for_drive
from src.outage_estimator import PreOutageState, estimate_initial_heading_from_pre_outage_gps


def run_dead_reckoning_and_metrics(
    pre_state: PreOutageState,
    in_outage_df: pd.DataFrame,
    pred_speed: np.ndarray,
    method_name: str,
    scenario_info: dict,
    h_sign: float = -1.0,  # -1.0 for corrected convention, +1.0 for legacy
):
    n_steps = len(in_outage_df)
    time_arr = in_outage_df["time_s"].values
    dt_arr = np.diff(time_arr, prepend=time_arr[0])
    dt_arr[0] = 0.0
    
    ref_east = in_outage_df["ref_east_m"].values
    ref_north = in_outage_df["ref_north_m"].values
    ref_speed = in_outage_df["ref_speed_mps"].values
    ref_heading = np.deg2rad(in_outage_df["ref_heading_deg"].values)
    
    seg_dists = np.sqrt(np.diff(ref_east)**2 + np.diff(ref_north)**2)
    total_length_m = float(np.sum(seg_dists))
    
    gyro_yaw_rate = (in_outage_df["gyro_pitch"].values.astype(np.float32) - pre_state.gyro_bias_rads)
    
    pe = np.zeros(n_steps, dtype=np.float64)
    pn = np.zeros(n_steps, dtype=np.float64)
    head = np.zeros(n_steps, dtype=np.float64)
    
    pe[0] = pre_state.pos_east_m
    pn[0] = pre_state.pos_north_m
    head[0] = np.deg2rad(pre_state.heading_deg)
    
    for k in range(1, n_steps):
        dt = dt_arr[k]
        d_psi = h_sign * float(gyro_yaw_rate[k] * dt)
        head[k] = (head[k-1] + d_psi) % (2.0 * np.pi)
        
        v_e = pred_speed[k] * np.sin(head[k])
        v_n = pred_speed[k] * np.cos(head[k])
        
        pe[k] = pe[k-1] + v_e * dt
        pn[k] = pn[k-1] + v_n * dt
        
    speed_err = pred_speed - ref_speed
    speed_mae = float(np.mean(np.abs(speed_err)))
    speed_rmse = float(np.sqrt(np.mean(speed_err**2)))
    speed_bias = float(np.mean(speed_err))
    
    pos_err = np.sqrt((pe - ref_east)**2 + (pn - ref_north)**2)
    ate_rmse = float(np.sqrt(np.mean(pos_err**2)))
    final_drift_m = float(pos_err[-1])
    drift_pct = float((final_drift_m / max(total_length_m, 1.0)) * 100.0)
    
    # Vector Error Decomposition at endpoint
    delta_e = pe[-1] - ref_east[-1]
    delta_n = pn[-1] - ref_north[-1]
    psi_f = ref_heading[-1]
    u_along = np.array([np.sin(psi_f), np.cos(psi_f)])
    u_cross = np.array([np.cos(psi_f), -np.sin(psi_f)])
    along_err = float(delta_e * u_along[0] + delta_n * u_along[1])
    cross_err = float(delta_e * u_cross[0] + delta_n * u_cross[1])
    
    conv_label = "corrected (-1)" if h_sign < 0 else "legacy (+1)"
    
    return {
        "scenario_id": scenario_info["scenario_id"],
        "driver_id": scenario_info["driver_id"],
        "session_name": scenario_info["session_name"],
        "duration_s": scenario_info["duration_s"],
        "method": method_name,
        "heading_convention": conv_label,
        "total_length_m": total_length_m,
        "speed_mae_mps": speed_mae,
        "speed_rmse_mps": speed_rmse,
        "speed_bias_mps": speed_bias,
        "ate_rmse_m": ate_rmse,
        "final_drift_m": final_drift_m,
        "along_track_err_m": along_err,
        "cross_track_err_m": cross_err,
        "drift_pct": drift_pct,
    }


def compute_bootstrap_ci(diff_series, n_bootstrap=2000, alpha=0.05, seed=42):
    """Computes paired bootstrap confidence interval for the mean difference."""
    rng = np.random.RandomState(seed)
    n = len(diff_series)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(diff_series, size=n, replace=True)
        boot_means.append(np.mean(sample))
    low = np.percentile(boot_means, 100 * (alpha / 2))
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(low), float(high)


def main():
    print("=" * 85)
    print("PHASE 12: EXPANDED INDEPENDENT VALIDATION BENCHMARK (DRIVER B + DRIVER D)")
    print("=" * 85)
    
    out_dir = Path("experiments/phase12_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = out_dir / "phase12_scenarios.json"
    with open(manifest_path) as f:
        scenarios = json.load(f)
    print(f"Loaded {len(scenarios)} total scenarios from manifest.")
    
    # Load raw driving data
    print("Loading continuous driving segments...")
    raw_b_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)"
    segs_b, _ = extract_aligned_segments_for_drive("M", raw_dir=raw_b_dir)
    df_b = segs_b[1] # M_seg1
    
    raw_d_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/Y (Driver D)/Y1"
    segs_d, _ = extract_aligned_segments_for_drive("Y1", raw_dir=raw_d_dir)
    df_d = segs_d[2] # Y1_seg3
    
    drive_dfs = {"M": df_b, "Y1": df_d}
    print(f"  Loaded Driver B (M_seg1): {len(df_b)} samples ({len(df_b)*0.1/60:.1f} mins)")
    print(f"  Loaded Driver D (Y1_seg3): {len(df_d)} samples ({len(df_d)*0.1/60:.1f} mins)")

    # Load frozen normalizer
    norm_meta = json.load(open("data/processed/sensor_normalizer_grouped.json"))
    norm_params = norm_meta["params"]
    feat_channels = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
    norm_mean = np.array([norm_params[ch]["mean"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
    norm_std = np.array([norm_params[ch]["std"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)

    # Load 9 frozen GPU checkpoints
    ckpt_dir = Path("experiments/phase8_corrected_speed_models/checkpoints")
    models = {}
    for d in sorted(ckpt_dir.iterdir()):
        pt_path = d / "best_model.pt"
        if pt_path.exists():
            m, mtype = load_model(pt_path)
            models[d.name] = (m, mtype)
    print(f"Loaded {len(models)} frozen GPU checkpoints.")

    # Execution loop
    all_metrics = []
    t_start = time.time()
    
    for sc_idx, sc in enumerate(scenarios):
        drive_key = sc["drive_id"]
        drive_df = drive_dfs[drive_key]
        st, end = sc["start_idx"], sc["end_idx"]
        dur = sc["duration_s"]
        sid = sc["scenario_id"]
        pre_st = max(0, st - sc["pre_window_samples"])
        
        pre_df = drive_df.iloc[pre_st:st].copy()
        in_outage_df = drive_df.iloc[st:end].copy()
        n_steps = len(in_outage_df)
        
        heading_deg, last_spd = estimate_initial_heading_from_pre_outage_gps(pre_df)
        pos_e0 = float(pre_df["ref_east_m"].iloc[-1])
        pos_n0 = float(pre_df["ref_north_m"].iloc[-1])
        gyro_bias = float(np.median(pre_df["gyro_pitch"]))
        
        pre_state = PreOutageState(
            t0_s=float(pre_df["time_s"].iloc[-1]),
            pos_east_m=pos_e0,
            pos_north_m=pos_n0,
            speed_mps=last_spd,
            heading_deg=heading_deg,
            gyro_bias_rads=gyro_bias,
        )
        
        # Baselines
        zero_speed_arr = np.zeros(n_steps, dtype=np.float32)
        zero_speed_arr[0] = last_spd
        const_speed_arr = np.full(n_steps, last_spd, dtype=np.float32)
        
        # Construct batched IMU tensor
        pre_imu = pre_df[feat_channels].values.astype(np.float32)
        raw_imu = in_outage_df[feat_channels].values.astype(np.float32)
        full_imu = np.vstack([pre_imu, raw_imu])
        
        windows = []
        for k in range(n_steps):
            w_end = len(pre_imu) + k + 1
            w_start = w_end - 100
            w = full_imu[w_start:w_end]
            windows.append((w - norm_mean) / norm_std)
        windows_tensor = torch.from_numpy(np.array(windows, dtype=np.float32))
        
        # Predict speeds across all 9 models
        model_speeds = {}
        for m_name, (m, _) in models.items():
            with torch.no_grad():
                preds = np.maximum(0.0, m(windows_tensor).squeeze(-1).cpu().numpy())
            preds[0] = last_spd
            model_speeds[m_name] = preds
            
        # Determine heading conventions to run
        # For Driver B, test both corrected (-1) and legacy (+1) for reconciliation
        # For Driver D, test corrected (-1)
        conventions = [-1.0]
        if sc["driver_id"] == "Driver B":
            conventions.append(+1.0)
            
        for h_sign in conventions:
            # Zero-Speed
            r_z = run_dead_reckoning_and_metrics(pre_state, in_outage_df, zero_speed_arr, "Zero-Speed", sc, h_sign)
            all_metrics.append(r_z)
            
            # Constant-Velocity
            r_c = run_dead_reckoning_and_metrics(pre_state, in_outage_df, const_speed_arr, "Constant-Velocity", sc, h_sign)
            all_metrics.append(r_c)
            
            # Neural models
            for m_name, preds in model_speeds.items():
                r_m = run_dead_reckoning_and_metrics(pre_state, in_outage_df, preds, m_name, sc, h_sign)
                all_metrics.append(r_m)
                
        if (sc_idx + 1) % 5 == 0 or sc_idx == len(scenarios) - 1:
            print(f"  [{sc_idx+1:2d}/{len(scenarios)}] Completed scenario {sid} ({dur}s, {sc['driver_id']})")
            
    elapsed = time.time() - t_start
    print(f"\nAll evaluations finished in {elapsed:.2f} seconds.")
    
    # 1. Save per-scenario metrics CSV
    per_scenario_df = pd.DataFrame(all_metrics)
    out_scen_csv = out_dir / "per_scenario_metrics.csv"
    per_scenario_df.to_csv(out_scen_csv, index=False)
    print(f"Saved per-scenario metrics to {out_scen_csv}")
    
    # 2. Per-Session and Pooled Summaries
    def get_arch(m):
        if "cnngru" in m: return "CNN-GRU"
        if "mamba2" in m: return "Mamba-2"
        if "moderntcn" in m: return "ModernTCN"
        return m
    per_scenario_df["arch"] = per_scenario_df["method"].apply(get_arch)

    # Focus on corrected convention
    corr_df = per_scenario_df[per_scenario_df["heading_convention"] == "corrected (-1)"].copy()
    
    per_session_summary = corr_df.groupby(["driver_id", "session_name", "duration_s", "arch"])[
        ["speed_mae_mps", "speed_rmse_mps", "speed_bias_mps", "ate_rmse_m", "final_drift_m", "drift_pct", "along_track_err_m", "cross_track_err_m"]
    ].mean().reset_index()
    
    out_sess_csv = out_dir / "per_session_summary.csv"
    per_session_summary.to_csv(out_sess_csv, index=False)
    print(f"Saved per-session summary to {out_sess_csv}")
    
    # 3. Statistical Comparison vs Constant-Velocity
    stat_rows = []
    
    # Evaluate by session and duration
    session_list = ["Driver B", "Driver D", "POOLED (Driver B + D)"]
    
    for sess in session_list:
        if sess == "POOLED (Driver B + D)":
            sess_df = corr_df
        else:
            sess_df = corr_df[corr_df["driver_id"] == sess]
            
        for dur in [10, 30, 60]:
            dur_df = sess_df[sess_df["duration_s"] == dur]
            
            # Pivot by scenario and arch
            pivot = dur_df.groupby(["scenario_id", "arch"])["final_drift_m"].mean().unstack("arch").reset_index()
            cv_drift = pivot["Constant-Velocity"].values
            
            for test_arch in ["ModernTCN", "CNN-GRU", "Mamba-2"]:
                if test_arch not in pivot.columns: continue
                nn_drift = pivot[test_arch].values
                diff = cv_drift - nn_drift # positive means model beats CV
                
                n = len(diff)
                mean_diff = float(np.mean(diff))
                med_diff = float(np.median(diff))
                std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
                
                # Paired t-test
                t_stat, p_t = stats.ttest_rel(cv_drift, nn_drift) if n > 1 else (np.nan, np.nan)
                
                # Wilcoxon signed rank test
                try:
                    w_stat, p_w = stats.wilcoxon(cv_drift, nn_drift)
                except Exception:
                    w_stat, p_w = np.nan, np.nan
                    
                # Bootstrap 95% CI
                boot_low, boot_high = compute_bootstrap_ci(diff) if n > 2 else (np.nan, np.nan)
                
                # Cohen's d effect size
                cohens_d = float(mean_diff / std_diff) if std_diff > 1e-6 else 0.0
                
                scen_won = int(np.sum(diff > 0))
                
                stat_rows.append({
                    "session": sess,
                    "duration_s": dur,
                    "model_architecture": test_arch,
                    "num_scenarios": n,
                    "scenarios_won": scen_won,
                    "win_rate_pct": float(scen_won / n * 100.0),
                    "mean_cv_drift_m": float(np.mean(cv_drift)),
                    "mean_model_drift_m": float(np.mean(nn_drift)),
                    "mean_advantage_m": mean_diff,
                    "median_advantage_m": med_diff,
                    "std_advantage_m": std_diff,
                    "ci95_bootstrap_low": boot_low,
                    "ci95_bootstrap_high": boot_high,
                    "cohens_d": cohens_d,
                    "paired_t_stat": float(t_stat),
                    "paired_t_pval": float(p_t),
                    "wilcoxon_stat": float(w_stat),
                    "wilcoxon_pval": float(p_w),
                })
                
    stat_df = pd.DataFrame(stat_rows)
    out_stat_csv = out_dir / "statistical_comparison.csv"
    stat_df.to_csv(out_stat_csv, index=False)
    print(f"Saved statistical comparison to {out_stat_csv}")
    
    # 4. Save Data Quality Audit JSON
    audit_data = {
        "audit_timestamp": "2026-10-29",
        "evaluated_independent_sessions": {
            "Driver_B_M_seg1": {
                "driver": "Driver B",
                "session": "M (Driver B)",
                "date": "2019-09-07",
                "segment": "M_seg1",
                "samples": 61714,
                "duration_minutes": 102.9,
                "mean_gravity_z": 9.81,
                "phone_mounting": "Flat horizontal console facing upwards",
                "yaw_rate_channel": "GYROSCOPE Pitch (rad/s)",
                "corr_gyro_yaw_rate": +0.8811,
                "corr_heading_rate": -0.6946,
                "time_shift_samples": 34,
                "status": "VALID_SYNCHRONIZED"
            },
            "Driver_D_Y1_seg3": {
                "driver": "Driver D",
                "session": "Y1 (Driver D)",
                "date": "2019-08-30",
                "segment": "Y1_seg3",
                "samples": 65761,
                "duration_minutes": 109.6,
                "mean_gravity_z": 9.81,
                "phone_mounting": "Flat horizontal console facing upwards",
                "yaw_rate_channel": "GYROSCOPE Pitch (rad/s)",
                "corr_gyro_yaw_rate": +0.6802,
                "corr_heading_rate": -0.6814,
                "time_shift_samples": 1149,
                "status": "VALID_SYNCHRONIZED"
            }
        },
        "unsuitable_sessions_documentation": {
            "Driver_E_Vfa01": "November 2019 session; post-alignment correlation near zero (0.27 max cross-correlation). Clock drift or different vehicle mounting.",
            "Driver_E_Vfa02": "November 2019 session; post-alignment correlation +0.011.",
            "Driver_E_Vta_series": "Checked all 30 Vta sessions; correlation with yaw rate < 0.25 across all drives.",
            "Driver_E_Vw_series": "Mismatched row counts (20476 vs 20475) and weak gyro correlation."
        }
    }
    
    out_audit_json = out_dir / "data_quality_audit.json"
    with open(out_audit_json, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"Saved data quality audit to {out_audit_json}")

    # Display Summary Results Table
    print("\n" + "=" * 105)
    print("PHASE 12 EXPANDED VALIDATION SUMMARY: MODERNTCN VS CONSTANT-VELOCITY")
    print("=" * 105)
    
    print(f"{'Session':24s} | {'Duration':10s} | {'CV Drift':12s} | {'ModernTCN':12s} | {'Advantage':14s} | {'Win Rate':10s} | {'95% Bootstrap CI':20s}")
    print("-" * 105)
    for _, r in stat_df[stat_df["model_architecture"] == "ModernTCN"].iterrows():
        sess = r["session"]
        dur = f"{int(r['duration_s'])}s"
        cv = f"{r['mean_cv_drift_m']:.2f} m"
        nn = f"{r['mean_model_drift_m']:.2f} m"
        adv = f"{r['mean_advantage_m']:+.2f} m"
        wr = f"{r['scenarios_won']}/{r['num_scenarios']} ({r['win_rate_pct']:.0f}%)"
        ci = f"[{r['ci95_bootstrap_low']:+.1f}, {r['ci95_bootstrap_high']:+.1f}] m"
        print(f"{sess:24s} | {dur:10s} | {cv:12s} | {nn:12s} | {adv:14s} | {wr:10s} | {ci:20s}")

if __name__ == "__main__":
    main()
