"""
Phase 15: State-of-the-Art Benchmarking, Speed Estimation Audit & Generalization.

Evaluates 6 Local Baselines:
1. Constant-Velocity (CV)
2. Acceleration-Integration (Pure Inertial Speed Propagation)
3. ModernTCN (3-seed ensemble)
4. Phase 14 Causal Hybrid
5. Constrained Causal Hybrid (physical slew-rate limit)
6. Complementary Filter (leaky inertial acceleration fusion)

Across 3 Gyroscope & Accelerometer Calibration Protocols:
- Protocol A: Zero-Bias Baseline (b = 0.0)
- Protocol B: Prior Calibration (frozen initial stationary stop at trip start)
- Protocol C: Causal Online Calibration (updated strictly causally prior to outage onset)

Across 30 Standardized Scenarios (Driver B and Driver D; 10s, 30s, 60s).
Generates:
- experiments/phase15_validation/per_scenario_metrics.csv
- experiments/phase15_validation/summary_metrics.csv
- experiments/phase15_validation/config.json
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.evaluate_phase10_independent import load_model
from src.corrected_synchronization import extract_aligned_segments_for_drive
from src.outage_estimator import PreOutageState, estimate_initial_heading_from_pre_outage_gps
from experiments.phase14_validation.evaluate_phase14_causal import (
    get_causal_online_bias,
    compute_causal_hybrid_speed,
    PRIOR_CALIBRATION_BIASES,
)

# Prior accelerometer bias from initial stationary interval at drive start
PRIOR_CALIBRATION_ACCEL_BIASES = {
    "M": +0.0241,   # Driver B initial stop accel_x mean
    "Y1": -0.0185,  # Driver D initial stop accel_x mean
}

def get_causal_online_accel_bias(drive_df, outage_start_idx, min_stat_samples=30):
    pre_data = drive_df.iloc[:outage_start_idx]
    is_stat = (pre_data["ref_speed_mps"] < 0.1).values
    curr_end = None
    for i in range(len(is_stat) - 1, -1, -1):
        if is_stat[i] and curr_end is None:
            curr_end = i
        elif not is_stat[i] and curr_end is not None:
            if (curr_end - i) >= min_stat_samples:
                stat_slice = pre_data.iloc[i+1 : curr_end+1]
                return float(stat_slice["accel_x"].mean())
            curr_end = None
    if curr_end is not None and curr_end >= min_stat_samples:
        stat_slice = pre_data.iloc[: curr_end+1]
        return float(stat_slice["accel_x"].mean())
    return 0.0

def run_causal_dead_reckoning_phase15(
    pre_state: PreOutageState,
    in_outage_df: pd.DataFrame,
    pred_speed: np.ndarray,
    gyro_bias_val: float,
    h_sign: float = -1.0,
):
    n_steps = len(in_outage_df)
    time_arr = in_outage_df["time_s"].values
    dt_arr = np.diff(time_arr, prepend=time_arr[0])
    dt_arr[0] = 0.0
    
    ref_east = in_outage_df["ref_east_m"].values
    ref_north = in_outage_df["ref_north_m"].values
    ref_speed = in_outage_df["ref_speed_mps"].values
    ref_heading = np.deg2rad(in_outage_df["ref_heading_deg"].values)
    
    gyro_yaw_rate = in_outage_df["gyro_pitch"].values.astype(np.float32) - gyro_bias_val
    
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
        
    pos_err = np.sqrt((pe - ref_east)**2 + (pn - ref_north)**2)
    final_drift_m = float(pos_err[-1])
    ate_rmse_m = float(np.sqrt(np.mean(pos_err**2)))
    
    speed_err = pred_speed - ref_speed
    speed_mae = float(np.mean(np.abs(speed_err)))
    speed_rmse = float(np.sqrt(np.mean(speed_err**2)))
    
    delta_e = pe[-1] - ref_east[-1]
    delta_n = pn[-1] - ref_north[-1]
    psi_f = ref_heading[-1]
    u_along = np.array([np.sin(psi_f), np.cos(psi_f)])
    u_cross = np.array([np.cos(psi_f), -np.sin(psi_f)])
    along_err = float(delta_e * u_along[0] + delta_n * u_along[1])
    cross_err = float(delta_e * u_cross[0] + delta_n * u_cross[1])
    
    head_diff = np.arctan2(np.sin(head - ref_heading), np.cos(head - ref_heading))
    final_heading_err_deg = float(np.rad2deg(np.abs(head_diff[-1])))
    mean_heading_err_deg = float(np.rad2deg(np.mean(np.abs(head_diff))))
    
    # Distance travelled calculation
    d_east = np.diff(ref_east)
    d_north = np.diff(ref_north)
    dist_travelled_m = float(np.sum(np.sqrt(d_east**2 + d_north**2)))
    drift_pct_of_distance = float((final_drift_m / max(1.0, dist_travelled_m)) * 100.0)
    
    return {
        "final_drift_m": final_drift_m,
        "ate_rmse_m": ate_rmse_m,
        "speed_mae_mps": speed_mae,
        "speed_rmse_mps": speed_rmse,
        "along_track_err_m": along_err,
        "cross_track_err_m": cross_err,
        "final_heading_err_deg": final_heading_err_deg,
        "mean_heading_err_deg": mean_heading_err_deg,
        "distance_travelled_m": dist_travelled_m,
        "drift_pct_of_distance": drift_pct_of_distance,
    }

def main():
    print("=" * 80)
    print("PHASE 15: STATE-OF-THE-ART BENCHMARKING & MULTI-PROTOCOL EVALUATION")
    print("=" * 80)
    
    manifest_path = repo_root / "experiments/phase12_validation/phase12_scenarios.json"
    with open(manifest_path) as f:
        scenarios = json.load(f)
        
    raw_b_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)"
    segs_b, _ = extract_aligned_segments_for_drive("M", raw_dir=raw_b_dir)
    df_b = segs_b[1]

    raw_d_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/Y (Driver D)/Y1"
    segs_d, _ = extract_aligned_segments_for_drive("Y1", raw_dir=raw_d_dir)
    df_d = segs_d[2]
    
    drive_dfs = {"M": df_b, "Y1": df_d}
    
    norm_meta = json.load(open("data/processed/sensor_normalizer_grouped.json"))
    norm_params = norm_meta["params"]
    feat_channels = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
    norm_mean = np.array([norm_params[ch]["mean"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
    norm_std = np.array([norm_params[ch]["std"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
    
    ckpt_dir = repo_root / "experiments/phase8_corrected_speed_models/checkpoints"
    mtcn_models = []
    for s in [42, 123, 456]:
        m, _ = load_model(ckpt_dir / f"moderntcnmodel_seed{s}/best_model.pt")
        mtcn_models.append(m)
        
    out_dir = repo_root / "experiments/phase15_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_records = []
    t0_start = time.time()
    
    for sc in scenarios:
        drive_key = sc["drive_id"]
        drive_df = drive_dfs[drive_key]
        st, end = sc["start_idx"], sc["end_idx"]
        dur = sc["duration_s"]
        sid = sc["scenario_id"]
        pre_st = max(0, st - sc["pre_window_samples"])
        
        pre_df = drive_df.iloc[pre_st:st].copy()
        in_outage_df = drive_df.iloc[st:end].copy()
        n_steps = len(in_outage_df)
        time_arr = in_outage_df["time_s"].values
        dt_arr = np.diff(time_arr, prepend=time_arr[0])
        dt_arr[0] = 0.0
        
        heading_deg, last_spd = estimate_initial_heading_from_pre_outage_gps(pre_df)
        pos_e0 = float(pre_df["ref_east_m"].iloc[-1])
        pos_n0 = float(pre_df["ref_north_m"].iloc[-1])
        
        pre_state = PreOutageState(
            t0_s=float(pre_df["time_s"].iloc[-1]),
            pos_east_m=pos_e0,
            pos_north_m=pos_n0,
            speed_mps=last_spd,
            heading_deg=heading_deg,
            gyro_bias_rads=0.0,
        )
        
        # Calibration Biases
        b_gyro_a = 0.0
        b_accel_a = 0.0
        
        b_gyro_b = PRIOR_CALIBRATION_BIASES[drive_key]
        b_accel_b = PRIOR_CALIBRATION_ACCEL_BIASES[drive_key]
        
        b_gyro_c, _, _ = get_causal_online_bias(drive_df, st)
        b_accel_c = get_causal_online_accel_bias(drive_df, st)
        
        calib_protocols = {
            "Protocol_A_ZeroBias": {"gyro": b_gyro_a, "accel": b_accel_a},
            "Protocol_B_PriorCalib": {"gyro": b_gyro_b, "accel": b_accel_b},
            "Protocol_C_CausalOnline": {"gyro": b_gyro_c, "accel": b_accel_c},
        }
        
        accel_x_in = in_outage_df["accel_x"].values
        
        # 1. Constant-Velocity
        spd_cv = np.full(n_steps, last_spd, dtype=np.float64)
        
        # 3. ModernTCN (3-seed ensemble)
        spd_mtcn = np.zeros(n_steps, dtype=np.float64)
        spd_mtcn[0] = last_spd
        for k in range(1, n_steps):
            w_start = (st + k) - 100
            w_end = st + k
            w_raw = drive_df.iloc[w_start:w_end][feat_channels].values.astype(np.float32)
            w_norm = (w_raw - norm_mean) / norm_std
            inp_t = torch.from_numpy(w_norm).unsqueeze(0)
            with torch.no_grad():
                preds = [m(inp_t).item() for m in mtcn_models]
            spd_mtcn[k] = max(0.0, float(np.mean(preds)))
            
        # 4. Phase 14 Causal Hybrid
        spd_hyb_p14 = compute_causal_hybrid_speed(
            time_arr, last_spd, spd_mtcn, accel_x=accel_x_in, t_switch=10.0, tau=2.0, window_len=5, brake_thr=-1.5
        )
        
        # 5. Constrained Causal Hybrid (slew-rate bounded)
        spd_hyb_constrained = np.zeros(n_steps, dtype=np.float64)
        spd_hyb_constrained[0] = last_spd
        for k in range(1, n_steps):
            dt = dt_arr[k]
            val = spd_hyb_p14[k]
            v_max = spd_hyb_constrained[k-1] + 4.0 * dt
            v_min = max(0.0, spd_hyb_constrained[k-1] - 8.0 * dt)
            spd_hyb_constrained[k] = np.clip(val, v_min, v_max)
            
        # Run across the 3 calibration protocols
        for proto_name, b_dict in calib_protocols.items():
            b_gyro = b_dict["gyro"]
            b_accel = b_dict["accel"]
            
            # 2. Pure Acceleration Integration (depends on b_accel of protocol)
            spd_accel = np.zeros(n_steps, dtype=np.float64)
            spd_accel[0] = last_spd
            for k in range(1, n_steps):
                dt = dt_arr[k]
                spd_accel[k] = max(0.0, spd_accel[k-1] + (accel_x_in[k] - b_accel) * dt)
                
            # 6. Complementary Filter (fusing acceleration with hybrid)
            spd_comp = np.zeros(n_steps, dtype=np.float64)
            spd_comp[0] = last_spd
            gamma = 0.15
            for k in range(1, n_steps):
                dt = dt_arr[k]
                v_in = spd_comp[k-1] + (accel_x_in[k] - b_accel) * dt
                spd_comp[k] = max(0.0, (1.0 - gamma) * v_in + gamma * spd_hyb_p14[k])
                
            estimators = {
                "Constant-Velocity": spd_cv,
                "Acceleration-Integration": spd_accel,
                "ModernTCN": spd_mtcn,
                "Phase14-Causal-Hybrid": spd_hyb_p14,
                "Constrained-Causal-Hybrid": spd_hyb_constrained,
                "Complementary-Filter": spd_comp,
            }
            
            for est_name, spd_vec in estimators.items():
                res = run_causal_dead_reckoning_phase15(
                    pre_state, in_outage_df, spd_vec, gyro_bias_val=b_gyro
                )
                metrics_records.append({
                    "scenario_id": sid,
                    "driver_id": sc["driver_id"],
                    "duration_s": dur,
                    "calibration_protocol": proto_name,
                    "estimator": est_name,
                    "gyro_bias_rads": b_gyro,
                    "accel_bias_mps2": b_accel,
                    **res,
                })
                
    elapsed = time.time() - t0_start
    print(f"Phase 15 full evaluation completed in {elapsed:.1f} s.")
    
    # Save per-scenario metrics
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(out_dir / "per_scenario_metrics.csv", index=False)
    print(f"Saved {out_dir / 'per_scenario_metrics.csv'} ({len(df_metrics)} records)")
    
    # Summary metrics across sessions and pooled
    summary_list = []
    for sess in ["Driver B", "Driver D", "POOLED"]:
        sess_df = df_metrics if sess == "POOLED" else df_metrics[df_metrics["driver_id"] == sess]
        
        for proto in ["Protocol_A_ZeroBias", "Protocol_B_PriorCalib", "Protocol_C_CausalOnline"]:
            for dur in [10, 30, 60]:
                for est in ["Constant-Velocity", "Acceleration-Integration", "ModernTCN", "Phase14-Causal-Hybrid", "Constrained-Causal-Hybrid", "Complementary-Filter"]:
                    sub = sess_df[
                        (sess_df["calibration_protocol"] == proto) &
                        (sess_df["duration_s"] == dur) &
                        (sess_df["estimator"] == est)
                    ]
                    if len(sub) == 0: continue
                    
                    drifts = sub["final_drift_m"].values
                    speed_maes = sub["speed_mae_mps"].values
                    speed_rmses = sub["speed_rmse_mps"].values
                    ate_rmses = sub["ate_rmse_m"].values
                    head_errs = sub["final_heading_err_deg"].values
                    along_errs = sub["along_track_err_m"].values
                    cross_errs = sub["cross_track_err_m"].values
                    dist_pcts = sub["drift_pct_of_distance"].values
                    
                    summary_list.append({
                        "session": sess,
                        "calibration_protocol": proto,
                        "duration_s": dur,
                        "estimator": est,
                        "n_scenarios": len(sub),
                        "mean_final_drift_m": float(np.mean(drifts)),
                        "median_final_drift_m": float(np.median(drifts)),
                        "std_final_drift_m": float(np.std(drifts, ddof=1)),
                        "iqr_final_drift_m": float(np.percentile(drifts, 75) - np.percentile(drifts, 25)),
                        "mean_along_track_err_m": float(np.mean(along_errs)),
                        "mean_cross_track_err_m": float(np.mean(cross_errs)),
                        "mean_speed_mae_mps": float(np.mean(speed_maes)),
                        "mean_speed_rmse_mps": float(np.mean(speed_rmses)),
                        "mean_ate_rmse_m": float(np.mean(ate_rmses)),
                        "mean_heading_err_deg": float(np.mean(head_errs)),
                        "mean_drift_pct_of_distance": float(np.mean(dist_pcts)),
                        "median_drift_pct_of_distance": float(np.median(dist_pcts)),
                    })
                    
    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    print(f"Saved {out_dir / 'summary_metrics.csv'} ({len(df_summary)} summary rows)")
    
    # Save config.json
    config = {
        "evaluation_phase": "Phase 15: State-of-the-Art Benchmarking & Speed Estimation Audit",
        "evaluation_timestamp": "2026-10-30",
        "reference_commit": "45230a3",
        "num_scenarios": len(scenarios),
        "protocols": {
            "Protocol_A": "Zero-bias baseline (b_gyro = 0.0 rad/s, b_accel = 0.0 m/s^2)",
            "Protocol_B": "Prior calibration from initial pre-departure stationary stop (frozen)",
            "Protocol_C": "Causal online calibration from verified stationary stop prior to outage",
        },
        "prior_biases": {
            "gyro_rads": PRIOR_CALIBRATION_BIASES,
            "accel_mps2": PRIOR_CALIBRATION_ACCEL_BIASES,
        },
        "baselines_evaluated": [
            "Constant-Velocity",
            "Acceleration-Integration",
            "ModernTCN",
            "Phase14-Causal-Hybrid",
            "Constrained-Causal-Hybrid",
            "Complementary-Filter"
        ],
        "models": ["moderntcnmodel_seed42", "moderntcnmodel_seed123", "moderntcnmodel_seed456"],
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved {out_dir / 'config.json'}")

if __name__ == "__main__":
    main()
