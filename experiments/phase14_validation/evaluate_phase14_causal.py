"""
Phase 14: Causal Bias Audit, Leakage Elimination & Hybrid Estimator Validation.

Evaluates 3 distinct Gyroscope Calibration Protocols:
- Protocol A: Zero-Bias Baseline (b = 0.0)
- Protocol B: Prior Calibration (frozen from initial stationary interval at drive start)
- Protocol C: Causal Online Calibration (updated strictly causally prior to outage onset)

Across 4 Navigation Estimators:
1. Constant-Velocity (CV)
2. ModernTCN (3-seed ensemble)
3. Existing Hybrid (Phase 13 specification)
4. Corrected Causal Hybrid (Phase 14 sustained braking filter)

Across 30 Standardized Scenarios (Driver B and Driver D).
Generates:
- experiments/phase14_validation/per_scenario_metrics.csv
- experiments/phase14_validation/summary_metrics.csv
- experiments/phase14_validation/bias_estimation_log.csv
- experiments/phase14_validation/config.json
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
from tests.test_gyro_bias_and_hybrid import compute_hybrid_speed

# Protocol B: Prior calibration values from initial stationary stop at t=0
PRIOR_CALIBRATION_BIASES = {
    "M": +0.003177,  # Driver B initial stationary stop (t = 0 to 40.8s)
    "Y1": -0.005010, # Driver D initial stationary stop (t = 0 to 12.6s)
}

def get_causal_online_bias(drive_df, outage_start_idx, min_stat_samples=30):
    """Protocol C: Finds the most recent stationary period strictly prior to outage_start_idx."""
    pre_data = drive_df.iloc[:outage_start_idx]
    is_stat = (pre_data["ref_speed_mps"] < 0.1).values
    curr_end = None
    for i in range(len(is_stat) - 1, -1, -1):
        if is_stat[i] and curr_end is None:
            curr_end = i
        elif not is_stat[i] and curr_end is not None:
            if (curr_end - i) >= min_stat_samples:
                stat_slice = pre_data.iloc[i+1 : curr_end+1]
                return float(stat_slice["gyro_pitch"].mean()), float(stat_slice["time_s"].iloc[-1]), (curr_end - i) * 0.1
            curr_end = None
    if curr_end is not None and curr_end >= min_stat_samples:
        stat_slice = pre_data.iloc[: curr_end+1]
        return float(stat_slice["gyro_pitch"].mean()), float(stat_slice["time_s"].iloc[-1]), curr_end * 0.1
    # Fallback to zero if no prior stationary stop found
    return 0.0, 0.0, 0.0

def compute_causal_hybrid_speed(time_s, v0, v_nn, accel_x, t_switch=10.0, tau=2.0, window_len=5, brake_thr=-1.5):
    """Corrected Causal Hybrid: Uses causal rolling mean for braking detection (no 1-sample spike false latching)."""
    n = len(time_s)
    dt_outage = time_s - time_s[0]
    alpha = 1.0 / (1.0 + np.exp(-(dt_outage - t_switch) / tau))
    
    # Causal rolling average of longitudinal acceleration
    is_braking = np.zeros(n, dtype=bool)
    for i in range(window_len, n):
        if np.mean(accel_x[i-window_len : i]) < brake_thr:
            is_braking[i] = True
            
    braking_latched = np.maximum.accumulate(is_braking)
    alpha_eff = np.where(braking_latched, 1.0, alpha)
    
    return (1.0 - alpha_eff) * v0 + alpha_eff * v_nn

def run_causal_dead_reckoning(
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
    
    return {
        "final_drift_m": final_drift_m,
        "ate_rmse_m": ate_rmse_m,
        "speed_mae_mps": speed_mae,
        "speed_rmse_mps": speed_rmse,
        "along_track_err_m": along_err,
        "cross_track_err_m": cross_err,
        "final_heading_err_deg": final_heading_err_deg,
        "mean_heading_err_deg": mean_heading_err_deg,
    }

def main():
    print("=" * 80)
    print("PHASE 14: CAUSAL BIAS AUDIT & RIGOROUS HYBRID ESTIMATOR VALIDATION")
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
    
    # Load normalizer
    norm_meta = json.load(open("data/processed/sensor_normalizer_grouped.json"))
    norm_params = norm_meta["params"]
    feat_channels = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
    norm_mean = np.array([norm_params[ch]["mean"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
    norm_std = np.array([norm_params[ch]["std"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
    
    # Load 3 ModernTCN models for ensemble
    ckpt_dir = repo_root / "experiments/phase8_corrected_speed_models/checkpoints"
    mtcn_models = []
    for s in [42, 123, 456]:
        m, _ = load_model(ckpt_dir / f"moderntcnmodel_seed{s}/best_model.pt")
        mtcn_models.append(m)
        
    out_dir = repo_root / "experiments/phase14_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_records = []
    bias_log_records = []
    
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
        outage_start_s = float(time_arr[0])
        
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
        
        # --- CALIBRATION PROTOCOLS ---
        # Protocol A: Zero bias
        b_proto_a = 0.0
        
        # Protocol B: Prior calibration (from initial stop at drive start)
        b_proto_b = PRIOR_CALIBRATION_BIASES[drive_key]
        
        # Protocol C: Causal online calibration (most recent stop prior to outage)
        b_proto_c, last_stop_s, stop_dur_s = get_causal_online_bias(drive_df, st)
        
        # Record bias log
        bias_log_records.append({
            "scenario_id": sid,
            "driver_id": sc["driver_id"],
            "duration_s": dur,
            "outage_start_time_s": outage_start_s,
            "protocol_a_bias_rads": b_proto_a,
            "protocol_b_bias_rads": b_proto_b,
            "protocol_b_source": "Trip start initial stationary period",
            "protocol_c_bias_rads": b_proto_c,
            "protocol_c_last_stop_time_s": last_stop_s,
            "protocol_c_stop_duration_s": stop_dur_s,
        })
        
        protocols = {
            "Protocol_A_ZeroBias": b_proto_a,
            "Protocol_B_PriorCalib": b_proto_b,
            "Protocol_C_CausalOnline": b_proto_c,
        }
        
        # --- SPEED ESTIMATORS ---
        # 1. Constant-Velocity
        spd_cv = np.full(n_steps, last_spd, dtype=np.float64)
        
        # 2. ModernTCN (3-seed ensemble)
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
            
        # 3. Existing Hybrid (Phase 13: 1-sample spike braking)
        accel_x_in = in_outage_df["accel_x"].values
        spd_hyb_exist = compute_hybrid_speed(
            time_arr, last_spd, spd_mtcn, accel_x=accel_x_in, t_switch=10.0, tau=2.0
        )
        
        # 4. Corrected Causal Hybrid (Phase 14: sustained rolling braking)
        spd_hyb_causal = compute_causal_hybrid_speed(
            time_arr, last_spd, spd_mtcn, accel_x=accel_x_in, t_switch=10.0, tau=2.0, window_len=5, brake_thr=-1.5
        )
        
        estimators = {
            "Constant-Velocity": spd_cv,
            "ModernTCN": spd_mtcn,
            "Existing-Hybrid": spd_hyb_exist,
            "Corrected-Causal-Hybrid": spd_hyb_causal,
        }
        
        # Run dead reckoning for each protocol x estimator
        for proto_name, b_val in protocols.items():
            for est_name, spd_vec in estimators.items():
                res = run_causal_dead_reckoning(
                    pre_state, in_outage_df, spd_vec, gyro_bias_val=b_val
                )
                metrics_records.append({
                    "scenario_id": sid,
                    "driver_id": sc["driver_id"],
                    "duration_s": dur,
                    "calibration_protocol": proto_name,
                    "estimator": est_name,
                    "bias_rads": b_val,
                    **res,
                })
                
    elapsed = time.time() - t0_start
    print(f"Evaluation finished in {elapsed:.1f} s.")
    
    # Save per-scenario metrics
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(out_dir / "per_scenario_metrics.csv", index=False)
    print(f"Saved {out_dir / 'per_scenario_metrics.csv'}")
    
    # Save bias log
    df_bias = pd.DataFrame(bias_log_records)
    df_bias.to_csv(out_dir / "bias_estimation_log.csv", index=False)
    print(f"Saved {out_dir / 'bias_estimation_log.csv'}")
    
    # Summary metrics across sessions and pooled
    summary_list = []
    
    for sess in ["Driver B", "Driver D", "POOLED"]:
        sess_df = df_metrics if sess == "POOLED" else df_metrics[df_metrics["driver_id"] == sess]
        
        for proto in ["Protocol_A_ZeroBias", "Protocol_B_PriorCalib", "Protocol_C_CausalOnline"]:
            for dur in [10, 30, 60]:
                for est in ["Constant-Velocity", "ModernTCN", "Existing-Hybrid", "Corrected-Causal-Hybrid"]:
                    sub = sess_df[
                        (sess_df["calibration_protocol"] == proto) &
                        (sess_df["duration_s"] == dur) &
                        (sess_df["estimator"] == est)
                    ]
                    if len(sub) == 0: continue
                    
                    drifts = sub["final_drift_m"].values
                    speed_maes = sub["speed_mae_mps"].values
                    ate_rmses = sub["ate_rmse_m"].values
                    head_errs = sub["final_heading_err_deg"].values
                    
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
                        "mean_speed_mae_mps": float(np.mean(speed_maes)),
                        "mean_ate_rmse_m": float(np.mean(ate_rmses)),
                        "mean_heading_err_deg": float(np.mean(head_errs)),
                    })
                    
    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    print(f"Saved {out_dir / 'summary_metrics.csv'}")
    
    # Save Config JSON
    config = {
        "evaluation_timestamp": "2026-10-30",
        "num_scenarios": len(scenarios),
        "protocols": {
            "Protocol_A": "Zero-bias baseline (b = 0.0)",
            "Protocol_B": "Prior calibration from initial stationary stop at trip start (frozen)",
            "Protocol_C": "Causal online calibration from verified stationary stop prior to outage",
        },
        "prior_calibration_values": PRIOR_CALIBRATION_BIASES,
        "hybrid_parameters": {
            "existing": {"t_switch_s": 10.0, "tau_s": 2.0, "braking_filter": "instantaneous_1sample", "thr": -1.5},
            "corrected_causal": {"t_switch_s": 10.0, "tau_s": 2.0, "braking_filter": "causal_rolling_5sample_0.5s", "thr": -1.5},
        },
        "models": ["moderntcnmodel_seed42", "moderntcnmodel_seed123", "moderntcnmodel_seed456"],
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved {out_dir / 'config.json'}")

if __name__ == "__main__":
    main()
