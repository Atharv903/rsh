"""
Phase 16 Controlled Ablation Study:
Uncertainty-Aware ESKF, Non-Holonomic Constraints & Competitive Benchmarking.

Evaluates 8 Controlled Configurations:
A. Constant-Velocity (CV baseline)
B. ModernTCN (3-seed ensemble, forward Euler)
C. Phase 14 Causal Hybrid (forward Euler)
D. ESKF-IMU-Only (pure inertial propagation with accelerometer & gyro)
E. ESKF-Learned-Speed (uncertainty-aware ModernTCN fusion, no NHC)
F. ESKF-Speed-NHC (learned speed + Non-Holonomic Constraint v_lat ~ 0)
G. ESKF-Speed-NHC-ZUPT (learned speed + NHC + causal ZUPT)
H. ESKF-Hybrid-NHC-ZUPT (Phase 14 hybrid speed + NHC + causal ZUPT)

Across 3 Calibration Protocols:
- Protocol A: Zero-Bias (b_gyro = 0, b_accel = 0)
- Protocol B: Prior Calibration (frozen initial stationary stop)
- Protocol C: Causal Online Calibration (updated prior to outage onset)

Across 30 Standardized Scenarios (Driver B and Driver D; 10s, 30s, 60s).
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
from src.eskf import VehicleESKF
from experiments.phase14_validation.evaluate_phase14_causal import (
    get_causal_online_bias,
    compute_causal_hybrid_speed,
    PRIOR_CALIBRATION_BIASES,
)
from experiments.phase15_validation.evaluate_phase15 import (
    get_causal_online_accel_bias,
    PRIOR_CALIBRATION_ACCEL_BIASES,
)

def evaluate_eskf_trajectory(
    eskf: VehicleESKF,
    in_outage_df: pd.DataFrame,
    speed_predictions: np.ndarray,
    speed_variances: np.ndarray,
    accel_bias_mps2: float = 0.0,
    use_accel_propagation: bool = False,
    enable_speed_update: bool = True,
    enable_nhc: bool = False,
    enable_zupt: bool = False,
    var_nhc: float = 0.10,          # (0.31 m/s)^2
    nhc_gate_threshold: float = 9.0, # 3-sigma gate
):
    n_steps = len(in_outage_df)
    time_arr = in_outage_df["time_s"].values
    dt_arr = np.diff(time_arr, prepend=time_arr[0])
    dt_arr[0] = 0.0
    
    ref_east = in_outage_df["ref_east_m"].values
    ref_north = in_outage_df["ref_north_m"].values
    ref_speed = in_outage_df["ref_speed_mps"].values
    ref_heading = np.deg2rad(in_outage_df["ref_heading_deg"].values)
    
    accel_x = in_outage_df["accel_x"].values
    gyro_pitch = in_outage_df["gyro_pitch"].values
    
    pe_hist = np.zeros(n_steps, dtype=np.float64)
    pn_hist = np.zeros(n_steps, dtype=np.float64)
    head_hist = np.zeros(n_steps, dtype=np.float64)
    spd_fwd_hist = np.zeros(n_steps, dtype=np.float64)
    
    pe_hist[0] = eskf.state.p_e
    pn_hist[0] = eskf.state.p_n
    head_hist[0] = eskf.state.psi
    spd_fwd_hist[0] = eskf.state.v_e * np.sin(eskf.state.psi) + eskf.state.v_n * np.cos(eskf.state.psi)
    
    # Causal stop detector: rolling 5 samples (0.5s)
    for k in range(1, n_steps):
        dt = dt_arr[k]
        
        # 1. Predict
        eskf.predict(
            dt=dt,
            gyro_pitch_rads=gyro_pitch[k],
            accel_x_mps2=accel_x[k],
            accel_bias_mps2=accel_bias_mps2,
            use_accel_propagation=use_accel_propagation,
        )
        
        # 2. Speed update
        if enable_speed_update and speed_predictions is not None:
            eskf.update_speed(
                v_pred_mps=speed_predictions[k],
                var_v=speed_variances[k],
                gate_threshold=16.0,
            )
            
        # 3. Non-Holonomic Constraint (NHC)
        if enable_nhc:
            eskf.update_nhc(
                var_nhc=var_nhc,
                gate_threshold=nhc_gate_threshold,
            )
            
        # 4. Zero-Velocity Update (ZUPT)
        if enable_zupt and k >= 5:
            # Check causal acceleration variance and predicted speed
            acc_window = accel_x[k-5 : k+1]
            if np.var(acc_window) < 0.05 and speed_predictions[k] < 0.2:
                eskf.update_zupt(var_zupt=0.0025)
                
        pe_hist[k] = eskf.state.p_e
        pn_hist[k] = eskf.state.p_n
        head_hist[k] = eskf.state.psi
        spd_fwd_hist[k] = eskf.state.v_e * np.sin(eskf.state.psi) + eskf.state.v_n * np.cos(eskf.state.psi)
        
    pos_err = np.sqrt((pe_hist - ref_east)**2 + (pn_hist - ref_north)**2)
    final_drift_m = float(pos_err[-1])
    ate_rmse_m = float(np.sqrt(np.mean(pos_err**2)))
    
    speed_err = spd_fwd_hist - ref_speed
    speed_mae = float(np.mean(np.abs(speed_err)))
    speed_rmse = float(np.sqrt(np.mean(speed_err**2)))
    
    delta_e = pe_hist[-1] - ref_east[-1]
    delta_n = pn_hist[-1] - ref_north[-1]
    psi_f = ref_heading[-1]
    u_along = np.array([np.sin(psi_f), np.cos(psi_f)])
    u_cross = np.array([np.cos(psi_f), -np.sin(psi_f)])
    along_err = float(delta_e * u_along[0] + delta_n * u_along[1])
    cross_err = float(delta_e * u_cross[0] + delta_n * u_cross[1])
    
    head_diff = np.arctan2(np.sin(head_hist - ref_heading), np.cos(head_hist - ref_heading))
    final_heading_err_deg = float(np.rad2deg(np.abs(head_diff[-1])))
    mean_heading_err_deg = float(np.rad2deg(np.mean(np.abs(head_diff))))
    
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
    print("PHASE 16: UNCERTAINTY-AWARE ESKF & CONTROLLED ABLATION BENCHMARK")
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
        
    out_dir = repo_root / "experiments/phase16_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_records = []
    diag_records = []
    
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
        heading_rad = np.deg2rad(heading_deg)
        
        # 1. Compute ModernTCN predictions and ensemble variance
        spd_mtcn = np.zeros(n_steps, dtype=np.float64)
        var_mtcn = np.zeros(n_steps, dtype=np.float64)
        spd_mtcn[0] = last_spd
        var_mtcn[0] = 0.25 # Initial speed variance
        
        for k in range(1, n_steps):
            w_start = (st + k) - 100
            w_end = st + k
            w_raw = drive_df.iloc[w_start:w_end][feat_channels].values.astype(np.float32)
            w_norm = (w_raw - norm_mean) / norm_std
            inp_t = torch.from_numpy(w_norm).unsqueeze(0)
            with torch.no_grad():
                preds = [m(inp_t).item() for m in mtcn_models]
            p_mean = max(0.0, float(np.mean(preds)))
            p_var = float(np.var(preds))
            spd_mtcn[k] = p_mean
            # Uncertainty = base validation residual variance (1.5^2 = 2.25) + 4.0 * ensemble variance
            var_mtcn[k] = 2.25 + 4.0 * p_var
            
        # 2. Compute Phase 14 Causal Hybrid speed
        accel_x_in = in_outage_df["accel_x"].values
        spd_hyb = compute_causal_hybrid_speed(
            time_arr, last_spd, spd_mtcn, accel_x=accel_x_in, t_switch=10.0, tau=2.0, window_len=5, brake_thr=-1.5
        )
        # Hybrid uncertainty: transitions from small initial variance to model variance
        alpha_t = 1.0 / (1.0 + np.exp(-((time_arr - time_arr[0]) - 10.0) / 2.0))
        var_hyb = (1.0 - alpha_t) * 0.25 + alpha_t * var_mtcn
        
        # Protocols
        b_gyro_a, b_accel_a = 0.0, 0.0
        b_gyro_b = PRIOR_CALIBRATION_BIASES[drive_key]
        b_accel_b = PRIOR_CALIBRATION_ACCEL_BIASES[drive_key]
        b_gyro_c, _, _ = get_causal_online_bias(drive_df, st)
        b_accel_c = get_causal_online_accel_bias(drive_df, st)
        
        protocols = {
            "Protocol_A_ZeroBias": {"gyro": b_gyro_a, "accel": b_accel_a},
            "Protocol_B_PriorCalib": {"gyro": b_gyro_b, "accel": b_accel_b},
            "Protocol_C_CausalOnline": {"gyro": b_gyro_c, "accel": b_accel_c},
        }
        
        for proto_name, b_dict in protocols.items():
            b_gyro = b_dict["gyro"]
            b_accel = b_dict["accel"]
            
            # --- CONFIGURATIONS A, B, C: Forward Euler Baselines ---
            from src.outage_estimator import PreOutageState
            from experiments.phase14_validation.evaluate_phase14_causal import run_causal_dead_reckoning
            pre_state = PreOutageState(
                t0_s=float(pre_df["time_s"].iloc[-1]),
                pos_east_m=pos_e0, pos_north_m=pos_n0,
                speed_mps=last_spd, heading_deg=heading_deg,
                gyro_bias_rads=0.0,
            )
            
            # Config A: Constant-Velocity
            spd_cv = np.full(n_steps, last_spd, dtype=np.float64)
            res_cv = run_causal_dead_reckoning(pre_state, in_outage_df, spd_cv, gyro_bias_val=b_gyro)
            metrics_records.append({
                "scenario_id": sid, "driver_id": sc["driver_id"], "duration_s": dur,
                "calibration_protocol": proto_name, "config_id": "A_Constant-Velocity",
                "estimator_name": "Constant-Velocity", **res_cv
            })
            
            # Config B: ModernTCN
            res_mtcn = run_causal_dead_reckoning(pre_state, in_outage_df, spd_mtcn, gyro_bias_val=b_gyro)
            metrics_records.append({
                "scenario_id": sid, "driver_id": sc["driver_id"], "duration_s": dur,
                "calibration_protocol": proto_name, "config_id": "B_ModernTCN",
                "estimator_name": "ModernTCN", **res_mtcn
            })
            
            # Config C: Phase 14 Causal Hybrid
            res_hyb = run_causal_dead_reckoning(pre_state, in_outage_df, spd_hyb, gyro_bias_val=b_gyro)
            metrics_records.append({
                "scenario_id": sid, "driver_id": sc["driver_id"], "duration_s": dur,
                "calibration_protocol": proto_name, "config_id": "C_Phase14-Causal-Hybrid",
                "estimator_name": "Phase14-Causal-Hybrid", **res_hyb
            })
            
            # --- CONFIGURATIONS D, E, F, G, H: ESKF Configurations ---
            def make_filter():
                return VehicleESKF(
                    init_p_e=pos_e0, init_p_n=pos_n0,
                    init_speed_mps=last_spd, init_heading_rad=heading_rad,
                    init_gyro_bias_rads=b_gyro,
                    sigma_pos_init=1.0, sigma_vel_init=0.5,
                    sigma_psi_init=np.deg2rad(1.5), sigma_bg_init=0.002,
                    q_pos=0.01, q_vel=0.20, q_psi=0.002, q_bg=1e-6,
                    h_sign=-1.0,
                )
                
            # Config D: ESKF-IMU-Only (pure inertial propagation using accel + gyro)
            eskf_d = make_filter()
            res_d = evaluate_eskf_trajectory(
                eskf_d, in_outage_df, speed_predictions=None, speed_variances=None,
                accel_bias_mps2=b_accel, use_accel_propagation=True,
                enable_speed_update=False, enable_nhc=False, enable_zupt=False,
            )
            metrics_records.append({
                "scenario_id": sid, "driver_id": sc["driver_id"], "duration_s": dur,
                "calibration_protocol": proto_name, "config_id": "D_ESKF-IMU-Only",
                "estimator_name": "ESKF-IMU-Only", **res_d
            })
            
            # Config E: ESKF-Learned-Speed (uncertainty-aware ModernTCN fusion, no NHC)
            eskf_e = make_filter()
            res_e = evaluate_eskf_trajectory(
                eskf_e, in_outage_df, speed_predictions=spd_mtcn, speed_variances=var_mtcn,
                accel_bias_mps2=b_accel, use_accel_propagation=False,
                enable_speed_update=True, enable_nhc=False, enable_zupt=False,
            )
            metrics_records.append({
                "scenario_id": sid, "driver_id": sc["driver_id"], "duration_s": dur,
                "calibration_protocol": proto_name, "config_id": "E_ESKF-Learned-Speed",
                "estimator_name": "ESKF-Learned-Speed", **res_e
            })
            
            # Config F: ESKF-Speed-NHC (learned speed + NHC v_lat ~ 0)
            eskf_f = make_filter()
            res_f = evaluate_eskf_trajectory(
                eskf_f, in_outage_df, speed_predictions=spd_mtcn, speed_variances=var_mtcn,
                accel_bias_mps2=b_accel, use_accel_propagation=False,
                enable_speed_update=True, enable_nhc=True, enable_zupt=False,
            )
            metrics_records.append({
                "scenario_id": sid, "driver_id": sc["driver_id"], "duration_s": dur,
                "calibration_protocol": proto_name, "config_id": "F_ESKF-Speed-NHC",
                "estimator_name": "ESKF-Speed-NHC", **res_f
            })
            
            # Config G: ESKF-Speed-NHC-ZUPT (learned speed + NHC + causal ZUPT)
            eskf_g = make_filter()
            res_g = evaluate_eskf_trajectory(
                eskf_g, in_outage_df, speed_predictions=spd_mtcn, speed_variances=var_mtcn,
                accel_bias_mps2=b_accel, use_accel_propagation=False,
                enable_speed_update=True, enable_nhc=True, enable_zupt=True,
            )
            metrics_records.append({
                "scenario_id": sid, "driver_id": sc["driver_id"], "duration_s": dur,
                "calibration_protocol": proto_name, "config_id": "G_ESKF-Speed-NHC-ZUPT",
                "estimator_name": "ESKF-Speed-NHC-ZUPT", **res_g
            })
            
            # Config H: ESKF-Hybrid-NHC-ZUPT (Phase 14 hybrid speed + NHC + causal ZUPT)
            eskf_h = make_filter()
            res_h = evaluate_eskf_trajectory(
                eskf_h, in_outage_df, speed_predictions=spd_hyb, speed_variances=var_hyb,
                accel_bias_mps2=b_accel, use_accel_propagation=False,
                enable_speed_update=True, enable_nhc=True, enable_zupt=True,
            )
            metrics_records.append({
                "scenario_id": sid, "driver_id": sc["driver_id"], "duration_s": dur,
                "calibration_protocol": proto_name, "config_id": "H_ESKF-Hybrid-NHC-ZUPT",
                "estimator_name": "ESKF-Hybrid-NHC-ZUPT", **res_h
            })
            
            # Log diagnostics for Config H
            diag_records.append({
                "scenario_id": sid, "driver_id": sc["driver_id"], "duration_s": dur,
                "calibration_protocol": proto_name, "config_id": "H_ESKF-Hybrid-NHC-ZUPT",
                "speed_updates_accepted": eskf_h.diagnostics["speed_updates_accepted"],
                "speed_updates_rejected": eskf_h.diagnostics["speed_updates_rejected"],
                "nhc_updates_accepted": eskf_h.diagnostics["nhc_updates_accepted"],
                "nhc_updates_rejected": eskf_h.diagnostics["nhc_updates_rejected"],
                "zupt_updates_accepted": eskf_h.diagnostics["zupt_updates_accepted"],
                "mean_speed_innovation": float(np.mean(eskf_h.diagnostics["innovations_speed"])) if eskf_h.diagnostics["innovations_speed"] else 0.0,
                "mean_nhc_innovation": float(np.mean(eskf_h.diagnostics["innovations_nhc"])) if eskf_h.diagnostics["innovations_nhc"] else 0.0,
                "final_cov_trace": eskf_h.diagnostics["covariance_trace"][-1] if eskf_h.diagnostics["covariance_trace"] else 0.0,
            })

    elapsed = time.time() - t0_start
    print(f"Phase 16 evaluation finished in {elapsed:.1f} s.")
    
    # Save per-scenario metrics
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(out_dir / "per_scenario_metrics.csv", index=False)
    print(f"Saved {out_dir / 'per_scenario_metrics.csv'} ({len(df_metrics)} records)")
    
    # Save diagnostics
    df_diag = pd.DataFrame(diag_records)
    df_diag.to_csv(out_dir / "filter_diagnostics.csv", index=False)
    print(f"Saved {out_dir / 'filter_diagnostics.csv'}")
    
    # Generate summary metrics
    summary_list = []
    configs = [
        "A_Constant-Velocity", "B_ModernTCN", "C_Phase14-Causal-Hybrid",
        "D_ESKF-IMU-Only", "E_ESKF-Learned-Speed", "F_ESKF-Speed-NHC",
        "G_ESKF-Speed-NHC-ZUPT", "H_ESKF-Hybrid-NHC-ZUPT"
    ]
    
    for sess in ["Driver B", "Driver D", "POOLED"]:
        sess_df = df_metrics if sess == "POOLED" else df_metrics[df_metrics["driver_id"] == sess]
        
        for proto in ["Protocol_A_ZeroBias", "Protocol_B_PriorCalib", "Protocol_C_CausalOnline"]:
            for dur in [10, 30, 60]:
                for cfg in configs:
                    sub = sess_df[
                        (sess_df["calibration_protocol"] == proto) &
                        (sess_df["duration_s"] == dur) &
                        (sess_df["config_id"] == cfg)
                    ]
                    if len(sub) == 0: continue
                    
                    drifts = sub["final_drift_m"].values
                    speed_maes = sub["speed_mae_mps"].values
                    ate_rmses = sub["ate_rmse_m"].values
                    head_errs = sub["final_heading_err_deg"].values
                    along_errs = sub["along_track_err_m"].values
                    cross_errs = sub["cross_track_err_m"].values
                    dist_pcts = sub["drift_pct_of_distance"].values
                    
                    summary_list.append({
                        "session": sess,
                        "calibration_protocol": proto,
                        "duration_s": dur,
                        "config_id": cfg,
                        "estimator_name": sub["estimator_name"].iloc[0],
                        "n_scenarios": len(sub),
                        "mean_final_drift_m": float(np.mean(drifts)),
                        "median_final_drift_m": float(np.median(drifts)),
                        "std_final_drift_m": float(np.std(drifts, ddof=1)),
                        "iqr_final_drift_m": float(np.percentile(drifts, 75) - np.percentile(drifts, 25)),
                        "mean_along_track_err_m": float(np.mean(along_errs)),
                        "mean_cross_track_err_m": float(np.mean(cross_errs)),
                        "mean_speed_mae_mps": float(np.mean(speed_maes)),
                        "mean_ate_rmse_m": float(np.mean(ate_rmses)),
                        "mean_heading_err_deg": float(np.mean(head_errs)),
                        "mean_drift_pct_of_distance": float(np.mean(dist_pcts)),
                        "median_drift_pct_of_distance": float(np.median(dist_pcts)),
                    })
                    
    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    print(f"Saved {out_dir / 'summary_metrics.csv'} ({len(df_summary)} rows)")
    
    # Save config.json
    config_dict = {
        "phase": "Phase 16: Uncertainty-Aware ESKF & Non-Holonomic Constraints",
        "timestamp": "2026-10-30",
        "baseline_commit": "3c5a181",
        "num_scenarios": len(scenarios),
        "configurations": configs,
        "protocols": list(protocols.keys()),
        "eskf_parameters": {
            "state_dimension": 6,
            "state_variables": ["p_e", "p_n", "v_e", "v_n", "psi", "b_g"],
            "process_noise": {"q_pos": 0.01, "q_vel": 0.20, "q_psi": 0.002, "q_bg": 1e-6},
            "measurement_noise": {"var_nhc": 0.10, "var_zupt": 0.0025},
            "innovation_gates": {"speed": 16.0, "nhc": 9.0},
        }
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved {out_dir / 'config.json'}")

if __name__ == "__main__":
    main()
