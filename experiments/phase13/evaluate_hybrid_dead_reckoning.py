"""
Phase 13D: Experimental Evaluation of Adaptive Hybrid Dead Reckoning.

Evaluates:
1. Constant-Velocity (Baseline)
2. ModernTCN (Neural speed estimator)
3. Adaptive-Hybrid (Dynamic blending with braking override)

Across:
- Driver B (15 scenarios)
- Driver D (15 scenarios)
Under both causal stationary bias handling and legacy bias handling.
"""

import json
import sys
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

def get_causal_stationary_bias(drive_df, outage_start_idx, min_stat_samples=30):
    pre_data = drive_df.iloc[:outage_start_idx]
    is_stat = (pre_data["ref_speed_mps"] < 0.1).values
    curr_end = None
    for i in range(len(is_stat) - 1, -1, -1):
        if is_stat[i] and curr_end is None:
            curr_end = i
        elif not is_stat[i] and curr_end is not None:
            if (curr_end - i) >= min_stat_samples:
                stat_slice = pre_data.iloc[i+1 : curr_end+1]
                return float(stat_slice["gyro_pitch"].mean())
            curr_end = None
    return 0.0

def run_causal_dr(
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
    print("PHASE 13D: EVALUATING ADAPTIVE HYBRID DEAD RECKONING")
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
    
    # Load ModernTCN models across 3 seeds to compute ensemble/mean
    ckpt_dir = repo_root / "experiments/phase8_corrected_speed_models/checkpoints"
    mtcn_models = []
    for s in [42, 123, 456]:
        m, _ = load_model(ckpt_dir / f"moderntcnmodel_seed{s}/best_model.pt")
        mtcn_models.append(m)
        
    records = []
    
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
        
        heading_deg, last_spd = estimate_initial_heading_from_pre_outage_gps(pre_df)
        pos_e0 = float(pre_df["ref_east_m"].iloc[-1])
        pos_n0 = float(pre_df["ref_north_m"].iloc[-1])
        
        # Biases
        b_causal = get_causal_stationary_bias(drive_df, st)
        b_legacy = float(np.median(pre_df["gyro_pitch"]))
        
        pre_state = PreOutageState(
            t0_s=float(pre_df["time_s"].iloc[-1]),
            pos_east_m=pos_e0,
            pos_north_m=pos_n0,
            speed_mps=last_spd,
            heading_deg=heading_deg,
            gyro_bias_rads=0.0,
        )
        
        # 1. Constant Velocity Speed
        spd_cv = np.full(n_steps, last_spd, dtype=np.float64)
        
        # 2. ModernTCN Speed (ensemble across 3 seeds)
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
            
        # 3. Adaptive Hybrid Speed
        accel_x_in = in_outage_df["accel_x"].values
        spd_hybrid = compute_hybrid_speed(
            time_arr, last_spd, spd_mtcn, accel_x=accel_x_in, t_switch=10.0, tau=2.0
        )
        
        eval_methods = {
            "Constant-Velocity": spd_cv,
            "ModernTCN": spd_mtcn,
            "Adaptive-Hybrid": spd_hybrid,
        }
        
        bias_strategies = {
            "causal_pre_stationary": b_causal,
            "legacy_pre_median": b_legacy,
        }
        
        for m_name, spd_vec in eval_methods.items():
            for b_name, b_val in bias_strategies.items():
                res = run_causal_dr(pre_state, in_outage_df, spd_vec, gyro_bias_val=b_val)
                records.append({
                    "scenario_id": sid,
                    "driver_id": sc["driver_id"],
                    "duration_s": dur,
                    "method": m_name,
                    "bias_strategy": b_name,
                    **res,
                })
                
    res_df = pd.DataFrame(records)
    out_dir = repo_root / "experiments/phase13"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "hybrid_evaluation_per_scenario.csv", index=False)
    
    # Summary Table
    summary = res_df.groupby(["bias_strategy", "duration_s", "method"])[
        ["speed_mae_mps", "ate_rmse_m", "final_drift_m", "along_track_err_m", "cross_track_err_m", "final_heading_err_deg"]
    ].mean().reset_index()
    
    summary.to_csv(out_dir / "hybrid_evaluation_summary.csv", index=False)
    print("\nADAPTIVE HYBRID EVALUATION SUMMARY:")
    print(summary.to_string())

if __name__ == "__main__":
    main()
