import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.evaluate_phase10_independent import load_model
from src.corrected_synchronization import extract_aligned_segments_for_drive
from src.outage_estimator import PreOutageState, estimate_initial_heading_from_pre_outage_gps
from experiments.phase14_validation.evaluate_phase14_causal import (
    get_causal_online_bias,
    compute_causal_hybrid_speed,
    run_causal_dead_reckoning,
    PRIOR_CALIBRATION_BIASES,
)

def get_causal_accel_bias(drive_df, outage_start_idx, min_stat_samples=30):
    """Finds the most recent stationary period strictly prior to outage_start_idx to estimate accel_x bias."""
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

def test_baselines_pilot():
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
    
    import torch
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
        
        # Gyro and accel bias under Protocol C
        b_gyro_c, _, _ = get_causal_online_bias(drive_df, st)
        b_accel_c = get_causal_accel_bias(drive_df, st)
        
        accel_x_in = in_outage_df["accel_x"].values
        
        # 1. Constant-Velocity
        spd_cv = np.full(n_steps, last_spd, dtype=np.float64)
        
        # 2. Pure Acceleration Integration
        spd_accel = np.zeros(n_steps, dtype=np.float64)
        spd_accel[0] = last_spd
        for k in range(1, n_steps):
            dt = dt_arr[k]
            # Integrate forward acceleration with bias removal
            spd_accel[k] = max(0.0, spd_accel[k-1] + (accel_x_in[k] - b_accel_c) * dt)
            
        # 3. ModernTCN
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
            
        # 4. Existing Causal Hybrid (Phase 14)
        spd_hyb_p14 = compute_causal_hybrid_speed(
            time_arr, last_spd, spd_mtcn, accel_x=accel_x_in, t_switch=10.0, tau=2.0, window_len=5, brake_thr=-1.5
        )
        
        # 5. Constrained Causal Hybrid (with slew-rate limit: max accel 4 m/s^2, max decel 8 m/s^2)
        spd_hyb_constrained = np.zeros(n_steps, dtype=np.float64)
        spd_hyb_constrained[0] = last_spd
        for k in range(1, n_steps):
            dt = dt_arr[k]
            val = spd_hyb_p14[k]
            v_max = spd_hyb_constrained[k-1] + 4.0 * dt
            v_min = max(0.0, spd_hyb_constrained[k-1] - 8.0 * dt)
            spd_hyb_constrained[k] = np.clip(val, v_min, v_max)
            
        # 6. Complementary Filter (fusing acceleration high-frequency with hybrid low-frequency anchor)
        # gamma = 0.15 gives time constant ~0.67s
        spd_complementary = np.zeros(n_steps, dtype=np.float64)
        spd_complementary[0] = last_spd
        gamma = 0.15
        for k in range(1, n_steps):
            dt = dt_arr[k]
            v_inertial = spd_complementary[k-1] + (accel_x_in[k] - b_accel_c) * dt
            v_anchor = spd_hyb_constrained[k]
            v_fused = (1.0 - gamma) * v_inertial + gamma * v_anchor
            spd_complementary[k] = max(0.0, v_fused)
            
        estimators = {
            "1_Constant-Velocity": spd_cv,
            "2_Acceleration-Integration": spd_accel,
            "3_ModernTCN": spd_mtcn,
            "4_Phase14-Causal-Hybrid": spd_hyb_p14,
            "5_Constrained-Causal-Hybrid": spd_hyb_constrained,
            "6_Complementary-Filter": spd_complementary,
        }
        
        for est_name, spd_vec in estimators.items():
            res = run_causal_dead_reckoning(
                pre_state, in_outage_df, spd_vec, gyro_bias_val=b_gyro_c
            )
            records.append({
                "scenario_id": sid,
                "driver_id": sc["driver_id"],
                "duration_s": dur,
                "estimator": est_name,
                **res
            })
            
    df_res = pd.DataFrame(records)
    print("\n--- BASELINE EVALUATION PILOT (Protocol C - Causal Online Bias) ---")
    summary = df_res.groupby(["duration_s", "estimator"])[["final_drift_m", "speed_mae_mps", "ate_rmse_m"]].agg(["mean", "median"])
    print(summary.to_string())

if __name__ == "__main__":
    test_baselines_pilot()
