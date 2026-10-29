import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def run_diagnostics():
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
    
    # Load 3 ModernTCN models
    import torch
    ckpt_dir = repo_root / "experiments/phase8_corrected_speed_models/checkpoints"
    mtcn_models = []
    for s in [42, 123, 456]:
        m, _ = load_model(ckpt_dir / f"moderntcnmodel_seed{s}/best_model.pt")
        mtcn_models.append(m)
        
    out_dir = repo_root / "experiments/phase15_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for time-series traces (for 60s scenarios)
    traces_60s = []
    regime_records = []
    ablation_records = []
    
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
        rel_time = time_arr - time_arr[0]
        dt_arr = np.diff(time_arr, prepend=time_arr[0])
        dt_arr[0] = 0.0
        
        heading_deg, last_spd = estimate_initial_heading_from_pre_outage_gps(pre_df)
        pos_e0 = float(pre_df["ref_east_m"].iloc[-1])
        pos_n0 = float(pre_df["ref_north_m"].iloc[-1])
        
        ref_east = in_outage_df["ref_east_m"].values
        ref_north = in_outage_df["ref_north_m"].values
        ref_speed = in_outage_df["ref_speed_mps"].values
        ref_heading = np.deg2rad(in_outage_df["ref_heading_deg"].values)
        
        # Protocol C bias
        b_proto_c, _, _ = get_causal_online_bias(drive_df, st)
        gyro_yaw_rate = in_outage_df["gyro_pitch"].values.astype(np.float32) - b_proto_c
        
        # 1. Constant-Velocity speed
        spd_cv = np.full(n_steps, last_spd, dtype=np.float64)
        
        # 2. ModernTCN speed
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
            
        # 3. Phase 14 Causal Hybrid speed
        accel_x_in = in_outage_df["accel_x"].values
        spd_hyb = compute_causal_hybrid_speed(
            time_arr, last_spd, spd_mtcn, accel_x=accel_x_in, t_switch=10.0, tau=2.0, window_len=5, brake_thr=-1.5
        )
        
        # 4. Integrate Trajectories for Diagnosis
        def integrate(spd_vector, gyro_rate, h_sign=-1.0):
            pe = np.zeros(n_steps, dtype=np.float64)
            pn = np.zeros(n_steps, dtype=np.float64)
            head = np.zeros(n_steps, dtype=np.float64)
            pe[0], pn[0], head[0] = pos_e0, pos_n0, np.deg2rad(heading_deg)
            for k in range(1, n_steps):
                dt = dt_arr[k]
                d_psi = h_sign * float(gyro_rate[k] * dt)
                head[k] = (head[k-1] + d_psi) % (2.0 * np.pi)
                v_e = spd_vector[k] * np.sin(head[k])
                v_n = spd_vector[k] * np.cos(head[k])
                pe[k] = pe[k-1] + v_e * dt
                pn[k] = pn[k-1] + v_n * dt
            return pe, pn, head

        pe_hyb, pn_hyb, head_hyb = integrate(spd_hyb, gyro_yaw_rate)
        pe_mtcn, pn_mtcn, head_mtcn = integrate(spd_mtcn, gyro_yaw_rate)
        pe_cv, pn_cv, head_cv = integrate(spd_cv, gyro_yaw_rate)
        
        # Ablations:
        # A: Perfect speed, realistic gyro
        pe_perf_spd, pn_perf_spd, _ = integrate(ref_speed, gyro_yaw_rate)
        # B: Realistic hybrid speed, perfect heading
        pe_perf_head = np.zeros(n_steps)
        pn_perf_head = np.zeros(n_steps)
        pe_perf_head[0], pn_perf_head[0] = pos_e0, pos_n0
        for k in range(1, n_steps):
            dt = dt_arr[k]
            pe_perf_head[k] = pe_perf_head[k-1] + spd_hyb[k] * np.sin(ref_heading[k]) * dt
            pn_perf_head[k] = pn_perf_head[k-1] + spd_hyb[k] * np.cos(ref_heading[k]) * dt
            
        drift_hyb = np.sqrt((pe_hyb[-1] - ref_east[-1])**2 + (pn_hyb[-1] - ref_north[-1])**2)
        drift_perf_spd = np.sqrt((pe_perf_spd[-1] - ref_east[-1])**2 + (pn_perf_spd[-1] - ref_north[-1])**2)
        drift_perf_head = np.sqrt((pe_perf_head[-1] - ref_east[-1])**2 + (pn_perf_head[-1] - ref_north[-1])**2)
        
        ablation_records.append({
            "scenario_id": sid,
            "duration_s": dur,
            "driver_id": sc["driver_id"],
            "drift_hybrid_actual_m": drift_hyb,
            "drift_perfect_speed_m": drift_perf_spd,   # isolate heading/gyro error
            "drift_perfect_heading_m": drift_perf_head, # isolate speed estimation error
        })
        
        # Analyze error regimes across dynamic phases
        accel_long = drive_df.iloc[st:end]["accel_x"].values
        is_accel = accel_long > 0.5
        is_brake = accel_long < -0.5
        is_cruise = (~is_accel) & (~is_brake)
        
        err_spd_mtcn = np.abs(spd_mtcn - ref_speed)
        err_spd_cv = np.abs(spd_cv - ref_speed)
        err_spd_hyb = np.abs(spd_hyb - ref_speed)
        
        regime_records.append({
            "scenario_id": sid,
            "duration_s": dur,
            "driver_id": sc["driver_id"],
            "speed_mae_overall": np.mean(err_spd_hyb),
            "speed_mae_cruise": np.mean(err_spd_hyb[is_cruise]) if np.any(is_cruise) else np.nan,
            "speed_mae_accel": np.mean(err_spd_hyb[is_accel]) if np.any(is_accel) else np.nan,
            "speed_mae_brake": np.mean(err_spd_hyb[is_brake]) if np.any(is_brake) else np.nan,
            "init_speed_error_mps": abs(last_spd - ref_speed[0]),
            "init_heading_error_deg": np.rad2deg(np.abs(np.arctan2(np.sin(np.deg2rad(heading_deg) - ref_heading[0]), np.cos(np.deg2rad(heading_deg) - ref_heading[0]))))
        })
        
        if dur == 60:
            # Calculate along-track and cross-track error traces
            pos_err = np.sqrt((pe_hyb - ref_east)**2 + (pn_hyb - ref_north)**2)
            along_err = []
            cross_err = []
            head_err = []
            for k in range(n_steps):
                de = pe_hyb[k] - ref_east[k]
                dn = pn_hyb[k] - ref_north[k]
                psi = ref_heading[k]
                u_along = np.array([np.sin(psi), np.cos(psi)])
                u_cross = np.array([np.cos(psi), -np.sin(psi)])
                along_err.append(de * u_along[0] + dn * u_along[1])
                cross_err.append(de * u_cross[0] + dn * u_cross[1])
                h_diff = np.arctan2(np.sin(head_hyb[k] - psi), np.cos(head_hyb[k] - psi))
                head_err.append(np.rad2deg(np.abs(h_diff)))
                
            traces_60s.append({
                "scenario_id": sid,
                "driver_id": sc["driver_id"],
                "time_s": rel_time,
                "pos_err": pos_err,
                "along_err": np.array(along_err),
                "cross_err": np.array(cross_err),
                "head_err": np.array(head_err),
                "spd_err": err_spd_hyb,
            })

    # Save summary data
    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv(out_dir / "error_ablation_summary.csv", index=False)
    
    df_regimes = pd.DataFrame(regime_records)
    df_regimes.to_csv(out_dir / "error_regime_summary.csv", index=False)
    
    print(f"Computed ablation and regime summaries for {len(scenarios)} scenarios.")
    
    # Generate diagnostic plot of error growth over 60 seconds
    plt.figure(figsize=(14, 10))
    
    # 1. Total Positional Drift vs Time
    plt.subplot(2, 2, 1)
    mean_pos_err = np.mean([t["pos_err"] for t in traces_60s], axis=0)
    for t in traces_60s:
        plt.plot(t["time_s"], t["pos_err"], color="gray", alpha=0.3, linewidth=1.0)
    plt.plot(traces_60s[0]["time_s"], mean_pos_err, color="blue", linewidth=2.5, label="Mean Drift (Protocol C)")
    plt.title("Total Positional Drift vs Time (60s Outages)", fontsize=11, fontweight="bold")
    plt.xlabel("Elapsed Outage Time (s)")
    plt.ylabel("Positional Drift (m)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # 2. Along-track vs Cross-track Error Growth
    plt.subplot(2, 2, 2)
    mean_along = np.mean([np.abs(t["along_err"]) for t in traces_60s], axis=0)
    mean_cross = np.mean([np.abs(t["cross_err"]) for t in traces_60s], axis=0)
    plt.plot(traces_60s[0]["time_s"], mean_along, color="crimson", linewidth=2.2, label="|Along-Track Error| (Speed)")
    plt.plot(traces_60s[0]["time_s"], mean_cross, color="darkgreen", linewidth=2.2, label="|Cross-Track Error| (Heading)")
    plt.title("Error Decomposition: Along-Track vs Cross-Track", fontsize=11, fontweight="bold")
    plt.xlabel("Elapsed Outage Time (s)")
    plt.ylabel("Mean Absolute Error (m)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # 3. Speed MAE vs Time
    plt.subplot(2, 2, 3)
    mean_spd_err = np.mean([t["spd_err"] for t in traces_60s], axis=0)
    for t in traces_60s:
        plt.plot(t["time_s"], t["spd_err"], color="gray", alpha=0.25, linewidth=0.8)
    plt.plot(traces_60s[0]["time_s"], mean_spd_err, color="purple", linewidth=2.5, label="Mean Speed Error (m/s)")
    plt.title("Hybrid Speed Absolute Error vs Time", fontsize=11, fontweight="bold")
    plt.xlabel("Elapsed Outage Time (s)")
    plt.ylabel("Speed Error (m/s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # 4. Heading Error Growth vs Time
    plt.subplot(2, 2, 4)
    mean_head_err = np.mean([t["head_err"] for t in traces_60s], axis=0)
    for t in traces_60s:
        plt.plot(t["time_s"], t["head_err"], color="gray", alpha=0.25, linewidth=0.8)
    plt.plot(traces_60s[0]["time_s"], mean_head_err, color="teal", linewidth=2.5, label="Mean Heading Error (deg)")
    plt.title("Integrated Heading Absolute Error vs Time", fontsize=11, fontweight="bold")
    plt.xlabel("Elapsed Outage Time (s)")
    plt.ylabel("Heading Error (degrees)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    fig_path = repo_root / "figures/phase15_error_decomposition.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved diagnostic figure to {fig_path}")

if __name__ == "__main__":
    run_diagnostics()
