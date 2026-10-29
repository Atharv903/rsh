import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.cnn_gru import CNNGRUModel
from src.models.modern_tcn import ModernTCNModel
from src.models.mamba2 import Mamba2Model
from src.corrected_synchronization import extract_aligned_segments_for_drive
from src.outage_estimator import (
    PreOutageState,
    estimate_initial_heading_from_pre_outage_gps,
    run_causal_dead_reckoning,
    make_zero_speed_estimator,
    make_last_known_speed_estimator,
)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    mtype = cfg.get("model_type")
    margs = cfg.get("model", {})
    if mtype == "CNNGRUModel":
        model = CNNGRUModel(**margs)
    elif mtype == "ModernTCNModel":
        model = ModernTCNModel(**margs)
    elif mtype == "Mamba2Model":
        model = Mamba2Model(**margs)
    else:
        raise ValueError(f"Unknown {mtype}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def main():
    print("=" * 70)
    print("PHASE 9 FORENSIC AUDIT: RE-EVALUATING COLAB GPU CHECKPOINTS ON REAL DRIVE S4 (15 SCENARIOS)")
    print("=" * 70)

    segs, _ = extract_aligned_segments_for_drive("S4", raw_dir="data/raw")
    test_df = segs[0]
    total_len = len(test_df)

    np.random.seed(42)
    durations = [10, 30, 60]
    scenarios = []

    for dur in durations:
        n_samples = dur * 10
        valid_starts = []
        for cand in range(500, total_len - n_samples - 500, 1000):
            seg_spd = test_df["ref_speed_mps"].iloc[cand : cand + n_samples].values
            if np.mean(seg_spd) > 3.0:
                valid_starts.append(cand)
        selected_starts = np.random.choice(valid_starts, size=min(5, len(valid_starts)), replace=False)
        selected_starts.sort()
        for idx, st in enumerate(selected_starts):
            scenarios.append({
                "scenario_id": f"S4_{dur}s_trial{idx+1}",
                "duration_s": dur,
                "start_idx": int(st),
                "end_idx": int(st + n_samples),
                "pre_window_samples": 100,
            })

    norm_meta = json.load(open("data/processed/sensor_normalizer_grouped.json"))
    norm_params = norm_meta["params"]
    feat_channels = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
    norm_mean = np.array([norm_params[ch]["mean"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)
    norm_std = np.array([norm_params[ch]["std"] for ch in feat_channels], dtype=np.float32).reshape(1, 6)

    ckpt_dir = Path("experiments/phase8_corrected_speed_models/checkpoints")
    models = {}
    for d in sorted(ckpt_dir.iterdir()):
        pt_path = d / "best_model.pt"
        if pt_path.exists():
            m, _ = load_model(pt_path)
            models[d.name] = m

    print(f"Loaded {len(models)} GPU checkpoints.")

    outage_rows = []
    for sc in scenarios:
        st, end = sc["start_idx"], sc["end_idx"]
        pre_st = max(0, st - sc["pre_window_samples"])
        pre_df = test_df.iloc[pre_st:st].copy()
        in_outage_df = test_df.iloc[st:end].copy()

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

        r_zero = run_causal_dead_reckoning(pre_state, in_outage_df, make_zero_speed_estimator(), "Zero-Speed", sc["scenario_id"], "S4")
        r_const = run_causal_dead_reckoning(pre_state, in_outage_df, make_last_known_speed_estimator(last_spd), "Constant-Velocity", sc["scenario_id"], "S4")

        outage_rows.append({"scenario_id": sc["scenario_id"], "duration_s": sc["duration_s"], "method": "Zero-Speed", "ate_rmse": r_zero.ate_rmse_m, "final_drift_m": r_zero.final_drift_m, "drift_pct": r_zero.drift_pct})
        outage_rows.append({"scenario_id": sc["scenario_id"], "duration_s": sc["duration_s"], "method": "Constant-Velocity", "ate_rmse": r_const.ate_rmse_m, "final_drift_m": r_const.final_drift_m, "drift_pct": r_const.drift_pct})

        for m_name, model in models.items():
            def make_model_fn(m):
                def _est(w):
                    w_norm = (w - norm_mean) / norm_std
                    tensor_in = torch.from_numpy(w_norm.astype(np.float32)).unsqueeze(0)
                    with torch.no_grad():
                        return float(m(tensor_in).item())
                return _est

            r_m = run_causal_dead_reckoning(
                pre_state,
                in_outage_df,
                make_model_fn(model),
                m_name,
                sc["scenario_id"],
                "S4",
            )
            outage_rows.append({
                "scenario_id": sc["scenario_id"],
                "duration_s": sc["duration_s"],
                "method": m_name,
                "ate_rmse": r_m.ate_rmse_m,
                "final_drift_m": r_m.final_drift_m,
                "drift_pct": r_m.drift_pct,
            })

    df_out = pd.DataFrame(outage_rows)
    df_out["arch"] = df_out["method"].apply(lambda m: "CNN-GRU" if "cnngru" in m else ("ModernTCN" if "moderntcn" in m else ("Mamba-2 SSD" if "mamba2" in m else m)))
    summary = df_out.groupby(["duration_s", "arch"])[["ate_rmse", "final_drift_m", "drift_pct"]].mean().round(2)
    print("\n--- Colab GPU Checkpoints Evaluated on the EXACT SAME 15 Scenarios of Real Drive S4 ---")
    print(summary)

    # Save to dedicated audit comparison file
    os.makedirs("reports", exist_ok=True)
    df_out.to_csv("reports/colab_checkpoints_on_real_s4_outages.csv", index=False)
    print("\nSaved detailed outage metrics to reports/colab_checkpoints_on_real_s4_outages.csv")


if __name__ == "__main__":
    main()
