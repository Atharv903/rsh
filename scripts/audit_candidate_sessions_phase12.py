"""
Phase 12C: Forensic Audit of Additional Independent Driving Sessions (Driver D and Driver E).

Audits all candidate sessions in the IO-VNBD dataset archive to find genuinely independent,
leakage-free, high-quality sessions for expanded independent dead-reckoning validation.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.inspection import sanitize_column_names

base_dir = Path("Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset")

def audit_candidate(s_path, v_path, label):
    try:
        s_raw = pd.read_csv(s_path, encoding="latin1")
        v_raw = pd.read_csv(v_path, encoding="latin1")
        s_df = sanitize_column_names(s_raw)
        v_df = v_raw.copy()
        v_df.columns = [c.strip() for c in v_df.columns]
    except Exception as e:
        return {"session": label, "status": "ERROR_LOADING", "error": str(e)}

    n_s, n_v = len(s_df), len(v_df)
    
    # Check date column
    date_col = [c for c in s_df.columns if "DATE" in c]
    t0_str = str(s_df[date_col[0]].iloc[0]) if date_col else "N/A"
    tend_str = str(s_df[date_col[0]].iloc[-1]) if date_col else "N/A"
    
    # Check time deltas in S
    t_ms = s_df["TIME SINCE START (ms)"].values.astype(float)
    dt_ms = np.diff(t_ms)
    s_resets = np.where(dt_ms < 0)[0]
    
    # Check time deltas in V
    v_time_col = [c for c in v_df.columns if "Time" in c or "time" in c]
    t_v = v_df[v_time_col[0]].values.astype(float) if v_time_col else None
    dt_v = np.diff(t_v) if t_v is not None else []
    v_gaps = np.where(dt_v > 0.5)[0] if len(dt_v) > 0 else []
    
    # Speed check
    spd_col = [c for c in v_df.columns if "Velocity" in c or "Speed" in c]
    v_spd = v_df[spd_col[0]].values.astype(float) / 3.6 if spd_col else None
    
    # Gravity check (mounting orientation)
    gz_col = [c for c in s_df.columns if "GRAVITY Z" in c]
    mean_gz = float(s_df[gz_col[0]].mean()) if gz_col else 0.0
    
    # Gyro check
    gp_col = [c for c in s_df.columns if "GYROSCOPE Pitch" in c]
    yr_col = [c for c in v_df.columns if "Yaw Rate" in c]
    
    corr_gp_yr = np.nan
    if gp_col and yr_col and min(n_s, n_v) > 100:
        min_n = min(n_s, n_v)
        corr_gp_yr = float(np.corrcoef(s_df[gp_col[0]].iloc[:min_n], v_df[yr_col[0]].iloc[:min_n])[0, 1])

    return {
        "session": label,
        "s_rows": n_s,
        "v_rows": n_v,
        "duration_minutes": float(n_s * 0.1 / 60.0),
        "start_time": t0_str,
        "end_time": tend_str,
        "phone_resets_count": int(len(s_resets)),
        "veh_time_gaps_gt_500ms": int(len(v_gaps)),
        "mean_gravity_z": mean_gz,
        "speed_mean_mps": float(np.mean(v_spd)) if v_spd is not None else 0.0,
        "speed_max_mps": float(np.max(v_spd)) if v_spd is not None else 0.0,
        "corr_gyro_pitch_yaw_rate": corr_gp_yr,
    }


def main():
    print("=" * 85)
    print("PHASE 12C: AUDITING CANDIDATE SESSIONS ACROSS DRIVER D AND DRIVER E")
    print("=" * 85)
    
    candidates = []
    
    # 1. Driver D (Y1)
    y1_s = base_dir / "Y (Driver D)/Y1/S-Y1.csv"
    y1_v = base_dir / "Y (Driver D)/Y1/V-Y1.csv"
    if y1_s.exists() and y1_v.exists():
        candidates.append((y1_s, y1_v, "Driver D (Y1)"))
        
    # 2. Driver E (Vf)
    for vf_name in ["V-Vfa01", "V-Vfa02"]:
        p_dir = base_dir / f"Vf (Driver E)/{vf_name}"
        if p_dir.exists():
            s_f = list(p_dir.glob("S-*.csv"))
            v_f = list(p_dir.glob("V-*.csv"))
            if s_f and v_f:
                candidates.append((s_f[0], v_f[0], f"Driver E ({vf_name})"))
                
    # 3. Driver E (Vta sample sessions)
    for vta_id in ["Vta01a", "Vta01b", "Vta02", "Vta03", "Vta05", "Vta10", "Vta15", "Vta20", "Vta25", "Vta30"]:
        p_dir = base_dir / f"Vta (Driver E)/{vta_id}"
        if p_dir.exists():
            s_f = list(p_dir.glob("S-*.csv"))
            v_f = list(p_dir.glob("V-*.csv"))
            if s_f and v_f:
                candidates.append((s_f[0], v_f[0], f"Driver E ({vta_id})"))

    # 4. Driver E (Vw sample sessions)
    for vw_id in ["Vw01", "Vw02", "Vw05", "Vw10"]:
        p_dir = base_dir / f"Vw (Driver E)/{vw_id}"
        if p_dir.exists():
            s_f = list(p_dir.glob("S-*.csv"))
            v_f = list(p_dir.glob("V-*.csv"))
            if s_f and v_f:
                candidates.append((s_f[0], v_f[0], f"Driver E ({vw_id})"))

    print(f"Discovered {len(candidates)} candidate session pairs. Auditing each...")
    results = []
    for s_path, v_path, label in candidates:
        info = audit_candidate(s_path, v_path, label)
        results.append(info)
        print(f"  {info['session']:25s} | {info['duration_minutes']:5.1f} min | Rows: S={info['s_rows']:5d}, V={info['v_rows']:5d} | Resets: {info['phone_resets_count']} | Gaps: {info['veh_time_gaps_gt_500ms']} | Corr(gp, yr): {info['corr_gyro_pitch_yaw_rate']:+.3f}")

    out_json = Path("experiments/phase12_validation/candidate_sessions_audit.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved candidates audit to {out_json}")

if __name__ == "__main__":
    main()
