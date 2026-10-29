"""
Rigorous S/V Signal Alignment and Timestamp Audit Script.
Analyzes timestamp relationships, clock offsets, clock resets, and cross-correlations
between smartphone and vehicle telemetry across all 6 drives in the IO-VNBD dataset.
"""

import os
import sys
import json
from pathlib import Path

# Add project root
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})


def parse_phone_seconds_of_day(date_series: pd.Series) -> pd.Series:
    """Parse 'YYYY-MO-DD HH-MI-SS_SSS' into seconds since start of day (UTC).
    Phone is logged in British Summer Time (BST = UTC+1 = +3600s).
    """
    times = date_series.str.strip().str.split(' ').str[1].str.replace(':', '-').str.replace('_', '-').str.split('-')
    h = times.str[0].astype(float)
    m = times.str[1].astype(float)
    s = times.str[2].astype(float)
    ms = times.str[3].astype(float)
    # Convert BST to UTC by subtracting 3600 seconds
    return (h * 3600.0 + m * 60.0 + s + ms / 1000.0) - 3600.0


def audit_drive_pair(drive_id: str):
    s_path = f"data/raw/S-{drive_id}.csv"
    v_path = f"data/raw/V-{drive_id}.csv"
    
    s_df = pd.read_csv(s_path, encoding="latin1")
    v_df = pd.read_csv(v_path, encoding="latin1")
    
    s_cols = {c.strip(): c for c in s_df.columns}
    v_cols = {c.strip(): c for c in v_df.columns}
    
    n_rows = len(s_df)
    
    phone_utc = parse_phone_seconds_of_day(s_df[s_cols["DATE (YYYY-MO-DD HH-MI-SS_SSS)"]])
    veh_utc = v_df[v_cols["Time Since Start of Day (seconds)"]].values.astype(float)
    
    delta_t = phone_utc.values - veh_utc
    
    # Identify resets in phone TIME SINCE START (ms)
    t_ms = s_df[s_cols["TIME SINCE START (ms)"]].values.astype(float)
    resets = np.where(np.diff(t_ms) < 0)[0]
    
    # Segment definitions based on resets
    seg_bounds = [0] + list(resets + 1) + [n_rows]
    segments = []
    
    v_yaw = v_df[v_cols["Yaw Rate (deg/sec)"]].values.astype(float)
    s_pitch = s_df[s_cols["GYROSCOPE Pitch (rad/s)"]].values.astype(float) * (180.0 / np.pi)
    s_spd = s_df[s_cols["GPS SPEED (Kmh)"]].values.astype(float)
    v_spd = v_df[v_cols["Velocity (km/hr)"]].values.astype(float)
    
    for i in range(len(seg_bounds) - 1):
        st, end = seg_bounds[i], seg_bounds[i+1]
        dt_seg = delta_t[st:end]
        mean_offset = float(np.mean(dt_seg))
        std_offset = float(np.std(dt_seg))
        
        # Cross-correlations
        # 1. Naive (zero shift)
        c_gyro_naive = float(np.corrcoef(s_pitch[st:end], v_yaw[st:end])[0, 1])
        c_spd_naive = float(np.corrcoef(s_spd[st:end], v_spd[st:end])[0, 1])
        
        # 2. Shifted by integer samples corresponding to round(mean_offset / 0.1)
        shift_smp = int(np.round(mean_offset / 0.1))
        
        # We need to shift vehicle relative to phone:
        # If phone_utc > veh_utc by offset, phone is ahead in index, so phone[k] matches veh[k + shift_smp]
        if st + shift_smp >= 0 and end + shift_smp <= n_rows:
            v_yaw_aligned = v_yaw[st + shift_smp : end + shift_smp]
            v_spd_aligned = v_spd[st + shift_smp : end + shift_smp]
            s_pitch_aligned = s_pitch[st:end]
            s_spd_aligned = s_spd[st:end]
            
            c_gyro_aligned = float(np.corrcoef(s_pitch_aligned, v_yaw_aligned)[0, 1])
            c_spd_aligned = float(np.corrcoef(s_spd_aligned, v_spd_aligned)[0, 1])
        else:
            c_gyro_aligned = c_gyro_naive
            c_spd_aligned = c_spd_naive
            
        segments.append({
            "segment_idx": i,
            "start_row": int(st),
            "end_row": int(end),
            "num_samples": int(end - st),
            "mean_offset_s": mean_offset,
            "std_offset_s": std_offset,
            "shift_samples": shift_smp,
            "corr_gyro_naive": c_gyro_naive,
            "corr_gyro_aligned": c_gyro_aligned,
            "corr_spd_naive": c_spd_naive,
            "corr_spd_aligned": c_spd_aligned,
        })
        
    return {
        "drive": drive_id,
        "total_rows": n_rows,
        "resets_count": int(len(resets)),
        "resets_rows": [int(r) for r in resets],
        "delta_t_overall": {
            "start_offset_s": float(delta_t[0]),
            "end_offset_s": float(delta_t[-1]),
            "total_drift_s": float(delta_t[-1] - delta_t[0]),
            "min_offset_s": float(np.min(delta_t)),
            "max_offset_s": float(np.max(delta_t)),
        },
        "segments": segments,
    }


def main():
    drives = ["S1", "S2", "S3a", "S3b", "S3c", "S4"]
    results = {}
    
    for d in drives:
        res = audit_drive_pair(d)
        results[d] = res
        print(f"Drive {d}: {len(res['segments'])} segment(s) | Resets: {res['resets_count']}")
        for seg in res["segments"]:
            print(f"  Seg {seg['segment_idx']} ({seg['start_row']}:{seg['end_row']}): "
                  f"Offset = {seg['mean_offset_s']:+.3f}s (Shift: {seg['shift_samples']:+d} smp) | "
                  f"Gyro Corr: naive={seg['corr_gyro_naive']:.3f} -> aligned={seg['corr_gyro_aligned']:.3f} | "
                  f"Speed Corr: naive={seg['corr_spd_naive']:.3f} -> aligned={seg['corr_spd_aligned']:.3f}")
                  
    # Save JSON results
    with open("reports/phase3_sv_alignment_audit.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved reports/phase3_sv_alignment_audit.json")
    
    # -------------------------------------------------------------------------
    # Generate Publication Figures
    # -------------------------------------------------------------------------
    os.makedirs("figures", exist_ok=True)
    
    # Figure 1: Timestamp Offsets Across All Drives
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=False)
    axes = axes.flatten()
    
    for i, d in enumerate(drives):
        s_path = f"data/raw/S-{d}.csv"
        v_path = f"data/raw/V-{d}.csv"
        s_df = pd.read_csv(s_path, encoding="latin1")
        v_df = pd.read_csv(v_path, encoding="latin1")
        s_cols = {c.strip(): c for c in s_df.columns}
        v_cols = {c.strip(): c for c in v_df.columns}
        
        p_utc = parse_phone_seconds_of_day(s_df[s_cols["DATE (YYYY-MO-DD HH-MI-SS_SSS)"]])
        v_utc = v_df[v_cols["Time Since Start of Day (seconds)"]].values
        dt_s = p_utc.values - v_utc
        
        time_hours = np.arange(len(dt_s)) * 0.1 / 3600.0
        
        ax = axes[i]
        ax.plot(time_hours, dt_s, color="#2b5c8f", linewidth=1.5, label="Raw UTC Difference ($t_{Phone} - t_{Veh}$)")
        ax.set_title(f"Drive {d} (N={len(dt_s):,} samples)")
        ax.set_xlabel("Elapsed Time (hours)")
        ax.set_ylabel("Offset $\Delta t$ (seconds)")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Mark resets
        resets = results[d]["resets_rows"]
        for r in resets:
            ax.axvline(r * 0.1 / 3600.0, color="red", linestyle=":", label="Clock Reset Jump" if r == resets[0] else "")
            
        ax.legend(loc="upper left")
        
    plt.suptitle("Raw Timestamp Offset $(t_{Phone} - t_{Vehicle})$ Across IO-VNBD Drives Showing Clock Resets", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/phase3_timestamp_offsets.png")
    plt.close()
    print("Saved figures/phase3_timestamp_offsets.png")
    
    # Figure 2: Signal Alignment Verification (Zoomed Overlay of Gyro vs Vehicle Yaw Rate)
    # Show Drive S1 and Drive S3c zoomed turn
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    for idx, d in enumerate(["S1", "S3c"]):
        s_path = f"data/raw/S-{d}.csv"
        v_path = f"data/raw/V-{d}.csv"
        s_df = pd.read_csv(s_path, encoding="latin1")
        v_df = pd.read_csv(v_path, encoding="latin1")
        s_cols = {c.strip(): c for c in s_df.columns}
        v_cols = {c.strip(): c for c in v_df.columns}
        
        s_pitch = s_df[s_cols["GYROSCOPE Pitch (rad/s)"]].values * (180.0 / np.pi)
        v_yaw = v_df[v_cols["Yaw Rate (deg/sec)"]].values
        
        # Pick an active 2-minute dynamic driving turn window
        start_smp = 15000
        window_len = 600  # 60 seconds
        t_win = np.arange(window_len) * 0.1
        
        ax = axes[idx]
        ax.plot(t_win, v_yaw[start_smp:start_smp+window_len], color="black", linewidth=2.0, label="Vehicle Telemetry Yaw Rate (deg/s)")
        ax.plot(t_win, s_pitch[start_smp:start_smp+window_len], color="#d95f02", linestyle="--", linewidth=1.5, label="Smartphone Gyroscope Pitch (deg/s)")
        ax.set_title(f"Drive {d}: Telemetry Yaw Rate vs. Smartphone Gyroscope Dynamics (60-Second Window)")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Angular Rate (deg/s)")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    plt.savefig("figures/phase3_signal_alignment_overlay.png")
    plt.close()
    print("Saved figures/phase3_signal_alignment_overlay.png")
    print("Phase 3 alignment script finished successfully!")


if __name__ == "__main__":
    main()
