"""
Audits heading mathematics across multiple left turns, right turns, and straight sections
in both continuous segments of Driver B (M).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.corrected_synchronization import extract_aligned_segments_for_drive

def audit_turns():
    raw_m_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)"
    segs, _ = extract_aligned_segments_for_drive("M", raw_dir=raw_m_dir)

    for seg_idx, seg in enumerate(segs):
        sid = seg["segment_id"].iloc[0]
        print("=" * 80)
        print(f"AUDITING HEADING MATHEMATICS IN SEGMENT {seg_idx}: {sid}")
        print("=" * 80)
        dt = 0.1
        h_deg = seg["ref_heading_deg"].values
        h_rad = np.unwrap(np.deg2rad(h_deg))
        yr_rad = np.deg2rad(seg["ref_yaw_rate_dps"].values)
        
        gyaw = seg["gyro_yaw"].values
        gpitch = seg["gyro_pitch"].values
        groll = seg["gyro_roll"].values
        
        w_len = 100
        right_turns = []
        left_turns = []
        straights = []
        
        for k in range(0, len(seg) - w_len, 200):
            dh = np.rad2deg(h_rad[k + w_len] - h_rad[k])
            mean_spd = seg["ref_speed_mps"].iloc[k : k + w_len].mean()
            if mean_spd > 5.0:
                if dh > 25.0 and len(right_turns) < 3:
                    right_turns.append(k)
                elif dh < -25.0 and len(left_turns) < 3:
                    left_turns.append(k)
                elif abs(dh) < 1.5 and mean_spd > 12.0 and len(straights) < 3:
                    straights.append(k)
                    
        for turn_type, indices in [("RIGHT TURN (+dHeading)", right_turns), ("LEFT TURN (-dHeading)", left_turns), ("STRAIGHT (~0 dHeading)", straights)]:
            print(f"\n--- {turn_type} (Count: {len(indices)}) ---")
            for idx in indices:
                dh = np.rad2deg(h_rad[idx + w_len] - h_rad[idx])
                int_yr = np.rad2deg(np.sum(yr_rad[idx : idx + w_len]) * dt)
                int_yaw = np.rad2deg(np.sum(gyaw[idx : idx + w_len]) * dt)
                int_pitch = np.rad2deg(np.sum(gpitch[idx : idx + w_len]) * dt)
                int_roll = np.rad2deg(np.sum(groll[idx : idx + w_len]) * dt)
                print(f"Window [{idx:5d}:{idx+w_len:5d}] | Ref dHeading: {dh:+6.1f} deg | Ref YawRate: {int_yr:+6.1f} deg | GyroYaw: {int_yaw:+6.1f} deg | GyroPitch: {int_pitch:+6.1f} deg | GyroRoll: {int_roll:+6.1f} deg")

if __name__ == "__main__":
    audit_turns()
