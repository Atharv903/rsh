"""
Generates 15 standardized, matched GNSS outage scenarios for Independent Driver B (M).

Criteria:
- 5 trials for 10-second outages
- 5 trials for 30-second outages
- 5 trials for 60-second outages
- Seed: 42 (deterministic, reproducible)
- Vehicle in motion (mean speed > 3.0 m/s)
- Adequate pre-outage window (100 samples = 10s)
"""

import os
import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from src.corrected_synchronization import extract_aligned_segments_for_drive

def generate_scenarios():
    raw_m_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)"
    segs, meta = extract_aligned_segments_for_drive("M", raw_dir=raw_m_dir)
    seg1 = segs[1]  # 61,714 samples, 102.9 minutes
    
    np.random.seed(42)
    durations = [10, 30, 60]
    scenarios = []

    for dur in durations:
        n_samples = dur * 10
        valid_starts = []
        for cand in range(500, len(seg1) - n_samples - 500, 1000):
            seg_spd = seg1["ref_speed_mps"].iloc[cand : cand + n_samples].values
            if np.mean(seg_spd) > 3.0:
                valid_starts.append(cand)
                
        selected = np.random.choice(valid_starts, size=min(5, len(valid_starts)), replace=False)
        selected.sort()
        for idx, st in enumerate(selected):
            sc_spd = seg1["ref_speed_mps"].iloc[st : st + n_samples].values
            scenarios.append({
                "scenario_id": f"DriverB_M_{dur}s_trial{idx+1}",
                "segment_id": "M_seg1",
                "drive_id": "M",
                "duration_s": dur,
                "start_idx": int(st),
                "end_idx": int(st + n_samples),
                "pre_window_samples": 100,
                "mean_speed_mps": float(np.mean(sc_spd)),
                "min_speed_mps": float(np.min(sc_spd)),
                "max_speed_mps": float(np.max(sc_spd)),
            })

    out_path = Path("experiments/phase10_independent_validation/scenarios/driver_b_outage_scenarios.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"Saved {len(scenarios)} scenarios to {out_path}")
    for sc in scenarios:
        sid = sc["scenario_id"]
        d = sc["duration_s"]
        st = sc["start_idx"]
        end = sc["end_idx"]
        m_spd = sc["mean_speed_mps"]
        mx_spd = sc["max_speed_mps"]
        print(f"{sid:24s} | Dur: {d:2d}s | Start: {st:5d} | End: {end:5d} | Mean Spd: {m_spd:5.2f} m/s | Max: {mx_spd:5.2f} m/s")

if __name__ == "__main__":
    generate_scenarios()
