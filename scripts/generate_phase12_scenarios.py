"""
Phase 12C: Generate Scenario Manifest for Expanded Independent Validation.

Generates 30 standardized outage scenarios:
- 15 from Driver B (M_seg1, Sept 7, 2019)
- 15 from Driver D (Y1_seg3, Aug 30, 2019)
Each containing:
- 5 x 10s outages
- 5 x 30s outages
- 5 x 60s outages
Seed: 42 (reproducible, deterministic).
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

from src.corrected_synchronization import extract_aligned_segments_for_drive

def generate_manifest():
    out_dir = Path("experiments/phase12_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Driver B (M_seg1) scenarios from Phase 10
    with open("experiments/phase10_independent_validation/scenarios/driver_b_outage_scenarios.json") as f:
        scenarios_b = json.load(f)
    for sc in scenarios_b:
        sc["driver_id"] = "Driver B"
        sc["session_name"] = "M (Driver B)"
        sc["collection_date"] = "2019-09-07"
        sc["segment_id"] = "M_seg1"

    # 2. Driver D (Y1_seg3) scenarios
    raw_d_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/Y (Driver D)/Y1"
    segs_d, _ = extract_aligned_segments_for_drive("Y1", raw_dir=raw_d_dir)
    seg3_d = segs_d[2] # Y1_seg3, 65,761 samples
    
    np.random.seed(42)
    durations = [10, 30, 60]
    scenarios_d = []

    for dur in durations:
        n_samples = dur * 10
        valid_starts = []
        for cand in range(500, len(seg3_d) - n_samples - 500, 1000):
            seg_spd = seg3_d["ref_speed_mps"].iloc[cand : cand + n_samples].values
            if np.mean(seg_spd) > 3.0: # vehicle in motion
                valid_starts.append(cand)
                
        selected = np.random.choice(valid_starts, size=min(5, len(valid_starts)), replace=False)
        selected.sort()
        for idx, st in enumerate(selected):
            sc_spd = seg3_d["ref_speed_mps"].iloc[st : st + n_samples].values
            scenarios_d.append({
                "scenario_id": f"DriverD_Y1_{dur}s_trial{idx+1}",
                "driver_id": "Driver D",
                "session_name": "Y1 (Driver D)",
                "collection_date": "2019-08-30",
                "segment_id": "Y1_seg3",
                "drive_id": "Y1",
                "duration_s": dur,
                "start_idx": int(st),
                "end_idx": int(st + n_samples),
                "pre_window_samples": 100,
                "mean_speed_mps": float(np.mean(sc_spd)),
                "min_speed_mps": float(np.min(sc_spd)),
                "max_speed_mps": float(np.max(sc_spd)),
            })

    all_scenarios = scenarios_b + scenarios_d
    
    out_manifest = out_dir / "phase12_scenarios.json"
    with open(out_manifest, "w") as f:
        json.dump(all_scenarios, f, indent=2)
    print(f"Generated {len(all_scenarios)} total scenarios across 2 independent drivers.")
    print(f"  Driver B (M_seg1) : {len(scenarios_b)} scenarios")
    print(f"  Driver D (Y1_seg3): {len(scenarios_d)} scenarios")
    
    for sc in scenarios_d:
        print(f"  {sc['scenario_id']:24s} | Dur: {sc['duration_s']:2d}s | Start: {sc['start_idx']:5d} | Mean Spd: {sc['mean_speed_mps']:5.2f} m/s | Max: {sc['max_speed_mps']:5.2f} m/s")

    # Save experiment config
    exp_config = {
        "experiment_name": "Phase 12 Navigation Correction & Expanded Independent Validation",
        "timestamp": "2026-10-29",
        "git_commit": "13a87aa",
        "dataset_archive": "Categorised IOVNB Dataset",
        "sessions": [
            {
                "session_name": "M (Driver B)",
                "driver": "Driver B",
                "date": "2019-09-07",
                "segment_evaluated": "M_seg1",
                "num_scenarios": 15,
                "heading_convention_tested": ["corrected (-1)", "legacy (+1)"]
            },
            {
                "session_name": "Y1 (Driver D)",
                "driver": "Driver D",
                "date": "2019-08-30",
                "segment_evaluated": "Y1_seg3",
                "num_scenarios": 15,
                "heading_convention_tested": ["corrected (-1)"]
            }
        ],
        "models_evaluated": [
            "Constant-Velocity",
            "Zero-Speed",
            "cnngrumodel_seed42", "cnngrumodel_seed123", "cnngrumodel_seed456",
            "mamba2model_seed42", "mamba2model_seed123", "mamba2model_seed456",
            "moderntcnmodel_seed42", "moderntcnmodel_seed123", "moderntcnmodel_seed456"
        ],
        "normalization": "data/processed/sensor_normalizer_grouped.json (fitted exclusively on S1+S2)",
        "durations_s": [10, 30, 60],
        "trials_per_duration": 5,
        "total_scenarios": len(all_scenarios),
        "total_runs": len(all_scenarios) * 11
    }
    
    cfg_path = out_dir / "experiment_config.json"
    with open(cfg_path, "w") as f:
        json.dump(exp_config, f, indent=2)
    print(f"Saved experiment configuration to {cfg_path}")

if __name__ == "__main__":
    generate_manifest()
