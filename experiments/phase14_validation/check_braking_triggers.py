import json
import sys
from pathlib import Path
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.corrected_synchronization import extract_aligned_segments_for_drive

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

triggered = 0
for sc in scenarios:
    df = drive_dfs[sc["drive_id"]]
    st, end = sc["start_idx"], sc["end_idx"]
    sub = df.iloc[st:end]
    min_ax = float(sub["accel_x"].min())
    num_braking = int((sub["accel_x"] < -1.5).sum())
    if num_braking > 0:
        triggered += 1
        print(f"{sc['scenario_id']} ({sc['duration_s']}s): min accel_x = {min_ax:.2f} m/s^2, {num_braking} samples < -1.5 m/s^2")

print(f"\nTotal scenarios triggering braking override: {triggered} / {len(scenarios)}")
