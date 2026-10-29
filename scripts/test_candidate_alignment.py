"""
Tests alignment and post-alignment correlation for Driver D and Driver E candidate sessions.
"""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.corrected_synchronization import extract_aligned_segments_for_drive
import numpy as np

base_dir = Path("Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset")

candidates = [
    ("Driver D (Y1)", base_dir / "Y (Driver D)/Y1", "Y1"),
    ("Driver E (Vfa01)", base_dir / "Vf (Driver E)/V-Vfa01", "Vfa01"),
    ("Driver E (Vfa02)", base_dir / "Vf (Driver E)/V-Vfa02", "Vfa02"),
    ("Driver E (Vta01a)", base_dir / "Vta (Driver E)/Vta01a", "Vta1a"),
    ("Driver E (Vta02)", base_dir / "Vta (Driver E)/Vta02", "Vta2"),
    ("Driver E (Vw01)", base_dir / "Vw (Driver E)/Vw01", "Vw1"),
    ("Driver E (Vw02)", base_dir / "Vw (Driver E)/Vw02", "Vw2"),
]

for label, p_dir, drive_id in candidates:
    try:
        segs, meta = extract_aligned_segments_for_drive(drive_id, raw_dir=str(p_dir))
        print(f"=== {label} ===")
        print(f"  Valid Segments: {len(segs)}")
        for s in meta["segments"]:
            sid = s.get("segment_id", s.get("segment_idx"))
            st = s["status"]
            n = s.get("num_samples", 0)
            off = s.get("offset_s", 0)
            sh = s.get("shift_samples", 0)
            print(f"    {sid}: status={st}, samples={n}, offset={off:.2f}s, shift={sh}")
            
        for seg in segs:
            sid = seg["segment_id"].iloc[0]
            yr = np.deg2rad(seg["ref_yaw_rate_dps"].values)
            gp = seg["gyro_pitch"].values
            gy = seg["gyro_yaw"].values
            gr = seg["gyro_roll"].values
            c_p = np.corrcoef(gp, yr)[0, 1]
            c_y = np.corrcoef(gy, yr)[0, 1]
            c_r = np.corrcoef(gr, yr)[0, 1]
            print(f"    [{sid}] Post-Align Corr: Pitch={c_p:+.4f}, Yaw={c_y:+.4f}, Roll={c_r:+.4f}")
    except Exception as e:
        print(f"=== {label} === ERROR: {e}")
