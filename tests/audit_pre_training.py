"""
Pre-training audit script to verify data split integrity, window isolation,
target coordinate units, and reference validity before model training.
"""

import json
from pathlib import Path
import numpy as np


def run_audit():
    print("=== STAGE 1 PRE-TRAINING AUDIT ===")
    
    # 1. Split manifest check
    manifest_path = Path("data/splits/drive_split_manifest.json")
    assert manifest_path.exists(), "Split manifest not found!"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    expected_train = set(manifest["splits"]["train"]["drives"])
    expected_val = set(manifest["splits"]["val"]["drives"])
    expected_test = set(manifest["splits"]["test"]["drives"])
    
    print(f"Manifest Drives -> Train: {expected_train}, Val: {expected_val}, Test: {expected_test}")
    assert len(expected_train.intersection(expected_val)) == 0, "Train and Val overlap!"
    assert len(expected_train.intersection(expected_test)) == 0, "Train and Test overlap!"
    assert len(expected_val.intersection(expected_test)) == 0, "Val and Test overlap!"
    print("[PASS] Drive splits are completely disjoint.")
    
    # 2. Window files check
    for split_name, npz_path, exp_drives in [
        ("train", "data/processed/train_windows.npz", expected_train),
        ("val", "data/processed/val_windows.npz", expected_val),
        ("test", "data/processed/test_windows.npz", expected_test),
    ]:
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"]
        y_disp = data["Y"] if "Y" in data else data["y"]
        y = y_disp / 10.0  # Window-average velocity in m/s
        meta = data["meta"] if "meta" in data else None
        
        if meta is not None:
            valid_mask = np.array([m.get("valid_ratio", 1.0) >= 0.99 for m in meta], dtype=bool)
            drives = set(m.get("drive_id", "") for m in meta)
        else:
            valid_mask = data["valid_mask"]
            drives = set(data["drive_id"])
        
        # Clean drive names if without .csv
        drives_norm = set(d + ".csv" if not d.endswith(".csv") else d for d in drives)
        
        print(f"\nAudit [{split_name.upper()}]:")
        print(f"  Total Windows: {len(X):,}")
        print(f"  Valid Reference Windows: {np.sum(valid_mask):,} ({np.mean(valid_mask)*100:.1f}%)")
        print(f"  X shape: {X.shape}, dtype: {X.dtype}")
        print(f"  y shape: {y.shape}, dtype: {y.dtype}")
        print(f"  Drives present: {drives_norm}")
        
        assert drives_norm.issubset(exp_drives), f"Leakage detected in {split_name}: {drives_norm} vs {exp_drives}"
        assert not np.isnan(X).any(), f"NaN found in {split_name} inputs!"
        assert not np.isinf(X).any(), f"Inf found in {split_name} inputs!"
        assert not np.isnan(y[valid_mask]).any(), f"NaN found in valid {split_name} targets!"
        
        # Verify velocity targets reasonable range (e.g. -50 to 50 m/s for road vehicle)
        v_mags = np.linalg.norm(y[valid_mask], axis=1)
        print(f"  Velocity Target Magnitude (valid): min={np.min(v_mags):.2f} m/s, max={np.max(v_mags):.2f} m/s, mean={np.mean(v_mags):.2f} m/s")
        assert np.max(v_mags) < 60.0, f"Unphysical velocity target {np.max(v_mags)} m/s!"
        print(f"  [PASS] {split_name} verified clean.")
        
    # 3. Normalizer check
    norm_path = Path("data/processed/sensor_normalizer.json")
    assert norm_path.exists(), "Sensor normalizer not found!"
    with open(norm_path, "r") as f:
        norm = json.load(f)
    print(f"\nSensor Normalizer Check:")
    channels = norm.get("channels", list(norm.get("params", {}).keys()))
    means = [norm["params"][ch]["mean"] for ch in channels]
    stds = [norm["params"][ch]["std"] for ch in channels]
    print(f"  Channels: {channels}")
    print(f"  Means: {[round(m, 4) for m in means]}")
    print(f"  Stds:  {[round(s, 4) for s in stds]}")
    assert len(means) == 6 and len(stds) == 6, "Normalizer channel count mismatch!"
    assert all(s > 1e-4 for s in stds), "Standard deviation near zero!"
    print("  [PASS] Normalizer parameters verified.")
    
    # 4. Outage Scenarios check
    outage_path = Path("data/processed/outage_scenarios.json")
    assert outage_path.exists(), "Outage scenarios not found!"
    with open(outage_path, "r") as f:
        scenarios = json.load(f)
    print(f"\nOutage Scenarios: {len(scenarios)} total")
    for dur in [10, 30, 60]:
        dur_sc = [s for s in scenarios if s["duration_s"] == dur]
        print(f"  {dur}s outages: {len(dur_sc)} scenarios")
        assert len(dur_sc) > 0, f"No scenarios for duration {dur}s!"
    print("  [PASS] Outage scenarios verified.")
    print("\n>>> AUDIT PASSED: ZERO LEAKAGE, CLEAN NORMALIZATION, ROBUST TARGETS. READY FOR TRAINING. <<<")


if __name__ == "__main__":
    run_audit()
