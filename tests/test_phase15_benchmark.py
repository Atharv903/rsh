"""
Unit and regression tests for Phase 15 benchmark outputs and causal integrity.
"""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
phase15_dir = repo_root / "experiments/phase15_validation"

def test_file_existence():
    """Verify all Phase 15 core artifacts exist."""
    required_files = [
        phase15_dir / "literature_comparison.csv",
        phase15_dir / "phase14_reproduction_audit.csv",
        phase15_dir / "phase14_reproduction_audit.md",
        phase15_dir / "error_ablation_summary.csv",
        phase15_dir / "error_regime_summary.csv",
        phase15_dir / "per_scenario_metrics.csv",
        phase15_dir / "summary_metrics.csv",
        phase15_dir / "statistical_paired_comparisons.csv",
        phase15_dir / "computational_benchmarks.json",
        phase15_dir / "config.json",
    ]
    for p in required_files:
        assert p.exists(), f"Missing required Phase 15 file: {p}"

def test_per_scenario_metrics_dimensions():
    """Verify exact count of 540 scenario runs (30 scenarios x 3 protocols x 6 estimators)."""
    df = pd.read_csv(phase15_dir / "per_scenario_metrics.csv")
    assert len(df) == 540, f"Expected 540 rows, got {len(df)}"
    assert set(df["duration_s"].unique()) == {10, 30, 60}
    assert len(df["scenario_id"].unique()) == 30
    assert len(df["calibration_protocol"].unique()) == 3
    assert len(df["estimator"].unique()) == 6

def test_summary_metrics_reconciliation():
    """Verify that summary_metrics mathematically match per-scenario aggregations."""
    df_scen = pd.read_csv(phase15_dir / "per_scenario_metrics.csv")
    df_sum = pd.read_csv(phase15_dir / "summary_metrics.csv")
    
    assert len(df_sum) == 162 # 3 sessions x 3 protocols x 3 durations x 6 estimators
    
    for _, r in df_sum.iterrows():
        sess = r["session"]
        proto = r["calibration_protocol"]
        dur = r["duration_s"]
        est = r["estimator"]
        
        sub = df_scen if sess == "POOLED" else df_scen[df_scen["driver_id"] == sess]
        sub = sub[(sub["calibration_protocol"] == proto) & (sub["duration_s"] == dur) & (sub["estimator"] == est)]
        
        calc_mean_drift = float(sub["final_drift_m"].mean())
        calc_med_drift = float(sub["final_drift_m"].median())
        
        assert abs(calc_mean_drift - r["mean_final_drift_m"]) < 1e-4, f"Mismatch in mean drift for {sess} {proto} {dur}s {est}"
        assert abs(calc_med_drift - r["median_final_drift_m"]) < 1e-4, f"Mismatch in median drift for {sess} {proto} {dur}s {est}"

def test_causal_protocol_and_physical_bounds():
    """Verify that physical constraints and metrics obey non-negativity."""
    df = pd.read_csv(phase15_dir / "per_scenario_metrics.csv")
    assert (df["final_drift_m"] >= 0.0).all(), "Negative drift found"
    assert (df["ate_rmse_m"] >= 0.0).all(), "Negative ATE RMSE found"
    assert (df["speed_mae_mps"] >= 0.0).all(), "Negative speed MAE found"
    assert (df["distance_travelled_m"] >= 0.0).all(), "Negative distance travelled found"
    assert (df["drift_pct_of_distance"] >= 0.0).all(), "Negative drift percentage found"

def test_literature_comparison_integrity():
    """Verify literature comparison table covers all 6 required state-of-the-art studies."""
    df_lit = pd.read_csv(phase15_dir / "literature_comparison.csv")
    assert len(df_lit) >= 6
    required_studies = {"DeGIN_2026", "OFFGRID_Carcajou_2025", "Xiao_2025", "Shin_2025", "AI_IMU_2020", "RoNIN_2020"}
    assert set(df_lit["study_id"].unique()) == required_studies

def test_computational_benchmark_structure():
    """Verify computational benchmarking has valid timings and parameters."""
    with open(phase15_dir / "computational_benchmarks.json") as f:
        data = json.load(f)
    assert "models_profile" in data
    assert len(data["models_profile"]) == 3
    for m in data["models_profile"]:
        assert m["parameter_count"] > 100000
        assert m["latency_mean_ms"] > 0.0
        assert m["throughput_inferences_per_sec"] > 10.0
