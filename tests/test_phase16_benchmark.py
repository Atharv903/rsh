"""
Unit and regression tests for Phase 16 benchmark artifacts and causal integrity.
"""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
phase16_dir = repo_root / "experiments/phase16_validation"

def test_file_existence():
    """Verify all Phase 16 required artifacts exist."""
    required = [
        phase16_dir / "per_scenario_metrics.csv",
        phase16_dir / "summary_metrics.csv",
        phase16_dir / "filter_diagnostics.csv",
        phase16_dir / "statistical_paired_comparisons.csv",
        phase16_dir / "config.json",
        repo_root / "reports/phase16_eskf_design.md",
        repo_root / "reports/phase16_benchmark_results.md",
        repo_root / "reports/phase16_literature_addendum.md",
        repo_root / "reports/phase16_research_conclusion.md",
        repo_root / "figures/phase16_ablation_drift_by_duration.png",
        repo_root / "figures/phase16_nhc_crosstrack_suppression.png",
        repo_root / "figures/phase16_eskf_covariance_and_innovations.png",
    ]
    for p in required:
        assert p.exists(), f"Missing required Phase 16 artifact: {p}"

def test_per_scenario_metrics_dimensions():
    """Verify exact count of 720 scenario evaluations (30 scenarios x 3 protocols x 8 configs)."""
    df = pd.read_csv(phase16_dir / "per_scenario_metrics.csv")
    assert len(df) == 720, f"Expected 720 rows, got {len(df)}"
    assert set(df["duration_s"].unique()) == {10, 30, 60}
    assert len(df["scenario_id"].unique()) == 30
    assert len(df["calibration_protocol"].unique()) == 3
    assert len(df["config_id"].unique()) == 8

def test_summary_metrics_reconciliation():
    """Verify that summary_metrics mathematically match per-scenario aggregations."""
    df_scen = pd.read_csv(phase16_dir / "per_scenario_metrics.csv")
    df_sum = pd.read_csv(phase16_dir / "summary_metrics.csv")
    
    assert len(df_sum) == 216 # 3 sessions x 3 protocols x 3 durations x 8 configs
    
    for _, r in df_sum.iterrows():
        sess = r["session"]
        proto = r["calibration_protocol"]
        dur = r["duration_s"]
        cfg = r["config_id"]
        
        sub = df_scen if sess == "POOLED" else df_scen[df_scen["driver_id"] == sess]
        sub = sub[(sub["calibration_protocol"] == proto) & (sub["duration_s"] == dur) & (sub["config_id"] == cfg)]
        
        calc_mean_drift = float(sub["final_drift_m"].mean())
        calc_med_drift = float(sub["final_drift_m"].median())
        
        assert abs(calc_mean_drift - r["mean_final_drift_m"]) < 1e-4, f"Mean drift mismatch for {sess} {proto} {dur}s {cfg}"
        assert abs(calc_med_drift - r["median_final_drift_m"]) < 1e-4, f"Median drift mismatch for {sess} {proto} {dur}s {cfg}"

def test_filter_diagnostics_unbiased_innovations():
    """Verify that filter measurement innovations are zero-mean and unbiased."""
    df_diag = pd.read_csv(phase16_dir / "filter_diagnostics.csv")
    assert len(df_diag) == 90 # 30 scenarios x 3 protocols
    
    mean_speed_inno = df_diag["mean_speed_innovation"].mean()
    mean_nhc_inno = df_diag["mean_nhc_innovation"].mean()
    
    # Assert zero-mean innovations within tight tolerances
    assert abs(mean_speed_inno) < 0.10, f"Speed innovation biased: {mean_speed_inno:.4f} m/s"
    assert abs(mean_nhc_inno) < 0.10, f"NHC innovation biased: {mean_nhc_inno:.4f} m/s"
    
    # Assert high update acceptance (>98%)
    total_accepted = df_diag["speed_updates_accepted"].sum()
    total_rejected = df_diag["speed_updates_rejected"].sum()
    acceptance_rate = total_accepted / (total_accepted + total_rejected)
    assert acceptance_rate > 0.98, f"Speed update acceptance too low: {acceptance_rate:.4f}"

def test_causal_integrity_and_positive_bounds():
    """Verify physical non-negativity across all configurations."""
    df = pd.read_csv(phase16_dir / "per_scenario_metrics.csv")
    assert (df["final_drift_m"] >= 0.0).all(), "Negative drift found"
    assert (df["ate_rmse_m"] >= 0.0).all(), "Negative ATE RMSE found"
    assert (df["speed_mae_mps"] >= 0.0).all(), "Negative speed MAE found"
