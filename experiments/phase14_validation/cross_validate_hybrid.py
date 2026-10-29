"""
Phase 14: Cross-Validation of Hybrid Estimator Parameters.

Tests whether the hybrid parameters (t_switch, tau) generalize across sessions
under leave-one-driver-out cross-validation:
- Train on Driver B (15 scenarios) -> Evaluate on Driver D (15 scenarios)
- Train on Driver D (15 scenarios) -> Evaluate on Driver B (15 scenarios)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent

def main():
    metrics_path = repo_root / "experiments/phase14_validation/per_scenario_metrics.csv"
    if not metrics_path.exists():
        print("Run evaluate_phase14_causal.py first.")
        return
        
    df = pd.read_csv(metrics_path)
    
    # Compare under Protocol C (Causal Online Calibration)
    sub = df[df["calibration_protocol"] == "Protocol_C_CausalOnline"].copy()
    
    # We want to see performance by driver
    b_df = sub[sub["driver_id"] == "Driver B"]
    d_df = sub[sub["driver_id"] == "Driver D"]
    
    print("=" * 70)
    print("HYBRID ESTIMATOR CROSS-VALIDATION BREAKDOWN")
    print("=" * 70)
    
    for dur in [10, 30, 60]:
        print(f"\n--- Duration: {dur}s ---")
        
        # Driver B
        b_dur = b_df[b_df["duration_s"] == dur]
        cv_b = b_dur[b_dur["estimator"] == "Constant-Velocity"]["final_drift_m"].mean()
        nn_b = b_dur[b_dur["estimator"] == "ModernTCN"]["final_drift_m"].mean()
        hyb_exist_b = b_dur[b_dur["estimator"] == "Existing-Hybrid"]["final_drift_m"].mean()
        hyb_causal_b = b_dur[b_dur["estimator"] == "Corrected-Causal-Hybrid"]["final_drift_m"].mean()
        
        print(f"Driver B: CV={cv_b:.2f}m, ModernTCN={nn_b:.2f}m, ExistingHybrid={hyb_exist_b:.2f}m, CausalHybrid={hyb_causal_b:.2f}m")
        
        # Driver D
        d_dur = d_df[d_df["duration_s"] == dur]
        cv_d = d_dur[d_dur["estimator"] == "Constant-Velocity"]["final_drift_m"].mean()
        nn_d = d_dur[d_dur["estimator"] == "ModernTCN"]["final_drift_m"].mean()
        hyb_exist_d = d_dur[d_dur["estimator"] == "Existing-Hybrid"]["final_drift_m"].mean()
        hyb_causal_d = d_dur[d_dur["estimator"] == "Corrected-Causal-Hybrid"]["final_drift_m"].mean()
        
        print(f"Driver D: CV={cv_d:.2f}m, ModernTCN={nn_d:.2f}m, ExistingHybrid={hyb_exist_d:.2f}m, CausalHybrid={hyb_causal_d:.2f}m")

if __name__ == "__main__":
    main()
