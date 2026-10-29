"""
Phase 14: Comprehensive Leakage and Provenance Audit Table.
Generates experiments/phase14_validation/leakage_audit.csv.
"""

from pathlib import Path
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent
out_dir = repo_root / "experiments/phase14_validation"
out_dir.mkdir(parents=True, exist_ok=True)

leakage_data = [
    {
        "Potential leakage": "Session-Wide Gyroscope Bias Averaging (Global Stationary)",
        "Code location": "experiments/phase13/audit_gyro_bias_experiments.py:110-115",
        "Evidence": "global_stat_biases={'M': +0.004763, 'Y1': -0.001160} was averaged over all 33/29 stops across the entire 100+ minute drives, including stops occurring long after evaluated outages.",
        "Severity": "Critical",
        "Required correction": "Restrict bias estimation strictly to intervals prior to outage start (Protocol B: Prior calibration at trip start; Protocol C: Causal online updates).",
        "Status": "Corrected in Phase 14 Protocols B & C"
    },
    {
        "Potential leakage": "Stationary Detection via Reference CAN-Bus / RTK-GPS Speed",
        "Code location": "experiments/phase13/evaluate_hybrid_dead_reckoning.py:33",
        "Evidence": "is_stat = (pre_data['ref_speed_mps'] < 0.1).values uses vehicle OBD/RTK reference speed to identify stationary intervals.",
        "Severity": "Moderate",
        "Required correction": "Implement autonomous, sensor-only stationary detection using IMU acceleration variance and angular rate magnitude (ZUPT/ZARU) or pre-outage GNSS fixes.",
        "Status": "Corrected in Phase 14 Protocol C"
    },
    {
        "Potential leakage": "Hybrid Transition Parameter Optimization on Evaluation Set",
        "Code location": "experiments/phase13/evaluate_hybrid_dead_reckoning.py:171",
        "Evidence": "t_switch=10.0s was selected after observing in Phase 12 that CV won at 10s while ModernTCN won at 30s on the exact same 30 test scenarios.",
        "Severity": "High",
        "Required correction": "Treat existing hybrid as exploratory; evaluate cross-validation (tune on Driver B, test on Driver D, and vice versa) and report generalization gaps.",
        "Status": "Audited & Cross-Validated in Phase 14"
    },
    {
        "Potential leakage": "Instantaneous Accelerometer Spike Latching in Braking Override",
        "Code location": "tests/test_gyro_bias_and_hybrid.py:39-42",
        "Evidence": "np.maximum.accumulate(accel_x < -1.5) latched alpha=1.0 for the rest of the outage from a single 0.1s road vibration noise spike in 24 of 30 scenarios.",
        "Severity": "Moderate",
        "Required correction": "Require sustained deceleration (rolling mean over >= 0.5s) to avoid false latching on vibration spikes.",
        "Status": "Corrected in Phase 14 Causal Hybrid"
    },
    {
        "Potential leakage": "Pre-Outage Window Dynamic Turn-Rate Bias Corruption",
        "Code location": "src/outage_estimator.py:192 / scripts/evaluate_phase12_expanded.py:192",
        "Evidence": "gyro_bias = np.median(pre_df['gyro_pitch']) took moving window turn rate (up to 4.25 deg/s) during active driving and subtracted it as sensor bias.",
        "Severity": "Critical",
        "Required correction": "Never subtract moving median during driving as bias; use Protocol A (zero bias), Protocol B (prior stationary), or Protocol C (causal ZARU).",
        "Status": "Corrected in Phase 14 Protocols A, B, C"
    },
    {
        "Potential leakage": "Post-Outage Telemetry Ingestion During Navigation",
        "Code location": "src/outage_estimator.py:136-160",
        "Evidence": "Dead-reckoning loop uses strictly causal sliding windows [st+k-100:st+k] and pred_speed[k]. No future IMU or reference telemetry enters the loop.",
        "Severity": "None",
        "Required correction": "None; verified strictly causal.",
        "Status": "Verified Clean"
    },
    {
        "Potential leakage": "GPS Fix Leakage During Denied Outage Interval",
        "Code location": "src/outage_estimator.py:145-165",
        "Evidence": "Reference GPS coordinates (ref_east_m, ref_north_m) and ref_speed_mps are accessed only after the outage simulation loop finishes, purely for metric evaluation.",
        "Severity": "None",
        "Required correction": "None; verified strictly causal.",
        "Status": "Verified Clean"
    },
    {
        "Potential leakage": "Normalizer Training Set Data Contamination",
        "Code location": "data/processed/sensor_normalizer_grouped.json",
        "Evidence": "Scaler mean and std were fit exclusively on training drives S1 and S2. Test sessions M (Driver B) and Y1 (Driver D) were held out entirely.",
        "Severity": "None",
        "Required correction": "None; verified strictly out-of-sample.",
        "Status": "Verified Clean"
    }
]

df_leakage = pd.DataFrame(leakage_data)
df_leakage.to_csv(out_dir / "leakage_audit.csv", index=False)
print(f"Saved leakage audit to {out_dir / 'leakage_audit.csv'}")
