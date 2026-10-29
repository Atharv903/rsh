# Phase 14 Reproduction & Integrity Audit

**Audit Timestamp:** October 30, 2026  
**Audited Baseline Commit:** `45230a3`  
**Repository:** [https://github.com/Atharv903/rsh](https://github.com/Atharv903/rsh)  
**Machine-Readable Audit Table:** [`experiments/phase15_validation/phase14_reproduction_audit.csv`](file:///c:/Users/shreya%20parmar/Downloads/research/experiments/phase15_validation/phase14_reproduction_audit.csv)

---

## 1. Executive Summary

Prior to proposing architectural or algorithmic interventions in Phase 15, we conducted a forensic verification of the baseline results established at commit `45230a3` (Phase 14).

Every scenario ID, duration, start timestamp, sensor normalizer parameter, calibration protocol, and summary metric was reconciled against executable code and underlying raw scenario tables.

### Audit Verdict: **FULL REPRODUCIBILITY PASS (Zero Discrepancies)**

All 360 scenario evaluations (30 scenarios $\times$ 3 calibration protocols $\times$ 4 navigation estimators) and 108 summary metric rows in `experiments/phase14_validation/summary_metrics.csv` match the scenario-level ground truth exactly (maximum absolute error $< 10^{-6}$).

---

## 2. Item-by-Item Verification Findings

| Audit Dimension | Evaluated Requirement | Audit Result | Status |
|:---|:---|:---|:---:|
| **1. Scenario Manifest Alignment** | 30 scenarios from `experiments/phase12_validation/phase12_scenarios.json` | Exact match on all IDs, durations (10s, 30s, 60s), and start/end indices. | **PASS** |
| **2. Driver & Session Independence** | Training on Driver S; Testing on Driver B & D | Strict holdout. Normalizer and checkpoints fit strictly on Driver S; tested on B and D. | **PASS** |
| **3. Initial State Conditioning** | Identical initial position, speed, and heading across estimators | All estimators share the identical pre-outage state $s_0 = [p_{e0}, p_{n0}, v_0, \psi_0]$. | **PASS** |
| **4. Gyroscope Bias Protocol A** | Uncalibrated baseline ($b \equiv 0.0$ rad/s) | Verified strictly $0.0$ rad/s across all 30 scenarios. | **PASS** |
| **5. Gyroscope Bias Protocol B** | Prior calibration from initial pre-departure stop | Fixed at $+0.003177$ rad/s (Driver B) and $-0.005010$ rad/s (Driver D). Zero future leakage. | **PASS** |
| **6. Gyroscope Bias Protocol C** | Causal online update from most recent pre-outage stop | Every calibration stop timestamp is strictly earlier than outage onset ($t_{\text{stop}} < t_{\text{outage}}$). | **PASS** |
| **7. Feature Normalizer Integrity** | Scaler parameters fit strictly on Driver S | `sensor_normalizer_grouped.json` fit on `S-S1` and `S-S2`. Zero test leakage. | **PASS** |
| **8. Summary Metric Aggregation** | Mean, median, std, and IQR match per-scenario metrics | Reconciled across all 108 summary rows with zero numerical discrepancy. | **PASS** |
| **9. Causal Inference & Data Isolation** | Window $[st+k-100 : st+k]$ contains only past IMU; zero GPS fixes | Verified strictly causal; no future samples or CAN reference speed accessed during outage. | **PASS** |
| **10. Parameter Cross-Validation** | Sigmoid transition parameters ($t_{\text{switch}}=10.0$s, $\tau=2.0$s) | Leave-One-Driver-Out Cross-Validation (LOOCV) confirms smooth generalization between B and D. | **PASS** |

---

## 3. Confirmed Historical Reference Metrics (Protocol C — Causal Online Calibration)

The verified reference performance at commit `45230a3` under the primary operational protocol (Protocol C) is:

| Outage Duration | Constant-Velocity Drift (m) | ModernTCN Drift (m) | Phase 14 Causal Hybrid Drift (m) | Winning Estimator |
|:---:|:---:|:---:|:---:|:---:|
| **10 Seconds** | **$12.98$ m** (med $11.08$) | $29.15$ m (med $23.11$) | **$12.94$ m** (med $11.08$) | **Causal Hybrid / CV** |
| **30 Seconds** | $96.57$ m (med $82.72$) | $72.56$ m (med $57.06$) | **$61.21$ m** (med $47.34$) | **Causal Hybrid ($-36.6\%$)** |
| **60 Seconds** | $141.41$ m (med $92.44$) | $162.32$ m (med $88.68$) | **$144.23$ m** (med **$75.69$ m**) | **Causal Hybrid (Lowest Median)** |

This confirms the frozen historical baseline for all Phase 15 comparisons.
