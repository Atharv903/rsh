# Phase 7 Deliverable: Fair Baselines and Standardized Outage Evaluation Metrics

**Date:** October 2026  
**Auditor:** Senior Research Engineer & Inertial Navigation Specialist  
**Deliverable Document:** [`reports/phase7_baselines_and_metrics.md`](file:///c:/Users/shreya%20parmar/Downloads/research/reports/phase7_baselines_and_metrics.md)  
**Associated Results:**
- Baseline Results CSV: [`reports/phase7_baseline_results.csv`](file:///c:/Users/shreya%20parmar/Downloads/research/reports/phase7_baseline_results.csv)
- Baseline Summary JSON: [`reports/phase7_baseline_summary.json`](file:///c:/Users/shreya%20parmar/Downloads/research/reports/phase7_baseline_summary.json)
- Trajectory Comparison Figure: [`figures/phase7_baseline_trajectory_comparison_60s.png`](file:///c:/Users/shreya%20parmar/Downloads/research/figures/phase7_baseline_trajectory_comparison_60s.png)

---

## 1. Executive Summary

This report establishes fair, scientifically rigorous navigation baselines and standardized trajectory evaluation metrics for smartphone-based GPS-denied navigation during simulated 10-second, 30-second, and 60-second outages.

### Critical Correction to Baseline Methodology:
The previous audit reported an unrealistic "0.596 m/s" persistence baseline. As proven in Phase 1 and Phase 3, that number was achieved by using the true ground-truth GPS velocity from 1 second prior at every step of a 90%-overlapping sliding window. During a genuine GPS outage, intermediate GPS fixes do not exist. Accessing them represents illegal future information leakage.

We replace that flawed formulation with **four strictly causal baselines** initialized strictly from legitimate pre-outage state:
1. **Zero-Speed Baseline:** Predicts vehicle remains stationary ($\hat{s}(t) = 0$).
2. **Last-Known-Speed Baseline:** Holds the pre-outage speed $s(t_0)$ constant throughout the outage, with causal gyroscope heading propagation.
3. **Constant-Velocity Baseline:** Holds both pre-outage speed $s(t_0)$ and pre-outage heading $\psi(t_0)$ constant (straight-line dead reckoning).
4. **IMU-Kinematic Baseline:** Damped integration of longitudinal accelerometer dynamics combined with causal gyroscope heading propagation.

---

## 2. Standardized Metric Definitions

All evaluations are conducted step-by-step using actual elapsed $\Delta t_k = t_k - t_{k-1}$:

1. **Speed Mean Absolute Error (MAE):**
   $$\text{MAE}_{\text{speed}} = \frac{1}{K} \sum_{k=1}^K | \hat{s}(t_k) - s^*(t_k) | \quad (\text{m/s})$$
2. **Speed Root Mean Square Error (RMSE):**
   $$\text{RMSE}_{\text{speed}} = \sqrt{ \frac{1}{K} \sum_{k=1}^K (\hat{s}(t_k) - s^*(t_k))^2 } \quad (\text{m/s})$$
3. **Speed Bias:**
   $$\text{Bias}_{\text{speed}} = \frac{1}{K} \sum_{k=1}^K (\hat{s}(t_k) - s^*(t_k)) \quad (\text{m/s})$$
4. **Absolute Trajectory Error (ATE RMSE):**
   Root mean square Euclidean distance between estimated and ground-truth Local ENU positions along the full outage trajectory:
   $$\text{ATE RMSE} = \sqrt{ \frac{1}{K} \sum_{k=1}^K \left[ (\hat{p}_E(t_k) - p_E^*(t_k))^2 + (\hat{p}_N(t_k) - p_N^*(t_k))^2 \right] } \quad (\text{m})$$
5. **Final Position Drift:**
   Euclidean error at the final step of the outage:
   $$\text{Final Drift} = \sqrt{ (\hat{p}_E(t_K) - p_E^*(t_K))^2 + (\hat{p}_N(t_K) - p_N^*(t_K))^2 } \quad (\text{m})$$
6. **Relative Drift Percentage:**
   $$\text{Drift } \% = \frac{\text{Final Drift}}{\max(\text{Total Ground-Truth Path Length}, 1.0\text{ m})} \times 100\%$$

---

## 3. Empirical Baseline Benchmark on Independent Holdout Test Set (Drive S4)

Evaluated across 15 matched simulated outage trials on Drive S4 (unseen independent day):

| Outage Duration | Navigation Baseline | Speed MAE (m/s) | Speed RMSE (m/s) | ATE RMSE (m) | Final Drift (m) | Relative Drift (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **10 Seconds** | **Zero-Speed** | 11.35 | 11.42 | 66.33 | 114.49 | 100.9% |
| | **Last-Known-Speed** | 1.82 | 2.12 | 16.67 | 32.37 | 28.6% |
| | **Constant-Velocity** | 1.82 | 2.12 | 12.01 | 21.79 | **19.2%** |
| | **IMU-Kinematic** | **1.55** | **1.86** | 16.41 | 32.27 | 28.3% |
| **30 Seconds** | **Zero-Speed** | 9.02 | 9.58 | 165.45 | 266.98 | 98.6% |
| | **Last-Known-Speed** | 2.44 | 2.98 | 125.11 | 238.46 | 98.0% |
| | **Constant-Velocity** | 2.44 | 2.98 | **67.22** | **131.07** | **61.9%** |
| | **IMU-Kinematic** | 2.90 | 3.51 | 120.64 | 233.11 | 91.9% |
| **60 Seconds** | **Zero-Speed** | 7.33 | 8.24 | 239.80 | 371.38 | 83.9% |
| | **Last-Known-Speed** | 3.50 | 4.22 | 219.63 | 388.84 | 88.8% |
| | **Constant-Velocity** | 3.50 | 4.22 | **178.12** | 360.95 | **84.8%** |
| | **IMU-Kinematic** | **3.23** | **3.94** | 203.80 | **359.04** | 79.5% |

---

## 4. Scientific Observations on Baseline Performance

1. **Short Outages (10s):** Constant-velocity and last-known-speed baselines achieve strong performance ($21.8\text{ m}$ drift, 19.2%), because vehicle speed and heading do not change drastically over 10 seconds.
2. **Medium to Long Outages (30s & 60s):** Constant velocity degrades as vehicles turn corners or alter speed, with drift exceeding $130 - 360\text{ metres}$ ($62\% - 85\%$).
3. **IMU Kinematic Integration:** Simple naive integration of noisy accelerometer features drifts quickly over 60 seconds without neural filtering, achieving $359\text{ m}$ final drift.
4. **The Target for Neural Models:** To claim genuine machine learning value, deep learning models (CNN-GRU, ModernTCN, Mamba-2) must beat the **Last-Known-Speed** and **Constant-Velocity** baselines on both Speed MAE ($<2.0\text{ m/s}$ at 30s) and Trajectory Drift ($<50\%$ at 60s).

*Phase 7 Baseline Benchmark complete. All requirements for Phases 4–7 satisfied.*
