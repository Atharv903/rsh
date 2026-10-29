# IO-VNBD GPS-Denied Vehicle Dead Reckoning: Project Handoff

**Milestone:** Final Research Freeze & Project Handoff  
**Date:** October 30, 2026  
**Repository:** [https://github.com/Atharv903/rsh](https://github.com/Atharv903/rsh)  
**Final Frozen Commit:** `0565426` (Phase 16)  
**Git Working Tree Status:** Clean; all artifacts version-controlled and pushed to `origin/main`  

---

## 1. Executive Status & Test Outcomes

### 1.1 Test Suite Verification
- **Total Tests Executed:** 30 automated pytest cases across 6 test modules.
- **Outcome:** **30 passed in 4.64s**, 0 failed, 0 warnings.
  - `tests/test_phase16_eskf.py` (6 passed): Validates ESKF state propagation, covariance symmetry, positive definiteness, speed update, NHC update, and ZUPT.
  - `tests/test_phase16_benchmark.py` (5 passed): Validates Phase 16 dimensions (720 records), math reconciliation, and unbiased innovations ($|\bar{y}| < 0.02\,\text{m/s}$).
  - `tests/test_phase15_benchmark.py` (6 passed): Validates Phase 15 benchmark outputs and literature integrity.
  - `tests/test_phase14_causal_integrity.py` (4 passed): Validates future IMU perturbation invariance and zero future stop leakage.
  - `tests/test_gyro_bias_and_hybrid.py` (3 passed): Validates hybrid sigmoid transition and braking override.
  - `tests/test_navigation_heading.py` (6 passed): Validates coordinate azimuth conventions and heading integration sign ($d\psi/dt = -\text{GyroPitch}$).

---

## 2. Final Validated Baseline & Experimental Alternative

### 2.1 Recommended Operational Baseline: Phase 14/15 Causal Hybrid (Forward Euler)
- **Mathematical Specification:**
  - Forward speed: Blends Constant-Velocity ($v_0$) with ModernTCN ensemble via a causal sigmoid transition ($t_{\text{switch}} = 10.0$s, $\tau = 2.0$s).
  - Braking override: Rolling mean deceleration ($\ge 0.5$s window) $< -1.5\,\text{m/s}^2$ latches $\alpha = 1.0$.
  - Heading integration: $\psi_k = \psi_{k-1} - (\tilde{\omega}_k - b_g) \Delta t$ under Protocol C (Causal Online Stationary Calibration).
- **Validated Performance (Protocol C):**
  - **10s Outage:** **$12.94\,\text{m}$** drift (matches Constant-Velocity, cuts ModernTCN penalty by 55.6%).
  - **30s Outage:** **$61.21\,\text{m}$** drift (median **$40.06\,\text{m}$**, a **$36.6\%$ error reduction** over CV).
  - **60s Outage:** **$144.23\,\text{m}$** drift (median **$75.69\,\text{m}$**, lowest median drift among all tested methods).
- **Role:** Lowest overall endpoint drift, lowest computational footprint ($4.3\,\text{ms}$ on CPU), zero filter lag.

### 2.2 Validated Experimental Alternative: Uncertainty-Aware ESKF with NHC
- **Mathematical Specification:**
  - 6D Error-State Kalman Filter ($\mathbf{x} = [p_e, p_n, v_e, v_n, \psi, b_g]^T$) with Joseph-stabilized covariance.
  - Measurement updates: Forward speed ($R_v = \sigma_{\text{base}}^2 + 4.0\sigma_{\text{ens}}^2$), Non-Holonomic Constraints ($v_{\text{lat}} \approx 0$ with $3\sigma$ innovation gate), and causal Zero-Velocity Updates (ZUPT).
- **Validated Performance (Protocol C):**
  - 60s Outage with NHC: **$213.49\,\text{m}$** drift (cross-track error suppressed to **$-0.39\,\text{m}$**).
- **Role:** Essential when formal state uncertainty bounds ($\mathbf{P}$) or multi-sensor fusion (e.g. wheel ticks, map matching) are required.

---

## 3. Directory of Key Source Files, Reports & Datasets

### 3.1 Primary Source Code Modules
- [`src/eskf.py`](file:///c:/Users/shreya%20parmar/Downloads/research/src/eskf.py): Modular Error-State Kalman Filter implementation.
- [`src/outage_estimator.py`](file:///c:/Users/shreya%20parmar/Downloads/research/src/outage_estimator.py): Causal pre-outage state estimation and dead reckoning.
- [`src/corrected_synchronization.py`](file:///c:/Users/shreya%20parmar/Downloads/research/src/corrected_synchronization.py): High-fidelity continuous 10 Hz timebase alignment.
- [`src/models/modern_tcn.py`](file:///c:/Users/shreya%20parmar/Downloads/research/src/models/modern_tcn.py): ModernTCN neural architecture.

### 3.2 Evaluation Scripts & Benchmarks
- [`experiments/phase16_validation/evaluate_phase16_ablation.py`](file:///c:/Users/shreya%20parmar/Downloads/research/experiments/phase16_validation/evaluate_phase16_ablation.py): Controlled 8-way ablation pipeline.
- [`experiments/phase16_validation/statistical_analysis_phase16.py`](file:///c:/Users/shreya%20parmar/Downloads/research/experiments/phase16_validation/statistical_analysis_phase16.py): Bootstrap CIs and paired tests.
- [`scripts/plot_phase16_figures.py`](file:///c:/Users/shreya%20parmar/Downloads/research/scripts/plot_phase16_figures.py): Publication figure generator.

### 3.3 Core Deliverables & Reports
- Final Master Report: [`reports/final_research_report.md`](file:///c:/Users/shreya%20parmar/Downloads/research/reports/final_research_report.md)
- Reproduction Guide: [`reports/final_reproduction_guide.md`](file:///c:/Users/shreya%20parmar/Downloads/research/reports/final_reproduction_guide.md)
- Literature Addendum: [`reports/phase16_literature_addendum.md`](file:///c:/Users/shreya%20parmar/Downloads/research/reports/phase16_literature_addendum.md)
- ESKF Design Specification: [`reports/phase16_eskf_design.md`](file:///c:/Users/shreya%20parmar/Downloads/research/reports/phase16_eskf_design.md)
- Phase 15 Error Decomposition: [`reports/phase15_error_decomposition.md`](file:///c:/Users/shreya%20parmar/Downloads/research/reports/phase15_error_decomposition.md)

### 3.4 Validated Machine-Readable CSVs
- Phase 16 Per-Scenario Results: [`experiments/phase16_validation/per_scenario_metrics.csv`](file:///c:/Users/shreya%20parmar/Downloads/research/experiments/phase16_validation/per_scenario_metrics.csv)
- Phase 16 Summary Metrics: [`experiments/phase16_validation/summary_metrics.csv`](file:///c:/Users/shreya%20parmar/Downloads/research/experiments/phase16_validation/summary_metrics.csv)
- Phase 16 Paired Statistics: [`experiments/phase16_validation/statistical_paired_comparisons.csv`](file:///c:/Users/shreya%20parmar/Downloads/research/experiments/phase16_validation/statistical_paired_comparisons.csv)
- Filter Diagnostics: [`experiments/phase16_validation/filter_diagnostics.csv`](file:///c:/Users/shreya%20parmar/Downloads/research/experiments/phase16_validation/filter_diagnostics.csv)

---

## 4. Known Critical Issues

- **Zero Critical Blockers:** All historical bugs (moving-median turn rate corruption, inverted heading sign convention, future-stop leakage, numerical covariance collapse) have been resolved, verified, and locked with unit tests.
- **Documented Caveats:**
  1. Test cohort is limited to Drivers B and D ($N=2$ drives, 30 scenarios).
  2. Neural speed estimation on unseen vehicles exhibits $2.7–3.3\,\text{m/s}$ MAE due to vehicle vibration domain mismatch.

---

## 5. Short List of Recommended Future Directions (Max 5 Items)

1. **Pre-Outage Attitude & Gravity Alignment:** Implement an explicit attitude observer (estimating the tilt quaternion from pre-outage gravity) to project phone IMU features directly into the vehicle's forward/lateral/vertical axes.
2. **Spectral Vibration Whitening:** Apply frequency-domain bandstop filtering or adaptive whitening to remove vehicle-specific engine combustion harmonics before feeding ModernTCN.
3. **Multi-Driver Cohort Expansion ($N \ge 10$):** Evaluate the frozen pipeline on 10+ open-source smartphone vehicle drives (e.g., OFFGRID dataset or Oxford RobotCar phone subset) to upgrade findings to population-level statistical significance.
4. **Road Network Vector Map Matching Track:** Add an optional, cleanly separated Map-Matching module (e.g. Hidden Markov Model on OpenStreetMap road centerlines) to constrain along-track and cross-track drift along road geometry.
5. **On-Device Android Embedded Deployment:** Convert the frozen ModernTCN ensemble and ESKF to ONNX Runtime Mobile / LiteRT and benchmark real-time power consumption and thermal throttling on a physical Android smartphone.
