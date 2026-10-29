# IO-VNBD GPS-Denied Dead Reckoning: Final Reproduction Guide

**Milestone:** Final Research Freeze  
**Date:** October 30, 2026  
**Repository:** [https://github.com/Atharv903/rsh](https://github.com/Atharv903/rsh)  
**Verified Git Commit:** `0565426`  

---

## 1. System Requirements & Environment Setup

### 1.1 Software & Hardware Requirements
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python Version:** Python 3.10.x (verified on 3.10.11)
- **Virtual Environment:** Recommended `.venv`
- **Hardware:** Standard multi-core x86_64 CPU (CPU inference only; no GPU required for evaluation)
- **RAM:** Minimum 8 GB (16 GB recommended)
- **Disk Space:** ~500 MB for repository code, checkpoints, and evaluation artifacts (raw dataset requires ~1.2 GB)

### 1.2 Python Dependencies
The virtual environment requires the following verified packages:
```bash
pip install torch==2.14.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scipy matplotlib pytest pyyaml
```

---

## 2. Running Automated Tests

All tests can be executed directly using `pytest` from the repository root:

### 2.1 Execute the Full Unit & Regression Test Suite (30 Tests)
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_phase16_eskf.py tests/test_phase16_benchmark.py tests/test_phase15_benchmark.py tests/test_phase14_causal_integrity.py tests/test_gyro_bias_and_hybrid.py tests/test_navigation_heading.py -v
```
*Expected Outcome:* **30 passed in ~4.5–5.0 seconds**, zero failures, zero warnings.

### 2.2 Execute Only the Phase 16 ESKF Filter Tests (6 Tests)
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_phase16_eskf.py -v
```
*Expected Outcome:* 6 passed in ~0.25 seconds (verifies state kinematics, covariance symmetry, positive definiteness, Joseph update, speed update, NHC, and ZUPT).

---

## 3. Reproducing Benchmark Evaluations

### 3.1 Reproduce the Phase 16 Controlled Ablation Benchmark (8 Configurations x 3 Protocols x 30 Scenarios)
To re-run the complete evaluation pipeline:
```bash
.\.venv\Scripts\python.exe experiments/phase16_validation/evaluate_phase16_ablation.py
```
*Expected Runtime:* ~3.5 minutes on desktop CPU (runs 720 scenario evaluations).  
*Generated Output Artifacts:*
- `experiments/phase16_validation/per_scenario_metrics.csv` (720 records)
- `experiments/phase16_validation/summary_metrics.csv` (216 summary rows)
- `experiments/phase16_validation/filter_diagnostics.csv` (90 scenario diagnostic rows)
- `experiments/phase16_validation/config.json`

### 3.2 Reproduce Paired Statistical Analysis & Bootstrap Confidence Intervals
```bash
.\.venv\Scripts\python.exe experiments/phase16_validation/statistical_analysis_phase16.py
```
*Expected Runtime:* ~1.2 minutes (runs 10,000 bootstrap iterations per comparison pair).  
*Generated Output Artifacts:*
- `experiments/phase16_validation/statistical_paired_comparisons.csv` (189 paired comparisons)

### 3.3 Regenerate Publication-Quality Figures
```bash
.\.venv\Scripts\python.exe scripts/plot_phase16_figures.py
```
*Expected Runtime:* ~5 seconds.  
*Generated Output Artifacts:*
- `figures/phase16_ablation_drift_by_duration.png`
- `figures/phase16_nhc_crosstrack_suppression.png`
- `figures/phase16_eskf_covariance_and_innovations.png`

---

## 4. Key Configuration Files & Data Paths

- **Scenario Manifest:** `experiments/phase12_validation/phase12_scenarios.json` (defines the 30 standardized outage scenarios: 5x10s, 5x30s, 5x60s for Driver B and Driver D).
- **Sensor Normalizer:** `data/processed/sensor_normalizer_grouped.json` (fit exclusively on Driver S training partitions).
- **Trained Neural Checkpoints:**
  - `experiments/phase8_corrected_speed_models/checkpoints/moderntcnmodel_seed42/best_model.pt`
  - `experiments/phase8_corrected_speed_models/checkpoints/moderntcnmodel_seed123/best_model.pt`
  - `experiments/phase8_corrected_speed_models/checkpoints/moderntcnmodel_seed456/best_model.pt`
- **Raw Evaluation Data:**
  - Driver B: `Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)`
  - Driver D: `Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/Y (Driver D)/Y1`

---

## 5. Known Hardware, Runtime & Data Limitations

1. **CPU Execution:** All evaluation code is CPU-compatible. GPU runtime is not required for inference.
2. **Evaluation Time:** A complete 720-scenario ablation run requires ~3.5 minutes on multi-core CPU.
3. **Data Access:** Raw dataset files must remain intact under `Synchronised V abd S datasets/`.
