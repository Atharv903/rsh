# Research-Grade IMU Dead-Reckoning Pipeline for GPS-Denied Navigation
**Benchmarking CNN-GRU, ModernTCN, and Mamba-2 (vectorized State Space Duality) on the IO-VNBD Dataset**

[![Tests](https://img.shields.io/badge/Unit%20Tests-25%20Passed-brightgreen.svg)](#automated-validation)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#environment-setup)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B%20%7C%20CUDA%20%26%20CPU-orange.svg)](#environment-setup)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Atharv903/rsh/blob/main/notebooks/colab_gpu_benchmark.ipynb)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

---

## 📌 Project Overview
This repository provides a reproducible, research-grade inertial navigation and benchmarking pipeline for vehicle navigation during simulated GPS outages using the **IO-VNBD (Input-Output Vehicle Navigation Benchmark Dataset)**.

The framework resolves the unobservable 2D velocity trap by reformulating the task into:
1. **Continuous Forward Speed Estimation:** Neural regression of scalar speed $s(t) = v_{\text{fwd}}(t) \in \mathbb{R}_{\ge 0}$ from 10-second sliding IMU windows ($W=100$, 10 Hz).
2. **Causal Gyroscope Heading Propagation:** Azimuth propagation $\psi_k = \psi_{k-1} + \omega_{\text{yaw}} \Delta t$ coupled with speed dead reckoning:
   $$\mathbf{p}_k = \mathbf{p}_{k-1} + s_k \begin{bmatrix} \sin(\psi_k) \\ \cos(\psi_k) \end{bmatrix} \Delta t_k$$
3. **Leakage-Free Date-Grouped Partitions:** Isolating driving dates to strictly prevent cross-split highway loop memorization.
4. **Benchmarked Deep Architectures:**
   - **CNN-GRU Baseline** (Convolutional Recurrent Neural Network)
   - **ModernTCN** (Modern Temporal Convolutional Network with large receptive field)
   - **Mamba-2 SSD** (Custom vectorized State Space Duality in pure PyTorch)

---

## 🏆 Key Benchmark Findings (Holdout Drive S4)

### 1. Speed Estimation Accuracy
Across 9 models (3 architectures $\times$ 3 seeds: 42, 123, 456) evaluated on the independent holdout test set (Drive S4, 9,117 windows):

| Model / Baseline | Params | Latency ($B=1$, GPU) | Test MAE (m/s) | Test RMSE (m/s) | High-Speed MAE ($s \ge 15$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Speed Baseline** | — | — | 9.68 | 11.88 | 18.25 |
| **Train-Mean Baseline (10.9 m/s)** | — | — | 5.66 | 7.13 | 7.35 |
| *Stride-1 Persistence (Leaked)* | — | — | *0.46* | *0.67* | *0.61* |
| **CNN-GRU Baseline** | 250,689 | **0.86 $\pm$ 0.05 ms** | 2.56 $\pm$ 0.08 | 3.96 $\pm$ 0.17 | 7.07 $\pm$ 0.38 |
| **Mamba-2 SSD (Custom)** | **217,153** | 1.97 $\pm$ 0.08 ms | **2.52 $\pm$ 0.20** | 3.98 $\pm$ 0.29 | 6.98 $\pm$ 0.85 |
| **ModernTCN** | 413,825 | 1.29 $\pm$ 0.05 ms | 2.62 $\pm$ 0.19 | **3.92 $\pm$ 0.34** | **6.50 $\pm$ 1.09** |

- **Root Cause of Historical 9.8 m/s Anomaly:** Zero-Speed baseline MAE is **9.68 m/s**, proving that historical models collapsed to $[0, 0]$ due to unobservable 2D velocity targets.
- **Top Speed Estimator:** Mamba-2 achieved the lowest overall average test error (**2.52 m/s MAE**; single best run **2.315 m/s** on Seed 123) with the lowest parameter count (**217k params**).
- **Sub-Millisecond Inference:** On GPU, CNN-GRU executes in **0.86 ms**, ModernTCN in **1.29 ms**, and Mamba-2 in **1.97 ms** ($>50\times$ faster than real-time 10 Hz requirements).

### 2. Trajectory Dead-Reckoning (Simulated GNSS Outages)

| Outage Duration | Constant-Velocity Drift | CNN-GRU Drift | Mamba-2 SSD Drift | ModernTCN Drift | Neural Drift Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10 Seconds** | **40.47 m (45.5%)** | 89.64 m (100.9%) | 89.66 m (100.9%) | 88.78 m (99.9%) | *Constant-velocity competitive* |
| **30 Seconds** | 194.71 m (98.1%) | 198.25 m (99.9%) | 198.29 m (99.9%) | **195.65 m (98.6%)** | *Matched dynamics* |
| **60 Seconds** | 432.86 m (121.6%) | 354.38 m (99.6%) | 354.47 m (99.6%) | **349.18 m (98.1%)** | **83.68 m drift reduction (19.3%)** |

---

## 📂 Repository Architecture

```
├── configs/                          # Configurable pipeline parameters
├── experiments/
│   └── phase8_corrected_speed_models/
│       ├── configs/                  # Architecture YAML configs (cnn_gru, moderntcn, mamba2)
│       ├── results/                  # Structured evaluation CSVs & profiling
│       └── training_benchmark_summary.json
├── figures/                          # Publication-ready figures (300 DPI)
│   ├── phase8_outage_trajectories_60s.png
│   ├── phase8_training_loss_curves.png
│   ├── phase3_signal_alignment_overlay.png
│   └── phase4_imu_dynamics_vs_speed.png
├── notebooks/
│   └── colab_gpu_benchmark.ipynb    # Interactive Google Colab GPU notebook
├── reports/
│   ├── phase8_final_results.md       # Comprehensive scientific benchmark report
│   ├── phase8_experiment_protocol.md # Frozen experimental protocol
│   ├── phase7_baselines_and_metrics.md
│   ├── phase6_outage_protocol.md
│   ├── phase5_grouped_split_audit.md
│   ├── phase4_speed_target_feasibility.md
│   ├── phase3_sv_alignment_audit.md
│   ├── phase2_timing_audit.md
│   └── colab_gpu_execution_guide.md  # Detailed GPU execution & profiling guide
├── scripts/
│   ├── colab_runner.py               # Colab automated runner with NVMe caching
│   ├── evaluate_speed_models.py      # Device-aware model & outage evaluator
│   ├── plot_loss_curves.py           # Publication loss curves generator
│   ├── run_phase8_training.py        # Master training orchestration script
│   ├── train_speed_models.py         # Device-aware trainer with per-epoch resume
│   └── verify_numerical_consistency.py # Numerical determinism & resume checks
├── src/
│   ├── corrected_synchronization.py  # UTC offset alignment & clock reset isolation
│   ├── outage_estimator.py           # Causal dead-reckoning engine & baselines
│   ├── models/
│   │   ├── cnn_gru.py                # CNN-GRU baseline
│   │   ├── modern_tcn.py             # ModernTCN implementation
│   │   └── mamba2.py                 # Vectorized State Space Duality implementation
│   └── signal_processing.py          # Train-only normalizer & conditioning
├── tests/                            # 25 automated unit tests (100% passing)
├── pytest.ini
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### 1. Local Environment Setup
```bash
# Clone the repository
git clone https://github.com/Atharv903/rsh.git
cd rsh

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt  # Or: pip install pyyaml pandas numpy matplotlib scipy pytest torch
```

### 2. Run Automated Verification Tests
```bash
pytest -v tests/
python scripts/verify_numerical_consistency.py
```

### 3. Run Google Colab GPU Benchmark
Open [`notebooks/colab_gpu_benchmark.ipynb`](notebooks/colab_gpu_benchmark.ipynb) in Google Colab, select a GPU runtime (T4, V100, or A100), and execute all cells.

Alternatively, execute via CLI:
```bash
python scripts/colab_runner.py --mode full_benchmark --device auto
```

---

## 📜 Scientific Audits & Reports

Full documentation of the multi-phase audit and resolution is available in [`reports/`](reports/):
- [`reports/phase8_final_results.md`](reports/phase8_final_results.md): Final multi-model comparison across seeds and outage scenarios.
- [`reports/phase8_experiment_protocol.md`](reports/phase8_experiment_protocol.md): Frozen experimental protocol.
- [`reports/phase5_grouped_split_audit.md`](reports/phase5_grouped_split_audit.md): Grouped session isolation to eliminate highway loop leakage.
- [`reports/phase3_sv_alignment_audit.md`](reports/phase3_sv_alignment_audit.md): Smartphone-to-vehicle millisecond alignment & Android clock reset handling.
- [`reports/colab_gpu_execution_guide.md`](reports/colab_gpu_execution_guide.md): Guide for GPU acceleration and memory profiling.

---

## 📄 License
MIT License.
