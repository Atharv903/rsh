# Research-Grade IMU Preprocessing Pipeline for GPS-Denied Navigation
**Benchmark Platform for CNN-GRU, ModernTCN, and Mamba Models on the IO-VNBD Dataset**

[![Tests](https://img.shields.io/badge/Unit%20Tests-12%20Passed-brightgreen.svg)](#automated-validation)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#environment-setup)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

---

## 📌 Project Overview
This repository provides a reproducible, research-grade preprocessing and validation pipeline for inertial navigation using smartphone IMU and GNSS data from the **IO-VNBD (Input-Output Vehicle Navigation Benchmark Dataset)**.

The pipeline prepares clean, synchronized, leakage-free time-series windows and simulated GPS outage benchmarks (10s, 30s, and 60s) for training and evaluating next-generation deep learning inertial dead-reckoning architectures:
- **CNN-GRU** (Convolutional Recurrent Neural Network)
- **ModernTCN** (Modern Temporal Convolutional Network)
- **Mamba** (Selective State Space Model for long-range sequence modeling)

---

## 📂 Multi-Drive Dataset Architecture

The pipeline processes and standardizes all 6 drive recordings in Driver A (~8.58 hours of driving data) into clean, drive-level train, validation, and test partitions:

```
research/
├── configs/
│   └── default_config.yaml           # Configurable pipeline parameters
├── data/
│   ├── raw/                          # Preserved raw recordings (never modified)
│   │   ├── S-S1.csv, V-S1.csv
│   │   ├── S-S2.csv, V-S2.csv
│   │   ├── S-S3a.csv, V-S3a.csv
│   │   ├── S-S3b.csv, V-S3b.csv
│   │   ├── S-S3c.csv, V-S3c.csv
│   │   └── S-S4.csv, V-S4.csv
│   ├── processed/                    # Conditioned datasets & model-ready tensors
│   │   ├── train_windows.npz         # 23,994 windows (S1, S2, S4)
│   │   ├── val_windows.npz           # 2,453 windows (S3a)
│   │   ├── test_windows.npz          # 4,381 windows (S3b, S3c)
│   │   ├── sensor_normalizer.json    # Fitted scaler parameters (train-only)
│   │   └── outage_scenarios.json     # 10s, 30s, 60s verified outage scenarios
│   └── splits/
│       ├── smoke_test_manifest.json  # Single-drive smoke test manifest
│       └── drive_split_manifest.json # Complete drive-level split manifest
├── figures/                          # Publication-ready figures (300 DPI)
│   ├── fig1_raw_imu_timeseries.png
│   ├── fig2_cleaned_vs_raw.png
│   ├── fig3_gps_trajectory_local_metres.png
│   ├── fig4_sampling_interval_distributions.png
│   ├── fig5_missing_and_invalid_labels.png
│   ├── fig6_outage_examples.png
│   ├── fig7_input_observation_reference_distinction.png
│   └── fig8_highrate_groundtruth_vs_phone_interpolation.png
├── reports/
│   ├── data_quality_report.md        # Comprehensive Phase 1 data inspection
│   ├── preprocessing_summary.md      # Summary of complete multi-drive pipeline
│   └── assumptions_limitations_leakage_report.md # Scientific analysis
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration loader
│   ├── dataset_generator.py          # Sliding windows & split manifests
│   ├── geodesy.py                    # WGS84 Geodetic to Local ENU projection
│   ├── inspection.py                 # Schema & data quality inspection
│   ├── outage_simulation.py          # 10s/30s/60s outage masking & verification
│   ├── pipeline.py                   # Main executable CLI pipeline
│   ├── signal_processing.py          # Conditioning, filtering, normalization
│   ├── synchronization.py            # High-rate GNSS reference alignment
│   └── visualization.py              # Publication plotting routines
├── tests/
│   ├── test_dataset_generator.py     # Window shapes & manifest tests
│   ├── test_geodesy.py               # WGS84 roundtrip & ENU origin tests
│   ├── test_outage_and_leakage.py    # Zero leakage & outage masking tests
│   ├── test_signal_processing.py     # Scaler train-only fit & clipping tests
│   └── test_synchronization.py       # Gap rejection & interpolation tests
├── pytest.ini
└── README.md
```

---

## 📊 Partition Manifest & Sample Allocations

| Partition | Recording Files | Total Samples | Driving Duration | Sliding Windows $(N, 100, 6)$ |
|---|---|---|---|---|
| **TRAIN** | `S-S1.csv`, `S-S2.csv`, `S-S4.csv` | 240,222 | ~6.67 hrs | **23,994 windows** |
| **VAL** | `S-S3a.csv` | 24,621 | ~41.0 min | **2,453 windows** |
| **TEST** | `S-S3b.csv`, `S-S3c.csv` | 43,996 | ~73.4 min | **4,381 windows** |
| **TOTAL** | All 6 Drives | **308,839** | **~8.58 hrs** | **30,828 windows** |

---

## 🎯 Ground Truth Fidelity: 10 Hz Vehicle GNSS vs 9s Phone Interpolation

The smartphone GNSS chip logged discrete fixes at ~0.11 Hz (~9 seconds). Over a 9-second interval at urban driving speeds ($45\text{ km/h}$), a vehicle traverses $112\text{ meters}$. Piecewise linear interpolation across 9 seconds cuts chords across curves, introducing up to $20\text{ meters}$ of lateral error on $90^\circ$ turns.

To ensure ground-truth evaluation integrity:
- The pipeline pairs each smartphone drive (`S-*.csv`) with its row-synchronized 10 Hz vehicle reference (`V-*.csv`), converting coordinates into Local ENU space for evaluation.
- `figures/fig8_highrate_groundtruth_vs_phone_interpolation.png` plots a zoomed turn illustrating this chord truncation phenomenon.
- **Model inputs are strictly 6 smartphone IMU channels** (`accel_x, accel_y, accel_z, gyro_yaw, gyro_pitch, gyro_roll`). Vehicle signals are strictly barred from model inputs.

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Single-Command Reproduction
Run the end-to-end multi-drive preprocessing pipeline:
```bash
python -m src.pipeline --config configs/default_config.yaml
```

### 3. Run Automated Validation Tests
Execute the full test suite (12 unit tests):
```bash
pytest -v
```
