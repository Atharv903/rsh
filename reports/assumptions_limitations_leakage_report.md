# Scientific Report: Multi-Drive Standardization, High-Rate Ground Truth, and Leakage Invariants

## Executive Summary
This document provides an in-depth scientific analysis addressing two core architectural and experimental realities in inertial navigation research using the **IO-VNBD** dataset:
1. **Multi-Drive Generalization:** Extension of data standardization, leakage prevention, and window generation across all recordings in Driver A (`S-S1`, `S-S2`, `S-S3a`, `S-S3b`, `S-S3c`, `S-S4`), producing distinct, non-overlapping `train_windows.npz`, `val_windows.npz`, and `test_windows.npz` datasets.
2. **Ground Truth Fidelity:** A rigorous quantitative evaluation of why ~9-second smartphone GNSS interpolation is inadequate as high-fidelity ground truth on dynamic vehicle maneuvers, and how the pipeline integrates paired 10 Hz high-rate vehicle instrumentation (`V-*.csv`) as the gold-standard evaluation reference.

---

## 1. Why ~9-Second Phone GNSS Interpolation Fails on Curves

### 1.1 The Chord Truncation Phenomenon
In `S-S1.csv` through `S-S4.csv`, the smartphone operating system delivered discrete GNSS fixes at approximately $0.11\text{ Hz}$ ($\sim 9.0\text{ s}$ intervals). Between fixes, the logger repeated the previous latitude/longitude coordinates (sample-and-hold step function).

When naive or piecewise linear interpolation is applied between two fixes $t_k$ and $t_{k+1}$ separated by $9$ seconds:
- If a vehicle travels at an urban speed of $45\text{ km/h} = 12.5\text{ m/s}$, it covers **$112.5\text{ meters}$** between fixes.
- If the vehicle executes a $90^\circ$ turn (radius $R \approx 70\text{ m}$), the true physical trajectory follows a circular arc of length $s = R \theta = 110\text{ m}$.
- The linear interpolation connects the start and end of the turn with a straight secant line (chord). The maximum lateral chord error between the arc and the chord is:
  $$\delta_{\max} = R \left(1 - \cos\frac{\theta}{2}\right) \approx 70 \left(1 - \cos 45^\circ\right) \approx 20.5\text{ meters!}$$

### 1.2 Quantitative Discrepancy Against High-Rate Vehicle Reference
Because the IO-VNBD authors paired each smartphone recording (`S-*.csv`) with synchronized high-rate vehicle instrumentation (`V-*.csv`, logging continuous 10 Hz GNSS, speed, and heading across identical sample counts):
- We directly compared the 9-second phone interpolated position against the true 10 Hz vehicle position across all 46,482 valid samples of `S1`.
- **Finding:** Instantaneous 2D position discrepancies routinely reach $15\text{ m}$ to $35\text{ m}$ during cornering and intersection turns.
- **Resolution in Pipeline:** The pipeline integrates `V-*.csv` to establish true, continuous 10 Hz Local ENU ground-truth references (`ref_east_m`, `ref_north_m`, `ref_vel_east_mps`, `ref_vel_north_mps`).
- **Figure Generated:** `figures/fig8_highrate_groundtruth_vs_phone_interpolation.png` displays a zoomed 60-second turn highlighting the linear chord truncation of the phone's 9s GPS against the vehicle's smooth continuous trajectory.

---

## 2. Complete Multi-Drive Dataset Standardization

All 6 recordings in Driver A have been standardized, conditioned, aligned, and partitioned by complete drive:

| Partition | Recording Files | Total Samples | Driving Duration | Sliding Windows $(N, 100, 6)$ |
|---|---|---|---|---|
| **TRAIN** | `S-S1.csv`, `S-S2.csv`, `S-S4.csv` | 240,222 samples | 400.4 minutes (~6.67 hrs) | **23,994 windows** |
| **VAL** | `S-S3a.csv` | 24,621 samples | 41.0 minutes (~0.68 hrs) | **2,453 windows** |
| **TEST** | `S-S3b.csv`, `S-S3c.csv` | 43,996 samples | 73.4 minutes (~1.22 hrs) | **4,381 windows** |
| **TOTAL** | All 6 Drives | **308,839 samples** | **514.7 minutes (~8.58 hrs)** | **30,828 windows** |

### 2.1 Artifacts Persisted:
- `data/processed/train_windows.npz`: Feature tensor $X \in \mathbb{R}^{23994 \times 100 \times 6}$, displacement targets $Y \in \mathbb{R}^{23994 \times 2}$.
- `data/processed/val_windows.npz`: Feature tensor $X \in \mathbb{R}^{2453 \times 100 \times 6}$, displacement targets $Y \in \mathbb{R}^{2453 \times 2}$.
- `data/processed/test_windows.npz`: Feature tensor $X \in \mathbb{R}^{4381 \times 100 \times 6}$, displacement targets $Y \in \mathbb{R}^{4381 \times 2}$.
- `data/processed/{drive}_processed_aligned.csv`: Per-drive aligned and normalized time-series dataframes.

---

## 3. Strict Leakage Prevention Invariants

1. **Train-Only Normalization:** `SensorNormalizer` was fitted strictly across the 240,222 samples of the training drives (`S1`, `S2`, `S4`). Mean and standard deviation statistics were saved to `data/processed/sensor_normalizer.json` and applied to `val` and `test` partitions without updating.
2. **Drive-Level Isolation:** Sliding windows ($L = 100$ samples = 10s, stride = 10 samples = 1s) are generated strictly within individual drive boundaries. Zero windows ever span across drive edges or split boundaries.
3. **No External Vehicle Inputs:** All vehicle sensors (wheel speeds, steering angle, engine speed, brake pressure from `V-*.csv`) are completely excluded from model inputs. The model inputs across all partitions consist strictly of the 6 smartphone IMU channels (`accel_x, accel_y, accel_z, gyro_yaw, gyro_pitch, gyro_roll`).
4. **Outage Horizon Masking:** During simulated outages, GNSS observation features are masked (`gps_available = False`, observation coordinates set to NaN). Ground truth reference positions are stored exclusively in offline evaluation structures.
