# Phase 8 Deliverable: Experimental Protocol and Benchmark Specification

**Date:** October 2026  
**Auditor:** Senior Research Engineer & Inertial Navigation Specialist  
**Deliverable Document:** [`reports/phase8_experiment_protocol.md`](file:///c:/Users/shreya%20parmar/Downloads/research/reports/phase8_experiment_protocol.md)  
**Experiment Root Directory:** `experiments/phase8_corrected_speed_models/`

---

## 1. Frozen Experimental Partitions and Leakage Prohibition

To guarantee that reported performance represents true generalizability across distinct days and road networks, all multi-hour trips in the IO-VNBD dataset are partitioned strictly by session:

| Partition | Session & Calendar Date | Drive Files | Aligned Duration | Causal Windows $(W=100, \text{stride}=10)$ | Speed Target Range (m/s) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **TRAIN** | Session 3 (Sept 8, 2019) | `S-S1.csv`, `S-S2.csv` | 145,533 samples ($\sim 4.04\text{ h}$) | **14,525 windows** | $0.00 - 29.10$ ($7.80 \pm 6.34$) |
| **VAL** | Session 1 (Sept 4, 2019) | `S-S3a.csv`, `S-S3b.csv`, `S-S3c.csv` | 68,495 samples ($\sim 1.90\text{ h}$) | **6,812 windows** | $0.00 - 32.53$ ($10.73 \pm 7.49$) |
| **TEST** | Session 2 (Sept 6, 2019) | `S-S4.csv` | 91,451 samples ($\sim 2.54\text{ h}$) | **9,117 windows** | $0.00 - 30.36$ ($9.68 \pm 7.48$) |

### Strict Protocols:
1. **Grouped Isolation:** All contiguous fragments of Session 1 (`S3a`, `S3b`, `S3c`) remain strictly united in the Validation set. No fragment is moved to Test.
2. **Frozen Test Holdout:** Drive `S4` is frozen. No test results may be used to alter hyperparameters, architectures, early-stopping patience, loss functions, or feature preprocessing.
3. **Normalizer Isolation:** Normalization parameters (`sensor_normalizer_grouped.json`) are fitted strictly on the 145,533 training samples of `S1` and `S2`.

---

## 2. Target and Prediction Timing Convention

- **Primary Supervised Target:** Nonnegative scalar vehicle forward speed $s(t) = v_{\text{fwd}}(t)$ in $\text{m/s}$ at the final timestamp $t$ of the causal 10-second IMU history window.
- **Window Causality:** For an input window spanning indices $[k-W+1 : k+1]$ ($t \in [t - 9.9\text{ s}, t]$), the target is $s(t)$, the instantaneous vehicle forward speed at the window's final sample. No observations later than $t$ are accessible to the network.
- **Model Input Allowlist (6 Channels Only):**
  1. `accel_x` ($\text{m/s}^2$)
  2. `accel_y` ($\text{m/s}^2$)
  3. `accel_z` ($\text{m/s}^2$)
  4. `gyro_yaw` ($\text{rad/s}$)
  5. `gyro_pitch` ($\text{rad/s}$)
  6. `gyro_roll` ($\text{rad/s}$)
- **Strictly Barred from Model Inputs:** Vehicle speed, vehicle heading, wheel speed, CAN accelerations, GPS coordinates, and GPS course over ground.

---

## 3. Matched Training Configuration Across Architectures

All three architectures are trained under identical conditions:
- **Optimization:** AdamW optimizer with initial learning rate $\eta = 10^{-3}$, weight decay $\lambda = 10^{-4}$.
- **Scheduler:** Cosine Annealing learning rate schedule ($\eta_{\min} = 10^{-6}$).
- **Loss Function:** Huber Loss ($\delta = 1.0$) on scalar speed $s(t)$ in $\text{m/s}$.
- **Gradient Clipping:** Maximum norm $1.0$.
- **Validation-Based Early Stopping:** Model checkpoints saved based on minimum validation speed MAE with patience of 7 epochs.
- **Random Seeds:** Three fixed independent seeds ($42, 123, 456$) per architecture.
- **Compute Budget:** 20 epochs per run with batch size 64.

### Architecture Specifications:
1. **CNN-GRU:** 1D Temporal Convolution (kernel 7, channels 64, BatchNorm, ReLU) $\to$ 2-layer Bidirectional GRU (hidden dim 64) $\to$ Dropout (0.2) $\to$ Linear projection to scalar speed $s \in \mathbb{R}^1$.
2. **ModernTCN:** Depthwise separable multi-scale temporal convolutions with inverted bottleneck depthwise-conv blocks and residual connections $\to$ Global average pooling $\to$ Linear projection.
3. **Mamba-2 (Custom PyTorch SSD):** Vectorized State Space Duality implementation in pure PyTorch using causal lower-triangular decay matrix recurrence $\to$ Linear projection. Explicitly documented as a custom SSD formulation.

*Phase 8 Protocol frozen and approved. Proceeding to Phase 3 (Learned Model Sanity Check).*
