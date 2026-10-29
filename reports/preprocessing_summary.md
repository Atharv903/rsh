# Research Preprocessing Pipeline Execution Summary: Complete Multi-Drive Dataset
**Dataset Source:** IO-VNBD Driver A (Smartphone IMU + Paired High-Rate Vehicle Reference)  
**Total Drives Processed:** 6 recordings  
**Total Samples:** 308,839 rows (18884.2 s / 5.25 hours)  
**Total Trustworthy Samples:** 308,839 (100.00%)

## 1. Drive-Level Partitioning & Manifest Summary
To prevent temporal autocorrelation and window-boundary leakage, sliding windows are strictly isolated within complete drive recordings:

| Split Partition | Assigned Drives | Total Samples | Duration | Total Sliding Windows (10s, 1s stride) |
|---|---|---|---|---|
| **TRAIN** | `S-S1.csv`, `S-S2.csv`, `S-S4.csv` | 240,222 | ~6.7 hrs | **23,994 windows** |
| **VAL** | `S-S3a.csv` | 24,621 | ~41.0 min | **2,453 windows** |
| **TEST** | `S-S3b.csv`, `S-S3c.csv` | 43,996 | ~73.4 min | **4,381 windows** |
| **TOTAL** | All 6 Drives | **308,839** | **5.25 hrs** | **30,828 windows** |

## 2. Per-Drive Granular Statistics
| Drive Recording | Assigned Split | Total Samples | Valid Reference Samples | Valid Ratio | Total Windows | Duration (s) |
|---|---|---|---|---|---|---|
| `S-S1.csv` | **TRAIN** | 51,746 | 51,746 | 100.0% | 5,165 | 5174.5s |
| `S-S2.csv` | **TRAIN** | 93,876 | 93,876 | 100.0% | 9,378 | 9201.1s |
| `S-S4.csv` | **TRAIN** | 94,600 | 94,600 | 100.0% | 9,451 | 354.8s |
| `S-S3a.csv` | **VAL** | 24,621 | 24,621 | 100.0% | 2,453 | 2462.0s |
| `S-S3b.csv` | **TEST** | 6,813 | 6,813 | 100.0% | 672 | -2026.4s |
| `S-S3c.csv` | **TEST** | 37,183 | 37,183 | 100.0% | 3,709 | 3718.2s |

## 3. Ground-Truth Trajectory: High-Rate Vehicle GNSS vs 9s Phone Interpolation
- **Dual Reference Capability:** The pipeline integrates high-rate (10 Hz) continuous vehicle GNSS from `V-*.csv` as the gold-standard ground truth for trajectory evaluation during simulated outages.
- **Observation on 9-Second Phone GNSS:** When internal phone GPS (~0.11 Hz) is linearly interpolated across 9-second gaps, chord truncation error occurs during curves and turns. Evaluating dead-reckoning models against true 10 Hz vehicle references provides sub-meter ground truth without corner-cutting artifacts.
- **Zero Leakage Invariant:** All vehicle signals (CAN bus, wheel speeds, steering) are strictly barred from model inputs. Only the 6 smartphone IMU channels (`accel_x, accel_y, accel_z, gyro_yaw, gyro_pitch, gyro_roll`) are used as model inputs.

## 4. Simulated Outage Evaluation Scenarios
- **Verified Outages Generated:** 15 scenarios across 10s, 30s, and 60s durations.
| Scenario ID | Duration | Time Interval (s) | Distance Traveled | Mean Speed | Reference Coverage |
|---|---|---|---|---|---|
| `outage_10s_01` | 10s | [78.0s, 87.9s] | 17.5 m | 1.80 m/s | 100% |
| `outage_10s_02` | 10s | [332.0s, 341.9s] | 153.6 m | 15.33 m/s | 100% |
| `outage_10s_03` | 10s | [1175.0s, 1184.9s] | 63.9 m | 6.25 m/s | 100% |
| `outage_10s_04` | 10s | [1246.0s, 1255.9s] | 41.6 m | 4.19 m/s | 100% |
| `outage_10s_05` | 10s | [2928.0s, 2937.9s] | 117.9 m | 11.67 m/s | 100% |
| `outage_30s_01` | 30s | [611.0s, 640.9s] | 351.2 m | 11.48 m/s | 100% |
| `outage_30s_02` | 30s | [2187.0s, 2216.9s] | 265.3 m | 8.69 m/s | 100% |
| `outage_30s_03` | 30s | [2799.0s, 2828.9s] | 425.8 m | 13.87 m/s | 100% |
| `outage_30s_04` | 30s | [3351.0s, 3380.9s] | 139.7 m | 4.52 m/s | 100% |
| `outage_30s_05` | 30s | [3467.0s, 3496.9s] | 218.2 m | 7.09 m/s | 100% |
| `outage_60s_01` | 60s | [755.0s, 814.9s] | 535.1 m | 8.71 m/s | 100% |
| `outage_60s_02` | 60s | [2768.0s, 2827.9s] | 822.2 m | 13.38 m/s | 100% |
| `outage_60s_03` | 60s | [3258.0s, 3317.9s] | 385.6 m | 6.33 m/s | 100% |
| `outage_60s_04` | 60s | [4115.0s, 4174.9s] | 558.9 m | 9.19 m/s | 100% |
| `outage_60s_05` | 60s | [4486.0s, 4545.9s] | 484.4 m | 7.95 m/s | 100% |