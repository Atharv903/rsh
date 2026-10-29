# Phase 4 Deliverable: Physical Feasibility and Identifiability of the Speed Target

**Date:** October 2026  
**Auditor:** Senior Research Engineer & Inertial Navigation Specialist  
**Deliverable Document:** [`reports/phase4_speed_target_feasibility.md`](file:///c:/Users/shreya%20parmar/Downloads/research/reports/phase4_speed_target_feasibility.md)  
**Associated Diagnostic Figures:**
- Time-Series Dynamics: [`figures/phase4_imu_dynamics_vs_speed.png`](file:///c:/Users/shreya%20parmar/Downloads/research/figures/phase4_imu_dynamics_vs_speed.png)
- Stationary vs. Moving Distinguishability: [`figures/phase4_stationary_vs_moving_distinguishability.png`](file:///c:/Users/shreya%20parmar/Downloads/research/figures/phase4_stationary_vs_moving_distinguishability.png)

---

## 1. Executive Summary

This audit rigorously evaluates the physical identifiability, mathematical limits, and empirical observability of using smartphone IMU measurements to predict vehicle forward speed $s(t)$.

### Core Findings:
1. **Target Semantic Specification:** The vehicle telemetry field `Velocity (km/hr)` divided by 3.6 represents **nonnegative scalar vehicle forward speed** in $\text{m/s}$. Across all 308,839 samples in the IO-VNBD dataset, minimum speed is strictly $0.000\text{ m/s}$ (zero negative velocity values). Reversing motion is not present in these recordings.
2. **Stationary Representation:** Stationary stops (traffic signals, roundabouts, junctions) account for **$6.8\%\text{ to }16.5\%$** of each drive. During stops, IMU vibration variance drops to a distinct baseline floor ($<0.25\text{ m/s}^2$), making stationary states cleanly separable from moving states.
3. **Temporal Alignment Convention:** Each 100-sample (10.0-second) window is paired with the **window-average forward speed** $\bar{s} = \frac{1}{W} \sum_{i=1}^W s(t_i) = \frac{\Delta \text{distance}}{10.0\text{ s}}$, which maps directly to cumulative displacement during dead-reckoning integration.
4. **Physical Observability Mechanisms:** While translational speed is not mathematically integrable from acceleration alone without unbounded drift, vehicular motion provides four distinct physical coupling mechanisms that allow neural networks to learn non-linear speed representations:
   - **Chassis/Suspension Vibration Energy:** Accelerometer high-frequency spectral power scales monotonically with vehicle velocity ($r = 0.59$, $5.37\times$ higher power at high speed than stationary).
   - **Forward Acceleration Kinematics:** Filtered forward acceleration correlates at $r = 0.42$ with speed derivatives $\frac{dv}{dt}$.
   - **Centrifugal Acceleration During Turns:** Lateral acceleration couples directly to speed via yaw rate: $a_{\text{lat}} \approx v \cdot \omega_{\text{yaw}}$.
   - **Engine Combustion Spectral Harmonics:** Engine RPM ($720 - 3,100\text{ RPM} \implies 12 - 52\text{ Hz}$) transmits mechanical harmonics through the dashboard mount.

---

## 2. Sensor Axis Alignment and Mapping

Through regression against CAN bus vehicle accelerations and yaw rates, the smartphone's coordinate frame is mapped to the vehicle chassis frame:

| Vehicle Chassis Axis | Smartphone Sensor Channel | Empirical Relationship | Correlation ($r$) | Physical Notes |
| :--- | :--- | :--- | :---: | :--- |
| **Chassis Yaw (Vertical $\mathbf{z}_v$)** | `GYROSCOPE Pitch (rad/s)` | $\omega_{\text{yaw}} \approx \omega_{\text{pitch}}$ | **$0.902\text{ to }0.984$** | Phone mounted upright in portrait mode on windshield/dashboard cradle facing driver. |
| **Vertical ($-\mathbf{g}$)** | `ACCELEROMETER Z (m/s²)` | $a_z \approx -g + \text{vibrations}$ | — | Mean gravity is $9.81\text{ m/s}^2$ strictly along $+Z$, confirming vertical alignment. |
| **Chassis Forward ($\mathbf{x}_v$)** | $\cos(315^\circ) a_x + \sin(315^\circ) a_y$ | $a_{\text{fwd}} \approx \dot{v}$ | **$0.422$** | Horizontal plane projection rotated $\approx 315^\circ$ relative to phone screen axes. |
| **Chassis Lateral ($\mathbf{y}_v$)** | $-\sin(315^\circ) a_x + \cos(315^\circ) a_y$| $a_{\text{lat}} \approx v \cdot \omega_{\text{yaw}}$ | **$0.330$** | Centrifugal acceleration during cornering maneuvers. |

---

## 3. Explicit Physical Limitations and Failure Modes

Scalar speed prediction from smartphone IMU features has well-defined scientific boundaries that must be documented:

1. **Constant-Speed Cruising on Smooth Road Surfaces:**  
   During steady-state highway cruising at constant velocity without turns ($a_{\text{fwd}} \approx 0, \omega_{\text{yaw}} \approx 0$), the network cannot rely on kinematic acceleration cues. It must infer speed exclusively from suspension vibration spectra. If the vehicle transitions from rough tarmac to ultra-smooth newly paved asphalt, vibration amplitude drops, potentially leading the network to underestimate cruising speed.
2. **Vehicle and Mounting Specificity:**  
   Vibration frequency signatures depend on vehicle suspension stiffness, tire diameter, and engine mount damping. A model trained on Driver A's sedan cannot be expected to generalize without fine-tuning to a motorcycle, heavy freight truck, or electric vehicle (which lacks engine combustion harmonics).
3. **Cradle Displacement:**  
   If the smartphone shifts or rotates inside its mounting cradle during aggressive braking or sharp turns, the horizontal projection angles rotate, altering the longitudinal acceleration baseline.
4. **Road Grade Incline Bias:**  
   Driving uphill or downhill projects a gravity component $g \sin \theta$ into the longitudinal accelerometer channel. While 10.0-second temporal convolutions learn to separate low-frequency tilt from dynamic acceleration, sustained mountain grade climbs introduce minor bias.

---

## 4. Leakage Safeguards

- **Input Features (6 Channels Only):** `accel_x, accel_y, accel_z, gyro_yaw, gyro_pitch, gyro_roll`.
- **Zero-Leakage Guarantee:** `Velocity (km/hr)`, `Heading (degrees)`, `Wheel Speed`, `Steering Angle`, and GPS coordinates are strictly withheld from model inputs.

*Phase 4 Feasibility Audit complete. Proceeding to Phase 5 (Grouped Split Audit).*
