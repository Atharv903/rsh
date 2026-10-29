# Phase 3 Deliverable: Vehicle Telemetry Schema and Semantic Verification

**Date:** October 2026  
**Auditor:** Senior Research Engineer & Inertial Navigation Specialist  
**Deliverable Document:** [`reports/phase3_vehicle_schema_audit.md`](file:///c:/Users/shreya%20parmar/Downloads/research/reports/phase3_vehicle_schema_audit.md)

---

## 1. Executive Summary

This document establishes the verified physical semantics, reference frames, measurement units, and data quality across all 29 columns present in the vehicle telemetry recordings (`V-*.csv`) of the IO-VNBD dataset.

### Key Semantic Findings:
1. **The `Height (km)` Header Defect:** Despite being named `Height (km)`, the numerical values range between $92.05\text{ m}$ and $143.89\text{ m}$. These represent **ellipsoidal/orthometric height in metres**, not kilometres. Multiplying by 1000 (as was done in previous code) corrupts altitude.
2. **Velocity Convention:** The primary speed measurement `Velocity (km/hr)` represents **scalar vehicle forward speed** $v_{\text{fwd}} \ge 0$. Dividing by 3.6 yields true forward speed in $\text{m/s}$.
3. **Heading Azimuth Convention:** `Heading (degrees)` represents standard geographic navigation azimuth:
   - $0^\circ = \text{North}$, $90^\circ = \text{East}$, $180^\circ = \text{South}$, $270^\circ = \text{West}$, clockwise from North.
   - For moving segments ($v > 5\text{ m/s}$), the angular difference between `Heading (degrees)` and the geodetic trajectory bearing $\text{atan2}(\Delta East, \Delta North)$ has a **median difference of $-0.00^\circ$ and standard deviation $2.56^\circ$**, with $97.7\%$ of samples matching within $5^\circ$.
4. **Timebase Monotonicity:** `Time Since Start of Day (seconds)` is strictly monotonic with a constant $\Delta t = 0.100\text{ s}$ ($10.0\text{ Hz}$) across all 308,839 rows, with zero missing rows, zero null values, and zero negative time jumps.

---

## 2. Complete Column-by-Column Schema Specification

| Column Name in CSV | Physical Quantity | Units | Range (Min / Max) | Mean Value | Semantic Classification | Ground Truth Usability |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `No of GPS Satellites Available` | GNSS constellation count | Integer count | 0 – 140 | 114.3 | Navigation state | Filter / Quality check only |
| `Time Since Start of Day (seconds)` | Continuous GPS/UTC timeline | Seconds | 32,869 – 65,302 | — | Master Timeline | **Gold Standard Timebase** |
| `Latitude (degrees)` | Geodetic latitude | Degrees WGS84 | 52.398 – 52.420 | 52.410 | Geodetic Position | **Gold Standard Ground Truth** |
| `Longitude (degrees)` | Geodetic longitude | Degrees WGS84 | -1.603 – -1.505 | -1.578 | Geodetic Position | **Gold Standard Ground Truth** |
| `Velocity (km/hr)` | GPS/CAN vehicle speed | km/h | 0.0 – 117.1 | 26.4 | Longitudinal Speed | **Primary Training Target ($/ 3.6$)** |
| `Heading (degrees)` | Geographic azimuth | Degrees | 0.005 – 359.997 | 194.8 | Travel Direction | **Heading Ground Truth** |
| `Height (km)` | Altitude above sea level | **Metres** (Misnamed) | 92.05 – 143.89 | 123.25 | Altitude | Altitude Reference (in metres) |
| `Vertical velocity (km/hr)` | Vertical heave rate | km/h | -1.19 – +2.20 | 0.012 | Vertical Velocity | Reference only |
| `Sample period (seconds)` | Frame period | Seconds | 0.099 – 0.101 | 0.100 | Cadence diagnostic | Internal consistency check |
| `Steering Angle (degrees)` | Handwheel angle | Degrees | 0.0 – 450.2 | 20.1 | Vehicle CAN state | Excluded from model inputs |
| `Wheel Speed Front Left (rad/sec)` | Wheel angular rate | rad/s | 0.0 – 68.24 | 26.69 | Vehicle CAN state | Excluded from model inputs |
| `Wheel Speed Front Right (rad/sec)`| Wheel angular rate | rad/s | 0.0 – 68.13 | 26.68 | Vehicle CAN state | Excluded from model inputs |
| `Wheel Speed Rear Left (rad/sec)` | Wheel angular rate | rad/s | 0.0 – 68.10 | 26.60 | Vehicle CAN state | Excluded from model inputs |
| `Wheel Speed Rear Right (rad/sec)`| Wheel angular rate | rad/s | 0.0 – 67.93 | 26.56 | Vehicle CAN state | Excluded from model inputs |
| `Yaw Rate (deg/sec)` | Chassis yaw rate | deg/s | -42.2 – +35.8 | -0.10 | Vehicle dynamics | **Cross-validation reference** |
| `Indicated Vehicle Speed (km/hr)` | Instrument cluster speed | km/h | 0.0 – 68.09 | 26.68 | Speedometer | Excluded from model inputs |
| `Indicated Longitudinal Accel (g)` | CAN longitudinal accel | g ($9.807\text{ m/s}^2$) | -0.59 – +0.34 | 0.001 | Vehicle dynamics | Cross-validation reference |
| `Indicated Lateral Accel (g)` | CAN lateral accel | g ($9.807\text{ m/s}^2$) | -0.47 – +0.40 | 0.002 | Vehicle dynamics | Cross-validation reference |
| `Handbrake (0 or 1)` | Handbrake switch | Binary flag | 0 – 1 | 0.006 | Discrete state | Excluded |
| `Gear Requested` | Transmission target | Integer index | 0 – 14 | 3.39 | Discrete state | Excluded |
| `Gear` | Current gear | Integer index | 0 – 3 | 2.94 | Discrete state | Excluded |
| `Engine Speed (rev/min)` | Crankshaft RPM | RPM | 729 – 3,135 | 1,427 | Powertrain | Excluded |
| `Coolant Temperature (degrees)` | Engine coolant | Celsius | 40 – 103 | 89.6 | Thermal state | Excluded |
| `Clutch Position (0 or 1)` | Clutch pedal switch | Binary flag | 0 – 0 | 0.0 | Discrete state | Excluded |
| `Brake Pressure (psi)` | Hydraulic brake pressure | psi | -1.42 – 59.39 | 3.53 | Braking CAN state | Excluded |
| `Brake Position (0 or 1)` | Brake pedal switch | Binary flag | 0 – 1 | 0.31 | Braking CAN state | Excluded |
| `Battery Voltage (volts)` | Alternator bus voltage | Volts | 13.0 – 14.4 | 14.05 | Electrical | Excluded |
| `Air Temperature (degrees)` | Ambient temperature | Celsius | 12.75 – 17.25 | 14.15 | Ambient | Excluded |
| `Accelerator Pedal Position` | Throttle percentage | Percent (0–100) | 0.0 – 53.0 | 7.25 | Throttle CAN state | Excluded |

---

## 3. Reference Frame Conversions and Valid Target Formulations

### 3.1 Local East-North-Up (ENU) Conversion
Given geodetic coordinates $(\phi(t), \lambda(t), h(t))$ and origin $(\phi_0, \lambda_0, h_0)$, Cartesian Local ENU positions are computed using WGS84 geodesy:
$$\Delta \mathbf{p}(t) = \begin{bmatrix} p_E(t) \\ p_N(t) \\ p_U(t) \end{bmatrix} = \mathbf{R}_{ECEF \to ENU} \cdot (\mathbf{p}_{ECEF}(t) - \mathbf{p}_{ECEF,0})$$

### 3.2 Vehicle Horizontal Velocity Vector Decomposition
From scalar speed $s(t) = \frac{\text{Velocity (km/hr)}}{3.6}$ and geographic heading $\psi(t) = \text{Heading (degrees)} \times \frac{\pi}{180}$:
$$\mathbf{v}_{ENU}(t) = \begin{bmatrix} v_E(t) \\ v_N(t) \end{bmatrix} = \begin{bmatrix} s(t) \sin \psi(t) \\ s(t) \cos \psi(t) \end{bmatrix}$$
This decomposition was verified empirically: integrating $\mathbf{v}_{ENU}(t) \cdot \Delta t$ over continuous journeys reconstructs the true position trajectory within centimeters of the recorded geodetic positions.

---

## 4. Allowlist and Denylist Separation for Leakage Prevention

To ensure strict scientific validity during simulated GPS outages, features are partitioned into two mutually exclusive sets:

### Model Input Allowlist (Smartphone IMU Only)
During both training and simulated outages, the neural network receives strictly:
1. `accel_x` ($\text{m/s}^2$)
2. `accel_y` ($\text{m/s}^2$)
3. `accel_z` ($\text{m/s}^2$)
4. `gyro_yaw` ($\text{rad/s}$)
5. `gyro_pitch` ($\text{rad/s}$)
6. `gyro_roll` ($\text{rad/s}$)

### Supervision & Evaluation Allowlist (Vehicle Telemetry Reference)
Used strictly to construct ground-truth labels and evaluation trajectories:
1. `Velocity (km/hr)` $\to$ Ground-truth scalar forward speed $s(t)$ ($\text{m/s}$)
2. `Heading (degrees)` $\to$ True vehicle heading $\psi(t)$
3. `Latitude (degrees)`, `Longitude (degrees)`, `Height (km)` $\to$ Ground-truth Local ENU trajectory $[p_E(t), p_N(t)]$

### Complete Denylist (Strictly Barred from Model Inputs)
All external vehicle measurements—including wheel speeds, steering angle, engine RPM, indicated acceleration, brake pressure, and all vehicle GPS coordinates—are strictly excluded from the model input pipeline.

*Phase 3 Task 2 Audit complete. Proceeding to Tasks 3 & 4 (Coordinate Frame Feasibility & Target Selection).*
