# Data Quality & Sensor Inspection Report
**Target Dataset File:** `data/raw\S-S1.csv`  
**Total Samples:** 51,746 rows | **Total Columns:** 24  
**Total Duration:** 5174.50 s (86.24 min)  
**File Size:** 9.19 MB

## 1. Schema & Column Categorization

| Column Name | Assigned Role | Standard Identifier | Unit | Description |
|---|---|---|---|---|
| `GPS LATITUDE (degrees)` | **reference_label** | `gps_lat` | degrees | GNSS coordinate |
| `GPS LONGITUDE (degrees)` | **reference_label** | `gps_lon` | degrees | GNSS coordinate |
| `GPS ALTITUDE (m)` | **reference_label** | `gps_alt` | meters | GNSS altitude |
| `GPS SPEED (Kmh)` | **reference_label** | `gps_speed_kmh` | km/h | GNSS speed |
| `GPS ACCURACY (m)` | **reference_quality** | `gps_accuracy_m` | meters | GNSS horizontal accuracy |
| `GPS ORIENTATION (°)` | **reference_label** | `gps_heading_deg` | degrees | GNSS course over ground |
| `GPS SATELLITES IN RANGE` | **reference_quality** | `gps_satellites` | count | GNSS satellite count |
| `TIME SINCE START (ms)` | **timestamp** | `time_ms` | ms | Monotonic millisecond clock |
| `DATE (YYYY-MO-DD HH-MI-SS_SSS)` | **metadata** | `date_str` | ISO text | Wall-clock datetime |
| `ACCELEROMETER X (m/s²)` | **input** | `accel_x` | m/s² | IMU accelerometer |
| `ACCELEROMETER Y (m/s²)` | **input** | `accel_y` | m/s² | IMU accelerometer |
| `ACCELEROMETER Z (m/s²)` | **input** | `accel_z` | m/s² | IMU accelerometer |
| `GRAVITY X (m/s²)` | **excluded_auxiliary** | `gravity_x` | m/s² | Device fused gravity vector |
| `GRAVITY Y (m/s²)` | **excluded_auxiliary** | `gravity_y` | m/s² | Device fused gravity vector |
| `GRAVITY Z (m/s²)` | **excluded_auxiliary** | `gravity_z` | m/s² | Device fused gravity vector |
| `GYROSCOPE Yaw (rad/s)` | **input** | `gyro_yaw` | rad/s | IMU gyroscope |
| `GYROSCOPE Pitch (rad/s)` | **input** | `gyro_pitch` | rad/s | IMU gyroscope |
| `GYROSCOPE Roll (rad/s)` | **input** | `gyro_roll` | rad/s | IMU gyroscope |
| `MAGNETIC FIELD X (μT)` | **excluded_auxiliary** | `mag_x` | μT | Magnetometer |
| `MAGNETIC FIELD Y (μT)` | **excluded_auxiliary** | `mag_y` | μT | Magnetometer |
| `MAGNETIC FIELD Z (μT)` | **excluded_auxiliary** | `mag_z` | μT | Magnetometer |
| `ORIENTATION (Yaw) (°)` | **excluded_auxiliary** | `orient_yaw` | degrees | Device fused orientation |
| `ORIENTATION (Pitch) (°)` | **excluded_auxiliary** | `orient_pitch` | degrees | Device fused orientation |
| `ORIENTATION (Roll ) (°)` | **excluded_auxiliary** | `orient_roll` | degrees | Device fused orientation |

## 2. Temporal & Sampling Analysis

- **Monotonic Increasing:** Yes (Strict)
- **Duplicate Timestamps:** 0
- **Nominal IMU Sampling Rate:** 10.00 Hz (Mean dt: 100.00 ms, std: 0.7646 ms)
- **IMU Sampling Interval Percentiles (ms):**
  - 1%: 99.0 ms | Median (50%): 100.0 ms | 99%: 101.0 ms | Max: 111.0 ms
- **IMU Timestamp Gaps (> 200 ms):** 0

## 3. GNSS / GPS Characteristics & Ground Truth Reference

- **Geographic Bounds:** Latitude [52.398403°, 52.419914°], Longitude [-1.603373°, -1.505290°]
- **Total Discrete GPS Fix Events:** 532
- **Effective GPS Update Interval:** Median: 9.00 s | Mean: 9.73 s | Max: 70.00 s
- **GPS Horizontal Accuracy:** Mean: 2.88 m | Median: 3.00 m | Max: 32.00 m
- **GPS Speed:** Max: 18.82 km/h (5.23 m/s) | Mean: 7.43 km/h
- **Invalid Coordinate Fixes (e.g. 0,0 or out-of-range):** 0

## 4. IMU Sensor Inputs (Initial 6 Channels)

| Sensor Channel | Mean | Std | Min | Median (50%) | Max | Unit |
|---|---|---|---|---|---|---|
| `ACCELEROMETER X (m/s²)` | 0.0423 | 1.0602 | -8.3689 | 0.0598 | 7.5736 | m/s² |
| `ACCELEROMETER Y (m/s²)` | 0.0622 | 1.0664 | -8.2137 | 0.0293 | 11.3521 | m/s² |
| `ACCELEROMETER Z (m/s²)` | 9.8474 | 0.5132 | 2.4242 | 9.8514 | 16.1533 | m/s² |
| `GYROSCOPE Yaw (rad/s)` | 0.0022 | 0.0999 | -1.1627 | 0.0017 | 1.3556 | rad/s |
| `GYROSCOPE Pitch (rad/s)` | -0.0072 | 0.1417 | -0.7799 | -0.0050 | 0.6330 | rad/s |
| `GYROSCOPE Roll (rad/s)` | 0.0025 | 0.0515 | -0.5837 | 0.0023 | 0.5546 | rad/s |

## 5. Leakage Prevention Protocol (Mandatory Rules)

1. **Strict Input Channel Isolation:** Only 6 channels (`accel_x`, `accel_y`, `accel_z`, `gyro_yaw`, `gyro_pitch`, `gyro_roll`) enter the model feature space.
2. **Zero GNSS Leakage:** GPS latitude, longitude, speed, altitude, accuracy, and heading are strictly isolated as evaluation-only reference labels during simulated outages.
3. **No External CAN/OBD Signals:** External vehicle signals (wheel speeds, steering angle, OBD throttle) are completely excluded.
4. **Coordinate Frame Distinction:** Smartphone body frame $(x_b, y_b, z_b)$ is distinct from vehicle frame $(x_v, y_v, z_v)$. The pipeline maintains explicit transformation documentation.

## 6. Data Quality Warnings & Flags

- ⚠️ **WARNING:** GPS accuracy degraded up to 32.0 meters.