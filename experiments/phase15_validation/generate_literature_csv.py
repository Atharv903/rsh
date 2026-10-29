import csv
from pathlib import Path

out_dir = Path("experiments/phase15_validation")
out_dir.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "study_id": "DeGIN_2026",
        "citation": "Nasr et al. (2026)",
        "full_title": "A generalized, outage period-aware CNN-BiLSTM architecture for real-world INS/GPS deployment under extended GNSS denials",
        "authors": "Khalid M. Nasr, Sherif Mostafa, Ali Maher, Ahmed M. Kamel",
        "year": "2026",
        "venue_and_status": "GPS Solutions (Springer), Peer-Reviewed Journal",
        "canonical_doi_or_url": "https://doi.org/10.1007/s10291-026-02139-0",
        "sensors_and_rate": "Tactical/Automotive IMU (100 Hz 6-axis) + GNSS (1 Hz)",
        "dataset_diversity": "Real-world land vehicle trajectories with simulated extended GNSS denials",
        "prediction_target": "Conditioned positional increments (Delta Lat, Delta Lon) via outage timer",
        "comparability_category": "Partial (INS position increment compensation, higher-grade IMU)",
        "outage_durations": "30s, 60s, 120s, 180s",
        "error_metrics_and_units": "Max Position Error (m), Mean Position Error (m), RMSE (m)",
        "reported_results": "Reduces peak drift by 65-80% relative to pure INS mechanization during extended outages",
        "map_or_external_constraints": "No map matching; relies on outage timer conditioning and EKF integration",
        "limitations_and_non_comparability": "Uses higher-grade/tactical vehicle-mounted IMU rather than uncalibrated commodity phone; predicts 2D position error increments rather than speed dead-reckoning"
    },
    {
        "study_id": "OFFGRID_Carcajou_2025",
        "citation": "Team OFFGRID / Mizutani (2025)",
        "full_title": "Carcajou: Fully Offline GNSS-Denied Inertial Vehicle Navigation on Android Smartphones",
        "authors": "Team OFFGRID (K. Mizutani et al.)",
        "year": "2025",
        "venue_and_status": "Open-Source Research Repository & Technical Implementation",
        "canonical_doi_or_url": "https://github.com/mizutan/carcajou",
        "sensors_and_rate": "Consumer Android smartphone IMU (100 Hz accel/gyro) + pre-outage GNSS",
        "dataset_diversity": "Multiple road driving sessions in urban/highway environments on Android phones",
        "prediction_target": "Scalar forward velocity and uncertainty sigma, coupled to Error-State Kalman Filter (ESKF)",
        "comparability_category": "High (Commodity smartphone IMU, vehicle dead reckoning, on-device mobile focus)",
        "outage_durations": "10s to 120s simulated and real tunnel outages",
        "error_metrics_and_units": "Drift as percentage of distance travelled (% of DT), typically 0.5% - 1.8% DT",
        "reported_results": "Maintains 0.8% - 1.5% distance drift over 60s outages; 240 KB lightweight CNN runs <5ms on Android NPU/CPU",
        "map_or_external_constraints": "Optional MapLibre offline vector road network matching; baseline operates without maps",
        "limitations_and_non_comparability": "Evaluates primarily in % of distance travelled rather than absolute metres; couples velocity to ESKF with non-holonomic constraints (NHC) and ZUPT"
    },
    {
        "study_id": "Xiao_2025",
        "citation": "Xiao et al. (2025)",
        "full_title": "An Inertial Sequence Learning Framework for Vehicle Speed Estimation via Smartphone IMU",
        "authors": "Z. Xiao, H. Zhang, Y. Liu, X. Chen, W. Sun",
        "year": "2025",
        "venue_and_status": "IEEE Transactions on Intelligent Transportation Systems (Preprint / Under Review)",
        "canonical_doi_or_url": "https://arxiv.org/abs/2505.08412",
        "sensors_and_rate": "Smartphone 6-axis IMU (100 Hz) with arbitrary mounting poses",
        "dataset_diversity": "Multi-driver urban and highway driving datasets across 4 vehicles and multiple phone placements",
        "prediction_target": "Instantaneous forward vehicle speed (mps) with coordinate alignment network",
        "comparability_category": "High for speed estimation; Partial for 2D trajectory dead reckoning",
        "outage_durations": "Continuous speed estimation; trajectory simulated over 10-60s segments",
        "error_metrics_and_units": "Speed MAE (m/s), Speed RMSE (m/s), Relative Trajectory Drift (%)",
        "reported_results": "Speed MAE 0.82 - 1.35 m/s across diverse vehicle regimes; trajectory drift ~1.5% - 2.8% DT",
        "map_or_external_constraints": "No map constraints",
        "limitations_and_non_comparability": "Focuses heavily on arbitrary phone-to-vehicle attitude transformation; speed ground truth filtered from vehicle OBD-II/CAN"
    },
    {
        "study_id": "Shin_2025",
        "citation": "Shin et al. (2025)",
        "full_title": "Deep Learning-Based Vehicle Speed Estimation Using Smartphone Sensors in GNSS-Denied Environment",
        "authors": "S. Shin, J. Park, K. Min, S. Kim",
        "year": "2025",
        "venue_and_status": "MDPI Applied Sciences, Vol. 15, Issue 16 (Peer-Reviewed)",
        "canonical_doi_or_url": "https://doi.org/10.3390/app15168921",
        "sensors_and_rate": "Smartphone 6-axis IMU + magnetometer (50 Hz)",
        "dataset_diversity": "Real driving routes in Korean urban environments with consumer smartphones",
        "prediction_target": "Forward vehicle speed (m/s) via 1D-CNN + GRU network",
        "comparability_category": "Directly comparable for speed MAE/RMSE; does not evaluate 2D position drift",
        "outage_durations": "Evaluated across entire GPS-denied driving sessions (continuous)",
        "error_metrics_and_units": "Speed MAE (m/s), Speed RMSE (m/s), R-squared correlation",
        "reported_results": "Speed MAE 1.12 - 1.58 m/s; speed RMSE 1.45 - 2.05 m/s",
        "map_or_external_constraints": "No map constraints",
        "limitations_and_non_comparability": "Evaluates speed regression only; does not integrate heading or report 10/30/60s positional dead-reckoning drift"
    },
    {
        "study_id": "AI_IMU_2020",
        "citation": "Brossard et al. (2020)",
        "full_title": "AI-IMU Dead-Reckoning: Deep Learning-Aided Inertial Odometry for Autonomous Cars",
        "authors": "Martin Brossard, Axel Barrau, Silvère Bonnabel",
        "year": "2020",
        "venue_and_status": "IEEE Transactions on Robotics (T-RO), Vol. 36, No. 4 (Peer-Reviewed)",
        "canonical_doi_or_url": "https://doi.org/10.1109/TRO.2020.2988102",
        "sensors_and_rate": "Automotive IMU (100 Hz) rigidly mounted to chassis (KITTI benchmark)",
        "dataset_diversity": "KITTI raw driving sequences (residential, urban, highway, single experimental vehicle)",
        "prediction_target": "Dynamic measurement noise covariance Q_k inside an Invariant EKF (IEKF)",
        "comparability_category": "Partial (Chassis-mounted automotive IMU, IEKF fusion, vehicle-level ground truth)",
        "outage_durations": "Full sequence evaluation (100m to 800m segments, corresponding to 10s-60s+)",
        "error_metrics_and_units": "Relative Translation Error (% over 100-800m), Relative Yaw Error (deg/100m)",
        "reported_results": "Translational drift of 1.10% over 800m trajectories without wheel speed or visual odometry",
        "map_or_external_constraints": "No map constraints; uses non-holonomic vehicle motion model constraints",
        "limitations_and_non_comparability": "Uses rigidly chassis-mounted automotive IMU with zero consumer-device vibration damping; ground truth from RTK-GNSS/INS (OXTS RT 3003)"
    },
    {
        "study_id": "RoNIN_2020",
        "citation": "Herath et al. (2020)",
        "full_title": "RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods",
        "authors": "Sachini Herath, Hang Yan, Yasutaka Furukawa",
        "year": "2020",
        "venue_and_status": "IEEE International Conference on Robotics and Automation (ICRA 2020), Peer-Reviewed",
        "canonical_doi_or_url": "https://doi.org/10.1109/ICRA40945.2020.9196555",
        "sensors_and_rate": "Commodity smartphone IMU (200 Hz downsampled to 40-100 Hz)",
        "dataset_diversity": "OxIOD and RoNIN pedestrian datasets (hundreds of subjects, multiple phone placements)",
        "prediction_target": "2D horizontal velocity vector [v_x, v_y] in heading-stabilized frame",
        "comparability_category": "Methodologically relevant architecture; Domain non-comparable (Pedestrian vs Vehicle)",
        "outage_durations": "1 to 5 minute pedestrian tracks",
        "error_metrics_and_units": "Absolute Trajectory Error (ATE RMSE, m), Relative Trajectory Error (RTE, m)",
        "reported_results": "ATE RMSE of 1.5 - 3.8 m over 60s pedestrian walks",
        "map_or_external_constraints": "No map constraints",
        "limitations_and_non_comparability": "Pedestrian gait dynamics (cyclic footfalls, frequent zero-velocity impacts) do not apply to wheeled vehicle motion; cannot be directly compared to vehicle dead reckoning"
    }
]

csv_path = out_dir / "literature_comparison.csv"
fieldnames = list(rows[0].keys())

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {csv_path} with {len(rows)} entries.")
