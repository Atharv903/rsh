import sys
from pathlib import Path
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.corrected_synchronization import extract_aligned_segments_for_drive

def test_imu_stationary_detector(df, drive_name, win_size=20, var_acc_thr=0.08, var_gyro_thr=0.0005, min_dwell=30):
    n = len(df)
    acc_norm = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2).values
    gyro_pitch = df["gyro_pitch"].values
    ref_speed = df["ref_speed_mps"].values
    
    is_stat_detected = np.zeros(n, dtype=bool)
    consec_stat = 0
    
    for i in range(win_size, n):
        a_slice = acc_norm[i-win_size:i]
        g_slice = gyro_pitch[i-win_size:i]
        
        var_a = np.var(a_slice)
        var_g = np.var(g_slice)
        mean_a = np.mean(a_slice)
        
        # Check stationary conditions: low acc variance, acc near 1g, low gyro variance
        if var_a < var_acc_thr and abs(mean_a - 9.81) < 0.4 and var_g < var_gyro_thr:
            consec_stat += 1
        else:
            consec_stat = 0
            
        if consec_stat >= min_dwell:
            is_stat_detected[i] = True
            
    # Evaluate agreement with true stationary periods (ref_speed < 0.1)
    true_stat = ref_speed < 0.1
    
    tp = np.sum(is_stat_detected & true_stat)
    fp = np.sum(is_stat_detected & (~true_stat))
    fn = np.sum((~is_stat_detected) & true_stat)
    tn = np.sum((~is_stat_detected) & (~true_stat))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print(f"=== {drive_name} IMU-Only Stationary Detector ===")
    print(f"  Detected stationary samples: {np.sum(is_stat_detected)} ({np.sum(is_stat_detected)*0.1:.1f} s)")
    print(f"  True stationary samples: {np.sum(true_stat)} ({np.sum(true_stat)*0.1:.1f} s)")
    print(f"  Precision: {precision*100:.1f}%, Recall: {recall*100:.1f}%")
    print(f"  False Positives (detecting stop while driving): {fp} samples ({fp*0.1:.1f} s)")
    
    return is_stat_detected

def main():
    raw_b_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)"
    segs_b, _ = extract_aligned_segments_for_drive("M", raw_dir=raw_b_dir)
    df_b = segs_b[1]

    raw_d_dir = "Synchronised V abd S datasets/Synchronised V abd S datasets/Categorised IOVNB Dataset/Y (Driver D)/Y1"
    segs_d, _ = extract_aligned_segments_for_drive("Y1", raw_dir=raw_d_dir)
    df_d = segs_d[2]
    
    test_imu_stationary_detector(df_b, "Driver B (M_seg1)")
    test_imu_stationary_detector(df_d, "Driver D (Y1_seg3)")

if __name__ == "__main__":
    main()
