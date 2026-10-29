import pandas as pd

df = pd.read_csv("experiments/phase14_validation/summary_metrics.csv")
pooled = df[df["session"] == "POOLED"]

for proto in ["Protocol_A_ZeroBias", "Protocol_B_PriorCalib", "Protocol_C_CausalOnline"]:
    print(f"\n=======================================================")
    print(f"POOLED SUMMARY FOR: {proto}")
    print(f"=======================================================")
    sub = pooled[pooled["calibration_protocol"] == proto]
    for dur in [10, 30, 60]:
        print(f"\n--- {dur}s Outage ---")
        dur_sub = sub[sub["duration_s"] == dur]
        for idx, r in dur_sub.iterrows():
            print(f"  {r['estimator']:<25} | Drift: {r['mean_final_drift_m']:6.2f}m (med {r['median_final_drift_m']:6.2f}m) | ATE: {r['mean_ate_rmse_m']:5.2f}m | Spd MAE: {r['mean_speed_mae_mps']:.2f} m/s | Head Err: {r['mean_heading_err_deg']:.2f}°")
