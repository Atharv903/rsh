import pandas as pd

df = pd.read_csv("experiments/phase16_validation/summary_metrics.csv")
sub = df[(df["session"] == "POOLED") & (df["calibration_protocol"] == "Protocol_C_CausalOnline")]
cols = ["duration_s", "config_id", "mean_final_drift_m", "median_final_drift_m", "mean_along_track_err_m", "mean_cross_track_err_m", "mean_speed_mae_mps", "mean_drift_pct_of_distance"]
print(sub[cols].to_string())
