import pandas as pd

df = pd.read_csv("experiments/phase15_validation/statistical_paired_comparisons.csv")
sub = df[(df["session"] == "POOLED") & (df["calibration_protocol"] == "Protocol_C_CausalOnline")]
cols = ["duration_s", "candidate_estimator", "baseline_estimator", "mean_drift_reduction_m", "median_drift_reduction_m", "ci_95_low_m", "ci_95_high_m", "p_value_paired_t", "p_value_wilcoxon", "p_value_holm_bonferroni_pooled_protocol_c"]
print(sub[cols].to_string())
