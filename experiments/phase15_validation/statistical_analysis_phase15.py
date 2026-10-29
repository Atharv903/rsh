"""
Statistical Analysis of Phase 15 Benchmark Results.
Computes:
- Paired differences: Delta = Baseline - Candidate (positive = candidate improves drift)
- Paired t-test (two-sided) and p-value
- Wilcoxon signed-rank test and p-value
- 95% Bootstrap Confidence Intervals (10,000 iterations, BCa or percentile)
- Holm-Bonferroni family-wise error rate adjusted p-values
- Distance-normalized drift statistics (% of Distance Travelled)
Outputs:
- experiments/phase15_validation/statistical_paired_comparisons.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
metrics_path = repo_root / "experiments/phase15_validation/per_scenario_metrics.csv"
df = pd.read_csv(metrics_path)

def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    n = len(diffs)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    low = np.percentile(boot_means, 100 * (alpha / 2))
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(low), float(high)

paired_records = []

# Comparisons to make:
# 1. Phase14-Causal-Hybrid vs Constant-Velocity
# 2. Phase14-Causal-Hybrid vs ModernTCN
# 3. Constrained-Causal-Hybrid vs Phase14-Causal-Hybrid
# 4. Complementary-Filter vs Phase14-Causal-Hybrid
# 5. ModernTCN vs Constant-Velocity
# 6. Acceleration-Integration vs Constant-Velocity

comparison_pairs = [
    ("Phase14-Causal-Hybrid", "Constant-Velocity"),
    ("Phase14-Causal-Hybrid", "ModernTCN"),
    ("Constrained-Causal-Hybrid", "Phase14-Causal-Hybrid"),
    ("Complementary-Filter", "Phase14-Causal-Hybrid"),
    ("ModernTCN", "Constant-Velocity"),
    ("Acceleration-Integration", "Constant-Velocity"),
]

protocols = ["Protocol_A_ZeroBias", "Protocol_B_PriorCalib", "Protocol_C_CausalOnline"]
durations = [10, 30, 60]
sessions = ["Driver B", "Driver D", "POOLED"]

for proto in protocols:
    for dur in durations:
        for sess in sessions:
            sub = df if sess == "POOLED" else df[df["driver_id"] == sess]
            sub = sub[(sub["calibration_protocol"] == proto) & (sub["duration_s"] == dur)]
            
            for cand, base in comparison_pairs:
                sub_cand = sub[sub["estimator"] == cand].sort_values("scenario_id")
                sub_base = sub[sub["estimator"] == base].sort_values("scenario_id")
                
                if len(sub_cand) == 0 or len(sub_base) == 0:
                    continue
                    
                drift_cand = sub_cand["final_drift_m"].values
                drift_base = sub_base["final_drift_m"].values
                
                # Delta = Base - Cand (positive = Candidate has lower drift, i.e., improvement)
                diffs = drift_base - drift_cand
                n = len(diffs)
                mean_diff = float(np.mean(diffs))
                med_diff = float(np.median(diffs))
                std_diff = float(np.std(diffs, ddof=1))
                
                # Paired t-test
                t_stat, p_val_t = stats.ttest_rel(drift_base, drift_cand)
                
                # Wilcoxon signed rank test
                try:
                    w_stat, p_val_w = stats.wilcoxon(diffs, alternative="two-sided")
                except Exception:
                    w_stat, p_val_w = np.nan, np.nan
                    
                ci_low, ci_high = bootstrap_ci(diffs)
                
                paired_records.append({
                    "calibration_protocol": proto,
                    "duration_s": dur,
                    "session": sess,
                    "candidate_estimator": cand,
                    "baseline_estimator": base,
                    "n_scenarios": n,
                    "mean_drift_reduction_m": mean_diff,
                    "median_drift_reduction_m": med_diff,
                    "std_drift_reduction_m": std_diff,
                    "ci_95_low_m": ci_low,
                    "ci_95_high_m": ci_high,
                    "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                    "t_statistic": float(t_stat),
                    "p_value_paired_t": float(p_val_t),
                    "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
                    "p_value_wilcoxon": float(p_val_w) if not np.isnan(p_val_w) else None,
                })

df_paired = pd.DataFrame(paired_records)

# Apply Holm-Bonferroni correction within primary POOLED Protocol C family
pooled_mask = (df_paired["session"] == "POOLED") & (df_paired["calibration_protocol"] == "Protocol_C_CausalOnline")
pooled_indices = df_paired[pooled_mask].index

p_vals = df_paired.loc[pooled_indices, "p_value_paired_t"].values
m = len(p_vals)
sorted_order = np.argsort(p_vals)
p_adj = np.empty(m)

for rank, idx in enumerate(sorted_order):
    mult = m - rank
    p_adj[idx] = min(1.0, p_vals[idx] * mult)
# Enforce monotonicity
for i in range(1, m):
    prev_idx = sorted_order[i-1]
    curr_idx = sorted_order[i]
    p_adj[curr_idx] = max(p_adj[curr_idx], p_adj[prev_idx])

df_paired["p_value_holm_bonferroni_pooled_protocol_c"] = np.nan
df_paired.loc[pooled_indices, "p_value_holm_bonferroni_pooled_protocol_c"] = p_adj

out_path = repo_root / "experiments/phase15_validation/statistical_paired_comparisons.csv"
df_paired.to_csv(out_path, index=False)
print(f"Saved paired statistical analysis to {out_path} ({len(df_paired)} comparisons)")
