"""
Phase 14: Comprehensive Statistical Analysis and Cross-Validation.

Computes:
- Paired comparisons vs Constant-Velocity and ModernTCN
- Student's t 95% CI and p-values
- Percentile bootstrap 95% CI (10,000 resamples, seed 42)
- Holm-Bonferroni adjusted p-values
- Cross-validation of hybrid parameters across Driver B and Driver D
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

repo_root = Path(__file__).resolve().parent.parent.parent

def compute_bootstrap_ci(diff_series, n_bootstrap=10000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n = len(diff_series)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(diff_series, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    low = np.percentile(boot_means, 100 * (alpha / 2))
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(low), float(high)

def compute_t_ci(diff_series, alpha=0.05):
    n = len(diff_series)
    mean = np.mean(diff_series)
    se = stats.sem(diff_series)
    h = se * stats.t.ppf(1 - alpha / 2, n - 1)
    return float(mean - h), float(mean + h)

def main():
    metrics_path = repo_root / "experiments/phase14_validation/per_scenario_metrics.csv"
    if not metrics_path.exists():
        print(f"{metrics_path} does not exist.")
        return
        
    df = pd.read_csv(metrics_path)
    out_dir = repo_root / "experiments/phase14_validation"
    
    paired_rows = []
    
    sessions = ["Driver B", "Driver D", "POOLED"]
    protocols = ["Protocol_A_ZeroBias", "Protocol_B_PriorCalib", "Protocol_C_CausalOnline"]
    durations = [10, 30, 60]
    
    for proto in protocols:
        proto_df = df[df["calibration_protocol"] == proto]
        
        for dur in durations:
            dur_df = proto_df[proto_df["duration_s"] == dur]
            
            for sess in sessions:
                s_df = dur_df if sess == "POOLED" else dur_df[dur_df["driver_id"] == sess]
                
                # Pivot by scenario_id and estimator
                piv = s_df.pivot(index="scenario_id", columns="estimator", values="final_drift_m").reset_index()
                
                cv_vals = piv["Constant-Velocity"].values
                mtcn_vals = piv["ModernTCN"].values
                hyb_exist_vals = piv["Existing-Hybrid"].values
                hyb_causal_vals = piv["Corrected-Causal-Hybrid"].values
                
                n = len(cv_vals)
                
                # Candidate estimators to compare against CV and ModernTCN
                candidates = [
                    ("ModernTCN", mtcn_vals),
                    ("Existing-Hybrid", hyb_exist_vals),
                    ("Corrected-Causal-Hybrid", hyb_causal_vals),
                ]
                
                for cand_name, cand_vals in candidates:
                    # 1. Advantage vs Constant-Velocity: CV - Candidate (positive = candidate wins)
                    diff_vs_cv = cv_vals - cand_vals
                    mean_adv_cv = float(np.mean(diff_vs_cv))
                    med_adv_cv = float(np.median(diff_vs_cv))
                    std_adv_cv = float(np.std(diff_vs_cv, ddof=1))
                    iqr_adv_cv = float(np.percentile(diff_vs_cv, 75) - np.percentile(diff_vs_cv, 25))
                    won_vs_cv = int(np.sum(diff_vs_cv > 0))
                    lost_vs_cv = int(np.sum(diff_vs_cv < 0))
                    
                    t_stat_cv, p_t_cv = stats.ttest_rel(cv_vals, cand_vals)
                    t_low_cv, t_high_cv = compute_t_ci(diff_vs_cv)
                    boot_low_cv, boot_high_cv = compute_bootstrap_ci(diff_vs_cv)
                    
                    # 2. Advantage vs ModernTCN: ModernTCN - Candidate (positive = candidate wins)
                    diff_vs_nn = mtcn_vals - cand_vals
                    mean_adv_nn = float(np.mean(diff_vs_nn))
                    med_adv_nn = float(np.median(diff_vs_nn))
                    std_adv_nn = float(np.std(diff_vs_nn, ddof=1))
                    iqr_adv_nn = float(np.percentile(diff_vs_nn, 75) - np.percentile(diff_vs_nn, 25))
                    won_vs_nn = int(np.sum(diff_vs_nn > 0))
                    lost_vs_nn = int(np.sum(diff_vs_nn < 0))
                    
                    t_stat_nn, p_t_nn = stats.ttest_rel(mtcn_vals, cand_vals) if cand_name != "ModernTCN" else (np.nan, np.nan)
                    t_low_nn, t_high_nn = compute_t_ci(diff_vs_nn) if cand_name != "ModernTCN" else (np.nan, np.nan)
                    boot_low_nn, boot_high_nn = compute_bootstrap_ci(diff_vs_nn) if cand_name != "ModernTCN" else (np.nan, np.nan)
                    
                    paired_rows.append({
                        "session": sess,
                        "calibration_protocol": proto,
                        "duration_s": dur,
                        "candidate_estimator": cand_name,
                        "n_scenarios": n,
                        "mean_candidate_drift_m": float(np.mean(cand_vals)),
                        "mean_cv_drift_m": float(np.mean(cv_vals)),
                        "mean_adv_vs_cv_m": mean_adv_cv,
                        "median_adv_vs_cv_m": med_adv_cv,
                        "std_adv_vs_cv_m": std_adv_cv,
                        "iqr_adv_vs_cv_m": iqr_adv_cv,
                        "won_vs_cv": won_vs_cv,
                        "lost_vs_cv": lost_vs_cv,
                        "win_rate_vs_cv_pct": float(won_vs_cv / n * 100.0),
                        "t_stat_vs_cv": float(t_stat_cv),
                        "pval_vs_cv": float(p_t_cv),
                        "t_ci95_vs_cv_low": t_low_cv,
                        "t_ci95_vs_cv_high": t_high_cv,
                        "boot_ci95_vs_cv_low": boot_low_cv,
                        "boot_ci95_vs_cv_high": boot_high_cv,
                        "mean_adv_vs_nn_m": mean_adv_nn,
                        "won_vs_nn": won_vs_nn,
                        "lost_vs_nn": lost_vs_nn,
                        "pval_vs_nn": float(p_t_nn),
                        "t_ci95_vs_nn_low": t_low_nn,
                        "t_ci95_vs_nn_high": t_high_nn,
                    })
                    
    df_paired = pd.DataFrame(paired_rows)
    
    # Apply Holm-Bonferroni within each session and protocol family
    for sess in sessions:
        for proto in protocols:
            mask = (df_paired["session"] == sess) & (df_paired["calibration_protocol"] == proto)
            pvals = df_paired.loc[mask, "pval_vs_cv"].values
            m = len(pvals)
            sorted_idx = np.argsort(pvals)
            adj_p = np.empty(m)
            for rank, idx in enumerate(sorted_idx):
                adj_p[idx] = min(1.0, pvals[idx] * (m - rank))
            for i in range(1, m):
                curr = sorted_idx[i]
                prev = sorted_idx[i-1]
                adj_p[curr] = max(adj_p[curr], adj_p[prev])
            df_paired.loc[mask, "holm_adj_pval_vs_cv"] = adj_p
            
    df_paired.to_csv(out_dir / "statistical_paired_comparisons.csv", index=False)
    print(f"Saved {out_dir / 'statistical_paired_comparisons.csv'}")

if __name__ == "__main__":
    main()
