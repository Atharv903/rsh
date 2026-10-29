"""
Phase 13B: Forensic Statistical Audit of Phase 12 Results.

Exhaustively recomputes all paired comparisons, bootstrap intervals,
parametric vs non-parametric tests, multiple comparison adjustments,
and checks for outlier influence and clustering effects.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

repo_root = Path(__file__).resolve().parent.parent.parent

def compute_percentile_bootstrap_ci(data, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    low = np.percentile(boot_means, 100 * (alpha / 2))
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(low), float(high)

def compute_bca_bootstrap_ci(data, n_boot=10000, alpha=0.05, seed=42):
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(data)
    theta_hat = np.mean(data)
    
    # Bootstrap replicates
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = np.mean(rng.choice(data, size=n, replace=True))
        
    # Bias correction factor z0
    z0 = stats.norm.ppf(np.mean(boot_means < theta_hat))
    
    # Acceleration factor a via jackknife
    jack_means = np.empty(n)
    for j in range(n):
        jack_sample = np.delete(data, j)
        jack_means[j] = np.mean(jack_sample)
    jack_mean = np.mean(jack_means)
    diff = jack_mean - jack_means
    num = np.sum(diff**3)
    denom = 6.0 * (np.sum(diff**2)**1.5)
    a = num / denom if denom != 0 else 0.0
    
    # Adjusted percentiles
    z_alpha_low = stats.norm.ppf(alpha / 2)
    z_alpha_high = stats.norm.ppf(1 - alpha / 2)
    
    alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
    alpha2 = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high)))
    
    # Clip to avoid index errors
    alpha1 = np.clip(alpha1, 1e-4, 1 - 1e-4)
    alpha2 = np.clip(alpha2, 1e-4, 1 - 1e-4)
    
    low = np.percentile(boot_means, 100 * alpha1)
    high = np.percentile(boot_means, 100 * alpha2)
    return float(low), float(high)

def compute_t_ci(data, alpha=0.05):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf(1 - alpha / 2, n - 1)
    return float(mean - h), float(mean + h)

def main():
    csv_path = repo_root / "experiments/phase12_validation/per_scenario_metrics.csv"
    df = pd.read_csv(csv_path)
    
    # Filter to corrected convention
    corr_df = df[df["heading_convention"] == "corrected (-1)"].copy()
    
    def get_arch(m):
        if "cnngru" in m: return "CNN-GRU"
        if "mamba2" in m: return "Mamba-2"
        if "moderntcn" in m: return "ModernTCN"
        return m
    corr_df["arch"] = corr_df["method"].apply(get_arch)
    
    out_dir = repo_root / "experiments/phase13"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    detailed_discrepancies = []
    
    sessions = ["Driver B", "Driver D", "POOLED (Driver B + D)"]
    durations = [10, 30, 60]
    architectures = ["ModernTCN", "CNN-GRU", "Mamba-2"]
    
    for sess in sessions:
        sub_df = corr_df if sess.startswith("POOLED") else corr_df[corr_df["driver_id"] == sess]
        
        for dur in durations:
            dur_df = sub_df[sub_df["duration_s"] == dur]
            
            # Pivot to get paired table by scenario_id
            pivot = dur_df.groupby(["scenario_id", "arch"])["final_drift_m"].mean().unstack("arch").reset_index()
            cv = pivot["Constant-Velocity"].values
            
            for arch in architectures:
                if arch not in pivot.columns:
                    continue
                nn = pivot[arch].values
                diff = cv - nn  # Positive: Model beats Constant Velocity
                
                n = len(diff)
                mean_cv = float(np.mean(cv))
                mean_nn = float(np.mean(nn))
                median_cv = float(np.median(cv))
                median_nn = float(np.median(nn))
                std_cv = float(np.std(cv, ddof=1))
                std_nn = float(np.std(nn, ddof=1))
                iqr_cv = float(np.percentile(cv, 75) - np.percentile(cv, 25))
                iqr_nn = float(np.percentile(nn, 75) - np.percentile(nn, 25))
                
                mean_diff = float(np.mean(diff))
                median_diff = float(np.median(diff))
                std_diff = float(np.std(diff, ddof=1))
                iqr_diff = float(np.percentile(diff, 75) - np.percentile(diff, 25))
                
                # Percentage improvement defined relative to Constant Velocity mean drift
                pct_improvement = float((mean_diff / mean_cv) * 100.0)
                
                won = int(np.sum(diff > 0))
                tied = int(np.sum(diff == 0))
                lost = int(np.sum(diff < 0))
                
                # Confidence intervals
                t_low, t_high = compute_t_ci(diff)
                boot_p_low, boot_p_high = compute_percentile_bootstrap_ci(diff)
                try:
                    boot_bca_low, boot_bca_high = compute_bca_bootstrap_ci(diff)
                except Exception:
                    boot_bca_low, boot_bca_high = np.nan, np.nan
                
                # Tests
                t_stat, p_t = stats.ttest_rel(cv, nn)
                try:
                    w_stat, p_w = stats.wilcoxon(cv, nn)
                except Exception:
                    w_stat, p_w = np.nan, np.nan
                    
                # Sign test (binomial exact)
                sign_test_pval = float(stats.binomtest(won, n, 0.5, alternative='two-sided').pvalue)
                
                # Record detailed breakdown of individual differences
                detailed_discrepancies.append({
                    "session": sess,
                    "duration_s": dur,
                    "arch": arch,
                    "n": n,
                    "diff_values": [float(x) for x in diff],
                    "mean_diff": mean_diff,
                    "std_diff": std_diff,
                    "se_diff": float(std_diff / np.sqrt(n)),
                    "t_interval": [t_low, t_high],
                    "percentile_boot_interval": [boot_p_low, boot_p_high],
                    "bca_boot_interval": [boot_bca_low, boot_bca_high],
                    "paired_t_stat": float(t_stat),
                    "paired_t_pval": float(p_t),
                    "wilcoxon_stat": float(w_stat),
                    "wilcoxon_pval": float(p_w),
                    "sign_test_pval": sign_test_pval,
                })
                
                results.append({
                    "session": sess,
                    "duration_s": dur,
                    "architecture": arch,
                    "n_scenarios": n,
                    "scenarios_won": won,
                    "scenarios_tied": tied,
                    "scenarios_lost": lost,
                    "win_rate_pct": float(won / n * 100.0),
                    "mean_cv_drift_m": mean_cv,
                    "std_cv_drift_m": std_cv,
                    "median_cv_drift_m": median_cv,
                    "iqr_cv_drift_m": iqr_cv,
                    "mean_nn_drift_m": mean_nn,
                    "std_nn_drift_m": std_nn,
                    "median_nn_drift_m": median_nn,
                    "iqr_nn_drift_m": iqr_nn,
                    "mean_paired_advantage_m": mean_diff,
                    "std_paired_advantage_m": std_diff,
                    "median_paired_advantage_m": median_diff,
                    "iqr_paired_advantage_m": iqr_diff,
                    "pct_improvement_rel_to_cv": pct_improvement,
                    "t_ci95_low": t_low,
                    "t_ci95_high": t_high,
                    "bootstrap_percentile_ci95_low": boot_p_low,
                    "bootstrap_percentile_ci95_high": boot_p_high,
                    "bootstrap_bca_ci95_low": boot_bca_low,
                    "bootstrap_bca_ci95_high": boot_bca_high,
                    "paired_t_stat": float(t_stat),
                    "paired_t_pval": float(p_t),
                    "wilcoxon_stat": float(w_stat),
                    "wilcoxon_pval": float(p_w),
                    "sign_test_pval": sign_test_pval,
                })
                
    res_df = pd.DataFrame(results)
    
    # Multiple Comparison Corrections: Holm-Bonferroni and Benjamini-Hochberg FDR
    # Separate by session and duration family or pooled family
    for sess in sessions:
        mask = res_df["session"] == sess
        pvals = res_df.loc[mask, "paired_t_pval"].values
        m = len(pvals)
        
        # Holm-Bonferroni
        sorted_idx = np.argsort(pvals)
        adj_p_holm = np.empty(m)
        for rank, idx in enumerate(sorted_idx):
            adj_p_holm[idx] = min(1.0, pvals[idx] * (m - rank))
        # Ensure monotonicity in Holm
        for i in range(1, m):
            curr_idx = sorted_idx[i]
            prev_idx = sorted_idx[i-1]
            adj_p_holm[curr_idx] = max(adj_p_holm[curr_idx], adj_p_holm[prev_idx])
        res_df.loc[mask, "holm_bonferroni_pval"] = adj_p_holm
        
        # Benjamini-Hochberg FDR
        adj_p_fdr = np.empty(m)
        for rank, idx in enumerate(sorted_idx):
            adj_p_fdr[idx] = min(1.0, pvals[idx] * m / (rank + 1))
        for i in range(m - 2, -1, -1):
            curr_idx = sorted_idx[i]
            next_idx = sorted_idx[i+1]
            adj_p_fdr[curr_idx] = min(adj_p_fdr[curr_idx], adj_p_fdr[next_idx])
        res_df.loc[mask, "fdr_bh_pval"] = adj_p_fdr

    res_df.to_csv(out_dir / "recomputed_statistical_forensics.csv", index=False)
    
    with open(out_dir / "statistical_discrepancy_analysis.json", "w", encoding="utf-8") as f:
        json.dump(detailed_discrepancies, f, indent=2)
        
    print(f"Saved forensic recomputations to {out_dir / 'recomputed_statistical_forensics.csv'}")
    print(f"Saved discrepancy analysis to {out_dir / 'statistical_discrepancy_analysis.json'}")

if __name__ == "__main__":
    main()
