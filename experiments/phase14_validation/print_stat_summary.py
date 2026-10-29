import pandas as pd

df = pd.read_csv("experiments/phase14_validation/statistical_paired_comparisons.csv")
pooled = df[(df["session"] == "POOLED") & (df["candidate_estimator"] == "Corrected-Causal-Hybrid")]

print(f"{'Protocol':<23} | {'Dur':<3} | {'Mean Adv vs CV':<14} | {'Win vs CV':<9} | {'Raw p':<6} | {'Holm p':<6} | {'Boot CI vs CV'}")
print("-" * 90)
for idx, r in pooled.iterrows():
    print(f"{r['calibration_protocol']:<23} | {r['duration_s']:<3} | {r['mean_adv_vs_cv_m']:+7.2f} m ({r['median_adv_vs_cv_m']:+6.2f}m) | {r['won_vs_cv']}/10 ({r['win_rate_vs_cv_pct']:3.0f}%) | {r['pval_vs_cv']:.4f} | {r['holm_adj_pval_vs_cv']:.4f} | [{r['boot_ci95_vs_cv_low']:+6.2f}, {r['boot_ci95_vs_cv_high']:+6.2f}] m")
