import pandas as pd

df = pd.read_csv('experiments/phase13/recomputed_statistical_forensics.csv')
for sess in ["Driver B", "Driver D", "POOLED (Driver B + D)"]:
    sub = df[df["session"] == sess]
    print(f"=== {sess} ===")
    for idx, r in sub.iterrows():
        print(f"  {r['duration_s']}s | {r['architecture']:<10} | Mean Adv: {r['mean_paired_advantage_m']:+7.2f}m | Win: {r['win_rate_pct']:3.0f}% | Raw p: {r['paired_t_pval']:.4f} | Holm p: {r['holm_bonferroni_pval']:.4f} | FDR p: {r['fdr_bh_pval']:.4f}")
