import json

with open('experiments/phase13/statistical_discrepancy_analysis.json', 'r') as f:
    data = json.load(f)

for row in data:
    if row['arch'] == 'ModernTCN':
        print(f"{row['session']} | {row['duration_s']}s | n={row['n']}")
        print(f"  diffs: {[round(x, 2) for x in row['diff_values']]}")
        print(f"  mean: {row['mean_diff']:.2f} m, std: {row['std_diff']:.2f} m, SE: {row['se_diff']:.2f} m")
        print(f"  Student t-CI95: [{row['t_interval'][0]:.2f}, {row['t_interval'][1]:.2f}] m")
        print(f"  Percentile Boot-CI95: [{row['percentile_boot_interval'][0]:.2f}, {row['percentile_boot_interval'][1]:.2f}] m")
        print(f"  BCa Boot-CI95: [{row['bca_boot_interval'][0]:.2f}, {row['bca_boot_interval'][1]:.2f}] m")
        print(f"  Paired t-test: t={row['paired_t_stat']:.3f}, p={row['paired_t_pval']:.4f}")
        print(f"  Wilcoxon test: W={row['wilcoxon_stat']}, p={row['wilcoxon_pval']:.4f}")
        print(f"  Sign test: p={row['sign_test_pval']:.4f}")
        print("-" * 60)
