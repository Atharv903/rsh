import pandas as pd

df = pd.read_csv("experiments/phase14_validation/statistical_paired_comparisons.csv")
pooled = df[(df["session"] == "POOLED") & (df["candidate_estimator"] == "Corrected-Causal-Hybrid")]

print(f"{'Protocol':<23} | {'Dur':<3} | {'Mean Adv vs NN':<14} | {'Win vs NN':<9} | {'Raw p':<6} | {'t-CI95 vs NN'}")
print("-" * 80)
for idx, r in pooled.iterrows():
    print(f"{r['calibration_protocol']:<23} | {r['duration_s']:<3} | {r['mean_adv_vs_nn_m']:+7.2f} m | {r['won_vs_nn']}/10 ({r['won_vs_nn']*10:3.0f}%) | {r['pval_vs_nn']:.4f} | [{r['t_ci95_vs_nn_low']:+6.2f}, {r['t_ci95_vs_nn_high']:+6.2f}] m")
