import pandas as pd

df = pd.read_csv("experiments/phase16_validation/filter_diagnostics.csv")
print("Filter Diagnostics Summary across all 90 runs:")
print(f"Total speed updates accepted: {df['speed_updates_accepted'].sum()}, rejected: {df['speed_updates_rejected'].sum()}")
print(f"Total NHC updates accepted: {df['nhc_updates_accepted'].sum()}, rejected: {df['nhc_updates_rejected'].sum()}")
print(f"Total ZUPT updates accepted: {df['zupt_updates_accepted'].sum()}")
print(f"Average final covariance trace: {df['final_cov_trace'].mean():.2f}")
print(f"Mean speed innovation: {df['mean_speed_innovation'].mean():.3f} m/s")
print(f"Mean NHC innovation: {df['mean_nhc_innovation'].mean():.3f} m/s")
