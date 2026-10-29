"""
Master orchestration script to train all 3 architectures across 3 random seeds (42, 123, 456).
Total runs: 9 models.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.train_speed_models import train_single_model


def main():
    print("=" * 70)
    print("STARTING PHASE 8 MULTI-ARCHITECTURE SPEED MODEL TRAINING")
    print("=" * 70)
    
    exp_dir = Path("experiments/phase8_corrected_speed_models")
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    models = [
        ("CNN-GRU", "experiments/phase8_corrected_speed_models/configs/cnn_gru.yaml"),
        ("ModernTCN", "experiments/phase8_corrected_speed_models/configs/moderntcn.yaml"),
        ("Mamba-2 SSD", "experiments/phase8_corrected_speed_models/configs/mamba2.yaml"),
    ]
    
    seeds = [42, 123, 456]
    all_results = []
    
    t0_global = time.perf_counter()
    
    for name, cfg_path in models:
        print(f"\n=======================================================")
        print(f"BENCHMARK ARCHITECTURE: {name}")
        print(f"=======================================================")
        for s in seeds:
            res = train_single_model(cfg_path, s, ckpt_dir)
            all_results.append(res)
            
    total_bench_time = time.perf_counter() - t0_global
    print(f"\nAll 9 model training runs completed in {total_bench_time/60.0:.2f} minutes.")
    
    # Save combined summary
    with open(exp_dir / "training_benchmark_summary.json", "w") as f:
        json.dump({
            "total_bench_time_s": total_bench_time,
            "models_trained": len(all_results),
            "runs": all_results,
        }, f, indent=2)
        
    print(f"Saved {exp_dir / 'training_benchmark_summary.json'}")


if __name__ == "__main__":
    main()
