"""
Master orchestration script to train all 3 architectures across 3 random seeds (42, 123, 456).
Total benchmark runs: 9 models.
Fully device-aware (CUDA when available, CPU fallback) with checkpoint resumption.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add project root
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from scripts.train_speed_models import train_single_model, get_default_device


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Multi-Architecture Training Benchmark")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--data_dir", type=str, default=None, help="Local fast directory for processed npz files")
    parser.add_argument("--output_dir", type=str, default="experiments/phase8_corrected_speed_models/checkpoints")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--architectures", type=str, nargs="+", default=["all"],
                        choices=["all", "cnn_gru", "moderntcn", "mamba2"])
    parser.add_argument("--no_resume", action="store_true", help="Disable resuming from last checkpoint")
    args = parser.parse_args()
    
    dev = None
    if args.device != "auto":
        dev = torch.device(args.device)
    else:
        dev = get_default_device()
        
    print("=" * 70, flush=True)
    print("STARTING PHASE 8 MULTI-ARCHITECTURE SPEED MODEL TRAINING", flush=True)
    print(f"Target Execution Device: {dev}", flush=True)
    if dev.type == "cuda":
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)", flush=True)
    print("=" * 70, flush=True)
    
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    all_models = [
        ("CNN-GRU", "experiments/phase8_corrected_speed_models/configs/cnn_gru.yaml", "cnn_gru"),
        ("ModernTCN", "experiments/phase8_corrected_speed_models/configs/moderntcn.yaml", "moderntcn"),
        ("Mamba-2 SSD", "experiments/phase8_corrected_speed_models/configs/mamba2.yaml", "mamba2"),
    ]
    
    if "all" not in args.architectures:
        selected_models = [m for m in all_models if m[2] in args.architectures]
    else:
        selected_models = all_models
        
    seeds = args.seeds
    all_results = []
    
    t0_global = time.perf_counter()
    
    for name, cfg_path, tag in selected_models:
        print(f"\n=======================================================", flush=True)
        print(f"BENCHMARK ARCHITECTURE: {name}", flush=True)
        print(f"=======================================================", flush=True)
        for s in seeds:
            res = train_single_model(
                cfg_path,
                s,
                ckpt_dir,
                device=dev,
                resume=not args.no_resume,
                data_dir_override=args.data_dir,
            )
            all_results.append(res)
            
    total_bench_time = time.perf_counter() - t0_global
    print(f"\nAll {len(all_results)} model training runs completed in {total_bench_time/60.0:.2f} minutes.", flush=True)
    
    summary_path = ckpt_dir.parent / "training_benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "device": str(dev),
            "total_bench_time_s": total_bench_time,
            "models_trained": len(all_results),
            "runs": all_results,
        }, f, indent=2)
        
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
