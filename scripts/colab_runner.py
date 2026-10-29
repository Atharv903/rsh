"""
Colab Runner Utility for Automated GPS-Denied Navigation Benchmarking.
Handles local NVMe cache copying, Google Drive checkpoint persistence,
GPU smoke testing, full benchmark execution, and profiling.
"""

import os
import sys
import shutil
import time
import json
import argparse
from pathlib import Path

# Add project root
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from scripts.train_speed_models import get_default_device, train_single_model
from scripts.evaluate_speed_models import run_full_evaluation
from scripts.verify_numerical_consistency import verify_forward_consistency, verify_checkpoint_resume


def setup_local_storage(source_data_dir: Path, local_target_dir: Path) -> Path:
    """Copies processed datasets to high-speed local runtime storage to prevent Drive FUSE bottleneck."""
    print("=" * 65)
    print("SETTING UP HIGH-SPEED LOCAL RUNTIME STORAGE")
    print(f"Source Directory: {source_data_dir}")
    print(f"Target Directory: {local_target_dir}")
    print("=" * 65)
    
    local_target_dir.mkdir(parents=True, exist_ok=True)
    files_to_copy = [
        "train_windows_grouped.npz",
        "val_windows_grouped.npz",
        "test_windows_grouped.npz",
        "sensor_normalizer_grouped.json",
    ]
    
    t0 = time.perf_counter()
    total_bytes = 0
    for fname in files_to_copy:
        src_file = source_data_dir / fname
        dst_file = local_target_dir / fname
        if src_file.exists():
            if not dst_file.exists() or src_file.stat().st_size != dst_file.stat().st_size:
                shutil.copy2(src_file, dst_file)
            total_bytes += dst_file.stat().st_size
            print(f"  Cached {fname} ({dst_file.stat().st_size / (1024**2):.1f} MB)")
        else:
            print(f"  [Warning] {fname} not found in source directory {source_data_dir}")
            
    elapsed = time.perf_counter() - t0
    print(f"Local storage setup completed: {total_bytes / (1024**2):.1f} MB transferred in {elapsed:.2f}s ({total_bytes / max(1e-4, elapsed) / (1024**2):.1f} MB/s)\n")
    return local_target_dir


def run_gpu_smoke_test(device: torch.device, data_dir: Path, output_dir: Path):
    """Runs a 1-epoch GPU sanity check across all 3 architectures."""
    print("=" * 65)
    print(f"EXECUTING GPU SMOKE TEST ON: {device}")
    if device.type == "cuda":
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 65)
    
    configs = [
        ("CNN-GRU", "experiments/phase8_corrected_speed_models/configs/cnn_gru.yaml"),
        ("ModernTCN", "experiments/phase8_corrected_speed_models/configs/moderntcn.yaml"),
        ("Mamba-2 SSD", "experiments/phase8_corrected_speed_models/configs/mamba2.yaml"),
    ]
    
    smoke_dir = output_dir / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_results = []
    
    for name, cfg_path in configs:
        import yaml
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        # Quick 1-epoch smoke test
        cfg["training"]["epochs"] = 1
        cfg["training"]["patience"] = 1
        temp_cfg = smoke_dir / f"smoke_{Path(cfg_path).name}"
        with open(temp_cfg, "w") as f:
            yaml.dump(cfg, f)
            
        t0 = time.perf_counter()
        res = train_single_model(
            str(temp_cfg),
            seed=42,
            output_dir=smoke_dir / "checkpoints",
            device=device,
            resume=False,
            data_dir_override=str(data_dir),
        )
        elapsed = time.perf_counter() - t0
        
        val_mae = res["best_val_mae"]
        assert np.isfinite(val_mae), f"[{name}] Non-finite validation loss in smoke test!"
        print(f"  [{name} Smoke Test] SUCCESS: Val MAE={val_mae:.3f} m/s in {elapsed:.1f}s")
        smoke_results.append({
            "model": name,
            "val_mae": val_mae,
            "elapsed_s": elapsed,
            "device": str(device),
        })
        
    print("\nAll 3 architecture GPU smoke tests completed successfully!")
    return smoke_results


def main():
    parser = argparse.ArgumentParser(description="Google Colab GPU Benchmark Runner")
    parser.add_argument("--mode", type=str, default="smoke_test",
                        choices=["smoke_test", "verify_consistency", "full_benchmark", "evaluate_only"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--source_data_dir", type=str, default="data/processed")
    parser.add_argument("--local_cache_dir", type=str, default="data/processed")
    parser.add_argument("--persistent_dir", type=str, default="experiments/phase8_corrected_speed_models")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device != "auto" else get_default_device()
    src_data = Path(args.source_data_dir)
    local_data = Path(args.local_cache_dir)
    persist_dir = Path(args.persistent_dir)
    
    # 1. Setup local fast storage
    cached_data_dir = setup_local_storage(src_data, local_data)
    
    if args.mode == "verify_consistency":
        verify_forward_consistency()
        verify_checkpoint_resume()
        
    elif args.mode == "smoke_test":
        verify_forward_consistency()
        run_gpu_smoke_test(device, cached_data_dir, persist_dir)
        
    elif args.mode == "full_benchmark":
        # 1. Verify smoke test first
        print("\n--- Running Mandatory Pre-Benchmark Smoke Test ---")
        run_gpu_smoke_test(device, cached_data_dir, persist_dir)
        
        # 2. Run Full 9-Model Benchmark
        print("\n--- Starting Full 9-Model GPU Benchmark ---")
        from scripts.train_speed_models import train_single_model
        
        models = [
            ("CNN-GRU", "experiments/phase8_corrected_speed_models/configs/cnn_gru.yaml"),
            ("ModernTCN", "experiments/phase8_corrected_speed_models/configs/moderntcn.yaml"),
            ("Mamba-2 SSD", "experiments/phase8_corrected_speed_models/configs/mamba2.yaml"),
        ]
        
        ckpt_dir = persist_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        all_results = []
        t0_bench = time.perf_counter()
        
        for name, cfg_path in models:
            for s in args.seeds:
                res = train_single_model(
                    cfg_path,
                    s,
                    ckpt_dir,
                    device=device,
                    resume=True,
                    data_dir_override=str(cached_data_dir),
                )
                all_results.append(res)
                
        total_time_s = time.perf_counter() - t0_bench
        with open(persist_dir / "training_benchmark_summary.json", "w") as f:
            json.dump({
                "device": str(device),
                "total_bench_time_s": total_time_s,
                "models_trained": len(all_results),
                "runs": all_results,
            }, f, indent=2)
            
        # 3. Trajectory Evaluation
        print("\n--- Starting Outage Trajectory Benchmark ---")
        run_full_evaluation(
            device_str=str(device),
            data_dir=str(cached_data_dir),
            ckpt_dir=str(ckpt_dir),
            results_dir=str(persist_dir / "results"),
        )
        
    elif args.mode == "evaluate_only":
        run_full_evaluation(
            device_str=str(device),
            data_dir=str(cached_data_dir),
            ckpt_dir=str(persist_dir / "checkpoints"),
            results_dir=str(persist_dir / "results"),
        )


if __name__ == "__main__":
    main()
