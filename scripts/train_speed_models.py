"""
Unified Training Script for Phase 8 Corrected Speed Models.
Trains CNN-GRU, ModernTCN, and Mamba-2 (custom SSD) on scalar speed targets.
Fully device-aware (CUDA with automatic CPU fallback) and fault-tolerant
with per-epoch checkpoints and auto-resume capability.
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add repository root to python search path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn_gru import CNNGRUModel
from src.models.modern_tcn import ModernTCNModel
from src.models.mamba2 import Mamba2Model


def get_default_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(config: Dict[str, Any]) -> nn.Module:
    m_type = config.get("model_type", "CNNGRUModel")
    m_args = config.get("model", {})
    if m_type == "CNNGRUModel":
        return CNNGRUModel(**m_args)
    elif m_type == "ModernTCNModel":
        return ModernTCNModel(**m_args)
    elif m_type == "Mamba2Model":
        return Mamba2Model(**m_args)
    else:
        raise ValueError(f"Unknown model type: {m_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_inference_latency(model: nn.Module, device: torch.device) -> Dict[str, float]:
    """Rigorous inference latency benchmark: 20 warmups, 100 repetitions.
    Uses CUDA synchronization when running on GPU.
    """
    model.eval()
    is_cuda = device.type == "cuda"
    
    # Batch size 1
    dummy_b1 = torch.randn(1, 100, 6, device=device)
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_b1)
            if is_cuda:
                torch.cuda.synchronize()
            
    times_b1 = []
    with torch.no_grad():
        for _ in range(100):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_b1)
            if is_cuda:
                torch.cuda.synchronize()
            times_b1.append((time.perf_counter() - t0) * 1000.0)  # ms
            
    # Batch size 32
    dummy_b32 = torch.randn(32, 100, 6, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_b32)
            if is_cuda:
                torch.cuda.synchronize()
            
    times_b32 = []
    with torch.no_grad():
        for _ in range(50):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_b32)
            if is_cuda:
                torch.cuda.synchronize()
            times_b32.append((time.perf_counter() - t0) * 1000.0)  # ms
            
    return {
        "device": str(device),
        "latency_b1_mean_ms": float(np.mean(times_b1)),
        "latency_b1_p95_ms": float(np.percentile(times_b1, 95)),
        "latency_b32_mean_ms": float(np.mean(times_b32)),
        "latency_b32_per_sample_ms": float(np.mean(times_b32) / 32.0),
    }


def train_single_model(
    config_path: str,
    seed: int,
    output_dir: Path,
    device: Optional[torch.device] = None,
    resume: bool = True,
    data_dir_override: Optional[str] = None,
) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(seed)
    if device is None:
        device = get_default_device()
        
    model_name = config.get("model_type", "Model")
    run_dir = output_dir / f"{model_name.lower()}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[{model_name} | Seed {seed}] Target Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}", flush=True)
        torch.cuda.reset_peak_memory_stats(0)
    
    # Resolve Data Paths (with optional override for fast local storage on Colab)
    train_npz = Path(config["data"]["train_npz"])
    val_npz = Path(config["data"]["val_npz"])
    if data_dir_override is not None:
        train_npz = Path(data_dir_override) / train_npz.name
        val_npz = Path(data_dir_override) / val_npz.name
        
    tr_data = np.load(train_npz, allow_pickle=True)
    va_data = np.load(val_npz, allow_pickle=True)
    
    X_tr = torch.from_numpy(tr_data["X"].astype(np.float32))
    y_tr = torch.from_numpy(tr_data["y"].astype(np.float32))
    
    X_va = torch.from_numpy(va_data["X"].astype(np.float32))
    y_va = torch.from_numpy(va_data["y"].astype(np.float32))
    
    # Subsample stride if specified
    sub_stride = config["training"].get("subsample_stride", 1)
    if sub_stride > 1:
        X_tr = X_tr[::sub_stride]
        y_tr = y_tr[::sub_stride]
        
    batch_size = config["training"].get("batch_size", 64)
    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(X_va, y_va),
        batch_size=128,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    
    # Build Model and Optimizer
    model = build_model(config).to(device)
    param_count = count_parameters(model)
    
    lr = float(config["training"].get("learning_rate", 1e-3))
    wd = float(config["training"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    epochs = config["training"].get("epochs", 12)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.HuberLoss(delta=1.0)
    patience = config["training"].get("patience", 4)
    grad_clip = config["training"].get("grad_clip", 1.0)
    
    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    epoch_logs = []
    start_epoch = 1
    
    # Check for resume checkpoint
    last_ckpt_path = run_dir / "last_checkpoint.pt"
    best_ckpt_path = run_dir / "best_model.pt"
    
    if resume and last_ckpt_path.exists():
        print(f"  [Resume] Found existing checkpoint at {last_ckpt_path}. Loading state...", flush=True)
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_mae = ckpt.get("best_val_mae", float("inf"))
        best_epoch = ckpt.get("best_epoch", 0)
        patience_counter = ckpt.get("patience_counter", 0)
        epoch_logs = ckpt.get("epoch_history", [])
        start_epoch = ckpt["epoch"] + 1
        print(f"  [Resume] Resuming training from epoch {start_epoch}/{epochs} (Best Val MAE: {best_val_mae:.3f} m/s at epoch {best_epoch})", flush=True)
        
    if start_epoch > epochs or (best_ckpt_path.exists() and run_dir.joinpath("run_metadata.json").exists() and not resume):
        if (run_dir / "run_metadata.json").exists():
            print(f"  [Skip] Run already complete. Returning saved results from run_metadata.json", flush=True)
            with open(run_dir / "run_metadata.json", "r") as f:
                return json.load(f)
                
    t_start = time.perf_counter()
    
    for epoch in range(start_epoch, epochs + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        n_batches = 0
        n_samples_trained = 0
        
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            n_samples_trained += len(bx)
            
        train_loss = total_loss / max(1, n_batches)
        scheduler.step()
        epoch_train_time_s = time.perf_counter() - t_epoch_start
        throughput_samples_per_s = n_samples_trained / max(1e-4, epoch_train_time_s)
        
        # Validation Evaluation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                p = model(bx)
                val_preds.append(p.detach().cpu().numpy())
                val_targets.append(by.detach().cpu().numpy())
                
        preds_arr = np.concatenate(val_preds, axis=0).flatten()
        targets_arr = np.concatenate(val_targets, axis=0).flatten()
        
        val_errors = np.abs(preds_arr - targets_arr)
        val_mae = float(np.mean(val_errors))
        val_rmse = float(np.sqrt(np.mean((preds_arr - targets_arr)**2)))
        val_bias = float(np.mean(preds_arr - targets_arr))
        
        max_gpu_mem_mb = 0.0
        if device.type == "cuda":
            max_gpu_mem_mb = float(torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0))
            
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_bias": val_bias,
            "lr": float(scheduler.get_last_lr()[0]),
            "epoch_train_time_s": epoch_train_time_s,
            "throughput_samples_per_s": throughput_samples_per_s,
            "max_gpu_mem_mb": max_gpu_mem_mb,
        })
        
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "seed": seed,
                "best_epoch": best_epoch,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_bias": val_bias,
                "device": str(device),
            }, best_ckpt_path)
            mark = "(*BEST*)"
        else:
            patience_counter += 1
            mark = ""
            
        # Save last checkpoint for fault-tolerant resumption
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "epoch_history": epoch_logs,
            "config": config,
            "seed": seed,
            "device": str(device),
        }, last_ckpt_path)
            
        print(f"  Epoch {epoch:2d}/{epochs:2d}: Train Loss={train_loss:.4f} | Val MAE={val_mae:.3f} m/s | Val RMSE={val_rmse:.3f} m/s ({epoch_train_time_s:.1f}s, {throughput_samples_per_s:.0f} samp/s) {mark}", flush=True)
        
        if patience_counter >= patience:
            print(f"  Early stopping triggered at epoch {epoch} (Patience: {patience}).", flush=True)
            break
            
    total_time_s = time.perf_counter() - t_start
    
    # Reload best model for latency benchmark and final logging
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        
    latency_info = benchmark_inference_latency(model, device)
    
    run_results = {
        "model_type": model_name,
        "seed": seed,
        "device": str(device),
        "param_count": param_count,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "total_training_time_s": total_time_s,
        "latency_benchmark": latency_info,
        "epoch_history": epoch_logs,
    }
    
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(run_results, f, indent=2)
        
    print(f"[{model_name} | Seed {seed}] Finished in {total_time_s:.1f}s on {device}. Best Val MAE={best_val_mae:.3f} m/s (Params: {param_count:,}).\n", flush=True)
    return run_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--output_dir", type=str, default="experiments/phase8_corrected_speed_models/checkpoints")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--data_dir", type=str, default=None, help="Optional override directory for processed datasets")
    parser.add_argument("--no_resume", action="store_true", help="Disable resuming from last checkpoint")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dev = None
    if args.device != "auto":
        dev = torch.device(args.device)
        
    for seed in args.seeds:
        train_single_model(
            args.config,
            seed,
            out_dir,
            device=dev,
            resume=not args.no_resume,
            data_dir_override=args.data_dir,
        )


if __name__ == "__main__":
    main()
