"""
Unified training script for Inertial Navigation models.
Supports CNN-GRU, ModernTCN, and Mamba-2 architectures.
"""

import os
import sys
import time
import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Tuple

# Add repository root to python search path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import IMUNavigationDataset
from src.models.cnn_gru import CNNGRUModel
from src.models.modern_tcn import ModernTCNModel
from src.models.mamba2 import Mamba2Model


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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()
        
        total_loss += loss.item() * len(batch_y)
        count += len(batch_y)
        
    return total_loss / max(1, count)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        
        total_loss += loss.item() * len(batch_y)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())
        
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    
    # 2D velocity magnitude error: norm(v_pred - v_true)
    vel_errors = np.linalg.norm(preds_arr - targets_arr, axis=1)
    mae = float(np.mean(vel_errors))
    rmse = float(np.sqrt(np.mean(vel_errors ** 2)))
    
    # Per component error
    e_mae = float(np.mean(np.abs(preds_arr[:, 0] - targets_arr[:, 0])))
    n_mae = float(np.mean(np.abs(preds_arr[:, 1] - targets_arr[:, 1])))
    
    return {
        "loss": total_loss / max(1, len(preds_arr)),
        "vel_mae": mae,
        "vel_rmse": rmse,
        "v_east_mae": e_mae,
        "v_north_mae": n_mae,
    }


def train_single_run(
    config: Dict[str, Any],
    seed: int,
    output_dir: Path,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_npz = config["data"]["train_npz"]
    val_npz = config["data"]["val_npz"]
    test_npz = config["data"]["test_npz"]
    filter_valid = config["data"].get("filter_valid_only", True)
    
    train_ds = IMUNavigationDataset(train_npz, filter_valid_only=filter_valid)
    val_ds = IMUNavigationDataset(val_npz, filter_valid_only=filter_valid)
    test_ds = IMUNavigationDataset(test_npz, filter_valid_only=filter_valid)
    
    # Subsample training windows by stride to reduce redundancy while preserving complete drive coverage
    train_stride = config["data"].get("subsample_stride", 5)
    if train_stride > 1 and not smoke_test:
        train_ds.X = train_ds.X[::train_stride]
        train_ds.y = train_ds.y[::train_stride]
        train_ds.t_start = train_ds.t_start[::train_stride]
        train_ds.t_end = train_ds.t_end[::train_stride]
        train_ds.drive_id = train_ds.drive_id[::train_stride]
        print(f"Applied train subsample stride {train_stride} -> {len(train_ds):,} windows.", flush=True)
    
    batch_size = 16 if smoke_test else config["training"].get("batch_size", 64)
    epochs = 2 if smoke_test else config["training"].get("epochs", 25)
    
    # If smoke test, subsample dataset
    if smoke_test:
        train_ds.X = train_ds.X[:128]
        train_ds.y = train_ds.y[:128]
        val_ds.X = val_ds.X[:64]
        val_ds.y = val_ds.y[:64]
        test_ds.X = test_ds.X[:64]
        test_ds.y = test_ds.y[:64]
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Build model
    model = build_model(config).to(device)
    param_count = count_parameters(model)
    
    # Optimizer & Scheduler
    lr = float(config["training"].get("learning_rate", 1e-3))
    wd = float(config["training"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    scheduler_type = config["training"].get("scheduler", "cosine")
    min_lr = float(config["training"].get("min_lr", 1e-6))
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    else:
        scheduler = None
        
    # Loss
    loss_name = config["training"].get("loss_fn", "huber")
    if loss_name == "huber":
        delta = float(config["training"].get("huber_delta", 1.0))
        criterion = nn.HuberLoss(delta=delta)
    else:
        criterion = nn.MSELoss()
        
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    patience = config["training"].get("patience", 8)
    
    run_dir = output_dir / f"{config['model_name']}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        
        if scheduler:
            scheduler.step()
            
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_vel_mae": val_metrics["vel_mae"],
            "val_vel_rmse": val_metrics["vel_rmse"],
            "lr": current_lr,
            "time_s": epoch_time,
        }
        history.append(log_entry)
        
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "seed": seed,
                "val_metrics": val_metrics,
            }, run_dir / "best_model.pt")
        else:
            patience_counter += 1
            
        print(f"[{config['model_name'].upper()} Seed {seed}] Epoch {epoch:02d}/{epochs:02d} - "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
              f"Val MAE: {val_metrics['vel_mae']:.3f} m/s | Time: {epoch_time:.1f}s "
              f"{'(*best*)' if is_best else ''}", flush=True)
              
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
            break
            
    total_train_time = time.time() - start_time
    
    # Load best model for test evaluation
    best_checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    
    # Test evaluation
    test_metrics = evaluate_epoch(model, test_loader, criterion, device)
    
    # Measure inference latency
    model.eval()
    sample_batch = torch.randn(1, 100, 6, device=device)
    # Warmup
    for _ in range(10):
        _ = model(sample_batch)
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model(sample_batch)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_latency_ms = float(np.mean(latencies))
    
    results = {
        "model_name": config["model_name"],
        "seed": seed,
        "parameters": param_count,
        "total_train_time_s": total_train_time,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_vel_mae": test_metrics["vel_mae"],
        "test_vel_rmse": test_metrics["vel_rmse"],
        "test_east_mae": test_metrics["v_east_mae"],
        "test_north_mae": test_metrics["v_north_mae"],
        "latency_ms": avg_latency_ms,
        "history": history,
    }
    
    with open(run_dir / "run_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return results


def main():
    parser = argparse.ArgumentParser(description="Train IMU navigation models.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed (runs all seeds if omitted)")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-epoch smoke test")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    output_dir = Path(args.output_dir)
    seeds = [args.seed] if args.seed is not None else config["training"].get("seeds", [42])
    
    all_runs = []
    for s in seeds:
        print(f"\n==========================================", flush=True)
        print(f"Starting Training: {config['model_name']} | Seed: {s}", flush=True)
        print(f"==========================================", flush=True)
        res = train_single_run(config, s, output_dir, smoke_test=args.smoke_test)
        all_runs.append(res)
        
    print(f"\nCompleted {len(all_runs)} runs for {config['model_name']}.", flush=True)


if __name__ == "__main__":
    main()
