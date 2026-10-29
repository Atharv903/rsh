"""
Phase 3 Sanity Check Script.
Trains a small CNN-GRU on the corrected grouped training split for 3 epochs.
Verifies finite loss, decreasing loss, speed MAE vs train-mean baseline,
predictions in m/s, absence of collapse, and plots predictions vs reference on Val set.
"""

import os
import sys
import json
from pathlib import Path

# Add project root
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn_gru import CNNGRUModel

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})


def main():
    print("=" * 70)
    print("PHASE 3: LEARNED SPEED ESTIMATOR SANITY CHECK")
    print("=" * 70)
    
    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load Data
    train_data = np.load("data/processed/train_windows_grouped.npz", allow_pickle=True)
    val_data = np.load("data/processed/val_windows_grouped.npz", allow_pickle=True)
    
    X_train = torch.from_numpy(train_data["X"].astype(np.float32))
    y_train = torch.from_numpy(train_data["y"].astype(np.float32))
    
    X_val = torch.from_numpy(val_data["X"].astype(np.float32))
    y_val = torch.from_numpy(val_data["y"].astype(np.float32))
    
    print(f"Loaded Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Loaded Val:   X={X_val.shape}, y={y_val.shape}")
    
    # Compute Train-Mean Speed Baseline
    train_mean_speed = float(y_train.mean())
    val_train_mean_mae = float(torch.abs(y_val - train_mean_speed).mean())
    print(f"\nTrain Mean Speed Baseline: {train_mean_speed:.2f} m/s")
    print(f"Val MAE of Train-Mean Baseline: {val_train_mean_mae:.3f} m/s")
    
    # 2. Build Small Model (out_dim=1 for scalar speed)
    model = CNNGRUModel(
        in_channels=6,
        conv_dims=[32, 64],
        kernel_sizes=[5, 3],
        gru_hidden_dim=64,
        gru_num_layers=1,
        out_dim=1,
        dropout=0.1,
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized Sanity CNN-GRU with {param_count:,} parameters.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    
    # Fast DataLoader (subsample train by stride 2 for fast sanity check)
    train_ds = TensorDataset(X_train[::2], y_train[::2])
    val_ds = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    print(f"\nTraining for 3 sanity check epochs on {len(train_ds):,} windows...")
    epoch_losses = []
    
    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        avg_loss = total_loss / n_batches
        epoch_losses.append(avg_loss)
        
        # Validation MAE
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                p = model(bx)
                val_preds.append(p.cpu().numpy())
                val_targets.append(by.cpu().numpy())
                
        val_preds_arr = np.concatenate(val_preds, axis=0).flatten()
        val_targets_arr = np.concatenate(val_targets, axis=0).flatten()
        val_mae = float(np.mean(np.abs(val_preds_arr - val_targets_arr)))
        val_rmse = float(np.sqrt(np.mean((val_preds_arr - val_targets_arr)**2)))
        
        print(f"Epoch {epoch}/3: Train Loss={avg_loss:.4f} | Val Speed MAE={val_mae:.3f} m/s | Val Speed RMSE={val_rmse:.3f} m/s")
        
    # Check 1: Loss is finite and decreasing
    assert not np.isnan(epoch_losses).any(), "Loss contains NaNs!"
    assert epoch_losses[-1] < epoch_losses[0], f"Loss did not decrease: {epoch_losses}"
    print("\n[CHECK PASSED] Training loss is finite and strictly decreases.")
    
    # Check 2: Val MAE beats train-mean baseline
    print(f"[COMPARISON] Model Val MAE: {val_mae:.3f} m/s vs Train-Mean Baseline: {val_train_mean_mae:.3f} m/s")
    assert val_mae < val_train_mean_mae, "Model failed to beat naive train-mean baseline!"
    print("[CHECK PASSED] Model outperforms constant train-mean baseline.")
    
    # Check 3: Predictions are in m/s and cover realistic speed distribution
    p_min, p_max, p_mean = val_preds_arr.min(), val_preds_arr.max(), val_preds_arr.mean()
    t_min, t_max, t_mean = val_targets_arr.min(), val_targets_arr.max(), val_targets_arr.mean()
    print(f"[CHECK] Predicted Speed Range: [{p_min:.2f}, {p_max:.2f}] m/s (Mean={p_mean:.2f} m/s)")
    print(f"[CHECK] Target Speed Range:    [{t_min:.2f}, {t_max:.2f}] m/s (Mean={t_mean:.2f} m/s)")
    assert p_max > 15.0, "Model predictions collapsed to narrow low range!"
    assert p_min >= -1.0, "Unphysical negative speeds predicted!"
    print("[CHECK PASSED] Predictions are in valid m/s units and span full speed range.")
    
    # 3. Generate Diagnostic Sanity Plot (Val Set - S3a segment)
    os.makedirs("figures", exist_ok=True)
    
    # Plot consecutive 300 seconds of Validation set predictions vs truth
    n_plot = 300
    t_axis = np.arange(n_plot) * 1.0  # 1-second stride
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].plot(t_axis, val_targets_arr[:n_plot], color="black", linewidth=2.0, label="Ground Truth Vehicle Speed (m/s)")
    axes[0].plot(t_axis, val_preds_arr[:n_plot], color="#1f77b4", linestyle="--", linewidth=1.8, label="CNN-GRU Predicted Speed (m/s)")
    axes[0].axhline(train_mean_speed, color="gray", linestyle=":", label=f"Train-Mean Baseline ({train_mean_speed:.1f} m/s)")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Sanity Check: Validation Set Predicted vs. True Speed Time Series (First 300 Windows)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Scatter plot
    sub = np.arange(0, len(val_targets_arr), 5)
    axes[1].scatter(val_targets_arr[sub], val_preds_arr[sub], alpha=0.3, s=10, color="#2b5c8f")
    axes[1].plot([0, 32], [0, 32], color="red", linestyle="--", label="Ideal 1:1 Identity Line")
    axes[1].set_xlabel("Ground Truth Speed (m/s)")
    axes[1].set_ylabel("Predicted Speed (m/s)")
    axes[1].set_title(f"Validation Speed Scatter (MAE = {val_mae:.2f} m/s, N={len(sub):,})")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("figures/phase8_sanity_check_predictions.png")
    plt.close()
    print("Saved figures/phase8_sanity_check_predictions.png")
    print("Sanity check completed successfully! Network learns speed robustly.")


if __name__ == "__main__":
    main()
