"""
Plot publication-quality training loss curves and validation MAE curves across all 9 models.
"""

import sys
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

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


def plot_curves(ckpt_dir_override=None):
    candidates = [
        Path(ckpt_dir_override) if ckpt_dir_override else None,
        Path("/content/drive/MyDrive/GPS_Denied_Navigation_Research/checkpoints"),
        Path("experiments/phase8_corrected_speed_models/checkpoints"),
        Path("/content/checkpoints"),
    ]
    ckpt_dir = None
    for cand in candidates:
        if cand is not None and cand.exists():
            ckpt_dir = cand
            break
            
    if ckpt_dir is None:
        print("[Warning] Checkpoints directory not found. Skipping loss curve plot.")
        return
        
    run_dirs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir()])
    
    if not run_dirs:
        print(f"No run directories found in {ckpt_dir}.")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        "CNNGRUModel": "#1f77b4",
        "ModernTCNModel": "#2ca02c",
        "Mamba2Model": "#d62728",
    }
    linestyles = {
        42: "-",
        123: "--",
        456: ":",
    }
    
    labels_plotted = set()
    
    for rdir in run_dirs:
        meta_file = rdir / "run_metadata.json"
        if not meta_file.exists():
            continue
            
        with open(meta_file, "r") as f:
            meta = json.load(f)
            
        history = meta.get("epoch_history", [])
        if not history:
            continue
            
        m_type = meta.get("model_type", "Unknown")
        seed = meta.get("seed", 42)
        
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_mae = [h["val_mae"] for h in history]
        
        c = colors.get(m_type, "black")
        ls = linestyles.get(seed, "-")
        
        lbl = m_type if m_type not in labels_plotted else None
        if lbl:
            labels_plotted.add(m_type)
            
        ax1.plot(epochs, train_loss, color=c, linestyle=ls, alpha=0.85, label=lbl)
        ax2.plot(epochs, val_mae, color=c, linestyle=ls, alpha=0.85, label=lbl)
        
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Smooth L1 Loss")
    ax1.set_title("Training Loss Convergence")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Speed MAE (m/s)")
    ax2.set_title("Validation Speed Estimation MAE")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    out_path = Path("figures/phase8_training_loss_curves.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved training curves to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss and validation curves")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    args = parser.parse_args()
    plot_curves(args.ckpt_dir)
