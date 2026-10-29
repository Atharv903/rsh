import os
import json
import torch
from pathlib import Path

ckpt_dir = Path("experiments/phase8_corrected_speed_models/checkpoints")
for d in sorted(ckpt_dir.iterdir()):
    if d.is_dir():
        pt_path = d / "best_model.pt"
        if pt_path.exists():
            ckpt = torch.load(pt_path, map_location="cpu")
            cfg = ckpt.get("config", {})
            model_type = cfg.get("model_type")
            seed = ckpt.get("seed")
            best_epoch = ckpt.get("best_epoch")
            val_mae = ckpt.get("val_mae")
            val_rmse = ckpt.get("val_rmse")
            dev = ckpt.get("device")
            state_dict = ckpt.get("model_state_dict", {})
            n_params = sum(p.numel() for p in state_dict.values())
            print(f"[{d.name}] Arch={model_type:15s} | Seed={seed} | BestEpoch={best_epoch:2d} | ValMAE={val_mae:.4f} | ValRMSE={val_rmse:.4f} | Device={dev} | Params={n_params:,d}")
