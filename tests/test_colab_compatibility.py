"""
Unit Tests for Colab GPU Migration, Device Awareness, and Checkpoint Resumption.
"""

import os
import tempfile
from pathlib import Path
import yaml
import torch
import pytest
import numpy as np

from scripts.train_speed_models import (
    get_default_device,
    build_model,
    set_seed,
    train_single_model,
    benchmark_inference_latency,
)
from scripts.colab_runner import setup_local_storage


def test_device_detection():
    dev = get_default_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ["cuda", "cpu"]
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    else:
        assert dev.type == "cpu"


def test_models_device_transfer():
    dev = get_default_device()
    configs = [
        ("CNN-GRU", "experiments/phase8_corrected_speed_models/configs/cnn_gru.yaml"),
        ("ModernTCN", "experiments/phase8_corrected_speed_models/configs/moderntcn.yaml"),
        ("Mamba-2 SSD", "experiments/phase8_corrected_speed_models/configs/mamba2.yaml"),
    ]
    
    for name, cfg_path in configs:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        model = build_model(cfg).to(dev)
        x = torch.randn(2, 100, 6, device=dev)
        with torch.no_grad():
            out = model(x)
            
        assert out.shape == (2, 1)
        assert out.device.type == dev.type
        assert torch.all(torch.isfinite(out))


def test_checkpoint_saving_and_resumption():
    dev = get_default_device()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg_path = "experiments/phase8_corrected_speed_models/configs/cnn_gru.yaml"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        test_cfg = dict(cfg)
        test_cfg["training"]["epochs"] = 1
        test_cfg["training"]["patience"] = 2
        
        cfg_file = tmp_path / "test_cfg.yaml"
        with open(cfg_file, "w") as f:
            yaml.dump(test_cfg, f)
            
        ckpt_dir = tmp_path / "ckpts"
        
        # 1. Train 1 epoch
        res1 = train_single_model(
            str(cfg_file),
            seed=42,
            output_dir=ckpt_dir,
            device=dev,
            resume=False,
        )
        
        run_folder = ckpt_dir / "cnngrumodel_seed42"
        last_ckpt = run_folder / "last_checkpoint.pt"
        best_ckpt = run_folder / "best_model.pt"
        
        assert last_ckpt.exists(), "last_checkpoint.pt must be created at every epoch!"
        assert best_ckpt.exists(), "best_model.pt must be created when new best achieved!"
        
        # 2. Resume with epoch count increased to 2
        test_cfg["training"]["epochs"] = 2
        with open(cfg_file, "w") as f:
            yaml.dump(test_cfg, f)
            
        res2 = train_single_model(
            str(cfg_file),
            seed=42,
            output_dir=ckpt_dir,
            device=dev,
            resume=True,
        )
        
        assert res2["best_epoch"] >= 1
        assert len(res2["epoch_history"]) == 2


def test_local_storage_cache_helper():
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        
        # Create mock dataset files
        for fname in ["train_windows_grouped.npz", "val_windows_grouped.npz", "test_windows_grouped.npz", "sensor_normalizer_grouped.json"]:
            with open(src_path / fname, "w") as f:
                f.write("mock_content")
                
        cached = setup_local_storage(src_path, dst_path)
        assert cached == dst_path
        for fname in ["train_windows_grouped.npz", "val_windows_grouped.npz", "test_windows_grouped.npz", "sensor_normalizer_grouped.json"]:
            assert (dst_path / fname).exists()
