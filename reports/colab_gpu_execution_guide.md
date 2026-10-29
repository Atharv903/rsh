# Google Colab GPU Execution & Migration Guide: GPS-Denied Vehicle Dead Reckoning

**Project:** GPS-Denied Vehicle Navigation on IO-VNBD  
**Date:** October 2026  
**Status:** Validated & Ready for GPU Execution  
**Notebook Deliverable:** [`notebooks/colab_gpu_benchmark.ipynb`](file:///c:/Users/shreya%20parmar/Downloads/research/notebooks/colab_gpu_benchmark.ipynb)  
**Orchestration Script:** [`scripts/colab_runner.py`](file:///c:/Users/shreya%20parmar/Downloads/research/scripts/colab_runner.py)  
**Test Suite Status:** 25 / 25 unit tests passing (`pytest -v`)  

---

## 1. Architecture & Design Principles for Colab Execution

Migrating research code from local development to Google Colab GPU environments requires addressing several cloud-specific failure modes without compromising scientific reproducibility or experimental methodology:

1. **Device Abstraction:** All model instantiations, batch tensors, and loss computations are fully device-aware. The pipeline automatically queries `torch.cuda.is_available()`, routing tensors to CUDA with pinned memory buffers on GPU runtimes, while maintaining identical CPU execution fallback.
2. **Storage Hierarchy & FUSE Bottleneck Prevention:** Reading datasets directly from Google Drive (`/content/drive/MyDrive/...`) introduces high-latency FUSE filesystem overhead during iterative DataLoader slicing. To maximize throughput, the pipeline incorporates a local caching routine that transfers processed `.npz` files and normalization parameters to the Colab VM's local high-speed NVMe storage (`/content/data/processed/`).
3. **Persistent Checkpointing & Interruption Recovery:** Free and Pro Colab sessions are subject to disconnections and kernel preemption. The training loop saves `last_checkpoint.pt` to persistent Google Drive after every single epoch. If an interruption occurs, rerunning the cell automatically loads model parameters, optimizer momentum, scheduler states, and epoch logs, resuming training from the exact interrupted epoch.
4. **Separation of Profiling:** Training throughput (samples/sec, epoch duration, peak GPU VRAM) and trajectory evaluation latency (ms/step, total outage simulation runtime) are profiled and logged into distinct JSON metadata files.

---

## 2. Validation & Behavioral Consistency Verification

Prior to full benchmark execution on Colab, the codebase underwent rigorous automated verification:

### Test Suite Execution
- **Total Unit Tests:** 25 passed in 117.05s.
- **Colab Compatibility Tests:**
  - `test_device_detection`: Verifies correct runtime typing and CUDA/CPU auto-detection.
  - `test_models_device_transfer`: Confirms tensor shapes `(B, 1)`, device alignment, and zero NaN/Inf outputs across CNN-GRU, ModernTCN, and Mamba-2.
  - `test_checkpoint_saving_and_resumption`: Verifies that training for 1 epoch, saving `last_checkpoint.pt`, and resuming for epoch 2 restores optimizer momentum and completes training without divergence.
  - `test_local_storage_cache_helper`: Proves reliable transfer between persistent and local runtime directories.

### Numerical Consistency Check
Executed via [`scripts/verify_numerical_consistency.py`](file:///c:/Users/shreya%20parmar/Downloads/research/scripts/verify_numerical_consistency.py):
- **CPU Determinism:** Max absolute difference between repeated runs with fixed seeds is exactly `0.0e+00`.
- **Checkpoint Fidelity:** Resumed training restores exact loss trajectory and validation metrics.
- **Device Parity:** When executed on CUDA, CPU vs. GPU forward outputs must match within numerical floating-point precision ($< 10^{-4}$).

---

## 3. Google Colab Notebook Structure (`colab_gpu_benchmark.ipynb`)

The notebook is divided into 10 structured, interactive cells:

| Cell # | Operation | Target Execution & Purpose |
| :---: | :--- | :--- |
| **1** | **Hardware Diagnostics** | Runs `nvidia-smi` and queries `torch.cuda` for device name, memory, and compute capability. |
| **2** | **Mount Google Drive** | Mounts `/content/drive` and creates persistent output directory `/content/drive/MyDrive/GPS_Denied_Navigation_Research`. |
| **3** | **Local Cache Transfer** | Copies `train_windows_grouped.npz`, `val_windows_grouped.npz`, `test_windows_grouped.npz`, and `sensor_normalizer_grouped.json` to `/content/data/processed/`. |
| **4** | **Install Dependencies** | Installs `pyyaml`, `pandas`, `numpy`, `matplotlib`, `scipy`, and `pytest`. |
| **5** | **Run Unit Test Suite** | Executes `!pytest -v tests/` to confirm 25 passing unit tests on the Colab environment. |
| **6** | **Consistency Verification** | Runs `scripts/verify_numerical_consistency.py` to prove forward pass determinism and checkpoint state fidelity. |
| **7** | **Fast GPU Smoke Test** | Executes 1-epoch smoke tests for CNN-GRU, ModernTCN, and Mamba-2 on CUDA, asserting finite loss and gradient flow. |
| **8** | **Full 9-Model Benchmark** | Trains all 3 architectures $\times$ 3 seeds (42, 123, 456) with Drive checkpointing after each epoch. |
| **9** | **Outage Trajectory Benchmark** | Evaluates test speed metrics and runs causal dead reckoning across 10s, 30s, and 60s GNSS outages on Drive S4. |
| **10** | **Figures & Metric Summary** | Generates training loss curves, trajectory comparisons, and displays structured summary tables. |

---

## 4. Local CPU vs. Target GPU Profiling Template

When executed on Google Colab, populate the empirical GPU measurements in the template below to compare against our local baseline:

### Table 1: Hardware Environment Comparison

| Attribute | Local Baseline Machine | Google Colab Target Environment |
| :--- | :---: | :---: |
| **Platform / OS** | Windows 10 x86_64 | Linux (Ubuntu 22.04 LTS) |
| **Compute Device** | Intel Core CPU (Multi-core) | NVIDIA GPU (e.g., T4, V100, A100, or L4) |
| **CUDA Available** | False (`torch.device("cpu")`) | True (`torch.device("cuda:0")`) |
| **Total Memory** | 16 GB System RAM | 12.0 – 40.0 GB High-Bandwidth VRAM |
| **PyTorch Version** | 2.14.1+cpu | 2.1+ / 2.2+ with CUDA 12.x |

### Table 2: Training Throughput & Resource Profiling (Measured vs. Target)

| Model Architecture | Local CPU Epoch Time | Local CPU Throughput | GPU Epoch Time (Target) | GPU Throughput (Target) | GPU Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN-GRU** | 35.5 s | ~205 samples/sec | *To be measured on Colab* | *To be measured on Colab* | *~150 MB* |
| **ModernTCN** | 41.2 s | ~176 samples/sec | *To be measured on Colab* | *To be measured on Colab* | *~280 MB* |
| **Mamba-2 (custom SSD)** | 31.8 s | ~228 samples/sec | *To be measured on Colab* | *To be measured on Colab* | *~190 MB* |

> [!IMPORTANT]
> **GPU Speedup Reporting Policy:** As required by scientific reproducibility guidelines, GPU speedup ratios must NOT be claimed theoretically; they will be recorded directly from Colab runtime execution logs via `epoch_train_time_s` and `throughput_samples_per_s` saved in `run_metadata.json`.

---

## 5. Command-Line Reproduction Guide

To run directly via terminal or headless bash scripts in Colab:

```bash
# 1. Run consistency & checkpoint verification
python scripts/verify_numerical_consistency.py

# 2. Execute GPU smoke test
python scripts/colab_runner.py --mode smoke_test --device auto

# 3. Execute full 9-model training benchmark
python scripts/run_phase8_training.py \
    --device auto \
    --data_dir /content/data/processed \
    --output_dir /content/drive/MyDrive/GPS_Denied_Navigation_Research/checkpoints \
    --seeds 42 123 456

# 4. Execute outage trajectory evaluation & profiling
python scripts/evaluate_speed_models.py \
    --device auto \
    --data_dir /content/data/processed \
    --ckpt_dir /content/drive/MyDrive/GPS_Denied_Navigation_Research/checkpoints \
    --results_dir /content/drive/MyDrive/GPS_Denied_Navigation_Research/results
```
