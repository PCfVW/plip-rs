# PLIP-rs: Instructions for RIKEN H100

## Overview

This document explains how to run the PLIP (Programming Language Internal Probing) experiment on the RIKEN H100 GPU using pre-built containers from GitHub Container Registry.

**Contact**: Eric Jacopin (Rennes, Brittany)
**Hardware**: NVIDIA H100 80GB
**Model**: CodeLlama-34B-Instruct (or StarCoder2-3B for comparison)

---

## Prerequisites

1. Docker with NVIDIA GPU support (`nvidia-docker2` or `nvidia-container-toolkit`)
2. Access to GitHub Container Registry (public, no authentication needed)
3. HuggingFace token (optional, for gated models)

Verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

---

## Quick Start

### 1. Pull the Pre-built Container

```bash
# Pull H100-optimized container from GitHub
docker pull ghcr.io/pcfvw/plip-rs:h100

# Verify
docker images | grep plip-rs
```

### 2. Prepare Directories

```bash
mkdir -p ~/plip-experiment/outputs
mkdir -p ~/plip-experiment/hf_cache
```

### 3. Run the Experiment

**With StarCoder2-3B (comparison with Rennes results):**
```bash
docker run --gpus all \
    -v ~/plip-experiment/outputs:/data/outputs \
    -v ~/plip-experiment/hf_cache:/data/hf_cache \
    ghcr.io/pcfvw/plip-rs:h100 \
    --model bigcode/starcoder2-3b \
    --output /data/outputs/starcoder2-3b
```

**With CodeLlama-34B (primary H100 experiment):**
```bash
docker run --gpus all \
    -v ~/plip-experiment/outputs:/data/outputs \
    -v ~/plip-experiment/hf_cache:/data/hf_cache \
    -e HF_TOKEN=${HF_TOKEN} \
    ghcr.io/pcfvw/plip-rs:h100 \
    --model codellama/CodeLlama-34b-Instruct-hf \
    --output /data/outputs/codellama-34b
```

### 4. Monitor Progress

```bash
# In another terminal
docker logs -f $(docker ps -q --filter ancestor=ghcr.io/pcfvw/plip-rs:h100)
```

---

## Expected Output

### Console Output
```
=== PLIP-rs: Programming Language Internal Probing ===
Model: codellama/CodeLlama-34b-Instruct-hf
Target: h100
GPU: NVIDIA H100 80GB HBM3
Corpus: 50 Python, 50 Rust samples

Processing Python samples... [====================] 50/50
Processing Rust samples...   [====================] 50/50

Layer  0:  51.25%
Layer  1:  56.25%
Layer  2:  63.75%
...
Layer 47:  96.25%

=== Results ===
Peak accuracy: 96.25% at layer 42
Results saved to /data/outputs/codellama-34b/
```

### Output Files
```
~/plip-experiment/outputs/codellama-34b/
├── plip_results.json       # Accuracy per layer
├── metadata.json           # Model, GPU, timestamp
└── activations/            # Optional: raw activations (large!)
    ├── layer_00.npy
    └── ...
```

---

## Return Results to Rennes

Please send:

1. **Results archive**:
   ```bash
   cd ~/plip-experiment/outputs
   tar -czf plip_results_riken.tar.gz codellama-34b/ starcoder2-3b/
   ```

2. **GPU verification**:
   ```bash
   nvidia-smi > gpu_info.txt
   ```

3. **Any error logs** if issues occurred

Transfer method: Email, shared drive, or `scp` to agreed location.

---

## Troubleshooting

### Container won't start
```bash
# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Check container logs
docker logs $(docker ps -aq --latest)
```

### Out of Memory (OOM)
```bash
# Use smaller model
docker run --gpus all \
    ... \
    --model bigcode/starcoder2-3b  # 6GB instead of 68GB
```

### Model download fails
```bash
# Set HuggingFace token for gated models
export HF_TOKEN=your_token_here
docker run --gpus all -e HF_TOKEN=${HF_TOKEN} ...
```

### Permission denied on output directory
```bash
# Fix permissions
sudo chown -R $(id -u):$(id -g) ~/plip-experiment/outputs
```

---

## Models to Run

| Priority | Model | VRAM | Purpose |
|----------|-------|------|---------|
| 1 | CodeLlama-34B-Instruct | ~68 GB | Primary experiment |
| 2 | StarCoder2-3B | ~6 GB | Comparison with Rennes |
| 3 | DeepSeek-Coder-33B | ~66 GB | Alternative code model |

---

## Timeline

| Task | Duration |
|------|----------|
| Pull container | ~2 min |
| Download CodeLlama-34B (first run) | ~30 min |
| Run experiment (CodeLlama-34B) | ~2-4 hours |
| Run experiment (StarCoder2-3B) | ~30 min |
| Package results | ~5 min |

---

## Contact

For issues or questions:
- GitHub Issues: https://github.com/PCfVW/plip-rs/issues
- Email: [your email]

Thank you for running this experiment!

---

*Last updated: January 29, 2026*
