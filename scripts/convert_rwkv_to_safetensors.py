#!/usr/bin/env python3
"""Convert RWKV-6 pytorch_model.bin to model.safetensors.

The RWKV/v6-Finch-1B6-HF model only ships pytorch_model.bin.
PLIP-RS requires safetensors format. Run this once before loading.

Usage:
    python scripts/convert_rwkv_to_safetensors.py [--model-id RWKV/v6-Finch-1B6-HF]

The converted model.safetensors will be placed in the HuggingFace cache
alongside the original pytorch_model.bin.
"""

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Convert RWKV pytorch_model.bin to safetensors")
    parser.add_argument(
        "--model-id",
        default="RWKV/v6-Finch-1B6-HF",
        help="HuggingFace model ID (default: RWKV/v6-Finch-1B6-HF)",
    )
    args = parser.parse_args()

    print(f"Downloading pytorch_model.bin from {args.model_id}...")
    bin_path = hf_hub_download(args.model_id, "pytorch_model.bin")
    bin_path = Path(bin_path)

    print(f"Loading weights from {bin_path}...")
    weights = torch.load(bin_path, map_location="cpu", weights_only=True)

    # Print weight names for inspection
    print(f"\nFound {len(weights)} tensors:")
    total_params = 0
    for name, tensor in sorted(weights.items()):
        print(f"  {name}: {list(tensor.shape)} ({tensor.dtype})")
        total_params += tensor.numel()
    print(f"\nTotal parameters: {total_params:,} ({total_params / 1e9:.2f}B)")

    # Save as safetensors in the same directory
    out_path = bin_path.parent / "model.safetensors"
    print(f"\nSaving to {out_path}...")
    save_file(weights, str(out_path))

    print(f"Done. Safetensors file: {out_path}")
    print(f"Size: {out_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
