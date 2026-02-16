#!/usr/bin/env python3
"""Generate CLT-426K reference encodings for Rust validation.

Loads CLT encoder weights from HuggingFace, encodes test residual
vectors, and saves reference activations for comparison with plip-rs
clt.rs.

Dependencies: torch, safetensors, huggingface_hub

Usage:
    python scripts/clt_reference.py

Output:
    scripts/clt_reference_426k.json
"""

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

CLT_REPO = "mntss/clt-gemma-2-2b-426k"
TEST_LAYERS = [0, 12, 25]
N_SEEDS_PER_LAYER = 3
TOP_K = 10


def main():
    print(f"CLT reference generation for {CLT_REPO}")
    print(f"Test layers: {TEST_LAYERS}, seeds per layer: {N_SEEDS_PER_LAYER}")
    print()

    results = {
        "clt_repo": CLT_REPO,
        "d_model": None,
        "n_features_per_layer": None,
        "test_cases": [],
    }

    for layer in TEST_LAYERS:
        # Download encoder weights (cached by huggingface_hub)
        enc_path = hf_hub_download(CLT_REPO, f"W_enc_{layer}.safetensors")
        weights = load_file(enc_path)

        w_enc = weights[f"W_enc_{layer}"].float()  # [n_features, d_model]
        b_enc = weights[f"b_enc_{layer}"].float()  # [n_features]

        n_features, d_model = w_enc.shape
        print(f"Layer {layer}: W_enc [{n_features}, {d_model}], b_enc [{b_enc.shape[0]}]")

        # Record dimensions (from first layer)
        if results["d_model"] is None:
            results["d_model"] = d_model
            results["n_features_per_layer"] = n_features

        for seed_idx in range(N_SEEDS_PER_LAYER):
            seed = seed_idx * 100 + layer
            torch.manual_seed(seed)
            residual = torch.randn(d_model)

            # Encode: pre_acts = W_enc @ residual + b_enc, acts = ReLU(pre_acts)
            pre_acts = w_enc @ residual + b_enc
            acts = torch.relu(pre_acts)

            n_active = int((acts > 0).sum())
            top_vals, top_idx = acts.topk(min(TOP_K, n_active))

            test_case = {
                "layer": layer,
                "seed": seed,
                "residual": residual.tolist(),
                "n_active": n_active,
                "top_10": [
                    {"index": int(idx), "activation": float(val)}
                    for idx, val in zip(top_idx, top_vals)
                ],
            }
            results["test_cases"].append(test_case)

            top_feat = f"L{layer}:{int(top_idx[0])}" if len(top_idx) > 0 else "none"
            top_act = f"{float(top_vals[0]):.4f}" if len(top_vals) > 0 else "N/A"
            print(
                f"  seed={seed:3d}: {n_active:5d} active features, "
                f"top={top_feat} ({top_act})"
            )

    # Save reference data
    out_path = Path(__file__).parent / "clt_reference_426k.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_cases = len(results["test_cases"])
    file_size = out_path.stat().st_size
    print(f"\nSaved {n_cases} test cases to {out_path} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
