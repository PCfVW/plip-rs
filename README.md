# PLIP-rs: Programming Language Internal Probing in Rust

[![CI](https://github.com/PCfVW/plip-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/PCfVW/plip-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.92+-DEA584.svg?logo=rust)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.1-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Models](https://img.shields.io/badge/Models-6-8b5cf6.svg)](README.md#supported-models)
[![Live Demo](https://img.shields.io/badge/Demo-Deloson-06b6d4.svg)](https://PCfVW.github.io/deloson/)

**PLIP** investigates how transformer models internally process test-related syntax, measuring attention patterns from test markers (Python `>>>`, Rust `#[test]`) to function tokens. Supplementary material for AIware 2026, developed as part of the [d-Heap Priority Queue](https://github.com/PCfVW/d-Heap-priority-queue) research project.

**Key Finding:** Python doctest markers show **2.8-4.4× stronger attention** to function tokens than Rust test attributes, with **p < 0.0002** in 4 of 6 tested architectures. Two models (Phi-3-mini, Code-LLaMA) show near-symmetric or reversed patterns, revealing architecture-dependent attention behavior.

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Hardware Requirements](#hardware-requirements)
- [Usage](#usage)
- [Universal Corpus Format](#universal-corpus-format)
- [Connection to AIware 2026](#connection-to-aiware-2026)
- [Development](#development)
- [Supported Models](#supported-models)
- [MI for the Rest of Us](#mi-for-the-rest-of-us)
- [License](#license)
- [Citation](#citation)

## Quick Start

### Universal Layer Scan (Recommended)

The **universal corpus format** works with ANY model without preprocessing:

```bash
# Prerequisites: Rust 1.92+, CUDA 13.1 (or --cpu for CPU mode)
cargo build --release

# Scan attention patterns for any model
cargo run --release --example layer_scan_universal -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct"

cargo run --release --example layer_scan_universal -- \
    --model "bigcode/starcoder2-3b"

cargo run --release --example layer_scan_universal -- \
    --model "google/codegemma-7b-it"
```

### CPU Mode

```bash
cargo build --release --no-default-features
cargo run --release --no-default-features --example layer_scan_universal -- --cpu
```

> **Note:** CPU mode is intended for CI and compilation checks, not for running experiments. A full layer scan on a 7B model takes minutes on GPU but can take hours on CPU. All examples default to CUDA and require a GPU with sufficient VRAM (see [Hardware Requirements](#hardware-requirements)). CPU mode also requires enough system RAM to hold the model weights (~6 GB for 3B models, ~14 GB for 7B models).

See [COMMANDS.md](COMMANDS.md) for the full list of examples (ablation, steering, generation, debug tools, and more).

## Project Structure

```
plip-rs/
├── src/
│   ├── lib.rs                  # Public API re-exports
│   ├── main.rs                 # CLI entrypoint
│   ├── model.rs                # PlipModel wrapper (multi-architecture)
│   ├── forward.rs              # StarCoder2 forward pass with activation capture
│   ├── forward_qwen2.rs        # Qwen2 forward pass with activation capture
│   ├── forward_gemma.rs        # Gemma forward pass with activation capture
│   ├── forward_llama.rs        # LLaMA forward pass with activation capture
│   ├── forward_phi3.rs         # Phi-3 forward pass with activation capture
│   ├── kv_cache.rs             # KV-cache for efficient autoregressive generation
│   ├── masks.rs                # Shared attention mask utilities (cached)
│   ├── positioning.rs          # Character → token position conversion
│   ├── cache.rs                # ActivationCache struct
│   ├── attention.rs            # Attention pattern capture and analysis
│   ├── intervention.rs         # Attention intervention (knockout, steering)
│   ├── steering.rs             # Steering calibration and dose-response
│   ├── logit_lens.rs           # Logit Lens for interpretability
│   ├── corpus.rs               # JSON corpus loading
│   ├── experiment.rs           # PLIP experiment runner
│   └── probe.rs                # Linear probing with linfa
├── corpus/
│   ├── attention_samples_universal.json  # Universal corpus (character positions)
│   ├── attention_samples.json            # Legacy corpus (token positions)
│   └── README.md                         # Corpus format documentation
├── examples/
│   ├── layer_scan_universal.rs # Scan layers with universal corpus (recommended)
│   ├── convert_corpus.rs       # Convert legacy to universal format
│   ├── verify_positions_universal.rs  # Verify position conversion
│   ├── logit_lens.rs           # Layer-by-layer prediction analysis
│   ├── attention_patterns.rs   # Attention weight extraction
│   ├── statistical_attention.rs # Statistical significance testing
│   └── ...                     # See COMMANDS.md for full list
├── outputs/                    # Generated results
├── docs/
│   ├── experiments/            # Ablation, steering, N=50 results
│   ├── roadmaps/              # Planning documents
│   ├── TEST_CHECKLIST.md
│   └── RIKEN_INSTRUCTIONS.md
├── Cargo.toml
├── CHANGELOG.md               # Release history
├── COMMANDS.md                 # Full command reference
└── README.md
```

## Hardware Requirements

| Model | VRAM Required | Tested On |
|-------|---------------|-----------|
| StarCoder2-3B | ~6 GB | RTX 5060 Ti (16GB) |
| Qwen2.5-Coder-3B | ~6 GB | RTX 5060 Ti (16GB) |
| Phi-3-mini-4k | ~8 GB | RTX 5060 Ti (16GB) |
| Code-LLaMA-7B | ~13 GB | RTX 5060 Ti (16GB) |
| Qwen2.5-Coder-7B | ~14 GB | RTX 5060 Ti (16GB) |
| CodeGemma-7B | ~14 GB | RTX 5060 Ti (16GB) |

> **Important:** The RTX 5060 Ti comes in 8GB and 16GB variants. The **16GB model is required** — the 7B-parameter models need ~14 GB VRAM for attention extraction, which exceeds the 8GB variant's capacity.

### Continuous Integration

The [CI workflow](https://github.com/PCfVW/plip-rs/actions/workflows/ci.yml) runs on every push: `cargo check`, `cargo test`, `cargo fmt`, and `cargo clippy` — all in CPU mode (`--no-default-features`). CUDA-dependent functionality (model loading, attention extraction, steering) is tested locally on RTX 5060 Ti 16GB before each release.

## Usage

### Attention Analysis (Primary Use Case)

```bash
# Scan layers to find optimal attention patterns
cargo run --release --example layer_scan_universal -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --output outputs/qwen7b_scan.json
```

### Sample Output

```
═══════════════════════════════════════════════════════════════════
  Universal Layer Scan - Model-Agnostic Attention Analysis
═══════════════════════════════════════════════════════════════════

Loading universal corpus from: "corpus/attention_samples_universal.json"
  Format version: 2.0
  Python doctest samples: 10
  Rust test samples:      10

Loading model: Qwen/Qwen2.5-Coder-7B-Instruct...
Model loaded: 28 layers

Converting character positions to token positions...
  Total samples: 20
  Successful conversions: 20
  Failed conversions: 0

┌───────┬────────────┬────────────┬─────────┬──────────┬──────────┬──────────┐
│ Layer │ Python μ   │ Rust μ     │  Ratio  │ t-stat   │ df       │ p-value  │
├───────┼────────────┼────────────┼─────────┼──────────┼──────────┼──────────┤
│    16 │      9.08% │      2.59% │   3.51× │    8.88 │    10.6 │  0.0000 *** │
│    17 │      8.89% │      2.53% │   3.51× │    8.39 │    10.4 │  0.0000 *** │
...
└───────┴────────────┴────────────┴─────────┴──────────┴──────────┴──────────┘

═══════════════════════════════════════════════════════════════════
  Best Layer: 16
═══════════════════════════════════════════════════════════════════
  Python >>> → params:  9.08% ± 2.21%  (n=10)
  Rust #[ → fn tokens: 2.59% ± 0.67%  (n=10)
  Ratio: 3.51×
  p-value: 0.000003 ✓ SIGNIFICANT
```

### Output Files

```
outputs/
├── layer_scan_universal_starcoder2.json   # Layer-by-layer statistics
├── layer_scan_universal_qwen3b.json
├── layer_scan_universal_qwen7b.json
├── layer_scan_universal_codegemma.json
├── layer_scan_universal_codellama.json
└── layer_scan_universal_phi3.json
```

## Universal Corpus Format

PLIP-rs uses a **model-agnostic corpus format** with character positions instead of token indices — because each tokenizer maps the same source code to different token sequences, making token-level annotations model-specific and fragile:

```json
{
  "_format_version": "2.0",
  "python_doctest": [
    {
      "id": "py_simple_add",
      "code": "def add(a, b):\n    \"\"\"\n    >>> add(2, 3)\n    5\n    \"\"\"\n    return a + b",
      "marker_char_pos": 27,
      "marker_pattern": ">>>",
      "target_char_positions": [0, 4, 8, 11]
    }
  ]
}
```

**Benefits:**
- Works with ANY model without preprocessing
- 100% position accuracy (no tokenizer mismatches)
- Single corpus file for all experiments

## Connection to AIware 2026

This tool supports the AIware 2026 submission on attention patterns in code LLMs:

1. **Finding**: Python inline doctests (`>>>`) show 2.8-4.4× stronger attention to function tokens than Rust `#[test]` attributes in 4 of 6 architectures (p < 0.0002). Two models (Phi-3-mini, Code-LLaMA) show near-symmetric or reversed patterns.
2. **Method**: Attention weight extraction at each layer with Welch's t-test for statistical significance across 6 models (5 architectures)
3. **Implication**: The Python attention advantage is architecture-dependent, suggesting test syntax processing varies with model design choices

See [RIGOR_EXPERIMENT.md](docs/experiments/RIGOR_EXPERIMENT.md) for full methodology and results.

**Visualization:** Layer scan results can be explored interactively with [Deloson](https://github.com/PCfVW/deloson), a companion web app. Try the [live demo](https://PCfVW.github.io/deloson/).

## Development

```bash
# Run tests
cargo test

# Run GPU tests (requires CUDA + downloaded models)
# --test-threads=1 prevents parallel GPU contention (OOM with concurrent model loads)
cargo test -- --ignored --test-threads=1

# Run with logging
RUST_LOG=debug cargo run --release --example layer_scan_universal

# Format
cargo fmt

# Lint
cargo clippy

# See all available commands
cat COMMANDS.md
```

## Supported Models

| Model | HuggingFace ID | Architecture |
|-------|----------------|--------------|
| StarCoder2 3B | `bigcode/starcoder2-3b` | StarCoder2 |
| Qwen2.5-Coder 3B | `Qwen/Qwen2.5-Coder-3B-Instruct` | Qwen2 |
| Phi-3-mini-4k | `microsoft/Phi-3-mini-4k-instruct` | Phi3 |
| Code-LLaMA 7B | `codellama/CodeLlama-7b-hf` | LLaMA |
| Qwen2.5-Coder 7B | `Qwen/Qwen2.5-Coder-7B-Instruct` | Qwen2 |
| CodeGemma 7B | `google/codegemma-7b-it` | Gemma |

**Why these models?** Mechanistic interpretability requires access to model internals (attention weights) that proprietary models (Claude, GPT-4) do not expose. Selection was constrained to open-source code LLMs that: (1) fit within 16GB VRAM, (2) are compatible with [candle](https://github.com/huggingface/candle) (Rust ML framework), and (3) demonstrate both Python and Rust code generation capability.

## MI for the Rest of Us

PLIP-rs demonstrates that meaningful mechanistic interpretability research is possible with **consumer hardware**. This wasn't easy—running 7B parameter models with full attention extraction on 16GB VRAM required:

- **KV-cache with hybrid steering**: Cache K,V tensors during prompt processing, then generate efficiently with full steering compatibility. Enables steering experiments without full sequence recomputation.
- **Shared mask caching**: Attention masks (16MB+ for seq_len=2048) are cached by `(seq_len, device, dtype)` and reused across all model backends, avoiding repeated allocations.
- **Memory-limited generation**: Automatic cache trimming to 75% when memory limits are exceeded, enabling long-context generation within VRAM constraints.
- **Rust/candle** instead of Python/PyTorch: no garbage collector means deterministic deallocation of tensors, giving precise control over peak VRAM usage.
- **Model-agnostic corpus format** to avoid redundant preprocessing per model.

The result: statistically significant findings (p < 0.0002) in 4 of 6 models, plus revealing architecture-dependent variation in the remaining 2, on hardware that costs ~$500, not $50,000.

**Why this matters:**
- Democratizes MI research beyond well-funded labs
- Proves consumer GPUs are viable for attention analysis
- Open-source tooling (candle + PLIP-rs) enables reproducibility
- Lowers the barrier for researchers to investigate model internals

If you're doing MI research on limited hardware, we hope PLIP-rs helps. PRs welcome for further memory optimizations.

## License

Apache 2.0

## Citation

```bibtex
@software{plip_rs,
  title = {PLIP-rs: Programming Language Internal Probing in Rust},
  author = {Jacopin, Eric and Claude},
  year = {2026},
  note = {Attention analysis for AIware 2026},
  url = {https://github.com/PCfVW/plip-rs}
}
```
