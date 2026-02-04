# PLIP-rs: Programming Language Internal Probing in Rust

[![CI](https://github.com/PCfVW/plip-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/PCfVW/plip-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**PLIP** investigates how transformer models internally process test-related syntax, measuring attention patterns from test markers (Python `>>>`, Rust `#[test]`) to function tokens. Supplementary material for AIware 2026, developed as part of the [d-Heap Priority Queue](https://github.com/PCfVW/d-Heap-priority-queue) research project.

**Key Finding:** Python doctest markers show **2.8-4.4× stronger attention** to function tokens than Rust test attributes, with **p < 0.0002** across all tested architectures.

## Quick Start

### Universal Layer Scan (Recommended)

The **universal corpus format** works with ANY model without preprocessing:

```bash
# Prerequisites: Rust 1.87+, CUDA 13.1 (or --cpu for CPU mode)
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
├── Cargo.toml
├── COMMANDS.md                 # Full command reference
└── README.md
```

## Hardware Requirements

| Model | VRAM Required | Tested On |
|-------|---------------|-----------|
| Qwen2.5-Coder-3B | ~6 GB | RTX 5060 Ti (16GB) |
| StarCoder2-3B | ~6 GB | RTX 5060 Ti (16GB) |
| Qwen2.5-Coder-7B | ~14 GB | RTX 5060 Ti (16GB) |
| CodeGemma-7B | ~14 GB | RTX 5060 Ti (16GB) |

> **Important:** The RTX 5060 Ti comes in 8GB and 16GB variants. The **16GB model is required** — the 7B-parameter models need ~14 GB VRAM for attention extraction, which exceeds the 8GB variant's capacity.

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
├── layer_scan_universal_qwen7b.json   # Layer-by-layer statistics
├── layer_scan_universal_starcoder.json
└── layer_scan_universal_codegemma.json
```

## Universal Corpus Format

PLIP-rs uses a **model-agnostic corpus format** with character positions instead of token indices:

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

## Connection to AIWare 2026

This tool supports the AIWare 2026 submission on attention patterns in code LLMs:

1. **Finding**: Python inline doctests (`>>>`) show 2.8-4.4× stronger attention to function tokens than Rust `#[test]` attributes
2. **Method**: Attention weight extraction at each layer with Welch's t-test for statistical significance
3. **Implication**: Test syntax structure (inline vs separated) affects how models learn function-test relationships

See [RIGOR_EXPERIMENT.md](RIGOR_EXPERIMENT.md) for full methodology and results.

## Development

```bash
# Run tests
cargo test

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
| Qwen2.5-Coder 7B | `Qwen/Qwen2.5-Coder-7B-Instruct` | Qwen2 |
| CodeGemma 7B | `google/codegemma-7b-it` | Gemma |

**Why these models?** Mechanistic interpretability requires access to model internals (attention weights) that proprietary models (Claude, GPT-4) do not expose. Selection was constrained to open-source code LLMs that: (1) fit within 16GB VRAM, (2) are compatible with [candle](https://github.com/huggingface/candle) (Rust ML framework), and (3) demonstrate both Python and Rust code generation capability.

## MI for the Rest of Us

PLIP-rs demonstrates that meaningful mechanistic interpretability research is possible with **consumer hardware**. This wasn't easy—running 7B parameter models with full attention extraction on 16GB VRAM required:

- **KV-cache with hybrid steering**: Cache K,V tensors during prompt processing, then generate efficiently with full steering compatibility. Enables steering experiments without full sequence recomputation.
- **Shared mask caching**: Attention masks (16MB+ for seq_len=2048) are cached by `(seq_len, device, dtype)` and reused across all model backends, avoiding repeated allocations.
- **Memory-limited generation**: Automatic cache trimming to 75% when memory limits are exceeded, enabling long-context generation within VRAM constraints.
- **Rust/candle** instead of Python/PyTorch for the fine-grained memory control needed.
- **Model-agnostic corpus format** to avoid redundant preprocessing per model.

The result: statistically significant findings (p < 0.0002) across 4 models on hardware that costs ~$500, not $50,000.

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
  note = {Attention analysis for AIWare 2026},
  url = {https://github.com/PCfVW/plip-rs}
}
```
