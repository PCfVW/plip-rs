# PLIP-rs - Command Reference

**Programming Language Internal Probing**

A Rust library for probing code model internals using linear probes, logit lens, attention analysis, and **attention intervention experiments** (knockout and steering). Designed for AIware 2026 research on test-awareness in language models.

## Table of Contents

- [Main CLI (plip-rs)](#main-cli-plip-rs)
- [Universal Corpus Format (Perfect Positioning)](#universal-corpus-format-perfect-positioning)
- [Example Binaries](#example-binaries)
  - [load_model](#load_model)
  - [tokenize](#tokenize)
  - [inference](#inference)
  - [logit_lens](#logit_lens)
  - [list_tensors](#list_tensors)
  - [test_emergence](#test_emergence)
  - [attention_patterns](#attention_patterns)
  - [multi_sample_attention](#multi_sample_attention)
  - [statistical_attention](#statistical_attention)
  - [verify_full_corpus](#verify_full_corpus)
  - [layer_scan](#layer_scan)
  - [list_models](#list_models)
  - [verify_positions](#verify_positions)
  - [verify_tokens](#verify_tokens)
- [Universal Positioning Tools](#universal-positioning-tools)
  - [convert_corpus](#convert_corpus)
  - [layer_scan_universal](#layer_scan_universal)
  - [verify_positions_universal](#verify_positions_universal)
- [Attention Intervention Tools](#attention-intervention-tools)
  - [attention_ablation](#attention_ablation)
  - [ablation_experiment](#ablation_experiment)
  - [steering_calibrate](#steering_calibrate)
  - [steering_experiment](#steering_experiment)
  - [steering_generate](#steering_generate)
- [Debug Tools](#debug-tools)
  - [debug_config](#debug_config)
  - [debug_weights](#debug_weights)
  - [debug_generation](#debug_generation)
- [Build Commands](#build-commands)
- [Environment Variables](#environment-variables)
- [Supported Models](#supported-models)

---

## Main CLI (plip-rs)

Run the complete PLIP experiment: load a model, process a corpus, train linear probes at each layer, and report accuracy.

```bash
plip-rs [OPTIONS]
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model <MODEL>` | `-m` | HuggingFace model ID | `bigcode/starcoder2-3b` |
| `--corpus <PATH>` | `-c` | Path to corpus JSON file | `corpus/samples.json` |
| `--output <PATH>` | `-o` | Output directory for results | `outputs` |
| `--verbose` | `-v` | Enable verbose (DEBUG) logging | false |
| `--cpu` | | Force CPU mode | false |
| `--version` | `-V` | Print version information | |
| `--help` | `-h` | Print help | |

### Examples

```bash
# Basic run with defaults
plip-rs

# Specify model and corpus
plip-rs --model "bigcode/starcoder2-3b" --corpus corpus/samples.json

# Run on CPU with verbose output
plip-rs --cpu --verbose

# Custom output directory
plip-rs --output results/experiment1/

# Full example with Qwen model
plip-rs \
    --model "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --corpus corpus/custom_samples.json \
    --output outputs/qwen_experiment \
    --verbose
```

---

## Universal Corpus Format (Perfect Positioning)

**NEW in v2.5:** PLIP-rs now supports a **model-agnostic corpus format** that uses character positions (byte offsets) instead of token indices. This eliminates the need for model-specific corpus files and ensures 100% accuracy across all tokenizer architectures.

### The Problem with Token Positions

Different models tokenize code differently:
- `">>>"` might be 1 token in Qwen but 3 tokens in StarCoder2
- Token boundaries vary by model architecture
- Required maintaining separate corpus files per model

### The Solution: Character Positions

The universal format stores **character positions** (byte offsets into the code string):

```json
{
  "_format_version": "2.0",
  "_description": "Universal corpus with character positions",
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

At runtime, PLIP-rs converts character positions to token positions using the model's actual tokenizer with offset mapping.

### Benefits

| Aspect | Legacy Format | Universal Format |
|--------|---------------|------------------|
| Corpus files needed | 1 per model | **1 for all models** |
| Position accuracy | ~85-95% | **100%** |
| New model support | Manual corpus creation | **Works automatically** |
| Maintenance | High | **Low** |

### Validated Results

With perfect positioning, all 4 tested models achieve p < 0.0002:

| Model | Best Layer | Python μ | Rust μ | Ratio | p-value |
|-------|------------|----------|--------|-------|---------|
| Qwen-7B | 16 | 9.08% | 2.59% | 3.51× | 0.000003 |
| Qwen-3B | 14 | 8.47% | 3.05% | 2.78× | 0.000009 |
| StarCoder2-3B | 23 | 7.19% | 2.41% | 2.98× | 0.000004 |
| CodeGemma-7B | 24 | 5.23% | 1.20% | 4.35× | 0.000114 |

### Quick Start

```bash
# Use the universal corpus (works with any model)
cargo run --release --example layer_scan_universal -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct"

cargo run --release --example layer_scan_universal -- \
    --model "bigcode/starcoder2-3b"

cargo run --release --example layer_scan_universal -- \
    --model "google/codegemma-7b-it"
```

---

## Example Binaries

Examples are run using `cargo run --example <name>`. Add `--release` for optimized builds and `--no-default-features` with `-- --cpu` for CPU-only mode.

### load_model

Load a model and print its configuration (architecture, layers, hidden dimensions, vocab size).

```bash
cargo run --example load_model [-- OPTIONS]
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model <MODEL>` | `-m` | HuggingFace model ID | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| `--cpu` | | Force CPU mode | false |

#### Examples

```bash
# Load default model (Qwen2.5-Coder-3B)
cargo run --release --example load_model

# Load StarCoder2 on CPU
cargo run --release --no-default-features --example load_model -- \
    --cpu --model "bigcode/starcoder2-3b"

# Load Qwen on CPU
cargo run --release --no-default-features --example load_model -- \
    --cpu --model "Qwen/Qwen2.5-Coder-3B-Instruct"
```

---

### tokenize

Tokenize sample Python and Rust code using the StarCoder2 tokenizer.

```bash
cargo run --example tokenize
```

#### Options

No command-line options. Uses hardcoded Fibonacci examples in Python and Rust.

#### Examples

```bash
cargo run --example tokenize
```

---

### inference

Run forward inference on code and extract layer activations.

```bash
cargo run --example inference
```

#### Options

No command-line options. Uses a hardcoded "Hello, world!" Rust example.

#### Examples

```bash
cargo run --release --example inference
```

---

### logit_lens

Logit Lens analysis: see what the model predicts at each transformer layer.

```bash
cargo run --example logit_lens [-- OPTIONS]
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model <MODEL>` | `-m` | HuggingFace model ID | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| `--code <CODE>` | `-c` | Code to analyze | `fn add(a: i32, b: i32) -> i32 { a + b }\n` |
| `--top-k <N>` | `-t` | Number of top predictions per layer | `5` |
| `--cpu` | | Force CPU mode | false |
| `--detailed` | | Show all layers and predictions | false |

#### Examples

```bash
# Basic analysis
cargo run --release --example logit_lens

# CPU mode with custom code
cargo run --release --no-default-features --example logit_lens -- \
    --cpu \
    --code "def hello():\n    print("

# Detailed output with StarCoder2
cargo run --release --no-default-features --example logit_lens -- \
    --cpu \
    --model "bigcode/starcoder2-3b" \
    --detailed

# Analyze test attribute completion
cargo run --release --example logit_lens -- --code "#[te" --top-k 10
```

---

### list_tensors

List tensor names from a model's safetensors file.

```bash
cargo run --example list_tensors
```

#### Options

No command-line options. Uses `bigcode/starcoder2-3b`.

#### Examples

```bash
cargo run --example list_tensors
```

---

### test_emergence

AIware 2026 experiment: Track when test-related tokens emerge in Logit Lens predictions.

```bash
cargo run --example test_emergence [-- OPTIONS]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cpu` | Use CPU instead of CUDA | false |
| `--model <MODEL>` | Model to use | `bigcode/starcoder2-3b` |
| `--top-k <N>` | Number of top predictions to consider | `100` |

#### Examples

```bash
# Default analysis
cargo run --release --example test_emergence

# CPU mode with Qwen
cargo run --release --no-default-features --example test_emergence -- \
    --cpu \
    --model "Qwen/Qwen2.5-Coder-3B-Instruct"

# More extensive search
cargo run --release --example test_emergence -- --top-k 500
```

---

### attention_patterns

AIware 2026 experiment: Analyze attention patterns for test-related tokens.

```bash
cargo run --example attention_patterns [-- OPTIONS]
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model <MODEL>` | `-m` | HuggingFace model ID | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| `--cpu` | | Use CPU instead of CUDA | false |
| `--layers <N>` | | Number of layers to analyze | Sample ~7 layers |

#### Examples

```bash
# Default analysis
cargo run --release --example attention_patterns

# CPU mode
cargo run --release --no-default-features --example attention_patterns -- --cpu

# Analyze all layers
cargo run --release --example attention_patterns -- --layers 36

# With StarCoder2
cargo run --release --example attention_patterns -- \
    --model "bigcode/starcoder2-3b"
```

---

### multi_sample_attention

Control 3 for AIware 2026: Multi-sample attention analysis with statistical significance.

```bash
cargo run --example multi_sample_attention [-- OPTIONS]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cpu` | Use CPU instead of CUDA | false |
| `--model <MODEL>` | Model to use | `Qwen/Qwen2.5-Coder-7B-Instruct` |

#### Examples

```bash
# Default analysis (Qwen 7B)
cargo run --release --example multi_sample_attention

# CPU mode with smaller model
cargo run --release --no-default-features --example multi_sample_attention -- \
    --cpu \
    --model "Qwen/Qwen2.5-Coder-3B-Instruct"
```

---

### statistical_attention

Full statistical attention analysis with Welch's t-test for AIware 2026.

```bash
cargo run --example statistical_attention [-- OPTIONS]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model <MODEL>` | Model to use | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `--corpus <PATH>` | Corpus JSON file | `corpus/attention_samples.json` |
| `--layer <N>` | Layer to analyze | `12` |
| `--output <PATH>` | Output JSON file | auto-generated |
| `--cpu` | Use CPU instead of CUDA | false |

#### Examples

```bash
# Default analysis (Qwen 7B, layer 12)
cargo run --release --example statistical_attention

# Specify layer and output
cargo run --release --example statistical_attention -- \
    --layer 14 \
    --output outputs/stats_layer14.json

# With StarCoder2 and custom corpus
cargo run --release --example statistical_attention -- \
    --model "bigcode/starcoder2-3b" \
    --corpus "corpus/attention_samples_bigcode_starcoder2_3b.json" \
    --layer 20
```

---

### verify_full_corpus

Verify token positions across the full corpus for a specific model.

```bash
cargo run --example verify_full_corpus [-- OPTIONS]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model <MODEL>` | Model to use | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `--corpus <PATH>` | Corpus JSON file | `corpus/attention_samples.json` |
| `--cpu` | Use CPU instead of CUDA | false |

#### Examples

```bash
# Verify for Qwen
cargo run --release --example verify_full_corpus

# Verify for StarCoder2
cargo run --release --example verify_full_corpus -- \
    --model "bigcode/starcoder2-3b"
```

---

### layer_scan

Scan multiple layers to find optimal attention patterns for any model.

```bash
cargo run --example layer_scan [-- OPTIONS]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model <MODEL>` | Model to use | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `--corpus <PATH>` | Corpus JSON file | `corpus/attention_samples.json` |
| `--output <PATH>` | Output JSON file | auto-generated |
| `--start-layer <N>` | First layer to scan | `n_layers / 3` |
| `--end-layer <N>` | Last layer to scan | `n_layers - 1` |
| `--cpu` | Use CPU instead of CUDA | false |

#### Examples

```bash
# Scan Qwen layers (auto-detects range)
cargo run --release --example layer_scan

# Scan StarCoder2 with corrected corpus
cargo run --release --example layer_scan -- \
    --model "bigcode/starcoder2-3b" \
    --corpus "corpus/attention_samples_bigcode_starcoder2_3b.json" \
    --output "outputs/layer_scan_starcoder2.json"

# Scan specific layer range
cargo run --release --example layer_scan -- \
    --start-layer 5 \
    --end-layer 20
```

---

### list_models

List locally cached HuggingFace models compatible with PLIP-rs.

```bash
cargo run --example list_models
```

#### Options

No command-line options. Scans the HuggingFace cache directory.

#### Output

Lists all cached models with their detected architecture:
- **StarCoder2**: BigCode starcoder2 models
- **Qwen2**: Qwen/Qwen2.5 models
- **Gemma**: Google Gemma/CodeGemma models
- **Unknown**: Other architectures (may not work)

#### Examples

```bash
# List cached models
cargo run --release --example list_models
```

#### Sample Output

```
┌────────────────────────────────────────────────┬────────────┐
│ Model ID                                       │ Architecture│
├────────────────────────────────────────────────┼────────────┤
│ ✓ Qwen/Qwen2.5-Coder-7B-Instruct              │      Qwen2 │
│ ✓ Qwen/Qwen2.5-Coder-3B-Instruct              │      Qwen2 │
│ ✓ bigcode/starcoder2-3b                        │  StarCoder2 │
│ ✓ google/codegemma-7b-it                       │      Gemma │
└────────────────────────────────────────────────┴────────────┘
```

---

### verify_positions

Verify and correct token positions for any model's tokenizer.

```bash
cargo run --example verify_positions [-- OPTIONS]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model <MODEL>` | Model to verify for | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `--corpus <PATH>` | Corpus JSON file | `corpus/attention_samples.json` |
| `--output <PATH>` | Output verification report JSON | none |
| `--fix` | Generate corrected corpus file | false |
| `--verbose` | Show detailed token listings | false |
| `--cpu` | Use CPU instead of CUDA | false |

#### Examples

```bash
# Verify positions for Qwen (should be 100% correct)
cargo run --release --example verify_positions

# Verify and fix positions for StarCoder2
cargo run --release --example verify_positions -- \
    --model "bigcode/starcoder2-3b" \
    --fix

# Verify for CodeGemma with verbose output
cargo run --release --example verify_positions -- \
    --model "google/codegemma-7b-it" \
    --verbose \
    --fix

# Save verification report
cargo run --release --example verify_positions -- \
    --model "bigcode/starcoder2-3b" \
    --output "outputs/verification_starcoder2.json"
```

#### Output Files

When `--fix` is specified, generates a corrected corpus file:
- `corpus/attention_samples_<model_name>.json`

Example: `corpus/attention_samples_bigcode_starcoder2_3b.json`

---

### verify_tokens

Print detailed tokenization output to verify token positions and marker detection.

```bash
cargo run --release --example verify_tokens
```

#### Options

No command-line options. Uses hardcoded Python and Rust samples with the Qwen tokenizer.

#### Description

Displays token-by-token breakdown of sample code with position annotations:
- Marks potential test marker tokens (`>>>`, `#[test]`)
- Highlights function-related tokens (`fn`, parameter names)
- Shows Python doctest, Rust test, and baseline samples

Useful for understanding how different code constructs tokenize and where markers land.

#### Example Output

```
═══════════════════════════════════════════════════════════════════
Python Doctest Sample
═══════════════════════════════════════════════════════════════════
def add(a, b):
    """
    >>> add(2, 3)
    5
    """
    return a + b

Tokens:
  0: "def"
  1: " add"
  2: "("
  3: "a"                ← PARAMETER
...
 12: ">>>"              ← POTENTIAL MARKER
```

---

## Universal Positioning Tools

These tools use the **universal corpus format** with character positions, providing model-agnostic analysis without preprocessing.

### convert_corpus

Convert a legacy token-position corpus to the universal character-position format.

```bash
cargo run --example convert_corpus [-- OPTIONS]
```

#### Options

| Option | Description | Required |
|--------|-------------|----------|
| `--input <PATH>` | Input legacy corpus file | Yes |
| `--output <PATH>` | Output universal corpus file | Yes |
| `--verbose` | Show detailed conversion info | No |

#### Examples

```bash
# Convert legacy corpus to universal format
cargo run --release --example convert_corpus -- \
    --input corpus/attention_samples.json \
    --output corpus/attention_samples_universal.json

# With verbose output
cargo run --release --example convert_corpus -- \
    --input corpus/attention_samples.json \
    --output corpus/attention_samples_universal.json \
    --verbose
```

#### Notes

- The converter extracts character positions directly from the code strings
- Token positions in the legacy file are ignored (recalculated from code)
- Supports Python doctest, Rust test, and baseline samples

---

### layer_scan_universal

**Recommended for all new analysis.** Scan multiple layers using the universal corpus format. Works with ANY model without preprocessing.

```bash
cargo run --example layer_scan_universal [-- OPTIONS]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model <MODEL>` | Model to analyze | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `--corpus <PATH>` | Universal corpus JSON file | `corpus/attention_samples_universal.json` |
| `--output <PATH>` | Output JSON file | auto-generated |
| `--start-layer <N>` | First layer to scan | `n_layers / 3` |
| `--end-layer <N>` | Last layer to scan | `n_layers - 1` |
| `--verbose` | Show position conversion details | false |
| `--cpu` | Use CPU instead of CUDA | false |

#### Examples

```bash
# Scan Qwen-7B (default)
cargo run --release --example layer_scan_universal

# Scan StarCoder2
cargo run --release --example layer_scan_universal -- \
    --model "bigcode/starcoder2-3b"

# Scan CodeGemma
cargo run --release --example layer_scan_universal -- \
    --model "google/codegemma-7b-it" \
    --output outputs/layer_scan_universal_codegemma.json

# Scan Qwen-3B with verbose output
cargo run --release --example layer_scan_universal -- \
    --model "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --verbose

# Scan specific layer range
cargo run --release --example layer_scan_universal -- \
    --start-layer 10 \
    --end-layer 25
```

#### Output

Produces a formatted table with statistical analysis per layer:

```
┌───────┬────────────┬────────────┬─────────┬──────────┬──────────┬──────────┐
│ Layer │ Python μ   │ Rust μ     │  Ratio  │ t-stat   │ df       │ p-value  │
├───────┼────────────┼────────────┼─────────┼──────────┼──────────┼──────────┤
│    16 │      9.08% │      2.59% │   3.51× │    8.88 │    10.6 │  0.0000 *** │
│    17 │      8.89% │      2.53% │   3.51× │    8.39 │    10.4 │  0.0000 *** │
...
└───────┴────────────┴────────────┴─────────┴──────────┴──────────┴──────────┘
```

Also saves position conversion statistics showing fuzzy match rates.

---

### verify_positions_universal

Verify that character positions in the universal corpus correctly convert to token positions for any model.

```bash
cargo run --example verify_positions_universal [-- OPTIONS]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model <MODEL>` | Model to verify positions for | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `--corpus <PATH>` | Universal corpus JSON file | `corpus/attention_samples_universal.json` |
| `--output <PATH>` | Output verification report JSON | none |
| `--verbose` | Show detailed token listings | false |
| `--cpu` | Use CPU instead of CUDA | false |

#### Examples

```bash
# Verify for Qwen-7B
cargo run --release --example verify_positions_universal

# Verify for StarCoder2
cargo run --release --example verify_positions_universal -- \
    --model "bigcode/starcoder2-3b"

# Verify for CodeGemma with verbose output
cargo run --release --example verify_positions_universal -- \
    --model "google/codegemma-7b-it" \
    --verbose

# Save verification report
cargo run --release --example verify_positions_universal -- \
    --model "bigcode/starcoder2-3b" \
    --output outputs/verification_universal_starcoder2.json
```

#### Output

Shows conversion success rates and any position mismatches:

```
═══════════════════════════════════════════════════════════════════
  Universal Position Verification
═══════════════════════════════════════════════════════════════════

Model: bigcode/starcoder2-3b
Corpus: corpus/attention_samples_universal.json

  Total samples: 20
  Successful conversions: 20
  Failed conversions: 0
  Fuzzy matches: 0

All positions convert correctly!
```

---

## Attention Intervention Tools

These tools implement causal intervention experiments to test the mechanistic role of attention patterns in test-awareness. Two intervention types are supported:

- **Knockout (Ablation)**: Remove attention edges completely (pre-softmax -inf masking)
- **Steering**: Boost or modify attention weights (post-softmax scaling with renormalization)

### attention_ablation

Interactive attention knockout experiments for understanding which attention edges are causally important.

```bash
cargo run --release --example attention_ablation [-- OPTIONS]
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model <MODEL>` | `-m` | HuggingFace model ID | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| `--cpu` | | Use CPU instead of CUDA | false |
| `--ablate-test-marker` | | Ablate attention from test marker tokens | false |
| `--layer <N>` | | Specific layer to ablate | Auto-select based on model |
| `--head <N>` | | Specific head to ablate | All heads |

#### Examples

```bash
# Default demo: show tokens and simple ablation
cargo run --release --example attention_ablation

# Ablate test marker attention for both Python/Rust
cargo run --release --example attention_ablation -- --ablate-test-marker

# Target specific layer and model
cargo run --release --example attention_ablation -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --layer 16 \
    --ablate-test-marker

# Ablate single head only
cargo run --release --example attention_ablation -- \
    --ablate-test-marker \
    --layer 14 \
    --head 0
```

#### Output

Shows KL divergence after ablation and top changed tokens:

```
=== Python Doctest Ablation ===

Found marker '>>>' at token position 12
Testing ablation at layer 14

Results:
  KL divergence: 0.012345

  Top changed tokens:
    'return': 0.8521 -> 0.7234 (diff: -0.1287)
    '+': 0.0823 -> 0.1456 (diff: 0.0633)
```

---

### ablation_experiment

Full ablation experiment with statistical analysis across corpus samples. Tests the causal hypothesis that attention from test markers to function tokens is critical for test preservation.

```bash
cargo run --release --example ablation_experiment [-- OPTIONS]
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model <MODEL>` | `-m` | HuggingFace model ID | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| `--corpus <PATH>` | | Path to corpus file | `corpus/attention_samples_universal.json` |
| `--layer <N>` | | Specific layer to test | Auto-select based on model size |
| `--layer-start <N>` | | Start layer for contiguous window knockout | - |
| `--layer-end <N>` | | End layer for contiguous window knockout (inclusive) | - |
| `--scan-layers` | | Test all layers and report per-layer effects | false |
| `--scan-windows` | | Scan windows of increasing size around best layer | false |
| `--window-center <N>` | | Center layer for window scan | Target layer |
| `--max-radius <N>` | | Maximum window radius for window scan | `5` |
| `--slide-window <N>` | | Slide a fixed-size window across all layers | - |
| `--all-edges` | | Knockout ALL outgoing attention from marker | false |
| `--all-heads` | | Target all heads | `true` |
| `--include-baselines` | | Include baseline samples (non-test code) | false |
| `--output <PATH>` | `-o` | Output JSON file for results | none |
| `--verbose` | `-v` | Verbose output | false |
| `--cpu` | | Use CPU instead of CUDA | false |

#### Examples

```bash
# Basic experiment at default layer
cargo run --release --example ablation_experiment

# Scan all layers to find most important ones
cargo run --release --example ablation_experiment -- --scan-layers

# Test with contiguous layer window
cargo run --release --example ablation_experiment -- \
    --layer-start 12 --layer-end 18

# Scan windows of increasing size around layer 14
cargo run --release --example ablation_experiment -- \
    --scan-windows --window-center 14 --max-radius 5

# Slide 3-layer window across model
cargo run --release --example ablation_experiment -- --slide-window 3

# Save results to JSON
cargo run --release --example ablation_experiment -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --layer 16 \
    --output outputs/ablation_qwen7b.json \
    --verbose
```

#### Output

```
============================================================
EXPERIMENT RESULTS
============================================================

Model: Qwen/Qwen2.5-Coder-3B-Instruct
Layer tested: 14
Number of heads: 16

=== Python Doctest Results ===
  Samples: 10
  Mean KL divergence: 0.008234
  Std deviation: 0.003421

=== Rust Test Results ===
  Samples: 10
  Mean KL divergence: 0.005123
  Std deviation: 0.002156

=== Statistical Comparison (Python vs Rust) ===
Welch's t-statistic: 2.4521
p-value: 0.024567
Significant difference (p < 0.05): YES

=== Interpretation ===
FINDING: Python doctests are MORE affected by knockout than Rust tests.
         This suggests different preservation mechanisms per language.
```

---

### steering_calibrate

Measure baseline attention levels from test markers to function tokens. Used to calibrate steering targets for dose-response experiments.

```bash
cargo run --release --example steering_calibrate [-- OPTIONS]
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model <MODEL>` | `-m` | HuggingFace model ID | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| `--corpus <PATH>` | | Path to universal corpus file | `corpus/attention_samples_universal.json` |
| `--layer <N>` | | Specific layer to measure | Auto-select based on model |
| `--verbose` | `-v` | Show per-sample attention levels | false |
| `--cpu` | | Use CPU instead of CUDA | false |

#### Examples

```bash
# Calibrate for Qwen-3B (default)
cargo run --release --example steering_calibrate

# Calibrate for Qwen-7B
cargo run --release --example steering_calibrate -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct"

# Calibrate specific layer with verbose output
cargo run --release --example steering_calibrate -- \
    --model "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --layer 20 \
    --verbose
```

#### Output

```
============================================================
CALIBRATION RESULTS
============================================================

Model: Qwen/Qwen2.5-Coder-3B-Instruct
Layer: 20
Samples: 10 Python, 10 Rust

=== Baseline Attention Levels ===

  Python doctest (>>> → fn):  5.70%
  Rust test (#[test] → fn):   2.32%
  Ratio (Python/Rust):        2.46×

=== Recommended Steering Targets ===

  Target (Python level):      5.70%
  Scale factor for Rust:      2.46×

=== Dose-Response Levels ===

  Scale  | Absolute Attention
  -------|-------------------
  0.5×   | 1.16%
  1.0×   | 2.32% ← Baseline
  2.0×   | 4.64%
  3.0×   | 6.96% ← Python level
```

---

### steering_experiment

Run dose-response experiments to measure how different attention steering intensities affect model behavior (via KL divergence). Tests whether Rust test marker attention can be boosted to Python doctest levels without disrupting model outputs.

```bash
cargo run --release --example steering_experiment [-- OPTIONS]
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model <MODEL>` | `-m` | HuggingFace model ID | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| `--corpus <PATH>` | | Path to universal corpus file | `corpus/attention_samples_universal.json` |
| `--layer <N>` | | Layer to intervene on | Auto-select based on model |
| `--target-attention <F>` | | Target attention level (0.0-1.0) for SetValue mode | Scale mode |
| `--output <PATH>` | | Output JSON file for results | none |
| `--max-samples <N>` | | Limit number of samples processed | All samples |
| `--verbose` | `-v` | Show per-sample KL divergence | false |
| `--cpu` | | Use CPU instead of CUDA | false |

#### Intervention Modes

1. **Scale mode** (default): Tests multiple scaling factors (0.5×, 1.0×, 2.0×, 3.0×, 4.0×, 6.0×)
2. **SetValue mode**: Sets attention to a specific target level (use `--target-attention`)

#### Examples

```bash
# Dose-response experiment for Qwen-3B (Scale mode)
cargo run --release --example steering_experiment

# Dose-response for Qwen-7B at layer 16
cargo run --release --example steering_experiment -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --layer 16

# Target specific attention level (SetValue mode)
cargo run --release --example steering_experiment -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --layer 16 \
    --target-attention 0.09

# Save results to JSON with verbose output
cargo run --release --example steering_experiment -- \
    --model "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --output outputs/steering_qwen3b.json \
    --verbose
```

#### Output

```
============================================================
DOSE-RESPONSE RESULTS
============================================================

Model: Qwen/Qwen2.5-Coder-3B-Instruct
Layer: 20
Intervention: Scale

Dose    | KL Divergence      | Attention    | Samples
--------|--------------------|--------------|---------
0.5×    | 470.42 ± 332.08    |   1.23%      | 10
1.0×    | 469.96 ± 331.31    |   2.32%      | 10
2.0×    | 470.52 ± 332.21    |   4.17%      | 10
3.0×    | 470.53 ± 332.22    |   5.71%      | 10
4.0×    | 470.52 ± 332.21    |   7.02%      | 10
6.0×    | 470.55 ± 332.22    |   9.17%      | 10

=== Analysis ===

Baseline (1.0×): KL=469.96, Attention=2.32%
Max KL at 6.0×: KL=470.55, Attention=9.17%
KL divergence shows non-monotonic relationship with dose
```

#### Key Findings

The steering experiments demonstrate:

1. **Attention boosting works**: Rust attention can be boosted from 2.3% to 9%+ (matching Python levels) via 3-4× scaling
2. **Flat KL divergence**: Model output distributions remain stable regardless of steering intensity
3. **Safe intervention**: Steering can be applied without catastrophically disrupting model behavior

See [STEERING_RESULTS.md](STEERING_RESULTS.md) for detailed experiment results.

---

### steering_generate

Generation proof-of-concept: tests whether attention steering during autoregressive generation affects test preservation behavior.

```bash
cargo run --release --example steering_generate [-- OPTIONS]
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model <MODEL>` | `-m` | HuggingFace model ID | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| `--layer <N>` | | Layer to apply steering | Auto-select based on model |
| `--scale <F>` | | Steering scale factor | `3.0` |
| `--max-tokens <N>` | | Maximum tokens to generate | `150` |
| `--temperature <F>` | | Temperature for sampling (0.0 = greedy) | `0.0` |
| `--chat` | | Use chat template formatting (for instruct models) | false |
| `--cpu` | | Force CPU mode | false |

#### Examples

```bash
# Default generation test
cargo run --release --example steering_generate

# With Qwen-7B and chat template
cargo run --release --example steering_generate -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --chat

# Different steering scale
cargo run --release --example steering_generate -- \
    --scale 5.0 \
    --layer 16

# More generation with temperature
cargo run --release --example steering_generate -- \
    --max-tokens 200 \
    --temperature 0.3
```

#### Output

Compares baseline generation vs steered generation across test prompts:

```
============================================================
SUMMARY
============================================================

| Sample          | Baseline #[test] | Steered #[test] | Change |
|-----------------|------------------|-----------------|--------|
| max_function    | No               | Yes             | ✓ GAINED |
| is_even         | No               | No              | = NONE |
| factorial       | Yes              | Yes             | = KEPT |

Baseline preservation: 1/3
Steered preservation:  2/3

✓ Steering IMPROVED test preservation!
```

---

## Debug Tools

Diagnostic utilities for troubleshooting model loading, configuration, and generation issues.

### debug_config

Inspect model configuration from HuggingFace config.json.

```bash
cargo run --release --example debug_config
```

#### Description

Loads and displays the raw config.json from a model, showing:
- Hidden size and intermediate size
- Number of attention heads and KV heads
- Number of layers and vocab size
- RoPE theta and scaling settings
- Computed head dimensions and KV ratios

Useful for verifying model architecture compatibility.

---

### debug_weights

Inspect model weight tensors and sharding.

```bash
cargo run --release --example debug_weights
```

#### Description

Lists safetensors weight names and shapes:
- Weight mapping from index file
- Tensor shapes and dtypes from first shard
- Expected weight names for the architecture

Useful for debugging weight loading issues.

---

### debug_generation

Debug token generation and decoding pipeline.

```bash
cargo run --release --example debug_generation
```

#### Description

Tests the full generation pipeline:
1. Tokenization roundtrip verification
2. Logit lens analysis
3. Forward pass shape checking
4. Single token generation
5. Comparison of logit lens predictions vs generation
6. Chat template formatting test

Useful for diagnosing why generation produces unexpected output.

---

## Build Commands

### Standard Build

```bash
# Debug build (faster compile, slower runtime)
cargo build

# Release build (slower compile, optimized runtime)
cargo build --release

# Build specific example
cargo build --release --example logit_lens
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `cuda` | Enable CUDA GPU acceleration | **On** |

```bash
# Disable CUDA (CPU only)
cargo build --release --no-default-features

# Explicitly enable CUDA
cargo build --release --features cuda
```

### Code Quality

```bash
# Format code
cargo fmt

# Lint
cargo clippy

# Run tests
cargo test
```

---

## Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `RUST_LOG` | Logging level (error, warn, info, debug, trace) | Debugging |
| `HF_HOME` | HuggingFace cache directory | Custom cache |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use (e.g., `0`, `0,1`) | Multi-GPU |

### Examples

```bash
# Debug logging
RUST_LOG=debug cargo run --release --example logit_lens

# Use specific GPU
CUDA_VISIBLE_DEVICES=1 cargo run --release -- --corpus corpus/samples.json
```

---

## Supported Models

| Model | HuggingFace ID | Architecture | VRAM |
|-------|----------------|--------------|------|
| StarCoder2 3B | `bigcode/starcoder2-3b` | StarCoder2 | ~6 GB |
| Qwen2.5-Coder 3B | `Qwen/Qwen2.5-Coder-3B-Instruct` | Qwen2 | ~6 GB |
| Qwen2.5-Coder 7B | `Qwen/Qwen2.5-Coder-7B-Instruct` | Qwen2 | ~14 GB |
| CodeGemma 7B | `google/codegemma-7b-it` | Gemma | ~14 GB |

---

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error (model loading, file I/O, etc.) |
| 101 | Panic (Rust runtime error) |

---

## See Also

- [README.md](README.md) - Project overview and quick start
- [corpus/README.md](corpus/README.md) - Corpus format documentation
- [STEERING_RESULTS.md](STEERING_RESULTS.md) - Detailed steering experiment results
- [ABLATION_RESULTS.md](ABLATION_RESULTS.md) - Attention knockout experiment results
- [SEGA experiment-runner](https://github.com/PCfVW/d-Heap-priority-queue/tree/master/experiment/experiment-runner) - Related experiment runner (in parent repository)
