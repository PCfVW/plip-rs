# PLIP-rs: Ablation Experiment Results (Attention Knockout & State Knockout)

**Created**: February 1, 2026
**Updated**: February 9, 2026 (RWKV-6 state knockout added)
**Purpose**: Test causal importance of test marker information flow for model predictions
**Hardware**: RTX 5060 Ti (16GB VRAM)

---

## Executive Summary

**Research Question**: Is the information flow from test markers (`>>>`, `#[test]`) to function tokens *causally necessary* for model predictions, or merely *correlational*?

**Key Findings**:

1. **Model-Specific Causal Pathways**: Different models handle test marker information very differently:
   - **Qwen-7B**: Layer 2 shows 189× stronger knockout effect for Python vs Rust
   - **StarCoder2**: Near-zero sensitivity (uses redundant pathways)
   - **CodeGemma**: Balanced effects (both languages equally affected)
   - **RWKV-6**: First statistically significant result (p = 0.018) — Python 4.6× more affected

2. **Layer Compensation**: In Qwen-7B, knocking out layer 2 alone causes 0.094% Python KL, but knocking out layers 1-3 together **reduces** the effect to 0.025% — demonstrating that adjacent layers compensate for knocked-out attention. In RWKV-6, the opposite: KL **increases** with window size (0.34% → 1.93%), reflecting the recurrent architecture's cumulative state.

3. **First Non-Transformer Model (RWKV-6)**: State knockout on the gated-linear RNN produces the **first statistically significant** Python vs Rust difference (p = 0.018, t = 2.84). This is semantically equivalent to transformer all-edge knockout: the marker position becomes invisible to all future tokens.

4. **Correlation ≠ Causation**: Despite Python showing 2.8-4.4× stronger attention *correlation* in all 4 code-specialized transformers (see RIGOR_EXPERIMENT.md), causal ablation reveals model-specific dependencies. High variance prevents statistical significance in transformers (all p > 0.05), but practical effect sizes vary dramatically across architectures.

---

## Background: From Correlation to Causation

### What We Knew (Attention Analysis - RIGOR_EXPERIMENT.md)

| Metric | Python `>>>` | Rust `#[test]` | Ratio | p-value |
|--------|-------------|----------------|-------|---------|
| Attention to function tokens (Qwen-7B, layer 16) | 9.08% | 2.59% | **3.51×** | **0.000003** |
| Range across 4 code-specialized models | 5.2-9.1% | 1.2-3.1% | **2.8-4.4×** | **<0.0002** |

**Interpretation**: Python doctest markers "look at" function tokens much more strongly than Rust test attributes across all 4 code-specialized models. Two non-code-specialized models (Code-LLaMA-7B, Phi-3-mini) do not show this effect (see RIGOR_EXPERIMENT.md Appendix C).

### What We Asked (Ablation Experiment)

> If we **remove** the marker's influence on subsequent tokens — by zeroing attention edges (transformers) or suppressing state writes (RWKV-6) — does the model's output change differently for Python vs Rust?

This tests whether the observed difference is:
- **Causally important**: Knockout should affect Python more than Rust
- **Merely correlational**: Knockout affects both equally (or neither)

---

## Methodology

### Intervention Mechanism: Attention Knockout (Transformers)

For transformer models, knockout is implemented by adding `-infinity` to attention scores **before softmax**:

```
NORMAL FORWARD PASS:
  attn_scores = Q @ K^T / sqrt(d_k)
  attn_scores += causal_mask
  attn_weights = softmax(attn_scores)     ← test marker attends to function
  output = attn_weights @ V

KNOCKOUT FORWARD PASS:
  attn_scores = Q @ K^T / sqrt(d_k)
  attn_scores += causal_mask
  attn_scores += knockout_mask            ← ADD -inf at (marker, function) positions
  attn_weights = softmax(attn_scores)     ← knocked out edges become exactly 0
  output = attn_weights @ V
```

### Intervention Mechanism: State Knockout (RWKV-6)

For the RWKV-6 gated-linear RNN, there are no attention matrices. Instead, information flows through recurrent state. State knockout suppresses the state write at targeted positions, following the Mamba Knockout methodology (Endy et al., ACL 2025):

```
NORMAL WKV RECURRENCE:
  state = kv + decay * state              ← position contributes to recurrent state

STATE KNOCKOUT RECURRENCE:
  state = decay * state                   ← position's kv contribution suppressed
```

State knockout is **semantically equivalent** to transformer all-edge knockout: the marker position becomes invisible to all future tokens while preserving the decay dynamics. This makes results directly comparable across architectures.

### Measurement: KL Divergence

We measure how much the model's next-token probability distribution changes:

```
KL(baseline || ablated) = Σ p_baseline(x) * log(p_baseline(x) / p_ablated(x))
```

| KL Value | Interpretation |
|----------|----------------|
| > 1% | Significant impact — information flow is causally important |
| 0.1% - 1% | Moderate impact — some causal role |
| < 0.1% | Minimal impact — information flow is redundant |

### Corpus

Using `corpus/attention_samples_universal.json`:
- 10 Python doctest samples
- 10 Rust test samples
- Character-based positions (model-agnostic)

---

## Results by Model

### Qwen/Qwen2.5-Coder-3B-Instruct (36 layers, 16 heads) — Transformer

#### Layer Scan: Finding Causally Important Layers

| Layer | Python KL | Rust KL | Python/Rust Ratio |
|-------|-----------|---------|-------------------|
| **1** | **0.221%** | 0.002% | 116× |
| **0** | **0.129%** | 0.011% | 11× |
| 19 | 0.003% | 0.002% | 1.9× |
| 14 | 0.010% | 0.002% | 4.6× |
| 20 | 0.038% | 0.002% | 17× |

**Key insight**: Early layers (0-1) show strongest knockout effects, not mid-layers (14) where attention *strength* is highest.

#### Full Experiment at Layer 1 (Most Causally Important)

| Metric | Python Doctest | Rust Test |
|--------|---------------|-----------|
| **N samples** | 10 | 10 |
| **Mean KL** | 0.714% | 0.819% |
| **Std Dev** | 1.256% | 2.372% |
| **Min** | 0.001% | 0.001% |
| **Max** | 4.285% | 7.932% |
| **Median** | 0.157% | 0.003% |

**Statistical Test (Welch's t-test)**:
- t-statistic: -0.117
- p-value: **0.908**
- Significant difference: **NO**

#### Per-Sample Results (Layer 1)

**Python Doctest Samples:**

| Sample ID | KL Divergence | Impact Level |
|-----------|---------------|--------------|
| py_string_manipulation | 4.285% | High |
| py_list_operations | 1.172% | Moderate |
| py_nested_structure | 0.955% | Moderate |
| py_complex_params | 0.398% | Moderate |
| py_simple_add | 0.221% | Moderate |
| py_single_char_param | 0.093% | Low |
| py_multi_param | 0.009% | Minimal |
| py_multiple_doctests | 0.005% | Minimal |
| py_default_args | 0.003% | Minimal |
| py_long_name | 0.001% | Minimal |

**Rust Test Samples:**

| Sample ID | KL Divergence | Impact Level |
|-----------|---------------|--------------|
| rust_result_type | **7.932%** | High (outlier) |
| rust_tuple_return | 0.221% | Moderate |
| rust_option_return | 0.020% | Low |
| rust_vec_operations | 0.003% | Minimal |
| rust_multiple_assertions | 0.003% | Minimal |
| rust_cfg_test_module | 0.003% | Minimal |
| rust_simple_add | 0.002% | Minimal |
| rust_should_panic | 0.001% | Minimal |
| rust_generic_complex | 0.001% | Minimal |
| rust_reference_params | 0.006% | Minimal |

**Observation**: High variance in both groups. One Rust sample (`rust_result_type`) shows the highest knockout effect overall.

#### Window Scan: Layer Compensation Analysis

Testing contiguous window knockout centered at layer 1:

| Window | Layers | Python KL | Rust KL | Ratio |
|--------|--------|-----------|---------|-------|
| Single | 1 only | 0.22% | 0.002% | **115×** |
| +1 | 0-2 | 0.23% | 0.014% | 16× |
| +2 | 0-3 | 0.24% | 0.015% | 16× |
| +3 | 0-4 | 0.23% | 0.009% | 24× |
| +4 | 0-5 | 0.23% | 0.014% | 16× |
| +5 | 0-6 | 0.23% | 0.012% | 19× |
| +6 | 0-7 | 0.23% | 0.014% | 16× |
| +7 | 0-8 | 0.22% | 0.014% | 15× |
| +8 | 0-9 | 0.23% | 0.011% | 21× |
| +9 | 0-10 | 0.23% | 0.014% | 16× |
| +10 | 0-11 | 0.23% | 0.011% | 20× |

**Key insight - No Compensation**: Unlike Qwen-7B, Qwen-3B shows **no layer compensation effect**. Python KL stays stable at ~0.22-0.24% regardless of how many layers are knocked out. This suggests:
- Python processing in Qwen-3B is concentrated in layer 1 only
- No redundant pathways in adjacent layers
- The 15-24× Python/Rust ratio is maintained at all scales

---

### Qwen/Qwen2.5-Coder-7B-Instruct (28 layers, 28 heads) — Transformer

#### Layer Scan: Finding Causally Important Layers

| Layer | Python KL | Rust KL | Python/Rust Ratio |
|-------|-----------|---------|-------------------|
| **2** | **0.094%** | 0.002% | **3917×** |
| 0 | 0.002% | 0.003% | 61× |
| 1 | 0.001% | 0.005% | 22× |
| 3 | 0.0005% | 0.002% | 30× |
| 15 | 0.0001% | 0.001% | 9× |

**Key insight**: Layer 2 shows extreme Python-specific effect (3917× ratio), much higher than Qwen-3B.

#### Full Experiment at Layer 2 (Most Causally Important)

| Metric | Python Doctest | Rust Test |
|--------|---------------|-----------|
| **N samples** | 10 | 10 |
| **Mean KL** | 0.946% | 0.005% |
| **Std Dev** | 2.813% | 0.007% |
| **Min** | 0.001% | 0.0002% |
| **Max** | 9.384% | 0.026% |
| **Median** | 0.004% | 0.003% |

**Statistical Test (Welch's t-test)**:
- t-statistic: 1.003
- p-value: **0.338**
- Significant difference: **NO** (high variance in Python samples)

**Notable finding**: Layer 2 shows extremely high Python effect (mean 0.95%) with almost no Rust effect (0.005%), but high variance prevents statistical significance.

#### Window Scan: Layer Compensation Analysis

Testing contiguous window knockout centered at layer 2:

| Window | Layers | Python KL | Rust KL | Ratio |
|--------|--------|-----------|---------|-------|
| Single | 2 only | **0.094%** | 0.00002% | **3884×** |
| ±1 | 1-3 | 0.025% | 0.008% | 3× |
| ±2 | 0-4 | 0.331% | 0.019% | 17× |
| ±3 | 0-5 | 0.308% | 0.016% | 20× |
| ±5 | 0-7 | 0.324% | 0.012% | 28× |
| ±8 | 0-10 | 0.303% | 0.015% | 20× |

**Key insight - Layer Compensation**: Knocking out layer 2 alone shows 0.094% Python effect, but adding adjacent layers (1-3) **reduces** the effect to 0.025%. This demonstrates that **adjacent layers compensate** for the knocked-out layer. Only when the entire early pipeline (0-4+) is knocked out does the effect stabilize at ~0.3%.

#### Sliding Window Scan: Locating the Critical Region

Sliding a fixed-size window across all 28 layers to find where the Python-specific effect is concentrated:

| Window Size | Peak Window | Python KL | Rust KL | Ratio |
|-------------|-------------|-----------|---------|-------|
| **w=2** | 2-3 | 0.081% | 0.0016% | **5223×** |
| **w=3** | 2-4 | 0.080% | 0.002% | **4090×** |
| **w=4** | 2-5 | 0.082% | 0.002% | **4195×** |
| **w=5** | 2-6 | 0.084% | 0.002% | **3897×** |

**Key insight - Layer 2 is the bottleneck**: Regardless of window size, the peak always occurs when the window **starts at layer 2**. The Python KL is remarkably stable (~0.08%) regardless of how many layers are included. This confirms that:
- **Layer 2 is the critical layer** for Python test marker processing
- Including layers 3, 4, 5, 6 doesn't increase the effect (no distributed processing)
- The ~0.08% Python KL is a **ceiling** - the maximum disruption possible from test marker knockout
- The Python/Rust ratio decreases as more layers are added (from 5223× to 3897×) because Rust remains near-zero

**Contrast with expanding window**: The earlier window scan (centered at layer 2, expanding) showed compensation effects. The sliding window shows that **the effect is localized to layer 2** - other regions of the network show negligible effects.

---

### bigcode/starcoder2-3b (30 layers, 24 heads) — Transformer

#### Layer Scan: Finding Causally Important Layers

| Layer | Python KL | Rust KL | Python/Rust Ratio |
|-------|-----------|---------|-------------------|
| 0 | 0.0004% | 0.0003% | 1.3× |
| 7 | 0.0001% | 0.0007% | 0.1× |
| 4-5 | ~0% | 0.0008% | ~0× |

**Key insight**: StarCoder2 shows **virtually no sensitivity** to test marker → function token attention knockout. All KL values are near 0 across all layers.

#### Full Experiment at Layer 0 (Highest Python Effect)

| Metric | Python Doctest | Rust Test |
|--------|---------------|-----------|
| **N samples** | 10 | 10 |
| **Mean KL** | 0.0002% | 0.0004% |
| **Std Dev** | 0.0001% | 0.0003% |
| **Min** | 0% | 0% |
| **Max** | 0.0004% | 0.001% |
| **Median** | 0.0001% | 0.0003% |

**Statistical Test (Welch's t-test)**:
- t-statistic: -1.895
- p-value: **0.084**
- Significant difference: **NO**

**Notable finding**: StarCoder2 shows ~500× less sensitivity to knockout than Qwen models. This suggests StarCoder2 uses:
- **Different attention patterns** for test marker processing
- **More redundant pathways** that compensate for knocked-out attention
- **Less reliance** on direct marker→function attention edges

#### Window Scan: Layer Compensation Analysis

Testing contiguous window knockout centered at layer 0:

| Window | Layers | Python KL | Rust KL | Ratio |
|--------|--------|-----------|---------|-------|
| Single | 0 only | 0.0004% | 0.0003% | 1.2× |
| +1 | 0-1 | 0.0001% | 0.0005% | 0.2× |
| +2 | 0-2 | 0.0002% | 0.0002% | 1.4× |
| +3 | 0-3 | 0.0003% | 0.0006% | 0.5× |
| +4 | 0-4 | 0.0002% | 0.0017% | 0.1× |
| +5 | 0-5 | 0.0002% | 0.0003% | 0.8× |
| +6 | 0-6 | 0.0002% | 0.0004% | 0.7× |
| +7 | 0-7 | 0.0004% | 0.0007% | 0.5× |
| +8 | 0-8 | 0.0002% | 0.0006% | 0.4× |
| +9 | 0-9 | 0.0001% | 0.0003% | 0.4× |
| +10 | 0-10 | 0.0004% | 0.0003% | 1.1× |

**Key insight - Extreme Redundancy**: Even knocking out 11 layers (0-10) produces KL values in the 0.0001-0.0004% range. StarCoder2's architecture is **completely resilient** to test marker attention knockout, suggesting it uses entirely different mechanisms for processing test context.

---

### google/codegemma-7b-it (28 layers, 16 heads) — Transformer

#### Layer Scan: Finding Causally Important Layers

| Layer | Python KL | Rust KL | Python/Rust Ratio |
|-------|-----------|---------|-------------------|
| **5** | **0.79%** | **0.76%** | **1.04×** |
| 9 | 0.78% | 0.03% | 26× |
| 19 | 0.78% | 0.04% | 20× |
| 1 | 0.01% | **0.77%** | 0.01× |

**Key insight**: Layer 5 shows strong effects for **both languages equally** - different from Qwen where Python dominated. Layer 1 shows the opposite pattern (Rust > Python).

#### Full Experiment at Layer 5 (Highest Combined Effect)

| Metric | Python Doctest | Rust Test |
|--------|---------------|-----------|
| **N samples** | 10 | 10 |
| **Mean KL** | 0.232% | 0.296% |
| **Std Dev** | 0.345% | 0.326% |
| **Min** | 0% | 0% |
| **Max** | 0.79% | 0.77% |
| **Median** | 0.005% | 0.14% |

**Statistical Test (Welch's t-test)**:
- t-statistic: -0.409
- p-value: **0.686**
- Significant difference: **NO**

**Notable finding**: CodeGemma shows **balanced knockout effects** between Python and Rust - both languages are similarly affected. This contrasts with Qwen's Python-heavy pattern and StarCoder2's near-zero sensitivity.

#### Window Scan: Layer Compensation Analysis

Testing contiguous window knockout centered at layer 5:

| Window | Layers | Python KL | Rust KL | Ratio |
|--------|--------|-----------|---------|-------|
| Single | 5 only | 0.79% | 0.76% | 1.0× |
| ±1 | 4-6 | 0.78% | 0.76% | 1.0× |
| ±2 | 3-7 | 0.015% | 0.038% | 0.4× |
| ±3 | 2-8 | 0.77% | 0.76% | 1.0× |
| ±4 | 1-9 | 0.0002% | 0.78% | 0.0× |
| Full | 0-10 | 0.20% | ~0% | **N/A** |
| Full | 0-11 | 0.78% | 0.013% | **60×** |
| Full | 0-12 | 0.14% | 0.013% | 11× |
| Full | 0-13 | 0.22% | 0.013% | 17× |
| Full | 0-14 | 0.21% | 0.029% | 7× |
| Full | 0-15 | 0.21% | ~0% | **14675×** |

**Key insight - Emergent Python Bias**: CodeGemma shows balanced effects at small windows, but when knocking out half the model (0-15), Python shows **14675× higher sensitivity** than Rust. This suggests:
- **Python processing**: Distributed across many layers with high redundancy in later layers
- **Rust processing**: Concentrated in early layers with extreme redundancy when knocked out
- **Asymmetric architecture**: The model's test marker processing differs fundamentally between languages at scale

---

### RWKV/v6-Finch-1B6-HF (24 layers, 32 heads) — Gated-linear RNN

**Architecture note**: RWKV-6 is the first non-transformer model tested. It uses a gated-linear RNN with recurrent state instead of attention matrices. State knockout is the equivalent of transformer all-edge knockout (see Methodology).

#### Layer Scan: Finding Causally Important Layers (State Knockout)

| Layer | Python KL | Rust KL | Python/Rust Ratio |
|-------|-----------|---------|-------------------|
| **2** | **0.336%** | 0.009% | **37×** |
| **22** | **0.327%** | 0.003% | **102×** |
| 9 | 0.275% | 0.008% | 33× |
| 3 | 0.189% | 0.004% | 44× |
| 5 | 0.146% | 0.013% | 11× |

Top Rust layers:

| Layer | Rust KL | Python KL |
|-------|---------|-----------|
| 12 | 0.095% | 0.013% |
| 7 | 0.060% | 0.099% |
| 13 | 0.058% | 0.099% |

**Key insight**: Two distinct Python-sensitive regions — early layers (2-3) and a late peak at layer 22. Rust sensitivity is distributed through the middle layers (7, 12-13). Unlike the transformers, no single layer dominates.

#### Full Experiment at Layer 2 (Highest Python Effect)

| Metric | Python Doctest | Rust Test |
|--------|---------------|-----------|
| **N samples** | 10 | 10 |
| **Mean KL** | 0.111% | 0.024% |
| **Std Dev** | 0.090% | 0.021% |
| **Min** | 0.040% | 0.001% |
| **Max** | 0.336% | 0.063% |
| **Median** | 0.078% | 0.019% |

**Statistical Test (Welch's t-test)**:
- t-statistic: **2.841**
- p-value: **0.018**
- Significant difference: **YES** (first statistically significant result across all models)

**Notable finding**: RWKV-6 is the **only model** where the Python vs Rust difference reaches statistical significance. The Python/Rust mean KL ratio (4.6×) is moderate, but the low variance in both groups drives significance — unlike the transformers where a few high-KL outliers inflate the standard deviation.

#### Per-Sample Results (Layer 2)

**Python Doctest Samples:**

| Sample ID | KL Divergence | Impact Level |
|-----------|---------------|--------------|
| py_simple_add | 0.336% | Moderate |
| py_long_name | 0.213% | Moderate |
| py_complex_params | 0.128% | Moderate |
| py_multiple_doctests | 0.093% | Low |
| py_multi_param | 0.078% | Low |
| py_list_operations | 0.078% | Low |
| py_default_args | 0.054% | Low |
| py_single_char_param | 0.052% | Low |
| py_string_manipulation | 0.044% | Low |
| py_nested_structure | 0.040% | Low |

**Rust Test Samples:**

| Sample ID | KL Divergence | Impact Level |
|-----------|---------------|--------------|
| rust_option_return | 0.063% | Low |
| rust_tuple_return | 0.048% | Low |
| rust_multiple_assertions | 0.047% | Low |
| rust_cfg_test_module | 0.031% | Low |
| rust_reference_params | 0.028% | Low |
| rust_simple_add | 0.009% | Minimal |
| rust_vec_operations | 0.007% | Minimal |
| rust_should_panic | 0.005% | Minimal |
| rust_result_type | 0.003% | Minimal |
| rust_generic_complex | 0.001% | Minimal |

**Observation**: Unlike the transformer models (especially Qwen-3B, CodeGemma), RWKV-6 shows **no extreme outliers**. All Python KL values fall within one order of magnitude (0.04-0.34%), and all Rust values within two (0.001-0.063%). This low variance is what enables statistical significance despite moderate effect sizes.

#### Window Scan: Cumulative Recurrent State Effects

Testing expanding window knockout centered at layer 2:

| Window | Layers | Python KL | Rust KL | Ratio |
|--------|--------|-----------|---------|-------|
| Single | 2 only | 0.336% | 0.009% | 37× |
| ±1 | 1-3 | 0.698% | 0.015% | 48× |
| ±2 | 0-4 | 0.742% | 0.015% | 48× |
| ±3 | 0-5 | 0.833% | 0.019% | 45× |
| ±4 | 0-6 | 1.281% | 0.010% | 127× |
| ±5 | 0-7 | 1.930% | 0.026% | 76× |

**Key insight - Cumulative Effect (No Compensation)**: Unlike transformer models (especially Qwen-7B) where expanding the window can **reduce** the effect through layer compensation, RWKV-6 shows the **opposite**: Python KL **grows monotonically** from 0.34% (single layer) to 1.93% (8 layers). This is a direct consequence of the recurrent architecture — each layer's state carries information from all previous positions, so knocking out more layers compounds the information loss. The Python/Rust ratio also increases, peaking at 127× for layers 0-6.

This contrasts sharply with:
- **Qwen-7B**: Window expansion from layer 2 alone (0.094%) → layers 1-3 (0.025%) = **compensation** (4× decrease)
- **RWKV-6**: Window expansion from layer 2 alone (0.336%) → layers 0-7 (1.930%) = **accumulation** (6× increase)

---

## All-Edge Knockout: Complete Marker Removal (Transformers)

### Motivation

The previous experiments knocked out only **specific edges** from the test marker to function tokens. But what if the marker's influence flows through other pathways? The all-edge knockout tests a stronger intervention:

> **Question**: What happens if subsequent tokens cannot see the test marker at all?

### Methodology

Instead of knocking out `marker → function_tokens`, we knock out **ALL attention TO the marker** from all subsequent positions:

```
SPECIFIC-EDGE KNOCKOUT (previous experiments):
  marker → function_name  ← knocked out
  marker → other_tokens   ← still active

ALL-EDGE KNOCKOUT (this experiment):
  ALL tokens → marker     ← knocked out (marker becomes "invisible")
```

This tests whether the marker's mere **presence in context** (being attended to by any token) affects model behavior.

### Qwen-3B All-Edge Knockout Layer Scan

| Layer | Python KL | Rust KL | Ratio |
|-------|-----------|---------|-------|
| **0** | **153.7%** | **0.19%** | **806×** (Python) |
| **1** | **10.1%** | **0.003%** | **3380×** (Python) |
| 2 | 0.009% | 0.001% | 8× |
| 34 | 0.004% | 0.002% | 2× |

**Key insight - Extreme Python bias in early layers**: Layer 0 shows 806× Python/Rust ratio, Layer 1 shows 3380× ratio. This confirms Python doctests rely heavily on early attention to the `>>>` marker.

### Qwen-7B All-Edge Knockout Layer Scan

| Layer | Python KL | Rust KL | Ratio |
|-------|-----------|---------|-------|
| 0 | 8.60% | 11.80% | ~0.7× (both high - embedding layer) |
| **1** | **6.76%** | **0.004%** | **1878×** |
| 2 | 0.043% | 0.13% | ~0.3× |
| **3** | **0.28%** | **0.002%** | **131×** |
| 25 | 0.046% | 0.0006% | 77× |
| 27 | 0.045% | 0.0008% | 56× |

**Key insight - Layer 1 is the critical Python layer**: When the test marker becomes completely invisible at layer 1, Python shows **1878× higher sensitivity** than Rust. This is dramatically higher than the specific-edge knockout ratios (~189× at layer 2).

### StarCoder2-3B All-Edge Knockout Layer Scan

| Layer | Python KL | Rust KL | Ratio |
|-------|-----------|---------|-------|
| **0** | **0.04%** | **2.81%** | **0.01× (Rust 74× higher!)** |
| 2 | 0.0001% | 0.001% | 0.1× |
| 25 | 0.0001% | 0.0009% | 0.1× |

**Key insight - REVERSED pattern**: StarCoder2 shows the **opposite** of Qwen. At layer 0, Rust is 74× more sensitive to marker knockout than Python. All other layers show near-zero effects for both languages.

This suggests StarCoder2 processes Rust's `#[test]` attribute through attention in layer 0, while Python doctests use entirely different mechanisms (perhaps syntactic/positional).

### CodeGemma-7B All-Edge Knockout Layer Scan

| Layer | Python KL | Rust KL | Ratio |
|-------|-----------|---------|-------|
| **4** | **1.46%** | **0.013%** | **113×** (Python) |
| 11 | 0.78% | 0.16% | 4.8× (Python) |
| 13 | 0.78% | 0.03% | 27× (Python) |
| **15** | **0.30%** | **0.78%** | **0.4× (Rust 2.5× higher)** |
| 27 | 0.78% | ~0% | High (Python) |

**Key insight - Distributed processing with layer-specific biases**:
- **Layer 4**: Strong Python bias (113× ratio)
- **Layer 15**: Rust bias (2.5× ratio) - one of the few layers where Rust > Python
- Effects are distributed across many layers (not concentrated like Qwen)

### Cross-Model All-Edge Knockout Summary

| Model | Architecture | Peak Python Layer | Peak Python KL | Peak Rust Layer | Peak Rust KL | Pattern |
|-------|-------------|-------------------|----------------|-----------------|--------------|---------|
| **Qwen-3B** | Transformer | 0 | **153.7%** | 0 | 0.19% | Extreme Python bias |
| **Qwen-7B** | Transformer | 1 | **6.76%** | 0 | 11.8% | Python bias (L1), shared (L0) |
| **StarCoder2-3B** | Transformer | 0 | 0.04% | **0** | **2.81%** | **Rust bias** (unique!) |
| **CodeGemma-7B** | Transformer | 4 | 1.46% | 15 | 0.78% | Distributed, mixed biases |
| **RWKV-6** | Gated-linear RNN | 2 | 0.336% | 12 | 0.095% | Python bias, distributed |

**Note on RWKV-6**: State knockout is semantically equivalent to all-edge knockout (marker becomes invisible to all future tokens). The RWKV-6 values are from single-layer state knockout; see the RWKV-6 section above for expanding window results up to 1.93% Python KL.

### Interpretation: Why All-Edge > Specific-Edge (Qwen-7B)

| Knockout Type | Python KL | Rust KL | Ratio |
|---------------|-----------|---------|-------|
| **Specific-edge** (marker→function) | 0.081% | 0.002% | ~50× |
| **All-edge** (marker invisible) | 6.76% | 0.004% | **1878×** |

The all-edge knockout effect is **83× stronger** for Python than specific-edge knockout. This reveals:

1. **The marker itself is critical, not just the marker→function edge**: Python relies on the test marker being visible to the entire forward context, not just for copying function tokens.

2. **Rust's marker is essentially decorative**: The `#[test]` attribute has virtually no causal effect on model predictions when removed from attention (0.004% KL).

3. **Different processing mechanisms**: Python doctests are processed through dedicated attention pathways (the marker must be "seen"), while Rust tests work through other mechanisms (perhaps residual connections or syntactic features).

### Commands Reference

```powershell
# All-edge knockout at a specific layer
cargo run --release --example ablation_experiment -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --layer 1 \
    --all-edges

# All-edge knockout layer scan
cargo run --release --example ablation_experiment -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --scan-layers \
    --all-edges
```

---

## Cross-Model Summary

| Model | Architecture | Best Layer | Python KL | Rust KL | Ratio | p-value | Significant? |
|-------|-------------|------------|-----------|---------|-------|---------|--------------|
| Qwen-3B | Transformer | 1 | 0.71% | 0.82% | 0.87× | 0.908 | **NO** |
| Qwen-7B | Transformer | 2 | 0.95% | 0.005% | **189×** | 0.338 | **NO** |
| StarCoder2-3B | Transformer | 0 | 0.0002% | 0.0004% | 0.5× | 0.084 | **NO** |
| CodeGemma-7B | Transformer | 5 | 0.23% | 0.30% | 0.78× | 0.686 | **NO** |
| **RWKV-6** | **Gated-linear RNN** | **2** | **0.111%** | **0.024%** | **4.6×** | **0.018** | **YES** |

### Model Architecture Comparison

| Model | Architecture | Sensitivity | Language Bias | Pattern |
|-------|-------------|-------------|---------------|---------|
| Qwen-3B | Transformer | Medium | None (equal) | Both languages affected equally |
| Qwen-7B | Transformer | High (Python) | **Strong Python** | Layer 2 critical for Python only |
| StarCoder2-3B | Transformer | **Minimal** | None | Uses redundant pathways |
| CodeGemma-7B | Transformer | Medium | None (equal) | Balanced effects |
| RWKV-6 | Gated-linear RNN | Low-Medium | **Python** (p < 0.05) | Consistent Python bias, low variance |

**Key findings**:
- Despite all 4 transformer models showing p > 0.05 (no statistically significant difference), Qwen-7B reveals a **large practical difference**: knocking out layer 2 causes 189× more disruption to Python than Rust. High variance prevents statistical significance.
- **RWKV-6 achieves the only statistically significant result** (p = 0.018). The effect size is moderate (4.6× ratio), but the recurrent architecture produces **consistently low variance** across samples — no extreme outliers like transformer models exhibit. This suggests the recurrent state carries marker information more uniformly than attention heads, which can be highly sample-dependent.

---

## Scientific Interpretation

### The Assembly Line Analogy

Imagine a neural network as a **factory assembly line**:

```
TRANSFORMER:
INPUT TOKENS → [Layer 0] → [Layer 1] → ... → [Layer N] → OUTPUT PREDICTION
                  ↑           ↑                  ↑
              Workers look back at previous parts (attention)

RWKV-6 (RNN):
INPUT TOKENS → [Layer 0] → [Layer 1] → ... → [Layer N] → OUTPUT PREDICTION
                  ↓           ↓                  ↓
              State passed forward through the line (recurrence)
```

**Attention Analysis** (RIGOR_EXPERIMENT): We observed that Worker A (Python `>>>`) spends 2.8-4.4× more time looking at Part X (function tokens) than Worker B (Rust `#[test]`) across 4 code-specialized transformer models.

**Ablation Experiment** (this document): We blindfolded workers (transformers) or blocked parts from being passed forward (RWKV-6) and measured if the assembly line still produces the same output.

**Transformer finding**: Both assembly lines produce equally similar outputs when workers are blindfolded. The attention *pattern* differs, but the *dependence* on that attention is statistically the same (p > 0.05 for all 4 transformers).

**RWKV-6 finding**: Blocking the Python marker from being passed forward has a **significantly larger** effect than blocking the Rust marker (p = 0.018). The recurrent state carries marker information more uniformly, allowing the difference to reach statistical significance.

### Why Might Transformers and RNNs Differ?

1. **Transformer redundancy**: Multiple attention heads can compensate for knocked-out edges
   - Other heads pick up the slack → high per-sample variance
   - Residual connections bypass attention entirely
   - Information already embedded in hidden states

2. **RNN consistency**: State flows through a single recurrent pathway per layer
   - No head-level redundancy → low per-sample variance
   - State knockout at a position affects all downstream tokens uniformly
   - Cumulative effect across layers (no compensation)

3. **Layer specialization**:
   - Transformers: Effect concentrated in 1-2 layers (bottleneck)
   - RWKV-6: Effect distributed, accumulates with more layers knocked out

### Implications for the Paper

**Original claim** (attention analysis):
> "Python doctests show 2.8-4.4× stronger attention to function tokens than Rust tests (p < 0.0002) across 4 code-specialized models"

**Nuanced claim** (after ablation across 5 models):
> "Python doctests show 2.8-4.4× stronger attention correlation in code-specialized transformers, but causal ablation reveals **both languages rely equally** on transformer attention pathways (p > 0.05). However, RWKV-6 state knockout achieves the first statistically significant result (p = 0.018), suggesting the causal asymmetry is real but masked by transformer redundancy."

### What This Means

| Finding | Implication |
|---------|-------------|
| Correlation ≠ Causation (transformers) | High attention doesn't mean necessary attention |
| Both languages equally affected (transformers) | Transformer redundancy masks causal differences |
| RWKV-6 significant (p = 0.018) | Causal asymmetry exists but requires non-redundant architecture to detect |
| Early layers more causal | Layer 2 matters across architectures (Qwen-7B, RWKV-6) |
| Recurrence reveals signal | Low variance in RNNs enables statistical power that transformers lack |

---

## Comparison: Correlation vs Causation

| Analysis | Architecture | What It Measures | Python | Rust | Ratio | p-value |
|----------|-------------|------------------|--------|------|-------|---------|
| **Attention Strength** (Qwen-7B, L16) | Transformer | How much marker "looks at" function | 9.08% | 2.59% | **3.51×** | **0.000003** |
| **Knockout Effect** (Qwen-3B, L1) | Transformer | How much prediction changes when attention removed | 0.71% | 0.82% | **0.87×** | 0.908 |
| **State Knockout** (RWKV-6, L2) | Gated-linear RNN | How much prediction changes when state removed | 0.111% | 0.024% | **4.6×** | **0.018** |

**Key insight**: Strong attention correlation does NOT imply strong causal dependence in transformers (all 4 models p > 0.05). However, state knockout in RWKV-6 — which is semantically equivalent to all-edge knockout — **does** show significant causal asymmetry (p = 0.018). This suggests the underlying causal difference between Python and Rust marker processing is real, but transformer redundancy prevents it from reaching significance.

---

## Limitations

1. **Single-edge knockout (transformers only)**: Transformer experiments knock out marker→function edges. The model may use other edges for the same information. (RWKV-6 state knockout is inherently all-edge.)

2. **Layer-by-layer**: Primary experiments knock out one layer at a time. Window scans partially address this.

3. **Sample size**: 10 samples per language. More samples would reduce variance and likely reveal significance in more models.

4. **Model scale**: Tested on 1.6B-7B models. Larger models may show different patterns.

5. **Ablation scope**: Ablation experiments cover 4 code-specialized transformers (Qwen-3B, Qwen-7B, StarCoder2-3B, CodeGemma-7B) and 1 non-code-specialized RNN (RWKV-6). Code-LLaMA-7B and Phi-3-mini — which show no significant attention effect (RIGOR_EXPERIMENT.md) — have not been tested with ablation. Extending ablation to these models would test whether the causal patterns differ in non-code-specialized transformers.

6. **Position granularity**: Character-to-token conversion may miss some edges.

7. **RWKV-6 is not code-specialized**: Unlike the 4 transformer models (all trained on code), RWKV-6 (World model, multilingual) was not specifically trained on code. The significant result may partly reflect the model's weaker code understanding rather than a fundamental architectural difference. Testing a code-specialized RNN (when available) would disambiguate.

---

## Future Work

1. ~~**Multi-layer knockout**: Knock out layers 0-5 simultaneously~~ ✅ **DONE** — Window scan implemented
2. ~~**All-edge knockout**: Remove ALL attention from marker token (not just to functions)~~ ✅ **DONE** — Shows 1878× Python/Rust ratio at layer 1 on Qwen-7B
3. ~~**Non-transformer model**: Test equivalent ablation on a non-attention architecture~~ ✅ **DONE** — RWKV-6 state knockout achieves first significant result (p = 0.018)
4. ~~**Amplification experiment**: Boost attention (the opposite of knockout) to see if preservation improves~~ ✅ **DONE** — RWKV-6 state steering generation tested at scales 1-9× with n=30 samples. **Null result**: amplification has no effect on generation output. The marker write is *necessary* (knockout p=0.018) but *not sufficient* (amplification indistinguishable from baseline). See [STEERING_RESULTS.md](STEERING_RESULTS.md) Section 8.
5. **Non-code-specialized transformers**: Run ablation on Code-LLaMA-7B and Phi-3-mini to test whether the causal independence also holds in models that show no attention correlation effect
6. **RWKV-6 effective attention** (Phase 5): Compute effective attention matrices for RWKV-6 to enable direct attention analysis (not just knockout)
7. **Larger models**: Test on 30B+ models if hardware permits
8. **Different test patterns**: pytest, unittest, Go tests, etc.
9. **Head-specific knockout**: Test which attention heads within a layer are most important (transformers only)

---

## Commands Reference

### Transformer Attention Knockout (`ablation_experiment`)

```powershell
# Layer scan (find most causally important layer)
cargo run --release --example ablation_experiment -- --scan-layers

# Full experiment at specific layer
cargo run --release --example ablation_experiment -- --layer 1 --verbose

# Save results to JSON
cargo run --release --example ablation_experiment -- --layer 1 --output outputs/results.json

# Different model
cargo run --release --example ablation_experiment -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --layer 2 \
    --verbose

# Window scan (test layer compensation around a center layer)
cargo run --release --example ablation_experiment -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --scan-windows \
    --window-center 2 \
    --max-radius 10

# Contiguous layer window knockout
cargo run --release --example ablation_experiment -- \
    --layer-start 0 \
    --layer-end 4 \
    --verbose

# Include baseline samples (non-test code)
cargo run --release --example ablation_experiment -- --layer 1 --include-baselines

# Sliding window scan (slide fixed-size window across all layers)
cargo run --release --example ablation_experiment -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --slide-window 3

# All-edge knockout (marker becomes invisible to subsequent tokens)
cargo run --release --example ablation_experiment -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --layer 1 \
    --all-edges

# All-edge knockout layer scan
cargo run --release --example ablation_experiment -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --scan-layers \
    --all-edges
```

### RWKV-6 State Knockout (`state_ablation_experiment`)

```powershell
# Layer scan (all 24 layers)
cargo run --release --example state_ablation_experiment -- --scan-layers

# Full experiment at specific layer
cargo run --release --example state_ablation_experiment -- --layer 2 --verbose

# Save results to JSON
cargo run --release --example state_ablation_experiment -- \
    --layer 2 \
    --output outputs/state_ablation_layer2.json

# Window scan (test cumulative effect around a center layer)
cargo run --release --example state_ablation_experiment -- \
    --scan-windows \
    --window-center 2 \
    --max-radius 5

# Sliding window scan
cargo run --release --example state_ablation_experiment -- --slide-window 3

# Contiguous layer window knockout
cargo run --release --example state_ablation_experiment -- \
    --layer-start 0 \
    --layer-end 7

# Include baseline samples
cargo run --release --example state_ablation_experiment -- --layer 2 --include-baselines
```

---

## Files

- `examples/ablation_experiment.rs` — Transformer attention knockout experiment
- `examples/state_ablation_experiment.rs` — RWKV-6 state knockout experiment
- `src/intervention.rs` — Knockout infrastructure (`KnockoutSpec`, `StateKnockoutSpec`, `create_knockout_mask`)
- `corpus/attention_samples_universal.json` — Test corpus with character positions (shared by both experiments)
- `outputs/ablation_layer1_full.json` — Qwen-3B results at layer 1
- `outputs/state_ablation_layer2.json` — RWKV-6 results at layer 2

---

*Updated: February 10, 2026 (amplification experiment marked done — null result, see STEERING_RESULTS.md Section 8)*
*Last transformer ablation run: February 1, 2026 — All-edge knockout layer scan on 4 code-specialized models*
*Last state knockout run: February 9, 2026 — RWKV-6 layer scan + full experiment at layer 2 (p = 0.018, first significant result)*
*Note: Ablation experiments have not been extended to Code-LLaMA-7B or Phi-3-mini (added in v1.1.0). See RIGOR_EXPERIMENT.md for their attention analysis results.*
*For: AIWare 2026 submission*
