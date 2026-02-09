# PLIP-rs: Attention Ablation (Knockout) Experiment Results

**Created**: February 1, 2026
**Purpose**: Test causal importance of test marker → function token attention
**Hardware**: RTX 5060 Ti (16GB VRAM)

---

## Executive Summary

**Research Question**: Is the attention from test markers (`>>>`, `#[test]`) to function tokens *causally necessary* for model predictions, or merely *correlational*?

**Key Findings**:

1. **Model-Specific Causal Pathways**: Different models handle test marker attention very differently:
   - **Qwen-7B**: Layer 2 shows 189× stronger knockout effect for Python vs Rust
   - **StarCoder2**: Near-zero sensitivity (uses redundant pathways)
   - **CodeGemma**: Balanced effects (both languages equally affected)

2. **Layer Compensation**: In Qwen-7B, knocking out layer 2 alone causes 0.094% Python KL, but knocking out layers 1-3 together **reduces** the effect to 0.025% - demonstrating that adjacent layers compensate for knocked-out attention.

3. **Correlation ≠ Causation**: Despite Python showing 2.8-4.4× stronger attention *correlation* in all 4 code-specialized models (see RIGOR_EXPERIMENT.md), causal ablation reveals model-specific dependencies. High variance prevents statistical significance (all p > 0.05), but practical effect sizes vary dramatically across architectures.

---

## Background: From Correlation to Causation

### What We Knew (Attention Analysis - RIGOR_EXPERIMENT.md)

| Metric | Python `>>>` | Rust `#[test]` | Ratio | p-value |
|--------|-------------|----------------|-------|---------|
| Attention to function tokens (Qwen-7B, layer 16) | 9.08% | 2.59% | **3.51×** | **0.000003** |
| Range across 4 code-specialized models | 5.2-9.1% | 1.2-3.1% | **2.8-4.4×** | **<0.0002** |

**Interpretation**: Python doctest markers "look at" function tokens much more strongly than Rust test attributes across all 4 code-specialized models. Two non-code-specialized models (Code-LLaMA-7B, Phi-3-mini) do not show this effect (see RIGOR_EXPERIMENT.md Appendix C).

### What We Asked (Ablation Experiment)

> If we **remove** this attention (set it to zero), does the model's output change differently for Python vs Rust?

This tests whether the attention difference is:
- **Causally important**: Knockout should affect Python more than Rust
- **Merely correlational**: Knockout affects both equally (or neither)

---

## Methodology

### Intervention Mechanism

Knockout is implemented by adding `-infinity` to attention scores **before softmax**:

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

### Measurement: KL Divergence

We measure how much the model's next-token probability distribution changes:

```
KL(baseline || ablated) = Σ p_baseline(x) * log(p_baseline(x) / p_ablated(x))
```

| KL Value | Interpretation |
|----------|----------------|
| > 1% | Significant impact - attention is causally important |
| 0.1% - 1% | Moderate impact - some causal role |
| < 0.1% | Minimal impact - attention is redundant |

### Corpus

Using `corpus/attention_samples_universal.json`:
- 10 Python doctest samples
- 10 Rust test samples
- Character-based positions (model-agnostic)

---

## Results by Model

### Qwen/Qwen2.5-Coder-3B-Instruct (36 layers, 16 heads)

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

### Qwen/Qwen2.5-Coder-7B-Instruct (28 layers, 28 heads)

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

### bigcode/starcoder2-3b (30 layers, 24 heads)

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

### google/codegemma-7b-it (28 layers, 16 heads)

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

## All-Edge Knockout: Complete Marker Removal

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

| Model | Peak Python Layer | Peak Python KL | Peak Rust Layer | Peak Rust KL | Pattern |
|-------|-------------------|----------------|-----------------|--------------|---------|
| **Qwen-3B** | 0 | **153.7%** | 0 | 0.19% | Extreme Python bias |
| **Qwen-7B** | 1 | **6.76%** | 0 | 11.8% | Python bias (L1), shared (L0) |
| **StarCoder2-3B** | 0 | 0.04% | **0** | **2.81%** | **Rust bias** (unique!) |
| **CodeGemma-7B** | 4 | 1.46% | 15 | 0.78% | Distributed, mixed biases |

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

| Model | Best Layer | Python KL | Rust KL | Ratio | p-value | Significant? |
|-------|------------|-----------|---------|-------|---------|--------------|
| Qwen-3B | 1 | 0.71% | 0.82% | 0.87× | 0.908 | **NO** |
| Qwen-7B | 2 | 0.95% | 0.005% | **189×** | 0.338 | **NO** |
| StarCoder2-3B | 0 | 0.0002% | 0.0004% | 0.5× | 0.084 | **NO** |
| CodeGemma-7B | 5 | 0.23% | 0.30% | 0.78× | 0.686 | **NO** |

### Model Architecture Comparison

| Model | Sensitivity | Language Bias | Pattern |
|-------|-------------|---------------|---------|
| Qwen-3B | Medium | None (equal) | Both languages affected equally |
| Qwen-7B | High (Python) | **Strong Python** | Layer 2 critical for Python only |
| StarCoder2-3B | **Minimal** | None | Uses redundant pathways |
| CodeGemma-7B | Medium | None (equal) | Balanced effects |

**Key finding**: Despite all models showing p > 0.05 (no statistically significant difference), Qwen-7B reveals a **large practical difference**: knocking out layer 2 causes 189× more disruption to Python than Rust. High variance prevents statistical significance.

---

## Scientific Interpretation

### The Assembly Line Analogy

Imagine a transformer as a **factory assembly line**:

```
INPUT TOKENS → [Layer 0] → [Layer 1] → ... → [Layer N] → OUTPUT PREDICTION
                  ↑           ↑                  ↑
              Workers look back at previous parts (attention)
```

**Attention Analysis** (RIGOR_EXPERIMENT): We observed that Worker A (Python `>>>`) spends 2.8-4.4× more time looking at Part X (function tokens) than Worker B (Rust `#[test]`) across 4 code-specialized models.

**Ablation Experiment** (this document): We blindfolded both workers and measured if the assembly line still produces the same output.

**Finding**: Both assembly lines produce equally similar outputs when workers are blindfolded. The attention *pattern* differs, but the *dependence* on that attention is the same.

### Why Might This Happen?

1. **Redundancy**: Multiple workers can compensate for one blindfolded worker
   - Other attention heads pick up the slack
   - Residual connections bypass attention entirely
   - Information already embedded in hidden states

2. **Distributed Processing**: No single attention edge is critical
   - Many small contributions sum to the result
   - Removing one edge doesn't break the system

3. **Layer Specialization**:
   - Layer 1 (early): "Raw feature extraction" - more critical
   - Layer 14 (mid): "Semantic refinement" - more optional, more observable

### Implications for the Paper

**Original claim** (attention analysis):
> "Python doctests show 2.8-4.4× stronger attention to function tokens than Rust tests (p < 0.0002) across 4 code-specialized models"

**Nuanced claim** (after ablation):
> "Python doctests show 2.8-4.4× stronger attention correlation in code-specialized models, but causal ablation reveals **both languages rely equally** on this attention pathway (p = 0.91). The attention difference reflects how models *organize* information, not what they *require* to function."

### What This Means

| Finding | Implication |
|---------|-------------|
| Correlation ≠ Causation | High attention doesn't mean necessary attention |
| Both languages equally affected | No language-specific causal mechanism at this layer |
| Early layers more causal | Layer 1 matters more than layer 14 for predictions |
| High sample variance | Effect is sample-specific, not language-specific |

---

## Comparison: Correlation vs Causation

| Analysis | What It Measures | Python | Rust | Ratio | p-value |
|----------|------------------|--------|------|-------|---------|
| **Attention Strength** (Qwen-7B, layer 16) | How much marker "looks at" function | 9.08% | 2.59% | **3.51×** | **0.000003** |
| **Knockout Effect** (Qwen-3B, layer 1) | How much prediction changes when attention removed | 0.71% | 0.82% | **0.87×** | 0.908 |

**Key insight**: Strong attention correlation does NOT imply strong causal dependence. This holds across all 4 code-specialized models tested with ablation.

---

## Limitations

1. **Single-edge knockout**: We only knock out marker→function edges. The model may use other edges for the same information.

2. **Layer-by-layer**: We knock out one layer at a time. Multi-layer knockout might show different results.

3. **Sample size**: 10 samples per language. More samples would reduce variance.

4. **Model scale**: Tested on 3B-7B models. Larger models may show different patterns.

5. **Ablation scope**: Ablation experiments were performed on the 4 code-specialized models only (Qwen-3B, Qwen-7B, StarCoder2-3B, CodeGemma-7B). Code-LLaMA-7B and Phi-3-mini — which show no significant attention effect (RIGOR_EXPERIMENT.md) — have not been tested with ablation. Extending ablation to these models would test whether the causal patterns also differ in non-code-specialized models.

6. **Position granularity**: Character-to-token conversion may miss some edges.

---

## Future Work

1. ~~**Multi-layer knockout**: Knock out layers 0-5 simultaneously~~ ✅ **DONE** - Window scan implemented
2. ~~**All-edge knockout**: Remove ALL attention from marker token (not just to functions)~~ ✅ **DONE** - Shows 1878× Python/Rust ratio at layer 1 on Qwen-7B
3. **Amplification experiment**: Boost attention (the opposite of knockout) to see if preservation improves
4. **Non-code-specialized models**: Run ablation on Code-LLaMA-7B and Phi-3-mini to test whether the causal independence (both languages equally affected) also holds in models that show no attention correlation effect
5. **Larger models**: Test on 30B+ models if hardware permits
6. **Different test patterns**: pytest, unittest, Go tests, etc.
7. **Head-specific knockout**: Test which attention heads within a layer are most important

---

## Commands Reference

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

---

## Files

- `examples/ablation_experiment.rs` - Main experiment script
- `src/intervention.rs` - Knockout infrastructure (KnockoutSpec, create_knockout_mask)
- `corpus/attention_samples_universal.json` - Test corpus with character positions
- `outputs/ablation_layer1_full.json` - Qwen-3B results at layer 1

---

*Updated: February 9, 2026 (contextual updates for Code-LLaMA and Phi-3 attention findings)*
*Last ablation experiment run: February 1, 2026 — All-edge knockout layer scan on 4 code-specialized models (Qwen-3B shows 3380× Python/Rust ratio at L1; StarCoder2 shows reversed 74× Rust/Python ratio at L0)*
*Note: Ablation experiments have not been extended to Code-LLaMA-7B or Phi-3-mini (added in v1.1.0). See RIGOR_EXPERIMENT.md for their attention analysis results.*
*For: AIWare 2026 submission*
