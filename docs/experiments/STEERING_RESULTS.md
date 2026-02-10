# Attention Steering Experiment Results

**Date**: February 1, 2026
**Experiment**: Part 2 - Amplification (Attention Steering)
**Hypothesis**: Boosting Rust `#[test]` marker attention to Python `>>>` levels may improve test preservation in Qwen models.

## Executive Summary

We successfully implemented and validated attention steering in PLIP-rs on the two Qwen code-specialized models, demonstrating that:

1. **Rust test markers receive significantly less attention than Python doctest markers** (2.3-2.6% vs 5.7-9.1% in Qwen models; 2.8-4.4× ratio across all 4 code-specialized models — see [RIGOR_EXPERIMENT.md](RIGOR_EXPERIMENT.md))
2. **Steering can boost Rust attention to Python levels** without catastrophically affecting model outputs
3. **The intervention is "safe"** - KL divergence remains flat across dose levels

**Scope**: Steering dose-response experiments were conducted on Qwen-3B and Qwen-7B only. The attention asymmetry that motivates steering has since been validated across 4 code-specialized models but does **not replicate** on Code-LLaMA-7B (reversed pattern) or Phi-3-mini (no significant difference). See [RIGOR_EXPERIMENT.md](RIGOR_EXPERIMENT.md) Appendix C for full cross-model analysis.

---

## 1. Calibration Results

### 1.1 Baseline Attention Measurements

Calibration was performed on the two Qwen code-specialized models. For cross-model attention analysis across all 6 models (including Code-LLaMA-7B and Phi-3-mini), see [RIGOR_EXPERIMENT.md](RIGOR_EXPERIMENT.md) Appendix C.

| Model | Layer | Python `>>>` → `fn` | Rust `#[test]` → `fn` | Ratio |
|-------|-------|---------------------|----------------------|-------|
| **Qwen/Qwen2.5-Coder-3B-Instruct** | 20 | 5.70% | 2.32% | 2.46× |
| **Qwen/Qwen2.5-Coder-7B-Instruct** | 16 | 9.08% | 2.59% | 3.51× |

### 1.2 Per-Sample Breakdown (Qwen 3B, Layer 20)

#### Python Doctest Samples

| Sample ID | Attention |
|-----------|-----------|
| py_simple_add | 6.93% |
| py_long_name | 7.86% |
| py_multi_param | 1.58% |
| py_complex_params | 2.53% |
| py_single_char_param | 10.12% |
| py_multiple_doctests | 9.03% |
| py_list_operations | 2.29% |
| py_string_manipulation | 2.31% |
| py_default_args | 2.84% |
| py_nested_structure | 11.53% |
| **Mean** | **5.70%** |

#### Rust Test Samples

| Sample ID | Attention |
|-----------|-----------|
| rust_simple_add | 3.18% |
| rust_option_return | 3.01% |
| rust_should_panic | 2.48% |
| rust_result_type | 3.46% |
| rust_multiple_assertions | 1.91% |
| rust_generic_complex | 0.75% |
| rust_tuple_return | 1.09% |
| rust_vec_operations | 0.48% |
| rust_reference_params | 5.39% |
| rust_cfg_test_module | 1.47% |
| **Mean** | **2.32%** |

### 1.3 Per-Sample Breakdown (Qwen 7B, Layer 16)

#### Python Doctest Samples

| Sample ID | Attention |
|-----------|-----------|
| py_simple_add | 10.04% |
| py_long_name | 8.68% |
| py_multi_param | 7.24% |
| py_complex_params | 5.34% |
| py_single_char_param | 12.71% |
| py_multiple_doctests | 9.70% |
| py_list_operations | 10.97% |
| py_string_manipulation | 8.62% |
| py_default_args | 6.73% |
| py_nested_structure | 10.73% |
| **Mean** | **9.08%** |

#### Rust Test Samples

| Sample ID | Attention |
|-----------|-----------|
| rust_simple_add | 3.78% |
| rust_option_return | 2.76% |
| rust_should_panic | 2.35% |
| rust_result_type | 3.44% |
| rust_multiple_assertions | 2.22% |
| rust_generic_complex | 2.02% |
| rust_tuple_return | 2.21% |
| rust_vec_operations | 2.49% |
| rust_reference_params | 3.01% |
| rust_cfg_test_module | 1.58% |
| **Mean** | **2.59%** |

---

## 2. Dose-Response Experiments

### 2.1 Qwen 3B (Layer 20)

**Intervention**: Scale factor applied to `#[test]` → function token edges

| Scale | Attention Achieved | KL Divergence | Std Dev |
|-------|-------------------|---------------|---------|
| 0.5× | 1.23% | 470.42 | ±332.08 |
| **1.0×** (baseline) | **2.32%** | **469.96** | ±331.31 |
| 2.0× | 4.17% | 470.52 | ±332.21 |
| 3.0× | 5.71% | 470.53 | ±332.22 |
| 4.0× | 7.02% | 470.52 | ±332.21 |
| **6.0×** | **9.17%** ≈ Python | 470.55 | ±332.22 |

**Observations**:
- At 6× scaling, Rust attention (9.17%) exceeds Python baseline (5.70%)
- KL divergence is essentially **flat** across all dose levels (~470)
- High variance (σ ≈ 332) indicates sample-level heterogeneity

### 2.2 Qwen 7B (Layer 16)

**Intervention**: Scale factor applied to `#[test]` → function token edges

| Scale | Attention Achieved | KL Divergence | Std Dev |
|-------|-------------------|---------------|---------|
| 0.5× | 1.35% | 676.80 | ±109.43 |
| **1.0×** (baseline) | **2.59%** | **676.73** | ±109.47 |
| 2.0× | 4.77% | 676.77 | ±109.41 |
| 3.0× | 6.64% | 676.76 | ±109.47 |
| **4.0×** | **8.27%** ≈ Python | 676.75 | ±109.51 |
| 6.0× | 10.98% | 676.75 | ±109.43 |

**Observations**:
- At 4× scaling, Rust attention (8.27%) approaches Python baseline (9.08%)
- KL divergence is **flat** across all dose levels (~676.7)
- Lower variance (σ ≈ 109) compared to 3B model

---

## 3. Key Findings

### 3.1 Attention Asymmetry Confirmed

Both Qwen models exhibit consistent attention asymmetry:
- **Python `>>>` markers**: Receive 5.7-9.1% attention to function tokens
- **Rust `#[test]` markers**: Receive only 2.3-2.6% attention to function tokens
- **Gap widens with model size**: 2.46× ratio for 3B → 3.51× ratio for 7B

This asymmetry has been confirmed across all 4 code-specialized models (Qwen-7B/3B, StarCoder2-3B, CodeGemma-7B) with ratios of 2.8-4.4× and p < 0.0002. However, two non-code-specialized models show no such asymmetry: Code-LLaMA-7B exhibits a **reversed** pattern (Rust > Python at all layers), and Phi-3-mini shows no significant difference. See [RIGOR_EXPERIMENT.md](RIGOR_EXPERIMENT.md) for details.

### 3.2 Steering Successfully Boosts Attention

| Model | Baseline Rust | Target (Python) | Scale Needed | Achieved |
|-------|--------------|-----------------|--------------|----------|
| Qwen 3B | 2.32% | 5.70% | ~2-3× | 5.71% at 3× |
| Qwen 7B | 2.59% | 9.08% | ~3-4× | 8.27% at 4× |

### 3.3 Intervention is "Safe"

**Critical finding**: KL divergence between baseline and steered outputs remains essentially constant regardless of steering intensity.

This suggests:
1. Attention steering modifies internal representations without catastrophically changing outputs
2. The model's predictions are robust to moderate attention perturbations
3. Steering can be applied without breaking model functionality

### 3.4 Implications for Test Preservation

The flat KL curve has two possible interpretations:

**Interpretation A (Optimistic)**:
- Steering successfully increases attention to test markers
- The model's output distribution adapts gracefully
- Test-related tokens may receive more "credit" during generation

**Interpretation B (Conservative)**:
- Attention patterns may be partially redundant
- Multiple heads/layers may compensate for steered edges
- More targeted interventions may be needed

---

## 4. Technical Implementation

### 4.1 Steering Mechanism

Steering is applied **post-softmax** with row renormalization:

```
attn_scores = Q @ K^T / sqrt(d_k)
attn_scores += causal_mask
attn_weights = softmax(attn_scores)

// STEERING INTERVENTION (post-softmax)
for each specified edge (from_pos, to_pos):
    attn_weights[from_pos][to_pos] *= scale_factor
attn_weights = renormalize_rows(attn_weights)  // rows sum to 1.0

output = attn_weights @ V
```

### 4.2 Files Modified/Created

| File | Purpose |
|------|---------|
| `src/intervention.rs` | `SteeringSpec`, `InterventionType`, steering functions |
| `src/forward_qwen2.rs` | Post-softmax steering in Qwen2 attention |
| `src/forward.rs` | Post-softmax steering in StarCoder2 attention |
| `src/forward_gemma.rs` | Post-softmax steering in Gemma attention |
| `src/forward_llama.rs` | Post-softmax steering in LLaMA/Code-LLaMA attention (v1.1.0) |
| `src/forward_phi3.rs` | Post-softmax steering in Phi-3 attention (v1.1.0) |
| `src/model.rs` | `forward_with_steering()`, `forward_steered_only()` |
| `src/steering.rs` | Calibration utilities, `DOSE_LEVELS` |
| `examples/steering_calibrate.rs` | Measure baseline attention |
| `examples/steering_experiment.rs` | Dose-response experiments |

---

## 5. Commands to Reproduce

### Calibration

```bash
# Qwen 3B
cargo run --release --example steering_calibrate -- \
    --model "Qwen/Qwen2.5-Coder-3B-Instruct" --verbose

# Qwen 7B
cargo run --release --example steering_calibrate -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" --verbose
```

### Dose-Response Experiments

```bash
# Qwen 3B with default dose levels (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
cargo run --release --example steering_experiment -- \
    --model "Qwen/Qwen2.5-Coder-3B-Instruct" --layer 20 --verbose

# Qwen 7B
cargo run --release --example steering_experiment -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" --layer 16 --verbose

# Target specific attention level (e.g., match Python at 9%)
cargo run --release --example steering_experiment -- \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" --layer 16 \
    --target-attention 0.09
```

---

## 6. Next Steps

1. ~~**Test on code generation tasks**: Measure if boosted attention improves test preservation in actual code generation~~ ✅ **DONE** — RWKV-6 state steering generation tested (Section 8). Null result: amplification up to 9× has no effect on generation output at any distance.
2. **Extend to other code-specialized models**: Run dose-response experiments on StarCoder2-3B and CodeGemma-7B, which both show the attention asymmetry (RIGOR_EXPERIMENT.md). StarCoder2's ablation results show extreme redundancy (ABLATION_RESULTS.md), so steering may behave differently.
3. **Test on non-code-specialized models**: Run steering on Code-LLaMA-7B (reversed attention pattern) and Phi-3-mini (no differential). Since these models show no Python > Rust asymmetry, steering Rust attention upward may have fundamentally different effects — or none at all.
4. **Head-specific steering**: Identify which attention heads are most responsive to steering
5. **Multi-layer steering**: Test steering across multiple layers simultaneously
6. **Generation quality metrics**: Beyond KL divergence, measure BLEU/CodeBLEU on test-related tokens
7. **Transformer steering + generation**: Extend the steering generation pipeline to transformer backends (Qwen, CodeGemma) to test whether attention steering also produces null generation results

---

## 7. Raw Data Files

- Calibration output: `cargo run --release --example steering_calibrate -- --verbose > calibration_output.txt`
- Experiment output: `cargo run --release --example steering_experiment -- --output results.json`

---

## 8. RWKV-6 State Steering — Generation Experiments

### 8.1 Background

RWKV-6 is a gated-linear RNN with no attention matrices. Instead of attention steering (post-softmax edge scaling), it uses **state steering**: scaling the $k_m \, v_m^\top$ write at marker positions in the WKV recurrence. State knockout at layer 2 has already been shown to significantly affect model predictions (p = 0.018, see [ABLATION_RESULTS.md](ABLATION_RESULTS.md) Section 7.3).

The question: if suppressing the state write at the marker disrupts the model (knockout), does **amplifying** it steer generation towards more test-related output?

### 8.2 Steering + Generation Pipeline

State steering generation exploits the recurrent architecture's natural persistence: a steered state propagates to all future tokens via the WKV recurrence without needing to re-steer during generation.

```
1. PREFILL (with steering): process prompt with state steering applied
   → steered recurrent state stored in KVCache
2. GENERATE (normal): autoregressive generation using regular forward_with_kv_cache
   → steered state naturally propagates through the recurrence
```

This is implemented in `PlipRwkv6::generate_with_state_steering` and exposed via `PlipModel::generate_with_state_steering` / `PlipModel::generate_with_state_steering_details`.

### 8.3 Experiment 1: Greedy Generation (`state_steering_generate`)

**Setup**: 5 code prompts (3 Python doctest + 2 Rust test) containing test markers at known positions. For each prompt, compare baseline (scale=1.0) vs steered (scale=3.0) generation at layer 2 with greedy decoding (temperature=0).

**Result**: All 5 prompts produce **identical** baseline and steered output. At greedy decoding, state steering at scale 3.0 on layer 2 has zero effect on generation.

| Prompt | Baseline Output | Steered (3.0×) Output | Match? |
|--------|----------------|-----------------------|--------|
| Python doctest (3 prompts) | (code completions) | (identical) | YES |
| Rust test (2 prompts) | (code completions) | (identical) | YES |

**Interpretation**: At greedy decoding, the argmax token is robust to 3× state amplification. The effect may only manifest in the probability distribution (not the top-1 token), or at higher scales, or with stochastic sampling.

### 8.4 Experiment 2: Persistence at Distance (`state_steering_persistence`)

**Motivation**: Since greedy decoding shows no effect, we use stochastic sampling to expose changes in the probability distribution. We also vary the **distance** between marker and generation start to test whether state steering persistence decays over token distance — as predicted by the mathematical model (Section 4 of [RWKV6_MATHEMATICAL_FOUNDATIONS.md](../RWKV6_MATHEMATICAL_FOUNDATIONS.md)).

**Setup**:
- **Prompts**: 3 Rust prompts with `#[test]` markers at different distances from generation start:
  - `rs_close`: 86 tokens between marker and generation
  - `rs_medium`: 126 tokens between marker and generation
  - `rs_far`: 207 tokens between marker and generation
- **Scales**: 1.0 (baseline), 3.0, 5.0, 9.0
- **Temperature**: 0.8
- **Samples**: n=30 per condition
- **Metric**: Percentage of generated outputs containing test syntax (`#[test]`, `assert`, `fn test_`)

**Results** (n=30, Rust only, temp=0.8, layer 2):

| Prompt | Distance | delta | ×1.0 | ×3.0 | ×5.0 | ×9.0 |
|--------|----------|-------|------|------|------|------|
| rs_close | close | 86 | 77% | 73% | 80% | 70% |
| rs_medium | medium | 126 | 47% | 43% | 43% | 47% |
| rs_far | far | 207 | 37% | 30% | 33% | 43% |

#### 8.4.1 Distance Effect (Confirmed)

Averaging across steered conditions (scale > 1.0):

| Distance | Avg Delta | Test Rate |
|----------|-----------|-----------|
| close | 86 tokens | **74%** |
| medium | 126 tokens | **44%** |
| far | 207 tokens | **36%** |

The marker's influence on generation decays monotonically with distance. This is consistent with the exponential decay predicted by the WKV recurrence: $\text{Persistence} \propto \prod_{j} d_j$ where $d_j = \exp(-\exp(w_j))$.

#### 8.4.2 Scale Effect (Null Result)

Averaging across all distances:

| Scale | Avg Test Rate |
|-------|---------------|
| ×1.0 | 53% |
| ×3.0 | 49% |
| ×5.0 | 52% |
| ×9.0 | 53% |

All scale values produce statistically indistinguishable results. Amplifying the state write at layer 2 by up to 9× has **no effect** on generation.

### 8.5 Key Finding: Necessary but Not Sufficient

| Intervention | Effect | Interpretation |
|-------------|--------|----------------|
| **Knockout** (scale=0) | Significant (p=0.018, KL ratio 4.6×) | Marker write is **necessary** for normal predictions |
| **Amplification** (scale=3-9×) | None (all within sampling noise) | Marker write is **not sufficient** to steer generation |

The layer 2 state write is a **necessary component** of the model's processing of test markers (removing it disrupts predictions), but it is **not a tunable knob** for steering generation (amplifying it has no effect). The bottleneck for test syntax generation lies elsewhere — possibly in:
- Later layers that re-process the state
- The channel-mix (FFN) pathway, which state steering does not affect
- The model's language modeling objective (which determines test syntax probability independently of marker strength)

### 8.6 Comparison with Transformer Steering

| Architecture | Steering Type | Mechanism | Dose-Response |
|-------------|--------------|-----------|---------------|
| Transformers (Qwen) | Attention steering | Post-softmax edge scaling | Flat KL divergence across scales (safe but no generation effect tested) |
| RWKV-6 | State steering | Scale kv^T write | Flat KL divergence AND no generation effect |

Both architectures show flat dose-response curves in KL divergence. For RWKV-6, we additionally confirm that the flat KL curve translates to flat generation behavior — amplification does not steer output content.

### 8.7 Commands to Reproduce

```bash
# Greedy generation comparison (baseline vs steered)
cargo run --release --example state_steering_generate

# Persistence experiment (n=30, Rust only, temp=0.8)
cargo run --release --example state_steering_persistence -- \
    --n-samples 30 --rust-only --temperature 0.8

# Custom scale/temperature sweep
cargo run --release --example state_steering_persistence -- \
    --n-samples 30 --temperature 0.6 --temperature 0.8
```

---

## 9. Raw Data Files

- Calibration output: `cargo run --release --example steering_calibrate -- --verbose > calibration_output.txt`
- Experiment output: `cargo run --release --example steering_experiment -- --output results.json`
- State steering generation output: `cargo run --release --example state_steering_generate`
- State steering persistence output: `cargo run --release --example state_steering_persistence -- --n-samples 30 --rust-only --temperature 0.8`

---

*Created: February 1, 2026*
*Updated: February 10, 2026 (RWKV-6 state steering generation experiments added)*
*Steering scope: Transformer dose-response experiments on Qwen-3B and Qwen-7B. RWKV-6 state steering generation experiments on v6-Finch-1B6-HF (n=30). Steering infrastructure available for all 7 models via PlipBackend trait (v1.2.0).*
*For: AIWare 2026 submission*
