# Ablation & Amplification Roadmap for PLIP-RS

**Date:** February 1, 2026
**Motivation:** Scientific method requires not just *observing* a correlation, but *testing* it via removal (ablation) and enhancement (amplification)

---

## Scientific Context

Your biologist friend identified two missing steps in the research methodology:

| Step | Scientific Question | PLIP-RS Implementation |
|------|--------------------|-----------------------|
| **Current** | *Does* Python `>>>` get more attention than Rust `#[test]`? | Observational analysis (complete) |
| **Step 1: Ablation** | *What happens* if we turn off that attention? | Attention Knockout experiments |
| **Step 2: Amplification** | *Can we improve* results by boosting attention? | Attention Steering experiments |

This follows the classic experimental biology pattern:
1. **Observe** the phenomenon (attention difference: 2.8-4.4×)
2. **Remove** the suspected cause (knockout → does effect disappear?)
3. **Enhance** the suspected cause (steering → does effect amplify?)

If both ablation and amplification produce the predicted effects, we have strong causal evidence.

---

## Current PLIP-RS Architecture

### Attention Flow (from `forward_qwen2.rs:196-209`)

```
Q, K, V projections
        ↓
RoPE (Rotary Position Embeddings)
        ↓
attn_scores = Q @ K^T / sqrt(d_k)        ← Line 198
        ↓
attn_scores += causal_mask               ← Line 202 [INTERVENTION POINT A: Pre-softmax knockout]
        ↓
attn_weights = softmax(attn_scores)      ← Line 204
        ↓                                   [INTERVENTION POINT B: Post-softmax steering]
output = attn_weights @ V                ← Line 205
        ↓
O projection
```

### Key Insight: Two Intervention Points

| Point | Location | Operation | Best For |
|-------|----------|-----------|----------|
| **A** | Pre-softmax | Add -∞ to specific edges | **Knockout** (complete removal) |
| **B** | Post-softmax | Scale specific edges + renormalize | **Steering** (gradual adjustment) |

---

## Part 1: Ablation (Attention Knockout)

### 1.1 Research Question

> If we zero out attention from test markers (`>>>`, `#[test]`) to function tokens (`def`, `fn`, parameters), what happens to:
> - Next-token logit distributions?
> - Test preservation during generation?

### 1.2 Hypothesis

**H1 (Ablation Effect):** Knocking out test→function attention will:
- Significantly change next-token predictions (measured by KL divergence)
- Reduce test preservation during generation
- Have a larger effect on Python (where attention is higher) than Rust

### 1.3 Implementation Plan

#### Phase 1A: Intervention Infrastructure

Create new module `src/intervention.rs`:

```rust
/// Specifies which attention edges to modify
pub struct AttentionMask {
    /// Layer index (0-indexed)
    pub layer: usize,
    /// Source token positions (attention "from")
    pub source_positions: Vec<usize>,
    /// Target token positions (attention "to")
    pub target_positions: Vec<usize>,
    /// Intervention type
    pub intervention: InterventionType,
}

pub enum InterventionType {
    /// Set attention to zero (pre-softmax: add -∞)
    Knockout,
    /// Multiply attention by factor (post-softmax)
    Scale(f32),
    /// Set attention to specific value (post-softmax, then renormalize)
    SetValue(f32),
}

/// Result of forward pass with intervention
pub struct InterventionResult {
    /// Output logits
    pub logits: Tensor,
    /// Attention patterns (with intervention applied)
    pub attention: AttentionCache,
    /// Top-k predictions
    pub top_k: Vec<(u32, f32)>,
}
```

#### Phase 1B: Modified Forward Pass

Modify `Attention::forward_with_attn()` to accept intervention:

```rust
impl Attention {
    /// Forward pass with optional attention intervention
    pub fn forward_with_intervention(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
        intervention: Option<&AttentionMask>,
    ) -> Result<(Tensor, Tensor)> {
        // ... Q, K, V projections and RoPE ...

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Causal mask
        let mask = create_causal_mask(seq_len, x.device(), x.dtype())?;
        let mut attn_weights = attn_weights.broadcast_add(&mask)?;

        // === INTERVENTION POINT A: Pre-softmax knockout ===
        if let Some(mask) = intervention {
            if matches!(mask.intervention, InterventionType::Knockout) {
                attn_weights = apply_knockout_mask(
                    &attn_weights,
                    &mask.source_positions,
                    &mask.target_positions,
                )?;
            }
        }

        let mut attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // === INTERVENTION POINT B: Post-softmax steering ===
        if let Some(mask) = intervention {
            match &mask.intervention {
                InterventionType::Scale(factor) => {
                    attn_weights = apply_scale_intervention(
                        &attn_weights,
                        &mask.source_positions,
                        &mask.target_positions,
                        *factor,
                    )?;
                }
                InterventionType::SetValue(target) => {
                    attn_weights = apply_set_value_intervention(
                        &attn_weights,
                        &mask.source_positions,
                        &mask.target_positions,
                        *target,
                    )?;
                }
                _ => {}
            }
        }

        let attn_output = attn_weights.matmul(&v)?;
        // ... output projection ...
    }
}
```

#### Phase 1C: Knockout Implementation

```rust
/// Apply knockout mask by setting specified edges to -infinity (pre-softmax)
fn apply_knockout_mask(
    attn_scores: &Tensor,
    source_positions: &[usize],
    target_positions: &[usize],
) -> Result<Tensor> {
    // attn_scores shape: [batch, heads, seq, seq]
    // We need to set attn_scores[..., source, target] = -inf

    let (batch, heads, seq_src, seq_tgt) = attn_scores.dims4()?;

    // Create knockout mask: 0 for knockout edges, 1 elsewhere
    let mut mask_data = vec![0.0f32; batch * heads * seq_src * seq_tgt];

    for &src in source_positions {
        for &tgt in target_positions {
            if src < seq_src && tgt < seq_tgt {
                for b in 0..batch {
                    for h in 0..heads {
                        let idx = b * heads * seq_src * seq_tgt
                                + h * seq_src * seq_tgt
                                + src * seq_tgt
                                + tgt;
                        mask_data[idx] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    let mask = Tensor::from_vec(mask_data, (batch, heads, seq_src, seq_tgt), attn_scores.device())?
        .to_dtype(attn_scores.dtype())?;

    attn_scores.add(&mask)
}
```

### 1.4 Experiment Design: Single Forward Pass

| Condition | Description | Measure |
|-----------|-------------|---------|
| **Baseline** | Normal forward pass | Logits, top-k predictions |
| **Knockout** | Zero `>>>`→function attention at layer 16 | Same |
| **Control** | Zero random edges (same count) | Same |

**Corpus:** Use existing `attention_samples_universal.json` (10 Python, 10 Rust samples)

**Metrics:**
1. **KL Divergence:** `KL(P_baseline || P_knockout)` - how much do logits change?
2. **Top-1 Stability:** Does the most likely next token change?
3. **Perplexity Delta:** `PPL_knockout - PPL_baseline`

### 1.5 Experiment Design: Generation

Once single-pass knockout works, extend to autoregressive generation:

```rust
impl PlipModel {
    /// Generate tokens with intervention applied at each step
    pub fn generate_with_intervention(
        &self,
        prompt: &str,
        intervention: &AttentionMask,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<GenerationResult> {
        let mut context = self.tokenize(prompt)?;
        let mut generated = Vec::new();

        for _ in 0..max_tokens {
            // Track which positions are test markers as context grows
            let updated_intervention = update_intervention_positions(
                intervention,
                &context,
            )?;

            let result = self.forward_with_intervention(&context, Some(&updated_intervention))?;
            let next_token = sample(&result.logits, temperature);

            generated.push(next_token);
            context.push(next_token);

            if next_token == self.eos_token() {
                break;
            }
        }

        Ok(GenerationResult { text: self.decode(&generated)?, tokens: generated })
    }
}
```

**Preservation Scoring:**

```rust
/// Score how well generated code preserves test content
pub fn score_preservation(prompt: &str, generated: &str, language: Language) -> PreservationScore {
    match language {
        Language::Python => {
            // Check: Does generated doctest call the function?
            let func_name = extract_function_name(prompt);
            let params = extract_parameters(prompt);

            PreservationScore {
                calls_function: generated.contains(&func_name),
                uses_params: params.iter().any(|p| generated.contains(p)),
                syntactically_valid: is_valid_doctest(generated),
            }
        }
        Language::Rust => {
            // Check: Does generated test call the function with correct types?
            PreservationScore {
                calls_function: /* ... */,
                uses_params: /* ... */,
                syntactically_valid: /* ... */,
            }
        }
    }
}
```

### 1.6 Expected Outcomes

| Outcome | Interpretation | Next Steps |
|---------|---------------|------------|
| Knockout → significant KL divergence | Attention is causally important | Proceed to Part 2 (amplification) |
| Knockout → no effect | Attention is epiphenomenal (correlated but not causal) | Investigate other mechanisms |
| Python knockout > Rust knockout | Higher attention = more causal importance | Supports the "inline syntax" hypothesis |

### 1.7 Files to Create/Modify

| File | Status | Purpose |
|------|--------|---------|
| `src/intervention.rs` | **New** | AttentionMask, InterventionType structs |
| `src/forward_qwen2.rs` | Modify | Add `forward_with_intervention()` |
| `src/forward.rs` | Modify | Same for StarCoder2 |
| `src/forward_gemma.rs` | Modify | Same for Gemma |
| `src/model.rs` | Modify | Expose intervention through PlipModel |
| `src/generation.rs` | **New** | Autoregressive generation with intervention |
| `examples/knockout_single.rs` | **New** | Single forward pass knockout experiment |
| `examples/knockout_generation.rs` | **New** | Generation knockout experiment |

### 1.8 Estimated Effort

| Task | Hours | Dependencies |
|------|-------|--------------|
| Design intervention structs | 2-4 | None |
| Implement knockout mask application | 4-6 | intervention.rs |
| Modify forward passes (3 architectures) | 8-12 | mask application |
| Single-pass experiment script | 4-6 | modified forward |
| KL divergence analysis | 2-4 | experiment script |
| Generation loop with intervention | 6-8 | modified forward |
| Preservation scoring | 4-6 | generation loop |
| Run full experiments (4 models × 3 conditions) | 4-8 | all above |
| Statistical analysis & write-up | 6-8 | experiment results |
| **Total** | **40-62 hours** | ~1.5-2 weeks |

---

## Part 2: Amplification (Attention Steering)

### 2.1 Research Question

> If we boost Rust `#[test]` attention to Python `>>>` levels (~9%), does test preservation improve?

### 2.2 Hypothesis

**H2 (Amplification Effect):** Steering Rust test→function attention upward will:
- Improve Rust test preservation during generation
- Show dose-dependent response (more steering → more improvement, up to a ceiling)
- Have diminishing returns beyond Python's natural level (~9%)

### 2.3 Implementation Plan

#### Phase 2A: Post-Softmax Steering

```rust
/// Scale attention weights for specified edges, then renormalize
fn apply_scale_intervention(
    attn_weights: &Tensor,
    source_positions: &[usize],
    target_positions: &[usize],
    scale_factor: f32,
) -> Result<Tensor> {
    // attn_weights shape: [batch, heads, seq, seq]
    // After softmax, each row sums to 1.0

    // 1. Scale specified edges
    let mut data = attn_weights.to_vec4::<f32>()?;

    for b in 0..data.len() {
        for h in 0..data[b].len() {
            for &src in source_positions {
                // Scale target edges
                for &tgt in target_positions {
                    data[b][h][src][tgt] *= scale_factor;
                }
                // Renormalize the row to sum to 1.0
                let row_sum: f32 = data[b][h][src].iter().sum();
                for val in &mut data[b][h][src] {
                    *val /= row_sum;
                }
            }
        }
    }

    Tensor::new(data, attn_weights.device())?.to_dtype(attn_weights.dtype())
}

/// Set attention to a specific target value, then renormalize
fn apply_set_value_intervention(
    attn_weights: &Tensor,
    source_positions: &[usize],
    target_positions: &[usize],
    target_value: f32,  // e.g., 0.09 for 9%
) -> Result<Tensor> {
    let mut data = attn_weights.to_vec4::<f32>()?;

    for b in 0..data.len() {
        for h in 0..data[b].len() {
            for &src in source_positions {
                // Calculate how much attention we're adding/removing
                let current_total: f32 = target_positions.iter()
                    .map(|&tgt| data[b][h][src][tgt])
                    .sum();
                let target_total = target_value * target_positions.len() as f32;
                let delta = target_total - current_total;

                // Set target values
                for &tgt in target_positions {
                    data[b][h][src][tgt] = target_value;
                }

                // Redistribute delta across non-target positions
                let other_positions: Vec<usize> = (0..data[b][h][src].len())
                    .filter(|i| !target_positions.contains(i))
                    .collect();
                let adjustment = delta / other_positions.len() as f32;
                for idx in other_positions {
                    data[b][h][src][idx] = (data[b][h][src][idx] - adjustment).max(0.0);
                }

                // Final renormalization
                let row_sum: f32 = data[b][h][src].iter().sum();
                for val in &mut data[b][h][src] {
                    *val /= row_sum;
                }
            }
        }
    }

    Tensor::new(data, attn_weights.device())?.to_dtype(attn_weights.dtype())
}
```

### 2.4 Experiment Design: Dose-Response Curve

| Condition | Rust `#[test]`→fn Attention | Python `>>>`→fn Attention |
|-----------|---------------------------|--------------------------|
| **Baseline** | Natural (~2.5%) | Natural (~9%) |
| **Steering 0.5×** | ~1.25% | N/A (control) |
| **Steering 2×** | ~5% | N/A |
| **Steering 3×** | ~7.5% | N/A |
| **Steering 4×** (Python level) | ~10% | N/A |
| **Steering 6×** | ~15% | N/A |

**Hypothesis:** Improvement should be monotonic up to Python level, then plateau or degrade.

### 2.5 Steering Calibration

First, measure exact attention levels for each model to set accurate targets:

```rust
/// Calibrate steering targets based on observed attention levels
pub fn calibrate_steering_targets(model: &PlipModel, corpus: &Corpus) -> SteeringCalibration {
    let python_attention = measure_mean_attention(model, &corpus.python_samples());
    let rust_attention = measure_mean_attention(model, &corpus.rust_samples());

    SteeringCalibration {
        python_baseline: python_attention,  // e.g., 9.08%
        rust_baseline: rust_attention,      // e.g., 2.59%
        recommended_target: python_attention, // Boost Rust to Python level
    }
}
```

### 2.6 Files to Create/Modify

| File | Status | Purpose |
|------|--------|---------|
| `src/steering.rs` | **New** | Steering logic, calibration |
| `src/intervention.rs` | Modify | Add Scale, SetValue variants |
| `examples/steering_calibrate.rs` | **New** | Measure baseline attention per model |
| `examples/steering_dose_response.rs` | **New** | Run dose-response curve experiment |
| `examples/steering_generation.rs` | **New** | Generate with steering, measure preservation |

### 2.7 Estimated Effort

| Task | Hours | Dependencies |
|------|-------|--------------|
| Implement post-softmax scaling | 4-6 | intervention.rs |
| Implement set-value intervention | 4-6 | scaling |
| Calibration script | 4-6 | modified forward |
| Dose-response experiment | 6-8 | calibration |
| Generation with steering | 4-6 | generation.rs |
| Quality assessment (manual review) | 8-12 | generated outputs |
| Statistical analysis | 4-6 | experiment results |
| **Total** | **34-50 hours** | ~1-1.5 weeks |

---

## Part 3: Combined Analysis

### 3.1 Causal Triangle

The ablation and amplification experiments together form a causal triangle:

```
                    Attention Level
                    (Observed: Python > Rust)
                           ↑
                           |
           [Knockout]      |      [Steering]
           (Remove)        |      (Enhance)
                ↓          |          ↓
    If preservation ←------+------→ If preservation
    drops, attention                  improves, attention
    is causal                         is malleable
```

### 3.2 Outcome Matrix

| Knockout Result | Steering Result | Interpretation |
|-----------------|-----------------|----------------|
| Strong effect | Improvement | **Attention is causal and malleable** (best case) |
| Strong effect | No improvement | Attention is causal but one-way (knockout disrupts, boost doesn't help) |
| No effect | Improvement | Attention is not causal but manipulation works (confounded) |
| No effect | No improvement | **Attention is epiphenomenal** (correlation, not causation) |

### 3.3 Statistical Considerations

**Sample Size:**
- 50 prompts per condition (per existing SEGA methodology)
- 4 models × 7 conditions (baseline + 3 knockout + 3 steering) = 28 experimental cells
- Total: 1,400 generation runs

**Statistical Tests:**
- McNemar's test for paired preservation comparisons
- Welch's t-test for attention level comparisons
- Bonferroni correction for multiple comparisons

**Effect Size:**
- Cohen's d for continuous measures
- Odds ratio for binary preservation outcomes

---

## Part 4: Implementation Priorities

### Recommended Order

```
Week 1:     ABLATION - Single Forward Pass
            ├── [Day 1-2] intervention.rs structs
            ├── [Day 3-4] Knockout mask implementation
            ├── [Day 5-6] Modify forward_qwen2.rs
            └── [Day 7] Single-pass experiment script

Week 2:     ABLATION - Generation
            ├── [Day 1-3] generation.rs with intervention
            ├── [Day 4-5] Run knockout experiments
            └── [Day 6-7] Analyze results, validate ablation hypothesis

Week 3:     AMPLIFICATION - Steering
            ├── [Day 1-2] Post-softmax steering implementation
            ├── [Day 3-4] Calibration script
            ├── [Day 5-6] Dose-response experiment
            └── [Day 7] Analyze steering results

Week 4:     INTEGRATION & WRITE-UP
            ├── [Day 1-2] Combined analysis (causal triangle)
            ├── [Day 3-5] Paper section draft
            └── [Day 6-7] Review, iterate
```

### Hardware Requirements

| Phase | GPU Memory | Compute Time |
|-------|------------|--------------|
| Ablation (single-pass) | 16GB | ~2 hours |
| Ablation (generation) | 16GB | ~12 hours |
| Amplification (all conditions) | 16GB | ~10 hours |
| **Total** | 16GB (RTX 5060 Ti compatible) | ~24 GPU-hours |

---

## Part 5: Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Tensor shape errors in masking | High | Medium | Extensive unit tests, print shapes at each step |
| Attention normalization issues | Medium | High | Verify row sums = 1.0 after each operation |
| Generation quality too noisy | Medium | Medium | Increase N, use multiple random seeds |
| No causal effect found | Medium | High | Report negative result honestly; it's still valuable science |
| Position tracking during generation | High | Medium | Log positions at each step, visual debugging |

---

## Part 6: Success Criteria

### For AIware 2026 Submission (Feb 12)

**Minimum:** Include knockout single-pass results in Section 5.5 (Attention Analysis):
- "Ablation experiments (zeroing test→function attention) produced KL divergence of X, confirming the attention mechanism is [causally important / not causally important]."

**Stretch:** Include generation results:
- "Knockout reduced Python doctest preservation from 100% to Y%, while Rust test preservation [showed no change / also dropped by Z%]."

### For Follow-up Publication

Full causal triangle analysis:
- Knockout establishes necessity: "Without this attention, preservation fails."
- Steering establishes sufficiency: "With boosted attention, preservation improves."
- Together: "Attention is a causal mechanism for test preservation, and a viable intervention target."

---

## Appendix A: Code Templates

### Example: Knockout Experiment Script

```rust
// examples/knockout_single.rs

use plip_rs::{PlipModel, AttentionMask, InterventionType};
use anyhow::Result;

fn main() -> Result<()> {
    let model = PlipModel::from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")?;

    let python_sample = r#"def add(a, b):
    """Add two numbers.

    >>> add(2, 3)
    5
    """
    return a + b"#;

    // Identify positions: >>> at position X, function tokens at positions Y, Z
    let (tokens, marker_pos, fn_positions) = model.tokenize_with_positions(python_sample)?;

    // Baseline forward pass
    let baseline_result = model.forward_with_attention(python_sample)?;

    // Knockout forward pass
    let knockout_mask = AttentionMask {
        layer: 16,  // Best layer from prior analysis
        source_positions: vec![marker_pos],
        target_positions: fn_positions,
        intervention: InterventionType::Knockout,
    };
    let knockout_result = model.forward_with_intervention(python_sample, Some(&knockout_mask))?;

    // Compare logits
    let kl_div = compute_kl_divergence(&baseline_result.logits, &knockout_result.logits)?;

    println!("KL Divergence (baseline vs knockout): {:.6}", kl_div);
    println!("Baseline top-1: {:?}", baseline_result.top_k[0]);
    println!("Knockout top-1: {:?}", knockout_result.top_k[0]);

    Ok(())
}
```

### Example: Steering Experiment Script

```rust
// examples/steering_dose_response.rs

use plip_rs::{PlipModel, AttentionMask, InterventionType};
use anyhow::Result;

fn main() -> Result<()> {
    let model = PlipModel::from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")?;

    let rust_sample = r#"fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[test]
fn test_add() {
    assert_eq!(add(2, 3), 5);
}"#;

    let (tokens, marker_pos, fn_positions) = model.tokenize_with_positions(rust_sample)?;

    // Measure baseline Rust attention
    let baseline = model.forward_with_attention(rust_sample)?;
    let baseline_attn = measure_attention_level(&baseline, marker_pos, &fn_positions);
    println!("Baseline Rust attention: {:.2}%", baseline_attn * 100.0);

    // Test different steering levels
    for scale in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0] {
        let mask = AttentionMask {
            layer: 16,
            source_positions: vec![marker_pos],
            target_positions: fn_positions.clone(),
            intervention: InterventionType::Scale(scale),
        };

        let result = model.forward_with_intervention(rust_sample, Some(&mask))?;
        let steered_attn = measure_attention_level(&result, marker_pos, &fn_positions);

        println!("Scale {:.1}×: attention = {:.2}%", scale, steered_attn * 100.0);
    }

    Ok(())
}
```

---

## Appendix B: Connection to Paper Outline

These experiments would strengthen the AIware 2026 paper by:

1. **Section 5.5.3 (Interpretation):** Add causal evidence:
   > "Ablation experiments confirm that attention is causally important: zeroing test→function attention at layer 16 produces KL divergence of X (p < 0.001)."

2. **Section 5.5.4 (Connection to Model Behavior):** Add intervention results:
   > "Steering Rust test attention to Python levels (~9%) improved test preservation from Y% to Z%, demonstrating that attention is not only correlated with but also sufficient for preservation."

3. **Section 7 (Conclusion):** Strengthen RQ4 answer:
   > "RQ4: Python doctests receive higher attention AND this attention is causally responsible for preservation (ablation) AND malleable (steering)."

4. **Future Work:** If full experiments don't complete by Feb 12:
   > "Preliminary ablation results suggest attention is causal; full dose-response steering experiments are ongoing."

---

*Created: February 1, 2026*
*For: PLIP-RS Causal Validation (Ablation & Amplification)*
*Prerequisite: INTERVENTION_ROADMAP.md (observational baseline)*
