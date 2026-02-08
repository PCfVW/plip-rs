# PLIP-RS Intervention Experiment Roadmap

**Date:** February 1, 2026
**Status:** Planning
**Prerequisite:** Attention observation complete (p < 0.0002 across 4 models)

---

## Executive Summary

This roadmap outlines the path from **correlational findings** (attention patterns correlate with test preservation) to **causal evidence** (attention patterns cause test preservation). The goal is to establish whether the 2.8-4.4× attention difference between Python doctests and Rust tests is mechanistically responsible for preservation differences during code generation.

---

## Current State (Phase 1 Complete)

### What We Have
- Attention measurement infrastructure (PLIP-RS)
- Model-agnostic corpus with perfect positioning
- Validated results across 4 models (p < 0.0002)
- Effect sizes: 2.8-4.4× (Python > Rust attention to function tokens)

### What We Claim
- **Correlation**: Python `>>>` markers have stronger attention to function parameters
- **Observation**: Python doctests are preserved at higher rates during generation

### What We Cannot Claim (Yet)
- **Causation**: That attention strength *causes* better preservation

---

## Phase 2: Attention Knockout (Single Forward Pass)

**Goal:** Does zeroing attention affect next-token logits?

### 2.1 Implementation

Add attention masking capability to PlipModel:

```rust
// src/intervention.rs

use tch::Tensor;

/// Defines which attention edges to modify
pub struct AttentionMask {
    /// Layer to apply the mask (0-indexed)
    pub layer: usize,
    /// Source token positions (attention "from")
    pub source_positions: Vec<usize>,
    /// Target token positions (attention "to")
    pub target_positions: Vec<usize>,
    /// Mask type
    pub mask_type: MaskType,
}

pub enum MaskType {
    /// Zero out attention weights
    Knockout,
    /// Multiply by factor (< 1.0 = reduce, > 1.0 = amplify)
    Scale(f32),
}

impl PlipModel {
    /// Forward pass with attention intervention
    pub fn forward_with_intervention(
        &self,
        input: &str,
        masks: &[AttentionMask],
    ) -> Result<InterventionResult> {
        // Hook attention computation
        // Apply masks before softmax
        // Return logits + attention patterns
    }
}

pub struct InterventionResult {
    pub logits: Tensor,
    pub attention_patterns: Vec<Tensor>,
    pub top_predictions: Vec<(String, f64)>,
}
```

### 2.2 Experiment Design

| Condition | Description | Expected Outcome |
|-----------|-------------|------------------|
| Baseline | Normal forward pass | Reference logits |
| Knockout | Zero `>>>` → function_params attention | Degraded predictions? |
| Control | Zero random attention edges | Minimal change |

### 2.3 Metrics

1. **KL Divergence**: `KL(P_baseline || P_knockout)` - how much do logits change?
2. **Top-k Change**: Does the top prediction change?
3. **Perplexity Delta**: Does perplexity increase after knockout?

### 2.4 Success Criteria

- If KL divergence is **high** after knockout: attention is causally important
- If KL divergence is **low**: attention is not the causal mechanism

### 2.5 Estimated Effort

| Task | Hours | Notes |
|------|-------|-------|
| Implement attention hooks | 8-12 | Requires modifying forward pass |
| Add mask application | 4-6 | Before softmax in attention |
| Create experiment script | 4-6 | |
| Run experiments (4 models) | 2-4 | GPU time |
| Analysis | 4-6 | |
| **Total** | **22-34** | ~1 week |

---

## Phase 3: Attention Knockout (Generation)

**Goal:** Does zeroing attention affect test preservation during autoregressive generation?

### 3.1 Implementation

Extend intervention to generation loop:

```rust
impl PlipModel {
    /// Generate with attention intervention applied at each step
    pub fn generate_with_intervention(
        &self,
        prompt: &str,
        masks: &[AttentionMask],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<GenerationResult> {
        let mut generated = Vec::new();
        let mut context = self.tokenize(prompt)?;

        for _ in 0..max_tokens {
            // Forward with intervention
            let result = self.forward_with_intervention(&context, masks)?;

            // Sample next token
            let next_token = sample_with_temperature(&result.logits, temperature);

            // Update masks for new context length
            // (source positions shift as context grows)

            generated.push(next_token);
            context.push(next_token);

            if next_token == self.eos_token() {
                break;
            }
        }

        Ok(GenerationResult {
            text: self.decode(&generated)?,
            tokens: generated,
        })
    }
}
```

### 3.2 Experiment Design

**Prompt Template:**
```
Complete this Python function with a doctest:

def add(a, b):
    """Add two numbers.

    >>>
```

**Conditions:**
1. **Baseline**: Normal generation
2. **Knockout**: Zero attention from `>>>` to `def`/`add`/params at layer 16
3. **Amplify**: Double attention (control for direction)

**Measure:**
- Does the doctest call the function correctly?
- Does it use the correct parameter names?
- Is the expected output reasonable?

### 3.3 Preservation Scoring

```python
def score_preservation(prompt: str, generated: str) -> dict:
    """Score how well the generated doctest preserves function info."""

    # Extract function name from prompt
    func_name = extract_function_name(prompt)
    params = extract_parameters(prompt)

    # Check generated doctest
    doctest_calls_function = func_name in generated
    doctest_uses_params = any(p in generated for p in params)

    return {
        "calls_function": doctest_calls_function,
        "uses_params": doctest_uses_params,
        "syntactically_valid": is_valid_doctest(generated),
    }
```

### 3.4 Statistical Design

- **N = 50** prompts per condition
- **3 conditions** (baseline, knockout, amplify)
- **4 models**
- Total: 600 generation runs

### 3.5 Success Criteria

| Outcome | Interpretation |
|---------|----------------|
| Knockout significantly reduces preservation | Attention is causal |
| Knockout has no effect | Attention is epiphenomenal |
| Amplify improves preservation | Attention is beneficial and malleable |

### 3.6 Estimated Effort

| Task | Hours | Notes |
|------|-------|-------|
| Extend to generation loop | 8-12 | Complex: mask position tracking |
| Create prompt corpus | 4-6 | 50 diverse prompts |
| Implement scoring | 4-6 | |
| Run experiments | 8-12 | ~15 min per prompt × 600 |
| Statistical analysis | 6-8 | |
| **Total** | **30-44** | ~1.5-2 weeks |

---

## Phase 4: Attention Steering (Rust Improvement)

**Goal:** Can we improve Rust test preservation by boosting attention?

### 4.1 Hypothesis

If Python's higher attention causes better preservation, then artificially boosting Rust `#[test]` attention to function tokens should improve Rust test preservation.

### 4.2 Implementation

```rust
pub struct AttentionSteering {
    /// Target attention level (e.g., Python's ~9%)
    pub target_attention: f32,
    /// Layer to apply steering
    pub layer: usize,
    /// Source positions (test markers)
    pub sources: Vec<usize>,
    /// Target positions (function tokens)
    pub targets: Vec<usize>,
}

impl PlipModel {
    pub fn generate_with_steering(
        &self,
        prompt: &str,
        steering: &AttentionSteering,
        max_tokens: usize,
    ) -> Result<String> {
        // During attention computation:
        // 1. Compute natural attention
        // 2. For specified edges, scale to target level
        // 3. Re-normalize attention row
    }
}
```

### 4.3 Experiment Design

**Prompt:**
```rust
fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[test]
fn test_add() {
```

**Conditions:**
1. **Baseline**: Natural Rust attention (~2.5%)
2. **Steered**: Boost to Python level (~9%)
3. **Over-steered**: Boost to 15% (ceiling effect?)

**Measure:**
- Does the test correctly call `add()`?
- Does it use correct argument types?
- Does `assert_eq!` have reasonable expected value?

### 4.4 Success Criteria

If steered Rust generation shows improved test quality, this confirms:
1. Attention is causally important
2. Attention is a viable intervention target
3. Rust test quality could potentially be improved by attention manipulation

### 4.5 Estimated Effort

| Task | Hours | Notes |
|------|-------|-------|
| Implement steering | 6-8 | Similar to knockout |
| Calibrate target levels | 4-6 | Find Python attention levels per model |
| Run experiments | 6-8 | |
| Quality assessment | 8-12 | Manual review of generated tests |
| **Total** | **24-34** | ~1 week |

---

## Phase 5: Activation Patching (Advanced)

**Goal:** Can we transplant Python's attention pattern into Rust context?

### 5.1 Concept

Instead of just scaling attention, take the actual attention distribution from a Python forward pass and "paste" it into a Rust forward pass.

### 5.2 Implementation

```rust
pub struct ActivationPatch {
    /// Layer to patch
    pub layer: usize,
    /// Captured attention pattern from donor context
    pub donor_attention: Tensor,
    /// Position mapping: donor_pos -> recipient_pos
    pub position_map: Vec<(usize, usize)>,
}

impl PlipModel {
    pub fn forward_with_activation_patch(
        &self,
        input: &str,
        patch: &ActivationPatch,
    ) -> Result<Tensor> {
        // 1. Run forward pass
        // 2. At specified layer, replace attention with donor pattern
        // 3. Continue forward pass
    }
}
```

### 5.3 Challenges

- **Position alignment**: Python and Rust have different token counts
- **Semantic alignment**: `>>>` vs `#[test]` are at different relative positions
- **Attention shape**: Different sequence lengths mean different attention matrix sizes

### 5.4 Estimated Effort

| Task | Hours | Notes |
|------|-------|-------|
| Design position mapping | 8-12 | Non-trivial |
| Implement patching | 12-16 | |
| Handle edge cases | 8-12 | |
| Experiments | 8-12 | |
| **Total** | **36-52** | ~2 weeks |

---

## Timeline Summary

```
Week 1:     Phase 2 - Single Forward Pass Knockout
            ├── Implement attention hooks
            ├── Run knockout experiments
            └── Analyze logit changes

Week 2-3:   Phase 3 - Generation Knockout
            ├── Extend to generation loop
            ├── Create prompt corpus
            ├── Run preservation experiments
            └── Statistical analysis

Week 4:     Phase 4 - Attention Steering
            ├── Implement steering
            ├── Calibrate target levels
            └── Test Rust improvement

Week 5-6:   Phase 5 - Activation Patching (Optional)
            ├── Design position mapping
            ├── Implement patching
            └── Cross-context experiments

Week 7:     Analysis & Write-up
            ├── Synthesize results
            ├── Update RIGOR_EXPERIMENT.md
            └── Draft paper section
```

---

## Hardware Requirements

| Phase | GPU Memory | Compute Time |
|-------|------------|--------------|
| Phase 2 | 16GB (fits 7B) | ~2 hours |
| Phase 3 | 16GB | ~10 hours |
| Phase 4 | 16GB | ~6 hours |
| Phase 5 | 16GB | ~8 hours |

**Total:** ~26 GPU-hours on RTX 5060 Ti (16GB)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Attention hooks too slow | Medium | Medium | Batch processing, optimize |
| No causal effect found | Medium | High | Report negative result honestly |
| Position tracking bugs | High | Medium | Extensive testing |
| Generation quality too noisy | Medium | Medium | Increase N, better prompts |

---

## Dependencies

### New Crates

```toml
[dependencies]
# For statistical tests on generation results
statrs = "0.16"

# For efficient tensor manipulation
ndarray = "0.15"
```

### New PLIP-RS Modules

```
src/
├── intervention.rs      # AttentionMask, MaskType
├── steering.rs          # AttentionSteering
├── patching.rs          # ActivationPatch
└── generation.rs        # generate_with_intervention
```

---

## Success Metrics

### Phase 2 Success
- [ ] Can measure KL divergence between baseline and knockout
- [ ] Knockout shows statistically significant logit change (p < 0.05)

### Phase 3 Success
- [ ] Knockout reduces preservation rate by >20%
- [ ] Effect replicates across 2+ models

### Phase 4 Success
- [ ] Steering improves Rust test quality
- [ ] Effect is dose-dependent (more steering = more improvement, up to ceiling)

### Overall Success
- [ ] Establish causal link between attention and preservation
- [ ] Publish results in follow-up paper or workshop

---

## Relation to AIWare 2026

**For AIWare submission (Feb 12):**
- Include Phase 1 results (observational) ✅
- Mention intervention roadmap in "Future Work"
- Do NOT wait for intervention results

**For follow-up work:**
- Intervention results could be a workshop paper
- Or extension to journal version

---

*Created: February 1, 2026*
*For: PLIP-RS Causal Validation*
