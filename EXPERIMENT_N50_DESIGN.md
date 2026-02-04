# N=50 Steering Generation Experiment Design

**Date**: February 2, 2026
**Model**: Qwen/Qwen2.5-Coder-3B-Instruct
**Objective**: Validate steering-induced test preservation improvement with statistical significance

---

## 1. Background & Motivation

### 1.1 Prior Results (n=5)

The initial proof-of-concept experiment (Section 5.5.7 of AIware 2026 paper) showed:

| Model | Baseline Preservation | Steered Preservation | Change |
|-------|----------------------|---------------------|--------|
| Qwen-3B | 0/5 (0%) | 2/5 (40%) | +40% |
| Qwen-7B | 1/5 (20%) | 2/5 (40%) | +20% |
| StarCoder2-3B | 0/5 (0%) | 0/5 (0%) | 0% |
| CodeGemma-7B | 0/5 (0%) | 0/5 (0%) | 0% |

**Limitation**: n=5 is insufficient for statistical significance. A +40% improvement could occur by chance with p = 0.08 (Fisher's exact test). We need larger sample size.

### 1.2 Why Qwen-3B?

1. **Only model showing improvement**: Both Qwen models improved; others showed no effect
2. **Fastest inference**: 3B parameters vs 7B = ~2x faster
3. **Highest relative improvement**: +40% vs +20% for Qwen-7B
4. **Architecture-specific**: Knockout experiments predict steering effectiveness for Qwen

### 1.3 Statistical Power Analysis

For detecting a 40% improvement (0% → 40%) with:
- Alpha = 0.05 (two-tailed)
- Power = 0.80

Fisher's exact test requires n ≥ 20 per condition. With n=50, we can detect:
- 25% improvement with >95% power
- 20% improvement with >80% power

---

## 2. Experimental Design

### 2.1 Two-Phase Design

**Phase 1: Dose Calibration (Fine-Tuning)**
- Objective: Find optimal scale factor using KL divergence and attention matching
- Samples: 10 Rust samples from universal corpus
- Scales tested: [2.0, 2.5, 3.0, 3.5, 4.0]
- Layer: 20 (optimal for Qwen-3B)

**Phase 2: Generation Experiment (n=50)**
- Objective: Measure test preservation improvement
- Samples: 50 diverse Rust code completion prompts
- Conditions: Baseline vs Optimal Scale
- Runs per prompt: 1 (temperature=0 for determinism)

### 2.2 Dose Selection Rationale

From prior calibration (STEERING_RESULTS.md):

| Scale | Rust Attention | Target Match | KL Divergence |
|-------|---------------|--------------|---------------|
| 1.0× | 2.32% | Baseline | 469.96 |
| 2.0× | 4.17% | Below Python (5.70%) | 470.52 |
| **2.5×** | ~5.2% | Close match | ~470.5 |
| **3.0×** | 5.71% | Python match | 470.53 |
| 4.0× | 7.02% | Above Python | 470.52 |

**Recommendation**: Test 2.5× and 3.0× as primary candidates.

---

## 3. Corpus Design (50 Prompts)

### 3.1 Diversity Dimensions

Each prompt is a Rust function with inline `#[test]` in doc comment:

| Dimension | Categories |
|-----------|------------|
| **Complexity** | Simple (1-line), Medium (2-5 lines), Complex (5+ lines) |
| **Parameters** | 0, 1, 2, 3+ parameters |
| **Param Types** | i32, &str, Vec<T>, Option<T>, &[T] |
| **Return Types** | Primitive, String, Vec, Option, Result, Tuple |
| **Test Types** | assert_eq!, assert!, should_panic |

### 3.2 Sample Distribution

| Category | Count |
|----------|-------|
| Arithmetic functions | 10 |
| String manipulation | 10 |
| Collection operations | 10 |
| Option/Result handling | 10 |
| Generic functions | 5 |
| Edge cases (should_panic) | 5 |
| **Total** | **50** |

### 3.3 Prompt Format

Each prompt follows the structure used in proof-of-concept:

```rust
/// Brief description
///
/// #[test]
/// fn test_name() {
///     assert_eq!(function_name(args), expected);
/// }
fn function_name(params) -> ReturnType {
```

The model completes the function body. Test preservation is measured by detecting `#[test]`, `assert_eq!`, `assert!`, or `fn test_` in the generated output.

---

## 4. Execution Protocol

### 4.1 Commands

```bash
cd experiment/plip-rs

# Phase 1: Dose calibration
cargo run --release --example steering_calibrate -- \
    --model "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --layer 20 \
    --verbose

# Phase 2a: Run with 2.5× scale
cargo run --release --example steering_generate_n50 -- \
    --model "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --layer 20 \
    --scale 2.5 \
    --chat \
    --output results/n50_scale_2.5.json

# Phase 2b: Run with 3.0× scale
cargo run --release --example steering_generate_n50 -- \
    --model "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --layer 20 \
    --scale 3.0 \
    --chat \
    --output results/n50_scale_3.0.json
```

### 4.2 Output Format

```json
{
  "experiment": "steering_generation_n50",
  "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
  "layer": 20,
  "scale": 3.0,
  "temperature": 0.0,
  "samples": [
    {
      "id": "arith_add",
      "baseline_has_test": false,
      "steered_has_test": true,
      "baseline_tokens": 45,
      "steered_tokens": 62,
      "baseline_output": "...",
      "steered_output": "..."
    }
  ],
  "summary": {
    "baseline_preserved": 5,
    "steered_preserved": 22,
    "total": 50,
    "improvement": 34,
    "improvement_pct": 34.0,
    "fisher_exact_p": 0.0003
  }
}
```

---

## 5. Statistical Analysis

### 5.1 Primary Metric

**Test preservation rate**: % of samples where generated output contains test-related content.

Detection criteria (same as proof-of-concept):
- Contains `#[test]`
- Contains `assert_eq!`
- Contains `assert!`
- Contains `fn test_`

### 5.2 Statistical Test

**Fisher's Exact Test** (one-tailed):
- H0: P(preserved|steered) ≤ P(preserved|baseline)
- H1: P(preserved|steered) > P(preserved|baseline)
- Alpha = 0.05

### 5.3 Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Statistical significance | p < 0.05 |
| Minimum improvement | +10% absolute |
| Practical significance | ≥10 additional samples preserved |

### 5.4 Expected Results

Based on n=5 results (0/5 → 2/5 = +40%), scaled to n=50:
- Baseline preservation: ~0-5 samples (0-10%)
- Steered preservation: ~15-25 samples (30-50%)
- Expected improvement: +20-40%

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Ceiling effect (baseline already preserves) | Diverse prompts to ensure low baseline |
| Floor effect (steering doesn't help) | Multiple scale factors tested |
| Prompt-specific effects | 50 diverse prompts across 6 categories |
| Generation quality degradation | Monitor KL divergence, output coherence |
| Hardware memory issues | 3B model fits in 16GB VRAM |

---

## 7. Timeline

| Phase | Duration |
|-------|----------|
| Corpus creation | 1 hour |
| Dose calibration | 10 minutes |
| Generation experiment (2 scales × 50 samples) | ~2 hours |
| Analysis & reporting | 30 minutes |
| **Total** | **~4 hours** |

---

## 8. Files Created

| File | Purpose |
|------|---------|
| `corpus/generation_prompts_n50.json` | 50 test prompts |
| `examples/steering_generate_n50.rs` | Experiment runner |
| `results/n50_scale_X.json` | Raw results per scale |
| `EXPERIMENT_N50_RESULTS.md` | Analysis report |

---

## Appendix: Quick Reference

**Model Parameters**:
- Model ID: `Qwen/Qwen2.5-Coder-3B-Instruct`
- Layers: 32 total, target layer 20
- Python baseline attention: 5.70%
- Rust baseline attention: 2.32%
- KL divergence at 3×: ~470

**Steering Parameters**:
- Intervention type: Scale (post-softmax)
- Primary scale: 3.0×
- Secondary scale: 2.5×
- Edges: `#[test]` → `fn` token positions
