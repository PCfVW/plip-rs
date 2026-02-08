# N=50 Steering Generation Experiment Results

**Date**: February 2, 2026
**Model**: Qwen/Qwen2.5-Coder-3B-Instruct
**Layer**: 20
**Scale**: 3.0x
**Duration**: 479.5 seconds (~8 minutes)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Baseline preservation** | 7/50 (14.0%) |
| **Steered preservation** | 10/50 (20.0%) |
| **Improvement** | +3 (+6.0%) |
| **Fisher's exact p-value** | 0.298 (NOT significant) |

**Conclusion**: Steering shows a modest positive trend but the effect is **not statistically significant** at p < 0.05. The n=5 proof-of-concept results (0/5 → 2/5 = +40%) appear to have been **optimistically biased by small sample variance**.

---

## Detailed Results

### Sample-Level Changes

| Change Type | Count | Samples |
|-------------|-------|---------|
| **GAINED** | 5 | string_04_lowercase, string_07_starts_with, string_09_repeat, string_10_trim, gen_01_identity |
| **LOST** | 2 | arith_07_abs, string_08_ends_with |
| **KEPT** | 5 | arith_01_add, arith_03_multiply, arith_04_divide, string_01_len, string_02_reverse |
| **NONE** | 38 | (no change, no test in either condition) |

**Net gain**: 5 gained - 2 lost = **+3**

---

### Category Analysis

| Category | Baseline | Steered | Change | Notes |
|----------|----------|---------|--------|-------|
| **string** | 3/10 (30%) | 6/10 (60%) | **+3** | Best responder |
| **generic** | 0/5 (0%) | 1/5 (20%) | +1 | Modest gain |
| **arithmetic** | 4/10 (40%) | 3/10 (30%) | -1 | Slight regression |
| **collection** | 0/10 (0%) | 0/10 (0%) | 0 | Non-responder |
| **option** | 0/10 (0%) | 0/10 (0%) | 0 | Non-responder |
| **edge_case** | 0/5 (0%) | 0/5 (0%) | 0 | Non-responder |

**Key insight**: **String functions are the primary beneficiaries** of steering (+3 samples). Collection, Option, and edge_case categories show **zero response** to steering intervention.

---

## Comparison with n=5 Results

| Metric | n=5 (proof-of-concept) | n=50 (this experiment) |
|--------|------------------------|------------------------|
| Baseline rate | 0% (0/5) | 14% (7/50) |
| Steered rate | 40% (2/5) | 20% (10/50) |
| Improvement | +40% | +6% |
| Statistical significance | N/A (underpowered) | p = 0.298 (not significant) |

**Interpretation**: The n=5 results were overly optimistic. The true effect size appears to be ~6% improvement, not 40%.

---

## Implications for AIware 2026 Paper

### Revisions Needed

1. **Section 5.5.7 (End-to-End Steering Generation)** must be updated:
   - Original claim: "steering improves test preservation from 0/5 to 2/5 (+40%)"
   - Revised finding: "n=50 follow-up shows +6% improvement (7/50 → 10/50), p = 0.30 (not significant)"

2. **Conclusions about Qwen-3B effectiveness** should be tempered:
   - Steering shows a **positive trend** but **not statistical significance**
   - Effect is **category-dependent** (string functions respond, others don't)

3. **New insight to add**: Category-specific response patterns
   - String manipulation functions: +30% (3/10 → 6/10)
   - Collection/Option operations: 0% (complete non-responders)

### Honest Reporting

The paper should acknowledge:
- Initial n=5 results were suggestive but underpowered
- n=50 follow-up shows modest, non-significant improvement
- Effect is heterogeneous across function categories
- Steering intervention may need category-specific tuning

---

## Technical Notes

### Generation Parameters

```
Model: Qwen/Qwen2.5-Coder-3B-Instruct
Layer: 20
Scale: 3.0x
Temperature: 0.0 (greedy)
Max tokens: 150
Chat template: enabled
```

### Test Detection Criteria

Generated output classified as "has test" if contains any of:
- `#[test]`
- `assert_eq!`
- `assert!`
- `fn test_`

---

## Raw Data

Results saved to: `results/n50_scale_3.0.json`

---

## Future Work

1. **Test higher scale factors** (4x, 6x) to see if stronger steering helps
2. **Category-specific steering**: Different layers/scales for different function types
3. **Longer generation**: Increase max_tokens to give model more opportunity to generate tests
4. **Multi-run experiment**: Run each sample 5x to measure within-sample variance
5. **Alternative models**: Test Qwen-7B with n=50 to compare with 3B

---

*Report generated: February 2, 2026*
