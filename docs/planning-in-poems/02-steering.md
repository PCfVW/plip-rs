# 2. Steering: Why Six Methods Failed and What Worked

This document covers the steering experiments: six methods that attempted
exact-word control (all 0% hit rate), the root-cause analysis, and the design
of the category-level approach that eventually succeeded.

---

## 2.1 What Methods 1--6 tried

All six methods target the same goal: inject CLT features at the planning site
(the newline token after line 3) to force a specific word as the line ending.
120 steering pairs (20 rhyme groups x 3 prompts x 2 conditions), tested at 7
strength levels (0--16).

| Method | Feature selection strategy | Result |
|--------|---------------------------|--------|
| 1 | Max-activation probes | 0% target hit |
| 2a | Decoder dot product | 0% target hit; catastrophic degradation (70% -> 10% rhyme rate) |
| 2b | Decoder cosine similarity | 0% target hit; milder degradation (66% -> 23% rhyme rate) |
| 3 | Planning-site activation + cosine filter | 0% target hit |
| 4 | Method 3 + multi-layer clamping | 0% target hit |
| 5 | Contrastive word probes | 0% target hit |
| 6 | Causal activation patching | Best causal effect: 9.72 logit units ablated; still 0% target hit |

For comparison, attention head steering (Layer 21, Heads 1/6/7) was also
tested. It preserved coherence better (~65% rhyme rate at all strengths) but
produced the same 0% target hit rate.

**Output files:**
- `outputs/method3_results.json`, `outputs/method4_results.json`, `outputs/method5_results.json` (all 0% hit rate)
- `outputs/method6_test.json` (causal effect: 9.72 logit units)
- `outputs/clt_steering_results.json` (CLT injection)
- `outputs/attention_steering_results.json` (attention steering)
- `outputs/steering_comparison.json` (cross-mechanism comparison)

```bash
cargo run --release --example poetry_clt_steering -- --model google/gemma-2-2b
cargo run --release --example poetry_attention_steering -- --model google/gemma-2-2b
cargo run --release --example evaluate_steering
```

## 2.2 Root cause

The logit gap between the model's natural top prediction and any target word
is typically ~25+ units. With 426K features split across 26 layers (~16K per
layer) covering a 256K vocabulary, no small group of features is specific
enough to act as a lexical selector. Each feature covers a broad semantic
subspace.

Method 6 proved that causally relevant features *do exist*: ablating them drops
the target logit by up to 9.72 units. But injecting them -- even these
causally verified features -- at strengths up to 4.0 does not bridge the
~25-unit gap. At high strength, generation degrades rather than shifting to the
target.

Two insights follow:

1. **The 426K CLT captures semantic neighborhoods, not individual words.** A
   feature that pushes toward "tree" also pushes toward "forest", "garden",
   "river". This is a liability for exact-word steering but an asset for
   group-level steering.

2. **Binary argmax is the wrong metric.** All six methods measured top-1 hit.
   Anthropic measures continuous probability. It is entirely possible that
   injection increased P(target) from 0.001 to 0.01 -- a 10x shift invisible
   to argmax but clearly visible on a probability plot.

## 2.3 What Anthropic actually does

From "Planning in Poems" (Lindsey et al., 2025), on Claude 3.5 Haiku with a
30M-feature CLT:

1. Plan features identified via **attribution graphs** (backward from output)
2. Steering uses **suppress + inject**: "negatively intervene on 'rabbit' and
   'habit' features, and positively on a 'green' feature"
3. Steering **only works at the newline**
4. **70% success rate** across 25 poems
5. The model "reasons backwards" from the planned target

| Aspect | Anthropic | Methods 1--6 |
|--------|-----------|--------------|
| CLT size | **30M** features | **426K** features (70x fewer) |
| Steering | **Suppress + inject** | Inject only |
| Metric | **Continuous probability** | Binary top-1 hit |
| Feature ID | Attribution graphs | Various proxies |

Methods 1--6 never suppressed the existing plan. They only added signal on top
of what the model already intended to do. And they measured success with a
metric that requires the target to win a ~25-unit logit race.

## 2.4 The bottom-up pivot

After six top-down methods failed, the approach reversed: instead of searching
for features that steer toward a known word, scan what the features *actually
encode*.

Scanning CLT decoder vectors against the full 256K vocabulary via cosine
similarity revealed that features naturally cluster around rhyme groups (see
[01-detection.md](01-detection.md), Section 1.4). This led to the
completion-prompt design and the 6.45x detection result.

## 2.5 Semantic category steering (Method 7)

Method 7 steers toward a *semantic category* (nature, emotion, light, motion)
rather than a specific word. This matches the CLT's natural granularity.

**Design:**
- Four categories, each with 30--50 words spanning multiple rhyme groups
- Category-neutral prompts where line 3 does not bias toward any one category
- Causal category verification (Method 6's ablation technique, but summing
  logit deltas across all words in a category)
- Suppress competing features + inject target-category features
- Measure category hit rate, not exact-word hit rate

This approach incorporates both lessons from the failure analysis: it uses
suppress + inject (not inject-only) and measures at the right granularity
(category, not word).

```bash
cargo run --release --example poetry_category_steering -- --model google/gemma-2-2b
```
**Output:** `outputs/category_probe_full.json`

## 2.6 Summary of lessons

| Lesson | Source |
|--------|--------|
| Binary argmax masks real probability shifts | Methods 1--6 all 0% hit, but Method 6 showed 9.72 logit-unit causal effects |
| Inject-only cannot overcome the existing plan | Anthropic uses suppress + inject; Methods 1--6 only injected |
| 426K features are semantic neighborhoods, not words | Feature granularity limits exact-word control but enables group-level steering |
| Bottom-up discovery beats top-down search | Six top-down methods failed; bottom-up scanning revealed rhyme-group structure |
| Observational designs fail when conditions diverge after measurement | A-vs-B attention comparison: Cohen's d = 0.000 (Section 1.3 of [01-detection.md](01-detection.md)) |
