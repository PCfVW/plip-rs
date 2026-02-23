# 3. Cross-Model Comparison -- The Commitment Hypothesis

**Author:** Eric Jacopin

**Models:** google/gemma-2-2b (26 layers, 2304 hidden), meta-llama/Llama-3.2-1B (16 layers, 2048 hidden)

**Hardware:** RTX 5060 Ti 16 GB, CUDA

This document covers Phase 4 (including 4b) of the planning circuit hunt: applying
the logit lens to Llama 3.2 1B, comparing with the Gemma 2 2B results from Phase 1,
and the layer-by-layer diversity analysis that reveals the commitment hypothesis.

---

## 3.1 PlipLlama fix (Phase 3 prerequisite)

Llama 3.2 1B uses `tie_word_embeddings: true` -- there is no separate `lm_head`
weight in the safetensors. PlipLlama was updated to handle this:

- `lm_head` changed from `Linear` to `Option<Linear>`
- `project_to_vocab()` falls back to `embed_tokens.embeddings().t()` when `lm_head`
  is `None` (tied embeddings)
- `forward_with_full_cache()` added, returning hidden states at every layer for
  every token position (required by the logit lens experiment)

This follows the same pattern as PlipQwen2, which already handled tied vs. untied
embeddings via an `Option<Linear>`.

---

## 3.2 Hypothesis

Based on the behavioral experiments (tragos branch) showing that Llama 3.2 1B uses
"constrained late selection" rather than forward planning, the pre-experiment
hypothesis was:

**Rhyme words never appear at the newline position in Llama at any layer.** The
model lacks the phonological planning circuit present in Gemma 2 2B.

---

## 3.3 Experiment design

**Experiment:** `examples/couplet_logit_lens.rs` with `--model meta-llama/Llama-3.2-1B`

**Output:** `outputs/couplet_logit_lens_llama32_1b.json`

Same corpus and protocol as Phase 1 (Section 1.1 of the logit lens document): ten
couplets, probe at the newline position, logit lens at every layer, check whether
any member of the target rhyme family appears in the top-50 predictions.

```bash
cargo run --release --example couplet_logit_lens -- \
    --model meta-llama/Llama-3.2-1B \
    --output outputs/couplet_logit_lens_llama32_1b.json
```

---

## 3.4 Result: hypothesis falsified

Llama 3.2 1B finds MORE couplets than Gemma 2 2B.

| Metric | Gemma 2 2B (26 layers) | Llama 3.2 1B (16 layers) |
|--------|------------------------|--------------------------|
| Rhyme found at probe | 6/10 (60%) | **7/10 (70%)** |
| Avg first-rhyme layer | 19.2 (74% depth) | 11.4 (71% depth) |
| Failures (probe) | sound, gold, world, earth | sound, gold, world |

The relative depth of first-rhyme emergence is similar (74% vs. 71%), suggesting
the planning computation occupies a comparable fraction of the network in both
models despite the absolute difference in layer count.

### Per-couplet comparison

| ID | Target | Gemma 2 2B first layer | Llama 3.2 1B first layer |
|----|--------|------------------------|--------------------------|
| 1  | light  | L22 (light) | **L13 (bright)** |
| 2  | play   | L22 (play) | **L13 (play)** |
| 3  | sound  | never | never |
| 4  | rain   | L16 (rain) | **L11 (rain)** |
| 6  | air    | L14 (there) | **L3 (share)** |
| 7  | gold   | never | never |
| 8  | fire   | L22 (fire) | L15 (fire) |
| 13 | truth  | L19 (truth) | **L10 (truth)** |
| 14 | world  | never | never |
| 15 | earth  | never | **L15 (earth)** |

Three shared failures (sound, gold, world) correspond to small, phonologically
unusual rhyme families (-ound, -old, -orld). The fourth Gemma failure (earth) is
resolved by Llama at L15 (the last layer).

The "air" couplet is the most striking: Llama activates "share" at L3 -- just 19%
depth -- nine layers earlier in relative terms than Gemma's first hit at L14 (54%
depth).

---

## 3.5 Phase 4b: layer-by-layer diversity analysis

The critical question is not just WHETHER rhyme words appear, but HOW MANY distinct
candidates appear and whether the candidate set EVOLVES across layers. A single
candidate appearing once is retrieval; multiple candidates appearing and switching
is search.

### Gemma 2 2B -- rhyme candidates per couplet across layers

| ID | Target | Distinct candidates | Layers active | Switching? | Pattern |
|----|--------|--------------------:|---------------|:----------:|---------|
| 1  | light  | 1 (light) | L22 only | no | single flash |
| 2  | play   | 1 (play) | L22 only | no | single flash |
| 3  | sound  | 0 | -- | -- | blank |
| 4  | rain   | 1 (rain) | L16-L25 | no | early onset, **persistent** |
| 6  | air    | 2 (there, where) | L14-L25 | **yes** | two candidates, settles on "there" |
| 7  | gold   | 0 | -- | -- | blank |
| 8  | fire   | 1 (fire) | L22 only | no | single flash |
| 13 | truth  | 1 (truth) | L19 only | no | single flash |
| 14 | world  | 0 | -- | -- | blank |
| 15 | earth  | 0 | -- | -- | blank |

### Llama 3.2 1B -- rhyme candidates per couplet across layers

| ID | Target | Distinct candidates | Layers active | Switching? | Pattern |
|----|--------|--------------------:|---------------|:----------:|---------|
| 1  | light  | 2 (bright, light) | L13-L15 | **yes** | bright to light switch |
| 2  | play   | 1 (play) | L13-L15 | no | stable |
| 3  | sound  | 0 | -- | -- | blank |
| 4  | rain   | 1 (rain) | L11-L15 | no | intermittent (gap at L14) |
| 6  | air    | 3 (share, where, there) | L3-L4, L11-L14 | **yes** | bimodal: share then gap then where+there |
| 7  | gold   | 0 | -- | -- | blank |
| 8  | fire   | 1 (fire) | L15 only | no | last layer only |
| 13 | truth  | 1 (truth) | L10-L14 | no | intermittent (gaps at L12, L15) |
| 14 | world  | 0 | -- | -- | blank |
| 15 | earth  | 1 (earth) | L15 only | no | last layer only |

---

## 3.6 Aggregate diversity metrics

| Metric | Gemma 2 2B | Llama 3.2 1B |
|--------|------------|--------------|
| Total distinct candidates (all couplets) | 7 | **10** |
| Avg candidates per couplet | 0.7 | **1.0** |
| Couplets with candidate switching | 1/10 | **2/10** |
| Couplets with any rhyme hit | 6/10 | **7/10** |
| Couplets with persistent signal (3+ layers) | 2 (rain, air) | 3 (rain, truth, air) |

By every diversity metric, Llama explores the rhyme space more broadly than Gemma.

---

## 3.7 The surprise: more exploration, worse performance

The behavioral experiments (tragos branch, Ollama) showed Gemma reliably produces
rhyming couplets while Llama does not. Yet Llama shows MORE total candidates (10
vs. 7), MORE candidate switching (2 vs. 1), and MORE couplets with any rhyme hit
(7 vs. 6).

How can more exploration lead to worse performance?

---

## 3.8 The key difference: signal persistence

The answer lies in what happens at the **final layers**.

### Gemma 2 2B -- signal persists through output

- rain: persistent L16-L25 (10 consecutive layers)
- air: "there" dominates L21-L25 (5 consecutive layers at the end)
- light, play, fire: single flash at L22, but L22 is only 4 layers from the end

Once a rhyme candidate appears in Gemma, it **persists through the final layers**
where it can influence generation.

### Llama 3.2 1B -- signal dissipates at output

- air: share (L3-4) then gap then where+there (L11-14) then **drops at L15**
- truth: intermittent L10-L14 then **absent at L15**
- rain: intermittent L11-L13, gap at L14, returns at L15
- fire, earth: appear only at L15 (last layer) with no buildup
- light: bright to light switch at L13-L15 (present but brief)

The rhyme signal is **intermittent and often drops at the output layer**. The
knowledge is there but it does not survive to the point where it drives token
selection.

---

## 3.9 Detailed comparison: the "air" couplet

This couplet shows the clearest search signature in both models and best
illustrates the persistence difference.

### Gemma 2 2B (26 layers)

```
L14: there (first appearance)
L15: where (switch)
L16: --
L17: there
L20: there + where (co-occurrence)
L21-L25: there (committed, 5 consecutive layers)
```

Pattern: explore (there/where) then commit ("there" persists to output).

### Llama 3.2 1B (16 layers)

```
L3-L4:  share (early exploration)
L5-L10: -- (signal lost)
L11:    where
L12:    where
L13:    where + there (co-occurrence)
L14:    where + there
L15:    -- (signal drops at output)
```

Pattern: explore (share/where/there) then **fail to commit** (signal lost at L15).

### Comparison

Llama explores THREE candidates (share, where, there) versus Gemma's two (there,
where). Llama activates rhyme candidates earlier (L3 vs. L14) and more broadly. But
Gemma locks onto "there" at L21 and sustains it for five consecutive layers through
the output. Llama loses the signal at the layer that matters most.

---

## 3.10 Interpretation: search vs. commitment

Both models perform phonological search at the newline position. The mechanistic
difference is not search breadth but **commitment** -- the ability to sustain a
selected candidate through the residual stream to the output.

| Property | Gemma 2 2B | Llama 3.2 1B |
|----------|------------|--------------|
| Search breadth | Narrower (0.7 candidates/couplet) | Wider (1.0 candidates/couplet) |
| Candidate switching | 1/10 couplets | 2/10 couplets |
| Signal persistence | Strong (rain: 10 layers, air: 5 final layers) | Weak (intermittent, drops at output) |
| Final-layer commitment | Yes (signal present at L22-L25) | No (signal often absent at L15) |
| Behavioral outcome | Reliable rhyming | Unreliable rhyming |

The planning difference between the two models is not about having phonological
knowledge (both have it) or searching for candidates (Llama searches more). It is
about **maintaining commitment**: Gemma's circuit locks onto a rhyme candidate and
sustains it through the final layers where it influences token selection. Llama's
circuit activates candidates but lets them dissipate before they can drive output.

This aligns with the behavioral characterization from the tragos branch: Gemma
shows "forward planning" (commit early, maintain through generation) while Llama
shows "constrained late selection" (knowledge is present but the commitment circuit
is absent or too weak to sustain the signal).

---

## 3.11 Revised planning model

The planning circuit has two stages:

1. **Search** (both models): activate phonological neighborhood members at the
   newline position, exploring candidates across layers.

2. **Commitment** (Gemma only): sustain the selected candidate through the final
   4-5 layers, ensuring it influences the output distribution.

Llama 3.2 1B has stage 1 (search) but lacks stage 2 (commitment). The behavioral
difference is not about knowledge or exploration -- it is about the ability to hold
a decision through the residual stream to the output.

Whether this absence reflects insufficient depth (16 layers vs. 26, leaving only
1-3 layers between search results and output) or a circuit that simply never formed
during training is an open question. Phase 5 (layer suppression on Gemma) will test
whether the commitment layers are causally necessary, not merely correlated.

---

## Reproduction

```bash
# Gemma 2 2B (Phase 1, for comparison)
cargo run --release --example couplet_logit_lens -- --model google/gemma-2-2b

# Llama 3.2 1B (Phase 4)
cargo run --release --example couplet_logit_lens -- \
    --model meta-llama/Llama-3.2-1B \
    --output outputs/couplet_logit_lens_llama32_1b.json
```

Results are written to the `outputs/` directory as JSON files.
