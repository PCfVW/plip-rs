# 1. Per-Position Logit Lens (Gemma 2 2B)

**Author:** Eric Jacopin

**Model:** google/gemma-2-2b (2.6B parameters, base, BF16, 26 layers)

**Hardware:** RTX 5060 Ti 16 GB, CUDA

This document covers Phase 1 of the planning circuit hunt: applying a
per-position logit lens to Gemma 2 2B to determine where in the network
rhyme planning first becomes linearly readable.

---

## 1.1 Experiment design

**Experiment:** `examples/couplet_logit_lens.rs` with `--model google/gemma-2-2b`

**Output:** `outputs/couplet_logit_lens.json`

Ten couplets from the Ollama behavioral baselines (the ones Gemma 2 2B
reliably completed with correct rhymes):

| ID | Target | Line 1 | Rhyme family size |
|----|--------|--------|-------------------|
| 1  | light  | The moon casts silver light, | 19 |
| 2  | play   | The children laugh and play, | 20 |
| 3  | sound  | The thunder makes a sound, | 13 |
| 4  | rain   | The clouds bring heavy rain, | 19 |
| 6  | air    | The geese fly through the air, | 22 |
| 7  | gold   | The sunset gleams like gold, | 13 |
| 8  | fire   | The embers feed the fire, | 17 |
| 13 | truth  | She spoke the honest truth, | 8 |
| 14 | world  | He traveled all the world, | 9 |
| 15 | earth  | The seeds lay in the earth, | 9 |

Each prompt is the line-1 text followed by `\n`. The probe position is the
last token (the newline), where the model must commit to a plan for line 2.

## 1.2 Method

For each couplet, collect the full activation cache (hidden states at every
layer for every token position). At two positions -- the probe position
(last token = `\n`) and a control position (a mid-line token) -- apply the
logit lens at every layer (0-25): project the intermediate hidden state
through the final RMSNorm and unembedding matrix, then check whether any
member of the target rhyme family appears in the top-50 predictions.

The control position tests whether any observed rhyme signal is specific to
the planning site or is merely a consequence of the prompt containing the
target word.

```bash
cargo run --release --example couplet_logit_lens -- --model google/gemma-2-2b
```

---

## 1.3 Per-couplet results

| ID | Target | Probe: first rhyme layer | First rhyme word | Control pos (token) | Control: first rhyme layer |
|----|--------|--------------------------|------------------|---------------------|---------------------------|
| 1  | light  | **22** | light | 4 (silver) | 4 |
| 2  | play   | **22** | play | 4 (and) | 11 |
| 3  | sound  | **never** | - | 4 (a) | 7 |
| 4  | rain   | **16** | rain | 4 (heavy) | 7 |
| 6  | air    | **14** | there | 4 (through) | 17 |
| 7  | gold   | **never** | - | 4 (ams) | 24 |
| 8  | fire   | **22** | fire | 4 (the) | 14 |
| 13 | truth  | **19** | truth | 4 (honest) | 0 |
| 14 | world  | **never** | - | 4 (the) | 15 |
| 15 | earth  | **never** | - | 4 (in) | never |

---

## 1.4 Key findings

1. **Rhyme planning is a late-layer phenomenon.** Among the 6 couplets where
   a rhyme word surfaces at the probe position, the average first-rhyme layer
   is **19.2** (74% network depth). Three of those six only show rhymes at
   layer 22 (85% depth).

2. **40% failure rate.** Four couplets (sound, gold, world, earth) never show
   any rhyme family member in the top-50 at the probe position. These tend to
   have smaller, phonologically unusual rhyme families (-ound, -old, -orld,
   -irth). The model may plan rhymes for these through a mechanism not visible
   in the linear logit lens projection.

3. **Control positions show earlier hits, but for semantic reasons.** The
   control position shows rhyme hits in 9/10 cases, often earlier than the
   probe. This is driven by semantic association (e.g., "silver" at position 4
   elicits "white", "honest" elicits "truth") rather than rhyme planning.

4. **Layer 14 is the earliest observed rhyme emergence** (couplet 6, "air"
   family, word "there"). The spread from layer 14 to layer 22 suggests
   different rhyme families may be resolved at different depths.

---

## 1.5 Interpretation

The logit lens confirms that rhyme planning happens in the **final quarter**
of the network (layers 14-22 out of 26), consistent with the hypothesis that
planning features write to late layers. However, the 40% failure rate means
the logit lens is an incomplete probe -- the circuit may encode rhyme
constraints in a subspace that does not project linearly to vocabulary.

This motivates two follow-up directions:

- **CLT feature mapping** (Phase 2): use cross-layer transcoder features
  rather than the linear logit lens to identify planning representations that
  may be encoded non-linearly.
- **Cross-model comparison** (Phase 4): apply the same logit lens to
  Llama 3.2 1B (16 layers) to test whether the late-layer rhyme signal is
  specific to Gemma or shared across architectures.

---

## Reproduction

```bash
cargo run --release --example couplet_logit_lens -- --model google/gemma-2-2b
```

Results are written to `outputs/couplet_logit_lens.json`.
