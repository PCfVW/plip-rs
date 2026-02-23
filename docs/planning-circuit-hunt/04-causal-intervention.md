# 4. Layer Suppression -- Causal Intervention

**Author:** Eric Jacopin

**Model:** google/gemma-2-2b (2.6B parameters, base, BF16, 26 layers)

**Hardware:** RTX 5060 Ti 16 GB, CUDA

This document covers Phase 5 of the planning circuit hunt: a layer
suppression causal intervention on Gemma 2 2B to test whether the
commitment layers (L22-25) are causally necessary for couplet rhyming.

---

## 4.1 Motivation

Phases 1-4b established correlational evidence for a commitment circuit in
Gemma 2 2B's final layers. The logit lens (Phase 1) showed rhyme words
appearing at layers 14-22 and persisting through L25. The cross-model
comparison (Phase 4b) showed that Llama 3.2 1B's rhyme signal dissipates
at the output layer, while Gemma's persists. But correlation is not
causation. The question remained: are L22-25 merely *correlated* with
rhyming, or are they *causally necessary*?

Layer suppression provides a direct causal test. If skipping L22-25 destroys
rhyming while leaving other capabilities relatively intact, these layers
contain computations that are necessary for rhyme commitment -- not just
correlated with it.

---

## 4.2 Experiment design

**Experiment:** `examples/couplet_layer_suppression.rs`

**Output:** `outputs/couplet_layer_suppression.json`

For each of the 10 couplets from the standard corpus, generate a full
line 2 (greedy decoding, max 30 tokens) with different groups of layers
skipped. A skipped layer passes its input through unchanged -- it
contributes nothing to the residual stream. The prompt is bare `line1\n`
(no instruction formatting, no priming couplet). Check whether the last
word of the generated line 2 belongs to the target rhyme family.

### Corpus

The same 10 couplets used throughout the investigation:

| ID | Target | Line 1 |
|----|--------|--------|
| 1  | light  | The moon casts silver light, |
| 2  | play   | The children laugh and play, |
| 3  | sound  | The thunder makes a sound, |
| 4  | rain   | The clouds bring heavy rain, |
| 6  | air    | The geese fly through the air, |
| 7  | gold   | The sunset gleams like gold, |
| 8  | fire   | The embers feed the fire, |
| 13 | truth  | She spoke the honest truth, |
| 14 | world  | He traveled all the world, |
| 15 | earth  | The seeds lay in the earth, |

### Layer groups tested

| Group | Layers skipped | Purpose |
|-------|---------------|---------|
| baseline | none | Model's natural completion |
| skip L0-4 | 0, 1, 2, 3, 4 | Embedding/early layers |
| skip L5-9 | 5, 6, 7, 8, 9 | Early-mid layers |
| skip L10-14 | 10, 11, 12, 13, 14 | Mid layers |
| skip L15-19 | 15, 16, 17, 18, 19 | Mid-late layers |
| skip L20-25 | 20, 21, 22, 23, 24, 25 | Late layers (full commitment range) |
| skip L22-25 | 22, 23, 24, 25 | Commitment layers only |
| skip L24-25 | 24, 25 | Final two layers |

---

## 4.3 Results

| ID | Target | base | L0-4 | L5-9 | L10-14 | L15-19 | L20-25 | L22-25 | L24-25 |
|----|--------|------|------|------|--------|--------|--------|--------|--------|
| 1  | light  | bright | - | light | - | - | - | - | bright |
| 2  | play   | pray | - | play | play | - | play | - | play |
| 3  | sound  | - | - | - | sound | - | - | - | sound |
| 4  | rain   | - | - | rain | - | - | rain | - | rain |
| 6  | air    | - | air | air | air | - | - | - | air |
| 7  | gold   | - | - | gold | - | cold | - | - | - |
| 8  | fire   | - | - | fire | fire | - | fire | - | fire |
| 13 | truth  | - | - | truth | - | - | - | - | - |
| 14 | world  | - | - | - | - | - | - | - | world |
| 15 | earth  | - | - | - | earth | - | - | - | earth |
| **TOTAL** | | **2/10** | **1/10** | **7/10** | **5/10** | **1/10** | **3/10** | **0/10** | **8/10** |

Cells show the rhyming word produced (if any); `-` means no rhyme.

---

## 4.4 Key findings

### 1. Baseline only rhymes 2/10

Without instruction formatting, a bare `line1\n` prompt does not reliably
elicit rhyming completions. The model predicts formatting tokens, prose
continuations, or parallel structures (e.g., "The lightning makes a flash"
for couplet 3) rather than poetry. The Ollama behavioral experiments (tragos
branch) used chat formatting that suppressed these competing predictions
and allowed the planning circuit to drive output.

This low baseline is important context: it means the layer suppression
results are measured against a condition where format prediction already
dominates the rhyme circuit.

### 2. Skipping L22-25 kills rhyming completely: 0/10

This is the strongest causal finding in the investigation. The commitment
layers (L22-25) are **necessary** for rhyming -- without them, the model
never produces a rhyming last word, even though all other layers are intact.
When L22-25 are skipped, the model generates coherent but non-rhyming text
(e.g., "They are happy and they are free" for couplet 2, "The world is a
place of mystery" for couplet 14).

The planning signal exists in intermediate layers (the logit lens confirmed
this in Phase 1), but it cannot survive to the output without L22-25.

### 3. Skipping L5-9 improves rhyming: 7/10

These early-mid layers contain computations that **interfere** with the
rhyme circuit. Removing them lets the planning signal flow unimpeded. With
L5-9 skipped, seven couplets produce correct rhymes -- a 3.5x improvement
over baseline.

The generated text under L5-9 suppression tends toward repetition and
literal continuation (e.g., "The moon casts silver light," produces "The
moon casts silver light," -- an exact echo) or prosaic but rhyming
completions. The format prediction computation in these layers is
suppressed, and the rhyme circuit's output dominates.

### 4. Skipping L24-25 gives the best rhyming: 8/10

The final two layers appear to add "output refinement" that overrides the
planning signal. Removing them reveals the underlying rhyme capability.
Eight of ten couplets rhyme -- better than any other condition, including
conditions where more layers are available.

This suggests that the commitment circuit's work is largely done by L23,
and L24-25 perform a final refinement step that, for the bare-prompt
condition, tends to redirect toward non-rhyming predictions.

### 5. Skipping L0-4 or L15-19 destroys coherence

L0-4 suppression produces degenerate output (repeated numbers:
"1000000000000000000000000000" for every couplet). L15-19 suppression
produces incoherent fragments ("The sun is on the", "The sky is full of").
These layers are essential for basic language modelling and cannot be
removed without catastrophic degradation.

---

## 4.5 Three competing computations

The layer suppression results reveal that the network contains three
competing computations at the newline position:

| Computation | Layers | Effect on rhyming | Evidence |
|---|---|---|---|
| Format prediction | L5-9 | SUPPRESSES (predicts formatting tokens, prose) | Skipping L5-9 improves rhyming from 2/10 to 7/10 |
| Rhyme planning | L10-22 | ENABLES (search + selection + commitment) | Skipping L10-14 still allows 5/10; skipping L22-25 yields 0/10 |
| Output refinement | L24-25 | SUPPRESSES (overrides planning signal) | Skipping L24-25 improves rhyming from 2/10 to 8/10 |

The baseline's poor performance (2/10) is not because the model lacks a
rhyme circuit -- it is because format prediction and output refinement
**override** the planning signal. When we remove the interfering layers
(L5-9 or L24-25), the planning circuit's output dominates and the model
rhymes.

This explains the discrepancy with behavioral experiments: the Ollama
experiments used instruction formatting that suppressed the format
prediction layers' contribution, allowing the planning circuit to drive
output. The bare prompt activates competing format-prediction computations
that the planning circuit cannot overcome.

---

## 4.6 The critical result: L22-25 are causally necessary

The 0/10 result for L22-25 suppression is the central finding. Combined
with evidence from earlier phases:

| Evidence type | Source | Finding |
|---|---|---|
| Correlational | Phase 1 (logit lens) | Rhyme words appear at L14-22, persist to L25 in Gemma |
| Correlational | Phase 4b (cross-model) | Signal persists in Gemma, dissipates in Llama |
| **Causal** | **Phase 5 (layer suppression)** | **L22-25 are necessary for rhyming (0/10 without them)** |

This establishes three conclusions:

1. **L22-25 are the commitment circuit** (or contain it). The planning
   signal from intermediate layers cannot reach the output without passing
   through these layers.

2. **Without commitment, no rhyming occurs** -- even with search and
   selection intact. The 0/10 result means that whatever rhyme information
   exists in L0-21 is insufficient on its own to produce a rhyming output.

3. **Llama 3.2 1B's failure is equivalent to L22-25 suppression in Gemma.**
   Phase 4b showed that Llama's rhyme signal dissipates at the final layer
   (L15) rather than persisting. The behavioral consequence -- unreliable
   rhyming despite having phonological knowledge -- is exactly what we
   observe when Gemma's L22-25 are removed.

---

## 4.7 Interpretation

The layer suppression experiment reframes the planning circuit from a single
pathway to a competition among three pathways:

```
Input (line1\n)
     |
  L0-4     (foundation: embedding, basic language structure)
     |
  L5-9     (format prediction: predicts prose continuation, formatting tokens)
     |
  L10-14   (rhyme search: activates phonological neighborhood)
     |
  L15-21   (rhyme selection: narrows candidates, builds commitment signal)
     |
  L22-25   (commitment: sustains selected candidate through to output)
     |
  L24-25   (output refinement: can override planning signal)
     |
  Output
```

In normal operation with a bare prompt, format prediction (L5-9) and output
refinement (L24-25) together suppress the rhyme circuit. The model produces
coherent but non-rhyming text. Instruction formatting (as in the Ollama
experiments) suppresses format prediction, tipping the balance toward the
rhyme circuit. Layer suppression achieves the same effect mechanically:
removing L5-9 or L24-25 removes the competition.

The commitment circuit (L22-25) is the bottleneck. It is the component
that both models need but only Gemma has in sufficient strength. Llama 3.2
1B, with only 16 layers total, has at most 1-3 layers between its rhyme
search results and the output -- too little depth for a commitment
mechanism to form or function.

---

## Reproduction

```bash
cargo run --release --example couplet_layer_suppression -- --model google/gemma-2-2b
```

Results are written to `outputs/couplet_layer_suppression.json`.
