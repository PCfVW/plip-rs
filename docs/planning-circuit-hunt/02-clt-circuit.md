# 2. CLT Feature Mapping and Decoder-to-Vocabulary Projection

**Model:** Gemma 2 2B (`google/gemma-2-2b`, 26 layers, 2304 hidden, 256K vocab)

**CLT:** `mntss/clt-gemma-2-2b-426k` (16,384 features/layer, 426K total)

**Hardware:** RTX 5060 Ti 16 GB, CUDA

This document covers CLT feature layer mapping (Phase 2), decoder-to-vocabulary
projection (Phase 2b), and the cross-phase synthesis into a three-stage circuit
model.

---

## 2.1 CLT feature layer mapping

**Experiment:** `examples/couplet_clt_circuit.rs`
**Output:** `outputs/couplet_clt_circuit.json`

### Method

For each of the ten couplets, encode the last-token activations (the newline
position) through the CLT encoder and score active features by cosine similarity
between their decoder vector and the target word embedding. Select the top-20
features per couplet. For each feature, record the source layer (where the CLT
encoder reads) and the decoder write strength (L2 norm) at every downstream
layer. Compute pairwise Jaccard similarity between the feature sets of all
couplet pairs.

### Source layer distribution: bimodal

The source layers show a strongly **bimodal** distribution -- heavy at both
ends, sparse in the middle:

| Layer range | Features | Percentage |
|-------------|----------|------------|
| L0+L1 (early) | 97 | 48.5% |
| L2-L23 (middle) | 27 | 13.5% |
| L24+L25 (late) | 76 | 38.0% |

Global histogram (layers with zero features omitted):

| Layer | Count | Layer | Count | Layer | Count |
|-------|-------|-------|-------|-------|-------|
| L0 | 83 | L6 | 5 | L22 | 3 |
| L1 | 14 | L7 | 1 | L23 | 4 |
| L3 | 2 | L9 | 3 | L24 | 16 |
| L4 | 1 | L11 | 1 | L25 | 60 |
| | | L13 | 3 | | |
| | | L20 | 2 | | |
| | | L21 | 2 | | |

This "dumbbell" shape was unexpected. The original hypothesis posited a
"plan register" at layers 14-22, but instead the circuit reads from the
embedding layer and the final layers, with only 13.5% of features in the
middle twenty-two layers.

### Two archetypes

The ten couplets divide into two groups based on their source layer profiles:

**Group A: L0-heavy (embedding recall)**

| Couplet | L0 | L25 | Character |
|---------|----|-----|-----------|
| 14 (world) | 15 | 2 | Strongest L0 |
| 8 (fire) | 11 | 0 | Pure L0 (zero L25) |
| 1 (light) | 10 | 1 | L0-heavy |
| 2 (play) | 10 | 2 | L0-heavy |
| 6 (air) | 10 | 2 | L0-heavy |
| 7 (gold) | 10 | 6 | L0-heavy with L25 |

**Group B: L25-heavy (late computation)**

| Couplet | L0 | L25 | Character |
|---------|----|-----|-----------|
| 13 (truth) | 1 | 16 | Strongest L25 |
| 3 (sound) | 5 | 13 | L25-heavy |
| 15 (earth) | 4 | 10 | L25-heavy |
| 4 (rain) | 7 | 8 | Mixed, L25-leaning |

Group A couplets (light, play, air, gold, fire, world) resolve rhyme primarily
from embedding-level patterns. Group B couplets (truth, sound, earth, rain)
require deep contextual processing in the final layers.

### Shared backbone: feature L24:15133

Feature L24:15133 appears in **9 out of 10 couplets** (all except couplet 13,
truth). Its top vocabulary projections (energy, scientific, tourism, music)
suggest it encodes general discourse-continuation rather than specific rhyme
content -- a near-universal "planning infrastructure" feature.

### U-shaped decoder norm profile

All L0-sourced features exhibit a **U-shaped write profile** across layers:
peak norm at L0, trough at L10-12, recovery at L25. Quantitatively, 98.5% of
L0 features have L25 norms exceeding 50% of their L0 norm. Some L0 features
(13018, 923, 1516, 4026) have peak write norms at L23-25 despite reading
from L0.

This overturns the naive interpretation that "L0 features write locally to
early layers." Instead, these features write a persistent signal into the
residual stream that is available at every layer from L0 through L25. The
U-shaped profile means L0 features broadcast across the **entire network**.

### Pairwise feature overlap (Jaccard similarity)

Full 10x10 Jaccard matrix (upper triangle):

| | light | play | sound | rain | air | gold | fire | truth | world | earth |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| light | - | 0.177 | 0.026 | 0.053 | 0.053 | 0.081 | 0.177 | 0.000 | 0.081 | 0.026 |
| play | | - | 0.053 | 0.053 | 0.081 | 0.111 | 0.177 | 0.026 | 0.111 | 0.026 |
| sound | | | - | 0.081 | 0.053 | 0.111 | 0.053 | 0.212 | 0.081 | 0.177 |
| rain | | | | - | 0.053 | 0.081 | 0.081 | 0.081 | 0.026 | 0.143 |
| air | | | | | - | 0.026 | 0.081 | 0.000 | 0.053 | 0.026 |
| gold | | | | | | - | 0.081 | 0.111 | 0.081 | 0.081 |
| fire | | | | | | | - | 0.000 | 0.143 | 0.026 |
| truth | | | | | | | | - | 0.000 | 0.081 |
| world | | | | | | | | | - | 0.026 |
| earth | | | | | | | | | | - |

**Statistics:**
- Average Jaccard: **0.075** (mixed regime -- neither fully shared nor fully independent)
- Maximum Jaccard: **0.212** (sound/truth)
- 41/45 pairs (91%) share at least one feature
- 4 zero-overlap pairs: all involve couplet 13 (truth) paired with L0-heavy couplets
- High-overlap cluster: {light, play, fire} (mutual Jaccard = 0.177, all L0-heavy)
- High-overlap cluster: {sound, truth, earth} (Jaccard 0.177-0.212, all L25-heavy)

The low average Jaccard (0.075) with high connectivity (91% share at least one
feature) indicates a circuit architecture with a shared backbone (L0 features,
responsible for the connectivity) and content-specific features (L25 features,
responsible for the low Jaccard).

```bash
cargo run --release --example couplet_clt_circuit
```

---

## 2.2 Decoder-to-vocabulary projection

**Experiment:** `examples/couplet_clt_decoder_vocab.rs`
**Output:** `outputs/couplet_clt_decoder_vocab.json`

### Method

For each couplet's top-20 planning features, project their L25 decoder vector
through the unembedding matrix (logit lens on the decoder vector itself) to see
which vocabulary words each feature pushes toward. Check how many of the top-50
vocabulary predictions fall within the target rhyme family. This tests whether
different features activate different rhyme candidates (evidence of search) or
all point to the same word (pure retrieval).

### Per-couplet results

| ID | Target | Features with hits | Distinct candidates | Candidates |
|----|--------|--------------------|---------------------|------------|
| 1 | light | 10/20 | 3 | flight, light, night |
| 2 | play | 8/20 | 7 | away, bay, day, pay, play, say, way |
| 3 | sound | 5/20 | 1 | sound |
| 4 | rain | 6/20 | 2 | brain, rain |
| 6 | air | 9/20 | 3 | air, there, where |
| 7 | gold | 0/20 | 0 | - |
| 8 | fire | 5/20 | 2 | fire, wire |
| 13 | truth | 0/20 | 0 | - |
| 14 | world | 1/20 | 1 | world |
| 15 | earth | 2/20 | 1 | earth |

Aggregate: average 2.0 distinct rhyme candidates per couplet.

### Three-tier pattern

**Tier 1: Multi-candidate (3-7 candidates)**
- **play** (7 candidates): the strongest evidence of multi-candidate activation.
  Eight different features each push toward different -ay words: L0:4885 pushes
  toward play+pay, L0:15301 toward day+bay+way, L0:13018 toward say, L0:955
  toward away.
- **light** (3): flight, light, night from 10 different features.
- **air** (3): air, there, where from 9 different features.

**Tier 2: Target-only or target+1 (1-2 candidates)**
- rain (2: brain, rain), fire (2: fire, wire), sound (1), world (1), earth (1).

**Tier 3: Silent (0 candidates)**
- **gold** and **truth**: 0/20 features have any rhyme family member in the
  top-50 vocabulary predictions.

### Weak signals: associative spreading activation

Although "play" has 7 distinct rhyme candidates, the rhyme words appear at
**rank 11-50**, not in the top-10 predictions. The actual top-10 for most
features are unrelated words:

| Feature | Top-3 predictions | Rhyme hit (rank) |
|---------|-------------------|------------------|
| L0:15301 | body, man, time | day, bay, way (rank 30-50) |
| L0:4885 | paint, flywheel, pyrolysis | play (rank 8), pay (rank 20-50) |
| L0:13018 | ship, space, sheet | say (rank 40-50) |
| L24:15133 | energy, scientific, tourism | none |

Most features' primary vocabulary projections are noise (code tokens,
multilingual strings, generic words). The rhyme hits are weak residual signals,
not dominant predictions. This is **associative spreading activation**: the
features weakly activate multiple members of the phonological neighborhood,
analogous to spreading activation in psycholinguistic models, but the
activation is diffuse rather than focused.

### Correlation with archetypes

The multi-candidate tier (play, light, air) consists entirely of **Group A
(L0-heavy)** couplets. The silent tier (gold, truth) includes one Group A
and one Group B couplet. The L25-heavy couplets (sound, rain, earth) cluster
in the target-only tier.

This may reflect the nature of L0 features: reading from the embedding layer,
they capture broad phonological neighborhoods. L25 features, reading from
fully contextualized representations, may be more narrowly tuned.

```bash
cargo run --release --example couplet_clt_decoder_vocab
```

---

## 2.3 Cross-phase synthesis: the three-stage circuit

Combining the logit lens (Phase 1, see
[01-logit-lens.md](01-logit-lens.md)), CLT feature mapping (Phase 2), and
decoder-to-vocabulary projection (Phase 2b), the rhyme planning circuit in
Gemma 2 2B operates in three stages:

### Stage 1: Encoding (L0 features)

At the newline position, CLT features reading from the embedding layer activate
a broad phonological neighborhood. These features encode "end-of-line, expect
rhyme" and weakly push toward multiple members of the relevant rhyme family.
They broadcast across all layers via the U-shaped decoder norm profile.

### Stage 2: Refinement (middle layers)

The logit lens shows no rhyme signal before layer 14 at the earliest. The
sparse middle-layer features (13.5% of total) may contribute to narrowing the
candidate set, but this process is not directly observable with the current
tools. The refinement stage is inferred from the gap between L0 encoding and
L22+ crystallization.

### Stage 3: Crystallization (L22-25)

The rhyme word first becomes linearly readable at layers 14-22 via the logit
lens. L25-heavy features provide content-specific refinement. The final
selection happens in the last 4-5 layers.

### Mechanism: stochastic search (fuzzy retrieval)

The circuit does not perform deliberate multi-candidate search as observed in
Claude 3.5 Haiku (Anthropic, "Planning in Poems"). Instead, it implements
**stochastic search**: multiple rhyme candidates are activated simultaneously
but weakly, and the final word is selected through a noisy competition process
in the final layers.

| Property | Pure retrieval | Gemma 2 2B (observed) | Deliberate search (Claude 3.5 Haiku) |
|----------|---------------|----------------------|--------------------------------------|
| Candidates per couplet | 1 | 0-7 (avg 2.0) | Many, explicitly tracked |
| Candidate strength | Top-1 dominant | Rank 11-50 (weak) | Top-ranked |
| Cross-token maintenance | No | Not tested | Yes (backward construction) |
| Different features push different words | No | Yes (play: 7 words from 8 features) | Yes |
| Coverage | All couplets | 8/10 (gold, truth silent) | All prompts |
| Circuit sharing | N/A | 91% pairs share features, avg Jaccard 0.075 | Unknown |

Gemma 2 2B occupies an intermediate position between pure retrieval and
deliberate search. It activates multiple candidates (unlike pure retrieval)
but holds them weakly at rank 11-50 (unlike Claude 3.5 Haiku, which maintains
top-ranked candidates across positions with backward construction). The
mechanism is closer to fuzzy phonological retrieval with stochastic selection
than to the explicit multi-candidate tracking and backward construction
reported for Claude 3.5 Haiku.

**Note on feature granularity.** The analysis above uses the 426K CLT, where
features encode rhyme groups rather than individual words. The
[2.5M CLT upgrade](../planning-in-poems/04-2.5M-word-level.md) closes this
gap: at 98,304 features per layer, every rhyme word has its own dedicated
feature (209 words, mean rank 1.0, zero cross-group contamination). The
Figure 13 experiment at 2.5M resolution achieves a 52.2% cross-group redirect
and a 3.78-trillion-fold spike ratio, confirming the same planning phenomenon
at word-level resolution.

### What Gemma 2 2B lacks for deliberate search

Three properties distinguish Claude 3.5 Haiku's planning from what is observed
here:

1. **Explicit multi-candidate maintenance.** In Gemma, candidates are weak
   rank-11-50 signals, not top-ranked predictions maintained across positions.
2. **Backward construction.** No evidence that Gemma works backward from the
   desired line-end to construct intermediate words.
3. **Cross-position coordination.** Planning appears concentrated at the
   newline position; whether it propagates across line-2 tokens has not been
   tested.

### Two-tier circuit architecture

The synthesis reveals a two-tier architecture:

1. **Shared backbone** (L0 features): encodes structural "end-of-line, expect
   rhyme" signals, broadly shared across couplets via the U-shaped broadcast
   profile. Responsible for the high connectivity (91% of pairs share at
   least one feature).
2. **Content-specific features** (L25 features): encodes the specific
   semantic/phonological context, pair-specific, responsible for the low
   average Jaccard (0.075).

The split into L0-heavy vs L25-heavy archetypes suggests that some rhyme
combinations are resolved primarily from embedding-level patterns (Group A:
light, play, air, gold, fire, world) while others require deep contextual
processing (Group B: truth, sound, earth, rain).

---

## 2.4 Limitations

1. **Cosine-based feature selection.** The top-20 features per couplet are
   selected by cosine similarity with the target word embedding. Features
   that contribute to rhyming through indirect mechanisms (phonological
   features that do not align with specific word embeddings) would be missed.

2. **No causal interventions.** The evidence in this document is correlational.
   Suppressing or amplifying specific features to test whether they causally
   determine the rhyme word is addressed in subsequent phases.

3. **Small corpus.** Ten couplets provide limited statistical power. The
   three-tier pattern (multi-candidate / target-only / silent) may partly
   reflect rhyme family size and word frequency.

4. **Logit lens is a linear probe.** The 40% failure rate in Phase 1 (four
   couplets where no rhyme word appears at the probe position) may reflect
   nonlinear encoding rather than absence of planning.

---

## Reproduction

```bash
# Phase 2: CLT circuit mapping
cargo run --release --example couplet_clt_circuit

# Phase 2b: Decoder-to-vocabulary projection
cargo run --release --example couplet_clt_decoder_vocab
```

Results are written to the `outputs/` directory as JSON files.

---

## References

- **Cross-Layer Transcoders (CLTs):** Lindsey et al., "On the Biology of a Large
  Language Model", Anthropic, 2025.
  [transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- **Planning in Poems:** Lindsey et al., Section on rhyme planning in Claude 3.5
  Haiku, describing deliberate multi-candidate search with backward construction.
  [transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poems)
