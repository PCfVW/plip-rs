# 1. Detecting the Planning Signal

**Model:** google/gemma-2-2b (2.6B parameters, base, BF16)

**CLT:** mntss/clt-gemma-2-2b-426k (16,384 features x 26 layers)

**Hardware:** RTX 5060 Ti 16 GB, CUDA 13.1

This document covers the detection phase: establishing that Gemma 2 2B
activates rhyme-planning features at the planning site, and how the
experimental design evolved to find them.

---

## 1.1 Baseline capability: priming is required

Gemma 2 2B is a base model with no built-in tendency to rhyme.

| Condition | Rhyme rate |
|-----------|:---:|
| Instruction-tuned model (via Ollama) | ~66% |
| Base model, bare single-line prompts | **0%** (continues as prose) |
| Base model, primed with one completed couplet | **100%** (5/5 prompts) |

All subsequent experiments use priming couplets. The priming couplet uses a
different rhyme ending than the target to avoid confounding.

## 1.2 CLT infrastructure

A Cross-Layer Transcoder (CLT) is a sparse dictionary that decomposes a model's
residual stream into interpretable features, with encoder and decoder vectors
spanning multiple layers (Lindsey et al., 2025; see
[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
for the underlying sparse-autoencoder approach). The 426K CLT used here was
trained by mntss on Gemma 2 2B and published as open weights.

Built `src/clt.rs` (1,640 lines) from scratch: lazy HuggingFace download,
stream-and-free encoder loading (~75 MB shards on GPU), sparse activation
encoding with ReLU threshold, decoder vector extraction with micro-cache,
and `CltInjectionSpec` for steered generation. Validated against Python
[Circuit Tracer](https://github.com/safety-research/circuit-tracer)
reference values: 90/90 top-10 features match (max relative error
1.2 x 10^-6). Acceptance test: KL divergence = 0.070 on GPU with CLT
injection.

**Validation script:**
```bash
python scripts/clt_reference.py          # generate reference (once)
cargo run --release --example validate_clt
cargo run --release --example clt_logit_shift
```

## 1.3 Attention layer scan: observational design fails

**Question:** Does Gemma 2 2B attend differently to line-ending words when
generating rhyming vs. non-rhyming text?

**Design:** 260 rhyming samples (Category A) and 260 non-rhyming controls
(Category B). Measured attention weight from the newline token to the
line-ending word at all 26 layers x 8 heads.

**Result:** Cohen's d = 0.000, p ~ 1.0 at every layer. The two conditions
produce mathematically identical representations at the newline because, in a
causal model, the difference (line 4) comes *after* the measurement point.

**Positive finding:** Line-ending words attract structurally elevated attention
(layer 21: 18.6x ratio), but this is a positional effect, not
rhyme-conditional.

**Lesson:** Observational A-vs-B designs cannot detect planning in a causal
model when conditions diverge only after the measurement point. This motivated
the pivot to interventional approaches.

```bash
cargo run --release --example poetry_layer_scan -- \
    --model google/gemma-2-2b \
    --output outputs/poetry_layer_scan_google_gemma_2_2b.json
```
**Output:** `outputs/poetry_layer_scan_google_gemma_2_2b.json`

## 1.4 Bottom-up feature discovery: reverse-engineering the CLT

After six steering methods produced 0% hit rates (see
[02-steering.md](02-steering.md)), the question changed. Anthropic's 30M-feature
CLT provides word-level resolution: they can find a feature for "green" and
another for "rabbit". With 426K features -- 70x fewer -- spread across 26 layers
and a 256K vocabulary, the same granularity cannot exist. Attempting to replicate
Anthropic's exact protocol with these tools is not a viable path.

The productive question is not "can we steer toward word X?" but "if planning
exists in this model, what granularity would the 426K CLT represent it at?"
Answering this requires understanding what the features actually encode, which
means reverse-engineering the CLT's decoder vectors against the model's full
vocabulary.

### The pipeline

Each CLT feature has a decoder vector (one per target layer) that, when added to
the residual stream, pushes the model's output in a particular direction. The
direction a feature pushes is defined by its cosine similarity with the
embedding vectors of all 256,768 tokens in Gemma 2's vocabulary.

**Step 1: Vocabulary scan.** For all 425,984 features (16,384 per layer x 26
layers), extract the decoder vector at the final layer (L25), normalize it, and
compute cosine similarity against every normalized token embedding via GPU
matmul. This produces a `[n_features, 256768]` similarity matrix, processed in
chunks of 4,096 features to fit in 16 GB VRAM. For each feature, extract the
top-20 highest-cosine tokens.

**Step 2: Linguistic filter.** From the top-1 token per feature, keep only clean
English words (ASCII alphabetic, length >= 2). Of 425,984 features scanned,
287 produce a clean English top-1 token.

**Step 3: Phoneme cross-reference.** Look up each word in the
[CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict), which
encodes pronunciations in
[ARPAbet](https://en.wikipedia.org/wiki/ARPABET) notation. In ARPAbet, each
phoneme is a short uppercase code: vowels carry a stress digit (1 = primary,
2 = secondary, 0 = unstressed) and consonants are plain letters. For example,
"around" is transcribed `AH0 R AW1 N D` -- the syllable "round" is the
diphthong `AW` with primary stress (`1`), followed by consonants `N` and `D`.

The rhyme ending is everything from the last stressed vowel onward. Examples:

| Word | ARPAbet | Rhyme ending | Intuition |
|------|---------|-------------|-----------|
| around | AH0 R **AW1** N D | AW1-N-D | rhymes with "found", "ground" |
| so | S **OW1** | OW1 | rhymes with "go", "know" |
| green | G R **IY1** N | IY1-N | rhymes with "seen", "mean" |

Features are grouped by rhyme ending, so all words that share the same
ending (and therefore rhyme) belong to the same group.

**Step 4: Selection.** Keep groups with 2+ members and cosine >= 0.3. For each
word, retain only the best feature (highest cosine). Result: 35 rhyme groups
containing 98 words.

### The key finding

CLT features cluster by **phonetic ending**, not by semantics or orthography.
A feature whose top token is "found" also scores high on "ground", "round", and
"around" -- words that share the -AW1-N-D ending but have no semantic
relationship. This is exactly the structure needed for rhyme planning.

10 clean groups were selected for the detection and steering experiments:

| Group | Words with CLT features |
|-------|-------------------------|
| -ound | found, around, ground, round |
| -ow | go, grow, know, slow, snow, so, though |
| -oo | do, new, ou, to, too, two, who |
| -ee | be, de, free, he, me, she, three, we |
| -ell | tell, well |
| -ate | great, straight, weight |
| -ill | still, until, will |
| -ire | entire, fire, require |
| -out | about, out |
| -ind | find, kind |

### Methodological lesson

The reverse-engineering answers the question posed at the start: at 426K
features, the CLT encodes **rhyme-group-level** planning, not word-level. This
constrains every subsequent experiment. Detection must measure in-group vs.
out-group activation (not word-specific). Steering must target phonetic groups
(not individual words). The Figure 13 replication must use suppress + inject
at the group level.

This is not a limitation to work around -- it is the correct experimental
design for these tools. The 6.45x detection result (Section 1.7) and the 48%
cross-group redirect (Section 3.5 of
[03-figure13-replication.md](03-figure13-replication.md)) were only possible
because the experiments were designed at the granularity the CLT actually
supports.

```bash
# Step 1: Scan decoder vectors against full vocabulary
cargo run --release --example poetry_category_steering -- \
    --mode explore-vocabulary \
    --model google/gemma-2-2b \
    --output outputs/explore_vocab_all_layers.json

# Steps 2-4: Cross-reference with CMU dict, form rhyme groups
cargo run --release --example poetry_category_steering -- \
    --mode find-rhyme-pairs \
    --explore-json outputs/explore_vocab_all_layers.json \
    --cmu-dict corpus/cmudict.dict \
    --min-cosine 0.3 \
    --output outputs/rhyme_pairs_all_layers.json
```
**Output:** `outputs/rhyme_pairs_all_layers.json`

## 1.5 Completion prompt design

Built 27 completion-style prompts following the priming protocol. Each has:
1. A priming couplet (different rhyme ending)
2. A third line ending with the target word
3. An incomplete fourth line (the model fills the gap word)

Example:
```
The stars were twinkling in the night,
The lanterns cast a golden light.
At last the missing piece was found,
Half buried in the frozen
```

The gap word should rhyme with "found" (-ound group). At T=0 (greedy
decoding), the model produces correct rhymes on **21/27 prompts (78%)**.

```bash
cargo run --release --example poetry_category_steering -- \
    --mode verify-rhyming \
    --model google/gemma-2-2b
```

## 1.6 Planning detection V1: single-line prompts (failed)

Used instruction-style single-line prompts. At the last token, encoded the
residual stream via the CLT and measured activation of features associated
with in-group vs. out-group words.

| Metric | Value |
|--------|:---:|
| In-group mean activation | 0.0223 |
| Out-group mean activation | 0.0256 |
| **Ratio** | **0.87** (out-group *higher*) |

No planning signal. The model has no priming context and no indication it
should rhyme.

```bash
cargo run --release --example poetry_category_steering -- \
    --mode detect-planning \
    --model google/gemma-2-2b \
    --rhyme-pairs outputs/rhyme_pairs_all_layers.json
```

## 1.7 Planning detection V2: completion-style prompts (success)

Replaced single-line prompts with the four-line completion prompts. 24 prompts
across 10 rhyme groups. Same measurement at the last token.

| Metric | Value |
|--------|:---:|
| In-group mean activation | 0.0236 |
| Out-group mean activation | 0.0037 |
| **Ratio** | **6.45** (7.4x improvement over V1) |

Three groups show clear positive signal:

| Group | In-group mean | Out-group mean | Ratio |
|-------|:---:|:---:|:---:|
| -out | 0.172 | 0.015 | **11.60** |
| -oo | 0.020 | 0.003 | **5.82** |
| -ow | 0.055 | 0.000 | **inf** |

The strongest individual activations:

| Prompt (last line) | Target | Feature fires for | Activation |
|--------------------|--------|-------------------|:---:|
| "There is so much we do not " | so (-ow) | **go** (-ow) | **0.983** |
| "And found a hidden passage " | about (-out) | **out** (-out) | **0.247** |
| "The truth was struggling to come " | shout (-out) | **out** (-out) | **0.192** |
| "Would come to find a way back " | who (-oo) | **ou** (-oo) | **0.359** |

The signal is sparse: 19/24 prompts show zero activation for all features. But
where features fire, they fire overwhelmingly for in-group (rhyming) words.

The detection V1 and V2 results are generated by the same command above (the
`detect-planning` mode runs both single-line and completion-style probes in
sequence).

## 1.8 HTML tag experiment (minor finding)

**Motivation:** Where does a base model encounter rhyming poetry in its training
data? Likely sources include web pages hosting poems, nursery rhymes, song
lyrics, and educational sites. In HTML, line-ending words in poems are often
marked up with emphasis tags (`<em>`, `<strong>`) for visual styling or
accessibility. If the model learned to associate rhyming structure with HTML
markup, then wrapping the target word in emphasis tags might act as an
additional rhyme cue -- a signal the model already knows from its training
distribution.

Tested whether HTML emphasis tags act as rhyme markers.

| Condition | Rhyme rate | Change |
|-----------|:---:|--------|
| T=0 baseline | 21/27 (78%) | -- |
| T=0 + `<em>` | 22/27 (81%) | +1 flip |
| T=0 + `<strong>` | 22/27 (81%) | +1 flip |

The effect is small (+3 percentage points) and consistent across tags. The
model associates HTML emphasis markup with phonologically salient words.

The rhyme rate verification (baseline and HTML-tagged) uses the same
`verify-rhyming` mode as Section 1.5.

---

## Summary

The planning signal in Gemma 2 2B exists but is weaker and sparser than
Anthropic's report for Claude 3.5 Haiku:

1. **Requires context.** Without priming, ratio is 0.87. With priming, 6.45.
2. **Sparse.** Only 5/24 prompts show non-zero activation. But when features
   fire, the in-group preference is strong (up to 11.6x).
3. **Correlational, not yet causal** at this stage. The causal tests follow
   in Versions A--D (see [03-figure13-replication.md](03-figure13-replication.md)).

| Dimension | Anthropic | This work |
|-----------|-----------|-----------|
| Signal density | ~50% of poems | ~21% of prompts (5/24) |
| Planning features | Word-level (30M CLT) | Rhyme-group-level (426K CLT) |
| Prompt style | Instruction-tuned, no priming | Base model, priming required |

---

## References

- **Cross-Layer Transcoders (CLTs):** Lindsey et al., "On the Biology of a Large
  Language Model", Anthropic, 2025.
  [transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- **Sparse autoencoders / Scaling Monosemanticity:** Templeton et al., "Scaling
  Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet",
  Anthropic, 2024.
  [transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- **Circuit Tracer:** Safety Research, reference implementation for CLT
  encoding / decoding.
  [github.com/safety-research/circuit-tracer](https://github.com/safety-research/circuit-tracer)
- **CMU Pronouncing Dictionary:** Carnegie Mellon University, ~134K English words
  with ARPAbet transcriptions.
  [github.com/cmusphinx/cmudict](https://github.com/cmusphinx/cmudict)
- **ARPAbet:** Phonetic transcription system using ASCII symbols for American
  English phonemes. Vowels carry stress marks (0/1/2); consonants are unmarked.
  [Wikipedia](https://en.wikipedia.org/wiki/ARPABET)
