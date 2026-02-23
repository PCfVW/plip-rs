# Planning Circuit Hunt

*Trágos (τράγος) — "goat", the animal whose song (tragōidia) became tragedy;
here, the pursuit of the phonological circuit across two models.*

**Author:** Eric Jacopin

**Models:** Gemma 2 2B (`google/gemma-2-2b`, 26 layers) and
Llama 3.2 1B (`meta-llama/Llama-3.2-1B`, 16 layers)

**Hardware:** RTX 5060 Ti 16 GB, CUDA

---

**Contents:**
[1. Overview](#1-overview) |
[2. The GOAP analogy](#2-the-goap-analogy) |
[3. Document map](#3-document-map) |
[4. File inventory](#4-file-inventory) |
[5. Reproduction](#5-reproduction) |
[6. Acknowledgments](#6-acknowledgments)

---

## 1. Overview

The [melomētis branch](../planning-in-poems/README.md) replicated Anthropic's
Figure 13 finding: suppress + inject CLT steering at the planning site
redirects the line ending. But behavioral experiments on Llama 3.2 1B
revealed a puzzle: Llama *knows* phonological neighborhoods (the rhyme
signal is present in intermediate layers) yet fails to produce rhyming
couplets reliably. Why?

This investigation hunts for the internal planning circuit across both
models, combining correlational tools (logit lens, CLT feature mapping)
with causal intervention (layer suppression, suppress + inject sweeps).
The central finding is that rhyme planning requires two stages:

1. **Search** (both models): activate phonological neighborhood members at
   the newline position. Multiple candidates appear across layers.

2. **Commitment** (Gemma only): sustain the selected candidate through the
   final 4-5 layers so that it influences the output distribution.

Llama 3.2 1B has stage 1 (search) but lacks stage 2 (commitment). Its
planning features are crammed into L15 (82% at the last layer), leaving no
room for an intermediate planning register. The behavioral difference is
not about knowledge or exploration -- it is about holding a decision
through the residual stream to the output.

---

## 2. The GOAP analogy

The three-stage pipeline (Search, Selection, Commitment) maps onto
classical AI planning for NPCs using Goal-Oriented Action Planning (GOAP):

| Classical NPC planner | LLM rhyme circuit |
|---|---|
| Full action set | Full vocabulary |
| Precondition filtering -> candidate actions | Search: activate phonological neighbors at newline |
| Cost evaluation -> rank candidates | Selection: narrow from multiple candidates to one |
| Execute winning action | Commitment: sustain selected candidate to output layer |

A striking parallel from the game AI literature: in F.E.A.R. (the first
game to ship with GOAP), the planner fails ~80% of attempts, and plans
rarely execute fully -- the game world is too dynamic. Yet players never
notice because **alarm-actions** fire to stitch behaviors together. This
maps directly:

| GOAP game | LLM rhyme generation |
|---|---|
| Planner fails 80% of the time | Rhyme search signal dissipates most of the time |
| Plans rarely execute fully | Even successful search may not commit to output |
| Alarm-actions stitch behavior | Base LM generation produces coherent fallback |
| Player never notices failures | Reader gets a coherent line (just not rhyming) |

The rhyme circuit is a **sparse overlay** on the base language model's
syntactic template-plan. When it succeeds (search + selection +
commitment), the output is a rhyming word. When it fails at any stage, the
base LM takes over as a fallback -- producing a grammatical, coherent, but
non-rhyming continuation.

---

## 3. Document map

| Phase | Document | Model | Method |
|:-----:|----------|-------|--------|
| 1 | [01-logit-lens.md](01-logit-lens.md) | Gemma 2 2B | Per-position logit lens at probe + control positions |
| 2, 2b | [02-clt-circuit.md](02-clt-circuit.md) | Gemma 2 2B | CLT feature mapping, decoder-to-vocabulary projection, three-stage circuit model |
| 3, 4, 4b | [03-cross-model.md](03-cross-model.md) | Gemma + Llama | Cross-model logit lens, layer-by-layer diversity, the commitment hypothesis |
| 5 | [04-causal-intervention.md](04-causal-intervention.md) | Gemma 2 2B | Layer suppression: L22-25 are causally necessary for rhyming |
| 6 | [05-llama-clt-and-sweep.md](05-llama-clt-and-sweep.md) | Llama 3.2 1B | CLT mapping, vocabulary exploration, suppress + inject position sweep |

**See also:** The Gemma 2 2B CLT analysis above uses the 426K CLT
(group-level features). The melomētis branch extends this to the
[2.5M CLT](../planning-in-poems/04-2.5M-word-level.md), where every rhyme
word has its own dedicated feature (209 words at word-level resolution).
The same Figure 13 experiment at 2.5M achieves a 52.2% cross-group redirect
and a 3.78-trillion-fold spike ratio.

---

## 4. File inventory

### Examples

| Example | Phase | Role |
|---------|:-----:|------|
| `couplet_logit_lens` | 1, 4 | Per-position logit lens (both models) |
| `couplet_clt_circuit` | 2, 6a | CLT feature layer mapping (both models) |
| `couplet_clt_decoder_vocab` | 2b, 6b | Decoder-to-vocabulary projection (both models) |
| `couplet_layer_suppression` | 5 | Layer suppression causal intervention (Gemma) |
| `suppress_inject_sweep` | 6d | Suppress + inject position sweep (both models) |
| `poetry_category_steering` | 6c | Vocabulary exploration and rhyme-pair discovery |
| `ollama_rhyme_probe` | Behavioral | Couplet rhyme baseline via Ollama |
| `ollama_prefix_stability` | Behavioral | Prefix-stability probe via Ollama |

### Output files

| File | Phase | Content |
|------|:-----:|---------|
| `couplet_logit_lens.json` | 1 | Gemma logit lens (10 couplets x 26 layers) |
| `couplet_logit_lens_llama32_1b.json` | 4 | Llama logit lens (10 couplets x 16 layers) |
| `couplet_clt_circuit.json` | 2 | Gemma CLT circuit mapping |
| `couplet_clt_circuit_llama.json` | 6a | Llama CLT circuit mapping |
| `couplet_clt_decoder_vocab.json` | 2b | Gemma decoder-to-vocab projection |
| `couplet_clt_decoder_vocab_llama.json` | 6b | Llama decoder-to-vocab projection |
| `couplet_clt_analysis.m` | 2 | Mathematica analysis notebook |
| `couplet_layer_suppression.json` | 5 | Layer suppression results |
| `suppress_inject_sweep_llama.json` | 6d | Llama suppress + inject (first run) |
| `suppress_inject_sweep_llama_v2.json` | 6d | Llama suppress + inject (final) |
| `explore_vocab_llama.json` | 6c | Vocabulary scan (1.2 GB, not committed) |
| `rhyme_pairs_llama.json` | 6c | Llama rhyme pairs from CMU dictionary |
| `llama_couplet_results.json` | Behavioral | Ollama couplet baseline |
| `llama_rhyme_results.json` | Behavioral | Ollama rhyme probe results |
| `llama_rhyme_results_v2.json` | Behavioral | Ollama rhyme probe (expanded) |
| `llama_prefix_stability.json` | Behavioral | Prefix-stability results |

### Corpus files

| File | Content |
|------|---------|
| `corpus/llama_couplet_probes.json` | 15 couplet probes for Ollama |
| `corpus/llama_rhyme_probes.json` | Single-word rhyme probes |
| `corpus/llama_rhyme_probes_v2.json` | Expanded rhyme probes |
| `corpus/llama_prompts.json` | Llama-specific quatrain prompts for suppress + inject |

### Figures

| File | Content |
|------|---------|
| `figures/ee_suppressed_group_L14_that_injected.png` | Suppress -ee, inject "that" L14 |
| `figures/oo_suppressed_group_L14_that_injected.png` | Suppress -oo, inject "that" L14 |
| `figures/ore_suppressed_group_L14_that_injected.png` | Suppress -ore, inject "that" L14 |
| `figures/at_suppressed_group_L1_for_injected.png` | Suppress -at, inject "for" L1 |
| `figures/at_suppressed_group_L6_are_injected.png` | Suppress -at, inject "are" L6 |
| `figures/at_suppressed_group_L13_will_injected.png` | Suppress -at, inject "will" L13 |

### Scripts

| File | Content |
|------|---------|
| `scripts/llama_position_sweep_data.wl` | Mathematica data + plotting code for figures |

### Large regenerable files (not committed)

`outputs/explore_vocab_llama.json` (1.2 GB) is excluded via `.gitignore`.
To regenerate it, run the vocabulary exploration command from
[05-llama-clt-and-sweep.md](05-llama-clt-and-sweep.md) (Phase 6c).
This file is not needed for the core experiments -- it is an intermediate
artifact used to discover rhyme pairs.

---

## 5. Reproduction

All experiments require the model weights and CLT weights to be cached
locally from HuggingFace. Timings assume both are already downloaded.

### Gemma experiments (Phases 1, 2, 2b, 5)

```bash
# Phase 1: Logit lens
cargo run --release --example couplet_logit_lens -- --model google/gemma-2-2b

# Phase 2: CLT circuit mapping
cargo run --release --example couplet_clt_circuit

# Phase 2b: Decoder-to-vocabulary projection
cargo run --release --example couplet_clt_decoder_vocab

# Phase 5: Layer suppression
cargo run --release --example couplet_layer_suppression -- --model google/gemma-2-2b
```

### Llama experiments (Phases 4, 6)

```bash
# Phase 4: Logit lens
cargo run --release --example couplet_logit_lens -- \
    --model meta-llama/Llama-3.2-1B \
    --output outputs/couplet_logit_lens_llama32_1b.json

# Phase 6a: CLT circuit mapping
cargo run --release --example couplet_clt_circuit -- \
    --model meta-llama/Llama-3.2-1B \
    --clt mntss/clt-llama-3.2-1b-524k \
    --output outputs/couplet_clt_circuit_llama.json

# Phase 6b: Decoder-to-vocabulary projection
cargo run --release --example couplet_clt_decoder_vocab -- \
    --model meta-llama/Llama-3.2-1B \
    --clt mntss/clt-llama-3.2-1b-524k \
    --output outputs/couplet_clt_decoder_vocab_llama.json

# Phase 6d: Suppress + inject sweep
cargo run --release --example suppress_inject_sweep -- \
    --model meta-llama/Llama-3.2-1B \
    --clt-repo mntss/clt-llama-3.2-1b-524k \
    --rhyme-pairs outputs/rhyme_pairs_llama.json \
    --prompts corpus/llama_prompts.json \
    --output outputs/suppress_inject_sweep_llama_v2.json
```

### Behavioral baselines (require Ollama with llama3.2:1b)

```bash
cargo run --release --example ollama_rhyme_probe
cargo run --release --example ollama_prefix_stability
```

---

## 6. Acknowledgments

- **Anthropic** -- the original "Planning in Poems" finding
  ([Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poems),
  Lindsey et al., 2025)
- **mntss** -- the [426K CLT](https://huggingface.co/mntss/clt-gemma-2-2b-426k)
  for Gemma 2 2B and the [524K CLT](https://huggingface.co/mntss/clt-llama-3.2-1b-524k)
  for Llama 3.2 1B
- **Meta** -- the [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) model
- **Google DeepMind** -- the [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b) model
- **HuggingFace [candle](https://github.com/huggingface/candle)** -- the Rust ML framework
- **Wolfram [Mathematica](https://www.wolfram.com/mathematica/) 14.3** for figures
