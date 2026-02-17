# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] — 2026-02-17

### Added

- **Gemma 2 2B backend** (`src/forward_gemma2.rs`, 1,570 lines): Gemma 2
  architecture with softcapped attention, GQA, alternating local/global sliding
  window, and per-layer activation capture. `PlipBackend` implementation with
  all forward pass variants plus CLT injection hooks.
- **Cross-Layer Transcoder infrastructure** (`src/clt.rs`, 1,640 lines): lazy
  HuggingFace download, stream-and-free encoder loading, sparse activation
  encoding with ReLU threshold, decoder vector extraction with micro-cache, and
  `CltInjectionSpec` for steered generation. Validated against Python Circuit
  Tracer reference: 90/90 top-10 features match (max relative error 1.2e-6).
- **Poetry corpus** (`corpus/`): 780 samples (260 rhyming, 260 non-rhyming,
  260 generation prompts) across 20 rhyme groups, generated from the CMU
  Pronouncing Dictionary.
- **13 new examples** covering the full replication pipeline: CLT inspection,
  encoding, validation, logit-shift acceptance test, poetry corpus verification,
  planning detection (layer scan), CLT steering (Methods 1--6), attention
  steering, semantic category steering (6 modes), cross-mechanism evaluation,
  multi-layer position sweep, suppress + inject (Figure 13), and offline
  analysis.
- **`docs/planning-in-poems/`**: four-part write-up replicating Anthropic's
  Figure 13 from "On the Biology of a Large Language Model" (Lindsey et al.,
  2025) using entirely open tools: Gemma 2 2B, mntss/clt-gemma-2-2b-426k
  (426K features), Rust + candle, RTX 5060 Ti 16 GB. Core result: 48.3%
  cross-group probability redirect at the planning site, ten-million-fold spike.

### Changed

- `ModelArchitecture` enum extended with `Gemma2` variant; detection for
  `"gemma-2"`, `"gemma2"` (before generic `"gemma"` match).
- `PlipBackend` trait extended with CLT-related methods: `clt_logit_shift`,
  `generate_with_clt_injection`, `get_all_position_activations`,
  `token_embedding` (all with default error implementations).
- `PlipTokenizer::encode` changed from `encode(text, false)` to
  `encode(text, true)` — adds special tokens (BOS for Gemma 2) as configured
  in `tokenizer.json`.
- `FullActivationCache` added to `src/cache.rs` for all-position activation
  storage.
- `CltInjectionSpec`, `CltLayerInjection`, `CltLogitShiftResult` added to
  `src/intervention.rs`.

## [1.2.0] — 2026-02-10

### Added

- **RWKV-6 backend** (`src/forward_rwkv6.rs`): first non-transformer architecture — a
  gated-linear RNN with data-dependent decay, fixed-size recurrent state `[b,h,d,d]`
  per layer, and squared-ReLU channel-mix MLP. Model: `RWKV/v6-Finch-1B6-HF`
  (hidden=2048, 24 layers, head_size=64, 32 heads, vocab=65536).
  `PlipBackend` implementation with `forward_with_cache`, `forward_with_kv_cache`,
  `forward_with_attention` (effective attention), `forward_with_state_knockout`,
  and `generate`.
- **RWKV World tokenizer** (`src/tokenizer_rwkv.rs`): custom Trie-based greedy
  longest-match tokenizer (~350 lines) for RWKV's `rwkv_vocab_v20230424.txt` vocab
  file. Supports Python literal parsing with `\xHH`, `\uHHHH`, `\UHHHHHHHH` escapes
  and `is_bytes` distinction for string vs. bytes literals.
- **`PlipTokenizer` enum** (`src/model.rs`): unified tokenizer abstraction over
  `HuggingFace(Box<Tokenizer>)` and `Rwkv(RwkvTokenizer)` — all call sites updated
  transparently.
- **State knockout** (`StateKnockoutSpec`, `StateAblationResult` in
  `src/intervention.rs`): at a target position, suppress the kv^T state write
  while preserving decay dynamics. Equivalent to all-edge knockout for transformers.
  First statistically significant result: p=0.018 (Welch's t-test, layer 2).
- **State steering** (`StateSteeringSpec`, `StateSteeringResult` in
  `src/intervention.rs`): scale the kv^T state write at target positions by
  configurable factors for dose-response experiments.
- **Effective attention** for RWKV-6: computes `[batch, heads, seq, seq]` effective
  attention matrices from the WKV recurrence, producing lower-triangular distributions
  compatible with `AttentionCache`. Uses log-space prefix sums for numerically stable
  cumulative decay and ReLU+L1 normalisation for signed r*k products.
- **`state_ablation_experiment`** example (~780 lines): state knockout experiment
  runner with layer scanning, window scanning, sliding windows, and statistical
  analysis (Welch's t-test with self-contained t-distribution CDF).
- **`state_steering_experiment`** example: dose-response experiment for RWKV-6
  state steering with configurable scale factors.
- **State steering generation** (`PlipRwkv6::generate_with_state_steering`): steered
  prefill + normal autoregressive generation. Exposed via `PlipModel::generate_with_state_steering`
  and `PlipModel::generate_with_state_steering_details` wrappers.
- **`state_steering_generate`** example: greedy generation comparison (baseline vs
  steered) across 5 code prompts. Result: identical output at scale=3.0 (greedy).
- **`state_steering_persistence`** example (~380 lines): distance × scale × temperature
  sweep with repeated sampling. Supports `--rust-only`, `--temperature` (multi-value),
  and `--n-samples` CLI args. Key finding: distance effect confirmed (close=74%,
  medium=44%, far=36% at n=30), scale effect null (all ~50-53%).
- **Weight conversion script** (`scripts/convert_rwkv_to_safetensors.py`): one-time
  conversion from `pytorch_model.bin` to `model.safetensors` for RWKV-6 (678 tensors,
  1.60B params, 3.20 GB).
- **Python validation script** (`scripts/rwkv6_validation.py`): standalone RWKV-6
  forward pass for generating reference data (token IDs, top-10 logits, 20 greedy
  tokens) since HF's `modeling_rwkv6.py` was incompatible with `transformers` 5.1.

### Changed

- `ModelArchitecture` enum extended with `Rwkv6` variant; detection for `"rwkv"`,
  `"finch"` in model ID.
- `PlipBackend` trait extended with `forward_with_state_knockout` (default: returns
  error for non-RWKV backends).
- README updated: 7 models across 6 architectures (5 transformer + 1 RNN), RWKV-6
  in hardware requirements and supported models tables, RWKV-specific notes.
- COMMANDS.md updated: new RWKV-6 State Intervention Tools section with
  `state_ablation_experiment`, `state_steering_experiment`, `state_steering_generate`,
  and `state_steering_persistence` documentation.
- STEERING_RESULTS.md updated: new Section 8 (RWKV-6 state steering generation).
- ABLATION_RESULTS.md updated: amplification experiment (Future Work #4) marked done.
- GPU integration tests (`tests/integration.rs`): all 8 GPU-dependent tests now use
  `#[serial]` from the `serial_test` crate to prevent concurrent execution and VRAM
  exhaustion. No more need for `--test-threads=1`.

## [1.1.1] — 2026-02-09

Documentation-only release: updates experiment result documents for consistency
with the v1.1.0 Code-LLaMA and Phi-3 attention findings.

### Changed

- **RIGOR_EXPERIMENT.md**: Expanded from 4 to 6 models. Key finding: the
  Python > Rust attention effect is code-specialization-dependent (4/6 models
  replicate; Code-LLaMA reversed, Phi-3 non-significant). Updated all sections
  including Phase 3/4 tables, decision point, Section 5.3 draft, Appendix C
  (layer scan details, hypothesis validation, publishable claims, limitations),
  and Appendix D.
- **ABLATION_RESULTS.md**: Updated attention reference numbers to match current
  RIGOR_EXPERIMENT.md data (2.8–4.4× across code-specialized models). Added
  code-specialization qualifiers throughout. Noted ablation scope limited to
  4 models; Code-LLaMA and Phi-3 not yet tested with ablation.
- **STEERING_RESULTS.md**: Added code-specialization context and scope notes.
  Referenced cross-model attention analysis. Added LLaMA and Phi-3 backends to
  files table. Expanded future work with steering on non-code-specialized models.
- **ROADMAP-v1.1-model-expansion.md**: Marked Phases 1–2 and v1.1.0 release
  checklists as complete. Updated current version to v1.1.0.

## [1.1.0] — 2026-02-08

### Added

- **PlipBackend trait** (`src/model.rs`): unified interface for all model backends,
  replacing per-architecture branching with dynamic dispatch (`Box<dyn PlipBackend>`).
  Methods: `forward`, `forward_with_cache`, `forward_knockout`, `forward_steering`,
  `forward_prompt_steering`, `generate`, `chat_template`, `embed_tokens`, `lm_head`,
  `num_layers`, `num_heads`, `head_dim`, `hidden_size`.
- **Code-LLaMA 7B backend** (`src/forward_llama.rs`): LLaMA architecture with
  no-bias attention, separate `lm_head`, GQA support, RoPE (theta=1M).
  `PlipBackend` implementation with all forward pass variants and chat template.
- **Phi-3-mini-4k-instruct backend** (`src/forward_phi3.rs`): Phi-3 architecture with
  fused QKV projection (`.narrow()` split), fused gate-up MLP, head_dim=96,
  RoPE (theta=10K). `PlipBackend` implementation with all forward pass variants
  and chat template (`<|system|>...<|end|><|user|>...<|end|><|assistant|>`).
- Experiment outputs for Code-LLaMA and Phi-3: layer scans, knockout, steering,
  and generation results.

### Changed

- All existing backends (StarCoder2, Qwen2, Gemma) refactored to implement
  `PlipBackend` trait. Inherent methods preserved; trait impl delegates to them.
- `PlipModel` now stores `Box<dyn PlipBackend>` instead of a per-architecture enum.
- `ModelArchitecture` enum extended with `LLaMA` and `Phi3` variants.
- README and COMMANDS.md updated to document 6 models across 5 architectures,
  with badges, clickable table of contents, and link to
  [Deloson](https://github.com/PCfVW/deloson) live demo.
- Documentation reorganized: experiment results and roadmaps moved to `docs/`
  subdirectories. Added `CHANGELOG.md`.
- Contiguous tensor fix applied to Gemma and LLaMA backends (required when
  `repeat_kv` returns non-contiguous tensors for full MHA with `n_rep=1`).

## [1.0.3] — 2026-02-05

### Fixed

- StarCoder2 attention head count was hardcoded to 24, now reads `num_attention_heads`
  from model config. This bug had no effect on StarCoder2-3B (which has 24 heads)
  but would have caused incorrect results for other StarCoder2 variants.

## [1.0.2] — 2026-02-05

### Changed

- Removed CUDA CI job (requires GPU hardware not available in GitHub Actions runners).
- Documented CI coverage scope in README.

## [1.0.1] — 2026-02-04

### Fixed

- CI workflow: corrected Rust toolchain action and committed `Cargo.lock`.
- Formatting fixes in example files (`rustfmt`).

### Changed

- Updated GPU references from RTX 3070 Ti to RTX 5060 Ti 16GB across README,
  Dockerfile, and docker-compose configuration.

## [1.0.0] — 2026-02-04

Initial public release of PLIP-rs: **P**robing **L**anguage model
**I**nternals for **P**rogramming language understanding.

### Added

- **Core library** for attention extraction from code-specialized LLMs using
  the [candle](https://github.com/huggingface/candle) ML framework (pure Rust, CUDA).
- **3 model backends**: StarCoder2 (BigCode), Qwen2/Qwen2.5-Coder (Alibaba),
  Gemma/CodeGemma (Google) — covering 4 models across 3 architectures.
- **Universal corpus** (`corpus/universal_v2.0/`): 10 Python doctest + 10 Rust test
  samples with character-level byte offsets for model-agnostic attention extraction.
- **Attention analysis pipeline**: per-layer, per-head attention weights over
  test-relevant token spans, with Welch's t-test for Python vs. Rust comparison.
- **Layer scan** (`layer_scan_universal`): automated sweep across all layers to
  identify where architectures differentiate Python and Rust test patterns.
- **Interventions**: causal ablation (attention knockout per layer/head) and
  activation steering (add/subtract direction vectors at target layers).
- **Logit lens**: vocabulary projection at intermediate layers.
- **KV-cache inference** and **text generation** with temperature sampling.
- **20+ example binaries** covering the full experimental pipeline:
  attention extraction, layer scanning, ablation, steering, generation,
  corpus verification, and model inspection utilities.
- GitHub Actions CI (CPU build + clippy + tests).
- Docker support for containerized GPU experiments.

[1.3.0]: https://github.com/PCfVW/plip-rs/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/PCfVW/plip-rs/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/PCfVW/plip-rs/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/PCfVW/plip-rs/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/PCfVW/plip-rs/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/PCfVW/plip-rs/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/PCfVW/plip-rs/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/PCfVW/plip-rs/releases/tag/v1.0.0
