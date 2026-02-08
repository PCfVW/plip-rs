# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.1.0]: https://github.com/PCfVW/plip-rs/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/PCfVW/plip-rs/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/PCfVW/plip-rs/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/PCfVW/plip-rs/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/PCfVW/plip-rs/releases/tag/v1.0.0
