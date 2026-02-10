# PLIP-RS Roadmap: Model Expansion and Backend Refactoring

**Current version:** v1.2.0
**Target version:** v1.3.0 (tentative — RWKV-6 state delta analysis)
**Status:** v1.2.0 released (Phases 0-5 complete + state steering generation); Phase 6 not started
**Last updated:** 2026-02-10

---

## Table of Contents

- [1. Motivation](#1-motivation)
- [2. Version Strategy](#2-version-strategy)
- [3. Phase 0: PlipBackend Trait Refactor](#3-phase-0-plipbackend-trait-refactor)
  - [3.1 Current Pain Points](#31-current-pain-points)
  - [3.2 Target Architecture](#32-target-architecture)
  - [3.3 Trait Definition](#33-trait-definition)
  - [3.4 Complications and Solutions](#34-complications-and-solutions)
  - [3.5 Files Touched](#35-files-touched)
  - [3.6 Validation](#36-validation)
- [4. Phase 1: Code-LLaMA 7B](#4-phase-1-code-llama-7b)
  - [4.1 Model Details](#41-model-details)
  - [4.2 Architecture Mapping](#42-architecture-mapping)
  - [4.3 Implementation Checklist](#43-implementation-checklist)
  - [4.4 Validation](#44-validation)
- [5. Phase 2: Phi-3-mini-4k-instruct](#5-phase-2-phi-3-mini-4k-instruct)
  - [5.1 Model Details](#51-model-details)
  - [5.2 Architecture Mapping](#52-architecture-mapping)
  - [5.3 Implementation Checklist](#53-implementation-checklist)
  - [5.4 Validation](#54-validation)
- [6. Release: v1.1.0](#6-release-v110)
- [7. Phase 3: RWKV-6 Basic Inference](#7-phase-3-rwkv-6-basic-inference)
  - [7.1 Model Details](#71-model-details)
  - [7.2 Candle Issue #3044](#72-candle-issue-3044)
  - [7.3 Architecture Differences from Transformers](#73-architecture-differences-from-transformers)
  - [7.4 Forward Pass Structure](#74-forward-pass-structure)
  - [7.5 Tokenizer Verification](#75-tokenizer-verification)
  - [7.6 Implementation Checklist](#76-implementation-checklist)
  - [7.7 Validation](#77-validation)
- [8. Phase 4: RWKV-6 State Knockout (Approach 3)](#8-phase-4-rwkv-6-state-knockout-approach-3)
  - [8.1 Intervention Design](#81-intervention-design)
  - [8.2 Infrastructure Changes](#82-infrastructure-changes)
  - [8.3 Implementation Checklist](#83-implementation-checklist)
  - [8.4 Validation](#84-validation)
- [9. Phase 5: RWKV-6 Effective Attention (Approach 1)](#9-phase-5-rwkv-6-effective-attention-approach-1)
  - [9.1 Mathematical Foundation](#91-mathematical-foundation)
  - [9.2 Prior Art](#92-prior-art)
  - [9.3 Numerical Stability](#93-numerical-stability)
  - [9.4 Implementation Checklist](#94-implementation-checklist)
  - [9.5 Validation](#95-validation)
- [10. Release: v1.2.0](#10-release-v120)
- [11. Phase 6: RWKV-6 State Delta Analysis (Approach 2) — Optional](#11-phase-6-rwkv-6-state-delta-analysis-approach-2--optional)
  - [11.1 Three Metrics](#111-three-metrics)
  - [11.2 Cross-Validation with Knockout](#112-cross-validation-with-knockout)
  - [11.3 Implementation Checklist](#113-implementation-checklist)
- [12. Dependency Graph](#12-dependency-graph)
- [13. Risk Register](#13-risk-register)
- [14. Model Summary After Completion](#14-model-summary-after-completion)
- [15. References](#15-references)

---

## 1. Motivation

PLIP-RS v1.0.3 supports 3 architecture families (StarCoder2, Qwen2, Gemma) across 4 models. This roadmap expands coverage to **6 architecture families across 7 models** (5 transformer + 1 gated-linear RNN), including a non-transformer architecture (RWKV-6).

Goals:

1. **Model-agnosticity**: Strengthen the claim that PLIP-RS findings generalise across architectures by testing on LLaMA and Phi-3 families.
2. **Architectural diversity**: Add RWKV-6 (gated-linear RNN) to compare transformer attention mechanisms with recurrent state dynamics.
3. **Maintainability**: Refactor the model backend to make adding new models plug-and-play, eliminating ~90 lines of boilerplate per new model.

---

## 2. Version Strategy

| Version | Content | Scope |
|---------|---------|-------|
| **v1.1.0** | Trait refactor + Code-LLaMA 7B + Phi-3-mini | Internal refactor (no public API change) + 2 new transformer models |
| **v1.2.0** | RWKV-6 (inference + knockout + effective attention) | New architecture paradigm + new intervention type (`StateKnockout`) |
| v1.3.0 (tentative) | RWKV-6 state delta analysis (Approach 2) | New metrics (write strength, persistence curves, SVD channel analysis) |

**Rationale:** The trait refactor and new transformer models are grouped into v1.1.0 because the refactor enables the models and both are backward-compatible. RWKV-6 gets v1.2.0 because it introduces a new intervention type (`StateKnockout`) and a fundamentally different model paradigm. State delta analysis (Approach 2) is tentative and scientifically the highest risk, so it gets its own version if pursued.

---

## 3. Phase 0: PlipBackend Trait Refactor

### 3.1 Current Pain Points

Adding a new model currently requires touching **~21 sites** in [model.rs](src/model.rs):

- 11 match arms in `ModelBackend` impl (lines 107–211)
- 1 `MemoryLimitedGeneration` impl block (~10 lines, identical boilerplate)
- 1 match arm in `from_pretrained_with_arch` (lines 280–293)
- 1 match arm in `apply_chat_template` (lines 340–361)
- 2 match blocks in `generate_with_steering` (lines 938–960, 978–1003)
- 1 match arm in `generate_with_memory_limit` (lines 1210–1235)
- 1 match arm in `generate_with_details` (lines 1126–1136)
- 2 additions to `ModelArchitecture` enum + `from_model_id` (lines 22–49)
- 1 `use` import

Total: **~90 lines of trivial boilerplate per new model.**

### 3.2 Target Architecture

Replace `ModelBackend` enum dispatch with a `PlipBackend` trait. After refactoring, adding a new model requires:

1. Write `forward_newarch.rs` implementing `PlipBackend` — already required
2. Add one variant to `ModelArchitecture` + one line in `from_model_id()` — for detection
3. Add one match arm in `from_pretrained_with_arch` — `Box::new(PlipNewArch::load(...)?)`

Three touch points. No other changes to model.rs.

### 3.3 Trait Definition

The trait unifies `ModelBackend` dispatch and `MemoryLimitedGeneration`:

```rust
pub trait PlipBackend {
    // --- Metadata ---
    fn n_layers(&self) -> usize;
    fn d_model(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn n_heads(&self) -> usize;

    // --- Forward passes ---
    fn forward_with_cache(&self, input_ids: &Tensor) -> Result<(Tensor, ActivationCache)>;
    fn forward_with_attention(&self, input_ids: &Tensor) -> Result<(Tensor, AttentionCache)>;
    fn forward_with_intervention(
        &self, input_ids: &Tensor, spec: &KnockoutSpec,
    ) -> Result<(Tensor, AttentionCache)>;

    // --- Logit lens ---
    fn logit_lens(&self, activation: &Tensor) -> Result<Tensor>;
    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor>;
    fn logit_lens_top_k(&self, activation: &Tensor, k: usize) -> Result<Vec<(u32, f32)>>;

    // --- Generation ---
    fn new_kv_cache(&self) -> KVCache;
    fn forward_with_kv_cache(&self, input_ids: &Tensor, kv_cache: &mut KVCache)
        -> Result<Tensor>;
    fn generate(
        &self, prompt_ids: &[u32], max_tokens: usize, temperature: f32,
        stop_tokens: &[u32], device: &Device,
    ) -> Result<Vec<u32>>;

    // --- Optional capabilities (default: unsupported) ---
    fn forward_with_steering(
        &self, _input_ids: &Tensor, _spec: &SteeringSpec,
    ) -> Result<(Tensor, AttentionCache)> {
        anyhow::bail!("Steering not supported for this architecture")
    }

    fn generate_with_prompt_steering(
        &self, _prompt_ids: &[u32], _max_tokens: usize, _temperature: f32,
        _stop_tokens: &[u32], _spec: &SteeringSpec, _device: &Device,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("Prompt steering not supported for this architecture")
    }

    fn chat_template(
        &self, _prompt: &str, _system_prompt: Option<&str>,
    ) -> Option<String> {
        None  // No chat template by default
    }
}
```

`PlipModel` stores `Box<dyn PlipBackend>` instead of `ModelBackend` enum.

### 3.4 Complications and Solutions

**C1: Steering is optional for RWKV.**
Solved by default trait methods returning errors (see trait definition above). All three existing transformer backends override these methods. RWKV does not.

**C2: KV-cache vs. recurrent state.**
The `generate()` method on the trait is self-contained — each backend manages its own inference state internally. `PlipModel` never touches the cache directly, except in `generate_with_memory_limit_impl` (lines 1246–1306). This method moves into the trait as a default implementation with memory-limit enforcement. The `sample_token` / `argmax` / `sample_with_temperature` helpers (lines 1309–1360) must be extracted from `PlipModel` into free functions so the trait's default `generate_with_memory_limit` can call them without access to `&self` on `PlipModel`.

**C3: `ModelArchitecture` used outside model.rs.**
Verified: `ModelArchitecture` appears only in `model.rs` (defined) and `lib.rs` (re-exported as public API). No behavior branching outside `model.rs`. The enum is safe to extend with new variants without side effects.

**C4: Chat template in PlipModel vs. backend.**
Currently `apply_chat_template` in `PlipModel` matches on `ModelArchitecture`. After refactor, each backend provides its own `chat_template()` method. `PlipModel::apply_chat_template` becomes:
```rust
fn apply_chat_template(&self, prompt: &str, system_prompt: Option<&str>) -> String {
    self.model.chat_template(prompt, system_prompt)
        .unwrap_or_else(|| prompt.to_string())
}
```

### 3.5 Files Touched

| File | Changes |
|------|---------|
| [src/model.rs](src/model.rs) | Define `PlipBackend` trait; replace `ModelBackend` enum with `Box<dyn PlipBackend>`; delete all match-dispatch methods; extract sampling helpers to free functions; simplify `apply_chat_template`, `generate_with_steering`, `generate_with_memory_limit`; delete `MemoryLimitedGeneration` trait |
| [src/forward.rs](src/forward.rs) | Add `impl PlipBackend for PlipStarCoder2` block (~20 lines delegating to existing methods) |
| [src/forward_qwen2.rs](src/forward_qwen2.rs) | Add `impl PlipBackend for PlipQwen2` block (~20 lines) |
| [src/forward_gemma.rs](src/forward_gemma.rs) | Add `impl PlipBackend for PlipGemma` block (~20 lines) |

**Net effect:** model.rs shrinks by ~150–200 lines. Each forward file gains ~20 lines.

### 3.6 Validation

- All existing tests pass unchanged (public API of `PlipModel` is identical).
- Run existing examples against all 4 currently-supported models to verify no regression.
- Verify `cargo clippy` and `cargo test` pass.

---

## 4. Phase 1: Code-LLaMA 7B

### 4.1 Model Details

| Property | Value |
|----------|-------|
| Model ID | `codellama/CodeLlama-7b-hf` (or `-Instruct-hf` / `-Python-hf` variants) |
| Architecture | LLaMA (Meta) |
| Parameters | 7B (~14GB fp16) |
| Candle support | Mature — flagship model family |
| Code-specialised | Yes |
| Safetensors | Yes (sharded) |

### 4.2 Architecture Mapping

| Component | Code-LLaMA | Closest existing backend |
|-----------|-----------|------------------------|
| Config fields | Same as Qwen2 (hidden_size, num_attention_heads, num_key_value_heads, etc.) | Qwen2 |
| RoPE theta | Likely 1,000,000.0 — **verify in config.json** (may differ between base/instruct/Python variants) | Qwen2 (if 1M) or StarCoder2 (if 10K) |
| Normalisation | RMSNorm (standard, no +1 bias) | StarCoder2 |
| MLP | SwiGLU (gate_proj, up_proj, down_proj with SiLU) | Qwen2 (identical) |
| Bias | No bias on Q/K/V/O projections | Gemma |
| GQA | Yes (num_key_value_heads < num_attention_heads for 7B) | Qwen2, Gemma |
| Tied embeddings | No (`lm_head` is separate) | StarCoder2 |
| Weight names | `model.layers.N.self_attn.{q,k,v,o}_proj.weight` | Qwen2, Gemma (identical) |

**Assessment:** Code-LLaMA is architecturally closest to Qwen2. The forward pass can be adapted from `forward_qwen2.rs` with minor changes (no Q/K/V bias, untied embeddings). RoPE theta must be verified from the actual `config.json` — the value differs across Code-LLaMA variants.

### 4.3 Implementation Checklist

- [x] Download and inspect `config.json` from `codellama/CodeLlama-7b-hf`
- [x] Create `src/forward_llama.rs` with `LlamaConfig` struct
- [x] Implement `PlipBackend` for `PlipLlama`
- [x] Handle sharded weights (reuse Qwen2's shard-detection logic)
- [x] Add `Llama` variant to `ModelArchitecture` + detection for `"llama"`, `"codellama"` in model ID
- [x] Add match arm in `from_pretrained_with_arch`
- [x] Test basic inference (greedy generation on a short prompt)
- [x] Test attention extraction (verify `[batch, heads, seq, seq]` output shape)
- [x] Test knockout (reuse existing knockout test patterns)
- [x] Run existing PLIP experiments on Code-LLaMA and compare with transformer baselines

### 4.4 Validation

- Compare greedy output of first 20 tokens against HuggingFace `transformers` Python library on the same prompt.
- Verify attention patterns are non-trivial (not uniform, not degenerate).
- Run a knockout experiment on a known Python code sample and verify KL divergence is non-zero.

---

## 5. Phase 2: Phi-3-mini-4k-instruct

### 5.1 Model Details

| Property | Value |
|----------|-------|
| Model ID | `microsoft/Phi-3-mini-4k-instruct` |
| Architecture | Phi-3 (Microsoft) |
| Parameters | 3.8B (~7.6GB fp16) |
| Candle support | Mature |
| Code-specialised | No (but trained with significant code data) |
| Safetensors | Yes |

### 5.2 Architecture Mapping

| Component | Phi-3 | Closest existing backend |
|-----------|-------|------------------------|
| Config fields | Similar to LLaMA with some Phi-specific additions | Qwen2 |
| RoPE | Standard RoPE (may use SuRoPE / longrope variant — verify in config) | Qwen2 |
| Normalisation | RMSNorm (standard) | StarCoder2 |
| MLP | SwiGLU (gate_up_proj fused, then down_proj) | Qwen2 — but **fused gate+up projection** |
| Q/K/V projections | **Fused QKV** (`qkv_proj` single weight) | **None** — new pattern |
| GQA | Yes | Qwen2, Gemma |
| Weight names | `model.layers.N.self_attn.qkv_proj.weight` (fused) | **Different** from all existing backends |

**Key difference: Fused QKV projection.** Phi-3 uses a single `qkv_proj` weight matrix instead of separate `q_proj`, `k_proj`, `v_proj`. The forward pass must split the output into Q, K, V segments:
```
qkv = qkv_proj(x)                     # [batch, seq, (q_dim + k_dim + v_dim)]
q, k, v = split(qkv, [q_dim, k_dim, v_dim], dim=-1)
```

Similarly, the MLP uses a fused `gate_up_proj` instead of separate `gate_proj` and `up_proj`.

### 5.3 Implementation Checklist

- [x] Download and inspect `config.json` from `microsoft/Phi-3-mini-4k-instruct`
- [x] Verify RoPE variant (standard vs. SuRoPE/longrope) — check for `rope_scaling` in config
- [x] Create `src/forward_phi3.rs` with `Phi3Config` struct
- [x] Implement fused QKV split logic in attention
- [x] Implement fused gate+up split logic in MLP
- [x] Implement `PlipBackend` for `PlipPhi3`
- [x] Add `Phi3` variant to `ModelArchitecture` + detection for `"phi-3"`, `"phi3"` in model ID
- [x] Add match arm in `from_pretrained_with_arch`
- [x] Implement Phi-3 chat template (`<|user|>\n{prompt}<|end|>\n<|assistant|>\n`)
- [x] Test basic inference
- [x] Test attention extraction
- [x] Test knockout
- [x] Run PLIP experiments

### 5.4 Validation

Same as Code-LLaMA: compare greedy output against HuggingFace Python, verify attention shape and knockout KL divergence.

---

## 6. Release: v1.1.0

**Content:** Trait refactor + Code-LLaMA 7B + Phi-3-mini

**Checklist before release:**
- [x] All existing tests pass
- [x] All 6 models produce correct inference output
- [x] Attention extraction works for all 6 models
- [x] Knockout experiments produce non-trivial results for all 6 models
- [x] `cargo clippy` clean
- [x] Update `Cargo.toml` version to `1.1.0`
- [x] Update README model support table

**Architecture coverage after v1.1.0:**

| Model | Architecture | Size | Family |
|-------|-------------|------|--------|
| Qwen2.5-Coder-7B-Instruct | Qwen2 | 7B | Qwen |
| Qwen2.5-Coder-3B-Instruct | Qwen2 | 3B | Qwen |
| StarCoder2-3B | StarCoder2 | 3B | BigCode |
| CodeGemma-7B-it | Gemma | 7B | Google |
| Code-LLaMA-7B | LLaMA | 7B | Meta |
| Phi-3-mini-4k-instruct | Phi-3 | 3.8B | Microsoft |

**5 architecture families, 6 models.**

---

## 7. Phase 3: RWKV-6 Basic Inference

### 7.1 Model Details

| Property | Value |
|----------|-------|
| Model ID | `RWKV/v6-Finch-1B6-HF` |
| Architecture | RWKV-6 / Finch (gated-linear RNN) |
| Parameters | 1.6B (~3.2GB fp16) |
| Candle support | Implemented but buggy (see below) |
| Code-specialised | No (World model, multilingual) |
| Safetensors | **No** — only `pytorch_model.bin` shipped; requires one-time conversion via `scripts/convert_rwkv_to_safetensors.py` |
| Config | `hidden_size=2048`, `num_hidden_layers=24`, `head_size=64` (32 heads), `vocab_size=65536` |

### 7.2 Candle Issue #3044

GitHub issue [#3044](https://github.com/huggingface/candle/issues/3044) reports nonsensical output from candle's RWKV example. **This is a model-level bug, not a framework bug** — other candle models work correctly on the same system. Likely causes:

- Confusing `num_attention_heads` semantics (means `head_size` in RWKV's HF config, not head count)
- Unofficial weight re-uploads with missing config fields
- Potential saturation in `exp(-exp(w))` decay computation

**Impact on PLIP-RS:** None as a blocker. PLIP-RS writes its own forward passes from scratch, using candle only for tensor operations (matmul, exp, etc.), which work correctly. We would not use candle's `rwkv_v6.rs`.

**Consequence:** We cannot use candle's RWKV implementation as a correctness reference. Validation must be done against the HuggingFace Python implementation (`modeling_rwkv6.py` from `RWKV/v6-Finch-1B6-HF`).

### 7.3 Architecture Differences from Transformers

| Component | Transformer (existing) | RWKV-6 |
|-----------|----------------------|--------|
| Position encoding | RotaryEmbedding | None (position implicit in recurrence) |
| Normalisation | RMSNorm | LayerNorm + GroupNorm in attention |
| Attention core | Q/K/V, scaled dot-product, softmax, `[b,h,s,s]` | R/K/V/gate, per-timestep state update loop, **no attention matrix** |
| Projections | Separate Q, K, V, O | Separate R, K, V, gate, O + data-dependent mixing via low-rank matrices (`time_mix_w1`/`w2`) |
| Time decay | N/A | Data-dependent: `time_decay` through `time_decay_w1`/`w2` producing per-timestep $w_t$ |
| MLP activation | GELU / SiLU / SwiGLU | Squared ReLU (`relu(x)^2`) |
| MLP structure | gate/up/down (3 projections) | key/value/receptance (channel mix with token-shift) |
| Inference state | Growing KV-cache `[b,h,seq,d]` | Fixed-size state `[b,h,d,d]` per layer |

### 7.4 Forward Pass Structure

```
For each timestep t:
    1. Token-shift: interpolate current and previous hidden states
       via data-dependent mixing (5 components: mw, mk, mv, mr, mg)
    2. Project to R, K, V, gate from mixed inputs
    3. Compute data-dependent decay w_t via time_decay_w1/w2
    4. State update: S_t = diag(exp(w_t)) * S_{t-1} + k_t^T * v_t
    5. Output: o_t = GroupNorm(r_t * (S_t @ ...))
    6. Apply gate (SiLU) and output projection
```

This is a sequential loop over sequence length. For PLIP-RS experiments (typically < 1024 tokens), this is acceptable.

**Estimated size:** 600–800 lines for `forward_rwkv6.rs`, compared to ~1,200–1,400 for transformer backends. **Actual:** ~1,005 lines (Phase 3), ~1,110 lines with Phase 4 state knockout additions.

### 7.5 Tokenizer Verification — Resolved

**Finding:** `RWKV/v6-Finch-1B6-HF` does **not** ship a `tokenizer.json`. It uses a custom Trie-based greedy longest-match tokenizer (`rwkv_vocab_v20230424.txt`) incompatible with the HuggingFace `tokenizers` crate (BPE/Unigram/WordPiece only).

**Solution implemented:**
1. Custom Rust Trie tokenizer in `src/tokenizer_rwkv.rs` (~350 lines including Python literal parser with `\xHH`, `\uHHHH`, `\UHHHHHHHH` escapes and `is_bytes` distinction for string vs. bytes literals)
2. `PlipTokenizer` enum in `src/model.rs` abstracting over `HuggingFace(Box<Tokenizer>)` and `Rwkv(RwkvTokenizer)` — all 18+ call sites updated transparently

### 7.6 Implementation Checklist

**Pre-work (Python, one-time):**
- [x] Write weight conversion script `scripts/convert_rwkv_to_safetensors.py` (1B6 only ships `pytorch_model.bin`, no safetensors)
- [x] Run conversion: 678 tensors, 1.60B params, 3.20 GB safetensors
- [x] Write standalone Python validation script `scripts/rwkv6_validation.py` (HF's custom `modeling_rwkv6.py` was incompatible with `transformers` 5.1; standalone forward pass used instead)
- [x] Capture reference data in `scripts/rwkv6_reference.json` (token IDs, top-10 logits, 20 greedy-generated tokens)

**Tokenizer (`src/tokenizer_rwkv.rs`):**
- [x] Verify tokenizer compatibility → **incompatible** (no `tokenizer.json`, see 7.5)
- [x] Implement RWKV Trie tokenizer in Rust: `RwkvTokenizer` with `from_file`, `encode`, `decode`, `encode_with_offsets`, `get_vocab`
- [x] Implement Python literal parser (`parse_python_literal`) with `is_bytes` flag for string vs. bytes literal distinction (`'\x80'` = U+0080 = 2 UTF-8 bytes vs. `b'\x80'` = 1 raw byte)
- [x] Add `\uHHHH` (4-digit) and `\UHHHHHHHH` (8-digit) Unicode escape support
- [x] Create `PlipTokenizer` enum in `src/model.rs` unifying HuggingFace and RWKV tokenizers; update all call sites

**Model backend (`src/forward_rwkv6.rs`):**
- [x] Download and inspect `config.json` from `RWKV/v6-Finch-1B6-HF`
- [x] Inspect weight names: `rwkv.` prefix for backbone, `head.` for LM head (678 tensors confirmed)
- [x] Create `src/forward_rwkv6.rs` with `Rwkv6Config` struct (hardcoded `TIME_MIX_EXTRA_DIM=32`, `TIME_DECAY_EXTRA_DIM=64`)
- [x] Implement token-shift (data-dependent mixing via `time_maa_w1`/`w2`) — required `.contiguous()` after `transpose(0,1)` for CUDA matmul
- [x] Implement data-dependent time decay via `time_decay_w1`/`w2`
- [x] Implement WKV state update loop (core recurrence, state in F32 for numerical stability)
- [x] Implement GroupNorm manually in attention output (~15 lines; eps = `layer_norm_epsilon * head_size_divisor^2`)
- [x] Implement channel-mix (MLP equivalent with squared ReLU)
- [x] Implement `PlipBackend` for `PlipRwkv6` — `forward_with_attention` returns error (Phase 5); `forward_with_intervention` returns error (state knockout replaces it per Phase 4)
- [x] Handle fixed-size state encoded in KVCache: `keys[i]` = concat(attn_x, ffn_x), `values[i]` = attn_kv

**Registration (`src/model.rs`, `src/lib.rs`):**
- [x] Add `Rwkv6` variant to `ModelArchitecture` + detection for `"rwkv"`, `"finch"` in model ID
- [x] Add match arm in `from_pretrained_with_arch` (loads vocab file for RWKV, `tokenizer.json` for all others)
- [x] Add `pub mod forward_rwkv6` / `pub mod tokenizer_rwkv` and re-exports in `src/lib.rs`

**Validation:**
- [x] Tokenizer test: token IDs match Python reference exactly (7 tokens for `"def fibonacci(n):\n    "`)
- [x] Forward logits test (GPU, F32): top-10 logits match Python within abs < 1.3e-5
- [x] Generation test (GPU, F32): 20/20 greedy tokens match Python exactly
- [x] Regression: all 48 unit tests + 4 integration tests pass
- [x] `cargo clippy` clean (zero warnings)

### 7.7 Validation — Complete

**Validated against standalone Python reference** (not HF `AutoModel`, which was incompatible with `transformers` 5.1).

| Test | Method | Result |
|------|--------|--------|
| Tokenizer | Compare token IDs for `"def fibonacci(n):\n    "` | 7/7 tokens match exactly |
| Forward logits | Compare top-10 logit values (GPU, F32) | All within abs < 1.3e-5 |
| Generation | Compare 20 greedy tokens (GPU, F32, temperature=0) | 20/20 match exactly |

**Reference data:** `scripts/rwkv6_reference.json` (generated by `scripts/rwkv6_validation.py`)

**Lessons learned:**
- HF's custom `modeling_rwkv6.py` required `bitsandbytes` and had a `RuntimeError` (tensor size mismatch in `extract_key_value`). Standalone Python forward pass was written instead (~250 lines).
- Data-dependent mixing requires `.contiguous()` after `transpose(0,1)` before batched matmul on CUDA (candle rejects non-contiguous tensors).
- Python literal parser needed `is_bytes` flag: string literal `'\x80'` encodes as UTF-8 (2 bytes), bytes literal `b'\x80'` is 1 raw byte. Also needed `\uHHHH` and `\UHHHHHHHH` Unicode escape support.

---

## 8. Phase 4: RWKV-6 State Knockout (Approach 3)

**Prerequisite:** Phase 3 (working RWKV-6 inference)

State knockout is the lowest-risk RWKV interpretability approach, with direct methodological precedent in Mamba Knockout (Endy et al., ACL 2025).

### 8.1 Intervention Design

At the marker position $m$, replace the normal state update:

$$S_m = k_m v_m^T + D_m \cdot S_{m-1} \quad\longrightarrow\quad S_m = D_m \cdot S_{m-1}$$

where $D_m = \text{diag}(\exp(-\exp(w_m)))$ is the data-dependent decay. The $k_m v_m^T$ write is suppressed while preserving the forgetting dynamics — matching the Mamba Knockout semantics (Endy et al., ACL 2025). All subsequent tokens process a state that never saw the marker. The metric is KL divergence between baseline and knocked-out output distributions — identical to the transformer knockout metric in Table 6 of the PLIP paper.

**Semantics:** This is equivalent to the **all-edge knockout** (type 2) from the PLIP paper — making the marker invisible to all future positions. There is no RWKV equivalent of specific-edge knockout (type 1).

### 8.2 Infrastructure Changes

**New intervention type.** The existing `KnockoutSpec` uses `AttentionEdge { from_pos, to_pos }`, which does not map to position-only state knockout. Add:

```rust
// In intervention.rs
pub struct StateKnockoutSpec {
    positions: Vec<usize>,     // Positions where state update is skipped
    layers: LayerSpec,         // Which layers to apply knockout
}
```

**PlipBackend extension.** Add an optional method:

```rust
fn forward_with_state_knockout(
    &self, _input_ids: &Tensor, _spec: &StateKnockoutSpec,
) -> Result<Tensor> {
    anyhow::bail!("State knockout not supported for this architecture")
}
```

Note: the return type is `Result<Tensor>` (ln_out-normalized hidden states), not `Result<(Tensor, AttentionCache)>`. The hidden states are fed through `PlipModel::compute_logits()` for last-token extraction and vocab projection. State knockout does not require attention patterns. Effective attention computation (Phase 5) is a separate capability added later. Only `PlipRwkv6` overrides this. Transformer backends use the default (unsupported).

**PlipModel integration.** Add a `forward_with_state_knockout` method that handles tokenisation and calls the backend. This method runs both baseline and knocked-out forward passes and computes KL divergence, following the existing `forward_with_intervention` pattern but returning logits-only results.

### 8.3 Implementation Checklist

**Intervention types (`src/intervention.rs`):**
- [x] Define `StateKnockoutSpec` — builder pattern with `positions: Vec<usize>`, `layers: LayerSpec`; methods: `position()`, `positions()`, `layer()`, `layers()`, `layer_range()`, `applies_to_layer()`, `position_set() -> HashSet<usize>`, `validate(n_layers, seq_len)`
- [x] Define `StateAblationResult` — fields: `baseline_logits`, `ablated_logits`, `spec`; methods: `kl_divergence()`, `logit_diff()`, `top_changed_tokens()`
- [x] Add 4 unit tests: `test_state_knockout_spec_builder`, `test_state_knockout_spec_validation`, `test_state_knockout_position_set`, `test_state_knockout_applies_to_layer`

**PlipBackend trait (`src/model.rs`):**
- [x] Add `forward_with_state_knockout` default method (returns error for non-RWKV backends)
- [x] Add `StateKnockoutSpec`, `StateAblationResult` to intervention imports

**RWKV-6 backend (`src/forward_rwkv6.rs`):**
- [x] Refactor `Rwkv6Attention::forward` → shared `forward_inner` with `Option<&HashSet<usize>>` parameter
- [x] Add `Rwkv6Attention::forward_with_knockout` (delegates to `forward_inner` with knockout set)
- [x] Core knockout logic: at knocked-out positions, `state = decay * state` (suppress kv write, preserve forgetting)
- [x] Add `Rwkv6Block::forward_with_knockout` (delegates attention knockout; FFN not knocked out — stateless)
- [x] Add `PlipRwkv6::forward_with_state_knockout` inherent method (per-layer conditional dispatch)
- [x] Wire into `impl PlipBackend for PlipRwkv6`

**PlipModel wrapper (`src/model.rs`):**
- [x] Add `PlipModel::forward_with_state_knockout` — dual forward pass (baseline via `forward_with_cache` + knockout), returns `StateAblationResult`
  - **Bugfix:** Initial implementation used `forward_with_kv_cache` for the baseline, which returns 2D logits for RWKV-6 (already projected to vocab) — incompatible with `compute_logits()` which expects 3D hidden states. Fixed to use `forward_with_cache`.

**Re-exports (`src/lib.rs`):**
- [x] Re-export `StateKnockoutSpec`, `StateAblationResult`

**Integration tests (`tests/integration.rs`):**
- [x] `test_state_knockout_spec_validation` — no GPU, validates builder and error cases
- [x] `test_rwkv6_state_knockout_kl` — GPU, knocks out position 0 on `"def add(a, b):\n    return a + b"`, asserts KL > 0

**Clippy compliance:**
- [x] Fixed `map_or` → `is_some_and` (clippy suggestion)
- [x] Added `#[allow(clippy::too_many_lines)]` on `forward_inner` (104 lines, cohesive WKV loop)
- [x] Fixed pre-existing `too_many_lines` warning on `parse_python_literal` in `tokenizer_rwkv.rs`
- [x] Zero warnings with `cargo clippy -- -W clippy::pedantic`

**Experiment script (`examples/state_ablation_experiment.rs`):**
- [x] Create `state_ablation_experiment.rs` (~780 lines) mirroring `ablation_experiment.rs` for transformers
- [x] CLI args: `--model`, `--layer`, `--scan-layers`, `--scan-windows`, `--slide-window`, `--layer-start/--layer-end`, `--output`, `--verbose`, `--include-baselines`
- [x] Structs: `SampleResult` (with `layers_knocked_out` instead of `edges_knocked_out`), `ExperimentResults` (with `architecture` and `intervention_type` instead of `n_heads`)
- [x] Statistics: `compute_stats`, `welch_t_test`, self-contained t-distribution CDF
- [x] Add `[[example]]` entry in `Cargo.toml`
- [x] Zero clippy pedantic warnings
- [x] GPU-validated: layer scan (24 layers), full experiment at layer 2 (p=0.018), window scan (KL grows 0.34% → 1.93%)

**Not planned but required:**
- [x] Design decision: `state = decay * state` (not `state = state`) — preserves forgetting dynamics, matches Mamba Knockout (Endy et al., ACL 2025) semantics
- [x] `forward_with_state_knockout` returns `ln_out`-normalized hidden states (not projected logits), compatible with `PlipModel::compute_logits` for last-token extraction
- [x] `LayerSpec::Range` uses inclusive end — test must use `layer_range(0, n_layers - 1)`, not `layer_range(0, n_layers)`

### 8.4 Validation

- [x] KL divergence = **4.96** when knocking out position 0 across all 24 layers on `"def add(a, b):\n    return a + b"` (GPU, F32)
- [x] Compare Python vs. Rust knockout KL divergences — **confirmed**: Python mean KL = 0.111%, Rust mean KL = 0.024% (4.6× ratio), p = 0.018 (Welch's t-test, **first statistically significant result** across all 5 models)
- [x] Cross-reference with transformer all-edge knockout results in ABLATION_RESULTS.md — **done**: RWKV-6 state knockout added to all cross-model tables and interpretation sections

**Post-Phase 4 Extension — State Steering Generation (v1.2.0):**
- [x] `forward_with_state_steering_kv_cache` — steered prefill with KVCache persistence
- [x] `generate_with_state_steering` — steered prefill + normal generation loop
- [x] `PlipModel::generate_with_state_steering` / `generate_with_state_steering_details` wrappers
- [x] `state_steering_generate` example — greedy baseline vs steered comparison (5 prompts, identical output at scale=3.0)
- [x] `state_steering_persistence` example — distance × scale × temperature sweep (n=30). Key finding: distance effect confirmed (close=74%, medium=44%, far=36%), scale effect null (amplification has no generation effect)

---

## 9. Phase 5: RWKV-6 Effective Attention (Approach 1)

**Prerequisite:** Phase 3 (working RWKV-6 inference)

Compute the effective attention matrix for RWKV-6, producing `[batch, heads, seq, seq]` tensors compatible with `AttentionCache`.

### 9.1 Mathematical Foundation

The effective attention is derived from the WKV recurrence. The output at position $t$ uses the accumulated state $S_{t-1}$:

$$o_t = r_t^\top \cdot \bigl[\text{diag}(u) \cdot k_t v_t^\top + S_{t-1}\bigr]$$

Unrolling $S_{t-1} = \sum_{i=0}^{t-1} \bigl(\prod_{j=i+1}^{t-1} \text{diag}(d_j)\bigr) \cdot k_i v_i^\top$ where $d_j = \exp(-\exp(w_j))$:

$$o_t = \sum_{i=0}^{t} \alpha_{\text{raw}}(t,i,h) \cdot v_i$$

where the **effective attention weight** per head $h$ is:

$$\alpha_{\text{raw}}(t,i,h) = \sum_d r_t[h,d] \cdot k_i[h,d] \cdot \prod_{j=i+1}^{t-1} d_j[h,d] \quad \text{for } i < t$$

$$\alpha_{\text{raw}}(t,t,h) = \sum_d r_t[h,d] \cdot k_t[h,d] \cdot u[h,d] \quad \text{(diagonal)}$$

**Key difference from RWKV-4:** Both $w_t$ and $k_t$ are **vectors** of dimension `head_size=64` per head, not scalars. The effective attention aggregates per-channel contributions via a dot product over the head-channel dimension $d$. The decay product runs from $i+1$ to $t-1$ (not $t$) because the output uses $S_{t-1}$.

**Normalisation:** Since $\alpha_{\text{raw}}$ can be negative (signed $r$ and $k$ components), we apply ReLU followed by L1 normalisation:

$$\alpha(t,i,h) = \frac{\max(0, \alpha_{\text{raw}}(t,i,h))}{\sum_{j \leq t} \max(0, \alpha_{\text{raw}}(t,j,h))}$$

This produces a lower-triangular $T \times T$ matrix per head per layer.

### 9.2 Prior Art

Zimerman, Ali & Wolf (ICLR 2025, arXiv:2405.16504) derived the effective attention matrix for RWKV, but only for the **RWKV-4 scalar-decay case** ($w$ fixed, position-independent). The RWKV-6 data-dependent extension ($w_t$ varying per timestep, vector-valued per head channel) has no published derivation and is a genuine contribution of this work.

### 9.3 Numerical Stability

The cumulative decay products can underflow for large distances since $d_j \in (0,1)$. The implementation uses log-space prefix sums:

```
log_decay[t,h,d] = -exp(w[t,h,d])    (= ln(decay[t,h,d]))
prefix[0] = 0
prefix[k] = prefix[k-1] + log_decay[k-1]   (cumulative sum of log-decays)
```

For source $i < t$, the cumulative decay product from $i+1$ to $t-1$ is:
```
cum_decay[t,i,h,d] = exp(prefix[t] - prefix[i+1])
```

Per-channel contributions `r_t[d] * k_i[d] * cum_decay[d]` are computed in linear F32 space after exponentiating the log-space prefix difference. The sum across channels yields a scalar per $(t,i,h)$ triplet. This avoids overflow while handling the signed `r * k` products correctly.

### 9.4 Implementation Checklist

- [x] Implement `Rwkv6Attention::forward_with_effective_attention` (~170 lines, duplicates preamble from `forward_inner` with added effective attention computation)
- [x] Pre-compute log-decay prefix sums for numerically stable cumulative decay
- [x] Handle the current-position bonus term $u$ (`time_faaaa`) for diagonal entries
- [x] Apply ReLU + L1 normalisation to handle signed `r * k` products
- [x] Implement `Rwkv6Block::forward_with_attention` delegating to attention module
- [x] Implement `PlipRwkv6::forward_with_attention` inherent method collecting per-layer effective attention into `AttentionCache`
- [x] Wire into `PlipBackend::forward_with_attention` (replace error stub with delegation)
- [x] Integration test `test_rwkv6_effective_attention`: shape `[1, 32, seq, seq]`, lower-triangular, rows sum to ~1.0
- [x] Integration test `test_rwkv6_effective_attention_output_unchanged`: hidden output matches `forward_with_cache`
- [x] `cargo clippy -- -W clippy::pedantic` zero warnings
- [x] All 52 unit tests + 5 non-GPU integration tests pass (no regression)
- [x] GPU validation: `test_rwkv6_effective_attention` — shape [1,32,seq,seq] ✓, causal ✓, row sums = 1.0 exactly ✓ (layer 0: 337/384 valid rows, layer 11: 253/384, layer 23: 235/384); `test_rwkv6_effective_attention_output_unchanged` — hidden output bit-exact match (diff = 0.0)
- [ ] Run PLIP attention probing experiments on RWKV-6

### 9.5 Validation

- Effective attention matrix should be lower-triangular with rows summing to ~1.0.
- Hidden state output from `forward_with_attention` must be identical to `forward_with_cache`.
- Nearby tokens should receive more attention weight than distant tokens (decay property).
- Attention to function tokens from test markers should show the expected Python > Rust asymmetry.
- Compare effective attention patterns qualitatively with transformer attention patterns.

---

## 10. Release: v1.2.0

**Content:** RWKV-6 basic inference + state knockout + effective attention

**Checklist before release:**
- [x] RWKV-6 inference validated against Python reference
- [x] State knockout produces meaningful results
- [x] Effective attention is numerically stable and produces valid distributions
- [x] All 7 models pass existing tests
- [x] `cargo clippy` clean
- [x] Update `Cargo.toml` version to `1.2.0`
- [x] Update README model support table
- [x] Document RWKV-specific methods and limitations

**Architecture coverage after v1.2.0:**

| Model | Architecture | Size | Family | Type |
|-------|-------------|------|--------|------|
| Qwen2.5-Coder-7B-Instruct | Qwen2 | 7B | Qwen | Transformer |
| Qwen2.5-Coder-3B-Instruct | Qwen2 | 3B | Qwen | Transformer |
| StarCoder2-3B | StarCoder2 | 3B | BigCode | Transformer |
| CodeGemma-7B-it | Gemma | 7B | Google | Transformer |
| Code-LLaMA-7B | LLaMA | 7B | Meta | Transformer |
| Phi-3-mini-4k-instruct | Phi-3 | 3.8B | Microsoft | Transformer |
| RWKV-6-Finch-1B6 | RWKV-6 | 1.6B | RWKV | Gated-linear RNN |

**5 transformer families + 1 RNN family, 7 models.**

---

## 11. Phase 6: RWKV-6 State Delta Analysis (Approach 2) — Optional

**Prerequisite:** Phases 3 + 4 (RWKV-6 inference + knockout)

This phase is the highest-novelty, highest-risk component. It introduces RWKV-native metrics that have no transformer equivalent.

### 11.1 Three Metrics

**Write strength** at marker position $m$:
$$\|k_m^\top \cdot v_m\|_F$$

**Persistence** — survival of marker's write at distance $\delta$:
$$\text{persistence}(\delta) = \frac{\| \text{proj}(S_{m+\delta},\; \Delta S_m) \|_F}{\|\Delta S_m\|_F}$$

**Channel selectivity** — spectral analysis of the write component:
$$\text{spectrum}(\Delta S_m) = \text{SVD}(k_m^\top \cdot v_m)$$

### 11.2 Cross-Validation with Knockout

The persistence metric must be validated against knockout results (Phase 4):

- Tokens with high write strength and high persistence should be the same tokens whose knockout causes maximal KL divergence.
- If this correlation holds, persistence is validated as a lighter-weight proxy for knockout experiments.
- If it does not hold, the persistence metric needs revision or the state dynamics are more complex than the linear projection model assumes.

### 11.3 Implementation Checklist

- [ ] Capture $\Delta S_m = k_m^\top \cdot v_m$ at specified positions during forward pass
- [ ] Implement write strength metric ($\|\Delta S_m\|_F$)
- [ ] Implement forward persistence tracking: project $\Delta S_m$ into $S_{m+\delta}$ at increasing distances
- [ ] Implement SVD-based channel selectivity analysis
- [ ] Define new result types for state delta analysis (not `AttentionCache`)
- [ ] Add `PlipModel` methods for state delta analysis
- [ ] Cross-validate persistence against knockout KL divergence
- [ ] Run experiments comparing Python vs. Rust marker write strength and persistence

---

## 12. Dependency Graph

```
Phase 0 (Trait Refactor)
    |
    +---> Phase 1 (Code-LLaMA) --+
    |                             |
    +---> Phase 2 (Phi-3) -------+---> v1.1.0 Release
    |
    +---> Phase 3 (RWKV-6 Inference)
              |
              +---> Phase 4 (State Knockout) ---+
              |                                 |
              +---> Phase 5 (Effective Attn) ---+---> v1.2.0 Release
                                                |
                                                +---> Phase 6 (State Delta)
                                                           |
                                                           +---> v1.3.0 Release
```

**Notes:**
- Phase 0 must come first — it is a prerequisite for all subsequent phases.
- Phases 1, 2, and 3 are all independent of each other after Phase 0 and can be done in any order or in parallel.
- Phase 3 does NOT depend on v1.1.0 release — it only depends on Phase 0 (the trait refactor). However, v1.1.0 is released before starting RWKV work as a checkpoint.
- Phases 4 and 5 are independent of each other and can be done in parallel after Phase 3.
- Phase 6 depends on Phase 4 (for cross-validation of persistence against knockout KL divergence).

---

## 13. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Phi-3 uses SuRoPE/longrope** that differs from standard RoPE | Medium | Medium — requires implementing a RoPE variant | Inspect `config.json` for `rope_scaling` field before committing to implementation |
| **RWKV-6 tokenizer incompatible** with HF `tokenizers` crate | Low–Medium | High — requires tokenizer conversion or custom loading | Verify `tokenizer.json` presence on `RWKV/v6-Finch-1B6-HF` before starting Phase 3 |
| **RWKV-6 inference produces incorrect output** | Medium | High — no working Rust reference to debug against | Validate layer-by-layer against Python implementation; start with small inputs |
| **Effective attention is numerically unstable** for long sequences | Low | Medium — limits sequence lengths for attention analysis | Log-space computation with logsumexp normalisation; test on sequences up to 1024 tokens |
| **State delta persistence does not correlate with knockout KL** | Medium | Medium — invalidates Approach 2 as a proxy metric | This is a scientific finding, not a failure; document the result and investigate alternative formulations |
| **Code-LLaMA 7B exceeds VRAM** during attention capture | Low | Low — already run Qwen-7B and CodeGemma-7B successfully | Use BF16; if tight, use the Python variant (7B, same architecture but potentially different tokenizer) |
| **Trait refactor introduces subtle dispatch bugs** | Low | Medium — regression in existing model behaviour | Run full test suite on all 4 existing models before and after refactor |

---

## 14. Model Summary After Completion

| # | Model | Arch Family | Type | Size | Attention Probing | Knockout | Steering | Effective Attention | State Delta |
|---|-------|------------|------|------|-------------------|----------|----------|--------------------:|-------------|
| 1 | Qwen2.5-Coder-7B | Qwen2 | Transformer | 7B | Yes | Yes | Yes | N/A | N/A |
| 2 | Qwen2.5-Coder-3B | Qwen2 | Transformer | 3B | Yes | Yes | Yes | N/A | N/A |
| 3 | StarCoder2-3B | StarCoder2 | Transformer | 3B | Yes | Yes | Yes | N/A | N/A |
| 4 | CodeGemma-7B-it | Gemma | Transformer | 7B | Yes | Yes | Yes | N/A | N/A |
| 5 | Code-LLaMA-7B | LLaMA | Transformer | 7B | Yes | Yes | Yes | N/A | N/A |
| 6 | Phi-3-mini | Phi-3 | Transformer | 3.8B | Yes | Yes | Yes | N/A | N/A |
| 7 | RWKV-6-Finch-1B6 | RWKV-6 | RNN | 1.6B | Via eff. attn | State knockout | No | Yes | Approach 2 |

---

## 15. References

**Architecture support:**
- Candle framework: https://github.com/huggingface/candle
- Candle issue #3044 (RWKV bug): https://github.com/huggingface/candle/issues/3044

**RWKV interpretability (see also [rwkv-attention-approaches.md](../rwkv-attention-approaches.md)):**
- Zimerman, Ali & Wolf (2025). "Explaining Modern Gated-Linear RNNs via a Unified Implicit Attention Formulation." ICLR 2025. arXiv:2405.16504
- Endy et al. (2025). "Mamba Knockout for Unraveling Factual Information Flow." ACL 2025. arXiv:2505.24244
- Paulo et al. (2025). "Do Transformer Interpretability Methods Transfer to RNNs?" AAAI 2025.

**Model sources:**
- Code-LLaMA: https://huggingface.co/codellama/CodeLlama-7b-hf
- Phi-3-mini: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- RWKV-6-Finch: https://huggingface.co/RWKV/v6-Finch-1B6-HF
