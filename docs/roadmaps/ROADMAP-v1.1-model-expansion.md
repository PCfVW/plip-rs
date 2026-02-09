# PLIP-RS Roadmap: Model Expansion and Backend Refactoring

**Current version:** v1.1.0
**Target version:** v1.2.0 (RWKV-6)
**Status:** v1.1.0 released (Phases 0-2 complete); Phases 3-6 not started
**Last updated:** 2026-02-09

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
| Safetensors | Yes (HuggingFace format) |
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

**Estimated size:** 600–800 lines for `forward_rwkv6.rs`, compared to ~1,200–1,400 for transformer backends.

### 7.5 Tokenizer Verification

**Must verify before implementation:** Does `RWKV/v6-Finch-1B6-HF` ship a `tokenizer.json` compatible with the HuggingFace `tokenizers` crate?

- If yes: no changes needed; PLIP-RS's existing tokenizer loading works.
- If no (only RWKV-native World tokenizer): need to either convert the tokenizer to HF format or add a tokenizer loading path. This would be a meaningful additional task.

### 7.6 Implementation Checklist

- [ ] Verify tokenizer compatibility (see 7.5)
- [ ] Download and inspect `config.json` from `RWKV/v6-Finch-1B6-HF`
- [ ] Inspect weight names in safetensors file (expected: `rwkv.blocks.N.attention.{key,receptance,value,gate,output}.weight`, etc.)
- [ ] Create `src/forward_rwkv6.rs` with `Rwkv6Config` struct
- [ ] Implement token-shift (data-dependent mixing via `time_mix_w1`/`w2`)
- [ ] Implement data-dependent time decay via `time_decay_w1`/`w2`
- [ ] Implement WKV state update loop (core recurrence)
- [ ] Implement GroupNorm in attention output
- [ ] Implement channel-mix (MLP equivalent with squared ReLU)
- [ ] Implement `PlipBackend` for `PlipRwkv6` — note: `forward_with_steering` and `generate_with_prompt_steering` use default (unsupported) implementations
- [ ] Handle fixed-size state in `generate()` (no KV-cache growth)
- [ ] Add `Rwkv6` variant to `ModelArchitecture` + detection for `"rwkv"`, `"finch"` in model ID
- [ ] Add match arm in `from_pretrained_with_arch`
- [ ] Test basic inference

### 7.7 Validation

**Critical: validate against Python reference.**

1. Run `RWKV/v6-Finch-1B6-HF` in Python with `transformers` library on a test prompt.
2. Capture logits for the first forward pass (no generation).
3. Compare with PLIP-RS logits on the same tokenised input.
4. Tolerance: logits should match within BF16 precision (~1e-2 relative error).

If logits diverge significantly, debug by comparing intermediate tensors (layer outputs, state matrices) between Python and Rust.

---

## 8. Phase 4: RWKV-6 State Knockout (Approach 3)

**Prerequisite:** Phase 3 (working RWKV-6 inference)

State knockout is the lowest-risk RWKV interpretability approach, with direct methodological precedent in Mamba Knockout (Endy et al., ACL 2025).

### 8.1 Intervention Design

At the marker position $m$, replace the normal state update:

$$S_m = f(S_{m-1}, x_m) \quad\longrightarrow\quad S_m = S_{m-1}$$

All subsequent tokens process a state that never saw the marker. The metric is KL divergence between baseline and knocked-out output distributions — identical to the transformer knockout metric in Table 6 of the PLIP paper.

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

Note: the return type is `Result<Tensor>` (logits only), not `Result<(Tensor, AttentionCache)>`. State knockout only needs output logits for KL divergence comparison — it does not require attention patterns. Effective attention computation (Phase 5) is a separate capability added later. Only `PlipRwkv6` overrides this. Transformer backends use the default (unsupported).

**PlipModel integration.** Add a `forward_with_state_knockout` method that handles tokenisation and calls the backend. This method runs both baseline and knocked-out forward passes and computes KL divergence, following the existing `forward_with_intervention` pattern but returning logits-only results.

### 8.3 Implementation Checklist

- [ ] Define `StateKnockoutSpec` in `intervention.rs`
- [ ] Add `forward_with_state_knockout` to `PlipBackend` trait (default: unsupported)
- [ ] Implement `forward_with_state_knockout` in `PlipRwkv6` — skip state update at specified positions, return logits
- [ ] Add `PlipModel::forward_with_state_knockout` wrapper method
- [ ] Test: knockout at a known marker position produces non-zero KL divergence
- [ ] Test: knockout at a random non-marker position produces different (typically smaller) KL divergence
- [ ] Run PLIP knockout experiments (Python vs. Rust test markers) on RWKV-6

### 8.4 Validation

- KL divergence must be > 0 for knocked-out marker positions (otherwise the marker has no effect, which is implausible).
- Compare Python vs. Rust knockout KL divergences — expect asymmetry consistent with co-location hypothesis.
- Cross-reference with transformer knockout results in Table 6.

---

## 9. Phase 5: RWKV-6 Effective Attention (Approach 1)

**Prerequisite:** Phase 3 (working RWKV-6 inference)

Compute the effective attention matrix for RWKV-6, producing `[batch, heads, seq, seq]` tensors compatible with `AttentionCache`.

### 9.1 Mathematical Foundation

For RWKV-6 with data-dependent decay $w_t$, the effective weight from position $t$ to position $i < t$ is:

$$\alpha_{t,i} \propto \exp\Bigl(\sum_{j=i+1}^{t} (-w_j) + k_i\Bigr)$$

Current position bonus: $\alpha_{t,t} \propto \exp(u + k_t)$ where $u$ is the learned bonus.

Normalisation: $\alpha_{t,i} = \alpha_{t,i}^{\text{raw}} / \sum_{j \leq t} \alpha_{t,j}^{\text{raw}}$

This produces a lower-triangular $T \times T$ matrix per head per layer.

### 9.2 Prior Art

Zimerman, Ali & Wolf (ICLR 2025, arXiv:2405.16504) derived the effective attention matrix for RWKV, but only for the **RWKV-4 scalar-decay case** ($w$ fixed, position-independent). The RWKV-6 data-dependent extension ($w_t$ varying per timestep) has no published derivation and is a genuine contribution of this work.

### 9.3 Numerical Stability

The log-space computation avoids overflow:

```
log_alpha_raw[t, i] = cumsum(-w[i+1..t]) + k[i]    for i < t
log_alpha_raw[t, t] = u + k[t]                      for current position
log_alpha[t, :] = log_alpha_raw[t, :] - logsumexp(log_alpha_raw[t, :])
alpha[t, :] = exp(log_alpha[t, :])
```

The `cumsum` over log-decay values replaces the product of per-step decays, keeping everything in log-space until the final exponentiation.

### 9.4 Implementation Checklist

- [ ] Implement effective attention computation in `forward_rwkv6.rs` (~30–50 lines per layer)
- [ ] Use log-space cumulative sums for numerical stability
- [ ] Handle the current-position bonus term $u$
- [ ] Return effective attention as `AttentionCache` from `forward_with_attention`
- [ ] Verify: effective attention rows sum to 1.0 (within numerical tolerance)
- [ ] Verify: effective attention is lower-triangular (causal)
- [ ] Verify: nearby positions have higher effective attention than distant positions (sanity check for decay)
- [ ] Run PLIP attention probing experiments on RWKV-6

### 9.5 Validation

- Effective attention matrix should be lower-triangular with rows summing to ~1.0.
- Nearby tokens should receive more attention weight than distant tokens (decay property).
- Attention to function tokens from test markers should show the expected Python > Rust asymmetry.
- Compare effective attention patterns qualitatively with transformer attention patterns.

---

## 10. Release: v1.2.0

**Content:** RWKV-6 basic inference + state knockout + effective attention

**Checklist before release:**
- [ ] RWKV-6 inference validated against Python reference
- [ ] State knockout produces meaningful results
- [ ] Effective attention is numerically stable and produces valid distributions
- [ ] All 7 models pass existing tests
- [ ] `cargo clippy` clean
- [ ] Update `Cargo.toml` version to `1.2.0`
- [ ] Update README model support table
- [ ] Document RWKV-specific methods and limitations

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
