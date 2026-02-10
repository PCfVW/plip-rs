//! RWKV-6 / Finch forward pass implementation
//!
//! Implements the RWKV-6 gated-linear RNN architecture from scratch using candle
//! tensor operations. Reference: `modeling_rwkv6.py` from `RWKV/v6-Finch-1B6-HF`.
//!
//! Key differences from transformer backends:
//! - No attention mechanism — uses recurrent state update (WKV)
//! - No positional encoding (position is implicit in recurrence)
//! - Fixed-size state per layer (no growing KV-cache)
//! - LayerNorm + GroupNorm instead of RMSNorm
//! - Squared ReLU activation in channel-mix (MLP)

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tracing::info;

use std::collections::HashSet;

use crate::attention::AttentionCache;
use crate::cache::ActivationCache;
use crate::intervention::{KnockoutSpec, StateKnockoutSpec, StateSteeringSpec};
use crate::kv_cache::KVCache;
use crate::model::PlipBackend;

// ============================================================================
// Config
// ============================================================================

/// Hardcoded extra dimensions for data-dependent mixing projections.
/// These match the Python reference and are not in config.json.
const TIME_MIX_EXTRA_DIM: usize = 32;
const TIME_DECAY_EXTRA_DIM: usize = 64;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Rwkv6Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    /// Per-head dimension. NOTE: In the HF config, `num_attention_heads` confusingly
    /// equals `head_size` (not the head count). We use `head_size` directly.
    pub head_size: usize,
    pub vocab_size: usize,
    /// Usually equal to hidden_size
    #[serde(default = "default_attention_hidden_size")]
    pub attention_hidden_size: usize,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,
    #[serde(default = "default_head_size_divisor")]
    pub head_size_divisor: usize,
    #[serde(default = "default_rescale_every")]
    pub rescale_every: usize,
    /// null in config → computed as int(hidden_size * 3.5 / 32) * 32
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_attention_hidden_size() -> usize {
    2048
}
fn default_layer_norm_epsilon() -> f64 {
    1e-5
}
fn default_head_size_divisor() -> usize {
    8
}
fn default_rescale_every() -> usize {
    6
}

impl Rwkv6Config {
    pub fn num_heads(&self) -> usize {
        self.attention_hidden_size / self.head_size
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
            .unwrap_or_else(|| (self.hidden_size * 7 / 2) / 32 * 32)
    }
}

// ============================================================================
// LayerNorm (candle_nn has this but we need explicit weight/bias access)
// ============================================================================

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        let bias = vb.get(size, "bias")?;
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        x_normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
            .map_err(Into::into)
    }
}

// ============================================================================
// GroupNorm (manual implementation — not available in candle-nn)
// ============================================================================

/// Apply group normalization.
///
/// Input shape: [batch * seq, channels]
/// Groups: num_heads (each group = head_size channels)
fn group_norm(
    x: &Tensor,
    num_groups: usize,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let (n, c) = x.dims2()?;
    let channels_per_group = c / num_groups;

    // Reshape to [n, num_groups, channels_per_group]
    let x = x.reshape((n, num_groups, channels_per_group))?;
    let mean = x.mean_keepdim(2)?;
    let x_centered = x.broadcast_sub(&mean)?;
    let var = x_centered.sqr()?.mean_keepdim(2)?;
    let x_normed = x_centered.broadcast_div(&(var + eps)?.sqrt()?)?;

    // Reshape back to [n, channels]
    let x_normed = x_normed.reshape((n, c))?;

    // Affine transform
    x_normed
        .broadcast_mul(&weight.unsqueeze(0)?)?
        .broadcast_add(&bias.unsqueeze(0)?)
        .map_err(Into::into)
}

// ============================================================================
// RWKV-6 Attention (Time-Mix)
// ============================================================================

struct Rwkv6Attention {
    // Data-dependent mixing parameters
    time_maa_x: Tensor, // [1, 1, hidden_size]
    time_maa_w: Tensor,
    time_maa_k: Tensor,
    time_maa_v: Tensor,
    time_maa_r: Tensor,
    time_maa_g: Tensor,

    // Low-rank mixing projections
    time_maa_w1: Tensor, // [hidden_size, TIME_MIX_EXTRA_DIM * 5]
    time_maa_w2: Tensor, // [5, TIME_MIX_EXTRA_DIM, hidden_size]

    // Time decay
    time_decay: Tensor,    // [1, 1, attention_hidden_size]
    time_decay_w1: Tensor, // [hidden_size, TIME_DECAY_EXTRA_DIM]
    time_decay_w2: Tensor, // [TIME_DECAY_EXTRA_DIM, attention_hidden_size]

    // Per-head "time first" bonus for current position
    time_faaaa: Tensor, // [num_heads, head_size]

    // Linear projections (no bias)
    receptance: Linear,
    key: Linear,
    value: Linear,
    gate: Linear,
    output: Linear,

    // GroupNorm parameters
    ln_x_weight: Tensor,
    ln_x_bias: Tensor,

    num_heads: usize,
    head_size: usize,
    group_norm_eps: f64,
}

impl Rwkv6Attention {
    fn load(vb: VarBuilder, config: &Rwkv6Config) -> Result<Self> {
        let h = config.hidden_size;
        let ah = config.attention_hidden_size;
        let nh = config.num_heads();
        let hs = config.head_size;

        let time_maa_x = vb.get((1, 1, h), "time_maa_x")?;
        let time_maa_w = vb.get((1, 1, h), "time_maa_w")?;
        let time_maa_k = vb.get((1, 1, h), "time_maa_k")?;
        let time_maa_v = vb.get((1, 1, h), "time_maa_v")?;
        let time_maa_r = vb.get((1, 1, h), "time_maa_r")?;
        let time_maa_g = vb.get((1, 1, h), "time_maa_g")?;

        let time_maa_w1 = vb.get((h, TIME_MIX_EXTRA_DIM * 5), "time_maa_w1")?;
        let time_maa_w2 = vb.get((5, TIME_MIX_EXTRA_DIM, h), "time_maa_w2")?;

        let time_decay = vb.get((1, 1, ah), "time_decay")?;
        let time_decay_w1 = vb.get((h, TIME_DECAY_EXTRA_DIM), "time_decay_w1")?;
        let time_decay_w2 = vb.get((TIME_DECAY_EXTRA_DIM, ah), "time_decay_w2")?;

        let time_faaaa = vb.get((nh, hs), "time_faaaa")?;

        let receptance = linear_no_bias(h, ah, vb.pp("receptance"))?;
        let key = linear_no_bias(h, ah, vb.pp("key"))?;
        let value = linear_no_bias(h, ah, vb.pp("value"))?;
        let gate = linear_no_bias(h, ah, vb.pp("gate"))?;
        let output = linear_no_bias(ah, h, vb.pp("output"))?;

        let ln_x_weight = vb.get(ah, "ln_x.weight")?;
        let ln_x_bias = vb.get(ah, "ln_x.bias")?;

        // GroupNorm eps = layer_norm_epsilon * head_size_divisor^2
        let group_norm_eps = config.layer_norm_epsilon * (config.head_size_divisor as f64).powi(2);

        Ok(Self {
            time_maa_x,
            time_maa_w,
            time_maa_k,
            time_maa_v,
            time_maa_r,
            time_maa_g,
            time_maa_w1,
            time_maa_w2,
            time_decay,
            time_decay_w1,
            time_decay_w2,
            time_faaaa,
            receptance,
            key,
            value,
            gate,
            output,
            ln_x_weight,
            ln_x_bias,
            num_heads: nh,
            head_size: hs,
            group_norm_eps,
        })
    }

    /// Forward pass for the time-mix (attention) block.
    ///
    /// # Arguments
    /// * `hidden` - Input tensor [batch, seq, hidden_size]
    /// * `attn_x_state` - Previous hidden for token-shift [batch, hidden_size] or None
    /// * `attn_kv_state` - Accumulated WKV state [batch, num_heads, head_size, head_size] or None
    ///
    /// # Returns
    /// (output, new_attn_x_state, new_attn_kv_state)
    fn forward(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.forward_inner(hidden, attn_x_state, attn_kv_state, None, None)
    }

    /// Forward with state knockout: positions in `knockout_positions` skip the kv state write.
    fn forward_with_knockout(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
        knockout_positions: &HashSet<usize>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.forward_inner(
            hidden,
            attn_x_state,
            attn_kv_state,
            Some(knockout_positions),
            None,
        )
    }

    /// Forward with state steering: scale the kv write at specified positions.
    fn forward_with_steering(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
        steering_positions: &HashSet<usize>,
        scale: f32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.forward_inner(
            hidden,
            attn_x_state,
            attn_kv_state,
            Some(steering_positions),
            Some(scale),
        )
    }

    /// Shared forward implementation.
    ///
    /// When `intervention_positions` is `Some`, the WKV state update at those
    /// positions is modified:
    /// - `kv_scale = None` (or 0.0): knockout — only decay, no kv write
    /// - `kv_scale = Some(s)`: steering — scale the kv write by `s`
    #[allow(clippy::too_many_lines)]
    fn forward_inner(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
        intervention_positions: Option<&HashSet<usize>>,
        kv_scale: Option<f32>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, t, c) = hidden.dims3()?;
        let h = self.num_heads;
        let s = self.head_size;

        // --- Token shift ---
        let shifted = if t == 1 {
            // Single token: use state
            match attn_x_state {
                Some(state) => state.unsqueeze(1)?,
                None => Tensor::zeros((b, 1, c), hidden.dtype(), hidden.device())?,
            }
        } else {
            // Multi-token: shift by padding top, cropping bottom
            let zeros = Tensor::zeros((b, 1, c), hidden.dtype(), hidden.device())?;
            let prev_tokens = hidden.i((.., ..t - 1, ..))?;
            let mut shifted = Tensor::cat(&[&zeros, &prev_tokens], 1)?;
            // If we have state, replace the first token's shift
            if let Some(state) = attn_x_state {
                let state_expanded = state.unsqueeze(1)?;
                // Replace shifted[:, 0, :] with state
                let rest = shifted.i((.., 1.., ..))?;
                shifted = Tensor::cat(&[&state_expanded, &rest], 1)?;
            }
            shifted
        };

        // Save last token as new attn_x_state
        let new_attn_x_state = hidden.i((.., t - 1, ..))?;

        // --- Data-dependent mixing ---
        let xx = shifted.broadcast_sub(hidden)?; // [b, t, c]

        // Step 1: Base mixing coefficient
        let xxx = hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_x)?)?; // [b, t, c]

        // Step 2: Project through w1 → tanh → reshape for 5 components
        let xxx_flat = xxx.reshape((b * t, c))?;
        let projected = xxx_flat.matmul(&self.time_maa_w1)?; // [b*t, 160]
        let projected = projected.tanh()?;
        let projected = projected.reshape((b * t, 5, TIME_MIX_EXTRA_DIM))?;
        let projected = projected.transpose(0, 1)?.contiguous()?; // [5, b*t, 32]

        // Step 3: Back-project through w2
        let mixed = projected.matmul(&self.time_maa_w2)?; // [5, b*t, c]
        let mixed = mixed.reshape((5, b, t, c))?;

        let mw = mixed.i(0)?; // [b, t, c]
        let mk = mixed.i(1)?;
        let mv = mixed.i(2)?;
        let mr = mixed.i(3)?;
        let mg = mixed.i(4)?;

        // Step 4: Apply mixing to produce inputs for each projection
        let time_decay_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_w.broadcast_add(&mw)?)?)?;
        let key_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_k.broadcast_add(&mk)?)?)?;
        let value_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_v.broadcast_add(&mv)?)?)?;
        let receptance_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_r.broadcast_add(&mr)?)?)?;
        let gate_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_g.broadcast_add(&mg)?)?)?;

        // --- Project to R, K, V, gate ---
        let r = self.receptance.forward(&receptance_input)?; // [b, t, ah]
        let k = self.key.forward(&key_input)?;
        let v = self.value.forward(&value_input)?;
        let gate_val = candle_nn::ops::silu(&self.gate.forward(&gate_input)?)?;

        // --- Data-dependent time decay ---
        let td_flat = time_decay_input.reshape((b * t, c))?;
        let td_proj = td_flat.matmul(&self.time_decay_w1)?.tanh()?; // [b*t, 64]
        let td_proj = td_proj.matmul(&self.time_decay_w2)?; // [b*t, ah]
        let td_proj = td_proj.reshape((b, t, self.num_heads * self.head_size))?;
        let w = self.time_decay.broadcast_add(&td_proj)?; // [b, t, ah]

        // decay = exp(-exp(w))
        let decay = w.to_dtype(DType::F32)?.exp()?.neg()?.exp()?; // [b, t, ah] in f32

        // --- Reshape for per-head computation ---
        // r, k, v: [b, t, ah] → [b, t, h, s]
        let r = r.to_dtype(DType::F32)?.reshape((b, t, h, s))?;
        let k = k.to_dtype(DType::F32)?.reshape((b, t, h, s))?;
        let v = v.to_dtype(DType::F32)?.reshape((b, t, h, s))?;
        let decay = decay.reshape((b, t, h, s))?;

        // time_faaaa: [h, s] → used as current-position bonus
        let time_first = self.time_faaaa.to_dtype(DType::F32)?; // [h, s]
        let time_first = time_first.reshape((1, 1, h, s))?;

        // --- WKV Recurrence Loop ---
        let mut state = match attn_kv_state {
            Some(s) => s.to_dtype(DType::F32)?.clone(),
            None => Tensor::zeros((b, h, s, s), DType::F32, hidden.device())?,
        };

        let mut outputs: Vec<Tensor> = Vec::with_capacity(t);

        for ti in 0..t {
            // Current timestep values: [b, h, s]
            let r_t = r.i((.., ti, .., ..))?; // [b, h, s]
            let k_t = k.i((.., ti, .., ..))?;
            let v_t = v.i((.., ti, .., ..))?;
            let decay_t = decay.i((.., ti, .., ..))?; // [b, h, s]
            let time_first_t = time_first.i((.., 0, .., ..))?; // [1, h, s]

            // kv = k_t^T @ v_t: outer product [b, h, s, 1] x [b, h, 1, s] → [b, h, s, s]
            let k_col = k_t.unsqueeze(D::Minus1)?; // [b, h, s, 1]
            let v_row = v_t.unsqueeze(2)?; // [b, h, 1, s]
            let kv = k_col.matmul(&v_row)?; // [b, h, s, s]

            // out_t = r_t @ (time_first * kv + state)
            // time_first * kv: element-wise on the k-dimension
            let time_first_expanded = time_first_t.unsqueeze(D::Minus1)?; // [1, h, s, 1]
            let weighted_kv = kv.broadcast_mul(&time_first_expanded)?; // [b, h, s, s]
            let combined = (&weighted_kv + &state)?; // [b, h, s, s]

            let r_row = r_t.unsqueeze(2)?; // [b, h, 1, s]
            let out_t = r_row.matmul(&combined)?; // [b, h, 1, s]
            let out_t = out_t.squeeze(2)?; // [b, h, s]

            outputs.push(out_t);

            // State update: state = kv + decay * state
            // Intervention: scale or suppress the kv write at specified positions
            let decay_expanded = decay_t.unsqueeze(D::Minus1)?; // [b, h, s, 1]
            let should_intervene = intervention_positions.is_some_and(|kp| kp.contains(&ti));
            if should_intervene {
                let scale = kv_scale.unwrap_or(0.0); // Default: knockout
                let decayed = state.broadcast_mul(&decay_expanded)?;
                if scale == 0.0 {
                    state = decayed;
                } else {
                    state = ((kv * f64::from(scale))? + decayed)?;
                }
            } else {
                state = (kv + state.broadcast_mul(&decay_expanded)?)?;
            }
        }

        // Stack outputs: [b, t, h, s]
        let out = Tensor::stack(&outputs, 1)?; // [b, t, h, s]
        let new_attn_kv_state = state; // [b, h, s, s]

        // --- GroupNorm per head ---
        let out = out.reshape((b * t, h * s))?;
        let out = group_norm(
            &out,
            self.num_heads,
            &self.ln_x_weight.to_dtype(DType::F32)?,
            &self.ln_x_bias.to_dtype(DType::F32)?,
            self.group_norm_eps,
        )?;
        let out = out.reshape((b, t, h * s))?.to_dtype(hidden.dtype())?;

        // --- Apply gate and output projection ---
        let out = (out * gate_val)?;
        let out = self.output.forward(&out)?;

        Ok((out, new_attn_x_state, new_attn_kv_state))
    }

    /// Forward pass that also computes the effective attention matrix.
    ///
    /// The effective attention is derived from the WKV recurrence. For query
    /// position t and source position i:
    ///
    ///   α_raw[t,i,h] = Σ_d r_t[h,d] · k_i[h,d] · Π_{j=i+1}^{t-1} decay_j[h,d]
    ///   α_raw[t,t,h] = Σ_d r_t[h,d] · k_t[h,d] · time_first[h,d]  (diagonal)
    ///
    /// Normalised via ReLU + L1: α[t,i] = max(0, α_raw[t,i]) / Σ_j max(0, α_raw[t,j])
    ///
    /// Returns (output, new_attn_x_state, new_attn_kv_state, effective_attention)
    /// where effective_attention is `[batch, heads, seq, seq]`.
    #[allow(clippy::too_many_lines)]
    fn forward_with_effective_attention(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (b, t, c) = hidden.dims3()?;
        let h = self.num_heads;
        let s = self.head_size;
        let device = hidden.device();

        // === Preamble: Token shift + mixing + projection (same as forward_inner) ===

        let shifted = if t == 1 {
            match attn_x_state {
                Some(state) => state.unsqueeze(1)?,
                None => Tensor::zeros((b, 1, c), hidden.dtype(), hidden.device())?,
            }
        } else {
            let zeros = Tensor::zeros((b, 1, c), hidden.dtype(), hidden.device())?;
            let prev_tokens = hidden.i((.., ..t - 1, ..))?;
            let mut shifted = Tensor::cat(&[&zeros, &prev_tokens], 1)?;
            if let Some(state) = attn_x_state {
                let state_expanded = state.unsqueeze(1)?;
                let rest = shifted.i((.., 1.., ..))?;
                shifted = Tensor::cat(&[&state_expanded, &rest], 1)?;
            }
            shifted
        };

        let new_attn_x_state = hidden.i((.., t - 1, ..))?;

        let xx = shifted.broadcast_sub(hidden)?;
        let xxx = hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_x)?)?;

        let xxx_flat = xxx.reshape((b * t, c))?;
        let projected = xxx_flat.matmul(&self.time_maa_w1)?;
        let projected = projected.tanh()?;
        let projected = projected.reshape((b * t, 5, TIME_MIX_EXTRA_DIM))?;
        let projected = projected.transpose(0, 1)?.contiguous()?;

        let mixed = projected.matmul(&self.time_maa_w2)?;
        let mixed = mixed.reshape((5, b, t, c))?;

        let mw = mixed.i(0)?;
        let mk = mixed.i(1)?;
        let mv = mixed.i(2)?;
        let mr = mixed.i(3)?;
        let mg = mixed.i(4)?;

        let time_decay_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_w.broadcast_add(&mw)?)?)?;
        let key_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_k.broadcast_add(&mk)?)?)?;
        let value_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_v.broadcast_add(&mv)?)?)?;
        let receptance_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_r.broadcast_add(&mr)?)?)?;
        let gate_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_g.broadcast_add(&mg)?)?)?;

        let r = self.receptance.forward(&receptance_input)?;
        let k = self.key.forward(&key_input)?;
        let v = self.value.forward(&value_input)?;
        let gate_val = candle_nn::ops::silu(&self.gate.forward(&gate_input)?)?;

        let td_flat = time_decay_input.reshape((b * t, c))?;
        let td_proj = td_flat.matmul(&self.time_decay_w1)?.tanh()?;
        let td_proj = td_proj.matmul(&self.time_decay_w2)?;
        let td_proj = td_proj.reshape((b, t, self.num_heads * self.head_size))?;
        let w = self.time_decay.broadcast_add(&td_proj)?;

        // === Decay and log-decay ===
        let neg_exp_w = w.to_dtype(DType::F32)?.exp()?.neg()?; // log(decay) = -exp(w)
        let log_decay = neg_exp_w.reshape((b, t, h, s))?;
        let decay = log_decay.exp()?; // decay = exp(log(decay))

        // === Reshape for per-head computation ===
        let r = r.to_dtype(DType::F32)?.reshape((b, t, h, s))?;
        let k = k.to_dtype(DType::F32)?.reshape((b, t, h, s))?;
        let v = v.to_dtype(DType::F32)?.reshape((b, t, h, s))?;

        let time_first = self.time_faaaa.to_dtype(DType::F32)?; // [h, s]
        let time_first_4d = time_first.reshape((1, 1, h, s))?;

        // === WKV Recurrence Loop (same as forward_inner, no knockout) ===
        let mut state = match attn_kv_state {
            Some(s) => s.to_dtype(DType::F32)?.clone(),
            None => Tensor::zeros((b, h, s, s), DType::F32, hidden.device())?,
        };

        let mut outputs: Vec<Tensor> = Vec::with_capacity(t);

        for ti in 0..t {
            let r_t = r.i((.., ti, .., ..))?;
            let k_t = k.i((.., ti, .., ..))?;
            let v_t = v.i((.., ti, .., ..))?;
            let decay_t = decay.i((.., ti, .., ..))?;
            let time_first_t = time_first_4d.i((.., 0, .., ..))?;

            let k_col = k_t.unsqueeze(D::Minus1)?;
            let v_row = v_t.unsqueeze(2)?;
            let kv = k_col.matmul(&v_row)?;

            let time_first_expanded = time_first_t.unsqueeze(D::Minus1)?;
            let weighted_kv = kv.broadcast_mul(&time_first_expanded)?;
            let combined = (&weighted_kv + &state)?;

            let r_row = r_t.unsqueeze(2)?;
            let out_t = r_row.matmul(&combined)?;
            let out_t = out_t.squeeze(2)?;

            outputs.push(out_t);

            let decay_expanded = decay_t.unsqueeze(D::Minus1)?;
            state = (kv + state.broadcast_mul(&decay_expanded)?)?;
        }

        let out = Tensor::stack(&outputs, 1)?;
        let new_attn_kv_state = state;

        // === Compute Effective Attention ===

        // Pre-compute log_decay prefix sums: prefix[k] = Σ_{j=0}^{k-1} log_decay[j]
        let mut cum_ld = Vec::with_capacity(t + 1);
        cum_ld.push(Tensor::zeros((b, h, s), DType::F32, device)?); // prefix[0] = 0
        for ti in 0..t {
            let ld_ti = log_decay.i((.., ti, .., ..))?; // [b, h, s]
            cum_ld.push((&cum_ld[ti] + &ld_ti)?);
        }
        let prefix = Tensor::stack(&cum_ld, 1)?; // [b, t+1, h, s]

        let mut eff_attn_rows: Vec<Tensor> = Vec::with_capacity(t);

        for ti in 0..t {
            let r_ti = r.i((.., ti, .., ..))?; // [b, h, s]
            let k_ti = k.i((.., ti, .., ..))?; // [b, h, s]

            // Diagonal: α_raw[ti,ti] = Σ_d r[d] * k[d] * time_first[d]
            let diag = (&r_ti * &k_ti)?.broadcast_mul(&time_first)?; // [b, h, s]
            let diag_alpha = diag.sum(D::Minus1)?.unsqueeze(D::Minus1)?; // [b, h, 1]

            let alpha_raw = if ti > 0 {
                // Past positions: α_raw[ti,si] for si = 0..ti
                let k_past = k.i((.., ..ti, .., ..))?; // [b, ti, h, s]
                let pref_ti = prefix.i((.., ti, .., ..))?.unsqueeze(1)?; // [b, 1, h, s]
                let pref_src = prefix.i((.., 1..=ti, .., ..))?; // [b, ti, h, s]
                let log_cd = pref_ti.broadcast_sub(&pref_src)?; // [b, ti, h, s]
                let cd = log_cd.exp()?; // [b, ti, h, s]

                let r_exp = r_ti.unsqueeze(1)?; // [b, 1, h, s]
                let per_ch = r_exp.broadcast_mul(&k_past)?.broadcast_mul(&cd)?; // [b, ti, h, s]
                let a_past = per_ch.sum(D::Minus1)?.transpose(1, 2)?; // [b, h, ti]

                Tensor::cat(&[&a_past, &diag_alpha], D::Minus1)? // [b, h, ti+1]
            } else {
                diag_alpha
            };

            // Pad with zeros for causal mask (positions after ti)
            let alpha_raw = if ti + 1 < t {
                let pad = Tensor::zeros((b, h, t - ti - 1), DType::F32, device)?;
                Tensor::cat(&[&alpha_raw, &pad], D::Minus1)? // [b, h, t]
            } else {
                alpha_raw
            };

            // ReLU + L1 normalisation
            let alpha_relu = alpha_raw.relu()?;
            let row_sum = (alpha_relu.sum_keepdim(D::Minus1)? + 1e-10)?; // [b, h, 1]
            let alpha_normed = alpha_relu.broadcast_div(&row_sum)?; // [b, h, t]

            eff_attn_rows.push(alpha_normed);
        }

        let eff_attn = Tensor::stack(&eff_attn_rows, 2)?; // [b, h, t_query, t_source]

        // === Postprocessing (same as forward_inner) ===
        let out = out.reshape((b * t, h * s))?;
        let out = group_norm(
            &out,
            self.num_heads,
            &self.ln_x_weight.to_dtype(DType::F32)?,
            &self.ln_x_bias.to_dtype(DType::F32)?,
            self.group_norm_eps,
        )?;
        let out = out.reshape((b, t, h * s))?.to_dtype(hidden.dtype())?;

        let out = (out * gate_val)?;
        let out = self.output.forward(&out)?;

        Ok((out, new_attn_x_state, new_attn_kv_state, eff_attn))
    }
}

// ============================================================================
// RWKV-6 Feed-Forward (Channel-Mix)
// ============================================================================

struct Rwkv6FeedForward {
    time_maa_k: Tensor, // [1, 1, hidden_size]
    time_maa_r: Tensor,

    key: Linear,        // hidden → intermediate (no bias)
    receptance: Linear, // hidden → hidden (no bias)
    value: Linear,      // intermediate → hidden (no bias)
}

impl Rwkv6FeedForward {
    fn load(vb: VarBuilder, config: &Rwkv6Config) -> Result<Self> {
        let h = config.hidden_size;
        let intermediate = config.intermediate_size();

        let time_maa_k = vb.get((1, 1, h), "time_maa_k")?;
        let time_maa_r = vb.get((1, 1, h), "time_maa_r")?;

        let key = linear_no_bias(h, intermediate, vb.pp("key"))?;
        let receptance = linear_no_bias(h, h, vb.pp("receptance"))?;
        let value = linear_no_bias(intermediate, h, vb.pp("value"))?;

        Ok(Self {
            time_maa_k,
            time_maa_r,
            key,
            receptance,
            value,
        })
    }

    /// Forward pass for channel-mix (MLP).
    ///
    /// # Arguments
    /// * `hidden` - Input [batch, seq, hidden_size]
    /// * `ffn_x_state` - Previous hidden for token-shift [batch, hidden_size] or None
    ///
    /// # Returns
    /// (output, new_ffn_x_state)
    fn forward(&self, hidden: &Tensor, ffn_x_state: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (b, t, c) = hidden.dims3()?;

        // --- Token shift ---
        let shifted = if t == 1 {
            match ffn_x_state {
                Some(state) => state.unsqueeze(1)?,
                None => Tensor::zeros((b, 1, c), hidden.dtype(), hidden.device())?,
            }
        } else {
            let zeros = Tensor::zeros((b, 1, c), hidden.dtype(), hidden.device())?;
            let prev_tokens = hidden.i((.., ..t - 1, ..))?;
            let mut shifted = Tensor::cat(&[&zeros, &prev_tokens], 1)?;
            if let Some(state) = ffn_x_state {
                let state_expanded = state.unsqueeze(1)?;
                let rest = shifted.i((.., 1.., ..))?;
                shifted = Tensor::cat(&[&state_expanded, &rest], 1)?;
            }
            shifted
        };

        let new_ffn_x_state = hidden.i((.., t - 1, ..))?;

        // --- Mixing ---
        let xx = shifted.broadcast_sub(hidden)?;
        let key_input = hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_k)?)?;
        let rec_input = hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_r)?)?;

        // key = relu(key_proj(key_input))^2 (squared ReLU)
        let k = self.key.forward(&key_input)?.relu()?.sqr()?;

        // value = value_proj(k)  — note: "key" output feeds into "value" projection
        let v = self.value.forward(&k)?;

        // receptance = sigmoid(receptance_proj(rec_input))
        let r = candle_nn::ops::sigmoid(&self.receptance.forward(&rec_input)?)?;

        // Output: receptance * value
        let out = (r * v)?;

        Ok((out, new_ffn_x_state))
    }
}

// ============================================================================
// RWKV-6 Block
// ============================================================================

struct Rwkv6Block {
    pre_ln: Option<LayerNorm>, // Only for block 0
    ln1: LayerNorm,
    ln2: LayerNorm,
    attention: Rwkv6Attention,
    feed_forward: Rwkv6FeedForward,
}

impl Rwkv6Block {
    fn load(vb: VarBuilder, config: &Rwkv6Config, layer_id: usize) -> Result<Self> {
        let eps = config.layer_norm_epsilon;
        let h = config.hidden_size;

        let pre_ln = if layer_id == 0 {
            Some(LayerNorm::load(h, eps, vb.pp("pre_ln"))?)
        } else {
            None
        };

        let ln1 = LayerNorm::load(h, eps, vb.pp("ln1"))?;
        let ln2 = LayerNorm::load(h, eps, vb.pp("ln2"))?;
        let attention = Rwkv6Attention::load(vb.pp("attention"), config)?;
        let feed_forward = Rwkv6FeedForward::load(vb.pp("feed_forward"), config)?;

        Ok(Self {
            pre_ln,
            ln1,
            ln2,
            attention,
            feed_forward,
        })
    }

    /// Forward pass for a single RWKV-6 block.
    ///
    /// # Arguments
    /// * `hidden` - Input [batch, seq, hidden_size]
    /// * `attn_x_state` - Previous hidden for attention shift
    /// * `attn_kv_state` - WKV state [batch, num_heads, head_size, head_size]
    /// * `ffn_x_state` - Previous hidden for FFN shift
    ///
    /// # Returns
    /// (output, new_attn_x, new_attn_kv, new_ffn_x)
    fn forward(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
        ffn_x_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        // Pre-LayerNorm (only first block)
        let hidden = if let Some(pre_ln) = &self.pre_ln {
            pre_ln.forward(hidden)?
        } else {
            hidden.clone()
        };

        // Attention with residual
        let (attn_out, new_attn_x, new_attn_kv) =
            self.attention
                .forward(&self.ln1.forward(&hidden)?, attn_x_state, attn_kv_state)?;
        let hidden = (&hidden + attn_out)?;

        // Feed-forward with residual
        let (ffn_out, new_ffn_x) = self
            .feed_forward
            .forward(&self.ln2.forward(&hidden)?, ffn_x_state)?;
        let hidden = (&hidden + ffn_out)?;

        Ok((hidden, new_attn_x, new_attn_kv, new_ffn_x))
    }

    /// Forward pass with state knockout at specified positions.
    ///
    /// The FFN is NOT knocked out — it has no accumulated state, only token-shift.
    fn forward_with_knockout(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
        ffn_x_state: Option<&Tensor>,
        knockout_positions: &HashSet<usize>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        // Pre-LayerNorm (only first block)
        let hidden = if let Some(pre_ln) = &self.pre_ln {
            pre_ln.forward(hidden)?
        } else {
            hidden.clone()
        };

        // Attention with knockout and residual
        let (attn_out, new_attn_x, new_attn_kv) = self.attention.forward_with_knockout(
            &self.ln1.forward(&hidden)?,
            attn_x_state,
            attn_kv_state,
            knockout_positions,
        )?;
        let hidden = (&hidden + attn_out)?;

        // Feed-forward with residual (no knockout — stateless per-token)
        let (ffn_out, new_ffn_x) = self
            .feed_forward
            .forward(&self.ln2.forward(&hidden)?, ffn_x_state)?;
        let hidden = (&hidden + ffn_out)?;

        Ok((hidden, new_attn_x, new_attn_kv, new_ffn_x))
    }

    /// Forward pass with state steering at specified positions.
    ///
    /// Like knockout, but scales the kv write instead of suppressing it entirely.
    fn forward_with_steering(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
        ffn_x_state: Option<&Tensor>,
        steering_positions: &HashSet<usize>,
        scale: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        // Pre-LayerNorm (only first block)
        let hidden = if let Some(pre_ln) = &self.pre_ln {
            pre_ln.forward(hidden)?
        } else {
            hidden.clone()
        };

        // Attention with steering and residual
        let (attn_out, new_attn_x, new_attn_kv) = self.attention.forward_with_steering(
            &self.ln1.forward(&hidden)?,
            attn_x_state,
            attn_kv_state,
            steering_positions,
            scale,
        )?;
        let hidden = (&hidden + attn_out)?;

        // Feed-forward with residual (no steering — stateless per-token)
        let (ffn_out, new_ffn_x) = self
            .feed_forward
            .forward(&self.ln2.forward(&hidden)?, ffn_x_state)?;
        let hidden = (&hidden + ffn_out)?;

        Ok((hidden, new_attn_x, new_attn_kv, new_ffn_x))
    }

    /// Forward pass that also computes effective attention.
    ///
    /// Returns (output, new_attn_x, new_attn_kv, new_ffn_x, effective_attention)
    /// where effective_attention is `[batch, heads, seq, seq]`.
    fn forward_with_attention(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
        ffn_x_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let hidden = if let Some(pre_ln) = &self.pre_ln {
            pre_ln.forward(hidden)?
        } else {
            hidden.clone()
        };

        let (attn_out, new_attn_x, new_attn_kv, eff_attn) =
            self.attention.forward_with_effective_attention(
                &self.ln1.forward(&hidden)?,
                attn_x_state,
                attn_kv_state,
            )?;
        let hidden = (&hidden + attn_out)?;

        let (ffn_out, new_ffn_x) = self
            .feed_forward
            .forward(&self.ln2.forward(&hidden)?, ffn_x_state)?;
        let hidden = (&hidden + ffn_out)?;

        Ok((hidden, new_attn_x, new_attn_kv, new_ffn_x, eff_attn))
    }
}

// ============================================================================
// RWKV-6 State (for packing into/unpacking from KVCache)
// ============================================================================

/// RWKV recurrent state for all layers.
struct Rwkv6State {
    /// Previous hidden for attention token-shift: Vec of [batch, hidden_size] per layer
    attn_x: Vec<Option<Tensor>>,
    /// Accumulated WKV state: Vec of [batch, num_heads, head_size, head_size] per layer
    attn_kv: Vec<Option<Tensor>>,
    /// Previous hidden for FFN token-shift: Vec of [batch, hidden_size] per layer
    ffn_x: Vec<Option<Tensor>>,
}

impl Rwkv6State {
    fn new(n_layers: usize) -> Self {
        Self {
            attn_x: vec![None; n_layers],
            attn_kv: vec![None; n_layers],
            ffn_x: vec![None; n_layers],
        }
    }

    /// Pack state into a KVCache for the PlipBackend trait interface.
    ///
    /// Encoding:
    /// - keys[i] = concat(attn_x[i], ffn_x[i]) → [batch, 2 * hidden_size]
    /// - values[i] = attn_kv[i] → [batch, num_heads, head_size, head_size]
    fn to_kv_cache(&self, n_layers: usize) -> KVCache {
        let mut cache = KVCache::new(n_layers);
        for i in 0..n_layers {
            if let (Some(ax), Some(fx)) = (&self.attn_x[i], &self.ffn_x[i]) {
                if let Ok(combined) = Tensor::cat(&[ax, fx], D::Minus1) {
                    cache.keys[i] = Some(combined);
                }
            }
            cache.values[i].clone_from(&self.attn_kv[i]);
        }
        cache
    }

    /// Unpack state from a KVCache.
    fn from_kv_cache(cache: &KVCache, hidden_size: usize) -> Self {
        let n = cache.n_layers();
        let mut state = Self::new(n);
        for i in 0..n {
            if let Some(combined) = &cache.keys[i] {
                // Split concat(attn_x, ffn_x) back into two tensors
                if let (Ok(ax), Ok(fx)) = (
                    combined.narrow(D::Minus1, 0, hidden_size),
                    combined.narrow(D::Minus1, hidden_size, hidden_size),
                ) {
                    state.attn_x[i] = Some(ax);
                    state.ffn_x[i] = Some(fx);
                }
            }
            state.attn_kv[i].clone_from(&cache.values[i]);
        }
        state
    }
}

// ============================================================================
// PlipRwkv6 — Main model struct
// ============================================================================

pub struct PlipRwkv6 {
    embeddings: Embedding,
    blocks: Vec<Rwkv6Block>,
    ln_out: LayerNorm,
    head: Linear,
    config: Rwkv6Config,
    num_heads: usize,
}

impl PlipRwkv6 {
    pub fn load(model_id: &str, device: &Device, dtype: DType) -> Result<Self> {
        info!("Loading RWKV-6 model: {}", model_id);

        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        // Load config
        let config_path = repo
            .get("config.json")
            .context("Failed to download config.json")?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Rwkv6Config = serde_json::from_str(&config_str)?;

        info!(
            "RWKV-6 config: hidden_size={}, layers={}, heads={}, head_size={}, vocab={}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_heads(),
            config.head_size,
            config.vocab_size
        );

        // Load weights (safetensors — requires prior conversion from pytorch_model.bin)
        let weights_path = repo.get("model.safetensors").context(
            "Failed to download model.safetensors. \
                 RWKV-6 1B6 only ships pytorch_model.bin — run \
                 scripts/convert_rwkv_to_safetensors.py first.",
        )?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)? };

        // Build model
        let vb_rwkv = vb.pp("rwkv");

        let embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb_rwkv.pp("embeddings"),
        )?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let block = Rwkv6Block::load(vb_rwkv.pp(format!("blocks.{i}")), &config, i)?;
            blocks.push(block);
        }

        let ln_out = LayerNorm::load(
            config.hidden_size,
            config.layer_norm_epsilon,
            vb_rwkv.pp("ln_out"),
        )?;

        // LM head: may be tied to embeddings
        let head = if config.tie_word_embeddings {
            // Use embedding weights transposed as the LM head
            let head_weight = embeddings.embeddings().clone();
            Linear::new(head_weight, None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("head"))?
        };

        let num_heads = config.num_heads();

        info!(
            "RWKV-6 model loaded: {} layers, {} heads, intermediate_size={}",
            config.num_hidden_layers,
            num_heads,
            config.intermediate_size()
        );

        Ok(Self {
            embeddings,
            blocks,
            ln_out,
            head,
            config,
            num_heads,
        })
    }

    // --- Metadata (inherent methods) ---

    pub fn n_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub fn d_model(&self) -> usize {
        self.config.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn n_heads(&self) -> usize {
        self.num_heads
    }

    // --- Forward passes ---

    /// Full-sequence forward pass with activation capture.
    pub fn forward_with_cache(&self, input_ids: &Tensor) -> Result<(Tensor, ActivationCache)> {
        let mut cache = ActivationCache::with_capacity(self.n_layers());
        let mut hidden = self.embeddings.forward(input_ids)?;
        let mut state = Rwkv6State::new(self.n_layers());

        for (i, block) in self.blocks.iter().enumerate() {
            let (new_hidden, new_attn_x, new_attn_kv, new_ffn_x) = block.forward(
                &hidden,
                state.attn_x[i].as_ref(),
                state.attn_kv[i].as_ref(),
                state.ffn_x[i].as_ref(),
            )?;
            hidden = new_hidden;
            state.attn_x[i] = Some(new_attn_x);
            state.attn_kv[i] = Some(new_attn_kv);
            state.ffn_x[i] = Some(new_ffn_x);

            // Capture last-token activation
            let seq_len = hidden.dim(1)?;
            let last_token = hidden.i((.., seq_len - 1, ..))?.squeeze(1)?;
            cache.push(last_token);
        }

        let output = self.ln_out.forward(&hidden)?;
        Ok((output, cache))
    }

    /// Full-sequence forward pass with effective attention capture.
    ///
    /// Returns (ln_out_normalized_hidden, attention_cache) where the cache
    /// contains one `[batch, heads, seq, seq]` tensor per layer.
    pub fn forward_with_attention(&self, input_ids: &Tensor) -> Result<(Tensor, AttentionCache)> {
        let mut attn_cache = AttentionCache::with_capacity(self.n_layers());
        let mut hidden = self.embeddings.forward(input_ids)?;
        let mut state = Rwkv6State::new(self.n_layers());

        for (i, block) in self.blocks.iter().enumerate() {
            let (new_hidden, new_attn_x, new_attn_kv, new_ffn_x, eff_attn) = block
                .forward_with_attention(
                    &hidden,
                    state.attn_x[i].as_ref(),
                    state.attn_kv[i].as_ref(),
                    state.ffn_x[i].as_ref(),
                )?;
            hidden = new_hidden;
            state.attn_x[i] = Some(new_attn_x);
            state.attn_kv[i] = Some(new_attn_kv);
            state.ffn_x[i] = Some(new_ffn_x);

            attn_cache.push(eff_attn); // [b, h, t, t]

            info!(
                "Layer {}/{}: effective attention captured",
                i + 1,
                self.n_layers()
            );
        }

        let output = self.ln_out.forward(&hidden)?;
        Ok((output, attn_cache))
    }

    /// Create a new (empty) KV-cache for RWKV state.
    pub fn new_kv_cache(&self) -> KVCache {
        KVCache::new(self.n_layers())
    }

    /// Incremental forward pass using KV-cache (for generation).
    pub fn forward_with_kv_cache(
        &self,
        input_ids: &Tensor,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let mut state = if kv_cache.is_empty() {
            Rwkv6State::new(self.n_layers())
        } else {
            Rwkv6State::from_kv_cache(kv_cache, self.config.hidden_size)
        };

        let mut hidden = self.embeddings.forward(input_ids)?;

        for (i, block) in self.blocks.iter().enumerate() {
            let (new_hidden, new_attn_x, new_attn_kv, new_ffn_x) = block.forward(
                &hidden,
                state.attn_x[i].as_ref(),
                state.attn_kv[i].as_ref(),
                state.ffn_x[i].as_ref(),
            )?;
            hidden = new_hidden;
            state.attn_x[i] = Some(new_attn_x);
            state.attn_kv[i] = Some(new_attn_kv);
            state.ffn_x[i] = Some(new_ffn_x);
        }

        // Store state back into KV-cache
        *kv_cache = state.to_kv_cache(self.n_layers());

        // Apply final norm and project to logits (last token only)
        let output = self.ln_out.forward(&hidden)?;
        let seq_len = output.dim(1)?;
        let last_hidden = output.i((.., seq_len - 1, ..))?.squeeze(1)?;
        self.head.forward(&last_hidden).map_err(Into::into)
    }

    /// Full-sequence forward pass with state knockout.
    ///
    /// At knocked-out positions in targeted layers, the WKV state update
    /// applies only decay (no kv write), suppressing the position's influence
    /// on future tokens.
    ///
    /// Returns ln_out-normalized hidden states (same shape as `forward_with_cache` output).
    pub fn forward_with_state_knockout(
        &self,
        input_ids: &Tensor,
        spec: &StateKnockoutSpec,
    ) -> Result<Tensor> {
        let knockout_positions = spec.position_set();
        let mut hidden = self.embeddings.forward(input_ids)?;
        let mut state = Rwkv6State::new(self.n_layers());

        for (i, block) in self.blocks.iter().enumerate() {
            let (new_hidden, new_attn_x, new_attn_kv, new_ffn_x) = if spec.applies_to_layer(i) {
                block.forward_with_knockout(
                    &hidden,
                    state.attn_x[i].as_ref(),
                    state.attn_kv[i].as_ref(),
                    state.ffn_x[i].as_ref(),
                    &knockout_positions,
                )?
            } else {
                block.forward(
                    &hidden,
                    state.attn_x[i].as_ref(),
                    state.attn_kv[i].as_ref(),
                    state.ffn_x[i].as_ref(),
                )?
            };
            hidden = new_hidden;
            state.attn_x[i] = Some(new_attn_x);
            state.attn_kv[i] = Some(new_attn_kv);
            state.ffn_x[i] = Some(new_ffn_x);
        }

        let output = self.ln_out.forward(&hidden)?;
        Ok(output)
    }

    /// Full-sequence forward pass with state steering.
    ///
    /// At steered positions in targeted layers, the WKV state update scales
    /// the kv write by the specified factor, amplifying or dampening the
    /// position's influence on future tokens.
    ///
    /// Returns ln_out-normalized hidden states (same shape as `forward_with_cache` output).
    pub fn forward_with_state_steering(
        &self,
        input_ids: &Tensor,
        spec: &StateSteeringSpec,
    ) -> Result<Tensor> {
        let steering_positions = spec.position_set();
        let mut hidden = self.embeddings.forward(input_ids)?;
        let mut state = Rwkv6State::new(self.n_layers());

        for (i, block) in self.blocks.iter().enumerate() {
            let (new_hidden, new_attn_x, new_attn_kv, new_ffn_x) = if spec.applies_to_layer(i) {
                block.forward_with_steering(
                    &hidden,
                    state.attn_x[i].as_ref(),
                    state.attn_kv[i].as_ref(),
                    state.ffn_x[i].as_ref(),
                    &steering_positions,
                    spec.scale,
                )?
            } else {
                block.forward(
                    &hidden,
                    state.attn_x[i].as_ref(),
                    state.attn_kv[i].as_ref(),
                    state.ffn_x[i].as_ref(),
                )?
            };
            hidden = new_hidden;
            state.attn_x[i] = Some(new_attn_x);
            state.attn_kv[i] = Some(new_attn_kv);
            state.ffn_x[i] = Some(new_ffn_x);
        }

        let output = self.ln_out.forward(&hidden)?;
        Ok(output)
    }

    /// Incremental forward pass with state steering and KV-cache support.
    ///
    /// Hybrid of `forward_with_state_steering` (applies steering at specified
    /// positions/layers) and `forward_with_kv_cache` (stores resulting state
    /// into the cache for subsequent generation steps).
    ///
    /// Returns logits for the last token (same as `forward_with_kv_cache`).
    pub fn forward_with_state_steering_kv_cache(
        &self,
        input_ids: &Tensor,
        kv_cache: &mut KVCache,
        spec: &StateSteeringSpec,
    ) -> Result<Tensor> {
        let steering_positions = spec.position_set();
        let mut state = if kv_cache.is_empty() {
            Rwkv6State::new(self.n_layers())
        } else {
            Rwkv6State::from_kv_cache(kv_cache, self.config.hidden_size)
        };

        let mut hidden = self.embeddings.forward(input_ids)?;

        for (i, block) in self.blocks.iter().enumerate() {
            let (new_hidden, new_attn_x, new_attn_kv, new_ffn_x) = if spec.applies_to_layer(i) {
                block.forward_with_steering(
                    &hidden,
                    state.attn_x[i].as_ref(),
                    state.attn_kv[i].as_ref(),
                    state.ffn_x[i].as_ref(),
                    &steering_positions,
                    spec.scale,
                )?
            } else {
                block.forward(
                    &hidden,
                    state.attn_x[i].as_ref(),
                    state.attn_kv[i].as_ref(),
                    state.ffn_x[i].as_ref(),
                )?
            };
            hidden = new_hidden;
            state.attn_x[i] = Some(new_attn_x);
            state.attn_kv[i] = Some(new_attn_kv);
            state.ffn_x[i] = Some(new_ffn_x);
        }

        // Store steered state back into KV-cache for generation
        *kv_cache = state.to_kv_cache(self.n_layers());

        // Apply final norm and project to logits (last token only)
        let output = self.ln_out.forward(&hidden)?;
        let seq_len = output.dim(1)?;
        let last_hidden = output.i((.., seq_len - 1, ..))?.squeeze(1)?;
        self.head.forward(&last_hidden).map_err(Into::into)
    }

    /// Autoregressive generation with state steering during prompt prefill.
    ///
    /// The prompt is processed with state steering applied at the specified
    /// positions and layers (scaling the kv^T state write by the spec's scale
    /// factor). The steered recurrent state then propagates naturally to all
    /// generated tokens — no re-steering is needed during generation.
    pub fn generate_with_state_steering(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        spec: &StateSteeringSpec,
        device: &Device,
    ) -> Result<Vec<u32>> {
        let mut kv_cache = self.new_kv_cache();
        let mut tokens = prompt_ids.to_vec();

        // Prefill: process entire prompt with state steering
        let prompt_tensor = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let logits =
            self.forward_with_state_steering_kv_cache(&prompt_tensor, &mut kv_cache, spec)?;

        // Sample first generated token
        let mut next_token = crate::model::sample_token(&logits, temperature)?;

        if stop_tokens.contains(&next_token) {
            return Ok(tokens);
        }
        tokens.push(next_token);

        // Generate remaining tokens (no steering — state propagates naturally)
        for _ in 1..max_tokens {
            let input_tensor = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let logits = self.forward_with_kv_cache(&input_tensor, &mut kv_cache)?;

            next_token = crate::model::sample_token(&logits, temperature)?;

            if stop_tokens.contains(&next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Autoregressive generation.
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        device: &Device,
    ) -> Result<Vec<u32>> {
        let mut kv_cache = self.new_kv_cache();
        let mut tokens = prompt_ids.to_vec();

        // Prefill: process entire prompt
        let prompt_tensor = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let logits = self.forward_with_kv_cache(&prompt_tensor, &mut kv_cache)?;

        // Sample first generated token
        let mut next_token = crate::model::sample_token(&logits, temperature)?;

        if stop_tokens.contains(&next_token) {
            return Ok(tokens);
        }
        tokens.push(next_token);

        // Generate remaining tokens
        for _ in 1..max_tokens {
            let input_tensor = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let logits = self.forward_with_kv_cache(&input_tensor, &mut kv_cache)?;

            next_token = crate::model::sample_token(&logits, temperature)?;

            if stop_tokens.contains(&next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    // --- Logit lens ---

    pub fn logit_lens(&self, activation: &Tensor) -> Result<Tensor> {
        // Apply final layer norm, then project to vocab
        let normed = self.ln_out.forward(&activation.unsqueeze(0)?)?;
        let normed = normed.squeeze(0)?;
        self.head.forward(&normed).map_err(Into::into)
    }

    pub fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        self.head.forward(hidden).map_err(Into::into)
    }

    pub fn logit_lens_top_k(&self, activation: &Tensor, k: usize) -> Result<Vec<(u32, f32)>> {
        let logits = self.logit_lens(activation)?;
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let logits_vec: Vec<f32> = logits_f32.flatten_all()?.to_vec1()?;

        // Softmax
        let max_val = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_vec.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|x| x / sum).collect();

        // Top-k
        let mut indexed: Vec<(u32, f32)> = probs
            .into_iter()
            .enumerate()
            .map(|(i, p)| (i as u32, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        Ok(indexed)
    }
}

// ============================================================================
// PlipBackend trait implementation
// ============================================================================

impl PlipBackend for PlipRwkv6 {
    fn n_layers(&self) -> usize {
        self.n_layers()
    }
    fn d_model(&self) -> usize {
        self.d_model()
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
    fn n_heads(&self) -> usize {
        self.n_heads()
    }

    fn forward_with_cache(&self, input_ids: &Tensor) -> Result<(Tensor, ActivationCache)> {
        self.forward_with_cache(input_ids)
    }

    fn forward_with_attention(&self, input_ids: &Tensor) -> Result<(Tensor, AttentionCache)> {
        self.forward_with_attention(input_ids)
    }

    fn forward_with_intervention(
        &self,
        _input_ids: &Tensor,
        _spec: &KnockoutSpec,
    ) -> Result<(Tensor, AttentionCache)> {
        anyhow::bail!(
            "Attention knockout not applicable to RWKV-6. \
             Use state knockout (Phase 4) instead."
        )
    }

    fn forward_with_state_knockout(
        &self,
        input_ids: &Tensor,
        spec: &StateKnockoutSpec,
    ) -> Result<Tensor> {
        self.forward_with_state_knockout(input_ids, spec)
    }

    fn forward_with_state_steering(
        &self,
        input_ids: &Tensor,
        spec: &StateSteeringSpec,
    ) -> Result<Tensor> {
        self.forward_with_state_steering(input_ids, spec)
    }

    fn generate_with_state_steering(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        spec: &StateSteeringSpec,
        device: &Device,
    ) -> Result<Vec<u32>> {
        self.generate_with_state_steering(
            prompt_ids,
            max_tokens,
            temperature,
            stop_tokens,
            spec,
            device,
        )
    }

    fn logit_lens(&self, activation: &Tensor) -> Result<Tensor> {
        self.logit_lens(activation)
    }
    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        self.project_to_vocab(hidden)
    }
    fn logit_lens_top_k(&self, activation: &Tensor, k: usize) -> Result<Vec<(u32, f32)>> {
        self.logit_lens_top_k(activation, k)
    }

    fn new_kv_cache(&self) -> KVCache {
        self.new_kv_cache()
    }
    fn forward_with_kv_cache(&self, input_ids: &Tensor, kv_cache: &mut KVCache) -> Result<Tensor> {
        self.forward_with_kv_cache(input_ids, kv_cache)
    }
    fn generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        device: &Device,
    ) -> Result<Vec<u32>> {
        self.generate(prompt_ids, max_tokens, temperature, stop_tokens, device)
    }
}
