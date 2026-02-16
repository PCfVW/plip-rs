//! Gemma 2 forward pass with per-layer activation capture
//!
//! Custom implementation for Gemma 2 2B (google/gemma-2-2b) that supports:
//! - Alternating sliding window / global attention per layer
//! - Attention logit soft-capping (50.0) and final logit soft-capping (30.0)
//! - Four-norm decoder layers (pre/post attention + pre/post MLP)
//! - GQA with explicit head_dim=256 (not derived from hidden_size/num_heads)
//! - CLT injection into the residual stream during forward passes

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::Rng;
use tracing::info;

use crate::attention::AttentionCache;
use crate::cache::{ActivationCache, FullActivationCache};
use crate::intervention::{apply_steering, CltInjectionSpec, KnockoutSpec, SteeringSpec};
use crate::kv_cache::KVCache;
use crate::masks::create_causal_mask;
use crate::model::PlipBackend;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Gemma 2 model configuration (matches HuggingFace config.json).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub attn_logit_softcapping: Option<f64>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,
    #[serde(default = "default_query_pre_attn_scalar")]
    pub query_pre_attn_scalar: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub attention_bias: bool,
}

fn default_rope_theta() -> f64 {
    10000.0
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_max_position_embeddings() -> usize {
    8192
}
fn default_query_pre_attn_scalar() -> usize {
    256
}

/// Safetensors index for sharded models.
#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// RmsNorm (Gemma-style: weight + 1.0)
// ---------------------------------------------------------------------------

struct GemmaRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl GemmaRmsNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let weight_plus_one = (&self.weight.to_dtype(internal_dtype)? + 1.0)?;
        Ok(x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&weight_plus_one.to_dtype(x_dtype)?)?)
    }
}

// ---------------------------------------------------------------------------
// Rotary Embedding
// ---------------------------------------------------------------------------

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0f32 / (rope_theta.powf(i as f64 / head_dim as f64) as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    n_rep: usize,
    head_dim: usize,
    attn_logit_softcapping: Option<f64>,
    scale: f64,
}

impl Attention {
    fn load(vb: VarBuilder, config: &Gemma2Config) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;

        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let scale = 1.0 / (config.query_pre_attn_scalar as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            n_rep: num_heads / num_kv_heads,
            head_dim,
            attn_logit_softcapping: config.attn_logit_softcapping,
            scale,
        })
    }

    /// Standard forward (for forward_with_cache, forward_with_clt_injection).
    fn forward(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        mask: &Tensor,
        start_pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary.apply(&q, &k, start_pos)?;

        // GQA: expand KV heads
        let k = repeat_kv(k, self.n_rep)?;
        let v = repeat_kv(v, self.n_rep)?;

        // Contiguous for matmul
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // Scaled dot-product
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        // Attention logit soft-capping
        if let Some(sc) = self.attn_logit_softcapping {
            attn_weights = ((attn_weights / sc)?.tanh()? * sc)?;
        }

        // Apply causal mask
        attn_weights = attn_weights.broadcast_add(mask)?;

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Weighted sum
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;

        Ok(self.o_proj.forward(&attn_output)?)
    }

    /// Forward returning attention weights for capture.
    fn forward_with_attn(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        mask: &Tensor,
        start_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary.apply(&q, &k, start_pos)?;

        let k = repeat_kv(k, self.n_rep)?;
        let v = repeat_kv(v, self.n_rep)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        if let Some(sc) = self.attn_logit_softcapping {
            attn_weights = ((attn_weights / sc)?.tanh()? * sc)?;
        }

        attn_weights = attn_weights.broadcast_add(mask)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;

        Ok((self.o_proj.forward(&attn_output)?, attn_weights))
    }

    /// Forward with intervention (knockout mask on attention).
    fn forward_with_intervention(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        mask: &Tensor,
        start_pos: usize,
        knockout_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary.apply(&q, &k, start_pos)?;

        let k = repeat_kv(k, self.n_rep)?;
        let v = repeat_kv(v, self.n_rep)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        if let Some(sc) = self.attn_logit_softcapping {
            attn_weights = ((attn_weights / sc)?.tanh()? * sc)?;
        }

        // Apply both causal mask and knockout mask
        attn_weights = attn_weights.broadcast_add(mask)?;
        attn_weights = (&attn_weights + knockout_mask)?;

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;

        Ok((self.o_proj.forward(&attn_output)?, attn_weights))
    }

    /// Forward with KV-cache (single token autoregressive generation).
    fn forward_with_cache(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
        k_cache: &mut Option<Tensor>,
        v_cache: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary.apply(&q, &k, start_pos)?;

        // Update KV cache
        let (k, v) = match (k_cache.as_ref(), v_cache.as_ref()) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            _ => (k, v),
        };
        *k_cache = Some(k.clone());
        *v_cache = Some(v.clone());

        // GQA expand
        let k = repeat_kv(k, self.n_rep)?;
        let v = repeat_kv(v, self.n_rep)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        if let Some(sc) = self.attn_logit_softcapping {
            attn_weights = ((attn_weights / sc)?.tanh()? * sc)?;
        }

        // No mask needed for single-token generation (seq_len=1)
        // For multi-token (prefill), we need a mask - but KV-cache generation
        // typically does prefill without cache, then single tokens with cache.
        // The mask handling is done at the model level.
        if seq_len > 1 {
            // During prefill, create causal mask
            let mask = create_causal_mask(seq_len, x.device(), x.dtype())?;
            attn_weights = attn_weights.broadcast_add(&mask)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;

        Ok(self.o_proj.forward(&attn_output)?)
    }

    /// Forward with post-softmax steering intervention.
    fn forward_with_steering(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        mask: &Tensor,
        start_pos: usize,
        steering_spec: &SteeringSpec,
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary.apply(&q, &k, start_pos)?;

        let k = repeat_kv(k, self.n_rep)?;
        let v = repeat_kv(v, self.n_rep)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        if let Some(sc) = self.attn_logit_softcapping {
            attn_weights = ((attn_weights / sc)?.tanh()? * sc)?;
        }

        attn_weights = attn_weights.broadcast_add(mask)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Post-softmax steering
        let attn_weights = if steering_spec.is_steering() {
            apply_steering(&attn_weights, steering_spec, self.num_heads, seq_len)?
        } else {
            attn_weights
        };

        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;

        Ok((self.o_proj.forward(&attn_output)?, attn_weights))
    }

    /// Forward with KV-cache and post-softmax steering.
    fn forward_with_cache_and_steering(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
        steering_spec: &SteeringSpec,
        k_cache: &mut Option<Tensor>,
        v_cache: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary.apply(&q, &k, start_pos)?;

        // Update KV cache
        let (k, v) = match (k_cache.as_ref(), v_cache.as_ref()) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            _ => (k, v),
        };
        *k_cache = Some(k.clone());
        *v_cache = Some(v.clone());

        // GQA expand
        let k = repeat_kv(k, self.n_rep)?;
        let v = repeat_kv(v, self.n_rep)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let total_seq_len = k.dim(2)?;

        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        if let Some(sc) = self.attn_logit_softcapping {
            attn_weights = ((attn_weights / sc)?.tanh()? * sc)?;
        }

        if seq_len > 1 {
            let mask = create_causal_mask(seq_len, x.device(), x.dtype())?;
            attn_weights = attn_weights.broadcast_add(&mask)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Post-softmax steering
        let attn_weights = if steering_spec.is_steering() {
            apply_steering(&attn_weights, steering_spec, self.num_heads, total_seq_len)?
        } else {
            attn_weights
        };

        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;

        Ok(self.o_proj.forward(&attn_output)?)
    }
}

/// Expand KV heads for GQA.
fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    let x = x
        .unsqueeze(2)?
        .expand((b, num_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((b, num_kv_heads * n_rep, seq_len, head_dim))?;
    Ok(x)
}

// ---------------------------------------------------------------------------
// MLP (GeGLU with gelu_pytorch_tanh)
// ---------------------------------------------------------------------------

#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    fn load(vb: VarBuilder, config: &Gemma2Config) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.gelu()?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        Ok(self.down_proj.forward(&hidden)?)
    }
}

// ---------------------------------------------------------------------------
// Decoder Layer (4 norms: pre/post attention + pre/post MLP)
// ---------------------------------------------------------------------------

struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
}

impl DecoderLayer {
    fn load(vb: VarBuilder, config: &Gemma2Config, _layer_idx: usize) -> Result<Self> {
        let self_attn = Attention::load(vb.pp("self_attn"), config)?;
        let mlp = MLP::load(vb.pp("mlp"), config)?;
        let input_layernorm = GemmaRmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = GemmaRmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = GemmaRmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = GemmaRmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    /// Standard forward.
    fn forward(
        &self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        mask: &Tensor,
        start_pos: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, rotary, mask, start_pos)?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.pre_feedforward_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_feedforward_layernorm.forward(&xs)?;
        Ok((residual + xs)?)
    }

    /// Forward with attention weight capture.
    fn forward_with_attn(
        &self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        mask: &Tensor,
        start_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let (xs, attn_weights) = self
            .self_attn
            .forward_with_attn(&xs, rotary, mask, start_pos)?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.pre_feedforward_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_feedforward_layernorm.forward(&xs)?;
        Ok(((residual + xs)?, attn_weights))
    }

    /// Forward with attention intervention (knockout mask).
    fn forward_with_intervention(
        &self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        mask: &Tensor,
        start_pos: usize,
        knockout_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let (xs, attn_weights) = self.self_attn.forward_with_intervention(
            &xs,
            rotary,
            mask,
            start_pos,
            knockout_mask,
        )?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.pre_feedforward_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_feedforward_layernorm.forward(&xs)?;
        Ok(((residual + xs)?, attn_weights))
    }

    /// Forward with KV-cache for autoregressive generation.
    fn forward_with_cache(
        &self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
        k_cache: &mut Option<Tensor>,
        v_cache: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_with_cache(&xs, rotary, start_pos, k_cache, v_cache)?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.pre_feedforward_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_feedforward_layernorm.forward(&xs)?;
        Ok((residual + xs)?)
    }

    /// Forward with post-softmax steering intervention.
    fn forward_with_steering(
        &self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        mask: &Tensor,
        start_pos: usize,
        steering_spec: &SteeringSpec,
    ) -> Result<(Tensor, Tensor)> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let (xs, attn_weights) =
            self.self_attn
                .forward_with_steering(&xs, rotary, mask, start_pos, steering_spec)?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.pre_feedforward_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_feedforward_layernorm.forward(&xs)?;
        Ok(((residual + xs)?, attn_weights))
    }

    /// Forward with KV-cache and post-softmax steering.
    fn forward_with_cache_and_steering(
        &self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
        steering_spec: &SteeringSpec,
        k_cache: &mut Option<Tensor>,
        v_cache: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_with_cache_and_steering(
            &xs,
            rotary,
            start_pos,
            steering_spec,
            k_cache,
            v_cache,
        )?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.pre_feedforward_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_feedforward_layernorm.forward(&xs)?;
        Ok((residual + xs)?)
    }
}

// ---------------------------------------------------------------------------
// Attention mask helpers
// ---------------------------------------------------------------------------

/// Create a sliding window causal mask.
///
/// Positions where `j > i` (future) or `i - j > window` (too far back) are -inf.
fn create_sliding_window_mask(
    seq_len: usize,
    window: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| {
                if j > i || i.saturating_sub(j) > window {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect();
    Ok(Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)?)
}

/// Create a global causal mask (standard causal, no window restriction).
fn create_global_causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Ok(Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)?)
}

// ---------------------------------------------------------------------------
// PlipGemma2 — main model struct
// ---------------------------------------------------------------------------

/// Gemma 2 model with per-layer activation capture and CLT injection support.
pub struct PlipGemma2 {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: GemmaRmsNorm,
    rotary: RotaryEmbedding,
    config: Gemma2Config,
}

impl PlipGemma2 {
    /// Load Gemma 2 model from HuggingFace.
    pub fn load(model_id: &str, device: &Device, dtype: DType) -> Result<Self> {
        info!("Loading Gemma 2 from: {}", model_id);

        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let config_path = repo
            .get("config.json")
            .context("Failed to download config.json")?;
        let config_str = std::fs::read_to_string(&config_path).context("Failed to read config")?;
        let config: Gemma2Config = serde_json::from_str(&config_str)?;

        info!(
            "Gemma 2 config: {} layers, hidden={}, heads={}, kv_heads={}, head_dim={}, vocab={}",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.vocab_size
        );
        info!(
            "  attn_softcap={:?}, final_softcap={:?}, sliding_window={:?}",
            config.attn_logit_softcapping, config.final_logit_softcapping, config.sliding_window
        );

        // Handle sharded vs single safetensors
        let weights_paths = if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            info!("Model is sharded, loading index...");
            let index_str = std::fs::read_to_string(&index_path).context("Failed to read index")?;
            let index: SafetensorsIndex = serde_json::from_str(&index_str)?;
            let mut shard_names: Vec<String> = index.weight_map.values().cloned().collect();
            shard_names.sort();
            shard_names.dedup();
            info!("Downloading {} shard files...", shard_names.len());
            let mut paths = Vec::new();
            for shard_name in &shard_names {
                let path = repo
                    .get(shard_name)
                    .with_context(|| format!("Failed to download {shard_name}"))?;
                paths.push(path);
            }
            paths
        } else {
            let path = repo
                .get("model.safetensors")
                .context("Failed to download model.safetensors")?;
            vec![path]
        };

        info!("Loading weights from {} file(s)...", weights_paths.len());

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_paths, dtype, device)? };
        let vb_model = vb.pp("model");

        // Build model components
        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb_model.pp("embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            if (i + 1) % 10 == 0 || i == 0 {
                info!("Loading layer {}/{}", i + 1, config.num_hidden_layers);
            }
            let layer = DecoderLayer::load(vb_model.pp(format!("layers.{i}")), &config, i)?;
            layers.push(layer);
        }

        let norm =
            GemmaRmsNorm::load(config.hidden_size, config.rms_norm_eps, vb_model.pp("norm"))?;

        let rotary = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
            dtype,
        )?;

        info!(
            "Gemma 2 loaded: {} layers (vocab_size: {})",
            config.num_hidden_layers, config.vocab_size
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
            config,
        })
    }

    /// Apply final logit soft-capping: `sc * tanh(logits / sc)`.
    fn apply_final_softcap(&self, logits: &Tensor) -> Result<Tensor> {
        match self.config.final_logit_softcapping {
            Some(sc) => Ok(((logits / sc)?.tanh()? * sc)?),
            None => Ok(logits.clone()),
        }
    }

    /// Get the mask for a specific layer (alternating sliding window / global).
    ///
    /// Even layers (0, 2, 4, ...): sliding window
    /// Odd layers (1, 3, 5, ...): global causal
    fn mask_for_layer(
        &self,
        layer_idx: usize,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        if layer_idx.is_multiple_of(2) {
            // Even: sliding window
            if let Some(window) = self.config.sliding_window {
                create_sliding_window_mask(seq_len, window, device, dtype)
            } else {
                create_global_causal_mask(seq_len, device, dtype)
            }
        } else {
            // Odd: global causal
            create_global_causal_mask(seq_len, device, dtype)
        }
    }

    // -----------------------------------------------------------------------
    // Forward passes
    // -----------------------------------------------------------------------

    /// Forward pass with per-layer activation capture.
    pub fn forward_with_cache(&self, input_ids: &Tensor) -> Result<(Tensor, ActivationCache)> {
        let seq_len = input_ids.dim(1)?;
        let device = input_ids.device();
        let dtype = self.embed_tokens.embeddings().dtype();

        // Embedding with sqrt(hidden_size) scaling
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let normalizer = (self.config.hidden_size as f64).sqrt();
        hidden = (hidden * normalizer)?;

        let mut cache = ActivationCache::with_capacity(self.config.num_hidden_layers);

        for (i, layer) in self.layers.iter().enumerate() {
            let mask = self.mask_for_layer(i, seq_len, device, dtype)?;
            hidden = layer.forward(&hidden, &self.rotary, &mask, 0)?;

            // Capture last-token activation
            let last_token = hidden.i((.., seq_len - 1, ..))?.squeeze(1)?;
            cache.push(last_token);
        }

        // Final norm
        let output = self.norm.forward(&hidden)?;
        Ok((output, cache))
    }

    /// Forward pass with all-position activation capture.
    ///
    /// Same as [`forward_with_cache`](Self::forward_with_cache) but stores the
    /// full residual stream at every token position per layer, not just the last token.
    /// Each cached tensor has shape `(seq_len, d_model)`.
    pub fn forward_with_full_cache(
        &self,
        input_ids: &Tensor,
    ) -> Result<(Tensor, FullActivationCache)> {
        let seq_len = input_ids.dim(1)?;
        let device = input_ids.device();
        let dtype = self.embed_tokens.embeddings().dtype();

        // Embedding with sqrt(hidden_size) scaling
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let normalizer = (self.config.hidden_size as f64).sqrt();
        hidden = (hidden * normalizer)?;

        let mut cache = FullActivationCache::with_capacity(self.config.num_hidden_layers);

        for (i, layer) in self.layers.iter().enumerate() {
            let mask = self.mask_for_layer(i, seq_len, device, dtype)?;
            hidden = layer.forward(&hidden, &self.rotary, &mask, 0)?;

            // Store all positions: squeeze batch dim [1, seq_len, d_model] → [seq_len, d_model]
            let all_positions = hidden.squeeze(0)?;
            cache.push(all_positions);
        }

        // Final norm
        let output = self.norm.forward(&hidden)?;
        Ok((output, cache))
    }

    /// Forward pass with attention weight capture.
    pub fn forward_with_attention(&self, input_ids: &Tensor) -> Result<(Tensor, AttentionCache)> {
        let seq_len = input_ids.dim(1)?;
        let device = input_ids.device();
        let dtype = self.embed_tokens.embeddings().dtype();

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let normalizer = (self.config.hidden_size as f64).sqrt();
        hidden = (hidden * normalizer)?;

        let mut attn_cache = AttentionCache::with_capacity(self.config.num_hidden_layers);

        for (i, layer) in self.layers.iter().enumerate() {
            let mask = self.mask_for_layer(i, seq_len, device, dtype)?;
            let (h, attn_weights) = layer.forward_with_attn(&hidden, &self.rotary, &mask, 0)?;
            hidden = h;
            attn_cache.push(attn_weights);
        }

        let output = self.norm.forward(&hidden)?;
        Ok((output, attn_cache))
    }

    /// Forward pass with attention knockout intervention.
    pub fn forward_with_intervention(
        &self,
        input_ids: &Tensor,
        spec: &KnockoutSpec,
    ) -> Result<(Tensor, AttentionCache)> {
        let seq_len = input_ids.dim(1)?;
        let device = input_ids.device();
        let dtype = self.embed_tokens.embeddings().dtype();

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let normalizer = (self.config.hidden_size as f64).sqrt();
        hidden = (hidden * normalizer)?;

        let mut attn_cache = AttentionCache::with_capacity(self.config.num_hidden_layers);

        for (i, layer) in self.layers.iter().enumerate() {
            let mask = self.mask_for_layer(i, seq_len, device, dtype)?;
            if spec.applies_to_layer(i) {
                let knockout_mask = crate::intervention::create_knockout_mask(
                    spec,
                    self.config.num_attention_heads,
                    seq_len,
                    device,
                    dtype,
                )?;
                let (h, attn_weights) = layer.forward_with_intervention(
                    &hidden,
                    &self.rotary,
                    &mask,
                    0,
                    &knockout_mask,
                )?;
                hidden = h;
                attn_cache.push(attn_weights);
            } else {
                let (h, attn_weights) = layer.forward_with_attn(&hidden, &self.rotary, &mask, 0)?;
                hidden = h;
                attn_cache.push(attn_weights);
            }
        }

        let output = self.norm.forward(&hidden)?;
        Ok((output, attn_cache))
    }

    /// Forward pass with KV-cache for autoregressive generation.
    pub fn forward_with_kv_cache(
        &self,
        input_ids: &Tensor,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let start_pos = kv_cache.seq_len();

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let normalizer = (self.config.hidden_size as f64).sqrt();
        hidden = (hidden * normalizer)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_with_cache(
                &hidden,
                &self.rotary,
                start_pos,
                &mut kv_cache.keys[i],
                &mut kv_cache.values[i],
            )?;
        }

        let output = self.norm.forward(&hidden)?;
        let last_hidden = output.i((.., seq_len - 1, ..))?.squeeze(1)?;
        let logits = last_hidden.matmul(&self.embed_tokens.embeddings().t()?)?;
        self.apply_final_softcap(&logits)
    }

    /// Forward pass with attention steering intervention (post-softmax).
    pub fn forward_with_steering(
        &self,
        input_ids: &Tensor,
        spec: &SteeringSpec,
    ) -> Result<(Tensor, AttentionCache)> {
        let seq_len = input_ids.dim(1)?;
        let device = input_ids.device();
        let dtype = self.embed_tokens.embeddings().dtype();

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let normalizer = (self.config.hidden_size as f64).sqrt();
        hidden = (hidden * normalizer)?;

        let mut attn_cache = AttentionCache::with_capacity(self.config.num_hidden_layers);

        for (i, layer) in self.layers.iter().enumerate() {
            let mask = self.mask_for_layer(i, seq_len, device, dtype)?;
            if spec.applies_to_layer(i) {
                let (h, attn_weights) =
                    layer.forward_with_steering(&hidden, &self.rotary, &mask, 0, spec)?;
                hidden = h;
                attn_cache.push(attn_weights);
            } else {
                let (h, attn_weights) = layer.forward_with_attn(&hidden, &self.rotary, &mask, 0)?;
                hidden = h;
                attn_cache.push(attn_weights);
            }
        }

        let output = self.norm.forward(&hidden)?;
        Ok((output, attn_cache))
    }

    /// Forward pass with KV-cache and steering for prompt-steered generation.
    pub fn forward_with_kv_cache_and_steering(
        &self,
        input_ids: &Tensor,
        kv_cache: &mut KVCache,
        steering_spec: &SteeringSpec,
    ) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let start_pos = kv_cache.seq_len();

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let normalizer = (self.config.hidden_size as f64).sqrt();
        hidden = (hidden * normalizer)?;

        for (i, layer) in self.layers.iter().enumerate() {
            if steering_spec.applies_to_layer(i) {
                hidden = layer.forward_with_cache_and_steering(
                    &hidden,
                    &self.rotary,
                    start_pos,
                    steering_spec,
                    &mut kv_cache.keys[i],
                    &mut kv_cache.values[i],
                )?;
            } else {
                hidden = layer.forward_with_cache(
                    &hidden,
                    &self.rotary,
                    start_pos,
                    &mut kv_cache.keys[i],
                    &mut kv_cache.values[i],
                )?;
            }
        }

        let output = self.norm.forward(&hidden)?;
        let last_hidden = output.i((.., seq_len - 1, ..))?.squeeze(1)?;
        let logits = last_hidden.matmul(&self.embed_tokens.embeddings().t()?)?;
        self.apply_final_softcap(&logits)
    }

    /// Generate with steering applied to prompt, then standard KV-cache generation.
    pub fn generate_with_prompt_steering(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        steering_spec: &SteeringSpec,
        device: &Device,
    ) -> Result<Vec<u32>> {
        let mut kv_cache = KVCache::new(self.config.num_hidden_layers);
        let mut tokens = prompt_ids.to_vec();

        // Prefill with steering
        let prompt_tensor = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let logits =
            self.forward_with_kv_cache_and_steering(&prompt_tensor, &mut kv_cache, steering_spec)?;

        let mut next_token = sample_token(&logits, temperature)?;
        if stop_tokens.contains(&next_token) {
            return Ok(tokens);
        }
        tokens.push(next_token);

        // Continue generation without steering (standard KV-cache)
        for _ in 1..max_tokens {
            let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let logits = self.forward_with_kv_cache(&input, &mut kv_cache)?;
            next_token = sample_token(&logits, temperature)?;
            if stop_tokens.contains(&next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Forward pass with CLT injection at specified layers, filling KV-cache.
    pub fn forward_with_clt_injection(
        &self,
        input_ids: &Tensor,
        clt_spec: &CltInjectionSpec,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let start_pos = kv_cache.seq_len();

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let normalizer = (self.config.hidden_size as f64).sqrt();
        hidden = (hidden * normalizer)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_with_cache(
                &hidden,
                &self.rotary,
                start_pos,
                &mut kv_cache.keys[i],
                &mut kv_cache.values[i],
            )?;

            // CLT INJECTION: add steering vectors at this layer
            for inj in clt_spec.injections_for_layer(i) {
                hidden = inject_at_position(&hidden, &inj.vector, inj.position)?;
            }
        }

        let output = self.norm.forward(&hidden)?;
        let last_hidden = output.i((.., seq_len - 1, ..))?.squeeze(1)?;
        let logits = last_hidden.matmul(&self.embed_tokens.embeddings().t()?)?;
        self.apply_final_softcap(&logits)
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
        let mut kv_cache = KVCache::new(self.config.num_hidden_layers);
        let mut tokens = prompt_ids.to_vec();

        // Prefill: process entire prompt
        let prompt_tensor = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let logits = self.forward_with_kv_cache(&prompt_tensor, &mut kv_cache)?;

        let mut next_token = sample_token(&logits, temperature)?;
        if stop_tokens.contains(&next_token) {
            return Ok(tokens);
        }
        tokens.push(next_token);

        // Autoregressive: one token at a time
        for _ in 1..max_tokens {
            let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let logits = self.forward_with_kv_cache(&input, &mut kv_cache)?;
            next_token = sample_token(&logits, temperature)?;
            if stop_tokens.contains(&next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Generate with CLT injection during prompt processing.
    pub fn generate_with_clt_injection(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        clt_spec: &CltInjectionSpec,
        device: &Device,
    ) -> Result<Vec<u32>> {
        let mut kv_cache = KVCache::new(self.config.num_hidden_layers);
        let mut tokens = prompt_ids.to_vec();

        // Prefill with CLT injection
        let prompt_tensor = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let logits = self.forward_with_clt_injection(&prompt_tensor, clt_spec, &mut kv_cache)?;

        let mut next_token = sample_token(&logits, temperature)?;
        if stop_tokens.contains(&next_token) {
            return Ok(tokens);
        }
        tokens.push(next_token);

        // Continue generation without injection (standard KV-cache)
        for _ in 1..max_tokens {
            let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let logits = self.forward_with_kv_cache(&input, &mut kv_cache)?;
            next_token = sample_token(&logits, temperature)?;
            if stop_tokens.contains(&next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    // -----------------------------------------------------------------------
    // Logit Lens
    // -----------------------------------------------------------------------

    /// Project intermediate activation to vocabulary (no final softcap).
    pub fn logit_lens(&self, activation: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward(activation)?;
        self.project_to_vocab(&normed)
    }

    /// Project hidden states to vocabulary logits (tied embeddings, no softcap).
    pub fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        Ok(hidden.matmul(&self.embed_tokens.embeddings().t()?)?)
    }

    /// Top-K token predictions from intermediate activation.
    pub fn logit_lens_top_k(&self, activation: &Tensor, k: usize) -> Result<Vec<(u32, f32)>> {
        let logits = self.logit_lens(activation)?;
        let logits_f32: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let mut indexed: Vec<(u32, f32)> = logits_f32
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u32, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(indexed.into_iter().take(k).collect())
    }
}

// ---------------------------------------------------------------------------
// PlipBackend trait implementation
// ---------------------------------------------------------------------------

impl PlipBackend for PlipGemma2 {
    fn n_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn d_model(&self) -> usize {
        self.config.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn n_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    fn forward_with_cache(&self, input_ids: &Tensor) -> Result<(Tensor, ActivationCache)> {
        self.forward_with_cache(input_ids)
    }

    fn forward_with_attention(&self, input_ids: &Tensor) -> Result<(Tensor, AttentionCache)> {
        self.forward_with_attention(input_ids)
    }

    fn forward_with_intervention(
        &self,
        input_ids: &Tensor,
        spec: &KnockoutSpec,
    ) -> Result<(Tensor, AttentionCache)> {
        self.forward_with_intervention(input_ids, spec)
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
        KVCache::new(self.config.num_hidden_layers)
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

    fn forward_with_clt_injection(
        &self,
        input_ids: &Tensor,
        clt_spec: &CltInjectionSpec,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        PlipGemma2::forward_with_clt_injection(self, input_ids, clt_spec, kv_cache)
    }

    fn generate_with_clt_injection(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        clt_spec: &CltInjectionSpec,
        device: &Device,
    ) -> Result<Vec<u32>> {
        PlipGemma2::generate_with_clt_injection(
            self,
            prompt_ids,
            max_tokens,
            temperature,
            stop_tokens,
            clt_spec,
            device,
        )
    }

    fn forward_with_full_cache(&self, input_ids: &Tensor) -> Result<(Tensor, FullActivationCache)> {
        self.forward_with_full_cache(input_ids)
    }

    fn embedding_vector(&self, token_id: u32) -> Result<Tensor> {
        let emb = self.embed_tokens.embeddings(); // [vocab_size, d_model]
        Ok(emb.i(token_id as usize)?)
    }

    fn forward_with_steering(
        &self,
        input_ids: &Tensor,
        spec: &SteeringSpec,
    ) -> Result<(Tensor, AttentionCache)> {
        self.forward_with_steering(input_ids, spec)
    }

    fn generate_with_prompt_steering(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        spec: &SteeringSpec,
        device: &Device,
    ) -> Result<Vec<u32>> {
        self.generate_with_prompt_steering(
            prompt_ids,
            max_tokens,
            temperature,
            stop_tokens,
            spec,
            device,
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Inject a vector into the residual stream at a specific position.
///
/// hidden: `[batch, seq_len, d_model]`
/// vector: `[d_model]` (F32)
fn inject_at_position(hidden: &Tensor, vector: &Tensor, position: usize) -> Result<Tensor> {
    let (batch, seq_len, d_model) = hidden.dims3()?;
    let pos_slice = hidden.narrow(1, position, 1)?; // [batch, 1, d_model]
    let vec_expanded = vector
        .to_dtype(hidden.dtype())?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .expand((batch, 1, d_model))?;
    let pos_updated = (&pos_slice + &vec_expanded)?;

    // Reassemble: before + updated_position + after
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if position > 0 {
        parts.push(hidden.narrow(1, 0, position)?);
    }
    parts.push(pos_updated);
    if position + 1 < seq_len {
        parts.push(hidden.narrow(1, position + 1, seq_len - position - 1)?);
    }
    Ok(Tensor::cat(&parts, 1)?)
}

/// Sample a token from logits with temperature.
fn sample_token(logits: &Tensor, temperature: f32) -> Result<u32> {
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let logits_vec: Vec<f32> = logits_f32.flatten_all()?.to_vec1()?;

    if temperature <= 0.0 {
        // Greedy
        return Ok(argmax(&logits_vec));
    }

    // Temperature scaling + softmax sampling
    let max_logit = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let temp = f64::from(temperature);
    let probs: Vec<f64> = logits_vec
        .iter()
        .map(|&l| ((f64::from(l) - f64::from(max_logit)) / temp).exp())
        .collect();
    let sum: f64 = probs.iter().sum();
    let probs: Vec<f64> = probs.iter().map(|p| p / sum).collect();

    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= r {
            return Ok(i as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u32)
}
