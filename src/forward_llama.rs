//! LLaMA forward pass with per-layer activation capture
//!
//! Custom implementation that runs layer-by-layer to capture
//! intermediate activations for PLIP probing experiments.
//!
//! Based on the Code-LLaMA architecture from Meta.
//! Adapted from forward_qwen2.rs with simplifications:
//! - No bias on any projections (Q, K, V, O, MLP)
//! - Always separate lm_head (no tie_word_embeddings)
//! - Full MHA (num_key_value_heads == num_attention_heads for 7B)

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, RmsNorm, VarBuilder};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::Rng;
use tracing::info;

use crate::attention::AttentionCache;
use crate::cache::ActivationCache;
use crate::intervention::{KnockoutSpec, SteeringSpec};
use crate::kv_cache::KVCache;
use crate::masks::{create_causal_mask, create_generation_mask};
use crate::model::PlipBackend;

/// Model configuration (matches HuggingFace config.json for Code-LLaMA)
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_max_position_embeddings() -> usize {
    16384
}

/// Rotary Position Embeddings (RoPE)
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dim: usize,
        max_seq_len: usize,
        theta: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let inv_freq: Vec<f64> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / dim as f64))
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?.to_dtype(dtype)?;

        let positions: Vec<f64> = (0..max_seq_len).map(|i| i as f64).collect();
        let positions = Tensor::new(positions, device)?.to_dtype(dtype)?;

        // [seq_len, dim/2]
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.i(start_pos..start_pos + seq_len)?;
        let sin = self.sin.i(start_pos..start_pos + seq_len)?;

        let q_embed = apply_rotary_emb(q, &cos, &sin)?;
        let k_embed = apply_rotary_emb(k, &cos, &sin)?;

        Ok((q_embed, k_embed))
    }
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, seq_len, head_dim) = x.dims4()?;
    let x_reshape = x.reshape(((), seq_len, head_dim / 2, 2))?;
    let x0 = x_reshape.i((.., .., .., 0))?;
    let x1 = x_reshape.i((.., .., .., 1))?;

    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    let out0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let out1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;

    let out = Tensor::stack(&[&out0, &out1], D::Minus1)?;
    Ok(out.reshape(x.shape())?)
}

/// Multi-head attention (no bias on any projection)
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, config: &LlamaConfig) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        // LLaMA has NO bias on any projection (unlike Qwen2 which has bias on Q/K/V)
        let q_proj = linear_no_bias(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_no_bias(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_no_bias(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_no_bias(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
        })
    }

    /// Forward pass that also returns attention weights
    /// Returns: (output, attention_weights) where attention_weights is [batch, heads, seq, seq]
    fn forward_with_attn(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (b, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = rotary.apply(&q, &k, start_pos)?;

        // Expand KV heads for grouped query attention
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        // Ensure tensors are contiguous for matmul
        // (needed when n_rep=1 in repeat_kv, since transpose leaves non-contiguous layout)
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;

        // Causal mask
        let mask = create_causal_mask(seq_len, x.device(), x.dtype())?;
        let attn_weights = attn_weights.broadcast_add(&mask)?;

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        Ok((self.o_proj.forward(&attn_output)?, attn_weights))
    }

    /// Forward pass with knockout mask applied (pre-softmax intervention)
    fn forward_with_intervention(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
        knockout_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (b, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = rotary.apply(&q, &k, start_pos)?;

        // Expand KV heads for grouped query attention
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        // Ensure tensors are contiguous for matmul
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;

        // Causal mask
        let mask = create_causal_mask(seq_len, x.device(), x.dtype())?;
        let attn_weights = attn_weights.broadcast_add(&mask)?;

        // Apply knockout mask if provided (INTERVENTION POINT)
        let attn_weights = if let Some(ko_mask) = knockout_mask {
            attn_weights.broadcast_add(ko_mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        Ok((self.o_proj.forward(&attn_output)?, attn_weights))
    }

    /// Forward pass with KV-cache for efficient generation
    fn forward_with_cache(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
        cache_k: &mut Option<Tensor>,
        cache_v: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = rotary.apply(&q, &k, start_pos)?;

        // Concatenate with cached K, V if available
        let (k, v) = if let (Some(prev_k), Some(prev_v)) = (cache_k.as_ref(), cache_v.as_ref()) {
            let k = Tensor::cat(&[prev_k, &k], 2)?;
            let v = Tensor::cat(&[prev_v, &v], 2)?;
            (k, v)
        } else {
            (k, v)
        };

        // Update cache
        *cache_k = Some(k.clone());
        *cache_v = Some(v.clone());

        // Expand KV heads for grouped query attention
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        // Ensure tensors are contiguous for matmul
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // Total sequence length (cached + new)
        let total_seq_len = k.dim(2)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;

        let mask =
            create_generation_mask(seq_len, total_seq_len, start_pos, x.device(), x.dtype())?;
        let attn_weights = attn_weights.broadcast_add(&mask)?;

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        Ok(self.o_proj.forward(&attn_output)?)
    }

    /// Forward pass with steering intervention applied (post-softmax)
    fn forward_with_steering(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
        steering_spec: &crate::intervention::SteeringSpec,
    ) -> Result<(Tensor, Tensor)> {
        use crate::intervention::apply_steering;

        let (b, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = rotary.apply(&q, &k, start_pos)?;

        // Expand KV heads for grouped query attention
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        // Ensure tensors are contiguous for matmul
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;

        // Causal mask
        let mask = create_causal_mask(seq_len, x.device(), x.dtype())?;
        let attn_weights = attn_weights.broadcast_add(&mask)?;

        // Apply softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // INTERVENTION POINT B: Post-softmax steering
        let attn_weights = if steering_spec.is_steering() {
            apply_steering(&attn_weights, steering_spec, self.num_heads, seq_len)?
        } else {
            attn_weights
        };

        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        Ok((self.o_proj.forward(&attn_output)?, attn_weights))
    }

    /// Forward pass with steering AND KV-cache
    fn forward_with_cache_and_steering(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
        steering_spec: &crate::intervention::SteeringSpec,
        cache_k: &mut Option<Tensor>,
        cache_v: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        use crate::intervention::apply_steering;

        let (b, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = rotary.apply(&q, &k, start_pos)?;

        // Concatenate with cached K, V if available
        let (k, v) = if let (Some(prev_k), Some(prev_v)) = (cache_k.as_ref(), cache_v.as_ref()) {
            let k = Tensor::cat(&[prev_k, &k], 2)?;
            let v = Tensor::cat(&[prev_v, &v], 2)?;
            (k, v)
        } else {
            (k, v)
        };

        // Update cache (before GQA expansion)
        *cache_k = Some(k.clone());
        *cache_v = Some(v.clone());

        // Expand KV heads for grouped query attention
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        // Ensure tensors are contiguous for matmul
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // Total sequence length
        let total_seq_len = k.dim(2)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;

        let mask =
            create_generation_mask(seq_len, total_seq_len, start_pos, x.device(), x.dtype())?;
        let attn_weights = attn_weights.broadcast_add(&mask)?;

        // Apply softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Apply steering (post-softmax)
        let attn_weights = if steering_spec.is_steering() {
            apply_steering(&attn_weights, steering_spec, self.num_heads, total_seq_len)?
        } else {
            attn_weights
        };

        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        Ok(self.o_proj.forward(&attn_output)?)
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    let x = x.unsqueeze(2)?;
    let x = x.expand((b, num_kv_heads, n_rep, seq_len, head_dim))?;
    Ok(x.reshape((b, num_kv_heads * n_rep, seq_len, head_dim))?)
}

/// MLP block (LLaMA style - SwiGLU, no bias)
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    fn load(vb: VarBuilder, config: &LlamaConfig) -> Result<Self> {
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
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        Ok(self.down_proj.forward(&hidden)?)
    }
}

/// Single decoder layer
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn load(vb: VarBuilder, config: &LlamaConfig) -> Result<Self> {
        let self_attn = Attention::load(vb.pp("self_attn"), config)?;
        let mlp = MLP::load(vb.pp("mlp"), config)?;
        let input_layernorm = candle_nn::rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = candle_nn::rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &RotaryEmbedding, start_pos: usize) -> Result<Tensor> {
        let (output, _) = self.forward_with_attn(x, rotary, start_pos)?;
        Ok(output)
    }

    fn forward_with_attn(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let (x, attn_weights) = self.self_attn.forward_with_attn(&x, rotary, start_pos)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok(((residual + x)?, attn_weights))
    }

    fn forward_with_intervention(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
        knockout_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let (x, attn_weights) =
            self.self_attn
                .forward_with_intervention(&x, rotary, start_pos, knockout_mask)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok(((residual + x)?, attn_weights))
    }

    fn forward_with_steering(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
        steering_spec: &crate::intervention::SteeringSpec,
    ) -> Result<(Tensor, Tensor)> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let (x, attn_weights) =
            self.self_attn
                .forward_with_steering(&x, rotary, start_pos, steering_spec)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok(((residual + x)?, attn_weights))
    }

    fn forward_with_cache(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
        cache_k: &mut Option<Tensor>,
        cache_v: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self
            .self_attn
            .forward_with_cache(&x, rotary, start_pos, cache_k, cache_v)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok((residual + x)?)
    }

    fn forward_with_cache_and_steering(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        start_pos: usize,
        steering_spec: &crate::intervention::SteeringSpec,
        cache_k: &mut Option<Tensor>,
        cache_v: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward_with_cache_and_steering(
            &x,
            rotary,
            start_pos,
            steering_spec,
            cache_k,
            cache_v,
        )?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok((residual + x)?)
    }
}

/// Safetensors index for sharded models
#[derive(Debug, serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

/// Custom LLaMA model with per-layer activation capture
pub struct PlipLlama {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear, // Always separate (Code-LLaMA never ties embeddings)
    rotary: RotaryEmbedding,
    n_layers: usize,
    n_heads: usize,
    hidden_size: usize,
    vocab_size: usize,
}

impl PlipLlama {
    /// Load model from HuggingFace
    pub fn load(model_id: &str, device: &Device, dtype: DType) -> Result<Self> {
        info!("Loading LLaMA from: {}", model_id);

        // Download model files
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let config_path = repo
            .get("config.json")
            .context("Failed to download config.json")?;

        // Load config
        let config_str = std::fs::read_to_string(&config_path).context("Failed to read config")?;
        let config: LlamaConfig = serde_json::from_str(&config_str)?;

        info!(
            "Model config: {} layers, {} hidden, {} vocab",
            config.num_hidden_layers, config.hidden_size, config.vocab_size
        );

        // Check for sharded vs single safetensors
        let weights_paths = if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            // Sharded model - parse index and download all shards
            info!("Model is sharded, loading index...");
            let index_str = std::fs::read_to_string(&index_path).context("Failed to read index")?;
            let index: SafetensorsIndex = serde_json::from_str(&index_str)?;

            // Get unique shard filenames
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
            // Single file model
            let path = repo
                .get("model.safetensors")
                .context("Failed to download model.safetensors")?;
            vec![path]
        };

        info!("Loading weights from {} file(s)...", weights_paths.len());

        // Load weights
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
            let layer = DecoderLayer::load(vb_model.pp(format!("layers.{i}")), &config)?;
            layers.push(layer);
        }

        let norm =
            candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, vb_model.pp("norm"))?;

        // LLaMA always has a separate lm_head
        info!("Loading separate lm_head...");
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
            dtype,
        )?;

        info!(
            "Model loaded successfully with {} layers (vocab_size: {})",
            config.num_hidden_layers, config.vocab_size
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            n_layers: config.num_hidden_layers,
            n_heads: config.num_attention_heads,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
        })
    }

    /// Forward pass with activation capture at each layer
    pub fn forward_with_cache(&self, input_ids: &Tensor) -> Result<(Tensor, ActivationCache)> {
        let mut cache = ActivationCache::with_capacity(self.n_layers);

        let mut hidden = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.rotary, 0)?;

            let seq_len = hidden.dim(1)?;
            let last_token = hidden.i((.., seq_len - 1, ..))?.squeeze(1)?;
            cache.push(last_token);

            if (i + 1) % 10 == 0 {
                info!("Processed layer {}/{}", i + 1, self.n_layers);
            }
        }

        let output = self.norm.forward(&hidden)?;
        Ok((output, cache))
    }

    /// Forward pass that captures attention weights from all layers
    pub fn forward_with_attention(&self, input_ids: &Tensor) -> Result<(Tensor, AttentionCache)> {
        let mut attn_cache = AttentionCache::with_capacity(self.n_layers);

        let mut hidden = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let (new_hidden, attn_weights) = layer.forward_with_attn(&hidden, &self.rotary, 0)?;
            hidden = new_hidden;
            attn_cache.push(attn_weights);

            if (i + 1) % 10 == 0 {
                info!(
                    "Processed layer {}/{} (with attention)",
                    i + 1,
                    self.n_layers
                );
            }
        }

        let output = self.norm.forward(&hidden)?;
        Ok((output, attn_cache))
    }

    /// Forward pass with attention knockout intervention
    pub fn forward_with_intervention(
        &self,
        input_ids: &Tensor,
        spec: &crate::intervention::KnockoutSpec,
    ) -> Result<(Tensor, AttentionCache)> {
        use crate::intervention::create_knockout_mask;

        let mut attn_cache = AttentionCache::with_capacity(self.n_layers);
        let seq_len = input_ids.dim(1)?;

        let mut hidden = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let knockout_mask = if spec.applies_to_layer(i) {
                Some(create_knockout_mask(
                    spec,
                    self.n_heads(),
                    seq_len,
                    input_ids.device(),
                    hidden.dtype(),
                )?)
            } else {
                None
            };

            let (new_hidden, attn_weights) = layer.forward_with_intervention(
                &hidden,
                &self.rotary,
                0,
                knockout_mask.as_ref(),
            )?;
            hidden = new_hidden;
            attn_cache.push(attn_weights);

            if (i + 1) % 10 == 0 {
                info!(
                    "Processed layer {}/{} (with intervention)",
                    i + 1,
                    self.n_layers
                );
            }
        }

        let output = self.norm.forward(&hidden)?;
        Ok((output, attn_cache))
    }

    /// Forward pass with attention steering intervention (post-softmax)
    pub fn forward_with_steering(
        &self,
        input_ids: &Tensor,
        spec: &crate::intervention::SteeringSpec,
    ) -> Result<(Tensor, AttentionCache)> {
        let mut attn_cache = AttentionCache::with_capacity(self.n_layers);

        let mut hidden = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let (new_hidden, attn_weights) = if spec.applies_to_layer(i) {
                layer.forward_with_steering(&hidden, &self.rotary, 0, spec)?
            } else {
                layer.forward_with_attn(&hidden, &self.rotary, 0)?
            };

            hidden = new_hidden;
            attn_cache.push(attn_weights);

            if (i + 1) % 10 == 0 {
                info!(
                    "Processed layer {}/{} (with steering)",
                    i + 1,
                    self.n_layers
                );
            }
        }

        let output = self.norm.forward(&hidden)?;
        Ok((output, attn_cache))
    }

    /// Number of layers
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Hidden dimension
    pub fn d_model(&self) -> usize {
        self.hidden_size
    }

    /// Vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Number of attention heads
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Apply logit lens: project intermediate activation through final norm + lm_head
    pub fn logit_lens(&self, activation: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward(activation)?;
        self.project_to_vocab(&normed)
    }

    /// Project hidden state directly to vocabulary logits (no normalization)
    pub fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        Ok(self.lm_head.forward(hidden)?)
    }

    /// Get top-k token predictions from logits
    pub fn top_k_from_logits(&self, logits: &Tensor, k: usize) -> Result<Vec<(u32, f32)>> {
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits_f32)?;
        let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;

        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k: Vec<(u32, f32)> = indexed
            .into_iter()
            .take(k)
            .map(|(idx, prob)| (idx as u32, prob))
            .collect();

        Ok(top_k)
    }

    /// Apply logit lens and return top-k predictions
    pub fn logit_lens_top_k(&self, activation: &Tensor, k: usize) -> Result<Vec<(u32, f32)>> {
        let logits = self.logit_lens(activation)?;
        self.top_k_from_logits(&logits, k)
    }

    /// Create a new KV-cache for this model
    pub fn new_kv_cache(&self) -> KVCache {
        KVCache::new(self.n_layers)
    }

    /// Forward pass with KV-cache for efficient generation
    pub fn forward_with_kv_cache(
        &self,
        input_ids: &Tensor,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let start_pos = kv_cache.seq_len();

        let mut hidden = self.embed_tokens.forward(input_ids)?;

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

        let seq_len = output.dim(1)?;
        let last_hidden = output.i((.., seq_len - 1, ..))?.squeeze(1)?;

        Ok(self.lm_head.forward(&last_hidden)?)
    }

    /// Generate tokens autoregressively with KV-cache
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

        let prompt_tensor = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let logits = self.forward_with_kv_cache(&prompt_tensor, &mut kv_cache)?;

        let mut next_token = sample_from_logits(&logits, temperature)?;

        if stop_tokens.contains(&next_token) {
            return Ok(tokens);
        }
        tokens.push(next_token);

        for _ in 1..max_tokens {
            let input_tensor = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let logits = self.forward_with_kv_cache(&input_tensor, &mut kv_cache)?;

            next_token = sample_from_logits(&logits, temperature)?;

            if stop_tokens.contains(&next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Forward pass with steering AND KV-cache for prompt-steered generation
    pub fn forward_with_kv_cache_and_steering(
        &self,
        input_ids: &Tensor,
        kv_cache: &mut KVCache,
        steering_spec: &crate::intervention::SteeringSpec,
    ) -> Result<Tensor> {
        let start_pos = kv_cache.seq_len();

        let mut hidden = self.embed_tokens.forward(input_ids)?;

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

        let seq_len = output.dim(1)?;
        let last_hidden = output.i((.., seq_len - 1, ..))?.squeeze(1)?;

        Ok(self.lm_head.forward(&last_hidden)?)
    }

    /// Generate with steering applied to prompt, then efficient KV-cache generation
    pub fn generate_with_prompt_steering(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        steering_spec: &crate::intervention::SteeringSpec,
        device: &Device,
    ) -> Result<Vec<u32>> {
        let mut kv_cache = self.new_kv_cache();
        let mut tokens = prompt_ids.to_vec();

        let prompt_tensor = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let logits =
            self.forward_with_kv_cache_and_steering(&prompt_tensor, &mut kv_cache, steering_spec)?;

        let mut next_token = sample_from_logits(&logits, temperature)?;

        if stop_tokens.contains(&next_token) {
            return Ok(tokens);
        }
        tokens.push(next_token);

        for _ in 1..max_tokens {
            let input_tensor = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let logits = self.forward_with_kv_cache(&input_tensor, &mut kv_cache)?;

            next_token = sample_from_logits(&logits, temperature)?;

            if stop_tokens.contains(&next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }
}

impl PlipBackend for PlipLlama {
    fn n_layers(&self) -> usize { self.n_layers() }
    fn d_model(&self) -> usize { self.d_model() }
    fn vocab_size(&self) -> usize { self.vocab_size() }
    fn n_heads(&self) -> usize { self.n_heads() }

    fn forward_with_cache(&self, input_ids: &Tensor) -> Result<(Tensor, ActivationCache)> {
        self.forward_with_cache(input_ids)
    }
    fn forward_with_attention(&self, input_ids: &Tensor) -> Result<(Tensor, AttentionCache)> {
        self.forward_with_attention(input_ids)
    }
    fn forward_with_intervention(&self, input_ids: &Tensor, spec: &KnockoutSpec) -> Result<(Tensor, AttentionCache)> {
        self.forward_with_intervention(input_ids, spec)
    }

    fn logit_lens(&self, activation: &Tensor) -> Result<Tensor> { self.logit_lens(activation) }
    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> { self.project_to_vocab(hidden) }
    fn logit_lens_top_k(&self, activation: &Tensor, k: usize) -> Result<Vec<(u32, f32)>> {
        self.logit_lens_top_k(activation, k)
    }

    fn new_kv_cache(&self) -> KVCache { self.new_kv_cache() }
    fn forward_with_kv_cache(&self, input_ids: &Tensor, kv_cache: &mut KVCache) -> Result<Tensor> {
        self.forward_with_kv_cache(input_ids, kv_cache)
    }
    fn generate(&self, prompt_ids: &[u32], max_tokens: usize, temperature: f32, stop_tokens: &[u32], device: &Device) -> Result<Vec<u32>> {
        self.generate(prompt_ids, max_tokens, temperature, stop_tokens, device)
    }

    fn forward_with_steering(&self, input_ids: &Tensor, spec: &SteeringSpec) -> Result<(Tensor, AttentionCache)> {
        self.forward_with_steering(input_ids, spec)
    }
    fn generate_with_prompt_steering(&self, prompt_ids: &[u32], max_tokens: usize, temperature: f32, stop_tokens: &[u32], spec: &SteeringSpec, device: &Device) -> Result<Vec<u32>> {
        self.generate_with_prompt_steering(prompt_ids, max_tokens, temperature, stop_tokens, spec, device)
    }

    fn chat_template(&self, _prompt: &str, _system_prompt: Option<&str>) -> Option<String> {
        // Code-LLaMA is a base model — no chat template
        None
    }
}

/// Sample a token from logits with temperature
fn sample_from_logits(logits: &Tensor, temperature: f32) -> Result<u32> {
    let logits_f32 = logits.to_dtype(DType::F32)?.flatten_all()?;
    let logits_vec: Vec<f32> = logits_f32.to_vec1()?;

    if temperature <= 0.0 {
        // Greedy: argmax
        let (max_idx, _) = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| anyhow::anyhow!("Empty logits"))?;
        return Ok(max_idx as u32);
    }

    // Temperature sampling
    let scaled: Vec<f32> = logits_vec.iter().map(|x| x / temperature).collect();
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|x| x / sum).collect();

    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;

    for (idx, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return Ok(idx as u32);
        }
    }

    Ok((probs.len() - 1) as u32)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_placeholder() {
        // GPU-dependent tests are in tests/integration.rs
        // This placeholder ensures the test module compiles
    }
}
