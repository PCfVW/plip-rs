//! PlipModel wrapper for activation extraction
//!
//! Supports multiple model backends (StarCoder2, Qwen2) with a unified interface.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use tracing::info;

use crate::attention::{AttentionAnalysis, AttentionCache};
use crate::cache::ActivationCache;
use crate::forward::PlipStarCoder2;
use crate::forward_gemma::PlipGemma;
use crate::forward_llama::PlipLlama;
use crate::forward_phi3::PlipPhi3;
use crate::forward_qwen2::PlipQwen2;
use crate::intervention::{
    measure_attention_to_targets, AblationResult, KnockoutSpec, SteeringResult, SteeringSpec,
};
use crate::kv_cache::KVCache;
use crate::logit_lens::{decode_predictions, LogitLensAnalysis, LogitLensResult};
use crate::positioning::EncodingWithOffsets;

/// Supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// StarCoder2 (BigCode)
    StarCoder2,
    /// Qwen2 / Qwen2.5 (Alibaba)
    Qwen2,
    /// Gemma / CodeGemma (Google)
    Gemma,
    /// LLaMA / Code-LLaMA (Meta)
    Llama,
    /// Phi-3 (Microsoft)
    Phi3,
}

impl ModelArchitecture {
    /// Detect architecture from model ID
    pub fn from_model_id(model_id: &str) -> Self {
        let model_lower = model_id.to_lowercase();
        if model_lower.contains("qwen") {
            ModelArchitecture::Qwen2
        } else if model_lower.contains("starcoder") || model_lower.contains("bigcode") {
            ModelArchitecture::StarCoder2
        } else if model_lower.contains("gemma") || model_lower.contains("codegemma") {
            ModelArchitecture::Gemma
        } else if model_lower.contains("llama") || model_lower.contains("codellama") {
            ModelArchitecture::Llama
        } else if model_lower.contains("phi") {
            ModelArchitecture::Phi3
        } else {
            // Default to Qwen2 for unknown models (more likely to work)
            info!(
                "Unknown model architecture for '{}', defaulting to Qwen2",
                model_id
            );
            ModelArchitecture::Qwen2
        }
    }

    /// Check if this architecture typically uses instruct/chat format
    pub fn is_instruct_model(model_id: &str) -> bool {
        let model_lower = model_id.to_lowercase();
        model_lower.contains("instruct") || model_lower.contains("chat")
    }
}

/// Unified backend trait for all model architectures.
///
/// Implementing this trait is the only requirement for adding a new model to PLIP-RS.
/// Optional capabilities (steering, chat template) have default implementations that
/// return errors or `None`, so non-transformer architectures can skip them.
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
        &self,
        input_ids: &Tensor,
        spec: &KnockoutSpec,
    ) -> Result<(Tensor, AttentionCache)>;

    // --- Logit lens ---
    fn logit_lens(&self, activation: &Tensor) -> Result<Tensor>;
    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor>;
    fn logit_lens_top_k(&self, activation: &Tensor, k: usize) -> Result<Vec<(u32, f32)>>;

    // --- Generation (KV-cache) ---
    fn new_kv_cache(&self) -> KVCache;
    fn forward_with_kv_cache(
        &self,
        input_ids: &Tensor,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor>;
    fn generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        device: &Device,
    ) -> Result<Vec<u32>>;

    // --- Optional capabilities (default: unsupported) ---

    fn forward_with_steering(
        &self,
        _input_ids: &Tensor,
        _spec: &SteeringSpec,
    ) -> Result<(Tensor, AttentionCache)> {
        anyhow::bail!("Steering not supported for this architecture")
    }

    fn generate_with_prompt_steering(
        &self,
        _prompt_ids: &[u32],
        _max_tokens: usize,
        _temperature: f32,
        _stop_tokens: &[u32],
        _spec: &SteeringSpec,
        _device: &Device,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("Prompt steering not supported for this architecture")
    }

    fn chat_template(&self, _prompt: &str, _system_prompt: Option<&str>) -> Option<String> {
        None
    }
}

/// High-level model wrapper for PLIP experiments
///
/// Supports multiple model architectures with a unified interface.
pub struct PlipModel {
    model: Box<dyn PlipBackend>,
    tokenizer: Tokenizer,
    device: Device,
    architecture: ModelArchitecture,
    model_id: String,
}

impl PlipModel {
    /// Load a model from HuggingFace (tries CUDA, falls back to CPU)
    ///
    /// Automatically detects model architecture from the model ID.
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        Self::from_pretrained_with_device(model_id, None)
    }

    /// Load with explicit device choice (None = auto-detect)
    ///
    /// Automatically detects model architecture from the model ID.
    pub fn from_pretrained_with_device(model_id: &str, force_cpu: Option<bool>) -> Result<Self> {
        let architecture = ModelArchitecture::from_model_id(model_id);
        Self::from_pretrained_with_arch(model_id, force_cpu, architecture)
    }

    /// Load with explicit architecture specification
    pub fn from_pretrained_with_arch(
        model_id: &str,
        force_cpu: Option<bool>,
        architecture: ModelArchitecture,
    ) -> Result<Self> {
        let (device, dtype) = if force_cpu == Some(true) {
            info!("Forcing CPU mode");
            (Device::Cpu, DType::F32)
        } else {
            match Device::cuda_if_available(0) {
                Ok(dev) if dev.is_cuda() => {
                    info!("Using CUDA device");
                    // Use BF16 for Qwen models (they're trained in bfloat16)
                    // F16 causes garbage output due to range/precision mismatch
                    (dev, DType::BF16)
                }
                _ => {
                    info!("CUDA not available, using CPU");
                    (Device::Cpu, DType::F32)
                }
            }
        };

        info!("Loading model: {}", model_id);
        info!("Architecture: {:?}", architecture);
        info!("Device: {:?}", device);
        info!("Dtype: {:?}", dtype);

        // Load tokenizer from HuggingFace
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {e}"))?;

        // Load model based on architecture
        let model: Box<dyn PlipBackend> = match architecture {
            ModelArchitecture::StarCoder2 => {
                Box::new(PlipStarCoder2::load(model_id, &device, dtype)?)
            }
            ModelArchitecture::Qwen2 => {
                Box::new(PlipQwen2::load(model_id, &device, dtype)?)
            }
            ModelArchitecture::Gemma => {
                Box::new(PlipGemma::load(model_id, &device, dtype)?)
            }
            ModelArchitecture::Llama => {
                Box::new(PlipLlama::load(model_id, &device, dtype)?)
            }
            ModelArchitecture::Phi3 => {
                Box::new(PlipPhi3::load(model_id, &device, dtype)?)
            }
        };

        Ok(Self {
            model,
            tokenizer,
            device,
            architecture,
            model_id: model_id.to_string(),
        })
    }

    /// Get the model architecture
    pub fn architecture(&self) -> ModelArchitecture {
        self.architecture
    }

    /// Get the model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Check if this is an instruct/chat model
    pub fn is_instruct_model(&self) -> bool {
        ModelArchitecture::is_instruct_model(&self.model_id)
    }

    /// Apply chat template formatting for instruct models
    ///
    /// For Qwen Instruct models, wraps the prompt in the chat template format:
    /// ```text
    /// <|im_start|>system
    /// You are a helpful assistant.<|im_end|>
    /// <|im_start|>user
    /// {prompt}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    ///
    /// For non-instruct models, returns the prompt unchanged.
    ///
    /// # Arguments
    /// * `prompt` - The user's prompt/request
    /// * `system_prompt` - Optional system prompt (defaults to code assistant)
    pub fn apply_chat_template(&self, prompt: &str, system_prompt: Option<&str>) -> String {
        if !self.is_instruct_model() {
            return prompt.to_string();
        }

        self.model
            .chat_template(prompt, system_prompt)
            .unwrap_or_else(|| prompt.to_string())
    }

    /// Get the EOS token ID for this model
    ///
    /// Returns the appropriate end-of-sequence token for generation stopping.
    pub fn eos_token_id(&self) -> Option<u32> {
        // Try to get from tokenizer's special tokens
        self.tokenizer
            .get_vocab(true)
            .get("<|im_end|>")
            .copied()
            .or_else(|| self.tokenizer.get_vocab(true).get("<|endoftext|>").copied())
            .or_else(|| self.tokenizer.get_vocab(true).get("</s>").copied())
            .or_else(|| self.tokenizer.get_vocab(true).get("<end_of_turn>").copied())
    }

    /// Get activations for a text input
    pub fn get_activations(&self, text: &str) -> Result<ActivationCache> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let input_tensor = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;

        // Forward pass with cache
        let (_, cache) = self.model.forward_with_cache(&input_tensor)?;

        Ok(cache)
    }

    /// Number of layers in the model
    pub fn n_layers(&self) -> usize {
        self.model.n_layers()
    }

    /// Hidden dimension of the model
    pub fn d_model(&self) -> usize {
        self.model.d_model()
    }

    /// Vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    /// Number of attention heads
    pub fn n_heads(&self) -> usize {
        self.model.n_heads()
    }

    /// Run logit lens analysis on text
    ///
    /// Returns predictions at each layer showing what the model would
    /// predict if it stopped processing at that layer.
    pub fn logit_lens(&self, text: &str, top_k: usize) -> Result<LogitLensAnalysis> {
        // Get activations for all layers
        let cache = self.get_activations(text)?;

        let mut analysis = LogitLensAnalysis::new(text.to_string(), cache.n_layers());

        // Apply logit lens at each layer
        for layer in 0..cache.n_layers() {
            if let Some(activation) = cache.get_layer(layer) {
                let predictions = self.model.logit_lens_top_k(activation, top_k)?;
                let decoded = decode_predictions(&predictions, &self.tokenizer);

                analysis.push(LogitLensResult {
                    layer,
                    predictions: decoded,
                });
            }
        }

        Ok(analysis)
    }

    /// Apply logit lens to a single activation tensor
    pub fn logit_lens_activation(
        &self,
        activation: &Tensor,
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let predictions = self.model.logit_lens_top_k(activation, top_k)?;
        let decoded = decode_predictions(&predictions, &self.tokenizer);
        Ok(decoded
            .into_iter()
            .map(|p| (p.token, p.probability))
            .collect())
    }

    /// Decode a token ID to string
    pub fn decode_token(&self, token_id: u32) -> String {
        self.tokenizer
            .decode(&[token_id], false)
            .unwrap_or_else(|_| format!("<{token_id}>"))
    }

    /// Tokenize text and return token strings
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        Ok(encoding
            .get_ids()
            .iter()
            .map(|&id| self.decode_token(id))
            .collect())
    }

    /// Tokenize text and return tokens with character offsets
    ///
    /// This enables model-agnostic position handling by providing the
    /// character offset mapping for each token. Use this with the
    /// `positioning` module to convert character positions to token indices.
    ///
    /// # Example
    /// ```ignore
    /// let encoding = model.tokenize_with_offsets("def add(a, b):")?;
    ///
    /// // Find which token contains character position 8 (parameter 'a')
    /// let token_idx = encoding.char_to_token(8);
    /// ```
    pub fn tokenize_with_offsets(&self, text: &str) -> Result<EncodingWithOffsets> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        let ids = encoding.get_ids().to_vec();
        let tokens: Vec<String> = ids.iter().map(|&id| self.decode_token(id)).collect();
        let offsets = encoding.get_offsets().to_vec();

        Ok(EncodingWithOffsets::new(ids, tokens, offsets))
    }

    /// Convert character position to token index for the given text
    ///
    /// This is a convenience method that tokenizes the text and then
    /// finds the token containing the given character position.
    pub fn char_to_token(&self, text: &str, char_pos: usize) -> Result<Option<usize>> {
        let encoding = self.tokenize_with_offsets(text)?;
        Ok(encoding.char_to_token(char_pos))
    }

    /// Convert multiple character positions to token indices
    ///
    /// This is more efficient than calling `char_to_token` multiple times
    /// as it only tokenizes once.
    pub fn chars_to_tokens(
        &self,
        text: &str,
        char_positions: &[usize],
    ) -> Result<Vec<Option<usize>>> {
        let encoding = self.tokenize_with_offsets(text)?;
        Ok(char_positions
            .iter()
            .map(|&pos| encoding.char_to_token(pos))
            .collect())
    }

    /// Get attention patterns for a text input
    pub fn get_attention(&self, text: &str) -> Result<AttentionCache> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let input_tensor = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;

        // Forward pass with attention capture
        let (_, attn_cache) = self.model.forward_with_attention(&input_tensor)?;

        Ok(attn_cache)
    }

    /// Forward pass returning logits and attention cache
    ///
    /// Simple forward pass without any intervention, returning both
    /// the output logits and the attention patterns.
    pub fn forward(&self, text: &str) -> Result<(Tensor, AttentionCache)> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let input_tensor = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;

        // Forward pass with attention capture
        self.model.forward_with_attention(&input_tensor)
    }

    /// Run attention pattern analysis on text
    ///
    /// Returns analysis with tokens and attention patterns for each layer
    pub fn analyze_attention(&self, text: &str) -> Result<AttentionAnalysis> {
        let tokens = self.tokenize(text)?;
        let cache = self.get_attention(text)?;
        let n_layers = self.n_layers();
        let n_heads = self.n_heads();

        Ok(AttentionAnalysis::new(tokens, cache, n_layers, n_heads))
    }

    /// Forward pass with attention knockout intervention
    ///
    /// Runs both baseline and intervened forward passes, returning
    /// an `AblationResult` for comparison.
    ///
    /// # Arguments
    /// * `text` - Input text to process
    /// * `spec` - Knockout specification defining which edges to remove
    ///
    /// # Returns
    /// `AblationResult` containing baseline and ablated logits
    ///
    /// # Example
    /// ```ignore
    /// use plip_rs::{PlipModel, KnockoutSpec};
    ///
    /// let model = PlipModel::from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")?;
    ///
    /// // Knockout attention from position 5 to positions 0-3 in layer 10
    /// let spec = KnockoutSpec::new()
    ///     .layer(10)
    ///     .from_to_positions(5, &[0, 1, 2, 3]);
    ///
    /// let result = model.forward_with_intervention("def add(a, b):", &spec)?;
    /// println!("KL divergence: {}", result.kl_divergence()?);
    /// ```
    pub fn forward_with_intervention(
        &self,
        text: &str,
        spec: &KnockoutSpec,
    ) -> Result<AblationResult> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq_len = input_ids.len();
        let input_tensor = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;

        // Validate spec
        spec.validate(self.n_layers(), self.n_heads(), seq_len)?;

        // Run baseline (no intervention)
        let (baseline_output, _) = self.model.forward_with_attention(&input_tensor)?;
        let baseline_logits = self.compute_logits(&baseline_output)?;

        // Run intervened
        let (ablated_output, ablated_attn) =
            self.model.forward_with_intervention(&input_tensor, spec)?;
        let ablated_logits = self.compute_logits(&ablated_output)?;

        Ok(AblationResult::new(
            baseline_logits,
            ablated_logits,
            spec.clone(),
            Some(ablated_attn),
        ))
    }

    /// Forward pass with intervention, returning only ablated results (no baseline)
    ///
    /// Use this when you want to run baseline separately or don't need comparison.
    pub fn forward_ablated_only(
        &self,
        text: &str,
        spec: &KnockoutSpec,
    ) -> Result<(Tensor, AttentionCache)> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq_len = input_ids.len();
        let input_tensor = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;

        spec.validate(self.n_layers(), self.n_heads(), seq_len)?;

        let (output, attn_cache) = self.model.forward_with_intervention(&input_tensor, spec)?;
        let logits = self.compute_logits(&output)?;

        Ok((logits, attn_cache))
    }

    /// Forward pass with attention steering intervention
    ///
    /// Runs both baseline and steered forward passes, returning
    /// a `SteeringResult` for comparison.
    ///
    /// # Arguments
    /// * `text` - Input text to process
    /// * `spec` - Steering specification defining which edges to modify
    ///
    /// # Returns
    /// `SteeringResult` containing baseline and steered logits
    ///
    /// # Example
    /// ```ignore
    /// use plip_rs::{PlipModel, SteeringSpec};
    ///
    /// let model = PlipModel::from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")?;
    ///
    /// // Boost attention from marker to function tokens by 3x
    /// let spec = SteeringSpec::scale(3.0)
    ///     .layer(16)
    ///     .from_to_positions(marker_pos, &function_positions);
    ///
    /// let result = model.forward_with_steering("...", &spec)?;
    /// println!("KL divergence: {}", result.kl_divergence()?);
    /// ```
    pub fn forward_with_steering(
        &self,
        text: &str,
        spec: &SteeringSpec,
    ) -> Result<SteeringResult> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq_len = input_ids.len();
        let input_tensor = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;

        // Validate spec
        spec.validate(self.n_layers(), self.n_heads(), seq_len)?;

        // Run baseline (no intervention)
        let (baseline_output, _) = self.model.forward_with_attention(&input_tensor)?;
        let baseline_logits = self.compute_logits(&baseline_output)?;

        // Run steered
        let (steered_output, steered_attn) =
            self.model.forward_with_steering(&input_tensor, spec)?;
        let steered_logits = self.compute_logits(&steered_output)?;

        Ok(SteeringResult::new(
            baseline_logits,
            steered_logits,
            spec.clone(),
            Some(steered_attn),
        ))
    }

    /// Forward pass with steering, returning only steered results (no baseline)
    ///
    /// Use this when you want to run baseline separately or don't need comparison.
    pub fn forward_steered_only(
        &self,
        text: &str,
        spec: &SteeringSpec,
    ) -> Result<(Tensor, AttentionCache)> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq_len = input_ids.len();
        let input_tensor = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;

        spec.validate(self.n_layers(), self.n_heads(), seq_len)?;

        let (output, attn_cache) = self.model.forward_with_steering(&input_tensor, spec)?;
        let logits = self.compute_logits(&output)?;

        Ok((logits, attn_cache))
    }

    /// Measure attention from a source position to target positions
    ///
    /// Returns the mean attention weight across all heads for the specified edges.
    pub fn measure_marker_attention(
        &self,
        text: &str,
        from_pos: usize,
        to_positions: &[usize],
        layer: usize,
    ) -> Result<f32> {
        let attn_cache = self.get_attention(text)?;
        measure_attention_to_targets(&attn_cache, from_pos, to_positions, layer)
    }

    /// Compute logits from final hidden state (last token position)
    ///
    /// NOTE: This method assumes `hidden` is ALREADY normalized (i.e., the output
    /// of a forward pass that applies the final layer norm). It does NOT apply
    /// additional normalization.
    fn compute_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        // Get last token position
        let seq_len = hidden.dim(1)?;
        let last_hidden = hidden.i((.., seq_len - 1, ..))?.squeeze(1)?;

        // Project directly to vocabulary (hidden is already normed by forward pass)
        // Do NOT call logit_lens here as it would apply norm again
        self.model.project_to_vocab(&last_hidden)
    }

    // ========================================================================
    // Generation Methods (for end-to-end steering benchmark)
    // ========================================================================

    /// Generate with chat template (for instruct models)
    ///
    /// Automatically applies the appropriate chat template for the model
    /// and uses the model's EOS token for stopping.
    ///
    /// # Arguments
    /// * `prompt` - User prompt (will be wrapped in chat template)
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature (0.0 = greedy)
    /// * `system_prompt` - Optional system prompt
    ///
    /// # Example
    /// ```ignore
    /// let model = PlipModel::from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")?;
    /// let output = model.generate_chat(
    ///     "Write a function to compute factorial",
    ///     200,
    ///     0.0,
    ///     None,
    /// )?;
    /// ```
    pub fn generate_chat(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        system_prompt: Option<&str>,
    ) -> Result<String> {
        let formatted = self.apply_chat_template(prompt, system_prompt);
        let stop_tokens: Vec<u32> = self.eos_token_id().into_iter().collect();

        let output = self.generate(&formatted, max_tokens, temperature, &stop_tokens)?;

        // Extract only the assistant's response (after the last assistant marker)
        if self.is_instruct_model() {
            if let Some(idx) = output.rfind("<|im_start|>assistant\n") {
                let start = idx + "<|im_start|>assistant\n".len();
                let response = &output[start..];
                // Remove trailing end marker if present
                let response = response.trim_end_matches("<|im_end|>");
                return Ok(response.to_string());
            }
        }

        Ok(output)
    }

    /// Generate with chat template and steering
    ///
    /// Combines chat template formatting with attention steering.
    pub fn generate_chat_with_steering(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        system_prompt: Option<&str>,
        steering: Option<&SteeringSpec>,
    ) -> Result<String> {
        let formatted = self.apply_chat_template(prompt, system_prompt);
        let stop_tokens: Vec<u32> = self.eos_token_id().into_iter().collect();

        let output = self.generate_with_steering(
            &formatted,
            max_tokens,
            temperature,
            &stop_tokens,
            steering,
        )?;

        // Extract only the assistant's response
        if self.is_instruct_model() {
            if let Some(idx) = output.rfind("<|im_start|>assistant\n") {
                let start = idx + "<|im_start|>assistant\n".len();
                let response = &output[start..];
                let response = response.trim_end_matches("<|im_end|>");
                return Ok(response.to_string());
            }
        }

        Ok(output)
    }

    /// Generate tokens autoregressively without steering
    ///
    /// Uses KV-cache for efficient generation - only computes attention
    /// for new tokens while reusing cached keys/values from previous positions.
    ///
    /// # Arguments
    /// * `prompt` - Initial text to continue from
    /// * `max_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (0.0 = greedy, higher = more random)
    /// * `stop_tokens` - Token IDs that stop generation (e.g., EOS)
    ///
    /// # Returns
    /// Generated text (including prompt)
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
    ) -> Result<String> {
        self.generate_with_steering(prompt, max_tokens, temperature, stop_tokens, None)
    }

    /// Generate tokens with optional steering intervention
    ///
    /// Uses KV-cache for efficient generation. At each step, if steering is
    /// enabled, the full context is processed with steering applied (no cache).
    ///
    /// # Arguments
    /// * `prompt` - Initial text to continue from
    /// * `max_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (0.0 = greedy)
    /// * `stop_tokens` - Token IDs that stop generation
    /// * `steering` - Optional steering specification (layer, edges, scale)
    ///
    /// # Returns
    /// Generated text (including prompt)
    ///
    /// # Example
    /// ```ignore
    /// use plip_rs::{PlipModel, SteeringSpec};
    ///
    /// let model = PlipModel::from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")?;
    ///
    /// // Generate with 3x attention boost to test markers
    /// let spec = SteeringSpec::scale(3.0)
    ///     .layer(20)
    ///     .from_to_positions(marker_pos, &fn_positions);
    ///
    /// let output = model.generate_with_steering(
    ///     "fn max(a: i32, b: i32) -> i32 {\n    // TODO",
    ///     100,
    ///     0.0,
    ///     &[],
    ///     Some(&spec),
    /// )?;
    /// ```
    pub fn generate_with_steering(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        steering: Option<&SteeringSpec>,
    ) -> Result<String> {
        // Tokenize prompt
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();

        // Use backend-specific KV-cache generation when no steering is applied
        // (Steering requires full context processing at each step for proper attention modification)
        // All backends now support KV-cache generation for efficient autoregressive generation
        if steering.is_none() {
            let tokens = self.model.generate(
                &prompt_ids,
                max_tokens,
                temperature,
                stop_tokens,
                &self.device,
            )?;

            let output = self
                .tokenizer
                .decode(&tokens, true)
                .map_err(|e| anyhow::anyhow!("Decode error: {e}"))?;

            return Ok(output);
        }

        let steering_spec = steering.unwrap();
        let prompt_len = prompt_ids.len();

        // HYBRID APPROACH: If steering only affects prompt positions, we can use
        // KV-cache for efficient generation after processing the prompt with steering.
        // This is O(prompt_len²) + O(n*context_len) instead of O(n*context_len²)
        let tokens = if steering_spec.is_prompt_only(prompt_len) {
            // Use KV-cache with prompt-only steering
            self.model.generate_with_prompt_steering(
                &prompt_ids,
                max_tokens,
                temperature,
                stop_tokens,
                steering_spec,
                &self.device,
            )?
        } else {
            // Steering affects generation positions - need full recomputation
            self.generate_with_steering_no_cache(
                &prompt_ids,
                max_tokens,
                temperature,
                stop_tokens,
                steering_spec,
            )?
        };

        let output = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {e}"))?;

        Ok(output)
    }

    /// Generate with steering (requires full context at each step)
    fn generate_with_steering_no_cache(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        steering: &SteeringSpec,
    ) -> Result<Vec<u32>> {
        let mut tokens = prompt_ids.to_vec();

        for _ in 0..max_tokens {
            let input_tensor = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
            let seq_len = tokens.len();

            let logits = if steering
                .validate(self.n_layers(), self.n_heads(), seq_len)
                .is_ok()
            {
                let (output, _) = self.model.forward_with_steering(&input_tensor, steering)?;
                self.compute_logits(&output)?
            } else {
                tracing::warn!("Steering validation failed, using baseline");
                let (output, _) = self.model.forward_with_attention(&input_tensor)?;
                self.compute_logits(&output)?
            };

            let next_token = sample_token(&logits, temperature)?;

            if stop_tokens.contains(&next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Generate and return both text and token-level details
    ///
    /// Useful for analyzing which tokens were generated and comparing
    /// baseline vs steered outputs. Uses KV-cache for efficient generation
    /// when no steering is applied.
    pub fn generate_with_details(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        steering: Option<&SteeringSpec>,
    ) -> Result<GenerationResult> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = prompt_ids.len();

        // Generate tokens
        let tokens = if let Some(spec) = steering {
            self.generate_with_steering_no_cache(
                &prompt_ids,
                max_tokens,
                temperature,
                stop_tokens,
                spec,
            )?
        } else {
            self.model.generate(
                &prompt_ids,
                max_tokens,
                temperature,
                stop_tokens,
                &self.device,
            )?
        };

        let generated_tokens = tokens[prompt_len..].to_vec();

        let full_text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {e}"))?;

        let generated_text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {e}"))?;

        Ok(GenerationResult {
            prompt: prompt.to_string(),
            full_text,
            generated_text,
            prompt_tokens: tokens[..prompt_len].to_vec(),
            generated_tokens,
            total_tokens: tokens.len(),
        })
    }

    /// Generate text with KV-cache memory limit enforcement
    ///
    /// This method is optimized for memory-constrained scenarios (e.g., 16GB VRAM).
    /// When the KV-cache exceeds the specified memory limit, older context is
    /// trimmed to keep memory usage within bounds.
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature (0.0 = greedy)
    /// * `stop_tokens` - Tokens that stop generation
    /// * `max_kv_cache_mb` - Maximum KV-cache size in megabytes
    ///
    /// # Memory Analysis (for reference)
    /// For a 7B model (typical hyperparameters):
    /// - num_kv_heads = 8 (GQA), head_dim = 128, num_layers = 32, dtype = BF16
    /// - Per token: 8 * 128 * 2 * 2 * 32 = 128KB
    /// - For 2048 tokens: ~256MB
    /// - Setting max_kv_cache_mb = 256 keeps context to ~2K tokens
    ///
    /// # Example
    /// ```ignore
    /// // Generate with 256MB KV-cache limit (good for 16GB VRAM)
    /// let output = model.generate_with_memory_limit(
    ///     "def fibonacci(n):",
    ///     100,
    ///     0.8,
    ///     &[],
    ///     256, // 256 MB max cache
    /// )?;
    /// ```
    pub fn generate_with_memory_limit(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        max_kv_cache_mb: usize,
    ) -> Result<String> {
        let max_bytes = max_kv_cache_mb * 1024 * 1024;

        // Tokenize prompt
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();

        // Use KV-cache generation with memory enforcement via trait
        let tokens = self.generate_with_memory_limit_impl(
            &*self.model,
            &prompt_ids,
            max_tokens,
            temperature,
            stop_tokens,
            max_bytes,
        )?;

        let output = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {e}"))?;

        Ok(output)
    }

    /// Internal implementation for memory-limited generation
    fn generate_with_memory_limit_impl(
        &self,
        model: &dyn PlipBackend,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        max_bytes: usize,
    ) -> Result<Vec<u32>> {
        let mut kv_cache = model.new_kv_cache();
        let mut tokens = prompt_ids.to_vec();

        // Process full prompt first
        let prompt_tensor = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let logits = model.forward_with_kv_cache(&prompt_tensor, &mut kv_cache)?;

        // Enforce memory limit after prompt processing
        let trimmed = kv_cache.enforce_memory_limit(max_bytes)?;
        if trimmed {
            tracing::info!(
                "Trimmed KV-cache after prompt processing to {} tokens",
                kv_cache.seq_len()
            );
        }

        // Sample first generated token
        let mut next_token = sample_token(&logits, temperature)?;

        if stop_tokens.contains(&next_token) {
            return Ok(tokens);
        }
        tokens.push(next_token);

        // Generate remaining tokens
        for _ in 1..max_tokens {
            let input_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = model.forward_with_kv_cache(&input_tensor, &mut kv_cache)?;

            // Periodically enforce memory limit (every 100 tokens)
            if tokens.len().is_multiple_of(100) {
                let trimmed = kv_cache.enforce_memory_limit(max_bytes)?;
                if trimmed {
                    tracing::debug!(
                        "Trimmed KV-cache at token {} to {} seq_len, {} bytes",
                        tokens.len(),
                        kv_cache.seq_len(),
                        kv_cache.memory_usage()
                    );
                }
            }

            next_token = sample_token(&logits, temperature)?;

            if stop_tokens.contains(&next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

}

/// Sample a token from logits
fn sample_token(logits: &Tensor, temperature: f32) -> Result<u32> {
    if temperature <= 0.0 {
        argmax(logits)
    } else {
        sample_with_temperature(logits, temperature)
    }
}

/// Argmax sampling (greedy)
fn argmax(logits: &Tensor) -> Result<u32> {
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let logits_vec: Vec<f32> = logits_f32.flatten_all()?.to_vec1()?;

    let (max_idx, _) = logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| anyhow::anyhow!("Empty logits"))?;

    Ok(max_idx as u32)
}

/// Temperature-based sampling
fn sample_with_temperature(logits: &Tensor, temperature: f32) -> Result<u32> {
    use rand::Rng;

    let logits_f32 = logits.to_dtype(DType::F32)?;
    let logits_vec: Vec<f32> = logits_f32.flatten_all()?.to_vec1()?;

    // Apply temperature
    let scaled: Vec<f32> = logits_vec.iter().map(|x| x / temperature).collect();

    // Softmax
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|x| x / sum).collect();

    // Sample from distribution
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (idx, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return Ok(idx as u32);
        }
    }

    // Fallback to last token
    Ok((probs.len() - 1) as u32)
}

/// Result of text generation with details
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Original prompt text
    pub prompt: String,
    /// Full output (prompt + generated)
    pub full_text: String,
    /// Only the generated portion
    pub generated_text: String,
    /// Token IDs from the prompt
    pub prompt_tokens: Vec<u32>,
    /// Token IDs that were generated
    pub generated_tokens: Vec<u32>,
    /// Total token count
    pub total_tokens: usize,
}
