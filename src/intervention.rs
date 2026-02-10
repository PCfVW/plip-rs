//! Attention Intervention for PLIP-rs
//!
//! Enables causal intervention experiments by surgically modifying
//! attention edges and measuring impact on model outputs.
//!
//! ## Intervention Types
//!
//! - **Knockout**: Remove attention edges (pre-softmax, add -inf)
//! - **Scale**: Multiply attention by a factor (post-softmax, then renormalize)
//! - **SetValue**: Set attention to a specific value (post-softmax, then renormalize)
//!
//! ## Intervention Mechanism
//!
//! Knockout is implemented by adding -infinity to specified attention
//! scores BEFORE softmax. After softmax, these edges become exactly 0,
//! completely removing their contribution to the output.
//!
//! Steering (Scale/SetValue) is applied AFTER softmax, modifying attention
//! weights and renormalizing rows to maintain valid probability distributions.
//!
//! ## Example: Knockout
//!
//! ```ignore
//! use plip_rs::{PlipModel, KnockoutSpec};
//!
//! let model = PlipModel::from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")?;
//!
//! // Knockout attention from token 5 to tokens 0-3 in layer 10
//! let spec = KnockoutSpec::new()
//!     .layer(10)
//!     .from_to_positions(5, &[0, 1, 2, 3]);
//!
//! let result = model.forward_with_intervention("def add(a, b):", &spec)?;
//! println!("KL divergence: {}", result.kl_divergence()?);
//! ```
//!
//! ## Example: Steering
//!
//! ```ignore
//! use plip_rs::{PlipModel, SteeringSpec, InterventionType};
//!
//! let model = PlipModel::from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")?;
//!
//! // Boost attention from marker to function tokens by 3x
//! let spec = SteeringSpec::new(InterventionType::Scale(3.0))
//!     .layer(16)
//!     .from_to_positions(marker_pos, &function_positions);
//!
//! let result = model.forward_with_steering("...", &spec)?;
//! println!("KL divergence: {}", result.kl_divergence()?);
//! ```

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Helper function to extract 4D tensor data to nested Vecs
///
/// Candle doesn't have to_vec4(), so we flatten and reshape manually.
fn tensor_to_vec4(tensor: &Tensor) -> Result<Vec<Vec<Vec<Vec<f32>>>>> {
    let dims = tensor.dims();
    if dims.len() != 4 {
        anyhow::bail!("Expected 4D tensor, got {}D", dims.len());
    }
    let (d0, d1, d2, d3) = (dims[0], dims[1], dims[2], dims[3]);

    // Flatten to 1D and extract
    let flat: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

    // Reshape to 4D using iterator chunks
    let mut result = Vec::with_capacity(d0);
    let mut iter = flat.into_iter();
    for _ in 0..d0 {
        let mut dim1 = Vec::with_capacity(d1);
        for _ in 0..d1 {
            let mut dim2 = Vec::with_capacity(d2);
            for _ in 0..d2 {
                let dim3: Vec<f32> = iter.by_ref().take(d3).collect();
                dim2.push(dim3);
            }
            dim1.push(dim2);
        }
        result.push(dim1);
    }

    Ok(result)
}

/// Specification for which attention edges to knockout
///
/// An "edge" is defined as attention from one token position to another.
/// Knockout removes the edge completely by setting its attention weight to 0.
#[derive(Debug, Clone)]
pub struct KnockoutSpec {
    /// Layer indices to apply intervention (empty = all layers)
    pub layers: LayerSpec,

    /// Head indices to apply intervention (empty = all heads)
    pub heads: HeadSpec,

    /// Attention edges to knockout: (from_position, to_position)
    pub edges: Vec<AttentionEdge>,
}

/// Specification for which layers to target
#[derive(Debug, Clone)]
pub enum LayerSpec {
    /// Apply to all layers
    All,
    /// Apply to specific layers
    Specific(Vec<usize>),
    /// Apply to a range of layers (inclusive)
    Range { start: usize, end: usize },
}

/// Specification for which heads to target
#[derive(Debug, Clone)]
pub enum HeadSpec {
    /// Apply to all heads
    All,
    /// Apply to specific heads
    Specific(Vec<usize>),
}

/// A single attention edge from one position to another
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionEdge {
    /// Token position that is attending (row in attention matrix)
    pub from_pos: usize,
    /// Token position being attended to (column in attention matrix)
    pub to_pos: usize,
}

impl AttentionEdge {
    /// Create a new edge
    pub fn new(from_pos: usize, to_pos: usize) -> Self {
        Self { from_pos, to_pos }
    }
}

// ============================================================================
// Part 2: Amplification (Attention Steering)
// ============================================================================

/// Type of intervention to apply to attention weights
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum InterventionType {
    /// Set attention to zero (pre-softmax: add -inf)
    #[default]
    Knockout,
    /// Multiply attention by factor (post-softmax, then renormalize)
    Scale(f32),
    /// Set attention to specific target value (post-softmax, then renormalize)
    SetValue(f32),
}

/// Specification for attention steering interventions
///
/// Unlike knockout which removes edges, steering modifies attention weights
/// by scaling or setting values, then renormalizing to maintain valid
/// probability distributions.
#[derive(Debug, Clone)]
pub struct SteeringSpec {
    /// Layer indices to apply intervention (empty = all layers)
    pub layers: LayerSpec,

    /// Head indices to apply intervention (empty = all heads)
    pub heads: HeadSpec,

    /// Attention edges to modify: (from_position, to_position)
    pub edges: Vec<AttentionEdge>,

    /// Type of intervention to apply
    pub intervention_type: InterventionType,
}

impl SteeringSpec {
    /// Create a new steering specification with the given intervention type
    pub fn new(intervention_type: InterventionType) -> Self {
        Self {
            layers: LayerSpec::All,
            heads: HeadSpec::All,
            edges: Vec::new(),
            intervention_type,
        }
    }

    /// Create a scaling intervention
    pub fn scale(factor: f32) -> Self {
        Self::new(InterventionType::Scale(factor))
    }

    /// Create a set-value intervention
    pub fn set_value(target: f32) -> Self {
        Self::new(InterventionType::SetValue(target))
    }

    /// Set specific layer(s) to target
    pub fn layer(mut self, layer: usize) -> Self {
        self.layers = LayerSpec::Specific(vec![layer]);
        self
    }

    /// Set multiple layers to target
    pub fn layers(mut self, layers: &[usize]) -> Self {
        self.layers = LayerSpec::Specific(layers.to_vec());
        self
    }

    /// Set layer range to target (inclusive)
    pub fn layer_range(mut self, start: usize, end: usize) -> Self {
        self.layers = LayerSpec::Range { start, end };
        self
    }

    /// Set specific head(s) to target
    pub fn head(mut self, head: usize) -> Self {
        self.heads = HeadSpec::Specific(vec![head]);
        self
    }

    /// Set multiple heads to target
    pub fn heads(mut self, heads: &[usize]) -> Self {
        self.heads = HeadSpec::Specific(heads.to_vec());
        self
    }

    /// Add a single edge to modify
    pub fn edge(mut self, from_pos: usize, to_pos: usize) -> Self {
        self.edges.push(AttentionEdge::new(from_pos, to_pos));
        self
    }

    /// Steer all attention FROM a specific position to all other positions
    pub fn from_position(mut self, from_pos: usize) -> Self {
        self.edges.push(AttentionEdge::new(from_pos, usize::MAX));
        self
    }

    /// Steer all attention TO a specific position from all other positions
    pub fn to_position(mut self, to_pos: usize) -> Self {
        self.edges.push(AttentionEdge::new(usize::MAX, to_pos));
        self
    }

    /// Add multiple edges from one position to several positions
    pub fn from_to_positions(mut self, from_pos: usize, to_positions: &[usize]) -> Self {
        for &to_pos in to_positions {
            self.edges.push(AttentionEdge::new(from_pos, to_pos));
        }
        self
    }

    /// Check if this layer should have intervention applied
    pub fn applies_to_layer(&self, layer: usize) -> bool {
        match &self.layers {
            LayerSpec::All => true,
            LayerSpec::Specific(layers) => layers.contains(&layer),
            LayerSpec::Range { start, end } => layer >= *start && layer <= *end,
        }
    }

    /// Check if this head should have intervention applied
    pub fn applies_to_head(&self, head: usize) -> bool {
        match &self.heads {
            HeadSpec::All => true,
            HeadSpec::Specific(heads) => heads.contains(&head),
        }
    }

    /// Validate the spec against model dimensions
    pub fn validate(&self, n_layers: usize, n_heads: usize, seq_len: usize) -> Result<()> {
        // Check layers
        match &self.layers {
            LayerSpec::Specific(layers) => {
                for &l in layers {
                    if l >= n_layers {
                        anyhow::bail!("Layer {l} out of range (model has {n_layers} layers)");
                    }
                }
            }
            LayerSpec::Range { start, end } => {
                if *end >= n_layers {
                    anyhow::bail!(
                        "Layer range end {end} out of range (model has {n_layers} layers)"
                    );
                }
                if start > end {
                    anyhow::bail!("Invalid layer range: start {start} > end {end}");
                }
            }
            LayerSpec::All => {}
        }

        // Check heads
        if let HeadSpec::Specific(heads) = &self.heads {
            for &h in heads {
                if h >= n_heads {
                    anyhow::bail!("Head {h} out of range (model has {n_heads} heads)");
                }
            }
        }

        // Check edges (skip sentinels)
        for edge in &self.edges {
            if edge.from_pos != usize::MAX && edge.from_pos >= seq_len {
                anyhow::bail!(
                    "Edge from_pos {} out of range (seq_len is {})",
                    edge.from_pos,
                    seq_len
                );
            }
            if edge.to_pos != usize::MAX && edge.to_pos >= seq_len {
                anyhow::bail!(
                    "Edge to_pos {} out of range (seq_len is {})",
                    edge.to_pos,
                    seq_len
                );
            }
        }

        // Validate intervention type parameters
        match self.intervention_type {
            InterventionType::Scale(factor) => {
                if factor < 0.0 {
                    anyhow::bail!("Scale factor must be non-negative, got {factor}");
                }
            }
            InterventionType::SetValue(value) => {
                if !(0.0..=1.0).contains(&value) {
                    anyhow::bail!("SetValue must be in [0, 1], got {value}");
                }
            }
            InterventionType::Knockout => {}
        }

        Ok(())
    }

    /// Get the intervention type
    pub fn intervention_type(&self) -> InterventionType {
        self.intervention_type
    }

    /// Check if this is a knockout intervention
    pub fn is_knockout(&self) -> bool {
        matches!(self.intervention_type, InterventionType::Knockout)
    }

    /// Check if this is a post-softmax steering intervention
    pub fn is_steering(&self) -> bool {
        matches!(
            self.intervention_type,
            InterventionType::Scale(_) | InterventionType::SetValue(_)
        )
    }

    /// Check if steering only affects positions within the prompt
    ///
    /// This is useful for KV-cache optimization: if all steering edges
    /// target prompt positions only (from_pos < prompt_len), we can:
    /// 1. Apply steering during prompt processing
    /// 2. Cache K,V from the steered forward pass
    /// 3. Generate using cache (no steering needed for generated tokens)
    ///
    /// # Arguments
    /// * `prompt_len` - Length of the prompt in tokens
    ///
    /// # Returns
    /// `true` if all edges have `from_pos < prompt_len`, `false` otherwise
    pub fn is_prompt_only(&self, prompt_len: usize) -> bool {
        for edge in &self.edges {
            // Skip sentinel values (they expand based on seq_len at runtime)
            if edge.from_pos == usize::MAX {
                // "FROM all positions" - not compatible with prompt-only
                return false;
            }
            if edge.from_pos >= prompt_len {
                return false;
            }
        }
        true
    }

    /// Get the maximum `from_pos` among all edges
    ///
    /// Useful for determining the minimum context length needed for steering.
    /// Returns `None` if there are no edges or all edges use sentinels.
    pub fn max_from_pos(&self) -> Option<usize> {
        self.edges
            .iter()
            .filter(|e| e.from_pos != usize::MAX)
            .map(|e| e.from_pos)
            .max()
    }

    /// Get the maximum `to_pos` among all edges
    ///
    /// Useful for determining the minimum context length needed for steering.
    /// Returns `None` if there are no edges or all edges use sentinels.
    pub fn max_to_pos(&self) -> Option<usize> {
        self.edges
            .iter()
            .filter(|e| e.to_pos != usize::MAX)
            .map(|e| e.to_pos)
            .max()
    }
}

impl KnockoutSpec {
    /// Create a new empty knockout specification
    pub fn new() -> Self {
        Self {
            layers: LayerSpec::All,
            heads: HeadSpec::All,
            edges: Vec::new(),
        }
    }

    /// Set specific layer(s) to target
    pub fn layer(mut self, layer: usize) -> Self {
        self.layers = LayerSpec::Specific(vec![layer]);
        self
    }

    /// Set multiple layers to target
    pub fn layers(mut self, layers: &[usize]) -> Self {
        self.layers = LayerSpec::Specific(layers.to_vec());
        self
    }

    /// Set layer range to target (inclusive)
    pub fn layer_range(mut self, start: usize, end: usize) -> Self {
        self.layers = LayerSpec::Range { start, end };
        self
    }

    /// Set specific head(s) to target
    pub fn head(mut self, head: usize) -> Self {
        self.heads = HeadSpec::Specific(vec![head]);
        self
    }

    /// Set multiple heads to target
    pub fn heads(mut self, heads: &[usize]) -> Self {
        self.heads = HeadSpec::Specific(heads.to_vec());
        self
    }

    /// Add a single edge to knockout
    pub fn edge(mut self, from_pos: usize, to_pos: usize) -> Self {
        self.edges.push(AttentionEdge::new(from_pos, to_pos));
        self
    }

    /// Knockout all attention FROM a specific position to all other positions
    pub fn from_position(mut self, from_pos: usize) -> Self {
        // Mark with sentinel - will expand at mask creation time based on seq_len
        self.edges.push(AttentionEdge::new(from_pos, usize::MAX));
        self
    }

    /// Knockout all attention TO a specific position from all other positions
    pub fn to_position(mut self, to_pos: usize) -> Self {
        // Mark with sentinel
        self.edges.push(AttentionEdge::new(usize::MAX, to_pos));
        self
    }

    /// Add multiple edges from one position to several positions
    pub fn from_to_positions(mut self, from_pos: usize, to_positions: &[usize]) -> Self {
        for &to_pos in to_positions {
            self.edges.push(AttentionEdge::new(from_pos, to_pos));
        }
        self
    }

    /// Check if this layer should have intervention applied
    pub fn applies_to_layer(&self, layer: usize) -> bool {
        match &self.layers {
            LayerSpec::All => true,
            LayerSpec::Specific(layers) => layers.contains(&layer),
            LayerSpec::Range { start, end } => layer >= *start && layer <= *end,
        }
    }

    /// Check if this head should have intervention applied
    pub fn applies_to_head(&self, head: usize) -> bool {
        match &self.heads {
            HeadSpec::All => true,
            HeadSpec::Specific(heads) => heads.contains(&head),
        }
    }

    /// Validate the spec against model dimensions
    pub fn validate(&self, n_layers: usize, n_heads: usize, seq_len: usize) -> Result<()> {
        // Check layers
        match &self.layers {
            LayerSpec::Specific(layers) => {
                for &l in layers {
                    if l >= n_layers {
                        anyhow::bail!("Layer {l} out of range (model has {n_layers} layers)");
                    }
                }
            }
            LayerSpec::Range { start, end } => {
                if *end >= n_layers {
                    anyhow::bail!(
                        "Layer range end {end} out of range (model has {n_layers} layers)"
                    );
                }
                if start > end {
                    anyhow::bail!("Invalid layer range: start {start} > end {end}");
                }
            }
            LayerSpec::All => {}
        }

        // Check heads
        if let HeadSpec::Specific(heads) = &self.heads {
            for &h in heads {
                if h >= n_heads {
                    anyhow::bail!("Head {h} out of range (model has {n_heads} heads)");
                }
            }
        }

        // Check edges (skip sentinels)
        for edge in &self.edges {
            if edge.from_pos != usize::MAX && edge.from_pos >= seq_len {
                anyhow::bail!(
                    "Edge from_pos {} out of range (seq_len is {})",
                    edge.from_pos,
                    seq_len
                );
            }
            if edge.to_pos != usize::MAX && edge.to_pos >= seq_len {
                anyhow::bail!(
                    "Edge to_pos {} out of range (seq_len is {})",
                    edge.to_pos,
                    seq_len
                );
            }
        }

        Ok(())
    }
}

impl Default for KnockoutSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of an ablation experiment
#[derive(Debug)]
pub struct AblationResult {
    /// Logits from baseline forward pass (no intervention)
    pub baseline_logits: Tensor,

    /// Logits from intervened forward pass
    pub ablated_logits: Tensor,

    /// The knockout specification used
    pub spec: KnockoutSpec,

    /// Attention weights from intervened pass (if captured)
    pub ablated_attention: Option<crate::AttentionCache>,
}

impl AblationResult {
    /// Create new ablation result
    pub fn new(
        baseline_logits: Tensor,
        ablated_logits: Tensor,
        spec: KnockoutSpec,
        ablated_attention: Option<crate::AttentionCache>,
    ) -> Self {
        Self {
            baseline_logits,
            ablated_logits,
            spec,
            ablated_attention,
        }
    }

    /// Compute KL divergence between baseline and ablated distributions
    ///
    /// Returns KL(baseline || ablated) for the last token position.
    /// Higher values indicate the knockout had more impact.
    pub fn kl_divergence(&self) -> Result<f32> {
        kl_divergence(&self.baseline_logits, &self.ablated_logits)
    }

    /// Compute logit difference for a specific token
    ///
    /// Returns (baseline_logit - ablated_logit) for the specified token.
    /// Positive values mean knockout decreased the token's probability.
    pub fn logit_diff(&self, token_id: u32) -> Result<f32> {
        let baseline_f32 = self.baseline_logits.to_dtype(DType::F32)?;
        let ablated_f32 = self.ablated_logits.to_dtype(DType::F32)?;

        let baseline_vec: Vec<f32> = baseline_f32.flatten_all()?.to_vec1()?;
        let ablated_vec: Vec<f32> = ablated_f32.flatten_all()?.to_vec1()?;

        let idx = token_id as usize;
        if idx >= baseline_vec.len() {
            anyhow::bail!("Token ID {token_id} out of range");
        }

        Ok(baseline_vec[idx] - ablated_vec[idx])
    }

    /// Get top-k tokens that changed most due to ablation
    ///
    /// Returns Vec of (token_id, baseline_prob, ablated_prob, abs_diff)
    pub fn top_changed_tokens(&self, k: usize) -> Result<Vec<(u32, f32, f32, f32)>> {
        let baseline_probs = softmax_to_vec(&self.baseline_logits)?;
        let ablated_probs = softmax_to_vec(&self.ablated_logits)?;

        let mut changes: Vec<(u32, f32, f32, f32)> = baseline_probs
            .iter()
            .zip(ablated_probs.iter())
            .enumerate()
            .map(|(idx, (&base, &abl))| (idx as u32, base, abl, (base - abl).abs()))
            .collect();

        changes.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        Ok(changes.into_iter().take(k).collect())
    }
}

/// Result of a steering experiment
#[derive(Debug)]
pub struct SteeringResult {
    /// Logits from baseline forward pass (no intervention)
    pub baseline_logits: Tensor,

    /// Logits from steered forward pass
    pub steered_logits: Tensor,

    /// The steering specification used
    pub spec: SteeringSpec,

    /// Attention weights from steered pass (if captured)
    pub steered_attention: Option<crate::AttentionCache>,

    /// Mean attention to target edges before steering
    pub baseline_attention_mean: Option<f32>,

    /// Mean attention to target edges after steering
    pub steered_attention_mean: Option<f32>,
}

impl SteeringResult {
    /// Create new steering result
    pub fn new(
        baseline_logits: Tensor,
        steered_logits: Tensor,
        spec: SteeringSpec,
        steered_attention: Option<crate::AttentionCache>,
    ) -> Self {
        Self {
            baseline_logits,
            steered_logits,
            spec,
            steered_attention,
            baseline_attention_mean: None,
            steered_attention_mean: None,
        }
    }

    /// Create with attention measurements
    pub fn with_attention_measurements(mut self, baseline_mean: f32, steered_mean: f32) -> Self {
        self.baseline_attention_mean = Some(baseline_mean);
        self.steered_attention_mean = Some(steered_mean);
        self
    }

    /// Compute KL divergence between baseline and steered distributions
    ///
    /// Returns KL(baseline || steered) for the last token position.
    /// Higher values indicate the steering had more impact.
    pub fn kl_divergence(&self) -> Result<f32> {
        kl_divergence(&self.baseline_logits, &self.steered_logits)
    }

    /// Compute logit difference for a specific token
    ///
    /// Returns (baseline_logit - steered_logit) for the specified token.
    /// Positive values mean steering decreased the token's probability.
    pub fn logit_diff(&self, token_id: u32) -> Result<f32> {
        let baseline_f32 = self.baseline_logits.to_dtype(DType::F32)?;
        let steered_f32 = self.steered_logits.to_dtype(DType::F32)?;

        let baseline_vec: Vec<f32> = baseline_f32.flatten_all()?.to_vec1()?;
        let steered_vec: Vec<f32> = steered_f32.flatten_all()?.to_vec1()?;

        let idx = token_id as usize;
        if idx >= baseline_vec.len() {
            anyhow::bail!("Token ID {token_id} out of range");
        }

        Ok(baseline_vec[idx] - steered_vec[idx])
    }

    /// Get top-k tokens that changed most due to steering
    ///
    /// Returns Vec of (token_id, baseline_prob, steered_prob, abs_diff)
    pub fn top_changed_tokens(&self, k: usize) -> Result<Vec<(u32, f32, f32, f32)>> {
        let baseline_probs = softmax_to_vec(&self.baseline_logits)?;
        let steered_probs = softmax_to_vec(&self.steered_logits)?;

        let mut changes: Vec<(u32, f32, f32, f32)> = baseline_probs
            .iter()
            .zip(steered_probs.iter())
            .enumerate()
            .map(|(idx, (&base, &steer))| (idx as u32, base, steer, (base - steer).abs()))
            .collect();

        changes.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        Ok(changes.into_iter().take(k).collect())
    }

    /// Get the attention change ratio
    ///
    /// Returns steered_mean / baseline_mean if both are available.
    pub fn attention_ratio(&self) -> Option<f32> {
        match (self.baseline_attention_mean, self.steered_attention_mean) {
            (Some(base), Some(steered)) if base > 1e-10 => Some(steered / base),
            _ => None,
        }
    }
}

/// Create a knockout mask tensor for the given specification
///
/// Returns a tensor of shape [1, n_heads, seq_len, seq_len] where:
/// - 0.0 = no knockout (attention allowed)
/// - -inf = knockout (attention blocked)
///
/// This mask is ADDED to the attention scores before softmax.
pub fn create_knockout_mask(
    spec: &KnockoutSpec,
    n_heads: usize,
    seq_len: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    // Start with zeros (no knockout)
    let mut mask_data = vec![0.0f32; n_heads * seq_len * seq_len];

    // Expand edges (handle sentinels)
    let expanded_edges = expand_edges(&spec.edges, seq_len);

    // Apply knockout to specified heads and edges
    for head in 0..n_heads {
        if !spec.applies_to_head(head) {
            continue;
        }

        for edge in &expanded_edges {
            if edge.from_pos < seq_len && edge.to_pos < seq_len {
                let idx = head * seq_len * seq_len + edge.from_pos * seq_len + edge.to_pos;
                mask_data[idx] = f32::NEG_INFINITY;
            }
        }
    }

    let mask = Tensor::from_vec(mask_data, (1, n_heads, seq_len, seq_len), device)?;
    Ok(mask.to_dtype(dtype)?)
}

/// Expand edge specifications, handling sentinels (usize::MAX)
fn expand_edges(edges: &[AttentionEdge], seq_len: usize) -> Vec<AttentionEdge> {
    let mut expanded = Vec::new();

    for edge in edges {
        match (edge.from_pos, edge.to_pos) {
            // Knockout all edges FROM a position
            (from, usize::MAX) if from != usize::MAX => {
                for to in 0..seq_len {
                    expanded.push(AttentionEdge::new(from, to));
                }
            }
            // Knockout all edges TO a position
            (usize::MAX, to) if to != usize::MAX => {
                for from in 0..seq_len {
                    expanded.push(AttentionEdge::new(from, to));
                }
            }
            // Regular edge
            (from, to) if from != usize::MAX && to != usize::MAX => {
                expanded.push(*edge);
            }
            _ => {} // Invalid sentinel combination, skip
        }
    }

    expanded
}

/// Compute KL divergence between two logit tensors
///
/// KL(P || Q) where P = softmax(baseline), Q = softmax(ablated)
pub fn kl_divergence(baseline_logits: &Tensor, ablated_logits: &Tensor) -> Result<f32> {
    let p = softmax_to_vec(baseline_logits)?;
    let q = softmax_to_vec(ablated_logits)?;

    let kl: f32 = p
        .iter()
        .zip(q.iter())
        .filter(|(&pi, &qi)| pi > 1e-10 && qi > 1e-10)
        .map(|(&pi, &qi)| pi * (pi / qi).ln())
        .sum();

    Ok(kl)
}

/// Convert logits to probability distribution (softmax)
fn softmax_to_vec(logits: &Tensor) -> Result<Vec<f32>> {
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let probs = candle_nn::ops::softmax_last_dim(&logits_f32)?;
    Ok(probs.flatten_all()?.to_vec1()?)
}

// ============================================================================
// Steering Application Functions
// ============================================================================

/// Apply steering intervention to attention weights (post-softmax)
///
/// This function modifies attention weights according to the steering spec
/// and renormalizes rows to maintain valid probability distributions.
///
/// # Arguments
/// * `attn_weights` - Attention weights tensor of shape [batch, heads, seq, seq]
/// * `spec` - Steering specification
/// * `n_heads` - Number of attention heads
/// * `seq_len` - Sequence length
///
/// # Returns
/// Modified attention weights tensor with same shape
pub fn apply_steering(
    attn_weights: &Tensor,
    spec: &SteeringSpec,
    n_heads: usize,
    seq_len: usize,
) -> Result<Tensor> {
    match spec.intervention_type {
        InterventionType::Scale(factor) => {
            apply_scale_steering(attn_weights, spec, n_heads, seq_len, factor)
        }
        InterventionType::SetValue(target) => {
            apply_set_value_steering(attn_weights, spec, n_heads, seq_len, target)
        }
        InterventionType::Knockout => {
            // Knockout should use pre-softmax mask, not post-softmax steering
            anyhow::bail!(
                "Knockout intervention should use create_knockout_mask, not apply_steering"
            )
        }
    }
}

/// Apply scaling to specified edges, then renormalize rows
///
/// # Arguments
/// * `attn_weights` - Attention weights [batch, heads, seq, seq]
/// * `spec` - Steering specification with edges to scale
/// * `n_heads` - Number of attention heads
/// * `seq_len` - Sequence length
/// * `scale_factor` - Factor to multiply attention weights by
pub fn apply_scale_steering(
    attn_weights: &Tensor,
    spec: &SteeringSpec,
    _n_heads: usize,
    seq_len: usize,
    scale_factor: f32,
) -> Result<Tensor> {
    // Convert to f32 for manipulation
    let attn_f32 = attn_weights.to_dtype(DType::F32)?;
    let original_dtype = attn_weights.dtype();
    let device = attn_weights.device();

    // Extract to Vec for manipulation
    // Shape: [batch, heads, seq, seq]
    let mut data: Vec<Vec<Vec<Vec<f32>>>> = tensor_to_vec4(&attn_f32)?;

    // Expand edges (handle sentinels)
    let expanded_edges = expand_edges(&spec.edges, seq_len);

    // Apply scaling and renormalization
    for batch_data in &mut data {
        for (h, head_data) in batch_data.iter_mut().enumerate() {
            if !spec.applies_to_head(h) {
                continue;
            }

            // Track which rows need renormalization
            let mut rows_modified: std::collections::HashSet<usize> =
                std::collections::HashSet::new();

            // Scale specified edges
            for edge in &expanded_edges {
                if edge.from_pos < seq_len && edge.to_pos < seq_len {
                    head_data[edge.from_pos][edge.to_pos] *= scale_factor;
                    rows_modified.insert(edge.from_pos);
                }
            }

            // Renormalize modified rows
            for row in rows_modified {
                let row_sum: f32 = head_data[row].iter().sum();
                if row_sum > 1e-10 {
                    for val in &mut head_data[row] {
                        *val /= row_sum;
                    }
                }
            }
        }
    }

    // Convert back to tensor
    let result = Tensor::new(data, device)?.to_dtype(original_dtype)?;
    Ok(result)
}

/// Set specified edges to a target value, redistributing mass from other edges
///
/// # Arguments
/// * `attn_weights` - Attention weights [batch, heads, seq, seq]
/// * `spec` - Steering specification with edges to set
/// * `n_heads` - Number of attention heads
/// * `seq_len` - Sequence length
/// * `target_value` - Value to set for each edge (e.g., 0.09 for 9%)
pub fn apply_set_value_steering(
    attn_weights: &Tensor,
    spec: &SteeringSpec,
    _n_heads: usize,
    seq_len: usize,
    target_value: f32,
) -> Result<Tensor> {
    // Convert to f32 for manipulation
    let attn_f32 = attn_weights.to_dtype(DType::F32)?;
    let original_dtype = attn_weights.dtype();
    let device = attn_weights.device();

    // Extract to Vec for manipulation
    let mut data: Vec<Vec<Vec<Vec<f32>>>> = tensor_to_vec4(&attn_f32)?;

    // Expand edges (handle sentinels)
    let expanded_edges = expand_edges(&spec.edges, seq_len);

    // Group edges by from_position for row-wise operations
    let mut edges_by_row: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for edge in &expanded_edges {
        if edge.from_pos < seq_len && edge.to_pos < seq_len {
            edges_by_row
                .entry(edge.from_pos)
                .or_default()
                .push(edge.to_pos);
        }
    }

    // Apply set-value and renormalization
    for batch_data in &mut data {
        for (h, head_data) in batch_data.iter_mut().enumerate() {
            if !spec.applies_to_head(h) {
                continue;
            }

            for (&row, target_cols) in &edges_by_row {
                // Calculate current sum of target edges
                let current_target_sum: f32 =
                    target_cols.iter().map(|&col| head_data[row][col]).sum();

                // Calculate target sum
                let new_target_sum = target_value * target_cols.len() as f32;

                // Calculate how much mass we're adding/removing
                let delta = new_target_sum - current_target_sum;

                // Get non-target positions
                let non_target_cols: Vec<usize> =
                    (0..seq_len).filter(|i| !target_cols.contains(i)).collect();

                // Set target values
                for &col in target_cols {
                    head_data[row][col] = target_value;
                }

                // Redistribute delta across non-target positions
                if !non_target_cols.is_empty() {
                    let adjustment = delta / non_target_cols.len() as f32;
                    for col in non_target_cols {
                        head_data[row][col] = (head_data[row][col] - adjustment).max(0.0);
                    }
                }

                // Final renormalization to ensure row sums to 1.0
                let row_sum: f32 = head_data[row].iter().sum();
                if row_sum > 1e-10 {
                    for val in &mut head_data[row] {
                        *val /= row_sum;
                    }
                }
            }
        }
    }

    // Convert back to tensor
    let result = Tensor::new(data, device)?.to_dtype(original_dtype)?;
    Ok(result)
}

/// Measure mean attention for specified edges
///
/// # Arguments
/// * `attn_weights` - Attention weights [batch, heads, seq, seq]
/// * `from_pos` - Source position (row)
/// * `to_positions` - Target positions (columns)
/// * `layer_idx` - Which layer's attention to measure (for multi-layer cache)
///
/// # Returns
/// Mean attention across all specified edges, averaged over heads
pub fn measure_attention_to_targets(
    attn_cache: &crate::AttentionCache,
    from_pos: usize,
    to_positions: &[usize],
    layer_idx: usize,
) -> Result<f32> {
    let attn_weights = attn_cache
        .get_layer(layer_idx)
        .ok_or_else(|| anyhow::anyhow!("Layer {layer_idx} not found in attention cache"))?;

    // attn_weights shape: [batch, heads, seq, seq]
    let attn_f32 = attn_weights.to_dtype(DType::F32)?;
    let data: Vec<Vec<Vec<Vec<f32>>>> = tensor_to_vec4(&attn_f32)?;

    // Get sequence length from innermost dimension
    let seq_len = data
        .first()
        .and_then(|b| b.first())
        .map_or(0, std::vec::Vec::len);

    if from_pos >= seq_len {
        anyhow::bail!("from_pos {from_pos} out of range (seq_len is {seq_len})");
    }

    let mut total = 0.0f32;
    let mut count = 0usize;

    for batch_data in &data {
        for head_data in batch_data {
            for &to_pos in to_positions {
                if to_pos < seq_len {
                    total += head_data[from_pos][to_pos];
                    count += 1;
                }
            }
        }
    }

    if count == 0 {
        Ok(0.0)
    } else {
        Ok(total / count as f32)
    }
}

/// Convert a KnockoutSpec to a SteeringSpec (for unified handling)
impl From<KnockoutSpec> for SteeringSpec {
    fn from(spec: KnockoutSpec) -> Self {
        SteeringSpec {
            layers: spec.layers,
            heads: spec.heads,
            edges: spec.edges,
            intervention_type: InterventionType::Knockout,
        }
    }
}

// ============================================================================
// Part 3: State Knockout (RWKV-6)
// ============================================================================

/// Specification for RWKV-6 state knockout intervention.
///
/// State knockout makes specific token positions invisible to all future
/// tokens by skipping the recurrent state update at those positions.
/// This is the RNN analogue of all-edge attention knockout in transformers.
///
/// # Example
/// ```ignore
/// use plip_rs::{PlipModel, StateKnockoutSpec};
///
/// let model = PlipModel::from_pretrained("RWKV/v6-Finch-1B6-HF")?;
///
/// // Knock out position 5 across all layers
/// let spec = StateKnockoutSpec::new()
///     .position(5);
///
/// let result = model.forward_with_state_knockout("def add(a, b):", &spec)?;
/// println!("KL divergence: {}", result.kl_divergence()?);
/// ```
#[derive(Debug, Clone)]
pub struct StateKnockoutSpec {
    /// Token positions where state update is skipped
    pub positions: Vec<usize>,
    /// Which layers to apply knockout
    pub layers: LayerSpec,
}

impl StateKnockoutSpec {
    /// Create a new empty spec (all layers, no positions yet).
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            layers: LayerSpec::All,
        }
    }

    /// Add a single position to knock out.
    pub fn position(mut self, pos: usize) -> Self {
        self.positions.push(pos);
        self
    }

    /// Add multiple positions to knock out.
    pub fn positions(mut self, positions: &[usize]) -> Self {
        self.positions.extend_from_slice(positions);
        self
    }

    /// Target a single layer.
    pub fn layer(mut self, layer: usize) -> Self {
        self.layers = LayerSpec::Specific(vec![layer]);
        self
    }

    /// Target multiple specific layers.
    pub fn layers(mut self, layers: &[usize]) -> Self {
        self.layers = LayerSpec::Specific(layers.to_vec());
        self
    }

    /// Target a range of layers (inclusive).
    pub fn layer_range(mut self, start: usize, end: usize) -> Self {
        self.layers = LayerSpec::Range { start, end };
        self
    }

    /// Check if knockout applies to this layer.
    pub fn applies_to_layer(&self, layer: usize) -> bool {
        match &self.layers {
            LayerSpec::All => true,
            LayerSpec::Specific(layers) => layers.contains(&layer),
            LayerSpec::Range { start, end } => layer >= *start && layer <= *end,
        }
    }

    /// Get knockout positions as a HashSet for O(1) lookup in the WKV loop.
    pub fn position_set(&self) -> std::collections::HashSet<usize> {
        self.positions.iter().copied().collect()
    }

    /// Validate the spec against model dimensions.
    pub fn validate(&self, n_layers: usize, seq_len: usize) -> Result<()> {
        // Check layers
        match &self.layers {
            LayerSpec::Specific(layers) => {
                for &l in layers {
                    if l >= n_layers {
                        anyhow::bail!("Layer {l} out of range (model has {n_layers} layers)");
                    }
                }
            }
            LayerSpec::Range { start, end } => {
                if *end >= n_layers {
                    anyhow::bail!(
                        "Layer range end {end} out of range (model has {n_layers} layers)"
                    );
                }
                if start > end {
                    anyhow::bail!("Invalid layer range: start {start} > end {end}");
                }
            }
            LayerSpec::All => {}
        }

        // Check positions
        for &pos in &self.positions {
            if pos >= seq_len {
                anyhow::bail!("Position {pos} out of range (seq_len is {seq_len})");
            }
        }

        if self.positions.is_empty() {
            anyhow::bail!("StateKnockoutSpec has no positions specified");
        }

        Ok(())
    }
}

impl Default for StateKnockoutSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a state knockout ablation experiment (RWKV-6).
///
/// Similar to `AblationResult` but without attention cache, since
/// RWKV-6 has no attention matrices.
#[derive(Debug)]
pub struct StateAblationResult {
    /// Logits from baseline forward pass (no intervention)
    pub baseline_logits: Tensor,
    /// Logits from state-knocked-out forward pass
    pub ablated_logits: Tensor,
    /// The state knockout specification used
    pub spec: StateKnockoutSpec,
}

impl StateAblationResult {
    pub fn new(baseline_logits: Tensor, ablated_logits: Tensor, spec: StateKnockoutSpec) -> Self {
        Self {
            baseline_logits,
            ablated_logits,
            spec,
        }
    }

    /// Compute KL divergence between baseline and ablated distributions.
    pub fn kl_divergence(&self) -> Result<f32> {
        kl_divergence(&self.baseline_logits, &self.ablated_logits)
    }

    /// Compute logit difference for a specific token (baseline - ablated).
    pub fn logit_diff(&self, token_id: u32) -> Result<f32> {
        let baseline_f32 = self.baseline_logits.to_dtype(DType::F32)?;
        let ablated_f32 = self.ablated_logits.to_dtype(DType::F32)?;
        let baseline_vec: Vec<f32> = baseline_f32.flatten_all()?.to_vec1()?;
        let ablated_vec: Vec<f32> = ablated_f32.flatten_all()?.to_vec1()?;
        let idx = token_id as usize;
        if idx >= baseline_vec.len() {
            anyhow::bail!("Token ID {token_id} out of range");
        }
        Ok(baseline_vec[idx] - ablated_vec[idx])
    }

    /// Get top-k tokens that changed most due to state knockout.
    ///
    /// Returns Vec of (token_id, baseline_prob, ablated_prob, abs_diff).
    pub fn top_changed_tokens(&self, k: usize) -> Result<Vec<(u32, f32, f32, f32)>> {
        let baseline_probs = softmax_to_vec(&self.baseline_logits)?;
        let ablated_probs = softmax_to_vec(&self.ablated_logits)?;
        let mut changes: Vec<(u32, f32, f32, f32)> = baseline_probs
            .iter()
            .zip(ablated_probs.iter())
            .enumerate()
            .map(|(idx, (&base, &abl))| (idx as u32, base, abl, (base - abl).abs()))
            .collect();
        changes.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        Ok(changes.into_iter().take(k).collect())
    }
}

// ============================================================================
// Part 4: State Steering (RWKV-6)
// ============================================================================

/// Specification for RWKV-6 state steering intervention.
///
/// State steering scales the kv write at specified positions, amplifying
/// or dampening the token's contribution to recurrent state. This is the
/// RNN analogue of post-softmax attention scaling in transformers.
///
/// - `scale = 0.0` → knockout (equivalent to `StateKnockoutSpec`)
/// - `scale = 1.0` → no-op (normal forward pass)
/// - `scale > 1.0` → amplify the token's state write
/// - `scale < 1.0` → dampen the token's state write
///
/// # Example
/// ```ignore
/// use plip_rs::{PlipModel, StateSteeringSpec};
///
/// let model = PlipModel::from_pretrained("RWKV/v6-Finch-1B6-HF")?;
///
/// // Amplify marker position's state write by 2× at layer 14
/// let spec = StateSteeringSpec::new(2.0)
///     .position(5)
///     .layer(14);
///
/// let result = model.forward_with_state_steering("def add(a, b):", &spec)?;
/// println!("KL divergence: {}", result.kl_divergence()?);
/// ```
#[derive(Debug, Clone)]
pub struct StateSteeringSpec {
    /// Token positions where state write is scaled
    pub positions: Vec<usize>,
    /// Which layers to apply steering
    pub layers: LayerSpec,
    /// Scale factor for kv write (0.0 = knockout, 1.0 = normal, >1.0 = amplify)
    pub scale: f32,
}

impl StateSteeringSpec {
    /// Create a new spec with the given scale factor (all layers, no positions yet).
    pub fn new(scale: f32) -> Self {
        Self {
            positions: Vec::new(),
            layers: LayerSpec::All,
            scale,
        }
    }

    /// Add a single position to steer.
    pub fn position(mut self, pos: usize) -> Self {
        self.positions.push(pos);
        self
    }

    /// Add multiple positions to steer.
    pub fn positions(mut self, positions: &[usize]) -> Self {
        self.positions.extend_from_slice(positions);
        self
    }

    /// Target a single layer.
    pub fn layer(mut self, layer: usize) -> Self {
        self.layers = LayerSpec::Specific(vec![layer]);
        self
    }

    /// Target multiple specific layers.
    pub fn layers(mut self, layers: &[usize]) -> Self {
        self.layers = LayerSpec::Specific(layers.to_vec());
        self
    }

    /// Target a range of layers (inclusive).
    pub fn layer_range(mut self, start: usize, end: usize) -> Self {
        self.layers = LayerSpec::Range { start, end };
        self
    }

    /// Check if steering applies to this layer.
    pub fn applies_to_layer(&self, layer: usize) -> bool {
        match &self.layers {
            LayerSpec::All => true,
            LayerSpec::Specific(layers) => layers.contains(&layer),
            LayerSpec::Range { start, end } => layer >= *start && layer <= *end,
        }
    }

    /// Get steering positions as a HashSet for O(1) lookup in the WKV loop.
    pub fn position_set(&self) -> std::collections::HashSet<usize> {
        self.positions.iter().copied().collect()
    }

    /// Validate the spec against model dimensions.
    pub fn validate(&self, n_layers: usize, seq_len: usize) -> Result<()> {
        match &self.layers {
            LayerSpec::Specific(layers) => {
                for &l in layers {
                    if l >= n_layers {
                        anyhow::bail!("Layer {l} out of range (model has {n_layers} layers)");
                    }
                }
            }
            LayerSpec::Range { start, end } => {
                if *end >= n_layers {
                    anyhow::bail!(
                        "Layer range end {end} out of range (model has {n_layers} layers)"
                    );
                }
                if start > end {
                    anyhow::bail!("Invalid layer range: start {start} > end {end}");
                }
            }
            LayerSpec::All => {}
        }

        for &pos in &self.positions {
            if pos >= seq_len {
                anyhow::bail!("Position {pos} out of range (seq_len is {seq_len})");
            }
        }

        if self.positions.is_empty() {
            anyhow::bail!("StateSteeringSpec has no positions specified");
        }

        Ok(())
    }
}

/// Result of a state steering experiment (RWKV-6).
///
/// Similar to `StateAblationResult` but for continuous-scale interventions
/// rather than binary knockout.
#[derive(Debug)]
pub struct StateSteeringResult {
    /// Logits from baseline forward pass (no intervention)
    pub baseline_logits: Tensor,
    /// Logits from steered forward pass
    pub steered_logits: Tensor,
    /// The state steering specification used
    pub spec: StateSteeringSpec,
}

impl StateSteeringResult {
    pub fn new(baseline_logits: Tensor, steered_logits: Tensor, spec: StateSteeringSpec) -> Self {
        Self {
            baseline_logits,
            steered_logits,
            spec,
        }
    }

    /// Compute KL divergence between baseline and steered distributions.
    pub fn kl_divergence(&self) -> Result<f32> {
        kl_divergence(&self.baseline_logits, &self.steered_logits)
    }

    /// Get top-k tokens that changed most due to state steering.
    ///
    /// Returns Vec of (token_id, baseline_prob, steered_prob, abs_diff).
    pub fn top_changed_tokens(&self, k: usize) -> Result<Vec<(u32, f32, f32, f32)>> {
        let baseline_probs = softmax_to_vec(&self.baseline_logits)?;
        let steered_probs = softmax_to_vec(&self.steered_logits)?;
        let mut changes: Vec<(u32, f32, f32, f32)> = baseline_probs
            .iter()
            .zip(steered_probs.iter())
            .enumerate()
            .map(|(idx, (&base, &steered))| (idx as u32, base, steered, (base - steered).abs()))
            .collect();
        changes.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        Ok(changes.into_iter().take(k).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knockout_spec_builder() {
        let spec = KnockoutSpec::new()
            .layer(5)
            .head(2)
            .edge(3, 1)
            .from_to_positions(4, &[0, 1, 2]);

        assert!(matches!(spec.layers, LayerSpec::Specific(_)));
        assert!(matches!(spec.heads, HeadSpec::Specific(_)));
        assert_eq!(spec.edges.len(), 4); // 1 + 3
    }

    #[test]
    fn test_layer_spec_applies() {
        let spec = KnockoutSpec::new().layer_range(5, 10);

        assert!(!spec.applies_to_layer(4));
        assert!(spec.applies_to_layer(5));
        assert!(spec.applies_to_layer(7));
        assert!(spec.applies_to_layer(10));
        assert!(!spec.applies_to_layer(11));
    }

    #[test]
    fn test_expand_edges() {
        let edges = vec![
            AttentionEdge::new(2, usize::MAX), // All from position 2
            AttentionEdge::new(1, 0),          // Specific edge
        ];

        let expanded = expand_edges(&edges, 4);

        // Should have 4 edges from position 2 + 1 specific edge = 5 edges
        assert_eq!(expanded.len(), 5);
    }

    #[test]
    fn test_create_knockout_mask() {
        let spec = KnockoutSpec::new().head(0).edge(2, 1);

        let mask = create_knockout_mask(&spec, 2, 4, &Device::Cpu, DType::F32).unwrap();

        assert_eq!(mask.dims(), &[1, 2, 4, 4]);

        // Check that only head 0, position (2,1) is knocked out
        let mask_vec: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Head 0, row 2, col 1 = index 0*16 + 2*4 + 1 = 9
        assert!(mask_vec[9].is_infinite() && mask_vec[9].is_sign_negative());

        // Head 1 should not be affected (index 1*16 + 2*4 + 1 = 25)
        assert_eq!(mask_vec[25], 0.0);
    }

    #[test]
    fn test_validation_catches_errors() {
        let spec = KnockoutSpec::new().layer(100).edge(50, 25);

        // Should fail validation for small model
        assert!(spec.validate(30, 16, 20).is_err());
    }

    #[test]
    fn test_validation_passes_valid_spec() {
        let spec = KnockoutSpec::new().layer(10).edge(5, 3);

        assert!(spec.validate(30, 16, 20).is_ok());
    }

    // ========== Steering Tests ==========

    #[test]
    fn test_steering_spec_builder() {
        let spec = SteeringSpec::scale(2.0)
            .layer(5)
            .head(2)
            .edge(3, 1)
            .from_to_positions(4, &[0, 1, 2]);

        assert!(matches!(spec.layers, LayerSpec::Specific(_)));
        assert!(matches!(spec.heads, HeadSpec::Specific(_)));
        assert_eq!(spec.edges.len(), 4); // 1 + 3
        assert!(
            matches!(spec.intervention_type, InterventionType::Scale(f) if (f - 2.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_steering_spec_validation() {
        // Valid scale factor
        let spec = SteeringSpec::scale(2.0).layer(10).edge(5, 3);
        assert!(spec.validate(30, 16, 20).is_ok());

        // Invalid negative scale factor
        let spec = SteeringSpec::scale(-1.0).layer(10).edge(5, 3);
        assert!(spec.validate(30, 16, 20).is_err());

        // Valid set value in [0, 1]
        let spec = SteeringSpec::set_value(0.09).layer(10).edge(5, 3);
        assert!(spec.validate(30, 16, 20).is_ok());

        // Invalid set value > 1
        let spec = SteeringSpec::set_value(1.5).layer(10).edge(5, 3);
        assert!(spec.validate(30, 16, 20).is_err());
    }

    #[test]
    fn test_steering_is_methods() {
        let knockout = SteeringSpec::new(InterventionType::Knockout);
        assert!(knockout.is_knockout());
        assert!(!knockout.is_steering());

        let scale = SteeringSpec::scale(2.0);
        assert!(!scale.is_knockout());
        assert!(scale.is_steering());

        let set_value = SteeringSpec::set_value(0.1);
        assert!(!set_value.is_knockout());
        assert!(set_value.is_steering());
    }

    #[test]
    fn test_apply_scale_steering() {
        // Create a simple 1x2x4x4 attention tensor
        // batch=1, heads=2, seq=4, seq=4
        let data: Vec<f32> = vec![
            // Head 0: uniform attention (each row sums to 1.0)
            0.25, 0.25, 0.25, 0.25, // row 0
            0.25, 0.25, 0.25, 0.25, // row 1
            0.25, 0.25, 0.25, 0.25, // row 2
            0.25, 0.25, 0.25, 0.25, // row 3
            // Head 1: same
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25,
        ];
        let tensor = Tensor::from_vec(data, (1, 2, 4, 4), &Device::Cpu).unwrap();

        // Scale edge (2, 1) by 2x
        let spec = SteeringSpec::scale(2.0).edge(2, 1);

        let result = apply_scale_steering(&tensor, &spec, 2, 4, 2.0).unwrap();
        let result_data: Vec<Vec<Vec<Vec<f32>>>> = tensor_to_vec4(&result).unwrap();

        // Row 2 should be modified: edge (2,1) scaled by 2, then renormalized
        // Before: [0.25, 0.25, 0.25, 0.25], edge (2,1) = 0.25
        // After scaling (2,1): [0.25, 0.50, 0.25, 0.25], sum = 1.25
        // After renorm: [0.20, 0.40, 0.20, 0.20]
        let row2 = &result_data[0][0][2];
        assert!((row2[0] - 0.20).abs() < 1e-5);
        assert!((row2[1] - 0.40).abs() < 1e-5);
        assert!((row2[2] - 0.20).abs() < 1e-5);
        assert!((row2[3] - 0.20).abs() < 1e-5);

        // Verify row sums to 1.0
        let row_sum: f32 = row2.iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_set_value_steering() {
        // Create a simple 1x2x4x4 attention tensor
        let data: Vec<f32> = vec![
            // Head 0: uniform attention
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, // Head 1: same
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25,
        ];
        let tensor = Tensor::from_vec(data, (1, 2, 4, 4), &Device::Cpu).unwrap();

        // Set edge (2, 1) to 0.5
        let spec = SteeringSpec::set_value(0.5).edge(2, 1);

        let result = apply_set_value_steering(&tensor, &spec, 2, 4, 0.5).unwrap();
        let result_data: Vec<Vec<Vec<Vec<f32>>>> = tensor_to_vec4(&result).unwrap();

        // Row 2 should have edge (2,1) set to ~0.5 (after renormalization)
        let row2 = &result_data[0][0][2];

        // Verify row sums to 1.0
        let row_sum: f32 = row2.iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "Row sum should be 1.0, got {}",
            row_sum
        );

        // Edge (2,1) should be the largest value in the row
        assert!(row2[1] > row2[0]);
        assert!(row2[1] > row2[2]);
        assert!(row2[1] > row2[3]);
    }

    #[test]
    fn test_knockout_to_steering_conversion() {
        let knockout = KnockoutSpec::new().layer(5).head(2).edge(3, 1);

        let steering: SteeringSpec = knockout.into();

        assert!(matches!(steering.layers, LayerSpec::Specific(ref v) if v == &[5]));
        assert!(matches!(steering.heads, HeadSpec::Specific(ref v) if v == &[2]));
        assert_eq!(steering.edges.len(), 1);
        assert!(steering.is_knockout());
    }

    #[test]
    fn test_is_prompt_only() {
        // Edges within prompt (positions 0-9)
        let spec = SteeringSpec::scale(2.0).edge(5, 2).edge(8, 3);

        // All edges are within prompt of length 10
        assert!(spec.is_prompt_only(10));

        // If prompt is only 6 tokens, edge (8, 3) is outside
        assert!(!spec.is_prompt_only(6));
    }

    #[test]
    fn test_is_prompt_only_with_sentinel() {
        // Using to_position with sentinel - this sets from_pos to usize::MAX
        let spec = SteeringSpec::scale(2.0).to_position(5); // This uses usize::MAX for from_pos

        // Sentinel (usize::MAX in from_pos) should make it NOT prompt-only
        // because it affects ALL positions (including generated ones)
        assert!(!spec.is_prompt_only(10));

        // Using from_position is fine for prompt-only (it sets to_pos to sentinel)
        // The steering applies to how position 5 attends, which is within prompt
        let spec2 = SteeringSpec::scale(2.0).from_position(5); // from_pos=5, to_pos=usize::MAX

        // This IS prompt-only because from_pos=5 < 10
        assert!(spec2.is_prompt_only(10));
    }

    #[test]
    fn test_max_positions() {
        let spec = SteeringSpec::scale(2.0).edge(5, 2).edge(8, 3).edge(3, 7);

        assert_eq!(spec.max_from_pos(), Some(8));
        assert_eq!(spec.max_to_pos(), Some(7));
    }

    #[test]
    fn test_max_positions_empty() {
        let spec = SteeringSpec::scale(2.0);

        assert_eq!(spec.max_from_pos(), None);
        assert_eq!(spec.max_to_pos(), None);
    }

    // --- State Knockout tests ---

    #[test]
    fn test_state_knockout_spec_builder() {
        let spec = StateKnockoutSpec::new().position(3).position(5).layer(10);
        assert_eq!(spec.positions, vec![3, 5]);
        assert!(matches!(spec.layers, LayerSpec::Specific(ref v) if v == &[10]));
    }

    #[test]
    fn test_state_knockout_spec_validation() {
        // Valid spec
        let spec = StateKnockoutSpec::new().position(5).layer(10);
        assert!(spec.validate(24, 20).is_ok());

        // Position out of range
        let spec = StateKnockoutSpec::new().position(25);
        assert!(spec.validate(24, 20).is_err());

        // Layer out of range
        let spec = StateKnockoutSpec::new().position(5).layer(30);
        assert!(spec.validate(24, 20).is_err());

        // Empty positions
        let spec = StateKnockoutSpec::new();
        assert!(spec.validate(24, 20).is_err());
    }

    #[test]
    fn test_state_knockout_position_set() {
        let spec = StateKnockoutSpec::new().position(3).position(5).position(3);
        let set = spec.position_set();
        assert_eq!(set.len(), 2); // deduplication
        assert!(set.contains(&3));
        assert!(set.contains(&5));
    }

    #[test]
    fn test_state_knockout_applies_to_layer() {
        let spec = StateKnockoutSpec::new().position(0).layer_range(5, 10);
        assert!(!spec.applies_to_layer(4));
        assert!(spec.applies_to_layer(5));
        assert!(spec.applies_to_layer(10));
        assert!(!spec.applies_to_layer(11));
    }
}
