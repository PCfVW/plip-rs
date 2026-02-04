//! Attention Pattern Analysis for PLIP-rs
//!
//! Captures and analyzes attention patterns to understand how
//! the model processes test-related tokens.

use candle_core::{DType, IndexOp, Tensor};

/// Cache for storing attention weights from each layer
#[derive(Debug)]
pub struct AttentionCache {
    /// Attention weights per layer: [batch, heads, seq, seq]
    patterns: Vec<Tensor>,
}

impl AttentionCache {
    /// Create new cache with expected capacity
    pub fn with_capacity(n_layers: usize) -> Self {
        Self {
            patterns: Vec::with_capacity(n_layers),
        }
    }

    /// Add attention pattern for a layer
    pub fn push(&mut self, pattern: Tensor) {
        self.patterns.push(pattern);
    }

    /// Number of layers captured
    pub fn n_layers(&self) -> usize {
        self.patterns.len()
    }

    /// Get attention pattern for a specific layer
    pub fn get_layer(&self, layer: usize) -> Option<&Tensor> {
        self.patterns.get(layer)
    }

    /// Get attention from a specific position to all other positions
    /// Returns average across all heads: [seq_len]
    pub fn attention_from_position(&self, layer: usize, position: usize) -> Option<Vec<f32>> {
        let pattern = self.patterns.get(layer)?;
        // pattern shape: [batch, heads, seq, seq]
        // We want [seq] for the given position, averaged across heads

        // Get attention row for this position, average across heads
        let attn_f32 = pattern.to_dtype(DType::F32).ok()?;
        let attn_row = attn_f32.i((0, .., position, ..)).ok()?; // [heads, seq]
        let avg_attn = attn_row.mean(0).ok()?; // [seq]
        avg_attn.to_vec1().ok()
    }

    /// Get attention TO a specific position from all other positions
    /// Returns average across all heads: [seq_len]
    pub fn attention_to_position(&self, layer: usize, position: usize) -> Option<Vec<f32>> {
        let pattern = self.patterns.get(layer)?;
        // pattern shape: [batch, heads, seq, seq]

        let attn_f32 = pattern.to_dtype(DType::F32).ok()?;
        let attn_col = attn_f32.i((0, .., .., position)).ok()?; // [heads, seq]
        let avg_attn = attn_col.mean(0).ok()?; // [seq]
        avg_attn.to_vec1().ok()
    }

    /// Get top-k positions that a given position attends to
    pub fn top_attended_positions(
        &self,
        layer: usize,
        from_position: usize,
        k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        let attn = self.attention_from_position(layer, from_position)?;
        let mut indexed: Vec<(usize, f32)> = attn.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Some(indexed.into_iter().take(k).collect())
    }
}

/// Analysis result for attention patterns
#[derive(Debug)]
pub struct AttentionAnalysis {
    /// Input tokens
    pub tokens: Vec<String>,
    /// Attention cache from all layers
    pub cache: AttentionCache,
    /// Number of layers
    pub n_layers: usize,
    /// Number of attention heads
    pub n_heads: usize,
}

impl AttentionAnalysis {
    /// Create new analysis
    pub fn new(
        tokens: Vec<String>,
        cache: AttentionCache,
        n_layers: usize,
        n_heads: usize,
    ) -> Self {
        Self {
            tokens,
            cache,
            n_layers,
            n_heads,
        }
    }

    /// Find what a specific token attends to most
    pub fn what_does_token_attend_to(
        &self,
        token_idx: usize,
        layer: usize,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        if let Some(top_positions) = self.cache.top_attended_positions(layer, token_idx, top_k) {
            top_positions
                .into_iter()
                .filter_map(|(pos, weight)| self.tokens.get(pos).map(|t| (t.clone(), weight)))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find what attends most to a specific token
    pub fn what_attends_to_token(&self, token_idx: usize, layer: usize) -> Vec<(String, f32)> {
        if let Some(attn) = self.cache.attention_to_position(layer, token_idx) {
            attn.into_iter()
                .enumerate()
                .filter_map(|(pos, weight)| self.tokens.get(pos).map(|t| (t.clone(), weight)))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Print attention summary for a specific token
    pub fn print_attention_for_token(&self, token_idx: usize, layer: usize, top_k: usize) {
        if token_idx >= self.tokens.len() {
            println!("Token index {token_idx} out of range");
            return;
        }

        let token = &self.tokens[token_idx];
        println!(
            "Layer {}: Token '{}' (position {}) attends to:",
            layer,
            token.replace('\n', "\\n"),
            token_idx
        );

        let attended = self.what_does_token_attend_to(token_idx, layer, top_k);
        for (t, w) in attended {
            println!("  {:.1}% -> '{}'", w * 100.0, t.replace('\n', "\\n"));
        }
    }

    /// Find the token index for a specific substring
    pub fn find_token(&self, substring: &str) -> Option<usize> {
        self.tokens.iter().position(|t| t.contains(substring))
    }

    /// Print summary of attention patterns for test-related tokens
    pub fn print_test_token_analysis(&self, layer: usize) {
        println!("\n=== Test Token Attention Analysis (Layer {layer}) ===\n");

        // Look for Rust test markers
        if let Some(test_idx) = self.find_token("#[test]") {
            self.print_attention_for_token(test_idx, layer, 5);
        } else if let Some(hash_idx) = self.find_token("#[") {
            println!("Found '#[' at position {hash_idx}");
            self.print_attention_for_token(hash_idx, layer, 5);
        }

        // Look for Python doctest markers
        if let Some(doctest_idx) = self.find_token(">>>") {
            self.print_attention_for_token(doctest_idx, layer, 5);
        }

        // Look for assert markers
        if let Some(assert_idx) = self.find_token("assert") {
            self.print_attention_for_token(assert_idx, layer, 5);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_cache() {
        let cache = AttentionCache::with_capacity(30);
        assert_eq!(cache.n_layers(), 0);
    }
}
