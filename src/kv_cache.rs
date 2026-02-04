//! KV-Cache for efficient autoregressive generation
//!
//! Stores key and value tensors from previous positions so they don't
//! need to be recomputed at each generation step. This enables efficient
//! token-by-token generation with O(1) complexity per token instead of O(n).
//!
//! ## Memory Layout
//!
//! Each layer stores:
//! - keys: `[batch, num_kv_heads, seq_len, head_dim]`
//! - values: `[batch, num_kv_heads, seq_len, head_dim]`
//!
//! ## Memory Estimation
//!
//! For a 7B model (typical hyperparameters):
//! - num_kv_heads = 8 (GQA)
//! - head_dim = 128
//! - num_layers = 32
//! - dtype = BF16 (2 bytes)
//!
//! Per token: 8 * 128 * 2 * 2 * 32 = 128KB
//! For 2048 tokens: ~256MB

use anyhow::Result;
use candle_core::Tensor;

/// KV-cache for efficient autoregressive generation
///
/// Stores the key and value tensors from previous positions so they don't
/// need to be recomputed at each generation step. Each layer has its own
/// cache entry.
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Cached key tensors per layer: [batch, num_kv_heads, seq_len, head_dim]
    pub keys: Vec<Option<Tensor>>,
    /// Cached value tensors per layer: [batch, num_kv_heads, seq_len, head_dim]
    pub values: Vec<Option<Tensor>>,
}

impl KVCache {
    /// Create a new empty cache for the given number of layers
    pub fn new(n_layers: usize) -> Self {
        Self {
            keys: vec![None; n_layers],
            values: vec![None; n_layers],
        }
    }

    /// Get the current sequence length from the cache (0 if empty)
    pub fn seq_len(&self) -> usize {
        self.keys
            .iter()
            .find_map(|k| k.as_ref())
            .map_or(0, |k| k.dim(2).unwrap_or(0))
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.keys.iter().all(std::option::Option::is_none)
    }

    /// Get the number of layers in the cache
    pub fn n_layers(&self) -> usize {
        self.keys.len()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        for k in &mut self.keys {
            *k = None;
        }
        for v in &mut self.values {
            *v = None;
        }
    }

    /// Get mutable references to the cache for a specific layer
    ///
    /// Returns a tuple of (cache_k, cache_v) for the specified layer.
    pub fn layer_mut(&mut self, layer: usize) -> (&mut Option<Tensor>, &mut Option<Tensor>) {
        (&mut self.keys[layer], &mut self.values[layer])
    }

    /// Estimate memory usage in bytes
    ///
    /// Returns the total memory used by all cached tensors.
    pub fn memory_usage(&self) -> usize {
        let key_mem: usize = self
            .keys
            .iter()
            .filter_map(|k| k.as_ref())
            .map(|k| k.elem_count() * k.dtype().size_in_bytes())
            .sum();
        let value_mem: usize = self
            .values
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|v| v.elem_count() * v.dtype().size_in_bytes())
            .sum();
        key_mem + value_mem
    }

    /// Trim the cache to keep only the last `max_seq_len` tokens
    ///
    /// Useful for memory-constrained scenarios with long sequences.
    /// Returns Ok(true) if trimming occurred, Ok(false) if no trimming was needed.
    pub fn trim_to(&mut self, max_seq_len: usize) -> Result<bool> {
        let current_len = self.seq_len();
        if current_len <= max_seq_len {
            return Ok(false);
        }

        let trim_start = current_len - max_seq_len;

        for tensor in self.keys.iter_mut().flatten() {
            *tensor = tensor.narrow(2, trim_start, max_seq_len)?;
        }
        for tensor in self.values.iter_mut().flatten() {
            *tensor = tensor.narrow(2, trim_start, max_seq_len)?;
        }
        Ok(true)
    }

    /// Check if cache exceeds memory limit and trim if needed
    ///
    /// Trims to ~75% of current length if memory limit is exceeded.
    /// Returns Ok(true) if trimming occurred.
    pub fn enforce_memory_limit(&mut self, max_bytes: usize) -> Result<bool> {
        let current = self.memory_usage();
        if current > max_bytes {
            // Trim to ~75% of current length
            let current_len = self.seq_len();
            let target_len = (current_len * 3) / 4;
            if target_len > 0 {
                self.trim_to(target_len)?;
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_cache() {
        let cache = KVCache::new(32);
        assert_eq!(cache.n_layers(), 32);
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.memory_usage(), 0);
    }

    #[test]
    fn test_clear_cache() {
        let mut cache = KVCache::new(4);
        // Even an empty cache should clear without error
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_layer_mut() {
        let mut cache = KVCache::new(4);
        let (k, v) = cache.layer_mut(2);
        assert!(k.is_none());
        assert!(v.is_none());
    }

    #[test]
    fn test_default() {
        let cache = KVCache::default();
        assert_eq!(cache.n_layers(), 0);
        assert!(cache.is_empty());
    }
}
