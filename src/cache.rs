//! Activation cache for storing intermediate transformer states

use anyhow::Result;
use candle_core::Tensor;

/// Stores activations from a forward pass
#[derive(Debug)]
pub struct ActivationCache {
    /// Residual stream activations per layer
    /// Each tensor is the last-token activation: shape (d_model,)
    activations: Vec<Tensor>,
}

impl ActivationCache {
    /// Create a new cache from collected activations
    pub fn new(activations: Vec<Tensor>) -> Result<Self> {
        Ok(Self { activations })
    }

    /// Create an empty cache with capacity for n_layers
    pub fn with_capacity(n_layers: usize) -> Self {
        Self {
            activations: Vec::with_capacity(n_layers),
        }
    }

    /// Add a layer's activation to the cache
    pub fn push(&mut self, tensor: Tensor) {
        self.activations.push(tensor);
    }

    /// Get activation for a specific layer
    pub fn get_layer(&self, layer: usize) -> Option<&Tensor> {
        self.activations.get(layer)
    }

    /// Get the number of cached layers
    pub fn n_layers(&self) -> usize {
        self.activations.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.activations.is_empty()
    }

    /// Get all activations
    pub fn activations(&self) -> &[Tensor] {
        &self.activations
    }

    /// Extract activations as f32 vectors
    ///
    /// Returns: Vec of (d_model,) vectors, one per layer
    pub fn to_f32_vecs(&self) -> Result<Vec<Vec<f32>>> {
        self.activations
            .iter()
            .map(|t| {
                let flat = t.flatten_all()?;
                let data: Vec<f32> = flat.to_dtype(candle_core::DType::F32)?.to_vec1()?;
                Ok(data)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_cache_basic() {
        let device = Device::Cpu;
        let t1 = Tensor::zeros((2048,), DType::F32, &device).unwrap();
        let t2 = Tensor::zeros((2048,), DType::F32, &device).unwrap();

        let cache = ActivationCache::new(vec![t1, t2]).unwrap();

        assert_eq!(cache.n_layers(), 2);
        assert!(cache.get_layer(0).is_some());
        assert!(cache.get_layer(1).is_some());
        assert!(cache.get_layer(2).is_none());
    }

    #[test]
    fn test_cache_push() {
        let device = Device::Cpu;
        let mut cache = ActivationCache::with_capacity(2);

        assert!(cache.is_empty());

        let t = Tensor::zeros((2048,), DType::F32, &device).unwrap();
        cache.push(t);

        assert_eq!(cache.n_layers(), 1);
        assert!(!cache.is_empty());
    }
}
