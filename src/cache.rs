//! Activation cache for storing intermediate transformer states

use anyhow::Result;
use candle_core::{IndexOp, Tensor};

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

/// Stores all-position activations from a forward pass.
///
/// Unlike [`ActivationCache`] which stores only the last-token activation per layer,
/// this cache stores the full residual stream at every token position.
/// Each tensor has shape `(seq_len, d_model)`.
#[derive(Debug)]
pub struct FullActivationCache {
    /// Residual stream activations per layer, each shape (seq_len, d_model)
    activations: Vec<Tensor>,
}

impl FullActivationCache {
    /// Create an empty cache with capacity for n_layers
    pub fn with_capacity(n_layers: usize) -> Self {
        Self {
            activations: Vec::with_capacity(n_layers),
        }
    }

    /// Add a layer's all-position activation to the cache.
    ///
    /// Tensor should have shape `(seq_len, d_model)`.
    pub fn push(&mut self, tensor: Tensor) {
        self.activations.push(tensor);
    }

    /// Get the full activation tensor for a specific layer.
    ///
    /// Returns shape `(seq_len, d_model)`.
    pub fn get_layer(&self, layer: usize) -> Option<&Tensor> {
        self.activations.get(layer)
    }

    /// Get the activation at a specific layer and token position.
    ///
    /// Returns shape `(d_model,)` — compatible with `clt.encode()`.
    pub fn get_position(&self, layer: usize, position: usize) -> Result<Tensor> {
        let layer_tensor = self
            .activations
            .get(layer)
            .ok_or_else(|| anyhow::anyhow!("Layer {layer} not in cache"))?;
        let seq_len = layer_tensor.dim(0)?;
        anyhow::ensure!(
            position < seq_len,
            "Position {position} out of range (seq_len={seq_len})"
        );
        Ok(layer_tensor.i(position)?)
    }

    /// Get the number of cached layers
    pub fn n_layers(&self) -> usize {
        self.activations.len()
    }

    /// Get the sequence length (from the first layer's tensor)
    pub fn seq_len(&self) -> Result<usize> {
        let first = self
            .activations
            .first()
            .ok_or_else(|| anyhow::anyhow!("Cache is empty"))?;
        Ok(first.dim(0)?)
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.activations.is_empty()
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

    #[test]
    fn test_full_cache_basic() {
        let device = Device::Cpu;
        let seq_len = 10;
        let d_model = 2304;

        let mut cache = FullActivationCache::with_capacity(2);
        assert!(cache.is_empty());

        let t1 = Tensor::zeros((seq_len, d_model), DType::F32, &device).unwrap();
        let t2 = Tensor::zeros((seq_len, d_model), DType::F32, &device).unwrap();
        cache.push(t1);
        cache.push(t2);

        assert_eq!(cache.n_layers(), 2);
        assert_eq!(cache.seq_len().unwrap(), seq_len);
        assert!(!cache.is_empty());

        // get_layer returns 2D tensor
        let layer0 = cache.get_layer(0).unwrap();
        assert_eq!(layer0.dims(), &[seq_len, d_model]);

        // get_position returns 1D tensor
        let pos = cache.get_position(0, 5).unwrap();
        assert_eq!(pos.dims(), &[d_model]);

        // out of range
        assert!(cache.get_position(0, seq_len).is_err());
        assert!(cache.get_position(5, 0).is_err());
    }
}
