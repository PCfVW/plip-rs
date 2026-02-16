//! Cross-Layer Transcoder (CLT) support for Gemma 2 2B
//!
//! Loads Hanna–Piotrowski CLT weights from HuggingFace safetensors,
//! encodes residual stream activations into sparse feature activations,
//! and injects decoder vectors into the residual stream for steering.
//!
//! Memory-efficient: uses stream-and-free for encoders (~75 MB/layer on GPU)
//! and a micro-cache for steering vectors (~450 KB for 50 features).
//!
//! # CLT Architecture
//!
//! A cross-layer transcoder at layer `l` implements:
//! ```text
//! Encode:  features = ReLU(W_enc[l] @ residual_mid[l] + b_enc[l])
//! Decode:  For each downstream layer l' >= l:
//!            mlp_out_hat[l'] += W_dec[l, l'] @ features + b_dec[l']
//! Inject:  residual[pos] += strength × W_dec[l, target_layer, feature_idx, :]
//! ```
//!
//! # Weight File Layout (discovered via inspect_clt)
//!
//! Each encoder file `W_enc_{l}.safetensors` contains:
//! - `W_enc_{l}`: shape `[n_features, d_model]` (BF16) — encoder weight matrix
//! - `b_enc_{l}`: shape `[n_features]` (BF16) — encoder bias
//! - `b_dec_{l}`: shape `[d_model]` (BF16) — decoder bias for target layer l
//!
//! Each decoder file `W_dec_{l}.safetensors` contains:
//! - `W_dec_{l}`: shape `[n_features, n_target_layers, d_model]` (BF16)
//!   where n_target_layers = n_layers - l (layer l writes to layers l..n_layers-1)

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use hf_hub::{api::sync::Api, Repo, RepoType};
use safetensors::tensor::SafeTensors;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::cache::ActivationCache;
use crate::intervention::CltInjectionSpec;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Identifies a single CLT feature by its source layer and index within that layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CltFeatureId {
    /// Source layer where this feature's encoder lives (0..n_layers)
    pub layer: usize,
    /// Feature index within the layer (0..n_features_per_layer)
    pub index: usize,
}

impl std::fmt::Display for CltFeatureId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L{}:{}", self.layer, self.index)
    }
}

/// Sparse representation of CLT feature activations.
///
/// Only features with non-zero activation (after ReLU) are stored,
/// sorted by activation magnitude in descending order.
pub struct SparseActivations {
    /// Active features with their activation magnitudes, sorted descending.
    pub features: Vec<(CltFeatureId, f32)>,
}

impl SparseActivations {
    /// Number of active features.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Whether no features are active.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

/// CLT configuration auto-detected from tensor shapes.
#[derive(Debug, Clone)]
pub struct CltConfig {
    /// Number of layers in the base model (26 for Gemma 2 2B)
    pub n_layers: usize,
    /// Hidden dimension of the base model (2304 for Gemma 2 2B)
    pub d_model: usize,
    /// Number of features per encoder layer (16384 for CLT-426K)
    pub n_features_per_layer: usize,
    /// Total feature count across all layers
    pub n_features_total: usize,
    /// Base model name from config.yaml
    pub model_name: String,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Currently loaded encoder weights on GPU.
struct LoadedEncoder {
    layer: usize,
    w_enc: Tensor, // [n_features, d_model]
    b_enc: Tensor, // [n_features]
}

// ---------------------------------------------------------------------------
// CrossLayerTranscoder
// ---------------------------------------------------------------------------

/// Cross-Layer Transcoder for Gemma 2 2B.
///
/// Loads CLT encoder/decoder weights on-demand from HuggingFace safetensors,
/// with memory-efficient streaming (only one encoder on GPU at a time)
/// and a micro-cache for steering vectors.
///
/// Downloads are lazy: `open()` only fetches config and the first encoder
/// for dimension detection. Subsequent files are downloaded as needed by
/// `load_encoder()`, `decoder_vector()`, and `cache_steering_vectors()`.
pub struct CrossLayerTranscoder {
    /// HuggingFace repository ID for on-demand downloads
    clt_repo: String,
    /// Local paths to already-downloaded encoder files (None = not yet downloaded)
    encoder_paths: Vec<Option<PathBuf>>,
    /// Local paths to already-downloaded decoder files (None = not yet downloaded)
    decoder_paths: Vec<Option<PathBuf>>,
    /// Auto-detected configuration
    config: CltConfig,
    /// Currently loaded encoder (stream-and-free: only one at a time)
    loaded_encoder: Option<LoadedEncoder>,
    /// Micro-cache: pre-extracted steering vectors pinned on GPU.
    /// Key: (feature_id, target_layer), Value: decoder vector [d_model] on device.
    steering_cache: HashMap<(CltFeatureId, usize), Tensor>,
}

impl CrossLayerTranscoder {
    /// Open a CLT from HuggingFace and detect its configuration.
    ///
    /// Only downloads `config.yaml` and `W_enc_0.safetensors` (~75 MB).
    /// All other encoder/decoder files are downloaded lazily on first use
    /// by `load_encoder()`, `decoder_vector()`, or `cache_steering_vectors()`.
    ///
    /// # Arguments
    /// * `clt_repo` - HuggingFace repository ID (e.g., `"mntss/clt-gemma-2-2b-426k"`)
    pub fn open(clt_repo: &str) -> Result<Self> {
        let api = Api::new().context("Failed to create HuggingFace API")?;
        let repo = api.repo(Repo::new(clt_repo.to_string(), RepoType::Model));

        // Parse config.yaml (simple string matching, no serde_yaml dep)
        let model_name = match repo.get("config.yaml") {
            Ok(path) => {
                let text = std::fs::read_to_string(&path)?;
                parse_yaml_value(&text, "model_name").unwrap_or_else(|| "unknown".to_string())
            }
            Err(_) => "unknown".to_string(),
        };

        // Detect n_layers from repo file listing (no file downloads needed)
        let info = repo
            .info()
            .context("Failed to get repo info for layer count detection")?;
        let n_layers = info
            .siblings
            .iter()
            .filter(|s| s.rfilename.starts_with("W_enc_") && s.rfilename.ends_with(".safetensors"))
            .count();
        anyhow::ensure!(n_layers > 0, "No CLT encoder files found in {clt_repo}");

        // Download only the first encoder for dimension detection (~75 MB)
        let enc0_path = repo
            .get("W_enc_0.safetensors")
            .context("Failed to download first encoder file")?;
        let data = std::fs::read(&enc0_path).context("Failed to read first encoder file")?;
        let tensors =
            SafeTensors::deserialize(&data).context("Failed to deserialize first encoder file")?;
        let w_enc_view = tensors
            .tensor("W_enc_0")
            .context("Tensor 'W_enc_0' not found in first encoder file")?;
        let shape = w_enc_view.shape();
        anyhow::ensure!(
            shape.len() == 2,
            "Expected 2D encoder weight, got shape {shape:?}"
        );
        let n_features_per_layer = shape[0];
        let d_model = shape[1];

        // Initialize paths: only first encoder known, rest downloaded lazily
        let mut encoder_paths: Vec<Option<PathBuf>> = vec![None; n_layers];
        encoder_paths[0] = Some(enc0_path);
        let decoder_paths: Vec<Option<PathBuf>> = vec![None; n_layers];

        let config = CltConfig {
            n_layers,
            d_model,
            n_features_per_layer,
            n_features_total: n_layers * n_features_per_layer,
            model_name,
        };
        info!(
            "CLT config: {} layers, d_model={}, features_per_layer={}, total={}",
            config.n_layers, config.d_model, config.n_features_per_layer, config.n_features_total
        );

        Ok(Self {
            clt_repo: clt_repo.to_string(),
            encoder_paths,
            decoder_paths,
            config,
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        })
    }

    /// Access the auto-detected CLT configuration.
    pub fn config(&self) -> &CltConfig {
        &self.config
    }

    // --- Lazy download helpers ---

    /// Ensure the encoder file for a given layer is downloaded. Returns the path.
    fn ensure_encoder_path(&mut self, layer: usize) -> Result<PathBuf> {
        if let Some(ref path) = self.encoder_paths[layer] {
            return Ok(path.clone());
        }
        let api = Api::new()?;
        let repo = api.repo(Repo::new(self.clt_repo.clone(), RepoType::Model));
        let filename = format!("W_enc_{layer}.safetensors");
        info!("Downloading {filename} from {}", self.clt_repo);
        let path = repo
            .get(&filename)
            .with_context(|| format!("Failed to download {filename}"))?;
        self.encoder_paths[layer] = Some(path.clone());
        Ok(path)
    }

    /// Ensure the decoder file for a given layer is downloaded. Returns the path.
    fn ensure_decoder_path(&mut self, layer: usize) -> Result<PathBuf> {
        if let Some(ref path) = self.decoder_paths[layer] {
            return Ok(path.clone());
        }
        let api = Api::new()?;
        let repo = api.repo(Repo::new(self.clt_repo.clone(), RepoType::Model));
        let filename = format!("W_dec_{layer}.safetensors");
        info!("Downloading {filename} from {}", self.clt_repo);
        let path = repo
            .get(&filename)
            .with_context(|| format!("Failed to download {filename}"))?;
        self.decoder_paths[layer] = Some(path.clone());
        Ok(path)
    }

    // --- Encoder loading (stream-and-free) ---

    /// Load a single encoder's weights to the specified device.
    ///
    /// Frees any previously loaded encoder first (stream-and-free pattern).
    /// Peak GPU overhead: ~75 MB for CLT-426K, ~450 MB for CLT-2.5M.
    pub fn load_encoder(&mut self, layer: usize, device: &Device) -> Result<()> {
        anyhow::ensure!(
            layer < self.config.n_layers,
            "Layer {layer} out of range (CLT has {} layers)",
            self.config.n_layers
        );

        // Skip if already loaded
        if let Some(ref enc) = self.loaded_encoder {
            if enc.layer == layer {
                return Ok(());
            }
        }

        // Drop previous encoder (frees GPU memory)
        self.loaded_encoder = None;

        info!("Loading CLT encoder for layer {layer}");

        let enc_path = self.ensure_encoder_path(layer)?;
        let data = std::fs::read(&enc_path)
            .with_context(|| format!("Failed to read encoder file for layer {layer}"))?;
        let st = SafeTensors::deserialize(&data)
            .with_context(|| format!("Failed to deserialize encoder file for layer {layer}"))?;

        let w_enc_name = format!("W_enc_{layer}");
        let b_enc_name = format!("b_enc_{layer}");

        let w_enc = tensor_from_view(&st.tensor(&w_enc_name)?, device)?;
        let b_enc = tensor_from_view(&st.tensor(&b_enc_name)?, device)?;

        self.loaded_encoder = Some(LoadedEncoder {
            layer,
            w_enc,
            b_enc,
        });

        Ok(())
    }

    /// Check whether an encoder is currently loaded and for which layer.
    pub fn loaded_encoder_layer(&self) -> Option<usize> {
        self.loaded_encoder.as_ref().map(|e| e.layer)
    }

    // --- Encoding ---

    /// Encode a residual stream activation into sparse CLT features.
    ///
    /// The residual should be the "residual mid" activation at the given layer
    /// (after attention, before MLP). Shape: `(d_model,)`.
    ///
    /// Returns all features that pass the ReLU threshold, sorted by
    /// activation magnitude in descending order.
    ///
    /// # Requires
    /// `load_encoder(layer)` must have been called first.
    pub fn encode(&self, residual: &Tensor, layer: usize) -> Result<SparseActivations> {
        let enc = self.loaded_encoder.as_ref().ok_or_else(|| {
            anyhow::anyhow!("No encoder loaded. Call load_encoder({layer}) first")
        })?;
        anyhow::ensure!(
            enc.layer == layer,
            "Loaded encoder is for layer {}, but layer {layer} was requested",
            enc.layer
        );

        // Compute pre-activations in F32 for numerical stability
        // W_enc: [n_features, d_model], residual: [d_model]
        // pre_acts = W_enc @ residual + b_enc → [n_features]
        // Flatten to 1D in case the activation has a leading batch dimension [1, d_model]
        let residual_f32 = residual.flatten_all()?.to_dtype(DType::F32)?;
        let w_enc_f32 = enc.w_enc.to_dtype(DType::F32)?;
        let b_enc_f32 = enc.b_enc.to_dtype(DType::F32)?;

        let pre_acts = w_enc_f32.matmul(&residual_f32.unsqueeze(1)?)?.squeeze(1)?;
        let pre_acts = (&pre_acts + &b_enc_f32)?;

        // ReLU activation
        let acts = pre_acts.relu()?;

        // Transfer to CPU for sparse extraction
        let acts_vec: Vec<f32> = acts.to_vec1()?;

        let mut features: Vec<(CltFeatureId, f32)> = acts_vec
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.0)
            .map(|(i, &v)| (CltFeatureId { layer, index: i }, v))
            .collect();

        // Sort by activation magnitude (descending)
        features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(SparseActivations { features })
    }

    /// Encode and return only the top-k most active features.
    ///
    /// # Requires
    /// `load_encoder(layer)` must have been called first.
    pub fn top_k(&self, residual: &Tensor, layer: usize, k: usize) -> Result<SparseActivations> {
        let mut sparse = self.encode(residual, layer)?;
        sparse.features.truncate(k);
        Ok(sparse)
    }

    // --- Decoder access ---

    /// Extract a single feature's decoder vector for a target downstream layer.
    ///
    /// Loads from safetensors on demand. Returns shape `(d_model,)` on the
    /// specified device. Checks the steering cache first to avoid redundant loads.
    ///
    /// # Arguments
    /// * `feature` - The CLT feature to extract the decoder for
    /// * `target_layer` - The downstream layer to decode to (must be >= feature.layer)
    /// * `device` - Device to place the resulting tensor on
    pub fn decoder_vector(
        &mut self,
        feature: &CltFeatureId,
        target_layer: usize,
        device: &Device,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            feature.layer < self.config.n_layers,
            "Feature source layer {} out of range",
            feature.layer
        );
        anyhow::ensure!(
            target_layer >= feature.layer && target_layer < self.config.n_layers,
            "Target layer {target_layer} must be >= source layer {} and < {}",
            feature.layer,
            self.config.n_layers
        );
        anyhow::ensure!(
            feature.index < self.config.n_features_per_layer,
            "Feature index {} out of range (max {})",
            feature.index,
            self.config.n_features_per_layer
        );

        // Check steering cache first
        let cache_key = (*feature, target_layer);
        if let Some(cached) = self.steering_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Compute target layer offset within the decoder tensor
        // W_dec_l has shape [n_features, n_layers - l, d_model]
        // target_layer_offset = target_layer - feature.layer
        let target_offset = target_layer - feature.layer;

        // Load the full decoder tensor to CPU, slice the needed vector
        let dec_path = self.ensure_decoder_path(feature.layer)?;
        let data = std::fs::read(&dec_path)
            .with_context(|| format!("Failed to read decoder file for layer {}", feature.layer))?;
        let st = SafeTensors::deserialize(&data).with_context(|| {
            format!(
                "Failed to deserialize decoder file for layer {}",
                feature.layer
            )
        })?;

        let dec_name = format!("W_dec_{}", feature.layer);
        let w_dec_view = st.tensor(&dec_name)?;

        // w_dec shape: [n_features, n_target_layers, d_model]
        // We need w_dec[feature.index, target_offset, :] → [d_model]
        let w_dec = tensor_from_view(&w_dec_view, &Device::Cpu)?;
        let column = w_dec.i((feature.index, target_offset))?;

        // Transfer to target device
        let column = column.to_device(device)?;

        Ok(column)
    }

    // --- Micro-cache ---

    /// Pre-load decoder vectors into the steering micro-cache.
    ///
    /// Each entry is a `(CltFeatureId, target_layer)` pair. Vectors are
    /// loaded to the specified device and kept pinned for repeated injection.
    ///
    /// Memory: 50 features × 2304 × 4 bytes = ~450 KB (negligible).
    pub fn cache_steering_vectors(
        &mut self,
        features: &[(CltFeatureId, usize)],
        device: &Device,
    ) -> Result<()> {
        // Group by source layer to batch decoder file reads
        let mut by_source: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (fid, target_layer) in features {
            by_source
                .entry(fid.layer)
                .or_default()
                .push((fid.index, *target_layer));
        }

        let mut loaded = 0usize;
        let n_source_layers = by_source.len();
        for (layer_idx, (source_layer, entries)) in by_source.iter().enumerate() {
            info!(
                "cache_steering_vectors: loading decoder for source layer {} ({}/{})",
                source_layer,
                layer_idx + 1,
                n_source_layers
            );
            // Group by target_layer to identify needed offsets
            let mut by_target: HashMap<usize, Vec<usize>> = HashMap::new();
            for &(index, target_layer) in entries {
                by_target.entry(target_layer).or_default().push(index);
            }

            // Load decoder file, extract needed columns as independent CPU
            // tensors, then drop the large file data BEFORE any GPU transfer.
            // This prevents OOM when cosine scoring selects features from many
            // layers (early-layer decoders can be >1.6 GB each).
            let mut cpu_columns: Vec<(CltFeatureId, usize, Tensor)> = Vec::new();
            {
                let dec_path = self.ensure_decoder_path(*source_layer)?;
                let data = std::fs::read(&dec_path).with_context(|| {
                    format!("Failed to read decoder file for layer {source_layer}")
                })?;
                info!(
                    "cache_steering_vectors: loaded {} MB for layer {}",
                    data.len() / (1024 * 1024),
                    source_layer
                );
                let st = SafeTensors::deserialize(&data)?;
                let dec_name = format!("W_dec_{source_layer}");
                let w_dec = tensor_from_view(&st.tensor(&dec_name)?, &Device::Cpu)?;

                for (target_layer, indices) in &by_target {
                    let target_offset = target_layer - source_layer;
                    for &index in indices {
                        let fid = CltFeatureId {
                            layer: *source_layer,
                            index,
                        };
                        let cache_key = (fid, *target_layer);
                        if !self.steering_cache.contains_key(&cache_key) {
                            // Extract as independent F32 tensor: to_dtype +
                            // to_vec1 copies data OUT of candle's Arc storage,
                            // so dropping w_dec truly frees the ~1.6 GB decoder.
                            let view = w_dec.i((index, target_offset))?;
                            let dims = view.dims().to_vec();
                            let values = view.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                            let independent =
                                Tensor::from_vec(values, dims.as_slice(), &Device::Cpu)?;
                            cpu_columns.push((fid, *target_layer, independent));
                        }
                    }
                }
                // data, st, w_dec all drop here — freeing the large decoder file
            }

            // Now move the small independent columns to GPU
            for (fid, target_layer, cpu_tensor) in cpu_columns {
                let cache_key = (fid, target_layer);
                if let std::collections::hash_map::Entry::Vacant(e) =
                    self.steering_cache.entry(cache_key)
                {
                    let gpu_tensor = cpu_tensor.to_device(device)?;
                    e.insert(gpu_tensor);
                    loaded += 1;
                }
            }
        }

        info!(
            "Cached {loaded} new steering vectors ({} total in cache)",
            self.steering_cache.len()
        );
        Ok(())
    }

    /// Cache steering vectors for ALL downstream layers of each feature.
    ///
    /// For each feature at source layer `l`, caches decoder vectors for every
    /// downstream target layer `l..n_layers` (offsets `0..n_layers - l`).
    /// This enables multi-layer "clamping" injection where the steering signal
    /// propagates through all downstream transformer layers, matching how CLT
    /// features naturally operate.
    ///
    /// Same OOM-safe pattern as [`cache_steering_vectors()`][Self::cache_steering_vectors]:
    /// loads each decoder file once per source layer, extracts rows as independent
    /// F32 tensors, drops the large file, then moves small tensors to GPU.
    ///
    /// # Arguments
    /// * `features` - Feature IDs to cache (no target_layer needed — all downstream layers are cached)
    /// * `device` - Device to store cached tensors on (typically GPU)
    pub fn cache_steering_vectors_all_downstream(
        &mut self,
        features: &[CltFeatureId],
        device: &Device,
    ) -> Result<()> {
        let n_layers = self.config.n_layers;

        // Group by source layer to batch decoder file reads
        let mut by_source: HashMap<usize, Vec<usize>> = HashMap::new();
        for fid in features {
            anyhow::ensure!(
                fid.layer < n_layers,
                "Feature source layer {} out of range (max {})",
                fid.layer,
                n_layers - 1
            );
            by_source.entry(fid.layer).or_default().push(fid.index);
        }

        let mut loaded = 0usize;
        let n_source_layers = by_source.len();
        for (layer_idx, (source_layer, indices)) in by_source.iter().enumerate() {
            let n_target_layers = n_layers - source_layer;
            info!(
                "cache_steering_vectors_all_downstream: loading decoder for source layer {} \
                 ({}/{}, {} downstream layers)",
                source_layer,
                layer_idx + 1,
                n_source_layers,
                n_target_layers
            );

            // Load decoder file, extract ALL offsets as independent CPU tensors, then drop
            let mut cpu_columns: Vec<(CltFeatureId, usize, Tensor)> = Vec::new();
            {
                let dec_path = self.ensure_decoder_path(*source_layer)?;
                let data = std::fs::read(&dec_path).with_context(|| {
                    format!("Failed to read decoder file for layer {source_layer}")
                })?;
                info!(
                    "cache_steering_vectors_all_downstream: loaded {} MB for layer {}",
                    data.len() / (1024 * 1024),
                    source_layer
                );
                let st = SafeTensors::deserialize(&data)?;
                let dec_name = format!("W_dec_{source_layer}");
                let w_dec = tensor_from_view(&st.tensor(&dec_name)?, &Device::Cpu)?;

                for &index in indices {
                    let fid = CltFeatureId {
                        layer: *source_layer,
                        index,
                    };
                    // Extract decoder vector for EVERY downstream layer
                    for target_offset in 0..n_target_layers {
                        let target_layer = source_layer + target_offset;
                        let cache_key = (fid, target_layer);
                        if !self.steering_cache.contains_key(&cache_key) {
                            let view = w_dec.i((index, target_offset))?;
                            let dims = view.dims().to_vec();
                            let values = view.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                            let independent =
                                Tensor::from_vec(values, dims.as_slice(), &Device::Cpu)?;
                            cpu_columns.push((fid, target_layer, independent));
                        }
                    }
                }
                // data, st, w_dec all drop here — freeing the large decoder file
            }

            // Move small independent columns to GPU
            for (fid, target_layer, cpu_tensor) in cpu_columns {
                let cache_key = (fid, target_layer);
                if let std::collections::hash_map::Entry::Vacant(e) =
                    self.steering_cache.entry(cache_key)
                {
                    let gpu_tensor = cpu_tensor.to_device(device)?;
                    e.insert(gpu_tensor);
                    loaded += 1;
                }
            }
        }

        info!(
            "Cached {loaded} new steering vectors across all downstream layers ({} total in cache)",
            self.steering_cache.len()
        );
        Ok(())
    }

    /// Clear all cached steering vectors, freeing device memory.
    pub fn clear_steering_cache(&mut self) {
        let count = self.steering_cache.len();
        self.steering_cache.clear();
        if count > 0 {
            info!("Cleared {count} steering vectors from cache");
        }
    }

    /// Number of vectors currently in the steering cache.
    pub fn steering_cache_len(&self) -> usize {
        self.steering_cache.len()
    }

    // --- Injection ---

    /// Inject cached steering vectors into the residual stream at a position.
    ///
    /// Returns a new tensor with the injection applied:
    /// `residual[:, position, :] += strength * Σ decoder_vectors`
    ///
    /// # Arguments
    /// * `residual` - Hidden states, shape `(batch, seq_len, d_model)`
    /// * `features` - List of `(feature, target_layer)` pairs to inject (must be cached)
    /// * `position` - Token position in the sequence to inject at
    /// * `strength` - Scalar multiplier for the steering vectors
    pub fn inject(
        &self,
        residual: &Tensor,
        features: &[(CltFeatureId, usize)],
        position: usize,
        strength: f32,
    ) -> Result<Tensor> {
        let (batch, seq_len, d_model) = residual.dims3()?;
        anyhow::ensure!(
            position < seq_len,
            "Injection position {position} out of range (seq_len={seq_len})"
        );
        anyhow::ensure!(
            d_model == self.config.d_model,
            "Residual d_model={d_model} doesn't match CLT d_model={}",
            self.config.d_model
        );

        // Accumulate all steering vectors into one vector (F32 for stability)
        let mut accumulated = Tensor::zeros((d_model,), DType::F32, residual.device())?;
        for (feature, target_layer) in features {
            let cache_key = (*feature, *target_layer);
            let cached = self.steering_cache.get(&cache_key).ok_or_else(|| {
                anyhow::anyhow!(
                    "Feature {feature} for target layer {target_layer} not in steering cache"
                )
            })?;
            let cached_f32 = cached.to_dtype(DType::F32)?;
            accumulated = (&accumulated + &cached_f32)?;
        }

        // Scale by strength
        let accumulated = (accumulated * f64::from(strength))?;

        // Convert to residual dtype
        let accumulated = accumulated.to_dtype(residual.dtype())?;

        // Build steering tensor: zeros everywhere, accumulated at [batch, position, :]
        // Use narrow + cat to assemble the modified residual
        let pos_slice = residual.narrow(1, position, 1)?; // [batch, 1, d_model]
        let steering_expanded = accumulated
            .unsqueeze(0)?
            .unsqueeze(0)?
            .expand((batch, 1, d_model))?; // [batch, 1, d_model]
        let pos_updated = (&pos_slice + &steering_expanded)?;

        // Reassemble: before + updated_position + after
        let mut parts: Vec<Tensor> = Vec::with_capacity(3);
        if position > 0 {
            parts.push(residual.narrow(1, 0, position)?);
        }
        parts.push(pos_updated);
        if position + 1 < seq_len {
            parts.push(residual.narrow(1, position + 1, seq_len - position - 1)?);
        }

        let result = Tensor::cat(&parts, 1)?;
        Ok(result)
    }

    /// Prepare a [`CltInjectionSpec`] from cached steering vectors.
    ///
    /// Looks up each feature's decoder vector from the micro-cache,
    /// accumulates per target layer, and scales by `strength`.
    /// The resulting spec can be passed to `forward_with_clt_injection()`
    /// without needing `&mut self`.
    ///
    /// Features must have been previously cached via [`cache_steering_vectors()`][Self::cache_steering_vectors].
    ///
    /// # Arguments
    /// * `features` - List of `(feature_id, target_layer)` pairs
    /// * `position` - Token position in the sequence to inject at
    /// * `strength` - Scalar multiplier for the accumulated steering vectors
    pub fn prepare_injection(
        &self,
        features: &[(CltFeatureId, usize)],
        position: usize,
        strength: f32,
    ) -> Result<CltInjectionSpec> {
        // Group features by target layer and accumulate their decoder vectors
        let mut per_layer: HashMap<usize, Tensor> = HashMap::new();
        for (feature, target_layer) in features {
            let cache_key = (*feature, *target_layer);
            let cached = self.steering_cache.get(&cache_key).ok_or_else(|| {
                anyhow::anyhow!(
                    "Feature {feature} for target layer {target_layer} not in steering cache. \
                     Call cache_steering_vectors() first."
                )
            })?;
            let cached_f32 = cached.to_dtype(DType::F32)?;
            if let Some(acc) = per_layer.get_mut(target_layer) {
                *acc = (acc as &Tensor + &cached_f32)?;
            } else {
                per_layer.insert(*target_layer, cached_f32);
            }
        }

        // Scale by strength and build the spec
        let mut spec = CltInjectionSpec::new();
        for (target_layer, accumulated) in per_layer {
            let scaled = (accumulated * f64::from(strength))?;
            spec.add(target_layer, position, scaled);
        }

        Ok(spec)
    }

    /// Inject cached steering vectors into a single-position residual.
    ///
    /// Convenience method for generation (where residual is shape `(d_model,)`
    /// for a single token). Returns the modified residual.
    pub fn inject_single(
        &self,
        residual: &Tensor,
        features: &[(CltFeatureId, usize)],
        strength: f32,
    ) -> Result<Tensor> {
        let d_model = residual.dim(0)?;
        anyhow::ensure!(
            d_model == self.config.d_model,
            "Residual d_model={d_model} doesn't match CLT d_model={}",
            self.config.d_model
        );

        let mut accumulated = Tensor::zeros((d_model,), DType::F32, residual.device())?;
        for (feature, target_layer) in features {
            let cache_key = (*feature, *target_layer);
            let cached = self.steering_cache.get(&cache_key).ok_or_else(|| {
                anyhow::anyhow!(
                    "Feature {feature} for target layer {target_layer} not in steering cache"
                )
            })?;
            let cached_f32 = cached.to_dtype(DType::F32)?;
            accumulated = (&accumulated + &cached_f32)?;
        }

        let accumulated = (accumulated * f64::from(strength))?;
        let accumulated = accumulated.to_dtype(residual.dtype())?;
        let result = (residual + &accumulated)?;
        Ok(result)
    }

    // --- Planning-site feature identification (Method 3) ---

    /// Identify planning-relevant features from a single prompt's activations.
    ///
    /// Method 3 (Planning-site activation + decoder filtering):
    /// 1. Encode hidden states through the CLT encoder at each layer
    /// 2. Collect features with activation above threshold
    /// 3. For each active feature, compute cosine similarity between its
    ///    final-layer decoder vector and the target word embedding
    /// 4. Return top-K features ranked by cosine score
    ///
    /// This combines context sensitivity (features the model actually fires
    /// at the planning site) with directional specificity (features whose
    /// decoder vectors point toward the target word).
    ///
    /// Memory-efficient: loads encoders and decoders sequentially (stream-and-free).
    /// Loads 26 encoders + up to 26 decoders per call.
    ///
    /// # Arguments
    /// * `activations` - Per-layer last-token activations from `model.get_activations()`
    /// * `target_embedding` - Target word embedding `[d_model]` from `model.token_embedding()`
    /// * `top_k` - Number of top features to return
    /// * `activation_threshold` - Minimum activation to consider (0.0 = any ReLU-positive)
    /// * `device` - Device for encoder computation
    pub fn identify_planning_features(
        &mut self,
        activations: &ActivationCache,
        target_embedding: &Tensor,
        top_k: usize,
        activation_threshold: f32,
        device: &Device,
    ) -> Result<Vec<(CltFeatureId, f32)>> {
        let d_model = self.config.d_model;
        let n_layers = self.config.n_layers;
        let final_layer = n_layers - 1;

        anyhow::ensure!(
            target_embedding.dims() == [d_model],
            "Target embedding must have shape [{d_model}], got {:?}",
            target_embedding.dims()
        );

        // Phase 1: Encode all layers to find active features
        let mut all_active: Vec<(CltFeatureId, f32)> = Vec::new();
        for layer in 0..n_layers {
            let residual = activations
                .get_layer(layer)
                .ok_or_else(|| anyhow::anyhow!("No activation at layer {layer}"))?;
            self.load_encoder(layer, device)?;
            let sparse = self.encode(residual, layer)?;
            for (fid, act) in &sparse.features {
                if *act > activation_threshold {
                    all_active.push((*fid, *act));
                }
            }
        }

        info!(
            "Planning features: {} active features across {} layers (threshold={activation_threshold})",
            all_active.len(),
            n_layers
        );

        if all_active.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 2: Score active features by decoder cosine similarity
        let target_f32 = target_embedding
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        let target_norm_sq: f32 = target_f32.sqr()?.sum_all()?.to_scalar()?;
        let target_norm = target_norm_sq.sqrt();

        // Group active features by source layer
        let mut by_source: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for (fid, act) in &all_active {
            by_source
                .entry(fid.layer)
                .or_default()
                .push((fid.index, *act));
        }

        let mut scored: Vec<(CltFeatureId, f32)> = Vec::new();

        for (source_layer, entries) in &by_source {
            // Skip if this source layer can't decode to final layer
            if final_layer < *source_layer {
                continue;
            }
            let target_offset = final_layer - source_layer;

            // Load decoder file to CPU, extract needed rows, drop file
            let dec_path = self.ensure_decoder_path(*source_layer)?;
            let data = std::fs::read(&dec_path)
                .with_context(|| format!("Failed to read decoder file for layer {source_layer}"))?;
            let st = SafeTensors::deserialize(&data)?;
            let dec_name = format!("W_dec_{source_layer}");
            let w_dec = tensor_from_view(&st.tensor(&dec_name)?, &Device::Cpu)?;
            let w_dec_f32 = w_dec.to_dtype(DType::F32)?;

            for &(index, _act) in entries {
                let dec_row = w_dec_f32.i((index, target_offset))?;
                // Cosine similarity
                let dot: f32 = (&dec_row * &target_f32)?.sum_all()?.to_scalar()?;
                let dec_norm_sq: f32 = dec_row.sqr()?.sum_all()?.to_scalar()?;
                let dec_norm = dec_norm_sq.sqrt();
                let cosine = if dec_norm > 1e-10 && target_norm > 1e-10 {
                    dot / (dec_norm * target_norm)
                } else {
                    0.0
                };

                if cosine.is_finite() {
                    scored.push((
                        CltFeatureId {
                            layer: *source_layer,
                            index,
                        },
                        cosine,
                    ));
                }
            }
            // data, st, w_dec, w_dec_f32 drop here
        }

        // Phase 3: Sort by cosine score descending, take top-K
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        info!(
            "Planning features: selected top-{} (best cosine={:.4})",
            scored.len(),
            scored.first().map_or(0.0, |(_, s)| *s)
        );

        Ok(scored)
    }

    /// Extract decoder vectors for a set of features at the specified target layer.
    ///
    /// Groups features by source layer, loads each decoder file **once**,
    /// and extracts the needed vectors as independent F32 CPU tensors.
    /// Memory-efficient: drops each large decoder file immediately after extraction
    /// (same OOM-safe pattern as [`cache_steering_vectors()`][Self::cache_steering_vectors]).
    ///
    /// Use with [`encode()`][Self::encode] to implement the batch path for
    /// Method 3 (planning-site activation + decoder filtering):
    /// 1. Encode all prompts at all layers → collect active features
    /// 2. Call this method to extract decoder vectors for all unique features
    /// 3. Compute cosine similarity per (prompt, target) pair using the returned map
    ///
    /// # Arguments
    /// * `features` - Feature IDs to extract decoder vectors for
    /// * `target_layer` - Downstream layer to decode to (typically the final layer)
    ///
    /// # Returns
    /// HashMap from `CltFeatureId` to decoder vector `[d_model]` on CPU (F32).
    pub fn extract_decoder_vectors(
        &mut self,
        features: &[CltFeatureId],
        target_layer: usize,
    ) -> Result<HashMap<CltFeatureId, Tensor>> {
        anyhow::ensure!(
            target_layer < self.config.n_layers,
            "Target layer {target_layer} out of range (max {})",
            self.config.n_layers - 1
        );

        // Group by source layer
        let mut by_source: HashMap<usize, Vec<usize>> = HashMap::new();
        for fid in features {
            anyhow::ensure!(
                fid.layer < self.config.n_layers,
                "Feature source layer {} out of range",
                fid.layer
            );
            anyhow::ensure!(
                target_layer >= fid.layer,
                "Target layer {target_layer} must be >= source layer {}",
                fid.layer
            );
            by_source.entry(fid.layer).or_default().push(fid.index);
        }

        let mut result: HashMap<CltFeatureId, Tensor> = HashMap::new();
        let n_source_layers = by_source.len();

        for (layer_idx, (source_layer, indices)) in by_source.iter().enumerate() {
            info!(
                "extract_decoder_vectors: loading decoder for source layer {} ({}/{})",
                source_layer,
                layer_idx + 1,
                n_source_layers
            );
            let target_offset = target_layer - source_layer;

            // Load decoder file to CPU, extract needed rows as independent tensors
            let dec_path = self.ensure_decoder_path(*source_layer)?;
            let data = std::fs::read(&dec_path)
                .with_context(|| format!("Failed to read decoder file for layer {source_layer}"))?;
            let st = SafeTensors::deserialize(&data)?;
            let dec_name = format!("W_dec_{source_layer}");
            let w_dec = tensor_from_view(&st.tensor(&dec_name)?, &Device::Cpu)?;

            for &index in indices {
                let fid = CltFeatureId {
                    layer: *source_layer,
                    index,
                };
                if let std::collections::hash_map::Entry::Vacant(e) = result.entry(fid) {
                    // Extract as independent F32 tensor (same pattern as cache_steering_vectors)
                    let view = w_dec.i((index, target_offset))?;
                    let dims = view.dims().to_vec();
                    let values = view.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                    let independent = Tensor::from_vec(values, dims.as_slice(), &Device::Cpu)?;
                    e.insert(independent);
                }
            }
            // data, st, w_dec drop here — freeing the large decoder file
        }

        info!(
            "Extracted {} decoder vectors across {} source layers",
            result.len(),
            n_source_layers
        );

        Ok(result)
    }

    // --- Batch decoder scoring ---

    /// Score all CLT features by how strongly their decoder vector at `target_layer`
    /// projects along a given direction (e.g., a target token's embedding vector).
    ///
    /// For each source layer, loads the full decoder file to CPU, extracts the
    /// `target_layer` slice `[n_features, d_model]`, and computes
    /// `scores = slice @ direction` → `[n_features]`. Returns the top-K features
    /// globally (across all source layers), sorted by score descending.
    ///
    /// This is the core operation for Method 2 (decoder projection) in §2.3.3:
    /// find features whose decoders have high logit weight for a target token.
    ///
    /// # Arguments
    /// * `direction` - `[d_model]` vector to project decoders along (e.g., token embedding)
    /// * `target_layer` - Downstream layer to examine decoders at (typically the final layer)
    /// * `top_k` - Number of top-scoring features to return
    ///
    /// # Memory
    /// Processes one decoder file at a time on CPU (up to ~2 GB for layer 0).
    /// No GPU memory required.
    pub fn score_features_by_decoder_projection(
        &mut self,
        direction: &Tensor,
        target_layer: usize,
        top_k: usize,
        cosine: bool,
    ) -> Result<Vec<(CltFeatureId, f32)>> {
        let d_model = self.config.d_model;
        anyhow::ensure!(
            direction.dims() == [d_model],
            "Direction vector must have shape [{d_model}], got {:?}",
            direction.dims()
        );
        anyhow::ensure!(
            target_layer < self.config.n_layers,
            "Target layer {target_layer} out of range (max {})",
            self.config.n_layers - 1
        );

        let direction_f32 = direction.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

        // Optionally normalize direction to unit length for cosine similarity
        let direction_norm = if cosine {
            let norm = direction_f32.sqr()?.sum_all()?.sqrt()?;
            let norm_val: f32 = norm.to_scalar()?;
            if norm_val > 1e-10 {
                direction_f32.broadcast_div(&Tensor::new(norm_val, &Device::Cpu)?)?
            } else {
                direction_f32.clone()
            }
        } else {
            direction_f32.clone()
        };

        let mut all_scores: Vec<(CltFeatureId, f32)> = Vec::new();

        for source_layer in 0..self.config.n_layers {
            if target_layer < source_layer {
                continue; // This source layer cannot decode to target_layer
            }
            let target_offset = target_layer - source_layer;

            // Load decoder file to CPU
            let dec_path = self.ensure_decoder_path(source_layer)?;
            let data = std::fs::read(&dec_path)
                .with_context(|| format!("Failed to read decoder file for layer {source_layer}"))?;
            let st = SafeTensors::deserialize(&data).with_context(|| {
                format!("Failed to deserialize decoder file for layer {source_layer}")
            })?;

            let dec_name = format!("W_dec_{source_layer}");
            let w_dec_view = st.tensor(&dec_name)?;
            // Shape: [n_features, n_target_layers, d_model]
            let w_dec = tensor_from_view(&w_dec_view, &Device::Cpu)?;
            let w_dec_f32 = w_dec.to_dtype(DType::F32)?;

            // Extract target layer slice: [n_features, d_model]
            let dec_slice = w_dec_f32.i((.., target_offset, ..))?;

            // raw_scores = dec_slice @ direction_norm → [n_features]
            let raw_scores = dec_slice
                .matmul(&direction_norm.unsqueeze(1)?)?
                .squeeze(1)?;

            let scores_vec: Vec<f32> = if cosine {
                // Divide by each decoder row's L2 norm → cosine similarity
                let dec_norms = dec_slice.sqr()?.sum(1)?.sqrt()?;
                let cosine_scores = raw_scores.broadcast_div(&dec_norms)?;
                cosine_scores.to_vec1()?
            } else {
                raw_scores.to_vec1()?
            };

            for (idx, &score) in scores_vec.iter().enumerate() {
                if score.is_finite() {
                    all_scores.push((
                        CltFeatureId {
                            layer: source_layer,
                            index: idx,
                        },
                        score,
                    ));
                }
            }

            info!(
                "Scored {} features at source layer {source_layer} (target layer {target_layer})",
                scores_vec.len()
            );
        }

        // Sort by score descending, take top-K
        all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all_scores.truncate(top_k);

        Ok(all_scores)
    }

    /// Batch version of [`score_features_by_decoder_projection`].
    ///
    /// Scores multiple direction vectors against all decoder files in a single
    /// pass. Each decoder file is loaded **once** for all directions, reducing
    /// I/O from `n_words × n_layers` file reads to just `n_layers`.
    ///
    /// Returns one `Vec<(CltFeatureId, f32)>` per direction (top-K per word).
    pub fn score_features_by_decoder_projection_batch(
        &mut self,
        directions: &[Tensor],
        target_layer: usize,
        top_k: usize,
        cosine: bool,
    ) -> Result<Vec<Vec<(CltFeatureId, f32)>>> {
        let d_model = self.config.d_model;
        let n_words = directions.len();
        anyhow::ensure!(n_words > 0, "At least one direction vector required");
        for (i, dir) in directions.iter().enumerate() {
            anyhow::ensure!(
                dir.dims() == [d_model],
                "Direction vector {i} must have shape [{d_model}], got {:?}",
                dir.dims()
            );
        }
        anyhow::ensure!(
            target_layer < self.config.n_layers,
            "Target layer {target_layer} out of range (max {})",
            self.config.n_layers - 1
        );

        // Stack directions to [n_words, d_model] on CPU
        let dirs_f32: Vec<Tensor> = directions
            .iter()
            .map(|d| Ok(d.to_dtype(DType::F32)?.to_device(&Device::Cpu)?) as Result<Tensor>)
            .collect::<Result<_>>()?;
        let stacked = Tensor::stack(&dirs_f32, 0)?; // [n_words, d_model]

        // For cosine: normalize direction vectors to unit length
        let stacked_norm = if cosine {
            let norms = stacked.sqr()?.sum(1)?.sqrt()?; // [n_words]
                                                        // Clamp norms to avoid division by zero
            let ones = Tensor::ones_like(&norms)?;
            let safe_norms = norms.maximum(&(&ones * 1e-10f64)?)?; // [n_words]
            stacked.broadcast_div(&safe_norms.unsqueeze(1)?)?
        } else {
            stacked
        };
        let directions_t = stacked_norm.t()?; // [d_model, n_words]

        // Per-word score accumulators
        let mut all_scores: Vec<Vec<(CltFeatureId, f32)>> =
            (0..n_words).map(|_| Vec::new()).collect();

        for source_layer in 0..self.config.n_layers {
            if target_layer < source_layer {
                continue;
            }
            let target_offset = target_layer - source_layer;

            // Load decoder file ONCE for all words
            let dec_path = self.ensure_decoder_path(source_layer)?;
            let data = std::fs::read(&dec_path)
                .with_context(|| format!("Failed to read decoder file for layer {source_layer}"))?;
            let st = SafeTensors::deserialize(&data).with_context(|| {
                format!("Failed to deserialize decoder file for layer {source_layer}")
            })?;
            let dec_name = format!("W_dec_{source_layer}");
            let w_dec = tensor_from_view(&st.tensor(&dec_name)?, &Device::Cpu)?;
            let w_dec_f32 = w_dec.to_dtype(DType::F32)?;
            let dec_slice = w_dec_f32.i((.., target_offset, ..))?; // [n_features, d_model]

            // Batch matmul: [n_features, d_model] × [d_model, n_words] = [n_features, n_words]
            let raw_scores = dec_slice.matmul(&directions_t)?;

            // Transpose to [n_words, n_features] for easy extraction
            let scores_2d: Vec<Vec<f32>> = if cosine {
                let dec_norms = dec_slice.sqr()?.sum(1)?.sqrt()?; // [n_features]
                let cosine_scores = raw_scores.broadcast_div(&dec_norms.unsqueeze(1)?)?;
                cosine_scores.t()?.to_vec2()?
            } else {
                raw_scores.t()?.to_vec2()?
            };

            for (w, word_scores) in scores_2d.iter().enumerate() {
                for (idx, &score) in word_scores.iter().enumerate() {
                    if score.is_finite() {
                        all_scores[w].push((
                            CltFeatureId {
                                layer: source_layer,
                                index: idx,
                            },
                            score,
                        ));
                    }
                }
            }

            info!(
                "Batch scored {} words × {} features at source layer {source_layer} (target layer {target_layer})",
                n_words,
                scores_2d.first().map_or(0, Vec::len)
            );
        }

        // Sort and truncate per word
        for word_scores in &mut all_scores {
            word_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            word_scores.truncate(top_k);
        }

        Ok(all_scores)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Parse a simple YAML `key: "value"` pair. No serde_yaml dependency needed.
fn parse_yaml_value(yaml_text: &str, key: &str) -> Option<String> {
    for line in yaml_text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key) {
            if let Some(rest) = rest.strip_prefix(':') {
                let value = rest.trim().trim_matches('"');
                return Some(value.to_string());
            }
        }
    }
    None
}

/// Convert a safetensors `TensorView` to a candle `Tensor` on the given device.
fn tensor_from_view(view: &safetensors::tensor::TensorView<'_>, device: &Device) -> Result<Tensor> {
    let shape: Vec<usize> = view.shape().to_vec();
    let dtype = match view.dtype() {
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::F32 => DType::F32,
        other => anyhow::bail!("Unsupported CLT tensor dtype: {other:?}"),
    };
    let tensor = Tensor::from_raw_buffer(view.data(), dtype, &shape, device)?;
    Ok(tensor)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Tier 1: Pure unit tests (no GPU, no downloads) ---

    #[test]
    fn test_feature_id_equality() {
        let a = CltFeatureId {
            layer: 5,
            index: 100,
        };
        let b = CltFeatureId {
            layer: 5,
            index: 100,
        };
        let c = CltFeatureId {
            layer: 5,
            index: 101,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_feature_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CltFeatureId {
            layer: 0,
            index: 42,
        });
        set.insert(CltFeatureId {
            layer: 0,
            index: 42,
        });
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_feature_id_display() {
        let fid = CltFeatureId {
            layer: 12,
            index: 4231,
        };
        assert_eq!(format!("{fid}"), "L12:4231");
    }

    #[test]
    fn test_sparse_activations_empty() {
        let sparse = SparseActivations { features: vec![] };
        assert!(sparse.is_empty());
        assert_eq!(sparse.len(), 0);
    }

    #[test]
    fn test_parse_yaml_value_basic() {
        let yaml = "model_name: \"google/gemma-2-2b\"\nmodel_kind: \"cross_layer_transcoder\"";
        assert_eq!(
            parse_yaml_value(yaml, "model_name"),
            Some("google/gemma-2-2b".to_string())
        );
        assert_eq!(
            parse_yaml_value(yaml, "model_kind"),
            Some("cross_layer_transcoder".to_string())
        );
        assert_eq!(parse_yaml_value(yaml, "nonexistent"), None);
    }

    #[test]
    fn test_parse_yaml_value_no_quotes() {
        let yaml = "model_kind: cross_layer_transcoder";
        assert_eq!(
            parse_yaml_value(yaml, "model_kind"),
            Some("cross_layer_transcoder".to_string())
        );
    }

    // --- Tier 2: Integration tests (auto-detected prerequisites) -----------
    //
    // These tests need real CLT weight files and (most of them) a CUDA GPU.
    // Instead of `#[ignore]`, they detect prerequisites at runtime:
    //
    //   - `clt_files_cached()` checks the local HuggingFace cache for
    //     `mntss/clt-gemma-2-2b-426k` files. Purely filesystem — no network.
    //   - `cuda_available()` checks for a CUDA device via candle.
    //
    // On a development machine with a GPU and cached weights (e.g. after
    // running any CLT experiment), `cargo test --lib` runs them automatically.
    // On CI or machines without GPU/weights, they skip with a message.
    //
    // This pattern is intended for the public candle-mi crate: users who have
    // the hardware and weights get full coverage; others get a clean skip
    // instead of download-triggered failures or manual `--ignored` flags.

    const CLT_REPO: &str = "mntss/clt-gemma-2-2b-426k";

    /// Check whether CLT weight files are already present in the local
    /// HuggingFace cache. Uses `hf_hub::Cache` which resolves the same
    /// `$HF_HOME` / `~/.cache/huggingface/hub/` path as the download API.
    /// No network access is performed.
    fn clt_files_cached() -> bool {
        let cache = hf_hub::Cache::default();
        let repo = cache.repo(Repo::new(CLT_REPO.to_string(), RepoType::Model));
        // W_enc_0 is the first file any test needs; if it's cached the rest
        // will be too (they're all downloaded together by experiments).
        repo.get("W_enc_0.safetensors").is_some()
    }

    /// Check whether a CUDA GPU is available to candle.
    fn cuda_available() -> bool {
        Device::cuda_if_available(0).is_ok()
    }

    #[test]
    fn test_clt_open_426k() {
        if !clt_files_cached() {
            eprintln!("  SKIP test_clt_open_426k: CLT files not in local cache");
            return;
        }
        let clt = CrossLayerTranscoder::open(CLT_REPO).unwrap();
        assert_eq!(clt.config().n_layers, 26);
        assert_eq!(clt.config().d_model, 2304);
        assert_eq!(clt.config().n_features_per_layer, 16384);
        assert_eq!(clt.config().n_features_total, 425_984);
        assert_eq!(clt.config().model_name, "google/gemma-2-2b");
    }

    #[test]
    fn test_clt_load_encoder_stream_and_free() {
        if !clt_files_cached() || !cuda_available() {
            eprintln!("  SKIP test_clt_load_encoder_stream_and_free: needs GPU + cached CLT");
            return;
        }
        let device = Device::cuda_if_available(0).unwrap();
        let mut clt = CrossLayerTranscoder::open(CLT_REPO).unwrap();

        // Load encoder 0
        clt.load_encoder(0, &device).unwrap();
        assert_eq!(clt.loaded_encoder_layer(), Some(0));

        // Load encoder 1 (should free encoder 0)
        clt.load_encoder(1, &device).unwrap();
        assert_eq!(clt.loaded_encoder_layer(), Some(1));

        // Re-loading same encoder should be a no-op
        clt.load_encoder(1, &device).unwrap();
        assert_eq!(clt.loaded_encoder_layer(), Some(1));
    }

    #[test]
    fn test_clt_encode_produces_sparse_activations() {
        if !clt_files_cached() || !cuda_available() {
            eprintln!("  SKIP test_clt_encode_produces_sparse_activations: needs GPU + cached CLT");
            return;
        }
        let device = Device::cuda_if_available(0).unwrap();
        let mut clt = CrossLayerTranscoder::open(CLT_REPO).unwrap();
        clt.load_encoder(0, &device).unwrap();

        // Random residual vector
        let residual = Tensor::randn(0.0f32, 1.0, (2304,), &device).unwrap();
        let sparse = clt.encode(&residual, 0).unwrap();

        // Should have some active features
        assert!(!sparse.is_empty(), "Expected some active features");
        // Should be sorted by magnitude descending
        for window in sparse.features.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "Features should be sorted descending"
            );
        }
        // All activations should be positive
        for (_, act) in &sparse.features {
            assert!(*act > 0.0, "ReLU activations should be positive");
        }
    }

    #[test]
    fn test_clt_top_k() {
        if !clt_files_cached() || !cuda_available() {
            eprintln!("  SKIP test_clt_top_k: needs GPU + cached CLT");
            return;
        }
        let device = Device::cuda_if_available(0).unwrap();
        let mut clt = CrossLayerTranscoder::open(CLT_REPO).unwrap();
        clt.load_encoder(0, &device).unwrap();

        let residual = Tensor::randn(0.0f32, 1.0, (2304,), &device).unwrap();
        let top5 = clt.top_k(&residual, 0, 5).unwrap();

        assert!(top5.features.len() <= 5);
    }

    #[test]
    fn test_clt_decoder_vector_shape() {
        if !clt_files_cached() || !cuda_available() {
            eprintln!("  SKIP test_clt_decoder_vector_shape: needs GPU + cached CLT");
            return;
        }
        let device = Device::cuda_if_available(0).unwrap();
        let mut clt = CrossLayerTranscoder::open(CLT_REPO).unwrap();

        let fid = CltFeatureId {
            layer: 25,
            index: 0,
        };
        let vec = clt.decoder_vector(&fid, 25, &device).unwrap();
        assert_eq!(vec.dims(), &[2304]);
    }

    #[test]
    fn test_clt_cache_and_inject() {
        if !clt_files_cached() || !cuda_available() {
            eprintln!("  SKIP test_clt_cache_and_inject: needs GPU + cached CLT");
            return;
        }
        let device = Device::cuda_if_available(0).unwrap();
        let mut clt = CrossLayerTranscoder::open(CLT_REPO).unwrap();

        let features = vec![
            (
                CltFeatureId {
                    layer: 25,
                    index: 0,
                },
                25usize,
            ),
            (
                CltFeatureId {
                    layer: 25,
                    index: 1,
                },
                25usize,
            ),
        ];
        clt.cache_steering_vectors(&features, &device).unwrap();
        assert_eq!(clt.steering_cache_len(), 2);

        // Create a dummy residual [batch=1, seq_len=5, d_model=2304]
        let residual = Tensor::zeros((1, 5, 2304), DType::F32, &device).unwrap();
        let result = clt.inject(&residual, &features, 2, 1.0).unwrap();
        assert_eq!(result.dims(), &[1, 5, 2304]);

        // The injected position should be non-zero
        let pos_slice: Vec<f32> = result
            .i((0, 2))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();
        let any_nonzero = pos_slice.iter().any(|&v| v != 0.0);
        assert!(any_nonzero, "Injection should produce non-zero values");

        // Non-injected positions should remain zero
        let other_slice: Vec<f32> = result
            .i((0, 0))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();
        let all_zero = other_slice.iter().all(|&v| v == 0.0);
        assert!(all_zero, "Non-injected positions should remain zero");

        // Test clear
        clt.clear_steering_cache();
        assert_eq!(clt.steering_cache_len(), 0);
    }

    #[test]
    fn test_clt_inject_single() {
        if !clt_files_cached() || !cuda_available() {
            eprintln!("  SKIP test_clt_inject_single: needs GPU + cached CLT");
            return;
        }
        let device = Device::cuda_if_available(0).unwrap();
        let mut clt = CrossLayerTranscoder::open(CLT_REPO).unwrap();

        let features = vec![(
            CltFeatureId {
                layer: 25,
                index: 0,
            },
            25usize,
        )];
        clt.cache_steering_vectors(&features, &device).unwrap();

        let residual = Tensor::zeros((2304,), DType::F32, &device).unwrap();
        let result = clt.inject_single(&residual, &features, 2.0).unwrap();
        assert_eq!(result.dims(), &[2304]);

        let vals: Vec<f32> = result.to_vec1().unwrap();
        let any_nonzero = vals.iter().any(|&v| v != 0.0);
        assert!(any_nonzero, "Injection should produce non-zero values");
    }

    #[test]
    fn test_clt_encode_latency() {
        if !clt_files_cached() || !cuda_available() {
            eprintln!("  SKIP test_clt_encode_latency: needs GPU + cached CLT");
            return;
        }
        let device = Device::cuda_if_available(0).unwrap();
        let mut clt = CrossLayerTranscoder::open(CLT_REPO).unwrap();
        clt.load_encoder(12, &device).unwrap();

        let residual = Tensor::randn(0.0f32, 1.0, (2304,), &device).unwrap();

        // Warmup
        for _ in 0..10 {
            let _ = clt.encode(&residual, 12).unwrap();
        }

        // Benchmark
        let n_iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..n_iters {
            let _ = clt.encode(&residual, 12).unwrap();
        }
        let elapsed = start.elapsed();
        let mean_ms = elapsed.as_secs_f64() * 1000.0 / f64::from(n_iters);

        println!(
            "CLT encode latency: {mean_ms:.3}ms mean over {n_iters} iterations \
             (total: {:.1}ms)",
            elapsed.as_secs_f64() * 1000.0
        );
        assert!(
            mean_ms < 5.0,
            "Encode latency {mean_ms:.3}ms exceeds 5ms target"
        );
    }

    #[test]
    fn test_clt_decode_round_trip() {
        if !clt_files_cached() || !cuda_available() {
            eprintln!("  SKIP test_clt_decode_round_trip: needs GPU + cached CLT");
            return;
        }
        let device = Device::cuda_if_available(0).unwrap();
        let mut clt = CrossLayerTranscoder::open(CLT_REPO).unwrap();
        clt.load_encoder(12, &device).unwrap();

        let residual = Tensor::randn(0.0f32, 1.0, (2304,), &device).unwrap();

        // Encode
        let sparse = clt.encode(&residual, 12).unwrap();
        assert!(!sparse.is_empty(), "Should have active features");

        // Decode top-10 features at target layer 12
        let top10: Vec<(CltFeatureId, f32)> = sparse.features.iter().take(10).copied().collect();
        let mut reconstruction = Tensor::zeros((2304,), DType::F32, &device).unwrap();
        for (fid, activation) in &top10 {
            let dec_vec = clt.decoder_vector(fid, 12, &device).unwrap();
            assert_eq!(dec_vec.dims(), &[2304]);
            let dec_f32 = dec_vec.to_dtype(DType::F32).unwrap();
            let scaled = (&dec_f32 * f64::from(*activation)).unwrap();
            reconstruction = (&reconstruction + &scaled).unwrap();
        }

        // Reconstruction should be non-zero
        let recon_vals: Vec<f32> = reconstruction.to_vec1().unwrap();
        let recon_norm: f32 = recon_vals.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            recon_norm > 0.0,
            "Reconstruction from top-10 features should be non-zero"
        );
        println!(
            "Decode round-trip: {} active features, top-10 reconstruction L2 norm = {recon_norm:.4}",
            sparse.len()
        );

        // Determinism: encoding the same residual again should produce identical results
        let sparse2 = clt.encode(&residual, 12).unwrap();
        assert_eq!(
            sparse.len(),
            sparse2.len(),
            "Repeat encoding should produce same number of active features"
        );
        for ((fid1, act1), (fid2, act2)) in sparse.features.iter().zip(sparse2.features.iter()) {
            assert_eq!(fid1, fid2, "Feature IDs should match on repeat encode");
            assert!(
                (act1 - act2).abs() < 1e-6,
                "Activations should match on repeat encode"
            );
        }
        println!("Determinism: verified (exact match on repeat encode)");
    }
}
