//! Experiment runner for PLIP probing experiments
//!
//! Coordinates corpus loading, activation extraction, and linear probing.

use anyhow::Result;
use tracing::{info, warn};

use crate::corpus::{CodeSample, Corpus};
use crate::model::PlipModel;
use crate::probe::{ProbeResults, ProbeTrainer};

/// Configuration for a PLIP experiment
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// Path to corpus JSON file
    pub corpus_path: String,
    /// Layer indices to probe (empty = all layers)
    pub layers: Vec<usize>,
    /// Train/test split ratio
    pub train_ratio: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            corpus_path: "corpus/samples.json".to_string(),
            layers: vec![],
            train_ratio: 0.8,
            seed: 42,
        }
    }
}

/// Results from a full PLIP experiment
#[derive(Debug)]
pub struct ExperimentResults {
    /// Per-layer probe results
    pub layer_results: Vec<(usize, ProbeResults)>,
    /// Best performing layer
    pub best_layer: usize,
    /// Best accuracy achieved
    pub best_accuracy: f64,
}

/// Main experiment runner
pub struct Experiment {
    config: ExperimentConfig,
    model: PlipModel,
}

impl Experiment {
    /// Create a new experiment
    pub fn new(model: PlipModel, config: ExperimentConfig) -> Self {
        Self { config, model }
    }

    /// Run the full experiment
    pub fn run(&mut self) -> Result<ExperimentResults> {
        info!("Starting PLIP experiment");
        info!("Config: {:?}", self.config);

        // Load corpus
        let corpus = Corpus::load(&self.config.corpus_path)?;
        info!(
            "Loaded corpus: {} Python, {} Rust samples",
            corpus.python_count(),
            corpus.rust_count()
        );

        // Determine which layers to probe
        let layers: Vec<usize> = if self.config.layers.is_empty() {
            (0..self.model.n_layers()).collect()
        } else {
            self.config.layers.clone()
        };

        info!("Probing {} layers", layers.len());

        // Extract activations for all samples
        let (train_samples, test_samples) = corpus.split(self.config.train_ratio, self.config.seed);
        info!(
            "Split: {} train, {} test samples",
            train_samples.len(),
            test_samples.len()
        );

        // Run probing for each layer
        let mut layer_results = Vec::new();
        let mut best_layer = 0;
        let mut best_accuracy = 0.0;

        for &layer in &layers {
            info!("Probing layer {}", layer);

            // Collect activations for this layer
            let train_data = self.collect_activations(&train_samples, layer)?;
            let test_data = self.collect_activations(&test_samples, layer)?;

            // Train and evaluate probe
            let trainer = ProbeTrainer::new();
            let results = trainer.train_and_evaluate(train_data, test_data)?;

            info!("Layer {} accuracy: {:.2}%", layer, results.accuracy * 100.0);

            if results.accuracy > best_accuracy {
                best_accuracy = results.accuracy;
                best_layer = layer;
            }

            layer_results.push((layer, results));
        }

        info!(
            "Best layer: {} with {:.2}% accuracy",
            best_layer,
            best_accuracy * 100.0
        );

        Ok(ExperimentResults {
            layer_results,
            best_layer,
            best_accuracy,
        })
    }

    /// Collect activations for samples at a specific layer
    fn collect_activations(
        &mut self,
        samples: &[CodeSample],
        layer: usize,
    ) -> Result<Vec<(Vec<f32>, bool)>> {
        let mut data = Vec::with_capacity(samples.len());

        for sample in samples {
            let cache = self.model.get_activations(&sample.code)?;

            match cache.get_layer(layer) {
                Some(activation) => {
                    // Convert from F16 to F32 before extracting
                    let flat: Vec<f32> = activation
                        .flatten_all()?
                        .to_dtype(candle_core::DType::F32)?
                        .to_vec1()?;
                    let is_rust = sample.language == "rust";
                    data.push((flat, is_rust));
                }
                None => {
                    warn!("Layer {} not found in cache, skipping sample", layer);
                }
            }
        }

        Ok(data)
    }
}
