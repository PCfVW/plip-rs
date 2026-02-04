//! Linear probing with linfa for language classification
//!
//! Trains logistic regression probes to classify Python vs Rust
//! from model activations.

use anyhow::{Context, Result};
use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Array2};
use tracing::debug;

/// Results from training and evaluating a probe
#[derive(Debug, Clone)]
pub struct ProbeResults {
    /// Classification accuracy on test set
    pub accuracy: f64,
    /// Number of correct predictions
    pub correct: usize,
    /// Total number of test samples
    pub total: usize,
    /// True positive count
    pub true_positives: usize,
    /// True negative count
    pub true_negatives: usize,
    /// False positive count
    pub false_positives: usize,
    /// False negative count
    pub false_negatives: usize,
}

impl ProbeResults {
    /// Compute precision (Rust = positive class)
    pub fn precision(&self) -> f64 {
        let denom = self.true_positives + self.false_positives;
        if denom == 0 {
            0.0
        } else {
            self.true_positives as f64 / denom as f64
        }
    }

    /// Compute recall (Rust = positive class)
    pub fn recall(&self) -> f64 {
        let denom = self.true_positives + self.false_negatives;
        if denom == 0 {
            0.0
        } else {
            self.true_positives as f64 / denom as f64
        }
    }

    /// Compute F1 score
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

/// Trainer for linear probes
pub struct ProbeTrainer {
    max_iterations: u64,
}

impl ProbeTrainer {
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
        }
    }

    /// Train a probe and evaluate on test data
    pub fn train_and_evaluate(
        &self,
        train_data: Vec<(Vec<f32>, bool)>,
        test_data: Vec<(Vec<f32>, bool)>,
    ) -> Result<ProbeResults> {
        // Convert to ndarray format
        let (train_x, train_y) = self.prepare_data(&train_data)?;
        let (test_x, test_y) = self.prepare_data(&test_data)?;

        debug!(
            "Training probe: {} samples, {} features",
            train_x.nrows(),
            train_x.ncols()
        );

        // Create dataset
        let train_dataset = Dataset::new(train_x, train_y);

        // Train logistic regression
        let model = LogisticRegression::default()
            .max_iterations(self.max_iterations)
            .fit(&train_dataset)
            .context("Failed to train logistic regression")?;

        // Predict on test set
        let predictions = model.predict(&test_x);

        // Compute metrics
        let results = self.compute_metrics(&predictions, &test_y);

        Ok(results)
    }

    /// Convert activation data to ndarray format
    fn prepare_data(&self, data: &[(Vec<f32>, bool)]) -> Result<(Array2<f64>, Array1<usize>)> {
        if data.is_empty() {
            anyhow::bail!("Empty dataset");
        }

        let n_samples = data.len();
        let n_features = data[0].0.len();

        // Build feature matrix
        let flat: Vec<f64> = data
            .iter()
            .flat_map(|(features, _)| features.iter().map(|&f| f64::from(f)))
            .collect();

        let x = Array2::from_shape_vec((n_samples, n_features), flat)
            .context("Failed to create feature matrix")?;

        // Build label vector (1 = Rust, 0 = Python)
        let y: Array1<usize> = data
            .iter()
            .map(|(_, is_rust)| usize::from(*is_rust))
            .collect();

        Ok((x, y))
    }

    /// Compute classification metrics
    fn compute_metrics(&self, predictions: &Array1<usize>, labels: &Array1<usize>) -> ProbeResults {
        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut r#fn = 0;

        for (pred, label) in predictions.iter().zip(labels.iter()) {
            match (*pred, *label) {
                (1, 1) => tp += 1,
                (0, 0) => tn += 1,
                (1, 0) => fp += 1,
                (0, 1) => r#fn += 1,
                _ => {}
            }
        }

        let correct = tp + tn;
        let total = predictions.len();
        let accuracy = correct as f64 / total as f64;

        ProbeResults {
            accuracy,
            correct,
            total,
            true_positives: tp,
            true_negatives: tn,
            false_positives: fp,
            false_negatives: r#fn,
        }
    }
}

impl Default for ProbeTrainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_results_metrics() {
        let results = ProbeResults {
            accuracy: 0.8,
            correct: 8,
            total: 10,
            true_positives: 4,
            true_negatives: 4,
            false_positives: 1,
            false_negatives: 1,
        };

        assert!((results.precision() - 0.8).abs() < 0.01);
        assert!((results.recall() - 0.8).abs() < 0.01);
    }
}
