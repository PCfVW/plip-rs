//! Corpus loading for PLIP experiments

use anyhow::Result;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// A single code sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSample {
    pub code: String,
    pub language: String,
}

/// Raw JSON structure for loading
#[derive(Debug, Deserialize)]
struct CorpusFile {
    samples: Vec<CodeSample>,
    #[allow(dead_code)]
    metadata: Option<serde_json::Value>,
}

/// Collection of code samples for PLIP
#[derive(Debug, Clone)]
pub struct Corpus {
    samples: Vec<CodeSample>,
}

impl Corpus {
    /// Load corpus from JSON file
    pub fn load(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let file: CorpusFile = serde_json::from_str(&content)?;
        Ok(Self {
            samples: file.samples,
        })
    }

    /// Count Python samples
    pub fn python_count(&self) -> usize {
        self.samples
            .iter()
            .filter(|s| s.language == "python")
            .count()
    }

    /// Count Rust samples
    pub fn rust_count(&self) -> usize {
        self.samples.iter().filter(|s| s.language == "rust").count()
    }

    /// Split corpus into train and test sets
    pub fn split(&self, train_ratio: f64, seed: u64) -> (Vec<CodeSample>, Vec<CodeSample>) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut samples = self.samples.clone();
        samples.shuffle(&mut rng);

        let split_idx = (samples.len() as f64 * train_ratio) as usize;
        let train = samples[..split_idx].to_vec();
        let test = samples[split_idx..].to_vec();

        (train, test)
    }

    /// Total number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if corpus is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get all samples
    pub fn samples(&self) -> &[CodeSample] {
        &self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_split() {
        let samples = vec![
            CodeSample {
                code: "a".into(),
                language: "python".into(),
            },
            CodeSample {
                code: "b".into(),
                language: "rust".into(),
            },
            CodeSample {
                code: "c".into(),
                language: "python".into(),
            },
            CodeSample {
                code: "d".into(),
                language: "rust".into(),
            },
        ];

        let corpus = Corpus { samples };
        let (train, test) = corpus.split(0.5, 42);

        assert_eq!(train.len(), 2);
        assert_eq!(test.len(), 2);
    }
}
