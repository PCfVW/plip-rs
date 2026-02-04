//! Integration tests for PLIP-rs
//!
//! Note: Tests marked with #[ignore] require GPU and model download.
//! Run them explicitly with: cargo test --ignored

use plip_rs::{Corpus, ExperimentConfig};
use std::io::Write;
use tempfile::NamedTempFile;

/// Test corpus loading from JSON
#[test]
fn test_corpus_loading() {
    // Create a temporary corpus file
    let mut file = NamedTempFile::new().unwrap();
    writeln!(
        file,
        r#"{{
        "samples": [
            {{"language": "python", "code": "def foo(): pass"}},
            {{"language": "rust", "code": "fn foo() {{}}"}}
        ]
    }}"#
    )
    .unwrap();

    let corpus = Corpus::load(file.path().to_str().unwrap()).unwrap();
    assert_eq!(corpus.len(), 2);
    assert_eq!(corpus.python_count(), 1);
    assert_eq!(corpus.rust_count(), 1);
}

/// Test corpus train/test split
#[test]
fn test_corpus_split() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(
        file,
        r#"{{
        "samples": [
            {{"language": "python", "code": "a"}},
            {{"language": "rust", "code": "b"}},
            {{"language": "python", "code": "c"}},
            {{"language": "rust", "code": "d"}},
            {{"language": "python", "code": "e"}},
            {{"language": "rust", "code": "f"}},
            {{"language": "python", "code": "g"}},
            {{"language": "rust", "code": "h"}},
            {{"language": "python", "code": "i"}},
            {{"language": "rust", "code": "j"}}
        ]
    }}"#
    )
    .unwrap();

    let corpus = Corpus::load(file.path().to_str().unwrap()).unwrap();
    let (train, test) = corpus.split(0.8, 42);

    assert_eq!(train.len(), 8);
    assert_eq!(test.len(), 2);
}

/// Test deterministic split with same seed
#[test]
fn test_split_deterministic() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(
        file,
        r#"{{
        "samples": [
            {{"language": "python", "code": "1"}},
            {{"language": "rust", "code": "2"}},
            {{"language": "python", "code": "3"}},
            {{"language": "rust", "code": "4"}}
        ]
    }}"#
    )
    .unwrap();

    let corpus = Corpus::load(file.path().to_str().unwrap()).unwrap();

    let (train1, test1) = corpus.split(0.5, 42);
    let (train2, test2) = corpus.split(0.5, 42);

    // Same seed should give same split
    for (a, b) in train1.iter().zip(train2.iter()) {
        assert_eq!(a.code, b.code);
    }
    for (a, b) in test1.iter().zip(test2.iter()) {
        assert_eq!(a.code, b.code);
    }
}

/// Test experiment config defaults
#[test]
fn test_experiment_config_defaults() {
    let config = ExperimentConfig::default();
    assert_eq!(config.train_ratio, 0.8);
    assert_eq!(config.seed, 42);
    assert!(config.layers.is_empty());
}

/// GPU-dependent test: model loading
#[test]
#[ignore = "requires GPU and model download"]
fn test_model_loading() {
    use plip_rs::PlipModel;

    let model = PlipModel::from_pretrained("bigcode/starcoder2-3b").unwrap();
    assert_eq!(model.n_layers(), 30);
    assert_eq!(model.d_model(), 3072); // StarCoder2-3B hidden_size
}

/// GPU-dependent test: activation extraction
#[test]
#[ignore = "requires GPU and model download"]
fn test_activation_extraction() {
    use plip_rs::PlipModel;

    let model = PlipModel::from_pretrained("bigcode/starcoder2-3b").unwrap();
    let code = "fn main() {}";

    let cache = model.get_activations(code).unwrap();

    assert_eq!(cache.n_layers(), 30);
    assert!(cache.get_layer(0).is_some());
    assert!(cache.get_layer(29).is_some());
    assert!(cache.get_layer(30).is_none());
}
