//! Integration tests for PLIP-rs
//!
//! Note: Tests marked with #[ignore] require GPU and model download.
//! Run them explicitly with: cargo test --ignored

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use plip_rs::{Corpus, ExperimentConfig};
use serial_test::serial;
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
    assert!((config.train_ratio - 0.8).abs() < f64::EPSILON);
    assert_eq!(config.seed, 42);
    assert!(config.layers.is_empty());
}

// ============================================================================
// RWKV-6 validation against Python reference
// ============================================================================

/// Validate RWKV-6 tokenizer produces the same token IDs as the Python reference.
#[test]
#[ignore = "requires model download (~3.2 GB)"]
#[serial]
fn test_rwkv6_tokenizer() {
    use plip_rs::RwkvTokenizer;
    use std::path::Path;

    let reference: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("scripts/rwkv6_reference.json")
            .expect("Run `python scripts/rwkv6_validation.py` first"),
    )
    .unwrap();

    // Load tokenizer from HF cache
    let vocab_path_str = hf_hub::api::sync::Api::new()
        .unwrap()
        .repo(hf_hub::Repo::new(
            "RWKV/v6-Finch-1B6-HF".to_string(),
            hf_hub::RepoType::Model,
        ))
        .get("rwkv_vocab_v20230424.txt")
        .unwrap();
    let tokenizer = RwkvTokenizer::from_file(Path::new(&vocab_path_str)).unwrap();

    let prompt = reference["test_prompt"].as_str().unwrap();
    let expected_ids: Vec<u32> = reference["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let rust_ids = tokenizer.encode(prompt).unwrap();
    assert_eq!(
        rust_ids, expected_ids,
        "Token IDs mismatch: Rust={rust_ids:?} vs Python={expected_ids:?}"
    );

    // Verify decode round-trip
    let decoded = tokenizer.decode(&rust_ids).unwrap();
    assert_eq!(decoded, prompt, "Decode round-trip mismatch");
}

/// Validate RWKV-6 forward pass produces matching logits against Python reference.
///
/// Loads the model in F32 on GPU (CUDA) and compares top-10 predictions.
#[test]
#[ignore = "requires GPU and model download (~3.2 GB)"]
#[serial]
fn test_rwkv6_forward_logits() {
    use candle_core::{DType, Device, Tensor};
    use plip_rs::PlipRwkv6;

    let reference: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("scripts/rwkv6_reference.json")
            .expect("Run `python scripts/rwkv6_validation.py` first"),
    )
    .unwrap();

    let token_ids: Vec<u32> = reference["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    // Load model in F32 on GPU for comparison with Python reference
    let device = Device::new_cuda(0).expect("CUDA device required");
    let model = PlipRwkv6::load("RWKV/v6-Finch-1B6-HF", &device, DType::F32)
        .expect("Model loading failed — run convert_rwkv_to_safetensors.py first");

    // Run forward pass
    let input_tensor = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let mut kv_cache = model.new_kv_cache();
    let logits = model
        .forward_with_kv_cache(&input_tensor, &mut kv_cache)
        .expect("Forward pass failed");

    // Get logits as Vec<f32> (squeeze batch dimension)
    let logits_vec: Vec<f32> = logits
        .squeeze(0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();

    // Compare top prediction
    let expected_top = &reference["top_predictions"][0];
    let expected_top_id = expected_top["token_id"].as_u64().unwrap() as usize;
    let expected_top_logit = expected_top["logit"].as_f64().unwrap() as f32;

    // Find argmax in Rust logits
    let (rust_top_id, rust_top_logit) = logits_vec
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    println!("Top prediction: Rust id={rust_top_id} logit={rust_top_logit:.4} vs Python id={expected_top_id} logit={expected_top_logit:.4}");

    // The top predicted token should match
    assert_eq!(
        rust_top_id, expected_top_id,
        "Top predicted token mismatch: Rust={rust_top_id} vs Python={expected_top_id}"
    );

    // Compare logit values for top-10 tokens (tolerance for f32 numerical differences)
    let expected_logit_values: Vec<f64> = reference["top_logit_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let expected_top_ids: Vec<usize> = reference["top_predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["token_id"].as_u64().unwrap() as usize)
        .collect();

    for (i, (&tid, &expected_logit)) in expected_top_ids
        .iter()
        .zip(expected_logit_values.iter())
        .enumerate()
    {
        let rust_logit = f64::from(logits_vec[tid]);
        let abs_diff = (rust_logit - expected_logit).abs();
        let rel_diff = abs_diff / expected_logit.abs().max(1e-6);
        println!(
            "  Top-{}: id={tid:>5} Rust={rust_logit:.4} Python={expected_logit:.4} (abs={abs_diff:.6}, rel={rel_diff:.4})",
            i + 1
        );

        // Allow reasonable tolerance for float32 accumulation differences
        // across 24 layers of recurrence
        assert!(
            abs_diff < 1.0 || rel_diff < 0.2,
            "Logit mismatch too large for token {tid} at rank {}: abs={abs_diff:.6}, rel={rel_diff:.4}",
            i + 1
        );
    }
}

/// Validate RWKV-6 greedy generation matches Python reference.
#[test]
#[ignore = "requires GPU and model download (~3.2 GB)"]
#[serial]
fn test_rwkv6_generation() {
    use candle_core::{DType, Device};
    use plip_rs::PlipRwkv6;

    let reference: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("scripts/rwkv6_reference.json")
            .expect("Run `python scripts/rwkv6_validation.py` first"),
    )
    .unwrap();

    let token_ids: Vec<u32> = reference["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let expected_generated: Vec<u32> = reference["generated_token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let device = Device::new_cuda(0).expect("CUDA device required");
    let model =
        PlipRwkv6::load("RWKV/v6-Finch-1B6-HF", &device, DType::F32).expect("Model loading failed");

    // Generate with temperature=0 (greedy), EOS=0
    let stop_tokens = vec![0u32];
    let all_tokens = model
        .generate(&token_ids, 20, 0.0, &stop_tokens, &device)
        .expect("Generation failed");

    let generated = &all_tokens[token_ids.len()..];
    println!("Generated tokens: {generated:?}");
    println!("Expected tokens:  {expected_generated:?}");

    // At minimum, the first generated token should match (most sensitive to errors)
    assert_eq!(
        generated[0], expected_generated[0],
        "First generated token mismatch: Rust={} vs Python={}",
        generated[0], expected_generated[0]
    );

    // Check how many tokens match in sequence
    let matching = generated
        .iter()
        .zip(expected_generated.iter())
        .take_while(|(a, b)| a == b)
        .count();
    println!("Matching tokens: {matching}/{}", expected_generated.len());

    // All generated tokens should match for greedy decoding with F32
    assert_eq!(
        generated,
        &expected_generated[..],
        "Generated sequence mismatch after {matching} matching tokens"
    );
}

// ============================================================================
// Existing GPU-dependent tests
// ============================================================================

/// GPU-dependent test: model loading
#[test]
#[ignore = "requires GPU and model download"]
#[serial]
fn test_model_loading() {
    use plip_rs::PlipModel;

    let model = PlipModel::from_pretrained("bigcode/starcoder2-3b").unwrap();
    assert_eq!(model.n_layers(), 30);
    assert_eq!(model.d_model(), 3072); // StarCoder2-3B hidden_size
}

/// GPU-dependent test: activation extraction
#[test]
#[ignore = "requires GPU and model download"]
#[serial]
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

// ============================================================================
// Phase 4: State Knockout — spec validation (no GPU)
// ============================================================================

/// Validate `StateKnockoutSpec` builder and validation logic.
#[test]
fn test_state_knockout_spec_validation() {
    use plip_rs::StateKnockoutSpec;

    // Valid spec
    let spec = StateKnockoutSpec::new()
        .position(2)
        .positions(&[3, 5])
        .layer(0)
        .layer(10);
    assert!(spec.validate(24, 10).is_ok());

    // Position out of range
    let spec_bad_pos = StateKnockoutSpec::new().position(10).layer(0);
    assert!(spec_bad_pos.validate(24, 10).is_err());

    // Layer out of range
    let spec_bad_layer = StateKnockoutSpec::new().position(0).layer(30);
    assert!(spec_bad_layer.validate(24, 10).is_err());

    // No positions
    let spec_empty = StateKnockoutSpec::new().layer(0);
    assert!(spec_empty.validate(24, 10).is_err());

    // position_set should deduplicate
    let spec_dup = StateKnockoutSpec::new()
        .position(2)
        .position(2)
        .positions(&[3, 2]);
    let pos_set = spec_dup.position_set();
    assert_eq!(pos_set.len(), 2); // {2, 3}
    assert!(pos_set.contains(&2));
    assert!(pos_set.contains(&3));
}

// ============================================================================
// Phase 4: State Knockout — GPU integration test
// ============================================================================

/// Validate RWKV-6 state knockout produces different logits (KL divergence > 0).
#[test]
#[ignore = "requires GPU and model download (~3.2 GB)"]
#[serial]
fn test_rwkv6_state_knockout_kl() {
    use candle_core::{DType, Device, IndexOp, Tensor};
    use plip_rs::{PlipRwkv6, StateKnockoutSpec};

    let device = Device::new_cuda(0).expect("CUDA device required");
    let model = PlipRwkv6::load("RWKV/v6-Finch-1B6-HF", &device, DType::F32)
        .expect("Model loading failed — run convert_rwkv_to_safetensors.py first");

    let prompt = "def add(a, b):\n    return a + b";
    let tokenizer = {
        let vocab_path_str = hf_hub::api::sync::Api::new()
            .unwrap()
            .repo(hf_hub::Repo::new(
                "RWKV/v6-Finch-1B6-HF".to_string(),
                hf_hub::RepoType::Model,
            ))
            .get("rwkv_vocab_v20230424.txt")
            .unwrap();
        plip_rs::RwkvTokenizer::from_file(std::path::Path::new(&vocab_path_str)).unwrap()
    };
    let token_ids = tokenizer.encode(prompt).unwrap();
    let seq_len = token_ids.len();
    let input_tensor = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    // Baseline: normal forward
    let mut baseline_cache = model.new_kv_cache();
    let baseline_output = model
        .forward_with_kv_cache(&input_tensor, &mut baseline_cache)
        .unwrap();

    // Knocked-out: suppress position 0 across all layers
    let spec = StateKnockoutSpec::new()
        .position(0)
        .layer_range(0, model.n_layers() - 1);
    spec.validate(model.n_layers(), seq_len).unwrap();

    let ablated_output = model
        .forward_with_state_knockout(&input_tensor, &spec)
        .unwrap();

    // Both should return logits of same shape
    // baseline_output is from forward_with_kv_cache: [vocab_size] (already last-token projected)
    // ablated_output is from forward_with_state_knockout: [batch, seq, hidden_size] (ln_out normalized)
    // We need to extract last-token logits from ablated_output the same way
    let ablated_seq_len = ablated_output.dim(1).unwrap();
    let ablated_last = ablated_output
        .i((.., ablated_seq_len - 1, ..))
        .unwrap()
        .squeeze(1)
        .unwrap();
    let ablated_logits = model.project_to_vocab(&ablated_last).unwrap();

    // Compute KL divergence between baseline and ablated
    let kl = plip_rs::kl_divergence(&baseline_output, &ablated_logits).unwrap();
    println!("State knockout KL divergence: {kl:.6}");

    // KL should be > 0 (knockout changes the output)
    assert!(
        kl > 0.0,
        "KL divergence should be positive when knocking out position 0, got {kl}"
    );
    // KL should be finite
    assert!(kl.is_finite(), "KL divergence should be finite, got {kl}");
}

// ============================================================================
// Phase 5: RWKV-6 Effective Attention
// ============================================================================

/// Validate RWKV-6 effective attention has correct shape, causality, and normalization.
#[test]
#[ignore = "requires GPU and model download (~3.2 GB)"]
#[serial]
fn test_rwkv6_effective_attention() {
    use candle_core::{DType, Device, Tensor};
    use plip_rs::PlipRwkv6;

    let device = Device::new_cuda(0).expect("CUDA device required");
    let model = PlipRwkv6::load("RWKV/v6-Finch-1B6-HF", &device, DType::F32)
        .expect("Model loading failed — run convert_rwkv_to_safetensors.py first");

    let tokenizer = {
        let vocab_path_str = hf_hub::api::sync::Api::new()
            .unwrap()
            .repo(hf_hub::Repo::new(
                "RWKV/v6-Finch-1B6-HF".to_string(),
                hf_hub::RepoType::Model,
            ))
            .get("rwkv_vocab_v20230424.txt")
            .unwrap();
        plip_rs::RwkvTokenizer::from_file(std::path::Path::new(&vocab_path_str)).unwrap()
    };
    let prompt = "def add(a, b):\n    return a + b";
    let token_ids = tokenizer.encode(prompt).unwrap();
    let seq_len = token_ids.len();
    let input_tensor = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let (_output, attn_cache) = model
        .forward_with_attention(&input_tensor)
        .expect("forward_with_attention failed");

    // Check layer count
    assert_eq!(
        attn_cache.n_layers(),
        model.n_layers(),
        "AttentionCache should have {} layers",
        model.n_layers()
    );

    // Check shape and properties for a few layers
    for layer_idx in [0, 11, 23] {
        let attn = attn_cache
            .get_layer(layer_idx)
            .unwrap_or_else(|| panic!("Layer {layer_idx} missing from attention cache"));
        let dims = attn.dims();
        assert_eq!(
            dims,
            &[1, model.n_heads(), seq_len, seq_len],
            "Layer {layer_idx}: expected [1, {}, {seq_len}, {seq_len}], got {dims:?}",
            model.n_heads()
        );

        // Flatten to 1D and manually index as [batch][heads][query][source]
        let n_heads = model.n_heads();
        let flat: Vec<f32> = attn
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Index helper: flat[(h * seq_len + q) * seq_len + s] (batch=0)
        let idx = |h: usize, q: usize, s: usize| -> f32 { flat[(h * seq_len + q) * seq_len + s] };

        // Check causality: entries where source > query should be zero
        for head in 0..n_heads {
            for query in 0..seq_len {
                for source in (query + 1)..seq_len {
                    let val = idx(head, query, source);
                    assert!(
                        val.abs() < 1e-6,
                        "Layer {layer_idx}, head {head}: non-causal entry [{query}][{source}] = {val}"
                    );
                }
            }
        }

        // Check row sums ≈ 1.0 (within tolerance; some rows may be all-zero if ReLU kills everything)
        let mut valid_rows = 0;
        let mut max_deviation = 0.0f32;
        for head in 0..n_heads {
            for query in 0..seq_len {
                let row_sum: f32 = (0..seq_len).map(|s| idx(head, query, s)).sum();
                if row_sum > 0.01 {
                    // Skip rows where ReLU killed everything
                    let deviation = (row_sum - 1.0).abs();
                    max_deviation = max_deviation.max(deviation);
                    valid_rows += 1;
                }
            }
        }

        println!(
            "Layer {layer_idx}: {valid_rows}/{} valid rows, max row-sum deviation = {max_deviation:.6}",
            n_heads * seq_len
        );

        assert!(
            max_deviation < 1e-4,
            "Layer {layer_idx}: row sum deviation too large: {max_deviation}"
        );
        assert!(
            valid_rows > 0,
            "Layer {layer_idx}: no valid rows (all attention zeroed by ReLU)"
        );
    }
}

// ============================================================================
// Phase 2b: Gemma 2 Attention Steering
// ============================================================================

/// Verify that steering with scale(1.0) produces identical logits to no steering.
///
/// This is a correctness test for the new `forward_with_steering` code path
/// in `forward_gemma2.rs`: identity steering must not change the output.
#[test]
#[ignore = "requires GPU and model download (~5 GB)"]
#[serial]
fn test_gemma2_steering_identity() {
    use candle_core::DType;
    use plip_rs::{PlipModel, SteeringSpec};

    let model = PlipModel::from_pretrained("google/gemma-2-2b").expect("Gemma 2 2B loading failed");

    let prompt = "The autumn leaves begin to fall,\n";
    let token_ids = model.encode(prompt).expect("Encoding failed");
    let seq_len = token_ids.len();

    // Identity steering: scale(1.0) on layer 21, heads 1,6,7
    // PlipModel::forward_with_steering runs both baseline and steered passes
    let spec = SteeringSpec::scale(1.0)
        .layer(21)
        .heads(&[1, 6, 7])
        .from_to_positions(seq_len - 1, &[0, 1, 2]);
    let result = model
        .forward_with_steering(prompt, &spec)
        .expect("Steering forward failed");

    // Compare logits — they should be identical for scale(1.0)
    // Cast to F32 first (model may produce BF16 logits on GPU)
    let baseline_f32 = result.baseline_logits.to_dtype(DType::F32).unwrap();
    let steered_f32 = result.steered_logits.to_dtype(DType::F32).unwrap();
    let diff = (&baseline_f32 - &steered_f32)
        .unwrap()
        .abs()
        .unwrap()
        .max(0)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let max_diff: f32 = diff.iter().copied().fold(0.0f32, f32::max);

    let kl = result.kl_divergence().unwrap();
    println!("Gemma 2 steering identity: max logit diff = {max_diff:.2e}, KL = {kl:.6}");

    assert!(
        max_diff < 1e-4,
        "scale(1.0) steering should not change logits, but max diff = {max_diff:.2e}"
    );
}

/// Verify that steering with scale(4.0) produces different logits than baseline.
///
/// This confirms that the steering code path actually has an effect on the output.
#[test]
#[ignore = "requires GPU and model download (~5 GB)"]
#[serial]
fn test_gemma2_steering_has_effect() {
    use candle_core::DType;
    use plip_rs::{PlipModel, SteeringSpec};

    let model = PlipModel::from_pretrained("google/gemma-2-2b").expect("Gemma 2 2B loading failed");

    let prompt = "The autumn leaves begin to fall,\n";
    let token_ids = model.encode(prompt).expect("Encoding failed");
    let seq_len = token_ids.len();

    // Strong steering: scale(4.0) on layer 21, heads 1,6,7
    // Steer from newline (last token) to first few tokens
    let spec = SteeringSpec::scale(4.0)
        .layer(21)
        .heads(&[1, 6, 7])
        .from_to_positions(seq_len - 1, &[0, 1, 2]);
    let result = model
        .forward_with_steering(prompt, &spec)
        .expect("Steering forward failed");

    let kl = result.kl_divergence().unwrap();

    // Compare logits — they should be different
    // Cast to F32 first (model may produce BF16 logits on GPU)
    let baseline_f32 = result.baseline_logits.to_dtype(DType::F32).unwrap();
    let steered_f32 = result.steered_logits.to_dtype(DType::F32).unwrap();
    let diff = (&baseline_f32 - &steered_f32)
        .unwrap()
        .abs()
        .unwrap()
        .max(0)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let max_diff: f32 = diff.iter().copied().fold(0.0f32, f32::max);

    println!("Gemma 2 steering effect: max logit diff = {max_diff:.4}, KL = {kl:.6}");

    assert!(
        max_diff > 0.01,
        "scale(4.0) steering should change logits, but max diff = {max_diff:.2e}"
    );
    assert!(
        kl > 0.0,
        "KL divergence should be positive with scale(4.0) steering, got {kl}"
    );
    assert!(kl.is_finite(), "KL divergence should be finite, got {kl}");
}

// ============================================================================
// Phase 5: RWKV-6 Effective Attention (continued)
// ============================================================================

/// Validate that `forward_with_attention` produces the same hidden output as `forward_with_cache`.
#[test]
#[ignore = "requires GPU and model download (~3.2 GB)"]
#[serial]
fn test_rwkv6_effective_attention_output_unchanged() {
    use candle_core::{DType, Device, Tensor};
    use plip_rs::PlipRwkv6;

    let device = Device::new_cuda(0).expect("CUDA device required");
    let model =
        PlipRwkv6::load("RWKV/v6-Finch-1B6-HF", &device, DType::F32).expect("Model loading failed");

    let tokenizer = {
        let vocab_path_str = hf_hub::api::sync::Api::new()
            .unwrap()
            .repo(hf_hub::Repo::new(
                "RWKV/v6-Finch-1B6-HF".to_string(),
                hf_hub::RepoType::Model,
            ))
            .get("rwkv_vocab_v20230424.txt")
            .unwrap();
        plip_rs::RwkvTokenizer::from_file(std::path::Path::new(&vocab_path_str)).unwrap()
    };
    let token_ids = tokenizer
        .encode("def add(a, b):\n    return a + b")
        .unwrap();
    let input_tensor = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    // Run both forward methods
    let (cache_output, _act_cache) = model.forward_with_cache(&input_tensor).unwrap();
    let (attn_output, _attn_cache) = model.forward_with_attention(&input_tensor).unwrap();

    // Compare outputs — they should be identical since effective attention
    // is computed independently of the WKV recurrence output
    let diff = (&cache_output - &attn_output)
        .unwrap()
        .abs()
        .unwrap()
        .max(0)
        .unwrap()
        .max(0)
        .unwrap()
        .max(0)
        .unwrap()
        .to_vec0::<f32>()
        .unwrap();

    println!(
        "Max absolute difference between forward_with_cache and forward_with_attention: {diff:.2e}"
    );

    assert!(
        diff < 1e-5,
        "Outputs should be identical, but max diff = {diff:.2e}"
    );
}
