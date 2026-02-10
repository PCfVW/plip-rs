// Pedantic clippy configuration for ML/math codebase
// These are acceptable in numerical/ML code:
#![allow(clippy::cast_precision_loss)] // usize→f64/f32 intentional in ML
#![allow(clippy::cast_possible_truncation)] // usize→u32 in tensor indexing
#![allow(clippy::cast_possible_wrap)] // usize→i64 in tensor ops
#![allow(clippy::many_single_char_names)] // x, y, i, j standard in math
#![allow(clippy::similar_names)] // related variables like `head`/`heads`
#![allow(clippy::module_name_repetitions)] // PlipModel in model.rs is fine
// Documentation pedantic - acceptable for research code:
#![allow(clippy::doc_markdown)] // backticks for every technical term is excessive
#![allow(clippy::missing_errors_doc)] // # Errors section for every Result fn
#![allow(clippy::missing_panics_doc)] // # Panics section for every panic
// Method style pedantic:
#![allow(clippy::must_use_candidate)] // #[must_use] on every pure fn is excessive
#![allow(clippy::return_self_not_must_use)] // #[must_use] on Self returns
#![allow(clippy::unused_self)] // &self for API consistency
#![allow(clippy::trivially_copy_pass_by_ref)] // &usize for API consistency
#![allow(clippy::struct_field_names)] // field postfix patterns
#![allow(clippy::needless_pass_by_value)] // value params for API flexibility
#![allow(clippy::unnecessary_wraps)] // Result for future error handling
#![allow(clippy::cast_sign_loss)] // f64→usize when value is known positive

//! PLIP-rs: Programming Language Internal Probing
//!
//! Investigates where transformer models internally distinguish
//! programming languages (Python vs Rust).
//!
//! ## Architecture
//!
//! - `model`: High-level PlipModel wrapper for activation extraction
//! - `forward`: StarCoder2 forward pass with activation capture hooks
//! - `forward_qwen2`: Qwen2 forward pass with activation capture hooks
//! - `forward_gemma`: Gemma/CodeGemma forward pass with activation capture hooks
//! - `forward_llama`: LLaMA/Code-LLaMA forward pass with activation capture hooks
//! - `forward_phi3`: Phi-3 forward pass with activation capture hooks
//! - `cache`: ActivationCache for storing layer activations
//! - `kv_cache`: KV-cache for efficient autoregressive generation
//! - `masks`: Shared attention mask utilities (causal masks, generation masks)
//! - `attention`: Attention pattern capture and analysis
//! - `corpus`: Code sample loading and management
//! - `probe`: Linear probing with linfa for language classification
//! - `experiment`: Experiment runner coordinating the full pipeline
//! - `logit_lens`: Logit Lens for interpretability through intermediate predictions
//! - `positioning`: Model-agnostic character-based position handling
//! - `intervention`: Attention intervention (knockout and steering) for causal experiments
//! - `steering`: Attention steering calibration and dose-response utilities

pub mod attention;
pub mod cache;
pub mod corpus;
pub mod experiment;
pub mod forward;
pub mod forward_gemma;
pub mod forward_llama;
pub mod forward_phi3;
pub mod forward_qwen2;
pub mod forward_rwkv6;
pub mod intervention;
pub mod kv_cache;
pub mod logit_lens;
pub mod masks;
pub mod model;
pub mod positioning;
pub mod probe;
pub mod steering;
pub mod tokenizer_rwkv;

pub use attention::{AttentionAnalysis, AttentionCache};
pub use cache::ActivationCache;
pub use corpus::{CodeSample, Corpus};
pub use experiment::{Experiment, ExperimentConfig, ExperimentResults};
pub use forward::PlipStarCoder2;
pub use forward_gemma::PlipGemma;
pub use forward_llama::PlipLlama;
pub use forward_phi3::PlipPhi3;
pub use forward_qwen2::PlipQwen2;
pub use forward_rwkv6::PlipRwkv6;
pub use intervention::{
    // Part 2: Amplification (Attention Steering)
    apply_scale_steering,
    apply_set_value_steering,
    apply_steering,
    create_knockout_mask,
    kl_divergence,
    measure_attention_to_targets,
    AblationResult,
    AttentionEdge,
    HeadSpec,
    InterventionType,
    KnockoutSpec,
    LayerSpec,
    StateAblationResult,
    StateKnockoutSpec,
    StateSteeringResult,
    StateSteeringSpec,
    SteeringResult,
    SteeringSpec,
};
pub use kv_cache::KVCache;
pub use logit_lens::{LogitLensAnalysis, LogitLensResult, TokenPrediction};
pub use masks::{clear_mask_caches, create_causal_mask, create_generation_mask};
pub use model::{GenerationResult, ModelArchitecture, PlipBackend, PlipModel, PlipTokenizer};
pub use tokenizer_rwkv::RwkvTokenizer;
pub use positioning::{EncodingWithOffsets, PositionConversion, TokenWithOffset};
pub use probe::{ProbeResults, ProbeTrainer};
pub use steering::{
    calibrate_from_samples, CalibrationSample, DoseResponseCurve, DoseResponsePoint,
    SteeringCalibration, DOSE_LEVELS,
};
