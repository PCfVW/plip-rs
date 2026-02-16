//! CLT logit shift acceptance test.
//!
//! **CLT Infrastructure** — verifies that Cross-Layer Transcoder injection produces measurable
//! logit shifts (§7.3 criterion: "injection produces expected logit shifts").
//!
//! Steps:
//!   1. Load Gemma 2 2B
//!   2. Open CLT dictionary
//!   3. Run forward pass on test prompt, encode residual at source layer
//!   4. Cache decoder vectors for top features → target layer
//!   5. Build `CltInjectionSpec` via `prepare_injection`
//!   6. Compare baseline vs injected logits (KL divergence)
//!
//! Usage:
//!   cargo run --release --example `clt_logit_shift`
//!   cargo run --release --example `clt_logit_shift` -- --prompt "The Eiffel Tower is in"

#![allow(clippy::doc_markdown)]

use anyhow::Result;
use candle_core::Device;
use clap::Parser;
use plip_rs::{CrossLayerTranscoder, PlipModel};

#[derive(Parser)]
struct Args {
    /// `HuggingFace` model to load
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// `HuggingFace` CLT repository
    #[arg(long, default_value = "mntss/clt-gemma-2-2b-426k")]
    clt_repo: String,

    /// Source layer for CLT encoding
    #[arg(long, default_value_t = 12)]
    source_layer: usize,

    /// Target layer for CLT injection
    #[arg(long, default_value_t = 25)]
    target_layer: usize,

    /// Number of top features to use
    #[arg(long, default_value_t = 5)]
    top_k: usize,

    /// Injection strength
    #[arg(long, default_value_t = 5.0)]
    strength: f32,

    /// Test prompt
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,

    /// Force CPU execution
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    println!("Device: {device:?}");

    // 1. Load Gemma 2 2B
    println!("\n=== Loading model: {} ===", args.model);
    let model = PlipModel::from_pretrained(&args.model)?;
    println!("Model loaded: {} layers", model.n_layers());

    // 2. Open CLT
    println!("\n=== Opening CLT: {} ===", args.clt_repo);
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let config = clt.config().clone();
    println!(
        "CLT: {} features/layer, d_model={}",
        config.n_features_per_layer, config.d_model
    );

    // 3. Forward pass to get residual at source layer
    println!("\n=== Forward pass on: \"{}\" ===", args.prompt);
    let activations = model.get_activations(&args.prompt)?;
    let residual = activations
        .get_layer(args.source_layer)
        .ok_or_else(|| anyhow::anyhow!("No activation at layer {}", args.source_layer))?
        .squeeze(0)?; // Remove batch dimension: [1, d_model] → [d_model]
    println!(
        "Residual at layer {}: shape {:?}",
        args.source_layer,
        residual.dims()
    );

    // 4. Encode residual → top features
    clt.load_encoder(args.source_layer, &device)?;
    let top = clt.top_k(&residual, args.source_layer, args.top_k)?;
    println!(
        "\nTop-{} features at layer {}:",
        args.top_k, args.source_layer
    );
    for (fid, act) in &top.features {
        println!("  {fid}: activation = {act:.4}");
    }

    // 5. Cache decoder vectors for each feature → target layer
    let feature_targets: Vec<_> = top
        .features
        .iter()
        .map(|(fid, _)| (*fid, args.target_layer))
        .collect();
    clt.cache_steering_vectors(&feature_targets, &device)?;
    println!(
        "\nCached {} decoder vectors for target layer {}",
        feature_targets.len(),
        args.target_layer
    );

    // 6. Build CltInjectionSpec
    let token_strs = model.tokenize(&args.prompt)?;
    let last_pos = token_strs.len() - 1;
    let spec = clt.prepare_injection(&feature_targets, last_pos, args.strength)?;
    println!(
        "Injection spec: {} layer entries, position {}, strength {}",
        spec.injections.len(),
        last_pos,
        args.strength
    );

    // 7. Compare baseline vs injected logits
    println!("\n=== Logit shift comparison ===");
    let result = model.clt_logit_shift(&args.prompt, &spec)?;

    let kl = result.kl_divergence()?;
    println!("KL divergence: {kl:.6}");

    let top_changed = result.top_changed_tokens(10)?;
    println!("\nTop-10 tokens with largest probability shift:");
    println!(
        "  {:>8}  {:>12}  {:>12}  {:>12}",
        "Token ID", "Baseline", "Injected", "Abs Diff"
    );
    for (tok_id, base_p, inj_p, diff) in &top_changed {
        // Try to decode the token
        let tok_str = model.decode_token(*tok_id);
        println!("  {tok_id:>8}  {base_p:>12.6}  {inj_p:>12.6}  {diff:>12.6}  \"{tok_str}\"");
    }

    // 8. Acceptance criterion: KL > 0 (injection actually changes logits)
    if kl > 0.0 {
        println!("\n*** PASS: CLT injection produces logit shift (KL = {kl:.6}) ***");
    } else {
        println!("\n*** FAIL: No logit shift detected ***");
        std::process::exit(1);
    }

    Ok(())
}
