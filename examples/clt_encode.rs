//! Demonstrate CLT feature extraction on a dummy residual vector.
//!
//! **CLT Infrastructure** — opens the CLT dictionary from `HuggingFace`,
//! loads an encoder, encodes a random residual vector, and prints the
//! top-K active features.
//!
//! Usage:
//!   cargo run --release --example `clt_encode`
//!   cargo run --release --example `clt_encode` -- --clt-repo mntss/clt-gemma-2-2b-2.5M --layer 12

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]

use anyhow::Result;
use candle_core::{Device, Tensor};
use clap::Parser;
use plip_rs::CrossLayerTranscoder;

#[derive(Parser)]
struct Args {
    /// `HuggingFace` CLT repository
    #[arg(long, default_value = "mntss/clt-gemma-2-2b-426k")]
    clt_repo: String,

    /// Layer to encode at
    #[arg(long, default_value_t = 12)]
    layer: usize,

    /// Number of top features to display
    #[arg(long, default_value_t = 10)]
    top_k: usize,

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

    // Open CLT (downloads on first run, cached by hf_hub)
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let config = clt.config().clone();
    println!(
        "\nCLT: {} layers, {} features/layer, d_model={}, total={}",
        config.n_layers, config.n_features_per_layer, config.d_model, config.n_features_total
    );
    println!("Base model: {}", config.model_name);

    // Load encoder for the requested layer
    clt.load_encoder(args.layer, &device)?;
    println!("\nEncoder loaded for layer {}", args.layer);

    // Create a dummy residual (in real use, this comes from Gemma 2 2B forward pass)
    let residual = Tensor::randn(0.0f32, 1.0, (config.d_model,), &device)?;
    println!("Residual shape: {:?}", residual.dims());

    // Encode
    let sparse = clt.encode(&residual, args.layer)?;
    println!(
        "\nActive features: {} / {} ({:.1}% sparsity)",
        sparse.len(),
        config.n_features_per_layer,
        100.0 * (1.0 - sparse.len() as f64 / config.n_features_per_layer as f64)
    );

    // Top-K
    let top = clt.top_k(&residual, args.layer, args.top_k)?;
    println!("\nTop-{} features:", args.top_k);
    for (fid, act) in &top.features {
        println!("  {fid}: activation = {act:.4}");
    }

    // Demonstrate decoder vector extraction
    if let Some((fid, _)) = top.features.first() {
        let target_layer = config.n_layers - 1; // last layer
        let dec_vec = clt.decoder_vector(fid, target_layer, &device)?;
        println!(
            "\nDecoder vector for {fid} → layer {target_layer}: shape {:?}",
            dec_vec.dims()
        );
    }

    Ok(())
}
