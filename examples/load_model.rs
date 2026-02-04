//! Example: Load a model and print configuration
//!
//! Run with: cargo run --release --no-default-features --example load_model -- --cpu
//! For Qwen2: cargo run --release --no-default-features --example load_model -- --cpu --model "Qwen/Qwen2.5-Coder-3B-Instruct"
//! For StarCoder2: cargo run --release --no-default-features --example load_model -- --cpu --model "bigcode/starcoder2-3b"

use anyhow::Result;
use clap::Parser;
use plip_rs::PlipModel;

#[derive(Parser)]
#[command(name = "load_model")]
#[command(about = "Load and inspect a code model")]
struct Args {
    /// HuggingFace model ID
    #[arg(short, long, default_value = "Qwen/Qwen2.5-Coder-3B-Instruct")]
    model: String,

    /// Force CPU mode
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();

    println!("Loading model: {}", args.model);
    println!("(This will download model files on first run)");
    if args.cpu {
        println!("Mode: CPU (forced)\n");
    } else {
        println!("Mode: Auto (CUDA if available)\n");
    }

    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;

    println!("\nModel loaded successfully!");
    println!("  Architecture: {:?}", model.architecture());
    println!("  Layers: {}", model.n_layers());
    println!("  Hidden dim: {}", model.d_model());
    println!("  Vocab size: {}", model.vocab_size());
    println!("  Attention heads: {}", model.n_heads());

    Ok(())
}
