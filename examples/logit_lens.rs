//! Logit Lens Example: See what the model predicts at each layer
//!
//! This demonstrates the first Rust implementation of Logit Lens,
//! showing how predictions evolve through the transformer layers.
//!
//! Usage:
//!   cargo run --release --no-default-features --example logit_lens -- --cpu
//!   cargo run --release --no-default-features --example logit_lens -- --cpu --model "Qwen/Qwen2.5-Coder-3B-Instruct"

use anyhow::Result;
use clap::Parser;
use plip_rs::PlipModel;

#[derive(Parser)]
#[command(name = "logit_lens")]
#[command(about = "Logit Lens analysis of code models")]
struct Args {
    /// HuggingFace model ID
    #[arg(short, long, default_value = "Qwen/Qwen2.5-Coder-3B-Instruct")]
    model: String,

    /// Code to analyze (what comes next?)
    #[arg(
        short,
        long,
        default_value = "fn add(a: i32, b: i32) -> i32 { a + b }\n"
    )]
    code: String,

    /// Number of top predictions to show per layer
    #[arg(short, long, default_value = "5")]
    top_k: usize,

    /// Force CPU mode
    #[arg(long)]
    cpu: bool,

    /// Show detailed output (all layers, all predictions)
    #[arg(long)]
    detailed: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();

    println!("=== Logit Lens: Rust Implementation ===\n");

    // Load model
    println!("Loading {}...", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    println!(
        "Model loaded: {} layers, {} hidden, {} vocab (architecture: {:?})\n",
        model.n_layers(),
        model.d_model(),
        model.vocab_size(),
        model.architecture()
    );

    // Show tokenization
    println!("Input code:");
    println!("  \"{}\"", args.code.replace('\n', "\\n"));

    let tokens = model.tokenize(&args.code)?;
    println!("\nTokenized ({} tokens):", tokens.len());
    for (i, tok) in tokens.iter().enumerate() {
        println!("  {}: \"{}\"", i, tok.replace('\n', "\\n"));
    }

    // Run logit lens
    println!("\nRunning Logit Lens analysis...\n");
    let analysis = model.logit_lens(&args.code, args.top_k)?;

    if args.detailed {
        analysis.print_detailed(args.top_k);
    } else {
        analysis.print_summary();
    }

    // Look for specific patterns
    println!("\n=== Pattern Analysis ===");

    // Check when #[test] appears
    if let Some(layer) = analysis.first_appearance("#[test]", args.top_k) {
        println!(
            "'#[test]' first appears in top-{} at layer {}",
            args.top_k, layer
        );
    } else {
        println!("'#[test]' never appears in top-{}", args.top_k);
    }

    // Check when #[cfg(test)] appears
    if let Some(layer) = analysis.first_appearance("#[cfg", args.top_k) {
        println!(
            "'#[cfg' first appears in top-{} at layer {}",
            args.top_k, layer
        );
    } else {
        println!("'#[cfg' never appears in top-{}", args.top_k);
    }

    // Check when fn/def appears (language-specific)
    if let Some(layer) = analysis.first_appearance("fn ", args.top_k) {
        println!(
            "'fn ' first appears in top-{} at layer {}",
            args.top_k, layer
        );
    }
    if let Some(layer) = analysis.first_appearance("def ", args.top_k) {
        println!(
            "'def ' first appears in top-{} at layer {}",
            args.top_k, layer
        );
    }

    println!("\n=== Analysis Complete ===");

    Ok(())
}
