//! PLIP-rs CLI: Programming Language Internal Probing

use anyhow::Result;
use clap::Parser;
use plip_rs::{Corpus, Experiment, ExperimentConfig, PlipModel};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "plip-rs")]
#[command(about = "Programming Language Internal Probing in Rust")]
#[command(version)]
struct Cli {
    /// Model ID from `HuggingFace` (e.g., "bigcode/starcoder2-3b")
    #[arg(short, long, default_value = "bigcode/starcoder2-3b")]
    model: String,

    /// Path to corpus JSON file
    #[arg(short, long, default_value = "corpus/samples.json")]
    corpus: PathBuf,

    /// Output directory for results
    #[arg(short, long, default_value = "outputs")]
    output: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Force CPU mode (slower but avoids CUDA issues)
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let level = if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    println!("=== PLIP-rs: Programming Language Internal Probing ===");
    println!("Model:  {}", cli.model);
    println!("Corpus: {}", cli.corpus.display());
    println!("Output: {}", cli.output.display());
    if cli.cpu {
        println!("Mode:   CPU (forced)");
    }

    // Load model
    info!("Loading model...");
    let model = PlipModel::from_pretrained_with_device(&cli.model, Some(cli.cpu))?;
    info!(
        "Model: {} layers, {} hidden",
        model.n_layers(),
        model.d_model()
    );

    // Load corpus (for validation)
    let corpus_path = cli.corpus.to_string_lossy();
    let corpus = Corpus::load(&corpus_path)?;
    info!(
        "Corpus: {} Python, {} Rust samples",
        corpus.python_count(),
        corpus.rust_count()
    );

    // Run experiment
    let config = ExperimentConfig {
        corpus_path: corpus_path.to_string(),
        ..Default::default()
    };

    let mut experiment = Experiment::new(model, config);
    let results = experiment.run()?;

    // Print results
    println!("\n=== Results ===");
    for (layer, probe_result) in &results.layer_results {
        println!("Layer {:2}: {:.1}%", layer, probe_result.accuracy * 100.0);
    }

    println!(
        "\nBest: layer {} with {:.1}% accuracy",
        results.best_layer,
        results.best_accuracy * 100.0
    );

    // Save results
    std::fs::create_dir_all(&cli.output)?;
    let results_path = cli.output.join("plip_results.json");
    let accuracies: Vec<f64> = results
        .layer_results
        .iter()
        .map(|(_, r)| r.accuracy)
        .collect();
    std::fs::write(&results_path, serde_json::to_string_pretty(&accuracies)?)?;
    info!("Results saved to {}", results_path.display());

    Ok(())
}
