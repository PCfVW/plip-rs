//! Steering Experiment: Dose-Response Testing
//!
//! This script runs dose-response experiments to measure how different
//! levels of attention steering affect model behavior (via KL divergence).
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example steering_experiment
//! cargo run --release --example steering_experiment -- --model "Qwen/Qwen2.5-Coder-7B-Instruct"
//! cargo run --release --example steering_experiment -- --layer 16 --target-attention 0.09
//! ```

use anyhow::Result;
use clap::Parser;
use plip_rs::{
    kl_divergence, AttentionEdge, HeadSpec, InterventionType, LayerSpec, PlipModel, SteeringSpec,
    DOSE_LEVELS,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "steering_experiment")]
#[command(about = "Run dose-response steering experiments")]
struct Args {
    /// HuggingFace model ID
    #[arg(short, long, default_value = "Qwen/Qwen2.5-Coder-3B-Instruct")]
    model: String,

    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,

    /// Specific layer to intervene on (default: auto-select based on model size)
    #[arg(long)]
    layer: Option<usize>,

    /// Target attention level for SetValue intervention (0.0-1.0)
    #[arg(long)]
    target_attention: Option<f32>,

    /// Path to corpus file
    #[arg(long)]
    corpus: Option<PathBuf>,

    /// Output file for results (JSON)
    #[arg(long)]
    output: Option<PathBuf>,

    /// Verbose output
    #[arg(long, short)]
    verbose: bool,

    /// Number of samples to process (default: all)
    #[arg(long)]
    max_samples: Option<usize>,
}

/// Corpus sample from JSON
#[derive(Debug, Deserialize)]
struct Sample {
    id: String,
    code: String,
    marker_char_pos: usize,
    #[allow(dead_code)]
    marker_pattern: String,
    target_char_positions: Vec<usize>,
}

/// Full corpus structure
#[derive(Debug, Deserialize)]
struct Corpus {
    #[allow(dead_code)]
    python_doctest: Vec<Sample>,
    rust_test: Vec<Sample>,
}

/// Results for a single dose level
#[derive(Debug, Clone, Serialize)]
struct DoseResult {
    scale_factor: f32,
    mean_kl_divergence: f32,
    std_kl_divergence: f32,
    mean_attention_achieved: f32,
    n_samples: usize,
}

/// Full experiment results
#[derive(Debug, Serialize)]
struct ExperimentResults {
    model: String,
    layer: usize,
    intervention_type: String,
    target_attention: Option<f32>,
    dose_results: Vec<DoseResult>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("=== Steering Experiment: Dose-Response Testing ===\n");

    // Find corpus file
    let corpus_path = args.corpus.clone().unwrap_or_else(|| {
        PathBuf::from("corpus/attention_samples_universal.json")
    });

    println!("Loading corpus from: {}", corpus_path.display());
    let corpus_content = fs::read_to_string(&corpus_path)?;
    let corpus: Corpus = serde_json::from_str(&corpus_content)?;

    let rust_samples: Vec<&Sample> = if let Some(max) = args.max_samples {
        corpus.rust_test.iter().take(max).collect()
    } else {
        corpus.rust_test.iter().collect()
    };

    println!("Using {} Rust test samples\n", rust_samples.len());

    // Load model
    println!("Loading model: {}", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    println!(
        "Model loaded: {} layers, {} heads\n",
        model.n_layers(),
        model.n_heads()
    );

    // Determine target layer
    let target_layer = args.layer.unwrap_or_else(|| {
        let n_layers = model.n_layers();
        if n_layers >= 32 {
            20
        } else if n_layers >= 28 {
            16
        } else {
            14
        }
    });

    println!("Intervening at layer {}\n", target_layer);

    // Determine intervention type
    let intervention_type_str = if args.target_attention.is_some() {
        "SetValue"
    } else {
        "Scale"
    };

    println!("Intervention type: {}", intervention_type_str);
    if let Some(target) = args.target_attention {
        println!("Target attention: {:.2}%\n", target * 100.0);
    } else {
        println!("Testing dose levels: {:?}\n", DOSE_LEVELS);
    }

    // Run dose-response experiment
    let mut dose_results = Vec::new();

    let dose_levels: Vec<f32> = if let Some(target) = args.target_attention {
        // For SetValue, we test the single target
        vec![target]
    } else {
        // For Scale, we test all dose levels
        DOSE_LEVELS.to_vec()
    };

    for &dose in &dose_levels {
        println!("Testing dose: {:.1}x...", dose);

        let mut kl_values = Vec::new();
        let mut attention_values = Vec::new();

        for sample in &rust_samples {
            // Get token positions
            let encoding = match model.tokenize_with_offsets(&sample.code) {
                Ok(enc) => enc,
                Err(e) => {
                    if args.verbose {
                        println!("  Skipping {}: tokenization error - {}", sample.id, e);
                    }
                    continue;
                }
            };

            let marker_token_pos = match encoding.char_to_token(sample.marker_char_pos) {
                Some(pos) => pos,
                None => {
                    if args.verbose {
                        println!("  Skipping {}: marker position not found", sample.id);
                    }
                    continue;
                }
            };

            let target_token_positions: Vec<usize> = sample
                .target_char_positions
                .iter()
                .filter_map(|&pos| encoding.char_to_token(pos))
                .collect();

            if target_token_positions.is_empty() {
                if args.verbose {
                    println!("  Skipping {}: no target positions found", sample.id);
                }
                continue;
            }

            // Create edges from marker to targets
            let edges: Vec<AttentionEdge> = target_token_positions
                .iter()
                .map(|&to_pos| AttentionEdge {
                    from_pos: marker_token_pos,
                    to_pos,
                })
                .collect();

            // Create steering spec
            let intervention_type = if args.target_attention.is_some() {
                InterventionType::SetValue(dose)
            } else {
                InterventionType::Scale(dose)
            };

            let spec = SteeringSpec {
                layers: LayerSpec::Specific(vec![target_layer]),
                heads: HeadSpec::All,
                edges: edges.clone(),
                intervention_type,
            };

            // Get baseline (no intervention) logits
            let baseline_result = match model.get_attention(&sample.code) {
                Ok(cache) => cache,
                Err(e) => {
                    if args.verbose {
                        println!("  Skipping {}: baseline forward error - {}", sample.id, e);
                    }
                    continue;
                }
            };

            // Get steered logits
            let steered_result = match model.forward_steered_only(&sample.code, &spec) {
                Ok((logits, cache)) => (logits, cache),
                Err(e) => {
                    if args.verbose {
                        println!("  Skipping {}: steered forward error - {}", sample.id, e);
                    }
                    continue;
                }
            };

            // Compute KL divergence between baseline and steered
            // We need to get the logits from baseline too
            let baseline_logits = match model.forward(&sample.code) {
                Ok((logits, _)) => logits,
                Err(e) => {
                    if args.verbose {
                        println!(
                            "  Skipping {}: baseline logits forward error - {}",
                            sample.id, e
                        );
                    }
                    continue;
                }
            };

            // Compute KL divergence at the last token position
            let kl = match kl_divergence(&baseline_logits, &steered_result.0) {
                Ok(kl) => kl,
                Err(e) => {
                    if args.verbose {
                        println!("  Skipping {}: KL computation error - {}", sample.id, e);
                    }
                    continue;
                }
            };

            kl_values.push(kl);

            // Measure achieved attention level
            if let Ok(achieved_attn) = plip_rs::measure_attention_to_targets(
                &steered_result.1,
                marker_token_pos,
                &target_token_positions,
                target_layer,
            ) {
                attention_values.push(achieved_attn);
            }

            // Also measure baseline attention for comparison
            if args.verbose {
                if let Ok(baseline_attn) = plip_rs::measure_attention_to_targets(
                    &baseline_result,
                    marker_token_pos,
                    &target_token_positions,
                    target_layer,
                ) {
                    let achieved = attention_values.last().unwrap_or(&0.0);
                    println!(
                        "  {}: baseline={:.2}%, achieved={:.2}%, KL={:.4}",
                        sample.id,
                        baseline_attn * 100.0,
                        achieved * 100.0,
                        kl
                    );
                }
            }
        }

        // Compute statistics
        if kl_values.is_empty() {
            println!("  No valid samples for this dose level");
            continue;
        }

        let n_samples = kl_values.len();
        let mean_kl = kl_values.iter().sum::<f32>() / n_samples as f32;
        let std_kl = if n_samples > 1 {
            let variance = kl_values
                .iter()
                .map(|&x| (x - mean_kl).powi(2))
                .sum::<f32>()
                / (n_samples - 1) as f32;
            variance.sqrt()
        } else {
            0.0
        };

        let mean_attention = if !attention_values.is_empty() {
            attention_values.iter().sum::<f32>() / attention_values.len() as f32
        } else {
            0.0
        };

        dose_results.push(DoseResult {
            scale_factor: dose,
            mean_kl_divergence: mean_kl,
            std_kl_divergence: std_kl,
            mean_attention_achieved: mean_attention,
            n_samples,
        });

        println!(
            "  KL: {:.4} ± {:.4}, Attention: {:.2}%, N={}",
            mean_kl,
            std_kl,
            mean_attention * 100.0,
            n_samples
        );
    }

    // Print summary table
    println!("\n============================================================");
    println!("DOSE-RESPONSE RESULTS");
    println!("============================================================\n");

    println!("Model: {}", args.model);
    println!("Layer: {}", target_layer);
    println!("Intervention: {}", intervention_type_str);
    println!();

    println!("Dose    | KL Divergence      | Attention    | Samples");
    println!("--------|--------------------|--------------|---------");

    for result in &dose_results {
        let dose_str = if args.target_attention.is_some() {
            format!("{:.2}%", result.scale_factor * 100.0)
        } else {
            format!("{:.1}×", result.scale_factor)
        };

        println!(
            "{:7} | {:.4} ± {:.4}     | {:6.2}%      | {}",
            dose_str,
            result.mean_kl_divergence,
            result.std_kl_divergence,
            result.mean_attention_achieved * 100.0,
            result.n_samples
        );
    }

    // Analysis summary
    if dose_results.len() > 1 {
        println!("\n=== Analysis ===\n");

        // Find baseline (scale = 1.0)
        let baseline_result = dose_results
            .iter()
            .find(|r| (r.scale_factor - 1.0).abs() < 0.1);

        if let Some(baseline) = baseline_result {
            println!(
                "Baseline (1.0×): KL={:.4}, Attention={:.2}%",
                baseline.mean_kl_divergence,
                baseline.mean_attention_achieved * 100.0
            );

            // Find max KL divergence
            if let Some(max_kl) = dose_results.iter().max_by(|a, b| {
                a.mean_kl_divergence
                    .partial_cmp(&b.mean_kl_divergence)
                    .unwrap()
            }) {
                println!(
                    "Max KL at {:.1}×: KL={:.4}, Attention={:.2}%",
                    max_kl.scale_factor,
                    max_kl.mean_kl_divergence,
                    max_kl.mean_attention_achieved * 100.0
                );
            }
        }

        // Trend analysis
        let kl_trend: Vec<f32> = dose_results.iter().map(|r| r.mean_kl_divergence).collect();
        let is_monotonic =
            kl_trend.windows(2).all(|w| w[0] <= w[1]) || kl_trend.windows(2).all(|w| w[0] >= w[1]);

        if is_monotonic {
            println!("KL divergence shows monotonic trend with dose");
        } else {
            println!("KL divergence shows non-monotonic relationship with dose");
        }
    }

    // Save results if output path specified
    let results = ExperimentResults {
        model: args.model.clone(),
        layer: target_layer,
        intervention_type: intervention_type_str.to_string(),
        target_attention: args.target_attention,
        dose_results,
    };

    if let Some(output_path) = args.output {
        let json = serde_json::to_string_pretty(&results)?;
        fs::write(&output_path, json)?;
        println!("\nResults saved to: {}", output_path.display());
    }

    println!("\n=== Suggested Next Steps ===\n");
    println!("1. Compare KL divergence trends across models");
    println!("2. Test with target-attention to match Python levels:");
    println!("   cargo run --release --example steering_experiment -- --target-attention 0.09");
    println!("3. Analyze if higher attention correlates with better test preservation");

    println!();

    Ok(())
}
