//! Steering Calibration: Measure baseline attention levels
//!
//! This script measures the baseline attention from test markers to function
//! tokens for both Python doctests and Rust tests. These measurements are used
//! to calibrate steering targets for dose-response experiments.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example steering_calibrate
//! cargo run --release --example steering_calibrate -- --model "Qwen/Qwen2.5-Coder-7B-Instruct"
//! cargo run --release --example steering_calibrate -- --layer 16
//! ```

use anyhow::Result;
use clap::Parser;
use plip_rs::{calibrate_from_samples, CalibrationSample, PlipModel};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "steering_calibrate")]
#[command(about = "Measure baseline attention levels for steering calibration")]
struct Args {
    /// HuggingFace model ID
    #[arg(short, long, default_value = "Qwen/Qwen2.5-Coder-3B-Instruct")]
    model: String,

    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,

    /// Specific layer to measure (default: auto-select based on model size)
    #[arg(long)]
    layer: Option<usize>,

    /// Path to corpus file
    #[arg(long)]
    corpus: Option<PathBuf>,

    /// Verbose output
    #[arg(long, short)]
    verbose: bool,
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
    python_doctest: Vec<Sample>,
    rust_test: Vec<Sample>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("=== Steering Calibration: Measure Baseline Attention Levels ===\n");

    // Find corpus file
    let corpus_path = args
        .corpus
        .clone()
        .unwrap_or_else(|| PathBuf::from("corpus/attention_samples_universal.json"));

    println!("Loading corpus from: {}", corpus_path.display());
    let corpus_content = fs::read_to_string(&corpus_path)?;
    let corpus: Corpus = serde_json::from_str(&corpus_content)?;

    println!(
        "Loaded {} Python samples, {} Rust samples\n",
        corpus.python_doctest.len(),
        corpus.rust_test.len()
    );

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

    println!("Measuring attention at layer {}\n", target_layer);

    // Convert corpus samples to CalibrationSample format
    let python_samples: Vec<CalibrationSample> = corpus
        .python_doctest
        .iter()
        .map(|s| {
            CalibrationSample::new(
                &s.id,
                &s.code,
                s.marker_char_pos,
                s.target_char_positions.clone(),
            )
        })
        .collect();

    let rust_samples: Vec<CalibrationSample> = corpus
        .rust_test
        .iter()
        .map(|s| {
            CalibrationSample::new(
                &s.id,
                &s.code,
                s.marker_char_pos,
                s.target_char_positions.clone(),
            )
        })
        .collect();

    // Run calibration
    println!("Measuring Python doctest attention...");
    if args.verbose {
        for sample in &python_samples {
            match model.get_attention(&sample.code) {
                Ok(attn_cache) => {
                    let encoding = model.tokenize_with_offsets(&sample.code)?;
                    if let Some(marker_pos) = encoding.char_to_token(sample.marker_char_pos) {
                        let target_positions: Vec<usize> = sample
                            .target_char_positions
                            .iter()
                            .filter_map(|&pos| encoding.char_to_token(pos))
                            .collect();

                        if let Ok(attn) = plip_rs::measure_attention_to_targets(
                            &attn_cache,
                            marker_pos,
                            &target_positions,
                            target_layer,
                        ) {
                            println!("  {}: {:.4}%", sample.id, attn * 100.0);
                        }
                    }
                }
                Err(e) => println!("  {}: Error - {}", sample.id, e),
            }
        }
    }

    println!("Measuring Rust test attention...");
    if args.verbose {
        for sample in &rust_samples {
            match model.get_attention(&sample.code) {
                Ok(attn_cache) => {
                    let encoding = model.tokenize_with_offsets(&sample.code)?;
                    if let Some(marker_pos) = encoding.char_to_token(sample.marker_char_pos) {
                        let target_positions: Vec<usize> = sample
                            .target_char_positions
                            .iter()
                            .filter_map(|&pos| encoding.char_to_token(pos))
                            .collect();

                        if let Ok(attn) = plip_rs::measure_attention_to_targets(
                            &attn_cache,
                            marker_pos,
                            &target_positions,
                            target_layer,
                        ) {
                            println!("  {}: {:.4}%", sample.id, attn * 100.0);
                        }
                    }
                }
                Err(e) => println!("  {}: Error - {}", sample.id, e),
            }
        }
    }

    // Get calibration results
    let calibration = calibrate_from_samples(&model, &python_samples, &rust_samples, target_layer)?;

    // Print results
    println!("\n============================================================");
    println!("CALIBRATION RESULTS");
    println!("============================================================\n");

    println!("Model: {}", args.model);
    println!("Layer: {}", calibration.layer);
    println!(
        "Samples: {} Python, {} Rust\n",
        calibration.n_python_samples, calibration.n_rust_samples
    );

    println!("=== Baseline Attention Levels ===\n");
    println!(
        "  Python doctest (>>> → fn):  {:.2}%",
        calibration.python_baseline * 100.0
    );
    println!(
        "  Rust test (#[test] → fn):   {:.2}%",
        calibration.rust_baseline * 100.0
    );
    println!(
        "  Ratio (Python/Rust):        {:.2}×",
        calibration.attention_ratio
    );

    println!("\n=== Recommended Steering Targets ===\n");
    println!(
        "  Target (Python level):      {:.2}%",
        calibration.recommended_target * 100.0
    );
    println!(
        "  Scale factor for Rust:      {:.2}×",
        calibration.scale_factor_to_python()
    );

    println!("\n=== Dose-Response Levels ===\n");
    println!("  Scale  | Absolute Attention");
    println!("  -------|-------------------");
    for (scale, absolute) in calibration.dose_levels_absolute() {
        let marker = if (scale - calibration.scale_factor_to_python()).abs() < 0.5 {
            " ← Python level"
        } else if (scale - 1.0).abs() < 0.1 {
            " ← Baseline"
        } else {
            ""
        };
        println!("  {:.1}×   | {:.2}%{}", scale, absolute * 100.0, marker);
    }

    println!("\n=== Suggested Experiment Commands ===\n");
    println!("  # Run dose-response experiment with these calibrated values:");
    println!("  cargo run --release --example steering_experiment -- \\",);
    println!("    --model \"{}\" \\", args.model);
    println!("    --layer {} \\", target_layer);
    println!(
        "    --target-attention {:.4}",
        calibration.recommended_target
    );

    println!();

    Ok(())
}
