//! State Ablation Experiment: Does removing recurrent state break preservation? (RWKV-6)
//!
//! This experiment tests the causal hypothesis that recurrent state flow from
//! test markers is critical for test preservation behavior in RWKV-6.
//!
//! ## Hypothesis
//!
//! If marker position state is causally important for "copying" expected values,
//! then knocking out the state write at the marker position should:
//! - Significantly change the model's next-token distribution (high KL divergence)
//! - Have different effects on Python doctests vs Rust tests (language-specific patterns)
//!
//! ## Methodology
//!
//! 1. Load samples from attention_samples_universal.json
//! 2. For each sample:
//!    - Convert character positions to token positions
//!    - Run baseline forward pass (no intervention)
//!    - Run state-knockout forward pass (suppress state write at marker position)
//!    - Compute KL divergence
//! 3. Compare Python vs Rust with statistical testing (Welch's t-test)
//!
//! ## Semantic Equivalence to Transformer Ablation
//!
//! RWKV-6 state knockout at a position is semantically equivalent to transformer
//! "all-edge" knockout: the marker position becomes invisible to all future tokens.
//! Results are directly comparable with `ablation_experiment.rs --all-edges`.
//!
//! Usage:
//!   cargo run --release --example state_ablation_experiment
//!   cargo run --release --example state_ablation_experiment -- --scan-layers
//!   cargo run --release --example state_ablation_experiment -- --layer 14 --output results.json

// Allow intentional float comparisons for division-by-zero guards
#![allow(clippy::suboptimal_flops)]
// Allow standard modulo pattern for even/odd checks (is_multiple_of is unstable)
#![allow(clippy::manual_is_multiple_of)]
// These are acceptable in numerical/statistics code:
#![allow(clippy::cast_precision_loss)] // usize→f32 intentional in statistics
#![allow(clippy::similar_names)] // am_new/az_new standard in math algorithms
#![allow(clippy::many_single_char_names)] // x, a, b, m standard in math
#![allow(clippy::doc_markdown)] // backticks for every technical term is excessive
#![allow(clippy::struct_excessive_bools)] // CLI args struct needs multiple bool flags
#![allow(clippy::unreadable_literal)] // float constants from reference papers

use anyhow::Result;
use clap::Parser;
use plip_rs::{PlipModel, StateKnockoutSpec};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "state_ablation_experiment")]
#[command(about = "Test whether RWKV-6 state knockout breaks test preservation")]
struct Args {
    /// HuggingFace model ID
    #[arg(short, long, default_value = "RWKV/v6-Finch-1B6-HF")]
    model: String,

    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,

    /// Specific layer to test (default: auto-select based on model size)
    #[arg(long)]
    layer: Option<usize>,

    /// Start layer for contiguous window knockout (use with --layer-end)
    #[arg(long)]
    layer_start: Option<usize>,

    /// End layer for contiguous window knockout (inclusive, use with --layer-start)
    #[arg(long)]
    layer_end: Option<usize>,

    /// Test all layers and report per-layer effects
    #[arg(long)]
    scan_layers: bool,

    /// Scan windows of increasing size around best layer
    #[arg(long)]
    scan_windows: bool,

    /// Center layer for window scan (required with --scan-windows)
    #[arg(long)]
    window_center: Option<usize>,

    /// Maximum window radius for window scan (default: 5)
    #[arg(long, default_value = "5")]
    max_radius: usize,

    /// Slide a fixed-size window across all layers (specify window size)
    #[arg(long)]
    slide_window: Option<usize>,

    /// Path to corpus file
    #[arg(long)]
    corpus: Option<PathBuf>,

    /// Output JSON file for results
    #[arg(long, short)]
    output: Option<PathBuf>,

    /// Include baseline samples (non-test code with similar markers)
    #[arg(long)]
    include_baselines: bool,

    /// Verbose output
    #[arg(long, short)]
    verbose: bool,
}

/// Corpus sample from JSON
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Sample {
    id: String,
    code: String,
    marker_char_pos: usize,
    marker_pattern: String,
    target_char_positions: Vec<usize>,
}

/// Full corpus structure
#[derive(Debug, Deserialize)]
struct Corpus {
    python_doctest: Vec<Sample>,
    rust_test: Vec<Sample>,
    #[serde(default)]
    python_baseline: Vec<Sample>,
    #[serde(default)]
    rust_baseline: Vec<Sample>,
}

/// Result for a single sample
#[derive(Debug, Serialize)]
struct SampleResult {
    id: String,
    language: String,
    is_baseline: bool,
    kl_divergence: f32,
    marker_token_pos: usize,
    target_token_positions: Vec<usize>,
    layers_knocked_out: usize,
    top_changed_tokens: Vec<TokenChange>,
}

/// Token probability change
#[derive(Debug, Serialize)]
struct TokenChange {
    token: String,
    baseline_prob: f32,
    ablated_prob: f32,
    diff: f32,
}

/// Statistical summary for a group
#[derive(Debug, Serialize)]
struct GroupStats {
    language: String,
    is_baseline: bool,
    n_samples: usize,
    mean_kl: f32,
    std_kl: f32,
    min_kl: f32,
    max_kl: f32,
    median_kl: f32,
}

/// Full experiment results (single-layer or specific-layers mode)
#[derive(Debug, Serialize)]
struct ExperimentResults {
    model: String,
    architecture: String,
    intervention_type: String,
    layer: usize,
    sample_results: Vec<SampleResult>,
    python_stats: GroupStats,
    rust_stats: GroupStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    python_baseline_stats: Option<GroupStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rust_baseline_stats: Option<GroupStats>,
    welch_t_statistic: f32,
    welch_p_value: f32,
    significant_difference: bool,
}

/// Window experiment results
#[derive(Debug, Serialize)]
struct WindowExperimentResults {
    model: String,
    architecture: String,
    intervention_type: String,
    layer_start: usize,
    layer_end: usize,
    n_layers_knocked_out: usize,
    sample_results: Vec<SampleResult>,
    python_stats: GroupStats,
    rust_stats: GroupStats,
    welch_t_statistic: f32,
    welch_p_value: f32,
    significant_difference: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("=== State Ablation Experiment: Does State Knockout Break Preservation? (RWKV-6) ===\n");

    // Find corpus file
    let corpus_path = args
        .corpus
        .clone()
        .unwrap_or_else(|| PathBuf::from("corpus/attention_samples_universal.json"));

    println!("Loading corpus from: {}", corpus_path.display());
    let corpus_content = fs::read_to_string(&corpus_path)?;
    let corpus: Corpus = serde_json::from_str(&corpus_content)?;

    println!(
        "Loaded {} Python samples, {} Rust samples",
        corpus.python_doctest.len(),
        corpus.rust_test.len()
    );
    if args.include_baselines {
        println!(
            "Including {} Python baselines, {} Rust baselines",
            corpus.python_baseline.len(),
            corpus.rust_baseline.len()
        );
    }

    // Load model
    println!("\nLoading model: {}", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    println!(
        "Model loaded: {} layers (RWKV-6 architecture)\n",
        model.n_layers()
    );

    // Determine target layer
    // Heuristic: ~60% through the model (where semantic processing happens)
    let target_layer = args.layer.unwrap_or_else(|| {
        let n_layers = model.n_layers();
        // 60% of 24 = 14.4 → 14
        (n_layers * 3) / 5
    });

    if args.scan_layers {
        run_layer_scan(&model, &corpus, &args)?;
    } else if let Some(window_size) = args.slide_window {
        run_sliding_window_scan(&model, &corpus, window_size, &args)?;
    } else if args.scan_windows {
        let center = args.window_center.unwrap_or(target_layer);
        run_window_scan(&model, &corpus, center, args.max_radius, &args)?;
    } else if args.layer_start.is_some() && args.layer_end.is_some() {
        // Contiguous layer window
        let start = args.layer_start.unwrap();
        let end = args.layer_end.unwrap();
        let results = run_experiment_window(&model, &corpus, start, end, &args)?;
        print_results_window(&results);

        if let Some(output_path) = &args.output {
            let json = serde_json::to_string_pretty(&results)?;
            fs::write(output_path, json)?;
            println!("\nResults saved to: {}", output_path.display());
        }
    } else {
        let results = run_experiment(&model, &corpus, target_layer, &args)?;
        print_results(&results);

        if let Some(output_path) = &args.output {
            let json = serde_json::to_string_pretty(&results)?;
            fs::write(output_path, json)?;
            println!("\nResults saved to: {}", output_path.display());
        }
    }

    Ok(())
}

/// Run the main state ablation experiment at a single layer
#[allow(clippy::too_many_lines, clippy::unnecessary_wraps)]
fn run_experiment(
    model: &PlipModel,
    corpus: &Corpus,
    layer: usize,
    args: &Args,
) -> Result<ExperimentResults> {
    println!("Running state knockout at layer {layer}\n");

    let mut sample_results = Vec::new();

    // Process Python samples
    println!("Processing Python doctest samples...");
    for sample in &corpus.python_doctest {
        match process_sample(model, sample, "python", false, layer, args) {
            Ok(result) => {
                if args.verbose {
                    println!("  {}: KL = {:.6}", sample.id, result.kl_divergence);
                }
                sample_results.push(result);
            }
            Err(e) => {
                eprintln!("  {} failed: {}", sample.id, e);
            }
        }
    }

    // Process Rust samples
    println!("Processing Rust test samples...");
    for sample in &corpus.rust_test {
        match process_sample(model, sample, "rust", false, layer, args) {
            Ok(result) => {
                if args.verbose {
                    println!("  {}: KL = {:.6}", sample.id, result.kl_divergence);
                }
                sample_results.push(result);
            }
            Err(e) => {
                eprintln!("  {} failed: {}", sample.id, e);
            }
        }
    }

    // Process baseline samples if requested
    if args.include_baselines {
        println!("Processing Python baseline samples...");
        for sample in &corpus.python_baseline {
            match process_sample(model, sample, "python", true, layer, args) {
                Ok(result) => {
                    if args.verbose {
                        println!("  {}: KL = {:.6}", sample.id, result.kl_divergence);
                    }
                    sample_results.push(result);
                }
                Err(e) => {
                    eprintln!("  {} failed: {}", sample.id, e);
                }
            }
        }

        println!("Processing Rust baseline samples...");
        for sample in &corpus.rust_baseline {
            match process_sample(model, sample, "rust", true, layer, args) {
                Ok(result) => {
                    if args.verbose {
                        println!("  {}: KL = {:.6}", sample.id, result.kl_divergence);
                    }
                    sample_results.push(result);
                }
                Err(e) => {
                    eprintln!("  {} failed: {}", sample.id, e);
                }
            }
        }
    }

    // Compute statistics
    let python_kls: Vec<f32> = sample_results
        .iter()
        .filter(|r| r.language == "python" && !r.is_baseline)
        .map(|r| r.kl_divergence)
        .collect();

    let rust_kls: Vec<f32> = sample_results
        .iter()
        .filter(|r| r.language == "rust" && !r.is_baseline)
        .map(|r| r.kl_divergence)
        .collect();

    let python_stats = compute_stats(&python_kls, "python", false);
    let rust_stats = compute_stats(&rust_kls, "rust", false);

    let python_baseline_stats = if args.include_baselines {
        let kls: Vec<f32> = sample_results
            .iter()
            .filter(|r| r.language == "python" && r.is_baseline)
            .map(|r| r.kl_divergence)
            .collect();
        Some(compute_stats(&kls, "python", true))
    } else {
        None
    };

    let rust_baseline_stats = if args.include_baselines {
        let kls: Vec<f32> = sample_results
            .iter()
            .filter(|r| r.language == "rust" && r.is_baseline)
            .map(|r| r.kl_divergence)
            .collect();
        Some(compute_stats(&kls, "rust", true))
    } else {
        None
    };

    // Welch's t-test
    let (t_stat, p_value) = welch_t_test(&python_kls, &rust_kls);
    let significant = p_value < 0.05;

    Ok(ExperimentResults {
        model: args.model.clone(),
        architecture: "rwkv6".to_string(),
        intervention_type: "state_knockout".to_string(),
        layer,
        sample_results,
        python_stats,
        rust_stats,
        python_baseline_stats,
        rust_baseline_stats,
        welch_t_statistic: t_stat,
        welch_p_value: p_value,
        significant_difference: significant,
    })
}

/// Process a single sample at a single layer
fn process_sample(
    model: &PlipModel,
    sample: &Sample,
    language: &str,
    is_baseline: bool,
    layer: usize,
    args: &Args,
) -> Result<SampleResult> {
    process_sample_layers(model, sample, language, is_baseline, layer, layer, args)
}

/// Process a single sample with a layer range
#[allow(clippy::too_many_arguments)]
fn process_sample_layers(
    model: &PlipModel,
    sample: &Sample,
    language: &str,
    is_baseline: bool,
    layer_start: usize,
    layer_end: usize,
    args: &Args,
) -> Result<SampleResult> {
    // Convert character positions to token positions
    let encoding = model.tokenize_with_offsets(&sample.code)?;

    // Find marker token position
    let marker_token_pos = encoding
        .char_to_token(sample.marker_char_pos)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Could not find marker token for char pos {}",
                sample.marker_char_pos
            )
        })?;

    // Find target token positions (metadata only — state knockout is position-based)
    let target_token_positions: Vec<usize> = sample
        .target_char_positions
        .iter()
        .filter_map(|&char_pos| encoding.char_to_token(char_pos))
        .collect();

    if args.verbose {
        eprintln!(
            "  [DEBUG] marker_char_pos={} -> token_pos={}, targets={:?}",
            sample.marker_char_pos, marker_token_pos, target_token_positions
        );
    }

    // Build state knockout spec
    let spec = if layer_start == layer_end {
        StateKnockoutSpec::new()
            .position(marker_token_pos)
            .layer(layer_start)
    } else {
        StateKnockoutSpec::new()
            .position(marker_token_pos)
            .layer_range(layer_start, layer_end)
    };

    let n_layers_knocked = layer_end - layer_start + 1;

    // Run state ablation (baseline + knockout in a single call)
    let result = model.forward_with_state_knockout(&sample.code, &spec)?;
    let kl = result.kl_divergence()?;

    // Get top changed tokens
    let changed = result.top_changed_tokens(5)?;
    let top_changed_tokens: Vec<TokenChange> = changed
        .iter()
        .map(|(token_id, base_prob, abl_prob, diff)| {
            let token = model.decode_token(*token_id);
            TokenChange {
                token: token.replace('\n', "\\n").replace('\t', "\\t"),
                baseline_prob: *base_prob,
                ablated_prob: *abl_prob,
                diff: *diff,
            }
        })
        .collect();

    Ok(SampleResult {
        id: sample.id.clone(),
        language: language.to_string(),
        is_baseline,
        kl_divergence: kl,
        marker_token_pos,
        target_token_positions,
        layers_knocked_out: n_layers_knocked,
        top_changed_tokens,
    })
}

/// Run a layer scan to find which layers matter most
#[allow(clippy::unnecessary_wraps)]
fn run_layer_scan(model: &PlipModel, corpus: &Corpus, args: &Args) -> Result<()> {
    println!("=== Layer Scan: Finding Most Important Layers (State Knockout) ===\n");

    let n_layers = model.n_layers();
    let mut layer_effects: Vec<(usize, f32, f32)> = Vec::new();

    // Sample one Python and one Rust sample for speed
    let py_sample = &corpus.python_doctest[0];
    let rust_sample = &corpus.rust_test[0];

    for layer in 0..n_layers {
        print!("Layer {layer:2}: ");

        let py_result = process_sample(model, py_sample, "python", false, layer, args);
        let rust_result = process_sample(model, rust_sample, "rust", false, layer, args);

        match (py_result, rust_result) {
            (Ok(py), Ok(rust)) => {
                println!(
                    "Python KL = {:.6}, Rust KL = {:.6}",
                    py.kl_divergence, rust.kl_divergence
                );
                layer_effects.push((layer, py.kl_divergence, rust.kl_divergence));
            }
            (Err(e), _) | (_, Err(e)) => {
                println!("Error: {e}");
            }
        }
    }

    // Find layers with highest effect
    println!("\n=== Top Layers by Effect ===\n");

    let mut by_python: Vec<_> = layer_effects.iter().collect();
    by_python.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("Top layers for Python doctest:");
    for (layer, py_kl, rust_kl) in by_python.iter().take(5) {
        println!(
            "  Layer {layer:2}: KL = {py_kl:.6} (Rust: {rust_kl:.6})"
        );
    }

    let mut by_rust: Vec<_> = layer_effects.iter().collect();
    by_rust.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop layers for Rust test:");
    for (layer, py_kl, rust_kl) in by_rust.iter().take(5) {
        println!(
            "  Layer {layer:2}: KL = {rust_kl:.6} (Python: {py_kl:.6})"
        );
    }

    Ok(())
}

/// Run experiment with a contiguous window of layers
#[allow(clippy::unnecessary_wraps)]
fn run_experiment_window(
    model: &PlipModel,
    corpus: &Corpus,
    layer_start: usize,
    layer_end: usize,
    args: &Args,
) -> Result<WindowExperimentResults> {
    println!(
        "Running state knockout on layers {}-{} (window of {} layers)\n",
        layer_start,
        layer_end,
        layer_end - layer_start + 1
    );

    let mut sample_results = Vec::new();

    // Process Python samples
    println!("Processing Python doctest samples...");
    for sample in &corpus.python_doctest {
        match process_sample_layers(model, sample, "python", false, layer_start, layer_end, args) {
            Ok(result) => {
                if args.verbose {
                    println!("  {}: KL = {:.6}", sample.id, result.kl_divergence);
                }
                sample_results.push(result);
            }
            Err(e) => {
                eprintln!("  {} failed: {}", sample.id, e);
            }
        }
    }

    // Process Rust samples
    println!("Processing Rust test samples...");
    for sample in &corpus.rust_test {
        match process_sample_layers(model, sample, "rust", false, layer_start, layer_end, args) {
            Ok(result) => {
                if args.verbose {
                    println!("  {}: KL = {:.6}", sample.id, result.kl_divergence);
                }
                sample_results.push(result);
            }
            Err(e) => {
                eprintln!("  {} failed: {}", sample.id, e);
            }
        }
    }

    // Compute statistics
    let python_kls: Vec<f32> = sample_results
        .iter()
        .filter(|r| r.language == "python" && !r.is_baseline)
        .map(|r| r.kl_divergence)
        .collect();

    let rust_kls: Vec<f32> = sample_results
        .iter()
        .filter(|r| r.language == "rust" && !r.is_baseline)
        .map(|r| r.kl_divergence)
        .collect();

    let python_stats = compute_stats(&python_kls, "python", false);
    let rust_stats = compute_stats(&rust_kls, "rust", false);

    // Welch's t-test
    let (t_stat, p_value) = welch_t_test(&python_kls, &rust_kls);
    let significant = p_value < 0.05;

    Ok(WindowExperimentResults {
        model: args.model.clone(),
        architecture: "rwkv6".to_string(),
        intervention_type: "state_knockout".to_string(),
        layer_start,
        layer_end,
        n_layers_knocked_out: layer_end - layer_start + 1,
        sample_results,
        python_stats,
        rust_stats,
        welch_t_statistic: t_stat,
        welch_p_value: p_value,
        significant_difference: significant,
    })
}

/// Scan windows of increasing size around a center layer
fn run_window_scan(
    model: &PlipModel,
    corpus: &Corpus,
    center: usize,
    max_radius: usize,
    args: &Args,
) -> Result<()> {
    println!(
        "=== Window Scan: Testing Increasing Windows Around Layer {center} (State Knockout) ===\n"
    );

    let n_layers = model.n_layers();
    let py_sample = &corpus.python_doctest[0];
    let rust_sample = &corpus.rust_test[0];

    println!("Window       | Layers     | Python KL  | Rust KL    | Ratio");
    println!("-------------|------------|------------|------------|-------");

    // First, single layer (radius 0)
    let py_result = process_sample(model, py_sample, "python", false, center, args)?;
    let rust_result = process_sample(model, rust_sample, "rust", false, center, args)?;
    println!(
        "Layer {}      | {:2}         | {:.6}   | {:.6}   | {:.2}x",
        center,
        center,
        py_result.kl_divergence,
        rust_result.kl_divergence,
        if rust_result.kl_divergence > 1e-10 {
            py_result.kl_divergence / rust_result.kl_divergence
        } else {
            0.0
        }
    );

    // Increasing window sizes
    for radius in 1..=max_radius {
        let start = center.saturating_sub(radius);
        let end = (center + radius).min(n_layers - 1);

        let py_result =
            process_sample_layers(model, py_sample, "python", false, start, end, args)?;
        let rust_result =
            process_sample_layers(model, rust_sample, "rust", false, start, end, args)?;

        let ratio = if rust_result.kl_divergence > 1e-10 {
            py_result.kl_divergence / rust_result.kl_divergence
        } else {
            0.0
        };

        println!(
            "Layers {}-{:<2}  | {:2}-{:<2}      | {:.6}   | {:.6}   | {:.2}x",
            start, end, start, end, py_result.kl_divergence, rust_result.kl_divergence, ratio
        );
    }

    println!("\n=== Interpretation ===");
    println!("If KL increases with window size: Causal effect is distributed across layers");
    println!("If KL stays constant: All causal effect is in the center layer");
    println!("If KL decreases: Adjacent layers compensate for the knocked-out layer");

    Ok(())
}

/// Slide a fixed-size window across all layers
fn run_sliding_window_scan(
    model: &PlipModel,
    corpus: &Corpus,
    window_size: usize,
    args: &Args,
) -> Result<()> {
    let n_layers = model.n_layers();

    if window_size > n_layers {
        anyhow::bail!(
            "Window size {window_size} exceeds number of layers {n_layers}"
        );
    }

    println!(
        "=== Sliding Window Scan: Window Size {window_size} (State Knockout) ===\n"
    );
    println!(
        "Model: {} ({} layers, RWKV-6)",
        args.model, n_layers
    );
    println!();

    let py_sample = &corpus.python_doctest[0];
    let rust_sample = &corpus.rust_test[0];

    println!("Start | Window     | Python KL  | Rust KL    | Ratio      | Peak?");
    println!("------|------------|------------|------------|------------|------");

    let mut results: Vec<(usize, usize, f32, f32)> = Vec::new();
    let mut max_py_kl: f32 = 0.0;
    let mut max_rust_kl: f32 = 0.0;

    // Slide window from 0 to n_layers - window_size
    for start in 0..=(n_layers - window_size) {
        let end = start + window_size - 1;

        let py_result =
            process_sample_layers(model, py_sample, "python", false, start, end, args)?;
        let rust_result =
            process_sample_layers(model, rust_sample, "rust", false, start, end, args)?;

        let py_kl = py_result.kl_divergence;
        let rust_kl = rust_result.kl_divergence;

        if py_kl > max_py_kl {
            max_py_kl = py_kl;
        }
        if rust_kl > max_rust_kl {
            max_rust_kl = rust_kl;
        }

        results.push((start, end, py_kl, rust_kl));
    }

    // Print results with peak markers
    for (start, end, py_kl, rust_kl) in &results {
        let ratio = if *rust_kl > 1e-10 {
            py_kl / rust_kl
        } else {
            0.0
        };

        let is_py_peak = (*py_kl - max_py_kl).abs() < 1e-10;
        let is_rust_peak = (*rust_kl - max_rust_kl).abs() < 1e-10;

        let peak_marker = match (is_py_peak, is_rust_peak) {
            (true, true) => "PY+RS",
            (true, false) => "PY",
            (false, true) => "RS",
            (false, false) => "",
        };

        println!(
            "{start:5} | {start:2}-{end:<2}      | {py_kl:.6}   | {rust_kl:.6}   | {ratio:8.2}x  | {peak_marker}"
        );
    }

    // Summary
    println!("\n=== Summary ===");
    println!("Window size: {window_size} layers");
    println!(
        "Max Python KL:  {:.6} (at window starting at layer {})",
        max_py_kl,
        results
            .iter()
            .find(|(_, _, py, _)| (*py - max_py_kl).abs() < 1e-10)
            .map_or(&0, |(s, _, _, _)| s)
    );
    println!(
        "Max Rust KL:    {:.6} (at window starting at layer {})",
        max_rust_kl,
        results
            .iter()
            .find(|(_, _, _, rs)| (*rs - max_rust_kl).abs() < 1e-10)
            .map_or(&0, |(s, _, _, _)| s)
    );

    // Find where Python effect is highest relative to Rust
    let max_ratio_entry = results
        .iter()
        .filter(|(_, _, _, rs)| *rs > 1e-10)
        .max_by(|a, b| {
            (a.2 / a.3)
                .partial_cmp(&(b.2 / b.3))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

    if let Some((start, end, py, rs)) = max_ratio_entry {
        println!(
            "Max Python/Rust ratio: {:.2}x at layers {}-{}",
            py / rs,
            start,
            end
        );
    }

    Ok(())
}

/// Print formatted results for single-layer experiment
fn print_results(results: &ExperimentResults) {
    println!("\n============================================================");
    println!("EXPERIMENT RESULTS (State Knockout)");
    println!("============================================================\n");

    println!("Model: {}", results.model);
    println!("Architecture: {}", results.architecture);
    println!("Intervention: {}", results.intervention_type);
    println!("Layer tested: {}\n", results.layer);

    println!("=== Python Doctest Results ===");
    print_stats(&results.python_stats);

    println!("\n=== Rust Test Results ===");
    print_stats(&results.rust_stats);

    if let Some(ref stats) = results.python_baseline_stats {
        println!("\n=== Python Baseline Results ===");
        print_stats(stats);
    }

    if let Some(ref stats) = results.rust_baseline_stats {
        println!("\n=== Rust Baseline Results ===");
        print_stats(stats);
    }

    println!("\n=== Statistical Comparison (Python vs Rust) ===");
    println!("Welch's t-statistic: {:.4}", results.welch_t_statistic);
    println!("p-value: {:.6}", results.welch_p_value);
    println!(
        "Significant difference (p < 0.05): {}",
        if results.significant_difference {
            "YES"
        } else {
            "NO"
        }
    );

    // Interpretation
    println!("\n=== Interpretation ===");

    let py_mean = results.python_stats.mean_kl;
    let rust_mean = results.rust_stats.mean_kl;

    if py_mean > 0.01 || rust_mean > 0.01 {
        println!("FINDING: State knockout has measurable impact on predictions.");
        println!("         This supports the hypothesis that marker state flow");
        println!("         is causally important for preservation behavior.");
    } else {
        println!("FINDING: State knockout has minimal impact on predictions.");
        println!("         This may indicate the model uses redundant pathways,");
        println!("         or the targeted layer is not the critical one.");
    }

    if results.significant_difference {
        if py_mean > rust_mean {
            println!("\nFINDING: Python doctests are MORE affected by state knockout than Rust tests.");
            println!("         This suggests different preservation mechanisms per language.");
        } else {
            println!("\nFINDING: Rust tests are MORE affected by state knockout than Python doctests.");
            println!("         This suggests different preservation mechanisms per language.");
        }
    } else {
        println!("\nFINDING: No significant difference between Python and Rust knockout effects.");
        println!("         Both languages may use similar preservation mechanisms at this layer.");
    }
}

/// Print formatted window results
fn print_results_window(results: &WindowExperimentResults) {
    println!("\n============================================================");
    println!("WINDOW EXPERIMENT RESULTS (State Knockout)");
    println!("============================================================\n");

    println!("Model: {}", results.model);
    println!("Architecture: {}", results.architecture);
    println!("Intervention: {}", results.intervention_type);
    println!(
        "Layer window: {}-{} ({} layers)\n",
        results.layer_start, results.layer_end, results.n_layers_knocked_out
    );

    println!("=== Python Doctest Results ===");
    print_stats(&results.python_stats);

    println!("\n=== Rust Test Results ===");
    print_stats(&results.rust_stats);

    println!("\n=== Statistical Comparison (Python vs Rust) ===");
    println!("Welch's t-statistic: {:.4}", results.welch_t_statistic);
    println!("p-value: {:.6}", results.welch_p_value);
    println!(
        "Significant difference (p < 0.05): {}",
        if results.significant_difference {
            "YES"
        } else {
            "NO"
        }
    );

    // Interpretation
    println!("\n=== Interpretation ===");

    let py_mean = results.python_stats.mean_kl;
    let rust_mean = results.rust_stats.mean_kl;

    if py_mean > 0.01 || rust_mean > 0.01 {
        println!("FINDING: Multi-layer state knockout has measurable impact on predictions.");
        println!(
            "         Knocking out layers {}-{} significantly changes model behavior.",
            results.layer_start, results.layer_end
        );
    } else {
        println!("FINDING: Multi-layer state knockout has minimal impact on predictions.");
        println!("         The model appears to use redundant pathways across these layers.");
    }

    if results.significant_difference {
        if py_mean > rust_mean {
            println!("\nFINDING: Python doctests are MORE affected by this window knockout.");
        } else {
            println!("\nFINDING: Rust tests are MORE affected by this window knockout.");
        }
    } else {
        println!("\nFINDING: No significant difference between Python and Rust.");
    }
}

fn print_stats(stats: &GroupStats) {
    println!("  Samples: {}", stats.n_samples);
    println!("  Mean KL divergence: {:.6}", stats.mean_kl);
    println!("  Std deviation: {:.6}", stats.std_kl);
    println!("  Min: {:.6}", stats.min_kl);
    println!("  Max: {:.6}", stats.max_kl);
    println!("  Median: {:.6}", stats.median_kl);
}

/// Compute statistics for a group of KL values
fn compute_stats(kls: &[f32], language: &str, is_baseline: bool) -> GroupStats {
    let n = kls.len();
    if n == 0 {
        return GroupStats {
            language: language.to_string(),
            is_baseline,
            n_samples: 0,
            mean_kl: 0.0,
            std_kl: 0.0,
            min_kl: 0.0,
            max_kl: 0.0,
            median_kl: 0.0,
        };
    }

    let mean: f32 = kls.iter().sum::<f32>() / n as f32;
    let variance: f32 = kls.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let std = variance.sqrt();

    let mut sorted = kls.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if n % 2 == 0 {
        f32::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    } else {
        sorted[n / 2]
    };

    GroupStats {
        language: language.to_string(),
        is_baseline,
        n_samples: n,
        mean_kl: mean,
        std_kl: std,
        min_kl: sorted[0],
        max_kl: sorted[n - 1],
        median_kl: median,
    }
}

/// Welch's t-test for unequal variances
fn welch_t_test(group1: &[f32], group2: &[f32]) -> (f32, f32) {
    let n1 = group1.len() as f32;
    let n2 = group2.len() as f32;

    if n1 < 2.0 || n2 < 2.0 {
        return (0.0, 1.0);
    }

    let mean1: f32 = group1.iter().sum::<f32>() / n1;
    let mean2: f32 = group2.iter().sum::<f32>() / n2;

    let var1: f32 = group1.iter().map(|&x| (x - mean1).powi(2)).sum::<f32>() / (n1 - 1.0);
    let var2: f32 = group2.iter().map(|&x| (x - mean2).powi(2)).sum::<f32>() / (n2 - 1.0);

    let se = ((var1 / n1) + (var2 / n2)).sqrt();
    if se < 1e-10 {
        return (0.0, 1.0);
    }

    let t = (mean1 - mean2) / se;

    // Welch-Satterthwaite degrees of freedom
    let v1 = var1 / n1;
    let v2 = var2 / n2;
    let df_num = (v1 + v2).powi(2);
    let df_denom = (v1.powi(2) / (n1 - 1.0)) + (v2.powi(2) / (n2 - 1.0));
    let df = if df_denom > 0.0 {
        df_num / df_denom
    } else {
        n1 + n2 - 2.0
    };

    // Approximate p-value using Student's t distribution
    let p_value = 2.0 * (1.0 - t_cdf(t.abs(), df));

    (t, p_value)
}

/// Cumulative distribution function for Student's t distribution (approximation)
fn t_cdf(t: f32, df: f32) -> f32 {
    // Use approximation based on the normal distribution for large df
    if df > 30.0 {
        // Standard normal approximation
        let x = t / (1.0 + t * t / df).sqrt();
        return normal_cdf(x);
    }

    // For smaller df, use a series approximation
    let x = df / (df + t * t);
    let a = df / 2.0;
    let b = 0.5;

    // Incomplete beta function approximation
    let result = 0.5 * regularized_incomplete_beta(x, a, b);
    if t > 0.0 {
        1.0 - result
    } else {
        result
    }
}

/// Standard normal CDF approximation
fn normal_cdf(x: f32) -> f32 {
    0.5 * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f32) -> f32 {
    let a1 = 0.254_829_6_f32;
    let a2 = -0.284_496_72_f32;
    let a3 = 1.421_413_8_f32;
    let a4 = -1.453_152_1_f32;
    let a5 = 1.061_405_4_f32;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Regularized incomplete beta function approximation
fn regularized_incomplete_beta(x: f32, a: f32, b: f32) -> f32 {
    // Simple approximation using continued fraction
    if x < (a + 1.0) / (a + b + 2.0) {
        beta_cf(x, a, b) * x.powf(a) * (1.0 - x).powf(b) / (a * beta(a, b))
    } else {
        1.0 - beta_cf(1.0 - x, b, a) * (1.0 - x).powf(b) * x.powf(a) / (b * beta(a, b))
    }
}

/// Beta function using log-gamma
fn beta(a: f32, b: f32) -> f32 {
    (lgamma(a) + lgamma(b) - lgamma(a + b)).exp()
}

/// Log-gamma function approximation (Stirling)
fn lgamma(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::INFINITY;
    }
    0.5 * (2.0 * std::f32::consts::PI / x).ln()
        + x * ((x + 1.0 / (12.0 * x - 1.0 / (10.0 * x))).ln() - 1.0)
}

/// Continued fraction for incomplete beta function
fn beta_cf(x: f32, a: f32, b: f32) -> f32 {
    let max_iter = 100;
    let eps = 1e-7;

    let mut am = 1.0;
    let mut bm = 1.0;
    let mut az = 1.0;
    let mut bz = 1.0 - (a + b) * x / (a + 1.0);

    for m in 1..max_iter {
        let m = m as f32;

        // Even step
        let em = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        let d = 1.0 + em * az / bz;
        if d.abs() < 1e-30 {
            continue;
        }
        let az_new = az + em * am;
        let bz_new = bz + em * bm;
        let am_new = az;
        let bm_new = bz;

        // Odd step
        let ep = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        let az_final = az_new + ep * am_new;
        let bz_final = bz_new + ep * bm_new;

        let ratio = az_final / bz_final;
        if (ratio - am / bm).abs() < eps * ratio.abs() {
            return ratio;
        }

        am = am_new;
        bm = bm_new;
        az = az_final;
        bz = bz_final;
    }

    az / bz
}
