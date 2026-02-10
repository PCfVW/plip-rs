//! State Steering Experiment: Dose-response curve for RWKV-6 state intervention
//!
//! This experiment tests whether scaling the kv write to recurrent state produces
//! graded effects on the model's predictions, extending the knockout-predicts-steering
//! hypothesis from transformer architectures to RWKV-6.
//!
//! ## Hypothesis
//!
//! If marker position state is causally important for test preservation behavior,
//! then amplifying (scale > 1) or dampening (scale < 1) the state write should
//! produce graded KL divergence that correlates with the scale factor, and the
//! effect should differ between Python doctests and Rust tests.
//!
//! ## Methodology
//!
//! 1. Load samples from attention_samples_universal.json
//! 2. For each scale factor (0.0, 0.5, 1.0, 2.0, 5.0, 9.0):
//!    - For each sample: run `forward_with_state_steering` at target layer
//!    - Compute KL divergence between baseline and steered predictions
//! 3. Produce dose-response table: scale -> Python KL, Rust KL, ratio, p-value
//!
//! ## Semantic Equivalence
//!
//! - scale=0.0: State knockout (should match `state_ablation_experiment`)
//! - scale=1.0: Identity (KL should be ~0, sanity check)
//! - scale>1.0: Amplification (analogous to transformer attention steering)
//!
//! Usage:
//!   cargo run --release --example state_steering_experiment
//!   cargo run --release --example state_steering_experiment -- --layer 14 --output results.json
//!   cargo run --release --example state_steering_experiment -- --scales 0.0,0.5,1.0,2.0,5.0,9.0

// Clippy configuration for ML/statistics code
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::print_literal)]

use anyhow::Result;
use clap::Parser;
use plip_rs::{PlipModel, StateSteeringSpec};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "state_steering_experiment")]
#[command(about = "Dose-response curve for RWKV-6 state steering intervention")]
struct Args {
    /// HuggingFace model ID
    #[arg(short, long, default_value = "RWKV/v6-Finch-1B6-HF")]
    model: String,

    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,

    /// Target layer for steering (default: 60% through model)
    #[arg(long)]
    layer: Option<usize>,

    /// Comma-separated scale factors (default: 0.0,0.5,1.0,2.0,5.0,9.0)
    #[arg(long, default_value = "0.0,0.5,1.0,2.0,5.0,9.0")]
    scales: String,

    /// Path to corpus file
    #[arg(long)]
    corpus: Option<PathBuf>,

    /// Output JSON file for results
    #[arg(long, short)]
    output: Option<PathBuf>,

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
}

/// Result for a single sample at a single scale
#[derive(Debug, Serialize)]
struct SampleResult {
    id: String,
    language: String,
    scale: f32,
    kl_divergence: f32,
    marker_token_pos: usize,
}

/// Statistics for a group at a given scale
#[derive(Debug, Serialize)]
struct GroupStats {
    language: String,
    n_samples: usize,
    mean_kl: f32,
    std_kl: f32,
    min_kl: f32,
    max_kl: f32,
    median_kl: f32,
}

/// Dose-response point: one scale factor's results
#[derive(Debug, Serialize)]
struct DosePoint {
    scale: f32,
    python_stats: GroupStats,
    rust_stats: GroupStats,
    welch_t_statistic: f32,
    welch_p_value: f32,
    significant_difference: bool,
    sample_results: Vec<SampleResult>,
}

/// Full experiment results
#[derive(Debug, Serialize)]
struct ExperimentResults {
    model: String,
    architecture: String,
    intervention_type: String,
    layer: usize,
    dose_response: Vec<DosePoint>,
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("=== State Steering Experiment: Dose-Response Curve (RWKV-6) ===\n");

    // Parse scale factors
    let scales: Vec<f32> = args
        .scales
        .split(',')
        .map(|s| s.trim().parse::<f32>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    println!("Scale factors: {:?}", scales);

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

    // Load model
    println!("\nLoading model: {}", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    let n_layers = model.n_layers();
    println!("Model loaded: {n_layers} layers (RWKV-6 architecture)\n");

    // Determine target layer (60% through the model)
    let target_layer = args.layer.unwrap_or_else(|| (n_layers * 3) / 5);
    println!("Target layer: {target_layer}\n");

    // Run dose-response experiment
    let mut dose_response = Vec::new();

    for &scale in &scales {
        println!("--- Scale = {scale:.1} ---");

        let mut sample_results = Vec::new();

        // Process Python samples
        if args.verbose {
            println!("  Processing Python doctest samples...");
        }
        for sample in &corpus.python_doctest {
            match process_sample(&model, sample, "python", target_layer, scale, &args) {
                Ok(result) => {
                    if args.verbose {
                        println!("    {}: KL = {:.6}", sample.id, result.kl_divergence);
                    }
                    sample_results.push(result);
                }
                Err(e) => {
                    eprintln!("    {} failed: {e}", sample.id);
                }
            }
        }

        // Process Rust samples
        if args.verbose {
            println!("  Processing Rust test samples...");
        }
        for sample in &corpus.rust_test {
            match process_sample(&model, sample, "rust", target_layer, scale, &args) {
                Ok(result) => {
                    if args.verbose {
                        println!("    {}: KL = {:.6}", sample.id, result.kl_divergence);
                    }
                    sample_results.push(result);
                }
                Err(e) => {
                    eprintln!("    {} failed: {e}", sample.id);
                }
            }
        }

        // Compute statistics per language
        let python_kls: Vec<f32> = sample_results
            .iter()
            .filter(|r| r.language == "python")
            .map(|r| r.kl_divergence)
            .collect();

        let rust_kls: Vec<f32> = sample_results
            .iter()
            .filter(|r| r.language == "rust")
            .map(|r| r.kl_divergence)
            .collect();

        let python_stats = compute_stats(&python_kls, "python");
        let rust_stats = compute_stats(&rust_kls, "rust");

        // Welch's t-test
        let (t_stat, p_value) = welch_t_test(&python_kls, &rust_kls);
        let significant = p_value < 0.05;

        // Print summary line
        let ratio = if rust_stats.mean_kl > 1e-10 {
            python_stats.mean_kl / rust_stats.mean_kl
        } else {
            0.0
        };
        println!(
            "  Python KL: {:.6} +/- {:.6}, Rust KL: {:.6} +/- {:.6}, ratio: {:.2}x, p={:.4}{}",
            python_stats.mean_kl,
            python_stats.std_kl,
            rust_stats.mean_kl,
            rust_stats.std_kl,
            ratio,
            p_value,
            if significant { " *" } else { "" }
        );

        dose_response.push(DosePoint {
            scale,
            python_stats,
            rust_stats,
            welch_t_statistic: t_stat,
            welch_p_value: p_value,
            significant_difference: significant,
            sample_results,
        });
    }

    // Print dose-response summary table
    print_dose_response_table(&dose_response);

    // Sanity checks
    print_sanity_checks(&dose_response);

    // Build results
    let results = ExperimentResults {
        model: args.model.clone(),
        architecture: "rwkv6".to_string(),
        intervention_type: "state_steering".to_string(),
        layer: target_layer,
        dose_response,
    };

    // Save JSON
    if let Some(output_path) = &args.output {
        let json = serde_json::to_string_pretty(&results)?;
        fs::write(output_path, json)?;
        println!("\nResults saved to: {}", output_path.display());
    }

    Ok(())
}

/// Process a single sample at a given scale
fn process_sample(
    model: &PlipModel,
    sample: &Sample,
    language: &str,
    layer: usize,
    scale: f32,
    args: &Args,
) -> Result<SampleResult> {
    // Convert character positions to token positions
    let encoding = model.tokenize_with_offsets(&sample.code)?;

    let marker_token_pos = encoding
        .char_to_token(sample.marker_char_pos)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Could not find marker token for char pos {}",
                sample.marker_char_pos
            )
        })?;

    if args.verbose {
        eprintln!(
            "    [DEBUG] marker_char_pos={} -> token_pos={}",
            sample.marker_char_pos, marker_token_pos
        );
    }

    // Build state steering spec
    let spec = StateSteeringSpec::new(scale)
        .position(marker_token_pos)
        .layer(layer);

    // Run steering (baseline + steered in a single call)
    let result = model.forward_with_state_steering(&sample.code, &spec)?;
    let kl = result.kl_divergence()?;

    Ok(SampleResult {
        id: sample.id.clone(),
        language: language.to_string(),
        scale,
        kl_divergence: kl,
        marker_token_pos,
    })
}

/// Print dose-response summary table
fn print_dose_response_table(dose_response: &[DosePoint]) {
    println!("\n============================================================");
    println!("DOSE-RESPONSE TABLE (State Steering)");
    println!("============================================================\n");

    println!(
        "{:<8} | {:<12} {:<12} | {:<12} {:<12} | {:<8} | {:<8} | {}",
        "Scale", "Py Mean", "Py Std", "Rs Mean", "Rs Std", "Ratio", "p-value", "Sig?"
    );
    println!("{}", "-".repeat(96));

    for point in dose_response {
        let ratio = if point.rust_stats.mean_kl > 1e-10 {
            point.python_stats.mean_kl / point.rust_stats.mean_kl
        } else {
            0.0
        };

        println!(
            "{:<8.1} | {:<12.6} {:<12.6} | {:<12.6} {:<12.6} | {:<8.2} | {:<8.4} | {}",
            point.scale,
            point.python_stats.mean_kl,
            point.python_stats.std_kl,
            point.rust_stats.mean_kl,
            point.rust_stats.std_kl,
            ratio,
            point.welch_p_value,
            if point.significant_difference {
                "YES"
            } else {
                "no"
            }
        );
    }
}

/// Print sanity check assessments
fn print_sanity_checks(dose_response: &[DosePoint]) {
    println!("\n=== Sanity Checks ===\n");

    // Check scale=1.0 gives KL ~0
    if let Some(identity) = dose_response.iter().find(|p| (p.scale - 1.0).abs() < 1e-6) {
        let py_kl = identity.python_stats.mean_kl;
        let rs_kl = identity.rust_stats.mean_kl;
        let max_kl = py_kl.max(rs_kl);
        if max_kl < 1e-4 {
            println!(
                "PASS: Scale=1.0 (identity) gives KL near zero (Py={:.2e}, Rs={:.2e})",
                py_kl, rs_kl
            );
        } else {
            println!(
                "WARN: Scale=1.0 should give KL~0 but got Py={:.6}, Rs={:.6}",
                py_kl, rs_kl
            );
        }
    } else {
        println!("SKIP: Scale=1.0 not tested (add 1.0 to --scales for sanity check)");
    }

    // Check monotonicity: does KL generally increase with |scale - 1|?
    let sorted: Vec<&DosePoint> = {
        let mut pts: Vec<&DosePoint> = dose_response.iter().collect();
        pts.sort_by(|a, b| {
            (a.scale - 1.0)
                .abs()
                .partial_cmp(&(b.scale - 1.0).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pts
    };

    if sorted.len() >= 3 {
        let mut monotonic_py = true;
        let mut monotonic_rs = true;
        for i in 1..sorted.len() {
            if sorted[i].python_stats.mean_kl < sorted[i - 1].python_stats.mean_kl * 0.8 {
                monotonic_py = false;
            }
            if sorted[i].rust_stats.mean_kl < sorted[i - 1].rust_stats.mean_kl * 0.8 {
                monotonic_rs = false;
            }
        }
        println!(
            "Monotonicity (Python): {}",
            if monotonic_py {
                "PASS - KL increases with distance from scale=1.0"
            } else {
                "MIXED - KL does not strictly increase (may indicate saturation)"
            }
        );
        println!(
            "Monotonicity (Rust):   {}",
            if monotonic_rs {
                "PASS - KL increases with distance from scale=1.0"
            } else {
                "MIXED - KL does not strictly increase (may indicate saturation)"
            }
        );
    }

    // Check if any scale gives significant Python vs Rust difference
    let n_significant = dose_response
        .iter()
        .filter(|p| p.significant_difference)
        .count();

    println!(
        "\nSignificant P vs R differences: {}/{} scales",
        n_significant,
        dose_response.len()
    );

    println!("\n=== Interpretation ===\n");
    println!("If dose-response shows graded KL: state steering produces controllable effects.");
    println!("If Python/Rust differ: language-specific state flow, consistent with knockout findings.");
    println!("If scale=0.0 matches knockout results: steering generalizes knockout correctly.");
}

/// Compute statistics for a group of KL values
fn compute_stats(kls: &[f32], language: &str) -> GroupStats {
    let n = kls.len();
    if n == 0 {
        return GroupStats {
            language: language.to_string(),
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

    let p_value = 2.0 * (1.0 - t_cdf(t.abs(), df));

    (t, p_value)
}

/// Cumulative distribution function for Student's t distribution (approximation)
fn t_cdf(t: f32, df: f32) -> f32 {
    if df > 30.0 {
        let x = t / (1.0 + t * t / df).sqrt();
        return normal_cdf(x);
    }

    let x = df / (df + t * t);
    let a = df / 2.0;
    let b = 0.5;

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

        let em = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        let d = 1.0 + em * az / bz;
        if d.abs() < 1e-30 {
            continue;
        }
        let az_new = az + em * am;
        let bz_new = bz + em * bm;
        let am_new = az;
        let bm_new = bz;

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
