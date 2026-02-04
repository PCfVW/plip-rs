//! Statistical Attention Analysis for PLIP-rs
//!
//! Loads corpus from JSON, analyzes attention patterns across multiple samples,
//! computes statistical significance (t-test), and saves detailed results.
//!
//! This implements Phase 2 of the RIGOR_EXPERIMENT.md plan.

use anyhow::{Context, Result};
use clap::Parser;
use plip_rs::PlipModel;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "statistical_attention")]
#[command(about = "Statistical attention analysis with t-test for AIware 2026")]
struct Args {
    /// Path to corpus JSON file
    #[arg(long, default_value = "corpus/attention_samples.json")]
    corpus: PathBuf,

    /// Output JSON file for results
    #[arg(long, default_value = "outputs/statistical_attention.json")]
    output: PathBuf,

    /// Model to use
    #[arg(long, default_value = "Qwen/Qwen2.5-Coder-7B-Instruct")]
    model: String,

    /// Layer to analyze (default: last layer)
    #[arg(long)]
    layer: Option<usize>,

    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,
}

#[derive(Deserialize)]
struct AttentionCorpus {
    python_doctest: Vec<AttentionSample>,
    rust_test: Vec<AttentionSample>,
    python_baseline: Vec<AttentionSample>,
    rust_baseline: Vec<AttentionSample>,
}

#[derive(Deserialize, Clone)]
struct AttentionSample {
    id: String,
    code: String,
    #[serde(alias = "doctest_token_pos")]
    #[serde(alias = "test_attr_token_pos")]
    #[serde(alias = "marker_token_pos")]
    marker_token_pos: usize,
    #[serde(alias = "function_param_positions")]
    #[serde(alias = "function_token_positions")]
    #[serde(alias = "struct_token_positions")]
    target_token_positions: Vec<usize>,
}

#[derive(Serialize)]
struct AttentionStatistics {
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    n: usize,
    samples: Vec<SampleResult>,
}

#[derive(Serialize)]
struct SampleResult {
    id: String,
    attention_to_targets: Vec<f64>,
    mean_attention: f64,
}

#[derive(Serialize)]
struct TTestResult {
    t: f64,
    df: f64,
    p_value: f64,
}

#[derive(Serialize)]
struct StatisticalResults {
    model: String,
    layer: usize,
    python_doctest: AttentionStatistics,
    rust_test: AttentionStatistics,
    python_baseline: AttentionStatistics,
    rust_baseline: AttentionStatistics,
    python_vs_rust: TTestResult,
    python_test_vs_baseline: TTestResult,
    rust_test_vs_baseline: TTestResult,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  PLIP-rs: Statistical Attention Analysis");
    println!("  AIware 2026 - Multi-Sample Mechanistic Study");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Load corpus
    println!("Loading corpus from: {:?}", args.corpus);
    let corpus_json = fs::read_to_string(&args.corpus)
        .with_context(|| format!("Failed to read corpus file: {:?}", args.corpus))?;
    let corpus: AttentionCorpus =
        serde_json::from_str(&corpus_json).context("Failed to parse corpus JSON")?;

    println!("  Python doctest samples: {}", corpus.python_doctest.len());
    println!("  Rust test samples:      {}", corpus.rust_test.len());
    println!("  Python baseline:        {}", corpus.python_baseline.len());
    println!("  Rust baseline:          {}\n", corpus.rust_baseline.len());

    // Load model
    println!("Loading model: {}", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    let n_layers = model.n_layers();
    let layer = args.layer.unwrap_or(n_layers - 1);
    println!("Model loaded: {} layers", n_layers);
    println!("Analyzing layer: {}\n", layer);

    // Analyze Python doctest samples
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Analyzing Python Doctest Samples (>>> marker)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    let python_stats = analyze_samples(&model, &corpus.python_doctest, layer, "Python >>>")?;

    // Analyze Rust test samples
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Analyzing Rust Test Samples (#[ marker)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    let rust_stats = analyze_samples(&model, &corpus.rust_test, layer, "Rust #[")?;

    // Analyze baselines
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Analyzing Python Baseline (non-test context)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    let python_baseline =
        analyze_samples(&model, &corpus.python_baseline, layer, "Python baseline")?;

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Analyzing Rust Baseline (non-test context)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    let rust_baseline = analyze_samples(&model, &corpus.rust_baseline, layer, "Rust baseline")?;

    // Compute t-tests
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Computing Statistical Tests");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let python_vs_rust = compute_t_test(&python_stats, &rust_stats);
    let python_test_vs_baseline = compute_t_test(&python_stats, &python_baseline);
    let rust_test_vs_baseline = compute_t_test(&rust_stats, &rust_baseline);

    // Report results
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  STATISTICAL RESULTS (Layer {})", layer);
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("┌─────────────────────────┬─────────┬─────────┬───────┬─────┐");
    println!("│ Condition               │   Mean  │ Std Dev │ Range │  N  │");
    println!("├─────────────────────────┼─────────┼─────────┼───────┼─────┤");
    println!(
        "│ Python >>> → params     │ {:>6.1}% │ {:>6.1}% │ {:>4.1}% │ {:>3} │",
        python_stats.mean * 100.0,
        python_stats.std_dev * 100.0,
        (python_stats.max - python_stats.min) * 100.0,
        python_stats.n
    );
    println!(
        "│ Rust #[ → fn tokens    │ {:>6.1}% │ {:>6.1}% │ {:>4.1}% │ {:>3} │",
        rust_stats.mean * 100.0,
        rust_stats.std_dev * 100.0,
        (rust_stats.max - rust_stats.min) * 100.0,
        rust_stats.n
    );
    println!(
        "│ Python baseline         │ {:>6.1}% │ {:>6.1}% │ {:>4.1}% │ {:>3} │",
        python_baseline.mean * 100.0,
        python_baseline.std_dev * 100.0,
        (python_baseline.max - python_baseline.min) * 100.0,
        python_baseline.n
    );
    println!(
        "│ Rust baseline           │ {:>6.1}% │ {:>6.1}% │ {:>4.1}% │ {:>3} │",
        rust_baseline.mean * 100.0,
        rust_baseline.std_dev * 100.0,
        (rust_baseline.max - rust_baseline.min) * 100.0,
        rust_baseline.n
    );
    println!("└─────────────────────────┴─────────┴─────────┴───────┴─────┘\n");

    println!("Effect Sizes:");
    let ratio = python_stats.mean / rust_stats.mean.max(0.001);
    println!("  Python vs Rust ratio: {:.2}×", ratio);
    println!(
        "  Difference: {:.1}%",
        (python_stats.mean - rust_stats.mean) * 100.0
    );

    println!("\nStatistical Significance Tests:\n");

    println!("1. Python >>> vs Rust #[test]:");
    print_t_test_result(&python_vs_rust);

    println!("\n2. Python doctest vs Python baseline:");
    print_t_test_result(&python_test_vs_baseline);

    println!("\n3. Rust #[test] vs Rust baseline:");
    print_t_test_result(&rust_test_vs_baseline);

    // Hypothesis validation
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  HYPOTHESIS VALIDATION");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let h1_pass = python_stats.mean > 0.15 && python_vs_rust.p_value < 0.05;
    let h2_pass = rust_stats.mean < 0.07 && python_vs_rust.p_value < 0.05;
    let h3_pass = python_vs_rust.p_value < 0.05;

    println!(
        "H1: Python >>> → params (μ > 15%, significant): {}",
        if h1_pass { "✓ PASS" } else { "✗ FAIL" }
    );
    println!(
        "H2: Rust #[ → tokens (μ < 7%, significant):    {}",
        if h2_pass { "✓ PASS" } else { "✗ FAIL" }
    );
    println!(
        "H3: Difference significant (p < 0.05):         {}",
        if h3_pass { "✓ PASS" } else { "✗ FAIL" }
    );

    // Save detailed results
    let results = StatisticalResults {
        model: args.model.clone(),
        layer,
        python_doctest: python_stats,
        rust_test: rust_stats,
        python_baseline,
        rust_baseline,
        python_vs_rust,
        python_test_vs_baseline,
        rust_test_vs_baseline,
    };

    // Create output directory if needed
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {:?}", parent))?;
    }

    fs::write(&args.output, serde_json::to_string_pretty(&results)?)
        .with_context(|| format!("Failed to write results to: {:?}", args.output))?;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("Results saved to: {:?}", args.output);
    println!("═══════════════════════════════════════════════════════════════════\n");

    Ok(())
}

fn analyze_samples(
    model: &PlipModel,
    samples: &[AttentionSample],
    layer: usize,
    label: &str,
) -> Result<AttentionStatistics> {
    let mut sample_results = Vec::new();

    for sample in samples {
        print!("  Analyzing: {} ... ", sample.id);

        // Get attention analysis
        let analysis = model
            .analyze_attention(&sample.code)
            .with_context(|| format!("Failed to analyze sample: {}", sample.id))?;

        // Get attention from marker position at specified layer
        let attn_from_marker = analysis
            .cache
            .attention_from_position(layer, sample.marker_token_pos)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Could not get attention for layer {} at position {}",
                    layer,
                    sample.marker_token_pos
                )
            })?;

        // Extract attention weights to target tokens
        let attentions: Vec<f64> = sample
            .target_token_positions
            .iter()
            .map(|&pos| {
                if pos < attn_from_marker.len() {
                    attn_from_marker[pos] as f64
                } else {
                    0.0
                }
            })
            .collect();

        let mean_attention = if !attentions.is_empty() {
            attentions.iter().sum::<f64>() / attentions.len() as f64
        } else {
            0.0
        };

        println!("{:.2}%", mean_attention * 100.0);

        sample_results.push(SampleResult {
            id: sample.id.clone(),
            attention_to_targets: attentions,
            mean_attention,
        });
    }

    // Compute statistics
    let means: Vec<f64> = sample_results.iter().map(|r| r.mean_attention).collect();
    let n = means.len();

    if n == 0 {
        return Ok(AttentionStatistics {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            n: 0,
            samples: vec![],
        });
    }

    let mean = means.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        means.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let std_dev = variance.sqrt();
    let min = means.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "\n  {} Statistics: μ={:.2}%, σ={:.2}%",
        label,
        mean * 100.0,
        std_dev * 100.0
    );

    Ok(AttentionStatistics {
        mean,
        std_dev,
        min,
        max,
        n,
        samples: sample_results,
    })
}

fn compute_t_test(stats1: &AttentionStatistics, stats2: &AttentionStatistics) -> TTestResult {
    let n1 = stats1.n as f64;
    let n2 = stats2.n as f64;

    if n1 == 0.0 || n2 == 0.0 {
        return TTestResult {
            t: 0.0,
            df: 0.0,
            p_value: 1.0,
        };
    }

    // Welch's t-test (unequal variances)
    let se1 = stats1.std_dev.powi(2) / n1;
    let se2 = stats2.std_dev.powi(2) / n2;
    let se_diff = (se1 + se2).sqrt();

    let t = if se_diff > 0.0 {
        (stats1.mean - stats2.mean) / se_diff
    } else {
        0.0
    };

    // Welch-Satterthwaite degrees of freedom
    let df = if se1 > 0.0 && se2 > 0.0 {
        let numerator = (se1 + se2).powi(2);
        let denominator = (se1.powi(2) / (n1 - 1.0)) + (se2.powi(2) / (n2 - 1.0));
        numerator / denominator
    } else {
        n1 + n2 - 2.0
    };

    // Compute p-value using t-distribution
    let p_value = if df > 0.0 {
        let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
        2.0 * (1.0 - t_dist.cdf(t.abs()))
    } else {
        1.0
    };

    TTestResult { t, df, p_value }
}

fn print_t_test_result(result: &TTestResult) {
    println!("   t-statistic: {:.3}", result.t);
    println!("   df:          {:.1}", result.df);
    println!("   p-value:     {:.4}", result.p_value);

    if result.p_value < 0.001 {
        println!("   → Highly significant (p < 0.001) ***");
    } else if result.p_value < 0.01 {
        println!("   → Very significant (p < 0.01) **");
    } else if result.p_value < 0.05 {
        println!("   → Significant (p < 0.05) *");
    } else {
        println!("   → Not significant (p ≥ 0.05)");
    }
}
