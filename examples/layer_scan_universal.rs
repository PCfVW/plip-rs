//! Universal Layer Scan - Model-Agnostic Attention Analysis
//!
//! This version uses character positions from the universal corpus format,
//! converting to token positions at runtime. Works with ANY model without
//! preprocessing.
//!
//! Usage:
//!   cargo run --release --example layer_scan_universal -- --model "Qwen/Qwen2.5-Coder-7B-Instruct"
//!   cargo run --release --example layer_scan_universal -- --model "bigcode/starcoder2-3b"
//!   cargo run --release --example layer_scan_universal -- --model "google/codegemma-7b-it"

use anyhow::Result;
use clap::Parser;
use plip_rs::PlipModel;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "layer_scan_universal")]
#[command(about = "Scan layers using universal corpus with character positions")]
struct Args {
    /// Model to analyze
    #[arg(long, default_value = "Qwen/Qwen2.5-Coder-7B-Instruct")]
    model: String,

    /// Path to universal corpus JSON file
    #[arg(long, default_value = "corpus/attention_samples_universal.json")]
    corpus: PathBuf,

    /// Output JSON file for results
    #[arg(long)]
    output: Option<PathBuf>,

    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,

    /// Start layer (default: n_layers / 3)
    #[arg(long)]
    start_layer: Option<usize>,

    /// End layer (default: n_layers - 1)
    #[arg(long)]
    end_layer: Option<usize>,

    /// Show position conversion details
    #[arg(long)]
    verbose: bool,
}

/// Universal corpus format with character positions
#[derive(Deserialize, Clone)]
#[allow(dead_code)]
struct UniversalCorpus {
    #[serde(default)]
    _format_version: Option<String>,
    #[serde(default)]
    _description: Option<String>,
    python_doctest: Vec<UniversalSample>,
    rust_test: Vec<UniversalSample>,
    #[serde(default)]
    python_baseline: Vec<UniversalSample>,
    #[serde(default)]
    rust_baseline: Vec<UniversalSample>,
}

/// Sample with character positions (model-agnostic)
#[derive(Deserialize, Clone)]
#[allow(dead_code)]
struct UniversalSample {
    id: String,
    code: String,
    /// Character position of the marker (byte offset)
    marker_char_pos: usize,
    /// The marker pattern (e.g., ">>>" or "#[test]")
    marker_pattern: String,
    /// Character positions of target tokens
    target_char_positions: Vec<usize>,
}

/// Sample with runtime-converted token positions
#[allow(dead_code)]
struct ConvertedSample {
    id: String,
    code: String,
    marker_token_pos: Option<usize>,
    target_token_positions: Vec<usize>,
    conversion_warnings: Vec<String>,
}

#[derive(Serialize, Clone)]
struct LayerResult {
    layer: usize,
    python_mean: f64,
    python_std: f64,
    python_n: usize,
    rust_mean: f64,
    rust_std: f64,
    rust_n: usize,
    ratio: f64,
    t_statistic: f64,
    df: f64,
    p_value: f64,
}

#[derive(Serialize)]
struct ScanResults {
    model: String,
    corpus_format: String,
    n_layers: usize,
    scanned_layers: Vec<usize>,
    best_layer: usize,
    best_p_value: f64,
    position_conversion_stats: ConversionStats,
    results: Vec<LayerResult>,
}

#[derive(Serialize, Default)]
struct ConversionStats {
    total_samples: usize,
    successful_conversions: usize,
    failed_conversions: usize,
    fuzzy_matches: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Universal Layer Scan - Model-Agnostic Attention Analysis");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Load corpus
    println!("Loading universal corpus from: {:?}", args.corpus);
    let corpus_json = fs::read_to_string(&args.corpus)?;
    let corpus: UniversalCorpus = serde_json::from_str(&corpus_json)?;

    let format_version = corpus._format_version.as_deref().unwrap_or("1.0");
    println!("  Format version: {}", format_version);
    println!("  Python doctest samples: {}", corpus.python_doctest.len());
    println!("  Rust test samples:      {}\n", corpus.rust_test.len());

    println!("Loading model: {}...", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    let n_layers = model.n_layers();
    println!("Model loaded: {} layers\n", n_layers);

    // Convert samples using this model's tokenizer
    println!("Converting character positions to token positions...\n");
    let mut stats = ConversionStats::default();

    let python_samples = convert_samples(&model, &corpus.python_doctest, &mut stats, args.verbose)?;
    let rust_samples = convert_samples(&model, &corpus.rust_test, &mut stats, args.verbose)?;

    println!("  Total samples: {}", stats.total_samples);
    println!("  Successful conversions: {}", stats.successful_conversions);
    println!("  Failed conversions: {}", stats.failed_conversions);
    println!("  Fuzzy matches: {}\n", stats.fuzzy_matches);

    // Determine layer range
    let start_layer = args.start_layer.unwrap_or(n_layers / 3);
    let end_layer = args.end_layer.unwrap_or(n_layers - 1);
    let layers_to_scan: Vec<usize> = (start_layer..=end_layer).collect();

    let mut results = Vec::new();

    println!("Scanning layers {} to {}...\n", start_layer, end_layer);
    println!("┌───────┬────────────┬────────────┬─────────┬──────────┬──────────┬──────────┐");
    println!("│ Layer │ Python μ   │ Rust μ     │  Ratio  │ t-stat   │ df       │ p-value  │");
    println!("├───────┼────────────┼────────────┼─────────┼──────────┼──────────┼──────────┤");

    for layer in &layers_to_scan {
        let (py_mean, py_std, py_n) = analyze_layer(&model, &python_samples, *layer)?;
        let (rust_mean, rust_std, rust_n) = analyze_layer(&model, &rust_samples, *layer)?;

        let ratio = if rust_mean > 0.001 {
            py_mean / rust_mean
        } else {
            0.0
        };
        let (t_stat, df, p_value) =
            compute_t_test(py_mean, py_std, py_n, rust_mean, rust_std, rust_n);

        let significance = if p_value < 0.001 {
            "***"
        } else if p_value < 0.01 {
            "** "
        } else if p_value < 0.05 {
            "*  "
        } else {
            "   "
        };

        println!(
            "│ {:>5} │ {:>9.2}% │ {:>9.2}% │ {:>6.2}× │ {:>7.2} │ {:>7.1} │ {:>7.4} {} │",
            layer,
            py_mean * 100.0,
            rust_mean * 100.0,
            ratio,
            t_stat,
            df,
            p_value,
            significance
        );

        results.push(LayerResult {
            layer: *layer,
            python_mean: py_mean,
            python_std: py_std,
            python_n: py_n,
            rust_mean,
            rust_std,
            rust_n,
            ratio,
            t_statistic: t_stat,
            df,
            p_value,
        });
    }

    println!("└───────┴────────────┴────────────┴─────────┴──────────┴──────────┴──────────┘");

    // Find best layer
    let best = results
        .iter()
        .filter(|r| r.python_mean > r.rust_mean)
        .min_by(|a, b| a.p_value.partial_cmp(&b.p_value).unwrap());

    let (best_layer, best_p) = if let Some(best) = best {
        println!("\n═══════════════════════════════════════════════════════════════════");
        println!("  Best Layer: {}", best.layer);
        println!("═══════════════════════════════════════════════════════════════════");
        println!(
            "  Python >>> → params:  {:.2}% ± {:.2}%  (n={})",
            best.python_mean * 100.0,
            best.python_std * 100.0,
            best.python_n
        );
        println!(
            "  Rust #[ → fn tokens: {:.2}% ± {:.2}%  (n={})",
            best.rust_mean * 100.0,
            best.rust_std * 100.0,
            best.rust_n
        );
        println!("  Ratio: {:.2}×", best.ratio);
        println!("  t-statistic: {:.3}, df: {:.1}", best.t_statistic, best.df);
        println!(
            "  p-value: {:.6} {}",
            best.p_value,
            if best.p_value < 0.05 {
                "✓ SIGNIFICANT"
            } else {
                "Not significant"
            }
        );
        (best.layer, best.p_value)
    } else {
        println!("\n⚠️  No layer found where Python > Rust");
        (0, 1.0)
    };

    // Save results
    let scan_results = ScanResults {
        model: args.model.clone(),
        corpus_format: format!("universal_v{}", format_version),
        n_layers,
        scanned_layers: layers_to_scan,
        best_layer,
        best_p_value: best_p,
        position_conversion_stats: stats,
        results,
    };

    let output_path = args.output.unwrap_or_else(|| {
        let model_name = args.model.replace("/", "_").replace("-", "_");
        PathBuf::from(format!("outputs/layer_scan_universal_{}.json", model_name))
    });

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&output_path, serde_json::to_string_pretty(&scan_results)?)?;
    println!("\nResults saved to: {:?}", output_path);

    Ok(())
}

/// Convert universal samples to token positions for the current model
fn convert_samples(
    model: &PlipModel,
    samples: &[UniversalSample],
    stats: &mut ConversionStats,
    verbose: bool,
) -> Result<Vec<ConvertedSample>> {
    let mut converted = Vec::new();

    for sample in samples {
        stats.total_samples += 1;
        let mut warnings = Vec::new();

        // Get encoding with offsets
        let encoding = model.tokenize_with_offsets(&sample.code)?;

        // Convert marker position
        let marker_token_pos = encoding.char_to_token(sample.marker_char_pos);

        if marker_token_pos.is_none() {
            // Try fuzzy match
            if let Some(fuzzy_pos) = encoding.char_to_token_fuzzy(sample.marker_char_pos) {
                warnings.push(format!(
                    "Marker at char {} fuzzy matched to token {}",
                    sample.marker_char_pos, fuzzy_pos
                ));
                stats.fuzzy_matches += 1;
            } else {
                warnings.push(format!(
                    "Failed to convert marker char pos {}",
                    sample.marker_char_pos
                ));
                stats.failed_conversions += 1;
                continue;
            }
        }

        // Convert target positions
        let mut target_token_positions = Vec::new();
        for &char_pos in &sample.target_char_positions {
            if let Some(token_pos) = encoding.char_to_token(char_pos) {
                target_token_positions.push(token_pos);
            } else if let Some(fuzzy_pos) = encoding.char_to_token_fuzzy(char_pos) {
                target_token_positions.push(fuzzy_pos);
                warnings.push(format!(
                    "Target at char {} fuzzy matched to token {}",
                    char_pos, fuzzy_pos
                ));
                stats.fuzzy_matches += 1;
            } else {
                warnings.push(format!("Failed to convert target char pos {}", char_pos));
            }
        }

        if verbose && !warnings.is_empty() {
            println!("  {} warnings:", sample.id);
            for w in &warnings {
                println!("    - {}", w);
            }
        }

        let final_marker_pos =
            marker_token_pos.or_else(|| encoding.char_to_token_fuzzy(sample.marker_char_pos));

        if final_marker_pos.is_some() && !target_token_positions.is_empty() {
            stats.successful_conversions += 1;
        }

        converted.push(ConvertedSample {
            id: sample.id.clone(),
            code: sample.code.clone(),
            marker_token_pos: final_marker_pos,
            target_token_positions,
            conversion_warnings: warnings,
        });
    }

    Ok(converted)
}

fn analyze_layer(
    model: &PlipModel,
    samples: &[ConvertedSample],
    layer: usize,
) -> Result<(f64, f64, usize)> {
    let mut attentions = Vec::new();

    for sample in samples {
        let marker_pos = match sample.marker_token_pos {
            Some(pos) => pos,
            None => continue,
        };

        if sample.target_token_positions.is_empty() {
            continue;
        }

        let analysis = model.analyze_attention(&sample.code)?;

        let attn_from_marker = match analysis.cache.attention_from_position(layer, marker_pos) {
            Some(attn) => attn,
            None => continue,
        };

        let target_attns: Vec<f64> = sample
            .target_token_positions
            .iter()
            .filter_map(|&pos| {
                if pos < attn_from_marker.len() {
                    Some(attn_from_marker[pos] as f64)
                } else {
                    None
                }
            })
            .collect();

        if !target_attns.is_empty() {
            let mean = target_attns.iter().sum::<f64>() / target_attns.len() as f64;
            attentions.push(mean);
        }
    }

    if attentions.is_empty() {
        return Ok((0.0, 0.0, 0));
    }

    let n = attentions.len();
    let mean = attentions.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        attentions.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    Ok((mean, std_dev, n))
}

fn compute_t_test(
    mean1: f64,
    std1: f64,
    n1: usize,
    mean2: f64,
    std2: f64,
    n2: usize,
) -> (f64, f64, f64) {
    let n1_f = n1 as f64;
    let n2_f = n2 as f64;

    if n1 == 0 || n2 == 0 {
        return (0.0, 0.0, 1.0);
    }

    let se1 = std1.powi(2) / n1_f;
    let se2 = std2.powi(2) / n2_f;
    let se_diff = (se1 + se2).sqrt();

    let t = if se_diff > 0.0 {
        (mean1 - mean2) / se_diff
    } else {
        0.0
    };

    let df = if se1 > 0.0 && se2 > 0.0 {
        let numerator = (se1 + se2).powi(2);
        let denominator = (se1.powi(2) / (n1_f - 1.0)) + (se2.powi(2) / (n2_f - 1.0));
        numerator / denominator
    } else {
        n1_f + n2_f - 2.0
    };

    let p_value = if df > 0.0 {
        let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
        2.0 * (1.0 - t_dist.cdf(t.abs()))
    } else {
        1.0
    };

    (t, df, p_value)
}
