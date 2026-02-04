//! Layer Scan for Optimal Attention Patterns
//!
//! Scans multiple layers to find which shows the strongest effect.
//! Works with any model - detects layer count automatically.
//!
//! Usage:
//!   cargo run --release --example layer_scan -- --model "Qwen/Qwen2.5-Coder-7B-Instruct"
//!   cargo run --release --example layer_scan -- --model "bigcode/starcoder2-3b"

use anyhow::Result;
use clap::Parser;
use plip_rs::PlipModel;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "layer_scan")]
#[command(about = "Scan layers to find optimal attention patterns")]
struct Args {
    /// Model to analyze
    #[arg(long, default_value = "Qwen/Qwen2.5-Coder-7B-Instruct")]
    model: String,

    /// Path to corpus JSON file
    #[arg(long, default_value = "corpus/attention_samples.json")]
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
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)]
struct AttentionCorpus {
    python_doctest: Vec<AttentionSample>,
    rust_test: Vec<AttentionSample>,
    #[serde(default)]
    python_baseline: Vec<AttentionSample>,
    #[serde(default)]
    rust_baseline: Vec<AttentionSample>,
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)]
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
    n_layers: usize,
    scanned_layers: Vec<usize>,
    best_layer: usize,
    best_p_value: f64,
    results: Vec<LayerResult>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Layer Scan for Optimal Attention Patterns");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Load corpus
    println!("Loading corpus from: {:?}", args.corpus);
    let corpus_json = fs::read_to_string(&args.corpus)?;
    let corpus: AttentionCorpus = serde_json::from_str(&corpus_json)?;
    println!("  Python doctest samples: {}", corpus.python_doctest.len());
    println!("  Rust test samples:      {}\n", corpus.rust_test.len());

    println!("Loading model: {}...", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    let n_layers = model.n_layers();
    println!("Model loaded: {} layers\n", n_layers);

    // Determine layer range (default: scan from 1/3 to end)
    let start_layer = args.start_layer.unwrap_or(n_layers / 3);
    let end_layer = args.end_layer.unwrap_or(n_layers - 1);
    let layers_to_scan: Vec<usize> = (start_layer..=end_layer).collect();

    let mut results = Vec::new();

    println!("Scanning layers {} to {}...\n", start_layer, end_layer);
    println!("┌───────┬────────────┬────────────┬─────────┬──────────┬──────────┬──────────┐");
    println!("│ Layer │ Python μ   │ Rust μ     │  Ratio  │ t-stat   │ df       │ p-value  │");
    println!("├───────┼────────────┼────────────┼─────────┼──────────┼──────────┼──────────┤");

    for layer in &layers_to_scan {
        let (py_mean, py_std, py_n) = analyze_layer(&model, &corpus.python_doctest, *layer)?;
        let (rust_mean, rust_std, rust_n) = analyze_layer(&model, &corpus.rust_test, *layer)?;

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

    // Find best layer (lowest p-value with Python > Rust)
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
        n_layers,
        scanned_layers: layers_to_scan,
        best_layer,
        best_p_value: best_p,
        results,
    };

    let output_path = args.output.unwrap_or_else(|| {
        let model_name = args.model.replace("/", "_").replace("-", "_");
        PathBuf::from(format!("outputs/layer_scan_{}.json", model_name))
    });

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&output_path, serde_json::to_string_pretty(&scan_results)?)?;
    println!("\nResults saved to: {:?}", output_path);

    Ok(())
}

fn analyze_layer(
    model: &PlipModel,
    samples: &[AttentionSample],
    layer: usize,
) -> Result<(f64, f64, usize)> {
    let mut attentions = Vec::new();

    for sample in samples {
        let analysis = model.analyze_attention(&sample.code)?;

        let attn_from_marker = match analysis
            .cache
            .attention_from_position(layer, sample.marker_token_pos)
        {
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
