//! Poetry Layer Scan - Newline→Ending-Word Attention Analysis
//!
//! Measures whether the model attends more to line-ending rhyme words at the
//! newline planning site when rhyming (Category A) vs non-rhyming (Category B)
//! continuations follow. Part of the Melometis experiment (Phase 1b).
//!
//! Metrics per layer:
//!   - Newline→ending attention (head-averaged)
//!   - Newline→non-ending attention (line-3 baseline)
//!   - Ratio (ending / non-ending)
//!   - Per-head breakdown
//!   - Welch's t-test + Cohen's d with 95% CI
//!
//! Usage:
//!   cargo run --release --example poetry_layer_scan -- --model google/gemma-2-2b
//!   cargo run --release --example poetry_layer_scan -- --model google/gemma-2-2b --cpu

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use anyhow::Result;
use candle_core::{DType, IndexOp};
use clap::Parser;
use plip_rs::PlipModel;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "poetry_layer_scan")]
#[command(about = "Scan layers for newline→ending-word attention in poetry corpus")]
struct Args {
    /// Model to analyze
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// Path to poetry corpus JSON file
    #[arg(long, default_value = "corpus/attention_samples_poetry.json")]
    corpus: PathBuf,

    /// Output JSON file for results
    #[arg(long)]
    output: Option<PathBuf>,

    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,

    /// Start layer (default: 0)
    #[arg(long)]
    start_layer: Option<usize>,

    /// End layer (default: n_layers - 1)
    #[arg(long)]
    end_layer: Option<usize>,

    /// Show position conversion details
    #[arg(long)]
    verbose: bool,
}

// ── Corpus deserialization ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct PoetryCorpus {
    #[serde(default, rename = "_format_version")]
    format_version: Option<String>,
    rhyming: Vec<PoetrySample>,
    non_rhyming: Vec<PoetrySample>,
    #[serde(default)]
    generation: Vec<PoetrySample>,
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)]
struct PoetrySample {
    id: String,
    code: String,
    priming_lines: usize,
    marker_char_pos: usize,
    marker_pattern: String,
    target_char_positions: Vec<usize>,
    rhyme_group: String,
    ending_word: String,
    rhyme_word: Option<String>,
    category: String,
    triplet_id: usize,
}

// ── Runtime types ───────────────────────────────────────────────────────────

struct ConvertedPoetrySample {
    #[allow(dead_code)]
    id: String,
    code: String,
    marker_token_pos: Option<usize>,
    ending_token_positions: Vec<usize>,
    non_ending_line3_token_positions: Vec<usize>,
}

/// Per-sample metrics extracted from a single forward pass for one layer
struct SampleMetrics {
    /// ending_attn / non_ending_attn (head-averaged)
    ratio: f64,
    /// Raw ending attention (head-averaged)
    ending_attn: f64,
    /// Raw non-ending attention (head-averaged)
    non_ending_attn: f64,
    /// Per-head: (ending_attn, non_ending_attn) for each head
    per_head: Vec<(f64, f64)>,
}

// ── Output types ────────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
struct HeadResult {
    head: usize,
    ending_attn_mean: f64,
    non_ending_attn_mean: f64,
    ratio: f64,
}

#[derive(Serialize, Clone)]
struct LayerResult {
    layer: usize,
    // Category A (rhyming) stats
    rhyming_ratio_mean: f64,
    rhyming_ratio_std: f64,
    rhyming_n: usize,
    // Category B (non-rhyming) stats
    non_rhyming_ratio_mean: f64,
    non_rhyming_ratio_std: f64,
    non_rhyming_n: usize,
    // Raw attention values (head-averaged, across samples)
    rhyming_ending_attn: f64,
    rhyming_non_ending_attn: f64,
    non_rhyming_ending_attn: f64,
    non_rhyming_non_ending_attn: f64,
    // Statistical tests
    t_statistic: f64,
    df: f64,
    p_value: f64,
    cohens_d: f64,
    cohens_d_ci_lo: f64,
    cohens_d_ci_hi: f64,
    // Per-head breakdown (averaged across all samples in both categories)
    per_head: Vec<HeadResult>,
}

#[derive(Serialize)]
struct ScanResults {
    model: String,
    corpus_format: String,
    n_layers: usize,
    n_heads: usize,
    scanned_layers: Vec<usize>,
    best_layer: usize,
    best_p_value: f64,
    best_cohens_d: f64,
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

// ── Per-layer accumulators ──────────────────────────────────────────────────

struct LayerAccumulators {
    rhyming_ratios: Vec<Vec<f64>>,
    non_rhyming_ratios: Vec<Vec<f64>>,
    rhyming_ending_sums: Vec<(f64, usize)>,
    rhyming_non_ending_sums: Vec<(f64, usize)>,
    non_rhyming_ending_sums: Vec<(f64, usize)>,
    non_rhyming_non_ending_sums: Vec<(f64, usize)>,
    // per_head_sums[layer][head] = (ending_sum, non_ending_sum, count)
    per_head_sums: Vec<Vec<(f64, f64, usize)>>,
}

impl LayerAccumulators {
    fn new(n_layers: usize, n_heads: usize) -> Self {
        Self {
            rhyming_ratios: vec![Vec::new(); n_layers],
            non_rhyming_ratios: vec![Vec::new(); n_layers],
            rhyming_ending_sums: vec![(0.0, 0); n_layers],
            rhyming_non_ending_sums: vec![(0.0, 0); n_layers],
            non_rhyming_ending_sums: vec![(0.0, 0); n_layers],
            non_rhyming_non_ending_sums: vec![(0.0, 0); n_layers],
            per_head_sums: vec![vec![(0.0, 0.0, 0); n_heads]; n_layers],
        }
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Poetry Layer Scan - Newline→Ending-Word Attention Analysis");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Load corpus
    println!("Loading poetry corpus from: {}", args.corpus.display());
    let corpus_json = fs::read_to_string(&args.corpus)?;
    let corpus: PoetryCorpus = serde_json::from_str(&corpus_json)?;

    let format_version = corpus.format_version.as_deref().unwrap_or("2.0");
    println!("  Format version: {format_version}");
    println!("  Rhyming samples (A):     {}", corpus.rhyming.len());
    println!("  Non-rhyming samples (B): {}", corpus.non_rhyming.len());
    println!(
        "  Generation samples (C):  {} (not used)\n",
        corpus.generation.len()
    );

    println!("Loading model: {}...", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    let n_layers = model.n_layers();
    let n_heads = model.n_heads();
    println!("Model loaded: {n_layers} layers, {n_heads} heads\n");

    // Convert character positions to token positions
    println!("Converting character positions to token positions...\n");
    let mut stats = ConversionStats::default();

    let rhyming_samples =
        convert_poetry_samples(&model, &corpus.rhyming, &mut stats, args.verbose)?;
    let non_rhyming_samples =
        convert_poetry_samples(&model, &corpus.non_rhyming, &mut stats, args.verbose)?;

    println!("  Total samples: {}", stats.total_samples);
    println!("  Successful conversions: {}", stats.successful_conversions);
    println!("  Failed conversions: {}", stats.failed_conversions);
    println!("  Fuzzy matches: {}\n", stats.fuzzy_matches);

    // Determine layer range
    let start_layer = args.start_layer.unwrap_or(0);
    let end_layer = args.end_layer.unwrap_or(n_layers - 1);
    let layers_to_scan: Vec<usize> = (start_layer..=end_layer).collect();

    // Single-pass accumulation: process each sample once, extract all layers
    println!(
        "Analyzing {} samples across layers {start_layer}..{end_layer}...\n",
        rhyming_samples.len() + non_rhyming_samples.len()
    );

    let mut acc = LayerAccumulators::new(n_layers, n_heads);

    // Process rhyming samples
    for (i, sample) in rhyming_samples.iter().enumerate() {
        if i % 50 == 0 {
            println!("  Rhyming sample {}/{}", i, rhyming_samples.len());
        }
        process_sample(&model, sample, &layers_to_scan, n_heads, true, &mut acc)?;
    }

    // Process non-rhyming samples
    for (i, sample) in non_rhyming_samples.iter().enumerate() {
        if i % 50 == 0 {
            println!("  Non-rhyming sample {}/{}", i, non_rhyming_samples.len());
        }
        process_sample(&model, sample, &layers_to_scan, n_heads, false, &mut acc)?;
    }

    println!();

    // Compute per-layer statistics and print table
    let mut results = Vec::new();

    println!("┌───────┬────────────┬────────────┬─────────┬──────────┬──────────┬───────────────┐");
    println!("│ Layer │ Rhym Ratio │ NonR Ratio │  Diff   │ t-stat   │ p-value  │ Cohen's d     │");
    println!("├───────┼────────────┼────────────┼─────────┼──────────┼──────────┼───────────────┤");

    for &layer in &layers_to_scan {
        let result = compute_layer_result(layer, n_heads, &acc);

        let significance = if result.p_value < 0.001 {
            "***"
        } else if result.p_value < 0.01 {
            "** "
        } else if result.p_value < 0.05 {
            "*  "
        } else {
            "   "
        };

        let diff = result.rhyming_ratio_mean - result.non_rhyming_ratio_mean;
        println!(
            "│ {:>5} │ {:>9.3} │ {:>9.3} │ {:>+7.3} │ {:>8.3} │ {:>8.5} │ {:>6.3} [{:>5.2},{:>5.2}] {} │",
            layer,
            result.rhyming_ratio_mean,
            result.non_rhyming_ratio_mean,
            diff,
            result.t_statistic,
            result.p_value,
            result.cohens_d,
            result.cohens_d_ci_lo,
            result.cohens_d_ci_hi,
            significance,
        );

        results.push(result);
    }

    println!("└───────┴────────────┴────────────┴─────────┴──────────┴──────────┴───────────────┘");

    // Find best layer (lowest p-value where rhyming ratio > non-rhyming ratio)
    let best = results
        .iter()
        .filter(|r| r.rhyming_ratio_mean > r.non_rhyming_ratio_mean)
        .min_by(|a, b| a.p_value.partial_cmp(&b.p_value).unwrap());

    let (best_layer, best_p, best_d) = if let Some(best) = best {
        println!("\n═══════════════════════════════════════════════════════════════════");
        println!("  Best Layer: {}", best.layer);
        println!("═══════════════════════════════════════════════════════════════════");
        println!(
            "  Rhyming ratio:     {:.4} +/- {:.4}  (n={})",
            best.rhyming_ratio_mean, best.rhyming_ratio_std, best.rhyming_n
        );
        println!(
            "  Non-rhyming ratio: {:.4} +/- {:.4}  (n={})",
            best.non_rhyming_ratio_mean, best.non_rhyming_ratio_std, best.non_rhyming_n
        );
        println!(
            "  Diff: {:.4}",
            best.rhyming_ratio_mean - best.non_rhyming_ratio_mean
        );
        println!(
            "  Raw attn (rhyming):     ending={:.5}, non-ending={:.5}",
            best.rhyming_ending_attn, best.rhyming_non_ending_attn
        );
        println!(
            "  Raw attn (non-rhyming): ending={:.5}, non-ending={:.5}",
            best.non_rhyming_ending_attn, best.non_rhyming_non_ending_attn
        );
        println!("  t-statistic: {:.3}, df: {:.1}", best.t_statistic, best.df);
        println!(
            "  p-value: {:.6} {}",
            best.p_value,
            if best.p_value < 0.01 {
                "SIGNIFICANT (p < 0.01)"
            } else if best.p_value < 0.05 {
                "SIGNIFICANT (p < 0.05)"
            } else {
                "Not significant"
            }
        );
        println!(
            "  Cohen's d: {:.4} [{:.4}, {:.4}]",
            best.cohens_d, best.cohens_d_ci_lo, best.cohens_d_ci_hi
        );

        // Top-3 heads by ratio
        let mut heads_sorted = best.per_head.clone();
        heads_sorted.sort_by(|a, b| b.ratio.partial_cmp(&a.ratio).unwrap());
        println!("\n  Top-3 heads by ending/non-ending ratio:");
        for h in heads_sorted.iter().take(3) {
            println!(
                "    Head {}: ratio={:.3} (ending={:.5}, non-ending={:.5})",
                h.head, h.ratio, h.ending_attn_mean, h.non_ending_attn_mean
            );
        }

        (best.layer, best.p_value, best.cohens_d)
    } else {
        println!("\nNo layer found where rhyming ratio > non-rhyming ratio");
        (0, 1.0, 0.0)
    };

    // Save results
    let scan_results = ScanResults {
        model: args.model.clone(),
        corpus_format: format!("poetry_v{format_version}"),
        n_layers,
        n_heads,
        scanned_layers: layers_to_scan,
        best_layer,
        best_p_value: best_p,
        best_cohens_d: best_d,
        position_conversion_stats: stats,
        results,
    };

    let output_path = args.output.unwrap_or_else(|| {
        let model_name = args.model.replace(['/', '-'], "_");
        PathBuf::from(format!("outputs/poetry_layer_scan_{model_name}.json"))
    });

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&output_path, serde_json::to_string_pretty(&scan_results)?)?;
    println!("\nResults saved to: {}", output_path.display());

    Ok(())
}

// ── Position conversion ─────────────────────────────────────────────────────

/// Find the character index where line 3 starts (after 2nd newline + 1)
fn compute_line3_start_char(code: &str) -> Option<usize> {
    let first_nl = code.find('\n')?;
    let second_nl = code[first_nl + 1..].find('\n')? + first_nl + 1;
    Some(second_nl + 1)
}

fn convert_poetry_samples(
    model: &PlipModel,
    samples: &[PoetrySample],
    stats: &mut ConversionStats,
    verbose: bool,
) -> Result<Vec<ConvertedPoetrySample>> {
    let mut converted = Vec::new();

    for sample in samples {
        stats.total_samples += 1;
        let mut warnings = Vec::new();

        let encoding = model.tokenize_with_offsets(&sample.code)?;

        // Convert marker position (newline between line 3 and 4)
        let mut marker_token_pos = encoding.char_to_token(sample.marker_char_pos);
        if marker_token_pos.is_none() {
            if let Some(fuzzy_pos) = encoding.char_to_token_fuzzy(sample.marker_char_pos) {
                marker_token_pos = Some(fuzzy_pos);
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
                if verbose {
                    println!("  {} FAILED: {}", sample.id, warnings.last().unwrap());
                }
                continue;
            }
        }

        // Convert ending word target positions
        let mut ending_token_positions = Vec::new();
        for &char_pos in &sample.target_char_positions {
            if let Some(token_pos) = encoding.char_to_token(char_pos) {
                ending_token_positions.push(token_pos);
            } else if let Some(fuzzy_pos) = encoding.char_to_token_fuzzy(char_pos) {
                ending_token_positions.push(fuzzy_pos);
                warnings.push(format!(
                    "Target at char {char_pos} fuzzy matched to token {fuzzy_pos}"
                ));
                stats.fuzzy_matches += 1;
            } else {
                warnings.push(format!("Failed to convert target char pos {char_pos}"));
            }
        }
        // Deduplicate (multiple char positions often map to the same token)
        ending_token_positions.sort_unstable();
        ending_token_positions.dedup();

        // Compute line-3 token range
        let line3_start = compute_line3_start_char(&sample.code);
        let non_ending_positions = if let Some(l3_start) = line3_start {
            let line3_tokens = encoding.char_range_to_tokens(l3_start, sample.marker_char_pos);
            let ending_set: HashSet<usize> = ending_token_positions.iter().copied().collect();
            line3_tokens
                .into_iter()
                .filter(|pos| !ending_set.contains(pos))
                .collect()
        } else {
            warnings.push("Could not determine line-3 start".to_string());
            Vec::new()
        };

        if verbose && !warnings.is_empty() {
            println!("  {} warnings:", sample.id);
            for w in &warnings {
                println!("    - {w}");
            }
        }

        if marker_token_pos.is_some()
            && !ending_token_positions.is_empty()
            && !non_ending_positions.is_empty()
        {
            stats.successful_conversions += 1;
        } else {
            stats.failed_conversions += 1;
            if verbose {
                println!(
                    "  {} incomplete: marker={}, ending={}, non_ending={}",
                    sample.id,
                    marker_token_pos.is_some(),
                    ending_token_positions.len(),
                    non_ending_positions.len()
                );
            }
            continue;
        }

        converted.push(ConvertedPoetrySample {
            id: sample.id.clone(),
            code: sample.code.clone(),
            marker_token_pos,
            ending_token_positions,
            non_ending_line3_token_positions: non_ending_positions,
        });
    }

    Ok(converted)
}

// ── Sample processing ───────────────────────────────────────────────────────

/// Extract metrics from a single sample for all layers (one forward pass)
fn extract_sample_metrics(
    analysis: &plip_rs::AttentionAnalysis,
    sample: &ConvertedPoetrySample,
    layer: usize,
    n_heads: usize,
) -> Option<SampleMetrics> {
    let marker_pos = sample.marker_token_pos?;

    // Head-averaged attention from marker to all positions
    let attn = analysis.cache.attention_from_position(layer, marker_pos)?;

    let ending_attn = mean_attention_at(&attn, &sample.ending_token_positions)?;
    let non_ending_attn = mean_attention_at(&attn, &sample.non_ending_line3_token_positions)?;

    if non_ending_attn < 1e-10 {
        return None;
    }

    let ratio = ending_attn / non_ending_attn;

    // Per-head attention
    let pattern = analysis.cache.get_layer(layer)?;
    let pattern_f32 = pattern.to_dtype(DType::F32).ok()?;

    let mut per_head = Vec::with_capacity(n_heads);
    for h in 0..n_heads {
        let head_attn: Vec<f32> = pattern_f32.i((0, h, marker_pos, ..)).ok()?.to_vec1().ok()?;
        let h_ending = mean_attention_at_f32(&head_attn, &sample.ending_token_positions)?;
        let h_non_ending =
            mean_attention_at_f32(&head_attn, &sample.non_ending_line3_token_positions)?;
        per_head.push((h_ending, h_non_ending));
    }

    Some(SampleMetrics {
        ratio,
        ending_attn,
        non_ending_attn,
        per_head,
    })
}

fn mean_attention_at(attn: &[f32], positions: &[usize]) -> Option<f64> {
    if positions.is_empty() {
        return None;
    }
    let mut sum = 0.0_f64;
    let mut count = 0;
    for &pos in positions {
        if pos < attn.len() {
            sum += f64::from(attn[pos]);
            count += 1;
        }
    }
    if count > 0 {
        Some(sum / f64::from(count))
    } else {
        None
    }
}

fn mean_attention_at_f32(attn: &[f32], positions: &[usize]) -> Option<f64> {
    mean_attention_at(attn, positions)
}

/// Process one sample: run forward pass once, accumulate metrics for all layers
fn process_sample(
    model: &PlipModel,
    sample: &ConvertedPoetrySample,
    layers: &[usize],
    n_heads: usize,
    is_rhyming: bool,
    acc: &mut LayerAccumulators,
) -> Result<()> {
    if sample.marker_token_pos.is_none()
        || sample.ending_token_positions.is_empty()
        || sample.non_ending_line3_token_positions.is_empty()
    {
        return Ok(());
    }

    let analysis = model.analyze_attention(&sample.code)?;

    for &layer in layers {
        if let Some(metrics) = extract_sample_metrics(&analysis, sample, layer, n_heads) {
            if is_rhyming {
                acc.rhyming_ratios[layer].push(metrics.ratio);
                acc.rhyming_ending_sums[layer].0 += metrics.ending_attn;
                acc.rhyming_ending_sums[layer].1 += 1;
                acc.rhyming_non_ending_sums[layer].0 += metrics.non_ending_attn;
                acc.rhyming_non_ending_sums[layer].1 += 1;
            } else {
                acc.non_rhyming_ratios[layer].push(metrics.ratio);
                acc.non_rhyming_ending_sums[layer].0 += metrics.ending_attn;
                acc.non_rhyming_ending_sums[layer].1 += 1;
                acc.non_rhyming_non_ending_sums[layer].0 += metrics.non_ending_attn;
                acc.non_rhyming_non_ending_sums[layer].1 += 1;
            }

            // Per-head: accumulate across both categories
            for (h, &(h_end, h_nend)) in metrics.per_head.iter().enumerate() {
                acc.per_head_sums[layer][h].0 += h_end;
                acc.per_head_sums[layer][h].1 += h_nend;
                acc.per_head_sums[layer][h].2 += 1;
            }
        }
    }

    Ok(())
}

// ── Statistics ──────────────────────────────────────────────────────────────

fn mean_std(values: &[f64]) -> (f64, f64, usize) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    (mean, variance.sqrt(), n)
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

fn compute_cohens_d(
    mean1: f64,
    std1: f64,
    n1: usize,
    mean2: f64,
    std2: f64,
    n2: usize,
) -> (f64, f64, f64) {
    let n1_f = n1 as f64;
    let n2_f = n2 as f64;

    if n1 < 2 || n2 < 2 {
        return (0.0, 0.0, 0.0);
    }

    let pooled_var =
        ((n1_f - 1.0) * std1.powi(2) + (n2_f - 1.0) * std2.powi(2)) / (n1_f + n2_f - 2.0);
    let pooled_std = pooled_var.sqrt();

    let d = if pooled_std > 1e-12 {
        (mean1 - mean2) / pooled_std
    } else {
        0.0
    };

    // SE of Cohen's d (Hedges & Olkin, 1985)
    let se_d = ((n1_f + n2_f) / (n1_f * n2_f) + d.powi(2) / (2.0 * (n1_f + n2_f - 2.0))).sqrt();
    let df = n1_f + n2_f - 2.0;

    let (ci_lo, ci_hi) = if df > 0.0 {
        let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
        let t_crit = t_dist.inverse_cdf(0.975);
        (d - t_crit * se_d, d + t_crit * se_d)
    } else {
        (d, d)
    };

    (d, ci_lo, ci_hi)
}

// ── Layer result computation ────────────────────────────────────────────────

fn compute_layer_result(layer: usize, n_heads: usize, acc: &LayerAccumulators) -> LayerResult {
    let (r_mean, r_std, r_n) = mean_std(&acc.rhyming_ratios[layer]);
    let (nr_mean, nr_std, nr_n) = mean_std(&acc.non_rhyming_ratios[layer]);

    let (t_stat, df, p_value) = compute_t_test(r_mean, r_std, r_n, nr_mean, nr_std, nr_n);
    let (cohens_d, ci_lo, ci_hi) = compute_cohens_d(r_mean, r_std, r_n, nr_mean, nr_std, nr_n);

    let r_end = safe_mean(acc.rhyming_ending_sums[layer]);
    let r_nend = safe_mean(acc.rhyming_non_ending_sums[layer]);
    let nr_end = safe_mean(acc.non_rhyming_ending_sums[layer]);
    let nr_nend = safe_mean(acc.non_rhyming_non_ending_sums[layer]);

    let per_head: Vec<HeadResult> = (0..n_heads)
        .map(|h| {
            let (end_sum, nend_sum, count) = acc.per_head_sums[layer][h];
            let end_mean = if count > 0 {
                end_sum / count as f64
            } else {
                0.0
            };
            let nend_mean = if count > 0 {
                nend_sum / count as f64
            } else {
                0.0
            };
            let ratio = if nend_mean > 1e-10 {
                end_mean / nend_mean
            } else {
                0.0
            };
            HeadResult {
                head: h,
                ending_attn_mean: end_mean,
                non_ending_attn_mean: nend_mean,
                ratio,
            }
        })
        .collect();

    LayerResult {
        layer,
        rhyming_ratio_mean: r_mean,
        rhyming_ratio_std: r_std,
        rhyming_n: r_n,
        non_rhyming_ratio_mean: nr_mean,
        non_rhyming_ratio_std: nr_std,
        non_rhyming_n: nr_n,
        rhyming_ending_attn: r_end,
        rhyming_non_ending_attn: r_nend,
        non_rhyming_ending_attn: nr_end,
        non_rhyming_non_ending_attn: nr_nend,
        t_statistic: t_stat,
        df,
        p_value,
        cohens_d,
        cohens_d_ci_lo: ci_lo,
        cohens_d_ci_hi: ci_hi,
        per_head,
    }
}

fn safe_mean(pair: (f64, usize)) -> f64 {
    if pair.1 > 0 {
        pair.0 / pair.1 as f64
    } else {
        0.0
    }
}
