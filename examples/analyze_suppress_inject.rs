//! Analyze `outputs/suppress_inject_sweep.json` (Version D results).
//!
//! Usage:
//!   cargo run --release --example `analyze_suppress_inject` -- \
//!       --input `outputs/suppress_inject_sweep.json`

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

#[derive(Parser)]
#[command(name = "analyze_suppress_inject")]
struct Args {
    #[arg(long)]
    input: PathBuf,
}

#[derive(Deserialize)]
struct SuppressInjectOutput {
    model: String,
    suppress_strength: f32,
    inject_strength: f32,
    n_pairs: usize,
    results: Vec<SuppressInjectResult>,
}

#[derive(Deserialize)]
struct SuppressInjectResult {
    prompt_text: String,
    natural_group: String,
    alt_group: String,
    inject_word: String,
    inject_feature: Feature,
    inject_source_layer: usize,
    #[serde(rename = "n_suppress_features")]
    _n_suppress_features: usize,
    baseline_p_inject: f32,
    max_steered_p_inject: f32,
    max_position: usize,
    #[serde(rename = "last_position_p")]
    _last_position_p: f32,
    positions: Vec<PositionResult>,
}

#[derive(Deserialize)]
struct Feature {
    #[serde(rename = "layer")]
    _layer: usize,
    index: usize,
}

#[derive(Deserialize)]
struct PositionResult {
    position: usize,
    token: String,
    p_inject: f32,
}

fn last_pos(r: &SuppressInjectResult) -> usize {
    r.positions.len().saturating_sub(1)
}

fn ratio(r: &SuppressInjectResult) -> f64 {
    if r.baseline_p_inject > 0.0 {
        f64::from(r.max_steered_p_inject) / f64::from(r.baseline_p_inject)
    } else {
        0.0
    }
}

fn prompt_label(r: &SuppressInjectResult) -> &str {
    if r.natural_group == "-out" {
        // Distinguish the two -out prompts by target word in line 3
        if r.prompt_text.contains("about,") {
            "about"
        } else {
            "shout"
        }
    } else if r.natural_group == "-ow" {
        "so"
    } else {
        "who"
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let text = fs::read_to_string(&args.input)
        .with_context(|| format!("reading {}", args.input.display()))?;
    let data: SuppressInjectOutput = serde_json::from_str(&text)?;

    println!("=== Version D: Suppress + Inject Sweep Analysis ===");
    println!(
        "Model: {}, suppress={}, inject={}, pairs={}",
        data.model, data.suppress_strength, data.inject_strength, data.n_pairs
    );

    // ── 1. Breakdown by prompt ──────────────────────────────────────────
    println!("\n── 1. Breakdown by prompt ──");
    let mut by_prompt: HashMap<String, Vec<&SuppressInjectResult>> = HashMap::new();
    for r in &data.results {
        let key = format!("{} ({})", r.natural_group, prompt_label(r));
        by_prompt.entry(key).or_default().push(r);
    }
    let mut prompt_keys: Vec<&String> = by_prompt.keys().collect();
    prompt_keys.sort();
    for key in &prompt_keys {
        let results = &by_prompt[*key];
        let lp = last_pos(results[0]);
        println!("  {:<16} {} pairs, last_pos={}", key, results.len(), lp);
    }

    // ── 2. Planning-site localization ────────────────────────────────────
    println!("\n── 2. Planning-site localization ──");
    let mut total_at_planning = 0usize;
    let mut total_elsewhere = 0usize;

    for key in &prompt_keys {
        let results = &by_prompt[*key];
        let lp = last_pos(results[0]);
        let at_planning = results.iter().filter(|r| r.max_position == lp).count();
        let elsewhere = results.len() - at_planning;
        total_at_planning += at_planning;
        total_elsewhere += elsewhere;
        println!(
            "  {:<16} planning-site: {}/{} ({:.0}%)  elsewhere: {}",
            key,
            at_planning,
            results.len(),
            100.0 * at_planning as f64 / results.len() as f64,
            elsewhere
        );
    }
    println!(
        "  TOTAL            planning-site: {}/{} ({:.0}%)  elsewhere: {}",
        total_at_planning,
        data.n_pairs,
        100.0 * total_at_planning as f64 / data.n_pairs as f64,
        total_elsewhere
    );

    // ── 3. Top 10 earlier-layer cross-group redirections ────────────────
    println!("\n── 3. Top 10 cross-group redirections (L<25, max at planning site) ──");
    let mut earlier_planning: Vec<&SuppressInjectResult> = data
        .results
        .iter()
        .filter(|r| r.inject_source_layer < 25 && r.max_position == last_pos(r))
        .collect();
    earlier_planning.sort_by(|a, b| {
        b.max_steered_p_inject
            .partial_cmp(&a.max_steered_p_inject)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "  {:>5} {:>6} {:>10} {:>8} {:>4} {:>12} {:>12} {:>6} {:>14}",
        "NatGr", "Prompt", "AltGr", "InjectW", "SrcL", "Baseline", "MaxP", "Pos", "Ratio"
    );
    for r in earlier_planning.iter().take(10) {
        println!(
            "  {:>5} {:>6} {:>10} {:>8} {:>4} {:>12.4e} {:>12.4e} {:>6} {:>14.1}x",
            r.natural_group,
            prompt_label(r),
            r.alt_group,
            r.inject_word,
            r.inject_source_layer,
            r.baseline_p_inject,
            r.max_steered_p_inject,
            r.max_position,
            ratio(r)
        );
    }

    // ── 4. "about" L16 results ──────────────────────────────────────────
    println!("\n── 4. \"about\" L16 results (cross-group redirection star) ──");
    let about_l16: Vec<&SuppressInjectResult> = data
        .results
        .iter()
        .filter(|r| r.inject_word == "about" && r.inject_source_layer == 16)
        .collect();
    println!(
        "  {:>5} {:>6} {:>12} {:>6} {:>14}",
        "NatGr", "Prompt", "MaxP", "Pos", "Ratio"
    );
    for r in &about_l16 {
        let lp = last_pos(r);
        let at_plan = if r.max_position == lp { "✓" } else { "" };
        println!(
            "  {:>5} {:>6} {:>12.4e} {:>6} {:>14.1}x  {}",
            r.natural_group,
            prompt_label(r),
            r.max_steered_p_inject,
            r.max_position,
            ratio(r),
            at_plan,
        );
    }

    // ── 5. "around" L22 results ─────────────────────────────────────────
    println!("\n── 5. \"around\" L22 results ──");
    let around_l22: Vec<&SuppressInjectResult> = data
        .results
        .iter()
        .filter(|r| r.inject_word == "around" && r.inject_source_layer == 22)
        .collect();
    println!(
        "  {:>5} {:>6} {:>12} {:>6} {:>14}",
        "NatGr", "Prompt", "MaxP", "Pos", "Ratio"
    );
    for r in &around_l22 {
        let lp = last_pos(r);
        let at_plan = if r.max_position == lp { "✓" } else { "" };
        println!(
            "  {:>5} {:>6} {:>12.4e} {:>6} {:>14.1}x  {}",
            r.natural_group,
            prompt_label(r),
            r.max_steered_p_inject,
            r.max_position,
            ratio(r),
            at_plan,
        );
    }

    // ── 6. Non-planning-site peaks with ratio > 100x ────────────────────
    println!("\n── 6. Non-planning-site peaks (ratio > 100x) ──");
    let mut non_planning: Vec<&SuppressInjectResult> = data
        .results
        .iter()
        .filter(|r| r.max_position != last_pos(r) && ratio(r) > 100.0)
        .collect();
    non_planning.sort_by(|a, b| {
        ratio(b)
            .partial_cmp(&ratio(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!(
        "  {:>5} {:>6} {:>8} {:>4} {:>6} {:>20} {:>14}",
        "NatGr", "Prompt", "InjectW", "SrcL", "MaxPos", "Token@MaxPos", "Ratio"
    );
    for r in &non_planning {
        let tok_at_max = r
            .positions
            .get(r.max_position)
            .map(|p| p.token.replace('\n', "\\n"))
            .unwrap_or_default();
        println!(
            "  {:>5} {:>6} {:>8} {:>4} {:>6} {:>20} {:>14.1}x",
            r.natural_group,
            prompt_label(r),
            r.inject_word,
            r.inject_source_layer,
            r.max_position,
            tok_at_max,
            ratio(r),
        );
    }

    // ── 7. Per-position data for best "around" L22 pair ─────────────────
    println!("\n── 7. Position sweep for best \"around\" L22 pair ──");
    if let Some(best) = around_l22.iter().max_by(|a, b| {
        a.max_steered_p_inject
            .partial_cmp(&b.max_steered_p_inject)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!(
            "  Prompt: {} ({}) → inject \"around\" L22:{}",
            best.natural_group,
            prompt_label(best),
            best.inject_feature.index
        );
        println!("  Baseline P(\"around\") = {:.4e}", best.baseline_p_inject);
        println!("\n  {:>3}  {:<20}  {:>14}", "Pos", "Token", "P(around)");
        println!("  {:-<3}  {:-<20}  {:-<14}", "", "", "");
        for p in &best.positions {
            let marker = if p.position == best.max_position {
                " <<<< MAX"
            } else {
                ""
            };
            println!(
                "  {:>3}  {:<20}  {:>14.6e}{}",
                p.position,
                p.token.replace('\n', "\\n"),
                p.p_inject,
                marker
            );
        }
    }

    // ── 8. Layer-depth gradient (planning-site pairs only) ──────────────
    println!("\n── 8. Layer-depth gradient (median ratio by source layer, planning-site only) ──");
    let mut by_layer: HashMap<usize, Vec<f64>> = HashMap::new();
    for r in &data.results {
        if r.max_position == last_pos(r) {
            by_layer
                .entry(r.inject_source_layer)
                .or_default()
                .push(ratio(r));
        }
    }
    let mut layers: Vec<usize> = by_layer.keys().copied().collect();
    layers.sort_unstable();
    println!(
        "  {:>4}  {:>5}  {:>14}  {:>14}  {:>14}",
        "SrcL", "Count", "Median ratio", "Min ratio", "Max ratio"
    );
    for layer in &layers {
        let ratios = by_layer.get_mut(layer).unwrap();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = ratios.len();
        let median = if n.is_multiple_of(2) {
            f64::midpoint(ratios[n / 2 - 1], ratios[n / 2])
        } else {
            ratios[n / 2]
        };
        println!(
            "  {:>4}  {:>5}  {:>14.1}x  {:>14.1}x  {:>14.1}x",
            layer,
            n,
            median,
            ratios[0],
            ratios[n - 1]
        );
    }

    Ok(())
}
