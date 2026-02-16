//! Cross-Mechanism Steering Evaluation (Melometis Phase 2c)
//!
//! Offline analysis tool that loads CLT and attention steering results from
//! Phases 2a/2b and produces cross-mechanism comparison metrics per §2.3.5.
//!
//! Two modes:
//!   - `validate`: Verify the 120-item prompt set distribution
//!   - `compare`:  Side-by-side CLT vs. attention evaluation with Fisher's exact tests
//!
//! Usage:
//!   cargo run --release --example evaluate_steering -- \
//!       --mode validate --corpus corpus/attention_samples_poetry.json
//!
//!   cargo run --release --example evaluate_steering -- \
//!       --mode compare \
//!       --clt-results outputs/clt_steering_results.json \
//!       --attention-results outputs/attention_steering_results.json \
//!       --output outputs/steering_comparison.json

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unreadable_literal)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "evaluate_steering")]
#[command(about = "Cross-mechanism steering evaluation (Melometis Phase 2c)")]
struct Args {
    /// Mode: validate | compare
    #[arg(long)]
    mode: String,

    /// Path to CLT steering results JSON (from poetry_clt_steering --mode run)
    #[arg(long)]
    clt_results: Option<PathBuf>,

    /// Path to attention steering results JSON (from poetry_attention_steering --mode run)
    #[arg(long)]
    attention_results: Option<PathBuf>,

    /// Path to poetry corpus JSON
    #[arg(long, default_value = "corpus/attention_samples_poetry.json")]
    corpus: PathBuf,

    /// Output file path (JSON report)
    #[arg(long)]
    output: Option<PathBuf>,
}

// ── Corpus deserialization ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct PoetryCorpus {
    #[serde(default, rename = "_format_version")]
    _format_version: Option<String>,
    rhyming: Vec<PoetrySample>,
    #[serde(default)]
    #[allow(dead_code)]
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

// ── Input types (shared with Phase 2a/2b) ───────────────────────────────────

#[derive(Deserialize)]
struct RunOutput {
    model: String,
    #[allow(dead_code)]
    clt_repo: String,
    mechanism: String,
    n_pairs: usize,
    strengths: Vec<f32>,
    n_samples: usize,
    results: Vec<ExperimentRecord>,
    #[allow(dead_code)]
    summary: serde_json::Value,
}

#[derive(Deserialize)]
struct ExperimentRecord {
    #[allow(dead_code)]
    mechanism: String,
    prompt_id: String,
    #[allow(dead_code)]
    prompt_text: String,
    #[allow(dead_code)]
    ending_word: String,
    rhyme_group: String,
    #[allow(dead_code)]
    target_word: String,
    condition: String,
    strength: f32,
    #[allow(dead_code)]
    sample_idx: usize,
    #[allow(dead_code)]
    generated_line: String,
    #[allow(dead_code)]
    generated_ending: String,
    target_hit: bool,
    rhyme_hit: bool,
    #[allow(dead_code)]
    features_used: Vec<String>,
}

// ── Validate output types ───────────────────────────────────────────────────

#[derive(Serialize)]
struct ValidateOutput {
    n_groups: usize,
    expected_groups: usize,
    n_pairs: usize,
    expected_pairs: usize,
    group_distribution: Vec<GroupDistEntry>,
    condition_distribution: CondDistEntry,
    all_checks_pass: bool,
    errors: Vec<String>,
}

#[derive(Serialize)]
struct GroupDistEntry {
    rhyme_group: String,
    n_prompts: usize,
    n_congruent: usize,
    n_incongruent: usize,
    n_total_pairs: usize,
    ok: bool,
}

#[derive(Serialize)]
struct CondDistEntry {
    total_congruent: usize,
    total_incongruent: usize,
    balanced: bool,
}

// ── Compare output types ────────────────────────────────────────────────────

#[derive(Serialize)]
struct CompareOutput {
    clt_model: String,
    attention_model: String,
    n_clt_records: usize,
    n_attention_records: usize,
    side_by_side: Vec<SideBySideEntry>,
    clt_efficiency: EfficiencyResult,
    attention_efficiency: EfficiencyResult,
    by_rhyme_group: Vec<RhymeGroupComparison>,
    fisher_tests: Vec<FisherTestEntry>,
    overall_winner: String,
    summary_text: String,
}

#[derive(Serialize)]
struct SideBySideEntry {
    strength: f32,
    condition: String,
    clt_target_hit_rate: f32,
    clt_rhyme_hit_rate: f32,
    clt_n: usize,
    attention_target_hit_rate: f32,
    attention_rhyme_hit_rate: f32,
    attention_n: usize,
    delta_target_hit_rate: f32,
}

#[derive(Serialize)]
struct EfficiencyResult {
    mechanism: String,
    congruent_threshold_strength: Option<f32>,
    any_threshold_strength: Option<f32>,
    best_target_hit_rate: f32,
    best_strength: f32,
}

#[derive(Serialize)]
struct RhymeGroupComparison {
    rhyme_group: String,
    clt_target_hit_rate: f32,
    clt_n: usize,
    attention_target_hit_rate: f32,
    attention_n: usize,
    fisher_p: Option<f64>,
}

#[derive(Serialize)]
struct FisherTestEntry {
    strength: f32,
    condition: String,
    clt_hits: usize,
    clt_n: usize,
    attention_hits: usize,
    attention_n: usize,
    fisher_p: Option<f64>,
    significant_at_05: bool,
    favors: String,
}

// ── Internal types ──────────────────────────────────────────────────────────

struct SteeringPair {
    #[allow(dead_code)]
    prompt_id: String,
    #[allow(dead_code)]
    prompt_text: String,
    #[allow(dead_code)]
    ending_word: String,
    rhyme_group: String,
    #[allow(dead_code)]
    target_word: String,
    condition: String,
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    match args.mode.as_str() {
        "validate" => mode_validate(&args),
        "compare" => mode_compare(&args),
        other => anyhow::bail!("Unknown mode: {other}. Use validate|compare"),
    }
}

// ── Validate mode ───────────────────────────────────────────────────────────

fn mode_validate(args: &Args) -> Result<()> {
    let corpus = load_corpus(&args.corpus)?;
    let pairs = build_steering_pairs(&corpus);

    eprintln!("Built {} steering pairs from corpus", pairs.len());

    let mut errors = Vec::new();

    // Check total count
    let expected_pairs = 120;
    let expected_groups = 20;
    if pairs.len() != expected_pairs {
        errors.push(format!(
            "Expected {expected_pairs} pairs, got {}",
            pairs.len()
        ));
    }

    // Aggregate by (rhyme_group, condition)
    let mut group_cond: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for pair in &pairs {
        let entry = group_cond.entry(pair.rhyme_group.clone()).or_insert((0, 0));
        if pair.condition == "congruent" {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }

    // Check group count
    if group_cond.len() != expected_groups {
        errors.push(format!(
            "Expected {expected_groups} rhyme groups, got {}",
            group_cond.len()
        ));
    }

    // Build distribution entries
    let mut group_distribution = Vec::new();
    let mut total_cong = 0_usize;
    let mut total_incong = 0_usize;

    println!(
        "\n{:>10}  {:>10}  {:>12}  {:>14}  {:>6}  {:>4}",
        "Group", "Prompts", "Congruent", "Incongruent", "Total", "OK?"
    );
    println!("{}", "-".repeat(64));

    for (group, (cong, incong)) in &group_cond {
        let n_prompts = cong + incong;
        // Each prompt generates 2 pairs (1 cong + 1 incong), so n_prompts/2 unique prompts
        let unique_prompts = n_prompts / 2;
        let ok = *cong == 3 && *incong == 3;

        if !ok {
            errors.push(format!(
                "Group \"{group}\": expected 3 congruent + 3 incongruent, got {cong} + {incong}"
            ));
        }

        println!(
            "{group:>10}  {unique_prompts:>10}  {cong:>12}  {incong:>14}  {n_prompts:>6}  {:>4}",
            if ok { "OK" } else { "FAIL" }
        );

        total_cong += cong;
        total_incong += incong;

        group_distribution.push(GroupDistEntry {
            rhyme_group: group.clone(),
            n_prompts: unique_prompts,
            n_congruent: *cong,
            n_incongruent: *incong,
            n_total_pairs: n_prompts,
            ok,
        });
    }

    let balanced = total_cong == total_incong;
    if !balanced {
        errors.push(format!(
            "Unbalanced conditions: {total_cong} congruent vs {total_incong} incongruent"
        ));
    }

    println!(
        "\nTotal: {total_cong} congruent + {total_incong} incongruent = {} pairs",
        total_cong + total_incong
    );

    let all_pass = errors.is_empty();
    if all_pass {
        println!("\n*** ALL CHECKS PASS ***");
    } else {
        println!("\n*** ERRORS ***");
        for e in &errors {
            println!("  - {e}");
        }
    }

    let output = ValidateOutput {
        n_groups: group_cond.len(),
        expected_groups,
        n_pairs: pairs.len(),
        expected_pairs,
        group_distribution,
        condition_distribution: CondDistEntry {
            total_congruent: total_cong,
            total_incongruent: total_incong,
            balanced,
        },
        all_checks_pass: all_pass,
        errors,
    };

    write_output(&output, args.output.as_deref(), "validate")?;
    Ok(())
}

// ── Compare mode ────────────────────────────────────────────────────────────

fn mode_compare(args: &Args) -> Result<()> {
    let clt_path = args
        .clt_results
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--clt-results is required for compare mode"))?;
    let attn_path = args
        .attention_results
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--attention-results is required for compare mode"))?;

    // Load both result sets
    let clt_json = fs::read_to_string(clt_path)
        .with_context(|| format!("Failed to read CLT results: {}", clt_path.display()))?;
    let clt_run: RunOutput =
        serde_json::from_str(&clt_json).context("Failed to parse CLT results JSON")?;

    let attn_json = fs::read_to_string(attn_path)
        .with_context(|| format!("Failed to read attention results: {}", attn_path.display()))?;
    let attn_run: RunOutput =
        serde_json::from_str(&attn_json).context("Failed to parse attention results JSON")?;

    eprintln!(
        "CLT: {} records ({} pairs, {} strengths, {} samples, mechanism: {})",
        clt_run.results.len(),
        clt_run.n_pairs,
        clt_run.strengths.len(),
        clt_run.n_samples,
        clt_run.mechanism
    );
    eprintln!(
        "Attention: {} records ({} pairs, {} strengths, {} samples, mechanism: {})",
        attn_run.results.len(),
        attn_run.n_pairs,
        attn_run.strengths.len(),
        attn_run.n_samples,
        attn_run.mechanism
    );

    // Determine shared strengths
    let clt_strengths: BTreeSet<String> = clt_run
        .strengths
        .iter()
        .map(|s| format!("{s:.1}"))
        .collect();
    let attn_strengths: BTreeSet<String> = attn_run
        .strengths
        .iter()
        .map(|s| format!("{s:.1}"))
        .collect();

    let clt_only: Vec<_> = clt_strengths.difference(&attn_strengths).collect();
    let attn_only: Vec<_> = attn_strengths.difference(&clt_strengths).collect();
    if !clt_only.is_empty() {
        eprintln!("WARNING: Strengths in CLT only: {clt_only:?}");
    }
    if !attn_only.is_empty() {
        eprintln!("WARNING: Strengths in attention only: {attn_only:?}");
    }

    let shared_strengths: Vec<f32> = clt_strengths
        .intersection(&attn_strengths)
        .filter_map(|s| s.parse::<f32>().ok())
        .collect();

    // Check prompt_id overlap
    let clt_prompts: BTreeSet<String> = clt_run
        .results
        .iter()
        .map(|r| r.prompt_id.clone())
        .collect();
    let attn_prompts: BTreeSet<String> = attn_run
        .results
        .iter()
        .map(|r| r.prompt_id.clone())
        .collect();
    if clt_prompts != attn_prompts {
        let clt_extra = clt_prompts.difference(&attn_prompts).count();
        let attn_extra = attn_prompts.difference(&clt_prompts).count();
        eprintln!(
            "WARNING: Prompt set mismatch — {clt_extra} CLT-only, {attn_extra} attention-only prompt IDs"
        );
    }

    // Compute analyses
    let side_by_side = compute_side_by_side(&clt_run.results, &attn_run.results, &shared_strengths);
    let clt_efficiency = compute_efficiency(&clt_run.results, &shared_strengths, "clt");
    let attn_efficiency = compute_efficiency(&attn_run.results, &shared_strengths, "attention");
    let by_rhyme_group = compute_rhyme_group_comparison(&clt_run.results, &attn_run.results);
    let fisher_tests = compute_fisher_tests(&clt_run.results, &attn_run.results, &shared_strengths);
    let (overall_winner, summary_text) =
        determine_winner(&clt_efficiency, &attn_efficiency, &fisher_tests);

    // Print side-by-side table
    println!(
        "\n{:>8}  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  {:>10}  {:>4}",
        "Strength",
        "Condition",
        "CLT Hit%",
        "CLT Rhy%",
        "Attn Hit%",
        "Attn Rhy%",
        "Delta",
        "Fisher p",
        "Sig?"
    );
    println!("{}", "-".repeat(96));

    for entry in &side_by_side {
        let fisher = fisher_tests
            .iter()
            .find(|f| (f.strength - entry.strength).abs() < 1e-4 && f.condition == entry.condition);
        let (p_str, sig_str) = match fisher {
            Some(f) => match f.fisher_p {
                Some(p) => (
                    format!("{p:.4}"),
                    if f.significant_at_05 { "*" } else { "" },
                ),
                None => ("n/a".to_string(), ""),
            },
            None => ("n/a".to_string(), ""),
        };

        println!(
            "{:>8.1}  {:>12}  {:>9.1}%  {:>9.1}%  {:>9.1}%  {:>9.1}%  {:>+7.1}%  {:>10}  {:>4}",
            entry.strength,
            entry.condition,
            entry.clt_target_hit_rate * 100.0,
            entry.clt_rhyme_hit_rate * 100.0,
            entry.attention_target_hit_rate * 100.0,
            entry.attention_rhyme_hit_rate * 100.0,
            entry.delta_target_hit_rate * 100.0,
            p_str,
            sig_str,
        );
    }

    // Print efficiency comparison
    println!("\n--- Steering Efficiency ---");
    println!(
        "{:>12}  {:>12}  {:>14}  {:>22}  {:>18}",
        "Mechanism", "Best Rate", "Best Strength", "50% Thresh (cong)", "50% Thresh (any)"
    );
    println!("{}", "-".repeat(84));
    print_efficiency(&clt_efficiency);
    print_efficiency(&attn_efficiency);

    // Print per-rhyme-group comparison
    println!(
        "\n{:>10}  {:>10}  {:>8}  {:>10}  {:>8}  {:>10}",
        "Group", "CLT Hit%", "CLT N", "Attn Hit%", "Attn N", "Fisher p"
    );
    println!("{}", "-".repeat(66));

    for rg in &by_rhyme_group {
        let p_str = match rg.fisher_p {
            Some(p) => format!("{p:.4}"),
            None => "n/a".to_string(),
        };
        println!(
            "{:>10}  {:>9.1}%  {:>8}  {:>9.1}%  {:>8}  {:>10}",
            rg.rhyme_group,
            rg.clt_target_hit_rate * 100.0,
            rg.clt_n,
            rg.attention_target_hit_rate * 100.0,
            rg.attention_n,
            p_str,
        );
    }

    // Print overall result
    println!("\n*** Winner: {overall_winner} ***");
    println!("{summary_text}");

    let output = CompareOutput {
        clt_model: clt_run.model.clone(),
        attention_model: attn_run.model.clone(),
        n_clt_records: clt_run.results.len(),
        n_attention_records: attn_run.results.len(),
        side_by_side,
        clt_efficiency,
        attention_efficiency: attn_efficiency,
        by_rhyme_group,
        fisher_tests,
        overall_winner,
        summary_text,
    };

    write_output(&output, args.output.as_deref(), "compare")?;
    Ok(())
}

fn print_efficiency(eff: &EfficiencyResult) {
    let thresh_cong = eff
        .congruent_threshold_strength
        .map_or("None".to_string(), |s| format!("{s:.1}"));
    let thresh_any = eff
        .any_threshold_strength
        .map_or("None".to_string(), |s| format!("{s:.1}"));
    println!(
        "{:>12}  {:>11.1}%  {:>14.1}  {:>22}  {:>18}",
        eff.mechanism,
        eff.best_target_hit_rate * 100.0,
        eff.best_strength,
        thresh_cong,
        thresh_any,
    );
}

// ── Analysis functions ──────────────────────────────────────────────────────

/// Aggregate hit counts for a result set by (strength_key, condition).
fn aggregate_by_strength_condition(
    results: &[ExperimentRecord],
) -> BTreeMap<(String, String), (usize, usize, usize)> {
    let mut map: BTreeMap<(String, String), (usize, usize, usize)> = BTreeMap::new();
    for r in results {
        let key = (format!("{:.1}", r.strength), r.condition.clone());
        let entry = map.entry(key).or_insert((0, 0, 0));
        entry.0 += 1;
        if r.target_hit {
            entry.1 += 1;
        }
        if r.rhyme_hit {
            entry.2 += 1;
        }
    }
    map
}

fn compute_side_by_side(
    clt_results: &[ExperimentRecord],
    attn_results: &[ExperimentRecord],
    strengths: &[f32],
) -> Vec<SideBySideEntry> {
    let clt_agg = aggregate_by_strength_condition(clt_results);
    let attn_agg = aggregate_by_strength_condition(attn_results);

    let conditions = ["congruent", "incongruent"];
    let mut entries = Vec::new();

    for &strength in strengths {
        let s_key = format!("{strength:.1}");
        for condition in &conditions {
            let cond_str = (*condition).to_string();

            let (clt_n, clt_t, clt_r) = clt_agg
                .get(&(s_key.clone(), cond_str.clone()))
                .copied()
                .unwrap_or((0, 0, 0));
            let (attn_n, attn_t, attn_r) = attn_agg
                .get(&(s_key.clone(), cond_str.clone()))
                .copied()
                .unwrap_or((0, 0, 0));

            let clt_target_rate = if clt_n > 0 {
                clt_t as f32 / clt_n as f32
            } else {
                0.0
            };
            let clt_rhyme_rate = if clt_n > 0 {
                clt_r as f32 / clt_n as f32
            } else {
                0.0
            };
            let attn_target_rate = if attn_n > 0 {
                attn_t as f32 / attn_n as f32
            } else {
                0.0
            };
            let attn_rhyme_rate = if attn_n > 0 {
                attn_r as f32 / attn_n as f32
            } else {
                0.0
            };

            entries.push(SideBySideEntry {
                strength,
                condition: cond_str,
                clt_target_hit_rate: clt_target_rate,
                clt_rhyme_hit_rate: clt_rhyme_rate,
                clt_n,
                attention_target_hit_rate: attn_target_rate,
                attention_rhyme_hit_rate: attn_rhyme_rate,
                attention_n: attn_n,
                delta_target_hit_rate: clt_target_rate - attn_target_rate,
            });
        }
    }

    entries
}

fn compute_efficiency(
    results: &[ExperimentRecord],
    strengths: &[f32],
    mechanism: &str,
) -> EfficiencyResult {
    let mut sorted_strengths = strengths.to_vec();
    sorted_strengths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_rate = 0.0_f32;
    let mut best_str = 0.0_f32;
    let mut cong_threshold: Option<f32> = None;
    let mut any_threshold: Option<f32> = None;

    for &s in &sorted_strengths {
        // Congruent only
        let cong: Vec<_> = results
            .iter()
            .filter(|r| (r.strength - s).abs() < 1e-4 && r.condition == "congruent")
            .collect();
        let cong_n = cong.len();
        let cong_hits = cong.iter().filter(|r| r.target_hit).count();
        let cong_rate = if cong_n > 0 {
            cong_hits as f32 / cong_n as f32
        } else {
            0.0
        };

        if cong_threshold.is_none() && cong_rate > 0.50 {
            cong_threshold = Some(s);
        }

        // Any condition
        let all: Vec<_> = results
            .iter()
            .filter(|r| (r.strength - s).abs() < 1e-4)
            .collect();
        let all_n = all.len();
        let all_hits = all.iter().filter(|r| r.target_hit).count();
        let all_rate = if all_n > 0 {
            all_hits as f32 / all_n as f32
        } else {
            0.0
        };

        if any_threshold.is_none() && all_rate > 0.50 {
            any_threshold = Some(s);
        }

        if all_rate > best_rate {
            best_rate = all_rate;
            best_str = s;
        }
    }

    EfficiencyResult {
        mechanism: mechanism.to_string(),
        congruent_threshold_strength: cong_threshold,
        any_threshold_strength: any_threshold,
        best_target_hit_rate: best_rate,
        best_strength: best_str,
    }
}

fn compute_rhyme_group_comparison(
    clt_results: &[ExperimentRecord],
    attn_results: &[ExperimentRecord],
) -> Vec<RhymeGroupComparison> {
    let mut clt_rg: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in clt_results {
        let entry = clt_rg.entry(r.rhyme_group.clone()).or_insert((0, 0));
        entry.0 += 1;
        if r.target_hit {
            entry.1 += 1;
        }
    }

    let mut attn_rg: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in attn_results {
        let entry = attn_rg.entry(r.rhyme_group.clone()).or_insert((0, 0));
        entry.0 += 1;
        if r.target_hit {
            entry.1 += 1;
        }
    }

    let all_groups: BTreeSet<String> = clt_rg.keys().chain(attn_rg.keys()).cloned().collect();

    let mut comparisons = Vec::new();
    for group in all_groups {
        let (clt_n, clt_hits) = clt_rg.get(&group).copied().unwrap_or((0, 0));
        let (attn_n, attn_hits) = attn_rg.get(&group).copied().unwrap_or((0, 0));

        let clt_rate = if clt_n > 0 {
            clt_hits as f32 / clt_n as f32
        } else {
            0.0
        };
        let attn_rate = if attn_n > 0 {
            attn_hits as f32 / attn_n as f32
        } else {
            0.0
        };

        let fisher_p = if clt_n > 0 && attn_n > 0 {
            fisher_exact_two_tailed(clt_hits, clt_n - clt_hits, attn_hits, attn_n - attn_hits)
        } else {
            None
        };

        comparisons.push(RhymeGroupComparison {
            rhyme_group: group,
            clt_target_hit_rate: clt_rate,
            clt_n,
            attention_target_hit_rate: attn_rate,
            attention_n: attn_n,
            fisher_p,
        });
    }

    comparisons
}

fn compute_fisher_tests(
    clt_results: &[ExperimentRecord],
    attn_results: &[ExperimentRecord],
    strengths: &[f32],
) -> Vec<FisherTestEntry> {
    let clt_agg = aggregate_by_strength_condition(clt_results);
    let attn_agg = aggregate_by_strength_condition(attn_results);

    let conditions = ["congruent", "incongruent"];
    let mut tests = Vec::new();

    for &strength in strengths {
        let s_key = format!("{strength:.1}");
        for condition in &conditions {
            let cond_str = (*condition).to_string();
            let key = (s_key.clone(), cond_str.clone());

            let (clt_n, clt_hits, _) = clt_agg.get(&key).copied().unwrap_or((0, 0, 0));
            let (attn_n, attn_hits, _) = attn_agg.get(&key).copied().unwrap_or((0, 0, 0));

            if clt_n == 0 || attn_n == 0 {
                tests.push(FisherTestEntry {
                    strength,
                    condition: cond_str,
                    clt_hits,
                    clt_n,
                    attention_hits: attn_hits,
                    attention_n: attn_n,
                    fisher_p: None,
                    significant_at_05: false,
                    favors: "n/a".to_string(),
                });
                continue;
            }

            let clt_misses = clt_n - clt_hits;
            let attn_misses = attn_n - attn_hits;

            let fisher_p = fisher_exact_two_tailed(clt_hits, clt_misses, attn_hits, attn_misses);

            let clt_rate = clt_hits as f64 / clt_n as f64;
            let attn_rate = attn_hits as f64 / attn_n as f64;
            let favors = if (clt_rate - attn_rate).abs() < 1e-10 {
                "tie".to_string()
            } else if clt_rate > attn_rate {
                "clt".to_string()
            } else {
                "attention".to_string()
            };

            let significant = fisher_p.is_some_and(|p| p < 0.05);

            tests.push(FisherTestEntry {
                strength,
                condition: cond_str,
                clt_hits,
                clt_n,
                attention_hits: attn_hits,
                attention_n: attn_n,
                fisher_p,
                significant_at_05: significant,
                favors,
            });
        }
    }

    tests
}

fn determine_winner(
    clt_eff: &EfficiencyResult,
    attn_eff: &EfficiencyResult,
    fisher_tests: &[FisherTestEntry],
) -> (String, String) {
    let clt_sig_wins = fisher_tests
        .iter()
        .filter(|f| f.significant_at_05 && f.favors == "clt")
        .count();
    let attn_sig_wins = fisher_tests
        .iter()
        .filter(|f| f.significant_at_05 && f.favors == "attention")
        .count();

    let total_sig = clt_sig_wins + attn_sig_wins;

    let winner = if clt_sig_wins > attn_sig_wins {
        "clt"
    } else if attn_sig_wins > clt_sig_wins {
        "attention"
    } else if clt_eff.best_target_hit_rate > attn_eff.best_target_hit_rate {
        "clt (by best rate, no significant difference)"
    } else if attn_eff.best_target_hit_rate > clt_eff.best_target_hit_rate {
        "attention (by best rate, no significant difference)"
    } else {
        "tie"
    };

    let summary = format!(
        "Significant cells: {total_sig}/{} total. \
         CLT wins {clt_sig_wins}, attention wins {attn_sig_wins}. \
         CLT best rate: {:.1}% at strength {:.1}. \
         Attention best rate: {:.1}% at strength {:.1}.",
        fisher_tests.len(),
        clt_eff.best_target_hit_rate * 100.0,
        clt_eff.best_strength,
        attn_eff.best_target_hit_rate * 100.0,
        attn_eff.best_strength,
    );

    (winner.to_string(), summary)
}

// ── Fisher's exact test ─────────────────────────────────────────────────────
// Adapted from examples/steering_generate_n50.rs

/// Fisher's exact test (two-tailed) for 2x2 contingency table.
fn fisher_exact_two_tailed(a_yes: usize, a_no: usize, b_yes: usize, b_no: usize) -> Option<f64> {
    let p1 = fisher_exact_one_tailed(a_yes, a_no, b_yes, b_no)?;
    let p2 = fisher_exact_one_tailed(b_yes, b_no, a_yes, a_no)?;
    Some((p1.min(p2) * 2.0).min(1.0))
}

/// Fisher's exact test (one-tailed) for 2x2 contingency table.
/// Returns p-value for testing if group B rate > group A rate.
fn fisher_exact_one_tailed(a_yes: usize, a_no: usize, b_yes: usize, b_no: usize) -> Option<f64> {
    let n1 = a_yes + a_no;
    let n2 = b_yes + b_no;
    let k = a_yes + b_yes;
    let n = n1 + n2;

    if n == 0 || k == 0 || k == n {
        return None;
    }

    let mut p_value = 0.0;
    let max_possible = std::cmp::min(k, n2);

    for x in b_yes..=max_possible {
        let log_prob = log_hypergeom_pmf(n, k, n2, x);
        p_value += log_prob.exp();
    }

    Some(p_value)
}

/// Log of hypergeometric PMF.
fn log_hypergeom_pmf(n: usize, k: usize, n2: usize, x: usize) -> f64 {
    log_binomial(k, x) + log_binomial(n - k, n2 - x) - log_binomial(n, n2)
}

/// Log of binomial coefficient.
fn log_binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }
    log_factorial(n) - log_factorial(k) - log_factorial(n - k)
}

/// Log factorial using lookup table for small values, Stirling for large.
fn log_factorial(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }

    if n <= 20 {
        let factorials: [f64; 21] = [
            1.0,
            1.0,
            2.0,
            6.0,
            24.0,
            120.0,
            720.0,
            5040.0,
            40320.0,
            362880.0,
            3628800.0,
            39916800.0,
            479001600.0,
            6227020800.0,
            87178291200.0,
            1307674368000.0,
            20922789888000.0,
            355687428096000.0,
            6402373705728000.0,
            121645100408832000.0,
            2432902008176640000.0,
        ];
        return factorials[n].ln();
    }

    let n_f = n as f64;
    n_f * n_f.ln() - n_f + 0.5 * (2.0 * std::f64::consts::PI * n_f).ln()
}

// ── Corpus helpers (shared with Phase 2a/2b) ────────────────────────────────

/// Build 120 steering pairs from the corpus (60 prompts x 2 conditions).
fn build_steering_pairs(corpus: &PoetryCorpus) -> Vec<SteeringPair> {
    // Collect rhyme vocabulary per group from Category A
    let mut group_vocab: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for sample in &corpus.rhyming {
        let entry = group_vocab.entry(sample.rhyme_group.clone()).or_default();
        entry.insert(sample.ending_word.to_lowercase());
        if let Some(ref rw) = sample.rhyme_word {
            entry.insert(rw.to_lowercase());
        }
    }

    let groups: Vec<String> = group_vocab.keys().cloned().collect();
    let n_groups = groups.len();

    // Select first 3 prompts per group from Category C
    let mut pairs = Vec::new();

    for (group_idx, group) in groups.iter().enumerate() {
        let mut group_prompts: Vec<&PoetrySample> = corpus
            .generation
            .iter()
            .filter(|s| &s.rhyme_group == group)
            .collect();
        group_prompts.sort_by_key(|s| s.triplet_id);
        group_prompts.truncate(3);

        let vocab = group_vocab.get(group).cloned().unwrap_or_default();

        // Incongruent group: offset by half
        let incong_group_idx = (group_idx + n_groups / 2) % n_groups;
        let incong_group = &groups[incong_group_idx];
        let incong_vocab = group_vocab.get(incong_group).cloned().unwrap_or_default();

        for prompt in &group_prompts {
            let ending_lower = prompt.ending_word.to_lowercase();

            // Congruent: first word from same group that differs from ending_word
            let congruent_target = vocab
                .iter()
                .find(|w| *w != &ending_lower)
                .cloned()
                .unwrap_or_else(|| ending_lower.clone());

            // Incongruent: first word from different group
            let incongruent_target = incong_vocab
                .iter()
                .next()
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());

            pairs.push(SteeringPair {
                prompt_id: prompt.id.clone(),
                prompt_text: prompt.code.clone(),
                ending_word: prompt.ending_word.clone(),
                rhyme_group: prompt.rhyme_group.clone(),
                target_word: congruent_target,
                condition: "congruent".to_string(),
            });

            pairs.push(SteeringPair {
                prompt_id: prompt.id.clone(),
                prompt_text: prompt.code.clone(),
                ending_word: prompt.ending_word.clone(),
                rhyme_group: prompt.rhyme_group.clone(),
                target_word: incongruent_target,
                condition: "incongruent".to_string(),
            });
        }
    }

    pairs
}

/// Load and parse the poetry corpus.
fn load_corpus(path: &PathBuf) -> Result<PoetryCorpus> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("Failed to read corpus: {}", path.display()))?;
    serde_json::from_str(&text).context("Failed to parse poetry corpus JSON")
}

/// Write JSON output to file or stdout.
fn write_output<T: Serialize>(data: &T, path: Option<&std::path::Path>, mode: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(data)?;
    if let Some(p) = path {
        fs::write(p, &json)
            .with_context(|| format!("Failed to write {mode} output to {}", p.display()))?;
        eprintln!("Wrote {mode} output to {}", p.display());
    } else {
        println!("{json}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_exact_equal_groups() {
        // Two identical groups should give p = 1.0
        let p = fisher_exact_two_tailed(5, 5, 5, 5);
        assert!(p.is_some());
        assert!((p.unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fisher_exact_extreme_difference() {
        // Extreme case: all hits vs all misses
        let p = fisher_exact_two_tailed(10, 0, 0, 10);
        assert!(p.is_some());
        assert!(p.unwrap() < 0.001);
    }

    #[test]
    fn test_fisher_exact_empty() {
        // All zeros should return None
        assert!(fisher_exact_two_tailed(0, 0, 0, 0).is_none());
    }

    #[test]
    fn test_fisher_exact_no_variation() {
        // All successes → k == n, should return None
        assert!(fisher_exact_two_tailed(5, 0, 5, 0).is_none());
    }

    #[test]
    fn test_log_factorial_small() {
        assert!((log_factorial(0) - 0.0).abs() < 1e-10);
        assert!((log_factorial(1) - 0.0).abs() < 1e-10);
        assert!((log_factorial(5) - 120.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_log_binomial_edges() {
        assert!((log_binomial(5, 0) - 0.0).abs() < 1e-10);
        assert!((log_binomial(5, 5) - 0.0).abs() < 1e-10);
        assert!(log_binomial(3, 5) == f64::NEG_INFINITY);
    }
}
