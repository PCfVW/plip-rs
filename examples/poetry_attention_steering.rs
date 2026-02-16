//! Attention Steering for Poetry Generation (Melometis Phase 2b)
//!
//! Comparison condition for CLT feature steering: boosts attention from the
//! newline planning site TO the ending word positions in structurally elevated
//! planning heads (identified in Phase 1b).
//!
//! Two modes:
//!   - `run`:       Main experiment (120 pairs × 7 strengths × 3 samples)
//!   - `evaluate`:  Compute metrics and go/no-go assessment
//!
//! Usage:
//!   cargo run --release --example poetry_attention_steering -- \
//!       --mode run --layer 21 --heads 1,6,7 \
//!       --output outputs/attention_steering_results.json
//!
//!   cargo run --release --example poetry_attention_steering -- \
//!       --mode evaluate --results outputs/attention_steering_results.json

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use plip_rs::{PlipModel, SteeringSpec};
use serde::{Deserialize, Serialize};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "poetry_attention_steering")]
#[command(about = "Attention steering for poetry generation (Melometis Phase 2b)")]
struct Args {
    /// Mode: run | evaluate
    #[arg(long)]
    mode: String,

    /// HuggingFace model ID
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// Path to poetry corpus JSON
    #[arg(long, default_value = "corpus/attention_samples_poetry.json")]
    corpus: PathBuf,

    /// Output file path
    #[arg(long)]
    output: Option<PathBuf>,

    /// Force CPU execution
    #[arg(long)]
    cpu: bool,

    /// Planning layer (default: 21 from Phase 1b analysis)
    #[arg(long, default_value_t = 21)]
    layer: usize,

    /// Comma-separated planning head indices (default: 1,6,7 from Phase 1b)
    #[arg(long, default_value = "1,6,7")]
    heads: String,

    /// Comma-separated steering strengths (Scale factors, 0.0 = baseline)
    #[arg(long, default_value = "0.0,0.5,1.0,2.0,4.0,8.0,16.0")]
    strengths: String,

    /// Samples per condition (run mode)
    #[arg(long, default_value_t = 3)]
    n_seeds: usize,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Max tokens to generate per completion
    #[arg(long, default_value_t = 50)]
    max_tokens: usize,

    /// Path to run output (evaluate mode)
    #[arg(long)]
    results: Option<PathBuf>,
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

// ── Output types (shared format with CLT version) ───────────────────────────

#[derive(Serialize, Deserialize)]
struct RunOutput {
    model: String,
    clt_repo: String,
    mechanism: String,
    n_pairs: usize,
    strengths: Vec<f32>,
    n_samples: usize,
    results: Vec<ExperimentRecord>,
    summary: ExperimentSummary,
}

#[derive(Serialize, Deserialize)]
struct ExperimentRecord {
    mechanism: String,
    prompt_id: String,
    prompt_text: String,
    ending_word: String,
    rhyme_group: String,
    target_word: String,
    condition: String,
    strength: f32,
    sample_idx: usize,
    generated_line: String,
    generated_ending: String,
    target_hit: bool,
    rhyme_hit: bool,
    features_used: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct ExperimentSummary {
    by_strength: Vec<StrengthSummary>,
    by_condition: ConditionSummary,
    by_rhyme_group: Vec<RhymeGroupSummary>,
    go_no_go: String,
}

#[derive(Serialize, Deserialize)]
struct StrengthSummary {
    strength: f32,
    target_hit_rate: f32,
    rhyme_hit_rate: f32,
    n: usize,
}

#[derive(Serialize, Deserialize)]
struct ConditionSummary {
    congruent_target_hit_rate: f32,
    congruent_rhyme_hit_rate: f32,
    congruent_n: usize,
    incongruent_target_hit_rate: f32,
    incongruent_rhyme_hit_rate: f32,
    incongruent_n: usize,
}

#[derive(Serialize, Deserialize)]
struct RhymeGroupSummary {
    rhyme_group: String,
    target_hit_rate: f32,
    rhyme_hit_rate: f32,
    n: usize,
}

// -- evaluate --

#[derive(Serialize, Deserialize)]
struct EvaluateOutput {
    by_strength_condition: Vec<StrengthConditionEntry>,
    by_rhyme_group: Vec<RhymeGroupSummary>,
    overall_best_target_hit_rate: f32,
    overall_best_strength: f32,
    go_no_go: String,
}

#[derive(Serialize, Deserialize)]
struct StrengthConditionEntry {
    strength: f32,
    condition: String,
    target_hit_rate: f32,
    rhyme_hit_rate: f32,
    n: usize,
}

// ── Internal types ──────────────────────────────────────────────────────────

struct SteeringPair {
    prompt_id: String,
    prompt_text: String,
    ending_word: String,
    rhyme_group: String,
    target_word: String,
    condition: String,
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    match args.mode.as_str() {
        "run" => mode_run(&args),
        "evaluate" => mode_evaluate(&args),
        other => anyhow::bail!("Unknown mode: {other}. Use run|evaluate"),
    }
}

// ── Run mode ────────────────────────────────────────────────────────────────

fn mode_run(args: &Args) -> Result<()> {
    let strengths = parse_strengths(&args.strengths)?;
    let heads = parse_heads(&args.heads)?;

    eprintln!(
        "Attention steering: layer {}, heads {:?}",
        args.layer, heads
    );

    // Load model
    eprintln!("Loading model: {}", args.model);
    let model = PlipModel::from_pretrained(&args.model)?;

    // Load corpus and build 120 pairs
    let corpus = load_corpus(&args.corpus)?;
    let pairs = build_steering_pairs(&corpus);
    eprintln!("Built {} steering pairs from corpus", pairs.len());

    // Find stop tokens
    let newline_token = find_newline_token(&model)?;
    let mut stop_tokens = vec![newline_token];
    if let Some(eos) = model.eos_token_id() {
        stop_tokens.push(eos);
    }

    // Features description for output records
    let features_str: Vec<String> = heads
        .iter()
        .map(|&h| format!("L{}:H{}", args.layer, h))
        .collect();

    // Run experiment
    let total_gens = pairs.len() * strengths.len() * args.n_seeds;
    eprintln!(
        "Running {} generations ({} pairs x {} strengths x {} seeds)...",
        total_gens,
        pairs.len(),
        strengths.len(),
        args.n_seeds
    );

    let mut results = Vec::with_capacity(total_gens);
    let mut gen_count = 0;

    for pair in &pairs {
        // Find ending word token positions via string search in prompt
        let ending_positions =
            find_ending_word_token_positions(&model, &pair.prompt_text, &pair.ending_word)?;

        // Newline position = last token of prompt
        let prompt_tokens = model.encode(&pair.prompt_text)?;
        let newline_pos = prompt_tokens.len().saturating_sub(1);

        for &strength in &strengths {
            // Build SteeringSpec (None for baseline, Some for active steering)
            let spec = if strength == 0.0 || ending_positions.is_empty() {
                None
            } else {
                Some(
                    SteeringSpec::scale(strength)
                        .layer(args.layer)
                        .heads(&heads)
                        .from_to_positions(newline_pos, &ending_positions),
                )
            };

            for seed_idx in 0..args.n_seeds {
                gen_count += 1;

                let generated = model.generate_with_steering(
                    &pair.prompt_text,
                    args.max_tokens,
                    args.temperature,
                    &stop_tokens,
                    spec.as_ref(),
                )?;

                let gen_line = extract_generated_line(&generated, &pair.prompt_text);
                let ending = extract_ending_word(&gen_line);
                let target_hit = ending.eq_ignore_ascii_case(&pair.target_word);
                let rhyme_hit = words_rhyme(&ending, &pair.rhyme_group);

                results.push(ExperimentRecord {
                    mechanism: "attention".to_string(),
                    prompt_id: pair.prompt_id.clone(),
                    prompt_text: pair.prompt_text.clone(),
                    ending_word: pair.ending_word.clone(),
                    rhyme_group: pair.rhyme_group.clone(),
                    target_word: pair.target_word.clone(),
                    condition: pair.condition.clone(),
                    strength,
                    sample_idx: seed_idx,
                    generated_line: gen_line,
                    generated_ending: ending,
                    target_hit,
                    rhyme_hit,
                    features_used: features_str.clone(),
                });

                if gen_count % 50 == 0 || gen_count == total_gens {
                    eprint!("\r  [{gen_count}/{total_gens}]");
                }
            }
        }
    }
    eprintln!();

    let summary = compute_summary(&results, &strengths);

    let output = RunOutput {
        model: args.model.clone(),
        clt_repo: String::new(),
        mechanism: "attention".to_string(),
        n_pairs: pairs.len(),
        strengths: strengths.clone(),
        n_samples: args.n_seeds,
        results,
        summary,
    };

    write_output(&output, args.output.as_deref(), "run")?;
    Ok(())
}

// ── Evaluate mode ───────────────────────────────────────────────────────────

fn mode_evaluate(args: &Args) -> Result<()> {
    let results_path = args
        .results
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--results is required for evaluate mode"))?;

    let json = fs::read_to_string(results_path)
        .with_context(|| format!("Failed to read results: {}", results_path.display()))?;
    let run: RunOutput = serde_json::from_str(&json)?;

    eprintln!(
        "Loaded {} results ({} pairs, {} strengths, {} samples, mechanism: {})",
        run.results.len(),
        run.n_pairs,
        run.strengths.len(),
        run.n_samples,
        run.mechanism
    );

    // By strength x condition
    let mut by_sc: BTreeMap<(String, String), (usize, usize, usize)> = BTreeMap::new();
    for r in &run.results {
        let key = (format!("{:.1}", r.strength), r.condition.clone());
        let entry = by_sc.entry(key).or_insert((0, 0, 0));
        entry.0 += 1;
        if r.target_hit {
            entry.1 += 1;
        }
        if r.rhyme_hit {
            entry.2 += 1;
        }
    }

    println!(
        "\n{:>8}  {:>12}  {:>6}  {:>14}  {:>12}",
        "Strength", "Condition", "N", "Target Hit %", "Rhyme Hit %"
    );
    println!("{}", "-".repeat(60));

    let mut strength_condition_entries = Vec::new();
    let mut overall_best_rate = 0.0_f32;
    let mut overall_best_strength = 0.0_f32;

    for ((s, cond), (n, t_hits, r_hits)) in &by_sc {
        let t_rate = *t_hits as f32 / (*n).max(1) as f32;
        let r_rate = *r_hits as f32 / (*n).max(1) as f32;
        println!(
            "{s:>8}  {cond:>12}  {n:>6}  {:>13.1}%  {:>11.1}%",
            t_rate * 100.0,
            r_rate * 100.0
        );
        if t_rate > overall_best_rate {
            overall_best_rate = t_rate;
            overall_best_strength = s.parse().unwrap_or(0.0);
        }
        strength_condition_entries.push(StrengthConditionEntry {
            strength: s.parse().unwrap_or(0.0),
            condition: cond.clone(),
            target_hit_rate: t_rate,
            rhyme_hit_rate: r_rate,
            n: *n,
        });
    }

    // By rhyme group
    let mut by_rg: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    for r in &run.results {
        let entry = by_rg.entry(r.rhyme_group.clone()).or_insert((0, 0, 0));
        entry.0 += 1;
        if r.target_hit {
            entry.1 += 1;
        }
        if r.rhyme_hit {
            entry.2 += 1;
        }
    }

    println!(
        "\n{:>10}  {:>6}  {:>14}  {:>12}",
        "Rhyme Grp", "N", "Target Hit %", "Rhyme Hit %"
    );
    println!("{}", "-".repeat(50));

    let mut rg_summaries = Vec::new();
    for (rg, (n, t_hits, r_hits)) in &by_rg {
        let t_rate = *t_hits as f32 / (*n).max(1) as f32;
        let r_rate = *r_hits as f32 / (*n).max(1) as f32;
        println!(
            "{rg:>10}  {n:>6}  {:>13.1}%  {:>11.1}%",
            t_rate * 100.0,
            r_rate * 100.0
        );
        rg_summaries.push(RhymeGroupSummary {
            rhyme_group: rg.clone(),
            target_hit_rate: t_rate,
            rhyme_hit_rate: r_rate,
            n: *n,
        });
    }

    // Go/no-go
    let go_no_go = if overall_best_rate >= 0.10 {
        format!(
            "PASS: best target_hit_rate = {:.1}% at strength {:.1} (>= 10%)",
            overall_best_rate * 100.0,
            overall_best_strength
        )
    } else {
        format!(
            "FAIL: best target_hit_rate = {:.1}% at strength {:.1} (< 10% — attention steering insufficient)",
            overall_best_rate * 100.0,
            overall_best_strength
        )
    };
    println!("\n*** {go_no_go} ***");

    let output = EvaluateOutput {
        by_strength_condition: strength_condition_entries,
        by_rhyme_group: rg_summaries,
        overall_best_target_hit_rate: overall_best_rate,
        overall_best_strength,
        go_no_go,
    };

    write_output(&output, args.output.as_deref(), "evaluate")?;
    Ok(())
}

// ── Helper functions ────────────────────────────────────────────────────────

/// Find token positions of the ending word in the prompt via string search.
///
/// Uses `rfind()` to locate the last occurrence of the ending word in the
/// prompt text, then determines which token positions correspond to it by
/// comparing tokenizations of the prefix before and through the word.
fn find_ending_word_token_positions(
    model: &PlipModel,
    prompt: &str,
    ending_word: &str,
) -> Result<Vec<usize>> {
    // Find last occurrence of the ending word in the prompt (case-insensitive)
    let prompt_lower = prompt.to_lowercase();
    let word_lower = ending_word.to_lowercase();
    let Some(char_start) = prompt_lower.rfind(&word_lower) else {
        return Ok(Vec::new());
    };
    let char_end = char_start + ending_word.len();

    // Tokenize prefix before the word and prefix through the word
    let prefix_before = &prompt[..char_start];
    let prefix_through = &prompt[..char_end];

    let tokens_before = model.encode(prefix_before)?;
    let tokens_through = model.encode(prefix_through)?;

    // The ending word spans token positions [len(before)..len(through))
    let start_pos = tokens_before.len();
    let end_pos = tokens_through.len();

    if start_pos >= end_pos {
        return Ok(Vec::new());
    }

    Ok((start_pos..end_pos).collect())
}

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

/// Extract the generated continuation after the prompt.
///
/// NOTE: This uses byte-level slicing, which assumes `decode(encode(prompt)) == prompt`.
/// Safe for ASCII poetry text with Gemma 2's SentencePiece tokenizer, but would need
/// token-level separation if used with prompts containing characters that the tokenizer
/// normalizes (e.g., non-ASCII whitespace, combining characters).
fn extract_generated_line(full_text: &str, prompt: &str) -> String {
    if full_text.len() > prompt.len() {
        full_text[prompt.len()..].trim().to_string()
    } else {
        String::new()
    }
}

/// Extract the last word from a line, stripping punctuation.
fn extract_ending_word(line: &str) -> String {
    line.split_whitespace()
        .last()
        .unwrap_or("")
        .trim_matches(|c: char| !c.is_alphabetic())
        .to_lowercase()
}

/// Check if a word rhymes (suffix-based match against known rhyme group suffixes).
fn words_rhyme(word: &str, rhyme_group: &str) -> bool {
    let word_lower = word.to_lowercase();
    word_lower.ends_with(rhyme_group)
}

/// Parse comma-separated strength values.
fn parse_strengths(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|v| {
            v.trim()
                .parse::<f32>()
                .with_context(|| format!("Invalid strength value: \"{v}\""))
        })
        .collect()
}

/// Parse comma-separated head indices.
fn parse_heads(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|v| {
            v.trim()
                .parse::<usize>()
                .with_context(|| format!("Invalid head index: \"{v}\""))
        })
        .collect()
}

/// Find the newline token ID for the model.
fn find_newline_token(model: &PlipModel) -> Result<u32> {
    let tokens = model.encode("\n")?;
    tokens
        .into_iter()
        .last()
        .ok_or_else(|| anyhow::anyhow!("Newline encodes to empty token sequence"))
}

/// Compute summary statistics from experiment results.
fn compute_summary(results: &[ExperimentRecord], strengths: &[f32]) -> ExperimentSummary {
    // By strength
    let by_strength: Vec<StrengthSummary> = strengths
        .iter()
        .map(|&s| {
            let matching: Vec<_> = results
                .iter()
                .filter(|r| (r.strength - s).abs() < 1e-4)
                .collect();
            let n = matching.len();
            let t_hits = matching.iter().filter(|r| r.target_hit).count();
            let r_hits = matching.iter().filter(|r| r.rhyme_hit).count();
            StrengthSummary {
                strength: s,
                target_hit_rate: t_hits as f32 / n.max(1) as f32,
                rhyme_hit_rate: r_hits as f32 / n.max(1) as f32,
                n,
            }
        })
        .collect();

    // By condition
    let cong: Vec<_> = results
        .iter()
        .filter(|r| r.condition == "congruent")
        .collect();
    let incong: Vec<_> = results
        .iter()
        .filter(|r| r.condition == "incongruent")
        .collect();
    let by_condition = ConditionSummary {
        congruent_target_hit_rate: cong.iter().filter(|r| r.target_hit).count() as f32
            / cong.len().max(1) as f32,
        congruent_rhyme_hit_rate: cong.iter().filter(|r| r.rhyme_hit).count() as f32
            / cong.len().max(1) as f32,
        congruent_n: cong.len(),
        incongruent_target_hit_rate: incong.iter().filter(|r| r.target_hit).count() as f32
            / incong.len().max(1) as f32,
        incongruent_rhyme_hit_rate: incong.iter().filter(|r| r.rhyme_hit).count() as f32
            / incong.len().max(1) as f32,
        incongruent_n: incong.len(),
    };

    // By rhyme group
    let mut rg_map: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    for r in results {
        let entry = rg_map.entry(r.rhyme_group.clone()).or_insert((0, 0, 0));
        entry.0 += 1;
        if r.target_hit {
            entry.1 += 1;
        }
        if r.rhyme_hit {
            entry.2 += 1;
        }
    }
    let by_rhyme_group: Vec<RhymeGroupSummary> = rg_map
        .into_iter()
        .map(|(rg, (n, t, r))| RhymeGroupSummary {
            rhyme_group: rg,
            target_hit_rate: t as f32 / n.max(1) as f32,
            rhyme_hit_rate: r as f32 / n.max(1) as f32,
            n,
        })
        .collect();

    // Go/no-go
    let best_rate = by_strength
        .iter()
        .map(|s| s.target_hit_rate)
        .fold(0.0_f32, f32::max);
    let best_str = by_strength
        .iter()
        .max_by(|a, b| {
            a.target_hit_rate
                .partial_cmp(&b.target_hit_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(0.0, |s| s.strength);

    let go_no_go = if best_rate >= 0.10 {
        format!(
            "PASS: target_hit_rate = {:.1}% at strength {:.1} (>= 10%)",
            best_rate * 100.0,
            best_str
        )
    } else {
        format!(
            "FAIL: best target_hit_rate = {:.1}% (< 10% — attention steering insufficient)",
            best_rate * 100.0
        )
    };

    ExperimentSummary {
        by_strength,
        by_condition,
        by_rhyme_group,
        go_no_go,
    }
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
