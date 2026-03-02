#![allow(clippy::cast_precision_loss)]
//! Recurrent block rhyme experiment for Llama 3.2 1B.
//!
//! Tests whether re-running a block of transformer layers (with optional
//! unembed feedback) improves rhyme word probability and generation quality.
//!
//! ## Conditions
//!
//! **A. Baseline** — standard generation
//! **B. Double pass (no feedback)** — `RecurrentPassSpec::no_feedback(start, end)`.
//!     Tests whether raw extra depth helps (DRC analog).
//! **C. Unembed feedback** — Feedback vector = averaged unembedding directions of
//!     rhyme-family words, injected at the planning site (last prompt token).
//!     Sweep strengths: 1.0, 2.0, 5.0, 10.0, 20.0.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example recurrent_block_rhyme
//! ```

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use clap::Parser;
use plip_rs::{PlipModel, RecurrentPassSpec};
use serde::Serialize;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Recurrent block rhyme experiment for Llama 3.2 1B")]
struct Args {
    /// `HuggingFace` model ID.
    #[arg(long, default_value = "meta-llama/Llama-3.2-1B")]
    model: String,

    /// Max tokens to generate for line 2.
    #[arg(long, default_value = "30")]
    max_tokens: usize,

    /// Generation temperature (0.0 = greedy).
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Path to write results JSON.
    #[arg(long, default_value = "outputs/recurrent_block_rhyme.json")]
    output: String,

    /// Force CPU mode.
    #[arg(long)]
    cpu: bool,
}

// ---------------------------------------------------------------------------
// Couplet definitions (all 15 from llama_couplet_probes.json)
// ---------------------------------------------------------------------------

struct CoupletDef {
    id: u32,
    target_word: &'static str,
    line1: &'static str,
    rhyme_family: &'static [&'static str],
}

#[allow(clippy::too_many_lines)]
fn couplet_defs() -> Vec<CoupletDef> {
    vec![
        CoupletDef {
            id: 1,
            target_word: "light",
            line1: "The moon casts silver light,",
            rhyme_family: &[
                "light", "night", "bright", "sight", "might", "flight", "right", "tight", "white",
                "bite", "kite", "quite", "knight", "delight", "blight", "plight", "slight",
                "fright", "height",
            ],
        },
        CoupletDef {
            id: 2,
            target_word: "play",
            line1: "The children laugh and play,",
            rhyme_family: &[
                "play", "day", "way", "say", "stay", "sway", "ray", "bay", "may", "lay", "pay",
                "gray", "away", "display", "pray", "stray", "clay", "hay", "decay", "delay",
            ],
        },
        CoupletDef {
            id: 3,
            target_word: "sound",
            line1: "The thunder makes a sound,",
            rhyme_family: &[
                "sound", "ground", "found", "round", "bound", "around", "mound", "pound", "hound",
                "wound", "profound", "abound", "astound",
            ],
        },
        CoupletDef {
            id: 4,
            target_word: "rain",
            line1: "The clouds bring heavy rain,",
            rhyme_family: &[
                "rain", "pain", "gain", "main", "vain", "plain", "chain", "train", "brain",
                "strain", "remain", "again", "drain", "lane", "crane", "bane", "wane", "reign",
                "feign",
            ],
        },
        CoupletDef {
            id: 5,
            target_word: "time",
            line1: "The old clock measures time,",
            rhyme_family: &[
                "time", "rhyme", "climb", "crime", "dime", "lime", "mime", "prime", "chime",
                "sublime", "paradigm", "thyme",
            ],
        },
        CoupletDef {
            id: 6,
            target_word: "air",
            line1: "The geese fly through the air,",
            rhyme_family: &[
                "air", "there", "fair", "care", "bare", "dare", "rare", "share", "stare", "where",
                "pair", "aware", "compare", "despair", "prayer", "hair", "chair", "bear", "wear",
                "spare", "snare", "glare",
            ],
        },
        CoupletDef {
            id: 7,
            target_word: "gold",
            line1: "The sunset gleams like gold,",
            rhyme_family: &[
                "gold",
                "old",
                "bold",
                "cold",
                "fold",
                "hold",
                "told",
                "sold",
                "mold",
                "behold",
                "unfold",
                "rolled",
                "controlled",
            ],
        },
        CoupletDef {
            id: 8,
            target_word: "fire",
            line1: "The embers feed the fire,",
            rhyme_family: &[
                "fire", "hire", "wire", "desire", "tire", "inspire", "acquire", "higher", "entire",
                "admire", "liar", "dire", "sire", "pyre", "mire", "conspire", "expire",
            ],
        },
        CoupletDef {
            id: 9,
            target_word: "stone",
            line1: "The castle walls of stone,",
            rhyme_family: &[
                "stone", "bone", "tone", "lone", "zone", "throne", "phone", "own", "known",
                "blown", "grown", "shown", "moan", "groan", "clone", "drone",
            ],
        },
        CoupletDef {
            id: 10,
            target_word: "dream",
            line1: "I wandered through a dream,",
            rhyme_family: &[
                "dream", "stream", "seem", "team", "beam", "cream", "gleam", "scheme", "theme",
                "extreme", "esteem", "scream", "steam",
            ],
        },
        CoupletDef {
            id: 11,
            target_word: "strange",
            line1: "The silence felt so strange,",
            rhyme_family: &[
                "strange", "change", "range", "arrange", "exchange", "grange",
            ],
        },
        CoupletDef {
            id: 12,
            target_word: "love",
            line1: "I never knew such love,",
            rhyme_family: &["love", "above", "dove", "of", "shove", "glove", "thereof"],
        },
        CoupletDef {
            id: 13,
            target_word: "truth",
            line1: "She spoke the honest truth,",
            rhyme_family: &[
                "truth", "youth", "tooth", "booth", "smooth", "sleuth", "ruth", "uncouth",
            ],
        },
        CoupletDef {
            id: 14,
            target_word: "world",
            line1: "He traveled all the world,",
            rhyme_family: &[
                "world", "curled", "unfurled", "whirled", "hurled", "swirled", "pearled", "furled",
                "twirled",
            ],
        },
        CoupletDef {
            id: 15,
            target_word: "earth",
            line1: "The seeds lay in the earth,",
            rhyme_family: &[
                "earth", "birth", "worth", "mirth", "berth", "girth", "dearth", "rebirth", "hearth",
            ],
        },
    ]
}

// ---------------------------------------------------------------------------
// Experiment condition spec
// ---------------------------------------------------------------------------

struct Condition {
    name: String,
    loop_range: Option<(usize, usize)>,
    feedback_strength: Option<f32>,
    sustained: bool,
}

fn experiment_conditions() -> Vec<Condition> {
    let mut conds = vec![
        // A. Baseline
        Condition {
            name: "baseline".to_string(),
            loop_range: None,
            feedback_strength: None,
            sustained: false,
        },
    ];

    // B + C. Loop ranges to test
    let loop_ranges: &[(usize, usize, &str)] = &[
        (12, 15, "12-15"),
        (10, 15, "10-15"),
        (8, 15, "8-15"),
        (14, 15, "14-15"),
    ];

    for &(start, end, label) in loop_ranges {
        // B. Double pass (no feedback)
        conds.push(Condition {
            name: format!("double_pass_{label}"),
            loop_range: Some((start, end)),
            feedback_strength: None,
            sustained: false,
        });

        // C. Unembed feedback at various strengths (prefill only)
        for strength in &[1.0_f32, 2.0, 5.0, 10.0, 20.0] {
            conds.push(Condition {
                name: format!("unembed_{label}_s={strength:.1}"),
                loop_range: Some((start, end)),
                feedback_strength: Some(*strength),
                sustained: false,
            });
        }
    }

    // D. Sustained feedback during generation (L14-15 only)
    for strength in &[0.5_f32, 1.0, 2.0] {
        conds.push(Condition {
            name: format!("sustained_14-15_s={strength:.1}"),
            loop_range: Some((14, 15)),
            feedback_strength: Some(*strength),
            sustained: true,
        });
    }

    conds
}

// ---------------------------------------------------------------------------
// Output data structures
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ExperimentOutput {
    model: String,
    n_layers: usize,
    n_couplets: usize,
    results: Vec<CoupletResult>,
}

#[derive(Serialize)]
struct CoupletResult {
    id: u32,
    target_word: String,
    rhyme_family: Vec<String>,
    conditions: Vec<ConditionResult>,
}

#[derive(Serialize)]
struct ConditionResult {
    condition: String,
    generated_text: String,
    last_word: String,
    rhymes: bool,
    target_word_prob: f32,
    rhyme_family_prob: f32,
    target_word_rank: usize,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the last word-like token from generated text.
fn extract_last_word(text: &str) -> String {
    text.split_whitespace()
        .next_back()
        .unwrap_or("")
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase()
}

/// Check if a word matches any member of the rhyme family.
fn word_rhymes(word: &str, rhyme_family: &[String]) -> bool {
    let clean = word
        .trim()
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase();
    rhyme_family.contains(&clean)
}

/// Compute L2-normalised average of a set of embedding vectors.
fn averaged_rhyme_direction(model: &PlipModel, rhyme_words: &[&str]) -> Result<Tensor> {
    let mut embeddings: Vec<Tensor> = Vec::new();

    for word in rhyme_words {
        // Tokenize with leading space (subword convention)
        let with_space = format!(" {word}");
        let ids = model.encode(&with_space)?;
        let token_id = *ids.last().context("word produced no tokens")?;
        let emb = model.token_embedding(token_id)?;
        embeddings.push(emb);
    }

    // Stack [n, d_model] and mean across dim 0 => [d_model]
    let stacked = Tensor::stack(&embeddings, 0)?;
    let avg = stacked.mean(0)?;

    // L2-normalise
    let avg_f32 = avg.to_dtype(DType::F32)?;
    let norm = avg_f32.sqr()?.sum_all()?.sqrt()?;
    let norm_val: f32 = norm.to_scalar()?;
    if norm_val > 1e-8 {
        Ok(avg_f32.affine(1.0 / f64::from(norm_val), 0.0)?)
    } else {
        Ok(avg_f32)
    }
}

/// Compute logit-level metrics from a hidden state.
///
/// Accepts either:
/// - 3-D `[batch, seq_len, d_model]` (from `forward_with_recurrent_pass`) — uses last token
/// - 2-D `[1, d_model]` (from `forward_with_layer_skip`) — uses the single vector
///
/// Returns `(target_prob, rhyme_family_prob, target_rank)`.
fn logit_metrics(
    model: &PlipModel,
    hidden: &Tensor,
    target_word: &str,
    rhyme_family: &[String],
) -> Result<(f32, f32, usize)> {
    // Extract last-token hidden as [1, d_model] (project_to_vocab requires 2-D)
    let last_hidden = match hidden.dims().len() {
        3 => {
            let seq_len = hidden.dim(1)?;
            hidden.i((.., seq_len - 1, ..))? // [1, d_model]
        }
        2 => hidden.clone(),       // already [1, d_model]
        1 => hidden.unsqueeze(0)?, // [d_model] → [1, d_model]
        n => anyhow::bail!("logit_metrics: unexpected hidden ndim={n}"),
    };

    let logits = model.project_to_vocab(&last_hidden)?;
    let logits_f32 = logits.to_dtype(DType::F32)?.flatten_all()?;

    // Softmax
    let max_val: f32 = logits_f32.max(0)?.to_scalar()?;
    let shifted = logits_f32.affine(1.0, f64::from(-max_val))?;
    let exps = shifted.exp()?;
    let sum_exp: f32 = exps.sum_all()?.to_scalar()?;
    let probs = exps.affine(1.0 / f64::from(sum_exp), 0.0)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

    // Target word token ID
    let target_with_space = format!(" {target_word}");
    let target_ids = model.encode(&target_with_space)?;
    let target_id = *target_ids
        .last()
        .context("target word produced no tokens")?;

    let target_prob = probs_vec[target_id as usize];

    // Target rank (0-indexed)
    let target_rank = probs_vec.iter().filter(|&&p| p > target_prob).count();

    // Rhyme family total probability
    let mut family_prob = 0.0_f32;
    for word in rhyme_family {
        let word_with_space = format!(" {word}");
        let ids = model.encode(&word_with_space)?;
        let token_id = *ids.last().context("rhyme word produced no tokens")?;
        family_prob += probs_vec[token_id as usize];
    }

    Ok((target_prob, family_prob, target_rank))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();
    println!("=== Recurrent Block Rhyme Experiment — Llama 3.2 1B ===\n");

    // Load model
    println!("Loading {}...", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let n_layers = model.n_layers();
    println!(
        "Model: {} layers, {} hidden, {} vocab",
        n_layers,
        model.d_model(),
        model.vocab_size(),
    );
    println!(
        "Generation: max_tokens={}, temperature={}\n",
        args.max_tokens, args.temperature
    );

    let output_dir = Path::new(&args.output)
        .parent()
        .context("invalid output path")?;
    fs::create_dir_all(output_dir)?;

    let couplets = couplet_defs();
    let conditions = experiment_conditions();
    let stop_tokens: Vec<u32> = model.eos_token_id().into_iter().collect();
    let mut results: Vec<CoupletResult> = Vec::new();

    for couplet in &couplets {
        println!(
            "=== Couplet {:>2}: '{}' (target: {}) ===",
            couplet.id, couplet.line1, couplet.target_word
        );

        let prompt = format!("{}\n", couplet.line1);
        let rhyme_set: Vec<String> = couplet
            .rhyme_family
            .iter()
            .map(|w| (*w).to_lowercase())
            .collect();

        // Compute rhyme direction once per couplet
        let rhyme_direction = averaged_rhyme_direction(&model, couplet.rhyme_family)?;

        // Find planning site = last token position of the prompt
        let prompt_ids = model.encode(&prompt)?;
        let planning_pos = prompt_ids.len() - 1;

        let mut cond_results: Vec<ConditionResult> = Vec::new();

        for cond in &conditions {
            // --- Generation ---
            let generated_text = match cond.loop_range {
                None => {
                    // Baseline: standard generation
                    let std_set = std::collections::HashSet::new();
                    model.generate_with_layer_skip(
                        &prompt,
                        args.max_tokens,
                        args.temperature,
                        &stop_tokens,
                        &std_set,
                    )?
                }
                Some((start, end)) => {
                    let mut spec =
                        RecurrentPassSpec::no_feedback(start, end).with_sustained(cond.sustained);
                    if let Some(strength) = cond.feedback_strength {
                        // Convert rhyme direction to device dtype
                        let vec_on_device = rhyme_direction.to_device(&device)?;
                        spec.add_feedback(planning_pos, vec_on_device, strength);
                    }
                    model.generate_with_recurrent_pass(
                        &prompt,
                        args.max_tokens,
                        args.temperature,
                        &stop_tokens,
                        &spec,
                    )?
                }
            };

            // Extract line 2
            let line2 = generated_text
                .strip_prefix(&prompt)
                .unwrap_or(&generated_text)
                .lines()
                .next()
                .unwrap_or("")
                .to_string();

            let last_word = extract_last_word(&line2);
            let rhymes = word_rhymes(&last_word, &rhyme_set);

            // --- Logit analysis ---
            // Note: logit analysis always uses prefill-only forward (no sustained),
            // because forward_with_recurrent_pass is a single-pass analysis method.
            let (target_prob, family_prob, target_rank) = match cond.loop_range {
                None => {
                    // Baseline: standard single-pass forward (empty skip = all layers)
                    let empty_skip = std::collections::HashSet::new();
                    let hidden = model.forward_with_layer_skip(&prompt, &empty_skip)?;
                    logit_metrics(&model, &hidden, couplet.target_word, &rhyme_set)?
                }
                Some((start, end)) => {
                    let mut spec = RecurrentPassSpec::no_feedback(start, end);
                    if let Some(strength) = cond.feedback_strength {
                        let vec_on_device = rhyme_direction.to_device(&device)?;
                        spec.add_feedback(planning_pos, vec_on_device, strength);
                    }
                    let hidden = model.forward_with_recurrent_pass(&prompt, &spec)?;
                    logit_metrics(&model, &hidden, couplet.target_word, &rhyme_set)?
                }
            };

            let rhyme_marker = if rhymes { "RHYME" } else { "-" };
            println!(
                "  {:<30} [{:<5}] P(tgt)={:.4e} P(fam)={:.4e} rank={:<6} last='{}'",
                cond.name, rhyme_marker, target_prob, family_prob, target_rank, last_word
            );

            cond_results.push(ConditionResult {
                condition: cond.name.clone(),
                generated_text: line2,
                last_word,
                rhymes,
                target_word_prob: target_prob,
                rhyme_family_prob: family_prob,
                target_word_rank: target_rank,
            });
        }

        println!();
        results.push(CoupletResult {
            id: couplet.id,
            target_word: couplet.target_word.to_string(),
            rhyme_family: rhyme_set,
            conditions: cond_results,
        });
    }

    // Build output
    let output = ExperimentOutput {
        model: args.model.clone(),
        n_layers,
        n_couplets: results.len(),
        results,
    };

    let json = serde_json::to_string_pretty(&output)?;
    fs::write(&args.output, &json)?;
    println!("Results written to {}", args.output);

    // Summary table
    println!("\n{:=<100}", "");
    println!("SUMMARY: Rhyme success rate per condition");
    println!("{:=<100}", "");

    for cond in &conditions {
        let count = output
            .results
            .iter()
            .filter(|r| {
                r.conditions
                    .iter()
                    .any(|c| c.condition == cond.name && c.rhymes)
            })
            .count();
        let avg_target_prob: f32 = output
            .results
            .iter()
            .filter_map(|r| {
                r.conditions
                    .iter()
                    .find(|c| c.condition == cond.name)
                    .map(|c| c.target_word_prob)
            })
            .sum::<f32>()
            / output.results.len() as f32;
        let avg_family_prob: f32 = output
            .results
            .iter()
            .filter_map(|r| {
                r.conditions
                    .iter()
                    .find(|c| c.condition == cond.name)
                    .map(|c| c.rhyme_family_prob)
            })
            .sum::<f32>()
            / output.results.len() as f32;

        println!(
            "  {:<35} rhymes={:>2}/{} avg_P(tgt)={:.4e} avg_P(fam)={:.4e}",
            cond.name,
            count,
            output.results.len(),
            avg_target_prob,
            avg_family_prob,
        );
    }

    Ok(())
}
