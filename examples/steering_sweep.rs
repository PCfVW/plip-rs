//! Version C of Anthropic Figure 13 replication: multi-layer causal steering sweep.
//!
//! For each token position in a prompt, injects a rhyme-planning CLT feature
//! at that position across ALL downstream layers (source layer L through 25)
//! and measures P(target word) in the output logits.
//!
//! Unlike Version B (which only injected at layer 25 — trivially position-specific
//! because the logit projection reads only the last position), Version C injects
//! at earlier layers where subsequent attention can propagate the signal.
//!
//! Usage:
//!   cargo run --release --example steering_sweep -- \
//!       --rhyme-pairs outputs/rhyme_pairs_all_layers.json \
//!       --output outputs/steering_sweep_multilayer.json

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp};
use clap::Parser;
use plip_rs::{CltFeatureId, CrossLayerTranscoder, PlipModel};
use serde::{Deserialize, Serialize};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "steering_sweep")]
#[command(about = "Figure 13 Version C: multi-layer causal steering sweep (Melometis)")]
struct Args {
    /// HuggingFace model ID
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// HuggingFace CLT repository
    #[arg(long, default_value = "mntss/clt-gemma-2-2b-426k")]
    clt_repo: String,

    /// Path to rhyme-pairs JSON (from find-rhyme-pairs mode)
    #[arg(long)]
    rhyme_pairs: PathBuf,

    /// Output file path
    #[arg(long)]
    output: Option<PathBuf>,

    /// Injection strength
    #[arg(long, default_value_t = 10.0)]
    strength: f32,

    /// Force CPU execution
    #[arg(long)]
    cpu: bool,
}

// ── Rhyme-pairs input types ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct RhymePairsOutput {
    rhyme_groups: Vec<RhymeGroup>,
}

#[derive(Deserialize)]
struct RhymeGroup {
    rhyme_ending: String,
    words: Vec<RhymeWord>,
}

#[derive(Deserialize)]
struct RhymeWord {
    word: String,
    feature: CltFeatureId,
}

// ── Output types ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct SteeringSweepOutput {
    model: String,
    clt_repo: String,
    strength: f32,
    n_layers: usize,
    n_candidates: usize,
    results: Vec<SteeringSweepResult>,
}

#[derive(Serialize)]
struct SteeringSweepResult {
    prompt_text: String,
    group: String,
    target_word: String,
    feature_word: String,
    target_feature: CltFeatureId,
    source_layer: usize,
    n_injection_layers: usize,
    target_token_id: u32,
    baseline_prob: f32,
    max_steered_prob: f32,
    max_steered_position: usize,
    last_position_prob: f32,
    positions: Vec<PositionProb>,
}

#[derive(Serialize)]
struct PositionProb {
    position: usize,
    token: String,
    steered_prob: f32,
}

// ── Prompts (same 4 candidates as Version A/B) ─────────────────────────────

struct CompletionPrompt {
    group: &'static str,
    rhyme_ending: &'static str,
    target_word: &'static str,
    text: &'static str,
}

fn make_sweep_prompts() -> Vec<CompletionPrompt> {
    vec![
        CompletionPrompt {
            group: "-ow",
            rhyme_ending: "OW1",
            target_word: "so",
            text: "A sailor sailed across the bay,\n\
                   And dreamed of home throughout the day.\n\
                   The world keeps spinning even so,\n\
                   There is so much we do not",
        },
        CompletionPrompt {
            group: "-out",
            rhyme_ending: "AW1 T",
            target_word: "about",
            text: "The stars were twinkling in the night,\n\
                   The lanterns cast a golden light.\n\
                   She wandered in the dark about,\n\
                   And found a hidden passage",
        },
        CompletionPrompt {
            group: "-out",
            rhyme_ending: "AW1 T",
            target_word: "shout",
            text: "A sailor sailed across the bay,\n\
                   And dreamed of home throughout the day.\n\
                   He raised his voice and gave a shout,\n\
                   The truth was struggling to come",
        },
        CompletionPrompt {
            group: "-oo",
            rhyme_ending: "UW1",
            target_word: "who",
            text: "The sun goes up, the sun goes down,\n\
                   The moon shines bright above the town.\n\
                   Nobody knows or remembers who,\n\
                   Would come to find a way back",
        },
    ]
}

// ── Sweep candidates ────────────────────────────────────────────────────────

struct SweepCandidate {
    group: String,
    target_word: String,
    feature_word: String,
    feature: CltFeatureId,
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    eprintln!("=== Steering Sweep: Figure 13 Version C (multi-layer) ===\n");

    // 1. Load rhyme pairs
    let rp_text = fs::read_to_string(&args.rhyme_pairs)
        .with_context(|| format!("reading rhyme pairs from {}", args.rhyme_pairs.display()))?;
    let rp: RhymePairsOutput = serde_json::from_str(&rp_text)?;

    // Build rhyme_ending → Vec<(word, feature)> lookup
    let mut ending_to_features: HashMap<String, Vec<(String, CltFeatureId)>> = HashMap::new();
    for g in &rp.rhyme_groups {
        let features: Vec<(String, CltFeatureId)> = g
            .words
            .iter()
            .map(|w| (w.word.clone(), w.feature))
            .collect();
        ending_to_features.insert(g.rhyme_ending.clone(), features);
    }

    // 2. Build candidates: for each prompt, include ALL features from its rhyme group
    let prompts = make_sweep_prompts();
    let mut candidates: Vec<(usize, SweepCandidate)> = Vec::new(); // (prompt_idx, candidate)

    for (prompt_idx, prompt) in prompts.iter().enumerate() {
        let features = ending_to_features.get(prompt.rhyme_ending).ok_or_else(|| {
            anyhow::anyhow!(
                "Rhyme ending \"{}\" not found in rhyme pairs",
                prompt.rhyme_ending
            )
        })?;

        for (word, feature) in features {
            candidates.push((
                prompt_idx,
                SweepCandidate {
                    group: prompt.group.to_string(),
                    target_word: prompt.target_word.to_string(),
                    feature_word: word.clone(),
                    feature: *feature,
                },
            ));
        }
    }

    // Sort by source layer (earliest first — most interesting)
    candidates.sort_by_key(|(_, c)| c.feature.layer);

    eprintln!("Candidates ({} total):", candidates.len());
    for (pi, c) in &candidates {
        eprintln!(
            "  [{:5}] prompt {} \"{}\" → inject \"{}\" L{}:{}",
            c.group, pi, c.target_word, c.feature_word, c.feature.layer, c.feature.index
        );
    }

    // 3. Load model
    eprintln!("\nLoading model: {}", args.model);
    let force_cpu = if args.cpu { Some(true) } else { None };
    let model = PlipModel::from_pretrained_with_device(&args.model, force_cpu)?;

    // 4. Open CLT and cache decoder vectors for ALL downstream layers
    eprintln!("Opening CLT: {}", args.clt_repo);
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let n_layers = clt.config().n_layers;
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    };

    let unique_features: Vec<CltFeatureId> = candidates
        .iter()
        .map(|(_, c)| c.feature)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    eprintln!(
        "Caching decoder vectors for {} unique features (all downstream layers)...",
        unique_features.len()
    );
    clt.cache_steering_vectors_all_downstream(&unique_features, &device)?;

    // 5. Run steering sweep for each candidate
    let mut results: Vec<SteeringSweepResult> = Vec::new();
    let total = candidates.len();

    for (sweep_idx, (prompt_idx, candidate)) in candidates.iter().enumerate() {
        let prompt = &prompts[*prompt_idx];
        let prompt_text = format!("{} ", prompt.text);
        let source_layer = candidate.feature.layer;
        let n_injection_layers = n_layers - source_layer;

        eprintln!(
            "\n--- [{}/{}] [{:5}] \"...{}\" → inject \"{}\" L{}:{} ({} downstream layers) ---",
            sweep_idx + 1,
            total,
            candidate.group,
            candidate.target_word,
            candidate.feature_word,
            source_layer,
            candidate.feature.index,
            n_injection_layers
        );

        // Tokenize
        let token_strs = model.tokenize(&prompt_text)?;
        let seq_len = token_strs.len();
        eprintln!("  Tokens: {seq_len}");

        // Find target token ID
        let target_token_id = find_target_token_id(&model, &candidate.feature_word)?;
        let target_token_str = model.decode_token(target_token_id);
        eprintln!("  Target token: \"{target_token_str}\" (id={target_token_id})");

        // Build multi-layer injection targets: (feature, L), (feature, L+1), ..., (feature, 25)
        let features_for_injection: Vec<(CltFeatureId, usize)> = (source_layer..n_layers)
            .map(|target_layer| (candidate.feature, target_layer))
            .collect();
        eprintln!(
            "  Injection: {} entries (layers {}–{})",
            features_for_injection.len(),
            source_layer,
            n_layers - 1
        );

        // Baseline
        eprintln!("  Running baseline...");
        let zero_spec = clt.prepare_injection(&features_for_injection, 0, 0.0)?;
        let baseline_result = model.clt_logit_shift(&prompt_text, &zero_spec)?;
        let baseline_prob = extract_token_prob(&baseline_result.baseline_logits, target_token_id)?;
        eprintln!("  Baseline P(\"{target_token_str}\") = {baseline_prob:.6e}");

        // Sweep: inject at each position
        eprintln!(
            "  Sweeping {} positions (strength={})...",
            seq_len, args.strength
        );
        let mut positions: Vec<PositionProb> = Vec::with_capacity(seq_len);

        for (pos, tok_str) in token_strs.iter().enumerate() {
            let spec = clt.prepare_injection(&features_for_injection, pos, args.strength)?;
            let result = model.clt_logit_shift(&prompt_text, &spec)?;
            let steered_prob = extract_token_prob(&result.injected_logits, target_token_id)?;

            positions.push(PositionProb {
                position: pos,
                token: tok_str.clone(),
                steered_prob,
            });
        }

        // Print table
        eprintln!(
            "\n  {:>3}  {:<20}  {:>14}  {:>14}",
            "Pos", "Token", "P(target)", "Delta"
        );
        eprintln!("  {:-<3}  {:-<20}  {:-<14}  {:-<14}", "", "", "", "");
        for p in &positions {
            let delta = p.steered_prob - baseline_prob;
            let marker = if delta > baseline_prob * 10.0 && delta > 1e-12 {
                " ***"
            } else if delta > baseline_prob && delta > 1e-12 {
                " *"
            } else {
                ""
            };
            let display = p.token.replace('\n', "\\n");
            eprintln!(
                "  {:>3}  {:<20}  {:>14.6e}  {:>+14.6e}{}",
                p.position, display, p.steered_prob, delta, marker
            );
        }

        // Summary stats
        let (max_pos, max_prob) = positions
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.steered_prob
                    .partial_cmp(&b.steered_prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or((0, 0.0), |(i, p)| (i, p.steered_prob));
        let last_prob = positions.last().map_or(0.0, |p| p.steered_prob);

        eprintln!("\n  Baseline:   {baseline_prob:.6e}");
        eprintln!(
            "  Max P:      {max_prob:.6e} at position {max_pos} (\"{}\")  ratio={:.1}x",
            token_strs[max_pos].replace('\n', "\\n"),
            if baseline_prob > 0.0 {
                max_prob / baseline_prob
            } else {
                0.0
            }
        );
        eprintln!(
            "  Last-token: {last_prob:.6e}  ratio={:.1}x",
            if baseline_prob > 0.0 {
                last_prob / baseline_prob
            } else {
                0.0
            }
        );

        results.push(SteeringSweepResult {
            prompt_text: prompt_text.clone(),
            group: candidate.group.clone(),
            target_word: candidate.target_word.clone(),
            feature_word: candidate.feature_word.clone(),
            target_feature: candidate.feature,
            source_layer,
            n_injection_layers,
            target_token_id,
            baseline_prob,
            max_steered_prob: max_prob,
            max_steered_position: max_pos,
            last_position_prob: last_prob,
            positions,
        });
    }

    // 6. Summary
    eprintln!("\n=== Summary (sorted by source layer) ===");
    eprintln!(
        "  {:>5}  {:>6}  {:>6}  {:>4}  {:>5}  {:>12}  {:>12}  {:>6}  {:>12}",
        "Group", "Target", "Feat", "SrcL", "#Lyr", "Baseline", "MaxP", "MaxPos", "Ratio"
    );
    for r in &results {
        let ratio = if r.baseline_prob > 0.0 {
            r.max_steered_prob / r.baseline_prob
        } else {
            0.0
        };
        eprintln!(
            "  {:>5}  {:>6}  {:>6}  {:>4}  {:>5}  {:>12.4e}  {:>12.4e}  {:>6}  {:>12.1}x",
            r.group,
            r.target_word,
            r.feature_word,
            r.source_layer,
            r.n_injection_layers,
            r.baseline_prob,
            r.max_steered_prob,
            r.max_steered_position,
            ratio
        );
    }

    // 7. JSON output
    let output = SteeringSweepOutput {
        model: args.model,
        clt_repo: args.clt_repo,
        strength: args.strength,
        n_layers,
        n_candidates: results.len(),
        results,
    };

    let json = serde_json::to_string_pretty(&output)?;
    if let Some(ref p) = args.output {
        fs::write(p, &json)?;
        eprintln!("\nOutput written to {}", p.display());
    } else {
        println!("{json}");
    }

    Ok(())
}

/// Find the token ID for a target word (tries " word" with leading space first,
/// then bare "word").
fn find_target_token_id(model: &PlipModel, word: &str) -> Result<u32> {
    let with_space = format!(" {word}");
    let ids = model.encode(&with_space)?;
    if ids.len() == 2 {
        return Ok(ids[1]);
    }
    let bare_ids = model.encode(word)?;
    if bare_ids.len() == 2 {
        return Ok(bare_ids[1]);
    }
    ids.last()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("Could not find token ID for \"{word}\""))
}

/// Extract P(token_id) from logits via softmax.
fn extract_token_prob(logits: &candle_core::Tensor, token_id: u32) -> Result<f32> {
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let probs = candle_nn::ops::softmax_last_dim(&logits_f32)?;
    let prob = probs
        .flatten_all()?
        .i(token_id as usize)?
        .to_scalar::<f32>()?;
    Ok(prob)
}
