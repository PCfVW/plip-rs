//! Version D of Anthropic Figure 13 replication: Suppress + Inject steering sweep.
//!
//! For each prompt, suppresses all features from the natural rhyme group
//! (negative strength, multi-layer) while injecting the best feature from an
//! alternative group (positive strength, multi-layer). Sweeps the combined
//! intervention position across all tokens and measures P(inject_word).
//!
//! This implements Anthropic's full suppress+inject protocol from Figure 13:
//! suppress the natural plan (e.g., "rabbit"/"habit") and inject an alternative
//! ("green"), testing whether the rhyme can be *redirected*.
//!
//! Usage:
//!   cargo run --release --example suppress_inject_sweep -- \
//!       --rhyme-pairs outputs/rhyme_pairs_all_layers.json \
//!       --output outputs/suppress_inject_sweep.json

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
use plip_rs::{CltFeatureId, CltInjectionSpec, CrossLayerTranscoder, PlipModel};
use serde::{Deserialize, Serialize};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "suppress_inject_sweep")]
#[command(about = "Figure 13 Version D: suppress + inject steering sweep (Melometis)")]
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

    /// Suppression strength (applied as negative)
    #[arg(long, default_value_t = 10.0)]
    suppress_strength: f32,

    /// Injection strength (applied as positive)
    #[arg(long, default_value_t = 10.0)]
    inject_strength: f32,

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
struct SuppressInjectOutput {
    model: String,
    clt_repo: String,
    suppress_strength: f32,
    inject_strength: f32,
    n_layers: usize,
    n_pairs: usize,
    results: Vec<SuppressInjectResult>,
}

#[derive(Serialize)]
struct SuppressInjectResult {
    prompt_text: String,
    natural_group: String,
    alt_group: String,
    inject_word: String,
    inject_feature: CltFeatureId,
    inject_source_layer: usize,
    n_suppress_features: usize,
    n_suppress_entries: usize,
    n_inject_entries: usize,
    inject_token_id: u32,
    baseline_p_inject: f32,
    max_steered_p_inject: f32,
    max_position: usize,
    last_position_p: f32,
    positions: Vec<PositionResult>,
}

#[derive(Serialize)]
struct PositionResult {
    position: usize,
    token: String,
    p_inject: f32,
}

// ── Prompts (same 4 as Versions A/B/C) ─────────────────────────────────────

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

// ── Experiment pairs ────────────────────────────────────────────────────────

struct ExperimentPair {
    prompt_idx: usize,
    natural_features: Vec<CltFeatureId>,
    alt_group_name: String,
    inject_feature: CltFeatureId,
    inject_word: String,
}

/// Map rhyme ending phonemes to display names using the prompts as reference.
fn build_ending_display_names(prompts: &[CompletionPrompt]) -> HashMap<String, String> {
    let mut names = HashMap::new();
    for p in prompts {
        names.insert(p.rhyme_ending.to_string(), p.group.to_string());
    }
    names
}

fn build_experiment_pairs(
    prompts: &[CompletionPrompt],
    ending_to_features: &HashMap<String, Vec<(String, CltFeatureId)>>,
    ending_display_names: &HashMap<String, String>,
) -> Vec<ExperimentPair> {
    let mut pairs = Vec::new();
    let all_endings: Vec<String> = ending_to_features.keys().cloned().collect();

    for (prompt_idx, prompt) in prompts.iter().enumerate() {
        let natural_ending = prompt.rhyme_ending;
        let natural_features: Vec<CltFeatureId> = ending_to_features
            .get(natural_ending)
            .map(|fs| fs.iter().map(|(_, f)| *f).collect())
            .unwrap_or_default();

        for alt_ending in &all_endings {
            if *alt_ending == natural_ending {
                continue;
            }
            let alt_features = match ending_to_features.get(alt_ending) {
                Some(fs) if !fs.is_empty() => fs,
                _ => continue,
            };

            // Pick the feature with the earliest source layer (most downstream propagation)
            let (inject_word, inject_feature) =
                alt_features.iter().min_by_key(|(_, f)| f.layer).unwrap();

            let alt_group_name = ending_display_names
                .get(alt_ending)
                .cloned()
                .unwrap_or_else(|| alt_ending.clone());

            pairs.push(ExperimentPair {
                prompt_idx,
                natural_features: natural_features.clone(),
                alt_group_name,
                inject_feature: *inject_feature,
                inject_word: inject_word.clone(),
            });
        }
    }

    // Sort by prompt index, then by inject source layer (earliest first)
    pairs.sort_by_key(|p| (p.prompt_idx, p.inject_feature.layer));
    pairs
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    eprintln!("=== Suppress + Inject Sweep: Figure 13 Version D ===\n");

    // 1. Load rhyme pairs
    let rp_text = fs::read_to_string(&args.rhyme_pairs)
        .with_context(|| format!("reading rhyme pairs from {}", args.rhyme_pairs.display()))?;
    let rp: RhymePairsOutput = serde_json::from_str(&rp_text)?;

    // Build ending → features lookup
    let mut ending_to_features: HashMap<String, Vec<(String, CltFeatureId)>> = HashMap::new();
    for g in &rp.rhyme_groups {
        let features: Vec<(String, CltFeatureId)> = g
            .words
            .iter()
            .map(|w| (w.word.clone(), w.feature))
            .collect();
        ending_to_features.insert(g.rhyme_ending.clone(), features);
    }

    // 2. Build experiment pairs
    let prompts = make_sweep_prompts();
    let ending_display_names = build_ending_display_names(&prompts);
    let pairs = build_experiment_pairs(&prompts, &ending_to_features, &ending_display_names);

    eprintln!("Experiment pairs ({} total):", pairs.len());
    for p in &pairs {
        let prompt = &prompts[p.prompt_idx];
        eprintln!(
            "  prompt {} \"{}\": suppress {} ({} features) → inject \"{}\" {} (L{}:{})",
            p.prompt_idx,
            prompt.target_word,
            prompt.group,
            p.natural_features.len(),
            p.inject_word,
            p.alt_group_name,
            p.inject_feature.layer,
            p.inject_feature.index,
        );
    }

    // 3. Load model
    eprintln!("\nLoading model: {}", args.model);
    let force_cpu = if args.cpu { Some(true) } else { None };
    let model = PlipModel::from_pretrained_with_device(&args.model, force_cpu)?;

    // 4. Open CLT and cache ALL needed decoder vectors
    eprintln!("Opening CLT: {}", args.clt_repo);
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let n_layers = clt.config().n_layers;
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    };

    // Collect ALL unique features (suppress + inject)
    let unique_features: Vec<CltFeatureId> = {
        let mut all = Vec::new();
        for p in &pairs {
            all.extend(&p.natural_features);
            all.push(p.inject_feature);
        }
        all.into_iter()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    };
    eprintln!(
        "Caching decoder vectors for {} unique features (all downstream layers)...",
        unique_features.len()
    );
    clt.cache_steering_vectors_all_downstream(&unique_features, &device)?;

    // 5. Run suppress+inject sweep for each pair
    let mut results: Vec<SuppressInjectResult> = Vec::new();
    let total = pairs.len();

    for (pair_idx, pair) in pairs.iter().enumerate() {
        let prompt = &prompts[pair.prompt_idx];
        let prompt_text = format!("{} ", prompt.text);
        let inject_source_layer = pair.inject_feature.layer;

        eprintln!(
            "\n--- [{}/{}] \"{}\" {} → suppress {} ({} feat) + inject \"{}\" {} L{}:{} ---",
            pair_idx + 1,
            total,
            prompt.target_word,
            prompt.group,
            prompt.group,
            pair.natural_features.len(),
            pair.inject_word,
            pair.alt_group_name,
            inject_source_layer,
            pair.inject_feature.index,
        );

        // Tokenize
        let token_strs = model.tokenize(&prompt_text)?;
        let seq_len = token_strs.len();
        eprintln!("  Tokens: {seq_len}");

        // Find inject word token ID
        let inject_token_id = find_target_token_id(&model, &pair.inject_word)?;
        let inject_token_str = model.decode_token(inject_token_id);
        eprintln!("  Inject token: \"{inject_token_str}\" (id={inject_token_id})");

        // Build multi-layer feature entries for suppress and inject
        let suppress_entries: Vec<(CltFeatureId, usize)> = pair
            .natural_features
            .iter()
            .flat_map(|f| (f.layer..n_layers).map(move |l| (*f, l)))
            .collect();
        let inject_entries: Vec<(CltFeatureId, usize)> = (inject_source_layer..n_layers)
            .map(|l| (pair.inject_feature, l))
            .collect();

        eprintln!(
            "  Suppress: {} entries across {} features",
            suppress_entries.len(),
            pair.natural_features.len()
        );
        eprintln!(
            "  Inject: {} entries (layers {}–{})",
            inject_entries.len(),
            inject_source_layer,
            n_layers - 1
        );

        // Baseline (no intervention)
        eprintln!("  Running baseline...");
        let zero_spec = clt.prepare_injection(&inject_entries, 0, 0.0)?;
        let baseline_result = model.clt_logit_shift(&prompt_text, &zero_spec)?;
        let baseline_p_inject =
            extract_token_prob(&baseline_result.baseline_logits, inject_token_id)?;
        eprintln!("  Baseline P(\"{inject_token_str}\") = {baseline_p_inject:.6e}");

        // Sweep: suppress + inject at each position
        eprintln!(
            "  Sweeping {} positions (suppress={}, inject={})...",
            seq_len, args.suppress_strength, args.inject_strength
        );
        let mut positions: Vec<PositionResult> = Vec::with_capacity(seq_len);

        for (pos, tok_str) in token_strs.iter().enumerate() {
            let suppress_spec =
                clt.prepare_injection(&suppress_entries, pos, -args.suppress_strength)?;
            let inject_spec = clt.prepare_injection(&inject_entries, pos, args.inject_strength)?;
            let merged_spec = merge_injection_specs(suppress_spec, inject_spec);

            let result = model.clt_logit_shift(&prompt_text, &merged_spec)?;
            let p_inject = extract_token_prob(&result.injected_logits, inject_token_id)?;

            positions.push(PositionResult {
                position: pos,
                token: tok_str.clone(),
                p_inject,
            });
        }

        // Print table
        eprintln!(
            "\n  {:>3}  {:<20}  {:>14}  {:>14}",
            "Pos", "Token", "P(inject)", "Delta"
        );
        eprintln!("  {:-<3}  {:-<20}  {:-<14}  {:-<14}", "", "", "", "");
        for p in &positions {
            let delta = p.p_inject - baseline_p_inject;
            let marker = if delta > baseline_p_inject * 10.0 && delta > 1e-12 {
                " ***"
            } else if delta > baseline_p_inject && delta > 1e-12 {
                " *"
            } else {
                ""
            };
            let display = p.token.replace('\n', "\\n");
            eprintln!(
                "  {:>3}  {:<20}  {:>14.6e}  {:>+14.6e}{}",
                p.position, display, p.p_inject, delta, marker
            );
        }

        // Summary stats
        let (max_pos, max_p) = positions
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.p_inject
                    .partial_cmp(&b.p_inject)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or((0, 0.0), |(i, p)| (i, p.p_inject));
        let last_p = positions.last().map_or(0.0, |p| p.p_inject);

        eprintln!("\n  Baseline:   {baseline_p_inject:.6e}");
        eprintln!(
            "  Max P:      {max_p:.6e} at position {max_pos} (\"{}\")  ratio={:.1}x",
            token_strs[max_pos].replace('\n', "\\n"),
            if baseline_p_inject > 0.0 {
                max_p / baseline_p_inject
            } else {
                0.0
            }
        );
        eprintln!(
            "  Last-token: {last_p:.6e}  ratio={:.1}x",
            if baseline_p_inject > 0.0 {
                last_p / baseline_p_inject
            } else {
                0.0
            }
        );

        results.push(SuppressInjectResult {
            prompt_text: prompt_text.clone(),
            natural_group: prompt.group.to_string(),
            alt_group: pair.alt_group_name.clone(),
            inject_word: pair.inject_word.clone(),
            inject_feature: pair.inject_feature,
            inject_source_layer,
            n_suppress_features: pair.natural_features.len(),
            n_suppress_entries: suppress_entries.len(),
            n_inject_entries: inject_entries.len(),
            inject_token_id,
            baseline_p_inject,
            max_steered_p_inject: max_p,
            max_position: max_pos,
            last_position_p: last_p,
            positions,
        });
    }

    // 6. Summary
    eprintln!("\n=== Summary (sorted by prompt, then inject source layer) ===");
    eprintln!(
        "  {:>5}  {:>6}  {:>5}  {:>8}  {:>4}  {:>5}  {:>12}  {:>12}  {:>6}  {:>12}",
        "NatGr",
        "Target",
        "AltGr",
        "InjectW",
        "SrcL",
        "#Sup",
        "Baseline",
        "MaxP",
        "MaxPos",
        "Ratio"
    );
    for r in &results {
        let ratio = if r.baseline_p_inject > 0.0 {
            r.max_steered_p_inject / r.baseline_p_inject
        } else {
            0.0
        };
        eprintln!(
            "  {:>5}  {:>6}  {:>5}  {:>8}  {:>4}  {:>5}  {:>12.4e}  {:>12.4e}  {:>6}  {:>12.1}x",
            r.natural_group,
            r.inject_word,
            r.alt_group,
            r.inject_word,
            r.inject_source_layer,
            r.n_suppress_features,
            r.baseline_p_inject,
            r.max_steered_p_inject,
            r.max_position,
            ratio
        );
    }

    // 7. JSON output
    let output = SuppressInjectOutput {
        model: args.model,
        clt_repo: args.clt_repo,
        suppress_strength: args.suppress_strength,
        inject_strength: args.inject_strength,
        n_layers,
        n_pairs: results.len(),
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

/// Merge two [`CltInjectionSpec`]s by concatenating their injection lists.
fn merge_injection_specs(mut a: CltInjectionSpec, b: CltInjectionSpec) -> CltInjectionSpec {
    a.injections.extend(b.injections);
    a
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
