//! CLT Feature Steering for Poetry Generation (Melometis Phase 2a)
//!
//! Implements cross-layer transcoder feature injection at newline planning
//! sites to steer rhyme line endings, per §2.3 of the experiment document.
//!
//! Four modes:
//!   - `identify`:  Find CLT features for a target word (Methods 1 & 2)
//!   - `calibrate`: Dose-response calibration (5 prompts × 30 completions × 7 strengths)
//!   - `run`:       Main experiment (120 pairs × 7 strengths × 3 samples)
//!   - `evaluate`:  Compute metrics and go/no-go assessment
//!
//! Feature identification methods (run mode):
//!   - Methods 1-2: Per-word (`--activation-search`, `--decoder-projection`)
//!   - Method 3: Per-(prompt, word) via planning-site activation + decoder
//!     filtering (`--method3`)
//!   - Method 4: Method 3 identification + multi-layer clamping injection
//!     (`--method4`)
//!   - Method 5: Target-word probe activation + contrastive scoring
//!     (`--method5`)
//!   - Method 6: Causal activation patching — ablate features to find
//!     production features (`--method6`)
//!
//! Usage:
//!   cargo run --release --example poetry_clt_steering -- \
//!       --mode identify --target-word wall --decoder-projection \
//!       --output outputs/features_wall.json
//!
//!   cargo run --release --example poetry_clt_steering -- \
//!       --mode calibrate --features outputs/features_wall.json \
//!       --output outputs/calibration_wall.json
//!
//!   cargo run --release --example poetry_clt_steering -- \
//!       --mode run --decoder-projection \
//!       --output outputs/steering_results.json
//!
//!   cargo run --release --example poetry_clt_steering -- \
//!       --mode run --method3 \
//!       --output outputs/method3_results.json
//!
//!   cargo run --release --example poetry_clt_steering -- \
//!       --mode run --method4 \
//!       --output outputs/method4_results.json
//!
//!   cargo run --release --example poetry_clt_steering -- \
//!       --mode run --method5 \
//!       --output outputs/method5_results.json
//!
//!   cargo run --release --example poetry_clt_steering -- \
//!       --mode run --method6 \
//!       --output outputs/method6_results.json
//!
//!   cargo run --release --example poetry_clt_steering -- \
//!       --mode evaluate --results outputs/steering_results.json

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::struct_excessive_bools)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use plip_rs::{ActivationCache, CltFeatureId, CltInjectionSpec, CrossLayerTranscoder, PlipModel};
use serde::{Deserialize, Serialize};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "poetry_clt_steering")]
#[command(about = "CLT feature steering for poetry generation (Melometis Phase 2a)")]
struct Args {
    /// Mode: identify | calibrate | run | evaluate
    #[arg(long)]
    mode: String,

    /// HuggingFace model ID
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// HuggingFace CLT repository
    #[arg(long, default_value = "mntss/clt-gemma-2-2b-426k")]
    clt_repo: String,

    /// Path to poetry corpus JSON
    #[arg(long, default_value = "corpus/attention_samples_poetry.json")]
    corpus: PathBuf,

    /// Output file path
    #[arg(long)]
    output: Option<PathBuf>,

    /// Force CPU execution
    #[arg(long)]
    cpu: bool,

    /// Target word (identify mode)
    #[arg(long)]
    target_word: Option<String>,

    /// Path to identify output (calibrate mode)
    #[arg(long)]
    features: Option<PathBuf>,

    /// Path to run output (evaluate mode)
    #[arg(long)]
    results: Option<PathBuf>,

    /// Enable Method 1: max-activation search
    #[arg(long)]
    activation_search: bool,

    /// Enable Method 2: decoder projection scoring
    #[arg(long)]
    decoder_projection: bool,

    /// Use cosine similarity instead of dot product for decoder projection scoring
    #[arg(long)]
    cosine: bool,

    /// Number of top features to select
    #[arg(long, default_value_t = 5)]
    n_features: usize,

    /// Completions per prompt (calibrate mode)
    #[arg(long, default_value_t = 30)]
    n_completions: usize,

    /// Comma-separated steering strengths
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

    /// Limit unique target words for diagnostics (run mode, 0 = no limit)
    #[arg(long, default_value_t = 0)]
    max_words: usize,

    /// Enable Method 3: planning-site activation + decoder filtering
    #[arg(long)]
    method3: bool,

    /// Enable Method 4: planning-site features + multi-layer clamping injection
    #[arg(long)]
    method4: bool,

    /// Enable Method 5: activation-based contrastive feature identification
    #[arg(long)]
    method5: bool,

    /// Enable Method 6: causal activation patching
    #[arg(long)]
    method6: bool,

    /// Number of candidate features to test per prompt in causal patching (Method 6)
    #[arg(long, default_value_t = 50)]
    n_candidates: usize,
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

// ── Output types ────────────────────────────────────────────────────────────

// -- identify --

#[derive(Serialize, Deserialize)]
struct IdentifyOutput {
    model: String,
    clt_repo: String,
    target_word: String,
    target_token_ids: Vec<u32>,
    method1_features: Vec<ScoredFeature>,
    method2_features: Vec<ScoredFeature>,
    selected_features: Vec<SelectedFeature>,
}

#[derive(Serialize, Deserialize, Clone)]
struct ScoredFeature {
    feature: CltFeatureId,
    score: f32,
    method: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct SelectedFeature {
    feature: CltFeatureId,
    target_layer: usize,
    score: f32,
    source_method: String,
}

// -- calibrate --

#[derive(Serialize, Deserialize)]
struct CalibrateOutput {
    model: String,
    clt_repo: String,
    target_word: String,
    features: Vec<SelectedFeature>,
    results_by_strength: Vec<StrengthResult>,
    best_strength: f32,
    best_target_hit_rate: f32,
    go_no_go: String,
}

#[derive(Serialize, Deserialize)]
struct StrengthResult {
    strength: f32,
    target_hit_rate: f32,
    rhyme_hit_rate: f32,
    n_completions: usize,
    completions: Vec<CompletionRecord>,
}

#[derive(Serialize, Deserialize)]
struct CompletionRecord {
    prompt_id: String,
    generated_line: String,
    ending_word: String,
    target_hit: bool,
    rhyme_hit: bool,
}

// -- run --

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

/// Return type for `mode_run_methods12` and `mode_run_method3`:
/// (pairs, prebuilt_specs, pair_feature_strs).
type RunPipelineResult = (
    Vec<SteeringPair>,
    Vec<Vec<CltInjectionSpec>>,
    Vec<Vec<String>>,
);

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    match args.mode.as_str() {
        "identify" => mode_identify(&args),
        "calibrate" => mode_calibrate(&args),
        "run" => mode_run(&args),
        "evaluate" => mode_evaluate(&args),
        other => anyhow::bail!("Unknown mode: {other}. Use identify|calibrate|run|evaluate"),
    }
}

// ── Identify mode ───────────────────────────────────────────────────────────

fn mode_identify(args: &Args) -> Result<()> {
    let target_word = args
        .target_word
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--target-word is required for identify mode"))?;

    anyhow::ensure!(
        args.activation_search || args.decoder_projection,
        "At least one of --activation-search or --decoder-projection must be set"
    );

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    eprintln!("Device: {device:?}");

    // Load model
    eprintln!("Loading model: {}", args.model);
    let model = PlipModel::from_pretrained(&args.model)?;
    let n_layers = model.n_layers();
    eprintln!("Model loaded: {n_layers} layers");

    // Tokenize target word (with leading space for SentencePiece context)
    let target_with_space = format!(" {target_word}");
    let target_token_ids = model.encode(&target_with_space)?;
    let scoring_token_id = *target_token_ids
        .last()
        .ok_or_else(|| anyhow::anyhow!("Target word tokenizes to empty sequence"))?;
    eprintln!(
        "Target word \"{target_word}\" → token IDs: {target_token_ids:?} (scoring: {scoring_token_id} = \"{}\")",
        model.decode_token(scoring_token_id)
    );

    // Open CLT
    eprintln!("Opening CLT: {}", args.clt_repo);
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let clt_config = clt.config().clone();
    eprintln!(
        "CLT: {} features/layer, {} layers",
        clt_config.n_features_per_layer, clt_config.n_layers
    );

    // Method 1: max-activation search
    let method1_features = if args.activation_search {
        eprintln!("\n=== Method 1: Max-activation search ===");
        let probe_text = format!("The word is {target_word}");
        let activations = model.get_activations(&probe_text)?;

        let mut all_features: Vec<(CltFeatureId, f32)> = Vec::new();
        for layer in 0..n_layers {
            let residual = activations
                .get_layer(layer)
                .ok_or_else(|| anyhow::anyhow!("No activation at layer {layer}"))?;
            clt.load_encoder(layer, &device)?;
            let top = clt.top_k(residual, layer, 20)?;
            for (fid, act) in &top.features {
                all_features.push((*fid, *act));
            }
            eprintln!(
                "  Layer {layer:>2}: top activation = {:.4}",
                top.features.first().map_or(0.0, |(_, a)| *a)
            );
        }

        all_features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all_features.dedup_by(|a, b| a.0 == b.0);
        all_features.truncate(20);

        let result: Vec<ScoredFeature> = all_features
            .into_iter()
            .map(|(feature, score)| ScoredFeature {
                feature,
                score,
                method: "activation_search".to_string(),
            })
            .collect();

        eprintln!("Method 1: {} features identified", result.len());
        for f in result.iter().take(5) {
            eprintln!("  {}: score = {:.4}", f.feature, f.score);
        }
        result
    } else {
        Vec::new()
    };

    // Method 2: decoder projection
    let method2_features = if args.decoder_projection {
        let scoring_mode = if args.cosine {
            "cosine similarity"
        } else {
            "dot product"
        };
        eprintln!("\n=== Method 2: Decoder projection ({scoring_mode}) ===");
        let embed_vec = model.token_embedding(scoring_token_id)?;
        let final_layer = n_layers - 1;
        eprintln!(
            "Scoring all features by decoder projection to layer {final_layer} for token {scoring_token_id}..."
        );

        let scored =
            clt.score_features_by_decoder_projection(&embed_vec, final_layer, 20, args.cosine)?;

        let result: Vec<ScoredFeature> = scored
            .into_iter()
            .map(|(feature, score)| ScoredFeature {
                feature,
                score,
                method: "decoder_projection".to_string(),
            })
            .collect();

        eprintln!("Method 2: {} features identified", result.len());
        for f in result.iter().take(5) {
            eprintln!("  {}: score = {:.4}", f.feature, f.score);
        }
        result
    } else {
        Vec::new()
    };

    // Combine and select top-N
    let selected = combine_features(
        &method1_features,
        &method2_features,
        args.n_features,
        n_layers,
    );
    eprintln!("\n=== Selected {} features ===", selected.len());
    for f in &selected {
        eprintln!(
            "  {} → layer {} (score={:.4}, method={})",
            f.feature, f.target_layer, f.score, f.source_method
        );
    }

    let output = IdentifyOutput {
        model: args.model.clone(),
        clt_repo: args.clt_repo.clone(),
        target_word: target_word.to_string(),
        target_token_ids,
        method1_features,
        method2_features,
        selected_features: selected,
    };

    write_output(&output, args.output.as_deref(), "identify")?;
    Ok(())
}

// ── Calibrate mode ──────────────────────────────────────────────────────────

fn mode_calibrate(args: &Args) -> Result<()> {
    let features_path = args
        .features
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--features is required for calibrate mode"))?;

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    // Load identify output
    let identify_json = fs::read_to_string(features_path)
        .with_context(|| format!("Failed to read features file: {}", features_path.display()))?;
    let identify: IdentifyOutput = serde_json::from_str(&identify_json)?;
    let target_word = &identify.target_word;
    let features = &identify.selected_features;

    eprintln!("Target word: \"{target_word}\"");
    eprintln!("Features: {} selected", features.len());

    // Parse strengths
    let strengths = parse_strengths(&args.strengths)?;

    // Load model
    eprintln!("Loading model: {}", args.model);
    let model = PlipModel::from_pretrained(&args.model)?;

    // Open CLT and cache steering vectors
    eprintln!("Opening CLT: {}", args.clt_repo);
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let feature_targets: Vec<(CltFeatureId, usize)> = features
        .iter()
        .map(|f| (f.feature, f.target_layer))
        .collect();
    clt.cache_steering_vectors(&feature_targets, &device)?;

    // Load corpus and select 5 calibration prompts
    let corpus = load_corpus(&args.corpus)?;
    let cal_prompts = select_calibration_prompts(&corpus, target_word, 5);
    eprintln!(
        "Selected {} calibration prompts (rhyme group match)",
        cal_prompts.len()
    );

    // Find stop tokens
    let newline_token = find_newline_token(&model)?;
    let mut stop_tokens = vec![newline_token];
    if let Some(eos) = model.eos_token_id() {
        stop_tokens.push(eos);
    }

    // Run calibration
    let mut results_by_strength = Vec::new();
    let total_gens = strengths.len() * cal_prompts.len() * args.n_completions;
    let mut gen_count = 0;

    for &strength in &strengths {
        let mut completions = Vec::new();

        for prompt in &cal_prompts {
            let tokens = model.encode(&prompt.code)?;
            let position = tokens.len().saturating_sub(1);

            let spec = if strength == 0.0 {
                CltInjectionSpec::new()
            } else {
                clt.prepare_injection(&feature_targets, position, strength)?
            };

            for _ in 0..args.n_completions {
                gen_count += 1;
                let generated = model.generate_with_clt_injection(
                    &prompt.code,
                    args.max_tokens,
                    args.temperature,
                    &stop_tokens,
                    &spec,
                )?;

                let gen_line = extract_generated_line(&generated, &prompt.code);
                let ending = extract_ending_word(&gen_line);
                let target_hit = ending.eq_ignore_ascii_case(target_word);
                let rhyme_hit = words_rhyme(&ending, &prompt.rhyme_group);

                completions.push(CompletionRecord {
                    prompt_id: prompt.id.clone(),
                    generated_line: gen_line,
                    ending_word: ending,
                    target_hit,
                    rhyme_hit,
                });

                if gen_count % 10 == 0 || gen_count == total_gens {
                    eprint!("\r  [{gen_count}/{total_gens}] strength={strength:.1}");
                }
            }
        }

        let n = completions.len();
        let target_hits = completions.iter().filter(|c| c.target_hit).count();
        let rhyme_hits = completions.iter().filter(|c| c.rhyme_hit).count();
        let target_hit_rate = target_hits as f32 / n.max(1) as f32;
        let rhyme_hit_rate = rhyme_hits as f32 / n.max(1) as f32;

        eprintln!(
            "\n  strength={strength:.1}: target={target_hits}/{n} ({:.1}%), rhyme={rhyme_hits}/{n} ({:.1}%)",
            target_hit_rate * 100.0,
            rhyme_hit_rate * 100.0
        );

        results_by_strength.push(StrengthResult {
            strength,
            target_hit_rate,
            rhyme_hit_rate,
            n_completions: n,
            completions,
        });
    }

    // Go/no-go — extract values before moving results_by_strength
    let (best_strength, best_target_hit_rate) = results_by_strength
        .iter()
        .max_by(|a, b| {
            a.target_hit_rate
                .partial_cmp(&b.target_hit_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or((0.0, 0.0), |b| (b.strength, b.target_hit_rate));

    let go_no_go = if best_target_hit_rate >= 0.10 {
        format!(
            "PASS: target_hit_rate = {:.1}% at strength {best_strength:.1} (>= 10% threshold)",
            best_target_hit_rate * 100.0,
        )
    } else {
        format!(
            "FAIL: best target_hit_rate = {:.1}% at strength {best_strength:.1} (< 10% threshold)",
            best_target_hit_rate * 100.0,
        )
    };

    eprintln!("\n{go_no_go}");

    let output = CalibrateOutput {
        model: args.model.clone(),
        clt_repo: args.clt_repo.clone(),
        target_word: target_word.clone(),
        features: features.clone(),
        results_by_strength,
        best_strength,
        best_target_hit_rate,
        go_no_go,
    };

    write_output(&output, args.output.as_deref(), "calibrate")?;
    Ok(())
}

// ── Run mode ────────────────────────────────────────────────────────────────

fn mode_run(args: &Args) -> Result<()> {
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    let strengths = parse_strengths(&args.strengths)?;

    // Load model
    eprintln!("Loading model: {}", args.model);
    let model = PlipModel::from_pretrained(&args.model)?;
    let n_layers = model.n_layers();
    let final_layer = n_layers - 1;

    // Load corpus
    let corpus = load_corpus(&args.corpus)?;

    // Open CLT
    eprintln!("Opening CLT: {}", args.clt_repo);
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;

    // ── Feature identification ──────────────────────────────────────────
    //
    // Two paths:
    //   Methods 1-2: features identified per target word (word-specific)
    //   Method 3:    features identified per (prompt, target_word) pair
    //
    // Both paths build steering pairs from the corpus and produce:
    //   pairs:              Vec<SteeringPair>
    //   prebuilt_specs:     Vec<Vec<CltInjectionSpec>> (per pair × per strength)
    //   pair_feature_strs:  Vec<Vec<String>>           (per pair feature IDs)

    let pairs: Vec<SteeringPair>;
    let prebuilt_specs: Vec<Vec<CltInjectionSpec>>;
    let pair_feature_strs: Vec<Vec<String>>;

    if args.method6 {
        // ── Method 6: Causal activation patching ──
        let result = mode_run_method6(
            args, &model, &mut clt, &corpus, &strengths, n_layers, &device,
        )?;
        pairs = result.0;
        prebuilt_specs = result.1;
        pair_feature_strs = result.2;
    } else if args.method5 {
        // ── Method 5: Activation-based contrastive feature identification ──
        let result = mode_run_method5(
            args, &model, &mut clt, &corpus, &strengths, n_layers, &device,
        )?;
        pairs = result.0;
        prebuilt_specs = result.1;
        pair_feature_strs = result.2;
    } else if args.method4 || args.method3 {
        // ── Method 3/4: Planning-site activation + decoder filtering ──
        // Method 4 adds multi-layer clamping injection on top of Method 3
        let result = mode_run_method3(
            args,
            &model,
            &mut clt,
            &corpus,
            &strengths,
            n_layers,
            final_layer,
            &device,
        )?;
        pairs = result.0;
        prebuilt_specs = result.1;
        pair_feature_strs = result.2;
    } else {
        // ── Methods 1-2: Per-word feature identification ─────────────
        let result = mode_run_methods12(
            args,
            &model,
            &mut clt,
            &corpus,
            &strengths,
            n_layers,
            final_layer,
            &device,
        )?;
        pairs = result.0;
        prebuilt_specs = result.1;
        pair_feature_strs = result.2;
    }

    // Drop CLT to free all decoder-related memory before generation
    drop(clt);
    eprintln!("Dropped CLT — freed decoder memory before generation");

    // ── Generation (common to all methods) ──────────────────────────────

    let newline_token = find_newline_token(&model)?;
    let mut stop_tokens = vec![newline_token];
    if let Some(eos) = model.eos_token_id() {
        stop_tokens.push(eos);
    }

    let total_gens = pairs.len() * strengths.len() * args.n_seeds;
    eprintln!(
        "Running {} generations ({} pairs × {} strengths × {} seeds)...",
        total_gens,
        pairs.len(),
        strengths.len(),
        args.n_seeds
    );

    let mut results = Vec::with_capacity(total_gens);
    let mut gen_count = 0;

    for (pair_idx, pair) in pairs.iter().enumerate() {
        let features_str = &pair_feature_strs[pair_idx];

        for (str_idx, &strength) in strengths.iter().enumerate() {
            let spec = &prebuilt_specs[pair_idx][str_idx];

            for seed_idx in 0..args.n_seeds {
                gen_count += 1;

                let generated = model.generate_with_clt_injection(
                    &pair.prompt_text,
                    args.max_tokens,
                    args.temperature,
                    &stop_tokens,
                    spec,
                )?;

                let gen_line = extract_generated_line(&generated, &pair.prompt_text);
                let ending = extract_ending_word(&gen_line);
                let target_hit = ending.eq_ignore_ascii_case(&pair.target_word);
                let rhyme_hit = words_rhyme(&ending, &pair.rhyme_group);

                results.push(ExperimentRecord {
                    mechanism: "clt".to_string(),
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
        clt_repo: args.clt_repo.clone(),
        mechanism: "clt".to_string(),
        n_pairs: pairs.len(),
        strengths: strengths.clone(),
        n_samples: args.n_seeds,
        results,
        summary,
    };

    write_output(&output, args.output.as_deref(), "run")?;
    Ok(())
}

/// Methods 1-2 feature identification pipeline (per-word).
///
/// Returns (pairs, prebuilt_specs, pair_feature_strs).
#[allow(clippy::too_many_arguments)]
fn mode_run_methods12(
    args: &Args,
    model: &PlipModel,
    clt: &mut CrossLayerTranscoder,
    corpus: &PoetryCorpus,
    strengths: &[f32],
    n_layers: usize,
    final_layer: usize,
    device: &Device,
) -> Result<RunPipelineResult> {
    let all_pairs = build_steering_pairs(corpus);

    // Collect unique target words
    let mut unique_targets: Vec<String> = all_pairs
        .iter()
        .map(|p| p.target_word.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if args.max_words > 0 && unique_targets.len() > args.max_words {
        eprintln!(
            "  --max-words {}: limiting from {} to {} unique targets",
            args.max_words,
            unique_targets.len(),
            args.max_words
        );
        unique_targets.truncate(args.max_words);
    }
    eprintln!(
        "Identifying features for {} unique target words...",
        unique_targets.len()
    );

    let use_m1 = args.activation_search;
    let use_m2 = args.decoder_projection || !args.activation_search;

    let mut target_features: HashMap<String, Vec<SelectedFeature>> = HashMap::new();

    if !use_m1 && use_m2 {
        // Batch scoring: load each decoder file ONCE for all words (26 reads, not N×26)
        eprintln!(
            "Using batch decoder projection ({} scoring)...",
            if args.cosine { "cosine" } else { "dot product" }
        );
        let embeddings: Vec<Tensor> = unique_targets
            .iter()
            .map(|target| {
                let target_with_space = format!(" {target}");
                let token_ids = model.encode(&target_with_space)?;
                let scoring_token_id = *token_ids.last().ok_or_else(|| {
                    anyhow::anyhow!("Target word \"{target}\" tokenizes to empty")
                })?;
                model.token_embedding(scoring_token_id)
            })
            .collect::<Result<_>>()?;

        let all_scores = clt.score_features_by_decoder_projection_batch(
            &embeddings,
            final_layer,
            20,
            args.cosine,
        )?;

        for (i, target) in unique_targets.iter().enumerate() {
            let selected: Vec<SelectedFeature> = all_scores[i]
                .iter()
                .take(args.n_features)
                .map(|(feature, score)| SelectedFeature {
                    feature: *feature,
                    target_layer: final_layer,
                    score: *score,
                    source_method: "decoder_projection".to_string(),
                })
                .collect();
            target_features.insert(target.clone(), selected);
        }
    } else {
        // Per-word scoring (needed when Method 1 is active)
        for (i, target) in unique_targets.iter().enumerate() {
            eprintln!(
                "  [{}/{}] Identifying features for \"{}\"...",
                i + 1,
                unique_targets.len(),
                target
            );
            let features = identify_features_for_word(
                model,
                clt,
                target,
                use_m1,
                use_m2,
                args.cosine,
                args.n_features,
                n_layers,
                device,
            )?;
            target_features.insert(target.clone(), features);
        }
    }
    eprintln!("[DIAG] Feature identification complete. Proceeding to cache...");

    // Print which layers the features span (diagnostic for OOM)
    {
        let mut layer_counts: BTreeMap<usize, usize> = BTreeMap::new();
        for feats in target_features.values() {
            for f in feats {
                *layer_counts.entry(f.feature.layer).or_default() += 1;
            }
        }
        eprintln!(
            "[DIAG] Features span {} unique source layers: {:?}",
            layer_counts.len(),
            layer_counts
        );
    }

    // Cache all steering vectors at once
    let all_feature_targets: Vec<(CltFeatureId, usize)> = target_features
        .values()
        .flat_map(|feats| feats.iter().map(|f| (f.feature, f.target_layer)))
        .collect();
    eprintln!(
        "[DIAG] About to cache {} steering vectors on GPU...",
        all_feature_targets.len()
    );
    clt.cache_steering_vectors(&all_feature_targets, device)?;
    eprintln!("Cached {} steering vectors", all_feature_targets.len());

    // Filter pairs to those whose target word has features (relevant when --max-words)
    let pairs: Vec<SteeringPair> = all_pairs
        .into_iter()
        .filter(|p| target_features.contains_key(&p.target_word))
        .collect();
    if args.max_words > 0 {
        eprintln!(
            "[DIAG] After --max-words filter: {} pairs remain",
            pairs.len()
        );
    }

    // Pre-build injection specs and feature strings per pair
    eprintln!("[DIAG] Pre-building injection specs...");
    let mut prebuilt_specs: Vec<Vec<CltInjectionSpec>> = Vec::with_capacity(pairs.len());
    let mut pair_feature_strs: Vec<Vec<String>> = Vec::with_capacity(pairs.len());

    for pair in &pairs {
        let features = target_features
            .get(&pair.target_word)
            .ok_or_else(|| anyhow::anyhow!("No features for target \"{}\"", pair.target_word))?;
        let feature_targets: Vec<(CltFeatureId, usize)> = features
            .iter()
            .map(|f| (f.feature, f.target_layer))
            .collect();
        let tokens = model.encode(&pair.prompt_text)?;
        let position = tokens.len().saturating_sub(1);

        let mut pair_specs = Vec::with_capacity(strengths.len());
        for &strength in strengths {
            let spec = if strength == 0.0 {
                CltInjectionSpec::new()
            } else {
                clt.prepare_injection(&feature_targets, position, strength)?
            };
            pair_specs.push(spec);
        }
        prebuilt_specs.push(pair_specs);
        pair_feature_strs.push(features.iter().map(|f| f.feature.to_string()).collect());
    }
    eprintln!(
        "Pre-built {} injection specs ({} pairs × {} strengths)",
        prebuilt_specs.len() * strengths.len(),
        pairs.len(),
        strengths.len()
    );

    Ok((pairs, prebuilt_specs, pair_feature_strs))
}

/// Method 3 feature identification pipeline (planning-site activation + decoder filtering).
///
/// Optimized batch path: 26 encoder loads + 26 decoder loads total,
/// regardless of the number of prompts or pairs.
///
/// Returns (pairs, prebuilt_specs, pair_feature_strs).
#[allow(clippy::too_many_arguments)]
fn mode_run_method3(
    args: &Args,
    model: &PlipModel,
    clt: &mut CrossLayerTranscoder,
    corpus: &PoetryCorpus,
    strengths: &[f32],
    n_layers: usize,
    final_layer: usize,
    device: &Device,
) -> Result<RunPipelineResult> {
    let multi_layer = args.method4;
    let tag = if multi_layer { "M4" } else { "M3" };
    if multi_layer {
        eprintln!("=== Method 4: Planning-site features + multi-layer clamping ===");
    } else {
        eprintln!("=== Method 3: Planning-site activation + decoder filtering ===");
    }

    let all_pairs = build_steering_pairs(corpus);

    // Apply --max-words filter: limit unique target words, then filter prompts
    let mut unique_targets: BTreeSet<String> =
        all_pairs.iter().map(|p| p.target_word.clone()).collect();
    if args.max_words > 0 && unique_targets.len() > args.max_words {
        eprintln!(
            "  --max-words {}: limiting from {} to {} unique targets",
            args.max_words,
            unique_targets.len(),
            args.max_words
        );
        let limited: BTreeSet<String> = unique_targets.into_iter().take(args.max_words).collect();
        unique_targets = limited;
    }
    let pairs: Vec<SteeringPair> = all_pairs
        .into_iter()
        .filter(|p| unique_targets.contains(&p.target_word))
        .collect();

    // Collect unique prompts from filtered pairs
    let unique_prompt_ids: Vec<String> = pairs
        .iter()
        .map(|p| p.prompt_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    eprintln!(
        "Method 3: {} pairs, {} unique prompts, {} unique targets",
        pairs.len(),
        unique_prompt_ids.len(),
        unique_targets.len()
    );

    // ── Phase 1: Forward passes for unique prompts ──────────────────────
    eprintln!(
        "[{tag}] Phase 1: Running forward passes for {} unique prompts...",
        unique_prompt_ids.len()
    );
    let mut prompt_activations: HashMap<String, ActivationCache> = HashMap::new();
    for (i, prompt_id) in unique_prompt_ids.iter().enumerate() {
        let prompt_text = pairs
            .iter()
            .find(|p| &p.prompt_id == prompt_id)
            .map(|p| p.prompt_text.as_str())
            .ok_or_else(|| anyhow::anyhow!("Prompt {prompt_id} not found in pairs"))?;
        let activations = model.get_activations(prompt_text)?;
        prompt_activations.insert(prompt_id.clone(), activations);
        if (i + 1) % 10 == 0 || i + 1 == unique_prompt_ids.len() {
            eprintln!(
                "  [{}/{}] forward passes complete",
                i + 1,
                unique_prompt_ids.len()
            );
        }
    }

    // ── Phase 2: Encode all layers (26 encoder loads total) ─────────────
    eprintln!(
        "[{tag}] Phase 2: Encoding {} prompts × {} layers...",
        unique_prompt_ids.len(),
        n_layers
    );
    let mut prompt_features: HashMap<String, Vec<(CltFeatureId, f32)>> = HashMap::new();
    for prompt_id in &unique_prompt_ids {
        prompt_features.insert(prompt_id.clone(), Vec::new());
    }

    for layer in 0..n_layers {
        clt.load_encoder(layer, device)?;
        for prompt_id in &unique_prompt_ids {
            let activations = &prompt_activations[prompt_id];
            if let Some(residual) = activations.get_layer(layer) {
                let sparse = clt.encode(residual, layer)?;
                let features = prompt_features.get_mut(prompt_id).unwrap();
                for (fid, act) in &sparse.features {
                    if *act > 0.0 {
                        features.push((*fid, *act));
                    }
                }
            }
        }
        if (layer + 1) % 5 == 0 || layer + 1 == n_layers {
            eprintln!("  [{}/{}] encoder layers complete", layer + 1, n_layers);
        }
    }

    // Report active feature counts
    let total_active: usize = prompt_features.values().map(Vec::len).sum();
    let avg_active = total_active as f64 / unique_prompt_ids.len().max(1) as f64;
    eprintln!(
        "[{tag}] Active features: {} total across {} prompts (avg {:.0}/prompt)",
        total_active,
        unique_prompt_ids.len(),
        avg_active
    );

    // ── Phase 3: Extract decoder vectors (26 decoder loads total) ────────
    let all_unique_features: Vec<CltFeatureId> = prompt_features
        .values()
        .flat_map(|v| v.iter().map(|(fid, _)| *fid))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    eprintln!(
        "[{tag}] Phase 3: Extracting decoder vectors for {} unique features at layer {}...",
        all_unique_features.len(),
        final_layer
    );
    let decoder_map = clt.extract_decoder_vectors(&all_unique_features, final_layer)?;

    // ── Phase 4: Score each pair by cosine similarity ────────────────────
    eprintln!(
        "[{tag}] Phase 4: Scoring {} pairs by cosine similarity...",
        pairs.len()
    );
    let mut pair_selected: Vec<Vec<SelectedFeature>> = Vec::with_capacity(pairs.len());

    for pair in &pairs {
        let active = &prompt_features[&pair.prompt_id];

        // Get target word embedding
        let target_with_space = format!(" {}", pair.target_word);
        let token_ids = model.encode(&target_with_space)?;
        let scoring_token_id = *token_ids.last().ok_or_else(|| {
            anyhow::anyhow!("Target word \"{}\" tokenizes to empty", pair.target_word)
        })?;
        let target_emb = model.token_embedding(scoring_token_id)?;
        let target_f32 = target_emb.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let target_norm = {
            let sq: f32 = target_f32.sqr()?.sum_all()?.to_scalar()?;
            sq.sqrt()
        };

        // Score each active feature
        let mut scored: Vec<(CltFeatureId, f32)> = Vec::new();
        for (fid, _act) in active {
            if let Some(dec_vec) = decoder_map.get(fid) {
                let dot: f32 = (dec_vec * &target_f32)?.sum_all()?.to_scalar()?;
                let dec_norm = {
                    let sq: f32 = dec_vec.sqr()?.sum_all()?.to_scalar()?;
                    sq.sqrt()
                };
                let cosine = if dec_norm > 1e-10 && target_norm > 1e-10 {
                    dot / (dec_norm * target_norm)
                } else {
                    0.0
                };
                if cosine.is_finite() {
                    scored.push((*fid, cosine));
                }
            }
        }

        // Sort by cosine descending, take top-K
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(args.n_features);

        let source_method = if multi_layer {
            "planning_site_clamped"
        } else {
            "planning_site"
        };
        let selected: Vec<SelectedFeature> = scored
            .into_iter()
            .map(|(feature, score)| SelectedFeature {
                feature,
                target_layer: final_layer,
                score,
                source_method: source_method.to_string(),
            })
            .collect();
        pair_selected.push(selected);
    }

    // Print layer distribution diagnostic
    {
        let mut layer_counts: BTreeMap<usize, usize> = BTreeMap::new();
        for feats in &pair_selected {
            for f in feats {
                *layer_counts.entry(f.feature.layer).or_default() += 1;
            }
        }
        eprintln!(
            "[{tag}] Features span {} unique source layers: {:?}",
            layer_counts.len(),
            layer_counts
        );
    }

    // ── Phase 5: Cache steering vectors + build injection specs ─────────
    let all_unique_features: Vec<CltFeatureId> = pair_selected
        .iter()
        .flat_map(|feats| feats.iter().map(|f| f.feature))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    if multi_layer {
        // Method 4: cache decoder vectors for ALL downstream layers per feature
        eprintln!(
            "[{tag}] Phase 5: Caching all downstream vectors for {} unique features on GPU...",
            all_unique_features.len()
        );
        clt.cache_steering_vectors_all_downstream(&all_unique_features, device)?;
    } else {
        // Method 3: cache only final-layer decoder vectors
        let all_feature_targets: Vec<(CltFeatureId, usize)> = all_unique_features
            .iter()
            .map(|f| (*f, final_layer))
            .collect();
        eprintln!(
            "[{tag}] Phase 5: Caching {} unique steering vectors on GPU...",
            all_feature_targets.len()
        );
        clt.cache_steering_vectors(&all_feature_targets, device)?;
    }

    let clt_n_layers = clt.config().n_layers;
    eprintln!("[{tag}] Pre-building injection specs...");
    let mut prebuilt_specs: Vec<Vec<CltInjectionSpec>> = Vec::with_capacity(pairs.len());
    let mut pair_feature_strs: Vec<Vec<String>> = Vec::with_capacity(pairs.len());

    for (pair_idx, pair) in pairs.iter().enumerate() {
        let features = &pair_selected[pair_idx];
        // Method 4: expand each feature to all downstream (feature, target_layer) pairs
        // Method 3: single (feature, final_layer) pair per feature
        let feature_targets: Vec<(CltFeatureId, usize)> = if multi_layer {
            features
                .iter()
                .flat_map(|f| (f.feature.layer..clt_n_layers).map(move |tl| (f.feature, tl)))
                .collect()
        } else {
            features
                .iter()
                .map(|f| (f.feature, f.target_layer))
                .collect()
        };
        let tokens = model.encode(&pair.prompt_text)?;
        let position = tokens.len().saturating_sub(1);

        let mut pair_specs = Vec::with_capacity(strengths.len());
        for &strength in strengths {
            let spec = if strength == 0.0 {
                CltInjectionSpec::new()
            } else {
                clt.prepare_injection(&feature_targets, position, strength)?
            };
            pair_specs.push(spec);
        }
        prebuilt_specs.push(pair_specs);
        pair_feature_strs.push(features.iter().map(|f| f.feature.to_string()).collect());
    }
    eprintln!(
        "[{tag}] Pre-built {} injection specs ({} pairs × {} strengths)",
        prebuilt_specs.len() * strengths.len(),
        pairs.len(),
        strengths.len()
    );

    Ok((pairs, prebuilt_specs, pair_feature_strs))
}

/// Method 5: Activation-based feature identification with contrastive scoring.
///
/// Instead of geometric proxies (decoder cosine similarity), this method:
/// 1. Runs each target word through the model to get per-layer activations
/// 2. Encodes activations through CLT encoders to find which features FIRE
/// 3. Uses contrastive scoring (target - mean others) to isolate word-specific features
/// 4. Injects using Method 4's multi-layer clamping
///
/// Returns (pairs, prebuilt_specs, pair_feature_strs).
#[allow(clippy::too_many_arguments)]
fn mode_run_method5(
    args: &Args,
    model: &PlipModel,
    clt: &mut CrossLayerTranscoder,
    corpus: &PoetryCorpus,
    strengths: &[f32],
    n_layers: usize,
    device: &Device,
) -> Result<RunPipelineResult> {
    eprintln!("=== Method 5: Activation-based contrastive feature identification ===");

    let all_pairs = build_steering_pairs(corpus);

    // Collect unique target words, apply --max-words filter
    let mut unique_targets: Vec<String> = all_pairs
        .iter()
        .map(|p| p.target_word.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if args.max_words > 0 && unique_targets.len() > args.max_words {
        eprintln!(
            "  --max-words {}: limiting from {} to {} unique targets",
            args.max_words,
            unique_targets.len(),
            args.max_words
        );
        unique_targets.truncate(args.max_words);
    }

    // Filter pairs to those whose target word survived the filter
    let unique_target_set: BTreeSet<String> = unique_targets.iter().cloned().collect();
    let pairs: Vec<SteeringPair> = all_pairs
        .into_iter()
        .filter(|p| unique_target_set.contains(&p.target_word))
        .collect();

    eprintln!(
        "[M5] {} pairs, {} unique targets",
        pairs.len(),
        unique_targets.len()
    );

    // ── Phase 1: Target word probe forward passes ────────────────────────
    eprintln!(
        "[M5] Phase 1: Running probe forward passes for {} unique target words...",
        unique_targets.len()
    );

    let mut target_activations: HashMap<String, ActivationCache> = HashMap::new();
    for (i, target) in unique_targets.iter().enumerate() {
        let probe_text = format!(" {target}");
        let activations = model.get_activations(&probe_text)?;
        target_activations.insert(target.clone(), activations);
        if (i + 1) % 10 == 0 || i + 1 == unique_targets.len() {
            eprintln!(
                "  [{}/{}] probe forward passes complete",
                i + 1,
                unique_targets.len()
            );
        }
    }

    // ── Phase 2: Encode probe activations through CLT (26 encoder loads) ─
    eprintln!(
        "[M5] Phase 2: Encoding {} targets × {} layers...",
        unique_targets.len(),
        n_layers
    );

    let mut target_features: HashMap<String, Vec<(CltFeatureId, f32)>> = HashMap::new();
    for target in &unique_targets {
        target_features.insert(target.clone(), Vec::new());
    }

    for layer in 0..n_layers {
        clt.load_encoder(layer, device)?;
        let mut layer_active_count = 0usize;
        for target in &unique_targets {
            let activations = &target_activations[target];
            if let Some(residual) = activations.get_layer(layer) {
                let sparse = clt.encode(residual, layer)?;
                let features = target_features.get_mut(target).unwrap();
                for (fid, act) in &sparse.features {
                    if *act > 0.0 {
                        features.push((*fid, *act));
                        layer_active_count += 1;
                    }
                }
            }
        }
        if (layer + 1) % 5 == 0 || layer + 1 == n_layers {
            eprintln!(
                "  [{}/{}] encoder layers complete ({} active this layer)",
                layer + 1,
                n_layers,
                layer_active_count
            );
        }
    }

    // Drop activations to free memory
    drop(target_activations);

    let total_active: usize = target_features.values().map(Vec::len).sum();
    let avg_active = total_active as f64 / unique_targets.len().max(1) as f64;
    eprintln!(
        "[M5] Active features: {} total across {} targets (avg {:.0}/target)",
        total_active,
        unique_targets.len(),
        avg_active
    );

    // ── Phase 3: Contrastive scoring + top-K selection ───────────────────
    eprintln!("[M5] Phase 3: Computing contrastive scores...");

    // Build inverted index: feature → { target_word → activation }
    let mut feature_index: HashMap<CltFeatureId, HashMap<String, f32>> = HashMap::new();
    for (target, features) in &target_features {
        for (fid, act) in features {
            feature_index
                .entry(*fid)
                .or_default()
                .insert(target.clone(), *act);
        }
    }

    let n_targets = unique_targets.len();
    let mut per_target_selected: HashMap<String, Vec<SelectedFeature>> = HashMap::new();
    let mut all_best_scores: Vec<f32> = Vec::new();

    for target in &unique_targets {
        let active_features = &target_features[target];

        let mut scored: Vec<(CltFeatureId, f32)> = Vec::new();

        for (fid, act_t) in active_features {
            let word_acts = &feature_index[fid];

            // Mean activation across all OTHER targets (inactive = 0.0)
            let sum_others: f32 = unique_targets
                .iter()
                .filter(|t| *t != target)
                .map(|t| word_acts.get(t).copied().unwrap_or(0.0))
                .sum();
            let mean_others = if n_targets > 1 {
                sum_others / (n_targets - 1) as f32
            } else {
                0.0
            };

            let contrastive_score = act_t - mean_others;
            scored.push((*fid, contrastive_score));
        }

        // Sort by contrastive score descending, take top-K
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(args.n_features);

        if let Some((_, best)) = scored.first() {
            all_best_scores.push(*best);
        }

        let selected: Vec<SelectedFeature> = scored
            .into_iter()
            .map(|(feature, score)| SelectedFeature {
                feature,
                target_layer: feature.layer, // placeholder; multi-layer expansion overrides
                score,
                source_method: "target_probe".to_string(),
            })
            .collect();

        per_target_selected.insert(target.clone(), selected);
    }

    // Diagnostic: contrastive score distribution
    all_best_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let best_cs = all_best_scores.first().copied().unwrap_or(0.0);
    let worst_cs = all_best_scores.last().copied().unwrap_or(0.0);
    let median_cs = if all_best_scores.is_empty() {
        0.0
    } else {
        all_best_scores[all_best_scores.len() / 2]
    };
    eprintln!(
        "[M5] Best contrastive scores per target: best={best_cs:.4}, median={median_cs:.4}, worst={worst_cs:.4} ({} targets)",
        all_best_scores.len()
    );

    // Layer distribution diagnostic
    {
        let mut layer_counts: BTreeMap<usize, usize> = BTreeMap::new();
        for feats in per_target_selected.values() {
            for f in feats {
                *layer_counts.entry(f.feature.layer).or_default() += 1;
            }
        }
        eprintln!(
            "[M5] Selected features span {} unique source layers: {:?}",
            layer_counts.len(),
            layer_counts
        );
    }

    // ── Phase 4: Cache steering vectors + build injection specs ──────────
    let all_unique_features: Vec<CltFeatureId> = per_target_selected
        .values()
        .flat_map(|feats| feats.iter().map(|f| f.feature))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    eprintln!(
        "[M5] Phase 4: Caching all downstream vectors for {} unique features on GPU...",
        all_unique_features.len()
    );
    clt.cache_steering_vectors_all_downstream(&all_unique_features, device)?;

    let clt_n_layers = clt.config().n_layers;
    eprintln!("[M5] Pre-building injection specs...");
    let mut prebuilt_specs: Vec<Vec<CltInjectionSpec>> = Vec::with_capacity(pairs.len());
    let mut pair_feature_strs: Vec<Vec<String>> = Vec::with_capacity(pairs.len());

    for pair in &pairs {
        let features = per_target_selected
            .get(&pair.target_word)
            .ok_or_else(|| anyhow::anyhow!("No features for target \"{}\"", pair.target_word))?;

        // Multi-layer clamping: expand each feature to all downstream layers
        let feature_targets: Vec<(CltFeatureId, usize)> = features
            .iter()
            .flat_map(|f| (f.feature.layer..clt_n_layers).map(move |tl| (f.feature, tl)))
            .collect();

        let tokens = model.encode(&pair.prompt_text)?;
        let position = tokens.len().saturating_sub(1);

        let mut pair_specs = Vec::with_capacity(strengths.len());
        for &strength in strengths {
            let spec = if strength == 0.0 {
                CltInjectionSpec::new()
            } else {
                clt.prepare_injection(&feature_targets, position, strength)?
            };
            pair_specs.push(spec);
        }
        prebuilt_specs.push(pair_specs);
        pair_feature_strs.push(features.iter().map(|f| f.feature.to_string()).collect());
    }

    eprintln!(
        "[M5] Pre-built {} injection specs ({} pairs × {} strengths)",
        prebuilt_specs.len() * strengths.len(),
        pairs.len(),
        strengths.len()
    );

    Ok((pairs, prebuilt_specs, pair_feature_strs))
}

/// Method 6: Causal activation patching for production feature identification.
///
/// Instead of proxies (cosine, contrastive), this method directly measures
/// each feature's causal effect on the target word's logit:
/// 1. Runs each corpus prompt to get per-layer activations + baseline logits
/// 2. Encodes activations through CLT to find which features are active
/// 3. Pre-filters top-N candidates by activation magnitude
/// 4. For each candidate: ablates it (injects negative contribution) and
///    measures the drop in the target word's logit
/// 5. Selects top-K features with the largest positive causal effect
/// 6. Injects using Method 4's multi-layer clamping
///
/// Returns (pairs, prebuilt_specs, pair_feature_strs).
#[allow(clippy::too_many_arguments)]
fn mode_run_method6(
    args: &Args,
    model: &PlipModel,
    clt: &mut CrossLayerTranscoder,
    corpus: &PoetryCorpus,
    strengths: &[f32],
    n_layers: usize,
    device: &Device,
) -> Result<RunPipelineResult> {
    eprintln!("=== Method 6: Causal activation patching ===");

    let all_pairs = build_steering_pairs(corpus);

    // Collect unique target words, apply --max-words filter
    let mut unique_targets: BTreeSet<String> =
        all_pairs.iter().map(|p| p.target_word.clone()).collect();
    if args.max_words > 0 && unique_targets.len() > args.max_words {
        eprintln!(
            "  --max-words {}: limiting from {} to {} unique targets",
            args.max_words,
            unique_targets.len(),
            args.max_words
        );
        let limited: BTreeSet<String> = unique_targets.into_iter().take(args.max_words).collect();
        unique_targets = limited;
    }

    // Filter pairs to those whose target word survived the filter
    let pairs: Vec<SteeringPair> = all_pairs
        .into_iter()
        .filter(|p| unique_targets.contains(&p.target_word))
        .collect();

    let unique_prompt_ids: Vec<String> = pairs
        .iter()
        .map(|p| p.prompt_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    eprintln!(
        "[M6] {} pairs, {} unique prompts, {} unique targets",
        pairs.len(),
        unique_prompt_ids.len(),
        unique_targets.len()
    );

    // ── Phase 1: Forward passes on corpus prompts ─────────────────────────
    eprintln!(
        "[M6] Phase 1: Running forward passes for {} unique prompts...",
        unique_prompt_ids.len()
    );
    let mut prompt_activations: HashMap<String, ActivationCache> = HashMap::new();
    for (i, prompt_id) in unique_prompt_ids.iter().enumerate() {
        let prompt_text = pairs
            .iter()
            .find(|p| &p.prompt_id == prompt_id)
            .map(|p| p.prompt_text.as_str())
            .ok_or_else(|| anyhow::anyhow!("Prompt {prompt_id} not found in pairs"))?;
        let activations = model.get_activations(prompt_text)?;
        prompt_activations.insert(prompt_id.clone(), activations);
        if (i + 1) % 10 == 0 || i + 1 == unique_prompt_ids.len() {
            eprintln!(
                "  [{}/{}] prompt forward passes complete",
                i + 1,
                unique_prompt_ids.len()
            );
        }
    }

    // ── Phase 2: Encode prompt activations through CLT ────────────────────
    eprintln!(
        "[M6] Phase 2: Encoding {} prompts × {} layers...",
        unique_prompt_ids.len(),
        n_layers
    );
    let mut prompt_features: HashMap<String, Vec<(CltFeatureId, f32)>> = HashMap::new();
    for prompt_id in &unique_prompt_ids {
        prompt_features.insert(prompt_id.clone(), Vec::new());
    }

    for layer in 0..n_layers {
        clt.load_encoder(layer, device)?;
        for prompt_id in &unique_prompt_ids {
            let activations = &prompt_activations[prompt_id];
            if let Some(residual) = activations.get_layer(layer) {
                let sparse = clt.encode(residual, layer)?;
                let features = prompt_features.get_mut(prompt_id).unwrap();
                for (fid, act) in &sparse.features {
                    if *act > 0.0 {
                        features.push((*fid, *act));
                    }
                }
            }
        }
        if (layer + 1) % 5 == 0 || layer + 1 == n_layers {
            eprintln!("  [{}/{}] encoder layers complete", layer + 1, n_layers);
        }
    }

    // Drop activations to free memory
    drop(prompt_activations);

    let total_active: usize = prompt_features.values().map(Vec::len).sum();
    let avg_active = total_active as f64 / unique_prompt_ids.len().max(1) as f64;
    eprintln!(
        "[M6] Active features: {} total across {} prompts (avg {:.0}/prompt)",
        total_active,
        unique_prompt_ids.len(),
        avg_active
    );

    // ── Phase 3: Pre-filter candidates by activation magnitude ────────────
    eprintln!(
        "[M6] Phase 3: Selecting top {} candidates per prompt by activation magnitude...",
        args.n_candidates
    );
    let mut prompt_candidates: HashMap<String, Vec<(CltFeatureId, f32)>> = HashMap::new();
    for (prompt_id, features) in &prompt_features {
        let mut sorted = features.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(args.n_candidates);
        if let (Some(first), Some(last)) = (sorted.first(), sorted.last()) {
            eprintln!(
                "  Prompt {}: {} candidates (activation range {:.4} – {:.4})",
                &prompt_id[..prompt_id.len().min(30)],
                sorted.len(),
                first.1,
                last.1
            );
        }
        prompt_candidates.insert(prompt_id.clone(), sorted);
    }

    // Candidate layer distribution
    {
        let mut layer_counts: BTreeMap<usize, usize> = BTreeMap::new();
        for candidates in prompt_candidates.values() {
            for (fid, _) in candidates {
                *layer_counts.entry(fid.layer).or_default() += 1;
            }
        }
        eprintln!(
            "[M6] Candidates span {} unique source layers: {:?}",
            layer_counts.len(),
            layer_counts
        );
    }

    // ── Phase 4: Causal verification via ablation ─────────────────────────
    eprintln!("[M6] Phase 4: Causal verification via ablation...");

    // Resolve target token IDs
    let mut target_token_ids: HashMap<String, u32> = HashMap::new();
    for target in &unique_targets {
        let target_with_space = format!(" {target}");
        let token_ids = model.encode(&target_with_space)?;
        let scoring_token = *token_ids
            .last()
            .ok_or_else(|| anyhow::anyhow!("Target \"{target}\" tokenizes to empty"))?;
        target_token_ids.insert(target.clone(), scoring_token);
    }

    // Build prompt → target words mapping
    let mut prompt_targets: HashMap<String, Vec<String>> = HashMap::new();
    for pair in &pairs {
        prompt_targets
            .entry(pair.prompt_id.clone())
            .or_default()
            .push(pair.target_word.clone());
    }
    // Deduplicate targets per prompt
    for targets in prompt_targets.values_mut() {
        targets.sort();
        targets.dedup();
    }

    // Causal effects: (prompt_id, target_word) → Vec<(CltFeatureId, causal_effect)>
    let mut causal_effects: HashMap<(String, String), Vec<(CltFeatureId, f32)>> = HashMap::new();

    let n_prompts = unique_prompt_ids.len();
    let clt_n_layers = clt.config().n_layers;

    for (prompt_idx, prompt_id) in unique_prompt_ids.iter().enumerate() {
        let prompt_text = pairs
            .iter()
            .find(|p| &p.prompt_id == prompt_id)
            .map(|p| p.prompt_text.as_str())
            .ok_or_else(|| anyhow::anyhow!("Prompt {prompt_id} not found"))?;

        let candidates = &prompt_candidates[prompt_id];
        if candidates.is_empty() {
            continue;
        }

        let tokens = model.encode(prompt_text)?;
        let position = tokens.len().saturating_sub(1);

        // Cache all candidate features' downstream decoder vectors
        let candidate_fids: Vec<CltFeatureId> = candidates.iter().map(|(fid, _)| *fid).collect();
        clt.cache_steering_vectors_all_downstream(&candidate_fids, device)?;

        // Get baseline logits
        let baseline_result = model.clt_logit_shift(prompt_text, &CltInjectionSpec::new())?;
        let baseline_logits_vec: Vec<f32> = baseline_result
            .baseline_logits
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1()?;

        // Extract baseline logits for each target word of this prompt
        let targets = &prompt_targets[prompt_id];
        let mut baseline_target_logits: HashMap<String, f32> = HashMap::new();
        for target in targets {
            let token_id = target_token_ids[target];
            baseline_target_logits.insert(target.clone(), baseline_logits_vec[token_id as usize]);
        }

        // Ablate each candidate and measure effect
        let mut best_effect = f32::NEG_INFINITY;
        let mut worst_effect = f32::INFINITY;

        for (c_idx, (fid, activation)) in candidates.iter().enumerate() {
            // Build ablation spec: inject -activation at all downstream layers
            let feature_targets: Vec<(CltFeatureId, usize)> =
                (fid.layer..clt_n_layers).map(|tl| (*fid, tl)).collect();
            let ablation_spec = clt.prepare_injection(&feature_targets, position, -activation)?;

            let ablation_result = model.clt_logit_shift(prompt_text, &ablation_spec)?;
            let ablated_logits_vec: Vec<f32> = ablation_result
                .injected_logits
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1()?;

            for target in targets {
                let token_id = target_token_ids[target] as usize;
                let causal_effect = baseline_target_logits[target] - ablated_logits_vec[token_id];
                causal_effects
                    .entry((prompt_id.clone(), target.clone()))
                    .or_default()
                    .push((*fid, causal_effect));

                if causal_effect > best_effect {
                    best_effect = causal_effect;
                }
                if causal_effect < worst_effect {
                    worst_effect = causal_effect;
                }
            }

            if (c_idx + 1) % 10 == 0 || c_idx + 1 == candidates.len() {
                eprintln!(
                    "  [{}/{}] prompt {}/{} ablated",
                    c_idx + 1,
                    candidates.len(),
                    prompt_idx + 1,
                    n_prompts
                );
            }
        }

        eprintln!(
            "  Prompt {}/{}: best_effect={:.4}, worst_effect={:.4}",
            prompt_idx + 1,
            n_prompts,
            best_effect,
            worst_effect
        );

        // Clear cache for next prompt
        clt.clear_steering_cache();
    }

    // ── Phase 5: Select production features per pair ──────────────────────
    eprintln!(
        "[M6] Phase 5: Selecting top {} production features per pair...",
        args.n_features
    );

    let mut pair_selected: Vec<Vec<SelectedFeature>> = Vec::with_capacity(pairs.len());
    let mut n_positive_pairs = 0usize;
    let mut all_best_effects: Vec<f32> = Vec::new();

    for pair in &pairs {
        let key = (pair.prompt_id.clone(), pair.target_word.clone());
        let effects = causal_effects.get(&key).cloned().unwrap_or_default();

        // Sort by causal effect descending, take top-K with positive effect
        let mut sorted = effects;
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Only keep features with positive causal effect
        let positive: Vec<(CltFeatureId, f32)> =
            sorted.into_iter().filter(|(_, e)| *e > 0.0).collect();

        if !positive.is_empty() {
            n_positive_pairs += 1;
            all_best_effects.push(positive[0].1);
        }

        let selected: Vec<SelectedFeature> = positive
            .into_iter()
            .take(args.n_features)
            .map(|(feature, score)| SelectedFeature {
                feature,
                target_layer: feature.layer,
                score,
                source_method: "causal_patching".to_string(),
            })
            .collect();
        pair_selected.push(selected);
    }

    all_best_effects.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let best_e = all_best_effects.first().copied().unwrap_or(0.0);
    let worst_e = all_best_effects.last().copied().unwrap_or(0.0);
    let median_e = if all_best_effects.is_empty() {
        0.0
    } else {
        all_best_effects[all_best_effects.len() / 2]
    };
    eprintln!(
        "[M6] Pairs with ≥1 positive-causal feature: {}/{} ({:.1}%)",
        n_positive_pairs,
        pairs.len(),
        100.0 * n_positive_pairs as f64 / pairs.len().max(1) as f64
    );
    eprintln!(
        "[M6] Best causal effects per pair: best={best_e:.4}, median={median_e:.4}, worst={worst_e:.4}"
    );

    // Layer distribution of selected features
    {
        let mut layer_counts: BTreeMap<usize, usize> = BTreeMap::new();
        for feats in &pair_selected {
            for f in feats {
                *layer_counts.entry(f.feature.layer).or_default() += 1;
            }
        }
        eprintln!(
            "[M6] Selected features span {} unique source layers: {:?}",
            layer_counts.len(),
            layer_counts
        );
    }

    // ── Phase 6: Cache steering vectors + build injection specs ───────────
    let all_unique_features: Vec<CltFeatureId> = pair_selected
        .iter()
        .flat_map(|feats| feats.iter().map(|f| f.feature))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    if all_unique_features.is_empty() {
        eprintln!("[M6] WARNING: No features with positive causal effect found!");
        eprintln!("[M6] All injection specs will be empty — steering will have no effect.");
    } else {
        eprintln!(
            "[M6] Phase 6: Caching all downstream vectors for {} unique features on GPU...",
            all_unique_features.len()
        );
        clt.cache_steering_vectors_all_downstream(&all_unique_features, device)?;
    }

    eprintln!("[M6] Pre-building injection specs...");
    let mut prebuilt_specs: Vec<Vec<CltInjectionSpec>> = Vec::with_capacity(pairs.len());
    let mut pair_feature_strs: Vec<Vec<String>> = Vec::with_capacity(pairs.len());

    for (pair_idx, pair) in pairs.iter().enumerate() {
        let features = &pair_selected[pair_idx];

        // Multi-layer clamping: expand each feature to all downstream layers
        let feature_targets: Vec<(CltFeatureId, usize)> = features
            .iter()
            .flat_map(|f| (f.feature.layer..clt_n_layers).map(move |tl| (f.feature, tl)))
            .collect();

        let tokens = model.encode(&pair.prompt_text)?;
        let position = tokens.len().saturating_sub(1);

        let mut pair_specs = Vec::with_capacity(strengths.len());
        for &strength in strengths {
            let spec = if strength == 0.0 || feature_targets.is_empty() {
                CltInjectionSpec::new()
            } else {
                clt.prepare_injection(&feature_targets, position, strength)?
            };
            pair_specs.push(spec);
        }
        prebuilt_specs.push(pair_specs);
        pair_feature_strs.push(features.iter().map(|f| f.feature.to_string()).collect());
    }

    eprintln!(
        "[M6] Pre-built {} injection specs ({} pairs × {} strengths)",
        prebuilt_specs.len() * strengths.len(),
        pairs.len(),
        strengths.len()
    );

    Ok((pairs, prebuilt_specs, pair_feature_strs))
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
        "Loaded {} results ({} pairs, {} strengths, {} samples)",
        run.results.len(),
        run.n_pairs,
        run.strengths.len(),
        run.n_samples
    );

    // By strength × condition
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
            "FAIL: best target_hit_rate = {:.1}% at strength {:.1} (< 10% — CLT steering insufficient for this model size)",
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

/// Combine Method 1 and Method 2 features, deduplicate, select top-N.
fn combine_features(
    m1: &[ScoredFeature],
    m2: &[ScoredFeature],
    n: usize,
    n_layers: usize,
) -> Vec<SelectedFeature> {
    let final_layer = n_layers - 1;

    // Merge with deduplication: if a feature appears in both, keep the higher score
    let mut best: HashMap<CltFeatureId, (f32, String)> = HashMap::new();
    for f in m1 {
        let entry = best.entry(f.feature).or_insert((f.score, f.method.clone()));
        if f.score > entry.0 {
            *entry = (f.score, f.method.clone());
        }
    }
    for f in m2 {
        let entry = best.entry(f.feature).or_insert((f.score, f.method.clone()));
        if f.score > entry.0 {
            *entry = (f.score, f.method.clone());
        }
    }

    let mut sorted: Vec<_> = best.into_iter().collect();
    sorted.sort_by(|a, b| {
        b.1 .0
            .partial_cmp(&a.1 .0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted.truncate(n);

    sorted
        .into_iter()
        .map(|(feature, (score, method))| SelectedFeature {
            feature,
            target_layer: final_layer,
            score,
            source_method: method,
        })
        .collect()
}

/// Identify features for a single target word (inline, for run mode).
#[allow(clippy::too_many_arguments)]
fn identify_features_for_word(
    model: &PlipModel,
    clt: &mut CrossLayerTranscoder,
    target_word: &str,
    use_m1: bool,
    use_m2: bool,
    cosine: bool,
    n_features: usize,
    n_layers: usize,
    device: &Device,
) -> Result<Vec<SelectedFeature>> {
    let target_with_space = format!(" {target_word}");
    let target_token_ids = model.encode(&target_with_space)?;
    let scoring_token_id = *target_token_ids
        .last()
        .ok_or_else(|| anyhow::anyhow!("Target word \"{target_word}\" tokenizes to empty"))?;

    let mut m1_features = Vec::new();
    let mut m2_features = Vec::new();

    if use_m1 {
        let probe_text = format!("The word is {target_word}");
        let activations = model.get_activations(&probe_text)?;
        let mut all: Vec<(CltFeatureId, f32)> = Vec::new();

        for layer in 0..n_layers {
            if let Some(residual) = activations.get_layer(layer) {
                clt.load_encoder(layer, device)?;
                let top = clt.top_k(residual, layer, 20)?;
                for (fid, act) in &top.features {
                    all.push((*fid, *act));
                }
            }
        }
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.dedup_by(|a, b| a.0 == b.0);
        all.truncate(20);

        m1_features = all
            .into_iter()
            .map(|(feature, score)| ScoredFeature {
                feature,
                score,
                method: "activation_search".to_string(),
            })
            .collect();
    }

    if use_m2 {
        let embed_vec = model.token_embedding(scoring_token_id)?;
        let final_layer = n_layers - 1;
        let scored =
            clt.score_features_by_decoder_projection(&embed_vec, final_layer, 20, cosine)?;

        m2_features = scored
            .into_iter()
            .map(|(feature, score)| ScoredFeature {
                feature,
                score,
                method: "decoder_projection".to_string(),
            })
            .collect();
    }

    Ok(combine_features(
        &m1_features,
        &m2_features,
        n_features,
        n_layers,
    ))
}

/// Build 120 steering pairs from the corpus (60 prompts × 2 conditions).
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

/// Select calibration prompts from Category C matching the target word's rhyme group.
fn select_calibration_prompts(
    corpus: &PoetryCorpus,
    target_word: &str,
    n: usize,
) -> Vec<PoetrySample> {
    let target_lower = target_word.to_lowercase();

    // Find which rhyme group this target belongs to
    let target_group = corpus
        .rhyming
        .iter()
        .find(|s| {
            s.ending_word.to_lowercase() == target_lower
                || s.rhyme_word
                    .as_deref()
                    .is_some_and(|rw| rw.to_lowercase() == target_lower)
        })
        .map(|s| s.rhyme_group.clone());

    let mut prompts: Vec<PoetrySample> = if let Some(ref group) = target_group {
        corpus
            .generation
            .iter()
            .filter(|s| &s.rhyme_group == group)
            .take(n)
            .cloned()
            .collect()
    } else {
        // Fall back to first N generation samples
        corpus.generation.iter().take(n).cloned().collect()
    };

    // If we got fewer than n from the target group, pad with others
    if prompts.len() < n {
        for s in &corpus.generation {
            if prompts.len() >= n {
                break;
            }
            if !prompts.iter().any(|p| p.id == s.id) {
                prompts.push(s.clone());
            }
        }
    }

    prompts
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
            "FAIL: best target_hit_rate = {:.1}% (< 10% — CLT steering insufficient)",
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
