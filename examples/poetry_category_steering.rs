//! Method 7: Semantic Category Steering with Causal Verification
//!
//! Steers Gemma 2 2B's poetry line endings toward semantic categories
//! (nature, emotion, light, motion) using CLT feature suppress + inject.
//!
//! Modes:
//!   - `probe-categories`: Analyze CLT dictionary for category-level structure
//!   - `explore-vocabulary`: Scan CLT features against full 256K vocabulary
//!     (bottom-up discovery of what the CLT actually encodes)
//!   - `baseline`: Generate unsteered completions (future)
//!   - `steer`: Suppress + inject category features (future)
//!
//! Usage:
//!   cargo run --release --example poetry_category_steering -- \
//!       --mode probe-categories \
//!       --output outputs/category_probe.json

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::BufRead;
use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use plip_rs::{ActivationCache, CltFeatureId, CrossLayerTranscoder, PlipModel};
use safetensors::tensor::SafeTensors;
use serde::{Deserialize, Serialize};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "poetry_category_steering")]
#[command(about = "Method 7: Semantic category CLT steering for poetry (Melometis)")]
struct Args {
    /// Mode: probe-categories, explore-vocabulary, find-rhyme-pairs, detect-planning
    #[arg(long)]
    mode: String,

    /// HuggingFace model ID
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// HuggingFace CLT repository
    #[arg(long, default_value = "mntss/clt-gemma-2-2b-426k")]
    clt_repo: String,

    /// Path to semantic categories JSON
    #[arg(long, default_value = "corpus/semantic_categories.json")]
    categories: PathBuf,

    /// Output file path
    #[arg(long)]
    output: Option<PathBuf>,

    /// Force CPU execution
    #[arg(long)]
    cpu: bool,

    /// Path to explore-vocabulary JSON (for find-rhyme-pairs mode)
    #[arg(long)]
    explore_json: Option<PathBuf>,

    /// Path to CMU Pronouncing Dictionary file
    #[arg(long, default_value = "corpus/cmudict.dict")]
    cmu_dict: PathBuf,

    /// Minimum cosine threshold for feature viability
    #[arg(long, default_value_t = 0.3)]
    min_cosine: f32,

    /// Path to rhyme-pairs JSON (for detect-planning mode)
    #[arg(long)]
    rhyme_pairs: Option<PathBuf>,

    /// Number of top features to report per category
    #[arg(long, default_value_t = 20)]
    top_k: usize,

    /// Feature sampling step (1 = every feature, 16 = every 16th)
    #[arg(long, default_value_t = 16)]
    sample_step: usize,

    /// Comma-separated layer indices to probe (default: all layers)
    #[arg(long)]
    layers: Option<String>,

    /// Comma-separated HTML tags to test as rhyme markers (e.g., "em,strong,b,i")
    #[arg(long)]
    tags: Option<String>,
}

// ── Category word list deserialization ───────────────────────────────────────

#[derive(Deserialize)]
struct CategoryFile {
    categories: BTreeMap<String, Vec<String>>,
}

// ── Output types ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ProbeOutput {
    model: String,
    clt_repo: String,
    n_layers: usize,
    n_features_per_layer: usize,
    sample_step: usize,
    categories: BTreeMap<String, CategoryTokenInfo>,
    /// Per-layer summary: how many features are dominated by each category
    layer_category_dominance: Vec<LayerDominance>,
    /// Top features per category (across all layers) ranked by category affinity
    top_features_per_category: BTreeMap<String, Vec<FeatureCategoryScore>>,
    /// Cross-category overlap: features that score highly for multiple categories
    cross_category_features: Vec<CrossCategoryFeature>,
}

#[derive(Serialize)]
struct CategoryTokenInfo {
    words: Vec<String>,
    token_ids: Vec<u32>,
    /// Words that are single-token (usable for logit measurement)
    single_token_words: Vec<String>,
    /// Words that are multi-token (excluded from logit measurement)
    multi_token_words: Vec<String>,
}

#[derive(Serialize)]
struct LayerDominance {
    layer: usize,
    /// Number of features whose strongest category affinity is for each category
    counts: BTreeMap<String, usize>,
    /// Number of features with no meaningful affinity (max score < threshold)
    neutral_count: usize,
}

#[derive(Serialize)]
struct FeatureCategoryScore {
    feature: CltFeatureId,
    /// Mean cosine similarity between this feature's decoder vector
    /// and unembedding directions of all single-token words in the category
    mean_cosine: f32,
    /// Max cosine similarity across category words
    max_cosine: f32,
    /// Number of category words with cosine > 0.1
    n_aligned_words: usize,
    /// Top 5 most-aligned words from this category
    top_words: Vec<(String, f32)>,
}

#[derive(Serialize)]
struct CrossCategoryFeature {
    feature: CltFeatureId,
    /// Category affinities (category_name -> mean_cosine), for categories with mean_cosine > 0.05
    affinities: BTreeMap<String, f32>,
}

// ── Explore-vocabulary output types ────────────────────────────────────────
// (also used by find-rhyme-pairs as input)

#[derive(Serialize, Deserialize)]
struct ExploreOutput {
    model: String,
    clt_repo: String,
    layers: Vec<usize>,
    sample_step: usize,
    vocab_size: usize,
    d_model: usize,
    n_features_scanned: usize,
    features: Vec<ExploreFeatureResult>,
}

#[derive(Serialize, Deserialize)]
struct ExploreFeatureResult {
    feature: CltFeatureId,
    max_cosine: f32,
    top_tokens: Vec<ExploreTokenScore>,
}

#[derive(Serialize, Deserialize)]
struct ExploreTokenScore {
    token_id: u32,
    text: String,
    cosine: f32,
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    match args.mode.as_str() {
        "probe-categories" => mode_probe_categories(&args),
        "explore-vocabulary" => mode_explore_vocabulary(&args),
        "find-rhyme-pairs" => mode_find_rhyme_pairs(&args),
        "detect-planning" => mode_detect_planning(&args),
        "position-sweep" => mode_position_sweep(&args),
        "verify-rhyming" => mode_verify_rhyming(&args),
        other => {
            anyhow::bail!("Unknown mode: {other}. Available: probe-categories, explore-vocabulary, find-rhyme-pairs, detect-planning, position-sweep, verify-rhyming")
        }
    }
}

// ── probe-categories mode ───────────────────────────────────────────────────

/// Analyze CLT dictionary for category-level structure.
///
/// For each CLT feature at each layer, computes the cosine similarity between
/// its decoder vector (projected to the last layer) and the unembedding
/// direction of each category word. This tells us whether the CLT has features
/// that naturally cluster around semantic categories.
fn mode_probe_categories(args: &Args) -> Result<()> {
    eprintln!("=== Method 7: CLT Category Probe ===");

    // 1. Load category word lists
    let cat_json = fs::read_to_string(&args.categories)
        .with_context(|| format!("Failed to read {}", args.categories.display()))?;
    let cat_file: CategoryFile = serde_json::from_str(&cat_json)?;

    let category_names: Vec<String> = cat_file.categories.keys().cloned().collect();
    eprintln!(
        "Loaded {} categories: {}",
        category_names.len(),
        category_names.join(", ")
    );

    // 2. Load model (needed for tokenizer + unembedding vectors)
    eprintln!("Loading model: {}", args.model);
    let force_cpu = if args.cpu { Some(true) } else { None };
    let model = PlipModel::from_pretrained_with_device(&args.model, force_cpu)?;

    // 3. Tokenize all category words and check single-token status
    //
    // Gemma 2's tokenizer (via model.encode) does NOT add BOS.
    // encode(" word") returns [▁word_token] (len=1) for single-token words,
    // or [▁word_part1, part2, ...] (len>1) for multi-token words.
    // We use the LAST token ID as the scoring token (same as Method 6).
    let mut category_info: BTreeMap<String, CategoryTokenInfo> = BTreeMap::new();
    let mut all_token_ids: Vec<(String, String, u32)> = Vec::new(); // (category, word, token_id)

    for (cat_name, words) in &cat_file.categories {
        let mut token_ids = Vec::new();
        let mut single_token = Vec::new();
        let mut multi_token = Vec::new();

        for word in words {
            let with_space = format!(" {word}");
            let ids = model.encode(&with_space)?;
            let scoring_token = *ids.last().unwrap();

            if ids.len() == 1 {
                single_token.push(word.clone());
            } else {
                multi_token.push(word.clone());
            }

            // Use the last token for scoring regardless of single/multi-token status.
            // For the probe we want ALL words, not just single-token ones.
            token_ids.push(scoring_token);
            all_token_ids.push((cat_name.clone(), word.clone(), scoring_token));
        }

        eprintln!(
            "  {cat_name}: {}/{} words are single-token, {} total scoring tokens",
            single_token.len(),
            words.len(),
            token_ids.len()
        );
        if !multi_token.is_empty() {
            eprintln!("    Multi-token: {}", multi_token.join(", "));
        }

        category_info.insert(
            cat_name.clone(),
            CategoryTokenInfo {
                words: words.clone(),
                token_ids: token_ids.clone(),
                single_token_words: single_token,
                multi_token_words: multi_token,
            },
        );
    }

    // 4. Get unembedding vectors for all category words (on CPU for efficiency)
    eprintln!(
        "Extracting unembedding vectors for {} tokens...",
        all_token_ids.len()
    );

    // Build a matrix of normalized unembedding vectors per category [n_words, d_model] on CPU
    let mut cat_unembed: BTreeMap<String, Vec<(String, Vec<f32>)>> = BTreeMap::new();
    for (cat, word, tid) in &all_token_ids {
        let emb = model.token_embedding(*tid)?;
        let emb_f32: Vec<f32> = emb.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let norm: f32 = emb_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normed: Vec<f32> = if norm > 1e-8 {
            emb_f32.iter().map(|x| x / norm).collect()
        } else {
            emb_f32
        };
        cat_unembed
            .entry(cat.clone())
            .or_default()
            .push((word.clone(), normed));
    }

    // 5. Open CLT for config, then probe decoder files directly
    eprintln!("Opening CLT: {}", args.clt_repo);
    let clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let n_layers = clt.config().n_layers;
    let n_features = clt.config().n_features_per_layer;
    let last_layer = n_layers - 1;
    let step = args.sample_step;

    // Parse --layers filter (if provided)
    let probe_layers: Vec<usize> = if let Some(ref layers_str) = args.layers {
        layers_str
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<usize>()
                    .with_context(|| format!("Invalid layer index: \"{s}\""))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        (0..n_layers).collect()
    };

    let n_probed = probe_layers.len();
    let features_per_layer = n_features / step;
    eprintln!(
        "CLT: {n_layers} layers x {n_features} features = {} total",
        n_layers * n_features
    );
    eprintln!(
        "Probing {n_probed} layers, step={step} ({features_per_layer} features/layer, {} total)",
        n_probed * features_per_layer
    );
    eprintln!("Probing decoder vectors at target layer {last_layer} (last layer)...");

    // Download decoder files via hf_hub (same approach as CLT internals)
    let api = Api::new()?;
    let repo = api.repo(Repo::new(args.clt_repo.clone(), RepoType::Model));

    let mut layer_dominance: Vec<LayerDominance> = Vec::with_capacity(n_probed);
    let mut global_scores: BTreeMap<(CltFeatureId, String), f32> = BTreeMap::new();
    let mut feature_word_scores: BTreeMap<(CltFeatureId, String), Vec<(String, f32)>> =
        BTreeMap::new();

    let d_model = clt.config().d_model;
    let probe_start = std::time::Instant::now();

    for (layer_idx, &source_layer) in probe_layers.iter().enumerate() {
        let n_target_layers = n_layers - source_layer;
        let target_offset = last_layer - source_layer;

        // Load decoder file ONCE for this source layer
        let dec_filename = format!("W_dec_{source_layer}.safetensors");
        eprintln!(
            "  [{}/{}] Layer {source_layer}: loading {dec_filename}...",
            layer_idx + 1,
            n_probed,
        );
        let dec_path = repo
            .get(&dec_filename)
            .with_context(|| format!("Failed to download {dec_filename}"))?;
        let dec_data =
            std::fs::read(&dec_path).with_context(|| format!("Failed to read {dec_filename}"))?;
        let dec_size_mb = dec_data.len() / (1024 * 1024);

        let dec_st = SafeTensors::deserialize(&dec_data)
            .with_context(|| format!("Failed to deserialize {dec_filename}"))?;
        let dec_name = format!("W_dec_{source_layer}");
        let dec_view = dec_st.tensor(&dec_name)?;
        // w_dec shape: [n_features, n_target_layers, d_model], BF16
        // Access raw bytes directly -- no candle Tensor allocation needed.
        // Each BF16 value is 2 bytes. Row-major layout:
        //   byte_offset(feat, tgt, dim) = ((feat * n_target_layers + tgt) * d_model + dim) * 2
        let raw_bytes = dec_view.data();
        let row_stride = n_target_layers * d_model * 2; // bytes per feature

        eprintln!("    Loaded {dec_size_mb} MB, scoring sampled features (zero-copy)...");

        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for cat in &category_names {
            counts.insert(cat.clone(), 0);
        }
        let mut neutral = 0usize;
        let threshold = 0.05_f32;
        let mut sampled = 0usize;

        for feat_idx in (0..n_features).step_by(step) {
            let fid = CltFeatureId {
                layer: source_layer,
                index: feat_idx,
            };

            // Extract decoder vector directly from raw BF16 bytes -> f32
            let byte_start = feat_idx * row_stride + target_offset * d_model * 2;
            let dec_vec = bf16_slice_to_f32(&raw_bytes[byte_start..byte_start + d_model * 2]);

            let dec_norm: f32 = dec_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if dec_norm < 1e-8 {
                neutral += 1;
                sampled += 1;
                continue;
            }

            // Normalize decoder vector
            let inv_norm = 1.0 / dec_norm;
            let dec_normed: Vec<f32> = dec_vec.iter().map(|x| x * inv_norm).collect();

            // Compute cosine similarity with each category's words (CPU dot products)
            let mut best_cat = String::new();
            let mut best_mean = 0.0_f32;

            for (cat_name, words_vecs) in &cat_unembed {
                let mut cosines: Vec<f32> = Vec::with_capacity(words_vecs.len());

                for (_, unembed_normed) in words_vecs {
                    let cos: f32 = dec_normed
                        .iter()
                        .zip(unembed_normed.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    cosines.push(cos);
                }

                let mean_cos = cosines.iter().sum::<f32>() / cosines.len().max(1) as f32;

                global_scores.insert((fid, cat_name.clone()), mean_cos);

                let mut word_scores: Vec<(String, f32)> = words_vecs
                    .iter()
                    .zip(cosines.iter())
                    .map(|((w, _), &c)| (w.clone(), c))
                    .collect();
                word_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                feature_word_scores.insert((fid, cat_name.clone()), word_scores);

                if mean_cos > best_mean {
                    best_mean = mean_cos;
                    best_cat.clone_from(cat_name);
                }
            }

            if best_mean >= threshold && !best_cat.is_empty() {
                *counts.get_mut(&best_cat).unwrap() += 1;
            } else {
                neutral += 1;
            }

            sampled += 1;
        }

        let elapsed = probe_start.elapsed().as_secs();
        let cat_count: usize = counts.values().sum();
        eprintln!(
            "    {sampled} features: {cat_count} categorized, {neutral} neutral  [{elapsed}s elapsed]"
        );

        layer_dominance.push(LayerDominance {
            layer: source_layer,
            counts,
            neutral_count: neutral,
        });
    }

    let total_elapsed = probe_start.elapsed().as_secs_f32();
    eprintln!("Probing complete in {total_elapsed:.1}s");

    // 6. Compute top features per category
    eprintln!("Computing top features per category...");

    let mut top_features: BTreeMap<String, Vec<FeatureCategoryScore>> = BTreeMap::new();

    for cat_name in &category_names {
        let mut scored: Vec<(CltFeatureId, f32)> = global_scores
            .iter()
            .filter(|((_, c), _)| c == cat_name)
            .map(|((fid, _), &score)| (*fid, score))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(args.top_k);

        let mut entries = Vec::new();
        for (fid, mean_cos) in &scored {
            let word_scores = feature_word_scores
                .get(&(*fid, cat_name.clone()))
                .cloned()
                .unwrap_or_default();

            let max_cos = word_scores.first().map_or(0.0, |(_, c)| *c);
            let n_aligned = word_scores.iter().filter(|(_, c)| *c > 0.1).count();
            let top_words: Vec<(String, f32)> = word_scores.into_iter().take(5).collect();

            entries.push(FeatureCategoryScore {
                feature: *fid,
                mean_cosine: *mean_cos,
                max_cosine: max_cos,
                n_aligned_words: n_aligned,
                top_words,
            });
        }

        if let Some(best) = entries.first() {
            eprintln!(
                "  {cat_name}: best feature {} (mean_cos={:.4}, max_cos={:.4}, aligned={})",
                best.feature, best.mean_cosine, best.max_cosine, best.n_aligned_words
            );
        }

        top_features.insert(cat_name.clone(), entries);
    }

    // 7. Find cross-category features (features with affinity to multiple categories)
    eprintln!("Finding cross-category features...");

    let all_features: Vec<CltFeatureId> = global_scores
        .keys()
        .map(|(fid, _)| *fid)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    let mut cross_features: Vec<CrossCategoryFeature> = Vec::new();
    let cross_threshold = 0.05_f32;

    for fid in &all_features {
        let mut affinities: BTreeMap<String, f32> = BTreeMap::new();
        for cat_name in &category_names {
            if let Some(&score) = global_scores.get(&(*fid, cat_name.clone())) {
                if score > cross_threshold {
                    affinities.insert(cat_name.clone(), score);
                }
            }
        }
        if affinities.len() >= 2 {
            cross_features.push(CrossCategoryFeature {
                feature: *fid,
                affinities,
            });
        }
    }

    cross_features.sort_by(|a, b| {
        let a_max = a.affinities.values().copied().fold(0.0_f32, f32::max);
        let b_max = b.affinities.values().copied().fold(0.0_f32, f32::max);
        b_max
            .partial_cmp(&a_max)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    eprintln!(
        "Found {} features with cross-category affinity (>= 2 categories above {cross_threshold})",
        cross_features.len()
    );

    cross_features.truncate(50);

    // 8. Build output
    let output = ProbeOutput {
        model: args.model.clone(),
        clt_repo: args.clt_repo.clone(),
        n_layers,
        n_features_per_layer: n_features,
        sample_step: step,
        categories: category_info,
        layer_category_dominance: layer_dominance,
        top_features_per_category: top_features,
        cross_category_features: cross_features,
    };

    // 9. Write output
    let json = serde_json::to_string_pretty(&output)?;
    if let Some(ref path) = args.output {
        fs::write(path, &json)
            .with_context(|| format!("Failed to write output to {}", path.display()))?;
        eprintln!("Output written to {}", path.display());
    } else {
        println!("{json}");
    }

    // 10. Print summary
    eprintln!("\n=== Summary ===");
    for cat_name in &category_names {
        let info = &output.categories[cat_name];
        eprintln!(
            "  {cat_name}: {} single-token, {} multi-token, {} total scoring tokens",
            info.single_token_words.len(),
            info.multi_token_words.len(),
            info.token_ids.len()
        );
    }

    eprintln!("\nLayer dominance (sampled, step={step}):");
    for ld in &output.layer_category_dominance {
        let total: usize = ld.counts.values().sum::<usize>() + ld.neutral_count;
        let dominant = ld
            .counts
            .iter()
            .max_by_key(|(_, v)| *v)
            .map(|(k, v)| format!("{k}={v}"))
            .unwrap_or_default();
        eprintln!(
            "  L{:02}: total={total}, dominant={dominant}, neutral={}",
            ld.layer, ld.neutral_count
        );
    }

    eprintln!("\nTop feature per category:");
    for (cat, feats) in &output.top_features_per_category {
        if let Some(f) = feats.first() {
            let words: String = f
                .top_words
                .iter()
                .map(|(w, c)| format!("{w}:{c:.3}"))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "  {cat}: {} mean={:.4} max={:.4} aligned={} words=[{words}]",
                f.feature, f.mean_cosine, f.max_cosine, f.n_aligned_words
            );
        }
    }

    Ok(())
}

/// Convert a slice of raw BF16 bytes (little-endian) to a Vec<f32>.
///
/// Each BF16 value is 2 bytes. BF16 → F32 conversion: shift left by 16 bits
/// (BF16 is the top 16 bits of F32).
fn bf16_slice_to_f32(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bf16_bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        let f32_bits = u32::from(bf16_bits) << 16;
        out.push(f32::from_bits(f32_bits));
    }
    out
}

/// Convert a safetensors `TensorView` to `Vec<f32>`, handling BF16/F16/F32 dtypes.
fn safetensors_to_f32(v: &safetensors::tensor::TensorView) -> Result<Vec<f32>> {
    let bytes = v.data();
    match v.dtype() {
        safetensors::Dtype::BF16 => Ok(bf16_slice_to_f32(bytes)),
        safetensors::Dtype::F16 => {
            let n = bytes.len() / 2;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                let sign = u32::from((bits >> 15) & 1);
                let exp = u32::from((bits >> 10) & 0x1f);
                let frac = u32::from(bits & 0x3ff);
                let f = if exp == 0 {
                    f32::from_bits(sign << 31)
                        + if frac != 0 {
                            #[allow(clippy::cast_precision_loss)]
                            let v = (frac as f32 / 1024.0) * 2.0_f32.powi(-14);
                            if sign == 1 {
                                -v
                            } else {
                                v
                            }
                        } else {
                            0.0
                        }
                } else if exp == 31 {
                    f32::from_bits((sign << 31) | 0x7F80_0000 | (frac << 13))
                } else {
                    let f32_exp = exp - 15 + 127;
                    f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
                };
                out.push(f);
            }
            Ok(out)
        }
        safetensors::Dtype::F32 => {
            let n = bytes.len() / 4;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(f32::from_le_bytes([
                    bytes[i * 4],
                    bytes[i * 4 + 1],
                    bytes[i * 4 + 2],
                    bytes[i * 4 + 3],
                ]));
            }
            Ok(out)
        }
        other => anyhow::bail!("Unsupported dtype: {other:?}"),
    }
}

/// Extract a named tensor from `SafeTensors`, convert to f32, validate shape.
fn extract_embedding(st: &SafeTensors, name: &str) -> Result<(Vec<f32>, usize, usize)> {
    let v = st.tensor(name)?;
    let (vocab, dm) = (v.shape()[0], v.shape()[1]);
    eprintln!(
        "  Tensor: [{vocab}, {dm}], dtype={:?}, data_bytes={}",
        v.dtype(),
        v.data().len()
    );
    let data = safetensors_to_f32(&v)?;
    anyhow::ensure!(
        data.len() == vocab * dm,
        "Embedding data length mismatch: {} vs {vocab}×{dm}={}",
        data.len(),
        vocab * dm
    );
    Ok((data, vocab, dm))
}

/// Load the full embedding matrix from model safetensors (handles sharded models).
///
/// Returns `(flat_f32_data, vocab_size, d_model)`.
fn load_embedding_matrix(model_id: &str) -> Result<(Vec<f32>, usize, usize)> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
    let tensor_name = "model.embed_tokens.weight";

    // Try single-file model
    if let Ok(path) = repo.get("model.safetensors") {
        let data = fs::read(&path)?;
        let st = SafeTensors::deserialize(&data)?;
        if st.tensor(tensor_name).is_ok() {
            return extract_embedding(&st, tensor_name);
        }
    }

    // Sharded model: consult index
    let idx_path = repo
        .get("model.safetensors.index.json")
        .context("No model.safetensors or index.json found")?;
    let idx: serde_json::Value = serde_json::from_str(&fs::read_to_string(&idx_path)?)?;
    let shard = idx["weight_map"][tensor_name]
        .as_str()
        .context("embed_tokens.weight not found in weight_map")?;

    eprintln!("  Embedding is in shard: {shard}");
    let shard_path = repo.get(shard)?;
    let data = fs::read(&shard_path)?;
    let st = SafeTensors::deserialize(&data)?;
    extract_embedding(&st, tensor_name)
}

// ── explore-vocabulary mode ────────────────────────────────────────────────

/// Scan CLT decoder vectors against the FULL 256K vocabulary to discover
/// what the CLT actually encodes, without imposing predefined categories.
///
/// Uses chunked candle matmul on GPU for fast cosine computation:
///   `dec_chunk [chunk, d_model] × embed_t [d_model, vocab] → [chunk, vocab]`
fn mode_explore_vocabulary(args: &Args) -> Result<()> {
    eprintln!("=== Explore: What does the CLT encode? ===\n");

    // 0. Device selection (GPU by default)
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    };
    eprintln!("Device: {device:?}");

    // 1. Load tokenizer (for decoding token IDs back to text)
    let api = Api::new()?;
    let model_repo = api.repo(Repo::new(args.model.clone(), RepoType::Model));
    eprintln!("Loading tokenizer...");
    let tok_path = model_repo.get("tokenizer.json")?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    // 2. Load full embedding matrix, normalize + transpose on device
    eprintln!("Loading embedding matrix from model safetensors...");
    let (embed_flat, vocab_size, d_model) = load_embedding_matrix(&args.model)?;
    eprintln!("  [{vocab_size} x {d_model}], transferring to {device:?}...");
    let embed = Tensor::from_vec(embed_flat, (vocab_size, d_model), &device)?;
    let norms = embed.sqr()?.sum_keepdim(1)?.affine(1.0, 1e-16)?.sqrt()?;
    let embed_normed = embed.broadcast_div(&norms)?;
    drop(embed);
    let embed_t = embed_normed.t()?.contiguous()?; // [d_model, vocab_size]
    drop(embed_normed);
    eprintln!("  Embedding matrix ready on {device:?} (normalized, transposed)");

    // 3. CLT config + layers
    let clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let n_layers = clt.config().n_layers;
    let n_feat = clt.config().n_features_per_layer;
    let step = args.sample_step;
    let clt_repo = api.repo(Repo::new(args.clt_repo.clone(), RepoType::Model));

    let layers: Vec<usize> = if let Some(ref s) = args.layers {
        s.split(',')
            .map(|x| {
                x.trim()
                    .parse::<usize>()
                    .with_context(|| format!("Invalid layer: \"{x}\""))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![n_layers - 1] // default: last layer only
    };

    // 4. Token decode cache
    let mut tok_cache: HashMap<u32, String> = HashMap::new();

    // 5. Scan each layer
    let mut all_results: Vec<ExploreFeatureResult> = Vec::new();
    let t0 = std::time::Instant::now();

    for &layer in &layers {
        let n_tgt = n_layers - layer;
        let tgt_off = (n_layers - 1) - layer;
        let fname = format!("W_dec_{layer}.safetensors");
        eprintln!("Layer {layer}: loading {fname}...");
        let path = clt_repo.get(&fname)?;
        let data = fs::read(&path)?;
        let st = SafeTensors::deserialize(&data)?;
        let view = st.tensor(&format!("W_dec_{layer}"))?;
        let raw = view.data();
        let row_stride = n_tgt * d_model * 2;

        // Collect sampled feature indices
        let feat_indices: Vec<usize> = (0..n_feat).step_by(step).collect();
        let n_sampled = feat_indices.len();
        eprintln!("  Scanning {n_sampled} features vs {vocab_size} tokens (chunked GPU matmul)...");

        // Process in chunks for efficient GPU matmul (4096 = ~36 MB decoder + ~4 GB result)
        let chunk_size = 4096;
        let layer_start = std::time::Instant::now();

        for (chunk_idx, chunk_feats) in feat_indices.chunks(chunk_size).enumerate() {
            // Build decoder matrix for this chunk: [chunk_len, d_model] (CPU extraction)
            let chunk_len = chunk_feats.len();
            let mut dec_flat: Vec<f32> = Vec::with_capacity(chunk_len * d_model);

            for &fi in chunk_feats {
                let off = fi * row_stride + tgt_off * d_model * 2;
                let dv = bf16_slice_to_f32(&raw[off..off + d_model * 2]);
                dec_flat.extend_from_slice(&dv);
            }

            // Transfer to device, normalize, matmul
            let dec = Tensor::from_vec(dec_flat, (chunk_len, d_model), &device)?;
            let dec_norms = dec.sqr()?.sum_keepdim(1)?.affine(1.0, 1e-16)?.sqrt()?;
            let dec_normed = dec.broadcast_div(&dec_norms)?;
            let cosines = dec_normed.matmul(&embed_t)?; // [chunk_len, vocab_size]
            let cos_cpu: Vec<f32> = cosines.flatten_all()?.to_vec1()?;

            // Extract top-20 per feature (on CPU)
            for (local_idx, &fi) in chunk_feats.iter().enumerate() {
                let row_start = local_idx * vocab_size;
                let row = &cos_cpu[row_start..row_start + vocab_size];
                let mut top20: Vec<(u32, f32)> = Vec::with_capacity(20);
                let mut min_cos = f32::NEG_INFINITY;
                let mut min_i = 0usize;

                for (tid, &cos) in row.iter().enumerate() {
                    if top20.len() < 20 {
                        #[allow(clippy::cast_possible_truncation)]
                        top20.push((tid as u32, cos));
                        if top20.len() == 20 {
                            for (i, &(_, c)) in top20.iter().enumerate() {
                                if i == 0 || c < min_cos {
                                    min_cos = c;
                                    min_i = i;
                                }
                            }
                        }
                    } else if cos > min_cos {
                        #[allow(clippy::cast_possible_truncation)]
                        let tid_u32 = tid as u32;
                        top20[min_i] = (tid_u32, cos);
                        min_cos = top20[0].1;
                        min_i = 0;
                        for (i, &(_, c)) in top20.iter().enumerate() {
                            if c < min_cos {
                                min_cos = c;
                                min_i = i;
                            }
                        }
                    }
                }

                top20.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let max_cos = top20.first().map_or(0.0, |x| x.1);

                let top_tokens: Vec<ExploreTokenScore> = top20
                    .iter()
                    .map(|(tid, c)| {
                        let text = tok_cache
                            .entry(*tid)
                            .or_insert_with(|| {
                                tokenizer
                                    .decode(&[*tid], true)
                                    .unwrap_or_else(|_| format!("<{tid}>"))
                            })
                            .clone();
                        ExploreTokenScore {
                            token_id: *tid,
                            text,
                            cosine: *c,
                        }
                    })
                    .collect();

                all_results.push(ExploreFeatureResult {
                    feature: CltFeatureId { layer, index: fi },
                    max_cosine: max_cos,
                    top_tokens,
                });
            }

            let done = (chunk_idx + 1) * chunk_size;
            if done % 4096 == 0 || done >= n_sampled {
                eprintln!(
                    "  {}/{n_sampled}  [{}s]",
                    done.min(n_sampled),
                    layer_start.elapsed().as_secs()
                );
            }
        }

        eprintln!(
            "  Layer {layer} done in {:.1}s",
            layer_start.elapsed().as_secs_f32()
        );
    }

    // 6. Sort by max cosine (most word-specific first)
    all_results.sort_by(|a, b| {
        b.max_cosine
            .partial_cmp(&a.max_cosine)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_n = args.top_k.min(all_results.len());
    eprintln!(
        "\n=== Top {top_n} most word-specific CLT features (of {} scanned) ===",
        all_results.len()
    );
    for (i, r) in all_results.iter().take(top_n).enumerate() {
        let tokens: String = r
            .top_tokens
            .iter()
            .take(10)
            .map(|t| format!("{}({:.3})", t.text.trim(), t.cosine))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "  #{:3}  {}  max={:.4}  [{tokens}]",
            i + 1,
            r.feature,
            r.max_cosine
        );
    }

    // 7. JSON output
    let total_elapsed = t0.elapsed().as_secs_f32();
    eprintln!("\nTotal time: {total_elapsed:.1}s");

    let output = ExploreOutput {
        model: args.model.clone(),
        clt_repo: args.clt_repo.clone(),
        layers: layers.clone(),
        sample_step: step,
        vocab_size,
        d_model,
        n_features_scanned: all_results.len(),
        features: all_results,
    };

    let json = serde_json::to_string_pretty(&output)?;
    if let Some(ref p) = args.output {
        fs::write(p, &json)?;
        eprintln!("Written to {}", p.display());
    } else {
        println!("{json}");
    }

    Ok(())
}

// ── find-rhyme-pairs output types ─────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct RhymePairsOutput {
    explore_json: String,
    cmu_dict: String,
    min_cosine: f32,
    n_features_loaded: usize,
    n_clean_english: usize,
    n_in_cmu_dict: usize,
    n_rhyme_groups: usize,
    n_rhyming_words: usize,
    rhyme_groups: Vec<RhymeGroup>,
}

#[derive(Serialize, Deserialize)]
struct RhymeGroup {
    /// Rhyme ending phonemes (e.g., "IY1" for words ending in -ee sound)
    rhyme_ending: String,
    /// Words in this group, sorted by cosine descending
    words: Vec<RhymeWord>,
}

#[derive(Serialize, Deserialize)]
struct RhymeWord {
    word: String,
    feature: CltFeatureId,
    cosine: f32,
    /// CMU phonemes for this word
    phonemes: String,
}

// ── find-rhyme-pairs mode ─────────────────────────────────────────────────

/// Find all CLT features that map to clean English words with rhyming partners.
///
/// Loads pre-computed explore-vocabulary JSON, cross-references with CMU
/// Pronouncing Dictionary, and discovers rhyming pairs where BOTH words
/// have dedicated CLT features (cosine > threshold).
fn mode_find_rhyme_pairs(args: &Args) -> Result<()> {
    eprintln!("=== Find Rhyme Pairs in CLT Dictionary ===\n");

    // 1. Load explore-vocabulary JSON
    let explore_path = args
        .explore_json
        .as_ref()
        .context("--explore-json is required for find-rhyme-pairs mode")?;
    eprintln!("Loading explore data: {}", explore_path.display());
    let explore_text = fs::read_to_string(explore_path)?;
    let explore: ExploreOutput = serde_json::from_str(&explore_text)?;
    eprintln!(
        "  {} features loaded (layers: {:?})",
        explore.features.len(),
        explore.layers
    );

    // 2. Load CMU Pronouncing Dictionary
    eprintln!("Loading CMU dict: {}", args.cmu_dict.display());
    let cmu = load_cmu_dict(&args.cmu_dict)?;
    eprintln!("  {} entries loaded", cmu.len());

    // 3. Extract clean English features above cosine threshold
    //    For each feature, take the top-1 token, strip whitespace, lowercase.
    //    Filter: ASCII alphabetic only, length >= 2, cosine >= min_cosine, in CMU dict.
    let min_cos = args.min_cosine;
    eprintln!("Filtering features: cosine >= {min_cos}, ASCII words, in CMU dict...");

    // Track best feature per word (a word may appear as top-1 of multiple features)
    let mut word_best: HashMap<String, (CltFeatureId, f32)> = HashMap::new();

    for feat in &explore.features {
        if feat.max_cosine < min_cos {
            continue;
        }
        let Some(top) = feat.top_tokens.first() else {
            continue;
        };

        let word = top.text.trim().to_lowercase();
        if word.len() < 2 || !word.chars().all(|c| c.is_ascii_alphabetic()) {
            continue;
        }

        // Keep the feature with highest cosine for this word
        let entry = word_best.entry(word).or_insert((feat.feature, top.cosine));
        if top.cosine > entry.1 {
            *entry = (feat.feature, top.cosine);
        }
    }

    let n_clean = word_best.len();
    eprintln!("  {n_clean} unique clean English words above threshold");

    // 4. Cross-reference with CMU dict and extract rhyme endings
    let mut rhyme_map: BTreeMap<String, Vec<(String, CltFeatureId, f32, String)>> = BTreeMap::new();
    let mut n_in_cmu = 0usize;

    for (word, &(fid, cos)) in &word_best {
        if let Some(phonemes) = cmu.get(word.as_str()) {
            n_in_cmu += 1;
            let ending = rhyme_ending(phonemes);
            if !ending.is_empty() {
                let phoneme_str = phonemes.join(" ");
                rhyme_map
                    .entry(ending)
                    .or_default()
                    .push((word.clone(), fid, cos, phoneme_str));
            }
        }
    }

    eprintln!("  {n_in_cmu} words found in CMU dict");

    // 5. Filter to groups with 2+ members (actual rhyme pairs)
    let mut groups: Vec<RhymeGroup> = Vec::new();
    let mut total_rhyming = 0usize;

    for (ending, mut members) in rhyme_map {
        if members.len() < 2 {
            continue;
        }
        // Sort by cosine descending
        members.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        total_rhyming += members.len();

        let words: Vec<RhymeWord> = members
            .into_iter()
            .map(|(w, fid, cos, ph)| RhymeWord {
                word: w,
                feature: fid,
                cosine: cos,
                phonemes: ph,
            })
            .collect();

        groups.push(RhymeGroup {
            rhyme_ending: ending,
            words,
        });
    }

    // Sort groups: biggest groups first, then by best cosine
    groups.sort_by(|a, b| {
        let size_cmp = b.words.len().cmp(&a.words.len());
        if size_cmp != std::cmp::Ordering::Equal {
            return size_cmp;
        }
        let a_best = a.words.first().map_or(0.0, |w| w.cosine);
        let b_best = b.words.first().map_or(0.0, |w| w.cosine);
        b_best
            .partial_cmp(&a_best)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    eprintln!("\n=== Results ===");
    eprintln!("  {} rhyme groups found", groups.len());
    eprintln!("  {total_rhyming} total rhyming words");

    // Print top groups to stderr
    let show_n = 30.min(groups.len());
    eprintln!("\nTop {show_n} rhyme groups:");
    for (i, g) in groups.iter().take(show_n).enumerate() {
        let words_str: String = g
            .words
            .iter()
            .map(|w| format!("{}({:.3})", w.word, w.cosine))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "  #{:2}  [{}]  ({} words): {words_str}",
            i + 1,
            g.rhyme_ending,
            g.words.len()
        );
    }

    // 6. JSON output
    let output = RhymePairsOutput {
        explore_json: explore_path.display().to_string(),
        cmu_dict: args.cmu_dict.display().to_string(),
        min_cosine: min_cos,
        n_features_loaded: explore.features.len(),
        n_clean_english: n_clean,
        n_in_cmu_dict: n_in_cmu,
        n_rhyme_groups: groups.len(),
        n_rhyming_words: total_rhyming,
        rhyme_groups: groups,
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

/// Load CMU Pronouncing Dictionary from file.
///
/// Returns `HashMap<word, phonemes>` where word is lowercase and phonemes
/// is the list of ARPAbet symbols. Only keeps the first pronunciation
/// for words with alternates (e.g., ignores "word(2)").
fn load_cmu_dict(path: &std::path::Path) -> Result<HashMap<String, Vec<String>>> {
    let file = fs::File::open(path)
        .with_context(|| format!("Failed to open CMU dict: {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut dict: HashMap<String, Vec<String>> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with(";;;") {
            continue;
        }
        let mut parts = line.splitn(2, ' ');
        let word = match parts.next() {
            Some(w) => w.to_lowercase(),
            None => continue,
        };
        let phonemes_str = match parts.next() {
            Some(p) => p.trim(),
            None => continue,
        };

        // Skip alternate pronunciations like "word(2)"
        if word.contains('(') {
            continue;
        }

        let phonemes: Vec<String> = phonemes_str.split_whitespace().map(String::from).collect();
        dict.entry(word).or_insert(phonemes);
    }

    Ok(dict)
}

/// Extract the rhyme ending from a list of ARPAbet phonemes.
///
/// The rhyme ending is everything from the last stressed vowel (with stress
/// marker 1 or 2) to the end. If no stressed vowel, uses the last vowel.
/// Vowels are phonemes containing a digit (0, 1, 2).
fn rhyme_ending(phonemes: &[String]) -> String {
    // Find the last vowel with primary stress (1), or secondary (2), or any vowel
    let is_vowel = |ph: &str| ph.chars().any(|c| c.is_ascii_digit());

    // Try primary stress first
    let mut last_stressed = None;
    for (i, ph) in phonemes.iter().enumerate() {
        if ph.ends_with('1') {
            last_stressed = Some(i);
        }
    }

    // Fall back to secondary stress
    if last_stressed.is_none() {
        for (i, ph) in phonemes.iter().enumerate() {
            if ph.ends_with('2') {
                last_stressed = Some(i);
            }
        }
    }

    // Fall back to any vowel
    if last_stressed.is_none() {
        for (i, ph) in phonemes.iter().enumerate().rev() {
            if is_vowel(ph) {
                last_stressed = Some(i);
                break;
            }
        }
    }

    match last_stressed {
        Some(idx) => phonemes[idx..].join(" "),
        None => String::new(),
    }
}

// ── detect-planning mode ──────────────────────────────────────────────────

#[derive(Serialize)]
struct PlanningPromptResult {
    text: String,
    line_ending_word: String,
    group: String,
    /// Activations of features for rhyme-partner words (same group, different word)
    in_group: BTreeMap<String, f32>,
    /// Activations of features from other groups (control)
    out_group: BTreeMap<String, f32>,
    in_group_mean: f32,
    out_group_mean: f32,
    ratio: f32,
}

#[derive(Serialize)]
struct PlanningOutput {
    model: String,
    clt_repo: String,
    n_prompts: usize,
    groups_tested: Vec<String>,
    per_prompt: Vec<PlanningPromptResult>,
    summary: PlanningSummary,
}

#[derive(Serialize)]
struct PlanningSummary {
    overall_in_group_mean: f32,
    overall_out_group_mean: f32,
    overall_ratio: f32,
    per_group: BTreeMap<String, GroupSummary>,
}

#[derive(Serialize)]
struct GroupSummary {
    n_prompts: usize,
    in_group_mean: f32,
    out_group_mean: f32,
    ratio: f32,
}

/// Detect planning: check if rhyme-partner CLT features activate at the
/// last token of a completion-style prompt (the space before the gap word).
///
/// Uses primed completion prompts (from `make_completion_prompts`) across
/// all 10 rhyme groups. Measures at the final token position, where the
/// model's residual stream should encode the planning signal for the
/// upcoming rhyming word.
fn mode_detect_planning(args: &Args) -> Result<()> {
    eprintln!("=== Detect Planning in CLT Feature Activations ===\n");

    // 1. Load rhyme pairs
    let rp_path = args
        .rhyme_pairs
        .as_ref()
        .context("--rhyme-pairs is required for detect-planning mode")?;
    let rp_text = fs::read_to_string(rp_path)?;
    let rp: RhymePairsOutput = serde_json::from_str(&rp_text)?;
    eprintln!(
        "Loaded {} rhyme groups from {}",
        rp.rhyme_groups.len(),
        rp_path.display()
    );

    // 2. Build feature map: human group name → [(word, feature_id)]
    //    Map CMU phoneme endings from rhyme-pairs JSON to human group names
    let cmu_to_human: HashMap<&str, &str> = [
        ("AW1 N D", "-ound"),
        ("OW1", "-ow"),
        ("UW1", "-oo"),
        ("IY1", "-ee"),
        ("EH1 L", "-ell"),
        ("EY1 T", "-ate"),
        ("IH1 L", "-ill"),
        ("AY1 ER0", "-ire"),
        ("AW1 T", "-out"),
        ("AY1 N D", "-ind"),
    ]
    .iter()
    .copied()
    .collect();

    let mut group_features: BTreeMap<String, Vec<(String, CltFeatureId)>> = BTreeMap::new();
    let mut needed_layers: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

    for g in &rp.rhyme_groups {
        let human_name = match cmu_to_human.get(g.rhyme_ending.as_str()) {
            Some(name) => *name,
            None => continue,
        };
        let mut feats = Vec::new();
        for w in &g.words {
            feats.push((w.word.clone(), w.feature));
            needed_layers.insert(w.feature.layer);
        }
        group_features.insert(human_name.to_string(), feats);
    }

    let test_groups: Vec<String> = group_features.keys().cloned().collect();
    eprintln!("Test groups ({}):", test_groups.len());
    for (g, feats) in &group_features {
        let words: String = feats
            .iter()
            .map(|(w, f)| format!("{w}({f})"))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("  [{g}]: {words}");
    }
    eprintln!("Layers needed: {needed_layers:?}");

    // 3. Generate completion-style prompts (skip V5ref control group)
    let all_prompts = make_completion_prompts();
    let prompts: Vec<&CompletionPrompt> =
        all_prompts.iter().filter(|p| p.group != "V5ref").collect();
    eprintln!("\n{} completion prompts (excl. V5ref)", prompts.len());

    // 4. Load model
    eprintln!("Loading model: {}", args.model);
    let force_cpu = if args.cpu { Some(true) } else { None };
    let model = PlipModel::from_pretrained_with_device(&args.model, force_cpu)?;

    // 5. Run forward passes — collect ActivationCaches
    //    Add trailing space so the last token is the space before the gap word
    eprintln!("Running forward passes...");
    let mut caches: Vec<ActivationCache> = Vec::with_capacity(prompts.len());
    for (i, p) in prompts.iter().enumerate() {
        let prompt_text = format!("{} ", p.text);
        let cache = model.get_activations(&prompt_text)?;
        caches.push(cache);
        if (i + 1) % 5 == 0 || i + 1 == prompts.len() {
            eprintln!("  {}/{} prompts done", i + 1, prompts.len());
        }
    }

    // 6. Open CLT and encode activations per layer
    eprintln!("\nOpening CLT: {}", args.clt_repo);
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    };

    // For each prompt, collect activations of ALL features we care about
    // Key: (prompt_idx, feature_id) → activation value
    let mut feature_acts: HashMap<(usize, CltFeatureId), f32> = HashMap::new();

    for &layer in &needed_layers {
        eprintln!("  Loading CLT encoder for layer {layer}...");
        clt.load_encoder(layer, &device)?;

        for (pi, cache) in caches.iter().enumerate() {
            let residual = cache
                .get_layer(layer)
                .context(format!("No activation for layer {layer}"))?;

            let sparse = clt.encode(residual, layer)?;
            for (fid, act) in &sparse.features {
                feature_acts.insert((pi, *fid), *act);
            }
        }
    }

    eprintln!("  CLT encoding complete");

    // 7. Compute per-prompt results
    let mut per_prompt: Vec<PlanningPromptResult> = Vec::new();

    for (pi, prompt) in prompts.iter().enumerate() {
        let my_group = prompt.group;
        let my_word = prompt.target_word;

        // In-group: features for OTHER words in the same rhyme group
        let mut in_group: BTreeMap<String, f32> = BTreeMap::new();
        if let Some(feats) = group_features.get(my_group) {
            for (word, fid) in feats {
                if word == my_word {
                    continue; // skip the line-ending word's own feature
                }
                let act = feature_acts.get(&(pi, *fid)).copied().unwrap_or(0.0);
                in_group.insert(word.clone(), act);
            }
        }

        // Out-group: features from ALL other test groups
        let mut out_group: BTreeMap<String, f32> = BTreeMap::new();
        for (g, feats) in &group_features {
            if g == my_group {
                continue;
            }
            for (word, fid) in feats {
                let act = feature_acts.get(&(pi, *fid)).copied().unwrap_or(0.0);
                out_group.insert(format!("{word}[{g}]"), act);
            }
        }

        let in_mean = if in_group.is_empty() {
            0.0
        } else {
            in_group.values().sum::<f32>() / in_group.len() as f32
        };
        let out_mean = if out_group.is_empty() {
            0.0
        } else {
            out_group.values().sum::<f32>() / out_group.len() as f32
        };
        let ratio = if out_mean > 1e-8 {
            in_mean / out_mean
        } else if in_mean > 0.0 {
            f32::INFINITY
        } else {
            1.0
        };

        per_prompt.push(PlanningPromptResult {
            text: format!("{} ", prompt.text),
            line_ending_word: my_word.to_string(),
            group: my_group.to_string(),
            in_group,
            out_group,
            in_group_mean: in_mean,
            out_group_mean: out_mean,
            ratio,
        });
    }

    // 8. Print per-prompt results
    eprintln!("\n=== Per-Prompt Results ===");
    for r in &per_prompt {
        let in_words: String = r
            .in_group
            .iter()
            .filter(|(_, &v)| v > 0.0)
            .map(|(w, v)| format!("{w}={v:.3}"))
            .collect::<Vec<_>>()
            .join(", ");
        let in_active = r.in_group.values().filter(|&&v| v > 0.0).count();
        let out_active = r.out_group.values().filter(|&&v| v > 0.0).count();
        eprintln!(
            "  [{:6}] \"...{}\" → in={:.4} ({}/{} active{}) out={:.4} ({}/{} active) ratio={:.2}",
            r.group,
            r.line_ending_word,
            r.in_group_mean,
            in_active,
            r.in_group.len(),
            if in_words.is_empty() {
                String::new()
            } else {
                format!(": {in_words}")
            },
            r.out_group_mean,
            out_active,
            r.out_group.len(),
            r.ratio
        );
    }

    // 9. Compute summary
    let overall_in: f32 =
        per_prompt.iter().map(|r| r.in_group_mean).sum::<f32>() / per_prompt.len() as f32;
    let overall_out: f32 =
        per_prompt.iter().map(|r| r.out_group_mean).sum::<f32>() / per_prompt.len() as f32;
    let overall_ratio = if overall_out > 1e-8 {
        overall_in / overall_out
    } else {
        0.0
    };

    let mut per_group_summary: BTreeMap<String, GroupSummary> = BTreeMap::new();
    for group_name in &test_groups {
        let group_results: Vec<&PlanningPromptResult> = per_prompt
            .iter()
            .filter(|r| r.group == *group_name)
            .collect();
        if group_results.is_empty() {
            continue;
        }
        let n = group_results.len();
        let g_in = group_results.iter().map(|r| r.in_group_mean).sum::<f32>() / n as f32;
        let g_out = group_results.iter().map(|r| r.out_group_mean).sum::<f32>() / n as f32;
        let g_ratio = if g_out > 1e-8 { g_in / g_out } else { 0.0 };
        per_group_summary.insert(
            group_name.clone(),
            GroupSummary {
                n_prompts: n,
                in_group_mean: g_in,
                out_group_mean: g_out,
                ratio: g_ratio,
            },
        );
    }

    eprintln!("\n=== Summary ===");
    eprintln!(
        "  Overall: in_group={overall_in:.6}, out_group={overall_out:.6}, ratio={overall_ratio:.2}"
    );
    for (g, s) in &per_group_summary {
        eprintln!(
            "  [{g}]: in={:.6}, out={:.6}, ratio={:.2} (n={})",
            s.in_group_mean, s.out_group_mean, s.ratio, s.n_prompts
        );
    }

    // 10. JSON output
    let output = PlanningOutput {
        model: args.model.clone(),
        clt_repo: args.clt_repo.clone(),
        n_prompts: prompts.len(),
        groups_tested: test_groups.clone(),
        per_prompt,
        summary: PlanningSummary {
            overall_in_group_mean: overall_in,
            overall_out_group_mean: overall_out,
            overall_ratio,
            per_group: per_group_summary,
        },
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

// ── verify-rhyming mode ─────────────────────────────────────────────────────

/// A completion-style prompt: line 1 ends with a target word, line 2 is
/// incomplete — the model should complete it with a rhyming word.
struct CompletionPrompt {
    group: &'static str,
    target_word: &'static str,
    /// Full couplet text up to (but not including) the final rhyming word.
    /// The model's first generated token(s) should complete the rhyme.
    text: &'static str,
}

/// Strip HTML tags from text (e.g. `<em>word</em>` → `word`).
fn strip_html_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_tag = false;
    for c in text.chars() {
        if c == '<' {
            in_tag = true;
        } else if c == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(c);
        }
    }
    result
}

/// Extract the first word from generated text, stripping HTML tags and punctuation.
fn first_word(text: &str) -> String {
    let clean = strip_html_tags(text);
    clean
        .split(|c: char| c.is_whitespace() || c == '\n')
        .find(|w| !w.is_empty())
        .unwrap_or("")
        .trim_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase()
}

/// Check if two words rhyme using the CMU dict.
fn words_rhyme(a: &str, b: &str, cmu: &HashMap<String, Vec<String>>) -> bool {
    let a_low = a.to_lowercase();
    let b_low = b.to_lowercase();
    if a_low == b_low {
        return true;
    }
    match (cmu.get(&a_low), cmu.get(&b_low)) {
        (Some(pa), Some(pb)) => {
            let ra = rhyme_ending(pa);
            let rb = rhyme_ending(pb);
            !ra.is_empty() && ra == rb
        }
        _ => false,
    }
}

/// Build completion-style prompts across all 10 rhyme groups.
///
/// Each prompt: primer couplet + target line 1 + incomplete line 2.
/// The model completes the last word, which should rhyme with target_word.
fn make_completion_prompts() -> Vec<CompletionPrompt> {
    vec![
        // ── -ound (found, round, around, ground) ────────────────────────
        CompletionPrompt {
            group: "-ound",
            target_word: "found",
            text: "The stars were twinkling in the night,\n\
                   The lanterns cast a golden light.\n\
                   At last the missing piece was found,\n\
                   Half buried in the frozen",
        },
        CompletionPrompt {
            group: "-ound",
            target_word: "around",
            text: "A sailor sailed across the bay,\n\
                   And dreamed of home throughout the day.\n\
                   The children danced and spun around,\n\
                   Their laughter was the only",
        },
        CompletionPrompt {
            group: "-ound",
            target_word: "ground",
            text: "She walked along the river's edge,\n\
                   And rested by the garden hedge.\n\
                   Upon the hill we held our ground,\n\
                   The echo made a thundering",
        },
        // ── -ow (so, grow, go, know, slow, though, snow) ────────────────
        CompletionPrompt {
            group: "-ow",
            target_word: "grow",
            text: "The sun goes up, the sun goes down,\n\
                   The moon shines bright above the town.\n\
                   The flowers in the garden grow,\n\
                   In winter covered deep in",
        },
        CompletionPrompt {
            group: "-ow",
            target_word: "go",
            text: "The stars were twinkling in the night,\n\
                   The lanterns cast a golden light.\n\
                   The time has come for us to go,\n\
                   The river keeps its steady",
        },
        CompletionPrompt {
            group: "-ow",
            target_word: "so",
            text: "A sailor sailed across the bay,\n\
                   And dreamed of home throughout the day.\n\
                   The world keeps spinning even so,\n\
                   There is so much we do not",
        },
        // ── -oo (to, who, do, too, two, new) ────────────────────────────
        CompletionPrompt {
            group: "-oo",
            target_word: "who",
            text: "The sun goes up, the sun goes down,\n\
                   The moon shines bright above the town.\n\
                   Nobody knows or remembers who,\n\
                   Would come to find a way back",
        },
        CompletionPrompt {
            group: "-oo",
            target_word: "new",
            text: "She walked along the river's edge,\n\
                   And rested by the garden hedge.\n\
                   The morning brings a day brand new,\n\
                   The sky above a brilliant",
        },
        CompletionPrompt {
            group: "-oo",
            target_word: "do",
            text: "The stars were twinkling in the night,\n\
                   The lanterns cast a golden light.\n\
                   The world has things it needs to do,\n\
                   And dreams it needs to follow",
        },
        // ── -ee (we, be, free, he, she, me) ─────────────────────────────
        CompletionPrompt {
            group: "-ee",
            target_word: "free",
            text: "The sun goes up, the sun goes down,\n\
                   The moon shines bright above the town.\n\
                   The birds above are flying free,\n\
                   As far as any eye can",
        },
        CompletionPrompt {
            group: "-ee",
            target_word: "free",
            text: "A sailor sailed across the bay,\n\
                   And dreamed of home throughout the day.\n\
                   The golden sun has set us free,\n\
                   Beneath the shade of every",
        },
        CompletionPrompt {
            group: "-ee",
            target_word: "be",
            text: "The stars were twinkling in the night,\n\
                   The lanterns cast a golden light.\n\
                   She told the world she wants to be,\n\
                   As happy as a bird at",
        },
        // ── -ell (tell, well) ───────────────────────────────────────────
        CompletionPrompt {
            group: "-ell",
            target_word: "tell",
            text: "The sun goes up, the sun goes down,\n\
                   The moon shines bright above the town.\n\
                   He had a secret he could tell,\n\
                   About a magic wishing",
        },
        CompletionPrompt {
            group: "-ell",
            target_word: "well",
            text: "A sailor sailed across the bay,\n\
                   And dreamed of home throughout the day.\n\
                   She rang the bell and gave a yell,\n\
                   The echo traveled down the",
        },
        // ── -ate (straight, great, weight) ──────────────────────────────
        CompletionPrompt {
            group: "-ate",
            target_word: "straight",
            text: "The sun goes up, the sun goes down,\n\
                   The moon shines bright above the town.\n\
                   The city walls were tall and straight,\n\
                   The traveler arrived too",
        },
        CompletionPrompt {
            group: "-ate",
            target_word: "great",
            text: "She walked along the river's edge,\n\
                   And rested by the garden hedge.\n\
                   The task ahead was truly great,\n\
                   The hero could no longer",
        },
        // ── -ill (will, still, until) ───────────────────────────────────
        CompletionPrompt {
            group: "-ill",
            target_word: "hill",
            text: "A sailor sailed across the bay,\n\
                   And dreamed of home throughout the day.\n\
                   Beyond the valley lies a hill,\n\
                   Where everything is calm and",
        },
        CompletionPrompt {
            group: "-ill",
            target_word: "still",
            text: "The stars were twinkling in the night,\n\
                   The lanterns cast a golden light.\n\
                   The winter wind was sharp and chill,\n\
                   The village rested on the",
        },
        // ── -ire (require, fire, entire) ────────────────────────────────
        CompletionPrompt {
            group: "-ire",
            target_word: "fire",
            text: "The sun goes up, the sun goes down,\n\
                   The moon shines bright above the town.\n\
                   The soldiers gathered by the fire,\n\
                   The smoke rose ever so much",
        },
        CompletionPrompt {
            group: "-ire",
            target_word: "fire",
            text: "She walked along the river's edge,\n\
                   And rested by the garden hedge.\n\
                   The situation was most dire,\n\
                   The flames consumed the raging",
        },
        // ── -out (about, out) ───────────────────────────────────────────
        CompletionPrompt {
            group: "-out",
            target_word: "shout",
            text: "A sailor sailed across the bay,\n\
                   And dreamed of home throughout the day.\n\
                   He raised his voice and gave a shout,\n\
                   The truth was struggling to come",
        },
        CompletionPrompt {
            group: "-out",
            target_word: "about",
            text: "The stars were twinkling in the night,\n\
                   The lanterns cast a golden light.\n\
                   She wandered in the dark about,\n\
                   And found a hidden passage",
        },
        // ── -ind (kind, find) ───────────────────────────────────────────
        CompletionPrompt {
            group: "-ind",
            target_word: "find",
            text: "The sun goes up, the sun goes down,\n\
                   The moon shines bright above the town.\n\
                   She searched and searched but couldn't find,\n\
                   A single thought within her",
        },
        CompletionPrompt {
            group: "-ind",
            target_word: "kind",
            text: "A sailor sailed across the bay,\n\
                   And dreamed of home throughout the day.\n\
                   The stranger was both wise and kind,\n\
                   And left the past far",
        },
        // ── V5 reference prompts (known successes from Phase 0a) ────────
        CompletionPrompt {
            group: "V5ref",
            target_word: "blue",
            text: "Roses are red, violets are blue,\n\
                   Sugar is sweet, and so are",
        },
        CompletionPrompt {
            group: "V5ref",
            target_word: "town",
            text: "The sun goes up, the sun goes down,\n\
                   The moon shines bright above the town.\n\
                   The wind blows east, the wind blows west,\n\
                   The birds fly home to find their",
        },
        CompletionPrompt {
            group: "V5ref",
            target_word: "down",
            text: "The sun goes up, the sun goes down,\n\
                   The moon shines bright above the",
        },
    ]
}

// ── position-sweep mode ──────────────────────────────────────────────────

/// Position sweep result for a single prompt: CLT feature activation at every
/// token position (Version A of Anthropic Figure 13 replication).
#[derive(Serialize)]
struct PositionActivation {
    position: usize,
    token: String,
    activation: f32,
}

#[derive(Serialize)]
struct PositionSweepResult {
    prompt_text: String,
    group: String,
    target_word: String,
    feature_word: String,
    target_feature: CltFeatureId,
    last_token_activation: f32,
    max_activation: f32,
    max_activation_position: usize,
    positions: Vec<PositionActivation>,
}

#[derive(Serialize)]
struct PositionSweepOutput {
    model: String,
    clt_repo: String,
    n_candidates: usize,
    results: Vec<PositionSweepResult>,
}

/// Sweep candidate: a (prompt, feature) pair to measure activation at every position.
struct SweepCandidate {
    group: &'static str,
    target_word: &'static str,
    feature_word: &'static str,
    feature: CltFeatureId,
}

/// Version A of Anthropic Figure 13 replication.
///
/// Measures CLT feature activation at every token position in a prompt, not just
/// the last. If the planning feature shows a sharp spike at the planning site
/// (last token) and near-zero elsewhere, we have correlational evidence that
/// planning is temporally localized — the analog of Anthropic's Figure 13.
fn mode_position_sweep(args: &Args) -> Result<()> {
    eprintln!("=== Position Sweep: Figure 13 Version A ===\n");

    // 1. Load rhyme pairs to resolve feature IDs dynamically
    let rp_path = args
        .rhyme_pairs
        .as_ref()
        .context("--rhyme-pairs is required for position-sweep mode")?;
    let rp_text = fs::read_to_string(rp_path)?;
    let rp: RhymePairsOutput = serde_json::from_str(&rp_text)?;

    // Build word→feature lookup from rhyme pairs
    let mut word_to_feature: HashMap<String, CltFeatureId> = HashMap::new();
    for g in &rp.rhyme_groups {
        for w in &g.words {
            word_to_feature.insert(w.word.clone(), w.feature);
        }
    }

    // 2. Define sweep candidates — the 4 best from planning_detection_v2.json
    //    For each: the prompt ends with `target_word`, and we measure the
    //    activation of `feature_word`'s CLT feature (a rhyme partner).
    let candidate_specs: Vec<(&str, &str, &str)> = vec![
        // (group, target_word in prompt, feature_word whose feature we track)
        ("-ow", "so", "go"),
        ("-out", "about", "out"),
        ("-out", "shout", "out"),
        ("-oo", "who", "ou"),
    ];

    let mut candidates: Vec<SweepCandidate> = Vec::new();
    for (group, target_word, feature_word) in &candidate_specs {
        match word_to_feature.get(*feature_word) {
            Some(&feature) => {
                candidates.push(SweepCandidate {
                    group,
                    target_word,
                    feature_word,
                    feature,
                });
                eprintln!(
                    "  Candidate: [{group}] \"{target_word}\" prompt → track \"{feature_word}\" feature ({feature})"
                );
            }
            None => {
                eprintln!(
                    "  WARNING: feature word \"{feature_word}\" not found in rhyme pairs, skipping"
                );
            }
        }
    }

    if candidates.is_empty() {
        anyhow::bail!("No valid sweep candidates found");
    }

    // 3. Get the matching prompts from make_completion_prompts()
    let all_prompts = make_completion_prompts();

    // 4. Load model
    eprintln!("\nLoading model: {}", args.model);
    let force_cpu = if args.cpu { Some(true) } else { None };
    let model = PlipModel::from_pretrained_with_device(&args.model, force_cpu)?;

    // 5. Open CLT
    eprintln!("Opening CLT: {}", args.clt_repo);
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    };

    // 6. Run position sweep for each candidate
    let mut results: Vec<PositionSweepResult> = Vec::new();

    for candidate in &candidates {
        // Find the matching prompt
        let prompt = all_prompts
            .iter()
            .find(|p| p.group == candidate.group && p.target_word == candidate.target_word)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No prompt found for group={} target_word={}",
                    candidate.group,
                    candidate.target_word
                )
            })?;

        // Add trailing space (same convention as detect-planning)
        let prompt_text = format!("{} ", prompt.text);

        eprintln!(
            "\n--- Sweep: [{:5}] \"...{}\" → feature {} (\"{}\") ---",
            candidate.group, candidate.target_word, candidate.feature, candidate.feature_word
        );

        // Tokenize for display
        let tokens = model.tokenize(&prompt_text)?;
        let seq_len = tokens.len();
        eprintln!("  Tokens ({seq_len}):");
        for (i, tok) in tokens.iter().enumerate() {
            let display = tok.replace('\n', "\\n");
            eprintln!("    [{i:3}] {display:?}");
        }

        // Forward pass with all-position cache
        eprintln!("  Running forward pass (all positions)...");
        let full_cache = model.get_all_position_activations(&prompt_text)?;
        let cache_seq_len = full_cache.seq_len()?;
        eprintln!(
            "  Cache: {} layers, seq_len={cache_seq_len}",
            full_cache.n_layers()
        );

        // Load CLT encoder for the target feature's layer
        let layer = candidate.feature.layer;
        clt.load_encoder(layer, &device)?;

        // Encode at each position and extract the target feature's activation
        eprintln!(
            "  Encoding at layer {layer} for feature {}...",
            candidate.feature
        );
        let mut positions: Vec<PositionActivation> = Vec::with_capacity(cache_seq_len);

        for pos in 0..cache_seq_len {
            let residual = full_cache.get_position(layer, pos)?;
            let sparse = clt.encode(&residual, layer)?;

            // Look up the target feature's activation
            let activation = sparse
                .features
                .iter()
                .find(|(fid, _)| *fid == candidate.feature)
                .map_or(0.0, |(_, act)| *act);

            let token_str = if pos < tokens.len() {
                tokens[pos].clone()
            } else {
                "???".to_string()
            };

            positions.push(PositionActivation {
                position: pos,
                token: token_str,
                activation,
            });
        }

        // Print table
        let last_act = positions.last().map_or(0.0, |p| p.activation);
        let (max_pos, max_act) = positions
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.activation
                    .partial_cmp(&b.activation)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or((0, 0.0), |(i, p)| (i, p.activation));

        eprintln!("\n  Position | Token                | Activation");
        eprintln!("  ---------+----------------------+-----------");
        for p in &positions {
            let marker = if (p.activation - max_act).abs() < f32::EPSILON && max_act > 0.0 {
                " <<<" // Mark the maximum
            } else {
                ""
            };
            let display = p.token.replace('\n', "\\n");
            eprintln!(
                "  {:>7}  | {:<20} | {:.6}{}",
                p.position, display, p.activation, marker
            );
        }
        eprintln!(
            "\n  Last-token activation: {last_act:.6}  |  Max activation: {max_act:.6} at position {max_pos}"
        );

        results.push(PositionSweepResult {
            prompt_text: prompt_text.clone(),
            group: candidate.group.to_string(),
            target_word: candidate.target_word.to_string(),
            feature_word: candidate.feature_word.to_string(),
            target_feature: candidate.feature,
            last_token_activation: last_act,
            max_activation: max_act,
            max_activation_position: max_pos,
            positions,
        });
    }

    // 7. Summary
    eprintln!("\n=== Summary ===");
    for r in &results {
        let is_last_max = (r.last_token_activation - r.max_activation).abs() < 1e-8;
        let verdict = if is_last_max {
            "LOCALIZED (max at last token)"
        } else {
            "NOT localized"
        };
        eprintln!(
            "  [{:5}] \"{:6}\" → \"{:4}\" feature: last={:.4}, max={:.4} at pos {} → {}",
            r.group,
            r.target_word,
            r.feature_word,
            r.last_token_activation,
            r.max_activation,
            r.max_activation_position,
            verdict
        );
    }

    // 8. JSON output
    let output = PositionSweepOutput {
        model: args.model.clone(),
        clt_repo: args.clt_repo.clone(),
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

/// Phase 0a replication: verify rhyming with completion-style prompts.
///
/// Uses the V5 protocol: priming couplet + target line + incomplete second line.
/// The model completes the last word, which should rhyme with the target word.
/// Tests at T=0 (greedy) then T=1 (sampling). Tracks per-group success rates.
fn mode_verify_rhyming(args: &Args) -> Result<()> {
    eprintln!("=== Phase 0a: Verify Rhyming (Completion-Style) ===\n");

    // Load CMU dict for rhyme checking
    let cmu = load_cmu_dict(&args.cmu_dict)?;
    eprintln!("CMU dict loaded: {} entries", cmu.len());

    let prompts = make_completion_prompts();
    eprintln!(
        "{} completion prompts across 10 rhyme groups\n",
        prompts.len()
    );

    // Load model
    eprintln!("Loading model: {}", args.model);
    let force_cpu = if args.cpu { Some(true) } else { None };
    let model = PlipModel::from_pretrained_with_device(&args.model, force_cpu)?;

    // Build conditions: baseline T=0, optional HTML tags, T=1
    let mut conditions: Vec<(f32, String, String)> =
        vec![(0.0, " ".to_string(), "T=0 (greedy)".to_string())];
    if let Some(ref tags_str) = args.tags {
        for tag in tags_str.split(',') {
            let tag = tag.trim();
            if !tag.is_empty() {
                conditions.push((0.0, format!(" <{tag}>"), format!("T=0 + <{tag}> tag")));
            }
        }
    }
    conditions.push((1.0, " ".to_string(), "T=1 (sampling)".to_string()));

    for (temp, suffix, temp_label) in &conditions {
        eprintln!("\n=== {temp_label} ===\n");

        let mut group_hits: BTreeMap<&str, (usize, usize)> = BTreeMap::new();

        for p in &prompts {
            // Prompt ends mid-sentence; model completes with the rhyming word
            // With HTML tags: test if emphasis markup acts as a rhyme marker
            let prompt = format!("{}{}", p.text, suffix);
            let output = model.generate(&prompt, 20, *temp, &[])?;
            let generated = &output[prompt.len()..];
            let completed_word = first_word(generated);
            let rhymes = words_rhyme(p.target_word, &completed_word, &cmu);

            let entry = group_hits.entry(p.group).or_insert((0, 0));
            entry.1 += 1;
            if rhymes {
                entry.0 += 1;
            }

            let mark = if rhymes { "Y" } else { "N" };
            // Show the incomplete line + completion
            let last_line_start = p.text.rfind('\n').map_or(0, |i| i + 1);
            let incomplete_line = &p.text[last_line_start..];
            eprintln!(
                "  [{:6}] \"{}...\" → \"{}\" [{}→{} {}]",
                p.group, incomplete_line, completed_word, p.target_word, completed_word, mark
            );
        }

        // Summary
        let total_hits: usize = group_hits.values().map(|(h, _)| h).sum();
        let total_n: usize = group_hits.values().map(|(_, n)| n).sum();
        eprintln!(
            "\n  Overall: {}/{} ({:.0}%)",
            total_hits,
            total_n,
            100.0 * total_hits as f64 / total_n as f64
        );
        for (g, (h, n)) in &group_hits {
            eprintln!("    {g:6}: {h}/{n} ({:.0}%)", 100.0 * *h as f64 / *n as f64);
        }
    }

    Ok(())
}
