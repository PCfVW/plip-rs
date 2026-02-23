//! CLT decoder→vocabulary projection (Planning Circuit Hunt Phase 2b).
//!
//! For each couplet's top-20 planning features, projects their L25 decoder
//! vectors through the unembedding matrix to see which vocabulary words each
//! feature pushes toward.  This tests whether different features activate
//! different rhyme candidates (evidence of search) or all point to the same
//! word (pure retrieval).
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example couplet_clt_decoder_vocab
//! ```

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::Device;
use clap::Parser;
use plip_rs::{CltFeatureId, CrossLayerTranscoder, PlipModel};
use serde::Serialize;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "CLT decoder→vocabulary projection for couplet rhyming")]
struct Args {
    /// `HuggingFace` model ID.
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// CLT repository ID.
    #[arg(long, default_value = "mntss/clt-gemma-2-2b-426k")]
    clt: String,

    /// Number of top planning features per couplet.
    #[arg(long, default_value = "20")]
    top_k: usize,

    /// Minimum activation threshold for CLT encoding.
    #[arg(long, default_value = "0.0")]
    threshold: f32,

    /// Number of top vocab predictions per decoder vector.
    #[arg(long, default_value = "50")]
    vocab_k: usize,

    /// Path to write results JSON.
    #[arg(long, default_value = "outputs/couplet_clt_decoder_vocab.json")]
    output: String,
}

// ---------------------------------------------------------------------------
// Couplet definitions (with rhyme families from Phase 1)
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
// Data structures
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct FeatureVocab {
    layer: usize,
    index: usize,
    cosine_score: f32,
    top_10_words: Vec<(String, f32)>,
    rhyme_hits: Vec<String>,
    n_rhyme_hits: usize,
}

#[derive(Serialize)]
struct CoupletDecoderResult {
    id: u32,
    target_word: String,
    line1: String,
    features: Vec<FeatureVocab>,
    distinct_rhyme_candidates: Vec<String>,
    n_distinct_candidates: usize,
    n_features_with_rhyme_hits: usize,
}

#[derive(Serialize)]
struct FullResults {
    couplets: Vec<CoupletDecoderResult>,
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
    println!("=== CLT Decoder→Vocab Projection (Phase 2b) ===\n");

    // Load model.
    println!("Loading model {}...", args.model);
    let model = PlipModel::from_pretrained(&args.model)?;
    let n_layers = model.n_layers();
    let final_layer = n_layers - 1;
    println!(
        "Model: {n_layers} layers, {} hidden, {} vocab\n",
        model.d_model(),
        model.vocab_size(),
    );

    // Open CLT.
    println!("Opening CLT {}...", args.clt);
    let mut clt = CrossLayerTranscoder::open(&args.clt)?;
    println!(
        "CLT: {} features/layer, {} total\n",
        clt.config().n_features_per_layer,
        clt.config().n_features_total,
    );

    let output_dir = Path::new(&args.output)
        .parent()
        .context("invalid output path")?;
    fs::create_dir_all(output_dir)?;

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let couplets = couplet_defs();
    let mut results: Vec<CoupletDecoderResult> = Vec::new();

    for couplet in &couplets {
        println!(
            "=== Couplet {:>2}: '{}' (target: {}) ===",
            couplet.id, couplet.line1, couplet.target_word
        );

        // Get last-token activations.
        let prompt = format!("{}\n", couplet.line1);
        let cache = model.get_activations(&prompt)?;

        // Target word embedding for identify_planning_features.
        let target_ids = model.encode(couplet.target_word)?;
        let target_id = *target_ids
            .last()
            .context("target word produced no tokens")?;
        let target_emb = model.token_embedding(target_id)?;

        // Identify planning features.
        let planning = clt.identify_planning_features(
            &cache,
            &target_emb,
            args.top_k,
            args.threshold,
            &device,
        )?;
        println!("  Planning features: {}", planning.len());

        // Batch-load decoder vectors at the final layer.
        let feature_ids: Vec<CltFeatureId> = planning.iter().map(|(fid, _)| *fid).collect();
        clt.cache_steering_vectors_all_downstream(&feature_ids, &device)?;

        // Build lowercase rhyme family for matching.
        let rhyme_set: HashSet<String> = couplet
            .rhyme_family
            .iter()
            .map(|w| (*w).to_lowercase())
            .collect();

        let mut feature_vocabs: Vec<FeatureVocab> = Vec::new();
        let mut all_rhyme_candidates: HashSet<String> = HashSet::new();
        let mut n_features_with_hits = 0;

        for (fid, cosine) in &planning {
            // Get decoder vector at the final layer.
            let dec_vec = clt.decoder_vector(fid, final_layer, &device)?;

            // Project through unembedding: logit_lens expects [1, d_model].
            // Decoder vectors are F32 but model weights may be BF16 — match dtype.
            let dec_2d = dec_vec.to_dtype(candle_core::DType::BF16)?.unsqueeze(0)?;
            let top_preds = model.logit_lens_activation(&dec_2d, args.vocab_k)?;

            // Check for rhyme family hits.
            let mut rhyme_hits: Vec<String> = Vec::new();
            for (tok, _score) in &top_preds {
                let clean = tok
                    .trim()
                    .trim_end_matches(|c: char| c.is_ascii_punctuation())
                    .to_lowercase();
                if rhyme_set.contains(&clean) && !rhyme_hits.contains(&clean) {
                    rhyme_hits.push(clean.clone());
                    all_rhyme_candidates.insert(clean);
                }
            }

            if !rhyme_hits.is_empty() {
                n_features_with_hits += 1;
            }

            let n_hits = rhyme_hits.len();
            let top_10: Vec<(String, f32)> = top_preds.into_iter().take(10).collect();

            // Print feature info.
            let top3_str: Vec<String> = top_10
                .iter()
                .take(3)
                .map(|(t, s)| format!("{}({:.1})", t.trim(), s))
                .collect();
            let hits_str = if rhyme_hits.is_empty() {
                String::new()
            } else {
                format!("  RHYME: {}", rhyme_hits.join(", "))
            };
            println!(
                "  L{}:{:<6} cos={:.4} → [{}]{hits_str}",
                fid.layer,
                fid.index,
                cosine,
                top3_str.join(", "),
            );

            feature_vocabs.push(FeatureVocab {
                layer: fid.layer,
                index: fid.index,
                cosine_score: *cosine,
                top_10_words: top_10,
                rhyme_hits,
                n_rhyme_hits: n_hits,
            });
        }

        // Collect distinct rhyme candidates.
        let mut distinct: Vec<String> = all_rhyme_candidates.into_iter().collect();
        distinct.sort();
        let n_distinct = distinct.len();

        println!(
            "  → {n_features_with_hits}/{} features have rhyme hits, {n_distinct} distinct candidates: [{}]",
            planning.len(),
            distinct.join(", "),
        );
        println!();

        results.push(CoupletDecoderResult {
            id: couplet.id,
            target_word: couplet.target_word.to_string(),
            line1: couplet.line1.to_string(),
            features: feature_vocabs,
            distinct_rhyme_candidates: distinct,
            n_distinct_candidates: n_distinct,
            n_features_with_rhyme_hits: n_features_with_hits,
        });

        // Free cached decoder vectors.
        clt.clear_steering_cache();
    }

    // Write results.
    let full = FullResults { couplets: results };
    let json = serde_json::to_string_pretty(&full)?;
    fs::write(&args.output, &json)?;
    println!("Results written to {}", args.output);

    // Summary table.
    println!("\n{:-<75}", "");
    println!(
        "{:<4} {:<8} {:<12} {:<12} Distinct candidates",
        "ID", "Target", "Feat w/hits", "# candidates"
    );
    println!("{:-<75}", "");
    for r in &full.couplets {
        let candidates_str = if r.distinct_rhyme_candidates.len() <= 5 {
            r.distinct_rhyme_candidates.join(", ")
        } else {
            let first5: Vec<&str> = r
                .distinct_rhyme_candidates
                .iter()
                .take(5)
                .map(String::as_str)
                .collect();
            format!(
                "{}, +{} more",
                first5.join(", "),
                r.distinct_rhyme_candidates.len() - 5
            )
        };
        println!(
            "{:<4} {:<8} {:<12} {:<12} {}",
            r.id,
            r.target_word,
            format!("{}/{}", r.n_features_with_rhyme_hits, r.features.len()),
            r.n_distinct_candidates,
            candidates_str,
        );
    }

    // Aggregate.
    #[allow(clippy::cast_precision_loss)]
    let avg_candidates: f64 = {
        let sum: usize = full.couplets.iter().map(|r| r.n_distinct_candidates).sum();
        sum as f64 / full.couplets.len() as f64
    };
    #[allow(clippy::cast_precision_loss)]
    let avg_features_with_hits: f64 = {
        let sum: usize = full
            .couplets
            .iter()
            .map(|r| r.n_features_with_rhyme_hits)
            .sum();
        sum as f64 / full.couplets.len() as f64
    };

    println!("\n--- Aggregate ---");
    println!("Avg distinct rhyme candidates per couplet: {avg_candidates:.1}");
    println!("Avg features with rhyme hits per couplet: {avg_features_with_hits:.1}");

    if avg_candidates > 3.0 {
        println!(
            "Interpretation: MULTIPLE CANDIDATES — features push toward different rhyme words."
        );
        println!("This is evidence of search-like behavior (parallel candidate activation).");
    } else if avg_candidates > 1.0 {
        println!("Interpretation: MIXED — some diversity but limited candidate set.");
    } else {
        println!("Interpretation: SINGLE TARGET — features converge on one word (pure retrieval).");
    }

    Ok(())
}
