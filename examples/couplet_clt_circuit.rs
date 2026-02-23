//! CLT feature layer mapping for couplet rhyming (Planning Circuit Hunt Phase 2).
//!
//! For each couplet line-1 prompt, identifies planning-relevant CLT features
//! at the newline position (last token) via `identify_planning_features()`.
//! Then measures where each feature reads from (encoder source layer) and
//! writes to (decoder norm per downstream layer).  Builds histograms and
//! computes feature overlap between couplets to distinguish a general
//! phonological circuit from pair-specific learned associations.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example couplet_clt_circuit
//! ```

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use plip_rs::{CltFeatureId, CrossLayerTranscoder, PlipModel};
use serde::Serialize;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "CLT feature layer mapping for couplet rhyming")]
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

    /// Path to write results JSON.
    #[arg(long, default_value = "outputs/couplet_clt_circuit.json")]
    output: String,
}

// ---------------------------------------------------------------------------
// Couplet definitions (same 10 as Phase 1)
// ---------------------------------------------------------------------------

struct CoupletDef {
    id: u32,
    target_word: &'static str,
    line1: &'static str,
}

fn couplet_defs() -> Vec<CoupletDef> {
    vec![
        CoupletDef {
            id: 1,
            target_word: "light",
            line1: "The moon casts silver light,",
        },
        CoupletDef {
            id: 2,
            target_word: "play",
            line1: "The children laugh and play,",
        },
        CoupletDef {
            id: 3,
            target_word: "sound",
            line1: "The thunder makes a sound,",
        },
        CoupletDef {
            id: 4,
            target_word: "rain",
            line1: "The clouds bring heavy rain,",
        },
        CoupletDef {
            id: 6,
            target_word: "air",
            line1: "The geese fly through the air,",
        },
        CoupletDef {
            id: 7,
            target_word: "gold",
            line1: "The sunset gleams like gold,",
        },
        CoupletDef {
            id: 8,
            target_word: "fire",
            line1: "The embers feed the fire,",
        },
        CoupletDef {
            id: 13,
            target_word: "truth",
            line1: "She spoke the honest truth,",
        },
        CoupletDef {
            id: 14,
            target_word: "world",
            line1: "He traveled all the world,",
        },
        CoupletDef {
            id: 15,
            target_word: "earth",
            line1: "The seeds lay in the earth,",
        },
    ]
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct CoupletCltResult {
    id: u32,
    target_word: String,
    line1: String,
    n_planning_features: usize,
    features: Vec<FeatureInfo>,
    source_layer_histogram: Vec<u32>,
}

#[derive(Serialize)]
struct FeatureInfo {
    layer: usize,
    index: usize,
    cosine_score: f32,
    /// L2 norm of decoder vector at each downstream layer.
    decoder_norms: Vec<(usize, f32)>,
    /// Layer with the strongest decoder write.
    peak_write_layer: usize,
}

#[derive(Clone, Serialize)]
struct OverlapEntry {
    couplet_a: u32,
    couplet_b: u32,
    shared_features: usize,
    total_a: usize,
    total_b: usize,
    jaccard: f32,
}

#[derive(Serialize)]
struct FullResults {
    couplets: Vec<CoupletCltResult>,
    global_source_histogram: Vec<u32>,
    global_target_histogram: Vec<u32>,
    overlap_matrix: Vec<OverlapEntry>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute L2 norm of a tensor on CPU.
fn l2_norm_cpu(t: &candle_core::Tensor) -> Result<f32> {
    let t_f32 = t.to_dtype(DType::F32)?;
    let norm_sq: f32 = t_f32.sqr()?.sum_all()?.to_scalar()?;
    Ok(norm_sq.sqrt())
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
    println!("=== Couplet CLT Circuit (Planning Circuit Hunt Phase 2) ===\n");

    // Load model.
    println!("Loading model {}...", args.model);
    let model = PlipModel::from_pretrained(&args.model)?;
    let n_layers = model.n_layers();
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
    let mut results: Vec<CoupletCltResult> = Vec::new();
    let mut global_source: Vec<u32> = vec![0; n_layers];
    let mut global_target: Vec<u32> = vec![0; n_layers];
    let mut all_feature_sets: Vec<(u32, HashSet<CltFeatureId>)> = Vec::new();

    for couplet in &couplets {
        println!(
            "=== Couplet {:>2}: '{}' (target: {}) ===",
            couplet.id, couplet.line1, couplet.target_word
        );

        // Get last-token activations (newline is last token).
        let prompt = format!("{}\n", couplet.line1);
        let cache = model.get_activations(&prompt)?;
        println!("  Activations: {} layers", cache.n_layers());

        // Get target word embedding for cosine scoring.
        let target_ids = model.encode(couplet.target_word)?;
        let target_id = *target_ids
            .last()
            .context("target word produced no tokens")?;
        let target_emb = model.token_embedding(target_id)?;
        println!(
            "  Target '{}' → token {} ('{}')",
            couplet.target_word,
            target_id,
            model.decode_token(target_id),
        );

        // Identify planning features.
        let planning = clt.identify_planning_features(
            &cache,
            &target_emb,
            args.top_k,
            args.threshold,
            &device,
        )?;
        println!("  Planning features: {}", planning.len());

        // Batch-load all decoder vectors for these features (one file read per
        // source layer instead of one per feature × target_layer).
        let feature_ids: Vec<CltFeatureId> = planning.iter().map(|(fid, _)| *fid).collect();
        clt.cache_steering_vectors_all_downstream(&feature_ids, &device)?;

        // Source layer histogram for this couplet.
        let mut source_hist: Vec<u32> = vec![0; n_layers];
        let mut feature_set: HashSet<CltFeatureId> = HashSet::new();
        let mut feature_infos: Vec<FeatureInfo> = Vec::new();

        for (fid, cosine) in &planning {
            source_hist[fid.layer] += 1;
            global_source[fid.layer] += 1;
            feature_set.insert(*fid);

            // Measure decoder write strength at each downstream layer.
            let mut decoder_norms: Vec<(usize, f32)> = Vec::new();
            let mut peak_layer = fid.layer;
            let mut peak_norm: f32 = 0.0;

            #[allow(clippy::needless_range_loop)]
            for target_layer in fid.layer..n_layers {
                let dec_vec = clt.decoder_vector(fid, target_layer, &device)?;
                let norm = l2_norm_cpu(&dec_vec)?;
                decoder_norms.push((target_layer, norm));
                global_target[target_layer] += 1;
                if norm > peak_norm {
                    peak_norm = norm;
                    peak_layer = target_layer;
                }
            }

            feature_infos.push(FeatureInfo {
                layer: fid.layer,
                index: fid.index,
                cosine_score: *cosine,
                decoder_norms,
                peak_write_layer: peak_layer,
            });
        }

        // Print top-5 features.
        for (i, fi) in feature_infos.iter().take(5).enumerate() {
            println!(
                "  #{}: L{}:{} cos={:.4} peak_write=L{}",
                i + 1,
                fi.layer,
                fi.index,
                fi.cosine_score,
                fi.peak_write_layer,
            );
        }

        // Print source histogram (compact).
        let active_sources: Vec<String> = source_hist
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(l, c)| format!("L{l}:{c}"))
            .collect();
        println!("  Source layers: {}", active_sources.join(", "));
        println!();

        all_feature_sets.push((couplet.id, feature_set));

        results.push(CoupletCltResult {
            id: couplet.id,
            target_word: couplet.target_word.to_string(),
            line1: couplet.line1.to_string(),
            n_planning_features: planning.len(),
            features: feature_infos,
            source_layer_histogram: source_hist,
        });

        // Free cached decoder vectors to reclaim VRAM before next couplet.
        clt.clear_steering_cache();
    }

    // Compute pairwise feature overlap (Jaccard similarity).
    let mut overlap_matrix: Vec<OverlapEntry> = Vec::new();
    for i in 0..all_feature_sets.len() {
        for j in (i + 1)..all_feature_sets.len() {
            let (id_a, set_a) = &all_feature_sets[i];
            let (id_b, set_b) = &all_feature_sets[j];
            let shared = set_a.intersection(set_b).count();
            let union = set_a.union(set_b).count();
            #[allow(clippy::cast_precision_loss)]
            let jaccard = if union > 0 {
                shared as f32 / union as f32
            } else {
                0.0
            };
            overlap_matrix.push(OverlapEntry {
                couplet_a: *id_a,
                couplet_b: *id_b,
                shared_features: shared,
                total_a: set_a.len(),
                total_b: set_b.len(),
                jaccard,
            });
        }
    }

    // Write results.
    let full = FullResults {
        couplets: results,
        global_source_histogram: global_source.clone(),
        global_target_histogram: global_target.clone(),
        overlap_matrix: overlap_matrix.clone(),
    };
    let json = serde_json::to_string_pretty(&full)?;
    fs::write(&args.output, &json)?;
    println!("Results written to {}", args.output);

    // Print global histograms.
    println!("\n=== Global Source Layer Histogram (where features read from) ===");
    for (l, &count) in global_source.iter().enumerate() {
        if count > 0 {
            let bar = "#".repeat(count as usize);
            println!("  L{l:>2}: {bar} ({count})");
        }
    }

    println!("\n=== Global Target Layer Histogram (where features write to) ===");
    let max_target = global_target.iter().copied().max().unwrap_or(1);
    for (l, &count) in global_target.iter().enumerate() {
        if count > 0 {
            // Scale to max 40 chars.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let bar_len = ((f64::from(count) / f64::from(max_target)) * 40.0) as usize;
            let bar = "#".repeat(bar_len.max(1));
            println!("  L{l:>2}: {bar} ({count})");
        }
    }

    // Print overlap matrix.
    println!("\n=== Feature Overlap (Jaccard Similarity) ===");
    println!(
        "{:<6} {:<6} {:<8} {:<8} {:<8} Jaccard",
        "ID-A", "ID-B", "Shared", "Total-A", "Total-B"
    );
    println!("{:-<55}", "");

    let mut nonzero_overlaps: u32 = 0;
    for entry in &overlap_matrix {
        if entry.shared_features > 0 {
            nonzero_overlaps += 1;
        }
        println!(
            "{:<6} {:<6} {:<8} {:<8} {:<8} {:.4}",
            entry.couplet_a,
            entry.couplet_b,
            entry.shared_features,
            entry.total_a,
            entry.total_b,
            entry.jaccard,
        );
    }

    let total_pairs = overlap_matrix.len();
    println!("\n--- Summary ---");
    #[allow(clippy::cast_precision_loss)]
    let pct = if total_pairs > 0 {
        f64::from(nonzero_overlaps) / total_pairs as f64 * 100.0
    } else {
        0.0
    };
    println!("Pairs with shared features: {nonzero_overlaps}/{total_pairs} ({pct:.0}%)");

    #[allow(clippy::cast_precision_loss)]
    let avg_jaccard: f64 = if overlap_matrix.is_empty() {
        0.0
    } else {
        let sum: f64 = overlap_matrix.iter().map(|e| f64::from(e.jaccard)).sum();
        sum / overlap_matrix.len() as f64
    };
    println!("Average Jaccard similarity: {avg_jaccard:.4}");

    if avg_jaccard < 0.05 {
        println!("Interpretation: PAIR-SPECIFIC — different features fire for different couplets.");
    } else if avg_jaccard > 0.2 {
        println!("Interpretation: GENERAL CIRCUIT — shared features across couplets.");
    } else {
        println!("Interpretation: MIXED — partial overlap suggests shared + specific components.");
    }

    Ok(())
}
