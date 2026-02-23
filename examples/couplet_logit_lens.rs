//! Per-position logit lens for couplet rhyming (Planning Circuit Hunt Phase 1).
//!
//! For each couplet line-1 prompt, extracts the `FullActivationCache` and
//! runs logit lens at every layer at the last-token position.  Records at
//! which layer a word from the target rhyme family first enters the top-k
//! predictions.  A control position (mid-line) is also probed for comparison.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example couplet_logit_lens -- --model google/gemma-2-2b
//! ```

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use clap::Parser;
use plip_rs::PlipModel;
use serde::Serialize;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Per-position logit lens for couplet rhyming")]
struct Args {
    /// `HuggingFace` model ID.
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// Number of top predictions to check at each layer.
    #[arg(long, default_value = "50")]
    top_k: usize,

    /// Path to write results JSON.
    #[arg(long, default_value = "outputs/couplet_logit_lens.json")]
    output: String,

    /// Force CPU mode.
    #[arg(long)]
    cpu: bool,
}

// ---------------------------------------------------------------------------
// Couplet definitions (the 10 successful from the Ollama baseline)
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
struct CoupletResult {
    id: u32,
    target_word: String,
    line1: String,
    tokens: Vec<String>,
    probe_position: usize,
    probe_token: String,
    control_position: usize,
    control_token: String,
    probe_layers: Vec<LayerResult>,
    first_rhyme_layer: Option<usize>,
    first_rhyme_word: Option<String>,
    control_layers: Vec<LayerResult>,
    control_first_rhyme_layer: Option<usize>,
}

#[derive(Serialize)]
struct LayerResult {
    layer: usize,
    top_5: Vec<(String, f32)>,
    rhyme_hits: Vec<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if a decoded token matches any word in the rhyme family.
///
/// Strips whitespace and trailing punctuation before comparing.
fn find_rhyme_hits(predictions: &[(String, f32)], rhyme_family: &[String]) -> Vec<String> {
    predictions
        .iter()
        .filter_map(|(tok, _)| {
            let clean = tok
                .trim()
                .trim_end_matches(|c: char| c.is_ascii_punctuation())
                .to_lowercase();
            if rhyme_family.contains(&clean) {
                Some(clean)
            } else {
                None
            }
        })
        .collect()
}

/// Run logit lens at every layer for one position, returning per-layer results
/// and the first layer with a rhyme hit.
fn probe_position(
    model: &PlipModel,
    cache: &plip_rs::FullActivationCache,
    position: usize,
    top_k: usize,
    rhyme_family: &[String],
) -> Result<(Vec<LayerResult>, Option<usize>, Option<String>)> {
    let mut layers = Vec::with_capacity(cache.n_layers());
    let mut first_layer: Option<usize> = None;
    let mut first_word: Option<String> = None;

    for layer in 0..cache.n_layers() {
        // unsqueeze to [1, d_model] — logit_lens expects 2D (ActivationCache
        // stores [batch, d_model], but FullActivationCache::get_position is 1D).
        let activation = cache.get_position(layer, position)?.unsqueeze(0)?;
        let top_k_preds = model.logit_lens_activation(&activation, top_k)?;
        let rhyme_hits = find_rhyme_hits(&top_k_preds, rhyme_family);

        if !rhyme_hits.is_empty() && first_layer.is_none() {
            first_layer = Some(layer);
            first_word = Some(rhyme_hits[0].clone());
        }

        let top_5: Vec<(String, f32)> = top_k_preds.into_iter().take(5).collect();

        layers.push(LayerResult {
            layer,
            top_5,
            rhyme_hits,
        });
    }

    Ok((layers, first_layer, first_word))
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
    println!("=== Couplet Logit Lens (Planning Circuit Hunt Phase 1) ===\n");

    // Load model.
    println!("Loading {}...", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    println!(
        "Model: {} layers, {} hidden, {} vocab\n",
        model.n_layers(),
        model.d_model(),
        model.vocab_size(),
    );

    let output_dir = Path::new(&args.output)
        .parent()
        .context("invalid output path")?;
    fs::create_dir_all(output_dir)?;

    let couplets = couplet_defs();
    let mut results: Vec<CoupletResult> = Vec::new();

    for couplet in &couplets {
        println!(
            "=== Couplet {:>2}: '{}' (target: {}) ===",
            couplet.id, couplet.line1, couplet.target_word
        );

        // Include trailing newline so the last-token position is the \n
        // where the model begins generating line 2.
        let prompt = format!("{}\n", couplet.line1);
        let tokens = model.tokenize(&prompt)?;
        let n_tokens = tokens.len();

        println!("  Tokens ({n_tokens}): {tokens:?}");

        // Probe position: last token (the newline).
        let probe_pos = n_tokens - 1;
        // Control position: roughly the middle of the line.
        let control_pos = n_tokens / 2;

        let probe_tok = tokens[probe_pos].replace('\n', "\\n");
        let control_tok = &tokens[control_pos];
        println!("  Probe pos: {probe_pos} ('{probe_tok}')");
        println!("  Control pos: {control_pos} ('{control_tok}')");

        // Get full activation cache.
        let cache = model.get_all_position_activations(&prompt)?;
        println!(
            "  Cache: {} layers, seq_len={}",
            cache.n_layers(),
            cache.seq_len()?
        );

        // Build lowercase rhyme family for matching.
        let rhyme_set: Vec<String> = couplet
            .rhyme_family
            .iter()
            .map(|w| (*w).to_lowercase())
            .collect();

        // Probe at the last-token position.
        let (probe_layers, first_rhyme_layer, first_rhyme_word) =
            probe_position(&model, &cache, probe_pos, args.top_k, &rhyme_set)?;

        // Print probe results layer by layer.
        for lr in &probe_layers {
            let hits_str = if lr.rhyme_hits.is_empty() {
                String::new()
            } else {
                format!("  RHYME: {}", lr.rhyme_hits.join(", "))
            };
            let t0 = &lr.top_5[0];
            let t1_tok = lr.top_5.get(1).map_or("", |p| p.0.as_str());
            let t1_val = lr.top_5.get(1).map_or(0.0, |p| p.1);
            let t2_tok = lr.top_5.get(2).map_or("", |p| p.0.as_str());
            let t2_val = lr.top_5.get(2).map_or(0.0, |p| p.1);
            println!(
                "  L{:>2}: {:>12} ({:>6.1}) | {:>12} ({:>6.1}) | {:>12} ({:>6.1}){hits_str}",
                lr.layer,
                t0.0.replace('\n', "\\n"),
                t0.1,
                t1_tok.replace('\n', "\\n"),
                t1_val,
                t2_tok.replace('\n', "\\n"),
                t2_val,
            );
        }

        // Probe at the control position.
        let (control_layers, control_first, _) =
            probe_position(&model, &cache, control_pos, args.top_k, &rhyme_set)?;

        // Summary for this couplet.
        if let Some(l) = first_rhyme_layer {
            println!(
                "  --> First rhyme in top-{}: layer {l} ('{}')",
                args.top_k,
                first_rhyme_word.as_deref().unwrap_or("?")
            );
        } else {
            println!("  --> No rhyme word in top-{} at any layer", args.top_k);
        }
        if let Some(cl) = control_first {
            println!("  --> Control also has rhyme at layer {cl}");
        }
        println!();

        results.push(CoupletResult {
            id: couplet.id,
            target_word: couplet.target_word.to_string(),
            line1: couplet.line1.to_string(),
            tokens: tokens.clone(),
            probe_position: probe_pos,
            probe_token: tokens[probe_pos].clone(),
            control_position: control_pos,
            control_token: tokens[control_pos].clone(),
            probe_layers,
            first_rhyme_layer,
            first_rhyme_word,
            control_layers,
            control_first_rhyme_layer: control_first,
        });
    }

    // Write results.
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&args.output, &json)?;
    println!("Results written to {}", args.output);

    // Summary table.
    println!("\n{:-<70}", "");
    println!(
        "{:<4} {:<8} {:<12} {:<18} Control",
        "ID", "Target", "First word", "First rhyme layer"
    );
    println!("{:-<70}", "");
    for r in &results {
        let layer_str = match r.first_rhyme_layer {
            Some(l) => format!("layer {l}"),
            None => "NEVER".to_string(),
        };
        let ctrl_str = match r.control_first_rhyme_layer {
            Some(l) => format!("layer {l}"),
            None => "-".to_string(),
        };
        let word = r.first_rhyme_word.as_deref().unwrap_or("-");
        println!(
            "{:<4} {:<8} {:<12} {:<18} {ctrl_str}",
            r.id, r.target_word, word, layer_str,
        );
    }

    // Aggregate statistics.
    let found_count = results
        .iter()
        .filter(|r| r.first_rhyme_layer.is_some())
        .count();
    let control_count = results
        .iter()
        .filter(|r| r.control_first_rhyme_layer.is_some())
        .count();
    #[allow(clippy::cast_precision_loss)]
    let avg_layer: f64 = {
        let layers: Vec<f64> = results
            .iter()
            .filter_map(|r| r.first_rhyme_layer.map(|l| l as f64))
            .collect();
        if layers.is_empty() {
            f64::NAN
        } else {
            layers.iter().sum::<f64>() / layers.len() as f64
        }
    };

    println!("\n--- Aggregate ---");
    println!(
        "Rhyme found at probe position: {found_count}/{} couplets",
        results.len()
    );
    println!(
        "Rhyme found at control position: {control_count}/{} couplets",
        results.len()
    );
    if !avg_layer.is_nan() {
        println!("Average first-rhyme layer (probe): {avg_layer:.1}");
    }

    Ok(())
}
