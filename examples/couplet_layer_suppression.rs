//! Layer-suppression causal intervention for couplet rhyming (Planning Circuit Hunt).
//!
//! For each couplet, generates line 2 with different layer groups skipped
//! and checks whether the generated text still rhymes.  Tests the hypothesis
//! that Gemma 2 2B's commitment circuit (L22–25) is causally necessary for
//! rhyming.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example couplet_layer_suppression -- --model google/gemma-2-2b
//! ```

use std::collections::HashSet;
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
#[command(about = "Layer-suppression causal intervention for couplet rhyming")]
struct Args {
    /// `HuggingFace` model ID.
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// Max tokens to generate for line 2.
    #[arg(long, default_value = "30")]
    max_tokens: usize,

    /// Generation temperature (0.0 = greedy).
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Path to write results JSON.
    #[arg(long, default_value = "outputs/couplet_layer_suppression.json")]
    output: String,

    /// Force CPU mode.
    #[arg(long)]
    cpu: bool,
}

// ---------------------------------------------------------------------------
// Couplet definitions (same 10 as other experiments)
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
// Layer groups to test
// ---------------------------------------------------------------------------

struct LayerGroup {
    name: &'static str,
    layers: Vec<usize>,
}

fn layer_groups(n_layers: usize) -> Vec<LayerGroup> {
    vec![
        LayerGroup {
            name: "baseline",
            layers: vec![],
        },
        LayerGroup {
            name: "skip L0-4",
            layers: (0..5).collect(),
        },
        LayerGroup {
            name: "skip L5-9",
            layers: (5..10).collect(),
        },
        LayerGroup {
            name: "skip L10-14",
            layers: (10..15).collect(),
        },
        LayerGroup {
            name: "skip L15-19",
            layers: (15..20).collect(),
        },
        LayerGroup {
            name: "skip L20-25",
            layers: (20..n_layers).collect(),
        },
        LayerGroup {
            name: "skip L22-25",
            layers: (22..n_layers).collect(),
        },
        LayerGroup {
            name: "skip L24-25",
            layers: (24..n_layers).collect(),
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
    groups: Vec<GroupResult>,
}

#[derive(Serialize)]
struct GroupResult {
    group_name: String,
    skipped_layers: Vec<usize>,
    generated_text: String,
    last_word: String,
    rhymes: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the last word-like token from generated text (strip punctuation).
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();
    println!("=== Couplet Layer Suppression — Full Generation ===\n");

    // Load model.
    println!("Loading {}...", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
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
    let groups = layer_groups(n_layers);
    let stop_tokens: Vec<u32> = model.eos_token_id().into_iter().collect();
    let mut results: Vec<CoupletResult> = Vec::new();

    for couplet in &couplets {
        println!(
            "=== Couplet {:>2}: '{}' (target: {}) ===",
            couplet.id, couplet.line1, couplet.target_word
        );

        let prompt = format!("{}\n", couplet.line1);

        // Build lowercase rhyme family.
        let rhyme_set: Vec<String> = couplet
            .rhyme_family
            .iter()
            .map(|w| (*w).to_lowercase())
            .collect();

        let mut group_results: Vec<GroupResult> = Vec::new();

        for group in &groups {
            let skip_set: HashSet<usize> = group.layers.iter().copied().collect();

            let generated = model.generate_with_layer_skip(
                &prompt,
                args.max_tokens,
                args.temperature,
                &stop_tokens,
                &skip_set,
            )?;

            // Extract only the generated part (after the prompt).
            let line2 = generated
                .strip_prefix(&prompt)
                .unwrap_or(&generated)
                .lines()
                .next()
                .unwrap_or("")
                .to_string();

            let last_word = extract_last_word(&line2);
            let rhymes = word_rhymes(&last_word, &rhyme_set);

            let rhyme_marker = if rhymes { "RHYME" } else { "-" };
            println!(
                "  {:<20} [{:<5}] last='{:<12}' | {}",
                group.name, rhyme_marker, last_word, line2
            );

            group_results.push(GroupResult {
                group_name: group.name.to_string(),
                skipped_layers: group.layers.clone(),
                generated_text: line2,
                last_word,
                rhymes,
            });
        }

        println!();
        results.push(CoupletResult {
            id: couplet.id,
            target_word: couplet.target_word.to_string(),
            line1: couplet.line1.to_string(),
            groups: group_results,
        });
    }

    // Write results.
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&args.output, &json)?;
    println!("Results written to {}", args.output);

    // Summary table.
    println!("\n{:=<100}", "");
    println!("SUMMARY: Does line 2 end with a rhyme word?");
    println!("{:=<100}", "");

    print!("{:<8}", "ID");
    for group in &groups {
        print!("{:<14}", group.name);
    }
    println!();
    println!("{:-<100}", "");

    for r in &results {
        print!("{:<8}", r.target_word);
        for g in &r.groups {
            let cell = if g.rhymes {
                g.last_word.clone()
            } else {
                "-".to_string()
            };
            print!("{cell:<14}");
        }
        println!();
    }

    // Count rhymes per group.
    println!("{:-<100}", "");
    print!("{:<8}", "TOTAL");
    for (gi, _group) in groups.iter().enumerate() {
        let count = results.iter().filter(|r| r.groups[gi].rhymes).count();
        print!("{count}/{:<12}", results.len());
    }
    println!();

    Ok(())
}
