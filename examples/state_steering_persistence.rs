//! State Steering Persistence Experiment: Distance × Scale × Temperature
//!
//! Tests whether RWKV-6 state steering can preserve test-syntax generation
//! at variable distances from the test marker position.
//!
//! ## Hypothesis
//!
//! The steered state decays exponentially with distance: each intervening
//! token applies decay factors d_j to the state. Higher steering scales
//! should overcome more decay, preserving test-awareness further from
//! the marker (Math Foundations Section 4).
//!
//! ## Design
//!
//! - **Distance**: close (δ≈0), medium (δ≈20 tokens), far (δ≈40 tokens)
//! - **Scale sweep**: 1.0 (baseline), 3.0, 5.0, 9.0
//! - **Temperature**: 0.0 (greedy argmax), 0.6 (sampling)
//! - At temperature>0: run n_samples times and report test-syntax rate
//!
//! ## Expected pattern
//!
//! If state steering causally affects generation:
//! - Close + high scale → highest test-syntax rate
//! - Far + low scale → lowest test-syntax rate
//! - Scale effect should decrease with distance (interaction effect)
//!
//! Usage:
//!   cargo run --release --example state_steering_persistence
//!   cargo run --release --example state_steering_persistence -- --layer 14 --n-samples 5
//!   cargo run --release --example state_steering_persistence -- --max-tokens 120

// Clippy configuration for ML/statistics code
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::print_literal)]

use anyhow::Result;
use clap::Parser;
use plip_rs::{PlipModel, StateSteeringSpec};

#[derive(Parser)]
#[command(name = "state_steering_persistence")]
#[command(about = "RWKV-6 state steering persistence across distance, scale, and temperature")]
struct Args {
    /// HuggingFace model ID
    #[arg(short, long, default_value = "RWKV/v6-Finch-1B6-HF")]
    model: String,

    /// Layer to apply state steering (default: layer 2, where knockout was significant)
    #[arg(long)]
    layer: Option<usize>,

    /// Maximum tokens to generate per sample
    #[arg(long, default_value = "80")]
    max_tokens: usize,

    /// Number of samples at temperature > 0
    #[arg(long, default_value = "3")]
    n_samples: usize,

    /// Temperature(s) to test (can be specified multiple times)
    /// Default: 0.0 and 0.6
    #[arg(long)]
    temperature: Vec<f32>,

    /// Only run Rust prompts (skip Python)
    #[arg(long)]
    rust_only: bool,

    /// Force CPU mode
    #[arg(long)]
    cpu: bool,
}

/// Scale factors to sweep: 1.0 = baseline (no intervention)
const SCALES: &[f32] = &[1.0, 3.0, 5.0, 9.0];

/// Temperatures to test: greedy and sampling
const TEMPERATURES: &[f32] = &[0.0, 0.6];

/// A prompt with a test marker at a known distance from the generation point.
struct DistancePrompt {
    name: &'static str,
    category: &'static str,
    distance: &'static str,
    text: &'static str,
    marker: &'static str,
}

/// Prompts organized by language × distance.
///
/// Each prompt has:
/// - A test marker (>>> or #[test]) early in the code
/// - Variable padding between marker and the continuation point
/// - An incomplete new function at the end where generation starts
const PROMPTS: &[DistancePrompt] = &[
    // ── Python doctest family ──────────────────────────────────────
    //
    // Close: marker → immediate new function
    DistancePrompt {
        name: "py_close",
        category: "python_doctest",
        distance: "close",
        marker: ">>>",
        text: "def add(a, b):\n    \"\"\"\n    >>> add(2, 3)\n    5\n    \"\"\"\n    return a + b\n\ndef multiply(a, b):\n    \"\"\"",
    },
    // Medium: ~20 tokens of unrelated code between marker and new function
    DistancePrompt {
        name: "py_medium",
        category: "python_doctest",
        distance: "medium",
        marker: ">>>",
        text: "def add(a, b):\n    \"\"\"\n    >>> add(2, 3)\n    5\n    \"\"\"\n    return a + b\n\nx = 42\ny = x + 1\nresult = x * y\nprint(result)\n\ndef multiply(a, b):\n    \"\"\"",
    },
    // Far: ~40 tokens of unrelated code (class definition) between marker and new function
    DistancePrompt {
        name: "py_far",
        category: "python_doctest",
        distance: "far",
        marker: ">>>",
        text: "def add(a, b):\n    \"\"\"\n    >>> add(2, 3)\n    5\n    \"\"\"\n    return a + b\n\nclass Config:\n    def __init__(self):\n        self.debug = False\n        self.verbose = True\n\n    def enable_debug(self):\n        self.debug = True\n\nconfig = Config()\nconfig.enable_debug()\n\ndef multiply(a, b):\n    \"\"\"",
    },
    // ── Rust test family ───────────────────────────────────────────
    //
    // Close: marker → immediate new function
    DistancePrompt {
        name: "rs_close",
        category: "rust_test",
        distance: "close",
        marker: "#[test]",
        text: "/// Returns the larger of two values\n///\n/// #[test]\n/// fn test_max() {\n///     assert_eq!(max(3, 5), 5);\n/// }\nfn max(a: i32, b: i32) -> i32 {\n    if a > b { a } else { b }\n}\n\n/// Returns the smaller of two values\nfn min(a: i32, b: i32) -> i32 {",
    },
    // Medium: struct definition as padding
    DistancePrompt {
        name: "rs_medium",
        category: "rust_test",
        distance: "medium",
        marker: "#[test]",
        text: "/// Returns the larger of two values\n///\n/// #[test]\n/// fn test_max() {\n///     assert_eq!(max(3, 5), 5);\n/// }\nfn max(a: i32, b: i32) -> i32 {\n    if a > b { a } else { b }\n}\n\nconst MAX_SIZE: usize = 1024;\nstatic COUNTER: i32 = 0;\n\nstruct Point {\n    x: f64,\n    y: f64,\n}\n\n/// Returns the smaller of two values\nfn min(a: i32, b: i32) -> i32 {",
    },
    // Far: struct + impl block as padding
    DistancePrompt {
        name: "rs_far",
        category: "rust_test",
        distance: "far",
        marker: "#[test]",
        text: "/// Returns the larger of two values\n///\n/// #[test]\n/// fn test_max() {\n///     assert_eq!(max(3, 5), 5);\n/// }\nfn max(a: i32, b: i32) -> i32 {\n    if a > b { a } else { b }\n}\n\nconst MAX_SIZE: usize = 1024;\nstatic COUNTER: i32 = 0;\n\nstruct Point {\n    x: f64,\n    y: f64,\n}\n\nimpl Point {\n    fn new(x: f64, y: f64) -> Self {\n        Point { x, y }\n    }\n\n    fn distance(&self, other: &Point) -> f64 {\n        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()\n    }\n}\n\n/// Returns the smaller of two values\nfn min(a: i32, b: i32) -> i32 {",
    },
];

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("============================================================");
    println!("STATE STEERING PERSISTENCE — RWKV-6");
    println!("Distance x Scale x Temperature Experiment");
    println!("============================================================\n");

    // Load model
    println!("Loading model: {}", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;

    let layer = args.layer.unwrap_or(2);

    // Use CLI temperatures if provided, otherwise defaults
    let temperatures: Vec<f32> = if args.temperature.is_empty() {
        TEMPERATURES.to_vec()
    } else {
        args.temperature.clone()
    };

    println!("Architecture: {:?}", model.architecture());
    println!("Layers: {}", model.n_layers());
    println!("Steering layer: {}", layer);
    println!("Scales: {:?}", SCALES);
    println!("Temperatures: {:?}", temperatures);
    println!("Max tokens: {}", args.max_tokens);
    println!("Samples at temp>0: {}", args.n_samples);
    if args.rust_only {
        println!("Filter: Rust prompts only");
    }
    println!();

    let mut results: Vec<ConditionResult> = Vec::new();

    for prompt in PROMPTS {
        // Skip Python prompts if --rust-only
        if args.rust_only && prompt.category != "rust_test" {
            continue;
        }
        println!("------------------------------------------------------------");
        println!(
            "Prompt: {} ({}, distance={})",
            prompt.name, prompt.category, prompt.distance
        );
        println!("------------------------------------------------------------");

        // Find marker position in token space
        let marker_pos = find_marker_position(&model, prompt.text, prompt.marker)?;
        if marker_pos.is_none() {
            println!("  (No marker found, skipping)\n");
            continue;
        }
        let marker_pos = marker_pos.unwrap();

        // Compute token distance: marker → end of prompt
        let prompt_token_count = model.tokenize(prompt.text)?.len();
        let token_distance = prompt_token_count.saturating_sub(marker_pos + 1);

        println!(
            "  Marker at token {} / {} (distance to generation start: {} tokens)",
            marker_pos, prompt_token_count, token_distance
        );

        for &scale in SCALES {
            for &temp in &temperatures {
                let n_runs = if temp == 0.0 { 1 } else { args.n_samples };
                let mut test_hits = 0;

                for run in 0..n_runs {
                    let spec = StateSteeringSpec::new(scale)
                        .position(marker_pos)
                        .layer(layer);

                    let result = model.generate_with_state_steering_details(
                        prompt.text,
                        args.max_tokens,
                        temp,
                        &[],
                        &spec,
                    )?;

                    let has_test = output_contains_test_syntax(&result.generated_text);
                    if has_test {
                        test_hits += 1;
                    }

                    // Show first sample output for each condition
                    if run == 0 {
                        println!(
                            "  scale={:.1} temp={:.1}: {:?}{}",
                            scale,
                            temp,
                            truncate_output(&result.generated_text, 100),
                            if has_test { " [TEST]" } else { "" }
                        );
                    }
                }

                let rate = test_hits as f32 / n_runs as f32;
                if n_runs > 1 {
                    println!(
                        "    -> {}/{} samples contain test syntax ({:.0}%)",
                        test_hits,
                        n_runs,
                        rate * 100.0
                    );
                }

                results.push(ConditionResult {
                    prompt_name: prompt.name.to_string(),
                    distance: prompt.distance.to_string(),
                    category: prompt.category.to_string(),
                    scale,
                    temperature: temp,
                    n_runs,
                    test_hits,
                    rate,
                    token_distance,
                });
            }
        }
        println!();
    }

    // === Summary tables ===
    print_summary(&results);

    // === JSON output ===
    print_json(&results)?;

    Ok(())
}

struct ConditionResult {
    prompt_name: String,
    distance: String,
    #[allow(dead_code)]
    category: String,
    scale: f32,
    temperature: f32,
    n_runs: usize,
    test_hits: usize,
    rate: f32,
    token_distance: usize,
}

fn print_summary(results: &[ConditionResult]) {
    println!("\n============================================================");
    println!("SUMMARY — Test Syntax Rate by Condition");
    println!("============================================================\n");

    // Extract unique temperatures from results
    let mut temps: Vec<f32> = results.iter().map(|r| r.temperature).collect();
    temps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    temps.dedup();

    // One table per temperature
    for &temp in &temps {
        let temp_label = if temp == 0.0 {
            "greedy"
        } else {
            "sampling"
        };
        println!("Temperature = {:.1} ({}):", temp, temp_label);
        println!();

        // Header row
        print!("{:12} {:8} {:>5}", "Prompt", "Dist", "delta");
        for &scale in SCALES {
            print!(" | {:>7}", format!("x{:.1}", scale));
        }
        println!();

        // Separator
        print!("{:-<12} {:-<8} {:-<5}", "", "", "");
        for _ in SCALES {
            print!("-+-{:-<7}", "");
        }
        println!();

        // Data rows
        for prompt in PROMPTS {
            let row: Vec<&ConditionResult> = results
                .iter()
                .filter(|r| r.prompt_name == prompt.name && r.temperature == temp)
                .collect();

            if row.is_empty() {
                continue;
            }

            print!(
                "{:12} {:8} {:>5}",
                prompt.name, prompt.distance, row[0].token_distance
            );

            for &scale in SCALES {
                if let Some(r) = row.iter().find(|r| (r.scale - scale).abs() < 0.01) {
                    if r.n_runs == 1 {
                        // Greedy: show Yes/No
                        print!(
                            " | {:>7}",
                            if r.test_hits > 0 { "Yes" } else { "No" }
                        );
                    } else {
                        // Sampling: show percentage
                        print!(" | {:>5.0}% ", r.rate * 100.0);
                    }
                } else {
                    print!(" | {:>7}", "—");
                }
            }
            println!();
        }
        println!();
    }

    // === DISTANCE EFFECT ===
    println!("============================================================");
    println!("DISTANCE EFFECT (averaging steered conditions, scale > 1.0)");
    println!("============================================================\n");

    for &temp in &temps {
        let temp_label = if temp == 0.0 {
            "greedy"
        } else {
            "sampling"
        };
        println!("Temperature = {:.1} ({}):", temp, temp_label);

        for dist in &["close", "medium", "far"] {
            let steered: Vec<&ConditionResult> = results
                .iter()
                .filter(|r| r.distance == *dist && r.temperature == temp && r.scale > 1.5)
                .collect();

            if steered.is_empty() {
                continue;
            }

            let avg_rate: f32 =
                steered.iter().map(|r| r.rate).sum::<f32>() / steered.len() as f32;
            let avg_delta: f32 = steered.iter().map(|r| r.token_distance as f32).sum::<f32>()
                / steered.len() as f32;

            println!(
                "  {:8} (avg delta={:.0} tokens): test rate = {:.0}% (across {} conditions)",
                dist,
                avg_delta,
                avg_rate * 100.0,
                steered.len()
            );
        }
        println!();
    }

    // === SCALE EFFECT ===
    println!("============================================================");
    println!("SCALE EFFECT (averaging across distances)");
    println!("============================================================\n");

    for &temp in &temps {
        let temp_label = if temp == 0.0 {
            "greedy"
        } else {
            "sampling"
        };
        println!("Temperature = {:.1} ({}):", temp, temp_label);

        for &scale in SCALES {
            let matched: Vec<&ConditionResult> = results
                .iter()
                .filter(|r| (r.scale - scale).abs() < 0.01 && r.temperature == temp)
                .collect();

            if matched.is_empty() {
                continue;
            }

            let avg_rate: f32 =
                matched.iter().map(|r| r.rate).sum::<f32>() / matched.len() as f32;
            println!(
                "  x{:.1}: avg test rate = {:.0}% (across {} prompts)",
                scale,
                avg_rate * 100.0,
                matched.len()
            );
        }
        println!();
    }
}

fn print_json(results: &[ConditionResult]) -> Result<()> {
    println!("============================================================");
    println!("JSON OUTPUT");
    println!("============================================================\n");

    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "prompt": r.prompt_name,
                "distance": r.distance,
                "token_distance": r.token_distance,
                "scale": r.scale,
                "temperature": r.temperature,
                "n_runs": r.n_runs,
                "test_hits": r.test_hits,
                "rate": r.rate,
            })
        })
        .collect();

    let output = serde_json::json!({
        "experiment": "state_steering_persistence",
        "conditions": {
            "scales": SCALES,
            "temperatures": TEMPERATURES,
            "distances": ["close", "medium", "far"],
        },
        "results": json_results,
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}

/// Find the test marker position in token space.
fn find_marker_position(
    model: &PlipModel,
    text: &str,
    marker: &str,
) -> Result<Option<usize>> {
    if let Some(char_pos) = text.find(marker) {
        return model.char_to_token(text, char_pos);
    }
    Ok(None)
}

/// Check if generated output contains test-related syntax.
fn output_contains_test_syntax(text: &str) -> bool {
    text.contains("#[test]")
        || text.contains("assert_eq!")
        || text.contains("assert!")
        || text.contains("fn test_")
        || text.contains(">>>")
        || text.contains("doctest")
}

/// Truncate output for display (UTF-8 safe).
fn truncate_output(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        text.to_string()
    } else {
        let truncated: String = chars[..max_chars].iter().collect();
        format!("{}...", truncated)
    }
}
