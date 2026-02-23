//! Prefix-stability probe: does the model commit to a rhyme word early?
//!
//! For each successful couplet, progressively reveals more of line 2 and
//! asks the model to complete the rest greedily.  Records the ending word
//! at each prefix length to see when/if it stabilises.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example ollama_prefix_stability
//! ```

use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Prefix-stability probe for couplet rhyming")]
struct Args {
    /// Path to write the results JSON.
    #[arg(long, default_value = "outputs/llama_prefix_stability.json")]
    output: String,
}

// ---------------------------------------------------------------------------
// Couplet definitions (the 10 successful couplets from the baseline)
// ---------------------------------------------------------------------------

struct Couplet {
    id: u32,
    rhyme_family_size: &'static str,
    target_word: &'static str,
    line1: &'static str,
    baseline_ending: &'static str,
}

fn successful_couplets() -> Vec<Couplet> {
    vec![
        Couplet {
            id: 1,
            rhyme_family_size: "large",
            target_word: "light",
            line1: "The moon casts silver light,",
            baseline_ending: "night",
        },
        Couplet {
            id: 2,
            rhyme_family_size: "large",
            target_word: "play",
            line1: "The children laugh and play,",
            baseline_ending: "sway",
        },
        Couplet {
            id: 3,
            rhyme_family_size: "large",
            target_word: "sound",
            line1: "The thunder makes a sound,",
            baseline_ending: "around",
        },
        Couplet {
            id: 4,
            rhyme_family_size: "large",
            target_word: "rain",
            line1: "The clouds bring heavy rain,",
            baseline_ending: "vain",
        },
        Couplet {
            id: 6,
            rhyme_family_size: "medium",
            target_word: "air",
            line1: "The geese fly through the air,",
            baseline_ending: "fair",
        },
        Couplet {
            id: 7,
            rhyme_family_size: "medium",
            target_word: "gold",
            line1: "The sunset gleams like gold,",
            baseline_ending: "hold",
        },
        Couplet {
            id: 8,
            rhyme_family_size: "medium",
            target_word: "fire",
            line1: "The embers feed the fire,",
            baseline_ending: "acquire",
        },
        Couplet {
            id: 13,
            rhyme_family_size: "small",
            target_word: "truth",
            line1: "She spoke the honest truth,",
            baseline_ending: "youth",
        },
        Couplet {
            id: 14,
            rhyme_family_size: "small",
            target_word: "world",
            line1: "He traveled all the world,",
            baseline_ending: "unfurl",
        },
        Couplet {
            id: 15,
            rhyme_family_size: "small",
            target_word: "earth",
            line1: "The seeds lay in the earth,",
            baseline_ending: "birth",
        },
    ]
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct StabilityResult {
    id: u32,
    rhyme_family_size: String,
    target_word: String,
    line1: String,
    baseline_ending: String,
    full_response: String,
    forks: Vec<ForkResult>,
    stabilises_at: Option<usize>,
}

#[derive(Serialize)]
struct ForkResult {
    prefix_words: usize,
    prefix_text: String,
    completion: String,
    ending_word: String,
    rhymes_with_target: bool,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

// ---------------------------------------------------------------------------
// Ollama helpers
// ---------------------------------------------------------------------------

const OLLAMA_URL: &str = "http://localhost:11434/api/generate";
const MODEL: &str = "llama3.2:1b";

/// Build the raw prompt with the Llama 3.2 chat template and an optional
/// assistant prefix (the partial line 2 we want the model to continue from).
fn build_raw_prompt(user_msg: &str, assistant_prefix: Option<&str>) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
    prompt.push_str("Cutting Knowledge Date: December 2023\n\n");
    prompt.push_str("<|eot_id|>");
    prompt.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
    prompt.push_str(user_msg);
    prompt.push_str("<|eot_id|>");
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    if let Some(prefix) = assistant_prefix {
        prompt.push_str(prefix);
    }
    prompt
}

/// Call Ollama with a raw prompt, greedy decoding (temperature=0).
fn call_ollama_greedy(raw_prompt: &str, max_tokens: u32) -> Result<OllamaResponse> {
    let body = serde_json::json!({
        "model": MODEL,
        "prompt": raw_prompt,
        "stream": false,
        "raw": true,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.0
        }
    });

    let resp = ureq::post(OLLAMA_URL)
        .set("Content-Type", "application/json")
        .send_json(&body)
        .context("Ollama unreachable")?;

    resp.into_json().context("Failed to parse Ollama response")
}

/// Extract the last word from a line of text (strip trailing punctuation).
fn last_word(text: &str) -> String {
    text.split_whitespace()
        .next_back()
        .unwrap_or("")
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase()
}

/// Naive rhyme check: do the last 2+ characters match?
fn naive_rhymes(word_a: &str, word_b: &str) -> bool {
    let a = word_a.to_lowercase();
    let b = word_b.to_lowercase();
    if a == b {
        return true;
    }
    let suffix_len = 2.min(a.len()).min(b.len());
    if suffix_len < 2 {
        return false;
    }
    if a[a.len() - suffix_len..] == b[b.len() - suffix_len..] {
        return true;
    }
    if a.len() >= 3 && b.len() >= 3 && a[a.len() - 3..] == b[b.len() - 3..] {
        return true;
    }
    false
}

/// Find the first prefix position from which the ending word never changes.
fn first_stable_position(forks: &[ForkResult]) -> Option<usize> {
    forks.iter().position(|f| {
        forks[f.prefix_words..]
            .iter()
            .all(|r| r.ending_word == forks[f.prefix_words].ending_word)
    })
}

// ---------------------------------------------------------------------------
// Per-couplet analysis
// ---------------------------------------------------------------------------

fn analyse_couplet(couplet: &Couplet) -> Result<StabilityResult> {
    println!(
        "=== Couplet {:>2}  [{:<6}]  '{}' (target: {}) ===",
        couplet.id, couplet.rhyme_family_size, couplet.line1, couplet.target_word
    );

    let user_msg = format!(
        "Write exactly one line of poetry that rhymes with this line:\n{}",
        couplet.line1
    );

    // Step 1: Generate full line 2 with greedy decoding.
    let full_prompt = build_raw_prompt(&user_msg, None);
    let full_resp = call_ollama_greedy(&full_prompt, 30)?;
    let full_line = full_resp.response.trim().to_string();
    let full_first_line = full_line.lines().next().unwrap_or("").to_string();
    let full_ending = last_word(&full_first_line);

    println!("  Full response: {full_first_line}");
    println!("  Ending word:   {full_ending}\n");

    // Step 2: Fork at each prefix length.
    let words: Vec<&str> = full_first_line.split_whitespace().collect();
    let mut forks: Vec<ForkResult> = Vec::new();

    for prefix_len in 0..words.len() {
        let prefix_text = if prefix_len == 0 {
            String::new()
        } else {
            words[..prefix_len].join(" ") + " "
        };

        let fork_prompt = build_raw_prompt(&user_msg, Some(&prefix_text));
        let start = Instant::now();
        let fork_resp = call_ollama_greedy(&fork_prompt, 30)?;
        let elapsed = start.elapsed().as_millis();

        let completion_raw = fork_resp.response.trim().to_string();
        let completion_first_line = completion_raw.lines().next().unwrap_or("");
        let full_line_reconstructed = format!("{prefix_text}{completion_first_line}");
        let ending = last_word(&full_line_reconstructed);
        let rhymes = naive_rhymes(&ending, couplet.target_word);

        let rhyme_marker = if rhymes { "Y" } else { " " };
        print!("  prefix={prefix_len:<2}  [{rhyme_marker}]  \"{ending:<12}\"");
        println!("  ({elapsed} ms)  {full_line_reconstructed}");

        forks.push(ForkResult {
            prefix_words: prefix_len,
            prefix_text: prefix_text.trim().to_string(),
            completion: completion_first_line.to_string(),
            ending_word: ending,
            rhymes_with_target: rhymes,
        });
    }

    let stabilises_at = first_stable_position(&forks);
    if let Some(pos) = stabilises_at {
        println!(
            "  --> Ending word stabilises at prefix={pos} ('{}')\n",
            forks[pos].ending_word
        );
    } else {
        println!("  --> Ending word NEVER stabilises\n");
    }

    Ok(StabilityResult {
        id: couplet.id,
        rhyme_family_size: couplet.rhyme_family_size.to_string(),
        target_word: couplet.target_word.to_string(),
        line1: couplet.line1.to_string(),
        baseline_ending: couplet.baseline_ending.to_string(),
        full_response: full_first_line,
        forks,
        stabilises_at,
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();
    let output_path = &args.output;

    let output_dir = Path::new(output_path.as_str())
        .parent()
        .context("invalid output path")?;
    fs::create_dir_all(output_dir)?;

    if ureq::get("http://localhost:11434/api/tags").call().is_err() {
        bail!("Cannot reach Ollama at localhost:11434. Is it running?");
    }
    println!("Ollama is reachable.\n");

    let couplets = successful_couplets();
    let mut results: Vec<StabilityResult> = Vec::new();

    for couplet in &couplets {
        results.push(analyse_couplet(couplet)?);
    }

    // Write results.
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(output_path, &json)?;
    println!("Results written to {output_path}");

    // Print summary.
    println!("\n{:-<60}", "");
    println!(
        "{:<4} {:<7} {:<8} {:<10} Stabilises at",
        "ID", "Family", "Target", "Full end"
    );
    println!("{:-<60}", "");
    for r in &results {
        let stab = match r.stabilises_at {
            Some(pos) => format!("prefix={pos}"),
            None => "NEVER".to_string(),
        };
        println!(
            "{:<4} {:<7} {:<8} {:<10} {stab}",
            r.id,
            r.rhyme_family_size,
            r.target_word,
            r.full_response.split_whitespace().next_back().unwrap_or(""),
        );
    }

    Ok(())
}
