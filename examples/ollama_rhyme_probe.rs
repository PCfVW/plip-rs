//! Behavioral baseline: probe Llama 3.2 1B rhyming ability via Ollama.
//!
//! Reads a probe JSON file, sends each prompt to the local Ollama instance,
//! and writes structured results to a JSON output file.
//!
//! # Usage
//!
//! ```bash
//! # v1 — basic rhyme probes (default)
//! cargo run --release --example ollama_rhyme_probe
//!
//! # v2 — strategy probes (couplet, few-shot, CoT, explicit, HTML, system)
//! cargo run --release --example ollama_rhyme_probe -- \
//!     --probes corpus/llama_rhyme_probes_v2.json \
//!     --output outputs/llama_rhyme_results_v2.json
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
#[command(about = "Probe Ollama rhyming ability")]
struct Args {
    /// Path to the probe JSON file.
    #[arg(long, default_value = "corpus/llama_rhyme_probes.json")]
    probes: String,

    /// Path to write the results JSON.
    #[arg(long, default_value = "outputs/llama_rhyme_results.json")]
    output: String,
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ProbeSet {
    model: String,
    probes: Vec<Probe>,
}

#[derive(Deserialize, Clone)]
struct Probe {
    id: u32,
    rhyme_family_size: String,
    scheme: String,
    target_word: String,
    prompt: String,

    // v1 compatibility: "style" field (instructed / raw / anti-rewrite).
    #[serde(default)]
    style: Option<String>,

    // v2 fields.
    #[serde(default)]
    strategy: Option<String>,
    #[serde(default)]
    raw_mode: Option<bool>,
    #[serde(default)]
    system_prompt: Option<String>,
}

impl Probe {
    /// Human-readable label for the strategy / style column.
    fn label(&self) -> String {
        self.strategy
            .clone()
            .or_else(|| self.style.clone())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Whether to send the prompt in raw mode (no chat template).
    fn is_raw(&self) -> bool {
        if let Some(raw) = self.raw_mode {
            return raw;
        }
        // v1 fallback: "raw" style means raw mode.
        self.style.as_deref() == Some("raw")
    }
}

#[derive(Serialize)]
struct ProbeResult {
    id: u32,
    rhyme_family_size: String,
    scheme: String,
    target_word: String,
    strategy: String,
    prompt: String,
    response: String,
    response_trimmed: String,
    duration_ms: u64,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

// ---------------------------------------------------------------------------
// Ollama API
// ---------------------------------------------------------------------------

fn call_ollama(
    model: &str,
    prompt: &str,
    raw: bool,
    system_prompt: Option<&str>,
) -> Result<OllamaResponse> {
    let mut body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "raw": raw,
        "options": {
            "num_predict": 120,
            "temperature": 0.7
        }
    });

    if let Some(sys) = system_prompt {
        body["system"] = serde_json::Value::String(sys.to_string());
    }

    let resp = ureq::post("http://localhost:11434/api/generate")
        .set("Content-Type", "application/json")
        .send_json(&body)
        .context("Failed to reach Ollama \u{2014} is it running on localhost:11434?")?;

    let ollama: OllamaResponse = resp
        .into_json()
        .context("Failed to parse Ollama response")?;

    Ok(ollama)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    let probe_path = &args.probes;
    let output_path = &args.output;

    // Ensure output directory exists.
    let output_dir = Path::new(output_path.as_str())
        .parent()
        .context("invalid output path")?;
    fs::create_dir_all(output_dir)?;

    // Load probes.
    let raw_json =
        fs::read_to_string(probe_path).with_context(|| format!("reading {probe_path}"))?;
    let probe_set: ProbeSet =
        serde_json::from_str(&raw_json).with_context(|| format!("parsing {probe_path}"))?;

    println!(
        "Loaded {} probes for model '{}'",
        probe_set.probes.len(),
        probe_set.model
    );

    // Quick connectivity check.
    let check = ureq::get("http://localhost:11434/api/tags").call();
    if check.is_err() {
        bail!("Cannot reach Ollama at localhost:11434. Is it running?");
    }
    println!("Ollama is reachable.\n");

    let mut results: Vec<ProbeResult> = Vec::with_capacity(probe_set.probes.len());

    for probe in &probe_set.probes {
        let label = probe.label();
        print!(
            "Probe {:>2}  [{:<6} {:>7} {:<16}]  target='{}'",
            probe.id, probe.rhyme_family_size, probe.scheme, label, probe.target_word,
        );

        let raw = probe.is_raw();
        let sys = probe.system_prompt.as_deref();

        let start = Instant::now();
        let ollama = call_ollama(&probe_set.model, &probe.prompt, raw, sys)?;
        let elapsed = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

        let trimmed = ollama.response.trim().to_string();
        let first_line = trimmed.lines().next().unwrap_or("");

        println!("  ({elapsed} ms)");
        println!("  -> {first_line}");
        let line_count = trimmed.lines().count();
        if line_count > 1 {
            println!("     ({} more lines)", line_count - 1);
        }
        println!();

        results.push(ProbeResult {
            id: probe.id,
            rhyme_family_size: probe.rhyme_family_size.clone(),
            scheme: probe.scheme.clone(),
            target_word: probe.target_word.clone(),
            strategy: label,
            prompt: probe.prompt.clone(),
            response: ollama.response,
            response_trimmed: trimmed,
            duration_ms: elapsed,
        });
    }

    // Write results.
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(output_path, &json)?;
    println!("Results written to {output_path}");

    // Print summary table.
    println!("\n{:-<80}", "");
    println!(
        "{:<4} {:<7} {:<8} {:<17} {:<8} First line of response",
        "ID", "Family", "Scheme", "Strategy", "Target"
    );
    println!("{:-<80}", "");
    for r in &results {
        let first = r.response_trimmed.lines().next().unwrap_or("");
        let truncated = if first.len() > 40 {
            format!("{}...", &first[..37])
        } else {
            first.to_string()
        };
        println!(
            "{:<4} {:<7} {:<8} {:<17} {:<8} {}",
            r.id, r.rhyme_family_size, r.scheme, r.strategy, r.target_word, truncated,
        );
    }

    Ok(())
}
