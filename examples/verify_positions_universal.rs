//! Universal Position Verification
//!
//! Verifies that character positions in the universal corpus correctly convert
//! to token positions for any model. This replaces the need for model-specific
//! corpus files.
//!
//! Usage:
//!   cargo run --release --example verify_positions_universal -- --model "Qwen/Qwen2.5-Coder-7B-Instruct"
//!   cargo run --release --example verify_positions_universal -- --model "bigcode/starcoder2-3b"
//!   cargo run --release --example verify_positions_universal -- --model "google/codegemma-7b-it"

use anyhow::{Context, Result};
use clap::Parser;
use plip_rs::PlipModel;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "verify_positions_universal")]
#[command(about = "Verify character position to token conversion for any model")]
struct Args {
    /// Model to verify positions for
    #[arg(long, default_value = "Qwen/Qwen2.5-Coder-7B-Instruct")]
    model: String,

    /// Path to universal corpus JSON file
    #[arg(long, default_value = "corpus/attention_samples_universal.json")]
    corpus: PathBuf,

    /// Output JSON file for verification report
    #[arg(long)]
    output: Option<PathBuf>,

    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,

    /// Show detailed token listings
    #[arg(long)]
    verbose: bool,
}

/// Universal corpus format with character positions
#[derive(Deserialize)]
struct UniversalCorpus {
    #[serde(default)]
    _format_version: Option<String>,
    #[serde(default)]
    _description: Option<String>,
    python_doctest: Vec<UniversalSample>,
    rust_test: Vec<UniversalSample>,
    #[serde(default)]
    python_baseline: Vec<UniversalSample>,
    #[serde(default)]
    rust_baseline: Vec<UniversalSample>,
}

#[derive(Deserialize)]
struct UniversalSample {
    id: String,
    code: String,
    marker_char_pos: usize,
    marker_pattern: String,
    target_char_positions: Vec<usize>,
}

#[derive(Serialize)]
struct VerificationReport {
    model: String,
    corpus_format: String,
    total_samples: usize,
    successful_conversions: usize,
    partial_conversions: usize,
    failed_conversions: usize,
    samples: Vec<SampleVerification>,
}

#[derive(Serialize)]
struct SampleVerification {
    id: String,
    category: String,
    marker_conversion: PositionVerification,
    target_conversions: Vec<PositionVerification>,
    overall_status: String,
}

#[derive(Serialize)]
struct PositionVerification {
    char_pos: usize,
    token_pos: Option<usize>,
    token_text: Option<String>,
    expected_pattern: Option<String>,
    pattern_found: bool,
    exact_match: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Universal Position Verification");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Load model
    println!("Loading model: {} (for tokenizer)...", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    println!("Model loaded: {} layers\n", model.n_layers());

    // Load corpus
    println!("Loading universal corpus from: {:?}\n", args.corpus);
    let corpus_json = fs::read_to_string(&args.corpus)
        .with_context(|| format!("Failed to read corpus file: {:?}", args.corpus))?;
    let corpus: UniversalCorpus =
        serde_json::from_str(&corpus_json).context("Failed to parse corpus JSON")?;

    let format_version = corpus._format_version.as_deref().unwrap_or("1.0");

    let mut report = VerificationReport {
        model: args.model.clone(),
        corpus_format: format!("universal_v{}", format_version),
        total_samples: 0,
        successful_conversions: 0,
        partial_conversions: 0,
        failed_conversions: 0,
        samples: Vec::new(),
    };

    // Verify Python doctest samples
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PYTHON DOCTEST SAMPLES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for sample in &corpus.python_doctest {
        let verification = verify_sample(&model, sample, "python_doctest", args.verbose)?;
        update_stats(&mut report, &verification);
        report.samples.push(verification);
    }

    // Verify Rust test samples
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RUST TEST SAMPLES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for sample in &corpus.rust_test {
        let verification = verify_sample(&model, sample, "rust_test", args.verbose)?;
        update_stats(&mut report, &verification);
        report.samples.push(verification);
    }

    // Verify Python baseline samples
    if !corpus.python_baseline.is_empty() {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("PYTHON BASELINE SAMPLES");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for sample in &corpus.python_baseline {
            let verification = verify_sample(&model, sample, "python_baseline", args.verbose)?;
            update_stats(&mut report, &verification);
            report.samples.push(verification);
        }
    }

    // Verify Rust baseline samples
    if !corpus.rust_baseline.is_empty() {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("RUST BASELINE SAMPLES");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for sample in &corpus.rust_baseline {
            let verification = verify_sample(&model, sample, "rust_baseline", args.verbose)?;
            update_stats(&mut report, &verification);
            report.samples.push(verification);
        }
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  VERIFICATION SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("Model: {}", args.model);
    println!("Corpus format: universal_v{}", format_version);
    println!("Total samples: {}", report.total_samples);
    println!(
        "Successful conversions: {} ({:.1}%)",
        report.successful_conversions,
        100.0 * report.successful_conversions as f64 / report.total_samples as f64
    );
    println!(
        "Partial conversions: {} ({:.1}%)",
        report.partial_conversions,
        100.0 * report.partial_conversions as f64 / report.total_samples as f64
    );
    println!(
        "Failed conversions: {} ({:.1}%)",
        report.failed_conversions,
        100.0 * report.failed_conversions as f64 / report.total_samples as f64
    );

    if report.failed_conversions == 0 && report.partial_conversions == 0 {
        println!("\n✓ All positions convert correctly for this model!");
        println!(
            "  The universal corpus works perfectly with {}.",
            args.model
        );
    } else if report.failed_conversions == 0 {
        println!("\n⚠  All markers convert, but some targets used fuzzy matching.");
        println!("  This is usually acceptable for attention analysis.");
    } else {
        println!("\n⚠️  Some samples have conversion issues.");
        println!("\nFailed samples:");
        for sample in &report.samples {
            if sample.overall_status == "failed" {
                println!("  - {} ({})", sample.id, sample.category);
            }
        }
    }

    // Save report if output specified
    if let Some(output_path) = &args.output {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(output_path, serde_json::to_string_pretty(&report)?)?;
        println!("\nReport saved to: {:?}", output_path);
    }

    Ok(())
}

fn verify_sample(
    model: &PlipModel,
    sample: &UniversalSample,
    category: &str,
    verbose: bool,
) -> Result<SampleVerification> {
    let encoding = model.tokenize_with_offsets(&sample.code)?;
    let tokens = &encoding.tokens;

    // Verify marker position
    let marker_token_pos = encoding.char_to_token(sample.marker_char_pos);
    let marker_exact = marker_token_pos.is_some();

    let marker_token_pos =
        marker_token_pos.or_else(|| encoding.char_to_token_fuzzy(sample.marker_char_pos));

    let marker_token_text = marker_token_pos.map(|pos| tokens[pos].clone());

    // Check if the marker pattern is in the token
    let marker_pattern_found = marker_token_text
        .as_ref()
        .map(|t| t.contains(&sample.marker_pattern) || sample.marker_pattern.starts_with(&**t))
        .unwrap_or(false);

    let marker_verification = PositionVerification {
        char_pos: sample.marker_char_pos,
        token_pos: marker_token_pos,
        token_text: marker_token_text.clone(),
        expected_pattern: Some(sample.marker_pattern.clone()),
        pattern_found: marker_pattern_found,
        exact_match: marker_exact,
    };

    // Verify target positions
    let mut target_verifications = Vec::new();
    for &char_pos in &sample.target_char_positions {
        let token_pos = encoding.char_to_token(char_pos);
        let exact = token_pos.is_some();
        let token_pos = token_pos.or_else(|| encoding.char_to_token_fuzzy(char_pos));
        let token_text = token_pos.map(|pos| tokens[pos].clone());

        target_verifications.push(PositionVerification {
            char_pos,
            token_pos,
            token_text,
            expected_pattern: None,
            pattern_found: token_pos.is_some(),
            exact_match: exact,
        });
    }

    // Determine overall status
    let all_targets_converted = target_verifications.iter().all(|v| v.token_pos.is_some());
    let all_exact = marker_exact && target_verifications.iter().all(|v| v.exact_match);

    let overall_status = if marker_token_pos.is_none() {
        "failed"
    } else if !all_targets_converted {
        "partial"
    } else if all_exact && marker_pattern_found {
        "perfect"
    } else {
        "ok"
    };

    // Print status
    let status_icon = match overall_status {
        "perfect" => "✓",
        "ok" => "~",
        "partial" => "⚠",
        _ => "✗",
    };

    println!(
        "{} {}: marker@{} → token {} {:?}",
        status_icon,
        sample.id,
        sample.marker_char_pos,
        marker_token_pos
            .map(|p| p.to_string())
            .unwrap_or("?".to_string()),
        marker_token_text.as_deref().unwrap_or("?")
    );

    if verbose {
        println!(
            "  Marker pattern '{}' {}",
            sample.marker_pattern,
            if marker_pattern_found {
                "found"
            } else {
                "NOT FOUND"
            }
        );
        println!("  Target conversions:");
        for (i, v) in target_verifications.iter().enumerate() {
            println!(
                "    [{}] char {} → token {} {:?} {}",
                i,
                v.char_pos,
                v.token_pos
                    .map(|p| p.to_string())
                    .unwrap_or("?".to_string()),
                v.token_text.as_deref().unwrap_or("?"),
                if v.exact_match { "(exact)" } else { "(fuzzy)" }
            );
        }

        // Show token context around marker
        if let Some(marker_pos) = marker_token_pos {
            println!("  Token context:");
            let start = marker_pos.saturating_sub(2);
            let end = (marker_pos + 5).min(tokens.len());
            for (i, token) in tokens.iter().enumerate().take(end).skip(start) {
                let marker = if i == marker_pos { " ← MARKER" } else { "" };
                println!("    {:3}: {:?}{}", i, token, marker);
            }
        }
    }

    Ok(SampleVerification {
        id: sample.id.clone(),
        category: category.to_string(),
        marker_conversion: marker_verification,
        target_conversions: target_verifications,
        overall_status: overall_status.to_string(),
    })
}

fn update_stats(report: &mut VerificationReport, verification: &SampleVerification) {
    report.total_samples += 1;
    match verification.overall_status.as_str() {
        "perfect" | "ok" => report.successful_conversions += 1,
        "partial" => report.partial_conversions += 1,
        _ => report.failed_conversions += 1,
    }
}
