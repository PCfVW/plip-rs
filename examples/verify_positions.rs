//! Token Position Verification for Any Model
//!
//! Verifies that corpus token positions are correct for the specified model's tokenizer.
//! Reports discrepancies and suggests corrections.
//!
//! Usage:
//!   cargo run --release --example verify_positions -- --model "Qwen/Qwen2.5-Coder-7B-Instruct"
//!   cargo run --release --example verify_positions -- --model "bigcode/starcoder2-3b"
//!   cargo run --release --example verify_positions -- --model "google/codegemma-7b-it"

use anyhow::{Context, Result};
use clap::Parser;
use plip_rs::PlipModel;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "verify_positions")]
#[command(about = "Verify token positions for a specific model's tokenizer")]
struct Args {
    /// Model to verify positions for
    #[arg(long, default_value = "Qwen/Qwen2.5-Coder-7B-Instruct")]
    model: String,

    /// Path to corpus JSON file
    #[arg(long, default_value = "corpus/attention_samples.json")]
    corpus: PathBuf,

    /// Output JSON file for corrected positions
    #[arg(long)]
    output: Option<PathBuf>,

    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,

    /// Show detailed token listings
    #[arg(long)]
    verbose: bool,

    /// Generate corrected corpus file
    #[arg(long)]
    fix: bool,
}

#[derive(Deserialize, Serialize, Clone)]
struct AttentionCorpus {
    python_doctest: Vec<PythonSample>,
    rust_test: Vec<RustSample>,
    python_baseline: Vec<PythonBaselineSample>,
    rust_baseline: Vec<RustBaselineSample>,
}

#[derive(Deserialize, Serialize, Clone)]
struct PythonSample {
    id: String,
    code: String,
    doctest_token_pos: usize,
    function_param_positions: Vec<usize>,
}

#[derive(Deserialize, Serialize, Clone)]
struct RustSample {
    id: String,
    code: String,
    test_attr_token_pos: usize,
    function_token_positions: Vec<usize>,
}

#[derive(Deserialize, Serialize, Clone)]
struct PythonBaselineSample {
    id: String,
    code: String,
    marker_token_pos: usize,
    function_param_positions: Vec<usize>,
}

#[derive(Deserialize, Serialize, Clone)]
struct RustBaselineSample {
    id: String,
    code: String,
    marker_token_pos: usize,
    struct_token_positions: Vec<usize>,
}

#[derive(Serialize)]
struct VerificationReport {
    model: String,
    total_samples: usize,
    correct_positions: usize,
    incorrect_positions: usize,
    samples: Vec<SampleVerification>,
}

#[derive(Serialize)]
struct SampleVerification {
    id: String,
    category: String,
    marker_correct: bool,
    expected_marker_pos: usize,
    actual_marker_pos: Option<usize>,
    marker_token: Option<String>,
    tokens_preview: Vec<String>,
}

/// Corrected corpus for a specific model's tokenizer
#[derive(Serialize)]
struct CorrectedCorpus {
    model: String,
    python_doctest: Vec<CorrectedPythonSample>,
    rust_test: Vec<CorrectedRustSample>,
    python_baseline: Vec<CorrectedPythonBaselineSample>,
    rust_baseline: Vec<CorrectedRustBaselineSample>,
}

#[derive(Serialize)]
struct CorrectedPythonSample {
    id: String,
    code: String,
    doctest_token_pos: usize,
    function_param_positions: Vec<usize>,
}

#[derive(Serialize)]
struct CorrectedRustSample {
    id: String,
    code: String,
    test_attr_token_pos: usize,
    function_token_positions: Vec<usize>,
}

#[derive(Serialize)]
struct CorrectedPythonBaselineSample {
    id: String,
    code: String,
    marker_token_pos: usize,
    function_param_positions: Vec<usize>,
}

#[derive(Serialize)]
struct CorrectedRustBaselineSample {
    id: String,
    code: String,
    marker_token_pos: usize,
    struct_token_positions: Vec<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  PLIP-rs: Token Position Verification");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Load model (for tokenizer)
    println!("Loading model: {} (for tokenizer only)...", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    println!("Model loaded: {} layers\n", model.n_layers());

    // Load corpus
    println!("Loading corpus from: {:?}\n", args.corpus);
    let corpus_json = fs::read_to_string(&args.corpus)
        .with_context(|| format!("Failed to read corpus file: {:?}", args.corpus))?;
    let corpus: AttentionCorpus =
        serde_json::from_str(&corpus_json).context("Failed to parse corpus JSON")?;

    let mut report = VerificationReport {
        model: args.model.clone(),
        total_samples: 0,
        correct_positions: 0,
        incorrect_positions: 0,
        samples: Vec::new(),
    };

    // Corrected corpus collections
    let mut corrected_python_doctest = Vec::new();
    let mut corrected_rust_test = Vec::new();
    let mut corrected_python_baseline = Vec::new();
    let mut corrected_rust_baseline = Vec::new();

    // Verify Python doctest samples
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PYTHON DOCTEST SAMPLES (looking for >>>)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for sample in &corpus.python_doctest {
        let (verification, corrected) =
            verify_and_correct_python_sample(&model, sample, "python_doctest", args.verbose)?;
        report.total_samples += 1;
        if verification.marker_correct {
            report.correct_positions += 1;
        } else {
            report.incorrect_positions += 1;
        }
        report.samples.push(verification);
        if let Some(c) = corrected {
            corrected_python_doctest.push(c);
        }
    }

    // Verify Rust test samples
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RUST TEST SAMPLES (looking for #[)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for sample in &corpus.rust_test {
        let (verification, corrected) =
            verify_and_correct_rust_sample(&model, sample, "rust_test", args.verbose)?;
        report.total_samples += 1;
        if verification.marker_correct {
            report.correct_positions += 1;
        } else {
            report.incorrect_positions += 1;
        }
        report.samples.push(verification);
        if let Some(c) = corrected {
            corrected_rust_test.push(c);
        }
    }

    // Verify Python baseline samples
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PYTHON BASELINE SAMPLES (looking for >>>)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for sample in &corpus.python_baseline {
        let (verification, corrected) =
            verify_and_correct_python_baseline(&model, sample, args.verbose)?;
        report.total_samples += 1;
        if verification.marker_correct {
            report.correct_positions += 1;
        } else {
            report.incorrect_positions += 1;
        }
        report.samples.push(verification);
        if let Some(c) = corrected {
            corrected_python_baseline.push(c);
        }
    }

    // Verify Rust baseline samples
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RUST BASELINE SAMPLES (looking for #[)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for sample in &corpus.rust_baseline {
        let (verification, corrected) =
            verify_and_correct_rust_baseline(&model, sample, args.verbose)?;
        report.total_samples += 1;
        if verification.marker_correct {
            report.correct_positions += 1;
        } else {
            report.incorrect_positions += 1;
        }
        report.samples.push(verification);
        if let Some(c) = corrected {
            corrected_rust_baseline.push(c);
        }
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  VERIFICATION SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("Model: {}", args.model);
    println!("Total samples: {}", report.total_samples);
    println!(
        "Correct positions: {} ({}%)",
        report.correct_positions,
        report.correct_positions * 100 / report.total_samples
    );
    println!(
        "Incorrect positions: {} ({}%)",
        report.incorrect_positions,
        report.incorrect_positions * 100 / report.total_samples
    );

    if report.incorrect_positions > 0 {
        println!("\n⚠️  Some positions need correction for this model!");
        println!("\nIncorrect samples:");
        for sample in &report.samples {
            if !sample.marker_correct {
                println!(
                    "  - {} ({}): expected pos {}, found {:?}",
                    sample.id,
                    sample.category,
                    sample.expected_marker_pos,
                    sample.actual_marker_pos
                );
            }
        }
    } else {
        println!("\n✓ All positions are correct for this model!");
    }

    // Save report if output specified
    if let Some(output_path) = &args.output {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(output_path, serde_json::to_string_pretty(&report)?)?;
        println!("\nReport saved to: {:?}", output_path);
    }

    // Generate corrected corpus if --fix specified
    if args.fix {
        let corrected = CorrectedCorpus {
            model: args.model.clone(),
            python_doctest: corrected_python_doctest,
            rust_test: corrected_rust_test,
            python_baseline: corrected_python_baseline,
            rust_baseline: corrected_rust_baseline,
        };

        let model_name = args.model.replace("/", "_").replace("-", "_");
        let output_path = PathBuf::from(format!("corpus/attention_samples_{}.json", model_name));

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&output_path, serde_json::to_string_pretty(&corrected)?)?;
        println!("\n═══════════════════════════════════════════════════════════════════");
        println!("  CORRECTED CORPUS GENERATED");
        println!("═══════════════════════════════════════════════════════════════════");
        println!("\nCorrected corpus saved to: {:?}", output_path);
        println!("Python doctest samples: {}", corrected.python_doctest.len());
        println!("Rust test samples: {}", corrected.rust_test.len());
        println!(
            "Python baseline samples: {}",
            corrected.python_baseline.len()
        );
        println!("Rust baseline samples: {}", corrected.rust_baseline.len());
    }

    Ok(())
}

fn verify_and_correct_python_sample(
    model: &PlipModel,
    sample: &PythonSample,
    category: &str,
    verbose: bool,
) -> Result<(SampleVerification, Option<CorrectedPythonSample>)> {
    let tokens = model.tokenize(&sample.code)?;

    // Find >>> marker
    let actual_pos = tokens.iter().position(|t| t.contains(">>>"));
    let marker_correct = actual_pos == Some(sample.doctest_token_pos);

    let status = if marker_correct { "✓" } else { "✗" };
    println!(
        "{} {}: expected pos {}, found {:?}",
        status, sample.id, sample.doctest_token_pos, actual_pos
    );

    if verbose || !marker_correct {
        println!("  Tokens around marker:");
        let start = actual_pos
            .unwrap_or(sample.doctest_token_pos)
            .saturating_sub(2);
        let end = (start + 8).min(tokens.len());
        for (i, token) in tokens.iter().enumerate().take(end).skip(start) {
            let marker = if Some(i) == actual_pos {
                " ← FOUND"
            } else if i == sample.doctest_token_pos {
                " ← EXPECTED"
            } else {
                ""
            };
            println!("    {:3}: {:?}{}", i, token, marker);
        }
    }

    // Create corrected sample if actual position was found
    let corrected = actual_pos.map(|actual| {
        // Calculate offset and apply to target positions
        let offset = actual as i64 - sample.doctest_token_pos as i64;
        let corrected_targets: Vec<usize> = sample
            .function_param_positions
            .iter()
            .filter_map(|&pos| {
                let new_pos = pos as i64 + offset;
                if new_pos >= 0 && (new_pos as usize) < tokens.len() {
                    Some(new_pos as usize)
                } else {
                    None
                }
            })
            .collect();

        CorrectedPythonSample {
            id: sample.id.clone(),
            code: sample.code.clone(),
            doctest_token_pos: actual,
            function_param_positions: corrected_targets,
        }
    });

    let verification = SampleVerification {
        id: sample.id.clone(),
        category: category.to_string(),
        marker_correct,
        expected_marker_pos: sample.doctest_token_pos,
        actual_marker_pos: actual_pos,
        marker_token: actual_pos.map(|i| tokens[i].clone()),
        tokens_preview: tokens.iter().take(15).cloned().collect(),
    };

    Ok((verification, corrected))
}

fn verify_and_correct_rust_sample(
    model: &PlipModel,
    sample: &RustSample,
    category: &str,
    verbose: bool,
) -> Result<(SampleVerification, Option<CorrectedRustSample>)> {
    let tokens = model.tokenize(&sample.code)?;

    // Find #[ marker (could be "#[" as single token or "#" followed by "[")
    let actual_pos = tokens.iter().position(|t| t.contains("#[")).or_else(|| {
        // Look for # followed by [
        tokens.iter().enumerate().find_map(|(i, t)| {
            if t.contains("#") && i + 1 < tokens.len() && tokens[i + 1].contains("[") {
                Some(i)
            } else {
                None
            }
        })
    });

    let marker_correct = actual_pos == Some(sample.test_attr_token_pos);

    let status = if marker_correct { "✓" } else { "✗" };
    println!(
        "{} {}: expected pos {}, found {:?}",
        status, sample.id, sample.test_attr_token_pos, actual_pos
    );

    if verbose || !marker_correct {
        println!("  Tokens around marker:");
        let start = actual_pos
            .unwrap_or(sample.test_attr_token_pos)
            .saturating_sub(2);
        let end = (start + 8).min(tokens.len());
        for (i, token) in tokens.iter().enumerate().take(end).skip(start) {
            let marker = if Some(i) == actual_pos {
                " ← FOUND"
            } else if i == sample.test_attr_token_pos {
                " ← EXPECTED"
            } else {
                ""
            };
            println!("    {:3}: {:?}{}", i, token, marker);
        }
    }

    // Create corrected sample if actual position was found
    let corrected = actual_pos.map(|actual| {
        // Calculate offset and apply to target positions
        let offset = actual as i64 - sample.test_attr_token_pos as i64;
        let corrected_targets: Vec<usize> = sample
            .function_token_positions
            .iter()
            .filter_map(|&pos| {
                let new_pos = pos as i64 + offset;
                if new_pos >= 0 && (new_pos as usize) < tokens.len() {
                    Some(new_pos as usize)
                } else {
                    None
                }
            })
            .collect();

        CorrectedRustSample {
            id: sample.id.clone(),
            code: sample.code.clone(),
            test_attr_token_pos: actual,
            function_token_positions: corrected_targets,
        }
    });

    let verification = SampleVerification {
        id: sample.id.clone(),
        category: category.to_string(),
        marker_correct,
        expected_marker_pos: sample.test_attr_token_pos,
        actual_marker_pos: actual_pos,
        marker_token: actual_pos.map(|i| tokens[i].clone()),
        tokens_preview: tokens.iter().take(15).cloned().collect(),
    };

    Ok((verification, corrected))
}

fn verify_and_correct_python_baseline(
    model: &PlipModel,
    sample: &PythonBaselineSample,
    verbose: bool,
) -> Result<(SampleVerification, Option<CorrectedPythonBaselineSample>)> {
    let tokens = model.tokenize(&sample.code)?;

    let actual_pos = tokens.iter().position(|t| t.contains(">>>"));
    let marker_correct = actual_pos == Some(sample.marker_token_pos);

    let status = if marker_correct { "✓" } else { "✗" };
    println!(
        "{} {}: expected pos {}, found {:?}",
        status, sample.id, sample.marker_token_pos, actual_pos
    );

    if verbose || !marker_correct {
        println!("  Tokens around marker:");
        let start = actual_pos
            .unwrap_or(sample.marker_token_pos)
            .saturating_sub(2);
        let end = (start + 8).min(tokens.len());
        for (i, token) in tokens.iter().enumerate().take(end).skip(start) {
            let marker = if Some(i) == actual_pos {
                " ← FOUND"
            } else if i == sample.marker_token_pos {
                " ← EXPECTED"
            } else {
                ""
            };
            println!("    {:3}: {:?}{}", i, token, marker);
        }
    }

    // Create corrected sample if actual position was found
    let corrected = actual_pos.map(|actual| {
        let offset = actual as i64 - sample.marker_token_pos as i64;
        let corrected_targets: Vec<usize> = sample
            .function_param_positions
            .iter()
            .filter_map(|&pos| {
                let new_pos = pos as i64 + offset;
                if new_pos >= 0 && (new_pos as usize) < tokens.len() {
                    Some(new_pos as usize)
                } else {
                    None
                }
            })
            .collect();

        CorrectedPythonBaselineSample {
            id: sample.id.clone(),
            code: sample.code.clone(),
            marker_token_pos: actual,
            function_param_positions: corrected_targets,
        }
    });

    let verification = SampleVerification {
        id: sample.id.clone(),
        category: "python_baseline".to_string(),
        marker_correct,
        expected_marker_pos: sample.marker_token_pos,
        actual_marker_pos: actual_pos,
        marker_token: actual_pos.map(|i| tokens[i].clone()),
        tokens_preview: tokens.iter().take(15).cloned().collect(),
    };

    Ok((verification, corrected))
}

fn verify_and_correct_rust_baseline(
    model: &PlipModel,
    sample: &RustBaselineSample,
    verbose: bool,
) -> Result<(SampleVerification, Option<CorrectedRustBaselineSample>)> {
    let tokens = model.tokenize(&sample.code)?;

    let actual_pos = tokens.iter().position(|t| t.contains("#[")).or_else(|| {
        tokens.iter().enumerate().find_map(|(i, t)| {
            if t.contains("#") && i + 1 < tokens.len() && tokens[i + 1].contains("[") {
                Some(i)
            } else {
                None
            }
        })
    });

    let marker_correct = actual_pos == Some(sample.marker_token_pos);

    let status = if marker_correct { "✓" } else { "✗" };
    println!(
        "{} {}: expected pos {}, found {:?}",
        status, sample.id, sample.marker_token_pos, actual_pos
    );

    if verbose || !marker_correct {
        println!("  Tokens around marker:");
        let start = actual_pos
            .unwrap_or(sample.marker_token_pos)
            .saturating_sub(2);
        let end = (start + 8).min(tokens.len());
        for (i, token) in tokens.iter().enumerate().take(end).skip(start) {
            let marker = if Some(i) == actual_pos {
                " ← FOUND"
            } else if i == sample.marker_token_pos {
                " ← EXPECTED"
            } else {
                ""
            };
            println!("    {:3}: {:?}{}", i, token, marker);
        }
    }

    // Create corrected sample if actual position was found
    let corrected = actual_pos.map(|actual| {
        let offset = actual as i64 - sample.marker_token_pos as i64;
        let corrected_targets: Vec<usize> = sample
            .struct_token_positions
            .iter()
            .filter_map(|&pos| {
                let new_pos = pos as i64 + offset;
                if new_pos >= 0 && (new_pos as usize) < tokens.len() {
                    Some(new_pos as usize)
                } else {
                    None
                }
            })
            .collect();

        CorrectedRustBaselineSample {
            id: sample.id.clone(),
            code: sample.code.clone(),
            marker_token_pos: actual,
            struct_token_positions: corrected_targets,
        }
    });

    let verification = SampleVerification {
        id: sample.id.clone(),
        category: "rust_baseline".to_string(),
        marker_correct,
        expected_marker_pos: sample.marker_token_pos,
        actual_marker_pos: actual_pos,
        marker_token: actual_pos.map(|i| tokens[i].clone()),
        tokens_preview: tokens.iter().take(15).cloned().collect(),
    };

    Ok((verification, corrected))
}
