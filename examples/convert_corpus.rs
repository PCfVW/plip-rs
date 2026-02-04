//! Corpus Format Converter
//!
//! Converts legacy token-position corpus to the universal character-position format.
//! This enables migration from model-specific corpus files to the universal format.
//!
//! Usage:
//!   cargo run --release --example convert_corpus -- --input corpus/attention_samples.json --output corpus/attention_samples_universal.json

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "convert_corpus")]
#[command(about = "Convert legacy corpus to universal character-position format")]
struct Args {
    /// Input legacy corpus file
    #[arg(long)]
    input: PathBuf,

    /// Output universal corpus file
    #[arg(long)]
    output: PathBuf,

    /// Show detailed conversion info
    #[arg(long)]
    verbose: bool,
}

// Legacy format structures (token positions are read by serde but we recalculate from code)
#[derive(Deserialize)]
#[allow(dead_code)]
struct LegacyCorpus {
    python_doctest: Vec<LegacyPythonSample>,
    rust_test: Vec<LegacyRustSample>,
    #[serde(default)]
    python_baseline: Vec<LegacyPythonBaselineSample>,
    #[serde(default)]
    rust_baseline: Vec<LegacyRustBaselineSample>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct LegacyPythonSample {
    id: String,
    code: String,
    doctest_token_pos: usize,
    function_param_positions: Vec<usize>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct LegacyRustSample {
    id: String,
    code: String,
    test_attr_token_pos: usize,
    function_token_positions: Vec<usize>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct LegacyPythonBaselineSample {
    id: String,
    code: String,
    marker_token_pos: usize,
    function_param_positions: Vec<usize>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct LegacyRustBaselineSample {
    id: String,
    code: String,
    marker_token_pos: usize,
    struct_token_positions: Vec<usize>,
}

// Universal format structures
#[derive(Serialize)]
struct UniversalCorpus {
    _format_version: String,
    _description: String,
    python_doctest: Vec<UniversalSample>,
    rust_test: Vec<UniversalSample>,
    python_baseline: Vec<UniversalSample>,
    rust_baseline: Vec<UniversalSample>,
}

#[derive(Serialize)]
struct UniversalSample {
    id: String,
    code: String,
    marker_char_pos: usize,
    marker_pattern: String,
    target_char_positions: Vec<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Corpus Format Converter: Legacy → Universal");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Load legacy corpus
    println!("Loading legacy corpus from: {:?}", args.input);
    let legacy_json = fs::read_to_string(&args.input)
        .with_context(|| format!("Failed to read input file: {:?}", args.input))?;
    let legacy: LegacyCorpus =
        serde_json::from_str(&legacy_json).context("Failed to parse legacy corpus JSON")?;

    println!("  Python doctest samples: {}", legacy.python_doctest.len());
    println!("  Rust test samples:      {}", legacy.rust_test.len());
    println!(
        "  Python baseline samples: {}",
        legacy.python_baseline.len()
    );
    println!(
        "  Rust baseline samples:   {}\n",
        legacy.rust_baseline.len()
    );

    // Convert samples
    println!("Converting to universal format...\n");

    let mut universal = UniversalCorpus {
        _format_version: "2.0".to_string(),
        _description: "Universal corpus with character positions - works with any model"
            .to_string(),
        python_doctest: Vec::new(),
        rust_test: Vec::new(),
        python_baseline: Vec::new(),
        rust_baseline: Vec::new(),
    };

    // Convert Python doctest samples
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PYTHON DOCTEST SAMPLES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for sample in &legacy.python_doctest {
        let converted = convert_python_sample(sample, args.verbose)?;
        universal.python_doctest.push(converted);
    }

    // Convert Rust test samples
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RUST TEST SAMPLES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for sample in &legacy.rust_test {
        let converted = convert_rust_sample(sample, args.verbose)?;
        universal.rust_test.push(converted);
    }

    // Convert Python baseline samples
    if !legacy.python_baseline.is_empty() {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("PYTHON BASELINE SAMPLES");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for sample in &legacy.python_baseline {
            let converted = convert_python_baseline(sample, args.verbose)?;
            universal.python_baseline.push(converted);
        }
    }

    // Convert Rust baseline samples
    if !legacy.rust_baseline.is_empty() {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("RUST BASELINE SAMPLES");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for sample in &legacy.rust_baseline {
            let converted = convert_rust_baseline(sample, args.verbose)?;
            universal.rust_baseline.push(converted);
        }
    }

    // Save universal corpus
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&args.output, serde_json::to_string_pretty(&universal)?)?;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  CONVERSION COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("\nUniversal corpus saved to: {:?}", args.output);
    println!("Format version: 2.0");
    println!(
        "Total samples converted: {}",
        universal.python_doctest.len()
            + universal.rust_test.len()
            + universal.python_baseline.len()
            + universal.rust_baseline.len()
    );

    Ok(())
}

fn convert_python_sample(sample: &LegacyPythonSample, verbose: bool) -> Result<UniversalSample> {
    // Find >>> in code
    let marker_char_pos = sample.code.find(">>>").unwrap_or(0);

    // Find function parameters by parsing the def line
    let target_char_positions = find_python_param_positions(&sample.code);

    println!(
        "✓ {}: marker@char {} (found >>>), {} targets",
        sample.id,
        marker_char_pos,
        target_char_positions.len()
    );

    if verbose {
        println!("  Target positions: {:?}", target_char_positions);
    }

    Ok(UniversalSample {
        id: sample.id.clone(),
        code: sample.code.clone(),
        marker_char_pos,
        marker_pattern: ">>>".to_string(),
        target_char_positions,
    })
}

fn convert_rust_sample(sample: &LegacyRustSample, verbose: bool) -> Result<UniversalSample> {
    // Find #[test] in code
    let marker_char_pos = sample
        .code
        .find("#[test]")
        .or_else(|| sample.code.find("#[cfg(test)]"))
        .unwrap_or(0);

    let marker_pattern = if sample.code[marker_char_pos..].starts_with("#[test]") {
        "#[test]"
    } else {
        "#[cfg(test)]"
    };

    // Find function tokens by parsing the fn line
    let target_char_positions = find_rust_fn_positions(&sample.code);

    println!(
        "✓ {}: marker@char {} (found {}), {} targets",
        sample.id,
        marker_char_pos,
        marker_pattern,
        target_char_positions.len()
    );

    if verbose {
        println!("  Target positions: {:?}", target_char_positions);
    }

    Ok(UniversalSample {
        id: sample.id.clone(),
        code: sample.code.clone(),
        marker_char_pos,
        marker_pattern: marker_pattern.to_string(),
        target_char_positions,
    })
}

fn convert_python_baseline(
    sample: &LegacyPythonBaselineSample,
    verbose: bool,
) -> Result<UniversalSample> {
    let marker_char_pos = sample.code.find(">>>").unwrap_or(0);
    let target_char_positions = find_python_param_positions(&sample.code);

    println!(
        "✓ {}: marker@char {}, {} targets",
        sample.id,
        marker_char_pos,
        target_char_positions.len()
    );

    if verbose {
        println!("  Target positions: {:?}", target_char_positions);
    }

    Ok(UniversalSample {
        id: sample.id.clone(),
        code: sample.code.clone(),
        marker_char_pos,
        marker_pattern: ">>>".to_string(),
        target_char_positions,
    })
}

fn convert_rust_baseline(
    sample: &LegacyRustBaselineSample,
    verbose: bool,
) -> Result<UniversalSample> {
    let marker_char_pos = sample.code.find("#[").unwrap_or(0);
    let target_char_positions = find_rust_struct_positions(&sample.code);

    println!(
        "✓ {}: marker@char {}, {} targets",
        sample.id,
        marker_char_pos,
        target_char_positions.len()
    );

    if verbose {
        println!("  Target positions: {:?}", target_char_positions);
    }

    Ok(UniversalSample {
        id: sample.id.clone(),
        code: sample.code.clone(),
        marker_char_pos,
        marker_pattern: "#[".to_string(),
        target_char_positions,
    })
}

/// Find character positions of function name and parameters in Python code
fn find_python_param_positions(code: &str) -> Vec<usize> {
    let mut positions = Vec::new();

    // Find def keyword
    if let Some(def_pos) = code.find("def ") {
        positions.push(def_pos); // 'def' itself

        // Find function name
        let after_def = &code[def_pos + 4..];
        if let Some(paren_pos) = after_def.find('(') {
            let fn_name_start = def_pos + 4;
            positions.push(fn_name_start); // function name

            // Find parameters
            let abs_paren = def_pos + 4 + paren_pos;
            if let Some(close_paren) = code[abs_paren..].find(')') {
                let params_str = &code[abs_paren + 1..abs_paren + close_paren];

                // Parse each parameter
                let mut current_offset = abs_paren + 1;
                for param in params_str.split(',') {
                    let trimmed = param.trim();
                    if !trimmed.is_empty() {
                        // Find the parameter name in the original string
                        let param_name = trimmed
                            .split('=')
                            .next()
                            .unwrap()
                            .split(':')
                            .next()
                            .unwrap()
                            .trim();
                        if let Some(param_offset) = code[current_offset..].find(param_name) {
                            positions.push(current_offset + param_offset);
                        }
                    }
                    current_offset += param.len() + 1;
                }
            }
        }
    }

    positions
}

/// Find character positions of fn keyword and function name in Rust code
fn find_rust_fn_positions(code: &str) -> Vec<usize> {
    let mut positions = Vec::new();

    // Find the first fn keyword (the function being tested)
    if let Some(fn_pos) = code.find("fn ") {
        positions.push(fn_pos); // 'fn' keyword

        // Find function name
        let after_fn = &code[fn_pos + 3..];
        if let Some(name_end) = after_fn.find(['(', '<']) {
            if name_end > 0 {
                positions.push(fn_pos + 3); // function name start
            }
        }

        // Find parameters
        if let Some(paren_pos) = after_fn.find('(') {
            let abs_paren = fn_pos + 3 + paren_pos;
            if let Some(close_paren) = code[abs_paren..].find(')') {
                let params_str = &code[abs_paren + 1..abs_paren + close_paren];

                // Parse each parameter
                let mut current_offset = abs_paren + 1;
                for param in params_str.split(',') {
                    let trimmed = param.trim();
                    if !trimmed.is_empty() {
                        // Find the parameter name (before the colon)
                        let param_name = trimmed.split(':').next().unwrap().trim();
                        if let Some(param_offset) = code[current_offset..].find(param_name) {
                            positions.push(current_offset + param_offset);
                        }
                    }
                    current_offset += param.len() + 1;
                }
            }
        }
    }

    positions
}

/// Find character positions of struct fields in Rust code
fn find_rust_struct_positions(code: &str) -> Vec<usize> {
    let mut positions = Vec::new();

    // Find struct or fn keywords
    if let Some(struct_pos) = code.find("struct ") {
        positions.push(struct_pos);

        // Find struct name
        let after_struct = &code[struct_pos + 7..];
        if let Some(name_end) = after_struct.find(['{', '(', ' ', '\t', '\n', '\r']) {
            if name_end > 0 {
                positions.push(struct_pos + 7);
            }
        }
    }

    // Also look for fn
    if let Some(fn_pos) = code.find("fn ") {
        positions.push(fn_pos);
        positions.push(fn_pos + 3); // function name
    }

    positions
}
