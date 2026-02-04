//! Attention Pattern Analysis for AIware 2026
//!
//! Investigates how code models attend to test-related tokens:
//! - What does #[test] attend to?
//! - What does >>> (doctest) attend to?
//! - Do attention patterns differ between Rust and Python test markers?
//!
//! Usage:
//!   cargo run --release --example attention_patterns
//!   cargo run --release --example attention_patterns -- --model "Qwen/Qwen2.5-Coder-3B-Instruct"

use anyhow::Result;
use clap::Parser;
use plip_rs::PlipModel;

#[derive(Parser)]
#[command(name = "attention_patterns")]
#[command(about = "Analyze attention patterns for test-related tokens")]
struct Args {
    /// HuggingFace model ID
    #[arg(short, long, default_value = "Qwen/Qwen2.5-Coder-3B-Instruct")]
    model: String,

    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,

    /// Number of layers to analyze (default: sample layers)
    #[arg(long)]
    layers: Option<usize>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("=== Attention Pattern Analysis for AIware 2026 ===\n");

    // Test code samples
    let rust_test_code = r#"fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[test]
fn test_add() {
    assert_eq!(add(2, 3), 5);
}"#;

    let python_doctest_code = r#"def add(a, b):
    """Add two numbers.

    >>> add(2, 3)
    5
    """
    return a + b"#;

    println!("Loading {}...", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    println!(
        "Model loaded: {} layers, architecture: {:?}\n",
        model.n_layers(),
        model.architecture()
    );

    // Determine layers to analyze based on model size
    let n_layers = model.n_layers();
    let layers_to_analyze: Vec<usize> = match args.layers {
        Some(n) => (0..n.min(n_layers)).collect(),
        None => {
            // Sample ~7 layers evenly distributed
            let step = n_layers / 6;
            (0..n_layers).step_by(step.max(1)).take(7).collect()
        }
    };

    println!("Analyzing layers: {:?}\n", layers_to_analyze);

    // Analyze Rust test code
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RUST TEST CODE ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("Code:\n{}\n", rust_test_code);

    let rust_analysis = model.analyze_attention(rust_test_code)?;

    // Show tokens
    println!("Tokens ({}):", rust_analysis.tokens.len());
    for (i, token) in rust_analysis.tokens.iter().enumerate() {
        println!("  {:2}: '{}'", i, token.replace('\n', "\\n"));
    }

    // Find test-related tokens
    println!("\n--- Attention Analysis ---");
    if let Some(test_attr_idx) = rust_analysis.find_token("#[test]") {
        println!("\nFound '#[test]' at position {}", test_attr_idx);
        for &layer in &layers_to_analyze {
            rust_analysis.print_attention_for_token(test_attr_idx, layer, 5);
        }
    } else if let Some(hash_idx) = rust_analysis.find_token("#[") {
        println!("\nFound '#[' at position {}", hash_idx);
        for &layer in &layers_to_analyze {
            rust_analysis.print_attention_for_token(hash_idx, layer, 5);
        }
    }

    if let Some(assert_idx) = rust_analysis.find_token("assert") {
        println!("\nFound 'assert' at position {}", assert_idx);
        for &layer in &layers_to_analyze {
            rust_analysis.print_attention_for_token(assert_idx, layer, 5);
        }
    }

    // Analyze Python doctest code
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PYTHON DOCTEST CODE ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("Code:\n{}\n", python_doctest_code);

    let python_analysis = model.analyze_attention(python_doctest_code)?;

    // Show tokens
    println!("Tokens ({}):", python_analysis.tokens.len());
    for (i, token) in python_analysis.tokens.iter().enumerate() {
        println!("  {:2}: '{}'", i, token.replace('\n', "\\n"));
    }

    // Find doctest-related tokens
    println!("\n--- Attention Analysis ---");
    if let Some(doctest_idx) = python_analysis.find_token(">>>") {
        println!("\nFound '>>>' at position {}", doctest_idx);
        for &layer in &layers_to_analyze {
            python_analysis.print_attention_for_token(doctest_idx, layer, 5);
        }
    }

    if let Some(def_idx) = python_analysis.find_token("def") {
        println!("\nFound 'def' at position {}", def_idx);
        for &layer in &layers_to_analyze {
            python_analysis.print_attention_for_token(def_idx, layer, 5);
        }
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════");
    println!("AIWARE 2026 INTERPRETATION");
    println!("═══════════════════════════════════════════════════\n");
    println!("Key questions answered by this analysis:");
    println!("1. Does #[test] attend to the function it's testing?");
    println!("2. Does >>> attend to the function signature?");
    println!("3. Do attention patterns change across layers?");
    println!("4. Are test markers processed similarly to regular code?");

    Ok(())
}
