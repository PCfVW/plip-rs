//! State Steering Generation: RWKV-6 generation with steered recurrent state
//!
//! Tests whether amplifying the test marker's state write during prompt processing
//! affects the model's code generation behavior. The steered recurrent state
//! propagates naturally to all generated tokens via the WKV recurrence.
//!
//! ## Hypothesis
//!
//! If marker position state is causally important for test-aware generation,
//! then amplifying the state write at test markers should make the model more
//! likely to produce test-related syntax (assert, #[test], >>>) in its output.
//!
//! ## Method
//!
//! For each prompt:
//! 1. Find the test marker position (`>>>` or `#[test]`) in the prompt
//! 2. Generate baseline output (no steering)
//! 3. Generate steered output (scale marker's kv^T state write by scale factor)
//! 4. Compare: does steering affect test syntax preservation?
//!
//! Usage:
//!   cargo run --release --example state_steering_generate
//!   cargo run --release --example state_steering_generate -- --layer 14 --scale 5.0
//!   cargo run --release --example state_steering_generate -- --max-tokens 200

// Clippy configuration for ML/statistics code
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::print_literal)]

use anyhow::Result;
use clap::Parser;
use plip_rs::{PlipModel, StateSteeringSpec};

#[derive(Parser)]
#[command(name = "state_steering_generate")]
#[command(about = "RWKV-6 generation with steered recurrent state")]
struct Args {
    /// HuggingFace model ID
    #[arg(short, long, default_value = "RWKV/v6-Finch-1B6-HF")]
    model: String,

    /// Layer to apply state steering (default: layer 2, where knockout was significant)
    #[arg(long)]
    layer: Option<usize>,

    /// Steering scale factor for kv^T state write
    #[arg(long, default_value = "3.0")]
    scale: f32,

    /// Maximum tokens to generate
    #[arg(long, default_value = "100")]
    max_tokens: usize,

    /// Temperature for sampling (0.0 = greedy)
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Force CPU mode
    #[arg(long)]
    cpu: bool,
}

/// Test prompts: code snippets with test markers that should influence generation.
/// Mix of Python doctest (>>>) and Rust #[test] patterns.
const TEST_PROMPTS: &[(&str, &str, &str)] = &[
    (
        "py_simple_add",
        "python_doctest",
        "def add(a, b):\n    \"\"\"\n    >>> add(2, 3)\n    5\n    \"\"\"\n    return a + b\n\ndef multiply(a, b):\n    \"\"\"",
    ),
    (
        "py_factorial",
        "python_doctest",
        "def factorial(n):\n    \"\"\"\n    >>> factorial(5)\n    120\n    >>> factorial(0)\n    1\n    \"\"\"\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\ndef fibonacci(n):\n    \"\"\"",
    ),
    (
        "rs_max_function",
        "rust_test",
        "/// Returns the larger of two values\n///\n/// #[test]\n/// fn test_max() {\n///     assert_eq!(max(3, 5), 5);\n/// }\nfn max(a: i32, b: i32) -> i32 {\n    if a > b { a } else { b }\n}\n\n/// Returns the smaller of two values\nfn min(a: i32, b: i32) -> i32 {",
    ),
    (
        "rs_is_even",
        "rust_test",
        "/// Returns true if the number is even\n///\n/// #[test]\n/// fn test_is_even() {\n///     assert!(is_even(4));\n///     assert!(!is_even(7));\n/// }\nfn is_even(n: i32) -> bool {\n    n % 2 == 0\n}\n\n/// Returns true if the number is odd\nfn is_odd(n: i32) -> bool {",
    ),
    (
        "rs_sum_array",
        "rust_test",
        "/// Sums all elements in a slice\n///\n/// #[test]\n/// fn test_sum() {\n///     assert_eq!(sum(&[1, 2, 3]), 6);\n/// }\nfn sum(arr: &[i32]) -> i32 {\n    arr.iter().sum()\n}\n\n/// Returns the product of all elements\nfn product(arr: &[i32]) -> i32 {",
    ),
];

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("============================================================");
    println!("STATE STEERING GENERATION — RWKV-6");
    println!("============================================================\n");

    // Load model
    println!("Loading model: {}", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;

    // Default layer: 2 (where state knockout showed p=0.018)
    let layer = args.layer.unwrap_or(2);

    println!("Architecture: {:?}", model.architecture());
    println!("Layers: {}", model.n_layers());
    println!("Steering layer: {}", layer);
    println!("Scale factor: {}x", args.scale);
    println!("Max tokens: {}", args.max_tokens);
    println!("Temperature: {}", args.temperature);
    println!();

    let mut results = Vec::new();

    for (name, category, prompt) in TEST_PROMPTS {
        println!("------------------------------------------------------------");
        println!("Sample: {} ({})", name, category);
        println!("------------------------------------------------------------\n");

        // Find marker position
        let marker_pos = find_marker_position(&model, prompt, category)?;

        if marker_pos.is_none() {
            println!("  (No marker found, skipping)");
            println!();
            continue;
        }
        let marker_pos = marker_pos.unwrap();
        println!("  Marker token position: {}", marker_pos);

        // === BASELINE GENERATION ===
        println!("  --- Baseline (no steering) ---");
        let baseline_result = model.generate_with_state_steering_details(
            prompt,
            args.max_tokens,
            args.temperature,
            &[],
            // scale=1.0 is identity (no intervention)
            &StateSteeringSpec::new(1.0).position(marker_pos).layer(layer),
        )?;

        let baseline_has_test = output_contains_test_syntax(&baseline_result.generated_text);
        println!(
            "  Generated {} tokens",
            baseline_result.generated_tokens.len()
        );
        println!("  Contains test syntax: {}", baseline_has_test);
        println!(
            "  Output: {:?}",
            truncate_output(&baseline_result.generated_text, 150)
        );

        // === STEERED GENERATION ===
        println!("  --- Steered ({}x scale) ---", args.scale);
        let steered_result = model.generate_with_state_steering_details(
            prompt,
            args.max_tokens,
            args.temperature,
            &[],
            &StateSteeringSpec::new(args.scale)
                .position(marker_pos)
                .layer(layer),
        )?;

        let steered_has_test = output_contains_test_syntax(&steered_result.generated_text);
        println!(
            "  Generated {} tokens",
            steered_result.generated_tokens.len()
        );
        println!("  Contains test syntax: {}", steered_has_test);
        println!(
            "  Output: {:?}\n",
            truncate_output(&steered_result.generated_text, 150)
        );

        results.push(SampleResult {
            name: name.to_string(),
            category: category.to_string(),
            baseline_has_test,
            steered_has_test,
            baseline_tokens: baseline_result.generated_tokens.len(),
            steered_tokens: steered_result.generated_tokens.len(),
        });
    }

    // Summary
    println!("\n============================================================");
    println!("SUMMARY");
    println!("============================================================\n");

    println!(
        "| {:15} | {:10} | {:16} | {:15} | {:6} |",
        "Sample", "Category", "Baseline test?", "Steered test?", "Change"
    );
    println!(
        "|{:-<17}|{:-<12}|{:-<18}|{:-<17}|{:-<8}|",
        "", "", "", "", ""
    );

    let mut baseline_count = 0;
    let mut steered_count = 0;

    for r in &results {
        let change = match (r.baseline_has_test, r.steered_has_test) {
            (false, true) => "+ GAINED",
            (true, false) => "- LOST",
            (true, true) => "= KEPT",
            (false, false) => "= NONE",
        };

        println!(
            "| {:15} | {:10} | {:16} | {:15} | {:6} |",
            r.name,
            r.category,
            if r.baseline_has_test { "Yes" } else { "No" },
            if r.steered_has_test { "Yes" } else { "No" },
            change
        );

        if r.baseline_has_test {
            baseline_count += 1;
        }
        if r.steered_has_test {
            steered_count += 1;
        }
    }

    println!();
    println!(
        "Baseline test preservation: {}/{}",
        baseline_count,
        results.len()
    );
    println!(
        "Steered test preservation:  {}/{}",
        steered_count,
        results.len()
    );

    if steered_count > baseline_count {
        println!("\nSteering IMPROVED test preservation.");
    } else if steered_count < baseline_count {
        println!("\nSteering DECREASED test preservation.");
    } else {
        println!("\nSteering had no effect on test preservation.");
    }

    Ok(())
}

#[derive(Clone)]
struct SampleResult {
    name: String,
    #[allow(dead_code)]
    category: String,
    baseline_has_test: bool,
    steered_has_test: bool,
    #[allow(dead_code)]
    baseline_tokens: usize,
    #[allow(dead_code)]
    steered_tokens: usize,
}

/// Find the test marker position in the prompt (character position → token position).
fn find_marker_position(
    model: &PlipModel,
    text: &str,
    category: &str,
) -> Result<Option<usize>> {
    let marker = if category.contains("python") {
        ">>>"
    } else {
        "#[test]"
    };

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
