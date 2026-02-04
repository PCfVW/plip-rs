//! Steering Generation Proof-of-Concept
//!
//! Tests whether attention steering during generation affects test preservation.
//! Compares baseline generation vs generation with boosted #[test] attention.
//!
//! Usage:
//!   cargo run --release --example steering_generate
//!   cargo run --release --example steering_generate -- --model "Qwen/Qwen2.5-Coder-7B-Instruct"

use anyhow::Result;
use clap::Parser;
use plip_rs::{PlipModel, SteeringSpec};

#[derive(Parser)]
#[command(name = "steering_generate")]
#[command(about = "Test generation with attention steering")]
struct Args {
    /// Model to use
    #[arg(short, long, default_value = "Qwen/Qwen2.5-Coder-3B-Instruct")]
    model: String,

    /// Layer to apply steering
    #[arg(long)]
    layer: Option<usize>,

    /// Steering scale factor
    #[arg(long, default_value = "3.0")]
    scale: f32,

    /// Maximum tokens to generate
    #[arg(long, default_value = "150")]
    max_tokens: usize,

    /// Temperature for sampling (0.0 = greedy)
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Force CPU mode
    #[arg(long)]
    cpu: bool,

    /// Use chat template formatting (for instruct models)
    #[arg(long)]
    chat: bool,
}

/// Test prompts with inline tests that should be preserved
const TEST_PROMPTS: &[(&str, &str)] = &[
    (
        "max_function",
        r#"/// Returns the larger of two values
///
/// # Tests
/// ```
/// #[test]
/// fn test_max() {
///     assert_eq!(max(3, 5), 5);
///     assert_eq!(max(7, 2), 7);
/// }
/// ```
fn max(a: i32, b: i32) -> i32 {"#,
    ),
    (
        "is_even",
        r#"/// Returns true if the number is even
///
/// #[test]
/// fn test_is_even() {
///     assert!(is_even(4));
///     assert!(!is_even(7));
/// }
fn is_even(n: i32) -> bool {"#,
    ),
    (
        "factorial",
        r#"/// Computes factorial of n
///
/// #[test]
/// fn test_factorial() {
///     assert_eq!(factorial(0), 1);
///     assert_eq!(factorial(5), 120);
/// }
fn factorial(n: u32) -> u32 {"#,
    ),
    (
        "reverse_string",
        r#"/// Reverses a string
///
/// #[test]
/// fn test_reverse() {
///     assert_eq!(reverse("hello"), "olleh");
/// }
fn reverse(s: &str) -> String {"#,
    ),
    (
        "sum_array",
        r#"/// Sums all elements in a slice
///
/// #[test]
/// fn test_sum() {
///     assert_eq!(sum(&[1, 2, 3]), 6);
///     assert_eq!(sum(&[]), 0);
/// }
fn sum(arr: &[i32]) -> i32 {"#,
    ),
];

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("============================================================");
    println!("STEERING GENERATION PROOF-OF-CONCEPT");
    println!("============================================================\n");

    // Load model
    println!("Loading model: {}", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;

    // Determine layer
    let layer = args.layer.unwrap_or_else(|| {
        let n = model.n_layers();
        // Use layer ~2/3 through the model (where we found strong attention patterns)
        match model.architecture() {
            plip_rs::ModelArchitecture::Qwen2 => {
                if n > 30 {
                    16
                } else {
                    20
                } // 7B vs 3B
            }
            _ => n * 2 / 3,
        }
    });

    println!("Architecture: {:?}", model.architecture());
    println!("Layers: {}", model.n_layers());
    println!("Steering layer: {}", layer);
    println!("Scale factor: {}×", args.scale);
    println!("Max tokens: {}", args.max_tokens);
    println!("Temperature: {}", args.temperature);
    println!(
        "Chat template: {}",
        if args.chat { "enabled" } else { "disabled" }
    );
    if args.chat && !model.is_instruct_model() {
        println!("⚠️  Warning: --chat specified but model doesn't appear to be an instruct model");
    }
    println!();

    // Get stop tokens for chat mode
    let stop_tokens: Vec<u32> = if args.chat {
        model.eos_token_id().into_iter().collect()
    } else {
        vec![]
    };

    // Process each test prompt
    let mut results = Vec::new();

    for (name, prompt) in TEST_PROMPTS {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Sample: {}", name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // Format prompt (with chat template if requested)
        let formatted_prompt = if args.chat {
            model.apply_chat_template(
                &format!("Complete this Rust function:\n\n{}", prompt),
                Some("You are a Rust programming assistant. Complete the function with idiomatic Rust code."),
            )
        } else {
            prompt.to_string()
        };

        // Find test marker position in the prompt (use original prompt for position finding)
        // We need to boost attention from #[test] to fn tokens
        let test_marker_pos = find_test_marker_position(&model, &formatted_prompt)?;
        let fn_positions = find_fn_positions(&model, &formatted_prompt)?;

        println!("Test marker position: {:?}", test_marker_pos);
        println!("Function positions: {:?}", fn_positions);
        println!();

        // === BASELINE GENERATION ===
        println!("--- Baseline (no steering) ---");
        let baseline_result = model.generate_with_details(
            &formatted_prompt,
            args.max_tokens,
            args.temperature,
            &stop_tokens,
            None,
        )?;

        let baseline_has_test = output_contains_test(&baseline_result.generated_text);
        println!(
            "Generated {} tokens",
            baseline_result.generated_tokens.len()
        );
        println!("Contains #[test]: {}", baseline_has_test);
        println!(
            "Generated: {:?}",
            truncate_output(&baseline_result.generated_text, 200)
        );

        // === STEERED GENERATION ===
        println!("--- Steered ({}× scale) ---", args.scale);

        // Build steering spec if we found the marker
        let steered_result =
            if let (Some(marker_pos), Some(ref fn_pos)) = (test_marker_pos, &fn_positions) {
                if !fn_pos.is_empty() {
                    let spec = SteeringSpec::scale(args.scale)
                        .layer(layer)
                        .from_to_positions(marker_pos, fn_pos);

                    model.generate_with_details(
                        &formatted_prompt,
                        args.max_tokens,
                        args.temperature,
                        &stop_tokens,
                        Some(&spec),
                    )?
                } else {
                    println!("(No function positions found, using baseline)");
                    baseline_result.clone()
                }
            } else {
                println!("(No test marker found, using baseline)");
                baseline_result.clone()
            };

        let steered_has_test = output_contains_test(&steered_result.generated_text);
        println!("Generated {} tokens", steered_result.generated_tokens.len());
        println!("Contains #[test]: {}", steered_has_test);
        println!(
            "\nGenerated:\n{}\n",
            truncate_output(&steered_result.generated_text, 300)
        );

        // Record result
        results.push(SampleResult {
            name: name.to_string(),
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

    println!("| Sample          | Baseline #[test] | Steered #[test] | Change |");
    println!("|-----------------|------------------|-----------------|--------|");

    let mut baseline_preserved = 0;
    let mut steered_preserved = 0;

    for r in &results {
        let change = match (r.baseline_has_test, r.steered_has_test) {
            (false, true) => "✓ GAINED",
            (true, false) => "✗ LOST",
            (true, true) => "= KEPT",
            (false, false) => "= NONE",
        };

        println!(
            "| {:15} | {:16} | {:15} | {:6} |",
            r.name,
            if r.baseline_has_test { "Yes" } else { "No" },
            if r.steered_has_test { "Yes" } else { "No" },
            change
        );

        if r.baseline_has_test {
            baseline_preserved += 1;
        }
        if r.steered_has_test {
            steered_preserved += 1;
        }
    }

    println!();
    println!(
        "Baseline preservation: {}/{}",
        baseline_preserved,
        results.len()
    );
    println!(
        "Steered preservation:  {}/{}",
        steered_preserved,
        results.len()
    );

    if steered_preserved > baseline_preserved {
        println!("\n✓ Steering IMPROVED test preservation!");
    } else if steered_preserved < baseline_preserved {
        println!("\n✗ Steering DECREASED test preservation.");
    } else {
        println!("\n= Steering had no effect on test preservation.");
    }

    Ok(())
}

#[derive(Clone)]
struct SampleResult {
    name: String,
    baseline_has_test: bool,
    steered_has_test: bool,
    #[allow(dead_code)]
    baseline_tokens: usize,
    #[allow(dead_code)]
    steered_tokens: usize,
}

/// Find the token position of #[test] marker in the prompt
fn find_test_marker_position(model: &PlipModel, text: &str) -> Result<Option<usize>> {
    // Look for #[test] in the text
    if let Some(char_pos) = text.find("#[test]") {
        let token_pos = model.char_to_token(text, char_pos)?;
        return Ok(token_pos);
    }
    Ok(None)
}

/// Find token positions of function-related tokens (fn, function name)
fn find_fn_positions(model: &PlipModel, text: &str) -> Result<Option<Vec<usize>>> {
    let mut positions = Vec::new();

    // Find "fn " keyword
    if let Some(char_pos) = text.rfind("fn ") {
        if let Some(token_pos) = model.char_to_token(text, char_pos)? {
            positions.push(token_pos);
        }
    }

    if positions.is_empty() {
        Ok(None)
    } else {
        Ok(Some(positions))
    }
}

/// Check if generated output contains test-related content
fn output_contains_test(text: &str) -> bool {
    text.contains("#[test]")
        || text.contains("assert_eq!")
        || text.contains("assert!")
        || text.contains("fn test_")
}

/// Truncate output for display (UTF-8 safe)
fn truncate_output(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        text.to_string()
    } else {
        let truncated: String = chars[..max_chars].iter().collect();
        format!("{}...", truncated)
    }
}
