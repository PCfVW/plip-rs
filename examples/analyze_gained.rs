//! Analyze GAINED samples - compare baseline vs steered outputs
//!
//! Runs only on the 5 samples that gained test preservation through steering.

use anyhow::Result;
use plip_rs::{PlipModel, SteeringSpec};

const GAINED_PROMPTS: &[(&str, &str)] = &[
    (
        "string_04_lowercase",
        r#"/// Converts string to lowercase
///
/// #[test]
/// fn test_lowercase() {
///     assert_eq!(to_lower("HELLO"), "hello");
///     assert_eq!(to_lower("World"), "world");
/// }
fn to_lower(s: &str) -> String {"#,
    ),
    (
        "string_07_starts_with",
        r#"/// Checks if string starts with prefix
///
/// #[test]
/// fn test_starts_with() {
///     assert!(starts_with("hello world", "hello"));
///     assert!(!starts_with("hello", "world"));
/// }
fn starts_with(s: &str, prefix: &str) -> bool {"#,
    ),
    (
        "string_09_repeat",
        r#"/// Repeats a string n times
///
/// #[test]
/// fn test_repeat() {
///     assert_eq!(repeat("ab", 3), "ababab");
///     assert_eq!(repeat("x", 0), "");
/// }
fn repeat(s: &str, n: usize) -> String {"#,
    ),
    (
        "string_10_trim",
        r#"/// Trims whitespace from both ends
///
/// #[test]
/// fn test_trim() {
///     assert_eq!(trim_str("  hello  "), "hello");
///     assert_eq!(trim_str("no_spaces"), "no_spaces");
/// }
fn trim_str(s: &str) -> String {"#,
    ),
    (
        "gen_01_identity",
        r#"/// Returns the input unchanged
///
/// #[test]
/// fn test_identity() {
///     assert_eq!(identity(42), 42);
///     assert_eq!(identity("hello"), "hello");
/// }
fn identity<T>(x: T) -> T {"#,
    ),
];

fn main() -> Result<()> {
    println!("============================================================");
    println!("ANALYZING GAINED SAMPLES - Baseline vs Steered");
    println!("============================================================\n");

    // Load model
    println!("Loading model: Qwen/Qwen2.5-Coder-3B-Instruct\n");
    let model =
        PlipModel::from_pretrained_with_device("Qwen/Qwen2.5-Coder-3B-Instruct", Some(false))?;

    let layer = 20;
    let scale = 3.0;
    let max_tokens = 200; // Slightly more tokens to see full output
    let temperature = 0.0;

    let stop_tokens: Vec<u32> = model.eos_token_id().into_iter().collect();

    for (name, prompt) in GAINED_PROMPTS {
        println!("\n======================================================================");
        println!("Sample: {}", name);
        println!("======================================================================\n");

        // Format prompt with chat template
        let formatted_prompt = model.apply_chat_template(
            &format!("Complete this Rust function:\n\n{}", prompt),
            Some("You are a Rust programming assistant. Complete the function with idiomatic Rust code."),
        );

        // Find positions
        let test_marker_pos = formatted_prompt.find("#[test]").and_then(|char_pos| {
            model
                .char_to_token(&formatted_prompt, char_pos)
                .ok()
                .flatten()
        });

        let fn_positions = formatted_prompt
            .rfind("fn ")
            .and_then(|char_pos| {
                model
                    .char_to_token(&formatted_prompt, char_pos)
                    .ok()
                    .flatten()
            })
            .map(|pos| vec![pos]);

        println!("Prompt (original):\n{}\n", prompt);
        println!(
            "Test marker pos: {:?}, Fn positions: {:?}\n",
            test_marker_pos, fn_positions
        );

        // === BASELINE ===
        println!("--- BASELINE (no steering) ---");
        let baseline = model.generate_with_details(
            &formatted_prompt,
            max_tokens,
            temperature,
            &stop_tokens,
            None,
        )?;

        let baseline_has_test = output_contains_test(&baseline.generated_text);
        println!("Has test: {}", baseline_has_test);
        println!("Tokens: {}", baseline.generated_tokens.len());
        println!("Output:\n{}\n", baseline.generated_text);

        // === STEERED ===
        println!("--- STEERED ({}x scale, layer {}) ---", scale, layer);

        let steered = if let (Some(marker_pos), Some(ref fn_pos)) = (test_marker_pos, &fn_positions)
        {
            let spec = SteeringSpec::scale(scale)
                .layer(layer)
                .from_to_positions(marker_pos, fn_pos);

            model.generate_with_details(
                &formatted_prompt,
                max_tokens,
                temperature,
                &stop_tokens,
                Some(&spec),
            )?
        } else {
            println!("(Could not find marker/fn positions, using baseline)");
            baseline.clone()
        };

        let steered_has_test = output_contains_test(&steered.generated_text);
        println!("Has test: {}", steered_has_test);
        println!("Tokens: {}", steered.generated_tokens.len());
        println!("Output:\n{}\n", steered.generated_text);

        // Analysis
        println!("--- ANALYSIS ---");
        if baseline_has_test && steered_has_test {
            println!("Both have tests - KEPT");
        } else if !baseline_has_test && steered_has_test {
            println!("Test GAINED through steering!");
            // Check what test content was added
            for pattern in ["#[test]", "assert_eq!", "assert!", "fn test_"] {
                if steered.generated_text.contains(pattern)
                    && !baseline.generated_text.contains(pattern)
                {
                    println!("  Added: {}", pattern);
                }
            }
        } else if baseline_has_test && !steered_has_test {
            println!("Test LOST through steering!");
        } else {
            println!("Neither has tests - NONE");
        }
    }

    println!("\n============================================================");
    println!("ANALYSIS COMPLETE");
    println!("============================================================");

    Ok(())
}

fn output_contains_test(text: &str) -> bool {
    text.contains("#[test]")
        || text.contains("assert_eq!")
        || text.contains("assert!")
        || text.contains("fn test_")
}
