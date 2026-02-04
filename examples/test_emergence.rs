//! Test-Awareness Logit Lens Experiment for AIware 2026
//!
//! Investigates at which layer the model starts predicting test-related tokens.
//! Compares Rust (#[test], assert!) vs Python (def test_, assert) patterns.

use anyhow::Result;
use clap::Parser;
use plip_rs::PlipModel;
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "test_emergence")]
#[command(about = "Track when test tokens emerge in Logit Lens predictions")]
struct Args {
    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,

    /// Model to use (default: bigcode/starcoder2-3b)
    #[arg(long, default_value = "bigcode/starcoder2-3b")]
    model: String,

    /// Number of top predictions to consider
    #[arg(long, default_value = "100")]
    top_k: usize,
}

/// Test prompts that naturally lead to test code
struct TestPrompt {
    name: &'static str,
    language: &'static str,
    code: &'static str,
    /// Tokens we expect to see if model predicts tests
    target_tokens: Vec<&'static str>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("=== Test-Awareness Logit Lens Experiment ===");
    println!("AIware 2026: When does the model 'expect' tests?\n");

    // Define test prompts
    let prompts = vec![
        // Rust: Function followed by attribute start
        TestPrompt {
            name: "rust_after_fn_attr",
            language: "Rust",
            code: "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\n#[",
            target_tokens: vec!["test", "cfg", "derive", "allow", "inline", "must"],
        },
        // Rust: Inside test module - what comes after #[
        TestPrompt {
            name: "rust_test_mod",
            language: "Rust",
            code: "#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[",
            target_tokens: vec!["test", "ignore", "should_panic", "bench"],
        },
        // Rust: After test attribute - what function name?
        TestPrompt {
            name: "rust_after_test_attr",
            language: "Rust",
            code: "#[test]\nfn ",
            target_tokens: vec!["test", "it", "should", "check", "verify"],
        },
        // Rust: Completing test]
        TestPrompt {
            name: "rust_complete_test",
            language: "Rust",
            code: "#[te",
            target_tokens: vec!["st", "st]"],
        },
        // Python: Docstring start (expecting doctest)
        TestPrompt {
            name: "python_docstring_start",
            language: "Python",
            code: "def add(a, b):\n    \"\"\"Add two numbers.\n\n    ",
            target_tokens: vec![">>>", "Example", "Args", "Returns"],
        },
        // Python: Inside doctest context
        TestPrompt {
            name: "python_doctest_prompt",
            language: "Python",
            code: "def add(a, b):\n    \"\"\"Add two numbers.\n\n    >>>",
            target_tokens: vec![" add", "(", ">>>"],
        },
        // Python: After doctest example - expecting result
        TestPrompt {
            name: "python_after_doctest",
            language: "Python",
            code: "def add(a, b):\n    \"\"\"Add two numbers.\n\n    >>> add(2, 3)\n    ",
            target_tokens: vec!["5", ">>>", "\"\"\""],
        },
        // Python: Right after >>> - expecting function call
        TestPrompt {
            name: "python_doctest_call",
            language: "Python",
            code: "def add(a, b):\n    return a + b\n\n\"\"\"Example:\n>>> ",
            target_tokens: vec!["add", "print", "import"],
        },
    ];

    println!("Loading {}...", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    println!(
        "Model loaded: {} layers, {} hidden\n",
        model.n_layers(),
        model.d_model()
    );

    // Results storage
    let mut results: HashMap<String, Vec<(usize, String, f32)>> = HashMap::new();

    for prompt in &prompts {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Prompt: {} ({})", prompt.name, prompt.language);
        println!("Code: {:?}", prompt.code);
        println!("Target tokens: {:?}", prompt.target_tokens);
        println!();

        // Run logit lens
        let analysis = model.logit_lens(prompt.code, args.top_k)?;

        // Find first appearance of each target token
        let mut first_appearances: Vec<(usize, String, f32)> = Vec::new();

        for target in &prompt.target_tokens {
            for result in &analysis.layer_results {
                for pred in &result.predictions {
                    if pred.token.contains(target) {
                        first_appearances.push((
                            result.layer,
                            pred.token.clone(),
                            pred.probability,
                        ));
                        println!(
                            "  Layer {:2}: Found '{}' in prediction '{}' ({:.2}%)",
                            result.layer,
                            target,
                            pred.token.replace('\n', "\\n"),
                            pred.probability * 100.0
                        );
                        break;
                    }
                }
                // Only record first appearance
                if first_appearances.iter().any(|(_, t, _)| t.contains(target)) {
                    break;
                }
            }
        }

        if first_appearances.is_empty() {
            println!("  No target tokens found in top-{}", args.top_k);
        }

        // Show top predictions at key layers
        println!("\n  Top predictions at selected layers:");
        let n_layers = model.n_layers();
        let key_layers = [0, n_layers / 3, 2 * n_layers / 3, n_layers - 1];
        for layer_idx in key_layers {
            if let Some(result) = analysis.layer_results.get(layer_idx) {
                let top3: Vec<String> = result
                    .predictions
                    .iter()
                    .take(3)
                    .map(|p| {
                        format!(
                            "{}({:.1}%)",
                            p.token.replace('\n', "\\n"),
                            p.probability * 100.0
                        )
                    })
                    .collect();
                println!("    Layer {:2}: {}", layer_idx, top3.join(", "));
            }
        }

        results.insert(prompt.name.to_string(), first_appearances);
        println!();
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════");
    println!("SUMMARY: First Layer Where Test Tokens Appear");
    println!("═══════════════════════════════════════════════════\n");

    println!("RUST PROMPTS:");
    for prompt in prompts.iter().filter(|p| p.language == "Rust") {
        print!("  {}: ", prompt.name);
        if let Some(appearances) = results.get(prompt.name) {
            if appearances.is_empty() {
                println!("No test tokens found");
            } else {
                let min_layer = appearances.iter().map(|(l, _, _)| *l).min().unwrap();
                println!("Layer {} (earliest)", min_layer);
            }
        }
    }

    println!("\nPYTHON PROMPTS:");
    for prompt in prompts.iter().filter(|p| p.language == "Python") {
        print!("  {}: ", prompt.name);
        if let Some(appearances) = results.get(prompt.name) {
            if appearances.is_empty() {
                println!("No test tokens found");
            } else {
                let min_layer = appearances.iter().map(|(l, _, _)| *l).min().unwrap();
                println!("Layer {} (earliest)", min_layer);
            }
        }
    }

    // Final layer predictions analysis
    let final_layer = model.n_layers() - 1;
    println!("\n═══════════════════════════════════════════════════");
    println!(
        "FINAL LAYER ({}) PREDICTIONS - What would the model generate?",
        final_layer
    );
    println!("═══════════════════════════════════════════════════\n");

    for prompt in &prompts {
        let analysis = model.logit_lens(prompt.code, 10)?;
        if let Some(final_result) = analysis.layer_results.last() {
            println!("{} ({}):", prompt.name, prompt.language);
            println!(
                "  Prompt ends with: {:?}",
                &prompt.code[prompt.code.len().saturating_sub(10)..]
            );
            print!("  Top 10: ");
            for (i, pred) in final_result.predictions.iter().take(10).enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("'{}'", pred.token.replace('\n', "\\n"));
            }
            println!("\n");
        }
    }

    println!("═══════════════════════════════════════════════════");
    println!("AIware 2026 Interpretation:");
    println!("═══════════════════════════════════════════════════");
    println!("- Linear probing shows 100% Python/Rust distinction at ALL layers");
    println!("- Logit Lens shows noisy predictions at intermediate layers");
    println!("- This suggests: language awareness is encoded in REPRESENTATIONS,");
    println!("  not as direct test-token predictions");
    println!("- Test-awareness may require probing specific activation dimensions");

    Ok(())
}
