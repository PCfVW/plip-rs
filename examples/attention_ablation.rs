//! Attention Ablation (Knockout) Experiments
//!
//! Demonstrates causal intervention on attention edges to understand
//! which attention patterns are causally important for predictions.
//!
//! Usage:
//!   cargo run --release --example attention_ablation
//!   cargo run --release --example attention_ablation -- --model "Qwen/Qwen2.5-Coder-3B-Instruct"
//!   cargo run --release --example attention_ablation -- --ablate-test-marker

use anyhow::Result;
use clap::Parser;
use plip_rs::{KnockoutSpec, PlipModel};

#[derive(Parser)]
#[command(name = "attention_ablation")]
#[command(about = "Causal intervention experiments via attention knockout")]
struct Args {
    /// HuggingFace model ID
    #[arg(short, long, default_value = "Qwen/Qwen2.5-Coder-3B-Instruct")]
    model: String,

    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,

    /// Ablate attention from test marker tokens
    #[arg(long)]
    ablate_test_marker: bool,

    /// Specific layer to ablate (default: best layer from prior analysis)
    #[arg(long)]
    layer: Option<usize>,

    /// Specific head to ablate (default: all heads)
    #[arg(long)]
    head: Option<usize>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("=== Attention Ablation Experiments ===\n");

    // Sample code with Python doctest
    let python_code = r#"def add(a, b):
    """Add two numbers.

    >>> add(2, 3)
    5
    """
    return a + b"#;

    // Sample code with Rust test
    let rust_code = r#"fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[test]
fn test_add() {
    assert_eq!(add(2, 3), 5);
}"#;

    println!("Loading {}...", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    println!(
        "Model loaded: {} layers, {} heads\n",
        model.n_layers(),
        model.n_heads()
    );

    if args.ablate_test_marker {
        // Run test marker ablation on both languages
        println!("=== Python Doctest Ablation ===\n");
        run_test_marker_ablation(&model, python_code, ">>>", &args)?;

        println!("\n=== Rust Test Ablation ===\n");
        run_test_marker_ablation(&model, rust_code, "#[test]", &args)?;
    } else {
        // Default: show tokens and run simple ablation demo
        println!("Python code:");
        show_tokens(&model, python_code)?;

        println!("\nRust code:");
        show_tokens(&model, rust_code)?;

        // Run a simple ablation demo
        println!("\n=== Simple Ablation Demo ===\n");
        simple_ablation_demo(&model, python_code, &args)?;
    }

    Ok(())
}

fn show_tokens(model: &PlipModel, code: &str) -> Result<()> {
    let tokens = model.tokenize(code)?;
    println!("Tokens ({}):", tokens.len());
    for (i, token) in tokens.iter().enumerate() {
        let display = token.replace('\n', "\\n").replace('\t', "\\t");
        println!("  {:2}: '{}'", i, display);
    }
    Ok(())
}

/// Ablate attention from test markers and measure impact
fn run_test_marker_ablation(
    model: &PlipModel,
    code: &str,
    marker: &str,
    args: &Args,
) -> Result<()> {
    println!("Code:\n{}\n", code);

    // Find test marker position
    let tokens = model.tokenize(code)?;
    let marker_pos = tokens
        .iter()
        .position(|t| t.contains(marker) || (marker == ">>>" && t.contains(">>>")));

    let marker_pos = match marker_pos {
        Some(pos) => {
            println!("Found marker '{}' at token position {}", marker, pos);
            pos
        }
        None => {
            // Try to find partial match
            if let Some(pos) = tokens.iter().position(|t| {
                t.contains("#[") || t.contains("test") || t.contains(">") || t.contains(">>")
            }) {
                println!(
                    "Found partial marker match at position {} (token: '{}')",
                    pos, tokens[pos]
                );
                pos
            } else {
                println!("Could not find test marker '{}' in tokens", marker);
                return Ok(());
            }
        }
    };

    // Determine which layer(s) to test
    let best_layer = args.layer.unwrap_or_else(|| {
        // Use layer that showed strongest signal in prior analysis
        // Qwen2.5-Coder-7B: layer 16, Qwen2.5-Coder-3B: layer 14
        if model.n_layers() >= 28 {
            16
        } else {
            14
        }
    });

    println!("Testing ablation at layer {}\n", best_layer);

    // Build knockout spec
    let mut spec = KnockoutSpec::new()
        .layer(best_layer)
        .from_position(marker_pos);

    if let Some(h) = args.head {
        spec = spec.head(h);
        println!("Targeting head {} only", h);
    } else {
        println!("Targeting all heads");
    }

    // Run ablation
    match model.forward_with_intervention(code, &spec) {
        Ok(result) => {
            let kl = result.kl_divergence()?;
            println!("\nResults:");
            println!("  KL divergence: {:.6}", kl);

            // Show top changed tokens
            let changed = result.top_changed_tokens(5)?;
            println!("\n  Top changed tokens:");
            for (token_id, base_prob, abl_prob, diff) in &changed {
                let token = model.decode_token(*token_id);
                let display = token.replace('\n', "\\n").replace('\t', "\\t");
                println!(
                    "    '{}': {:.4} -> {:.4} (diff: {:.4})",
                    display, base_prob, abl_prob, diff
                );
            }
        }
        Err(e) => {
            println!("Ablation failed: {}", e);
        }
    }

    Ok(())
}

/// Simple ablation demo showing the API
fn simple_ablation_demo(model: &PlipModel, code: &str, args: &Args) -> Result<()> {
    let layer = args.layer.unwrap_or(10);

    println!(
        "Ablating attention at layer {} from position 5 to positions 0-4\n",
        layer
    );

    let spec = KnockoutSpec::new()
        .layer(layer)
        .from_to_positions(5, &[0, 1, 2, 3, 4]);

    match model.forward_with_intervention(code, &spec) {
        Ok(result) => {
            let kl = result.kl_divergence()?;
            println!("KL divergence: {:.6}", kl);

            if kl > 0.01 {
                println!("  -> Significant impact detected");
            } else if kl > 0.001 {
                println!("  -> Moderate impact detected");
            } else {
                println!("  -> Minimal impact detected");
            }
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    Ok(())
}
