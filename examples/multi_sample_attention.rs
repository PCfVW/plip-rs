//! Control 3: Multi-Sample Attention Analysis for AIware 2026
//!
//! Runs attention analysis on N samples per language to compute
//! mean and standard deviation for statistical significance.

use anyhow::Result;
use clap::Parser;
use plip_rs::PlipModel;

#[derive(Parser)]
#[command(name = "multi_sample_attention")]
#[command(about = "Control 3: Multi-sample attention analysis with statistics")]
struct Args {
    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,

    /// Model to use
    #[arg(long, default_value = "Qwen/Qwen2.5-Coder-7B-Instruct")]
    model: String,
}

struct CodeSample {
    name: &'static str,
    language: &'static str,
    code: &'static str,
    /// Token to analyze attention FROM
    marker_token: &'static str,
    /// Tokens to measure attention TO (function params)
    fn_param_tokens: Vec<&'static str>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  CONTROL 3: Multi-Sample Attention Analysis");
    println!("  AIware 2026 - Statistical Significance Test");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Define 5 Python samples with doctests
    let python_samples = vec![
        CodeSample {
            name: "add",
            language: "Python",
            code: "def add(a, b):\n    \"\"\"Add two numbers.\n\n    >>> add(2, 3)\n    5\n    \"\"\"\n    return a + b",
            marker_token: ">>>",
            fn_param_tokens: vec!["a", "b", "add"],
        },
        CodeSample {
            name: "multiply",
            language: "Python",
            code: "def multiply(x, y):\n    \"\"\"Multiply two numbers.\n\n    >>> multiply(3, 4)\n    12\n    \"\"\"\n    return x * y",
            marker_token: ">>>",
            fn_param_tokens: vec!["x", "y", "multiply"],
        },
        CodeSample {
            name: "subtract",
            language: "Python",
            code: "def subtract(a, b):\n    \"\"\"Subtract b from a.\n\n    >>> subtract(10, 3)\n    7\n    \"\"\"\n    return a - b",
            marker_token: ">>>",
            fn_param_tokens: vec!["a", "b", "subtract"],
        },
        CodeSample {
            name: "divide",
            language: "Python",
            code: "def divide(num, denom):\n    \"\"\"Divide num by denom.\n\n    >>> divide(10, 2)\n    5.0\n    \"\"\"\n    return num / denom",
            marker_token: ">>>",
            fn_param_tokens: vec!["num", "denom", "divide"],
        },
        CodeSample {
            name: "power",
            language: "Python",
            code: "def power(base, exp):\n    \"\"\"Raise base to exp.\n\n    >>> power(2, 3)\n    8\n    \"\"\"\n    return base ** exp",
            marker_token: ">>>",
            fn_param_tokens: vec!["base", "exp", "power"],
        },
    ];

    // Define 5 Rust samples with #[test]
    let rust_samples = vec![
        CodeSample {
            name: "add",
            language: "Rust",
            code: "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\n#[test]\nfn test_add() {\n    assert_eq!(add(2, 3), 5);\n}",
            marker_token: "#[",
            fn_param_tokens: vec!["a", "b", "add"],
        },
        CodeSample {
            name: "multiply",
            language: "Rust",
            code: "fn multiply(x: i32, y: i32) -> i32 {\n    x * y\n}\n\n#[test]\nfn test_multiply() {\n    assert_eq!(multiply(3, 4), 12);\n}",
            marker_token: "#[",
            fn_param_tokens: vec!["x", "y", "multiply"],
        },
        CodeSample {
            name: "subtract",
            language: "Rust",
            code: "fn subtract(a: i32, b: i32) -> i32 {\n    a - b\n}\n\n#[test]\nfn test_subtract() {\n    assert_eq!(subtract(10, 3), 7);\n}",
            marker_token: "#[",
            fn_param_tokens: vec!["a", "b", "subtract"],
        },
        CodeSample {
            name: "divide",
            language: "Rust",
            code: "fn divide(num: f64, denom: f64) -> f64 {\n    num / denom\n}\n\n#[test]\nfn test_divide() {\n    assert_eq!(divide(10.0, 2.0), 5.0);\n}",
            marker_token: "#[",
            fn_param_tokens: vec!["num", "denom", "divide"],
        },
        CodeSample {
            name: "power",
            language: "Rust",
            code: "fn power(base: i32, exp: u32) -> i32 {\n    base.pow(exp)\n}\n\n#[test]\nfn test_power() {\n    assert_eq!(power(2, 3), 8);\n}",
            marker_token: "#[",
            fn_param_tokens: vec!["base", "exp", "power"],
        },
    ];

    println!("Loading {}...", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;
    let n_layers = model.n_layers();
    let final_layer = n_layers - 1;
    println!("Model loaded: {} layers\n", n_layers);

    // Analyze Python samples
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PYTHON DOCTEST SAMPLES (>>> marker)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut python_attentions: Vec<f32> = Vec::new();

    for sample in &python_samples {
        let attn = analyze_sample(&model, sample, final_layer)?;
        println!(
            "  {} ({}): {} → fn params = {:.2}%",
            sample.name,
            sample.language,
            sample.marker_token,
            attn * 100.0
        );
        python_attentions.push(attn);
    }

    let (python_mean, python_std) = compute_stats(&python_attentions);
    println!(
        "\n  Python Mean: {:.2}% ± {:.2}%",
        python_mean * 100.0,
        python_std * 100.0
    );

    // Analyze Rust samples
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("RUST TEST SAMPLES (#[ marker)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut rust_attentions: Vec<f32> = Vec::new();

    for sample in &rust_samples {
        let attn = analyze_sample(&model, sample, final_layer)?;
        println!(
            "  {} ({}): {} → fn params = {:.2}%",
            sample.name,
            sample.language,
            sample.marker_token,
            attn * 100.0
        );
        rust_attentions.push(attn);
    }

    let (rust_mean, rust_std) = compute_stats(&rust_attentions);
    println!(
        "\n  Rust Mean: {:.2}% ± {:.2}%",
        rust_mean * 100.0,
        rust_std * 100.0
    );

    // Statistical summary
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  STATISTICAL SUMMARY (Layer {})", final_layer);
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("┌──────────┬────────┬────────────────────────┬─────────┬─────┐");
    println!("│ Language │ Token  │ Mean Attn to Fn Params │ Std Dev │  N  │");
    println!("├──────────┼────────┼────────────────────────┼─────────┼─────┤");
    println!(
        "│ Python   │ >>>    │ {:>20.2}% │ {:>6.2}% │ {:>3} │",
        python_mean * 100.0,
        python_std * 100.0,
        python_attentions.len()
    );
    println!(
        "│ Rust     │ #[     │ {:>20.2}% │ {:>6.2}% │ {:>3} │",
        rust_mean * 100.0,
        rust_std * 100.0,
        rust_attentions.len()
    );
    println!("└──────────┴────────┴────────────────────────┴─────────┴─────┘");

    // Effect size (Cohen's d)
    let pooled_std = ((python_std.powi(2) + rust_std.powi(2)) / 2.0).sqrt();
    let cohens_d = if pooled_std > 0.0 {
        (python_mean - rust_mean).abs() / pooled_std
    } else {
        0.0
    };

    println!("\nEffect size (Cohen's d): {:.2}", cohens_d);
    if cohens_d > 0.8 {
        println!("  → Large effect size (d > 0.8)");
    } else if cohens_d > 0.5 {
        println!("  → Medium effect size (0.5 < d < 0.8)");
    } else if cohens_d > 0.2 {
        println!("  → Small effect size (0.2 < d < 0.5)");
    } else {
        println!("  → Negligible effect size (d < 0.2)");
    }

    // Simple significance test: non-overlapping error bars
    let python_low = python_mean - python_std;
    let python_high = python_mean + python_std;
    let rust_low = rust_mean - rust_std;
    let rust_high = rust_mean + rust_std;

    println!("\nConfidence intervals (mean ± 1 std):");
    println!(
        "  Python: [{:.2}%, {:.2}%]",
        python_low * 100.0,
        python_high * 100.0
    );
    println!(
        "  Rust:   [{:.2}%, {:.2}%]",
        rust_low * 100.0,
        rust_high * 100.0
    );

    if python_low > rust_high || rust_low > python_high {
        println!("\n  ✓ Non-overlapping intervals → Statistically meaningful difference");
    } else {
        println!("\n  ⚠ Overlapping intervals → Difference may not be significant");
    }

    // Raw data for paper
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  RAW DATA (for paper appendix)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!(
        "Python attention values: {:?}",
        python_attentions
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "Rust attention values:   {:?}",
        rust_attentions
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    Ok(())
}

fn analyze_sample(model: &PlipModel, sample: &CodeSample, layer: usize) -> Result<f32> {
    // Get attention analysis
    let analysis = model.analyze_attention(sample.code)?;

    // Find marker token position using the analysis tokens (already strings)
    let marker_pos = analysis
        .tokens
        .iter()
        .position(|t| t.contains(sample.marker_token));

    if marker_pos.is_none() {
        eprintln!(
            "Warning: Marker '{}' not found in sample '{}'",
            sample.marker_token, sample.name
        );
        return Ok(0.0);
    }
    let marker_pos = marker_pos.unwrap();

    // Find function parameter positions
    let param_positions: Vec<usize> = analysis
        .tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| {
            sample
                .fn_param_tokens
                .iter()
                .any(|p| t.contains(p) && t.len() < 10) // Avoid matching in longer tokens
        })
        .map(|(i, _)| i)
        .collect();

    if param_positions.is_empty() {
        eprintln!("Warning: No fn params found in sample '{}'", sample.name);
        return Ok(0.0);
    }

    // Get attention from marker position at specified layer
    let attn_from_marker = analysis
        .cache
        .attention_from_position(layer, marker_pos)
        .ok_or_else(|| anyhow::anyhow!("Could not get attention for layer {}", layer))?;

    // Sum attention from marker position to all param positions
    let mut total_attn: f32 = 0.0;
    for &param_pos in &param_positions {
        if param_pos < attn_from_marker.len() {
            total_attn += attn_from_marker[param_pos];
        }
    }

    Ok(total_attn)
}

fn compute_stats(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();

    (mean, std_dev)
}
