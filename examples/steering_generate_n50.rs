//! N=50 Steering Generation Experiment
//!
//! Runs steering generation on 50 diverse Rust prompts to validate
//! test preservation improvement with statistical significance.
//!
//! Usage:
//!   cargo run --release --example steering_generate_n50
//!   cargo run --release --example steering_generate_n50 -- --scale 2.5 --output results.json

use anyhow::Result;
use clap::Parser;
use plip_rs::{PlipModel, SteeringSpec};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "steering_generate_n50")]
#[command(about = "N=50 steering generation experiment")]
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

    /// Path to corpus file
    #[arg(long, default_value = "corpus/generation_prompts_n50.json")]
    corpus: PathBuf,

    /// Output JSON file for results
    #[arg(long)]
    output: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Limit number of samples (for testing)
    #[arg(long)]
    limit: Option<usize>,
}

/// Corpus format
#[derive(Deserialize)]
struct Corpus {
    prompts: Vec<PromptEntry>,
}

#[derive(Deserialize)]
struct PromptEntry {
    id: String,
    category: String,
    prompt: String,
}

/// Result for a single sample
#[derive(Serialize, Clone)]
struct SampleResult {
    id: String,
    category: String,
    baseline_has_test: bool,
    steered_has_test: bool,
    baseline_tokens: usize,
    steered_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline_output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    steered_output: Option<String>,
    change: String,
}

/// Summary statistics
#[derive(Serialize)]
struct ExperimentSummary {
    baseline_preserved: usize,
    steered_preserved: usize,
    total: usize,
    improvement: i32,
    improvement_pct: f64,
    baseline_rate: f64,
    steered_rate: f64,
    fisher_exact_p: Option<f64>,
}

/// Full experiment results
#[derive(Serialize)]
struct ExperimentResults {
    experiment: String,
    model: String,
    layer: usize,
    scale: f32,
    temperature: f32,
    max_tokens: usize,
    chat_template: bool,
    duration_secs: f64,
    samples: Vec<SampleResult>,
    summary: ExperimentSummary,
    by_category: std::collections::HashMap<String, CategoryStats>,
}

#[derive(Serialize, Default)]
struct CategoryStats {
    total: usize,
    baseline_preserved: usize,
    steered_preserved: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("============================================================");
    println!("N=50 STEERING GENERATION EXPERIMENT");
    println!("============================================================\n");

    // Load corpus
    let corpus_path = if args.corpus.is_absolute() {
        args.corpus.clone()
    } else {
        std::env::current_dir()?.join(&args.corpus)
    };
    println!("Loading corpus from: {}", corpus_path.display());
    let corpus_content = fs::read_to_string(&corpus_path)?;
    let corpus: Corpus = serde_json::from_str(&corpus_content)?;

    let prompts: Vec<_> = if let Some(limit) = args.limit {
        corpus.prompts.into_iter().take(limit).collect()
    } else {
        corpus.prompts
    };

    println!("Loaded {} prompts\n", prompts.len());

    // Load model
    println!("Loading model: {}", args.model);
    let model = PlipModel::from_pretrained_with_device(&args.model, Some(args.cpu))?;

    // Determine layer
    let layer = args.layer.unwrap_or_else(|| {
        let n = model.n_layers();
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
    println!("Scale factor: {}x", args.scale);
    println!("Max tokens: {}", args.max_tokens);
    println!("Temperature: {}", args.temperature);
    println!(
        "Chat template: {}",
        if args.chat { "enabled" } else { "disabled" }
    );
    println!();

    // Get stop tokens for chat mode
    let stop_tokens: Vec<u32> = if args.chat {
        model.eos_token_id().into_iter().collect()
    } else {
        vec![]
    };

    // Process all prompts
    let mut results = Vec::with_capacity(prompts.len());
    let mut category_stats: std::collections::HashMap<String, CategoryStats> =
        std::collections::HashMap::new();

    let start_time = Instant::now();

    for (idx, entry) in prompts.iter().enumerate() {
        if args.verbose {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!(
                "[{}/{}] Sample: {} ({})",
                idx + 1,
                prompts.len(),
                entry.id,
                entry.category
            );
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        } else {
            print!(
                "\r[{}/{}] Processing: {}...",
                idx + 1,
                prompts.len(),
                entry.id
            );
            use std::io::Write;
            std::io::stdout().flush()?;
        }

        // Format prompt
        let formatted_prompt = if args.chat {
            model.apply_chat_template(
                &format!("Complete this Rust function:\n\n{}", entry.prompt),
                Some("You are a Rust programming assistant. Complete the function with idiomatic Rust code."),
            )
        } else {
            entry.prompt.clone()
        };

        // Find test marker and function positions
        let test_marker_pos = find_test_marker_position(&model, &formatted_prompt)?;
        let fn_positions = find_fn_positions(&model, &formatted_prompt)?;

        if args.verbose {
            println!("Test marker position: {:?}", test_marker_pos);
            println!("Function positions: {:?}", fn_positions);
        }

        // === BASELINE GENERATION ===
        let baseline_result = model.generate_with_details(
            &formatted_prompt,
            args.max_tokens,
            args.temperature,
            &stop_tokens,
            None,
        )?;

        let baseline_has_test = output_contains_test(&baseline_result.generated_text);

        if args.verbose {
            println!("--- Baseline ---");
            println!(
                "Tokens: {}, Has test: {}",
                baseline_result.generated_tokens.len(),
                baseline_has_test
            );
            println!(
                "Output: {:?}",
                truncate_output(&baseline_result.generated_text, 150)
            );
        }

        // === STEERED GENERATION ===
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
                    baseline_result.clone()
                }
            } else {
                baseline_result.clone()
            };

        let steered_has_test = output_contains_test(&steered_result.generated_text);

        if args.verbose {
            println!("--- Steered ({}x) ---", args.scale);
            println!(
                "Tokens: {}, Has test: {}",
                steered_result.generated_tokens.len(),
                steered_has_test
            );
            println!(
                "Output: {:?}\n",
                truncate_output(&steered_result.generated_text, 150)
            );
        }

        // Record result
        let change = match (baseline_has_test, steered_has_test) {
            (false, true) => "GAINED",
            (true, false) => "LOST",
            (true, true) => "KEPT",
            (false, false) => "NONE",
        };

        // Update category stats
        let cat_stats = category_stats.entry(entry.category.clone()).or_default();
        cat_stats.total += 1;
        if baseline_has_test {
            cat_stats.baseline_preserved += 1;
        }
        if steered_has_test {
            cat_stats.steered_preserved += 1;
        }

        results.push(SampleResult {
            id: entry.id.clone(),
            category: entry.category.clone(),
            baseline_has_test,
            steered_has_test,
            baseline_tokens: baseline_result.generated_tokens.len(),
            steered_tokens: steered_result.generated_tokens.len(),
            baseline_output: if args.verbose {
                Some(baseline_result.generated_text)
            } else {
                None
            },
            steered_output: if args.verbose {
                Some(steered_result.generated_text)
            } else {
                None
            },
            change: change.to_string(),
        });
    }

    let duration = start_time.elapsed();
    println!("\n");

    // Compute summary statistics
    let baseline_preserved: usize = results.iter().filter(|r| r.baseline_has_test).count();
    let steered_preserved: usize = results.iter().filter(|r| r.steered_has_test).count();
    let total = results.len();
    let improvement = steered_preserved as i32 - baseline_preserved as i32;
    let improvement_pct = (improvement as f64 / total as f64) * 100.0;

    // Fisher's exact test (one-tailed)
    // H0: steered rate <= baseline rate
    // Using hypergeometric distribution approximation
    let fisher_p = fisher_exact_one_tailed(
        baseline_preserved,
        total - baseline_preserved,
        steered_preserved,
        total - steered_preserved,
    );

    let summary = ExperimentSummary {
        baseline_preserved,
        steered_preserved,
        total,
        improvement,
        improvement_pct,
        baseline_rate: baseline_preserved as f64 / total as f64 * 100.0,
        steered_rate: steered_preserved as f64 / total as f64 * 100.0,
        fisher_exact_p: fisher_p,
    };

    // Print summary
    println!("============================================================");
    println!("SUMMARY");
    println!("============================================================\n");

    println!("| Sample ID        | Category    | Baseline | Steered | Change |");
    println!("|------------------|-------------|----------|---------|--------|");

    for r in &results {
        println!(
            "| {:16} | {:11} | {:8} | {:7} | {:6} |",
            truncate_str(&r.id, 16),
            truncate_str(&r.category, 11),
            if r.baseline_has_test { "Yes" } else { "No" },
            if r.steered_has_test { "Yes" } else { "No" },
            r.change
        );
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("STATISTICS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Total samples:       {}", total);
    println!(
        "Baseline preserved:  {} ({:.1}%)",
        baseline_preserved, summary.baseline_rate
    );
    println!(
        "Steered preserved:   {} ({:.1}%)",
        steered_preserved, summary.steered_rate
    );
    println!(
        "Improvement:         {:+} ({:+.1}%)",
        improvement, improvement_pct
    );
    if let Some(p) = fisher_p {
        println!("Fisher's exact p:    {:.6}", p);
        if p < 0.05 {
            println!("                     *** SIGNIFICANT (p < 0.05) ***");
        } else if p < 0.10 {
            println!("                     * marginally significant (p < 0.10)");
        }
    }
    println!();

    // Category breakdown
    println!("By Category:");
    for (cat, stats) in &category_stats {
        let cat_improvement = stats.steered_preserved as i32 - stats.baseline_preserved as i32;
        println!(
            "  {:12}: baseline {}/{}, steered {}/{} ({:+})",
            cat,
            stats.baseline_preserved,
            stats.total,
            stats.steered_preserved,
            stats.total,
            cat_improvement
        );
    }
    println!();

    println!(
        "Duration: {:.1}s ({:.2}s per sample)",
        duration.as_secs_f64(),
        duration.as_secs_f64() / total as f64
    );

    // Overall assessment
    println!();
    if improvement > 0 {
        if fisher_p.map_or(false, |p| p < 0.05) {
            println!("RESULT: Steering SIGNIFICANTLY improved test preservation!");
        } else {
            println!(
                "RESULT: Steering improved test preservation (not statistically significant)."
            );
        }
    } else if improvement < 0 {
        println!("RESULT: Steering DECREASED test preservation.");
    } else {
        println!("RESULT: Steering had no effect on test preservation.");
    }

    // Save results to JSON if requested
    if let Some(output_path) = args.output {
        let experiment_results = ExperimentResults {
            experiment: "steering_generation_n50".to_string(),
            model: args.model.clone(),
            layer,
            scale: args.scale,
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            chat_template: args.chat,
            duration_secs: duration.as_secs_f64(),
            samples: results,
            summary,
            by_category: category_stats,
        };

        let json = serde_json::to_string_pretty(&experiment_results)?;
        fs::write(&output_path, &json)?;
        println!("\nResults saved to: {}", output_path.display());
    }

    Ok(())
}

/// Find the token position of #[test] marker in the prompt
fn find_test_marker_position(model: &PlipModel, text: &str) -> Result<Option<usize>> {
    if let Some(char_pos) = text.find("#[test]") {
        let token_pos = model.char_to_token(text, char_pos)?;
        return Ok(token_pos);
    }
    Ok(None)
}

/// Find token positions of function-related tokens
fn find_fn_positions(model: &PlipModel, text: &str) -> Result<Option<Vec<usize>>> {
    let mut positions = Vec::new();

    // Find "fn " keyword - get the last one (the actual function definition)
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

/// Truncate string for table display
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Fisher's exact test (one-tailed) for 2x2 contingency table
/// Returns p-value for testing if steered rate > baseline rate
fn fisher_exact_one_tailed(
    baseline_yes: usize,
    baseline_no: usize,
    steered_yes: usize,
    steered_no: usize,
) -> Option<f64> {
    // Using hypergeometric distribution
    // This is a simplified implementation - for production use a stats library

    let n1 = baseline_yes + baseline_no; // total baseline
    let n2 = steered_yes + steered_no; // total steered
    let k = baseline_yes + steered_yes; // total "yes"
    let n = n1 + n2; // grand total

    if n == 0 || k == 0 || k == n {
        return None;
    }

    // Calculate p-value using hypergeometric probability
    // P(X >= steered_yes) where X ~ Hypergeometric(n, k, n2)

    let mut p_value = 0.0;
    let max_possible = std::cmp::min(k, n2);

    // Calculate log-probability to avoid overflow
    for x in steered_yes..=max_possible {
        let log_prob = log_hypergeom_pmf(n, k, n2, x);
        p_value += log_prob.exp();
    }

    Some(p_value)
}

/// Log of hypergeometric PMF
fn log_hypergeom_pmf(n: usize, k: usize, n2: usize, x: usize) -> f64 {
    // P(X = x) = C(k,x) * C(n-k, n2-x) / C(n, n2)
    log_binomial(k, x) + log_binomial(n - k, n2 - x) - log_binomial(n, n2)
}

/// Log of binomial coefficient using Stirling approximation for large values
fn log_binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }

    log_factorial(n) - log_factorial(k) - log_factorial(n - k)
}

/// Log factorial using Stirling approximation
fn log_factorial(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }

    // Use lookup table for small values
    if n <= 20 {
        let factorials: [f64; 21] = [
            1.0,
            1.0,
            2.0,
            6.0,
            24.0,
            120.0,
            720.0,
            5040.0,
            40320.0,
            362880.0,
            3628800.0,
            39916800.0,
            479001600.0,
            6227020800.0,
            87178291200.0,
            1307674368000.0,
            20922789888000.0,
            355687428096000.0,
            6402373705728000.0,
            121645100408832000.0,
            2432902008176640000.0,
        ];
        return factorials[n].ln();
    }

    // Stirling approximation for large n
    let n_f = n as f64;
    n_f * n_f.ln() - n_f + 0.5 * (2.0 * std::f64::consts::PI * n_f).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_contains_test() {
        assert!(output_contains_test("fn test_foo() { }"));
        assert!(output_contains_test("assert_eq!(1, 1);"));
        assert!(output_contains_test("#[test]"));
        assert!(output_contains_test("assert!(true);"));
        assert!(!output_contains_test("fn main() { }"));
    }

    #[test]
    fn test_fisher_exact() {
        // Test case: baseline 0/10, steered 4/10
        let p = fisher_exact_one_tailed(0, 10, 4, 6);
        assert!(p.is_some());
        let p = p.unwrap();
        // Should be significant
        assert!(p < 0.1);
    }
}
