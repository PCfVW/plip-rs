//! Full reproduction pipeline for the Melometis experiments.
//!
//! Runs every step from corpus verification through the Figure 13
//! suppress + inject sweep, with timing and progress reporting.
//!
//! By default, outputs go to `outputs/`. Use `--output-dir` to redirect
//! to a scratch directory for an independent reproduction without
//! overwriting existing results.
//!
//! Usage:
//!   cargo run --release --example reproduce_pipeline
//!   cargo run --release --example reproduce_pipeline -- --output-dir outputs/_repro
//!   cargo run --release --example reproduce_pipeline -- --start-step 8
//!   cargo run --release --example reproduce_pipeline -- --start-step 8 --end-step 9
//!   cargo run --release --example reproduce_pipeline -- --skip-clt-validation

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use clap::Parser;

const TOTAL_STEPS: usize = 13;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "reproduce_pipeline")]
#[command(about = "Full Melometis reproduction pipeline")]
struct Args {
    /// Output directory for all generated files
    #[arg(long, default_value = "outputs")]
    output_dir: PathBuf,

    /// HuggingFace model ID
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// HuggingFace CLT repository
    #[arg(long, default_value = "mntss/clt-gemma-2-2b-426k")]
    clt_repo: String,

    /// Skip CLT validation (requires Python-generated reference file)
    #[arg(long)]
    skip_clt_validation: bool,

    /// First step to run (1-13)
    #[arg(long, default_value_t = 1)]
    start_step: usize,

    /// Last step to run (1-13)
    #[arg(long, default_value_t = TOTAL_STEPS)]
    end_step: usize,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Print a section header with step number and description.
fn header(step: usize, title: &str) {
    let bar = "═".repeat(60);
    eprintln!("\n{bar}");
    eprintln!("  [{step}/{TOTAL_STEPS}] {title}");
    eprintln!("{bar}\n");
}

/// Print a skip notice for a step outside the requested range.
fn skip_notice(step: usize, title: &str) {
    eprintln!("  [{step}/{TOTAL_STEPS}] {title} — skipped");
}

/// Print elapsed time for a step.
fn report_elapsed(step_name: &str, elapsed: std::time::Duration) {
    let secs = elapsed.as_secs();
    let mins = secs / 60;
    let rem = secs % 60;
    if mins > 0 {
        eprintln!("  => {step_name} completed in {mins}m {rem}s");
    } else {
        eprintln!("  => {step_name} completed in {rem}s");
    }
}

/// Find the cargo binary path. Uses `CARGO` env var (set by cargo itself when
/// running examples) or falls back to "cargo" on PATH.
fn cargo_bin() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())
}

/// Run a cargo example with the given arguments. Returns Ok(()) on success,
/// Err with stderr on failure.
fn run_example(name: &str, args: &[&str]) -> Result<(), String> {
    let cargo = cargo_bin();
    let mut cmd = Command::new(&cargo);
    cmd.args(["run", "--release", "--example", name, "--"]);
    cmd.args(args);

    eprintln!("  $ cargo run --release --example {name} -- {}", args.join(" "));
    eprintln!();

    let status = cmd
        .status()
        .map_err(|e| format!("Failed to launch {name}: {e}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("{name} exited with {status}"))
    }
}

/// Ensure a directory exists, creating it if necessary.
fn ensure_dir(path: &Path) -> Result<(), String> {
    if !path.exists() {
        std::fs::create_dir_all(path)
            .map_err(|e| format!("Failed to create {}: {e}", path.display()))?;
    }
    Ok(())
}

/// Check whether a step should run.
fn should_run(step: usize, start: usize, end: usize) -> bool {
    step >= start && step <= end
}

/// Run a step or skip it. On failure, exits the process.
fn run_step(
    step: usize,
    title: &str,
    timing_label: &str,
    start: usize,
    end: usize,
    f: impl FnOnce() -> Result<(), String>,
) {
    if !should_run(step, start, end) {
        skip_notice(step, title);
        return;
    }
    header(step, title);
    let t = Instant::now();
    if let Err(e) = f() {
        eprintln!("FAILED: {e}");
        std::process::exit(1);
    }
    report_elapsed(timing_label, t.elapsed());
}

// ── Pipeline steps ──────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    let out = &args.output_dir;
    let start = args.start_step;
    let end = args.end_step;
    let pipeline_start = Instant::now();

    assert!(
        (1..=TOTAL_STEPS).contains(&start),
        "--start-step must be between 1 and {TOTAL_STEPS}"
    );
    assert!(
        (start..=TOTAL_STEPS).contains(&end),
        "--end-step must be between {start} and {TOTAL_STEPS}"
    );

    ensure_dir(out).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });

    let range_str = if start == 1 && end == TOTAL_STEPS {
        "all".to_string()
    } else {
        format!("{start}-{end}")
    };

    eprintln!("╔════════════════════════════════════════════════════════════╗");
    eprintln!("║          Melometis Reproduction Pipeline                  ║");
    eprintln!("║                                                          ║");
    eprintln!("║  Model:  {}{}║", &args.model, " ".repeat(42_usize.saturating_sub(args.model.len())));
    eprintln!("║  Output: {}{}║", out.display(), " ".repeat(42_usize.saturating_sub(out.display().to_string().len())));
    eprintln!("║  Steps:  {}{}║", &range_str, " ".repeat(42_usize.saturating_sub(range_str.len())));
    eprintln!("╚════════════════════════════════════════════════════════════╝");

    // ── Step 1: Verify poetry corpus (no GPU) ───────────────────────────────

    run_step(1, "Verify poetry corpus (no GPU)", "Corpus verification", start, end, || {
        run_example("verify_poetry_corpus", &[])
    });

    // ── Step 2: Validate CLT encoding against Python reference ──────────────

    if args.skip_clt_validation {
        if should_run(2, start, end) {
            skip_notice(2, "CLT validation (--skip-clt-validation)");
        }
    } else {
        run_step(2, "Validate CLT encoding against Python reference", "CLT validation", start, end, || {
            run_example("validate_clt", &[])
        });
    }

    // ── Step 3: CLT logit-shift acceptance test ─────────────────────────────

    run_step(3, "CLT logit-shift acceptance test (GPU)", "CLT logit-shift", start, end, || {
        run_example("clt_logit_shift", &["--model", &args.model])
    });

    // ── Step 4: Phase 1 — detect planning signal via attention ──────────────

    let layer_scan_output = out.join("poetry_layer_scan_google_gemma_2_2b.json");
    let layer_scan_out_str = layer_scan_output.display().to_string();

    run_step(4, "Phase 1: Detect planning signal via attention (GPU)", "Attention layer scan", start, end, || {
        run_example(
            "poetry_layer_scan",
            &["--model", &args.model, "--output", &layer_scan_out_str],
        )
    });

    // ── Step 5: Phase 2a — CLT steering Methods 3-6 ────────────────────────

    let clt_results_path = out.join("clt_steering_results.json");
    let clt_results_str = clt_results_path.display().to_string();

    run_step(5, "Phase 2a: CLT steering experiments, Methods 3-6 (GPU)", "CLT steering (5 runs)", start, end, || {
        let methods: &[(&str, &str)] = &[
            ("--method3", "method3_results.json"),
            ("--method4", "method4_results.json"),
            ("--method5", "method5_results.json"),
            ("--method6", "method6_test.json"),
        ];

        for (flag, filename) in methods {
            let output_path = out.join(filename);
            let output_str = output_path.display().to_string();
            eprintln!("  --- {flag} ---");
            run_example(
                "poetry_clt_steering",
                &[
                    "--mode", "run",
                    "--model", &args.model,
                    flag,
                    "--output", &output_str,
                ],
            )?;
        }

        // Also run decoder-projection (produces clt_steering_results.json for evaluate_steering)
        eprintln!("  --- --decoder-projection ---");
        run_example(
            "poetry_clt_steering",
            &[
                "--mode", "run",
                "--model", &args.model,
                "--decoder-projection",
                "--output", &clt_results_str,
            ],
        )
    });

    // ── Step 6: Phase 2b — Attention steering comparison ────────────────────

    let attn_results_path = out.join("attention_steering_results.json");
    let attn_results_str = attn_results_path.display().to_string();

    run_step(6, "Phase 2b: Attention steering comparison (GPU)", "Attention steering", start, end, || {
        run_example(
            "poetry_attention_steering",
            &[
                "--mode", "run",
                "--model", &args.model,
                "--output", &attn_results_str,
            ],
        )
    });

    // ── Step 7: Phase 2c — Cross-mechanism evaluation ───────────────────────

    let comparison_path = out.join("steering_comparison.json");
    let comparison_str = comparison_path.display().to_string();

    run_step(7, "Phase 2c: Cross-mechanism evaluation", "Cross-mechanism evaluation", start, end, || {
        run_example(
            "evaluate_steering",
            &[
                "--mode", "compare",
                "--clt-results", &clt_results_str,
                "--attention-results", &attn_results_str,
                "--output", &comparison_str,
            ],
        )
    });

    // ── Step 8: Bottom-up feature discovery — explore vocabulary ─────────────
    //
    // The explore-vocabulary mode defaults to sample-step=16 and last layer
    // only. The full scan needs sample-step=1 across all 26 layers to find
    // enough clean English features for rhyme group formation.

    let explore_vocab_path = out.join("explore_vocab_all_layers.json");
    let explore_vocab_str = explore_vocab_path.display().to_string();
    let all_layers = (0..26).map(|i| i.to_string()).collect::<Vec<_>>().join(",");

    run_step(8, "Bottom-up: Scan CLT features against vocabulary (GPU)", "Vocabulary exploration", start, end, || {
        run_example(
            "poetry_category_steering",
            &[
                "--mode", "explore-vocabulary",
                "--model", &args.model,
                "--sample-step", "1",
                "--layers", &all_layers,
                "--output", &explore_vocab_str,
            ],
        )
    });

    // ── Step 9: Find rhyme pairs via CMU dictionary ─────────────────────────

    let rhyme_pairs_path = out.join("rhyme_pairs_all_layers.json");
    let rhyme_pairs_str = rhyme_pairs_path.display().to_string();

    run_step(9, "Find rhyme pairs via CMU Pronouncing Dictionary", "Rhyme pair discovery", start, end, || {
        run_example(
            "poetry_category_steering",
            &[
                "--mode", "find-rhyme-pairs",
                "--explore-json", &explore_vocab_str,
                "--cmu-dict", "corpus/cmudict.dict",
                "--min-cosine", "0.3",
                "--output", &rhyme_pairs_str,
            ],
        )
    });

    // ── Step 10: Planning detection V2 (completion prompts) ─────────────────

    run_step(10, "Planning detection V2: completion-style probes (GPU)", "Planning detection", start, end, || {
        run_example(
            "poetry_category_steering",
            &[
                "--mode", "detect-planning",
                "--model", &args.model,
                "--rhyme-pairs", &rhyme_pairs_str,
            ],
        )
    });

    // ── Step 11: Version C — multi-layer causal position sweep ──────────────

    let sweep_path = out.join("steering_sweep_multilayer.json");
    let sweep_str = sweep_path.display().to_string();

    run_step(11, "Version C: Multi-layer causal position sweep (GPU)", "Steering sweep (Version C)", start, end, || {
        run_example(
            "steering_sweep",
            &[
                "--model", &args.model,
                "--rhyme-pairs", &rhyme_pairs_str,
                "--output", &sweep_str,
            ],
        )
    });

    // ── Step 12: Version D — suppress + inject (THE MAIN RESULT) ────────────

    let si_path = out.join("suppress_inject_sweep.json");
    let si_str = si_path.display().to_string();

    run_step(12, "Version D: Suppress + inject sweep (GPU)", "Suppress + inject (Version D)", start, end, || {
        run_example(
            "suppress_inject_sweep",
            &[
                "--model", &args.model,
                "--rhyme-pairs", &rhyme_pairs_str,
                "--output", &si_str,
            ],
        )
    });

    // ── Step 13: Analyze Version D results ──────────────────────────────────

    run_step(13, "Analyze suppress + inject results", "Analysis", start, end, || {
        run_example(
            "analyze_suppress_inject",
            &["--input", &si_str],
        )
    });

    // ── Summary ─────────────────────────────────────────────────────────────

    let total_elapsed = pipeline_start.elapsed();
    let total_secs = total_elapsed.as_secs();
    let total_mins = total_secs / 60;
    let total_rem = total_secs % 60;

    eprintln!();
    eprintln!("╔════════════════════════════════════════════════════════════╗");
    eprintln!("║  Pipeline complete!                                      ║");
    eprintln!("║                                                          ║");
    eprintln!("║  Steps:  {range_str:>7} ({start}-{end} of {TOTAL_STEPS})                            ║");
    eprintln!("║  Total time: {total_mins:>3}m {total_rem:>2}s                                   ║");
    eprintln!("║  Output dir: {}{}║", out.display(), " ".repeat(42_usize.saturating_sub(out.display().to_string().len())));
    eprintln!("╚════════════════════════════════════════════════════════════╝");
}
