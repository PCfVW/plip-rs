//! Verify the poetry corpus for Melometis Phase 1 (**Corpus Validation**).
//!
//! Checks:
//!   1. Schema: 780 samples (260 per category)
//!   2. Position validity: marker_char_pos → '\n', target within ending word
//!   3. Triplet matching: A/B/C share priming + line 3
//!   4. Rhyme group distribution (~13 per group)
//!   5. Priming confound check (priming rhyme ≠ target rhyme)
//!   6. Optional: tokenization test with --model
//!
//! Usage:
//!   cargo run --release --example verify_poetry_corpus
//!   cargo run --release --example verify_poetry_corpus -- --model google/gemma-2-2b

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

#[derive(Parser)]
struct Args {
    /// Path to poetry corpus JSON
    #[arg(long, default_value = "corpus/attention_samples_poetry.json")]
    corpus: String,

    /// Optional model for tokenization test
    #[arg(long)]
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PoetrySample {
    id: String,
    code: String,
    priming_lines: usize,
    marker_char_pos: usize,
    marker_pattern: String,
    target_char_positions: Vec<usize>,
    rhyme_group: String,
    ending_word: String,
    rhyme_word: Option<String>,
    category: String,
    triplet_id: usize,
}

#[derive(Debug, Deserialize)]
struct PoetryCorpus {
    #[allow(dead_code)]
    _format_version: String,
    rhyming: Vec<PoetrySample>,
    non_rhyming: Vec<PoetrySample>,
    generation: Vec<PoetrySample>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Poetry Corpus Verification ===\n");

    // Load corpus
    let corpus_json = fs::read_to_string(&args.corpus).context("Failed to read corpus JSON")?;
    let corpus: PoetryCorpus =
        serde_json::from_str(&corpus_json).context("Failed to parse corpus JSON")?;

    let n_rhyming = corpus.rhyming.len();
    let n_non_rhyming = corpus.non_rhyming.len();
    let n_generation = corpus.generation.len();
    let total = n_rhyming + n_non_rhyming + n_generation;

    println!("Loaded: {total} samples");
    println!("  rhyming:     {n_rhyming}");
    println!("  non_rhyming: {n_non_rhyming}");
    println!("  generation:  {n_generation}");

    // Check 1: Sample counts
    let mut errors = 0u32;
    if n_rhyming != 260 {
        eprintln!("ERROR: expected 260 rhyming samples, got {n_rhyming}");
        errors += 1;
    }
    if n_non_rhyming != 260 {
        eprintln!("ERROR: expected 260 non_rhyming samples, got {n_non_rhyming}");
        errors += 1;
    }
    if n_generation != 260 {
        eprintln!("ERROR: expected 260 generation samples, got {n_generation}");
        errors += 1;
    }
    println!(
        "\n[1] Sample counts: {}",
        if errors == 0 { "OK" } else { "FAIL" }
    );

    // Check 2: Position validity
    let mut pos_errors = 0u32;
    let all_samples: Vec<&PoetrySample> = corpus
        .rhyming
        .iter()
        .chain(corpus.non_rhyming.iter())
        .chain(corpus.generation.iter())
        .collect();

    for s in &all_samples {
        let code = s.code.as_bytes();
        let marker = s.marker_char_pos;

        // marker_char_pos should be within bounds
        if marker > code.len() {
            eprintln!(
                "  ERROR {}: marker_char_pos={marker} > code.len()={}",
                s.id,
                code.len()
            );
            pos_errors += 1;
            continue;
        }

        // For Category C (generation), marker is the trailing \n at end of string
        if marker == code.len() {
            if s.category != "C" {
                eprintln!(
                    "  ERROR {}: marker at end of string but category is {}",
                    s.id, s.category
                );
                pos_errors += 1;
            }
        } else if code[marker] != b'\n' {
            eprintln!(
                "  ERROR {}: code[{marker}] = {:?}, expected '\\n'",
                s.id, code[marker] as char
            );
            pos_errors += 1;
        }

        // marker_pattern should be "\n"
        if s.marker_pattern != "\n" {
            eprintln!(
                "  ERROR {}: marker_pattern = {:?}, expected \"\\n\"",
                s.id, s.marker_pattern
            );
            pos_errors += 1;
        }

        // target_char_positions should be within bounds and contiguous
        if s.target_char_positions.is_empty() {
            eprintln!("  ERROR {}: target_char_positions is empty", s.id);
            pos_errors += 1;
        } else {
            for (i, &pos) in s.target_char_positions.iter().enumerate() {
                if pos >= code.len() {
                    eprintln!(
                        "  ERROR {}: target_char_pos[{i}]={pos} >= code.len()={}",
                        s.id,
                        code.len()
                    );
                    pos_errors += 1;
                }
                // Check contiguity
                if i > 0 && pos != s.target_char_positions[i - 1] + 1 {
                    eprintln!(
                        "  ERROR {}: target positions not contiguous at index {i}",
                        s.id
                    );
                    pos_errors += 1;
                }
            }

            // Check that target positions form the ending_word
            let start = s.target_char_positions[0];
            let end = *s.target_char_positions.last().unwrap() + 1;
            if end <= code.len() {
                let extracted = std::str::from_utf8(&code[start..end]).unwrap_or("<invalid utf8>");
                if extracted != s.ending_word {
                    eprintln!(
                        "  ERROR {}: extracted '{}' != ending_word '{}'",
                        s.id, extracted, s.ending_word
                    );
                    pos_errors += 1;
                }
            }

            // Check target positions are before the marker (ending word precedes \n)
            if *s.target_char_positions.last().unwrap() >= marker {
                eprintln!("  ERROR {}: target positions extend past marker", s.id);
                pos_errors += 1;
            }
        }

        // priming_lines should be 2
        if s.priming_lines != 2 {
            eprintln!(
                "  ERROR {}: priming_lines={}, expected 2",
                s.id, s.priming_lines
            );
            pos_errors += 1;
        }
    }
    errors += pos_errors;
    println!(
        "[2] Position validity: {} ({} samples checked)",
        if pos_errors == 0 { "OK" } else { "FAIL" },
        all_samples.len()
    );

    // Check 3: Triplet matching
    let mut triplet_errors = 0u32;
    let a_by_triplet: HashMap<usize, &PoetrySample> =
        corpus.rhyming.iter().map(|s| (s.triplet_id, s)).collect();
    let b_by_triplet: HashMap<usize, &PoetrySample> = corpus
        .non_rhyming
        .iter()
        .map(|s| (s.triplet_id, s))
        .collect();
    let c_by_triplet: HashMap<usize, &PoetrySample> = corpus
        .generation
        .iter()
        .map(|s| (s.triplet_id, s))
        .collect();

    for tid in 0..260 {
        let a = a_by_triplet.get(&tid);
        let b = b_by_triplet.get(&tid);
        let c = c_by_triplet.get(&tid);

        if a.is_none() || b.is_none() || c.is_none() {
            eprintln!("  ERROR: triplet {tid} missing a category");
            triplet_errors += 1;
            continue;
        }
        let a = a.unwrap();
        let b = b.unwrap();
        let c = c.unwrap();

        // A, B, C should share the same first 3 lines (priming + line 3)
        let a_lines: Vec<&str> = a.code.split('\n').collect();
        let b_lines: Vec<&str> = b.code.split('\n').collect();
        let c_lines: Vec<&str> = c.code.split('\n').collect();

        // Lines 0, 1, 2 should match across A, B, C
        for line_idx in 0..3 {
            if a_lines.get(line_idx) != b_lines.get(line_idx) {
                eprintln!("  ERROR triplet {tid}: line {line_idx} differs between A and B");
                triplet_errors += 1;
            }
            if a_lines.get(line_idx) != c_lines.get(line_idx) {
                eprintln!("  ERROR triplet {tid}: line {line_idx} differs between A and C");
                triplet_errors += 1;
            }
        }

        // A should have 4 lines, B should have 4 lines, C should have 3 lines + trailing empty
        if a_lines.len() != 4 {
            eprintln!(
                "  ERROR triplet {tid}: Cat A has {} lines, expected 4",
                a_lines.len()
            );
            triplet_errors += 1;
        }
        if b_lines.len() != 4 {
            eprintln!(
                "  ERROR triplet {tid}: Cat B has {} lines, expected 4",
                b_lines.len()
            );
            triplet_errors += 1;
        }
        // Cat C ends with \n, so split produces 4 elements with last being empty
        if c_lines.len() != 4 || !c_lines[3].is_empty() {
            eprintln!(
                "  ERROR triplet {tid}: Cat C should end with trailing newline (got {} lines, last={:?})",
                c_lines.len(),
                c_lines.last()
            );
            triplet_errors += 1;
        }

        // A and B should share marker_char_pos and target_char_positions
        if a.marker_char_pos != b.marker_char_pos {
            eprintln!(
                "  ERROR triplet {tid}: marker_char_pos differs: A={}, B={}",
                a.marker_char_pos, b.marker_char_pos
            );
            triplet_errors += 1;
        }
        if a.target_char_positions != b.target_char_positions {
            eprintln!("  ERROR triplet {tid}: target_char_positions differ between A and B");
            triplet_errors += 1;
        }

        // Same rhyme_group and ending_word across all three
        if a.rhyme_group != b.rhyme_group || a.rhyme_group != c.rhyme_group {
            eprintln!("  ERROR triplet {tid}: rhyme_group differs across categories");
            triplet_errors += 1;
        }
        if a.ending_word != b.ending_word || a.ending_word != c.ending_word {
            eprintln!("  ERROR triplet {tid}: ending_word differs across categories");
            triplet_errors += 1;
        }

        // A should have a rhyme_word, B and C should not
        if a.rhyme_word.is_none() {
            eprintln!("  ERROR triplet {tid}: Cat A missing rhyme_word");
            triplet_errors += 1;
        }
        if b.rhyme_word.is_some() {
            eprintln!("  ERROR triplet {tid}: Cat B should not have rhyme_word");
            triplet_errors += 1;
        }
        if c.rhyme_word.is_some() {
            eprintln!("  ERROR triplet {tid}: Cat C should not have rhyme_word");
            triplet_errors += 1;
        }
    }
    errors += triplet_errors;
    println!(
        "[3] Triplet matching: {} (260 triplets checked)",
        if triplet_errors == 0 { "OK" } else { "FAIL" }
    );

    // Check 4: Rhyme group distribution
    let mut group_counts: HashMap<&str, usize> = HashMap::new();
    for s in &corpus.rhyming {
        *group_counts.entry(s.rhyme_group.as_str()).or_default() += 1;
    }
    let n_groups = group_counts.len();
    let mut dist_errors = 0u32;
    println!("\n  Rhyme group distribution ({n_groups} groups):");
    let mut sorted_groups: Vec<_> = group_counts.iter().collect();
    sorted_groups.sort_by_key(|(name, _)| *name);
    for (group, count) in &sorted_groups {
        let status = if **count == 13 { "ok" } else { "MISMATCH" };
        println!("    {group:>8}: {count:>3}  {status}");
        if **count != 13 {
            dist_errors += 1;
        }
    }
    if n_groups != 20 {
        eprintln!("  ERROR: expected 20 rhyme groups, got {n_groups}");
        dist_errors += 1;
    }
    errors += dist_errors;
    println!(
        "[4] Rhyme group distribution: {}",
        if dist_errors == 0 { "OK" } else { "FAIL" }
    );

    // Check 5: Priming confound (priming rhyme ≠ target rhyme)
    // We detect the priming couplet's rhyme by checking if the priming lines
    // end with words that belong to the target rhyme group
    println!("\n[5] Priming confound check: (manual inspection recommended)");
    println!("  Spot-checking first 5 triplets:");
    for s in corpus.rhyming.iter().take(5) {
        let lines: Vec<&str> = s.code.split('\n').collect();
        let priming_end_1 = lines[0].split_whitespace().last().unwrap_or("");
        let priming_end_2 = lines[1].split_whitespace().last().unwrap_or("");
        println!(
            "  triplet {:>3} (target group: {:>5}): priming ends with '{}' / '{}'",
            s.triplet_id, s.rhyme_group, priming_end_1, priming_end_2
        );
    }

    // Check 6: Optional tokenization test
    if let Some(model_id) = &args.model {
        println!("\n=== Tokenization Test with {model_id} ===");
        let model = plip_rs::PlipModel::from_pretrained_with_device(model_id, Some(false))?;

        let mut tok_errors = 0u32;
        let mut tok_success = 0u32;

        for s in all_samples.iter().take(30) {
            // Tokenize with offsets
            let encoding = model.tokenize_with_offsets(&s.code)?;

            // Check marker position resolves to a token
            let marker_token = encoding.char_to_token(s.marker_char_pos);
            if marker_token.is_none() {
                eprintln!(
                    "  WARN {}: marker_char_pos={} didn't map to token",
                    s.id, s.marker_char_pos
                );
                tok_errors += 1;
            }

            // Check target positions resolve
            for &pos in &s.target_char_positions {
                if encoding.char_to_token(pos).is_none() {
                    // Try fuzzy
                    if encoding.char_to_token_fuzzy(pos).is_none() {
                        eprintln!("  WARN {}: target_char_pos={pos} didn't map to token", s.id);
                        tok_errors += 1;
                    }
                }
            }
            tok_success += 1;
        }
        println!("[6] Tokenization: {tok_success} samples tested, {tok_errors} warnings");
        errors += tok_errors;
    } else {
        println!("\n[6] Tokenization: skipped (use --model to enable)");
    }

    // Summary
    println!("\n{}", "=".repeat(60));
    if errors == 0 {
        println!("*** ALL CHECKS PASSED ({total} samples) ***");
    } else {
        println!("*** {errors} ERRORS FOUND ***");
        std::process::exit(1);
    }

    Ok(())
}
