//! Verify token positions for ALL samples in the full corpus

use anyhow::Result;
use plip_rs::PlipModel;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct Corpus {
    python_doctest: Vec<Sample>,
    rust_test: Vec<Sample>,
    python_baseline: Vec<Sample>,
    rust_baseline: Vec<Sample>,
}

#[derive(Deserialize)]
struct Sample {
    id: String,
    code: String,
}

fn main() -> Result<()> {
    println!("Loading model (for tokenizer)...\n");
    let model =
        PlipModel::from_pretrained_with_device("Qwen/Qwen2.5-Coder-7B-Instruct", Some(false))?;

    let corpus_json = fs::read_to_string("corpus/attention_samples.json")?;
    let corpus: Corpus = serde_json::from_str(&corpus_json)?;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("PYTHON DOCTEST SAMPLES");
    println!("═══════════════════════════════════════════════════════════════════\n");

    for sample in &corpus.python_doctest {
        let analysis = model.analyze_attention(&sample.code)?;
        let tokens = &analysis.tokens;

        // Find >>> marker
        let marker_pos = tokens.iter().position(|t| t.contains(">>>"));

        println!("--- {} ---", sample.id);
        println!("  >>> marker at: {:?}", marker_pos);
        println!("  First 12 tokens:");
        for (i, token) in tokens.iter().take(12).enumerate() {
            let note = if Some(i) == marker_pos {
                " ← >>>"
            } else {
                ""
            };
            println!("    {:2}: {:?}{}", i, token, note);
        }
        println!();
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("RUST TEST SAMPLES");
    println!("═══════════════════════════════════════════════════════════════════\n");

    for sample in &corpus.rust_test {
        let analysis = model.analyze_attention(&sample.code)?;
        let tokens = &analysis.tokens;

        // Find #[ marker
        let marker_pos = tokens.iter().position(|t| t.contains("#["));

        println!("--- {} ---", sample.id);
        println!("  #[ marker at: {:?}", marker_pos);
        println!(
            "  First 5 tokens (fn signature): {:?}",
            &tokens[..5.min(tokens.len())]
        );
        if let Some(pos) = marker_pos {
            println!(
                "  Tokens around #[: {:?}",
                &tokens[pos.saturating_sub(1)..(pos + 3).min(tokens.len())]
            );
        }
        println!();
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("PYTHON BASELINE SAMPLES");
    println!("═══════════════════════════════════════════════════════════════════\n");

    for sample in &corpus.python_baseline {
        let analysis = model.analyze_attention(&sample.code)?;
        let tokens = &analysis.tokens;

        let marker_pos = tokens.iter().position(|t| t.contains(">>>"));

        println!("--- {} ---", sample.id);
        println!("  >>> marker at: {:?}", marker_pos);
        println!("  First 10 tokens: {:?}", &tokens[..10.min(tokens.len())]);
        println!();
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("RUST BASELINE SAMPLES");
    println!("═══════════════════════════════════════════════════════════════════\n");

    for sample in &corpus.rust_baseline {
        let analysis = model.analyze_attention(&sample.code)?;
        let tokens = &analysis.tokens;

        let marker_pos = tokens.iter().position(|t| t.contains("#["));

        println!("--- {} ---", sample.id);
        println!("  #[ marker at: {:?}", marker_pos);
        println!("  First 10 tokens: {:?}", &tokens[..10.min(tokens.len())]);
        println!();
    }

    Ok(())
}
