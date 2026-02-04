//! Example: Tokenize code samples with StarCoder2 tokenizer
//!
//! Run with: cargo run --example tokenize

use anyhow::{Context, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    println!("Loading StarCoder2 tokenizer...");

    // Download tokenizer from HuggingFace
    let api = Api::new()?;
    let repo = api.repo(Repo::new(
        "bigcode/starcoder2-3b".to_string(),
        RepoType::Model,
    ));
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("Failed to download tokenizer.json")?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

    // Sample Python code
    let python_code = r#"def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"#;

    // Sample Rust code
    let rust_code = r#"fn fibonacci(n: u32) -> u32 {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}"#;

    println!("\n=== Python Code ===");
    println!("{}", python_code);
    let py_enc = tokenizer
        .encode(python_code, false)
        .map_err(|e| anyhow::anyhow!("Encoding error: {}", e))?;
    println!("\nTokens: {:?}", py_enc.get_tokens());
    println!("Length: {} tokens", py_enc.get_ids().len());

    println!("\n=== Rust Code ===");
    println!("{}", rust_code);
    let rs_enc = tokenizer
        .encode(rust_code, false)
        .map_err(|e| anyhow::anyhow!("Encoding error: {}", e))?;
    println!("\nTokens: {:?}", rs_enc.get_tokens());
    println!("Length: {} tokens", rs_enc.get_ids().len());

    Ok(())
}
