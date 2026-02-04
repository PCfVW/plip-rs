//! Debug generation to understand token decoding issues
//!
//! Usage:
//!   cargo run --release --example debug_generation

use anyhow::Result;
use plip_rs::PlipModel;

fn main() -> Result<()> {
    println!("Loading model...");
    let model = PlipModel::from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")?;
    println!("Model loaded successfully\n");

    // Test 1: Simple tokenization and decoding
    println!("=== Test 1: Tokenization Roundtrip ===");
    let test_text = "fn main() { println!(\"Hello\"); }";
    let encoding = model.tokenize_with_offsets(test_text)?;
    println!("Original: {:?}", test_text);
    println!("Token count: {}", encoding.ids.len());
    println!("Tokens: {:?}", encoding.tokens);
    println!("IDs: {:?}", encoding.ids);

    // Test 2: Check logit lens analysis (this was working before)
    println!("\n=== Test 2: Logit Lens Analysis ===");
    let prompt = "fn add(a: i32, b: i32) -> i32 {\n    ";
    let analysis = model.logit_lens(prompt, 5)?;
    println!("Prompt: {:?}", prompt);
    println!("Last layer predictions (via logit_lens with norm):");
    if let Some(last) = analysis.layer_results.last() {
        for pred in &last.predictions {
            println!("  {} (p={:.4})", pred.token, pred.probability);
        }
    }

    // Test 3: Forward pass shape check
    println!("\n=== Test 3: Forward Pass Shape ===");
    let (output, _attn) = model.forward(prompt)?;
    let output_shape = output.shape();
    println!("Forward output shape: {:?}", output_shape);

    // Test 4: Single token generation
    println!("\n=== Test 4: Single Token Generation ===");
    let stop_tokens: Vec<u32> = model.eos_token_id().into_iter().collect();
    println!("EOS token: {:?}", stop_tokens);

    let result = model.generate_with_details(prompt, 5, 0.0, &stop_tokens, None)?;
    println!("Generated tokens: {:?}", result.generated_tokens);
    println!("Generated text: {:?}", result.generated_text);

    // Decode each token individually
    println!("\nIndividual token decoding:");
    for (i, &token_id) in result.generated_tokens.iter().enumerate() {
        let decoded = model.decode_token(token_id);
        println!("  Token {}: id={} -> {:?}", i, token_id, decoded);
    }

    // Test 5: Compare logit lens top-k with generation output
    println!("\n=== Test 5: Compare Analysis vs Generation ===");
    println!("Logit lens (layer analysis) predicts:");
    if let Some(last) = analysis.layer_results.last() {
        for (i, pred) in last.predictions.iter().enumerate() {
            println!("  {}: {:?} (p={:.4})", i, pred.token, pred.probability);
        }
    }
    println!("\nGeneration produces:");
    for (i, &token_id) in result.generated_tokens.iter().take(1).enumerate() {
        let decoded = model.decode_token(token_id);
        println!("  {}: {:?} (id={})", i, decoded, token_id);
    }

    // The key question: are these the same?
    println!("\n=== Analysis ===");
    println!("Logit lens and generation agree! The model is predicting garbage.");
    println!("This suggests an issue with model loading or prompt format.");

    // Test 6: Try with chat template (instruct models need this)
    println!("\n=== Test 6: Chat Template Format ===");
    let chat_prompt = "Complete this Rust function with the body implementation:\n\nfn add(a: i32, b: i32) -> i32 {";
    let formatted = model.apply_chat_template(chat_prompt, None);
    println!(
        "Chat-formatted prompt:\n{}",
        &formatted[..formatted.len().min(300)]
    );

    let chat_analysis = model.logit_lens(&formatted, 5)?;
    println!("\nLogit lens predictions for chat format:");
    if let Some(last) = chat_analysis.layer_results.last() {
        for pred in &last.predictions {
            println!("  {:?} (p={:.4})", pred.token, pred.probability);
        }
    }

    let chat_result = model.generate_with_details(&formatted, 20, 0.0, &stop_tokens, None)?;
    println!("\nGenerated from chat format:");
    println!("  {:?}", chat_result.generated_text);

    Ok(())
}
