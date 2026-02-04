//! Debug model configuration loading
//!
//! Usage:
//!   cargo run --release --example debug_config

use anyhow::Result;
use hf_hub::{api::sync::Api, Repo, RepoType};

fn main() -> Result<()> {
    let model_id = "Qwen/Qwen2.5-Coder-3B-Instruct";

    println!("Loading config for: {}", model_id);

    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
    let config_path = repo.get("config.json")?;

    let config_str = std::fs::read_to_string(&config_path)?;
    println!("\n=== Raw config.json ===\n{}", config_str);

    // Parse with our struct
    #[derive(Debug, serde::Deserialize)]
    struct Qwen2Config {
        pub hidden_size: usize,
        pub intermediate_size: usize,
        pub num_attention_heads: usize,
        pub num_key_value_heads: usize,
        pub num_hidden_layers: usize,
        pub vocab_size: usize,
        #[serde(default)]
        pub rope_theta: Option<f64>,
        #[serde(default)]
        pub rope_scaling: Option<serde_json::Value>,
        #[serde(default)]
        pub rms_norm_eps: Option<f64>,
        #[serde(default)]
        pub max_position_embeddings: Option<usize>,
        #[serde(default)]
        pub tie_word_embeddings: Option<bool>,
    }

    let config: Qwen2Config = serde_json::from_str(&config_str)?;

    println!("\n=== Parsed config ===");
    println!("hidden_size: {}", config.hidden_size);
    println!("intermediate_size: {}", config.intermediate_size);
    println!("num_attention_heads: {}", config.num_attention_heads);
    println!("num_key_value_heads: {}", config.num_key_value_heads);
    println!("num_hidden_layers: {}", config.num_hidden_layers);
    println!("vocab_size: {}", config.vocab_size);
    println!("rope_theta: {:?}", config.rope_theta);
    println!("rope_scaling: {:?}", config.rope_scaling);
    println!("rms_norm_eps: {:?}", config.rms_norm_eps);
    println!(
        "max_position_embeddings: {:?}",
        config.max_position_embeddings
    );
    println!("tie_word_embeddings: {:?}", config.tie_word_embeddings);

    // Check head_dim calculation
    let head_dim = config.hidden_size / config.num_attention_heads;
    println!(
        "\nhead_dim: {} = {} / {}",
        head_dim, config.hidden_size, config.num_attention_heads
    );

    // KV head ratio
    let kv_ratio = config.num_attention_heads / config.num_key_value_heads;
    println!(
        "KV head ratio: {} = {} / {}",
        kv_ratio, config.num_attention_heads, config.num_key_value_heads
    );

    Ok(())
}
