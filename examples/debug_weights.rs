//! Debug model weight loading
//!
//! Usage:
//!   cargo run --release --example debug_weights

use anyhow::Result;
use hf_hub::{api::sync::Api, Repo, RepoType};
use safetensors::SafeTensors;

fn main() -> Result<()> {
    let model_id = "Qwen/Qwen2.5-Coder-3B-Instruct";

    println!("Loading weights for: {}", model_id);

    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    // Check for index file (sharded model)
    let index_path = repo.get("model.safetensors.index.json")?;
    let index_str = std::fs::read_to_string(&index_path)?;

    #[derive(serde::Deserialize)]
    struct SafetensorsIndex {
        weight_map: std::collections::HashMap<String, String>,
    }

    let index: SafetensorsIndex = serde_json::from_str(&index_str)?;

    println!("\n=== Weight mapping (first 50 entries) ===");
    let mut weights: Vec<_> = index.weight_map.iter().collect();
    weights.sort_by_key(|(k, _)| k.as_str());

    for (i, (weight_name, file_name)) in weights.iter().take(50).enumerate() {
        println!("{}: {} -> {}", i, weight_name, file_name);
    }

    println!("\n... ({} total weights)", index.weight_map.len());

    // Load first shard and check some weight shapes
    let shard_names: std::collections::HashSet<_> = index.weight_map.values().collect();
    if let Some(first_shard) = shard_names.iter().next() {
        let shard_path = repo.get(first_shard)?;
        let shard_data = std::fs::read(&shard_path)?;
        let tensors = SafeTensors::deserialize(&shard_data)?;

        println!("\n=== First shard tensor shapes ===");
        let mut tensor_names: Vec<_> = tensors.names().into_iter().collect();
        tensor_names.sort();

        for name in tensor_names.iter().take(30) {
            let tensor = tensors.tensor(name)?;
            println!(
                "{}: {:?} (dtype: {:?})",
                name,
                tensor.shape(),
                tensor.dtype()
            );
        }
    }

    // Check expected weight names for Qwen2
    println!("\n=== Expected Qwen2 weight names ===");
    println!("Embeddings: model.embed_tokens.weight");
    println!("Layer 0 attn: model.layers.0.self_attn.{{q_proj,k_proj,v_proj,o_proj}}.weight");
    println!("Layer 0 mlp: model.layers.0.mlp.{{gate_proj,up_proj,down_proj}}.weight");
    println!("Layer 0 norms: model.layers.0.{{input_layernorm,post_attention_layernorm}}.weight");
    println!("Final norm: model.norm.weight");
    println!("LM head: lm_head.weight (or tied to embed_tokens)");

    Ok(())
}
