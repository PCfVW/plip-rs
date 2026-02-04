//! List tensor names from StarCoder2 safetensors

use anyhow::Result;
use hf_hub::{api::sync::Api, Repo, RepoType};
use safetensors::tensor::SafeTensors;

fn main() -> Result<()> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(
        "bigcode/starcoder2-3b".to_string(),
        RepoType::Model,
    ));
    let weights_path = repo.get("model.safetensors")?;

    let data = std::fs::read(&weights_path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    println!("=== Tensor names in model.safetensors ===\n");

    for name in tensors.names() {
        if name.contains("lm_head") || name.contains("embed") || name.contains("out") {
            let info = tensors.tensor(name)?;
            println!("{}: {:?}", name, info.shape());
        }
    }

    println!("\n=== All tensor names ===");
    let names: Vec<&String> = tensors.names().into_iter().collect();
    for name in names.iter().take(20) {
        println!("  {}", name);
    }
    println!("  ... and {} more", names.len().saturating_sub(20));

    Ok(())
}
