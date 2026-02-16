//! Inspect CLT safetensors files to discover tensor names, shapes, and dtypes.
//!
//! **CLT Infrastructure** — prerequisite for building clt.rs; we need to know
//! exact tensor naming conventions before implementing loading logic.
//!
//! Usage:
//!   cargo run --release --example `inspect_clt`
//!   cargo run --release --example `inspect_clt` -- --clt-repo mntss/clt-gemma-2-2b-2.5M

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]

use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use safetensors::tensor::SafeTensors;

#[derive(Parser)]
struct Args {
    /// `HuggingFace` CLT repository
    #[arg(long, default_value = "mntss/clt-gemma-2-2b-426k")]
    clt_repo: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let api = Api::new()?;
    let repo = api.repo(Repo::new(args.clt_repo.clone(), RepoType::Model));

    println!("=== CLT Repository: {} ===\n", args.clt_repo);

    // 1. Download and print config.yaml
    match repo.get("config.yaml") {
        Ok(path) => {
            let content = std::fs::read_to_string(&path)?;
            println!("--- config.yaml ---");
            println!("{content}");
            println!();
        }
        Err(e) => println!("No config.yaml found: {e}\n"),
    }

    // 2. Discover how many encoder/decoder files exist
    let mut n_layers: usize = 0;
    loop {
        let enc_name = format!("W_enc_{n_layers}.safetensors");
        if repo.get(&enc_name).is_ok() {
            n_layers += 1;
        } else {
            break;
        }
    }
    println!(
        "Found {n_layers} encoder files (W_enc_0..W_enc_{})\n",
        n_layers.saturating_sub(1)
    );

    // 3. Inspect first encoder file (W_enc_0)
    if n_layers > 0 {
        let enc_path = repo.get("W_enc_0.safetensors")?;
        println!("--- W_enc_0.safetensors ---");
        println!(
            "File size: {} bytes ({:.1} MB)",
            std::fs::metadata(&enc_path)?.len(),
            std::fs::metadata(&enc_path)?.len() as f64 / 1_048_576.0
        );

        let data = std::fs::read(&enc_path).context("Failed to read W_enc_0.safetensors")?;
        let tensors =
            SafeTensors::deserialize(&data).context("Failed to deserialize W_enc_0.safetensors")?;

        let names = tensors.names();
        println!("Tensor count: {}", names.len());
        println!();
        for name in &names {
            let info = tensors.tensor(name)?;
            println!("  {name}:");
            println!("    shape: {:?}", info.shape());
            println!("    dtype: {:?}", info.dtype());
            println!("    data bytes: {}", info.data().len());
        }
        println!();
    }

    // 4. Inspect smallest decoder file (W_dec_{n_layers-1}) first to avoid large download
    if n_layers > 0 {
        let last_dec = format!("W_dec_{}.safetensors", n_layers - 1);
        let dec_path = repo.get(&last_dec)?;
        println!("--- {last_dec} (smallest decoder, layer writes only to itself) ---");
        println!(
            "File size: {} bytes ({:.1} MB)",
            std::fs::metadata(&dec_path)?.len(),
            std::fs::metadata(&dec_path)?.len() as f64 / 1_048_576.0
        );

        let data = std::fs::read(&dec_path).context("Failed to read last decoder file")?;
        let tensors =
            SafeTensors::deserialize(&data).context("Failed to deserialize last decoder file")?;

        let names = tensors.names();
        println!("Tensor count: {}", names.len());
        println!();
        for name in &names {
            let info = tensors.tensor(name)?;
            println!("  {name}:");
            println!("    shape: {:?}", info.shape());
            println!("    dtype: {:?}", info.dtype());
            println!("    data bytes: {}", info.data().len());
        }
        println!();
    }

    // 5. Also inspect first decoder file (W_dec_0) — largest, has most target layers
    if n_layers > 0 {
        println!("--- W_dec_0.safetensors (largest decoder, layer 0 writes to all downstream) ---");
        let dec0_path = repo.get("W_dec_0.safetensors")?;
        println!(
            "File size: {} bytes ({:.1} MB)",
            std::fs::metadata(&dec0_path)?.len(),
            std::fs::metadata(&dec0_path)?.len() as f64 / 1_048_576.0
        );

        // This file is ~1.96 GB — read it (OS virtual memory handles it)
        let data = std::fs::read(&dec0_path).context("Failed to read W_dec_0.safetensors")?;
        let tensors =
            SafeTensors::deserialize(&data).context("Failed to deserialize W_dec_0.safetensors")?;

        let names = tensors.names();
        println!("Tensor count: {}", names.len());
        println!();
        // Print first 10 and last 5 tensor names to see the pattern
        for (i, name) in names.iter().enumerate() {
            let info = tensors.tensor(name)?;
            if i < 10 || i >= names.len().saturating_sub(5) {
                println!("  {name}:");
                println!("    shape: {:?}", info.shape());
                println!("    dtype: {:?}", info.dtype());
                println!("    data bytes: {}", info.data().len());
            } else if i == 10 {
                println!("  ... ({} more tensors) ...", names.len() - 15);
            }
        }
        println!();
    }

    // 6. Summary
    println!("=== Summary ===");
    println!("Layers: {n_layers}");
    if n_layers > 0 {
        let enc_path = repo.get("W_enc_0.safetensors")?;
        let data = std::fs::read(&enc_path)?;
        let tensors = SafeTensors::deserialize(&data)?;
        for name in tensors.names() {
            let info = tensors.tensor(name)?;
            let shape = info.shape();
            if shape.len() == 2 {
                println!(
                    "Encoder weight shape: {:?} → n_features_per_layer={}, d_model={}",
                    shape, shape[0], shape[1]
                );
                println!(
                    "Total features: {} × {} = {}",
                    shape[0],
                    n_layers,
                    shape[0] * n_layers
                );
                break;
            }
        }
    }

    Ok(())
}
