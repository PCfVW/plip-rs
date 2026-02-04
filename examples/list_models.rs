//! List locally cached HuggingFace models
//!
//! Scans the HuggingFace cache directory and lists models that are compatible
//! with PLIP-rs (StarCoder2, Qwen2, Gemma/CodeGemma architectures).

use std::fs;
use std::path::PathBuf;

fn get_hf_cache_dir() -> Option<PathBuf> {
    // Check standard locations
    if let Some(home) = dirs::home_dir() {
        let hf_cache = home.join(".cache").join("huggingface").join("hub");
        if hf_cache.exists() {
            return Some(hf_cache);
        }
    }

    // Windows alternative
    if let Some(local_app_data) = dirs::data_local_dir() {
        let hf_cache = local_app_data.join("huggingface").join("hub");
        if hf_cache.exists() {
            return Some(hf_cache);
        }
    }

    None
}

fn detect_architecture(model_id: &str) -> &'static str {
    let model_lower = model_id.to_lowercase();
    if model_lower.contains("qwen") {
        "Qwen2"
    } else if model_lower.contains("starcoder") || model_lower.contains("bigcode") {
        "StarCoder2"
    } else if model_lower.contains("gemma") || model_lower.contains("codegemma") {
        "Gemma"
    } else {
        "Unknown"
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  PLIP-rs: List Locally Cached Models");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let cache_dir = match get_hf_cache_dir() {
        Some(dir) => dir,
        None => {
            println!("Could not find HuggingFace cache directory.");
            println!("Expected locations:");
            println!("  - ~/.cache/huggingface/hub");
            println!("  - %LOCALAPPDATA%/huggingface/hub");
            return;
        }
    };

    println!("Cache directory: {}\n", cache_dir.display());

    // List all model directories
    let entries = match fs::read_dir(&cache_dir) {
        Ok(entries) => entries,
        Err(e) => {
            println!("Failed to read cache directory: {}", e);
            return;
        }
    };

    let mut models: Vec<(String, &'static str)> = Vec::new();

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();

        // HuggingFace cache uses "models--org--name" format
        if name.starts_with("models--") {
            let model_id = name
                .strip_prefix("models--")
                .unwrap_or(&name)
                .replace("--", "/");

            let arch = detect_architecture(&model_id);
            models.push((model_id, arch));
        }
    }

    // Sort by architecture, then by name
    models.sort_by(|a, b| {
        if a.1 != b.1 {
            a.1.cmp(b.1)
        } else {
            a.0.cmp(&b.0)
        }
    });

    // Print results
    println!("┌────────────────────────────────────────────────┬────────────┐");
    println!("│ Model ID                                       │ Architecture│");
    println!("├────────────────────────────────────────────────┼────────────┤");

    let mut plip_compatible = 0;
    for (model_id, arch) in &models {
        let compatible = *arch != "Unknown";
        if compatible {
            plip_compatible += 1;
        }

        let marker = if compatible { "✓" } else { " " };
        println!(
            "│ {} {:<44} │ {:>10} │",
            marker,
            if model_id.len() > 44 {
                format!("{}...", &model_id[..41])
            } else {
                model_id.to_string()
            },
            arch
        );
    }

    println!("└────────────────────────────────────────────────┴────────────┘");
    println!("\nTotal models: {}", models.len());
    println!(
        "PLIP-compatible: {} (StarCoder2, Qwen2, Gemma)",
        plip_compatible
    );

    // List PLIP-compatible models for easy copy-paste
    println!("\n─── PLIP-Compatible Models (copy-paste ready) ───\n");
    for (model_id, arch) in &models {
        if *arch != "Unknown" {
            println!("  {} ({})", model_id, arch);
        }
    }
}
