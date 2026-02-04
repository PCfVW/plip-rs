//! Example: Run inference and extract activations
//!
//! Run with: cargo run --example inference

use anyhow::Result;
use plip_rs::PlipModel;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("Loading model...");
    let model = PlipModel::from_pretrained("bigcode/starcoder2-3b")?;

    let code = r#"fn main() {
    println!("Hello, world!");
}"#;

    println!("\n=== Input Code ===");
    println!("{}", code);

    println!("\n=== Extracting Activations ===");
    let cache = model.get_activations(code)?;

    println!("Collected activations for {} layers", cache.n_layers());

    for layer in 0..cache.n_layers() {
        if let Some(activation) = cache.get_layer(layer) {
            println!("  Layer {:2}: shape {:?}", layer, activation.dims());
        }
    }

    Ok(())
}
