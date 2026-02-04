//! Token Position Verification
//! Prints actual tokenization from Qwen tokenizer to verify positions

use anyhow::Result;
use plip_rs::PlipModel;

fn main() -> Result<()> {
    println!("Loading Qwen tokenizer...\n");
    let model =
        PlipModel::from_pretrained_with_device("Qwen/Qwen2.5-Coder-7B-Instruct", Some(false))?;

    // Python test sample
    let py_code = r#"def add(a, b):
    """
    >>> add(2, 3)
    5
    """
    return a + b"#;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("Python Doctest Sample");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("{}\n", py_code);

    let analysis = model.analyze_attention(py_code)?;
    println!("Tokens:");
    for (i, token) in analysis.tokens.iter().enumerate() {
        let marker = if token.contains(">>>")
            || token.contains(">") && i > 0 && analysis.tokens[i - 1].contains(">")
        {
            " ← POTENTIAL MARKER"
        } else if token == "a" || token == "b" {
            " ← PARAMETER"
        } else {
            ""
        };
        println!("{:3}: {:?}{}", i, token, marker);
    }

    // Rust test sample
    let rust_code = r#"fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[test]
fn test_add() {
    assert_eq!(add(2, 3), 5);
}"#;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("Rust Test Sample");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("{}\n", rust_code);

    let analysis = model.analyze_attention(rust_code)?;
    println!("Tokens:");
    for (i, token) in analysis.tokens.iter().enumerate() {
        let marker = if token.contains("#[")
            || (token.contains("#")
                && i < analysis.tokens.len() - 1
                && analysis.tokens[i + 1].contains("["))
        {
            " ← POTENTIAL MARKER"
        } else if token == "fn" || token == "add" || token == "(" {
            if i < 10 {
                " ← FUNCTION TOKEN"
            } else {
                ""
            }
        } else {
            ""
        };
        println!("{:3}: {:?}{}", i, token, marker);
    }

    // Python baseline
    let py_baseline = r#"def add(a, b):
    # >>> this is just a comment
    return a + b"#;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("Python Baseline");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("{}\n", py_baseline);

    let analysis = model.analyze_attention(py_baseline)?;
    println!("Tokens:");
    for (i, token) in analysis.tokens.iter().enumerate() {
        let marker = if token.contains(">>>") || token.contains(">") {
            " ← MARKER"
        } else if token == "a" || token == "b" {
            " ← PARAMETER"
        } else {
            ""
        };
        println!("{:3}: {:?}{}", i, token, marker);
    }

    Ok(())
}
