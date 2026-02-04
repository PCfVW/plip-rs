//! Logit Lens: Interpretability through intermediate layer predictions
//!
//! Projects activations from any layer through the final layer norm and
//! unembedding matrix to see what the model would predict at that layer.
//!
//! This is the first implementation of Logit Lens using an all-Rust stack.

use tokenizers::Tokenizer;

/// Result of applying logit lens at a single layer
#[derive(Debug, Clone)]
pub struct LogitLensResult {
    /// Layer index (0-indexed)
    pub layer: usize,
    /// Top-k token predictions with probabilities
    pub predictions: Vec<TokenPrediction>,
}

/// A single token prediction
#[derive(Debug, Clone)]
pub struct TokenPrediction {
    /// Token ID
    pub token_id: u32,
    /// Decoded token string
    pub token: String,
    /// Probability (0.0 - 1.0)
    pub probability: f32,
}

/// Full logit lens analysis across all layers
#[derive(Debug)]
pub struct LogitLensAnalysis {
    /// Input text that was analyzed
    pub input_text: String,
    /// Results for each layer
    pub layer_results: Vec<LogitLensResult>,
    /// Number of layers analyzed
    pub n_layers: usize,
}

impl LogitLensAnalysis {
    /// Create a new analysis
    pub fn new(input_text: String, n_layers: usize) -> Self {
        Self {
            input_text,
            layer_results: Vec::with_capacity(n_layers),
            n_layers,
        }
    }

    /// Add a layer's result
    pub fn push(&mut self, result: LogitLensResult) {
        self.layer_results.push(result);
    }

    /// Get the top prediction at each layer
    pub fn top_predictions(&self) -> Vec<(&str, f32)> {
        self.layer_results
            .iter()
            .filter_map(|r| r.predictions.first())
            .map(|p| (p.token.as_str(), p.probability))
            .collect()
    }

    /// Find at which layer a specific token first appears in top-k
    pub fn first_appearance(&self, token: &str, k: usize) -> Option<usize> {
        for result in &self.layer_results {
            let in_top_k = result
                .predictions
                .iter()
                .take(k)
                .any(|p| p.token.contains(token));
            if in_top_k {
                return Some(result.layer);
            }
        }
        None
    }

    /// Print a summary of the analysis
    pub fn print_summary(&self) {
        println!("=== Logit Lens Analysis ===");
        println!("Input: {}", self.input_text);
        println!("\nTop prediction at each layer:");
        for result in &self.layer_results {
            if let Some(top) = result.predictions.first() {
                println!(
                    "  Layer {:2}: {:>12} ({:.1}%)",
                    result.layer,
                    format!("\"{}\"", top.token.replace('\n', "\\n")),
                    top.probability * 100.0
                );
            }
        }
    }

    /// Print detailed predictions for each layer
    pub fn print_detailed(&self, top_k: usize) {
        println!("=== Logit Lens Detailed Analysis ===");
        println!("Input: {}", self.input_text);
        for result in &self.layer_results {
            println!("\nLayer {}:", result.layer);
            for (i, pred) in result.predictions.iter().take(top_k).enumerate() {
                println!(
                    "  {}. {:>15} ({:.2}%)",
                    i + 1,
                    format!("\"{}\"", pred.token.replace('\n', "\\n")),
                    pred.probability * 100.0
                );
            }
        }
    }
}

/// Decode token IDs to strings using the tokenizer
pub fn decode_predictions(
    predictions: &[(u32, f32)],
    tokenizer: &Tokenizer,
) -> Vec<TokenPrediction> {
    predictions
        .iter()
        .map(|(token_id, prob)| {
            let token = tokenizer
                .decode(&[*token_id], false)
                .unwrap_or_else(|_| format!("<{token_id}>"));
            TokenPrediction {
                token_id: *token_id,
                token,
                probability: *prob,
            }
        })
        .collect()
}

/// Helper to format a token for display
pub fn format_token(token: &str) -> String {
    token
        .replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logit_lens_result() {
        let result = LogitLensResult {
            layer: 0,
            predictions: vec![
                TokenPrediction {
                    token_id: 1,
                    token: "fn".to_string(),
                    probability: 0.5,
                },
                TokenPrediction {
                    token_id: 2,
                    token: "def".to_string(),
                    probability: 0.3,
                },
            ],
        };

        assert_eq!(result.layer, 0);
        assert_eq!(result.predictions.len(), 2);
        assert_eq!(result.predictions[0].token, "fn");
    }

    #[test]
    fn test_first_appearance() {
        let mut analysis = LogitLensAnalysis::new("test".to_string(), 3);
        analysis.push(LogitLensResult {
            layer: 0,
            predictions: vec![TokenPrediction {
                token_id: 1,
                token: "a".to_string(),
                probability: 0.5,
            }],
        });
        analysis.push(LogitLensResult {
            layer: 1,
            predictions: vec![TokenPrediction {
                token_id: 2,
                token: "#[test]".to_string(),
                probability: 0.5,
            }],
        });

        assert_eq!(analysis.first_appearance("#[test]", 1), Some(1));
        assert_eq!(analysis.first_appearance("notfound", 1), None);
    }
}
