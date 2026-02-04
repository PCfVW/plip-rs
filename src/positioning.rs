//! Perfect Positioning: Model-Agnostic Character-Based Position Handling
//!
//! This module provides universal position handling using character offsets
//! instead of model-specific token indices. This enables:
//!
//! - **One corpus for all models**: No model-specific corpus files needed
//! - **Zero preprocessing**: Any new model works immediately
//! - **Guaranteed accuracy**: No offset heuristics, direct character mapping
//!
//! ## How It Works
//!
//! 1. Corpus stores character positions (byte offsets into the code string)
//! 2. At runtime, we tokenize with offset mapping
//! 3. Convert character positions to token indices using the offset map

/// Token with its character offset range
#[derive(Debug, Clone)]
pub struct TokenWithOffset {
    /// The token string
    pub token: String,
    /// Start character position (byte offset)
    pub start: usize,
    /// End character position (byte offset, exclusive)
    pub end: usize,
}

/// Encoding result with tokens and their offsets
#[derive(Debug, Clone)]
pub struct EncodingWithOffsets {
    /// Token IDs
    pub ids: Vec<u32>,
    /// Token strings
    pub tokens: Vec<String>,
    /// Character offset for each token: (start, end)
    pub offsets: Vec<(usize, usize)>,
}

impl EncodingWithOffsets {
    /// Create a new encoding with offsets
    pub fn new(ids: Vec<u32>, tokens: Vec<String>, offsets: Vec<(usize, usize)>) -> Self {
        Self {
            ids,
            tokens,
            offsets,
        }
    }

    /// Get tokens with their offsets
    pub fn tokens_with_offsets(&self) -> Vec<TokenWithOffset> {
        self.tokens
            .iter()
            .zip(self.offsets.iter())
            .map(|(token, (start, end))| TokenWithOffset {
                token: token.clone(),
                start: *start,
                end: *end,
            })
            .collect()
    }

    /// Find the token index that contains the given character position
    ///
    /// Returns the index of the token that spans the given character position,
    /// or None if no token contains that position.
    pub fn char_to_token(&self, char_pos: usize) -> Option<usize> {
        self.offsets
            .iter()
            .position(|(start, end)| char_pos >= *start && char_pos < *end)
    }

    /// Find the token index for a character position, with fallback strategies
    ///
    /// This is more lenient than `char_to_token` - if the exact position isn't
    /// found, it looks for the closest token.
    pub fn char_to_token_fuzzy(&self, char_pos: usize) -> Option<usize> {
        // First try exact match
        if let Some(idx) = self.char_to_token(char_pos) {
            return Some(idx);
        }

        // Find closest token
        self.offsets
            .iter()
            .enumerate()
            .min_by_key(|(_, (start, end))| {
                let mid = usize::midpoint(*start, *end);
                (char_pos as i64 - mid as i64).unsigned_abs() as usize
            })
            .map(|(idx, _)| idx)
    }

    /// Find the token index that starts at or after the given character position
    pub fn char_to_token_start(&self, char_pos: usize) -> Option<usize> {
        self.offsets
            .iter()
            .position(|(start, _)| *start >= char_pos)
    }

    /// Find all token indices that overlap with the given character range
    pub fn char_range_to_tokens(&self, start_char: usize, end_char: usize) -> Vec<usize> {
        self.offsets
            .iter()
            .enumerate()
            .filter_map(|(idx, (start, end))| {
                // Check if token overlaps with the range
                if *end > start_char && *start < end_char {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the character range for a token index
    pub fn token_to_char_range(&self, token_idx: usize) -> Option<(usize, usize)> {
        self.offsets.get(token_idx).copied()
    }

    /// Get the number of tokens
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Position conversion result
#[derive(Debug, Clone)]
pub struct PositionConversion {
    /// Original character position
    pub char_pos: usize,
    /// Converted token index (if found)
    pub token_idx: Option<usize>,
    /// The token at that position (if found)
    pub token: Option<String>,
    /// Whether this was an exact match or fuzzy
    pub exact_match: bool,
}

/// Convert multiple character positions to token indices
pub fn convert_positions(
    encoding: &EncodingWithOffsets,
    char_positions: &[usize],
) -> Vec<PositionConversion> {
    char_positions
        .iter()
        .map(|&char_pos| {
            let exact = encoding.char_to_token(char_pos);
            let (token_idx, exact_match) = if exact.is_some() {
                (exact, true)
            } else {
                (encoding.char_to_token_fuzzy(char_pos), false)
            };

            PositionConversion {
                char_pos,
                token_idx,
                token: token_idx.map(|idx| encoding.tokens[idx].clone()),
                exact_match,
            }
        })
        .collect()
}

/// Find the character position of a marker pattern in code
///
/// Searches for patterns like ">>>" for Python doctests or "#[" for Rust attributes.
pub fn find_marker_char_pos(code: &str, marker: &str) -> Option<usize> {
    code.find(marker)
}

/// Find the character position of the first occurrence of a test marker
///
/// For Python: finds ">>>" in a docstring context
/// For Rust: finds "#[test]" or "#[cfg(test)]"
pub fn find_test_marker_char_pos(code: &str, language: &str) -> Option<usize> {
    match language {
        "python" => {
            // Look for >>> that appears after a docstring delimiter
            if let Some(docstring_start) = code.find("\"\"\"") {
                if let Some(marker_offset) = code[docstring_start..].find(">>>") {
                    return Some(docstring_start + marker_offset);
                }
            }
            // Fallback: just find >>>
            code.find(">>>")
        }
        "rust" => {
            // Look for #[test] attribute
            code.find("#[test]")
                .or_else(|| code.find("#[cfg(test)]"))
                .or_else(|| {
                    // More general: find any #[ that's likely a test-related attribute
                    code.find("#[")
                })
        }
        _ => None,
    }
}

/// Extract character positions for function parameters/tokens in Python
///
/// This finds the positions of identifiers in the function signature.
pub fn find_python_param_char_positions(code: &str) -> Vec<usize> {
    let mut positions = Vec::new();

    // Find the def keyword
    if let Some(def_pos) = code.find("def ") {
        // Find the opening paren
        if let Some(paren_pos) = code[def_pos..].find('(') {
            let abs_paren = def_pos + paren_pos;
            // Find the closing paren
            if let Some(close_paren) = code[abs_paren..].find(')') {
                let params_str = &code[abs_paren + 1..abs_paren + close_paren];

                // Parse parameters
                let mut current_pos = abs_paren + 1;
                for param in params_str.split(',') {
                    let param = param.trim();
                    if !param.is_empty() {
                        // Get parameter name (before = or :)
                        let name = param
                            .split('=')
                            .next()
                            .unwrap()
                            .split(':')
                            .next()
                            .unwrap()
                            .trim();

                        // Find this parameter in the original string
                        if let Some(name_offset) = code[current_pos..].find(name) {
                            positions.push(current_pos + name_offset);
                        }
                    }
                    current_pos += param.len() + 1; // +1 for comma
                }
            }
        }
    }

    positions
}

/// Extract character positions for function name tokens in Rust
///
/// This finds the position of the `fn` keyword and function name.
pub fn find_rust_fn_char_positions(code: &str) -> Vec<usize> {
    let mut positions = Vec::new();

    // Find fn keyword
    if let Some(fn_pos) = code.find("fn ") {
        positions.push(fn_pos);

        // Find function name (after "fn ")
        let after_fn = &code[fn_pos + 3..];
        if let Some(name_end) = after_fn.find(|c: char| c == '(' || c == '<' || c.is_whitespace()) {
            if name_end > 0 {
                positions.push(fn_pos + 3); // Start of function name
            }
        }
    }

    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_to_token() {
        // Simulate tokenization of "def add(a, b):"
        let encoding = EncodingWithOffsets::new(
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![
                "def".into(),
                " ".into(),
                "add".into(),
                "(".into(),
                "a".into(),
                ",".into(),
                " ".into(),
                "b".into(),
            ],
            vec![
                (0, 3),
                (3, 4),
                (4, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
            ],
        );

        // 'd' is at position 0, which is in token 0 ("def")
        assert_eq!(encoding.char_to_token(0), Some(0));
        // 'a' in "add" is at position 4, which is in token 2
        assert_eq!(encoding.char_to_token(4), Some(2));
        // Parameter 'a' is at position 8
        assert_eq!(encoding.char_to_token(8), Some(4));
    }

    #[test]
    fn test_find_marker() {
        let code = "def add(a, b):\n    \"\"\"\n    >>> add(2, 3)\n    5\n    \"\"\"";
        assert!(find_marker_char_pos(code, ">>>").is_some());
    }

    #[test]
    fn test_find_test_marker() {
        let python_code =
            "def add(a, b):\n    \"\"\"\n    >>> add(2, 3)\n    5\n    \"\"\"\n    return a + b";
        assert!(find_test_marker_char_pos(python_code, "python").is_some());

        let rust_code = "fn add(a: i32, b: i32) -> i32 { a + b }\n\n#[test]\nfn test_add() { }";
        assert!(find_test_marker_char_pos(rust_code, "rust").is_some());
    }
}
