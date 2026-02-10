//! RWKV World tokenizer (Trie-based greedy longest-match)
//!
//! The RWKV-6 models use a custom tokenizer that is incompatible with the
//! HuggingFace `tokenizers` crate. This module implements the same algorithm
//! in Rust: a prefix trie built from the vocabulary, with greedy longest-match
//! encoding.
//!
//! Reference: `hf_rwkv_tokenizer.py` in `RWKV/v6-Finch-1B6-HF`

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// A node in the prefix trie
#[derive(Default)]
struct TrieNode {
    children: HashMap<u8, TrieNode>,
    /// If this node represents a complete token, store its ID
    token_id: Option<u32>,
}

/// RWKV World tokenizer using trie-based greedy longest-match
pub struct RwkvTokenizer {
    root: TrieNode,
    /// Token ID → byte sequence
    idx2token: Vec<Vec<u8>>,
    /// For eos_token_id lookups: string representation → token ID
    vocab_map: HashMap<String, u32>,
}

impl RwkvTokenizer {
    /// Load the tokenizer from an RWKV vocabulary file.
    ///
    /// The vocabulary file format (one line per token):
    /// ```text
    /// <index> <python_repr> <byte_length>
    /// ```
    /// Example lines:
    /// ```text
    /// 0 '\x00' 1
    /// 10 '\t' 1
    /// 257 '\t\t' 2
    /// 1000 'Es' 2
    /// 256 b'\xff' 1
    /// ```
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read vocab file: {}", path.display()))?;

        let mut root = TrieNode::default();
        // Pre-allocate for 65536 tokens
        let mut idx2token: Vec<Vec<u8>> = Vec::new();
        let mut vocab_map = HashMap::new();
        let mut max_idx: usize = 0;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Parse: <idx> <token_repr> <byte_length>
            let first_space = line
                .find(' ')
                .with_context(|| format!("Invalid vocab line (no space): {line}"))?;
            let idx: usize = line[..first_space]
                .parse()
                .with_context(|| format!("Invalid index in vocab line: {line}"))?;

            let rest = &line[first_space + 1..];
            let last_space = rest
                .rfind(' ')
                .with_context(|| format!("Invalid vocab line (no second space): {line}"))?;

            let token_repr = &rest[..last_space];
            let expected_len: usize = rest[last_space + 1..]
                .trim()
                .parse()
                .with_context(|| format!("Invalid byte length in vocab line: {line}"))?;

            // Parse the Python string literal into bytes
            let token_bytes = parse_python_literal(token_repr)
                .with_context(|| format!("Failed to parse token repr in line: {line}"))?;

            if token_bytes.len() != expected_len {
                anyhow::bail!(
                    "Token length mismatch for idx {idx}: parsed {} bytes, expected {expected_len}",
                    token_bytes.len()
                );
            }

            // Track max index for vector sizing
            if idx > max_idx {
                max_idx = idx;
            }

            // Grow idx2token if needed
            if idx >= idx2token.len() {
                idx2token.resize(idx + 1, Vec::new());
            }
            idx2token[idx].clone_from(&token_bytes);

            // Build vocab map (for eos_token_id lookups)
            if let Ok(s) = String::from_utf8(token_bytes.clone()) {
                vocab_map.insert(s, idx as u32);
            }

            // Insert into trie
            let token_id = idx as u32;
            let mut node = &mut root;
            for &byte in &token_bytes {
                node = node.children.entry(byte).or_default();
            }
            node.token_id = Some(token_id);
        }

        tracing::info!(
            "Loaded RWKV vocabulary: {} tokens (max idx {})",
            vocab_map.len(),
            max_idx
        );

        Ok(Self {
            root,
            idx2token,
            vocab_map,
        })
    }

    /// Encode text into token IDs using greedy longest-match.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let src = text.as_bytes();
        let mut tokens = Vec::new();
        let mut idx = 0;

        while idx < src.len() {
            let (match_end, token_id) = self.find_longest_match(src, idx);
            if match_end == idx {
                anyhow::bail!(
                    "No matching token at byte position {idx} (byte value 0x{:02x})",
                    src[idx]
                );
            }
            tokens.push(token_id);
            idx = match_end;
        }

        Ok(tokens)
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let id_usize = id as usize;
            if id_usize >= self.idx2token.len() {
                anyhow::bail!("Token ID {id} out of range (vocab size {})", self.idx2token.len());
            }
            bytes.extend_from_slice(&self.idx2token[id_usize]);
        }
        String::from_utf8(bytes)
            .map_err(|e| anyhow::anyhow!("UTF-8 decode error: {e}"))
    }

    /// Get vocabulary mapping (string → token ID) for special token lookups.
    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab_map.clone()
    }

    /// Encode text and return token IDs with byte offsets.
    ///
    /// Each offset pair (start, end) is in bytes (not characters).
    #[allow(clippy::type_complexity)]
    pub fn encode_with_offsets(&self, text: &str) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
        let src = text.as_bytes();
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let mut idx = 0;

        while idx < src.len() {
            let (match_end, token_id) = self.find_longest_match(src, idx);
            if match_end == idx {
                anyhow::bail!(
                    "No matching token at byte position {idx} (byte value 0x{:02x})",
                    src[idx]
                );
            }
            tokens.push(token_id);
            offsets.push((idx, match_end));
            idx = match_end;
        }

        Ok((tokens, offsets))
    }

    /// Find the longest matching token starting at position `start`.
    ///
    /// Returns (end_position, token_id). If no match, returns (start, 0).
    fn find_longest_match(&self, src: &[u8], start: usize) -> (usize, u32) {
        let mut node = &self.root;
        let mut best_end = start;
        let mut best_id = 0u32;
        let mut idx = start;

        while idx < src.len() {
            if let Some(child) = node.children.get(&src[idx]) {
                node = child;
                idx += 1;
                if let Some(token_id) = node.token_id {
                    best_end = idx;
                    best_id = token_id;
                }
            } else {
                break;
            }
        }

        (best_end, best_id)
    }
}

/// Parse a Python string/bytes literal into raw bytes.
///
/// Handles:
/// - `'...'` — Python string literal (with escape sequences)
/// - `b'...'` — Python bytes literal
/// - Escape sequences: `\x##`, `\t`, `\n`, `\r`, `\\`, `\'`, `\"`
#[allow(clippy::too_many_lines)]
fn parse_python_literal(repr: &str) -> Result<Vec<u8>> {
    let repr = repr.trim();

    // Determine if it's a bytes literal (b'...') or string literal ('...')
    let (inner, is_bytes) = if let Some(stripped) = repr.strip_prefix("b'") {
        (
            stripped
                .strip_suffix('\'')
                .with_context(|| format!("Unterminated bytes literal: {repr}"))?,
            true,
        )
    } else if let Some(stripped) = repr.strip_prefix('\'') {
        (
            stripped
                .strip_suffix('\'')
                .with_context(|| format!("Unterminated string literal: {repr}"))?,
            false,
        )
    } else if let Some(stripped) = repr.strip_prefix('"') {
        (
            stripped
                .strip_suffix('"')
                .with_context(|| format!("Unterminated string literal: {repr}"))?,
            false,
        )
    } else {
        anyhow::bail!("Unexpected token representation format: {repr}");
    };

    // Parse escape sequences
    let mut bytes = Vec::new();
    let mut chars = inner.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('x') => {
                    // \xHH — hex byte (or Unicode codepoint for string literals)
                    let h1 = chars
                        .next()
                        .with_context(|| format!("Incomplete \\x escape in: {repr}"))?;
                    let h2 = chars
                        .next()
                        .with_context(|| format!("Incomplete \\x escape in: {repr}"))?;
                    let hex_str: String = [h1, h2].iter().collect();
                    let byte = u8::from_str_radix(&hex_str, 16)
                        .with_context(|| format!("Invalid hex in \\x escape: {hex_str}"))?;
                    if is_bytes {
                        // Bytes literal: raw byte
                        bytes.push(byte);
                    } else {
                        // String literal: \xHH is Unicode codepoint, encode as UTF-8
                        let ch = char::from(byte);
                        let mut buf = [0u8; 4];
                        let encoded = ch.encode_utf8(&mut buf);
                        bytes.extend_from_slice(encoded.as_bytes());
                    }
                }
                Some('u') => {
                    // \uHHHH — 4-digit Unicode escape
                    let mut hex_str = String::with_capacity(4);
                    for _ in 0..4 {
                        hex_str.push(
                            chars
                                .next()
                                .with_context(|| format!("Incomplete \\u escape in: {repr}"))?,
                        );
                    }
                    let codepoint = u32::from_str_radix(&hex_str, 16)
                        .with_context(|| format!("Invalid hex in \\u escape: {hex_str}"))?;
                    let ch = char::from_u32(codepoint)
                        .with_context(|| format!("Invalid Unicode codepoint: U+{hex_str}"))?;
                    let mut buf = [0u8; 4];
                    let encoded = ch.encode_utf8(&mut buf);
                    bytes.extend_from_slice(encoded.as_bytes());
                }
                Some('U') => {
                    // \UHHHHHHHH — 8-digit Unicode escape
                    let mut hex_str = String::with_capacity(8);
                    for _ in 0..8 {
                        hex_str.push(
                            chars
                                .next()
                                .with_context(|| format!("Incomplete \\U escape in: {repr}"))?,
                        );
                    }
                    let codepoint = u32::from_str_radix(&hex_str, 16)
                        .with_context(|| format!("Invalid hex in \\U escape: {hex_str}"))?;
                    let ch = char::from_u32(codepoint)
                        .with_context(|| format!("Invalid Unicode codepoint: U+{hex_str}"))?;
                    let mut buf = [0u8; 4];
                    let encoded = ch.encode_utf8(&mut buf);
                    bytes.extend_from_slice(encoded.as_bytes());
                }
                Some('t') => bytes.push(b'\t'),
                Some('n') => bytes.push(b'\n'),
                Some('r') => bytes.push(b'\r'),
                Some('\\') => bytes.push(b'\\'),
                Some('\'') => bytes.push(b'\''),
                Some('"') => bytes.push(b'"'),
                Some('0') => bytes.push(0),
                Some('a') => bytes.push(0x07), // bell
                Some('b') => bytes.push(0x08), // backspace
                Some('f') => bytes.push(0x0C), // form feed
                Some('v') => bytes.push(0x0B), // vertical tab
                Some(other) => {
                    anyhow::bail!("Unknown escape sequence \\{other} in: {repr}");
                }
                None => {
                    anyhow::bail!("Trailing backslash in: {repr}");
                }
            }
        } else {
            // Regular character — encode as UTF-8 bytes
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            bytes.extend_from_slice(encoded.as_bytes());
        }
    }

    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_python_literal_simple() {
        assert_eq!(parse_python_literal("'hello'").unwrap(), b"hello");
        assert_eq!(parse_python_literal("'a'").unwrap(), b"a");
        assert_eq!(parse_python_literal("' '").unwrap(), b" ");
    }

    #[test]
    fn test_parse_python_literal_escapes() {
        assert_eq!(parse_python_literal("'\\t'").unwrap(), b"\t");
        assert_eq!(parse_python_literal("'\\n'").unwrap(), b"\n");
        assert_eq!(parse_python_literal("'\\r'").unwrap(), b"\r");
        assert_eq!(parse_python_literal("'\\x00'").unwrap(), vec![0u8]);
        // String literal '\\x7f' → U+007F → single UTF-8 byte 0x7F
        assert_eq!(parse_python_literal("'\\x7f'").unwrap(), vec![0x7F]);
        // String literal '\\x80' → U+0080 → two UTF-8 bytes [0xC2, 0x80]
        assert_eq!(parse_python_literal("'\\x80'").unwrap(), vec![0xC2, 0x80]);
        // String literal '\\xff' → U+00FF → two UTF-8 bytes [0xC3, 0xBF]
        assert_eq!(
            parse_python_literal("'\\xff'").unwrap(),
            vec![0xC3, 0xBF]
        );
        assert_eq!(parse_python_literal("'\\t\\t'").unwrap(), b"\t\t");
    }

    #[test]
    fn test_parse_python_literal_bytes() {
        assert_eq!(parse_python_literal("b'\\xff'").unwrap(), vec![0xFF]);
        assert_eq!(parse_python_literal("b'\\x00'").unwrap(), vec![0u8]);
    }

    #[test]
    fn test_trie_basic_encoding() {
        // Build a tiny trie for testing
        let mut root = TrieNode::default();

        // Add single bytes: 'h'=0, 'e'=1, 'l'=2, 'o'=3
        for (byte, id) in [(b'h', 0u32), (b'e', 1), (b'l', 2), (b'o', 3)] {
            let node = root.children.entry(byte).or_default();
            node.token_id = Some(id);
        }
        // Add multi-byte: "he"=4, "hel"=5, "hello"=6
        {
            let h = root.children.entry(b'h').or_default();
            let he = h.children.entry(b'e').or_default();
            he.token_id = Some(4);
            let hel = he.children.entry(b'l').or_default();
            hel.token_id = Some(5);
            let hell = hel.children.entry(b'l').or_default();
            let hello = hell.children.entry(b'o').or_default();
            hello.token_id = Some(6);
        }

        let tokenizer = RwkvTokenizer {
            root,
            idx2token: vec![
                b"h".to_vec(),
                b"e".to_vec(),
                b"l".to_vec(),
                b"o".to_vec(),
                b"he".to_vec(),
                b"hel".to_vec(),
                b"hello".to_vec(),
            ],
            vocab_map: HashMap::new(),
        };

        // "hello" should match as one token (longest match)
        let ids = tokenizer.encode("hello").unwrap();
        assert_eq!(ids, vec![6]);

        // "helo" → "hel" + "o"
        let ids = tokenizer.encode("helo").unwrap();
        assert_eq!(ids, vec![5, 3]);

        // Decode round-trip
        let decoded = tokenizer.decode(&[6]).unwrap();
        assert_eq!(decoded, "hello");
    }
}
