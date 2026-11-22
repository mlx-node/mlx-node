//! Adapter bridging HuggingFace `tokenizers::Tokenizer` into `toktrie::TokEnv`.
//!
//! Qwen3 / Qwen3.5 and other GPT-2 family models use ByteLevel BPE where
//! vocabulary strings encode raw bytes through the standard 256-entry
//! byte-to-unicode table. Decoding token strings directly through the inverse
//! byte-level map preserves exact raw UTF-8 byte prefixes (including partial
//! multi-byte codepoints), which `llguidance` and `toktrie` require for exact
//! byte-level constrained decoding.

use std::collections::HashMap;
use std::sync::Arc;

use napi::{Error, Result};
use serde_json::Value;
use tokenizers::Tokenizer;
use toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};

/// Tokenizer-to-bytes decoding strategy inferred from the HF tokenizer pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeMode {
    /// GPT-2 / Qwen / Llama-3 ByteLevel BPE encoding.
    ByteLevel,
    /// SentencePiece / Metaspace / `<0xHH>` ByteFallback encoding.
    ByteFallback { space_char: char },
    /// Plain UTF-8 token strings.
    Raw,
}

/// `toktrie::TokenizerEnv` backed by a HuggingFace `tokenizers::Tokenizer`.
pub struct HfTokEnv {
    tokenizer: Arc<Tokenizer>,
    trie: TokTrie,
}

impl HfTokEnv {
    /// Access the underlying HuggingFace tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

impl TokenizerEnv for HfTokEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.trie.tokenize_with_greedy_fallback(s, |text| {
            self.tokenizer
                .encode(text, false)
                .map(|enc| enc.get_ids().to_vec())
                .unwrap_or_else(|_| self.trie.greedy_tokenize(text.as_bytes()))
        })
    }

    /// Mask-only constrained decoding must allow every valid tokenization of the
    /// grammar (including non-canonical BPE splits) so logprobs and token-level
    /// scoring are never distorted and normalizer differences (e.g. NFC) never
    /// trigger forced-token mismatches.
    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}

fn is_byte_level_self_mapped(byte: u8) -> bool {
    matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF)
}

/// Build the inverse GPT-2 ByteLevel unicode-char -> raw byte table.
pub(crate) fn build_byte_level_char_map() -> HashMap<char, u8> {
    let mut map = HashMap::with_capacity(256);
    let mut next_codepoint: u32 = 0x100;
    for byte in 0..=255u8 {
        if is_byte_level_self_mapped(byte) {
            map.insert(char::from(byte), byte);
        } else {
            if let Some(ch) = char::from_u32(next_codepoint) {
                map.insert(ch, byte);
            }
            next_codepoint += 1;
        }
    }
    map
}

fn inspect_pipeline_json(val: &Value) -> Option<DecodeMode> {
    let ty = val.get("type").and_then(Value::as_str);
    match ty {
        Some("ByteLevel") => return Some(DecodeMode::ByteLevel),
        Some("ByteFallback") => return Some(DecodeMode::ByteFallback { space_char: '▁' }),
        Some("Metaspace") => {
            let space_char = val
                .get("replacement")
                .and_then(Value::as_str)
                .and_then(|s| s.chars().next())
                .unwrap_or('▁');
            return Some(DecodeMode::ByteFallback { space_char });
        }
        Some("Sequence") => {
            let children = val
                .get("decoders")
                .or_else(|| val.get("pretokenizers"))
                .and_then(Value::as_array);
            if let Some(items) = children {
                let mut is_byte_fallback = false;
                let mut space_char = '▁';
                for item in items {
                    match item.get("type").and_then(Value::as_str) {
                        Some("ByteLevel") => return Some(DecodeMode::ByteLevel),
                        Some("ByteFallback") => is_byte_fallback = true,
                        Some("Metaspace") => {
                            is_byte_fallback = true;
                            if let Some(ch) = item
                                .get("replacement")
                                .and_then(Value::as_str)
                                .and_then(|s| s.chars().next())
                            {
                                space_char = ch;
                            }
                        }
                        Some("Replace")
                            if item.get("content").and_then(Value::as_str) == Some(" ") =>
                        {
                            if let Some(pattern) = item
                                .get("pattern")
                                .and_then(|p| p.get("String"))
                                .and_then(Value::as_str)
                            {
                                let mut chars = pattern.chars();
                                if let (Some(ch), None) = (chars.next(), chars.next()) {
                                    space_char = ch;
                                    is_byte_fallback = true;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if is_byte_fallback {
                    return Some(DecodeMode::ByteFallback { space_char });
                }
            }
        }
        _ => {}
    }
    None
}

fn detect_decode_mode(tokenizer: &Tokenizer) -> DecodeMode {
    if let Some(decoder) = tokenizer.get_decoder()
        && let Ok(val) = serde_json::to_value(decoder)
        && let Some(mode) = inspect_pipeline_json(&val)
    {
        return mode;
    }
    if let Some(pre_tok) = tokenizer.get_pre_tokenizer()
        && let Ok(val) = serde_json::to_value(pre_tok)
        && let Some(mode) = inspect_pipeline_json(&val)
    {
        return mode;
    }
    DecodeMode::Raw
}

fn decode_vocab_token(
    tok_str: &str,
    mode: DecodeMode,
    byte_level_map: &HashMap<char, u8>,
) -> Vec<u8> {
    match mode {
        DecodeMode::ByteLevel => {
            let mut bytes = Vec::with_capacity(tok_str.len());
            for ch in tok_str.chars() {
                if let Some(&b) = byte_level_map.get(&ch) {
                    bytes.push(b);
                } else {
                    // Fallback if a non-ByteLevel control string appears in base vocab.
                    return tok_str.as_bytes().to_vec();
                }
            }
            bytes
        }
        DecodeMode::ByteFallback { space_char } => {
            if tok_str.len() == 6
                && tok_str.starts_with("<0x")
                && tok_str.ends_with('>')
                && let Ok(byte) = u8::from_str_radix(&tok_str[3..5], 16)
            {
                return vec![byte];
            }
            tok_str.replace(space_char, " ").into_bytes()
        }
        DecodeMode::Raw => tok_str.as_bytes().to_vec(),
    }
}

/// Build a `toktrie::TokEnv` directly from a HuggingFace `tokenizers::Tokenizer`.
///
/// * `vocab_size`: Model vocabulary size (may be larger than the tokenizer's
///   vocab when model embeddings are padded, e.g. Qwen3 151,936 vs 151,669).
/// * `eos_token_id`: Primary EOS token ID (must be `< vocab_size`).
/// * `stop_token_ids`: Optional additional stop token IDs (e.g. `<|im_end|>`,
///   `<|endoftext|>`). `eos_token_id` is always included first.
pub fn build_tok_env(
    tokenizer: Arc<Tokenizer>,
    vocab_size: usize,
    eos_token_id: u32,
    stop_token_ids: &[u32],
) -> Result<TokEnv> {
    if vocab_size == 0 {
        return Err(Error::from_reason("vocab_size must be > 0"));
    }
    if vocab_size >= 0x00ff_ffff {
        return Err(Error::from_reason(format!(
            "vocab_size {vocab_size} exceeds TokTrie maximum (16,777,214)"
        )));
    }
    if (eos_token_id as usize) >= vocab_size {
        return Err(Error::from_reason(format!(
            "eos_token_id {eos_token_id} out of range for vocab_size {vocab_size}"
        )));
    }

    let mut eos_tokens = Vec::with_capacity(stop_token_ids.len() + 1);
    eos_tokens.push(eos_token_id);
    for &id in stop_token_ids {
        if (id as usize) < vocab_size && !eos_tokens.contains(&id) {
            eos_tokens.push(id);
        }
    }

    let mode = detect_decode_mode(&tokenizer);
    let byte_level_map = if mode == DecodeMode::ByteLevel {
        build_byte_level_char_map()
    } else {
        HashMap::new()
    };

    let mut words: Vec<Vec<u8>> = vec![Vec::new(); vocab_size];
    let mut is_added = vec![false; vocab_size];

    let added_decoder = tokenizer.get_added_vocabulary().get_added_tokens_decoder();
    for (&id, added) in added_decoder {
        let idx = id as usize;
        if idx >= vocab_size {
            continue;
        }
        let mut bytes = added.content.as_bytes().to_vec();
        if added.special || eos_tokens.contains(&id) {
            bytes.insert(0, TokTrie::SPECIAL_TOKEN_MARKER);
        }
        words[idx] = bytes;
        is_added[idx] = true;
    }

    for (tok_str, id) in tokenizer.get_vocab(false) {
        let idx = id as usize;
        if idx >= vocab_size || is_added[idx] {
            continue;
        }
        words[idx] = decode_vocab_token(&tok_str, mode, &byte_level_map);
    }

    let info = TokRxInfo::new(vocab_size as u32, eos_token_id);
    let trie = TokTrie::from(&info, &words).with_eos_tokens(&eos_tokens);

    Ok(Arc::new(HfTokEnv { tokenizer, trie }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn clef_tokenizer_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../__test__/fixtures/clef/tokenizer.json")
    }

    #[test]
    fn test_clef_tokenizer_round_trip() {
        let path = clef_tokenizer_path();
        let tokenizer = Arc::new(Tokenizer::from_file(&path).expect("load clef tokenizer.json"));
        let vocab_size = tokenizer.get_vocab_size(true);
        let tok_env = build_tok_env(tokenizer.clone(), vocab_size, 0, &[0]).expect("build_tok_env");
        let trie = tok_env.tok_trie();

        assert_eq!(trie.vocab_size(), vocab_size);
        assert_eq!(trie.eos_token(), 0);
        assert!(trie.is_special_token(0));

        let samples = [
            "Hello, world!",
            r#"{"key": "value with \"quotes\" and \\backslash\\ and \n newline"}"#,
            "12345.6789 -0.001 1e+10",
            "café naïve résumé 日本語 한국어 🚀✨",
            "true false null",
        ];

        for text in samples {
            let ids = tok_env.tokenize(text);
            assert!(!ids.is_empty(), "empty encoding for {text:?}");
            let decoded = trie.decode(&ids);
            assert_eq!(
                String::from_utf8(decoded).expect("valid utf8"),
                text,
                "round-trip mismatch for {text:?}"
            );
        }
    }

    #[test]
    fn test_qwen_im_end_eos_and_padded_vocab() {
        // Build a ByteLevel tokenizer with `<|im_end|>` added token (like Qwen3/Qwen3.5)
        // by cloning the real clef tokenizer and adding `<|im_end|>` + `<|endoftext|>`.
        let path = clef_tokenizer_path();
        let mut tok = Tokenizer::from_file(&path).expect("load clef tokenizer.json");
        tok.add_special_tokens([
            tokenizers::AddedToken::from("<|endoftext|>", true),
            tokenizers::AddedToken::from("<|im_end|>", true),
        ])
        .expect("add special tokens");
        let endoftext_id = tok.token_to_id("<|endoftext|>").expect("endoftext id");
        let im_end_id = tok.token_to_id("<|im_end|>").expect("im_end id");
        // Pad vocab_size beyond tokenizer vocab (like Qwen3 151936 vs 151669).
        let padded_vocab_size = tok.get_vocab_size(true) + 128;

        let tok_env = build_tok_env(
            Arc::new(tok),
            padded_vocab_size,
            im_end_id,
            &[im_end_id, endoftext_id],
        )
        .expect("build_tok_env with im_end");

        let trie = tok_env.tok_trie();
        assert_eq!(trie.vocab_size(), padded_vocab_size);
        assert_eq!(trie.eos_token(), im_end_id);
        assert_eq!(trie.eos_tokens(), &[im_end_id, endoftext_id]);
        assert!(trie.is_special_token(im_end_id));
        assert!(trie.is_special_token(endoftext_id));
        assert_eq!(trie.get_special_token("<|im_end|>"), Some(im_end_id));
        assert_eq!(trie.decode_str(&[im_end_id]), "<|im_end|>");
    }

    #[test]
    fn test_full_qwen3_tokenizer_if_present() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../test-output/final/tokenizer.json");
        if !path.exists() {
            return;
        }
        let tokenizer = Arc::new(Tokenizer::from_file(&path).expect("load qwen3 tokenizer.json"));
        let im_end = tokenizer
            .token_to_id("<|im_end|>")
            .expect("qwen3 <|im_end|>");
        let endoftext = tokenizer
            .token_to_id("<|endoftext|>")
            .expect("qwen3 <|endoftext|>");
        let vocab_size = tokenizer.get_vocab_size(true).max(151_936);

        let tok_env = build_tok_env(tokenizer, vocab_size, im_end, &[im_end, endoftext])
            .expect("build qwen3 tok_env");
        let trie = tok_env.tok_trie();
        assert_eq!(trie.eos_token(), im_end);
        assert!(trie.is_special_token(im_end));

        let text = r#"{"amount": 12.5, "date": "2024-03-15", "note": "café 日本語 \"ok\" 🚀"}"#;
        let ids = tok_env.tokenize(text);
        assert_eq!(String::from_utf8(trie.decode(&ids)).expect("utf8"), text);
    }

    #[test]
    fn test_invalid_inputs_return_err() {
        let path = clef_tokenizer_path();
        let tokenizer = Arc::new(Tokenizer::from_file(&path).expect("load clef tokenizer.json"));
        assert!(build_tok_env(tokenizer.clone(), 0, 0, &[]).is_err());
        assert!(build_tok_env(tokenizer, 10, 10, &[]).is_err());
    }
}
