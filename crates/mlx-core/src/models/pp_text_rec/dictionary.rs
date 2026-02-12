//! Character Dictionary
//!
//! Loads the character-to-index mapping from a text file.
//! Each line in the dictionary file contains one character.
//! Index 0 is reserved for the CTC blank token.

use napi::bindgen_prelude::*;
use std::fs;
use std::path::Path;

/// Character dictionary for CTC decoding.
pub struct CharDictionary {
    /// Index → character mapping (index 0 = blank)
    chars: Vec<String>,
}

impl CharDictionary {
    /// Load a dictionary from a text file.
    ///
    /// Each line contains one character. The blank token is automatically
    /// prepended at index 0.
    ///
    /// `use_space_char` controls whether a space character is appended to the
    /// dictionary. PaddleOCR's `BaseRecLabelDecode.__init__()` only adds space
    /// when `use_space_char=True` (see `rec_postprocess.py`).
    /// PP-OCRv5_server_rec.yml sets `use_space_char: true`, so the default is true.
    pub fn load(dict_path: &str) -> Result<Self> {
        Self::load_with_options(dict_path, true)
    }

    /// Load a dictionary with explicit `use_space_char` control.
    ///
    /// When `use_space_char` is true, a space token `" "` is appended after all
    /// dictionary characters, matching PaddleOCR's behavior.
    pub fn load_with_options(dict_path: &str, use_space_char: bool) -> Result<Self> {
        let path = Path::new(dict_path);
        if !path.exists() {
            return Err(Error::from_reason(format!(
                "Dictionary file not found: {}",
                dict_path
            )));
        }

        let content = fs::read_to_string(path)
            .map_err(|e| Error::from_reason(format!("Failed to read dictionary: {e}")))?;

        // Index 0 = blank token
        let mut chars = vec!["".to_string()];

        for line in content.lines() {
            // Only strip trailing \r (lines() already strips \n).
            // Do NOT use trim() — it strips Unicode whitespace like U+3000
            // (ideographic space) which is a valid dictionary character.
            let ch = line.trim_end_matches('\r');
            if !ch.is_empty() {
                chars.push(ch.to_string());
            }
        }

        // PaddleOCR only adds space when use_space_char=True
        // (see BaseRecLabelDecode.__init__ in rec_postprocess.py)
        if use_space_char {
            chars.push(" ".to_string());
        }

        Ok(Self { chars })
    }

    /// Create a dictionary from an explicit character list.
    pub fn from_chars(chars: Vec<String>) -> Self {
        Self { chars }
    }

    /// Get the number of classes (including blank).
    pub fn num_classes(&self) -> usize {
        self.chars.len()
    }

    /// Decode character indices into a string.
    ///
    /// Skips index 0 (blank token) and any out-of-range indices.
    pub fn decode(&self, indices: &[usize]) -> String {
        let mut result = String::new();
        for &idx in indices {
            if idx > 0 && idx < self.chars.len() {
                result.push_str(&self.chars[idx]);
            }
        }
        result
    }

    /// Decode with confidence score (mean of character probabilities).
    pub fn decode_with_score(&self, indices: &[usize], scores: &[f32]) -> (String, f32) {
        let text = self.decode(indices);
        let avg_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        };
        (text, avg_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode() {
        let dict = CharDictionary::from_chars(vec![
            "".to_string(), // blank
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ]);
        assert_eq!(dict.num_classes(), 4);
        assert_eq!(dict.decode(&[1, 2, 3]), "abc");
        assert_eq!(dict.decode(&[0, 1, 0, 2]), "ab"); // blanks skipped
        assert_eq!(dict.decode(&[]), "");
    }

    #[test]
    fn test_decode_with_score() {
        let dict =
            CharDictionary::from_chars(vec!["".to_string(), "h".to_string(), "i".to_string()]);
        let (text, score) = dict.decode_with_score(&[1, 2], &[0.9, 0.8]);
        assert_eq!(text, "hi");
        assert!((score - 0.85).abs() < 1e-6);
    }
}
