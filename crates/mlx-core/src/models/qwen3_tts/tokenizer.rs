use napi::{Error, Result};
use std::path::Path;
use tokenizers::{
    AddedToken, SplitDelimiterBehavior, Tokenizer,
    models::bpe::BPE,
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
};

pub struct TextTokenizer(Tokenizer);
impl TextTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        if path.join("tokenizer.json").is_file() {
            return Tokenizer::from_file(path.join("tokenizer.json"))
                .map(Self)
                .map_err(|e| Error::from_reason(e.to_string()));
        }
        let vocab = path.join("vocab.json");
        let merges = path.join("merges.txt");
        let model = BPE::from_file(&vocab.to_string_lossy(), &merges.to_string_lossy())
            .build()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let mut tokenizer = Tokenizer::new(model);
        // Qwen2 byte-level BPE pre-tokenization contract. Kept with the family
        // tokenizer rather than imposing a regex on other model families.
        let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let split = Split::new(
            SplitPattern::Regex(pattern.into()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| Error::from_reason(e.to_string()))?;
        tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
            split.into(),
            ByteLevel::new(false, true, false).into(),
        ])));
        let config: serde_json::Value =
            super::config::read_json(&path.join("tokenizer_config.json"))?;
        if let Some(tokens) = config
            .get("added_tokens_decoder")
            .and_then(|x| x.as_object())
        {
            let mut entries = tokens
                .iter()
                .map(|(id, data)| {
                    Ok((
                        id.parse::<u32>()
                            .map_err(|e| Error::from_reason(e.to_string()))?,
                        data,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            entries.sort_by_key(|(id, _)| *id);
            for (id, data) in entries {
                let content = data
                    .get("content")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| Error::from_reason("Added token has no content"))?;
                let flag = |name| data.get(name).and_then(|x| x.as_bool()).unwrap_or(false);
                let token = AddedToken::from(content, flag("special"))
                    .single_word(flag("single_word"))
                    .lstrip(flag("lstrip"))
                    .rstrip(flag("rstrip"))
                    .normalized(flag("normalized"));
                tokenizer
                    .add_tokens([token])
                    .map_err(|e| Error::from_reason(e.to_string()))?;
                if tokenizer.token_to_id(content) != Some(id) {
                    return Err(Error::from_reason(format!(
                        "Tokenizer ID mismatch for {content}"
                    )));
                }
            }
        }
        Ok(Self(tokenizer))
    }
    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        self.0
            .encode(text, false)
            .map(|e| e.get_ids().iter().map(|&id| id as i32).collect())
            .map_err(|e| Error::from_reason(e.to_string()))
    }
}
