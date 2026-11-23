//! Shared GGUF shard-discovery and embedded-tokenizer helpers for streamed
//! GGUF chat families (`deepseek_v41`, `hy_v4_flash`).

use std::path::{Path, PathBuf};

use napi::{Error, Result};

use crate::tokenizer::Qwen3Tokenizer;
use crate::utils::gguf::{GgufMetaValue, embedded_gpt2_tokenizer_json, parse_gguf_reader};

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("gguf_chat: {msg}"))
}

/// Every shard of the model at `path`: any shard of a `-NNNNN-of-MMMMM.gguf`
/// set (siblings found by name), a directory holding one such set or one
/// `.gguf`, or a single-file GGUF.
pub(crate) fn shard_paths(path: &Path) -> Result<Vec<PathBuf>> {
    let path = if path.is_dir() {
        let mut v: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|e| bad(format!("'{}': {e}", path.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
            .collect();
        v.sort();
        let first = v
            .iter()
            .find(|p| p.to_string_lossy().contains("-00001-of-"));
        match (first, v.len()) {
            (Some(f), _) => f.clone(),
            (None, 1) => v.swap_remove(0),
            _ => return Err(bad(format!("'{}': no single GGUF set", path.display()))),
        }
    } else {
        path.to_path_buf()
    };
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| bad("non-UTF8 model path"))?;
    let split = name
        .strip_suffix(".gguf")
        .and_then(|s| s.rsplit_once("-of-"))
        .and_then(|(head, total)| Some((head.rsplit_once('-')?, total)))
        .filter(|((_, idx), total)| {
            [idx, total]
                .iter()
                .all(|s| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit()))
        });
    let Some(((prefix, _), total)) = split else {
        return Ok(vec![path]);
    };
    let n: usize = total
        .parse()
        .map_err(|_| bad(format!("shard count in '{name}'")))?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let paths: Vec<PathBuf> = (1..=n)
        .map(|i| dir.join(format!("{prefix}-{i:05}-of-{total}.gguf")))
        .collect();
    if let Some(missing) = paths.iter().find(|p| !p.is_file()) {
        return Err(bad(format!("shard '{}' missing", missing.display())));
    }
    Ok(paths)
}

/// The GGUF-embedded tokenizer and chat template of shard 1 (header only,
/// F_NOCACHE), built in memory.
pub(crate) fn gguf_tokenizer(first: &Path) -> Result<(Qwen3Tokenizer, u32)> {
    let file = crate::moe_stream::io::open_nocache(first)?;
    let g = parse_gguf_reader(std::io::BufReader::new(file))?;
    let json = embedded_gpt2_tokenizer_json(&g.metadata)?
        .ok_or_else(|| bad("GGUF lacks a supported embedded tokenizer"))?;
    let bytes = serde_json::to_vec(&json).map_err(|e| bad(e.to_string()))?;
    let tok = tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| bad(e.to_string()))?;
    let template = match g.metadata.get("tokenizer.chat_template") {
        Some(GgufMetaValue::String(t)) => Some(t.clone()),
        _ => None,
    };
    let eos = g
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(GgufMetaValue::as_u32)
        .ok_or_else(|| bad("GGUF lacks tokenizer.ggml.eos_token_id"))?;
    Ok((Qwen3Tokenizer::from_tokenizer(tok, template)?, eos))
}
