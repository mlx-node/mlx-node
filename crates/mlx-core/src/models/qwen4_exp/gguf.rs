use super::{config::Config, weights::Store};
use crate::utils::gguf::{
    GgufMetaValue, native_gguf_cache_root, source_file_identity_digest,
    write_embedded_gpt2_tokenizer,
};
use napi::{Error, Result};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub fn integers(m: &HashMap<String, GgufMetaValue>, key: &str) -> Result<Vec<u64>> {
    let v = match m.get(key) {
        Some(GgufMetaValue::ArrayU64(v)) => v.clone(),
        Some(GgufMetaValue::ArrayI64(v)) if v.iter().all(|&n| n >= 0) => {
            v.iter().map(|&n| n as u64).collect()
        }
        Some(GgufMetaValue::ArrayU32(v)) => v.iter().map(|&n| n as u64).collect(),
        Some(GgufMetaValue::ArrayI32(v)) if v.iter().all(|&n| n >= 0) => {
            v.iter().map(|&n| n as u64).collect()
        }
        _ => {
            return Err(Error::from_reason(format!(
                "Missing/invalid GGUF integer array {key}"
            )));
        }
    };
    Ok(v)
}
pub fn config(store: &Store) -> Result<Config> {
    let m = &store.metadata;
    let integer = |key: &str| {
        m.get(key)
            .and_then(GgufMetaValue::as_u64)
            .ok_or_else(|| Error::from_reason(format!("Missing GGUF metadata {key}")))
    };
    let mut text = serde_json::Map::new();
    for (hf, gg) in [
        ("hidden_size", "embedding_length"),
        ("num_hidden_layers", "block_count"),
        ("num_attention_heads", "attention.head_count"),
        ("num_key_value_heads", "attention.head_count_kv"),
        ("head_dim", "attention.key_length"),
        ("num_experts", "expert_count"),
        ("num_experts_per_tok", "expert_used_count"),
        ("moe_intermediate_size", "expert_feed_forward_length"),
        (
            "shared_expert_intermediate_size",
            "expert_shared_feed_forward_length",
        ),
        ("linear_num_key_heads", "ssm.group_count"),
        ("linear_num_value_heads", "ssm.time_step_rank"),
        ("linear_key_head_dim", "ssm.state_size"),
        ("linear_conv_kernel_dim", "ssm.conv_kernel"),
        ("hc_count", "hyper_connection.count"),
        ("hc_lowrank", "hyper_connection.low_rank"),
        ("max_position_embeddings", "context_length"),
        ("full_attention_interval", "full_attention_interval"),
        ("indexer_n_heads", "attention.indexer.head_count"),
        ("indexer_head_dim", "attention.indexer.key_length"),
        ("indexer_budget", "attention.indexer.top_k"),
        ("ple_conv_kernel_size", "ple.conv_kernel"),
        ("ngram_size", "ple.ngram_size"),
        ("heads_per_ngram", "ple.heads_per_ngram"),
        ("eos_token_id", "ple.eos_token_id"),
    ] {
        let key = format!("qwen4exp.{gg}");
        let n = integer(&key)?;
        text.insert(hf.into(), json!(n));
    }
    let ratios = integers(m, "qwen4exp.attention.compress_ratios")?;
    let ratio = ratios
        .iter()
        .find(|&&n| n > 0)
        .copied()
        .ok_or_else(|| Error::from_reason("Missing QSA compression ratio"))?;
    if ratios.len() as u64 != integer("qwen4exp.block_count")?
        || ratios.iter().any(|&r| r != 0 && r != ratio)
    {
        return Err(Error::from_reason(
            "Unsupported per-layer QSA compression ratios",
        ));
    }
    text.insert("indexer_compress_ratio".into(), json!(ratio));
    text.insert(
        "layer_types".into(),
        json!(
            ratios
                .iter()
                .map(|&r| if r == 0 {
                    "linear_attention"
                } else {
                    "full_attention"
                })
                .collect::<Vec<_>>()
        ),
    );
    let vheads = integer("qwen4exp.ssm.time_step_rank")?;
    let inner = m
        .get("qwen4exp.ssm.inner_size")
        .and_then(GgufMetaValue::as_u64)
        .ok_or_else(|| Error::from_reason("Missing SSM inner_size"))?;
    if vheads == 0 || !inner.is_multiple_of(vheads) {
        return Err(Error::from_reason("Invalid SSM inner_size"));
    }
    text.insert("linear_value_head_dim".into(), json!(inner / vheads));
    let eps = m
        .get("qwen4exp.attention.layer_norm_rms_epsilon")
        .and_then(GgufMetaValue::as_f32)
        .ok_or_else(|| Error::from_reason("Missing RMS epsilon"))?;
    text.insert("rms_norm_eps".into(), json!(eps));
    let dims = m
        .get("qwen4exp.rope.dimension_count")
        .and_then(GgufMetaValue::as_u64)
        .ok_or_else(|| Error::from_reason("Missing RoPE dimensions"))?;
    let theta = m
        .get("qwen4exp.rope.freq_base")
        .and_then(GgufMetaValue::as_f32)
        .ok_or_else(|| Error::from_reason("Missing RoPE base"))?;
    let head_dim = integer("qwen4exp.attention.key_length")?;
    if head_dim == 0 {
        return Err(Error::from_reason("Invalid GGUF attention head dimension"));
    }
    text.insert(
        "rope_parameters".into(),
        json!({"rope_theta":theta,"partial_rotary_factor":dims as f64/head_dim as f64}),
    );
    text.insert(
        "ple_layer_ids".into(),
        json!(
            integers(m, "qwen4exp.ple.layers")?
                .iter()
                .map(|n| n
                    .checked_add(1)
                    .ok_or_else(|| Error::from_reason("PLE layer index overflow")))
                .collect::<Result<Vec<_>>>()?
        ),
    );
    let sizes = integers(m, "qwen4exp.ple.head_vocab_sizes")?;
    text.insert(
        "ngram_vocab_size_base".into(),
        json!(sizes.first().copied().unwrap_or(0)),
    );
    text.insert("split_ngram_parts".into(), json!(1));
    text.insert(
        "ple_embed_dim".into(),
        json!(store.descriptor("per_layer_token_embd.weight")?.width()? * sizes.len()),
    );
    text.insert(
        "vocab_size".into(),
        json!(store.descriptor("token_embd.weight")?.rows()?),
    );
    Config::parse(&json!({"model_type":"qwen4_exp","text_config":Value::Object(text)}))
}

/// Tokenizer-only assets in the application cache; the checkpoint stays read-only.
pub fn assets(first: &Path, store: &Store, c: &Config) -> Result<PathBuf> {
    assets_in(first, store, c, &native_gguf_cache_root()?)
}

fn assets_in(first: &Path, store: &Store, c: &Config, root: &Path) -> Result<PathBuf> {
    let source = first
        .canonicalize()
        .map_err(|e| Error::from_reason(e.to_string()))?;
    let metadata = std::fs::metadata(&source).map_err(|e| Error::from_reason(e.to_string()))?;
    let identity = source_file_identity_digest(&source, &metadata);
    let dir = root.join(format!(".mlx-qwen4-assets-v2-{identity}"));
    if dir.join("complete").exists() {
        return Ok(dir);
    }
    // Publish a complete directory in one rename. Simultaneous loads may race,
    // but neither reader can observe another load's partly written tokenizer.
    let temp = root.join(format!(".mlx-qwen4-assets-tmp-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&temp).map_err(|e| Error::from_reason(e.to_string()))?;
    let result = (|| -> Result<PathBuf> {
        if !write_embedded_gpt2_tokenizer(&store.metadata, &temp)? {
            return Err(Error::from_reason(
                "GGUF lacks a supported embedded tokenizer",
            ));
        }
        std::fs::write(
            temp.join("config.json"),
            serde_json::to_vec_pretty(&json!({"model_type":"qwen4_exp","text_config":c}))
                .map_err(|e| Error::from_reason(e.to_string()))?,
        )
        .map_err(|e| Error::from_reason(e.to_string()))?;
        std::fs::write(temp.join("complete"), b"1")
            .map_err(|e| Error::from_reason(e.to_string()))?;
        if let Err(e) = std::fs::rename(&temp, &dir)
            && !dir.join("complete").exists()
        {
            return Err(Error::from_reason(e.to_string()));
        }
        Ok(dir)
    })();
    let _ = std::fs::remove_dir_all(&temp);
    result
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::fs::{self, FileTimes, OpenOptions};
    use std::os::unix::fs::PermissionsExt;

    fn tokenizer_fixture() -> (Store, Config) {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp");
        let mut store = Store::open_metadata(&fixture, None).unwrap();
        let raw = serde_json::from_slice(&fs::read(fixture.join("config.json")).unwrap()).unwrap();
        let config = Config::parse(&raw).unwrap();
        store.metadata = HashMap::from([
            (
                "tokenizer.ggml.model".into(),
                GgufMetaValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetaValue::ArrayString(vec!["old".into()]),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                GgufMetaValue::ArrayI32(vec![1]),
            ),
            (
                "tokenizer.ggml.merges".into(),
                GgufMetaValue::ArrayString(Vec::new()),
            ),
        ]);
        (store, config)
    }

    #[test]
    fn asset_cache_rebuilds_after_same_size_same_mtime_source_replacement() {
        let root = std::env::temp_dir().join(format!("qwen4-assets-{}", uuid::Uuid::new_v4()));
        let cache = root.join("cache");
        fs::create_dir_all(&cache).unwrap();
        let source = root.join("model.gguf");
        let replacement = root.join("replacement.gguf");
        fs::write(&source, b"old").unwrap();
        fs::write(&replacement, b"new").unwrap();
        let original_metadata = fs::metadata(&source).unwrap();
        let original_modified = original_metadata.modified().unwrap();
        let (mut store, config) = tokenizer_fixture();
        let original = assets_in(&source, &store, &config, &cache).unwrap();
        assert_eq!(
            assets_in(&source, &store, &config, &cache).unwrap(),
            original
        );

        OpenOptions::new()
            .write(true)
            .open(&replacement)
            .unwrap()
            .set_times(FileTimes::new().set_modified(original_modified))
            .unwrap();
        fs::rename(&replacement, &source).unwrap();
        let replacement_metadata = fs::metadata(&source).unwrap();
        assert_eq!(replacement_metadata.len(), original_metadata.len());
        assert_eq!(replacement_metadata.modified().unwrap(), original_modified);
        store.metadata.insert(
            "tokenizer.ggml.tokens".into(),
            GgufMetaValue::ArrayString(vec!["new".into()]),
        );

        let rebuilt = assets_in(&source, &store, &config, &cache).unwrap();
        assert_ne!(original, rebuilt);
        let tokenizer: Value =
            serde_json::from_slice(&fs::read(rebuilt.join("tokenizer.json")).unwrap()).unwrap();
        assert_eq!(tokenizer["model"]["vocab"]["new"], 0);
        assert!(tokenizer["model"]["vocab"].get("old").is_none());
        assert!(rebuilt.join("complete").is_file());
        assert_eq!(store.bytes_read, 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn asset_cache_never_writes_beside_read_only_source() {
        let root = std::env::temp_dir().join(format!("qwen4-assets-{}", uuid::Uuid::new_v4()));
        let source = root.join("source");
        let cache = root.join("cache");
        fs::create_dir_all(&source).unwrap();
        fs::create_dir(&cache).unwrap();
        let input = source.join("model.gguf");
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp/model.gguf");
        fs::copy(&fixture, &input).unwrap();
        let (store, config) = tokenizer_fixture();

        fs::set_permissions(&input, fs::Permissions::from_mode(0o444)).unwrap();
        fs::set_permissions(&source, fs::Permissions::from_mode(0o555)).unwrap();
        let prepared = assets_in(&input, &store, &config, &cache);
        fs::set_permissions(&source, fs::Permissions::from_mode(0o755)).unwrap();
        let output = prepared.unwrap();

        assert!(output.starts_with(&cache));
        assert_eq!(assets_in(&input, &store, &config, &cache).unwrap(), output);
        assert_eq!(fs::read_dir(&source).unwrap().count(), 1);
        assert_eq!(fs::read(&input).unwrap(), fs::read(&fixture).unwrap());
        let tokenizer: Value =
            serde_json::from_slice(&fs::read(output.join("tokenizer.json")).unwrap()).unwrap();
        assert_eq!(tokenizer["model"]["vocab"]["old"], 0);
        let generated: Value =
            serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
        assert_eq!(
            generated["text_config"],
            serde_json::to_value(&config).unwrap()
        );
        assert!(output.join("complete").is_file());
        assert_eq!(fs::read_dir(&cache).unwrap().count(), 1);
        assert_eq!(store.bytes_read, 0);
        fs::remove_dir_all(root).unwrap();
    }
}
