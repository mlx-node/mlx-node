use super::{config::Config, weights::Store};
use crate::utils::gguf::{GgufMetaValue, write_embedded_gpt2_tokenizer};
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
        let n = m
            .get(&key)
            .and_then(GgufMetaValue::as_u64)
            .ok_or_else(|| Error::from_reason(format!("Missing GGUF metadata {key}")))?;
        text.insert(hf.into(), json!(n));
    }
    let ratios = integers(m, "qwen4exp.attention.compress_ratios")?;
    let ratio = ratios
        .iter()
        .find(|&&n| n > 0)
        .copied()
        .ok_or_else(|| Error::from_reason("Missing QSA compression ratio"))?;
    if ratios.len() != text["num_hidden_layers"].as_u64().unwrap() as usize
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
    let vheads = text["linear_num_value_heads"].as_u64().unwrap();
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
    text.insert("rope_parameters".into(),json!({"rope_theta":theta,"partial_rotary_factor":dims as f64/text["head_dim"].as_u64().unwrap() as f64}));
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
        json!(store.descriptor("per_layer_token_embd.weight")?.width() * sizes.len()),
    );
    text.insert(
        "vocab_size".into(),
        json!(store.descriptor("token_embd.weight")?.rows()),
    );
    Config::parse(&json!({"model_type":"qwen4_exp","text_config":Value::Object(text)}))
}

/// Tokenizer-only assets: never convert or duplicate the 111 GB weight payload.
pub fn assets(first: &Path, store: &Store, c: &Config) -> Result<PathBuf> {
    use std::hash::{Hash, Hasher};
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    first
        .canonicalize()
        .map_err(|e| Error::from_reason(e.to_string()))?
        .hash(&mut hash);
    std::fs::metadata(first)
        .and_then(|m| m.modified())
        .map_err(|e| Error::from_reason(e.to_string()))?
        .hash(&mut hash);
    let dir = first
        .parent()
        .unwrap()
        .join(format!(".mlx-qwen4-assets-v2-{:016x}", hash.finish()));
    if dir.join("complete").exists() {
        return Ok(dir);
    }
    // Publish a complete directory in one rename. Simultaneous loads may race,
    // but neither reader can observe another load's partly written tokenizer.
    let temp = first
        .parent()
        .unwrap()
        .join(format!(".mlx-qwen4-assets-tmp-{}", uuid::Uuid::new_v4()));
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
