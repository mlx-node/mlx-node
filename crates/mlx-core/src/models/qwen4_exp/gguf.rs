use super::{config::Config, weights::Store};
use crate::utils::gguf::{
    GgufMetaValue, native_gguf_cache_cap_bytes, native_gguf_cache_root,
    native_gguf_publish_with_cap, source_file_identity_digest, write_embedded_gpt2_tokenizer,
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

/// Per-sidecar flock shared with the native-GGUF sweep: a held lock makes the
/// sweep skip the directory, so the hit check and the publish below cannot
/// race a mid-flight eviction. `Some(_)` keeps the fd as a guard (inner None
/// on non-Unix, where no sweep runs); `None` means contention — the caller
/// retries briefly rather than publishing into a possible mid-delete.
#[cfg(unix)]
fn assets_sidecar_lock(root: &Path, dir_name: &str) -> Option<Option<std::fs::File>> {
    use std::os::fd::AsRawFd;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join(format!("{dir_name}.lock")))
        .ok()?;
    // SAFETY: `file` owns the descriptor for the flock call, and the returned
    // guard keeps it — and the lock — alive until the caller drops it.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return None;
    }
    Some(Some(file))
}

#[cfg(not(unix))]
fn assets_sidecar_lock(_root: &Path, _dir_name: &str) -> Option<Option<std::fs::File>> {
    Some(None)
}

fn assets_in(first: &Path, store: &Store, c: &Config, root: &Path) -> Result<PathBuf> {
    let source = first
        .canonicalize()
        .map_err(|e| Error::from_reason(e.to_string()))?;
    let metadata = std::fs::metadata(&source).map_err(|e| Error::from_reason(e.to_string()))?;
    let identity = source_file_identity_digest(&source, &metadata);
    let dir_name = format!(".mlx-qwen4-assets-v2-{identity}");
    let dir = root.join(&dir_name);
    // Contention here is not an ordinary cache miss: a busy sidecar lock can
    // be a capped-cache sweep mid-`remove_dir_all` on this very directory.
    // Publishing unguarded could land `rename(temp, dir)` while the sweep's
    // delete is in flight, returning a path that no longer exists. Spin
    // briefly — a sweep's delete is short — and publish only under the lock.
    const ASSETS_LOCK_RETRIES: u32 = 40;
    const ASSETS_LOCK_RETRY_SLEEP: std::time::Duration = std::time::Duration::from_millis(50);
    let mut guard = assets_sidecar_lock(root, &dir_name);
    for _ in 0..ASSETS_LOCK_RETRIES {
        if guard.is_some() {
            break;
        }
        std::thread::sleep(ASSETS_LOCK_RETRY_SLEEP);
        guard = assets_sidecar_lock(root, &dir_name);
    }
    let Some(_guard) = guard else {
        return Err(Error::from_reason(format!(
            "GGUF assets sidecar lock for {dir_name} stayed contended for ~2s; refusing to publish unguarded"
        )));
    };
    if dir.join("complete").exists() {
        // Stamp recency on hits so the native-GGUF sweep's LRU never eats a
        // live sidecar directory; the held lock blocks that same sweep.
        if let Ok(file) = std::fs::OpenOptions::new()
            .write(true)
            .open(dir.join("complete"))
        {
            let _ = file.set_modified(std::time::SystemTime::now());
        }
        return Ok(dir);
    }
    // Publish a complete directory in one rename while the sidecar lock is
    // held: the sweep cannot delete `dir` between the rename and the return.
    // Simultaneous loads serialize on the lock rather than racing temp dirs.
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
        // With a cap configured the rename runs inside the shared
        // count→evict→publish section: admission counts the root under the
        // root-wide flock, so a sidecar cannot land between a sibling
        // conversion's admission count and its entry rename. Refusal
        // returns `Ok(temp)`: the temp already contains `complete`, so the
        // load reads it in place — and the sweep's `.mlx-qwen4-assets-tmp-`
        // grammar only ages it out after 24 h. The cap lock nests inside
        // the sidecar lock, matching `prepare_native_gguf_inner`'s
        // key-lock → admission-flock order.
        if let Some(cap) = native_gguf_cache_cap_bytes() {
            return native_gguf_publish_with_cap(root, &temp, &dir, cap)
                .map_err(|e| Error::from_reason(e.to_string()));
        }
        if let Err(e) = std::fs::rename(&temp, &dir)
            && !dir.join("complete").exists()
        {
            return Err(Error::from_reason(e.to_string()));
        }
        Ok(dir)
    })();
    // Keep the temp only when a refused admission returned it as the load
    // path; every other outcome either renamed it away or failed.
    if result.as_ref().is_ok_and(|path| *path != temp) || result.is_err() {
        let _ = std::fs::remove_dir_all(&temp);
    }
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
    fn asset_cache_hit_stamps_complete_marker_recency() {
        let root = std::env::temp_dir().join(format!("qwen4-assets-{}", uuid::Uuid::new_v4()));
        let cache = root.join("cache");
        fs::create_dir_all(&cache).unwrap();
        let source = root.join("model.gguf");
        fs::write(&source, b"weights").unwrap();
        let (store, config) = tokenizer_fixture();
        let dir = assets_in(&source, &store, &config, &cache).unwrap();
        let marker = dir.join("complete");
        let stale = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 24 * 3600);
        OpenOptions::new()
            .write(true)
            .open(&marker)
            .unwrap()
            .set_modified(stale)
            .unwrap();
        assert_eq!(assets_in(&source, &store, &config, &cache).unwrap(), dir);
        assert!(
            fs::metadata(&marker).unwrap().modified().unwrap() > stale,
            "cache hits must refresh the complete marker for LRU recency"
        );
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
        // The sidecar dir plus its hit-path lock file — nothing else may be
        // written beside the read-only source or in the cache root.
        let names: Vec<String> = fs::read_dir(&cache)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert_eq!(store.bytes_read, 0);
        fs::remove_dir_all(root).unwrap();
    }

    /// A busy sidecar lock is not an ordinary cache miss: it can be a capped
    /// sweep mid-`remove_dir_all` on the very directory a publish would land
    /// in. `assets_in` must spin on the lock and fail loudly rather than
    /// publishing unguarded into a possible mid-delete (which could return a
    /// phantom path the sweep then deletes).
    #[test]
    fn assets_in_never_publishes_while_the_sidecar_lock_is_held() {
        use std::os::fd::AsRawFd;
        let root = std::env::temp_dir().join(format!("qwen4-assets-{}", uuid::Uuid::new_v4()));
        let cache = root.join("cache");
        fs::create_dir_all(&cache).unwrap();
        let source = root.join("model.gguf");
        fs::write(&source, b"weights").unwrap();
        let (store, config) = tokenizer_fixture();

        // Hold the same lock file the sweep would hold while deleting the
        // sidecar: `.mlx-qwen4-assets-v2-{identity}.lock`. flock is
        // per-open-description, so this fd blocks assets_sidecar_lock.
        let canonical = source.canonicalize().unwrap();
        let metadata = fs::metadata(&canonical).unwrap();
        let dir_name = format!(
            ".mlx-qwen4-assets-v2-{}",
            source_file_identity_digest(&canonical, &metadata)
        );
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(cache.join(format!("{dir_name}.lock")))
            .unwrap();
        assert_eq!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );

        let err = assets_in(&source, &store, &config, &cache).unwrap_err();
        assert!(
            err.reason.contains("sidecar lock"),
            "expected a contention error, got: {}",
            err.reason
        );
        // Nothing may have been published or left behind while unguarded.
        assert!(!cache.join(&dir_name).exists());
        assert_eq!(
            fs::read_dir(&cache).unwrap().count(),
            1,
            "only the lock file itself may exist"
        );

        // Once the lock frees, publishing proceeds normally.
        drop(lock);
        let published = assets_in(&source, &store, &config, &cache).unwrap();
        assert_eq!(published, cache.join(&dir_name));
        assert!(published.join("complete").is_file());
        fs::remove_dir_all(root).unwrap();
    }

    /// With `MLX_NATIVE_GGUF_CACHE_MAX_GB` set the sidecar publish runs the
    /// shared cap-admission section instead of a bare rename: a cap that
    /// cannot fit the sidecar refuses admission and returns the completed
    /// temp dir (swept only by the 24 h tmp grammar), while a roomy cap
    /// still publishes under the versioned name.
    #[test]
    fn assets_in_publish_respects_the_cache_cap() {
        use crate::utils::gguf::{SweepEnvVar, sweep_env_lock};
        let _env_guard = sweep_env_lock().lock().unwrap();
        let root = std::env::temp_dir().join(format!("qwen4-assets-{}", uuid::Uuid::new_v4()));
        let cache = root.join("cache");
        fs::create_dir_all(&cache).unwrap();
        let source = root.join("model.gguf");
        fs::write(&source, b"weights").unwrap();
        let (store, config) = tokenizer_fixture();

        // Tiny cap: admission refuses, the load path is the temp dir, and
        // nothing is published under the versioned sidecar name.
        let _env = SweepEnvVar::set_max_gb("0.0000001");
        let refused = assets_in(&source, &store, &config, &cache).unwrap();
        let refused_name = refused.file_name().unwrap().to_string_lossy();
        assert!(
            refused_name.starts_with(".mlx-qwen4-assets-tmp-"),
            "a refused admission returns the temp dir, got {}",
            refused.display()
        );
        assert!(refused.join("complete").is_file());
        assert!(
            fs::read_dir(&cache).unwrap().flatten().all(|entry| !entry
                .file_name()
                .to_string_lossy()
                .starts_with(".mlx-qwen4-assets-v2-")
                || entry.file_name().to_string_lossy().ends_with(".lock")),
            "no sidecar dir may be published over the cap"
        );

        // Roomy cap: the normal publish path still lands.
        drop(_env);
        let _env = SweepEnvVar::set_max_gb("64");
        let published = assets_in(&source, &store, &config, &cache).unwrap();
        assert_eq!(published.parent(), Some(cache.as_path()));
        assert!(
            published
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(".mlx-qwen4-assets-v2-")
        );
        assert!(published.join("complete").is_file());
        fs::remove_dir_all(root).unwrap();
    }
}
