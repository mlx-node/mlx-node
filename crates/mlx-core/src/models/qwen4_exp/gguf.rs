use super::{config::Config, weights::Store};
use crate::utils::gguf::{
    GgufMetaValue, acquire_leased_gguf_assets_sidecar, native_gguf_cache_root,
    write_embedded_gpt2_tokenizer,
};
#[cfg(all(test, unix))]
use crate::utils::gguf::{
    NATIVE_GGUF_ASSETS_PREFIX, native_gguf_assets_dir_name, source_file_identity_digest,
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
    let mut rope = json!({"rope_theta":theta,"partial_rotary_factor":dims as f64/head_dim as f64});
    // Splash `GgufImage`: a set `rope.scaling.type` other than `none` is an
    // already scaled target, which `Config::validate` refuses.
    if let Some(scaling) = m
        .get("qwen4exp.rope.scaling.type")
        .and_then(GgufMetaValue::as_str)
        .filter(|scaling| *scaling != "none")
    {
        rope["rope_type"] = json!(scaling);
    }
    text.insert("rope_parameters".into(), rope);
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
    acquire_leased_gguf_assets_sidecar(first, root, |temp| {
        if !write_embedded_gpt2_tokenizer(&store.metadata, temp)? {
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
        Ok(())
    })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::utils::native_gguf_unlease_in_use;
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
        use crate::utils::gguf::sweep_env_lock;
        // Serialize against native-GGUF sweep tests: `assets_in` holding the
        // sidecar mutex would make their `try_lock` skip the sweep.
        let _env_guard = sweep_env_lock().lock().unwrap();
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
        use crate::utils::gguf::sweep_env_lock;
        let _env_guard = sweep_env_lock().lock().unwrap();
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
            "cache hits must refresh the complete marker recency"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn asset_cache_never_writes_beside_read_only_source() {
        use crate::utils::gguf::sweep_env_lock;
        let _env_guard = sweep_env_lock().lock().unwrap();
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
        // The sidecar dir plus its lock and lease sentinels — nothing else
        // may be written beside the read-only source or in the cache root.
        let names: Vec<String> = fs::read_dir(&cache)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 3, "{names:?}");
        assert_eq!(store.bytes_read, 0);
        fs::remove_dir_all(root).unwrap();
    }

    /// A busy sidecar lock is not an ordinary cache miss: it can be a
    /// sweep mid-`remove_dir_all` on the very directory a publish would land
    /// in. `assets_in` must spin on the lock and fail loudly rather than
    /// publishing unguarded into a possible mid-delete (which could return a
    /// phantom path the sweep then deletes).
    #[test]
    fn assets_in_never_publishes_while_the_sidecar_lock_is_held() {
        use crate::utils::gguf::sweep_env_lock;
        use std::os::fd::AsRawFd;
        let _env_guard = sweep_env_lock().lock().unwrap();
        let root = std::env::temp_dir().join(format!("qwen4-assets-{}", uuid::Uuid::new_v4()));
        let cache = root.join("cache");
        fs::create_dir_all(&cache).unwrap();
        let source = root.join("model.gguf");
        fs::write(&source, b"weights").unwrap();
        let (store, config) = tokenizer_fixture();

        // Hold the same lock file the sweep would hold while deleting the
        // sidecar: `.mlx-qwen4-assets-v3-{identity}.lock`. flock is
        // per-open-description, so this fd blocks assets_sidecar_lock.
        let canonical = source.canonicalize().unwrap();
        let metadata = fs::metadata(&canonical).unwrap();
        let dir_name =
            native_gguf_assets_dir_name(&source_file_identity_digest(&canonical, &metadata));
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

        let err = crate::utils::gguf::with_test_sidecar_retry_budget(
            3,
            std::time::Duration::from_millis(5),
            || assets_in(&source, &store, &config, &cache),
        )
        .unwrap_err();
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

    /// Marker loss on leased outputs promotes fresh sidecars to the base
    /// live name and then the next generation, never deleting either lease.
    #[test]
    fn qwen4_sidecar_promotes_live_when_leased() {
        use crate::utils::gguf::sweep_env_lock;
        let _env_guard = sweep_env_lock().lock().unwrap();
        let root = std::env::temp_dir().join(format!("qwen4-assets-{}", uuid::Uuid::new_v4()));
        let cache = root.join("cache");
        fs::create_dir_all(&cache).unwrap();
        let source = root.join("model.gguf");
        fs::write(&source, b"weights").unwrap();
        let (store, config) = tokenizer_fixture();

        let published = assets_in(&source, &store, &config, &cache).unwrap();
        assert!(
            published
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(NATIVE_GGUF_ASSETS_PREFIX)
        );
        fs::remove_file(published.join("complete")).unwrap();

        let live = assets_in(&source, &store, &config, &cache).unwrap();
        let live_name = live.file_name().unwrap().to_string_lossy();
        assert!(
            live_name.starts_with(".mlx-qwen4-assets-live-")
                && live_name.ends_with(&std::process::id().to_string()),
            "a leased versioned output must promote to the base live dir, got {}",
            live.display()
        );
        assert!(live.join("complete").is_file());
        assert!(
            published.exists(),
            "the markerless but leased versioned dir was deleted"
        );

        fs::remove_file(live.join("complete")).unwrap();
        let rebuilt = assets_in(&source, &store, &config, &cache).unwrap();
        assert!(
            rebuilt
                .file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(&format!("-{}-g2", std::process::id())),
            "a leased base live dir must promote to generation 2, got {}",
            rebuilt.display()
        );
        assert!(rebuilt.join("complete").is_file());
        assert!(live.exists(), "the markerless leased live dir was deleted");
        assert_eq!(
            assets_in(&source, &store, &config, &cache).unwrap(),
            rebuilt,
            "a completed live generation must be reused"
        );
        fs::remove_dir_all(root).unwrap();
    }

    /// A published sidecar must never be returned without its keyed lease:
    /// an unopenable `.{dir_name}.lease` sentinel (a directory there blocks
    /// the open) turns every hit and publish return into an error — the
    /// model must not hold a directory a foreign sweep can evict.
    #[test]
    fn assets_in_never_returns_a_sidecar_its_lease_could_not_pin() {
        use crate::utils::gguf::sweep_env_lock;
        let _env_guard = sweep_env_lock().lock().unwrap();
        let root = std::env::temp_dir().join(format!("qwen4-assets-{}", uuid::Uuid::new_v4()));
        let cache = root.join("cache");
        fs::create_dir_all(&cache).unwrap();
        let source = root.join("model.gguf");
        fs::write(&source, b"weights").unwrap();
        let (store, config) = tokenizer_fixture();

        let canonical = source.canonicalize().unwrap();
        let metadata = fs::metadata(&canonical).unwrap();
        let dir_name =
            native_gguf_assets_dir_name(&source_file_identity_digest(&canonical, &metadata));
        // A directory at the sentinel path: opening it for write fails on
        // every platform, so the keyed lease fails closed.
        fs::create_dir_all(cache.join(format!("{dir_name}.lease"))).unwrap();

        let err = crate::utils::gguf::with_test_sidecar_retry_budget(
            3,
            std::time::Duration::from_millis(5),
            || assets_in(&source, &store, &config, &cache),
        )
        .unwrap_err();
        assert!(
            err.reason.contains("lease"),
            "the load must fail rather than return an unleasable sidecar, got: {}",
            err.reason
        );
        fs::remove_dir_all(root).unwrap();
    }

    /// Concurrent same-process loads serialize on `native_gguf_sidecar_mutex`:
    /// the sidecar flock alone does not (a BSD flock is per-process — a
    /// second same-process fd "acquires" LOCK_EX while the first holds it).
    /// Without serialization both callers could miss the `complete` checks,
    /// build racing temp dirs, and the loser's publish could remove the dir
    /// the winner already returned.
    #[test]
    fn assets_in_serializes_concurrent_same_process_loads() {
        use crate::utils::gguf::sweep_env_lock;
        let _env_guard = sweep_env_lock().lock().unwrap();
        let root = std::env::temp_dir().join(format!("qwen4-assets-{}", uuid::Uuid::new_v4()));
        let cache = root.join("cache");
        fs::create_dir_all(&cache).unwrap();
        let source = root.join("model.gguf");
        fs::write(&source, b"weights").unwrap();

        let barrier = std::sync::Barrier::new(2);
        let results: Vec<Result<PathBuf>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2)
                .map(|_| {
                    let barrier = &barrier;
                    let source = &source;
                    let cache = &cache;
                    scope.spawn(move || {
                        barrier.wait();
                        let (store, config) = tokenizer_fixture();
                        assets_in(source, &store, &config, cache)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let first = results[0].as_ref().unwrap();
        let second = results[1].as_ref().unwrap();
        assert_eq!(
            first, second,
            "both concurrent loads must land on the same directory"
        );
        assert!(
            first.join("complete").is_file(),
            "the returned path must still be a completed sidecar"
        );
        assert_eq!(
            fs::read(first.join("tokenizer.json")).unwrap(),
            fs::read(second.join("tokenizer.json")).unwrap(),
            "both returned paths must serve identical content"
        );
        let tokenizer: Value =
            serde_json::from_slice(&fs::read(first.join("tokenizer.json")).unwrap()).unwrap();
        assert_eq!(tokenizer["model"]["vocab"]["old"], 0);
        fs::remove_dir_all(root).unwrap();
    }

    /// Holding the sidecar mutex makes the sweep skip even same-process
    /// dead-pid cleanup; releasing it allows the normal debris pass.
    #[test]
    fn assets_in_survives_concurrent_in_process_sweep() {
        use crate::utils::gguf::{
            native_gguf_sidecar_mutex, native_gguf_unlease_in_use, sweep_env_lock,
            sweep_native_gguf_cache,
        };
        let _env_guard = sweep_env_lock().lock().unwrap();
        let root = std::env::temp_dir().join(format!("qwen4-assets-{}", uuid::Uuid::new_v4()));
        let cache = root.join("cache");
        fs::create_dir_all(&cache).unwrap();
        let source = root.join("model.gguf");
        fs::write(&source, b"weights").unwrap();
        let (store, config) = tokenizer_fixture();
        let dir = assets_in(&source, &store, &config, &cache).unwrap();
        assert!(dir.join("complete").is_file());

        let debris = cache.join(format!(
            ".mlx-qwen4-assets-live-{}-{}",
            "d".repeat(64),
            i32::MAX
        ));
        fs::create_dir_all(&debris).unwrap();
        fs::write(debris.join("complete"), b"1").unwrap();

        let sidecar_guard = native_gguf_sidecar_mutex()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        sweep_native_gguf_cache(&cache, "no-such-key");
        assert!(
            debris.exists(),
            "the sweep must skip while a sidecar publish could be in flight"
        );
        drop(sidecar_guard);

        assert!(
            crate::test_support::sweep_until_removed(&cache, "no-such-key", &[&debris]),
            "dead-pid debris must be reclaimed"
        );
        assert!(dir.join("complete").is_file());
        native_gguf_unlease_in_use(&dir);
        fs::remove_dir_all(root).unwrap();
    }

    /// A completed sidecar of the previous version (special USER_DEFINED
    /// rows), published or as this pid's live fallback, is never served:
    /// the load rebuilds and the tags come back not special.
    #[test]
    fn asset_cache_never_reuses_an_older_sidecar_version() {
        use crate::utils::gguf::sweep_env_lock;
        let _env_guard = sweep_env_lock().lock().unwrap();
        let root = std::env::temp_dir().join(format!("qwen4-assets-{}", uuid::Uuid::new_v4()));
        let cache = root.join("cache");
        fs::create_dir_all(&cache).unwrap();
        let source = root.join("model.gguf");
        fs::write(&source, b"weights").unwrap();
        let (mut store, config) = tokenizer_fixture();
        store.metadata = crate::utils::gguf_test_fixtures::qwen_tag_tokenizer_metadata(&[]);
        let canonical = source.canonicalize().unwrap();
        let identity = source_file_identity_digest(&canonical, &fs::metadata(&canonical).unwrap());
        let stale = |dir: &Path| {
            fs::create_dir_all(dir).unwrap();
            fs::write(dir.join("tokenizer.json"), b"{\"stale\":true}").unwrap();
            fs::write(dir.join("complete"), b"1").unwrap();
        };
        let published = cache.join(format!(".mlx-qwen4-assets-v2-{identity}"));
        let live = cache.join(format!(
            ".mlx-qwen4-assets-live-{identity}-{}",
            std::process::id()
        ));
        stale(&published);
        stale(&live);

        let dir = assets_in(&source, &store, &config, &cache).unwrap();
        assert_ne!(dir, published);
        assert_ne!(dir, live);
        let tokenizer: Value =
            serde_json::from_slice(&fs::read(dir.join("tokenizer.json")).unwrap()).unwrap();
        let special = |content: &str| {
            tokenizer["added_tokens"]
                .as_array()
                .unwrap()
                .iter()
                .find(|token| token["content"] == content)
                .map(|token| token["special"].as_bool().unwrap())
        };
        assert_eq!(special("<tool_call>"), Some(false));
        assert_eq!(special("</think>"), Some(false));
        assert_eq!(special("<|im_end|>"), Some(true));
        native_gguf_unlease_in_use(&dir);
        fs::remove_dir_all(root).unwrap();
    }
}
