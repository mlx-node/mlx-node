//! Real-weights cold-tier restart-parity gate for Qwen3.
//!
//! Validates the full A1–A4 cold-tier pipeline end to end at the model
//! level: a persist-enabled instance captures its full paged KV blocks to
//! the SSD cold tier on turn finalize (A3), and a SECOND fresh instance —
//! standing in for a process restart, with an empty in-memory hot cache —
//! restores that persisted prefix through `find_cached_prefix` (A4) and
//! reports it in `cached_tokens`. The restored greedy decode must be
//! byte-identical to the original, and to a THIRD instance that runs the
//! same prompt with persistence disabled.
//!
//! Instances 1 and 2 load from the SAME on-disk clone (persist on) so their
//! cold-tier fingerprints — parsed config bytes + a bounded per-shard
//! weight-content sample + pool geometry/dtype (see
//! `cold_tier::build_model_fingerprint`) — are byte-identical (the clone's
//! weight files are symlinks the sampler follows to the real bytes), which is
//! what makes the restart lookup hit. Instance
//! 3 loads from a separate clone with persistence off; with no
//! `ColdTierContext` its adapter never touches the tier, so it is a clean
//! fresh-prefill baseline.
//!
//! The tier root is redirected to a per-process temp dir via
//! `MLX_COLD_CACHE_DIR`. The manager is a process-global `OnceLock`
//! initialized ONCE from that env on first use, so the env MUST be set
//! before the first persist-enabled load (which calls `enable_cold_tier`
//! -> `global_cold_cache()`), and this must be the only thing in the
//! process touching the tier — hence `#[ignore]` plus `--test-threads=1`.
//!
//! Gated on `MLX_TEST_MODEL_PATH`: a plain `cargo test --ignored` without
//! it still passes (the early-return short-circuits before any load).
//!
//! Run locally with:
//!
//! ```shell
//! MLX_COLD_CACHE_DIR=$(mktemp -d) \
//!     MLX_TEST_MODEL_PATH=./.cache/models/qwen3-0.6b-mlx-bf16 \
//!     cargo test -p mlx-core --test qwen3_cold_tier_parity \
//!     -- --ignored --test-threads=1 --nocapture
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mlx_core::cold_tier::cold_cache_stats_snapshot;
use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3::persistence::load_with_thread as qwen3_load_with_thread;
use mlx_core::tokenizer::ChatMessage;

/// Paged block size pinned into both clones' `config.json`. The cold tier
/// captures/restores whole blocks only, so the restored prefix asserted
/// below is a multiple of this.
const BLOCK_SIZE: u32 = 16;

// ---------------------------------------------------------------------------
// Test fixture helpers
// ---------------------------------------------------------------------------

/// Copy the source Qwen3 checkpoint directory into a fresh tempdir (under
/// the workspace `target/` so the OS doesn't garbage-collect it mid-run)
/// and patch `config.json` to force the block-paged adapter on and set the
/// `persist_paged_cache` flag. Weight files are symlinked (only
/// `config.json` is mutated per clone), so `enable_cold_tier`'s shard-size
/// hashing still sees the real byte sizes through the links.
fn clone_model_dir(src: &Path, suffix: &str, persist: bool) -> Result<PathBuf, String> {
    let pid = std::process::id();
    let workspace_target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let manifest = std::env::var("CARGO_MANIFEST_DIR")
                .expect("CARGO_MANIFEST_DIR must be set when running cargo test");
            let mut p = PathBuf::from(manifest);
            p.pop();
            p.pop();
            p.join("target")
        });

    let dst = workspace_target.join(format!("cold-tier-parity-{pid}-{suffix}"));
    if dst.exists() {
        let _ = fs::remove_dir_all(&dst);
    }
    fs::create_dir_all(&dst).map_err(|e| format!("create_dir_all({}): {e}", dst.display()))?;

    let read_dir = fs::read_dir(src).map_err(|e| format!("read_dir({}): {e}", src.display()))?;
    for entry in read_dir {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_file() {
            if entry.file_name() == "config.json" {
                fs::copy(&from, &to)
                    .map_err(|e| format!("copy({} -> {}): {e}", from.display(), to.display()))?;
            } else {
                std::os::unix::fs::symlink(&from, &to)
                    .map_err(|e| format!("symlink({} -> {}): {e}", from.display(), to.display()))?;
            }
        }
    }

    {
        let cfg_path = dst.join("config.json");
        let raw = fs::read_to_string(&cfg_path)
            .map_err(|e| format!("read config.json: {e} (path={})", cfg_path.display()))?;
        let mut cfg: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("parse config.json: {e} (path={})", cfg_path.display()))?;
        cfg["use_block_paged_cache"] = serde_json::Value::Bool(true);
        cfg["persist_paged_cache"] = serde_json::Value::Bool(persist);
        // Bound the adapter pool so the test stays light, and pin the block
        // size the restore assertion below is stated in terms of.
        cfg["paged_cache_memory_mb"] = serde_json::Value::from(256u32);
        cfg["paged_block_size"] = serde_json::Value::from(BLOCK_SIZE);
        let pretty = serde_json::to_string_pretty(&cfg)
            .map_err(|e| format!("serialize config.json: {e}"))?;
        fs::write(&cfg_path, pretty)
            .map_err(|e| format!("write config.json: {e} (path={})", cfg_path.display()))?;
    }

    Ok(dst)
}

/// Greedy decode, no penalties, fixed token budget — same knobs the
/// paged-vs-flat parity gate uses, so any divergence is attributable to the
/// cache backend rather than sampling noise.
fn parity_chat_config(max_new_tokens: i32) -> ChatConfig {
    ChatConfig {
        cache_owner_id: None,
        cache_root_owner_id: None,
        max_new_tokens: Some(max_new_tokens),
        temperature: Some(0.0),
        top_k: None,
        top_p: None,
        min_p: None,
        repetition_penalty: Some(1.0),
        repetition_context_size: None,
        presence_penalty: Some(0.0),
        presence_context_size: None,
        frequency_penalty: Some(0.0),
        frequency_context_size: None,
        max_consecutive_tokens: None,
        max_ngram_repeats: None,
        ngram_size: None,
        tools: None,
        reasoning_effort: None,
        thinking_token_budget: Some(32),
        include_reasoning: Some(true),
        report_performance: Some(false),
        reuse_cache: Some(true),
        enable_mtp: None,
        mtp_depth: None,
        mtp_adaptive_depth: None,
    }
}

fn user_message(content: &str) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: content.to_string(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        thinking_enabled: None,
        images: None,
        audio: None,
    }
}

/// A single prompt long enough that, after the chat template wraps it, the
/// tokenized prompt spans several full 16-token blocks — so the restore
/// across the restart covers comfortably more than `BLOCK_SIZE * 2`.
const PROMPT: &str = "Please explain, in a few clear sentences, how a block-paged \
    key-value cache stores attention state across many transformer layers, why \
    persisting warm prefixes to local solid-state storage can speed up a later \
    process restart, and what tradeoffs an engineer should weigh when choosing the \
    block size for such a cache.";

/// Resolve the source model path from `MLX_TEST_MODEL_PATH`, returning
/// `None` (and logging a skip notice) when unset or missing.
fn resolve_source_model() -> Option<PathBuf> {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_MODEL_PATH unset (point it at e.g. \
             ./.cache/models/qwen3-0.6b-mlx-bf16)"
        );
        return None;
    };
    let p = PathBuf::from(&model_path);
    if !p.exists() {
        eprintln!(
            "skipping: MLX_TEST_MODEL_PATH does not exist: {}",
            p.display()
        );
        return None;
    }
    Some(p)
}

/// Block on the process-global cold tier's background writer until it has
/// committed the enqueued captures to disk. Capture is asynchronous — the
/// blocks land on a write queue during turn finalize and are fsync'd +
/// index-published off-thread — so the restart read must wait for
/// `bytes_written` to appear and then quiesce, or it races an empty tier.
async fn wait_for_cold_writes_drained() {
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut last_written = u64::MAX;
    let mut stable_since: Option<Instant> = None;
    loop {
        let (enqueued, written) = cold_cache_stats_snapshot()
            .map(|s| (s.enqueued, s.bytes_written))
            .unwrap_or((0, 0));
        if enqueued > 0 && written > 0 {
            if written == last_written {
                let since = stable_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= Duration::from_millis(300) {
                    return;
                }
            } else {
                stable_since = None;
            }
        }
        last_written = written;
        if Instant::now() >= deadline {
            eprintln!(
                "warning: cold-tier drain wait timed out (enqueued={enqueued} \
                 bytes_written={written}); proceeding — restore will report the miss"
            );
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Panic with a first-differing-byte repro hint when two greedy outputs
/// diverge. A real restore fault (wrong positions, dropped/duplicated KV, a
/// silent cold-prefill) diverges at byte 0; float non-associativity near a
/// late argmax tie would diverge deep into the stream.
fn assert_text_eq(label_a: &str, a: &str, label_b: &str, b: &str) {
    if a != b {
        let first_diff = a
            .as_bytes()
            .iter()
            .zip(b.as_bytes().iter())
            .position(|(x, y)| x != y);
        panic!(
            "TEXT MISMATCH {label_a} vs {label_b}. first_diff_byte={first_diff:?}\n\
             {label_a} = {a:?}\n\
             {label_b} = {b:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Cold-tier restart parity gate
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3 checkpoint; run with --test-threads=1"]
async fn qwen3_cold_tier_restart_parity() {
    let Some(src) = resolve_source_model() else {
        return;
    };

    // Redirect the cold tier to a per-process temp dir. This MUST run before
    // the first persist-enabled load below (which is the first
    // `global_cold_cache()` caller in the process). `#[ignore]` +
    // `--test-threads=1` guarantee this is the only test in the binary that
    // initializes the process-global tier.
    let cold_root = std::env::temp_dir().join(format!("mlx-cold-parity-{}", std::process::id()));
    let _ = fs::remove_dir_all(&cold_root);
    fs::create_dir_all(&cold_root).expect("create cold-cache temp root");
    // SAFETY: set before any model load and thus before any thread reads the
    // env or the process-global tier; `#[ignore]` + `--test-threads=1` keep
    // this the sole toucher in the process.
    unsafe { std::env::set_var("MLX_COLD_CACHE_DIR", &cold_root) };

    // Instances 1 and 2 share this clone so their fingerprints match exactly.
    let persist_dir = match clone_model_dir(&src, "persist", true) {
        Ok(p) => p,
        Err(e) => panic!("failed to clone persist model dir: {e}"),
    };
    let nopersist_dir = match clone_model_dir(&src, "nopersist", false) {
        Ok(p) => p,
        Err(e) => panic!("failed to clone no-persist model dir: {e}"),
    };

    // Instance 1: persistence on. Fresh prefill; captures full blocks to the
    // cold tier on turn finalize. Dropped before the restart.
    let text_a = {
        let model1 = qwen3_load_with_thread(&persist_dir.to_string_lossy())
            .await
            .expect("failed to load persist instance 1");
        let r = model1
            .chat_session_start(vec![user_message(PROMPT)], Some(parity_chat_config(32)))
            .await
            .expect("instance 1 chat_session_start failed");
        eprintln!(
            "instance 1 (persist, capture): num_tokens={} cached={} finish={}",
            r.num_tokens, r.cached_tokens, r.finish_reason
        );
        r.text
    };

    // Let the background writer commit the captures to disk before the restart
    // reads them back.
    wait_for_cold_writes_drained().await;

    // Instance 2: fresh model (empty in-memory hot cache) standing in for a
    // process restart. Its `find_cached_prefix` must miss the hot cache and
    // restore the persisted prefix from the cold tier.
    let (text_b, cached_b) = {
        let model2 = qwen3_load_with_thread(&persist_dir.to_string_lossy())
            .await
            .expect("failed to load persist instance 2 (restart)");
        let r = model2
            .chat_session_start(vec![user_message(PROMPT)], Some(parity_chat_config(32)))
            .await
            .expect("instance 2 chat_session_start failed");
        eprintln!(
            "instance 2 (restart, restore): num_tokens={} cached={} finish={}",
            r.num_tokens, r.cached_tokens, r.finish_reason
        );
        (r.text, r.cached_tokens)
    };

    // Instance 3: persistence off — a clean fresh-prefill baseline that never
    // touches the tier (no `ColdTierContext`).
    let text_c = {
        let model3 = qwen3_load_with_thread(&nopersist_dir.to_string_lossy())
            .await
            .expect("failed to load no-persist instance 3");
        let r = model3
            .chat_session_start(vec![user_message(PROMPT)], Some(parity_chat_config(32)))
            .await
            .expect("instance 3 chat_session_start failed");
        eprintln!(
            "instance 3 (no-persist, baseline): num_tokens={} cached={} finish={}",
            r.num_tokens, r.cached_tokens, r.finish_reason
        );
        r.text
    };

    // Cold restore must have actually engaged across the "restart": at least
    // two full blocks of the prompt prefix served from disk. Zero here would
    // mean a silent cold-prefill fallback that passes the byte comparison
    // while proving nothing about persistence.
    assert!(
        cached_b >= BLOCK_SIZE * 2,
        "cold restore did not engage across restart: cached_tokens={cached_b} \
         (expected >= {})",
        BLOCK_SIZE * 2
    );

    // Byte-for-byte greedy determinism: the restored decode must reproduce the
    // original, and match the persistence-off baseline.
    assert_text_eq(
        "instance 1 (capture)",
        &text_a,
        "instance 2 (restore)",
        &text_b,
    );
    assert_text_eq(
        "instance 1 (capture)",
        &text_a,
        "instance 3 (no-persist)",
        &text_c,
    );

    eprintln!("Qwen3 cold-tier restart parity: cached_tokens={cached_b}, text matched a==b==c");

    // Best-effort cleanup. SAFETY: single-threaded teardown, no other reader.
    unsafe { std::env::remove_var("MLX_COLD_CACHE_DIR") };
    let _ = fs::remove_dir_all(&cold_root);
}
