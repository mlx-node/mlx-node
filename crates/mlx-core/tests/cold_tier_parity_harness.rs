//! Reusable restart-parity gate for the SSD cold tier, shared by every family.
//!
//! Lifted verbatim (behaviour-wise) out of `qwen3_cold_tier_parity.rs` so a
//! family joining `cold_tier::COLD_RESTORE_FAMILIES` arrives against a gate
//! that is already trusted, instead of hand-rolling its own and accidentally
//! weakening an assertion.
//!
//! # The three instances
//!
//! | # | persist | role |
//! |---|---------|------|
//! | 1 | on  | fresh prefill; captures full paged blocks to the tier on finalize |
//! | 2 | on  | fresh model = process-restart stand-in; MUST restore from disk |
//! | 3 | off | never attaches a `ColdTierContext`; clean fresh-prefill baseline |
//!
//! Instances 1 and 2 load from the SAME on-disk clone so their cold-tier
//! fingerprints — parsed config bytes, a full per-shard weight-content digest,
//! and pool geometry/dtype (`cold_tier::build_model_fingerprint`) — are
//! byte-identical, which is what makes the restart lookup hit. Weight files in
//! a clone are symlinks (only `config.json` is rewritten per clone) and the
//! clone carries no download marker, so the digest follows the links to the
//! real bytes and this exercises the full-hash fallback.
//!
//! # What the gate proves
//!
//! 1. **Restore engaged.** `cached_tokens` on instance 2 covers at least
//!    `min_restored_tokens` (default two full blocks). Zero here is a silent
//!    cold-prefill fallback that would pass the text comparison while proving
//!    nothing about persistence.
//! 2. **Restore engaged *soundly*.** The process-global cold stats gained at
//!    least one `hit` across instance 2, and recorded ZERO `corruptions` over
//!    the whole run. Without this a fail-open restore path — one that swallows
//!    a malformed on-disk object, counts the corruption and quietly recomputes
//!    — masquerades as a pass. That is exactly the failure mode the cold-tier
//!    work is defending against, so it is asserted, not merely logged.
//! 3. **Parity.** `text` is byte-for-byte equal across all three instances and
//!    `num_tokens` matches, under greedy/no-penalty decode so any divergence is
//!    attributable to the cache backend rather than sampling noise.
//!
//! # Process-global constraints
//!
//! The tier manager is a process-global `OnceLock` initialized ONCE from
//! `MLX_COLD_CACHE_DIR` on first use, so the root must be fixed before the
//! first persist-enabled load (the first `enable_cold_tier` ->
//! `global_cold_cache()` caller), and the scenario must be the only thing in
//! the process touching the tier. Hence every family test wrapping this is
//! `#[ignore]`d and run with `--test-threads=1`.
//!
//! `MLX_COLD_CACHE_DIR`, when already set by the caller, is honoured as-is and
//! left in place; otherwise a per-process temp root is created and removed on
//! success.
//!
//! # Usage
//!
//! ```ignore
//! mod cold_tier_parity_harness;
//! use cold_tier_parity_harness as harness;
//!
//! #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
//! #[ignore = "needs MLX_TEST_MODEL_PATH; run with --test-threads=1"]
//! async fn my_family_cold_tier_restart_parity() {
//!     harness::run_restart_parity(
//!         harness::ColdTierParitySpec::new("my_family"),
//!         |model_dir, messages, config| async move {
//!             let model = my_family_load_with_thread(&model_dir.to_string_lossy()).await?;
//!             model.chat_session_start(messages, Some(config)).await
//!         },
//!     )
//!     .await;
//! }
//! ```
//!
//! The closure owns the family-specific typing (each family's `chat_session_start`
//! is an inherent method emitted by `chat_napi_surface!`, not a trait method) and
//! MUST drop the model before it returns, so instance 2 really does start from an
//! empty in-memory hot cache.
//!
//! Cargo auto-discovers every `tests/*.rs` as a test target, so this file also
//! builds as a standalone binary with zero tests. That is harmless — and it
//! type-checks the harness even when no family test is being built.

#![allow(dead_code)]

use std::fs;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mlx_core::cold_tier::{cold_cache_drain, cold_cache_stats_snapshot};
use mlx_core::engine::types::{ChatConfig, ChatResult};
use mlx_core::tokenizer::ChatMessage;

/// Default paged block size pinned into both clones' `config.json`. The cold
/// tier captures/restores whole blocks only, so the restored prefix a family
/// asserts is a multiple of this.
pub const DEFAULT_BLOCK_SIZE: u32 = 16;

/// A prompt long enough that, after a chat template wraps it, the tokenized
/// prompt spans several full 16-token blocks — so the restore across the
/// restart covers comfortably more than two blocks.
pub const DEFAULT_PROMPT: &str = "Please explain, in a few clear sentences, how a block-paged \
    key-value cache stores attention state across many transformer layers, why \
    persisting warm prefixes to local solid-state storage can speed up a later \
    process restart, and what tradeoffs an engineer should weigh when choosing the \
    block size for such a cache.";

/// What the RESTORE instance (instance 2) did, handed to a family's
/// [`ColdTierParitySpec::with_restore_inspector`] callback.
///
/// The shared gate can only see `ChatResult`, which says *how much* prefix came
/// back but nothing about *what backed it*. A hybrid family's interesting claim
/// lives one level down — which auxiliary source primed the sliding/recurrent
/// half, and whether any replay was still paid — and the only place that is
/// observable from outside the crate is the inference-trace channel. So the
/// harness slices the trace to exactly instance 2's turn and hands it over; the
/// family decides what the lines have to say.
pub struct RestoreObservation<'a> {
    pub family: &'a str,
    /// Instance 2's result. `cached_tokens` is the adapter's
    /// `cached_token_count` verbatim.
    pub result: &'a ChatResult,
    /// Everything appended to `MLX_INFERENCE_TRACE_FILE` while instance 2 ran.
    ///
    /// EMPTY when the trace channel is not configured (`MLX_INFERENCE_TRACE` /
    /// `MLX_INFERENCE_TRACE_FILE` unset, or latched off earlier in this
    /// process). An inspector that needs the channel must say so itself — the
    /// harness deliberately does not turn tracing on for families that did not
    /// ask, because every other family's gate would then run instrumented.
    pub trace: &'a str,
}

/// See [`ColdTierParitySpec::with_restore_inspector`]. Boxed rather than a
/// generic parameter so adding one does not change `run_restart_parity`'s
/// signature for the families that pass none.
pub type RestoreInspector = Box<dyn Fn(&RestoreObservation<'_>) + Send + Sync>;

/// Per-family knobs for [`run_restart_parity`].
///
/// Everything here is a *fixture* dial. The gate's assertions themselves are
/// fixed: no family may opt out of the parity, engagement or corruption checks.
pub struct ColdTierParitySpec {
    /// Family label, used only in log lines and panic messages.
    pub family: &'static str,
    /// Env var naming the source checkpoint. Each family gets its own test
    /// binary, so they can all share `MLX_TEST_MODEL_PATH`.
    pub model_path_env: &'static str,
    /// `paged_block_size` forced into both clones.
    pub block_size: u32,
    /// `paged_cache_memory_mb` forced into both clones — bounded so the test
    /// stays light.
    pub pool_memory_mb: u32,
    /// Extra `config.json` overrides applied on top of the fixed set
    /// (`use_block_paged_cache`, `persist_paged_cache`, `paged_cache_memory_mb`,
    /// `paged_block_size`) for families that need more to reach the paged path.
    pub extra_config: Vec<(String, serde_json::Value)>,
    /// Single-turn prompt run identically by all three instances.
    pub prompt: &'static str,
    /// Decode budget. Short: the gate is about the prefix, not the tail.
    pub max_new_tokens: i32,
    /// Thinking budget, for families whose template opens a think block.
    pub thinking_token_budget: Option<i32>,
    /// Minimum `cached_tokens` instance 2 must report. `None` => `block_size * 2`.
    pub min_restored_tokens: Option<u32>,
    /// Extra persist-enabled turns run BEFORE instance 1, purely to deepen the
    /// persisted chain. Default 0 — qwen3 is bit-for-bit unaffected.
    ///
    /// This is a fixture dial, not an assertion knob: the cold writer's queue
    /// is bounded (`DEFAULT_QUEUE_DEPTH`) and
    /// `ColdTierWalk::capture_chain` STOPS at the first block the queue
    /// refuses, so a single turn only ever persists the first handful of
    /// blocks no matter how long the prompt is. Blocks already on disk are
    /// `contains`-skipped without re-enqueueing, so each further turn advances
    /// the frontier by another queue's worth. A family whose auxiliary state is
    /// only anchored at a deep boundary (gemma4's long-prompt gate targets the
    /// decode-cadence checkpoint at one whole `sliding_window`) therefore
    /// cannot reach that boundary in one turn, and would fail the gate for a
    /// reason that has nothing to do with its restore path.
    ///
    /// Warm-up turns are neither compared nor asserted on — the three measured
    /// instances below are unchanged.
    pub capture_warmup_turns: usize,
    /// Optional family-specific assertion over the RESTORE instance, run after
    /// the shared engagement/soundness checks and before the parity checks.
    ///
    /// `None` for every family that ships one — they are fully covered by the
    /// fixed assertions — so this is inert by default and no existing gate's
    /// behaviour changes.
    pub inspect_restore: Option<RestoreInspector>,
}

impl ColdTierParitySpec {
    /// Defaults matching the original Qwen3 gate.
    pub fn new(family: &'static str) -> Self {
        Self {
            family,
            model_path_env: "MLX_TEST_MODEL_PATH",
            block_size: DEFAULT_BLOCK_SIZE,
            pool_memory_mb: 256,
            extra_config: Vec::new(),
            prompt: DEFAULT_PROMPT,
            max_new_tokens: 32,
            thinking_token_budget: Some(32),
            min_restored_tokens: None,
            capture_warmup_turns: 0,
            inspect_restore: None,
        }
    }

    /// Add one `config.json` override applied to both clones.
    pub fn with_config(mut self, key: &str, value: serde_json::Value) -> Self {
        self.extra_config.push((key.to_string(), value));
        self
    }

    pub fn with_block_size(mut self, block_size: u32) -> Self {
        self.block_size = block_size;
        self
    }

    pub fn with_pool_memory_mb(mut self, pool_memory_mb: u32) -> Self {
        self.pool_memory_mb = pool_memory_mb;
        self
    }

    pub fn with_prompt(mut self, prompt: &'static str) -> Self {
        self.prompt = prompt;
        self
    }

    pub fn with_max_new_tokens(mut self, max_new_tokens: i32) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    pub fn with_min_restored_tokens(mut self, tokens: u32) -> Self {
        self.min_restored_tokens = Some(tokens);
        self
    }

    /// See [`Self::capture_warmup_turns`].
    pub fn with_capture_warmup_turns(mut self, turns: usize) -> Self {
        self.capture_warmup_turns = turns;
        self
    }

    /// Add a family-specific assertion over the restore instance. See
    /// [`RestoreObservation`]; the callback is expected to panic on failure,
    /// like every other assertion in this gate.
    pub fn with_restore_inspector<F>(mut self, inspect: F) -> Self
    where
        F: Fn(&RestoreObservation<'_>) + Send + Sync + 'static,
    {
        self.inspect_restore = Some(Box::new(inspect));
        self
    }

    fn min_restored(&self) -> u32 {
        self.min_restored_tokens
            .unwrap_or_else(|| self.block_size.saturating_mul(2))
    }
}

/// One user turn, shared by all three instances.
pub fn user_message(content: &str) -> ChatMessage {
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

/// Greedy decode, no penalties, fixed token budget — the same knobs the
/// paged-vs-flat parity gates use, so any divergence is attributable to the
/// cache backend rather than sampling noise. Every field left at
/// `ChatConfig::default()` is `None`.
pub fn parity_chat_config(spec: &ColdTierParitySpec) -> ChatConfig {
    ChatConfig {
        max_new_tokens: Some(spec.max_new_tokens),
        temperature: Some(0.0),
        repetition_penalty: Some(1.0),
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        thinking_token_budget: spec.thinking_token_budget,
        include_reasoning: Some(true),
        report_performance: Some(false),
        reuse_cache: Some(true),
        ..ChatConfig::default()
    }
}

/// Resolve the source model path from `spec.model_path_env`, returning `None`
/// (and logging a skip notice) when unset or missing, so a plain
/// `cargo test --ignored` without a checkpoint still passes.
fn resolve_source_model(spec: &ColdTierParitySpec) -> Option<PathBuf> {
    let env = spec.model_path_env;
    let Ok(model_path) = std::env::var(env) else {
        eprintln!(
            "skipping {}: {env} unset (point it at a real {} checkpoint)",
            spec.family, spec.family
        );
        return None;
    };
    let p = PathBuf::from(&model_path);
    if !p.exists() {
        eprintln!(
            "skipping {}: {env} does not exist: {}",
            spec.family,
            p.display()
        );
        return None;
    }
    Some(p)
}

/// Copy the source checkpoint directory into a fresh dir under the workspace
/// `target/` (so the OS doesn't garbage-collect it mid-run) and patch
/// `config.json` to force the block-paged adapter on and set the
/// `persist_paged_cache` flag. Weight files are symlinked, so the cold tier's
/// full-shard digest still hashes the real bytes through the links.
fn clone_model_dir(
    src: &Path,
    spec: &ColdTierParitySpec,
    suffix: &str,
    persist: bool,
) -> Result<PathBuf, String> {
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

    let dst = workspace_target.join(format!("cold-tier-parity-{}-{pid}-{suffix}", spec.family));
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
        // size the restore assertion is stated in terms of.
        cfg["paged_cache_memory_mb"] = serde_json::Value::from(spec.pool_memory_mb);
        cfg["paged_block_size"] = serde_json::Value::from(spec.block_size);
        for (key, value) in &spec.extra_config {
            cfg[key.as_str()] = value.clone();
        }
        let pretty = serde_json::to_string_pretty(&cfg)
            .map_err(|e| format!("serialize config.json: {e}"))?;
        fs::write(&cfg_path, pretty)
            .map_err(|e| format!("write config.json: {e} (path={})", cfg_path.display()))?;
    }

    Ok(dst)
}

/// Block until the process-global tier's background writer has committed the
/// enqueued captures to disk. Capture is asynchronous — the blocks land on a
/// write queue during turn finalize and are fsync'd + index-published
/// off-thread — so the restart read must wait, or it races an empty tier.
///
/// Two layers: an explicit writer barrier (`cold_cache_drain`), then a
/// `bytes_written`-quiesced poll, so a barrier that is admitted before the
/// captures are enqueued still cannot let the restart read run early.
async fn wait_for_cold_writes_drained() {
    let drained = tokio::task::spawn_blocking(|| cold_cache_drain(20_000))
        .await
        .unwrap_or(false);
    if !drained {
        eprintln!("warning: cold-tier write barrier did not ack within 20s");
    }

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

/// The inference-trace sink, when the caller configured one.
///
/// `mlx_core::inference_trace` appends every `[MLX_TRACE]` line to this file,
/// so byte offsets into it delimit turns: snapshot the length before an
/// instance runs and everything after that offset belongs to it. Only read
/// when a family passed a [`RestoreInspector`]; otherwise the harness never
/// touches tracing at all.
fn inference_trace_path() -> Option<PathBuf> {
    let raw = std::env::var("MLX_INFERENCE_TRACE_FILE").ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(PathBuf::from(trimmed))
}

/// Current length of the trace file, or 0 when there is no file yet.
fn inference_trace_len(path: Option<&Path>) -> u64 {
    path.and_then(|p| fs::metadata(p).ok())
        .map(|meta| meta.len())
        .unwrap_or(0)
}

/// Everything appended to the trace file since `offset`.
///
/// Lossy UTF-8 rather than a hard error: this feeds an assertion message, and
/// a torn multi-byte tail must never turn a real restore result into an
/// unrelated panic.
fn inference_trace_since(path: Option<&Path>, offset: u64) -> String {
    let Some(path) = path else {
        return String::new();
    };
    let Ok(bytes) = fs::read(path) else {
        return String::new();
    };
    let start = usize::try_from(offset)
        .unwrap_or(usize::MAX)
        .min(bytes.len());
    String::from_utf8_lossy(&bytes[start..]).into_owned()
}

/// Where the tier root came from, so cleanup only removes what we created.
enum ColdRoot {
    /// `MLX_COLD_CACHE_DIR` was already set; caller owns the directory.
    Inherited(PathBuf),
    /// We created a per-process temp root and set the env var.
    Created(PathBuf),
}

impl ColdRoot {
    fn path(&self) -> &Path {
        match self {
            ColdRoot::Inherited(p) | ColdRoot::Created(p) => p,
        }
    }
}

/// Fix the tier root BEFORE any model load. The manager is a process-global
/// `OnceLock` initialized once from this env, so a later change is ignored.
fn prepare_cold_root() -> ColdRoot {
    match std::env::var("MLX_COLD_CACHE_DIR") {
        Ok(dir) if !dir.trim().is_empty() => {
            let path = PathBuf::from(dir);
            fs::create_dir_all(&path).expect("create caller-supplied MLX_COLD_CACHE_DIR");
            ColdRoot::Inherited(path)
        }
        _ => {
            let path = std::env::temp_dir().join(format!("mlx-cold-parity-{}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create cold-cache temp root");
            // SAFETY: set before any model load and thus before any thread
            // reads the env or the process-global tier; `#[ignore]` +
            // `--test-threads=1` keep this the sole toucher in the process.
            unsafe { std::env::set_var("MLX_COLD_CACHE_DIR", &path) };
            ColdRoot::Created(path)
        }
    }
}

/// Run the three-instance cold-tier restart-parity gate for one family.
///
/// `run_turn` loads a FRESH model from the given directory, runs exactly one
/// turn, and drops the model before returning — instance 2 only stands in for a
/// process restart if its in-memory hot cache really is empty.
///
/// Returns without asserting anything (logging a skip notice) when the
/// checkpoint env var is unset or points nowhere.
pub async fn run_restart_parity<F, Fut>(spec: ColdTierParitySpec, run_turn: F)
where
    F: Fn(PathBuf, Vec<ChatMessage>, ChatConfig) -> Fut,
    Fut: Future<Output = napi::Result<ChatResult>>,
{
    let Some(src) = resolve_source_model(&spec) else {
        return;
    };

    let cold_root = prepare_cold_root();
    eprintln!(
        "[{}] cold tier root: {}",
        spec.family,
        cold_root.path().display()
    );

    // Instances 1 and 2 share this clone so their fingerprints match exactly.
    let persist_dir = match clone_model_dir(&src, &spec, "persist", true) {
        Ok(p) => p,
        Err(e) => panic!("[{}] failed to clone persist model dir: {e}", spec.family),
    };
    let nopersist_dir = match clone_model_dir(&src, &spec, "nopersist", false) {
        Ok(p) => p,
        Err(e) => panic!(
            "[{}] failed to clone no-persist model dir: {e}",
            spec.family
        ),
    };

    let turn = |dir: &PathBuf| {
        run_turn(
            dir.clone(),
            vec![user_message(spec.prompt)],
            parity_chat_config(&spec),
        )
    };

    // Chain warm-up (opt-in; see `capture_warmup_turns`). Each turn advances
    // the persisted chain's frontier by another writer-queue's worth of
    // blocks, because blocks already on disk are skipped without re-enqueueing.
    // Nothing here is asserted on — this only deepens what instance 2 can find.
    for turn_index in 0..spec.capture_warmup_turns {
        let result = turn(&persist_dir).await.unwrap_or_else(|e| {
            panic!(
                "[{}] capture warm-up turn {} failed: {e}",
                spec.family,
                turn_index + 1
            )
        });
        // Drain per turn: leaving the writer queue full would starve the next
        // turn's capture of the very slots it needs to advance the frontier.
        wait_for_cold_writes_drained().await;
        eprintln!(
            "[{}] warm-up turn {}/{}: cached={} (chain deepening; not asserted)",
            spec.family,
            turn_index + 1,
            spec.capture_warmup_turns,
            result.cached_tokens
        );
    }

    // Instance 1: persistence on. Fresh prefill; captures full blocks to the
    // cold tier on turn finalize. Dropped by `run_turn` before the restart.
    let result_a = turn(&persist_dir)
        .await
        .unwrap_or_else(|e| panic!("[{}] instance 1 (capture) failed: {e}", spec.family));
    eprintln!(
        "[{}] instance 1 (persist, capture): num_tokens={} cached={} finish={}",
        spec.family, result_a.num_tokens, result_a.cached_tokens, result_a.finish_reason
    );

    // Let the background writer commit the captures to disk before the restart
    // reads them back.
    wait_for_cold_writes_drained().await;

    let stats_before = cold_cache_stats_snapshot();

    // Only families that asked for an inspector pay any attention to tracing;
    // for everyone else these stay `None`/0 and nothing is read.
    let trace_path = spec
        .inspect_restore
        .as_ref()
        .and_then(|_| inference_trace_path());
    let trace_offset = inference_trace_len(trace_path.as_deref());

    // Instance 2: fresh model (empty in-memory hot cache) standing in for a
    // process restart. Its `find_cached_prefix*` must miss the hot cache and
    // restore the persisted prefix from the cold tier.
    let result_b = turn(&persist_dir)
        .await
        .unwrap_or_else(|e| panic!("[{}] instance 2 (restore) failed: {e}", spec.family));
    let restore_trace = inference_trace_since(trace_path.as_deref(), trace_offset);
    eprintln!(
        "[{}] instance 2 (restart, restore): num_tokens={} cached={} finish={}",
        spec.family, result_b.num_tokens, result_b.cached_tokens, result_b.finish_reason
    );

    let stats_after = cold_cache_stats_snapshot();

    // Instance 3: persistence off — a clean fresh-prefill baseline that never
    // touches the tier (no `ColdTierContext`).
    let result_c = turn(&nopersist_dir)
        .await
        .unwrap_or_else(|e| panic!("[{}] instance 3 (no-persist) failed: {e}", spec.family));
    eprintln!(
        "[{}] instance 3 (no-persist, baseline): num_tokens={} cached={} finish={}",
        spec.family, result_c.num_tokens, result_c.cached_tokens, result_c.finish_reason
    );

    // ---- 1. Restore engaged at all ---------------------------------------
    let min_restored = spec.min_restored();
    assert!(
        result_b.cached_tokens >= min_restored,
        "[{}] cold restore did not engage across restart: cached_tokens={} (expected >= {})",
        spec.family,
        result_b.cached_tokens,
        min_restored
    );

    // ---- 2. Restore engaged SOUNDLY --------------------------------------
    // A fail-open restore counts the bad object and recomputes, which still
    // produces correct text — so text parity alone cannot see it. Require a
    // real hit across the restart and zero corruptions over the whole run.
    let after = stats_after.unwrap_or_else(|| {
        panic!(
            "[{}] cold tier never initialized: no stats snapshot after the restart instance",
            spec.family
        )
    });
    let hits_before = stats_before.as_ref().map(|s| s.hits).unwrap_or(0);
    assert!(
        after.hits > hits_before,
        "[{}] no cold-tier hit recorded across the restart: hits {hits_before} -> {} \
         (misses={}, corruptions={}, bytes_restored={}) — cached_tokens={} came from \
         somewhere other than the tier",
        spec.family,
        after.hits,
        after.misses,
        after.corruptions,
        after.bytes_restored,
        result_b.cached_tokens
    );
    assert_eq!(
        after.corruptions, 0,
        "[{}] cold tier recorded {} corruption(s): a malformed on-disk object was \
         swallowed and the prefix silently recomputed — the restore path fell open, \
         which text parity alone cannot detect",
        spec.family, after.corruptions
    );
    eprintln!(
        "[{}] cold stats after restart: hits={} misses={} enqueued={} queue_drops={} \
         bytes_written={} bytes_restored={} evictions={} corruptions={}",
        spec.family,
        after.hits,
        after.misses,
        after.enqueued,
        after.queue_drops,
        after.bytes_written,
        after.bytes_restored,
        after.evictions,
        after.corruptions
    );

    // ---- 2b. Family-specific view of the restore -------------------------
    // Runs after the shared checks (so a plain "nothing restored" failure
    // reports itself first, in the words every family shares) and before
    // parity (so a family that can name *why* the restore is wrong gets to
    // say it before a text diff does).
    if let Some(inspect) = spec.inspect_restore.as_ref() {
        inspect(&RestoreObservation {
            family: spec.family,
            result: &result_b,
            trace: &restore_trace,
        });
    }

    // ---- 3. Byte-for-byte greedy parity ----------------------------------
    assert_text_eq(
        "instance 1 (capture)",
        &result_a.text,
        "instance 2 (restore)",
        &result_b.text,
    );
    assert_text_eq(
        "instance 1 (capture)",
        &result_a.text,
        "instance 3 (no-persist)",
        &result_c.text,
    );
    assert_eq!(
        result_a.num_tokens, result_b.num_tokens,
        "[{}] num_tokens diverged: instance 1 = {}, instance 2 (restore) = {}",
        spec.family, result_a.num_tokens, result_b.num_tokens
    );
    assert_eq!(
        result_a.num_tokens, result_c.num_tokens,
        "[{}] num_tokens diverged: instance 1 = {}, instance 3 (no-persist) = {}",
        spec.family, result_a.num_tokens, result_c.num_tokens
    );

    eprintln!(
        "[{}] cold-tier restart parity PASS: cached_tokens={} hits={} corruptions=0, \
         text and num_tokens matched a==b==c",
        spec.family, result_b.cached_tokens, after.hits
    );

    // Best-effort cleanup; only touches what this run created.
    let _ = fs::remove_dir_all(&persist_dir);
    let _ = fs::remove_dir_all(&nopersist_dir);
    if let ColdRoot::Created(path) = &cold_root {
        // SAFETY: single-threaded teardown, no other reader.
        unsafe { std::env::remove_var("MLX_COLD_CACHE_DIR") };
        let _ = fs::remove_dir_all(path);
    }
}
