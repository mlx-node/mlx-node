//! Reusable restart-parity gate for the SSD cold tier, shared by every family.
//!
//! Lifted verbatim (behaviour-wise) out of `qwen3_cold_tier_parity.rs` so a
//! family joining `cold_tier::COLD_RESTORE_FAMILIES` arrives against a gate
//! that is already trusted, instead of hand-rolling its own and accidentally
//! weakening an assertion.
//!
//! # The three instances
//!
//! | # | persist | prompt | role |
//! |---|---------|--------|------|
//! | 1 | on  | `prompt` | fresh prefill; captures full paged blocks to the tier on finalize |
//! | 2 | on  | `restore_prompt` | fresh model = process-restart stand-in; MUST restore from disk |
//! | 3 | off | `restore_prompt` | never attaches a `ColdTierContext`; clean fresh-prefill baseline |
//!
//! [`ColdTierParitySpec::restore_prompt`] defaults to `None`, in which case it
//! IS `prompt` and all three instances run the same text — the original
//! behaviour, unchanged for every family that does not set it.
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
//!    attributable to the cache backend rather than sampling noise. In
//!    `restore_prompt` mode instance 1 answers a different question, so the
//!    comparison narrows to 2-vs-3 — a turn that reconciled down onto a
//!    mid-ladder boundary must produce the same bytes as a clean prefill of the
//!    same prompt.
//! 4. **The restore anchored SHALLOW.** `restore_prompt` mode only. The restore
//!    landed on a checkpoint rung strictly shallower than the capture prompt's
//!    deepest one. Without this the gate is satisfied by an implementation that
//!    publishes exactly ONE boundary at the prompt's end — which is what the
//!    checkpoint ladder replaced, and what a run on a ~90-token prompt cannot
//!    tell apart from a ladder, because one turn's chain already reaches the
//!    prompt's end. See [`ColdTierParitySpec::restore_prompt`].
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

use mlx_core::cold_tier::{
    ColdSidecarTelemetry, cold_cache_drain, cold_cache_stats_snapshot, cold_sidecar_telemetry,
};
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

/// Capture-side prompt for a family that wants the ladder gate — the prompt
/// instance 1 (and every warm-up turn) runs.
///
/// Long on purpose. The blindness this exists to fix is a length problem:
/// [`DEFAULT_PROMPT`] is ~90 tokens, whose whole ladder is `[16, 80]`, and one
/// turn's chain reach (~8 blocks = 128 tokens) already covers the deepest rung,
/// so the restore always anchors at the prompt's end and the shallow rungs are
/// never touched.
///
/// MEASURED on `qwen3.5-0.8b-mlx-bf16`, not estimated: 1259 tokens after the
/// chat template, ladder `[16, 64, 304, 1248]`, deepest rung 78 blocks in —
/// far past anything a bounded writer queue drains in a handful of turns. The
/// observed run anchors the restore at 304 (rung 3 of 4). Tokenizers differ
/// between families, so treat that as the shape, not a constant; every
/// assertion reads the runtime `prompt_tokens` rather than any number here.
pub fn ladder_capture_prompt() -> &'static str {
    ladder_prompt("Ultimately, weigh the checksum cost against the eviction policy")
}

/// Restore-side prompt — byte-identical body, different closing question.
///
/// The two token streams agree for a long prefix and then part. That
/// divergence is a HARD ceiling on how deep `kv_chain_upper_bound` can
/// reconcile (it breaks at the first block key not on disk), independent of how
/// fast the writer queue drained during capture. So the restore is forced onto
/// a rung below the divergence no matter what the machine's I/O did — which is
/// what makes the shallow-rung assertion kill the "one boundary at the prompt's
/// end" implementation unconditionally, rather than only on a slow disk.
pub fn ladder_restore_prompt() -> &'static str {
    ladder_prompt("Separately, rank the restore latency below the block size")
}

/// A fixed 15-note body plus a closing `tail`, leaked to `'static` because
/// [`ColdTierParitySpec::prompt`] is a `&'static str`.
///
/// Lives here rather than in each family's test file so dense and MoE cannot
/// drift into two differently-sized fixtures making two different claims. The
/// notes are numbered so the text is not a degenerate repetition, which would
/// make the block-chain identity uninteresting.
///
/// `tail` is the ONLY thing that differs between the two prompts, and the two
/// tails differ from their FIRST word, so the divergence sits exactly at the
/// tail boundary rather than smeared past it.
///
/// The load-bearing sizing requirement is not the tail's own length but what
/// follows it: the shared closing sentence after `tail` must be at least two
/// paged blocks long, so `kv_chain_upper_bound` — which breaks at the first
/// block key not on disk — is capped strictly BELOW the capture prompt's
/// deepest rung. That is what forces the restore onto a shallow rung no matter
/// how fast the writer queue drained. If it ever shrinks, assertion 1b's
/// `cached_tokens < deepest` check starts firing as a "FIXTURE problem" on a
/// healthy build. `harness_tests::the_two_ladder_prompts_diverge_at_least_two_
/// blocks_before_the_end` pins that margin offline, in milliseconds, so nobody
/// discovers it from a 20-minute GPU gate.
pub fn ladder_prompt(tail: &str) -> &'static str {
    let mut prompt =
        String::from("Answer with a single short paragraph. First, read these notes.\n\n");
    for index in 1..=15 {
        prompt.push_str(&format!(
            "Note {index}: when a block-paged key-value cache persists warm prefixes to local \
             solid-state storage, the engineer reviewing revision {index} must weigh the block \
             size, the eviction policy, the checksum cost, the durability barrier each object \
             pays, and the exact boundary at which any out-of-pool recurrent or sliding-window \
             state can be resumed soundly rather than guessed from a deeper snapshot.\n",
        ));
    }
    prompt.push('\n');
    prompt.push_str(tail);
    prompt.push_str(
        ", then say which of the two a reviewer should treat as the binding constraint on a \
         machine whose storage barrier dominates, and explain the reasoning in one sentence \
         that a reader with no cache background could follow without further context.",
    );
    Box::leak(prompt.into_boxed_str())
}

/// Ascending checkpoint boundaries a cold-start prefill of `prompt_tokens`
/// publishes, restated from
/// `qwen3_5::paged_forward::gdn_prefill_checkpoint_boundaries` with
/// `cached_prefix_len = 0`.
///
/// A COPY, not a call, for two reasons. `paged_forward` is `pub(crate)` and out
/// of reach from an integration test — and more importantly, a gate that
/// imported the constants it exists to pin would move with the bug: collapsing
/// `GDN_CHECKPOINT_LADDER_RUNGS` to 1 would collapse the expectation too and
/// the assertion would still pass.
///
/// Warm-up and capture turns that reuse a prefix publish
/// `gdn_prefill_checkpoint_boundaries(L, cached, bs)`, which starts from the
/// same deepest rung and truncates earlier (`next <= cached_prefix_len`
/// breaks) — a suffix subset of this list. So every boundary any turn on this
/// prompt could have anchored a sidecar at is a member of what this returns.
fn expected_checkpoint_ladder(prompt_tokens: u32, block_size: u32) -> Vec<u32> {
    const LADDER_RATIO: u32 = 4;
    const LADDER_RUNGS: u32 = 4;
    if block_size == 0 || prompt_tokens == 0 {
        return Vec::new();
    }
    // `gdn_checkpoint_target`: the largest full block strictly before the end
    // of the prompt, mirroring production's `prompt.len() - 1` reuse cap.
    let deepest = (prompt_tokens - 1) / block_size * block_size;
    if deepest == 0 {
        return Vec::new();
    }
    let mut rungs = Vec::with_capacity(LADDER_RUNGS as usize);
    let mut rung = deepest;
    for _ in 0..LADDER_RUNGS {
        rungs.push(rung);
        let next = rung / LADDER_RATIO / block_size * block_size;
        if next == 0 || next >= rung {
            break;
        }
        rung = next;
    }
    rungs.reverse();
    rungs
}

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
    /// Single-turn prompt run by the warm-up turns and by instance 1 (capture).
    /// Also by instances 2 and 3 unless [`Self::restore_prompt`] is set.
    pub prompt: &'static str,
    /// Prompt for the RESTORE instance (2) and the BASELINE instance (3).
    ///
    /// `None` (default) => all three instances share [`Self::prompt`], the
    /// original behaviour, bit-for-bit.
    ///
    /// `Some(p)` => instance 1 captures with `prompt`, instances 2 and 3 run
    /// `p`, and parity narrows to 2-vs-3. `p` MUST share a long token prefix
    /// with `prompt` and then diverge — see [`ladder_restore_prompt`]. The
    /// shared part is what `kv_chain_upper_bound` can still match on disk, and
    /// the divergence is a hard ceiling on how deep the restore may reconcile,
    /// so the restore CANNOT land on the capture prompt's deepest rung however
    /// far the writer queue got. That is what assertion 1b turns into a claim
    /// about the ladder rather than about one boundary.
    pub restore_prompt: Option<&'static str>,
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
    /// In `restore_prompt` mode the chain only has to reach a SHALLOW rung, not
    /// the prompt's own endpoint, so fewer turns are needed here, not more.
    /// Warm-up turn N's logged `cached_tokens` is the rung turn N-1 anchored
    /// at, which is the fastest way to tell a mis-sized fixture apart from a
    /// real failure — see the diagnostics printed by [`run_restart_parity`].
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
    /// Require the restart instance to INSTALL the sidecar it restored, not
    /// merely find, read, checksum and layout-validate it.
    ///
    /// Off by default so families with no auxiliary state are bit-for-bit
    /// unchanged. Every family whose `ColdTierContext` carries a
    /// `ColdSidecarPolicy` should set it, because without it their gate's whole
    /// claim is unbacked: `install_*_cold_sidecar` has a long tail of
    /// `Ok(false)` / `Ok(None)` arms, each falling through to a full O(prefix)
    /// replay that reconstructs CORRECT state. Assertions 1, 1b, 2 and 3 are all
    /// satisfied by that fall-through — the restore walk already counted the
    /// hits and set `cached_tokens` before the install ran, and the replay's
    /// text matches. The counter is the only thing that moves.
    pub expect_sidecar_install: bool,
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
            restore_prompt: None,
            max_new_tokens: 32,
            thinking_token_budget: Some(32),
            min_restored_tokens: None,
            capture_warmup_turns: 0,
            inspect_restore: None,
            expect_sidecar_install: false,
        }
    }

    /// See [`Self::expect_sidecar_install`].
    pub fn expecting_sidecar_install(mut self) -> Self {
        self.expect_sidecar_install = true;
        self
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

    /// See [`Self::restore_prompt`]. [`Self::prompt`] stays the capture prompt.
    pub fn with_restore_prompt(mut self, restore_prompt: &'static str) -> Self {
        self.restore_prompt = Some(restore_prompt);
        self
    }

    /// The prompt instances 2 and 3 run.
    fn measured_prompt(&self) -> &'static str {
        self.restore_prompt.unwrap_or(self.prompt)
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

/// Resolve the source model from `spec.model_path_env`.
///
/// The old shape returned `None` for two situations that are not the same
/// thing, and only one of them is a legitimate skip:
///
/// * **the var is UNSET** — the caller has no checkpoint. `#[ignore]` is already
///   the opt-in gate, so a plain `cargo test --ignored` on a machine with no
///   weights still passes. This is the honest skip and it is kept.
/// * **the var is SET but points nowhere** — nobody sets a checkpoint path they
///   do not mean. This is a typo in an invocation that believed it was gating
///   the cold tier; skipping prints one line into a long log, exits 0, and
///   reports a pass having asserted NOTHING. Panic instead.
///
/// The direction has in-repo precedent: `gate_inference_trace_path` in
/// `gemma4_cold_tier_parity.rs` refuses to run rather than assert nothing.
/// `is_dir` rather than `exists` — a plain file at that path is the same class
/// of typo, and would fail much later with an unrelated message.
fn resolve_source_model(spec: &ColdTierParitySpec) -> Option<PathBuf> {
    let env = spec.model_path_env;
    let Ok(model_path) = std::env::var(env) else {
        eprintln!(
            "skipping {}: {env} unset (point it at a real {} checkpoint)",
            spec.family, spec.family
        );
        return None;
    };
    let trimmed = model_path.trim();
    // `VAR=` is the shell's way of saying "no value", not "this path" — a
    // command substitution that produced nothing, or a CI `env:` fed by an empty
    // expression. Panicking here would tell the operator to fix a path that was
    // never given. It joins the unset skip.
    if trimmed.is_empty() {
        eprintln!(
            "skipping {}: {env} set but empty (point it at a real {} checkpoint)",
            spec.family, spec.family
        );
        return None;
    }
    let p = PathBuf::from(trimmed);
    assert!(
        p.is_dir(),
        "[{}] {env}={:?} is not a readable checkpoint directory. This gate was invoked WITH a \
         checkpoint path, so skipping here would report a pass having asserted nothing about the \
         cold tier. Fix the path, or unset {env} if the skip is what you meant.",
        spec.family,
        trimmed
    );
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
        // `zip` stops at the shorter string, so a `None` here does NOT mean
        // "equal" — it means one output is a strict prefix of the other and the
        // two stopped at different points. Named explicitly, because the
        // offset-based triage below says nothing about that case.
        let first_diff = a
            .as_bytes()
            .iter()
            .zip(b.as_bytes().iter())
            .position(|(x, y)| x != y);
        let shape = match first_diff {
            Some(0) => "byte 0",
            Some(_) => "deep in the stream",
            None => {
                "NO DIFFERING BYTE — one output is a strict PREFIX of the other, so they \
                     agreed on every token they both produced and one simply stopped earlier. \
                     Read that as a stop-condition difference (finish_reason / token budget / \
                     an EOS at a near-tie), not as either bullet below; compare the \
                     num_tokens and finish= lines printed above"
            }
        };
        panic!(
            "TEXT MISMATCH {label_a} vs {label_b}. first_diff_byte={first_diff:?} \
             ({} B vs {} B) => {shape}\n\
             Triage by that offset before touching anything:\n\
             \x20 byte 0 => a RESTORE FAULT. The two runs disagreed on their very first \
             sampled token, so the reused prefix itself is wrong — wrong positions, dropped or \
             duplicated K/V blocks, or an out-of-pool state seated at the wrong boundary.\n\
             \x20 deep in the stream => FLOAT DRIFT is the LIKELY cause, but this is a \
             heuristic, not a proof: a restore that seats a slightly wrong out-of-pool state \
             also diverges late. A persist-on turn splits its prefill at every checkpoint rung \
             it crosses while a persist-off turn runs one shot; the chunk length IS the GEMM's \
             M, so kernel selection and the reduction order change. That is algebraically \
             transparent but not bit-identical, and it lands as an argmax flip at a near-tie \
             late in the decode. CONFIRM it by re-running with persistence off on BOTH \
             instances: if they still differ, the split was never the cause.\n\
             {label_a} = {a:?}\n\
             {label_b} = {b:?}",
            a.len(),
            b.len()
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

/// Difference between two [`ColdSidecarTelemetry`] snapshots. Saturating: the
/// counters are process-global and monotonic, but a wrap would be a diagnostic
/// line, never a reason to abort a real run.
fn sidecar_telemetry_delta(
    before: &ColdSidecarTelemetry,
    after: &ColdSidecarTelemetry,
) -> ColdSidecarTelemetry {
    ColdSidecarTelemetry {
        capture_reached: after.capture_reached.saturating_sub(before.capture_reached),
        chain_empty: after.chain_empty.saturating_sub(before.chain_empty),
        boundary_skips: after.boundary_skips.saturating_sub(before.boundary_skips),
        already_persisted: after
            .already_persisted
            .saturating_sub(before.already_persisted),
        enqueued: after.enqueued.saturating_sub(before.enqueued),
        queue_drops: after.queue_drops.saturating_sub(before.queue_drops),
        installed: after.installed.saturating_sub(before.installed),
        restore_suppressed: after
            .restore_suppressed
            .saturating_sub(before.restore_suppressed),
    }
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

/// Fail loudly when a SINGLE capture turn can reach the prompt's END.
///
/// This is the invariant the whole `restore_prompt` fixture rests on and the
/// one that silently rotted: a turn anchors ONE sidecar, at the deepest rung it
/// reached, so a turn that covers the full prompt writes only the deepest rung
/// — the one a diverged restore prompt can never name — and instance 2 restores
/// zero. That surfaces as "restore did not engage", pointing the reader at the
/// restore path, when the cause is that nothing shallow was ever written.
///
/// Checked against the LADDER rather than the raw block count because the
/// deepest rung, not the prompt's end, is what a restore can address.
///
/// The bound is on ONE turn, not on all of them together — see the comment in
/// the body for why the cumulative form rejected working fixtures.
fn assert_capture_reach_leaves_room(spec: &ColdTierParitySpec, ladder: &[u32], prompt_tokens: u32) {
    let Some(&deepest) = ladder.last() else {
        return;
    };
    // The FIRST turn's reach, not the cumulative reach of all of them.
    //
    // Reach ratchets across turns — `paged_kv_cache_adapter.rs` charges the
    // budget only for blocks it actually enqueues (`outcome.enqueued`), while
    // already-persisted blocks are re-walked for free — so turn N reaches
    // roughly `N x CAPTURE_BLOCKS_PER_TURN`. But each turn anchors its sidecar
    // at the deepest rung ITS OWN reach allows, under a key derived from that
    // boundary, so the turns write DISTINCT shallow rungs rather than
    // overwriting one. Turn 1 landing below `deepest` is therefore what
    // guarantees a shallow sidecar exists.
    //
    // The cumulative form this replaces was strictly stronger, and wrong in the
    // rejecting direction: with `capture_warmup_turns >= 6` it fires on a
    // fixture whose shallow rungs are on disk and restorable.
    let reach_tokens = CAPTURE_BLOCKS_PER_TURN as u32 * spec.block_size;
    assert!(
        reach_tokens < deepest,
        "[{}] FIXTURE, not a product fault: one capture turn reaches {} blocks x {} tok = {} tok, \
         which already covers this prompt's deepest ladder rung ({} tok of {} prompt tok). That \
         turn anchors its ONE sidecar at the deepest rung, so the shallow rungs instance 2 needs \
         after its prompt diverges are never written and the restore necessarily finds nothing. \
         Lengthen the capture prompt or lower CAPTURE_BLOCKS_PER_TURN — do NOT relax the restore \
         assertions below, which would leave this gate green against a real regression.",
        spec.family,
        CAPTURE_BLOCKS_PER_TURN,
        spec.block_size,
        reach_tokens,
        deepest,
        prompt_tokens
    );
}

/// Blocks one capture turn may persist, forced by [`prepare_cold_root`].
///
/// The fixture needs the chain to reach its frontier over SEVERAL turns: a turn
/// anchors exactly one sidecar, at the deepest rung it reached, so a turn that
/// walks the whole prompt writes only the deepest rung and leaves the ladder's
/// shallow rungs empty — which is precisely what instance 2 needs after its
/// prompt diverges. That used to happen for free, because the walk stopped
/// wherever the bounded writer queue refused (~12 blocks here). It is now a
/// policy defaulting to 128 blocks, so on any machine whose disk keeps up, one
/// turn covers a fixture-sized prompt and the gate fails by construction.
///
/// Pinned rather than left to the default so the stop condition is ARITHMETIC.
/// With this below the prompt's block count the walk always stops on `Budget`,
/// so no disk or CPU speed can change which rung gets written. [`LADDER_RATIO`]
/// governs how far apart the rungs are; this only has to be shallow enough that
/// `(warm-up turns + 1) x this` stays under the deepest rung, which
/// [`assert_capture_reach_leaves_room`] enforces.
const CAPTURE_BLOCKS_PER_TURN: usize = 12;

/// Wall-clock ceiling forced alongside [`CAPTURE_BLOCKS_PER_TURN`].
///
/// NOT a lengthened timeout hiding a race: with the block budget below the
/// prompt's block count the walk stops on blocks every time, so this exists
/// only to take the clock OUT of the outcome. A walk that would trip a 60 s
/// deadline is a real hang, not a slow runner.
const CAPTURE_BUDGET_MS: u64 = 60_000;

/// Fix the tier root and the capture budget BEFORE any model load. Both are
/// process-global `OnceLock`s resolved on first use, so a later change is
/// silently ignored.
///
/// Set through `install_*` rather than `std::env::set_var`. Every caller is an
/// `#[tokio::test(flavor = "multi_thread", worker_threads = 4)]` and the body
/// runs ON a pool worker, so at least three other threads are alive here;
/// `--test-threads=1` serializes test CASES, not runtime threads, so it never
/// made `setenv` safe. The installers write the `OnceLock` directly, which is
/// thread-safe, and they RETURN whether they won — turning "the harness thinks
/// it pinned a depth but the run used the default" from a silent wrong-green
/// into an assertion.
fn prepare_cold_root() -> ColdRoot {
    assert!(
        mlx_core::cold_tier::install_cold_capture_budget(
            CAPTURE_BLOCKS_PER_TURN,
            std::time::Duration::from_millis(CAPTURE_BUDGET_MS),
        ),
        "the capture budget was already resolved before the harness pinned it, so this run used \
         the {}-block DEFAULT and the ladder arithmetic below is meaningless. Something loaded a \
         model or touched the cold tier before prepare_cold_root().",
        128,
    );

    let root = match std::env::var("MLX_COLD_CACHE_DIR") {
        Ok(dir) if !dir.trim().is_empty() => {
            let path = PathBuf::from(dir);
            fs::create_dir_all(&path).expect("create caller-supplied MLX_COLD_CACHE_DIR");
            ColdRoot::Inherited(path)
        }
        _ => {
            let path = std::env::temp_dir().join(format!("mlx-cold-parity-{}", std::process::id()));
            // Wipe it ONLY on the first gate in this process.
            //
            // The path is pid-scoped, so a second gate in the same binary
            // computes the SAME one — and the tier manager installed by the
            // first gate holds a DESCRIPTOR for that directory, not its name.
            // Unlinking and recreating the pathname therefore leaves the live
            // manager pointed at an unlinked directory, where every name lookup
            // is ENOENT: the second gate stores nothing, restores nothing, and
            // fails at "cold restore did not engage" while the real fault is
            // this line. Sharing the root across gates is what the module doc
            // already assumes — the cold keys are content-derived, so different
            // prompts occupy disjoint chains.
            if mlx_core::cold_tier::installed_cold_cache_root() != Some(path.as_path()) {
                let _ = fs::remove_dir_all(&path);
            }
            fs::create_dir_all(&path).expect("create cold-cache temp root");
            ColdRoot::Created(path)
        }
    };
    assert!(
        mlx_core::cold_tier::install_cold_cache_root(root.path()),
        "the cold tier was already opened before the harness pinned its root, so this run wrote \
         to the DEFAULT location instead of {}",
        root.path().display(),
    );
    root
}

/// Run the three-instance cold-tier restart-parity gate for one family.
///
/// `run_turn` loads a FRESH model from the given directory, runs exactly one
/// turn, and drops the model before returning — instance 2 only stands in for a
/// process restart if its in-memory hot cache really is empty.
///
/// Returns without asserting anything (logging a skip notice) ONLY when the
/// checkpoint env var is unset. A var that is set but points nowhere panics —
/// see [`resolve_source_model`].
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

    // Two closures rather than one, so the capture side and the measured side
    // cannot silently re-converge on a single prompt after a later edit. With
    // `restore_prompt: None` they build the same message and every existing
    // family is bit-for-bit unchanged.
    let capture_turn = |dir: &PathBuf| {
        run_turn(
            dir.clone(),
            vec![user_message(spec.prompt)],
            parity_chat_config(&spec),
        )
    };
    let measured_turn = |dir: &PathBuf| {
        run_turn(
            dir.clone(),
            vec![user_message(spec.measured_prompt())],
            parity_chat_config(&spec),
        )
    };

    // Chain warm-up (opt-in; see `capture_warmup_turns`). Each turn advances
    // the persisted chain's frontier by another writer-queue's worth of
    // blocks, because blocks already on disk are skipped without re-enqueueing.
    // Nothing here is asserted on — this only deepens what instance 2 can find.
    for turn_index in 0..spec.capture_warmup_turns {
        let result = capture_turn(&persist_dir).await.unwrap_or_else(|e| {
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
            "[{}] warm-up turn {}/{}: cached={} (= the rung the PREVIOUS turn anchored at; \
             not asserted). If this already equals the capture prompt's deepest rung, one turn \
             covered the whole prompt and the fixture is too short.",
            spec.family,
            turn_index + 1,
            spec.capture_warmup_turns,
            result.cached_tokens
        );
    }

    let sidecar_before = cold_sidecar_telemetry();

    // Instance 1: persistence on. Fresh prefill; captures full blocks to the
    // cold tier on turn finalize. Dropped by `run_turn` before the restart.
    let result_a = capture_turn(&persist_dir)
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
    // A SECOND sidecar window, narrower than `sidecar_before` on purpose: it
    // opens after instance 1 has finished, so its delta is instance 2's alone.
    // The install assertion needs that. Instance 1 is itself a fresh model load
    // and can install a warm-up's sidecar, so the 1+2 window would stay green
    // with the restart instance installing nothing.
    let restore_sidecar_before = cold_sidecar_telemetry();

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
    let result_b = measured_turn(&persist_dir)
        .await
        .unwrap_or_else(|e| panic!("[{}] instance 2 (restore) failed: {e}", spec.family));
    let restore_trace = inference_trace_since(trace_path.as_deref(), trace_offset);
    eprintln!(
        "[{}] instance 2 (restart, restore): num_tokens={} cached={} finish={}",
        spec.family, result_b.num_tokens, result_b.cached_tokens, result_b.finish_reason
    );

    let stats_after = cold_cache_stats_snapshot();
    let sidecar_after = cold_sidecar_telemetry();
    let restore_installs = sidecar_after
        .installed
        .saturating_sub(restore_sidecar_before.installed);

    // Instance 3: persistence off — a clean fresh-prefill baseline that never
    // touches the tier (no `ColdTierContext`).
    let result_c = measured_turn(&nopersist_dir)
        .await
        .unwrap_or_else(|e| panic!("[{}] instance 3 (no-persist) failed: {e}", spec.family));
    eprintln!(
        "[{}] instance 3 (no-persist, baseline): num_tokens={} cached={} finish={}",
        spec.family, result_c.num_tokens, result_c.cached_tokens, result_c.finish_reason
    );

    // Sidecar telemetry deltas across instance 1 + instance 2. Printed BEFORE
    // the assertions because a 3-model-load gate that fails ambiguously costs an
    // hour per iteration, and these counters separate the failure modes:
    //
    //   enqueued == 0 && already_persisted == 0 && boundary_skips > 0  =>  every
    //       turn found the shallowest published boundary out of the chain's
    //       reach and nothing was ever on disk to dedup against. That is the
    //       ladder having collapsed to its deepest rung — a real defect.
    //   already_persisted > 0  =>  a rung WAS written (by an earlier turn, very
    //       likely a warm-up, which is outside this window) and later turns
    //       re-selected it. `enqueued == 0` here is the healthy steady state,
    //       NOT a collapse — which is exactly why that arm has its own counter.
    //   enqueued >= 1 && warm-up cached == the deepest rung  =>  the writer kept
    //       up and one turn covered the whole prompt. That is the FIXTURE being
    //       too short, not a defect.
    //
    // Read without forcing the tier open, unlike `cold_cache_stats_snapshot`.
    let sidecar_delta = sidecar_telemetry_delta(&sidecar_before, &sidecar_after);
    eprintln!(
        "[{}] sidecar telemetry over instances 1+2: capture_reached={} chain_empty={} \
         boundary_skips={} already_persisted={} enqueued={} queue_drops={} installed={} \
         restore_suppressed={}",
        spec.family,
        sidecar_delta.capture_reached,
        sidecar_delta.chain_empty,
        sidecar_delta.boundary_skips,
        sidecar_delta.already_persisted,
        sidecar_delta.enqueued,
        sidecar_delta.queue_drops,
        sidecar_delta.installed,
        sidecar_delta.restore_suppressed
    );

    // ---- 1b-diagnostic. The ladder, printed BEFORE any assertion ----------
    // `restore_prompt` mode only; inert for every family that does not set it.
    //
    // Deliberately ahead of assertion 1. The headline mutation this gate exists
    // to kill — a checkpoint ladder collapsed to a single endpoint rung — does
    // NOT surface as "the restore anchored on the wrong rung". It surfaces as
    // NOTHING being written at all: the one published boundary sits at the
    // prompt's end, tens of blocks past what a bounded writer queue drains in a
    // few turns, so every turn skips and instance 2 restores zero. That trips
    // assertion 1, whose message is about the restore path. Printing the ladder
    // first is what stops the next reader from starting the hunt in the wrong
    // subsystem.
    //
    // `prompt_tokens` on a sync turn is the FULL prompt length including any
    // reused prefix (`engine/paged_turn.rs`), so instance 1 reports the capture
    // prompt's length and the ladder that prompt published is computable here.
    let ladder = if spec.restore_prompt.is_some() {
        let ladder = expected_checkpoint_ladder(result_a.prompt_tokens, spec.block_size);
        eprintln!(
            "[{}] ladder gate: capture prompt {} tok, restore prompt {} tok, block_size {}, \
             expected ladder {:?}, restore anchored at {}",
            spec.family,
            result_a.prompt_tokens,
            result_b.prompt_tokens,
            spec.block_size,
            ladder,
            result_b.cached_tokens
        );
        assert_capture_reach_leaves_room(&spec, &ladder, result_a.prompt_tokens);
        ladder
    } else {
        Vec::new()
    };

    // ---- 0. The WRITE path worked ----------------------------------------
    //
    // Ordered ahead of every restore assertion deliberately. A capture that
    // could not store what it enqueued makes each read assertion below
    // downstream noise, and the read ones have the longer, more specific
    // messages — so whichever fires first is where the next reader starts
    // looking. Proven, not assumed: a torn tier root produced exactly this run,
    //
    //     [gemma4-sub-window] cold restore did not engage across restart:
    //     cached_tokens=0 (expected >= 32)
    //
    // which sent the reader to the restore path for a fault in `prepare_cold_root`.
    //
    // A torn root is the usual cause: the directory unlinked out from under the
    // live manager, which retains a descriptor rather than a pathname, so every
    // `openat`/`renameat`/`unlinkat` returns ENOENT while `fsync` and `getdents`
    // still succeed — the cache stores nothing and reads nothing, quietly.
    //
    // A missing snapshot is NOT silently skipped here: assertion 2 below panics
    // on it explicitly.
    if let Some(stats) = stats_after.as_ref() {
        assert_eq!(
            stats.write_errors, 0,
            "[{}] cold tier recorded {} write error(s): the capture path could not store what it \
             enqueued, so everything below about restore is downstream of a broken WRITE. The \
             usual cause is a tier root that no longer exists at the descriptor the manager \
             holds — check that nothing deleted or recreated it between gates.",
            spec.family, stats.write_errors
        );
    }

    // ---- 1. Restore engaged at all ---------------------------------------
    let min_restored = spec.min_restored();
    assert!(
        result_b.cached_tokens >= min_restored,
        "[{}] cold restore did not engage across restart: cached_tokens={} (expected >= {}).\n\
         In restore_prompt mode read the `ladder gate:` and `sidecar telemetry` lines above \
         FIRST — cached_tokens=0 with expected ladder {:?} and enqueued=0/already_persisted=0 \
         means no sidecar was ever WRITTEN, which is the checkpoint ladder having collapsed to \
         its deepest rung (no shallow rung within the chain's reach), not a fault in the restore \
         path. A cached_tokens that is small but nonzero and is a member of that ladder is a \
         legitimate shallow-rung restore that merely fell below this floor; lower \
         `min_restored_tokens` rather than chasing the restore walk.",
        spec.family,
        result_b.cached_tokens,
        min_restored,
        ladder
    );

    // ---- 1b. The restore anchored on a SHALLOW ladder rung ----------------
    //
    // Instance 2's prompt diverges from the capture prompt well before that
    // prompt's endpoint, so `kv_chain_upper_bound` — which breaks at the first
    // block key not on disk — cannot reach the deepest rung. Landing on a
    // shallower one is the ONLY way it can restore at all, which is exactly
    // what an implementation that publishes a single endpoint boundary removes.
    if spec.restore_prompt.is_some() {
        let deepest = ladder.last().copied().unwrap_or(0);
        assert!(
            ladder.len() >= 2,
            "[{}] FIXTURE TOO SHORT: a {}-token capture prompt publishes ladder {:?}. With \
             fewer than two rungs this gate structurally cannot tell a ladder apart from a \
             single endpoint boundary, so it would pass either way. Lengthen the capture prompt.",
            spec.family,
            result_a.prompt_tokens,
            ladder
        );
        assert!(
            ladder.contains(&result_b.cached_tokens),
            "[{}] the restore anchored at {} tokens, which is NOT a rung of the capture \
             prompt's checkpoint ladder {:?}. Two causes, and the telemetry line above \
             separates them: with enqueued=0 AND already_persisted=0 the prefill never \
             published the shallow rungs at all (fix the checkpoint ladder, not the test); \
             with enqueued>0 or already_persisted>0 a sidecar was written and the two fixture \
             prompts do not share the token prefix this gate assumes (fix the fixture).",
            spec.family,
            result_b.cached_tokens,
            ladder
        );
        assert!(
            result_b.cached_tokens < deepest,
            "[{}] the restore anchored at {} = the capture prompt's DEEPEST rung, so this run \
             proves nothing about the shallower ones and the gate is back to being blind. \
             This is a FIXTURE problem, not a code one: either one turn's persisted chain \
             already covered the whole capture prompt (lengthen it — check the warm-up \
             `cached=` lines above, which show the rung each turn anchored at), or the two \
             prompts diverge above {} tokens (move the divergence earlier in the shared body).",
            spec.family,
            result_b.cached_tokens,
            deepest
        );
    }

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
    // `write_errors` and `restore_declines` are printed with the rest because
    // they are the two that explain an otherwise contradictory row: a restart
    // with `hits 0 / misses 0` is a REFUSED restore when `restore_declines`
    // moved, and a "successful" capture that stored nothing is a broken cache
    // root when `write_errors` moved. Neither had a number before.
    eprintln!(
        "[{}] cold stats after restart: hits={} misses={} enqueued={} queue_drops={} \
         bytes_written={} bytes_restored={} evictions={} corruptions={} write_errors={} \
         restore_declines={}",
        spec.family,
        after.hits,
        after.misses,
        after.enqueued,
        after.queue_drops,
        after.bytes_written,
        after.bytes_restored,
        after.evictions,
        after.corruptions,
        after.write_errors,
        after.restore_declines
    );

    // ---- 2a. The restored sidecar was INSTALLED, not just read -----------
    //
    // Everything above this line is satisfied by a restart that finds the
    // sidecar on disk, checksums it, layout-validates it, and then throws it
    // away: `cached_tokens` and `hits` are set by the restore walk BEFORE the
    // install runs, and the fall-through replay reconstructs correct state, so
    // the text and `num_tokens` parity below match too. Only this counter
    // separates "restored and used" from "restored and re-derived from
    // scratch" — which is an O(prefix) GDN / sliding replay on every restart,
    // i.e. the entire point of the feature, silently gone.
    //
    // Division of labour, deliberately: the counter proves the decoded state
    // reached `self.caches`. It does NOT prove the state is CORRECT — that is
    // assertion 3's job, since stale or absent recurrent state changes the text.
    if spec.expect_sidecar_install {
        assert!(
            restore_installs >= 1,
            "[{}] the restart instance restored {} token(s) from the tier but installed NO \
             sidecar: every `install_*_cold_sidecar` arm declined and the turn fell through to \
             a full O(prefix) replay. Output is still correct, which is why nothing else here \
             fires — but the recurrent/sliding half of the restore did no work. Check the \
             group, boundary == cached_prefix_len, layout-vs-geometry and decode arms of that \
             function.",
            spec.family,
            result_b.cached_tokens
        );
    }
    eprintln!(
        "[{}] sidecar installs by the restart instance alone: {} (expected >= 1: {})",
        spec.family, restore_installs, spec.expect_sidecar_install
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
    if spec.restore_prompt.is_some() {
        // Instance 1 answered a DIFFERENT question, so its text is not
        // comparable to anything here. It is still the fixture the other two
        // instances rest on, so assert it produced a real turn rather than
        // leaving it wholly unchecked: a capture that emitted nothing would
        // still have prefilled and captured, so the rest of the gate could pass
        // while the fixture was quietly degenerate.
        assert!(
            result_a.num_tokens > 0,
            "[{}] instance 1 (capture) produced no tokens (finish={}); the capture fixture is \
             degenerate even though its prefill may have captured normally",
            spec.family,
            result_a.finish_reason
        );

        // The claim that matters is unchanged in force: a turn that reconciled
        // down onto a mid-ladder boundary must produce the same bytes as a
        // clean single-shot prefill of the same prompt.
        assert_text_eq(
            "instance 3 (no-persist)",
            &result_c.text,
            "instance 2 (restore)",
            &result_b.text,
        );
        assert_eq!(
            result_c.num_tokens, result_b.num_tokens,
            "[{}] num_tokens diverged: instance 3 (no-persist) = {}, instance 2 (restore) = {}",
            spec.family, result_c.num_tokens, result_b.num_tokens
        );
    } else {
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
    }

    eprintln!(
        "[{}] cold-tier restart parity PASS: cached_tokens={} hits={} corruptions=0, \
         text and num_tokens matched",
        spec.family, result_b.cached_tokens, after.hits
    );

    // Best-effort cleanup; only touches what this run created.
    let _ = fs::remove_dir_all(&persist_dir);
    let _ = fs::remove_dir_all(&nopersist_dir);

    // The tier ROOT is deliberately left in place.
    //
    // It used to be deleted here, alongside a `remove_var("MLX_COLD_CACHE_DIR")`
    // that undid the `set_var` this harness no longer does. Deleting it is
    // actively wrong now that a second gate in the same binary reuses the
    // installed tier: `ColdRoot::Created` is `temp_dir()/mlx-cold-parity-{pid}`,
    // so the next gate recreates the SAME pathname while the live manager still
    // holds a descriptor for the directory this line unlinked. `install_cold_cache_root`
    // sees a matching path and correctly reports the tier installed, but the two
    // no longer refer to the same directory.
    //
    // It is pid-scoped and under the OS temp directory, and the per-instance
    // model clones above — the actual bulk — are still removed. `--test-threads=1`
    // plus one root per process means nothing else collides with it.
}

/// Offline cover for [`expected_checkpoint_ladder`].
///
/// The gate it serves needs real weights and several model loads, so the helper
/// would otherwise be unverified code deciding whether a slow gate passes. These
/// pin the exact ladders the fixture sizing argument depends on, so a typo in
/// the restated recurrence breaks a one-microsecond test instead of quietly
/// widening the set of boundaries assertion 1b will accept.
///
/// Each family's test binary compiles this module separately (`mod
/// cold_tier_parity_harness;`), so these run once per binary. They are pure
/// arithmetic.
#[cfg(test)]
mod harness_tests {
    use super::{
        DEFAULT_BLOCK_SIZE, expected_checkpoint_ladder, ladder_capture_prompt,
        ladder_restore_prompt, prepare_cold_root,
    };

    /// A second gate in the same binary must not replace the tier directory the
    /// first gate's manager is already holding open.
    ///
    /// `ColdRoot::Created` is `temp_dir()/mlx-cold-parity-{pid}`, so both gates
    /// compute the SAME path, and `prepare_cold_root` used to `remove_dir_all`
    /// it unconditionally. The live manager holds a DESCRIPTOR, not a name: once
    /// that directory is unlinked, every `openat`/`renameat`/`unlinkat`/`statat`
    /// through it returns ENOENT (measured; `fsync` and `getdents` still work,
    /// which is why it fails quietly). The second gate then stores nothing,
    /// restores nothing, and trips "cold restore did not engage" — blaming the
    /// restore path for a setup fault two functions away.
    ///
    /// Asserted on the INODE rather than on existence, because the broken
    /// version left a directory at the same pathname; only the identity differs.
    ///
    /// Costs microseconds and needs no checkpoint, so unlike the gates it guards
    /// it runs on the ordinary `cargo test` leg — where those gates never run at
    /// all (`ci.yml` names only the qwen3 and qwen3_5 cold-tier binaries, both
    /// single-gate, so nothing in CI exercises two gates in one process).
    #[test]
    fn preparing_the_root_twice_keeps_the_same_directory() {
        use std::os::unix::fs::MetadataExt;

        let first = prepare_cold_root();
        let ino_first = std::fs::metadata(first.path())
            .expect("tier root must exist after the first prepare")
            .ino();

        // Second gate in the same process.
        let second = prepare_cold_root();
        let ino_second = std::fs::metadata(second.path())
            .expect("tier root must exist after the second prepare")
            .ino();

        assert_eq!(
            first.path(),
            second.path(),
            "both gates must resolve the same pid-scoped root, or this test is not exercising \
             the shared-root case at all"
        );
        assert_eq!(
            ino_first, ino_second,
            "the second prepare replaced the tier directory ({ino_first} -> {ino_second}): the \
             manager installed by the first gate still holds a descriptor for the unlinked one, \
             so every cache write and read in the second gate fails with ENOENT"
        );
    }

    /// The fixture's whole R-independence argument rests on the two prompts
    /// parting company far enough before the end that instance 2's chain
    /// ceiling lands below the capture prompt's deepest rung. That margin is
    /// currently ~300 characters of shared closing sentence, which is
    /// incidental to how the prose was written — nothing enforced it. A
    /// copy-edit that shortened the closing sentence, or a `block_size` raised
    /// to 64, would silently turn assertion 1b into a flake on a healthy build.
    ///
    /// Bounded in characters because the harness has no tokenizer. Four
    /// characters per token is conservative for English prose (real ratios run
    /// 3.5-4.5), so this is a floor on the token margin, not an estimate.
    #[test]
    fn the_two_ladder_prompts_diverge_at_least_two_blocks_before_the_end() {
        let capture = ladder_capture_prompt();
        let restore = ladder_restore_prompt();
        assert_ne!(capture, restore);

        let shared = capture
            .as_bytes()
            .iter()
            .zip(restore.as_bytes())
            .position(|(a, b)| a != b)
            .expect("neither prompt may be a prefix of the other");
        let capture_tail = capture.len() - shared;
        let restore_tail = restore.len() - shared;

        const CHARS_PER_TOKEN: usize = 4;
        let floor = 2 * DEFAULT_BLOCK_SIZE as usize * CHARS_PER_TOKEN;
        assert!(
            capture_tail >= floor && restore_tail >= floor,
            "the prompts diverge at byte {shared} with only {capture_tail}/{restore_tail} \
             bytes left. Under {floor} bytes (~2 blocks) the restore's chain ceiling can reach \
             the capture prompt's deepest rung and the shallow-rung claim evaporates. Lengthen \
             the shared closing sentence in `ladder_prompt`."
        );
    }

    /// The sizing the qwen3_5 / MoE fixtures are built around, and the reason
    /// the gate needs a long prompt at all.
    ///
    /// The literals are shared with
    /// `qwen3_5::paged_forward::gdn_checkpoint_tests::ladder_rungs_are_quarters_of_the_one_above`,
    /// which asserts the SAME numbers against the production function. That
    /// pairing is what keeps this restated copy honest: the copy exists so a
    /// collapsed ladder cannot collapse the expectation with it, but a copy
    /// nothing cross-checks is just a second place for the recurrence to be
    /// wrong. Change one of these two lists and the other must be changed by
    /// hand, deliberately.
    #[test]
    fn a_long_capture_prompt_publishes_four_well_separated_rungs() {
        assert_eq!(
            expected_checkpoint_ladder(1400, DEFAULT_BLOCK_SIZE),
            vec![16, 80, 336, 1392]
        );
        assert_eq!(
            expected_checkpoint_ladder(4096, DEFAULT_BLOCK_SIZE),
            vec![48, 240, 1008, 4080]
        );
    }

    /// Why `DEFAULT_PROMPT` cannot exercise this gate: at ~90 tokens the whole
    /// ladder is two rungs, and one turn's chain reach (~8 blocks = 128 tokens)
    /// already covers the deeper one — so the restore always anchors at the
    /// prompt's end and the shallow rung is never used. Assertion 1b's
    /// `ladder.len() >= 2` is the floor; the real requirement is that the
    /// deepest rung sit far past one turn's reach, which 90 tokens does not.
    #[test]
    fn the_default_prompt_length_is_too_short_to_prove_anything() {
        let ladder = expected_checkpoint_ladder(90, DEFAULT_BLOCK_SIZE);
        assert_eq!(ladder, vec![16, 80]);
        let deepest = *ladder.last().expect("two rungs");
        assert!(
            deepest < 8 * DEFAULT_BLOCK_SIZE,
            "a {deepest}-token deepest rung is inside one turn's ~128-token chain reach"
        );
    }

    /// The recurrence stops when the next rung would be zero after
    /// block-alignment, so short prompts degenerate rather than loop.
    #[test]
    fn short_prompts_degenerate_to_one_rung_or_none() {
        const NONE: [u32; 0] = [];
        // 63/16*16 = 48; a quarter of 48 block-aligns to 0, so the ladder stops.
        assert_eq!(expected_checkpoint_ladder(64, DEFAULT_BLOCK_SIZE), vec![48]);
        assert_eq!(expected_checkpoint_ladder(17, DEFAULT_BLOCK_SIZE), vec![16]);
        assert_eq!(expected_checkpoint_ladder(16, DEFAULT_BLOCK_SIZE), NONE);
        assert_eq!(expected_checkpoint_ladder(1, DEFAULT_BLOCK_SIZE), NONE);
        assert_eq!(expected_checkpoint_ladder(0, DEFAULT_BLOCK_SIZE), NONE);
        assert_eq!(expected_checkpoint_ladder(1400, 0), NONE);
    }

    /// Every rung is a real block boundary strictly inside the prompt — the
    /// production cap is `prompt_len - 1`, so a block-aligned prompt backs off
    /// one block rather than publishing a boundary no restore can ask for.
    #[test]
    fn every_rung_is_block_aligned_and_strictly_inside_the_prompt() {
        for tokens in [17u32, 64, 65, 256, 1024, 1400, 4096] {
            let ladder = expected_checkpoint_ladder(tokens, DEFAULT_BLOCK_SIZE);
            for rung in &ladder {
                assert_eq!(rung % DEFAULT_BLOCK_SIZE, 0, "{tokens} -> {ladder:?}");
                assert!(*rung < tokens, "{tokens} -> {ladder:?}");
            }
            assert!(ladder.windows(2).all(|w| w[0] < w[1]), "{ladder:?}");
            assert!(ladder.len() <= 4, "{ladder:?}");
        }
    }
}
