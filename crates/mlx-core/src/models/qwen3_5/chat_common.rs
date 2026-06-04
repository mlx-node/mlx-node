//! Shared chat/decode infrastructure for Qwen3.5 Dense and MoE models.
//!
//! Extracts identical boilerplate from the session entry points
//! (`chat_session_start_sync` / `chat_session_continue_sync` /
//! `chat_session_continue_tool_sync` and their `chat_stream_*` streaming
//! counterparts) across both model variants: config extraction, penalty
//! application, performance metrics, result finalization, and cache
//! management.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::OnceLock;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::model_thread::StreamTx;
use crate::sampling::{
    self, SamplingConfig, apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};
use crate::tools;

use super::layer_cache::Qwen3_5LayerCache;
use super::model::{ChatConfig, ChatResult, ChatStreamChunk};

/// Load-bearing typed error prefix used when `chat_session_continue_sync`
/// rejects an image parameter because images are changing mid-session.
///
/// Wire contract: when the Rust session-continue path detects that the
/// caller is trying to switch the active image set after a session has
/// already been initialized with different images, it returns a
/// `napi::Error` whose message begins with this prefix. The TypeScript
/// session layer pattern-matches the prefix to recognize the condition
/// and trigger an image-change restart (tearing down the old session
/// state and re-entering the `chat_session_start` path).
///
/// Because TS matches on the literal prefix, this constant MUST NOT
/// change without a coordinated update on both sides of the NAPI
/// boundary.
pub(crate) const IMAGE_CHANGE_RESTART_PREFIX: &str = "IMAGE_CHANGE_REQUIRES_SESSION_RESTART:";

// ---------------------------------------------------------------------------
// MTP runtime flag inventory
// ---------------------------------------------------------------------------
//
// Runtime knobs gating individual MTP optimizations. Boolean env flags are
// read at most once per process and cached. The truthy vocabulary is uniform:
// trim() + `1` / `true` / `on` (case-insensitive). The primary adaptive depth
// knob is surfaced through the TypeScript `ChatConfig.mtpAdaptiveDepth` field
// because it interacts with the user-set `mtpDepth` and needs per-session
// resolution.
//
// | Knob                          | Default | Opt direction |
// |-------------------------------|---------|---------------|
// | `MLX_MTP_USE_TAPE_REPLAY`     | ON      | opt-OUT       |
// | `mtpAdaptiveDepth` (TS field) | OFF*    | per-session   |
// | `MLX_MTP_ADAPTIVE_DEPTH_MODE` | throughput | opt-IN EV  |
// | `MLX_MTP_CHAINED_CYCLES`      | M5+ ON, M1–M4 OFF | gen-gated |
// | `MLX_MTP_VERIFY_ASYNC_EVAL`   | ON      | opt-OUT       |
// | `MLX_MTP_DEFER_VERIFY_HIDDEN` | ON      | opt-OUT       |
// | `MLX_MTP_HISTORY_POLICY`      | committed | opt-IN window |
// | `MLX_MTP_SPARSE_ACCEPT`       | ON      | opt-OUT       |
// | `MLX_MTP_BATCH_TARGET_ARRAYS` | ON      | opt-OUT       |
// | `MLX_MTP_TRACE_ACCEPTANCE`    | OFF     | opt-IN        |
//
// * adaptive depth is opt-in. When unset, MTP pins depth 1 because current
//   Apple Silicon measurements show depth-1 has the best deterministic
//   throughput on the bf16 MTP-head lane. If `mtpAdaptiveDepth=true`,
//   `MLX_MTP_ADAPTIVE_DEPTH_MODE=expected-value` switches from the throughput
//   state machine to the MTPLX-style intra-cycle expected-value gate. The EV
//   gate starts at `MLX_MTP_EV_BASE_DEPTH` and deepens toward `mtpDepth` per
//   the EV cost model by default (temperature-0 byte-parity safe); set
//   `MLX_MTP_EV_ALLOW_DEEPEN=0` to pin the base depth.
//
// Interaction notes:
//   - `MLX_MTP_USE_TAPE_REPLAY=0` falls back to the K+1 replay path; safe to
//     combine with all other flags.
//   - `MLX_MTP_CHAINED_CYCLES` is GPU-generation-gated: default ON on M5+
//     (arch gen >= 17), default OFF on M1–M4 (gen 13–16). Force OFF with
//     `MLX_MTP_CHAINED_CYCLES=0` (even on M5+) or ON with `=1` (even on
//     M1–M4) — see `mtp_chained_cycles_enabled()`. It is CROSS-CYCLE
//     hidden-state export: each cycle's `verify_hidden[K]` slice seeds the
//     next cycle's first MTP draft (batched into the next-cycle `async_eval`;
//     see `eval_step_with_chained_hidden` below). The chained 1-forward-per-
//     cycle shape is the canonical MTPLX/vLLM design and is T=0 correctness-
//     safe (the verify forward is ground truth; the chained seed only changes
//     acceptance RATE, never the committed tokens). On M5+ it is net-positive
//     (affine +16%, nvfp4 byte-identical to AR). On M1–M4 it helps only at
//     depth 1 and REGRESSES depth-3 acceptance (a lazy-slice eval-scheduling
//     stall), so it stays OFF there pending that fix.
//   - `MLX_MTP_VERIFY_ASYNC_EVAL=1` overlaps verify dispatch with the
//     accept loop's CPU-side graph construction; composes cleanly with
//     all other flags.

// Tape-replay rollback for GDN linear-attention.
//
// Replaces the K+1 main-model forwards on every partial-accept verify cycle.
// When ON (default), the cycle arms tape recording on the dense compiled path
// BEFORE verify; the per-step `(tape, k, g, qkv)` tensors are accumulated
// during the D+1 verify forwards; on rejection a single Metal kernel replays
// only the accepted prefix into the pre-verify snapshot state and the conv
// state is rebuilt by slicing the recorded qkv. Saves ~30% of cycle wall-time.
//
// Opt-out: `MLX_MTP_USE_TAPE_REPLAY=0` (or `false`) falls back to the K+1
// replay path. The env var is read once per process and cached.
pub(crate) fn mtp_use_tape_replay() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_USE_TAPE_REPLAY") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true, // default ON
    })
}

// Async verify-eval pipeline.
//
// Replaces the synchronous `verify_logits.eval()` at the end of
// `run_mtp_cycle_inner`'s verify step with a single batched
// `mlx::core::async_eval` over `(verify_logits, verify_hiddens)`. The
// dispatch is non-blocking, so CPU control flow continues into the
// accept loop's penalty / softmax / slice graph construction while the
// GPU is still running the verify command buffer. The first downstream
// `eval()` (the accept loop's `p_target.eval()`) then implicitly
// synchronizes. Semantic equivalent of MTPLX's `LAZY_VERIFY_LOGITS`
// (`MTPLX/mtplx/generation.py:49, 3894`) — both defer the verify-logits
// sync until the accept loop's first downstream `.eval()`.
//
// Opt-out: `MLX_MTP_VERIFY_ASYNC_EVAL=0` (or `false` / `off`) reverts
// to the synchronous `verify_logits.eval()` barrier (byte-identical
// acceptance). Default ON. The env var is read once per process and cached.
pub(crate) fn mtp_verify_async_eval() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_VERIFY_ASYNC_EVAL") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true, // default ON — overlaps verify dispatch with accept-loop graph construction
    })
}

// Defer verify hidden materialization.
//
// The T=0 sparse-accept path needs verifier logits for one batched
// argmax, but it does not need the full `[1, D+1, hidden]` tensor
// eagerly. The commit graph consumes only the accepted prefix, and the
// chained path consumes only the K-th hidden slice. Default ON to match
// MTPLX's "logits first, accepted hidden slice later" policy; opt out
// for bisecting lazy-graph scheduling issues.
pub(crate) fn mtp_defer_verify_hidden_eval() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_DEFER_VERIFY_HIDDEN") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    })
}

// MTPLX-style stochastic verify scheduling.
//
// In the T>0 sparse-accept path, target top-k distributions are the first
// consumer of verifier logits. Evaluating the full `[D+1, vocab]` logits tensor
// before that duplicates the synchronization MTPLX avoids with
// `MTPLX_DEFER_VERIFY_HIDDEN_EVAL=1`: it builds/evals the target distribution
// directly from lazy verifier logits, then materializes only the accepted hidden
// prefix later during commit/chaining.
//
// Opt-in only: on this MLX native path the lazy sparse-distribution graph can be
// more expensive than the explicit verify eval plus top-k pass. Keep it as a
// measurement knob rather than a default.
pub(crate) fn mtp_target_distribution_first_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(
        || match std::env::var("MLX_MTP_TARGET_DISTRIBUTION_FIRST") {
            Ok(v) => {
                let v = v.trim();
                v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
            }
            Err(_) => false,
        },
    )
}

// Opt-OUT gate for the paged-pool MTP verify graph. Default ON. Set
// `MLX_MTP_VERIFY_PAGED_ATTN` to `0` / `false` / `off` (case-insensitive,
// surrounding whitespace ignored) to fall back to the dense BHTD verify
// path. The Rust gate mirrors the C++ `mtp_verify_paged_attn_enabled()`
// reader in `mlx_qwen35.cpp`; both must agree per process for the
// dispatcher to route through the new graph.
pub(crate) fn mtp_verify_paged_attn_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_VERIFY_PAGED_ATTN") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    })
}

/// Minimum GPU architecture generation for chained MTP cycles to default ON.
/// M5+ (gen >= 17): chained is measured net-positive (affine +16%, nvfp4 byte-
/// identical to AR). On M1–M4 (gen 13–16) a lazy-slice eval-scheduling stall makes
/// chained regress depth-3 acceptance, so it defaults OFF there pending that fix.
/// Override either way with MLX_MTP_CHAINED_CYCLES=0/1.
const CHAINED_CYCLES_MIN_GPU_GEN: i32 = 17;

// Chained cycles via verify-hidden export.
//
// Once MTP caches use committed-history and the verifier exports
// `verify_hidden[K]`, chaining avoids paying the Step-A target forward at the
// start of every speculative cycle. That hidden slice is fused into the same
// `async_eval` batch as `(token, g_compiled_caches)` at end-of-iteration (see
// `eval_step_with_chained_hidden` in `MtpOps`) so the slice becomes a sibling
// of the next-cycle draft's first inputs rather than a late dependency
// materialized inside the draft graph build.
//
// Default ON on M5+ (GPU arch gen >= 17), where chaining is measured
// net-positive (affine +16%, nvfp4 byte-identical to AR). Default OFF on M1–M4
// (gen 13–16), where a lazy-slice eval-scheduling stall makes chained regress
// depth-3 acceptance — pending that fix.
//
// Override either direction with the env var: explicit `0` / `false` / `off`
// forces OFF even on M5+; explicit `1` / `true` / `on` forces ON even on M1–M4
// (e.g. for parity bisects).
//
// The env var (and the GPU-gen fallback) is read once per process and cached.
pub(crate) fn mtp_chained_cycles_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_CHAINED_CYCLES") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => {
            let gpu_gen = unsafe { mlx_sys::mlx_gpu_architecture_gen() };
            gpu_gen >= CHAINED_CYCLES_MIN_GPU_GEN
        }
    })
}

// Prompt-prefix MTP prefill opt-OUT.
//
// `MLX_MTP_NO_PROMPT_PREFILL=1` (or `true` / `on`) disables committing
// the prompt prefix into the MTP committed-history cache: the prefill
// stays logits-only and the MTP heads build history only from
// decode-produced tokens (the pre-prompt-prefill behaviour). Default
// OFF (prompt-prefill enabled). Read once per process and cached.
pub(crate) fn mtp_no_prompt_prefill() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_NO_PROMPT_PREFILL") {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        }
        Err(_) => false, // default OFF — prompt-prefill enabled
    })
}

// MTPLX-style committed MTP history policy.
//
// The committed-history cache remains active. This policy only decides how
// much prompt-side history is seeded before decode:
//   - committed: seed the full `[prompt[1..], first_sample]` run.
//   - last_window: seed only the tail of that run and carry an absolute
//     position base so RoPE positions stay aligned with the real sequence.
//   - auto: use last_window once the prompt crosses a threshold.
//
// Decode-time appends continue from the seeded tail. This mirrors MTPLX's
// normal decode path; their window is re-applied when the serving engine
// explicitly rebases/restores prompt state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MtpHistoryPolicy {
    Committed,
    LastWindow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MtpPromptHistorySelection {
    pub policy: MtpHistoryPolicy,
    pub keep_tokens: usize,
    pub position_base: usize,
}

impl MtpPromptHistorySelection {
    pub(crate) fn hidden_start_token_index(self) -> usize {
        self.position_base
    }
}

fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|raw| {
        let raw = raw.trim();
        if raw.is_empty() {
            None
        } else {
            raw.parse::<usize>().ok()
        }
    })
}

fn normalize_mtp_history_policy(raw: &str) -> Option<&'static str> {
    match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "" => None,
        "auto" => Some("auto"),
        "committed" | "full" => Some("committed"),
        "last_window" | "lastwindow" | "window" => Some("last_window"),
        // Keep MTPLX's opt-out spelling as an alias for the existing explicit
        // `MLX_MTP_NO_PROMPT_PREFILL` escape hatch, not as a silent mode switch.
        "cycle" | "none" | "off" => Some("committed"),
        _ => None,
    }
}

pub(crate) fn resolve_mtp_prompt_history_selection(
    requested_policy: &str,
    prompt_len: usize,
    window_tokens: usize,
    threshold_tokens: usize,
) -> MtpPromptHistorySelection {
    let normalized = normalize_mtp_history_policy(requested_policy).unwrap_or("committed");
    let policy = match normalized {
        "last_window" => MtpHistoryPolicy::LastWindow,
        "auto" if prompt_len >= threshold_tokens.max(1) => MtpHistoryPolicy::LastWindow,
        _ => MtpHistoryPolicy::Committed,
    };
    match policy {
        MtpHistoryPolicy::Committed => MtpPromptHistorySelection {
            policy,
            keep_tokens: prompt_len,
            position_base: 0,
        },
        MtpHistoryPolicy::LastWindow => {
            let keep_tokens = prompt_len.min(window_tokens.max(1));
            MtpPromptHistorySelection {
                policy,
                keep_tokens,
                position_base: prompt_len.saturating_sub(keep_tokens),
            }
        }
    }
}

pub(crate) fn mtp_prompt_history_selection(prompt_len: usize) -> MtpPromptHistorySelection {
    static POLICY: OnceLock<String> = OnceLock::new();
    static WINDOW: OnceLock<usize> = OnceLock::new();
    static THRESHOLD: OnceLock<usize> = OnceLock::new();

    let policy = POLICY.get_or_init(|| match std::env::var("MLX_MTP_HISTORY_POLICY") {
        Ok(raw) if normalize_mtp_history_policy(&raw).is_some() => raw,
        Ok(raw) => {
            tracing::warn!(
                target: "mlx_core::mtp",
                value = %raw,
                "Ignoring invalid MLX_MTP_HISTORY_POLICY; using committed"
            );
            "committed".to_string()
        }
        Err(_) => "committed".to_string(),
    });
    let window = *WINDOW.get_or_init(|| {
        parse_env_usize("MLX_MTP_HISTORY_LAST_WINDOW")
            .filter(|v| *v > 0)
            .unwrap_or(8192)
    });
    let threshold = *THRESHOLD.get_or_init(|| {
        parse_env_usize("MLX_MTP_HISTORY_LAST_WINDOW_THRESHOLD")
            .filter(|v| *v > 0)
            .unwrap_or(16384)
    });

    resolve_mtp_prompt_history_selection(policy, prompt_len, window, threshold)
}

// Accept-loop sync collapse via on-device sparse top-K / batched
// argmax (MTPLX-style).
//
// Replaces the legacy per-position accept loop's D forced GPU syncs
// (each materializing a full-vocab softmax of ~151k floats) with ONE
// batched on-device op over all `D+1` verify positions. On the T=0
// (greedy) path this is `argmax(verify_logits, axis=-1)` → `[1, D+1]`
// int32, evaluated once. On T>0 we keep the legacy per-position
// path (residual sampling still needs the full target distribution
// to draw from `(p_target - p_draft)+`).
//
// Eligibility (T=0 fast path):
//   - `temperature <= 1e-6` (matches `accept_with_residual`'s argmax
//     shortcut).
//   - All penalties at defaults (repetition=1.0, presence=0.0,
//     frequency=0.0). When any penalty is active, the per-position
//     `apply_all_penalties` call depends on `hist_extended` which
//     mutates inside the accept loop — we cannot precompute the
//     argmax in one shot without re-applying the penalty per
//     position.
//
// Default ON for the deterministic fast path. At T=0 with default
// penalties, acceptance only needs verifier argmax IDs, so this avoids
// D per-position full-vocab softmax materializations. Set
// `MLX_MTP_SPARSE_ACCEPT=0` / `false` / `off` to force the legacy
// per-position path for parity debugging or A/B measurements. The env
// var is read once per process and cached.
pub(crate) fn mtp_sparse_accept_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_SPARSE_ACCEPT") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    })
}

// Indirection over the sparse-accept gate so tests can drive the
// production `use_sparse_accept` commit path hermetically — independent
// of the process-wide `MLX_MTP_SPARSE_ACCEPT` env var / OnceLock cache.
// In non-test builds this is a zero-cost `#[inline]` passthrough, so the
// decode path's behavior and codegen are identical to calling
// `mtp_sparse_accept_enabled()` directly.
#[cfg(not(test))]
#[inline]
fn sparse_accept_gate() -> bool {
    mtp_sparse_accept_enabled()
}

#[cfg(test)]
thread_local! {
    /// Test-only override for [`sparse_accept_gate`]. `None` defers to the
    /// real env-backed [`mtp_sparse_accept_enabled`]; `Some(b)` forces the
    /// gate so a test deterministically exercises the intended accept path.
    static TEST_FORCE_SPARSE_ACCEPT: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn sparse_accept_gate() -> bool {
    TEST_FORCE_SPARSE_ACCEPT
        .with(std::cell::Cell::get)
        .unwrap_or_else(mtp_sparse_accept_enabled)
}

/// RAII guard that forces [`sparse_accept_gate`] for the current thread and
/// restores the prior value on drop (panic-safe). Used by the C2 T=0 safety
/// test to guarantee it drives the production sparse-accept commit path
/// regardless of `MLX_MTP_SPARSE_ACCEPT`.
#[cfg(test)]
pub(crate) struct ForceSparseAcceptGuard(Option<bool>);

#[cfg(test)]
impl ForceSparseAcceptGuard {
    pub(crate) fn force(value: bool) -> Self {
        let prev = TEST_FORCE_SPARSE_ACCEPT.with(|c| c.replace(Some(value)));
        ForceSparseAcceptGuard(prev)
    }
}

#[cfg(test)]
impl Drop for ForceSparseAcceptGuard {
    fn drop(&mut self) {
        TEST_FORCE_SPARSE_ACCEPT.with(|c| c.set(self.0));
    }
}

// MTPLX-style stochastic accept fast path.
//
// At T>0 with default penalties and a bounded top-k sampler, exact
// probability-ratio acceptance does not need dense `[vocab]` CPU copies. We
// keep `top_k` token IDs/probabilities per verifier row, copy only that tiny
// `[D+1, top_k]` table, and run accept/residual/bonus sampling on CPU. This
// mirrors MTPLX's `MTPLX_BATCH_TARGET_ARRAYS=1` path while preserving this
// runtime's compiled sampler semantics. Opt out with
// `MLX_MTP_BATCH_TARGET_ARRAYS=0` / `false` / `off`.
pub(crate) fn mtp_batch_target_arrays_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_BATCH_TARGET_ARRAYS") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    })
}

// Native stochastic verifier sparse-target output.
//
// This is an opt-out gate for the dense Qwen compiled path. When the sampler is
// in MTPLX parity mode, the verifier can return compact
// `[depth+1, top_k]` target ids/probabilities directly from the native graph
// instead of surfacing full `[1, depth+1, vocab]` logits and rebuilding the same
// sparse rows on the Rust side.
pub(crate) fn mtp_native_sparse_verify_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_NATIVE_SPARSE_VERIFY") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    })
}

// Greedy verifier output fast path.
//
// At T=0 with default penalties, accept/reject only needs target top-1 ids.
// This opt-in gate allows the dense verifier to return `[1, depth+1]` argmax
// ids plus hiddens without surfacing full `[1, depth+1, vocab]` logits.
// Diagnostics that need logits disable this path at the call site. It is
// disabled by default because the current MLX graph evaluates this form slower
// than the full-logits verifier on M5 Max.
pub(crate) fn mtp_greedy_argmax_only_verify_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(
        || match std::env::var("MLX_MTP_GREEDY_ARGMAX_ONLY_VERIFY") {
            Ok(v) => {
                let v = v.trim();
                v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
            }
            Err(_) => false,
        },
    )
}

fn parse_env_f64(name: &str) -> Option<f64> {
    std::env::var(name).ok().and_then(|raw| {
        let raw = raw.trim();
        if raw.is_empty() {
            None
        } else {
            raw.parse::<f64>().ok().filter(|v| v.is_finite())
        }
    })
}

fn parse_env_i32(name: &str) -> Option<i32> {
    std::env::var(name).ok().and_then(|raw| {
        let raw = raw.trim();
        if raw.is_empty() {
            None
        } else {
            raw.parse::<i32>().ok()
        }
    })
}

fn mtp_draft_temperature_scale() -> Option<f64> {
    static CACHE: OnceLock<Option<f64>> = OnceLock::new();
    *CACHE.get_or_init(|| parse_env_f64("MLX_MTP_DRAFT_TEMPERATURE_SCALE"))
}

fn mtp_draft_temperature_override() -> Option<f64> {
    static CACHE: OnceLock<Option<f64>> = OnceLock::new();
    *CACHE.get_or_init(|| parse_env_f64("MLX_MTP_DRAFT_TEMPERATURE"))
}

fn mtp_draft_top_p_override() -> Option<f64> {
    static CACHE: OnceLock<Option<f64>> = OnceLock::new();
    *CACHE.get_or_init(|| parse_env_f64("MLX_MTP_DRAFT_TOP_P"))
}

fn mtp_draft_top_k_override() -> Option<i32> {
    static CACHE: OnceLock<Option<i32>> = OnceLock::new();
    *CACHE.get_or_init(|| parse_env_i32("MLX_MTP_DRAFT_TOP_K"))
}

fn mtp_draft_sampling_config(
    target: crate::sampling::SamplingConfig,
) -> crate::sampling::SamplingConfig {
    let mut draft = target;
    if let Some(scale) = mtp_draft_temperature_scale()
        && scale > 0.0
    {
        draft.temperature = Some(target.temperature.unwrap_or(1.0) * scale);
    }
    if let Some(temperature) = mtp_draft_temperature_override()
        && temperature >= 0.0
    {
        draft.temperature = Some(temperature);
    }
    if let Some(top_p) = mtp_draft_top_p_override()
        && top_p >= 0.0
    {
        draft.top_p = Some(top_p);
    }
    if let Some(top_k) = mtp_draft_top_k_override()
        && top_k >= 0
    {
        draft.top_k = Some(top_k);
    }
    draft
}

// Diagnostic — per-committed-token top-2 logit trace.
//
// `MLX_MTP_TRACE_LOGITS=1` (or `true` / `on`) enables an env-gated
// per-token logit trace emitted to stderr. For each committed decode
// token it logs the position index, the committed token id, and the
// top-2 (token id + logit value) of the forward that produced it:
//   * the AR `decode_loop!` logs the single-token decode forward;
//   * `run_mtp_cycle_inner` logs the batched verify forward, per
//     verify slot.
//
// The trace exists to resolve whether an AR-vs-MTP argmax flip is a
// benign batched-vs-single kernel near-tie (both forwards have the
// SAME top-2 set with logits agreeing within bf16 epsilon) or a real
// verify-path bug (the verify forward computes a substantially
// different logit vector). Default OFF; read once per process and
// cached. Lines are prefixed `MTP_TRACE_LOGITS` for easy grep.
pub(crate) fn mtp_trace_logits() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_TRACE_LOGITS") {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        }
        Err(_) => false, // default OFF — diagnostic instrumentation
    })
}

pub(crate) fn mtp_verify_top1_check_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_VERIFY_TOP1_CHECK") {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        }
        Err(_) => false,
    })
}

pub(crate) fn mtp_trace_acceptance() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_TRACE_ACCEPTANCE") {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        }
        Err(_) => false,
    })
}

/// Top-2 entries `(id, logit)` of a logits vector — used by the
/// `MLX_MTP_TRACE_LOGITS` diagnostic.
pub(crate) struct Top2 {
    pub top1_id: i32,
    pub top1_logit: f32,
    pub top2_id: i32,
    pub top2_logit: f32,
}

/// Compute the top-2 `(id, logit)` of a 1-D logits array.
///
/// `logits_1d` must be a `[vocab]` array (any float dtype — values are
/// read back as f32). Uses a descending sort of the indices via
/// `argsort` then a single `.eval()`; the two winning logit values are
/// read by flat index from an f32 copy of the logits. No `.unwrap()` /
/// `.expect()` — every fallible step propagates with `?`, so this is
/// safe to call from the decode path.
pub(crate) fn trace_top2(logits_1d: &MxArray, vocab: i64) -> Result<Top2> {
    use crate::array::DType;

    // argsort is ascending; the last two entries are the top-2.
    let order = logits_1d.argsort(Some(-1))?;
    let logits_f32 = logits_1d.astype(DType::Float32)?;
    order.eval();
    logits_f32.eval();

    let last = (vocab - 1).max(0) as usize;
    let second = (vocab - 2).max(0) as usize;
    let top1_id = order.item_at_int32(last)?;
    let top2_id = order.item_at_int32(second)?;
    let top1_logit = logits_f32.item_at_float32(top1_id as usize)?;
    let top2_logit = logits_f32.item_at_float32(top2_id as usize)?;
    Ok(Top2 {
        top1_id,
        top1_logit,
        top2_id,
        top2_logit,
    })
}

fn trace_json_f64(value: f64) -> serde_json::Value {
    serde_json::Number::from_f64(value)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

fn trace_acceptance_emit(payload: serde_json::Value) {
    eprintln!("MTP_TRACE_ACCEPTANCE {}", payload);
}

fn trace_acceptance_greedy(
    depth: usize,
    slot: usize,
    token_history_len: usize,
    last_committed_id: u32,
    draft_id: i32,
    target_id: i32,
    accepted: bool,
    top2: Option<&Top2>,
) {
    trace_acceptance_emit(serde_json::json!({
        "schema_version": 1,
        "path": "greedy_sparse",
        "depth": depth,
        "slot": slot,
        "position": token_history_len + slot,
        "last_committed_id": last_committed_id,
        "draft_id": draft_id,
        "target_argmax": target_id,
        "target_rank": if accepted { Some(1usize) } else { None },
        "target_top1_id": top2.map(|t| t.top1_id).unwrap_or(target_id),
        "target_top1_logit": top2
            .map(|t| trace_json_f64(f64::from(t.top1_logit)))
            .unwrap_or(serde_json::Value::Null),
        "target_top2_id": top2.map(|t| t.top2_id),
        "target_top2_logit": top2
            .map(|t| trace_json_f64(f64::from(t.top2_logit)))
            .unwrap_or(serde_json::Value::Null),
        "target_logit_gap": top2
            .map(|t| trace_json_f64(f64::from(t.top1_logit - t.top2_logit)))
            .unwrap_or(serde_json::Value::Null),
        "target_prob_for_draft": if accepted { trace_json_f64(1.0) } else { trace_json_f64(0.0) },
        "draft_prob_for_draft": serde_json::Value::Null,
        "accept_prob": if accepted { trace_json_f64(1.0) } else { trace_json_f64(0.0) },
        "accepted": accepted,
        "out_token": if accepted { draft_id } else { target_id },
    }));
}

fn trace_acceptance_sparse(
    path: &'static str,
    depth: usize,
    slot: usize,
    token_history_len: usize,
    last_committed_id: u32,
    draft_id: i32,
    target_p: crate::sampling::SparseDistributionRef<'_>,
    draft_q: crate::sampling::SparseDistributionRef<'_>,
    accepted: bool,
    out_tok: i32,
) {
    let p = target_p.probability(draft_id);
    let q = draft_q.probability(draft_id);
    let accept_prob = crate::sampling::acceptance_probability_from_probs(p, q);
    let target_top = target_p.top_entry();
    let draft_top = draft_q.top_entry();

    trace_acceptance_emit(serde_json::json!({
        "schema_version": 1,
        "path": path,
        "depth": depth,
        "slot": slot,
        "position": token_history_len + slot,
        "last_committed_id": last_committed_id,
        "draft_id": draft_id,
        "target_rank": target_p.positive_rank(draft_id),
        "draft_rank": draft_q.positive_rank(draft_id),
        "target_top1_id": target_top.map(|(id, _)| id),
        "target_top1_prob": target_top
            .map(|(_, prob)| trace_json_f64(prob))
            .unwrap_or(serde_json::Value::Null),
        "draft_top1_id": draft_top.map(|(id, _)| id),
        "draft_top1_prob": draft_top
            .map(|(_, prob)| trace_json_f64(prob))
            .unwrap_or(serde_json::Value::Null),
        "target_prob_for_draft": trace_json_f64(p),
        "draft_prob_for_draft": trace_json_f64(q),
        "accept_prob": trace_json_f64(accept_prob),
        "accepted": accepted,
        "out_token": out_tok,
    }));
}

fn trace_acceptance_dense(
    depth: usize,
    slot: usize,
    token_history_len: usize,
    last_committed_id: u32,
    draft_id: i32,
    p_target: &MxArray,
    p_draft: &MxArray,
    sampling_config: &SamplingConfig,
    accepted: bool,
    out_tok: i32,
) -> Result<()> {
    use crate::array::DType;

    let p_target_f32 = p_target.astype(DType::Float32)?;
    let p_draft_f32 = p_draft.astype(DType::Float32)?;
    p_target_f32.eval();
    p_draft_f32.eval();

    let idx = draft_id as usize;
    let p = f64::from(p_target_f32.item_at_float32(idx)?);
    let q = f64::from(p_draft_f32.item_at_float32(idx)?);

    let target_argmax = p_target_f32.argmax(0, None)?;
    let draft_argmax = p_draft_f32.argmax(0, None)?;
    target_argmax.eval();
    draft_argmax.eval();
    let target_top1_id = target_argmax.item_at_int32(0)?;
    let draft_top1_id = draft_argmax.item_at_int32(0)?;
    let target_top1_prob = if target_top1_id >= 0 {
        f64::from(p_target_f32.item_at_float32(target_top1_id as usize)?)
    } else {
        0.0
    };
    let draft_top1_prob = if draft_top1_id >= 0 {
        f64::from(p_draft_f32.item_at_float32(draft_top1_id as usize)?)
    } else {
        0.0
    };

    let greedy = crate::sampling::is_greedy_temperature(sampling_config.temperature.unwrap_or(1.0));
    let accept_prob = if greedy {
        if target_top1_id == draft_id { 1.0 } else { 0.0 }
    } else {
        crate::sampling::acceptance_probability_from_probs(p, q)
    };

    trace_acceptance_emit(serde_json::json!({
        "schema_version": 1,
        "path": "legacy_dense",
        "depth": depth,
        "slot": slot,
        "position": token_history_len + slot,
        "last_committed_id": last_committed_id,
        "draft_id": draft_id,
        "target_argmax": target_top1_id,
        "draft_argmax": draft_top1_id,
        "target_rank": if target_top1_id == draft_id { Some(1usize) } else { None },
        "draft_rank": if draft_top1_id == draft_id { Some(1usize) } else { None },
        "target_top1_id": target_top1_id,
        "target_top1_prob": trace_json_f64(target_top1_prob),
        "draft_top1_id": draft_top1_id,
        "draft_top1_prob": trace_json_f64(draft_top1_prob),
        "target_prob_for_draft": trace_json_f64(p),
        "draft_prob_for_draft": trace_json_f64(q),
        "accept_prob": trace_json_f64(accept_prob),
        "accepted": accepted,
        "out_token": out_tok,
    }));

    Ok(())
}

/// Hash raw image bytes to a u64 key for cache lookup.
fn hash_image_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// Combine individual image hashes into a single cache key.
/// Order matters: different orderings of the same images produce different keys.
fn combine_image_hashes(hashes: &[u64]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for h in hashes {
        h.hash(&mut hasher);
    }
    hasher.finish()
}

/// Compute a combined cache key from raw image bytes.
pub(crate) fn compute_image_cache_key(all_images: &[Vec<u8>]) -> u64 {
    let individual_hashes: Vec<u64> = all_images.iter().map(|img| hash_image_bytes(img)).collect();
    combine_image_hashes(&individual_hashes)
}

/// Build per-block extra_keys for the paged adapter's prefix-cache walk.
///
/// Multimodal cache isolation: when the prompt contains image tokens,
/// the per-block extra_keys ensure that "same prompt + different image"
/// produces a cache miss (preventing stale-image KV reuse). For
/// text-only prompts (`token_image_positions` is empty), every block gets
/// an empty extra_keys vec — bit-equal to passing `&[]` uniformly to the
/// uniform `find_cached_prefix` / `finalize_turn_keep_live` API.
///
/// `total_tokens` is the FULL prompt length (cached prefix + new suffix
/// the request will write). The number of full blocks covered is
/// `total_tokens / block_size`; the trailing partial block (if any) is
/// not registered until full and so gets no entry here.
///
/// `token_image_positions` should be sorted by `token_pos` for stable
/// hashes (the helper preserves input order; reordered inputs would
/// produce different hashes). Today's Qwen3.5 paged dispatch is text-only
/// (image-bearing turns are routed to the flat path), so the production
/// call always passes `&[]` here. The hook stays in place so that when
/// VLM-paged forward integration lands, the call site only needs to swap
/// in the real image positions.
pub(crate) fn build_paged_extra_keys(
    total_tokens: usize,
    block_size: u32,
    token_image_positions: &[(u32, u64)],
) -> Vec<Vec<u64>> {
    let block_size_us = block_size as usize;
    if block_size_us == 0 {
        return Vec::new();
    }
    // Cover every block the request might register (full blocks only).
    // The adapter's per-block API tolerates an over-long vec by indexing
    // only what it needs, so erring high is safe.
    let num_blocks = total_tokens.div_ceil(block_size_us);
    crate::transformer::paged_kv_cache_adapter::compute_per_block_image_extra_keys(
        token_image_positions,
        num_blocks,
        block_size,
    )
}

/// Report a guard-violation error through the stream channel.
///
/// Used by the streaming session entry points (`chat_stream_session_*`
/// and `chat_stream_tokens_delta_sync`) to surface pre-decode guard
/// failures — text-only violations, missing tokenizer special tokens,
/// reuse_cache=false, empty delta, etc.
///
/// Sends an `Err(napi::Error::from_reason(message))` item into the
/// mpsc so the NAPI forwarding task invokes the TS callback with
/// `(err, null)`. On the TS side, `_runChatStream` pushes the error
/// onto its queue and throws it from the async generator, which
/// `ChatSession.sendStream` catches in its `try { ... } finally`
/// block. The finally clears `inFlight`, `sawFinal` stays false, and
/// `turnCount` is NOT incremented — so the next `sendStream()` call
/// re-routes through `chatStreamSessionStart` instead of trying to
/// continue a session that never initialized. The exception also
/// re-throws to the caller so the failure is observable.
///
/// Important: historically this helper emitted a fake `done: true`
/// `ChatStreamChunk` with `finish_reason: "error"`, which the TS side
/// treated as a successful final chunk and caused the session to
/// advance to a bricked turn 1. Do NOT reintroduce that pattern —
/// guard failures MUST come through as `Err` so the error path is
/// exercised.
pub(crate) fn send_stream_error(stream_tx: &StreamTx<ChatStreamChunk>, message: &str) {
    let _ = stream_tx.send(Err(napi::Error::from_reason(message.to_string())));
}

/// Build a synthetic `ChatMessage` wrapping a user-role text-only message.
///
/// Used by the session-continue paths to feed a single user turn through
/// `Qwen3Tokenizer::sanitize_messages_public` without leaking any of the
/// extended optional fields (tool calls, images, etc.) that a real client
/// request might carry. Those fields are deliberately set to `None` so
/// the sanitization pass only has to police the textual `content` field.
pub(crate) fn build_synthetic_user_message(user: &str) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: user.to_string(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        images: None,
    }
}

/// Build the ChatML wire-format delta text for a session-continue turn.
///
/// The cached history ends on `<|im_end|>` (because `chat_session_start_sync`
/// uses `im_end_id` as eos). The leading `\n` closes that turn's line; then
/// we open a new user turn and prime an assistant turn.
///
/// When thinking mode is explicitly enabled (`reasoning_effort ∈ {"medium",
/// "high"}`) or left as default, the Qwen3.5 jinja template inserts
/// `<think>\n` after the assistant prelude — mirror that here so the delta
/// stays template-equivalent. When thinking is explicitly disabled
/// (`Some(false)`), omit the prefix so the first generated token is a
/// plain content token.
///
/// `sanitized_user` MUST already be passed through
/// `Qwen3Tokenizer::sanitize_messages_public` by the caller — this helper
/// does not re-sanitize.
pub(crate) fn build_chatml_continue_delta_text(
    sanitized_user: &str,
    enable_thinking: Option<bool>,
) -> String {
    let thinking_prefix = match enable_thinking {
        Some(false) => "",
        // None = template default (Qwen3.5: thinking on) and
        // Some(true) both take the thinking path.
        _ => "<think>\n",
    };
    format!(
        "\n<|im_start|>user\n{sanitized_user}<|im_end|>\n<|im_start|>assistant\n{thinking_prefix}",
    )
}

/// Build the ChatML wire-format delta text for a tool-result turn.
///
/// Qwen3.5's chat template renders tool-role messages as a `user` turn
/// wrapping the tool result in `<tool_response>` tags:
///
/// ```text
/// <|im_start|>user
/// <tool_response>
/// {content}
/// </tool_response><|im_end|>
/// ```
///
/// The `tool_call_id` is NOT rendered anywhere by the template — Qwen
/// identifies tool responses purely by position and wrapper tags, so we
/// intentionally drop it here. Callers may still log it for their own
/// bookkeeping, but it does not enter the wire format.
///
/// Like `build_chatml_continue_delta_text`, this helper assumes the cached
/// history ends on `<|im_end|>` and emits a leading `\n` to close that
/// turn's line. After the tool response we open an assistant turn ready
/// for the next generation step.
///
/// Thinking-prefix handling mirrors `build_chatml_continue_delta_text`:
/// when thinking mode is explicitly disabled (`Some(false)`), omit the
/// `<think>\n` prefix so the first generated token is a plain content
/// token. Otherwise (`None` / `Some(true)`) emit the `<think>\n` prefix,
/// matching what the Qwen3.5 jinja template does after the assistant
/// opener. Callers resolve `enable_thinking` from the current
/// `ChatConfig` via `resolve_enable_thinking` before calling this helper.
///
/// `is_error` is the model-facing failure cue: when `Some(true)`, the
/// shared [`crate::tokenizer::TOOL_ERROR_MARKER`] is prepended to
/// `content` inside the `<tool_response>` wrapper. The structured
/// `ChatMessage::is_error` field on the originating message is the
/// authoritative signal; the marker injection here only affects the
/// wire bytes the model decodes. `None` / `Some(false)` produce the
/// unmarked wire format and stay byte-equal to the pre-feature output.
pub(crate) fn build_chatml_tool_delta_text(
    _tool_call_id: &str,
    content: &str,
    enable_thinking: Option<bool>,
    is_error: Option<bool>,
) -> String {
    let thinking_prefix = match enable_thinking {
        Some(false) => "",
        // None = template default (Qwen3.5: thinking on) and
        // Some(true) both take the thinking path.
        _ => "<think>\n",
    };
    let rendered_content = crate::tokenizer::apply_tool_error_marker(content, is_error);
    format!(
        "\n<|im_start|>user\n<tool_response>\n{rendered_content}\n</tool_response><|im_end|>\n<|im_start|>assistant\n{thinking_prefix}",
    )
}

/// Extracted chat parameters with defaults applied.
pub(crate) struct ChatParams {
    pub max_new_tokens: i32,
    pub repetition_penalty: f64,
    pub repetition_context_size: i32,
    pub presence_penalty: f64,
    pub presence_context_size: i32,
    pub frequency_penalty: f64,
    pub frequency_context_size: i32,
    pub max_consecutive_tokens: i32,
    pub max_ngram_repeats: i32,
    pub ngram_size: i32,
    pub sampling_config: Option<SamplingConfig>,
    pub report_performance: bool,
    pub reuse_cache: bool,
    pub thinking_token_budget: Option<i32>,
    pub include_reasoning: bool,
    /// MTP: opt-in flag enabling the Multi-Token Prediction speculative
    /// decode loop. Effective only on the dense compiled path AND when
    /// the model checkpoint carries an MTP head
    /// (`Qwen35Inner::has_mtp_weights`). The eager Rust forward, the
    /// paged path, MoE, and VLM decode loops all continue to use the
    /// single-token `decode_loop!` macro regardless. Default: `false`.
    pub enable_mtp: bool,
    /// MTP: number of draft tokens per speculative cycle, fed to the
    /// `forward_mtp_draft_compiled` / `forward_mtp_verify_compiled` FFI.
    /// Must be in `[1, 5]` to satisfy the verify-FFI contract. Default:
    /// 1 on the current bf16 MTP-head lane. When
    /// `mtp_adaptive_depth = true`, this value is only used as the
    /// initial depth — the `AdaptiveDepthPolicy` picks per-cycle.
    pub mtp_depth: usize,
    /// MTP: when true, the decode loop runs an `AdaptiveDepthPolicy`
    /// (`adaptive_depth.rs`) that picks the draft depth per cycle by
    /// maximising per-depth EMA of `accepted_tokens / cycle_wall_ns`.
    /// When false, the loop pins `mtp_depth` for every cycle.
    ///
    /// Default resolution (`extract_chat_params`):
    ///   * User set `mtpAdaptiveDepth` explicitly → use that.
    ///   * User set `mtpDepth` (a fixed numeric depth) but NOT
    ///     `mtpAdaptiveDepth` → `false` (pin to user's depth).
    ///   * Neither field set → `false` (pin to default depth 1). Set
    ///     `mtpAdaptiveDepth=true` explicitly to enable the adaptive
    ///     policy.
    pub mtp_adaptive_depth: bool,
}

/// Resolve the effective `enable_thinking` value from `reasoning_effort`.
///
/// In vLLM, `enable_thinking` is a low-level template kwarg nested inside
/// `chat_template_kwargs`. `reasoning_effort` is the user-facing control that
/// drives it. This function maps the user-facing API to the template parameter.
pub(crate) fn resolve_enable_thinking(config: &ChatConfig) -> Option<bool> {
    match config.reasoning_effort.as_deref() {
        Some("none") | Some("low") => Some(false),
        Some("medium") | Some("high") => Some(true),
        _ => None, // not set → default (template decides, typically true)
    }
}

/// Default thinking-token budget for models whose chat template CANNOT suppress thinking
/// (e.g. LFM2). None = unlimited. Qwen3.5 must NOT call this (its template honors enable_thinking).
pub(crate) fn default_thinking_budget_for_effort(reasoning_effort: Option<&str>) -> Option<i32> {
    match reasoning_effort {
        Some("none") => Some(0),  // force </think> ASAP → minimal thinking
        Some("low") => Some(256), // small cap; short reasoning still leaves room to answer
        _ => None,                // medium/high/unset → unlimited (preserves current default)
    }
}

/// Resolve `include_reasoning` from config, with `reasoning_effort: "none"` default.
pub(crate) fn resolve_include_reasoning(config: &ChatConfig) -> bool {
    config
        .include_reasoning
        .unwrap_or(!matches!(config.reasoning_effort.as_deref(), Some("none")))
}

/// Extract ChatConfig fields into flat variables with defaults.
pub(crate) fn extract_chat_params(config: &ChatConfig) -> ChatParams {
    ChatParams {
        // Nonpositive budgets clamp to 0 (AR-equivalent empty completion)
        // so downstream max_kv_len / cache sizing / the decode macro never
        // see a negative budget. This single point feeds qwen3_5
        // dense/paged + qwen3_5_moe (incl. MTP); it is the core backstop
        // behind the server-side reject in `/v1/responses`.
        max_new_tokens: config.max_new_tokens.unwrap_or(2048).max(0),
        repetition_penalty: config.repetition_penalty.unwrap_or(1.0),
        repetition_context_size: config.repetition_context_size.unwrap_or(256),
        presence_penalty: config.presence_penalty.unwrap_or(0.0),
        presence_context_size: config.presence_context_size.unwrap_or(20),
        frequency_penalty: config.frequency_penalty.unwrap_or(0.0),
        frequency_context_size: config.frequency_context_size.unwrap_or(20),
        max_consecutive_tokens: config.max_consecutive_tokens.unwrap_or(16),
        max_ngram_repeats: config.max_ngram_repeats.unwrap_or(3),
        ngram_size: config.ngram_size.unwrap_or(64),
        sampling_config: Some(SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
        }),
        report_performance: config.report_performance.unwrap_or(false),
        reuse_cache: config.reuse_cache.unwrap_or(true),
        thinking_token_budget: config.thinking_token_budget,
        include_reasoning: resolve_include_reasoning(config),
        // MTP defaults OFF. When MTP is enabled and the caller does not
        // choose a depth, pin depth 1: current M5 Max measurements show
        // deeper bf16 MTP-head cycles lose more verify/draft time than
        // they recover from acceptance.
        enable_mtp: config.enable_mtp.unwrap_or(false),
        // Clamp the SIGNED depth before casting to usize: a negative
        // `mtpDepth` (reachable via the public `ChatConfig` surface) would
        // otherwise wrap (`-1 as usize` == usize::MAX) and clamp UP to 5,
        // forcing the slowest/deepest MTP path. Clamping first preserves the
        // documented `<1 → 1` behavior.
        mtp_depth: config
            .mtp_depth
            .map(|d| d.clamp(1, 5) as usize)
            .unwrap_or(1),
        // Adaptive depth policy is opt-in by default. An explicit
        // `mtpAdaptiveDepth` always wins. See
        // `ChatParams::mtp_adaptive_depth` docs.
        mtp_adaptive_depth: config.mtp_adaptive_depth.unwrap_or(false),
    }
}

/// Round a `(prefill_len + max_new_tokens)` token budget up to the next multiple of 256
/// for KV-cache capacity sizing. Computed in i64 so a hostile or absurd `max_new_tokens`
/// near `i32::MAX` cannot overflow the i32 sum (which would panic in debug / silently wrap
/// in release) before cache initialization. Inputs are floored at 0 (callers already clamp
/// budgets to >= 0 via `extract_chat_params`; this is defense in depth — for any
/// non-negative input the result is byte-identical to the legacy
/// `((prefill_len + max_new_tokens + 255) / 256) * 256`).
///
/// Returns `Err` when the rounded capacity would exceed `i32::MAX`, since the native
/// cache/FFI APIs are i32-typed; the caller surfaces this as a normal request error
/// instead of overflowing.
pub fn kv_capacity_round_up(prefill_len: i32, max_new_tokens: i32) -> Result<i32> {
    let total = (prefill_len.max(0) as i64) + (max_new_tokens.max(0) as i64) + 255;
    let rounded = (total / 256) * 256;
    if rounded > i32::MAX as i64 {
        return Err(Error::from_reason(format!(
            "requested KV-cache capacity {rounded} (prefill_len={prefill_len} + \
             max_new_tokens={max_new_tokens}, rounded up to a multiple of 256) exceeds the \
             maximum supported size of {}",
            i32::MAX
        )));
    }
    Ok(rounded as i32)
}

/// Saturating variant for DISPLAY/TRACE only — never errors. Clamps to the largest
/// multiple of 256 representable in i32. MUST NOT be used to size a real allocation.
pub fn kv_capacity_round_up_saturating(prefill_len: i32, max_new_tokens: i32) -> i32 {
    kv_capacity_round_up(prefill_len, max_new_tokens).unwrap_or((i32::MAX / 256) * 256)
}

/// Eager-allocation cap for generated-output `Vec::with_capacity` hints. A
/// `Vec::with_capacity` reserves memory immediately, so a hostile-but-accepted
/// token budget near `i32::MAX` would otherwise reserve gigabytes up front
/// (~8 GiB for a `Vec<u32>`). Real budgets up to this cap pre-allocate exactly;
/// larger budgets pre-allocate this much then grow via amortized doubling (a few
/// reallocs of a few KiB — negligible next to multi-second decode). Bounding the
/// HINT changes no observable behavior because the Vec still grows to hold every
/// generated token.
pub const GENERATED_CAPACITY_HINT_CAP: usize = 8192;

/// Bounded `Vec::with_capacity` hint for a generated-output buffer (tokens or
/// logprobs) sized from an untrusted `max_new_tokens` budget. Floors negatives at
/// 0 (so a negative budget can never produce a `usize::MAX` capacity that aborts)
/// and caps the eager reservation at [`GENERATED_CAPACITY_HINT_CAP`]; the buffer
/// still grows as needed during decode.
pub fn generated_capacity_hint(max_new_tokens: i32) -> usize {
    (max_new_tokens.max(0) as usize).min(GENERATED_CAPACITY_HINT_CAP)
}

/// Apply repetition + presence + frequency penalties to logits.
pub(crate) fn apply_all_penalties(
    mut logits: MxArray,
    token_history: &[u32],
    params: &ChatParams,
) -> Result<MxArray> {
    if params.repetition_penalty != 1.0 && !token_history.is_empty() {
        logits = apply_repetition_penalty(
            &logits,
            token_history,
            params.repetition_penalty,
            Some(params.repetition_context_size),
        )?;
    }
    if params.presence_penalty != 0.0 {
        logits = apply_presence_penalty(
            &logits,
            token_history,
            params.presence_penalty,
            Some(params.presence_context_size),
        )?;
    }
    if params.frequency_penalty != 0.0 {
        logits = apply_frequency_penalty(
            &logits,
            token_history,
            params.frequency_penalty,
            Some(params.frequency_context_size),
        )?;
    }
    Ok(logits)
}

/// Tracks reasoning vs content state during token-by-token generation.
///
/// For Qwen3.5: the template injects `<think>\n` when thinking is enabled.
/// The model generates thinking tokens, then emits `</think>` (think_end_id),
/// then generates content. This tracker detects the transition at the TOKEN
/// level — no text parsing needed during decoding.
pub(crate) struct ReasoningTracker {
    in_thinking: bool,
    thinking_token_count: i32,
    budget: Option<i32>,
    think_end_id: Option<u32>,
    force_think_end: bool,
    /// Set after `should_force_think_end` is consumed, prevents re-triggering
    /// from subsequent `observe_token` calls before the forced token is extracted.
    end_scheduled: bool,
}

impl ReasoningTracker {
    /// Create a new tracker.
    ///
    /// `starts_in_thinking`: true when the template injected `<think>\n` (thinking enabled).
    /// `budget`: maximum thinking tokens before forcing `</think>`. None = unlimited.
    /// `think_end_id`: token ID for `</think>` from the tokenizer vocabulary.
    pub fn new(starts_in_thinking: bool, budget: Option<i32>, think_end_id: Option<u32>) -> Self {
        // Budget=0 means "no thinking tokens at all" — force </think> immediately
        // on the first decode step, before any thinking token is generated.
        let force_immediately = starts_in_thinking && budget == Some(0) && think_end_id.is_some();
        Self {
            in_thinking: starts_in_thinking,
            thinking_token_count: 0,
            budget,
            think_end_id,
            force_think_end: force_immediately,
            end_scheduled: false,
        }
    }

    /// Process a generated token. Returns whether this token is reasoning content.
    ///
    /// Call AFTER extracting the token ID from the GPU each decode step.
    pub fn observe_token(&mut self, token_id: u32) -> bool {
        if !self.in_thinking {
            return false;
        }

        if self.think_end_id == Some(token_id) {
            self.in_thinking = false;
            self.force_think_end = false;
            self.end_scheduled = false;
            return true; // </think> itself is part of reasoning
        }

        self.thinking_token_count += 1;
        if let Some(budget) = self.budget
            && self.thinking_token_count >= budget
            && !self.end_scheduled
        {
            self.force_think_end = true;
        }
        true
    }

    /// Whether the next token should be forced to think_end_id.
    /// Consumes the flag — returns true at most once per budget trigger.
    ///
    /// Check this BEFORE building the next decode step's graph.
    pub fn should_force_think_end(&mut self) -> bool {
        if self.force_think_end && self.think_end_id.is_some() {
            self.force_think_end = false;
            self.end_scheduled = true;
            true
        } else {
            false
        }
    }

    /// Non-consuming peek: whether a think-end force is currently pending.
    /// Unlike `should_force_think_end`, this does NOT clear the flag or set
    /// `end_scheduled`, so it is safe to call during routing/defer decisions.
    /// The single consuming call must remain at the actual token-insertion site.
    pub fn force_think_end_pending(&self) -> bool {
        self.force_think_end && self.think_end_id.is_some()
    }

    /// The think_end token ID to force. Only valid when `should_force_think_end()` returned true.
    pub fn forced_token_id(&self) -> Result<u32> {
        self.think_end_id.ok_or_else(|| {
            napi::Error::from_reason("should_force_think_end was true but think_end_id is None")
        })
    }

    /// Number of tokens generated during reasoning (inside <think>...</think>).
    pub fn reasoning_token_count(&self) -> u32 {
        self.thinking_token_count.max(0) as u32
    }
}

/// Compute TTFT / prefill tok/s / decode tok/s performance metrics.
pub(crate) fn compute_performance_metrics(
    generation_start: Option<std::time::Instant>,
    first_token_instant: Option<std::time::Instant>,
    prefill_tokens_len: usize,
    generated_tokens_len: usize,
) -> Option<crate::profiling::PerformanceMetrics> {
    let (gen_start, first_tok) = match (generation_start, first_token_instant) {
        (Some(gs), Some(ft)) => (gs, ft),
        _ => return None,
    };
    let generation_end = std::time::Instant::now();
    let actual_prefill_toks = prefill_tokens_len as f64;
    let gen_toks = generated_tokens_len as f64;
    let ttft_ms = first_tok.duration_since(gen_start).as_secs_f64() * 1000.0;
    let decode_ms = generation_end.duration_since(first_tok).as_secs_f64() * 1000.0;
    Some(crate::profiling::PerformanceMetrics {
        ttft_ms,
        prefill_tokens_per_second: if ttft_ms > 0.0 {
            actual_prefill_toks / (ttft_ms / 1000.0)
        } else {
            0.0
        },
        decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
            (gen_toks - 1.0) / (decode_ms / 1000.0)
        } else {
            0.0
        },
        // MTP acceptance is filled in post-hoc by the MTP decode paths
        // via `DecodeProfiler::fill_mtp_acceptance` — the profiler is in
        // scope there but not here. Stays `None` on autoregressive runs.
        mtp_mean_accepted_tokens: None,
        mtp_mean_accepted_tokens_total: None,
        mtp_acceptance_by_position: None,
        mtp_cycles: None,
        mtp_mean_depth: None,
        profile_phases: None,
    })
}

/// Shared finalization: parse thinking + tool calls from decoded text.
///
/// Four-way branching based on the request's reasoning state:
/// 1. `!thinking_enabled`: no-thinking mode — all text is content, no reasoning parsing.
/// 2. `thinking_enabled` + `</think>` token confirmed: split at token-confirmed boundary.
/// 3. `thinking_enabled` + no `</think>` token + `think_end_id` exists: truncated generation.
/// 4. `thinking_enabled` + no `think_end_id` in vocab: text-level fallback via `split_at_think_end`.
///
/// `include_reasoning`: when false, thinking field is suppressed (set to None).
pub(crate) fn parse_thinking_and_tools(
    text: &str,
    generated_tokens: &[u32],
    thinking_enabled: bool,
    think_end_id: Option<u32>,
    think_end_str: Option<&str>,
    include_reasoning: bool,
) -> (String, Vec<tools::ToolCallResult>, Option<String>) {
    let (clean_text, tool_calls, thinking) = if !thinking_enabled {
        // No-thinking mode: all text is content, passed through verbatim.
        // Any literal <think> tags are normal model output, not markup.
        let (clean, calls) = tools::parse_tool_calls(text);
        (clean, calls, None)
    } else if tools::has_think_end_token(generated_tokens, think_end_id) {
        // Thinking mode with confirmed </think>: split at token boundary.
        tools::split_at_think_end(text, think_end_str)
    } else if think_end_id.is_some() {
        // Thinking mode, truncated (no </think> before EOS/max_tokens):
        // entire output is reasoning, no content.
        let thinking_text = text.trim();
        // Strip leading <think>/<longcat_think> from old-style templates
        // that emit it in the generated text.
        let thinking_text = thinking_text
            .strip_prefix("<think>")
            .or_else(|| thinking_text.strip_prefix("<longcat_think>"))
            .unwrap_or(thinking_text)
            .trim();
        let thinking = if thinking_text.is_empty() {
            None
        } else {
            Some(thinking_text.to_string())
        };
        (String::new(), vec![], thinking)
    } else {
        // No think_end_id in vocab — cannot do token-level detection. Isolate reasoning
        // from content at the text level BEFORE extracting tool calls, so a `<tool_call>`
        // nested inside a reasoning block is NOT surfaced as an executable tool call.
        // This matches the token-confirmed path (which parses tool calls only from the
        // post-`</think>` content) and the `raw_text` scrub, so on this fallback the
        // `text`, `tool_calls`, and `raw_text` fields stay consistent. A standalone
        // `<tool_call>` outside reasoning is preserved and still extracted.
        let content = tools::strip_reasoning_preserving_tools(text);
        let (clean, calls) = tools::parse_tool_calls(&content);
        // Thinking field keeps the prior fallback derivation (reasoning parsed from the
        // tool-stripped text, so an in-argument think tag of a standalone tool call does
        // not masquerade as reasoning).
        let (text_without_tools, _) = tools::parse_tool_calls(text);
        let thinking = tools::parse_thinking(&text_without_tools).1;
        (clean.trim().to_string(), calls, thinking)
    };

    // Suppress reasoning if not requested
    let thinking = if include_reasoning { thinking } else { None };

    (clean_text, tool_calls, thinking)
}

/// Build the `raw_text` field with the reasoning span removed when reasoning is
/// not requested.
///
/// `raw_text` is normally the verbatim decoded generation (including
/// `<think>…</think>`). When `include_reasoning` is false we additionally strip
/// the reasoning span so a direct `raw_text` consumer cannot recover the model's
/// chain-of-thought — matching the suppression already applied to the parsed
/// `thinking` field and the streamed reasoning deltas.
///
/// The post-`</think>` content is kept VERBATIM (tool-call markup, whitespace,
/// the model's exact bytes) so `raw_text`'s downstream uses (e.g. tool-call
/// markup recovery) keep working. The branch structure mirrors
/// `parse_thinking_and_tools` so the boundary is identical to the one used for
/// the parsed `thinking`/`text` fields.
pub(crate) fn raw_text_with_reasoning_suppressed(
    text: &str,
    generated_tokens: &[u32],
    thinking_enabled: bool,
    think_end_id: Option<u32>,
    think_end_str: Option<&str>,
    include_reasoning: bool,
) -> String {
    // Reasoning requested, or no-thinking mode (all output is content): verbatim.
    if include_reasoning || !thinking_enabled {
        return text.to_string();
    }
    if tools::has_think_end_token(generated_tokens, think_end_id) {
        // Confirmed </think>: keep everything after the FIRST occurrence verbatim.
        if let Some(tag) = think_end_str
            && let Some(close_pos) = text.find(tag)
        {
            return text[close_pos + tag.len()..].to_string();
        }
        // Token confirmed but tag string unavailable/unlocatable: fall through to
        // the text-level strip below.
    } else if think_end_id.is_some() {
        // Truncated generation (no </think> before EOS/max): all reasoning.
        return String::new();
    }
    // No think_end_id in vocab (or tag unlocatable): text-level scrub. Strips EVERY
    // reasoning block of BOTH `<think>`/`<longcat_think>` families (parse_thinking
    // alone only handles the first family) while preserving `<tool_call>…</tool_call>`
    // spans verbatim — so reasoning-looking tags inside tool arguments can't corrupt
    // the tool markup that `raw_text` consumers (e.g. server tool-call recovery) rely on.
    tools::strip_reasoning_preserving_tools(text)
}

/// Decode tokens, parse thinking/tool_calls, build ChatResult.
pub(crate) fn finalize_chat_result(
    tokenizer: &Qwen3Tokenizer,
    generated_tokens: &[u32],
    finish_reason: String,
    think_end_id: Option<u32>,
    think_end_str: Option<&str>,
    performance: Option<crate::profiling::PerformanceMetrics>,
    include_reasoning: bool,
    thinking_enabled: bool,
    prompt_tokens: u32,
    reasoning_tokens: u32,
) -> Result<ChatResult> {
    let text = tokenizer
        .decode_sync(generated_tokens, true)
        .unwrap_or_else(|e| {
            tracing::warn!("Failed to decode generated tokens: {}", e);
            String::new()
        });

    let num_tokens = generated_tokens.len() as u32;

    let (clean_text, tool_calls, thinking) = parse_thinking_and_tools(
        &text,
        generated_tokens,
        thinking_enabled,
        think_end_id,
        think_end_str,
        include_reasoning,
    );

    // If we have valid tool calls, override finish reason
    let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
        "tool_calls".to_string()
    } else {
        finish_reason
    };

    let raw_text = raw_text_with_reasoning_suppressed(
        &text,
        generated_tokens,
        thinking_enabled,
        think_end_id,
        think_end_str,
        include_reasoning,
    );

    Ok(ChatResult {
        text: clean_text,
        tool_calls,
        thinking,
        num_tokens,
        prompt_tokens,
        reasoning_tokens,
        finish_reason,
        raw_text,
        // Callers that reused a cached prefix overwrite this via their own
        // `cached_prefix_len as u32` after this function returns. Defaulting
        // to zero keeps the behavior of callers that do not (yet) thread
        // the value through intact.
        cached_tokens: 0,
        performance,
    })
}

/// Whether the compiled init should re-apply the saved M-RoPE offset
/// (`cached_rope_deltas`) after building the decode graph.
///
/// The offset is saved only when a VLM prefill ran, so `has_saved_delta`
/// is effectively "the live KV cache encodes image attention". Two
/// callers need to re-apply it:
///   - **Fresh VLM prefill reusing a cached prefix** (`has_images &&
///     cached_prefix_len > 0`): the new turn shares its image grid with
///     the cached one, and the saved offset carries the image-adjusted
///     M-RoPE position forward into the rebuilt compiled graph.
///   - **Session delta continuation** (`is_delta`): the delta prefill
///     just ran on top of the live KV caches, which still encode the
///     prior VLM prefill's image attention. Without re-applying the
///     offset, the newly-built compiled graph would decode at a
///     sequential M-RoPE position and misposition all generated tokens
///     relative to the cached image patches.
///
/// Pure function — extracted so the decision can be unit-tested
/// without instantiating the compiled decoder.
pub(crate) fn should_reapply_rope_delta(
    has_saved_delta: bool,
    is_delta: bool,
    has_images: bool,
    cached_prefix_len: usize,
) -> bool {
    has_saved_delta && (is_delta || (has_images && cached_prefix_len > 0))
}

/// Whether the compiled init should clear `cached_rope_deltas` after
/// building the decode graph.
///
/// Only fresh text-only prefills clear the offset: they signal that the
/// non-delta cache-prefix verify dropped any prior image-bearing cache,
/// so the stored offset is stale. Delta continuations preserve the
/// offset so chained text-only turns on an image session keep the
/// image-adjusted M-RoPE position.
///
/// Pure function — extracted so the decision can be unit-tested.
pub(crate) fn should_clear_rope_delta(is_delta: bool, has_images: bool) -> bool {
    !has_images && !is_delta
}

/// Direct-ownership version of `save_cache_state` for dedicated-thread models.
///
/// Takes `&mut` refs instead of `Arc<RwLock<>>`. Used by Qwen3.5 Dense on
/// its dedicated model thread.
pub(crate) fn save_cache_state_direct(
    reuse_cache: bool,
    has_images: bool,
    generated_tokens: &[u32],
    finish_reason: &str,
    tokens: &[u32],
    expanded_tokens: Option<&[u32]>,
    image_cache_key: u64,
    cached_token_history: &mut Vec<u32>,
    cached_image_key: &mut Option<u64>,
    cached_rope_deltas: &mut Option<i32>,
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
) {
    if reuse_cache {
        let mut full_history = if has_images {
            expanded_tokens.unwrap_or(tokens).to_vec()
        } else {
            tokens.to_vec()
        };
        let history_tokens = if finish_reason == "length" && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            generated_tokens
        };
        full_history.extend_from_slice(history_tokens);
        *cached_token_history = full_history;
        *cached_image_key = if has_images {
            Some(image_cache_key)
        } else {
            None
        };
    } else {
        *caches = None;
        cached_token_history.clear();
        *cached_image_key = None;
        *cached_rope_deltas = None;
    }
}

/// Commit session state after a text-only delta continuation.
///
/// The delta path (`chat_tokens_delta_sync` / `chat_stream_tokens_delta_sync`)
/// appends a text delta on top of the live KV caches without touching the
/// image attention state baked in by the preceding prefill. The "current
/// turn is text-only" signal (`has_images == false`) MUST NOT be conflated
/// with "the session has no image context" — the KV caches still encode
/// every image patch from the earlier `chat_session_start` / VLM prefill,
/// and clearing `cached_image_key` here would make the next cache-prefix
/// verify think the session is pure text and accept a future image-carrying
/// turn via the delta path (which produces garbage because the mrope
/// offset `cached_rope_deltas` is stale for the new image grid).
///
/// This helper is identical to [`save_cache_state_direct`] except that it
/// leaves `cached_image_key` untouched on the `reuse_cache=true` branch.
/// The full-reset `reuse_cache=false` branch still clears everything —
/// same invariant as the prefill helper.
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_cache_state_after_delta(
    reuse_cache: bool,
    generated_tokens: &[u32],
    finish_reason: &str,
    save_tokens: &[u32],
    cached_token_history: &mut Vec<u32>,
    cached_image_key: &mut Option<u64>,
    cached_rope_deltas: &mut Option<i32>,
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
) {
    if reuse_cache {
        let mut full_history = save_tokens.to_vec();
        let history_tokens = if finish_reason == "length" && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            generated_tokens
        };
        full_history.extend_from_slice(history_tokens);
        *cached_token_history = full_history;
        // `cached_image_key` intentionally preserved — see doc comment.
    } else {
        *caches = None;
        cached_token_history.clear();
        *cached_image_key = None;
        *cached_rope_deltas = None;
    }
}

/// Direct-ownership version of `verify_cache_prefix` for dedicated-thread models.
///
/// Takes direct refs instead of `Arc<RwLock<>>`. Used by Qwen3.5 Dense on
/// its dedicated model thread.
///
/// # Return-value invariant (load-bearing)
///
/// This helper returns **either `0` (cache miss — caller MUST reset caches
/// before prefill) or `cached.len()` (exact-append hit — the new prompt
/// strictly extends the cached history)**. It **never** returns an
/// intermediate value such as "the first K tokens match, rewind to K".
///
/// That all-or-nothing contract is what makes it safe to drive Qwen3.5's
/// **hybrid linear + attention stack**. The Gated Delta Net (GDN) layers
/// carry a *recurrent* state (`conv_state`, `recurrent_state` in
/// [`super::layer_cache::Qwen3_5LayerCache::Linear`]) that folds every
/// absorbed token irreversibly into its hidden state — unlike a standard
/// KV cache, a GDN cache **cannot be trimmed or rewound mid-sequence**
/// without corrupting the representation. A non-zero return from this
/// function therefore always means "the incoming tokens are a *pure append*
/// on top of the cached state; continue decoding from the current live
/// caches". No mid-sequence rewind ever happens.
///
/// Any future modification that would relax this contract (e.g. returning
/// a prefix count less than `cached.len()`) MUST simultaneously ensure the
/// caller either (a) restricts the relaxation to pure-KVCache models or
/// (b) introduces GDN-state checkpointing to enable mid-sequence rewinds.
/// Neither has been done — the invariant here is the sole reason the
/// refactor that moves `reset_caches_sync()` from the outer session-start
/// path into the `cached_prefix_len == 0` branch of `chat_sync_core` is
/// safe for Qwen3.5 Dense and MoE.
pub(crate) fn verify_cache_prefix_direct(
    reuse_cache: bool,
    has_images: bool,
    tokens: &[u32],
    tokens_for_matching: &[u32],
    image_cache_key: u64,
    cached_token_history: &[u32],
    cached_image_key: &Option<u64>,
    has_caches: bool,
) -> usize {
    if !reuse_cache {
        return 0;
    }
    let cached = cached_token_history;
    if has_images {
        if let Some(cached_key) = *cached_image_key
            && cached_key == image_cache_key
            && !cached.is_empty()
            && tokens_for_matching.len() >= cached.len()
            && tokens_for_matching[..cached.len()] == cached[..]
            && has_caches
        {
            return cached.len();
        }
        0
    } else if !cached.is_empty()
        && tokens.len() >= cached.len()
        && tokens[..cached.len()] == cached[..]
        && has_caches
    {
        cached.len()
    } else {
        0
    }
}

/// Closures for model-specific operations in the decode loop.
///
/// `F`: forward pass — takes (input_ids [1,1], embedding_weight) → Result<(logits, needs_squeeze)>.
/// `E`: eval step — takes (next_token, logits, budget_forced) → schedules async eval.
pub(crate) struct DecodeOps<F, E>
where
    F: FnMut(&MxArray, &MxArray) -> Result<(MxArray, bool)>,
    E: Fn(&MxArray, &MxArray, bool),
{
    pub forward: F,
    pub eval_step: E,
}

/// Pipelined decode loop shared across all Qwen3.5 model variants.
///
/// Generates the token-by-token decode loop with:
/// - Pipelining: builds step N+1's graph before blocking on step N
/// - Budget enforcement via ReasoningTracker
/// - Penalty application via apply_all_penalties
/// - Stop conditions: EOS, repetition cutoff
/// - Every-256-step synchronize_and_clear_cache
/// - Profiler instrumentation
///
/// The optional `streaming:` block adds callback emission, cancellation,
/// incremental detokenization, and is_reasoning tagging.
macro_rules! decode_loop {
    (
        ops: $ops:expr,
        y: $y:expr,
        embedding_weight: $emb:expr,
        params: $p:expr,
        reasoning_tracker: $tracker:expr,
        profiler: $profiler:expr,
        max_new_tokens: $max:expr,
        eos_id: $eos:expr,
        generated_tokens: $gen:expr,
        token_history: $hist:expr,
        finish_reason: $reason:expr,
        first_token_instant: $first_tok:expr,
        report_perf: $report:expr,
        generation_stream: $stream:expr
        $(, streaming: {
            callback: $cb:expr,
            cancelled: $cancelled:expr,
            decode_stream: $ds:expr,
            tokenizer: $tok:expr,
            streamed_text_len: $slen:expr,
            last_is_reasoning: $last_r:expr
        })?
    ) => {{
        for step in 0..$max {
            let next_y = if step + 1 < $max {
                let _stream_ctx = $crate::stream::StreamContext::new($stream);

                $profiler.begin("forward");
                let next_ids = $y.reshape(&[1, 1])?;
                let (mut logits, needs_squeeze) = ($ops.forward)(&next_ids, &$emb)?;
                if needs_squeeze {
                    logits = logits.squeeze(Some(&[1]))?;
                }
                $profiler.end();

                let (next_token, budget_forced) =
                    if $tracker.should_force_think_end() {
                        let forced_id = $tracker.forced_token_id()? as i32;
                        ($crate::array::MxArray::from_int32(&[forced_id], &[1])?, true)
                    } else {
                        $profiler.begin("rep_penalty");
                        logits = $crate::models::qwen3_5::chat_common::apply_all_penalties(
                            logits, &$hist, &$p,
                        )?;
                        $profiler.end();

                        $profiler.begin("sample");
                        let t = $crate::sampling::sample(&logits, $p.sampling_config)?;
                        $profiler.end();
                        (t, false)
                    };

                $profiler.begin("eval_caches");
                ($ops.eval_step)(&next_token, &logits, budget_forced);
                $profiler.end();

                // Diagnostic — `MLX_MTP_TRACE_LOGITS=1` per-token AR
                // top-2 logit trace. `logits` is the post-penalty
                // single-token decode forward that PREDICTS the token
                // at position `$hist.len() + 1` (the current `$y` sits
                // at `$hist.len()`). `budget_forced` skips the real
                // logits, so only trace the sampled path.
                if !budget_forced
                    && $crate::models::qwen3_5::chat_common::mtp_trace_logits()
                {
                    let logits_1d = if logits.ndim()? == 2 {
                        logits.squeeze(Some(&[0]))?
                    } else {
                        logits.clone()
                    };
                    let vocab = logits_1d.shape_at(0)?;
                    match $crate::models::qwen3_5::chat_common::trace_top2(
                        &logits_1d, vocab,
                    ) {
                        Ok(t2) => {
                            next_token.eval();
                            let predicted = next_token.item_at_int32(0)?;
                            eprintln!(
                                "MTP_TRACE_LOGITS source=AR pos={} token_id={} \
                                 top1_id={} top1_logit={:.6} top2_id={} \
                                 top2_logit={:.6} gap={:.6}",
                                $hist.len() + 1,
                                predicted,
                                t2.top1_id,
                                t2.top1_logit,
                                t2.top2_id,
                                t2.top2_logit,
                                t2.top1_logit - t2.top2_logit,
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "MTP_TRACE_LOGITS source=AR pos={} ERROR {}",
                                $hist.len() + 1,
                                e.reason,
                            );
                        }
                    }
                }

                Some(next_token)
            } else {
                None
            };

            $profiler.begin("eval_token");
            $y.eval();
            $profiler.end();

            $profiler.begin("extract");
            let token_id = $y.item_at_int32(0)? as u32;
            $profiler.end();
            $profiler.mark_first_token();
            if $report && $first_tok.is_none() {
                $first_tok = Some(std::time::Instant::now());
            }

            $gen.push(token_id);
            $hist.push(token_id);
            let _is_reasoning = $tracker.observe_token(token_id);

            // Throttled per-step decode trace (AR / single-token loop).
            // Logs every 32 steps so long decode runs leave a sparse
            // breadcrumb trail (step idx, sampled token, cache offset
            // from the dense compiled global — MoE callers can ignore
            // the offset field).
            if step % 32 == 0 {
                let cache_offset = unsafe { mlx_sys::mlx_qwen35_get_cache_offset() };
                tracing::info!(
                    "Qwen3.5 decode AR step={} sampled_token_id={} cache_offset={} gen_len={}",
                    step,
                    token_id,
                    cache_offset,
                    $gen.len(),
                );
            }

            // Streaming-only block (conditionally compiled via macro repetition)
            $(
                $last_r = _is_reasoning;

                if $cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                    $reason = String::from("cancelled");
                    break;
                }

                let token_text = $crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                    &mut $ds,
                    $tok.inner(),
                    token_id,
                    &$gen,
                    $slen,
                );
                $slen += token_text.len();
                // Suppress reasoning (<think>…</think>) deltas from the stream
                // when include_reasoning == false. Detokenize + length-advance
                // above stay OUTSIDE this gate so DecodeStream sees every token.
                if $p.include_reasoning || !_is_reasoning {
                    $cb.call(
                        Ok($crate::models::qwen3_5::model::ChatStreamChunk {
                            text: token_text,
                            done: false,
                            finish_reason: None,
                            tool_calls: None,
                            thinking: None,
                            num_tokens: None,
                            prompt_tokens: None,
                            reasoning_tokens: None,
                            raw_text: None,
                            cached_tokens: None,
                            performance: None,
                            is_reasoning: Some(_is_reasoning),
                        }),
                        napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                    );
                }
            )?

            if token_id == $eos {
                $reason = String::from("stop");
                break;
            }

            if let Some(reason) = $crate::sampling::check_repetition_cutoff(
                &$gen,
                $p.max_consecutive_tokens,
                $p.max_ngram_repeats,
                $p.ngram_size,
            ) {
                $reason = reason.to_string();
                break;
            }

            match next_y {
                Some(next) => $y = next,
                None => break,
            }

            $profiler.step();

            if (step + 1) % 256 == 0 {
                $crate::array::synchronize_and_clear_cache();
            }
        }

        $profiler.snapshot_memory_after();
        $profiler.report();
    }};
}

pub(crate) use decode_loop;

// =============================================================================
// MTP speculative decode loop (dense compiled path only).
//
// Sister to `decode_loop!` above; preserves every behavior of the
// single-token loop (penalties, ReasoningTracker / budget, EOS,
// repetition cutoff, every-256-step cache clear, streaming +
// cancellation) while emitting up to `mtp_depth + 1` tokens per
// outer iteration via the draft + verify FFI plus the
// `accept_with_residual` sampler.
//
// Cache-rollback strategy (compiled path):
//   - The verify FFI advances the MAIN compiled K/V offset by
//     `depth + 1`. On any rejection we rewind the main offset by
//     `accepted_count - (depth + 1)` (a negative delta) via
//     `mlx_qwen35_compiled_adjust_offset`. We do NOT zero the K/V
//     buffer entries at positions `[accepted .. depth+1]` — the
//     next forward simply overwrites them.
//   - The MTP draft FFI advances its OWN offset by 1 per draft
//     step. On any rejection we rewind by `accepted_count - depth`
//     via `mlx_qwen35_mtp_compiled_adjust_offset`. The MTP path is
//     by design 1-token behind the main path's accepted prefix.
//   - `Qwen3_5LayerCache::snapshot_all` / `restore_all` is the
//     EAGER-PATH rollback primitive. On the compiled path the live
//     K/V lives in `g_compiled_caches` (C++), not `self.caches`, so
//     the snapshot/restore is intentionally NOT used here.
//
// Tracker / budget invariant:
//   - `should_force_think_end()` is checked BEFORE starting each
//     draft cycle. If forced, the macro emits ONE forced token via
//     the normal main-path forward + sampler and skips the cycle.
//   - It is also checked BEFORE accepting each individual verified
//     token. If forced mid-cycle, the macro aborts the remaining
//     accepted tokens, rewinds the offsets to (already-emitted + 1)
//     committed positions on top of the forced token, and emits the
//     forced token through the normal path.
// =============================================================================

/// Closure bundle for the MTP cycle. Mirrors `DecodeOps` but adds
/// draft / verify / rollback hooks that are only meaningful on the
/// dense compiled path.
///
/// `F`  : single main-path forward step returning `(logits, hidden,
///        needs_squeeze)`. `hidden` is `[1, hidden_size]` bf16.
/// `D`  : MTP draft step returning `(h_next, draft_logits)` where
///        `h_next` is `[1, 1, hidden]` and `draft_logits` is
///        `[1, vocab]`.
/// `V`  : MTP verify step returning verify logits of shape
///        `[1, depth + 1, vocab]`.
/// `R`  : rollback hook receiving `(accepted, depth)`. On rejection
///        the implementor calls
///        `mlx_qwen35_mtp_compiled_adjust_offset(accepted as i32 -
///        depth as i32)` (MTP path only). The MAIN path's offset is
///        rewound by `restore_and_replay_main` via the snapshot taken
///        in `snapshot_main_linear` — DO NOT also call
///        `mlx_qwen35_compiled_adjust_offset` here, or the main
///        offset will double-rewind.
/// `E`  : eval-step hook (same contract as `DecodeOps::eval_step`)
///        called after every emitted token to flush the lazy graph.
/// `B`  : begin-cycle hook called once per outer iteration, BEFORE
///        the draft steps, AFTER Step A's main-path forward. The
///        implementor reads the main path's current offset (via
///        `mlx_qwen35_get_cache_offset` / `mlx_qwen35_moe_get_cache_offset`)
///        and calls the corresponding `*_begin_cycle(main_offset)` FFI
///        to zero the MTP K/V caches and re-anchor the MTP offset.
///        Required for correctness: without the reset the MTP offset
///        lags the main offset by 2 per cycle (mid-stream divergence).
/// `S`  : snapshot hook for the main path's GDN linear-attention caches
///        plus the main decode offset. Called once per cycle AFTER
///        Step A and BEFORE verify. Implementor calls
///        `mlx_qwen35_compiled_snapshot_linear_caches` (or the MoE
///        equivalent). The snapshot is consumed by
///        `restore_and_replay_main` on rejection — without it the
///        GDN recurrent state stays polluted with rejected draft
///        positions and the next Step A produces wrong logits.
/// `RR` : restore + replay hook called on rejection (any
///        `accepted_drafts < depth`) AFTER `rollback`. Receives the
///        list of accepted draft token IDs (NOT including the residual
///        sample, NOT including `last_committed`) and the embedding
///        weight. Implementor:
///          1. Calls `mlx_qwen35_compiled_restore_linear_caches`
///             (rewinds linear caches AND main offset to the
///             snapshot point);
///          2. For each accepted draft, runs ONE
///             `mlx_qwen35_forward_compiled` (via
///             `forward_with_hidden` to keep the implementation
///             path-agnostic) so the main linear state catches up to
///             "after Step A + K accepted drafts" and the main offset
///             reaches `snapshot_offset + K`.
///        On full-accept the macro skips this hook (verify already
///        left the linear state advanced through all D drafts).
pub(crate) struct MtpOps<F, D, V, R, E, EX, B, S, RR, CM, RU>
where
    F: FnMut(&MxArray, &MxArray) -> Result<(MxArray, MxArray, bool)>,
    D: FnMut(&MxArray, &MxArray) -> Result<(MxArray, MxArray)>,
    // verify returns logits, verify_hiddens, and optionally precomputed
    // greedy target ids. Logits shape is `[1, depth+1, vocab]`; hiddens
    // shape is `[1, depth+1, hidden]`; target_argmax shape is
    // `[1, depth+1]` int32 when the backend can return it from the compiled
    // verify graph. In the greedy argmax-only and stochastic native sparse-target
    // paths, logits may be absent; those paths carry `target_argmax` or
    // `target_sparse` respectively.
    //
    // The hiddens carry the post-final-norm output at EVERY verify position.
    // After `run_mtp_cycle_inner` computes the number of accepted drafts K,
    // it slices `verify_hiddens[:, K, :]` to seed the next cycle's first MTP
    // draft — that hidden is the prediction context for the committed token
    // at position K+1 (bonus on full-accept, residual on rejection), matching
    // the MTP head's training contract.
    V: FnMut(&MxArray, &MxArray, usize) -> Result<MtpVerifyOutput>,
    R: FnMut(usize, usize),
    // Invoked from `decode_loop_mtp!` ONLY when a mid-cycle stop (EOS /
    // cancel / length / repetition cutoff) ends the emit loop after some
    // but not all of `outcome.tokens` have been emitted. Receives the
    // count of accepted-but-unemitted tokens. The paged path uses this to
    // truncate the live paged adapter so future turns don't reuse KV for
    // tokens the user never received. Dense / MoE / tests pass a no-op
    // closure (the dense compiled cursor is driven by `commit_mtp`, which
    // already ran for the full cycle — no extra rollback is required at
    // this boundary for dense KV consistency).
    RU: FnMut(usize),
    E: Fn(&MxArray, &MxArray, bool),
    // Same shape as `eval_step` but folds the chained `verify_hidden[K]`
    // slice into the SAME `async_eval` batch as `(token, g_compiled_caches)`.
    // Used by the chained-cycles macro path at end-of-iteration so the
    // slice is co-materialized with the next cycle's first-draft input
    // sources, eliminating the mid-cycle Metal command-buffer roundtrip
    // the lazy path forces.
    //
    // Contract: `token` is the just-set `$y` (CPU-only integer array
    // of the last accepted token); `chained_hidden` is the lazy
    // `verify_hiddens[:, K, :]` slice the prior cycle produced. The
    // closure MUST schedule `async_eval` (or equivalent) over the
    // dense / MoE compiled caches PLUS both arrays so the slice
    // becomes a sibling of the next-cycle draft graph rather than a
    // late dependency.
    EX: Fn(&MxArray, &MxArray),
    B: FnMut(bool),
    S: FnMut(),
    RR: FnMut(&[u32], &MxArray) -> Result<()>,
    // Committed-history commit hook. Called once per cycle AFTER the
    // accept loop computes `accepted_drafts` (K) and BEFORE `rollback`,
    // on BOTH the full-accept and reject paths. Receives
    // `(prev_hidden_in [1,1,hidden], verify_hiddens [1,D+1,hidden],
    // committed_ids [K+2], k_accepted, embedding_weight)`.
    //
    // `committed_ids` is the FULL committed sequence emitted by this
    // outer iteration — `[last_committed_id, d_0..d_{K-1}, boundary]`
    // (length K+2). `prev_hidden_in` is the cycle's seed hidden (Step
    // A's hidden output = h(token before last_committed_id)). The
    // implementor assembles the K+2 hidden / embedding rows and invokes
    // the commit FFI, which appends K+2 exact committed K/V slots to the
    // persistent MTP cache. A no-op closure disables committed-history
    // (MoE path + tests stay cycle-policy).
    CM: FnMut(MtpCommitAnchor, &MxArray, &MxArray, &[u32], usize, &MxArray) -> Result<()>,
{
    pub forward_with_hidden: F,
    pub draft_step: D,
    pub verify_step: V,
    pub verify_step_argmax_only:
        Option<Box<dyn FnMut(&MxArray, &MxArray, usize) -> Result<MtpVerifyOutput>>>,
    pub verify_step_sparse: Option<
        Box<
            dyn FnMut(
                &MxArray,
                &MxArray,
                usize,
                &sampling::SamplingConfig,
            ) -> Result<MtpVerifyOutput>,
        >,
    >,
    pub rollback: R,
    pub eval_step: E,
    pub eval_step_with_chained_hidden: EX,
    pub begin_cycle: B,
    pub snapshot_main_linear: S,
    pub restore_and_replay_main: RR,
    /// Committed-history commit hook. See the `CM` bound above for the
    /// contract. Pass a no-op closure to keep the cycle-history policy
    /// (MoE path, tests).
    pub commit_mtp: CM,
    /// Set `true` on the dense path where `commit_mtp` runs the real
    /// committed-history commit. When `true`,
    /// `run_mtp_cycle_inner` uses the per-step `draft_step` path whose
    /// attention mask is `chain_start <= pos <= offset` with
    /// `chain_start = 0` — correct under committed-history. MoE / tests
    /// leave this `false` (legacy cycle-history policy).
    pub committed_history_active: bool,
    pub rollback_unemitted: RU,
}

/// Commit payload policy for committed-history MTP.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MtpCommitAnchor {
    /// Step-A path: commit `[last_committed] ++ accepted_tokens`.
    IncludeAnchor,
    /// Chained path: `last_committed` is the prior cycle's already
    /// committed boundary, so commit only the newly emitted
    /// `accepted_tokens`.
    SkipAlreadyCommittedAnchor,
}

/// Outcome of `run_mtp_cycle_inner` — the list of accepted tokens for
/// this cycle plus whether a rejection forced a rollback (used by the
/// macro to log / observe).
pub(crate) struct MtpCycleOutcome {
    /// Accepted token IDs in emission order. Always at least one
    /// element on success (residual sample on full reject, or
    /// bonus token on full accept).
    pub tokens: Vec<u32>,
    /// Draft depth requested by the outer policy before intra-cycle gates.
    pub requested_depth: usize,
    /// Draft depth actually verified this cycle after intra-cycle gates.
    pub effective_depth: usize,
}

pub(crate) struct MtpVerifyOutput {
    pub logits: Option<MxArray>,
    pub hiddens: MxArray,
    pub target_argmax: Option<MxArray>,
    pub target_sparse: Option<sampling::SparseDistributionRows>,
}

impl MtpVerifyOutput {
    /// Verify produced dense logits only, with no precomputed target — the
    /// MoE compiled path and the dense chained-cycle path.
    pub(crate) fn logits_only(logits: MxArray, hiddens: MxArray) -> Self {
        Self {
            logits: Some(logits),
            hiddens,
            target_argmax: None,
            target_sparse: None,
        }
    }

    /// Verify produced dense logits plus the precomputed greedy-argmax target.
    pub(crate) fn logits_with_argmax(
        logits: MxArray,
        hiddens: MxArray,
        target_argmax: MxArray,
    ) -> Self {
        Self {
            logits: Some(logits),
            hiddens,
            target_argmax: Some(target_argmax),
            target_sparse: None,
        }
    }

    /// Verify produced only the greedy-argmax target (no dense logits) — the
    /// `argmax_only` fast path.
    pub(crate) fn argmax_only(hiddens: MxArray, target_argmax: MxArray) -> Self {
        Self {
            logits: None,
            hiddens,
            target_argmax: Some(target_argmax),
            target_sparse: None,
        }
    }

    /// Verify produced a precomputed sparse target distribution (no dense
    /// logits) — the native sparse-verify fast path.
    pub(crate) fn sparse(
        hiddens: MxArray,
        target_sparse: sampling::SparseDistributionRows,
    ) -> Self {
        Self {
            logits: None,
            hiddens,
            target_argmax: None,
            target_sparse: Some(target_sparse),
        }
    }
}

/// One MTP draft+verify cycle. Pure helper — the caller drives the
/// per-token streaming, EOS, cancellation, and tracker bookkeeping
/// inside `decode_loop_mtp!`.
///
/// Contract:
/// * `prev_hidden_in` / `prev_emb_in` are `[1, 1, hidden]` bf16.
/// * `last_committed_id` is the token at the end of `token_history`
///   — used as the first column of the verify input.
/// * `depth` MUST be `>= 1` and `<= 5` (verify FFI clamps).
/// * `embedding_weight` is the model's embedding table (already
///   resolved to the LM head when `tie_word_embeddings=false` on the
///   caller side — same arg as `forward_with_hidden`).
///
/// On error any partial offset / K/V drift from the partial cycle
/// is the caller's problem; production callers fold the cycle inside
/// `DENSE_COMPILED_MUTEX` so a `?` early-return drops the
/// `CompiledResetGuard` and wipes the C++ state cleanly.
pub(crate) fn run_mtp_cycle_inner<F, D, V, R, E, EX, B, S, RR, CM, RU>(
    ops: &mut MtpOps<F, D, V, R, E, EX, B, S, RR, CM, RU>,
    prev_hidden_in: MxArray,
    prev_emb_in: MxArray,
    last_committed_id: u32,
    embedding_weight: &MxArray,
    token_history: &[u32],
    params: &ChatParams,
    rng: &mut impl rand::Rng,
    profiler: &mut crate::decode_profiler::DecodeProfiler,
    depth: usize,
    mut ev_depth_policy: Option<
        &mut crate::models::qwen3_5::adaptive_depth::ExpectedValueDepthPolicy,
    >,
    commit_anchor: MtpCommitAnchor,
) -> Result<(MtpCycleOutcome, MxArray)>
where
    F: FnMut(&MxArray, &MxArray) -> Result<(MxArray, MxArray, bool)>,
    D: FnMut(&MxArray, &MxArray) -> Result<(MxArray, MxArray)>,
    V: FnMut(&MxArray, &MxArray, usize) -> Result<MtpVerifyOutput>,
    R: FnMut(usize, usize),
    E: Fn(&MxArray, &MxArray, bool),
    EX: Fn(&MxArray, &MxArray),
    B: FnMut(bool),
    S: FnMut(),
    RR: FnMut(&[u32], &MxArray) -> Result<()>,
    CM: FnMut(MtpCommitAnchor, &MxArray, &MxArray, &[u32], usize, &MxArray) -> Result<()>,
    RU: FnMut(usize),
{
    use crate::array::{DType, MxArray as A};
    use crate::sampling;

    debug_assert!(depth >= 1, "run_mtp_cycle_inner: depth must be >= 1");

    // Keep the ORIGINAL cycle-seed hidden alive for the committed-history
    // commit. `prev_hidden_in` is h(token before `last_committed_id`) —
    // the correct hidden to pair with the
    // embedding of `last_committed_id` for that token's MTP slot. The
    // draft loop below moves `prev_hidden_in` into the mutable
    // `prev_hidden` local and overwrites it step by step, so clone the
    // (cheap, refcounted) handle now before that happens.
    let commit_seed_hidden = prev_hidden_in.clone();

    // Step 1: D draft steps via the per-step `draft_step` loop.
    profiler.begin("mtp_draft_total");
    let temperature = params
        .sampling_config
        .and_then(|c| c.temperature)
        .unwrap_or(1.0);
    let sampling_cfg = params.sampling_config.unwrap_or_default();
    let draft_sampling_cfg = mtp_draft_sampling_config(sampling_cfg);
    // Fast-path eligibility: at T=0 with all penalties at defaults, the
    // per-position accept decision collapses to
    // `argmax(verify_logits[i]) == draft_id[i]` (the argmax shortcut in
    // `accept_with_residual`). Compute this before draft construction so
    // the deterministic path can avoid building unused draft probability
    // tensors.
    let penalties_no_op = params.repetition_penalty == 1.0
        && params.presence_penalty == 0.0
        && params.frequency_penalty == 0.0;
    let use_sparse_accept =
        sparse_accept_gate() && sampling::is_greedy_temperature(temperature) && penalties_no_op;
    let use_sparse_stochastic_accept = mtp_batch_target_arrays_enabled()
        && !sampling::is_greedy_temperature(temperature)
        && penalties_no_op
        && sampling::sparse_distribution_supported(&sampling_cfg)
        && sampling::sparse_distribution_supported(&draft_sampling_cfg);
    let mut prev_hidden = prev_hidden_in;
    let mut prev_emb = prev_emb_in;
    let mut draft_ids: Vec<i32> = Vec::with_capacity(depth);
    let mut draft_probs: Vec<MxArray> = if use_sparse_accept || use_sparse_stochastic_accept {
        Vec::new()
    } else {
        Vec::with_capacity(depth)
    };
    let mut draft_sparse_probs: Vec<sampling::SparseDistribution> = if use_sparse_stochastic_accept
    {
        Vec::with_capacity(depth)
    } else {
        Vec::new()
    };
    // `step_input_id` is the token whose hidden/embedding seed this
    // draft step: `last_committed_id` for step 0, then each prior
    // drafted id. Logged per step so a debug run can reconstruct
    // the full draft chain.
    let mut step_input_id = last_committed_id as i32;
    for step in 0..depth {
        let (h_next, draft_logits) = (ops.draft_step)(&prev_hidden, &prev_emb)?;
        let logits_1d = if use_sparse_accept {
            None
        } else {
            // draft_logits is [1, vocab]; squeeze to [vocab] for the
            // probability distribution consumed by accept/reject.
            Some(draft_logits.squeeze(Some(&[0]))?)
        };
        let probs = if use_sparse_accept || use_sparse_stochastic_accept {
            None
        } else {
            // The legacy stochastic accept path consumes this `probs` as
            // the proposal density `q` inside `accept_with_residual`
            // (`min(1, p/q)` + `(p - q)+` residual). For Leviathan-Chen
            // exactness `q` MUST be the distribution the
            // draft token was actually drawn from. The draft id below (T>0
            // branch) is drawn via `sampling::sample(&draft_logits, ..)`
            // → `mlx_compiled_sample_full`, which converts logits→logprobs,
            // applies the top_k/top_p/min_p filters ON THE LOGPROBS, then
            // applies temperature ONLY at the final categorical draw.
            //
            // A `softmax(apply_sampling(logits))` rebuild did NOT match that
            // draw: `apply_sampling` scales by temperature FIRST and then
            // filters (and it ERRORS at T=0 because `apply_temperature`
            // rejects `temperature <= 0`). Build `q` from the SAME compiled
            // filter chain instead, via `sampling::sampling_distribution`,
            // which returns `softmax(filtered_logits / temperature)` under the
            // active `sampler_parity_mode()` — matching the draw by
            // construction for ALL configs (incl. the common `top_k==0` plain
            // temperature/top_p case) and both parity modes.
            //
            // NOTE: at T=0 the legacy `else` accept branch is only reached
            // when `MLX_MTP_SPARSE_ACCEPT` is disabled; in that case
            // `accept_with_residual` takes its argmax-only shortcut and never
            // reads `q`. `sampling_distribution` at T=0 returns the (valid,
            // 1D `[vocab]`) one-hot argmax distribution — it does NOT error,
            // and is ignored by the accept shortcut — so every T=0 commit
            // decision stays byte-identical. Only the T>0 probability-ratio
            // path is corrected.
            let raw_1d = logits_1d.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "MTP draft logits_1d unexpectedly None (sparse-accept gating mismatch)",
                )
            })?;
            // `sample()` at the draw site uses `params.sampling_config` (the
            // target config), so build `q` from the SAME config — not
            // `draft_sampling_cfg`, which only feeds the sparse path's draw.
            Some(
                sampling::sampling_distribution(raw_1d, params.sampling_config)?
                    .astype(DType::Float32)?,
            )
        };
        let mut sparse_draft = None;
        let tok_id = if use_sparse_stochastic_accept {
            let sparse_rows = sampling::sparse_distributions_from_logits(
                logits_1d.as_ref().ok_or_else(|| {
                    Error::from_reason(
                        "MTP draft logits_1d unexpectedly None (sparse-accept gating mismatch)",
                    )
                })?,
                &draft_sampling_cfg,
            )?
            .ok_or_else(|| {
                Error::from_reason(
                    "MTP sparse stochastic draft path became ineligible after gating",
                )
            })?;
            let draft_dist = sparse_rows.row_owned(0)?;
            let sampled = draft_dist.as_row().sample(rng)?;
            sparse_draft = Some(draft_dist);
            sampled
        } else {
            // Sample the drafted token using the same sampling pipeline
            // the main path uses — drafter and verifier must agree on
            // their proposal distribution for Leviathan-Chen.
            let tok = sampling::sample(&draft_logits, params.sampling_config)?;
            tok.eval();
            tok.item_at_int32(0)?
        };
        let draft_metrics = crate::models::qwen3_5::adaptive_depth::DraftMetrics {
            top1_prob_topk: sparse_draft
                .as_ref()
                .and_then(|dist| dist.as_row().top_entry().map(|(_, prob)| prob)),
        };
        tracing::trace!(
            target: "mlx_core::mtp::draft",
            step,
            input_id = step_input_id,
            drafted_id = tok_id,
            "MTP per-step draft"
        );
        draft_ids.push(tok_id);
        if let Some(sparse_draft) = sparse_draft {
            draft_sparse_probs.push(sparse_draft);
        }
        if let Some(probs) = probs {
            draft_probs.push(probs);
        }
        // Keep the draft step's hidden/embedding handles alive even if the
        // EV gate stops here. The fixed-depth path always retains these
        // handles through the cycle tail; matching that lifetime matters
        // for MLX's lazy compiled cache writes.
        prev_hidden = h_next;
        let id_arr = A::from_int32(&[tok_id], &[1])?;
        let emb_2d = embedding_weight.take(&id_arr, 0)?; // [1, hidden]
        let hidden = emb_2d.shape_at(1)?;
        prev_emb = emb_2d.reshape(&[1, 1, hidden])?;
        step_input_id = tok_id;
        if let Some(policy) = ev_depth_policy.as_mut()
            && draft_ids.len() < depth
        {
            profiler.begin("mtp_draft_gate");
            let decision =
                policy.should_continue_after_draft(draft_ids.len(), depth, draft_metrics);
            profiler.end();
            tracing::trace!(
                target: "mlx_core::mtp::adaptive",
                drafted_depth = draft_ids.len(),
                next_depth = decision.next_depth,
                expected_extra_accept = decision.expected_extra_accept,
                required_extra_accept = decision.required_extra_accept,
                continue_drafting = decision.continue_drafting,
                "MTP EV depth gate"
            );
            if !decision.continue_drafting {
                break;
            }
        }
    }
    profiler.end();
    let effective_depth = draft_ids.len();
    debug_assert!(
        effective_depth >= 1,
        "MTP EV depth gate must leave at least one draft token"
    );
    // `trace!` not `debug!` — the full `draft_ids` vector is per-token
    // detail; one record per cycle would flood a long decode at debug.
    tracing::trace!(
        target: "mlx_core::mtp",
        depth,
        effective_depth,
        draft_ids = ?draft_ids,
        "MTP draft phase complete"
    );

    // Step 2: build verify input [last_committed_id, d_0, ..., d_{D-1}].
    let mut verify_ids: Vec<i32> = Vec::with_capacity(effective_depth + 1);
    verify_ids.push(last_committed_id as i32);
    verify_ids.extend(draft_ids.iter().copied());
    let verify_in = A::from_int32(&verify_ids, &[1, (effective_depth + 1) as i64])?;
    // `trace!` not `debug!` — the full `verify_ids` vector is per-token
    // detail; keep debug to compact once-per-cycle summaries.
    tracing::trace!(
        target: "mlx_core::mtp",
        depth,
        effective_depth,
        last_committed_id,
        verify_ids = ?verify_ids,
        "MTP verify input built"
    );
    // Snapshot the main path's GDN linear caches + offset BEFORE verify
    // runs its D+1 sequential forwards. Verify mutates `g_compiled_caches`
    // in place; on rejection we restore from this snapshot and replay only
    // the K accepted drafts so the linear recurrent state matches the
    // committed token stream. On full accept the snapshot is discarded —
    // verify already left the linear state correctly advanced.
    profiler.begin("mtp_tape_snapshot");
    (ops.snapshot_main_linear)();
    profiler.end();
    tracing::trace!(
        target: "mlx_core::mtp",
        depth,
        "MTP main-linear caches + offset snapshot taken (pre-verify)"
    );
    // Verify returns BOTH logits and per-position hiddens.
    // Logits: `[1, depth+1, vocab]`; hiddens: `[1, depth+1, hidden]`.
    // We hold off on slicing the hidden until after the accept loop
    // computes K (= number of accepted drafts) so we can pick
    // `verify_hiddens[:, K, :]` — the correct prediction context for
    // the next cycle's first MTP draft.
    // The gap between `mtp_cycle` and this floor is the headroom
    // available to algorithmic work.
    let verify_only_t0 = std::time::Instant::now();
    profiler.begin("mtp_verify_dispatch");
    let trace_logits = mtp_trace_logits();
    let trace_acceptance = mtp_trace_acceptance();
    let use_native_sparse_verify = use_sparse_stochastic_accept
        && mtp_native_sparse_verify_enabled()
        && sampling::sampler_parity_is_mtplx()
        && !trace_logits;
    let use_greedy_argmax_only_verify = use_sparse_accept
        && mtp_greedy_argmax_only_verify_enabled()
        && !trace_logits
        && !trace_acceptance
        && !mtp_verify_top1_check_enabled();
    let verify_step_res = if use_greedy_argmax_only_verify
        && let Some(ref mut verify_step_argmax_only) = ops.verify_step_argmax_only
    {
        profiler.begin("mtp_verify_dispatch_argmax_only");
        let res = verify_step_argmax_only(&verify_in, embedding_weight, effective_depth);
        profiler.end();
        res
    } else if use_native_sparse_verify
        && let Some(ref mut verify_step_sparse) = ops.verify_step_sparse
    {
        verify_step_sparse(&verify_in, embedding_weight, effective_depth, &sampling_cfg)
    } else {
        (ops.verify_step)(&verify_in, embedding_weight, effective_depth)
    };
    profiler.end();
    let MtpVerifyOutput {
        logits: verify_logits,
        hiddens: verify_hiddens,
        target_argmax: verify_target_argmax,
        target_sparse: verify_target_sparse,
    } = verify_step_res?;
    tracing::debug!(
        target: "mlx_core::mtp",
        depth = effective_depth,
        requested_depth = depth,
        verify_tokens = effective_depth + 1,
        "MTP verify dispatched (batched target forward over depth+1 tokens)"
    );
    // Async-eval over verify outputs. By default we dispatch verify
    // (logits + hiddens) via `async_eval` instead of the synchronous
    // `eval()` below. The kernel launch returns immediately, letting the
    // CPU construct the accept loop's penalty / softmax / slice graph
    // while the verify command buffer is still executing on the GPU. The
    // first downstream `eval()` (the accept loop's `p_target.eval()` at
    // the per-position softmax) syncs on completion. Semantic equivalent
    // of MTPLX's `LAZY_VERIFY_LOGITS` (`MTPLX/mtplx/generation.py:49,
    // 3894`).
    //
    // We batch `verify_hiddens` into the same async_eval call so MLX's
    // scheduler can fuse it with the verify logits graph (they share
    // the per-position `final_norm` outputs). Only the post-accept
    // `verify_hiddens[:, K, :]` slice is actually realised on-device
    // by the chained-cycle path; for the default Step-A path the
    // batch eval is still cheap (one extra command-buffer entry).
    //
    // `MLX_MTP_VERIFY_ASYNC_EVAL=0` reverts to the synchronous
    // `verify_logits.eval()` barrier — byte-identical for
    // parity-debugging or hardware where the overlap budget is negligible.
    // Fast-path acceptance. When eligible, collapse the D+1 per-position
    // softmax materializations into ONE batched
    // `argmax(verify_logits, axis=-1)` op + one `.eval()` reading
    // D+1 int32 values.
    //
    // Why this is safe:
    //   * T=0 → `accept_with_residual` only reads `argmax(p_target)`
    //     vs `draft_id`. `softmax` is monotone so `argmax(softmax(x))
    //     == argmax(x)`. No probabilities are ever consumed.
    //   * Penalties default → `apply_all_penalties` is the identity,
    //     so `hist_extended` does NOT affect the per-position logits.
    //     We can compute all D+1 argmaxes BEFORE the accept loop.
    //   * Bonus token on full-accept = argmax at position D, also a
    //     trivial readout from the same batched array.
    //
    // When ineligible (T>0, or any penalty non-default), fall through to
    // the legacy per-position path below.

    let sparse_verify_argmax = if use_sparse_accept {
        verify_target_argmax.as_ref()
    } else {
        None
    };
    let verify_logits_ref = verify_logits.as_ref();

    profiler.begin("mtp_verify_eval");
    let defer_hidden = mtp_defer_verify_hidden_eval();
    let target_distribution_first = use_sparse_stochastic_accept
        && defer_hidden
        && mtp_target_distribution_first_enabled()
        && verify_logits_ref.is_some()
        && !trace_logits;
    if target_distribution_first {
        tracing::debug!(
            target: "mlx_core::mtp::verify_async_eval",
            depth = effective_depth,
            requested_depth = depth,
            "W6.23 target-distribution-first verify scheduling"
        );
    } else if mtp_verify_async_eval() {
        tracing::debug!(
            target: "mlx_core::mtp::verify_async_eval",
            depth = effective_depth,
            requested_depth = depth,
            defer_hidden,
            "W6.9 async_eval verify outputs"
        );
        if let Some(argmax_arr) = sparse_verify_argmax {
            let mut eval_arrays: Vec<&MxArray> =
                Vec::with_capacity(1 + usize::from(trace_logits) + usize::from(!defer_hidden));
            eval_arrays.push(argmax_arr);
            if trace_logits && let Some(verify_logits) = verify_logits_ref {
                eval_arrays.push(verify_logits);
            }
            if !defer_hidden {
                eval_arrays.push(&verify_hiddens);
            }
            MxArray::async_eval_arrays(&eval_arrays);
        } else if let Some(verify_logits) = verify_logits_ref {
            if defer_hidden {
                MxArray::async_eval_arrays(&[verify_logits]);
            } else {
                MxArray::async_eval_arrays(&[verify_logits, &verify_hiddens]);
            }
        } else if !defer_hidden {
            MxArray::async_eval_arrays(&[&verify_hiddens]);
        }
    } else {
        // We materialize logits now so per-position slicing reads
        // from a CPU-resident buffer for penalty application. The
        // hiddens ride on the same compiled graph; we only eval the
        // K-th slice below.
        //
        // Note: the sparse-accept path also benefits from this eager
        // eval — folding verify materialization into the accept-loop
        // argmax op (one combined sync) measured ~10% slower than two
        // separate syncs. The eager eval here lets MLX's scheduler
        // pipeline the verify command buffer with the subsequent argmax
        // dispatch build, which the combined-eval variant defeats. Kept
        // unconditional.
        if let Some(argmax_arr) = sparse_verify_argmax {
            argmax_arr.eval();
            if trace_logits && let Some(verify_logits) = verify_logits_ref {
                verify_logits.eval();
            }
        } else if let Some(verify_logits) = verify_logits_ref {
            verify_logits.eval();
        } else if !defer_hidden {
            verify_hiddens.eval();
        }
        tracing::debug!(
            target: "mlx_core::mtp::verify_async_eval",
            depth = effective_depth,
            requested_depth = depth,
            sparse_argmax = sparse_verify_argmax.is_some(),
            "verify eval (synchronous; async-eval disabled)"
        );
    }
    profiler.end();
    profiler.record_duration("mtp_verify_floor", verify_only_t0.elapsed());
    let vocab = if let Some(verify_logits) = verify_logits_ref {
        verify_logits.shape_at(2)?
    } else if let Some(target_sparse) = verify_target_sparse.as_ref() {
        target_sparse.vocab_size() as i64
    } else {
        embedding_weight.shape_at(0)?
    };

    // Step 3: per-position accept/reject. Build extended history as
    // we accept; rejecting at position i halts the loop.
    let mut accepted_tokens: Vec<u32> = Vec::with_capacity(effective_depth + 1);
    let mut all_accepted = true;
    let mut rejection_residual: Option<i32> = None;

    if use_sparse_accept {
        // ONE batched argmax over all D+1 verify positions. Shape
        // `[1, D+1, vocab]` → `[1, D+1]` int32. At T=0 we care only
        // about per-position argmax — no full-vocab softmax
        // materialization needed.
        //
        // `verify_logits` may still be lazy from the verify dispatch
        // (especially under `MLX_MTP_VERIFY_ASYNC_EVAL=1`). The
        // `.eval()` below is the SINGLE sync point for the accept
        // loop — vs the legacy D × per-position `p_target.eval()`
        // path that forced D full-vocab softmaxes through Metal.
        profiler.begin("mtp_accept_argmax");
        let fallback_argmax;
        let argmax_arr = if let Some(argmax_arr) = sparse_verify_argmax {
            argmax_arr
        } else {
            let verify_logits = verify_logits_ref.ok_or_else(|| {
                Error::from_reason(
                    "MTP greedy sparse accept requires verifier logits or precomputed target argmax",
                )
            })?;
            fallback_argmax = verify_logits.argmax(-1, None)?;
            &fallback_argmax
        };
        argmax_arr.eval();

        // Extract D+1 int32s into a CPU buffer. `verify_logits` was
        // `[1, D+1, vocab]`; the argmax over the last axis yields
        // `[1, D+1]`. We read flat positions 0..=depth.
        let mut target_argmax: Vec<i32> = Vec::with_capacity(effective_depth + 1);
        for i in 0..=effective_depth {
            target_argmax.push(argmax_arr.item_at_int32(i)?);
        }
        if sparse_verify_argmax.is_some() && mtp_verify_top1_check_enabled() {
            let verify_logits = verify_logits_ref.ok_or_else(|| {
                Error::from_reason("MTP verifier top1 check requires verifier logits")
            })?;
            let fallback_argmax = verify_logits.argmax(-1, None)?;
            fallback_argmax.eval();
            for (i, &compiled_id) in target_argmax.iter().enumerate() {
                let fallback_id = fallback_argmax.item_at_int32(i)?;
                if compiled_id != fallback_id {
                    return Err(Error::from_reason(format!(
                        "MTP verifier top1 mismatch at slot {i}: compiled={compiled_id}, fallback={fallback_id}"
                    )));
                }
            }
        }
        profiler.end();

        // Accept loop runs entirely on CPU buffers — no further GPU
        // syncs. The Leviathan-Chen accept-reject coin is unused at
        // T=0 (deterministic argmax decision); `rng` is intentionally
        // not advanced, matching `accept_with_residual`'s T=0
        // shortcut (zero RNG consumed).
        profiler.begin("mtp_accept_loop");
        for i in 0..effective_depth {
            let target_id = target_argmax[i];
            let accept = target_id == draft_ids[i];
            if trace_acceptance {
                let top2 = verify_logits_ref.and_then(|verify_logits| {
                    verify_logits
                        .slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])
                        .and_then(|s| s.squeeze(Some(&[0, 1])))
                        .and_then(|v1d| trace_top2(&v1d, vocab))
                        .ok()
                });
                trace_acceptance_greedy(
                    effective_depth,
                    i,
                    token_history.len(),
                    last_committed_id,
                    draft_ids[i],
                    target_id,
                    accept,
                    top2.as_ref(),
                );
            }
            tracing::trace!(
                target: "mlx_core::mtp::accept",
                pos = i,
                draft_id = draft_ids[i],
                target_id,
                accepted = accept,
                "MTP sparse accept position"
            );
            if accept {
                let id_u = target_id as u32;
                accepted_tokens.push(id_u);
            } else {
                all_accepted = false;
                rejection_residual = Some(target_id);
                accepted_tokens.push(target_id as u32);
                break;
            }
        }
        if all_accepted {
            // Bonus token = argmax at position D. Same batched
            // array, no extra ops, no extra eval.
            let bonus_id = target_argmax[effective_depth] as u32;
            tracing::trace!(
                target: "mlx_core::mtp::accept",
                bonus_id,
                "MTP bonus token (full accept, sparse path)"
            );
            accepted_tokens.push(bonus_id);
        }
        profiler.end();
    } else if use_sparse_stochastic_accept {
        profiler.begin("mtp_accept_sparse_probs");
        let target_sparse_from_logits;
        let target_sparse = if let Some(rows) = verify_target_sparse.as_ref() {
            rows.validate_for_accept(effective_depth + 1, vocab as usize, &sampling_cfg)?;
            rows
        } else {
            let verify_logits = verify_logits_ref.ok_or_else(|| {
                Error::from_reason(
                    "MTP sparse stochastic target path requires verifier logits or precomputed sparse rows",
                )
            })?;
            target_sparse_from_logits =
                sampling::sparse_distributions_from_logits(verify_logits, &sampling_cfg)?
                    .ok_or_else(|| {
                        Error::from_reason(
                            "MTP sparse stochastic target path became ineligible after gating",
                        )
                    })?;
            &target_sparse_from_logits
        };
        profiler.end();

        // Exact stochastic accept loop over tiny CPU-side top-k distributions.
        // No per-position full-vocab softmax/eval; rejection residuals and the
        // full-accept bonus sample from the same precomputed target rows.
        profiler.begin("mtp_accept_loop");
        // `i` indexes several parallel collections (`target_sparse`,
        // `draft_sparse_probs`, `draft_ids`) and doubles as the trace `pos`,
        // so a single `enumerate()` over one of them would not be clearer.
        #[allow(clippy::needless_range_loop)]
        for i in 0..effective_depth {
            let target_p = target_sparse.row(i)?;
            let draft_q = draft_sparse_probs
                .get(i)
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "MTP sparse stochastic draft distribution missing at position {}",
                        i
                    ))
                })?
                .as_row();
            let (accept, out_tok) =
                sampling::accept_with_residual_sparse(target_p, draft_q, draft_ids[i], rng)?;
            if trace_acceptance {
                trace_acceptance_sparse(
                    "sparse_stochastic",
                    effective_depth,
                    i,
                    token_history.len(),
                    last_committed_id,
                    draft_ids[i],
                    target_p,
                    draft_q,
                    accept,
                    out_tok,
                );
            }
            tracing::trace!(
                target: "mlx_core::mtp::accept",
                pos = i,
                draft_id = draft_ids[i],
                out_tok,
                accepted = accept,
                "MTP sparse stochastic accept position"
            );
            if accept {
                let id_u = out_tok as u32;
                accepted_tokens.push(id_u);
            } else {
                all_accepted = false;
                rejection_residual = Some(out_tok);
                accepted_tokens.push(out_tok as u32);
                break;
            }
        }

        if all_accepted {
            let bonus_id = target_sparse.row(effective_depth)?.sample(rng)? as u32;
            tracing::trace!(
                target: "mlx_core::mtp::accept",
                bonus_id,
                "MTP bonus token (full accept, sparse stochastic path)"
            );
            accepted_tokens.push(bonus_id);
        }
        profiler.end();
    } else {
        let verify_logits = verify_logits_ref
            .ok_or_else(|| Error::from_reason("MTP legacy accept requires verifier logits"))?;
        let mut hist_extended: Vec<u32> = token_history.to_vec();
        // Legacy per-position path. Kept for T>0 (where residual
        // sampling needs the full target distribution) and for
        // penalty-active configurations (where `hist_extended`
        // mutates the per-position logits inside the loop).
        // Note: this wrap includes the full-accept bonus-token sample
        // (sample + eval), whereas the sparse-accept branch's bonus is
        // a CPU buffer read inside the same phase name.
        profiler.begin("mtp_accept_loop");
        for i in 0..effective_depth {
            // verify_logits[0, i, :] → [vocab]
            let v_slice = verify_logits.slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])?;
            let v_logits_1d = v_slice.squeeze(Some(&[0, 1]))?;
            let penalized = apply_all_penalties(v_logits_1d, &hist_extended, params)?;
            // The target density `p` consumed by `accept_with_residual`
            // (`min(1, p/q)` + `(p - q)+` residual) MUST match the
            // distribution the verify/bonus token is drawn from. The
            // bonus on full-accept (and the residual draw on rejection) is
            // sampled via `sampling::sample(&penalized, ..)` →
            // `mlx_compiled_sample_full`, which filters logprobs then applies
            // temperature at the categorical draw. A raw `softmax(penalized)`
            // (no temperature, no top_k/top_p/min_p) did NOT match that draw,
            // biasing accept/reject and the residual resample whenever
            // temperature != 1 and/or filters are active. Build `p` from the
            // SAME compiled filter chain via `sampling::sampling_distribution`.
            //
            // At T=0 `accept_with_residual` only reads `argmax(p_target)`;
            // `sampling_distribution` returns the one-hot argmax there, so the
            // argmax (and thus the T=0 commit decision) is byte-identical to
            // the prior `softmax` while never erroring at T=0.
            let p_target = sampling::sampling_distribution(&penalized, params.sampling_config)?
                .astype(DType::Float32)?;
            p_target.eval();

            let sampling_cfg = params.sampling_config.unwrap_or_default();
            let (accept, out_tok) = sampling::accept_with_residual(
                &p_target,
                &draft_probs[i],
                draft_ids[i],
                &sampling_cfg,
                rng,
            )?;
            if trace_acceptance
                && let Err(e) = trace_acceptance_dense(
                    effective_depth,
                    i,
                    token_history.len(),
                    last_committed_id,
                    draft_ids[i],
                    &p_target,
                    &draft_probs[i],
                    &sampling_cfg,
                    accept,
                    out_tok,
                )
            {
                trace_acceptance_emit(serde_json::json!({
                    "schema_version": 1,
                    "path": "legacy_dense",
                    "depth": effective_depth,
                    "requested_depth": depth,
                    "slot": i,
                    "position": token_history.len() + i,
                    "last_committed_id": last_committed_id,
                    "draft_id": draft_ids[i],
                    "accepted": accept,
                    "out_token": out_tok,
                    "error": e.reason,
                }));
            }
            tracing::trace!(
                target: "mlx_core::mtp::accept",
                pos = i,
                draft_id = draft_ids[i],
                out_tok,
                accepted = accept,
                "MTP legacy accept position"
            );
            if accept {
                let id_u = out_tok as u32;
                accepted_tokens.push(id_u);
                hist_extended.push(id_u);
            } else {
                all_accepted = false;
                rejection_residual = Some(out_tok);
                accepted_tokens.push(out_tok as u32);
                break;
            }
        }

        if all_accepted {
            // Step 4 (bonus): sample from verify position D (after all
            // drafts accepted). Apply penalties consistent with the
            // extended history.
            let i = effective_depth;
            let v_slice = verify_logits.slice(&[0, i as i64, 0], &[1, (i + 1) as i64, vocab])?;
            let v_logits_1d = v_slice.squeeze(Some(&[0, 1]))?;
            let penalized = apply_all_penalties(v_logits_1d, &hist_extended, params)?;
            let bonus = sampling::sample(&penalized, params.sampling_config)?;
            bonus.eval();
            let bonus_id = bonus.item_at_int32(0)? as u32;
            tracing::trace!(
                target: "mlx_core::mtp::accept",
                bonus_id,
                "MTP bonus token (full accept, legacy path)"
            );
            accepted_tokens.push(bonus_id);
        }
        profiler.end();
    }

    // Diagnostic — `MLX_MTP_TRACE_LOGITS=1` per-committed-token verify
    // top-2 logit trace. Runs AFTER the accept loop so it is read-only
    // and does not perturb the sparse/legacy accept hot path. Each
    // `accepted_tokens[j]` was committed from verify slot `j` of the
    // batched verify forward; `verify_logits` is `[1, depth+1, vocab]`.
    // The first `K` slots are accepted drafts; the final slot is the
    // boundary token (bonus on full accept, residual on rejection).
    // Position label `token_history.len() + j` aligns with the AR
    // loop's `$hist.len() + 1` numbering (same prompt base).
    if mtp_trace_logits() {
        let verify_logits = verify_logits_ref
            .ok_or_else(|| Error::from_reason("MTP_TRACE_LOGITS requires verifier logits"))?;
        for (j, &committed_id) in accepted_tokens.iter().enumerate() {
            let slot = j as i64;
            let source = if all_accepted && j + 1 == accepted_tokens.len() {
                "verify-bonus"
            } else if !all_accepted && j + 1 == accepted_tokens.len() {
                "verify-residual"
            } else {
                "verify-draft"
            };
            let v_slice_res = verify_logits
                .slice(&[0, slot, 0], &[1, slot + 1, vocab])
                .and_then(|s| s.squeeze(Some(&[0, 1])));
            match v_slice_res.and_then(|v1d| trace_top2(&v1d, vocab)) {
                Ok(t2) => {
                    eprintln!(
                        "MTP_TRACE_LOGITS source={} verify_slot={} pos={} \
                         token_id={} top1_id={} top1_logit={:.6} top2_id={} \
                         top2_logit={:.6} gap={:.6}",
                        source,
                        j,
                        token_history.len() + j,
                        committed_id,
                        t2.top1_id,
                        t2.top1_logit,
                        t2.top2_id,
                        t2.top2_logit,
                        t2.top1_logit - t2.top2_logit,
                    );
                }
                Err(e) => {
                    eprintln!(
                        "MTP_TRACE_LOGITS source={} verify_slot={} pos={} ERROR {}",
                        source,
                        j,
                        token_history.len() + j,
                        e.reason,
                    );
                }
            }
        }
    }

    // Step 5: rollback. `accepted_drafts` is the number of draft
    // tokens (out of `effective_depth`) whose K/V we are KEEPING in BOTH the
    // main and the MTP draft caches. The rest must be discarded.
    //
    // Layout BEFORE this cycle (right after the macro's Step A):
    //   - Main offset advanced by 1 (Step A wrote K/V for `y`, the
    //     prior cycle's last accepted token, at the next free slot).
    //   - MTP draft offset unchanged since the prior cycle's
    //     rollback (the MTP path mirrors a snapshot of the main
    //     offset and only moves on draft / rollback).
    //
    // Verify wrote K/V for ALL `effective_depth + 1` inputs of
    // `[last_committed_id, d_0, .., d_{effective_depth-1}]` into the
    // MAIN cache (advancing main offset by `effective_depth + 1`). Draft
    // steps wrote K/V for the `effective_depth` drafted tokens into the
    // MTP cache (advancing MTP offset by `effective_depth`).
    //
    //   - On full accept: ALL `effective_depth + 1` verify positions are kept
    //     in main (last_committed + `effective_depth` drafts) and ALL `effective_depth`
    //     draft positions are kept in MTP. The bonus token has no
    //     K/V written this cycle — its K/V will be laid down by the
    //     NEXT cycle's Step A.
    //   - On rejection after `K` accepted drafts: we keep the
    //     last_committed slot + the first `K` draft slots in main
    //     (= `K + 1` main verify slots) and the first `K` slots in
    //     MTP. The REJECTED draft's K/V is discarded by offset
    //     rewind in BOTH caches. The verifier's residual sample is
    //     emitted as a token but has no K/V written this cycle —
    //     its K/V will be laid down by the NEXT cycle's Step A.
    //
    // Both deltas reduce to `accepted_drafts - effective_depth`:
    //   - main_delta = (K + 1) - (effective_depth + 1) = K - effective_depth
    //   - mtp_delta  = K       - effective_depth
    let accepted_drafts = if all_accepted {
        effective_depth
    } else {
        // accepted_tokens contains `K` accepted drafts + 1 residual.
        accepted_tokens.len() - 1
    };
    if let Some(policy) = ev_depth_policy.as_mut() {
        policy.observe(effective_depth, accepted_drafts);
    }
    // Per-cycle acceptance: feeds the profiler's acceptance summary
    // (surfaced on `PerformanceMetrics` + the stderr report).
    profiler.record_mtp_cycle(effective_depth, accepted_drafts);
    tracing::debug!(
        target: "mlx_core::mtp",
        depth = effective_depth,
        requested_depth = depth,
        accepted_drafts,
        all_accepted,
        committed = accepted_tokens.len(),
        "MTP cycle accept result"
    );

    // Committed-history commit.
    //
    // Step-A cycles commit the full newly emitted sequence
    // `[last_committed_id] ++ accepted_tokens`: Step A sampled
    // `last_committed_id`, so it is not in the persistent MTP cache yet.
    //
    // Chained cycles skip Step A. Their `last_committed_id` is the prior
    // cycle's boundary token, already committed by that prior cycle. The
    // commit must therefore skip the anchor and append only
    // `accepted_tokens`, advancing `g_mtp_committed_len` by the number of
    // newly emitted tokens. Re-committing the anchor would drift the MTP
    // RoPE base by one slot per chained cycle.
    let committed_ids: Vec<u32> = match commit_anchor {
        MtpCommitAnchor::IncludeAnchor => {
            let mut ids = Vec::with_capacity(accepted_tokens.len() + 1);
            ids.push(last_committed_id);
            ids.extend(accepted_tokens.iter().copied());
            ids
        }
        MtpCommitAnchor::SkipAlreadyCommittedAnchor => accepted_tokens.clone(),
    };
    profiler.begin("mtp_commit");
    let commit_res = (ops.commit_mtp)(
        commit_anchor,
        &commit_seed_hidden,
        &verify_hiddens,
        &committed_ids,
        accepted_drafts,
        embedding_weight,
    );
    profiler.end();
    commit_res?;

    profiler.begin("mtp_rollback");
    (ops.rollback)(accepted_drafts, effective_depth);
    profiler.end();
    tracing::debug!(
        target: "mlx_core::mtp",
        accepted_drafts,
        depth = effective_depth,
        requested_depth = depth,
        offset_delta = accepted_drafts as i64 - effective_depth as i64,
        "MTP rollback applied"
    );

    // On rejection, restore the main path's GDN linear caches (back to
    // "after Step A": Step A processed `y_N` and the snapshot was taken
    // right after) and replay the K + 1 committed tokens that verify
    // processed but the restore discarded:
    //   * `last_committed_id` (= y_{N+1}, the token Step A sampled
    //     and the cycle treated as the verify-position-0 anchor),
    //   * `d_0..d_{K-1}` (the K accepted drafts).
    // The residual sample R is NOT replayed — its K/V will be laid
    // down by the NEXT outer iteration's Step A (it becomes `y` at
    // the loop boundary).
    //
    // Post-replay main offset = snapshot_offset + K + 1, matching
    // what the previous direct `adjust_offset(K - depth)` rollback
    // produced. Post-replay linear state = AR equivalent for the
    // `[y_N, y_{N+1}, d_0..d_{K-1}]` token prefix.
    //
    // On full accept the rollback hook receives `(accepted_drafts=depth,
    // depth)` and may still normalize the main linear state from the
    // recorded tape. The verifier's full window is logically kept, but
    // the dense GDN recurrent cache must remain byte-compatible with
    // serial AR across the next Step A.
    if !all_accepted {
        let mut replay_ids: Vec<u32> = Vec::with_capacity(accepted_drafts + 1);
        replay_ids.push(last_committed_id);
        // accepted_tokens = [d_0, .., d_{K-1}, residual]; we replay
        // only the K accepted drafts (NOT the residual).
        replay_ids.extend_from_slice(&accepted_tokens[..accepted_drafts]);
        tracing::debug!(
            target: "mlx_core::mtp",
            replay_token_count = replay_ids.len(),
            last_committed_id,
            "MTP tape replay (restore main caches + replay accepted prefix)"
        );
        profiler.begin("mtp_tape_replay");
        let replay_res = (ops.restore_and_replay_main)(&replay_ids, embedding_weight);
        profiler.end();
        replay_res?;
    }

    let _ = rejection_residual; // documented above; only used for clarity
    // `prev_hidden` / `prev_emb` are no longer needed (they were the
    // INPUTS to the cycle's drafts; the verify pass downstream of
    // them is already evaluated). They drop at end-of-function with
    // the rest of the locals; the underlying lazy MLX arrays stay
    // alive as long as any other handle still holds them.

    // Pick the position-K slice of `verify_hiddens` and return it so the
    // caller (the `decode_loop_mtp!` macro) can chain cycles: the NEXT
    // cycle's first MTP draft uses this hidden as `prev_hidden`,
    // eliminating the per-cycle main-model "Step A" forward.
    //
    // Semantics: `verify_hiddens[K]` is the post-final-norm hidden at
    // verify position K — the prediction context for the committed
    // token at position K+1 of `[last_committed, d_0, ..., d_{D-1}]`,
    // i.e. the BONUS token on full-accept (K=D, position K+1 = bonus's
    // would-be slot) or the RESIDUAL token on rejection (K<D, position
    // K+1 = rejected draft's slot, replaced by residual). Either way,
    // the next cycle's MTP draft gets `(prev_hidden=verify_hiddens[K],
    // prev_emb=embed(committed_K+1))` which matches the training
    // contract of the MTP head: `MTP(h_t, embed(t+1)) -> logits at
    // t+2`.
    //
    // Why K (not D, not D+1): position D only matches when ALL drafts
    // are accepted (K==D). Chaining a partial-accept cycle from
    // position D's hidden — the prediction context for the rejected
    // draft — diverges the MTP head's drafts from main, dropping mean
    // acceptance from ~1.5 to ~0.8 tokens/cycle.
    let hidden_dim = verify_hiddens.shape_at(2)?;
    let verify_hidden_k = verify_hiddens.slice(
        &[0, accepted_drafts as i64, 0],
        &[1, (accepted_drafts + 1) as i64, hidden_dim],
    )?;
    Ok((
        MtpCycleOutcome {
            tokens: accepted_tokens,
            requested_depth: depth,
            effective_depth,
        },
        verify_hidden_k,
    ))
}

/// MTP speculative decode loop. See `decode_loop!` for the
/// single-token sister macro this mirrors.
///
/// Required arguments mirror `decode_loop!`. Adds:
///   - `mtp_ops`: an [`MtpOps`] struct.
///   - `mtp_depth`: initial / fixed number of draft tokens per cycle
///     (`>= 1`, `<= 5`). When the adaptive policy is ON (see
///     `params.mtp_adaptive_depth`), this value seeds
///     `AdaptiveDepthPolicy::new(...)` and the per-cycle depth is then
///     chosen by `pick_depth()`. When adaptive is OFF, this value is
///     used unchanged for every cycle.
///   - `mtp_rng`: an `&mut impl rand::Rng` driving the acceptance
///     coin flip. The caller picks the seed strategy
///     (`rand::rng()` — thread-local CSPRNG — is the typical
///     production choice; `StdRng::seed_from_u64(seed)` is preferred
///     in tests for determinism).
///
/// The `streaming` block is OPTIONAL — same shape as `decode_loop!`.
macro_rules! decode_loop_mtp {
    (
        mtp_ops: $mtp:expr,
        mtp_depth: $depth:expr,
        mtp_rng: $rng:expr,
        y: $y:expr,
        embedding_weight: $emb:expr,
        params: $p:expr,
        reasoning_tracker: $tracker:expr,
        profiler: $profiler:expr,
        max_new_tokens: $max:expr,
        eos_id: $eos:expr,
        generated_tokens: $gen:expr,
        token_history: $hist:expr,
        finish_reason: $reason:expr,
        first_token_instant: $first_tok:expr,
        report_perf: $report:expr,
        generation_stream: $stream:expr
        $(, streaming: {
            callback: $cb:expr,
            cancelled: $cancelled:expr,
            decode_stream: $ds:expr,
            tokenizer: $tok:expr,
            streamed_text_len: $slen:expr,
            last_is_reasoning: $last_r:expr
        })?
    ) => {{
        // Emit the FIRST token via a normal main-path forward+hidden.
        // The MTP loop needs an established last-committed token AND
        // its post-final-norm hidden state to seed the first draft.
        // After this initial forward, `prev_hidden` / `prev_emb`
        // carry the seed for the next cycle.
        let mut prev_hidden_opt: Option<$crate::array::MxArray>;
        let mut prev_emb_opt: Option<$crate::array::MxArray>;
        let mut last_committed_id_opt: Option<u32>;

        // Chained-cycle state. `run_mtp_cycle_inner` slices
        // `verify_hiddens[:, K, :]` and returns it; we stash that
        // `[1, 1, hidden]` here so the NEXT outer iteration can skip
        // Step A's ~150 ms main-model forward and feed the chained
        // hidden directly into the cycle's first MTP draft.
        //
        // K = number of accepted drafts this cycle. Semantics:
        // `verify_hiddens[K]` is the prediction context for the
        // committed token at position K+1 (bonus on full-accept,
        // residual on rejection) — i.e. for the LAST emitted token of
        // this cycle. The next cycle's MTP draft is therefore
        // `MTP(prev_hidden=verify_hiddens[K], prev_emb=embed($y)) ->
        // next-next logits`, matching the head's training contract.
        //
        // Chaining is GPU-gen-gated (default ON M5+, OFF M1–M4); override
        // with `MLX_MTP_CHAINED_CYCLES=0/1`. The position-K slice makes it
        // SEMANTICALLY correct — byte-exact T=0 parity holds in both modes.
        //
        // Invariants:
        //   - `None` on the FIRST iteration (no prior verify) — Step A
        //     runs unconditionally and re-seeds the hidden from a real
        //     main forward.
        //   - `None` when forced-think-end fires — that path needs Step
        //     A's forward to write `$y`'s K/V before injecting the
        //     forced token. (See the force-end branch below.)
        //   - `Some(hidden)` after every successful cycle, to be drained
        //     by the NEXT iteration before its cycle runs.
        //
        // The hidden is a lazy MLX array referencing the verify's
        // position-K `final_norm` graph node; it stays alive because
        // `g_compiled_caches` (its upstream) is alive for the rest of
        // the decode loop. See `mlx_qwen35_mtp_verify_compiled_with_hidden`
        // for the C++ lifetime contract.
        let chained_cycles_enabled: bool =
            $crate::models::qwen3_5::chat_common::mtp_chained_cycles_enabled();
        let mut chained_hidden_opt: Option<$crate::array::MxArray> = None;

        // Adaptive MTP depth policy. When `mtp_adaptive_depth` is true
        // (explicit opt-in from ChatConfig), the policy picks the
        // per-cycle draft depth from a per-depth EMA of
        // `accepted_tokens / cycle_wall_ns` plus a DFlash-style 3-state
        // machine (`full | reduced | probe`). When false, the policy is
        // constructed but `pick_depth()` returns `$p.mtp_depth` on every
        // call (no transitions ever fire because `record_cycle` is gated
        // below).
        //
        // The compiled MTP verify graphs are pre-warmed at model load
        // for every depth `D ∈ {1..=5}`, so swinging the depth freely
        // between cycles is zero-cost from the compile side.
        let mut mtp_depth_policy =
            $crate::models::qwen3_5::adaptive_depth::AdaptiveDepthPolicy::new(
                $depth.min(u8::MAX as usize) as u8,
            );
        let mtp_adaptive_depth_mode =
            $crate::models::qwen3_5::adaptive_depth::adaptive_depth_mode_from_env();
        let mut mtp_ev_depth_policy =
            $crate::models::qwen3_5::adaptive_depth::ExpectedValueDepthPolicy::new(
                $depth.min(u8::MAX as usize) as u8,
            );

        // Track cycles for the every-256-emitted-token cache clear.
        // We use the running `gen.len()` rather than a separate step
        // counter so MTP and non-MTP loops stay byte-equivalent on
        // the cache-clear cadence.
        let mut last_clear_at: usize = $gen.len();

        // PARITY-FIX (budget): `$max` is the raw `max_new_tokens: i32`
        // with no upstream clamp on the chat/MTP path, so it can be `0`
        // or NEGATIVE (reachable via `ChatConfig.maxNewTokens` and
        // `/v1/responses` `max_output_tokens`). AR's `decode_loop!` uses
        // `for step in 0..$max` — an empty range for `$max <= 0` — and
        // therefore emits 0 tokens. The MTP loop below compares
        // `$gen.len()` against the budget via `as usize`; a NEGATIVE
        // `$max` would wrap to a huge `usize` and never trip the length
        // cap (effectively unbounded). Clamp negatives to 0 ONCE here
        // and use this value for every budget comparison so MTP matches
        // AR's "0 new tokens for a nonpositive budget" semantics. For
        // `$max >= 1` this is numerically identical to `($max as usize)`
        // ⇒ byte-for-byte identical behavior for valid budgets.
        let max_as_usize: usize = ($max).max(0) as usize;

        // PARITY-FIX: emit the initial `$y` (sampled from the prefill's
        // last logits BEFORE this macro was entered) before Step A's
        // first iteration. AR's `decode_loop!` macro emits its input
        // `$y` at the top of each iteration; MTP's Step A only emits
        // the SAMPLED next token, which means the very first token of
        // the generation (the prefill's seed sample) never reached
        // `$gen`. Without this push MTP's output is the AR output
        // shifted left by one token. We mirror the per-token bookkeeping
        // Step A does (eval, stream
        // callback, tracker.observe_token, profiler) so the initial
        // token participates identically. The stop checks (EOS,
        // length, cancel, repetition) run at the top of the loop body
        // below — they read `$gen` so the initial push is visible.
        //
        // Guarded on the budget: at macro entry `$gen` holds only
        // generated tokens (0 here), so `$gen.len() < max_as_usize` is
        // `0 < 0 == false` when `$max <= 0` ⇒ NO initial push, matching
        // AR. For `$max >= 1` the guard is `0 < max` (true) ⇒ the push
        // runs exactly as before.
        if $gen.len() < max_as_usize {
            let _stream_ctx = $crate::stream::StreamContext::new($stream);
            $profiler.begin("extract");
            $y.eval();
            let initial_token_id = $y.item_at_int32(0)? as u32;
            $profiler.end();
            $profiler.mark_first_token();
            if $report && $first_tok.is_none() {
                $first_tok = Some(std::time::Instant::now());
            }
            $gen.push(initial_token_id);
            $hist.push(initial_token_id);
            let _is_reasoning = $tracker.observe_token(initial_token_id);
            $(
                $last_r = _is_reasoning;
                if !$cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                    let token_text = $crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                        &mut $ds, $tok.inner(), initial_token_id, &$gen, $slen,
                    );
                    $slen += token_text.len();
                    // Suppress reasoning (<think>…</think>) deltas from the stream
                    // when include_reasoning == false, matching the AR decode loop.
                    // Detokenize + length-advance stay OUTSIDE this gate so
                    // DecodeStream sees every token.
                    if $p.include_reasoning || !_is_reasoning {
                        $cb.call(
                            Ok($crate::models::qwen3_5::model::ChatStreamChunk {
                                text: token_text, done: false, finish_reason: None,
                                tool_calls: None, thinking: None, num_tokens: None,
                                prompt_tokens: None, reasoning_tokens: None,
                                raw_text: None, cached_tokens: None, performance: None,
                                is_reasoning: Some(_is_reasoning),
                            }),
                            napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                        );
                    }
                }
            )?
            $profiler.step();
        }

        loop {
            // Zero budget (nonpositive clamped to 0): AR's `for step in 0..$max`
            // never iterates and never observes cancel/EOS/repetition, so its
            // finish_reason stays "length". `max_as_usize` is loop-invariant, so
            // for any budget >= 1 this is a dead branch (no behavior change);
            // only a 0 budget short-circuits here, before the cancelled check.
            if max_as_usize == 0 {
                if $reason.is_empty() { $reason = String::from("length"); }
                break;
            }
            // PARITY-FIX: re-check the same stop conditions Step A
            // uses, BEFORE the forward, so the initial push (above)
            // and any prior-iteration push that landed us on a stop
            // condition exit cleanly without one more forward.
            if let Some(&last) = $gen.last() {
                if last == $eos {
                    $reason = String::from("stop");
                    break;
                }
            }
            $(
                if $cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                    $reason = String::from("cancelled");
                    break;
                }
            )?
            if let Some(reason) = $crate::sampling::check_repetition_cutoff(
                &$gen, $p.max_consecutive_tokens, $p.max_ngram_repeats, $p.ngram_size,
            ) {
                $reason = reason.to_string();
                break;
            }
            if $gen.len() >= max_as_usize {
                if $reason.is_empty() { $reason = String::from("length"); }
                break;
            }

            // ---- Step A vs. chained-hidden decision. ------------------
            // When chained cycles are enabled (`chained_cycles_enabled`,
            // GPU-generation-gated: default ON on M5+/gen>=17, OFF on
            // M1–M4; override via `MLX_MTP_CHAINED_CYCLES=0/1`): skip
            // Step A's full main-model forward when a chained verify
            // hidden is available from the prior cycle, unless the
            // tracker is about to force a think-end token (the
            // forced-token path needs Step A to forward `$y` so its K/V
            // is committed before we inject the forced token). The gate
            // is a non-consuming `force_think_end_pending()` peek so the
            // pending flag survives into Step A's single consume below.
            //
            // When chained cycles are disabled (M1–M4 default, or
            // `MLX_MTP_CHAINED_CYCLES=0`): always Step A, byte-exact with
            // the non-chained path. On M1–M4 the chained path still
            // regresses depth-3 acceptance (a lazy-slice eval-scheduling
            // stall), see the comment block at the top of
            // `decode_loop_mtp!` for details.
            //
            // On the chained path the prior cycle's verify already
            // committed all accepted tokens' K/V, and the next cycle's
            // verify will write `$y`'s K/V at its position-0 input.
            // The MTP draft seeds from `chained_hidden_opt`
            // (`verify_hiddens[K]` — the prediction context for the
            // committed token at position K+1, i.e. $y itself). T=0
            // parity is preserved because verify (= main model) is the
            // ground truth and at T=0 the residual-sampler picks the
            // same token regardless of draft accuracy.
            let do_step_a = !chained_cycles_enabled
                || chained_hidden_opt.is_none()
                || $tracker.force_think_end_pending();
            let cycle_seed_was_chained = !do_step_a;

            let _stream_ctx = $crate::stream::StreamContext::new($stream);

            if do_step_a {
                $profiler.begin("forward");
                let next_ids = $y.reshape(&[1, 1])?;
                let (mut logits, hidden, needs_squeeze) =
                    ($mtp.forward_with_hidden)(&next_ids, &$emb)?;
                if needs_squeeze {
                    logits = logits.squeeze(Some(&[1]))?;
                }
                $profiler.end();

                let (next_token, budget_forced) =
                    if $tracker.should_force_think_end() {
                        let forced_id = $tracker.forced_token_id()? as i32;
                        tracing::debug!(
                            target: "mlx_core::mtp",
                            forced_id,
                            "MTP Step A: forcing think-end token (reasoning budget tripped)"
                        );
                        ($crate::array::MxArray::from_int32(&[forced_id], &[1])?, true)
                    } else {
                        $profiler.begin("rep_penalty");
                        logits = $crate::models::qwen3_5::chat_common::apply_all_penalties(
                            logits, &$hist, &$p,
                        )?;
                        $profiler.end();

                        $profiler.begin("sample");
                        let t = $crate::sampling::sample(&logits, $p.sampling_config)?;
                        $profiler.end();
                        (t, false)
                    };

                $profiler.begin("eval_caches");
                ($mtp.eval_step)(&next_token, &logits, budget_forced);
                $profiler.end();

                $profiler.begin("eval_token");
                next_token.eval();
                $profiler.end();

                $profiler.begin("extract");
                let token_id = next_token.item_at_int32(0)? as u32;
                $profiler.end();
                $profiler.mark_first_token();
                if $report && $first_tok.is_none() {
                    $first_tok = Some(std::time::Instant::now());
                }

                $gen.push(token_id);
                $hist.push(token_id);
                let _is_reasoning = $tracker.observe_token(token_id);

                $(
                    $last_r = _is_reasoning;
                    if $cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                        $reason = String::from("cancelled");
                        break;
                    }
                    let token_text = $crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                        &mut $ds, $tok.inner(), token_id, &$gen, $slen,
                    );
                    $slen += token_text.len();
                    // Suppress reasoning deltas when include_reasoning == false,
                    // matching the AR decode loop (detokenize + length-advance
                    // above stay outside the gate so DecodeStream sees every token).
                    if $p.include_reasoning || !_is_reasoning {
                        $cb.call(
                            Ok($crate::models::qwen3_5::model::ChatStreamChunk {
                                text: token_text, done: false, finish_reason: None,
                                tool_calls: None, thinking: None, num_tokens: None,
                                prompt_tokens: None, reasoning_tokens: None,
                                raw_text: None, cached_tokens: None, performance: None,
                                is_reasoning: Some(_is_reasoning),
                            }),
                            napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                        );
                    }
                )?

                if token_id == $eos {
                    $reason = String::from("stop");
                    break;
                }
                if let Some(reason) = $crate::sampling::check_repetition_cutoff(
                    &$gen, $p.max_consecutive_tokens, $p.max_ngram_repeats, $p.ngram_size,
                ) {
                    $reason = reason.to_string();
                    break;
                }
                if $gen.len() >= max_as_usize {
                    if $reason.is_empty() { $reason = String::from("length"); }
                    break;
                }

                // Seed for MTP cycles using the hidden returned from this
                // forward. `hidden` is `[1, hidden_size]`; reshape to
                // `[1, 1, hidden]` for the draft FFI's `[B, T, hidden]`
                // contract.
                let hidden_dim = hidden.shape_at(1)?;
                prev_hidden_opt = Some(hidden.reshape(&[1, 1, hidden_dim])?);
                // prev_emb is the embedding of the JUST-emitted token.
                let id_arr = $crate::array::MxArray::from_int32(&[token_id as i32], &[1])?;
                let emb_2d = $emb.take(&id_arr, 0)?;
                let h = emb_2d.shape_at(1)?;
                prev_emb_opt = Some(emb_2d.reshape(&[1, 1, h])?);
                last_committed_id_opt = Some(token_id);
                $y = next_token;
            } else {
                // ---- Chained path: skip Step A entirely. --------------
                // `$y` already holds the prior cycle's last accepted
                // token (set by that cycle's tail update). That token
                // has already been pushed to `$gen` / `$hist` /
                // `tracker` AND streamed to the callback. Its K/V will
                // be written by THIS cycle's verify at position 0.
                //
                // We just need to seed the cycle's MTP draft inputs
                // from the chained hidden and the embedding of `$y`.
                let chained_h = chained_hidden_opt.take().ok_or_else(|| {
                    napi::Error::from_reason(
                        "chained_hidden_opt is Some on the chained path \
                         (guarded by do_step_a)",
                    )
                })?;
                // `run_mtp_cycle_inner` already sliced the K-th
                // position out of the verify hiddens, so `chained_h`
                // arrives shaped `[1, 1, hidden]` — the same shape the
                // draft FFI's `[B, T, hidden]` contract expects, no
                // reshape needed.
                prev_hidden_opt = Some(chained_h);

                // Read `$y`'s id without re-evaluating it; the prior
                // cycle tail already ran `MxArray::from_int32(...)` to
                // produce a fully materialised `[1]` int32 array, so
                // `item_at_int32(0)` here is a CPU-only read.
                $y.eval();
                let token_id = $y.item_at_int32(0)? as u32;

                let id_arr = $crate::array::MxArray::from_int32(&[token_id as i32], &[1])?;
                let emb_2d = $emb.take(&id_arr, 0)?;
                let h = emb_2d.shape_at(1)?;
                prev_emb_opt = Some(emb_2d.reshape(&[1, 1, h])?);
                last_committed_id_opt = Some(token_id);
                // Note: no `$y =` assignment — `$y` is already correct.
                // No tracker.observe_token / no $gen.push / no callback —
                // the prior cycle's emit loop already handled all of
                // that for the same `token_id`.
            }
            $profiler.step();

            // ---- Step B: ONE MTP draft+verify cycle. -------------------
            // On the chained path the prior verify already committed
            // bonus/residual; this cycle's verify writes the chained
            // `$y`'s K/V at position 0 and extends the prefix by D more
            // drafts. On full accept per cycle we emit D+1 tokens for
            // D draft steps + 1 verify (one fewer main forward when
            // chaining).
            if $gen.len() >= max_as_usize {
                if $reason.is_empty() { $reason = String::from("length"); }
                break;
            }
            if $tracker.force_think_end_pending() {
                // Budget tripped during Step A's observe (after Step A's
                // consume) — defer the forced token to the NEXT cycle's
                // Step A. This is a NON-consuming peek: the flag stays set,
                // so next cycle's routing peek (`do_step_a`) forces Step A
                // and the single consuming call there emits `</think>`.
                tracing::debug!(
                    target: "mlx_core::mtp",
                    "MTP cycle skipped: think-end queued, deferring to next Step A"
                );
                continue;
            }

            let prev_h = prev_hidden_opt.take().ok_or_else(|| {
                napi::Error::from_reason("prev_hidden seeded by Step A or chained path")
            })?;
            let prev_e = prev_emb_opt.take().ok_or_else(|| {
                napi::Error::from_reason("prev_emb seeded by Step A or chained path")
            })?;
            let last_id = last_committed_id_opt.ok_or_else(|| {
                napi::Error::from_reason("last_committed seeded by Step A or chained path")
            })?;
            // Re-anchor the MTP cache to the main path's CURRENT offset
            // before launching this cycle's drafts. On the Step-A path
            // the main offset has
            // advanced by 1 (Step A's forward) + the prior cycle's
            // verify advancement. On the chained path the main offset
            // has only advanced by the prior cycle's verify (Step A
            // was skipped). EITHER way, this resets the MTP K/V and
            // sets the MTP offset = current main offset, which is
            // exactly the contract `begin_cycle` is documented to
            // honour. Without it the MTP draft RoPE positions diverge
            // and drafts produce gibberish.
            // The `begin_cycle` closure emits its own
            // `mlx_core::mtp` trace (old/new MTP offset) — it is the
            // only site that knows the dense-vs-MoE offset getters.
            ($mtp.begin_cycle)(cycle_seed_was_chained && $mtp.committed_history_active);
            // Per-cycle depth selection. When adaptive is OFF,
            // `pick_depth()` returns the seed depth unchanged
            // (`record_cycle` is gated below). When adaptive is ON, the
            // policy hill-climbs across depth-EMA + manages the
            // `full | reduced | probe` state machine.
            let cycle_depth: usize = if $p.mtp_adaptive_depth {
                match mtp_adaptive_depth_mode {
                    $crate::models::qwen3_5::adaptive_depth::AdaptiveDepthMode::Throughput => {
                        mtp_depth_policy.pick_depth() as usize
                    }
                    $crate::models::qwen3_5::adaptive_depth::AdaptiveDepthMode::ExpectedValue => {
                        mtp_ev_depth_policy.max_depth() as usize
                    }
                }
            } else {
                $depth
            };
            // Near-tail budget cap. The compiled verify writes `depth+1`
            // target-cache slots BEFORE the post-verify truncation to
            // `max_new_tokens`; when fewer than `depth+1` main-cache
            // slots remain near the tail the write can overrun the
            // rounded `max_kv_len` allocation. Cap the effective cycle
            // depth so the verify never needs more main-cache slots than
            // the remaining generation budget can absorb. `remaining` is
            // `>= 1` here (the `$gen.len() >= $max` check above already
            // broke the loop otherwise). With `effective_depth =
            // remaining - 1` the verify writes exactly `remaining`
            // slots and the cycle emits at most `remaining` tokens.
            let remaining: usize = max_as_usize.saturating_sub($gen.len());
            let cycle_depth: usize = cycle_depth.min(remaining.saturating_sub(1));
            if cycle_depth < 1 {
                // Only 1 token of budget left — an MTP cycle would
                // draft+verify more tokens than can be emitted. Fall
                // back to single-token AR decode: skip this cycle and
                // let the next iteration's Step A emit the final token
                // (its post-emit `$gen.len() >= $max` check then breaks
                // the loop with reason "length"). `chained_hidden_opt`
                // is still `None` here, so Step A runs unconditionally.
                tracing::debug!(
                    target: "mlx_core::mtp",
                    remaining,
                    "MTP cycle skipped near tail: AR-decoding the final token(s)"
                );
                continue;
            }
            $profiler.begin("mtp_cycle");
            let cycle_started_at = std::time::Instant::now();
            let commit_anchor = if cycle_seed_was_chained && $mtp.committed_history_active {
                $crate::models::qwen3_5::chat_common::MtpCommitAnchor::SkipAlreadyCommittedAnchor
            } else {
                $crate::models::qwen3_5::chat_common::MtpCommitAnchor::IncludeAnchor
            };
            let ev_depth_policy = if $p.mtp_adaptive_depth
                && matches!(
                    mtp_adaptive_depth_mode,
                    $crate::models::qwen3_5::adaptive_depth::AdaptiveDepthMode::ExpectedValue
                )
            {
                Some(&mut mtp_ev_depth_policy)
            } else {
                None
            };
            let cycle_res =
                $crate::models::qwen3_5::chat_common::run_mtp_cycle_inner(
                    &mut $mtp,
                    prev_h,
                    prev_e,
                    last_id,
                    &$emb,
                    &$hist,
                    &$p,
                    &mut $rng,
                    &mut $profiler,
                    cycle_depth,
                    ev_depth_policy,
                    commit_anchor,
                );
            $profiler.end();
            // `run_mtp_cycle_inner` returns the verify-final hidden so
            // the NEXT outer iteration can skip Step A's ~150 ms
            // main-model forward. We stash it into `chained_hidden_opt`;
            // the iteration boundary's `do_step_a` check will drain it.
            let (outcome, verify_last_hidden) = cycle_res?;
            chained_hidden_opt = Some(verify_last_hidden);

            // Throttled per-cycle MTP trace. Mirrors the AR loop's
            // every-32-steps cadence in token-count units so MTP and
            // AR runs leave comparable breadcrumb density. Reports the
            // dense compiled main-path cache offset; on the MoE path
            // this is the dense global and will read 0 — verify MoE
            // offsets via `mtp_chained_hidden_opt`-flavoured tooling.
            if ($gen.len() / 32) != (($gen.len() + outcome.tokens.len()) / 32) {
                let cache_offset = unsafe { mlx_sys::mlx_qwen35_get_cache_offset() };
                let first_tok = outcome.tokens.first().copied().unwrap_or(0);
                tracing::info!(
                    "Qwen3.5 decode MTP cycle gen_len={} depth={} committed={} \
                     first_tok={} cache_offset={}",
                    $gen.len(),
                    outcome.effective_depth,
                    outcome.tokens.len(),
                    first_tok,
                    cache_offset,
                );
            }
            // Feed observation to the policy AFTER the cycle's tokens
            // have been counted but BEFORE the emit loop's stop checks
            // (so the record always runs even on partial-emit due to
            // EOS / length / cancel). `committed` is the number
            // of tokens the cycle actually produced (drafts accepted +
            // residual/bonus); range `[1, depth+1]`.
            let cycle_wall_ns: u64 = cycle_started_at
                .elapsed()
                .as_nanos()
                .min(u128::from(u64::MAX)) as u64;
            let cycle_committed: u32 = outcome.tokens.len() as u32;
            if $p.mtp_adaptive_depth
                && matches!(
                    mtp_adaptive_depth_mode,
                    $crate::models::qwen3_5::adaptive_depth::AdaptiveDepthMode::Throughput
                )
            {
                mtp_depth_policy.record_cycle(
                    $crate::models::qwen3_5::adaptive_depth::CycleStats {
                        depth: outcome.effective_depth as u8,
                        committed: cycle_committed,
                        wall_ns: cycle_wall_ns,
                    },
                );
                tracing::debug!(
                    target: "mlx_core::mtp::adaptive",
                    state = mtp_depth_policy.state_label(),
                    depth = outcome.effective_depth,
                    requested_depth = outcome.requested_depth,
                    committed = cycle_committed,
                    wall_ms = (cycle_wall_ns as f64) / 1_000_000.0,
                    next_depth = mtp_depth_policy.pick_depth(),
                    "W6.8 cycle"
                );
            }

            // Emit each accepted token through the same stop /
            // streaming pipeline as the single-token loop.
            let mut hit_stop = false;
            let mut cycle_emitted: usize = 0;
            $profiler.begin("mtp_emit_loop");
            for tok_id in outcome.tokens.iter().copied() {
                if $gen.len() >= max_as_usize {
                    if $reason.is_empty() { $reason = String::from("length"); }
                    hit_stop = true;
                    break;
                }
                $gen.push(tok_id);
                $hist.push(tok_id);
                cycle_emitted += 1;
                let _is_reasoning = $tracker.observe_token(tok_id);
                $(
                    $last_r = _is_reasoning;
                    if $cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                        $reason = String::from("cancelled");
                        hit_stop = true;
                        break;
                    }
                    let token_text = $crate::tokenizer::Qwen3Tokenizer::step_decode_stream(
                        &mut $ds, $tok.inner(), tok_id, &$gen, $slen,
                    );
                    $slen += token_text.len();
                    // Suppress reasoning deltas when include_reasoning == false,
                    // matching the AR decode loop (detokenize + length-advance
                    // above stay outside the gate so DecodeStream sees every token).
                    if $p.include_reasoning || !_is_reasoning {
                        $cb.call(
                            Ok($crate::models::qwen3_5::model::ChatStreamChunk {
                                text: token_text, done: false, finish_reason: None,
                                tool_calls: None, thinking: None, num_tokens: None,
                                prompt_tokens: None, reasoning_tokens: None,
                                raw_text: None, cached_tokens: None, performance: None,
                                is_reasoning: Some(_is_reasoning),
                            }),
                            napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
                        );
                    }
                )?
                if tok_id == $eos {
                    $reason = String::from("stop");
                    hit_stop = true;
                    break;
                }
                if let Some(reason) = $crate::sampling::check_repetition_cutoff(
                    &$gen, $p.max_consecutive_tokens, $p.max_ngram_repeats, $p.ngram_size,
                ) {
                    $reason = reason.to_string();
                    hit_stop = true;
                    break;
                }
            }
            $profiler.end();
            tracing::debug!(
                target: "mlx_core::mtp",
                cycle_committed,
                gen_len = $gen.len(),
                hit_stop,
                cycle_emitted,
                "MTP cycle emit loop done"
            );

            // Every-256-emitted-token cache clear (matches the
            // single-token loop's cadence in token-count units).
            if $gen.len() >= last_clear_at + 256 {
                $crate::array::synchronize_and_clear_cache();
                last_clear_at = $gen.len();
            }

            if hit_stop {
                let unemitted = outcome.tokens.len().saturating_sub(cycle_emitted);
                if unemitted > 0 {
                    ($mtp.rollback_unemitted)(unemitted);
                }
                break;
            }
            // Set `$y` to the last accepted token so the next Step A
            // feeds the right token through main-path forward.
            // (Step A unconditionally re-seeds `prev_hidden_opt` /
            // `prev_emb_opt` / `last_committed_id_opt`, so no explicit
            // drain here.)
            let last = *outcome
                .tokens
                .last()
                .ok_or_else(|| napi::Error::from_reason("at least one accepted"))?
                as i32;
            $y = $crate::array::MxArray::from_int32(&[last], &[1])?;

            // When chaining IS enabled, flush the main path's KV-cache
            // lazy graph BEFORE the next cycle starts AND fuse the
            // chained `verify_hidden[K]` slice into the SAME `async_eval`
            // batch as `(token, g_compiled_caches)`.
            //
            // A plain `eval_step($y, h, false)` here would leave the
            // chained hidden LAZY across the iteration boundary (that
            // helper ignores `h` unless `budget_forced`), so when the
            // next cycle's `(ops.draft_step)(...)` built its graph against
            // `prev_hidden = chained_hidden`, materializing the slice
            // forced a mid-cycle Metal command-buffer roundtrip that the
            // Step-A bypass doesn't pay (Step A's `forward_with_hidden`
            // returns `(logits, hidden)` as siblings of the same compiled
            // forward and `eval_step` co-schedules them via
            // `next_token → logits → hidden`).
            //
            // `eval_step_with_chained_hidden` extends the same dispatch
            // to include the slice — so it becomes a sibling of
            // `(token, caches)`, the kernel scheduler can overlap its
            // materialization with the next cycle's draft graph
            // construction, and the chained path stops paying the
            // per-cycle roundtrip.
            //
            // On the Step-A path this whole branch is dead anyway:
            // Step A's `eval_step` call at the top of the NEXT
            // iteration handles the cache flush, and there is no
            // chained hidden to fold. The branch is only entered when
            // `chained_cycles_enabled=true` AND `chained_hidden_opt`
            // is `Some(...)`.
            if chained_cycles_enabled {
                if let Some(ref h) = chained_hidden_opt {
                    ($mtp.eval_step_with_chained_hidden)(&$y, h);
                }
            }
            $profiler.step();
        }

        $profiler.snapshot_memory_after();
        $profiler.report();
    }};
}

pub(crate) use decode_loop_mtp;

/// Policy decision for the C++ compiled paged forward fallback.
///
/// Inputs:
/// * `compiled_step_completed` — whether ANY compiled C++ paged step
///   has succeeded earlier in this turn.
///
/// Output:
/// * `true` — propagate the forward error as fatal. Returned when a
///   compiled step has previously succeeded; the C++ side has advanced
///   its per-layer GDN linear-cache globals (conv_state /
///   recurrent_state) but those updates are never imported back into
///   `self.caches`. Falling back to the pure-Rust paged decode after
///   that point would read stale pre-step state and silently corrupt
///   the response.
/// * `false` — safe to fall back to the pure-Rust paged decode.
///   Returned when no compiled step has succeeded yet; the only failure
///   mode at that point is an init/configuration mismatch caught at
///   first dispatch, which leaves `self.caches` consistent with
///   `paged_adapter` after a `rollback_last_tokens(1)`.
///
/// This mirrors the policy applied identically in the dense and MoE
/// sync + streaming decode loops; extracting it as a stand-alone helper
/// keeps the tests in lockstep.
#[inline]
pub(crate) fn should_propagate_compiled_paged_error(compiled_step_completed: bool) -> bool {
    compiled_step_completed
}

#[cfg(test)]
mod mtp_history_policy_tests {
    use super::{
        MtpHistoryPolicy, MtpPromptHistorySelection, resolve_mtp_prompt_history_selection,
    };

    #[test]
    fn committed_keeps_full_prompt_run() {
        assert_eq!(
            resolve_mtp_prompt_history_selection("committed", 4096, 8192, 16384),
            MtpPromptHistorySelection {
                policy: MtpHistoryPolicy::Committed,
                keep_tokens: 4096,
                position_base: 0,
            }
        );
    }

    #[test]
    fn auto_switches_to_last_window_at_threshold() {
        assert_eq!(
            resolve_mtp_prompt_history_selection("auto", 20000, 8192, 16384),
            MtpPromptHistorySelection {
                policy: MtpHistoryPolicy::LastWindow,
                keep_tokens: 8192,
                position_base: 11808,
            }
        );
    }

    #[test]
    fn last_window_caps_prompt_tail() {
        assert_eq!(
            resolve_mtp_prompt_history_selection("last-window", 10, 4, 100),
            MtpPromptHistorySelection {
                policy: MtpHistoryPolicy::LastWindow,
                keep_tokens: 4,
                position_base: 6,
            }
        );
    }
}

#[cfg(test)]
mod mtp_params_tests {
    //! MTP defaults + override plumbing for `ChatParams`. No Metal
    //! required; purely tests the `ChatConfig → ChatParams` extraction.

    use super::extract_chat_params;
    use crate::models::qwen3_5::model::ChatConfig;

    fn base_config() -> ChatConfig {
        ChatConfig {
            max_new_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: None,
            repetition_context_size: None,
            presence_penalty: None,
            presence_context_size: None,
            frequency_penalty: None,
            frequency_context_size: None,
            max_consecutive_tokens: None,
            max_ngram_repeats: None,
            ngram_size: None,
            tools: None,
            reasoning_effort: None,
            thinking_token_budget: None,
            include_reasoning: None,
            report_performance: None,
            reuse_cache: None,
            enable_mtp: None,
            mtp_depth: None,
            mtp_adaptive_depth: None,
        }
    }

    /// Defaults: MTP off, depth 1.
    #[test]
    fn defaults_disable_mtp() {
        let cfg = base_config();
        let p = extract_chat_params(&cfg);
        assert!(!p.enable_mtp, "enable_mtp must default to false");
        assert_eq!(p.mtp_depth, 1, "mtp_depth must default to 1");
    }

    /// User override: `enable_mtp=true`, `mtp_depth=2` flows through.
    #[test]
    fn user_overrides_pass_through() {
        let mut cfg = base_config();
        cfg.enable_mtp = Some(true);
        cfg.mtp_depth = Some(2);
        let p = extract_chat_params(&cfg);
        assert!(p.enable_mtp);
        assert_eq!(p.mtp_depth, 2);
    }

    /// Depth clamping: <1 clamps to 1, >5 clamps to 5.
    #[test]
    fn depth_clamps_to_verify_ffi_range() {
        let mut cfg = base_config();
        cfg.mtp_depth = Some(0);
        let p = extract_chat_params(&cfg);
        assert_eq!(p.mtp_depth, 1, "mtp_depth=0 must clamp to 1");

        cfg.mtp_depth = Some(99);
        let p = extract_chat_params(&cfg);
        assert_eq!(
            p.mtp_depth, 5,
            "mtp_depth=99 must clamp to verify-FFI max 5"
        );

        // Negative depths are "<1" and must clamp to 1 — NOT wrap to
        // usize::MAX and clamp up to the slowest depth 5.
        cfg.mtp_depth = Some(-1);
        let p = extract_chat_params(&cfg);
        assert_eq!(p.mtp_depth, 1, "mtp_depth=-1 must clamp to 1, not 5");

        cfg.mtp_depth = Some(i32::MIN);
        let p = extract_chat_params(&cfg);
        assert_eq!(p.mtp_depth, 1, "mtp_depth=i32::MIN must clamp to 1");
    }

    /// `mtp_adaptive_depth` default resolution.
    ///
    ///   * Neither `mtpAdaptiveDepth` nor `mtpDepth` set → adaptive OFF.
    ///   * `mtpDepth` set, `mtpAdaptiveDepth` unset → adaptive OFF
    ///     (caller pinned a depth).
    ///   * `mtpAdaptiveDepth = Some(true)`, `mtpDepth` set → adaptive
    ///     ON (explicit field wins; depth becomes initial seed).
    ///   * `mtpAdaptiveDepth = Some(false)`, `mtpDepth` unset → OFF.
    #[test]
    fn adaptive_depth_default_resolution() {
        // Default: no fields set → adaptive OFF.
        let cfg = base_config();
        let p = extract_chat_params(&cfg);
        assert!(
            !p.mtp_adaptive_depth,
            "mtp_adaptive_depth must default to false when neither field is set"
        );

        // User pinned depth → adaptive OFF.
        let mut cfg = base_config();
        cfg.mtp_depth = Some(4);
        let p = extract_chat_params(&cfg);
        assert!(
            !p.mtp_adaptive_depth,
            "setting mtpDepth alone must pin (adaptive OFF)"
        );
        assert_eq!(p.mtp_depth, 4);

        // Explicit adaptive=true with pinned depth → adaptive ON.
        let mut cfg = base_config();
        cfg.mtp_depth = Some(2);
        cfg.mtp_adaptive_depth = Some(true);
        let p = extract_chat_params(&cfg);
        assert!(p.mtp_adaptive_depth);
        assert_eq!(p.mtp_depth, 2, "depth becomes the initial seed");

        // Explicit adaptive=false with no depth → OFF (uses default 1).
        let mut cfg = base_config();
        cfg.mtp_adaptive_depth = Some(false);
        let p = extract_chat_params(&cfg);
        assert!(!p.mtp_adaptive_depth);
        assert_eq!(p.mtp_depth, 1);
    }
}

#[cfg(test)]
mod mtp_cycle_tests {
    //! `run_mtp_cycle_inner` smoke tests with mock draft / verify
    //! closures. Each test invokes the helper with a tiny synthetic
    //! vocab and validates: emitted token count, rollback callback
    //! receives the expected `(accepted_drafts, depth)` pair. Drafter
    //! and verifier closures track call counts so the tests double as
    //! wiring assertions.
    //!
    //! The rollback contract: `accepted_drafts` is the number of
    //! draft positions whose K/V we keep (range `0..=depth`). The
    //! dispatch-site callback in `model.rs` translates this into the
    //! single shared offset delta `accepted_drafts - depth` and
    //! applies it to BOTH the main and MTP compiled offsets.
    //!
    //! These are Metal-light: they DO use MLX softmax / sample /
    //! take inside the helper, so the tests skip cleanly when Metal
    //! is unavailable (mirrors the `compiled_ffi_tests` pattern in
    //! `mtp.rs`).
    //!
    //! Full token-for-token parity vs the eager Rust MTP forward is
    //! intentionally out of scope here — covered by the integration
    //! smoke that exercises real weights.

    use std::cell::RefCell;

    use crate::array::MxArray;
    use crate::decode_profiler::DecodeProfiler;
    use crate::models::qwen3_5::adaptive_depth::ExpectedValueDepthPolicy;
    use crate::models::qwen3_5::chat_common::{
        MtpCommitAnchor, MtpCycleOutcome, MtpOps, MtpVerifyOutput, extract_chat_params,
        run_mtp_cycle_inner,
    };
    use crate::models::qwen3_5::model::ChatConfig;
    use crate::sampling::{SamplingConfig, SparseDistributionRows};

    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const VOCAB: i64 = 8;
    const HIDDEN: i64 = 4;

    fn default_params() -> super::ChatParams {
        extract_chat_params(&ChatConfig {
            max_new_tokens: None,
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: None,
            repetition_context_size: None,
            presence_penalty: None,
            presence_context_size: None,
            frequency_penalty: None,
            frequency_context_size: None,
            max_consecutive_tokens: None,
            max_ngram_repeats: None,
            ngram_size: None,
            tools: None,
            reasoning_effort: None,
            thinking_token_budget: None,
            include_reasoning: None,
            report_performance: None,
            reuse_cache: None,
            enable_mtp: Some(true),
            mtp_depth: Some(3),
            mtp_adaptive_depth: None,
        })
    }

    /// Build a fake embedding table — vocab x hidden, deterministic
    /// distinct values per row so `take` returns recognizable embs.
    fn fake_embedding() -> Option<MxArray> {
        let mut data = Vec::with_capacity((VOCAB * HIDDEN) as usize);
        for v in 0..VOCAB {
            for h in 0..HIDDEN {
                data.push((v * 10 + h) as f32);
            }
        }
        MxArray::from_float32(&data, &[VOCAB, HIDDEN]).ok()
    }

    fn fake_hidden() -> Option<MxArray> {
        let data = vec![0.5f32; HIDDEN as usize];
        MxArray::from_float32(&data, &[1, 1, HIDDEN]).ok()
    }

    /// Construct a draft-step closure that always returns peaked
    /// logits at `draft_id_per_step[step]` and a constant h_next.
    /// Tracks call count in `counter`.
    fn make_draft<'a>(
        draft_id_per_step: &'a [i32],
        counter: &'a RefCell<usize>,
    ) -> impl FnMut(&MxArray, &MxArray) -> napi::Result<(MxArray, MxArray)> + 'a {
        move |_prev_h: &MxArray, _prev_e: &MxArray| {
            let step = *counter.borrow();
            *counter.borrow_mut() += 1;
            let id = draft_id_per_step[step % draft_id_per_step.len()];
            let mut logits = vec![-10.0f32; VOCAB as usize];
            logits[id as usize] = 10.0;
            let draft_logits = MxArray::from_float32(&logits, &[1, VOCAB]).expect("draft logits");
            let h_next = MxArray::from_float32(&vec![0.0f32; HIDDEN as usize], &[1, 1, HIDDEN])
                .expect("h_next");
            Ok((h_next, draft_logits))
        }
    }

    /// Construct a verify-step closure that returns logits peaked at
    /// `verify_id_per_position[i]` for each verify position, plus a
    /// zero `[1, depth+1, hidden]` stand-in for the per-position
    /// verify hiddens (the production closure exports the post-final-norm
    /// hidden at EVERY verify position so the caller can slice
    /// `verify_hiddens[:, K, :]` for chaining). For the mock tests below
    /// we don't care about contents, only shape, so a fresh zeros tensor
    /// matches the `[1, depth+1, hidden_size]` contract
    /// `run_mtp_cycle_inner` slices from.
    fn make_verify<'a>(
        verify_id_per_position: &'a [i32],
        counter: &'a RefCell<usize>,
    ) -> impl FnMut(&MxArray, &MxArray, usize) -> napi::Result<MtpVerifyOutput> + 'a {
        move |_ids: &MxArray, _emb: &MxArray, depth: usize| {
            *counter.borrow_mut() += 1;
            let positions = depth + 1;
            assert_eq!(verify_id_per_position.len(), positions);
            let mut data = vec![-10.0f32; positions * VOCAB as usize];
            for (i, &id) in verify_id_per_position.iter().enumerate() {
                data[i * VOCAB as usize + id as usize] = 10.0;
            }
            let arr =
                MxArray::from_float32(&data, &[1, positions as i64, VOCAB]).expect("verify logits");
            // Per-position verify hiddens: [1, depth+1, hidden].
            // Mirrors the production stacked `[1, D+1, hidden_size]`
            // contract `mlx_qwen35_mtp_verify_compiled_with_hidden`
            // ships.
            let zero_hiddens = vec![0.0f32; positions * HIDDEN as usize];
            let hiddens = MxArray::from_float32(&zero_hiddens, &[1, positions as i64, HIDDEN])
                .expect("verify hiddens stub");
            Ok(MtpVerifyOutput::logits_only(arr, hiddens))
        }
    }

    fn make_sparse_verify<'a>(
        target_sparse: SparseDistributionRows,
        counter: &'a RefCell<usize>,
    ) -> impl FnMut(&MxArray, &MxArray, usize) -> napi::Result<MtpVerifyOutput> + 'a {
        move |_ids: &MxArray, _emb: &MxArray, depth: usize| {
            *counter.borrow_mut() += 1;
            let positions = depth + 1;
            let zero_hiddens = vec![0.0f32; positions * HIDDEN as usize];
            let hiddens = MxArray::from_float32(&zero_hiddens, &[1, positions as i64, HIDDEN])
                .expect("verify hiddens stub");
            Ok(MtpVerifyOutput::sparse(hiddens, target_sparse.clone()))
        }
    }

    fn skip_if_metal_unavailable<T, E: std::fmt::Display>(
        label: &str,
        r: Result<T, E>,
    ) -> Option<T> {
        match r {
            Ok(v) => Some(v),
            Err(e) => {
                let msg = format!("{}", e);
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping {label} (Metal unavailable): {msg}");
                    None
                } else {
                    panic!("unexpected failure in {label}: {msg}");
                }
            }
        }
    }

    fn observed_commit_payload(
        label: &str,
        depth: usize,
        draft_ids: Vec<i32>,
        verify_ids: Vec<i32>,
        anchor: MtpCommitAnchor,
    ) -> Option<(MtpCycleOutcome, (Vec<u32>, usize))> {
        let emb = fake_embedding()?;
        let prev_h = fake_hidden()?;
        let prev_e = fake_hidden()?;
        let draft_ctr = RefCell::new(0usize);
        let verify_ctr = RefCell::new(0usize);
        let rollback_seen = RefCell::new(None::<(usize, usize)>);
        let commit_seen = RefCell::new(None::<(Vec<u32>, usize)>);
        let params = default_params();
        let mut rng = StdRng::seed_from_u64(0x51CED);
        let mut profiler = DecodeProfiler::new("test", "test");

        let res = {
            let mut ops = MtpOps {
                forward_with_hidden: |_ids: &MxArray,
                                      _emb: &MxArray|
                 -> napi::Result<(MxArray, MxArray, bool)> {
                    unreachable!("forward_with_hidden is not called inside run_mtp_cycle_inner")
                },
                draft_step: make_draft(&draft_ids, &draft_ctr),
                verify_step: make_verify(&verify_ids, &verify_ctr),
                verify_step_argmax_only: None,
                verify_step_sparse: None,
                rollback: |a: usize, d: usize| {
                    *rollback_seen.borrow_mut() = Some((a, d));
                },
                eval_step: |_t: &MxArray, _l: &MxArray, _b: bool| {},
                eval_step_with_chained_hidden: |_t: &MxArray, _h: &MxArray| {},
                begin_cycle: |_| {},
                snapshot_main_linear: || {},
                restore_and_replay_main: |_: &[u32], _: &MxArray| Ok(()),
                commit_mtp: |_: MtpCommitAnchor,
                             _: &MxArray,
                             _: &MxArray,
                             committed_ids: &[u32],
                             accepted_drafts: usize,
                             _: &MxArray| {
                    *commit_seen.borrow_mut() = Some((committed_ids.to_vec(), accepted_drafts));
                    Ok(())
                },
                committed_history_active: true,
                rollback_unemitted: |_: usize| {},
            };
            run_mtp_cycle_inner(
                &mut ops,
                prev_h,
                prev_e,
                0u32,
                &emb,
                &[],
                &params,
                &mut rng,
                &mut profiler,
                depth,
                None,
                anchor,
            )
        };
        let (outcome, _) = skip_if_metal_unavailable(label, res)?;
        let commit = commit_seen
            .into_inner()
            .expect("commit_mtp must be called exactly once");
        Some((outcome, commit))
    }

    /// All-accept path: drafter and verifier agree on every drafted
    /// token; cycle emits `depth + 1` tokens and the rollback
    /// callback fires with `(accepted_drafts=depth, depth=depth)` so
    /// the resulting `accepted_drafts - depth = 0` delta leaves both
    /// the main and the MTP offsets where the verify pass left them.
    #[test]
    fn all_accept_emits_depth_plus_one_tokens() {
        let depth = 3usize;
        let Some(emb) = fake_embedding() else { return };
        let Some(prev_h) = fake_hidden() else { return };
        let Some(prev_e) = fake_hidden() else { return };

        let draft_ids = vec![1i32, 2, 3];
        let verify_ids = vec![1i32, 2, 3, 4];
        let draft_ctr = RefCell::new(0usize);
        let verify_ctr = RefCell::new(0usize);
        let rollback_seen = RefCell::new(None::<(usize, usize)>);
        let commit_seen = RefCell::new(None::<(Vec<u32>, usize)>);

        let mut ops = MtpOps {
            forward_with_hidden: |_ids: &MxArray,
                                  _emb: &MxArray|
             -> napi::Result<(MxArray, MxArray, bool)> {
                unreachable!("forward_with_hidden is not called inside run_mtp_cycle_inner")
            },
            draft_step: make_draft(&draft_ids, &draft_ctr),
            verify_step: make_verify(&verify_ids, &verify_ctr),
            verify_step_argmax_only: None,
            verify_step_sparse: None,
            rollback: |a: usize, d: usize| {
                *rollback_seen.borrow_mut() = Some((a, d));
            },
            eval_step: |_t: &MxArray, _l: &MxArray, _b: bool| {},
            eval_step_with_chained_hidden: |_t: &MxArray, _h: &MxArray| {},
            begin_cycle: |_| {},
            snapshot_main_linear: || {},
            restore_and_replay_main: |_: &[u32], _: &MxArray| Ok(()),
            commit_mtp: |_: MtpCommitAnchor,
                         _: &MxArray,
                         _: &MxArray,
                         committed_ids: &[u32],
                         accepted_drafts: usize,
                         _: &MxArray| {
                *commit_seen.borrow_mut() = Some((committed_ids.to_vec(), accepted_drafts));
                Ok(())
            },
            // Cycle-level acceptance tests use the legacy cycle-history
            // policy, so committed-history is inactive.
            committed_history_active: false,
            rollback_unemitted: |_: usize| {},
        };
        let params = default_params();
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let mut profiler = DecodeProfiler::new("test", "test");

        let res = run_mtp_cycle_inner(
            &mut ops,
            prev_h,
            prev_e,
            0u32,
            &emb,
            &[],
            &params,
            &mut rng,
            &mut profiler,
            depth,
            None,
            MtpCommitAnchor::IncludeAnchor,
        );
        let Some((outcome, _verify_hidden)) = skip_if_metal_unavailable("all_accept", res) else {
            return;
        };
        assert_eq!(*draft_ctr.borrow(), depth, "must run depth draft steps");
        assert_eq!(*verify_ctr.borrow(), 1, "must run exactly one verify step");
        assert_eq!(
            outcome.tokens.len(),
            depth + 1,
            "all-accept must emit depth+1 tokens (depth drafts + 1 bonus)"
        );
        assert_eq!(outcome.tokens, vec![1u32, 2, 3, 4]);
        assert_eq!(
            *rollback_seen.borrow(),
            Some((depth, depth)),
            "rollback callback must receive (accepted_drafts=depth, depth=depth) on full \
             accept — produces zero offset delta in the dispatch-site formula \
             `accepted_drafts - depth`"
        );
        assert_eq!(
            *commit_seen.borrow(),
            Some((vec![0u32, 1, 2, 3, 4], depth)),
            "full-accept committed-history payload must be [last_committed, all drafts, bonus]"
        );
    }

    /// Expected-value gate path: caller requests depth 3, but the policy
    /// stops after the first draft, so verify/rollback/commit must all use
    /// `effective_depth=1` rather than the requested depth.
    #[test]
    fn ev_depth_gate_shortens_effective_depth_contract() {
        let depth = 3usize;
        let Some(emb) = fake_embedding() else { return };
        let Some(prev_h) = fake_hidden() else { return };
        let Some(prev_e) = fake_hidden() else { return };

        let draft_ids = vec![1i32, 2, 3];
        let verify_ids = vec![1i32, 4];
        let draft_ctr = RefCell::new(0usize);
        let verify_ctr = RefCell::new(0usize);
        let rollback_seen = RefCell::new(None::<(usize, usize)>);
        let commit_seen = RefCell::new(None::<(Vec<u32>, usize)>);

        let mut ops = MtpOps {
            forward_with_hidden: |_ids: &MxArray,
                                  _emb: &MxArray|
             -> napi::Result<(MxArray, MxArray, bool)> {
                unreachable!("forward_with_hidden is not called inside run_mtp_cycle_inner")
            },
            draft_step: make_draft(&draft_ids, &draft_ctr),
            verify_step: make_verify(&verify_ids, &verify_ctr),
            verify_step_argmax_only: None,
            verify_step_sparse: None,
            rollback: |a: usize, d: usize| {
                *rollback_seen.borrow_mut() = Some((a, d));
            },
            eval_step: |_t: &MxArray, _l: &MxArray, _b: bool| {},
            eval_step_with_chained_hidden: |_t: &MxArray, _h: &MxArray| {},
            begin_cycle: |_| {},
            snapshot_main_linear: || {},
            restore_and_replay_main: |_: &[u32], _: &MxArray| Ok(()),
            commit_mtp: |_: MtpCommitAnchor,
                         _: &MxArray,
                         _: &MxArray,
                         committed_ids: &[u32],
                         accepted_drafts: usize,
                         _: &MxArray| {
                *commit_seen.borrow_mut() = Some((committed_ids.to_vec(), accepted_drafts));
                Ok(())
            },
            committed_history_active: false,
            rollback_unemitted: |_: usize| {},
        };
        let params = default_params();
        let mut rng = StdRng::seed_from_u64(0xE11E);
        let mut profiler = DecodeProfiler::new("test", "test");
        let mut ev_policy =
            ExpectedValueDepthPolicy::for_test(3, 1, [0.70, 0.10, 0.05, 0.05, 0.05], 0.30);

        let res = run_mtp_cycle_inner(
            &mut ops,
            prev_h,
            prev_e,
            0u32,
            &emb,
            &[],
            &params,
            &mut rng,
            &mut profiler,
            depth,
            Some(&mut ev_policy),
            MtpCommitAnchor::IncludeAnchor,
        );
        let Some((outcome, _verify_hidden)) = skip_if_metal_unavailable("ev_depth_gate", res)
        else {
            return;
        };
        assert_eq!(*draft_ctr.borrow(), 1, "EV gate must stop after one draft");
        assert_eq!(*verify_ctr.borrow(), 1, "must run exactly one verify step");
        assert_eq!(outcome.requested_depth, 3);
        assert_eq!(outcome.effective_depth, 1);
        assert_eq!(
            outcome.tokens,
            vec![1u32, 4u32],
            "shortened full-accept cycle emits the accepted draft plus bonus"
        );
        assert_eq!(
            *rollback_seen.borrow(),
            Some((1, 1)),
            "rollback must receive the shortened effective depth"
        );
        assert_eq!(
            *commit_seen.borrow(),
            Some((vec![0u32, 1, 4], 1)),
            "commit payload must match the shortened verify window"
        );
    }

    /// Depth-1 degeneracy: 1 draft + 1 verify position still works.
    #[test]
    fn depth_one_degenerates_correctly() {
        let depth = 1usize;
        let Some(emb) = fake_embedding() else { return };
        let Some(prev_h) = fake_hidden() else { return };
        let Some(prev_e) = fake_hidden() else { return };

        let draft_ids = vec![5i32];
        let verify_ids = vec![5i32, 7];
        let draft_ctr = RefCell::new(0usize);
        let verify_ctr = RefCell::new(0usize);
        let rollback_seen = RefCell::new(None::<(usize, usize)>);
        let commit_seen = RefCell::new(None::<(Vec<u32>, usize)>);

        let mut ops = MtpOps {
            forward_with_hidden: |_ids: &MxArray,
                                  _emb: &MxArray|
             -> napi::Result<(MxArray, MxArray, bool)> {
                unreachable!()
            },
            draft_step: make_draft(&draft_ids, &draft_ctr),
            verify_step: make_verify(&verify_ids, &verify_ctr),
            verify_step_argmax_only: None,
            verify_step_sparse: None,
            rollback: |a: usize, d: usize| {
                *rollback_seen.borrow_mut() = Some((a, d));
            },
            eval_step: |_t: &MxArray, _l: &MxArray, _b: bool| {},
            eval_step_with_chained_hidden: |_t: &MxArray, _h: &MxArray| {},
            begin_cycle: |_| {},
            snapshot_main_linear: || {},
            restore_and_replay_main: |_: &[u32], _: &MxArray| Ok(()),
            commit_mtp: |_: MtpCommitAnchor,
                         _: &MxArray,
                         _: &MxArray,
                         committed_ids: &[u32],
                         accepted_drafts: usize,
                         _: &MxArray| {
                *commit_seen.borrow_mut() = Some((committed_ids.to_vec(), accepted_drafts));
                Ok(())
            },
            // Cycle-level acceptance tests use the legacy cycle-history
            // policy, so committed-history is inactive.
            committed_history_active: false,
            rollback_unemitted: |_: usize| {},
        };
        let params = default_params();
        let mut rng = StdRng::seed_from_u64(0xBADC0DE);
        let mut profiler = DecodeProfiler::new("test", "test");

        let res = run_mtp_cycle_inner(
            &mut ops,
            prev_h,
            prev_e,
            0u32,
            &emb,
            &[],
            &params,
            &mut rng,
            &mut profiler,
            depth,
            None,
            MtpCommitAnchor::IncludeAnchor,
        );
        let Some((outcome, _verify_hidden)) = skip_if_metal_unavailable("depth_one", res) else {
            return;
        };
        assert_eq!(*draft_ctr.borrow(), 1);
        assert_eq!(outcome.tokens.len(), 2, "depth=1 + full accept = 2 tokens");
        assert_eq!(outcome.tokens, vec![5u32, 7u32]);
        assert_eq!(*rollback_seen.borrow(), Some((1, 1)));
        assert_eq!(
            *commit_seen.borrow(),
            Some((vec![0u32, 5, 7], 1)),
            "depth=1 full-accept commit payload must include last_committed, draft, and bonus"
        );
    }

    #[test]
    fn sparse_stochastic_accept_uses_precomputed_target_rows_without_logits() {
        let depth = 1usize;
        let Some(emb) = fake_embedding() else { return };
        let Some(prev_h) = fake_hidden() else { return };
        let Some(prev_e) = fake_hidden() else { return };

        let draft_ids = vec![1i32];
        let target_sparse = SparseDistributionRows::from_precomputed(
            vec![1, 0, 2, 0],
            vec![1.0, 0.0, 1.0, 0.0],
            2,
            2,
            VOCAB as usize,
            "sparse_stochastic_accept_uses_precomputed_target_rows_without_logits",
        )
        .expect("target sparse rows");
        let draft_ctr = RefCell::new(0usize);
        let verify_ctr = RefCell::new(0usize);
        let rollback_seen = RefCell::new(None::<(usize, usize)>);
        let commit_seen = RefCell::new(None::<(Vec<u32>, usize)>);

        let mut ops = MtpOps {
            forward_with_hidden: |_ids: &MxArray,
                                  _emb: &MxArray|
             -> napi::Result<(MxArray, MxArray, bool)> {
                unreachable!()
            },
            draft_step: make_draft(&draft_ids, &draft_ctr),
            verify_step: make_sparse_verify(target_sparse, &verify_ctr),
            verify_step_argmax_only: None,
            verify_step_sparse: None,
            rollback: |a: usize, d: usize| {
                *rollback_seen.borrow_mut() = Some((a, d));
            },
            eval_step: |_t: &MxArray, _l: &MxArray, _b: bool| {},
            eval_step_with_chained_hidden: |_t: &MxArray, _h: &MxArray| {},
            begin_cycle: |_| {},
            snapshot_main_linear: || {},
            restore_and_replay_main: |_: &[u32], _: &MxArray| Ok(()),
            commit_mtp: |_: MtpCommitAnchor,
                         _: &MxArray,
                         _: &MxArray,
                         committed_ids: &[u32],
                         accepted_drafts: usize,
                         _: &MxArray| {
                *commit_seen.borrow_mut() = Some((committed_ids.to_vec(), accepted_drafts));
                Ok(())
            },
            committed_history_active: false,
            rollback_unemitted: |_: usize| {},
        };
        let mut params = default_params();
        params.sampling_config = Some(SamplingConfig {
            temperature: Some(1.0),
            top_k: Some(2),
            top_p: Some(1.0),
            min_p: Some(0.0),
        });
        let mut rng = StdRng::seed_from_u64(0x5A57);
        let mut profiler = DecodeProfiler::new("test", "test");

        let res = run_mtp_cycle_inner(
            &mut ops,
            prev_h,
            prev_e,
            0u32,
            &emb,
            &[],
            &params,
            &mut rng,
            &mut profiler,
            depth,
            None,
            MtpCommitAnchor::IncludeAnchor,
        );
        let Some((outcome, _verify_hidden)) =
            skip_if_metal_unavailable("sparse_precomputed_target", res)
        else {
            return;
        };
        assert_eq!(*draft_ctr.borrow(), 1);
        assert_eq!(*verify_ctr.borrow(), 1);
        assert_eq!(outcome.tokens, vec![1u32, 2u32]);
        assert_eq!(*rollback_seen.borrow(), Some((1, 1)));
        assert_eq!(*commit_seen.borrow(), Some((vec![0u32, 1, 2], 1)));
    }

    /// All-reject path: drafter and verifier argmaxes disagree on
    /// position 0 — cycle emits exactly 1 residual token and the
    /// rollback callback reports `accepted_drafts=0` so the
    /// dispatch-site delta `0 - depth = -depth` rewinds the full
    /// draft window on BOTH the main and MTP offsets.
    #[test]
    fn all_reject_emits_one_residual() {
        let depth = 3usize;
        let Some(emb) = fake_embedding() else { return };
        let Some(prev_h) = fake_hidden() else { return };
        let Some(prev_e) = fake_hidden() else { return };

        let draft_ids = vec![1i32, 2, 3];
        let verify_ids = vec![6i32, 7, 0, 0];
        let draft_ctr = RefCell::new(0usize);
        let verify_ctr = RefCell::new(0usize);
        let rollback_seen = RefCell::new(None::<(usize, usize)>);
        let commit_seen = RefCell::new(None::<(Vec<u32>, usize)>);

        let mut ops = MtpOps {
            forward_with_hidden: |_ids: &MxArray,
                                  _emb: &MxArray|
             -> napi::Result<(MxArray, MxArray, bool)> {
                unreachable!()
            },
            draft_step: make_draft(&draft_ids, &draft_ctr),
            verify_step: make_verify(&verify_ids, &verify_ctr),
            verify_step_argmax_only: None,
            verify_step_sparse: None,
            rollback: |a: usize, d: usize| {
                *rollback_seen.borrow_mut() = Some((a, d));
            },
            eval_step: |_t: &MxArray, _l: &MxArray, _b: bool| {},
            eval_step_with_chained_hidden: |_t: &MxArray, _h: &MxArray| {},
            begin_cycle: |_| {},
            snapshot_main_linear: || {},
            restore_and_replay_main: |_: &[u32], _: &MxArray| Ok(()),
            commit_mtp: |_: MtpCommitAnchor,
                         _: &MxArray,
                         _: &MxArray,
                         committed_ids: &[u32],
                         accepted_drafts: usize,
                         _: &MxArray| {
                *commit_seen.borrow_mut() = Some((committed_ids.to_vec(), accepted_drafts));
                Ok(())
            },
            // Cycle-level acceptance tests use the legacy cycle-history
            // policy, so committed-history is inactive.
            committed_history_active: false,
            rollback_unemitted: |_: usize| {},
        };
        let params = default_params();
        let mut rng = StdRng::seed_from_u64(0xDEAD);
        let mut profiler = DecodeProfiler::new("test", "test");

        let res = run_mtp_cycle_inner(
            &mut ops,
            prev_h,
            prev_e,
            0u32,
            &emb,
            &[],
            &params,
            &mut rng,
            &mut profiler,
            depth,
            None,
            MtpCommitAnchor::IncludeAnchor,
        );
        let Some((outcome, _verify_hidden)) = skip_if_metal_unavailable("all_reject", res) else {
            return;
        };
        assert_eq!(*draft_ctr.borrow(), depth);
        assert_eq!(
            outcome.tokens.len(),
            1,
            "all-reject at position 0 emits 1 residual token"
        );
        assert_eq!(
            *rollback_seen.borrow(),
            Some((0, depth)),
            "rollback must report accepted_drafts=0 on first-position reject so the \
             dispatch-site delta `0 - depth = -depth` rewinds the full draft window \
             on both caches"
        );
        assert_eq!(
            *commit_seen.borrow(),
            Some((vec![0u32, 6], 0)),
            "all-reject committed-history payload must be [last_committed, residual]"
        );
    }

    /// Partial-reject regression: drafter and verifier agree on the
    /// first two positions, then diverge at position 2 (out of
    /// `depth=3`). Cycle emits 3 tokens — 2 accepted drafts plus 1
    /// verifier residual — and the rollback callback reports
    /// `accepted_drafts=2`, NOT `accepted_drafts=3` (the pre-fix
    /// bug). This regression locks in the invariant that the
    /// residual sample does NOT count toward `accepted_drafts`,
    /// because the residual has no draft K/V slot; its K/V will be
    /// laid down by the NEXT cycle's Step A.
    #[test]
    fn partial_reject_reports_accepted_draft_count() {
        let depth = 3usize;
        let Some(emb) = fake_embedding() else { return };
        let Some(prev_h) = fake_hidden() else { return };
        let Some(prev_e) = fake_hidden() else { return };

        // Drafter argmaxes at 1, 2, 3; verifier agrees on the first
        // two positions (1, 2) and diverges (argmax 6) at position 2.
        // Position 3 is the bonus slot — never sampled on
        // partial-reject. The accept loop walks verify positions
        // 0..depth and compares each against `draft_ids[i]`.
        let draft_ids = vec![1i32, 2, 3];
        let verify_ids = vec![1i32, 2, 6, 0];
        let draft_ctr = RefCell::new(0usize);
        let verify_ctr = RefCell::new(0usize);
        let rollback_seen = RefCell::new(None::<(usize, usize)>);
        let commit_seen = RefCell::new(None::<(Vec<u32>, usize)>);

        let mut ops = MtpOps {
            forward_with_hidden: |_ids: &MxArray,
                                  _emb: &MxArray|
             -> napi::Result<(MxArray, MxArray, bool)> {
                unreachable!()
            },
            draft_step: make_draft(&draft_ids, &draft_ctr),
            verify_step: make_verify(&verify_ids, &verify_ctr),
            verify_step_argmax_only: None,
            verify_step_sparse: None,
            rollback: |a: usize, d: usize| {
                *rollback_seen.borrow_mut() = Some((a, d));
            },
            eval_step: |_t: &MxArray, _l: &MxArray, _b: bool| {},
            eval_step_with_chained_hidden: |_t: &MxArray, _h: &MxArray| {},
            begin_cycle: |_| {},
            snapshot_main_linear: || {},
            restore_and_replay_main: |_: &[u32], _: &MxArray| Ok(()),
            commit_mtp: |_: MtpCommitAnchor,
                         _: &MxArray,
                         _: &MxArray,
                         committed_ids: &[u32],
                         accepted_drafts: usize,
                         _: &MxArray| {
                *commit_seen.borrow_mut() = Some((committed_ids.to_vec(), accepted_drafts));
                Ok(())
            },
            // Cycle-level acceptance tests use the legacy cycle-history
            // policy, so committed-history is inactive.
            committed_history_active: false,
            rollback_unemitted: |_: usize| {},
        };
        let params = default_params();
        let mut rng = StdRng::seed_from_u64(0xFEED);
        let mut profiler = DecodeProfiler::new("test", "test");

        let res = run_mtp_cycle_inner(
            &mut ops,
            prev_h,
            prev_e,
            0u32,
            &emb,
            &[],
            &params,
            &mut rng,
            &mut profiler,
            depth,
            None,
            MtpCommitAnchor::IncludeAnchor,
        );
        let Some((outcome, _verify_hidden)) = skip_if_metal_unavailable("partial_reject", res)
        else {
            return;
        };
        assert_eq!(*draft_ctr.borrow(), depth);
        // 2 accepted drafts (positions 0,1) + 1 residual at position 2.
        assert_eq!(
            outcome.tokens.len(),
            3,
            "partial-reject K=2 must emit 2 accepted drafts + 1 residual = 3 tokens"
        );
        // On accept, `accept_with_residual` returns the original
        // drafted id (not the verifier's argmax).
        assert_eq!(
            outcome.tokens[0], 1u32,
            "first accepted draft is draft_ids[0]"
        );
        assert_eq!(
            outcome.tokens[1], 2u32,
            "second accepted draft is draft_ids[1]"
        );
        // Residual is sampled from `(p_target - p_draft)+`; with
        // sharply-peaked argmax disagreement at position 2 it MUST
        // equal the verifier argmax (6).
        assert_eq!(
            outcome.tokens[2], 6u32,
            "residual must be the verifier argmax under peaked disagreement"
        );
        assert_eq!(
            *rollback_seen.borrow(),
            Some((2, depth)),
            "rollback must report accepted_drafts=K=2 on partial reject (NOT K + 1 = 3); \
             the dispatch-site delta `2 - 3 = -1` rewinds exactly the rejected draft slot \
             on both caches. Locks in the pre-fix off-by-one regression."
        );
        assert_eq!(
            *commit_seen.borrow(),
            Some((vec![0u32, 1, 2, 6], 2)),
            "partial-reject committed-history payload must be [last_committed, accepted drafts, residual]"
        );
    }

    #[test]
    fn chained_full_accept_skips_already_committed_anchor() {
        let Some((outcome, commit)) = observed_commit_payload(
            "chained_full_accept",
            3,
            vec![1, 2, 3],
            vec![1, 2, 3, 4],
            MtpCommitAnchor::SkipAlreadyCommittedAnchor,
        ) else {
            return;
        };
        assert_eq!(outcome.tokens, vec![1u32, 2, 3, 4]);
        assert_eq!(
            commit,
            (vec![1u32, 2, 3, 4], 3),
            "chained full-accept must commit only newly emitted tokens"
        );
    }

    #[test]
    fn chained_partial_reject_skips_already_committed_anchor() {
        let Some((outcome, commit)) = observed_commit_payload(
            "chained_partial_reject",
            3,
            vec![1, 2, 3],
            vec![1, 2, 6, 0],
            MtpCommitAnchor::SkipAlreadyCommittedAnchor,
        ) else {
            return;
        };
        assert_eq!(outcome.tokens, vec![1u32, 2, 6]);
        assert_eq!(
            commit,
            (vec![1u32, 2, 6], 2),
            "chained partial-reject must not re-commit the anchor token"
        );
    }

    #[test]
    fn chained_all_reject_commits_single_residual() {
        let Some((outcome, commit)) = observed_commit_payload(
            "chained_all_reject",
            3,
            vec![1, 2, 3],
            vec![6, 7, 0, 0],
            MtpCommitAnchor::SkipAlreadyCommittedAnchor,
        ) else {
            return;
        };
        assert_eq!(outcome.tokens, vec![6u32]);
        assert_eq!(
            commit,
            (vec![6u32], 0),
            "chained all-reject must use the new one-token commit path"
        );
    }

    // ---- EV intra-cycle deepen — T=0 byte-equivalence gate ----

    /// T=0 (greedy / `use_sparse_accept`) params. Penalties stay at
    /// their no-op defaults so `penalties_no_op == true`, and
    /// `temperature == 0.0` drives the sparse-accept argmax commit path.
    /// This is the production path that decides committed tokens at T=0,
    /// and the exact path the EV deepen gate (allow_deepen) controls.
    fn t0_params() -> super::ChatParams {
        extract_chat_params(&ChatConfig {
            max_new_tokens: None,
            temperature: Some(0.0),
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: None,
            repetition_context_size: None,
            presence_penalty: None,
            presence_context_size: None,
            frequency_penalty: None,
            frequency_context_size: None,
            max_consecutive_tokens: None,
            max_ngram_repeats: None,
            ngram_size: None,
            tools: None,
            reasoning_effort: None,
            thinking_token_budget: None,
            include_reasoning: None,
            report_performance: None,
            reuse_cache: None,
            enable_mtp: Some(true),
            mtp_depth: Some(3),
            mtp_adaptive_depth: Some(true),
        })
    }

    /// Verify-step closure backed by a FIXED per-position target table.
    /// For whatever `depth` it is invoked with it returns
    /// `[1, depth+1, vocab]` logits peaked at `target_table[i]` for each
    /// position `i in 0..=depth`. Crucially the argmax at position `i`
    /// is therefore INDEPENDENT of `depth` — the causal-attention
    /// property the T=0 safety proof relies on. `target_table` must
    /// have at least `max_depth + 1` entries.
    fn make_verify_table<'a>(
        target_table: &'a [i32],
        counter: &'a RefCell<usize>,
    ) -> impl FnMut(&MxArray, &MxArray, usize) -> napi::Result<MtpVerifyOutput> + 'a {
        move |_ids: &MxArray, _emb: &MxArray, depth: usize| {
            *counter.borrow_mut() += 1;
            let positions = depth + 1;
            assert!(
                target_table.len() >= positions,
                "target_table must cover depth+1 positions (causal table)"
            );
            let mut data = vec![-10.0f32; positions * VOCAB as usize];
            for (i, &id) in target_table.iter().take(positions).enumerate() {
                data[i * VOCAB as usize + id as usize] = 10.0;
            }
            let arr =
                MxArray::from_float32(&data, &[1, positions as i64, VOCAB]).expect("verify logits");
            let zero_hiddens = vec![0.0f32; positions * HIDDEN as usize];
            let hiddens = MxArray::from_float32(&zero_hiddens, &[1, positions as i64, HIDDEN])
                .expect("verify hiddens stub");
            Ok(MtpVerifyOutput::logits_only(arr, hiddens))
        }
    }

    /// Drive one full `run_mtp_cycle_inner` at T=0 with the EV depth
    /// policy and a chosen `allow_deepen`. Returns
    /// `(effective_depth, emitted_tokens, committed_payload)`. The
    /// drafter argmaxes at `draft_table[step]`; the verifier argmaxes at
    /// `target_table[position]` (both depth-independent). With the two
    /// tables equal on the overlapping positions the cycle full-accepts,
    /// letting the deepen gate run to `max_depth` when enabled.
    #[allow(clippy::type_complexity)]
    fn run_ev_t0_cycle(
        label: &str,
        requested_depth: usize,
        draft_table: &[i32],
        target_table: &[i32],
        allow_deepen: bool,
    ) -> Option<(usize, Vec<u32>, (Vec<u32>, usize))> {
        let emb = fake_embedding()?;
        let prev_h = fake_hidden()?;
        let prev_e = fake_hidden()?;
        let draft_ctr = RefCell::new(0usize);
        let verify_ctr = RefCell::new(0usize);
        let commit_seen = RefCell::new(None::<(Vec<u32>, usize)>);

        // base_depth=1 so the gate is the SOLE thing extending depth;
        // accept_ewma pinned high + costs zeroed (via `for_test`) so the
        // EV cost model always votes to deepen when allowed. The only
        // difference between the two runs is `allow_deepen`.
        let mut ev_policy = ExpectedValueDepthPolicy::for_test(
            requested_depth as u8,
            1,
            [0.99, 0.99, 0.99, 0.99, 0.99],
            0.0,
        );
        ev_policy.set_allow_deepen(allow_deepen);

        // Force the production sparse-accept gate ON for this thread so the
        // cycle deterministically drives the T=0 `use_sparse_accept` commit
        // path regardless of `MLX_MTP_SPARSE_ACCEPT` or its process-wide
        // cache. The `ran_phase` assertion below fails closed if, despite
        // this, the sparse branch was not taken.
        let _force_sparse = super::ForceSparseAcceptGuard::force(true);
        // Enable the profiler so `ran_phase` can observe which accept branch
        // executed (timing instrumentation only — never changes committed
        // tokens).
        let mut profiler = DecodeProfiler::new("test", "test");
        profiler.enable_for_test();

        let res = {
            let mut ops = MtpOps {
                forward_with_hidden: |_ids: &MxArray,
                                      _emb: &MxArray|
                 -> napi::Result<(MxArray, MxArray, bool)> {
                    unreachable!("forward_with_hidden is not called inside run_mtp_cycle_inner")
                },
                draft_step: make_draft(draft_table, &draft_ctr),
                verify_step: make_verify_table(target_table, &verify_ctr),
                verify_step_argmax_only: None,
                verify_step_sparse: None,
                rollback: |_a: usize, _d: usize| {},
                eval_step: |_t: &MxArray, _l: &MxArray, _b: bool| {},
                eval_step_with_chained_hidden: |_t: &MxArray, _h: &MxArray| {},
                begin_cycle: |_| {},
                snapshot_main_linear: || {},
                restore_and_replay_main: |_: &[u32], _: &MxArray| Ok(()),
                commit_mtp: |_: MtpCommitAnchor,
                             _: &MxArray,
                             _: &MxArray,
                             committed_ids: &[u32],
                             accepted_drafts: usize,
                             _: &MxArray| {
                    *commit_seen.borrow_mut() = Some((committed_ids.to_vec(), accepted_drafts));
                    Ok(())
                },
                committed_history_active: false,
                rollback_unemitted: |_: usize| {},
            };
            let params = t0_params();
            // Fixed seed is belt-and-suspenders: at T=0 the sparse-accept
            // commit path consumes ZERO RNG.
            let mut rng = StdRng::seed_from_u64(0x7E50);
            run_mtp_cycle_inner(
                &mut ops,
                prev_h,
                prev_e,
                0u32,
                &emb,
                &[],
                &params,
                &mut rng,
                &mut profiler,
                requested_depth,
                Some(&mut ev_policy),
                MtpCommitAnchor::IncludeAnchor,
            )
        };
        let (outcome, _) = skip_if_metal_unavailable(label, res)?;
        // Fail closed: this gate is only meaningful if it actually drove the
        // production T=0 sparse-accept commit path. `"mtp_accept_argmax"` is
        // begun ONLY inside the `if use_sparse_accept` branch, so its absence
        // means the cycle silently took the legacy/stochastic accept path and
        // the byte-equivalence assertions below would be testing the wrong
        // code.
        assert!(
            profiler.ran_phase("mtp_accept_argmax"),
            "{label}: C2 gate must exercise the T=0 sparse-accept commit path \
             (`use_sparse_accept`); it did not, so the byte-equivalence checks \
             would give false coverage confidence"
        );
        let commit = commit_seen
            .into_inner()
            .expect("commit_mtp must be called exactly once");
        Some((outcome.effective_depth, outcome.tokens, commit))
    }

    /// The T=0 safety gate that graduates `MLX_MTP_EV_ALLOW_DEEPEN`.
    ///
    /// Reconciles the empirical claim in `adaptive_depth.rs` ("real M5
    /// Max traces showed intra-cycle deepening can violate the
    /// temperature-0 byte-equivalence safety contract") against the
    /// static analysis (every committed slot `i` is `target_argmax[i]`,
    /// causally INDEPENDENT of effective_depth; at T=0 the drafter is a
    /// deterministic argmax; the sparse-accept commit consumes zero RNG).
    ///
    /// Invariant under test: deepening can only EXTEND the committed
    /// sequence — it must NEVER mutate an already-committed prefix. We
    /// run the SAME deterministic T=0 cycle twice, differing only in
    /// `allow_deepen`:
    ///   - shallow (`false`): the gate stops at `base_depth = 1`.
    ///   - deep (`true`):     the gate extends to `max_depth = 3`.
    ///
    /// With the drafter and verifier argmaxes equal on every overlapping
    /// position the cycle full-accepts in both runs, so the deepen gate
    /// is free to run to `max_depth`. The shallow token/commit window
    /// MUST be a byte-identical prefix of the deep one.
    ///
    /// GREEN => the Rust accept/commit layer the EV gate controls is
    /// depth-invariant at T=0 => the deepen flag is safe to graduate.
    /// RED   => a REAL safety violation; keep the flag OFF and
    /// root-cause. Do NOT weaken this test to make it pass.
    #[test]
    fn ev_deepen_t0_committed_tokens_byte_identical() {
        let requested_depth = 3usize;
        // Drafter and verifier agree on every overlapping position →
        // full accept at every depth → the gate is the only thing
        // choosing how far the cycle drafts. Positions: draft d_0=1,
        // d_1=2, d_2=3; verifier argmax matches those plus a bonus
        // argmax (4) at the final verify slot.
        let draft_table = vec![1i32, 2, 3];
        let target_table = vec![1i32, 2, 3, 4];

        let Some((shallow_depth, shallow_tokens, shallow_commit)) = run_ev_t0_cycle(
            "ev_deepen_shallow",
            requested_depth,
            &draft_table,
            &target_table,
            false,
        ) else {
            return;
        };
        let Some((deep_depth, deep_tokens, deep_commit)) = run_ev_t0_cycle(
            "ev_deepen_deep",
            requested_depth,
            &draft_table,
            &target_table,
            true,
        ) else {
            return;
        };

        // Sanity: the flag actually changed the drafted depth. If both
        // collapsed to the same depth the test would be vacuous.
        assert_eq!(
            shallow_depth, 1,
            "allow_deepen=false must stop the gate at base_depth=1"
        );
        assert_eq!(
            deep_depth, requested_depth,
            "allow_deepen=true must extend the gate to max_depth on a full-accept chain"
        );

        // Shallow run: base_depth=1 full accept → [d_0, bonus@1].
        assert_eq!(shallow_tokens, vec![1u32, 2u32]);
        // Deep run: full accept to depth 3 → [d_0, d_1, d_2, bonus@3].
        assert_eq!(deep_tokens, vec![1u32, 2u32, 3u32, 4u32]);

        // THE SAFETY INVARIANT: the shallow emitted-token window is a
        // byte-identical PREFIX of the deep window. Deepening only
        // appended tokens; it never rewrote slot 0 (the only committed
        // draft the shallow run produced beyond the bonus boundary).
        // Note the shallow run's bonus token at slot K_s=1 (value 2)
        // equals the deep run's resolved token at slot 1 (value 2 =
        // target_argmax[1]) — exactly the proof's claim that the shallow
        // full-accept bonus equals whatever deeper drafting commits at
        // that slot.
        assert!(
            deep_tokens.starts_with(&shallow_tokens[..1]),
            "deepen must not mutate the first committed token: shallow={shallow_tokens:?} \
             deep={deep_tokens:?}"
        );
        assert_eq!(
            shallow_tokens[1], deep_tokens[1],
            "shallow full-accept bonus@K_s must equal the token deeper drafting commits at \
             that slot (both are target_argmax[1]) — the T=0 exactness property"
        );

        // Commit payloads carry the same invariant (IncludeAnchor →
        // [last_committed, emitted...]). Shallow payload must be a
        // byte-identical prefix of the deep payload up to the shallow
        // boundary slot.
        let (shallow_ids, shallow_accepted) = shallow_commit;
        let (deep_ids, deep_accepted) = deep_commit;
        assert_eq!(shallow_ids, vec![0u32, 1, 2]);
        assert_eq!(deep_ids, vec![0u32, 1, 2, 3, 4]);
        assert_eq!(shallow_accepted, 1);
        assert_eq!(deep_accepted, 3);
        assert_eq!(
            shallow_ids[..2],
            deep_ids[..2],
            "committed anchor + first accepted draft must be byte-identical across depths"
        );
    }
}

#[cfg(test)]
mod compiled_paged_fallback_policy_tests {
    use super::should_propagate_compiled_paged_error;

    /// Regression test: mid-turn fallback after a successful compiled
    /// step would corrupt the GDN linear cache state. The policy must
    /// propagate the error as fatal once any compiled step has completed;
    /// only the first-step failure is safe to fall back to pure-Rust decode.
    #[test]
    fn no_compiled_step_yet_allows_fallback() {
        assert!(
            !should_propagate_compiled_paged_error(false),
            "first-step compiled forward failure must allow fallback to pure-Rust paged decode \
             (self.caches is still consistent with paged_adapter pre-rollback)"
        );
    }

    #[test]
    fn after_successful_compiled_step_propagates_as_fatal() {
        assert!(
            should_propagate_compiled_paged_error(true),
            "compiled forward failure AFTER a successful compiled step must propagate as fatal: \
             the C++ GDN linear-cache globals advanced but self.caches is stale, so a pure-Rust \
             fallback would silently corrupt the response"
        );
    }
}

#[cfg(test)]
mod tool_delta_marker_tests {
    //! Guard the structured `is_error` channel on
    //! `build_chatml_tool_delta_text`. The renderer injects the
    //! `TOOL_ERROR_MARKER` cue into the `<tool_response>` wire content
    //! only when the caller passes `Some(true)`. `None` and
    //! `Some(false)` keep the output byte-equal to the pre-feature
    //! behavior — guarding both the hot (successful) path and the
    //! explicit-false path against accidental drift.

    use super::build_chatml_tool_delta_text;
    use crate::tokenizer::TOOL_ERROR_MARKER;

    #[test]
    fn tool_delta_injects_marker_when_is_error_true() {
        // `Some(true)` must produce the marker prefix inside the
        // `<tool_response>` wrapper. The marker is the single shared
        // constant — using it directly here keeps the test in sync
        // with any future rename.
        let payload = "boom: connection refused";
        let rendered = build_chatml_tool_delta_text("call_fail", payload, None, Some(true));
        let expected_inner = format!("{TOOL_ERROR_MARKER}{payload}");
        assert!(
            rendered.contains(&expected_inner),
            "expected error marker inside <tool_response> wrapper; got:\n{rendered}",
        );
        // The wrapper itself must stay correct (we don't want to ship
        // a malformed delta that only the unflagged path renders right).
        assert!(
            rendered.contains("<tool_response>\n"),
            "wrapper open missing"
        );
        assert!(
            rendered.contains("</tool_response>"),
            "wrapper close missing"
        );
    }

    #[test]
    fn tool_delta_skips_marker_when_is_error_none() {
        // None = default; pre-feature output. The marker MUST NOT
        // appear anywhere in the wire text.
        let payload = "{\"temperature\": 72}";
        let rendered = build_chatml_tool_delta_text("call_ok", payload, None, None);
        assert!(
            !rendered.contains(TOOL_ERROR_MARKER),
            "marker leaked into unflagged delta:\n{rendered}",
        );
        assert!(
            rendered.contains(payload),
            "original content missing from delta:\n{rendered}",
        );
    }

    #[test]
    fn tool_delta_skips_marker_when_is_error_some_false() {
        // Explicit `Some(false)` is the same as `None` — only
        // `Some(true)` flips the marker on.
        let payload = "ok";
        let rendered = build_chatml_tool_delta_text("call_ok", payload, None, Some(false));
        assert!(
            !rendered.contains(TOOL_ERROR_MARKER),
            "marker leaked into Some(false) delta:\n{rendered}",
        );
    }

    #[test]
    fn tool_delta_does_not_remark_content_that_resembles_marker() {
        // The structured channel removes the collision concern: a
        // successful tool result whose literal content begins with the
        // marker text must NOT double-prefix the marker on its way
        // through the renderer.
        let suspicious = format!("{TOOL_ERROR_MARKER}this is a successful payload");
        let rendered = build_chatml_tool_delta_text("call_ok", &suspicious, None, None);
        // Exactly one occurrence — the original payload — no extra
        // prefix.
        let occurrences = rendered.matches(TOOL_ERROR_MARKER).count();
        assert_eq!(
            occurrences, 1,
            "marker count should be 1 (the original literal); got {occurrences} in:\n{rendered}",
        );
    }

    #[test]
    fn tool_delta_marker_interacts_correctly_with_thinking_prefix() {
        // The marker and the `<think>\n` prefix occupy different slots
        // in the delta. Both must render together when both are active:
        // marker inside `<tool_response>`, `<think>\n` after the
        // assistant opener.
        let rendered = build_chatml_tool_delta_text("call_fail", "boom", Some(true), Some(true));
        assert!(
            rendered.contains(&format!("{TOOL_ERROR_MARKER}boom")),
            "marker missing from thinking-enabled delta:\n{rendered}",
        );
        assert!(
            rendered.contains("<|im_start|>assistant\n<think>\n"),
            "thinking prefix missing from thinking-enabled delta:\n{rendered}",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THINK_END_ID: u32 = 151668; // example </think> token ID

    /// Legacy formula the helper must reproduce byte-for-byte on the valid
    /// (non-overflowing, non-negative) range.
    fn legacy_round_up(prefill_len: i32, max_new_tokens: i32) -> i32 {
        ((prefill_len + max_new_tokens + 255) / 256) * 256
    }

    #[test]
    fn kv_capacity_round_up_matches_legacy_formula() {
        // Spread of normal (prefill_len, max_new_tokens) pairs that cannot
        // overflow i32. For every non-negative non-overflowing input the
        // helper is byte-identical to the legacy idiom.
        let cases = [
            (0, 0),
            (1, 1),
            (255, 1),
            (256, 1),
            (1000, 2048),
            (10, 2038),
            (4096, 0),
            (0, 4096),
            (8192, 8192),
            (123_456, 654_321),
        ];
        for (p, m) in cases {
            let expected = legacy_round_up(p, m);
            assert_eq!(
                kv_capacity_round_up(p, m).unwrap(),
                expected,
                "kv_capacity_round_up({p}, {m}) must match legacy formula"
            );
        }
        // Spot-check a couple of the trickier ones by hand.
        assert_eq!(kv_capacity_round_up(0, 0).unwrap(), 0);
        assert_eq!(kv_capacity_round_up(1, 1).unwrap(), 256);
        assert_eq!(kv_capacity_round_up(255, 1).unwrap(), 256);
        assert_eq!(kv_capacity_round_up(256, 1).unwrap(), 512);
        // 10 + 2038 + 255 = 2303; 2303 / 256 = 8; 8 * 256 = 2048.
        assert_eq!(kv_capacity_round_up(10, 2038).unwrap(), 2048);
    }

    #[test]
    fn kv_capacity_round_up_boundary_exact() {
        // Largest representable multiple of 256 in i32.
        assert_eq!((i32::MAX / 256) * 256, 2_147_483_392);

        // Already a multiple of 256 (after +255 it lands exactly on i32::MAX,
        // which floors back to 2_147_483_392) — no overflow.
        assert_eq!(
            kv_capacity_round_up(0, 2_147_483_392).unwrap(),
            2_147_483_392
        );

        // One more token rounds up to 2_147_483_648 (> i32::MAX) -> Err.
        assert!(kv_capacity_round_up(0, 2_147_483_393).is_err());

        // The exact review finding: any non-empty prompt + an i32::MAX budget
        // would overflow the i32 sum in the legacy formula. Now a clean Err.
        assert!(kv_capacity_round_up(1, i32::MAX).is_err());

        // i32::MAX budget alone already rounds up past i32::MAX -> Err.
        assert!(kv_capacity_round_up(0, i32::MAX).is_err());
    }

    #[test]
    fn kv_capacity_round_up_saturating_never_panics() {
        // Saturating variant clamps to the largest in-range multiple of 256
        // instead of erroring — for display/trace only.
        assert_eq!(kv_capacity_round_up_saturating(1, i32::MAX), 2_147_483_392);
        // Valid inputs pass through unchanged.
        assert_eq!(kv_capacity_round_up_saturating(1, 1), 256);
    }

    #[test]
    fn kv_capacity_round_up_floors_negative_inputs() {
        // Defense in depth: negative inputs are floored at 0 (callers already
        // clamp, but the helper must never produce a negative or wrapped size).
        assert_eq!(kv_capacity_round_up(-5, -5).unwrap(), 0);
        assert_eq!(kv_capacity_round_up(-1, 1).unwrap(), 256);
        // (256, -1000) floors to (256, 0): 256 + 0 + 255 = 511; 511/256 = 1; *256 = 256.
        assert_eq!(kv_capacity_round_up(256, -1000).unwrap(), 256);
        // Negative input must produce the SAME result as the floored positive input.
        assert_eq!(
            kv_capacity_round_up(256, -1000).unwrap(),
            kv_capacity_round_up(256, 0).unwrap()
        );
    }

    #[test]
    fn generated_capacity_hint_caps_and_floors() {
        // [high] scenario: a hostile-but-accepted budget near i32::MAX must NOT
        // trigger a multi-GiB eager reservation — the hint is capped.
        assert_eq!(generated_capacity_hint(i32::MAX), 8192);
        assert_eq!(
            generated_capacity_hint(i32::MAX),
            GENERATED_CAPACITY_HINT_CAP
        );
        // Negative budgets floor at 0 (never wrap to usize::MAX → abort).
        assert_eq!(generated_capacity_hint(-5), 0);
        assert_eq!(generated_capacity_hint(i32::MIN), 0);
        // Below-cap budgets pass through exactly (behavior-neutral pre-alloc).
        assert_eq!(generated_capacity_hint(0), 0);
        assert_eq!(generated_capacity_hint(100), 100);
        assert_eq!(generated_capacity_hint(2048), 2048);
        // Exact cap boundary.
        assert_eq!(generated_capacity_hint(8192), 8192);
        assert_eq!(generated_capacity_hint(8193), 8192);
    }

    #[test]
    fn test_tracker_starts_in_thinking() {
        let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // reasoning
        assert!(tracker.observe_token(200)); // reasoning
        assert!(!tracker.should_force_think_end());
    }

    #[test]
    fn test_tracker_transitions_on_think_end() {
        let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // reasoning
        assert!(tracker.observe_token(THINK_END_ID)); // </think> is still reasoning
        assert!(!tracker.observe_token(300)); // now content
        assert!(!tracker.observe_token(400)); // still content
    }

    #[test]
    fn test_stream_reasoning_gate_predicate() {
        // Drives the boundary semantics the streaming send-gate relies on:
        // `observe_token` returns true for reasoning tokens INCLUDING the
        // `</think>` closer, and false for the first content token after.
        // The send-gate is `include_reasoning || !is_reasoning`.
        //
        // Token ids are chosen distinct from THINK_END_ID for the
        // reasoning/content tokens.
        let seq = [101u32, 102, THINK_END_ID, 301, 302];

        // include_reasoning == false: suppress the 3 reasoning tokens
        // (including the </think> closer), emit the 2 content tokens.
        {
            let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
            let include_reasoning = false;
            let gate: Vec<bool> = seq
                .iter()
                .map(|&tok| {
                    let is_reasoning = tracker.observe_token(tok);
                    include_reasoning || !is_reasoning
                })
                .collect();
            assert_eq!(gate, vec![false, false, false, true, true]);
        }

        // include_reasoning == true: emit everything.
        {
            let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
            let include_reasoning = true;
            let gate: Vec<bool> = seq
                .iter()
                .map(|&tok| {
                    let is_reasoning = tracker.observe_token(tok);
                    include_reasoning || !is_reasoning
                })
                .collect();
            assert_eq!(gate, vec![true, true, true, true, true]);
        }
    }

    #[test]
    fn test_tracker_starts_in_content() {
        let mut tracker = ReasoningTracker::new(false, None, Some(THINK_END_ID));
        assert!(!tracker.observe_token(100));
        assert!(!tracker.observe_token(200));
        assert!(!tracker.should_force_think_end());
    }

    #[test]
    fn test_tracker_budget_enforcement() {
        // Budget=3: allows exactly 3 thinking tokens, then forces on the 3rd.
        let mut tracker = ReasoningTracker::new(true, Some(3), Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // count→1
        assert!(!tracker.should_force_think_end());
        assert!(tracker.observe_token(200)); // count→2
        assert!(!tracker.should_force_think_end());
        assert!(tracker.observe_token(300)); // count→3, 3>=3 → force!
        assert!(tracker.should_force_think_end());
        assert_eq!(tracker.forced_token_id().unwrap(), THINK_END_ID);
    }

    #[test]
    fn test_default_thinking_budget_for_effort() {
        // none → Some(0): force </think> ASAP (minimal thinking).
        assert_eq!(default_thinking_budget_for_effort(Some("none")), Some(0));
        // low → Some(256): small cap.
        assert_eq!(default_thinking_budget_for_effort(Some("low")), Some(256));
        // medium / high / unset / unknown → None (unlimited; preserves default).
        assert_eq!(default_thinking_budget_for_effort(Some("medium")), None);
        assert_eq!(default_thinking_budget_for_effort(Some("high")), None);
        assert_eq!(default_thinking_budget_for_effort(None), None);
        assert_eq!(default_thinking_budget_for_effort(Some("bogus")), None);
    }

    #[test]
    fn test_tracker_budget_zero() {
        // Budget=0: force is set in new() — triggers BEFORE any thinking token.
        let mut tracker = ReasoningTracker::new(true, Some(0), Some(THINK_END_ID));
        assert!(tracker.should_force_think_end()); // immediate, no observe needed
    }

    #[test]
    fn test_tracker_budget_zero_vs_one() {
        // Budget=0: force immediately (0 thinking tokens allowed).
        let mut t0 = ReasoningTracker::new(true, Some(0), Some(THINK_END_ID));
        assert!(t0.should_force_think_end()); // before any observe

        // Budget=1: allows exactly 1 thinking token before forcing.
        let mut t1 = ReasoningTracker::new(true, Some(1), Some(THINK_END_ID));
        assert!(!t1.should_force_think_end()); // not yet
        assert!(t1.observe_token(100)); // count→1, 1>=1 → force!
        assert!(t1.should_force_think_end()); // triggers after 1st token
    }

    #[test]
    fn test_tracker_budget_clears_on_think_end() {
        let mut tracker = ReasoningTracker::new(true, Some(2), Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // count→1
        assert!(!tracker.should_force_think_end());
        assert!(tracker.observe_token(200)); // count→2, 2>=2 → force!
        assert!(tracker.should_force_think_end());
        // When the forced think_end token is generated:
        assert!(tracker.observe_token(THINK_END_ID)); // transitions to content
        assert!(!tracker.should_force_think_end()); // force cleared
        assert!(!tracker.observe_token(300)); // now content
    }

    #[test]
    fn test_tracker_no_double_force_with_pipeline_lag() {
        // Simulates pipelined decode: after should_force_think_end() is consumed,
        // the pipeline extracts an over-budget token before the forced </think>
        // arrives. The tracker must NOT re-trigger forcing.
        let mut tracker = ReasoningTracker::new(true, Some(3), Some(THINK_END_ID));
        tracker.observe_token(100); // count→1
        tracker.observe_token(200); // count→2
        tracker.observe_token(300); // count→3, 3>=3 → force=true

        // Phase A of step N+1: consume the force flag
        assert!(tracker.should_force_think_end()); // returns true, sets end_scheduled
        assert!(!tracker.should_force_think_end()); // already consumed — must be false

        // Phase B of step N+1: the pipeline extracts the over-budget token (not </think>)
        assert!(tracker.observe_token(400)); // still reasoning, count→4
        // Must NOT re-trigger forcing despite count(4) >= budget(3)
        assert!(!tracker.should_force_think_end());

        // Phase B of step N+2: the forced </think> token is finally extracted
        assert!(tracker.observe_token(THINK_END_ID)); // transitions to content
        assert!(!tracker.should_force_think_end());

        // Phase B of step N+3: normal content token
        assert!(!tracker.observe_token(500)); // content
    }

    #[test]
    fn test_tracker_no_budget() {
        let mut tracker = ReasoningTracker::new(true, None, Some(THINK_END_ID));
        for i in 0..1000 {
            assert!(tracker.observe_token(i));
            assert!(!tracker.should_force_think_end());
        }
    }

    #[test]
    fn test_tracker_no_think_end_id() {
        let mut tracker = ReasoningTracker::new(true, Some(5), None);
        // Without think_end_id, should_force_think_end is always false
        for i in 0..100 {
            tracker.observe_token(i);
            assert!(!tracker.should_force_think_end());
        }
    }

    #[test]
    fn test_tracker_no_think_end_id_labels_as_reasoning() {
        // When thinking is enabled but think_end_id is missing (tokenizer
        // renders </think> as multiple tokens), observe_token should still
        // return true (reasoning) for every token — consistent with the
        // text-level finalization that will find reasoning via parsing.
        let mut tracker = ReasoningTracker::new(true, None, None);
        assert!(tracker.observe_token(100)); // reasoning
        assert!(tracker.observe_token(200)); // reasoning
        assert!(tracker.observe_token(300)); // reasoning
        // Never transitions — no think_end_id to match
        assert!(!tracker.should_force_think_end()); // budget disabled
    }

    #[test]
    fn test_raw_text_with_reasoning_suppressed() {
        // Token sequences: a sequence CONTAINING THINK_END_ID confirms </think>
        // (has_think_end_token == true); a sequence WITHOUT it but with a
        // think_end_id provided is a truncated generation.
        let confirmed_tokens = [101u32, 102, THINK_END_ID, 301, 302];
        let truncated_tokens = [101u32, 102, 103, 104]; // no THINK_END_ID

        // 1. include_reasoning == true → verbatim (reasoning span intact).
        let text = "<think>secret reasoning</think>\nVisible answer";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &confirmed_tokens,
            true, // thinking_enabled
            Some(THINK_END_ID),
            Some("</think>"),
            true, // include_reasoning
        );
        assert_eq!(out, text, "include_reasoning=true must keep raw verbatim");

        // 2. include_reasoning == false + confirmed </think>: keep everything
        //    after the FIRST </think> VERBATIM, including a <tool_call> that
        //    lives in the content portion.
        let text = "<think>secret reasoning</think>\n<tool_call>{\"name\":\"f\"}</tool_call>";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &confirmed_tokens,
            true,
            Some(THINK_END_ID),
            Some("</think>"),
            false,
        );
        assert_eq!(
            out, "\n<tool_call>{\"name\":\"f\"}</tool_call>",
            "must keep post-</think> content (incl. tool markup) verbatim"
        );
        assert!(
            !out.contains("<think>"),
            "no opening think tag should remain"
        );
        assert!(
            !out.contains("</think>"),
            "no closing think tag should remain"
        );
        assert!(
            out.contains("<tool_call>"),
            "tool-call markup must be preserved"
        );

        // 3. include_reasoning == false + truncated generation (think_end_id in
        //    vocab but NOT present in generated tokens): all output is reasoning.
        let text = "<think>unterminated reasoning that hit EOS";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &truncated_tokens,
            true,
            Some(THINK_END_ID),
            Some("</think>"),
            false,
        );
        assert_eq!(out, "", "truncated generation scrubs to empty string");

        // 4. include_reasoning == false + thinking disabled: all output is
        //    content, so raw_text stays verbatim (even literal think tags).
        let text = "<think> is just literal text here";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &confirmed_tokens,
            false, // thinking_enabled == false
            Some(THINK_END_ID),
            Some("</think>"),
            false,
        );
        assert_eq!(out, text, "no-thinking mode keeps raw verbatim");

        // 5. Text-level fallback: no think_end_id in vocab and no tag string.
        //    Paired <think>…</think> is stripped. The fallback delegates to
        //    `tools::parse_thinking` (the SAME function parse_thinking_and_tools'
        //    fallback uses) so the boundary is identical to the parsed `text`
        //    field — which means the remainder is trimmed (parse_thinking's
        //    strip_tag_blocks trims), e.g. "\nABC" -> "ABC". Boundary fidelity is
        //    the priority here over byte-verbatimness; no reasoning leaks.
        let text = "<think>r</think>\nABC";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &confirmed_tokens,
            true,
            None, // think_end_id
            None, // think_end_str
            false,
        );
        assert_eq!(out, "ABC", "text-level fallback strips reasoning span");
        assert!(!out.contains("<think>"));
        assert!(!out.contains("</think>"));
        assert!(!out.contains('r'), "reasoning content must be gone");
    }

    #[test]
    fn test_strip_reasoning_span_multi_block() {
        // Text-level fallback (think_end_id == None && think_end_str == None)
        // must strip ALL paired <think>…</think> blocks, not just the first.
        // A single leftover block would leak chain-of-thought into raw_text even
        // though the parsed `thinking` field is fully suppressed.
        let text = "<think>a</think>\nmid\n<think>b</think>\nanswer";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &[101u32, 102, 103], // no think_end_id present (forces fallback anyway)
            true,                // thinking_enabled
            None,                // think_end_id → text-level fallback
            None,                // think_end_str
            false,               // include_reasoning
        );
        assert!(
            !out.contains("<think>"),
            "no opening think tag may remain: {out:?}"
        );
        assert!(
            !out.contains("</think>"),
            "no closing think tag may remain: {out:?}"
        );
        assert!(
            out.contains("mid"),
            "non-reasoning content between blocks must survive: {out:?}"
        );
        assert!(
            out.contains("answer"),
            "trailing non-reasoning content must survive: {out:?}"
        );
    }

    #[test]
    fn test_raw_text_fallback_mixed_unmatched_tag() {
        // Regression: an UNMATCHED `<think>` opener appearing BEFORE a valid
        // `<longcat_think>…</longcat_think>` block must NOT prevent the longcat
        // reasoning from being stripped. The earlier hand-rolled scanner picked
        // the earliest opener of either family and bailed when it had no close,
        // leaking the later block. Delegating to `tools::parse_thinking` (which
        // checks each tag family separately) strips the longcat block while
        // leaving the stray `<think>` literal — exactly matching the parsed
        // `thinking` boundary, so no reasoning CONTENT leaks.
        let text = "prefix <think> literal <longcat_think>secret</longcat_think> suffix";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &[101u32, 102, 103], // no think_end_id present → text-level fallback
            true,                // thinking_enabled
            None,                // think_end_id
            None,                // think_end_str
            false,               // include_reasoning
        );
        assert!(
            !out.contains("secret"),
            "longcat reasoning content must not leak: {out:?}"
        );
        assert!(
            !out.contains("longcat_think"),
            "no longcat reasoning tag may remain: {out:?}"
        );
        assert!(
            out.contains("prefix") && out.contains("suffix"),
            "non-reasoning content must survive: {out:?}"
        );
    }

    #[test]
    fn test_fallback_excludes_reasoning_nested_tool_call() {
        // No think_end_id in vocab (text-level fallback): a `<tool_call>` nested inside
        // a reasoning block must NOT be surfaced as an executable tool call — it is the
        // model THINKING about a call, not emitting one. This matches the token-confirmed
        // path (tool calls parsed only from post-`</think>` content) and the raw_text
        // scrub, keeping `tool_calls` consistent with `raw_text` on the fallback.
        let text =
            "<think>maybe I should <tool_call>{\"name\":\"f\"}</tool_call></think>\nfinal answer";
        let (clean, calls, thinking) = parse_thinking_and_tools(
            text,
            &[101u32, 102, 103], // no think_end_id present → fallback
            true,                // thinking_enabled
            None,                // think_end_id
            None,                // think_end_str
            true,                // include_reasoning (thinking field populated)
        );
        assert!(
            calls.is_empty(),
            "reasoning-nested tool call must not leak into tool_calls: {calls:?}"
        );
        assert!(
            clean.contains("final answer"),
            "post-reasoning content survives as text: {clean:?}"
        );
        assert!(
            thinking.is_some_and(|t| t.contains("maybe I should")),
            "reasoning is captured in the thinking field"
        );
    }

    #[test]
    fn test_fallback_extracts_standalone_tool_call_after_reasoning() {
        // Companion to the above: a `<tool_call>` OUTSIDE reasoning on the fallback path
        // is still extracted (the fix isolates reasoning, it does not drop real calls).
        let text = "<think>let me call it</think>\n<tool_call>{\"name\":\"f\"}</tool_call>";
        let (_clean, calls, _thinking) =
            parse_thinking_and_tools(text, &[101u32, 102, 103], true, None, None, false);
        assert_eq!(
            calls.len(),
            1,
            "standalone tool call is extracted: {calls:?}"
        );
        assert_eq!(calls[0].name, "f");
    }

    #[test]
    fn test_fallback_straddling_tool_call_does_not_leak() {
        // A `<tool_call>` opens inside `<think>` but its `</tool_call>`
        // lands after `</think>`, so the span straddles the reasoning
        // boundary. On the no-think_end_id fallback, neither `tool_calls` nor the `text`
        // field may surface a call that began in reasoning, and the reasoning prefix must
        // not leak into `text`.
        let text = "<think>secret <tool_call><function=leak></think>\n<parameter=q>1</parameter></function></tool_call>";
        let (clean, calls, _thinking) =
            parse_thinking_and_tools(text, &[101u32, 102, 103], true, None, None, false);
        assert!(
            calls.is_empty(),
            "straddling reasoning-started tool call must not be extracted: {calls:?}"
        );
        assert!(
            !clean.contains("secret") && !clean.contains("<think>"),
            "reasoning prefix must not leak into text: {clean:?}"
        );
    }

    #[test]
    fn test_raw_text_straddling_tool_call_does_not_leak() {
        // Same straddling shape through the raw_text scrubber: with include_reasoning=false
        // the reasoning prefix and the reasoning-started tool markup must both be gone.
        let text = "<think>secret <tool_call><function=leak></think>\n<parameter=q>1</parameter></function></tool_call>";
        let out = raw_text_with_reasoning_suppressed(
            text,
            &[101u32, 102, 103], // no think_end_id present → text-level fallback
            true,                // thinking_enabled
            None,                // think_end_id
            None,                // think_end_str
            false,               // include_reasoning
        );
        assert!(
            !out.contains("secret") && !out.contains("<think>"),
            "reasoning prefix must not leak into raw_text: {out:?}"
        );
    }

    #[test]
    fn test_force_think_end_pending_is_non_consuming() {
        // The non-consuming peek used by the MTP routing/defer decisions
        // must report a pending force WITHOUT clearing it. Repeated peeks
        // stay true; only the single consuming call clears it.
        let mut tracker = ReasoningTracker::new(true, Some(2), Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // count→1
        assert!(!tracker.force_think_end_pending()); // not yet tripped
        assert!(tracker.observe_token(200)); // count→2, 2>=2 → force=true

        // Peek repeatedly — must stay true, never consuming.
        assert!(tracker.force_think_end_pending());
        assert!(tracker.force_think_end_pending());
        assert!(tracker.force_think_end_pending());

        // The single consuming call returns true exactly once and clears.
        assert!(tracker.should_force_think_end());
        assert!(!tracker.should_force_think_end()); // consumed
        assert!(!tracker.force_think_end_pending()); // peek now reflects cleared flag
    }

    #[test]
    fn test_force_think_end_pending_mirrors_chained_routing() {
        // Mirrors the chained-ON MTP path: the budget trips, the routing
        // peek (`do_step_a`) and the defer guard both poll the NON-consuming
        // predicate (possibly many times across cycles), and only the single
        // token-insertion site consumes + forces exactly once.
        let mut tracker = ReasoningTracker::new(true, Some(1), Some(THINK_END_ID));
        assert!(tracker.observe_token(100)); // count→1, 1>=1 → force=true

        // Routing peek for cycle N (do_step_a), then a defer-guard peek, then
        // routing peek for cycle N+1 — all non-consuming, all stay true.
        for _ in 0..5 {
            assert!(
                tracker.force_think_end_pending(),
                "peek must remain true until the single consume fires"
            );
        }

        // Step A's token-insertion site consumes and forces exactly once.
        assert!(tracker.should_force_think_end());
        assert_eq!(tracker.forced_token_id().unwrap(), THINK_END_ID);

        // After the consume, no further force fires from peeks or consumes.
        assert!(!tracker.force_think_end_pending());
        assert!(!tracker.should_force_think_end());
    }
}

#[cfg(test)]
mod save_cache_state_after_delta_tests {
    //! Guards the sticky-`cached_image_key` invariant on the text-only
    //! delta path. Before the fix, `save_cache_state_direct(has_images:
    //! false, ...)` was called after every delta continuation, which
    //! cleared `cached_image_key` even though the live KV cache still
    //! encoded the prior prefill's image attention state. That
    //! contradicted the TS `ChatSession` routing contract (warm cache
    //! across text-only follow-ups) and caused the delta path to fail
    //! with a cryptic "chat_tokens_delta_sync is text-only; session
    //! currently holds image state" on the very next turn.
    use super::save_cache_state_after_delta;

    #[test]
    fn delta_preserves_cached_image_key_on_reuse_cache_true() {
        let mut cached_history: Vec<u32> = vec![1, 2, 3];
        let mut cached_image_key: Option<u64> = Some(0xdeadbeef);
        let mut cached_rope_deltas: Option<i32> = Some(5);
        let mut caches: Option<Vec<super::Qwen3_5LayerCache>> =
            Some(vec![super::Qwen3_5LayerCache::new_full_attention()]);

        save_cache_state_after_delta(
            /* reuse_cache */ true,
            /* generated_tokens */ &[10, 11],
            /* finish_reason */ "stop",
            /* save_tokens */ &[1, 2, 3, 4],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        // Token history extended: pre-decode snapshot + generated tokens
        assert_eq!(cached_history, vec![1, 2, 3, 4, 10, 11]);
        // Image key preserved — THE invariant under test
        assert_eq!(cached_image_key, Some(0xdeadbeef));
        // Other cache state untouched
        assert_eq!(cached_rope_deltas, Some(5));
        assert!(caches.is_some());
    }

    #[test]
    fn delta_drops_trailing_generated_token_on_length_stop() {
        // Matches `save_cache_state_direct` truncation semantics: if the
        // decode terminated at max_new_tokens, the last generated token
        // was cut off mid-stream and must not be persisted.
        let mut cached_history: Vec<u32> = vec![];
        let mut cached_image_key: Option<u64> = Some(42);
        let mut cached_rope_deltas: Option<i32> = None;
        let mut caches: Option<Vec<super::Qwen3_5LayerCache>> = None;

        save_cache_state_after_delta(
            true,
            &[10, 11, 12],
            "length",
            &[1, 2],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        assert_eq!(cached_history, vec![1, 2, 10, 11]);
        assert_eq!(cached_image_key, Some(42));
    }

    #[test]
    fn delta_full_reset_clears_everything_when_reuse_cache_false() {
        // `reuse_cache=false` is the cold-path invariant from the prefill
        // helper — when the caller opts out of cache reuse, every piece
        // of session state must be cleared regardless of whether the
        // image key was previously populated.
        let mut cached_history: Vec<u32> = vec![1, 2, 3];
        let mut cached_image_key: Option<u64> = Some(0xabc);
        let mut cached_rope_deltas: Option<i32> = Some(7);
        let mut caches: Option<Vec<super::Qwen3_5LayerCache>> =
            Some(vec![super::Qwen3_5LayerCache::new_linear()]);

        save_cache_state_after_delta(
            false,
            &[10],
            "stop",
            &[1],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        assert!(cached_history.is_empty());
        assert!(cached_image_key.is_none());
        assert!(cached_rope_deltas.is_none());
        assert!(caches.is_none());
    }

    #[test]
    fn delta_with_text_only_session_keeps_key_none() {
        // Sanity: if the session never had images, the delta must not
        // fabricate a key either.
        let mut cached_history: Vec<u32> = vec![];
        let mut cached_image_key: Option<u64> = None;
        let mut cached_rope_deltas: Option<i32> = None;
        let mut caches: Option<Vec<super::Qwen3_5LayerCache>> = None;

        save_cache_state_after_delta(
            true,
            &[42],
            "stop",
            &[1, 2],
            &mut cached_history,
            &mut cached_image_key,
            &mut cached_rope_deltas,
            &mut caches,
        );

        assert_eq!(cached_image_key, None);
        assert_eq!(cached_history, vec![1, 2, 42]);
    }
}

#[cfg(test)]
mod rope_delta_gate_tests {
    //! Guards the M-RoPE offset lifecycle across the compiled decode
    //! init branch. The prior bug hard-coded `has_images: false` on the
    //! delta path and unconditionally cleared `cached_rope_deltas`,
    //! which caused the compiled graph to decode text-only deltas at a
    //! sequential position instead of the image-adjusted position —
    //! mispositioning every generated token relative to the cached
    //! image patches baked in by the earlier VLM prefill.
    use super::{should_clear_rope_delta, should_reapply_rope_delta};

    // ---- should_reapply_rope_delta ----

    #[test]
    fn reapply_skipped_when_no_saved_delta() {
        // Text-only session, nothing to re-apply.
        assert!(!should_reapply_rope_delta(false, false, false, 0));
        // Image session with delta, but saved offset missing (fresh VLM
        // prefill clears it before setting — we never enter the gated
        // branch without a saved offset).
        assert!(!should_reapply_rope_delta(false, true, false, 0));
        assert!(!should_reapply_rope_delta(false, false, true, 100));
    }

    #[test]
    fn reapply_fires_on_fresh_vlm_cache_prefix_reuse() {
        // Fresh VLM prefill reusing a cached prefix: both `has_images`
        // AND a non-zero `cached_prefix_len` must be present. The saved
        // offset was written on the prior turn's VLM prefill, so a
        // matching key + prefix means we rebuild the compiled graph at
        // the same image-adjusted position.
        assert!(should_reapply_rope_delta(true, false, true, 100));
    }

    #[test]
    fn reapply_skipped_on_fresh_vlm_without_prefix_match() {
        // VLM prefill without prefix reuse (cached_prefix_len == 0):
        // the compiled init already ran the fresh prefill path, which
        // computed the offset from scratch via M-RoPE. No re-apply.
        assert!(!should_reapply_rope_delta(true, false, true, 0));
    }

    #[test]
    fn reapply_skipped_on_fresh_text_prefill() {
        // Fresh text prefill with no image state: the cache-prefix
        // verify already dropped any prior image-bearing cache, so the
        // saved offset is stale. `should_clear_rope_delta` handles that
        // case by nulling it; re-apply stays off.
        assert!(!should_reapply_rope_delta(true, false, false, 50));
        assert!(!should_reapply_rope_delta(true, false, false, 0));
    }

    #[test]
    fn reapply_fires_on_delta_continuation_with_saved_offset() {
        // THE invariant this fix introduces: delta continuations on an
        // image-bearing session re-apply the saved offset regardless of
        // `has_images` (which is always false on the delta path by
        // construction — delta prefills are text-only) and regardless
        // of `cached_prefix_len` (which is always 0 on the delta path
        // because the live KV cache already contains the full prior
        // history and the delta bypasses the prefix-match flow).
        assert!(should_reapply_rope_delta(true, true, false, 0));
    }

    #[test]
    fn reapply_fires_on_chained_delta_turns() {
        // Chained text-only deltas on the same image session: each
        // turn's compiled init must re-apply the offset so the session
        // stays positioned correctly. The save helper preserves
        // `cached_rope_deltas` on the reuse_cache branch, so the next
        // turn sees `has_saved_delta=true`.
        assert!(should_reapply_rope_delta(true, true, false, 0));
    }

    // ---- should_clear_rope_delta ----

    #[test]
    fn clear_fires_only_on_fresh_text_prefill() {
        // The ONE case where the saved offset is stale: a non-delta
        // text prefill. The cache-prefix verify already dropped any
        // prior image cache, so the offset has nothing valid to apply
        // to on the next turn.
        assert!(should_clear_rope_delta(false, false));
    }

    #[test]
    fn clear_skipped_on_delta_path() {
        // Delta continuations (text-only by construction) preserve the
        // offset — regression gate for the bug this fix addresses. The
        // live KV cache still encodes the prior VLM prefill's image
        // attention, so the next delta turn (and the one after that)
        // must re-apply the same saved offset.
        assert!(!should_clear_rope_delta(true, false));
    }

    #[test]
    fn clear_skipped_on_vlm_prefill() {
        // VLM prefill sets a fresh offset and must not nuke it after
        // init. The `is_delta` axis is false on the non-delta prefill
        // path; the `has_images` axis guards the clear.
        assert!(!should_clear_rope_delta(false, true));
    }

    #[test]
    fn clear_skipped_on_vlm_delta_combination() {
        // Belt-and-suspenders: even if a future caller ever set
        // `is_delta=true, has_images=true`, the clear stays off. No
        // current caller does this — the delta path rejects images at
        // entry — but the gate is written defensively.
        assert!(!should_clear_rope_delta(true, true));
    }
}

#[cfg(test)]
mod verify_cache_prefix_invariant_tests {
    //! Guards the all-or-nothing return-value invariant of
    //! `verify_cache_prefix_direct` documented on its rustdoc. The Qwen3.5
    //! chat_session_start refactor — which moves the unconditional
    //! `reset_caches_sync()` out of the outer session-start path and
    //! relies on verify returning either `0` or the full cached length
    //! to drive the in-core reset-on-miss branch — is **only** safe as
    //! long as this function never returns a mid-sequence prefix length.
    //! A regression here would silently let the caller resume decoding on
    //! a GDN recurrent state that no longer corresponds to the token
    //! prefix in the KV cache, corrupting every generated token.
    use super::verify_cache_prefix_direct;

    #[test]
    fn returns_zero_when_reuse_cache_disabled() {
        // `reuse_cache = false` short-circuits; everything else is
        // irrelevant. This is the "caller explicitly opted out" path.
        assert_eq!(
            verify_cache_prefix_direct(
                false,
                false,
                &[1, 2, 3, 4],
                &[1, 2, 3, 4],
                0,
                &[1, 2, 3],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_when_no_caches() {
        // `has_caches = false` means the model has no live KV caches to
        // resume from — a full prefill is required even if the history
        // matches.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[1, 2, 3, 4],
                &[1, 2, 3, 4],
                0,
                &[1, 2, 3],
                &None,
                false,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_on_empty_history() {
        // First session-start turn: nothing cached yet, so we must
        // prefill the whole prompt.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[1, 2, 3, 4],
                &[1, 2, 3, 4],
                0,
                &[],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_on_first_token_mismatch() {
        // Histories diverge at index 0 — no reusable prefix.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[9, 2, 3, 4],
                &[9, 2, 3, 4],
                0,
                &[1, 2, 3],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_on_midsequence_mismatch() {
        // CRITICAL: histories match for 2 tokens then diverge. The
        // function MUST return 0 (full miss), NOT 2 (partial hit).
        // A partial hit would signal the caller to reuse only the first
        // 2 positions of the KV cache — which for the GDN linear layers
        // would require rewinding the recurrent state, which is
        // impossible. The all-or-nothing contract is what keeps this
        // safe.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[1, 2, 7, 4],
                &[1, 2, 7, 4],
                0,
                &[1, 2, 3],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_zero_on_shorter_new_prompt() {
        // New prompt is shorter than the cached history — can't be a
        // forward extension. Rewinding is infeasible (see above), so
        // return 0 and force a fresh prefill.
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &[1, 2],
                &[1, 2],
                0,
                &[1, 2, 3, 4, 5],
                &None,
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_full_length_on_exact_append_hit() {
        // Happy path: the new prompt is `cached + [extra]`. The function
        // returns `cached.len()` so the caller prefills only the delta
        // tail. This is the whole point of the cache-reuse machinery.
        let cached = vec![1u32, 2, 3, 4];
        let new_prompt = vec![1u32, 2, 3, 4, 5, 6];
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                false,
                &new_prompt,
                &new_prompt,
                0,
                &cached,
                &None,
                true,
            ),
            cached.len(),
        );
    }

    #[test]
    fn returns_full_length_on_exact_match() {
        // Edge case: new prompt is byte-identical to cached. Returns
        // `cached.len()` — the caller's zero-delta guard then takes
        // over (see the matching comment in `qwen3_5/model.rs` and
        // `qwen3_5_moe/model.rs`).
        let cached = vec![1u32, 2, 3, 4];
        assert_eq!(
            verify_cache_prefix_direct(true, false, &cached, &cached, 0, &cached, &None, true,),
            cached.len(),
        );
    }

    #[test]
    fn returns_zero_on_image_key_mismatch() {
        // VLM path: cached image key differs from the current turn's
        // key — the images changed, so the cached KV state no longer
        // represents the new prompt's image attention. Full reset.
        let cached = vec![1u32, 2, 3];
        let new_prompt = vec![1u32, 2, 3, 4];
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                true,
                &new_prompt,
                &new_prompt,
                /* new image key */ 999,
                &cached,
                &Some(42),
                true,
            ),
            0,
        );
    }

    #[test]
    fn returns_full_length_on_vlm_image_key_match() {
        // VLM happy path: same images, new text tail. Returns the
        // cached prefix length so the caller prefills only the delta.
        let cached = vec![1u32, 2, 3];
        let new_prompt = vec![1u32, 2, 3, 4, 5];
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                true,
                &new_prompt,
                &new_prompt,
                42,
                &cached,
                &Some(42),
                true,
            ),
            cached.len(),
        );
    }

    #[test]
    fn returns_zero_on_vlm_missing_image_key() {
        // VLM turn but cached state carries no image key — the cache
        // came from a prior text-only exchange, not a VLM prefill.
        // Safety requires a fresh VLM prefill, not a reuse.
        let cached = vec![1u32, 2, 3];
        let new_prompt = vec![1u32, 2, 3, 4];
        assert_eq!(
            verify_cache_prefix_direct(
                true,
                true,
                &new_prompt,
                &new_prompt,
                42,
                &cached,
                &None,
                true,
            ),
            0,
        );
    }

    /// The contract-level invariant: across a broad sweep of inputs the
    /// return value is ALWAYS either `0` or `cached.len()`. Any
    /// intermediate value would corrupt GDN recurrent state on reuse.
    ///
    /// This property-style sweep is belt-and-suspenders on top of the
    /// targeted unit tests above: even if a future refactor changes
    /// branch structure, the invariant holds by construction.
    #[test]
    fn invariant_return_value_is_always_zero_or_cached_len() {
        let cached = vec![10u32, 20, 30, 40, 50];
        // Every prefix-plus-suffix combination and a selection of
        // divergent inputs.
        let candidates: Vec<Vec<u32>> = vec![
            vec![],
            vec![10],
            vec![10, 20],
            vec![10, 20, 30],
            vec![10, 20, 30, 40],
            cached.clone(),
            [cached.clone(), vec![60]].concat(),
            [cached.clone(), vec![60, 70, 80]].concat(),
            vec![99, 20, 30, 40, 50, 60],
            vec![10, 20, 99, 40, 50, 60],
            vec![10, 20, 30, 40, 99, 60],
        ];

        for candidate in &candidates {
            let result = verify_cache_prefix_direct(
                true, false, candidate, candidate, 0, &cached, &None, true,
            );
            assert!(
                result == 0 || result == cached.len(),
                "invariant violated: result={} for candidate={:?} (expected 0 or {})",
                result,
                candidate,
                cached.len(),
            );
        }
    }
}

#[cfg(test)]
mod compute_performance_metrics_tests {
    //! `compute_performance_metrics` is a faithful numerator / ttft divider:
    //! `prefill_tokens_per_second = prefill_tokens_len / (ttft_ms/1000)`. This
    //! documents that the LFM2-paged telemetry fix (using the full-prompt count
    //! as the numerator) MUST live at the call site — the divider here applies
    //! whatever numerator it is given, verbatim. Cheap, deterministic, no GPU.
    use super::compute_performance_metrics;
    use std::time::{Duration, Instant};

    #[test]
    fn compute_performance_metrics_uses_given_numerator_directly() {
        let t0 = Instant::now();
        let ttft = Duration::from_millis(150);
        let first_tok = t0 + ttft;

        // Full-prompt numerator (1000) -> ~6667 tok/s; ttft ~= 150ms.
        let m = compute_performance_metrics(Some(t0), Some(first_tok), 1000, 8)
            .expect("metrics present when both instants are Some");
        assert!(
            (m.ttft_ms - 150.0).abs() < 5.0,
            "ttft_ms should reflect first_tok - gen_start (~150ms), got {}",
            m.ttft_ms
        );
        let expected_full = 1000.0 / 0.150;
        assert!(
            (m.prefill_tokens_per_second - expected_full).abs() / expected_full < 0.05,
            "full-prompt numerator must divide directly: expected ~{expected_full:.0}, got {}",
            m.prefill_tokens_per_second
        );

        // Suffix-scale numerator (6) -> ~40 tok/s: the exact bogus value the
        // LFM2-paged bug produced (6 / 0.150). Proves the function is a plain
        // divider and the numerator is the load-bearing choice.
        let m_suffix =
            compute_performance_metrics(Some(t0), Some(first_tok), 6, 8).expect("metrics present");
        let expected_suffix = 6.0 / 0.150;
        assert!(
            (m_suffix.prefill_tokens_per_second - expected_suffix).abs() / expected_suffix < 0.05,
            "suffix numerator divides directly: expected ~{expected_suffix:.0}, got {}",
            m_suffix.prefill_tokens_per_second
        );
        // And the full-prompt value is >5x the suffix value, i.e. the bug
        // under-reported by exactly the cached-prefix ratio.
        assert!(
            m.prefill_tokens_per_second > 5.0 * m_suffix.prefill_tokens_per_second,
            "full-prompt tok/s ({}) must be >5x suffix tok/s ({})",
            m.prefill_tokens_per_second,
            m_suffix.prefill_tokens_per_second
        );
    }
}
