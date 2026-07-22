//! Qwen3.5 MTP (Multi-Token Prediction) speculative-decode machinery.
//!
//! Family-specific in refactor phase 1: the cached MTP env-flag readers,
//! draft/verify helpers, history-policy resolution, and the shared cycle
//! data types (`MtpCommitAnchor` / `MtpCycleOutcome` / `MtpVerifyOutput`)
//! that only the MTP path consumes. The engine-owned propose/verify loop
//! (`crate::engine::mtp_turn::run_mtp_turn` / `run_mtp_cycle`) drives them.
//! The model-neutral AR decode infrastructure lives in
//! [`crate::engine`]; shared items needed by both the AR and MTP paths
//! (`apply_all_penalties`, the `mtp_trace_logits` / `trace_top2` trace
//! helpers) are imported from there.

use std::sync::OnceLock;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::decode::Top2;
use crate::sampling::{self, SamplingConfig};

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

// Async verify-eval pipeline.
//
// Replaces the synchronous `verify_logits.eval()` at the end of
// the MTP cycle's verify step with a single batched
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
// `async_eval` batch as `(token, main layer caches)` at end-of-iteration (see
// the `eval_step_with_chained_hidden` stepper hook) so the slice becomes a sibling
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
// Replaces the per-position accept loop's D forced GPU syncs
// (each materializing a full-vocab softmax of ~151k floats) with ONE
// batched on-device op over all `D+1` verify positions. On the T=0
// (greedy) path this is `argmax(verify_logits, axis=-1)` → `[1, D+1]`
// int32, evaluated once. On T>0 we keep the per-position
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
// `MLX_MTP_SPARSE_ACCEPT=0` / `false` / `off` to force the per-position
// path for parity debugging or A/B measurements. The env
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
pub(crate) fn sparse_accept_gate() -> bool {
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
pub(crate) fn sparse_accept_gate() -> bool {
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
// This is an opt-out gate for the native sparse-verify fast path
// (`MtpStepper::verify_step_sparse`). When the sampler is in MTPLX parity
// mode, such a verifier can return compact `[depth+1, top_k]` target
// ids/probabilities directly from the native graph instead of surfacing full
// `[1, depth+1, vocab]` logits and rebuilding the same sparse rows on the
// Rust side. No eager stepper implements `verify_step_sparse` today, so the
// gate is currently inert; kept for a future native sparse verifier.
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

pub(crate) fn mtp_draft_sampling_config(
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

fn trace_json_f64(value: f64) -> serde_json::Value {
    serde_json::Number::from_f64(value)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

pub(crate) fn trace_acceptance_emit(payload: serde_json::Value) {
    eprintln!("MTP_TRACE_ACCEPTANCE {}", payload);
}

pub(crate) fn trace_acceptance_greedy(
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

pub(crate) fn trace_acceptance_sparse(
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

pub(crate) fn trace_acceptance_dense(
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

// =============================================================================
// Eager AR decode driver (`DecodeOps` + `decode_loop!`) — the token-by-token
// decode loop for the qwen3_5 dense/MoE MTP and vision whole-turn cores.
//
// The qwen3_5 dense/MoE whole-turn cores behind the engine's `mtp_turn` /
// `vision_turn` probes (`vision_mtp_whole_turn_core` and the delta/streaming
// twins in `models/qwen3_5/model.rs` and `models/qwen3_5_moe/model.rs`) invoke
// `decode_loop!` for their AR arms (plain AR turns; vision turns;
// the MTP-ineligible delta shapes). The MTP propose/verify loop
// those cores interleave with now lives in `crate::engine::mtp_turn`, so the
// AR macro lives HERE next to the MTP draft/verify helpers it shares.
// =============================================================================

/// Closures for model-specific operations in the AR decode loop.
///
/// `F`: forward pass — takes (input_ids [1,1], embedding_weight) → Result<(logits, needs_squeeze)>.
/// `E`: eval step — takes (next_token, logits, budget_forced) → schedules async eval.
///
/// The engine's generic flow uses [`crate::engine::backend::DecodeStep`];
/// `DecodeOps` is built by the `decode_loop!` call sites below.
pub(crate) struct DecodeOps<F, E>
where
    F: FnMut(&MxArray, &MxArray) -> Result<(MxArray, bool)>,
    E: Fn(&MxArray, &MxArray, bool),
{
    pub forward: F,
    pub eval_step: E,
}

/// Pipelined eager decode loop for the qwen3_5 dense/MoE MTP and vision
/// whole-turn cores (see the banner above; the engine's generic chat flow
/// uses [`crate::engine::decode::run_decode_loop`]).
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
        last_in_cache: $last_in_cache:ident,
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
                        logits = $crate::engine::penalties::apply_all_penalties(
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
                    && $crate::engine::decode::mtp_trace_logits()
                {
                    let logits_1d = if logits.ndim()? == 2 {
                        logits.squeeze(Some(&[0]))?
                    } else {
                        logits.clone()
                    };
                    let vocab = logits_1d.shape_at(0)?;
                    match $crate::engine::decode::trace_top2(
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
            // breadcrumb trail (step idx, sampled token, gen length).
            if step % 32 == 0 {
                tracing::info!(
                    "Qwen3.5 decode AR step={} sampled_token_id={} gen_len={}",
                    step,
                    token_id,
                    $gen.len(),
                );
            }

            // Streaming-only block (conditionally compiled via macro repetition)
            $(
                $last_r = _is_reasoning;

                if $cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                    $reason = String::from("cancelled");
                    $last_in_cache = step + 1 < $max;
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
                        Ok($crate::engine::types::ChatStreamChunk {
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

            if token_id == $eos || $p.extra_eos_ids.contains(&token_id) {
                $reason = String::from("stop");
                // The token just pushed was forwarded into the physical KV/GDN
                // cache iff this iteration ran a forward (`step + 1 < $max`).
                // On the final step (incl. `max_new_tokens == 1`, where step 0
                // is final) no forward runs, so the stop token is unforwarded.
                $last_in_cache = step + 1 < $max;
                break;
            }

            if let Some(reason) = $crate::sampling::check_repetition_cutoff(
                &$gen,
                $p.max_consecutive_tokens,
                $p.max_ngram_repeats,
                $p.ngram_size,
            ) {
                $reason = reason.to_string();
                $last_in_cache = step + 1 < $max;
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

/// Outcome of `crate::engine::mtp_turn::run_mtp_cycle` — the accepted
/// tokens for this cycle plus the requested / effective draft depth (used
/// by the engine loop to log / observe).
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
    /// Verify produced dense logits only, with no precomputed target. The
    /// eager dense and MoE verify paths build only this variant; the
    /// per-position accept loop derives any argmax/sparse target from the
    /// dense logits.
    pub(crate) fn logits_only(logits: MxArray, hiddens: MxArray) -> Self {
        Self {
            logits: Some(logits),
            hiddens,
            target_argmax: None,
            target_sparse: None,
        }
    }
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
