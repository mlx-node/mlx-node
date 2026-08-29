//! Qwen3.5 MTP (Multi-Token Prediction) speculative-decode machinery.
//!
//! Holds the cached MTP env-flag readers, the draft/verify helpers, and the
//! shared cycle data types (`MtpCommitAnchor` / `MtpCycleOutcome` /
//! `MtpVerifyOutput`) that only the MTP path consumes. The engine-owned
//! propose/verify loop (`crate::engine::mtp_turn::run_mtp_turn` /
//! `run_mtp_cycle`) drives them. The model-neutral AR decode infrastructure
//! lives in [`crate::engine`]; shared items needed by both the AR and MTP
//! paths (`apply_all_penalties`, the `mtp_trace_logits` / `trace_top2` trace
//! helpers) are imported from there.

use std::sync::OnceLock;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::decode::Top2;
use crate::nn::Embedding;
use crate::sampling::SamplingConfig;

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
// | `mtpAdaptiveDepth` (TS field) | OFF*    | per-session   |
// | `MLX_MTP_ADAPTIVE_DEPTH_MODE` | throughput | opt-IN EV  |
// | `MLX_MTP_CHAINED_CYCLES`      | M5+ ON, M1–M4 OFF | gen-gated |
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

/// Break-even first-draft acceptance rate for the MTP acceptance gate.
///
/// The gate signal is the per-position acceptance at draft slot 0 (the
/// FIRST draft's acceptance rate) — the depth-agnostic measure of whether
/// speculation pays. At depth 1 it equals the accepted/attempted ratio,
/// where the break-even is ~0.6 (a depth-1 verify costs ~1.4-1.6× an AR
/// step). At depth > 1 the verify amortizes over more tokens, so the
/// first-draft rate is the right comparison: a head accepting its first
/// draft at ~73% is profitable (docs' depth-3 workload: per-position
/// [0.735, 0.471, 0.235]) even though the accepted/attempted average is
/// only ~0.48. Measured first-draft acceptance on real checkpoints:
/// 1.000 (qwen3.5-4b, counting prompt), 0.756 (qwen3.5-4b, complex task),
/// 0.000 (qwen3.5-0.8b — MTP is a net loss there and the gate is what
/// keeps it from auto-enabling).
pub(crate) const MTP_ACCEPT_GATE_THRESHOLD: f64 = 0.6;

/// After this many consecutive gated (MTP-disabled) turns, the MTP
/// acceptance gate re-probes: the model resets its recorded acceptance to
/// `None` so the next turn runs speculation again. Without this, one hard
/// first turn would permanently disable MTP for the model's lifetime — a
/// genuinely weak head (e.g. qwen3.5-0.8b, acceptance 0.0) re-gates after
/// one probe, while a prompt-dependent head (qwen3.5-4b: 0.756 on complex
/// tasks, 1.0 on counting) can recover on an easier later turn.
pub(crate) const MTP_ACCEPT_GATE_REPROBE_TURNS: u32 = 3;

/// Cap on the aggregated acceptance-history sample the MTP gate carries.
///
/// Without a bound, lifetime counters let a long healthy phase drown out
/// a later degradation: after 10,000 accepted drafts a head would need
/// ~6,667 consecutive rejects just to pull the raw rate below 0.6.
/// When the aggregate exceeds this cap the counters are halved (rate
/// preserved, sample bounded), so the gate's confidence bound reflects
/// roughly the most recent ~512-1024 depth-1 drafts and reacts to a
/// sustained degradation within a turn or two.
pub(crate) const MTP_ACCEPT_GATE_HISTORY_CAP: u64 = 512;

/// Bound the aggregated gate history: halve both counters until the
/// sample fits under [`MTP_ACCEPT_GATE_HISTORY_CAP`], preserving the
/// rate while keeping the window finite. Integer-only (no float drift).
pub(crate) fn mtp_bound_gate_history(accepted: &mut u64, attempted: &mut u64) {
    while *attempted > MTP_ACCEPT_GATE_HISTORY_CAP {
        *accepted /= 2;
        *attempted /= 2;
    }
}

/// MTP acceptance gate — `MLX_MTP_ACCEPT_GATE` (default ON).
///
/// When ON, the model AGGREGATES the first-draft acceptance counts
/// (accepted / attempted at draft slot 0, see
/// [`MTP_ACCEPT_GATE_THRESHOLD`]) across completed FIXED depth-1 turns
/// and disables speculative decoding for the NEXT fixed depth-1 turn
/// only when an exact-binomial test at the 5% level shows the aggregate
/// is inconsistent with the break-even rate — falling back to the exact
/// target autoregressive path. The exact test preserves a UNIFORM
/// false-gate bound at every reachable sample size: a 1-cycle turn
/// (exactly 0.0 or 1.0), a 2-of-4 streak from a healthy 0.756 head
/// (~25% of turns), and 0-of-2 / 0-of-3 / 1-of-5 aggregates (~5.95% /
/// 1.45% / 1.43% false rates) never gate; only extreme aggregates like
/// 0-of-4 (false rate ~0.35%) do. The aggregate is BOUNDED
/// ([`MTP_ACCEPT_GATE_HISTORY_CAP`]) so a long healthy phase cannot
/// drown out a later degradation. The gate is **depth-1-scoped and
/// adaptive-exempt**: the 0.6 threshold is depth-1 calibrated, and at
/// depth > 1 (or when `mtpAdaptiveDepth` sweeps depths 1-5) the verify
/// cost vs deeper-slot acceptance economics are not captured by a single
/// threshold — such turns are never gated and do not publish gate
/// history. The first turn of a model load has no history and probes;
/// after [`MTP_ACCEPT_GATE_REPROBE_TURNS`] consecutive gated turns the
/// gate re-probes. A full session reset (`reset_caches`) clears the
/// history so a new independent chat starts fresh. The state is
/// **per-model** (one loaded checkpoint, shared by every ChatSession over
/// it), not per-session. The gate reuses the existing "unsupported
/// combination disables speculation for this turn" routing.
///
/// Opt-out: `MLX_MTP_ACCEPT_GATE=0` (or `false` / `off`) disables the
/// gate so MTP always runs when requested. Read once per process and
/// cached.
pub(crate) fn mtp_accept_gate_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| match std::env::var("MLX_MTP_ACCEPT_GATE") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true, // default ON — a head that never accepts must not tax every turn
    })
}

/// Exact-binomial MTP acceptance-gate decision: gate only when the
/// observed (accepted, attempted) first-draft counts are inconsistent
/// with the break-even rate at the 5% level — i.e. when
/// `P(Binomial(attempted, 0.6) <= accepted) < 0.05`.
///
/// This preserves a UNIFORM false-gate bound at every reachable
/// aggregate size, unlike a fixed sample floor (which only protects
/// exactly at its own threshold: 1-of-5 would otherwise gate with a
/// ~1.43% false rate for a 0.756 head) or the normal-approximation
/// Wilson bound (which under-covers small/extreme samples). Verified
/// behavior: 0-of-1/2/3 and 1-of-4, 2-of-4, 1-of-5, 2-of-5 never gate
/// (p-value >= 0.064); 0-of-4 (0.0256) and 0-of-5+ gate — the false
/// rate for the documented 0.756 head stays ~0.35% and below.
pub(crate) fn mtp_accept_gate_blocks(accepted: u64, attempted: u64) -> bool {
    if attempted == 0 {
        return false;
    }
    let n = attempted;
    let k = accepted.min(n);
    let p = MTP_ACCEPT_GATE_THRESHOLD;
    let q = 1.0 - p;
    // Iterative binomial PMF: pmf(0) = q^n;
    // pmf(i+1) = pmf(i) * (n-i)/(i+1) * p/q. Accumulate the lower tail
    // and early-exit once it reaches the significance level (later terms
    // are all positive, so the tail can only grow).
    let mut pmf = q.powi(n as i32);
    let mut cdf = pmf;
    for i in 1..=k {
        pmf *= (n - i + 1) as f64 / i as f64 * p / q;
        cdf += pmf;
        if cdf >= 0.05 {
            return false;
        }
    }
    cdf < 0.05
}

/// MTP acceptance gate — see [`mtp_accept_gate_enabled`].
/// `false` means the aggregated first-draft acceptance rate is below
/// the break-even bound WITH 95% confidence, so this turn should run
/// plain AR instead of paying the verify cost for zero speedup.
/// Depth-1-scoped: the 0.6 threshold is depth-1 calibrated, and at
/// depth > 1 the verify cost vs deeper-slot acceptance economics are
/// not captured by a single threshold — the gate never blocks a
/// depth>1 turn. First turn (no history) probes; after
/// [`MTP_ACCEPT_GATE_REPROBE_TURNS`] consecutive gated turns the gate
/// re-probes; the env knob disables the gate entirely. The counters are
/// `&mut` because a blocked turn advances the gated-turn counter and
/// may trigger the re-probe reset.
pub(crate) fn mtp_gate_allows(
    accepted: &mut u64,
    attempted: &mut u64,
    gated_turns: &mut u32,
    requested_depth: u32,
) -> bool {
    if !mtp_accept_gate_enabled() || requested_depth > 1 {
        return true;
    }
    if *attempted == 0 {
        return true; // no history — probe
    }
    if !mtp_accept_gate_blocks(*accepted, *attempted) {
        return true; // not confident the head is below break-even
    }
    *gated_turns += 1;
    if *gated_turns >= MTP_ACCEPT_GATE_REPROBE_TURNS {
        *gated_turns = 0;
        *accepted = 0;
        *attempted = 0; // re-probe next turn
    }
    false
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
// The gate is unconditional on the deterministic path; tests override it
// per-thread so they can drive the per-position accept loop instead.
#[cfg(not(test))]
#[inline]
pub(crate) fn sparse_accept_gate() -> bool {
    true
}

#[cfg(test)]
thread_local! {
    /// Test-only override for [`sparse_accept_gate`]. `None` keeps the
    /// production answer; `Some(b)` forces the gate so a test
    /// deterministically exercises the intended accept path.
    static TEST_FORCE_SPARSE_ACCEPT: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn sparse_accept_gate() -> bool {
    TEST_FORCE_SPARSE_ACCEPT
        .with(std::cell::Cell::get)
        .unwrap_or(true)
}

/// RAII guard that forces [`sparse_accept_gate`] for the current thread and
/// restores the prior value on drop (panic-safe). Used by the T=0 safety
/// test to guarantee it drives the production sparse-accept commit path.
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
/// `F`: forward pass — takes (input_ids [1,1], embedding) → Result<(logits, needs_squeeze)>.
/// `E`: eval step — takes (next_token, logits, budget_forced) → schedules async eval.
///
/// The engine's generic flow uses [`crate::engine::backend::DecodeStep`];
/// `DecodeOps` is built by the `decode_loop!` call sites below.
pub(crate) struct DecodeOps<F, E>
where
    F: FnMut(&MxArray, &Embedding) -> Result<(MxArray, bool)>,
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
///
/// The optional `cancel:` fragment (H2, MUTUALLY EXCLUSIVE with
/// `streaming:` by convention — streaming already polls its own flag)
/// takes an `Option<&AtomicBool>` and compiles in a per-step cancel poll
/// at the SAME loop position as the streaming block's poll, breaking
/// with `finish_reason = "cancelled"`. SYNC whole-turn cores pass their
/// installed `self.turn_cancel` clone here so a client disconnect stops
/// the decode instead of burning the whole budget.
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
        $(, cancel: $cancel_flag:expr)?
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
            $profiler.step();
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

            // Sync-turn cancel poll (H2; conditionally compiled via the
            // `cancel:` macro repetition). Same snapshot point as the
            // streaming block's poll below: after the sampled token is
            // pushed/observed, before EOS/repetition checks.
            $(
                if $cancel_flag
                    .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
                {
                    $reason = String::from("cancelled");
                    $last_in_cache = step + 1 < $max;
                    break;
                }
            )?

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
                            thinking_enabled: None,
                            num_tokens: None,
                            prompt_tokens: None,
                            reasoning_tokens: None,
                            raw_text: None,
                            public_raw_text: None,
                            text_authoritative: None,
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

/// Dense verify logits `[1, depth+1, vocab]` plus per-position hiddens
/// `[1, depth+1, hidden]`. The accept loop derives its argmax / sparse
/// target from the logits.
pub(crate) struct MtpVerifyOutput {
    pub logits: MxArray,
    pub hiddens: MxArray,
}

impl MtpVerifyOutput {
    pub(crate) fn logits_only(logits: MxArray, hiddens: MxArray) -> Self {
        Self { logits, hiddens }
    }
}

#[cfg(test)]
mod mtp_accept_gate_tests {
    use super::{MTP_ACCEPT_GATE_HISTORY_CAP, mtp_accept_gate_blocks, mtp_bound_gate_history};

    #[test]
    fn confidence_gate_ignores_undersampled_and_marginal_rates() {
        // 2-of-4 and 1-of-4 are NOT confidently below break-even — a
        // healthy 0.756 head hits 2-of-4 in ~25% of 4-cycle turns, so
        // those must not gate. Small aggregates never gate, even at
        // rate 0 (0-of-2 has p-value 0.16, 0-of-3 0.064).
        assert!(!mtp_accept_gate_blocks(2, 4), "2-of-4 must not gate");
        assert!(!mtp_accept_gate_blocks(1, 4), "1-of-4 must not gate");
        assert!(!mtp_accept_gate_blocks(0, 1), "0-of-1 must not gate");
        assert!(!mtp_accept_gate_blocks(0, 2), "0-of-2 must not gate");
        assert!(!mtp_accept_gate_blocks(0, 3), "0-of-3 must not gate");
        // 1-of-5 (p-value 0.087) and 2-of-5 must not gate either — the
        // exact test preserves the false-gate bound at every size, not
        // just at four samples.
        assert!(
            !mtp_accept_gate_blocks(1, 5),
            "1-of-5 must not gate (~1.43% false rate for a 0.756 head)"
        );
        assert!(!mtp_accept_gate_blocks(2, 5), "2-of-5 must not gate");
    }

    #[test]
    fn confidence_gate_gates_only_extreme_rates() {
        // 0-of-4 gates (p-value 0.0256); the false-gate rate for a
        // 0.756 head is P(0-of-4) ≈ 0.35%.
        assert!(mtp_accept_gate_blocks(0, 4), "0-of-4 must gate");
        assert!(mtp_accept_gate_blocks(0, 64), "0-of-64 must gate");
        // A clearly-below-break-even head gates once the sample is large
        // enough for the test to have power (19/64 = 0.297).
        assert!(mtp_accept_gate_blocks(19, 64), "19-of-64 must gate");
    }

    #[test]
    fn confidence_gate_never_gates_a_healthy_head() {
        // The documented 0.756 head: at ~0.75 observed rates the p-value
        // is large at any sample size.
        assert!(!mtp_accept_gate_blocks(3, 4));
        assert!(!mtp_accept_gate_blocks(24, 32));
        assert!(!mtp_accept_gate_blocks(193, 256));
    }

    #[test]
    fn gate_history_is_bounded_and_preserves_rate() {
        // A long healthy phase must not persist forever: halving preserves
        // the rate while capping the window.
        let (mut a, mut t) = (10_000u64, 20_000u64); // rate 0.5
        mtp_bound_gate_history(&mut a, &mut t);
        assert!(t <= MTP_ACCEPT_GATE_HISTORY_CAP, "sample bounded");
        assert!(t > 0, "sample never empties");
        let rate = a as f64 / t as f64;
        assert!((rate - 0.5).abs() < 0.01, "rate preserved, got {rate}");
    }

    #[test]
    fn gate_history_under_cap_is_untouched() {
        let (mut a, mut t) = (300u64, 500u64); // under cap
        mtp_bound_gate_history(&mut a, &mut t);
        assert_eq!((a, t), (300, 500));
    }
}

#[cfg(test)]
mod decode_loop_sync_cancel_tests {
    use std::cell::Cell;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use napi::bindgen_prelude::Result;

    use super::DecodeOps;
    use crate::array::MxArray;
    use crate::engine::params::ChatParams;
    use crate::engine::penalties::ReasoningTracker;
    use crate::nn::Embedding;
    use crate::sampling::SamplingConfig;
    use crate::stream::{DeviceType, Stream};

    /// Greedy T=0 params with every penalty/cutoff neutral, mirroring the
    /// engine turn tests' `greedy_params`.
    fn greedy_params(max_new_tokens: i32) -> ChatParams {
        ChatParams {
            cache_salt: 0,
            cache_owner_id: String::new(),
            cache_root_owner_id: None,
            max_new_tokens,
            repetition_penalty: 1.0,
            repetition_context_size: 0,
            presence_penalty: 0.0,
            presence_context_size: 0,
            frequency_penalty: 0.0,
            frequency_context_size: 0,
            max_consecutive_tokens: 0,
            max_ngram_repeats: 0,
            ngram_size: 0,
            sampling_config: Some(SamplingConfig {
                temperature: Some(0.0),
                top_k: Some(0),
                top_p: Some(1.0),
                min_p: Some(0.0),
            }),
            report_performance: false,
            reuse_cache: true,
            include_reasoning: true,
            extra_eos_ids: Vec::new(),
            enable_mtp: false,
            mtp_depth: 0,
            mtp_adaptive_depth: false,
        }
    }

    /// `[1, vocab]` logits whose T=0 argmax is `id`.
    fn logits_row(vocab: i64, id: i32) -> Result<MxArray> {
        let mut row = vec![0.0f32; vocab as usize];
        row[id as usize] = 1.0;
        MxArray::from_float32(&row, &[1, vocab])
    }

    /// H2: the `cancel:` fragment must stop the flat AR `decode_loop!`
    /// (the dense/MoE SYNC whole-turn AR arm) at the per-step poll — the
    /// SAME loop position as the `streaming:` block's poll — with
    /// `finish_reason == "cancelled"`. The mock forward flips the shared
    /// flag, emulating a client disconnect landing during the pipelined
    /// step-0 forward.
    ///
    /// Named mutation this catches: reverting the `cancel:` fragment (or
    /// a family call site dropping `cancel: turn_cancel.as_deref()` — the
    /// pre-fix behavior): the loop then runs the full 16-token budget and
    /// finishes "length" with 16 committed tokens.
    #[test]
    fn decode_loop_sync_cancel_flag_breaks_with_finish_reason_cancelled() -> Result<()> {
        let vocab: i64 = 8;
        let eos_id: u32 = 7;
        let max_new_tokens: i32 = 16;

        let cancel = Arc::new(AtomicBool::new(false));
        let flip = Arc::clone(&cancel);
        let forward_calls = Cell::new(0usize);

        let mut ops = DecodeOps {
            forward: |_ids: &MxArray, _emb: &Embedding| -> Result<(MxArray, bool)> {
                forward_calls.set(forward_calls.get() + 1);
                // Disconnect lands DURING the step-0 forward — the poll
                // right after the seed's push/observe must see it.
                flip.store(true, Ordering::Relaxed);
                // Never-EOS argmax (6): an uncancelled loop walks the
                // whole budget.
                Ok((logits_row(vocab, 6)?, false))
            },
            eval_step: |_t: &MxArray, _l: &MxArray, _b: bool| {},
        };

        let mut y = MxArray::from_int32(&[3], &[1])?;
        let embedding = Embedding::from_weight(&MxArray::from_float32(
            &vec![0.0f32; vocab as usize],
            &[vocab, 1],
        )?)?;
        let p = greedy_params(max_new_tokens);
        let mut tracker = ReasoningTracker::new(false, None, None);
        let mut profiler = crate::decode_profiler::DecodeProfiler::new("decode_loop_test", "test");
        let mut generated: Vec<u32> = Vec::new();
        let mut hist: Vec<u32> = Vec::new();
        let mut finish_reason = String::from("length");
        let mut last_in_cache = true;
        let mut first_tok: Option<std::time::Instant> = None;
        let generation_stream = Stream::new(DeviceType::Gpu);

        decode_loop!(
            ops: ops,
            y: y,
            embedding_weight: embedding,
            params: p,
            reasoning_tracker: tracker,
            profiler: profiler,
            max_new_tokens: max_new_tokens,
            eos_id: eos_id,
            generated_tokens: generated,
            token_history: hist,
            finish_reason: finish_reason,
            last_in_cache: last_in_cache,
            first_token_instant: first_tok,
            report_perf: false,
            generation_stream: generation_stream,
            cancel: Some(cancel.as_ref())
        );

        // Silence the "value assigned is never read" path: the loop
        // reassigns `y` on non-terminal steps; a cancelled step 0 never
        // does.
        let _ = y;

        assert_eq!(finish_reason, "cancelled");
        assert_eq!(
            generated,
            vec![3],
            "only the seed commits — the poll fires on the seed's own step"
        );
        assert_eq!(hist, generated, "history stays in lockstep");
        assert!(
            last_in_cache,
            "the pipelined step-0 forward already ran (step + 1 < max), so the \
             seed's K/V is in the cache"
        );
        assert_eq!(
            forward_calls.get(),
            1,
            "exactly the pipelined step-0 forward ran before the poll broke the loop"
        );
        Ok(())
    }
}
