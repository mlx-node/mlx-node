//! Adaptive MTP depth policy.
//!
//! Per-session policy that picks the MTP draft depth `D ∈ {1..=5}` for
//! each cycle by maintaining an EMA of the effective decode rate
//! (`accepted_tokens / cycle_wall_ns`) per depth.
//!
//! State machine: `Explore → Full → {NeighborProbe | Reduced → Probe}`.
//!
//! * `Explore` — initial bootstrap. Sweeps every depth in `{1..=5}`
//!   for `MIN_COLD_SAMPLES` cycles each so every per-depth EMA gets
//!   real observations BEFORE the first hill-climb decision. Without
//!   this the policy is stuck at its seed depth, since `pick_depth()`
//!   returns `current_depth`, so non-current depths never get samples
//!   and can never win the hill-climb.
//! * `Full` — run at `current_depth` and re-evaluate the hill-climb
//!   every cycle. Every `FULL_REPROBE_INTERVAL` cycles, launch a
//!   `NeighborProbe` burst to refresh adjacent depths' EMAs.
//! * `NeighborProbe` — short burst at a neighbor of `current_depth`
//!   to keep its EMA current. Returns to `Full`.
//! * `Reduced` / `Probe` — DFlash-style 3-state fallback for
//!   low-acceptance prompts. Drop to `MIN_DEPTH`, periodically probe
//!   back up to `current_depth`.
//!
//! Reference: `dflash-mlx/dflash_mlx/engine/spec_epoch.py`
//! `_AdaptiveBlockPolicy`. The DFlash policy is keyed on a "block length"
//! (full vs. min vs. probe), where ours is keyed on draft depth. The
//! drop-acceptance threshold (`0.75`) and probe-interval pattern are lifted
//! directly.
//!
//! The MTP verify pass is pure-Rust eager — there are no per-depth
//! compiled graphs to pre-warm, so switching depth between cycles carries
//! no setup cost and the policy can swing freely.

use std::time::Duration;

/// Min and max draft depths supported by the verify FFI contract.
/// Mirrors the clamp in `extract_chat_params`.
pub(crate) const MIN_DEPTH: u8 = 1;
pub(crate) const MAX_DEPTH: u8 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AdaptiveDepthMode {
    Throughput,
    ExpectedValue,
}

pub(crate) fn adaptive_depth_mode_from_env() -> AdaptiveDepthMode {
    match std::env::var("MLX_MTP_ADAPTIVE_DEPTH_MODE") {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "expected-value" | "expected_value" | "ev" => AdaptiveDepthMode::ExpectedValue,
            _ => AdaptiveDepthMode::Throughput,
        },
        Err(_) => AdaptiveDepthMode::Throughput,
    }
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn env_u8(name: &str, default: u8) -> u8 {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<u8>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(raw) => {
            let value = raw.trim();
            value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("on")
        }
        Err(_) => default,
    }
}

fn env_accept_priors(max_depth: u8) -> [f64; MAX_DEPTH as usize] {
    let mut priors = [0.70, 0.40, 0.18, 0.10, 0.05];
    if let Ok(raw) = std::env::var("MLX_MTP_EV_ACCEPT_PRIORS") {
        let mut last = priors[0];
        for (idx, part) in raw.split(',').enumerate().take(MAX_DEPTH as usize) {
            if let Ok(value) = part.trim().parse::<f64>()
                && value.is_finite()
            {
                last = value.clamp(0.0, 1.0);
                priors[idx] = last;
            }
        }
        let max = max_depth.clamp(MIN_DEPTH, MAX_DEPTH) as usize;
        for slot in priors.iter_mut().take(MAX_DEPTH as usize).skip(max) {
            *slot = last;
        }
    }
    priors
}

/// DFlash drop threshold: when rolling acceptance over a window drops
/// below this, we leave `Full` for `Reduced`. Reference:
/// `_ADAPTIVE_DROP_ACCEPTANCE_THRESHOLD = 0.75`.
const DROP_ACCEPT_THRESHOLD: f64 = 0.75;

/// Rolling window for acceptance + rate stats inside a single state
/// (matches DFlash's `window_size = 4`).
const STATE_WINDOW: usize = 4;

/// Cycles to spend at the `Reduced` depth before probing back up to a
/// deeper depth. DFlash uses 24; we use a much shorter probe interval
/// because our cycles are O(50 ms) each and 200-token smoke turns only
/// run ~50-100 cycles total — a 24-cycle lockout would never probe.
const REDUCED_CYCLES_BEFORE_PROBE: u32 = 8;

/// Probe length: how many cycles to spend at the probe (deeper) depth
/// before deciding whether to commit (return to `Full` at that depth)
/// or revert (back to `Reduced`).
const PROBE_CYCLES: u32 = 3;

/// Minimum cycles a depth must be tried before EMA-decay kicks in for
/// that depth's rate. This guards against cold-start jitter where the
/// first observed sample for a depth is anomalous (e.g. cache warmup).
const MIN_COLD_SAMPLES: u32 = 2;

/// EMA decay factor: `new_ema = (1 - alpha) * old_ema + alpha * sample`.
/// Higher = more responsive, noisier. 0.3 picked empirically — gives
/// the policy ~10 cycles of meaningful history before swing.
const EMA_ALPHA: f64 = 0.3;

/// Cycles to spend in `Full` before launching a `NeighborProbe` burst
/// to refresh adjacent depths' EMAs. Picked so that on a 200-token
/// turn (~50 cycles total) we get one or two refresh rounds.
const FULL_REPROBE_INTERVAL: u32 = 20;

/// Cycles spent at a neighbor depth during a `NeighborProbe` burst.
/// Single cycle is enough to update the EMA at a rate-stable depth;
/// `MIN_COLD_SAMPLES` ensures we still hit the cold-mean path if the
/// neighbor was never seen before.
const NEIGHBOR_PROBE_CYCLES: u32 = MIN_COLD_SAMPLES;

/// Adaptive state machine.
///
/// Why an explicit `Explore` state: the engine's `run_mtp_cycle`
/// only ever records observations at the depth `pick_depth()` returned, so
/// without bootstrap the EMA for unsampled depths stays 0.0 forever and the
/// hill-climb can never discover a non-seed depth (a 3-seeded policy would
/// only oscillate between {3, 1}, never trying 2/4/5).
///
/// `Explore` runs through every depth in `{1..=5}` for `MIN_COLD_SAMPLES`
/// cycles each at decode start, seeding all per-depth EMAs, then commits to
/// the best-rate depth in `Full`. Periodic `NeighborProbe` re-checks adjacent
/// depths so a mid-decode regime change (e.g. high-acceptance code → low-
/// acceptance prose) gets re-discovered without dropping all the way to
/// `Reduced`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AdaptiveState {
    /// Bootstrap: cycle through each depth in `{1..=5}` for
    /// `MIN_COLD_SAMPLES` cycles each. After all depths are seeded,
    /// transition to `Full` at the best-EMA depth.
    Explore,
    /// Run at the policy's current chosen depth (`current_depth`).
    /// Re-evaluates the hill-climb every cycle. Periodically (every
    /// `FULL_REPROBE_INTERVAL` cycles) drops a `NeighborProbe` to
    /// re-test the immediately adjacent depths so we can adapt to
    /// drift in the optimum without paying a full `Reduced` cycle.
    Full,
    /// Run a single cycle at a neighbor of `current_depth` to refresh
    /// that depth's EMA. After `NEIGHBOR_PROBE_CYCLES` cycles, return
    /// to `Full` (which re-runs the hill-climb with the fresh data).
    NeighborProbe,
    /// Run at `MIN_DEPTH`. Entered when rolling acceptance at `Full`
    /// dropped below `DROP_ACCEPT_THRESHOLD`. After
    /// `REDUCED_CYCLES_BEFORE_PROBE` cycles, transitions to `Probe`.
    Reduced,
    /// One-shot test: temporarily run at `probe_depth` (the depth we
    /// dropped from). After `PROBE_CYCLES` cycles, compare the probe's
    /// rate EMA against `Reduced`'s rate and either commit back to
    /// `Full` at `probe_depth` (probe won) or revert to `Reduced`
    /// (probe lost).
    Probe,
}

/// Per-cycle observation passed to the policy.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CycleStats {
    /// Depth used this cycle (the `depth` arg to `run_mtp_cycle`).
    pub depth: u8,
    /// Tokens committed this cycle. On full-accept this is `D + 1`
    /// (the D drafts + bonus). On partial-accept it's `K + 1`
    /// (K accepted drafts + residual). Range: `[1, D + 1]`.
    pub committed: u32,
    /// Wall-clock time spent on this cycle's draft+verify path.
    /// Measured by the caller with `std::time::Instant`. Must be
    /// non-zero (the policy divides by it).
    pub wall_ns: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DraftMetrics {
    pub top1_prob_topk: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpectedValueDecision {
    pub continue_drafting: bool,
    pub next_depth: u8,
    pub expected_extra_accept: f64,
    pub required_extra_accept: f64,
}

/// MTPLX-style cost-aware intra-cycle depth gate.
///
/// The throughput state machine chooses one depth before the cycle starts. This
/// policy instead lets the draft loop start at `max_depth`, then stop after the
/// base depth when the next draft slot is unlikely to repay its draft+verify
/// cost. It never changes sampling semantics: it only shortens the proposal
/// prefix before target verification.
pub(crate) struct ExpectedValueDepthPolicy {
    max_depth: u8,
    base_depth: u8,
    accept_ewma: [f64; MAX_DEPTH as usize],
    ewma_alpha: f64,
    draft_cost_s: f64,
    extra_verify_cost_s: f64,
    baseline_tok_s: f64,
    safety_margin: f64,
    confidence_weight: f64,
    min_extra_accept_probability: f64,
    allow_deepen: bool,
}

impl ExpectedValueDepthPolicy {
    pub fn new(max_depth: u8) -> Self {
        let max_depth = max_depth.clamp(MIN_DEPTH, MAX_DEPTH);
        let base_depth = env_u8("MLX_MTP_EV_BASE_DEPTH", 1).clamp(MIN_DEPTH, max_depth);
        let ewma_alpha = env_f64("MLX_MTP_EV_EWMA_ALPHA", 0.12).clamp(0.001, 1.0);
        Self {
            max_depth,
            base_depth,
            accept_ewma: env_accept_priors(max_depth),
            ewma_alpha,
            draft_cost_s: env_f64("MLX_MTP_EV_DRAFT_COST_S", 0.0048).max(0.0),
            extra_verify_cost_s: env_f64("MLX_MTP_EV_EXTRA_VERIFY_COST_S", 0.0060).max(0.0),
            baseline_tok_s: env_f64("MLX_MTP_EV_BASELINE_TOK_S", 24.0).max(1e-6),
            safety_margin: env_f64("MLX_MTP_EV_SAFETY_MARGIN", 0.10).max(0.0),
            confidence_weight: env_f64("MLX_MTP_EV_CONFIDENCE_WEIGHT", 0.25).clamp(0.0, 1.0),
            min_extra_accept_probability: env_f64("MLX_MTP_EV_MIN_EXTRA_ACCEPT_PROBABILITY", 0.30)
                .clamp(0.0, 1.0),
            // Default ON: intra-cycle deepening produces byte-identical T=0
            // output ON vs OFF — it only extends the proposal prefix; the
            // accept/commit layer is unchanged. `MLX_MTP_EV_ALLOW_DEEPEN=0`
            // opts out. Only consulted in EV mode
            // (MLX_MTP_ADAPTIVE_DEPTH_MODE=ev); the default Throughput mode
            // never reaches this gate.
            allow_deepen: env_bool("MLX_MTP_EV_ALLOW_DEEPEN", true),
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        max_depth: u8,
        base_depth: u8,
        accept_ewma: [f64; MAX_DEPTH as usize],
        min_extra_accept_probability: f64,
    ) -> Self {
        let mut policy = Self::new(max_depth);
        policy.base_depth = base_depth.clamp(MIN_DEPTH, policy.max_depth);
        policy.accept_ewma = accept_ewma;
        policy.min_extra_accept_probability = min_extra_accept_probability.clamp(0.0, 1.0);
        policy.draft_cost_s = 0.0;
        policy.extra_verify_cost_s = 0.0;
        policy.allow_deepen = true;
        policy
    }

    /// Test-only override for the intra-cycle deepen gate. The production
    /// default is sourced from `MLX_MTP_EV_ALLOW_DEEPEN` (in `new`); tests
    /// must NOT mutate that env var (unsafe in Rust 2024, racy under parallel
    /// `cargo test`, and the value may be cached). This setter drives the
    /// shallow (`false`, stop at `base_depth`) vs deep (`true`, extend to
    /// `max_depth`) policies in-process.
    #[cfg(test)]
    pub(crate) fn set_allow_deepen(&mut self, allow_deepen: bool) {
        self.allow_deepen = allow_deepen;
    }

    pub fn max_depth(&self) -> u8 {
        self.max_depth
    }

    pub fn should_continue_after_draft(
        &self,
        drafted_depth: usize,
        cycle_max_depth: usize,
        metrics: DraftMetrics,
    ) -> ExpectedValueDecision {
        let drafted_depth = drafted_depth.clamp(1, self.max_depth as usize);
        let cycle_max_depth = cycle_max_depth.clamp(1, self.max_depth as usize);
        let next_depth = (drafted_depth + 1).min(self.max_depth as usize) as u8;

        if drafted_depth >= cycle_max_depth {
            return ExpectedValueDecision {
                continue_drafting: false,
                next_depth,
                expected_extra_accept: 0.0,
                required_extra_accept: 0.0,
            };
        }

        if drafted_depth < self.base_depth as usize {
            return ExpectedValueDecision {
                continue_drafting: true,
                next_depth,
                expected_extra_accept: 1.0,
                required_extra_accept: 0.0,
            };
        }

        if !self.allow_deepen {
            return ExpectedValueDecision {
                continue_drafting: false,
                next_depth,
                expected_extra_accept: 0.0,
                required_extra_accept: self.min_extra_accept_probability,
            };
        }

        let mut prefix_probability = 1.0;
        for idx in 0..drafted_depth {
            prefix_probability *= self.accept_ewma[idx].clamp(0.0, 1.0);
        }
        let next_probability = self.accept_ewma[next_depth as usize - 1].clamp(0.0, 1.0);
        let confidence_factor = self.confidence_factor(metrics);
        let expected_extra_accept =
            (prefix_probability * next_probability * confidence_factor).clamp(0.0, 0.999);
        let extra_cost_s = self.draft_cost_s + self.extra_verify_cost_s;
        let required_extra_accept = self
            .min_extra_accept_probability
            .max(extra_cost_s * self.baseline_tok_s * (1.0 + self.safety_margin));
        ExpectedValueDecision {
            continue_drafting: expected_extra_accept >= required_extra_accept,
            next_depth,
            expected_extra_accept,
            required_extra_accept,
        }
    }

    pub fn observe(&mut self, attempted_depth: usize, accepted_drafts: usize) {
        let attempted_depth = attempted_depth.clamp(1, self.max_depth as usize);
        let accepted_drafts = accepted_drafts.min(attempted_depth);
        for idx in 0..attempted_depth {
            let accepted = if accepted_drafts > idx { 1.0 } else { 0.0 };
            self.accept_ewma[idx] =
                (1.0 - self.ewma_alpha) * self.accept_ewma[idx] + self.ewma_alpha * accepted;
        }
    }

    fn confidence_factor(&self, metrics: DraftMetrics) -> f64 {
        let Some(top1_prob) = metrics.top1_prob_topk else {
            return 1.0;
        };
        let centered = 2.0 * top1_prob.clamp(0.0, 1.0) - 1.0;
        (1.0 + self.confidence_weight * centered).clamp(0.50, 1.50)
    }
}

impl CycleStats {
    /// Acceptance ratio for this cycle: `accepted_drafts / depth`.
    /// `accepted_drafts = committed - 1` (one of the `committed`
    /// tokens is always the residual/bonus, not a draft accept).
    pub fn acceptance(&self) -> f64 {
        if self.depth == 0 {
            return 0.0;
        }
        let accepted_drafts = self.committed.saturating_sub(1) as f64;
        accepted_drafts / (self.depth as f64)
    }

    /// Effective decode rate (tokens / second). The policy's primary
    /// objective function.
    pub fn rate_tps(&self) -> f64 {
        if self.wall_ns == 0 {
            return 0.0;
        }
        (self.committed as f64) / Duration::from_nanos(self.wall_ns).as_secs_f64()
    }
}

/// Rolling-window stats for a single (state, depth) bucket.
#[derive(Clone, Debug, Default)]
struct WindowStats {
    acceptances: Vec<f64>,
    rates: Vec<f64>,
}

impl WindowStats {
    fn push(&mut self, c: &CycleStats) {
        self.acceptances.push(c.acceptance());
        self.rates.push(c.rate_tps());
        if self.acceptances.len() > STATE_WINDOW {
            self.acceptances.remove(0);
            self.rates.remove(0);
        }
    }

    fn mean_acceptance(&self) -> f64 {
        if self.acceptances.is_empty() {
            return 1.0; // No data → assume "good" so we don't drop prematurely.
        }
        self.acceptances.iter().sum::<f64>() / (self.acceptances.len() as f64)
    }

    fn mean_rate(&self) -> f64 {
        if self.rates.is_empty() {
            return 0.0;
        }
        self.rates.iter().sum::<f64>() / (self.rates.len() as f64)
    }

    fn full(&self) -> bool {
        self.acceptances.len() >= STATE_WINDOW
    }

    fn clear(&mut self) {
        self.acceptances.clear();
        self.rates.clear();
    }
}

/// Adaptive depth policy state. One per chat session / decode-loop
/// invocation. Lives on the stack inside the engine's `run_mtp_turn` loop
/// and is dropped at decode end.
pub(crate) struct AdaptiveDepthPolicy {
    /// Per-depth EMA of tokens/sec rate. Indexed by `depth - 1`.
    rate_ema: [f64; MAX_DEPTH as usize],
    /// Per-depth observation count. EMA-decay only kicks in once a
    /// depth has been observed at least `MIN_COLD_SAMPLES` times.
    sample_count: [u32; MAX_DEPTH as usize],
    /// Per-depth `total_cycles` value at the most recent observation of
    /// that depth. Drives the age-based freshness gate in
    /// `stale_neighbor` — a neighbor whose EMA has not been refreshed
    /// for `>= FULL_REPROBE_INTERVAL` cycles is considered drifted and
    /// becomes a reprobe candidate. `0` means "never sampled" (which,
    /// once `total_cycles >= FULL_REPROBE_INTERVAL`, also reads as
    /// aged-out, matching the under-seeded path). Indexed by `depth - 1`.
    last_sampled_cycle: [u64; MAX_DEPTH as usize],
    /// The depth currently being used (and the depth the policy would
    /// commit to in `Full` state).
    current_depth: u8,
    /// State machine state.
    state: AdaptiveState,
    /// Rolling-window stats for the current state. Cleared on every
    /// state transition.
    window: WindowStats,
    /// In `Explore`: tracks which depth is currently being sampled.
    /// In `Reduced`: cycles elapsed since the last `Probe`. Used to
    /// trigger the next probe at `REDUCED_CYCLES_BEFORE_PROBE`.
    /// In `Probe` / `NeighborProbe`: cycles spent in that burst.
    /// In `Full`: cycles since entering Full (drives `FULL_REPROBE_INTERVAL`).
    cycles_in_state: u32,
    /// Explore state: the depth to sample on this cycle. Advanced
    /// every `MIN_COLD_SAMPLES` cycles. Range `[MIN_DEPTH, MAX_DEPTH]`.
    explore_depth: u8,
    /// Depth to use during the next `Probe` or `NeighborProbe`. In
    /// `Reduced`→`Probe`, set to the depth we dropped from. In
    /// `Full`→`NeighborProbe`, set to a neighbor of `current_depth`.
    probe_depth: u8,
    /// Average rate measured during the last `Reduced` burst — used as
    /// the comparison baseline when `Probe` exits.
    reduced_baseline_rate: f64,
    /// Average rate measured during the in-progress `Probe` — populated
    /// by `record_cycle` while `state == Probe`.
    probe_rate_ema: f64,
    /// Total cycles seen across the lifetime of this policy. Diagnostic
    /// only; included in `debug!` logs.
    total_cycles: u64,
}

impl AdaptiveDepthPolicy {
    /// Construct with an initial depth (typically the user's
    /// `mtpDepth` value or the default `3`).
    ///
    /// Starts in `Explore` state, which sweeps every depth in `{1..=5}`
    /// for `MIN_COLD_SAMPLES` cycles each before transitioning to `Full`
    /// at the best-rate depth, guaranteeing every depth gets observations
    /// (a `Full`-start design could never sample non-seed depths because
    /// `pick_depth()` only returns `current_depth`).
    pub fn new(initial_depth: u8) -> Self {
        let d = initial_depth.clamp(MIN_DEPTH, MAX_DEPTH);
        Self {
            rate_ema: [0.0; MAX_DEPTH as usize],
            sample_count: [0; MAX_DEPTH as usize],
            last_sampled_cycle: [0; MAX_DEPTH as usize],
            current_depth: d,
            state: AdaptiveState::Explore,
            window: WindowStats::default(),
            cycles_in_state: 0,
            explore_depth: MIN_DEPTH,
            probe_depth: d,
            reduced_baseline_rate: 0.0,
            probe_rate_ema: 0.0,
            total_cycles: 0,
        }
    }

    /// Construct a NO-OP policy that always returns `fixed_depth`.
    /// Used in tests to exercise the bounds-clamping path; the engine's
    /// `run_mtp_cycle` skips `record_cycle` when adaptive
    /// is off so the same `pick_depth()` always returns the seed.
    #[cfg(test)]
    pub fn fixed(fixed_depth: u8) -> Self {
        Self::new(fixed_depth)
    }

    /// Depth to use for the next cycle. Cheap call — pure read.
    pub fn pick_depth(&self) -> u8 {
        match self.state {
            AdaptiveState::Explore => self.explore_depth,
            AdaptiveState::Full => self.current_depth,
            AdaptiveState::NeighborProbe => self.probe_depth,
            AdaptiveState::Reduced => MIN_DEPTH,
            AdaptiveState::Probe => self.probe_depth,
        }
    }

    /// Diagnostic snapshot for logging.
    pub fn state_label(&self) -> &'static str {
        match self.state {
            AdaptiveState::Explore => "explore",
            AdaptiveState::Full => "full",
            AdaptiveState::NeighborProbe => "neighbor_probe",
            AdaptiveState::Reduced => "reduced",
            AdaptiveState::Probe => "probe",
        }
    }

    /// Diagnostic snapshot of per-depth EMA. Used by tests; the engine's
    /// `run_mtp_cycle` logs the chosen depth + state
    /// label per cycle via `tracing::debug!`.
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn rate_ema_snapshot(&self) -> [f64; MAX_DEPTH as usize] {
        self.rate_ema
    }

    /// Record a cycle's observation and update the state machine + EMA.
    pub fn record_cycle(&mut self, c: CycleStats) {
        self.total_cycles += 1;
        let d = c.depth.clamp(MIN_DEPTH, MAX_DEPTH);
        let idx = (d - 1) as usize;
        // Stamp the freshness clock for this depth: it was just sampled
        // at the current cycle count. Read by `stale_neighbor`'s
        // age-based drift gate. Done for EVERY sampled depth regardless
        // of which EMA branch runs below.
        self.last_sampled_cycle[idx] = self.total_cycles;

        // Per-depth rate EMA: cold-start exact mean for first
        // MIN_COLD_SAMPLES samples, then EMA.
        let rate = c.rate_tps();
        let prev_count = self.sample_count[idx];
        if prev_count < MIN_COLD_SAMPLES {
            // Cumulative mean over the cold-start period.
            let new_count = prev_count + 1;
            let prev = self.rate_ema[idx];
            self.rate_ema[idx] =
                prev * (prev_count as f64 / new_count as f64) + rate / (new_count as f64);
            self.sample_count[idx] = new_count;
        } else {
            let prev = self.rate_ema[idx];
            self.rate_ema[idx] = (1.0 - EMA_ALPHA) * prev + EMA_ALPHA * rate;
        }

        // Per-state window.
        self.window.push(&c);
        self.cycles_in_state = self.cycles_in_state.saturating_add(1);

        // State transitions.
        let prev_state = self.state_label();
        match self.state {
            AdaptiveState::Explore => self.maybe_explore_transition(),
            AdaptiveState::Full => self.maybe_full_transition(),
            AdaptiveState::NeighborProbe => self.maybe_neighbor_probe_transition(),
            AdaptiveState::Reduced => self.maybe_reduced_transition(rate),
            AdaptiveState::Probe => self.maybe_probe_transition(rate),
        }
        let new_state = self.state_label();
        if prev_state != new_state {
            tracing::debug!(
                target: "mlx_core::mtp::adaptive",
                from = prev_state,
                to = new_state,
                total_cycles = self.total_cycles,
                next_depth = self.pick_depth(),
                "MTP adaptive-depth state transition"
            );
        }
    }

    /// `Explore` → `Full`. Advance to the next depth after
    /// `MIN_COLD_SAMPLES` cycles; when every depth has been sampled,
    /// pick the best-rate depth and switch to `Full`.
    fn maybe_explore_transition(&mut self) {
        // Advance after MIN_COLD_SAMPLES cycles at the current
        // explore_depth.
        let need = MIN_COLD_SAMPLES;
        let cur = self.explore_depth;
        if self.sample_count[(cur - 1) as usize] >= need {
            if cur < MAX_DEPTH {
                self.explore_depth = cur + 1;
                // Reset the per-state window so the new depth's
                // observations don't get mixed in.
                self.window.clear();
            } else {
                // All depths sampled — pick the winner and enter Full.
                self.current_depth = self.best_seeded_depth();
                self.enter_full();
            }
        }
    }

    /// Pick the depth with the highest EMA among those with at least
    /// `MIN_COLD_SAMPLES` observations. Falls back to `current_depth`
    /// when nothing is seeded yet (only happens before exploration
    /// finishes).
    fn best_seeded_depth(&self) -> u8 {
        let mut best_depth = self.current_depth;
        let mut best_rate = f64::NEG_INFINITY;
        for d in MIN_DEPTH..=MAX_DEPTH {
            let i = (d - 1) as usize;
            if self.sample_count[i] >= MIN_COLD_SAMPLES && self.rate_ema[i] > best_rate {
                best_rate = self.rate_ema[i];
                best_depth = d;
            }
        }
        best_depth
    }

    /// `Full` → `Reduced` when rolling acceptance drops below
    /// threshold over a full window. Otherwise re-pick `current_depth`
    /// from the per-depth rate EMA (hill-climb on stale data — keeps
    /// adapting to drift) and periodically launch a `NeighborProbe`
    /// burst to refresh neighbor EMAs.
    fn maybe_full_transition(&mut self) {
        // First: hill-climb to the current best-seeded depth.
        let best = self.best_seeded_depth();
        if best != self.current_depth {
            self.current_depth = best;
            // Don't drop the window — the acceptance check below uses
            // the rolling window which briefly mixes old + new depth
            // samples until it rotates.
        }

        // Second: drop to `Reduced` if acceptance is consistently bad.
        if self.window.full() && self.window.mean_acceptance() < DROP_ACCEPT_THRESHOLD {
            self.probe_depth = self.current_depth;
            self.enter_reduced();
            return;
        }

        // Third: periodic NeighborProbe to keep adjacent depths' EMA
        // fresh in case the optimum drifted. Probes are spaced
        // `FULL_REPROBE_INTERVAL` cycles apart; pick the most-stale
        // neighbor within ±1 (under-seeded first, else oldest EMA).
        if self.cycles_in_state >= FULL_REPROBE_INTERVAL {
            if let Some(neighbor) = self.stale_neighbor() {
                self.probe_depth = neighbor;
                self.enter_neighbor_probe();
            } else {
                // Both neighbors fresh enough — reset the counter to
                // avoid hot-looping and try again in another window.
                self.cycles_in_state = 0;
            }
        }
    }

    /// Return the most-stale neighbor of `current_depth` (`±1` clamped
    /// to `[MIN_DEPTH, MAX_DEPTH]`), or `None` if no in-range neighbor
    /// warrants a reprobe right now.
    ///
    /// Freshness is age-based. A neighbor `d` is a reprobe *candidate*
    /// when it is EITHER under-seeded — `sample_count[d-1] <
    /// MIN_COLD_SAMPLES`, i.e. still on the cold-start cumulative-mean
    /// path — OR aged out:
    /// `total_cycles - last_sampled_cycle[d-1] >= FULL_REPROBE_INTERVAL`,
    /// i.e. its EMA has not been refreshed for a full reprobe interval so
    /// the optimum may have drifted out from under it.
    ///
    /// Of the candidates we refresh the most-stale: an under-seeded
    /// neighbor is always preferred over a merely-aged one, and among
    /// equally-seeded neighbors the one with the oldest (smallest)
    /// `last_sampled_cycle` wins. When NO in-range neighbor qualifies —
    /// e.g. a neighbor was just touched by a hill-climb move or a recent
    /// probe — we return `None`, which lets `maybe_full_transition` take
    /// its "both neighbors fresh enough" branch (so that branch is
    /// genuinely reachable). The age subtraction is underflow-safe:
    /// `last_sampled_cycle` is only ever assigned a past `total_cycles`,
    /// so `total_cycles >= last_sampled_cycle` always holds — and we use
    /// `saturating_sub` to make that obvious at the call site.
    ///
    /// Note: `record_cycle` SATURATES `sample_count` at `MIN_COLD_SAMPLES`
    /// (it increments only on the cold-start branch, then switches to the EMA
    /// branch), so after a normal `Explore` sweep every depth sits at exactly
    /// `MIN_COLD_SAMPLES`. A count-only filter (`sample_count <
    /// MIN_COLD_SAMPLES`) would therefore be empty forever and never reprobe.
    /// Gating on freshness (under-seeded OR aged out by `FULL_REPROBE_INTERVAL`
    /// cycles) keeps the periodic drift reprobe alive while still letting this
    /// return `None` when a neighbor was recently sampled.
    fn stale_neighbor(&self) -> Option<u8> {
        let cur = self.current_depth;
        let mut candidates: Vec<u8> = Vec::with_capacity(2);
        if cur > MIN_DEPTH {
            candidates.push(cur - 1);
        }
        if cur < MAX_DEPTH {
            candidates.push(cur + 1);
        }
        candidates
            .into_iter()
            .filter(|&d| {
                let i = (d - 1) as usize;
                // Under-seeded (never left the cold-start path) OR the
                // EMA has aged out (>= a full reprobe interval since the
                // last observation) ⇒ worth a refresh.
                let under_seeded = self.sample_count[i] < MIN_COLD_SAMPLES;
                let age = self.total_cycles.saturating_sub(self.last_sampled_cycle[i]);
                under_seeded || age >= FULL_REPROBE_INTERVAL as u64
            })
            // Most-stale first: prefer the under-seeded neighbor (sorts
            // before any seeded one), then — among equally-seeded
            // neighbors — the one with the oldest (smallest)
            // `last_sampled_cycle`.
            .min_by_key(|&d| {
                let i = (d - 1) as usize;
                let seeded = self.sample_count[i] >= MIN_COLD_SAMPLES;
                (seeded, self.last_sampled_cycle[i])
            })
    }

    /// `NeighborProbe` → `Full`. Just run for the configured number of
    /// cycles to push fresh observations through the EMA, then hand
    /// back to `Full` for the next hill-climb evaluation.
    fn maybe_neighbor_probe_transition(&mut self) {
        if self.cycles_in_state >= NEIGHBOR_PROBE_CYCLES {
            self.enter_full();
        }
    }

    /// `Reduced` → `Probe` after a fixed burst. The `Reduced` burst
    /// gives the EMA-rate-at-MIN_DEPTH time to stabilise so we have a
    /// real baseline to compare the probe against.
    fn maybe_reduced_transition(&mut self, _rate: f64) {
        if self.cycles_in_state >= REDUCED_CYCLES_BEFORE_PROBE {
            self.reduced_baseline_rate = self.window.mean_rate();
            self.enter_probe();
        }
    }

    /// `Probe` → `Full` (commit) or `Probe` → `Reduced` (revert).
    /// Decision: if the probe depth's rate during the burst beats the
    /// `Reduced` baseline rate by any margin, commit back. Otherwise
    /// fall back to `Reduced` and wait another burst before re-probing.
    fn maybe_probe_transition(&mut self, rate: f64) {
        // Accumulate probe rate as a simple EMA over the burst.
        if self.cycles_in_state == 1 {
            self.probe_rate_ema = rate;
        } else {
            self.probe_rate_ema = (1.0 - EMA_ALPHA) * self.probe_rate_ema + EMA_ALPHA * rate;
        }

        if self.cycles_in_state >= PROBE_CYCLES {
            if self.probe_rate_ema > self.reduced_baseline_rate {
                // Probe wins — commit. Stay at `probe_depth` in `Full`.
                self.current_depth = self.probe_depth;
                self.enter_full();
            } else {
                // Probe lost — back to `Reduced` for another burst.
                self.enter_reduced();
            }
        }
    }

    fn enter_full(&mut self) {
        self.state = AdaptiveState::Full;
        self.window.clear();
        self.cycles_in_state = 0;
    }

    fn enter_neighbor_probe(&mut self) {
        self.state = AdaptiveState::NeighborProbe;
        self.window.clear();
        self.cycles_in_state = 0;
    }

    fn enter_reduced(&mut self) {
        self.state = AdaptiveState::Reduced;
        self.window.clear();
        self.cycles_in_state = 0;
    }

    fn enter_probe(&mut self) {
        self.state = AdaptiveState::Probe;
        self.window.clear();
        self.cycles_in_state = 0;
        self.probe_rate_ema = 0.0;
    }
}

#[cfg(test)]
mod tests {
    //! Pure-Rust unit tests for the adaptive depth policy.
    //! No Metal, no MLX, no model load — these run in `cargo test
    //! -p mlx-core --lib adaptive_depth`.
    //!
    //! IMPORTANT: the engine's `run_mtp_cycle` only ever
    //! calls `record_cycle` with the depth `pick_depth()` returned. The
    //! `drive_cycles` helper below enforces that contract; any test
    //! that manually injects observations at a depth `pick_depth()`
    //! did NOT return is testing a non-production code path and will
    //! say so in its docstring.

    use super::*;

    /// Drive the policy for `n` cycles, always asking `pick_depth()`
    /// for the depth and computing `(committed, wall_ns)` from the
    /// caller-supplied closure. Mirrors the production loop in
    /// the engine's `run_mtp_cycle` byte-for-byte.
    fn drive_cycles(
        p: &mut AdaptiveDepthPolicy,
        n: u32,
        mut make_stats: impl FnMut(u8) -> (u32, u64),
    ) {
        for _ in 0..n {
            let depth = p.pick_depth();
            let (committed, wall_ns) = make_stats(depth);
            p.record_cycle(CycleStats {
                depth,
                committed,
                wall_ns,
            });
        }
    }

    /// Total cycles required to finish exploration: 5 depths *
    /// MIN_COLD_SAMPLES cycles each.
    const EXPLORE_TOTAL: u32 = (MAX_DEPTH as u32) * MIN_COLD_SAMPLES;

    #[test]
    fn expected_value_gate_stops_low_value_next_depth() {
        let mut p = ExpectedValueDepthPolicy::new(3);
        p.base_depth = 1;
        p.accept_ewma = [0.70, 0.30, 0.10, 0.05, 0.05];
        p.min_extra_accept_probability = 0.30;
        p.draft_cost_s = 0.0;
        p.extra_verify_cost_s = 0.0;

        let decision = p.should_continue_after_draft(1, 3, DraftMetrics::default());
        assert!(
            !decision.continue_drafting,
            "low D2 expected value should stop after D1"
        );
    }

    #[test]
    fn expected_value_gate_continues_high_value_next_depth() {
        let mut p = ExpectedValueDepthPolicy::new(3);
        p.base_depth = 1;
        p.accept_ewma = [0.95, 0.80, 0.50, 0.10, 0.05];
        p.min_extra_accept_probability = 0.30;
        p.draft_cost_s = 0.0;
        p.extra_verify_cost_s = 0.0;
        p.allow_deepen = true;

        let decision = p.should_continue_after_draft(1, 3, DraftMetrics::default());
        assert!(
            decision.continue_drafting,
            "high D2 expected value should continue past D1"
        );
    }

    #[test]
    fn expected_value_observe_updates_attempted_slots_only() {
        let mut p = ExpectedValueDepthPolicy::new(3);
        p.ewma_alpha = 0.5;
        p.accept_ewma = [0.50, 0.50, 0.50, 0.50, 0.50];

        p.observe(2, 1);
        assert!((p.accept_ewma[0] - 0.75).abs() < 1e-9);
        assert!((p.accept_ewma[1] - 0.25).abs() < 1e-9);
        assert!((p.accept_ewma[2] - 0.50).abs() < 1e-9);
    }

    /// Cold-start: first MIN_COLD_SAMPLES samples per depth use a
    /// cumulative mean, not EMA. Verify the rate stored is the exact
    /// mean. Drives the policy directly (bypassing `pick_depth`) since
    /// we want to exercise the cumulative-mean math at a single depth.
    #[test]
    fn cold_start_uses_cumulative_mean() {
        let mut p = AdaptiveDepthPolicy::new(3);
        p.record_cycle(CycleStats {
            depth: 3,
            committed: 4,
            wall_ns: 40_000_000, // 100 tps
        });
        p.record_cycle(CycleStats {
            depth: 3,
            committed: 4,
            wall_ns: 20_000_000, // 200 tps
        });
        let ema_idx = 2; // depth 3
        assert!(
            (p.rate_ema[ema_idx] - 150.0).abs() < 1e-6,
            "expected cumulative mean 150.0, got {}",
            p.rate_ema[ema_idx]
        );
        assert_eq!(p.sample_count[ema_idx], MIN_COLD_SAMPLES);
    }

    /// Bootstrap: the policy must spend `EXPLORE_TOTAL` cycles in
    /// `Explore` and visit every depth in `{1..=5}` before entering
    /// `Full`. The policy is driven entirely via `pick_depth()`, never
    /// manually fed unpicked depths.
    #[test]
    fn explore_visits_every_depth_then_enters_full() {
        let mut p = AdaptiveDepthPolicy::new(3);
        assert_eq!(p.state, AdaptiveState::Explore);

        // Drive exactly the explore-burst length with a non-trivial
        // rate function (rate increases with depth ⇒ depth 5 wins).
        drive_cycles(&mut p, EXPLORE_TOTAL, |d| (d as u32 + 1, 30_000_000));

        // After exploration: every depth has MIN_COLD_SAMPLES samples
        // and the policy entered Full at the best-rate depth.
        for d in MIN_DEPTH..=MAX_DEPTH {
            assert_eq!(
                p.sample_count[(d - 1) as usize],
                MIN_COLD_SAMPLES,
                "depth {d} should have {MIN_COLD_SAMPLES} samples after exploration",
            );
        }
        assert_eq!(p.state, AdaptiveState::Full);
        // Higher depth ⇒ higher rate (committed = depth+1, wall same)
        // ⇒ depth 5 wins.
        assert_eq!(p.current_depth, MAX_DEPTH);
    }

    /// State transition: after exploration, `Full` → `Reduced` when
    /// rolling acceptance over a full window falls below 0.75.
    /// **Production flow only.**
    #[test]
    fn full_to_reduced_on_low_acceptance() {
        let mut p = AdaptiveDepthPolicy::new(3);
        // Exploration with low acceptance (committed = 2 = 1 accepted
        // draft + 1 residual) regardless of depth. Rate ~constant.
        drive_cycles(&mut p, EXPLORE_TOTAL, |_d| (2, 30_000_000));
        assert_eq!(p.state, AdaptiveState::Full);
        // Now Full with low acceptance — window should fill and drop.
        drive_cycles(&mut p, STATE_WINDOW as u32, |_d| (2, 30_000_000));
        assert_eq!(p.state, AdaptiveState::Reduced);
        assert_eq!(p.pick_depth(), MIN_DEPTH);
        // probe_depth = the depth we dropped from. Since exploration
        // saw uniform rates, hill-climb keeps current_depth at the
        // first-tied winner — depth 1 in this case (lowest depth
        // with rate equal to the others). Either way it's in [1, 5].
        assert!((MIN_DEPTH..=MAX_DEPTH).contains(&p.probe_depth));
    }

    /// `Full` stays in `Full` when acceptance is good.
    #[test]
    fn full_stays_full_on_good_acceptance() {
        let mut p = AdaptiveDepthPolicy::new(3);
        drive_cycles(&mut p, EXPLORE_TOTAL, |d| (d as u32 + 1, 30_000_000));
        let pre_state = p.state;
        // A few more cycles at full-accept — should stay in Full
        // until the FULL_REPROBE_INTERVAL fires (which only switches
        // to NeighborProbe momentarily).
        drive_cycles(&mut p, STATE_WINDOW as u32, |d| (d as u32 + 1, 30_000_000));
        assert_eq!(pre_state, AdaptiveState::Full);
        assert!(matches!(
            p.state,
            AdaptiveState::Full | AdaptiveState::NeighborProbe
        ));
    }

    /// State transition: `Reduced` → `Probe` after the burst, then
    /// `Probe` → `Reduced` if the probe's rate is worse than the
    /// reduced baseline. **Production flow only.**
    #[test]
    fn reduced_to_probe_and_back_when_probe_loses() {
        let mut p = AdaptiveDepthPolicy::new(3);
        // Bad-acceptance exploration ⇒ Full at some depth.
        drive_cycles(&mut p, EXPLORE_TOTAL, |_d| (2, 30_000_000));
        // Drive Full → Reduced.
        drive_cycles(&mut p, STATE_WINDOW as u32, |_d| (2, 30_000_000));
        assert_eq!(p.state, AdaptiveState::Reduced);

        // Reduced burst: pick_depth() returns MIN_DEPTH=1; deliver a
        // FAST rate (200 tps) to make Reduced look good.
        drive_cycles(&mut p, REDUCED_CYCLES_BEFORE_PROBE, |_d| {
            (2, 10_000_000) // 200 tps
        });
        assert_eq!(p.state, AdaptiveState::Probe);

        // Probe burst: pick_depth() returns probe_depth. Deliver a
        // SLOW rate (25 tps) so the probe loses vs. the 200 tps
        // baseline ⇒ revert to Reduced.
        drive_cycles(&mut p, PROBE_CYCLES, |_d| (2, 80_000_000));
        assert_eq!(
            p.state,
            AdaptiveState::Reduced,
            "probe lost → back to reduced"
        );
    }

    /// State transition: `Probe` → `Full` when the probe rate exceeds
    /// the Reduced baseline. **Production flow only.**
    #[test]
    fn probe_to_full_when_probe_wins() {
        let mut p = AdaptiveDepthPolicy::new(3);
        drive_cycles(&mut p, EXPLORE_TOTAL, |_d| (2, 30_000_000));
        drive_cycles(&mut p, STATE_WINDOW as u32, |_d| (2, 30_000_000));
        assert_eq!(p.state, AdaptiveState::Reduced);

        // Reduced is SLOW (50 tps baseline).
        drive_cycles(&mut p, REDUCED_CYCLES_BEFORE_PROBE, |_d| {
            (2, 40_000_000) // 50 tps
        });
        assert_eq!(p.state, AdaptiveState::Probe);

        // Probe is FAST (200 tps).
        drive_cycles(&mut p, PROBE_CYCLES, |_d| (4, 20_000_000));
        assert_eq!(p.state, AdaptiveState::Full);
    }

    /// The policy must be able to discover a non-seed depth as the
    /// optimum, driven entirely through the production
    /// `pick_depth()`→`record_cycle()` loop. Bound the acceptance to
    /// "good" so we stay out of Reduced and let Explore + NeighborProbe
    /// do their job.
    ///
    /// Setup: rate is monotonically increasing in depth. Seed = 1
    /// (intentionally the worst). The policy must discover depth 5
    /// is the optimum.
    #[test]
    fn discovers_non_seed_best_depth_via_production_flow() {
        let mut p = AdaptiveDepthPolicy::new(1);
        // committed = depth + 1 (full accept), wall_ns constant ⇒
        // rate proportional to (depth + 1). Higher depth wins.
        drive_cycles(&mut p, EXPLORE_TOTAL + 4, |d| (d as u32 + 1, 30_000_000));
        assert_eq!(p.state, AdaptiveState::Full);
        assert_eq!(
            p.current_depth, MAX_DEPTH,
            "production flow with monotone-in-depth rate must discover MAX_DEPTH"
        );
    }

    /// Periodic `NeighborProbe` fires after `FULL_REPROBE_INTERVAL`
    /// cycles in `Full` *when a neighbor is under-seeded*, and pushes
    /// fresh observations through that neighbor's EMA.
    ///
    /// Under the age-based `stale_neighbor` rule, under-seeded is still a
    /// trigger AND ranks ahead of a merely-aged neighbor. We hand-seed
    /// the policy into `Full` at depth 3 with the upper neighbor (depth
    /// 4) left under-seeded. The lower neighbor (depth 2) is seeded; by
    /// the time the interval fires it has also aged out (it is never
    /// re-sampled in `Full`), but the under-seeded-first tie-break makes
    /// depth 4 the unambiguous probe target regardless.
    #[test]
    fn full_launches_neighbor_probe_at_interval() {
        let mut p = AdaptiveDepthPolicy::new(3);
        // Seed depths 1,2,3 (and 5) to the seeding bar but leave the
        // upper neighbor (depth 4) under-seeded so a reprobe is owed.
        for d in [MIN_DEPTH, 2, 3, MAX_DEPTH] {
            p.sample_count[(d - 1) as usize] = MIN_COLD_SAMPLES;
            p.rate_ema[(d - 1) as usize] = 100.0;
        }
        // Depth 4 stays at 0 samples (the stale neighbor). Make sure
        // the hill-climb keeps `current_depth` at 3 by giving it the
        // strictly-best rate among seeded depths.
        p.rate_ema[(3 - 1) as usize] = 200.0;
        p.current_depth = 3;
        p.enter_full();
        let cur_before = p.current_depth;
        assert_eq!(cur_before, 3);

        // Drive `FULL_REPROBE_INTERVAL` cycles in Full. We should
        // enter NeighborProbe at least once.
        let mut entered_probe = false;
        let mut probe_target = None;
        for _ in 0..(FULL_REPROBE_INTERVAL + NEIGHBOR_PROBE_CYCLES + 4) {
            let d = p.pick_depth();
            // Full accept (committed = depth + 1) keeps acceptance at
            // 1.0 so the policy never drops to Reduced before the
            // reprobe interval fires.
            p.record_cycle(CycleStats {
                depth: d,
                committed: (d as u32) + 1,
                wall_ns: 30_000_000,
            });
            if p.state == AdaptiveState::NeighborProbe {
                entered_probe = true;
                probe_target = Some(p.pick_depth());
            }
        }
        assert!(
            entered_probe,
            "NeighborProbe must fire within FULL_REPROBE_INTERVAL + buffer cycles when a neighbor is under-seeded"
        );
        // The under-seeded neighbor (depth 4) must be the probe target.
        assert_eq!(
            probe_target,
            Some(4),
            "the stale (under-seeded) neighbor must be the reprobe target"
        );
    }

    /// Else-branch reachable: `stale_neighbor` returns `None` (and so
    /// `maybe_full_transition` takes its "both neighbors fresh enough"
    /// branch) when both in-range neighbors are well-seeded AND were
    /// sampled recently (within the last `FULL_REPROBE_INTERVAL`
    /// cycles). Proves a recently-sampled, well-seeded neighbor is NOT a
    /// reprobe candidate, so the else-branch is live.
    ///
    /// Hand-seeds the per-depth state and calls `stale_neighbor`
    /// directly (the "both fresh" configuration is unreachable via the
    /// production Explore sweep, which leaves the current depth's
    /// neighbors un-sampled in `Full`).
    #[test]
    fn stale_neighbor_none_when_neighbors_seeded_and_fresh() {
        let mut p = AdaptiveDepthPolicy::new(3);
        p.current_depth = 3;
        // Both neighbors (depths 2 and 4) seeded to the bar...
        for d in MIN_DEPTH..=MAX_DEPTH {
            p.sample_count[(d - 1) as usize] = MIN_COLD_SAMPLES;
        }
        // ...and freshly sampled: their last-sampled cycle equals the
        // current cycle count, so age == 0 < FULL_REPROBE_INTERVAL.
        p.total_cycles = 100;
        p.last_sampled_cycle[(2 - 1) as usize] = p.total_cycles;
        p.last_sampled_cycle[(4 - 1) as usize] = p.total_cycles;

        assert_eq!(
            p.stale_neighbor(),
            None,
            "stale_neighbor must return None when both neighbors are well-seeded AND freshly sampled"
        );
    }

    /// Drift reprobe restored. With every depth seeded to
    /// `MIN_COLD_SAMPLES`, the policy enters `Full` at depth 3 and runs
    /// constant-rate, full-accept cycles at depth 3 only. The neighbors
    /// (depths 2 and 4) therefore go un-sampled, and once they age out by
    /// `>= FULL_REPROBE_INTERVAL` cycles the periodic drift `NeighborProbe`
    /// MUST fire — this fails under a count-only filter (which never
    /// re-fires once every depth saturates at `MIN_COLD_SAMPLES`).
    /// Acceptance is kept maximal (committed = depth + 1) so the policy
    /// never drops to `Reduced` before the reprobe fires.
    #[test]
    fn full_relaunches_neighbor_probe_on_drift() {
        let mut p = AdaptiveDepthPolicy::new(3);
        // Seed every depth to the bar; make depth 3 the strict winner so
        // the hill-climb pins current_depth at 3 (its neighbors are the
        // ones that drift).
        for d in MIN_DEPTH..=MAX_DEPTH {
            p.sample_count[(d - 1) as usize] = MIN_COLD_SAMPLES;
            p.rate_ema[(d - 1) as usize] = 100.0;
        }
        p.rate_ema[(3 - 1) as usize] = 200.0;
        p.current_depth = 3;
        p.enter_full();

        // Run depth-3-only cycles past the reprobe interval. Neighbors 2
        // and 4 are never re-sampled ⇒ they age out ⇒ drift reprobe.
        let mut entered_probe = false;
        for _ in 0..(FULL_REPROBE_INTERVAL + NEIGHBOR_PROBE_CYCLES + 4) {
            let d = p.pick_depth();
            p.record_cycle(CycleStats {
                depth: d,
                committed: (d as u32) + 1,
                wall_ns: 30_000_000,
            });
            if p.state == AdaptiveState::NeighborProbe {
                entered_probe = true;
            }
        }
        assert!(
            entered_probe,
            "drift reprobe must fire: aged-out neighbors trigger a NeighborProbe even when every depth is seeded to MIN_COLD_SAMPLES"
        );
    }

    /// Bounds: `pick_depth` always returns a value in
    /// `[MIN_DEPTH, MAX_DEPTH]` across every state.
    #[test]
    fn pick_depth_in_bounds() {
        let mut p = AdaptiveDepthPolicy::new(99);
        assert!((MIN_DEPTH..=MAX_DEPTH).contains(&p.pick_depth()));
        // Drive many cycles through real production flow.
        drive_cycles(&mut p, EXPLORE_TOTAL + 50, |d| {
            // Mix of good and bad cycles so we exercise all branches.
            if (d as u32).is_multiple_of(2) {
                (d as u32 + 1, 30_000_000) // full accept
            } else {
                (2, 30_000_000) // mostly reject
            }
        });
        assert!((MIN_DEPTH..=MAX_DEPTH).contains(&p.pick_depth()));
    }

    /// Fixed-depth helper bounds-clamps the input; the seed is the
    /// initial `current_depth`, but exploration will still run on the
    /// first record (production callers pin `mtp_adaptive_depth=false`
    /// to skip exploration entirely).
    #[test]
    fn fixed_constructor_clamps() {
        let p = AdaptiveDepthPolicy::fixed(0);
        // In Explore state, `pick_depth()` returns `explore_depth`
        // which starts at MIN_DEPTH.
        assert_eq!(p.pick_depth(), MIN_DEPTH);
        let p = AdaptiveDepthPolicy::fixed(7);
        assert_eq!(p.pick_depth(), MIN_DEPTH);
        assert_eq!(p.current_depth, MAX_DEPTH);
    }
}
