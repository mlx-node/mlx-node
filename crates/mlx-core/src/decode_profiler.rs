//! Decode loop performance profiler.
//!
//! Provides structured timing for token generation decode loops.
//! Activated via `MLX_PROFILE_DECODE=1` environment variable, or
//! programmatically via the `profiling` module's `set_profiling_enabled()`.
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::decode_profiler::DecodeProfiler;
//!
//! let mut profiler = DecodeProfiler::new("chat_compiled", "qwen3_5");
//! profiler.set_prompt_tokens(42);
//! profiler.snapshot_memory_before();
//! profiler.begin_prefill();
//! // ... prefill ...
//! profiler.end_prefill();
//!
//! for step in 0..max_tokens {
//!     profiler.begin("eval_token");
//!     y.eval();
//!     profiler.end();
//!
//!     profiler.begin("extract");
//!     let token_id = y.item_at_int32(0)?;
//!     profiler.end();
//!
//!     profiler.mark_first_token();
//!
//!     profiler.begin("forward");
//!     let logits = forward_inner(...);
//!     profiler.end();
//!
//!     profiler.begin("sample");
//!     let next = sample(&logits, ...)?;
//!     profiler.end();
//!
//!     profiler.begin("async_eval");
//!     MxArray::async_eval_arrays(&[&next]);
//!     profiler.end();
//!
//!     profiler.step();
//! }
//!
//! profiler.snapshot_memory_after();
//! profiler.report();
//! ```

use std::collections::HashMap;
use std::time::Instant;

use crate::profiling;
use crate::profiling::{GenerationProfile, MemorySnapshot, PhaseProfile};

/// Controls whether decode profiling is active via env var.
/// Cached on first access for fast repeated checks.
fn is_env_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MLX_PROFILE_DECODE").is_ok())
}

/// Lightweight decode loop profiler.
///
/// When enabled (env var or programmatic API), records per-phase wall-clock
/// times, prefill timing, TTFT, and memory snapshots. When disabled, all
/// methods are no-ops with ~1ns overhead (branch predictor eliminates).
pub struct DecodeProfiler {
    enabled: bool,
    env_enabled: bool,
    label: &'static str,
    model_type: &'static str,
    phases: HashMap<&'static str, PhaseStats>,
    phase_order: Vec<&'static str>,
    /// Stack of in-flight phases. Each frame stores its own `(name, start)`,
    /// so nested begin/end pairs compose correctly (e.g. outer `mtp_cycle`
    /// can wrap inner `draft`, `verify`, etc. without losing the cycle total).
    phase_stack: Vec<(&'static str, Instant)>,
    loop_start: Instant,
    num_tokens: u64,
    prompt_tokens: u32,
    prefill_start: Option<Instant>,
    prefill_ms: f64,
    first_token_time: Option<Instant>,
    first_token_marked: bool,
    memory_before: Option<MemorySnapshot>,
    memory_after: Option<MemorySnapshot>,
    /// MTP speculative-decode acceptance counters. Updated by
    /// `record_mtp_cycle` once per draft+verify cycle. `mtp_cycles == 0`
    /// means no MTP cycle ran (a plain autoregressive decode).
    mtp_cycles: u64,
    /// Sum of accepted draft tokens (K) across all cycles.
    mtp_accepted_drafts_total: u64,
    /// Sum of attempted draft depth (D) across all cycles.
    mtp_depth_total: u64,
    /// Per draft-position: how many cycles attempted that position.
    mtp_attempt_by_position: Vec<u64>,
    /// Per draft-position: how many cycles accepted that position.
    mtp_accept_by_position: Vec<u64>,
}

struct PhaseStats {
    total_us: u64,
    count: u64,
}

impl DecodeProfiler {
    /// Create a new profiler. `label` identifies the decode loop variant
    /// (e.g. "chat_compiled", "chat_rust", "generate_compiled").
    /// `model_type` identifies the model (e.g. "qwen3_5", "qwen3_5_moe", "qwen3").
    pub fn new(label: &'static str, model_type: &'static str) -> Self {
        let env_enabled = is_env_enabled();
        let enabled = env_enabled || profiling::is_active();
        if enabled {
            tracing::info!(target: "mlx_core::decode", label, "decode profiling enabled");
        }
        Self {
            enabled,
            env_enabled,
            label,
            model_type,
            phases: HashMap::new(),
            phase_order: Vec::new(),
            phase_stack: Vec::new(),
            loop_start: Instant::now(),
            num_tokens: 0,
            prompt_tokens: 0,
            prefill_start: None,
            prefill_ms: 0.0,
            first_token_time: None,
            first_token_marked: false,
            memory_before: None,
            memory_after: None,
            mtp_cycles: 0,
            mtp_accepted_drafts_total: 0,
            mtp_depth_total: 0,
            mtp_attempt_by_position: Vec::new(),
            mtp_accept_by_position: Vec::new(),
        }
    }

    /// Update the label (e.g. after branching compiled vs rust).
    #[inline]
    pub fn set_label(&mut self, label: &'static str) {
        self.label = label;
    }

    /// Record the number of prompt tokens.
    #[inline]
    pub fn set_prompt_tokens(&mut self, n: u32) {
        if !self.enabled {
            return;
        }
        self.prompt_tokens = n;
    }

    /// Start timing the prefill phase.
    #[inline]
    pub fn begin_prefill(&mut self) {
        if !self.enabled {
            return;
        }
        self.prefill_start = Some(Instant::now());
    }

    /// End timing the prefill phase.
    #[inline]
    pub fn end_prefill(&mut self) {
        if !self.enabled {
            return;
        }
        if let Some(start) = self.prefill_start.take() {
            self.prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
        }
        // Reset loop_start to measure decode time from here
        self.loop_start = Instant::now();
    }

    /// Take a memory snapshot before generation.
    #[inline]
    pub fn snapshot_memory_before(&mut self) {
        if !self.enabled {
            return;
        }
        self.memory_before = Some(profiling::snapshot_memory());
    }

    /// Take a memory snapshot after generation.
    #[inline]
    pub fn snapshot_memory_after(&mut self) {
        if !self.enabled {
            return;
        }
        self.memory_after = Some(profiling::snapshot_memory());
    }

    /// Mark that the first token has been extracted (for TTFT).
    /// Only records the first call; subsequent calls are no-ops.
    #[inline]
    pub fn mark_first_token(&mut self) {
        if !self.enabled || self.first_token_marked {
            return;
        }
        self.first_token_marked = true;
        self.first_token_time = Some(Instant::now());
    }

    /// Start timing a named phase. Pairs with `end()`.
    ///
    /// Nestable: each `begin` pushes a new frame onto an internal stack so an
    /// outer phase (e.g. `mtp_cycle`) can wrap inner sub-phases. Each frame
    /// carries its own start `Instant`; `end()` pops the top frame.
    ///
    /// First-seen ordering is established here on `begin()` so that an outer
    /// phase (begun first, ended last) still appears before inner phases in
    /// `phase_order`.
    #[inline]
    pub fn begin(&mut self, phase: &'static str) {
        if !self.enabled {
            return;
        }
        self.register_phase(phase);
        self.phase_stack.push((phase, Instant::now()));
    }

    /// End the most-recently-begun phase and accumulate its time. No-op if
    /// the stack is empty (matches prior behavior on stray `end()`).
    #[inline]
    pub fn end(&mut self) {
        if !self.enabled {
            return;
        }
        if let Some((phase, start)) = self.phase_stack.pop() {
            let elapsed_us = start.elapsed().as_micros() as u64;
            self.accumulate(phase, elapsed_us);
        }
    }

    /// Record a precomputed duration under `name` without touching the
    /// begin/end stack. Useful for paths that already have an
    /// `Instant::elapsed()` (e.g. FFI calls that return their own timings).
    /// Same enabled/disabled gating and accumulation semantics as `end()`.
    #[inline]
    pub fn record_duration(&mut self, name: &'static str, dur: std::time::Duration) {
        if !self.enabled {
            return;
        }
        self.register_phase(name);
        let elapsed_us = dur.as_micros() as u64;
        self.accumulate(name, elapsed_us);
    }

    /// Register `name` in `phase_order` on first sight, and ensure a
    /// `PhaseStats` entry exists. Idempotent on repeated calls.
    #[inline]
    fn register_phase(&mut self, name: &'static str) {
        let phase_order = &mut self.phase_order;
        self.phases.entry(name).or_insert_with(|| {
            phase_order.push(name);
            PhaseStats {
                total_us: 0,
                count: 0,
            }
        });
    }

    /// Accumulate `elapsed_us` into `phases[name]`. Callers MUST have run
    /// `register_phase(name)` first; an unregistered name is silently dropped.
    #[inline]
    fn accumulate(&mut self, name: &'static str, elapsed_us: u64) {
        if let Some(stats) = self.phases.get_mut(name) {
            stats.total_us += elapsed_us;
            stats.count += 1;
        }
    }

    /// Mark one token as completed. Call once per decode step.
    #[inline]
    pub fn step(&mut self) {
        self.num_tokens += 1;
    }

    /// Record one MTP speculative draft+verify cycle: `depth` draft
    /// tokens were attempted and `accepted_drafts` (K) of them were
    /// accepted by verify. Acceptance is prefix-monotone — positions
    /// `0..K` accepted, positions `K..depth` attempted-but-rejected.
    ///
    /// Unlike the timing methods, this is **not** gated on `enabled`:
    /// it is pure CPU integer arithmetic (negligible cost) and the
    /// resulting acceptance summary must reach `PerformanceMetrics`
    /// whenever `reportPerformance` is set, independent of whether
    /// `MLX_PROFILE_DECODE` enabled the timing profiler.
    #[inline]
    pub fn record_mtp_cycle(&mut self, depth: usize, accepted_drafts: usize) {
        let k = accepted_drafts.min(depth);
        self.mtp_cycles += 1;
        self.mtp_accepted_drafts_total += k as u64;
        self.mtp_depth_total += depth as u64;
        if self.mtp_attempt_by_position.len() < depth {
            self.mtp_attempt_by_position.resize(depth, 0);
        }
        if self.mtp_accept_by_position.len() < depth {
            self.mtp_accept_by_position.resize(depth, 0);
        }
        for slot in self.mtp_attempt_by_position.iter_mut().take(depth) {
            *slot += 1;
        }
        for slot in self.mtp_accept_by_position.iter_mut().take(k) {
            *slot += 1;
        }
    }

    /// Test-only: force the profiler on so `begin`/`end` record phases.
    /// `begin` no-ops while `!enabled`, so a test that wants to observe
    /// which decode branch ran (via [`ran_phase`]) must call this first.
    /// Enabling only turns on timing instrumentation — it never changes
    /// decode control flow or which tokens are committed.
    ///
    /// [`ran_phase`]: Self::ran_phase
    #[cfg(test)]
    pub(crate) fn enable_for_test(&mut self) {
        self.enabled = true;
    }

    /// Test-only: did a phase with this exact name run (i.e. was
    /// `begin(name)` reached while the profiler was enabled)? Used by
    /// accept-path coverage tests to assert which decode branch executed
    /// — e.g. `"mtp_accept_argmax"` is unique to the sparse-accept path.
    /// Requires [`enable_for_test`] to have been called on this profiler.
    ///
    /// [`enable_for_test`]: Self::enable_for_test
    #[cfg(test)]
    pub(crate) fn ran_phase(&self, name: &str) -> bool {
        self.phases.contains_key(name)
    }

    /// MTP acceptance summary: `(mean_accepted_per_cycle,
    /// per_position_rate, cycles)`. `None` when no MTP cycle ran.
    pub fn mtp_acceptance_summary(&self) -> Option<(f64, Vec<f64>, u32)> {
        if self.mtp_cycles == 0 {
            return None;
        }
        let mean = self.mtp_accepted_drafts_total as f64 / self.mtp_cycles as f64;
        let per_pos: Vec<f64> = self
            .mtp_accept_by_position
            .iter()
            .zip(self.mtp_attempt_by_position.iter())
            .map(|(&a, &t)| if t > 0 { a as f64 / t as f64 } else { 0.0 })
            .collect();
        Some((
            mean,
            per_pos,
            self.mtp_cycles.min(u64::from(u32::MAX)) as u32,
        ))
    }

    /// Mean *committed* tokens per MTP cycle INCLUDING the always-verified
    /// token: `mtp_accepted_drafts_total / mtp_cycles + 1.0`. `None` when
    /// no MTP cycle ran. This is the mlx-vlm-comparable headline accept
    /// rate — it equals mlx-vlm's `mean_accepted_tokens =
    /// (accepted_drafts + rounds) / rounds`
    /// (`mlx-vlm/mlx_vlm/speculative/common.py:247`), with `mtp_cycles`
    /// the 1:1 analog of mlx-vlm's `rounds`. The drafts-only value
    /// (`mtp_acceptance_summary().0`) stays available as the historical
    /// secondary metric.
    pub fn mtp_mean_accepted_tokens_total(&self) -> Option<f64> {
        if self.mtp_cycles == 0 {
            return None;
        }
        Some(self.mtp_accepted_drafts_total as f64 / self.mtp_cycles as f64 + 1.0)
    }

    /// Mean attempted draft depth per MTP cycle. `None` when no MTP
    /// cycle ran.
    pub fn mtp_mean_depth(&self) -> Option<f64> {
        if self.mtp_cycles == 0 {
            return None;
        }
        Some(self.mtp_depth_total as f64 / self.mtp_cycles as f64)
    }

    /// Build the public phase profile vector for `PerformanceMetrics`
    /// and `GenerationProfile`.
    fn phase_profiles(&self) -> Vec<PhaseProfile> {
        let n = (self.num_tokens as f64).max(1.0);
        self.phase_order
            .iter()
            .filter_map(|&name| {
                self.phases.get(name).map(|stats| PhaseProfile {
                    name: name.to_string(),
                    total_ms: stats.total_us as f64 / 1000.0,
                    avg_us_per_token: stats.total_us as f64 / n,
                    count: stats.count as u32,
                })
            })
            .collect()
    }

    /// Copy the MTP acceptance summary into a `PerformanceMetrics`.
    /// No-op when no MTP cycle ran (leaves the fields `None`).
    pub fn fill_mtp_acceptance(&self, m: &mut crate::profiling::PerformanceMetrics) {
        if let Some((mean, per_pos, cycles)) = self.mtp_acceptance_summary() {
            m.mtp_mean_accepted_tokens = Some(mean);
            m.mtp_mean_accepted_tokens_total = self.mtp_mean_accepted_tokens_total();
            m.mtp_acceptance_by_position = Some(per_pos);
            m.mtp_cycles = Some(cycles);
            m.mtp_mean_depth = self.mtp_mean_depth();
        }
        if self.enabled && self.num_tokens > 0 {
            let phases = self.phase_profiles();
            if !phases.is_empty() {
                m.profile_phases = Some(phases);
            }
        }
    }

    /// Print a summary and/or push to the global profiling store.
    ///
    /// - If env var is set → print to stderr (backward compat)
    /// - If programmatic profiling is active → push `GenerationProfile` to store
    pub fn report(&self) {
        if !self.enabled || self.num_tokens == 0 {
            return;
        }

        let n = self.num_tokens as f64;
        let decode_ms = self.loop_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = self.prefill_ms + decode_ms;
        let tok_s = n / (decode_ms / 1000.0);

        let ttft_ms = self
            .first_token_time
            .map(|t| {
                // TTFT = prefill time + time from decode loop start to first token
                let from_loop_start =
                    t.saturating_duration_since(self.loop_start).as_secs_f64() * 1000.0;
                self.prefill_ms + from_loop_start
            })
            .unwrap_or(0.0);

        // Stderr output (backward compat when env var set)
        if self.env_enabled {
            self.print_stderr_report(n, decode_ms, tok_s);
        }

        // Push to global store (when programmatic profiling is active)
        if profiling::is_active() {
            let phases = self.phase_profiles();
            let (mtp_mean_accepted_tokens, mtp_acceptance_by_position, mtp_cycles) = self
                .mtp_acceptance_summary()
                .map(|(mean, per_pos, cycles)| (Some(mean), Some(per_pos), Some(cycles)))
                .unwrap_or((None, None, None));

            profiling::push_generation(GenerationProfile {
                label: self.label.to_string(),
                model_type: self.model_type.to_string(),
                num_tokens: self.num_tokens as u32,
                prompt_tokens: self.prompt_tokens,
                prefill_ms: self.prefill_ms,
                decode_ms,
                total_ms,
                tokens_per_second: tok_s,
                time_to_first_token_ms: ttft_ms,
                phases,
                mtp_mean_accepted_tokens,
                mtp_mean_accepted_tokens_total: self.mtp_mean_accepted_tokens_total(),
                mtp_acceptance_by_position,
                mtp_cycles,
                mtp_mean_depth: self.mtp_mean_depth(),
                memory_before: self.memory_before.clone(),
                memory_after: self.memory_after.clone(),
            });
        }
    }

    /// Check if the profiler is enabled (for testing).
    #[cfg(test)]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn print_stderr_report(&self, n: f64, wall_ms: f64, wall_tok_s: f64) {
        let mut lines = Vec::new();

        if self.prefill_ms > 0.0 {
            lines.push(format!(
                "\n[PROFILE] {} ({} prompt tokens, prefill {:.1}ms):",
                self.label, self.prompt_tokens, self.prefill_ms
            ));
        }

        lines.push(format!(
            "\n[PROFILE] {} decode loop ({} tokens, {:.0}ms wall, {:.1} tok/s):",
            self.label, self.num_tokens, wall_ms, wall_tok_s
        ));

        let mut cpu_total_us: u64 = 0;
        for phase in &self.phase_order {
            if let Some(stats) = self.phases.get(phase) {
                cpu_total_us += stats.total_us;
                let ms = stats.total_us as f64 / 1000.0;
                let us_per_tok = stats.total_us as f64 / n;
                lines.push(format!(
                    "  {:<20} {:>8.1}ms ({:>7.1}us/tok, {} calls)",
                    phase, ms, us_per_tok, stats.count
                ));
            }
        }

        let cpu_ms = cpu_total_us as f64 / 1000.0;
        let cpu_tok_s = n / (cpu_total_us as f64 / 1_000_000.0);
        lines.push(format!(
            "  {:<20} {:>8.1}ms ({:>7.1}us/tok = {:.1} tok/s)",
            "TOTAL (measured)",
            cpu_ms,
            cpu_total_us as f64 / n,
            cpu_tok_s
        ));

        if let Some((mean, per_pos, cycles)) = self.mtp_acceptance_summary() {
            let mean_depth = self.mtp_depth_total as f64 / self.mtp_cycles as f64;
            let mean_total = self.mtp_mean_accepted_tokens_total().unwrap_or(mean + 1.0);
            let per_pos_str: Vec<String> = per_pos.iter().map(|p| format!("{:.3}", p)).collect();
            // Headline: mlx-vlm-comparable mean accepted tokens/cycle
            // (incl. the always-verified token) — matches mlx-vlm's
            // `(accepted_drafts + rounds)/rounds` (common.py:247).
            lines.push(format!(
                "  [PROFILE] MTP accept: cycles={} mean accepted tokens/cycle={:.2} \
                 (incl. verified, mlx-vlm-comparable) mean_drafts/cycle={:.2} \
                 mean_depth={:.2} per_position=[{}]",
                cycles,
                mean_total,
                mean,
                mean_depth,
                per_pos_str.join(", "),
            ));
        }

        let report = lines.join("\n");
        eprintln!("{}", report);
        tracing::info!(target: "mlx_core::decode", "{}", report);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::thread;
    use std::time::Duration;

    /// Helper: enable programmatic profiling, run closure, disable.
    fn with_profiling<F: FnOnce()>(f: F) {
        profiling::PROFILING_ENABLED.store(true, Ordering::Relaxed);
        profiling::PROFILING_STORE.lock().unwrap().clear();
        f();
        profiling::PROFILING_ENABLED.store(false, Ordering::Relaxed);
    }

    #[test]
    fn test_disabled_by_default() {
        // When neither env var nor programmatic API is enabled,
        // profiler should be disabled (unless MLX_PROFILE_DECODE is set in CI).
        let profiler = DecodeProfiler::new("test", "test_model");
        // We can't assert `!profiler.is_enabled()` because the env var
        // might be set in CI. Just verify it doesn't crash.
        drop(profiler);
    }

    #[test]
    fn test_enabled_via_programmatic_api() {
        with_profiling(|| {
            let profiler = DecodeProfiler::new("test_api", "qwen3");
            assert!(profiler.is_enabled());
        });
    }

    #[test]
    fn test_set_label() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("initial", "qwen3_5");
            profiler.set_label("changed");

            // Simulate a decode step so report() actually pushes
            profiler.step();
            profiler.report();

            let store = profiling::PROFILING_STORE.lock().unwrap();
            let last = store.last().unwrap();
            assert_eq!(last.label, "changed");
        });
    }

    #[test]
    fn test_set_prompt_tokens() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("test_prompt", "qwen3");
            profiler.set_prompt_tokens(42);
            profiler.step();
            profiler.report();

            let store = profiling::PROFILING_STORE.lock().unwrap();
            let last = store.last().unwrap();
            assert_eq!(last.prompt_tokens, 42);
        });
    }

    #[test]
    fn test_prefill_timing() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("test_prefill", "qwen3_5");
            profiler.begin_prefill();
            thread::sleep(Duration::from_millis(10));
            profiler.end_prefill();

            profiler.step();
            profiler.report();

            let store = profiling::PROFILING_STORE.lock().unwrap();
            let last = store.last().unwrap();
            // Prefill should have recorded at least ~10ms
            assert!(
                last.prefill_ms >= 5.0,
                "prefill_ms {} should be >= 5ms",
                last.prefill_ms
            );
        });
    }

    #[test]
    fn test_decode_timing() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("test_decode", "qwen3_5");
            // end_prefill resets loop_start for decode measurement
            profiler.end_prefill();

            thread::sleep(Duration::from_millis(10));
            profiler.step();
            profiler.report();

            let store = profiling::PROFILING_STORE.lock().unwrap();
            let last = store.last().unwrap();
            assert!(
                last.decode_ms >= 5.0,
                "decode_ms {} should be >= 5ms",
                last.decode_ms
            );
        });
    }

    #[test]
    fn test_total_ms_equals_prefill_plus_decode() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("test_total", "qwen3_5");
            profiler.begin_prefill();
            thread::sleep(Duration::from_millis(5));
            profiler.end_prefill();

            thread::sleep(Duration::from_millis(5));
            profiler.step();
            profiler.report();

            let store = profiling::PROFILING_STORE.lock().unwrap();
            let last = store.last().unwrap();
            let expected_total = last.prefill_ms + last.decode_ms;
            assert!(
                (last.total_ms - expected_total).abs() < 0.01,
                "total_ms {} should equal prefill_ms {} + decode_ms {}",
                last.total_ms,
                last.prefill_ms,
                last.decode_ms
            );
        });
    }

    #[test]
    fn test_tokens_per_second() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("test_tps", "qwen3");
            profiler.end_prefill(); // reset loop_start

            // Generate 10 "tokens"
            for _ in 0..10 {
                profiler.step();
            }
            thread::sleep(Duration::from_millis(10));
            profiler.report();

            let store = profiling::PROFILING_STORE.lock().unwrap();
            let last = store.last().unwrap();
            assert_eq!(last.num_tokens, 10);
            assert!(
                last.tokens_per_second > 0.0,
                "tok/s should be positive: {}",
                last.tokens_per_second
            );
        });
    }

    #[test]
    fn test_mark_first_token_ttft() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("test_ttft", "qwen3_5");
            profiler.begin_prefill();
            thread::sleep(Duration::from_millis(5));
            profiler.end_prefill();

            // First token extracted after a small delay
            thread::sleep(Duration::from_millis(5));
            profiler.mark_first_token();

            profiler.step();
            profiler.report();

            let store = profiling::PROFILING_STORE.lock().unwrap();
            let last = store.last().unwrap();
            // TTFT = prefill_ms + time from loop_start to mark_first_token
            assert!(
                last.time_to_first_token_ms >= 5.0,
                "ttft {} should be >= 5ms",
                last.time_to_first_token_ms
            );
        });
    }

    #[test]
    fn test_mark_first_token_only_first_call() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("test_ttft_once", "qwen3_5");
            profiler.end_prefill();

            profiler.mark_first_token();
            let t1 = profiler.first_token_time;

            thread::sleep(Duration::from_millis(5));
            profiler.mark_first_token(); // should be no-op
            let t2 = profiler.first_token_time;

            // Both should be the same instant (second call was no-op)
            assert_eq!(t1, t2);
        });
    }

    #[test]
    fn test_phase_tracking() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("test_phases", "qwen3");

            for _ in 0..5 {
                profiler.begin("forward");
                thread::sleep(Duration::from_micros(100));
                profiler.end();

                profiler.begin("sample");
                profiler.end();

                profiler.step();
            }

            profiler.report();

            let store = profiling::PROFILING_STORE.lock().unwrap();
            let last = store.last().unwrap();

            assert_eq!(last.phases.len(), 2);

            let forward = last.phases.iter().find(|p| p.name == "forward").unwrap();
            assert_eq!(forward.count, 5);
            assert!(forward.total_ms > 0.0);
            assert!(forward.avg_us_per_token > 0.0);

            let sample = last.phases.iter().find(|p| p.name == "sample").unwrap();
            assert_eq!(sample.count, 5);
        });
    }

    #[test]
    fn test_phase_order_preserved() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("test_order", "qwen3_5");

            profiler.begin("eval_token");
            profiler.end();
            profiler.begin("extract");
            profiler.end();
            profiler.begin("forward");
            profiler.end();
            profiler.begin("sample");
            profiler.end();
            profiler.step();

            profiler.report();

            let store = profiling::PROFILING_STORE.lock().unwrap();
            let last = store.last().unwrap();

            let phase_names: Vec<&str> = last.phases.iter().map(|p| p.name.as_str()).collect();
            assert_eq!(
                phase_names,
                vec!["eval_token", "extract", "forward", "sample"]
            );
        });
    }

    #[test]
    fn test_nested_phases() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("test_nested", "qwen3_5");

            // Open outer phase A, then nested phase B.
            profiler.begin("outer_a");
            profiler.begin("inner_b");
            thread::sleep(Duration::from_millis(5));
            profiler.end(); // closes inner_b (records >= 5ms)
            thread::sleep(Duration::from_millis(5));
            profiler.end(); // closes outer_a (records >= 10ms total)

            profiler.step();
            profiler.report();

            let store = profiling::PROFILING_STORE.lock().unwrap();
            let last = store.last().unwrap();

            // Both phases must be present.
            let outer = last
                .phases
                .iter()
                .find(|p| p.name == "outer_a")
                .expect("outer_a phase should be recorded");
            let inner = last
                .phases
                .iter()
                .find(|p| p.name == "inner_b")
                .expect("inner_b phase should be recorded");

            // Outer's total time encompasses inner's, so outer >= inner.
            assert!(
                outer.total_ms >= inner.total_ms,
                "outer_a total_ms {} should be >= inner_b total_ms {}",
                outer.total_ms,
                inner.total_ms,
            );
            // Inner slept ~5ms; allow generous slack for CI timer jitter.
            assert!(
                inner.total_ms >= 2.0,
                "inner_b total_ms {} should be >= 2ms",
                inner.total_ms,
            );
            // Outer slept ~10ms total (5ms before inner closed + 5ms after).
            assert!(
                outer.total_ms >= 5.0,
                "outer_a total_ms {} should be >= 5ms",
                outer.total_ms,
            );

            // phase_order is first-seen: outer_a was begun first, so it must
            // appear before inner_b in the recorded order.
            let names: Vec<&str> = last.phases.iter().map(|p| p.name.as_str()).collect();
            let pos_outer = names
                .iter()
                .position(|&n| n == "outer_a")
                .expect("outer_a in order");
            let pos_inner = names
                .iter()
                .position(|&n| n == "inner_b")
                .expect("inner_b in order");
            assert!(
                pos_outer < pos_inner,
                "outer_a (pos {}) should appear before inner_b (pos {}) in first-seen order",
                pos_outer,
                pos_inner,
            );
        });
    }

    #[test]
    fn test_record_duration() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("test_record_duration", "qwen3_5");

            profiler.record_duration("foo", Duration::from_micros(1234));
            // Must not touch the begin/end stack.
            assert!(profiler.phase_stack.is_empty());

            profiler.step();
            profiler.report();

            let store = profiling::PROFILING_STORE.lock().unwrap();
            let last = store.last().unwrap();

            let foo = last
                .phases
                .iter()
                .find(|p| p.name == "foo")
                .expect("foo phase should be recorded");
            assert_eq!(foo.count, 1);
            // 1234us = 1.234ms; allow tiny float slack.
            assert!(
                (foo.total_ms - 1.234).abs() < 0.001,
                "foo total_ms {} should be 1.234",
                foo.total_ms,
            );
        });
    }

    #[test]
    fn test_record_mtp_cycle() {
        // record_mtp_cycle is ungated — works on a disabled profiler.
        let mut profiler = DecodeProfiler::new("test_mtp", "qwen3_5");

        // No cycles yet → no summary.
        assert!(profiler.mtp_acceptance_summary().is_none());
        assert!(profiler.mtp_mean_depth().is_none());

        // depth=3 cycles with K = 3, 1, 2 → mean accepted = 6/3 = 2.0.
        profiler.record_mtp_cycle(3, 3);
        profiler.record_mtp_cycle(3, 1);
        profiler.record_mtp_cycle(3, 2);

        let (mean, per_pos, cycles) = profiler
            .mtp_acceptance_summary()
            .expect("summary after 3 cycles");
        assert_eq!(cycles, 3);
        assert!((mean - 2.0).abs() < 1e-9, "mean accepted {mean} != 2.0");
        assert!(
            (profiler.mtp_mean_depth().expect("mean depth") - 3.0).abs() < 1e-9,
            "mean depth should track attempted draft depth"
        );
        // Pos 0 accepted in all 3 cycles (K>=1): 3/3. Pos 1 when K>=2
        // (K=3,2): 2/3. Pos 2 when K>=3 (K=3 only): 1/3.
        assert_eq!(per_pos.len(), 3);
        assert!((per_pos[0] - 1.0).abs() < 1e-9);
        assert!((per_pos[1] - 2.0 / 3.0).abs() < 1e-9);
        assert!((per_pos[2] - 1.0 / 3.0).abs() < 1e-9);

        // fill_mtp_acceptance copies the summary onto PerformanceMetrics.
        // mlx-vlm-comparable: drafts-only mean (2.0) + 1.0 always-verified.
        assert!(
            (profiler
                .mtp_mean_accepted_tokens_total()
                .expect("total after 3 cycles")
                - 3.0)
                .abs()
                < 1e-9,
            "mlx-vlm-comparable total should be drafts-only + 1.0"
        );

        let mut m = crate::profiling::PerformanceMetrics {
            ttft_ms: 0.0,
            prefill_tokens_per_second: 0.0,
            decode_tokens_per_second: 0.0,
            mtp_mean_accepted_tokens: None,
            mtp_mean_accepted_tokens_total: None,
            mtp_acceptance_by_position: None,
            mtp_cycles: None,
            mtp_mean_depth: None,
            profile_phases: None,
        };
        profiler.fill_mtp_acceptance(&mut m);
        assert_eq!(m.mtp_cycles, Some(3));
        assert!((m.mtp_mean_depth.expect("mean depth") - 3.0).abs() < 1e-9);
        assert!((m.mtp_mean_accepted_tokens.expect("mean") - 2.0).abs() < 1e-9);
        assert!(
            (m.mtp_mean_accepted_tokens_total.expect("total") - 3.0).abs() < 1e-9,
            "PerformanceMetrics total == drafts-only + 1.0"
        );
        assert_eq!(
            m.mtp_acceptance_by_position
                .as_ref()
                .expect("per_pos")
                .len(),
            3
        );
    }

    #[test]
    fn test_fill_mtp_acceptance_copies_phase_profiles_when_enabled() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("test_perf_phases", "qwen3_5");
            profiler.begin("forward");
            thread::sleep(Duration::from_micros(100));
            profiler.end();
            profiler.record_mtp_cycle(2, 1);
            profiler.step();

            let mut m = crate::profiling::PerformanceMetrics {
                ttft_ms: 0.0,
                prefill_tokens_per_second: 0.0,
                decode_tokens_per_second: 0.0,
                mtp_mean_accepted_tokens: None,
                mtp_mean_accepted_tokens_total: None,
                mtp_acceptance_by_position: None,
                mtp_cycles: None,
                mtp_mean_depth: None,
                profile_phases: None,
            };
            profiler.fill_mtp_acceptance(&mut m);

            assert_eq!(m.mtp_cycles, Some(1));
            assert_eq!(m.mtp_mean_depth, Some(2.0));
            let phases = m.profile_phases.expect("phase profiles");
            assert_eq!(phases.len(), 1);
            assert_eq!(phases[0].name, "forward");
            assert_eq!(phases[0].count, 1);
        });
    }

    #[test]
    fn test_record_mtp_cycle_clamps_overaccept() {
        let mut profiler = DecodeProfiler::new("test_mtp_clamp", "qwen3_5");
        // accepted_drafts > depth is clamped to depth.
        profiler.record_mtp_cycle(2, 5);
        let (mean, per_pos, cycles) = profiler.mtp_acceptance_summary().expect("summary");
        assert_eq!(cycles, 1);
        assert!((mean - 2.0).abs() < 1e-9);
        assert_eq!(per_pos, vec![1.0, 1.0]);
    }

    /// The mlx-vlm-comparable total is exactly the drafts-only mean + 1.0
    /// for any non-zero-cycle state, and `None` when no cycle ran.
    #[test]
    fn test_mtp_mean_accepted_tokens_total_is_drafts_plus_one() {
        let mut profiler = DecodeProfiler::new("test_mtp_total", "qwen3_5");
        // No cycle recorded yet → total is None (matches drafts-only None).
        assert!(profiler.mtp_mean_accepted_tokens_total().is_none());
        assert!(profiler.mtp_acceptance_summary().is_none());

        // Mixed depths/accepts: drafts-only mean = (2 + 0 + 1) / 3 = 1.0.
        profiler.record_mtp_cycle(2, 2);
        profiler.record_mtp_cycle(1, 0);
        profiler.record_mtp_cycle(2, 1);

        let (drafts_only, _per_pos, cycles) = profiler.mtp_acceptance_summary().expect("summary");
        assert_eq!(cycles, 3);
        assert!((drafts_only - 1.0).abs() < 1e-9);

        let total = profiler
            .mtp_mean_accepted_tokens_total()
            .expect("total after cycles");
        // mlx-vlm: (accepted_drafts + rounds) / rounds == drafts_only + 1.0.
        assert!(
            (total - (drafts_only + 1.0)).abs() < 1e-9,
            "total {total} != drafts_only {drafts_only} + 1.0"
        );
        assert!((total - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_memory_snapshots() {
        with_profiling(|| {
            let mut profiler = DecodeProfiler::new("test_memory", "qwen3_5");
            profiler.snapshot_memory_before();
            profiler.step();
            profiler.snapshot_memory_after();
            profiler.report();

            let store = profiling::PROFILING_STORE.lock().unwrap();
            let last = store.last().unwrap();
            assert!(last.memory_before.is_some());
            assert!(last.memory_after.is_some());

            let before = last.memory_before.as_ref().unwrap();
            assert!(before.active_bytes >= 0.0);
            assert!(before.peak_bytes >= 0.0);
            assert!(before.cache_bytes >= 0.0);
        });
    }

    #[test]
    fn test_report_does_nothing_when_disabled() {
        // When disabled, report() should not push to the store
        profiling::PROFILING_ENABLED.store(false, Ordering::Relaxed);
        let initial_count = profiling::PROFILING_STORE.lock().unwrap().len();

        let mut profiler = DecodeProfiler::new("test_disabled", "qwen3");
        profiler.step();
        profiler.report();

        let new_count = profiling::PROFILING_STORE.lock().unwrap().len();
        assert_eq!(
            new_count, initial_count,
            "store should not grow when disabled"
        );
    }

    #[test]
    fn test_report_does_nothing_with_zero_tokens() {
        with_profiling(|| {
            let initial_count = profiling::PROFILING_STORE.lock().unwrap().len();

            let profiler = DecodeProfiler::new("test_zero", "qwen3");
            // No step() calls — num_tokens == 0
            profiler.report();

            let new_count = profiling::PROFILING_STORE.lock().unwrap().len();
            assert_eq!(
                new_count, initial_count,
                "store should not grow with zero tokens"
            );
        });
    }

    #[test]
    fn test_disabled_methods_are_noops() {
        profiling::PROFILING_ENABLED.store(false, Ordering::Relaxed);

        let mut profiler = DecodeProfiler::new("test_noop", "qwen3");
        // All of these should be no-ops and not panic
        profiler.set_prompt_tokens(100);
        profiler.begin_prefill();
        profiler.end_prefill();
        profiler.snapshot_memory_before();
        profiler.snapshot_memory_after();
        profiler.mark_first_token();
        profiler.begin("forward");
        profiler.end();
        profiler.step();
        profiler.report();

        // prompt_tokens should remain 0 (was gated by enabled check)
        assert_eq!(profiler.prompt_tokens, 0);
    }

    #[test]
    fn test_multiple_reports_push_multiple_profiles() {
        with_profiling(|| {
            let initial_count = profiling::PROFILING_STORE.lock().unwrap().len();

            // First generation
            let mut profiler1 = DecodeProfiler::new("gen1", "qwen3_5");
            profiler1.set_prompt_tokens(10);
            for _ in 0..5 {
                profiler1.step();
            }
            profiler1.report();

            // Second generation
            let mut profiler2 = DecodeProfiler::new("gen2", "qwen3_5");
            profiler2.set_prompt_tokens(20);
            for _ in 0..10 {
                profiler2.step();
            }
            profiler2.report();

            let store = profiling::PROFILING_STORE.lock().unwrap();
            assert_eq!(store.len(), initial_count + 2);

            let p1 = &store[initial_count];
            let p2 = &store[initial_count + 1];
            assert_eq!(p1.label, "gen1");
            assert_eq!(p1.num_tokens, 5);
            assert_eq!(p1.prompt_tokens, 10);
            assert_eq!(p2.label, "gen2");
            assert_eq!(p2.num_tokens, 10);
            assert_eq!(p2.prompt_tokens, 20);
        });
    }
}
