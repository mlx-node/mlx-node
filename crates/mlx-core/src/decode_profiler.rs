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
    current_phase: Option<&'static str>,
    phase_start: Instant,
    loop_start: Instant,
    num_tokens: u64,
    prompt_tokens: u32,
    prefill_start: Option<Instant>,
    prefill_ms: f64,
    first_token_time: Option<Instant>,
    first_token_marked: bool,
    memory_before: Option<MemorySnapshot>,
    memory_after: Option<MemorySnapshot>,
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
            current_phase: None,
            phase_start: Instant::now(),
            loop_start: Instant::now(),
            num_tokens: 0,
            prompt_tokens: 0,
            prefill_start: None,
            prefill_ms: 0.0,
            first_token_time: None,
            first_token_marked: false,
            memory_before: None,
            memory_after: None,
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
    #[inline]
    pub fn begin(&mut self, phase: &'static str) {
        if !self.enabled {
            return;
        }
        self.current_phase = Some(phase);
        self.phase_start = Instant::now();
    }

    /// End the current phase and accumulate its time.
    #[inline]
    pub fn end(&mut self) {
        if !self.enabled {
            return;
        }
        if let Some(phase) = self.current_phase.take() {
            let elapsed_us = self.phase_start.elapsed().as_micros() as u64;
            let stats = self.phases.entry(phase).or_insert_with(|| {
                self.phase_order.push(phase);
                PhaseStats {
                    total_us: 0,
                    count: 0,
                }
            });
            stats.total_us += elapsed_us;
            stats.count += 1;
        }
    }

    /// Mark one token as completed. Call once per decode step.
    #[inline]
    pub fn step(&mut self) {
        self.num_tokens += 1;
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
                let from_loop_start = t.duration_since(self.loop_start).as_secs_f64() * 1000.0;
                self.prefill_ms + from_loop_start
            })
            .unwrap_or(0.0);

        // Stderr output (backward compat when env var set)
        if self.env_enabled {
            self.print_stderr_report(n, decode_ms, tok_s);
        }

        // Push to global store (when programmatic profiling is active)
        if profiling::is_active() {
            let phases: Vec<PhaseProfile> = self
                .phase_order
                .iter()
                .filter_map(|&name| {
                    self.phases.get(name).map(|stats| PhaseProfile {
                        name: name.to_string(),
                        total_ms: stats.total_us as f64 / 1000.0,
                        avg_us_per_token: stats.total_us as f64 / n,
                        count: stats.count as u32,
                    })
                })
                .collect();

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
