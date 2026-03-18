//! Global profiling store and NAPI exports.
//!
//! Provides programmatic control over decode loop profiling.
//! When enabled, `DecodeProfiler` pushes `GenerationProfile` entries
//! into a global store which can be retrieved from JavaScript.
//!
//! Zero overhead when disabled: a single `AtomicBool::load(Relaxed)` (~1ns).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use napi_derive::napi;

/// Fast check — inlined, ~1ns per call.
pub(crate) static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Session start time (reset on `reset_profiling_data`).
pub(crate) static SESSION_START: LazyLock<Mutex<Instant>> =
    LazyLock::new(|| Mutex::new(Instant::now()));

/// Accumulated generation profiles.
pub(crate) static PROFILING_STORE: LazyLock<Mutex<Vec<GenerationProfile>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

// ── Data structures ────────────────────────────────────────────────

/// Lightweight performance metrics returned by chat/chatStream when
/// `reportPerformance: true` is set in the config.
#[napi(object)]
#[derive(Clone, Debug)]
pub struct PerformanceMetrics {
    /// Time to first token (ms) — wall-clock from generation start to
    /// first token extracted. Includes tokenization, prefill (lazy graph
    /// construction + first GPU eval), and first sample.
    pub ttft_ms: f64,
    /// Prefill throughput: prompt_tokens / (ttft_ms / 1000).
    pub prefill_tokens_per_second: f64,
    /// Decode throughput: (generated_tokens - 1) / decode_time.
    /// Excludes the first token (counted as prefill).
    pub decode_tokens_per_second: f64,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct MemorySnapshot {
    /// Active (non-cached) memory in bytes.
    pub active_bytes: f64,
    /// Peak memory usage in bytes.
    pub peak_bytes: f64,
    /// Cache memory in bytes.
    pub cache_bytes: f64,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct PhaseProfile {
    /// Phase name (e.g. "forward", "sample", "eval_token").
    pub name: String,
    /// Total wall-clock time spent in this phase (ms).
    pub total_ms: f64,
    /// Average time per invocation (µs).
    pub avg_us_per_token: f64,
    /// Number of invocations.
    pub count: u32,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct GenerationProfile {
    /// Label identifying the decode loop variant.
    pub label: String,
    /// Model type (e.g. "qwen3_5", "qwen3_5_moe", "qwen3").
    pub model_type: String,
    /// Number of tokens generated.
    pub num_tokens: u32,
    /// Number of prompt tokens.
    pub prompt_tokens: u32,
    /// Prefill wall-clock time (ms).
    pub prefill_ms: f64,
    /// Decode wall-clock time (ms).
    pub decode_ms: f64,
    /// Total wall-clock time (prefill + decode) (ms).
    pub total_ms: f64,
    /// Tokens per second (decode only).
    pub tokens_per_second: f64,
    /// Time to first token (ms) — from decode loop start to first token extracted.
    pub time_to_first_token_ms: f64,
    /// Per-phase breakdown.
    pub phases: Vec<PhaseProfile>,
    /// Memory snapshot before generation.
    pub memory_before: Option<MemorySnapshot>,
    /// Memory snapshot after generation.
    pub memory_after: Option<MemorySnapshot>,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct GpuInfo {
    /// GPU architecture generation (M1=13, M2=14, M3=15, M4=16, M5=17).
    pub architecture_gen: i32,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct ProfilingSummary {
    /// Total tokens generated across all generations.
    pub total_tokens: u32,
    /// Total prompt tokens across all generations.
    pub total_prompt_tokens: u32,
    /// Average tokens per second.
    pub avg_tokens_per_second: f64,
    /// Average time to first token (ms).
    pub avg_time_to_first_token_ms: f64,
    /// Average prefill time (ms).
    pub avg_prefill_ms: f64,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct ProfilingSession {
    /// GPU hardware info.
    pub gpu_info: GpuInfo,
    /// Total session duration (ms).
    pub total_duration_ms: f64,
    /// Individual generation profiles.
    pub generations: Vec<GenerationProfile>,
    /// Aggregate summary.
    pub summary: ProfilingSummary,
}

// ── NAPI exports ───────────────────────────────────────────────────

/// Enable or disable profiling globally.
#[napi]
pub fn set_profiling_enabled(enabled: bool) {
    PROFILING_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Check whether profiling is currently enabled.
#[napi]
pub fn is_profiling_enabled() -> bool {
    PROFILING_ENABLED.load(Ordering::Relaxed)
}

/// Retrieve all collected profiling data as a `ProfilingSession`.
#[napi]
pub fn get_profiling_data() -> ProfilingSession {
    let generations = PROFILING_STORE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let session_ms = SESSION_START
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .elapsed()
        .as_secs_f64()
        * 1000.0;

    let gpu_info = GpuInfo {
        architecture_gen: unsafe { mlx_sys::mlx_gpu_architecture_gen() },
    };

    let summary = build_summary(&generations);

    ProfilingSession {
        gpu_info,
        total_duration_ms: session_ms,
        generations,
        summary,
    }
}

/// Clear all collected profiling data and reset session timer.
#[napi]
pub fn reset_profiling_data() {
    PROFILING_STORE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
    *SESSION_START.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
}

// ── Internal API (called by DecodeProfiler) ────────────────────────

/// Fast inline check used by `DecodeProfiler` to decide whether to record.
#[inline]
pub fn is_active() -> bool {
    PROFILING_ENABLED.load(Ordering::Relaxed)
}

/// Push a completed generation profile into the global store.
pub fn push_generation(profile: GenerationProfile) {
    PROFILING_STORE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(profile);
}

/// Take a memory snapshot at this instant.
pub fn snapshot_memory() -> MemorySnapshot {
    use crate::array::{get_active_memory, get_cache_memory, get_peak_memory};
    MemorySnapshot {
        active_bytes: get_active_memory(),
        peak_bytes: get_peak_memory(),
        cache_bytes: get_cache_memory(),
    }
}

// ── Helpers ────────────────────────────────────────────────────────

fn build_summary(generations: &[GenerationProfile]) -> ProfilingSummary {
    if generations.is_empty() {
        return ProfilingSummary {
            total_tokens: 0,
            total_prompt_tokens: 0,
            avg_tokens_per_second: 0.0,
            avg_time_to_first_token_ms: 0.0,
            avg_prefill_ms: 0.0,
        };
    }

    let n = generations.len() as f64;
    let total_tokens: u32 = generations.iter().map(|g| g.num_tokens).sum();
    let total_prompt_tokens: u32 = generations.iter().map(|g| g.prompt_tokens).sum();
    let avg_tps: f64 = generations.iter().map(|g| g.tokens_per_second).sum::<f64>() / n;
    let avg_ttft: f64 = generations
        .iter()
        .map(|g| g.time_to_first_token_ms)
        .sum::<f64>()
        / n;
    let avg_prefill: f64 = generations.iter().map(|g| g.prefill_ms).sum::<f64>() / n;

    ProfilingSummary {
        total_tokens,
        total_prompt_tokens,
        avg_tokens_per_second: avg_tps,
        avg_time_to_first_token_ms: avg_ttft,
        avg_prefill_ms: avg_prefill,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a test GenerationProfile with controlled values.
    fn make_profile(
        label: &str,
        model_type: &str,
        num_tokens: u32,
        prompt_tokens: u32,
        prefill_ms: f64,
        decode_ms: f64,
        tps: f64,
        ttft_ms: f64,
    ) -> GenerationProfile {
        GenerationProfile {
            label: label.to_string(),
            model_type: model_type.to_string(),
            num_tokens,
            prompt_tokens,
            prefill_ms,
            decode_ms,
            total_ms: prefill_ms + decode_ms,
            tokens_per_second: tps,
            time_to_first_token_ms: ttft_ms,
            phases: vec![],
            memory_before: None,
            memory_after: None,
        }
    }

    #[test]
    fn test_build_summary_empty() {
        let summary = build_summary(&[]);
        assert_eq!(summary.total_tokens, 0);
        assert_eq!(summary.total_prompt_tokens, 0);
        assert_eq!(summary.avg_tokens_per_second, 0.0);
        assert_eq!(summary.avg_time_to_first_token_ms, 0.0);
        assert_eq!(summary.avg_prefill_ms, 0.0);
    }

    #[test]
    fn test_build_summary_single_generation() {
        let profiles = vec![make_profile(
            "chat", "qwen3_5", 128, 42, 200.0, 2000.0, 64.0, 210.0,
        )];
        let summary = build_summary(&profiles);

        assert_eq!(summary.total_tokens, 128);
        assert_eq!(summary.total_prompt_tokens, 42);
        assert!((summary.avg_tokens_per_second - 64.0).abs() < 0.01);
        assert!((summary.avg_time_to_first_token_ms - 210.0).abs() < 0.01);
        assert!((summary.avg_prefill_ms - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_build_summary_multiple_generations() {
        let profiles = vec![
            make_profile("chat", "qwen3_5", 100, 40, 200.0, 2000.0, 50.0, 210.0),
            make_profile("chat", "qwen3_5", 200, 60, 300.0, 3000.0, 66.67, 310.0),
        ];
        let summary = build_summary(&profiles);

        assert_eq!(summary.total_tokens, 300);
        assert_eq!(summary.total_prompt_tokens, 100);
        // avg tps = (50 + 66.67) / 2 = 58.335
        assert!((summary.avg_tokens_per_second - 58.335).abs() < 0.01);
        // avg ttft = (210 + 310) / 2 = 260
        assert!((summary.avg_time_to_first_token_ms - 260.0).abs() < 0.01);
        // avg prefill = (200 + 300) / 2 = 250
        assert!((summary.avg_prefill_ms - 250.0).abs() < 0.01);
    }

    #[test]
    fn test_is_active_reflects_atomic() {
        // Note: since these are global statics, tests may interact.
        // We save/restore state.
        let was_active = is_active();

        PROFILING_ENABLED.store(true, Ordering::Relaxed);
        assert!(is_active());

        PROFILING_ENABLED.store(false, Ordering::Relaxed);
        assert!(!is_active());

        // Restore
        PROFILING_ENABLED.store(was_active, Ordering::Relaxed);
    }

    #[test]
    fn test_push_and_clear_generation() {
        // Save current state
        let was_active = is_active();
        let prev_count = PROFILING_STORE.lock().unwrap().len();

        // Push a profile
        let profile = make_profile("test", "test_model", 10, 5, 10.0, 100.0, 100.0, 12.0);
        push_generation(profile);

        let new_count = PROFILING_STORE.lock().unwrap().len();
        assert_eq!(new_count, prev_count + 1);

        // Verify the last entry
        let store = PROFILING_STORE.lock().unwrap();
        let last = store.last().unwrap();
        assert_eq!(last.label, "test");
        assert_eq!(last.model_type, "test_model");
        assert_eq!(last.num_tokens, 10);
        assert_eq!(last.prompt_tokens, 5);
        drop(store);

        // Clear
        PROFILING_STORE.lock().unwrap().clear();
        assert_eq!(PROFILING_STORE.lock().unwrap().len(), 0);

        // Restore
        PROFILING_ENABLED.store(was_active, Ordering::Relaxed);
    }

    #[test]
    fn test_memory_snapshot_returns_valid_values() {
        let snap = snapshot_memory();
        // Memory values should be non-negative
        assert!(snap.active_bytes >= 0.0);
        assert!(snap.peak_bytes >= 0.0);
        assert!(snap.cache_bytes >= 0.0);
        // Peak should be >= active (peak tracks the max)
        assert!(snap.peak_bytes >= snap.active_bytes);
    }

    #[test]
    fn test_generation_profile_with_phases() {
        let profile = GenerationProfile {
            label: "test_phases".to_string(),
            model_type: "qwen3".to_string(),
            num_tokens: 50,
            prompt_tokens: 20,
            prefill_ms: 100.0,
            decode_ms: 500.0,
            total_ms: 600.0,
            tokens_per_second: 100.0,
            time_to_first_token_ms: 110.0,
            phases: vec![
                PhaseProfile {
                    name: "forward".to_string(),
                    total_ms: 300.0,
                    avg_us_per_token: 6000.0,
                    count: 50,
                },
                PhaseProfile {
                    name: "sample".to_string(),
                    total_ms: 50.0,
                    avg_us_per_token: 1000.0,
                    count: 50,
                },
                PhaseProfile {
                    name: "eval_token".to_string(),
                    total_ms: 100.0,
                    avg_us_per_token: 2000.0,
                    count: 50,
                },
            ],
            memory_before: Some(MemorySnapshot {
                active_bytes: 1e9,
                peak_bytes: 1.5e9,
                cache_bytes: 0.0,
            }),
            memory_after: Some(MemorySnapshot {
                active_bytes: 1.2e9,
                peak_bytes: 2.0e9,
                cache_bytes: 5e8,
            }),
        };

        assert_eq!(profile.phases.len(), 3);
        assert_eq!(profile.phases[0].name, "forward");
        assert_eq!(profile.phases[0].count, 50);
        assert!(profile.memory_before.is_some());
        assert!(profile.memory_after.is_some());
        assert_eq!(profile.memory_after.as_ref().unwrap().cache_bytes, 5e8);
    }

    #[test]
    fn test_generation_profile_without_memory() {
        let profile = make_profile("no_mem", "qwen3", 10, 5, 10.0, 100.0, 100.0, 12.0);
        assert!(profile.memory_before.is_none());
        assert!(profile.memory_after.is_none());
    }
}
