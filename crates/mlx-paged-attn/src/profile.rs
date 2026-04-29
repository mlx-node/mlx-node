//! Profile-run auto-sizer for the block pool (Phase 3, vLLM-aligned).
//!
//! Replaces the old static `gpu_memory_mb` / `calculate_num_blocks` config
//! knob with a runtime profile that:
//!
//! 1. Reads total unified-memory size from MLX (sysctl-backed on Apple
//!    Silicon).
//! 2. Resets the MLX peak-memory counter.
//! 3. Runs a caller-supplied dummy forward at `(batch=1, seq=max_seq)`,
//!    forcing eval so peak memory reflects weights + activations + working
//!    set the model needs at max-context decode.
//! 4. Reads the post-forward peak (this is the "non-KV" memory the model
//!    will permanently consume during real serving).
//! 5. Computes
//!    `available = total * util - peak_non_kv - safety_margin`
//!    and divides by
//!    `2 * num_layers * num_kv_heads * head_size * block_size * dtype_size`
//!    to get the maximum number of paged blocks the pool can carry.
//!
//! References:
//! - vLLM `vllm/v1/worker/gpu_worker.py::Worker::determine_available_memory`
//!   (lines 332–484) for the formula and env-knob shape (`gpu_memory_
//!   utilization`, the equivalent of `MLX_KV_MEMORY_UTILIZATION`).
//! - vLLM `vllm/v1/worker/gpu_model_runner.py::profile_run` for the
//!   dummy-forward pattern (single max-seq request to produce an
//!   accurate peak-activation estimate).
//!
//! The model loader is supplied as a closure rather than a trait — Phase 3
//! is the auto-sizer utility, not a per-model wiring (Phases 4-9 will hook
//! their own model.forward(synthetic_input) closure into this).
//!
//! Env knobs:
//! - `MLX_KV_MEMORY_UTILIZATION` (default `0.85`): fraction of total memory
//!   the auto-sizer is allowed to budget for everything (weights +
//!   activations + KV pool). The remaining fraction stays free for the OS
//!   and other processes — vLLM's docs explicitly call out the same
//!   reserve-fraction logic. Values outside `(0.0, 1.0]` return
//!   `ProfileError::InvalidUtil`.
//! - `MLX_KV_SAFETY_MARGIN_BYTES` (default `1 GiB`): extra reserve subtracted
//!   from the budget AFTER the utilization haircut. Catches MLX cache
//!   regrowth between profile and steady-state, plus per-step temporaries
//!   the profile can miss because they free immediately and don't move
//!   the peak.
//!
//! On hosts where MLX/Metal aren't available (no GPU, no `hw.memsize`),
//! `mlx_total_system_memory()` returns 0 and we surface
//! `ProfileError::TotalMemoryUnavailable` so the caller can fall back to
//! a static heuristic. The auto-sizer is opt-in: callers that don't want
//! it keep using `PagedAttentionConfig::calculate_num_blocks`.

use crate::metal::MetalDtype;

#[cfg(not(target_os = "macos"))]
use std::marker::PhantomData;

/// Default fraction of total memory we let the model + KV pool use
/// (matches vLLM's `gpu_memory_utilization=0.85` default).
pub const DEFAULT_KV_MEMORY_UTILIZATION: f64 = 0.85;

/// Default extra reserve subtracted after the utilization haircut. 1 GiB is
/// vLLM's "non_kv_extra" practical default — covers MLX cache regrowth
/// between profile and steady-state plus per-step temporaries that the
/// profile can miss.
pub const DEFAULT_SAFETY_MARGIN_BYTES: u64 = 1024 * 1024 * 1024;

const ENV_UTIL: &str = "MLX_KV_MEMORY_UTILIZATION";
const ENV_SAFETY: &str = "MLX_KV_SAFETY_MARGIN_BYTES";

/// Errors the auto-sizer can return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileError {
    /// `mlx_total_system_memory()` returned 0 — non-macOS host, or sysctl
    /// failed. Caller should fall back to a static heuristic.
    TotalMemoryUnavailable,
    /// `MLX_KV_MEMORY_UTILIZATION` parsed but outside `(0.0, 1.0]`. The
    /// value rounds to zero or exceeds total — neither is meaningful.
    InvalidUtil(String),
    /// `MLX_KV_SAFETY_MARGIN_BYTES` parsed but couldn't be converted to
    /// u64.
    InvalidSafetyMargin(String),
    /// The caller-supplied dummy forward closure returned an error string.
    /// We propagate verbatim so model wiring bugs are easy to debug.
    DummyForwardFailed(String),
    /// After subtracting peak + safety margin, no budget remains for KV
    /// blocks. Either the model's weights + activations alone exceed the
    /// utilization-fraction budget, or the safety margin is too large.
    InsufficientMemory {
        total_bytes: u64,
        budget_bytes: u64,
        peak_non_kv_bytes: u64,
        safety_margin_bytes: u64,
    },
    /// The configured layer/head/block parameters multiply to zero — the
    /// per-block size would divide by zero. Refuse to compute a block
    /// count. This is a programmer error rather than a runtime condition.
    InvalidShape(String),
    /// Profile produced an estimate of fewer than one block. The auto-sizer
    /// will not return zero (it would silently disable paged attention).
    /// Caller should reduce `num_layers`/`num_kv_heads`/`head_size`, raise
    /// `MLX_KV_MEMORY_UTILIZATION`, or fall back to a static heuristic.
    NotEnoughBlocks {
        budget_bytes: u64,
        bytes_per_block: u64,
    },
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileError::TotalMemoryUnavailable => write!(
                f,
                "total system memory unavailable (mlx_total_system_memory returned 0); falling \
                 back to static heuristic"
            ),
            ProfileError::InvalidUtil(s) => {
                write!(f, "invalid {ENV_UTIL}={s}: must parse as f64 in (0.0, 1.0]")
            }
            ProfileError::InvalidSafetyMargin(s) => write!(
                f,
                "invalid {ENV_SAFETY}={s}: must parse as non-negative integer bytes"
            ),
            ProfileError::DummyForwardFailed(s) => {
                write!(f, "dummy forward closure failed: {s}")
            }
            ProfileError::InsufficientMemory {
                total_bytes,
                budget_bytes,
                peak_non_kv_bytes,
                safety_margin_bytes,
            } => write!(
                f,
                "insufficient memory for KV pool: total={total_bytes}, budget={budget_bytes}, \
                 peak_non_kv={peak_non_kv_bytes}, safety_margin={safety_margin_bytes}; \
                 weights/activations alone exceed the utilization-fraction budget"
            ),
            ProfileError::InvalidShape(s) => write!(f, "invalid shape parameter: {s}"),
            ProfileError::NotEnoughBlocks {
                budget_bytes,
                bytes_per_block,
            } => write!(
                f,
                "profile budget {budget_bytes} B is smaller than per-block size \
                 {bytes_per_block} B; would yield 0 blocks"
            ),
        }
    }
}

impl std::error::Error for ProfileError {}

/// Bytes-per-element for the KV cache storage dtype. Mirrors the kernel
/// instantiation list — see `crates/mlx-paged-attn/src/metal/state.rs`'s
/// `MetalDtype::size`.
pub fn dtype_size_for(dtype: MetalDtype) -> u32 {
    match dtype {
        // Non-FP8 caches are 2 bytes/element regardless of input
        // (kernels only instantiate (half, half) and (bfloat16_t, bfloat16_t)).
        MetalDtype::Float16 | MetalDtype::BFloat16 => 2,
        // FP8 E4M3 cache: 1 byte/element.
        MetalDtype::UChar => 1,
        // Float32 caches are not supported (no kernel instantiation). The
        // profile-run path treats them as 4 bytes for consistency, but
        // `LayerKVPool::new` rejects them at construction so we never
        // actually return a num_blocks for an F32 pool.
        MetalDtype::Float32 => 4,
    }
}

/// Read `MLX_KV_MEMORY_UTILIZATION` (default `DEFAULT_KV_MEMORY_UTILIZATION`)
/// and validate the value. Pulled out of the main flow so unit tests can
/// hit the rejection paths without spawning a process per case.
pub fn read_util_env() -> Result<f64, ProfileError> {
    match std::env::var(ENV_UTIL) {
        Ok(s) => {
            let v: f64 = s
                .parse()
                .map_err(|_| ProfileError::InvalidUtil(s.clone()))?;
            if !v.is_finite() || v <= 0.0 || v > 1.0 {
                return Err(ProfileError::InvalidUtil(s));
            }
            Ok(v)
        }
        Err(_) => Ok(DEFAULT_KV_MEMORY_UTILIZATION),
    }
}

/// Read `MLX_KV_SAFETY_MARGIN_BYTES` (default `DEFAULT_SAFETY_MARGIN_BYTES`).
pub fn read_safety_margin_env() -> Result<u64, ProfileError> {
    match std::env::var(ENV_SAFETY) {
        Ok(s) => s
            .parse::<u64>()
            .map_err(|_| ProfileError::InvalidSafetyMargin(s)),
        Err(_) => Ok(DEFAULT_SAFETY_MARGIN_BYTES),
    }
}

/// Compute per-block bytes following vLLM's KV pool layout:
/// `2 * num_layers * num_kv_heads * head_size * block_size * dtype_size`
/// (factor of 2 covers K and V).
///
/// Returns `Err(InvalidShape)` if any factor is zero.
pub fn bytes_per_block(
    num_layers: u32,
    num_kv_heads: u32,
    head_size: u32,
    block_size: u32,
    dtype: MetalDtype,
) -> Result<u64, ProfileError> {
    if num_layers == 0 || num_kv_heads == 0 || head_size == 0 || block_size == 0 {
        return Err(ProfileError::InvalidShape(format!(
            "num_layers={num_layers}, num_kv_heads={num_kv_heads}, \
             head_size={head_size}, block_size={block_size}; all must be > 0"
        )));
    }
    let dt = dtype_size_for(dtype) as u64;
    Ok(2u64 * num_layers as u64 * num_kv_heads as u64 * head_size as u64 * block_size as u64 * dt)
}

/// Pure-formula step. Given the measured peak and the env-derived knobs,
/// compute `(num_blocks, available_kv_bytes)`. Pulled out so we can test
/// the math without mocking MLX or running a forward.
pub fn compute_num_blocks_from_measurements(
    total_bytes: u64,
    peak_non_kv_bytes: u64,
    util: f64,
    safety_margin_bytes: u64,
    bytes_per_block: u64,
) -> Result<(u32, u64), ProfileError> {
    if bytes_per_block == 0 {
        // Caller built `bytes_per_block` via our helper above which rejects
        // zero, so this is defense-in-depth against a future caller passing
        // a precomputed zero.
        return Err(ProfileError::InvalidShape(
            "bytes_per_block == 0 (zero divisor)".to_string(),
        ));
    }

    let total_f = total_bytes as f64;
    let budget_f = total_f * util;
    // Two ways the subtraction can underflow: `peak_non_kv` already exceeds
    // the utilization budget, or the safety margin alone exceeds whatever
    // remains. Both surface as `InsufficientMemory` with the input values
    // for a debuggable error message.
    let after_peak = budget_f - peak_non_kv_bytes as f64;
    if after_peak <= 0.0 {
        return Err(ProfileError::InsufficientMemory {
            total_bytes,
            budget_bytes: budget_f as u64,
            peak_non_kv_bytes,
            safety_margin_bytes,
        });
    }
    let kv_bytes_f = after_peak - safety_margin_bytes as f64;
    if kv_bytes_f <= 0.0 {
        return Err(ProfileError::InsufficientMemory {
            total_bytes,
            budget_bytes: budget_f as u64,
            peak_non_kv_bytes,
            safety_margin_bytes,
        });
    }

    let kv_bytes = kv_bytes_f as u64;
    let num_blocks_u64 = kv_bytes / bytes_per_block;
    if num_blocks_u64 == 0 {
        return Err(ProfileError::NotEnoughBlocks {
            budget_bytes: kv_bytes,
            bytes_per_block,
        });
    }
    let num_blocks = u32::try_from(num_blocks_u64).unwrap_or(u32::MAX);
    Ok((num_blocks, kv_bytes))
}

/// Probe MLX/Metal for the total system memory budget. On macOS reads
/// `mlx_total_system_memory()` (= `sysctl hw.memsize`); on other platforms
/// returns `Err(TotalMemoryUnavailable)`.
fn read_total_memory_bytes() -> Result<u64, ProfileError> {
    #[cfg(target_os = "macos")]
    {
        let total = unsafe { mlx_sys::mlx_total_system_memory() };
        if total == 0 {
            return Err(ProfileError::TotalMemoryUnavailable);
        }
        Ok(total as u64)
    }

    #[cfg(not(target_os = "macos"))]
    {
        Err(ProfileError::TotalMemoryUnavailable)
    }
}

/// MLX's GPU-visible working-set bound (`MTLDevice
/// recommendedMaxWorkingSetSize`), if available.
///
/// On Apple Silicon this is normally ~75% of unified memory — Metal
/// will refuse to commit allocations beyond this point (or, worse,
/// silently page out). The auto-sizer uses it as the upper bound when
/// `MLX_KV_MEMORY_UTILIZATION` × `hw.memsize` would otherwise
/// over-commit the GPU.
///
/// Returns `None` when:
/// - Metal is unavailable (CPU-only build, no GPU);
/// - the device_info map doesn't have `max_recommended_working_set_size`;
/// - the C++ helper returns 0 for any reason (defensive: the caller
///   then falls back to the physical-RAM budget).
#[cfg(target_os = "macos")]
fn read_working_set_bytes() -> Option<u64> {
    let ws = unsafe { mlx_sys::mlx_max_recommended_working_set_size() };
    if ws == 0 { None } else { Some(ws as u64) }
}

/// Pure-formula step that combines the physical-RAM budget with the
/// optional working-set budget and returns the smaller. Pulled out of the
/// main flow so unit tests can pin the clamp behaviour without spawning
/// a Metal-aware process.
///
/// Returns `(num_blocks, kv_bytes, physical_budget_bytes,
/// working_set_budget_bytes)` so the caller can log all three for
/// auditability. `working_set_budget_bytes` is `None` when no working-set
/// bound was reported (the formula then falls through to the physical
/// budget alone).
pub fn compute_num_blocks_with_working_set(
    total_bytes: u64,
    working_set_bytes: Option<u64>,
    peak_non_kv_bytes: u64,
    util: f64,
    safety_margin_bytes: u64,
    bytes_per_block: u64,
) -> Result<(u32, u64, u64, Option<u64>), ProfileError> {
    // Compute the physical-RAM budget first using the existing math.
    let (physical_blocks, physical_kv_bytes) = compute_num_blocks_from_measurements(
        total_bytes,
        peak_non_kv_bytes,
        util,
        safety_margin_bytes,
        bytes_per_block,
    )?;
    let physical_budget_bytes = physical_kv_bytes;

    let Some(ws_total) = working_set_bytes else {
        // No working-set bound reported → use the physical budget as-is.
        return Ok((
            physical_blocks,
            physical_kv_bytes,
            physical_budget_bytes,
            None,
        ));
    };

    // Compute the working-set budget the same way: util-haircut against
    // the working-set total, minus peak, minus safety margin. We treat
    // the working_set as the "memory ceiling" — util scales it the same
    // way it scales hw.memsize so the operator's `MLX_KV_MEMORY_UTILIZATION`
    // intent is preserved across both branches. Underflow here surfaces
    // as a working-set budget of zero, which falls through to the
    // physical budget (the only remaining option).
    let ws_total_f = ws_total as f64;
    let ws_budget_f = ws_total_f * util;
    let ws_after_peak = ws_budget_f - peak_non_kv_bytes as f64;
    let ws_kv_bytes_f = ws_after_peak - safety_margin_bytes as f64;
    let ws_kv_bytes = if ws_kv_bytes_f <= 0.0 {
        0u64
    } else {
        ws_kv_bytes_f as u64
    };

    // Choose the smaller of the two budgets. The working-set side is
    // strictly tighter on Apple Silicon (recommended < total), so it
    // wins in normal operation; we still keep the physical fallback
    // for the (rare) case where working_set_bytes is misreported as
    // larger than hw.memsize (e.g. driver bug, Paravirtual VM).
    let chosen_kv_bytes = physical_kv_bytes.min(ws_kv_bytes);
    if chosen_kv_bytes < bytes_per_block {
        return Err(ProfileError::NotEnoughBlocks {
            budget_bytes: chosen_kv_bytes,
            bytes_per_block,
        });
    }
    let num_blocks_u64 = chosen_kv_bytes / bytes_per_block;
    let num_blocks = u32::try_from(num_blocks_u64).unwrap_or(u32::MAX);
    Ok((
        num_blocks,
        chosen_kv_bytes,
        physical_budget_bytes,
        Some(ws_kv_bytes),
    ))
}

/// Reset the MLX peak-memory counter. Called immediately before the
/// caller's dummy forward so the post-forward `mlx_get_peak_memory()` read
/// reflects only the model's max footprint during one max-seq forward.
#[cfg(target_os = "macos")]
fn reset_peak_memory() {
    unsafe { mlx_sys::mlx_reset_peak_memory() };
}

/// Read MLX's peak memory counter (in bytes). Includes weights +
/// activations + intermediate buffers — everything the allocator counted
/// as "in flight at once" since the last `mlx_reset_peak_memory()`.
#[cfg(target_os = "macos")]
fn read_peak_memory() -> u64 {
    unsafe { mlx_sys::mlx_get_peak_memory() as u64 }
}

/// Run a caller-supplied dummy forward, measure peak memory, and compute
/// the auto-sized block count.
///
/// `dummy_forward(max_position_embeddings)` MUST:
/// 1. Construct synthetic input shaped `(batch=1, seq=max_position_embeddings)`
///    using the value passed in (NOT a hard-coded constant — the
///    auto-sizer derives the right number from the model config and
///    threads it through here so callers can't accidentally desync the
///    profile shape from the runtime serving shape).
/// 2. Run the model forward through to logits.
/// 3. Call MLX `eval()` so the lazy graph materializes (peak memory only
///    accounts for materialized data — without `eval()` the graph never
///    moves the counter).
///
/// Phase 3 makes this caller-supplied because the model loader is a model-
/// specific concern (Phases 4-9 will plug their own model into this slot).
/// Passing `max_position_embeddings` as a closure argument (rather than
/// expecting the caller to capture it) ensures it always reaches the
/// closure body and matches the value the auto-sizer used in its
/// budget math.
///
/// The peak-memory and budget math is also clamped against MLX's GPU-
/// visible working-set bound (`MTLDevice
/// recommendedMaxWorkingSetSize`). On Apple Silicon this is normally
/// ~75% of unified memory — without the clamp, sizing the pool against
/// raw `hw.memsize` would over-commit the GPU, surface as
/// `MTLBuffer` allocation failures during serving rather than at
/// profile time. The clamp is silent (we just take the smaller of the
/// two budgets) but logged so the chosen budget is auditable.
///
/// Logging: emits an `info!`-level line on success with the measured peak,
/// physical/working-set budgets, the chosen budget, and resulting block
/// count. Errors are returned to the caller — we don't `error!` here
/// because the caller often wants to fall back to a static heuristic on
/// `TotalMemoryUnavailable`.
#[allow(clippy::too_many_arguments)]
pub fn profile_run_and_compute_num_blocks<F>(
    dummy_forward: F,
    max_position_embeddings: u32,
    num_layers: u32,
    num_kv_heads: u32,
    head_size: u32,
    kv_dtype: MetalDtype,
    block_size: u32,
) -> Result<u32, ProfileError>
where
    F: FnOnce(u32) -> Result<(), String>,
{
    let total = read_total_memory_bytes()?;
    let util = read_util_env()?;
    let safety_margin = read_safety_margin_env()?;
    let bpb = bytes_per_block(num_layers, num_kv_heads, head_size, block_size, kv_dtype)?;

    #[cfg(target_os = "macos")]
    {
        reset_peak_memory();
        dummy_forward(max_position_embeddings).map_err(ProfileError::DummyForwardFailed)?;
        let peak = read_peak_memory();
        // Clamp against MLX's recommended working-set bound. Apple Silicon
        // reports ~75% of unified memory as the upper bound MTLDevice will
        // happily commit — past that we get `MTLBuffer` allocation
        // failures during serving rather than at profile time.
        let working_set = read_working_set_bytes();
        let (num_blocks, kv_bytes, physical_bytes, working_set_bytes) =
            compute_num_blocks_with_working_set(
                total,
                working_set,
                peak,
                util,
                safety_margin,
                bpb,
            )?;

        let total_mib = total / (1024 * 1024);
        let peak_mib = peak / (1024 * 1024);
        let kv_mib = kv_bytes / (1024 * 1024);
        let phys_mib = physical_bytes / (1024 * 1024);
        // `n/a` when no working-set bound was reported (CPU-only build
        // or device_info missing the entry); the formula then falls
        // through to the physical budget alone.
        let ws_str = match working_set_bytes {
            Some(b) => format!("{} MiB", b / (1024 * 1024)),
            None => "n/a".to_string(),
        };
        // MEMORY: log via eprintln rather than tracing because mlx-paged-attn
        // is lower in the dependency tree than the workspace tracing setup
        // and we don't want to drag tracing-subscriber into this crate.
        // Callers (mlx-core) bridge into `tracing::info!` from their wrappers
        // when desirable.
        eprintln!(
            "[paged-profile] total {total_mib} MiB, peak_non_kv {peak_mib} MiB, util {util:.3}, \
             safety_margin {safety_margin} B, physical_budget {phys_mib} MiB, \
             working_set_budget {ws_str}, kv_budget {kv_mib} MiB, num_blocks {num_blocks}, \
             block_size {block_size}, bytes_per_block {bpb}, max_seq {max_position_embeddings}"
        );
        Ok(num_blocks)
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Surface the unsupported platform — callers should fall back to a
        // static heuristic. We still call the dummy forward closure so any
        // platform-portable model-loading bugs surface deterministically
        // even without MLX/Metal (the closure receives a pre-platform-checked
        // failure and can pass through `Err`s from its own loader).
        let _ = dummy_forward(max_position_embeddings);
        let _ = (total, util, safety_margin, bpb);
        let _: PhantomData<F> = PhantomData;
        Err(ProfileError::TotalMemoryUnavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_size_table() {
        assert_eq!(dtype_size_for(MetalDtype::Float16), 2);
        assert_eq!(dtype_size_for(MetalDtype::BFloat16), 2);
        assert_eq!(dtype_size_for(MetalDtype::UChar), 1);
        assert_eq!(dtype_size_for(MetalDtype::Float32), 4);
    }

    #[test]
    fn bytes_per_block_formula() {
        // num_layers=32, num_kv_heads=8, head_size=64, block_size=16,
        // bf16 -> 2 bytes/element. Per-block bytes = 2 * 32 * 8 * 64 * 16
        // * 2 = 1,048,576 = 1 MiB.
        let bpb = bytes_per_block(32, 8, 64, 16, MetalDtype::BFloat16).unwrap();
        assert_eq!(bpb, 2 * 32 * 8 * 64 * 16 * 2);
        assert_eq!(bpb, 1024 * 1024);
    }

    #[test]
    fn bytes_per_block_rejects_zero() {
        assert!(matches!(
            bytes_per_block(0, 8, 64, 16, MetalDtype::BFloat16),
            Err(ProfileError::InvalidShape(_))
        ));
        assert!(matches!(
            bytes_per_block(32, 0, 64, 16, MetalDtype::BFloat16),
            Err(ProfileError::InvalidShape(_))
        ));
        assert!(matches!(
            bytes_per_block(32, 8, 0, 16, MetalDtype::BFloat16),
            Err(ProfileError::InvalidShape(_))
        ));
        assert!(matches!(
            bytes_per_block(32, 8, 64, 0, MetalDtype::BFloat16),
            Err(ProfileError::InvalidShape(_))
        ));
    }

    #[test]
    fn formula_basic() {
        // Hand-checked: total=128 GiB, peak=20 GiB, util=0.85, safety=1 GiB.
        // budget = 128 * 0.85 = 108.8 GiB
        // after_peak = 108.8 - 20 = 88.8 GiB
        // kv = 88.8 - 1 = 87.8 GiB ≈ 94,253,830,963 bytes.
        // bpb = 1 MiB → num_blocks ≈ 89,907.
        let total = 128u64 * 1024 * 1024 * 1024;
        let peak = 20u64 * 1024 * 1024 * 1024;
        let util = 0.85;
        let safety = 1024u64 * 1024 * 1024;
        let bpb = 1024u64 * 1024;
        let (num_blocks, kv) =
            compute_num_blocks_from_measurements(total, peak, util, safety, bpb).unwrap();
        // Allow ±1 fp rounding.
        assert!((89_900..=89_915).contains(&num_blocks), "got {num_blocks}");
        assert!(kv > 0);
    }

    #[test]
    fn formula_rejects_peak_exceeds_budget() {
        // peak ≥ total*util → no budget for KV.
        let total = 32u64 * 1024 * 1024 * 1024;
        let peak = 30u64 * 1024 * 1024 * 1024;
        let util = 0.85; // budget = 27.2 GiB
        let safety = 0u64;
        let bpb = 1024u64;
        assert!(matches!(
            compute_num_blocks_from_measurements(total, peak, util, safety, bpb),
            Err(ProfileError::InsufficientMemory { .. })
        ));
    }

    #[test]
    fn formula_rejects_safety_eats_budget() {
        // peak fits, but safety wipes out the remainder.
        let total = 32u64 * 1024 * 1024 * 1024;
        let peak = 25u64 * 1024 * 1024 * 1024;
        let util = 0.85; // budget = 27.2 GiB; after_peak = 2.2 GiB
        let safety = 4u64 * 1024 * 1024 * 1024; // > 2.2 GiB
        let bpb = 1024u64;
        assert!(matches!(
            compute_num_blocks_from_measurements(total, peak, util, safety, bpb),
            Err(ProfileError::InsufficientMemory { .. })
        ));
    }

    #[test]
    fn formula_rejects_too_few_blocks() {
        // Budget < bpb — would round to zero blocks. Auto-sizer refuses to
        // disable paged attention silently.
        let gib = 1024u64 * 1024 * 1024;
        let total = 4 * gib;
        let peak = gib;
        let util = 1.0;
        let safety = 0u64;
        // budget = 4 GiB - 1 GiB = 3 GiB; bpb = 4 GiB.
        let bpb = 4 * gib;
        assert!(matches!(
            compute_num_blocks_from_measurements(total, peak, util, safety, bpb),
            Err(ProfileError::NotEnoughBlocks { .. })
        ));
    }

    #[test]
    fn formula_rejects_zero_bpb() {
        let res = compute_num_blocks_from_measurements(1024, 0, 1.0, 0, 0);
        assert!(matches!(res, Err(ProfileError::InvalidShape(_))));
    }

    #[test]
    fn util_env_default() {
        // Don't mutate global env state in tests — just verify the parser
        // accepts the default constant. Env-overriding tests run in
        // `tests/profile_run.rs` where they're isolated per process.
        let _ = read_util_env(); // value depends on host env; just must not panic
    }

    /// vLLM's reference value: `gpu_memory_utilization=0.9`, peak=10GB,
    /// total=80GB, safety=1GB. Verifies our formula matches their docs.
    #[test]
    fn formula_matches_vllm_reference() {
        let total = 80u64 * 1024 * 1024 * 1024;
        let peak = 10u64 * 1024 * 1024 * 1024;
        let util = 0.9; // budget = 72 GB
        let safety = 1024u64 * 1024 * 1024;
        let bpb = 16 * 1024 * 1024; // 16 MiB / block
        let (num_blocks, kv) =
            compute_num_blocks_from_measurements(total, peak, util, safety, bpb).unwrap();
        // 72 - 10 - 1 = 61 GiB / 16 MiB ≈ 3904 blocks.
        assert!((3900..=3910).contains(&num_blocks), "got {num_blocks}");
        assert!(kv > 0);
    }

    /// No working-set bound reported (CPU-only build / device_info missing
    /// the entry) → the function falls through to the physical-RAM
    /// budget and returns the same num_blocks as
    /// `compute_num_blocks_from_measurements`.
    #[test]
    fn working_set_none_falls_through_to_physical() {
        let total = 64u64 * 1024 * 1024 * 1024;
        let peak = 10u64 * 1024 * 1024 * 1024;
        let util = 0.85;
        let safety = 1024u64 * 1024 * 1024;
        let bpb = 1024u64 * 1024;
        let (phys_blocks, _) =
            compute_num_blocks_from_measurements(total, peak, util, safety, bpb).unwrap();
        let (num_blocks, _, _, ws_bytes) =
            compute_num_blocks_with_working_set(total, None, peak, util, safety, bpb).unwrap();
        assert_eq!(num_blocks, phys_blocks);
        assert_eq!(ws_bytes, None);
    }

    /// Working set < total → working-set budget bounds the result. The
    /// Apple-Silicon-typical case: hw.memsize = 64 GiB, working set =
    /// 48 GiB. A naive auto-sizer would over-commit by 16 GiB; the
    /// clamp keeps the result inside the working-set budget.
    #[test]
    fn working_set_smaller_clamps_blocks() {
        let total = 64u64 * 1024 * 1024 * 1024;
        let working_set = 48u64 * 1024 * 1024 * 1024;
        let peak = 10u64 * 1024 * 1024 * 1024;
        let util = 0.85;
        let safety = 1024u64 * 1024 * 1024;
        let bpb = 1024u64 * 1024;

        // Physical budget alone: 64 * 0.85 - 10 - 1 = 43.4 GiB ≈ 44_447 blocks.
        // Working-set budget:    48 * 0.85 - 10 - 1 = 29.8 GiB ≈ 30_515 blocks.
        // Working set is strictly tighter → wins.
        let (phys_blocks, _) =
            compute_num_blocks_from_measurements(total, peak, util, safety, bpb).unwrap();
        let (clamped_blocks, kv_bytes, phys_budget, ws_budget) =
            compute_num_blocks_with_working_set(total, Some(working_set), peak, util, safety, bpb)
                .unwrap();

        assert!(
            clamped_blocks < phys_blocks,
            "expected clamp to reduce blocks: physical={phys_blocks}, clamped={clamped_blocks}"
        );
        // Sanity: ~30,500 blocks for the working-set budget.
        assert!(
            (30_000..=31_000).contains(&clamped_blocks),
            "expected ~30,515 blocks for working-set budget, got {clamped_blocks}"
        );
        // kv_bytes equals the working-set budget (since it's the smaller).
        let ws_kv = ws_budget.expect("working-set budget reported");
        assert_eq!(kv_bytes, ws_kv.min(phys_budget));
        assert!(ws_kv < phys_budget);
    }

    /// Working set >= total (the rare misreport case: VM driver bug) →
    /// the physical budget is tighter and wins. We still report both
    /// budgets through the return tuple so the eprintln log can show
    /// which side dominated.
    #[test]
    fn working_set_larger_falls_back_to_physical() {
        let total = 32u64 * 1024 * 1024 * 1024;
        let working_set = 64u64 * 1024 * 1024 * 1024; // > total: misreport
        let peak = 4u64 * 1024 * 1024 * 1024;
        let util = 0.85;
        let safety = 1024u64 * 1024 * 1024;
        let bpb = 1024u64 * 1024;
        let (phys_blocks, _) =
            compute_num_blocks_from_measurements(total, peak, util, safety, bpb).unwrap();
        let (num_blocks, _, _, ws_bytes) =
            compute_num_blocks_with_working_set(total, Some(working_set), peak, util, safety, bpb)
                .unwrap();
        assert_eq!(num_blocks, phys_blocks);
        // Working-set budget IS still reported, but it's the larger of
        // the two so the physical budget bounds the result.
        assert!(ws_bytes.is_some());
    }

    /// Both budgets reject the same way when peak exceeds the
    /// utilization haircut: physical leg returns InsufficientMemory
    /// before we even consider the working set. Verifies the
    /// short-circuit behaviour.
    #[test]
    fn working_set_rejects_when_physical_does() {
        let total = 32u64 * 1024 * 1024 * 1024;
        let peak = 30u64 * 1024 * 1024 * 1024; // > 32 * 0.85 = 27.2
        let util = 0.85;
        let safety = 0u64;
        let bpb = 1024u64;
        let res = compute_num_blocks_with_working_set(
            total,
            Some(48u64 * 1024 * 1024 * 1024),
            peak,
            util,
            safety,
            bpb,
        );
        assert!(matches!(res, Err(ProfileError::InsufficientMemory { .. })));
    }

    /// Physical budget passes, but the working-set budget is too small
    /// to fit even one block. The auto-sizer refuses to silently
    /// disable paged attention even with a generous physical budget.
    #[test]
    fn working_set_too_small_rejects() {
        let gib = 1024u64 * 1024 * 1024;
        let total = 64 * gib;
        // Working set so small the post-haircut budget can't fit one block.
        let working_set = 5 * gib; // util * 5 GiB - 4 GiB peak - 1 GiB safety = -0.75 GiB
        let peak = 4 * gib;
        let util = 0.85;
        let safety = gib;
        let bpb = 4 * gib;
        let res =
            compute_num_blocks_with_working_set(total, Some(working_set), peak, util, safety, bpb);
        // The clamp drives chosen_kv_bytes to 0; we surface NotEnoughBlocks
        // (NOT InsufficientMemory — that's the physical-only short-circuit).
        assert!(matches!(res, Err(ProfileError::NotEnoughBlocks { .. })));
    }
}
