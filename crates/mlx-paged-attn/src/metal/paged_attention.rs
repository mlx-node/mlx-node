//! paged_attention Metal kernel dispatch
//!
//! This kernel computes attention using the paged KV cache.
//! Supports both V1 (no partitioning) and V2 (with partitioning) modes.

use super::command_buffer::observe;
use super::reshape_and_cache::RawBufferInfo;
use super::state::{MetalDtype, MetalState};
use metal::foreign_types::ForeignTypeRef;
use metal::{Buffer, BufferRef, MTLSize};
use std::ffi::c_void;
use std::sync::OnceLock;

/// Convert IEEE 754 half-precision (f16) to single-precision (f32)
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal - convert to normalized f32
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = ((127 - 15 + 1 + e) as u32) << 23;
            f32::from_bits((sign << 31) | f32_exp | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
    } else {
        // Normalized
        let f32_exp = ((exp as i32 - 15 + 127) as u32) << 23;
        f32::from_bits((sign << 31) | f32_exp | (mant << 13))
    }
}

/// Convert bfloat16 (bf16) to single-precision (f32)
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    // bf16 is just the upper 16 bits of f32
    f32::from_bits((bits as u32) << 16)
}

/// Parameters for paged_attention kernel
pub struct PagedAttentionParams {
    /// Number of sequences in the batch
    pub num_seqs: u32,
    /// Number of query heads
    pub num_heads: u32,
    /// Number of KV heads
    pub num_kv_heads: u32,
    /// Size of each head
    pub head_size: u32,
    /// Block size for paged attention
    pub block_size: u32,
    /// Maximum sequence length
    pub max_seq_len: u32,
    /// Maximum number of blocks per sequence
    pub max_num_blocks_per_seq: u32,
    /// Attention scale factor (1/sqrt(head_size))
    pub scale: f32,
    /// Softcapping value (1.0 = disabled)
    pub softcapping: f32,
    /// Query stride (num_heads * head_size)
    pub q_stride: i32,
    /// KV block stride
    pub kv_block_stride: i32,
    /// KV head stride
    pub kv_head_stride: i32,
    /// K scale for FP8 quantization (1.0 for non-FP8)
    pub k_scale: f32,
    /// V scale for FP8 quantization (1.0 for non-FP8)
    pub v_scale: f32,
    /// Sliding-window mask. When > 0, K positions older than
    /// `context_len - sliding_window` are excluded from attention. When
    /// 0 (default), no sliding mask is applied (full-context attention).
    pub sliding_window: i32,
}

impl Default for PagedAttentionParams {
    fn default() -> Self {
        Self {
            num_seqs: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_size: 128,
            block_size: 16,
            max_seq_len: 0,
            max_num_blocks_per_seq: 0,
            scale: 1.0,
            softcapping: 1.0,
            q_stride: 0,
            kv_block_stride: 0,
            kv_head_stride: 0,
            k_scale: 1.0,
            v_scale: 1.0,
            sliding_window: 0,
        }
    }
}

/// Partition size for V2 kernel
const PARTITION_SIZE: u32 = 512;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GroupedPagedAttentionKind {
    Qwen35D256,
    D512Direct,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GroupedD512Mode {
    Disabled,
    Auto,
    Force,
}

/// Per-dispatch compute route. This changes only which kernel reads the
/// authoritative paged pool.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PagedAttentionRouteHint {
    #[default]
    Auto,
    /// Legacy compatibility name for forcing the validated grouped D512
    /// route. The canonical production pipeline now uses direct K/V reads.
    ForceD512Staged,
    ForceGeneric,
}

fn grouped_d512_stripe_override_value(value: Option<&str>) -> Option<u32> {
    value
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| matches!(value, 4 | 8 | 16 | 32 | 64 | 128 | 256))
}

fn grouped_d512_stripe_override() -> Option<u32> {
    static STRIPES: OnceLock<Option<u32>> = OnceLock::new();
    *STRIPES.get_or_init(|| {
        grouped_d512_stripe_override_value(
            std::env::var("MLX_PAGED_GROUPED_D512_STRIPES")
                .ok()
                .or_else(|| std::env::var("MLX_PAGED_GROUPED_GEMMA4_STRIPES").ok())
                .as_deref(),
        )
    })
}

// Mirror the graph dispatcher and model-free diagnostics. A fixed-session A/B
// measured Hq16/Hkv2 at 91,795 context for 512 generated tokens: 128 stripes
// delivered 32.8416 tok/s versus 31.0484 for 64 (+5.78%, 3/3 paired wins); the
// raw 112K sweep did not regress.
const D512_HQ16_HKV2_WIDE_STRIPE_CONTEXT: u32 = 88 * 1024 + 1;

fn grouped_d512_default_stripe_count(
    max_context_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
) -> u32 {
    let base_stripes = match max_context_len {
        0..=4096 => 32,
        4097..=8192 => 64,
        _ => 128,
    };
    let stripes = (base_stripes / num_kv_heads.max(1)).max(4);
    if (num_q_heads, num_kv_heads) == (16, 2)
        && max_context_len >= D512_HQ16_HKV2_WIDE_STRIPE_CONTEXT
    {
        128
    } else {
        stripes
    }
}

fn grouped_d512_resolved_stripe_count(
    override_stripes: Option<u32>,
    max_context_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
) -> u32 {
    override_stripes.unwrap_or_else(|| {
        grouped_d512_default_stripe_count(max_context_len, num_q_heads, num_kv_heads)
    })
}

fn grouped_stripe_count(
    kind: GroupedPagedAttentionKind,
    max_context_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
) -> u32 {
    if kind == GroupedPagedAttentionKind::D512Direct {
        return grouped_d512_resolved_stripe_count(
            grouped_d512_stripe_override(),
            max_context_len,
            num_q_heads,
            num_kv_heads,
        );
    }
    match max_context_len {
        0..=4096 => 32,
        4097..=8192 => 64,
        8193..=16383 => 128,
        16384..=32768 => 256,
        32769..=65536 => 512,
        _ => 1024,
    }
}

fn grouped_qwen35_env_enabled_value(value: Option<&str>) -> bool {
    value != Some("0")
}

fn grouped_qwen35_paged_attention_enabled() -> bool {
    // Default-on escape hatch for A/B profiling and driver-specific rollback.
    // Cache once: dispatch is on the token hot path and must not read the
    // process environment for every attention layer/token.
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        grouped_qwen35_env_enabled_value(std::env::var("MLX_PAGED_GROUPED_QWEN35").ok().as_deref())
    })
}

fn grouped_d512_env_mode_value(value: Option<&str>) -> GroupedD512Mode {
    match value {
        Some("1" | "on" | "auto" | "true") => GroupedD512Mode::Auto,
        Some("force") => GroupedD512Mode::Force,
        _ => GroupedD512Mode::Disabled,
    }
}

fn grouped_d512_paged_attention_mode() -> GroupedD512Mode {
    // Cache the generic setting once on the token hot path. The former
    // Gemma-specific variable remains a compatibility alias.
    static MODE: OnceLock<GroupedD512Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        grouped_d512_env_mode_value(
            std::env::var("MLX_PAGED_GROUPED_D512")
                .ok()
                .or_else(|| std::env::var("MLX_PAGED_GROUPED_GEMMA4").ok())
                .as_deref(),
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn grouped_qwen35_shape_matches(
    enabled: bool,
    io_dtype: MetalDtype,
    cache_dtype: MetalDtype,
    num_seqs: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    block_size: u32,
    query_rows: u32,
    max_context_len: u32,
) -> bool {
    let dense_heads = (num_heads, num_kv_heads) == (24, 4);
    let moe_heads = (num_heads, num_kv_heads) == (16, 2);
    let min_context = match (query_rows, moe_heads) {
        (1, true) => 32_768,
        (2, true) => 16_384,
        (1, false) => 16_384,
        _ => 8_192,
    };
    enabled
        && io_dtype == MetalDtype::BFloat16
        && cache_dtype == MetalDtype::BFloat16
        && num_seqs == 1
        && (dense_heads || moe_heads)
        && head_size == 256
        && block_size == 16
        && matches!(query_rows, 1 | 2)
        && max_context_len >= min_context
}

#[allow(clippy::too_many_arguments)]
fn use_grouped_qwen35_paged_attention(
    io_dtype: MetalDtype,
    cache_dtype: MetalDtype,
    num_seqs: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    block_size: u32,
    query_rows: u32,
    max_context_len: u32,
) -> bool {
    grouped_qwen35_shape_matches(
        grouped_qwen35_paged_attention_enabled(),
        io_dtype,
        cache_dtype,
        num_seqs,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        query_rows,
        max_context_len,
    )
}

#[allow(clippy::too_many_arguments)]
fn grouped_d512_shape_matches(
    mode: GroupedD512Mode,
    io_dtype: MetalDtype,
    cache_dtype: MetalDtype,
    num_seqs: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    block_size: u32,
    query_rows: u32,
    max_context_len: u32,
) -> bool {
    let supported_heads = matches!(
        (num_heads, num_kv_heads),
        (8, 1) | (16, 1) | (16, 2) | (32, 4)
    );
    let exact_shape = io_dtype == MetalDtype::BFloat16
        && cache_dtype == MetalDtype::BFloat16
        && num_seqs == 1
        && supported_heads
        && head_size == 512
        && block_size == 16
        && query_rows == 1
        && max_context_len > PARTITION_SIZE;
    exact_shape
        && mode != GroupedD512Mode::Disabled
        && (mode == GroupedD512Mode::Force
            || ((num_heads, num_kv_heads) == (16, 1)
                && (3_072..=16_384).contains(&max_context_len)))
}

#[allow(clippy::too_many_arguments)]
fn select_grouped_paged_attention(
    io_dtype: MetalDtype,
    cache_dtype: MetalDtype,
    num_seqs: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    block_size: u32,
    query_rows: u32,
    max_context_len: u32,
    route_hint: PagedAttentionRouteHint,
) -> Option<GroupedPagedAttentionKind> {
    if route_hint == PagedAttentionRouteHint::ForceGeneric {
        return None;
    }
    if route_hint == PagedAttentionRouteHint::ForceD512Staged {
        return grouped_d512_shape_matches(
            GroupedD512Mode::Force,
            io_dtype,
            cache_dtype,
            num_seqs,
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            query_rows,
            max_context_len,
        )
        .then_some(GroupedPagedAttentionKind::D512Direct);
    }
    if use_grouped_qwen35_paged_attention(
        io_dtype,
        cache_dtype,
        num_seqs,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        query_rows,
        max_context_len,
    ) {
        return Some(GroupedPagedAttentionKind::Qwen35D256);
    }
    if grouped_d512_shape_matches(
        grouped_d512_paged_attention_mode(),
        io_dtype,
        cache_dtype,
        num_seqs,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        query_rows,
        max_context_len,
    ) {
        return Some(GroupedPagedAttentionKind::D512Direct);
    }
    None
}

fn grouped_kernel_name(kind: GroupedPagedAttentionKind) -> &'static str {
    match kind {
        GroupedPagedAttentionKind::Qwen35D256 => {
            MetalState::paged_attention_grouped_qwen35_kernel_name()
        }
        GroupedPagedAttentionKind::D512Direct => {
            MetalState::paged_attention_grouped_d512_kernel_name()
        }
    }
}

fn grouped_reduce_kernel_name(kind: GroupedPagedAttentionKind) -> &'static str {
    match kind {
        GroupedPagedAttentionKind::Qwen35D256 => {
            MetalState::paged_attention_grouped_qwen35_reduce_kernel_name()
        }
        GroupedPagedAttentionKind::D512Direct => {
            MetalState::paged_attention_grouped_d512_reduce_kernel_name()
        }
    }
}

fn grouped_pipelines_supported(
    state: &MetalState,
    kind: GroupedPagedAttentionKind,
    num_heads: u32,
    num_kv_heads: u32,
) -> bool {
    // MetalState owns one process-wide device. Cache this immutable
    // capability result so dispatch does not take two extra pipeline-cache
    // locks per layer and token after the first grouped candidate.
    // A missing/invalid specialized pipeline is an optional-optimization
    // failure, not an inference failure. Cache `false`, warn once through the
    // normal tracing subscriber, and leave every later token on generic V2.
    static QWEN35_LIMITS: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    static GEMMA4_LIMITS: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    let limits = match kind {
        GroupedPagedAttentionKind::Qwen35D256 => &QWEN35_LIMITS,
        GroupedPagedAttentionKind::D512Direct => &GEMMA4_LIMITS,
    };
    let Some((stage_threads, reduce_threads)) = *limits.get_or_init(|| {
        let stage = match state.get_pipeline(grouped_kernel_name(kind)) {
            Ok(stage) => stage,
            Err(error) => {
                tracing::warn!(
                    target: "mlx_paged_attn::metal",
                    %error,
                    ?kind,
                    "grouped paged-attention stage pipeline unavailable; using generic V2"
                );
                return None;
            }
        };
        let reduce = match state.get_pipeline(grouped_reduce_kernel_name(kind)) {
            Ok(reduce) => reduce,
            Err(error) => {
                tracing::warn!(
                    target: "mlx_paged_attn::metal",
                    %error,
                    ?kind,
                    "grouped paged-attention reducer pipeline unavailable; using generic V2"
                );
                return None;
            }
        };
        let stage_threads = stage.max_total_threads_per_threadgroup();
        let reduce_threads = reduce.max_total_threads_per_threadgroup();
        Some((stage_threads, reduce_threads))
    }) else {
        return false;
    };
    if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
        return false;
    }
    let required_stage_threads = u64::from(32 * (num_heads / num_kv_heads));
    let supported = stage_threads >= required_stage_threads && reduce_threads >= 1024;
    if !supported {
        static QWEN35_WARNED: OnceLock<()> = OnceLock::new();
        static GEMMA4_WARNED: OnceLock<()> = OnceLock::new();
        let warned = match kind {
            GroupedPagedAttentionKind::Qwen35D256 => &QWEN35_WARNED,
            GroupedPagedAttentionKind::D512Direct => &GEMMA4_WARNED,
        };
        warned.get_or_init(|| {
            tracing::warn!(
                target: "mlx_paged_attn::metal",
                ?kind,
                stage_threads,
                required_stage_threads,
                reduce_threads,
                "grouped paged-attention threadgroup limits unsupported; using generic V2"
            );
        });
    }
    supported
}

#[cfg(test)]
mod grouped_selection_tests {
    use super::*;

    #[test]
    fn env_escape_hatch_is_default_on_and_zero_disables() {
        assert!(grouped_qwen35_env_enabled_value(None));
        assert!(grouped_qwen35_env_enabled_value(Some("1")));
        assert!(grouped_qwen35_env_enabled_value(Some("true")));
        assert!(!grouped_qwen35_env_enabled_value(Some("0")));
        assert_eq!(grouped_d512_env_mode_value(None), GroupedD512Mode::Disabled);
        assert_eq!(
            grouped_d512_env_mode_value(Some("1")),
            GroupedD512Mode::Auto
        );
        assert_eq!(
            grouped_d512_env_mode_value(Some("force")),
            GroupedD512Mode::Force
        );
        assert_eq!(grouped_d512_stripe_override_value(Some("4")), Some(4));
        assert_eq!(grouped_d512_stripe_override_value(Some("128")), Some(128));
        assert_eq!(grouped_d512_stripe_override_value(Some("3")), None);
        assert_eq!(grouped_d512_stripe_override_value(Some("oops")), None);
    }

    #[test]
    fn exact_qwen35_dense_and_moe_shapes_select_only_when_enabled() {
        let matches = |enabled, num_heads, num_kv_heads, query_rows, max_context_len| {
            grouped_qwen35_shape_matches(
                enabled,
                MetalDtype::BFloat16,
                MetalDtype::BFloat16,
                1,
                num_heads,
                num_kv_heads,
                256,
                16,
                query_rows,
                max_context_len,
            )
        };
        for (num_heads, num_kv_heads, decode_min, verify_min) in
            [(24, 4, 16_384, 8_192), (16, 2, 32_768, 16_384)]
        {
            assert!(matches(true, num_heads, num_kv_heads, 1, decode_min));
            assert!(!matches(true, num_heads, num_kv_heads, 1, decode_min - 1));
            assert!(matches(true, num_heads, num_kv_heads, 2, verify_min));
            assert!(!matches(true, num_heads, num_kv_heads, 2, verify_min - 1));
            assert!(!matches(false, num_heads, num_kv_heads, 1, decode_min));
            assert!(!matches(false, num_heads, num_kv_heads, 2, verify_min));
            assert!(!matches(true, num_heads, num_kv_heads, 3, decode_min));
        }
        assert!(!matches(true, 16, 4, 1, 16_384));
        assert!(!matches(true, 24, 2, 1, 16_384));

        assert!(!grouped_qwen35_shape_matches(
            true,
            MetalDtype::Float16,
            MetalDtype::BFloat16,
            1,
            24,
            4,
            256,
            16,
            1,
            16_384,
        ));
        assert!(!grouped_qwen35_shape_matches(
            true,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
            1,
            24,
            4,
            128,
            16,
            1,
            16_384,
        ));
        assert!(!grouped_qwen35_shape_matches(
            true,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
            1,
            16,
            2,
            256,
            32,
            1,
            16_384,
        ));
    }

    #[test]
    fn exact_d512_shapes_are_default_off_and_auto_is_bounded() {
        let matches = |mode, num_heads, num_kv_heads, query_rows, max_context_len| {
            grouped_d512_shape_matches(
                mode,
                MetalDtype::BFloat16,
                MetalDtype::BFloat16,
                1,
                num_heads,
                num_kv_heads,
                512,
                16,
                query_rows,
                max_context_len,
            )
        };
        for (num_heads, num_kv_heads) in [(8, 1), (16, 1), (16, 2), (32, 4)] {
            assert!(!matches(
                GroupedD512Mode::Disabled,
                num_heads,
                num_kv_heads,
                1,
                3_458
            ));
            assert!(!matches(
                GroupedD512Mode::Auto,
                num_heads,
                num_kv_heads,
                1,
                3_071
            ));
            assert_eq!(
                matches(GroupedD512Mode::Auto, num_heads, num_kv_heads, 1, 3_072),
                (num_heads, num_kv_heads) == (16, 1)
            );
            assert_eq!(
                matches(GroupedD512Mode::Auto, num_heads, num_kv_heads, 1, 16_384),
                (num_heads, num_kv_heads) == (16, 1)
            );
            assert!(!matches(
                GroupedD512Mode::Auto,
                num_heads,
                num_kv_heads,
                1,
                16_385
            ));
            assert!(!matches(
                GroupedD512Mode::Force,
                num_heads,
                num_kv_heads,
                1,
                PARTITION_SIZE
            ));
            assert!(matches(
                GroupedD512Mode::Force,
                num_heads,
                num_kv_heads,
                1,
                PARTITION_SIZE + 1
            ));
            assert!(!matches(
                GroupedD512Mode::Force,
                num_heads,
                num_kv_heads,
                2,
                3_458
            ));
        }
        for (num_heads, num_kv_heads) in [(8, 2), (16, 4), (24, 4), (32, 2)] {
            assert!(!matches(
                GroupedD512Mode::Force,
                num_heads,
                num_kv_heads,
                1,
                8_193
            ));
        }
    }

    #[test]
    fn d512_stripes_account_for_kv_head_parallelism() {
        for (context, expected) in [
            (4_096, [32, 16, 8]),
            (4_097, [64, 32, 16]),
            (8_193, [128, 64, 32]),
        ] {
            for ((q_heads, kv_heads), expected) in
                [(16, 1), (16, 2), (32, 4)].into_iter().zip(expected)
            {
                assert_eq!(
                    grouped_stripe_count(
                        GroupedPagedAttentionKind::D512Direct,
                        context,
                        q_heads,
                        kv_heads,
                    ),
                    expected
                );
            }
        }
    }

    #[test]
    fn d512_wide_stripe_policy_matches_graph_and_planner_boundaries() {
        for (context, expected) in [(90_112, 64), (90_113, 128), (91_795, 128), (112_000, 128)] {
            assert_eq!(
                grouped_d512_resolved_stripe_count(None, context, 16, 2),
                expected
            );
        }
        assert_eq!(
            grouped_d512_resolved_stripe_count(None, 90_113, 8, 1),
            128,
            "Hq8/Hkv1 is unchanged"
        );
        assert_eq!(
            grouped_d512_resolved_stripe_count(None, 90_113, 16, 1),
            128,
            "Hq16/Hkv1 is unchanged"
        );
        assert_eq!(
            grouped_d512_resolved_stripe_count(None, 90_113, 32, 4),
            32,
            "Hkv4 is unchanged"
        );
        assert_eq!(
            grouped_d512_resolved_stripe_count(Some(32), 91_795, 16, 2),
            32,
            "an explicit validated override remains authoritative"
        );
    }

    #[test]
    fn explicit_route_hint_bypasses_environment_and_context_only() {
        let select = |hint, num_heads, num_kv_heads, context| {
            select_grouped_paged_attention(
                MetalDtype::BFloat16,
                MetalDtype::BFloat16,
                1,
                num_heads,
                num_kv_heads,
                512,
                16,
                1,
                context,
                hint,
            )
        };
        assert_eq!(
            select(PagedAttentionRouteHint::ForceD512Staged, 16, 2, 91_765),
            Some(GroupedPagedAttentionKind::D512Direct)
        );
        assert_eq!(
            select(PagedAttentionRouteHint::ForceGeneric, 16, 2, 8_193),
            None
        );
        assert_eq!(
            select(PagedAttentionRouteHint::ForceD512Staged, 16, 4, 91_765),
            None,
            "a route hint must not bypass geometry validation"
        );
    }
}

/// Output from paged attention dispatch
///
/// Contains the output buffer and metadata for creating an MLX array.
pub struct PagedAttentionOutput {
    /// Output buffer [num_seqs, num_heads, head_size]
    pub buffer: Buffer,
    /// Number of sequences
    pub num_seqs: u32,
    /// Number of query heads
    pub num_heads: u32,
    /// Head size
    pub head_size: u32,
    /// Data type
    pub dtype: MetalDtype,
    /// Whether this dispatch actually selected the narrow grouped Qwen3.5
    /// long-context route. This is dispatch metadata (not a hot-path probe),
    /// used by integration tests to distinguish numerical parity from a
    /// silent generic fallback.
    #[doc(hidden)]
    pub used_grouped_qwen35: bool,
    /// Compatibility alias for callers that previously observed the
    /// Gemma-specific D512 route bit.
    #[doc(hidden)]
    pub used_grouped_gemma4: bool,
    /// Whether this dispatch selected the geometry-based grouped D512 route.
    #[doc(hidden)]
    pub used_grouped_d512: bool,
}

impl PagedAttentionOutput {
    /// Get the raw buffer pointer
    pub fn buffer_ptr(&self) -> *mut c_void {
        self.buffer.as_ptr() as *mut c_void
    }

    /// Get the total number of elements
    pub fn num_elements(&self) -> usize {
        (self.num_seqs * self.num_heads * self.head_size) as usize
    }

    /// Get the buffer size in bytes
    pub fn size_bytes(&self) -> usize {
        self.num_elements() * self.dtype.size()
    }

    /// Get output shape as [num_seqs, num_heads, head_size]
    pub fn shape(&self) -> [i64; 3] {
        [
            self.num_seqs as i64,
            self.num_heads as i64,
            self.head_size as i64,
        ]
    }

    /// Copy output data to host memory as f32
    ///
    /// This copies the GPU-private buffer to CPU-accessible memory and converts to f32.
    /// Use this to create an MLX array from the attention output.
    ///
    /// # Performance Note
    /// This involves a GPU->CPU copy. For best performance, consider batching
    /// multiple layers' attention outputs before copying to host.
    ///
    /// # Returns
    /// Vector of f32 values with shape [num_seqs, num_heads, head_size]
    pub fn to_host_f32(&self) -> Result<Vec<f32>, String> {
        let state = MetalState::get()?;
        let size_bytes = self.size_bytes();

        // Create shared (CPU-accessible) buffer
        let shared_buffer = state.device.new_buffer(
            size_bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Blit copy from private to shared

        let command_buffer = state.command_queue.new_command_buffer();
        let blit_encoder = command_buffer.new_blit_command_encoder();

        blit_encoder.copy_from_buffer(&self.buffer, 0, &shared_buffer, 0, size_bytes as u64);
        blit_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
        // Checked before the shared buffer is read, so an aborted copy is
        // never decoded as attention output.
        observe(&command_buffer, "PagedAttentionOutput::to_host_f32")?;

        // Read data from shared buffer
        let num_elements = self.num_elements();
        let mut result = vec![0.0f32; num_elements];

        match self.dtype {
            MetalDtype::Float32 => {
                // Direct copy for f32
                let ptr = shared_buffer.contents() as *const f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), num_elements);
                }
            }
            MetalDtype::Float16 => {
                // Convert f16 to f32
                let ptr = shared_buffer.contents() as *const u16;
                let src = unsafe { std::slice::from_raw_parts(ptr, num_elements) };
                for (dst, &bits) in result.iter_mut().zip(src.iter()) {
                    *dst = f16_to_f32(bits);
                }
            }
            MetalDtype::BFloat16 => {
                // Convert bf16 to f32
                let ptr = shared_buffer.contents() as *const u16;
                let src = unsafe { std::slice::from_raw_parts(ptr, num_elements) };
                for (dst, &bits) in result.iter_mut().zip(src.iter()) {
                    *dst = bf16_to_f32(bits);
                }
            }
            MetalDtype::UChar => {
                // FP8 E4M3 format cannot be correctly converted to f32 without the quantization scales.
                // The raw byte cast (byte as f32) produces completely wrong values because FP8 E4M3
                // has a different bit layout (1 sign, 4 exponent, 3 mantissa bits).
                // Proper dequantization requires: f32_value = fp8_to_f32(byte) / scale
                // where scale was used during quantization.
                return Err(
                    "FP8 (UChar) to f32 conversion not yet implemented. \
                    FP8 dequantization requires k_scale/v_scale values which are not available here. \
                    Use the raw buffer directly or implement to_host_f32_with_scales() for FP8 data."
                        .to_string(),
                );
            }
        }

        Ok(result)
    }

    /// Convert output to an MLX array
    ///
    /// Creates a new MLX array from the attention output data.
    /// This involves copying data from GPU to CPU and back to MLX.
    ///
    /// # Safety
    /// The returned pointer must be managed by the caller (typically wrapped in MxArray).
    ///
    /// # Performance Note
    /// This involves GPU->CPU->GPU copies. For zero-copy integration,
    /// use the raw buffer pointer with MLX's metal operations directly.
    pub unsafe fn to_mlx_array(&self) -> Result<*mut mlx_sys::mlx_array, String> {
        let data = self.to_host_f32()?;
        let shape = self.shape();

        // SAFETY: data and shape are valid pointers with correct length
        let arr = unsafe {
            mlx_sys::mlx_array_from_float32(
                data.as_ptr(),
                shape.as_ptr(),
                3, // ndim
            )
        };

        if arr.is_null() {
            return Err("Failed to create MLX array from attention output".to_string());
        }

        Ok(arr)
    }

    /// Convert output to an MLX array view without a GPU → host roundtrip.
    ///
    /// The returned MLX array retains the underlying `MTL::Buffer` through
    /// `mlx_array_from_metal_buffer_view`, so it remains valid after this
    /// `PagedAttentionOutput` wrapper is dropped.
    ///
    /// # Safety
    /// The returned pointer must be managed by the caller (typically wrapped in
    /// `MxArray`).
    pub unsafe fn to_mlx_array_view(&self) -> Result<*mut mlx_sys::mlx_array, String> {
        let shape = self.shape();
        let dtype_code = match self.dtype {
            MetalDtype::Float32 => 0,
            MetalDtype::Float16 => 2,
            MetalDtype::BFloat16 => 3,
            MetalDtype::UChar => 5,
        };
        let arr = unsafe {
            mlx_sys::mlx_array_from_metal_buffer_view(
                self.buffer_ptr(),
                shape.as_ptr(),
                shape.len(),
                dtype_code,
            )
        };
        if arr.is_null() {
            return Err(
                "Failed to create zero-copy MLX view from paged attention output".to_string(),
            );
        }
        Ok(arr)
    }
}

/// Dispatch paged_attention V1 with raw buffer pointers
///
/// This variant is used when integrating with MLX arrays.
///
/// # Safety
///
/// - queries must be a valid MTLBuffer* with shape [num_seqs, num_heads, head_size]
/// - key_cache and value_cache must be valid cache buffers
/// - block_tables must have shape [num_seqs, max_blocks_per_seq]
/// - context_lens must have shape [num_seqs]
///
/// # Returns
///
/// A new buffer containing the attention output [num_seqs, num_heads, head_size]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_paged_attention_v1_raw(
    queries: &RawBufferInfo,
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_tables: &Buffer,
    context_lens: &Buffer,
    params: &PagedAttentionParams,
    io_dtype: MetalDtype,
    cache_dtype: MetalDtype,
) -> Result<PagedAttentionOutput, String> {
    let state = MetalState::get()?;

    // Allocate output buffer using io_dtype.size() (not the cache dtype) —
    // the kernel writes the io type. For FP8 (cache=uchar) the kernel
    // dequantizes internally and writes io_dtype-sized elements. Using
    // cache_dtype.size() here would under-allocate by 2x for FP8 caches with
    // half/bf16 io.
    let output_element_size = io_dtype.size() as u64;
    let output_size =
        (params.num_seqs * params.num_heads * params.head_size) as u64 * output_element_size;
    let output = state
        .device
        .new_buffer(output_size, metal::MTLResourceOptions::StorageModePrivate);

    // Get V1 pipeline (partition_size = 0)
    let kernel_name = MetalState::paged_attention_v1_kernel_name(
        io_dtype,
        cache_dtype,
        params.head_size,
        params.block_size,
        false, // use_alibi
    );
    let pipeline = state.get_pipeline(&kernel_name)?;

    // Create command buffer and encoder

    let command_buffer = state.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    // Create dummy buffers for exp_sums/max_logits (unused in V1)
    let dummy_float: f32 = 0.0;
    let dummy_buffer = state
        .device
        .new_buffer_with_value(&dummy_float, metal::MTLResourceOptions::StorageModeShared);

    // Convert queries raw pointer to BufferRef
    // SAFETY: Caller guarantees queries.ptr is a valid MTLBuffer*
    let queries_ref: &BufferRef = unsafe { ForeignTypeRef::from_ptr(queries.ptr as *mut _) };

    // Set buffers
    encoder.set_buffer(0, Some(&dummy_buffer), 0); // exp_sums (unused)
    encoder.set_buffer(1, Some(&dummy_buffer), 0); // max_logits (unused)
    encoder.set_buffer(2, Some(&output), 0);
    encoder.set_buffer(3, Some(queries_ref), queries.offset as u64);
    encoder.set_buffer(4, Some(key_cache), 0);
    encoder.set_buffer(5, Some(value_cache), 0);

    // k_scale and v_scale - use params values for FP8 support
    let k_scale_buffer = state.device.new_buffer_with_value(
        &params.k_scale,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let v_scale_buffer = state.device.new_buffer_with_value(
        &params.v_scale,
        metal::MTLResourceOptions::StorageModeShared,
    );
    encoder.set_buffer(6, Some(&k_scale_buffer), 0);
    encoder.set_buffer(7, Some(&v_scale_buffer), 0);

    // Set constant buffers
    let create_int_buffer = |value: i32| {
        state
            .device
            .new_buffer_with_value(&value, metal::MTLResourceOptions::StorageModeShared)
    };

    let create_float_buffer = |value: f32| {
        state
            .device
            .new_buffer_with_value(&value, metal::MTLResourceOptions::StorageModeShared)
    };

    let num_kv_heads_buf = create_int_buffer(params.num_kv_heads as i32);
    let scale_buf = create_float_buffer(params.scale);
    let softcapping_buf = create_float_buffer(params.softcapping);
    let max_num_blocks_buf = create_int_buffer(params.max_num_blocks_per_seq as i32);
    let q_stride_buf = create_int_buffer(params.q_stride);
    let kv_block_stride_buf = create_int_buffer(params.kv_block_stride);
    let kv_head_stride_buf = create_int_buffer(params.kv_head_stride);

    encoder.set_buffer(8, Some(&num_kv_heads_buf), 0);
    encoder.set_buffer(9, Some(&scale_buf), 0);
    encoder.set_buffer(10, Some(&softcapping_buf), 0);
    encoder.set_buffer(11, Some(block_tables), 0);
    encoder.set_buffer(12, Some(context_lens), 0);
    encoder.set_buffer(13, Some(&max_num_blocks_buf), 0);

    // alibi_slopes (unused, set to dummy)
    encoder.set_buffer(14, Some(&dummy_buffer), 0);

    encoder.set_buffer(15, Some(&q_stride_buf), 0);
    encoder.set_buffer(16, Some(&kv_block_stride_buf), 0);
    encoder.set_buffer(17, Some(&kv_head_stride_buf), 0);

    // sliding-window mask. 0 = full context (default).
    let sliding_window_buf = create_int_buffer(params.sliding_window);
    encoder.set_buffer(18, Some(&sliding_window_buf), 0);

    // Calculate threadgroup memory size.
    //
    // The kernel reuses `shared_mem` for two phases:
    //   1. QK softmax — `logits[max_seq_len]` f32s + `red_smem[2*NUM_WARPS]` f32s.
    //   2. V output reduction — reinterprets `shared_mem` as
    //      `out_smem[mid * HEAD_SIZE]` f32s where `mid = NUM_WARPS/2 = 4`
    //      at the first reduction step.
    //
    // The original allocation only sized phase (1), which silently
    // truncates phase (2) when `max_seq_len * 4 < (NUM_WARPS/2) * HEAD_SIZE * 4`.
    // In production max_seq_len always dominates (8192 × 4 = 32 KiB ≫
    // 4 × 128 × 4 = 2 KiB), so this never surfaced. Small-context decode
    // (e.g. 24 tokens × HEAD_SIZE 64 → 96 < 1024) hits the truncation:
    // upper warps' writes during the reduction tree
    // land at undefined positions — empirically, only some lanes' rows
    // get reduced correctly, so the gathered output equals the kernel-warp-0
    // contribution for those rows and the multi-block sum gets dropped
    // for the remainder. Take the per-phase max of (1) and (2) plus the
    // dedicated red_smem scratchpad.
    const NUM_WARPS_FOR_REDUCE: usize = 8; // NUM_THREADS (256) / NUM_SIMD_LANES (32)
    let logits_bytes = params.max_seq_len as usize * std::mem::size_of::<f32>();
    let v_reduce_bytes =
        (NUM_WARPS_FOR_REDUCE / 2) * params.head_size as usize * std::mem::size_of::<f32>();
    let red_smem_bytes = 2 * NUM_WARPS_FOR_REDUCE * std::mem::size_of::<f32>();
    let threadgroup_mem_size = logits_bytes.max(v_reduce_bytes) + red_smem_bytes;
    encoder.set_threadgroup_memory_length(0, threadgroup_mem_size as u64);

    // Dispatch: (num_heads, num_seqs, 1) threadgroups, 256 threads each
    let threads_per_threadgroup = MTLSize::new(256, 1, 1);
    let threadgroups = MTLSize::new(
        params.num_heads as u64,
        params.num_seqs as u64,
        1, // No partitioning in V1
    );

    encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    observe(&command_buffer, "dispatch_paged_attention_v1_raw")?;

    // Output dtype is the io_dtype the kernel was templated on (not the
    // cache dtype). For FP8 (cache=uchar) the kernel dequantizes internally
    // and writes io_dtype-sized elements.
    Ok(PagedAttentionOutput {
        buffer: output,
        num_seqs: params.num_seqs,
        num_heads: params.num_heads,
        head_size: params.head_size,
        dtype: io_dtype,
        used_grouped_qwen35: false,
        used_grouped_gemma4: false,
        used_grouped_d512: false,
    })
}

/// Dispatch paged_attention V2 with raw buffer pointers (for long sequences)
///
/// This variant uses partitioning for sequences longer than PARTITION_SIZE tokens.
///
/// # Safety
///
/// Same requirements as V1.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_paged_attention_v2_raw(
    queries: &RawBufferInfo,
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_tables: &Buffer,
    context_lens: &Buffer,
    params: &PagedAttentionParams,
    io_dtype: MetalDtype,
    cache_dtype: MetalDtype,
) -> Result<PagedAttentionOutput, String> {
    unsafe {
        dispatch_paged_attention_v2_raw_with_route(
            queries,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            params,
            io_dtype,
            cache_dtype,
            PagedAttentionRouteHint::Auto,
        )
    }
}

/// Route-hinted sibling of [`dispatch_paged_attention_v2_raw`].
///
/// # Safety
///
/// Same requirements as V1.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_paged_attention_v2_raw_with_route(
    queries: &RawBufferInfo,
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_tables: &Buffer,
    context_lens: &Buffer,
    params: &PagedAttentionParams,
    io_dtype: MetalDtype,
    cache_dtype: MetalDtype,
    route_hint: PagedAttentionRouteHint,
) -> Result<PagedAttentionOutput, String> {
    let state = MetalState::get()?;

    // Allocate output buffer sized by io_dtype — kernel writes io_dtype
    // elements even for FP8 caches (dequantization happens inside the
    // kernel). Using cache_dtype.size() would under-allocate by 2x for FP8
    // with half/bf16 io.
    let output_element_size = io_dtype.size() as u64;
    let output_size =
        (params.num_seqs * params.num_heads * params.head_size) as u64 * output_element_size;
    let output = state
        .device
        .new_buffer(output_size, metal::MTLResourceOptions::StorageModePrivate);

    let grouped_kind = select_grouped_paged_attention(
        io_dtype,
        cache_dtype,
        params.num_seqs,
        params.num_heads,
        params.num_kv_heads,
        params.head_size,
        params.block_size,
        1,
        params.max_seq_len,
        route_hint,
    );
    let use_grouped = grouped_kind.is_some_and(|kind| {
        grouped_pipelines_supported(state, kind, params.num_heads, params.num_kv_heads)
    });
    let max_num_partitions = if let Some(kind) = grouped_kind.filter(|_| use_grouped) {
        grouped_stripe_count(
            kind,
            params.max_seq_len,
            params.num_heads,
            params.num_kv_heads,
        )
    } else {
        params.max_seq_len.div_ceil(PARTITION_SIZE)
    };

    // Allocate temporary buffers. `tmp_out` holds partition outputs in the
    // io dtype (NOT the cache dtype) — the reduce kernel reads io-typed
    // elements.
    let exp_sums_size = (params.num_seqs * params.num_heads * max_num_partitions) as usize
        * std::mem::size_of::<f32>();
    let max_logits_size = exp_sums_size;
    let tmp_out_size = (params.num_seqs * params.num_heads * max_num_partitions * params.head_size)
        as usize
        * io_dtype.size();

    let exp_sums = state.device.new_buffer(
        exp_sums_size as u64,
        metal::MTLResourceOptions::StorageModePrivate,
    );
    let max_logits = state.device.new_buffer(
        max_logits_size as u64,
        metal::MTLResourceOptions::StorageModePrivate,
    );
    let tmp_out = state.device.new_buffer(
        tmp_out_size as u64,
        metal::MTLResourceOptions::StorageModePrivate,
    );

    // Convert queries raw pointer to BufferRef
    // SAFETY: Caller guarantees queries.ptr is a valid MTLBuffer*
    let queries_ref: &BufferRef = unsafe { ForeignTypeRef::from_ptr(queries.ptr as *mut _) };

    // Phase 1: Compute partitioned attention
    {
        let kernel_name = if let Some(kind) = grouped_kind.filter(|_| use_grouped) {
            grouped_kernel_name(kind).to_string()
        } else {
            MetalState::paged_attention_v2_kernel_name(
                io_dtype,
                cache_dtype,
                params.head_size,
                params.block_size,
                false,
            )
        };
        let pipeline = state.get_pipeline(&kernel_name)?;

        let command_buffer = state.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);

        // Set buffers
        encoder.set_buffer(0, Some(&exp_sums), 0);
        encoder.set_buffer(1, Some(&max_logits), 0);
        encoder.set_buffer(2, Some(&tmp_out), 0);
        encoder.set_buffer(3, Some(queries_ref), queries.offset as u64);
        encoder.set_buffer(4, Some(key_cache), 0);
        encoder.set_buffer(5, Some(value_cache), 0);

        // k_scale and v_scale - use params values for FP8 support
        let k_scale_buffer = state.device.new_buffer_with_value(
            &params.k_scale,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let v_scale_buffer = state.device.new_buffer_with_value(
            &params.v_scale,
            metal::MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(6, Some(&k_scale_buffer), 0);
        encoder.set_buffer(7, Some(&v_scale_buffer), 0);

        // Set constant buffers
        let create_int_buffer = |value: i32| {
            state
                .device
                .new_buffer_with_value(&value, metal::MTLResourceOptions::StorageModeShared)
        };

        let create_float_buffer = |value: f32| {
            state
                .device
                .new_buffer_with_value(&value, metal::MTLResourceOptions::StorageModeShared)
        };

        let num_kv_heads_buf = create_int_buffer(params.num_kv_heads as i32);
        let scale_buf = create_float_buffer(params.scale);
        let softcapping_buf = create_float_buffer(params.softcapping);
        let max_num_blocks_buf = create_int_buffer(params.max_num_blocks_per_seq as i32);
        let q_stride_buf = create_int_buffer(params.q_stride);
        let kv_block_stride_buf = create_int_buffer(params.kv_block_stride);
        let kv_head_stride_buf = create_int_buffer(params.kv_head_stride);

        let dummy_float: f32 = 0.0;
        let dummy_buffer = state
            .device
            .new_buffer_with_value(&dummy_float, metal::MTLResourceOptions::StorageModeShared);

        encoder.set_buffer(8, Some(&num_kv_heads_buf), 0);
        encoder.set_buffer(9, Some(&scale_buf), 0);
        encoder.set_buffer(10, Some(&softcapping_buf), 0);
        encoder.set_buffer(11, Some(block_tables), 0);
        encoder.set_buffer(12, Some(context_lens), 0);
        encoder.set_buffer(13, Some(&max_num_blocks_buf), 0);
        encoder.set_buffer(14, Some(&dummy_buffer), 0); // alibi_slopes
        encoder.set_buffer(15, Some(&q_stride_buf), 0);
        encoder.set_buffer(16, Some(&kv_block_stride_buf), 0);
        encoder.set_buffer(17, Some(&kv_head_stride_buf), 0);

        // sliding-window mask. 0 = full context (default).
        let sliding_window_buf = create_int_buffer(params.sliding_window);
        encoder.set_buffer(18, Some(&sliding_window_buf), 0);

        if !use_grouped {
            // Threadgroup memory — same dual-purpose layout as V1 (see
            // `dispatch_paged_attention_v1_raw`). V2 partitions context into
            // PARTITION_SIZE chunks, so logits is sized by PARTITION_SIZE
            // rather than max_seq_len. V-reduction still needs
            // `(NUM_WARPS/2) * HEAD_SIZE` f32s; for HEAD_SIZE >= 256 that
            // exceeds PARTITION_SIZE * 4 bytes (4*256*4 = 4096 > 512*4 =
            // 2048), so the per-phase max keeps the largest models safe.
            const NUM_WARPS_V2: usize = 8;
            let logits_bytes_v2 = PARTITION_SIZE as usize * std::mem::size_of::<f32>();
            let v_reduce_bytes_v2 =
                (NUM_WARPS_V2 / 2) * params.head_size as usize * std::mem::size_of::<f32>();
            let red_smem_bytes_v2 = 2 * NUM_WARPS_V2 * std::mem::size_of::<f32>();
            let threadgroup_mem_size = logits_bytes_v2.max(v_reduce_bytes_v2) + red_smem_bytes_v2;
            encoder.set_threadgroup_memory_length(0, threadgroup_mem_size as u64);
        }

        let gqa_factor = params.num_heads / params.num_kv_heads;
        let threads_per_threadgroup = if use_grouped {
            MTLSize::new(32, gqa_factor as u64, 1)
        } else {
            MTLSize::new(256, 1, 1)
        };
        let threadgroups = if use_grouped {
            MTLSize::new(
                params.num_kv_heads as u64,
                params.num_seqs as u64,
                max_num_partitions as u64,
            )
        } else {
            MTLSize::new(
                params.num_heads as u64,
                params.num_seqs as u64,
                max_num_partitions as u64,
            )
        };

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
        observe(
            &command_buffer,
            "dispatch_paged_attention_v2_raw (partition)",
        )?;
    }

    // Phase 2: Reduce partitions
    {
        let kernel_name = if let Some(kind) = grouped_kind.filter(|_| use_grouped) {
            grouped_reduce_kernel_name(kind).to_string()
        } else {
            MetalState::paged_attention_v2_reduce_kernel_name(io_dtype, params.head_size)
        };
        let pipeline = state.get_pipeline(&kernel_name)?;

        let command_buffer = state.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_buffer(0, Some(&output), 0);
        encoder.set_buffer(1, Some(&exp_sums), 0);
        encoder.set_buffer(2, Some(&max_logits), 0);
        encoder.set_buffer(3, Some(&tmp_out), 0);
        encoder.set_buffer(4, Some(context_lens), 0);

        let max_num_partitions_buf = state.device.new_buffer_with_value(
            &(max_num_partitions as i32),
            metal::MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(5, Some(&max_num_partitions_buf), 0);

        if !use_grouped {
            // Threadgroup memory for the generic contiguous-partition reduce.
            let threadgroup_mem_size =
                2 * (max_num_partitions as usize) * std::mem::size_of::<f32>();
            encoder.set_threadgroup_memory_length(0, threadgroup_mem_size as u64);
        }

        // Dispatch: (num_heads, num_seqs, 1)
        let threads_per_threadgroup = MTLSize::new(if use_grouped { 1024 } else { 256 }, 1, 1);
        let threadgroups = MTLSize::new(params.num_heads as u64, params.num_seqs as u64, 1);

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
        observe(&command_buffer, "dispatch_paged_attention_v2_raw (reduce)")?;
    }

    // Output dtype is io_dtype (kernel writes io-typed elements). FP8
    // caches dequantize internally — output is still io_dtype.
    Ok(PagedAttentionOutput {
        buffer: output,
        num_seqs: params.num_seqs,
        num_heads: params.num_heads,
        head_size: params.head_size,
        dtype: io_dtype,
        used_grouped_qwen35: use_grouped
            && grouped_kind == Some(GroupedPagedAttentionKind::Qwen35D256),
        used_grouped_gemma4: use_grouped
            && grouped_kind == Some(GroupedPagedAttentionKind::D512Direct),
        used_grouped_d512: use_grouped
            && grouped_kind == Some(GroupedPagedAttentionKind::D512Direct),
    })
}

/// Automatically dispatch V1 or V2 based on max context length
///
/// Uses V1 for short sequences (< PARTITION_SIZE tokens), V2 for longer.
///
/// # Safety
/// - `queries` must contain a valid MTLBuffer pointer
/// - `key_cache` and `value_cache` must be valid cache buffers
/// - `block_tables` must have shape [num_seqs, max_blocks_per_seq]
/// - `context_lens` must have shape [num_seqs]
/// - All buffers must remain valid until the kernel completes
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_paged_attention_auto(
    queries: &RawBufferInfo,
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_tables: &Buffer,
    context_lens: &Buffer,
    max_context_len: u32,
    params: &PagedAttentionParams,
    io_dtype: MetalDtype,
    cache_dtype: MetalDtype,
) -> Result<PagedAttentionOutput, String> {
    unsafe {
        dispatch_paged_attention_auto_with_route(
            queries,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            max_context_len,
            params,
            io_dtype,
            cache_dtype,
            PagedAttentionRouteHint::Auto,
        )
    }
}

/// Route-hinted sibling of [`dispatch_paged_attention_auto`].
///
/// # Safety
/// Same requirements as [`dispatch_paged_attention_auto`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_paged_attention_auto_with_route(
    queries: &RawBufferInfo,
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_tables: &Buffer,
    context_lens: &Buffer,
    max_context_len: u32,
    params: &PagedAttentionParams,
    io_dtype: MetalDtype,
    cache_dtype: MetalDtype,
    route_hint: PagedAttentionRouteHint,
) -> Result<PagedAttentionOutput, String> {
    if max_context_len <= PARTITION_SIZE {
        // SAFETY: Caller guarantees all buffer pointers are valid
        unsafe {
            dispatch_paged_attention_v1_raw(
                queries,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                params,
                io_dtype,
                cache_dtype,
            )
        }
    } else {
        // SAFETY: Caller guarantees all buffer pointers are valid
        unsafe {
            dispatch_paged_attention_v2_raw_with_route(
                queries,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                params,
                io_dtype,
                cache_dtype,
                route_hint,
            )
        }
    }
}

/// Parameters for the varlen (multi-row) paged_attention dispatch.
///
/// For Multi-Token Prediction the verify pass needs D+1 query tokens per
/// sequence, so the kernel grid is `(num_q_heads, total_queries,
/// num_partitions)` and the query tensor is a flat ragged
/// `[total_queries, num_q_heads, head_size]` layout.
///
/// `cu_seqlens_q` is the standard FlashAttention-style cumulative query
/// length array: `cu_seqlens_q[0] = 0`, `cu_seqlens_q[i+1] -
/// cu_seqlens_q[i] = q_len_i`, and `cu_seqlens_q[num_seqs] =
/// total_queries`. Every query token discovers its source sequence by
/// binary-searching this array on the GPU.
pub struct PagedAttentionVarlenParams {
    /// Number of sequences in the batch (length of context_lens).
    pub num_seqs: u32,
    /// Total number of query tokens across all sequences (equals
    /// `cu_seqlens_q[num_seqs]`). Determines the y-dimension of the
    /// dispatch grid.
    pub total_queries: u32,
    /// Number of query heads.
    pub num_heads: u32,
    /// Number of KV heads.
    pub num_kv_heads: u32,
    /// Size of each head.
    pub head_size: u32,
    /// Block size for paged attention.
    pub block_size: u32,
    /// Max effective_context_len across all queries. Used to size V2's
    /// `max_num_partitions` and to choose V1 vs V2.
    pub max_seq_len: u32,
    /// Maximum number of blocks per sequence.
    pub max_num_blocks_per_seq: u32,
    /// Attention scale factor (1/sqrt(head_size)).
    pub scale: f32,
    /// Softcapping value (1.0 = disabled).
    pub softcapping: f32,
    /// Query stride (num_heads * head_size — same as the single-row
    /// kernel, since Q is still row-major in [num_heads, head_size]).
    pub q_stride: i32,
    /// KV block stride.
    pub kv_block_stride: i32,
    /// KV head stride.
    pub kv_head_stride: i32,
    /// K scale for FP8 quantization (1.0 for non-FP8).
    pub k_scale: f32,
    /// V scale for FP8 quantization (1.0 for non-FP8).
    pub v_scale: f32,
    /// Sliding-window mask. Same semantics as the single-row kernel,
    /// referenced to `effective_context_len` per query token.
    pub sliding_window: i32,
}

impl Default for PagedAttentionVarlenParams {
    fn default() -> Self {
        Self {
            num_seqs: 0,
            total_queries: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_size: 128,
            block_size: 16,
            max_seq_len: 0,
            max_num_blocks_per_seq: 0,
            scale: 1.0,
            softcapping: 1.0,
            q_stride: 0,
            kv_block_stride: 0,
            kv_head_stride: 0,
            k_scale: 1.0,
            v_scale: 1.0,
            sliding_window: 0,
        }
    }
}

/// Output of the varlen dispatch — shape `[total_queries, num_heads,
/// head_size]`. Reuses `PagedAttentionOutput` because the only semantic
/// change is that `num_seqs` actually carries `total_queries`; downstream
/// consumers only ever multiply the three fields back together to size
/// blits.
pub type PagedAttentionVarlenOutput = PagedAttentionOutput;

/// Dispatch the varlen V1 paged_attention kernel (no partitioning, used
/// when `max_seq_len <= PARTITION_SIZE`).
///
/// # Safety
/// - `queries` must be a valid MTLBuffer* with shape
///   `[total_queries, num_heads, head_size]` (ragged Q layout).
/// - `cu_seqlens_q` must hold `num_seqs + 1` int32 cumulative counts.
/// - Same K/V/block_tables/context_lens layout as the single-row
///   dispatchers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_paged_attention_varlen_v1_raw(
    queries: &RawBufferInfo,
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_tables: &Buffer,
    context_lens: &Buffer,
    cu_seqlens_q: &Buffer,
    params: &PagedAttentionVarlenParams,
    io_dtype: MetalDtype,
    cache_dtype: MetalDtype,
) -> Result<PagedAttentionVarlenOutput, String> {
    let state = MetalState::get()?;

    // Output sized by total_queries — every query token gets its own
    // [num_heads, head_size] slab. Allocated in io_dtype regardless of
    // cache_dtype (FP8 dequantizes internally).
    let output_element_size = io_dtype.size() as u64;
    let output_size = (params.total_queries as u64)
        * (params.num_heads as u64)
        * (params.head_size as u64)
        * output_element_size;
    let output = state
        .device
        .new_buffer(output_size, metal::MTLResourceOptions::StorageModePrivate);

    let kernel_name = MetalState::paged_attention_varlen_v1_kernel_name(
        io_dtype,
        cache_dtype,
        params.head_size,
        params.block_size,
        false, // use_alibi — not exercised by varlen
    );
    let pipeline = state.get_pipeline(&kernel_name)?;

    let command_buffer = state.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    let dummy_float: f32 = 0.0;
    let dummy_buffer = state
        .device
        .new_buffer_with_value(&dummy_float, metal::MTLResourceOptions::StorageModeShared);

    // SAFETY: caller guarantees queries.ptr is a valid MTLBuffer*.
    let queries_ref: &BufferRef = unsafe { ForeignTypeRef::from_ptr(queries.ptr as *mut _) };

    encoder.set_buffer(0, Some(&dummy_buffer), 0); // exp_sums (unused in V1)
    encoder.set_buffer(1, Some(&dummy_buffer), 0); // max_logits (unused in V1)
    encoder.set_buffer(2, Some(&output), 0);
    encoder.set_buffer(3, Some(queries_ref), queries.offset as u64);
    encoder.set_buffer(4, Some(key_cache), 0);
    encoder.set_buffer(5, Some(value_cache), 0);

    let k_scale_buffer = state.device.new_buffer_with_value(
        &params.k_scale,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let v_scale_buffer = state.device.new_buffer_with_value(
        &params.v_scale,
        metal::MTLResourceOptions::StorageModeShared,
    );
    encoder.set_buffer(6, Some(&k_scale_buffer), 0);
    encoder.set_buffer(7, Some(&v_scale_buffer), 0);

    let create_int_buffer = |value: i32| {
        state
            .device
            .new_buffer_with_value(&value, metal::MTLResourceOptions::StorageModeShared)
    };
    let create_float_buffer = |value: f32| {
        state
            .device
            .new_buffer_with_value(&value, metal::MTLResourceOptions::StorageModeShared)
    };

    let num_kv_heads_buf = create_int_buffer(params.num_kv_heads as i32);
    let scale_buf = create_float_buffer(params.scale);
    let softcapping_buf = create_float_buffer(params.softcapping);
    let max_num_blocks_buf = create_int_buffer(params.max_num_blocks_per_seq as i32);
    let q_stride_buf = create_int_buffer(params.q_stride);
    let kv_block_stride_buf = create_int_buffer(params.kv_block_stride);
    let kv_head_stride_buf = create_int_buffer(params.kv_head_stride);

    encoder.set_buffer(8, Some(&num_kv_heads_buf), 0);
    encoder.set_buffer(9, Some(&scale_buf), 0);
    encoder.set_buffer(10, Some(&softcapping_buf), 0);
    encoder.set_buffer(11, Some(block_tables), 0);
    encoder.set_buffer(12, Some(context_lens), 0);
    encoder.set_buffer(13, Some(&max_num_blocks_buf), 0);
    encoder.set_buffer(14, Some(&dummy_buffer), 0); // alibi_slopes
    encoder.set_buffer(15, Some(&q_stride_buf), 0);
    encoder.set_buffer(16, Some(&kv_block_stride_buf), 0);
    encoder.set_buffer(17, Some(&kv_head_stride_buf), 0);

    let sliding_window_buf = create_int_buffer(params.sliding_window);
    encoder.set_buffer(18, Some(&sliding_window_buf), 0);

    // New buffers for varlen: cu_seqlens_q + num_seqs.
    encoder.set_buffer(19, Some(cu_seqlens_q), 0);
    let num_seqs_buf = create_int_buffer(params.num_seqs as i32);
    encoder.set_buffer(20, Some(&num_seqs_buf), 0);

    // Threadgroup memory layout matches the single-row V1 dispatcher.
    // The `logits` buffer is still sized by `max_seq_len` (the cross-batch
    // worst case) — individual query tokens may use fewer slots when
    // their effective_context_len is smaller, but the allocation must
    // cover the longest.
    const NUM_WARPS_FOR_REDUCE: usize = 8;
    let logits_bytes = params.max_seq_len as usize * std::mem::size_of::<f32>();
    let v_reduce_bytes =
        (NUM_WARPS_FOR_REDUCE / 2) * params.head_size as usize * std::mem::size_of::<f32>();
    let red_smem_bytes = 2 * NUM_WARPS_FOR_REDUCE * std::mem::size_of::<f32>();
    let threadgroup_mem_size = logits_bytes.max(v_reduce_bytes) + red_smem_bytes;
    encoder.set_threadgroup_memory_length(0, threadgroup_mem_size as u64);

    // Grid: (num_heads, total_queries, 1) — one threadgroup per
    // (head, query_token). The y-dim is the only structural change vs.
    // the single-row dispatcher.
    let threads_per_threadgroup = MTLSize::new(256, 1, 1);
    let threadgroups = MTLSize::new(params.num_heads as u64, params.total_queries as u64, 1);

    encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    observe(&command_buffer, "dispatch_paged_attention_varlen_v1_raw")?;

    Ok(PagedAttentionVarlenOutput {
        buffer: output,
        // `num_seqs` in PagedAttentionOutput is overloaded to mean "the
        // leading dim of the output". For varlen that's total_queries.
        num_seqs: params.total_queries,
        num_heads: params.num_heads,
        head_size: params.head_size,
        dtype: io_dtype,
        used_grouped_qwen35: false,
        used_grouped_gemma4: false,
        used_grouped_d512: false,
    })
}

/// Dispatch the varlen V2 paged_attention kernel (with partitioning, used
/// when `max_seq_len > PARTITION_SIZE`).
///
/// # Safety
/// Same as `dispatch_paged_attention_varlen_v1_raw`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_paged_attention_varlen_v2_raw(
    queries: &RawBufferInfo,
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_tables: &Buffer,
    context_lens: &Buffer,
    cu_seqlens_q: &Buffer,
    params: &PagedAttentionVarlenParams,
    io_dtype: MetalDtype,
    cache_dtype: MetalDtype,
) -> Result<PagedAttentionVarlenOutput, String> {
    let state = MetalState::get()?;

    let output_element_size = io_dtype.size() as u64;
    let output_size = (params.total_queries as u64)
        * (params.num_heads as u64)
        * (params.head_size as u64)
        * output_element_size;
    let output = state
        .device
        .new_buffer(output_size, metal::MTLResourceOptions::StorageModePrivate);

    let grouped_kind = (params.total_queries == 2)
        .then(|| {
            select_grouped_paged_attention(
                io_dtype,
                cache_dtype,
                params.num_seqs,
                params.num_heads,
                params.num_kv_heads,
                params.head_size,
                params.block_size,
                params.total_queries,
                params.max_seq_len,
                PagedAttentionRouteHint::Auto,
            )
        })
        .flatten();
    let use_grouped = grouped_kind.is_some_and(|kind| {
        grouped_pipelines_supported(state, kind, params.num_heads, params.num_kv_heads)
    });

    // Sized off the worst-case effective_context_len (the caller's
    // `max_seq_len`), so every query token fits in the allocated grid.
    // Per-token short-context queries simply leave some partition slots
    // empty — the reduce kernel skips them via the
    // `effective_context_len`-derived `num_partitions` it computes
    // independently.
    let max_num_partitions = if let Some(kind) = grouped_kind.filter(|_| use_grouped) {
        grouped_stripe_count(
            kind,
            params.max_seq_len,
            params.num_heads,
            params.num_kv_heads,
        )
    } else {
        params.max_seq_len.div_ceil(PARTITION_SIZE)
    };

    // Note: the V2 main kernel writes exp_sums / max_logits / tmp_out
    // indexed by q_token_idx (NOT seq_idx), so size by total_queries.
    let exp_sums_size = (params.total_queries * params.num_heads * max_num_partitions) as usize
        * std::mem::size_of::<f32>();
    let max_logits_size = exp_sums_size;
    let tmp_out_size =
        (params.total_queries * params.num_heads * max_num_partitions * params.head_size) as usize
            * io_dtype.size();

    let exp_sums = state.device.new_buffer(
        exp_sums_size as u64,
        metal::MTLResourceOptions::StorageModePrivate,
    );
    let max_logits = state.device.new_buffer(
        max_logits_size as u64,
        metal::MTLResourceOptions::StorageModePrivate,
    );
    let tmp_out = state.device.new_buffer(
        tmp_out_size as u64,
        metal::MTLResourceOptions::StorageModePrivate,
    );

    // SAFETY: caller guarantees queries.ptr is a valid MTLBuffer*.
    let queries_ref: &BufferRef = unsafe { ForeignTypeRef::from_ptr(queries.ptr as *mut _) };

    // Phase 1: partitioned varlen attention.
    {
        let kernel_name = if let Some(kind) = grouped_kind.filter(|_| use_grouped) {
            grouped_kernel_name(kind).to_string()
        } else {
            MetalState::paged_attention_varlen_v2_kernel_name(
                io_dtype,
                cache_dtype,
                params.head_size,
                params.block_size,
                false,
            )
        };
        let pipeline = state.get_pipeline(&kernel_name)?;

        let command_buffer = state.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_buffer(0, Some(&exp_sums), 0);
        encoder.set_buffer(1, Some(&max_logits), 0);
        encoder.set_buffer(2, Some(&tmp_out), 0);
        encoder.set_buffer(3, Some(queries_ref), queries.offset as u64);
        encoder.set_buffer(4, Some(key_cache), 0);
        encoder.set_buffer(5, Some(value_cache), 0);

        let k_scale_buffer = state.device.new_buffer_with_value(
            &params.k_scale,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let v_scale_buffer = state.device.new_buffer_with_value(
            &params.v_scale,
            metal::MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(6, Some(&k_scale_buffer), 0);
        encoder.set_buffer(7, Some(&v_scale_buffer), 0);

        let create_int_buffer = |value: i32| {
            state
                .device
                .new_buffer_with_value(&value, metal::MTLResourceOptions::StorageModeShared)
        };
        let create_float_buffer = |value: f32| {
            state
                .device
                .new_buffer_with_value(&value, metal::MTLResourceOptions::StorageModeShared)
        };

        let num_kv_heads_buf = create_int_buffer(params.num_kv_heads as i32);
        let scale_buf = create_float_buffer(params.scale);
        let softcapping_buf = create_float_buffer(params.softcapping);
        let max_num_blocks_buf = create_int_buffer(params.max_num_blocks_per_seq as i32);
        let q_stride_buf = create_int_buffer(params.q_stride);
        let kv_block_stride_buf = create_int_buffer(params.kv_block_stride);
        let kv_head_stride_buf = create_int_buffer(params.kv_head_stride);

        let dummy_float: f32 = 0.0;
        let dummy_buffer = state
            .device
            .new_buffer_with_value(&dummy_float, metal::MTLResourceOptions::StorageModeShared);

        encoder.set_buffer(8, Some(&num_kv_heads_buf), 0);
        encoder.set_buffer(9, Some(&scale_buf), 0);
        encoder.set_buffer(10, Some(&softcapping_buf), 0);
        encoder.set_buffer(11, Some(block_tables), 0);
        encoder.set_buffer(12, Some(context_lens), 0);
        encoder.set_buffer(13, Some(&max_num_blocks_buf), 0);
        encoder.set_buffer(14, Some(&dummy_buffer), 0); // alibi_slopes
        encoder.set_buffer(15, Some(&q_stride_buf), 0);
        encoder.set_buffer(16, Some(&kv_block_stride_buf), 0);
        encoder.set_buffer(17, Some(&kv_head_stride_buf), 0);

        let sliding_window_buf = create_int_buffer(params.sliding_window);
        encoder.set_buffer(18, Some(&sliding_window_buf), 0);

        encoder.set_buffer(19, Some(cu_seqlens_q), 0);
        let num_seqs_buf = create_int_buffer(params.num_seqs as i32);
        encoder.set_buffer(20, Some(&num_seqs_buf), 0);

        if !use_grouped {
            const NUM_WARPS_V2: usize = 8;
            let logits_bytes_v2 = PARTITION_SIZE as usize * std::mem::size_of::<f32>();
            let v_reduce_bytes_v2 =
                (NUM_WARPS_V2 / 2) * params.head_size as usize * std::mem::size_of::<f32>();
            let red_smem_bytes_v2 = 2 * NUM_WARPS_V2 * std::mem::size_of::<f32>();
            let threadgroup_mem_size = logits_bytes_v2.max(v_reduce_bytes_v2) + red_smem_bytes_v2;
            encoder.set_threadgroup_memory_length(0, threadgroup_mem_size as u64);
        }

        let gqa_factor = params.num_heads / params.num_kv_heads;
        let threads_per_threadgroup = if use_grouped {
            MTLSize::new(32, gqa_factor as u64, 1)
        } else {
            MTLSize::new(256, 1, 1)
        };
        let threadgroups = if use_grouped {
            MTLSize::new(
                params.num_kv_heads as u64,
                params.total_queries as u64,
                max_num_partitions as u64,
            )
        } else {
            MTLSize::new(
                params.num_heads as u64,
                params.total_queries as u64,
                max_num_partitions as u64,
            )
        };

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
        observe(
            &command_buffer,
            "dispatch_paged_attention_varlen_v2_raw (partition)",
        )?;
    }

    // Phase 2: reduce partitions. The varlen reduce kernel walks the
    // ragged query layout the same way as the main kernel (y =
    // q_token_idx, binary-search to seq_idx).
    {
        let kernel_name = if let Some(kind) = grouped_kind.filter(|_| use_grouped) {
            grouped_reduce_kernel_name(kind).to_string()
        } else {
            MetalState::paged_attention_varlen_v2_reduce_kernel_name(io_dtype, params.head_size)
        };
        let pipeline = state.get_pipeline(&kernel_name)?;

        let command_buffer = state.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_buffer(0, Some(&output), 0);
        encoder.set_buffer(1, Some(&exp_sums), 0);
        encoder.set_buffer(2, Some(&max_logits), 0);
        encoder.set_buffer(3, Some(&tmp_out), 0);
        encoder.set_buffer(4, Some(context_lens), 0);

        let max_num_partitions_buf = state.device.new_buffer_with_value(
            &(max_num_partitions as i32),
            metal::MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(5, Some(&max_num_partitions_buf), 0);

        encoder.set_buffer(6, Some(cu_seqlens_q), 0);
        let num_seqs_buf = state.device.new_buffer_with_value(
            &(params.num_seqs as i32),
            metal::MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(7, Some(&num_seqs_buf), 0);

        if !use_grouped {
            let threadgroup_mem_size =
                2 * (max_num_partitions as usize) * std::mem::size_of::<f32>();
            encoder.set_threadgroup_memory_length(0, threadgroup_mem_size as u64);
        }

        let threads_per_threadgroup = MTLSize::new(if use_grouped { 1024 } else { 256 }, 1, 1);
        let threadgroups = MTLSize::new(params.num_heads as u64, params.total_queries as u64, 1);

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
        observe(
            &command_buffer,
            "dispatch_paged_attention_varlen_v2_raw (reduce)",
        )?;
    }

    Ok(PagedAttentionVarlenOutput {
        buffer: output,
        num_seqs: params.total_queries,
        num_heads: params.num_heads,
        head_size: params.head_size,
        dtype: io_dtype,
        used_grouped_qwen35: use_grouped
            && grouped_kind == Some(GroupedPagedAttentionKind::Qwen35D256),
        used_grouped_gemma4: use_grouped
            && grouped_kind == Some(GroupedPagedAttentionKind::D512Direct),
        used_grouped_d512: use_grouped
            && grouped_kind == Some(GroupedPagedAttentionKind::D512Direct),
    })
}

/// Automatically pick V1 or V2 for varlen dispatch based on `max_seq_len`.
///
/// Threshold matches the single-row auto-dispatcher (PARTITION_SIZE=512)
/// so the V1/V2 cutover stays consistent across the codebase.
///
/// # Safety
/// Same as the underlying `dispatch_paged_attention_varlen_*_raw` calls.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_paged_attention_varlen_auto(
    queries: &RawBufferInfo,
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_tables: &Buffer,
    context_lens: &Buffer,
    cu_seqlens_q: &Buffer,
    max_context_len: u32,
    params: &PagedAttentionVarlenParams,
    io_dtype: MetalDtype,
    cache_dtype: MetalDtype,
) -> Result<PagedAttentionVarlenOutput, String> {
    if max_context_len <= PARTITION_SIZE {
        // SAFETY: caller guarantees all buffer pointers are valid.
        unsafe {
            dispatch_paged_attention_varlen_v1_raw(
                queries,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                cu_seqlens_q,
                params,
                io_dtype,
                cache_dtype,
            )
        }
    } else {
        // SAFETY: caller guarantees all buffer pointers are valid.
        unsafe {
            dispatch_paged_attention_varlen_v2_raw(
                queries,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                cu_seqlens_q,
                params,
                io_dtype,
                cache_dtype,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params() {
        let params = PagedAttentionParams {
            num_seqs: 1,
            num_heads: 12,
            num_kv_heads: 2,
            head_size: 128,
            block_size: 16,
            max_seq_len: 2048,
            max_num_blocks_per_seq: 128,
            scale: 0.088388, // 1/sqrt(128)
            softcapping: 1.0,
            q_stride: 1536, // 12 * 128
            kv_block_stride: 32768,
            kv_head_stride: 16384,
            k_scale: 1.0,
            v_scale: 1.0,
            sliding_window: 0,
        };
        assert_eq!(params.num_heads / params.num_kv_heads, 6); // GQA ratio
        assert_eq!(params.sliding_window, 0);
    }

    #[test]
    fn test_fp8_params() {
        let params = PagedAttentionParams {
            k_scale: 0.5,  // FP8 quantization scale
            v_scale: 0.25, // FP8 quantization scale
            ..Default::default()
        };
        assert_eq!(params.k_scale, 0.5);
        assert_eq!(params.v_scale, 0.25);
    }

    #[test]
    fn test_sliding_window_params() {
        let params = PagedAttentionParams {
            sliding_window: 1024,
            ..Default::default()
        };
        assert_eq!(params.sliding_window, 1024);
    }
}
