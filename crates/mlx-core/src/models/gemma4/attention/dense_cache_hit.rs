//! Dense cache-hit attention keeps K/V, its absolute origin, and the group's
//! sliding window together. The paged-pool reader gathers only positions that
//! at least one query can attend; the host fallback retains the full context.
//! Both still need a per-query causal/window mask. In particular, the host
//! gather can contain retired null blocks whose contents are undefined.
//!
//! Private fields prevent callers from accidentally discarding the window.
//! The production dispatcher regression test covers both dense routes; adapter
//! readers also remain public, so privacy alone is not a complete guard.

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::mask::create_causal_mask;
use crate::transformer::paged_kv_cache_adapter::{DenseAttentionWindow, PagedKVCacheAdapter};

/// Window argument for a **dense** (gathered-K/V) cache-hit prefill route.
///
/// `None` means "plain causal is already exact" and the caller may keep the
/// fused causal fast path; `Some(w)` means the route must apply an explicit
/// windowed keep-mask. Both dense routes -- paged-pool SDPA and the host-read
/// fallback -- go through this one function, so the window can be dropped for
/// dense attention in exactly zero places.
///
/// Returning `None` is load-bearing in **two** cases, not one, and for the
/// same reason: MLX dispatches different kernels with and without an explicit
/// mask, and the mask-bearing kernel uses a different BF16 reduction order, so
/// asking for a mask that changes no value still moves paged-vs-flat parity by
/// a few ULP per layer across every sliding layer.
///
/// 1. A **global** group (`window == 0`) has no window to apply and must keep
///    the exact kernel it used before this plumbing existed.
/// 2. A windowed group whose window **cannot bite** on this chunk, i.e.
///    `cached_prefix_len + seq_len <= window`. The oldest key any query row in
///    this chunk can see is position 0, and `q_abs - 0 < window` holds for
///    every row, so the windowed mask is pointwise identical to plain causal.
///    This is exactly the predicate the flat path already applies in
///    `sliding_mask_offset_for_chunk` (`prior_len + seq_len > window`, where
///    `prior_len = min(cache_offset, window)` -- the same inequality once
///    `cached_prefix_len >= window` makes both sides trivially true). Keeping
///    the two in step means paged and flat agree on the *kernel* as well as on
///    the value: for a gemma-4-12b-it prompt of 513..1024 tokens the single
///    body chunk lands here, and before this check the paged route took the
///    mask-bearing kernel while flat took the fused causal one.
///
/// Nothing is lost by the second case: the null-block placeholders only exist
/// once `prune_sliding_window_for` has fired, which requires the recorded
/// context to exceed the window -- precisely when this returns `Some`.
pub(super) fn cache_hit_dense_window_arg(
    window: DenseAttentionWindow,
    cached_prefix_len: u32,
    seq_len: i64,
) -> Option<i32> {
    if !window.is_windowed() {
        return None;
    }
    let total_ctx = u64::from(cached_prefix_len).saturating_add(seq_len.max(0) as u64);
    (total_ctx > u64::from(window.tokens())).then_some(window.tokens() as i32)
}

/// Kernel a dense cache-hit route uses when no window mask is needed.
///
/// The two dense routes historically differed here and must keep differing:
/// MLX dispatches a different kernel when an explicit mask is present, with a
/// different BF16 reduction order, so changing either one's unmasked kernel
/// would drift paged-vs-flat parity. This selects the kernel for the cases
/// where `cache_hit_dense_window_arg` answers `None` -- a global group, or a
/// windowed group whose window cannot bite on this chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GlobalDenseKernel {
    /// Fused causal SDPA, no explicit mask -- the paged-pool SDPA route.
    FusedCausal,
    /// Explicit full-causal mask -- the host-read fallback.
    ExplicitCausalMask,
}

/// A dense cache-hit gather with its window and absolute key origin.
///
/// There is no way to read `keys`/`values` out of this type in production code
/// -- the fields are private to this module and there is no accessor. The only
/// way to turn it into numbers is [`Self::attention`], which cannot forget the
/// window because it holds it. Constructed only by the two adapter readers that
/// return a window alongside the data.
pub(super) struct DenseCacheHitKv {
    keys: MxArray,
    values: MxArray,
    /// This group's own window, taken from the same adapter call that produced
    /// `keys`/`values`. [`DenseAttentionWindow`] has no public constructor, so
    /// a literal `0` cannot be written here either.
    window: DenseAttentionWindow,
    key_start: u32,
}

impl DenseCacheHitKv {
    /// Gather through graph-native paged-pool views: the route production
    /// `Auto` actually selects at gemma-4 prefill sizes.
    pub(super) fn gather_through_paged_pool(
        adapter: &mut PagedKVCacheAdapter,
        paged_idx: u32,
        total_ctx: u32,
        query_len: u32,
    ) -> std::result::Result<Self, String> {
        let (keys, values, window, key_start) =
            adapter.gather_kv_for_windowed_prefill(paged_idx, total_ctx, query_len)?;
        Ok(Self {
            keys,
            values,
            window,
            key_start,
        })
    }

    /// Materialize through the synchronous host read: the last-resort arm,
    /// reachable from EVERY mode because it runs whenever no earlier arm
    /// produced an output.
    pub(super) fn read_through_host(
        adapter: &mut PagedKVCacheAdapter,
        paged_idx: u32,
        total_ctx: u32,
    ) -> std::result::Result<Self, String> {
        let (keys, values, window) =
            adapter.read_kv_range_for_dense_attention(paged_idx, total_ctx)?;
        Ok(Self {
            keys,
            values,
            window,
            key_start: 0,
        })
    }

    /// Attention over this gather. The only exit from this type.
    ///
    /// There is deliberately no `window()` accessor: the caller's routing
    /// estimate (and its mask-byte charge) is computed BEFORE the gather, off
    /// `adapter.dense_attention_window()`, so nothing in production needs to
    /// read the window back out here. Add one only when a caller genuinely
    /// does, and never a `keys()`/`values()` pair -- that is the hole this
    /// module exists to close.
    pub(super) fn attention(
        &self,
        queries_bhtd: &MxArray,
        seq_len: i64,
        cached_prefix_len: u32,
        global_kernel: GlobalDenseKernel,
    ) -> Result<MxArray> {
        dense_cache_hit_attention(
            queries_bhtd,
            &self.keys,
            &self.values,
            seq_len,
            cached_prefix_len
                .checked_sub(self.key_start)
                .ok_or_else(|| {
                    Error::from_reason("dense attention origin exceeds query position")
                })?,
            self.window,
            global_kernel,
        )
    }
}

/// Apply causal/window attention to gathered K/V. `cached_prefix_len` is
/// relative to the gather's first key, not necessarily absolute position zero.
/// Its sum with `seq_len` must equal the gather width. The mask excludes both
/// keys beyond each query and keys that have aged out of that query's window.
/// Tests can call this helper directly; production uses `DenseCacheHitKv` so
/// the data, origin and window come from the same adapter read.
pub(super) fn dense_cache_hit_attention(
    queries_bhtd: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    seq_len: i64,
    cached_prefix_len: u32,
    window: DenseAttentionWindow,
    global_kernel: GlobalDenseKernel,
) -> Result<MxArray> {
    match cache_hit_dense_window_arg(window, cached_prefix_len, seq_len) {
        Some(window) => {
            let mask =
                create_causal_mask(seq_len as i32, Some(cached_prefix_len as i32), Some(window))?;
            scaled_dot_product_attention(queries_bhtd, keys, values, 1.0, Some(&mask))
        }
        None => match global_kernel {
            GlobalDenseKernel::FusedCausal => {
                scaled_dot_product_attention_causal(queries_bhtd, keys, values, 1.0)
            }
            GlobalDenseKernel::ExplicitCausalMask => {
                let mask =
                    create_causal_mask(seq_len as i32, Some(cached_prefix_len as i32), None)?;
                scaled_dot_product_attention(queries_bhtd, keys, values, 1.0, Some(&mask))
            }
        },
    }
}
