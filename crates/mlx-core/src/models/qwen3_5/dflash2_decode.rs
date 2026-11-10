//! DFlash2 whole-turn integration for dense Qwen3.8.

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::engine::backend::{DsparkBackend, DsparkProposal, DsparkStepper, DsparkVerifyOutput};
use crate::engine::memory_plan::EXTENT_TARGET_BYTES;
use crate::stream::WiredLimitContext;
use crate::transformer::KvFormat;
use crate::transformer::paged_kv_cache_adapter::PagedPrefillMemorySnapshot;
use crate::transformer::paged_policy::{
    estimate_paged_pool_sdpa_bytes_with_portable, estimate_varlen_paged_attention_bytes,
};

use super::dflash2::DFlash2ContextCache;
use super::dflash2_state::trace::PrefixTracker;
use super::dflash2_turn::{DFlash2DecodeKv, DFlash2Target, DFlash2TurnState};
use super::gdn_blob::GdnStateBlobs;
use super::layer_cache::{
    Qwen3_5LayerCache, Qwen3_5LayerSnapshot, replay_mtp_snapshot_to, rewind_full_attention_to,
    snapshot_all_mtp,
};
use super::model::{PREFILL_STEP_SIZE, Qwen35Inner};

pub(crate) struct Qwen35DFlash2Stepper<'a> {
    inner: &'a mut Qwen35Inner,
    context: DFlash2ContextCache,
    next_position: i32,
    tap_layers: Vec<usize>,
    snapshot: Option<Vec<Qwen3_5LayerSnapshot>>,
    tape: Option<Vec<Option<super::gated_delta_net::GdnLayerTape>>>,
    tapped: Option<Vec<MxArray>>,
    verified_ids: Option<Vec<u32>>,
    /// Device-resident verify ids `[1+L]` set by [`DsparkStepper::verify_device`]
    /// — the token provenance [`Self::commit`] materializes once the verify
    /// graph has been forced (post-acceptance read = plain copy, no sync).
    verified_ids_device: Option<MxArray>,
    /// Packed GDN state of every linear layer (the per-layer cache slots are
    /// views of it) — the pre-verify state the fused commit replays from.
    /// `None` keeps the per-layer replay path.
    gdn_blobs: Option<GdnStateBlobs>,
    kv: DFlash2DecodeKv,
}

/// Packed GDN state is on unless `MLX_DFLASH2_GDN_BLOB=0` (A/B switch).
fn dflash2_gdn_blob_enabled() -> bool {
    std::env::var("MLX_DFLASH2_GDN_BLOB")
        .ok()
        .is_none_or(|value| value.trim() != "0")
}

pub(crate) fn reusable_dflash2_prefix(
    is_delta: bool,
    tokens: &[u32],
    prior_cached: usize,
    verified_hit: usize,
    retained_context_tokens: Option<&[u32]>,
    flat_attention_frontier: Option<usize>,
    flat_lane_authoritative: bool,
) -> usize {
    let candidate = if is_delta { prior_cached } else { verified_hit };
    if candidate > 0
        && candidate < tokens.len()
        && flat_lane_authoritative
        && flat_attention_frontier == Some(candidate)
        && retained_context_tokens
            .is_some_and(|retained| retained.len() == candidate && retained == &tokens[..candidate])
    {
        candidate
    } else {
        0
    }
}

pub(crate) fn constrain_dflash2_context_params(
    prompt_tokens: usize,
    target_capacity: i32,
    draft_capacity: usize,
    published_window: Option<u32>,
    params: &mut crate::engine::params::ChatParams,
) -> Result<usize> {
    let target_capacity = usize::try_from(target_capacity.max(0)).unwrap_or(usize::MAX);
    let window = published_window.map_or(usize::MAX, |window| window as usize);
    let capacity = target_capacity.min(draft_capacity).min(window);
    super::model::constrain_paged_context_params(
        "Qwen3.8 DFlash2",
        prompt_tokens,
        u32::try_from(capacity).unwrap_or(u32::MAX),
        params,
    )?;
    Ok(capacity)
}

pub(crate) fn dflash2_final_token_fits_context(logical_len: i32, capacity: usize) -> bool {
    usize::try_from(logical_len).is_ok_and(|logical_len| logical_len < capacity)
}

pub(crate) fn dflash2_memory_snapshot() -> PagedPrefillMemorySnapshot {
    let (mut active, mut cached, mut limit) = (0u64, 0u64, 0u64);
    // SAFETY: each probe writes one u64 through a valid pointer and reports
    // failure through its return code.
    let allocator_ok = unsafe {
        mlx_sys::mlx_get_active_memory(&mut active) == 0
            && mlx_sys::mlx_get_cache_memory(&mut cached) == 0
            && mlx_sys::mlx_get_memory_limit(&mut limit) == 0
            && limit > 0
    };
    let recommended = WiredLimitContext::get_max_working_set_size() as u64;
    PagedPrefillMemorySnapshot {
        allocator_active_bytes: allocator_ok.then_some(active),
        allocator_cached_bytes: allocator_ok.then_some(cached),
        allocator_limit_bytes: allocator_ok.then_some(limit),
        metal_recommended_working_set_bytes: (recommended > 0).then_some(recommended),
        metal_current_allocated_bytes: None,
        // Private paged pools are invisible to MLX's active counter.
        paged_pool_allocated_bytes: Some(crate::cache_limit::coordinator().registered_pool_bytes()),
    }
}

/// Blocks one growth adds to a runway paged pool: the whole blocks nearest
/// Splash's extent target, at least one.
pub(crate) fn runway_pool_extent_blocks(bytes_per_block: u64) -> u32 {
    let bytes_per_block = bytes_per_block.max(1);
    ((EXTENT_TARGET_BYTES + bytes_per_block / 2) / bytes_per_block).clamp(1, u64::from(u32::MAX))
        as u32
}

/// Bytes the copy of one layer allocates while a live flat target KV grows
/// to `rows` rows: `KVCache::grow_exact` builds each layer's new buffer
/// beside its old one.
pub(crate) fn dflash2_kv_layer_copy_bytes(rows: u64, row_bytes: u64, fa_layers: u64) -> u64 {
    rows.saturating_mul(row_bytes).div_ceil(fa_layers.max(1))
}

/// Bytes one token row costs across every full-attention K and V tensor:
/// `head_dim x activation bytes` per (layer, head, tensor) for BF16 rows,
/// `head_dim + 4` (int8 row plus its fp32 scale) for the int8 format.
pub(crate) fn kv_row_bytes(
    full_attention_layers: usize,
    kv_heads: i32,
    head_dim: i32,
    dtype: DType,
    format: KvFormat,
) -> u64 {
    (full_attention_layers as u64)
        .saturating_mul(2)
        .saturating_mul(kv_heads.max(0) as u64)
        .saturating_mul(format.row_bytes(i64::from(head_dim), dtype.byte_size() as u64))
}

/// Per-cycle short-lived allocations of one DFlash2 verify cycle. Feeds
/// `cache_limit::decode_cache_limit` with [`Self::total`].
pub(crate) struct DFlash2CycleTransient {
    /// Logits, GDN tapes and one layer's scratch set: geometry-only.
    pub core_bytes: u64,
    /// Two-pass attention partials: grows with context length; one set per
    /// full-attention layer can be in flight at once. Zero below the
    /// two-pass threshold.
    pub partials_bytes: u64,
}

impl DFlash2CycleTransient {
    pub fn total(&self) -> u64 {
        self.core_bytes.saturating_add(self.partials_bytes)
    }
}

/// First-order estimate of the short-lived bytes one DFlash2 verify cycle
/// allocates and frees, used to size the decode-time allocator cache
/// ceiling. `context_rows` is the turn's upper bound on attended keys
/// (the rows the target KV holds).
///
/// A cycle runs the target over `rows = block_size + 1` tokens — the anchor
/// plus the draft's `block_size` proposals (the loaded `block_size` is the
/// checkpoint's minus one, see `DFlash2Model::load`) — and keeps these
/// transient outputs alive until commit:
///
/// ```text
/// logits   = rows × vocab × 4 B × 2        f32 full-span logits + the
///                                           penalty/argmax copy
/// tapes    = layers × rows × (hidden        bf16 residual tap and the GDN
///            + conv_dim + 2·k_heads·k_dim    (q, k, v, qkv-window) tape kept
///            + v_heads·v_dim) × 2 B          per layer for the rollback replay
/// scratch  = rows × 3 × (hidden             one layer's attention/MLP
///            + intermediate) × 2 B           intermediates; identical sizes
///                                           are reused layer to layer, so
///                                           only one set is ever cached
/// partials = fa_layers × heads × rows        two-pass SDPA stage-1 output
///            × partitions × head_dim × 2 B   (`segmented_sdpa` / MLX vector
///                                           SDPA); only when the context is
///                                           long enough for two-pass (≥ 4096
///                                           keys), with partitions ≈ one per
///                                           64 keys clamped to [32, 1024]
///                                           (the tile kernel's count; the
///                                           vector kernel's 128/256/512 steps
///                                           sit at or below it)
/// ```
///
/// Weights and KV caches are resident (never freed per cycle) and are
/// deliberately excluded; the per-layer GDN recurrent state is re-allocated
/// each cycle but layer `i`'s freed state is reused by layer `i + 1`, so it
/// behaves like scratch and is not counted separately. Qwen3.8-27B (8 rows)
/// at a 1 K prompt: ≈ 15 + 25 + 1 MiB, no partials → `decode_cache_limit`'s
/// 128 MiB floor wins; at a 6 K prompt (6347 keys): + 16 × 24 × 8 × 128 ×
/// 256 × 2 B = 192 MiB of partials → 466 MiB cap (logged in production),
/// above the 340 MiB the unbounded pool holds there (see
/// `decode_cache_limit` for why that is intended).
fn dflash2_cycle_transient(
    config: &super::config::Qwen3_5Config,
    block_size: usize,
    context_rows: u64,
) -> DFlash2CycleTransient {
    let dim = |v: i32| v.max(0) as u64;
    let rows = block_size.saturating_add(1) as u64;
    let logits = rows
        .saturating_mul(dim(config.vocab_size))
        .saturating_mul(4)
        .saturating_mul(2);
    let key_width = dim(config.linear_num_key_heads)
        .saturating_mul(dim(config.linear_key_head_dim))
        .saturating_mul(2);
    let value_width =
        dim(config.linear_num_value_heads).saturating_mul(dim(config.linear_value_head_dim));
    let conv_dim = key_width.saturating_add(value_width);
    let tape_width = dim(config.hidden_size)
        .saturating_add(conv_dim)
        .saturating_add(key_width)
        .saturating_add(value_width);
    let tapes = dim(config.num_layers)
        .saturating_mul(rows)
        .saturating_mul(tape_width)
        .saturating_mul(2);
    let scratch = rows
        .saturating_mul(3)
        .saturating_mul(dim(config.hidden_size).saturating_add(dim(config.intermediate_size)))
        .saturating_mul(2);
    let partials = if context_rows >= SDPA_TWO_PASS_MIN_KEYS {
        let partitions = context_rows
            .div_ceil(SDPA_KEYS_PER_PARTITION)
            .next_multiple_of(32)
            .clamp(32, 1024);
        let fa_layers = (0..dim(config.num_layers) as usize)
            .filter(|&index| !config.is_linear_layer(index))
            .count() as u64;
        fa_layers
            .saturating_mul(dim(config.num_heads))
            .saturating_mul(rows)
            .saturating_mul(partitions)
            .saturating_mul(dim(config.head_dim))
            .saturating_mul(2)
    } else {
        0
    };
    DFlash2CycleTransient {
        core_bytes: logits.saturating_add(tapes).saturating_add(scratch),
        partials_bytes: partials,
    }
}

/// Decode-phase allocator free-pool ceiling for one DFlash2 turn: the
/// transient estimate above through `cache_limit::decode_cache_limit`.
pub(crate) fn dflash2_decode_cache_limit(
    config: &super::config::Qwen3_5Config,
    block_size: usize,
    context_rows: u64,
) -> u64 {
    crate::cache_limit::decode_cache_limit(
        dflash2_cycle_transient(config, block_size, context_rows).total(),
    )
}

/// Bytes the decode phase holds besides KV at `context_rows` attended keys:
/// one verify cycle and the free pool [`dflash2_decode_cache_limit`] lets
/// it keep. MLX collects that pool only at min(0.95·WS, its memory limit)
/// and no load lowers that limit to the plan, so the pool is resident
/// memory the hard budget must charge (Splash charges every live buffer).
pub(crate) fn dflash2_cycle_transient_bytes(
    config: &super::config::Qwen3_5Config,
    block_size: usize,
    context_rows: u64,
) -> u64 {
    let cycle = dflash2_cycle_transient(config, block_size, context_rows).total();
    cycle.saturating_add(crate::cache_limit::decode_cache_limit(cycle))
}

/// Bytes a DFlash2 turn allocates and frees besides its KV at its peak,
/// `context_rows` being the rows the target KV holds: the larger of a
/// prefill chunk (one layer's attention and MLP scratch over the chunk rows,
/// the draft's tapped hidden rows of that chunk and, for int8 rows, one
/// full-attention layer's prefix dequantized beside its concatenation with
/// the chunk) and one verify cycle.
pub(crate) fn dflash2_turn_transient_bytes(
    config: &super::config::Qwen3_5Config,
    block_size: usize,
    prompt_len: usize,
    context_rows: u64,
    kv_format: KvFormat,
    tap_layers: usize,
) -> u64 {
    prefill_transient_bytes(config, prompt_len as u64, kv_format, tap_layers).max(
        dflash2_cycle_transient_bytes(config, block_size, context_rows),
    )
}

/// One prefill chunk of a flat turn over `prompt` rows: one layer's
/// attention and MLP scratch, `tap_layers` tapped hidden rows of the chunk
/// and, for int8 rows, one full-attention layer's prefix dequantized beside
/// its concatenation with the chunk.
fn prefill_transient_bytes(
    config: &super::config::Qwen3_5Config,
    prompt: u64,
    kv_format: KvFormat,
    tap_layers: usize,
) -> u64 {
    let dim = |v: i32| v.max(0) as u64;
    let chunk = prompt.min(PREFILL_STEP_SIZE as u64);
    let taps = chunk
        .saturating_mul(tap_layers as u64)
        .saturating_mul(dim(config.hidden_size))
        .saturating_mul(2);
    let dequantized_prefix = match kv_format {
        KvFormat::Int8 => prompt
            .saturating_mul(2)
            .saturating_add(chunk)
            .saturating_mul(dim(config.num_kv_heads))
            .saturating_mul(dim(config.head_dim))
            .saturating_mul(2 * 2),
        KvFormat::Bf16 => 0,
    };
    prefill_chunk_scratch_bytes(config, prompt)
        .saturating_add(taps)
        .saturating_add(dequantized_prefix)
}

/// Which fused SDPA kernels a paged prefill chunk can take, as
/// `select_cache_hit_prefill_plan`'s estimators read them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillSdpaKernels {
    pub d256_full: bool,
    pub portable_d256: bool,
}

impl PrefillSdpaKernels {
    /// The kernels a BF16 paged prefill of `head_dim`-wide heads can take on
    /// this device.
    pub(crate) fn probe(head_dim: i32) -> Self {
        Self {
            d256_full: super::attention::d256_full_sdpa_available(false),
            portable_d256: head_dim == 256
                && unsafe { mlx_sys::mlx_metal_portable_d256_sdpa_available() },
        }
    }
}

/// One layer's attention and MLP scratch over a prefill chunk of the first
/// `window` rows.
fn prefill_chunk_scratch_bytes(config: &super::config::Qwen3_5Config, window: u64) -> u64 {
    let dim = |v: i32| v.max(0) as u64;
    window
        .min(PREFILL_STEP_SIZE as u64)
        .saturating_mul(3)
        .saturating_mul(dim(config.hidden_size).saturating_add(dim(config.intermediate_size)))
        .saturating_mul(2)
}

/// Bytes a paged pool turn of `window` tokens allocates besides KV at its
/// peak: the last prefill chunk's scratch with the cheaper of the two
/// attention routes the paged prefill picks from, or one decode step.
pub(crate) fn pool_lane_transient(
    config: &super::config::Qwen3_5Config,
    dtype: DType,
    kernels: PrefillSdpaKernels,
    window: u32,
) -> u64 {
    let dim = |v: i32| v.max(0) as u64;
    let window = u64::from(window);
    let chunk = window.min(PREFILL_STEP_SIZE as u64);
    let (heads, kv_heads, head_dim) = (
        dim(config.num_heads),
        dim(config.num_kv_heads),
        dim(config.head_dim),
    );
    let dtype_bytes = dtype.byte_size() as u64;
    let gathered = estimate_paged_pool_sdpa_bytes_with_portable(
        chunk,
        window,
        heads,
        kv_heads,
        head_dim,
        dtype_bytes,
        kernels.d256_full,
        kernels.portable_d256,
    );
    let varlen = estimate_varlen_paged_attention_bytes(
        chunk,
        window,
        heads,
        kv_heads,
        head_dim,
        dtype_bytes,
    );
    let prefill = prefill_chunk_scratch_bytes(config, window).saturating_add(gathered.min(varlen));
    prefill.max(dflash2_cycle_transient(config, 0, window).total())
}

/// Bytes a flat autoregressive turn of `window` tokens allocates besides KV
/// at its peak: its last prefill chunk, or one decode step. No DFlash2
/// decode cache limit is pushed on this lane.
pub(crate) fn flat_lane_transient(
    config: &super::config::Qwen3_5Config,
    kv_format: KvFormat,
    window: u32,
) -> u64 {
    let window = u64::from(window);
    prefill_transient_bytes(config, window, kv_format, 0)
        .max(dflash2_cycle_transient(config, 0, window).total())
}

/// Bytes a DFlash2 turn of `window` tokens allocates besides KV at its peak:
/// its prefill over the window, or one verify cycle over the window and its
/// scratch rows.
pub(crate) fn dflash2_lane_transient(
    config: &super::config::Qwen3_5Config,
    block_size: usize,
    kv_format: KvFormat,
    tap_layers: usize,
    window: u32,
) -> u64 {
    dflash2_turn_transient_bytes(
        config,
        block_size,
        window as usize,
        u64::from(window).saturating_add(block_size as u64),
        kv_format,
        tap_layers,
    )
}

/// Below this many keys the attention kernels run one-pass and allocate no
/// partials (`sdpa_vector_uses_two_pass` in `mlx_segmented_sdpa_plan.h`).
const SDPA_TWO_PASS_MIN_KEYS: u64 = 4096;
/// Keys per stage-1 partition of the segmented verify tile kernel
/// (`segmented_verify_tile_partitions`: `8 × max(tile_n, 8)`).
const SDPA_KEYS_PER_PARTITION: u64 = 64;

/// Smallest full-attention buffer capacity, `None` without full-attention
/// caches.
pub(crate) fn min_fa_capacity(caches: &Option<Vec<Qwen3_5LayerCache>>) -> Result<Option<i64>> {
    let mut min = None;
    for cache in caches.iter().flatten() {
        if let Qwen3_5LayerCache::FullAttention(kv) = cache {
            let capacity = kv.capacity()?;
            min = Some(min.map_or(capacity, |m: i64| m.min(capacity)));
        }
    }
    Ok(min)
}

/// Smallest rows every full-attention cache holds without growing,
/// counting the size a cold cache's first allocation is set to; `None`
/// without full-attention caches.
pub(crate) fn min_fa_reserved_rows(caches: &Option<Vec<Qwen3_5LayerCache>>) -> Result<Option<i64>> {
    let mut min = None;
    for cache in caches.iter().flatten() {
        if let Qwen3_5LayerCache::FullAttention(kv) = cache {
            let rows = kv.reserved_rows()?;
            min = Some(min.map_or(rows, |m: i64| m.min(rows)));
        }
    }
    Ok(min)
}

/// Grow every full-attention cache to exactly `rows` rows, one layer at a
/// time, releasing each layer's old buffer before the next copy.
pub(crate) fn grow_dflash2_target_kv(
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    rows: i64,
) -> Result<()> {
    let caches = caches
        .as_mut()
        .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?;
    for cache in caches.iter_mut() {
        if let Qwen3_5LayerCache::FullAttention(kv) = cache {
            let live = kv.capacity()? > 0;
            kv.grow_exact(rows)?;
            if live {
                crate::array::clear_cache();
            }
        }
    }
    Ok(())
}

impl Qwen35DFlash2Stepper<'_> {
    fn ensure_clean(&self, operation: &str) -> Result<()> {
        if self.snapshot.is_some()
            || self.tape.is_some()
            || self.tapped.is_some()
            || self.verified_ids.is_some()
            || self.verified_ids_device.is_some()
        {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 {operation}: prior verify was not committed"
            )));
        }
        Ok(())
    }

    fn target_forward(
        &mut self,
        ids: &[u32],
    ) -> Result<(
        MxArray,
        Vec<MxArray>,
        Vec<Option<super::gated_delta_net::GdnLayerTape>>,
    )> {
        let ids = ids.iter().map(|&id| id as i32).collect::<Vec<_>>();
        let input = MxArray::from_int32(&ids, &[1, ids.len() as i64])?;
        super::model::forward_dflash2_with_taps(
            self.inner,
            &input,
            &self.tap_layers,
            true,
            super::model::DFlash2LogitsSpan::All,
            self.gdn_blobs.as_ref().map(GdnStateBlobs::destinations),
            None,
        )
        .map(|(logits, taps, tape, _)| (logits, taps, tape))
    }

    fn append_tapped(&mut self, tapped: &[MxArray], token_ids: &[u32]) -> Result<()> {
        let keep = token_ids.len();
        let mut kept = Vec::with_capacity(tapped.len());
        for hidden in tapped {
            kept.push(hidden.slice_axis(1, 0, keep as i64)?);
        }
        let draft = self
            .inner
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?;
        let fused = draft.fuse_context(&kept)?;
        self.context
            .append(draft, &fused, self.next_position, token_ids)?;
        self.next_position = self.next_position.saturating_add(keep as i32);
        Ok(())
    }
}

impl DsparkStepper for Qwen35DFlash2Stepper<'_> {
    fn propose(
        &mut self,
        anchor_id: u32,
        max_len: usize,
        params: &crate::engine::params::ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        let draft = self
            .inner
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?;
        let temperature = params
            .sampling_config
            .unwrap_or_default()
            .temperature
            .unwrap_or(1.0);
        // Mirror `accept_dspark_proposal`'s `greedy_fast` predicate: the
        // device-resident path is only offered when acceptance will consume
        // it without a mid-cycle host read. Sampled or penalized-greedy
        // turns keep the host walk (their accept path needs host ids anyway).
        let greedy_fast = crate::sampling::is_greedy_temperature(temperature)
            && params.repetition_penalty == 1.0
            && params.presence_penalty == 0.0
            && params.frequency_penalty == 0.0;
        let (path, draft_sparse_dists) = draft.propose(
            &self.inner.embedding,
            self.inner.lm_head.as_ref(),
            &mut self.context,
            anchor_id,
            max_len,
            temperature,
            greedy_fast,
            None,
            rng,
        )?;
        let (draft_ids, device_draft_ids) = match path {
            super::dflash2::SelectorPath::Device(ids) => {
                // Queue the draft before the verify graph is built so the GPU
                // is not idle during that host work. Off Metal this call is a
                // blocking eval, which would only serialize the two builds.
                if unsafe { mlx_sys::mlx_metal_is_available() } {
                    MxArray::async_eval_arrays(&[&ids]);
                }
                (Vec::new(), Some(ids))
            }
            super::dflash2::SelectorPath::Host(ids) => (ids, None),
        };
        Ok(DsparkProposal {
            draft_ids,
            device_draft_ids,
            draft_dists: Vec::new(),
            draft_sparse_dists,
            keep_probabilities: None,
        })
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        if verify_ids.is_empty() {
            return Err(Error::from_reason(
                "Qwen3.8 DFlash2 verify block must not be empty",
            ));
        }
        self.ensure_clean("verify")?;
        self.kv.ensure_ahead(self.inner, verify_ids.len())?;
        let snapshot = snapshot_all_mtp(
            self.inner
                .caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?,
            false,
        )?;
        let (logits, tapped, tape) = self.target_forward(verify_ids)?;
        self.snapshot = Some(snapshot);
        self.tape = Some(tape);
        self.tapped = Some(tapped);
        self.verified_ids = Some(verify_ids.to_vec());
        Ok(DsparkVerifyOutput { logits })
    }

    /// Device-resident verify: `verify_ids` is the `[1+L]` concat of the
    /// anchor and the selector's device path — the whole propose → verify
    /// chain stays one lazy graph (no proposal readback in between).
    /// Provenance materializes in `commit`, after acceptance has forced the
    /// shared roots.
    fn verify_device(&mut self, verify_ids: &MxArray) -> Result<DsparkVerifyOutput> {
        self.ensure_clean("verify")?;
        let rows = usize::try_from(verify_ids.shape_at(0)?)
            .map_err(|_| Error::from_reason("Qwen3.8 DFlash2 negative verify rows"))?;
        self.kv.ensure_ahead(self.inner, rows)?;
        let snapshot = snapshot_all_mtp(
            self.inner
                .caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?,
            false,
        )?;
        let input = verify_ids.reshape(&[1, verify_ids.shape_at(0)?])?;
        let phase_time = std::env::var("MLX_DFLASH2_PHASE_TIME").is_ok();
        let t0 = std::time::Instant::now();
        let (logits, tapped, tape, _) = super::model::forward_dflash2_with_taps(
            self.inner,
            &input,
            &self.tap_layers,
            true,
            super::model::DFlash2LogitsSpan::All,
            self.gdn_blobs.as_ref().map(GdnStateBlobs::destinations),
            None,
        )?;
        if phase_time {
            eprintln!("[dflash2-phase] verify-build: {:?}", t0.elapsed());
        }
        self.snapshot = Some(snapshot);
        self.tape = Some(tape);
        self.tapped = Some(tapped);
        self.verified_ids_device = Some(verify_ids.clone());
        if phase_time {
            let t = std::time::Instant::now();
            MxArray::eval_arrays(&[&logits])?;
            eprintln!("[dflash2-phase] verify: {:?}", t.elapsed());
        }
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        if keep == 0 || keep > total_written {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 invalid commit keep={keep}, total={total_written}"
            )));
        }
        let snapshot = self
            .snapshot
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 commit has no target snapshot"))?;
        let tape = self
            .tape
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 commit has no GDN tape"))?;
        let tapped = self
            .tapped
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 commit has no target taps"))?;
        // The verify graph has already been forced by acceptance, so a
        // device-resident provenance read here is a plain copy, not a sync.
        let verified_ids = match (self.verified_ids.take(), self.verified_ids_device.take()) {
            (Some(ids), None) => ids,
            (None, Some(ids)) => ids
                .to_int32()?
                .as_ref()
                .iter()
                .map(|&id| id as u32)
                .collect(),
            _ => {
                return Err(Error::from_reason(
                    "Qwen3.8 DFlash2 commit has no token provenance",
                ));
            }
        };
        if verified_ids.len() != total_written {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 commit wrote {total_written} rows for {} token ids",
                verified_ids.len()
            )));
        }
        let caches = self
            .inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?;
        // Packed path. Full accept (every verified row kept; the bonus token
        // is not in the state yet): the verify already wrote each layer's
        // window-final state into the spare blob rows, and with the f32
        // carry that equals replaying the whole window
        // (`gated_delta::tests::ar_chain_matches_windowed_kernel_f32`), so
        // adopting is a parity swap. Otherwise every linear layer's replay
        // (plus its conv rebuild) runs in one fused primitive from the
        // pre-verify blobs. The full-attention offsets rewind as before. A
        // declined kernel drops back to the per-layer replay for the rest of
        // the turn.
        let next_blobs = match &self.gdn_blobs {
            Some(blobs) => {
                let adopted = if keep == total_written {
                    blobs.adopt(&tape)?
                } else {
                    None
                };
                match adopted {
                    Some(next) => Some(next),
                    None => blobs.commit(&tape, keep)?,
                }
            }
            None => None,
        };
        match next_blobs {
            Some(next) => {
                // Rewind first: a failure here leaves the linear slots on the
                // pre-commit blobs, consistent with `self.gdn_blobs`.
                rewind_full_attention_to(caches, &snapshot, keep, "Qwen3.8 DFlash2 commit")?;
                next.apply_views(caches)?;
                self.gdn_blobs = Some(next);
            }
            None => {
                if self.gdn_blobs.take().is_some() {
                    tracing::debug!(
                        keep,
                        "Qwen3.8 DFlash2 fused GDN commit declined; per-layer replay for the rest of the turn"
                    );
                }
                replay_mtp_snapshot_to(
                    caches,
                    &snapshot,
                    &tape,
                    keep,
                    false,
                    "Qwen3.8 DFlash2 commit",
                )?;
            }
        }
        self.append_tapped(&tapped, &verified_ids[..keep])
    }

    fn commit_with_provenance(
        &mut self,
        keep: usize,
        total_written: usize,
        verified_ids: &[u32],
    ) -> Result<()> {
        if verified_ids.len() != total_written {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 commit wrote {total_written} rows for {} supplied token ids",
                verified_ids.len()
            )));
        }
        match (&self.verified_ids, &self.verified_ids_device) {
            (Some(expected), None) if expected.as_slice() == verified_ids => {}
            (None, Some(device))
                if device.ndim()? == 1 && device.shape_at(0)? as usize == total_written => {}
            (Some(_), None) => {
                return Err(Error::from_reason(
                    "Qwen3.8 DFlash2 supplied commit provenance disagrees with host verify input",
                ));
            }
            _ => {
                return Err(Error::from_reason(
                    "Qwen3.8 DFlash2 commit has invalid token provenance ownership",
                ));
            }
        }
        // Acceptance has already copied these ids. Replace the device verify
        // provenance with that host vector so `commit` keeps its established
        // validation and settlement path without another device copy.
        self.verified_ids = Some(verified_ids.to_vec());
        self.verified_ids_device = None;
        self.commit(keep, total_written)
    }

    fn finish(self) -> Result<()> {
        self.ensure_clean("finish")?;
        self.inner.dflash2_context = Some(self.context);
        Ok(())
    }

    fn eval_boundary(&self, token: &MxArray) {
        let mut arrays: Vec<&MxArray> = Vec::new();
        crate::models::forward::CollectCacheArrays::collect_arrays(&self.inner.caches, &mut arrays);
        self.context.collect_arrays(&mut arrays);
        arrays.push(token);
        MxArray::async_eval_arrays(&arrays);
    }
}

impl DsparkBackend for Qwen35Inner {
    type DsparkDecode<'a>
        = Qwen35DFlash2Stepper<'a>
    where
        Self: 'a;

    fn begin_dspark_decode(&mut self, block_size: usize) -> Result<Self::DsparkDecode<'_>> {
        let expected = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 has no loaded DFlash2 companion"))?
            .config
            .block_size;
        if block_size != expected {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 block size {block_size} does not match checkpoint {expected}"
            )));
        }
        let state = self.dflash2_turn_state.take().ok_or_else(|| {
            Error::from_reason("Qwen3.8 DFlash2 decode requires tapped prefill state")
        })?;
        let tap_layers = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("checked DFlash2 companion"))?
            .config
            .target_layers
            .clone();
        // Pack the prefill's per-layer GDN states once per turn (two concats);
        // every cycle's commit then rewrites the blobs in one fused primitive
        // and re-points the per-layer slots at them. The DFlash2 stepper
        // always runs on the flat caches (see the `paged = false` snapshots).
        let gdn_blobs = match &mut self.caches {
            Some(caches) if dflash2_gdn_blob_enabled() => {
                let blobs = GdnStateBlobs::pack(caches)?;
                match &blobs {
                    Some(blobs) => blobs.apply_views(caches)?,
                    None => tracing::debug!(
                        "Qwen3.8 DFlash2: GDN state not packable; per-layer commit path"
                    ),
                }
                blobs
            }
            _ => None,
        };
        let kv = DFlash2DecodeKv::begin(&*self, state.kv)?;
        Ok(Qwen35DFlash2Stepper {
            inner: self,
            context: state.context,
            next_position: state.next_position,
            tap_layers,
            snapshot: None,
            tape: None,
            tapped: None,
            verified_ids: None,
            verified_ids_device: None,
            gdn_blobs,
            kv,
        })
    }
}

impl DFlash2Target for Qwen35Inner {
    const PREFILL_STEP_SIZE: i64 = PREFILL_STEP_SIZE;

    fn label(&self) -> &'static str {
        "Qwen3.8"
    }

    fn dflash2(&self) -> Option<&super::dflash2::DFlash2Model> {
        self.dflash2.as_ref()
    }

    fn caches(&self) -> &Option<Vec<Qwen3_5LayerCache>> {
        &self.caches
    }

    fn caches_mut(&mut self) -> &mut Option<Vec<Qwen3_5LayerCache>> {
        &mut self.caches
    }

    fn dflash2_context(&self) -> Option<&DFlash2ContextCache> {
        self.dflash2_context.as_ref()
    }

    fn dflash2_context_mut(&mut self) -> &mut Option<DFlash2ContextCache> {
        &mut self.dflash2_context
    }

    fn dflash2_turn_state_mut(&mut self) -> &mut Option<DFlash2TurnState> {
        &mut self.dflash2_turn_state
    }

    fn prefix_tracker(&self) -> &PrefixTracker {
        &self.dflash2_prefix
    }

    fn prefix_tracker_mut(&mut self) -> &mut PrefixTracker {
        &mut self.dflash2_prefix
    }

    fn dflash2_states(&self) -> &crate::models::qwen3_5::dflash2_state::flat::FlatStates {
        &self.dflash2_states
    }

    fn dflash2_states_mut(
        &mut self,
    ) -> &mut crate::models::qwen3_5::dflash2_state::flat::FlatStates {
        &mut self.dflash2_states
    }

    fn turn_cancel(&self) -> Option<&std::sync::atomic::AtomicBool> {
        self.turn_cancel.as_deref()
    }

    fn kv_heads(&self) -> i32 {
        self.config.num_kv_heads
    }

    fn head_dim(&self) -> i32 {
        self.config.head_dim
    }

    fn max_position_embeddings(&self) -> i32 {
        self.config.max_position_embeddings
    }

    fn dflash2_window(&self) -> Option<u32> {
        self.speculative_window()
    }

    fn memory_plan(&self) -> Option<&crate::engine::memory_plan::MemoryPlan> {
        self.memory_plan.as_ref()
    }

    fn vision_cache(&self) -> &crate::vision::qwen::cache::VisionCache {
        &self.vision_cache
    }

    fn dense_config(&self) -> std::borrow::Cow<'_, crate::models::qwen3_5::Qwen3_5Config> {
        std::borrow::Cow::Borrowed(&self.config)
    }

    fn set_cached_token_history(&mut self, history: Vec<u32>) {
        self.cached_token_history = history;
    }

    fn config_kv_format(&self) -> KvFormat {
        self.config.kv_format()
    }

    fn embedding_dtype(&self) -> Result<DType> {
        self.embedding
            .forward(&MxArray::from_int32(&[0], &[1, 1])?)?
            .dtype()
    }

    fn forward_with_taps(
        &mut self,
        input_ids: &MxArray,
        tap_layers: &[usize],
        capture: Option<&super::gated_delta_net::GdnCapturePlan<'_>>,
    ) -> Result<(MxArray, Vec<MxArray>, Vec<super::gated_delta_net::GdnCell>)> {
        let (logits, taps, _, cells) = super::model::forward_dflash2_with_taps(
            self,
            input_ids,
            tap_layers,
            false,
            super::model::DFlash2LogitsSpan::LastRow,
            None,
            capture,
        )?;
        Ok((logits, taps, cells))
    }

    fn init_caches_sync(&mut self) -> Result<()> {
        Qwen35Inner::init_caches_sync(self)
    }

    fn reset_caches_sync(&mut self) -> Result<()> {
        Qwen35Inner::reset_caches_sync(self)
    }

    fn paged_adapter_ref(
        &self,
    ) -> Option<&crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter> {
        self.paged_adapter.as_ref()
    }

    fn paged_adapter_mut_ref(
        &mut self,
    ) -> Option<&mut crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter> {
        self.paged_adapter.as_mut()
    }
    fn release_paged_requests(&mut self) -> Result<()> {
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter.release_all_requests().map_err(|e| {
                Error::from_reason(format!("DFlash2 paged request release failed: {e}"))
            })?;
        }
        Ok(())
    }

    fn flat_lane_authoritative(&self) -> bool {
        !self.paged_full_attn_caches_dirty && !self.flat_mtp_caches_desynced
    }

    fn claim_flat_lane(&mut self) {
        self.paged_full_attn_caches_dirty = false;
        self.flat_mtp_caches_desynced = false;
    }

    fn oldest_gdn_state(
        &self,
        at_cap: bool,
        skip: &std::collections::HashSet<crate::engine::reclaim::UnitKey>,
        keep: crate::models::qwen3_5::dflash2_turn::Keep<'_>,
        in_use: bool,
    ) -> Option<crate::engine::reclaim::Unit> {
        Self::oldest_gdn_state_unit(self, at_cap, skip, keep, in_use)
    }

    fn drop_gdn_state(&mut self, key: &crate::engine::reclaim::UnitKey) -> bool {
        Self::drop_gdn_state(self, key)
    }

    fn sync_gdn_state_marks(&self) {
        Self::sync_gdn_state_marks(self);
    }

    fn gdn_state_bytes(
        &self,
        keep: crate::models::qwen3_5::dflash2_turn::Keep<'_>,
        in_use: bool,
    ) -> u64 {
        Self::gdn_state_bytes(self, keep, in_use)
    }

    fn marked_hit_blocks(&self, keep: crate::models::qwen3_5::dflash2_turn::Keep<'_>) -> u32 {
        Self::marked_hit_blocks(self, keep)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::array::DType;
    use crate::engine::backend::{ChatBackend, SpecFrontier, TurnOutput, WholeTurnArgs};
    use crate::engine::extract_chat_params;
    use crate::engine::types::ChatConfig;
    use crate::models::forward::PREFILL_CANCELLED;
    use crate::models::quantized_linear::LinearProj;
    use crate::models::qwen3_5::dflash2_turn::{
        LIVE_DRAFT_APPENDS, PrefillCapture, dflash2_chat_turn, dflash2_prefill,
        flat_attention_frontier,
    };
    use crate::models::qwen3_5::gated_delta_net::{GdnCapturePlan, GdnCell};
    use crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache;
    use crate::nn::Linear;
    use crate::stream::{Stream, StreamContext};
    use rand::SeedableRng;

    /// The turn's KV budget for a memory snapshot.
    impl Qwen35DFlash2Stepper<'_> {
        fn frontier(&self) -> Option<SpecFrontier> {
            Some(SpecFrontier {
                attn_tokens: flat_attention_frontier(self.inner)
                    .and_then(|frontier| u64::try_from(frontier).ok())?,
                recurrent_tokens: Some(self.next_position.max(0) as u64),
            })
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    pub(crate) struct ArTrace {
        logits: Vec<Vec<u32>>,
        target_cache: Vec<(Vec<i64>, Vec<u32>)>,
        draft_cache: Vec<(Vec<i64>, Vec<u32>)>,
        context_tokens: Vec<u32>,
        frontier: SpecFrontier,
        continuation_draft: Vec<i32>,
    }

    pub(crate) fn assert_trace_eq(actual: &ArTrace, expected: &ArTrace, label: &str) {
        fn words(actual: &[u32], expected: &[u32], label: &str) {
            assert_eq!(actual.len(), expected.len(), "{label}: element count");
            if let Some((index, (actual, expected))) = actual
                .iter()
                .zip(expected)
                .enumerate()
                .find(|(_, (actual, expected))| actual != expected)
            {
                panic!(
                    "{label}: first mismatch at element {index}: actual={actual:#010x} ({}), expected={expected:#010x} ({})",
                    f32::from_bits(*actual),
                    f32::from_bits(*expected),
                );
            }
        }

        assert_eq!(
            actual.logits.len(),
            expected.logits.len(),
            "{label}: cycles"
        );
        for (cycle, (actual, expected)) in actual.logits.iter().zip(&expected.logits).enumerate() {
            words(actual, expected, &format!("{label}: logits cycle={cycle}"));
        }
        for (name, actual, expected) in [
            ("target cache", &actual.target_cache, &expected.target_cache),
            ("draft cache", &actual.draft_cache, &expected.draft_cache),
        ] {
            assert_eq!(actual.len(), expected.len(), "{label}: {name} array count");
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(actual.0, expected.0, "{label}: {name} array={index} shape");
                words(
                    &actual.1,
                    &expected.1,
                    &format!("{label}: {name} array={index} shape={:?}", actual.0),
                );
            }
        }
        assert_eq!(
            actual.context_tokens, expected.context_tokens,
            "{label}: tokens"
        );
        assert_eq!(actual.frontier, expected.frontier, "{label}: frontier");
        assert_eq!(
            actual.continuation_draft, expected.continuation_draft,
            "{label}: continuation draft"
        );
    }

    pub(crate) fn array_fingerprints(arrays: Vec<&MxArray>) -> Result<Vec<(Vec<i64>, Vec<u32>)>> {
        arrays
            .into_iter()
            .map(|array| {
                let values = array.astype(DType::Float32)?;
                values.eval();
                Ok((
                    array.shape()?.as_ref().to_vec(),
                    values
                        .to_float32()?
                        .iter()
                        .map(|value| value.to_bits())
                        .collect(),
                ))
            })
            .collect()
    }

    /// Bit goldens of the DFlash2 fixture traces. They are device-specific,
    /// so they live outside git: `MLX_DFLASH2_GOLDEN_DIR=<dir>` with
    /// `MLX_DFLASH2_GOLDEN_MODE=write|check`. Without the directory the
    /// golden tests do nothing.
    pub(crate) mod golden {
        use std::fmt::Write as _;
        use std::path::PathBuf;

        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        enum Mode {
            Write,
            Check,
        }

        pub(crate) struct GoldenRun {
            dir: PathBuf,
            mode: Mode,
            failures: Vec<String>,
        }

        /// One case: labelled u32 words compared in order.
        #[derive(Default)]
        pub(crate) struct GoldenRecord {
            entries: Vec<(String, Vec<u32>)>,
        }

        impl GoldenRecord {
            pub(crate) fn push(&mut self, label: String, words: impl IntoIterator<Item = u32>) {
                assert!(
                    !label.contains(['\t', '\n']),
                    "golden label must not contain tabs or newlines: {label:?}"
                );
                self.entries.push((label, words.into_iter().collect()));
            }

            pub(crate) fn push_i32(&mut self, label: String, values: &[i32]) {
                self.push(label, values.iter().map(|&value| value as u32));
            }

            pub(crate) fn push_u64(&mut self, label: String, value: u64) {
                self.push(label, [value as u32, (value >> 32) as u32]);
            }

            pub(crate) fn push_fingerprints(
                &mut self,
                label: &str,
                arrays: &[(Vec<i64>, Vec<u32>)],
            ) {
                self.push_u64(format!("{label}.count"), arrays.len() as u64);
                for (index, (shape, words)) in arrays.iter().enumerate() {
                    self.push(
                        format!("{label}[{index}] shape={shape:?}"),
                        words.iter().copied(),
                    );
                }
            }

            fn encode(&self) -> String {
                let mut out = String::new();
                for (label, words) in &self.entries {
                    out.push_str(label);
                    out.push('\t');
                    for (index, word) in words.iter().enumerate() {
                        if index > 0 {
                            out.push(' ');
                        }
                        write!(out, "{word:08x}").expect("write to String");
                    }
                    out.push('\n');
                }
                out
            }

            fn decode(text: &str) -> Result<Self, String> {
                let mut entries = Vec::new();
                for (line_no, line) in text.lines().enumerate() {
                    let (label, words) = line
                        .split_once('\t')
                        .ok_or_else(|| format!("line {}: missing tab", line_no + 1))?;
                    let words = words
                        .split_ascii_whitespace()
                        .map(|word| u32::from_str_radix(word, 16))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|error| format!("line {}: {error}", line_no + 1))?;
                    entries.push((label.to_string(), words));
                }
                Ok(Self { entries })
            }

            fn compare(&self, expected: &Self) -> Result<(), String> {
                for (index, ((label, words), (expected_label, expected_words))) in
                    self.entries.iter().zip(&expected.entries).enumerate()
                {
                    if label != expected_label {
                        return Err(format!(
                            "entry {index}: label {label:?}, golden {expected_label:?}"
                        ));
                    }
                    if words.len() != expected_words.len() {
                        return Err(format!(
                            "{label}: {} words, golden {}",
                            words.len(),
                            expected_words.len()
                        ));
                    }
                    if let Some((word, (actual, golden))) = words
                        .iter()
                        .zip(expected_words)
                        .enumerate()
                        .find(|(_, (actual, golden))| actual != golden)
                    {
                        return Err(format!(
                            "{label}: first mismatch at word {word}: actual={actual:#010x}, golden={golden:#010x}"
                        ));
                    }
                }
                if self.entries.len() != expected.entries.len() {
                    return Err(format!(
                        "{} entries, golden {}",
                        self.entries.len(),
                        expected.entries.len()
                    ));
                }
                Ok(())
            }
        }

        impl GoldenRun {
            /// `None` when `MLX_DFLASH2_GOLDEN_DIR` is unset.
            pub(crate) fn from_env() -> Option<Self> {
                let dir = std::env::var_os("MLX_DFLASH2_GOLDEN_DIR")?;
                let mode = match std::env::var("MLX_DFLASH2_GOLDEN_MODE").as_deref() {
                    Ok("write") => Mode::Write,
                    Ok("check") => Mode::Check,
                    other => panic!(
                        "MLX_DFLASH2_GOLDEN_DIR is set: MLX_DFLASH2_GOLDEN_MODE must be write or check, got {other:?}"
                    ),
                };
                Some(Self {
                    dir: PathBuf::from(dir),
                    mode,
                    failures: Vec::new(),
                })
            }

            /// Write the case, or check it against the stored golden. Check
            /// failures are collected so one run reports every case.
            pub(crate) fn case(&mut self, family: &str, case: &str, record: &GoldenRecord) {
                let path = self.dir.join(family).join(format!("{case}.golden"));
                match self.mode {
                    Mode::Write => {
                        let parent = path.parent().expect("golden path has a parent");
                        std::fs::create_dir_all(parent)
                            .unwrap_or_else(|error| panic!("create {}: {error}", parent.display()));
                        std::fs::write(&path, record.encode())
                            .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
                        eprintln!("golden written: {}", path.display());
                    }
                    Mode::Check => {
                        let result = std::fs::read_to_string(&path)
                            .map_err(|error| format!("read: {error}"))
                            .and_then(|text| GoldenRecord::decode(&text))
                            .and_then(|expected| record.compare(&expected));
                        match result {
                            Ok(()) => eprintln!("golden ok: {}", path.display()),
                            Err(error) => self
                                .failures
                                .push(format!("{family}/{case} ({}): {error}", path.display())),
                        }
                    }
                }
            }

            pub(crate) fn finish(self) {
                assert!(
                    self.failures.is_empty(),
                    "DFlash2 golden mismatches:\n{}",
                    self.failures.join("\n")
                );
            }
        }
    }

    fn push_trace(record: &mut golden::GoldenRecord, prefix: &str, trace: &ArTrace) {
        for (cycle, logits) in trace.logits.iter().enumerate() {
            record.push(format!("{prefix}.logits[{cycle}]"), logits.iter().copied());
        }
        record.push_fingerprints(&format!("{prefix}.target"), &trace.target_cache);
        record.push_fingerprints(&format!("{prefix}.draft"), &trace.draft_cache);
        record.push(
            format!("{prefix}.context_tokens"),
            trace.context_tokens.iter().copied(),
        );
        record.push_u64(
            format!("{prefix}.frontier.attn"),
            trace.frontier.attn_tokens,
        );
        record.push(
            format!("{prefix}.frontier.recurrent"),
            trace
                .frontier
                .recurrent_tokens
                .map(|tokens| [1, tokens as u32, (tokens >> 32) as u32])
                .into_iter()
                .flatten(),
        );
        record.push_i32(
            format!("{prefix}.continuation_draft"),
            &trace.continuation_draft,
        );
    }

    fn reset_flat_fixture(inner: &mut Qwen35Inner) {
        inner.caches = Some(Qwen3_5LayerCache::fresh_layer_caches(&inner.config));
        inner.dflash2_context = None;
        inner.dflash2_turn_state = None;
    }

    fn tapped_prefill_with_capture(
        inner: &mut Qwen35Inner,
        warm: bool,
        plan: Option<&GdnCapturePlan<'_>>,
    ) -> Result<(Vec<(Vec<i64>, Vec<u32>)>, Vec<GdnCell>)> {
        let tap_layers = inner
            .dflash2
            .as_ref()
            .expect("fixture draft")
            .config
            .target_layers
            .clone();
        reset_flat_fixture(inner);
        let ids = |v: &[i32]| MxArray::from_int32(v, &[1, v.len() as i64]);
        if warm {
            super::super::model::forward_dflash2_with_taps(
                inner,
                &ids(&[3, 1, 4, 1, 5])?,
                &tap_layers,
                false,
                super::super::model::DFlash2LogitsSpan::LastRow,
                None,
                None,
            )?;
            crate::models::forward::eval_layer_caches(&inner.caches)?;
        }
        let chunk: Vec<i32> = (0..12).map(|i| (i * 5 + 2) % 16).collect();
        let (logits, taps, _, cells) = super::super::model::forward_dflash2_with_taps(
            inner,
            &ids(&chunk)?,
            &tap_layers,
            false,
            super::super::model::DFlash2LogitsSpan::LastRow,
            None,
            plan,
        )?;
        let mut arrays = vec![&logits];
        arrays.extend(taps.iter());
        for cache in inner.caches.as_ref().expect("fixture caches") {
            cache.collect_arrays(&mut arrays);
        }
        Ok((array_fingerprints(arrays)?, cells))
    }

    /// A capture plan on the tapped prefill (tiny fixture: every GDN layer on
    /// the chain route) leaves the logits, taps and every cache array bit
    /// for bit as without it (X0), cold and after a carried prefix. It
    /// yields one cell per row with a conv history and an f32 state per GDN
    /// layer. The window-end cell equals the live GDN slots but shares no
    /// buffer with them, and no cell changes when a later chunk runs.
    #[test]
    fn gdn_capture_plan_keeps_tapped_prefill_bits() -> Result<()> {
        let mut inner = tiny_dflash_inner(17);
        let rows = [1usize, 3, 9, 12];
        let plan = GdnCapturePlan { rows: &rows };
        for warm in [false, true] {
            let (plain, none) = tapped_prefill_with_capture(&mut inner, warm, None)?;
            assert!(none.is_empty());
            let (captured, cells) = tapped_prefill_with_capture(&mut inner, warm, Some(&plan))?;
            assert_eq!(captured, plain, "warm={warm}: outputs or caches moved");
            let live: Vec<(MxArray, MxArray)> = inner
                .caches
                .as_ref()
                .expect("fixture caches")
                .iter()
                .filter_map(|cache| match cache {
                    Qwen3_5LayerCache::Linear(arrays) => Some((
                        arrays.get(0).expect("conv slot").clone(),
                        arrays.get(1).expect("state slot").clone(),
                    )),
                    _ => None,
                })
                .collect();
            assert!(!live.is_empty());
            assert_eq!(cells.len(), rows.len());
            let mut before = Vec::new();
            for (cell, &row) in cells.iter().zip(&rows) {
                assert_eq!(cell.row, row);
                assert_eq!(cell.conv.len(), live.len());
                assert_eq!(cell.rec.len(), live.len());
                for (layer, ((conv, rec), (live_conv, live_rec))) in
                    cell.conv.iter().zip(&cell.rec).zip(&live).enumerate()
                {
                    assert_eq!(conv.shape()?.as_ref(), live_conv.shape()?.as_ref());
                    assert_eq!(rec.shape()?.as_ref(), live_rec.shape()?.as_ref());
                    assert_eq!(rec.dtype()?, DType::Float32);
                    MxArray::eval_arrays(&[conv, rec, live_conv, live_rec])?;
                    let shares = |a: &MxArray, b: &MxArray| unsafe {
                        mlx_sys::mlx_array_aliases(a.as_raw_ptr(), b.as_raw_ptr())
                    };
                    assert!(
                        !shares(conv, live_conv) && !shares(rec, live_rec),
                        "warm={warm} row={row} layer={layer}: cell shares a live GDN buffer"
                    );
                    if row == 12 {
                        assert_eq!(
                            array_fingerprints(vec![conv, rec])?,
                            array_fingerprints(vec![live_conv, live_rec])?,
                            "warm={warm} layer={layer}: window-end cell != live state"
                        );
                    }
                }
                let mut arrays: Vec<&MxArray> = cell.conv.iter().collect();
                arrays.extend(cell.rec.iter());
                before.push(array_fingerprints(arrays)?);
            }
            let tap_layers = inner
                .dflash2
                .as_ref()
                .expect("fixture draft")
                .config
                .target_layers
                .clone();
            super::super::model::forward_dflash2_with_taps(
                &mut inner,
                &MxArray::from_int32(&[9, 2, 6, 5, 3, 5, 8], &[1, 7])?,
                &tap_layers,
                false,
                super::super::model::DFlash2LogitsSpan::LastRow,
                None,
                None,
            )?;
            crate::models::forward::eval_layer_caches(&inner.caches)?;
            for (cell, before) in cells.iter().zip(before) {
                let mut arrays: Vec<&MxArray> = cell.conv.iter().collect();
                arrays.extend(cell.rec.iter());
                assert_eq!(
                    array_fingerprints(arrays)?,
                    before,
                    "warm={warm} row={}: cell changed after a later chunk",
                    cell.row
                );
            }
        }
        Ok(())
    }

    fn tiny_dflash_inner(seed: u64) -> Qwen35Inner {
        tiny_dflash_inner_with_attention_head_dim(seed, 32)
    }

    pub(crate) fn tiny_dflash_inner_with_attention_head_dim(
        seed: u64,
        head_dim: i32,
    ) -> Qwen35Inner {
        tiny_dflash_inner_with_config(seed, |config| config.head_dim = head_dim)
    }

    pub(crate) fn tiny_dflash_inner_with_config(
        seed: u64,
        adjust: impl FnOnce(&mut super::super::config::Qwen3_5Config),
    ) -> Qwen35Inner {
        unsafe { mlx_sys::mlx_seed(seed) };
        let mut inner = super::super::model::scheduled_mtp::seeded_inner_with_config(seed, adjust);
        inner.paged_adapter = None;

        // The shared scheduled fixture deliberately installs a constant head;
        // replace it so logits remain sensitive to hidden/cache divergence.
        let mut head = Linear::new(
            inner.config.hidden_size as u32,
            inner.config.vocab_size as u32,
            Some(false),
        )
        .expect("construct sensitive tiny LM head");
        head.set_weight(
            &MxArray::random_normal(
                &[
                    inner.config.vocab_size as i64,
                    inner.config.hidden_size as i64,
                ],
                0.0,
                0.02,
                Some(DType::BFloat16),
            )
            .expect("tiny LM head weights"),
        )
        .expect("install tiny LM head weights");
        inner.lm_head = Some(LinearProj::Standard(head));
        let draft = super::super::dflash2::tiny_dflash2_model_for_stepper_test(&inner.config)
            .expect("construct tiny DFlash2 companion");
        draft
            .validate_target(&inner.config)
            .expect("tiny DFlash2 companion must match target");
        inner.dflash2 = Some(draft);
        inner
    }

    fn run_retained_prefix_trace(inner: &mut Qwen35Inner, first_keep: usize) -> Result<ArTrace> {
        Ok(run_prefix_trace(inner, &[1, 2, 3, 4], None, first_keep)?.trace)
    }

    /// A prefix trace plus what only a window capture changes: the prefill's
    /// last-row logits, the draft windows after the prefill and after the
    /// forced cycles (live rows only: the buffer layout follows the append
    /// schedule), and the snapshot.
    struct PrefixRun {
        trace: ArTrace,
        prefill_logits: Vec<u32>,
        prefill_windows: Vec<(Vec<i64>, Vec<u32>)>,
        decode_windows: Vec<(Vec<i64>, Vec<u32>)>,
        snapshot: Option<DFlash2ContextCache>,
        /// `(first row, rows)` of every live draft-context append.
        appends: Vec<(usize, usize)>,
    }

    fn window_fingerprints(context: &DFlash2ContextCache) -> Result<Vec<(Vec<i64>, Vec<u32>)>> {
        let views = context.window_views();
        let mut arrays = Vec::new();
        for (keys, values) in views.iter().flatten() {
            arrays.extend([keys, values]);
        }
        array_fingerprints(arrays)
    }

    fn logits_bits(logits: &MxArray) -> Result<Vec<u32>> {
        let floats = logits.astype(DType::Float32)?;
        floats.eval();
        Ok(floats.to_float32()?.iter().map(|x| x.to_bits()).collect())
    }

    fn run_prefix_trace(
        inner: &mut Qwen35Inner,
        prompt: &[u32],
        capture_row: Option<usize>,
        first_keep: usize,
    ) -> Result<PrefixRun> {
        reset_flat_fixture(inner);
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut capture = capture_row.map(PrefillCapture::at);
        LIVE_DRAFT_APPENDS.with(|log| *log.borrow_mut() = Some(Vec::new()));
        let prefilled = dflash2_prefill(inner, prompt, 0, capture.as_mut(), stream);
        let appends = LIVE_DRAFT_APPENDS
            .with(|log| log.borrow_mut().take())
            .unwrap_or_default();
        let (prefill_logits, state) = prefilled?;
        let prefill_logits = logits_bits(&prefill_logits)?;
        let prefill_windows = window_fingerprints(&state.context)?;
        inner.dflash2_turn_state = Some(state);
        let block_size = inner
            .dflash2
            .as_ref()
            .expect("fixture draft")
            .config
            .block_size;
        let mut step = inner.begin_dspark_decode(block_size)?;
        // Force retained widths independently of model acceptance, including
        // keep=1 immediately followed by keep=8 and multiple draft-window wraps.
        let mut logits_trace = Vec::new();
        for (cycle, keep) in [first_keep, 1, 8, first_keep].into_iter().enumerate() {
            let ids = (0..8)
                .map(|row| (row + cycle + 1) as u32)
                .collect::<Vec<_>>();
            let logits = step.verify(&ids)?.logits;
            let floats = logits.astype(DType::Float32)?;
            floats.eval();
            logits_trace.push(floats.to_float32()?.iter().map(|x| x.to_bits()).collect());
            step.commit_with_provenance(keep, ids.len(), &ids)?;
        }
        let frontier = step.frontier().expect("fixture frontier");
        let mut target_arrays = Vec::new();
        for cache in step.inner.caches.as_ref().expect("fixture caches") {
            cache.collect_arrays(&mut target_arrays);
        }
        let target_cache = array_fingerprints(target_arrays)?;
        let mut draft_arrays = Vec::new();
        step.context.collect_arrays(&mut draft_arrays);
        let draft_cache = array_fingerprints(draft_arrays)?;
        let decode_windows = window_fingerprints(&step.context)?;
        let context_tokens = step.context.token_history().to_vec();
        step.finish()?;

        let context = inner
            .dflash2_context
            .take()
            .expect("retained draft context");
        let next_position = context.logical_len();
        inner.dflash2_turn_state = Some(DFlash2TurnState {
            context,
            next_position,
            kv: None,
        });
        let mut continuation = inner.begin_dspark_decode(block_size)?;
        let params = extract_chat_params(&ChatConfig {
            temperature: Some(0.0),
            repetition_penalty: Some(1.01),
            ..ChatConfig::default()
        });
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let proposal = continuation.propose(3, 2, &params, &mut rng)?;
        assert!(proposal.device_draft_ids.is_none());
        let continuation_draft = proposal.draft_ids;
        // An ordinary one-row verify must consume the committed state and
        // use its own compiled sequence-length key when supported.
        let logits = continuation.verify(&[3])?.logits;
        let floats = logits.astype(DType::Float32)?;
        floats.eval();
        logits_trace.push(floats.to_float32()?.iter().map(|x| x.to_bits()).collect());
        continuation.commit(1, 1)?;
        continuation.finish()?;
        Ok(PrefixRun {
            trace: ArTrace {
                logits: logits_trace,
                target_cache,
                draft_cache,
                context_tokens,
                frontier,
                continuation_draft,
            },
            prefill_logits,
            prefill_windows,
            decode_windows,
            snapshot: capture.and_then(|capture| capture.window),
            appends,
        })
    }

    /// Forced-keep cycles on the live state a whole turn left: one 8-row
    /// verify per entry of `keeps` (ids drawn from `ids_seed`), committed
    /// with that keep. With `continuation`, a proposal and a one-row verify
    /// follow, as in [`run_prefix_trace`]. The history then follows the
    /// committed rows, so the state stays a parkable live state.
    pub(crate) fn run_live_extension(
        inner: &mut Qwen35Inner,
        keeps: &[usize],
        ids_seed: u32,
        continuation: bool,
    ) -> Result<ArTrace> {
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let block_size = inner
            .dflash2
            .as_ref()
            .expect("fixture draft")
            .config
            .block_size;
        let context = inner.dflash2_context.take().expect("live draft context");
        let next_position = context.logical_len();
        inner.dflash2_turn_state = Some(DFlash2TurnState {
            context,
            next_position,
            kv: None,
        });
        let mut step = inner.begin_dspark_decode(block_size)?;
        let mut logits_trace = Vec::new();
        for (cycle, &keep) in keeps.iter().enumerate() {
            let ids = (0..8u32)
                .map(|row| (ids_seed + row * 5 + cycle as u32 * 3) % 15 + 1)
                .collect::<Vec<_>>();
            let logits = step.verify(&ids)?.logits;
            logits_trace.push(logits_bits(&logits)?);
            step.commit_with_provenance(keep, ids.len(), &ids)?;
        }
        let frontier = step.frontier().expect("live frontier");
        let mut target_arrays = Vec::new();
        for cache in step.inner.caches.as_ref().expect("live caches") {
            cache.collect_arrays(&mut target_arrays);
        }
        let target_cache = array_fingerprints(target_arrays)?;
        let mut draft_arrays = Vec::new();
        step.context.collect_arrays(&mut draft_arrays);
        let draft_cache = array_fingerprints(draft_arrays)?;
        let context_tokens = step.context.token_history().to_vec();
        step.finish()?;

        let mut continuation_draft = Vec::new();
        if continuation {
            let context = inner
                .dflash2_context
                .take()
                .expect("retained draft context");
            let next_position = context.logical_len();
            inner.dflash2_turn_state = Some(DFlash2TurnState {
                context,
                next_position,
                kv: None,
            });
            let mut step = inner.begin_dspark_decode(block_size)?;
            let mut rng = rand::rngs::StdRng::seed_from_u64(11);
            let params = extract_chat_params(&ChatConfig {
                temperature: Some(0.0),
                repetition_penalty: Some(1.01),
                ..ChatConfig::default()
            });
            let proposal = step.propose(3, 2, &params, &mut rng)?;
            assert!(proposal.device_draft_ids.is_none());
            continuation_draft = proposal.draft_ids;
            logits_trace.push(logits_bits(&step.verify(&[3])?.logits)?);
            step.commit(1, 1)?;
            step.finish()?;
        }
        inner.cached_token_history = inner
            .dflash2_context
            .as_ref()
            .expect("retained draft context")
            .token_history()
            .to_vec();
        Ok(ArTrace {
            logits: logits_trace,
            target_cache,
            draft_cache,
            context_tokens,
            frontier,
            continuation_draft,
        })
    }

    fn assert_snapshot_shape(
        snapshot: &DFlash2ContextCache,
        prompt: &[u32],
        row: usize,
        window_rows: usize,
        label: &str,
    ) {
        assert_eq!(snapshot.logical_len(), row as i32, "{label}: length");
        assert_eq!(
            snapshot.token_history(),
            &prompt[..row],
            "{label}: provenance"
        );
        for view in snapshot.window_views() {
            let (keys, values) = view.expect("snapshot window rows");
            assert_eq!(
                keys.shape_at(2).unwrap(),
                row.min(window_rows) as i64,
                "{label}"
            );
            assert_eq!(
                values.shape_at(2).unwrap(),
                row.min(window_rows) as i64,
                "{label}"
            );
        }
    }

    /// Fingerprints of rows `[from, from + rows)` of every window.
    fn window_rows_fingerprints(
        context: &DFlash2ContextCache,
        from: i64,
        rows: i64,
    ) -> Result<Vec<(Vec<i64>, Vec<u32>)>> {
        let mut arrays = Vec::new();
        for (keys, values) in context.window_views().into_iter().flatten() {
            arrays.push(keys.slice_axis(2, from, from + rows)?);
            arrays.push(values.slice_axis(2, from, from + rows)?);
        }
        array_fingerprints(arrays.iter().collect())
    }

    fn sliding_rows(inner: &Qwen35Inner) -> usize {
        inner
            .dflash2
            .as_ref()
            .expect("fixture draft")
            .config
            .sliding_window
            - 1
    }

    /// Arming a window capture at `R < N - window` needs draft rows an
    /// unarmed prefill only records; the live context still makes the
    /// unarmed appends, row for row, since append widths set draft K/V bits
    /// on the tiled quantized kernels. Every bit of the turn equals the run
    /// without a capture (X0): prefill logits, the live draft windows after
    /// prefill and after forced cycles, the full draft buffers, every
    /// per-cycle verify logit, the target caches, provenance, frontier and
    /// the continuation draft. The snapshot holds
    /// length R, provenance `[.., R)` and `min(R, window)` rows, and at
    /// `R = N` it is the prefill's own window bit for bit.
    #[test]
    fn armed_window_capture_keeps_turn_bits() -> Result<()> {
        for (label, mut inner) in [
            (
                "bf16_d64",
                tiny_dflash_inner_with_attention_head_dim(0xDFA5_C301, 64),
            ),
            ("int8_d256", tiny_int8_dflash_inner(0xDFA5_C302)),
        ] {
            let window_rows = sliding_rows(&inner);
            let prompt: Vec<u32> = (0..24).map(|i| (i * 7 + 3) % 15 + 1).collect();
            assert!(10 + window_rows < prompt.len(), "{label}: R < N - window");
            let plain = run_prefix_trace(&mut inner, &prompt, None, 3)?;
            assert!(plain.snapshot.is_none());
            assert_eq!(
                plain.appends,
                [(prompt.len() - window_rows, window_rows)],
                "{label}: the unarmed prefill appends only the newest window"
            );
            for row in [10usize, 3, 17, 20, 24] {
                let case = format!("{label} R={row}");
                let armed = run_prefix_trace(&mut inner, &prompt, Some(row), 3)?;
                assert_eq!(armed.appends, plain.appends, "{case}: append widths");
                assert_eq!(
                    armed.prefill_logits, plain.prefill_logits,
                    "{case}: prefill logits"
                );
                assert_eq!(
                    armed.prefill_windows, plain.prefill_windows,
                    "{case}: prefill windows"
                );
                assert_eq!(
                    armed.decode_windows, plain.decode_windows,
                    "{case}: decode windows"
                );
                assert_trace_eq(&armed.trace, &plain.trace, &case);
                let snapshot = armed.snapshot.expect("armed capture snapshot");
                assert_snapshot_shape(&snapshot, &prompt, row, window_rows, &case);
                if row == prompt.len() {
                    assert_eq!(
                        window_fingerprints(&snapshot)?,
                        plain.prefill_windows,
                        "{case}: snapshot at the prompt end"
                    );
                }
            }
            for row in [0usize, 25] {
                let unarmed = run_prefix_trace(&mut inner, &prompt, Some(row), 3)?;
                assert!(unarmed.snapshot.is_none(), "{label} R={row}: out of range");
                assert_eq!(unarmed.prefill_windows, plain.prefill_windows);
            }
        }
        Ok(())
    }

    /// Capture rows across the 2048-row prefill chunk edge and on a warm
    /// prefill whose capture gap is shorter than the window (so the
    /// snapshot reads the previous turn's window rows). The prefill keeps
    /// its live draft appends row for row, and its logits, target caches and
    /// live draft windows bit for bit, vs an unarmed prefill (X0). Each snapshot holds length R,
    /// provenance `[.., R)`, `min(R, window)` rows, and every row it shares
    /// with the final window, or with the previous turn's window, is
    /// bitwise that row.
    #[test]
    fn window_capture_across_prefill_chunks() -> Result<()> {
        let mut inner = tiny_dflash_inner_with_attention_head_dim(0xDFA5_C303, 64);
        let window_rows = sliding_rows(&inner);
        let w = window_rows as i64;
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prefill = |inner: &mut Qwen35Inner,
                       prompt: &[u32],
                       base: i32,
                       row: Option<usize>|
         -> Result<(
            (Vec<u32>, Vec<(usize, usize)>),
            Vec<(Vec<i64>, Vec<u32>)>,
            DFlash2ContextCache,
            Option<DFlash2ContextCache>,
        )> {
            let mut capture = row.map(PrefillCapture::at);
            LIVE_DRAFT_APPENDS.with(|log| *log.borrow_mut() = Some(Vec::new()));
            let prefilled = dflash2_prefill(inner, prompt, base, capture.as_mut(), stream);
            let appends = LIVE_DRAFT_APPENDS
                .with(|log| log.borrow_mut().take())
                .unwrap_or_default();
            let (logits, state) = prefilled?;
            let logits = logits_bits(&logits)?;
            let target =
                live_target_cache_fingerprints(inner.caches.as_ref().expect("fixture caches"))?;
            Ok((
                (logits, appends),
                target,
                state.context,
                capture.and_then(|capture| capture.window),
            ))
        };

        let prompt: Vec<u32> = (0..2052).map(|i| (i * 7 + 1) % 15 + 1).collect();
        let n = prompt.len() as i64;
        reset_flat_fixture(&mut inner);
        let (plain_logits, plain_target, plain_context, none) =
            prefill(&mut inner, &prompt, 0, None)?;
        assert!(none.is_none());
        let plain_windows = window_fingerprints(&plain_context)?;
        for row in [2052usize, 2050, 2048, 2046, 1000, 3] {
            let case = format!("cold R={row}");
            reset_flat_fixture(&mut inner);
            let (logits, target, context, snapshot) = prefill(&mut inner, &prompt, 0, Some(row))?;
            assert_eq!(logits, plain_logits, "{case}: logits, draft appends");
            assert_eq!(target, plain_target, "{case}: target caches");
            assert_eq!(
                window_fingerprints(&context)?,
                plain_windows,
                "{case}: windows"
            );
            let snapshot = snapshot.expect("armed capture snapshot");
            assert_snapshot_shape(&snapshot, &prompt, row, window_rows, &case);
            let r = row as i64;
            let shared = r - (n - w).max(r - w);
            if shared > 0 {
                assert_eq!(
                    window_rows_fingerprints(&snapshot, w - shared, shared)?,
                    window_rows_fingerprints(&context, 0, shared)?,
                    "{case}: rows shared with the final window"
                );
            }
        }

        let (first, second) = prompt[..40].split_at(30);
        let cold = |inner: &mut Qwen35Inner| -> Result<DFlash2ContextCache> {
            reset_flat_fixture(inner);
            let (_, _, context, _) = prefill(inner, first, 0, None)?;
            Ok(context)
        };
        let previous = cold(&mut inner)?;
        let previous_tail = window_rows_fingerprints(&previous, w - 3, 3)?;
        inner.dflash2_context = Some(previous);
        let (plain_logits, plain_target, plain_context, _) = prefill(&mut inner, second, 30, None)?;
        let plain_windows = window_fingerprints(&plain_context)?;
        let previous = cold(&mut inner)?;
        inner.dflash2_context = Some(previous);
        let (logits, target, context, snapshot) = prefill(&mut inner, second, 30, Some(34))?;
        assert_eq!(logits, plain_logits, "warm: logits, draft appends");
        assert_eq!(target, plain_target, "warm: target caches");
        assert_eq!(
            window_fingerprints(&context)?,
            plain_windows,
            "warm: windows"
        );
        let snapshot = snapshot.expect("armed warm capture snapshot");
        assert_snapshot_shape(&snapshot, &prompt[..40], 34, window_rows, "warm R=34");
        assert_eq!(
            window_rows_fingerprints(&snapshot, 0, 3)?,
            previous_tail,
            "warm: rows from the previous window"
        );
        assert_eq!(
            window_rows_fingerprints(&snapshot, w - 1, 1)?,
            window_rows_fingerprints(&context, 0, 1)?,
            "warm: row shared with the final window"
        );
        Ok(())
    }

    /// The compiled verify adds each layer's MLP delta inside the next
    /// layer's input norm (the final norm for the last layer) and takes every
    /// tap from those fused sums. Logits, every layer's tap and every GDN tape
    /// array must equal the eager verify (plain adds and norms) bit for bit.
    #[test]
    fn compiled_verify_matches_eager_verify_bitwise() -> Result<()> {
        if !crate::engine::persistence::compiled_forward_backend_available()
            || unsafe { mlx_sys::mlx_default_device() } != 1
        {
            eprintln!(
                "SKIP compiled_verify_matches_eager_verify_bitwise: Metal must be the default device"
            );
            return Ok(());
        }
        assert!(
            std::env::var_os("MLX_DISABLE_COMPILE").is_none(),
            "compiled verify parity requires MLX_DISABLE_COMPILE to be unset"
        );
        let mut inner = tiny_dflash_inner_with_attention_head_dim(0xDFA5_5E3A, 64);
        let tap_layers: Vec<usize> = (0..inner.layers.len()).collect();
        let run = |inner: &mut Qwen35Inner, ids: &[i32]| -> Result<Vec<(Vec<i64>, Vec<u32>)>> {
            reset_flat_fixture(inner);
            let stream = Stream::generation();
            let _ctx = StreamContext::new(stream);
            let (prefill_logits, state) = dflash2_prefill(inner, &[1, 2, 3, 4], 0, None, stream)?;
            prefill_logits.eval();
            inner.dflash2_turn_state = Some(state);
            let input = MxArray::from_int32(ids, &[1, ids.len() as i64])?;
            let (logits, taps, tape, _) = super::super::model::forward_dflash2_with_taps(
                inner,
                &input,
                &tap_layers,
                true,
                super::super::model::DFlash2LogitsSpan::All,
                None,
                None,
            )?;
            let mut arrays = vec![&logits];
            arrays.extend(taps.iter());
            for layer in tape.iter().flatten() {
                let k = &layer.kernel;
                arrays.extend([&k.q, &k.k, &k.v, &k.g, &k.beta, &layer.qkv]);
            }
            array_fingerprints(arrays)
        };
        for ids in [vec![5, 6, 7, 8, 9, 10, 11, 12], vec![9, 3, 14, 2, 6, 11]] {
            Qwen35Inner::take_dflash2_compiled_test_counts();
            inner.dflash2_compiled_verify_disabled = false;
            let compiled = run(&mut inner, &ids)?;
            assert_eq!(
                Qwen35Inner::take_dflash2_compiled_test_counts().0,
                1,
                "the compiled verify must run"
            );
            inner.dflash2_compiled_verify_disabled = true;
            let eager = run(&mut inner, &ids)?;
            assert_eq!(Qwen35Inner::take_dflash2_compiled_test_counts(), (0, 0));
            assert_eq!(
                compiled.len(),
                1 + tap_layers.len() + 6 * inner.layers.iter().filter(|l| l.is_linear()).count()
            );
            for (index, (a, b)) in compiled.iter().zip(eager.iter()).enumerate() {
                assert_eq!(a, b, "verify output {index} differs (rows {})", ids.len());
            }
        }
        Ok(())
    }

    #[test]
    fn compiled_verifier_state_and_continuation_repeat_for_every_keep() -> Result<()> {
        repeat_retained_prefix_trace(64)
    }

    #[test]
    fn unfused_attention_state_and_continuation_repeat_for_every_keep() -> Result<()> {
        repeat_retained_prefix_trace(32)
    }

    fn repeat_retained_prefix_trace(head_dim: i32) -> Result<()> {
        if !crate::engine::persistence::compiled_forward_backend_available()
            || unsafe { mlx_sys::mlx_default_device() } != 1
        {
            eprintln!("SKIP retained-prefix verifier regression: Metal must be the default device");
            return Ok(());
        }
        let require_compiled = head_dim == 64;
        if require_compiled {
            // Presence disables MLX compilation, including a value of "0".
            // This regression must fail explicitly rather than silently
            // validating an eager run under a compiled-test name.
            assert!(
                std::env::var_os("MLX_DISABLE_COMPILE").is_none(),
                "compiled verifier regression requires MLX_DISABLE_COMPILE to be unset"
            );
        }
        let mut inner = tiny_dflash_inner_with_attention_head_dim(0xDFA5_2203, head_dim);
        // D=64 uses fused vector attention and can reuse the shapeless verifier.
        // D=32 must stay eager: its unfused causal mask contains the current
        // prefix length. Reusing that trace caused intermittent state errors.
        for layer in &inner.layers {
            if let super::super::decoder_layer::AttentionType::Full(attention) = &layer.attn {
                for seq_len in [1, 8] {
                    assert_eq!(attention.verify_can_be_shapeless(seq_len)?, head_dim == 64);
                }
            }
        }
        Qwen35Inner::take_dflash2_compiled_test_counts();
        for keep in 1..=8 {
            let expected = run_retained_prefix_trace(&mut inner, keep)?;
            // Four eight-row verifies and one ordinary one-row verify each
            // invoke compilation. The first trace builds both sequence-length
            // keys; subsequent traces reuse them despite changing prefixes.
            let expected_counts = if require_compiled {
                (5, 2 * usize::from(keep == 1))
            } else {
                (0, 0)
            };
            assert_eq!(
                Qwen35Inner::take_dflash2_compiled_test_counts(),
                expected_counts,
                "initial trace must use the required verifier route: head_dim={head_dim} keep={keep}"
            );
            let actual = run_retained_prefix_trace(&mut inner, keep)?;
            assert_eq!(
                Qwen35Inner::take_dflash2_compiled_test_counts(),
                if require_compiled { (5, 0) } else { (0, 0) },
                "resetting caches must preserve compiled replay or eager-only routing: head_dim={head_dim} keep={keep}"
            );
            assert_trace_eq(
                &actual,
                &expected,
                &format!("repeated retained prefix head_dim={head_dim} keep={keep}"),
            );
        }
        Ok(())
    }

    /// The fixture at the production head dim (256) with its full-attention
    /// caches in the int8 format: the only geometry the int8 segmented
    /// kernels serve, so both the compiled and the eager verify read int8
    /// rows.
    pub(crate) fn tiny_int8_dflash_inner(seed: u64) -> Qwen35Inner {
        // The DFlash2 fixture needs hidden % head_dim == 0 (its conv groups);
        // the format lives in the caches (`reset_flat_fixture` rebuilds them
        // from the config).
        tiny_dflash_inner_with_config(seed, |config| {
            config.head_dim = 256;
            config.hidden_size = 256;
            config.kv_format = Some("int8".to_string());
        })
    }

    /// Int8 K/V: the compiled verify (int8 prefix + scales as graph inputs,
    /// the block quantized inside the graph) equals the eager verify (the
    /// cache quantizes on write) bit for bit — logits, taps and GDN tape —
    /// and the caches hold identical int8 rows and scales afterwards.
    #[test]
    fn compiled_int8_verify_matches_eager_int8_verify_bitwise() -> Result<()> {
        if !crate::engine::persistence::compiled_forward_backend_available()
            || unsafe { mlx_sys::mlx_default_device() } != 1
        {
            eprintln!("SKIP compiled_int8_verify: Metal must be the default device");
            return Ok(());
        }
        assert!(std::env::var_os("MLX_DISABLE_COMPILE").is_none());
        let mut inner = tiny_int8_dflash_inner(0xDFA5_1E78);
        let tap_layers: Vec<usize> = (0..inner.layers.len()).collect();
        let run = |inner: &mut Qwen35Inner, ids: &[i32]| -> Result<Vec<(Vec<i64>, Vec<u32>)>> {
            reset_flat_fixture(inner);
            for cache in inner.caches.as_ref().expect("caches") {
                if let Qwen3_5LayerCache::FullAttention(kv) = cache {
                    assert_eq!(
                        kv.format(),
                        KvFormat::Int8,
                        "fixture caches follow the config"
                    );
                }
            }
            let stream = Stream::generation();
            let _ctx = StreamContext::new(stream);
            let (prefill_logits, state) =
                dflash2_prefill(inner, &[1, 2, 3, 4, 5, 6], 0, None, stream)?;
            prefill_logits.eval();
            inner.dflash2_turn_state = Some(state);
            let input = MxArray::from_int32(ids, &[1, ids.len() as i64])?;
            let (logits, taps, tape, _) = super::super::model::forward_dflash2_with_taps(
                inner,
                &input,
                &tap_layers,
                true,
                super::super::model::DFlash2LogitsSpan::All,
                None,
                None,
            )?;
            let mut arrays = vec![&logits];
            arrays.extend(taps.iter());
            for layer in tape.iter().flatten() {
                let k = &layer.kernel;
                arrays.extend([&k.q, &k.k, &k.v, &k.g, &k.beta, &layer.qkv]);
            }
            let mut cache_arrays = Vec::new();
            for cache in inner.caches.as_ref().expect("caches") {
                if let Qwen3_5LayerCache::FullAttention(kv) = cache {
                    let view = kv.int8_view().expect("int8 cache allocated");
                    assert_eq!(view.len()?, 6 + ids.len() as i64);
                    cache_arrays.push(view);
                }
            }
            assert!(
                !cache_arrays.is_empty(),
                "the fixture has full-attention layers"
            );
            for view in &cache_arrays {
                arrays.extend(view.arrays());
            }
            array_fingerprints(arrays)
        };
        for ids in [vec![5, 6, 7, 8, 9, 10, 11, 12], vec![9, 3, 14, 2, 6, 11]] {
            Qwen35Inner::take_dflash2_compiled_test_counts();
            inner.dflash2_compiled_verify_disabled = false;
            let compiled = run(&mut inner, &ids)?;
            assert_eq!(
                Qwen35Inner::take_dflash2_compiled_test_counts().0,
                1,
                "the compiled int8 verify must run (rows {})",
                ids.len()
            );
            assert!(
                !inner.dflash2_compiled_verify_disabled,
                "the int8 compiled verify must not fall back"
            );
            inner.dflash2_compiled_verify_disabled = true;
            let eager = run(&mut inner, &ids)?;
            assert_eq!(Qwen35Inner::take_dflash2_compiled_test_counts(), (0, 0));
            assert_eq!(compiled.len(), eager.len());
            for (index, (a, b)) in compiled.iter().zip(eager.iter()).enumerate() {
                assert_eq!(a.0, b.0, "output {index} shape (rows {})", ids.len());
                assert_eq!(
                    a.1,
                    b.1,
                    "verify output {index} differs (rows {})",
                    ids.len()
                );
            }
        }
        Ok(())
    }

    /// Int8 K/V through the whole retained-prefix DFlash2 trace (eight-row
    /// verifies with every keep width, rewinds, draft-window wraps and a
    /// one-row continuation): the compiled shapeless tape is built once per
    /// sequence length and replays bit-identically as the int8 prefix grows.
    #[test]
    fn compiled_int8_verifier_state_and_continuation_repeat_for_every_keep() -> Result<()> {
        if !crate::engine::persistence::compiled_forward_backend_available()
            || unsafe { mlx_sys::mlx_default_device() } != 1
        {
            eprintln!(
                "SKIP int8 retained-prefix verifier regression: Metal must be the default device"
            );
            return Ok(());
        }
        assert!(std::env::var_os("MLX_DISABLE_COMPILE").is_none());
        let mut inner = tiny_int8_dflash_inner(0xDFA5_1E79);
        for layer in &inner.layers {
            if let super::super::decoder_layer::AttentionType::Full(attention) = &layer.attn {
                for seq_len in [1, 8] {
                    assert!(attention.verify_can_be_shapeless(seq_len)?);
                }
            }
        }
        Qwen35Inner::take_dflash2_compiled_test_counts();
        for keep in 1..=8 {
            let expected = run_retained_prefix_trace(&mut inner, keep)?;
            assert_eq!(
                Qwen35Inner::take_dflash2_compiled_test_counts(),
                (5, 2 * usize::from(keep == 1)),
                "int8 compiled verifier route: keep={keep}"
            );
            assert!(!inner.dflash2_compiled_verify_disabled);
            let actual = run_retained_prefix_trace(&mut inner, keep)?;
            assert_eq!(
                Qwen35Inner::take_dflash2_compiled_test_counts(),
                (5, 0),
                "int8 compiled replay must survive a cache reset: keep={keep}"
            );
            assert_trace_eq(
                &actual,
                &expected,
                &format!("repeated int8 retained prefix keep={keep}"),
            );
            // The trace's target-cache fingerprint covers int8 rows AND
            // scales (collect_arrays includes the scale buffers).
            let fa_layers = inner.config.full_attention_layer_count();
            assert!(
                expected.target_cache.len() >= 4 * fa_layers,
                "int8 caches contribute rows and scales to the fingerprint"
            );
        }
        Ok(())
    }

    const QWEN38_ROW_BYTES: u64 = 64 * 1024;

    /// Qwen3.8-27B text geometry (the DFlash2 production target):
    /// 64 layers (16 full-attention), 24 heads × 256, GDN 16 key heads /
    /// 48 value heads × 128, vocab 248 320.
    pub(crate) fn qwen38_27b_cfg() -> super::super::config::Qwen3_5Config {
        super::super::config::Qwen3_5Config {
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
            vocab_size: 248_320,
            hidden_size: 5120,
            num_layers: 64,
            num_heads: 24,
            num_kv_heads: 4,
            intermediate_size: 17_408,
            rms_norm_eps: 1e-6,
            head_dim: 256,
            tie_word_embeddings: false,
            attention_bias: false,
            max_position_embeddings: 262_144,
            pad_token_id: 0,
            eos_token_id: 0,
            bos_token_id: 0,
            linear_num_value_heads: 48,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.25,
            rope_theta: 10_000_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
            persist_paged_cache: None,
            n_mtp_layers: 0,
        }
    }

    /// Loaded draft `block_size` for the Qwen3.8-27B DFlash2 checkpoint
    /// (checkpoint block 8 → 7 proposals + the anchor = 8 verify rows).
    const QWEN38_DRAFT_BLOCK: usize = 7;

    #[test]
    fn cycle_transient_estimate_matches_the_documented_terms() {
        let cfg = qwen38_27b_cfg();
        let rows = 8u64;
        let logits = rows * 248_320 * 4 * 2;
        let conv_dim = 16 * 128 * 2 + 48 * 128;
        let tape_width = 5120 + conv_dim + 16 * 128 * 2 + 48 * 128;
        let tapes = 64 * rows * tape_width * 2;
        let scratch = rows * 3 * (5120 + 17_408) * 2;
        let core = logits + tapes + scratch;

        // 1 K prompt + 1 K generation: one-pass attention, no partials.
        let short = dflash2_cycle_transient(&cfg, QWEN38_DRAFT_BLOCK, 2048);
        assert_eq!(short.core_bytes, core);
        assert_eq!(short.partials_bytes, 0);
        assert_eq!(short.total(), core);
        // Tens of MiB — same order as the measured per-cycle churn and under
        // half the 128 MiB floor, so the floor is the short-prompt cap.
        assert!((40 << 20) < core && core < (64 << 20), "{core}");
        assert_eq!(
            dflash2_decode_cache_limit(&cfg, QWEN38_DRAFT_BLOCK, 2048),
            crate::cache_limit::DECODE_CACHE_LIMIT_FLOOR
        );

        // 6 K prompt (6219) + 128 new → 6347 keys: two-pass → 16
        // full-attention layers × 24 heads × 8 rows × 128 partitions
        // (ceil(6347 / 64) = 100 → next multiple of 32) × 256 × 2 B.
        let six_k = dflash2_cycle_transient(&cfg, QWEN38_DRAFT_BLOCK, 6347);
        assert_eq!(six_k.core_bytes, core);
        assert_eq!(six_k.partials_bytes, 16 * 24 * rows * 128 * 256 * 2);
        let cap = dflash2_decode_cache_limit(&cfg, QWEN38_DRAFT_BLOCK, 6347);
        assert_eq!(cap, 2 * (core + six_k.partials_bytes));
        // 466 MiB — the value the coordinator logged for this turn in
        // production; above the 340 MiB the unbounded pool holds at 6 K, so
        // the ceiling does not bind there (see `decode_cache_limit`).
        assert_eq!(cap >> 20, 466, "{cap}");
        // 32 K prompt (32 488) + 1024 new → 33 512 keys → 544 partitions:
        // 1714 MiB, likewise logged in production.
        assert_eq!(
            dflash2_decode_cache_limit(&cfg, QWEN38_DRAFT_BLOCK, 33_512) >> 20,
            1714
        );

        // Partition count is clamped: 1024 at huge contexts, 64 at the knee.
        assert_eq!(
            dflash2_cycle_transient(&cfg, QWEN38_DRAFT_BLOCK, 1 << 20).partials_bytes,
            16 * 24 * rows * 1024 * 256 * 2
        );
        assert_eq!(
            dflash2_cycle_transient(&cfg, QWEN38_DRAFT_BLOCK, 4096).partials_bytes,
            16 * 24 * rows * 64 * 256 * 2
        );
        assert_eq!(
            dflash2_cycle_transient(&cfg, QWEN38_DRAFT_BLOCK, 4095).partials_bytes,
            0
        );
    }

    #[test]
    fn cycle_transient_estimate_scales_with_rows_and_layers() {
        let cfg = qwen38_27b_cfg();
        let one = dflash2_cycle_transient(&cfg, QWEN38_DRAFT_BLOCK, 2048).core_bytes;
        // Twice the layers → tapes double, logits/scratch unchanged.
        let deeper = super::super::config::Qwen3_5Config {
            num_layers: 128,
            ..cfg.clone()
        };
        let two = dflash2_cycle_transient(&deeper, QWEN38_DRAFT_BLOCK, 2048).core_bytes;
        assert!(two > one && two < 2 * one, "{one} → {two}");
        // Bigger block → every term grows linearly in rows.
        let wide = dflash2_cycle_transient(&cfg, 15, 2048).core_bytes;
        assert_eq!(wide, one * 2);
        // Negative / zero dims are clamped, never wrap.
        let degenerate = super::super::config::Qwen3_5Config {
            vocab_size: -1,
            num_layers: 0,
            ..cfg
        };
        let scratch_only = dflash2_cycle_transient(&degenerate, QWEN38_DRAFT_BLOCK, 1 << 20);
        assert_eq!(scratch_only.core_bytes, 8 * 3 * (5120 + 17_408) * 2);
        assert_eq!(scratch_only.partials_bytes, 0, "no layers → no partials");
    }

    #[test]
    fn kv_row_bytes_counts_every_full_attention_layer_and_dtype() {
        assert_eq!(
            kv_row_bytes(16, 4, 256, DType::BFloat16, KvFormat::Bf16),
            QWEN38_ROW_BYTES
        );
        assert_eq!(
            kv_row_bytes(16, 4, 256, DType::Float32, KvFormat::Bf16),
            2 * QWEN38_ROW_BYTES
        );
        assert_eq!(kv_row_bytes(0, 4, 256, DType::BFloat16, KvFormat::Bf16), 0);
        // int8: one byte per element plus one fp32 scale per row, whatever
        // the activation dtype.
        assert_eq!(
            kv_row_bytes(16, 4, 256, DType::BFloat16, KvFormat::Int8),
            16 * 2 * 4 * (256 + 4)
        );
        assert_eq!(
            kv_row_bytes(16, 4, 256, DType::Float32, KvFormat::Int8),
            kv_row_bytes(16, 4, 256, DType::BFloat16, KvFormat::Int8)
        );
    }

    fn live_target_cache_fingerprints(
        caches: &[Qwen3_5LayerCache],
    ) -> Result<Vec<(Vec<i64>, Vec<u32>)>> {
        let mut arrays = Vec::new();
        for cache in caches {
            match cache {
                Qwen3_5LayerCache::FullAttention(kv) => {
                    let rows = kv.get_offset() as i64;
                    for buffer in [kv.keys_ref(), kv.values_ref()].into_iter().flatten() {
                        arrays.push(buffer.slice_axis(2, 0, rows)?);
                    }
                }
                Qwen3_5LayerCache::Linear(_) => {
                    let mut linear = Vec::new();
                    cache.collect_arrays(&mut linear);
                    arrays.extend(linear.into_iter().cloned());
                }
            }
        }
        array_fingerprints(arrays.iter().collect())
    }

    fn run_reserved_growth_trace(
        inner: &mut Qwen35Inner,
        reserve_rows: Option<i64>,
    ) -> Result<(ArTrace, Vec<i64>)> {
        reset_flat_fixture(inner);
        if let Some(rows) = reserve_rows {
            grow_dflash2_target_kv(&mut inner.caches, rows)?;
        }
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prompt = (0..250).map(|i| (i % 15 + 1) as u32).collect::<Vec<_>>();
        let (prefill_logits, state) = dflash2_prefill(inner, &prompt, 0, None, stream)?;
        prefill_logits.eval();
        inner.dflash2_turn_state = Some(state);
        let mut capacities = vec![min_fa_capacity(&inner.caches)?.expect("fixture FA caches")];
        let block_size = inner
            .dflash2
            .as_ref()
            .expect("fixture draft")
            .config
            .block_size;
        let mut step = inner.begin_dspark_decode(block_size)?;
        let mut logits_trace = Vec::new();
        // The first verify block writes rows 250..258 and so crosses the
        // unreserved 256-row buffer inside a verify commit.
        for (cycle, keep) in [8, 3, 8, 8, 1, 8, 5, 8].into_iter().enumerate() {
            let ids = (0..8)
                .map(|row| ((row + cycle) % 15 + 1) as u32)
                .collect::<Vec<_>>();
            let logits = step.verify(&ids)?.logits;
            let floats = logits.astype(DType::Float32)?;
            floats.eval();
            logits_trace.push(floats.to_float32()?.iter().map(|x| x.to_bits()).collect());
            step.commit_with_provenance(keep, ids.len(), &ids)?;
            capacities.push(min_fa_capacity(&step.inner.caches)?.expect("fixture FA caches"));
        }
        let frontier = step.frontier().expect("fixture frontier");
        let target_cache =
            live_target_cache_fingerprints(step.inner.caches.as_ref().expect("fixture caches"))?;
        let mut draft_arrays = Vec::new();
        step.context.collect_arrays(&mut draft_arrays);
        let draft_cache = array_fingerprints(draft_arrays)?;
        let context_tokens = step.context.token_history().to_vec();
        step.finish()?;

        let context = inner
            .dflash2_context
            .take()
            .expect("retained draft context");
        let next_position = context.logical_len();
        inner.dflash2_turn_state = Some(DFlash2TurnState {
            context,
            next_position,
            kv: None,
        });
        let mut continuation = inner.begin_dspark_decode(block_size)?;
        let params = extract_chat_params(&ChatConfig {
            temperature: Some(0.0),
            repetition_penalty: Some(1.01),
            ..ChatConfig::default()
        });
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let proposal = continuation.propose(3, 2, &params, &mut rng)?;
        let continuation_draft = proposal.draft_ids;
        continuation.finish()?;
        Ok((
            ArTrace {
                logits: logits_trace,
                target_cache,
                draft_cache,
                context_tokens,
                frontier,
                continuation_draft,
            },
            capacities,
        ))
    }

    fn assert_reserved_trace_is_bit_identical(head_dim: i32) -> Result<()> {
        if !crate::engine::persistence::compiled_forward_backend_available()
            || unsafe { mlx_sys::mlx_default_device() } != 1
        {
            eprintln!("SKIP reserved KV trace: Metal must be the default device");
            return Ok(());
        }
        let mut inner = tiny_dflash_inner_with_attention_head_dim(0xDFA5_2204, head_dim);
        Qwen35Inner::take_dflash2_compiled_test_counts();
        run_reserved_growth_trace(&mut inner, None)?;
        Qwen35Inner::take_dflash2_compiled_test_counts();
        let (reserved, reserved_capacities) = run_reserved_growth_trace(&mut inner, Some(1024))?;
        let reserved_counts = Qwen35Inner::take_dflash2_compiled_test_counts();
        let (grown, grown_capacities) = run_reserved_growth_trace(&mut inner, None)?;
        let grown_counts = Qwen35Inner::take_dflash2_compiled_test_counts();
        assert_trace_eq(
            &reserved,
            &grown,
            &format!("reserved vs grown head_dim={head_dim}"),
        );
        assert_eq!(
            reserved_counts, grown_counts,
            "reservation must not change verifier routing or traces: head_dim={head_dim}"
        );
        assert_eq!(
            grown_counts.0,
            if head_dim == 64 { 8 } else { 0 },
            "verifier route: head_dim={head_dim}"
        );
        assert!(
            reserved_capacities.iter().all(|&capacity| capacity == 1024),
            "reserved capacity must never change: {reserved_capacities:?}"
        );
        assert_eq!(grown_capacities.first(), Some(&256));
        assert_eq!(
            grown_capacities.last(),
            Some(&(250 + 256)),
            "the unreserved twin must grow inside decode"
        );
        Ok(())
    }

    #[test]
    fn reserved_target_kv_trace_is_bit_identical_across_growth_boundary_compiled() -> Result<()> {
        assert!(
            std::env::var_os("MLX_DISABLE_COMPILE").is_none(),
            "compiled verifier regression requires MLX_DISABLE_COMPILE to be unset"
        );
        assert_reserved_trace_is_bit_identical(64)
    }

    #[test]
    fn reserved_target_kv_trace_is_bit_identical_across_growth_boundary_eager() -> Result<()> {
        assert_reserved_trace_is_bit_identical(32)
    }

    /// Every verify, host or device, grows the target KV first to whole
    /// extents of the turn's plan, capped at its rows, so a verify never
    /// grows it by the cache's own 256-row step; the decode ceiling follows
    /// the grown size. Mutations: remove the `ensure_ahead` call in
    /// `verify_device` (or in `verify`) → red (capacity 506, off the
    /// extent); drop the `push_decode_limit` after a growth in
    /// `DFlash2DecodeKv::ensure_ahead` → red (ceiling sized for 256 of 280
    /// rows).
    #[test]
    fn stepper_ensures_before_every_verify() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        const EXTENT: i64 = 40;
        const CAP: i64 = 330;
        for device in [false, true] {
            let mut inner = tiny_dflash_inner_with_attention_head_dim(0xDFA5_2205, 64);
            reset_flat_fixture(&mut inner);
            let stream = Stream::generation();
            let _ctx = StreamContext::new(stream);
            let prompt = (0..250).map(|i| (i % 15 + 1) as u32).collect::<Vec<_>>();
            let (logits, state) = dflash2_prefill(&mut inner, &prompt, 0, None, stream)?;
            logits.eval();
            assert_eq!(min_fa_capacity(&inner.caches)?, Some(256));
            inner.dflash2_turn_state = Some(DFlash2TurnState {
                kv: Some(super::super::dflash2_turn::DFlash2KvPlan {
                    cap_rows: CAP,
                    extent_rows: EXTENT,
                }),
                ..state
            });
            let block_size = inner.dflash2.as_ref().expect("draft").config.block_size;
            let mut step = inner.begin_dspark_decode(block_size)?;
            let mut grew = false;
            for (cycle, keep) in [8, 3, 8, 8, 1, 8, 5, 8, 8, 2].into_iter().enumerate() {
                let ids = (0..8)
                    .map(|row| ((row + cycle) % 15 + 1) as u32)
                    .collect::<Vec<_>>();
                let frontier = step.frontier().expect("frontier").attn_tokens as i64;
                let logits = if device {
                    let ids = ids.iter().map(|&id| id as i32).collect::<Vec<_>>();
                    step.verify_device(&MxArray::from_int32(&ids, &[8])?)?
                        .logits
                } else {
                    step.verify(&ids)?.logits
                };
                logits.eval();
                let capacity = min_fa_capacity(&step.inner.caches)?.expect("FA caches");
                assert!(capacity >= frontier + 8, "device={device} cycle={cycle}");
                assert!(
                    capacity == 256 || capacity == CAP || capacity % EXTENT == 0,
                    "device={device} cycle={cycle}: capacity {capacity} is off the extent"
                );
                assert_eq!(
                    step.kv.decode_limit_rows(),
                    min_fa_reserved_rows(&step.inner.caches)?,
                    "device={device} cycle={cycle}: the decode ceiling lags the KV"
                );
                grew |= capacity != 256;
                step.commit_with_provenance(keep, ids.len(), &ids)?;
            }
            assert!(grew, "device={device}: the trace must grow the KV");
            step.finish()?;
        }
        Ok(())
    }

    #[test]
    fn context_budget_uses_the_smaller_target_or_draft_window() {
        let mut params = extract_chat_params(&ChatConfig {
            max_new_tokens: Some(64),
            ..ChatConfig::default()
        });
        let capacity = constrain_dflash2_context_params(10, 16, 12, None, &mut params)
            .expect("valid prompt is clamped");
        assert_eq!(capacity, 12);
        assert_eq!(params.max_new_tokens, 2);

        let mut too_long = extract_chat_params(&ChatConfig::default());
        let error = constrain_dflash2_context_params(13, 16, 12, None, &mut too_long)
            .expect_err("prompt beyond draft context must fail before prefill");
        assert!(error.reason.contains("effective active context is 12"));

        assert!(dflash2_final_token_fits_context(11, capacity));
        assert!(
            !dflash2_final_token_fits_context(12, capacity),
            "the final sampled token may be returned at capacity but must not be forwarded"
        );
    }

    /// Gate: the published window bounds the DFlash2 clamp, with Splash's
    /// `window - prompt` output budget. Mutation: ignore `published_window`
    /// in `constrain_dflash2_context_params` → red (capacity
    /// 262,144 ≠ 148,464).
    #[test]
    fn context_budget_is_bounded_by_the_published_window() {
        let mut params = extract_chat_params(&ChatConfig {
            max_new_tokens: Some(100_000),
            ..ChatConfig::default()
        });
        let capacity =
            constrain_dflash2_context_params(140_000, 262_144, 262_144, Some(148_464), &mut params)
                .expect("prompt inside the window");
        assert_eq!(capacity, 148_464);
        assert_eq!(params.max_new_tokens, 148_464 - 140_000);

        let mut at_window = extract_chat_params(&ChatConfig::default());
        let error = constrain_dflash2_context_params(
            148_464,
            262_144,
            262_144,
            Some(148_464),
            &mut at_window,
        )
        .expect_err("a prompt filling the window leaves no room to generate");
        assert!(
            error
                .reason
                .starts_with("context_length_exceeded: rendered prompt has 148464 tokens"),
            "{}",
            error.reason
        );
    }

    /// Mutation: drop the concat copy (`2 * prompt + chunk` → `prompt`) from
    /// the int8 prefill term of `dflash2_turn_transient_bytes` → red.
    #[test]
    fn int8_prefill_transient_counts_the_prefix_and_concat_copies() {
        // One query head and a small vocab keep the decode term below the
        // prefill term, so the int8 copies are what the estimate returns.
        let cfg = super::super::config::Qwen3_5Config {
            num_heads: 1,
            vocab_size: 1024,
            ..qwen38_27b_cfg()
        };
        let prompt = 72_000usize;
        let transient = |format| {
            dflash2_turn_transient_bytes(&cfg, QWEN38_DRAFT_BLOCK, prompt, prompt as u64, format, 0)
        };
        let scratch = 2048 * 3 * (5120 + 17_408) * 2;
        let copies = (2 * prompt as u64 + 2048) * 4 * 256 * 2 * 2;
        assert_eq!(transient(KvFormat::Int8), scratch + copies);
        assert!(transient(KvFormat::Bf16) < scratch + copies);
    }

    #[test]
    fn continuation_requires_matching_prefix_and_flat_frontier() {
        let tokens = (0..14).collect::<Vec<u32>>();
        let retained = tokens[..10].to_vec();
        assert_eq!(
            reusable_dflash2_prefix(true, &tokens, 10, 0, Some(&retained), Some(10), true),
            10
        );
        assert_eq!(
            reusable_dflash2_prefix(true, &tokens, 10, 0, None, Some(10), true),
            0
        );
        assert_eq!(
            reusable_dflash2_prefix(false, &tokens, 0, 10, Some(&retained), Some(10), true),
            10
        );
        assert_eq!(
            reusable_dflash2_prefix(false, &tokens, 0, 10, Some(&retained), Some(9), true),
            0
        );
    }

    #[test]
    fn same_length_different_prefix_and_paged_owner_are_cold_misses() {
        let tokens = (0..14).collect::<Vec<u32>>();
        let unrelated = (100..110).collect::<Vec<u32>>();
        assert_eq!(
            reusable_dflash2_prefix(true, &tokens, 10, 0, Some(&unrelated), Some(10), true,),
            0,
            "equal lengths do not establish cache provenance"
        );
        assert_eq!(
            reusable_dflash2_prefix(true, &tokens, 10, 0, Some(&tokens[..10]), Some(10), false,),
            0,
            "a paged-owned target frontier cannot reuse flat DFlash2 state"
        );
    }

    #[test]
    fn commit_provenance_validation_is_non_mutating_and_published_state_survives_consumption() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            eprintln!("skipping DFlash2 commit durability: Metal unavailable");
            return;
        }

        fn begin_verified_step(inner: &mut Qwen35Inner) -> Result<(Qwen35DFlash2Stepper<'_>, u32)> {
            reset_flat_fixture(inner);
            let stream = Stream::generation();
            let _ctx = StreamContext::new(stream);
            let (prefill_logits, state) = dflash2_prefill(inner, &[1, 2, 3, 4], 0, None, stream)?;
            prefill_logits.eval();
            let anchor = prefill_logits.argmax(-1, None)?.item_at_int32(0)? as u32;
            inner.dflash2_turn_state = Some(state);
            let block_size = inner
                .dflash2
                .as_ref()
                .expect("tiny companion")
                .config
                .block_size;
            let mut step = inner.begin_dspark_decode(block_size)?;
            step.verify(&[anchor])?.logits.eval();
            Ok((step, anchor))
        }

        let mut inner = tiny_dflash_inner(0xDFA5_2202);
        let params = extract_chat_params(&ChatConfig {
            temperature: Some(0.0),
            repetition_penalty: Some(1.01),
            ..ChatConfig::default()
        });
        // Establish a valid-commit reference on the same weights, consuming
        // its roots while the stepper still owns them. The actual trace below
        // resets caches and consumes roots only after finish transfers ownership.
        let (mut reference, expected_anchor) =
            begin_verified_step(&mut inner).expect("reference verify");
        reference.commit(1, 1).expect("reference commit");
        let mut reference_target_arrays = Vec::new();
        for cache in reference
            .inner
            .caches
            .as_ref()
            .expect("reference target caches")
        {
            cache.collect_arrays(&mut reference_target_arrays);
        }
        let expected_target =
            array_fingerprints(reference_target_arrays).expect("reference target state");
        let mut reference_draft_arrays = Vec::new();
        reference
            .context
            .collect_arrays(&mut reference_draft_arrays);
        let expected_draft =
            array_fingerprints(reference_draft_arrays).expect("reference draft state");
        let expected_tokens = reference.context.token_history().to_vec();
        let expected_frontier = reference.frontier();
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let expected_proposal = reference
            .propose(expected_anchor, 2, &params, &mut rng)
            .expect("reference continuation");
        assert!(expected_proposal.device_draft_ids.is_none());
        reference.finish().expect("publish reference context");

        let (mut step, anchor) = begin_verified_step(&mut inner).expect("actual verify");
        assert_eq!(anchor, expected_anchor);
        let assert_pending = |step: &Qwen35DFlash2Stepper<'_>| {
            assert!(
                step.snapshot.is_some(),
                "snapshot was consumed on validation error"
            );
            assert!(step.tape.is_some(), "tape was consumed on validation error");
            assert!(
                step.tapped.is_some(),
                "taps were consumed on validation error"
            );
            assert!(
                step.verified_ids.is_some(),
                "provenance was consumed on validation error"
            );
        };
        assert!(
            step.commit_with_provenance(1, 1, &[anchor, anchor])
                .is_err(),
            "wrong provenance length must fail"
        );
        assert_pending(&step);
        assert!(
            step.commit_with_provenance(1, 1, &[anchor.wrapping_add(1)])
                .is_err(),
            "host provenance mismatch must fail"
        );
        assert_pending(&step);
        step.verified_ids_device =
            Some(MxArray::from_int32(&[anchor as i32], &[1]).expect("duplicate device provenance"));
        assert!(
            step.commit_with_provenance(1, 1, &[anchor]).is_err(),
            "dual provenance ownership must fail"
        );
        assert_pending(&step);
        step.verified_ids_device = None;

        step.commit_with_provenance(1, 1, &[anchor])
            .expect("valid provenance commit");
        assert_eq!(step.frontier(), expected_frontier);
        step.finish().expect("publish committed context");

        let mut target_arrays = Vec::new();
        for cache in inner.caches.as_ref().expect("target caches after finish") {
            cache.collect_arrays(&mut target_arrays);
        }
        let context = inner
            .dflash2_context
            .as_ref()
            .expect("published draft context");
        let mut draft_arrays = Vec::new();
        context.collect_arrays(&mut draft_arrays);
        assert!(!target_arrays.is_empty() && !draft_arrays.is_empty());
        assert_eq!(context.token_history(), expected_tokens.as_slice());
        // Finish preserves lazy publication. Consuming these roots must still
        // reconstruct the exact committed state; pre-consumption availability
        // is intentionally not part of the normal stepper contract.
        assert_eq!(
            array_fingerprints(target_arrays).expect("consume target roots"),
            expected_target
        );
        assert_eq!(
            array_fingerprints(draft_arrays).expect("consume draft roots"),
            expected_draft
        );

        let context = inner.dflash2_context.take().expect("retained context");
        let next_position = context.logical_len();
        inner.dflash2_turn_state = Some(DFlash2TurnState {
            context,
            next_position,
            kv: None,
        });
        let block_size = inner
            .dflash2
            .as_ref()
            .expect("tiny companion")
            .config
            .block_size;
        let mut continuation = inner
            .begin_dspark_decode(block_size)
            .expect("resume published state");
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let actual_proposal = continuation
            .propose(anchor, 2, &params, &mut rng)
            .expect("continuation after published-root consumption");
        assert!(actual_proposal.device_draft_ids.is_none());
        assert_eq!(actual_proposal.draft_ids, expected_proposal.draft_ids);
        continuation.finish().expect("publish resumed context");
    }

    pub(crate) fn metal_is_default_device() -> bool {
        crate::engine::persistence::compiled_forward_backend_available()
            && unsafe { mlx_sys::mlx_default_device() } == 1
    }

    fn is_available(array: &MxArray) -> bool {
        unsafe { mlx_sys::mlx_array_is_available(array.as_raw_ptr()) }
    }

    fn greedy_params() -> crate::engine::params::ChatParams {
        extract_chat_params(&ChatConfig {
            temperature: Some(0.0),
            ..ChatConfig::default()
        })
    }

    /// Prefill `[1, 2, 3, 4]` and open a stepper. The caller must hold a
    /// `StreamContext` on the generation stream for the whole test, so
    /// `synchronize()` waits on the stream the stepper submits to.
    fn begin_seeded_step(
        inner: &mut Qwen35Inner,
        stream: Stream,
    ) -> Result<(Qwen35DFlash2Stepper<'_>, u32)> {
        reset_flat_fixture(inner);
        let (prefill_logits, state) = dflash2_prefill(inner, &[1, 2, 3, 4], 0, None, stream)?;
        let anchor = prefill_logits.argmax(-1, None)?;
        anchor.eval();
        let anchor = anchor.item_at_int32(0)? as u32;
        super::super::model::eval_layer_caches(&inner.caches)?;
        inner.dflash2_turn_state = Some(state);
        let block_size = inner
            .dflash2
            .as_ref()
            .expect("tiny companion")
            .config
            .block_size;
        Ok((inner.begin_dspark_decode(block_size)?, anchor))
    }

    #[test]
    fn device_proposal_is_submitted_before_verify() -> Result<()> {
        if !metal_is_default_device() {
            eprintln!("SKIP device proposal submission: Metal must be the default device");
            return Ok(());
        }
        let mut inner = tiny_dflash_inner(0xDFA5_2204);
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let (mut step, anchor) = begin_seeded_step(&mut inner, stream)?;
        let ids = std::iter::once(anchor).chain(1..8).collect::<Vec<_>>();
        step.verify(&ids)?.logits.eval();
        step.commit_with_provenance(5, ids.len(), &ids)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let proposal = step.propose(ids[4], 2, &greedy_params(), &mut rng)?;
        let device_ids = proposal
            .device_draft_ids
            .as_ref()
            .expect("greedy no-penalty proposal must stay device-resident");
        crate::array::synchronize();
        assert!(
            is_available(device_ids),
            "the device draft must be queued by propose, not by the verify eval"
        );
        Ok(())
    }

    #[test]
    fn eval_boundary_submits_draft_context_append() -> Result<()> {
        if !metal_is_default_device() {
            eprintln!("SKIP boundary draft-context submission: Metal must be the default device");
            return Ok(());
        }
        let mut inner = tiny_dflash_inner(0xDFA5_2205);
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let (mut step, anchor) = begin_seeded_step(&mut inner, stream)?;
        let ids = std::iter::once(anchor).chain(1..8).collect::<Vec<_>>();
        step.verify(&ids)?.logits.eval();
        step.commit_with_provenance(5, ids.len(), &ids)?;
        let token = MxArray::from_int32(&[ids[5] as i32], &[1])?;
        step.eval_boundary(&token);
        crate::array::synchronize();
        let mut draft_arrays = Vec::new();
        step.context.collect_arrays(&mut draft_arrays);
        assert!(!draft_arrays.is_empty());
        for (index, array) in draft_arrays.iter().enumerate() {
            assert!(
                is_available(array),
                "draft context root {index} was left for the next proposal eval"
            );
        }
        let mut target_arrays = Vec::new();
        crate::models::forward::CollectCacheArrays::collect_arrays(
            &step.inner.caches,
            &mut target_arrays,
        );
        assert!(!target_arrays.is_empty());
        for (index, array) in target_arrays.iter().enumerate() {
            assert!(is_available(array), "target cache root {index} not settled");
        }
        Ok(())
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum CycleScheduling {
        /// Draft, verify and the draft-context append share one eval; the
        /// boundary submits only the target caches.
        Deferred,
        /// The stepper's own propose/eval_boundary scheduling.
        Stepper,
    }

    fn run_forced_keep_cycles(
        inner: &mut Qwen35Inner,
        scheduling: CycleScheduling,
    ) -> Result<(Vec<Vec<i32>>, ArTrace)> {
        const DRAFT_LEN: usize = 7;
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let params = greedy_params();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (mut step, mut anchor) = begin_seeded_step(inner, stream)?;

        fn propose_device(
            step: &mut Qwen35DFlash2Stepper<'_>,
            scheduling: CycleScheduling,
            anchor: u32,
            params: &crate::engine::params::ChatParams,
            rng: &mut rand::rngs::StdRng,
        ) -> Result<MxArray> {
            match scheduling {
                CycleScheduling::Stepper => Ok(step
                    .propose(anchor, DRAFT_LEN, params, rng)?
                    .device_draft_ids
                    .expect("greedy no-penalty proposal must stay device-resident")),
                CycleScheduling::Deferred => {
                    let draft = step.inner.dflash2.as_ref().expect("tiny companion");
                    let (path, _) = draft.propose(
                        &step.inner.embedding,
                        step.inner.lm_head.as_ref(),
                        &mut step.context,
                        anchor,
                        DRAFT_LEN,
                        0.0,
                        true,
                        None,
                        rng,
                    )?;
                    match path {
                        super::super::dflash2::SelectorPath::Device(ids) => Ok(ids),
                        super::super::dflash2::SelectorPath::Host(_) => {
                            panic!("greedy device proposal expected")
                        }
                    }
                }
            }
        }

        let mut drafts = Vec::new();
        let mut logits_trace = Vec::new();
        // keep=1 exercises the single-row append, keep>=2 the multi-row one,
        // and a full keep wraps the sliding draft window.
        for keep in [1, 3, DRAFT_LEN + 1, 2, 5, 1] {
            let device_ids = propose_device(&mut step, scheduling, anchor, &params, &mut rng)?;
            let anchor_arr = MxArray::from_int32(&[anchor as i32], &[1])?;
            let verify_ids = MxArray::concatenate(&anchor_arr, &device_ids, 0)?;
            let logits = step.verify_device(&verify_ids)?.logits;
            let argmax = logits.argmax(-1, None)?;
            argmax.eval();
            let draft = device_ids.to_int32()?.as_ref().to_vec();
            let floats = logits.astype(DType::Float32)?;
            floats.eval();
            logits_trace.push(floats.to_float32()?.iter().map(|x| x.to_bits()).collect());
            let mut verified = vec![anchor];
            verified.extend(draft.iter().map(|&id| id as u32));
            step.commit_with_provenance(keep, verified.len(), &verified)?;
            anchor = argmax.item_at_int32(keep - 1)? as u32;
            let token = MxArray::from_int32(&[anchor as i32], &[1])?;
            match scheduling {
                CycleScheduling::Stepper => step.eval_boundary(&token),
                CycleScheduling::Deferred => {
                    super::super::model::async_eval_layer_caches(&step.inner.caches);
                    MxArray::async_eval_arrays(&[&token]);
                }
            }
            drafts.push(draft);
        }
        let frontier = step.frontier().expect("fixture frontier");
        let mut target_arrays = Vec::new();
        for cache in step.inner.caches.as_ref().expect("fixture caches") {
            cache.collect_arrays(&mut target_arrays);
        }
        let target_cache = array_fingerprints(target_arrays)?;
        let mut draft_arrays = Vec::new();
        step.context.collect_arrays(&mut draft_arrays);
        let draft_cache = array_fingerprints(draft_arrays)?;
        let context_tokens = step.context.token_history().to_vec();
        step.finish()?;

        let context = inner
            .dflash2_context
            .take()
            .expect("retained draft context");
        let next_position = context.logical_len();
        inner.dflash2_turn_state = Some(DFlash2TurnState {
            context,
            next_position,
            kv: None,
        });
        let block_size = inner
            .dflash2
            .as_ref()
            .expect("tiny companion")
            .config
            .block_size;
        let mut continuation = inner.begin_dspark_decode(block_size)?;
        let continuation_draft =
            propose_device(&mut continuation, scheduling, anchor, &params, &mut rng)?
                .to_int32()?
                .as_ref()
                .to_vec();
        continuation.finish()?;
        Ok((
            drafts,
            ArTrace {
                logits: logits_trace,
                target_cache,
                draft_cache,
                context_tokens,
                frontier,
                continuation_draft,
            },
        ))
    }

    fn assert_cycle_scheduling_is_bit_exact(head_dim: i32, seed: u64) -> Result<()> {
        if !metal_is_default_device() {
            eprintln!("SKIP cycle scheduling exactness: Metal must be the default device");
            return Ok(());
        }
        let mut inner = tiny_dflash_inner_with_attention_head_dim(seed, head_dim);
        let (expected_drafts, expected) =
            run_forced_keep_cycles(&mut inner, CycleScheduling::Deferred)?;
        let (actual_drafts, actual) = run_forced_keep_cycles(&mut inner, CycleScheduling::Stepper)?;
        assert_eq!(
            actual_drafts, expected_drafts,
            "head_dim={head_dim}: per-cycle draft ids"
        );
        assert_trace_eq(
            &actual,
            &expected,
            &format!("stepper scheduling head_dim={head_dim}"),
        );
        Ok(())
    }

    #[test]
    fn compiled_verify_cycle_scheduling_is_bit_exact() -> Result<()> {
        assert_cycle_scheduling_is_bit_exact(64, 0xDFA5_2206)
    }

    #[test]
    fn eager_verify_cycle_scheduling_is_bit_exact() -> Result<()> {
        // Seeds whose tiny drafts ignore the anchor would hide a wrong-anchor
        // proposal; this one does not.
        assert_cycle_scheduling_is_bit_exact(32, 0xDFA5_2208)
    }

    /// A cancel raised before the prefill keeps the live session: no Command
    /// reset and no prefill chunk.
    #[test]
    fn cancel_before_prefill_keeps_the_session_and_runs_no_chunk() {
        let mut inner = tiny_dflash_inner(0xDFA5_2209);
        reset_flat_fixture(&mut inner);
        ChatBackend::set_turn_cancel_flag(
            &mut inner,
            Some(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                true,
            ))),
        );

        let err = dflash2_prefill(&mut inner, &[1, 2, 3, 4], 0, None, Stream::generation())
            .err()
            .expect("a cancelled prefill must not run its first chunk");
        assert_eq!(err.reason, PREFILL_CANCELLED);
        assert_eq!(flat_attention_frontier(&inner), Some(0));

        reset_flat_fixture(&mut inner);
        inner.cached_token_history = vec![9, 9, 9];
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let config = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(1),
            ..ChatConfig::default()
        };
        let params = ChatBackend::resolve_params(&inner, &config);
        let thinking = ChatBackend::thinking_setup(&inner, &config);
        let mut args = WholeTurnArgs {
            tokens: &[1, 2, 3, 4],
            tokenizer: &tokenizer,
            eos_id: u32::MAX,
            config: &config,
            params: &params,
            thinking,
            plan: crate::engine::plan::TurnPlan {
                is_delta: false,
                input_media: crate::engine::plan::MediaCapabilities::NONE,
                context_media: crate::engine::plan::MediaCapabilities::NONE,
                use_paged_attention: false,
                decoder: crate::engine::plan::DecoderPlan::Speculative(
                    crate::engine::plan::SpeculativeKind::DraftModel,
                ),
            },
            sink: None,
            cancelled: None,
            media: crate::engine::plan::MediaInputs {
                images: &[],
                audio: &[],
            },
        };
        let err = match dflash2_chat_turn(&mut inner, &mut args) {
            Ok(_) => panic!("a turn cancelled before its prefill must not complete"),
            Err(e) => e,
        };
        assert_eq!(err.reason, PREFILL_CANCELLED);
        assert_eq!(inner.cached_token_history, [9, 9, 9], "no Command reset");
        assert_eq!(flat_attention_frontier(&inner), Some(0), "no prefill chunk");
    }

    /// A cancel raised before a prefix-hit delta prefill keeps the committed
    /// turn instead of reaching the prefill's fail-closed Command reset.
    #[test]
    fn cancel_before_delta_prefill_keeps_the_committed_turn() {
        let mut inner = tiny_dflash_inner(0xDFA5_220A);
        reset_flat_fixture(&mut inner);
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let config = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(1),
            ..ChatConfig::default()
        };
        let params = ChatBackend::resolve_params(&inner, &config);
        let plan = |is_delta| crate::engine::plan::TurnPlan {
            is_delta,
            input_media: crate::engine::plan::MediaCapabilities::NONE,
            context_media: crate::engine::plan::MediaCapabilities::NONE,
            use_paged_attention: false,
            decoder: crate::engine::plan::DecoderPlan::Speculative(
                crate::engine::plan::SpeculativeKind::DraftModel,
            ),
        };
        let run = |inner: &mut Qwen35Inner, tokens: &[u32], is_delta| {
            let mut args = WholeTurnArgs {
                tokens,
                tokenizer: &tokenizer,
                eos_id: u32::MAX,
                config: &config,
                params: &params,
                thinking: ChatBackend::thinking_setup(inner, &config),
                plan: plan(is_delta),
                sink: None,
                cancelled: None,
                media: crate::engine::plan::MediaInputs {
                    images: &[],
                    audio: &[],
                },
            };
            dflash2_chat_turn(inner, &mut args).map(|_| ())
        };

        run(&mut inner, &[1, 2, 3, 4], false).expect("uncancelled first turn");
        let history = inner.cached_token_history.clone();
        let frontier = flat_attention_frontier(&inner);
        assert_eq!(frontier, Some(history.len()));

        ChatBackend::set_turn_cancel_flag(
            &mut inner,
            Some(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                true,
            ))),
        );
        let mut next = history.clone();
        next.push(5);
        let err = run(&mut inner, &next, true).expect_err("cancelled delta turn");
        assert_eq!(err.reason, PREFILL_CANCELLED);
        assert_eq!(inner.cached_token_history, history, "no fail-closed reset");
        assert_eq!(
            flat_attention_frontier(&inner),
            frontier,
            "no prefill chunk"
        );
        assert!(inner.dflash2_context.is_some(), "draft context kept");
    }

    /// Gate: a DFlash2 turn clamps on its memory plan's window. A prompt
    /// that fills it is rejected before the prefill; one row less runs.
    /// Mutation: pass `None` for `target.dflash2_window()` in
    /// `dflash2_chat_turn` → red.
    #[test]
    fn dflash2_chat_turn_rejects_prompt_past_the_published_window() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let window =
            |resident| crate::engine::memory_plan::MemoryPlan::with_speculative_window(resident, 8);
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let mut inner = tiny_dflash_inner(0xDFA5_220B);
        reset_flat_fixture(&mut inner);
        inner.memory_plan = Some(window(4));
        assert_eq!(DFlash2Target::dflash2_window(&inner), Some(4));
        let err = match run_tiny_dflash_turn(&mut inner, &tokenizer, &[1, 2, 3, 4], false, 1) {
            Ok(_) => panic!("a prompt that fills the published window must be rejected"),
            Err(err) => err,
        };
        assert!(
            err.reason.starts_with("context_length_exceeded")
                && err.reason.contains("effective active context is 4 tokens"),
            "{}",
            err.reason
        );
        assert_eq!(flat_attention_frontier(&inner), Some(0), "no prefill chunk");

        inner.memory_plan = Some(window(5));
        let result = run_tiny_dflash_turn(&mut inner, &tokenizer, &[1, 2, 3, 4], false, 8)?;
        assert_eq!(result.num_tokens, 1, "the window leaves one output row");
        Ok(())
    }

    /// The greedy config [`run_tiny_dflash_turn`] runs a turn with.
    pub(crate) fn tiny_dflash_turn_config(reuse_cache: bool, max_new_tokens: i32) -> ChatConfig {
        ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(max_new_tokens),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            mtp_adaptive_depth: Some(false),
            reuse_cache: Some(reuse_cache),
            reasoning_effort: Some("none".to_string()),
            ..ChatConfig::default()
        }
    }

    pub(crate) fn run_tiny_dflash_turn(
        inner: &mut Qwen35Inner,
        tokenizer: &std::sync::Arc<crate::tokenizer::Qwen3Tokenizer>,
        tokens: &[u32],
        reuse_cache: bool,
        max_new_tokens: i32,
    ) -> Result<crate::engine::types::ChatResult> {
        use crate::engine::plan::{
            DecoderPlan, MediaCapabilities, MediaInputs, SpeculativeKind, TurnPlan,
        };
        let config = tiny_dflash_turn_config(reuse_cache, max_new_tokens);
        let params = ChatBackend::resolve_params(inner, &config);
        let thinking = ChatBackend::thinking_setup(inner, &config);
        let mut args = WholeTurnArgs {
            tokens,
            tokenizer,
            eos_id: u32::MAX,
            config: &config,
            params: &params,
            thinking,
            plan: TurnPlan {
                is_delta: false,
                input_media: MediaCapabilities::NONE,
                context_media: MediaCapabilities::NONE,
                use_paged_attention: false,
                decoder: DecoderPlan::Speculative(SpeculativeKind::DraftModel),
            },
            sink: None,
            cancelled: None,
            media: MediaInputs {
                images: &[],
                audio: &[],
            },
        };
        match dflash2_chat_turn(inner, &mut args)? {
            TurnOutput::Complete(result) => Ok(*result),
            TurnOutput::Streamed => panic!("sync DFlash2 turn returned streamed output"),
        }
    }

    #[test]
    fn dense_reuse_false_turn_leaves_no_retained_context() -> Result<()> {
        if !metal_is_default_device() {
            return Ok(());
        }
        let mut inner = tiny_dflash_inner(0xDFA5_3506);
        let hidden = inner.config.hidden_size as i64;
        let vocab = inner.config.vocab_size as i64;
        let mut head = Linear::new(hidden as u32, vocab as u32, Some(true))?;
        head.set_weight(&MxArray::zeros(&[vocab, hidden], Some(DType::BFloat16))?)?;
        let mut bias = vec![0.0f32; vocab as usize];
        bias[7] = 1.0;
        let bias = MxArray::from_float32(&bias, &[vocab])?.astype(DType::BFloat16)?;
        head.set_bias(Some(&bias))?;
        inner.lm_head = Some(LinearProj::Standard(head));

        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let first = run_tiny_dflash_turn(&mut inner, &tokenizer, &[1, 2, 3, 4], false, 1)?;
        assert_eq!(first.finish_reason, "length");
        assert_eq!(first.num_tokens, 1);
        assert_eq!(first.cached_tokens, 0);
        assert!(inner.dflash2_context.is_none());
        assert!(inner.cached_token_history.is_empty());

        // The fixed head makes turn 1's history [1, 2, 3, 4, 7]. Turn 2
        // strictly extends it and would reuse five tokens if the
        // reuse_cache=false turn had left its flat state behind.
        let second = run_tiny_dflash_turn(&mut inner, &tokenizer, &[1, 2, 3, 4, 7, 8], true, 1)?;
        assert_eq!(
            second.cached_tokens, 0,
            "a non-reused completion must leave no prefix for the next request"
        );
        Ok(())
    }

    fn prefix_turn_tokens(seed: u32) -> Vec<u32> {
        (0..20).map(|i| (seed + i * 3) % 15 + 1).collect()
    }

    /// A1 B1 A2 A3: A2 resumes on A1's parked end state across B1, A3 on
    /// the live state.
    #[test]
    fn prefix_event_reports_parked_end_then_live() -> Result<()> {
        use crate::models::qwen3_5::dflash2_state::trace::capture::capture_prefix_events;
        if !metal_is_default_device() {
            return Ok(());
        }
        let mut inner = tiny_dflash_inner(0xDFA5_3601);
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let mut a1 = Vec::new();
        let (results, events) = capture_prefix_events(|| -> Result<Vec<u32>> {
            let mut cached = Vec::new();
            run_tiny_dflash_turn(&mut inner, &tokenizer, &prefix_turn_tokens(0), true, 2)?;
            a1 = inner.cached_token_history.clone();
            run_tiny_dflash_turn(&mut inner, &tokenizer, &prefix_turn_tokens(1), true, 2)?;
            let a2: Vec<u32> = a1.iter().copied().chain([2, 3]).collect();
            cached.push(run_tiny_dflash_turn(&mut inner, &tokenizer, &a2, true, 2)?.cached_tokens);
            let a3: Vec<u32> = inner
                .cached_token_history
                .iter()
                .copied()
                .chain([4, 5])
                .collect();
            cached.push(run_tiny_dflash_turn(&mut inner, &tokenizer, &a3, true, 2)?.cached_tokens);
            Ok(cached)
        });
        let cached = results?;
        let field = |turn: usize, name: &str| events[turn][name].clone();
        assert_eq!(events.len(), 4, "one event per turn: {events:?}");
        assert_eq!(field(0, "miss_reason"), "cold_first");
        assert_eq!(field(1, "miss_reason"), "cold_first");
        assert_eq!(field(2, "hit_kind"), "E");
        assert_eq!(field(2, "miss_reason"), "none");
        assert_eq!(field(2, "shadow_hit"), "16");
        assert_eq!(field(2, "diverge_at"), "0");
        assert_eq!(field(2, "store_entries"), "1", "B1 parked by A2");
        assert_eq!(cached[0], a1.len() as u32);
        assert_eq!(field(2, "hit_tokens"), cached[0].to_string());
        assert_eq!(field(3, "hit_kind"), "live");
        assert_eq!(field(3, "miss_reason"), "none");
        assert!(cached[1] > 0);
        assert_eq!(field(3, "cached_tokens"), cached[1].to_string());
        assert_eq!(field(3, "hit_tokens"), cached[1].to_string());
        assert_eq!(
            field(3, "prefilled_tokens"),
            (field(3, "prompt_tokens").parse::<u32>().unwrap() - cached[1]).to_string()
        );
        Ok(())
    }

    /// A paged AR turn parks the DFlash2 end state it displaces, so the next
    /// DFlash2 turn on that lineage resumes on it.
    #[test]
    fn prefix_event_reports_the_end_state_a_paged_turn_parked() -> Result<()> {
        use crate::engine::plan::{DecoderPlan, MediaCapabilities, MediaInputs, TurnPlan};
        use crate::models::qwen3_5::dflash2_state::trace::capture::capture_prefix_events;
        if !metal_is_default_device() {
            return Ok(());
        }
        let mut inner = tiny_dflash_inner(0xDFA5_3602);
        inner.initialize_paged_adapter()?;
        assert!(inner.paged_adapter.is_some());
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let mut first = Vec::new();
        let (result, events) = capture_prefix_events(|| -> Result<u32> {
            run_tiny_dflash_turn(&mut inner, &tokenizer, &prefix_turn_tokens(0), true, 2)?;
            first = inner.cached_token_history.clone();
            let paged: Vec<u32> = inner
                .cached_token_history
                .iter()
                .copied()
                .chain([2, 3])
                .collect();
            let config = ChatConfig {
                temperature: Some(0.0),
                max_new_tokens: Some(2),
                reasoning_effort: Some("none".to_string()),
                ..ChatConfig::default()
            };
            let params = ChatBackend::resolve_params(&inner, &config);
            let thinking = ChatBackend::thinking_setup(&inner, &config);
            let mut args = WholeTurnArgs {
                tokens: &paged,
                tokenizer: &tokenizer,
                eos_id: u32::MAX,
                config: &config,
                params: &params,
                thinking,
                plan: TurnPlan {
                    is_delta: false,
                    input_media: MediaCapabilities::NONE,
                    context_media: MediaCapabilities::NONE,
                    use_paged_attention: true,
                    decoder: DecoderPlan::Autoregressive,
                },
                sink: None,
                cancelled: None,
                media: MediaInputs {
                    images: &[],
                    audio: &[],
                },
            };
            ChatBackend::run_paged_turn(&mut inner, &mut args)?;
            assert!(inner.paged_full_attn_caches_dirty);
            let next: Vec<u32> = inner
                .cached_token_history
                .iter()
                .copied()
                .chain([4, 5, 6])
                .collect();
            Ok(run_tiny_dflash_turn(&mut inner, &tokenizer, &next, true, 2)?.cached_tokens)
        });
        let cached = result?;
        assert_eq!(events.len(), 2, "one event per DFlash2 turn: {events:?}");
        assert_eq!(events[1]["hit_kind"], "E");
        assert_eq!(events[1]["miss_reason"], "none");
        assert_eq!(cached, first.len() as u32);
        assert!(
            !inner.paged_full_attn_caches_dirty,
            "the restore claims the lane"
        );
        Ok(())
    }

    /// Record the result and the retained flat state of one whole turn.
    pub(crate) fn push_turn_state(
        record: &mut golden::GoldenRecord,
        prefix: &str,
        result: &crate::engine::types::ChatResult,
        history: &[u32],
        target: Vec<&MxArray>,
        context: Option<&DFlash2ContextCache>,
    ) -> Result<()> {
        record.push(
            format!("{prefix}.result"),
            [
                result.num_tokens,
                result.cached_tokens,
                result.prompt_tokens,
            ],
        );
        record.push(format!("{prefix}.history"), history.iter().copied());
        record.push_fingerprints(&format!("{prefix}.target"), &array_fingerprints(target)?);
        let mut draft = Vec::new();
        if let Some(context) = context {
            context.collect_arrays(&mut draft);
            record.push(
                format!("{prefix}.context_tokens"),
                context.token_history().iter().copied(),
            );
        }
        record.push_fingerprints(&format!("{prefix}.draft"), &array_fingerprints(draft)?);
        Ok(())
    }

    /// Prompts of the golden whole-turn sequence: a cold turn, a turn that
    /// extends its saved history, then an unrelated prompt (prefix miss).
    pub(crate) fn golden_turn_prompts(history: &[u32], turn: usize) -> Vec<u32> {
        match turn {
            0 => vec![1, 2, 3, 4, 5, 6],
            1 => history.iter().copied().chain([9, 10]).collect(),
            _ => vec![3, 1, 4, 1, 5],
        }
    }

    #[test]
    fn dflash2_golden_traces() -> Result<()> {
        let Some(mut goldens) = golden::GoldenRun::from_env() else {
            return Ok(());
        };
        assert!(
            metal_is_default_device(),
            "DFlash2 goldens are recorded on Metal as the default device"
        );
        assert!(
            std::env::var_os("MLX_DISABLE_COMPILE").is_none(),
            "DFlash2 goldens are recorded with MLX compilation enabled"
        );
        const FAMILY: &str = "qwen3_5";

        for (case, mut inner) in [
            (
                "retained_prefix_bf16_d64",
                tiny_dflash_inner_with_attention_head_dim(0xDFA5_2203, 64),
            ),
            (
                "retained_prefix_bf16_d32",
                tiny_dflash_inner_with_attention_head_dim(0xDFA5_2203, 32),
            ),
            (
                "retained_prefix_int8_d256",
                tiny_int8_dflash_inner(0xDFA5_1E79),
            ),
        ] {
            let mut record = golden::GoldenRecord::default();
            for keep in 1..=8 {
                let trace = run_retained_prefix_trace(&mut inner, keep)?;
                push_trace(&mut record, &format!("keep{keep}"), &trace);
            }
            goldens.case(FAMILY, case, &record);
        }

        // A cold prefill with a window capture armed at R < N - window: the
        // snapshot reads draft rows the live schedule only records.
        for (case, mut inner) in [
            (
                "capture_armed_bf16_d64",
                tiny_dflash_inner_with_attention_head_dim(0xDFA5_C304, 64),
            ),
            (
                "capture_armed_int8_d256",
                tiny_int8_dflash_inner(0xDFA5_C305),
            ),
        ] {
            let prompt: Vec<u32> = (0..24).map(|i| (i * 5 + 2) % 15 + 1).collect();
            let run = run_prefix_trace(&mut inner, &prompt, Some(10), 3)?;
            let mut record = golden::GoldenRecord::default();
            push_trace(&mut record, "trace", &run.trace);
            let snapshot = run.snapshot.expect("armed capture snapshot");
            record.push_fingerprints("snapshot", &window_fingerprints(&snapshot)?);
            record.push(
                "snapshot.tokens".to_string(),
                snapshot.token_history().iter().copied(),
            );
            goldens.case(FAMILY, case, &record);
        }

        for (head_dim, seed) in [(64, 0xDFA5_2206), (32, 0xDFA5_2208)] {
            let mut inner = tiny_dflash_inner_with_attention_head_dim(seed, head_dim);
            let (drafts, trace) = run_forced_keep_cycles(&mut inner, CycleScheduling::Stepper)?;
            let mut record = golden::GoldenRecord::default();
            for (cycle, draft) in drafts.iter().enumerate() {
                record.push_i32(format!("draft[{cycle}]"), draft);
            }
            push_trace(&mut record, "trace", &trace);
            goldens.case(FAMILY, &format!("forced_keep_d{head_dim}"), &record);
        }

        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        for (case, mut inner) in [
            (
                "chat_turns_bf16_d64",
                tiny_dflash_inner_with_attention_head_dim(0xDFA5_3504, 64),
            ),
            (
                "chat_turns_bf16_d32",
                tiny_dflash_inner_with_attention_head_dim(0xDFA5_3504, 32),
            ),
            ("chat_turns_int8_d256", tiny_int8_dflash_inner(0xDFA5_3505)),
        ] {
            let mut record = golden::GoldenRecord::default();
            for turn in 0..3 {
                let prompt = golden_turn_prompts(&inner.cached_token_history, turn);
                let result = run_tiny_dflash_turn(&mut inner, &tokenizer, &prompt, true, 12)?;
                let mut target = Vec::new();
                for cache in inner.caches.iter().flatten() {
                    cache.collect_arrays(&mut target);
                }
                push_turn_state(
                    &mut record,
                    &format!("turn{turn}"),
                    &result,
                    &inner.cached_token_history,
                    target,
                    inner.dflash2_context.as_ref(),
                )?;
            }
            goldens.case(FAMILY, case, &record);
        }

        goldens.finish();
        Ok(())
    }
}
