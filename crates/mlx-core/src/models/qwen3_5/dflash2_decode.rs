//! DFlash2 whole-turn integration for dense Qwen3.8.

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::engine::backend::{
    DsparkBackend, DsparkProposal, DsparkStepper, DsparkTreeGreedyResult, DsparkVerifyOutput,
};
use crate::engine::dspark_timeline::{self as timeline, Mark};
use crate::engine::memory_plan::EXTENT_TARGET_BYTES;

use super::dflash2::DFlash2ContextCache;
use super::dflash2_turn::{DFlash2Target, DFlash2TurnState};
use super::gdn_blob::GdnStateBlobs;
use super::model::{PREFILL_STEP_SIZE, Qwen35Inner};

pub(crate) struct Qwen35DFlash2Stepper<'a> {
    inner: &'a mut Qwen35Inner,
    context: DFlash2ContextCache,
    next_position: i32,
    tap_layers: Vec<usize>,
    tape: Option<Vec<Option<super::gated_delta_net::GdnLayerTape>>>,
    tapped: Option<Vec<MxArray>>,
    verified_ids: Option<Vec<u32>>,
    verified_device_ids: Option<MxArray>,
    /// The lane's two GDN cells and pending tape: each verify folds the
    /// last commit's retained rows, each commit swaps parity. Lent to the
    /// target across the verify's claim ([`lend_lane`]).
    gdn_blobs: Option<GdnStateBlobs>,
    flat_snapshot: Option<super::layer_cache::Qwen35FlatSnapshot>,
    /// CUDA flat-verify deferred GDN fold slots (one per layer). `Some` when
    /// `eager_flat && cuda_ops_available()` and every linear layer satisfies
    /// `mlx_gdn_fused_step_deferred_cuda`'s contract; each cycle's commit
    /// records `(tape, keep)` here so the next verify folds it in-register
    /// instead of launching a separate 48-layer replay pass on commit.
    cuda_pending_gdn: Option<Vec<Option<super::layer_cache::CudaPendingGdn>>>,
    eager_flat: bool,
    tree_enabled: bool,
    tree_max_nodes: usize,
    lookup_enabled: bool,
    lookup_index: super::verify_tree::LookupIndex,
    lookup_acceptance: super::verify_tree::LookupAcceptance,
    verify_budget: super::verify_tree::VerifyBudget,
    last_tree_is_chain: bool,
    last_tree_len: usize,
    paged: PagedVerifyLane,
}

/// A DFlash2 lane's two GDN cells with their pending rows.
pub(crate) struct DFlash2Lane {
    gdn: GdnStateBlobs,
}

/// Splash `Runtime::suspend` → `QwenStateStorage::releaseLane`: across a
/// verify's claim the lane sits on the target, so the vacate of a suspend
/// drops its GDN cells and gives its draft ring to the idle lane before the
/// wait and the replay. Returns the cells and the ring, `empty` in its place.
pub(crate) fn lend_lane(
    gdn: &mut Option<GdnStateBlobs>,
    ring: &mut DFlash2ContextCache,
    empty: DFlash2ContextCache,
) -> (Option<DFlash2Lane>, DFlash2ContextCache) {
    #[cfg(test)]
    lane_probe::record(gdn.as_ref(), ring);
    let ring = std::mem::replace(ring, empty);
    (gdn.take().map(|gdn| DFlash2Lane { gdn }), ring)
}

/// The cells and the ring the claim left on the target, when no suspend
/// took them.
pub(crate) fn reclaim_lane(
    lent: Option<DFlash2Lane>,
    ring: Option<DFlash2ContextCache>,
    gdn: &mut Option<GdnStateBlobs>,
    context: &mut DFlash2ContextCache,
) {
    if let Some(lane) = lent {
        *gdn = Some(lane.gdn);
    }
    if let Some(ring) = ring {
        *context = ring;
    }
}

pub(crate) fn missing_cells() -> Error {
    Error::from_reason("DFlash2 lane has no GDN cells")
}

/// A verify failure the next paged verify reports after its claim, as MLX
/// reports a failed command at the accept's wait.
#[cfg(test)]
pub(crate) mod verify_fault {
    use std::cell::RefCell;

    thread_local! {
        static NEXT: RefCell<Option<String>> = const { RefCell::new(None) };
    }

    pub(crate) fn set(reason: &str) {
        NEXT.with(|next| *next.borrow_mut() = Some(reason.to_string()));
    }

    pub(crate) fn take() -> Option<String> {
        NEXT.with(|next| next.borrow_mut().take())
    }
}

#[cfg(test)]
pub(crate) mod lane_probe {
    use std::cell::RefCell;
    use std::sync::{Arc, Weak};

    use crate::array::{MxArray, MxHandle};
    use crate::models::qwen3_5::dflash2::DFlash2ContextCache;
    use crate::models::qwen3_5::gdn_blob::GdnStateBlobs;

    type Weaks = Vec<Weak<MxHandle>>;

    thread_local! {
        static LENT: RefCell<(Weaks, Weaks)> = const { RefCell::new((Vec::new(), Vec::new())) };
    }

    fn weak(arrays: &[&MxArray]) -> Weaks {
        arrays
            .iter()
            .map(|array| Arc::downgrade(&array.handle))
            .collect()
    }

    /// The GDN cells and the draft ring the last verify lent.
    pub(crate) fn record(gdn: Option<&GdnStateBlobs>, ring: &DFlash2ContextCache) {
        let cells = gdn.map_or_else(Vec::new, |gdn| weak(&gdn.lane_arrays()));
        let mut arrays = Vec::new();
        ring.collect_arrays(&mut arrays);
        let ring = weak(&arrays);
        LENT.with(|lent| *lent.borrow_mut() = (cells, ring));
    }

    /// (recorded, still held), for the cells and for the ring.
    pub(crate) fn held() -> ((usize, usize), (usize, usize)) {
        let count = |weaks: &Weaks| {
            let alive = weaks.iter().filter(|weak| weak.strong_count() > 0).count();
            (weaks.len(), alive)
        };
        LENT.with(|lent| {
            let lent = lent.borrow();
            (count(&lent.0), count(&lent.1))
        })
    }
}

/// The stepper's verify on the paged pool: one 8-row graph per cycle (Splash
/// `Runtime::decodeAsync`), the rows booked before the write, rolled back
/// past the keep after it.
pub(crate) struct PagedVerifyLane {
    pub(crate) seq: u32,
    pub(crate) context_window: u32,
    pub(crate) open_cycle: Option<crate::engine::spec_paged::VerifyTicket>,
    /// The cycle's whole draft: the proposals past the driver's cap are the
    /// verify's tail rows.
    pub(crate) drafted: Option<DraftedProposals>,
    pub(crate) _decode_limit: crate::cache_limit::DecodeCacheLimitGuard,
}

pub(crate) enum DraftedProposals {
    Host(Vec<u32>),
    Device(MxArray),
}

impl DraftedProposals {
    pub(crate) fn host(self) -> Result<Vec<u32>> {
        match self {
            Self::Host(ids) => Ok(ids),
            Self::Device(ids) => Ok(ids.to_int32()?.iter().map(|&id| id as u32).collect()),
        }
    }
}

/// The 8 verify ids: `ids` (anchor + the driver's proposals) then the rest of
/// the cycle's draft.
pub(crate) fn full_verify_ids(ids: &[u32], drafted: &[u32]) -> Result<Vec<u32>> {
    use super::dflash2::DFLASH2_VERIFY_ROWS;
    let proposals = ids.len().saturating_sub(1);
    if ids.is_empty() || ids.len() > DFLASH2_VERIFY_ROWS || drafted.len() + 1 < DFLASH2_VERIFY_ROWS
    {
        return Err(Error::from_reason(format!(
            "DFlash2 verify of {} ids over a draft of {}",
            ids.len(),
            drafted.len()
        )));
    }
    let mut full = ids.to_vec();
    full.extend_from_slice(&drafted[proposals..DFLASH2_VERIFY_ROWS - 1]);
    Ok(full)
}

/// [`full_verify_ids`] on the device: the driver's `[1+L]` ids then the rest
/// of the cycle's device draft.
pub(crate) fn full_device_verify_ids(ids: &MxArray, drafted: &MxArray) -> Result<MxArray> {
    use super::dflash2::DFLASH2_VERIFY_ROWS;
    let rows = if ids.ndim()? == 1 {
        ids.shape_at(0)?
    } else {
        -1
    };
    let drafted_rows = if drafted.ndim()? == 1 {
        drafted.shape_at(0)?
    } else {
        -1
    };
    let verify_rows = DFLASH2_VERIFY_ROWS as i64;
    if rows < 1 || rows > verify_rows || drafted_rows + 1 < verify_rows {
        return Err(Error::from_reason(format!(
            "DFlash2 verify of {rows} device ids over a device draft of {drafted_rows}"
        )));
    }
    if rows == verify_rows {
        return Ok(ids.clone());
    }
    let tail = drafted.slice_axis(0, rows - 1, verify_rows - 1)?;
    MxArray::concatenate(ids, &tail, 0)
}

/// What a cycle's verify embeds.
enum VerifyIds<'a> {
    Host(&'a [u32]),
    /// The driver's `[1+L]` device ids over the cycle's device draft.
    Device(&'a MxArray),
}

/// The 8 verify rows of one cycle, as built.
enum CycleInput {
    Host {
        ids: Vec<u32>,
        full: Vec<u32>,
    },
    Device {
        ids: MxArray,
        drafted: MxArray,
        full: MxArray,
    },
}

#[cfg(test)]
pub(crate) mod verify_order {
    use std::cell::Cell;

    thread_local! {
        /// The verify reads device ids back before its graph build (no
        /// overlap), the reference of the overlap tests.
        pub(crate) static READ_BACK_FIRST: Cell<bool> = const { Cell::new(false) };
    }
}

/// Blocks one growth adds to a runway paged pool: the whole blocks nearest
/// Splash's extent target, at least one.
pub(crate) fn runway_pool_extent_blocks(bytes_per_block: u64) -> u32 {
    let bytes_per_block = bytes_per_block.max(1);
    ((EXTENT_TARGET_BYTES + bytes_per_block / 2) / bytes_per_block).clamp(1, u64::from(u32::MAX))
        as u32
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
/// peak: the last prefill chunk's scratch with the in-place attention's
/// workspace bound, or one decode step. With a DFlash2 draft (`draft_taps`
/// tapped layers) the prefill also holds the chunk's taps, and the decode
/// step is the 8-row verify on the pool with the free pool its decode cache
/// limit keeps, the prefill-sized workspace the adapter holds and the
/// pending GDN tape the next verify folds (Splash prices the one 8-row
/// verify graph and `DecodeTensor::GdnTape`). With `lanes` DFlash2 lanes
/// verifying in one decode step, each lane past the first adds
/// [`pool_verify_lane_bytes`].
pub(crate) fn pool_lane_transient(
    config: &super::config::Qwen3_5Config,
    draft_taps: Option<usize>,
    window: u32,
    lanes: u32,
) -> u64 {
    let lanes = lanes.max(1);
    let dim = |v: i32| v.max(0) as u64;
    let window = u64::from(window);
    let chunk = window.min(PREFILL_STEP_SIZE as u64);
    let prefill = prefill_chunk_scratch_bytes(config, window).saturating_add(pool_attention_bytes(
        config,
        chunk,
        mlx_paged_attn::prefill_plan::Pass::Prefill,
    ));
    match draft_taps {
        Some(taps) => {
            let tapped = chunk
                .saturating_mul(taps as u64)
                .saturating_mul(dim(config.hidden_size))
                .saturating_mul(2);
            let cycle = pool_verify_cycle_bytes(config, window);
            // The adapter's one workspace keeps the prefill bound through
            // every verify (Splash `fixedRuntimeBytes` prices the prefill
            // arena's `PrefillTensor::AttentionPartials` beside the decode
            // arena's). It is live, not cache: outside `decode_cache_limit`.
            let held = pool_held_workspace_bytes(config);
            let pending = gdn_pending_tape_bytes(config);
            let others = pool_verify_lane_bytes(config).saturating_mul(u64::from(lanes - 1));
            prefill.saturating_add(tapped).max(
                cycle
                    .saturating_add(held)
                    .saturating_add(pending)
                    .saturating_add(crate::cache_limit::decode_cache_limit(cycle))
                    .saturating_add(others),
            )
        }
        None => prefill.max(dflash2_cycle_transient(config, 0, window).total()),
    }
}

/// Bytes one more DFlash2 lane's verify holds in a decode step: its 8 rows'
/// logits, GDN tape and one layer's scratch, from its verify (alone, or in
/// a wave whose activations stack per lane) to the step's commit. The
/// verify attention's workspace is the adapter's one, reused per launch.
pub(crate) fn pool_verify_lane_bytes(config: &super::config::Qwen3_5Config) -> u64 {
    dflash2_cycle_transient(config, super::dflash2::DFLASH2_VERIFY_ROWS - 1, 0).core_bytes
}

/// One decode step of `rows` target rows over `window` keys, the rows of
/// every lane in the step together: logits, GDN tapes, one layer's scratch
/// and the attention partials. One row is the single-lane AR step
/// [`pool_lane_transient`] prices.
pub(crate) fn pool_decode_step_bytes(
    config: &super::config::Qwen3_5Config,
    rows: u64,
    window: u32,
) -> u64 {
    let block = usize::try_from(rows.max(1) - 1).unwrap_or(usize::MAX);
    dflash2_cycle_transient(config, block, u64::from(window)).total()
}

/// The pending GDN tape a DFlash2 lane holds between cycles: per linear
/// layer, 8 rows of raw conv inputs, k and v (bf16), decay (f32) and beta
/// (bf16) (Splash `gdn_tape_layer_bytes`). The verify's own tape is in the
/// cycle term; this is the other slot of Splash's two.
pub(crate) fn gdn_pending_tape_bytes(config: &super::config::Qwen3_5Config) -> u64 {
    let dim = |v: i32| v.max(0) as u64;
    let keys = dim(config.linear_num_key_heads).saturating_mul(dim(config.linear_key_head_dim));
    let values =
        dim(config.linear_num_value_heads).saturating_mul(dim(config.linear_value_head_dim));
    let conv_dim = keys.saturating_mul(2).saturating_add(values);
    let row = conv_dim
        .saturating_add(keys)
        .saturating_add(values)
        .saturating_mul(2)
        .saturating_add(dim(config.linear_num_value_heads).saturating_mul(6));
    let linear = (0..dim(config.num_layers) as usize)
        .filter(|&index| config.is_linear_layer(index))
        .count() as u64;
    linear
        .saturating_mul(super::dflash2::DFLASH2_VERIFY_ROWS as u64)
        .saturating_mul(row)
}

/// Bytes the prefill workspace bound holds past the verify bound.
fn pool_held_workspace_bytes(config: &super::config::Qwen3_5Config) -> u64 {
    use mlx_paged_attn::prefill_plan::{Pass, workspace_bound_floats};
    let (heads, head_dim) = (
        config.num_heads.max(0) as u32,
        config.head_dim.max(0) as u32,
    );
    workspace_bound_floats(Pass::Prefill, heads, head_dim)
        .saturating_sub(workspace_bound_floats(Pass::Verify, heads, head_dim))
        .saturating_mul(4)
}

/// One 8-row DFlash2 verify on the pool over `window` keys: logits, GDN
/// tapes and one layer's scratch, and the verify attention's workspace
/// bound. The stepper's decode cache limit is sized from it.
pub(crate) fn pool_verify_cycle_bytes(config: &super::config::Qwen3_5Config, window: u64) -> u64 {
    let rows = super::dflash2::DFLASH2_VERIFY_ROWS;
    dflash2_cycle_transient(config, rows - 1, window)
        .core_bytes
        .saturating_add(pool_attention_bytes(
            config,
            rows as u64,
            mlx_paged_attn::prefill_plan::Pass::Verify,
        ))
}

/// The in-place pool attention of a launch of up to `rows` rows (Splash
/// `prefillWorkspace` / `verifyWorkspaceBound` with the query staging, the
/// output and the K / V rows): no history term, both page formats.
pub(crate) fn pool_attention_bytes(
    config: &super::config::Qwen3_5Config,
    rows: u64,
    pass: mlx_paged_attn::prefill_plan::Pass,
) -> u64 {
    mlx_paged_attn::prefill_plan::attention_bytes(
        u32::try_from(rows).unwrap_or(u32::MAX),
        config.num_heads.max(0) as u32,
        config.num_kv_heads.max(0) as u32,
        config.head_dim.max(0) as u32,
        pass,
    )
}

/// Below this many keys the attention kernels run one-pass and allocate no
/// partials (`sdpa_vector_uses_two_pass` in `mlx_segmented_sdpa_plan.h`).
const SDPA_TWO_PASS_MIN_KEYS: u64 = 4096;
/// Keys per stage-1 partition of the segmented verify tile kernel
/// (`segmented_verify_tile_partitions`: `8 × max(tile_n, 8)`).
const SDPA_KEYS_PER_PARTITION: u64 = 64;

impl Qwen35DFlash2Stepper<'_> {
    /// The verify's claim suspended and replayed the lane: the stepper takes
    /// the draft window the tapped replay rebuilt and packs the GDN state
    /// the replay left in the caches into fresh cells. The old cells and
    /// their pending rows went with the vacate: the replay fed their
    /// tokens.
    fn adopt_replayed_lane(&mut self) -> Result<()> {
        let Some(state) = self.inner.dflash2_turn_state.take() else {
            return Ok(());
        };
        self.context = state.context;
        self.next_position = state.next_position;
        let tiled = self.inner.config.qwen35_gguf_gdn_layout.as_deref() == Some("tiled");
        let caches = self
            .inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?;
        let gdn = GdnStateBlobs::pack(caches, tiled)?.ok_or_else(|| {
            Error::from_reason("Qwen3.8 DFlash2 needs packed GDN state (bf16 conv, f32 recurrent)")
        })?;
        gdn.apply_views(caches)?;
        self.gdn_blobs = Some(gdn);
        Ok(())
    }

    fn ensure_clean(&self, operation: &str) -> Result<()> {
        if self.tape.is_some()
            || self.tapped.is_some()
            || self.verified_ids.is_some()
            || self.verified_device_ids.is_some()
            || self.flat_snapshot.is_some()
        {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 {operation}: prior verify was not committed"
            )));
        }
        Ok(())
    }

    /// A ring with no buffers, standing in while the lane holds the real one.
    fn empty_ring(&self) -> Result<DFlash2ContextCache> {
        self.inner
            .dflash2
            .as_ref()
            .map(|draft| DFlash2ContextCache::new(&draft.config))
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))
    }

    /// A greedy draft of the 7 proposals after `anchor` for a cycle the
    /// driver ran without one (Splash drafts every cycle).
    fn draft_tail(&mut self, anchor: u32) -> Result<Vec<u32>> {
        let draft = self
            .inner
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?;
        draft_greedy_block(
            draft,
            &self.inner.embedding,
            self.inner.lm_head.as_ref(),
            &mut self.context,
            anchor,
        )
    }

    /// Verify through the device-agnostic flat forward. The snapshot owns
    /// the pre-verify recurrent state and the logical full-attention offsets;
    /// commit trims both to the accepted prefix.
    fn flat_verify_array(&mut self, input: &MxArray) -> Result<DsparkVerifyOutput> {
        let phase_time = std::env::var("MLX_DFLASH2_PHASE_TIME").is_ok();
        let started = std::time::Instant::now();
        if input.size()? == 0 {
            return Err(Error::from_reason(
                "Qwen3.8 DFlash2 verify block must not be empty",
            ));
        }
        self.ensure_clean("verify")?;
        debug_assert!(self.inner.paged_adapter.is_none());
        let snapshot = super::layer_cache::snapshot_flat_dflash2(
            self.inner
                .caches
                .as_deref()
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?,
        )?;
        let (logits, taps, tape) = super::model::forward_dflash2_eager_with_taps_deferred(
            self.inner,
            input,
            &self.tap_layers,
            true,
            super::model::DFlash2LogitsSpan::All,
            self.cuda_pending_gdn.as_deref_mut(),
        )?;
        self.flat_snapshot = Some(snapshot);
        self.tape = Some(tape);
        self.tapped = Some(taps);
        if phase_time {
            eprintln!("[dflash2-phase] flat-verify-build: {:?}", started.elapsed());
            let eval_started = std::time::Instant::now();
            MxArray::eval_arrays(&[&logits])?;
            eprintln!(
                "[dflash2-phase] flat-verify-eval: {:?}",
                eval_started.elapsed()
            );
        }
        Ok(DsparkVerifyOutput { logits })
    }

    fn flat_verify(&mut self, ids: &[u32]) -> Result<DsparkVerifyOutput> {
        let input = MxArray::from_uint32(ids, &[1, ids.len() as i64])?;
        let output = self.flat_verify_array(&input)?;
        self.verified_ids = Some(ids.to_vec());
        self.verified_device_ids = None;
        Ok(output)
    }

    fn flat_verify_device(&mut self, ids: &MxArray) -> Result<DsparkVerifyOutput> {
        let input = ids.reshape(&[1, ids.size()? as i64])?;
        let output = self.flat_verify_array(&input)?;
        self.verified_ids = None;
        self.verified_device_ids = Some(input);
        Ok(output)
    }

    /// The verify on the pool: the 8 rows booked (Splash `Engine::workEndOf`
    /// for a decode command: committed + 8), stored and attended, the
    /// driver's `ids.len()` logits rows returned.
    fn paged_verify(&mut self, ids: VerifyIds<'_>) -> Result<DsparkVerifyOutput> {
        use super::dflash2::DFLASH2_VERIFY_ROWS;
        let empty = match &ids {
            VerifyIds::Host(ids) => ids.is_empty(),
            VerifyIds::Device(ids) => ids.size()? == 0,
        };
        if empty {
            return Err(Error::from_reason(
                "Qwen3.8 DFlash2 verify block must not be empty",
            ));
        }
        self.ensure_clean("verify")?;
        let input = match ids {
            VerifyIds::Host(ids) => {
                let drafted = match self.paged.drafted.take() {
                    Some(drafted) => drafted.host()?,
                    None => self.draft_tail(ids[0])?,
                };
                CycleInput::Host {
                    ids: ids.to_vec(),
                    full: full_verify_ids(ids, &drafted)?,
                }
            }
            VerifyIds::Device(ids) => {
                let Some(DraftedProposals::Device(drafted)) = self.paged.drafted.take() else {
                    return Err(Error::from_reason(
                        "Qwen3.8 DFlash2 device verify has no device draft",
                    ));
                };
                let full = full_device_verify_ids(ids, &drafted)?;
                CycleInput::Device {
                    ids: ids.clone(),
                    drafted,
                    full,
                }
            }
        };
        timeline::mark(Mark::VerifyInput);
        #[cfg(test)]
        let cycle_scope = crate::array::host_waits::scope("verify_cycle");
        let (seq, window) = (self.paged.seq, self.paged.context_window);
        let empty = self.empty_ring()?;
        let (lent, ring) = lend_lane(&mut self.gdn_blobs, &mut self.context, empty);
        self.inner.dflash2_lane = lent;
        self.inner.dflash2_context = Some(ring);
        let claimed = super::dflash2_turn::paged_step_claimed(&mut *self.inner, |inner| {
            inner
                .paged_adapter
                .as_mut()
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 paged adapter is absent"))?
                .reserve_rows(DFLASH2_VERIFY_ROWS as u32)
                .map(|_| ())
                .map_err(Error::from_reason)
        });
        reclaim_lane(
            self.inner.dflash2_lane.take(),
            self.inner.dflash2_context.take(),
            &mut self.gdn_blobs,
            &mut self.context,
        );
        claimed?;
        self.adopt_replayed_lane()?;
        #[cfg(test)]
        if let Some(reason) = verify_fault::take() {
            return Err(Error::from_reason(reason));
        }
        let ticket = crate::engine::spec_paged::SpecPagedCache::open_core_write_cycle(
            self,
            seq,
            DFLASH2_VERIFY_ROWS,
        )
        .map_err(Error::from_reason)?;
        timeline::mark(Mark::VerifyReserve);
        let mut tape = Vec::new();
        let phase_time = std::env::var("MLX_DFLASH2_PHASE_TIME").is_ok();
        let t0 = std::time::Instant::now();
        let verify_input = match &input {
            CycleInput::Host { full, .. } => super::paged_forward::VerifyInput::Host(full),
            CycleInput::Device { full, .. } => super::paged_forward::VerifyInput::Device {
                ids: full,
                rows: DFLASH2_VERIFY_ROWS,
            },
        };
        let result = {
            let inner = &mut *self.inner;
            let rope_deltas = inner.cached_rope_deltas.unwrap_or(0);
            match (
                inner.caches.as_mut(),
                inner.paged_adapter.as_mut(),
                self.gdn_blobs.as_ref(),
            ) {
                (Some(caches), Some(adapter), Some(gdn_blobs)) => {
                    super::paged_forward::run_paged_dflash2_verify_step(
                        verify_input,
                        &inner.embedding,
                        &mut inner.layers,
                        caches,
                        &inner.final_norm,
                        &inner.lm_head,
                        &inner.layer_kinds,
                        adapter,
                        &mut tape,
                        rope_deltas,
                        &self.tap_layers,
                        gdn_blobs,
                        window,
                    )
                }
                _ => Err(Error::from_reason(
                    "Qwen3.8 DFlash2 paged verify lost its caches",
                )),
            }
        };
        let (logits, taps) = match result {
            Ok(output) => output,
            Err(error) => {
                let _ =
                    crate::engine::spec_paged::SpecPagedCache::commit_cycle(self, seq, ticket, 0);
                return Err(error);
            }
        };
        if phase_time {
            eprintln!("[dflash2-phase] verify-build: {:?}", t0.elapsed());
            let t = std::time::Instant::now();
            MxArray::eval_arrays(&[&logits])?;
            eprintln!("[dflash2-phase] verify: {:?}", t.elapsed());
        }
        timeline::mark(Mark::VerifyGraph);
        #[cfg(test)]
        drop(cycle_scope);
        let ids = match input {
            CycleInput::Host { ids, .. } => ids,
            CycleInput::Device { ids, drafted, .. } => {
                // The verify is queued before the host waits on its ids.
                if unsafe { mlx_sys::mlx_metal_is_available() } {
                    MxArray::async_eval_arrays(&[&logits]);
                }
                match self.patch_device_rows(ids, drafted) {
                    Ok(ids) => ids,
                    Err(error) => {
                        let _ = crate::engine::spec_paged::SpecPagedCache::commit_cycle(
                            self, seq, ticket, 0,
                        );
                        return Err(error);
                    }
                }
            }
        };
        self.paged.open_cycle = Some(ticket);
        self.tape = Some(tape);
        self.tapped = Some(taps);
        let logits = logits.slice_axis(1, 0, ids.len() as i64)?;
        self.verified_ids = Some(ids);
        Ok(DsparkVerifyOutput { logits })
    }

    /// The device cycle's ids on the host: the driver's verify ids, and all
    /// 8 rows written over the pool's placeholder records.
    fn patch_device_rows(&mut self, ids: MxArray, drafted: MxArray) -> Result<Vec<u32>> {
        let ids = DraftedProposals::Device(ids).host()?;
        let drafted = DraftedProposals::Device(drafted).host()?;
        let full = full_verify_ids(&ids, &drafted)?;
        timeline::mark(Mark::ProposalHost);
        self.inner
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 paged adapter is absent"))?
            .patch_last_recorded_tokens(&full)
            .map_err(Error::from_reason)?;
        Ok(ids)
    }

    /// Close the cycle the paged verify opened, keeping `keep` of its 8 rows
    /// (Splash `Runtime::Impl::finalizeDecode`: the length advances by the
    /// retained rows, no page is freed, the next store overwrites).
    fn commit_paged_cycle(&mut self, keep: usize) -> Result<()> {
        let seq = self.paged.seq;
        let ticket =
            self.paged.open_cycle.take().ok_or_else(|| {
                Error::from_reason("Qwen3.8 DFlash2 commit has no open verify cycle")
            })?;
        crate::engine::spec_paged::SpecPagedCache::commit_cycle(self, seq, ticket, keep)
            .map_err(Error::from_reason)
    }

    fn append_tapped(&mut self, tapped: &[MxArray], token_ids: &[u32]) -> Result<()> {
        let draft = self
            .inner
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?;
        self.next_position = append_kept_taps(
            draft,
            &mut self.context,
            self.next_position,
            tapped,
            token_ids,
        )?;
        Ok(())
    }

    fn append_tapped_tree(
        &mut self,
        tapped: &[MxArray],
        accepted_nodes: &[usize],
        is_contiguous_prefix: bool,
        token_ids: &[u32],
    ) -> Result<()> {
        let draft = self
            .inner
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?;
        self.next_position = append_kept_tree_taps(
            draft,
            &mut self.context,
            self.next_position,
            tapped,
            accepted_nodes,
            is_contiguous_prefix,
            token_ids,
        )?;
        Ok(())
    }

    fn commit_flat(&mut self, keep: usize, total_written: usize) -> Result<()> {
        if keep == 0 || keep > total_written {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 invalid commit keep={keep}, total={total_written}"
            )));
        }
        let snapshot = self.flat_snapshot.take().ok_or_else(|| {
            Error::from_reason("Qwen3.8 DFlash2 flat commit has no target snapshot")
        })?;
        let tape = self
            .tape
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 commit has no GDN tape"))?;
        let tapped = self
            .tapped
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 commit has no target taps"))?;
        let verified_ids = match self.verified_ids.take() {
            Some(ids) => ids,
            None => {
                let ids = self.verified_device_ids.take().ok_or_else(|| {
                    Error::from_reason("Qwen3.8 DFlash2 commit has no token provenance")
                })?;
                // Acceptance has already synchronized the target graph. Read
                // provenance here instead of serializing draft and verify.
                DraftedProposals::Device(ids).host()?
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
            .as_deref_mut()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?;
        if let Some(pending) = self.cuda_pending_gdn.as_mut() {
            super::layer_cache::commit_flat_dflash2_deferred(
                caches,
                &snapshot,
                &tape,
                keep,
                pending,
                "Qwen3.8 DFlash2 flat commit",
            )?;
        } else {
            super::layer_cache::replay_flat_dflash2_snapshot_to(
                caches,
                &snapshot,
                &tape,
                keep,
                "Qwen3.8 DFlash2 flat commit",
            )?;
        }
        self.append_tapped(&tapped, &verified_ids[..keep])
    }

    /// Asynchronously flush `prev_commit + draft_propose` onto CUDA's single
    /// default GPU stream right before `flat_verify_array` spends 1.5-2.5 ms on
    /// CPU building the 64-layer target verify graph.
    ///
    /// Calls `mlx_sys::mlx_async_eval` directly (not `MxArray::async_eval_arrays`,
    /// which falls back to synchronous blocking eval off Metal). Single-stream
    /// safety: on CUDA `Stream::generation()` collapses to
    /// `Stream::default(DeviceType::Gpu)` and `async_eval` records a GPU
    /// `CopyableCudaEvent` (`signal_value == 1`, zero cross-stream
    /// `AtomicEvent::wait`), so `flat-verify-eval` executes strictly in stream
    /// order right behind `prev_commit + draft_propose`.
    fn cuda_async_flush_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("MLX_DFLASH2_CUDA_ASYNC_FLUSH").map_or(true, |v| v != "0")
        })
    }

    fn cuda_gpu_tree_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("MLX_DFLASH2_GPU_TREE").map_or(true, |v| v != "0"))
    }

    fn async_flush_cuda_tree_select(&self, sel: &crate::kernels::dflash2::SelectTreeOutputs) {
        if !Self::cuda_async_flush_enabled() {
            return;
        }
        let mut arrays: Vec<&MxArray> = vec![
            &sel.tree_tokens,
            &sel.parent,
            &sel.depths,
            &sel.ancestors_mask,
            &sel.conv_taps,
            &sel.node_meta,
        ];
        self.context.collect_arrays(&mut arrays);
        let mut handles: Vec<*mut mlx_sys::mlx_array> =
            arrays.iter().map(|arr| arr.as_raw_ptr()).collect();
        unsafe {
            mlx_sys::mlx_async_eval(handles.as_mut_ptr(), handles.len());
        }
    }

    fn async_flush_cuda_propose(&self, proposal: &DsparkProposal) {
        if !Self::cuda_async_flush_enabled() {
            return;
        }
        let mut arrays: Vec<&MxArray> = Vec::new();
        if let Some(ids) = proposal.device_draft_ids.as_ref() {
            arrays.push(ids);
        }
        if let Some(sparse) = proposal.device_draft_sparse.as_ref() {
            arrays.push(&sparse.ids);
            arrays.push(&sparse.probs);
        }
        self.context.collect_arrays(&mut arrays);
        crate::models::forward::CollectCacheArrays::collect_arrays(&self.inner.caches, &mut arrays);
        if arrays.is_empty() {
            return;
        }
        let mut handles: Vec<*mut mlx_sys::mlx_array> =
            arrays.iter().map(|arr| arr.as_raw_ptr()).collect();
        unsafe {
            mlx_sys::mlx_async_eval(handles.as_mut_ptr(), handles.len());
        }
    }

    /// Asynchronously flush `commit_tree`'s KV compaction, GDN conv replay,
    /// and draft-context update onto CUDA's single default stream so they
    /// overlap with host token emission, SAM updates, and the next cycle's
    /// `propose_tree_lattice` graph build.
    fn async_flush_cuda_state(&self) {
        if !Self::cuda_async_flush_enabled() {
            return;
        }
        let mut arrays: Vec<&MxArray> = Vec::new();
        self.context.collect_arrays(&mut arrays);
        crate::models::forward::CollectCacheArrays::collect_arrays(&self.inner.caches, &mut arrays);
        if arrays.is_empty() {
            return;
        }
        let mut handles: Vec<*mut mlx_sys::mlx_array> =
            arrays.iter().map(|arr| arr.as_raw_ptr()).collect();
        unsafe {
            mlx_sys::mlx_async_eval(handles.as_mut_ptr(), handles.len());
        }
    }
}

/// A greedy draft of the 7 proposals after `anchor` from `ring` (no
/// sampling draw): the tail of a cycle whose driver ran without a draft.
pub(crate) fn draft_greedy_block(
    draft: &super::dflash2::DFlash2Model,
    embedding: &crate::nn::Embedding,
    lm_head: Option<&crate::models::quantized_linear::LinearProj>,
    ring: &mut DFlash2ContextCache,
    anchor: u32,
) -> Result<Vec<u32>> {
    let (path, _) = draft.propose(
        embedding,
        lm_head,
        ring,
        anchor,
        super::dflash2::DFLASH2_DRAFT_PROPOSALS,
        0.0,
        false,
        None,
        &mut rand::rng(),
    )?;
    match path {
        super::dflash2::SelectorPath::Host(ids) => {
            Ok(ids.into_iter().map(|id| id as u32).collect())
        }
        super::dflash2::SelectorPath::Device(ids)
        | super::dflash2::SelectorPath::Sampled { ids, .. } => DraftedProposals::Device(ids).host(),
    }
}

/// A cycle's whole draft of 7 proposals after `anchor` from `ring` (Splash
/// `Runtime::decodeAsync`: one physical graph), queued on Metal, and the
/// proposal the driver reads.
pub(crate) fn propose_block(
    draft: &super::dflash2::DFlash2Model,
    embedding: &crate::nn::Embedding,
    lm_head: Option<&crate::models::quantized_linear::LinearProj>,
    ring: &mut DFlash2ContextCache,
    anchor_id: u32,
    params: &crate::engine::params::ChatParams,
    rng: &mut dyn rand::Rng,
) -> Result<(DsparkProposal, DraftedProposals)> {
    let temperature = params
        .sampling_config
        .unwrap_or_default()
        .temperature
        .unwrap_or(1.0);
    // The device path is offered only when acceptance consumes it without a
    // mid-cycle host read: penalized turns need host ids for their history,
    // so they keep the host walk.
    let device_path = params.repetition_penalty == 1.0
        && params.presence_penalty == 0.0
        && params.frequency_penalty == 0.0;
    let (path, draft_sparse_dists) = draft.propose(
        embedding,
        lm_head,
        ring,
        anchor_id,
        super::dflash2::DFLASH2_DRAFT_PROPOSALS,
        temperature,
        device_path,
        None,
        rng,
    )?;
    timeline::mark(Mark::DraftGraph);
    let (draft_ids, device_draft_ids, device_draft_sparse) = match path {
        super::dflash2::SelectorPath::Device(ids) => (Vec::new(), Some(ids), None),
        super::dflash2::SelectorPath::Sampled { ids, rows } => (Vec::new(), Some(ids), Some(rows)),
        super::dflash2::SelectorPath::Host(ids) => (ids, None, None),
    };
    // Queue the draft before the verify graph is built so the GPU is not
    // idle during that host work. Off Metal this call is a blocking eval,
    // which would only serialize the two builds.
    if let Some(ids) = &device_draft_ids
        && unsafe { mlx_sys::mlx_metal_is_available() }
    {
        MxArray::async_eval_arrays(&[ids]);
    }
    let proposal = DsparkProposal {
        draft_ids,
        device_draft_ids,
        device_draft_sparse,
        draft_dists: Vec::new(),
        draft_sparse_dists,
        keep_probabilities: None,
    };
    let drafted = match &proposal.device_draft_ids {
        Some(ids) => DraftedProposals::Device(ids.clone()),
        None => DraftedProposals::Host(proposal.draft_ids.iter().map(|&id| id as u32).collect()),
    };
    Ok((proposal, drafted))
}

/// Appends the first `token_ids.len()` rows of a verify's `tapped` hidden
/// states to `ring` at `position`; returns the position past them.
pub(crate) fn append_kept_taps(
    draft: &super::dflash2::DFlash2Model,
    ring: &mut DFlash2ContextCache,
    position: i32,
    tapped: &[MxArray],
    token_ids: &[u32],
) -> Result<i32> {
    let keep = token_ids.len();
    let mut kept = Vec::with_capacity(tapped.len());
    for hidden in tapped {
        kept.push(hidden.slice_axis(1, 0, keep as i64)?);
    }
    let fused = draft.fuse_context(&kept)?;
    ring.append(draft, &fused, position, token_ids)?;
    Ok(position.saturating_add(keep as i32))
}

/// Tree-verify counterpart to [`append_kept_taps`]: reuses the zero-copy
/// `slice_axis` fast path when `is_contiguous_prefix`, otherwise gathers the
/// accepted tree rows via `take(axis = 1)`.
pub(crate) fn append_kept_tree_taps(
    draft: &super::dflash2::DFlash2Model,
    ring: &mut DFlash2ContextCache,
    position: i32,
    tapped: &[MxArray],
    accepted_nodes: &[usize],
    is_contiguous_prefix: bool,
    token_ids: &[u32],
) -> Result<i32> {
    if is_contiguous_prefix {
        return append_kept_taps(draft, ring, position, tapped, token_ids);
    }
    let keep = token_ids.len();
    if accepted_nodes.len() != keep {
        return Err(Error::from_reason(format!(
            "Qwen3.8 DFlash2 tree taps length mismatch: nodes={}, tokens={keep}",
            accepted_nodes.len()
        )));
    }
    let indices: Vec<i32> = accepted_nodes.iter().map(|&n| n as i32).collect();
    let idx = MxArray::from_int32(&indices, &[keep as i64])?;
    let mut kept = Vec::with_capacity(tapped.len());
    for hidden in tapped {
        kept.push(hidden.take(&idx, 1)?);
    }
    let fused = draft.fuse_context(&kept)?;
    ring.append(draft, &fused, position, token_ids)?;
    Ok(position.saturating_add(keep as i32))
}

impl DsparkStepper for Qwen35DFlash2Stepper<'_> {
    fn device_sampled_rows(&self) -> bool {
        true
    }

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
        // Every cycle drafts the whole block; the driver gets its cap.
        let (mut proposal, drafted) = propose_block(
            draft,
            &self.inner.embedding,
            self.inner.lm_head.as_ref(),
            &mut self.context,
            anchor_id,
            params,
            rng,
        )?;
        if !self.eager_flat {
            self.paged.drafted = Some(drafted);
        }
        proposal.truncate(max_len);
        if self.eager_flat && crate::accelerator::cuda_ops_available() {
            self.async_flush_cuda_propose(&proposal);
        }
        timeline::mark(Mark::DraftSubmit);
        Ok(proposal)
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        if self.eager_flat {
            return self.flat_verify(verify_ids);
        }
        self.paged_verify(VerifyIds::Host(verify_ids))
    }

    /// The verify graph embeds the device ids and is built while the draft
    /// runs; the host ids are read back once the verify is queued. On the
    /// off-Metal eager flat lane, verify runs directly against the flat cache.
    fn verify_device(&mut self, verify_ids: &MxArray) -> Result<DsparkVerifyOutput> {
        if self.eager_flat {
            return self.flat_verify_device(verify_ids);
        }
        let device_draft = matches!(self.paged.drafted, Some(DraftedProposals::Device(_)));
        #[cfg(test)]
        let device_draft = device_draft && !verify_order::READ_BACK_FIRST.with(|first| first.get());
        if device_draft {
            return self.paged_verify(VerifyIds::Device(verify_ids));
        }
        let ids = DraftedProposals::Device(verify_ids.clone()).host()?;
        timeline::mark(Mark::ProposalReadback);
        self.paged_verify(VerifyIds::Host(&ids))
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        if self.eager_flat {
            return self.commit_flat(keep, total_written);
        }
        if keep == 0 || keep > total_written {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 invalid commit keep={keep}, total={total_written}"
            )));
        }
        let tape = self
            .tape
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 commit has no GDN tape"))?;
        let tapped = self
            .tapped
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 commit has no target taps"))?;
        let verified_ids = self
            .verified_ids
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 commit has no token provenance"))?;
        if verified_ids.len() != total_written {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 commit wrote {total_written} rows for {} token ids",
                verified_ids.len()
            )));
        }
        self.commit_paged_cycle(keep)?;
        timeline::mark(Mark::CommitKv);
        self.gdn_blobs
            .as_mut()
            .ok_or_else(missing_cells)?
            .retain(&tape, keep)?;
        timeline::mark(Mark::CommitGdn);
        self.append_tapped(&tapped, &verified_ids[..keep])?;
        timeline::mark(Mark::CommitRing);
        Ok(())
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
        if self.verified_ids.is_none()
            && let Some(device) = self.verified_device_ids.as_ref()
        {
            #[cfg(any(debug_assertions, test))]
            {
                let expected = DraftedProposals::Device(device.clone()).host()?;
                if expected.as_slice() != verified_ids {
                    return Err(Error::from_reason(
                        "Qwen3.8 DFlash2 supplied commit provenance disagrees with device verify input",
                    ));
                }
            }
            let _ = device;
            self.verified_device_ids = None;
            self.verified_ids = Some(verified_ids.to_vec());
        }
        match &self.verified_ids {
            Some(expected) if expected.as_slice() == verified_ids => {}
            Some(_) => {
                return Err(Error::from_reason(
                    "Qwen3.8 DFlash2 supplied commit provenance disagrees with host verify input",
                ));
            }
            None => {
                return Err(Error::from_reason(
                    "Qwen3.8 DFlash2 commit has no token provenance",
                ));
            }
        }
        self.commit(keep, total_written)
    }

    fn finish(mut self) -> Result<()> {
        self.ensure_clean("finish")?;
        if self.eager_flat {
            if let Some(mut pending) = self.cuda_pending_gdn.take() {
                let caches = self.inner.caches.as_mut().ok_or_else(|| {
                    Error::from_reason("Qwen3.8 DFlash2 target caches are absent")
                })?;
                super::layer_cache::settle_flat_dflash2_deferred(caches, &mut pending)?;
            }
            let config = self
                .inner
                .dflash2
                .as_ref()
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?
                .config
                .clone();
            let context = std::mem::replace(&mut self.context, DFlash2ContextCache::new(&config));
            self.inner.dflash2_context = Some(context);
            return Ok(());
        }
        let caches = self
            .inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?;
        self.gdn_blobs
            .as_mut()
            .ok_or_else(missing_cells)?
            .settle(caches)?;
        let config = self
            .inner
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?
            .config
            .clone();
        let context = std::mem::replace(&mut self.context, DFlash2ContextCache::new(&config));
        self.inner.dflash2_context = Some(context);
        Ok(())
    }

    fn eval_boundary(&self, token: &MxArray) {
        if !crate::accelerator::supports_async_overlap() {
            return;
        }
        let mut arrays: Vec<&MxArray> = Vec::new();
        crate::models::forward::CollectCacheArrays::collect_arrays(&self.inner.caches, &mut arrays);
        self.context.collect_arrays(&mut arrays);
        arrays.push(token);
        MxArray::schedule_overlap(&arrays);
    }

    fn supports_tree_verify(&self) -> bool {
        self.tree_enabled
    }

    fn observe_emitted_tokens(&mut self, tokens: &[u32]) {
        if self.tree_enabled && self.lookup_enabled && !tokens.is_empty() {
            self.lookup_index.append(tokens);
        }
    }

    fn verify_tree_greedy(
        &mut self,
        anchor: u32,
        max_nodes: usize,
        max_depth: usize,
    ) -> Result<DsparkTreeGreedyResult> {
        let phase_time = std::env::var("MLX_DFLASH2_PHASE_TIME").is_ok();
        let started = std::time::Instant::now();
        if !self.tree_enabled || self.cuda_pending_gdn.is_none() {
            return Err(Error::from_reason(
                "Qwen3.8 DFlash2 tree verification is not enabled on this stepper",
            ));
        }
        self.ensure_clean("verify_tree")?;
        debug_assert!(self.inner.paged_adapter.is_none());

        let draft_slots = self
            .inner
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?
            .config
            .block_size;
        let (candidates, scores) = {
            let draft = self
                .inner
                .dflash2
                .as_ref()
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?;
            draft.propose_tree_lattice_arrays(
                &self.inner.embedding,
                self.inner.lm_head.as_ref(),
                &mut self.context,
                anchor,
                draft_slots,
            )?
        };
        let after_propose = std::time::Instant::now();

        let lookup_info = self.lookup_index.match_info();
        let eff_max_depth = max_depth.min(super::verify_tree::MAX_LOOKUP_CHAIN);
        let node_budget = max_nodes
            .min(self.tree_max_nodes)
            .clamp(1, super::verify_tree::MAX_TREE_NODES);
        let lookup_cap = super::verify_tree::MAX_LOOKUP_CHAIN.min(eff_max_depth);
        let lookup_proposals = if self.lookup_enabled && lookup_cap > 0 {
            self.lookup_index.proposals(lookup_cap)
        } else {
            Vec::new()
        };
        let lookup_len = if lookup_info.match_len >= super::verify_tree::LookupIndex::MIN_MATCH {
            lookup_proposals
                .len()
                .min(super::verify_tree::MAX_LOOKUP_CHAIN)
        } else {
            0
        };

        let gpu_sel = if crate::accelerator::cuda_ops_available() && Self::cuda_gpu_tree_enabled() {
            let mut lt = [0i32; super::verify_tree::MAX_TREE_NODES];
            for (dst, &tok) in lt.iter_mut().zip(lookup_proposals.iter().take(lookup_len)) {
                *dst = tok as i32;
            }
            let mut lr = [0.0f32; super::verify_tree::MAX_TREE_NODES * 4];
            if lookup_len > 0 {
                let agreements = [
                    super::verify_tree::LookupAgreement::Top,
                    super::verify_tree::LookupAgreement::Candidate,
                    super::verify_tree::LookupAgreement::Absent,
                    super::verify_tree::LookupAgreement::Unknown,
                ];
                for d in 0..lookup_len {
                    for &ag in &agreements {
                        lr[d * 4 + ag as usize] =
                            self.lookup_acceptance
                                .rate(false, lookup_info.match_len, d, ag);
                    }
                }
            }
            let budget_acc = self.verify_budget.acceptance(false);
            let mut bi = [0.0f32; 8];
            bi[..super::verify_tree::DRAFT_SLOTS].copy_from_slice(&budget_acc.intercepts);

            let lookup_tokens_arr =
                MxArray::from_int32(&lt, &[super::verify_tree::MAX_TREE_NODES as i64])?;
            let lookup_rates_arr =
                MxArray::from_float32(&lr, &[super::verify_tree::MAX_TREE_NODES as i64, 4])?;
            let budget_intercepts_arr = MxArray::from_float32(&bi, &[8])?;
            crate::kernels::dflash2::select_tree_cuda(
                &candidates,
                &scores,
                &lookup_tokens_arr,
                &lookup_rates_arr,
                &budget_intercepts_arr,
                anchor,
                lookup_len as i32,
                node_budget as i32,
                eff_max_depth as i32,
                budget_acc.slope,
            )?
        } else {
            None
        };

        let (
            tree,
            snapshot,
            tape,
            taps,
            walk,
            is_chain,
            is_full_chain,
            after_build_tree,
            after_verify_build,
            gpu_path,
        ) = if let Some(sel) = gpu_sel {
            // Immediately flush `prev_commit + draft + select_tree_cuda` onto
            // CUDA's single stream before building the 64-layer target graph.
            self.async_flush_cuda_tree_select(&sel);
            let snapshot = super::layer_cache::snapshot_flat_dflash2(
                self.inner.caches.as_deref().ok_or_else(|| {
                    Error::from_reason("Qwen3.8 DFlash2 target caches are absent")
                })?,
            )?;
            let views = super::verify_tree::TreeDeviceViews {
                depths_i32: sel.depths,
                mask_u32: sel.ancestors_mask,
                parent_i32: sel.parent.clone(),
                conv_taps_i32: sel.conv_taps,
                tree_len: node_budget,
            };
            let after_build_tree = std::time::Instant::now();
            let (logits, taps, tape) = super::model::forward_dflash2_eager_with_taps_deferred_tree(
                self.inner,
                &sel.tree_tokens,
                &self.tap_layers,
                true,
                super::model::DFlash2LogitsSpan::All,
                self.cuda_pending_gdn.as_deref_mut(),
                Some(&views),
            )?;
            let argmax = logits.argmax(-1, None)?;
            let after_verify_build = std::time::Instant::now();
            MxArray::eval_arrays_with_context(
                &[&argmax, &sel.tree_tokens, &sel.parent, &sel.node_meta],
                "speculative_tree_greedy_accept",
            )?;
            timeline::mark(Mark::VerifyWait);

            let target_argmax_all: Vec<u32> = argmax.to_uint32()?.to_vec();
            let raw_tokens: Vec<u32> = sel.tree_tokens.to_uint32()?.to_vec();
            let raw_parent: Vec<i32> = sel.parent.to_int32()?.to_vec();
            let raw_meta: Vec<f32> = sel.node_meta.to_float32()?.to_vec();
            if target_argmax_all.len() != node_budget {
                return Err(Error::from_reason(format!(
                    "Qwen3.8 DFlash2 GPU tree argmax length {} != node_budget {}",
                    target_argmax_all.len(),
                    node_budget
                )));
            }
            let (tree, agreements, is_chain) = super::verify_tree::VerifyTree::from_gpu_readback(
                &raw_tokens,
                &raw_parent,
                &raw_meta,
            )?;
            let walk = tree.greedy_walk(&target_argmax_all[..tree.len()]);

            let labels = tree.budget_labels(&walk.emitted_tokens);
            self.verify_budget.observe(false, &labels);
            if lookup_len > 0 {
                let outcome = tree.lookup_outcome_with_agreements(
                    &lookup_proposals[..lookup_len],
                    &walk.emitted_tokens,
                    &agreements,
                );
                self.lookup_acceptance.observe(
                    false,
                    lookup_info.match_len,
                    &outcome.verified,
                    outcome.accepted,
                );
            }
            let is_full_chain = is_chain && tree.len() == node_budget;
            (
                tree,
                snapshot,
                tape,
                taps,
                walk,
                is_chain,
                is_full_chain,
                after_build_tree,
                after_verify_build,
                true,
            )
        } else {
            let lattice = {
                let draft = self
                    .inner
                    .dflash2
                    .as_ref()
                    .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?;
                draft.lattice_from_arrays(&candidates, &scores, draft_slots)?
            };
            let tree = super::verify_tree::build_verify_tree(
                anchor,
                &lattice,
                &lookup_proposals,
                lookup_info.match_len,
                node_budget,
                Some(eff_max_depth),
                false,
                &self.verify_budget,
                &self.lookup_acceptance,
            );
            let views = tree.to_device_views()?;
            let ids = MxArray::from_uint32(&tree.tokens, &[1, tree.len() as i64])?;
            let snapshot = super::layer_cache::snapshot_flat_dflash2(
                self.inner.caches.as_deref().ok_or_else(|| {
                    Error::from_reason("Qwen3.8 DFlash2 target caches are absent")
                })?,
            )?;
            let after_build_tree = std::time::Instant::now();
            let (logits, taps, tape) = super::model::forward_dflash2_eager_with_taps_deferred_tree(
                self.inner,
                &ids,
                &self.tap_layers,
                true,
                super::model::DFlash2LogitsSpan::All,
                self.cuda_pending_gdn.as_deref_mut(),
                Some(&views),
            )?;
            let argmax = logits.argmax(-1, None)?;
            let after_verify_build = std::time::Instant::now();
            MxArray::eval_arrays_with_context(&[&argmax], "speculative_tree_greedy_accept")?;
            timeline::mark(Mark::VerifyWait);
            let target_argmax: Vec<u32> = argmax.to_uint32()?.to_vec();
            if target_argmax.len() != tree.len() {
                return Err(Error::from_reason(format!(
                    "Qwen3.8 DFlash2 tree argmax length {} != tree length {}",
                    target_argmax.len(),
                    tree.len()
                )));
            }
            let walk = tree.greedy_walk(&target_argmax);

            let labels = tree.budget_labels(&walk.emitted_tokens);
            self.verify_budget.observe(false, &labels);
            if lookup_len > 0 {
                let outcome = tree.lookup_outcome(
                    &lookup_proposals[..lookup_len],
                    &walk.emitted_tokens,
                    &lattice,
                );
                self.lookup_acceptance.observe(
                    false,
                    lookup_info.match_len,
                    &outcome.verified,
                    outcome.accepted,
                );
            }
            let is_chain = tree.is_chain();
            (
                tree,
                snapshot,
                tape,
                taps,
                walk,
                is_chain,
                is_chain,
                after_build_tree,
                after_verify_build,
                false,
            )
        };

        let draft_count = tree.depth.iter().copied().max().unwrap_or(0).max(0) as usize;
        let accepted_draft_count = walk.accepted_nodes.len().saturating_sub(1);
        let tree_nodes = tree.len();
        if phase_time {
            eprintln!(
                "[dflash2-phase] tree(gpu={gpu_path},T={tree_nodes},chain={is_chain},match={},acc={accepted_draft_count}/{draft_count}): propose={:?} build_tree={:?} verify_build={:?} verify_eval={:?}",
                lookup_info.match_len,
                after_propose.duration_since(started),
                after_build_tree.duration_since(after_propose),
                after_verify_build.duration_since(after_build_tree),
                after_verify_build.elapsed(),
            );
        }

        self.flat_snapshot = Some(snapshot);
        self.tape = Some(tape);
        self.tapped = Some(taps);
        self.verified_ids = Some(tree.tokens);
        self.verified_device_ids = None;
        self.last_tree_is_chain = is_full_chain;
        self.last_tree_len = tree_nodes;

        Ok(DsparkTreeGreedyResult {
            emitted_tokens: walk.emitted_tokens,
            accepted_nodes: walk.accepted_nodes,
            is_contiguous_prefix: walk.is_contiguous_prefix,
            draft_count,
            tree_nodes,
            accepted_draft_count,
        })
    }

    fn commit_tree(&mut self, accepted_nodes: &[usize], is_contiguous_prefix: bool) -> Result<()> {
        let keep = accepted_nodes.len();
        if keep == 0 || keep > self.last_tree_len || accepted_nodes[0] != 0 {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 invalid tree commit accepted_nodes={accepted_nodes:?}, tree_len={}",
                self.last_tree_len
            )));
        }
        let snapshot = self.flat_snapshot.take().ok_or_else(|| {
            Error::from_reason("Qwen3.8 DFlash2 tree commit has no target snapshot")
        })?;
        let tape = self
            .tape
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 tree commit has no GDN tape"))?;
        let tapped = self
            .tapped
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 tree commit has no target taps"))?;
        let verified_ids = self.verified_ids.take().ok_or_else(|| {
            Error::from_reason("Qwen3.8 DFlash2 tree commit has no token provenance")
        })?;
        self.verified_device_ids = None;
        if verified_ids.len() != self.last_tree_len {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 tree commit wrote {} rows for {} token ids",
                self.last_tree_len,
                verified_ids.len()
            )));
        }
        let caches = self
            .inner
            .caches
            .as_deref_mut()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?;
        let pending = self.cuda_pending_gdn.as_mut().ok_or_else(|| {
            Error::from_reason("Qwen3.8 DFlash2 tree commit requires CUDA deferred GDN state")
        })?;
        super::layer_cache::commit_tree_dflash2_deferred(
            caches,
            &snapshot,
            &tape,
            accepted_nodes,
            is_contiguous_prefix,
            self.last_tree_is_chain,
            pending,
            "qwen3_5::dflash2::commit_tree",
        )?;
        let kept_tokens: Vec<u32> = if is_contiguous_prefix {
            verified_ids[..keep].to_vec()
        } else {
            accepted_nodes.iter().map(|&n| verified_ids[n]).collect()
        };
        self.append_tapped_tree(&tapped, accepted_nodes, is_contiguous_prefix, &kept_tokens)?;
        if self.eager_flat && crate::accelerator::cuda_ops_available() {
            self.async_flush_cuda_state();
        }
        Ok(())
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
        let eager_flat = self.paged_adapter.is_none();
        // Metal packs the prefill's per-layer GDN states into the paged lane's
        // current cell. The off-Metal lane snapshots those arrays per verify.
        let gdn_blobs = if eager_flat {
            None
        } else {
            let tiled = self.config.qwen35_gguf_gdn_layout.as_deref() == Some("tiled");
            let caches = self
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?;
            let blobs = GdnStateBlobs::pack(caches, tiled)?.ok_or_else(|| {
                Error::from_reason(
                    "Qwen3.8 DFlash2 needs packed GDN state (bf16 conv, f32 recurrent)",
                )
            })?;
            blobs.apply_views(caches)?;
            Some(blobs)
        };
        let seq = if eager_flat {
            0
        } else {
            self.paged_adapter
                .as_ref()
                .and_then(
                    crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter::active_seq_id,
                )
                .ok_or_else(|| {
                    Error::from_reason("Qwen3.8 DFlash2 paged turn has no active request")
                })?
        };
        let context_window = self.pool_context_limit();
        let paged = PagedVerifyLane {
            seq,
            context_window,
            open_cycle: None,
            drafted: None,
            _decode_limit: crate::cache_limit::coordinator().push_decode_limit(
                crate::cache_limit::decode_cache_limit(pool_verify_cycle_bytes(
                    &self.config,
                    u64::from(context_window),
                )),
            ),
        };
        if !eager_flat && self.active_scheduled_seq.is_none() {
            self.serial_decode = Some(crate::models::qwen3_5::dflash2_turn::SerialDecode::DFlash2);
        }
        let cuda_deferred_eligible = eager_flat
            && crate::accelerator::cuda_ops_available()
            && !std::env::var("MLX_DFLASH2_CUDA_DEFERRED").is_ok_and(|v| v == "0")
            && expected <= 16
            && self.config.linear_key_head_dim == 128
            && self.config.linear_value_head_dim == 128
            && self.config.linear_num_key_heads > 0
            && self.config.linear_num_value_heads > 0
            && self.config.linear_num_value_heads % self.config.linear_num_key_heads == 0
            && self.caches.as_deref().is_some_and(|caches| {
                caches.iter().all(|c| match c {
                    super::layer_cache::Qwen3_5LayerCache::FullAttention(_) => true,
                    super::layer_cache::Qwen3_5LayerCache::Linear(arrays) => {
                        arrays.get(0).and_then(|a| a.dtype().ok()) == Some(DType::BFloat16)
                            && arrays.get(1).and_then(|a| a.dtype().ok()) == Some(DType::Float32)
                    }
                })
            })
            && self.layers.iter().all(|layer| match &layer.attn {
                super::decoder_layer::AttentionType::Full(_) => true,
                super::decoder_layer::AttentionType::Linear(gdn) => gdn.is_cuda_deferred_eligible(),
            });
        let cuda_pending_gdn = if cuda_deferred_eligible {
            Some(vec![None; self.layers.len()])
        } else {
            None
        };
        let tree_enabled = cuda_pending_gdn.is_some()
            && std::env::var("MLX_DFLASH2_TREE").ok().as_deref() != Some("0");
        let tree_max_nodes = std::env::var("MLX_DFLASH2_TREE_MAX_NODES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(super::verify_tree::MAX_TREE_NODES)
            .clamp(2, super::verify_tree::MAX_TREE_NODES);
        let lookup_enabled = std::env::var("MLX_DFLASH2_LOOKUP").ok().as_deref() != Some("0");
        timeline::begin_turn();
        Ok(Qwen35DFlash2Stepper {
            inner: self,
            context: state.context,
            next_position: state.next_position,
            tap_layers,
            tape: None,
            tapped: None,
            verified_ids: None,
            verified_device_ids: None,
            gdn_blobs,
            flat_snapshot: None,
            cuda_pending_gdn,
            eager_flat,
            tree_enabled,
            tree_max_nodes,
            lookup_enabled,
            lookup_index: super::verify_tree::LookupIndex::new(),
            lookup_acceptance: super::verify_tree::LookupAcceptance::new(),
            verify_budget: super::verify_tree::VerifyBudget::new(),
            last_tree_is_chain: true,
            last_tree_len: 0,
            paged,
        })
    }
}

impl Drop for Qwen35DFlash2Stepper<'_> {
    fn drop(&mut self) {
        timeline::end_turn();
        self.inner.serial_decode = None;
        if let Some(mut pending) = self.cuda_pending_gdn.take()
            && let Some(caches) = self.inner.caches.as_mut()
        {
            let _ = super::layer_cache::settle_flat_dflash2_deferred(caches, &mut pending);
        }
        // A turn that failed between the verify write and its commit leaves
        // the cycle open; keeping every row closes the ticket, and the
        // turn's abort releases the request.
        if let Some(ticket) = self.paged.open_cycle.take() {
            let (seq, rows) = (ticket.seq_id(), ticket.rows());
            let _ =
                crate::engine::spec_paged::SpecPagedCache::commit_cycle(self, seq, ticket, rows);
        }
        // A stepper that ends without `finish` leaves its ring on the lane,
        // which the turn's end gives to the idle lane.
        if self.inner.dflash2_context.is_none()
            && self.context.buffer_bytes() > 0
            && let Some(config) = self
                .inner
                .dflash2
                .as_ref()
                .map(|draft| draft.config.clone())
        {
            let ring = std::mem::replace(&mut self.context, DFlash2ContextCache::new(&config));
            self.inner.dflash2_context = Some(ring);
        }
    }
}

impl crate::engine::spec_paged::SpecPagedCache for Qwen35DFlash2Stepper<'_> {
    fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> std::result::Result<bool, String> {
        let rows = u32::try_from(rows).unwrap_or(u32::MAX);
        match self.dflash2_pool(seq_id)?.reserve_rows(rows) {
            Ok(_) => Ok(true),
            Err(e) if e.starts_with("context_length_exceeded:") => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn record_rows(&mut self, seq_id: u32, _tokens: &[u32]) -> std::result::Result<(), String> {
        Err(format!(
            "Qwen3.8 DFlash2 sequence {seq_id}: the verify core writes its own rows"
        ))
    }

    fn record_verify(
        &mut self,
        seq_id: u32,
        _tokens: &[u32],
    ) -> std::result::Result<crate::engine::spec_paged::VerifyTicket, String> {
        Err(format!(
            "Qwen3.8 DFlash2 sequence {seq_id}: the verify core writes its own rows"
        ))
    }

    fn rollback_rows(&mut self, seq_id: u32, rows: usize) -> std::result::Result<(), String> {
        let rows = u32::try_from(rows)
            .map_err(|_| format!("Qwen3.8 DFlash2 rollback of {rows} rows does not fit u32"))?;
        self.dflash2_pool(seq_id)?.rollback_last_tokens(rows)
    }

    fn settle_committed(
        &mut self,
        seq_id: u32,
        committed_tokens: u64,
    ) -> std::result::Result<(), String> {
        let recorded = self.dflash2_pool(seq_id)?.request_tokens().len() as u64;
        if committed_tokens > recorded {
            return Err(format!(
                "Qwen3.8 DFlash2 settle_committed: committed frontier {committed_tokens} \
                 exceeds recorded token count {recorded}"
            ));
        }
        Ok(())
    }

    fn settle_captures_durable_state(&self) -> bool {
        false
    }

    fn frontier(&self, seq_id: u32) -> Option<crate::engine::backend::SpecFrontier> {
        let lane = Some(&self.paged).filter(|lane| lane.seq == seq_id)?;
        let adapter = self
            .inner
            .paged_adapter
            .as_ref()
            .filter(|adapter| adapter.active_seq_id() == Some(lane.seq))?;
        Some(crate::engine::backend::SpecFrontier {
            attn_tokens: adapter.request_tokens().len() as u64,
            recurrent_tokens: None,
        })
    }
}

impl Qwen35DFlash2Stepper<'_> {
    fn dflash2_pool(
        &mut self,
        seq_id: u32,
    ) -> std::result::Result<
        &mut crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter,
        String,
    > {
        if self.paged.seq != seq_id {
            return Err(format!(
                "Qwen3.8 DFlash2: sequence {seq_id} is not this turn's paged sequence"
            ));
        }
        self.inner
            .paged_adapter
            .as_mut()
            .filter(|adapter| adapter.active_seq_id() == Some(seq_id))
            .ok_or_else(|| format!("Qwen3.8 DFlash2: sequence {seq_id} is not the active request"))
    }
}

impl Qwen35Inner {
    /// The tapped prefill of `suffix_tokens` past `prefix`: the draft ring
    /// it advances is the restored one, else the idle lane's; it is left in
    /// the turn state for the stepper.
    pub(crate) fn tapped_dflash2_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &<Self as crate::engine::backend::PagedBackend>::PrefixState,
    ) -> Result<MxArray> {
        use crate::engine::backend::PagedPrefix;
        let config = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 has no loaded DFlash2 companion"))?
            .config
            .clone();
        let cached = prefix.effective_cached_prefix_len();
        let context = match self.dflash2_context.take() {
            Some(context) if cached > 0 && usize::try_from(context.logical_len()) == Ok(cached) => {
                context
            }
            _ if cached == 0 => self.dflash2_states.idle_lane.take_ring(&config),
            _ => {
                return Err(Error::from_reason(format!(
                    "Qwen3.8 DFlash2 cached prefix {cached} has no retained draft context"
                )));
            }
        };
        let (logits, context) = self.dense_paged_prefill(suffix_tokens, prefix, Some(context))?;
        let context = context
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 prefill lost its draft context"))?;
        self.dflash2_turn_state = Some(DFlash2TurnState {
            context,
            next_position: i32::try_from(cached + suffix_tokens.len()).unwrap_or(i32::MAX),
        });
        Ok(logits)
    }
}

impl crate::engine::dspark_turn::PagedDsparkBackend for Qwen35Inner {
    fn paged_prefill_with_draft_state(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        _stream: crate::stream::Stream,
    ) -> Result<MxArray> {
        crate::models::qwen3_5::dflash2_turn::suspendable_prefill(self, prefix.tokens(), |inner| {
            inner.tapped_dflash2_prefill(suffix_tokens, prefix)
        })
    }

    fn dspark_block_size(&self) -> Result<usize> {
        self.dflash2
            .as_ref()
            .map(|draft| draft.config.block_size)
            .ok_or_else(|| Error::from_reason("Qwen3.8 has no loaded DFlash2 companion"))
    }

    fn paged_spec_seq_id(&self) -> u32 {
        self.paged_adapter
            .as_ref()
            .and_then(
                crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter::active_seq_id,
            )
            .unwrap_or(0)
    }

    fn prime_dspark_prefix_state(
        &mut self,
        tokens: &[u32],
        reuse_cache: bool,
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        let seq = self.active_scheduled_seq.unwrap_or(0);
        crate::models::qwen3_5::dflash2_turn::prime_dflash2_turn(
            self,
            tokens,
            reuse_cache,
            cache_salt,
            seq,
            |inner, hit| inner.prime_dense_prefix(tokens, cache_salt, Some(hit)),
        )
    }
}

impl DFlash2Target for Qwen35Inner {
    fn dflash2(&self) -> Option<&super::dflash2::DFlash2Model> {
        self.dflash2.as_ref()
    }

    fn lane_seq(&self) -> u32 {
        self.active_scheduled_seq.unwrap_or(0)
    }

    fn held_prefill_bytes(&self) -> u64 {
        self.prefill_runs
            .values()
            .fold(0u64, |sum, run| sum.saturating_add(run.held_bytes()))
    }

    fn dflash2_states(&self) -> &crate::models::qwen3_5::dflash2_state::paged::DFlash2States {
        &self.dflash2_states
    }

    fn dflash2_states_mut(
        &mut self,
    ) -> &mut crate::models::qwen3_5::dflash2_state::paged::DFlash2States {
        &mut self.dflash2_states
    }

    fn vision_cache(&self) -> &crate::vision::qwen::cache::VisionCache {
        &self.vision_cache
    }

    fn dense_config(&self) -> std::borrow::Cow<'_, crate::models::qwen3_5::Qwen3_5Config> {
        std::borrow::Cow::Borrowed(&self.config)
    }

    fn embedding_dtype(&self) -> Result<DType> {
        self.embedding
            .forward(&MxArray::from_int32(&[0], &[1, 1])?)?
            .dtype()
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
    fn drop_gdn_state(&mut self, key: &crate::engine::governor::ledger::UnitKey) -> bool {
        Self::drop_gdn_state(self, key)
    }

    fn sync_gdn_state_marks(&self) {
        Self::sync_gdn_state_marks(self);
    }

    fn fresh_layer_caches(&self) -> Vec<crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache> {
        self.fresh_dflash2_layer_caches()
    }

    fn adopt_parked_state(
        &mut self,
        caches: Vec<crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache>,
        window: crate::models::qwen3_5::dflash2::DFlash2ContextCache,
        history: Vec<u32>,
    ) {
        self.adopt_parked_dflash2_state(caches, window, history);
    }

    fn replay_state_exact(&self) -> bool {
        self.dflash2_replay_state_exact()
    }

    fn paged_block_keys(&self, tokens: usize) -> Vec<Vec<u64>> {
        crate::engine::build_paged_extra_keys(
            tokens,
            self.paged_adapter
                .as_ref()
                .map_or(1, |adapter| adapter.block_size()),
            &self.cached_paged_image_token_positions,
        )
    }

    fn dflash2_turn_state_clear(&mut self) {
        let turn = self.dflash2_turn_state.take();
        self.dflash2_states
            .idle_lane
            .give_ring(turn.map(|turn| turn.context));
    }

    fn clear_draft_context(&mut self) {
        let ring = self.dflash2_context.take();
        self.dflash2_states.idle_lane.give_ring(ring);
    }

    fn snapshot_live_linear(
        &mut self,
    ) -> Result<Vec<crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache>> {
        crate::models::qwen3_5::paged_forward::snapshot_live_linear(self.caches.as_deref())
    }

    fn gdn_state_units(
        &self,
        keep: crate::models::qwen3_5::dflash2_turn::Keep<'_>,
        wanted: &crate::engine::governor::ledger::WantedChains,
    ) -> Vec<crate::models::qwen3_5::dflash2_turn::GdnStateUnit> {
        Self::gdn_state_units(self, keep, wanted)
    }

    fn memory_registration(
        &self,
    ) -> Option<&crate::engine::governor::governor::ModelGuard<'static>> {
        self.memory_registration.as_ref()
    }

    fn serial_decode(&self) -> Option<crate::models::qwen3_5::dflash2_turn::SerialDecode> {
        self.serial_decode.filter(|_| self.serial_text_lane())
    }

    fn serial_text_lane(&self) -> bool {
        self.active_scheduled_seq.is_none()
            && self.cached_image_key.is_none()
            && self.cached_paged_image_token_positions.is_empty()
    }

    fn set_serial_decode(
        &mut self,
        decode: Option<crate::models::qwen3_5::dflash2_turn::SerialDecode>,
    ) {
        self.serial_decode = decode;
    }

    fn drop_serial_lane_state(&mut self) {
        self.dflash2_lane = None;
        self.vacate_dense_lane();
    }

    fn replay_serial_lane(
        &mut self,
        tokens: &[u32],
        cache_salt: u64,
        plan: &crate::transformer::paged_kv_cache_adapter::PagedTurnPlan,
        decode: crate::models::qwen3_5::dflash2_turn::SerialDecode,
        stored: Option<crate::models::qwen3_5::dflash2_state::store::EntryId>,
    ) -> Result<MxArray> {
        self.replay_dense_serial_lane(tokens, cache_salt, plan, decode, stored)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::models::quantized_linear::LinearProj;
    use crate::nn::Linear;

    /// The seeded fixture without a pool, a sensitive random head and a
    /// tiny draft installed.
    pub(crate) fn tiny_dflash_inner_with_config(
        seed: u64,
        adjust: impl FnOnce(&mut super::super::config::Qwen3_5Config),
    ) -> Qwen35Inner {
        unsafe { mlx_sys::mlx_seed(seed) };
        let mut inner = super::super::model::scheduled_mtp::seeded_inner_with_config(seed, adjust);
        inner.paged_adapter = None;
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

    /// Qwen3.8-27B text geometry (the DFlash2 production target):
    /// 64 layers (16 full-attention), 24 heads × 256, GDN 16 key heads /
    /// 48 value heads × 128, vocab 248 320.
    pub(crate) fn qwen38_27b_cfg() -> super::super::config::Qwen3_5Config {
        super::super::config::Qwen3_5Config {
            rope_scaling: None,
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
            crate::cache_limit::decode_cache_limit(short.total()),
            crate::cache_limit::DECODE_CACHE_LIMIT_FLOOR
        );

        // 6 K prompt (6219) + 128 new → 6347 keys: two-pass → 16
        // full-attention layers × 24 heads × 8 rows × 128 partitions
        // (ceil(6347 / 64) = 100 → next multiple of 32) × 256 × 2 B.
        let six_k = dflash2_cycle_transient(&cfg, QWEN38_DRAFT_BLOCK, 6347);
        assert_eq!(six_k.core_bytes, core);
        assert_eq!(six_k.partials_bytes, 16 * 24 * rows * 128 * 256 * 2);
        let cap = crate::cache_limit::decode_cache_limit(six_k.total());
        assert_eq!(cap, 2 * (core + six_k.partials_bytes));
        // 466 MiB — the value the coordinator logged for this turn in
        // production; above the 340 MiB the unbounded pool holds at 6 K, so
        // the ceiling does not bind there (see `decode_cache_limit`).
        assert_eq!(cap >> 20, 466, "{cap}");
        // 32 K prompt (32 488) + 1024 new → 33 512 keys → 544 partitions:
        // 1714 MiB, likewise logged in production.
        assert_eq!(
            crate::cache_limit::decode_cache_limit(
                dflash2_cycle_transient(&cfg, QWEN38_DRAFT_BLOCK, 33_512).total()
            ) >> 20,
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

    pub(crate) fn metal_is_default_device() -> bool {
        crate::engine::persistence::compiled_forward_backend_available()
            && unsafe { mlx_sys::mlx_default_device() } == 1
    }

    /// Word-level `t0`..`t15` plus `<|im_end|>` (16, outside the 16-token
    /// head); the template renders each message's content and the
    /// generation prompt `t2 t3`.
    pub(crate) fn content_tokenizer() -> std::sync::Arc<crate::tokenizer::Qwen3Tokenizer> {
        word_tokenizer(None)
    }

    /// [`content_tokenizer`] whose id [`THINK_END`] is `</think>`: a
    /// thinking budget ends thinking with that one token.
    pub(crate) fn think_tokenizer() -> std::sync::Arc<crate::tokenizer::Qwen3Tokenizer> {
        word_tokenizer(Some(THINK_END))
    }

    /// The close of [`think_tokenizer`], inside the 16-token head.
    pub(crate) const THINK_END: u32 = 14;

    fn word_tokenizer(close: Option<u32>) -> std::sync::Arc<crate::tokenizer::Qwen3Tokenizer> {
        let vocab = (0..16)
            .map(|i| {
                if close == Some(i) {
                    format!("\"</think>\": {i}")
                } else {
                    format!("\"t{i}\": {i}")
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let json = format!(
            r#"{{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": {{ "type": "Whitespace" }},
                "post_processor": null,
                "decoder": null,
                "model": {{
                    "type": "WordLevel",
                    "vocab": {{ {vocab}, "<|im_end|>": 16 }},
                    "unk_token": "t0"
                }}
            }}"#
        );
        let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
            .unwrap_or_else(|error| panic!("fixture tokenizer: {error}"));
        std::sync::Arc::new(
            crate::tokenizer::Qwen3Tokenizer::from_tokenizer(
                tokenizer,
                Some(
                    "{% for message in messages %}{{ message.content }} {% endfor %}\
                     {% if add_generation_prompt %}t2 t3{% endif %}"
                        .to_string(),
                ),
            )
            .expect("tokenizer"),
        )
    }

    /// The generation prompt [`content_tokenizer`]'s template appends.
    pub(crate) const GENERATION_PROMPT: [u32; 2] = [2, 3];

    /// The prompt [`content_tokenizer`] renders for one user message of
    /// `body`.
    pub(crate) fn rendered_prompt(body: &[u32]) -> Vec<u32> {
        let mut tokens = body.to_vec();
        tokens.extend(GENERATION_PROMPT);
        tokens
    }

    fn flat_greedy_reference(
        inner: &mut Qwen35Inner,
        prompt: &[u32],
        count: usize,
    ) -> Result<Vec<u32>> {
        inner.reset_caches_sync()?;
        inner.init_caches_sync()?;
        let tap_layers = inner
            .dflash2
            .as_ref()
            .expect("tiny draft")
            .config
            .target_layers
            .clone();
        let argmax = |logits: &MxArray| -> Result<u32> {
            let ids = logits.argmax(-1, None)?;
            ids.eval();
            Ok(ids.to_int32()?[0] as u32)
        };
        let mut forward = |ids: &[u32]| -> Result<u32> {
            let input = MxArray::from_uint32(ids, &[1, ids.len() as i64])?;
            let (logits, _, _) = super::super::model::forward_dflash2_eager_with_taps(
                inner,
                &input,
                &tap_layers,
                false,
                super::super::model::DFlash2LogitsSpan::LastRow,
            )?;
            argmax(&logits)
        };
        let mut generated = vec![forward(prompt)?];
        while generated.len() < count {
            let last = *generated.last().expect("one generated token");
            generated.push(forward(&[last])?);
        }
        Ok(generated)
    }

    #[test]
    fn eager_flat_dflash2_turn_matches_greedy_ar_and_saves_aligned_caches() -> Result<()> {
        use crate::engine::backend::{ChatBackend, TurnOutput, WholeTurnArgs};
        use crate::engine::plan::{
            DecoderPlan, MediaCapabilities, MediaInputs, SpeculativeKind, TurnPlan,
        };
        use crate::stream::{Stream, StreamContext};

        const N: usize = 24;
        let prompt = (0..40u32).map(|i| (i * 7 + 3) % 16).collect::<Vec<_>>();
        let mut ar = tiny_dflash_inner_with_config(0xDFA5_F147, |config| {
            config.vocab_size = 16;
        });
        let reference = flat_greedy_reference(&mut ar, &prompt, N)?;
        let mut inner = tiny_dflash_inner_with_config(0xDFA5_F147, |config| {
            config.vocab_size = 16;
        });
        let tokenizer = content_tokenizer();
        let config = crate::engine::types::ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(N as i32),
            reasoning_effort: Some("none".to_string()),
            report_performance: Some(true),
            reuse_cache: Some(true),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            ..Default::default()
        };
        let params = ChatBackend::resolve_params(&inner, &config);
        let thinking = ChatBackend::thinking_setup(&inner, &config);
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut args = WholeTurnArgs {
            tokens: &prompt,
            tokenizer: &tokenizer,
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
        let TurnOutput::Complete(result) =
            ChatBackend::run_speculative_turn(&mut inner, &mut args)?
        else {
            panic!("a sync flat DFlash2 turn completes");
        };
        assert_eq!(result.num_tokens as usize, N);
        assert_eq!(result.prompt_tokens as usize, prompt.len());
        assert_eq!(result.cached_tokens, 0, "the off-Metal draft starts cold");
        assert_eq!(result.finish_reason, "length");
        assert!(
            result
                .performance
                .as_ref()
                .and_then(|performance| performance.mtp_cycles)
                .is_some_and(|cycles| cycles > 0),
            "the turn must execute DFlash2 cycles"
        );

        let mut expected_history = prompt.clone();
        expected_history.extend_from_slice(&reference[..N - 1]);
        assert_eq!(inner.cached_token_history, expected_history);
        for cache in inner.caches.as_deref().unwrap_or(&[]) {
            if let crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache::FullAttention(kv) = cache
            {
                assert_eq!(kv.get_offset() as usize, inner.cached_token_history.len());
            }
        }
        drop(_ctx);
        drop(inner);
        drop(ar);
        // Direct test threads do not pass through model_thread's Windows
        // shutdown hook; release cuDNN before their TLS destructors run.
        #[cfg(windows)]
        unsafe {
            mlx_sys::mlx_clear_streams();
        }
        Ok(())
    }

    /// A greedy `chatSessionStart` of one user message of `body` for
    /// `owner`, its draft off, and its reply.
    #[allow(clippy::type_complexity)]
    pub(crate) fn session_start(
        owner: &str,
        body: &[u32],
        max_new_tokens: i32,
    ) -> Result<(
        crate::engine::cmd::ChatCmd,
        tokio::sync::oneshot::Receiver<Result<crate::engine::types::ChatResult>>,
    )> {
        let (command, result, _) = session_start_with(owner, body, max_new_tokens, false)?;
        Ok((command, result))
    }

    /// [`session_start`] with its draft on when `speculate`, and its cancel
    /// flag.
    #[allow(clippy::type_complexity)]
    pub(crate) fn session_start_with(
        owner: &str,
        body: &[u32],
        max_new_tokens: i32,
        speculate: bool,
    ) -> Result<(
        crate::engine::cmd::ChatCmd,
        tokio::sync::oneshot::Receiver<Result<crate::engine::types::ChatResult>>,
        std::sync::Arc<std::sync::atomic::AtomicBool>,
    )> {
        let content = body
            .iter()
            .map(|id| format!("t{id}"))
            .collect::<Vec<_>>()
            .join(" ");
        let message: crate::tokenizer::ChatMessage =
            serde_json::from_value(serde_json::json!({"role": "user", "content": content}))
                .map_err(|error| Error::from_reason(error.to_string()))?;
        let (reply, result) = tokio::sync::oneshot::channel();
        let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let command = crate::engine::cmd::ChatCmd::SessionStart {
            messages: vec![message],
            config: crate::engine::types::ChatConfig {
                max_new_tokens: Some(max_new_tokens),
                temperature: Some(0.0),
                enable_mtp: Some(speculate),
                report_performance: speculate.then_some(true),
                max_consecutive_tokens: Some(0),
                max_ngram_repeats: Some(0),
                reasoning_effort: Some("none".to_string()),
                cache_owner_id: Some(owner.to_string()),
                ..Default::default()
            },
            reply,
            cancelled: std::sync::Arc::clone(&cancelled),
        };
        Ok((command, result, cancelled))
    }

    /// One `chatSessionStart` per `(owner, body)` (Splash `Engine::submit`),
    /// all queued before the first drive pass, driven through the model
    /// thread's scheduler until every one replies. Returns the replies and
    /// the most turns live in one pass.
    pub(crate) fn drive_session_starts<B>(
        state: &mut crate::engine::hybrid_scheduler::HybridSchedulerState<B>,
        starts: &[(&str, &[u32])],
        max_new_tokens: i32,
    ) -> Result<(Vec<crate::engine::types::ChatResult>, usize)>
    where
        B: crate::engine::hybrid_scheduler::HybridSchedulerBackend,
    {
        use crate::engine::model_command::ModelCommand;
        use crate::model_thread::LoopControl;
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let mut replies = Vec::new();
        for (owner, body) in starts {
            let (command, result) = session_start(owner, body, max_new_tokens)?;
            if sender
                .send(ModelCommand::<B::FamilyCommand>::from_chat(command))
                .is_err()
            {
                return Err(Error::from_reason("the command channel closed"));
            }
            replies.push((result, None));
        }
        // An idle pass with every sender gone ends the loop instead of
        // blocking this thread.
        drop(sender);
        let mut most_live = 0;
        for _ in 0..256 {
            if !matches!(state.drive(&mut receiver), LoopControl::Continue) {
                return Err(Error::from_reason("the scheduler stopped"));
            }
            most_live = most_live.max(state.live_turns());
            for (result, done) in &mut replies {
                if done.is_none()
                    && let Ok(reply) = result.try_recv()
                {
                    *done = Some(reply?);
                }
            }
            if replies.iter().all(|(_, done)| done.is_some()) {
                let results = replies.into_iter().filter_map(|(_, done)| done).collect();
                return Ok((results, most_live));
            }
        }
        Err(Error::from_reason("the scheduled turns did not finish"))
    }
}

#[cfg(test)]
pub(crate) mod paged_dflash2_tests {
    use super::*;
    use crate::engine::backend::{PagedBackend, PagedPrefix};
    use crate::engine::dspark_turn::PagedDsparkBackend;
    use crate::models::quantized_linear::LinearProj;
    use crate::models::qwen3_5::dflash2::{DFLASH2_DRAFT_PROPOSALS, DFLASH2_VERIFY_ROWS};
    use crate::models::qwen3_5::dflash2_state::paged::LaneVerifies;
    use crate::models::qwen3_5::paged_forward::{
        PREFILL_CHUNK_BLOCKS, PREFILL_CHUNK_ROWS_OVERRIDE, run_paged_dflash2_verify_step,
    };
    use crate::nn::Linear;
    use crate::stream::{Stream, StreamContext};
    use rand::SeedableRng;

    pub(super) const VOCAB: u32 = 16;

    /// GDN head order of the fixture: Hk = Hv = 2, or Hk 2 / Hv 4 grouped
    /// (MLX safetensors) or tiled (GGUF rows).
    #[derive(Clone, Copy, Debug)]
    enum Heads {
        Same,
        Grouped,
        Tiled,
    }

    /// The seeded paged fixture on an int8 pool, a sensitive random head and
    /// a draft that proposes the 7 tokens a cycle drafts.
    pub(crate) fn tiny_paged_int8_dflash_inner(seed: u64) -> Qwen35Inner {
        tiny_paged_int8_dflash_inner_heads(seed, Heads::Same)
    }

    fn tiny_paged_int8_dflash_inner_heads(seed: u64, heads: Heads) -> Qwen35Inner {
        tiny_paged_int8_dflash_inner_with(seed, heads, |_| {})
    }

    /// [`tiny_paged_int8_dflash_inner_heads`] whose config `adjust` changes
    /// last.
    fn tiny_paged_int8_dflash_inner_with(
        seed: u64,
        heads: Heads,
        adjust: impl FnOnce(&mut super::super::config::Qwen3_5Config),
    ) -> Qwen35Inner {
        let mut inner =
            super::super::model::scheduled_mtp::seeded_inner_with_config(seed, |config| {
                config.head_dim = 256;
                config.hidden_size = 256;
                config.kv_format = Some("int8".to_string());
                // The 27B's GDN head width, so the verify takes the deferred
                // complete kernel.
                config.linear_key_head_dim = 128;
                config.linear_value_head_dim = 128;
                config.linear_num_key_heads = 2;
                config.linear_num_value_heads = match heads {
                    Heads::Same => 2,
                    Heads::Grouped | Heads::Tiled => 4,
                };
                if matches!(heads, Heads::Tiled) {
                    config.qwen35_gguf_gdn_layout = Some("tiled".to_string());
                }
                adjust(config);
            });
        let mut head =
            Linear::new(inner.config.hidden_size as u32, VOCAB, Some(false)).expect("tiny head");
        head.set_weight(
            &MxArray::random_normal(
                &[i64::from(VOCAB), inner.config.hidden_size as i64],
                0.0,
                0.02,
                Some(DType::BFloat16),
            )
            .expect("tiny head weights"),
        )
        .expect("install tiny head");
        inner.lm_head = Some(LinearProj::Standard(head));
        let draft = super::super::dflash2::tiny_dflash2_model_with_block(
            &inner.config,
            DFLASH2_DRAFT_PROPOSALS,
        )
        .expect("tiny 7-proposal draft");
        draft
            .validate_target(&inner.config)
            .expect("draft fits target");
        inner.dflash2 = Some(draft);
        assert_eq!(
            inner.paged_adapter.as_ref().expect("pool").cache_dtype(),
            mlx_paged_attn::metal::MetalDtype::Int8,
            "the fixture's pool must be int8"
        );
        inner
    }

    pub(super) fn prompt(len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| (i * 7 + 3) % VOCAB).collect()
    }

    pub(super) fn argmax_rows(logits: &MxArray) -> Result<Vec<u32>> {
        let ids = logits.argmax(-1, None)?;
        ids.eval();
        Ok(ids.to_int32()?.iter().map(|&id| id as u32).collect())
    }

    fn cursor(inner: &Qwen35Inner) -> u32 {
        inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .current_token_count()
    }

    fn blocks(inner: &Qwen35Inner) -> usize {
        inner
            .paged_adapter
            .as_ref()
            .and_then(|adapter| adapter.block_table())
            .map_or(0, |table| table.num_blocks())
    }

    /// Greedy paged AR: `n` tokens and the cursor after them.
    pub(super) fn paged_ar(
        inner: &mut Qwen35Inner,
        prompt: &[u32],
        n: usize,
    ) -> Result<(Vec<u32>, u32)> {
        paged_ar_logits(inner, prompt, n).map(|(tokens, cursor, _)| (tokens, cursor))
    }

    /// The last row of `logits` as f32.
    fn last_row(logits: &MxArray) -> Result<Vec<f32>> {
        let f = logits.astype(DType::Float32)?;
        f.eval();
        let values = f.to_float32()?;
        let width = VOCAB as usize;
        Ok(values[values.len() - width..].to_vec())
    }

    /// [`paged_ar`] with each generated token's logits row.
    fn paged_ar_logits(
        inner: &mut Qwen35Inner,
        prompt: &[u32],
        n: usize,
    ) -> Result<(Vec<u32>, u32, Vec<Vec<f32>>)> {
        let prefix = inner.prime_dense_prefix(prompt, 0, None)?;
        let suffix = &prompt[prefix.effective_cached_prefix_len()..];
        let logits = inner.dense_paged_prefill(suffix, &prefix, None)?.0;
        let mut rows = vec![last_row(&logits)?];
        let mut tokens = argmax_rows(&logits)?;
        while tokens.len() < n {
            let last = *tokens.last().expect("one token");
            let embed = inner.embedding.clone();
            let caches = inner.caches.as_mut().expect("caches");
            let adapter = inner.paged_adapter.as_mut().expect("pool");
            let logits = super::super::paged_forward::run_paged_decode_step(
                last,
                &embed,
                &mut inner.layers,
                caches,
                &inner.final_norm,
                &inner.lm_head,
                &inner.layer_kinds,
                adapter,
                0,
            )?;
            rows.push(last_row(&logits)?);
            tokens.extend(argmax_rows(&logits)?);
        }
        Ok((tokens, cursor(inner), rows))
    }

    struct Run {
        tokens: Vec<u32>,
        cursor: u32,
        cached: usize,
        logits: Vec<Vec<u32>>,
        /// The logits row of each generated token, as AR scores it.
        path: Vec<Vec<f32>>,
        /// Deferred GDN launches per cycle, and the linear layer count.
        deferred: Vec<u64>,
        linear: u64,
        gdn: crate::models::qwen3_5::gdn_blob::trace::GdnTrace,
    }

    pub(super) fn bits(array: &MxArray) -> Result<Vec<u32>> {
        let f = array.astype(DType::Float32)?;
        f.eval();
        Ok(f.to_float32()?
            .iter()
            .map(|value| value.to_bits())
            .collect())
    }

    /// Paged DFlash2 cycles where the first `correct` proposals of each are
    /// the reference's tokens and the next one is wrong, so every cycle keeps
    /// `min(correct, l_cap) + 1` rows.
    fn paged_dflash2(
        inner: &mut Qwen35Inner,
        prompt: &[u32],
        reference: &[u32],
        correct: usize,
    ) -> Result<Run> {
        use crate::models::qwen3_5::gdn_blob::trace::{GdnTrace, cells};
        let n = reference.len();
        let linear = inner
            .layer_kinds
            .iter()
            .filter(|kind| matches!(kind, super::super::decoder_layer::Qwen3_5LayerKind::Linear))
            .count() as u64;
        let prefix = inner.prime_dspark_prefix_state(prompt, true, 0)?;
        let cached = prefix.effective_cached_prefix_len();
        let suffix = prompt[cached..].to_vec();
        let logits =
            inner.paged_prefill_with_draft_state(&suffix, &prefix, Stream::generation())?;
        let mut all_logits = vec![bits(&logits)?];
        let mut path = vec![last_row(&logits)?];
        let mut generated = argmax_rows(&logits)?;
        let params = crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
            temperature: Some(0.0),
            ..Default::default()
        });
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
        let mut gdn = GdnTrace::begin(
            step.gdn_blobs.as_ref().expect("cells"),
            step.inner.caches.as_ref().expect("caches"),
        );
        let mut deferred = Vec::new();
        while generated.len() < n {
            let anchor = *generated.last().expect("one token");
            let at = generated.len();
            let l_cap = DFLASH2_DRAFT_PROPOSALS.min(n - at - 1);
            if l_cap > 0 {
                step.propose(anchor, l_cap, &params, &mut rng)?;
            }
            let mut ids = vec![anchor];
            for i in 0..l_cap {
                let reference_id = reference[at + i];
                ids.push(if i < correct {
                    reference_id
                } else {
                    (reference_id + 1) % VOCAB
                });
            }
            unsafe { mlx_sys::mlx_test_kquant_counting(true) };
            let verified = step.verify(&ids).and_then(|out| {
                let row_bits = bits(&out.logits)?;
                Ok((out.logits, row_bits))
            });
            deferred.push(unsafe {
                mlx_sys::mlx_test_kquant_family_count(c"gdn_fused_deferred".as_ptr())
            });
            unsafe { mlx_sys::mlx_test_kquant_counting(false) };
            let (verified, row_bits) = verified?;
            all_logits.push(row_bits);
            let rows = argmax_rows(&verified)?;
            assert_eq!(rows.len(), ids.len(), "the driver sees 1 + L logits rows");
            let mut accepted = 0;
            while accepted < l_cap && rows[accepted] == ids[accepted + 1] {
                accepted += 1;
            }
            let values = verified.astype(DType::Float32)?;
            values.eval();
            let values = values.to_float32()?;
            for row in values.chunks(VOCAB as usize).take(accepted + 1) {
                path.push(row.to_vec());
            }
            let tape = step.tape.clone().expect("verify tape");
            step.commit_with_provenance(accepted + 1, ids.len(), &ids)?;
            gdn.cycles.push((tape, accepted + 1));
            generated.extend_from_slice(&ids[1..=accepted]);
            generated.push(rows[accepted]);
        }
        gdn.before_finish(
            step.gdn_blobs.as_ref().expect("cells"),
            step.inner.caches.as_ref().expect("caches"),
        );
        step.finish()?;
        gdn.after = cells(inner.caches.as_ref().expect("caches"), &gdn.layers);
        Ok(Run {
            tokens: generated,
            cursor: cursor(inner),
            cached,
            logits: all_logits,
            path,
            deferred,
            linear,
            gdn,
        })
    }

    /// Speculation on the int8 pool, every keep and head order: paged
    /// DFlash2 lands on the AR cursor, every cycle launches one deferred GDN
    /// kernel per linear layer, the cells after `finish` equal every cycle's
    /// kept rows replayed from the packed start, and with Hk = Hv the tokens
    /// are the paged AR tokens. The Hv 4 fixtures leave AR within 64 tokens
    /// on rounding alone (8-row verify projections and attention vs 1-row
    /// AR kernels; grouped heads also decode AR through the MLX conv chain,
    /// whose q/k norm and decay round up to one step away from the verify
    /// kernel's), so only their cursor and cells are compared.
    /// Mutation K1 (the retained count is 8 whatever the keep) turns it red.
    #[test]
    fn paged_dflash2_matches_paged_ar_for_every_keep() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "paged_dflash2_matches_paged_ar_for_every_keep",
        ) {
            return Ok(());
        }
        use crate::models::qwen3_5::gdn_blob::trace::{assert_cells, replay_chain};
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prompt = prompt(40);
        let n = 64;
        for heads in [Heads::Same, Heads::Grouped, Heads::Tiled] {
            let mut ar_inner = tiny_paged_int8_dflash_inner_heads(0x5_8001, heads);
            let (reference, ar_cursor) = paged_ar(&mut ar_inner, &prompt, n)?;
            for correct in 0..=DFLASH2_DRAFT_PROPOSALS {
                let tag = format!("{heads:?} with {correct} correct proposals per cycle");
                let mut inner = tiny_paged_int8_dflash_inner_heads(0x5_8001, heads);
                let run = paged_dflash2(&mut inner, &prompt, &reference, correct)?;
                assert!(
                    run.deferred.iter().all(|&launches| launches == run.linear),
                    "{tag}: deferred launches per cycle {:?} for {} linear layers",
                    run.deferred,
                    run.linear
                );
                assert_cells(&run.gdn.after, &replay_chain(&run.gdn), &tag);
                assert_eq!(run.cursor, ar_cursor, "{tag}: cursor");
                if matches!(heads, Heads::Same) {
                    assert_eq!(run.tokens[..n], reference[..], "{tag}: tokens");
                }
            }
        }
        Ok(())
    }

    /// The int8 DFlash2 fixture with sharp attention and INT8 queries from
    /// KV index `from` on (activations auto), or `None` on a device that
    /// scores none.
    fn int8_query_inner(seed: u64, from: u32) -> Option<Qwen35Inner> {
        let mut inner = tiny_paged_int8_dflash_inner(seed);
        for layer in inner.layers.iter_mut() {
            if let crate::models::qwen3_5::decoder_layer::AttentionType::Full(attn) =
                &mut layer.attn
            {
                crate::models::qwen3_5::model::scheduled_mtp::sharpen_attention(attn);
            }
        }
        let adapter = inner.paged_adapter.as_mut().expect("pool");
        adapter.set_activations(crate::transformer::Activations::Auto);
        adapter.int8_queries_from()?;
        adapter.set_int8_query_threshold(from);
        Some(inner)
    }

    /// `[tile q8q launches, coded decode rows]` of a fixture.
    fn int8_query_work(inner: &Qwen35Inner) -> [u64; 2] {
        let adapter = inner.paged_adapter.as_ref().expect("pool");
        [
            adapter.tile_query_launches()[1],
            adapter.decode_query_rows()[1],
        ]
    }

    /// The seeds of the sharp INT8-query fixtures. Each is screened: with
    /// bf16 queries its DFlash2 and AR runs agree on every token below the
    /// threshold.
    const INT8_QUERY_SEEDS: [u64; 4] = [0x5_8001, 0x5_800A, 0x5_8021, 0x5_8033];

    /// Speculation with INT8 queries from 64 tokens on, sharp attention (so
    /// the rounding of the 8-row verify against the 1-row AR step alone
    /// moves tokens, with bf16 queries too): a 40-token prompt and 64 tokens
    /// cross the threshold inside verify blocks (one split at 64) and AR
    /// decode steps, on every seed of `INT8_QUERY_SEEDS`. For every keep,
    /// the largest logits gap between the DFlash2 and AR rows of each
    /// generated token, over the tokens on which both modes' runs agree
    /// (they part past the threshold), stays within `BOUND` times the same
    /// gap with bf16 queries, below the threshold and from it on; the runs
    /// scored INT8 queries. Measured INT8 / bf16 from the threshold: 0.90 to
    /// 2.18 over the seeds; rows16 4.16 to 7.61, v-all and ss-head past 20,
    /// each on three seeds.
    /// Mutations v-all (every tile row coded), rows16 (q8q iff 16 rows or
    /// more), ss-head (every head reads head 0's score scale), and ar-bf16 /
    /// verify-bf16 (no coded AR row / no q8q verify) → red.
    #[test]
    fn paged_dflash2_with_int8_queries_matches_paged_ar() -> Result<()> {
        const BOUND: f32 = 4.0;
        if crate::test_support::paged_prefill_unsupported(
            "paged_dflash2_with_int8_queries_matches_paged_ar",
        ) {
            return Ok(());
        }
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prompt = prompt(40);
        let n = 64;
        if int8_query_inner(INT8_QUERY_SEEDS[0], 64).is_none() {
            return Ok(());
        }
        // The token whose logits row scores row 64: the prompt is 40 rows.
        const CROSSING: usize = 24;
        let keeps = [0, 3, DFLASH2_DRAFT_PROPOSALS];
        let mut failures = Vec::new();
        for seed in INT8_QUERY_SEEDS {
            // [mode][keep]: (where the tokens part, AR's rows, DFlash2's rows).
            let mut runs: [Vec<(usize, Vec<Vec<f32>>, Vec<Vec<f32>>)>; 2] =
                [Vec::new(), Vec::new()];
            for (slot, activations) in [
                crate::transformer::Activations::Bf16,
                crate::transformer::Activations::Auto,
            ]
            .into_iter()
            .enumerate()
            {
                let build = || {
                    let mut inner = int8_query_inner(seed, 64).expect("rule");
                    inner
                        .paged_adapter
                        .as_mut()
                        .expect("pool")
                        .set_activations(activations);
                    inner
                };
                let mut ar_inner = build();
                let (reference, ar_cursor, ar_rows) = paged_ar_logits(&mut ar_inner, &prompt, n)?;
                for &correct in &keeps {
                    let tag = format!("{seed:#x} {activations:?}, {correct} correct proposals");
                    let mut inner = build();
                    let run = paged_dflash2(&mut inner, &prompt, &reference, correct)?;
                    assert_eq!(run.cursor, ar_cursor, "{tag}: cursor");
                    let until = first_divergence(&run.tokens[..n], &reference).unwrap_or(n);
                    eprintln!("{tag}: tokens part at {until} of {n}");
                    if activations == crate::transformer::Activations::Auto {
                        let [tile, decode] =
                            [int8_query_work(&inner)[0], int8_query_work(&ar_inner)[1]];
                        assert!(
                            tile > 0 && decode > 0,
                            "{tag}: q8q {tile}, coded AR rows {decode}"
                        );
                    }
                    if until <= CROSSING {
                        failures.push(format!(
                            "{tag}: the runs part at {until}, before the threshold"
                        ));
                    }
                    runs[slot].push((until, ar_rows.clone(), run.path));
                }
            }
            for (k, &correct) in keeps.iter().enumerate() {
                // Both modes over the tokens on which both runs agree.
                let until = runs[0][k].0.min(runs[1][k].0).max(CROSSING);
                let gaps = runs.each_ref().map(|runs| {
                    let (_, ar_rows, path) = &runs[k];
                    let gap = |range: std::ops::Range<usize>| {
                        range
                            .flat_map(|g| {
                                ar_rows[g].iter().zip(&path[g]).map(|(a, b)| (a - b).abs())
                            })
                            .fold(0f32, f32::max)
                    };
                    (gap(0..CROSSING), gap(CROSSING..until))
                });
                let [(bf16_below, bf16_from), (below, from)] = gaps;
                let ratio = |a: f32, b: f32| a / b.max(f32::MIN_POSITIVE);
                eprintln!(
                    "{seed:#x}, {correct} correct proposals, tokens {CROSSING}..{until}: logits gap \
                     bf16 {bf16_below:e} / {bf16_from:e}, INT8 {below:e} / {from:e}; INT8 / bf16 \
                     below {:.2}, from the threshold {:.2}",
                    ratio(below, bf16_below),
                    ratio(from, bf16_from)
                );
                if below > BOUND * bf16_below || from > BOUND * bf16_from {
                    failures.push(format!(
                        "{seed:#x}, {correct} correct proposals: logits gap {below:e} / {from:e} \
                         past {BOUND} x bf16 queries' {bf16_below:e} / {bf16_from:e}"
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        Ok(())
    }

    /// Teacher-forced logits rows of `ids` through the paged prefill in
    /// chunks of `chunk` rows (`None`: one chunk), one row per position.
    #[cfg(target_os = "macos")]
    fn forced_prefill_rows(
        inner: &mut Qwen35Inner,
        ids: &[u32],
        chunk: Option<i32>,
    ) -> Result<Vec<Vec<f32>>> {
        use crate::quality::runner::EvalBackend;
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|cell| cell.set(chunk));
        let hidden = inner.prefill_hidden_once(ids);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|cell| cell.set(None));
        let logits = inner.project_head(&hidden?)?.astype(DType::Float32)?;
        logits.eval();
        Ok(logits
            .to_float32()?
            .chunks(VOCAB as usize)
            .map(<[f32]>::to_vec)
            .collect())
    }

    /// Teacher-forced logits rows of `ids` through a paged prefill of the
    /// first `prompt` and one AR decode step per later token: rows for
    /// positions `prompt - 1 ..` (the first from the prefill).
    #[cfg(target_os = "macos")]
    fn forced_decode_rows(
        inner: &mut Qwen35Inner,
        ids: &[u32],
        prompt: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let prefix = inner.prime_dense_prefix(&ids[..prompt], 0, None)?;
        let suffix = &ids[prefix.effective_cached_prefix_len()..prompt];
        let logits = inner.dense_paged_prefill(suffix, &prefix, None)?.0;
        let mut rows = vec![last_row(&logits)?];
        for &token in &ids[prompt..] {
            let embed = inner.embedding.clone();
            let caches = inner.caches.as_mut().expect("caches");
            let adapter = inner.paged_adapter.as_mut().expect("pool");
            let logits = super::super::paged_forward::run_paged_decode_step(
                token,
                &embed,
                &mut inner.layers,
                caches,
                &inner.final_norm,
                &inner.lm_head,
                &inner.layer_kinds,
                adapter,
                0,
            )?;
            rows.push(last_row(&logits)?);
        }
        Ok(rows)
    }

    /// Root mean square of the element differences of two row sets.
    #[cfg(target_os = "macos")]
    fn rows_rms(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
        assert_eq!(a.len(), b.len());
        let (sum, count) = a
            .iter()
            .zip(b)
            .flat_map(|(x, y)| x.iter().zip(y))
            .fold((0f64, 0usize), |(sum, count), (x, y)| {
                (sum + (f64::from(*x) - f64::from(*y)).powi(2), count + 1)
            });
        (sum / count.max(1) as f64).sqrt()
    }

    /// Every pass scores a row in the load's mode, seen in the logits alone
    /// (no launch counter): teacher-forced over 128 tokens with INT8 queries
    /// from 64 on, sharp attention, every seed of `INT8_QUERY_SEEDS`: the
    /// cold prefill (one chunk, split at 64), 4-row pieces, 48-row chunks,
    /// the portable varlen prefill, and a 40-token prefill followed by one
    /// AR decode step per token (warm rows: the KV a turn decoded). For each
    /// path and the rows from 64 on:
    /// - auto moves every row off the same path's bf16-query bits, by at
    ///   least `REACH` times what it moves the cold prefill (the path codes
    ///   its queries; a path that keeps bf16 queries moves nothing);
    /// - its RMS distance to the cold prefill is at most `CLASS` times the
    ///   same distance with bf16 queries (INT8 queries widen the
    ///   launch-shape class by a bounded factor);
    /// - rows below 64 of the paths whose launches end at or below 64 keep
    ///   the bf16 bits (pieces, portable, decode).
    ///
    /// On this fixture the launch shapes move the logits as much as the query
    /// format does, so no path is nearer its own mode's cold prefill than the
    /// other mode's; the check is the move itself. Both bounds come from the
    /// measurement.
    /// Mutations pieces-bf16 (a prefill launch under 64 rows keeps bf16
    /// queries), prefill-bf16, portable-bf16, ar-bf16, v-all, rows16, ss-head
    /// and ss-row → red.
    #[cfg(target_os = "macos")]
    #[test]
    fn every_path_scores_its_rows_in_one_mode() -> Result<()> {
        const REACH: f64 = 0.5;
        const CLASS: f64 = 2.5;
        const N: usize = 128;
        const FROM: usize = 64;
        const PROMPT: usize = 40;
        if crate::test_support::paged_prefill_unsupported("every_path_scores_its_rows_in_one_mode")
        {
            return Ok(());
        }
        use crate::transformer::Activations;
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        if int8_query_inner(INT8_QUERY_SEEDS[0], FROM as u32).is_none() {
            return Ok(());
        }
        let ids = prompt(N);
        const PATHS: [&str; 5] = ["cold", "pieces", "chunks", "portable", "decode"];
        let mut failures = Vec::new();
        for seed in INT8_QUERY_SEEDS {
            // [mode][path] rows for positions 0..N.
            let mut rows: Vec<Vec<Vec<Vec<f32>>>> = Vec::new();
            for activations in [Activations::Bf16, Activations::Auto] {
                let build = || {
                    let mut inner = int8_query_inner(seed, FROM as u32).expect("rule");
                    inner
                        .paged_adapter
                        .as_mut()
                        .expect("pool")
                        .set_activations(activations);
                    inner
                };
                let mut paths = Vec::new();
                for path in PATHS {
                    let mut inner = build();
                    let out = match path {
                        "cold" => forced_prefill_rows(&mut inner, &ids, None)?,
                        "pieces" => forced_prefill_rows(&mut inner, &ids, Some(4))?,
                        "chunks" => forced_prefill_rows(&mut inner, &ids, Some(48))?,
                        "portable" => {
                            inner
                                .paged_adapter
                                .as_mut()
                                .expect("pool")
                                .force_portable_prefill();
                            forced_prefill_rows(&mut inner, &ids, None)?
                        }
                        _ => {
                            let mut out = vec![Vec::new(); PROMPT - 1];
                            out.extend(forced_decode_rows(&mut inner, &ids, PROMPT)?);
                            out
                        }
                    };
                    assert_eq!(out.len(), N, "{seed:#x} {path}: rows");
                    paths.push(out);
                }
                rows.push(paths);
            }
            let tail = |path: usize, mode: usize| &rows[mode][path][FROM..];
            let modes = rows_rms(tail(0, 1), tail(0, 0));
            for (path, name) in PATHS.iter().enumerate() {
                let reach = rows_rms(tail(path, 1), tail(path, 0));
                let kept = (FROM..N)
                    .filter(|&r| {
                        rows[0][path][r]
                            .iter()
                            .zip(&rows[1][path][r])
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    })
                    .count();
                let same = if path == 0 {
                    [0.0; 2]
                } else {
                    [
                        rows_rms(tail(path, 0), tail(0, 0)),
                        rows_rms(tail(path, 1), tail(0, 1)),
                    ]
                };
                eprintln!(
                    "{seed:#x} {name}: auto vs bf16 {reach:.3e} ({:.2} of cold's {modes:.3e}), \
                     {kept} rows from {FROM} with bf16 bits; to the cold prefill bf16 {:.3e}, \
                     auto {:.3e} (x{:.2})",
                    reach / modes.max(f64::MIN_POSITIVE),
                    same[0],
                    same[1],
                    same[1] / same[0].max(f64::MIN_POSITIVE)
                );
                if kept > 0 || reach < REACH * modes {
                    failures.push(format!(
                        "{seed:#x} {name}: auto moved its rows from {FROM} by {reach:e} (cold \
                         {modes:e}); {kept} kept the bf16 bits"
                    ));
                }
                if path > 0 && same[1] > CLASS * same[0] {
                    failures.push(format!(
                        "{seed:#x} {name}: auto {:e} past {CLASS} x bf16 {:e}",
                        same[1], same[0]
                    ));
                }
                if matches!(*name, "pieces" | "portable" | "decode") {
                    let first = if *name == "decode" { PROMPT - 1 } else { 0 };
                    let kept = (first..FROM).all(|r| {
                        rows[0][path][r]
                            .iter()
                            .zip(&rows[1][path][r])
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    });
                    if !kept {
                        failures.push(format!("{seed:#x} {name}: a row below {FROM} changed"));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        Ok(())
    }

    /// `finish` folds the last cycle's retained rows once: the cells after it
    /// equal the deferred kernel's fold of the pending tape (Splash's fold
    /// itself, into throwaway cells) and differ from the cells before it.
    /// `second_dflash2_turn_resumes_the_live_window` covers the resumed turn.
    /// Mutation U1 (`finish` skips the settle) turns it red.
    #[test]
    fn finish_settles_the_pending_rows() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("finish_settles_the_pending_rows") {
            return Ok(());
        }
        use crate::models::qwen3_5::gdn_blob::trace::{assert_cells, kernel_fold};
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prompt = prompt(40);
        let n = 20;
        for heads in [Heads::Same, Heads::Grouped, Heads::Tiled] {
            let mut ar_inner = tiny_paged_int8_dflash_inner_heads(0x5_8011, heads);
            let (reference, _) = paged_ar(&mut ar_inner, &prompt, n)?;
            for correct in [0, 3, 7] {
                let tag = format!("{heads:?} correct={correct}");
                let mut inner = tiny_paged_int8_dflash_inner_heads(0x5_8011, heads);
                let run = paged_dflash2(&mut inner, &prompt, &reference, correct)?;
                assert_cells(&run.gdn.after, &kernel_fold(&run.gdn), &tag);
            }
        }
        Ok(())
    }

    /// The pending GDN tape is priced beside the verify (Splash
    /// `DecodeTensor::GdnTape`): 297,216 B per linear layer on the 27B
    /// (`gdn_tape_layer_bytes`), 48 layers. Mutation X1 (the term left out
    /// of the lane) turns it red.
    #[test]
    fn pool_lane_charges_the_pending_gdn_tape() {
        let cfg = super::super::dflash2_decode::tests::qwen38_27b_cfg();
        assert_eq!(gdn_pending_tape_bytes(&cfg), 48 * 297_216);
        assert_eq!(gdn_pending_tape_bytes(&cfg), 14_266_368);
        for window in [256u32, 2048, 65_536] {
            let w = u64::from(window);
            let chunk = w.min(PREFILL_STEP_SIZE as u64);
            let prefill = prefill_chunk_scratch_bytes(&cfg, w)
                + pool_attention_bytes(&cfg, chunk, mlx_paged_attn::prefill_plan::Pass::Prefill)
                + chunk * 3 * 5120 * 2;
            let cycle = pool_verify_cycle_bytes(&cfg, w);
            let decode = cycle
                + pool_held_workspace_bytes(&cfg)
                + 14_266_368
                + crate::cache_limit::decode_cache_limit(cycle);
            if window == 256 {
                assert!(decode > prefill, "the decode arm prices a short lane");
            }
            assert_eq!(
                pool_lane_transient(&cfg, Some(3), window, 1),
                decode.max(prefill),
                "window {window}"
            );
        }
    }

    /// Splash `compiledGdn` fails closed at plan time: a draft on a target
    /// whose GDN the deferred complete kernel cannot run (here: an f16
    /// gated-norm weight, i.e. f16 activations) is refused at load, not
    /// after a turn's prefill. Mutation L2 (no GDN check) turns it red.
    #[test]
    fn dflash2_companion_needs_a_fused_gdn_target() {
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8013);
        inner
            .require_dflash2_pool()
            .expect("the fixture's GDN runs the deferred kernel");
        let layer = inner
            .layers
            .iter_mut()
            .find_map(|layer| match &mut layer.attn {
                super::super::decoder_layer::AttentionType::Linear(gdn) => Some(gdn),
                _ => None,
            })
            .expect("a linear layer");
        let w = layer.get_norm_weight().astype(DType::Float16).expect("f16");
        layer.set_norm_weight(&w, DType::Float16).expect("f16 norm");
        let refused = inner
            .require_dflash2_pool()
            .expect_err("an f16 GDN target refuses the draft");
        assert!(
            refused.reason.contains("bf16 activations"),
            "{}",
            refused.reason
        );
    }

    /// A 3-proposal cycle runs the 8-row graph and commits 3; an `l_cap == 0`
    /// cycle drafts its tail itself and also writes 8 rows. Mutation R1 (the
    /// verify writes only the driver's rows) turns it red.
    #[test]
    fn dflash2_cycle_runs_eight_rows_and_commits_keep() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "dflash2_cycle_runs_eight_rows_and_commits_keep",
        ) {
            return Ok(());
        }
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8004);
        let prompt = prompt(40);
        let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
        let logits = inner.paged_prefill_with_draft_state(&prompt, &prefix, stream)?;
        let anchor = argmax_rows(&logits)?[0];
        let open = cursor(&inner);
        assert_eq!(open, 40);
        let params = crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
            temperature: Some(0.0),
            ..Default::default()
        });
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
        let proposal = step.propose(anchor, 3, &params, &mut rng)?;
        assert_eq!(proposal.draft_len()?, 3, "the driver gets its cap");
        let mut proposal = proposal;
        proposal.materialize_draft_ids()?;
        let mut ids = vec![anchor];
        ids.extend(proposal.draft_ids.iter().map(|&id| id as u32));
        let logits = step.verify(&ids)?.logits;
        assert_eq!(logits.shape()?.as_ref(), &[1, 4, i64::from(VOCAB)]);
        logits.eval();
        let written = step.inner.paged_adapter.as_ref().expect("pool");
        assert_eq!(
            written.current_token_count(),
            open + DFLASH2_VERIFY_ROWS as u32
        );
        let booked = written.block_table().map_or(0, |table| table.num_blocks());
        step.commit_with_provenance(2, ids.len(), &ids)?;
        let committed = step.inner.paged_adapter.as_ref().expect("pool");
        assert_eq!(committed.current_token_count(), open + 2);
        assert_eq!(
            committed
                .block_table()
                .map_or(0, |table| table.num_blocks()),
            booked,
            "a partial keep frees no page"
        );
        assert!(booked >= (open as usize + DFLASH2_VERIFY_ROWS).div_ceil(16));
        let anchor = ids[2];
        let logits = step.verify(&[anchor])?.logits;
        assert_eq!(logits.shape()?.as_ref(), &[1, 1, i64::from(VOCAB)]);
        logits.eval();
        assert_eq!(
            step.inner
                .paged_adapter
                .as_ref()
                .expect("pool")
                .current_token_count(),
            open + 2 + DFLASH2_VERIFY_ROWS as u32
        );
        step.commit_with_provenance(1, 1, &[anchor])?;
        assert_eq!(
            step.inner
                .paged_adapter
                .as_ref()
                .expect("pool")
                .current_token_count(),
            open + 3
        );
        step.finish()?;
        Ok(())
    }

    /// Every prefill chunk launches the in-place attention with the prefill
    /// split rule (Splash `prefillSplits`, max 32) and the 8-row DFlash2
    /// verify with the verify rule (Splash `kv::verifyAttentionSplits`, max
    /// 128), one launch per full-attention layer but a non-final chunk's
    /// store-only last layer. Mutation:
    /// `paged_verify_core` passes `PagedPass::Prefill` → red (no verify
    /// launch).
    #[test]
    fn verify_takes_the_verify_split_rule() -> Result<()> {
        // The launch probe counts every paged prefill in the process.
        if !crate::test_support::in_own_process(
            "models::qwen3_5::dflash2_decode::paged_dflash2_tests::verify_takes_the_verify_split_rule",
        ) {
            return Ok(());
        }
        if crate::test_support::paged_prefill_unsupported("verify_takes_the_verify_split_rule") {
            return Ok(());
        }
        let probe = |kind| unsafe { mlx_sys::mlx_paged_prefill_test_probe_count(kind) };
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8010);
        let layers = inner.config.full_attention_layer_count() as u64;
        let prompt = prompt(40);
        let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
        unsafe { mlx_sys::mlx_paged_prefill_test_probe_reset() };
        crate::models::qwen3_5::paged_forward::PREFILL_TAILS_STORED.with(|count| count.set(0));
        let logits = inner.paged_prefill_with_draft_state(&prompt, &prefix, stream)?;
        let anchor = argmax_rows(&logits)?[0];
        let chunks = probe(0);
        let tails = crate::models::qwen3_5::paged_forward::PREFILL_TAILS_STORED
            .with(std::cell::Cell::get) as u64;
        assert!(
            chunks > 0 && (chunks + tails).is_multiple_of(layers),
            "{chunks} prefill launches and {tails} store-only tails over {layers} layers"
        );
        assert_eq!(probe(1), 0, "a prefill chunk is no verify");
        assert_eq!(
            unsafe { mlx_sys::mlx_paged_prefill_test_last_max_splits() },
            32
        );
        let open = cursor(&inner);
        let params = crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
            temperature: Some(0.0),
            ..Default::default()
        });
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
        let mut proposal = step.propose(anchor, DFLASH2_DRAFT_PROPOSALS, &params, &mut rng)?;
        proposal.materialize_draft_ids()?;
        let mut ids = vec![anchor];
        ids.extend(proposal.draft_ids.iter().map(|&id| id as u32));
        step.verify(&ids)?.logits.eval();
        assert_eq!(probe(1), layers, "one verify launch per layer");
        assert_eq!(probe(0), chunks);
        assert_eq!(
            unsafe { mlx_sys::mlx_paged_prefill_test_last_splits() },
            mlx_paged_attn::prefill_plan::verify_splits(open)
        );
        assert_eq!(
            unsafe { mlx_sys::mlx_paged_prefill_test_last_max_splits() },
            128
        );
        step.commit_with_provenance(1, ids.len(), &ids)?;
        step.finish()?;
        Ok(())
    }

    /// Splash `PagedAttention::addVerify`: a lane chunk that is not the 8
    /// verify rows, a page table short of them, or rows past the window and
    /// its scratch rows are refused before anything is recorded. Mutation K1
    /// (the row check removed) turns it red.
    #[test]
    fn paged_dflash2_verify_refuses_a_chunk_that_is_not_eight_rows() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "paged_dflash2_verify_refuses_a_chunk_that_is_not_eight_rows",
        ) {
            return Ok(());
        }
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8005);
        let prompt = prompt(44);
        let prefix = inner.prime_dense_prefix(&prompt, 0, None)?;
        inner.dense_paged_prefill(&prompt, &prefix, None)?.0.eval();
        assert_eq!(cursor(&inner), 44);
        let taps = inner
            .dflash2
            .as_ref()
            .expect("draft")
            .config
            .target_layers
            .clone();
        let verify = |inner: &mut Qwen35Inner, ids: &[u32], window: u32| -> Result<()> {
            let embed = inner.embedding.clone();
            let caches = inner.caches.as_mut().expect("caches");
            let blobs = GdnStateBlobs::pack(caches, false)?.expect("packable GDN state");
            blobs.apply_views(caches)?;
            let adapter = inner.paged_adapter.as_mut().expect("pool");
            let mut tape = Vec::new();
            run_paged_dflash2_verify_step(
                super::super::paged_forward::VerifyInput::Host(ids),
                &embed,
                &mut inner.layers,
                caches,
                &inner.final_norm,
                &inner.lm_head,
                &inner.layer_kinds,
                adapter,
                &mut tape,
                0,
                &taps,
                &blobs,
                window,
            )
            .map(|(logits, _)| logits.eval())
        };
        let eight = [1u32; DFLASH2_VERIFY_ROWS];
        let short = verify(&mut inner, &eight, 4096).expect_err("3 blocks hold 48 of 52 rows");
        assert!(
            short
                .reason
                .contains("a paged verify lane's chunk is not a verify chunk")
                && short.reason.contains("page table too short"),
            "{}",
            short.reason
        );
        assert_eq!(cursor(&inner), 44);
        inner
            .paged_adapter
            .as_mut()
            .expect("pool")
            .reserve_rows(DFLASH2_VERIFY_ROWS as u32)
            .map_err(Error::from_reason)?;
        let four = verify(&mut inner, &[1, 2, 3, 4], 4096).expect_err("4 rows are no verify chunk");
        assert!(
            four.reason
                .contains("a paged verify lane's chunk is not a verify chunk")
                && four.reason.contains("4 rows"),
            "{}",
            four.reason
        );
        assert_eq!(cursor(&inner), 44, "a refused chunk records nothing");
        let past = verify(&mut inner, &eight, 44).expect_err("52 rows past a 44-token window + 7");
        assert!(
            past.reason.contains("context out of range"),
            "{}",
            past.reason
        );
        verify(&mut inner, &eight, 4096)?;
        assert_eq!(cursor(&inner), 44 + DFLASH2_VERIFY_ROWS as u32);
        Ok(())
    }

    /// Splash `Engine::workEndOf`: a DFlash2 prefill books each chunk's end
    /// (cut at the GDN checkpoint boundary), never the whole prompt; the
    /// first verify then books committed + 8. Mutation P1 (the prime books
    /// the whole prompt) turns it red.
    #[test]
    fn dflash2_prefill_books_each_chunk_end() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("dflash2_prefill_books_each_chunk_end") {
            return Ok(());
        }
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8006);
        let prompt = prompt(150);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(32)));
        PREFILL_CHUNK_BLOCKS.with(|log| *log.borrow_mut() = Some(Vec::new()));
        let result = (|| -> Result<u32> {
            let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
            let logits = inner.paged_prefill_with_draft_state(&prompt, &prefix, stream)?;
            Ok(argmax_rows(&logits)?[0])
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        let seen = PREFILL_CHUNK_BLOCKS
            .with(|log| log.borrow_mut().take())
            .unwrap_or_default();
        let anchor = result?;
        assert_eq!(
            seen,
            vec![2, 4, 6, 7, 9, 10],
            "blocks held before each chunk's forward: chunks end at 32, 64, 96, \
             112 and 144, past the checkpoints at 48 and 80"
        );
        let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
        step.verify(&[anchor])?.logits.eval();
        step.commit_with_provenance(1, 1, &[anchor])?;
        step.finish()?;
        assert_eq!(blocks(&inner), 158usize.div_ceil(16));
        Ok(())
    }

    /// The tapped prefill is the plain one: same chunks, same logits bits.
    /// With 128-row chunks the tapped prefill plans no checkpoint, so both
    /// end chunks at 128 and 144 (the plain one's GDN rung, the tapped one's
    /// replay point).
    /// Mutation T0 (the tapped prefill takes its own chunk schedule) turns it
    /// red.
    #[test]
    fn tapped_paged_prefill_keeps_plain_prefill_bits() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "tapped_paged_prefill_keeps_plain_prefill_bits",
        ) {
            return Ok(());
        }
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prompt = prompt(150);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(128)));
        let result = (|| -> Result<(Vec<u32>, Vec<u32>)> {
            let mut plain = tiny_paged_int8_dflash_inner(0x5_8007);
            let prefix = plain.prime_dense_prefix(&prompt, 0, None)?;
            let plain_logits = plain.dense_paged_prefill(&prompt, &prefix, None)?.0;
            let mut tapped = tiny_paged_int8_dflash_inner(0x5_8007);
            let prefix = tapped.prime_dspark_prefix_state(&prompt, true, 0)?;
            let tapped_logits = tapped.paged_prefill_with_draft_state(&prompt, &prefix, stream)?;
            let bits = |logits: &MxArray| -> Result<Vec<u32>> {
                let f = logits.astype(DType::Float32)?;
                f.eval();
                Ok(f.to_float32()?
                    .iter()
                    .map(|value| value.to_bits())
                    .collect())
            };
            Ok((bits(&plain_logits)?, bits(&tapped_logits)?))
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        let (plain, tapped) = result?;
        assert_eq!(
            plain, tapped,
            "the tapped prefill changed the target's logits"
        );
        Ok(())
    }

    /// A draft uses the pool on Metal and the speculative flat lane off
    /// Metal. Mutation L1 (requiring a pool in both cases) turns this red.
    #[test]
    fn dflash2_companion_selects_pool_or_flat_backend() {
        use crate::engine::backend::ChatBackend;
        use crate::engine::plan::{MediaCapabilities, TurnPath, TurnPlan, TurnRequest};
        if super::tests::metal_is_default_device() {
            let inner = tiny_paged_int8_dflash_inner(0x5_8008);
            inner
                .require_dflash2_pool()
                .expect("a pool admits the draft");
            let plan = inner.execution_plan();
            assert!(
                plan.speculative
                    .expect("draft plan")
                    .supports_paged_attention
            );
            let turn = TurnPlan::resolve(
                plan,
                TurnRequest {
                    is_delta: false,
                    input_media: MediaCapabilities::NONE,
                    context_media: MediaCapabilities::NONE,
                    speculative_requested: true,
                    streaming: false,
                },
            );
            assert!(matches!(turn.path(), TurnPath::Paged));
        }
        let inner = super::tests::tiny_dflash_inner_with_config(0x5_8008, |_| {});
        inner
            .require_dflash2_pool()
            .expect("the eager flat lane admits the draft off Metal");
        let flat = inner.execution_plan();
        assert!(
            !flat
                .speculative
                .expect("flat draft plan")
                .supports_paged_attention
        );
        let flat_turn = TurnPlan::resolve(
            flat,
            TurnRequest {
                is_delta: false,
                input_media: MediaCapabilities::NONE,
                context_media: MediaCapabilities::NONE,
                speculative_requested: true,
                streaming: false,
            },
        );
        assert!(matches!(flat_turn.path(), TurnPath::Speculative));
    }

    /// The paged DSpark driver runs on the session's params, the copy a
    /// context preflight may have clamped, not a fresh read of the config:
    /// a 3-token budget in `args.params` under a config of 12 emits 3.
    /// Mutation C11: the driver re-resolves the config → red (12 tokens).
    #[test]
    fn paged_dflash2_turn_honors_the_session_params() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "paged_dflash2_turn_honors_the_session_params",
        ) {
            return Ok(());
        }
        use crate::engine::backend::{ChatBackend, TurnOutput, WholeTurnArgs};
        use crate::engine::plan::{
            DecoderPlan, MediaCapabilities, MediaInputs, SpeculativeKind, TurnPlan,
        };
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8009);
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let tokens = prompt(40);
        let config = crate::engine::types::ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(12),
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        };
        let mut params = ChatBackend::resolve_params(&inner, &config);
        assert_eq!(params.max_new_tokens, 12);
        params.max_new_tokens = 3;
        let thinking = ChatBackend::thinking_setup(&inner, &config);
        let mut args = WholeTurnArgs {
            tokens: &tokens,
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
                decoder: DecoderPlan::Speculative(SpeculativeKind::DraftModel),
            },
            sink: None,
            cancelled: None,
            media: MediaInputs {
                images: &[],
                audio: &[],
            },
        };
        let TurnOutput::Complete(result) = ChatBackend::run_paged_turn(&mut inner, &mut args)?
        else {
            panic!("a sync turn completes");
        };
        assert_eq!(result.num_tokens, 3, "the turn ran on args.params");
        Ok(())
    }

    /// One whole turn of `tokens` through `run_paged_turn` on `decoder`;
    /// returns the history it saved: the prompt and the generated tokens
    /// but the last.
    fn whole_turn(
        inner: &mut Qwen35Inner,
        tokens: &[u32],
        decoder: crate::engine::plan::DecoderPlan,
    ) -> Result<Vec<u32>> {
        whole_turn_with(inner, tokens, decoder, None).map(|(history, _)| history)
    }

    /// [`whole_turn`] with the request's `reuse_cache`; also the raw text.
    fn whole_turn_with(
        inner: &mut Qwen35Inner,
        tokens: &[u32],
        decoder: crate::engine::plan::DecoderPlan,
        reuse_cache: Option<bool>,
    ) -> Result<(Vec<u32>, String)> {
        use crate::engine::backend::{ChatBackend, TurnOutput, WholeTurnArgs};
        use crate::engine::plan::{MediaCapabilities, MediaInputs, TurnPlan};
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let config = crate::engine::types::ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(20),
            reasoning_effort: Some("none".to_string()),
            reuse_cache,
            ..Default::default()
        };
        let params = ChatBackend::resolve_params(inner, &config);
        let thinking = ChatBackend::thinking_setup(inner, &config);
        let mut args = WholeTurnArgs {
            tokens,
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
                decoder,
            },
            sink: None,
            cancelled: None,
            media: MediaInputs {
                images: &[],
                audio: &[],
            },
        };
        let TurnOutput::Complete(result) = ChatBackend::run_paged_turn(inner, &mut args)? else {
            panic!("a sync turn completes");
        };
        assert_eq!(result.num_tokens, 20);
        Ok((inner.cached_token_history.clone(), result.raw_text))
    }

    fn block_ids(inner: &Qwen35Inner) -> Vec<u32> {
        inner
            .paged_adapter
            .as_ref()
            .and_then(|adapter| adapter.block_table())
            .map_or_else(Vec::new, |table| table.block_ids())
    }

    fn parked_depths(inner: &Qwen35Inner) -> Vec<usize> {
        inner
            .dflash2_states
            .store
            .entries()
            .map(|entry| entry.depth())
            .collect()
    }

    /// A1 parks its prompt's replay point (208) on its own chain; an
    /// unrelated Q then runs on a pool that must grow for it; A2 (A1's
    /// prompt to 208, then new text) resumes from the parked state on A1's
    /// blocks on the installed GDN cell, and its tokens and teacher-forced
    /// DFlash2 logits equal those of a run with no Q in between, bit for bit.
    /// (Paged AR is no reference: its decode kernel is not the verify's
    /// in-place attention.)
    /// Mutation MK1 (parked states mark no block) → red (Q's prefill takes
    /// A1's chain instead of growing the pool; A2's install fails).
    /// Mutation GI1 (the install leaves the GDN slots fresh) → red (A2
    /// replays the GDN prefix).
    #[test]
    fn parked_anchor_resume_matches_an_uninterrupted_run() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "parked_anchor_resume_matches_an_uninterrupted_run",
        ) {
            return Ok(());
        }
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::models::qwen3_5::model::dflash2_park_tests::{install_pool, passing_probe};
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let a1 = prompt(210);
        let q: Vec<u32> = (0..210u32).map(|i| (i * 5 + 1) % VOCAB).collect();
        let mut a2 = a1[..208].to_vec();
        a2.extend((0..30u32).map(|i| (i * 3 + 2) % VOCAB));
        // The AR reference resumes A2 at 208 through A1's ladder rung, so
        // both sides read A1's rows back from the int8 pool.
        let mut ar = tiny_paged_int8_dflash_inner(0x5_8101);
        install_pool(&mut ar, 16, 64, passing_probe())?;
        whole_turn(&mut ar, &a1, dflash2)?;
        ar.dflash2_states.clear();
        ar.sync_gdn_state_marks();
        let (reference, _) = paged_ar(&mut ar, &a2, 24)?;
        let run = |with_q: bool| -> Result<Run> {
            let mut inner = tiny_paged_int8_dflash_inner(0x5_8101);
            install_pool(&mut inner, 16, 64, passing_probe())?;
            whole_turn(&mut inner, &a1, dflash2)?;
            inner.drop_gdn_states_for_test();
            assert_eq!(parked_depths(&inner), vec![208]);
            if with_q {
                whole_turn(&mut inner, &q, dflash2)?;
            }
            let run = paged_dflash2(&mut inner, &a2, &reference, 3)?;
            assert_eq!(
                inner.last_gdn_prefix_prepare_state, "last_history",
                "A2 resumes on the installed GDN cell, with no replay"
            );
            Ok(run)
        };
        let x = run(true)?;
        let y = run(false)?;
        assert_eq!(
            (x.cached, y.cached),
            (208, 208),
            "A2 resumes on A1's blocks"
        );
        assert_eq!(x.tokens, y.tokens);
        assert!(x.logits == y.logits, "teacher-forced logits differ");
        Ok(())
    }

    /// Two sessions on one prefix share its blocks: B resumes from A1's
    /// parked replay point and writes only its own tail.
    /// Mutation CP1 (a stored-state prepare skips the lookup) → red (B
    /// re-prefills its own copy: no shared ids, 31 >= 30 blocks).
    #[test]
    fn two_sessions_share_the_parked_prefix_blocks() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "two_sessions_share_the_parked_prefix_blocks",
        ) {
            return Ok(());
        }
        use crate::engine::backend::{ChatBackend, PagedPrefix};
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let a1 = prompt(210);
        let mut b = a1[..208].to_vec();
        b.extend((0..40u32).map(|i| (i * 11 + 4) % VOCAB));
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8102);
        whole_turn(&mut inner, &a1, dflash2)?;
        // The turn released its lane; a chain is read back from a prime of
        // its own prompt, which restores at its replay point.
        let primed = |inner: &mut Qwen35Inner, tokens: &[u32], at: usize| -> Result<Vec<u32>> {
            let salt = ChatBackend::resolve_params(&*inner, &Default::default()).cache_salt;
            let prefix = inner.prime_dspark_prefix_state(tokens, true, salt)?;
            assert_eq!(prefix.effective_cached_prefix_len(), at);
            let ids = block_ids(inner);
            inner.abort_paged_turn();
            Ok(ids)
        };
        let a1_ids = primed(&mut inner, &a1, 208)?;
        let allocated = |inner: &Qwen35Inner| {
            inner
                .paged_adapter
                .as_ref()
                .expect("pool")
                .block_telemetry()
                .expect("telemetry")
                .allocated_blocks as usize
        };
        whole_turn(&mut inner, &b, dflash2)?;
        let b_ids = primed(&mut inner, &b, 240)?;
        assert_eq!(b_ids[..13], a1_ids[..13], "B sits on A1's chain");
        assert!(
            allocated(&inner) < 30,
            "{} blocks for two sessions",
            allocated(&inner)
        );
        Ok(())
    }

    /// A qwen3_5 SSD tier holds DFlash2 states only (Splash keeps no
    /// per-turn KV on disk): an AR turn and a DFlash2 turn with the tier open
    /// write no KV object and no sidecar, and the adapter captures nothing.
    /// Mutation PT1: the tier's context backed `KvOnly` → red.
    #[test]
    fn qwen35_turns_write_no_kv_per_turn() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("qwen35_turns_write_no_kv_per_turn") {
            return Ok(());
        }
        use crate::disk_tier::{CacheDisk, DiskTierOptions, open_disk_tier};
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let root = std::env::temp_dir().join(format!(
            "mlx-qwen35-no-kv-per-turn-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("temp root");
        let options = DiskTierOptions {
            cache_disk: CacheDisk::Bytes(1 << 30),
            temp_root: Some(root.clone()),
            ..DiskTierOptions::default()
        };
        let tier = open_disk_tier(&options, || None, 1 << 20).expect("a temporary tier");
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        for decoder in [DecoderPlan::Autoregressive, dflash2] {
            let mut inner = tiny_paged_int8_dflash_inner(0x7_1010);
            inner
                .paged_adapter
                .as_mut()
                .expect("pool")
                .set_cold_tier(super::super::dflash2_state::disk::state_context(&tier));
            whole_turn(&mut inner, &prompt(100), decoder)?;
            let adapter = inner.paged_adapter.as_ref().expect("pool");
            assert_eq!(adapter.cold_captured_blocks(), 0, "{decoder:?}");
            assert_eq!(adapter.cold_capture().enqueued, 0, "{decoder:?}");
        }
        assert!(tier.manager.drain(std::time::Duration::from_secs(10)));
        for group in [
            mlx_paged_attn::ColdGroup::Kv,
            mlx_paged_attn::ColdGroup::ConvState,
            mlx_paged_attn::ColdGroup::SlidingWindow,
        ] {
            assert!(tier.manager.keys_in(group).is_empty(), "{group:?}");
        }
        assert_eq!(tier.manager.total_bytes(), 0);
        drop(tier);
        let _ = std::fs::remove_dir_all(&root);
        Ok(())
    }

    /// The next turn of the prompt that published R restores the draft from
    /// R's 4-bit window (Splash `planDraftContext`: its first point, the
    /// prompt's end, lies within a window of R): the ring holds the
    /// projection of the window's dequantized newest rows, at R, and the
    /// lane's window takes R's rows. A turn whose first point lies a window
    /// or more past R starts with an empty ring at R (Splash
    /// `restoreLengths(.., false)`), and its tokens are paged AR's.
    /// Mutation RD1 (the restore ignores the flag) → red.
    #[test]
    fn restore_rebuilds_the_draft_or_starts_empty() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "restore_rebuilds_the_draft_or_starts_empty",
        ) {
            return Ok(());
        }
        use crate::engine::backend::{ChatBackend, PagedPrefix};
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let first = prompt(50);
        let mut far = first[..48].to_vec();
        far.extend((0..40u32).map(|i| (i * 5 + 2) % VOCAB));
        let turn_one = |decoder| -> Result<Qwen35Inner> {
            let mut inner = tiny_paged_int8_dflash_inner(0x5_8120);
            whole_turn(&mut inner, &first, decoder)?;
            Ok(inner)
        };
        let view_bits = |ring: &DFlash2ContextCache| -> Result<Vec<Vec<u32>>> {
            let mut out = Vec::new();
            for view in ring.window_views() {
                let (keys, values) = view.ok_or_else(|| Error::from_reason("empty ring"))?;
                out.push(bits(&keys)?);
                out.push(bits(&values)?);
            }
            Ok(out)
        };

        let mut near = turn_one(dflash2)?;
        assert_eq!(parked_depths(&near), vec![48]);
        let salt = ChatBackend::resolve_params(&near, &Default::default()).cache_salt;
        let prefix = near.prime_dspark_prefix_state(&first, true, salt)?;
        assert_eq!(prefix.effective_cached_prefix_len(), 48);
        let draft = near.dflash2.as_ref().expect("draft");
        let window = &near
            .dflash2_states
            .store
            .entries()
            .next()
            .expect("R")
            .ram()
            .expect("a RAM copy")
            .window
            .as_ref()
            .expect("a window");
        let begin = window
            .begin()
            .max(48 - (draft.config.sliding_window as u32 - 1));
        let mut want = DFlash2ContextCache::new(&draft.config);
        want.record_only(&first[..begin as usize])?;
        let rows = window.load_rows(begin, 48 - begin)?.reshape(&[
            1,
            i64::from(48 - begin),
            i64::from(window.width()),
        ])?;
        want.append(draft, &rows, begin as i32, &first[begin as usize..48])?;
        let ring = near.dflash2_context.as_ref().expect("restored ring");
        assert_eq!(ring.logical_len(), 48);
        assert!(view_bits(ring)? == view_bits(&want)?, "restored ring rows");
        let lane = near.dflash2_states.lane_window.as_ref().expect("lane");
        assert_eq!((lane.begin(), lane.end()), (window.begin(), 48));
        assert_eq!(lane.host_bytes()?, window.host_bytes()?);

        let mut cold = turn_one(dflash2)?;
        let prefix = cold.prime_dspark_prefix_state(&far, true, salt)?;
        assert_eq!(prefix.effective_cached_prefix_len(), 48);
        let ring = cold.dflash2_context.as_ref().expect("ring");
        assert_eq!(ring.logical_len(), 48);
        assert!(
            ring.window_views().iter().all(Option::is_none),
            "a far first point starts the ring empty"
        );
        assert!(cold.dflash2_states.lane_window.is_none());

        let mut spec = turn_one(dflash2)?;
        let mut ar = turn_one(DecoderPlan::Autoregressive)?;
        assert_eq!(
            whole_turn(&mut spec, &far, dflash2)?,
            whole_turn(&mut ar, &far, DecoderPlan::Autoregressive)?
        );
        Ok(())
    }

    /// An autoregressive turn on a draft load publishes the points a DFlash2
    /// turn of its prompt does, each with its draft window (Splash
    /// `configureDraftStatePlan` runs for every request), and decodes the
    /// tokens of a twin that keeps no state; its conversation's next
    /// autoregressive turn resumes at its replay point.
    /// Mutation MDL1 (`draft_lane_turn` false) → red (nothing parked).
    #[test]
    fn ar_turn_on_a_draft_load_publishes_draft_states() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "ar_turn_on_a_draft_load_publishes_draft_states",
        ) {
            return Ok(());
        }
        use crate::engine::backend::ChatBackend;
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let first = prompt(100);
        let mut spec = tiny_paged_int8_dflash_inner(0x5_8140);
        whole_turn(
            &mut spec,
            &first,
            DecoderPlan::Speculative(SpeculativeKind::DraftModel),
        )?;
        let mut ar = tiny_paged_int8_dflash_inner(0x5_8140);
        let (history, text) =
            whole_turn_with(&mut ar, &first, DecoderPlan::Autoregressive, Some(true))?;
        let depths = parked_depths(&ar);
        assert!(!depths.is_empty(), "the AR turn published nothing");
        assert_eq!(depths, parked_depths(&spec));
        assert!(
            ar.dflash2_states
                .store
                .entries()
                .all(|entry| entry.ram().is_some_and(|state| state.window.is_some())),
            "every state carries the draft window"
        );
        let mut cold = tiny_paged_int8_dflash_inner(0x5_8140);
        let (_, cold_text) =
            whole_turn_with(&mut cold, &first, DecoderPlan::Autoregressive, Some(false))?;
        assert!(parked_depths(&cold).is_empty());
        assert_eq!(text, cold_text, "publishing changed the tokens");
        let mut next = history.clone();
        next.extend((0..12u32).map(|i| (i * 7 + 3) % VOCAB));
        let salt = ChatBackend::resolve_params(&ar, &Default::default()).cache_salt;
        let prefix = PagedBackend::prime_prefix_state(&mut ar, &next, true, 0, &[], salt)?;
        assert_eq!(
            prefix.effective_cached_prefix_len(),
            *depths.iter().max().expect("a depth")
        );
        Ok(())
    }

    /// One stored state: its depth, its GDN slots' bits, and its window's
    /// bounds and rows' bits.
    type DraftState = (usize, Vec<Vec<u32>>, (u32, u32), Vec<u32>);

    fn window_state(
        window: &crate::models::qwen3_5::draft_context::DraftContextWindow,
    ) -> Result<((u32, u32), Vec<u32>)> {
        let bounds = (window.begin(), window.end());
        let rows = bits(&window.load_rows(window.begin(), window.end() - window.begin())?)?;
        Ok((bounds, rows))
    }

    /// The states on `tokens`' lineage, by depth.
    fn draft_states(inner: &Qwen35Inner, tokens: &[u32]) -> Result<Vec<DraftState>> {
        let mut states = Vec::new();
        for entry in inner.dflash2_states.store.entries() {
            if !tokens.starts_with(entry.tokens()) {
                continue;
            }
            let anchor = entry.ram().expect("a RAM copy");
            let mut arrays = Vec::new();
            for cache in &anchor.linear {
                if matches!(
                    cache,
                    crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache::Linear(_)
                ) {
                    cache.collect_arrays(&mut arrays);
                }
            }
            let cell = arrays.into_iter().map(bits).collect::<Result<Vec<_>>>()?;
            let (bounds, rows) = window_state(anchor.window.as_ref().expect("a draft window"))?;
            states.push((entry.depth(), cell, bounds, rows));
        }
        states.sort_by_key(|state| state.0);
        Ok(states)
    }

    /// The fixture with a `rows`-row draft window, so a point's window
    /// spans prefill slices.
    fn wide_draft_inner(seed: u64, rows: usize) -> Qwen35Inner {
        let mut inner = tiny_paged_int8_dflash_inner(seed);
        inner.dflash2.as_mut().expect("draft").config.sliding_window = rows;
        inner
    }

    /// The states a whole autoregressive turn of `tokens` on a draft load
    /// publishes (task B's tapped prime and prefill), and its logits.
    fn whole_turn_states(seed: u64, tokens: &[u32]) -> Result<(Vec<DraftState>, Vec<u32>)> {
        let mut inner = wide_draft_inner(seed, 40);
        let prefix = inner.prime_dspark_prefix_state(tokens, true, 0)?;
        assert_eq!(prefix.effective_cached_prefix_len(), 0);
        let logits = inner.paged_prefill_with_draft_state(tokens, &prefix, Stream::generation())?;
        Ok((draft_states(&inner, tokens)?, bits(&logits)?))
    }

    /// The scheduled prime of `tokens` on `seq` (Splash `admit` on a lane of
    /// its own), the sequence parked after it.
    fn scheduled_prime(
        inner: &mut Qwen35Inner,
        seq: u32,
        tokens: &[u32],
    ) -> Result<<Qwen35Inner as PagedBackend>::PrefixState> {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        HybridSchedulerBackend::activate_scheduled_recurrent(inner, seq)?;
        let base = PagedBackend::prime_prefix_state(inner, tokens, true, 16, &[], 0)?;
        HybridSchedulerBackend::park_active_scheduled_recurrent(inner)?;
        Ok(base)
    }

    /// The scheduler's prefill slice `[start, end)` of `seq`'s request, the
    /// sequence parked after it; returns the slice's logits, which only the
    /// prompt's last slice gives.
    fn scheduled_slice(
        inner: &mut Qwen35Inner,
        seq: u32,
        tokens: &[u32],
        base: &<Qwen35Inner as PagedBackend>::PrefixState,
        start: usize,
        end: usize,
    ) -> Result<Option<MxArray>> {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        HybridSchedulerBackend::activate_paged_seq(inner, seq)?;
        let logits = HybridSchedulerBackend::run_scheduled_prefill_slice(
            inner,
            seq,
            tokens,
            base,
            start,
            end,
            Stream::default(crate::stream::DeviceType::Gpu),
            start == base.effective_cached_prefix_len(),
        )?;
        if let Some(logits) = logits.as_ref() {
            logits.eval();
        }
        HybridSchedulerBackend::park_active_scheduled_recurrent(inner)?;
        Ok(logits)
    }

    /// [`scheduled_slice`] of the prompt's last unit `[start, len)`.
    fn scheduled_last_slice(
        inner: &mut Qwen35Inner,
        seq: u32,
        tokens: &[u32],
        base: &<Qwen35Inner as PagedBackend>::PrefixState,
        start: usize,
    ) -> Result<MxArray> {
        scheduled_slice(inner, seq, tokens, base, start, tokens.len())?
            .ok_or_else(|| Error::from_reason("the prompt's last unit gave no logits"))
    }

    /// Splash `configureDraftStatePlan` and `publishReachedStateBoundaries`
    /// for scheduled requests on a draft load whose draft is off: two
    /// sequences whose prefill slices interleave each publish the states a
    /// whole turn of their prompt does, each with the whole turn's GDN cell
    /// and draft window, the window carried across the slices on its own
    /// sequence; the turn's end gives the window to the idle lane.
    /// Mutations SD1 (a scheduled prefill on a draft load keeps no sink),
    /// SW1 (a slice starts its window empty), EW1 (the turn's end keeps the
    /// sequence's window) and MA1 (the accounts leave the sequences'
    /// windows out) → red.
    #[test]
    fn scheduled_ar_on_a_draft_load_publishes_draft_states() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "scheduled_ar_on_a_draft_load_publishes_draft_states",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8150;
        let prompts = [
            (1u32, prompt(150)),
            (
                2u32,
                (0..150u32).map(|i| (i * 5 + 1) % VOCAB).collect::<Vec<_>>(),
            ),
        ];
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            let mut references = Vec::new();
            for (_, tokens) in &prompts {
                references.push(whole_turn_states(seed, tokens)?.0);
            }
            let mut inner = wide_draft_inner(seed, 40);
            let mut bases = Vec::new();
            for (seq, tokens) in &prompts {
                let base = scheduled_prime(&mut inner, *seq, tokens)?;
                assert_eq!(base.effective_cached_prefix_len(), 0);
                bases.push(base);
            }
            for (start, end) in [(0, 48), (48, 112), (112, 150)] {
                for ((seq, tokens), base) in prompts.iter().zip(&bases) {
                    scheduled_slice(&mut inner, *seq, tokens, base, start, end)?;
                }
            }
            assert!(inner.dflash2.is_some(), "the draft is back");
            for ((seq, tokens), reference) in prompts.iter().zip(&references) {
                let depths: Vec<usize> = reference.iter().map(|state| state.0).collect();
                assert!(
                    depths.iter().any(|&depth| depth > 48 && depth - 40 < 48)
                        && depths.iter().any(|&depth| depth > 112 && depth - 40 < 112),
                    "a window spans each slice end: {depths:?}"
                );
                assert_eq!(
                    &draft_states(&inner, tokens)?,
                    reference,
                    "sequence {seq} publishes the whole turn's states"
                );
            }
            assert_eq!(inner.dflash2_states.seq_windows.len(), 2);
            let window = inner
                .dflash2
                .as_ref()
                .expect("draft")
                .config
                .context_window_bytes();
            let held = crate::models::qwen3_5::dflash2_turn::memory_accounts(&inner, 0).states;
            for (seq, _) in &prompts {
                HybridSchedulerBackend::scheduled_turn_closed(&mut inner, *seq);
            }
            assert_eq!(
                crate::models::qwen3_5::dflash2_turn::memory_accounts(&inner, 0).states,
                held - window,
                "the lanes' windows were on the books; the idle lane keeps one"
            );
            assert!(inner.dflash2_states.seq_windows.is_empty());
            assert!(inner.dflash2_states.turns.is_empty());
            assert!(inner.dflash2_states.idle_lane.window.is_some());
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// Splash `Cache::adopt` → `acquireDeepest` → `QwenStateStorage::restore`
    /// for a scheduled request on a draft load: after a restart, a scheduled
    /// request whose prompt extends a stored one resumes at its replay point
    /// (144) from disk with the stored window in its lane; the point it
    /// publishes 32 rows on (176) continues that window, and it and the
    /// logits equal a whole turn's of the longer prompt; the read turns
    /// resident.
    /// Mutations SR1 (the scheduled prime on a draft load resumes at no
    /// state), SR2 (its lane takes no stored window) and SP1 (the promotion
    /// snapshots the serial lane's window) → red.
    #[test]
    fn scheduled_ar_on_a_draft_load_resumes_after_a_restart() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "scheduled_ar_on_a_draft_load_resumes_after_a_restart",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        use crate::models::qwen3_5::dflash2_state::disk::tests::{
            TestTier, attach_tier, persistent_test_tier,
        };
        use crate::models::qwen3_5::dflash2_state::disk::{adopt_all, close_cache};
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8160;
        let on_tier = |tier: &TestTier| -> Result<Qwen35Inner> {
            let mut inner = wide_draft_inner(seed, 40);
            let window = inner
                .dflash2
                .as_ref()
                .expect("draft")
                .config
                .context_window()?;
            attach_tier(
                &mut inner.dflash2_states,
                &inner.config,
                inner.paged_adapter.as_ref().expect("pool"),
                tier,
                window.nbytes(),
                window.slots(),
            );
            inner.paged_adapter.as_mut().expect("pool").set_cold_tier(
                crate::models::qwen3_5::dflash2_state::disk::state_context(&tier.tier),
            );
            Ok(inner)
        };
        let first = prompt(150);
        let longer = prompt(180);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            let (reference, reference_logits) = whole_turn_states(seed, &longer)?;
            let tier = persistent_test_tier("state-tier-sched-draft-restart", 1 << 30);
            let mut inner = on_tier(&tier)?;
            let base = scheduled_prime(&mut inner, 1, &first)?;
            for (start, end) in [(0, 64), (64, 150)] {
                scheduled_slice(&mut inner, 1, &first, &base, start, end)?;
            }
            HybridSchedulerBackend::release_scheduled_cache(&mut inner, 1)?;
            HybridSchedulerBackend::scheduled_turn_closed(&mut inner, 1);
            let stored = draft_states(&inner, &first)?;
            let replay = stored
                .iter()
                .find(|state| state.0 == 144)
                .cloned()
                .expect("the replay point at 144");
            assert!(close_cache(
                &mut inner,
                std::time::Instant::now() + std::time::Duration::from_secs(30)
            ));
            drop(inner);

            let tier = tier.reopen("state-tier-sched-draft-restart", 1 << 30);
            let mut inner = on_tier(&tier)?;
            assert!(adopt_all(&mut inner) >= 1);
            let base = scheduled_prime(&mut inner, 4, &longer)?;
            assert_eq!(base.effective_cached_prefix_len(), 144);
            assert_eq!(
                window_state(
                    inner
                        .dflash2_states
                        .seq_windows
                        .get(&4)
                        .expect("the lane's window")
                )?,
                (replay.2, replay.3.clone()),
                "the lane holds the stored window"
            );
            let logits = scheduled_last_slice(&mut inner, 4, &longer, &base, 144)?;
            let disk = inner.dflash2_states.disk.as_ref().expect("tier").counters;
            assert_eq!((disk.promotions, disk.invalidations), (1, 0));
            let states = draft_states(&inner, &longer)?;
            assert!(
                states.contains(&replay),
                "the promoted state is the stored one"
            );
            let past: Vec<_> = states.into_iter().filter(|state| state.0 > 144).collect();
            let expected: Vec<_> = reference
                .into_iter()
                .filter(|state| state.0 > 144)
                .collect();
            assert!(
                expected.iter().any(|state| state.0 == 176),
                "the replay point at 176"
            );
            assert_eq!(
                past, expected,
                "the state past the resume continues the window"
            );
            assert_eq!(bits(&logits)?, reference_logits);
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// A draft ring: its logical length, its provenance and each layer's
    /// live K and V bits (empty for an empty layer).
    type RingState = (i32, Vec<u32>, Vec<Vec<u32>>);

    fn ring_state(ring: &DFlash2ContextCache) -> Result<RingState> {
        let mut views = Vec::new();
        for view in ring.window_views() {
            match view {
                Some((keys, values)) => {
                    views.push(bits(&keys)?);
                    views.push(bits(&values)?);
                }
                None => views.extend([Vec::new(), Vec::new()]),
            }
        }
        Ok((ring.logical_len(), ring.token_history().to_vec(), views))
    }

    /// The draft ring a whole DFlash2 turn of `tokens` on `inner` holds for
    /// its decode after its tapped prefill, the rows it resumed at, and
    /// whether its restore put rows in the ring.
    fn whole_turn_ring(
        inner: &mut Qwen35Inner,
        tokens: &[u32],
    ) -> Result<(usize, bool, RingState)> {
        let prefix = inner.prime_dspark_prefix_state(tokens, true, 0)?;
        let cached = prefix.effective_cached_prefix_len();
        let restored = inner
            .dflash2_context
            .as_ref()
            .is_some_and(|ring| ring.window_views().iter().any(Option::is_some));
        inner.paged_prefill_with_draft_state(&tokens[cached..], &prefix, Stream::generation())?;
        let turn = inner
            .dflash2_turn_state
            .as_ref()
            .ok_or_else(|| Error::from_reason("the whole turn left no ring"))?;
        Ok((cached, restored, ring_state(&turn.context)?))
    }

    /// A scheduled sequence with a DFlash2 decode lane on a draft load: two
    /// sequences whose prefill slices interleave each leave in their lane
    /// the ring a whole DFlash2 turn's tapped prefill of their prompt
    /// holds, bit for bit, its newest rows across two slices, with the
    /// lane's position at the prompt's end; the states they publish are the
    /// whole turn's. Released, both rings go to the idle lane.
    #[test]
    fn scheduled_dflash2_slices_fill_the_sequence_ring() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "scheduled_dflash2_slices_fill_the_sequence_ring",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8155;
        let prompts = [
            (1u32, prompt(150)),
            (
                2u32,
                (0..150u32).map(|i| (i * 5 + 1) % VOCAB).collect::<Vec<_>>(),
            ),
        ];
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            let mut references = Vec::new();
            for (_, tokens) in &prompts {
                let mut whole = wide_draft_inner(seed, 40);
                let (cached, _, ring) = whole_turn_ring(&mut whole, tokens)?;
                assert_eq!(cached, 0);
                assert!(
                    ring.2.iter().all(|layer| !layer.is_empty()),
                    "the whole turn's ring holds rows"
                );
                references.push((ring, draft_states(&whole, tokens)?));
            }
            let mut inner = wide_draft_inner(seed, 40);
            let mut bases = Vec::new();
            for (seq, tokens) in &prompts {
                let base = scheduled_prime(&mut inner, *seq, tokens)?;
                assert_eq!(base.effective_cached_prefix_len(), 0);
                assert!(HybridSchedulerBackend::begin_scheduled_speculation(
                    &mut inner, *seq, 0
                )?);
                bases.push(base);
            }
            for (start, end) in [(0, 48), (48, 112), (112, 150)] {
                for ((seq, tokens), base) in prompts.iter().zip(&bases) {
                    scheduled_slice(&mut inner, *seq, tokens, base, start, end)?;
                }
            }
            assert!(inner.dflash2.is_some(), "the draft is back");
            for ((seq, tokens), (ring, states)) in prompts.iter().zip(&references) {
                let lane = inner.dflash2_states.seq_lanes.get(seq).expect("the lane");
                assert_eq!(
                    &ring_state(&lane.ring)?,
                    ring,
                    "sequence {seq} holds the whole turn's ring"
                );
                assert_eq!(lane.next_position, 150);
                assert_eq!(
                    &draft_states(&inner, tokens)?,
                    states,
                    "sequence {seq} publishes the whole turn's states"
                );
            }
            for (seq, _) in &prompts {
                HybridSchedulerBackend::release_scheduled_speculation(&mut inner, *seq);
                HybridSchedulerBackend::scheduled_turn_closed(&mut inner, *seq);
            }
            assert!(inner.dflash2_states.seq_lanes.is_empty());
            assert!(inner.dflash2_states.idle_lane.ring.is_some());
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// A scheduled DFlash2 lane resumed at a stored state (144) holds the
    /// ring a whole DFlash2 turn resumed there holds after the same prompt:
    /// rebuilt from the state's window when the turn's first point (176)
    /// lies within a draft window (40) of the resume, from a RAM state and
    /// from the disk after a restart; empty at the resume when its first
    /// point (192) lies a window or more past it (Splash `planDraftContext`
    /// `restoresDraftState`, as `restore_rebuilds_the_draft_or_starts_empty`).
    #[test]
    fn a_scheduled_dflash2_lane_restores_its_ring_at_a_stored_state() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_scheduled_dflash2_lane_restores_its_ring_at_a_stored_state",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        use crate::models::qwen3_5::dflash2_state::disk::tests::{
            TestTier, attach_tier, persistent_test_tier,
        };
        use crate::models::qwen3_5::dflash2_state::disk::{adopt_all, close_cache};
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8165;
        let on_tier = |tier: &TestTier| -> Result<Qwen35Inner> {
            let mut inner = wide_draft_inner(seed, 40);
            let window = inner
                .dflash2
                .as_ref()
                .expect("draft")
                .config
                .context_window()?;
            attach_tier(
                &mut inner.dflash2_states,
                &inner.config,
                inner.paged_adapter.as_ref().expect("pool"),
                tier,
                window.nbytes(),
                window.slots(),
            );
            inner.paged_adapter.as_mut().expect("pool").set_cold_tier(
                crate::models::qwen3_5::dflash2_state::disk::state_context(&tier.tier),
            );
            Ok(inner)
        };
        let first = prompt(150);
        let near = prompt(180);
        let mut far = first[..144].to_vec();
        far.extend((0..56u32).map(|i| (i * 5 + 2) % VOCAB));
        // The first turn publishes its replay point at 144.
        let stored = |inner: &mut Qwen35Inner| -> Result<()> {
            let base = scheduled_prime(inner, 1, &first)?;
            for (start, end) in [(0, 64), (64, 150)] {
                scheduled_slice(inner, 1, &first, &base, start, end)?;
            }
            HybridSchedulerBackend::release_scheduled_cache(inner, 1)?;
            HybridSchedulerBackend::scheduled_turn_closed(inner, 1);
            Ok(())
        };
        let lane_ring = |inner: &mut Qwen35Inner, tokens: &[u32]| -> Result<RingState> {
            let base = scheduled_prime(inner, 4, tokens)?;
            assert_eq!(base.effective_cached_prefix_len(), 144);
            assert!(HybridSchedulerBackend::begin_scheduled_speculation(
                inner, 4, 144
            )?);
            scheduled_slice(inner, 4, tokens, &base, 144, tokens.len())?;
            let lane = inner.dflash2_states.seq_lanes.get(&4).expect("the lane");
            assert_eq!(lane.next_position as usize, tokens.len());
            ring_state(&lane.ring)
        };
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(64)));
        let result = (|| -> Result<()> {
            let mut references = Vec::new();
            for (tokens, restores) in [(&near, true), (&far, false)] {
                let mut whole = wide_draft_inner(seed, 40);
                stored(&mut whole)?;
                let (cached, restored, ring) = whole_turn_ring(&mut whole, tokens)?;
                assert_eq!((cached, restored), (144, restores));
                references.push(ring);
            }
            for (tokens, reference, case) in [
                (&near, &references[0], "near, RAM"),
                (&far, &references[1], "far, RAM"),
            ] {
                let mut inner = wide_draft_inner(seed, 40);
                stored(&mut inner)?;
                assert_eq!(&lane_ring(&mut inner, tokens)?, reference, "{case}");
            }

            let tier = persistent_test_tier("state-tier-sched-dflash2-ring", 1 << 30);
            let mut inner = on_tier(&tier)?;
            stored(&mut inner)?;
            assert!(close_cache(
                &mut inner,
                std::time::Instant::now() + std::time::Duration::from_secs(30)
            ));
            drop(inner);
            let tier = tier.reopen("state-tier-sched-dflash2-ring", 1 << 30);
            let mut inner = on_tier(&tier)?;
            assert!(adopt_all(&mut inner) >= 1);
            assert_eq!(&lane_ring(&mut inner, &near)?, &references[0], "near, disk");
            let disk = inner.dflash2_states.disk.as_ref().expect("tier").counters;
            assert_eq!((disk.promotions, disk.invalidations), (1, 0));
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// The states a whole autoregressive turn of `tokens` on a draft load
    /// publishes when its generation prompt is the template's.
    fn whole_turn_states_with_prompt(seed: u64, tokens: &[u32]) -> Result<Vec<DraftState>> {
        let mut inner = wide_draft_inner(seed, 40);
        inner.dflash2_states.generation_suffix =
            Some(std::sync::Arc::from(&super::tests::GENERATION_PROMPT[..]));
        let prefix = inner.prime_dspark_prefix_state(tokens, true, 0)?;
        assert_eq!(prefix.effective_cached_prefix_len(), 0);
        inner.paged_prefill_with_draft_state(tokens, &prefix, Stream::generation())?;
        draft_states(&inner, tokens)
    }

    /// Splash `Engine::submit` → `admit` → `configureDraftStatePlan` →
    /// `publishReachedStateBoundaries` → `release` through the model
    /// thread's scheduler on a draft load, the sessions' draft off: two
    /// sessions started together
    /// run as two live sequences, and each publishes the states, GDN cells
    /// and draft windows a whole turn of its own prompt does; their windows
    /// go back to the idle lane at their ends. After a restart (Splash
    /// `Cache::adopt` → `acquireDeepest` → `QwenStateStorage::restore`) a
    /// session whose prompt extends the first one's resumes at that one's
    /// replay point (144) from disk, and the states it publishes past it,
    /// windows reaching below 144 included, are the whole turn's of the
    /// longer prompt.
    /// Mutation DS1 (the scheduled prime on a draft load primes plain) → red.
    #[test]
    fn scheduler_runs_draft_load_ar_sessions_on_the_state_tier() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "scheduler_runs_draft_load_ar_sessions_on_the_state_tier",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        use super::tests::{content_tokenizer, drive_session_starts, rendered_prompt};
        use crate::engine::governor::governor::Governor;
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        use crate::engine::hybrid_scheduler::HybridSchedulerState;
        use crate::models::qwen3_5::dflash2_state::disk::tests::{
            TestTier, attach_tier, persistent_test_tier,
        };
        use crate::models::qwen3_5::dflash2_state::disk::{adopt_all, close_cache};
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_81A0;
        let scheduler_on = |tier: &TestTier| -> Result<HybridSchedulerState<Qwen35Inner>> {
            let mut inner = wide_draft_inner(seed, 40);
            let window = inner
                .dflash2
                .as_ref()
                .expect("draft")
                .config
                .context_window()?;
            attach_tier(
                &mut inner.dflash2_states,
                &inner.config,
                inner.paged_adapter.as_ref().expect("pool"),
                tier,
                window.nbytes(),
                window.slots(),
            );
            inner.paged_adapter.as_mut().expect("pool").set_cold_tier(
                crate::models::qwen3_5::dflash2_state::disk::state_context(&tier.tier),
            );
            inner.tokenizer = Some(content_tokenizer());
            let io = FakeIo::new(
                stats(0),
                host(32 * G, crate::engine::governor::host::Pressure::Normal),
            );
            let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
            inner.memory_registration =
                Some(governor.register_model(model(16 * G, 0)).expect("register"));
            let mut state = HybridSchedulerState::new(inner)?;
            state.enable_continuous_batching();
            Ok(state)
        };
        let first = prompt(150);
        let second: Vec<u32> = (0..150u32).map(|i| (i * 5 + 1) % VOCAB).collect();
        let longer = prompt(180);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            let references = [
                whole_turn_states_with_prompt(seed, &rendered_prompt(&first))?,
                whole_turn_states_with_prompt(seed, &rendered_prompt(&second))?,
            ];
            for reference in &references {
                assert_eq!(
                    reference.iter().map(|state| state.0).collect::<Vec<_>>(),
                    vec![80, 144]
                );
            }
            let longer_reference = whole_turn_states_with_prompt(seed, &rendered_prompt(&longer))?;
            let tier = persistent_test_tier("state-tier-sched-drive-draft", 1 << 30);
            let mut state = scheduler_on(&tier)?;
            let (results, most_live) =
                drive_session_starts(&mut state, &[("a", &first), ("b", &second)], 2)?;
            assert_eq!(most_live, 2, "the two sessions ran as two live sequences");
            for (result, (body, reference)) in results
                .iter()
                .zip([&first, &second].into_iter().zip(&references))
            {
                assert_eq!(result.cached_tokens, 0);
                assert_eq!(
                    &draft_states(&state.inner, &rendered_prompt(body))?,
                    reference,
                    "each session publishes its own whole turn's states"
                );
            }
            assert!(state.inner.dflash2.is_some(), "the draft is back");
            assert!(state.inner.dflash2_states.seq_windows.is_empty());
            assert!(state.inner.dflash2_states.turns.is_empty());
            assert!(state.inner.dflash2_states.idle_lane.window.is_some());
            let replay = draft_states(&state.inner, &rendered_prompt(&first))?
                .into_iter()
                .find(|stored| stored.0 == 144)
                .expect("the replay point at 144");
            assert!(close_cache(
                &mut state.inner,
                std::time::Instant::now() + std::time::Duration::from_secs(30)
            ));
            drop(state);

            let tier = tier.reopen("state-tier-sched-drive-draft", 1 << 30);
            let mut state = scheduler_on(&tier)?;
            assert!(adopt_all(&mut state.inner) >= 1);
            let (results, _) = drive_session_starts(&mut state, &[("c", &longer)], 2)?;
            assert_eq!(
                results[0].cached_tokens, 144,
                "the session resumed from disk"
            );
            let disk = state
                .inner
                .dflash2_states
                .disk
                .as_ref()
                .expect("tier")
                .counters;
            assert_eq!((disk.promotions, disk.invalidations), (1, 0));
            let states = draft_states(&state.inner, &rendered_prompt(&longer))?;
            assert!(
                states.contains(&replay),
                "the promoted state is the stored one"
            );
            let past: Vec<_> = states.into_iter().filter(|stored| stored.0 > 144).collect();
            let expected: Vec<_> = longer_reference
                .into_iter()
                .filter(|stored| stored.0 > 144)
                .collect();
            assert!(
                expected.iter().any(|stored| stored.2.0 < 144),
                "a window past the resume reaches below it"
            );
            assert_eq!(
                past, expected,
                "the states past the resume continue the stored window"
            );
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// Splash ranks a queued request by a cache probe: session c, whose
    /// prompt extends session b's stored conversation and is longer than
    /// session a's cold prompt, ranks by the rows it resumes at. It leads
    /// a's prefill, which needs two units, and answers while a still
    /// prefills. Ranked as uncached, c would wait out a's whole prefill.
    #[test]
    fn a_resumed_start_leads_a_long_prefill() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("a_resumed_start_leads_a_long_prefill")
            || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        use super::tests::{
            content_tokenizer, drive_session_starts, rendered_prompt, session_start,
        };
        use crate::engine::governor::governor::Governor;
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        use crate::engine::hybrid_scheduler::{HybridSchedulerBackend, HybridSchedulerState};
        use crate::engine::model_command::ModelCommand;
        use crate::model_thread::LoopControl;
        let _ctx = StreamContext::new(Stream::generation());
        // A window past one 2,048-row unit, so a cold prompt takes two.
        let mut inner = tiny_paged_int8_dflash_inner_with(0x5_81B0, Heads::Same, |config| {
            config.max_position_embeddings = 4_096;
        });
        inner.tokenizer = Some(content_tokenizer());
        inner.grant_scheduled_lanes(4);
        let io = FakeIo::new(
            stats(0),
            host(32 * G, crate::engine::governor::host::Pressure::Normal),
        );
        let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
        inner.memory_registration =
            Some(governor.register_model(model(16 * G, 0)).expect("register"));
        let mut state = HybridSchedulerState::new(inner)?;
        state.enable_continuous_batching();

        drive_session_starts(&mut state, &[("b", &prompt(2_560))], 1)?;
        let resumed = rendered_prompt(&prompt(2_600));
        let cold_body: Vec<u32> = (0..2_500u32).map(|i| (i * 5 + 1) % VOCAB).collect();
        let cold = rendered_prompt(&cold_body);
        let depth =
            crate::models::qwen3_5::dflash2_turn::stored_resume_depth(&state.inner, &resumed, 0)
                as usize;
        assert!(
            cold.len() < resumed.len() && resumed.len() - depth < cold.len(),
            "c is longer than a but resumes past a's length: {depth} of {} stored",
            resumed.len()
        );

        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let mut replies = Vec::new();
        for (owner, body) in [("a", &cold_body), ("c", &prompt(2_600))] {
            let (command, reply) = session_start(owner, body, 1)?;
            sender
                .send(ModelCommand::<
                    <Qwen35Inner as HybridSchedulerBackend>::FamilyCommand,
                >::from_chat(command))
                .map_err(|_| Error::from_reason("the command channel closed"))?;
            replies.push(reply);
        }
        let [mut a, mut c] =
            <[_; 2]>::try_from(replies).map_err(|_| Error::from_reason("two replies"))?;
        let (mut a_done, mut c_done) = (None, None);
        let mut a_prefilling_at_c = None;
        for _ in 0..256 {
            if !matches!(state.drive(&mut receiver), LoopControl::Continue) {
                return Err(Error::from_reason("the scheduler stopped"));
            }
            if c_done.is_none()
                && let Ok(reply) = c.try_recv()
            {
                c_done = Some(reply?);
                let seq = state.owner_sequences.get("a").copied();
                a_prefilling_at_c = Some(
                    seq.and_then(|seq| state.turn_mut(seq))
                        .is_some_and(|turn| turn.num_computed_tokens < turn.prompt_tokens),
                );
            }
            if a_done.is_none()
                && let Ok(reply) = a.try_recv()
            {
                a_done = Some(reply?);
            }
            if a_done.is_some() && c_done.is_some() {
                break;
            }
        }
        let (Some(a_done), Some(c_done)) = (a_done, c_done) else {
            return Err(Error::from_reason("the sessions did not finish"));
        };
        assert_eq!(
            c_done.cached_tokens as usize, depth,
            "c resumed at its state"
        );
        assert_eq!(a_done.cached_tokens, 0, "a was cold");
        assert_eq!(
            a_prefilling_at_c,
            Some(true),
            "c answered while a still prefilled"
        );
        Ok(())
    }

    /// A scheduled lane whose window did not come back with its state
    /// draws no point's window from rows below its resume (48): only the
    /// points a whole window (40) past it stay.
    /// Mutation KD1 (every point stays) → red.
    #[test]
    fn a_lane_without_its_window_keeps_only_drawable_points() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_lane_without_its_window_keeps_only_drawable_points",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let mut inner = wide_draft_inner(0x5_8170, 40);
        let tokens = prompt(150);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            scheduled_prime(&mut inner, 3, &tokens)?;
            HybridSchedulerBackend::activate_paged_seq(&mut inner, 3)?;
            let points = |inner: &Qwen35Inner| -> Vec<u32> {
                inner.dflash2_states.turns[&3]
                    .plan
                    .boundaries
                    .iter()
                    .map(|point| point.tokens)
                    .collect()
            };
            let before = points(&inner);
            assert_eq!(before, vec![80, 144]);
            crate::models::qwen3_5::dflash2_turn::keep_drawable_points(&mut inner, 48, 48)?;
            assert_eq!(points(&inner), before, "a lane at its resume keeps them");
            crate::models::qwen3_5::dflash2_turn::keep_drawable_points(&mut inner, 48, 0)?;
            assert_eq!(points(&inner), vec![144]);
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// The tapped prefill ends a chunk at the prompt's replay point (Splash
    /// `Engine::workEndOf`) and parks the GDN cell and draft window there,
    /// before the generation prompt (Splash `Engine::promptReplayBoundary`).
    /// A twin prompt cut right after that row parks the same state bits.
    /// Mutation CA1 (the capture row is no chunk break) → red (no chunk ends
    /// at 192, nothing parked).
    #[test]
    fn tapped_prefill_parks_the_replay_point() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("tapped_prefill_parks_the_replay_point") {
            return Ok(());
        }
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let suffix = [11u32, 12, 13, 14, 15];
        let mut full = prompt(205);
        full.extend(suffix);
        let prefill = |tokens: &[u32]| -> Result<(Vec<usize>, Qwen35Inner)> {
            let mut inner = tiny_paged_int8_dflash_inner(0x5_8103);
            inner.dflash2_states.generation_suffix = Some(std::sync::Arc::from(&suffix[..]));
            PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(128)));
            PREFILL_CHUNK_BLOCKS.with(|log| *log.borrow_mut() = Some(Vec::new()));
            let result = (|| -> Result<()> {
                let prefix = inner.prime_dspark_prefix_state(tokens, true, 0)?;
                inner
                    .paged_prefill_with_draft_state(tokens, &prefix, stream)?
                    .eval();
                Ok(())
            })();
            PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
            let seen = PREFILL_CHUNK_BLOCKS
                .with(|log| log.borrow_mut().take())
                .unwrap_or_default();
            result?;
            Ok((seen, inner))
        };
        let (seen, inner) = prefill(&full)?;
        assert_eq!(seen, vec![8, 12, 14], "chunks end at 128 and 192");
        let (twin_seen, twin) = prefill(&full[..193])?;
        assert_eq!(twin_seen, vec![8, 12, 13]);
        let anchor_bits = |inner: &Qwen35Inner| -> Result<Vec<Vec<u32>>> {
            let entry = inner
                .dflash2_states
                .store
                .entries()
                .next()
                .ok_or_else(|| Error::from_reason("no parked state"))?;
            assert_eq!(entry.tokens(), &full[..192]);
            let anchor = entry.ram().expect("a RAM copy");
            assert_eq!(anchor.window.as_ref().expect("a window").end(), 192);
            let mut arrays = Vec::new();
            anchor.collect_arrays(&mut arrays);
            arrays.into_iter().map(bits).collect()
        };
        assert!(anchor_bits(&inner)? == anchor_bits(&twin)?);
        Ok(())
    }

    /// The tapped prefill of `prompt` on a fresh fixture with 4-row chunks.
    fn tapped_prefill_by_four(inner: &mut Qwen35Inner, prompt: &[u32]) -> Result<()> {
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(4)));
        let result = (|| -> Result<()> {
            let prefix = inner.prime_dspark_prefix_state(prompt, true, 0)?;
            inner
                .paged_prefill_with_draft_state(prompt, &prefix, Stream::generation())?
                .eval();
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// A 100-token prompt with no generation prompt plans checkpoints at
    /// 16..80 and its replay point at 96 (draft window 8: interval 16,
    /// margin 4). Each checkpoint retires the one before it (Splash
    /// `Engine::retireCheckpoint`); the replay point keeps the lane's last
    /// one. The store ends with 80 (checkpoint) and 96 (ordinary, in use),
    /// each on a cached block.
    /// Mutations RC1 (no checkpoint retires: 6 entries) and RP1 (the replay
    /// point retires it too: 80 gone) → red.
    #[test]
    fn rolling_checkpoint_keeps_one_and_the_one_below_r() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "rolling_checkpoint_keeps_one_and_the_one_below_r",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8130);
        tapped_prefill_by_four(&mut inner, &prompt(100))?;
        let states = &inner.dflash2_states;
        let entries: Vec<(usize, bool, bool)> = states
            .store
            .entries()
            .map(|entry| {
                (
                    entry.depth(),
                    entry.checkpoint(),
                    states.store.in_use(entry.mark()),
                )
            })
            .collect();
        assert_eq!(entries, vec![(80, true, false), (96, false, true)]);
        assert_eq!(states.counters.checkpoint_publications, 5);
        assert_eq!(states.counters.checkpoint_retirements, 4);
        assert_eq!(states.counters.replay_state_publications, 1);
        let adapter = inner.paged_adapter.as_ref().expect("pool");
        assert!(
            states
                .store
                .entries()
                .all(|entry| adapter.prefix_cached(entry.mark())),
            "each state sits on a cached block"
        );
        Ok(())
    }

    /// A prefill cancelled after the chunk ending at 52 errors as a cancel,
    /// and what earlier chunks published stays (Splash publishes after each
    /// command): the checkpoint at 48 on cached blocks. The next turn of the
    /// prompt resumes there and emits paged AR's greedy tokens.
    /// Mutation CN1 (the abort removes the turn's publications) → red.
    #[test]
    fn cancelled_prefill_keeps_reached_states() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("cancelled_prefill_keeps_reached_states")
        {
            return Ok(());
        }
        use crate::models::qwen3_5::paged_forward::PREFILL_CANCEL_AT;
        let _ctx = StreamContext::new(Stream::generation());
        let prompt = prompt(100);
        let (reference, _) = paged_ar(&mut tiny_paged_int8_dflash_inner(0x5_8131), &prompt, 24)?;
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8131);
        PREFILL_CANCEL_AT.with(|row| row.set(Some(52)));
        let result = tapped_prefill_by_four(&mut inner, &prompt);
        PREFILL_CANCEL_AT.with(|row| row.set(None));
        let error = result.expect_err("the prefill is cancelled");
        assert_eq!(error.reason, "prefill cancelled");
        inner.abort_paged_turn();
        assert_eq!(parked_depths(&inner), vec![48]);
        let adapter = inner.paged_adapter.as_ref().expect("pool");
        assert!(
            inner
                .dflash2_states
                .store
                .entries()
                .all(|entry| adapter.prefix_cached(entry.mark())),
            "the checkpoint's blocks stay cached"
        );
        let run = paged_dflash2(&mut inner, &prompt, &reference, 3)?;
        assert_eq!(run.cached, 48);
        assert_eq!(run.tokens, reference);
        Ok(())
    }

    /// A fixture whose claims go through a ledger of its own: 32 GiB of
    /// host memory, normal pressure, nothing scripted or logged yet.
    fn with_ledger(inner: &mut Qwen35Inner) {
        use crate::engine::governor::governor::Governor;
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        use crate::models::qwen3_5::dflash2_turn::{STATE_CLAIM_LOG, STATE_CLAIM_SCRIPT};
        let io = FakeIo::new(
            stats(0),
            host(32 * G, crate::engine::governor::host::Pressure::Normal),
        );
        let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
        inner.memory_registration =
            Some(governor.register_model(model(16 * G, 0)).expect("register"));
        STATE_CLAIM_LOG.with(|log| log.borrow_mut().clear());
        STATE_CLAIM_SCRIPT.with(|script| script.borrow_mut().clear());
    }

    fn take_claim_log() -> Vec<crate::models::qwen3_5::dflash2_turn::ClaimLog> {
        crate::models::qwen3_5::dflash2_turn::STATE_CLAIM_LOG
            .with(|log| std::mem::take(&mut *log.borrow_mut()))
    }

    fn script_claims(claims: &[crate::models::qwen3_5::dflash2_turn::ScriptedClaim]) {
        crate::models::qwen3_5::dflash2_turn::STATE_CLAIM_SCRIPT
            .with(|script| *script.borrow_mut() = claims.iter().copied().collect());
    }

    /// Splash `Engine::stateClaim` per captured point of the 100-token
    /// prompt: checkpoints 16..80 Optional and Immediate, the replay point
    /// 96 in service and reaching the states turns use; the first claim
    /// settles, every later one tries once, since the lane then holds a
    /// checkpoint to take it from (`reserveCapture`). The next turn of the
    /// prompt claims the lane's buffers for its restore under the turn's
    /// claim, only what the idle lane lacks (Splash `admitIdleLane`): the
    /// cell beside an idle ring and window, all three without them. A
    /// failed claim sends the turn cold; one refused while memory is on its
    /// way back waits and restores (Splash `admit`: `judge`).
    /// Mutations SC1 (checkpoints claimed as InService), RB2 (the restore
    /// claims the ring and window beside idle ones) and RW1 (every refused
    /// restore goes cold) → red.
    #[test]
    fn capture_claims_follow_state_claim() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("capture_claims_follow_state_claim") {
            return Ok(());
        }
        use crate::engine::governor::allocate::Refusal;
        use crate::engine::governor::ledger::{ClaimClass, NeedTiming, ReachClass};
        use crate::models::qwen3_5::dflash2_state::paged::anchor_bytes;
        use crate::models::qwen3_5::dflash2_turn::ScriptedClaim;
        let _ctx = StreamContext::new(Stream::generation());
        let prompt = prompt(100);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8140);
        with_ledger(&mut inner);
        tapped_prefill_by_four(&mut inner, &prompt)?;
        let log = take_claim_log();
        let optional = (
            ClaimClass::Optional,
            NeedTiming::Immediate,
            ReachClass::Ordinary,
        );
        let in_use = (
            ClaimClass::InService,
            NeedTiming::CanWait,
            ReachClass::InUse,
        );
        assert_eq!(
            log.iter()
                .map(|claim| (claim.cls, claim.timing, claim.reach))
                .collect::<Vec<_>>(),
            [vec![optional; 5], vec![in_use]].concat()
        );
        assert_eq!(
            log.iter().map(|claim| claim.refusal).collect::<Vec<_>>(),
            [vec![Refusal::Settled], vec![Refusal::SettledOnce; 5]].concat()
        );
        inner.abort_paged_turn();
        assert_eq!(parked_depths(&inner), vec![80, 96]);

        let restore_log = |log: Vec<crate::models::qwen3_5::dflash2_turn::ClaimLog>| {
            log.iter()
                .map(|claim| (claim.cls, claim.refusal, claim.bytes))
                .collect::<Vec<_>>()
        };
        let cell = inner.config.recurrent_state_bytes();
        script_claims(&[ScriptedClaim::Refuse]);
        let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
        assert_eq!(
            restore_log(take_claim_log()),
            vec![(ClaimClass::InService, Refusal::Waits, cell)],
            "the idle lane holds the ring and the window"
        );
        assert_eq!(
            prefix.effective_cached_prefix_len(),
            0,
            "a failed restore goes cold"
        );
        inner.abort_paged_turn();

        inner.dflash2_states.idle_lane = Default::default();
        script_claims(&[ScriptedClaim::Wait, ScriptedClaim::Grant { freed: false }]);
        let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
        let draft = inner.dflash2.as_ref().expect("draft");
        let bytes = anchor_bytes(&inner.config, Some(&draft.config)) + draft.config.kv_ring_bytes();
        assert_eq!(
            restore_log(take_claim_log()),
            vec![(ClaimClass::InService, Refusal::Waits, bytes); 2]
        );
        assert_eq!(
            prefix.effective_cached_prefix_len(),
            96,
            "a restore refused while memory is on its way back waits"
        );
        inner.abort_paged_turn();
        Ok(())
    }

    /// The ledger refuses the capture of checkpoint 32 while the lane holds
    /// 16 (one try): the chunk takes no snapshot, 16 retires to give its
    /// buffers, and 32 is taken from the lane after the chunk under a claim
    /// of its own (Splash `fromLane`), recycled when that claim freed units.
    /// Refused that claim too, 32 is not published and the lane holds no
    /// checkpoint until 48.
    /// Mutations FL1 (`FromLane` treated as `Refused`) and FL2 (the chunk
    /// snapshots a `FromLane` point) → red.
    #[test]
    fn refused_checkpoint_comes_from_the_lane() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("refused_checkpoint_comes_from_the_lane")
        {
            return Ok(());
        }
        use crate::models::qwen3_5::dflash2_turn::{LANE_SNAPSHOTS, ScriptedClaim};
        let _ctx = StreamContext::new(Stream::generation());
        let prompt = prompt(100);
        let run = |claims: &[ScriptedClaim]| -> Result<(Qwen35Inner, usize)> {
            let mut inner = tiny_paged_int8_dflash_inner(0x5_8141);
            with_ledger(&mut inner);
            script_claims(claims);
            LANE_SNAPSHOTS.with(|count| count.set(0));
            tapped_prefill_by_four(&mut inner, &prompt)?;
            Ok((inner, LANE_SNAPSHOTS.with(std::cell::Cell::get)))
        };
        let entries = |inner: &Qwen35Inner| -> Vec<(usize, bool)> {
            inner
                .dflash2_states
                .store
                .entries()
                .map(|entry| (entry.depth(), entry.checkpoint()))
                .collect()
        };
        let granted = ScriptedClaim::Grant { freed: false };
        for freed in [true, false] {
            let (inner, snapshots) = run(&[
                granted,
                ScriptedClaim::Refuse,
                ScriptedClaim::Grant { freed },
            ])?;
            let counters = inner.dflash2_states.counters;
            assert_eq!(snapshots, 6, "five armed points and one from the lane");
            assert_eq!(entries(&inner), vec![(80, true), (96, false)]);
            assert_eq!(counters.checkpoint_publications, 5);
            assert_eq!(counters.checkpoint_retirements, 4);
            assert_eq!(counters.checkpoint_publication_failures, 0);
            assert_eq!(counters.recycled_state_publications, u64::from(freed));
        }
        let (inner, snapshots) = run(&[granted, ScriptedClaim::Refuse, ScriptedClaim::Refuse])?;
        let counters = inner.dflash2_states.counters;
        assert_eq!(snapshots, 5);
        assert_eq!(entries(&inner), vec![(80, true), (96, false)]);
        assert_eq!(counters.checkpoint_publications, 4);
        assert_eq!(counters.checkpoint_publication_failures, 1);
        assert_eq!(
            counters.checkpoint_retirements, 3,
            "16 retired for 32; 48 found no checkpoint to retire"
        );
        Ok(())
    }

    /// The chunk ending at checkpoint 16 grows the pool (its one block is
    /// held by another request), and the ledger has room for the grow or
    /// for the checkpoint's buffers, not both, with nothing to reclaim. The
    /// chunk's KV is booked before the capture is claimed (Splash
    /// `Engine::prepare`: `growLane`, then `reserveCaptures`), so the
    /// prefill completes and every state is published.
    /// Mutation KV1 (the capture is claimed before the chunk's KV) → red
    /// (the held capture leaves the grow no room: capacity_exhausted).
    #[test]
    fn chunk_kv_is_booked_before_its_capture() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("chunk_kv_is_booked_before_its_capture") {
            return Ok(());
        }
        use crate::engine::governor::governor::Governor;
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        use crate::models::qwen3_5::model::dflash2_park_tests::{install_pool, passing_probe};
        use crate::transformer::paged_kv_cache_adapter::pool_grow_reserved_bytes;
        let _ctx = StreamContext::new(Stream::generation());
        let prompt = prompt(100);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8142);
        install_pool(&mut inner, 1, 16, passing_probe())?;
        let grow = {
            let adapter = inner.paged_adapter.as_mut().expect("pool");
            let block_size = adapter.block_size();
            assert_eq!(block_size, 16, "the chunk ending at 16 takes one block");
            adapter.set_extent_growth(15);
            adapter.begin_request(9).map_err(Error::from_reason)?;
            adapter
                .allocate_suffix_blocks_for(9, block_size)
                .map_err(Error::from_reason)?;
            let pool = adapter.layer_kv_pool();
            let layers = pool.layer_bytes();
            let new_layer = pool.layer_bytes_for(16).map_err(Error::from_reason)?;
            new_layer * layers.len() as u64 - pool.total_bytes()
                + pool_grow_reserved_bytes(&layers, new_layer)
        };
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
            let anchor = inner
                .dflash2_states
                .turns
                .get(&0)
                .expect("the turn's plan")
                .anchor_bytes;
            let limit = grow.max(anchor) + grow.min(anchor) / 2;
            let io = FakeIo::new(
                stats(0),
                host(32 * G, crate::engine::governor::host::Pressure::Normal),
            );
            let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
            inner.memory_registration =
                Some(governor.register_model(model(limit, 0)).expect("register"));
            inner
                .paged_adapter
                .as_mut()
                .expect("pool")
                .set_admission(std::sync::Arc::new(governor));
            inner
                .paged_prefill_with_draft_state(&prompt, &prefix, Stream::generation())?
                .eval();
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result?;
        let counters = inner.dflash2_states.counters;
        assert_eq!(counters.checkpoint_publications, 5);
        assert_eq!(counters.replay_state_publications, 1);
        assert_eq!(counters.checkpoint_publication_failures, 0);
        let telemetry = inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .block_telemetry()
            .expect("telemetry");
        assert_eq!(telemetry.total_blocks, 16, "the chunk grew the pool");
        Ok(())
    }

    /// The ledger refuses both the capture of checkpoint 32 and its claim
    /// from the lane: no RAM copy takes it, so the tier takes it from the
    /// lane (Splash `publishStateToDisk` with `snapshotToDisk`). 32 is a
    /// disk-only checkpoint whose object, once landed, holds the bytes a
    /// granted twin's RAM state at 32 encodes to; the replay point 48 keeps
    /// it. With the staging buffer out (a write in flight) 32 fails as
    /// before the tier.
    /// Mutation DP1 (no disk arm) → red.
    #[test]
    fn refused_capture_goes_to_disk_from_the_lane() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "refused_capture_goes_to_disk_from_the_lane",
        ) {
            return Ok(());
        }
        use crate::models::qwen3_5::dflash2_state::disk::tests::{attach_tier, test_tier};
        use crate::models::qwen3_5::dflash2_state::disk::{
            encode_anchor, hold_staging, poll_offload, read_object,
        };
        use crate::models::qwen3_5::dflash2_turn::ScriptedClaim;
        let _ctx = StreamContext::new(Stream::generation());
        let prompt = prompt(52);
        let tier = test_tier("state-tier-from-lane", 1 << 30);
        let run = |claims: &[ScriptedClaim], held: bool| -> Result<Qwen35Inner> {
            let mut inner = tiny_paged_int8_dflash_inner(0x5_8191);
            with_ledger(&mut inner);
            let window = inner
                .dflash2
                .as_ref()
                .expect("draft")
                .config
                .context_window()?;
            attach_tier(
                &mut inner.dflash2_states,
                &inner.config,
                inner.paged_adapter.as_ref().expect("pool"),
                &tier,
                window.nbytes(),
                window.slots(),
            );
            if held {
                assert!(hold_staging(&mut inner.dflash2_states).is_some());
            }
            script_claims(claims);
            tapped_prefill_by_four(&mut inner, &prompt)?;
            Ok(inner)
        };
        let entries = |inner: &Qwen35Inner| -> Vec<(usize, bool, bool)> {
            inner
                .dflash2_states
                .store
                .entries()
                .map(|entry| (entry.depth(), entry.checkpoint(), entry.ram().is_some()))
                .collect()
        };
        let granted = ScriptedClaim::Grant { freed: false };

        let twin = run(&[], false)?;
        assert_eq!(entries(&twin), vec![(32, true, true), (48, false, true)]);
        let want = {
            let entry = twin
                .dflash2_states
                .store
                .entries()
                .find(|entry| entry.depth() == 32)
                .expect("the twin's 32");
            encode_anchor(
                &twin.dflash2_states,
                &twin.config,
                entry.ram().expect("a RAM copy"),
            )
            .expect("the twin's bytes")
        };

        let mut inner = run(
            &[granted, ScriptedClaim::Refuse, ScriptedClaim::Refuse],
            false,
        )?;
        let counters = inner.dflash2_states.counters;
        assert_eq!(counters.disk_state_publications, 1);
        assert_eq!(counters.checkpoint_publication_failures, 0);
        assert_eq!(entries(&inner), vec![(32, true, false), (48, false, true)]);
        assert!(tier.tier.manager.drain(std::time::Duration::from_secs(10)));
        poll_offload(&mut inner);
        let id = inner
            .dflash2_states
            .store
            .entries()
            .find(|entry| entry.depth() == 32)
            .map(|entry| entry.id())
            .expect("32");
        assert!(
            inner
                .dflash2_states
                .store
                .get(id)
                .and_then(|entry| entry.disk())
                .is_some_and(|copy| copy.landed)
        );
        assert!(
            read_object(&inner.dflash2_states, id) == Some(want),
            "the object holds the lane's state at 32"
        );

        let inner = run(
            &[granted, ScriptedClaim::Refuse, ScriptedClaim::Refuse],
            true,
        )?;
        let counters = inner.dflash2_states.counters;
        assert_eq!(counters.disk_state_publications, 0);
        assert_eq!(counters.checkpoint_publication_failures, 1);
        assert_eq!(entries(&inner), vec![(48, false, true)]);
        Ok(())
    }

    /// A fresh fixture whose states and cold KV are on `tier`; with
    /// `ledger` its claims go through a ledger of its own.
    fn disk_fixture(
        seed: u64,
        tier: &crate::models::qwen3_5::dflash2_state::disk::tests::TestTier,
        ledger: bool,
    ) -> Result<Qwen35Inner> {
        use crate::models::qwen3_5::dflash2_state::disk::tests::attach_tier;
        let mut inner = tiny_paged_int8_dflash_inner(seed);
        if ledger {
            with_ledger(&mut inner);
        }
        let window = inner
            .dflash2
            .as_ref()
            .expect("draft")
            .config
            .context_window()?;
        attach_tier(
            &mut inner.dflash2_states,
            &inner.config,
            inner.paged_adapter.as_ref().expect("pool"),
            tier,
            window.nbytes(),
            window.slots(),
        );
        inner.paged_adapter.as_mut().expect("pool").set_cold_tier(
            crate::models::qwen3_5::dflash2_state::disk::state_context(&tier.tier),
        );
        Ok(inner)
    }

    fn state_at(
        inner: &Qwen35Inner,
        depth: usize,
    ) -> Option<crate::models::qwen3_5::dflash2_state::store::EntryId> {
        inner
            .dflash2_states
            .store
            .entries()
            .find(|entry| entry.depth() == depth)
            .map(|entry| entry.id())
    }

    /// Erases every state but `keep`.
    fn keep_only(
        inner: &mut Qwen35Inner,
        keep: crate::models::qwen3_5::dflash2_state::store::EntryId,
    ) {
        let others: Vec<_> = inner
            .dflash2_states
            .store
            .entries()
            .map(|entry| entry.id())
            .filter(|&id| id != keep)
            .collect();
        for id in others {
            crate::models::qwen3_5::dflash2_state::disk::erase(&mut inner.dflash2_states, id);
        }
        inner.sync_gdn_state_marks();
    }

    /// `id`'s RAM copy as the tier writes it.
    fn ram_bytes(
        inner: &Qwen35Inner,
        id: crate::models::qwen3_5::dflash2_state::store::EntryId,
    ) -> Option<Vec<u8>> {
        use crate::models::qwen3_5::dflash2_state::disk::encode_anchor;
        let anchor = inner.dflash2_states.store.get(id)?.ram()?;
        encode_anchor(&inner.dflash2_states, &inner.config, anchor)
    }

    /// Writes `id` to the tier as a reclaim does, lands its chain, and
    /// evicts every block no other state holds. Returns its bytes.
    fn to_disk_alone(
        inner: &mut Qwen35Inner,
        tier: &crate::models::qwen3_5::dflash2_state::disk::tests::TestTier,
        id: crate::models::qwen3_5::dflash2_state::store::EntryId,
    ) -> Vec<u8> {
        use crate::models::qwen3_5::dflash2_state::disk::tests::{disk_copy, drained};
        use crate::models::qwen3_5::dflash2_state::disk::{Reclaimed, Tick, reclaim, tick};
        let want = ram_bytes(inner, id).expect("a RAM copy");
        assert_eq!(reclaim(inner, id), Reclaimed::Written);
        for _ in 0..64 {
            drained(tier);
            tick(inner, Tick::Full { command_pages: 0 });
            if disk_copy(inner, id).is_some_and(|copy| copy.landed && !copy.chain_owed) {
                break;
            }
        }
        assert!(
            disk_copy(inner, id).is_some_and(|copy| copy.landed && !copy.chain_owed),
            "the state and its chain are on the tier"
        );
        let adapter = inner.paged_adapter.as_mut().expect("pool");
        let _ = adapter.release_request_for(0);
        let allocator = std::sync::Arc::clone(adapter.allocator());
        let mut guard = allocator.lock().expect("allocator");
        while guard.evict_oldest_dead_leaf().is_some() {}
        want
    }

    /// The tier's object for `id`'s disk copy, as a file.
    fn object_file(
        inner: &Qwen35Inner,
        tier: &crate::models::qwen3_5::dflash2_state::disk::tests::TestTier,
        id: crate::models::qwen3_5::dflash2_state::store::EntryId,
    ) -> std::path::PathBuf {
        let key = inner
            .dflash2_states
            .store
            .get(id)
            .and_then(|entry| entry.disk().copied())
            .expect("a disk copy")
            .key
            .to_hex();
        let root = tier.tier.manager.root().to_path_buf();
        std::fs::read_dir(&root)
            .expect("the tier's root")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(&key))
            })
            .expect("the state's object")
    }

    /// R (96) of a 100-token turn goes to the tier: its chain lands and
    /// every block of it leaves RAM. The same prompt's next turn resumes
    /// at R from the tier: the KV chain comes back from the walk, the lane
    /// gets R's exact cell and window (its promoted RAM copy is the lane's
    /// copy), and the greedy tokens equal paged AR's (03-PLAN gate). With
    /// another state's write in flight, the read waits for its landing and
    /// restores the same bytes.
    /// Mutations RT1 (the writer drops the window span: the payload check
    /// refuses, the turn goes cold) and SW1 (the read does not wait for
    /// the staging buffer) → red.
    #[test]
    fn disk_state_restores_byte_equal_and_greedy_equal() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "disk_state_restores_byte_equal_and_greedy_equal",
        ) {
            return Ok(());
        }
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::models::qwen3_5::dflash2_state::disk::tests::{insert_state, test_tier};
        use crate::models::qwen3_5::dflash2_state::disk::{Staged, stage};
        use crate::models::qwen3_5::model::dflash2_park_tests::chain_tokens;
        let _ctx = StreamContext::new(Stream::generation());
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let first = prompt(100);
        let seed = 0x5_8200;
        let tier = test_tier("state-tier-restore", 1 << 30);
        let reference = whole_turn(
            &mut tiny_paged_int8_dflash_inner(seed),
            &first,
            DecoderPlan::Autoregressive,
        )?;

        let mut inner = disk_fixture(seed, &tier, false)?;
        whole_turn(&mut inner, &first, dflash2)?;
        let r = state_at(&inner, 96).expect("the replay point at 96");
        keep_only(&mut inner, r);
        let want = to_disk_alone(&mut inner, &tier, r);
        let mark = inner.dflash2_states.store.get(r).expect("R").mark();
        assert!(
            !inner
                .paged_adapter
                .as_ref()
                .expect("pool")
                .prefix_cached(mark),
            "R's blocks left RAM"
        );
        let hits = tier.tier.manager.stats().hits;

        assert_eq!(whole_turn(&mut inner, &first, dflash2)?, reference);
        assert_eq!(
            tier.tier.manager.stats().hits - hits,
            6,
            "R's chain came back from the tier"
        );
        let counters = inner.dflash2_states.disk.as_ref().expect("tier").counters;
        assert_eq!((counters.promotions, counters.invalidations), (1, 0));
        assert!(ram_bytes(&inner, r) == Some(want.clone()), "R's bytes");

        // R on disk alone again, its blocks resident; another state's write
        // holds the staging buffer.
        assert_eq!(
            crate::models::qwen3_5::dflash2_state::disk::reclaim(&mut inner, r),
            crate::models::qwen3_5::dflash2_state::disk::Reclaimed::RamDropped
        );
        let block_size = inner.paged_adapter.as_ref().expect("pool").block_size();
        let other = insert_state(
            &mut inner,
            &chain_tokens(77, 1, block_size),
            crate::models::qwen3_5::dflash2_state::store::StateClass::Ordinary,
        );
        assert_eq!(
            stage(&mut inner, other, mlx_paged_attn::MakeRoom::Evict, false),
            Staged::Started
        );
        assert_eq!(whole_turn(&mut inner, &first, dflash2)?, reference);
        let counters = inner.dflash2_states.disk.as_ref().expect("tier").counters;
        assert_eq!((counters.promotions, counters.invalidations), (2, 0));
        assert!(
            ram_bytes(&inner, r) == Some(want),
            "R's bytes after the wait"
        );
        Ok(())
    }

    /// The restore of a disk state is promoted under one optional,
    /// immediate claim of its RAM copy and lineage, settled (Splash
    /// `promotedCopy`): granted, R holds RAM and disk copies; refused, R
    /// stays on disk and the next turn reads the tier again.
    /// Mutation PR1 (the promotion is claimed in service) → red.
    #[test]
    fn promotion_is_optional_and_immediate() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("promotion_is_optional_and_immediate") {
            return Ok(());
        }
        use crate::engine::governor::allocate::Refusal;
        use crate::engine::governor::ledger::{ClaimClass, NeedTiming};
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::models::qwen3_5::dflash2_state::disk::tests::test_tier;
        use crate::models::qwen3_5::dflash2_turn::ScriptedClaim;
        let _ctx = StreamContext::new(Stream::generation());
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let first = prompt(100);
        let promotions = |log: &[crate::models::qwen3_5::dflash2_turn::ClaimLog], bytes: u64| {
            log.iter()
                .filter(|claim| claim.bytes == bytes && claim.refusal == Refusal::Settled)
                .map(|claim| (claim.cls, claim.timing))
                .collect::<Vec<_>>()
        };
        let optional = (ClaimClass::Optional, NeedTiming::Immediate);
        for granted in [true, false] {
            let tier = test_tier("state-tier-promote", 1 << 30);
            let mut inner = disk_fixture(0x5_8210, &tier, true)?;
            whole_turn(&mut inner, &first, dflash2)?;
            let r = state_at(&inner, 96).expect("R");
            keep_only(&mut inner, r);
            to_disk_alone(&mut inner, &tier, r);
            let bytes = inner
                .dflash2_states
                .store
                .get(r)
                .expect("R")
                .resident_bytes();
            take_claim_log();
            let promotion = if granted {
                ScriptedClaim::Grant { freed: false }
            } else {
                ScriptedClaim::Refuse
            };
            script_claims(&[ScriptedClaim::Grant { freed: false }, promotion]);
            whole_turn(&mut inner, &first, dflash2)?;
            assert_eq!(promotions(&take_claim_log(), bytes), vec![optional]);
            let entry = inner.dflash2_states.store.get(r).expect("R stays");
            let counters = inner.dflash2_states.disk.as_ref().expect("tier").counters;
            assert!(entry.disk().is_some());
            if granted {
                assert!(entry.ram().is_some());
                assert_eq!((counters.promotions, counters.promotions_skipped), (1, 0));
                continue;
            }
            assert!(entry.ram().is_none());
            assert_eq!((counters.promotions, counters.promotions_skipped), (0, 1));
            whole_turn(&mut inner, &first, dflash2)?;
            let counters = inner.dflash2_states.disk.as_ref().expect("tier").counters;
            assert_eq!(counters.promotions, 1, "the next turn read the tier again");
            assert!(ram_bytes(&inner, r).is_some());
        }
        Ok(())
    }

    /// A disk state that does not read back (one byte of its object
    /// flipped) is discarded and the turn resumes at the next deepest state,
    /// the RAM checkpoint at 80, with paged AR's greedy tokens; so is one
    /// whose chain lost a KV object.
    /// Mutation BR1 (a failed read keeps the state) → red.
    #[test]
    fn a_bad_read_discards_the_state_and_resumes_lower() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_bad_read_discards_the_state_and_resumes_lower",
        ) {
            return Ok(());
        }
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::models::qwen3_5::dflash2_state::disk::tests::{kv_key, test_tier};
        let _ctx = StreamContext::new(Stream::generation());
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let first = prompt(100);
        let seed = 0x5_8220;
        let reference = whole_turn(
            &mut tiny_paged_int8_dflash_inner(seed),
            &first,
            DecoderPlan::Autoregressive,
        )?;
        for damage in ["object", "chain"] {
            let tier = test_tier("state-tier-bad-read", 1 << 30);
            let mut inner = disk_fixture(seed, &tier, false)?;
            PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(4)));
            let turn = whole_turn(&mut inner, &first, dflash2);
            PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
            turn?;
            let r = state_at(&inner, 96).expect("R");
            let checkpoint = state_at(&inner, 80).expect("the checkpoint at 80");
            to_disk_alone(&mut inner, &tier, r);
            if damage == "object" {
                let path = object_file(&inner, &tier, r);
                let mut bytes = std::fs::read(&path)?;
                let last = bytes.len() - 1;
                bytes[last] ^= 0x5a;
                std::fs::write(&path, bytes)?;
            } else {
                assert!(tier.tier.manager.remove(&kv_key(&inner, &tier, r, 5)));
            }
            let salt =
                crate::engine::backend::ChatBackend::resolve_params(&inner, &Default::default())
                    .cache_salt;
            let prefix = inner.prime_dspark_prefix_state(&first, true, salt)?;
            assert_eq!(
                prefix.effective_cached_prefix_len(),
                80,
                "{damage}: the turn resumes at the checkpoint"
            );
            inner.abort_paged_turn();
            let counters = inner.dflash2_states.disk.as_ref().expect("tier").counters;
            assert_eq!(counters.invalidations, 1, "{damage}");
            assert!(
                inner.dflash2_states.store.get(r).is_none(),
                "{damage}: R is gone"
            );
            assert!(inner.dflash2_states.store.get(checkpoint).is_some());
            assert_eq!(
                whole_turn(&mut inner, &first, dflash2)?,
                reference,
                "{damage}"
            );
        }
        Ok(())
    }

    /// An autoregressive turn after a DFlash2 turn whose replay point (96)
    /// is on the tier alone, its chain there and out of RAM: the KV lookup
    /// restores the chain below 96 from the tier and the GDN prime takes
    /// the cell from R's object, replaying nothing.
    /// Mutation AR2 (disk states are no GDN candidates) → red (the prime
    /// replays the prefix).
    #[test]
    fn ar_turn_reads_a_disk_state() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("ar_turn_reads_a_disk_state") {
            return Ok(());
        }
        use crate::engine::backend::ChatBackend;
        use crate::engine::governor::ledger::UnitKey;
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::models::qwen3_5::dflash2_state::disk::tests::test_tier;
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let first = prompt(100);
        let tier = test_tier("state-tier-ar-read", 1 << 30);
        let mut inner = disk_fixture(0x5_8230, &tier, false)?;
        let salt = ChatBackend::resolve_params(&inner, &Default::default()).cache_salt;
        whole_turn(
            &mut inner,
            &first,
            DecoderPlan::Speculative(SpeculativeKind::DraftModel),
        )?;
        let r = state_at(&inner, 96).expect("the replay point at 96");
        keep_only(&mut inner, r);
        to_disk_alone(&mut inner, &tier, r);
        let hits = tier.tier.manager.stats().hits;
        let mut second = first.clone();
        second.extend((0..20u32).map(|i| (i * 5 + 1) % VOCAB));
        let prefix = inner.prime_dense_prefix(&second, salt, None)?;
        assert_eq!(
            crate::engine::backend::PagedPrefix::effective_cached_prefix_len(&prefix),
            96
        );
        assert_eq!(
            tier.tier.manager.stats().hits - hits,
            6,
            "the chain came back"
        );
        assert_eq!(inner.last_gdn_prefix_prepare_state, "checkpoint");
        assert_eq!(inner.gdn_in_use_for_test(0), Some(&UnitKey::Store(r)));
        assert!(
            inner
                .dflash2_states
                .store
                .get(r)
                .is_some_and(|entry| entry.ram().is_none()),
            "an autoregressive read promotes nothing"
        );
        inner.abort_paged_turn();
        Ok(())
    }

    /// An autoregressive turn whose deepest disk state lost a block of its
    /// chain discards it at the lookup (Splash `pollRestores` →
    /// `discardState`), so no later turn arms it or walks the tier for it.
    /// Mutation DD1 (the disk arm takes a state its chain cannot restore)
    /// → red (R stays).
    #[test]
    fn ar_turn_discards_a_disk_state_whose_chain_is_gone() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "ar_turn_discards_a_disk_state_whose_chain_is_gone",
        ) {
            return Ok(());
        }
        use crate::engine::backend::ChatBackend;
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::models::qwen3_5::dflash2_state::disk::tests::{kv_key, test_tier};
        let _ctx = StreamContext::new(Stream::generation());
        let first = prompt(100);
        let tier = test_tier("state-tier-ar-gone", 1 << 30);
        let mut inner = disk_fixture(0x5_8231, &tier, false)?;
        let salt = ChatBackend::resolve_params(&inner, &Default::default()).cache_salt;
        whole_turn(
            &mut inner,
            &first,
            DecoderPlan::Speculative(SpeculativeKind::DraftModel),
        )?;
        let r = state_at(&inner, 96).expect("the replay point at 96");
        keep_only(&mut inner, r);
        to_disk_alone(&mut inner, &tier, r);
        assert!(tier.tier.manager.remove(&kv_key(&inner, &tier, r, 5)));
        let mut second = first.clone();
        second.extend((0..20u32).map(|i| (i * 5 + 1) % VOCAB));
        let _prefix = inner.prime_dense_prefix(&second, salt, None)?;
        inner.abort_paged_turn();
        assert!(inner.dflash2_states.store.get(r).is_none(), "R is gone");
        let counters = inner.dflash2_states.disk.as_ref().expect("tier").counters;
        assert_eq!(counters.invalidations, 1);
        Ok(())
    }

    /// The napi object's handle names the namespace directory the model's
    /// persistent state tier opened, read from the tier; after `closeCache`,
    /// and for a temporary tier, it names none.
    /// Mutation: the tier names its namespace after its close → red.
    #[test]
    fn the_tier_handle_names_the_open_namespace() -> Result<()> {
        use crate::models::qwen3_5::dflash2_state::disk::close_cache;
        use crate::models::qwen3_5::dflash2_state::disk::tests::{persistent_test_tier, test_tier};
        let _ctx = StreamContext::new(Stream::generation());
        let tier = persistent_test_tier("state-tier-namespace", 1 << 30);
        let mut inner = disk_fixture(0x5_8251, &tier, false)?;
        let handle = inner.dflash2_states.tier_handle();
        let dir = tier.tier.dir();
        assert_eq!(
            handle.namespace().as_deref(),
            dir.file_name().and_then(|name| name.to_str())
        );
        assert!(close_cache(
            &mut inner,
            std::time::Instant::now() + std::time::Duration::from_secs(2)
        ));
        assert_eq!(handle.namespace(), None);
        let temporary = test_tier("state-tier-namespace-temp", 1 << 30);
        let inner = disk_fixture(0x5_8252, &temporary, false)?;
        assert_eq!(inner.dflash2_states.tier_handle().namespace(), None);
        Ok(())
    }

    /// The model's `closeCache` command at a clean stop, its deadline in
    /// milliseconds on `Date.now()`'s clock: a persistent tier writes the
    /// waiting point with its chain before that deadline and reports it.
    /// Mutation CC1 (the deadline is read as now) → red.
    #[test]
    fn close_cache_command_flushes_by_its_deadline() -> Result<()> {
        use crate::engine::model_command::FamilyCommand;
        use crate::models::qwen3_5::dflash2_state::disk::tests::{
            disk_copy, park_blocks, persistent_test_tier,
        };
        use crate::models::qwen3_5::dflash2_state::store::StateClass;
        use crate::models::qwen3_5::dflash2_state::write_behind::{MINIMUM_TOKENS, published};
        use crate::models::qwen3_5::model::Qwen35FamilyCommand;
        let _ctx = StreamContext::new(Stream::generation());
        let tier = persistent_test_tier("state-tier-close-command", 1 << 30);
        let mut inner = disk_fixture(0x5_8250, &tier, false)?;
        let block_size = inner.paged_adapter.as_ref().expect("pool").block_size();
        let blocks = MINIMUM_TOKENS.div_ceil(block_size);
        let p = park_blocks(&mut inner, 140, blocks, StateClass::Ordinary);
        published(&mut inner.dflash2_states, p, blocks * block_size, 0.0);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            * 1000.0;
        let (reply, mut flushed) = tokio::sync::oneshot::channel();
        Qwen35FamilyCommand::CloseCache {
            deadline_ms: now_ms + 6000.0,
            reply,
        }
        .execute(&mut inner);
        assert!(flushed.try_recv().expect("a reply")?);
        assert!(disk_copy(&inner, p).is_some_and(|copy| copy.landed && !copy.chain_owed));
        assert!(!inner.dflash2_states.disk.as_ref().expect("tier").writable());
        Ok(())
    }

    /// A persistent tier's next load takes back what the last one closed
    /// (Splash `Cache::adopt`): R (96) and a checkpoint on another chain
    /// reach the disk at `closeCache`, the checkpoint by a reclaim and R by
    /// the write-behind flush. The next load has both on disk alone, stamps
    /// continuing past R's, and a turn on the same prompt resumes at R from
    /// the tier with paged AR's greedy tokens. With one KV object of the
    /// checkpoint's chain gone, only R comes back.
    /// Mutation AD1 (a state comes back without its whole chain) → red.
    #[test]
    fn restart_adopts_states_with_whole_chains() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("restart_adopts_states_with_whole_chains")
        {
            return Ok(());
        }
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::models::qwen3_5::dflash2_state::disk::tests::{
            disk_copy, kv_key, park_blocks, persistent_test_tier,
        };
        use crate::models::qwen3_5::dflash2_state::disk::{
            Reclaimed, adopt_all, close_cache, reclaim,
        };
        use crate::models::qwen3_5::dflash2_state::store::StateClass;
        let _ctx = StreamContext::new(Stream::generation());
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let first = prompt(100);
        let seed = 0x5_8240;
        let reference = whole_turn(
            &mut tiny_paged_int8_dflash_inner(seed),
            &first,
            DecoderPlan::Autoregressive,
        )?;
        for broken in [false, true] {
            let tier = persistent_test_tier("state-tier-adopt", 1 << 30);
            let mut inner = disk_fixture(seed, &tier, false)?;
            whole_turn(&mut inner, &first, dflash2)?;
            let r = state_at(&inner, 96).expect("the replay point at 96");
            keep_only(&mut inner, r);
            let c = park_blocks(&mut inner, 77, 2, StateClass::Checkpoint);
            assert_eq!(reclaim(&mut inner, c), Reclaimed::Written);
            assert!(close_cache(
                &mut inner,
                std::time::Instant::now() + std::time::Duration::from_secs(6)
            ));
            let r_copy = disk_copy(&inner, r).expect("R on disk");
            let c_copy = disk_copy(&inner, c).expect("the checkpoint on disk");
            assert!(r_copy.landed && !r_copy.chain_owed);
            assert!(c_copy.landed && !c_copy.chain_owed);
            let (_, r_label) = tier
                .tier
                .manager
                .state_label(r_copy.key, tier.tier.state_fingerprint)
                .expect("R's label");
            let lost = kv_key(&inner, &tier, c, 1);
            drop(inner);
            if broken {
                assert!(tier.tier.manager.remove(&lost));
            }

            let tier = tier.reopen("state-tier-adopt", 1 << 30);
            let mut inner = disk_fixture(seed, &tier, false)?;
            assert_eq!(adopt_all(&mut inner), if broken { 1 } else { 2 });
            let disk = inner.dflash2_states.disk.as_ref().expect("tier");
            assert_eq!(disk.counters.adopted, if broken { 1 } else { 2 });
            let entries: Vec<(usize, bool, bool)> = inner
                .dflash2_states
                .store
                .entries()
                .map(|entry| (entry.depth(), entry.checkpoint(), entry.ram().is_some()))
                .collect();
            let block = inner.paged_adapter.as_ref().expect("pool").block_size() as usize;
            assert_eq!(entries.len(), if broken { 1 } else { 2 });
            assert!(entries.contains(&(96, false, false)), "R on disk alone");
            assert_eq!(
                entries.contains(&(2 * block, true, false)),
                !broken,
                "the checkpoint only with its whole chain"
            );
            let adopted = state_at(&inner, 96).expect("R");
            let copy = disk_copy(&inner, adopted).expect("R's disk copy");
            assert_eq!(
                (copy.key, copy.window_begin, copy.landed, copy.chain_owed),
                (r_copy.key, r_copy.window_begin, true, false)
            );
            assert!(inner.dflash2_states.store.next_stamp() > r_label.last_used);

            let hits = tier.tier.manager.stats().hits;
            assert_eq!(whole_turn(&mut inner, &first, dflash2)?, reference);
            assert_eq!(
                tier.tier.manager.stats().hits - hits,
                6,
                "R's chain came back from the tier"
            );
            let counters = inner.dflash2_states.disk.as_ref().expect("tier").counters;
            assert_eq!((counters.promotions, counters.invalidations), (1, 0));
        }
        Ok(())
    }

    /// A (200 tokens) keeps 176 (checkpoint) and 192 (its replay point) and
    /// registers its whole chain. B shares A's first 150 tokens: it resumes
    /// at 0, its KV hit is 144 with a state below it (Splash `Cache::lookup`
    /// junction), and it publishes the junction there. C, sharing the same
    /// 150, resumes at 144. D, whose replay point (160) lies below its KV
    /// hit (176) with a state below the hit and none at it, finds the
    /// junction but its plan rejects it.
    /// Mutation JN1 (`state_below` sees nothing) → red (B finds no junction).
    #[test]
    fn junction_at_the_kv_hit() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("junction_at_the_kv_hit") {
            return Ok(());
        }
        use crate::engine::backend::ChatBackend;
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let a = prompt(200);
        let tail = |len: u32, seed: u32| -> Vec<u32> {
            (0..len).map(|i| (i * seed + 5) % VOCAB).collect()
        };
        let mut b = a[..150].to_vec();
        b.extend(tail(90, 11));
        let mut c = a[..150].to_vec();
        c.extend(tail(30, 13));
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8150);
        let salt = ChatBackend::resolve_params(&inner, &Default::default()).cache_salt;
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(4)));
        let result = (|| -> Result<()> {
            whole_turn(&mut inner, &a, dflash2)?;
            let depths: Vec<(usize, bool)> = inner
                .dflash2_states
                .store
                .entries()
                .map(|entry| (entry.depth(), entry.checkpoint()))
                .collect();
            assert_eq!(depths, vec![(176, true), (192, false)]);
            whole_turn(&mut inner, &b, dflash2)?;
            let counters = inner.dflash2_states.counters;
            assert_eq!(counters.lazy_junctions, 1);
            assert_eq!(counters.junction_materializations, 1);
            assert!(
                inner
                    .dflash2_states
                    .store
                    .entries()
                    .any(|entry| entry.depth() == 144 && !entry.checkpoint()),
                "B published the junction at 144"
            );
            let prefix = inner.prime_dspark_prefix_state(&c, true, salt)?;
            assert_eq!(prefix.effective_cached_prefix_len(), 144);
            inner.abort_paged_turn();

            let cp = inner
                .dflash2_states
                .store
                .entries()
                .find(|entry| entry.depth() == 176)
                .map(|entry| entry.id())
                .expect("A's checkpoint");
            crate::models::qwen3_5::dflash2_turn::drop_parked_state(&mut inner, cp);
            let d = a[..186].to_vec();
            inner.dflash2_states.generation_suffix = Some(std::sync::Arc::from(&a[166..186]));
            let prefix = inner.prime_dspark_prefix_state(&d, true, salt)?;
            assert_eq!(inner.dflash2_states.counters.lazy_junctions, 2);
            let points: Vec<u32> = inner
                .dflash2_states
                .turns
                .get(&0)
                .expect("D's turn")
                .plan
                .boundaries
                .iter()
                .map(|point| point.tokens)
                .collect();
            assert_eq!(points, vec![160], "176 lies past D's replay point");
            let cached = prefix.effective_cached_prefix_len();
            inner
                .paged_prefill_with_draft_state(&d[cached..], &prefix, stream)?
                .eval();
            assert_eq!(inner.dflash2_states.counters.junction_materializations, 1);
            inner.abort_paged_turn();
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// A DFlash2 turn ends by registering its blocks, releasing its lane and
    /// then ending its replay point's use (Splash `Engine::release`): no
    /// live request, ring or GDN state is left, the dirty latch is clear,
    /// and no history entry is formed. The next turn, which extends the
    /// first one's history, resumes at the first one's replay point (32),
    /// publishes its own (64), and emits paged AR's greedy tokens.
    /// Mutations LV1 (the DFlash2 finalize keeps the request live) and SV1
    /// (the history save checks the frontier of a released lane) → red.
    #[test]
    fn dflash2_turns_resume_at_the_deepest_state() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "dflash2_turns_resume_at_the_deepest_state",
        ) {
            return Ok(());
        }
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::models::qwen3_5::dflash2_turn::TURN_END_LOG;
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let first = prompt(40);
        let mut spec = tiny_paged_int8_dflash_inner(0x5_800A);
        let mut ar = tiny_paged_int8_dflash_inner(0x5_800A);
        let (_, invalidations) = spec.gdn_dirty_for_test();
        TURN_END_LOG.with(|log| log.borrow_mut().clear());
        let history = whole_turn(&mut spec, &first, dflash2)?;
        assert_eq!(
            TURN_END_LOG.with(|log| std::mem::take(&mut *log.borrow_mut())),
            vec!["register", "release", "end_use"]
        );
        assert_eq!(
            history,
            whole_turn(&mut ar, &first, DecoderPlan::Autoregressive)?
        );
        assert_eq!(history.len(), first.len() + 19);
        assert!(
            !spec
                .paged_adapter
                .as_ref()
                .expect("pool")
                .is_live_for_continue_for(0),
            "the lane is released"
        );
        assert!(spec.dflash2_context.is_none());
        assert!(spec.dflash2_turn_state.is_none());
        assert_eq!(spec.gdn_dirty_for_test(), (false, invalidations));
        assert!(spec.history_stamp_for_test().is_none(), "no history entry");
        assert_eq!(parked_depths(&spec), vec![32]);

        let mut second = history.clone();
        second.extend(prompt(9).iter().rev());
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(4)));
        PREFILL_CHUNK_BLOCKS.with(|log| *log.borrow_mut() = Some(Vec::new()));
        let spec_history = whole_turn(&mut spec, &second, dflash2);
        let seen = PREFILL_CHUNK_BLOCKS
            .with(|log| log.borrow_mut().take())
            .unwrap_or_default();
        let ar_history = whole_turn(&mut ar, &second, DecoderPlan::Autoregressive);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        assert_eq!(seen.len(), 9, "the prefill runs [32, 68) by four");
        assert_eq!(spec.dflash2_states.counters.replay_state_publications, 2);
        assert!(parked_depths(&spec).contains(&64));
        assert_eq!(spec_history?, ar_history?);
        Ok(())
    }

    /// The first index where `a` and `b` differ; their shorter length when
    /// one is the other's prefix.
    fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
        (0..a.len().max(b.len())).find(|&i| a.get(i) != b.get(i))
    }

    /// With INT8 queries from 64 tokens on (sharp attention), a DFlash2 turn
    /// that resumes at its replay point and prefills [32, 68) by four rows
    /// gives the tokens of AR on the same two turns, with bf16 and with INT8
    /// queries; the INT8 runs scored INT8 queries. Warm rows against a cold
    /// prefill are `every_path_scores_its_rows_in_one_mode`'s.
    /// Mutations v-all, rows16 and ar-bf16 → red.
    #[test]
    fn int8_query_turns_resume_like_a_cold_turn() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "int8_query_turns_resume_like_a_cold_turn",
        ) {
            return Ok(());
        }
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::transformer::Activations;
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let ar_plan = DecoderPlan::Autoregressive;
        let first = prompt(40);
        if int8_query_inner(0x5_800A, 64).is_none() {
            return Ok(());
        }
        for activations in [Activations::Bf16, Activations::Auto] {
            let build = || {
                let mut inner = int8_query_inner(0x5_800A, 64).expect("rule");
                inner
                    .paged_adapter
                    .as_mut()
                    .expect("pool")
                    .set_activations(activations);
                inner
            };
            let (mut spec, mut ar) = (build(), build());
            let history = whole_turn(&mut spec, &first, dflash2)?;
            assert_eq!(history, whole_turn(&mut ar, &first, ar_plan)?);
            let mut second = history.clone();
            second.extend(prompt(9).iter().rev());
            PREFILL_CHUNK_ROWS_OVERRIDE.with(|cell| cell.set(Some(4)));
            let spec_history = whole_turn(&mut spec, &second, dflash2);
            let ar_history = whole_turn(&mut ar, &second, ar_plan);
            PREFILL_CHUNK_ROWS_OVERRIDE.with(|cell| cell.set(None));
            if activations == Activations::Auto {
                assert!(int8_query_work(&spec)[0] > 0, "spec: no q8q launch");
                let [tile, decode] = int8_query_work(&ar);
                assert!(
                    tile > 0 && decode > 0,
                    "ar: q8q {tile}, coded rows {decode}"
                );
            }
            assert_eq!(
                spec_history?, ar_history?,
                "{activations:?}: resumed DFlash2 vs resumed AR"
            );
        }
        Ok(())
    }

    /// A DFlash2 lane's draft ring and context window go to the idle lane
    /// when the turn ends and the next turn takes them (Splash
    /// `QwenStateStorage::releaseLane`, `acquire`): two turns allocate each
    /// once.
    /// Mutation LB1 (a lane never takes the idle buffers) → red.
    #[test]
    fn lane_buffers_are_reused_between_turns() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("lane_buffers_are_reused_between_turns") {
            return Ok(());
        }
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::models::qwen3_5::dflash2::RING_ALLOCATIONS;
        use crate::models::qwen3_5::draft_context::WINDOW_ALLOCATIONS;
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8220);
        RING_ALLOCATIONS.with(|count| count.set(0));
        WINDOW_ALLOCATIONS.with(|count| count.set(0));
        let first = prompt(40);
        let mut second = whole_turn(&mut inner, &first, dflash2)?;
        assert!(inner.dflash2_states.idle_lane.ring.is_some());
        assert!(inner.dflash2_states.idle_lane.window.is_some());
        let rings = RING_ALLOCATIONS.with(std::cell::Cell::get);
        assert!(rings > 0, "one ring buffer per draft layer");
        second.extend(prompt(9).iter().rev());
        whole_turn(&mut inner, &second, dflash2)?;
        assert_eq!(RING_ALLOCATIONS.with(std::cell::Cell::get), rings);
        assert_eq!(WINDOW_ALLOCATIONS.with(std::cell::Cell::get), 1);
        Ok(())
    }

    /// An aborted DFlash2 turn's draft ring and window go to the idle lane
    /// (Splash `Engine::release` → `QwenStateStorage::releaseLane`), whether
    /// the turn state holds the ring (after the prefill) or the decode's
    /// stepper does (an error ends it without `finish`).
    /// Mutations AB1 (the abort drops the lane's buffers) and SD1 (a
    /// dropped stepper keeps its ring) → red.
    #[test]
    fn aborted_turns_leave_their_lane_buffers_idle() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "aborted_turns_leave_their_lane_buffers_idle",
        ) {
            return Ok(());
        }
        use crate::engine::backend::PagedPrefix;
        use crate::engine::dspark_turn::PagedDsparkBackend;
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prompt = prompt(40);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8230);
        let idle_ring = |inner: &Qwen35Inner| {
            inner
                .dflash2_states
                .idle_lane
                .ring
                .as_ref()
                .map_or(0, DFlash2ContextCache::buffer_bytes)
        };
        let ring_bytes = inner
            .dflash2
            .as_ref()
            .expect("draft")
            .config
            .kv_ring_bytes();
        tapped_prefill_by_four(&mut inner, &prompt)?;
        inner.dflash2_states.idle_lane = Default::default();
        inner.abort_paged_turn();
        assert_eq!(idle_ring(&inner), ring_bytes, "after the prefill");

        let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
        let cached = prefix.effective_cached_prefix_len();
        inner
            .paged_prefill_with_draft_state(&prompt[cached..], &prefix, stream)?
            .eval();
        let block = inner.dflash2.as_ref().expect("draft").config.block_size;
        let step = inner.begin_dspark_decode(block)?;
        drop(step);
        inner.dflash2_states.idle_lane = Default::default();
        inner.abort_paged_turn();
        assert_eq!(idle_ring(&inner), ring_bytes, "after a decode error");
        Ok(())
    }

    /// The idle lane's ring and window count in the model's states on the
    /// governor's books (Splash keeps one lane's start buffers).
    /// Mutation IA1 (the idle lane is not counted) → red.
    #[test]
    fn idle_lane_counts_in_the_states_account() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("idle_lane_counts_in_the_states_account")
        {
            return Ok(());
        }
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::models::qwen3_5::dflash2_turn::memory_accounts;
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8240);
        whole_turn(
            &mut inner,
            &prompt(40),
            DecoderPlan::Speculative(SpeculativeKind::DraftModel),
        )?;
        let draft = &inner.dflash2.as_ref().expect("draft").config;
        let lane = draft.kv_ring_bytes() + draft.context_window_bytes();
        let with = memory_accounts(&inner, 0).states;
        inner.dflash2_states.idle_lane = Default::default();
        assert_eq!(with - memory_accounts(&inner, 0).states, lane);
        Ok(())
    }

    /// One verify's draft-layer taps of 8 rows, as the lane's commit takes
    /// them.
    fn verify_taps(inner: &Qwen35Inner) -> Result<Vec<MxArray>> {
        let draft = &inner.dflash2.as_ref().expect("draft").config;
        let hidden = i64::from(inner.config.hidden_size);
        draft
            .target_layers
            .iter()
            .map(|_| {
                MxArray::random_normal(
                    &[1, DFLASH2_VERIFY_ROWS as i64, hidden],
                    0.0,
                    1.0,
                    Some(DType::BFloat16),
                )
            })
            .collect()
    }

    /// The 40-token prompts of scheduled sequences 1 and 2: no shared
    /// block, so each prefills from 0.
    fn lane_prompts() -> [(u32, Vec<u32>); 2] {
        [
            (1, prompt(40)),
            (2, (0..40u32).map(|i| (i * 5 + 1) % VOCAB).collect()),
        ]
    }

    /// Scheduled sequences 1 and 2 prefilled on [`lane_prompts`], each with
    /// a decode lane whose ring holds its prompt and 3 kept rows; sequence
    /// 1's lane holds its packed GDN cells.
    fn two_sequence_lanes(inner: &mut Qwen35Inner) -> Result<()> {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        for (seq, tokens) in &lane_prompts() {
            let base = scheduled_prime(inner, *seq, tokens)?;
            assert_eq!(base.effective_cached_prefix_len(), 0);
            scheduled_slice(inner, *seq, tokens, &base, 0, tokens.len())?;
        }
        inner.dflash2_states.idle_lane = Default::default();
        let config = inner.dflash2.as_ref().expect("draft").config.clone();
        let taps = verify_taps(inner)?;
        for (seq, tokens) in &lane_prompts() {
            let draft = inner.dflash2.as_ref().expect("draft");
            let lane = inner.dflash2_states.begin_lane(*seq, &config);
            lane.ring.record_only(tokens)?;
            lane.next_position = tokens.len() as i32;
            lane.append_kept(draft, &taps, &prompt(3))?;
            lane.ring.eval()?;
        }
        HybridSchedulerBackend::activate_paged_seq(inner, 1)?;
        let tiled = inner.config.qwen35_gguf_gdn_layout.as_deref() == Some("tiled");
        let gdn = GdnStateBlobs::pack(inner.caches.as_ref().expect("caches"), tiled)?;
        HybridSchedulerBackend::park_active_scheduled_recurrent(inner)?;
        inner.dflash2_states.lane_mut(1).expect("lane 1").gdn = Some(gdn.expect("packed cells"));
        Ok(())
    }

    /// A scheduled sequence's decode lane gives its ring to the idle lane
    /// when released, rows dropped, and the next lane takes it (Splash
    /// `QwenStateStorage::releaseLane`, `acquire`); its cells drop and its
    /// peer's lane stays.
    #[test]
    fn a_released_sequence_lane_goes_to_the_idle_lane() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_released_sequence_lane_goes_to_the_idle_lane",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8250);
        two_sequence_lanes(&mut inner)?;
        let config = inner.dflash2.as_ref().expect("draft").config.clone();
        let ring_bytes = config.kv_ring_bytes();
        assert_eq!(
            HybridSchedulerBackend::scheduled_speculation_owners(&inner),
            2
        );

        HybridSchedulerBackend::release_scheduled_speculation(&mut inner, 1);
        let states = &inner.dflash2_states;
        assert!(!states.seq_lanes.contains_key(&1));
        let peer = states.seq_lanes.get(&2).expect("the peer's lane stays");
        assert_eq!(peer.ring.logical_len(), 40 + 3);
        let idle = states.idle_lane.ring.as_ref().expect("the ring went idle");
        assert_eq!(idle.buffer_bytes(), ring_bytes);
        assert_eq!(idle.logical_len(), 0, "its rows dropped");
        assert_eq!(
            HybridSchedulerBackend::scheduled_speculation_owners(&inner),
            1
        );

        let lane = inner.dflash2_states.begin_lane(3, &config);
        assert_eq!(
            lane.ring.buffer_bytes(),
            ring_bytes,
            "the next lane takes it"
        );
        assert!(inner.dflash2_states.idle_lane.ring.is_none());
        Ok(())
    }

    /// A lane stamps each kept row's ring append with its own position and
    /// moves it past the rows, as the whole turn's stepper does: its ring
    /// equals one appended at the committed positions, bit for bit.
    #[test]
    fn a_sequence_lane_stamps_its_ring_at_its_position() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_sequence_lane_stamps_its_ring_at_its_position",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8255);
        let tokens = prompt(40);
        let config = inner.dflash2.as_ref().expect("draft").config.clone();
        let cycles = [
            (verify_taps(&inner)?, prompt(3)),
            (verify_taps(&inner)?, prompt(5)),
        ];
        let mut reference = DFlash2ContextCache::new(&config);
        reference.record_only(&tokens)?;
        let mut position = tokens.len() as i32;
        let draft = inner.dflash2.as_ref().expect("draft");
        for (taps, ids) in &cycles {
            position = append_kept_taps(draft, &mut reference, position, taps, ids)?;
        }
        let lane = inner.dflash2_states.begin_lane(7, &config);
        lane.ring.record_only(&tokens)?;
        lane.next_position = tokens.len() as i32;
        for (taps, ids) in &cycles {
            lane.append_kept(draft, taps, ids)?;
        }
        assert_eq!(lane.next_position, tokens.len() as i32 + 8);
        assert_eq!(lane.next_position, position);
        let (got, want) = (lane.ring.window_views(), reference.window_views());
        assert_eq!(got.len(), want.len());
        for (got, want) in got.iter().zip(&want) {
            let ((gk, gv), (wk, wv)) = (
                got.as_ref().expect("lane rows"),
                want.as_ref().expect("reference rows"),
            );
            assert_eq!(bits(gk)?, bits(wk)?);
            assert_eq!(bits(gv)?, bits(wv)?);
        }
        Ok(())
    }

    /// Each scheduled sequence's decode lane counts its ring and its second
    /// GDN cell in the model's states on the governor's books; the
    /// current cell is the sequence's scheduled cell, counted there.
    #[test]
    fn sequence_lanes_count_in_the_states_account() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "sequence_lanes_count_in_the_states_account",
        ) {
            return Ok(());
        }
        use crate::models::qwen3_5::dflash2_turn::memory_accounts;
        let _ctx = StreamContext::new(Stream::generation());
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8260);
        two_sequence_lanes(&mut inner)?;
        let ring = inner
            .dflash2
            .as_ref()
            .expect("draft")
            .config
            .kv_ring_bytes();
        let cell = inner.config.recurrent_state_bytes();
        assert!(ring > 0 && cell > 0);
        let with = memory_accounts(&inner, 0).states;
        let lanes = std::mem::take(&mut inner.dflash2_states.seq_lanes);
        assert_eq!(
            with - memory_accounts(&inner, 0).states,
            2 * ring + cell,
            "two rings and sequence 1's second cell"
        );
        inner.dflash2_states.seq_lanes = lanes;
        Ok(())
    }

    /// A one-chunk DFlash2 prefill with no capture leaves its ring and
    /// window lazy; the idle lane keeps them materialized, so a restore's
    /// claim credits only storage that exists (Splash
    /// `QwenStateStorage::releaseLane` pools allocated rings).
    #[test]
    fn the_idle_lane_holds_materialized_buffers() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "the_idle_lane_holds_materialized_buffers",
        ) {
            return Ok(());
        }
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prompt = prompt(40);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8245);
        let prefix = inner.prime_dspark_prefix_state(&prompt, false, 0)?;
        inner
            .paged_prefill_with_draft_state(&prompt, &prefix, stream)?
            .eval();
        inner.abort_paged_turn();
        let ready = |a: &&MxArray| unsafe { mlx_sys::mlx_array_is_available(a.as_raw_ptr()) };
        let idle = &inner.dflash2_states.idle_lane;
        let mut ring = Vec::new();
        idle.ring
            .as_ref()
            .expect("idle ring")
            .collect_arrays(&mut ring);
        assert!(!ring.is_empty() && ring.iter().all(ready), "lazy idle ring");
        let mut window = Vec::new();
        idle.window
            .as_ref()
            .expect("idle window")
            .collect_arrays(&mut window);
        assert!(window.iter().all(ready), "lazy idle window");
        Ok(())
    }

    /// A DFlash2 turn without cache reuse plans no point yet still has its
    /// turn state, so its end releases the lane: its prefill publishes
    /// nothing.
    /// Mutation RU1 (the plan ignores reuse) → red.
    #[test]
    fn a_turn_without_reuse_publishes_no_state() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("a_turn_without_reuse_publishes_no_state")
        {
            return Ok(());
        }
        use crate::engine::backend::PagedPrefix;
        use crate::engine::dspark_turn::PagedDsparkBackend;
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prompt = prompt(100);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8250);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(4)));
        let result = (|| -> Result<()> {
            let prefix = inner.prime_dspark_prefix_state(&prompt, false, 0)?;
            let cached = prefix.effective_cached_prefix_len();
            inner
                .paged_prefill_with_draft_state(&prompt[cached..], &prefix, stream)?
                .eval();
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result?;
        let turn = inner.dflash2_states.turns.get(&0).expect("the turn state");
        assert!(
            turn.plan.boundaries.is_empty(),
            "{:?}",
            turn.plan.boundaries
        );
        assert_eq!(inner.dflash2_states.store.bytes().entries, 0);
        inner.abort_paged_turn();
        Ok(())
    }

    /// A DFlash2 turn that ends on `length` runs no target forward after
    /// its last verify: the paged steppers drop the final token
    /// (`FinalTokenPolicy::AlwaysDrop`), so the saved history leaves it out.
    /// Mutation TF1 (`PagedStepper::materialize_final` runs a step for an
    /// `AlwaysDrop` model) → red.
    #[test]
    fn length_exit_runs_no_terminal_forward() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("length_exit_runs_no_terminal_forward") {
            return Ok(());
        }
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::models::qwen3_5::model::PAGED_STEPS;
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let first = prompt(40);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8230);
        PAGED_STEPS.with(|steps| steps.set(0));
        let history = whole_turn(
            &mut inner,
            &first,
            DecoderPlan::Speculative(SpeculativeKind::DraftModel),
        )?;
        assert_eq!(PAGED_STEPS.with(std::cell::Cell::get), 0);
        assert_eq!(history.len(), first.len() + 19);
        Ok(())
    }

    /// An autoregressive turn after a DFlash2 turn restores its GDN cell
    /// from the DFlash2 turn's stored replay point (96), the deepest state
    /// within its cached prefix, and holds it in use for the turn: the
    /// governor lists it only at `InUse` (Splash `StateCache::useState`).
    /// Mutation AR1 (store states are no GDN candidates) → red (the prime
    /// replays the prefix).
    #[test]
    fn ar_turn_restores_gdn_from_a_composite_state() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "ar_turn_restores_gdn_from_a_composite_state",
        ) {
            return Ok(());
        }
        use crate::engine::backend::ChatBackend;
        use crate::engine::governor::ledger::{
            CertainReuse, Need, Stage, TakeRights, UnitKey, VictimQuery,
        };
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        use crate::models::qwen3_5::dflash2_turn::{Keep, model_units};
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let first = prompt(100);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8180);
        let salt = ChatBackend::resolve_params(&inner, &Default::default()).cache_salt;
        whole_turn(
            &mut inner,
            &first,
            DecoderPlan::Speculative(SpeculativeKind::DraftModel),
        )?;
        assert!(inner.ladder_entries_for_test().is_empty(), "no rungs");
        let (id, mark) = inner
            .dflash2_states
            .store
            .entries()
            .find(|entry| entry.depth() == 96)
            .map(|entry| (entry.id(), entry.mark()))
            .expect("the replay point at 96");
        let mut second = first.clone();
        second.extend((0..20u32).map(|i| (i * 5 + 1) % VOCAB));
        let prefix = inner.prime_dense_prefix(&second, salt, None)?;
        assert_eq!(
            crate::engine::backend::PagedPrefix::effective_cached_prefix_len(&prefix),
            96
        );
        assert_eq!(inner.last_gdn_prefix_prepare_state, "checkpoint");
        assert_eq!(inner.gdn_in_use_for_test(0), Some(&UnitKey::Store(id)));
        assert!(inner.dflash2_states.store.in_use(mark));
        let query = VictimQuery {
            rights: TakeRights {
                units: true,
                in_use: true,
                certain_reuse: CertainReuse::All,
                ..TakeRights::default()
            },
            need: Need {
                bytes: 1,
                ..Need::default()
            },
        };
        let stages: Vec<Stage> = model_units(&inner, Keep::default(), &query)
            .into_units()
            .into_iter()
            .filter(|unit| unit.key == UnitKey::Store(id))
            .map(|unit| unit.stage)
            .collect();
        assert_eq!(stages, vec![Stage::InUse]);
        inner.abort_paged_turn();
        assert!(!inner.dflash2_states.store.in_use(mark));
        Ok(())
    }

    /// One lane's run: the rows its start resumed at, its prefill logits,
    /// each cycle's driver logits rows (bits), keep and ring after the
    /// commit, each draft's proposals, its tokens, and its GDN cells
    /// settled at the end.
    #[derive(PartialEq)]
    struct LaneRun {
        resumed: usize,
        prefill: Vec<u32>,
        cycles: Vec<(Vec<u32>, usize, RingState)>,
        proposals: Vec<Vec<i32>>,
        tokens: Vec<u32>,
        cells: Vec<Vec<u32>>,
    }

    impl LaneRun {
        fn empty() -> Self {
            Self {
                resumed: 0,
                prefill: Vec::new(),
                cycles: Vec::new(),
                proposals: Vec::new(),
                tokens: Vec::new(),
                cells: Vec::new(),
            }
        }
    }

    /// A lane's prompt, its output length, its temperature (above 0 the
    /// draft samples its proposals from the lane's rng), the tokens its
    /// forced proposals copy, and per cycle (round robin) how many of the
    /// driver's proposals are those tokens.
    struct LanePlan {
        seq: u32,
        prompt: Vec<u32>,
        n: usize,
        temperature: f64,
        reference: Vec<u32>,
        schedule: Vec<usize>,
    }

    fn greedy_params() -> crate::engine::params::ChatParams {
        crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
            temperature: Some(0.0),
            ..Default::default()
        })
    }

    fn lane_params(plan: &LanePlan) -> crate::engine::params::ChatParams {
        crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
            temperature: Some(plan.temperature),
            ..Default::default()
        })
    }

    /// The lane's own rng, seeded by its sequence: the same alone and
    /// scheduled.
    fn lane_rng(plan: &LanePlan) -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(7 + u64::from(plan.seq))
    }

    /// The driver ids of the cycle at `at`: the anchor, then `l_cap`
    /// proposals, the first `correct` of them the reference's tokens and the
    /// next one not, so the cycle keeps `min(correct, l_cap) + 1` rows when
    /// the reference is the target's.
    fn forced_ids(
        anchor: u32,
        reference: &[u32],
        at: usize,
        l_cap: usize,
        correct: usize,
    ) -> Vec<u32> {
        let mut ids = vec![anchor];
        for i in 0..l_cap {
            let token = reference[at + i];
            ids.push(if i < correct {
                token
            } else {
                (token + 1) % VOCAB
            });
        }
        ids
    }

    /// Every linear layer's conv and recurrent bits.
    pub(crate) fn cell_bits(
        caches: &[crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache],
    ) -> Result<Vec<Vec<u32>>> {
        let mut cells = Vec::new();
        for cache in caches {
            if let crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache::Linear(arrays) = cache {
                for slot in 0..2 {
                    cells.push(bits(arrays.get(slot).expect("a GDN slot"))?);
                }
            }
        }
        Ok(cells)
    }

    fn linear_layers(inner: &Qwen35Inner) -> u64 {
        inner
            .layer_kinds
            .iter()
            .filter(|kind| matches!(kind, super::super::decoder_layer::Qwen3_5LayerKind::Linear))
            .count() as u64
    }

    fn cycle_plan(run_cycles: usize, plan: &LanePlan, at: usize) -> (usize, usize) {
        (
            DFLASH2_DRAFT_PROPOSALS.min(plan.n - at - 1),
            plan.schedule[run_cycles % plan.schedule.len()],
        )
    }

    /// `plan`'s lane alone: a whole DFlash2 turn (prime, tapped prefill of
    /// the rows past its resume, stepper cycles, `finish`) to `plan.n`
    /// tokens.
    fn whole_turn_lane(inner: &mut Qwen35Inner, plan: &LanePlan) -> Result<LaneRun> {
        let prefix = inner.prime_dspark_prefix_state(&plan.prompt, true, 0)?;
        let resumed = prefix.effective_cached_prefix_len();
        let logits = inner.paged_prefill_with_draft_state(
            &plan.prompt[resumed..],
            &prefix,
            Stream::generation(),
        )?;
        let prefill = bits(&logits)?;
        let mut tokens = argmax_rows(&logits)?;
        let params = lane_params(plan);
        let mut rng = lane_rng(plan);
        let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
        let mut cycles = Vec::new();
        let mut proposals = Vec::new();
        while tokens.len() < plan.n {
            let anchor = *tokens.last().expect("a token");
            let at = tokens.len();
            let (l_cap, correct) = cycle_plan(cycles.len(), plan, at);
            if l_cap > 0 {
                let mut proposal = step.propose(anchor, l_cap, &params, &mut rng)?;
                proposal.materialize_draft_ids()?;
                proposals.push(proposal.draft_ids);
            }
            let ids = forced_ids(anchor, &plan.reference, at, l_cap, correct);
            let out = step.verify(&ids)?;
            let target = argmax_rows(&out.logits)?;
            let accepted = (0..l_cap).take_while(|&i| target[i] == ids[i + 1]).count();
            step.commit_with_provenance(accepted + 1, ids.len(), &ids)?;
            cycles.push((bits(&out.logits)?, accepted + 1, ring_state(&step.context)?));
            tokens.extend_from_slice(&ids[1..=accepted]);
            tokens.push(target[accepted]);
        }
        step.finish()?;
        Ok(LaneRun {
            resumed,
            prefill,
            cycles,
            proposals,
            tokens,
            cells: cell_bits(inner.caches.as_ref().expect("caches"))?,
        })
    }

    /// One verify wave: each row's driver ids, the wave's deferred GDN
    /// launches, its lane-kernel verify forwards, and its verify waves with
    /// their lanes.
    #[derive(Debug)]
    struct Wave {
        rows: Vec<usize>,
        launches: u64,
        forwards: u64,
        waves: (u64, u64),
    }

    /// The rows of a wave that keep nothing: none, one sequence's commit at
    /// keep 0, the whole wave retracted after its verify, or a verify that
    /// fails (its rows run again).
    #[derive(Clone, Copy)]
    enum Void {
        None,
        KeepZero(u32),
        Retract,
        Fail,
    }

    fn verify_waves() -> (u64, u64) {
        crate::models::qwen3_5::paged_forward::DFLASH2_VERIFY_WAVES.with(std::cell::Cell::get)
    }

    /// An autoregressive peer (its draft off): its sequence, prompt and
    /// length; each decode step's logits bits, its tokens, and its cells at
    /// the end.
    struct ArPeer {
        seq: u32,
        prompt: Vec<u32>,
        n: usize,
        steps: Vec<Vec<u32>>,
        tokens: Vec<u32>,
        cells: Vec<Vec<u32>>,
    }

    impl ArPeer {
        fn new(seq: u32, prompt: Vec<u32>, n: usize) -> Self {
            Self {
                seq,
                prompt,
                n,
                steps: Vec::new(),
                tokens: Vec::new(),
                cells: Vec::new(),
            }
        }
    }

    /// `peer` alone: its scheduled prefill in one slice, then one-row
    /// scheduled decodes (the forward its decode takes when no draft runs)
    /// to its length.
    fn ar_peer_alone(inner: &mut Qwen35Inner, mut peer: ArPeer) -> Result<ArPeer> {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let base = scheduled_prime(inner, peer.seq, &peer.prompt)?;
        let logits = scheduled_last_slice(
            inner,
            peer.seq,
            &peer.prompt,
            &base,
            base.effective_cached_prefix_len(),
        )?;
        peer.tokens = argmax_rows(&logits)?;
        while peer.tokens.len() < peer.n {
            assert!(HybridSchedulerBackend::reserve_scheduled_speculation(
                inner, peer.seq, 1
            )?);
            let token = *peer.tokens.last().expect("a token");
            let logits =
                HybridSchedulerBackend::run_paged_decode_step_batched(inner, &[(peer.seq, token)])?;
            peer.steps.push(bits(&logits)?);
            peer.tokens.extend(argmax_rows(&logits)?);
        }
        HybridSchedulerBackend::activate_paged_seq(inner, peer.seq)?;
        peer.cells = cell_bits(inner.caches.as_ref().expect("caches"))?;
        Ok(peer)
    }

    /// Asserts `peer` is `alone`, naming the first part that differs.
    fn assert_ar_peer(peer: &ArPeer, alone: &ArPeer) {
        assert_eq!(peer.steps.len(), alone.steps.len(), "peer: steps");
        for (step, (got, want)) in peer.steps.iter().zip(&alone.steps).enumerate() {
            assert!(got == want, "peer: step {step} logits bits");
        }
        assert_eq!(peer.tokens, alone.tokens, "peer: tokens");
        assert!(peer.cells == alone.cells, "peer: cells");
    }

    /// Scheduled lanes stepped through the scheduler's hooks as
    /// `execute_speculative_rows` calls them; a lane rides the waves once
    /// its prefill's last slice ran, until it has its `n` tokens. Its
    /// autoregressive `peers` ride the same waves, each row ahead of the
    /// lanes', once prefilled, to their lengths.
    struct LaneDriver<'a> {
        plans: &'a [LanePlan],
        runs: Vec<LaneRun>,
        rngs: Vec<rand::rngs::StdRng>,
        waves: Vec<Wave>,
        /// Lanes whose sequence the scheduler released: no wave, no settle.
        retired: Vec<bool>,
        peers: Vec<ArPeer>,
        /// When set, after every commit each committed lane's sequence and
        /// the bits of both its cells and its pending rows.
        states: Option<Vec<(u32, Vec<Vec<u32>>)>>,
        /// The last [`Void::Fail`] wave's verify error.
        failure: Option<String>,
    }

    impl<'a> LaneDriver<'a> {
        fn new(plans: &'a [LanePlan]) -> Self {
            Self {
                plans,
                runs: plans.iter().map(|_| LaneRun::empty()).collect(),
                rngs: plans.iter().map(lane_rng).collect(),
                waves: Vec::new(),
                retired: vec![false; plans.len()],
                peers: Vec::new(),
                states: None,
                failure: None,
            }
        }

        /// Peer `peer` primed (no lane) and prefilled in one slice.
        fn prefill_peer(&mut self, inner: &mut Qwen35Inner, peer: usize) -> Result<()> {
            let peer = &mut self.peers[peer];
            let base = scheduled_prime(inner, peer.seq, &peer.prompt)?;
            let logits = scheduled_last_slice(
                inner,
                peer.seq,
                &peer.prompt,
                &base,
                base.effective_cached_prefix_len(),
            )?;
            peer.tokens = argmax_rows(&logits)?;
            Ok(())
        }

        /// The peers, each with its cells.
        fn finish_peers(&mut self, inner: &mut Qwen35Inner) -> Result<Vec<ArPeer>> {
            use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
            let mut peers = std::mem::take(&mut self.peers);
            for peer in &mut peers {
                HybridSchedulerBackend::activate_paged_seq(inner, peer.seq)?;
                peer.cells = cell_bits(inner.caches.as_ref().expect("caches"))?;
            }
            Ok(peers)
        }

        /// Primes lane `lane`'s sequence and begins its DFlash2 lane at the
        /// rows it resumes at; returns the prime's prefix.
        fn prime(
            &mut self,
            inner: &mut Qwen35Inner,
            lane: usize,
        ) -> Result<<Qwen35Inner as PagedBackend>::PrefixState> {
            use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
            let plan = &self.plans[lane];
            let base = scheduled_prime(inner, plan.seq, &plan.prompt)?;
            let resumed = base.effective_cached_prefix_len();
            assert!(HybridSchedulerBackend::begin_scheduled_speculation(
                inner,
                plan.seq,
                u32::try_from(resumed).expect("resume rows")
            )?);
            self.runs[lane].resumed = resumed;
            Ok(base)
        }

        /// Lane `lane`'s prefill slice `[start, end)`; the prompt's last
        /// slice gives the lane its prefill logits and first token.
        fn slice(
            &mut self,
            inner: &mut Qwen35Inner,
            lane: usize,
            base: &<Qwen35Inner as PagedBackend>::PrefixState,
            start: usize,
            end: usize,
        ) -> Result<()> {
            let plan = &self.plans[lane];
            let logits = scheduled_slice(inner, plan.seq, &plan.prompt, base, start, end)?;
            if end == plan.prompt.len() {
                let logits = logits.ok_or_else(|| Error::from_reason("no prefill logits"))?;
                self.runs[lane].prefill = bits(&logits)?;
                self.runs[lane].tokens = argmax_rows(&logits)?;
            }
            Ok(())
        }

        /// Lane `lane` primed and prefilled in one slice.
        fn prefill(&mut self, inner: &mut Qwen35Inner, lane: usize) -> Result<()> {
            let base = self.prime(inner, lane)?;
            let start = self.runs[lane].resumed;
            let end = self.plans[lane].prompt.len();
            self.slice(inner, lane, &base, start, end)
        }

        /// One wave of every prefilled lane short of its length: each books
        /// its verify for its cap, proposes, and rides one
        /// `run_scheduled_verify`; one `commit_scheduled_verify` closes the
        /// wave. False when no lane is live.
        fn wave(&mut self, inner: &mut Qwen35Inner) -> Result<bool> {
            self.wave_with(inner, Void::None)
        }

        /// [`Self::wave`] with `void` rows that keep nothing: their cycle
        /// runs again in a later wave.
        fn wave_with(&mut self, inner: &mut Qwen35Inner, void: Void) -> Result<bool> {
            use crate::engine::hybrid_scheduler::{
                HybridSchedulerBackend, ScheduledVerifyCommit, ScheduledVerifyRow,
            };
            let voided = |seq: u32| match void {
                Void::None => false,
                Void::KeepZero(lane) => lane == seq,
                Void::Retract | Void::Fail => true,
            };
            let live: Vec<usize> = (0..self.plans.len())
                .filter(|&lane| {
                    let tokens = self.runs[lane].tokens.len();
                    !self.retired[lane] && tokens > 0 && tokens < self.plans[lane].n
                })
                .collect();
            let peers: Vec<usize> = (0..self.peers.len())
                .filter(|&peer| {
                    let tokens = self.peers[peer].tokens.len();
                    tokens > 0 && tokens < self.peers[peer].n
                })
                .collect();
            if live.is_empty() && peers.is_empty() {
                return Ok(false);
            }
            let mut rows = Vec::new();
            for &peer in &peers {
                let peer = &self.peers[peer];
                assert!(
                    HybridSchedulerBackend::reserve_scheduled_speculation(inner, peer.seq, 1)?,
                    "peer {} books its row",
                    peer.seq
                );
                rows.push(ScheduledVerifyRow {
                    seq_id: peer.seq,
                    first_position: cursor_of(inner, peer.seq),
                    tokens: vec![*peer.tokens.last().expect("a token")],
                    speculative: false,
                });
            }
            for &lane in &live {
                let plan = &self.plans[lane];
                let anchor = *self.runs[lane].tokens.last().expect("a token");
                let at = self.runs[lane].tokens.len();
                let (l_cap, correct) = cycle_plan(self.runs[lane].cycles.len(), plan, at);
                assert!(
                    HybridSchedulerBackend::reserve_scheduled_speculation(
                        inner,
                        plan.seq,
                        l_cap + 1
                    )?,
                    "lane {} books its verify",
                    plan.seq
                );
                if l_cap > 0 {
                    let mut proposal = HybridSchedulerBackend::propose_scheduled(
                        inner,
                        plan.seq,
                        anchor,
                        l_cap,
                        &lane_params(plan),
                        &mut self.rngs[lane],
                        false,
                    )?;
                    if proposal.device_draft_sparse.is_some() {
                        LANE_DEVICE_DRAWS.with(|draws| draws.set(draws.get() + 1));
                    }
                    proposal.materialize_draft_ids()?;
                    assert_eq!(proposal.draft_ids.len(), l_cap);
                    if !voided(plan.seq) {
                        self.runs[lane].proposals.push(proposal.draft_ids);
                    }
                }
                rows.push(ScheduledVerifyRow {
                    seq_id: plan.seq,
                    first_position: inner
                        .paged_adapter
                        .as_ref()
                        .expect("pool")
                        .current_token_count_for(plan.seq)
                        .expect("the lane's frontier"),
                    tokens: forced_ids(anchor, &plan.reference, at, l_cap, correct),
                    speculative: true,
                });
            }
            let (forwards, waves) = (verify_forwards(), verify_waves());
            unsafe { mlx_sys::mlx_test_kquant_counting(true) };
            let logits =
                HybridSchedulerBackend::run_scheduled_verify(inner, &rows).inspect(|logits| {
                    logits.eval();
                });
            let launches =
                unsafe { mlx_sys::mlx_test_kquant_family_count(c"gdn_fused_deferred".as_ptr()) };
            unsafe { mlx_sys::mlx_test_kquant_counting(false) };
            let after = verify_waves();
            self.waves.push(Wave {
                rows: rows.iter().map(|row| row.tokens.len()).collect(),
                launches,
                forwards: verify_forwards() - forwards,
                waves: (after.0 - waves.0, after.1 - waves.1),
            });
            if matches!(void, Void::Fail) {
                return match logits {
                    Err(error) => {
                        self.failure = Some(error.reason);
                        Ok(true)
                    }
                    Ok(_) => Err(Error::from_reason("the wave meant to fail verified")),
                };
            }
            let logits = logits?;
            if matches!(void, Void::Retract) {
                crate::models::qwen3_5::scheduled_dflash2::retract_open_wave(inner)?;
                return Ok(true);
            }
            let mut commits = Vec::new();
            let mut decided = Vec::new();
            let mut offset = 0i64;
            for row in &rows {
                let end = offset + row.tokens.len() as i64;
                let lane_logits = logits
                    .slice_axis(0, offset, end)?
                    .transpose(Some(&[1, 0, 2]))?;
                offset = end;
                let target = argmax_rows(&lane_logits)?;
                let accepted = (0..row.tokens.len() - 1)
                    .take_while(|&i| target[i] == row.tokens[i + 1])
                    .count();
                commits.push(ScheduledVerifyCommit {
                    seq_id: row.seq_id,
                    keep: if voided(row.seq_id) { 0 } else { accepted + 1 },
                });
                decided.push((bits(&lane_logits)?, accepted, target[accepted]));
            }
            for result in HybridSchedulerBackend::commit_scheduled_verify(inner, &commits)? {
                result?;
            }
            if let Some(states) = self.states.as_mut() {
                for row in rows.iter().filter(|row| row.speculative) {
                    let gdn = inner.dflash2_states.seq_lanes[&row.seq_id]
                        .gdn
                        .as_ref()
                        .expect("a committed lane's cells");
                    let cells = gdn
                        .lane_arrays()
                        .into_iter()
                        .map(bits)
                        .collect::<Result<Vec<_>>>()?;
                    states.push((row.seq_id, cells));
                }
            }
            for (index, (row, (logits, accepted, boundary))) in rows.iter().zip(decided).enumerate()
            {
                if voided(row.seq_id) {
                    continue;
                }
                if let Some(&peer) = peers.get(index) {
                    let peer = &mut self.peers[peer];
                    peer.steps.push(logits);
                    peer.tokens.push(boundary);
                    continue;
                }
                let lane = live[index - peers.len()];
                let ring = ring_state(&inner.dflash2_states.seq_lanes[&row.seq_id].ring)?;
                let run = &mut self.runs[lane];
                run.cycles.push((logits, accepted + 1, ring));
                run.tokens.extend_from_slice(&row.tokens[1..=accepted]);
                run.tokens.push(boundary);
            }
            Ok(true)
        }

        /// Each lane's cells settled; the runs and the waves.
        fn finish(mut self, inner: &mut Qwen35Inner) -> Result<(Vec<LaneRun>, Vec<Wave>)> {
            use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
            for ((plan, run), &retired) in self.plans.iter().zip(&mut self.runs).zip(&self.retired)
            {
                if retired {
                    continue;
                }
                HybridSchedulerBackend::activate_paged_seq(inner, plan.seq)?;
                let caches = inner.caches.as_mut().expect("caches");
                inner
                    .dflash2_states
                    .seq_lanes
                    .get_mut(&plan.seq)
                    .and_then(|lane| lane.gdn.as_mut())
                    .expect("the lane's cells")
                    .settle(caches)?;
                run.cells = cell_bits(caches)?;
            }
            Ok((self.runs, self.waves))
        }
    }

    /// The lanes of `plans` each primed and prefilled in one slice, then
    /// stepped together in waves to their lengths.
    fn scheduled_lanes(
        inner: &mut Qwen35Inner,
        plans: &[LanePlan],
    ) -> Result<(Vec<LaneRun>, Vec<Wave>)> {
        let mut driver = LaneDriver::new(plans);
        for lane in 0..plans.len() {
            driver.prefill(inner, lane)?;
        }
        while driver.wave(inner)? {}
        driver.finish(inner)
    }

    /// Asserts `run` is `alone`, naming the first part that differs.
    fn assert_lane_run(run: &LaneRun, alone: &LaneRun, tag: &str) {
        assert_eq!(run.resumed, alone.resumed, "{tag}: resumed rows");
        assert!(run.prefill == alone.prefill, "{tag}: prefill logits");
        assert_eq!(run.proposals.len(), alone.proposals.len(), "{tag}: drafts");
        for (cycle, (got, want)) in run.proposals.iter().zip(&alone.proposals).enumerate() {
            assert_eq!(got, want, "{tag}: draft {cycle} proposals");
        }
        assert_eq!(run.cycles.len(), alone.cycles.len(), "{tag}: cycles");
        for (cycle, (got, want)) in run.cycles.iter().zip(&alone.cycles).enumerate() {
            assert!(got.0 == want.0, "{tag}: cycle {cycle} logits bits");
            assert_eq!(got.1, want.1, "{tag}: cycle {cycle} keep");
            assert!(got.2 == want.2, "{tag}: cycle {cycle} ring");
        }
        assert_eq!(run.tokens, alone.tokens, "{tag}: tokens");
        assert!(run.cells == alone.cells, "{tag}: settled cells");
        assert!(run == alone, "{tag}");
    }

    fn keeps(run: &LaneRun) -> std::collections::BTreeSet<usize> {
        run.cycles.iter().map(|cycle| cycle.1).collect()
    }

    /// A scheduled DFlash2 lane's cycles through the scheduler's hooks are
    /// the whole turn's, every head order: each cycle's logits bits, keep
    /// and ring, the tokens and the settled cells, over keeps 1..=8. The
    /// driver's ids replace the draft's proposals (a thinking cut's shape),
    /// so the verify carries the driver's rows and the draft's tail.
    #[test]
    fn a_scheduled_dflash2_lane_matches_the_whole_turn_at_every_keep() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_scheduled_dflash2_lane_matches_the_whole_turn_at_every_keep",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let n = 48;
        for heads in [Heads::Same, Heads::Grouped, Heads::Tiled] {
            let tag = format!("{heads:?}");
            let prompt = prompt(40);
            let (reference, _) = paged_ar(
                &mut tiny_paged_int8_dflash_inner_heads(seed, heads),
                &prompt,
                n,
            )?;
            let plan = LanePlan {
                seq: 1,
                prompt,
                n,
                temperature: 0.0,
                reference,
                schedule: (0..=DFLASH2_DRAFT_PROPOSALS).collect(),
            };
            let alone =
                whole_turn_lane(&mut tiny_paged_int8_dflash_inner_heads(seed, heads), &plan)?;
            let mut inner = tiny_paged_int8_dflash_inner_heads(seed, heads);
            let linear = linear_layers(&inner);
            let (runs, waves) = scheduled_lanes(&mut inner, std::slice::from_ref(&plan))?;
            assert_lane_run(&runs[0], &alone, &tag);
            assert!(
                waves
                    .iter()
                    .all(|wave| wave.launches == wave.rows.len() as u64 * linear),
                "{tag}: one 8-row verify per lane per wave: {waves:?}"
            );
            if matches!(heads, Heads::Same) {
                assert_eq!(
                    keeps(&alone),
                    (1..=DFLASH2_VERIFY_ROWS).collect(),
                    "{tag}: every keep"
                );
            }
        }
        Ok(())
    }

    /// Two scheduled DFlash2 lanes in one wave per step, one 8-row verify
    /// each: every cycle's logits bits, keep and ring, the tokens and the
    /// settled cells of each lane are its whole turn's alone, over keeps
    /// 1..=8 for both.
    #[test]
    fn two_scheduled_dflash2_lanes_match_each_alone() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "two_scheduled_dflash2_lanes_match_each_alone",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let n = 48;
        let mut plans = Vec::new();
        for ((seq, prompt), schedule) in lane_prompts().into_iter().zip([
            (0..=DFLASH2_DRAFT_PROPOSALS).collect::<Vec<_>>(),
            (0..=DFLASH2_DRAFT_PROPOSALS).rev().collect(),
        ]) {
            let (reference, _) = paged_ar(&mut tiny_paged_int8_dflash_inner(seed), &prompt, n)?;
            plans.push(LanePlan {
                seq,
                prompt,
                n,
                temperature: 0.0,
                reference,
                schedule,
            });
        }
        let mut alone = Vec::new();
        for plan in &plans {
            alone.push(whole_turn_lane(
                &mut tiny_paged_int8_dflash_inner(seed),
                plan,
            )?);
        }
        let mut inner = tiny_paged_int8_dflash_inner(seed);
        let linear = linear_layers(&inner);
        let (runs, waves) = scheduled_lanes(&mut inner, &plans)?;
        for ((run, alone), plan) in runs.iter().zip(&alone).zip(&plans) {
            let tag = format!("lane {}", plan.seq);
            assert_lane_run(run, alone, &tag);
            assert_eq!(
                keeps(alone),
                (1..=DFLASH2_VERIFY_ROWS).collect(),
                "{tag}: every keep"
            );
        }
        assert!(waves.iter().any(|wave| wave.rows.len() == 2), "{waves:?}");
        assert!(
            waves
                .iter()
                .all(|wave| wave.launches == wave.rows.len() as u64 * linear),
            "one 8-row verify per lane per wave: {waves:?}"
        );
        Ok(())
    }

    /// The plans of two lanes on [`lane_prompts`] at `temperature`, each
    /// with its length and schedule and its paged AR reference.
    fn two_lane_plans(
        seed: u64,
        temperature: f64,
        lanes: [(usize, Vec<usize>); 2],
    ) -> Result<Vec<LanePlan>> {
        let mut plans = Vec::new();
        for ((seq, prompt), (n, schedule)) in lane_prompts().into_iter().zip(lanes) {
            let (reference, _) = paged_ar(&mut tiny_paged_int8_dflash_inner(seed), &prompt, n)?;
            plans.push(LanePlan {
                seq,
                prompt,
                n,
                temperature,
                reference,
                schedule,
            });
        }
        Ok(plans)
    }

    thread_local! {
        /// Lane proposals drawn on the device with their rows, this thread.
        static LANE_DEVICE_DRAWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// Each plan's lane alone on a fresh fixture, and the plans' lanes
    /// scheduled together on another.
    fn alone_and_scheduled(
        seed: u64,
        plans: &[LanePlan],
    ) -> Result<(Vec<LaneRun>, Vec<LaneRun>, Vec<Wave>)> {
        let mut alone = Vec::new();
        for plan in plans {
            alone.push(whole_turn_lane(
                &mut tiny_paged_int8_dflash_inner(seed),
                plan,
            )?);
        }
        let (runs, waves) = scheduled_lanes(&mut tiny_paged_int8_dflash_inner(seed), plans)?;
        Ok((alone, runs, waves))
    }

    /// A wave whose first lane verifies fewer than 8 driver rows (its cap
    /// below 7, down to 0 at its last token) ahead of a lane at cap 7: each
    /// lane gets its own logits rows, so both lanes' cycles are their whole
    /// turns' alone; the first lane runs at every width 1..=8 ahead of the
    /// full lane.
    #[test]
    fn a_short_dflash2_row_ahead_of_a_full_row_keeps_each_lanes_logits() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_short_dflash2_row_ahead_of_a_full_row_keeps_each_lanes_logits",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = two_lane_plans(seed, 0.0, [(10, vec![0]), (48, vec![3, 0, 7, 1])])?;
        let (alone, runs, waves) = alone_and_scheduled(seed, &plans)?;
        for ((run, alone), plan) in runs.iter().zip(&alone).zip(&plans) {
            assert_lane_run(run, alone, &format!("lane {}", plan.seq));
        }
        let widths: std::collections::BTreeSet<usize> = waves
            .iter()
            .filter(|wave| wave.rows.len() == 2 && wave.rows[1] == DFLASH2_VERIFY_ROWS)
            .map(|wave| wave.rows[0])
            .collect();
        assert_eq!(
            widths,
            (1..=DFLASH2_VERIFY_ROWS).collect(),
            "lane 1 at every width ahead of lane 2 at cap 7: {waves:?}"
        );
        Ok(())
    }

    /// Sampled lanes (temperature 1): each lane's draft samples its
    /// proposals from its own turn's rng, so two lanes scheduled together
    /// draft exactly what each drafts alone from the same seed, and every
    /// cycle's logits bits, keep, ring, tokens and cells follow.
    #[test]
    fn sampled_dflash2_lanes_draft_from_their_own_rng() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "sampled_dflash2_lanes_draft_from_their_own_rng",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = two_lane_plans(
            seed,
            1.0,
            [
                (32, (0..=DFLASH2_DRAFT_PROPOSALS).collect()),
                (32, (0..=DFLASH2_DRAFT_PROPOSALS).rev().collect()),
            ],
        )?;
        LANE_DEVICE_DRAWS.with(|draws| draws.set(0));
        let (alone, runs, waves) = alone_and_scheduled(seed, &plans)?;
        for ((run, alone), plan) in runs.iter().zip(&alone).zip(&plans) {
            let tag = format!("lane {}", plan.seq);
            assert!(
                alone.proposals.len() >= 3,
                "{tag}: the lane drafts: {}",
                alone.proposals.len()
            );
            assert_lane_run(run, alone, &tag);
        }
        assert!(waves.iter().any(|wave| wave.rows.len() == 2), "{waves:?}");
        let draws = LANE_DEVICE_DRAWS.with(std::cell::Cell::get);
        let drafted: usize = runs.iter().map(|run| run.proposals.len()).sum();
        if unsafe { mlx_sys::mlx_metal_is_available() } {
            assert_eq!(
                draws, drafted,
                "every sampled lane draft draws on the device"
            );
        }
        Ok(())
    }

    /// DFlash2 lanes in verify waves with INT8 queries from KV row 44 on
    /// (40-row prompts, so the first cycles straddle it): each lane is its
    /// whole turn alone, and the verifies scored INT8 queries.
    #[test]
    fn int8_query_dflash2_lanes_in_a_wave_equal_alone() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "int8_query_dflash2_lanes_in_a_wave_equal_alone",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let from = 44;
        if int8_query_inner(seed, from).is_none() {
            return Ok(());
        }
        let plans = two_lane_plans(seed, 0.0, [(32, vec![7, 3, 0, 7]), (32, vec![2, 7, 7, 1])])?;
        let mut alone = Vec::new();
        for plan in &plans {
            alone.push(whole_turn_lane(
                &mut int8_query_inner(seed, from).expect("INT8 queries"),
                plan,
            )?);
        }
        let mut inner = int8_query_inner(seed, from).expect("INT8 queries");
        let (runs, waves) = scheduled_lanes(&mut inner, &plans)?;
        assert!(
            int8_query_work(&inner)[0] > 0,
            "no verify scored INT8 queries"
        );
        for ((run, alone), plan) in runs.iter().zip(&alone).zip(&plans) {
            assert_lane_run(run, alone, &format!("lane {}", plan.seq));
        }
        assert!(waves.iter().any(|wave| wave.rows.len() == 2), "{waves:?}");
        Ok(())
    }

    /// The app's turn shape at the hook level: lane 1 decodes while lane
    /// 2 arrives, and each of lane 2's prefill slices runs between two of
    /// lane 1's decode waves; then both decode in shared waves. Lane 2's
    /// prompt shares no block with lane 1's, or extends it (it resumes at
    /// the state lane 1's prefill published at 32). The draft window (40)
    /// reaches back past lane 2's last slice. Each lane's resume, prefill
    /// logits, proposals, every cycle's logits bits, keep and ring, tokens
    /// and settled cells are its whole turn's alone (lane 2's after a whole
    /// turn of lane 1's prompt published the same states).
    #[test]
    fn a_peer_prefill_between_dflash2_waves_leaves_each_lane_exact() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_peer_prefill_between_dflash2_waves_leaves_each_lane_exact",
        ) {
            return Ok(());
        }
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let fixture = || wide_draft_inner(seed, 40);
        let disjoint: Vec<u32> = (0..80u32).map(|i| (i * 5 + 1) % VOCAB).collect();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            for (case, second, resumes, ends) in [
                ("disjoint", disjoint, 0, vec![32, 64, 80]),
                ("shared", prompt(64), 32, vec![48, 64]),
            ] {
                let mut plans = Vec::new();
                for (seq, prompt, n, schedule) in [
                    (
                        1,
                        prompt(40),
                        48,
                        (0..=DFLASH2_DRAFT_PROPOSALS).rev().collect(),
                    ),
                    (
                        2,
                        second,
                        24,
                        (0..=DFLASH2_DRAFT_PROPOSALS).collect::<Vec<_>>(),
                    ),
                ] {
                    let (reference, _) = paged_ar(&mut fixture(), &prompt, n)?;
                    plans.push(LanePlan {
                        seq,
                        prompt,
                        n,
                        temperature: 0.0,
                        reference,
                        schedule,
                    });
                }
                let first_alone = whole_turn_lane(&mut fixture(), &plans[0])?;
                let mut whole = fixture();
                whole_turn(
                    &mut whole,
                    &plans[0].prompt,
                    DecoderPlan::Speculative(SpeculativeKind::DraftModel),
                )?;
                let second_alone = whole_turn_lane(&mut whole, &plans[1])?;
                assert_eq!(second_alone.resumed, resumes, "{case}: alone");

                let mut inner = fixture();
                let mut driver = LaneDriver::new(&plans);
                driver.prefill(&mut inner, 0)?;
                let base = driver.prime(&mut inner, 1)?;
                let mut start = driver.runs[1].resumed;
                for &end in &ends {
                    assert!(driver.wave(&mut inner)?, "{case}: lane 1 decodes");
                    driver.slice(&mut inner, 1, &base, start, end)?;
                    start = end;
                }
                while driver.wave(&mut inner)? {}
                let (runs, waves) = driver.finish(&mut inner)?;
                assert_lane_run(&runs[0], &first_alone, &format!("{case}: lane 1"));
                assert_lane_run(&runs[1], &second_alone, &format!("{case}: lane 2"));
                let (before, after) = waves.split_at(ends.len());
                assert!(
                    before.iter().all(|wave| wave.rows.len() == 1),
                    "{case}: one wave of lane 1 before each slice of lane 2: {waves:?}"
                );
                assert!(
                    after.iter().any(|wave| wave.rows.len() == 2),
                    "{case}: shared waves: {waves:?}"
                );
            }
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// A lane at its last output token (remaining 1: the engine books one
    /// row and drafts nothing) books its 8 rows and verifies the anchor with
    /// the draft's greedy tail, as the whole turn does.
    #[test]
    fn a_dflash2_lane_at_its_last_output_token_finishes() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_dflash2_lane_at_its_last_output_token_finishes",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8320;
        let n = 2;
        let prompt = prompt(44);
        let (reference, _) = paged_ar(&mut tiny_paged_int8_dflash_inner(seed), &prompt, n)?;
        let plan = LanePlan {
            seq: 1,
            prompt,
            n,
            temperature: 0.0,
            reference,
            schedule: vec![0],
        };
        let alone = whole_turn_lane(&mut tiny_paged_int8_dflash_inner(seed), &plan)?;
        let (runs, _) = scheduled_lanes(
            &mut tiny_paged_int8_dflash_inner(seed),
            std::slice::from_ref(&plan),
        )?;
        assert_eq!(alone.cycles.len(), 1);
        assert_lane_run(&runs[0], &alone, "last token");
        Ok(())
    }

    /// A lane near its window (prompt 61 of a 64-token window) runs to
    /// length as the whole turn does: its verifies end up to the window's 7
    /// scratch rows past it (61 + 8 … 63 + 8 = 71); a lane at the window is
    /// refused, not given a short verify.
    #[test]
    fn a_dflash2_lane_at_the_window_end_finishes_at_length() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_dflash2_lane_at_the_window_end_finishes_at_length",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8330;
        let n = 4;
        let windowed = || {
            let mut inner = tiny_paged_int8_dflash_inner(seed);
            inner.max_context = Some(64);
            assert_eq!(inner.pool_context_limit(), 64);
            inner
        };
        let prompt = prompt(61);
        let (reference, _) = paged_ar(&mut windowed(), &prompt, n)?;
        let plan = LanePlan {
            seq: 1,
            prompt,
            n,
            temperature: 0.0,
            reference,
            schedule: vec![0],
        };
        let alone = whole_turn_lane(&mut windowed(), &plan)?;
        let mut inner = windowed();
        let (runs, _) = scheduled_lanes(&mut inner, std::slice::from_ref(&plan))?;
        assert_eq!(alone.tokens.len(), n);
        assert_lane_run(&runs[0], &alone, "window end");
        let adapter = inner.paged_adapter.as_ref().expect("pool");
        assert_eq!(adapter.current_token_count_for(1), Some(64));
        let booked = blocks(&inner);
        assert!(!HybridSchedulerBackend::reserve_scheduled_speculation(
            &mut inner, 1, 1
        )?);
        assert_eq!(blocks(&inner), booked, "a refusal books nothing");
        Ok(())
    }

    /// Releasing a lane whose verify is open retracts the wave: its rows
    /// roll back to the committed frontier, its lane goes idle, and the
    /// last lane drops the decode cache limit.
    #[test]
    fn a_released_dflash2_lane_retracts_its_open_verify() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_released_dflash2_lane_retracts_its_open_verify",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::{HybridSchedulerBackend, ScheduledVerifyRow};
        let _ctx = StreamContext::new(Stream::generation());
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8340);
        let plan = LanePlan {
            seq: 1,
            prompt: prompt(40),
            n: 2,
            temperature: 0.0,
            reference: Vec::new(),
            schedule: vec![0],
        };
        let mut driver = LaneDriver::new(std::slice::from_ref(&plan));
        driver.prefill(&mut inner, 0)?;
        assert_eq!(driver.runs[0].resumed, 0);
        let anchor = driver.runs[0].tokens[0];
        let committed = cursor_of(&inner, 1);
        assert!(HybridSchedulerBackend::reserve_scheduled_speculation(
            &mut inner,
            1,
            DFLASH2_VERIFY_ROWS
        )?);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut proposal = HybridSchedulerBackend::propose_scheduled(
            &mut inner,
            1,
            anchor,
            DFLASH2_DRAFT_PROPOSALS,
            &greedy_params(),
            &mut rng,
            false,
        )?;
        proposal.materialize_draft_ids()?;
        let mut tokens = vec![anchor];
        tokens.extend(proposal.draft_ids.iter().map(|&id| id as u32));
        HybridSchedulerBackend::run_scheduled_verify(
            &mut inner,
            &[ScheduledVerifyRow {
                seq_id: 1,
                first_position: committed,
                tokens,
                speculative: true,
            }],
        )?
        .eval();
        assert_eq!(cursor_of(&inner, 1), committed + DFLASH2_VERIFY_ROWS as u32);
        assert!(inner.dflash2_states.lane_decode_limit.is_some());

        HybridSchedulerBackend::release_scheduled_speculation(&mut inner, 1);
        assert_eq!(cursor_of(&inner, 1), committed, "the open rows rolled back");
        assert!(inner.dflash2_states.lane_batch.is_none());
        assert!(inner.dflash2_states.seq_lanes.is_empty());
        assert!(inner.dflash2_states.idle_lane.ring.is_some());
        assert!(inner.dflash2_states.lane_decode_limit.is_none());
        Ok(())
    }

    fn cursor_of(inner: &Qwen35Inner, seq: u32) -> u32 {
        inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .current_token_count_for(seq)
            .expect("the sequence's frontier")
    }

    /// A zero-keep commit (one lane of a wave) and a retracted wave (every
    /// lane) leave each lane as it was before the cycle: its frontier, its
    /// ring and its cells with their pending rows, so the cycle run again
    /// gives the whole turn's logits bits, keeps, rings, tokens and settled
    /// cells alone.
    #[test]
    fn a_zero_keep_cycle_leaves_the_lane_as_before() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_zero_keep_cycle_leaves_the_lane_as_before",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let n = 48;
        let mut plans = Vec::new();
        for ((seq, prompt), schedule) in lane_prompts().into_iter().zip([
            (0..=DFLASH2_DRAFT_PROPOSALS).collect::<Vec<_>>(),
            (0..=DFLASH2_DRAFT_PROPOSALS).rev().collect(),
        ]) {
            let (reference, _) = paged_ar(&mut tiny_paged_int8_dflash_inner(seed), &prompt, n)?;
            plans.push(LanePlan {
                seq,
                prompt,
                n,
                temperature: 0.0,
                reference,
                schedule,
            });
        }
        let alone = plans
            .iter()
            .map(|plan| whole_turn_lane(&mut tiny_paged_int8_dflash_inner(seed), plan))
            .collect::<Result<Vec<_>>>()?;
        let mut inner = tiny_paged_int8_dflash_inner(seed);
        let mut driver = LaneDriver::new(&plans);
        driver.prefill(&mut inner, 0)?;
        driver.prefill(&mut inner, 1)?;
        let mut voids = 0;
        for wave in 0.. {
            let void = match wave {
                2 | 5 => Void::KeepZero(1),
                3 => Void::Retract,
                _ => Void::None,
            };
            let frontiers = [cursor_of(&inner, 1), cursor_of(&inner, 2)];
            if !driver.wave_with(&mut inner, void)? {
                break;
            }
            match void {
                Void::None => {}
                Void::KeepZero(_) => {
                    assert_eq!(cursor_of(&inner, 1), frontiers[0], "keep 0 rolls back");
                    voids += 1;
                }
                Void::Retract | Void::Fail => {
                    assert_eq!(
                        [cursor_of(&inner, 1), cursor_of(&inner, 2)],
                        frontiers,
                        "the retracted wave rolls back"
                    );
                    voids += 1;
                }
            }
        }
        assert_eq!(voids, 3, "every void wave ran");
        let (runs, _) = driver.finish(&mut inner)?;
        for (lane, (run, alone)) in runs.iter().zip(&alone).enumerate() {
            assert_lane_run(run, alone, &format!("lane {}", lane + 1));
        }
        Ok(())
    }

    /// A sequence whose recurrent cells are released (an idle owner's
    /// release, a preemption, a failed turn) loses its DFlash2 lane with
    /// them: no lane outlives its cells, its ring goes idle, and the peer's
    /// lane runs on to its whole turn alone.
    #[test]
    fn a_dflash2_lane_ends_with_its_recurrent_cells() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_dflash2_lane_ends_with_its_recurrent_cells",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let n = 40;
        let mut plans = Vec::new();
        for (seq, prompt) in lane_prompts() {
            let (reference, _) = paged_ar(&mut tiny_paged_int8_dflash_inner(seed), &prompt, n)?;
            plans.push(LanePlan {
                seq,
                prompt,
                n,
                temperature: 0.0,
                reference,
                schedule: (0..=DFLASH2_DRAFT_PROPOSALS).collect(),
            });
        }
        let second_alone = whole_turn_lane(&mut tiny_paged_int8_dflash_inner(seed), &plans[1])?;
        let mut inner = tiny_paged_int8_dflash_inner(seed);
        let mut driver = LaneDriver::new(&plans);
        driver.prefill(&mut inner, 0)?;
        driver.prefill(&mut inner, 1)?;
        for _ in 0..3 {
            assert!(driver.wave(&mut inner)?);
        }
        assert!(ring_state(&inner.dflash2_states.seq_lanes[&1].ring)?.0 > 0);
        assert!(inner.dflash2_states.idle_lane.ring.is_none());
        HybridSchedulerBackend::release_scheduled_recurrent_for(&mut inner, 1);
        assert!(!HybridSchedulerBackend::has_scheduled_recurrent(&inner, 1));
        assert!(
            !inner.dflash2_states.seq_lanes.contains_key(&1),
            "the lane went with its cells"
        );
        let idle = inner.dflash2_states.idle_lane.ring.as_ref();
        assert!(
            idle.is_some_and(|ring| ring.logical_len() == 0),
            "the ring went idle, its rows dropped"
        );
        HybridSchedulerBackend::release_scheduled_speculation(&mut inner, 1);
        HybridSchedulerBackend::release_scheduled_cache(&mut inner, 1)?;
        driver.retired[0] = true;
        while driver.wave(&mut inner)? {}
        let (runs, _) = driver.finish(&mut inner)?;
        assert_lane_run(&runs[1], &second_alone, "lane 2");
        Ok(())
    }

    /// An autoregressive peer (its draft off) rides the DFlash2 lane's waves
    /// as one `[1, 1]` row with the forward its decode takes alone: it
    /// prefills between two of the lane's waves, decodes with its row ahead
    /// of the lane's 8, keeps 0 once, rides a retracted wave, and outlives
    /// the lane in waves of its own. Its every step's logits bits, its
    /// tokens and its cells are its alone run's; the lane's cycles are its
    /// whole turn's.
    #[test]
    fn an_ar_peer_rides_a_dflash2_wave() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("an_ar_peer_rides_a_dflash2_wave") {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let n = 48;
        let peer_n = 32;
        let [(seq, prompt), (peer_seq, peer_prompt)] = lane_prompts();
        let (reference, _) = paged_ar(&mut tiny_paged_int8_dflash_inner(seed), &prompt, n)?;
        let plans = [LanePlan {
            seq,
            prompt,
            n,
            temperature: 0.0,
            reference,
            schedule: (0..=DFLASH2_DRAFT_PROPOSALS).collect(),
        }];
        let alone = whole_turn_lane(&mut tiny_paged_int8_dflash_inner(seed), &plans[0])?;
        let peer_alone = ar_peer_alone(
            &mut tiny_paged_int8_dflash_inner(seed),
            ArPeer::new(peer_seq, peer_prompt.clone(), peer_n),
        )?;
        let mut inner = tiny_paged_int8_dflash_inner(seed);
        let mut driver = LaneDriver::new(&plans);
        driver.prefill(&mut inner, 0)?;
        driver
            .peers
            .push(ArPeer::new(peer_seq, peer_prompt, peer_n));
        let (mut beside, mut own, mut voids) = (0, 0, 0);
        for wave in 0.. {
            if wave == 2 {
                driver.prefill_peer(&mut inner, 0)?;
            }
            let void = match wave {
                4 => Void::KeepZero(peer_seq),
                6 => Void::Retract,
                _ => Void::None,
            };
            let lane_live = driver.runs[0].tokens.len() < n;
            let peer_live = (2..).contains(&wave) && driver.peers[0].tokens.len() < peer_n;
            let frontier = peer_live.then(|| cursor_of(&inner, peer_seq));
            if !driver.wave_with(&mut inner, void)? {
                break;
            }
            if !matches!(void, Void::None) {
                assert_eq!(
                    Some(cursor_of(&inner, peer_seq)),
                    frontier,
                    "the peer's void row rolls back"
                );
                voids += 1;
            }
            match (lane_live, peer_live) {
                (true, true) => beside += 1,
                (false, true) => own += 1,
                _ => {}
            }
        }
        assert_eq!(voids, 2, "every void wave ran");
        assert!(beside > 2, "the peer decoded beside the lane");
        assert!(own > 0, "the peer outlived the lane");
        assert!(inner.dflash2_states.peer_cells.is_empty());
        let peer = driver.finish_peers(&mut inner)?.remove(0);
        let (runs, _) = driver.finish(&mut inner)?;
        assert_lane_run(&runs[0], &alone, "lane");
        assert_ar_peer(&peer, &peer_alone);
        Ok(())
    }

    /// The tiny fixture with its verify waves on or off. Its projections
    /// are plain bf16, so a wave runs each projection once per lane: the
    /// lane tier's bits are the wave tests' in `paged_forward`.
    fn wave_inner(seed: u64, waves: bool) -> Qwen35Inner {
        let mut inner = tiny_paged_int8_dflash_inner(seed);
        inner.grant_scheduled_lanes(4);
        inner.dflash2_wave_ready = waves;
        inner
    }

    /// Asserts two state records (sequence, cell and pending bits after each
    /// commit) are equal, naming the first commit that differs.
    fn assert_states(got: &[(u32, Vec<Vec<u32>>)], want: &[(u32, Vec<Vec<u32>>)], tag: &str) {
        assert_eq!(got.len(), want.len(), "{tag}: commits");
        for (index, (got, want)) in got.iter().zip(want).enumerate() {
            assert_eq!(got.0, want.0, "{tag}: commit {index} sequence");
            assert!(
                got.1 == want.1,
                "{tag}: commit {index} sequence {}: cells and pending rows",
                got.0
            );
        }
    }

    /// Runs `drive` on a driver of `plans` over [`wave_inner`] with waves on
    /// and off: the runs, waves, per-commit states and each lane's verify
    /// counts of each.
    #[allow(clippy::type_complexity)]
    fn waved_and_per_owner(
        seed: u64,
        plans: &[LanePlan],
        drive: impl Fn(&mut LaneDriver<'_>, &mut Qwen35Inner) -> Result<()>,
    ) -> Result<
        [(
            Vec<LaneRun>,
            Vec<Wave>,
            Vec<(u32, Vec<Vec<u32>>)>,
            Vec<LaneVerifies>,
        ); 2],
    > {
        let mut out = Vec::new();
        for waves in [true, false] {
            let mut inner = wave_inner(seed, waves);
            let mut driver = LaneDriver::new(plans);
            driver.states = Some(Vec::new());
            drive(&mut driver, &mut inner)?;
            let states = driver.states.take().unwrap_or_default();
            let (runs, waves) = driver.finish(&mut inner)?;
            let verifies = plans
                .iter()
                .map(|plan| inner.dflash2_states.seq_lanes[&plan.seq].verifies)
                .collect();
            out.push((runs, waves, states, verifies));
        }
        let per_owner = out.pop().expect("the per-owner run");
        let waved = out.pop().expect("the waved run");
        Ok([waved, per_owner])
    }

    /// Two lanes that reach verify in the same step run ONE verify wave
    /// (no lane-kernel forward), and each lane's every cycle is its own:
    /// logits bits, keep (the accepted count), ring, its cells and pending
    /// rows after every commit, its tokens and settled cells equal its whole
    /// turn alone, over keeps 1..=8 for both; the per-commit states equal
    /// the per-owner lanes' from the same state.
    #[test]
    fn two_lanes_verify_in_one_wave_and_match_each_alone() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "two_lanes_verify_in_one_wave_and_match_each_alone",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let n = 48;
        let plans = two_lane_plans(
            seed,
            0.0,
            [
                (n, (0..=DFLASH2_DRAFT_PROPOSALS).collect()),
                (n, (0..=DFLASH2_DRAFT_PROPOSALS).rev().collect()),
            ],
        )?;
        let alone = plans
            .iter()
            .map(|plan| whole_turn_lane(&mut tiny_paged_int8_dflash_inner(seed), plan))
            .collect::<Result<Vec<_>>>()?;
        let linear = linear_layers(&tiny_paged_int8_dflash_inner(seed));
        let [
            (runs, waves, states, verifies),
            (owner_runs, owner_waves, owner_states, owner_verifies),
        ] = waved_and_per_owner(seed, &plans, |driver, inner| {
            for lane in 0..2 {
                driver.prefill(inner, lane)?;
            }
            while driver.wave(inner)? {}
            Ok(())
        })?;
        for ((run, alone), plan) in runs.iter().zip(&alone).zip(&plans) {
            let tag = format!("lane {}", plan.seq);
            assert_lane_run(run, alone, &tag);
            assert_eq!(
                keeps(alone),
                (1..=DFLASH2_VERIFY_ROWS).collect(),
                "{tag}: every keep"
            );
        }
        assert_states(&states, &owner_states, "waved against per-owner");
        let shared = waves.iter().filter(|wave| wave.rows.len() == 2).count();
        assert!(shared >= 4, "shared steps: {waves:?}");
        for (wave, owner) in waves.iter().zip(&owner_waves) {
            let lanes = wave.rows.len() as u64;
            let expected = if lanes == 2 { (0, (1, 2)) } else { (1, (0, 0)) };
            assert_eq!(
                (wave.forwards, wave.waves),
                expected,
                "a shared step is one wave, a lone lane its kernel: {wave:?}"
            );
            assert_eq!(
                (owner.forwards, owner.waves),
                (lanes, (0, 0)),
                "waves off: one lane kernel per lane: {owner:?}"
            );
            assert_eq!(
                wave.launches,
                lanes * linear,
                "one complete GDN launch per lane per layer: {wave:?}"
            );
        }
        let shared = shared as u64;
        for (lane, plan) in plans.iter().enumerate() {
            let cycles = runs[lane].cycles.len() as u64;
            assert_eq!(
                verifies[lane],
                LaneVerifies {
                    waves: [shared, 0, 0],
                    alone: cycles - shared,
                },
                "lane {}: its verify counts",
                plan.seq
            );
            assert_eq!(
                verifies[lane].fields(),
                format!(
                    "wave_cycles={shared} lanes2={shared} lanes3=0 lanes4=0 alone_cycles={}",
                    cycles - shared
                )
            );
            assert_eq!(
                owner_verifies[lane],
                LaneVerifies {
                    waves: [0; 3],
                    alone: owner_runs[lane].cycles.len() as u64,
                },
                "lane {}: waves off, every verify alone",
                plan.seq
            );
        }
        Ok(())
    }

    /// A load whose projections the lane tier cannot hold (the tiny
    /// fixture's plain bf16 ones) is not wave-ready, and its lanes verify
    /// alone: every shared step is one lane kernel per lane, no wave, and
    /// each lane equals its whole turn alone.
    #[test]
    fn a_load_off_the_lane_tier_verifies_each_lane_alone() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_load_off_the_lane_tier_verifies_each_lane_alone",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = two_lane_plans(
            seed,
            0.0,
            [
                (24, (0..=DFLASH2_DRAFT_PROPOSALS).collect()),
                (24, (0..=DFLASH2_DRAFT_PROPOSALS).rev().collect()),
            ],
        )?;
        let alone = plans
            .iter()
            .map(|plan| whole_turn_lane(&mut tiny_paged_int8_dflash_inner(seed), plan))
            .collect::<Result<Vec<_>>>()?;
        let mut inner = wave_inner(seed, false);
        let hidden = i64::from(inner.config.hidden_size);
        let reason =
            crate::models::qwen3_5::scheduled_dflash2::wave_unready(&inner.layers, hidden)?;
        assert!(
            reason
                .as_deref()
                .is_some_and(|reason| reason.starts_with("layer 0: ")),
            "the first layer is off the tier: {reason:?}"
        );
        inner.dflash2_wave_ready =
            crate::models::qwen3_5::scheduled_dflash2::wave_ready_at_load(&inner);
        assert!(!inner.dflash2_wave_ready, "not wave-ready");
        let (runs, waves) = scheduled_lanes(&mut inner, &plans)?;
        for ((run, alone), plan) in runs.iter().zip(&alone).zip(&plans) {
            assert_lane_run(run, alone, &format!("lane {}", plan.seq));
        }
        assert!(
            waves.iter().any(|wave| wave.rows.len() == 2),
            "shared steps: {waves:?}"
        );
        for wave in &waves {
            assert_eq!(
                (wave.forwards, wave.waves),
                (wave.rows.len() as u64, (0, 0)),
                "every lane alone: {wave:?}"
            );
        }
        Ok(())
    }

    /// Lane A runs alone for 3 cycles (its lane kernel), lane B prefills
    /// between two of A's cycles and joins, its first cycle packing its
    /// cells inside the wave; both verify in waves until B ends; A goes on
    /// alone. Each lane's cycles, tokens and settled cells equal its whole
    /// turn alone, and every commit's state equals the per-owner run's.
    #[test]
    fn a_lane_joining_and_leaving_a_wave_keeps_its_bits() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_lane_joining_and_leaving_a_wave_keeps_its_bits",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = two_lane_plans(
            seed,
            0.0,
            [
                (48, (0..=DFLASH2_DRAFT_PROPOSALS).collect()),
                (20, (0..=DFLASH2_DRAFT_PROPOSALS).rev().collect()),
            ],
        )?;
        let alone = plans
            .iter()
            .map(|plan| whole_turn_lane(&mut tiny_paged_int8_dflash_inner(seed), plan))
            .collect::<Result<Vec<_>>>()?;
        let joined = std::cell::Cell::new(None);
        let [(runs, waves, states, _), (_, _, owner_states, _)] =
            waved_and_per_owner(seed, &plans, |driver, inner| {
                driver.prefill(inner, 0)?;
                for _ in 0..3 {
                    assert!(driver.wave(inner)?, "A decodes alone");
                }
                driver.prefill(inner, 1)?;
                let b = plans[1].seq;
                assert!(
                    inner.dflash2_states.seq_lanes[&b].gdn.is_none(),
                    "B joins with its cells unpacked"
                );
                assert!(driver.wave(inner)?, "B joins");
                assert!(inner.dflash2_states.seq_lanes[&b].gdn.is_some());
                if joined.get().is_none() {
                    joined.set(Some(driver.waves.len() - 1));
                }
                while driver.wave(inner)? {}
                Ok(())
            })?;
        for ((run, alone), plan) in runs.iter().zip(&alone).zip(&plans) {
            assert_lane_run(run, alone, &format!("lane {}", plan.seq));
        }
        assert_states(&states, &owner_states, "waved against per-owner");
        let joined = joined.get().expect("the join wave");
        assert_eq!(joined, 3);
        let (before, after) = waves.split_at(joined);
        assert!(
            before
                .iter()
                .all(|wave| wave.rows.len() == 1 && (wave.forwards, wave.waves) == (1, (0, 0))),
            "A alone runs its lane kernel: {waves:?}"
        );
        assert_eq!(
            (after[0].rows.len(), after[0].forwards, after[0].waves),
            (2, 0, (1, 2)),
            "B's first cycle is in the wave: {waves:?}"
        );
        let last_shared = after
            .iter()
            .rposition(|wave| wave.rows.len() == 2)
            .expect("shared waves");
        let tail = &after[last_shared + 1..];
        assert!(
            !tail.is_empty()
                && tail
                    .iter()
                    .all(|wave| wave.rows.len() == 1 && (wave.forwards, wave.waves) == (1, (0, 0))),
            "A goes on alone on its lane kernel after B ends: {waves:?}"
        );
        assert!(
            after[..=last_shared]
                .iter()
                .all(|wave| (wave.forwards, wave.waves) == (0, (1, 2))),
            "every shared step is one wave: {waves:?}"
        );
        Ok(())
    }

    /// A wave of one runs the lane kernel and nothing else: a lone lane on
    /// a wave-ready load runs its lane kernel at every step and equals its
    /// whole turn alone.
    #[test]
    fn a_wave_of_one_runs_the_lane_kernel() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("a_wave_of_one_runs_the_lane_kernel") {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = two_lane_plans(
            seed,
            0.0,
            [
                (48, (0..=DFLASH2_DRAFT_PROPOSALS).collect()),
                (48, (0..=DFLASH2_DRAFT_PROPOSALS).rev().collect()),
            ],
        )?;
        let alone = whole_turn_lane(&mut tiny_paged_int8_dflash_inner(seed), &plans[0])?;
        let mut inner = wave_inner(seed, true);
        let linear = linear_layers(&inner);
        let (runs, waves) = scheduled_lanes(&mut inner, &plans[..1])?;
        assert_lane_run(&runs[0], &alone, "lone lane");
        assert!(
            waves
                .iter()
                .all(|wave| (wave.forwards, wave.waves) == (1, (0, 0)) && wave.launches == linear),
            "a lone lane runs its lane kernel only: {waves:?}"
        );
        Ok(())
    }

    /// A lane that kept 0 (its commit clamped to 0, as a cancel or a stop
    /// inside the cycle does) or whose wave was retracted after its verify
    /// verifies in the very next wave: every step of two lanes is one wave.
    /// Each lane equals its whole turn alone, every commit's state equals the
    /// per-owner run's, and each lane's counts are its committed cycles (a
    /// cycle that kept nothing is not one).
    #[test]
    fn a_zero_keep_lane_rejoins_the_next_wave() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("a_zero_keep_lane_rejoins_the_next_wave")
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = two_lane_plans(
            seed,
            0.0,
            [
                (48, (0..=DFLASH2_DRAFT_PROPOSALS).collect()),
                (48, (0..=DFLASH2_DRAFT_PROPOSALS).rev().collect()),
            ],
        )?;
        let alone = plans
            .iter()
            .map(|plan| whole_turn_lane(&mut tiny_paged_int8_dflash_inner(seed), plan))
            .collect::<Result<Vec<_>>>()?;
        let voids = |wave: usize| match wave {
            2 => Void::KeepZero(plans[0].seq),
            4 => Void::KeepZero(plans[1].seq),
            6 => Void::Retract,
            _ => Void::None,
        };
        let [(runs, waves, states, verifies), (_, _, owner_states, _)] =
            waved_and_per_owner(seed, &plans, |driver, inner| {
                for lane in 0..2 {
                    driver.prefill(inner, lane)?;
                }
                for wave in 0.. {
                    if !driver.wave_with(inner, voids(wave))? {
                        break;
                    }
                }
                Ok(())
            })?;
        for ((run, alone), plan) in runs.iter().zip(&alone).zip(&plans) {
            assert_lane_run(run, alone, &format!("lane {}", plan.seq));
        }
        assert_states(&states, &owner_states, "waved against per-owner");
        for (step, wave) in waves.iter().enumerate().take(9) {
            assert_eq!(
                (wave.rows.len(), wave.forwards, wave.waves),
                (2, 0, (1, 2)),
                "step {step}: both lanes in one wave: {waves:?}"
            );
        }
        let shared = waves.iter().filter(|wave| wave.waves == (1, 2)).count() as u64;
        for (lane, plan) in plans.iter().enumerate() {
            let void_steps = if lane == 0 { [2, 6] } else { [4, 6] };
            let committed_shared = shared - void_steps.len() as u64;
            assert_eq!(
                verifies[lane],
                LaneVerifies {
                    waves: [committed_shared, 0, 0],
                    alone: runs[lane].cycles.len() as u64 - committed_shared,
                },
                "lane {}: its counts are its committed cycles",
                plan.seq
            );
        }
        Ok(())
    }

    /// Two lanes verify in one wave in every step an autoregressive peer
    /// rides (the peer's row first, its own one-row forward beside the
    /// wave), also when the peer's row keeps 0 or the step is retracted
    /// after its verify. Each lane equals its whole turn alone, the peer
    /// equals its run alone, and every commit's state equals the per-owner
    /// run's.
    #[test]
    fn lanes_wave_beside_an_ar_peer() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("lanes_wave_beside_an_ar_peer") {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = two_lane_plans(
            seed,
            0.0,
            [
                (48, (0..=DFLASH2_DRAFT_PROPOSALS).collect()),
                (48, (0..=DFLASH2_DRAFT_PROPOSALS).rev().collect()),
            ],
        )?;
        let alone = plans
            .iter()
            .map(|plan| whole_turn_lane(&mut tiny_paged_int8_dflash_inner(seed), plan))
            .collect::<Result<Vec<_>>>()?;
        let peer_seq = 3;
        // No block in common with either lane's prompt.
        let peer_prompt: Vec<u32> = (0..40u32).map(|i| (i * 3 + 2) % VOCAB).collect();
        let peer_n = 12;
        let peer_alone = ar_peer_alone(
            &mut tiny_paged_int8_dflash_inner(seed),
            ArPeer::new(peer_seq, peer_prompt.clone(), peer_n),
        )?;
        let peers = std::cell::RefCell::new(Vec::new());
        let [
            (runs, waves, states, _),
            (owner_runs, owner_waves, owner_states, _),
        ] = waved_and_per_owner(seed, &plans, |driver, inner| {
            driver
                .peers
                .push(ArPeer::new(peer_seq, peer_prompt.clone(), peer_n));
            driver.prefill_peer(inner, 0)?;
            driver.prefill(inner, 0)?;
            driver.prefill(inner, 1)?;
            for wave in 0.. {
                let void = match wave {
                    3 => Void::KeepZero(peer_seq),
                    5 => Void::Retract,
                    _ => Void::None,
                };
                if !driver.wave_with(inner, void)? {
                    break;
                }
            }
            assert!(inner.dflash2_states.peer_cells.is_empty());
            peers
                .borrow_mut()
                .push(driver.finish_peers(inner)?.remove(0));
            Ok(())
        })?;
        for ((run, alone), plan) in runs.iter().zip(&alone).zip(&plans) {
            assert_lane_run(run, alone, &format!("beside a peer: lane {}", plan.seq));
        }
        for (run, owner) in runs.iter().zip(&owner_runs) {
            assert_lane_run(run, owner, "waved against per-owner");
        }
        assert_states(&states, &owner_states, "waved against per-owner");
        for peer in peers.borrow().iter() {
            assert_ar_peer(peer, &peer_alone);
        }
        assert_eq!(waves.len(), owner_waves.len());
        let mut beside = 0;
        for (step, (wave, owner)) in waves.iter().zip(&owner_waves).enumerate() {
            if wave.rows.len() != 3 || wave.rows[0] != 1 {
                continue;
            }
            beside += 1;
            assert_eq!(
                (wave.forwards, wave.waves),
                (0, (1, 2)),
                "step {step}: the lanes wave beside the peer: {waves:?}"
            );
            assert_eq!(
                (owner.forwards, owner.waves),
                (2, (0, 0)),
                "waves off, step {step}: {owner_waves:?}"
            );
        }
        assert!(beside >= 8, "steps of the peer and both lanes: {waves:?}");
        Ok(())
    }

    /// A wave whose forward fails at its last GDN layer (lane 2's state
    /// there cast to bf16, which the complete kernel declines), after every
    /// earlier layer adopted both lanes' slots and stored their K/V: the
    /// verify fails, both lanes' rows roll back to their frontiers, no
    /// verify stays open, both lanes' slots are their cells before the wave
    /// (the bad state gone), and both lanes rerun the cycle in a wave, then
    /// in waves, to their whole turns alone. Each lane's counts are its
    /// committed cycles: the failed wave is not one.
    #[test]
    fn a_wave_error_aborts_every_lane() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("a_wave_error_aborts_every_lane") {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = two_lane_plans(
            seed,
            0.0,
            [
                (40, (0..=DFLASH2_DRAFT_PROPOSALS).collect()),
                (40, (0..=DFLASH2_DRAFT_PROPOSALS).rev().collect()),
            ],
        )?;
        let alone = plans
            .iter()
            .map(|plan| whole_turn_lane(&mut tiny_paged_int8_dflash_inner(seed), plan))
            .collect::<Result<Vec<_>>>()?;
        let mut inner = wave_inner(seed, true);
        let last_linear = inner
            .layer_kinds
            .iter()
            .rposition(|kind| matches!(kind, super::super::decoder_layer::Qwen3_5LayerKind::Linear))
            .expect("a linear layer");
        let mut driver = LaneDriver::new(&plans);
        driver.prefill(&mut inner, 0)?;
        driver.prefill(&mut inner, 1)?;
        for _ in 0..2 {
            assert!(driver.wave(&mut inner)?);
        }
        let seqs = [plans[0].seq, plans[1].seq];
        let frontiers = seqs.map(|seq| cursor_of(&inner, seq));
        let mut before = Vec::new();
        for seq in seqs {
            HybridSchedulerBackend::activate_paged_seq(&mut inner, seq)?;
            before.push(cell_bits(inner.caches.as_ref().expect("caches"))?);
        }
        let caches = inner.caches.as_mut().expect("lane 2's caches");
        let slot = caches[last_linear]
            .as_arrays_cache_mut()
            .expect("a GDN slot");
        let state = slot.get(1).expect("a state").astype(DType::BFloat16)?;
        slot.set(1, state)?;

        assert!(driver.wave_with(&mut inner, Void::Fail)?);
        let reason = driver.failure.take().unwrap_or_default();
        assert!(
            reason.contains("the complete kernel declined lane 1"),
            "the wave failed at lane 2's bad state: {reason}"
        );
        let failed = driver.waves.last().expect("the failed wave");
        assert_eq!((failed.forwards, failed.waves), (0, (1, 2)), "{failed:?}");
        assert_eq!(
            seqs.map(|seq| cursor_of(&inner, seq)),
            frontiers,
            "every lane's rows rolled back"
        );
        assert!(inner.dflash2_states.lane_batch.is_none());
        for (seq, before) in seqs.iter().zip(&before) {
            let lane = &inner.dflash2_states.seq_lanes[seq];
            assert!(lane.open.is_none(), "sequence {seq}: no open verify");
            HybridSchedulerBackend::activate_paged_seq(&mut inner, *seq)?;
            assert!(
                cell_bits(inner.caches.as_ref().expect("caches"))? == *before,
                "sequence {seq}: its slots are its cells before the wave"
            );
        }
        let failed_at = driver.waves.len() - 1;
        while driver.wave(&mut inner)? {}
        let (runs, waves) = driver.finish(&mut inner)?;
        for ((run, alone), plan) in runs.iter().zip(&alone).zip(&plans) {
            assert_lane_run(run, alone, &format!("lane {}", plan.seq));
        }
        assert_eq!(
            (waves[failed_at + 1].forwards, waves[failed_at + 1].waves),
            (0, (1, 2)),
            "the rerun verifies both lanes in one wave: {waves:?}"
        );
        let committed_waves = waves
            .iter()
            .enumerate()
            .filter(|&(step, wave)| step != failed_at && wave.waves == (1, 2))
            .count() as u64;
        for (run, plan) in runs.iter().zip(&plans) {
            let verifies = inner.dflash2_states.seq_lanes[&plan.seq].verifies;
            assert_eq!(
                verifies,
                LaneVerifies {
                    waves: [committed_waves, 0, 0],
                    alone: run.cycles.len() as u64 - committed_waves,
                },
                "lane {}: its counts are its committed cycles",
                plan.seq
            );
        }
        Ok(())
    }

    /// The tiny fixture with `lanes` granted and its verify waves on or off.
    fn lanes_inner(seed: u64, waves: bool, lanes: usize) -> Qwen35Inner {
        let mut inner = tiny_paged_int8_dflash_inner(seed);
        inner.grant_scheduled_lanes(lanes);
        inner.dflash2_wave_ready = waves;
        inner
    }

    /// Sequence `seq`'s prompt of `len` tokens: no first block in common
    /// with another sequence's, so each prefills from 0.
    fn wave_prompt(seq: u32, len: usize) -> Vec<u32> {
        let (mul, add) = [(7, 3), (5, 1), (3, 2), (11, 5), (13, 7)][(seq as usize - 1) % 5];
        (0..len as u32).map(|i| (i * mul + add) % VOCAB).collect()
    }

    /// One plan per `(prompt length, output length, schedule)`, sequences
    /// 1.., each with its paged AR reference.
    fn wave_plans(
        seed: u64,
        temperature: f64,
        lanes: &[(usize, usize, Vec<usize>)],
    ) -> Result<Vec<LanePlan>> {
        let mut plans = Vec::new();
        for (index, (len, n, schedule)) in lanes.iter().enumerate() {
            let seq = index as u32 + 1;
            let prompt = wave_prompt(seq, *len);
            let (reference, _) = paged_ar(&mut tiny_paged_int8_dflash_inner(seed), &prompt, *n)?;
            plans.push(LanePlan {
                seq,
                prompt,
                n: *n,
                temperature,
                reference,
                schedule: schedule.clone(),
            });
        }
        Ok(plans)
    }

    /// Each plan's lane alone on a fresh fixture: its whole turn.
    fn wave_alone(seed: u64, plans: &[LanePlan]) -> Result<Vec<LaneRun>> {
        plans
            .iter()
            .map(|plan| whole_turn_lane(&mut tiny_paged_int8_dflash_inner(seed), plan))
            .collect()
    }

    type VfyStates = Vec<(u32, Vec<Vec<u32>>)>;

    /// `drive` on `fixture(true)` (waves on) and `fixture(false)` (each lane
    /// alone, the per-owner route): the runs, steps and per-commit states of
    /// each.
    #[allow(clippy::type_complexity)]
    fn wave_and_per_owner_runs(
        plans: &[LanePlan],
        fixture: impl Fn(bool) -> Qwen35Inner,
        drive: impl Fn(&mut LaneDriver<'_>, &mut Qwen35Inner) -> Result<()>,
    ) -> Result<[(Vec<LaneRun>, Vec<Wave>, VfyStates); 2]> {
        let mut out = Vec::new();
        for waves in [true, false] {
            let mut inner = fixture(waves);
            let mut driver = LaneDriver::new(plans);
            driver.states = Some(Vec::new());
            drive(&mut driver, &mut inner)?;
            let states = driver.states.take().unwrap_or_default();
            let (runs, waves) = driver.finish(&mut inner)?;
            out.push((runs, waves, states));
        }
        let per_owner = out.pop().expect("the per-owner run");
        let waved = out.pop().expect("the waved run");
        Ok([waved, per_owner])
    }

    /// What a step of `lanes` plain lanes runs on a wave-ready load: its
    /// lane-kernel forwards and its (waves, lanes in waves).
    fn wave_step(lanes: usize) -> (u64, (u64, u64)) {
        match lanes {
            1 => (1, (0, 0)),
            2..=4 => (0, (1, lanes as u64)),
            _ => (lanes as u64 - 4, (1, 4)),
        }
    }

    /// Another sequence, or none, is the active one before a step: the
    /// lanes of the step are lent from the active slot and from the table in
    /// every arrangement.
    fn wave_shift_active(
        inner: &mut Qwen35Inner,
        driver: &LaneDriver<'_>,
        step: usize,
    ) -> Result<()> {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let prefilled: Vec<u32> = (0..driver.plans.len())
            .filter(|&lane| !driver.retired[lane] && !driver.runs[lane].tokens.is_empty())
            .map(|lane| driver.plans[lane].seq)
            .collect();
        match step % 3 {
            0 => HybridSchedulerBackend::park_active_scheduled_recurrent(inner),
            1 => match prefilled.first() {
                Some(&seq) => HybridSchedulerBackend::activate_paged_seq(inner, seq),
                None => Ok(()),
            },
            _ => Ok(()),
        }
    }

    /// Three lanes wave together; the middle one is cancelled mid-run (its
    /// commit clamped to keep 0 inside a cycle, or between two steps while
    /// it is the active sequence) and released. The two that stay go on in
    /// waves of two, and each equals its whole turn alone and the per-owner
    /// run from the same steps.
    #[test]
    fn three_lanes_one_cancelled_mid_run() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("three_lanes_one_cancelled_mid_run") {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = wave_plans(
            seed,
            0.0,
            &[
                (40, 40, vec![0, 1, 2]),
                (40, 40, vec![2, 0, 1]),
                (56, 40, vec![1, 3, 0]),
            ],
        )?;
        let alone = wave_alone(seed, &plans)?;
        for clamp in [true, false] {
            let tag = if clamp {
                "cancel inside a cycle"
            } else {
                "cancel between steps"
            };
            let [(runs, waves, states), (owner_runs, _, owner_states)] = wave_and_per_owner_runs(
                &plans,
                |waves| lanes_inner(seed, waves, 4),
                |driver, inner| {
                    for lane in 0..3 {
                        driver.prefill(inner, lane)?;
                    }
                    for _ in 0..2 {
                        assert!(driver.wave(inner)?);
                    }
                    let cancelled = plans[1].seq;
                    if clamp {
                        assert!(driver.wave_with(inner, Void::KeepZero(cancelled))?);
                    } else {
                        HybridSchedulerBackend::activate_paged_seq(inner, cancelled)?;
                    }
                    HybridSchedulerBackend::release_scheduled_recurrent_for(inner, cancelled);
                    HybridSchedulerBackend::release_scheduled_speculation(inner, cancelled);
                    HybridSchedulerBackend::release_scheduled_cache(inner, cancelled)?;
                    driver.retired[1] = true;
                    while driver.wave(inner)? {}
                    Ok(())
                },
            )?;
            for lane in [0, 2] {
                assert_lane_run(
                    &runs[lane],
                    &alone[lane],
                    &format!("{tag}: lane {} against alone", plans[lane].seq),
                );
            }
            for lane in 0..3 {
                assert_lane_run(
                    &runs[lane],
                    &owner_runs[lane],
                    &format!("{tag}: lane {} against per-owner", plans[lane].seq),
                );
            }
            assert_states(&states, &owner_states, tag);
            let together = if clamp { 3 } else { 2 };
            for (step, wave) in waves.iter().enumerate() {
                let lanes = wave.rows.len();
                if step < together {
                    assert_eq!(lanes, 3, "{tag}: step {step}: {waves:?}");
                } else {
                    assert!(lanes <= 2, "{tag}: step {step}: {waves:?}");
                }
                assert_eq!(
                    (wave.forwards, wave.waves),
                    wave_step(lanes),
                    "{tag}: step {step}: {waves:?}"
                );
            }
            let pairs = waves[together..]
                .iter()
                .filter(|wave| wave.rows.len() == 2)
                .count();
            assert!(pairs >= 4, "{tag}: the two that stay wave on: {waves:?}");
        }
        Ok(())
    }

    /// Five owners at unlike depths (24 .. 90 prompt rows) join one by one
    /// and leave at unlike times, with another sequence (or none) active
    /// before every step: steps of 1, 2, 3, 4 and 5 lanes (a wave of 4 and
    /// one lane kernel), each joiner packed inside its first wave. Every
    /// lane equals its whole turn alone and the per-owner run.
    #[test]
    fn lanes_at_unlike_depths_join_and_leave() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("lanes_at_unlike_depths_join_and_leave") {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = wave_plans(
            seed,
            0.0,
            &[
                (24, 48, vec![0, 1, 0, 2]),
                (40, 14, vec![1, 0]),
                (71, 30, (0..=DFLASH2_DRAFT_PROPOSALS).collect()),
                (57, 22, (0..=DFLASH2_DRAFT_PROPOSALS).rev().collect()),
                (90, 26, vec![2, 0, 1]),
            ],
        )?;
        let alone = wave_alone(seed, &plans)?;
        let linear = linear_layers(&tiny_paged_int8_dflash_inner(seed));
        let [
            (runs, waves, states),
            (owner_runs, owner_waves, owner_states),
        ] = wave_and_per_owner_runs(
            &plans,
            |waves| lanes_inner(seed, waves, 8),
            |driver, inner| {
                let mut step = 0;
                for lane in 0..plans.len() {
                    driver.prefill(inner, lane)?;
                    assert!(
                        inner.dflash2_states.seq_lanes[&plans[lane].seq]
                            .gdn
                            .is_none(),
                        "lane {} joins with its cells unpacked",
                        plans[lane].seq
                    );
                    for _ in 0..2 {
                        wave_shift_active(inner, driver, step)?;
                        step += 1;
                        assert!(driver.wave(inner)?);
                    }
                }
                loop {
                    wave_shift_active(inner, driver, step)?;
                    step += 1;
                    if !driver.wave(inner)? {
                        return Ok(());
                    }
                }
            },
        )?;
        for (lane, plan) in plans.iter().enumerate() {
            assert_lane_run(
                &runs[lane],
                &alone[lane],
                &format!("lane {} against alone", plan.seq),
            );
            assert_lane_run(
                &runs[lane],
                &owner_runs[lane],
                &format!("lane {} against per-owner", plan.seq),
            );
        }
        assert_states(&states, &owner_states, "waved against per-owner");
        assert_eq!(waves.len(), owner_waves.len());
        let mut sizes = std::collections::BTreeSet::new();
        for (step, (wave, owner)) in waves.iter().zip(&owner_waves).enumerate() {
            let lanes = wave.rows.len();
            sizes.insert(lanes);
            assert_eq!(
                (wave.forwards, wave.waves),
                wave_step(lanes),
                "step {step}: {waves:?}"
            );
            assert_eq!(
                (owner.forwards, owner.waves),
                (lanes as u64, (0, 0)),
                "waves off, step {step}: {owner_waves:?}"
            );
            assert_eq!(
                wave.launches,
                lanes as u64 * linear,
                "step {step}: one complete GDN launch per lane per layer"
            );
        }
        assert_eq!(sizes, (1..=5).collect(), "steps of every size: {waves:?}");
        assert!(
            runs[0].cycles.len() >= 12
                && waves.iter().filter(|wave| wave.rows.len() > 1).count() >= 8,
            "the first lane lives through its peers' joins and ends: {waves:?}"
        );
        Ok(())
    }

    /// One of three lanes keeps 0 (the first, the middle or the last row of
    /// the step): the next step runs all three in one wave again, and so
    /// does the step after a retracted wave. Every lane equals its whole
    /// turn alone and the per-owner run.
    #[test]
    fn a_zero_keep_lane_rejoins_the_wave() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("a_zero_keep_lane_rejoins_the_wave") {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = wave_plans(
            seed,
            0.0,
            &[
                (40, 44, vec![0, 1, 2]),
                (40, 44, vec![2, 0, 1]),
                (56, 44, vec![1, 2, 0]),
            ],
        )?;
        let alone = wave_alone(seed, &plans)?;
        for void_row in 0..3 {
            let tag = format!("row {void_row} keeps 0");
            let [(runs, waves, states), (owner_runs, _, owner_states)] = wave_and_per_owner_runs(
                &plans,
                |waves| lanes_inner(seed, waves, 4),
                |driver, inner| {
                    for lane in 0..3 {
                        driver.prefill(inner, lane)?;
                    }
                    for wave in 0.. {
                        let void = match wave {
                            2 => Void::KeepZero(plans[void_row].seq),
                            5 => Void::Retract,
                            _ => Void::None,
                        };
                        if !driver.wave_with(inner, void)? {
                            break;
                        }
                    }
                    Ok(())
                },
            )?;
            for (lane, plan) in plans.iter().enumerate() {
                assert_lane_run(
                    &runs[lane],
                    &alone[lane],
                    &format!("{tag}: lane {} against alone", plan.seq),
                );
                assert_lane_run(
                    &runs[lane],
                    &owner_runs[lane],
                    &format!("{tag}: lane {} against per-owner", plan.seq),
                );
            }
            assert_states(&states, &owner_states, &tag);
            let seen: Vec<_> = waves[..8]
                .iter()
                .map(|wave| (wave.rows.len(), wave.forwards, wave.waves))
                .collect();
            assert_eq!(seen, vec![(3, 0, (1, 3)); 8], "{tag}: {waves:?}");
        }
        Ok(())
    }

    /// The recurrent state of `seq` at decoder layer `layer` cast to bf16:
    /// the complete GDN kernel declines it, so the verify forward of `seq`
    /// fails at that layer.
    fn wave_break_state(inner: &mut Qwen35Inner, seq: u32, layer: usize) -> Result<()> {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        HybridSchedulerBackend::activate_paged_seq(inner, seq)?;
        let slot = inner.caches.as_mut().expect("caches")[layer]
            .as_arrays_cache_mut()
            .expect("a GDN slot");
        let state = slot.get(1).expect("a state").astype(DType::BFloat16)?;
        slot.set(1, state)?;
        Ok(())
    }

    /// A wave of three (also the step after one lane kept 0) whose forward
    /// fails at one lane, first or middle, at the first or the last GDN
    /// layer, with that lane parked, active, or no sequence active: every
    /// lane's rows roll back, no verify stays open, every lent set is back
    /// where it was (the active sequence and its slot unchanged, every
    /// sequence's cells in the table), every lane's slots are its cells
    /// before the wave, and every owner reruns in a wave of three to its
    /// whole turn alone.
    #[test]
    fn a_wave_error_at_any_lane_aborts_all_and_all_rerun() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_wave_error_at_any_lane_aborts_all_and_all_rerun",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = wave_plans(
            seed,
            0.0,
            &[
                (40, 36, vec![0, 2, 1]),
                (40, 36, vec![1, 0, 3]),
                (56, 36, vec![2, 1, 0]),
            ],
        )?;
        let alone = wave_alone(seed, &plans)?;
        let seqs: Vec<u32> = plans.iter().map(|plan| plan.seq).collect();
        for (case, bad, last_layer, active, voided_row) in [
            (
                "first lane, first GDN layer, parked",
                0usize,
                false,
                Some(2usize),
                None,
            ),
            (
                "middle lane, last GDN layer, active",
                1,
                true,
                Some(1),
                None,
            ),
            ("first lane, no active sequence", 0, true, None, None),
            (
                "first lane, the step after the last lane kept 0",
                0,
                true,
                Some(0),
                Some(2usize),
            ),
        ] {
            let mut inner = lanes_inner(seed, true, 4);
            let linear: Vec<usize> = inner
                .layer_kinds
                .iter()
                .enumerate()
                .filter(|(_, kind)| {
                    matches!(kind, super::super::decoder_layer::Qwen3_5LayerKind::Linear)
                })
                .map(|(index, _)| index)
                .collect();
            assert!(!linear.is_empty(), "{case}: the fixture has GDN layers");
            let layer = if last_layer {
                linear[linear.len() - 1]
            } else {
                linear[0]
            };
            let mut driver = LaneDriver::new(&plans);
            for lane in 0..3 {
                driver.prefill(&mut inner, lane)?;
            }
            for _ in 0..2 {
                assert!(driver.wave(&mut inner)?);
            }
            if let Some(row) = voided_row {
                assert!(driver.wave_with(&mut inner, Void::KeepZero(seqs[row]))?);
            }
            let frontiers: Vec<u32> = seqs.iter().map(|&seq| cursor_of(&inner, seq)).collect();
            let mut before = Vec::new();
            for &seq in &seqs {
                HybridSchedulerBackend::activate_paged_seq(&mut inner, seq)?;
                before.push(cell_bits(inner.caches.as_ref().expect("caches"))?);
            }
            wave_break_state(&mut inner, seqs[bad], layer)?;
            match active {
                Some(lane) => HybridSchedulerBackend::activate_paged_seq(&mut inner, seqs[lane])?,
                None => HybridSchedulerBackend::park_active_scheduled_recurrent(&mut inner)?,
            }
            let active_before = inner.active_scheduled_seq;
            assert_eq!(active_before, active.map(|lane| seqs[lane]), "{case}");

            assert!(driver.wave_with(&mut inner, Void::Fail)?);
            let reason = driver.failure.take().unwrap_or_default();
            assert!(
                reason.contains(&format!("the complete kernel declined lane {bad}")),
                "{case}: the wave failed at the bad lane: {reason}"
            );
            let failed = driver.waves.last().expect("the failed wave");
            assert_eq!(
                (failed.forwards, failed.waves),
                (0, (1, 3)),
                "{case}: {failed:?}"
            );
            assert_eq!(
                inner.active_scheduled_seq, active_before,
                "{case}: the active sequence"
            );
            assert_eq!(
                inner.caches.is_some(),
                active_before.is_some(),
                "{case}: the active sequence's cells are in its slot"
            );
            assert!(inner.dflash2_states.lane_batch.is_none(), "{case}");
            for (index, &seq) in seqs.iter().enumerate() {
                assert_eq!(
                    cursor_of(&inner, seq),
                    frontiers[index],
                    "{case}: sequence {seq}: its rows rolled back"
                );
                let lane = &inner.dflash2_states.seq_lanes[&seq];
                assert!(
                    lane.open.is_none(),
                    "{case}: sequence {seq}: no open verify"
                );
                assert!(
                    HybridSchedulerBackend::has_scheduled_recurrent(&inner, seq),
                    "{case}: sequence {seq}: its cells are back"
                );
            }
            for (index, &seq) in seqs.iter().enumerate() {
                HybridSchedulerBackend::activate_paged_seq(&mut inner, seq)?;
                assert!(
                    cell_bits(inner.caches.as_ref().expect("caches"))? == before[index],
                    "{case}: sequence {seq}: its slots are its cells before the wave"
                );
            }
            let failed_at = driver.waves.len() - 1;
            while driver.wave(&mut inner)? {}
            let (runs, waves) = driver.finish(&mut inner)?;
            for (lane, plan) in plans.iter().enumerate() {
                assert_lane_run(
                    &runs[lane],
                    &alone[lane],
                    &format!("{case}: lane {}", plan.seq),
                );
            }
            assert_eq!(
                (waves[failed_at + 1].forwards, waves[failed_at + 1].waves),
                (0, (1, 3)),
                "{case}: every owner reruns in one wave of three: {waves:?}"
            );
        }
        Ok(())
    }

    /// Five lanes: a step is a wave of four and the fifth row's lane
    /// kernel. The lane kernel fails after the wave verified (its state cast
    /// to bf16): the wave's lanes are retracted too, every lane's rows roll
    /// back, no verify stays open, the wave lanes' slots are their cells
    /// before the step. With the bad lane's slots put back on its cells,
    /// every owner reruns as a wave of four and a lane kernel to its whole
    /// turn alone; each lane's counts are its committed cycles (the wave of
    /// the retracted step is not one).
    #[test]
    fn a_lane_kernel_error_after_a_wave_retracts_the_wave_lanes() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_lane_kernel_error_after_a_wave_retracts_the_wave_lanes",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = wave_plans(
            seed,
            0.0,
            &[
                (40, 36, vec![0, 2, 1]),
                (40, 36, vec![1, 0, 3]),
                (56, 36, vec![2, 1, 0]),
                (24, 36, vec![1, 2, 0]),
                (72, 36, vec![0, 1, 2]),
            ],
        )?;
        let alone = wave_alone(seed, &plans)?;
        let seqs: Vec<u32> = plans.iter().map(|plan| plan.seq).collect();
        let mut inner = lanes_inner(seed, true, 8);
        let layer = inner
            .layer_kinds
            .iter()
            .rposition(|kind| matches!(kind, super::super::decoder_layer::Qwen3_5LayerKind::Linear))
            .expect("a linear layer");
        let mut driver = LaneDriver::new(&plans);
        for lane in 0..plans.len() {
            driver.prefill(&mut inner, lane)?;
        }
        for _ in 0..2 {
            assert!(driver.wave(&mut inner)?);
        }
        let bad = seqs[4];
        let frontiers: Vec<u32> = seqs.iter().map(|&seq| cursor_of(&inner, seq)).collect();
        let mut before = Vec::new();
        for &seq in &seqs {
            HybridSchedulerBackend::activate_paged_seq(&mut inner, seq)?;
            before.push(cell_bits(inner.caches.as_ref().expect("caches"))?);
        }
        wave_break_state(&mut inner, bad, layer)?;

        assert!(driver.wave_with(&mut inner, Void::Fail)?);
        let reason = driver.failure.take().unwrap_or_default();
        assert!(
            reason.contains("declined") && !reason.contains("declined lane"),
            "the lane kernel failed: {reason}"
        );
        let failed = driver.waves.last().expect("the failed step");
        assert_eq!(
            (failed.forwards, failed.waves),
            (1, (1, 4)),
            "the wave ran, then the lane kernel: {failed:?}"
        );
        assert!(inner.dflash2_states.lane_batch.is_none());
        for (index, &seq) in seqs.iter().enumerate() {
            assert_eq!(
                cursor_of(&inner, seq),
                frontiers[index],
                "sequence {seq}: its rows rolled back"
            );
            let lane = &inner.dflash2_states.seq_lanes[&seq];
            assert!(lane.open.is_none(), "sequence {seq}: no open verify");
        }
        for (index, &seq) in seqs.iter().enumerate().take(4) {
            HybridSchedulerBackend::activate_paged_seq(&mut inner, seq)?;
            assert!(
                cell_bits(inner.caches.as_ref().expect("caches"))? == before[index],
                "sequence {seq}: its slots are its cells before the step"
            );
        }
        HybridSchedulerBackend::activate_paged_seq(&mut inner, bad)?;
        inner.dflash2_states.seq_lanes[&bad]
            .gdn
            .as_ref()
            .expect("the bad lane's cells")
            .apply_views(inner.caches.as_mut().expect("caches"))?;
        assert!(cell_bits(inner.caches.as_ref().expect("caches"))? == before[4]);
        let failed_at = driver.waves.len() - 1;
        while driver.wave(&mut inner)? {}
        let (runs, waves) = driver.finish(&mut inner)?;
        for (lane, plan) in plans.iter().enumerate() {
            assert_lane_run(&runs[lane], &alone[lane], &format!("lane {}", plan.seq));
            let verifies = inner.dflash2_states.seq_lanes[&plan.seq].verifies;
            assert_eq!(
                verifies.wave_cycles() + verifies.alone,
                runs[lane].cycles.len() as u64,
                "lane {}: its counts are its committed cycles, the retracted step none",
                plan.seq
            );
        }
        assert_eq!(
            (waves[failed_at + 1].forwards, waves[failed_at + 1].waves),
            (1, (1, 4)),
            "every owner reruns, a wave of four and a lane kernel: {waves:?}"
        );
        Ok(())
    }

    /// Six lanes in one step run two waves (4 and 2), with another sequence
    /// or none active before every step; every lane equals its whole turn
    /// alone and the per-owner run. Then the second wave of a step fails at
    /// its last lane after the first wave verified: all six lanes roll
    /// back, none has an open verify, every lane's slots are its cells
    /// before the step, and all six rerun in two waves.
    #[test]
    fn six_lanes_run_two_waves_and_the_second_can_fail() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "six_lanes_run_two_waves_and_the_second_can_fail",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let schedules = [
            vec![0, 1, 2],
            vec![2, 0, 1],
            vec![1, 2, 0],
            vec![0, 2, 1],
            vec![1, 0, 2],
        ];
        // Sequence 6 extends no other prompt: its first block is its own.
        let mut plans = wave_plans(
            seed,
            0.0,
            &[
                (40, 30, schedules[0].clone()),
                (40, 30, schedules[1].clone()),
                (56, 30, schedules[2].clone()),
                (24, 30, schedules[3].clone()),
                (72, 30, schedules[4].clone()),
            ],
        )?;
        let sixth: Vec<u32> = (0..48u32).map(|i| (i * 9 + 4) % VOCAB).collect();
        let (reference, _) = paged_ar(&mut tiny_paged_int8_dflash_inner(seed), &sixth, 30)?;
        plans.push(LanePlan {
            seq: 6,
            prompt: sixth,
            n: 30,
            temperature: 0.0,
            reference,
            schedule: vec![2, 1, 0],
        });
        let alone = wave_alone(seed, &plans)?;
        let [(runs, waves, states), (owner_runs, _, owner_states)] = wave_and_per_owner_runs(
            &plans,
            |waves| lanes_inner(seed, waves, 8),
            |driver, inner| {
                for lane in 0..plans.len() {
                    driver.prefill(inner, lane)?;
                }
                for step in 0.. {
                    wave_shift_active(inner, driver, step)?;
                    if !driver.wave(inner)? {
                        break;
                    }
                }
                Ok(())
            },
        )?;
        for (lane, plan) in plans.iter().enumerate() {
            assert_lane_run(
                &runs[lane],
                &alone[lane],
                &format!("lane {} against alone", plan.seq),
            );
            assert_lane_run(
                &runs[lane],
                &owner_runs[lane],
                &format!("lane {} against per-owner", plan.seq),
            );
        }
        assert_states(&states, &owner_states, "waved against per-owner");
        let sixes = waves
            .iter()
            .filter(|wave| (wave.rows.len(), wave.forwards, wave.waves) == (6, 0, (2, 6)))
            .count();
        assert!(sixes >= 6, "steps of six lanes in two waves: {waves:?}");
        assert!(
            waves
                .iter()
                .all(|wave| wave.rows.len() != 6 || (wave.forwards, wave.waves) == (0, (2, 6))),
            "{waves:?}"
        );

        let seqs: Vec<u32> = plans.iter().map(|plan| plan.seq).collect();
        let mut inner = lanes_inner(seed, true, 8);
        let layer = inner
            .layer_kinds
            .iter()
            .rposition(|kind| matches!(kind, super::super::decoder_layer::Qwen3_5LayerKind::Linear))
            .expect("a linear layer");
        let mut driver = LaneDriver::new(&plans);
        for lane in 0..plans.len() {
            driver.prefill(&mut inner, lane)?;
        }
        for _ in 0..2 {
            assert!(driver.wave(&mut inner)?);
        }
        let frontiers: Vec<u32> = seqs.iter().map(|&seq| cursor_of(&inner, seq)).collect();
        let mut before = Vec::new();
        for &seq in &seqs {
            HybridSchedulerBackend::activate_paged_seq(&mut inner, seq)?;
            before.push(cell_bits(inner.caches.as_ref().expect("caches"))?);
        }
        wave_break_state(&mut inner, seqs[5], layer)?;
        HybridSchedulerBackend::activate_paged_seq(&mut inner, seqs[1])?;
        assert!(driver.wave_with(&mut inner, Void::Fail)?);
        let reason = driver.failure.take().unwrap_or_default();
        assert!(
            reason.contains("the complete kernel declined lane 1"),
            "the second wave failed at its last lane: {reason}"
        );
        let failed = driver.waves.last().expect("the failed step");
        assert_eq!((failed.forwards, failed.waves), (0, (2, 6)), "{failed:?}");
        // The retraction puts the first wave's lanes back on their cells, the
        // first row last.
        assert_eq!(inner.active_scheduled_seq, Some(seqs[0]));
        assert!(inner.caches.is_some());
        assert!(inner.dflash2_states.lane_batch.is_none());
        for (index, &seq) in seqs.iter().enumerate() {
            assert_eq!(
                cursor_of(&inner, seq),
                frontiers[index],
                "sequence {seq}: its rows rolled back"
            );
            let lane = &inner.dflash2_states.seq_lanes[&seq];
            assert!(lane.open.is_none(), "sequence {seq}: no open verify");
        }
        for (index, &seq) in seqs.iter().enumerate() {
            HybridSchedulerBackend::activate_paged_seq(&mut inner, seq)?;
            assert!(
                cell_bits(inner.caches.as_ref().expect("caches"))? == before[index],
                "sequence {seq}: its slots are its cells before the step"
            );
        }
        let failed_at = driver.waves.len() - 1;
        while driver.wave(&mut inner)? {}
        let (runs, waves) = driver.finish(&mut inner)?;
        for (lane, plan) in plans.iter().enumerate() {
            assert_lane_run(
                &runs[lane],
                &alone[lane],
                &format!("after the failure: lane {}", plan.seq),
            );
        }
        assert_eq!(
            (waves[failed_at + 1].forwards, waves[failed_at + 1].waves),
            (0, (2, 6)),
            "every owner reruns in two waves: {waves:?}"
        );
        Ok(())
    }

    /// Two owners with EQUAL prompts, lengths and schedules: the second
    /// resumes on the first one's blocks, and both verify the same ids at
    /// the same depth in every wave. Each equals the per-owner run, the
    /// first its whole turn alone, and the second its whole turn after a
    /// whole turn of the same prompt.
    #[test]
    fn two_lanes_on_equal_prompts() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("two_lanes_on_equal_prompts") {
            return Ok(());
        }
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let base = || wide_draft_inner(seed, 40);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            let body = prompt(40);
            let n = 40;
            let (reference, _) = paged_ar(&mut base(), &body, n)?;
            let plans: Vec<LanePlan> = [1u32, 2]
                .iter()
                .map(|&seq| LanePlan {
                    seq,
                    prompt: body.clone(),
                    n,
                    temperature: 0.0,
                    reference: reference.clone(),
                    schedule: vec![3, 0, 7, 1, 5],
                })
                .collect();
            let first_alone = whole_turn_lane(&mut base(), &plans[0])?;
            let mut whole = base();
            whole_turn(
                &mut whole,
                &body,
                DecoderPlan::Speculative(SpeculativeKind::DraftModel),
            )?;
            let second_alone = whole_turn_lane(&mut whole, &plans[1])?;
            let [(runs, waves, states), (owner_runs, _, owner_states)] = wave_and_per_owner_runs(
                &plans,
                |waves| {
                    let mut inner = base();
                    inner.grant_scheduled_lanes(4);
                    inner.dflash2_wave_ready = waves;
                    inner
                },
                |driver, inner| {
                    driver.prefill(inner, 0)?;
                    driver.prefill(inner, 1)?;
                    while driver.wave(inner)? {}
                    Ok(())
                },
            )?;
            println!(
                "equal prompts: resumed {} / {}, alone {} / {}",
                runs[0].resumed, runs[1].resumed, first_alone.resumed, second_alone.resumed
            );
            assert!(
                runs[1].resumed > 0,
                "the second owner resumes on the first one's blocks"
            );
            for lane in 0..2 {
                assert_lane_run(
                    &runs[lane],
                    &owner_runs[lane],
                    &format!("lane {} against per-owner", lane + 1),
                );
            }
            assert_states(&states, &owner_states, "waved against per-owner");
            assert_lane_run(&runs[0], &first_alone, "lane 1 against alone");
            assert_lane_run(&runs[1], &second_alone, "lane 2 against alone");
            let shared = waves.iter().filter(|wave| wave.rows.len() == 2).count();
            assert!(shared >= 6, "shared steps: {waves:?}");
            for (step, wave) in waves.iter().enumerate() {
                assert_eq!(
                    (wave.forwards, wave.waves),
                    wave_step(wave.rows.len()),
                    "step {step}: {waves:?}"
                );
            }
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// Sampled lanes (temperature 1) in waves of three: each lane's draft
    /// samples from its own rng and its own ring, so its proposals, every
    /// cycle's logits bits, keep and ring, its tokens and cells are its
    /// whole turn's alone from the same seed, and the per-owner run's.
    #[test]
    fn sampled_lanes_in_a_wave_draft_from_their_own_rng() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "sampled_lanes_in_a_wave_draft_from_their_own_rng",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8300;
        let plans = wave_plans(
            seed,
            1.0,
            &[
                (40, 32, (0..=DFLASH2_DRAFT_PROPOSALS).collect()),
                (40, 32, (0..=DFLASH2_DRAFT_PROPOSALS).rev().collect()),
                (56, 32, vec![1, 4, 0, 2]),
            ],
        )?;
        let alone = wave_alone(seed, &plans)?;
        let [(runs, waves, states), (owner_runs, _, owner_states)] = wave_and_per_owner_runs(
            &plans,
            |waves| lanes_inner(seed, waves, 4),
            |driver, inner| {
                for lane in 0..3 {
                    driver.prefill(inner, lane)?;
                }
                while driver.wave(inner)? {}
                Ok(())
            },
        )?;
        for (lane, plan) in plans.iter().enumerate() {
            assert!(
                alone[lane].proposals.len() >= 3,
                "lane {}: the lane drafts: {}",
                plan.seq,
                alone[lane].proposals.len()
            );
            assert_lane_run(
                &runs[lane],
                &alone[lane],
                &format!("lane {} against alone", plan.seq),
            );
            assert_lane_run(
                &runs[lane],
                &owner_runs[lane],
                &format!("lane {} against per-owner", plan.seq),
            );
        }
        assert_states(&states, &owner_states, "waved against per-owner");
        assert!(
            alone[0].proposals != alone[1].proposals,
            "the lanes draft unlike proposals"
        );
        let threes = waves
            .iter()
            .filter(|wave| (wave.rows.len(), wave.forwards, wave.waves) == (3, 0, (1, 3)))
            .count();
        assert!(threes >= 3, "waves of three: {waves:?}");
        Ok(())
    }

    /// [`scheduler_on`] on a wave-ready load with `lanes` granted.
    fn wave_scheduler_on(
        mut inner: Qwen35Inner,
        lanes: usize,
    ) -> Result<crate::engine::hybrid_scheduler::HybridSchedulerState<Qwen35Inner>> {
        use crate::engine::governor::governor::Governor;
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        inner.tokenizer = Some(super::tests::content_tokenizer());
        inner.grant_scheduled_lanes(lanes);
        inner.dflash2_wave_ready = true;
        let io = FakeIo::new(
            stats(0),
            host(32 * G, crate::engine::governor::host::Pressure::Normal),
        );
        let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
        inner.memory_registration =
            Some(governor.register_model(model(16 * G, 0)).expect("register"));
        let mut state = crate::engine::hybrid_scheduler::HybridSchedulerState::new(inner)?;
        state.enable_continuous_batching();
        Ok(state)
    }

    /// The verify waves since `last`, by their lane count, added to `sizes`.
    fn count_waves(last: &mut (u64, u64), sizes: &mut std::collections::BTreeMap<u64, u64>) {
        let now = verify_waves();
        let (waves, lanes) = (now.0 - last.0, now.1 - last.1);
        if waves > 0 {
            *sizes.entry(lanes / waves).or_insert(0) += waves;
        }
        *last = now;
    }

    /// Three sessions behind the model thread's scheduler on a wave-ready
    /// load (`execute_speculative_rows` end to end): they verify in waves of
    /// three; one is cancelled while all three decode; the two that stay
    /// verify in waves of two and give their whole turns' tokens alone.
    #[test]
    fn three_sessions_wave_and_one_is_cancelled() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "three_sessions_wave_and_one_is_cancelled",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let bodies = [wave_prompt(1, 40), wave_prompt(2, 40), wave_prompt(3, 40)];
        let n = 40;
        let (a_alone, _) = whole_turn_session(seed, &bodies[0], n)?;
        let (c_alone, _) = whole_turn_session(seed, &bodies[2], n)?;
        let mut state = wave_scheduler_on(tiny_paged_int8_dflash_inner(seed), 4)?;
        let mut last = verify_waves();
        let mut sizes = std::collections::BTreeMap::new();
        let mut cancelled_at = None;
        let forwards = verify_forwards();
        let (mut results, saw_lane) = drive_sessions(
            &mut state,
            &[
                ("a", &bodies[0], n),
                ("b", &bodies[1], n),
                ("c", &bodies[2], n),
            ],
            true,
            |state, cancels| {
                count_waves(&mut last, &mut sizes);
                let lanes = state.inner.dflash2_states.seq_lanes.len();
                let Some(&seq) = state.owner_sequences.get("b") else {
                    return;
                };
                let generated = state
                    .turn_mut(seq)
                    .map_or(0, |turn| turn.payload.generated_tokens.len());
                if cancelled_at.is_none()
                    && generated >= 2
                    && lanes == 3
                    && sizes.get(&3).is_some_and(|&waves| waves >= 2)
                {
                    cancels[1].store(true, std::sync::atomic::Ordering::Relaxed);
                    cancelled_at = Some(generated);
                }
            },
        )?;
        let forwards = verify_forwards() - forwards;
        println!("three sessions: waves by lanes {sizes:?}, lane-kernel forwards {forwards}");
        assert!(saw_lane);
        assert!(
            cancelled_at.is_some(),
            "b was cancelled while three lanes waved: {sizes:?}"
        );
        assert!(
            sizes.get(&2).is_some_and(|&waves| waves >= 2),
            "the two that stay wave on: {sizes:?}"
        );
        let c = results.pop().expect("c's reply")?;
        let cancelled = results.pop().expect("b's reply");
        assert!(
            cancelled.is_err_and(|error| error.reason == "chat session cancelled"),
            "b ended cancelled"
        );
        let a = results.pop().expect("a's reply")?;
        for (tag, result, alone) in [("a", &a, &a_alone), ("c", &c, &c_alone)] {
            assert!(drafted(result) > 0, "{tag} drafted");
            assert_eq!(result.text, alone.text, "{tag}: its whole turn's tokens");
            assert_eq!(drafted(result), drafted(alone), "{tag}: its cycles");
            assert_eq!(accepted(result), accepted(alone), "{tag}: its accepted");
        }
        assert!(state.inner.dflash2_states.seq_lanes.is_empty());
        assert!(state.inner.dflash2_states.lane_batch.is_none());
        assert!(state.inner.dflash2_states.lane_decode_limit.is_none());
        Ok(())
    }

    /// A lane preempted while it and its peer verify in waves replays,
    /// begins its lane again and verifies in waves again (its cells packed
    /// inside its first wave); both sessions give their whole turns' tokens
    /// alone (a's 19 tokens end before its first bf16 near-tie, as in
    /// `a_preempted_dflash2_lane_replays_and_drafts_again`).
    #[test]
    fn a_preempted_session_rejoins_the_waves() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("a_preempted_session_rejoins_the_waves")
            || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let [(_, first), (_, second)] = lane_prompts();
        let (a_n, b_n) = (19, 40);
        let (a_alone, _) = whole_turn_session(seed, &first, a_n)?;
        let (b_alone, _) = whole_turn_session(seed, &second, b_n)?;
        let mut state = wave_scheduler_on(tiny_paged_int8_dflash_inner(seed), 2)?;
        let mut last = verify_waves();
        let mut sizes = std::collections::BTreeMap::new();
        let mut preempted_at = None;
        let mut waves_before = 0;
        let mut waves_after = 0;
        let mut replay_first = None;
        let (mut results, saw_lane) = drive_sessions(
            &mut state,
            &[("a", &first, a_n), ("b", &second, b_n)],
            true,
            |state, _| {
                let before: u64 = sizes.values().sum();
                count_waves(&mut last, &mut sizes);
                let formed = sizes.values().sum::<u64>() - before;
                let (Some(&a), Some(&b)) = (
                    state.owner_sequences.get("a"),
                    state.owner_sequences.get("b"),
                ) else {
                    return;
                };
                if preempted_at.is_some() {
                    waves_after += formed;
                    if replay_first.is_none()
                        && let Some(lane) = state.inner.dflash2_states.seq_lanes.get(&a)
                        && lane.verifies != LaneVerifies::default()
                    {
                        replay_first = Some((
                            lane.verifies,
                            state.inner.dflash2_states.seq_lanes.contains_key(&b),
                        ));
                    }
                    return;
                }
                waves_before += formed;
                let packed =
                    |state: &crate::engine::hybrid_scheduler::HybridSchedulerState<Qwen35Inner>,
                     seq: u32| {
                        state
                            .inner
                            .dflash2_states
                            .seq_lanes
                            .get(&seq)
                            .is_some_and(|lane| lane.gdn.is_some())
                    };
                let generated = state
                    .turn_mut(a)
                    .map_or(0, |turn| turn.payload.generated_tokens.len());
                if generated >= 8 && packed(state, a) && packed(state, b) && waves_before >= 2 {
                    assert!(state.preempt_running(a), "a is preempted");
                    assert!(
                        !state.inner.dflash2_states.seq_lanes.contains_key(&a),
                        "the preemption ended a's lane"
                    );
                    state.resume_preempted();
                    preempted_at = Some(generated);
                }
            },
        )?;
        println!(
            "preempted session: waves by lanes {sizes:?}, before {waves_before}, after {waves_after}"
        );
        assert!(saw_lane);
        assert!(
            preempted_at.is_some(),
            "a was preempted while both waved: {sizes:?}"
        );
        assert_eq!(state.preemptions(), 1);
        assert!(
            waves_after > 0,
            "a's replayed lane verified in a wave again"
        );
        let (first_cycle, beside_b) = replay_first.expect("a's replayed lane committed a cycle");
        assert!(beside_b, "b decoded beside a's replay");
        assert_eq!(
            first_cycle,
            LaneVerifies {
                waves: [1, 0, 0],
                alone: 0,
            },
            "the replay's first cycle is in a wave (its cells packed there)"
        );
        let b = results.pop().expect("b's reply")?;
        let a = results.pop().expect("a's reply")?;
        assert!(drafted(&a) > 0 && drafted(&b) > 0);
        assert_eq!(a.text, a_alone.text, "a's whole turn's tokens");
        assert_eq!(b.text, b_alone.text, "b's whole turn's tokens");
        assert!(state.inner.dflash2_states.seq_lanes.is_empty());
        assert!(state.inner.dflash2_states.lane_batch.is_none());
        Ok(())
    }

    /// Two DFlash2 sessions and one session with its draft off on a
    /// wave-ready load: the passes in which the autoregressive session
    /// decodes beside both lanes form a wave of the two lanes; once it ends
    /// the two lanes go on in waves; each session gives its tokens alone.
    #[test]
    fn an_ar_session_beside_two_waving_sessions() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "an_ar_session_beside_two_waving_sessions",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let bodies = [wave_prompt(1, 40), wave_prompt(2, 40), wave_prompt(3, 40)];
        let (n, peer_n) = (40, 10);
        let (a_alone, _) = whole_turn_session(seed, &bodies[0], n)?;
        let (c_alone, _) = whole_turn_session(seed, &bodies[2], n)?;
        let b_alone = {
            let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
            let (mut results, saw_lane) =
                drive_sessions(&mut state, &[("b", &bodies[1], peer_n)], false, |_, _| {})?;
            assert!(!saw_lane);
            results.remove(0)?
        };
        let mut state = wave_scheduler_on(tiny_paged_int8_dflash_inner(seed), 4)?;
        let mut last = verify_waves();
        let mut sizes = std::collections::BTreeMap::new();
        let mut generated = 0;
        let (mut beside, mut waved_beside) = (0, 0);
        let (mut results, saw_lane) = drive_sessions_with(
            &mut state,
            &[
                ("a", &bodies[0], n, true),
                ("b", &bodies[1], peer_n, false),
                ("c", &bodies[2], n, true),
            ],
            |state, _| {
                let before: u64 = sizes.values().sum();
                count_waves(&mut last, &mut sizes);
                let formed = sizes.values().sum::<u64>() - before;
                let Some(&seq) = state.owner_sequences.get("b") else {
                    return;
                };
                let now = state
                    .turn_mut(seq)
                    .map_or(generated, |turn| turn.payload.generated_tokens.len());
                if now > generated && state.inner.dflash2_states.seq_lanes.len() == 2 {
                    beside += 1;
                    waved_beside += usize::from(formed > 0);
                }
                generated = now;
            },
        )?;
        println!(
            "ar beside: waves by lanes {sizes:?}, peer steps beside two lanes {beside}, waves in them {waved_beside}"
        );
        assert!(saw_lane);
        assert!(beside > 2, "b decoded beside both lanes");
        assert!(
            waved_beside > 2 && waved_beside + 2 >= beside,
            "the steps the peer rides form a wave: {waved_beside} of {beside}"
        );
        assert!(
            sizes.get(&2).is_some_and(|&waves| waves >= 2),
            "the lanes wave once the peer ends: {sizes:?}"
        );
        let c = results.pop().expect("c's reply")?;
        let b = results.pop().expect("b's reply")?;
        let a = results.pop().expect("a's reply")?;
        assert_eq!(a.text, a_alone.text, "a's whole turn's tokens");
        assert_eq!(c.text, c_alone.text, "c's whole turn's tokens");
        assert_eq!(b.text, b_alone.text, "b's tokens alone");
        assert!(drafted(&a) > 0 && drafted(&c) > 0);
        assert!(state.inner.dflash2_states.seq_lanes.is_empty());
        assert!(state.inner.dflash2_states.peer_cells.is_empty());
        assert!(state.inner.dflash2_states.lane_decode_limit.is_none());
        Ok(())
    }

    /// A wave fails behind the model thread's scheduler (one session's
    /// recurrent state cast to bf16 while both verify in waves): every
    /// session of the step ends with the verify's error, no lane, verify
    /// batch or decode cache limit stays behind, and the next session on
    /// the same load verifies and gives its whole turn's tokens alone.
    #[test]
    fn a_failed_wave_fails_its_sessions_and_the_next_session_is_exact() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_failed_wave_fails_its_sessions_and_the_next_session_is_exact",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let [(_, first), (_, second)] = lane_prompts();
        let third = wave_prompt(3, 40);
        let (n, next_n) = (40, 24);
        let (next_alone, _) = whole_turn_session(seed, &third, next_n)?;
        let mut state = wave_scheduler_on(tiny_paged_int8_dflash_inner(seed), 4)?;
        let layer = state
            .inner
            .layer_kinds
            .iter()
            .position(|kind| matches!(kind, super::super::decoder_layer::Qwen3_5LayerKind::Linear))
            .expect("a linear layer");
        let mut last = verify_waves();
        let mut sizes = std::collections::BTreeMap::new();
        let mut broken_at = None;
        let (results, saw_lane) = drive_sessions(
            &mut state,
            &[("a", &first, n), ("b", &second, n)],
            true,
            |state, _| {
                count_waves(&mut last, &mut sizes);
                if broken_at.is_some() || sizes.get(&2).is_none_or(|&waves| waves < 2) {
                    return;
                }
                let (Some(&a), Some(&b)) = (
                    state.owner_sequences.get("a"),
                    state.owner_sequences.get("b"),
                ) else {
                    return;
                };
                let packed = [a, b].iter().all(|seq| {
                    state
                        .inner
                        .dflash2_states
                        .seq_lanes
                        .get(seq)
                        .is_some_and(|lane| lane.gdn.is_some())
                });
                let room = [a, b].iter().all(|&seq| {
                    state
                        .turn_mut(seq)
                        .is_some_and(|turn| turn.payload.generated_tokens.len() + 12 < n as usize)
                });
                if packed && room {
                    wave_break_state(&mut state.inner, b, layer).expect("b's state cast");
                    broken_at = Some(sizes.clone());
                }
            },
        )?;
        println!("failed wave: waves by lanes {sizes:?}, broken at {broken_at:?}");
        assert!(saw_lane);
        assert!(broken_at.is_some(), "b's state was cast while both waved");
        for (tag, result) in ["a", "b"].iter().zip(results) {
            let reason = result.err().map(|error| error.reason);
            println!("failed wave: {tag} ended with {reason:?}");
            assert!(
                reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains("the complete kernel declined lane")),
                "{tag} ends with the wave's error: {reason:?}"
            );
        }
        assert!(state.inner.dflash2_states.seq_lanes.is_empty());
        assert!(state.inner.dflash2_states.lane_batch.is_none());
        assert!(state.inner.dflash2_states.peer_cells.is_empty());
        assert!(state.inner.dflash2_states.lane_decode_limit.is_none());

        let before = verify_forwards();
        let (mut results, saw_lane) =
            drive_sessions(&mut state, &[("c", &third, next_n)], true, |_, _| {})?;
        let next = results.remove(0)?;
        assert!(saw_lane, "the next session ran on a lane");
        assert!(verify_forwards() > before, "the next session verified");
        assert_eq!(next.text, next_alone.text, "its whole turn's tokens");
        assert_eq!(drafted(&next), drafted(&next_alone), "its cycles");
        assert!(state.inner.dflash2_states.seq_lanes.is_empty());
        Ok(())
    }

    /// Owner `k`'s prompt of `len` tokens: no first block in common with
    /// another owner's, so each prefills from 0 as it does alone.
    fn owner_prompt(k: usize, len: usize) -> Vec<u32> {
        let (mul, add) = [
            (7, 3),
            (5, 1),
            (3, 2),
            (11, 5),
            (13, 7),
            (9, 4),
            (15, 6),
            (1, 0),
            (7, 9),
        ][k % 9];
        (0..len as u32).map(|i| (i * mul + add) % VOCAB).collect()
    }

    /// One owner is cancelled while three sessions verify in waves of three
    /// behind the scheduler: between two steps (the step's cancel snapshot
    /// sees it), or inside a step of a wave (after the snapshot: its commit
    /// is clamped at its first token). The two that stay go on in waves and
    /// give their whole turns' tokens, cycles and accepted drafts alone.
    #[test]
    fn a_cancel_between_or_inside_waved_steps_leaves_the_others_exact() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_cancel_between_or_inside_waved_steps_leaves_the_others_exact",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let bodies = [
            owner_prompt(0, 40),
            owner_prompt(1, 40),
            owner_prompt(2, 40),
        ];
        let n = 40;
        let (a_alone, _) = whole_turn_session(seed, &bodies[0], n)?;
        let (c_alone, _) = whole_turn_session(seed, &bodies[2], n)?;
        for inside in [false, true] {
            let tag = if inside {
                "inside a step"
            } else {
                "between steps"
            };
            let mut state = wave_scheduler_on(tiny_paged_int8_dflash_inner(seed), 4)?;
            let mut last = verify_waves();
            let mut sizes = std::collections::BTreeMap::new();
            let mut cancelled_at = None;
            let mut cancel_step = None;
            let (mut results, saw_lane) = drive_sessions(
                &mut state,
                &[
                    ("a", &bodies[0], n),
                    ("b", &bodies[1], n),
                    ("c", &bodies[2], n),
                ],
                true,
                |state, cancels| {
                    count_waves(&mut last, &mut sizes);
                    let lanes = state.inner.dflash2_states.seq_lanes.len();
                    let Some(&seq) = state.owner_sequences.get("b") else {
                        return;
                    };
                    let generated = state
                        .turn_mut(seq)
                        .map_or(0, |turn| turn.payload.generated_tokens.len());
                    if cancelled_at.is_some()
                        || generated < 2
                        || lanes != 3
                        || sizes.get(&3).is_none_or(|&waves| waves < 2)
                    {
                        return;
                    }
                    cancelled_at = Some(generated);
                    if !inside {
                        cancels[1].store(true, std::sync::atomic::Ordering::Relaxed);
                        return;
                    }
                    let flag = std::sync::Arc::clone(&cancels[1]);
                    let before = verify_waves();
                    assert!(
                        state.step_after_snapshot(|| {
                            flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        }),
                        "the step ran"
                    );
                    let after = verify_waves();
                    cancel_step = Some((after.0 - before.0, after.1 - before.1));
                    last = after;
                },
            )?;
            println!("cancel {tag}: waves by lanes {sizes:?}, the cancel's step {cancel_step:?}");
            assert!(saw_lane);
            assert!(
                cancelled_at.is_some(),
                "{tag}: b was cancelled while three lanes waved"
            );
            if inside {
                assert_eq!(
                    cancel_step,
                    Some((1, 3)),
                    "{tag}: the cancel's step is one wave of three"
                );
            }
            assert!(
                sizes.get(&2).is_some_and(|&waves| waves >= 2),
                "{tag}: the two that stay wave on: {sizes:?}"
            );
            let c = results.pop().expect("c's reply")?;
            let cancelled = results.pop().expect("b's reply");
            match &cancelled {
                Ok(result) => assert_eq!(result.finish_reason, "cancelled", "{tag}: b"),
                Err(error) => assert_eq!(error.reason, "chat session cancelled", "{tag}: b"),
            }
            let a = results.pop().expect("a's reply")?;
            for (owner, result, alone) in [("a", &a, &a_alone), ("c", &c, &c_alone)] {
                assert!(drafted(result) > 0, "{tag}: {owner} drafted");
                assert_eq!(
                    result.text, alone.text,
                    "{tag}: {owner}: its whole turn's tokens"
                );
                assert_eq!(
                    drafted(result),
                    drafted(alone),
                    "{tag}: {owner}: its cycles"
                );
                assert_eq!(
                    accepted(result),
                    accepted(alone),
                    "{tag}: {owner}: its accepted"
                );
            }
            lane_sessions::clean(&state, tag);
        }
        Ok(())
    }

    /// 5, 6 and 9 owners behind the scheduler with as many lanes granted,
    /// at unlike prompt and output lengths: every step verifies its lanes in
    /// groups of at most four (5 = 4 + a lane kernel, 6 = 4 + 2, 8 = 4 + 4;
    /// the scheduler runs at most 8 sequences, so the ninth owner waits and
    /// nine never verify in one step); owners end in steps where others wave
    /// on. Every owner's reply is its whole turn's alone (tokens, cycles,
    /// accepted), and every owner's parked cell is its whole turn's settled
    /// cell.
    #[test]
    fn five_six_and_nine_owners_wave_in_groups_of_at_most_four() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "five_six_and_nine_owners_wave_in_groups_of_at_most_four",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let lens = [40usize, 24, 56, 32, 48, 40, 24, 64, 32];
        let ns = [44i32, 40, 40, 36, 36, 32, 28, 24, 20];
        let bodies: Vec<Vec<u32>> = (0..9).map(|k| owner_prompt(k, lens[k])).collect();
        let alone = (0..9)
            .map(|k| whole_turn_session(seed, &bodies[k], ns[k]))
            .collect::<Result<Vec<_>>>()?;
        let names: Vec<String> = (0..9).map(|k| format!("o{k}")).collect();
        for (owners, full) in [(5usize, (1u64, 4u64)), (6, (2, 6)), (9, (2, 8))] {
            let mut state = wave_scheduler_on(tiny_paged_int8_dflash_inner(seed), owners)?;
            let starts: Vec<(&str, &[u32], i32)> = (0..owners)
                .map(|k| (names[k].as_str(), bodies[k].as_slice(), ns[k]))
                .collect();
            let mut last = verify_waves();
            let mut steps = std::collections::BTreeSet::new();
            let mut live = std::collections::BTreeSet::new();
            let mut ended_beside_waves = 0;
            let (results, saw_lane) = drive_sessions(&mut state, &starts, true, |state, _| {
                let now = verify_waves();
                let step = (now.0 - last.0, now.1 - last.1);
                last = now;
                steps.insert(step);
                let lanes: std::collections::BTreeSet<u32> = state
                    .inner
                    .dflash2_states
                    .seq_lanes
                    .keys()
                    .copied()
                    .collect();
                if step.0 > 0 && !lanes.is_empty() && live.difference(&lanes).next().is_some() {
                    ended_beside_waves += 1;
                }
                live = lanes;
            })?;
            println!(
                "{owners} owners: (waves, lanes in waves) per pass {steps:?}, owners ended beside waves in {ended_beside_waves} passes"
            );
            assert!(saw_lane);
            assert!(
                steps.iter().all(|&(waves, lanes)| lanes <= 4 * waves),
                "{owners} owners: no wave of more than four lanes: {steps:?}"
            );
            assert!(
                steps.contains(&full),
                "{owners} owners: a step of every owner in groups of at most four: {steps:?}"
            );
            assert!(
                ended_beside_waves > 0,
                "{owners} owners: an owner ended in a step where others waved on"
            );
            for (k, result) in results.into_iter().enumerate() {
                let result = result?;
                let (whole, _) = &alone[k];
                let tag = format!("{owners} owners: o{k}");
                assert!(drafted(&result) > 0, "{tag}: drafted");
                assert_eq!(result.text, whole.text, "{tag}: its whole turn's tokens");
                assert_eq!(drafted(&result), drafted(whole), "{tag}: its cycles");
                assert_eq!(accepted(&result), accepted(whole), "{tag}: its accepted");
            }
            let mut parked = 0;
            for k in 0..owners {
                let seq = state.owner_sequences[&names[k]];
                if !HybridSchedulerBackend::has_scheduled_recurrent(&state.inner, seq) {
                    continue;
                }
                HybridSchedulerBackend::activate_scheduled_recurrent(&mut state.inner, seq)?;
                assert!(
                    cell_bits(state.inner.caches.as_ref().expect("caches"))? == alone[k].1.1,
                    "{owners} owners: o{k}: the parked cell is its whole turn's settled cell"
                );
                parked += 1;
            }
            assert!(
                parked >= owners - 1,
                "{owners} owners: parked cells {parked}"
            );
            lane_sessions::clean(&state, "groups of at most four");
        }
        Ok(())
    }

    /// Two thinking sessions on a wave-ready load, each cut at its thinking
    /// budget while both verify in waves: the pass that closes each thought
    /// formed a wave of the two lanes, and each reply (thought, close, text,
    /// cycles, accepted drafts) is its whole turn's alone.
    #[test]
    fn thinking_cuts_ride_a_wave() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("thinking_cuts_ride_a_wave")
            || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        use lane_sessions::{clean, run, start_cmd, take};
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let [(_, first), (_, second)] = lane_prompts();
        let n = 24;
        let thinking = |budget: i32| {
            move |config: &mut crate::engine::types::ChatConfig| {
                config.reasoning_effort = Some("high".to_string());
                config.thinking_token_budget = Some(budget);
            }
        };
        let state_on = || -> Result<lane_sessions::State> {
            let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
            state.inner.tokenizer = Some(super::tests::think_tokenizer());
            Ok(state)
        };
        let budgets = [("a", &first, 6), ("b", &second, 9)];
        let mut alone = Vec::new();
        for &(owner, body, budget) in &budgets {
            let mut state = state_on()?;
            state.serve_whole_turns();
            let (command, mut reply, _) = start_cmd(owner, body, n, thinking(budget))?;
            let mut done = None;
            run(&mut state, vec![command], false, |_, _| {
                take(&mut reply, &mut done);
                Ok(done.is_some())
            })?;
            alone.push(done.expect("whole turn")?);
        }
        let mut state = state_on()?;
        state.inner.dflash2_wave_ready = true;
        let mut commands = Vec::new();
        let mut replies = Vec::new();
        for &(owner, body, budget) in &budgets {
            let (command, reply, _) = start_cmd(owner, body, n, thinking(budget))?;
            commands.push(command);
            replies.push(reply);
        }
        let mut done = [None, None];
        let mut last = verify_waves();
        let mut closed = [None, None];
        run(&mut state, commands, false, |state, _| {
            let now = verify_waves();
            let step = (now.0 - last.0, now.1 - last.1);
            last = now;
            for (index, &(owner, _, _)) in budgets.iter().enumerate() {
                if closed[index].is_some() {
                    continue;
                }
                let Some(&seq) = state.owner_sequences.get(owner) else {
                    continue;
                };
                if state
                    .turn_mut(seq)
                    .is_some_and(|turn| turn.payload.reasoning_tracker.outcome().cut_at.is_some())
                {
                    closed[index] = Some(step);
                }
            }
            for (reply, slot) in replies.iter_mut().zip(done.iter_mut()) {
                take(reply, slot);
            }
            Ok(done.iter().all(Option::is_some))
        })?;
        println!("thinking cuts: the closing passes' (waves, lanes in waves) {closed:?}");
        for ((owner, _, budget), ((slot, whole), close)) in
            budgets.iter().zip(done.into_iter().zip(&alone).zip(closed))
        {
            let lane = slot.expect("reply")?;
            for result in [&lane, whole] {
                assert_eq!(
                    result.reasoning_close.as_deref(),
                    Some("budget"),
                    "{owner}: {}",
                    result.raw_text
                );
                assert_eq!(result.reasoning_cut_at, Some(*budget as u32), "{owner}");
            }
            assert_eq!(
                close,
                Some((1, 2)),
                "{owner}: the cut rode a wave of both lanes"
            );
            assert!(drafted(&lane) > 0, "{owner}: drafted");
            assert_eq!(
                lane.raw_text, whole.raw_text,
                "{owner}: the whole turn's tokens"
            );
            assert_eq!(lane.thinking, whole.thinking, "{owner}: thought");
            assert_eq!(lane.text, whole.text, "{owner}: text");
            assert_eq!(drafted(&lane), drafted(whole), "{owner}: cycles");
            assert_eq!(accepted(&lane), accepted(whole), "{owner}: accepted drafts");
        }
        clean(&state, "thinking cut in a wave");
        Ok(())
    }

    /// A load that is not wave-ready (the tiny fixture's projections are
    /// off the lane tier) says so in its load line, with the first layer
    /// that fails; two sessions behind the scheduler on it verify per owner
    /// (no wave, lane-kernel verifies) and give their whole turns' tokens
    /// alone.
    #[test]
    fn a_load_that_is_not_wave_ready_says_so_and_verifies_per_owner() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_load_that_is_not_wave_ready_says_so_and_verifies_per_owner",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        use crate::models::qwen3_5::scheduled_dflash2::{wave_load_line, wave_ready_at_load};
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let [(_, first), (_, second)] = lane_prompts();
        let n = 24;
        let (a_alone, _) = whole_turn_session(seed, &first, n)?;
        let (b_alone, _) = whole_turn_session(seed, &second, n)?;
        let inner = tiny_paged_int8_dflash_inner(seed);
        let (ready, line) = wave_load_line(&inner);
        println!("load line: {line}");
        assert!(!ready, "not wave-ready: {line}");
        assert!(
            line.starts_with(
                "Qwen3.8 DFlash2 lanes: verify waves off, each lane verifies alone (layer 0: "
            ),
            "{line}"
        );
        let mut state = scheduler_on(inner)?;
        state.inner.dflash2_wave_ready = wave_ready_at_load(&state.inner);
        assert!(!state.inner.dflash2_wave_ready);
        let (waves, forwards) = (verify_waves(), verify_forwards());
        let mut most = 0;
        let (mut results, saw_lane) = drive_sessions(
            &mut state,
            &[("a", &first, n), ("b", &second, n)],
            true,
            |state, _| {
                most = most.max(state.inner.dflash2_states.seq_lanes.len());
            },
        )?;
        assert!(saw_lane);
        assert_eq!(most, 2, "two live lanes");
        assert_eq!(verify_waves(), waves, "no verify wave");
        assert!(
            verify_forwards() > forwards,
            "the lanes verified on their lane kernels"
        );
        let b = results.pop().expect("b's reply")?;
        let a = results.pop().expect("a's reply")?;
        assert_eq!(a.text, a_alone.text, "a's whole turn's tokens");
        assert_eq!(b.text, b_alone.text, "b's whole turn's tokens");
        assert_eq!(drafted(&a), drafted(&a_alone), "a's cycles");
        assert_eq!(drafted(&b), drafted(&b_alone), "b's cycles");
        lane_sessions::clean(&state, "not wave-ready");
        Ok(())
    }

    /// A fixture behind the model thread's scheduler: continuous batching
    /// on, two lanes granted, a registered memory model.
    fn scheduler_on(
        mut inner: Qwen35Inner,
    ) -> Result<crate::engine::hybrid_scheduler::HybridSchedulerState<Qwen35Inner>> {
        use crate::engine::governor::governor::Governor;
        use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
        inner.tokenizer = Some(super::tests::content_tokenizer());
        inner.grant_scheduled_lanes(2);
        let io = FakeIo::new(
            stats(0),
            host(32 * G, crate::engine::governor::host::Pressure::Normal),
        );
        let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
        inner.memory_registration =
            Some(governor.register_model(model(16 * G, 0)).expect("register"));
        let mut state = crate::engine::hybrid_scheduler::HybridSchedulerState::new(inner)?;
        state.enable_continuous_batching();
        Ok(state)
    }

    /// One greedy `chatSessionStart` per `(owner, body, max_new_tokens)`, its
    /// draft on when `speculate`, all queued before the first pass and
    /// driven until each replies; `on_pass` sees the state and the cancel
    /// flags after every pass. Returns the replies (a cancelled session's is
    /// an error) and whether a pass saw a scheduled DFlash2 lane.
    fn drive_sessions(
        state: &mut crate::engine::hybrid_scheduler::HybridSchedulerState<Qwen35Inner>,
        starts: &[(&str, &[u32], i32)],
        speculate: bool,
        on_pass: impl FnMut(
            &mut crate::engine::hybrid_scheduler::HybridSchedulerState<Qwen35Inner>,
            &[std::sync::Arc<std::sync::atomic::AtomicBool>],
        ),
    ) -> Result<(Vec<Result<crate::engine::types::ChatResult>>, bool)> {
        let starts = starts
            .iter()
            .map(|&(owner, body, max_new_tokens)| (owner, body, max_new_tokens, speculate))
            .collect::<Vec<_>>();
        drive_sessions_with(state, &starts, on_pass)
    }

    /// [`drive_sessions`] with each session's draft on or off.
    #[allow(clippy::type_complexity)]
    fn drive_sessions_with(
        state: &mut crate::engine::hybrid_scheduler::HybridSchedulerState<Qwen35Inner>,
        starts: &[(&str, &[u32], i32, bool)],
        mut on_pass: impl FnMut(
            &mut crate::engine::hybrid_scheduler::HybridSchedulerState<Qwen35Inner>,
            &[std::sync::Arc<std::sync::atomic::AtomicBool>],
        ),
    ) -> Result<(Vec<Result<crate::engine::types::ChatResult>>, bool)> {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        use crate::engine::model_command::ModelCommand;
        use crate::model_thread::LoopControl;
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let mut replies = Vec::new();
        let mut cancels = Vec::new();
        for &(owner, body, max_new_tokens, speculate) in starts {
            let (command, reply, cancel) =
                super::tests::session_start_with(owner, body, max_new_tokens, speculate)?;
            sender
                .send(ModelCommand::<
                    <Qwen35Inner as HybridSchedulerBackend>::FamilyCommand,
                >::from_chat(command))
                .map_err(|_| Error::from_reason("the command channel closed"))?;
            replies.push((reply, None));
            cancels.push(cancel);
        }
        drop(sender);
        let mut saw_lane = false;
        for _ in 0..512 {
            if !matches!(state.drive(&mut receiver), LoopControl::Continue) {
                return Err(Error::from_reason("the scheduler stopped"));
            }
            saw_lane |= !state.inner.dflash2_states.seq_lanes.is_empty();
            on_pass(state, &cancels);
            for (reply, done) in &mut replies {
                if done.is_none()
                    && let Ok(result) = reply.try_recv()
                {
                    *done = Some(result);
                }
            }
            if replies.iter().all(|(_, done)| done.is_some()) {
                let results = replies.into_iter().filter_map(|(_, done)| done).collect();
                return Ok((results, saw_lane));
            }
        }
        Err(Error::from_reason("the sessions did not finish"))
    }

    /// The history checkpoint's tokens and cell bits.
    fn checkpoint_of(inner: &Qwen35Inner) -> Result<(Vec<u32>, Vec<Vec<u32>>)> {
        let (tokens, caches) = inner
            .history_checkpoint_for_test()
            .ok_or_else(|| Error::from_reason("no history checkpoint"))?;
        Ok((tokens.to_vec(), cell_bits(caches)?))
    }

    fn drafted(result: &crate::engine::types::ChatResult) -> u32 {
        result
            .performance
            .as_ref()
            .and_then(|metrics| metrics.mtp_cycles)
            .unwrap_or(0)
    }

    /// `body`'s session alone on a fresh fixture as the whole DFlash2 turn
    /// (`MLX_SERVE_FORCE_SERIAL=1`): its reply, its saved history and the
    /// settled cells its finalize released (the whole turn stores no history
    /// checkpoint).
    #[allow(clippy::type_complexity)]
    fn whole_turn_session(
        seed: u64,
        body: &[u32],
        max_new_tokens: i32,
    ) -> Result<(crate::engine::types::ChatResult, (Vec<u32>, Vec<Vec<u32>>))> {
        whole_turn_on(tiny_paged_int8_dflash_inner(seed), body, max_new_tokens)
    }

    /// [`whole_turn_session`] on `inner`.
    #[allow(clippy::type_complexity)]
    fn whole_turn_on(
        inner: Qwen35Inner,
        body: &[u32],
        max_new_tokens: i32,
    ) -> Result<(crate::engine::types::ChatResult, (Vec<u32>, Vec<Vec<u32>>))> {
        let mut state = scheduler_on(inner)?;
        state.serve_whole_turns();
        let (mut results, saw_lane) =
            drive_sessions(&mut state, &[("a", body, max_new_tokens)], true, |_, _| {})?;
        assert!(
            !saw_lane,
            "a serve-whole-turns draft-load turn is the whole turn"
        );
        let result = results.remove(0)?;
        assert!(drafted(&result) > 0, "the whole turn drafted");
        let cells = crate::models::qwen3_5::dflash2_turn::RELEASED_LANE
            .with(|lane| lane.borrow_mut().take())
            .ok_or_else(|| Error::from_reason("the whole turn released no lane"))?;
        Ok((
            result,
            (state.inner.cached_token_history.clone(), cell_bits(&cells)?),
        ))
    }

    /// The turn's end reads the lane's cells after `scheduled_turn_closed`
    /// folded the last commit's pending rows into them: the history
    /// checkpoint the finalize stores and the cell the owner parks are the
    /// whole DFlash2 turn's settled cells, and the turn ends on the
    /// keep-live branch (its request stays the owner's live continuation,
    /// as a scheduled AR turn on a draft load does).
    #[test]
    fn a_finished_dflash2_lane_settles_before_the_turn_end_reads_it() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_finished_dflash2_lane_settles_before_the_turn_end_reads_it",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let body = prompt(40);
        let n = 24;
        let (alone, alone_checkpoint) = whole_turn_session(seed, &body, n)?;
        {
            let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
            let (mut results, saw_lane) =
                drive_sessions(&mut state, &[("a", &body, n)], true, |_, _| {})?;
            let result = results.remove(0)?;
            assert!(saw_lane, "the turn ran on a scheduled DFlash2 lane");
            assert!(drafted(&result) > 0, "the lane drafted");
            assert_eq!(result.text, alone.text, "the whole turn's tokens");
            assert!(state.inner.dflash2_states.seq_lanes.is_empty());
            assert!(state.inner.dflash2_states.idle_lane.ring.is_some());
            let seq = state.owner_sequences["a"];
            assert!(
                state
                    .inner
                    .paged_adapter
                    .as_ref()
                    .expect("pool")
                    .request_tokens_for(seq)
                    .is_some(),
                "the request stays live for the owner's next turn"
            );
            let checkpoint = checkpoint_of(&state.inner)?;
            assert_eq!(checkpoint.0, alone_checkpoint.0, "checkpoint tokens");
            assert!(
                checkpoint.1 == alone_checkpoint.1,
                "the history checkpoint holds the settled cells"
            );
            HybridSchedulerBackend::activate_scheduled_recurrent(&mut state.inner, seq)?;
            assert!(
                cell_bits(state.inner.caches.as_ref().expect("caches"))? == alone_checkpoint.1,
                "the parked owner cell is the settled one"
            );
            Ok(())
        }
    }

    /// A cancelled session's lane ends alone: its peer, decoding in the same
    /// waves, keeps its lane and gives its whole turn's tokens and settled
    /// cells; the cancelled lane's ring goes idle.
    #[test]
    fn a_cancelled_dflash2_lane_leaves_its_peer_exact() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_cancelled_dflash2_lane_leaves_its_peer_exact",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let [(_, first), (_, second)] = lane_prompts();
        let n = 40;
        let (alone, alone_checkpoint) = whole_turn_session(seed, &first, n)?;
        {
            let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
            let mut cancelled_at = None;
            let mut shared_passes = 0;
            let (mut results, saw_lane) = drive_sessions(
                &mut state,
                &[("a", &first, n), ("b", &second, n)],
                true,
                |state, cancels| {
                    let lanes = state.inner.dflash2_states.seq_lanes.len();
                    if lanes == 2 {
                        shared_passes += 1;
                    }
                    let Some(&seq) = state.owner_sequences.get("b") else {
                        return;
                    };
                    let generated = state
                        .turn_mut(seq)
                        .map_or(0, |turn| turn.payload.generated_tokens.len());
                    if cancelled_at.is_none() && generated >= 2 && lanes == 2 {
                        cancels[1].store(true, std::sync::atomic::Ordering::Relaxed);
                        cancelled_at = Some(generated);
                    }
                },
            )?;
            assert!(saw_lane);
            assert!(cancelled_at.is_some(), "b was cancelled while both decoded");
            assert!(shared_passes > 0, "both lanes lived in one pass");
            let cancelled = results.pop().expect("b's reply");
            assert!(
                cancelled.is_err_and(|error| error.reason == "chat session cancelled"),
                "b ended cancelled"
            );
            let a = results.pop().expect("a's reply")?;
            assert_eq!(a.text, alone.text, "a's whole turn's tokens");
            assert!(drafted(&a) > 0);
            let checkpoint = checkpoint_of(&state.inner)?;
            assert_eq!(checkpoint.0, alone_checkpoint.0, "a ended last");
            assert!(
                checkpoint.1 == alone_checkpoint.1,
                "a's settled cells are its whole turn's"
            );
            assert!(state.inner.dflash2_states.seq_lanes.is_empty());
            assert!(state.inner.dflash2_states.idle_lane.ring.is_some());
            Ok(())
        }
    }

    /// A session with its draft off decodes in the DFlash2 session's waves
    /// and goes on in waves of its own after that session ends: each gives
    /// its tokens alone, the autoregressive one also its history
    /// checkpoint's cells, and no decode cache limit stays behind the last
    /// lane.
    #[test]
    fn an_ar_session_keeps_its_tokens_beside_a_dflash2_session() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "an_ar_session_keeps_its_tokens_beside_a_dflash2_session",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let [(_, first), (_, second)] = lane_prompts();
        let (a_n, b_n) = (24, 40);
        let (a_alone, _) = whole_turn_session(seed, &first, a_n)?;
        let (b_alone, b_checkpoint) = {
            let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
            let (mut results, saw_lane) =
                drive_sessions(&mut state, &[("b", &second, b_n)], false, |_, _| {})?;
            assert!(!saw_lane);
            (results.remove(0)?, checkpoint_of(&state.inner)?)
        };
        {
            let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
            let mut generated = 0;
            let (mut rode, mut own) = (0, 0);
            let (mut results, saw_lane) = drive_sessions_with(
                &mut state,
                &[("a", &first, a_n, true), ("b", &second, b_n, false)],
                |state, _| {
                    let Some(&seq) = state.owner_sequences.get("b") else {
                        return;
                    };
                    let now = state
                        .turn_mut(seq)
                        .map_or(0, |turn| turn.payload.generated_tokens.len());
                    if now > generated {
                        if state.inner.dflash2_states.seq_lanes.is_empty() {
                            own += 1;
                        } else {
                            rode += 1;
                        }
                    }
                    generated = now;
                },
            )?;
            assert!(saw_lane);
            assert!(rode > 2, "b decoded while a's lane lived");
            assert!(own > 0, "b decoded after a's lane ended");
            let b = results.pop().expect("b's reply")?;
            let a = results.pop().expect("a's reply")?;
            assert!(drafted(&a) > 0, "a drafted");
            assert_eq!(a.text, a_alone.text, "a's whole turn's tokens");
            assert_eq!(b.text, b_alone.text, "b's tokens alone");
            let checkpoint = checkpoint_of(&state.inner)?;
            assert_eq!(checkpoint.0, b_checkpoint.0, "b ended last");
            assert!(
                checkpoint.1 == b_checkpoint.1,
                "b's cells are its own alone"
            );
            assert!(state.inner.dflash2_states.seq_lanes.is_empty());
            assert!(state.inner.dflash2_states.peer_cells.is_empty());
            assert!(
                state.inner.dflash2_states.lane_decode_limit.is_none(),
                "the decode cache limit went with the last lane"
            );
            Ok(())
        }
    }

    /// A DFlash2 lane preempted while it and its peer decode replays its
    /// history through the prefill, begins its lane again at the replay's
    /// first slice and drafts again from the ring the replay filled (the
    /// scheduled twin of the whole turn's `adopt_replayed_lane`); both
    /// sessions give their whole turns' tokens alone. A replay's K/V and
    /// cells come from a prefill, so a's tokens equal alone only where its
    /// trajectory has no bf16 near-tie: its 19 tokens end before token 19,
    /// whose top-2 logits are 2 ULP apart (token 28's are 1 ULP apart, and a
    /// replayed run flips there).
    #[test]
    fn a_preempted_dflash2_lane_replays_and_drafts_again() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_preempted_dflash2_lane_replays_and_drafts_again",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let [(_, first), (_, second)] = lane_prompts();
        let (a_n, b_n) = (19, 40);
        let (a_alone, _) = whole_turn_session(seed, &first, a_n)?;
        let (b_alone, _) = whole_turn_session(seed, &second, b_n)?;
        {
            let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
            let mut preempted_at = None;
            let mut drafted_after = 0;
            let (mut results, saw_lane) = drive_sessions(
                &mut state,
                &[("a", &first, a_n), ("b", &second, b_n)],
                true,
                |state, _| {
                    let (Some(&a), Some(&b)) = (
                        state.owner_sequences.get("a"),
                        state.owner_sequences.get("b"),
                    ) else {
                        return;
                    };
                    let verified =
                        |state: &crate::engine::hybrid_scheduler::HybridSchedulerState<
                            Qwen35Inner,
                        >,
                         seq: u32| {
                            state
                                .inner
                                .dflash2_states
                                .seq_lanes
                                .get(&seq)
                                .is_some_and(|lane| lane.gdn.is_some())
                        };
                    if preempted_at.is_some() {
                        drafted_after += usize::from(verified(state, a));
                        return;
                    }
                    let generated = state
                        .turn_mut(a)
                        .map_or(0, |turn| turn.payload.generated_tokens.len());
                    if generated >= 8 && verified(state, a) && verified(state, b) {
                        assert!(state.preempt_running(a), "a is preempted");
                        assert!(
                            !state.inner.dflash2_states.seq_lanes.contains_key(&a),
                            "the preemption ended a's lane"
                        );
                        state.resume_preempted();
                        preempted_at = Some(generated);
                    }
                },
            )?;
            assert!(saw_lane);
            assert!(preempted_at.is_some(), "a was preempted while both drafted");
            assert_eq!(state.preemptions(), 1);
            assert!(drafted_after > 0, "a's replayed lane verified again");
            let b = results.pop().expect("b's reply")?;
            let a = results.pop().expect("a's reply")?;
            assert!(drafted(&a) > 0 && drafted(&b) > 0);
            assert_eq!(a.text, a_alone.text, "a's whole turn's tokens");
            assert_eq!(b.text, b_alone.text, "b's whole turn's tokens");
            assert!(state.inner.dflash2_states.seq_lanes.is_empty());
            assert!(state.inner.dflash2_states.idle_lane.ring.is_some());
            Ok(())
        }
    }

    /// A scheduled DFlash2 turn on a draft load books its lane's whole
    /// draft state at admission: its sequence's GDN cell, the second cell,
    /// the draft ring, the context window and the pending GDN tape. While
    /// two lanes decode, each lane's held bytes are on the states account
    /// (ring buffers, second cell and pending tape, at exactly the priced
    /// terms) and, with its window, inside its turn's charge.
    #[test]
    fn a_draft_lane_reserves_its_draft_state() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("a_draft_lane_reserves_its_draft_state")
            || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        use crate::models::qwen3_5::dflash2_turn::memory_accounts;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let [(_, first), (_, second)] = lane_prompts();
        {
            let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
            let cell = state.inner.config.recurrent_state_bytes();
            let tape = gdn_pending_tape_bytes(&state.inner.config);
            let (ring, window) = {
                let draft = &state.inner.dflash2.as_ref().expect("draft").config;
                (draft.kv_ring_bytes(), draft.context_window_bytes())
            };
            assert!(cell > 0 && tape > 0 && ring > 0 && window > 0);
            let charge = cell + cell + ring + window + tape;
            let mut charged = std::collections::BTreeSet::new();
            let mut both = 0;
            let (results, saw_lane) = drive_sessions(
                &mut state,
                &[("a", &first, 24), ("b", &second, 24)],
                true,
                |state, _| {
                    let pending: Vec<u32> = state
                        .inner
                        .dflash2_states
                        .seq_lanes
                        .iter()
                        .filter(|(_, lane)| {
                            lane.gdn.as_ref().is_some_and(|gdn| gdn.pending_bytes() > 0)
                        })
                        .map(|(&seq, _)| seq)
                        .collect();
                    if pending.len() == 2 {
                        both += 1;
                    }
                    for seq in pending {
                        let window_bytes = state
                            .inner
                            .dflash2_states
                            .seq_windows
                            .get(&seq)
                            .map_or(0, |window| window.nbytes());
                        let with = memory_accounts(&state.inner, 0).states;
                        let lane = state
                            .inner
                            .dflash2_states
                            .seq_lanes
                            .remove(&seq)
                            .expect("the lane");
                        let without = memory_accounts(&state.inner, 0).states;
                        state.inner.dflash2_states.seq_lanes.insert(seq, lane);
                        assert_eq!(
                            with - without,
                            ring + cell + tape,
                            "sequence {seq}: ring, second cell and pending tape on the account"
                        );
                        assert!(window_bytes <= window, "sequence {seq}: window");
                        let turn = state.turn_mut(seq).expect("the lane's turn");
                        assert_eq!(
                            turn.recurrent_state_bytes, charge,
                            "sequence {seq}: the turn's charge"
                        );
                        assert!(cell + ring + cell + tape + window_bytes <= charge);
                        charged.insert(seq);
                    }
                },
            )?;
            assert!(saw_lane);
            for result in results {
                assert!(drafted(&result?) > 0, "each lane drafted");
            }
            assert_eq!(charged.len(), 2, "both lanes were read with pending rows");
            assert!(both > 0, "both lanes held pending rows in one pass");
            Ok(())
        }
    }

    fn accepted(result: &crate::engine::types::ChatResult) -> u64 {
        result.performance.as_ref().map_or(0, |metrics| {
            (metrics.mtp_mean_accepted_tokens.unwrap_or(0.0)
                * f64::from(metrics.mtp_cycles.unwrap_or(0)))
            .round() as u64
        })
    }

    /// The fixture whose head lifts token 15 over every hidden: the draft
    /// and the target agree on most tokens, so cycles accept drafts (1 to 3
    /// a cycle; below a lift of 1.2 this draft accepts almost none), and
    /// other tokens still win where a hidden reaches past the lift.
    fn accepting_inner(seed: u64) -> Result<Qwen35Inner> {
        let mut inner = tiny_paged_int8_dflash_inner(seed);
        lift_last_token(&mut inner)?;
        Ok(inner)
    }

    fn lift_last_token(inner: &mut Qwen35Inner) -> Result<()> {
        let mut lift = vec![0.0f32; VOCAB as usize];
        lift[VOCAB as usize - 1] = 1.3;
        let bias = MxArray::from_float32(&lift, &[i64::from(VOCAB)])?.astype(DType::BFloat16)?;
        match inner.lm_head.as_mut() {
            Some(LinearProj::Standard(head)) => head.set_bias(Some(&bias)),
            _ => Err(Error::from_reason("the fixture's head is a plain linear")),
        }
    }

    fn verify_forwards() -> u64 {
        crate::models::qwen3_5::paged_forward::DFLASH2_VERIFY_FORWARDS.with(std::cell::Cell::get)
    }

    /// A DFlash2 session alone on the scheduled lane runs the whole turn's
    /// forwards: its tokens, cycles, accepted drafts and DFlash2 verify
    /// forwards equal the whole turn's (`MLX_SERVE_FORCE_SERIAL=1`); a wave
    /// of one is the whole turn's `[1, 8]` verify and nothing else.
    #[test]
    fn a_lone_dflash2_session_runs_the_whole_turns_forwards() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_lone_dflash2_session_runs_the_whole_turns_forwards",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let body = prompt(40);
        let n = 24;
        let before = verify_forwards();
        let (alone, alone_checkpoint) = whole_turn_on(accepting_inner(seed)?, &body, n)?;
        let whole = verify_forwards() - before;
        let mut state = scheduler_on(accepting_inner(seed)?)?;
        let before = verify_forwards();
        let (mut results, saw_lane) =
            drive_sessions(&mut state, &[("a", &body, n)], true, |_, _| {})?;
        let scheduled = verify_forwards() - before;
        let lane = results.remove(0)?;
        assert!(saw_lane, "the session ran on a scheduled DFlash2 lane");
        assert_eq!(lane.text, alone.text, "the whole turn's tokens");
        assert!(drafted(&alone) > 0 && accepted(&alone) > 0);
        assert_eq!(drafted(&lane), drafted(&alone), "the whole turn's cycles");
        assert_eq!(
            accepted(&lane),
            accepted(&alone),
            "the whole turn's accepted drafts"
        );
        assert_eq!(scheduled, whole, "the whole turn's verify forwards");
        assert_eq!(state.scheduler_stats().max_batch_occupancy, 1);
        let checkpoint = checkpoint_of(&state.inner)?;
        assert_eq!(checkpoint.0, alone_checkpoint.0, "checkpoint tokens");
        assert!(
            checkpoint.1 == alone_checkpoint.1,
            "the whole turn's settled cells"
        );
        Ok(())
    }

    /// The fixture of the two-lane session test: a 4,096-row window (room
    /// for a prompt past one 2,048-row prefill unit), a 40-row draft window,
    /// the accepting head ([`accepting_inner`]), two lanes, the tier `tier`
    /// when given.
    fn two_lane_state(
        seed: u64,
        tier: Option<&crate::models::qwen3_5::dflash2_state::disk::tests::TestTier>,
    ) -> Result<crate::engine::hybrid_scheduler::HybridSchedulerState<Qwen35Inner>> {
        use crate::models::qwen3_5::dflash2_state::disk::tests::attach_tier;
        let mut inner = tiny_paged_int8_dflash_inner_with(seed, Heads::Same, |config| {
            config.max_position_embeddings = 4_096;
        });
        inner.dflash2.as_mut().expect("draft").config.sliding_window = 40;
        lift_last_token(&mut inner)?;
        if let Some(tier) = tier {
            let window = inner
                .dflash2
                .as_ref()
                .expect("draft")
                .config
                .context_window()?;
            attach_tier(
                &mut inner.dflash2_states,
                &inner.config,
                inner.paged_adapter.as_ref().expect("pool"),
                tier,
                window.nbytes(),
                window.slots(),
            );
            inner.paged_adapter.as_mut().expect("pool").set_cold_tier(
                crate::models::qwen3_5::dflash2_state::disk::state_context(&tier.tier),
            );
        }
        scheduler_on(inner)
    }

    /// `body`'s session alone as the whole DFlash2 turn on
    /// [`two_lane_state`]: its reply and the states it published.
    fn two_lane_alone(
        seed: u64,
        body: &[u32],
        max_new_tokens: i32,
    ) -> Result<(crate::engine::types::ChatResult, Vec<DraftState>)> {
        let mut state = two_lane_state(seed, None)?;
        state.serve_whole_turns();
        let (mut results, saw_lane) =
            drive_sessions(&mut state, &[("a", body, max_new_tokens)], true, |_, _| {})?;
        assert!(!saw_lane);
        let result = results.remove(0)?;
        let states = draft_states(&state.inner, &super::tests::rendered_prompt(body))?;
        Ok((result, states))
    }

    /// What one drive pass saw of a session.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    struct Seen {
        computed: u32,
        prompt: u32,
        generated: usize,
        lane_position: Option<i32>,
    }

    fn seen(
        state: &mut crate::engine::hybrid_scheduler::HybridSchedulerState<Qwen35Inner>,
        owner: &str,
    ) -> Option<Seen> {
        let seq = *state.owner_sequences.get(owner)?;
        let lane_position = state
            .inner
            .dflash2_states
            .seq_lanes
            .get(&seq)
            .map(|lane| lane.next_position);
        let turn = state.turn_mut(seq)?;
        Some(Seen {
            computed: turn.num_computed_tokens,
            prompt: turn.prompt_tokens,
            generated: turn.payload.generated_tokens.len(),
            lane_position,
        })
    }

    /// The app's turn shape on the fixture: session a decodes with its
    /// draft on; session b, its draft on, arrives once a has decoded, its
    /// prompt past one prefill unit. b's prefill runs in slices between a's
    /// verify waves (a drafts in each of those passes), then both lanes
    /// verify in the same passes. Each session's tokens, cycles and accepted
    /// drafts are its whole turn's alone; each published the whole turn's
    /// states; the scheduler stepped with two decode rows; no lane is left
    /// and the idle lane holds the released ring. After a restart a session
    /// extending a's prompt resumes at a's replay point from disk and drafts
    /// from the ring its window restores.
    #[test]
    fn scheduler_runs_two_dflash2_sessions_as_two_lanes() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "scheduler_runs_two_dflash2_sessions_as_two_lanes",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        use super::tests::{rendered_prompt, session_start_with};
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        use crate::engine::model_command::ModelCommand;
        use crate::model_thread::LoopControl;
        use crate::models::qwen3_5::dflash2_state::disk::tests::persistent_test_tier;
        use crate::models::qwen3_5::dflash2_state::disk::{adopt_all, close_cache};
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let a_body = prompt(150);
        let b_body: Vec<u32> = (0..2_600u32).map(|i| (i * 5 + 1) % VOCAB).collect();
        let longer = prompt(180);
        let (a_n, b_n) = (40, 24);
        let (a_alone, a_states) = two_lane_alone(seed, &a_body, a_n)?;
        let (b_alone, b_states) = two_lane_alone(seed, &b_body, b_n)?;

        let tier = persistent_test_tier("two-dflash2-lanes", 1 << 30);
        let mut state = two_lane_state(seed, Some(&tier))?;
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let send = |sender: &tokio::sync::mpsc::UnboundedSender<_>,
                    owner: &str,
                    body: &[u32],
                    n: i32|
         -> Result<_> {
            let (command, reply, _) = session_start_with(owner, body, n, true)?;
            sender
                .send(ModelCommand::<
                    <Qwen35Inner as HybridSchedulerBackend>::FamilyCommand,
                >::from_chat(command))
                .map_err(|_| Error::from_reason("the command channel closed"))?;
            Ok(reply)
        };
        let mut a_reply = send(&sender, "a", &a_body, a_n)?;
        // b's send drops the last sender: an idle pass then ends the loop
        // instead of blocking this thread.
        let mut sender = Some(sender);
        let mut b_reply: Option<
            tokio::sync::oneshot::Receiver<Result<crate::engine::types::ChatResult>>,
        > = None;
        let (mut a_done, mut b_done) = (None, None);
        let (mut a_seen, mut b_seen) = (Seen::default(), Seen::default());
        let mut b_arrived_at = None;
        let mut slices_beside = 0;
        let mut both_drafted = 0;
        let mut most_lanes = 0;
        for _ in 0..512 {
            if !matches!(state.drive(&mut receiver), LoopControl::Continue) {
                return Err(Error::from_reason("the scheduler stopped"));
            }
            most_lanes = most_lanes.max(state.inner.dflash2_states.seq_lanes.len());
            let a_now = seen(&mut state, "a");
            let b_now = seen(&mut state, "b");
            if let (Some(a), Some(b)) = (a_now, b_now) {
                let a_drafted =
                    a.lane_position > a_seen.lane_position && a_seen.lane_position.is_some();
                let b_prefilling = b_seen.computed < b.prompt;
                if b_prefilling && b.computed > b_seen.computed && a_drafted {
                    slices_beside += 1;
                }
                let b_drafted =
                    b.lane_position > b_seen.lane_position && b_seen.lane_position.is_some();
                if !b_prefilling && a_drafted && b_drafted {
                    both_drafted += 1;
                }
            }
            if let Some(a) = a_now {
                a_seen = a;
            }
            if let Some(b) = b_now {
                b_seen = b;
            }
            if a_done.is_none()
                && let Ok(reply) = a_reply.try_recv()
            {
                a_done = Some(reply?);
            }
            if b_done.is_none()
                && let Some(Ok(reply)) = b_reply.as_mut().map(|reply| reply.try_recv())
            {
                b_done = Some(reply?);
            }
            if a_done.is_some() && b_done.is_some() {
                break;
            }
            if let Some(sender) = sender.take_if(|_| a_seen.generated >= 2 || a_done.is_some()) {
                b_reply = Some(send(&sender, "b", &b_body, b_n)?);
                b_arrived_at = Some(a_done.is_none());
            }
        }
        let (Some(a), Some(b)) = (a_done, b_done) else {
            return Err(Error::from_reason("the sessions did not finish"));
        };
        assert_eq!(b_arrived_at, Some(true), "b arrived while a decoded");
        assert!(
            slices_beside >= 2,
            "b's prefill ran in slices between a's waves: {slices_beside}"
        );
        assert!(both_drafted > 0, "both lanes verified in the same passes");
        assert_eq!(most_lanes, 2, "two live lanes");
        let stats = state.scheduler_stats();
        assert!(stats.global_steps > 0);
        assert!(
            stats.max_batch_occupancy >= 2,
            "the scheduler stepped two decode rows: {}",
            stats.max_batch_occupancy
        );
        for (tag, result, alone) in [("a", &a, &a_alone), ("b", &b, &b_alone)] {
            assert!(drafted(result) > 0 && accepted(result) > 0, "{tag} drafted");
            assert_eq!(result.text, alone.text, "{tag}: its whole turn's tokens");
            assert_eq!(
                drafted(result),
                drafted(alone),
                "{tag}: its whole turn's cycles"
            );
            assert_eq!(
                accepted(result),
                accepted(alone),
                "{tag}: its whole turn's accepted drafts"
            );
        }
        assert_eq!(
            draft_states(&state.inner, &rendered_prompt(&a_body))?,
            a_states,
            "a published its whole turn's states"
        );
        assert_eq!(
            draft_states(&state.inner, &rendered_prompt(&b_body))?,
            b_states,
            "b published its whole turn's states"
        );
        assert!(state.inner.dflash2_states.seq_lanes.is_empty());
        assert!(state.inner.dflash2_states.seq_windows.is_empty());
        assert!(state.inner.dflash2_states.turns.is_empty());
        assert!(state.inner.dflash2_states.peer_cells.is_empty());
        assert!(state.inner.dflash2_states.lane_decode_limit.is_none());
        assert!(state.inner.dflash2_states.idle_lane.ring.is_some());
        assert!(state.inner.dflash2_states.idle_lane.window.is_some());
        let replay = draft_states(&state.inner, &rendered_prompt(&a_body))?
            .into_iter()
            .map(|stored| stored.0)
            .max()
            .expect("a's replay point");
        assert!(close_cache(
            &mut state.inner,
            std::time::Instant::now() + std::time::Duration::from_secs(30)
        ));
        drop(state);

        let tier = tier.reopen("two-dflash2-lanes", 1 << 30);
        let mut state = two_lane_state(seed, Some(&tier))?;
        assert!(adopt_all(&mut state.inner) >= 1);
        let (mut results, saw_lane) =
            drive_sessions(&mut state, &[("c", &longer, 24)], true, |_, _| {})?;
        let c = results.remove(0)?;
        assert!(saw_lane);
        assert_eq!(c.cached_tokens as usize, replay, "c resumed from disk");
        assert!(
            drafted(&c) > 0 && accepted(&c) > 0,
            "c drafted after its resume"
        );
        Ok(())
    }

    /// What one owner's lane-state reservation was over the passes.
    #[derive(Debug, Default)]
    struct LaneBooks {
        /// Passes its lane had begun and not verified: its turn held the
        /// lane reservation.
        held: u32,
        /// Of those, passes inside its prefill.
        in_prefill: u32,
        /// Passes after its lane's first verify: no lane reservation left.
        verified: u32,
    }

    /// Reads `owner`'s lane-state reservation after a pass: from its lane's
    /// begin (its first step) until its first verify its turn holds
    /// `lane_state` bytes on the governor's books; after it, none.
    fn read_lane_books(
        state: &mut lane_sessions::State,
        owner: &str,
        lane_state: u64,
        books: &mut LaneBooks,
    ) {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let Some(&seq) = state.owner_sequences.get(owner) else {
            return;
        };
        let Some(verified) = state
            .inner
            .dflash2_states
            .seq_lanes
            .get(&seq)
            .map(|lane| lane.gdn.is_some())
        else {
            return;
        };
        let reserved = state
            .inner
            .memory_governor()
            .expect("the fixture's governor")
            .reserved_bytes();
        let Some(turn) = state.turn_mut(seq) else {
            return;
        };
        let held = turn
            .payload
            .draft_lane
            .as_ref()
            .map(crate::engine::governor::governor::Reservation::bytes);
        if verified {
            assert_eq!(
                held, None,
                "{owner}: the reservation ended at its first verify"
            );
            books.verified += 1;
        } else {
            assert_eq!(
                held,
                Some(lane_state),
                "{owner}: second cell, ring, window and tape reserved until the first verify"
            );
            assert!(reserved >= lane_state, "{owner}: on the governor's books");
            books.held += 1;
            if turn.num_computed_tokens < turn.prompt_tokens {
                books.in_prefill += 1;
            }
        }
    }

    /// The lane state a DFlash2 lane allocates at its first verify, from the
    /// fixture's terms: the second GDN cell, the ring, the context window
    /// and the pending GDN tape.
    fn lane_state_terms(inner: &Qwen35Inner) -> u64 {
        let draft = &inner.dflash2.as_ref().expect("draft").config;
        let terms = [
            inner.config.recurrent_state_bytes(),
            draft.kv_ring_bytes(),
            draft.context_window_bytes(),
            gdn_pending_tape_bytes(&inner.config),
        ];
        assert!(terms.iter().all(|&bytes| bytes > 0));
        terms.iter().sum()
    }

    /// A scheduled DFlash2 lane books its lane state (second cell, ring,
    /// window, pending tape) from its admission until the step of its first
    /// verify, which allocates the second cell and the tape: through every
    /// prefill slice of a prompt past one prefill unit, and for a lane
    /// beside a decoding lane.
    #[test]
    fn a_dflash2_lane_books_its_lane_state_until_its_first_verify() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_dflash2_lane_books_its_lane_state_until_its_first_verify",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let a_body = prompt(150);
        let b_body: Vec<u32> = (0..2_600u32).map(|i| (i * 5 + 1) % VOCAB).collect();
        let mut state = two_lane_state(seed, None)?;
        let lane_state = lane_state_terms(&state.inner);
        let (mut a_books, mut b_books) = (LaneBooks::default(), LaneBooks::default());
        let (results, saw_lane) = drive_sessions(
            &mut state,
            &[("a", &a_body, 24), ("b", &b_body, 16)],
            true,
            |state, _| {
                read_lane_books(state, "a", lane_state, &mut a_books);
                read_lane_books(state, "b", lane_state, &mut b_books);
            },
        )?;
        assert!(saw_lane);
        for result in results {
            assert!(drafted(&result?) > 0, "each lane drafted");
        }
        for (owner, books) in [("a", &a_books), ("b", &b_books)] {
            assert!(books.held > 0 && books.verified > 0, "{owner}: {books:?}");
        }
        assert!(
            b_books.in_prefill > 0,
            "b held it between its prefill slices: {b_books:?}"
        );
        Ok(())
    }

    /// An owner's next turn on its live sequence (its GDN cell live, so the
    /// admission reserves no cell) still books its new lane's state until
    /// that lane's first verify.
    #[test]
    fn a_continued_dflash2_lane_books_its_lane_state() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_continued_dflash2_lane_books_its_lane_state",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        use lane_sessions::{run, take, words};
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let mut state = two_lane_state(seed, None)?;
        let lane_state = lane_state_terms(&state.inner);
        let (mut results, _) =
            drive_sessions_with(&mut state, &[("a", &prompt(40), 16, true)], |_, _| {})?;
        assert!(drafted(&results.remove(0)?) > 0);
        let seq = state.owner_sequences["a"];
        assert!(
            state.inner.has_scheduled_recurrent(seq),
            "the owner's cell is live"
        );
        let history = state
            .inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .request_tokens_for(seq)
            .expect("a stays live")
            .to_vec();
        let (command, _, cancelled) = super::tests::session_start_with("a", &[], 16, true)?;
        let crate::engine::cmd::ChatCmd::SessionStart { config, .. } = command else {
            return Err(Error::from_reason("not a start"));
        };
        let message = |content: String| -> Result<crate::tokenizer::ChatMessage> {
            serde_json::from_value(serde_json::json!({"role": "user", "content": content}))
                .map_err(|error| Error::from_reason(error.to_string()))
        };
        let second: Vec<u32> = (0..12u32).map(|i| (i * 5 + 1) % VOCAB).collect();
        let (reply, mut result) = tokio::sync::oneshot::channel();
        let command = crate::engine::cmd::ChatCmd::SessionContinue {
            messages: vec![message(words(&history))?, message(words(&second))?],
            config,
            reply,
            cancelled,
        };
        let mut done = None;
        let mut books = LaneBooks::default();
        run(&mut state, vec![command], false, |state, _| {
            read_lane_books(state, "a", lane_state, &mut books);
            take(&mut result, &mut done);
            Ok(done.is_some())
        })?;
        assert_eq!(state.owner_sequences["a"], seq, "the same live sequence");
        assert!(drafted(&done.expect("turn 2")?) > 0, "turn 2 drafted");
        assert!(books.held > 0 && books.verified > 0, "{books:?}");
        Ok(())
    }

    /// The load of a draft model plans every scheduled lane as a DFlash2
    /// lane: the first lane its cell and the draft's lane state, each later
    /// lane its cell, a second cell, a ring, a window and a pending tape.
    #[test]
    fn a_draft_load_plans_each_lane_as_a_dflash2_lane() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_draft_load_plans_each_lane_as_a_dflash2_lane",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8350);
        inner.paged_adapter = None;
        inner.initialize_paged_adapter()?;
        let pool = inner
            .memory_plan
            .as_ref()
            .and_then(|plan| plan.pool)
            .expect("the load planned a pool");
        assert!(pool.lanes >= 2, "the device grants lanes: {}", pool.lanes);
        let cell = inner.config.recurrent_state_bytes();
        let first = crate::engine::memory_plan::dflash2_lane_state_bytes(
            cell,
            inner.dflash2.as_ref().expect("draft").config.lane_bytes(),
        );
        let later = cell + lane_state_terms(&inner);
        assert_eq!(
            pool.lane.lane_bytes,
            cell + first + u64::from(pool.lanes - 1) * later,
            "{} lanes",
            pool.lanes
        );
        Ok(())
    }

    /// Two thinking sessions (the app's turns think) on two lanes, each cut
    /// at its thinking budget: the end token rides the lane's verify, and
    /// each session's reply (thought, close, text, cycles) is its whole
    /// turn's alone.
    #[test]
    fn a_dflash2_lane_cuts_thinking_as_its_whole_turn() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_dflash2_lane_cuts_thinking_as_its_whole_turn",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        use lane_sessions::{clean, run, start_cmd, take};
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let [(_, first), (_, second)] = lane_prompts();
        let n = 24;
        let thinking = |budget: i32| {
            move |config: &mut crate::engine::types::ChatConfig| {
                config.reasoning_effort = Some("high".to_string());
                config.thinking_token_budget = Some(budget);
            }
        };
        for accepting in [false, true] {
            let state_on = || -> Result<lane_sessions::State> {
                let inner = if accepting {
                    accepting_inner(seed)?
                } else {
                    tiny_paged_int8_dflash_inner(seed)
                };
                let mut state = scheduler_on(inner)?;
                state.inner.tokenizer = Some(super::tests::think_tokenizer());
                Ok(state)
            };
            // Below each session's own close (the accepting head closes b's
            // thought at its 4th token).
            let budgets = [("a", &first, 3), ("b", &second, 2)];
            let mut alone = Vec::new();
            for &(owner, body, budget) in &budgets {
                let mut state = state_on()?;
                state.serve_whole_turns();
                let (command, mut reply, _) = start_cmd(owner, body, n, thinking(budget))?;
                let mut done = None;
                let mut saw_lane = false;
                run(&mut state, vec![command], false, |state, _| {
                    saw_lane |= !state.inner.dflash2_states.seq_lanes.is_empty();
                    take(&mut reply, &mut done);
                    Ok(done.is_some())
                })?;
                assert!(!saw_lane);
                alone.push(done.expect("whole turn")?);
            }
            let mut state = state_on()?;
            let mut commands = Vec::new();
            let mut replies = Vec::new();
            for &(owner, body, budget) in &budgets {
                let (command, reply, _) = start_cmd(owner, body, n, thinking(budget))?;
                commands.push(command);
                replies.push(reply);
            }
            let mut done = [None, None];
            let mut most = 0;
            run(&mut state, commands, false, |state, _| {
                most = most.max(state.inner.dflash2_states.seq_lanes.len());
                for (reply, slot) in replies.iter_mut().zip(done.iter_mut()) {
                    take(reply, slot);
                }
                Ok(done.iter().all(Option::is_some))
            })?;
            assert_eq!(most, 2, "accepting {accepting}: two live lanes");
            for ((owner, _, budget), (slot, whole)) in
                budgets.iter().zip(done.into_iter().zip(&alone))
            {
                let lane = slot.expect("reply")?;
                let tag = format!("accepting {accepting}, {owner}");
                for result in [&lane, whole] {
                    assert_eq!(
                        result.reasoning_close.as_deref(),
                        Some("budget"),
                        "{tag}: {}",
                        result.raw_text
                    );
                    assert_eq!(result.reasoning_cut_at, Some(*budget as u32), "{tag}");
                }
                assert!(drafted(&lane) > 0, "{tag}: drafted");
                assert_eq!(
                    lane.raw_text, whole.raw_text,
                    "{tag}: the whole turn's tokens"
                );
                assert_eq!(lane.thinking, whole.thinking, "{tag}: thought");
                assert_eq!(lane.text, whole.text, "{tag}: text");
                assert_eq!(drafted(&lane), drafted(whole), "{tag}: cycles");
                assert_eq!(accepted(&lane), accepted(whole), "{tag}: accepted drafts");
            }
            clean(&state, "thinking cut");
        }
        Ok(())
    }

    /// Two sampled thinking sessions (the app's T 0.6, top-k 20, top-p 0.95)
    /// on two lanes, each lane's rng seeded from its owner, each cut at its
    /// thinking budget: the close lands at the budget, every cycle (the
    /// cut's too) takes the device rows, a seed repeats each reply, and each
    /// lane's reply is the reply its session gives alone.
    #[test]
    fn sampled_lanes_cut_thinking_at_the_budget_on_the_device_rows() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "sampled_lanes_cut_thinking_at_the_budget_on_the_device_rows",
        ) || !super::tests::metal_is_default_device()
        {
            return Ok(());
        }
        use crate::engine::dspark_turn::{TEST_SAMPLED_ACCEPTS, TEST_SAMPLED_ROWS};
        use crate::engine::hybrid_scheduler::TEST_LANE_SEED;
        use lane_sessions::{clean, run, start_cmd, take};
        struct Seed;
        impl Drop for Seed {
            fn drop(&mut self) {
                TEST_LANE_SEED.with(|seed| seed.set(None));
            }
        }
        let _seed = Seed;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_8350;
        let [(_, first), (_, second)] = lane_prompts();
        let n = 24;
        let budgets = [("a", &first, 3), ("b", &second, 2)];
        let sessions = |picked: &[usize]| -> Result<Vec<crate::engine::types::ChatResult>> {
            let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
            state.inner.tokenizer = Some(super::tests::think_tokenizer());
            let mut commands = Vec::new();
            let mut replies = Vec::new();
            for &index in picked {
                let (owner, body, budget) = budgets[index];
                let (command, reply, _) = start_cmd(owner, body, n, |config| {
                    config.reasoning_effort = Some("high".to_string());
                    config.thinking_token_budget = Some(budget);
                    config.temperature = Some(0.6);
                    config.top_k = Some(20);
                    config.top_p = Some(0.95);
                    config.min_p = Some(0.0);
                })?;
                commands.push(command);
                replies.push(reply);
            }
            let mut done: Vec<Option<Result<crate::engine::types::ChatResult>>> =
                picked.iter().map(|_| None).collect();
            let mut most = 0;
            run(&mut state, commands, false, |state, _| {
                most = most.max(state.inner.dflash2_states.seq_lanes.len());
                for (reply, slot) in replies.iter_mut().zip(done.iter_mut()) {
                    take(reply, slot);
                }
                Ok(done.iter().all(Option::is_some))
            })?;
            assert_eq!(most, picked.len(), "{picked:?}: one live lane per session");
            clean(&state, "sampled thinking cut");
            done.into_iter().map(|slot| slot.expect("reply")).collect()
        };
        TEST_LANE_SEED.with(|base| base.set(Some(0x5EED)));
        TEST_SAMPLED_ROWS.with(|c| c.set(0));
        TEST_SAMPLED_ACCEPTS.with(|c| c.set((0, 0)));
        let pair = sessions(&[0, 1])?;
        let rows = TEST_SAMPLED_ROWS.with(|c| c.get());
        let chain = TEST_SAMPLED_ACCEPTS.with(|c| c.get());
        assert!(
            rows > 0 && chain == (0, 0),
            "sampled lanes take the device rows: {rows} cycles, chain {chain:?}"
        );
        for ((owner, _, budget), lane) in budgets.iter().zip(&pair) {
            assert_eq!(
                lane.reasoning_close.as_deref(),
                Some("budget"),
                "{owner}: {}",
                lane.raw_text
            );
            assert_eq!(lane.reasoning_cut_at, Some(*budget as u32), "{owner}");
            assert!(drafted(lane) > 0, "{owner}: drafted");
        }
        let again = sessions(&[0, 1])?;
        for (index, (lane, repeat)) in pair.iter().zip(&again).enumerate() {
            let owner = budgets[index].0;
            assert_eq!(
                lane.raw_text, repeat.raw_text,
                "{owner}: a seed repeats its reply"
            );
            let alone = sessions(&[index])?.pop().expect("one reply");
            assert_eq!(lane.raw_text, alone.raw_text, "{owner}: its reply alone");
            assert_eq!(lane.thinking, alone.thinking, "{owner}: thought alone");
            assert_eq!(lane.text, alone.text, "{owner}: text alone");
            assert_eq!(drafted(lane), drafted(&alone), "{owner}: cycles alone");
        }
        Ok(())
    }

    /// `seq` primed on `tokens` and its DFlash2 lane begun where it
    /// resumes.
    fn pass_lane(
        inner: &mut Qwen35Inner,
        seq: u32,
        tokens: &[u32],
    ) -> Result<<Qwen35Inner as PagedBackend>::PrefixState> {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let base = scheduled_prime(inner, seq, tokens)?;
        let resumed = u32::try_from(base.effective_cached_prefix_len()).expect("rows");
        assert!(HybridSchedulerBackend::begin_scheduled_speculation(
            inner, seq, resumed
        )?);
        Ok(base)
    }

    /// One pass of `seq`'s unit `[start, end)` within `budget_ms`, the
    /// sequence parked after it.
    fn unit_pass(
        inner: &mut Qwen35Inner,
        seq: u32,
        tokens: &[u32],
        base: &<Qwen35Inner as PagedBackend>::PrefixState,
        (start, end): (usize, usize),
        budget_ms: f64,
    ) -> Result<crate::engine::hybrid_scheduler::ScheduledPrefillPass> {
        use crate::engine::hybrid_scheduler::{HybridSchedulerBackend, ScheduledPrefillPass};
        HybridSchedulerBackend::activate_paged_seq(inner, seq)?;
        let prefix = HybridSchedulerBackend::build_scheduled_prefix(
            inner,
            base,
            start,
            end - start,
            tokens.to_vec(),
            start == base.effective_cached_prefix_len(),
        );
        let pass = inner.dense_paged_prefill_pass(&prefix, start, end, Some(budget_ms))?;
        if let ScheduledPrefillPass::Done(Some(logits)) = &pass {
            logits.eval();
        }
        HybridSchedulerBackend::park_active_scheduled_recurrent(inner)?;
        Ok(pass)
    }

    /// [`unit_pass`] of a unit that ends the prompt: its logits once it is
    /// done.
    fn one_pass(
        inner: &mut Qwen35Inner,
        seq: u32,
        tokens: &[u32],
        base: &<Qwen35Inner as PagedBackend>::PrefixState,
        unit: (usize, usize),
        budget_ms: f64,
    ) -> Result<Option<MxArray>> {
        use crate::engine::hybrid_scheduler::ScheduledPrefillPass;
        match unit_pass(inner, seq, tokens, base, unit, budget_ms)? {
            ScheduledPrefillPass::Paused => Ok(None),
            ScheduledPrefillPass::Done(Some(logits)) => Ok(Some(logits)),
            ScheduledPrefillPass::Done(None) => Err(Error::from_reason("a unit gave no logits")),
        }
    }

    /// What a lane holds once its prompt is prefilled: the logits bits, its
    /// cells, its ring, its window, the states on its lineage, its booked
    /// rows, and the logits of one decode step after it (which reads every
    /// K/V page it wrote).
    #[allow(clippy::type_complexity)]
    fn lane_after_prefill(
        inner: &mut Qwen35Inner,
        seq: u32,
        tokens: &[u32],
        logits: &MxArray,
    ) -> Result<(
        Vec<u32>,
        Vec<Vec<u32>>,
        RingState,
        ((u32, u32), Vec<u32>),
        Vec<DraftState>,
        u32,
        Vec<u32>,
    )> {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        HybridSchedulerBackend::activate_paged_seq(inner, seq)?;
        let cells = cell_bits(inner.caches.as_ref().expect("caches"))?;
        let ring = ring_state(&inner.dflash2_states.seq_lanes[&seq].ring)?;
        let window = window_state(&inner.dflash2_states.seq_windows[&seq])?;
        let states = draft_states(inner, tokens)?;
        let booked = inner
            .paged_adapter
            .as_ref()
            .and_then(|adapter| adapter.current_token_count_for(seq))
            .expect("booked rows");
        let token = argmax_rows(logits)?[0];
        inner
            .paged_adapter
            .as_mut()
            .expect("pool")
            .reserve_rows_for(seq, 1)
            .map_err(Error::from_reason)?;
        let next = HybridSchedulerBackend::run_paged_decode_step_batched(inner, &[(seq, token)])?;
        Ok((
            bits(logits)?,
            cells,
            ring,
            window,
            states,
            booked,
            bits(&next)?,
        ))
    }

    /// A prefill unit run in one-layer passes (Splash `PassCommand`) is the
    /// one call bit for bit: its logits, the cells, the lane's ring and
    /// window, the states it published, its booked rows and the next decode
    /// step's logits (every K/V page) — over ten 16-row launches whose
    /// state points the 40-row window spans, the draft back on the model
    /// between passes.
    /// Mutations, each → red: a pass resumes one layer on; a launch's draft
    /// context is taken twice.
    #[test]
    fn a_unit_in_one_layer_passes_equals_one_call() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_unit_in_one_layer_passes_equals_one_call",
        ) {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_9200;
        let tokens = prompt(150);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            let mut lone = wide_draft_inner(seed, 40);
            let base = pass_lane(&mut lone, 1, &tokens)?;
            let logits = scheduled_last_slice(&mut lone, 1, &tokens, &base, 0)?;
            let want = lane_after_prefill(&mut lone, 1, &tokens, &logits)?;
            assert!(!want.4.is_empty(), "the unit publishes states");

            let mut passed = wide_draft_inner(seed, 40);
            let base = pass_lane(&mut passed, 1, &tokens)?;
            let layers = passed.config.num_layers as usize;
            use crate::models::qwen3_5::paged_forward::PREFILL_PASSES;
            PREFILL_PASSES.with(|log| *log.borrow_mut() = Some(Vec::new()));
            let logits = loop {
                if let Some(logits) =
                    one_pass(&mut passed, 1, &tokens, &base, (0, tokens.len()), 0.0)?
                {
                    break logits;
                }
                assert!(passed.dflash2.is_some(), "the draft is back between passes");
            };
            let log = PREFILL_PASSES
                .with(|log| log.borrow_mut().take())
                .expect("the pass log");
            assert!(log.iter().all(|&ran| ran <= 1), "one layer a pass: {log:?}");
            assert_eq!(
                log.iter().filter(|&&ran| ran == 1).count(),
                10 * layers,
                "one pass a layer of each launch"
            );
            // A launch ending at an armed capture does its end work in a
            // pass of its own.
            let end_work = log.iter().filter(|&&ran| ran == 0).count();
            assert!(end_work > 0, "the window spans an armed capture: {log:?}");
            assert!(end_work < 10, "only launches ending at a capture: {log:?}");
            assert!(passed.prefill_runs.is_empty());
            let got = lane_after_prefill(&mut passed, 1, &tokens, &logits)?;
            assert!(got.0 == want.0, "prefill logits");
            assert!(got.1 == want.1, "cells");
            assert!(got.2 == want.2, "ring");
            assert!(got.3 == want.3, "window");
            assert!(got.4 == want.4, "published states");
            assert_eq!(got.5, want.5, "booked rows");
            assert!(got.6 == want.6, "the next step's logits");
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// One-layer passes store the one call's store-only tails, with INT8
    /// queries from a row inside the unit: the same tail count, every tail's
    /// pool writes evaluated by the eval that ends its launch (also where a
    /// launch pauses at an armed capture after its tail), and the one call's
    /// bits.
    #[test]
    fn passes_store_the_tails_of_one_call() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("passes_store_the_tails_of_one_call") {
            return Ok(());
        }
        use crate::models::qwen3_5::paged_forward::{
            PREFILL_PASSES, PREFILL_TAIL_WRITES_READY, PREFILL_TAILS_STORED,
        };
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_9201;
        let tokens = prompt(150);
        let arm = |passes: bool| -> Result<Option<_>> {
            let Some(mut inner) = int8_query_inner(seed, 70) else {
                return Ok(None);
            };
            inner.dflash2.as_mut().expect("draft").config.sliding_window = 40;
            PREFILL_TAILS_STORED.with(|count| count.set(0));
            PREFILL_TAIL_WRITES_READY.with(|log| log.borrow_mut().clear());
            PREFILL_PASSES.with(|log| *log.borrow_mut() = Some(Vec::new()));
            let base = pass_lane(&mut inner, 1, &tokens)?;
            let logits = if passes {
                loop {
                    if let Some(logits) =
                        one_pass(&mut inner, 1, &tokens, &base, (0, tokens.len()), 0.0)?
                    {
                        break logits;
                    }
                }
            } else {
                scheduled_last_slice(&mut inner, 1, &tokens, &base, 0)?
            };
            let end_work = PREFILL_PASSES
                .with(|log| log.borrow_mut().take())
                .expect("the pass log")
                .iter()
                .filter(|&&ran| ran == 0)
                .count();
            let stored = PREFILL_TAILS_STORED.with(std::cell::Cell::get);
            let ready =
                PREFILL_TAIL_WRITES_READY.with(|log| std::mem::take(&mut *log.borrow_mut()));
            let int8 = int8_query_work(&inner)[0];
            let after = lane_after_prefill(&mut inner, 1, &tokens, &logits)?;
            Ok(Some((after, stored, ready, int8, end_work)))
        };
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            let Some(lone) = arm(false)? else {
                return Ok(());
            };
            let passed = arm(true)?.expect("INT8 queries on the second fixture");
            assert!(lone.1 > 0, "the one call stored no tail");
            assert_eq!(passed.1, lone.1, "store-only tails");
            assert!(lone.3 > 0 && passed.3 > 0, "INT8 queries scored");
            assert!(passed.4 > 0, "a launch pauses at an armed capture");
            assert!(
                !passed.2.is_empty() && passed.2.iter().all(|&ready| ready),
                "{:?}",
                passed.2
            );
            assert!(lone.2.iter().all(|&ready| ready), "{:?}", lone.2);
            let (got, want) = (passed.0, lone.0);
            assert!(got.0 == want.0, "prefill logits");
            assert!(got.1 == want.1, "cells");
            assert!(got.2 == want.2, "ring");
            assert!(got.3 == want.3, "window");
            assert!(got.4 == want.4, "published states");
            assert_eq!(got.5, want.5, "booked rows");
            assert!(got.6 == want.6, "the next step's logits");
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// A prompt cut into scheduler units whose ends fall on the one call's
    /// launch ends (64, a single-launch unit; 144, a state point): each unit
    /// short of the prompt's end gives no logits and its last launch stores
    /// its last layer's K/V only, as the one call's launches inside it do,
    /// with the pool writes evaluated at the unit's end; alone and in
    /// one-layer passes the lane holds the one call's bits.
    /// Mutations (each red here): a short unit still projects logits; its
    /// last layer runs in full; its end eval leaves out the pool writes; its
    /// store-only last layer stores the K rotated one row on.
    #[test]
    fn units_short_of_the_prompt_store_tails_and_give_the_one_call_bits() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "units_short_of_the_prompt_store_tails_and_give_the_one_call_bits",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::ScheduledPrefillPass;
        use crate::models::qwen3_5::paged_forward::{
            PREFILL_TAIL_WRITES_READY, PREFILL_TAILS_STORED,
        };
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_9230;
        let tokens = prompt(150);
        let one_call = [(0usize, tokens.len())];
        let units = [(0usize, 64usize), (64, 144), (144, tokens.len())];
        let arm = |cuts: &[(usize, usize)], passes: bool| -> Result<_> {
            let mut inner = wide_draft_inner(seed, 40);
            let base = pass_lane(&mut inner, 1, &tokens)?;
            PREFILL_TAILS_STORED.with(|count| count.set(0));
            PREFILL_TAIL_WRITES_READY.with(|log| log.borrow_mut().clear());
            let mut given = Vec::new();
            let mut last = None;
            for &unit in cuts {
                let logits = if passes {
                    loop {
                        if let ScheduledPrefillPass::Done(logits) =
                            unit_pass(&mut inner, 1, &tokens, &base, unit, 0.0)?
                        {
                            break logits;
                        }
                    }
                } else {
                    scheduled_slice(&mut inner, 1, &tokens, &base, unit.0, unit.1)?
                };
                given.push(logits.is_some());
                last = logits;
            }
            let stored = PREFILL_TAILS_STORED.with(std::cell::Cell::get);
            let ready =
                PREFILL_TAIL_WRITES_READY.with(|log| std::mem::take(&mut *log.borrow_mut()));
            let logits = last.ok_or_else(|| Error::from_reason("no prefill logits"))?;
            let after = lane_after_prefill(&mut inner, 1, &tokens, &logits)?;
            Ok((after, stored, ready, given))
        };
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(64)));
        let result = (|| -> Result<()> {
            let (want, one_call_tails, _, _) = arm(&one_call, false)?;
            let depths: Vec<usize> = want.4.iter().map(|state| state.0).collect();
            assert!(
                depths.contains(&144) && depths.iter().all(|&depth| depth > 64),
                "a state point at the second unit's end, none in the first: {depths:?}"
            );
            assert!(one_call_tails > 0, "the one call stored no tail");
            for (passes, case) in [(false, "alone"), (true, "passes")] {
                let (got, tails, ready, given) = arm(&units, passes)?;
                assert_eq!(
                    tails, one_call_tails,
                    "{case}: a tail where the one call stores one"
                );
                assert_eq!(
                    given,
                    [false, false, true],
                    "{case}: logits only at the prompt's end"
                );
                assert!(
                    !ready.is_empty() && ready.iter().all(|&ready| ready),
                    "{case}: pool writes evaluated at each tail's end: {ready:?}"
                );
                assert!(got.0 == want.0, "{case}: prefill logits");
                assert!(got.1 == want.1, "{case}: cells");
                assert!(got.2 == want.2, "{case}: ring");
                assert!(got.3 == want.3, "{case}: window");
                assert!(got.4 == want.4, "{case}: published states");
                assert_eq!(got.5, want.5, "{case}: booked rows");
                assert!(got.6 == want.6, "{case}: the next step's logits");
            }
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// A turn cancelled between its units, after a unit that ended short of
    /// the prompt at a state point (144) and gave no logits: a later request
    /// whose prompt extends it resumes at that state, and the states it
    /// publishes past it and its logits are a whole turn's of its prompt,
    /// which restores nothing.
    /// Mutation (red here): a short unit ending at a state point does not
    /// reach it.
    #[test]
    fn a_cancel_after_a_short_unit_leaves_a_state_that_restores_as_none() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_cancel_after_a_short_unit_leaves_a_state_that_restores_as_none",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_9231;
        let first = prompt(150);
        let longer = prompt(180);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(64)));
        let result = (|| -> Result<()> {
            let (reference, reference_logits) = whole_turn_states(seed, &longer)?;
            let mut inner = wide_draft_inner(seed, 40);
            let base = scheduled_prime(&mut inner, 1, &first)?;
            for (start, end) in [(0, 64), (64, 144)] {
                assert!(
                    scheduled_slice(&mut inner, 1, &first, &base, start, end)?.is_none(),
                    "a unit short of the prompt gives no logits"
                );
            }
            HybridSchedulerBackend::release_scheduled_cache(&mut inner, 1)?;
            HybridSchedulerBackend::scheduled_turn_closed(&mut inner, 1);
            let base = scheduled_prime(&mut inner, 4, &longer)?;
            assert_eq!(
                base.effective_cached_prefix_len(),
                144,
                "the request resumes at the state the short unit reached"
            );
            let logits = scheduled_last_slice(&mut inner, 4, &longer, &base, 144)?;
            let past: Vec<_> = draft_states(&inner, &longer)?
                .into_iter()
                .filter(|state| state.0 > 144)
                .collect();
            let expected: Vec<_> = reference
                .into_iter()
                .filter(|state| state.0 > 144)
                .collect();
            assert!(!expected.is_empty(), "a state past the resume");
            assert_eq!(past, expected, "the states past the resume");
            assert_eq!(bits(&logits)?, reference_logits, "the prefill logits");
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// One-layer passes over launches of 64 rows run the fused GDN prefill's
    /// row scan as often as the one call does, and give the one call's bits.
    /// Mutation: the GDN arm of `forward_paged_prefill` takes the tape route
    /// (no fused prefill) in passes only → red at "row scans".
    #[test]
    fn passes_of_64_row_launches_take_the_gdn_row_scan() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "passes_of_64_row_launches_take_the_gdn_row_scan",
        ) || !crate::models::qwen3_5::gated_delta::tests::fused_prefill_launches()
        {
            return Ok(());
        }
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_9202;
        let tokens = prompt(150);
        let arm = |passes: bool| -> Result<_> {
            let mut inner = wide_draft_inner(seed, 40);
            let base = pass_lane(&mut inner, 1, &tokens)?;
            // SAFETY: test-only thread-local counter switch.
            unsafe { mlx_sys::mlx_test_kquant_counting(true) };
            let logits = (|| -> Result<MxArray> {
                if !passes {
                    return scheduled_last_slice(&mut inner, 1, &tokens, &base, 0);
                }
                loop {
                    if let Some(logits) =
                        one_pass(&mut inner, 1, &tokens, &base, (0, tokens.len()), 0.0)?
                    {
                        return Ok(logits);
                    }
                }
            })();
            // SAFETY: NUL-terminated name; test-only counter.
            let scans =
                unsafe { mlx_sys::mlx_test_kquant_family_count(c"gdn_prefill_row_scan".as_ptr()) };
            // SAFETY: as above.
            unsafe { mlx_sys::mlx_test_kquant_counting(false) };
            let logits = logits?;
            let after = lane_after_prefill(&mut inner, 1, &tokens, &logits)?;
            Ok((after, scans))
        };
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(64)));
        let result = (|| -> Result<()> {
            let (want, lone_scans) = arm(false)?;
            let (got, pass_scans) = arm(true)?;
            assert!(lone_scans > 0, "the one call ran no row scan");
            assert_eq!(pass_scans, lone_scans, "row scans");
            assert!(got.0 == want.0, "prefill logits");
            assert!(got.1 == want.1, "cells");
            assert!(got.2 == want.2, "ring");
            assert!(got.3 == want.3, "window");
            assert!(got.4 == want.4, "published states");
            assert_eq!(got.5, want.5, "booked rows");
            assert!(got.6 == want.6, "the next step's logits");
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// A unit ended between passes (cancel, release, preemption) leaves its
    /// launch start: the booked rows and the cells as they were before the
    /// launch in flight, no state published past it, and the lane's window
    /// and ring back on the sequence; a preemption registers no row past
    /// it.
    /// Mutations, each → red: the launch in flight is not rolled back; a
    /// preemption registers before the roll back.
    #[test]
    fn a_unit_ended_between_passes_leaves_its_launch_start() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported(
            "a_unit_ended_between_passes_leaves_its_launch_start",
        ) {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let seed = 0x5_9210;
        let tokens = prompt(150);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            // The cells after two whole launches, from a lone unit of 32 rows.
            let mut lone = wide_draft_inner(seed, 40);
            let base = pass_lane(&mut lone, 1, &tokens)?;
            scheduled_slice(&mut lone, 1, &tokens, &base, 0, 32)?;
            HybridSchedulerBackend::activate_paged_seq(&mut lone, 1)?;
            let want_cells = cell_bits(lone.caches.as_ref().expect("caches"))?;
            for preempt in [false, true] {
                let mut inner = wide_draft_inner(seed, 40);
                let base = pass_lane(&mut inner, 1, &tokens)?;
                assert!(inner.config.num_layers > 1);
                let booked = |inner: &Qwen35Inner| {
                    inner
                        .paged_adapter
                        .as_ref()
                        .and_then(|adapter| adapter.current_token_count_for(1))
                };
                use crate::models::qwen3_5::paged_forward::PREFILL_TAILS_STORED;
                PREFILL_TAILS_STORED.with(|count| count.set(0));
                // Passes until the third launch is booked: its first layer ran.
                while booked(&inner) != Some(48) {
                    assert!(
                        one_pass(&mut inner, 1, &tokens, &base, (0, tokens.len()), 0.0)?.is_none()
                    );
                }
                assert!(
                    PREFILL_TAILS_STORED.with(std::cell::Cell::get) > 0,
                    "the launches before the one in flight stored no tail"
                );
                if preempt {
                    HybridSchedulerBackend::preempt_scheduled_cache(
                        &mut inner,
                        1,
                        0,
                        crate::engine::scheduler::PreemptionMode::Recompute,
                    )?;
                    assert!(inner.prefill_runs.is_empty());
                    let adapter = inner.paged_adapter.as_ref().expect("pool");
                    let block = adapter.block_size() as usize;
                    let registered = inner
                        .dflash2_states
                        .store
                        .chain(crate::models::qwen3_5::dflash2_state::store::Lineage::text(
                            &tokens, 0,
                        ))
                        .iter()
                        .take_while(|&&hash| adapter.prefix_cached(hash))
                        .count()
                        * block;
                    assert_eq!(
                        registered,
                        32 - 32 % block,
                        "a preemption registers the launches that ended, not the one in flight"
                    );
                    continue;
                }
                inner.end_prefill_run(1)?;
                assert!(inner.prefill_runs.is_empty());
                assert_eq!(booked(&inner), Some(32), "back at the launch start");
                HybridSchedulerBackend::activate_paged_seq(&mut inner, 1)?;
                assert!(
                    cell_bits(inner.caches.as_ref().expect("caches"))? == want_cells,
                    "the cells of the launch start"
                );
                assert!(
                    draft_states(&inner, &tokens)?
                        .iter()
                        .all(|state| state.0 <= 32),
                    "no state past the launch start"
                );
                assert!(inner.dflash2_states.seq_windows.contains_key(&1));
                assert!(inner.dflash2_states.seq_lanes.contains_key(&1));
                assert_eq!(
                    inner.dflash2_states.seq_lanes[&1].ring.logical_len(),
                    32,
                    "the ring at the launch start"
                );
            }
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    /// While a unit is held between passes its lane's window, ring and
    /// decode lane, and its launch's hidden rows, taps and the launch-start
    /// cells of the layers it ran, stay on the governor's `states` account
    /// and the lane stays a speculation owner; the unit's end leaves both
    /// as they were, and a lone unit holds nothing. Mutations, each → red: the held
    /// unit's bytes are not counted; its lane is not an owner; the launch
    /// is not counted; cells of layers not run counted too; a tap that is
    /// the hidden rows counted again.
    #[test]
    fn a_held_unit_keeps_its_lane_on_the_books() -> Result<()> {
        if crate::test_support::paged_prefill_unsupported("a_held_unit_keeps_its_lane_on_the_books")
        {
            return Ok(());
        }
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let _ctx = StreamContext::new(Stream::generation());
        let tokens = prompt(150);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        let result = (|| -> Result<()> {
            let mut inner = wide_draft_inner(0x5_9220, 40);
            let base = pass_lane(&mut inner, 1, &tokens)?;
            scheduled_slice(&mut inner, 1, &tokens, &base, 0, 64)?;
            assert!(inner.prefill_runs.is_empty());
            assert_eq!(
                crate::models::qwen3_5::dflash2_turn::DFlash2Target::held_prefill_bytes(&inner),
                0,
                "a lone unit holds nothing"
            );
            let books = |inner: &Qwen35Inner| {
                (
                    HybridSchedulerBackend::memory_accounts(inner)
                        .expect("accounts")
                        .states,
                    HybridSchedulerBackend::scheduled_speculation_owners(inner),
                )
            };
            let before = books(&inner);
            assert_eq!(before.1, 1);
            for _ in 0..2 {
                assert!(one_pass(&mut inner, 1, &tokens, &base, (64, 150), 0.0)?.is_none());
            }
            let rows = inner.prefill_runs[&1]
                .driver
                .in_flight_rows()
                .expect("a launch in flight");
            assert!(!inner.dflash2_states.seq_lanes.contains_key(&1));
            let ran = 2usize;
            HybridSchedulerBackend::activate_paged_seq(&mut inner, 1)?;
            let (ran_cells, unrun_cells) = {
                let caches = inner.caches.as_ref().expect("caches");
                (
                    crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache::linear_state_bytes(
                        &caches[..ran],
                    ),
                    crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache::linear_state_bytes(
                        &caches[ran..],
                    ),
                )
            };
            HybridSchedulerBackend::park_active_scheduled_recurrent(&mut inner)?;
            let hidden = u64::from(rows)
                * inner.config.hidden_size as u64
                * crate::models::qwen3_5::dflash2_turn::DFlash2Target::embedding_dtype(&inner)?
                    .byte_size() as u64;
            let taps = inner
                .dflash2
                .as_ref()
                .expect("draft")
                .config
                .target_layers
                .iter()
                .filter(|&&layer| layer + 1 < ran)
                .count() as u64;
            let launch = hidden * (1 + taps) + ran_cells;
            assert!(ran_cells > 0 && unrun_cells > hidden);
            let held = books(&inner);
            assert!(
                held.0 >= before.0 + launch && held.0 < before.0 + launch + hidden,
                "held: {} bytes, before: {} bytes, launch: {launch} bytes \
                 (hidden {hidden}, taps {taps}, cells of layers run {ran_cells}), \
                 cells of layers not run: {unrun_cells} bytes",
                held.0,
                before.0
            );
            assert_eq!(held.1, before.1, "speculation owners");
            inner.end_prefill_run(1)?;
            assert_eq!(books(&inner), before, "the unit's end");
            Ok(())
        })();
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        result
    }

    mod wave_edges;

    mod lane_timeline;

    mod lane_sessions {
        use super::*;

        pub(super) type State = crate::engine::hybrid_scheduler::HybridSchedulerState<Qwen35Inner>;
        pub(super) type Reply =
            tokio::sync::oneshot::Receiver<Result<crate::engine::types::ChatResult>>;
        pub(super) type Sender = tokio::sync::mpsc::UnboundedSender<
            crate::engine::model_command::ModelCommand<
                <Qwen35Inner as crate::engine::hybrid_scheduler::HybridSchedulerBackend>::FamilyCommand,
            >,
        >;

        fn skip(name: &str) -> bool {
            crate::test_support::paged_prefill_unsupported(name)
                || !super::super::tests::metal_is_default_device()
        }

        pub(super) fn start_cmd(
            owner: &str,
            body: &[u32],
            n: i32,
            tweak: impl FnOnce(&mut crate::engine::types::ChatConfig),
        ) -> Result<(
            crate::engine::cmd::ChatCmd,
            Reply,
            std::sync::Arc<std::sync::atomic::AtomicBool>,
        )> {
            let (mut command, reply, cancel) =
                super::super::tests::session_start_with(owner, body, n, true)?;
            if let crate::engine::cmd::ChatCmd::SessionStart { config, .. } = &mut command {
                tweak(config);
            }
            Ok((command, reply, cancel))
        }

        pub(super) fn send(sender: &Sender, command: crate::engine::cmd::ChatCmd) -> Result<()> {
            sender
                .send(crate::engine::model_command::ModelCommand::from_chat(
                    command,
                ))
                .map_err(|_| Error::from_reason("the command channel closed"))
        }

        /// Sends `commands`, then drives until `on_pass` says done. `on_pass`
        /// may send more through the sender it is handed; it must drop the
        /// sender (set it to `None`) once it has nothing more to send.
        pub(super) fn run(
            state: &mut State,
            commands: Vec<crate::engine::cmd::ChatCmd>,
            keep_sender: bool,
            mut on_pass: impl FnMut(&mut State, &mut Option<Sender>) -> Result<bool>,
        ) -> Result<()> {
            use crate::model_thread::LoopControl;
            let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
            for command in commands {
                send(&sender, command)?;
            }
            let mut sender = keep_sender.then_some(sender);
            for _ in 0..2048 {
                if !matches!(state.drive(&mut receiver), LoopControl::Continue) {
                    return Err(Error::from_reason("the scheduler stopped"));
                }
                if on_pass(state, &mut sender)? {
                    return Ok(());
                }
            }
            Err(Error::from_reason("the sessions did not finish"))
        }

        pub(super) fn take(
            reply: &mut Reply,
            slot: &mut Option<Result<crate::engine::types::ChatResult>>,
        ) {
            if slot.is_none()
                && let Ok(result) = reply.try_recv()
            {
                *slot = Some(result);
            }
        }

        pub(super) fn clean(state: &State, tag: &str) {
            let states = &state.inner.dflash2_states;
            assert!(states.seq_lanes.is_empty(), "{tag}: a lane is left");
            assert!(states.lane_batch.is_none(), "{tag}: an open wave is left");
            assert!(states.peer_cells.is_empty(), "{tag}: peer cells are left");
            assert!(
                states.lane_decode_limit.is_none(),
                "{tag}: decode limit left"
            );
            assert!(states.turns.is_empty(), "{tag}: a turn is left");
        }

        /// A lane that ends while another lane's sequence is the active one
        /// (its own cells parked) still folds its pending rows: the owner's
        /// parked cell is its whole turn's settled cell.
        #[test]
        fn a_lane_that_ends_parked_leaves_settled_cells() -> Result<()> {
            if skip("a_lane_that_ends_parked_leaves_settled_cells") {
                return Ok(());
            }
            use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
            let _ctx = StreamContext::new(Stream::generation());
            let seed = 0x5_8350;
            let [(_, first), (_, second)] = lane_prompts();
            let (a_n, b_n) = (12, 40);
            let (a_alone, a_cells) = whole_turn_session(seed, &first, a_n)?;
            let (b_alone, b_cells) = whole_turn_session(seed, &second, b_n)?;
            let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
            let mut a_parked_with_lane = 0;
            let (mut results, saw_lane) = drive_sessions(
                &mut state,
                &[("a", &first, a_n), ("b", &second, b_n)],
                true,
                |state, _| {
                    if let Some(&seq) = state.owner_sequences.get("a")
                        && state.inner.dflash2_states.seq_lanes.contains_key(&seq)
                        && state.inner.active_scheduled_seq != Some(seq)
                    {
                        a_parked_with_lane += 1;
                    }
                },
            )?;
            let b = results.pop().expect("b")?;
            let a = results.pop().expect("a")?;
            assert!(saw_lane);
            assert!(a_parked_with_lane > 0, "a's lane lived while a was parked");
            assert_eq!(a.text, a_alone.text, "a's tokens");
            assert_eq!(b.text, b_alone.text, "b's tokens");
            assert_eq!(drafted(&a), drafted(&a_alone), "a's cycles");
            assert_eq!(drafted(&b), drafted(&b_alone), "b's cycles");
            let checkpoint = checkpoint_of(&state.inner)?;
            assert_eq!(checkpoint.0, b_cells.0, "b ended last");
            assert!(checkpoint.1 == b_cells.1, "b's settled cells");
            for (owner, cells) in [("a", &a_cells.1), ("b", &b_cells.1)] {
                let seq = state.owner_sequences[owner];
                HybridSchedulerBackend::activate_scheduled_recurrent(&mut state.inner, seq)?;
                assert!(
                    cell_bits(state.inner.caches.as_ref().expect("caches"))? == *cells,
                    "{owner}: the parked owner cell is its whole turn's settled cell"
                );
            }
            clean(&state, "parked");
            Ok(())
        }

        /// The app's sampling (Qwen defaults: T 0.6 / 1.0, top-p 0.95, top-k
        /// 20, presence penalty 1.5) on two overlapping lanes: both finish,
        /// both drafted, nothing is left.
        #[test]
        fn sampled_sessions_overlap_on_lanes() -> Result<()> {
            if skip("sampled_sessions_overlap_on_lanes") {
                return Ok(());
            }
            let _ctx = StreamContext::new(Stream::generation());
            let seed = 0x5_8350;
            let [(_, first), (_, second)] = lane_prompts();
            for (tag, temperature, presence, accepting) in [
                ("coding", 0.6, 0.0, false),
                ("general", 1.0, 1.5, false),
                ("coding-accepting", 0.6, 0.0, true),
                ("general-accepting", 1.0, 1.5, true),
            ] {
                let inner = if accepting {
                    accepting_inner(seed)?
                } else {
                    tiny_paged_int8_dflash_inner(seed)
                };
                let mut state = scheduler_on(inner)?;
                let tweak = move |config: &mut crate::engine::types::ChatConfig| {
                    config.temperature = Some(temperature);
                    config.top_p = Some(0.95);
                    config.top_k = Some(20);
                    config.presence_penalty = Some(presence);
                };
                let (a_cmd, mut a_reply, _) = start_cmd("a", &first, 40, tweak)?;
                let (b_cmd, mut b_reply, _) = start_cmd("b", &second, 24, tweak)?;
                let (mut a, mut b) = (None, None);
                let mut most = 0;
                run(&mut state, vec![a_cmd, b_cmd], false, |state, _| {
                    most = most.max(state.inner.dflash2_states.seq_lanes.len());
                    take(&mut a_reply, &mut a);
                    take(&mut b_reply, &mut b);
                    Ok(a.is_some() && b.is_some())
                })?;
                let a = a.expect("a")?;
                let b = b.expect("b")?;
                assert_eq!(most, 2, "{tag}: two live lanes");
                assert_eq!(a.num_tokens, 40, "{tag}: a's tokens");
                assert_eq!(b.num_tokens, 24, "{tag}: b's tokens");
                assert!(drafted(&a) > 0 && drafted(&b) > 0, "{tag}: both drafted");
                clean(&state, tag);
            }
            Ok(())
        }

        /// A greedy lane beside a sampled lane (the wave leaves the all-greedy
        /// accept): the greedy session keeps its whole turn's tokens, cycles
        /// and accepted drafts.
        #[test]
        fn a_greedy_lane_beside_a_sampled_lane_keeps_its_tokens() -> Result<()> {
            if skip("a_greedy_lane_beside_a_sampled_lane_keeps_its_tokens") {
                return Ok(());
            }
            let _ctx = StreamContext::new(Stream::generation());
            let seed = 0x5_8350;
            let [(_, first), (_, second)] = lane_prompts();
            for accepting in [false, true] {
                let build = || -> Result<Qwen35Inner> {
                    if accepting {
                        accepting_inner(seed)
                    } else {
                        Ok(tiny_paged_int8_dflash_inner(seed))
                    }
                };
                let mut whole = scheduler_on(build()?)?;
                whole.serve_whole_turns();
                let (mut results, _) =
                    drive_sessions(&mut whole, &[("a", &first, 40)], true, |_, _| {})?;
                let a_alone = results.remove(0)?;

                let mut state = scheduler_on(build()?)?;
                let (a_cmd, mut a_reply, _) = start_cmd("a", &first, 40, |_| {})?;
                let (b_cmd, mut b_reply, _) = start_cmd("b", &second, 40, |config| {
                    config.temperature = Some(1.0);
                    config.top_p = Some(0.95);
                    config.top_k = Some(20);
                    config.presence_penalty = Some(1.5);
                })?;
                let (mut a, mut b) = (None, None);
                let mut both = 0;
                run(&mut state, vec![a_cmd, b_cmd], false, |state, _| {
                    if state.inner.dflash2_states.seq_lanes.len() == 2 {
                        both += 1;
                    }
                    take(&mut a_reply, &mut a);
                    take(&mut b_reply, &mut b);
                    Ok(a.is_some() && b.is_some())
                })?;
                let a = a.expect("a")?;
                let b = b.expect("b")?;
                assert!(both > 4, "accepting {accepting}: the lanes overlapped");
                assert_eq!(b.num_tokens, 40, "accepting {accepting}: b's tokens");
                assert_eq!(a.text, a_alone.text, "accepting {accepting}: a's tokens");
                assert_eq!(
                    drafted(&a),
                    drafted(&a_alone),
                    "accepting {accepting}: a's cycles"
                );
                assert_eq!(
                    accepted(&a),
                    accepted(&a_alone),
                    "accepting {accepting}: a's accepted drafts"
                );
                clean(&state, "greedy beside sampled");
            }
            Ok(())
        }

        /// A greedy lane and a sampled lane on the device rows verify in one
        /// step whose sampled-rows eval fails. MLX throws a failed command's
        /// error once, so both turns fail with that error: the greedy lane
        /// commits and streams no token read from that step's logits.
        #[test]
        fn a_failed_sampled_rows_eval_fails_the_greedy_lane_of_the_step() -> Result<()> {
            if skip("a_failed_sampled_rows_eval_fails_the_greedy_lane_of_the_step") {
                return Ok(());
            }
            use crate::engine::dspark_turn::TEST_SAMPLED_ROWS;
            // A failed command that is not out of memory: the step's other
            // rows complete, no model reset fails them.
            const TIMEOUT: &str = "[METAL] Command buffer execution failed: Caused GPU Timeout \
                                   Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)";
            assert!(!crate::engine::governor::is_gpu_out_of_memory(TIMEOUT));
            let _ctx = StreamContext::new(Stream::generation());
            let seed = 0x5_8350;
            let [(_, first), (_, second)] = lane_prompts();
            let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
            let (a_cmd, mut a_reply, _) = start_cmd("a", &first, 40, |_| {})?;
            let (b_cmd, mut b_reply, _) = start_cmd("b", &second, 40, |config| {
                config.temperature = Some(1.0);
                config.top_p = Some(0.95);
                config.top_k = Some(20);
            })?;
            let generated = |state: &mut State, owner: &str| {
                state
                    .owner_sequences
                    .get(owner)
                    .copied()
                    .and_then(|seq| state.turn_mut(seq))
                    .map_or(0, |turn| turn.payload.generated_tokens.len())
            };
            TEST_SAMPLED_ROWS.with(|c| c.set(0));
            let (mut a, mut b) = (None, None);
            let mut armed = None;
            run(&mut state, vec![a_cmd, b_cmd], false, |state, _| {
                take(&mut a_reply, &mut a);
                take(&mut b_reply, &mut b);
                if armed.is_none()
                    && state.inner.dflash2_states.seq_lanes.len() == 2
                    && generated(state, "a") > 0
                    && generated(state, "b") > 0
                    && TEST_SAMPLED_ROWS.with(|c| c.get()) > 0
                {
                    armed = Some(crate::array::fail_native_call_at(
                        "scheduled_verify_sampled_rows",
                        TIMEOUT,
                    ));
                }
                Ok(a.is_some() && b.is_some())
            })?;
            assert!(armed.is_some(), "the two lanes decoded together");
            for (owner, reply) in [("a", a), ("b", b)] {
                let error = match reply.expect("a reply") {
                    Ok(result) => panic!(
                        "{owner}: the turn went on past the failed step: {} tokens",
                        result.num_tokens
                    ),
                    Err(error) => error,
                };
                assert!(
                    error.reason.contains("scheduled_verify_sampled_rows")
                        && error.reason.contains(TIMEOUT),
                    "{owner}: {}",
                    error.reason
                );
            }
            clean(&state, "failed sampled rows");
            Ok(())
        }

        /// Two streamed sessions (the app streams) on two lanes: each
        /// terminal chunk carries its whole turn's text.
        #[test]
        fn streamed_sessions_on_lanes_give_the_whole_turns_text() -> Result<()> {
            if skip("streamed_sessions_on_lanes_give_the_whole_turns_text") {
                return Ok(());
            }
            let _ctx = StreamContext::new(Stream::generation());
            let seed = 0x5_8350;
            let [(_, first), (_, second)] = lane_prompts();
            let n = 24;
            let (a_alone, _) = whole_turn_session(seed, &first, n)?;
            let (b_alone, _) = whole_turn_session(seed, &second, n)?;
            let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
            let stream = |owner: &str, body: &[u32]| -> Result<_> {
                let (command, _, cancelled) =
                    super::super::tests::session_start_with(owner, body, n, true)?;
                let crate::engine::cmd::ChatCmd::SessionStart {
                    messages, config, ..
                } = command
                else {
                    return Err(Error::from_reason("not a start"));
                };
                let (stream_tx, rx) = crate::model_thread::stream_channel(4096);
                Ok((
                    crate::engine::cmd::ChatCmd::StreamSessionStart {
                        messages,
                        config,
                        stream_tx,
                        cancelled,
                    },
                    rx,
                ))
            };
            let (a_cmd, mut a_rx) = stream("a", &first)?;
            let (b_cmd, mut b_rx) = stream("b", &second)?;
            let mut done: [Option<crate::engine::types::ChatStreamChunk>; 2] = [None, None];
            let mut deltas = [String::new(), String::new()];
            let mut most = 0;
            run(&mut state, vec![a_cmd, b_cmd], false, |state, _| {
                most = most.max(state.inner.dflash2_states.seq_lanes.len());
                for (index, rx) in [&mut a_rx, &mut b_rx].into_iter().enumerate() {
                    while let Ok(chunk) = rx.try_recv() {
                        let chunk = chunk?;
                        if chunk.done {
                            done[index] = Some(chunk);
                        } else {
                            deltas[index].push_str(&chunk.text);
                        }
                    }
                }
                Ok(done.iter().all(Option::is_some))
            })?;
            assert_eq!(most, 2, "two live lanes");
            for (index, alone) in [&a_alone, &b_alone].into_iter().enumerate() {
                let chunk = done[index].as_ref().expect("terminal chunk");
                assert_eq!(
                    chunk.text, alone.text,
                    "session {index}: the whole turn's text"
                );
                assert_eq!(chunk.num_tokens, Some(alone.num_tokens));
            }
            clean(&state, "stream");
            Ok(())
        }

        /// b is cancelled inside its prefill (its lane begun, never verified)
        /// while a decodes on its lane: a keeps its whole turn's tokens and
        /// cycles, b's lane is gone.
        #[test]
        fn a_cancel_during_prefill_leaves_the_decoder_exact() -> Result<()> {
            if skip("a_cancel_during_prefill_leaves_the_decoder_exact") {
                return Ok(());
            }
            let _ctx = StreamContext::new(Stream::generation());
            let seed = 0x5_8350;
            let a_body = prompt(150);
            let b_body: Vec<u32> = (0..2_600u32).map(|i| (i * 5 + 1) % VOCAB).collect();
            let (a_alone, _) = two_lane_alone(seed, &a_body, 40)?;
            let mut state = two_lane_state(seed, None)?;
            let (a_cmd, mut a_reply, _) = start_cmd("a", &a_body, 40, |_| {})?;
            let (b_cmd, mut b_reply, b_cancel) = start_cmd("b", &b_body, 24, |_| {})?;
            let mut b_cmd = Some(b_cmd);
            let (mut a, mut b) = (None, None);
            let mut cancelled_at = None;
            run(&mut state, vec![a_cmd], true, |state, sender| {
                take(&mut a_reply, &mut a);
                take(&mut b_reply, &mut b);
                let a_generated = state
                    .owner_sequences
                    .get("a")
                    .copied()
                    .and_then(|seq| state.turn_mut(seq))
                    .map_or(0, |turn| turn.payload.generated_tokens.len());
                if (a_generated >= 2 || a.is_some())
                    && let Some(command) = b_cmd.take()
                    && let Some(sender) = sender.take()
                {
                    send(&sender, command)?;
                }
                if cancelled_at.is_none()
                    && let Some(&seq) = state.owner_sequences.get("b")
                {
                    let lane = state
                        .inner
                        .dflash2_states
                        .seq_lanes
                        .get(&seq)
                        .map(|lane| lane.gdn.is_some());
                    if let Some(turn) = state.turn_mut(seq)
                        && turn.num_computed_tokens > 0
                        && turn.num_computed_tokens < turn.prompt_tokens
                        && lane == Some(false)
                    {
                        b_cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                        cancelled_at =
                            Some((turn.num_computed_tokens, turn.prompt_tokens, a.is_none()));
                    }
                }
                Ok(a.is_some() && b.is_some())
            })?;
            let (computed, prompt, a_decoding) =
                cancelled_at.expect("b was cancelled inside its prefill");
            assert!(computed < prompt);
            assert!(a_decoding, "a was still decoding");
            assert!(
                b.expect("b")
                    .is_err_and(|error| error.reason == "chat session cancelled"),
                "b ended cancelled"
            );
            let a = a.expect("a")?;
            assert_eq!(a.text, a_alone.text, "a's whole turn's tokens");
            assert_eq!(drafted(&a), drafted(&a_alone), "a's cycles");
            assert_eq!(accepted(&a), accepted(&a_alone), "a's accepted drafts");
            clean(&state, "cancel-in-prefill");
            assert!(state.inner.dflash2_states.seq_windows.is_empty());
            Ok(())
        }

        /// The app's request shape: a second request of ANOTHER owner extends
        /// a finished first request's prompt while that request is still the
        /// first owner's live sequence (no restart). It resumes where the
        /// whole turn resumes and gives the whole turn's tokens.
        #[test]
        fn a_second_owner_extends_a_live_first_owner() -> Result<()> {
            if skip("a_second_owner_extends_a_live_first_owner") {
                return Ok(());
            }
            let _ctx = StreamContext::new(Stream::generation());
            let seed = 0x5_8350;
            let a_body = prompt(150);
            let longer = prompt(180);
            let mut whole = two_lane_state(seed, None)?;
            whole.serve_whole_turns();
            let (mut results, _) =
                drive_sessions(&mut whole, &[("a", &a_body, 24)], true, |_, _| {})?;
            let a_whole = results.remove(0)?;
            let (mut results, saw) =
                drive_sessions(&mut whole, &[("c", &longer, 24)], true, |_, _| {})?;
            let c_whole = results.remove(0)?;
            assert!(!saw);

            let mut state = two_lane_state(seed, None)?;
            let (mut results, saw) =
                drive_sessions(&mut state, &[("a", &a_body, 24)], true, |_, _| {})?;
            let a = results.remove(0)?;
            assert!(saw);
            let a_seq = state.owner_sequences["a"];
            assert!(
                state
                    .inner
                    .paged_adapter
                    .as_ref()
                    .expect("pool")
                    .request_tokens_for(a_seq)
                    .is_some(),
                "a stays its owner's live sequence"
            );
            let (mut results, saw) =
                drive_sessions(&mut state, &[("c", &longer, 24)], true, |_, _| {})?;
            let c = results.remove(0)?;
            assert!(saw, "c ran on a lane");
            assert_eq!(a.text, a_whole.text, "a's tokens");
            assert_eq!(c.cached_tokens, c_whole.cached_tokens, "c's resume depth");
            assert_eq!(c.text, c_whole.text, "c's tokens");
            assert_eq!(drafted(&c), drafted(&c_whole), "c's cycles");
            assert_eq!(accepted(&c), accepted(&c_whole), "c's accepted drafts");
            clean(&state, "second owner");
            Ok(())
        }

        pub(super) fn words(tokens: &[u32]) -> String {
            tokens
                .iter()
                .map(|id| format!("t{id}"))
                .collect::<Vec<_>>()
                .join(" ")
        }

        /// One owner's second turn (a live continuation) with its draft on:
        /// returns turn 1, turn 2 and whether turn 2 ran on a lane.
        #[allow(clippy::type_complexity)]
        fn two_turns(
            seed: u64,
            speculate: bool,
            whole_turns: bool,
        ) -> Result<(
            crate::engine::types::ChatResult,
            crate::engine::types::ChatResult,
            bool,
            usize,
        )> {
            let body = prompt(40);
            let second: Vec<u32> = (0..12u32).map(|i| (i * 5 + 1) % VOCAB).collect();
            let mut state = two_lane_state(seed, None)?;
            if whole_turns {
                state.serve_whole_turns();
            }
            let (mut results, _) =
                drive_sessions_with(&mut state, &[("a", &body, 16, speculate)], |_, _| {})?;
            let first = results.remove(0)?;
            let history = if whole_turns {
                state.inner.cached_token_history.clone()
            } else {
                let seq = state.owner_sequences["a"];
                state
                    .inner
                    .paged_adapter
                    .as_ref()
                    .expect("pool")
                    .request_tokens_for(seq)
                    .expect("a stays live")
                    .to_vec()
            };
            let (command, _, cancelled) =
                super::super::tests::session_start_with("a", &[], 16, speculate)?;
            let crate::engine::cmd::ChatCmd::SessionStart { config, .. } = command else {
                return Err(Error::from_reason("not a start"));
            };
            let message = |content: String| -> Result<crate::tokenizer::ChatMessage> {
                serde_json::from_value(serde_json::json!({"role": "user", "content": content}))
                    .map_err(|error| Error::from_reason(error.to_string()))
            };
            let (reply, mut result) = tokio::sync::oneshot::channel();
            let command = crate::engine::cmd::ChatCmd::SessionContinue {
                messages: vec![message(words(&history))?, message(words(&second))?],
                config,
                reply,
                cancelled,
            };
            let mut done = None;
            let mut saw_lane = false;
            run(&mut state, vec![command], false, |state, _| {
                saw_lane |= !state.inner.dflash2_states.seq_lanes.is_empty();
                take(&mut result, &mut done);
                Ok(done.is_some())
            })?;
            if !whole_turns {
                clean(&state, "continue");
            }
            Ok((first, done.expect("turn 2")?, saw_lane, history.len()))
        }

        /// A same-owner continuation (a delta turn over the owner's live
        /// sequence) drafts on a lane begun past position 0: it resumes where
        /// the whole turn resumes and gives the whole turn's tokens, cycles
        /// and accepted drafts.
        #[test]
        fn the_same_owner_continues_on_a_lane() -> Result<()> {
            if skip("the_same_owner_continues_on_a_lane") {
                return Ok(());
            }
            let _ctx = StreamContext::new(Stream::generation());
            let seed = 0x5_8350;
            let (ar_first, ar_second, ar_lane, _) = two_turns(seed, false, false)?;
            let (first, second, lane, history) = two_turns(seed, true, false)?;
            let (w_first, w_second, w_lane, w_history) = two_turns(seed, true, true)?;
            assert!(!ar_lane && !w_lane && lane, "turn 2 ran on a lane");
            assert!(drafted(&second) > 0, "turn 2 drafted");
            assert_eq!(history, w_history, "the history turn 2 continues");
            assert!(second.cached_tokens as usize <= history);
            assert_eq!(
                second.cached_tokens, w_second.cached_tokens,
                "the whole turn's resume depth"
            );
            assert_eq!(first.text, w_first.text, "turn 1's whole turn's tokens");
            assert_eq!(second.text, w_second.text, "turn 2's whole turn's tokens");
            assert_eq!(drafted(&second), drafted(&w_second), "turn 2's cycles");
            assert_eq!(accepted(&second), accepted(&w_second), "turn 2's accepted");
            assert_eq!(second.cached_tokens, ar_second.cached_tokens);
            assert_eq!(first.text, ar_first.text, "turn 1 against scheduled AR");
            assert_eq!(second.text, ar_second.text, "turn 2 against scheduled AR");
            Ok(())
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
#[path = "dflash2_overlap_tests.rs"]
mod overlap_tests;
