//! DFlash2 whole-turn integration for dense Qwen3.8.

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::engine::backend::{DsparkBackend, DsparkProposal, DsparkStepper, DsparkVerifyOutput};
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
    /// The lane's two GDN cells and pending tape: each verify folds the
    /// last commit's retained rows, each commit swaps parity. Lent to the
    /// target across the verify's claim ([`lend_lane`]).
    gdn_blobs: Option<GdnStateBlobs>,
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
/// verify graph and `DecodeTensor::GdnTape`).
pub(crate) fn pool_lane_transient(
    config: &super::config::Qwen3_5Config,
    draft_taps: Option<usize>,
    window: u32,
) -> u64 {
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
            prefill.saturating_add(tapped).max(
                cycle
                    .saturating_add(held)
                    .saturating_add(pending)
                    .saturating_add(crate::cache_limit::decode_cache_limit(cycle)),
            )
        }
        None => prefill.max(dflash2_cycle_transient(config, 0, window).total()),
    }
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
        if self.tape.is_some() || self.tapped.is_some() || self.verified_ids.is_some() {
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
        let (path, _) = draft.propose(
            &self.inner.embedding,
            self.inner.lm_head.as_ref(),
            &mut self.context,
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
            super::dflash2::SelectorPath::Device(ids) => DraftedProposals::Device(ids).host(),
        }
    }

    /// The verify on the pool: the 8 rows booked (Splash `Engine::workEndOf`
    /// for a decode command: committed + 8), stored and attended, the
    /// driver's `ids.len()` logits rows returned.
    fn paged_verify(&mut self, ids: &[u32]) -> Result<DsparkVerifyOutput> {
        use super::dflash2::DFLASH2_VERIFY_ROWS;
        if ids.is_empty() {
            return Err(Error::from_reason(
                "Qwen3.8 DFlash2 verify block must not be empty",
            ));
        }
        self.ensure_clean("verify")?;
        let drafted = match self.paged.drafted.take() {
            Some(drafted) => drafted.host()?,
            None => self.draft_tail(ids[0])?,
        };
        let full = full_verify_ids(ids, &drafted)?;
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
        let ticket =
            crate::engine::spec_paged::SpecPagedCache::open_core_write_cycle(self, seq, full.len())
                .map_err(Error::from_reason)?;
        let mut tape = Vec::new();
        let phase_time = std::env::var("MLX_DFLASH2_PHASE_TIME").is_ok();
        let t0 = std::time::Instant::now();
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
                        &full,
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
        self.paged.open_cycle = Some(ticket);
        self.tape = Some(tape);
        self.tapped = Some(taps);
        self.verified_ids = Some(ids.to_vec());
        let logits = logits.slice_axis(1, 0, ids.len() as i64)?;
        if phase_time {
            eprintln!("[dflash2-phase] verify-build: {:?}", t0.elapsed());
            let t = std::time::Instant::now();
            MxArray::eval_arrays(&[&logits])?;
            eprintln!("[dflash2-phase] verify: {:?}", t.elapsed());
        }
        Ok(DsparkVerifyOutput { logits })
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
        // Every cycle drafts the whole block (Splash `Runtime::decodeAsync`:
        // one physical graph); the driver gets its cap.
        let (path, draft_sparse_dists) = draft.propose(
            &self.inner.embedding,
            self.inner.lm_head.as_ref(),
            &mut self.context,
            anchor_id,
            super::dflash2::DFLASH2_DRAFT_PROPOSALS,
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
        let mut proposal = DsparkProposal {
            draft_ids,
            device_draft_ids,
            draft_dists: Vec::new(),
            draft_sparse_dists,
            keep_probabilities: None,
        };
        self.paged.drafted = Some(match &proposal.device_draft_ids {
            Some(ids) => DraftedProposals::Device(ids.clone()),
            None => {
                DraftedProposals::Host(proposal.draft_ids.iter().map(|&id| id as u32).collect())
            }
        });
        proposal.truncate(max_len);
        Ok(proposal)
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        self.paged_verify(verify_ids)
    }

    /// The pool records host ids before its write, so the device ids are
    /// read back here.
    fn verify_device(&mut self, verify_ids: &MxArray) -> Result<DsparkVerifyOutput> {
        let ids = DraftedProposals::Device(verify_ids.clone()).host()?;
        self.paged_verify(&ids)
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
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
        self.gdn_blobs
            .as_mut()
            .ok_or_else(missing_cells)?
            .retain(&tape, keep)?;
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
        // Pack the prefill's per-layer GDN states once per turn (two concats)
        // into the lane's current cell; nothing is pending after a prefill.
        let tiled = self.config.qwen35_gguf_gdn_layout.as_deref() == Some("tiled");
        let caches = self
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?;
        let gdn_blobs = GdnStateBlobs::pack(caches, tiled)?.ok_or_else(|| {
            Error::from_reason("Qwen3.8 DFlash2 needs packed GDN state (bf16 conv, f32 recurrent)")
        })?;
        gdn_blobs.apply_views(caches)?;
        let seq = self
            .paged_adapter
            .as_ref()
            .and_then(
                crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter::active_seq_id,
            )
            .ok_or_else(|| {
                Error::from_reason("Qwen3.8 DFlash2 paged turn has no active request")
            })?;
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
        if self.active_scheduled_seq.is_none() {
            self.serial_decode = Some(crate::models::qwen3_5::dflash2_turn::SerialDecode::DFlash2);
        }
        Ok(Qwen35DFlash2Stepper {
            inner: self,
            context: state.context,
            next_position: state.next_position,
            tap_layers,
            tape: None,
            tapped: None,
            verified_ids: None,
            gdn_blobs: Some(gdn_blobs),
            paged,
        })
    }
}

impl Drop for Qwen35DFlash2Stepper<'_> {
    fn drop(&mut self) {
        self.inner.serial_decode = None;
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
        let vocab = (0..16)
            .map(|i| format!("\"t{i}\": {i}"))
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
        use crate::engine::cmd::ChatCmd;
        use crate::engine::model_command::ModelCommand;
        use crate::model_thread::LoopControl;
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let mut replies = Vec::new();
        for (owner, body) in starts {
            let content = body
                .iter()
                .map(|id| format!("t{id}"))
                .collect::<Vec<_>>()
                .join(" ");
            let message: crate::tokenizer::ChatMessage =
                serde_json::from_value(serde_json::json!({"role": "user", "content": content}))
                    .map_err(|error| Error::from_reason(error.to_string()))?;
            let (reply, result) = tokio::sync::oneshot::channel();
            let command = ChatCmd::SessionStart {
                messages: vec![message],
                config: crate::engine::types::ChatConfig {
                    max_new_tokens: Some(max_new_tokens),
                    temperature: Some(0.0),
                    enable_mtp: Some(false),
                    max_consecutive_tokens: Some(0),
                    max_ngram_repeats: Some(0),
                    reasoning_effort: Some("none".to_string()),
                    cache_owner_id: Some((*owner).to_string()),
                    ..Default::default()
                },
                reply,
                cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            };
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
mod paged_dflash2_tests {
    use super::*;
    use crate::engine::backend::{PagedBackend, PagedPrefix};
    use crate::engine::dspark_turn::PagedDsparkBackend;
    use crate::models::quantized_linear::LinearProj;
    use crate::models::qwen3_5::dflash2::{DFLASH2_DRAFT_PROPOSALS, DFLASH2_VERIFY_ROWS};
    use crate::models::qwen3_5::paged_forward::{
        PREFILL_CHUNK_BLOCKS, PREFILL_CHUNK_ROWS_OVERRIDE, run_paged_dflash2_verify_step,
    };
    use crate::nn::Linear;
    use crate::stream::{Stream, StreamContext};
    use rand::SeedableRng;

    const VOCAB: u32 = 16;

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
    fn tiny_paged_int8_dflash_inner(seed: u64) -> Qwen35Inner {
        tiny_paged_int8_dflash_inner_heads(seed, Heads::Same)
    }

    fn tiny_paged_int8_dflash_inner_heads(seed: u64, heads: Heads) -> Qwen35Inner {
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

    fn prompt(len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| (i * 7 + 3) % VOCAB).collect()
    }

    fn argmax_rows(logits: &MxArray) -> Result<Vec<u32>> {
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
    fn paged_ar(inner: &mut Qwen35Inner, prompt: &[u32], n: usize) -> Result<(Vec<u32>, u32)> {
        let prefix = inner.prime_dense_prefix(prompt, 0, None)?;
        let suffix = &prompt[prefix.effective_cached_prefix_len()..];
        let logits = inner.dense_paged_prefill(suffix, &prefix, None)?.0;
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
            tokens.extend(argmax_rows(&logits)?);
        }
        Ok((tokens, cursor(inner)))
    }

    struct Run {
        tokens: Vec<u32>,
        cursor: u32,
        cached: usize,
        logits: Vec<Vec<u32>>,
        /// Deferred GDN launches per cycle, and the linear layer count.
        deferred: Vec<u64>,
        linear: u64,
        gdn: crate::models::qwen3_5::gdn_blob::trace::GdnTrace,
    }

    fn bits(array: &MxArray) -> Result<Vec<u32>> {
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
                pool_lane_transient(&cfg, Some(3), window),
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
    /// 128), one launch per full-attention layer. Mutation R1:
    /// `paged_verify_core` passes `PagedPass::Prefill` → red (no verify
    /// launch).
    #[test]
    fn verify_takes_the_verify_split_rule() -> Result<()> {
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
        let logits = inner.paged_prefill_with_draft_state(&prompt, &prefix, stream)?;
        let anchor = argmax_rows(&logits)?[0];
        let chunks = probe(0);
        assert!(
            chunks > 0 && chunks % layers == 0,
            "{chunks} prefill launches over {layers} layers"
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
                ids,
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

    /// A draft needs the pool, and with one the plan routes DFlash2 turns to
    /// the paged lane. Mutation L1 (the check passes without a pool) turns
    /// it red.
    #[test]
    fn dflash2_companion_needs_the_pool() {
        use crate::engine::backend::ChatBackend;
        use crate::engine::plan::{MediaCapabilities, TurnPath, TurnPlan, TurnRequest};
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8008);
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
        inner.paged_adapter = None;
        let refused = inner.require_dflash2_pool().expect_err("no pool, no draft");
        assert!(
            refused.reason.contains("runs on the block-paged KV pool"),
            "{}",
            refused.reason
        );
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

    /// The prefill slice `[start, end)` of `seq`'s request, the sequence
    /// parked after it; returns the slice's logits.
    fn scheduled_slice(
        inner: &mut Qwen35Inner,
        seq: u32,
        tokens: &[u32],
        base: &<Qwen35Inner as PagedBackend>::PrefixState,
        start: usize,
        end: usize,
    ) -> Result<MxArray> {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        HybridSchedulerBackend::activate_paged_seq(inner, seq)?;
        let prefix = HybridSchedulerBackend::build_scheduled_prefix(
            inner,
            base,
            start,
            end - start,
            tokens.to_vec(),
            start == base.effective_cached_prefix_len(),
        );
        let logits = PagedBackend::paged_prefill(
            inner,
            &tokens[start..end],
            &prefix,
            Stream::default(crate::stream::DeviceType::Gpu),
        )?;
        logits.eval();
        HybridSchedulerBackend::park_active_scheduled_recurrent(inner)?;
        Ok(logits)
    }

    /// Splash `configureDraftStatePlan` and `publishReachedStateBoundaries`
    /// for scheduled requests on a draft load, which decode
    /// autoregressively (the scheduler never speculates there): two
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
            let logits = scheduled_slice(&mut inner, 4, &longer, &base, 144, 180)?;
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
    /// thread's scheduler on a draft load, which decodes autoregressively
    /// (the scheduler never speculates there): two sessions started together
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

    /// A one-chunk DFlash2 prefill with no capture leaves its ring and
    /// window lazy; the idle lane keeps them materialized, so a restore's
    /// claim credits only storage that exists (Splash
    /// `QwenStateStorage::releaseLane` pools allocated rings).
    /// Mutations IR1 (the idle ring is kept lazy) and IR2 (the idle window
    /// is) → red.
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
}
