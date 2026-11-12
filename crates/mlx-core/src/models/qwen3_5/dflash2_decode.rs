//! DFlash2 whole-turn integration for dense Qwen3.8.

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::engine::backend::{DsparkBackend, DsparkProposal, DsparkStepper, DsparkVerifyOutput};
use crate::engine::memory_plan::EXTENT_TARGET_BYTES;
use crate::transformer::KvFormat;
use crate::transformer::paged_policy::{
    estimate_paged_pool_sdpa_bytes_with_portable, estimate_varlen_paged_attention_bytes,
};

use super::dflash2::DFlash2ContextCache;
use super::dflash2_turn::{DFlash2Target, DFlash2TurnState};
use super::gdn_blob::GdnStateBlobs;
use super::layer_cache::{Qwen3_5LayerSnapshot, replay_mtp_snapshot_to, snapshot_all_mtp};
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
    /// Packed GDN state of every linear layer (the per-layer cache slots are
    /// views of it) — the pre-verify state the fused commit replays from.
    /// `None` keeps the per-layer replay path.
    gdn_blobs: Option<GdnStateBlobs>,
    paged: PagedVerifyLane,
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

#[cfg(test)]
thread_local! {
    /// Paged commits that adopted the verify's spare GDN cell.
    pub(crate) static PAGED_ADOPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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

/// Packed GDN state is on unless `MLX_DFLASH2_GDN_BLOB=0` (A/B switch).
fn dflash2_gdn_blob_enabled() -> bool {
    std::env::var("MLX_DFLASH2_GDN_BLOB")
        .ok()
        .is_none_or(|value| value.trim() != "0")
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
/// peak: the last prefill chunk's scratch with its attention route, or one
/// decode step. A bf16 pool takes the cheaper of the two routes the paged
/// prefill picks from; an int8 pool always reads through the varlen kernel
/// with its bounded workspace. With a DFlash2 draft (`draft_taps` tapped
/// layers) the prefill also holds the chunk's taps, and the decode step is
/// the 8-row verify on the pool with the free pool its decode cache limit
/// keeps (Splash prices the one 8-row verify graph).
pub(crate) fn pool_lane_transient(
    config: &super::config::Qwen3_5Config,
    format: KvFormat,
    kernels: PrefillSdpaKernels,
    draft_taps: Option<usize>,
    window: u32,
) -> u64 {
    let dim = |v: i32| v.max(0) as u64;
    let window = u64::from(window);
    let chunk = window.min(PREFILL_STEP_SIZE as u64);
    let prefill = prefill_chunk_scratch_bytes(config, window)
        .saturating_add(pool_attention_bytes(config, format, kernels, chunk, window));
    match draft_taps {
        Some(taps) => {
            let tapped = chunk
                .saturating_mul(taps as u64)
                .saturating_mul(dim(config.hidden_size))
                .saturating_mul(2);
            let cycle = pool_verify_cycle_bytes(config, format, kernels, window);
            prefill
                .saturating_add(tapped)
                .max(cycle.saturating_add(crate::cache_limit::decode_cache_limit(cycle)))
        }
        None => prefill.max(dflash2_cycle_transient(config, 0, window).total()),
    }
}

/// One 8-row DFlash2 verify on the pool over `window` keys: logits, GDN
/// tapes and one layer's scratch, and the attention workspace of the 8
/// rows. The stepper's decode cache limit is sized from it.
pub(crate) fn pool_verify_cycle_bytes(
    config: &super::config::Qwen3_5Config,
    format: KvFormat,
    kernels: PrefillSdpaKernels,
    window: u64,
) -> u64 {
    let rows = super::dflash2::DFLASH2_VERIFY_ROWS;
    dflash2_cycle_transient(config, rows - 1, window)
        .core_bytes
        .saturating_add(pool_attention_bytes(
            config,
            format,
            kernels,
            rows as u64,
            window,
        ))
}

/// Attention workspace of `queries` rows over `window` keys on the pool.
fn pool_attention_bytes(
    config: &super::config::Qwen3_5Config,
    format: KvFormat,
    kernels: PrefillSdpaKernels,
    queries: u64,
    window: u64,
) -> u64 {
    let dim = |v: i32| v.max(0) as u64;
    let (heads, kv_heads, head_dim) = (
        dim(config.num_heads),
        dim(config.num_kv_heads),
        dim(config.head_dim),
    );
    match format {
        KvFormat::Int8 => crate::transformer::paged_policy::estimate_int8_varlen_prefill_bytes(
            queries, window, heads, head_dim,
        ),
        KvFormat::Bf16 => {
            let dtype_bytes = DType::BFloat16.byte_size() as u64;
            let gathered = estimate_paged_pool_sdpa_bytes_with_portable(
                queries,
                window,
                heads,
                kv_heads,
                head_dim,
                dtype_bytes,
                kernels.d256_full,
                kernels.portable_d256,
            );
            let varlen = estimate_varlen_paged_attention_bytes(
                queries,
                window,
                heads,
                kv_heads,
                head_dim,
                dtype_bytes,
            );
            gathered.min(varlen)
        }
    }
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

/// Below this many keys the attention kernels run one-pass and allocate no
/// partials (`sdpa_vector_uses_two_pass` in `mlx_segmented_sdpa_plan.h`).
const SDPA_TWO_PASS_MIN_KEYS: u64 = 4096;
/// Keys per stage-1 partition of the segmented verify tile kernel
/// (`segmented_verify_tile_partitions`: `8 × max(tile_n, 8)`).
const SDPA_KEYS_PER_PARTITION: u64 = 64;

impl Qwen35DFlash2Stepper<'_> {
    fn ensure_clean(&self, operation: &str) -> Result<()> {
        if self.snapshot.is_some()
            || self.tape.is_some()
            || self.tapped.is_some()
            || self.verified_ids.is_some()
        {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 {operation}: prior verify was not committed"
            )));
        }
        Ok(())
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
        super::dflash2_turn::paged_step_reclaiming(&mut *self.inner, |inner| {
            inner
                .paged_adapter
                .as_mut()
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 paged adapter is absent"))?
                .reserve_rows(DFLASH2_VERIFY_ROWS as u32)
                .map(|_| ())
                .map_err(Error::from_reason)
        })?;
        let snapshot = snapshot_all_mtp(
            self.inner
                .caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?,
            true,
        )?;
        let ticket =
            crate::engine::spec_paged::SpecPagedCache::open_core_write_cycle(self, seq, full.len())
                .map_err(Error::from_reason)?;
        let mut tape = Vec::new();
        let phase_time = std::env::var("MLX_DFLASH2_PHASE_TIME").is_ok();
        let t0 = std::time::Instant::now();
        let result = {
            let inner = &mut *self.inner;
            let rope_deltas = inner.cached_rope_deltas.unwrap_or(0);
            match (inner.caches.as_mut(), inner.paged_adapter.as_mut()) {
                (Some(caches), Some(adapter)) => {
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
                        self.gdn_blobs.as_ref().map(GdnStateBlobs::destinations),
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
        self.snapshot = Some(snapshot);
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
        // The verify writes 8 rows; only keeping all of them is the
        // window-final state its spare cell holds.
        let written = super::dflash2::DFLASH2_VERIFY_ROWS;
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
                let adopted = if keep == written {
                    blobs.adopt(&tape)?
                } else {
                    None
                };
                #[cfg(test)]
                if adopted.is_some() {
                    PAGED_ADOPTS.with(|count| count.set(count.get() + 1));
                }
                match adopted {
                    Some(next) => Some(next),
                    None => blobs.commit(&tape, keep)?,
                }
            }
            None => None,
        };
        match next_blobs {
            Some(next) => {
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
                    true,
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
        // Pack the prefill's per-layer GDN states once per turn (two concats);
        // every cycle's commit then rewrites the blobs in one fused primitive
        // and re-points the per-layer slots at them.
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
                    self.config.kv_format(),
                    PrefillSdpaKernels::probe(self.config.head_dim),
                    u64::from(context_window),
                )),
            ),
        };
        Ok(Qwen35DFlash2Stepper {
            inner: self,
            context: state.context,
            next_position: state.next_position,
            tap_layers,
            snapshot: None,
            tape: None,
            tapped: None,
            verified_ids: None,
            gdn_blobs,
            paged,
        })
    }
}

impl Drop for Qwen35DFlash2Stepper<'_> {
    fn drop(&mut self) {
        // A turn that failed between the verify write and its commit leaves
        // the cycle open; keeping every row closes the ticket, and the
        // turn's abort releases the request.
        if let Some(ticket) = self.paged.open_cycle.take() {
            let (seq, rows) = (ticket.seq_id(), ticket.rows());
            let _ =
                crate::engine::spec_paged::SpecPagedCache::commit_cycle(self, seq, ticket, rows);
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

/// The prefix a DFlash2 turn on the pool resumes at: the live paged request
/// of `seq` and the retained draft window end at the same tokens, under the
/// turn's salt (Splash resumes a state only at its exact boundary).
pub(crate) fn reusable_paged_dflash2_prefix(
    tokens: &[u32],
    reuse_cache: bool,
    cache_salt: u64,
    adapter: &crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter,
    seq: u32,
    retained: Option<&DFlash2ContextCache>,
) -> usize {
    if !reuse_cache
        || !adapter.is_live_for_continue_for(seq)
        || adapter.request_cache_salt_for(seq) != Some(cache_salt)
    {
        return 0;
    }
    let Some(live) = adapter.request_tokens_for(seq) else {
        return 0;
    };
    let window_ends_there = retained.is_some_and(|context| {
        usize::try_from(context.logical_len()) == Ok(live.len()) && context.token_history() == live
    });
    if window_ends_there {
        live_window_prefix(tokens, live)
    } else {
        0
    }
}

/// The prefix a turn of `tokens` continues on a live request of `live`
/// tokens whose draft window ends with it: the whole live request when the
/// prompt extends it by at least one token, else nothing.
pub(crate) fn live_window_prefix(tokens: &[u32], live: &[u32]) -> usize {
    if !live.is_empty() && live.len() < tokens.len() && tokens.starts_with(live) {
        live.len()
    } else {
        0
    }
}

impl crate::engine::dspark_turn::PagedDsparkBackend for Qwen35Inner {
    fn paged_prefill_with_draft_state(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        _stream: crate::stream::Stream,
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
            _ if cached == 0 => DFlash2ContextCache::new(&config),
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
        use crate::engine::backend::PagedPrefix;
        let seq = self.active_scheduled_seq.unwrap_or(0);
        let live = self.paged_adapter.as_ref().map_or(0, |adapter| {
            reusable_paged_dflash2_prefix(
                tokens,
                reuse_cache,
                cache_salt,
                adapter,
                seq,
                self.dflash2_context.as_ref(),
            )
        });
        let (hit, stored) = crate::models::qwen3_5::dflash2_turn::resolve_dflash2_resume(
            self,
            tokens,
            reuse_cache,
            cache_salt,
            seq,
            live,
        )?;
        let prefix = self.prime_dense_prefix(tokens, cache_salt, Some(hit))?;
        let cached = prefix.effective_cached_prefix_len();
        if cached == 0 {
            self.dflash2_context = None;
        }
        crate::models::qwen3_5::dflash2_turn::use_resumed_replay_point(
            self, tokens, stored, cached,
        );
        Ok(prefix)
    }

    fn set_generation_suffix(&mut self, suffix: Option<std::sync::Arc<[u32]>>) {
        self.dflash2_states.generation_suffix = suffix;
    }
}

impl DFlash2Target for Qwen35Inner {
    fn dflash2(&self) -> Option<&super::dflash2::DFlash2Model> {
        self.dflash2.as_ref()
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

    fn ladder_marks(&self) -> std::collections::HashSet<u64> {
        Self::ladder_marks(self)
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

    fn dflash2_turn_state_clear(&mut self) {
        self.dflash2_turn_state = None;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::models::quantized_linear::LinearProj;
    use crate::nn::Linear;

    pub(crate) fn tiny_dflash_inner_with_attention_head_dim(
        seed: u64,
        head_dim: i32,
    ) -> Qwen35Inner {
        tiny_dflash_inner_with_config(seed, |config| config.head_dim = head_dim)
    }

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

    /// The seeded paged fixture on an int8 pool, a sensitive random head and
    /// a draft that proposes the 7 tokens a cycle drafts.
    fn tiny_paged_int8_dflash_inner(seed: u64) -> Qwen35Inner {
        let mut inner =
            super::super::model::scheduled_mtp::seeded_inner_with_config(seed, |config| {
                config.head_dim = 256;
                config.hidden_size = 256;
                config.kv_format = Some("int8".to_string());
                // The 27B's GDN head geometry, so the verify takes the complete
                // kernel that writes the spare cell.
                config.linear_key_head_dim = 128;
                config.linear_value_head_dim = 128;
                config.linear_num_key_heads = 2;
                config.linear_num_value_heads = 2;
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
        let prefix = inner.prime_prefix_state(prompt, true, 0, &[], 0)?;
        let suffix = &prompt[prefix.effective_cached_prefix_len()..];
        let logits = inner.paged_prefill(suffix, &prefix, Stream::generation())?;
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
        adopts: usize,
        cached: usize,
        logits: Vec<Vec<u32>>,
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
        PAGED_ADOPTS.with(|count| count.set(0));
        let n = reference.len();
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
            let verified = step.verify(&ids)?.logits;
            all_logits.push(bits(&verified)?);
            let rows = argmax_rows(&verified)?;
            assert_eq!(rows.len(), ids.len(), "the driver sees 1 + L logits rows");
            let mut accepted = 0;
            while accepted < l_cap && rows[accepted] == ids[accepted + 1] {
                accepted += 1;
            }
            step.commit_with_provenance(accepted + 1, ids.len(), &ids)?;
            generated.extend_from_slice(&ids[1..=accepted]);
            generated.push(rows[accepted]);
        }
        step.finish()?;
        Ok(Run {
            tokens: generated,
            cursor: cursor(inner),
            adopts: PAGED_ADOPTS.with(std::cell::Cell::get),
            cached,
            logits: all_logits,
        })
    }

    /// Lossless speculation on the int8 pool, every keep: paged DFlash2 emits
    /// the paged AR tokens and lands on the AR cursor, and a full accept
    /// adopts the spare GDN cell. Mutations: M1 (the commit keeps all 8
    /// rows), M2 (the 8-row chunk skips the pool store) and A1 (the verify
    /// writes no spare cell) turn it red.
    #[test]
    fn paged_dflash2_matches_paged_ar_for_every_keep() -> Result<()> {
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prompt = prompt(40);
        let n = 64;
        let mut ar_inner = tiny_paged_int8_dflash_inner(0x5_8001);
        let (reference, ar_cursor) = paged_ar(&mut ar_inner, &prompt, n)?;
        for correct in 0..=DFLASH2_DRAFT_PROPOSALS {
            let mut inner = tiny_paged_int8_dflash_inner(0x5_8001);
            let run = paged_dflash2(&mut inner, &prompt, &reference, correct)?;
            assert_eq!(
                run.tokens[..n],
                reference[..],
                "paged DFlash2 with {correct} correct proposals per cycle left the AR tokens"
            );
            assert_eq!(
                run.cursor, ar_cursor,
                "paged DFlash2 with {correct} correct proposals left the AR cursor"
            );
            if correct == DFLASH2_DRAFT_PROPOSALS {
                assert!(
                    run.adopts > 0,
                    "a full accept must adopt the spare GDN cell the verify wrote"
                );
            }
        }
        Ok(())
    }

    /// A 3-proposal cycle runs the 8-row graph and commits 3; an `l_cap == 0`
    /// cycle drafts its tail itself and also writes 8 rows. Mutation R1 (the
    /// verify writes only the driver's rows) turns it red.
    #[test]
    fn dflash2_cycle_runs_eight_rows_and_commits_keep() -> Result<()> {
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

    /// Splash `PagedAttention::addVerify`: a lane chunk that is not the 8
    /// verify rows, a page table short of them, or rows past the window and
    /// its scratch rows are refused before anything is recorded. Mutation K1
    /// (the row check removed) turns it red.
    #[test]
    fn paged_dflash2_verify_refuses_a_chunk_that_is_not_eight_rows() -> Result<()> {
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8005);
        let prompt = prompt(44);
        let prefix = inner.prime_prefix_state(&prompt, true, 0, &[], 0)?;
        inner.paged_prefill(&prompt, &prefix, stream)?.eval();
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
                None,
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
            vec![2, 4, 6, 8, 9, 10],
            "blocks held before each chunk's forward"
        );
        let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
        step.verify(&[anchor])?.logits.eval();
        step.commit_with_provenance(1, 1, &[anchor])?;
        step.finish()?;
        assert_eq!(blocks(&inner), 158usize.div_ceil(16));
        Ok(())
    }

    /// The tapped prefill is the plain one: same chunks, same logits bits.
    /// Mutation T0 (the tapped prefill takes its own chunk schedule) turns it
    /// red.
    #[test]
    fn tapped_paged_prefill_keeps_plain_prefill_bits() -> Result<()> {
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prompt = prompt(150);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(32)));
        let result = (|| -> Result<(Vec<u32>, Vec<u32>)> {
            let mut plain = tiny_paged_int8_dflash_inner(0x5_8007);
            let prefix = plain.prime_prefix_state(&prompt, true, 0, &[], 0)?;
            let plain_logits = plain.paged_prefill(&prompt, &prefix, stream)?;
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
    /// returns the live request it leaves.
    fn whole_turn(
        inner: &mut Qwen35Inner,
        tokens: &[u32],
        decoder: crate::engine::plan::DecoderPlan,
    ) -> Result<Vec<u32>> {
        use crate::engine::backend::{ChatBackend, TurnOutput, WholeTurnArgs};
        use crate::engine::plan::{MediaCapabilities, MediaInputs, TurnPlan};
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let config = crate::engine::types::ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(20),
            reasoning_effort: Some("none".to_string()),
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
        Ok(inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .request_tokens()
            .to_vec())
    }

    /// A second DFlash2 turn resumes the live request and the draft window
    /// the first left at the same token (Splash resumes a state only at its
    /// exact boundary), and both turns emit what two paged AR turns emit.
    /// Mutations W0 (`finish` drops the draft window) and R0 (the prime
    /// never reuses the live request) turn it red.
    #[test]
    fn second_dflash2_turn_resumes_the_live_window() -> Result<()> {
        use crate::engine::backend::ChatBackend;
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let first = prompt(40);
        let mut spec = tiny_paged_int8_dflash_inner(0x5_800A);
        let mut ar = tiny_paged_int8_dflash_inner(0x5_800A);
        let live = whole_turn(&mut spec, &first, dflash2)?;
        assert_eq!(
            live,
            whole_turn(&mut ar, &first, DecoderPlan::Autoregressive)?
        );
        assert_eq!(live.len(), first.len() + 19, "the last token has no K/V");
        let window = spec
            .dflash2_context
            .as_ref()
            .expect("the turn keeps its window");
        assert_eq!(window.token_history(), live.as_slice());
        let mut second = live.clone();
        second.extend(prompt(9).iter().rev());
        let adapter = spec.paged_adapter.as_ref().expect("pool");
        assert_eq!(
            reusable_paged_dflash2_prefix(
                &second,
                true,
                ChatBackend::resolve_params(&spec, &Default::default()).cache_salt,
                adapter,
                0,
                spec.dflash2_context.as_ref(),
            ),
            live.len(),
            "the second turn resumes the whole live request"
        );
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(4)));
        PREFILL_CHUNK_BLOCKS.with(|log| *log.borrow_mut() = Some(Vec::new()));
        let spec_live = whole_turn(&mut spec, &second, dflash2);
        let seen = PREFILL_CHUNK_BLOCKS
            .with(|log| log.borrow_mut().take())
            .unwrap_or_default();
        let ar_live = whole_turn(&mut ar, &second, DecoderPlan::Autoregressive);
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        assert_eq!(
            seen.len(),
            3,
            "the second prefill runs only the 9 new tokens"
        );
        assert_eq!(spec_live?, ar_live?);
        Ok(())
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
    /// blocks on the installed GDN cell, emits the paged AR tokens of A2,
    /// and its teacher-forced DFlash2 logits equal those of a run with no Q
    /// in between, bit for bit.
    /// Mutation MK1 (parked states mark no block) → red (Q's prefill takes
    /// A1's chain instead of growing the pool; A2's install fails).
    /// Mutation GI1 (the install leaves the GDN slots fresh) → red (A2
    /// replays the GDN prefix).
    #[test]
    fn parked_anchor_resume_matches_an_uninterrupted_run() -> Result<()> {
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
        let run = |with_q: bool| -> Result<(Run, Vec<u32>, Vec<u32>)> {
            let mut inner = tiny_paged_int8_dflash_inner(0x5_8101);
            install_pool(&mut inner, 16, 64, passing_probe())?;
            whole_turn(&mut inner, &a1, dflash2)?;
            let chain = block_ids(&inner)[..13].to_vec();
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
            Ok((run, chain, block_ids(&inner)))
        };
        let (x, x_chain, x_ids) = run(true)?;
        let (y, y_chain, y_ids) = run(false)?;
        assert_eq!((x.cached, y.cached), (208, 208));
        assert_eq!(x_ids[..13], x_chain[..], "A2 resumes on A1's blocks");
        assert_eq!(y_ids[..13], y_chain[..]);
        assert_eq!(
            x.tokens[..reference.len()],
            reference[..],
            "A2 left the AR tokens"
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
        use crate::engine::plan::{DecoderPlan, SpeculativeKind};
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let dflash2 = DecoderPlan::Speculative(SpeculativeKind::DraftModel);
        let a1 = prompt(210);
        let mut b = a1[..208].to_vec();
        b.extend((0..40u32).map(|i| (i * 11 + 4) % VOCAB));
        let mut inner = tiny_paged_int8_dflash_inner(0x5_8102);
        whole_turn(&mut inner, &a1, dflash2)?;
        let a1_ids = block_ids(&inner);
        assert_eq!(a1_ids.len(), 15);
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
        let b_ids = block_ids(&inner);
        assert_eq!(b_ids[..13], a1_ids[..13], "B sits on A1's chain");
        assert!(
            allocated(&inner) < 2 * a1_ids.len(),
            "{} blocks for two sessions",
            allocated(&inner)
        );
        Ok(())
    }

    /// The tapped prefill ends a chunk at the prompt's replay point (Splash
    /// `Engine::workEndOf`) and parks the GDN cell and draft window there,
    /// before the generation prompt (Splash `Engine::promptReplayBoundary`).
    /// A twin prompt cut right after that row parks the same state bits.
    /// Mutation CA1 (the capture row is no chunk break) → red (no chunk ends
    /// at 192, nothing parked).
    #[test]
    fn tapped_prefill_parks_the_replay_point() -> Result<()> {
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
        assert_eq!(seen, vec![8, 12, 13, 14], "chunks end at 128, 192, 208");
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
            assert_eq!(entry.state().window.logical_len(), 192);
            let mut arrays = Vec::new();
            entry.state().collect_arrays(&mut arrays);
            arrays.into_iter().map(bits).collect()
        };
        assert!(anchor_bits(&inner)? == anchor_bits(&twin)?);
        Ok(())
    }
}
