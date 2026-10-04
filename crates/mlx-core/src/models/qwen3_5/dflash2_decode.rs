//! DFlash2 whole-turn integration for dense Qwen3.8.

use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChatBackend, DsparkBackend, DsparkProposal, DsparkStepper, DsparkVerifyOutput, FinalizeArgs,
    ResetScope, SpecFrontier, TurnOutput, WholeTurnArgs,
};
use crate::engine::decode::TurnStreaming;
use crate::engine::dspark_turn::{DsparkTurnArgs, run_dspark_turn};
use crate::engine::finalize::compute_performance_metrics;
use crate::engine::params::{generated_capacity_hint, kv_capacity_round_up};
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::stream::{Stream, StreamContext, WiredLimitContext};
use crate::transformer::paged_kv_cache_adapter::PagedPrefillMemorySnapshot;
use crate::transformer::paged_policy::live_prefill_headroom;

use super::dflash2::DFlash2ContextCache;
use super::layer_cache::{
    Qwen3_5LayerCache, Qwen3_5LayerSnapshot, replay_mtp_snapshot_to, snapshot_all_mtp,
};
use super::model::{PREFILL_STEP_SIZE, Qwen35Inner};

pub(crate) struct DFlash2TurnState {
    context: DFlash2ContextCache,
    next_position: i32,
}

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
}

fn reusable_dflash2_prefix(
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

fn flat_attention_frontier(inner: &Qwen35Inner) -> Option<usize> {
    let mut frontiers = inner.caches.as_ref()?.iter().filter_map(|cache| {
        if matches!(
            cache,
            super::layer_cache::Qwen3_5LayerCache::FullAttention(_)
        ) {
            usize::try_from(cache.offset().max(0)).ok()
        } else {
            None
        }
    });
    let first = frontiers.next()?;
    frontiers.all(|frontier| frontier == first).then_some(first)
}

fn constrain_dflash2_context_params(
    prompt_tokens: usize,
    target_capacity: i32,
    draft_capacity: usize,
    params: &mut crate::engine::params::ChatParams,
) -> Result<usize> {
    let target_capacity = usize::try_from(target_capacity.max(0)).unwrap_or(usize::MAX);
    let capacity = target_capacity.min(draft_capacity);
    super::model::constrain_paged_context_params(
        "Qwen3.8 DFlash2",
        prompt_tokens,
        u32::try_from(capacity).unwrap_or(u32::MAX),
        params,
    )?;
    Ok(capacity)
}

fn dflash2_final_token_fits_context(logical_len: i32, capacity: usize) -> bool {
    usize::try_from(logical_len).is_ok_and(|logical_len| logical_len < capacity)
}

/// Process reserve kept out of the live headroom before sizing the target KV
/// reservation; 90% of the rest is usable, leaving room for prefill and
/// verify transients.
const DFLASH2_KV_HEADROOM_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// Growth step of the flat `KVCache`; a smaller look-ahead saves nothing.
const DFLASH2_KV_MIN_AHEAD_ROWS: u64 = 256;

fn dflash2_memory_snapshot() -> PagedPrefillMemorySnapshot {
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

/// Bytes the target KV reservation may claim, or `None` when the device
/// limits cannot be read.
fn dflash2_kv_budget_bytes(snapshot: PagedPrefillMemorySnapshot) -> Option<u64> {
    live_prefill_headroom(snapshot)
        .selected_bytes
        .map(|headroom| headroom.saturating_sub(DFLASH2_KV_HEADROOM_RESERVE_BYTES) / 10 * 9)
}

fn kv_row_bytes(full_attention_layers: usize, kv_heads: i32, head_dim: i32, dtype: DType) -> u64 {
    (full_attention_layers as u64)
        .saturating_mul(2)
        .saturating_mul(kv_heads.max(0) as u64)
        .saturating_mul(head_dim.max(0) as u64)
        .saturating_mul(dtype.byte_size() as u64)
}

/// Target rows to reserve for one DFlash2 turn: the whole turn
/// (`prompt + max_new`, rounded to the growth step) when the budget covers
/// it, else as many look-ahead rows as fit. `None` keeps the growth path.
/// Growing a live buffer allocates the whole new buffer while the old one is
/// still alive, so every prompt row is charged unless `resident_rows` already
/// hold the whole turn.
fn dflash2_kv_reserve_rows(
    prompt_len: usize,
    max_new_tokens: i32,
    resident_rows: i64,
    row_bytes: u64,
    budget_bytes: Option<u64>,
) -> Result<Option<i64>> {
    let prompt = i32::try_from(prompt_len).map_err(|_| {
        Error::from_reason(format!(
            "Qwen3.8 DFlash2 prompt of {prompt_len} tokens is too long"
        ))
    })?;
    let whole_turn = kv_capacity_round_up(prompt, max_new_tokens)? as i64;
    if resident_rows > 0 && resident_rows >= whole_turn {
        return Ok(Some(whole_turn));
    }
    let Some(budget) = budget_bytes else {
        return Ok(None);
    };
    if row_bytes == 0 {
        return Ok(None);
    }
    let ahead_cap = budget.saturating_sub((prompt as u64).saturating_mul(row_bytes)) / row_bytes;
    if ahead_cap < DFLASH2_KV_MIN_AHEAD_ROWS {
        return Ok(None);
    }
    let ahead = ahead_cap.min(max_new_tokens.max(0) as u64) as i32;
    Ok(Some(kv_capacity_round_up(prompt, ahead)? as i64))
}

/// Smallest full-attention buffer capacity, `None` without full-attention
/// caches.
fn min_fa_capacity(caches: &Option<Vec<Qwen3_5LayerCache>>) -> Result<Option<i64>> {
    let mut min = None;
    for cache in caches.iter().flatten() {
        if let Qwen3_5LayerCache::FullAttention(kv) = cache {
            let capacity = kv.capacity()?;
            min = Some(min.map_or(capacity, |m: i64| m.min(capacity)));
        }
    }
    Ok(min)
}

fn reserve_dflash2_target_kv(caches: &mut Option<Vec<Qwen3_5LayerCache>>, rows: i64) -> Result<()> {
    let caches = caches
        .as_mut()
        .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?;
    for cache in caches.iter_mut() {
        if let Some(kv) = cache.as_kv_cache_mut() {
            kv.reserve(rows)?;
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
        )
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

    fn attention_frontier(&self) -> Option<u64> {
        flat_attention_frontier(self.inner).and_then(|frontier| u64::try_from(frontier).ok())
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
            &self.context,
            anchor_id,
            max_len,
            temperature,
            greedy_fast,
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
        let (logits, tapped, tape) = super::model::forward_dflash2_with_taps(
            self.inner,
            &input,
            &self.tap_layers,
            true,
            super::model::DFlash2LogitsSpan::All,
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
        // Replay is required even on full accept: the windowed verify kernel
        // carries the recurrent state in f32 across the whole window and rounds
        // to bf16 once at the end, while replay re-rounds per token to restore
        // the AR-exact state serial decode would leave. Skipping it would let a
        // sub-ULP divergence compound across cycles.
        let caches = self
            .inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?;
        replay_mtp_snapshot_to(
            caches,
            &snapshot,
            &tape,
            keep,
            false,
            "Qwen3.8 DFlash2 commit",
        )?;
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

    fn frontier(&self) -> Option<SpecFrontier> {
        Some(SpecFrontier {
            attn_tokens: self.attention_frontier()?,
            recurrent_tokens: Some(self.next_position.max(0) as u64),
        })
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
        })
    }
}

impl Qwen35Inner {
    fn dflash2_prefill(
        &mut self,
        tokens: &[u32],
        position_base: i32,
        stream: Stream,
    ) -> Result<(MxArray, DFlash2TurnState)> {
        if tokens.is_empty() {
            return Err(Error::from_reason(
                "Qwen3.8 DFlash2 requires at least one prefill token",
            ));
        }
        let draft = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 has no loaded DFlash2 companion"))?;
        let tap_layers = draft.config.target_layers.clone();
        let draft_config = draft.config.clone();
        let mut context = match self.dflash2_context.take() {
            Some(context) if context.logical_len() == position_base => context,
            Some(context) => {
                return Err(Error::from_reason(format!(
                    "Qwen3.8 DFlash2 context length {} does not match cached prefix {position_base}",
                    context.logical_len()
                )));
            }
            None if position_base == 0 => DFlash2ContextCache::new(&draft_config),
            None => {
                return Err(Error::from_reason(format!(
                    "Qwen3.8 DFlash2 cached prefix {position_base} has no retained draft context"
                )));
            }
        };
        // Draft-context retention: each layer's sliding cache keeps only the
        // last `sliding_window - 1` rows, so rows below `keep_from` would be
        // evicted by later appends without ever being read. Their fc fusion
        // and per-layer K/V projections are dead work — skip the appends but
        // still advance logical length and token provenance (`record_only`).
        let base_usize = position_base.max(0) as usize;
        let window_rows = draft_config.sliding_window.saturating_sub(1);
        let keep_from = (base_usize + tokens.len())
            .saturating_sub(window_rows)
            .max(base_usize);
        let mut offset = 0usize;
        let mut last_logits = None;
        while offset < tokens.len() {
            if offset > 0
                && self
                    .turn_cancel
                    .as_ref()
                    .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
            {
                return Err(Error::from_reason("prefill cancelled"));
            }
            let end = (offset + PREFILL_STEP_SIZE as usize).min(tokens.len());
            let chunk = tokens[offset..end]
                .iter()
                .map(|&id| id as i32)
                .collect::<Vec<_>>();
            let input = MxArray::from_int32(&chunk, &[1, chunk.len() as i64])?;
            let (logits, taps, _) = {
                let _stream = StreamContext::new(stream);
                super::model::forward_dflash2_with_taps(
                    self,
                    &input,
                    &tap_layers,
                    false,
                    super::model::DFlash2LogitsSpan::LastRow,
                )?
            };
            let draft = self
                .dflash2
                .as_ref()
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?;
            let chunk_end = base_usize + end;
            if chunk_end <= keep_from {
                context.record_only(&tokens[offset..end])?;
            } else {
                let head = keep_from.saturating_sub(base_usize + offset);
                if head > 0 {
                    context.record_only(&tokens[offset..offset + head])?;
                }
                // Rows at or past `keep_from` survive in the sliding window:
                // fuse + project only that tail (RoPE keys are per-row at
                // absolute base + row, so a sliced append is identical).
                let fused = if head == 0 {
                    draft.fuse_context(&taps)?
                } else {
                    let tail_taps = taps
                        .iter()
                        .map(|tap| tap.slice_axis(1, head as i64, (end - offset) as i64))
                        .collect::<Result<Vec<_>>>()?;
                    draft.fuse_context(&tail_taps)?
                };
                context.append(
                    draft,
                    &fused,
                    (base_usize + offset + head) as i32,
                    &tokens[offset + head..end],
                )?;
            }
            // `LastRow` span already reduced the logits to the chunk's final
            // row ([1, 1, vocab]); only non-final chunks' rows are dropped.
            let vocab = logits.shape_at(2)?;
            last_logits = Some(logits.reshape(&[vocab])?);
            if end < tokens.len() {
                super::model::eval_layer_caches(&self.caches)?;
                context.eval()?;
                crate::array::clear_cache();
            }
            offset = end;
        }
        Ok((
            last_logits.ok_or_else(|| Error::from_reason("non-empty DFlash2 prefill"))?,
            DFlash2TurnState {
                context,
                next_position: position_base.saturating_add(tokens.len() as i32),
            },
        ))
    }

    /// K/V element type the target's full-attention caches hold or will be
    /// allocated with: activations follow the embedding output dtype.
    fn dflash2_target_kv_dtype(&self) -> Result<DType> {
        let allocated = self.caches.iter().flatten().find_map(|cache| match cache {
            Qwen3_5LayerCache::FullAttention(kv) => kv.keys_ref(),
            Qwen3_5LayerCache::Linear(_) => None,
        });
        match allocated {
            Some(keys) => keys.dtype(),
            None => self
                .embedding
                .forward(&MxArray::from_int32(&[0], &[1, 1])?)?
                .dtype(),
        }
    }

    /// Size every flat full-attention cache for this turn so decode never
    /// grows it. Runs before prefill: on a cold turn it only records the
    /// size for the first allocation; a warm turn copies its prefix once.
    fn reserve_dflash2_turn_kv(&mut self, prompt_len: usize, max_new_tokens: i32) -> Result<()> {
        let Some(resident_rows) = min_fa_capacity(&self.caches)? else {
            return Ok(());
        };
        let full_attention_layers = self
            .caches
            .iter()
            .flatten()
            .filter(|cache| matches!(cache, Qwen3_5LayerCache::FullAttention(_)))
            .count();
        let row_bytes = kv_row_bytes(
            full_attention_layers,
            self.config.num_kv_heads,
            self.config.head_dim,
            self.dflash2_target_kv_dtype()?,
        );
        let budget_bytes = dflash2_kv_budget_bytes(dflash2_memory_snapshot());
        let rows = dflash2_kv_reserve_rows(
            prompt_len,
            max_new_tokens,
            resident_rows,
            row_bytes,
            budget_bytes,
        )?;
        tracing::debug!(
            target: "mlx_core::inference",
            prompt_len,
            max_new_tokens,
            resident_rows,
            row_bytes,
            budget_bytes,
            reserved_rows = rows,
            "DFlash2 target KV reservation"
        );
        match rows {
            Some(rows) => reserve_dflash2_target_kv(&mut self.caches, rows),
            None => Ok(()),
        }
    }

    fn dflash2_fail_closed(&mut self, error: Error) -> Error {
        let _ = ChatBackend::reset_caches(self, ResetScope::Command);
        self.dflash2_turn_state = None;
        error
    }

    fn dflash2_materialize_final(&mut self, token: u32, stream: Stream) -> Result<()> {
        let tap_layers = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?
            .config
            .target_layers
            .clone();
        let input = MxArray::from_int32(&[token as i32], &[1, 1])?;
        let _stream = StreamContext::new(stream);
        let (_, taps, _) = super::model::forward_dflash2_with_taps(
            self,
            &input,
            &tap_layers,
            false,
            super::model::DFlash2LogitsSpan::LastRow,
        )?;
        let fused = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?
            .fuse_context(&taps)?;
        let mut context = self.dflash2_context.take().ok_or_else(|| {
            Error::from_reason("Qwen3.8 DFlash2 final token has no retained draft context")
        })?;
        let base = context.logical_len();
        let append_result = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))
            .and_then(|draft| context.append(draft, &fused, base, &[token]));
        self.dflash2_context = Some(context);
        append_result?;
        super::model::eval_layer_caches(&self.caches)
    }

    pub(crate) fn dflash2_chat_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        let tokenizer = args.tokenizer.clone();
        let tokens = args.tokens.to_vec();
        let is_delta = args.plan.is_delta;
        let is_streaming = args.sink.is_some();
        let mut params = ChatBackend::resolve_params(self, args.config);
        params.extra_eos_ids = ChatBackend::extra_eos_ids(self);
        let dflash_context_capacity = constrain_dflash2_context_params(
            tokens.len(),
            self.config.max_position_embeddings,
            self.dflash2
                .as_ref()
                .ok_or_else(|| Error::from_reason("loaded DFlash2 companion"))?
                .max_position_embeddings(),
            &mut params,
        )?;
        let prior_cached = if is_delta {
            self.cached_token_history.len()
        } else {
            0
        };
        let verified_hit = if is_delta {
            0
        } else {
            ChatBackend::verify_cache_prefix(self, &tokens, params.reuse_cache)
        };
        let flat_lane_authoritative =
            !self.paged_full_attn_caches_dirty && !self.flat_mtp_caches_desynced;
        let cached_prefix = reusable_dflash2_prefix(
            is_delta,
            &tokens,
            prior_cached,
            verified_hit,
            self.dflash2_context
                .as_ref()
                .map(DFlash2ContextCache::token_history),
            flat_attention_frontier(self),
            flat_lane_authoritative,
        );
        let prefill = if cached_prefix > 0 {
            tokens[cached_prefix..].to_vec()
        } else {
            // A previous turn may have occupied the target's paged lane.
            // Command reset also releases/purges that request before the
            // DFlash2 flat-cache prefill rebuilds the complete prompt.
            ChatBackend::reset_caches(self, ResetScope::Command)?;
            self.init_caches_sync()?;
            tokens.clone()
        };

        let generation_stream = Stream::generation();
        let report_performance = params.report_performance;
        let generation_start = report_performance.then(Instant::now);
        let mut first_token_instant = None;
        let mut profiler = DecodeProfiler::new(
            ChatBackend::profiler_label(self, is_delta, is_streaming),
            ChatBackend::family_name(self),
        );
        profiler.set_prompt_tokens(prefill.len() as u32);
        profiler.snapshot_memory_before();
        let reserved = {
            let _stream = StreamContext::new(generation_stream);
            self.reserve_dflash2_turn_kv(tokens.len(), params.max_new_tokens)
        };
        if let Err(error) = reserved {
            return Err(self.dflash2_fail_closed(error));
        }
        profiler.begin_prefill();
        let (last_logits, state) =
            match self.dflash2_prefill(&prefill, cached_prefix as i32, generation_stream) {
                Ok(value) => value,
                Err(error) => return Err(self.dflash2_fail_closed(error)),
            };
        profiler.end_prefill();

        let mut token_history = tokens.clone();
        let y = match apply_all_penalties(last_logits, &token_history, &params)
            .and_then(|logits| crate::sampling::sample(&logits, params.sampling_config))
        {
            Ok(token) => token,
            Err(error) => return Err(self.dflash2_fail_closed(error)),
        };
        y.eval();
        if let Err(error) = super::model::eval_layer_caches(&self.caches) {
            return Err(self.dflash2_fail_closed(error));
        }
        if report_performance {
            first_token_instant = Some(Instant::now());
        }
        self.dflash2_turn_state = Some(state);

        let mut generated = Vec::with_capacity(generated_capacity_hint(params.max_new_tokens));
        let mut finish_reason = String::from("length");
        let mut reasoning = ReasoningTracker::from_setup(&args.thinking, tokenizer.think_end_id());
        let stream_skip_special = ChatBackend::stream_skip_special_tokens(self);
        // One streaming bundle — detokenizer + cursors + emitter — built
        // only when a sink exists so the sync path never runs the emitter
        // hook.
        let mut turn_streaming = args.sink.map(|_| {
            TurnStreaming::new(
                self,
                tokenizer.inner(),
                args.thinking.enabled,
                stream_skip_special,
            )
        });
        let turn_token_observer = ChatBackend::turn_token_observer(self);
        let block_size = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("loaded DFlash2 companion"))?
            .config
            .block_size;
        let mut rng = rand::rng();
        let outcome = {
            let streaming = turn_streaming
                .as_mut()
                .and_then(|ts| ts.ctx(args.sink, args.cancelled));
            run_dspark_turn(
                self,
                &mut rng,
                DsparkTurnArgs {
                    y,
                    block_size,
                    params: &params,
                    reasoning_tracker: &mut reasoning,
                    profiler: &mut profiler,
                    max_new_tokens: params.max_new_tokens,
                    eos_id: args.eos_id,
                    generated_tokens: &mut generated,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant: &mut first_token_instant,
                    report_perf: report_performance,
                    generation_stream,
                    cancel_flag: args.cancelled,
                    turn_token_observer,
                },
                streaming,
            )
        };
        let mut last_in_cache = match outcome {
            Ok(outcome) => outcome.last_in_cache,
            Err(error) => return Err(self.dflash2_fail_closed(error)),
        };
        let final_token_fits_context = self.dflash2_context.as_ref().is_some_and(|context| {
            dflash2_final_token_fits_context(context.logical_len(), dflash_context_capacity)
        });
        if finish_reason == "length"
            && !last_in_cache
            && final_token_fits_context
            && let Some(&last) = generated.last()
        {
            if let Err(error) = self.dflash2_materialize_final(last, generation_stream) {
                return Err(self.dflash2_fail_closed(error));
            }
            last_in_cache = true;
        }
        let saved_generated = if !last_in_cache && !generated.is_empty() {
            &generated[..generated.len() - 1]
        } else {
            generated.as_slice()
        };
        let mut history = tokens.clone();
        history.extend_from_slice(saved_generated);
        self.cached_token_history = history;

        let performance = if report_performance {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                prefill.len(),
                generated.len(),
            )
            .map(|mut metrics| {
                ChatBackend::augment_performance(self, &profiler, &mut metrics);
                metrics
            })
        } else {
            None
        };
        if let (Some(sink), Some(ts)) = (args.sink, turn_streaming.as_mut()) {
            let decoded = tokenizer
                .decode_sync(&generated, stream_skip_special)
                .unwrap_or_default();
            ts.flush_residual(&decoded, params.include_reasoning, sink);
        }
        let prompt_tokens = if is_delta && is_streaming {
            ChatBackend::stream_delta_prompt_tokens(self, tokens.len(), tokens.len() - prior_cached)
        } else {
            tokens.len() as u32
        };
        let mut result = ChatBackend::finalize_turn(
            self,
            FinalizeArgs {
                tokenizer: &tokenizer,
                generated_tokens: &generated,
                finish_reason,
                think_end_id: tokenizer.think_end_id(),
                think_end_str: tokenizer.think_end_str(),
                think_end_extra_ids: &[],
                performance,
                include_reasoning: params.include_reasoning,
                thinking_enabled: args.thinking.enabled,
                prompt_tokens,
                reasoning_tokens: reasoning.reasoning_token_count(),
            },
        )?;
        result.cached_tokens = cached_prefix as u32;
        if let (Some(sink), Some(ts)) = (args.sink, turn_streaming.as_mut()) {
            ts.emitter.finish(&result, sink);
            Ok(TurnOutput::Streamed)
        } else {
            Ok(TurnOutput::Complete(Box::new(result)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;
    use crate::engine::extract_chat_params;
    use crate::engine::types::ChatConfig;
    use crate::models::quantized_linear::LinearProj;
    use crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache;
    use crate::nn::Linear;
    use rand::SeedableRng;

    #[derive(Debug, PartialEq, Eq)]
    struct ArTrace {
        logits: Vec<Vec<u32>>,
        target_cache: Vec<(Vec<i64>, Vec<u32>)>,
        draft_cache: Vec<(Vec<i64>, Vec<u32>)>,
        context_tokens: Vec<u32>,
        frontier: SpecFrontier,
        continuation_draft: Vec<i32>,
    }

    fn assert_trace_eq(actual: &ArTrace, expected: &ArTrace, label: &str) {
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

    fn array_fingerprints(arrays: Vec<&MxArray>) -> Result<Vec<(Vec<i64>, Vec<u32>)>> {
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

    fn reset_flat_fixture(inner: &mut Qwen35Inner) {
        inner.caches = Some(
            (0..inner.config.num_layers as usize)
                .map(|index| {
                    if inner.config.is_linear_layer(index) {
                        Qwen3_5LayerCache::new_linear()
                    } else {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
                .collect(),
        );
        inner.dflash2_context = None;
        inner.dflash2_turn_state = None;
    }

    fn tiny_dflash_inner(seed: u64) -> Qwen35Inner {
        tiny_dflash_inner_with_attention_head_dim(seed, 32)
    }

    fn tiny_dflash_inner_with_attention_head_dim(seed: u64, head_dim: i32) -> Qwen35Inner {
        unsafe { mlx_sys::mlx_seed(seed) };
        let mut inner = super::super::model::scheduled_mtp::seeded_inner_with_attention_head_dim(
            seed, head_dim,
        );
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
        reset_flat_fixture(inner);
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let (prefill_logits, state) = inner.dflash2_prefill(&[1, 2, 3, 4], 0, stream)?;
        prefill_logits.eval();
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
        Ok(ArTrace {
            logits: logits_trace,
            target_cache,
            draft_cache,
            context_tokens,
            frontier,
            continuation_draft,
        })
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
            let (prefill_logits, state) = inner.dflash2_prefill(&[1, 2, 3, 4], 0, stream)?;
            prefill_logits.eval();
            inner.dflash2_turn_state = Some(state);
            let input = MxArray::from_int32(ids, &[1, ids.len() as i64])?;
            let (logits, taps, tape) = super::super::model::forward_dflash2_with_taps(
                inner,
                &input,
                &tap_layers,
                true,
                super::super::model::DFlash2LogitsSpan::All,
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

    const GIB: u64 = 1024 * 1024 * 1024;
    const QWEN38_ROW_BYTES: u64 = 64 * 1024;

    #[test]
    fn kv_row_bytes_counts_every_full_attention_layer_and_dtype() {
        assert_eq!(kv_row_bytes(16, 4, 256, DType::BFloat16), QWEN38_ROW_BYTES);
        assert_eq!(
            kv_row_bytes(16, 4, 256, DType::Float32),
            2 * QWEN38_ROW_BYTES
        );
        assert_eq!(kv_row_bytes(0, 4, 256, DType::BFloat16), 0);
    }

    #[test]
    fn kv_budget_uses_the_lower_device_limit_minus_active_and_reserve() {
        let snapshot = PagedPrefillMemorySnapshot {
            allocator_active_bytes: Some(30 * GIB),
            allocator_cached_bytes: Some(7 * GIB),
            allocator_limit_bytes: Some(200 * GIB),
            metal_recommended_working_set_bytes: Some(100 * GIB),
            metal_current_allocated_bytes: None,
            paged_pool_allocated_bytes: Some(5 * GIB),
        };
        // min(200, 95% of 100) - 30 active - 5 pool - 2 reserve = 58 GiB; 90% usable.
        assert_eq!(dflash2_kv_budget_bytes(snapshot), Some(58 * GIB / 10 * 9));
        let lower_mlx_limit = PagedPrefillMemorySnapshot {
            allocator_limit_bytes: Some(60 * GIB),
            ..snapshot
        };
        assert_eq!(
            dflash2_kv_budget_bytes(lower_mlx_limit),
            Some(23 * GIB / 10 * 9)
        );
        let exhausted = PagedPrefillMemorySnapshot {
            allocator_active_bytes: Some(96 * GIB),
            ..snapshot
        };
        assert_eq!(dflash2_kv_budget_bytes(exhausted), Some(0));
        let unreadable = PagedPrefillMemorySnapshot {
            allocator_active_bytes: None,
            allocator_limit_bytes: None,
            ..snapshot
        };
        assert_eq!(dflash2_kv_budget_bytes(unreadable), None);
    }

    #[test]
    fn kv_reserve_rows_covers_turn_and_caps_lookahead() {
        let rows = |prompt, max_new, resident, budget| {
            dflash2_kv_reserve_rows(prompt, max_new, resident, QWEN38_ROW_BYTES, budget)
                .expect("valid reservation input")
        };
        let plenty = Some(64 * GIB);
        assert_eq!(rows(87, 1024, 0, plenty), Some(1280));
        assert_eq!(rows(6219, 1024, 0, plenty), Some(7424));
        assert_eq!(rows(32488, 1024, 0, plenty), Some(33536));
        // The budget first pays for the prompt rows that are not allocated
        // yet; the rest bounds the look-ahead.
        assert_eq!(
            rows(100, 100_000, 0, Some((100 + 8192) * QWEN38_ROW_BYTES)),
            Some(8448)
        );
        assert_eq!(
            rows(6000, 1024, 6144, Some(300 * QWEN38_ROW_BYTES)),
            None,
            "growing a resident buffer pays for the whole new buffer"
        );
        assert_eq!(
            rows(6000, 1024, 6144, Some((6000 + 400) * QWEN38_ROW_BYTES)),
            Some(6400)
        );
        // Less than one growth step of look-ahead keeps the growth path.
        assert_eq!(
            rows(100, 1024, 0, Some((100 + 255) * QWEN38_ROW_BYTES)),
            None
        );
        assert_eq!(rows(32488, 1024, 0, Some(GIB)), None);
        assert_eq!(rows(87, 1024, 0, None), None);
        assert_eq!(rows(0, 0, 0, plenty), Some(0));
        assert_eq!(
            dflash2_kv_reserve_rows(87, 1024, 0, 0, plenty).expect("no full attention"),
            None
        );
        assert!(dflash2_kv_reserve_rows(1, i32::MAX, 0, QWEN38_ROW_BYTES, plenty).is_err());
    }

    #[test]
    fn kv_reserve_warm_copy_fits_live_headroom() {
        // A warm 100K-row prompt already resident: its old buffer is in
        // `active` and stays live while `reserve` copies it into the new one.
        let resident = 100_096;
        let snapshot = PagedPrefillMemorySnapshot {
            allocator_active_bytes: Some(20 * GIB),
            allocator_cached_bytes: Some(0),
            allocator_limit_bytes: Some(36 * GIB),
            metal_recommended_working_set_bytes: None,
            metal_current_allocated_bytes: None,
            paged_pool_allocated_bytes: Some(0),
        };
        let headroom = live_prefill_headroom(snapshot)
            .selected_bytes
            .expect("fake limits are readable");
        assert_eq!(headroom, 16 * GIB);
        let budget = dflash2_kv_budget_bytes(snapshot);
        let rows = dflash2_kv_reserve_rows(100_000, 200_000, resident, QWEN38_ROW_BYTES, budget)
            .expect("valid reservation input")
            .expect("budget admits look-ahead");
        assert!(rows > resident, "the reservation copies the warm prefix");
        assert!(
            rows as u64 * QWEN38_ROW_BYTES <= headroom,
            "a {rows}-row replacement buffer exceeds {headroom} bytes of headroom"
        );
        // Without the copy the budget still covers resident-only turns.
        assert_eq!(
            dflash2_kv_reserve_rows(6000, 1024, 7168, QWEN38_ROW_BYTES, Some(0))
                .expect("valid reservation input"),
            Some(7168)
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
            reserve_dflash2_target_kv(&mut inner.caches, rows)?;
        }
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prompt = (0..250).map(|i| (i % 15 + 1) as u32).collect::<Vec<_>>();
        let (prefill_logits, state) = inner.dflash2_prefill(&prompt, 0, stream)?;
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

    #[test]
    fn context_budget_uses_the_smaller_target_or_draft_window() {
        let mut params = extract_chat_params(&ChatConfig {
            max_new_tokens: Some(64),
            ..ChatConfig::default()
        });
        let capacity = constrain_dflash2_context_params(10, 16, 12, &mut params)
            .expect("valid prompt is clamped");
        assert_eq!(capacity, 12);
        assert_eq!(params.max_new_tokens, 3);

        let mut too_long = extract_chat_params(&ChatConfig::default());
        let error = constrain_dflash2_context_params(13, 16, 12, &mut too_long)
            .expect_err("prompt beyond draft context must fail before prefill");
        assert!(error.reason.contains("effective active context is 12"));

        assert!(dflash2_final_token_fits_context(11, capacity));
        assert!(
            !dflash2_final_token_fits_context(12, capacity),
            "the final sampled token may be returned at capacity but must not be forwarded"
        );
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
            let (prefill_logits, state) = inner.dflash2_prefill(&[1, 2, 3, 4], 0, stream)?;
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

    fn metal_is_default_device() -> bool {
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
        let (prefill_logits, state) = inner.dflash2_prefill(&[1, 2, 3, 4], 0, stream)?;
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
                        &step.context,
                        anchor,
                        DRAFT_LEN,
                        0.0,
                        true,
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
}
