//! DFlash2 whole-turn integration for dense Qwen3.8.

use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChatBackend, DsparkBackend, DsparkProposal, DsparkStepper, DsparkVerifyOutput, FinalizeArgs,
    ResetScope, SpecFrontier, TurnOutput, WholeTurnArgs,
};
use crate::engine::decode::TurnStreaming;
use crate::engine::dspark_turn::{DsparkTurnArgs, run_dspark_turn};
use crate::engine::finalize::compute_performance_metrics;
use crate::engine::params::generated_capacity_hint;
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::stream::{DeviceType, Stream, StreamContext};

use super::dflash2::DFlash2ContextCache;
use super::layer_cache::{Qwen3_5LayerSnapshot, replay_mtp_snapshot_to, snapshot_all_mtp};
use super::model::{PREFILL_STEP_SIZE, Qwen35Inner, async_eval_layer_caches};

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
            super::dflash2::SelectorPath::Device(ids) => (Vec::new(), Some(ids)),
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
        async_eval_layer_caches(&self.inner.caches);
        MxArray::async_eval_arrays(&[token]);
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

        let generation_stream = Stream::new(DeviceType::Gpu);
        let report_performance = params.report_performance;
        let generation_start = report_performance.then(Instant::now);
        let mut first_token_instant = None;
        let mut profiler = DecodeProfiler::new(
            ChatBackend::profiler_label(self, is_delta, is_streaming),
            ChatBackend::family_name(self),
        );
        profiler.set_prompt_tokens(prefill.len() as u32);
        profiler.snapshot_memory_before();
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
        let stream = Stream::new(DeviceType::Gpu);
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
        let draft_cache = array_fingerprints(step.context.cache_arrays_for_stepper_test())?;
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
                    assert_eq!(attention.verify_can_be_shapeless(seq_len), head_dim == 64);
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
            let stream = Stream::new(DeviceType::Gpu);
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
        let expected_draft = array_fingerprints(reference.context.cache_arrays_for_stepper_test())
            .expect("reference draft state");
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
        let draft_arrays = context.cache_arrays_for_stepper_test();
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
}
