//! DFlash2 whole-turn integration for Qwen3.5/3.6/3.8 MoE.

use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChatBackend, DsparkBackend, DsparkProposal, DsparkStepper, DsparkVerifyOutput, FinalizeArgs,
    ResetScope, TurnOutput, WholeTurnArgs,
};
use crate::engine::decode::TurnStreaming;
use crate::engine::dspark_turn::{DsparkTurnArgs, run_dspark_turn};
use crate::engine::finalize::compute_performance_metrics;
use crate::engine::params::generated_capacity_hint;
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::stream::{Stream, StreamContext};
use crate::transformer::KvFormat;

use super::layer_cache::{
    Qwen3_5LayerCache, Qwen3_5LayerSnapshot, replay_mtp_snapshot_to, rewind_full_attention_to,
    snapshot_all_mtp,
};
use super::model::{PREFILL_STEP_SIZE, Qwen35MoeInner};
use crate::models::qwen3_5::dflash2::DFlash2ContextCache;
use crate::models::qwen3_5::gdn_blob::GdnStateBlobs;

pub(crate) struct DFlash2TurnState {
    context: DFlash2ContextCache,
    next_position: i32,
}

struct PendingGdn {
    tape: Vec<Option<super::gated_delta_net::GdnLayerTape>>,
    keep: usize,
    keep_array: MxArray,
    base_position: i32,
}

struct DeferredGdnState {
    empty: Vec<Option<super::gated_delta_net::GdnLayerTape>>,
    zero: MxArray,
    pending: Option<PendingGdn>,
    physical_position: i32,
    verifying: bool,
}

pub(crate) struct Qwen35MoeDFlash2Stepper<'a> {
    inner: &'a mut Qwen35MoeInner,
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
    /// Packed recurrent and convolution state for one fused accepted-prefix commit.
    gdn_blobs: Option<GdnStateBlobs>,
    deferred_gdn: Option<DeferredGdnState>,
    verify_confidence: Option<MxArray>,
    verify_row_budget: Option<MxArray>,
}

use crate::models::qwen3_5::dflash2_decode::{
    constrain_dflash2_context_params, dflash2_decode_cache_limit, dflash2_final_token_fits_context,
    dflash2_kv_budget_bytes, dflash2_kv_reserve_rows, dflash2_memory_snapshot, kv_row_bytes,
    min_fa_capacity, reserve_dflash2_target_kv, reusable_dflash2_prefix,
};

fn dflash2_gdn_blob_enabled() -> bool {
    std::env::var("MLX_DFLASH2_GDN_BLOB")
        .ok()
        .is_none_or(|v| v.trim() != "0")
}

// FC+merged-KV costs cross over between retained heights 6 and 7.
fn prefer_full_width_context(keep: usize) -> bool {
    (7..=8).contains(&keep)
}

fn flat_attention_frontier(inner: &Qwen35MoeInner) -> Option<usize> {
    let mut frontiers = inner
        .caches
        .as_ref()?
        .iter()
        .filter_map(|cache| match cache {
            Qwen3_5LayerCache::FullAttention(kv) => usize::try_from(kv.get_offset()).ok(),
            Qwen3_5LayerCache::Linear(_) => None,
        });
    let first = frontiers.next()?;
    frontiers.all(|frontier| frontier == first).then_some(first)
}

impl Qwen35MoeDFlash2Stepper<'_> {
    fn ensure_clean(&self, operation: &str) -> Result<()> {
        if self.snapshot.is_some()
            || self.tape.is_some()
            || self.tapped.is_some()
            || self.verified_ids.is_some()
            || self.verified_ids_device.is_some()
            || self
                .deferred_gdn
                .as_ref()
                .is_some_and(|state| state.verifying)
        {
            return Err(Error::from_reason(format!(
                "Qwen3.5 MoE DFlash2 {operation}: prior verify was not committed"
            )));
        }
        Ok(())
    }

    /// Publish a physically representable frontier before any ordinary consumer.
    /// A failure leaves the pending owner alive until the outer turn reset.
    fn settle_pending_gdn(&mut self, force: bool) -> Result<()> {
        let Some(state) = self.deferred_gdn.as_ref() else {
            return Ok(());
        };
        let Some(pending) = state.pending.as_ref() else {
            return Ok(());
        };
        if pending.keep == 0
            || pending.keep > 8
            || pending.base_position != state.physical_position
            || pending.base_position.checked_add(pending.keep as i32) != Some(self.next_position)
        {
            return Err(Error::from_reason(
                "deferred GDN settlement frontier mismatch",
            ));
        }
        let blobs = self
            .gdn_blobs
            .as_ref()
            .ok_or_else(|| Error::from_reason("deferred GDN blobs missing"))?;
        let next = blobs
            .commit(&pending.tape, pending.keep)?
            .ok_or_else(|| Error::from_reason("deferred GDN settlement declined"))?;
        if force {
            MxArray::eval_arrays(&[&next.conv, &next.recurrent])?;
        }
        next.apply_views(
            self.inner
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("deferred GDN caches missing"))?,
        )?;
        self.gdn_blobs = Some(next);
        let state = self
            .deferred_gdn
            .as_mut()
            .ok_or_else(|| Error::from_reason("deferred GDN state missing"))?;
        state.pending = None;
        state.physical_position = self.next_position;
        Ok(())
    }

    fn prepare_gdn_verify(&mut self, rows: usize) -> Result<()> {
        if self.deferred_gdn.is_none() {
            return Ok(());
        }
        let deferred = rows == 8
            && self.gdn_blobs.is_some()
            && !self.inner.dflash2_compiled_verify_disabled
            && !std::env::var("MLX_MOE_DFLASH2_COMPILE").is_ok_and(|v| v == "0")
            && crate::engine::persistence::compiled_forward_backend_available();
        if !deferred {
            self.settle_pending_gdn(false)?;
        }
        let state = self
            .deferred_gdn
            .as_mut()
            .ok_or_else(|| Error::from_reason("deferred GDN state missing"))?;
        let logical = match state.pending.as_ref() {
            Some(pending) if pending.base_position == state.physical_position => {
                pending.base_position.checked_add(pending.keep as i32)
            }
            None => Some(state.physical_position),
            _ => None,
        };
        if logical != Some(self.next_position) {
            return Err(Error::from_reason("deferred GDN verify frontier mismatch"));
        }
        state.verifying = deferred;
        Ok(())
    }

    fn target_forward_array(
        &mut self,
        input: &MxArray,
        budget: Option<&MxArray>,
    ) -> Result<(
        MxArray,
        Vec<MxArray>,
        Vec<Option<super::gated_delta_net::GdnLayerTape>>,
    )> {
        if let Some(state) = self.deferred_gdn.as_ref().filter(|state| state.verifying) {
            let pending = crate::models::qwen3_5::gdn_blob::GdnDeferredInput {
                tape: state
                    .pending
                    .as_ref()
                    .map_or(state.empty.as_slice(), |p| p.tape.as_slice()),
                keep: state
                    .pending
                    .as_ref()
                    .map_or(&state.zero, |p| &p.keep_array),
            };
            let destinations = self
                .gdn_blobs
                .as_ref()
                .ok_or_else(|| Error::from_reason("deferred GDN destinations missing"))?
                .destinations();
            super::model::forward_dflash2_deferred(
                self.inner,
                input,
                &self.tap_layers,
                destinations,
                budget,
                &pending,
            )
        } else {
            super::model::forward_dflash2_with_budget(
                self.inner,
                input,
                &self.tap_layers,
                true,
                super::model::DFlash2LogitsSpan::All,
                self.gdn_blobs.as_ref().map(GdnStateBlobs::destinations),
                budget,
            )
        }
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
        self.target_forward_array(&input, None)
    }

    fn append_tapped(&mut self, tapped: &[MxArray], token_ids: &[u32]) -> Result<()> {
        let keep = token_ids.len();
        let draft = self
            .inner
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE DFlash2 model disappeared"))?;
        let full_width = prefer_full_width_context(keep)
            && draft.config.block_size == 7
            && tapped
                .first()
                .is_some_and(|hidden| hidden.shape_at(1).is_ok_and(|rows| rows == 8));
        if full_width {
            let fused = draft.fuse_context(tapped)?;
            self.context
                .append_full_width_prefix(draft, &fused, self.next_position, token_ids)?;
        } else {
            let kept = tapped
                .iter()
                .map(|hidden| hidden.slice_axis(1, 0, keep as i64))
                .collect::<Result<Vec<_>>>()?;
            let fused = draft.fuse_context(&kept)?;
            self.context
                .append(draft, &fused, self.next_position, token_ids)?;
        }
        self.next_position = self.next_position.saturating_add(keep as i32);
        Ok(())
    }
}

impl DsparkStepper for Qwen35MoeDFlash2Stepper<'_> {
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
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE DFlash2 model disappeared"))?;
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
        self.verify_confidence = None;
        self.verify_row_budget = None;
        // Preserve eight physical rows; shorter tail blocks and non-greedy
        // paths keep complete verification and their established semantics.
        let adaptive = params.mtp_adaptive_depth && greedy_fast && max_len == 7;
        let (path, draft_sparse_dists) = draft.propose(
            &self.inner.embedding,
            self.inner.lm_head.as_ref(),
            &mut self.context,
            anchor_id,
            max_len,
            temperature,
            greedy_fast,
            adaptive.then_some(&mut self.verify_confidence),
            rng,
        )?;
        if let Some(confidence) = self.verify_confidence.as_ref() {
            let policy = self.inner.dflash2_verify_budget.parameters(false, true);
            let mut values = Vec::with_capacity(18);
            values.push(policy.acceptance.slope);
            values.extend(policy.acceptance.intercepts);
            values.extend(policy.costs);
            values.push(if policy.every_row { 1.0 } else { 0.0 });
            let parameters = MxArray::from_float32(&values, &[18])?;
            self.verify_row_budget = Some(MxArray::from_handle(
                unsafe {
                    mlx_sys::mlx_dflash2_verify_budget(
                        confidence.as_raw_ptr(),
                        parameters.as_raw_ptr(),
                    )
                },
                "dflash2_verify_budget",
            )?);
        }
        let (draft_ids, device_draft_ids) = match path {
            crate::models::qwen3_5::dflash2::SelectorPath::Device(ids) => {
                // Queue the draft before the verify graph is built so the GPU
                // is not idle during that host work. Off Metal this call is a
                // blocking eval, which would only serialize the two builds.
                if unsafe { mlx_sys::mlx_metal_is_available() } {
                    let mut roots = vec![&ids];
                    roots.extend(self.verify_row_budget.as_ref());
                    MxArray::async_eval_arrays(&roots);
                }
                (Vec::new(), Some(ids))
            }
            crate::models::qwen3_5::dflash2::SelectorPath::Host(ids) => (ids, None),
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
                "Qwen3.5 MoE DFlash2 verify block must not be empty",
            ));
        }
        self.ensure_clean("verify")?;
        // Host verification includes teacher forcing and target-only steps.
        // Its tokens need not be the selector's path, so never mask or label it.
        self.verify_confidence = None;
        self.verify_row_budget = None;
        self.prepare_gdn_verify(verify_ids.len())?;
        let snapshot = snapshot_all_mtp(
            self.inner.caches.as_ref().ok_or_else(|| {
                Error::from_reason("Qwen3.5 MoE DFlash2 target caches are absent")
            })?,
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
        self.prepare_gdn_verify(
            usize::try_from(verify_ids.shape_at(0)?)
                .map_err(|_| Error::from_reason("negative verify rows"))?,
        )?;
        let snapshot = snapshot_all_mtp(
            self.inner.caches.as_ref().ok_or_else(|| {
                Error::from_reason("Qwen3.5 MoE DFlash2 target caches are absent")
            })?,
            false,
        )?;
        let input = verify_ids.reshape(&[1, verify_ids.shape_at(0)?])?;
        let budget = self.verify_row_budget.clone();
        let (logits, tapped, tape) = self.target_forward_array(&input, budget.as_ref())?;
        self.snapshot = Some(snapshot);
        self.tape = Some(tape);
        self.tapped = Some(tapped);
        self.verified_ids_device = Some(verify_ids.clone());
        Ok(DsparkVerifyOutput { logits })
    }

    fn device_verification_budget(&self) -> Option<&MxArray> {
        self.verify_row_budget.as_ref()
    }

    fn observe_device_verification(
        &mut self,
        rows: usize,
        raw_matches: usize,
        milliseconds: f64,
    ) -> Result<()> {
        let Some(confidence) = self.verify_confidence.take() else {
            return Ok(());
        };
        self.verify_row_budget = None;
        // Budget is a dependent root of confidence, already forced at acceptance.
        let values = confidence.to_float32()?;
        let confidence: [f32; 7] = values
            .as_ref()
            .try_into()
            .map_err(|_| Error::from_reason("DFlash2 confidence width changed after verify"))?;
        self.inner
            .dflash2_verify_budget
            .observe(
                super::verify_budget::CycleObservation {
                    confidence,
                    verify_rows: rows as u32,
                    accepted: raw_matches as u32,
                    labels: true,
                },
                false,
                milliseconds,
            )
            .map_err(Error::from_reason)?;
        Ok(())
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        if keep == 0 || keep > total_written {
            return Err(Error::from_reason(format!(
                "Qwen3.5 MoE DFlash2 invalid commit keep={keep}, total={total_written}"
            )));
        }
        let snapshot = self.snapshot.take().ok_or_else(|| {
            Error::from_reason("Qwen3.5 MoE DFlash2 commit has no target snapshot")
        })?;
        let tape = self
            .tape
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE DFlash2 commit has no GDN tape"))?;
        let tapped = self
            .tapped
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE DFlash2 commit has no target taps"))?;
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
                    "Qwen3.5 MoE DFlash2 commit has no token provenance",
                ));
            }
        };
        if verified_ids.len() != total_written {
            return Err(Error::from_reason(format!(
                "Qwen3.5 MoE DFlash2 commit wrote {total_written} rows for {} token ids",
                verified_ids.len()
            )));
        }
        let caches =
            self.inner.caches.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen3.5 MoE DFlash2 target caches are absent")
            })?;
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
        let deferred = self
            .deferred_gdn
            .as_ref()
            .is_some_and(|state| state.verifying);
        if deferred && (total_written != 8 || keep > 8) {
            return Err(Error::from_reason("deferred GDN commit width changed"));
        }
        let next_blobs = if deferred {
            let blobs = self
                .gdn_blobs
                .as_ref()
                .ok_or_else(|| Error::from_reason("deferred GDN commit blobs missing"))?;
            Some(blobs.adopt(&tape)?.ok_or_else(|| {
                Error::from_reason("deferred GDN settled destinations were not evaluated")
            })?)
        } else {
            match &self.gdn_blobs {
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
            }
        };
        match next_blobs {
            Some(next) => {
                // Rewind first: a failure here leaves the linear slots on the
                // pre-commit blobs, consistent with `self.gdn_blobs`.
                rewind_full_attention_to(caches, &snapshot, keep, "Qwen3.5 MoE DFlash2 commit")?;
                next.apply_views(caches)?;
                self.gdn_blobs = Some(next);
            }
            None => {
                if self.gdn_blobs.take().is_some() {
                    tracing::debug!(
                        keep,
                        "Qwen3.5 MoE DFlash2 fused GDN commit declined; per-layer replay for the rest of the turn"
                    );
                }
                replay_mtp_snapshot_to(
                    caches,
                    &snapshot,
                    &tape,
                    keep,
                    false,
                    "Qwen3.5 MoE DFlash2 commit",
                )?;
            }
        }
        if deferred {
            let state = self
                .deferred_gdn
                .as_mut()
                .ok_or_else(|| Error::from_reason("deferred GDN state missing"))?;
            state.physical_position = self.next_position;
            state.pending = Some(PendingGdn {
                tape,
                keep,
                keep_array: MxArray::from_int32(&[keep as i32], &[1])?,
                base_position: self.next_position,
            });
            state.verifying = false;
        }
        self.append_tapped(&tapped, &verified_ids[..keep])?;
        if !deferred && let Some(state) = self.deferred_gdn.as_mut() {
            state.physical_position = self.next_position;
        }
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
                "Qwen3.5 MoE DFlash2 commit wrote {total_written} rows for {} supplied token ids",
                verified_ids.len()
            )));
        }
        match (&self.verified_ids, &self.verified_ids_device) {
            (Some(expected), None) if expected.as_slice() == verified_ids => {}
            (None, Some(device))
                if device.ndim()? == 1 && device.shape_at(0)? as usize == total_written => {}
            (Some(_), None) => {
                return Err(Error::from_reason(
                    "Qwen3.5 MoE DFlash2 supplied commit provenance disagrees with host verify input",
                ));
            }
            _ => {
                return Err(Error::from_reason(
                    "Qwen3.5 MoE DFlash2 commit has invalid token provenance ownership",
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

    fn finish(mut self) -> Result<()> {
        self.ensure_clean("finish")?;
        self.settle_pending_gdn(true)?;
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

impl DsparkBackend for Qwen35MoeInner {
    type DsparkDecode<'a>
        = Qwen35MoeDFlash2Stepper<'a>
    where
        Self: 'a;

    fn begin_dspark_decode(&mut self, block_size: usize) -> Result<Self::DsparkDecode<'_>> {
        let expected = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE has no loaded DFlash2 companion"))?
            .config
            .block_size;
        if block_size != expected {
            return Err(Error::from_reason(format!(
                "Qwen3.5 MoE DFlash2 block size {block_size} does not match checkpoint {expected}"
            )));
        }
        let state = self.dflash2_turn_state.take().ok_or_else(|| {
            Error::from_reason("Qwen3.5 MoE DFlash2 decode requires tapped prefill state")
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
                        "Qwen3.5 MoE DFlash2: GDN state not packable; per-layer commit path"
                    ),
                }
                blobs
            }
            _ => None,
        };
        let deferred_gdn = if std::env::var_os("MLX_MOE_DFLASH2_DEFER_GDN").as_deref()
            == Some(std::ffi::OsStr::new("1"))
            && self.config.hidden_size == 2048
            && self.config.num_layers == 40
            && self.config.linear_num_key_heads == 16
            && self.config.linear_num_value_heads == 32
            && self.config.linear_key_head_dim == 128
            && self.config.linear_value_head_dim == 128
            && self.config.linear_conv_kernel_dim == 4
            && let Some(blobs) = gdn_blobs.as_ref()
        {
            Some(DeferredGdnState {
                empty: blobs.empty_deferred_tape(self.layers.len())?,
                zero: MxArray::from_int32(&[0], &[1])?,
                pending: None,
                physical_position: state.next_position,
                verifying: false,
            })
        } else {
            None
        };
        Ok(Qwen35MoeDFlash2Stepper {
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
            deferred_gdn,
            verify_confidence: None,
            verify_row_budget: None,
        })
    }
}

impl Qwen35MoeInner {
    fn dflash2_prefill(
        &mut self,
        tokens: &[u32],
        position_base: i32,
        stream: Stream,
    ) -> Result<(MxArray, DFlash2TurnState)> {
        if tokens.is_empty() {
            return Err(Error::from_reason(
                "Qwen3.5 MoE DFlash2 requires at least one prefill token",
            ));
        }
        let draft = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE has no loaded DFlash2 companion"))?;
        let tap_layers = draft.config.target_layers.clone();
        let draft_config = draft.config.clone();
        let mut context = match self.dflash2_context.take() {
            Some(context) if context.logical_len() == position_base => context,
            Some(context) => {
                return Err(Error::from_reason(format!(
                    "Qwen3.5 MoE DFlash2 context length {} does not match cached prefix {position_base}",
                    context.logical_len()
                )));
            }
            None if position_base == 0 => DFlash2ContextCache::new(&draft_config),
            None => {
                return Err(Error::from_reason(format!(
                    "Qwen3.5 MoE DFlash2 cached prefix {position_base} has no retained draft context"
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
                    None,
                )?
            };
            let draft = self
                .dflash2
                .as_ref()
                .ok_or_else(|| Error::from_reason("Qwen3.5 MoE DFlash2 model disappeared"))?;
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
                crate::models::forward::eval_layer_caches(&self.caches)?;
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

    /// Activation element type the target's full-attention caches hold (BF16
    /// format) or quantize from (int8 format): the embedding output dtype,
    /// or the allocated BF16 buffer's own.
    fn dflash2_target_kv_dtype(&self) -> Result<DType> {
        let allocated = self.caches.iter().flatten().find_map(|cache| match cache {
            Qwen3_5LayerCache::FullAttention(kv) if kv.format() == KvFormat::Bf16 => kv.keys_ref(),
            _ => None,
        });
        match allocated {
            Some(keys) => keys.dtype(),
            None => self
                .embedding
                .forward(&MxArray::from_int32(&[0], &[1, 1])?)?
                .dtype(),
        }
    }

    /// Row format of the target's full-attention caches (every FA slot
    /// shares the config's format).
    fn dflash2_target_kv_format(&self) -> KvFormat {
        self.caches
            .iter()
            .flatten()
            .find_map(|cache| match cache {
                Qwen3_5LayerCache::FullAttention(kv) => Some(kv.format()),
                Qwen3_5LayerCache::Linear(_) => None,
            })
            .unwrap_or_else(|| self.config.to_dense_config().kv_format())
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
            self.dflash2_target_kv_format(),
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
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE DFlash2 model disappeared"))?
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
            None,
        )?;
        let fused = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE DFlash2 model disappeared"))?
            .fuse_context(&taps)?;
        let mut context = self.dflash2_context.take().ok_or_else(|| {
            Error::from_reason("Qwen3.5 MoE DFlash2 final token has no retained draft context")
        })?;
        let base = context.logical_len();
        let append_result = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE DFlash2 model disappeared"))
            .and_then(|draft| context.append(draft, &fused, base, &[token]));
        self.dflash2_context = Some(context);
        append_result?;
        crate::models::forward::eval_layer_caches(&self.caches)
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
            self.dflash2_flat_lane_authoritative && !self.flat_mtp_caches_desynced;
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
        if let Err(error) = crate::models::forward::eval_layer_caches(&self.caches) {
            return Err(self.dflash2_fail_closed(error));
        }
        if report_performance {
            first_token_instant = Some(Instant::now());
        }
        self.dflash2_turn_state = Some(state);
        self.dflash2_flat_lane_authoritative = true;

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
            // Decode-phase allocator ceiling: prefill (2048-row chunks) has
            // already run under the load-time cap; the verify/propose loop
            // only needs room to recycle one cycle's transients, so the
            // free-pool is capped for exactly this scope and lifted on drop.
            let _decode_cache_cap =
                crate::cache_limit::coordinator().push_decode_limit(dflash2_decode_cache_limit(
                    &self.config.to_dense_config(),
                    block_size,
                    tokens.len() as u64 + params.max_new_tokens.max(0) as u64,
                ));
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
        // This specialized path bypasses the generic save hook's non-reuse cleanup.
        if !params.reuse_cache {
            ChatBackend::reset_caches(self, ResetScope::Command)?;
        }

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
    use crate::engine::backend::SpecFrontier;
    use crate::engine::extract_chat_params;
    use crate::engine::plan::{DecoderPlan, MediaCapabilities, MediaInputs, TurnPlan};
    use crate::engine::types::{ChatConfig, ChatResult};
    use crate::models::quantized_linear::LinearProj;
    use crate::nn::Linear;
    use crate::tokenizer::Qwen3Tokenizer;
    use rand::SeedableRng;

    impl Qwen35MoeDFlash2Stepper<'_> {
        fn attention_frontier(&self) -> Option<u64> {
            flat_attention_frontier(self.inner).and_then(|frontier| u64::try_from(frontier).ok())
        }

        fn frontier(&self) -> Option<SpecFrontier> {
            Some(SpecFrontier {
                attn_tokens: self.attention_frontier()?,
                recurrent_tokens: Some(self.next_position.max(0) as u64),
            })
        }
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

    fn reset_flat_fixture(inner: &mut Qwen35MoeInner) {
        inner.caches = Some(Qwen3_5LayerCache::fresh_layer_caches(
            &inner.config.to_dense_config(),
        ));
        inner.dflash2_context = None;
        inner.dflash2_turn_state = None;
    }

    fn tiny_dflash_inner(seed: u64) -> Qwen35MoeInner {
        tiny_dflash_inner_with_head_dim(seed, 32)
    }

    fn tiny_dflash_inner_with_head_dim(seed: u64, head_dim: i32) -> Qwen35MoeInner {
        tiny_dflash_inner_with_config(seed, |cfg| cfg.head_dim = head_dim)
    }

    fn tiny_dflash_inner_with_config(
        seed: u64,
        adjust: impl FnOnce(&mut super::super::config::Qwen3_5MoeConfig),
    ) -> Qwen35MoeInner {
        let mut inner = super::super::model::scheduled_mtp::seeded_inner_with_config(seed, adjust);
        inner.paged_adapter = None;
        let mut head = Linear::new(
            inner.config.hidden_size as u32,
            inner.config.vocab_size as u32,
            Some(false),
        )
        .unwrap();
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
            .unwrap(),
        )
        .unwrap();
        inner.lm_head = Some(LinearProj::Standard(head));
        let dense = inner.config.to_dense_config();
        let draft =
            crate::models::qwen3_5::dflash2::tiny_dflash2_model_for_stepper_test(&dense).unwrap();
        draft.validate_target(&dense).unwrap();
        inner.dflash2 = Some(draft);
        inner
    }

    fn run_tiny_dflash_turn(
        inner: &mut Qwen35MoeInner,
        tokenizer: &std::sync::Arc<Qwen3Tokenizer>,
        tokens: &[u32],
        reuse_cache: bool,
    ) -> Result<ChatResult> {
        let config = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(1),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            mtp_adaptive_depth: Some(false),
            reuse_cache: Some(reuse_cache),
            reasoning_effort: Some("none".to_string()),
            ..ChatConfig::default()
        };
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
                decoder: DecoderPlan::Speculative(crate::engine::plan::SpeculativeKind::DraftModel),
            },
            sink: None,
            cancelled: None,
            media: MediaInputs {
                images: &[],
                audio: &[],
            },
        };
        match inner.dflash2_chat_turn(&mut args)? {
            TurnOutput::Complete(result) => Ok(*result),
            TurnOutput::Streamed => panic!("sync DFlash2 turn returned streamed output"),
        }
    }

    #[test]
    fn full_width_context_policy_crosses_over_at_keep_seven() {
        for keep in 0..=9 {
            assert_eq!(
                prefer_full_width_context(keep),
                matches!(keep, 7 | 8),
                "keep={keep}"
            );
        }
    }

    #[test]
    fn commit_provenance_validation_is_non_mutating_and_published_state_survives_consumption() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            eprintln!("skipping DFlash2 commit durability: Metal unavailable");
            return;
        }

        fn begin_verified_step(
            inner: &mut Qwen35MoeInner,
        ) -> Result<(Qwen35MoeDFlash2Stepper<'_>, u32)> {
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
        let assert_pending = |step: &Qwen35MoeDFlash2Stepper<'_>| {
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

    #[test]
    fn moe_dflash_rejected_suffix_and_full_accept_match_per_layer_replay() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let mut inner = tiny_dflash_inner_with_config(0xDFA5_3501, |cfg| {
            cfg.linear_key_head_dim = 128;
            cfg.linear_value_head_dim = 128;
        });
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        for keep in [1, 2, 3] {
            let mut traces = Vec::new();
            for packed in [false, true] {
                reset_flat_fixture(&mut inner);
                let (_, state) = inner.dflash2_prefill(&[1, 2, 3, 4], 0, stream)?;
                inner.dflash2_turn_state = Some(state);
                let mut step = inner.begin_dspark_decode(2)?;
                assert!(
                    step.gdn_blobs.is_some(),
                    "fixture must activate packed GDN state"
                );
                if !packed {
                    step.gdn_blobs = None;
                }
                let ids = [3, 7, 8];
                step.verify(&ids)?.logits.eval();
                step.commit_with_provenance(keep, ids.len(), &ids)?;
                if packed {
                    assert!(
                        step.gdn_blobs.is_some(),
                        "packed commit must not silently fall back"
                    );
                }
                assert_eq!(step.frontier().unwrap().attn_tokens, 4 + keep as u64);
                assert_eq!(
                    step.frontier().unwrap().recurrent_tokens,
                    Some(4 + keep as u64)
                );
                assert_eq!(
                    step.context.token_history(),
                    &[&[1, 2, 3, 4][..], &ids[..keep]].concat()
                );
                let mut target = Vec::new();
                crate::models::forward::CollectCacheArrays::collect_arrays(
                    &step.inner.caches,
                    &mut target,
                );
                let target = array_fingerprints(target)?;
                let mut draft = Vec::new();
                step.context.collect_arrays(&mut draft);
                let draft = array_fingerprints(draft)?;
                // The next verify reads the settled state, including any rollback.
                let logits = step.verify(&[6])?.logits;
                let logits = array_fingerprints(vec![&logits])?;
                step.commit(1, 1)?;
                step.finish()?;
                traces.push((target, draft, logits));
            }
            assert_eq!(
                traces[0], traces[1],
                "keep={keep}: packed commit versus replay"
            );
        }
        Ok(())
    }

    #[test]
    fn moe_dflash_state_reset_fails_closed_and_preserves_draft_weights() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let mut inner = tiny_dflash_inner(0xDFA5_3502);
        reset_flat_fixture(&mut inner);
        let stream = Stream::generation();
        let (_, state) = inner.dflash2_prefill(&[1, 2, 3, 4], 0, stream)?;
        inner.dflash2_turn_state = Some(state);
        let mut step = inner.begin_dspark_decode(2)?;
        step.verify(&[3, 4, 5])?.logits.eval();
        assert!(
            step.verify(&[3]).is_err(),
            "uncommitted verify cannot be overwritten"
        );
        drop(step);
        inner.dflash2_flat_lane_authoritative = true;
        let err = inner.dflash2_fail_closed(Error::from_reason("injected verify failure"));
        assert_eq!(err.reason, "injected verify failure");
        assert!(inner.caches.is_none());
        assert!(inner.cached_token_history.is_empty());
        assert!(inner.dflash2_context.is_none());
        assert!(inner.dflash2_turn_state.is_none());
        assert!(!inner.dflash2_flat_lane_authoritative);
        assert!(
            inner.dflash2.is_some(),
            "cache reset retains loaded draft weights"
        );
        assert!(
            inner.begin_dspark_decode(2).is_err(),
            "fresh tapped prefill is required"
        );
        Ok(())
    }

    #[test]
    fn moe_dflash_plan_and_params_use_companion_width() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let inner = tiny_dflash_inner(0xDFA5_3503);
        let plan = ChatBackend::execution_plan(&inner).speculative.unwrap();
        assert_eq!(plan.kind, crate::engine::plan::SpeculativeKind::DraftModel);
        assert!(!plan.supports_paged_attention);
        assert!(plan.supports_streaming);
        assert!(plan.supported_input_media.is_empty());
        let params = ChatBackend::resolve_params(
            &inner,
            &ChatConfig {
                mtp_depth: Some(99),
                mtp_adaptive_depth: Some(true),
                temperature: Some(0.0),
                max_new_tokens: Some(19),
                ..ChatConfig::default()
            },
        );
        assert_eq!(params.mtp_depth, 2);
        assert!(params.mtp_adaptive_depth);
        assert_eq!(params.max_new_tokens, 19);
        Ok(())
    }

    #[test]
    fn reuse_false_completion_does_not_seed_a_later_dflash_prefix() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let mut inner = tiny_dflash_inner(0xDFA5_3504);
        let hidden = inner.config.hidden_size as i64;
        let vocab = inner.config.vocab_size as i64;
        let mut head = Linear::new(hidden as u32, vocab as u32, Some(true))?;
        head.set_weight(&MxArray::zeros(&[vocab, hidden], Some(DType::BFloat16))?)?;
        let mut bias = [0.0f32; 16];
        bias[7] = 1.0;
        let bias = MxArray::from_float32(&bias, &[vocab])?.astype(DType::BFloat16)?;
        head.set_bias(Some(&bias))?;
        inner.lm_head = Some(LinearProj::Standard(head));

        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        ChatBackend::set_cache_owner_id(&mut inner, "owner", Some("root"));
        let first = run_tiny_dflash_turn(&mut inner, &tokenizer, &[1, 2, 3, 4], false)?;
        assert_eq!(first.finish_reason, "length");
        assert_eq!(first.num_tokens, 1);
        assert_eq!(first.cached_tokens, 0);

        // The fixed head makes turn 1's saved history [1, 2, 3, 4, 7].
        // Turn 2 strictly extends it and would reuse five tokens if that
        // history survived the completed reuse_cache=false request.
        ChatBackend::set_cache_owner_id(&mut inner, "owner", Some("root"));
        let second = run_tiny_dflash_turn(&mut inner, &tokenizer, &[1, 2, 3, 4, 7, 8], true)?;
        assert_eq!(
            second.cached_tokens, 0,
            "a non-reused completion must leave no prefix for the next request"
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
        let run = |inner: &mut Qwen35MoeInner,
                   prefix: &[u32],
                   ids: &[i32]|
         -> Result<Vec<(Vec<i64>, Vec<u32>)>> {
            reset_flat_fixture(inner);
            let stream = Stream::generation();
            let _ctx = StreamContext::new(stream);
            let (prefill_logits, state) = inner.dflash2_prefill(prefix, 0, stream)?;
            prefill_logits.eval();
            inner.dflash2_turn_state = Some(state);
            let input = MxArray::from_int32(ids, &[1, ids.len() as i64])?;
            let tap_layers: Vec<usize> = (0..inner.layers.len()).collect();
            let (logits, taps, tape) = super::super::model::forward_dflash2_with_taps(
                inner,
                &input,
                &tap_layers,
                true,
                super::super::model::DFlash2LogitsSpan::All,
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
        // D64 covers the vector FA path; D256 covers the segmented path used
        // by the real target in both KV formats. Width8 then6 then8 must reuse
        // the original width8 trace even though its prefix length changed.
        for (head_dim, kv_format) in [(64, "bf16"), (256, "bf16"), (256, "int8")] {
            let mut inner = tiny_dflash_inner_with_config(0xDFA5_5E3A, |cfg| {
                cfg.head_dim = head_dim;
                cfg.hidden_size = cfg.hidden_size.max(head_dim);
                cfg.kv_format = Some(kv_format.to_owned());
            });
            let cases: [(Vec<u32>, Vec<i32>, usize); 3] = [
                (vec![1, 2, 3, 4], vec![5, 6, 7, 8, 9, 10, 11, 12], 1),
                (vec![1, 2, 3, 4, 5, 6, 7], vec![9, 3, 14, 2, 6, 11], 1),
                (
                    vec![2, 4, 6, 8, 10, 12, 14, 3, 5],
                    vec![7, 8, 9, 10, 11, 12, 13, 14],
                    0,
                ),
            ];
            for (prefix, ids, expected_builds) in cases {
                super::super::model::take_dflash2_compiled_test_counts();
                inner.dflash2_compiled_verify_disabled = false;
                let compiled = run(&mut inner, &prefix, &ids)?;
                assert_eq!(
                    super::super::model::take_dflash2_compiled_test_counts(),
                    (1, expected_builds),
                    "compiled trace/replay mismatch: D{head_dim} {kv_format}, prefix{} rows{}",
                    prefix.len(),
                    ids.len()
                );
                inner.dflash2_compiled_verify_disabled = true;
                let eager = run(&mut inner, &prefix, &ids)?;
                assert_eq!(
                    super::super::model::take_dflash2_compiled_test_counts(),
                    (0, 0)
                );
                assert_eq!(
                    compiled.len(),
                    1 + inner.layers.len()
                        + 6 * inner.layers.iter().filter(|l| l.is_linear()).count()
                );
                for (index, (a, b)) in compiled.iter().zip(eager.iter()).enumerate() {
                    assert_eq!(
                        a,
                        b,
                        "verify output {index} differs: D{head_dim} {kv_format}, prefix{} rows{}",
                        prefix.len(),
                        ids.len()
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    fn compiled_budget_replays_dynamic_rows_without_changing_physical_width() -> Result<()> {
        if !crate::engine::persistence::compiled_forward_backend_available()
            || unsafe { mlx_sys::mlx_default_device() } != 1
        {
            return Ok(());
        }
        let mut inner = tiny_dflash_inner_with_head_dim(0xDFA5_35B1, 64);
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let ids = MxArray::from_int32(&[5, 6, 7, 8, 9, 10, 11, 12], &[1, 8])?;
        let mut reference = None;
        for (rows, builds) in [(None, 1), (Some(8), 1), (Some(1), 0), (Some(4), 0)] {
            reset_flat_fixture(&mut inner);
            let (prefill, state) = inner.dflash2_prefill(&[1, 2, 3, 4], 0, stream)?;
            prefill.eval();
            inner.dflash2_turn_state = Some(state);
            let budget = rows
                .map(|rows| MxArray::from_int32(&[rows], &[1]))
                .transpose()?;
            super::super::model::take_dflash2_compiled_test_counts();
            let (logits, _, _) = super::super::model::forward_dflash2_with_budget(
                &mut inner,
                &ids,
                &[1, 3],
                true,
                super::super::model::DFlash2LogitsSpan::All,
                None,
                budget.as_ref(),
            )?;
            logits.eval();
            assert_eq!(
                super::super::model::take_dflash2_compiled_test_counts(),
                (1, builds)
            );
            assert_eq!(logits.shape_at(1)?, 8);
            assert_eq!(flat_attention_frontier(&inner), Some(12));
            let valid = rows.unwrap_or(8) as i64;
            let current = logits
                .slice_axis(1, 0, valid)?
                .astype(DType::Float32)?
                .to_float32()?;
            if let Some(reference) = &reference {
                let expected: &Vec<f32> = reference;
                assert_eq!(current.as_ref(), &expected[..current.len()]);
            } else {
                reference = Some(current.to_vec());
            }
        }
        Ok(())
    }

    #[test]
    fn compiled_store_failure_is_not_retried_on_mutated_caches() -> Result<()> {
        if !crate::engine::persistence::compiled_forward_backend_available()
            || unsafe { mlx_sys::mlx_default_device() } != 1
        {
            return Ok(());
        }
        let mut inner = tiny_dflash_inner_with_head_dim(0xDFA5_35F1, 64);
        reset_flat_fixture(&mut inner);
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let (prefill_logits, state) = inner.dflash2_prefill(&[1, 2, 3, 4], 0, stream)?;
        // Production loading materializes every weight before tracing. Force
        // the random fixture's target weights before the shapeless invoke too.
        prefill_logits.eval();
        inner.dflash2_turn_state = Some(state);
        let mut step = inner.begin_dspark_decode(2)?;
        super::super::model::inject_dflash2_store_failure();
        let error = match step.verify(&[3, 4, 5]) {
            Err(error) => error,
            Ok(_) => panic!("injected compiled store failure must escape; no eager retry"),
        };
        assert!(
            error
                .reason
                .contains("injected compiled MoE KV store failure")
        );
        assert!(step.inner.dflash2_compiled_verify_disabled);
        assert_eq!(
            step.attention_frontier(),
            Some(4),
            "published offsets roll back before propagating failure"
        );
        drop(step);
        let _ = inner.dflash2_fail_closed(error);
        assert!(inner.caches.is_none());
        assert!(inner.dflash2_context.is_none());
        assert!(!inner.dflash2_flat_lane_authoritative);
        Ok(())
    }
    #[test]
    fn retained_draft_context_cannot_cross_owner_or_root_identity() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let mut inner = tiny_dflash_inner(0xDFA5_35A1);
        ChatBackend::set_cache_owner_id(&mut inner, "owner-a", Some("root-a"));
        inner.dflash2_context = Some(DFlash2ContextCache::new(
            &inner.dflash2.as_ref().unwrap().config,
        ));
        inner.dflash2_flat_lane_authoritative = true;
        ChatBackend::set_cache_owner_id(&mut inner, "owner-a", Some("root-a"));
        assert!(
            inner.dflash2_context.is_some(),
            "same owner keeps the retained draft"
        );
        ChatBackend::set_cache_owner_id(&mut inner, "owner-a", Some("root-b"));
        assert!(
            inner.dflash2_context.is_none(),
            "root identity change invalidates draft"
        );
        assert!(!inner.dflash2_flat_lane_authoritative);
        inner.dflash2_context = Some(DFlash2ContextCache::new(
            &inner.dflash2.as_ref().unwrap().config,
        ));
        ChatBackend::set_cache_owner_id(&mut inner, "owner-b", Some("root-b"));
        assert!(
            inner.dflash2_context.is_none(),
            "another owner cannot take retained flat state"
        );
        Ok(())
    }
    #[test]
    fn full_width_context_commit_then_failure_resets_and_recovers() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let mut inner = tiny_dflash_inner(0x8FC0_3501);
        inner.dflash2.as_mut().unwrap().config.block_size = 7;
        let stream = Stream::generation();
        let _stream = StreamContext::new(stream);
        let mut reference = None;
        for pass in 0..2 {
            reset_flat_fixture(&mut inner);
            let (logits, state) = inner.dflash2_prefill(&[1, 2, 3, 4], 0, stream)?;
            logits.eval();
            inner.dflash2_turn_state = Some(state);
            let mut step = inner.begin_dspark_decode(7)?;
            let ids = [3, 7, 8, 9, 10, 11, 12, 13];
            let keep = 7;
            step.verify(&ids)?.logits.eval();
            step.commit_with_provenance(keep, 8, &ids)?;
            assert_eq!(
                step.context.token_history(),
                &[&[1, 2, 3, 4][..], &ids[..keep]].concat()
            );
            assert_eq!(step.frontier().unwrap().attn_tokens, 4 + keep as u64);
            let mut target = Vec::new();
            crate::models::forward::CollectCacheArrays::collect_arrays(
                &step.inner.caches,
                &mut target,
            );
            let target = array_fingerprints(target)?;
            let mut draft = Vec::new();
            step.context.collect_arrays(&mut draft);
            let draft = array_fingerprints(draft)?;
            if pass == 0 {
                reference = Some((target, draft));
                // Model an abort after a later verification has written target
                // state but before it can commit its draft/context prefix.
                step.verify(&ids)?.logits.eval();
                drop(step);
                inner.dflash2_flat_lane_authoritative = true;
                let error = inner.dflash2_fail_closed(Error::from_reason(
                    "injected interruption after full-width context commit",
                ));
                assert!(error.reason.contains("injected interruption"));
                assert!(inner.caches.is_none());
                assert!(inner.dflash2_context.is_none());
                assert!(inner.dflash2_turn_state.is_none());
                assert!(inner.cached_token_history.is_empty());
                assert!(!inner.dflash2_flat_lane_authoritative);
                assert!(inner.dflash2.is_some());
                ChatBackend::set_cache_owner_id(
                    &mut inner,
                    "recovered-owner",
                    Some("recovered-root"),
                );
            } else {
                assert_eq!(
                    reference.as_ref().unwrap(),
                    &(target, draft),
                    "fresh owner must recover the exact committed target and draft state"
                );
                step.finish()?;
            }
        }
        Ok(())
    }
    fn tiny_deferred_inner(seed: u64) -> Qwen35MoeInner {
        let mut inner = tiny_dflash_inner_with_config(seed, |cfg| {
            cfg.head_dim = 64;
            cfg.linear_num_key_heads = 2;
            cfg.linear_num_value_heads = 2;
            cfg.linear_key_head_dim = 128;
            cfg.linear_value_head_dim = 128;
            cfg.linear_conv_kernel_dim = 4;
        });
        inner.dflash2.as_mut().unwrap().config.block_size = 7;
        inner
    }
    fn enable_tiny_deferred(step: &mut Qwen35MoeDFlash2Stepper<'_>) -> Result<()> {
        let blobs = step
            .gdn_blobs
            .as_ref()
            .ok_or_else(|| Error::from_reason("tiny deferred blobs absent"))?;
        step.deferred_gdn = Some(DeferredGdnState {
            empty: blobs.empty_deferred_tape(step.inner.layers.len())?,
            zero: MxArray::from_int32(&[0], &[1])?,
            pending: None,
            physical_position: step.next_position,
            verifying: false,
        });
        Ok(())
    }
    #[test]
    fn deferred_stepper_settles_finish_and_short_tail_before_warm_publication() -> Result<()> {
        if !crate::engine::persistence::compiled_forward_backend_available()
            || unsafe { mlx_sys::mlx_default_device() } != 1
        {
            return Ok(());
        }
        let mut inner = tiny_deferred_inner(0xDEF0_3501);
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        for tail in [false, true] {
            let mut reference = None;
            for deferred in [false, true] {
                reset_flat_fixture(&mut inner);
                let (prefill, state) = inner.dflash2_prefill(&[1, 2, 3, 4], 0, stream)?;
                prefill.eval();
                inner.dflash2_turn_state = Some(state);
                let mut step = inner.begin_dspark_decode(7)?;
                if deferred {
                    enable_tiny_deferred(&mut step)?;
                }
                let mut logits = Vec::new();
                for keep in [3, 8, 1, 5] {
                    let ids = [3, 4, 5, 6, 7, 8, 9, 10];
                    let verify = step.verify(&ids)?;
                    logits.push(array_fingerprints(vec![&verify.logits])?);
                    step.commit_with_provenance(keep, 8, &ids)?;
                    if deferred {
                        let state = step.deferred_gdn.as_ref().unwrap();
                        let pending = state
                            .pending
                            .as_ref()
                            .expect("accepted rows remain pending");
                        assert_eq!(pending.keep, keep);
                        assert_eq!(state.physical_position + keep as i32, step.next_position);
                        assert!(
                            step.inner.dflash2_context.is_none(),
                            "unsettled context cannot be published"
                        );
                    }
                }
                if tail {
                    let ids = [6, 7, 8];
                    let verify = step.verify(&ids)?;
                    logits.push(array_fingerprints(vec![&verify.logits])?);
                    if deferred {
                        assert!(
                            step.deferred_gdn.as_ref().unwrap().pending.is_none(),
                            "ordinary tail must settle first"
                        );
                    }
                    step.commit_with_provenance(2, 3, &ids)?;
                }
                let frontier = step.frontier();
                let tokens = step.context.token_history().to_vec();
                step.finish()?;
                let mut target = Vec::new();
                crate::models::forward::CollectCacheArrays::collect_arrays(
                    &inner.caches,
                    &mut target,
                );
                let target = array_fingerprints(target)?;
                let context = inner
                    .dflash2_context
                    .as_ref()
                    .expect("finish published settled context");
                let mut draft = Vec::new();
                context.collect_arrays(&mut draft);
                let result = (logits, target, array_fingerprints(draft)?, tokens, frontier);
                if let Some(reference) = &reference {
                    assert_eq!(
                        reference, &result,
                        "deferred settlement changed warm cache contents"
                    );
                } else {
                    reference = Some(result);
                }
            }
        }
        Ok(())
    }
    #[test]
    fn deferred_stepper_failure_drops_pending_and_recovers_new_owner() -> Result<()> {
        if !crate::engine::persistence::compiled_forward_backend_available()
            || unsafe { mlx_sys::mlx_default_device() } != 1
        {
            return Ok(());
        }
        let mut inner = tiny_deferred_inner(0xDEF0_3502);
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut reference = None;
        for pass in 0..3 {
            reset_flat_fixture(&mut inner);
            let (prefill, state) = inner.dflash2_prefill(&[1, 2, 3, 4], 0, stream)?;
            prefill.eval();
            inner.dflash2_turn_state = Some(state);
            let mut step = inner.begin_dspark_decode(7)?;
            enable_tiny_deferred(&mut step)?;
            let ids = [3, 4, 5, 6, 7, 8, 9, 10];
            let verify = step.verify(&ids)?;
            let fingerprint = array_fingerprints(vec![&verify.logits])?;
            if let Some(reference) = &reference {
                assert_eq!(
                    reference, &fingerprint,
                    "fresh owner must recover exact verifier output"
                );
            } else {
                reference = Some(fingerprint);
            }
            step.commit_with_provenance(3, 8, &ids)?;
            assert!(step.deferred_gdn.as_ref().unwrap().pending.is_some());
            let error = if pass == 0 {
                super::super::model::inject_dflash2_store_failure();
                let error = step
                    .verify(&ids)
                    .err()
                    .expect("injected store failure must fail closed");
                assert!(
                    error
                        .reason
                        .contains("injected compiled MoE KV store failure")
                );
                assert!(step.inner.dflash2_context.is_none());
                drop(step);
                error
            } else if pass == 1 {
                // Settlement failure cannot publish context ahead of physical state.
                step.deferred_gdn
                    .as_mut()
                    .unwrap()
                    .pending
                    .as_mut()
                    .unwrap()
                    .base_position -= 1;
                step.finish()
                    .expect_err("invalid pending frontier must reject publication")
            } else {
                step.finish()?;
                break;
            };
            let _ = inner.dflash2_fail_closed(error);
            assert!(inner.caches.is_none());
            assert!(inner.dflash2_context.is_none());
            assert!(inner.dflash2_turn_state.is_none());
            assert!(inner.cached_token_history.is_empty());
            assert!(!inner.dflash2_flat_lane_authoritative);
            ChatBackend::set_cache_owner_id(&mut inner, "deferred-recovery", Some("new-root"));
            assert!(
                inner.dflash2.is_some(),
                "failure retains the companion weights"
            );
            // Recreate the tiny target to exercise deferred retry without clearing
            // the production object's fail-closed compiled-graph latch.
            inner = tiny_deferred_inner(0xDEF0_3502);
        }
        Ok(())
    }
}
