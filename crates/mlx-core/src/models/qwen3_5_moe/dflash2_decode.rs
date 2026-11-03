//! DFlash2 whole-turn integration for Qwen3.5/3.6/3.8 MoE.

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::engine::backend::{DsparkBackend, DsparkProposal, DsparkStepper, DsparkVerifyOutput};
use crate::transformer::KvFormat;

use super::layer_cache::{
    Qwen3_5LayerCache, Qwen3_5LayerSnapshot, replay_mtp_snapshot_to, rewind_full_attention_to,
    snapshot_all_mtp,
};
use super::model::{PREFILL_STEP_SIZE, Qwen35MoeInner};
use crate::models::qwen3_5::dflash2::DFlash2ContextCache;
use crate::models::qwen3_5::dflash2_state::trace::PrefixTracker;
use crate::models::qwen3_5::dflash2_turn::{DFlash2Target, DFlash2TurnState};
use crate::models::qwen3_5::gdn_blob::GdnStateBlobs;

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

fn dflash2_gdn_blob_enabled() -> bool {
    std::env::var("MLX_DFLASH2_GDN_BLOB")
        .ok()
        .is_none_or(|v| v.trim() != "0")
}

// FC+merged-KV costs cross over between retained heights 6 and 7.
fn prefer_full_width_context(keep: usize) -> bool {
    (7..=8).contains(&keep)
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

impl DFlash2Target for Qwen35MoeInner {
    const PREFILL_STEP_SIZE: i64 = PREFILL_STEP_SIZE;

    fn label(&self) -> &'static str {
        "Qwen3.5 MoE"
    }

    fn dflash2(&self) -> Option<&crate::models::qwen3_5::dflash2::DFlash2Model> {
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

    fn dense_config(&self) -> std::borrow::Cow<'_, crate::models::qwen3_5::Qwen3_5Config> {
        std::borrow::Cow::Owned(self.config.to_dense_config())
    }

    fn set_cached_token_history(&mut self, history: Vec<u32>) {
        self.cached_token_history = history;
    }

    fn config_kv_format(&self) -> KvFormat {
        self.config.to_dense_config().kv_format()
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
    ) -> Result<(MxArray, Vec<MxArray>)> {
        let (logits, taps, _) = super::model::forward_dflash2_with_taps(
            self,
            input_ids,
            tap_layers,
            false,
            super::model::DFlash2LogitsSpan::LastRow,
            None,
        )?;
        Ok((logits, taps))
    }

    fn init_caches_sync(&mut self) -> Result<()> {
        Qwen35MoeInner::init_caches_sync(self)
    }

    fn flat_lane_authoritative(&self) -> bool {
        self.dflash2_flat_lane_authoritative && !self.flat_mtp_caches_desynced
    }

    fn on_flat_prefill_done(&mut self) {
        self.dflash2_flat_lane_authoritative = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::{ChatBackend, SpecFrontier, TurnOutput, WholeTurnArgs};
    use crate::engine::extract_chat_params;
    use crate::engine::plan::{DecoderPlan, MediaCapabilities, MediaInputs, TurnPlan};
    use crate::engine::types::{ChatConfig, ChatResult};
    use crate::models::forward::PREFILL_CANCELLED;
    use crate::models::quantized_linear::LinearProj;
    use crate::models::qwen3_5::dflash2_turn::{
        dflash2_chat_turn, dflash2_fail_closed, dflash2_prefill, flat_attention_frontier,
    };
    use crate::nn::Linear;
    use crate::stream::{Stream, StreamContext};
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
        max_new_tokens: i32,
    ) -> Result<ChatResult> {
        let config = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(max_new_tokens),
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
        match dflash2_chat_turn(inner, &mut args)? {
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
            let (prefill_logits, state) = dflash2_prefill(inner, &[1, 2, 3, 4], 0, stream)?;
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
                let (_, state) = dflash2_prefill(&mut inner, &[1, 2, 3, 4], 0, stream)?;
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
        let (_, state) = dflash2_prefill(&mut inner, &[1, 2, 3, 4], 0, stream)?;
        inner.dflash2_turn_state = Some(state);
        let mut step = inner.begin_dspark_decode(2)?;
        step.verify(&[3, 4, 5])?.logits.eval();
        assert!(
            step.verify(&[3]).is_err(),
            "uncommitted verify cannot be overwritten"
        );
        drop(step);
        inner.dflash2_flat_lane_authoritative = true;
        let err = dflash2_fail_closed(&mut inner, Error::from_reason("injected verify failure"));
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
        let first = run_tiny_dflash_turn(&mut inner, &tokenizer, &[1, 2, 3, 4], false, 1)?;
        assert_eq!(first.finish_reason, "length");
        assert_eq!(first.num_tokens, 1);
        assert_eq!(first.cached_tokens, 0);

        // The fixed head makes turn 1's saved history [1, 2, 3, 4, 7].
        // Turn 2 strictly extends it and would reuse five tokens if that
        // history survived the completed reuse_cache=false request.
        ChatBackend::set_cache_owner_id(&mut inner, "owner", Some("root"));
        let second = run_tiny_dflash_turn(&mut inner, &tokenizer, &[1, 2, 3, 4, 7, 8], true, 1)?;
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
            let (prefill_logits, state) = dflash2_prefill(inner, prefix, 0, stream)?;
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
            let (prefill, state) = dflash2_prefill(&mut inner, &[1, 2, 3, 4], 0, stream)?;
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
        let (prefill_logits, state) = dflash2_prefill(&mut inner, &[1, 2, 3, 4], 0, stream)?;
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
        let _ = dflash2_fail_closed(&mut inner, error);
        assert!(inner.caches.is_none());
        assert!(inner.dflash2_context.is_none());
        assert!(!inner.dflash2_flat_lane_authoritative);
        Ok(())
    }
    #[test]
    fn prefix_event_names_the_owner_flip() -> Result<()> {
        use crate::models::qwen3_5::dflash2_state::trace::capture::capture_prefix_events;
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let mut inner = tiny_dflash_inner(0xDFA5_3603);
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let (result, events) = capture_prefix_events(|| -> Result<u32> {
            let first: Vec<u32> = (0..20).map(|i| (i * 3) % 15 + 1).collect();
            ChatBackend::set_cache_owner_id(&mut inner, "owner-a", Some("root"));
            run_tiny_dflash_turn(&mut inner, &tokenizer, &first, true, 2)?;
            let second: Vec<u32> = inner
                .cached_token_history
                .iter()
                .copied()
                .chain([2, 3])
                .collect();
            ChatBackend::set_cache_owner_id(&mut inner, "owner-b", Some("root"));
            Ok(run_tiny_dflash_turn(&mut inner, &tokenizer, &second, true, 2)?.cached_tokens)
        });
        assert_eq!(result?, 0);
        assert_eq!(events.len(), 2, "one event per turn: {events:?}");
        assert_eq!(events[0]["miss_reason"], "cold_first");
        assert_eq!(events[1]["hit_kind"], "none");
        assert_eq!(events[1]["miss_reason"], "invalidated:set_cache_owner_id");
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
            let (logits, state) = dflash2_prefill(&mut inner, &[1, 2, 3, 4], 0, stream)?;
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
                let error = dflash2_fail_closed(
                    &mut inner,
                    Error::from_reason("injected interruption after full-width context commit"),
                );
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
                let (prefill, state) = dflash2_prefill(&mut inner, &[1, 2, 3, 4], 0, stream)?;
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
            let (prefill, state) = dflash2_prefill(&mut inner, &[1, 2, 3, 4], 0, stream)?;
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
            let _ = dflash2_fail_closed(&mut inner, error);
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

    /// A cancel raised before the prefill keeps the live session: no Command
    /// reset and no prefill chunk.
    #[test]
    fn cancel_before_prefill_keeps_the_session_and_runs_no_chunk() {
        let mut inner = tiny_dflash_inner(0xDFA5_3509);
        reset_flat_fixture(&mut inner);
        ChatBackend::set_turn_cancel_flag(
            &mut inner,
            Some(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                true,
            ))),
        );

        let err = dflash2_prefill(&mut inner, &[1, 2, 3, 4], 0, Stream::generation())
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

    #[test]
    fn dflash2_golden_traces() -> Result<()> {
        use crate::models::qwen3_5::dflash2_decode::tests::{
            golden, golden_turn_prompts, push_turn_state,
        };
        let Some(mut goldens) = golden::GoldenRun::from_env() else {
            return Ok(());
        };
        assert!(
            crate::engine::persistence::compiled_forward_backend_available()
                && unsafe { mlx_sys::mlx_default_device() } == 1,
            "DFlash2 goldens are recorded on Metal as the default device"
        );
        assert!(
            std::env::var_os("MLX_DISABLE_COMPILE").is_none(),
            "DFlash2 goldens are recorded with MLX compilation enabled"
        );
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        for head_dim in [64, 32] {
            let mut inner = tiny_dflash_inner_with_head_dim(0xDFA5_3504, head_dim);
            let mut record = golden::GoldenRecord::default();
            for turn in 0..3 {
                let prompt = golden_turn_prompts(&inner.cached_token_history, turn);
                ChatBackend::set_cache_owner_id(&mut inner, "owner", Some("root"));
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
            goldens.case(
                "qwen3_5_moe",
                &format!("chat_turns_bf16_d{head_dim}"),
                &record,
            );
        }
        goldens.finish();
        Ok(())
    }
}
