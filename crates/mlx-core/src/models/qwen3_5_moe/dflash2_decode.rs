//! DFlash2 whole-turn integration for Qwen3.5/3.6/3.8 MoE.

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::engine::backend::{DsparkBackend, DsparkProposal, DsparkStepper, DsparkVerifyOutput};

use super::layer_cache::{Qwen3_5LayerSnapshot, replay_mtp_snapshot_to, snapshot_all_mtp};
use super::model::Qwen35MoeInner;
use crate::models::qwen3_5::dflash2::DFlash2ContextCache;
use crate::models::qwen3_5::dflash2_decode::{DraftedProposals, PagedVerifyLane, full_verify_ids};
use crate::models::qwen3_5::dflash2_turn::{DFlash2Target, DFlash2TurnState};
use crate::models::qwen3_5::gdn_blob::GdnStateBlobs;

pub(crate) struct Qwen35MoeDFlash2Stepper<'a> {
    inner: &'a mut Qwen35MoeInner,
    context: DFlash2ContextCache,
    next_position: i32,
    tap_layers: Vec<usize>,
    snapshot: Option<Vec<Qwen3_5LayerSnapshot>>,
    tape: Option<Vec<Option<super::gated_delta_net::GdnLayerTape>>>,
    tapped: Option<Vec<MxArray>>,
    verified_ids: Option<Vec<u32>>,
    /// Packed recurrent and convolution state for one fused accepted-prefix commit.
    gdn_blobs: Option<GdnStateBlobs>,
    verify_confidence: Option<MxArray>,
    verify_row_budget: Option<MxArray>,
    paged: PagedVerifyLane,
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
        {
            return Err(Error::from_reason(format!(
                "Qwen3.5 MoE DFlash2 {operation}: prior verify was not committed"
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
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE DFlash2 model disappeared"))?;
        let (path, _) = draft.propose(
            &self.inner.embedding,
            self.inner.lm_head.as_ref(),
            &mut self.context,
            anchor,
            crate::models::qwen3_5::dflash2::DFLASH2_DRAFT_PROPOSALS,
            0.0,
            false,
            None,
            &mut rand::rng(),
        )?;
        match path {
            crate::models::qwen3_5::dflash2::SelectorPath::Host(ids) => {
                Ok(ids.into_iter().map(|id| id as u32).collect())
            }
            crate::models::qwen3_5::dflash2::SelectorPath::Device(ids) => {
                DraftedProposals::Device(ids).host()
            }
        }
    }

    /// The verify on the pool (Splash `Engine::workEndOf` for a decode
    /// command books committed + 8); `budgeted` keeps the selector's row
    /// budget, clamped to the driver's `ids.len()` rows (Splash
    /// `decode_accept_dflash`: `min(remaining, verify_rows)`).
    fn paged_verify(&mut self, ids: &[u32], budgeted: bool) -> Result<DsparkVerifyOutput> {
        use crate::models::qwen3_5::dflash2::DFLASH2_VERIFY_ROWS;
        if ids.is_empty() {
            return Err(Error::from_reason(
                "Qwen3.5 MoE DFlash2 verify block must not be empty",
            ));
        }
        self.ensure_clean("verify")?;
        if budgeted {
            if let Some(budget) = self.verify_row_budget.take() {
                let rows = MxArray::from_int32(&[ids.len() as i32], &[1])?;
                self.verify_row_budget = Some(budget.minimum(&rows)?);
            }
        } else {
            self.verify_confidence = None;
            self.verify_row_budget = None;
        }
        let drafted = match self.paged.drafted.take() {
            Some(drafted) => drafted.host()?,
            None => self.draft_tail(ids[0])?,
        };
        let full = full_verify_ids(ids, &drafted)?;
        let (seq, window) = (self.paged.seq, self.paged.context_window);
        crate::models::qwen3_5::dflash2_turn::paged_step_reclaiming(&mut *self.inner, |inner| {
            inner
                .paged_adapter
                .as_mut()
                .ok_or_else(|| Error::from_reason("Qwen3.5 MoE DFlash2 paged adapter is absent"))?
                .reserve_rows(DFLASH2_VERIFY_ROWS as u32)
                .map(|_| ())
                .map_err(Error::from_reason)
        })?;
        let snapshot = snapshot_all_mtp(
            self.inner.caches.as_ref().ok_or_else(|| {
                Error::from_reason("Qwen3.5 MoE DFlash2 target caches are absent")
            })?,
            true,
        )?;
        let ticket =
            crate::engine::spec_paged::SpecPagedCache::open_core_write_cycle(self, seq, full.len())
                .map_err(Error::from_reason)?;
        let mut tape = Vec::new();
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
                        super::paged_forward::PagedVerifyExtras {
                            tap_layers: &self.tap_layers,
                            gdn_dst: self.gdn_blobs.as_ref().map(GdnStateBlobs::destinations),
                            row_budget: self.verify_row_budget.as_ref(),
                        },
                        window,
                    )
                }
                _ => Err(Error::from_reason(
                    "Qwen3.5 MoE DFlash2 paged verify lost its caches",
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
        Ok(DsparkVerifyOutput { logits })
    }

    /// Close the cycle the paged verify opened, keeping `keep` of its 8 rows
    /// (Splash `Runtime::Impl::finalizeDecode`).
    fn commit_paged_cycle(&mut self, keep: usize) -> Result<()> {
        let seq = self.paged.seq;
        let ticket = self.paged.open_cycle.take().ok_or_else(|| {
            Error::from_reason("Qwen3.5 MoE DFlash2 commit has no open verify cycle")
        })?;
        crate::engine::spec_paged::SpecPagedCache::commit_cycle(self, seq, ticket, keep)
            .map_err(Error::from_reason)
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
        // Every cycle drafts the whole block (Splash `Runtime::decodeAsync`:
        // one physical graph); the driver gets its cap.
        let adaptive = params.mtp_adaptive_depth && greedy_fast;
        let (path, draft_sparse_dists) = draft.propose(
            &self.inner.embedding,
            self.inner.lm_head.as_ref(),
            &mut self.context,
            anchor_id,
            crate::models::qwen3_5::dflash2::DFLASH2_DRAFT_PROPOSALS,
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

    /// Host verification includes teacher forcing and target-only steps: its
    /// tokens need not be the selector's path, so it is never budgeted.
    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        self.paged_verify(verify_ids, false)
    }

    /// The pool records host ids before its write, so the device ids are
    /// read back here.
    fn verify_device(&mut self, verify_ids: &MxArray) -> Result<DsparkVerifyOutput> {
        let ids = DraftedProposals::Device(verify_ids.clone()).host()?;
        self.paged_verify(&ids, true)
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
        let verified_ids = self.verified_ids.take().ok_or_else(|| {
            Error::from_reason("Qwen3.5 MoE DFlash2 commit has no token provenance")
        })?;
        if verified_ids.len() != total_written {
            return Err(Error::from_reason(format!(
                "Qwen3.5 MoE DFlash2 commit wrote {total_written} rows for {} token ids",
                verified_ids.len()
            )));
        }
        self.commit_paged_cycle(keep)?;
        // The verify writes 8 rows; only keeping all of them is the
        // window-final state its spare cell holds.
        let written = crate::models::qwen3_5::dflash2::DFLASH2_VERIFY_ROWS;
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
        let next_blobs = match &self.gdn_blobs {
            Some(blobs) => {
                let adopted = if keep == written {
                    blobs.adopt(&tape)?
                } else {
                    None
                };
                #[cfg(test)]
                if adopted.is_some() {
                    crate::models::qwen3_5::dflash2_decode::PAGED_ADOPTS
                        .with(|count| count.set(count.get() + 1));
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
                        "Qwen3.5 MoE DFlash2 fused GDN commit declined; per-layer replay for the rest of the turn"
                    );
                }
                replay_mtp_snapshot_to(
                    caches,
                    &snapshot,
                    &tape,
                    keep,
                    true,
                    "Qwen3.5 MoE DFlash2 commit",
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
                "Qwen3.5 MoE DFlash2 commit wrote {total_written} rows for {} supplied token ids",
                verified_ids.len()
            )));
        }
        match &self.verified_ids {
            Some(expected) if expected.as_slice() == verified_ids => {}
            Some(_) => {
                return Err(Error::from_reason(
                    "Qwen3.5 MoE DFlash2 supplied commit provenance disagrees with host verify input",
                ));
            }
            None => {
                return Err(Error::from_reason(
                    "Qwen3.5 MoE DFlash2 commit has no token provenance",
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
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE DFlash2 model disappeared"))?
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
        // and re-points the per-layer slots at them.
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
        let seq = self
            .paged_adapter
            .as_ref()
            .and_then(
                crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter::active_seq_id,
            )
            .ok_or_else(|| {
                Error::from_reason("Qwen3.5 MoE DFlash2 paged turn has no active request")
            })?;
        let context_window = self.pool_context_limit();
        let paged = PagedVerifyLane {
            seq,
            context_window,
            open_cycle: None,
            drafted: None,
            _decode_limit: crate::cache_limit::coordinator().push_decode_limit(
                crate::cache_limit::decode_cache_limit(
                    crate::models::qwen3_5::dflash2_decode::pool_verify_cycle_bytes(
                        &self.config.to_dense_config(),
                        u64::from(context_window),
                    ),
                ),
            ),
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
            gdn_blobs,
            verify_confidence: None,
            verify_row_budget: None,
            paged,
        })
    }
}

impl Drop for Qwen35MoeDFlash2Stepper<'_> {
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

impl Qwen35MoeDFlash2Stepper<'_> {
    fn dflash2_pool(
        &mut self,
        seq_id: u32,
    ) -> std::result::Result<
        &mut crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter,
        String,
    > {
        if self.paged.seq != seq_id {
            return Err(format!(
                "Qwen3.5 MoE DFlash2: sequence {seq_id} is not this turn's paged sequence"
            ));
        }
        self.inner
            .paged_adapter
            .as_mut()
            .filter(|adapter| adapter.active_seq_id() == Some(seq_id))
            .ok_or_else(|| {
                format!("Qwen3.5 MoE DFlash2: sequence {seq_id} is not the active request")
            })
    }
}

impl crate::engine::spec_paged::SpecPagedCache for Qwen35MoeDFlash2Stepper<'_> {
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
            "Qwen3.5 MoE DFlash2 sequence {seq_id}: the verify core writes its own rows"
        ))
    }

    fn record_verify(
        &mut self,
        seq_id: u32,
        _tokens: &[u32],
    ) -> std::result::Result<crate::engine::spec_paged::VerifyTicket, String> {
        Err(format!(
            "Qwen3.5 MoE DFlash2 sequence {seq_id}: the verify core writes its own rows"
        ))
    }

    fn rollback_rows(&mut self, seq_id: u32, rows: usize) -> std::result::Result<(), String> {
        let rows = u32::try_from(rows)
            .map_err(|_| format!("Qwen3.5 MoE DFlash2 rollback of {rows} rows does not fit u32"))?;
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
                "Qwen3.5 MoE DFlash2 settle_committed: committed frontier {committed_tokens} \
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

impl crate::engine::dspark_turn::PagedDsparkBackend for Qwen35MoeInner {
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
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE has no loaded DFlash2 companion"))?
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
                    "Qwen3.5 MoE DFlash2 cached prefix {cached} has no retained draft context"
                )));
            }
        };
        let (logits, context) = self.moe_paged_prefill(suffix_tokens, prefix, Some(context))?;
        let context = context.ok_or_else(|| {
            Error::from_reason("Qwen3.5 MoE DFlash2 prefill lost its draft context")
        })?;
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
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE has no loaded DFlash2 companion"))
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
            crate::models::qwen3_5::dflash2_decode::reusable_paged_dflash2_prefix(
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
        let prefix = self.prime_moe_prefix(tokens, cache_salt, Some(hit))?;
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

impl DFlash2Target for Qwen35MoeInner {
    fn dflash2(&self) -> Option<&crate::models::qwen3_5::dflash2::DFlash2Model> {
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
        std::borrow::Cow::Owned(self.config.to_dense_config())
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
mod tests {
    use super::*;
    use crate::engine::backend::ChatBackend;
    use crate::engine::types::ChatConfig;
    use crate::models::quantized_linear::LinearProj;
    use crate::nn::Linear;

    fn tiny_dflash_inner(seed: u64) -> Qwen35MoeInner {
        tiny_dflash_inner_with_config(seed, |_| {})
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
    fn moe_dflash_plan_and_params_use_companion_width() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let inner = tiny_dflash_inner(0xDFA5_3503);
        let plan = ChatBackend::execution_plan(&inner).speculative.unwrap();
        assert_eq!(plan.kind, crate::engine::plan::SpeculativeKind::DraftModel);
        assert!(plan.supports_paged_attention);
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
}

#[cfg(test)]
mod paged_dflash2_tests {
    use super::*;
    use crate::engine::backend::{PagedBackend, PagedPrefix};
    use crate::engine::dspark_turn::PagedDsparkBackend;
    use crate::models::quantized_linear::LinearProj;
    use crate::models::qwen3_5::dflash2::DFLASH2_DRAFT_PROPOSALS;
    use crate::models::qwen3_5::dflash2_decode::PAGED_ADOPTS;
    use crate::nn::Linear;
    use crate::stream::{Stream, StreamContext};
    use rand::SeedableRng;

    const VOCAB: u32 = 16;

    /// The seeded MoE paged fixture on an int8 pool, a sensitive random head
    /// and a draft that proposes the 7 tokens a cycle drafts.
    fn tiny_paged_int8_moe_dflash_inner(seed: u64) -> Qwen35MoeInner {
        let mut inner =
            super::super::model::scheduled_mtp::seeded_inner_with_config(seed, |config| {
                config.head_dim = 256;
                config.hidden_size = 256;
                config.kv_format = Some("int8".to_string());
                // The GDN head geometry whose verify takes the complete kernel
                // that writes the spare cell.
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
        let dense = inner.config.to_dense_config();
        let draft = crate::models::qwen3_5::dflash2::tiny_dflash2_model_with_block(
            &dense,
            DFLASH2_DRAFT_PROPOSALS,
        )
        .expect("tiny 7-proposal draft");
        draft.validate_target(&dense).expect("draft fits target");
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

    fn cursor(inner: &Qwen35MoeInner) -> u32 {
        inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .current_token_count()
    }

    /// Every prefill chunk launches the in-place attention with the prefill
    /// split rule (Splash `prefillSplits`, max 32) and the 8-row DFlash2
    /// verify with the verify rule (Splash `kv::verifyAttentionSplits`, max
    /// 128), one launch per full-attention layer. Mutation R1:
    /// `paged_verify_core` passes `PagedPass::Prefill` → red (no verify
    /// launch).
    #[test]
    fn verify_takes_the_verify_split_rule() -> Result<()> {
        let probe = |kind| unsafe { mlx_sys::mlx_paged_prefill_test_probe_count(kind) };
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8010);
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

    fn greedy() -> crate::engine::params::ChatParams {
        crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
            temperature: Some(0.0),
            ..Default::default()
        })
    }

    /// Greedy paged AR: `n` tokens and the cursor after them.
    fn paged_ar(inner: &mut Qwen35MoeInner, prompt: &[u32], n: usize) -> Result<(Vec<u32>, u32)> {
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

    /// Paged MoE DFlash2 cycles where the first `correct` proposals of each
    /// are the reference's tokens and the next one is wrong.
    fn paged_dflash2(
        inner: &mut Qwen35MoeInner,
        prompt: &[u32],
        reference: &[u32],
        correct: usize,
    ) -> Result<(Vec<u32>, u32, usize)> {
        PAGED_ADOPTS.with(|count| count.set(0));
        let n = reference.len();
        let prefix = inner.prime_dspark_prefix_state(prompt, true, 0)?;
        let suffix = prompt[prefix.effective_cached_prefix_len()..].to_vec();
        let logits =
            inner.paged_prefill_with_draft_state(&suffix, &prefix, Stream::generation())?;
        let mut generated = argmax_rows(&logits)?;
        let params = greedy();
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
            let rows = argmax_rows(&step.verify(&ids)?.logits)?;
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
        Ok((
            generated,
            cursor(inner),
            PAGED_ADOPTS.with(std::cell::Cell::get),
        ))
    }

    /// MoE speculation on the int8 pool, every keep: paged DFlash2 lands on
    /// the AR cursor, and a full accept adopts the spare GDN cell. Its tokens
    /// are not compared with paged AR: the verify reads through the in-place
    /// prefill attention, the AR decode through the decode kernel. Mutation
    /// MM1 (the commit keeps all 8 rows) turns it red.
    #[test]
    fn paged_moe_dflash2_lands_on_the_ar_cursor_for_every_keep() -> Result<()> {
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let prompt = prompt(40);
        let n = 64;
        let mut ar_inner = tiny_paged_int8_moe_dflash_inner(0x5_8101);
        let (reference, ar_cursor) = paged_ar(&mut ar_inner, &prompt, n)?;
        for correct in 0..=DFLASH2_DRAFT_PROPOSALS {
            let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8101);
            let (tokens, end, adopts) = paged_dflash2(&mut inner, &prompt, &reference, correct)?;
            assert!(tokens.len() >= n, "{correct}: {} tokens", tokens.len());
            assert_eq!(
                end, ar_cursor,
                "paged MoE DFlash2 with {correct} correct proposals left the AR cursor"
            );
            if correct == DFLASH2_DRAFT_PROPOSALS {
                assert!(
                    adopts > 0,
                    "a full accept must adopt the spare GDN cell the verify wrote"
                );
            }
        }
        Ok(())
    }

    /// Splash `decode_accept_dflash`: the verify's row budget is
    /// `min(budget, 1 + L)` for a driver handed L proposals. Mutation B2 (the
    /// clamp removed) turns it red.
    #[test]
    fn paged_moe_verify_clamps_the_row_budget() -> Result<()> {
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8102);
        let prompt = prompt(40);
        let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
        let logits = inner.paged_prefill_with_draft_state(&prompt, &prefix, stream)?;
        let anchor = argmax_rows(&logits)?[0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
        let mut proposal = step.propose(anchor, 2, &greedy(), &mut rng)?;
        assert_eq!(proposal.draft_len()?, 2, "the driver gets its cap");
        proposal.materialize_draft_ids()?;
        let mut ids = vec![anchor as i32];
        ids.extend(proposal.draft_ids.iter().copied());
        step.verify_row_budget = Some(MxArray::from_int32(&[5], &[1])?);
        let device = MxArray::from_int32(&ids, &[ids.len() as i64])?;
        let logits = step.verify_device(&device)?.logits;
        assert_eq!(logits.shape()?.as_ref(), &[1, 3, i64::from(VOCAB)]);
        logits.eval();
        let budget = step
            .device_verification_budget()
            .expect("a budgeted verify keeps its budget")
            .to_int32()?;
        assert_eq!(budget.as_ref(), &[3], "min(budget 5, 1 + 2 proposals)");
        let host: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        step.commit_with_provenance(1, host.len(), &host)?;
        step.finish()?;
        Ok(())
    }

    /// A poolless load refuses the companion.
    #[test]
    fn moe_dflash2_companion_needs_the_pool() {
        let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8103);
        inner
            .require_dflash2_pool()
            .expect("a pool admits the companion");
        inner.paged_adapter = None;
        let error = inner
            .require_dflash2_pool()
            .expect_err("no pool refuses the companion");
        assert!(
            error.reason.contains("block-paged KV pool"),
            "{}",
            error.reason
        );
    }

    /// The MoE turn parks its prompt's replay point once its blocks
    /// register, and the next DFlash2 turn on that prefix resumes there
    /// though the live window ends elsewhere (Splash `Cache::lookup`).
    /// Mutation MS1 (the prime skips the parked-state lookup) → red (hit 0).
    #[test]
    fn moe_parked_replay_point_resumes_the_next_session() -> Result<()> {
        let stream = Stream::generation();
        let _ctx = StreamContext::new(stream);
        let a1 = prompt(40);
        let mut ar_inner = tiny_paged_int8_moe_dflash_inner(0x5_8104);
        let (reference, _) = paged_ar(&mut ar_inner, &a1, 12)?;
        let mut inner = tiny_paged_int8_moe_dflash_inner(0x5_8104);
        paged_dflash2(&mut inner, &a1, &reference, 2)?;
        inner.finalize_paged_turn(true, 0);
        let parked: Vec<usize> = inner
            .dflash2_states
            .store
            .entries()
            .map(|entry| entry.depth())
            .collect();
        assert_eq!(parked, vec![32]);
        let mut a2 = a1[..32].to_vec();
        a2.extend((0..20u32).map(|i| (i * 5 + 2) % VOCAB));
        let prefix = inner.prime_dspark_prefix_state(&a2, true, 0)?;
        assert_eq!(prefix.effective_cached_prefix_len(), 32);
        let logits = inner.paged_prefill_with_draft_state(&a2[32..], &prefix, stream)?;
        logits.eval();
        Ok(())
    }
}
