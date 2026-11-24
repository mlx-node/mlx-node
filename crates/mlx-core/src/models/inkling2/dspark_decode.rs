//! Whole-turn DSpark speculative decoding for Inkling-2-Mini (`DsparkBackend`,
//! `PagedDsparkBackend`, `Inkling2DsparkStepper`) guarded by `DsparkBreakEvenPolicy`
//! and `ShortConvTape` rollback.

use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};
use crate::engine::backend::{
    DsparkBackend, DsparkProposal, DsparkStepper, DsparkTreeGreedyResult, DsparkVerifyOutput,
    PagedBackend,
};
use crate::engine::dspark_turn::PagedDsparkBackend;
use crate::engine::params::ChatParams;
use crate::engine::spec_paged::{SpecPagedCache, VerifyTicket};
use crate::models::qwen3_5::verify_tree::cuda::{TreeLookupSource, finalize_tree_greedy_walk};
use crate::stream::{Stream, StreamContext};
use crate::transformer::paged_kv_cache_adapter::{PagedRaggedRow, SeqId};

use super::dflash2::{
    Inkling2Dflash2Context, Inkling2DraftContext, Inkling2VerifyTreePolicy, is_strict_linear_chain,
    verify_tree_to_inkling2_rows,
};
use super::dspark::{AdaptiveDraftDepth, Inkling2DsparkContext};
use super::model::Inkling2Inner;
use super::paged_forward::Inkling2VerifyTreeInput;
use super::short_conv::Inkling2ShortConvTape;

/// Open verify-tree (or adaptive zero-sync 1D fast-path) state held between
/// [`Inkling2DsparkStepper::verify_tree_greedy`] and [`Inkling2DsparkStepper::commit_tree`].
struct OpenInkling2TreeVerify {
    ticket: VerifyTicket,
    tape: Inkling2ShortConvTape,
    taps: Vec<MxArray>,
    tree_tokens: Vec<u32>,
    anchor_pos: u32,
    tree_len: usize,
    used_placeholders: bool,
}

/// Per-turn state for [`Inkling2DsparkStepper`] (serves both DFlash2 and DSpark drafters).
pub(crate) struct Inkling2DsparkStepper<'a> {
    inner: &'a mut Inkling2Inner,
    seq_id: SeqId,
    ctx: Inkling2DraftContext,
    adaptive_depth: AdaptiveDraftDepth,
    tap_visits: Vec<usize>,
    tree_policy: Inkling2VerifyTreePolicy,
    ar_fallback: bool,
    open_ticket: Option<VerifyTicket>,
    open_tape: Option<Inkling2ShortConvTape>,
    open_taps: Option<Vec<MxArray>>,
    open_device_ids: Option<MxArray>,
    open_host_ids: Option<Vec<u32>>,
    open_tree: Option<OpenInkling2TreeVerify>,
    ar_probe_pending: bool,
    ar_probe_taps: Option<Vec<MxArray>>,
    ar_probe_token: Option<u32>,
}

impl Inkling2DsparkStepper<'_> {
    fn ensure_clean(&self, op: &str) -> Result<()> {
        if self.open_ticket.is_some()
            || self.open_tape.is_some()
            || self.open_taps.is_some()
            || self.open_device_ids.is_some()
            || self.open_host_ids.is_some()
            || self.open_tree.is_some()
            || self.ar_probe_pending
        {
            return Err(Error::from_reason(format!(
                "Inkling2 speculative {op}: prior verify was not committed"
            )));
        }
        Ok(())
    }

    fn verify_host_inner(&mut self, verify_ids: &[u32], greedy_argmax: bool) -> Result<MxArray> {
        self.ensure_clean("verify")?;
        if verify_ids.is_empty() {
            return Err(Error::from_reason(
                "Inkling2 DSpark verify_ids must not be empty",
            ));
        }

        // Fast AR fallback path (post break-even decision): bypass verify/tape machinery cleanly.
        if self.ar_fallback && verify_ids.len() == 1 {
            return if greedy_argmax {
                self.inner
                    .paged_decode_step_greedy(self.seq_id, verify_ids[0])
            } else {
                let logits_2d = self.inner.paged_decode_step(self.seq_id, verify_ids[0])?;
                let vocab = logits_2d.shape()?[1];
                logits_2d.reshape(&[1, 1, vocab])
            };
        }

        let first_pos = {
            let rt = self.inner.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2 DSpark verify: paged runtime missing")
            })?;
            rt.ensure_conv_state(self.seq_id, &self.inner.config)?;
            rt.coordinator
                .request_token_count_all(self.seq_id)
                .map_err(Error::from_reason)?
        };
        let ticket = self
            .inner
            .record_verify_exclusive(self.seq_id, verify_ids)
            .map_err(Error::from_reason)?;

        let row = PagedRaggedRow {
            seq_id: self.seq_id,
            first_logical_position: first_pos,
            query_len: verify_ids.len() as u32,
        };
        let rows = [(row, verify_ids.to_vec())];
        let tap_indices: &[usize] = if self.ar_fallback {
            &[]
        } else {
            &self.tap_visits
        };
        let forward_res = if greedy_argmax {
            self.inner
                .paged_verify_ragged_taped_greedy(&rows, tap_indices)
        } else {
            self.inner.paged_verify_ragged_taped(&rows, tap_indices)
        };
        let (out_arr, taps, mut tapes) = match forward_res {
            Ok(out) => out,
            Err(err) => {
                if let Some(rt) = self.inner.paged.as_mut() {
                    let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                }
                return Err(err);
            }
        };

        let result_arr = if greedy_argmax {
            out_arr
        } else {
            let total_rows = verify_ids.len() as i64;
            let vocab = out_arr.shape()?[1];
            out_arr.reshape(&[1, total_rows, vocab])?
        };
        let tape = tapes.pop().ok_or_else(|| {
            Error::from_reason("Inkling2 DSpark verify did not produce a ShortConvTape")
        })?;

        self.open_ticket = Some(ticket);
        self.open_tape = Some(tape);
        self.open_taps = if self.ar_fallback { None } else { Some(taps) };
        self.open_host_ids = Some(verify_ids.to_vec());
        Ok(result_arr)
    }

    fn verify_device_inner(
        &mut self,
        verify_ids: &MxArray,
        greedy_argmax: bool,
    ) -> Result<MxArray> {
        self.ensure_clean("verify_device")?;
        let len = verify_ids.size()? as usize;
        if len == 0 {
            return Err(Error::from_reason(
                "Inkling2 DSpark verify_device: verify_ids must not be empty",
            ));
        }
        if self.ar_fallback && len == 1 {
            let id = verify_ids.item_at_int32(0)? as u32;
            return self.verify_host_inner(&[id], greedy_argmax);
        }

        let placeholders = vec![u32::MAX; len];
        let first_pos = {
            let rt = self.inner.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2 DSpark verify_device: paged runtime missing")
            })?;
            rt.ensure_conv_state(self.seq_id, &self.inner.config)?;
            rt.coordinator
                .request_token_count_all(self.seq_id)
                .map_err(Error::from_reason)?
        };
        let ticket = self
            .inner
            .record_verify_exclusive(self.seq_id, &placeholders)
            .map_err(Error::from_reason)?;

        let row = PagedRaggedRow {
            seq_id: self.seq_id,
            first_logical_position: first_pos,
            query_len: len as u32,
        };
        let tap_indices: &[usize] = if self.ar_fallback {
            &[]
        } else {
            &self.tap_visits
        };
        let forward_res = self.inner.paged_verify_ragged_taped_device(
            &[row],
            verify_ids,
            tap_indices,
            greedy_argmax,
        );
        let (out_arr, taps, mut tapes) = match forward_res {
            Ok(out) => out,
            Err(err) => {
                if let Some(rt) = self.inner.paged.as_mut() {
                    let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                }
                return Err(err);
            }
        };

        let result_arr = if greedy_argmax {
            out_arr
        } else {
            let total_rows = len as i64;
            let vocab = out_arr.shape()?[1];
            out_arr.reshape(&[1, total_rows, vocab])?
        };
        let tape = tapes.pop().ok_or_else(|| {
            Error::from_reason("Inkling2 DSpark verify_device did not produce a ShortConvTape")
        })?;

        self.open_ticket = Some(ticket);
        self.open_tape = Some(tape);
        self.open_taps = if self.ar_fallback { None } else { Some(taps) };
        self.open_device_ids = Some(verify_ids.clone());
        Ok(result_arr)
    }

    fn commit_inner(
        &mut self,
        keep: usize,
        total_written: usize,
        verified_ids: Option<&[u32]>,
    ) -> Result<()> {
        if self.ar_fallback && self.open_ticket.is_none() {
            let _ = self.inner.settle_paged_step(self.seq_id)?;
            return Ok(());
        }
        let ticket = self.open_ticket.take().ok_or_else(|| {
            Error::from_reason("Inkling2 DSpark commit without open verify ticket")
        })?;
        let tape = self
            .open_tape
            .take()
            .ok_or_else(|| Error::from_reason("Inkling2 DSpark commit without ShortConvTape"))?;
        let taps = self.open_taps.take();
        let dev_ids = self.open_device_ids.take();
        let host_ids_saved = self.open_host_ids.take();

        let ticket_rows = ticket.rows();
        if keep == 0 || keep > total_written || total_written != ticket_rows {
            if let Some(rt) = self.inner.paged.as_mut() {
                let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                if let Some(conv) = rt.conv_states.get_mut(&self.seq_id) {
                    let _ = tape.commit_prefix(0, conv);
                }
            }
            return Err(Error::from_reason(format!(
                "Inkling2 speculative invalid commit keep={keep}, total_written={total_written}, ticket_rows={ticket_rows}"
            )));
        }

        // If `verify_device` booked placeholder KV entries, patch them with exact token IDs
        // BEFORE `commit_cycle` truncates the rejected tail.
        let fallback_host_ids;
        let patch_ids: Option<&[u32]> = if let Some(dev_arr) = dev_ids {
            if let Some(host_ids) = verified_ids {
                if host_ids.len() != total_written {
                    if let Some(rt) = self.inner.paged.as_mut() {
                        let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                        if let Some(conv) = rt.conv_states.get_mut(&self.seq_id) {
                            let _ = tape.commit_prefix(0, conv);
                        }
                    }
                    return Err(Error::from_reason(format!(
                        "Inkling2 speculative commit_with_provenance: verified_ids len {} != total_written {total_written}",
                        host_ids.len()
                    )));
                }
                Some(host_ids)
            } else {
                let raw = match dev_arr.to_int32() {
                    Ok(r) => r,
                    Err(err) => {
                        if let Some(rt) = self.inner.paged.as_mut() {
                            let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                            if let Some(conv) = rt.conv_states.get_mut(&self.seq_id) {
                                let _ = tape.commit_prefix(0, conv);
                            }
                        }
                        return Err(err);
                    }
                };
                fallback_host_ids = raw.as_ref().iter().map(|&id| id as u32).collect::<Vec<_>>();
                Some(&fallback_host_ids)
            }
        } else {
            None
        };

        let committed_tokens = {
            let rt = self.inner.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2 speculative commit: paged runtime missing")
            })?;
            if let Some(ids) = patch_ids {
                rt.coordinator
                    .patch_last_recorded_tokens_all(self.seq_id, ids)
                    .map_err(Error::from_reason)?;
            }
            // 1. Roll back rejected KV rows across both Global & SWA pools
            rt.coordinator
                .commit_cycle(self.seq_id, ticket, keep)
                .map_err(Error::from_reason)?;

            // 2. Roll back 48-visit ShortConv state to exact `keep`-token prefix via tape!
            let conv = rt.conv_states.get_mut(&self.seq_id).ok_or_else(|| {
                Error::from_reason("Inkling2 speculative commit: missing ShortConv state")
            })?;
            tape.commit_prefix(keep, conv)?;

            let committed = rt
                .coordinator
                .request_token_count_all(self.seq_id)
                .map_err(Error::from_reason)?;
            rt.coordinator
                .settle_committed(self.seq_id, u64::from(committed))
                .map_err(Error::from_reason)?;
            rt.record_block_boundary_conv_from_tape(self.seq_id, committed, keep, &tape)?;
            committed
        };

        // 3. Append ONLY the `keep` accepted tokens' tapped hiddens into the draft context!
        if !self.ar_fallback
            && let Some(taps) = taps
        {
            match &mut self.ctx {
                Inkling2DraftContext::Dspark(dctx) => {
                    if let Some(ref draft) = self.inner.dspark {
                        dctx.append_taps(draft, &taps, keep)?;
                        debug_assert_eq!(dctx.committed_len as u32, committed_tokens);
                    }
                }
                Inkling2DraftContext::Dflash2(dctx) => {
                    if let Some(ref draft) = self.inner.dflash2 {
                        let kept_ids: &[u32] = patch_ids
                            .or(verified_ids)
                            .or(host_ids_saved.as_deref())
                            .ok_or_else(|| {
                                Error::from_reason(
                                    "Inkling2 DFlash2 commit missing verified token ids",
                                )
                            })?;
                        if kept_ids.len() < keep {
                            return Err(Error::from_reason(format!(
                                "Inkling2 DFlash2 commit token provenance len {} < keep {keep}",
                                kept_ids.len()
                            )));
                        }
                        dctx.append_taps(draft, &taps, &kept_ids[..keep])?;
                        debug_assert_eq!(dctx.committed_len() as u32, committed_tokens);
                    }
                }
            }
        }
        let drafted = total_written.saturating_sub(1);
        if !self.ar_fallback && drafted > 0 {
            self.adaptive_depth.observe(drafted, keep.saturating_sub(1));
        }
        Ok(())
    }
}

impl DsparkStepper for Inkling2DsparkStepper<'_> {
    fn supports_adaptive_ar_fallback(&self) -> bool {
        true
    }

    fn enter_ar_fallback(&mut self) -> Result<()> {
        self.ensure_clean("enter_ar_fallback")?;
        self.ar_fallback = true;
        Ok(())
    }

    fn device_sampled_rows(&self) -> bool {
        true
    }

    fn materialize_adaptive_state(&self) -> Result<()> {
        self.ctx.eval_state()
    }

    fn reserve_cycle_lookahead(&mut self, rows: usize) -> Result<bool> {
        self.inner
            .reserve_lookahead_exclusive(self.seq_id, rows)
            .map_err(Error::from_reason)
    }

    fn propose(
        &mut self,
        anchor_id: u32,
        max_len: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        self.ensure_clean("propose")?;
        if self.ar_fallback || max_len == 0 {
            return Ok(DsparkProposal {
                draft_ids: Vec::new(),
                device_draft_ids: None,
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            });
        }
        let prop = match &mut self.ctx {
            Inkling2DraftContext::Dspark(dctx) => {
                let effective_len = if params.mtp_adaptive_depth {
                    self.adaptive_depth.current_depth(max_len)
                } else {
                    max_len
                };
                let draft =
                    self.inner.dspark.as_ref().ok_or_else(|| {
                        Error::from_reason("Inkling2 DSpark draft model disappeared")
                    })?;
                dctx.propose_block(
                    draft,
                    &self.inner.embed_tokens,
                    &self.inner.embed_norm,
                    &self.inner.lm_head,
                    anchor_id,
                    effective_len,
                    params,
                    rng,
                )?
            }
            Inkling2DraftContext::Dflash2(dctx) => {
                let draft = self.inner.dflash2.as_ref().ok_or_else(|| {
                    Error::from_reason("Inkling2 DFlash2 draft model disappeared")
                })?;
                let mut view = self.inner.target_view();
                dctx.propose_block(draft, &mut view, anchor_id, max_len, params, rng, false)?
            }
        };
        // Kick ONE async_eval on `device_draft_ids` right after `propose_block` returns
        // (before target `verify` builds its 48-visit graph) so the GPU executes
        // `commit (append_taps) + propose_block` in ONE continuous Metal CB!
        if let Some(ref dev_ids) = prop.device_draft_ids {
            MxArray::schedule_overlap(&[dev_ids]);
        }
        Ok(prop)
    }

    fn verify_ar_probe(&mut self, anchor_id: u32) -> Result<DsparkVerifyOutput> {
        self.ensure_clean("verify_ar_probe")?;
        let first_pos = {
            let rt = self.inner.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2 speculative AR probe: paged runtime missing")
            })?;
            rt.ensure_conv_state(self.seq_id, &self.inner.config)?;
            let pos = rt
                .coordinator
                .request_token_count_all(self.seq_id)
                .map_err(Error::from_reason)?;
            if let Some(snap) = rt.conv_states.get(&self.seq_id).cloned() {
                rt.prev_conv_snapshots.insert(self.seq_id, (pos, snap));
            }
            pos
        };
        self.inner
            .record_tokens_all_exclusive(self.seq_id, &[anchor_id])
            .map_err(Error::from_reason)?;
        let (logits_2d, taps) = self.inner.forward_paged_single_chunk_with_taps(
            self.seq_id,
            &[anchor_id],
            None,
            first_pos,
            true,
            &self.tap_visits,
        )?;
        let logits_2d = logits_2d.ok_or_else(|| {
            Error::from_reason("Inkling2 speculative AR probe produced no logits")
        })?;
        let vocab = logits_2d.shape()?[1];
        let logits = logits_2d.reshape(&[1, 1, vocab])?;
        if let Some(rt) = self.inner.paged.as_mut() {
            rt.record_block_boundary_conv(self.seq_id, first_pos.saturating_add(1));
        }
        self.ar_probe_pending = true;
        self.ar_probe_taps = Some(taps);
        self.ar_probe_token = Some(anchor_id);
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit_ar_probe(&mut self) -> Result<()> {
        if !self.ar_probe_pending {
            return Err(Error::from_reason(
                "Inkling2 speculative commit_ar_probe without open AR probe",
            ));
        }
        self.ar_probe_pending = false;
        let taps = self.ar_probe_taps.take();
        let probe_tok = self.ar_probe_token.take();
        let _ = self.inner.settle_paged_step(self.seq_id)?;
        if !self.ar_fallback
            && let Some(taps) = taps
        {
            match &mut self.ctx {
                Inkling2DraftContext::Dspark(dctx) => {
                    if let Some(ref draft) = self.inner.dspark {
                        dctx.append_taps(draft, &taps, 1)?;
                    }
                }
                Inkling2DraftContext::Dflash2(dctx) => {
                    if let Some(ref draft) = self.inner.dflash2 {
                        let tok = probe_tok.ok_or_else(|| {
                            Error::from_reason("Inkling2 DFlash2 AR probe missing anchor token")
                        })?;
                        dctx.append_taps(draft, &taps, &[tok])?;
                    }
                }
            }
        }
        Ok(())
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        let logits = self.verify_host_inner(verify_ids, false)?;
        Ok(DsparkVerifyOutput { logits })
    }

    fn verify_device(&mut self, verify_ids: &MxArray) -> Result<DsparkVerifyOutput> {
        let logits = self.verify_device_inner(verify_ids, false)?;
        Ok(DsparkVerifyOutput { logits })
    }

    fn supports_fused_greedy_verify(&self) -> bool {
        true
    }

    fn verify_greedy(&mut self, verify_ids: &[u32]) -> Result<MxArray> {
        self.verify_host_inner(verify_ids, true)
    }

    fn verify_device_greedy(&mut self, verify_ids: &MxArray) -> Result<MxArray> {
        self.verify_device_inner(verify_ids, true)
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        self.commit_inner(keep, total_written, None)
    }

    fn commit_with_provenance(
        &mut self,
        keep: usize,
        total_written: usize,
        verified_ids: &[u32],
    ) -> Result<()> {
        self.commit_inner(keep, total_written, Some(verified_ids))
    }

    fn finish(self) -> Result<()> {
        self.ensure_clean("finish")?;
        if !self.ar_fallback {
            self.inner
                .dspark_turn_contexts
                .insert(self.seq_id, self.ctx);
        } else {
            self.inner.dspark_turn_contexts.remove(&self.seq_id);
        }
        Ok(())
    }

    fn eval_boundary(&self, _token: &MxArray) {
        // Deliberately no-op: flushing 199 leaf arrays (192 conv states + 7 draft ctx arrays)
        // here costs ~0.6-0.8 ms of synchronous CPU graph traversal + submits a tiny Metal CB
        // right before `propose_block`. Instead, `propose()` kicks ONE `schedule_overlap` on
        // `device_draft_ids` so `commit + propose_block` execute in ONE continuous Metal CB
        // overlapped with CPU building the 48-visit target `verify` graph.
    }

    fn supports_tree_verify(&self) -> bool {
        matches!(self.ctx, Inkling2DraftContext::Dflash2(_))
            && self.tree_policy.tree_path_active()
            && !self.ar_fallback
    }

    fn observe_emitted_tokens(&mut self, tokens: &[u32]) {
        if self.tree_policy.tree_path_active()
            && self.tree_policy.lookup_enabled
            && !self.ar_fallback
            && !tokens.is_empty()
            && let Inkling2DraftContext::Dflash2(dctx) = &mut self.ctx
        {
            dctx.observe_tokens(tokens);
        }
    }

    fn verify_tree_greedy(
        &mut self,
        anchor: u32,
        max_nodes: usize,
        max_depth: usize,
    ) -> Result<DsparkTreeGreedyResult> {
        self.ensure_clean("verify_tree")?;
        if !self.supports_tree_verify() {
            return Err(Error::from_reason(
                "Inkling2 DFlash2 tree verification is not enabled on this stepper",
            ));
        }

        // O(1) adaptive routing (< 1 us pure CPU): in `Auto` mode, stay on the zero-sync
        // overlapped 1D DFlash2 chain (`M=8`, `p ~ 0.5 ms`) whenever SAM prompt lookup has
        // no strong suffix match, and switch seamlessly into `M=8..16` Verify Tree + SAM
        // the instant SAM finds a match >= `sam_trigger_len` with `remaining >= 4`.
        let (use_tree_cycle, peek_match_len) = {
            let Inkling2DraftContext::Dflash2(dctx) = &self.ctx else {
                return Err(Error::from_reason(
                    "Inkling2 verify_tree_greedy requires DFlash2 draft context",
                ));
            };
            let match_len = dctx.peek_lookup_match_len();
            (
                self.tree_policy.should_use_tree_cycle(match_len, max_depth),
                match_len,
            )
        };

        if !use_tree_cycle {
            // Fast path: zero-sync overlapped 1D DFlash2 chain (`M = 1 + cap <= 8`).
            let draft_slots = self
                .inner
                .dflash2
                .as_ref()
                .ok_or_else(|| Error::from_reason("Inkling2 DFlash2 draft model disappeared"))?
                .config
                .block_size;
            let cap = draft_slots.min(max_depth).min(max_nodes.saturating_sub(1));
            let verify_len = 1 + cap;
            let mut prop_opt = if cap > 0 {
                let Inkling2DraftContext::Dflash2(dctx) = &mut self.ctx else {
                    return Err(Error::from_reason(
                        "Inkling2 verify_tree_greedy requires DFlash2 draft context",
                    ));
                };
                let draft = self.inner.dflash2.as_ref().ok_or_else(|| {
                    Error::from_reason("Inkling2 DFlash2 draft model disappeared")
                })?;
                let mut view = self.inner.target_view();
                Some(dctx.propose_greedy_chain(draft, &mut view, anchor, cap, true, false)?)
            } else {
                None
            };
            let device_verify_ids = if let Some(ref prop) = prop_opt
                && let Some(ref dev_drafts) = prop.device_draft_ids
            {
                let anchor_arr = MxArray::from_int32(&[anchor as i32], &[1])?;
                let v_ids = MxArray::concatenate(&anchor_arr, dev_drafts, 0)?;
                // Overlap `prev_commit (append_taps) + 5-layer NAX draft + 7-row lm_head`
                // on Metal with CPU building the 48-visit target verify graph!
                MxArray::schedule_overlap(&[&v_ids]);
                Some(v_ids)
            } else {
                None
            };

            let host_verify_ids = if device_verify_ids.is_none() {
                let mut ids = Vec::with_capacity(verify_len);
                ids.push(anchor);
                if let Some(ref mut prop) = prop_opt {
                    prop.materialize_draft_ids()?;
                    ids.extend(prop.draft_ids.iter().map(|&id| id as u32));
                }
                Some(ids)
            } else {
                None
            };

            let (anchor_pos, rec_ids, used_placeholders) = {
                let rt = self.inner.paged.as_mut().ok_or_else(|| {
                    Error::from_reason("Inkling2 DFlash2 hybrid 1D verify: paged runtime missing")
                })?;
                rt.ensure_conv_state(self.seq_id, &self.inner.config)?;
                let pos = rt
                    .coordinator
                    .request_token_count_all(self.seq_id)
                    .map_err(Error::from_reason)?;
                let (rec_ids, placeholders) = if let Some(ref host_ids) = host_verify_ids {
                    (host_ids.clone(), false)
                } else {
                    (vec![u32::MAX; verify_len], true)
                };
                (pos, rec_ids, placeholders)
            };
            let ticket = self
                .inner
                .record_verify_exclusive(self.seq_id, &rec_ids)
                .map_err(Error::from_reason)?;

            let row = PagedRaggedRow {
                seq_id: self.seq_id,
                first_logical_position: anchor_pos,
                query_len: verify_len as u32,
            };
            let forward_res = if let Some(ref dev_ids) = device_verify_ids {
                self.inner
                    .paged_verify_ragged_taped_device(&[row], dev_ids, &self.tap_visits, true)
            } else if let Some(ref host_ids) = host_verify_ids {
                self.inner
                    .paged_verify_ragged_taped_greedy(&[(row, host_ids.clone())], &self.tap_visits)
            } else {
                unreachable!("either device_verify_ids or host_verify_ids is populated");
            };
            let (argmax, taps, mut tapes) = match forward_res {
                Ok(out) => out,
                Err(err) => {
                    if let Some(rt) = self.inner.paged.as_mut() {
                        let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                    }
                    return Err(err);
                }
            };
            let tape = match tapes.pop() {
                Some(t) => t,
                None => {
                    if let Some(rt) = self.inner.paged.as_mut() {
                        let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                    }
                    return Err(Error::from_reason(
                        "Inkling2 DFlash2 hybrid 1D verify did not produce a ShortConvTape",
                    ));
                }
            };

            // Single GPU->CPU sync for BOTH argmax and device draft IDs!
            let eval_res: Result<(Vec<u32>, Vec<u32>)> = (|| {
                if let Some(ref dev_ids) = device_verify_ids {
                    MxArray::eval_arrays_with_context(
                        &[&argmax, dev_ids],
                        "inkling2_dflash2_hybrid_1d_accept",
                    )?;
                    let am = match argmax.dtype()? {
                        DType::Uint32 => argmax.to_uint32()?.to_vec(),
                        _ => argmax.to_int32()?.iter().map(|&v| v as u32).collect(),
                    };
                    let ids = dev_ids.to_int32()?.iter().map(|&v| v as u32).collect();
                    Ok((am, ids))
                } else {
                    MxArray::eval_arrays_with_context(
                        &[&argmax],
                        "inkling2_dflash2_hybrid_1d_accept",
                    )?;
                    let am = match argmax.dtype()? {
                        DType::Uint32 => argmax.to_uint32()?.to_vec(),
                        _ => argmax.to_int32()?.iter().map(|&v| v as u32).collect(),
                    };
                    let ids = host_verify_ids.unwrap_or_default();
                    Ok((am, ids))
                }
            })();
            let (target_argmax, verified_ids) = match eval_res {
                Ok((am, ids)) if am.len() == verify_len && ids.len() == verify_len => (am, ids),
                Ok((am, ids)) => {
                    if let Some(rt) = self.inner.paged.as_mut() {
                        let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                        if let Some(conv) = rt.conv_states.get_mut(&self.seq_id) {
                            let _ = tape.commit_prefix(0, conv);
                        }
                    }
                    return Err(Error::from_reason(format!(
                        "Inkling2 DFlash2 hybrid 1D length mismatch: argmax={} ids={} verify_len={verify_len}",
                        am.len(),
                        ids.len()
                    )));
                }
                Err(err) => {
                    if let Some(rt) = self.inner.paged.as_mut() {
                        let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                        if let Some(conv) = rt.conv_states.get_mut(&self.seq_id) {
                            let _ = tape.commit_prefix(0, conv);
                        }
                    }
                    return Err(err);
                }
            };

            let mut k = 0usize;
            while k < cap && target_argmax[k] == verified_ids[k + 1] {
                k += 1;
            }
            let mut emitted_tokens = Vec::with_capacity(k + 1);
            emitted_tokens.extend_from_slice(&verified_ids[1..=k]);
            emitted_tokens.push(target_argmax[k]);
            let accepted_nodes: Vec<usize> = (0..=k).collect();

            if self.tree_policy.log_stats {
                eprintln!(
                    "[inkling2-tree] mode=1d_fast T={verify_len} match={peek_match_len} acc={k}/{cap}"
                );
            }

            self.open_tree = Some(OpenInkling2TreeVerify {
                ticket,
                tape,
                taps,
                tree_tokens: verified_ids,
                anchor_pos,
                tree_len: verify_len,
                used_placeholders,
            });

            return Ok(DsparkTreeGreedyResult {
                emitted_tokens,
                accepted_nodes,
                is_contiguous_prefix: true,
                draft_count: cap,
                tree_nodes: verify_len,
                accepted_draft_count: k,
            });
        }

        // 1. Propose transition lattice from compiled 5-layer Metal NAX DFlash2 draft.
        let lattice = {
            let Inkling2DraftContext::Dflash2(dctx) = &mut self.ctx else {
                return Err(Error::from_reason(
                    "Inkling2 verify_tree_greedy requires DFlash2 draft context",
                ));
            };
            let draft =
                self.inner.dflash2.as_ref().ok_or_else(|| {
                    Error::from_reason("Inkling2 DFlash2 draft model disappeared")
                })?;
            let draft_slots = draft.config.block_size;
            let mut view = self.inner.target_view();
            let (candidates, scores) =
                dctx.propose_tree_lattice(draft, &mut view, anchor, draft_slots)?;
            draft.lattice_from_arrays(&candidates, &scores, draft_slots)?
        };

        // 2. Query suffix-automaton prompt lookup and build cost-aware verify tree (M <= 16).
        let (lookup, mut tree) = {
            let Inkling2DraftContext::Dflash2(dctx) = &self.ctx else {
                return Err(Error::from_reason(
                    "Inkling2 verify_tree_greedy requires DFlash2 draft context",
                ));
            };
            let lookup = dctx.prepare_lookup(self.tree_policy.lookup_enabled, max_depth);
            let tree = self.tree_policy.build_tree(
                anchor,
                &lattice,
                &lookup,
                max_nodes,
                &dctx.verify_budget,
                &dctx.lookup_acc,
            );
            (lookup, tree)
        };

        // 3. Ensure KV lookahead can accommodate `tree.len()` rows (reclaiming idle owners on
        //    the exclusive lane before shrinking gracefully if the paged pool is near capacity on
        //    `M > 8` trees).
        let mut reserved = self
            .inner
            .reserve_lookahead_exclusive(self.seq_id, tree.len())
            .map_err(Error::from_reason)?;
        if !reserved && tree.len() > 1 {
            for fallback_cap in [8usize, 4, 1] {
                if fallback_cap >= tree.len() {
                    continue;
                }
                tree = {
                    let Inkling2DraftContext::Dflash2(dctx) = &self.ctx else {
                        return Err(Error::from_reason(
                            "Inkling2 verify_tree_greedy requires DFlash2 draft context",
                        ));
                    };
                    self.tree_policy.build_tree(
                        anchor,
                        &lattice,
                        &lookup,
                        max_nodes.min(fallback_cap),
                        &dctx.verify_budget,
                        &dctx.lookup_acc,
                    )
                };
                reserved = self
                    .inner
                    .reserve_lookahead_exclusive(self.seq_id, tree.len())
                    .map_err(Error::from_reason)?;
                if reserved {
                    break;
                }
            }
        }
        if !reserved {
            return Err(Error::from_reason(format!(
                "Inkling2 DFlash2 verify_tree: paged KV pool cannot reserve {} rows",
                tree.len()
            )));
        }

        // 4. Record verify rows and run 48-visit target tree verification.
        let tree_len = tree.len();
        let anchor_pos = {
            let rt = self.inner.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2 DFlash2 verify_tree: paged runtime missing")
            })?;
            rt.ensure_conv_state(self.seq_id, &self.inner.config)?;
            rt.coordinator
                .request_token_count_all(self.seq_id)
                .map_err(Error::from_reason)?
        };
        let ticket = self
            .inner
            .record_verify_exclusive(self.seq_id, &tree.tokens)
            .map_err(Error::from_reason)?;

        let is_chain = is_strict_linear_chain(&tree);
        let encoded_tree = if is_chain {
            None
        } else {
            match verify_tree_to_inkling2_rows(&tree) {
                Ok(pair) => Some(pair),
                Err(err) => {
                    if let Some(rt) = self.inner.paged.as_mut() {
                        let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                    }
                    return Err(err);
                }
            }
        };
        let tree_input = encoded_tree
            .as_ref()
            .map(|(tree_rows, depths)| Inkling2VerifyTreeInput { tree_rows, depths });

        let row = PagedRaggedRow {
            seq_id: self.seq_id,
            first_logical_position: anchor_pos,
            query_len: tree_len as u32,
        };
        let forward_res = self.inner.paged_verify_ragged_taped_tree(
            &[(row, tree.tokens.clone())],
            tree_input,
            &self.tap_visits,
            true,
        );
        let (argmax, taps, mut tapes) = match forward_res {
            Ok(out) => out,
            Err(err) => {
                if let Some(rt) = self.inner.paged.as_mut() {
                    let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                }
                return Err(err);
            }
        };
        let tape = match tapes.pop() {
            Some(t) => t,
            None => {
                if let Some(rt) = self.inner.paged.as_mut() {
                    let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                }
                return Err(Error::from_reason(
                    "Inkling2 DFlash2 verify_tree did not produce a ShortConvTape",
                ));
            }
        };

        // 5. Evaluate greedy argmax (`[T]` Int32 from `inkling2_lm_head_softcap_argmax`) and walk tree.
        let target_argmax: Vec<u32> = match (|| -> Result<Vec<u32>> {
            MxArray::eval_arrays_with_context(&[&argmax], "inkling2_dflash2_tree_greedy_accept")?;
            match argmax.dtype()? {
                DType::Uint32 => Ok(argmax.to_uint32()?.to_vec()),
                _ => Ok(argmax.to_int32()?.iter().map(|&v| v as u32).collect()),
            }
        })() {
            Ok(ids) if ids.len() == tree_len => ids,
            Ok(ids) => {
                if let Some(rt) = self.inner.paged.as_mut() {
                    let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                    if let Some(conv) = rt.conv_states.get_mut(&self.seq_id) {
                        let _ = tape.commit_prefix(0, conv);
                    }
                }
                return Err(Error::from_reason(format!(
                    "Inkling2 DFlash2 verify_tree argmax length {} != tree_len {tree_len}",
                    ids.len()
                )));
            }
            Err(err) => {
                if let Some(rt) = self.inner.paged.as_mut() {
                    let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                    if let Some(conv) = rt.conv_states.get_mut(&self.seq_id) {
                        let _ = tape.commit_prefix(0, conv);
                    }
                }
                return Err(err);
            }
        };

        let (walk, lookup_accepted) = {
            let Inkling2DraftContext::Dflash2(dctx) = &mut self.ctx else {
                if let Some(rt) = self.inner.paged.as_mut() {
                    let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                    if let Some(conv) = rt.conv_states.get_mut(&self.seq_id) {
                        let _ = tape.commit_prefix(0, conv);
                    }
                }
                return Err(Error::from_reason(
                    "Inkling2 verify_tree_greedy lost DFlash2 draft context",
                ));
            };
            finalize_tree_greedy_walk(
                &tree,
                &target_argmax,
                &lookup,
                TreeLookupSource::Lattice(&lattice),
                &mut dctx.verify_budget,
                &mut dctx.lookup_acc,
            )
        };

        let draft_count = tree.depth.iter().copied().max().unwrap_or(0).max(0) as usize;
        let accepted_draft_count = walk.accepted_nodes.len().saturating_sub(1);
        if self.tree_policy.log_stats {
            eprintln!(
                "[inkling2-tree] T={tree_len} chain={is_chain} depth={draft_count} match={} acc={accepted_draft_count} lookup_acc={lookup_accepted} contig={}",
                lookup.match_len, walk.is_contiguous_prefix,
            );
        }

        self.open_tree = Some(OpenInkling2TreeVerify {
            ticket,
            tape,
            taps,
            tree_tokens: tree.tokens,
            anchor_pos,
            tree_len,
            used_placeholders: false,
        });

        Ok(DsparkTreeGreedyResult {
            emitted_tokens: walk.emitted_tokens,
            accepted_nodes: walk.accepted_nodes,
            is_contiguous_prefix: walk.is_contiguous_prefix,
            draft_count,
            tree_nodes: tree_len,
            accepted_draft_count,
        })
    }

    fn commit_tree(&mut self, accepted_nodes: &[usize], is_contiguous_prefix: bool) -> Result<()> {
        let open = self.open_tree.take().ok_or_else(|| {
            Error::from_reason("Inkling2 DFlash2 commit_tree without open verify tree")
        })?;
        let keep = accepted_nodes.len();
        let valid_path = keep >= 1
            && keep <= open.tree_len
            && accepted_nodes[0] == 0
            && accepted_nodes.iter().all(|&row| row < open.tree_len)
            && accepted_nodes.windows(2).all(|w| w[0] < w[1]);
        if !valid_path {
            if let Some(rt) = self.inner.paged.as_mut() {
                let _ = rt.coordinator.commit_cycle(self.seq_id, open.ticket, 0);
                if let Some(conv) = rt.conv_states.get_mut(&self.seq_id) {
                    let _ = open.tape.commit_prefix(0, conv);
                }
            }
            return Err(Error::from_reason(format!(
                "Inkling2 DFlash2 invalid tree commit accepted_nodes={accepted_nodes:?}, tree_len={}",
                open.tree_len
            )));
        }

        let contiguous = is_contiguous_prefix
            && accepted_nodes
                .iter()
                .enumerate()
                .all(|(idx, &row)| row == idx);

        let committed_tokens = {
            let rt = self.inner.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2 DFlash2 commit_tree: paged runtime missing")
            })?;
            // 0. If the zero-sync 1D fast path booked placeholder KV entries, patch real
            //    verified token IDs BEFORE compaction / commit_cycle truncation.
            if open.used_placeholders
                && let Err(err) = rt
                    .coordinator
                    .patch_last_recorded_tokens_all(self.seq_id, &open.tree_tokens)
            {
                let _ = rt.coordinator.commit_cycle(self.seq_id, open.ticket, 0);
                if let Some(conv) = rt.conv_states.get_mut(&self.seq_id) {
                    let _ = open.tape.commit_prefix(0, conv);
                }
                return Err(Error::from_reason(err));
            }
            // 1. Compact non-contiguous accepted KV slots & recorded tokens BEFORE commit_cycle truncates.
            if !contiguous {
                if let Err(err) =
                    rt.coordinator
                        .compact_kv_path_all(self.seq_id, open.anchor_pos, accepted_nodes)
                {
                    let _ = rt.coordinator.commit_cycle(self.seq_id, open.ticket, 0);
                    if let Some(conv) = rt.conv_states.get_mut(&self.seq_id) {
                        let _ = open.tape.commit_prefix(0, conv);
                    }
                    return Err(Error::from_reason(err));
                }
                if let Err(err) = rt.coordinator.compact_recorded_tree_tokens_all(
                    self.seq_id,
                    open.tree_len,
                    accepted_nodes,
                ) {
                    let _ = rt.coordinator.commit_cycle(self.seq_id, open.ticket, 0);
                    if let Some(conv) = rt.conv_states.get_mut(&self.seq_id) {
                        let _ = open.tape.commit_prefix(0, conv);
                    }
                    return Err(Error::from_reason(err));
                }
            }
            // 2. Truncate rejected KV tail across both Global & SWA pools.
            rt.coordinator
                .commit_cycle(self.seq_id, open.ticket, keep)
                .map_err(Error::from_reason)?;

            // 3. Commit 48-visit ShortConv state along the exact accepted tree path!
            let conv = rt.conv_states.get_mut(&self.seq_id).ok_or_else(|| {
                Error::from_reason("Inkling2 DFlash2 commit_tree: missing ShortConv state")
            })?;
            if contiguous {
                open.tape.commit_prefix(keep, conv)?;
            } else {
                open.tape.commit_path(accepted_nodes, conv)?;
            }

            // 4. Prune SWA frontier and capture block-boundary ShortConv snapshot if crossed.
            let committed = rt
                .coordinator
                .request_token_count_all(self.seq_id)
                .map_err(Error::from_reason)?;
            rt.coordinator
                .settle_committed(self.seq_id, u64::from(committed))
                .map_err(Error::from_reason)?;
            if contiguous {
                rt.record_block_boundary_conv_from_tape(self.seq_id, committed, keep, &open.tape)?;
            } else {
                rt.record_block_boundary_conv_from_tape_path(
                    self.seq_id,
                    committed,
                    accepted_nodes,
                    &open.tape,
                )?;
            }
            committed
        };

        // 5. Gather and fuse ONLY the accepted path's target taps into the DFlash2 KV ring.
        if !self.ar_fallback
            && let Inkling2DraftContext::Dflash2(dctx) = &mut self.ctx
            && let Some(ref draft) = self.inner.dflash2
        {
            let kept_tokens: Vec<u32> = if contiguous {
                open.tree_tokens[..keep].to_vec()
            } else {
                accepted_nodes
                    .iter()
                    .map(|&row| open.tree_tokens[row])
                    .collect()
            };
            dctx.append_taps_path(draft, &open.taps, accepted_nodes, &kept_tokens)?;
            debug_assert_eq!(dctx.committed_len() as u32, committed_tokens);
        }
        Ok(())
    }
}

impl DsparkBackend for Inkling2Inner {
    type DsparkDecode<'a>
        = Inkling2DsparkStepper<'a>
    where
        Self: 'a;

    fn begin_dspark_decode(&mut self, block_size: usize) -> Result<Self::DsparkDecode<'_>> {
        let expected = self
            .active_draft_block_size()
            .ok_or_else(|| Error::from_reason("Inkling2 has no loaded speculative draft model"))?;
        if block_size != expected {
            return Err(Error::from_reason(format!(
                "Inkling2 speculative block_size {block_size} != draft checkpoint {expected}"
            )));
        }
        let tap_visits = self
            .active_draft_tap_visits()
            .ok_or_else(|| Error::from_reason("Inkling2 has no loaded speculative draft model"))?
            .to_vec();
        let seq_id = self
            .paged
            .as_ref()
            .ok_or_else(|| {
                Error::from_reason("Inkling2 speculative decode requires paged runtime")
            })?
            .active_seq;
        let mut ctx = self.dspark_turn_contexts.remove(&seq_id).ok_or_else(|| {
            Error::from_reason(format!(
                "Inkling2 speculative turn context missing for seq {seq_id} (prefill did not seed it)"
            ))
        })?;
        if let Inkling2DraftContext::Dflash2(ref mut dctx) = ctx {
            // Reset SAM index so `run_dspark_turn`'s `observe_emitted_tokens(hist)`
            // populates the full conversation history without duplicating prior turns.
            dctx.reset_lookup_for_turn();
        }
        Ok(Inkling2DsparkStepper {
            inner: self,
            seq_id,
            ctx,
            adaptive_depth: AdaptiveDraftDepth::new(expected),
            tap_visits,
            tree_policy: Inkling2VerifyTreePolicy::from_env(),
            ar_fallback: false,
            open_ticket: None,
            open_tape: None,
            open_taps: None,
            open_device_ids: None,
            open_host_ids: None,
            open_tree: None,
            ar_probe_pending: false,
            ar_probe_taps: None,
            ar_probe_token: None,
        })
    }
}

impl Inkling2Inner {
    fn new_turn_draft_context(&self) -> Result<Inkling2DraftContext> {
        if let Some(ref draft) = self.dflash2 {
            Ok(Inkling2DraftContext::Dflash2(Box::new(
                Inkling2Dflash2Context::new(&draft.config),
            )))
        } else if let Some(ref draft) = self.dspark {
            Ok(Inkling2DraftContext::Dspark(Inkling2DsparkContext::new(
                draft.config.num_hidden_layers,
            )))
        } else {
            Err(Error::from_reason(
                "Inkling2 has no loaded speculative draft model",
            ))
        }
    }
}

impl PagedDsparkBackend for Inkling2Inner {
    fn dspark_block_size(&self) -> Result<usize> {
        self.active_draft_block_size()
            .ok_or_else(|| Error::from_reason("Inkling2 has no loaded speculative draft model"))
    }

    fn paged_spec_seq_id(&self) -> u32 {
        self.paged.as_ref().map_or(0, |rt| rt.active_seq)
    }

    fn prime_dspark_prefix_state(
        &mut self,
        tokens: &[u32],
        reuse_cache: bool,
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        let seq_id = self
            .paged
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?
            .active_seq;

        // Only reuse a cached prefix on the whole-turn speculative path when `seq_id` already holds
        // a matching live draft context at the exact prefix boundary; otherwise start fresh from
        // position 0 so the draft context receives every token's tapped hiddens.
        let prefix = self.prime_prefix_state(tokens, reuse_cache, 0, &[], cache_salt)?;
        let boundary = prefix.effective_cached_prefix_len;
        let has_matching_draft_ctx = boundary > 0
            && prefix.live_continuation
            && self
                .dspark_turn_contexts
                .get(&seq_id)
                .is_some_and(|ctx| ctx.committed_len() as usize == boundary);
        if boundary == 0 || has_matching_draft_ctx {
            if boundary == 0 {
                let ctx = self.new_turn_draft_context()?;
                self.dspark_turn_contexts.insert(seq_id, ctx);
            }
            return Ok(prefix);
        }

        // Prefix hit had no matching draft context (e.g. cold SSD restore): restart fresh so
        // speculative prefill seeds the draft context cleanly from position 0.
        self.dspark_turn_contexts.remove(&seq_id);
        let fresh = self.prime_prefix_state(tokens, false, 0, &[], cache_salt)?;
        let ctx = self.new_turn_draft_context()?;
        self.dspark_turn_contexts.insert(seq_id, ctx);
        Ok(fresh)
    }

    fn paged_prefill_with_draft_state(
        &mut self,
        suffix_tokens: &[u32],
        _prefix: &Self::PrefixState,
        stream: Stream,
    ) -> Result<MxArray> {
        let _ctx = StreamContext::new(stream);
        let seq_id = self
            .paged
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 paged runtime missing"))?
            .active_seq;
        if self.dflash2.is_some() {
            let mut draft_ctx = match self.dspark_turn_contexts.remove(&seq_id) {
                Some(Inkling2DraftContext::Dflash2(ctx)) => ctx,
                _ => {
                    let draft = self.dflash2.as_ref().ok_or_else(|| {
                        Error::from_reason("Inkling2 has no loaded DFlash2 draft model")
                    })?;
                    Box::new(Inkling2Dflash2Context::new(&draft.config))
                }
            };
            let logits_2d =
                self.paged_prefill_dflash2_with_taps(seq_id, suffix_tokens, &mut draft_ctx)?;
            draft_ctx.eval_state()?;
            self.dspark_turn_contexts
                .insert(seq_id, Inkling2DraftContext::Dflash2(draft_ctx));
            return logits_2d.squeeze(Some(&[0]));
        }

        let (tap_visits, n_draft_layers) = {
            let draft = self
                .dspark
                .as_ref()
                .ok_or_else(|| Error::from_reason("Inkling2 has no loaded DSpark draft model"))?;
            (
                draft.config.target_layer_ids.clone(),
                draft.config.num_hidden_layers,
            )
        };
        let mut draft_ctx = match self.dspark_turn_contexts.remove(&seq_id) {
            Some(Inkling2DraftContext::Dspark(ctx)) => ctx,
            _ => Inkling2DsparkContext::new(n_draft_layers),
        };

        let (logits_2d, _) = self.paged_prefill_sequence_with_taps(
            seq_id,
            suffix_tokens,
            None,
            &tap_visits,
            |draft, taps, chunk_len| draft_ctx.append_taps(draft, taps, chunk_len),
        )?;
        draft_ctx.eval_state()?;
        self.dspark_turn_contexts
            .insert(seq_id, Inkling2DraftContext::Dspark(draft_ctx));
        logits_2d.squeeze(Some(&[0]))
    }
}
