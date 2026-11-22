//! Whole-turn DSpark speculative decoding for Inkling-2-Mini (`DsparkBackend`,
//! `PagedDsparkBackend`, `Inkling2DsparkStepper`) guarded by `DsparkBreakEvenPolicy`
//! and `ShortConvTape` rollback.

use napi::bindgen_prelude::{Error, Result};

use crate::array::MxArray;
use crate::engine::backend::{
    DsparkBackend, DsparkProposal, DsparkStepper, DsparkVerifyOutput, PagedBackend,
};
use crate::engine::dspark_turn::PagedDsparkBackend;
use crate::engine::params::ChatParams;
use crate::engine::spec_paged::{SpecPagedCache, VerifyTicket};
use crate::stream::{Stream, StreamContext};
use crate::transformer::paged_kv_cache_adapter::{PagedRaggedRow, SeqId};

use super::dspark::{Inkling2DsparkContext, Inkling2DsparkModel};
use super::model::Inkling2Inner;
use super::short_conv::Inkling2ShortConvTape;

/// Per-turn state for [`Inkling2DsparkStepper`].
pub(crate) struct Inkling2DsparkStepper<'a> {
    inner: &'a mut Inkling2Inner,
    seq_id: SeqId,
    ctx: Inkling2DsparkContext,
    tap_visits: Vec<usize>,
    ar_fallback: bool,
    open_ticket: Option<VerifyTicket>,
    open_tape: Option<Inkling2ShortConvTape>,
    open_taps: Option<Vec<MxArray>>,
    ar_probe_pending: bool,
    ar_probe_taps: Option<Vec<MxArray>>,
}

impl Inkling2DsparkStepper<'_> {
    fn ensure_clean(&self, op: &str) -> Result<()> {
        if self.open_ticket.is_some()
            || self.open_tape.is_some()
            || self.open_taps.is_some()
            || self.ar_probe_pending
        {
            return Err(Error::from_reason(format!(
                "Inkling2 DSpark {op}: prior verify was not committed"
            )));
        }
        Ok(())
    }

    fn require_draft(&self) -> Result<&Inkling2DsparkModel> {
        self.inner
            .dspark
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 DSpark draft model disappeared"))
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
        let rt =
            self.inner.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2 DSpark reserve: paged runtime missing")
            })?;
        rt.coordinator
            .reserve_lookahead(self.seq_id, rows)
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
        let draft = self.require_draft()?;
        self.ctx.propose_block(
            draft,
            &self.inner.embed_tokens,
            &self.inner.embed_norm,
            &self.inner.lm_head,
            anchor_id,
            max_len,
            params,
            rng,
        )
    }

    fn verify_ar_probe(&mut self, anchor_id: u32) -> Result<DsparkVerifyOutput> {
        self.ensure_clean("verify_ar_probe")?;
        let first_pos = {
            let rt = self.inner.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2 DSpark AR probe: paged runtime missing")
            })?;
            rt.ensure_conv_state(self.seq_id, &self.inner.config)?;
            let pos = rt
                .coordinator
                .request_token_count_all(self.seq_id)
                .map_err(Error::from_reason)?;
            if let Some(snap) = rt.conv_states.get(&self.seq_id).cloned() {
                rt.prev_conv_snapshots.insert(self.seq_id, (pos, snap));
            }
            rt.coordinator
                .record_tokens_all(self.seq_id, &[anchor_id])
                .map_err(Error::from_reason)?;
            pos
        };
        let (logits_2d, taps) = self.inner.forward_paged_single_chunk_with_taps(
            self.seq_id,
            &[anchor_id],
            None,
            first_pos,
            true,
            &self.tap_visits,
        )?;
        let logits_2d = logits_2d
            .ok_or_else(|| Error::from_reason("Inkling2 DSpark AR probe produced no logits"))?;
        let vocab = logits_2d.shape()?[1];
        let logits = logits_2d.reshape(&[1, 1, vocab])?;
        if let Some(rt) = self.inner.paged.as_mut() {
            rt.record_block_boundary_conv(self.seq_id, first_pos.saturating_add(1));
        }
        self.ar_probe_pending = true;
        self.ar_probe_taps = Some(taps);
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit_ar_probe(&mut self) -> Result<()> {
        if !self.ar_probe_pending {
            return Err(Error::from_reason(
                "Inkling2 DSpark commit_ar_probe without open AR probe",
            ));
        }
        self.ar_probe_pending = false;
        let taps = self.ar_probe_taps.take();
        let _ = self.inner.settle_paged_step(self.seq_id)?;
        if !self.ar_fallback
            && let Some(taps) = taps
            && let Some(ref draft) = self.inner.dspark
        {
            self.ctx.append_taps(draft, &taps, 1)?;
        }
        Ok(())
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        self.ensure_clean("verify")?;
        if verify_ids.is_empty() {
            return Err(Error::from_reason(
                "Inkling2 DSpark verify_ids must not be empty",
            ));
        }

        // Fast AR fallback path (post break-even decision): bypass verify/tape machinery cleanly.
        if self.ar_fallback && verify_ids.len() == 1 {
            let logits_2d = self.inner.paged_decode_step(self.seq_id, verify_ids[0])?;
            let vocab = logits_2d.shape()?[1];
            let logits = logits_2d.reshape(&[1, 1, vocab])?;
            return Ok(DsparkVerifyOutput { logits });
        }

        let (first_pos, ticket) = {
            let rt = self.inner.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2 DSpark verify: paged runtime missing")
            })?;
            rt.ensure_conv_state(self.seq_id, &self.inner.config)?;
            let pos = rt
                .coordinator
                .request_token_count_all(self.seq_id)
                .map_err(Error::from_reason)?;
            let ticket = rt
                .coordinator
                .record_verify(self.seq_id, verify_ids)
                .map_err(Error::from_reason)?;
            (pos, ticket)
        };

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
        let forward_res = self.inner.paged_verify_ragged_taped(&rows, tap_indices);
        let (logits_2d, taps, mut tapes) = match forward_res {
            Ok(out) => out,
            Err(err) => {
                if let Some(rt) = self.inner.paged.as_mut() {
                    let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                }
                return Err(err);
            }
        };

        let total_rows = verify_ids.len() as i64;
        let vocab = logits_2d.shape()?[1];
        let logits = logits_2d.reshape(&[1, total_rows, vocab])?;
        let tape = tapes.pop().ok_or_else(|| {
            Error::from_reason("Inkling2 DSpark verify did not produce a ShortConvTape")
        })?;

        self.open_ticket = Some(ticket);
        self.open_tape = Some(tape);
        self.open_taps = if self.ar_fallback { None } else { Some(taps) };
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
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

        let ticket_rows = ticket.rows();
        if keep == 0 || keep > total_written || total_written != ticket_rows {
            if let Some(rt) = self.inner.paged.as_mut() {
                let _ = rt.coordinator.commit_cycle(self.seq_id, ticket, 0);
                if let Some(conv) = rt.conv_states.get_mut(&self.seq_id) {
                    let _ = tape.commit_prefix(0, conv);
                }
            }
            return Err(Error::from_reason(format!(
                "Inkling2 DSpark invalid commit keep={keep}, total_written={total_written}, ticket_rows={ticket_rows}"
            )));
        }

        let committed_tokens = {
            let rt = self.inner.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Inkling2 DSpark commit: paged runtime missing")
            })?;
            // 1. Roll back rejected KV rows across both Global & SWA pools
            rt.coordinator
                .commit_cycle(self.seq_id, ticket, keep)
                .map_err(Error::from_reason)?;

            // 2. Roll back 48-visit ShortConv state to exact `keep`-token prefix via tape!
            let conv = rt.conv_states.get_mut(&self.seq_id).ok_or_else(|| {
                Error::from_reason("Inkling2 DSpark commit: missing ShortConv state")
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
            && let Some(ref draft) = self.inner.dspark
        {
            self.ctx.append_taps(draft, &taps, keep)?;
            debug_assert_eq!(self.ctx.committed_len as u32, committed_tokens);
        }
        Ok(())
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

    fn eval_boundary(&self, token: &MxArray) {
        if !crate::accelerator::supports_async_overlap() {
            return;
        }
        let mut arrays: Vec<&MxArray> = Vec::new();
        if let Some(ref rt) = self.inner.paged
            && let Some(conv) = rt.conv_states.get(&self.seq_id)
        {
            for v in &conv.visits {
                arrays.push(&v.k);
                arrays.push(&v.v);
                arrays.push(&v.attn_post);
                arrays.push(&v.mlp_post);
            }
        }
        self.ctx.collect_arrays(&mut arrays);
        arrays.push(token);
        MxArray::schedule_overlap(&arrays);
    }
}

impl DsparkBackend for Inkling2Inner {
    type DsparkDecode<'a>
        = Inkling2DsparkStepper<'a>
    where
        Self: 'a;

    fn begin_dspark_decode(&mut self, block_size: usize) -> Result<Self::DsparkDecode<'_>> {
        let draft = self
            .dspark
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 has no loaded DSpark draft model"))?;
        let expected = draft.config.num_draft_tokens();
        if block_size != expected {
            return Err(Error::from_reason(format!(
                "Inkling2 DSpark block_size {block_size} != draft checkpoint {expected}"
            )));
        }
        let tap_visits = draft.config.target_layer_ids.clone();
        let seq_id = self
            .paged
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 DSpark requires paged runtime"))?
            .active_seq;
        let ctx = self.dspark_turn_contexts.remove(&seq_id).ok_or_else(|| {
            Error::from_reason(format!(
                "Inkling2 DSpark turn context missing for seq {seq_id} (prefill did not seed it)"
            ))
        })?;
        Ok(Inkling2DsparkStepper {
            inner: self,
            seq_id,
            ctx,
            tap_visits,
            ar_fallback: false,
            open_ticket: None,
            open_tape: None,
            open_taps: None,
            ar_probe_pending: false,
            ar_probe_taps: None,
        })
    }
}

impl PagedDsparkBackend for Inkling2Inner {
    fn dspark_block_size(&self) -> Result<usize> {
        let draft = self
            .dspark
            .as_ref()
            .ok_or_else(|| Error::from_reason("Inkling2 has no loaded DSpark draft model"))?;
        Ok(draft.config.num_draft_tokens())
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

        // Only reuse a cached prefix on the whole-turn DSpark path when `seq_id` already holds
        // a matching live `Inkling2DsparkContext` at the exact prefix boundary; otherwise start
        // fresh from position 0 so the draft context receives every token's tapped hiddens.
        let prefix = self.prime_prefix_state(tokens, reuse_cache, 0, &[], cache_salt)?;
        let boundary = prefix.effective_cached_prefix_len;
        let has_matching_draft_ctx = boundary > 0
            && prefix.live_continuation
            && self
                .dspark_turn_contexts
                .get(&seq_id)
                .is_some_and(|ctx| ctx.committed_len as usize == boundary);
        if boundary == 0 || has_matching_draft_ctx {
            if boundary == 0 {
                let n_layers = self
                    .dspark
                    .as_ref()
                    .map_or(3, |d| d.config.num_hidden_layers);
                self.dspark_turn_contexts
                    .insert(seq_id, Inkling2DsparkContext::new(n_layers));
            }
            return Ok(prefix);
        }

        // Prefix hit had no matching draft context (e.g. cold SSD restore): restart fresh so
        // DSpark prefill seeds the draft context cleanly from position 0.
        self.dspark_turn_contexts.remove(&seq_id);
        let fresh = self.prime_prefix_state(tokens, false, 0, &[], cache_salt)?;
        let n_layers = self
            .dspark
            .as_ref()
            .map_or(3, |d| d.config.num_hidden_layers);
        self.dspark_turn_contexts
            .insert(seq_id, Inkling2DsparkContext::new(n_layers));
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
        let mut draft_ctx = self
            .dspark_turn_contexts
            .remove(&seq_id)
            .unwrap_or_else(|| Inkling2DsparkContext::new(n_draft_layers));

        let (logits_2d, _) = self.paged_prefill_sequence_with_taps(
            seq_id,
            suffix_tokens,
            None,
            &tap_visits,
            |draft, taps, chunk_len| draft_ctx.append_taps(draft, taps, chunk_len),
        )?;
        draft_ctx.eval_state()?;
        self.dspark_turn_contexts.insert(seq_id, draft_ctx);
        logits_2d.squeeze(Some(&[0]))
    }
}
