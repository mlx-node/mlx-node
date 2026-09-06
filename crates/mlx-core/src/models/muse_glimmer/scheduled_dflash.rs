//! Paged, owner-isolated DFlash context and target verification.
use super::{MuseGlimmerInner, MusePagedSettle};
use crate::array::MxArray;
use crate::engine::backend::{DsparkProposal, SpecFrontier};
use crate::engine::hybrid_scheduler::{ScheduledVerifyCommit, ScheduledVerifyRow};
use crate::engine::params::ChatParams;
use crate::engine::scheduled_verify::{ScheduledDraftVerify, ScheduledVerifyBatch};
use crate::engine::spec_paged::SpecPagedCache;
use crate::models::gemma4::dspark::DsparkTap;
use crate::models::muse_glimmer::dflash::DFlashContextCache;
use crate::models::muse_glimmer::kv_cache::PagedWindowSlot;
use crate::transformer::paged_kv_cache_adapter::PagedRaggedRow;
use napi::bindgen_prelude::{Error, Result};

pub(crate) struct ScheduledDFlashState {
    context: DFlashContextCache,
}

pub(crate) struct MuseSpecPagedCache<'a>(&'a mut MuseGlimmerInner);

impl<'a> MuseSpecPagedCache<'a> {
    pub(crate) fn new(inner: &'a mut MuseGlimmerInner) -> Self {
        Self(inner)
    }
}

impl SpecPagedCache for MuseSpecPagedCache<'_> {
    fn reserve_lookahead(&mut self, seq: u32, rows: usize) -> std::result::Result<bool, String> {
        let paged = self.0.paged.as_mut().ok_or("Muse paged runtime missing")?;
        match paged.coordinator.reserve_rows_all(
            seq,
            u32::try_from(rows).map_err(|_| "verify width overflow")?,
        ) {
            Ok(_) => Ok(true),
            Err(error) if error.starts_with("context_length_exceeded:") => Ok(false),
            Err(error) => Err(error),
        }
    }
    fn record_rows(&mut self, seq: u32, tokens: &[u32]) -> std::result::Result<(), String> {
        self.0
            .paged
            .as_mut()
            .ok_or("Muse paged runtime missing")?
            .coordinator
            .record_tokens_all(seq, tokens)
    }
    fn rollback_rows(&mut self, seq: u32, rows: usize) -> std::result::Result<(), String> {
        self.0
            .paged
            .as_mut()
            .ok_or("Muse paged runtime missing")?
            .coordinator
            .rollback_last_tokens_all(
                seq,
                u32::try_from(rows).map_err(|_| "rollback width overflow")?,
            )
    }
    fn settle_committed(&mut self, seq: u32, tokens: u64) -> std::result::Result<(), String> {
        let tokens = u32::try_from(tokens).map_err(|_| "committed frontier overflow")?;
        self.0
            .settle_paged_kv_step(seq, MusePagedSettle::Committed(tokens))
            .map_err(|error| error.reason)
    }
    fn settle_captures_durable_state(&self) -> bool {
        true
    }
    fn frontier(&self, seq: u32) -> Option<SpecFrontier> {
        self.0.paged.as_ref()?.coordinator.spec_frontier(seq)
    }
}

impl MuseGlimmerInner {
    pub(crate) fn clear_scheduled_dflash(&mut self) -> Result<()> {
        if let Some(batch) = self.scheduled_dflash_verify.take() {
            self.retract_verify_batch(batch)?;
        }
        self.scheduled_dflash_states.clear();
        Ok(())
    }
    pub(crate) fn begin_scheduled_dflash(&mut self, seq: u32, position: u32) -> Result<()> {
        if self.scheduled_dflash_verify.is_some() {
            return Err(Error::from_reason("cannot seed DFlash during verification"));
        }
        let draft = self
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("DFlash is not loaded"))?;
        let position =
            i32::try_from(position).map_err(|_| Error::from_reason("DFlash position overflow"))?;
        self.scheduled_dflash_states.insert(
            seq,
            ScheduledDFlashState {
                context: DFlashContextCache::new_at(&draft.config, position),
            },
        );
        Ok(())
    }

    pub(crate) fn propose_scheduled_dflash(
        &mut self,
        seq: u32,
        anchor: u32,
        max_drafts: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        let draft = self
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("DFlash is not loaded"))?;
        let state = self
            .scheduled_dflash_states
            .get(&seq)
            .ok_or_else(|| Error::from_reason("DFlash owner context missing"))?;
        let (draft_ids, draft_dists) = draft.propose(
            &self.embed_tokens,
            self.lm_head.as_ref(),
            &self.config.text_config,
            &state.context,
            anchor,
            max_drafts,
            &params.sampling_config.unwrap_or_default(),
            rng,
        )?;
        Ok(DsparkProposal {
            draft_ids,
            draft_dists,
            draft_sparse_dists: Vec::new(),
            keep_probabilities: None,
        })
    }

    pub(crate) fn prefill_scheduled_dflash(
        &mut self,
        seq: u32,
        tokens: &[u32],
        position: u32,
    ) -> Result<MxArray> {
        let ids = self
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("DFlash is not loaded"))?
            .config
            .target_layers
            .clone();
        let mut tap = DsparkTap::new(&ids);
        let hidden = self.run_paged_layer_loop(
            tokens,
            position,
            true,
            MusePagedSettle::Cursor,
            Some(&mut tap),
        )?;
        self.append_committed_taps(seq, &tap.captured, tokens.len())?;
        self.project_logits(&hidden, true)?.squeeze(Some(&[0, 1]))
    }

    pub(crate) fn verify_scheduled_dflash(
        &mut self,
        rows: &[ScheduledVerifyRow],
    ) -> Result<MxArray> {
        self.verify_scheduled_rows(rows)
    }
    pub(crate) fn commit_scheduled_dflash(
        &mut self,
        rows: &[ScheduledVerifyCommit],
    ) -> Result<Vec<Result<()>>> {
        self.commit_scheduled_rows(rows)
    }
}

impl ScheduledDraftVerify for MuseGlimmerInner {
    type Cache<'a> = MuseSpecPagedCache<'a>;
    fn verify_cache(&mut self) -> Self::Cache<'_> {
        MuseSpecPagedCache(self)
    }
    fn take_verify_batch(&mut self) -> Option<ScheduledVerifyBatch> {
        self.scheduled_dflash_verify.take()
    }
    fn store_verify_batch(&mut self, batch: ScheduledVerifyBatch) {
        self.scheduled_dflash_verify = Some(batch);
    }
    fn validate_draft_owner(&self, row: &ScheduledVerifyRow) -> Result<()> {
        let state = self.scheduled_dflash_states.get(&row.seq_id);
        if row.speculative != state.is_some()
            || state.is_some_and(|state| state.context.logical_len() != row.first_position as i32)
        {
            return Err(Error::from_reason(
                "DFlash owner or context frontier mismatch",
            ));
        }
        Ok(())
    }
    fn discard_draft_owner(&mut self, seq: u32) {
        self.scheduled_dflash_states.remove(&seq);
    }
    fn append_committed_taps(&mut self, seq: u32, taps: &[MxArray], keep: usize) -> Result<()> {
        let draft = self
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("DFlash is not loaded"))?;
        let kept = taps
            .iter()
            .map(|tap| tap.slice_axis(1, 0, keep as i64))
            .collect::<Result<Vec<_>>>()?;
        let fused = draft.fuse_context(&kept)?;
        let state = self
            .scheduled_dflash_states
            .get_mut(&seq)
            .ok_or_else(|| Error::from_reason("DFlash owner context missing"))?;
        let position = state.context.logical_len();
        state.context.append(draft, &fused, position)
    }
    fn run_packed_target(
        &mut self,
        rows: &[(PagedRaggedRow, Vec<u32>)],
    ) -> Result<(MxArray, Vec<MxArray>)> {
        let tap_layers = self
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("DFlash is not loaded"))?
            .config
            .target_layers
            .clone();
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut metadata = Vec::with_capacity(rows.len());
        for (row, ids) in rows {
            if ids.is_empty() || ids.len() != row.query_len as usize {
                return Err(Error::from_reason("invalid Muse verifier width"));
            }
            metadata.push(*row);
            tokens.extend_from_slice(ids);
            for offset in 0..row.query_len {
                positions.push(
                    row.first_logical_position
                        .checked_add(offset)
                        .and_then(|p| i32::try_from(p).ok())
                        .ok_or_else(|| Error::from_reason("Muse verifier position overflow"))?,
                );
            }
        }
        let input = MxArray::from_uint32(&tokens, &[tokens.len() as i64, 1])?;
        let offsets = MxArray::from_int32(&positions, &[positions.len() as i64])?;
        let mut hidden = self.scaleless_rms_norm(&self.embed_tokens.forward(&input)?)?;
        let mut taps = Vec::with_capacity(tap_layers.len());
        for index in 0..self.layers.len() {
            let paged = self
                .paged
                .as_mut()
                .ok_or_else(|| Error::from_reason("Muse paged runtime missing"))?;
            let route = paged
                .routes
                .get(index)
                .ok_or_else(|| Error::from_reason("Muse layer route missing"))?;
            let window: PagedWindowSlot = paged.decode_windows[route.group_id];
            let adapter = paged
                .coordinator
                .adapter_mut(route.group_id)
                .map_err(Error::from_reason)?;
            hidden = self.layers[index].forward_paged_ragged(
                &hidden,
                adapter,
                route.physical_layer_ordinal as u32,
                &metadata,
                &offsets,
                window,
                self.row_exact_decode_projections,
            )?;
            if tap_layers.binary_search(&index).is_ok() {
                taps.push(hidden.clone());
            }
        }
        if taps.len() != tap_layers.len() {
            return Err(Error::from_reason("Muse verifier missed DFlash taps"));
        }
        let logits = if self.row_exact_decode_projections {
            crate::models::muse_glimmer::row_exact::forward_owner_spans(
                &hidden,
                &metadata,
                |span| self.project_logits(span, false),
            )?
        } else {
            self.project_logits(&hidden, false)?
        };
        Ok((logits, taps))
    }
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) mod tests {
    use super::*;
    use crate::engine::hybrid_scheduler::{HybridSchedulerBackend, ScheduledPrefixAdmission};
    use crate::models::gemma4::quantized_linear::LinearProj;
    use crate::models::muse_glimmer::{
        attention::MuseGlimmerAttention,
        config::MuseGlimmerDFlashConfig,
        decoder_layer::MuseGlimmerDecoderLayer,
        dflash::{DFlashAttention, DFlashLayer, DFlashModel},
        kv_cache::{WindowCarrier, paged_window_for_kind},
        mlp::MuseGlimmerMlp,
    };
    use crate::nn::{Linear, RMSNorm};
    use crate::stream::{DeviceType, Stream};
    use crate::transformer::AttentionKind;

    fn linear(input: u32, output: u32) -> LinearProj {
        let weight = Linear::new(input, output, Some(false))
            .unwrap()
            .get_weight()
            .astype(crate::array::DType::BFloat16)
            .unwrap();
        LinearProj::Standard(Linear::from_weights(&weight, None).unwrap())
    }
    fn norm(width: u32) -> RMSNorm {
        let weight = MxArray::from_float32(&vec![1.0; width as usize], &[i64::from(width)])
            .unwrap()
            .astype(crate::array::DType::BFloat16)
            .unwrap();
        RMSNorm::from_weight(&weight, Some(1e-5)).unwrap()
    }
    fn mlp() -> MuseGlimmerMlp {
        MuseGlimmerMlp::new(linear(8, 16), linear(8, 16), linear(16, 8))
    }

    pub(crate) fn seeded_inner(seed: u64) -> Option<MuseGlimmerInner> {
        unsafe { mlx_sys::mlx_seed(seed) };
        let mut inner = super::super::spec_paged_settle_tests::maybe_tiny_inner()?;
        let embedding = inner
            .embed_tokens
            .get_weight()
            .astype(crate::array::DType::BFloat16)
            .unwrap();
        inner.embed_tokens.set_weight(&embedding).unwrap();
        inner.final_norm = norm(8);
        for index in 0..4 {
            let attention = MuseGlimmerAttention::from_projections(
                &inner.config.text_config,
                index,
                false,
                linear(8, 64),
                linear(8, 32),
                linear(8, 32),
                linear(64, 8),
                linear(8, 64),
            )
            .unwrap();
            inner.layers.push(MuseGlimmerDecoderLayer::new(
                attention,
                mlp(),
                norm(8),
                norm(8),
                norm(8),
                norm(8),
            ));
        }
        let draft = MuseGlimmerDFlashConfig {
            num_hidden_layers: 1,
            block_size: 3,
            hidden_size: 8,
            intermediate_size: 16,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 32,
            sliding_window: 16,
            max_position_embeddings: 256,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            target_layers: vec![0, 2],
            mask_token_id: 63,
            causal: false,
        };
        let attention = DFlashAttention::new(
            &draft,
            linear(8, 64),
            linear(8, 32),
            linear(8, 32),
            linear(64, 8),
            norm(32),
            norm(32),
        );
        inner.dflash = Some(DFlashModel::from_loaded(
            draft,
            linear(16, 8),
            norm(8),
            vec![DFlashLayer::new(attention, mlp(), norm(8), norm(8))],
            norm(8),
        ));
        let paged = inner.paged.as_mut().unwrap();
        let kinds = [
            AttentionKind::Full,
            AttentionKind::SlidingWindow { sliding_window: 16 },
        ];
        paged.prefill_windows = kinds
            .iter()
            .map(|&kind| paged_window_for_kind(kind, WindowCarrier::ExplicitMask).unwrap())
            .collect();
        paged.decode_windows = kinds
            .iter()
            .map(|&kind| paged_window_for_kind(kind, WindowCarrier::KernelArgument).unwrap())
            .collect();
        Some(inner)
    }

    fn seed(inner: &mut MuseGlimmerInner, seq: u32, tokens: &[u32], draft: bool) {
        let prefix = match inner
            .prepare_scheduled_prefix(seq, tokens, &[], false, 0, 8)
            .unwrap()
        {
            ScheduledPrefixAdmission::Ready(prefix) => prefix,
            _ => unreachable!(),
        };
        inner.activate_paged_seq(seq).unwrap();
        if draft {
            inner.begin_scheduled_dflash(seq, 0).unwrap();
        }
        inner
            .run_scheduled_prefill_slice(
                seq,
                tokens,
                &prefix,
                0,
                tokens.len(),
                Stream::default(DeviceType::Gpu),
                true,
            )
            .unwrap()
            .unwrap()
            .eval();
    }

    #[test]
    fn ragged_dflash_preserves_owners_rejection_and_sliding_frontiers() {
        let Some(mut packed) = seeded_inner(1906) else {
            return;
        };
        let mut scalar = seeded_inner(1906).unwrap();
        for inner in [&mut packed, &mut scalar] {
            seed(inner, 51, &(0..13).collect::<Vec<_>>(), true);
            seed(inner, 52, &(4..23).collect::<Vec<_>>(), true);
        }
        let rows = [
            ScheduledVerifyRow {
                seq_id: 51,
                first_position: 13,
                tokens: vec![13, 14, 15],
                speculative: true,
            },
            ScheduledVerifyRow {
                seq_id: 52,
                first_position: 19,
                tokens: vec![23, 24],
                speculative: true,
            },
        ];
        let actual = packed.verify_scheduled_rows(&rows).unwrap();
        actual.eval();
        let actual = actual.to_float32().unwrap();
        let mut expected = Vec::new();
        for (row, keep) in rows.iter().zip([2, 1]) {
            let logits = scalar
                .verify_scheduled_rows(std::slice::from_ref(row))
                .unwrap();
            logits.eval();
            expected.extend_from_slice(&logits.to_float32().unwrap());
            scalar
                .commit_scheduled_rows(&[ScheduledVerifyCommit {
                    seq_id: row.seq_id,
                    keep,
                }])
                .unwrap()
                .pop()
                .unwrap()
                .unwrap();
        }
        for result in packed
            .commit_scheduled_rows(&[
                ScheduledVerifyCommit {
                    seq_id: 51,
                    keep: 2,
                },
                ScheduledVerifyCommit {
                    seq_id: 52,
                    keep: 1,
                },
            ])
            .unwrap()
        {
            result.unwrap();
        }
        let error = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(error < 0.02, "packed versus independent error={error}");
        for seq in [51, 52] {
            assert_eq!(
                packed.paged_adapter().unwrap().request_tokens_for(seq),
                scalar.paged_adapter().unwrap().request_tokens_for(seq)
            );
            assert_eq!(
                packed.scheduled_dflash_states[&seq].context.logical_len(),
                if seq == 51 { 15 } else { 20 }
            );
        }
        // Invalid commit retracts both owners, and no draft context advances.
        let next = [
            ScheduledVerifyRow {
                seq_id: 51,
                first_position: 15,
                tokens: vec![30, 31],
                speculative: true,
            },
            ScheduledVerifyRow {
                seq_id: 52,
                first_position: 20,
                tokens: vec![32],
                speculative: true,
            },
        ];
        packed.verify_scheduled_rows(&next).unwrap().eval();
        assert!(
            packed
                .commit_scheduled_rows(&[ScheduledVerifyCommit {
                    seq_id: 51,
                    keep: 3
                }])
                .is_err()
        );
        assert_eq!(
            packed.scheduled_dflash_states[&51].context.logical_len(),
            15
        );
        assert_eq!(packed.verify_cache().frontier(51).unwrap().attn_tokens, 15);
        assert_eq!(packed.verify_cache().frontier(52).unwrap().attn_tokens, 20);
    }
}
