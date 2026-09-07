use super::*;
use crate::models::qwen3_5::gated_delta_net::GdnLayerTape;
use crate::models::qwen3_5::scheduled_mtp::{ScheduledMtpState, ScheduledMtpTarget};
use crate::transformer::paged_kv_cache_adapter::PagedRaggedRow;

impl ScheduledMtpTarget for Qwen35MoeInner {
    fn mtp_state(&self) -> &ScheduledMtpState {
        &self.scheduled_mtp
    }
    fn mtp_state_mut(&mut self) -> &mut ScheduledMtpState {
        &mut self.scheduled_mtp
    }
    fn mtp_adapter(&self) -> Option<&PagedKVCacheAdapter> {
        self.paged_adapter.as_ref()
    }
    fn mtp_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter> {
        self.paged_adapter.as_mut()
    }
    fn park_mtp_target(&mut self) -> Result<()> {
        self.park_active_scheduled_recurrent()
    }
    fn mtp_target_caches(&self, seq: u32) -> Result<&[Qwen3_5LayerCache]> {
        self.scheduled_recurrent
            .live(seq)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                Error::from_reason(format!("MTP owner {seq} has no resident target state"))
            })
    }
    fn replace_mtp_target_caches(
        &mut self,
        seq: u32,
        caches: Vec<Qwen3_5LayerCache>,
    ) -> Result<()> {
        self.scheduled_recurrent
            .insert_live(seq, self.config.recurrent_state_bytes(), caches)
            .map(|_| ())
            .map_err(Error::from_reason)
    }
    fn fresh_mtp_draft_caches(&self) -> Vec<Qwen3_5LayerCache> {
        Qwen3_5MoeMTPModule::fresh_caches(&self.config)
    }
    fn embed_mtp_token(&self, ids: &MxArray) -> Result<MxArray> {
        self.embedding.forward(ids)
    }
    fn run_mtp_draft_hidden(
        &mut self,
        hidden: &MxArray,
        embedding: &MxArray,
        caches: &mut [Qwen3_5LayerCache],
    ) -> Result<MxArray> {
        self.mtp
            .as_mut()
            .ok_or_else(|| Error::from_reason("scheduled MTP draft disappeared"))?
            .forward(hidden, embedding, Some(caches))
    }
    fn project_mtp_logits(&self, hidden: &MxArray) -> Result<MxArray> {
        super::forward::project_logits_from_hidden(hidden, &self.lm_head, &self.embedding)
    }
    #[cfg(test)]
    fn run_reference_mtp_target(
        &mut self,
        seq: u32,
        ids: &[u32],
        caches: &mut [Qwen3_5LayerCache],
        tape: &mut Vec<Option<GdnLayerTape>>,
    ) -> Result<MxArray> {
        let adapter = self.paged_adapter.as_mut().unwrap();
        adapter.activate_request(seq).map_err(Error::from_reason)?;
        crate::models::qwen3_5_moe::paged_forward::run_paged_verify_step(
            ids,
            &self.embedding,
            &mut self.layers,
            caches,
            &self.final_norm,
            &self.lm_head,
            &self.layer_kinds,
            adapter,
            tape,
            0,
        )
        .map(|output| output.logits)
    }
    fn run_mtp_target_bucket(
        &mut self,
        rows: &[(PagedRaggedRow, Vec<u32>)],
        caches: &mut [Qwen3_5LayerCache],
        tape: &mut [Option<GdnLayerTape>],
    ) -> Result<(MxArray, MxArray)> {
        if rows.is_empty()
            || caches.len() != self.layers.len()
            || tape.len() != self.layers.len()
            || self.layer_kinds.len() != self.layers.len()
        {
            return Err(Error::from_reason("invalid scheduled MTP target layout"));
        }
        if self.cached_rope_deltas.unwrap_or(0) != 0 {
            return Err(Error::from_reason(
                "scheduled MTP requires text-only position state",
            ));
        }
        let width = rows[0].1.len();
        if rows.iter().any(|(_, ids)| ids.len() != width) {
            return Err(Error::from_reason(
                "MTP GDN bucket has unequal query widths",
            ));
        }
        let ids = rows
            .iter()
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect::<Vec<_>>();
        let positions = rows
            .iter()
            .map(|(row, _)| (row.seq_id, row.first_logical_position))
            .collect::<Vec<_>>();
        let ids = MxArray::from_uint32(&ids, &[rows.len() as i64, width as i64])?;
        let mut hidden = self.embedding.forward(&ids)?;
        for index in 0..self.layers.len() {
            let adapter = self
                .paged_adapter
                .as_mut()
                .ok_or_else(|| Error::from_reason("MTP target has no paged adapter"))?;
            hidden = self.layers[index].forward_paged_batched_with_tape(
                &hidden,
                self.layer_kinds[index],
                adapter,
                &positions,
                Some(&mut caches[index]),
                self.row_exact_decode_projections,
                Some(&mut tape[index]),
            )?;
        }
        let hidden = self.final_norm.forward(&hidden)?;
        let logits =
            super::forward::project_logits_from_hidden(&hidden, &self.lm_head, &self.embedding)?;
        Ok((logits, hidden))
    }
}

#[cfg(test)]
pub(crate) fn seeded_inner(seed: u64) -> Qwen35MoeInner {
    // Tests run serially because MLX owns one shared device and PRNG.
    unsafe { mlx_sys::mlx_seed(seed) };
    let mut config = super::paged_construction_tests::tiny_paged_forward_moe_cfg();
    config.n_mtp_layers = 1;
    config.vocab_size = 16;
    let mut inner = Qwen35MoeInner::new(config).unwrap();
    super::paged_construction_tests::cast_moe_inner_weights_bf16(&mut inner);
    // Numerical parity is checked separately against the original verifier.
    // A fixed non-uniform head makes this fixture isolate scheduler/RNG
    // behavior from GEMV/GEMM rounding when a peer leaves the batch.
    let mut head = Linear::new(inner.config.hidden_size as u32, 16, Some(true)).unwrap();
    head.set_weight(
        &MxArray::zeros(
            &[16, inner.config.hidden_size as i64],
            Some(crate::array::DType::BFloat16),
        )
        .unwrap(),
    )
    .unwrap();
    let bias = MxArray::from_float32(&(0..16).map(|i| i as f32 / 16.0).collect::<Vec<_>>(), &[16])
        .unwrap()
        .astype(crate::array::DType::BFloat16)
        .unwrap();
    head.set_bias(Some(&bias)).unwrap();
    inner.lm_head = Some(LinearProj::Standard(head));
    inner.initialize_paged_adapter().unwrap();
    inner
}
