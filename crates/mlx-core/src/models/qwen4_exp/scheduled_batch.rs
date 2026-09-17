//! Stateful attention remains per owner; stateless HC/MLP projections share a
//! layer across ready decode rows. No owner sees another owner's cache/positions.
use super::*;

impl Decoder {
    pub(in super::super) fn scheduled_start(&mut self, token: u32) -> Result<MxArray> {
        self.check_cancelled()?;
        if token as usize >= self.config.vocab_size
            || self.history.len() >= self.config.effective_context_limit()
        {
            return Err(Error::from_reason(
                "Qwen4 scheduled token exceeds vocabulary/context",
            ));
        }
        let adapter = self
            .paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen4 scheduled row has no pages"))?;
        if adapter.request_tokens() != self.history {
            return Err(Error::from_reason("Qwen4 scheduled frontier mismatch"));
        }
        adapter
            .record_tokens(&[token])
            .map_err(Error::from_reason)?;
        MxArray::tile(
            &self.embed_token(token)?,
            &[1, 1, self.config.hc_count as i32],
        )
    }
    pub(in super::super) fn scheduled_attention(
        &mut self,
        x: MxArray,
        token: u32,
        layer: usize,
    ) -> Result<MxArray> {
        self.check_cancelled()?;
        self.paged_chunk_start = self.history.len();
        let mut cache = std::mem::take(&mut self.caches[layer]);
        let output = self.attention_block(x, token, layer, &mut cache)?;
        if !self.config.linear(layer) {
            let index = (0..layer).filter(|&i| !self.config.linear(i)).count() as u32;
            let shape = [
                1,
                self.config.num_key_value_heads as i64,
                self.config.head_dim as i64,
            ];
            let k = cache
                .keys
                .take()
                .ok_or_else(|| Error::from_reason("Qwen4 scheduled attention produced no keys"))?
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&shape)?
                .astype(DType::BFloat16)?;
            let v = cache
                .values
                .take()
                .ok_or_else(|| Error::from_reason("Qwen4 scheduled attention produced no values"))?
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&shape)?
                .astype(DType::BFloat16)?;
            let adapter = self.paged.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen4 scheduled attention has no paged cache")
            })?;
            super::super::paged::write_rows(adapter, index, &k, &v, self.history.len() as u32)?;
            // Owners share the pool. Complete the write before activating another.
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
        }
        self.caches[layer] = cache;
        Ok(output)
    }
    pub(in super::super) fn scheduled_mlp(
        &mut self,
        rows: &[MxArray],
        layer: usize,
    ) -> Result<Vec<MxArray>> {
        self.mlp_batch(rows, layer)
    }
    pub(in super::super) fn scheduled_finish(
        &mut self,
        x: &MxArray,
        token: u32,
    ) -> Result<MxArray> {
        self.last_chunk_hidden = vec![x.clone()];
        let (mixed, _) = self.hyper(x, "hyper_connection_mixer", "output_hc", false)?;
        let logits = self.linear(&mixed, "lm_head.weight", "output.weight")?;
        MxArray::eval_arrays_with_context(&[&logits], "qwen4::scheduled::logits")?;
        self.history.push(token);
        Ok(logits)
    }
}
