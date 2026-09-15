//! Prepare the MTP attention cache from trusted target hidden states. The
//! discarded draft MLP/output head cannot affect future attention state.
use super::*;

impl Decoder {
    pub(in super::super) fn draft_prefill(
        &mut self,
        hidden: &[MxArray],
        tokens: &[u32],
        position: usize,
        cache: &mut LayerCache,
    ) -> Result<()> {
        if hidden.len() != tokens.len()
            || tokens.is_empty()
            || tokens.len() > self.prefill_chunk_size
        {
            return Err(Error::from_reason(
                "Qwen4 MTP history window exceeds admission",
            ));
        }
        let mut embeddings = Vec::with_capacity(tokens.len());
        for &token in tokens {
            self.check_cancelled()?;
            embeddings.push(self.embed_token(token)?);
        }
        let embedding = MxArray::concatenate_many(embeddings.iter().collect(), Some(1))?;
        let hidden = MxArray::concatenate_many(hidden.iter().collect(), Some(1))?;
        let pages = self.paged.take();
        let old_position = self.position_override.take();
        self.scope = "mtp.".into();
        let result = (|| {
            let c = self.config.clone();
            let n = tokens.len() as i64;
            let e = self.norm(
                &embedding,
                "pre_fc_norm_embedding.weight",
                "",
                c.hidden_size,
                true,
            )?;
            let e = self.linear(&e, "fc_embedding.weight", "")?;
            let h = self
                .norm(
                    &hidden,
                    "pre_fc_norm_hidden.weight",
                    "",
                    c.hidden_size * c.hc_count,
                    true,
                )?
                .reshape(&[1, n, c.hc_count as i64, c.hidden_size as i64])?;
            let h = self.linear(&h, "fc_hidden.weight", "")?;
            let x =
                h.add(&e.expand_dims(2)?)?
                    .reshape(&[1, n, (c.hc_count * c.hidden_size) as i64])?;
            let (input, _) = self.hyper(&x, "layers.0.attn_hyper_connection", "", false)?;
            let k = self
                .linear(&input, "layers.0.self_attn.k_proj.weight", "")?
                .reshape(&[1, n, c.num_key_value_heads as i64, c.head_dim as i64])?;
            let k = self
                .norm(&k, "layers.0.self_attn.k_norm.weight", "", c.head_dim, true)?
                .transpose(Some(&[0, 2, 1, 3]))?;
            let k = self.rope_window(&k, position, tokens.len())?;
            let v = self
                .linear(&input, "layers.0.self_attn.v_proj.weight", "")?
                .reshape(&[1, n, c.num_key_value_heads as i64, c.head_dim as i64])?
                .transpose(Some(&[0, 2, 1, 3]))?;
            let key = self.key("layers.0.self_attn.indexer.index_qk_proj.weight", "");
            let ik = self
                .weights
                .read(
                    &key,
                    c.indexer_n_heads * c.indexer_head_dim,
                    c.indexer_head_dim,
                )?
                .linear(&input)?;
            let tail = cache.index_tail.len();
            let old_blocks = cache.index_blocks.len();
            let mut raw = cache.index_tail.clone();
            raw.push(ik.reshape(&[n, c.indexer_head_dim as i64])?);
            let raw = MxArray::concatenate_many(raw.iter().collect(), Some(0))?;
            let ratio = c.indexer_compress_ratio;
            let used = (tail + tokens.len()) / ratio * ratio;
            if used > 0 {
                let means = raw
                    .slice_axis(0, 0, used as i64)?
                    .reshape(&[-1, ratio as i64, c.indexer_head_dim as i64])?
                    .astype(DType::Float32)?
                    .mean(Some(&[1]), Some(false))?
                    .astype(input.dtype()?)?;
                let means = self.norm(
                    &means,
                    "layers.0.self_attn.indexer.k_layernorm.weight",
                    "",
                    c.indexer_head_dim,
                    true,
                )?;
                for block in 0..used / ratio {
                    cache.index_blocks.push(self.rope(
                        &means.slice_axis(0, block as i64, block as i64 + 1)?,
                        position - tail + block * ratio,
                    )?);
                }
            }
            cache.index_tail = (used..tail + tokens.len())
                .map(|r| raw.slice_axis(0, r as i64, r as i64 + 1))
                .collect::<Result<_>>()?;
            cache.keys = Some(match &cache.keys {
                Some(old) => MxArray::concatenate(old, &k, 2)?,
                None => k,
            });
            cache.values = Some(match &cache.values {
                Some(old) => MxArray::concatenate(old, &v, 2)?,
                None => v,
            });
            let arrays: Vec<_> = cache
                .keys
                .iter()
                .chain(cache.values.iter())
                .chain(cache.index_tail.iter())
                .chain(cache.index_blocks[old_blocks..].iter())
                .collect();
            MxArray::eval_arrays_with_context(&arrays, "qwen4::mtp::history_cache")
        })();
        self.scope.clear();
        self.paged = pages;
        self.position_override = old_position;
        result
    }
}
