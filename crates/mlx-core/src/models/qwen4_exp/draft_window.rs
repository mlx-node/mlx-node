//! The MTP head over consecutive (target hidden, next token) rows. One pass
//! shares the input projections across its rows. Rows that only extend the
//! draft history skip the head: its MLP/output cannot affect attention state.
use super::*;

impl Decoder {
    /// Every row extends `cache`, evaluated before returning.
    pub(in super::super) fn draft_prefill(
        &mut self,
        hidden: &[MxArray],
        tokens: &[u32],
        position: usize,
        cache: &mut LayerCache,
    ) -> Result<()> {
        self.draft_rows(hidden, tokens, position, cache, false)?;
        let arrays: Vec<_> = cache.arrays().collect();
        MxArray::eval_arrays_with_context(&arrays, "qwen4::mtp::history_cache")
    }

    /// Rows before the last extend `cache`. The last row runs the head over a
    /// copy of the extended cache, returned with the next hidden and the
    /// logits. Nothing is evaluated here.
    pub fn draft_step(
        &mut self,
        hidden: &[MxArray],
        tokens: &[u32],
        position: usize,
        cache: &mut LayerCache,
    ) -> Result<(MxArray, MxArray, LayerCache)> {
        let (next, mixed, draft) = self
            .draft_rows(hidden, tokens, position, cache, true)?
            .ok_or_else(|| Error::from_reason("Qwen4 MTP draft step produced no head"))?;
        let logits = self.linear(&mixed, "lm_head.weight", "output.weight")?;
        Ok((next, logits, draft))
    }

    /// Scope, position and the target page adapter are restored on every
    /// error path.
    fn draft_rows(
        &mut self,
        hidden: &[MxArray],
        tokens: &[u32],
        position: usize,
        cache: &mut LayerCache,
        head: bool,
    ) -> Result<Option<(MxArray, MxArray, LayerCache)>> {
        if hidden.len() != tokens.len()
            || tokens.is_empty()
            || tokens.len() > self.prefill_chunk_size
        {
            return Err(Error::from_reason(
                "Qwen4 MTP history window exceeds admission",
            ));
        }
        let rows = match (cache.draft_rows, &cache.keys) {
            (Some(rows), _) => rows,
            (None, None) => 0,
            (None, Some(_)) => {
                return Err(Error::from_reason(
                    "Qwen4 MTP history is not a draft history",
                ));
            }
        };
        if rows != position {
            return Err(Error::from_reason(
                "Qwen4 MTP history does not end at the window start",
            ));
        }
        cache.draft_rows = Some(rows);
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
            let n = tokens.len();
            let e = self.norm(
                &embedding,
                "pre_fc_norm_embedding.weight",
                "",
                c.hidden_size,
                true,
            )?;
            let e = self.linear(&e, "fc_embedding.weight", "")?;
            // One RMS per hyper-connection stream, as the target's hc_norm
            // and llama.cpp's MTP graph: a joint RMS over the streams
            // (mlx-vlm) loses ~2 % of the accepted drafts on the checkpoint.
            let h = self
                .norm(
                    &hidden,
                    "pre_fc_norm_hidden.weight",
                    "",
                    c.hidden_size,
                    true,
                )?
                .reshape(&[1, n as i64, c.hc_count as i64, c.hidden_size as i64])?;
            let h = self.linear(&h, "fc_hidden.weight", "")?;
            let x = h.add(&e.expand_dims(2)?)?.reshape(&[
                1,
                n as i64,
                (c.hc_count * c.hidden_size) as i64,
            ])?;
            let (input, inject) = self.hyper(&x, "layers.0.attn_hyper_connection", "", head)?;
            let mut started = vec![&input];
            started.extend(inject.iter());
            self.submit(&started, "qwen4::mtp::draft_input")?;
            let history = n - usize::from(head);
            let rows = |a: &MxArray, start: usize, end: usize| {
                if start == 0 && end == n {
                    Ok(a.clone())
                } else {
                    a.slice_axis(1, start as i64, end as i64)
                }
            };
            if history > 0 {
                self.extend_draft_history(&rows(&input, 0, history)?, position, history, cache)?;
            }
            if !head {
                return Ok(None);
            }
            let mut draft = cache.clone();
            self.position_override = Some(position + history);
            let mut x = rows(&x, history, n)?;
            let attn = self.attention(&rows(&input, history, n)?, 0, &mut draft)?;
            let inject = inject.ok_or_else(|| {
                Error::from_reason("Qwen4 MTP attention is missing its injection gate")
            })?;
            x = Self::inject(&x, &attn, &rows(&inject, history, n)?)?;
            let (input, inject) = self.hyper(&x, "layers.0.mlp_hyper_connection", "", true)?;
            let mlp = self.moe(&input, 0)?;
            let inject = inject
                .ok_or_else(|| Error::from_reason("Qwen4 MTP MLP is missing its injection gate"))?;
            x = Self::inject(&x, &mlp, &inject)?;
            let (mixed, _) = self.hyper(&x, "hyper_connection_mixer", "", false)?;
            Ok(Some((x, mixed, draft)))
        })();
        self.scope.clear();
        self.paged = pages;
        self.position_override = old_position;
        result
    }

    /// Append `count` rows of draft K/V and indexer keys from `position`.
    fn extend_draft_history(
        &mut self,
        input: &MxArray,
        position: usize,
        count: usize,
        cache: &mut LayerCache,
    ) -> Result<()> {
        let c = self.config.clone();
        let n = count as i64;
        let k = self
            .linear(input, "layers.0.self_attn.k_proj.weight", "")?
            .reshape(&[1, n, c.num_key_value_heads as i64, c.head_dim as i64])?;
        let k = self
            .norm(&k, "layers.0.self_attn.k_norm.weight", "", c.head_dim, true)?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let k = self.rope_window(&k, position, count)?;
        let v = self
            .linear(input, "layers.0.self_attn.v_proj.weight", "")?
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
            .linear(input)?;
        let tail = cache.index_tail.len();
        let mut raw = cache.index_tail.clone();
        raw.push(ik.reshape(&[n, c.indexer_head_dim as i64])?);
        let raw = MxArray::concatenate_many(raw.iter().collect(), Some(0))?;
        let ratio = c.indexer_compress_ratio;
        let used = (tail + count) / ratio * ratio;
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
            let keys = (0..used / ratio)
                .map(|block| {
                    self.rope(
                        &means.slice_axis(0, block as i64, block as i64 + 1)?,
                        position - tail + block * ratio,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            cache.append_index_keys(MxArray::concatenate_many(keys.iter().collect(), Some(0))?)?;
        }
        cache.index_tail = (used..tail + count)
            .map(|r| raw.slice_axis(0, r as i64, r as i64 + 1))
            .collect::<Result<_>>()?;
        cache.append_draft_kv(&k, &v)?;
        Ok(())
    }
}
