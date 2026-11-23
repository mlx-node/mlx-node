use super::*;

impl Decoder {
    /// `[multipliers, head sizes, head offsets]` of PLE layer `i`.
    fn ple_constants(&self, i: usize) -> Result<[Vec<u64>; 3]> {
        let p = format!("layers.{i}.ple");
        let constants = |name: &str| -> Result<Vec<u64>> {
            if self.gguf() {
                super::super::gguf::integers(
                    &self.weights.metadata,
                    &format!("qwen4exp.ple.{name}"),
                )
            } else {
                self.weights.integers(&format!(
                    "{p}.ple_embedding.{}",
                    match name {
                        "head_offsets" => "ngram_heads_offsets",
                        "head_vocab_sizes" => "ngram_heads_vocab_sizes",
                        _ => name,
                    }
                ))
            }
        };
        Ok([
            constants("layer_multipliers")?,
            constants("head_vocab_sizes")?,
            constants("head_offsets")?,
        ])
    }

    /// Hash ids of `tokens` continuing the current history, in gather order.
    fn ple_hash_ids(&self, i: usize, tokens: &[u32]) -> Result<Vec<u64>> {
        self.ple_hash_ids_after(i, &self.history, tokens)
    }

    /// The hash ids of `tokens` following `history` (the ngram context is
    /// its tail).
    fn ple_hash_ids_after(&self, i: usize, history: &[u32], tokens: &[u32]) -> Result<Vec<u64>> {
        let c = &self.config;
        let [multipliers, sizes, offsets] = self.ple_constants(i)?;
        let mut previous = history[history.len().saturating_sub(c.ngram_size - 1)..].to_vec();
        let mut ids = Vec::new();
        for &token in tokens {
            self.check_cancelled()?;
            ids.extend(math::hash_ids(
                token,
                &previous,
                c.eos_token_id,
                c.ngram_size,
                c.heads_per_ngram,
                &multipliers,
                &sizes,
                &offsets,
            )?);
            previous.push(token);
        }
        Ok(ids)
    }

    /// Tree-aware n-gram hash ids for `tokens` with tree parents `parent`
    /// (`parent[0] = -1`, `0 <= parent[j] < j`) following committed `history`.
    /// Each node walks `ngram_size - 1` ancestors back through `parent` (and
    /// into `history` past root) so branching paths compute exact AR hashes.
    fn ple_hash_ids_tree(
        &self,
        i: usize,
        history: &[u32],
        tokens: &[u32],
        parent: &[i32],
    ) -> Result<Vec<u64>> {
        let c = &self.config;
        let [multipliers, sizes, offsets] = self.ple_constants(i)?;
        let need = c.ngram_size.saturating_sub(1);
        let mut ids = Vec::new();
        let mut previous = Vec::with_capacity(need);
        for (j, &token) in tokens.iter().enumerate() {
            self.check_cancelled()?;
            previous.clear();
            let mut cur = parent[j];
            for _ in 0..need {
                if cur >= 0 {
                    previous.push(tokens[cur as usize]);
                    cur = parent[cur as usize];
                } else {
                    let back = (-cur) as usize;
                    if back <= history.len() {
                        previous.push(history[history.len() - back]);
                    } else {
                        break;
                    }
                    cur -= 1;
                }
            }
            previous.reverse();
            ids.extend(math::hash_ids(
                token,
                &previous,
                c.eos_token_id,
                c.ngram_size,
                c.heads_per_ngram,
                &multipliers,
                &sizes,
                &offsets,
            )?);
        }
        Ok(ids)
    }

    /// Prefetch tree-aware PLE rows across every PLE layer before Layer 0 runs.
    pub(super) fn prefetch_tree_ple_rows(&mut self, tokens: &[u32], parent: &[i32]) -> Result<()> {
        let history = self.history.clone();
        let mut keys = Vec::new();
        for &layer in &self.config.ple_layer_ids {
            let Some(i) = layer.checked_sub(1) else {
                continue;
            };
            let ids = self.ple_hash_ids_tree(i, &history, tokens, parent)?;
            keys.extend(self.ple_row_keys(i, &ids)?);
        }
        self.weights.prefetch_rows(&keys)
    }

    /// The table row behind each hash id: the one GGUF table, or the
    /// safetensors shard that holds it.
    fn ple_row_keys(&self, i: usize, ids: &[u64]) -> Result<Vec<(String, usize)>> {
        if self.gguf() {
            return Ok(ids
                .iter()
                .map(|&id| ("per_layer_token_embd.weight".to_string(), id as usize))
                .collect());
        }
        let p = format!("layers.{i}.ple");
        let shards = (0..self.config.split_ngram_parts)
            .map(|shard| {
                let key = format!("{p}.ple_embedding.ngram_embedding.shard_{shard}.weight");
                let count = self.weights.descriptor(&key)?.rows()?;
                Ok((key, count))
            })
            .collect::<Result<Vec<_>>>()?;
        ids.iter()
            .map(|&id| {
                let mut local = id as usize;
                for (key, count) in &shards {
                    if local < *count {
                        return Ok((key.clone(), local));
                    }
                    local -= count;
                }
                Err(Error::from_reason("PLE row outside checkpoint vocabulary"))
            })
            .collect()
    }

    /// Hand the reader threads the PLE rows `tokens` will need at their PLE
    /// layer, before the layers ahead of it are encoded. The embedding row
    /// is needed at once, so the model thread reads it itself.
    pub(super) fn prefetch_ple_rows(&mut self, tokens: &[u32]) -> Result<()> {
        let keys = self.ple_row_keys_after(&self.history.clone(), tokens)?;
        self.weights.prefetch_rows(&keys)
    }

    /// The PLE rows of `tokens` following `history`, over every PLE layer.
    pub(super) fn ple_row_keys_after(
        &self,
        history: &[u32],
        tokens: &[u32],
    ) -> Result<Vec<(String, usize)>> {
        let mut keys = Vec::new();
        for &layer in &self.config.ple_layer_ids {
            let Some(i) = layer.checked_sub(1) else {
                continue;
            };
            let ids = self.ple_hash_ids_after(i, history, tokens)?;
            keys.extend(self.ple_row_keys(i, &ids)?);
        }
        Ok(keys)
    }

    /// A prefill window starts: request this window's rows, then, queued
    /// behind them, the rows of the window after it, so the reader threads
    /// fetch those while this window's layers run instead of the model
    /// thread reading them one `pread` at a time at the next window's start
    /// (1,024 embedding rows and up to 16k PLE rows per window). Two
    /// requests, not one sorted batch: the rows this window needs now must
    /// not queue behind rows needed a second later. A window given its
    /// embeddings (media) has no embedding rows to read.
    pub(super) fn prefetch_window_rows(
        &mut self,
        tokens: &[u32],
        next: &[u32],
        embed: bool,
    ) -> Result<()> {
        let embedding = self.key("embed_tokens.weight", "token_embd.weight");
        if embed {
            let embed_keys: Vec<_> = tokens
                .iter()
                .map(|&t| (embedding.clone(), t as usize))
                .collect();
            self.weights.prefetch_rows(&embed_keys)?;
        }
        let history = self.history.clone();
        let ple_keys = self.ple_row_keys_after(&history, tokens)?;
        self.weights.prefetch_rows(&ple_keys)?;
        if next.is_empty() {
            return Ok(());
        }
        if embed {
            let next_embed: Vec<_> = next
                .iter()
                .map(|&t| (embedding.clone(), t as usize))
                .collect();
            self.weights.prefetch_rows(&next_embed)?;
        }
        let mut through = history;
        through.extend_from_slice(tokens);
        let next_ple = self.ple_row_keys_after(&through, next)?;
        self.weights.prefetch_rows(&next_ple)
    }

    /// Compute PLE delta and add to `x` (`x + ple`), fusing the residual add into
    /// `ple_gate_conv_cuda` on CUDA when eligible.
    pub(in super::super) fn ple_apply(
        &mut self,
        x: &MxArray,
        tokens: &[u32],
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        let embeddings = self.ple_embeddings(tokens, i, x.dtype()?)?;
        let (key, value) = self.ple_projections(&embeddings, i)?;
        self.ple_mix_with_residual(x, &key, &value, i, cache, true)
    }

    /// The `[1, T, ple_embed_dim]` embeddings of `tokens` following the history.
    pub(super) fn ple_embeddings(
        &mut self,
        tokens: &[u32],
        i: usize,
        dtype: DType,
    ) -> Result<MxArray> {
        let ids = self.ple_hash_ids(i, tokens)?;
        self.ple_embeddings_for_ids(&ids, tokens.len() as i64, i, dtype)
    }

    /// Tree-aware `[1, T, ple_embed_dim]` PLE embeddings for `tokens`/`parent`.
    pub(super) fn ple_embeddings_tree(
        &mut self,
        tokens: &[u32],
        parent: &[i32],
        i: usize,
        dtype: DType,
    ) -> Result<MxArray> {
        let ids = self.ple_hash_ids_tree(i, &self.history, tokens, parent)?;
        self.ple_embeddings_for_ids(&ids, tokens.len() as i64, i, dtype)
    }

    fn ple_embeddings_for_ids(
        &mut self,
        ids: &[u64],
        t: i64,
        i: usize,
        dtype: DType,
    ) -> Result<MxArray> {
        let c = self.config.clone();
        let mut groups: std::collections::BTreeMap<String, (Vec<usize>, Vec<usize>)> =
            Default::default();
        for (position, (name, row)) in self.ple_row_keys(i, ids)?.into_iter().enumerate() {
            let (rows, positions) = groups.entry(name).or_default();
            rows.push(row);
            positions.push(position);
        }
        let emb = if groups.len() == 1
            && let Some((name, (rows, _))) = groups.pop_first()
        {
            self.weights.lookup_rows(&name, &rows)?
        } else {
            let mut rows = vec![None; ids.len()];
            for (name, (indices, positions)) in groups {
                let bank = self.weights.lookup_rows(&name, &indices)?;
                for (row, position) in positions.into_iter().enumerate() {
                    rows[position] = Some(bank.slice_axis(0, row as i64, row as i64 + 1)?);
                }
            }
            let rows = rows
                .iter()
                .map(|row| {
                    row.as_ref().ok_or_else(|| {
                        Error::from_reason("Qwen4 PLE lookup did not populate every embedding row")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            MxArray::concatenate_many(rows, Some(0))?
        };
        emb.reshape(&[1, t, c.ple_embed_dim as i64])?.astype(dtype)
    }

    /// The PLE key and value projections of a `[1, T, ple_embed_dim]` window.
    pub(super) fn ple_projections(
        &mut self,
        emb: &MxArray,
        i: usize,
    ) -> Result<(MxArray, MxArray)> {
        let p = format!("layers.{i}.ple");
        let g = format!("blk.{i}");
        let key_name = self.key(
            &format!("{p}.key_proj.weight"),
            &format!("{g}.ple_key.weight"),
        );
        let value_name = self.key(
            &format!("{p}.value_proj.weight"),
            &format!("{g}.ple_value.weight"),
        );
        if super::super::cuda_qwen4_fast_enabled() && self.verification.is_none() {
            // On CUDA prefill (`M >= 17`), route directly through the batched cuBLAS
            // dense prefill path instead of slicing into 64× M=16 `affine_q8_few` calls.
            let key = self.weights.linear(&key_name, emb)?;
            let value = self.weights.linear(&value_name, emb)?;
            return Ok((key, value));
        }
        let key = self.weights.linear_ple_window(&key_name, emb)?;
        let value = self.weights.linear_ple_window(&value_name, emb)?;
        Ok((key, value))
    }

    /// The PLE gate and convolution over `x` (`[1, T, 4H]`) given the
    /// window's key and value projections.
    pub(super) fn ple_mix(
        &mut self,
        x: &MxArray,
        key: &MxArray,
        value: &MxArray,
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        self.ple_mix_with_residual(x, key, value, i, cache, false)
    }

    pub(super) fn ple_mix_with_residual(
        &mut self,
        x: &MxArray,
        key: &MxArray,
        value: &MxArray,
        i: usize,
        cache: &mut LayerCache,
        fuse_residual: bool,
    ) -> Result<MxArray> {
        let c = self.config.clone();
        let t = x.shape_at(1)?;
        let p = format!("layers.{i}.ple");
        let g = format!("blk.{i}");
        if super::super::cuda_qwen4_fast_enabled()
            && c.hidden_size == 2560
            && c.hc_count == 4
            && c.ple_conv_kernel_size == 4
            && c.ngram_size == 3
            && x.dtype()? == DType::BFloat16
            && key.dtype()? == DType::BFloat16
            && value.dtype()? == DType::BFloat16
        {
            let norm_key_w = self.dense(
                &format!("{p}.norm_key.weight"),
                &format!("{g}.ple_norm_key.weight"),
            )?;
            let norm_query_w = self.dense(
                &format!("{p}.norm_query.weight"),
                &format!("{g}.ple_norm_query.weight"),
            )?;
            let norm_conv_w = self.dense(
                &format!("{p}.norm_conv.weight"),
                &format!("{g}.ple_norm_conv.weight"),
            )?;
            let conv_w = self.dense(
                &format!("{p}.conv1d.weight"),
                &format!("{g}.ple_conv1d.weight"),
            )?;
            let weight_offset = if self.gguf() { 0.0 } else { 1.0 };
            if let Some((out, next_hist)) = crate::kernels::qwen4::ple_gate_conv_cuda(
                x,
                key,
                value,
                &norm_key_w,
                &norm_query_w,
                &norm_conv_w,
                &conv_w,
                cache.ple_conv.as_ref(),
                c.rms_norm_eps as f32,
                weight_offset,
                fuse_residual,
            )? {
                cache.ple_conv = Some(next_hist);
                return Ok(out);
            }
        }
        let key = self
            .norm(
                key,
                &format!("{p}.norm_key.weight"),
                &format!("{g}.ple_norm_key.weight"),
                c.hidden_size,
                true,
            )?
            .reshape(&[1, t, c.hc_count as i64, c.hidden_size as i64])?;
        let query = self
            .norm(
                x,
                &format!("{p}.norm_query.weight"),
                &format!("{g}.ple_norm_query.weight"),
                c.hidden_size,
                true,
            )?
            .reshape(&[1, t, c.hc_count as i64, c.hidden_size as i64])?;
        let score = key
            .mul(&query)?
            .sum(Some(&[-1]), Some(true))?
            .div_scalar((c.hidden_size as f64).sqrt())?;
        let score = score
            .sign()?
            .mul(&score.abs()?.clip(Some(1e-6), None)?.sqrt()?)?;
        let value = Activations::sigmoid(&score)?
            .mul(&value.expand_dims(-2)?)?
            .reshape(&x.shape()?)?;
        let normalized = self.norm(
            &value,
            &format!("{p}.norm_conv.weight"),
            &format!("{g}.ple_norm_conv.weight"),
            c.hidden_size,
            true,
        )?;
        let weight = self.dense(
            &format!("{p}.conv1d.weight"),
            &format!("{g}.ple_conv1d.weight"),
        )?;
        let convolved = math::conv_window(
            &normalized,
            &weight,
            &mut cache.ple_conv,
            c.ple_conv_kernel_size,
            c.ngram_size,
        )?;
        let ple = value.add(&convolved)?;
        if fuse_residual { x.add(&ple) } else { Ok(ple) }
    }

    /// Tree-aware Layer-1 PLE gate + dilated depthwise conv (`k=4, d=3`) with
    /// fused residual add (`x + ple`). Returns `(out_x [1, T, 10240], conv_in [T, 10240])`
    /// without mutating `cache.ple_conv` so `commit_tree_deferred` can rebuild
    /// exact `[9, 10240]` history along any accepted branch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ple_mix_tree_cuda(
        &mut self,
        x: &MxArray,
        key: &MxArray,
        value: &MxArray,
        i: usize,
        pre_ple_conv: Option<&MxArray>,
        conv_taps: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let c = self.config.clone();
        if !super::super::cuda_qwen4_fast_enabled()
            || c.hidden_size != 2560
            || c.hc_count != 4
            || c.ple_conv_kernel_size != 4
            || c.ngram_size != 3
            || x.dtype()? != DType::BFloat16
            || key.dtype()? != DType::BFloat16
            || value.dtype()? != DType::BFloat16
        {
            return Err(Error::from_reason(
                "Qwen4 tree PLE requires flat CUDA BF16 (H=2560, HC=4, k=4, ngram=3)",
            ));
        }
        let p = format!("layers.{i}.ple");
        let g = format!("blk.{i}");
        let norm_key_w = self.dense(
            &format!("{p}.norm_key.weight"),
            &format!("{g}.ple_norm_key.weight"),
        )?;
        let norm_query_w = self.dense(
            &format!("{p}.norm_query.weight"),
            &format!("{g}.ple_norm_query.weight"),
        )?;
        let norm_conv_w = self.dense(
            &format!("{p}.norm_conv.weight"),
            &format!("{g}.ple_norm_conv.weight"),
        )?;
        let conv_w = self.dense(
            &format!("{p}.conv1d.weight"),
            &format!("{g}.ple_conv1d.weight"),
        )?;
        let weight_offset = if self.gguf() { 0.0 } else { 1.0 };
        crate::kernels::qwen4::ple_gate_conv_tree_cuda(
            x,
            key,
            value,
            &norm_key_w,
            &norm_query_w,
            &norm_conv_w,
            &conv_w,
            pre_ple_conv,
            Some(conv_taps),
            c.rms_norm_eps as f32,
            weight_offset,
            true,
        )?
        .ok_or_else(|| Error::from_reason("Qwen4 CUDA ple_gate_conv_tree_cuda failed"))
    }
}
