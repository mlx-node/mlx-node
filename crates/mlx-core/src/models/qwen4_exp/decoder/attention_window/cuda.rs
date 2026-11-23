use super::super::*;

impl Decoder {
    /// Cached `[head_dim]` BF16 Q/K RMSNorm weight for fused CUDA `qk_norm_rope_kv_store`
    /// (with `+1.0` baked in for non-GGUF / HF weights).
    pub(in super::super) fn cuda_qk_norm_weight_bf16(
        &mut self,
        hf: &str,
        gg: &str,
    ) -> Result<MxArray> {
        let key = self.key(hf, gg);
        if let Some(w) = self.cuda_qk_norm_bf16.get(&key) {
            return Ok(w.clone());
        }
        let raw = self.dense(hf, gg)?;
        let bf16 = if self.gguf() {
            if raw.dtype()? == DType::BFloat16 {
                raw
            } else {
                raw.astype(DType::BFloat16)?
            }
        } else {
            raw.astype(DType::Float32)?
                .add_scalar(1.0)?
                .astype(DType::BFloat16)?
        };
        MxArray::eval_arrays_with_context(&[&bf16], "qwen4::cuda::qk_norm_bf16")?;
        self.cuda_qk_norm_bf16.insert(key, bf16.clone());
        Ok(bf16)
    }

    /// Fused CUDA QK-Norm + partial RoPE + in-place flat BF16 KV cache store.
    ///
    /// Returns `(q_sdpa [1, Hq, T, D], k_buf [1, Hkv, cap, D], v_buf [1, Hkv, cap, D], kv_len)`
    /// when admitted, or `Ok(None)` when disabled / off-contract (e.g. MRoPE, YaRN, paged, draft).
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn cuda_flat_qk_rope_store(
        &mut self,
        qg: &MxArray,
        k_proj: &MxArray,
        v_proj: &MxArray,
        i: usize,
        cache: &mut LayerCache,
        base: usize,
        t: usize,
        submit_context: &str,
    ) -> Result<Option<(MxArray, MxArray, MxArray, usize)>> {
        if !super::super::super::cuda_qwen4_fast_enabled()
            || self.paged.is_some()
            || cache.draft_rows.is_some()
            || !self.positions.is_empty()
            || self.yarn.is_some()
            || qg.dtype()? != DType::BFloat16
            || k_proj.dtype()? != DType::BFloat16
            || v_proj.dtype()? != DType::BFloat16
        {
            return Ok(None);
        }
        let existing = cache.flat_kv_len()?;
        if existing != base {
            return Ok(None);
        }
        let c = self.config.clone();
        let (nh, kh, hd) = (
            c.num_attention_heads as i64,
            c.num_key_value_heads as i64,
            c.head_dim as i64,
        );
        let p = format!("layers.{i}.self_attn");
        let g = format!("blk.{i}");
        let qw = self.cuda_qk_norm_weight_bf16(
            &format!("{p}.q_norm.weight"),
            &format!("{g}.attn_q_norm.weight"),
        )?;
        let kw = self.cuda_qk_norm_weight_bf16(
            &format!("{p}.k_norm.weight"),
            &format!("{g}.attn_k_norm.weight"),
        )?;
        let (offset, k_buf, v_buf) = cache.ensure_flat_kv_capacity(kh, hd, t, DType::BFloat16)?;
        let q_in = qg.slice_axis(3, 0, hd)?;
        let k_in = k_proj.reshape(&[1, t as i64, kh, hd])?;
        let v_in = v_proj.reshape(&[1, t as i64, kh, hd])?;
        let Some((q_out, new_k, new_v)) = crate::kernels::kv::qk_norm_rope_kv_store(
            &q_in,
            &k_in,
            &v_in,
            &qw,
            &kw,
            &k_buf,
            &v_buf,
            c.rms_norm_eps as f32,
            c.rope_dims() as i32,
            c.rope_theta() as f32,
            1.0,
            offset as i32,
        )?
        else {
            return Ok(None);
        };
        let kv_len = offset + t;
        cache.keys = Some(new_k.clone());
        cache.values = Some(new_v.clone());
        cache.flat_kv_rows = Some(kv_len);
        self.submit_state(&[&new_k, &new_v], submit_context)?;
        let _ = nh;
        Ok(Some((q_out, new_k, new_v, kv_len)))
    }

    /// Fused CUDA dense (`sdpa_causal_few_q1_to_16` / `sdpa_prefill_d256`) or
    /// zero-gather sparse QSA (`qsa_sparse_sdpa_cuda`) over flat BF16 KV buffers.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn cuda_flat_sdpa(
        &mut self,
        q: &MxArray,
        k_buf: &MxArray,
        v_buf: &MxArray,
        kv_len: usize,
        iq: &MxArray,
        i: usize,
        cache: &LayerCache,
        base: usize,
        t: usize,
        sparse: bool,
    ) -> Result<Option<MxArray>> {
        if !super::super::super::cuda_qwen4_fast_enabled()
            || q.dtype()? != DType::BFloat16
            || k_buf.dtype()? != DType::BFloat16
            || v_buf.dtype()? != DType::BFloat16
        {
            return Ok(None);
        }
        let c = self.config.clone();
        if c.head_dim != 256 {
            return Ok(None);
        }
        let scale = (c.head_dim as f32).powf(-0.5);
        if sparse {
            if c.indexer_n_heads != 4
                || c.indexer_head_dim != 128
                || c.indexer_compress_ratio != 4
                || c.indexer_budget != 2048
            {
                return Ok(None);
            }
            let Some(pooled_k) = cache.index_keys.as_ref() else {
                return Ok(None);
            };
            let num_blocks = cache.index_block_count()? as i32;
            let iq_rot = self.qsa_query(i, iq, base, t)?;
            let idx_scale = (c.indexer_head_dim as f32).powf(-0.5);
            return crate::kernels::qwen4::qsa_sparse_sdpa_cuda(
                q,
                k_buf,
                v_buf,
                &iq_rot,
                pooled_k,
                None,
                kv_len as i32,
                num_blocks,
                scale,
                idx_scale,
                0,
            );
        }
        let k_active = if k_buf.shape_at(2)? as usize == kv_len {
            k_buf.clone()
        } else {
            k_buf.slice_axis(2, 0, kv_len as i64)?
        };
        let v_active = if v_buf.shape_at(2)? as usize == kv_len {
            v_buf.clone()
        } else {
            v_buf.slice_axis(2, 0, kv_len as i64)?
        };
        let gqa = (c.num_attention_heads / c.num_key_value_heads) as i64;
        if t <= 16 {
            crate::kernels::attention::sdpa_causal_few_q1_to_16(
                q, &k_active, &v_active, 256, gqa, t as i64, scale,
            )
        } else {
            crate::kernels::attention::sdpa_prefill_d256(
                q, &k_active, &v_active, 256, t as i64, scale,
            )
        }
    }

    /// Batched CUDA flat-KV attention for `verify_layer` (`R in 2..=16`): runs ONE
    /// `qk_norm_rope_kv_store`, ONE indexer block update, and ONE causal dense/sparse
    /// SDPA across all `R` rows while recording exact per-row rollback snapshots in `saved`.
    pub(in super::super) fn cuda_verify_attention_batch(
        &mut self,
        x: &MxArray,
        i: usize,
        cache: &mut LayerCache,
        projections: &AttentionProjections,
        saved: &mut [LayerCache],
    ) -> Result<Option<MxArray>> {
        let t = saved.len();
        if !super::super::super::cuda_qwen4_fast_enabled()
            || self.paged.is_some()
            || cache.draft_rows.is_some()
            || !self.positions.is_empty()
            || self.yarn.is_some()
            || self.position_override.is_some()
            || x.dtype()? != DType::BFloat16
            || !(1..=16).contains(&t)
        {
            return Ok(None);
        }
        let c = self.config.clone();
        if c.head_dim != 256 {
            return Ok(None);
        }
        let base = self.history.len();
        let nh = c.num_attention_heads as i64;
        let hd = c.head_dim as i64;
        let qg = projections.qg.reshape(&[1, t as i64, nh, 2 * hd])?;
        let Some((q, k_buf, v_buf, kv_len)) = self.cuda_flat_qk_rope_store(
            &qg,
            &projections.k,
            &projections.v,
            i,
            cache,
            base,
            t,
            "qwen4::decoder::kv",
        )?
        else {
            return Ok(None);
        };
        let p = format!("layers.{i}.self_attn");
        let g = format!("blk.{i}");
        let id = c.indexer_head_dim as i64;
        let ratio = c.indexer_compress_ratio;
        let old_blocks = cache.index_block_count()?;
        let tail = cache.index_tail.len();
        let mut raw_parts = cache.index_tail.clone();
        raw_parts.push(projections.ik.reshape(&[t as i64, id])?);
        let raw = MxArray::concatenate_many(raw_parts.iter().collect(), Some(0))?;
        let total_new_blocks = (tail + t) / ratio;
        let used = total_new_blocks * ratio;
        if total_new_blocks > 0 {
            let means = raw
                .slice_axis(0, 0, used as i64)?
                .reshape(&[total_new_blocks as i64, ratio as i64, id])?
                .astype(DType::Float32)?
                .mean(Some(&[1]), Some(false))?
                .astype(x.dtype()?)?;
            let norms = self.norm(
                &means,
                &format!("{p}.indexer.k_layernorm.weight"),
                &format!("{g}.indexer.k_norm.weight"),
                c.indexer_head_dim,
                true,
            )?;
            let rotated = self
                .rope_window_positions(
                    &norms.reshape(&[1, 1, total_new_blocks as i64, id])?,
                    (0..total_new_blocks).map(|b| base - tail + b * ratio),
                )?
                .reshape(&[total_new_blocks as i64, id])?;
            let keys = self.append_index_blocks(i, cache, rotated)?;
            self.submit_state(&[&keys], "qwen4::decoder::block")?;
        }
        cache.index_tail = (used..tail + t)
            .map(|r| raw.slice_axis(0, r as i64, r as i64 + 1))
            .collect::<Result<_>>()?;
        for (j, slot) in saved.iter_mut().enumerate() {
            let n_j = tail + j + 1;
            let blocks_j = n_j / ratio;
            let used_j = blocks_j * ratio;
            slot.keys = cache.keys.clone();
            slot.values = cache.values.clone();
            slot.flat_kv_rows = Some(base + j + 1);
            slot.index_keys = cache.index_keys.clone();
            slot.flat_index_blocks = Some(old_blocks + blocks_j);
            slot.index_blocks = cache.index_blocks;
            slot.index_tail = (used_j..n_j)
                .map(|r| raw.slice_axis(0, r as i64, r as i64 + 1))
                .collect::<Result<_>>()?;
        }
        let topk = c.indexer_budget / ratio;
        let sparse = cache.index_block_count()? > topk;
        let out = if let Some(cuda_out) = self.cuda_flat_sdpa(
            &q,
            &k_buf,
            &v_buf,
            kv_len,
            &projections.iq,
            i,
            cache,
            base,
            t,
            sparse,
        )? {
            cuda_out
        } else if sparse {
            let (selected, query_ends) = self.qsa_select(i, cache, &projections.iq, base, t)?;
            let k_active = k_buf.slice_axis(2, 0, kv_len as i64)?;
            let v_active = v_buf.slice_axis(2, 0, kv_len as i64)?;
            math::qsa_attention_flat(
                &q,
                &k_active,
                &v_active,
                &selected,
                &query_ends,
                base,
                (c.head_dim as f64).powf(-0.5),
                ratio,
            )?
        } else {
            let k_active = k_buf.slice_axis(2, 0, kv_len as i64)?;
            let v_active = v_buf.slice_axis(2, 0, kv_len as i64)?;
            crate::array::scaled_dot_product_attention_causal(
                &q,
                &k_active,
                &v_active,
                (c.head_dim as f64).powf(-0.5),
            )?
        };
        let gated = math::attention_output(&out, &qg)?;
        Ok(Some(gated))
    }

    /// Tree-verify CUDA flat-KV attention (`T in 1..=16`):
    /// 1. Fused QK-Norm + tree-depth RoPE (`pre_kv_len + depths[j]`) + flat BF16 KV store
    ///    at `pre_kv_len..pre_kv_len + T`.
    /// 2. Dense (`sdpa_tree_few`) when `pre_index_blocks <= topk`, or zero-gather sparse QSA
    ///    (`qsa_sparse_sdpa_cuda` with `mask_mode = 1` and tree-depth RoPE'd `iq`) when
    ///    `pre_index_blocks > topk`.
    /// 3. Does NOT mutate `index_keys`/`index_tail` yet; returns [`VerifyAttnLayerTape`]
    ///    so `commit_tree_deferred` compacts non-contiguous branches and appends committed
    ///    4-token indexer blocks cleanly.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn cuda_verify_attention_tree(
        &mut self,
        x: &MxArray,
        i: usize,
        cache: &mut LayerCache,
        projections: &AttentionProjections,
        tree: &crate::models::qwen3_5::verify_tree::VerifyTree,
        views: &crate::models::qwen3_5::verify_tree::TreeDeviceViews,
    ) -> Result<(MxArray, VerifyAttnLayerTape)> {
        let t = tree.len();
        if !super::super::super::cuda_qwen4_fast_enabled()
            || self.paged.is_some()
            || cache.draft_rows.is_some()
            || !self.positions.is_empty()
            || self.yarn.is_some()
            || self.position_override.is_some()
            || x.dtype()? != DType::BFloat16
            || !(1..=16).contains(&t)
        {
            return Err(Error::from_reason(
                "Qwen4 tree attention verify requires flat CUDA BF16 (T in 1..=16, no MRoPE/YaRN)",
            ));
        }
        let c = self.config.clone();
        if c.head_dim != 256 {
            return Err(Error::from_reason(format!(
                "Qwen4 tree attention verify expects head_dim=256, got {}",
                c.head_dim
            )));
        }
        let pre_kv_len = cache.flat_kv_len()?;
        let base = self.history.len();
        if pre_kv_len != base {
            return Err(Error::from_reason(format!(
                "Qwen4 tree attention flat_kv_len ({pre_kv_len}) != history.len() ({base})"
            )));
        }
        let (nh, kh, hd) = (
            c.num_attention_heads as i64,
            c.num_key_value_heads as i64,
            c.head_dim as i64,
        );
        let p = format!("layers.{i}.self_attn");
        let g = format!("blk.{i}");
        let qw = self.cuda_qk_norm_weight_bf16(
            &format!("{p}.q_norm.weight"),
            &format!("{g}.attn_q_norm.weight"),
        )?;
        let kw = self.cuda_qk_norm_weight_bf16(
            &format!("{p}.k_norm.weight"),
            &format!("{g}.attn_k_norm.weight"),
        )?;
        let (offset, k_buf, v_buf) = cache.ensure_flat_kv_capacity(kh, hd, t, DType::BFloat16)?;
        debug_assert_eq!(offset, pre_kv_len);
        let qg = projections.qg.reshape(&[1, t as i64, nh, 2 * hd])?;
        let q_in = qg.slice_axis(3, 0, hd)?;
        let k_in = projections.k.reshape(&[1, t as i64, kh, hd])?;
        let v_in = projections.v.reshape(&[1, t as i64, kh, hd])?;
        let (q, new_k, new_v) = crate::kernels::kv::qk_norm_rope_kv_store_tree(
            &q_in,
            &k_in,
            &v_in,
            &qw,
            &kw,
            &k_buf,
            &v_buf,
            &views.depths_i32,
            c.rms_norm_eps as f32,
            c.rope_dims() as i32,
            c.rope_theta() as f32,
            1.0,
            offset as i32,
        )?
        .ok_or_else(|| Error::from_reason("Qwen4 CUDA qk_norm_rope_kv_store_tree failed"))?;
        let kv_len = pre_kv_len + t;
        cache.keys = Some(new_k.clone());
        cache.values = Some(new_v.clone());
        cache.flat_kv_rows = Some(kv_len);

        let id = c.indexer_head_dim as i64;
        let ratio = c.indexer_compress_ratio;
        let pre_index_blocks = cache.index_block_count()?;
        let pre_index_tail = cache.index_tail.clone();
        let ik_rows = projections.ik.reshape(&[t as i64, id])?;

        let topk = c.indexer_budget / ratio;
        let scale = (c.head_dim as f32).powf(-0.5);
        let gqa = (c.num_attention_heads / c.num_key_value_heads) as i64;
        let out = if pre_index_blocks > topk {
            if c.indexer_n_heads != 4
                || c.indexer_head_dim != 128
                || c.indexer_compress_ratio != 4
                || c.indexer_budget != 2048
            {
                return Err(Error::from_reason(
                    "Qwen4 sparse tree QSA requires ih=4, id=128, ratio=4, budget=2048",
                ));
            }
            let pooled_k = cache
                .index_keys
                .as_ref()
                .ok_or_else(|| Error::from_reason("Qwen4 sparse tree QSA missing pooled_k"))?;
            let iq_rot = self.qsa_query_positions(
                i,
                &projections.iq,
                t,
                tree.depth.iter().map(|&d| pre_kv_len + d as usize),
            )?;
            let idx_scale = (c.indexer_head_dim as f32).powf(-0.5);
            crate::kernels::qwen4::qsa_sparse_sdpa_cuda(
                &q,
                &new_k,
                &new_v,
                &iq_rot,
                pooled_k,
                Some(&views.mask_u32),
                kv_len as i32,
                pre_index_blocks as i32,
                scale,
                idx_scale,
                1,
            )?
            .ok_or_else(|| Error::from_reason("Qwen4 CUDA qsa_sparse_sdpa_cuda(tree) failed"))?
        } else {
            let k_active = if new_k.shape_at(2)? as usize == kv_len {
                new_k.clone()
            } else {
                new_k.slice_axis(2, 0, kv_len as i64)?
            };
            let v_active = if new_v.shape_at(2)? as usize == kv_len {
                new_v.clone()
            } else {
                new_v.slice_axis(2, 0, kv_len as i64)?
            };
            crate::kernels::attention::sdpa_tree_few(
                &q,
                &k_active,
                &v_active,
                &views.mask_u32,
                pre_kv_len as i32,
                256,
                gqa,
                t as i64,
                scale,
            )?
            .ok_or_else(|| Error::from_reason("Qwen4 CUDA sdpa_tree_few failed"))?
        };
        let branch = math::attention_output(&out, &qg)?;
        Ok((
            branch,
            VerifyAttnLayerTape {
                pre_kv_len,
                pre_index_tail,
                ik_rows,
            },
        ))
    }

    /// Commit `keep` accepted nodes on full-attention layer `i` after tree verify:
    /// trims flat KV in zero-copy when `is_contiguous_prefix`, otherwise compacts
    /// accepted KV rows via `kernels::kv::kv_compact_rows` and reassigns handles;
    /// then updates `index_keys` and `index_tail` from the `keep` accepted `ik` rows.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn commit_attn_tree_layer(
        &mut self,
        i: usize,
        cache: &mut LayerCache,
        tape: VerifyAttnLayerTape,
        keep: usize,
        is_contiguous_prefix: bool,
        accepted_i32: Option<&MxArray>,
    ) -> Result<()> {
        if is_contiguous_prefix {
            cache.flat_kv_rows = Some(tape.pre_kv_len + keep);
        } else {
            let idx = accepted_i32.ok_or_else(|| {
                Error::from_reason("Qwen4 non-contiguous tree commit missing accepted_i32")
            })?;
            let (Some(k_buf), Some(v_buf)) = (&cache.keys, &cache.values) else {
                return Err(Error::from_reason(
                    "Qwen4 non-contiguous tree commit missing KV buffers",
                ));
            };
            let (new_k, new_v) =
                crate::kernels::kv::kv_compact_rows(k_buf, v_buf, tape.pre_kv_len as i32, idx)?
                    .ok_or_else(|| Error::from_reason("Qwen4 CUDA kv_compact_rows failed"))?;
            cache.keys = Some(new_k);
            cache.values = Some(new_v);
            cache.flat_kv_rows = Some(tape.pre_kv_len + keep);
        }

        let c = self.config.clone();
        let ratio = c.indexer_compress_ratio;
        let id = c.indexer_head_dim as i64;
        let kept_ik = if is_contiguous_prefix {
            tape.ik_rows.slice_axis(0, 0, keep as i64)?
        } else {
            let idx = accepted_i32.ok_or_else(|| {
                Error::from_reason("Qwen4 non-contiguous tree commit missing accepted_i32")
            })?;
            tape.ik_rows.take(idx, 0)?
        };
        let tail = tape.pre_index_tail.len();
        let total_new_blocks = (tail + keep) / ratio;
        if total_new_blocks == 0 {
            let mut next_tail = tape.pre_index_tail;
            for r in 0..keep {
                next_tail.push(kept_ik.slice_axis(0, r as i64, r as i64 + 1)?);
            }
            cache.index_tail = next_tail;
            return Ok(());
        }
        let mut raw_parts = tape.pre_index_tail;
        raw_parts.push(kept_ik);
        let raw = MxArray::concatenate_many(raw_parts.iter().collect(), Some(0))?;
        let used = total_new_blocks * ratio;
        let p = format!("layers.{i}.self_attn");
        let g = format!("blk.{i}");
        let means = raw
            .slice_axis(0, 0, used as i64)?
            .reshape(&[total_new_blocks as i64, ratio as i64, id])?
            .astype(DType::Float32)?
            .mean(Some(&[1]), Some(false))?
            .astype(raw.dtype()?)?;
        let norms = self.norm(
            &means,
            &format!("{p}.indexer.k_layernorm.weight"),
            &format!("{g}.indexer.k_norm.weight"),
            c.indexer_head_dim,
            true,
        )?;
        let rotated = if self.scope.is_empty() && self.positions.is_empty() && self.yarn.is_none() {
            if total_new_blocks == 1 {
                self.rope(&norms, tape.pre_kv_len - tail)?
            } else {
                let mut rows = Vec::with_capacity(total_new_blocks);
                for b in 0..total_new_blocks {
                    let row = norms.slice_axis(0, b as i64, b as i64 + 1)?;
                    rows.push(self.rope(&row, tape.pre_kv_len - tail + b * ratio)?);
                }
                MxArray::concatenate_many(rows.iter().collect(), Some(0))?
            }
        } else {
            self.rope_window_positions(
                &norms.reshape(&[1, 1, total_new_blocks as i64, id])?,
                (0..total_new_blocks).map(|b| tape.pre_kv_len - tail + b * ratio),
            )?
            .reshape(&[total_new_blocks as i64, id])?
        };
        let _ = self.append_index_blocks(i, cache, rotated)?;
        cache.index_tail = (used..tail + keep)
            .map(|r| raw.slice_axis(0, r as i64, r as i64 + 1))
            .collect::<Result<_>>()?;
        Ok(())
    }
}
