//! Window-wide causal attention. Pages remain the persistent storage; only
//! the current window and a bounded view of its prefix feed SDPA.
use super::*;
use crate::models::qwen4_exp::runtime_flags;

impl Decoder {
    pub(super) fn rope_window(&self, x: &MxArray, start: usize, count: usize) -> Result<MxArray> {
        self.rope_window_positions(x, start..start + count)
    }

    fn use_batch_rotary(&self) -> bool {
        self.batch_rotary && !runtime_flags::is_zero(c"MLX_QWEN4_PREFILL_ROTARY")
    }

    fn rope_window_positions(
        &self,
        x: &MxArray,
        offsets: impl Iterator<Item = usize>,
    ) -> Result<MxArray> {
        let (positions, sections, interleaved) = self.rotary_window_parameters(offsets);
        if self.use_batch_rotary() {
            return self.rotary_tables.borrow_mut().apply(
                x,
                positions,
                self.config.rope_dims(),
                self.config.rope_theta(),
                sections,
                interleaved,
            );
        }
        math::mrope_window(
            x,
            &positions,
            self.config.rope_dims(),
            self.config.rope_theta(),
            sections,
            interleaved,
        )
    }

    fn rotary_window_parameters(
        &self,
        offsets: impl Iterator<Item = usize>,
    ) -> (Vec<[i64; 3]>, [usize; 3], bool) {
        let media = self.scope.is_empty() && !self.positions.is_empty();
        let positions: Vec<_> = offsets
            .map(|pos| {
                if media {
                    self.positions
                        .get(pos)
                        .copied()
                        .unwrap_or([pos as i64 + self.rope_delta; 3])
                } else {
                    [pos as i64; 3]
                }
            })
            .collect();
        let v = &self.config.rope_parameters;
        let sections = std::array::from_fn(|i| {
            if media {
                v["mrope_section"][i].as_u64().unwrap_or(0) as usize
            } else {
                0
            }
        });
        (
            positions,
            sections,
            v["mrope_interleaved"].as_bool().unwrap_or(true),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_norm_rotary(
        &mut self,
        x: &MxArray,
        hf: &str,
        gg: &str,
        start: usize,
        count: usize,
        window: bool,
    ) -> Result<MxArray> {
        if self.gguf()
            && self.config.head_dim == 256
            && x.dtype()? == DType::BFloat16
            && runtime_flags::is_one(c"MLX_QWEN4_ATTENTION_NORM_ROTARY")
            && !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_POINTWISE")
        {
            let weight = self.dense(hf, gg)?;
            let (positions, sections, interleaved) =
                self.rotary_window_parameters(start..start + count);
            if let Some(out) = self.rotary_tables.borrow_mut().apply_normalized(
                x,
                &weight,
                positions,
                self.config.rope_dims(),
                self.config.rope_theta(),
                sections,
                interleaved,
                self.config.rms_norm_eps,
            )? {
                return Ok(out);
            }
        }
        let normalized = self.norm(x, hf, gg, self.config.head_dim, true)?;
        if window {
            self.rope_window(&normalized, start, count)
        } else {
            self.rope(&normalized, start)
        }
    }

    pub(super) fn attention_window(
        &mut self,
        x: &MxArray,
        i: usize,
        cache: &mut LayerCache,
        projections: &AttentionProjections,
    ) -> Result<Option<MxArray>> {
        let c = self.config.clone();
        let base = self.history.len();
        let t = x.shape()?[1] as usize;
        if runtime_flags::is_zero(c"MLX_QWEN4_ATTENTION_WINDOW")
            // F32 fixtures retain singleton SDPA accumulation (wide NAX uses TF32).
            || x.dtype()? == DType::Float32
            || base + t > c.indexer_budget || self.verification.is_some()
        {
            return Ok(None);
        }
        let (nh, kh, hd) = (
            c.num_attention_heads as i64,
            c.num_key_value_heads as i64,
            c.head_dim as i64,
        );
        let p = format!("layers.{i}.self_attn");
        let g = format!("blk.{i}");
        let qg = projections.qg.reshape(&[1, t as i64, nh, 2 * hd])?;
        let q = self.attention_norm_rotary(
            &qg.slice_axis(3, 0, hd)?.transpose(Some(&[0, 2, 1, 3]))?,
            &format!("{p}.q_norm.weight"),
            &format!("{g}.attn_q_norm.weight"),
            base,
            t,
            true,
        )?;
        let k = self.attention_norm_rotary(
            &projections
                .k
                .reshape(&[1, t as i64, kh, hd])?
                .transpose(Some(&[0, 2, 1, 3]))?,
            &format!("{p}.k_norm.weight"),
            &format!("{g}.attn_k_norm.weight"),
            base,
            t,
            true,
        )?;
        let v = projections
            .v
            .reshape(&[1, t as i64, kh, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        // Keep the same complete-block compression and its physical first
        // position, including blocks split across windows or media positions.
        let id = c.indexer_head_dim as i64;
        let tail = cache.index_tail.len();
        let mut raw = cache.index_tail.clone();
        raw.push(projections.ik.reshape(&[t as i64, id])?);
        let raw = MxArray::concatenate_many(raw.iter().collect(), Some(0))?;
        let ratio = c.indexer_compress_ratio;
        let blocks = (tail + t) / ratio;
        let used = blocks * ratio;
        if blocks > 0 {
            let means = raw
                .slice_axis(0, 0, used as i64)?
                .reshape(&[blocks as i64, ratio as i64, id])?
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
            let rotated = if self.use_batch_rotary() {
                Some(
                    self.rope_window_positions(
                        &norms.reshape(&[1, 1, blocks as i64, id])?,
                        (0..blocks).map(|b| base - tail + b * ratio),
                    )?
                    .reshape(&[blocks as i64, id])?,
                )
            } else {
                None
            };
            for b in 0..blocks {
                if let Some(keys) = &rotated {
                    cache
                        .index_blocks
                        .push(keys.slice_axis(0, b as i64, b as i64 + 1)?);
                    continue;
                }
                let key = self.rope(
                    &norms.slice_axis(0, b as i64, b as i64 + 1)?,
                    base - tail + b * ratio,
                )?;
                cache.index_blocks.push(key);
            }
        }
        cache.index_tail = (used..tail + t)
            .map(|r| raw.slice_axis(0, r as i64, r as i64 + 1))
            .collect::<Result<_>>()?;
        let (keys, values) = if let Some(adapter) = &self.paged {
            cache.keys = Some(k.clone());
            cache.values = Some(v.clone());
            if base == 0 {
                (k, v)
            } else {
                let layer = (0..i).filter(|&l| !c.linear(l)).count() as u32;
                let ids: Vec<_> = (0..base as i32).collect();
                let (pk, pv) = super::super::paged::gather_selected(
                    adapter,
                    layer,
                    &ids,
                    c.num_key_value_heads,
                    c.head_dim,
                )?;
                (
                    MxArray::concatenate(&pk.astype(x.dtype()?)?, &k, 2)?,
                    MxArray::concatenate(&pv.astype(x.dtype()?)?, &v, 2)?,
                )
            }
        } else {
            let keys = match &cache.keys {
                Some(old) => MxArray::concatenate(old, &k, 2)?,
                None => k,
            };
            let values = match &cache.values {
                Some(old) => MxArray::concatenate(old, &v, 2)?,
                None => v,
            };
            cache.keys = Some(keys.clone());
            cache.values = Some(values.clone());
            (keys, values)
        };
        let out = crate::array::scaled_dot_product_attention_causal(
            &q,
            &keys,
            &values,
            (c.head_dim as f64).powf(-0.5),
        )?;
        Ok(Some(math::attention_output(&out, &qg)?))
    }
}
