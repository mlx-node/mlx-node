use super::*;

pub(super) struct AttentionProjections {
    pub(super) qg: MxArray,
    pub(super) k: MxArray,
    pub(super) v: MxArray,
    pub(super) iq: MxArray,
    pub(super) ik: MxArray,
}

pub(super) struct GdnProjections {
    pub(super) qkv: MxArray,
    pub(super) z: MxArray,
    pub(super) a: MxArray,
    pub(super) b: MxArray,
}
impl GdnProjections {
    pub(super) fn row(&self, row: usize) -> Result<Self> {
        let slice = |a: &MxArray| a.slice_axis(1, row as i64, row as i64 + 1);
        Ok(Self {
            qkv: slice(&self.qkv)?,
            z: slice(&self.z)?,
            a: slice(&self.a)?,
            b: slice(&self.b)?,
        })
    }
}
impl AttentionProjections {
    pub(super) fn row(&self, row: usize) -> Result<Self> {
        let slice = |a: &MxArray| a.slice_axis(1, row as i64, row as i64 + 1);
        Ok(Self {
            qg: slice(&self.qg)?,
            k: slice(&self.k)?,
            v: slice(&self.v)?,
            iq: slice(&self.iq)?,
            ik: slice(&self.ik)?,
        })
    }
}

const DRAFT_HISTORY_STEP: usize = 256;
const DRAFT_HEAD_ROWS: usize = 3;
/// Preallocation growth quantum (tokens) for non-paged flat BF16 KV buffers.
const FLAT_KV_STEP: usize = 256;
/// Preallocation growth quantum (blocks) for non-paged flat indexer keys.
const FLAT_INDEX_STEP: usize = 64;

impl LayerCache {
    pub(super) fn index_block_count(&self) -> Result<usize> {
        if let Some(blocks) = self.flat_index_blocks {
            return Ok(blocks);
        }
        match &self.index_keys {
            Some(keys) => Ok(keys.shape_at(0)? as usize),
            None => Ok(self.index_blocks),
        }
    }
    /// Active `[blocks, indexer_head_dim]` view of non-paged `index_keys`.
    pub(in super::super) fn active_index_keys(&self) -> Result<Option<MxArray>> {
        let Some(keys) = &self.index_keys else {
            return Ok(None);
        };
        if let Some(blocks) = self.flat_index_blocks {
            if blocks == 0 {
                return Ok(None);
            }
            return Ok(Some(keys.slice_axis(0, 0, blocks as i64)?));
        }
        Ok(Some(keys.clone()))
    }
    /// Append complete indexer block keys (`[n, indexer_head_dim]`) into a
    /// preallocated growable buffer via `slice_assign_axis_inplace`, avoiding
    /// O(T^2) concatenations during decode. Returns the backing buffer.
    pub(super) fn append_index_keys(&mut self, keys: MxArray) -> Result<&MxArray> {
        let rows = self.index_block_count()?;
        let add = keys.shape_at(0)? as usize;
        let end = rows + add;
        let capacity = match &self.index_keys {
            Some(existing) => existing.shape_at(0)? as usize,
            None => 0,
        };
        if end > capacity {
            let target = end.next_multiple_of(FLAT_INDEX_STEP);
            let dim = keys.shape_at(1)?;
            let fresh = MxArray::zeros(&[(target - rows) as i64, dim], Some(keys.dtype()?))?;
            let grown = match &self.index_keys {
                Some(old) if rows > 0 => {
                    MxArray::concatenate(&old.slice_axis(0, 0, rows as i64)?, &fresh, 0)?
                }
                _ => fresh,
            };
            self.index_keys = Some(grown);
        }
        let buf = self
            .index_keys
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen4 flat indexer cache lost its buffer"))?;
        buf.slice_assign_axis_inplace(0, rows as i64, end as i64, &keys)?;
        self.flat_index_blocks = Some(end);
        Ok(buf)
    }
    /// Active token rows stored in non-paged flat `keys`/`values`.
    pub(in super::super) fn flat_kv_len(&self) -> Result<usize> {
        if let Some(rows) = self.flat_kv_rows {
            return Ok(rows);
        }
        match &self.keys {
            Some(keys) => Ok(keys.shape_at(2)? as usize),
            None => Ok(0),
        }
    }
    /// Append `k`/`v` (`[1, kv_heads, t, head_dim]`) into preallocated growable
    /// flat buffers via `slice_assign_axis_inplace` (zero O(T^2) concat copies).
    /// Returns active `[1, kv_heads, end, head_dim]` views.
    pub(in super::super) fn append_flat_kv(
        &mut self,
        k: &MxArray,
        v: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let rows = self.flat_kv_len()?;
        let add = k.shape_at(2)? as usize;
        let end = rows + add;
        let capacity = match &self.keys {
            Some(keys) => keys.shape_at(2)? as usize,
            None => 0,
        };
        if end > capacity {
            let target = end.next_multiple_of(FLAT_KV_STEP);
            let grow = |old: &Option<MxArray>, update: &MxArray| -> Result<MxArray> {
                let shape = [
                    update.shape_at(0)?,
                    update.shape_at(1)?,
                    (target - rows) as i64,
                    update.shape_at(3)?,
                ];
                let fresh = MxArray::zeros(&shape, Some(update.dtype()?))?;
                match old {
                    Some(old) if rows > 0 => {
                        MxArray::concatenate(&old.slice_axis(2, 0, rows as i64)?, &fresh, 2)
                    }
                    _ => Ok(fresh),
                }
            };
            self.keys = Some(grow(&self.keys, k)?);
            self.values = Some(grow(&self.values, v)?);
        }
        let (Some(keys), Some(values)) = (self.keys.as_mut(), self.values.as_mut()) else {
            return Err(Error::from_reason("Qwen4 flat KV cache lost its buffer"));
        };
        keys.slice_assign_axis_inplace(2, rows as i64, end as i64, k)?;
        values.slice_assign_axis_inplace(2, rows as i64, end as i64, v)?;
        self.flat_kv_rows = Some(end);
        Ok((
            keys.slice_axis(2, 0, end as i64)?,
            values.slice_axis(2, 0, end as i64)?,
        ))
    }
    /// Ensure preallocated non-paged flat KV buffers have capacity for `add`
    /// more tokens without mutating `flat_kv_rows` yet. Returns
    /// `(offset, keys_buf, values_buf)` for fused in-place CUDA KV store.
    pub(in super::super) fn ensure_flat_kv_capacity(
        &mut self,
        kv_heads: i64,
        head_dim: i64,
        add: usize,
        dtype: DType,
    ) -> Result<(usize, MxArray, MxArray)> {
        let rows = self.flat_kv_len()?;
        let end = rows + add;
        let capacity = match &self.keys {
            Some(keys) => keys.shape_at(2)? as usize,
            None => 0,
        };
        if end > capacity {
            let target = end.next_multiple_of(FLAT_KV_STEP);
            let grow = |old: &Option<MxArray>| -> Result<MxArray> {
                let fresh = MxArray::zeros(
                    &[1, kv_heads, (target - rows) as i64, head_dim],
                    Some(dtype),
                )?;
                match old {
                    Some(old) if rows > 0 => {
                        MxArray::concatenate(&old.slice_axis(2, 0, rows as i64)?, &fresh, 2)
                    }
                    _ => Ok(fresh),
                }
            };
            self.keys = Some(grow(&self.keys)?);
            self.values = Some(grow(&self.values)?);
            // Keep `flat_kv_len()` anchored at `rows` (critical when pre-reserving
            // at `rows == 0` before chunk 0, otherwise `flat_kv_len()` would see
            // `flat_kv_rows == None` and return the allocated capacity!).
            self.flat_kv_rows = Some(rows);
        }
        let (Some(keys), Some(values)) = (&self.keys, &self.values) else {
            return Err(Error::from_reason("Qwen4 flat KV cache lost its buffer"));
        };
        Ok((rows, keys.clone(), values.clone()))
    }
    /// Ensure preallocated non-paged flat `index_keys` has capacity for
    /// `add_blocks` more indexer blocks without advancing `flat_index_blocks`.
    /// Returns the newly allocated buffer if growth occurred.
    pub(in super::super) fn ensure_flat_index_capacity(
        &mut self,
        dim: i64,
        add_blocks: usize,
        dtype: DType,
    ) -> Result<Option<MxArray>> {
        let rows = self.index_block_count()?;
        let end = rows + add_blocks;
        let capacity = match &self.index_keys {
            Some(existing) => existing.shape_at(0)? as usize,
            None => 0,
        };
        if end <= capacity {
            return Ok(None);
        }
        let target = end.next_multiple_of(FLAT_INDEX_STEP);
        let fresh = MxArray::zeros(&[(target - rows) as i64, dim], Some(dtype))?;
        let grown = match &self.index_keys {
            Some(old) if rows > 0 => {
                MxArray::concatenate(&old.slice_axis(0, 0, rows as i64)?, &fresh, 0)?
            }
            _ => fresh,
        };
        self.index_keys = Some(grown.clone());
        self.flat_index_blocks = Some(rows);
        Ok(Some(grown))
    }
    /// Trim non-paged flat KV / indexer cursors back to `rows` tokens (keeping
    /// preallocated storage intact for subsequent decode or speculative verify).
    #[allow(dead_code)]
    pub(in super::super) fn trim_flat(&mut self, rows: usize, compress_ratio: usize) {
        if let Some(cur) = self.flat_kv_rows {
            self.flat_kv_rows = Some(cur.min(rows));
        }
        if let Some(cur) = self.flat_index_blocks {
            let keep_blocks = rows / compress_ratio.max(1);
            self.flat_index_blocks = Some(cur.min(keep_blocks));
        }
        self.index_tail.clear();
    }
    /// Write draft K/V rows after this cache's rows. A buffer too small for
    /// them is replaced in this cache only: other holders keep reading theirs.
    pub(in super::super) fn append_draft_kv(
        &mut self,
        k: &MxArray,
        v: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let rows = self.draft_rows.unwrap_or(0);
        let end = rows + k.shape_at(2)? as usize;
        let capacity = match &self.keys {
            Some(keys) => keys.shape_at(2)? as usize,
            None => 0,
        };
        if end > capacity {
            let target = (end + DRAFT_HEAD_ROWS).next_multiple_of(DRAFT_HISTORY_STEP);
            let grow = |old: &Option<MxArray>, update: &MxArray| -> Result<MxArray> {
                let shape = [
                    update.shape_at(0)?,
                    update.shape_at(1)?,
                    (target - rows) as i64,
                    update.shape_at(3)?,
                ];
                let fresh = MxArray::zeros(&shape, Some(update.dtype()?))?;
                match old {
                    Some(old) if rows > 0 => {
                        MxArray::concatenate(&old.slice_axis(2, 0, rows as i64)?, &fresh, 2)
                    }
                    _ => Ok(fresh),
                }
            };
            self.keys = Some(grow(&self.keys, k)?);
            self.values = Some(grow(&self.values, v)?);
        }
        let (Some(keys), Some(values)) = (self.keys.as_mut(), self.values.as_mut()) else {
            return Err(Error::from_reason(
                "Qwen4 MTP draft history lost its buffer",
            ));
        };
        keys.slice_assign_axis_inplace(2, rows as i64, end as i64, k)?;
        values.slice_assign_axis_inplace(2, rows as i64, end as i64, v)?;
        self.draft_rows = Some(end);
        Ok((keys.clone(), values.clone()))
    }
}

impl Decoder {
    /// Cached FP32 `(scale, dt)` for GDN layer `i`.
    pub(in super::super) fn gdn_scale_dt(
        &mut self,
        i: usize,
        p: &str,
        g: &str,
    ) -> Result<(MxArray, MxArray)> {
        if self.scope.is_empty()
            && let Some(pair) = self.cuda_gdn_scale_dt.get(&i)
        {
            return Ok(pair.clone());
        }
        let a_log = self
            .dense(&format!("{p}.A_log"), &format!("{g}.ssm_a"))?
            .astype(DType::Float32)?;
        let scale = if self.gguf() {
            a_log
        } else {
            a_log.exp()?.negative()?
        };
        let dt = self
            .dense(&format!("{p}.dt_bias"), &format!("{g}.ssm_dt.bias"))?
            .astype(DType::Float32)?;
        if self.scope.is_empty() {
            self.cuda_gdn_scale_dt
                .insert(i, (scale.clone(), dt.clone()));
        }
        Ok((scale, dt))
    }

    pub(super) fn rope(&self, x: &MxArray, pos: usize) -> Result<MxArray> {
        if !self.scope.is_empty() || self.positions.is_empty() {
            return math::rope(
                x,
                pos,
                self.config.rope_dims(),
                self.config.rope_theta(),
                self.yarn.as_ref(),
            );
        }
        let v = &self.config.rope_parameters;
        let p = self
            .positions
            .get(pos)
            .copied()
            .unwrap_or([pos as i64 + self.rope_delta; 3]);
        let sections =
            std::array::from_fn(|i| v["mrope_section"][i].as_u64().unwrap_or(0) as usize);
        let interleaved = v["mrope_interleaved"].as_bool().unwrap_or(true);
        math::mrope(
            x,
            p,
            self.config.rope_dims(),
            self.config.rope_theta(),
            sections,
            interleaved,
            self.yarn.as_ref(),
        )
    }
    #[cfg(test)]
    pub(in super::super) fn rope_row(&self, x: &MxArray, pos: usize) -> Result<MxArray> {
        self.rope(x, pos)
    }
    pub(super) fn gguf(&self) -> bool {
        self.weights.gguf && self.scope.is_empty()
    }
    pub(super) fn key(&self, hf: &str, gg: &str) -> String {
        if self.gguf() {
            gg.into()
        } else {
            format!("{}{hf}", self.scope)
        }
    }
    pub(super) fn dense(&mut self, hf: &str, gg: &str) -> Result<MxArray> {
        self.weights.dense(&self.key(hf, gg))
    }
    pub(super) fn linear(&mut self, x: &MxArray, hf: &str, gg: &str) -> Result<MxArray> {
        self.weights.linear(&self.key(hf, gg), x)
    }
    pub(super) fn linear_pair(
        &mut self,
        x: &MxArray,
        a: (&str, &str),
        b: (&str, &str),
    ) -> Result<(MxArray, MxArray)> {
        self.weights
            .linear_pair(x, &self.key(a.0, a.1), &self.key(b.0, b.1))
    }
    /// Row-exact projections of a verification window (`[1, R, K]`, `R <= 4`):
    /// row `j` carries the bits of the single-row projection. Outside a
    /// verification pass a window is a prefill window, whatever its row
    /// count, and takes the batched projection: the row kernels admit four
    /// rows, and every other row would run a single-row projection of its own
    /// (a 6k prompt dispatched 1.5 M of them).
    pub(super) fn linear_rows(&mut self, x: &MxArray, hf: &str, gg: &str) -> Result<MxArray> {
        if self.verification.is_none() {
            return self.linear(x, hf, gg);
        }
        self.weights.linear_rows(&self.key(hf, gg), x)
    }
    pub(super) fn linear_pair_rows(
        &mut self,
        x: &MxArray,
        a: (&str, &str),
        b: (&str, &str),
    ) -> Result<(MxArray, MxArray)> {
        if self.verification.is_none() {
            return self.linear_pair(x, a, b);
        }
        self.weights
            .linear_pair_rows(x, &self.key(a.0, a.1), &self.key(b.0, b.1))
    }
    pub(super) fn norm(
        &mut self,
        x: &MxArray,
        hf: &str,
        gg: &str,
        group: usize,
        centered: bool,
    ) -> Result<MxArray> {
        let w = self.dense(hf, gg)?;
        math::norm(
            x,
            &w,
            group,
            self.config.rms_norm_eps,
            centered && !self.gguf(),
        )
    }
    pub fn check_cancelled(&self) -> Result<()> {
        if self.decode_pass {
            return Ok(());
        }
        if self
            .cancelled
            .as_ref()
            .is_some_and(|c| c.load(Ordering::Relaxed))
        {
            return Err(Error::from_reason("prefill cancelled"));
        }
        Ok(())
    }
    fn shape(&self, hf: &str, gg: &str, want: &[usize]) -> Result<()> {
        let key = self.key(hf, gg);
        let shape = &self.weights.descriptor(&key)?.shape;
        if shape != want {
            return Err(Error::from_reason(format!(
                "qwen4_exp tensor {key}: expected {want:?}, got {shape:?}"
            )));
        }
        Ok(())
    }
    pub(super) fn preflight(&self) -> Result<()> {
        let c = &self.config;
        let h = c.hidden_size;
        let hc = h * c.hc_count;
        self.shape(
            "embed_tokens.weight",
            "token_embd.weight",
            &[c.vocab_size, h],
        )?;
        self.shape("lm_head.weight", "output.weight", &[c.vocab_size, h])?;
        for (hf, gg, shape) in [
            (
                "hyper_connection_mixer.hc_norm.weight",
                "output_hc_norm.weight",
                vec![hc],
            ),
            (
                "hyper_connection_mixer.input_mix_weight_down.weight",
                "output_hc_down.weight",
                vec![c.hc_lowrank, hc],
            ),
            (
                "hyper_connection_mixer.input_mix_weight_up.weight",
                "output_hc_up.weight",
                vec![hc, c.hc_lowrank],
            ),
        ] {
            self.shape(hf, gg, &shape)?;
        }
        for i in 0..c.num_hidden_layers {
            for (hf, gg) in [
                ("attn_hyper_connection", "hc_attn"),
                ("mlp_hyper_connection", "hc_ffn"),
            ] {
                self.shape(
                    &format!("layers.{i}.{hf}.input_mix_weight_down.weight"),
                    &format!("blk.{i}.{gg}_down.weight"),
                    &[c.hc_lowrank, hc],
                )?;
                self.shape(
                    &format!("layers.{i}.{hf}.input_mix_weight_up.weight"),
                    &format!("blk.{i}.{gg}_up.weight"),
                    &[hc, c.hc_lowrank],
                )?;
                self.shape(
                    &format!("layers.{i}.{hf}.hc_norm.weight"),
                    &format!("blk.{i}.{gg}_norm.weight"),
                    &[hc],
                )?;
                self.shape(
                    &format!("layers.{i}.{hf}.block_inject_weight.weight"),
                    &format!("blk.{i}.{gg}_inject.weight"),
                    &[c.hc_count, hc],
                )?;
            }
            let kd = c.linear_num_key_heads * c.linear_key_head_dim;
            let vd = c.linear_num_value_heads * c.linear_value_head_dim;
            let p = format!("layers.{i}");
            let g = format!("blk.{i}");
            if c.linear(i) {
                for (hf, gg, shape) in [
                    (
                        "linear_attn.in_proj_qkv.weight",
                        "attn_qkv.weight",
                        vec![2 * kd + vd, h],
                    ),
                    (
                        "linear_attn.in_proj_z.weight",
                        "attn_gate.weight",
                        vec![vd, h],
                    ),
                    (
                        "linear_attn.in_proj_a.weight",
                        "ssm_alpha.weight",
                        vec![c.linear_num_value_heads, h],
                    ),
                    (
                        "linear_attn.in_proj_b.weight",
                        "ssm_beta.weight",
                        vec![c.linear_num_value_heads, h],
                    ),
                    ("linear_attn.out_proj.weight", "ssm_out.weight", vec![h, vd]),
                    ("linear_attn.A_log", "ssm_a", vec![c.linear_num_value_heads]),
                    (
                        "linear_attn.dt_bias",
                        "ssm_dt.bias",
                        vec![c.linear_num_value_heads],
                    ),
                    (
                        "linear_attn.norm.weight",
                        "ssm_norm.weight",
                        vec![c.linear_value_head_dim],
                    ),
                ] {
                    self.shape(&format!("{p}.{hf}"), &format!("{g}.{gg}"), &shape)?;
                }
                self.shape(
                    &format!("{p}.linear_attn.conv1d.weight"),
                    &format!("{g}.ssm_conv1d.weight"),
                    &if self.gguf() {
                        vec![2 * kd + vd, c.linear_conv_kernel_dim]
                    } else {
                        vec![2 * kd + vd, 1, c.linear_conv_kernel_dim]
                    },
                )?;
            } else {
                for (hf, gg, shape) in [
                    (
                        "self_attn.q_proj.weight",
                        "attn_q.weight",
                        vec![2 * c.num_attention_heads * c.head_dim, h],
                    ),
                    (
                        "self_attn.k_proj.weight",
                        "attn_k.weight",
                        vec![c.num_key_value_heads * c.head_dim, h],
                    ),
                    (
                        "self_attn.v_proj.weight",
                        "attn_v.weight",
                        vec![c.num_key_value_heads * c.head_dim, h],
                    ),
                    (
                        "self_attn.o_proj.weight",
                        "attn_output.weight",
                        vec![h, c.num_attention_heads * c.head_dim],
                    ),
                    (
                        "self_attn.q_norm.weight",
                        "attn_q_norm.weight",
                        vec![c.head_dim],
                    ),
                    (
                        "self_attn.k_norm.weight",
                        "attn_k_norm.weight",
                        vec![c.head_dim],
                    ),
                    (
                        "self_attn.indexer.q_layernorm.weight",
                        "indexer.q_norm.weight",
                        vec![c.indexer_head_dim],
                    ),
                    (
                        "self_attn.indexer.k_layernorm.weight",
                        "indexer.k_norm.weight",
                        vec![c.indexer_head_dim],
                    ),
                ] {
                    self.shape(&format!("{p}.{hf}"), &format!("{g}.{gg}"), &shape)?;
                }
                if self.gguf() {
                    self.shape(
                        "",
                        &format!("{g}.indexer.q_proj.weight"),
                        &[c.indexer_n_heads * c.indexer_head_dim, h],
                    )?;
                    self.shape(
                        "",
                        &format!("{g}.indexer.k_proj.weight"),
                        &[c.indexer_head_dim, h],
                    )?;
                } else {
                    self.shape(
                        &format!("{p}.self_attn.indexer.index_qk_proj.weight"),
                        "",
                        &[(c.indexer_n_heads + 1) * c.indexer_head_dim, h],
                    )?;
                }
            }
            for (hf, gg, shape) in [
                (
                    "shared_expert.gate_proj.weight",
                    "ffn_gate_shexp.weight",
                    vec![c.shared_expert_intermediate_size, h],
                ),
                (
                    "shared_expert.up_proj.weight",
                    "ffn_up_shexp.weight",
                    vec![c.shared_expert_intermediate_size, h],
                ),
                (
                    "shared_expert.down_proj.weight",
                    "ffn_down_shexp.weight",
                    vec![h, c.shared_expert_intermediate_size],
                ),
            ] {
                self.shape(&format!("{p}.mlp.{hf}"), &format!("{g}.{gg}"), &shape)?;
            }
            self.shape(
                &format!("{p}.mlp.shared_expert_gate.weight"),
                &format!("{g}.ffn_gate_inp_shexp.weight"),
                &if self.gguf() { vec![h] } else { vec![1, h] },
            )?;
            if c.ple_layer_ids.contains(&(i + 1)) {
                for (hf, gg, shape) in [
                    (
                        "key_proj.weight",
                        "ple_key.weight",
                        vec![hc, c.ple_embed_dim],
                    ),
                    (
                        "value_proj.weight",
                        "ple_value.weight",
                        vec![h, c.ple_embed_dim],
                    ),
                    ("norm_key.weight", "ple_norm_key.weight", vec![hc]),
                    ("norm_query.weight", "ple_norm_query.weight", vec![hc]),
                    ("norm_conv.weight", "ple_norm_conv.weight", vec![hc]),
                ] {
                    self.shape(&format!("{p}.ple.{hf}"), &format!("{g}.{gg}"), &shape)?;
                }
                self.shape(
                    &format!("{p}.ple.conv1d.weight"),
                    &format!("{g}.ple_conv1d.weight"),
                    &if self.gguf() {
                        vec![hc, c.ple_conv_kernel_size]
                    } else {
                        vec![hc, 1, c.ple_conv_kernel_size]
                    },
                )?;
                let heads = (c.ngram_size - 1) * c.heads_per_ngram;
                let mut rows = 0usize;
                if self.gguf() {
                    let table = self.weights.descriptor("per_layer_token_embd.weight")?;
                    if table.shape.len() != 2 || table.width()? != c.ple_embed_dim / heads {
                        return Err(Error::from_reason("Invalid GGUF PLE shape"));
                    }
                    rows = table.rows()?;
                } else {
                    for shard in 0..c.split_ngram_parts {
                        let table = self.weights.descriptor(&format!(
                            "{p}.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
                        ))?;
                        if table.shape.len() != 2 || table.width()? != c.ple_embed_dim / heads {
                            return Err(Error::from_reason("Invalid PLE shard shape"));
                        }
                        rows = rows
                            .checked_add(table.rows()?)
                            .ok_or_else(|| Error::from_reason("PLE row count overflow"))?;
                    }
                }
                let constants = |hf: &str, gg: &str| {
                    if self.gguf() {
                        super::super::gguf::integers(
                            &self.weights.metadata,
                            &format!("qwen4exp.ple.{gg}"),
                        )
                    } else {
                        self.weights
                            .integers(&format!("{p}.ple.ple_embedding.{hf}"))
                    }
                };
                let sizes = constants("ngram_heads_vocab_sizes", "head_vocab_sizes")?;
                let offsets = constants("ngram_heads_offsets", "head_offsets")?;
                let multipliers = constants("layer_multipliers", "layer_multipliers")?;
                math::hash_ids(
                    0,
                    &[],
                    c.eos_token_id,
                    c.ngram_size,
                    c.heads_per_ngram,
                    &multipliers,
                    &sizes,
                    &offsets,
                )?;
                let mut end = 0u64;
                for (&size, &offset) in sizes.iter().zip(&offsets) {
                    if offset != end {
                        return Err(Error::from_reason(
                            "PLE hash vocabularies overlap or contain gaps",
                        ));
                    }
                    end = end
                        .checked_add(size)
                        .ok_or_else(|| Error::from_reason("PLE vocabulary size overflow"))?;
                }
                if end > rows as u64 {
                    return Err(Error::from_reason(
                        "PLE table does not cover its hashed vocabulary",
                    ));
                }
            }
            self.shape(
                &format!("layers.{i}.mlp.gate.weight"),
                &format!("blk.{i}.ffn_gate_inp.weight"),
                &[c.num_experts, h],
            )?;
            if self.gguf() {
                self.shape(
                    "",
                    &format!("blk.{i}.ffn_gate_exps.weight"),
                    &[c.num_experts, c.moe_intermediate_size, h],
                )?;
                self.shape(
                    "",
                    &format!("blk.{i}.ffn_up_exps.weight"),
                    &[c.num_experts, c.moe_intermediate_size, h],
                )?;
            } else {
                self.shape(
                    &format!("layers.{i}.mlp.experts.gate_up_proj"),
                    "",
                    &[c.num_experts, 2 * c.moe_intermediate_size, h],
                )?;
            }
            self.shape(
                &format!("layers.{i}.mlp.experts.down_proj"),
                &format!("blk.{i}.ffn_down_exps.weight"),
                &[c.num_experts, h, c.moe_intermediate_size],
            )?;
        }
        Ok(())
    }

    pub(super) fn submit(&self, arrays: &[&MxArray], context: &str) -> Result<()> {
        #[cfg(test)]
        self.submit_log.borrow_mut().push(context.to_owned());
        if crate::engine::persistence::compiled_forward_backend_available() {
            MxArray::async_eval_arrays(arrays);
            Ok(())
        } else if super::super::cuda_qwen4_fast_enabled() {
            Self::cuda_async_eval(arrays);
            Ok(())
        } else {
            MxArray::eval_arrays_with_context(arrays, context)
        }
    }
    pub(super) fn submit_state(&self, arrays: &[&MxArray], context: &str) -> Result<()> {
        // The reference submits whole layer groups. Inside a checked device
        // transaction, its final join also completes every auxiliary state
        // before bank readers are released or a failed window is replayed.
        if self.deferred_state_submission() || self.cuda_deferred_state() {
            Ok(())
        } else {
            self.submit(arrays, context)
        }
    }

    /// A test event (`submit_log`): the group's frontier evaluated, the
    /// route commit, end, trip, switch and the tail's start, in the order
    /// they ran.
    pub(super) fn note(&self, event: &str) {
        #[cfg(test)]
        self.submit_log.borrow_mut().push(event.to_owned());
        #[cfg(not(test))]
        let _ = event;
    }

    pub(super) fn gdn_projections(
        &mut self,
        x: &MxArray,
        p: &str,
        g: &str,
    ) -> Result<GdnProjections> {
        let (qkv, z) = self.linear_pair_rows(
            x,
            (
                &format!("{p}.in_proj_qkv.weight"),
                &format!("{g}.attn_qkv.weight"),
            ),
            (
                &format!("{p}.in_proj_z.weight"),
                &format!("{g}.attn_gate.weight"),
            ),
        )?;
        let z = z.reshape(&[
            1,
            x.shape_at(1)?,
            self.config.linear_num_value_heads as i64,
            self.config.linear_value_head_dim as i64,
        ])?;
        let (a, b) = if super::super::cuda_qwen4_fast_enabled() && x.dtype()? == DType::BFloat16 {
            // Pair ssm_alpha + ssm_beta ([96, 2560] BF16) into one `bf16_matmul_few`
            // launch and pass BF16 directly into `GdnPrepareCuda` (no F32 cast!).
            self.linear_pair_rows(
                x,
                (
                    &format!("{p}.in_proj_a.weight"),
                    &format!("{g}.ssm_alpha.weight"),
                ),
                (
                    &format!("{p}.in_proj_b.weight"),
                    &format!("{g}.ssm_beta.weight"),
                ),
            )?
        } else {
            let a = self
                .linear_rows(
                    x,
                    &format!("{p}.in_proj_a.weight"),
                    &format!("{g}.ssm_alpha.weight"),
                )?
                .astype(DType::Float32)?;
            let b = self
                .linear_rows(
                    x,
                    &format!("{p}.in_proj_b.weight"),
                    &format!("{g}.ssm_beta.weight"),
                )?
                .astype(DType::Float32)?;
            (a, b)
        };
        Ok(GdnProjections { qkv, z, a, b })
    }

    pub(super) fn attention_projections(
        &mut self,
        x: &MxArray,
        i: usize,
    ) -> Result<AttentionProjections> {
        let p = format!("layers.{i}.self_attn");
        let g = format!("blk.{i}");
        let qg = self.linear_rows(
            x,
            &format!("{p}.q_proj.weight"),
            &format!("{g}.attn_q.weight"),
        )?;
        let (k, v) = self.linear_pair_rows(
            x,
            (&format!("{p}.k_proj.weight"), &format!("{g}.attn_k.weight")),
            (&format!("{p}.v_proj.weight"), &format!("{g}.attn_v.weight")),
        )?;
        let ih = self.config.indexer_n_heads as i64;
        let id = self.config.indexer_head_dim as i64;
        let (iq, ik) = if self.gguf() {
            (
                self.linear_rows(x, "", &format!("{g}.indexer.q_proj.weight"))?,
                self.linear_rows(x, "", &format!("{g}.indexer.k_proj.weight"))?,
            )
        } else {
            let qk = self.linear_rows(x, &format!("{p}.indexer.index_qk_proj.weight"), "")?;
            (
                qk.slice_axis(2, 0, ih * id)?,
                qk.slice_axis(2, ih * id, (ih + 1) * id)?,
            )
        };
        Ok(AttentionProjections { qg, k, v, iq, ik })
    }

    pub(super) fn shared_expert_parts(
        &mut self,
        x: &MxArray,
        i: usize,
    ) -> Result<(MxArray, MxArray)> {
        let p = format!("layers.{i}.mlp");
        let g = format!("blk.{i}");
        let (gate, up) = self.linear_pair_rows(
            x,
            (
                &format!("{p}.shared_expert.gate_proj.weight"),
                &format!("{g}.ffn_gate_shexp.weight"),
            ),
            (
                &format!("{p}.shared_expert.up_proj.weight"),
                &format!("{g}.ffn_up_shexp.weight"),
            ),
        )?;
        let activation = math::swiglu(&gate, &up)?;
        let shared = self.linear_rows(
            &activation,
            &format!("{p}.shared_expert.down_proj.weight"),
            &format!("{g}.ffn_down_shexp.weight"),
        )?;
        drop(activation);
        let gate = self.linear_rows(
            x,
            &format!("{p}.shared_expert_gate.weight"),
            &format!("{g}.ffn_gate_inp_shexp.weight"),
        )?;
        Ok((shared, gate))
    }
}
