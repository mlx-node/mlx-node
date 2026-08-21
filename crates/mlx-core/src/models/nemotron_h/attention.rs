//! NemotronH GQA attention mixer.
//!
//! Port of the HuggingFace NemotronHAttention: GQA (32 query / 2 KV heads,
//! head_dim 128), bias-free projections, BF16. No Q/K RMSNorm and no
//! softcap for this family.
//!
//! NemotronH attention is NoPE - no positional rotation is applied
//! anywhere. All three references agree: HF `NemotronHAttention.forward`
//! projects Q/K/V, updates the cache and calls the attention interface
//! (`apply_rotary_pos_emb` is defined in that module but never called from
//! this class), vLLM's `NemotronHAttention.forward(self, hidden_states,
//! **kwargs)` takes no `positions` at all, and mlx-lm's `__call__` goes
//! straight from the projections to SDPA. Position information reaches the
//! model through the Mamba-2 mixers instead.
//!
//! The same layer serves both the backbone attention blocks and the MTP
//! head's attention layer. The MTP head is a real decoder layer with its
//! OWN KV cache group (vLLM `NemotronHMTPAttentionDecoderLayer` ->
//! `NemotronHAttention` -> its own `Attention(...)`), so it drives the very
//! same `forward` as the backbone, with its own per-layer cache.

use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::{DType, MxArray};
use crate::models::qwen3_5_moe::quantized_linear::LinearProj;
use crate::nn::Linear;
use crate::transformer::KVCache;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};
use napi::bindgen_prelude::*;

use super::config::NemotronHConfig;

/// The dtype the block-paged KV pool stores, and therefore the dtype the
/// flat `KVCache` has to end up holding for the two lanes to run the same
/// attention arithmetic.
///
/// This is a hard constraint on the paged side, not a preference:
/// `PagedKVCacheAdapter` rejects anything but Float16/BFloat16 at every K/V
/// boundary ("kernel io_type is half-precision") and the pool is
/// byte-allocated for 2-byte elements. The FLAT cache has no dtype of its
/// own - `KVCache::update_and_fetch` allocates its buffer with
/// `keys.dtype()` - so it holds whatever the residual stream hands the K/V
/// projections. Keeping that stream in the model dtype (bf16) is what keeps
/// the lanes aligned; see `gated_rmsnorm` in `mamba2.rs`.
/// `pub(crate)` so the end-to-end loader seam gate in `persistence.rs` can
/// assert the FLAT cache against the paged writer's OWN constant rather than
/// re-typing the literal — otherwise a future change to the pool dtype would
/// leave that assertion quietly checking the old value.
pub(crate) const PAGED_KV_IO_DTYPE: DType = DType::BFloat16;

pub struct NemotronHAttention {
    pub(crate) q_proj: LinearProj,
    pub(crate) k_proj: LinearProj,
    pub(crate) v_proj: LinearProj,
    pub(crate) o_proj: LinearProj,
    pub(crate) num_heads: i32,
    pub(crate) num_kv_heads: i32,
    pub(crate) head_dim: i32,
    pub(crate) scale: f64,
}

impl NemotronHAttention {
    pub fn new(config: &NemotronHConfig) -> Result<Self> {
        let h = config.hidden_size as u32;
        let q_dim = (config.num_attention_heads * config.head_dim) as u32;
        let kv_dim = (config.num_key_value_heads * config.head_dim) as u32;

        let q_proj = LinearProj::Standard(Linear::new(h, q_dim, Some(false))?);
        let k_proj = LinearProj::Standard(Linear::new(h, kv_dim, Some(false))?);
        let v_proj = LinearProj::Standard(Linear::new(h, kv_dim, Some(false))?);
        let o_proj = LinearProj::Standard(Linear::new(q_dim, h, Some(false))?);

        let scale = (config.head_dim as f64).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            scale,
        })
    }
}

impl NemotronHAttention {
    /// Standard causal attention forward with an optional flat KVCache.
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;

        let keys = keys.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;

        let values = values.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        let (keys, values) = if let Some(c) = cache {
            c.update_and_fetch(&keys, &values)?
        } else {
            (keys, values)
        };

        let output = if let Some(m) = mask {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale, Some(m))?
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(&queries, &keys, &values, self.scale)?
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale, None)?
        };

        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        self.o_proj.forward(&output)
    }

    /// Whether any projection holds a quantized backend.
    pub fn is_quantized(&self) -> bool {
        self.q_proj.is_quantized()
            || self.k_proj.is_quantized()
            || self.v_proj.is_quantized()
            || self.o_proj.is_quantized()
    }
    /// Block-paged forward driven by the PagedKVCacheAdapter.
    ///
    /// Mirrors the LFM2 forward_paged contract: x is already pre-normalized,
    /// attn_layer_idx is the ATTENTION-LAYER ORDINAL into the adapter's
    /// LayerKVPool (0..6, NOT the absolute decoder index), and the caller must
    /// have recorded the suffix in the adapter BEFORE this call so the
    /// update_keys_values alignment passes. K/V are written into the shared
    /// pool; attention reads the pool through the graph-native gather/prefill
    /// bridges (with the synchronous fallbacks).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let queries_bhtd = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let keys_bhtd = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values = values.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values_bhtd = values.transpose(Some(&[0, 2, 1, 3]))?;

        // Paged layout [num_tokens, n_kv_heads, head_dim] (batch == 1 on the
        // single-row prefill path).
        let keys_paged = keys_bhtd.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            batch * seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values_paged = values_bhtd.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            batch * seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;

        // The KV pool stores 2-byte elements (bf16); cast at the write
        // boundary so an f32 upstream stream (unit-test fixtures) is accepted
        // and a production bf16 stream is an identity no-op.
        let keys_paged = keys_paged.astype(PAGED_KV_IO_DTYPE)?;
        let values_paged = values_paged.astype(PAGED_KV_IO_DTYPE)?;
        let native_written = crate::models::lfm2::attention::native_kv_write_enabled()
            && adapter
                .update_keys_values_native(
                    attn_layer_idx,
                    &keys_paged,
                    &values_paged,
                    first_logical_position,
                )
                .is_ok();
        if !native_written {
            adapter
                .update_keys_values(
                    attn_layer_idx,
                    &keys_paged,
                    &values_paged,
                    first_logical_position,
                )
                .map_err(napi::Error::from_reason)?;
        }

        let attn_bhtd = if is_prefill {
            if cached_prefix_len == 0 {
                if seq_len > 1 {
                    scaled_dot_product_attention_causal(
                        &queries_bhtd,
                        &keys_bhtd,
                        &values_bhtd,
                        self.scale,
                    )?
                } else {
                    scaled_dot_product_attention(
                        &queries_bhtd,
                        &keys_bhtd,
                        &values_bhtd,
                        self.scale,
                        None,
                    )?
                }
            } else {
                // Cache-hit prefill: the suffix was just written above; the
                // paged kernel attends over cached prefix + fresh suffix.
                let total_ctx = cached_prefix_len + (seq_len as u32);
                let queries_paged = queries_bhtd
                    .squeeze(Some(&[0]))?
                    .transpose(Some(&[1, 0, 2]))?;
                let maybe_paged_attn = adapter
                    .gather_kv_for_prefill_chunk(
                        attn_layer_idx,
                        &queries_paged,
                        cached_prefix_len,
                        self.scale as f32,
                    )
                    .ok()
                    .map(|attn_t_h_d| {
                        let target_dtype = x.dtype()?;
                        let attn_t_h_d = attn_t_h_d.astype(target_dtype)?;
                        attn_t_h_d.transpose(Some(&[1, 0, 2]))?.reshape(&[
                            batch,
                            self.num_heads as i64,
                            seq_len,
                            self.head_dim as i64,
                        ])
                    })
                    .transpose()?;
                match maybe_paged_attn {
                    Some(attn) => attn,
                    None => {
                        let (k_full, v_full) = adapter
                            .read_kv_range(attn_layer_idx, 0, total_ctx)
                            .map_err(napi::Error::from_reason)?;
                        let mask = crate::array::mask::create_causal_mask(
                            seq_len as i32,
                            Some(cached_prefix_len as i32),
                            None,
                        )?;
                        scaled_dot_product_attention(
                            &queries_bhtd,
                            &k_full,
                            &v_full,
                            self.scale,
                            Some(&mask),
                        )?
                    }
                }
            }
        } else {
            // Decode: gather full historical K/V via the paged kernel (the
            // kernels are half-precision; production bf16 streams pass through
            // this cast unchanged).
            let queries_3d = queries_bhtd
                .squeeze(Some(&[2]))?
                .reshape(&[1, self.num_heads as i64, self.head_dim as i64])?
                .astype(PAGED_KV_IO_DTYPE)?;
            let attn_3d = if crate::models::lfm2::attention::graph_decode_gather_enabled() {
                match adapter.gather_kv_for_decode_graph(
                    attn_layer_idx,
                    &queries_3d,
                    self.scale as f32,
                    /* softcap */ 1.0,
                ) {
                    Ok(attn_3d) => attn_3d,
                    Err(_) => adapter
                        .gather_kv_for_decode(
                            attn_layer_idx,
                            &queries_3d,
                            self.scale as f32,
                            /* softcap */ 1.0,
                        )
                        .map_err(napi::Error::from_reason)?,
                }
            } else {
                adapter
                    .gather_kv_for_decode(
                        attn_layer_idx,
                        &queries_3d,
                        self.scale as f32,
                        /* softcap */ 1.0,
                    )
                    .map_err(napi::Error::from_reason)?
            };
            let target_dtype = x.dtype()?;
            let attn_3d = attn_3d.astype(target_dtype)?;
            attn_3d.reshape(&[1, self.num_heads as i64, 1, self.head_dim as i64])?
        };

        let output = attn_bhtd.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;
        self.o_proj.forward(&output)
    }

    /// Uniform batched paged decode for the continuous-batching lane.
    ///
    /// One token per row: queries [N, H, D], K/V [N, kvH, D], one batched
    /// native K/V write and one graph-native batched attention gather. No
    /// serial fallback - a genuine N-row wave must share the weight stream.
    /// `rows` carries the per-row (seq id, logical position) pairs the
    /// adapter needs for slot mapping; no rotation is applied to them.
    pub(crate) fn forward_paged_batched(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        rows: &[(SeqId, u32)],
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        if rows.is_empty()
            || shape.as_ref().len() != 3
            || shape[0] != rows.len() as i64
            || shape[1] != 1
        {
            return Err(Error::from_reason(format!(
                "NemotronHAttention::forward_paged_batched expects [N,1,H] for {} rows, got {:?}",
                rows.len(),
                shape.as_ref()
            )));
        }
        if !crate::models::lfm2::attention::native_kv_write_enabled()
            || !crate::models::lfm2::attention::graph_decode_gather_enabled()
        {
            return Err(Error::from_reason(
                "NemotronH batched decode requires native K/V writes and graph decode gather",
            ));
        }

        let batch = rows.len() as i64;
        let seq_ids = rows.iter().map(|&(seq_id, _)| seq_id).collect::<Vec<_>>();

        let queries = self
            .q_proj
            .forward(x)?
            .reshape(&[batch, 1, self.num_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let keys = self
            .k_proj
            .forward(x)?
            .reshape(&[batch, 1, self.num_kv_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let values = self
            .v_proj
            .forward(x)?
            .reshape(&[batch, 1, self.num_kv_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;

        let queries = queries.squeeze(Some(&[2]))?.astype(PAGED_KV_IO_DTYPE)?;
        let keys = keys.squeeze(Some(&[2]))?.astype(PAGED_KV_IO_DTYPE)?;
        let values = values.squeeze(Some(&[2]))?.astype(PAGED_KV_IO_DTYPE)?;
        adapter
            .update_keys_values_native_batched(attn_layer_idx, &keys, &values, rows)
            .map_err(Error::from_reason)?;
        let attended = adapter
            .gather_kv_for_decode_graph_batched(
                attn_layer_idx,
                &queries,
                &seq_ids,
                self.scale as f32,
                /* softcap */ 1.0,
            )
            .map_err(Error::from_reason)?
            .astype(x.dtype()?)?
            .reshape(&[batch, 1, (self.num_heads * self.head_dim) as i64])?;
        self.o_proj.forward(&attended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;
    use crate::models::nemotron_h::config::NemotronHConfig;

    fn tiny_cfg() -> NemotronHConfig {
        NemotronHConfig {
            vocab_size: 32,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 64,
            layer_norm_epsilon: 1e-5,
            rope_theta: 10000.0,
            layers_block_type: vec!["full_attention".into()],
            mamba_num_heads: 2,
            mamba_head_dim: 2,
            ssm_state_size: 2,
            n_groups: 1,
            conv_kernel: 4,
            chunk_size: 2,
            time_step_min: 0.001,
            time_step_limit: None,
            n_routed_experts: 2,
            num_experts_per_tok: 1,
            routed_scaling_factor: 1.0,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            intermediate_size: 8,
            moe_shared_expert_intermediate_size: 8,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_ids: vec![2],
            pad_token_id: 0,
            num_logits_to_keep: 1,
            mtp_layers_block_type: Vec::new(),
            n_mtp_layers: 0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
        }
    }

    /// Attention with a fresh cache over a 3-token sequence must produce
    /// per-position outputs, and a cached decode at the 4th position must
    /// equal the same forward computed from the full 4-token sequence.
    #[test]
    fn attention_kv_cache_continuity() {
        let cfg = tiny_cfg();
        let attn = NemotronHAttention::new(&cfg).expect("attention builds");
        let h = cfg.hidden_size as usize;

        let x: Vec<f32> = (0..3 * h)
            .map(|i| ((i as f32) * 0.57) % 1.0 - 0.5)
            .collect();
        let mx = MxArray::from_float32(&x, &[1, 3, h as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

        let mut cache = KVCache::new();
        let out3 = attn.forward(&mx, None, Some(&mut cache)).unwrap();
        assert_eq!(cache.get_offset(), 3);

        // 4th token: decode via cache vs full 4-token forward.
        let x4: Vec<f32> = (0..4 * h)
            .map(|i| ((i as f32) * 0.57) % 1.0 - 0.5)
            .collect();
        let mx4 = MxArray::from_float32(&x4, &[1, 4, h as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let mut cache_full = KVCache::new();
        let _ = attn.forward(&mx4, None, Some(&mut cache_full)).unwrap();

        let last = MxArray::from_float32(&x4[3 * h..], &[1, 1, h as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let out_decode = attn.forward(&last, None, Some(&mut cache)).unwrap();
        let out_full = attn.forward(&mx4, None, None).unwrap();
        let full_last = out_full
            .slice_axis(1, 3, 4)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        let decode = out_decode.to_float32().unwrap().to_vec();
        let mut max_d = 0.0f32;
        for (a, b) in decode.iter().zip(full_last.iter()) {
            max_d = max_d.max((a - b).abs());
        }
        assert!(
            max_d <= 5e-3,
            "decode-via-cache vs full forward at position 3: max |diff| = {max_d}"
        );
        let _ = out3;
    }

    /// Regression: NemotronH attention is NoPE (HF `NemotronHAttention.forward`
    /// never calls `apply_rotary_pos_emb`; vLLM's takes no `positions`; mlx-lm
    /// goes straight from the projections to SDPA). Without a positional
    /// rotation the decode output depends only on the SET of cached (K, V)
    /// pairs, so feeding the same two-token history in the opposite ORDER must
    /// give a bit-comparable result. With RoPE the two histories rotate K by
    /// swapped offsets and the outputs diverge.
    #[test]
    fn attention_has_no_positional_rotation() {
        let cfg = tiny_cfg();
        let attn = NemotronHAttention::new(&cfg).expect("attention builds");
        let h = cfg.hidden_size as i64;
        let tok = |seed: f32| -> Vec<f32> {
            (0..h)
                .map(|i| ((i as f32 + seed) * 0.29) % 1.0 - 0.5)
                .collect()
        };
        let (a, b, q) = (tok(0.0), tok(3.0), tok(7.0));

        let pair = |first: &[f32], second: &[f32]| -> MxArray {
            let mut buf = first.to_vec();
            buf.extend_from_slice(second);
            MxArray::from_float32(&buf, &[1, 2, h]).unwrap()
        };
        let query = MxArray::from_float32(&q, &[1, 1, h]).unwrap();

        let run = |hist: MxArray| -> Vec<f32> {
            let mut cache = KVCache::new();
            let _ = attn.forward(&hist, None, Some(&mut cache)).unwrap();
            assert_eq!(cache.get_offset(), 2);
            attn.forward(&query, None, Some(&mut cache))
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec()
        };

        let ab = run(pair(&a, &b));
        let ba = run(pair(&b, &a));
        let mut max_d = 0.0f32;
        for (l, r) in ab.iter().zip(ba.iter()) {
            max_d = max_d.max((l - r).abs());
        }
        assert!(
            max_d <= 1e-5,
            "attention is order-sensitive, i.e. a positional rotation is applied: max |diff| = {max_d}"
        );
    }

    /// Attention with dense BF16 projections, matching the released
    /// checkpoint (q/k/v/o carry no `.scales` key there).
    fn bf16_attention(cfg: &NemotronHConfig) -> NemotronHAttention {
        let mut attn = NemotronHAttention::new(cfg).expect("attention builds");
        let h = cfg.hidden_size as i64;
        let q_dim = (cfg.num_attention_heads * cfg.head_dim) as i64;
        let kv_dim = (cfg.num_key_value_heads * cfg.head_dim) as i64;
        let mk = |rows: i64, cols: i64, seed: f32| -> LinearProj {
            let w: Vec<f32> = (0..rows * cols)
                .map(|i| ((i as f32 + seed) * 0.13) % 1.0 - 0.5)
                .collect();
            let w = MxArray::from_float32(&w, &[rows, cols])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            LinearProj::Standard(Linear::from_weights(&w, None).unwrap())
        };
        attn.q_proj = mk(q_dim, h, 0.0);
        attn.k_proj = mk(kv_dim, h, 5.0);
        attn.v_proj = mk(kv_dim, h, 11.0);
        attn.o_proj = mk(h, q_dim, 17.0);
        attn
    }

    /// PINS THE FLAT/PAGED KV DTYPE SEAM.
    ///
    /// The flat `KVCache` has no dtype of its own: `update_and_fetch`
    /// allocates its buffer with `keys.dtype()`, so it stores exactly what
    /// `k_proj` produced - i.e. the residual stream's dtype. The paged lane
    /// has no such freedom: `forward_paged` casts every K/V write to
    /// `PAGED_KV_IO_DTYPE` because the pool is byte-allocated for 2-byte
    /// elements and the kernels are half-precision. So the moment anything
    /// upstream promotes the residual to f32, the two lanes silently run
    /// different attention arithmetic - flat over f32 K/V, paged over bf16.
    ///
    /// Mutation caught: any `astype(Float32)` introduced into this forward,
    /// or a paged pool dtype that stops matching what the flat lane stores.
    /// The UPSTREAM promotion (an f32 mixer output) is guarded in `mamba2.rs`
    /// by `mamba_mixer_is_dtype_transparent`; the second half of this test
    /// shows why that guard has to live there and not here.
    #[test]
    fn flat_kv_cache_stores_the_paged_pool_dtype() {
        let cfg = tiny_cfg();
        let h = cfg.hidden_size as i64;
        let xs: Vec<f32> = (0..3 * h)
            .map(|i| ((i as f32) * 0.57) % 1.0 - 0.5)
            .collect();

        let attn = bf16_attention(&cfg);
        let x = MxArray::from_float32(&xs, &[1, 3, h])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let mut cache = KVCache::new();
        let out = attn.forward(&x, None, Some(&mut cache)).unwrap();

        assert_eq!(
            out.dtype().unwrap(),
            DType::BFloat16,
            "attention must be dtype-transparent: a bf16 stream in, a bf16 stream out"
        );
        assert_eq!(
            cache
                .keys_ref()
                .expect("the forward must populate the cache")
                .dtype()
                .unwrap(),
            PAGED_KV_IO_DTYPE,
            "the flat KV cache must hold the same dtype the paged pool stores"
        );
        assert_eq!(
            cache
                .values_ref()
                .expect("the forward must populate the cache")
                .dtype()
                .unwrap(),
            PAGED_KV_IO_DTYPE,
            "the flat KV cache must hold the same dtype the paged pool stores"
        );

        // Teeth, and the reason no cast is added at the flat write: the flat
        // cache mirrors whatever the residual stream is. Run the very same
        // forward on an f32 stream and it stores f32 - the exact divergence
        // this test exists to catch. Casting here would paper over it while
        // leaving `out_proj`, the MoE and `lm_head` in f32, so the fix belongs
        // upstream at the mixer boundary.
        let attn_f32 = NemotronHAttention::new(&cfg).expect("attention builds");
        let x32 = MxArray::from_float32(&xs, &[1, 3, h]).unwrap();
        let mut cache_f32 = KVCache::new();
        let _ = attn_f32.forward(&x32, None, Some(&mut cache_f32)).unwrap();
        assert_eq!(
            cache_f32
                .keys_ref()
                .expect("the forward must populate the cache")
                .dtype()
                .unwrap(),
            DType::Float32,
            "the flat cache adopts the projection dtype, so the bf16 assertion \
             above is not vacuous"
        );
    }
}
