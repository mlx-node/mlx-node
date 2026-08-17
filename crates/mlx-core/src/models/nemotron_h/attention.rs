//! NemotronH GQA attention mixer.
//!
//! Port of the HuggingFace NemotronHAttention: GQA (32 query / 2 KV heads,
//! head_dim 128), standard RoPE (theta 10000, full rotary), bias-free
//! projections, BF16. No Q/K RMSNorm and no softcap for this family.
//!
//! The same layer serves both the backbone attention blocks and the MTP
//! head's attention layer. The MTP attention additionally exposes a
//! read-only forward that attends over a TARGET KV cache (the backbone's
//! final attention layer) at the next absolute position without writing
//! K/V - the vLLM NemotronH-MTP "attention at positions t+1 reading
//! target KV" contract.

use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::{DType, MxArray};
use crate::models::qwen3_5_moe::quantized_linear::LinearProj;
use crate::nn::{Linear, RoPE};
use crate::transformer::KVCache;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};
use napi::bindgen_prelude::*;

use super::config::NemotronHConfig;

pub struct NemotronHAttention {
    pub(crate) q_proj: LinearProj,
    pub(crate) k_proj: LinearProj,
    pub(crate) v_proj: LinearProj,
    pub(crate) o_proj: LinearProj,
    pub(crate) rope: RoPE,
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

        let rope = RoPE::new(
            config.head_dim,
            Some(false), // traditional=false (neox-style)
            Some(config.rope_theta),
            None,
        );

        let scale = (config.head_dim as f64).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
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

        let offset = cache.as_ref().map_or(0, |c| c.get_offset());
        let queries = self.rope.forward(&queries, Some(offset))?;
        let keys = self.rope.forward(&keys, Some(offset))?;

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

    /// Read-only attention over a TARGET KV cache (the MTP draft path).
    ///
    /// x is [1, 1, hidden]; the queries are placed at absolute RoPE
    /// position position and attend over ALL stored K/V positions in kv
    /// (no causal mask - every stored position precedes the query).
    /// K/V are never written.
    pub fn forward_read_only(&self, x: &MxArray, kv: &KVCache, position: i32) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        let queries = self.q_proj.forward(x)?;
        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let queries = self.rope.forward(&queries, Some(position))?;

        let (keys, values) = kv.keys_ref().zip(kv.values_ref()).ok_or_else(|| {
            Error::from_reason(
                "NemotronH MTP attention: target KV cache is empty (no committed tokens)",
            )
        })?;

        let output = scaled_dot_product_attention(&queries, keys, values, self.scale, None)?;
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

        let rope_offset = first_logical_position as i32;
        let queries_bhtd = self.rope.forward(&queries_bhtd, Some(rope_offset))?;
        let keys_bhtd = self.rope.forward(&keys_bhtd, Some(rope_offset))?;

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
        let keys_paged = keys_paged.astype(DType::BFloat16)?;
        let values_paged = values_paged.astype(DType::BFloat16)?;
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
                .astype(DType::BFloat16)?;
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
    /// One token per row: queries [N, H, D], K/V [N, kvH, D] with per-row
    /// RoPE offsets, one batched native K/V write and one graph-native
    /// batched attention gather. No serial fallback - a genuine N-row wave
    /// must share the weight stream.
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
        let offsets = rows
            .iter()
            .map(|&(seq_id, position)| {
                i32::try_from(position).map_err(|_| {
                    Error::from_reason(format!(
                        "NemotronH batched decode sequence {seq_id} position {position} exceeds i32::MAX"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = MxArray::from_int32(&offsets, &[batch])?;
        let seq_ids = rows.iter().map(|&(seq_id, _)| seq_id).collect::<Vec<_>>();

        let queries = self
            .q_proj
            .forward(x)?
            .reshape(&[batch, 1, self.num_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let queries = self.rope.forward_with_offsets(&queries, &offsets)?;
        let keys = self
            .k_proj
            .forward(x)?
            .reshape(&[batch, 1, self.num_kv_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let keys = self.rope.forward_with_offsets(&keys, &offsets)?;
        let values = self
            .v_proj
            .forward(x)?
            .reshape(&[batch, 1, self.num_kv_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;

        let queries = queries.squeeze(Some(&[2]))?.astype(DType::BFloat16)?;
        let keys = keys.squeeze(Some(&[2]))?.astype(DType::BFloat16)?;
        let values = values.squeeze(Some(&[2]))?.astype(DType::BFloat16)?;
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

    /// Read-only MTP attention: empty target KV fails closed; populated
    /// target KV returns the projected output at the next position.
    #[test]
    fn read_only_attention_reads_target_kv() {
        let cfg = tiny_cfg();
        let attn = NemotronHAttention::new(&cfg).expect("attention builds");
        let h = cfg.hidden_size as usize;

        let kv = KVCache::new();
        let x = MxArray::from_float32(&vec![0.1f32; h], &[1, 1, h as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        assert!(
            attn.forward_read_only(&x, &kv, 0).is_err(),
            "empty target KV must fail closed"
        );

        let mut kv = KVCache::new();
        let seq: Vec<f32> = (0..2 * h)
            .map(|i| ((i as f32) * 0.31) % 1.0 - 0.5)
            .collect();
        let mx = MxArray::from_float32(&seq, &[1, 2, h as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let attn2 = NemotronHAttention::new(&cfg).expect("attention builds 2");
        let _ = attn2.forward(&mx, None, Some(&mut kv)).unwrap();
        assert_eq!(kv.get_offset(), 2);

        let out = attn
            .forward_read_only(&x, &kv, 2)
            .expect("read-only attention with populated KV");
        let vals = out.to_float32().unwrap().to_vec();
        assert_eq!(vals.len(), h);
        assert!(vals.iter().all(|v| v.is_finite()));
    }
}
