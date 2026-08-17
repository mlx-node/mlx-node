use crate::array::MxArray;
use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::mask::create_causal_mask;
use crate::models::gemma4::layer_cache::Gemma4LayerCache;
use crate::models::gemma4::quantized_linear::LinearProj;
use crate::nn::{Activations, RoPE};
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};
use napi::bindgen_prelude::*;

use super::config::{LayerKind, MuseGlimmerTextConfig};
use super::kv_cache::{PagedWindowSlot, WindowCarrier};

/// Muse-Glimmer's gated GQA attention. Q/K normalization is weightless in the
/// HF model; the official GGUF synthesizes scale tensors, which the converter
/// deliberately omits and represents through `qk_scale_factor` here.
pub struct MuseGlimmerAttention {
    q_proj: LinearProj,
    k_proj: LinearProj,
    v_proj: LinearProj,
    o_proj: LinearProj,
    gate_proj: LinearProj,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    qk_scale_factor: f64,
    qk_norm_eps: f32,
    sliding_window: Option<i32>,
    rope: Option<RoPE>,
}

impl MuseGlimmerAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn from_projections(
        config: &MuseGlimmerTextConfig,
        layer_index: usize,
        rope_traditional: bool,
        q_proj: LinearProj,
        k_proj: LinearProj,
        v_proj: LinearProj,
        o_proj: LinearProj,
        gate_proj: LinearProj,
    ) -> Result<Self> {
        let kind = config.layer_kinds[layer_index];
        let rope = config.rope_theta_for(layer_index).map(|theta| {
            RoPE::new(
                config.head_dim as i32,
                Some(rope_traditional),
                Some(theta as f64),
                None,
            )
        });
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            num_heads: config.num_attention_heads as i64,
            num_kv_heads: config.num_key_value_heads as i64,
            head_dim: config.head_dim as i64,
            qk_scale_factor: config.qk_scale_factor as f64,
            qk_norm_eps: config.rms_norm_eps,
            sliding_window: (kind == LayerKind::Sliding).then_some(
                i32::try_from(config.sliding_window).map_err(|_| {
                    Error::from_reason("Muse-Glimmer sliding window exceeds i32::MAX")
                })?,
            ),
            rope,
        })
    }

    fn scaleless_rms_norm(&self, x: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            mlx_sys::mlx_fast_rms_norm(x.as_raw_ptr(), std::ptr::null_mut(), self.qk_norm_eps)
        };
        MxArray::from_handle(handle, "muse_glimmer_qk_rms_norm")
    }

    pub fn forward(&self, x: &MxArray, cache: &mut Gemma4LayerCache) -> Result<MxArray> {
        let shape = x.shape()?;
        if shape.len() != 3 {
            return Err(Error::from_reason(format!(
                "Muse-Glimmer attention expects [B,T,H], got {:?}",
                shape.as_ref()
            )));
        }
        let (batch, seq_len) = (shape[0], shape[1]);
        let offset = cache.get_offset();

        let q =
            self.q_proj
                .forward(x)?
                .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k =
            self.k_proj
                .forward(x)?
                .reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;
        let v =
            self.v_proj
                .forward(x)?
                .reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;

        let q = self
            .scaleless_rms_norm(&q)?
            .mul_scalar(self.qk_scale_factor)?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let k = self
            .scaleless_rms_norm(&k)?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let v = v.transpose(Some(&[0, 2, 1, 3]))?;
        let (q, k) = match self.rope.as_ref() {
            Some(rope) => (
                rope.forward(&q, Some(offset))?,
                rope.forward(&k, Some(offset))?,
            ),
            None => (q, k),
        };
        let (k, v) = cache.update_and_fetch(&k, &v)?;

        let mask = if seq_len > 1 {
            let full = create_causal_mask(seq_len as i32, Some(offset), self.sliding_window)?;
            let kv_len = k.shape_at(2)?;
            let full_len = full.shape_at(1)?;
            Some(if kv_len < full_len {
                full.slice_axis(1, full_len - kv_len, full_len)?
            } else {
                full
            })
        } else {
            None
        };
        let attended = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            1.0 / (self.head_dim as f64).sqrt(),
            mask.as_ref(),
        )?
        .transpose(Some(&[0, 2, 1, 3]))?
        .reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;

        let gate = Activations::sigmoid(&self.gate_proj.forward(x)?)?;
        self.o_proj.forward(&attended.mul(&gate)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        paged_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
        window: PagedWindowSlot,
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        if shape.as_ref().len() != 3 || shape[0] != 1 {
            return Err(Error::from_reason(format!(
                "Muse-Glimmer paged attention expects [1,T,H], got {:?}",
                shape.as_ref()
            )));
        }
        let seq_len = shape[1];
        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[1, seq_len, self.num_heads, self.head_dim])?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[1, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[1, seq_len, self.num_kv_heads, self.head_dim])?;
        let q = self
            .scaleless_rms_norm(&q)?
            .mul_scalar(self.qk_scale_factor)?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let k = self
            .scaleless_rms_norm(&k)?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let v = v.transpose(Some(&[0, 2, 1, 3]))?;
        let (q, k) = match self.rope.as_ref() {
            Some(rope) => (
                rope.forward(&q, Some(first_logical_position as i32))?,
                rope.forward(&k, Some(first_logical_position as i32))?,
            ),
            None => (q, k),
        };
        let k_paged = k.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ])?;
        let v_paged = v.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ])?;
        if let Err(error) =
            adapter.update_keys_values_native(paged_idx, &k_paged, &v_paged, first_logical_position)
        {
            adapter
                .update_keys_values(paged_idx, &k_paged, &v_paged, first_logical_position)
                .map_err(|fallback| {
                    Error::from_reason(format!(
                        "Muse-Glimmer paged K/V write failed: {error}; fallback failed: {fallback}"
                    ))
                })?;
        }

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attended = if is_prefill {
            if window.carrier() != WindowCarrier::ExplicitMask {
                return Err(Error::from_reason(
                    "Muse-Glimmer paged prefill requires an admitted explicit-mask window slot",
                ));
            }
            let (keys, values) = if cached_prefix_len == 0 {
                (k, v)
            } else {
                let total_context = cached_prefix_len.saturating_add(seq_len as u32);
                let (keys, values, adapter_window) = adapter
                    .gather_kv_for_dense_cache_hit_prefill(paged_idx, total_context)
                    .map_err(Error::from_reason)?;
                let expected = window.mask_window().unwrap_or(0) as u32;
                if adapter_window.tokens() != expected {
                    return Err(Error::from_reason(format!(
                        "Muse-Glimmer paged prefill window mismatch: admitted {expected}, adapter {}",
                        adapter_window.tokens()
                    )));
                }
                (keys, values)
            };
            let mask = window
                .mask_window()
                .map(|width| {
                    create_causal_mask(
                        seq_len as i32,
                        (first_logical_position != 0).then_some(first_logical_position as i32),
                        Some(width),
                    )
                })
                .transpose()?;
            if mask.is_none() && cached_prefix_len == 0 {
                scaled_dot_product_attention_causal(&q, &keys, &values, scale)?
            } else {
                let mask = match mask {
                    Some(mask) => mask,
                    None => create_causal_mask(
                        seq_len as i32,
                        Some(first_logical_position as i32),
                        None,
                    )?,
                };
                scaled_dot_product_attention(&q, &keys, &values, scale, Some(&mask))?
            }
        } else {
            if window.carrier() != WindowCarrier::KernelArgument {
                return Err(Error::from_reason(
                    "Muse-Glimmer paged decode requires an admitted kernel-window slot",
                ));
            }
            let query = q.squeeze(Some(&[2]))?;
            adapter
                .gather_kv_for_decode_graph(paged_idx, &query, scale as f32, 1.0)
                .or_else(|_| {
                    adapter
                        .gather_kv_for_decode(paged_idx, &query, scale as f32, 1.0)
                        .map_err(Error::from_reason)
                })?
                .astype(x.dtype()?)?
                .reshape(&[1, self.num_heads, 1, self.head_dim])?
        };
        let attended = attended.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            1,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;
        let gate = Activations::sigmoid(&self.gate_proj.forward(x)?)?;
        self.o_proj.forward(&attended.mul(&gate)?)
    }

    pub(crate) fn forward_paged_batched(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        paged_idx: u32,
        rows: &[(SeqId, u32)],
        window: PagedWindowSlot,
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        if rows.is_empty()
            || shape.as_ref().len() != 3
            || shape[0] != rows.len() as i64
            || shape[1] != 1
        {
            return Err(Error::from_reason(format!(
                "Muse-Glimmer batched paged attention expects [N,1,H] for {} rows, got {:?}",
                rows.len(),
                shape.as_ref()
            )));
        }
        if window.carrier() != WindowCarrier::KernelArgument {
            return Err(Error::from_reason(
                "Muse-Glimmer batched decode requires an admitted kernel-window slot",
            ));
        }
        let batch = rows.len() as i64;
        let offsets = rows
            .iter()
            .map(|&(seq_id, position)| {
                i32::try_from(position).map_err(|_| {
                    Error::from_reason(format!(
                        "Muse-Glimmer sequence {seq_id} position {position} exceeds i32::MAX"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = MxArray::from_int32(&offsets, &[batch])?;
        let seq_ids = rows.iter().map(|&(seq_id, _)| seq_id).collect::<Vec<_>>();
        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[batch, 1, self.num_heads, self.head_dim])?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[batch, 1, self.num_kv_heads, self.head_dim])?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[batch, 1, self.num_kv_heads, self.head_dim])?;
        let q = self
            .scaleless_rms_norm(&q)?
            .mul_scalar(self.qk_scale_factor)?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let k = self
            .scaleless_rms_norm(&k)?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let v = v.transpose(Some(&[0, 2, 1, 3]))?;
        let (q, k) = match self.rope.as_ref() {
            Some(rope) => (
                rope.forward_with_offsets(&q, &offsets)?,
                rope.forward_with_offsets(&k, &offsets)?,
            ),
            None => (q, k),
        };
        let q = q.squeeze(Some(&[2]))?;
        let k = k.squeeze(Some(&[2]))?;
        let v = v.squeeze(Some(&[2]))?;
        adapter
            .update_keys_values_native_batched(paged_idx, &k, &v, rows)
            .map_err(Error::from_reason)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attended = adapter
            .gather_kv_for_decode_graph_batched(paged_idx, &q, &seq_ids, scale as f32, 1.0)
            .map_err(Error::from_reason)?
            .astype(x.dtype()?)?
            .reshape(&[batch, 1, self.num_heads * self.head_dim])?;
        let gate = Activations::sigmoid(&self.gate_proj.forward(x)?)?;
        self.o_proj.forward(&attended.mul(&gate)?)
    }
}
