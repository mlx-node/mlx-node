use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::cache::{DraftKvWindow, store_window_tail};
use crate::array::MxArray;
use crate::array::attention::scaled_dot_product_attention;
use crate::engine::dspark_turn::non_causal_sliding_mask;
use crate::models::quantized_linear::LinearProj;
use crate::models::qwen3_5::decoder_layer::DecoderLayer;
use crate::nn::{Activations, RMSNorm, RoPE};

/// The block's attention mask, keyed by what it depends on, which every
/// draft layer shares and which repeats from cycle to cycle once the window
/// is full: so it is built once and every layer of every such propose
/// reuses the same array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MaskKey {
    /// `(query_base - key_base) + i - j < window` over keys in logical
    /// order.
    Sliding {
        gap: i64,
        query_len: i64,
        key_len: i64,
        window: i64,
    },
    /// The ring's slots in physical order, then the block.
    Ring {
        slot: i64,
        live: i64,
        gap: i64,
        rows: i64,
        slots: i64,
        window: i64,
    },
}

pub(super) type SlidingMaskMemo = Option<(MaskKey, Option<MxArray>)>;

pub(super) fn memo_sliding_mask(
    memo: &mut SlidingMaskMemo,
    query_base: i32,
    query_len: i64,
    key_base: i32,
    key_len: i64,
    window: i64,
) -> Result<Option<MxArray>> {
    let key = MaskKey::Sliding {
        gap: i64::from(query_base) - i64::from(key_base),
        query_len,
        key_len,
        window,
    };
    if let Some((cached, mask)) = memo.as_ref()
        && *cached == key
    {
        return Ok(mask.clone());
    }
    // Relative positions: the same booleans as the absolute ones, for any
    // base the memo key is later matched at.
    let gap = i32::try_from(i64::from(query_base) - i64::from(key_base))
        .map_err(|_| Error::from_reason("DFlash2 sliding mask: query/key base gap overflows"))?;
    let mask = non_causal_sliding_mask(gap, query_len, 0, key_len, window)?;
    *memo = Some((key, mask.clone()));
    Ok(mask)
}

/// Splash `draft_attention_split_phase` over a wrapped ring: query row r
/// (position `query_base + r`) sees the ring's live rows `[start, end)` no
/// more than `window - 1` behind it, read in slot order, and every block
/// row. Row p sits at slot p % slots.
pub(super) fn memo_ring_mask(
    memo: &mut SlidingMaskMemo,
    query_base: i64,
    rows: i64,
    start: i64,
    end: i64,
    slots: i64,
    window: i64,
) -> Result<Option<MxArray>> {
    let key = MaskKey::Ring {
        slot: end % slots,
        live: end - start,
        gap: query_base - end,
        rows,
        slots,
        window,
    };
    if let Some((cached, mask)) = memo.as_ref()
        && *cached == key
    {
        return Ok(mask.clone());
    }
    let key_len = slots + rows;
    let mut visible = Vec::with_capacity((rows * key_len) as usize);
    for r in 0..rows {
        let query = query_base + r;
        for slot in 0..slots {
            let row = end - 1 - (end - 1 - slot).rem_euclid(slots);
            visible.push(u8::from(row >= start && row < end && query - row < window));
        }
        for j in 0..rows {
            visible.push(u8::from(query - (end + j) < window));
        }
    }
    let mask = MxArray::from_uint8(&visible, &[1, 1, rows, key_len])?
        .not_equal(&MxArray::from_uint8(&[0], &[1])?)?;
    *memo = Some((key, Some(mask.clone())));
    Ok(Some(mask))
}

pub(super) struct DFlash2Attention {
    pub(super) tensor_attention_enabled: bool,
    pub(super) q_proj: LinearProj,
    pub(super) k_proj: LinearProj,
    pub(super) v_proj: LinearProj,
    pub(super) o_proj: LinearProj,
    /// Row-merged `q|k|v` projection: one quantized matmul for the block
    /// where the three packed formats merge (`None` keeps three).
    pub(super) qkv_proj: Option<LinearProj>,
    pub(super) q_norm: RMSNorm,
    pub(super) k_norm: RMSNorm,
    pub(super) rope: RoPE,
    pub(super) num_heads: i64,
    pub(super) num_kv_heads: i64,
    pub(super) head_dim: i64,
}

impl DFlash2Attention {
    /// `rope(k_norm(keys))` as `[B, HK, T, D]` from `[B, T, HK * D]` rows.
    fn finish_keys(&self, keys: &MxArray, base: i32) -> Result<MxArray> {
        let keys = keys.reshape(&[
            keys.shape_at(0)?,
            keys.shape_at(1)?,
            self.num_kv_heads,
            self.head_dim,
        ])?;
        let keys = self.k_norm.forward(&keys)?.transpose(Some(&[0, 2, 1, 3]))?;
        self.rope.forward(&keys, Some(base))
    }

    /// `[B, HK, T, D]` values from `[B, T, HK * D]` rows.
    pub(super) fn finish_values(&self, values: &MxArray) -> Result<MxArray> {
        values
            .reshape(&[
                values.shape_at(0)?,
                values.shape_at(1)?,
                self.num_kv_heads,
                self.head_dim,
            ])?
            .transpose(Some(&[0, 2, 1, 3]))
    }

    pub(super) fn project_context(&self, x: &MxArray, base: i32) -> Result<(MxArray, MxArray)> {
        self.project_context_rows(x, base, None)
    }

    pub(super) fn project_context_rows(
        &self,
        x: &MxArray,
        base: i32,
        retained_rows: Option<i64>,
    ) -> Result<(MxArray, MxArray)> {
        let projected_keys = self.k_proj.forward(x)?;
        let key_prefix = retained_rows
            .map(|rows| projected_keys.slice_axis(1, 0, rows))
            .transpose()?;
        let keys = self.finish_keys(key_prefix.as_ref().unwrap_or(&projected_keys), base)?;
        let projected_values = self.v_proj.forward(x)?;
        let value_prefix = retained_rows
            .map(|rows| projected_values.slice_axis(1, 0, rows))
            .transpose()?;
        Ok((
            keys,
            self.finish_values(value_prefix.as_ref().unwrap_or(&projected_values))?,
        ))
    }

    /// `rope(q_norm(q))`, `rope(k_norm(k))` in one dispatch
    /// (`mlx_qk_norm_rope` on Metal / `mlx_qk_norm_rope_cuda` on CUDA, bit-identical
    /// to the four-op chain) from `[B, T, H, D]` inputs to `[B, H, T, D]`;
    /// `None` on a contract miss.
    #[cfg(test)]
    pub(super) fn fused_qk_norm_rope(
        &self,
        queries: &MxArray,
        keys: &MxArray,
        query_base: i32,
    ) -> Option<(MxArray, MxArray)> {
        if self.rope.traditional || self.q_norm.eps_f32() != self.k_norm.eps_f32() {
            return None;
        }
        let batch = queries.shape_at(0).ok()?;
        let offsets = MxArray::from_int32(&vec![query_base; batch as usize], &[batch]).ok()?;
        self.fused_qk_norm_rope_offsets(queries, keys, &offsets)
    }

    fn fused_qk_norm_rope_offsets(
        &self,
        queries: &MxArray,
        keys: &MxArray,
        offsets: &MxArray,
    ) -> Option<(MxArray, MxArray)> {
        if self.rope.traditional || self.q_norm.eps_f32() != self.k_norm.eps_f32() {
            return None;
        }
        crate::kernels::attention::qk_norm_rope_offsets(
            queries,
            keys,
            self.q_norm.weight(),
            self.k_norm.weight(),
            offsets,
            self.q_norm.eps_f32(),
            self.rope.base,
            self.rope.scale,
            self.rope.dims,
        )
        .ok()
        .flatten()
    }

    /// The block's roped queries `[B, HQ, T, D]` and its K/V rows
    /// `[B, HK, T, D]` at positions `query_base + t`. The merged `q|k|v`
    /// matmul serves up to `max_rows` rows (see `DFlash2Model::merged_rows`).
    pub(super) fn project_block(
        &self,
        x: &MxArray,
        query_base: i32,
        max_rows: i64,
    ) -> Result<(MxArray, MxArray, MxArray)> {
        let batch = x.shape_at(0)?;
        let offsets = MxArray::from_int32(&vec![query_base; batch as usize], &[batch])?;
        self.project_block_offsets(x, &offsets, max_rows)
    }

    fn project_block_offsets(
        &self,
        x: &MxArray,
        offsets: &MxArray,
        max_rows: i64,
    ) -> Result<(MxArray, MxArray, MxArray)> {
        let batch = x.shape_at(0)?;
        let seq = x.shape_at(1)?;
        let (queries, keys, values) = match self.qkv_proj.as_ref().filter(|_| seq <= max_rows) {
            Some(qkv_proj) => {
                let q_width = self.num_heads * self.head_dim;
                let kv_width = self.num_kv_heads * self.head_dim;
                let mut parts = qkv_proj
                    .forward(x)?
                    .split_sections(&[q_width, q_width + kv_width], -1)?
                    .into_iter();
                match (parts.next(), parts.next(), parts.next()) {
                    (Some(q), Some(k), Some(v)) => (q, k, v),
                    _ => return Err(Error::from_reason("DFlash2 q|k|v split arity")),
                }
            }
            None => (
                self.q_proj.forward(x)?,
                self.k_proj.forward(x)?,
                self.v_proj.forward(x)?,
            ),
        };
        let queries = queries.reshape(&[batch, seq, self.num_heads, self.head_dim])?;
        let keys = keys.reshape(&[batch, seq, self.num_kv_heads, self.head_dim])?;
        let (queries, keys) = match self.fused_qk_norm_rope_offsets(&queries, &keys, offsets) {
            Some(fused) => fused,
            None => {
                let queries = self
                    .q_norm
                    .forward(&queries)?
                    .transpose(Some(&[0, 2, 1, 3]))?;
                let keys = self.k_norm.forward(&keys)?.transpose(Some(&[0, 2, 1, 3]))?;
                (
                    self.rope.forward_with_offsets(&queries, offsets)?,
                    self.rope.forward_with_offsets(&keys, offsets)?,
                )
            }
        };
        Ok((queries, keys, self.finish_values(&values)?))
    }

    /// Sliding-window attention of `queries` `[B, HQ, T, D]` over `keys` /
    /// `values` `[B, HK, N, D]` whose first row sits at `key_base`.
    #[cfg(test)]
    pub(super) fn attend(
        &self,
        queries: &MxArray,
        keys: &MxArray,
        values: &MxArray,
        key_base: i32,
        query_base: i32,
        window: i64,
        masks: &mut SlidingMaskMemo,
    ) -> Result<MxArray> {
        let seq = queries.shape_at(2)?;
        let mask = memo_sliding_mask(masks, query_base, seq, key_base, keys.shape_at(2)?, window)?;
        self.attend_masked(queries, keys, values, mask.as_ref())
    }

    pub(super) fn attend_masked(
        &self,
        queries: &MxArray,
        keys: &MxArray,
        values: &MxArray,
        mask: Option<&MxArray>,
    ) -> Result<MxArray> {
        self.attend_masked_with_broadcast(queries, keys, values, mask, None)
    }

    // MLX's generic SDPA inserts a Broadcast with the construction-time
    // [B,H,Q,K] shape. During compiled replay, a pre-broadcast explicit input
    // makes that operation a no-op, allowing masked K to grow dynamically.
    // The custom tensor-op path keeps its singleton-head mask contract.
    pub(super) fn attend_masked_with_broadcast(
        &self,
        queries: &MxArray,
        keys: &MxArray,
        values: &MxArray,
        mask: Option<&MxArray>,
        broadcast_mask: Option<&MxArray>,
    ) -> Result<MxArray> {
        let batch = queries.shape_at(0)?;
        let seq = queries.shape_at(2)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attended = match crate::kernels::attention::sdpa_draft_masked(
            queries,
            keys,
            values,
            mask,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            seq,
            scale as f32,
            self.tensor_attention_enabled,
        )? {
            Some(out) => out,
            None => {
                scaled_dot_product_attention(queries, keys, values, scale, broadcast_mask.or(mask))?
            }
        }
        .transpose(Some(&[0, 2, 1, 3]))?
        .reshape(&[batch, seq, self.num_heads * self.head_dim])?;
        self.o_proj.forward(&attended)
    }
}

pub(super) struct DFlash2Mlp {
    pub(super) gate_proj: LinearProj,
    pub(super) up_proj: LinearProj,
    pub(super) down_proj: LinearProj,
    /// Row-merged `gate|up` projection and its split point (one quantized
    /// matmul where the two packed formats merge; `None` keeps two).
    pub(super) gate_up: Option<(LinearProj, i64)>,
}

impl DFlash2Mlp {
    pub(super) fn forward(&self, hidden: &MxArray, max_rows: i64) -> Result<MxArray> {
        self.down_proj.forward(&self.gated(hidden, max_rows)?)
    }

    pub(super) fn gated(&self, hidden: &MxArray, max_rows: i64) -> Result<MxArray> {
        let rows = hidden.shape_at(1)?;
        match self.gate_up.as_ref().filter(|_| rows <= max_rows) {
            Some((gate_up, split)) => {
                let parts = gate_up.forward(hidden)?.split_sections(&[*split], -1)?;
                Activations::swiglu_compiled(&parts[0], &parts[1])
            }
            None => Activations::swiglu_compiled(
                &self.gate_proj.forward(hidden)?,
                &self.up_proj.forward(hidden)?,
            ),
        }
    }
}

pub(super) struct GroupedDynamicCausalConv {
    pub(super) base_kernel: MxArray,
    pub(super) kernel_projection: LinearProj,
    pub(super) kernel_size: usize,
    pub(super) group_size: usize,
}

impl GroupedDynamicCausalConv {
    pub(super) fn finish_output(
        &self,
        hidden: &MxArray,
        dynamic: &MxArray,
        residual: &MxArray,
    ) -> Result<DraftLayerOutput> {
        Ok(DraftLayerOutput(
            residual.clone(),
            self.finish(hidden, dynamic)?,
        ))
    }

    /// Fused single-dispatch conv; `None` when the FFI rejects the inputs
    /// (non-Metal build, dtype mismatch, unusual dims) so the caller can run
    /// the elementwise chain instead.
    pub(super) fn fused_convolve(
        &self,
        hidden: &MxArray,
        dynamic: &MxArray,
        side: usize,
    ) -> Option<MxArray> {
        if !crate::accelerator::host_has_metal() && !crate::accelerator::cuda_ops_available() {
            return None;
        }
        let dtype = hidden.dtype().ok()?;
        if dynamic.dtype().ok()? != dtype || self.base_kernel.dtype().ok()? != dtype {
            return None;
        }
        let mut out = std::ptr::null_mut();
        let ok = unsafe {
            sys::mlx_dflash2_conv(
                hidden.as_raw_ptr(),
                dynamic.as_raw_ptr(),
                self.base_kernel.as_raw_ptr(),
                side as i32,
                &mut out,
            )
        };
        if !ok || out.is_null() {
            return None;
        }
        MxArray::from_handle(out, "dflash2_conv").ok()
    }

    /// `dynamic` is the full kernel-projection output `[B, L, 2, K, G]`;
    /// `side` selects which of the two tap blocks applies.
    fn convolve(&self, hidden: &MxArray, dynamic: &MxArray, side: usize) -> Result<MxArray> {
        if let Some(out) = self.fused_convolve(hidden, dynamic, side) {
            return Ok(out);
        }
        self.convolve_elementwise(hidden, dynamic, side)
    }

    pub(super) fn convolve_elementwise(
        &self,
        hidden: &MxArray,
        dynamic: &MxArray,
        side: usize,
    ) -> Result<MxArray> {
        let batch = hidden.shape_at(0)?;
        let length = hidden.shape_at(1)?;
        let hidden_size = hidden.shape_at(2)?;
        let groups = hidden_size / self.group_size as i64;
        let blocks = hidden.reshape(&[batch, length, groups, self.group_size as i64])?;
        let dynamic = dynamic
            .slice_axis(2, side as i64, side as i64 + 1)?
            .squeeze(Some(&[2]))?;
        let mut output = MxArray::zeros(blocks.shape()?.as_ref(), Some(hidden.dtype()?))?;
        for offset in 0..self.kernel_size {
            let values = if offset == 0 {
                blocks.clone()
            } else {
                blocks
                    .pad(&[0, 0, offset as i32, 0, 0, 0, 0, 0], 0.0)?
                    .slice_axis(1, 0, length)?
            };
            let base = self
                .base_kernel
                .slice(
                    &[side as i64, offset as i64, 0],
                    &[side as i64 + 1, offset as i64 + 1, hidden_size],
                )?
                .reshape(&[1, 1, groups, self.group_size as i64])?
                .astype(hidden.dtype()?)?;
            let correction = dynamic
                .slice(
                    &[0, 0, offset as i64, 0],
                    &[batch, length, offset as i64 + 1, groups],
                )?
                .reshape(&[batch, length, groups, 1])?;
            output = output.add(&base.add(&correction)?.mul(&values)?)?;
        }
        output.reshape(&[batch, length, hidden_size])
    }

    pub(super) fn prepare(&self, hidden: &MxArray) -> Result<(MxArray, MxArray)> {
        let batch = hidden.shape_at(0)?;
        let length = hidden.shape_at(1)?;
        let groups = hidden.shape_at(2)? / self.group_size as i64;
        // The full `[B, L, 2, K, G]` projection is carried into `finish` so the
        // fused conv can index both tap blocks without a materializing slice.
        let dynamic = self.kernel_projection.forward(hidden)?.reshape(&[
            batch,
            length,
            2,
            self.kernel_size as i64,
            groups,
        ])?;
        let prepared = self.convolve(hidden, &dynamic, 0)?;
        Ok((prepared, dynamic))
    }

    pub(super) fn finish(&self, hidden: &MxArray, dynamic: &MxArray) -> Result<MxArray> {
        self.convolve(hidden, dynamic, 1)
    }
}

pub(super) struct DFlash2Layer {
    pub(super) attention: DFlash2Attention,
    pub(super) mlp: DFlash2Mlp,
    pub(super) input_norm: RMSNorm,
    pub(super) post_attention_norm: RMSNorm,
    pub(super) attention_conv: GroupedDynamicCausalConv,
    pub(super) mlp_conv: GroupedDynamicCausalConv,
}

/// A layer's input: the materialized hidden state, or the previous layer's
/// residual and its MLP delta, whose sum is fused into this layer's input
/// norm.
pub(super) enum DraftResidual<'a> {
    Hidden(&'a MxArray),
    Pending(&'a MxArray, &'a MxArray),
}

/// Pending `(residual, delta)` left for the next layer's fused residual norm.
pub(super) struct DraftLayerOutput(MxArray, MxArray);

impl DraftLayerOutput {
    pub(super) fn as_input(&self) -> DraftResidual<'_> {
        DraftResidual::Pending(&self.0, &self.1)
    }

    pub(super) fn hidden_and_normalized(&self, norm: &RMSNorm) -> Result<(MxArray, MxArray)> {
        DecoderLayer::add_residual_norm(norm, &self.0, &self.1)
    }
}

impl DFlash2Layer {
    pub(super) fn forward_compiled(
        &self,
        input: DraftResidual<'_>,
        window: &[MxArray],
        offsets: &MxArray,
        mask: Option<&MxArray>,
        broadcast_mask: Option<&MxArray>,
        merged_rows: i64,
    ) -> Result<(DraftLayerOutput, MxArray, MxArray)> {
        let (residual, normed) = match input {
            DraftResidual::Hidden(hidden) => (hidden.clone(), self.input_norm.forward(hidden)?),
            DraftResidual::Pending(hidden, delta) => {
                DecoderLayer::add_residual_norm(&self.input_norm, hidden, delta)?
            }
        };
        let (prepared, dynamic) = self.attention_conv.prepare(&normed)?;
        let (queries, keys, values) =
            self.attention
                .project_block_offsets(&prepared, offsets, merged_rows)?;
        let stored = store_window_tail(window, &keys, &values)?;
        let attention = self.attention.attend_masked_with_broadcast(
            &queries,
            &stored[2],
            &stored[3],
            mask,
            broadcast_mask,
        )?;
        let (hidden, normed) = self
            .attention_conv
            .finish_output(&attention, &dynamic, &residual)?
            .hidden_and_normalized(&self.post_attention_norm)?;
        let (prepared, dynamic) = self.mlp_conv.prepare(&normed)?;
        let output = self.mlp_conv.finish_output(
            &self.mlp.forward(&prepared, merged_rows)?,
            &dynamic,
            &hidden,
        )?;
        Ok((output, stored[0].clone(), stored[1].clone()))
    }

    /// Returns either complete hidden state or a pending residual and delta
    /// for the next norm, according to the selected convolution arithmetic.
    pub(super) fn forward(
        &self,
        input: DraftResidual<'_>,
        window: &mut DraftKvWindow,
        query_base: i32,
        mask: Option<&MxArray>,
        merged_rows: i64,
    ) -> Result<DraftLayerOutput> {
        let (residual, normed) = match input {
            DraftResidual::Hidden(hidden) => (hidden.clone(), self.input_norm.forward(hidden)?),
            DraftResidual::Pending(hidden, delta) => {
                DecoderLayer::add_residual_norm(&self.input_norm, hidden, delta)?
            }
        };
        let (prepared, dynamic) = self.attention_conv.prepare(&normed)?;
        let (queries, block_keys, block_values) =
            self.attention
                .project_block(&prepared, query_base, merged_rows)?;
        let (keys, values) = window.with_block(&block_keys, &block_values)?;
        let attention = self
            .attention
            .attend_masked(&queries, &keys, &values, mask)?;
        let (hidden, normed) = self
            .attention_conv
            .finish_output(&attention, &dynamic, &residual)?
            .hidden_and_normalized(&self.post_attention_norm)?;
        let (prepared, dynamic) = self.mlp_conv.prepare(&normed)?;
        self.mlp_conv.finish_output(
            &self.mlp.forward(&prepared, merged_rows)?,
            &dynamic,
            &hidden,
        )
    }
}
