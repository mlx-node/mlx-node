//! External DFlash2 drafter for dense Qwen3.8 targets.
//!
//! DFlash2 consumes five post-layer target residual streams, runs a five-layer
//! parallel draft transformer, and chooses a Markov path through the top-16
//! token candidates at each position. The drafter shares the target embedding
//! and language-model head; its checkpoint therefore contains only the draft
//! transformer, two-tap grouped dynamic convolutions, and selector codebooks.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use mlx_sys as sys;
use napi::bindgen_prelude::*;
use rand::{Rng, RngExt};
use serde::Deserialize;

use crate::array::attention::scaled_dot_product_attention;
use crate::array::{DType, MxArray};
use crate::models::quantized_linear::{LinearProj, QuantizedLinear};
use crate::models::qwen3_5::decoder_layer::DecoderLayer;
use crate::nn::{Activations, Embedding, Linear, RMSNorm, RoPE};
use crate::sampling::{SparseDistribution, is_greedy_temperature};
use crate::utils::safetensors::load_safetensors_lazy;

#[derive(Clone, Debug, Deserialize)]
struct RawDFlashConfig {
    block_size: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    mask_token_id: usize,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
struct RawConfig {
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_target_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    max_position_embeddings: usize,
    sliding_window: usize,
    layer_types: Vec<String>,
    #[serde(default)]
    is_causal: bool,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,
    dflash_config: RawDFlashConfig,
}

#[derive(Clone, Debug)]
pub(crate) struct DFlash2Config {
    pub(crate) block_size: usize,
    pub(crate) mask_token_id: usize,
    pub(crate) target_layers: Vec<usize>,
    target_num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    max_position_embeddings: usize,
    pub(crate) sliding_window: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    selector_rank: usize,
    selector_top_k: usize,
    rope_theta: f64,
}

impl DFlash2Config {
    fn from_raw(raw: RawConfig) -> Result<Self> {
        if !raw
            .architectures
            .iter()
            .any(|name| name == "DFlash2DraftModel")
        {
            return Err(Error::from_reason(
                "DFlash2 checkpoint architectures must contain DFlash2DraftModel",
            ));
        }
        let draft = raw.dflash_config;
        let rope_theta = raw
            .rope_theta
            .or_else(|| {
                raw.rope_parameters
                    .as_ref()
                    .and_then(|value| value.get("rope_theta"))
                    .and_then(serde_json::Value::as_f64)
            })
            .unwrap_or(10_000.0);
        let valid = raw.hidden_size > 0
            && raw.intermediate_size > 0
            && raw.num_hidden_layers > 0
            && raw.num_attention_heads > 0
            && raw.num_key_value_heads > 0
            && raw
                .num_attention_heads
                .is_multiple_of(raw.num_key_value_heads)
            && raw.head_dim > 0
            && raw.vocab_size > 0
            && raw.sliding_window > 1
            && raw.layer_types.len() == raw.num_hidden_layers
            && raw
                .layer_types
                .iter()
                .all(|kind| kind == "sliding_attention")
            && !raw.is_causal
            && draft.block_size > 1
            && draft.conv_kernel_size == 2
            && draft.conv_group_size > 0
            && raw.hidden_size.is_multiple_of(draft.conv_group_size)
            && draft.selector_rank > 0
            && draft.selector_top_k > 0
            && draft.selector_top_k <= raw.vocab_size
            && !draft.target_layer_ids.is_empty()
            && raw.num_target_layers > 0
            && draft
                .target_layer_ids
                .iter()
                .all(|&layer| layer < raw.num_target_layers)
            && draft
                .target_layer_ids
                .windows(2)
                .all(|pair| pair[0] < pair[1]);
        if !valid {
            return Err(Error::from_reason(
                "unsupported DFlash2 configuration (requires non-causal sliding attention, two-tap grouped convolution, and a non-empty selector)",
            ));
        }
        Ok(Self {
            // z-lab names the total verify width (anchor + proposals)
            // `block_size`. The engine's DSpark width counts proposals only.
            block_size: draft.block_size - 1,
            mask_token_id: draft.mask_token_id,
            target_layers: draft.target_layer_ids,
            target_num_layers: raw.num_target_layers,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            vocab_size: raw.vocab_size,
            rms_norm_eps: raw.rms_norm_eps,
            max_position_embeddings: raw.max_position_embeddings,
            sliding_window: raw.sliding_window,
            conv_group_size: draft.conv_group_size,
            conv_kernel_size: draft.conv_kernel_size,
            selector_rank: draft.selector_rank,
            selector_top_k: draft.selector_top_k,
            rope_theta,
        })
    }
}

fn parallel_query_ids(anchor: u32, mask: usize, draft_len: usize) -> Vec<i32> {
    let mut ids = Vec::with_capacity(draft_len.saturating_add(1));
    ids.push(anchor as i32);
    ids.resize(draft_len.saturating_add(1), mask as i32);
    ids
}

fn non_causal_sliding_mask(
    query_base: i32,
    query_len: i64,
    key_base: i32,
    key_len: i64,
    window: i64,
) -> Result<Option<MxArray>> {
    if key_len <= window {
        return Ok(None);
    }
    let queries = MxArray::arange(
        f64::from(query_base),
        f64::from(query_base) + query_len as f64,
        None,
        None,
    )?
    .reshape(&[query_len, 1])?;
    let keys = MxArray::arange(
        f64::from(key_base),
        f64::from(key_base) + key_len as f64,
        None,
        None,
    )?
    .reshape(&[1, key_len])?;
    let window = MxArray::scalar_int(window as i32)?;
    Ok(Some(
        queries
            .sub(&keys)?
            .less(&window)?
            .reshape(&[1, 1, query_len, key_len])?,
    ))
}

/// The sliding mask is `(query_base - key_base) + i - j < window`: it depends
/// only on the base gap, `query_len`, `key_len` and `window`, which every
/// draft layer shares and which repeat from cycle to cycle once the window
/// is full, so it is built once and every layer of every such propose reuses
/// the same array.
type SlidingMaskMemo = Option<((i64, i64, i64, i64), Option<MxArray>)>;

fn memo_sliding_mask(
    memo: &mut SlidingMaskMemo,
    query_base: i32,
    query_len: i64,
    key_base: i32,
    key_len: i64,
    window: i64,
) -> Result<Option<MxArray>> {
    let key = (
        i64::from(query_base) - i64::from(key_base),
        query_len,
        key_len,
        window,
    );
    if let Some((cached, mask)) = memo.as_ref()
        && *cached == key
    {
        return Ok(mask.clone());
    }
    // Relative positions: the same booleans as the absolute ones, for any
    // base the memo key is later matched at.
    let gap = i32::try_from(key.0)
        .map_err(|_| Error::from_reason("DFlash2 sliding mask: query/key base gap overflows"))?;
    let mask = non_causal_sliding_mask(gap, query_len, 0, key_len, window)?;
    *memo = Some((key, mask.clone()));
    Ok(mask)
}

struct DFlash2Attention {
    q_proj: LinearProj,
    k_proj: LinearProj,
    v_proj: LinearProj,
    o_proj: LinearProj,
    /// Row-merged `q|k|v` projection: one quantized matmul for the block
    /// where the three packed formats merge (`None` keeps three).
    qkv_proj: Option<LinearProj>,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    rope: RoPE,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    sliding_window: i64,
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
    fn finish_values(&self, values: &MxArray) -> Result<MxArray> {
        values
            .reshape(&[
                values.shape_at(0)?,
                values.shape_at(1)?,
                self.num_kv_heads,
                self.head_dim,
            ])?
            .transpose(Some(&[0, 2, 1, 3]))
    }

    fn project_context(&self, x: &MxArray, base: i32) -> Result<(MxArray, MxArray)> {
        Ok((
            self.finish_keys(&self.k_proj.forward(x)?, base)?,
            self.finish_values(&self.v_proj.forward(x)?)?,
        ))
    }

    /// `rope(q_norm(q))`, `rope(k_norm(k))` in one dispatch
    /// (`mlx_qk_norm_rope`, bit-identical to the four-op chain) from
    /// `[B, T, H, D]` inputs to `[B, H, T, D]`; `None` on a contract miss.
    fn fused_qk_norm_rope(
        &self,
        queries: &MxArray,
        keys: &MxArray,
        query_base: i32,
    ) -> Option<(MxArray, MxArray)> {
        if !unsafe { sys::mlx_metal_is_available() }
            || self.rope.traditional
            || self.q_norm.eps_f32() != self.k_norm.eps_f32()
        {
            return None;
        }
        let batch = queries.shape_at(0).ok()?;
        let offsets = MxArray::from_int32(&vec![query_base; batch as usize], &[batch]).ok()?;
        let mut out_q = std::ptr::null_mut();
        let mut out_k = std::ptr::null_mut();
        // SAFETY: every handle is a live array for the call; the outputs are
        // owned handles or stay null when the call reports false.
        let ok = unsafe {
            sys::mlx_qk_norm_rope(
                queries.as_raw_ptr(),
                keys.as_raw_ptr(),
                self.q_norm.weight().as_raw_ptr(),
                self.k_norm.weight().as_raw_ptr(),
                offsets.as_raw_ptr(),
                self.q_norm.eps_f32(),
                self.rope.base,
                self.rope.scale,
                self.rope.dims,
                &mut out_q,
                &mut out_k,
            )
        };
        if !ok {
            return None;
        }
        Some((
            MxArray::from_handle(out_q, "dflash2 qk_norm_rope:q").ok()?,
            MxArray::from_handle(out_k, "dflash2 qk_norm_rope:k").ok()?,
        ))
    }

    /// The block's roped queries `[B, HQ, T, D]` and its K/V rows
    /// `[B, HK, T, D]` at positions `query_base + t`. The merged `q|k|v`
    /// matmul serves up to `max_rows` rows (see `DFlash2Model::merged_rows`).
    fn project_block(
        &self,
        x: &MxArray,
        query_base: i32,
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
        let (queries, keys) = match self.fused_qk_norm_rope(&queries, &keys, query_base) {
            Some(fused) => fused,
            None => {
                let queries = self
                    .q_norm
                    .forward(&queries)?
                    .transpose(Some(&[0, 2, 1, 3]))?;
                let keys = self.k_norm.forward(&keys)?.transpose(Some(&[0, 2, 1, 3]))?;
                (
                    self.rope.forward(&queries, Some(query_base))?,
                    self.rope.forward(&keys, Some(query_base))?,
                )
            }
        };
        Ok((queries, keys, self.finish_values(&values)?))
    }

    /// Sliding-window attention of `queries` `[B, HQ, T, D]` over `keys` /
    /// `values` `[B, HK, N, D]` whose first row sits at `key_base`.
    fn attend(
        &self,
        queries: &MxArray,
        keys: &MxArray,
        values: &MxArray,
        key_base: i32,
        query_base: i32,
        masks: &mut SlidingMaskMemo,
    ) -> Result<MxArray> {
        let batch = queries.shape_at(0)?;
        let seq = queries.shape_at(2)?;
        let mask = memo_sliding_mask(
            masks,
            query_base,
            seq,
            key_base,
            keys.shape_at(2)?,
            self.sliding_window,
        )?;
        let attended = scaled_dot_product_attention(
            queries,
            keys,
            values,
            1.0 / (self.head_dim as f64).sqrt(),
            mask.as_ref(),
        )?
        .transpose(Some(&[0, 2, 1, 3]))?
        .reshape(&[batch, seq, self.num_heads * self.head_dim])?;
        self.o_proj.forward(&attended)
    }
}

struct DFlash2Mlp {
    gate_proj: LinearProj,
    up_proj: LinearProj,
    down_proj: LinearProj,
    /// Row-merged `gate|up` projection and its split point (one quantized
    /// matmul where the two packed formats merge; `None` keeps two).
    gate_up: Option<(LinearProj, i64)>,
}

impl DFlash2Mlp {
    fn forward(&self, hidden: &MxArray, max_rows: i64) -> Result<MxArray> {
        let rows = hidden.shape_at(1)?;
        let gated = match self.gate_up.as_ref().filter(|_| rows <= max_rows) {
            Some((gate_up, split)) => {
                let parts = gate_up.forward(hidden)?.split_sections(&[*split], -1)?;
                Activations::swiglu_compiled(&parts[0], &parts[1])?
            }
            None => Activations::swiglu_compiled(
                &self.gate_proj.forward(hidden)?,
                &self.up_proj.forward(hidden)?,
            )?,
        };
        self.down_proj.forward(&gated)
    }
}

struct GroupedDynamicCausalConv {
    base_kernel: MxArray,
    kernel_projection: LinearProj,
    kernel_size: usize,
    group_size: usize,
}

impl GroupedDynamicCausalConv {
    /// Fused single-dispatch conv; `None` when the FFI rejects the inputs
    /// (non-Metal build, dtype mismatch, unusual dims) so the caller can run
    /// the elementwise chain instead.
    fn fused_convolve(&self, hidden: &MxArray, dynamic: &MxArray, side: usize) -> Option<MxArray> {
        // On non-Metal builds the FFI throws and this falls back anyway —
        // skip the throw/catch + stderr line per call (~4 per layer).
        if !unsafe { sys::mlx_metal_is_available() } {
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

    fn convolve_elementwise(
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

    fn prepare(&self, hidden: &MxArray) -> Result<(MxArray, MxArray)> {
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
        Ok((self.convolve(hidden, &dynamic, 0)?, dynamic))
    }

    fn finish(&self, hidden: &MxArray, dynamic: &MxArray) -> Result<MxArray> {
        self.convolve(hidden, dynamic, 1)
    }
}

struct DFlash2Layer {
    attention: DFlash2Attention,
    mlp: DFlash2Mlp,
    input_norm: RMSNorm,
    post_attention_norm: RMSNorm,
    attention_conv: GroupedDynamicCausalConv,
    mlp_conv: GroupedDynamicCausalConv,
}

/// A layer's input: the materialized hidden state, or the previous layer's
/// residual and its MLP delta, whose sum is fused into this layer's input
/// norm.
enum DraftResidual<'a> {
    Hidden(&'a MxArray),
    Pending(&'a MxArray, &'a MxArray),
}

impl DFlash2Layer {
    /// Returns this layer's post-attention residual and MLP delta; the caller
    /// adds them inside the next norm.
    fn forward(
        &self,
        input: DraftResidual<'_>,
        window: &mut DraftKvWindow,
        context_base: i32,
        query_base: i32,
        masks: &mut SlidingMaskMemo,
        merged_rows: i64,
    ) -> Result<(MxArray, MxArray)> {
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
        let key_base = if window.len() > 0 {
            context_base
        } else {
            query_base
        };
        let (keys, values) = window.with_block(&block_keys, &block_values)?;
        let attention = self
            .attention
            .attend(&queries, &keys, &values, key_base, query_base, masks)?;
        let (hidden, normed) = DecoderLayer::add_residual_norm(
            &self.post_attention_norm,
            &residual,
            &self.attention_conv.finish(&attention, &dynamic)?,
        )?;
        let (prepared, dynamic) = self.mlp_conv.prepare(&normed)?;
        let delta = self
            .mlp_conv
            .finish(&self.mlp.forward(&prepared, merged_rows)?, &dynamic)?;
        Ok((hidden, delta))
    }
}

struct CandidateSelector {
    predecessor_codebook: Embedding,
    successor_codebook: Embedding,
    hidden_projection: LinearProj,
    top_k: usize,
    rank: usize,
    vocab_size: usize,
}

fn normalized_selector_probs(scores: &[f32], temperature: f64) -> Result<Vec<f64>> {
    let scale = temperature.max(f64::MIN_POSITIVE);
    let max = scores
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(f32::total_cmp)
        .ok_or_else(|| Error::from_reason("DFlash2 selector scores are all non-finite"))?;
    let mut probs = scores
        .iter()
        .map(|&score| ((f64::from(score) - f64::from(max)) / scale).exp())
        .collect::<Vec<_>>();
    let total = probs.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::from_reason(
            "DFlash2 selector has no positive probability mass",
        ));
    }
    for prob in &mut probs {
        *prob /= total;
    }
    Ok(probs)
}

fn sample_selector_index<R: Rng + ?Sized>(probs: &[f64], rng: &mut R) -> usize {
    let draw = rng.random::<f64>();
    let mut cumulative = 0.0;
    let mut last = 0usize;
    for (index, &prob) in probs.iter().enumerate() {
        if prob <= 0.0 {
            continue;
        }
        cumulative += prob;
        last = index;
        if draw < cumulative {
            return index;
        }
    }
    last
}

/// The selector's drafted token path.
pub(crate) enum SelectorPath {
    /// Host-walked ids — sampled turns need the score rows on CPU anyway, so
    /// the conditional predecessor walk pays the same readback either way.
    Host(Vec<i32>),
    /// Device-resident greedy path `[L]` i32: the conditional predecessor
    /// walk ran inside the lazy graph (gather + argmax per position), so no
    /// per-cycle GPU→CPU readback gates the verify block on the proposal.
    Device(MxArray),
}

impl CandidateSelector {
    /// Sharded top-16 over the vocab via `mlx_dflash2_topk16`: two Metal
    /// dispatches instead of argpartition's full per-row merge sort. Returns
    /// (ids [1,L,16] i32, values [1,L,16] f32) ascending by value — the same
    /// contract the argpartition slice produces. `None` falls back to the
    /// generic path (non-16 K, non-Metal, shape/dtype mismatch).
    fn fused_topk16(&self, logits: &MxArray) -> Option<(MxArray, MxArray)> {
        if self.top_k != 16 || !unsafe { sys::mlx_metal_is_available() } {
            return None;
        }
        let shape = logits.shape().ok()?;
        if shape.len() != 3 || shape[0] != 1 {
            return None;
        }
        let mut ids = std::ptr::null_mut();
        let mut values = std::ptr::null_mut();
        if !unsafe { sys::mlx_dflash2_topk16(logits.as_raw_ptr(), &mut ids, &mut values) }
            || ids.is_null()
            || values.is_null()
        {
            return None;
        }
        let ids = MxArray::from_handle(ids, "dflash2_topk16:ids").ok()?;
        let values = MxArray::from_handle(values, "dflash2_topk16:values").ok()?;
        // The kernel emits f32 values (widened for the merge compare), while
        // the take_along_axis fallback returns logits.dtype(). bf16→f32 is
        // lossless, so casting back restores bit-parity: `scores`' add then
        // runs in the model dtype instead of promoting to f32 before the
        // final astype — a different rounding that flips near-tie picks.
        let values = values.astype(logits.dtype().ok()?).ok()?;
        let dims = [1, shape[1], self.top_k as i64];
        Some((ids.reshape(&dims).ok()?, values.reshape(&dims).ok()?))
    }

    /// One-dispatch version of the device-resident greedy predecessor walk.
    /// The Metal kernel scans each selected score row in reverse with strict
    /// `>`, exactly matching the lazy fallback's reversed MLX argmax: the last
    /// non-NaN maximum wins, NaNs do not win, and an all-NaN/-inf row selects
    /// the final slot. `None` keeps the existing lazy graph as the fallback.
    fn fused_greedy_path(&self, candidates: &MxArray, scores: &MxArray) -> Option<MxArray> {
        if !unsafe { sys::mlx_metal_is_available() } {
            return None;
        }
        let mut path = std::ptr::null_mut();
        if !unsafe {
            sys::mlx_dflash2_greedy_path(candidates.as_raw_ptr(), scores.as_raw_ptr(), &mut path)
        } || path.is_null()
        {
            return None;
        }
        MxArray::from_handle(path, "dflash2_greedy_path").ok()
    }

    /// Original lazy device walk, retained both as the contract fallback and
    /// as an independent parity oracle for the fused kernel.
    fn lazy_greedy_path(
        &self,
        candidates: &MxArray,
        scores: &MxArray,
        length: usize,
    ) -> Result<MxArray> {
        let mut predecessor = MxArray::from_int32(&[0], &[1])?;
        // Preserve the established device path's last-maximum rule by
        // reversing before argmax (which otherwise returns the first max).
        // This matches the host max_by walk for ordinary finite ties, but
        // host total_cmp distinguishes signed zero and ranks NaNs. MLX
        // argmax treats signed zeros as equal and, since upstream a124ac096,
        // returns the first NaN (here the last original NaN slot); the fused
        // kernel preserves these device-path semantics.
        let descending = MxArray::from_int32(
            &(0..self.top_k as i32).rev().collect::<Vec<_>>(),
            &[self.top_k as i64],
        )?;
        let last_index = MxArray::from_int32(&[self.top_k as i32 - 1], &[1])?;
        let mut tokens = Vec::with_capacity(length);
        for position in 0..length {
            let scores_i = scores.slice_axis(0, position as i64, position as i64 + 1)?;
            let row = scores_i.take(&predecessor, 1)?; // [1, 1, K]
            let selected = last_index
                .sub(
                    &row.take(&descending, -1)?
                        .argmax(-1, Some(false))?
                        .reshape(&[1])?,
                )?
                .astype(DType::Int32)?; // [1]
            let cand_i = candidates.slice_axis(1, position as i64, position as i64 + 1)?;
            tokens.push(cand_i.take(&selected, 2)?.reshape(&[1])?); // [1]
            predecessor = selected;
        }
        MxArray::concatenate_many(tokens.iter().collect(), Some(0))
    }

    fn select<R: Rng + ?Sized>(
        &self,
        hidden: &MxArray,
        logits: &MxArray,
        anchor: u32,
        temperature: f64,
        device_path: bool,
        rng: &mut R,
    ) -> Result<(SelectorPath, Vec<SparseDistribution>)> {
        let length = hidden.shape_at(1)? as usize;
        let (candidates, unary) = match self.fused_topk16(logits) {
            Some((ids, values)) => (ids, values),
            None => {
                let candidates = logits
                    .argpartition(-(self.top_k as i32), Some(-1))?
                    .slice_axis(
                        2,
                        self.vocab_size as i64 - self.top_k as i64,
                        self.vocab_size as i64,
                    )?;
                let unary = logits.take_along_axis(&candidates, -1)?;
                (candidates, unary)
            }
        };
        let projected = self.hidden_projection.forward(hidden)?.reshape(&[
            length as i64,
            1,
            self.rank as i64,
        ])?;
        let candidate_flat = candidates.reshape(&[(length * self.top_k) as i64])?;
        let successors = self
            .successor_codebook
            .forward(&candidate_flat)?
            .reshape(&[length as i64, self.top_k as i64, self.rank as i64])?;
        let anchor_ids = MxArray::from_int32(&[anchor as i32], &[1])?;
        let anchor_embedding = self
            .predecessor_codebook
            .forward(&anchor_ids)?
            .reshape(&[1, 1, self.rank as i64])?
            .broadcast_to(&[1, self.top_k as i64, self.rank as i64])?;
        let predecessor_ids = if length > 1 {
            candidates.slice_axis(1, 0, length as i64 - 1)?
        } else {
            MxArray::from_int32(&[], &[0, self.top_k as i64])?
        };
        let predecessors = if length > 1 {
            let previous = self
                .predecessor_codebook
                .forward(&predecessor_ids.reshape(&[((length - 1) * self.top_k) as i64])?)?
                .reshape(&[length as i64 - 1, self.top_k as i64, self.rank as i64])?;
            MxArray::concatenate(&anchor_embedding, &previous, 0)?
        } else {
            anchor_embedding
        };
        let edges = predecessors
            .mul(&projected)?
            .matmul(&successors.transpose(Some(&[0, 2, 1]))?)?;
        let scores = edges.add(&unary.reshape(&[length as i64, 1, self.top_k as i64])?)?;
        let candidates = candidates.astype(DType::Int32)?;
        let scores = scores.astype(DType::Float32)?;

        let greedy = is_greedy_temperature(temperature);
        if greedy && device_path {
            // One custom dispatch replaces the per-position gather + reverse
            // + argmax + gather chain. Contract misses retain that lazy graph,
            // and either result stays device-resident into target verify.
            let path = match self.fused_greedy_path(&candidates, &scores) {
                Some(path) => path,
                None => self.lazy_greedy_path(&candidates, &scores, length)?,
            };
            return Ok((SelectorPath::Device(path), Vec::new()));
        }

        MxArray::eval_arrays(&[&candidates, &scores])?;
        let candidate_ids = candidates.to_int32()?;
        let edge_scores = scores.to_float32()?;

        let mut path = Vec::with_capacity(length);
        let mut rows = Vec::with_capacity(if greedy { 0 } else { length });
        let mut predecessor_index = 0usize;
        for position in 0..length {
            let start = (position * self.top_k + predecessor_index) * self.top_k;
            let row = &edge_scores[start..start + self.top_k];
            let selected = if greedy {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(index, _)| index)
                    .unwrap_or(0)
            } else {
                let probs = normalized_selector_probs(row, temperature)?;
                let support_start = position * self.top_k;
                rows.push(SparseDistribution::from_parts(
                    candidate_ids[support_start..support_start + self.top_k].to_vec(),
                    probs.clone(),
                    self.vocab_size,
                )?);
                sample_selector_index(&probs, rng)
            };
            let token = candidate_ids[position * self.top_k + selected];
            path.push(token);
            predecessor_index = selected;
        }
        Ok((SelectorPath::Host(path), rows))
    }
}

/// Rows a draft window buffer grows by.
const DRAFT_KV_STEP: i64 = 256;

/// Whether draft projections of the given row widths may be row-merged (and
/// the merge sliced back into views): only Tiled64 K-quant/affine linears,
/// whose kernels decode and reduce each output row the same way at any
/// width, and whose layout concatenates and slices in whole 64-row tiles, so
/// every width must be a tile multiple. A projection still on MLX's affine
/// route (no Metal, or a shape the tiled kernels do not take) is not merged:
/// its per-row `qmv` / GEMM choice follows the width.
fn kquant_rows_merge(packed: &QuantizedLinear, widths: &[i64]) -> bool {
    use crate::models::quant_dispatch::{KQUANT_TILE_ROWS, split_kquant_layout};
    split_kquant_layout(packed.mode()).1 && widths.iter().all(|w| w % KQUANT_TILE_ROWS == 0)
}

/// One draft layer's attention context: the newest `window` rows of K and V
/// in temporal order, rows `[start, end)` of flat `[B, H, cap, D]` buffers.
/// Appends write their rows in place (`mlx_kv_store_rows`, one dispatch for
/// K and V) and the attention view is a slice, so neither the commit nor the
/// propose copies the window; the buffer is only reallocated when the rows
/// run past its end (every `DRAFT_KV_STEP` rows once the window is full).
pub(crate) struct DraftKvWindow {
    keys: Option<MxArray>,
    values: Option<MxArray>,
    start: i64,
    end: i64,
    window: i64,
    /// Rows kept free past `end` so a propose can place its block in the
    /// buffer without reallocating.
    reserve: i64,
}

impl DraftKvWindow {
    fn new(window: i64, reserve: i64) -> Self {
        Self {
            keys: None,
            values: None,
            start: 0,
            end: 0,
            window: window.max(1),
            reserve: reserve.max(0),
        }
    }

    /// Live rows.
    pub(crate) fn len(&self) -> i64 {
        self.end - self.start
    }

    fn capacity(&self) -> Result<i64> {
        self.keys.as_ref().map_or(Ok(0), |keys| keys.shape_at(2))
    }

    /// The window `[start, end)` as `[B, H, len, D]` views, `None` when empty.
    pub(crate) fn view(&self) -> Option<(MxArray, MxArray)> {
        if self.end == self.start {
            return None;
        }
        Some((
            self.keys
                .as_ref()?
                .slice_axis(2, self.start, self.end)
                .ok()?,
            self.values
                .as_ref()?
                .slice_axis(2, self.start, self.end)
                .ok()?,
        ))
    }

    /// Append `[B, H, T, D]` rows (any strides) after the window; only the
    /// newest `window` rows stay live.
    fn append(&mut self, keys: &MxArray, values: &MxArray) -> Result<()> {
        let mut rows = keys.shape_at(2)?;
        let tail;
        let (keys, values) = if rows > self.window {
            self.start = 0;
            self.end = 0;
            tail = (
                keys.slice_axis(2, rows - self.window, rows)?,
                values.slice_axis(2, rows - self.window, rows)?,
            );
            rows = self.window;
            (&tail.0, &tail.1)
        } else {
            (keys, values)
        };
        if self.end + rows + self.reserve > self.capacity()? {
            // Move the rows that stay live to the front of fresh buffers.
            let keep = self.len().min(self.window - rows);
            let needed = keep + rows + self.reserve;
            let capacity = ((needed + DRAFT_KV_STEP - 1) / DRAFT_KV_STEP * DRAFT_KV_STEP
                + DRAFT_KV_STEP)
                .min(self.window + self.reserve + DRAFT_KV_STEP)
                .max(needed);
            let fresh = |rows: &MxArray, old: Option<&MxArray>| -> Result<MxArray> {
                let mut buffer = MxArray::zeros(
                    &[
                        rows.shape_at(0)?,
                        rows.shape_at(1)?,
                        capacity,
                        rows.shape_at(3)?,
                    ],
                    Some(rows.dtype()?),
                )?;
                if let Some(old) = old
                    && keep > 0
                {
                    let live = old.slice_axis(2, self.end - keep, self.end)?;
                    buffer.slice_assign_axis_inplace(2, 0, keep, &live)?;
                }
                Ok(buffer)
            };
            let new_keys = fresh(keys, self.keys.as_ref())?;
            let new_values = fresh(values, self.values.as_ref())?;
            self.keys = Some(new_keys);
            self.values = Some(new_values);
            self.start = 0;
            self.end = keep;
        }
        let at = self.end;
        self.store_rows(at, keys, values)?;
        self.end += rows;
        self.start = self.start.max(self.end - self.window);
        Ok(())
    }

    /// The attention view for a block: the window followed by the block's
    /// `[B, H, T, D]` rows. The block lands in the buffer's free rows past
    /// `end` (in place, one dispatch) and the view is one slice; a block that
    /// does not fit, or an empty window, falls back to a concatenation. The
    /// written rows are dead to the window and are overwritten by the next
    /// append, whose store is ordered after this one by the shared handle.
    fn with_block(&mut self, keys: &MxArray, values: &MxArray) -> Result<(MxArray, MxArray)> {
        let rows = keys.shape_at(2)?;
        let Some((context_keys, context_values)) = self.view() else {
            return Ok((keys.clone(), values.clone()));
        };
        if self.end + rows > self.capacity()? {
            return Ok((
                MxArray::concatenate(&context_keys, keys, 2)?,
                MxArray::concatenate(&context_values, values, 2)?,
            ));
        }
        let at = self.end;
        self.store_rows(at, keys, values)?;
        let (Some(buffer_keys), Some(buffer_values)) = (&self.keys, &self.values) else {
            return Err(Error::from_reason("DFlash2 draft window lost its buffers"));
        };
        Ok((
            buffer_keys.slice_axis(2, self.start, self.end + rows)?,
            buffer_values.slice_axis(2, self.start, self.end + rows)?,
        ))
    }

    /// Write `keys` / `values` rows at `offset` in place and adopt the store
    /// primitive's handles (same buffers); without Metal, slice_update.
    fn store_rows(&mut self, offset: i64, keys: &MxArray, values: &MxArray) -> Result<()> {
        let rows = keys.shape_at(2)?;
        let (Some(buffer_keys), Some(buffer_values)) = (&self.keys, &self.values) else {
            return Err(Error::from_reason("DFlash2 draft window has no buffers"));
        };
        let dst = [buffer_keys.as_raw_ptr(), buffer_values.as_raw_ptr()];
        let src = [keys.as_raw_ptr(), values.as_raw_ptr()];
        let offsets = [offset as i32; 2];
        let mut out: [*mut sys::mlx_array; 2] = [std::ptr::null_mut(); 2];
        // SAFETY: every pointer is a live array handle for the call; `out`
        // receives owned handles (same buffers as `dst`) or stays null.
        let fused = unsafe { sys::mlx_metal_is_available() }
            && unsafe {
                sys::mlx_kv_store_rows(
                    2,
                    dst.as_ptr(),
                    src.as_ptr(),
                    offsets.as_ptr(),
                    out.as_mut_ptr(),
                )
            };
        if fused {
            self.keys = Some(MxArray::from_handle(out[0], "draft_kv_store:keys")?);
            self.values = Some(MxArray::from_handle(out[1], "draft_kv_store:values")?);
            return Ok(());
        }
        let (Some(buffer_keys), Some(buffer_values)) = (&mut self.keys, &mut self.values) else {
            return Err(Error::from_reason("DFlash2 draft window has no buffers"));
        };
        buffer_keys.slice_assign_axis_inplace(2, offset, offset + rows, keys)?;
        buffer_values.slice_assign_axis_inplace(2, offset, offset + rows, values)?;
        Ok(())
    }

    fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        out.extend(self.keys.as_ref());
        out.extend(self.values.as_ref());
    }
}

pub(crate) struct DFlash2ContextCache {
    layers: Vec<DraftKvWindow>,
    /// The sliding mask depends only on the window/block geometry, which
    /// repeats once the window is full, so it is built once and reused.
    mask_memo: SlidingMaskMemo,
    logical_len: i32,
    /// Exact token prefix represented by every layer cache above.
    ///
    /// A length alone is not cache provenance: an unrelated paged request can
    /// end at the same position. Keep the ids beside the draft K/V so a warm
    /// DFlash2 continuation is admitted only for the prefix that produced it.
    token_history: Vec<u32>,
}

impl DFlash2ContextCache {
    pub(crate) fn new(config: &DFlash2Config) -> Self {
        Self {
            layers: (0..config.num_hidden_layers)
                .map(|_| {
                    DraftKvWindow::new(
                        config.sliding_window as i64 - 1,
                        config.block_size as i64 + 1,
                    )
                })
                .collect(),
            mask_memo: None,
            logical_len: 0,
            token_history: Vec::new(),
        }
    }

    pub(crate) fn append(
        &mut self,
        model: &DFlash2Model,
        fused_context: &MxArray,
        base: i32,
        token_ids: &[u32],
    ) -> Result<()> {
        if base != self.logical_len {
            return Err(Error::from_reason(format!(
                "DFlash2 context append starts at {base}, expected {}",
                self.logical_len
            )));
        }
        if usize::try_from(self.logical_len).ok() != Some(self.token_history.len()) {
            return Err(Error::from_reason(format!(
                "DFlash2 context provenance length {} does not match logical length {}",
                self.token_history.len(),
                self.logical_len
            )));
        }
        let rows = fused_context.shape_at(1)? as usize;
        if rows != token_ids.len() {
            return Err(Error::from_reason(format!(
                "DFlash2 context append has {rows} hidden rows for {} token ids",
                token_ids.len()
            )));
        }
        for ((keys, values), window) in model
            .project_context(fused_context, base)?
            .iter()
            .zip(self.layers.iter_mut())
        {
            window.append(keys, values)?;
        }
        self.logical_len = self
            .logical_len
            .checked_add(i32::try_from(rows).map_err(|_| {
                Error::from_reason(format!(
                    "DFlash2 context append row count {rows} exceeds i32"
                ))
            })?)
            .ok_or_else(|| Error::from_reason("DFlash2 context length overflow"))?;
        self.token_history.extend_from_slice(token_ids);
        Ok(())
    }

    /// Advance logical length and token provenance WITHOUT writing draft
    /// K/V. Prefill uses this for rows older than the sliding window's
    /// retained tail: those rows would be evicted by later appends unread,
    /// so their fc fusion + per-layer K/V projections are dead work.
    pub(crate) fn record_only(&mut self, token_ids: &[u32]) -> Result<()> {
        self.logical_len = self
            .logical_len
            .checked_add(i32::try_from(token_ids.len()).map_err(|_| {
                Error::from_reason(format!(
                    "DFlash2 context skip row count {} exceeds i32",
                    token_ids.len()
                ))
            })?)
            .ok_or_else(|| Error::from_reason("DFlash2 context length overflow"))?;
        self.token_history.extend_from_slice(token_ids);
        Ok(())
    }

    pub(crate) fn logical_len(&self) -> i32 {
        self.logical_len
    }

    pub(crate) fn token_history(&self) -> &[u32] {
        &self.token_history
    }

    pub(crate) fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        for window in &self.layers {
            window.collect_arrays(out);
        }
    }

    pub(crate) fn eval(&self) -> Result<()> {
        let mut arrays = Vec::new();
        self.collect_arrays(&mut arrays);
        if arrays.is_empty() {
            Ok(())
        } else {
            MxArray::eval_arrays(&arrays)
        }
    }
}

pub(crate) struct DFlash2Model {
    pub(crate) config: DFlash2Config,
    pub(crate) weight_bytes: u64,
    fc: LinearProj,
    hidden_norm: RMSNorm,
    layers: Vec<DFlash2Layer>,
    norm: RMSNorm,
    selector: CandidateSelector,
    /// Every layer's `k|v` context projection row-merged (`k0|v0|k1|v1|…`):
    /// the accepted rows' draft K/V for all layers in one quantized matmul.
    /// `None` when the packed formats do not merge (per-layer projections).
    context_kv: Option<LinearProj>,
}

impl DFlash2Model {
    pub(crate) fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }

    pub(crate) fn validate_target(&self, target: &super::config::Qwen3_5Config) -> Result<()> {
        let invalid_tap = self
            .config
            .target_layers
            .iter()
            .copied()
            .find(|&layer| layer >= target.num_layers as usize);
        if self.config.hidden_size != target.hidden_size as usize
            || self.config.vocab_size != target.vocab_size as usize
            || self.config.mask_token_id >= self.config.vocab_size
            || self.config.target_num_layers != target.num_layers as usize
            || invalid_tap.is_some()
        {
            return Err(Error::from_reason(format!(
                "DFlash2/target mismatch: draft hidden={} vocab={} taps={:?}; target hidden={} vocab={} layers={}",
                self.config.hidden_size,
                self.config.vocab_size,
                self.config.target_layers,
                target.hidden_size,
                target.vocab_size,
                target.num_layers,
            )));
        }
        Ok(())
    }

    pub(crate) fn fuse_context(&self, taps: &[MxArray]) -> Result<MxArray> {
        if taps.len() != self.config.target_layers.len() || taps.is_empty() {
            return Err(Error::from_reason(format!(
                "DFlash2 expects {} target taps, got {}",
                self.config.target_layers.len(),
                taps.len()
            )));
        }
        let refs = taps.iter().collect::<Vec<_>>();
        self.hidden_norm.forward(
            &self
                .fc
                .forward(&MxArray::concatenate_many(refs, Some(2))?)?,
        )
    }

    /// Row-merge the projections that share an input so each runs as one
    /// quantized matmul: `q|k|v` and `gate|up` per layer, and every layer's
    /// `k|v` for the context append. The merge needs K-quant projections
    /// whose widths are whole Tiled64 tiles ([`kquant_rows_merge`]) and
    /// agreeing packed formats; otherwise the separate projections stay.
    /// Below the GEMM threshold the tiled kernels decode and reduce each
    /// output row the same way at any width except the M = 8 tensor-op
    /// route's split-K count, which follows N (`qmm_m8_nax_splits`), so a
    /// merged block of 8 rows may differ from the separate projections in
    /// the last bf16 bit. Per-layer merges turn the originals into row views
    /// of the merged buffer (no extra bytes); the cross-layer `k|v` merge is
    /// an extra resident copy, counted in `weight_bytes`.
    fn merge_projections(&mut self) -> Result<()> {
        let merge_rows = |first: &LinearProj, widths: &[i64]| -> bool {
            match first {
                LinearProj::Quantized(packed) => kquant_rows_merge(packed, widths),
                LinearProj::Standard(_) => false,
            }
        };
        for layer in &mut self.layers {
            let attention = &mut layer.attention;
            let q_rows = attention.num_heads * attention.head_dim;
            let kv_rows = attention.num_kv_heads * attention.head_dim;
            if merge_rows(&attention.q_proj, &[q_rows, kv_rows])
                && let Some(qk) = attention.q_proj.concat_rows(&attention.k_proj)?
                && let Some(qkv) = qk.concat_rows(&attention.v_proj)?
            {
                attention.q_proj = qkv.slice_rows(0, q_rows)?;
                attention.k_proj = qkv.slice_rows(q_rows, q_rows + kv_rows)?;
                attention.v_proj = qkv.slice_rows(q_rows + kv_rows, q_rows + 2 * kv_rows)?;
                attention.qkv_proj = Some(qkv);
            }
            let mlp = &mut layer.mlp;
            let split = self.config.intermediate_size as i64;
            if merge_rows(&mlp.gate_proj, &[split])
                && let Some(gate_up) = mlp.gate_proj.concat_rows(&mlp.up_proj)?
            {
                mlp.gate_proj = gate_up.slice_rows(0, split)?;
                mlp.up_proj = gate_up.slice_rows(split, 2 * split)?;
                mlp.gate_up = Some((gate_up, split));
            }
        }
        let Some(first) = self.layers.first() else {
            return Ok(());
        };
        let kv_rows = first.attention.num_kv_heads * first.attention.head_dim;
        if !merge_rows(&first.attention.k_proj, &[kv_rows]) {
            return Ok(());
        }
        let mut merged: Option<LinearProj> = None;
        for layer in &self.layers {
            for proj in [&layer.attention.k_proj, &layer.attention.v_proj] {
                merged = match merged {
                    None => match proj {
                        LinearProj::Quantized(_) => {
                            Some(proj.slice_rows(0, proj.packed_out_features()?)?)
                        }
                        LinearProj::Standard(_) => return Ok(()),
                    },
                    Some(acc) => match acc.concat_rows(proj)? {
                        Some(next) => Some(next),
                        None => return Ok(()),
                    },
                };
            }
        }
        if let Some(LinearProj::Quantized(packed)) = &merged {
            let extra = [
                Some(packed.get_weight()),
                Some(packed.get_scales()),
                packed.get_biases(),
            ]
            .into_iter()
            .flatten()
            .map(|a| a.nbytes() as u64)
            .sum::<u64>();
            self.weight_bytes = self.weight_bytes.saturating_add(extra);
        }
        self.context_kv = merged;
        Ok(())
    }

    /// Row heights the merged projections serve: a decode block (anchor +
    /// proposals), the heights the merges were made for. Taller (prefill)
    /// inputs take the separate projections, whose GEMM split-K then follows
    /// each projection's own width.
    fn merged_rows(&self) -> i64 {
        self.config.block_size as i64 + 1
    }

    /// The accepted rows' draft K/V for every layer: `(keys, values)` per
    /// layer as `[B, HK, T, D]`, keys normed and roped at `base + t`. With
    /// the merged projection this is one quantized matmul, one k_norm per
    /// layer and one RoPE over all layers' heads; otherwise per layer.
    fn project_context(
        &self,
        fused_context: &MxArray,
        base: i32,
    ) -> Result<Vec<(MxArray, MxArray)>> {
        let Some(context_kv) = self.context_kv.as_ref().filter(|_| {
            fused_context
                .shape_at(1)
                .is_ok_and(|rows| rows <= self.merged_rows())
        }) else {
            return self
                .layers
                .iter()
                .map(|layer| layer.attention.project_context(fused_context, base))
                .collect();
        };
        let Some(first) = self.layers.first() else {
            return Ok(Vec::new());
        };
        let kv_width = first.attention.num_kv_heads * first.attention.head_dim;
        let bounds = (1..2 * self.layers.len() as i64)
            .map(|i| i * kv_width)
            .collect::<Vec<_>>();
        let parts = context_kv
            .forward(fused_context)?
            .split_sections(&bounds, -1)?;
        let mut keys = Vec::with_capacity(self.layers.len());
        let mut values = Vec::with_capacity(self.layers.len());
        for (layer, pair) in self.layers.iter().zip(parts.as_chunks::<2>().0) {
            let attention = &layer.attention;
            let normed = attention.k_norm.forward(&pair[0].reshape(&[
                pair[0].shape_at(0)?,
                pair[0].shape_at(1)?,
                attention.num_kv_heads,
                attention.head_dim,
            ])?)?;
            keys.push(normed.transpose(Some(&[0, 2, 1, 3]))?);
            values.push(attention.finish_values(&pair[1])?);
        }
        let roped = first.attention.rope.forward(
            &MxArray::concatenate_many(keys.iter().collect(), Some(1))?,
            Some(base),
        )?;
        let head_bounds = (1..self.layers.len() as i64)
            .map(|i| i * first.attention.num_kv_heads)
            .collect::<Vec<_>>();
        Ok(roped
            .split_sections(&head_bounds, 1)?
            .into_iter()
            .zip(values)
            .collect())
    }

    fn forward_hidden(
        &self,
        target_embedding: &Embedding,
        block_ids: &MxArray,
        query_base: i32,
        context: &mut DFlash2ContextCache,
    ) -> Result<MxArray> {
        let embedded = target_embedding.forward(block_ids)?;
        let mut pending: Option<(MxArray, MxArray)> = None;
        let DFlash2ContextCache {
            layers: windows,
            mask_memo,
            logical_len,
            ..
        } = context;
        for (layer, window) in self.layers.iter().zip(windows.iter_mut()) {
            let context_base = logical_len.saturating_sub(window.len() as i32);
            let input = match &pending {
                Some((hidden, delta)) => DraftResidual::Pending(hidden, delta),
                None => DraftResidual::Hidden(&embedded),
            };
            let next = layer.forward(
                input,
                window,
                context_base,
                query_base,
                mask_memo,
                self.merged_rows(),
            )?;
            pending = Some(next);
        }
        match &pending {
            Some((hidden, delta)) => {
                Ok(DecoderLayer::add_residual_norm(&self.norm, hidden, delta)?.1)
            }
            None => self.norm.forward(&embedded),
        }
    }

    pub(crate) fn propose<R: Rng + ?Sized>(
        &self,
        target_embedding: &Embedding,
        target_lm_head: Option<&LinearProj>,
        context: &mut DFlash2ContextCache,
        anchor: u32,
        max_len: usize,
        temperature: f64,
        device_path: bool,
        rng: &mut R,
    ) -> Result<(SelectorPath, Vec<SparseDistribution>)> {
        let query_end = context
            .logical_len
            .saturating_add(max_len.saturating_add(1) as i32);
        if query_end as usize > self.config.max_position_embeddings {
            return Err(Error::from_reason(format!(
                "DFlash2 query end {query_end} exceeds trained context {}",
                self.config.max_position_embeddings
            )));
        }
        let ids = parallel_query_ids(anchor, self.config.mask_token_id, max_len);
        let phase_time = std::env::var("MLX_DFLASH2_PHASE_TIME").is_ok();
        let t0 = std::time::Instant::now();
        let block = MxArray::from_int32(&ids, &[1, ids.len() as i64])?;
        let query_base = context.logical_len;
        let hidden = self.forward_hidden(target_embedding, &block, query_base, context)?;
        // The head runs on the whole block (anchor + proposals, 8 rows), the
        // height its K-quant M = 8 route serves, and the anchor row's logits
        // are dropped: at 7 rows the head took the per-row kernel and cost
        // more than at 8.
        let logits = match target_lm_head {
            Some(head) => head.forward(&hidden)?,
            None => target_embedding.as_linear(&hidden)?,
        };
        let rows = max_len as i64 + 1;
        let hidden = hidden.slice_axis(1, 1, rows)?;
        let logits = logits.slice_axis(1, 1, rows)?;
        let out = self
            .selector
            .select(&hidden, &logits, anchor, temperature, device_path, rng)?;
        if phase_time {
            eprintln!("[dflash2-phase] draft-build: {:?}", t0.elapsed());
            let t = std::time::Instant::now();
            if let SelectorPath::Device(ids) = &out.0 {
                MxArray::eval_arrays(&[ids])?;
            } else {
                MxArray::eval_arrays(&[&logits])?;
            }
            eprintln!("[dflash2-phase] draft+selector: {:?}", t.elapsed());
        }
        Ok(out)
    }
}

#[cfg(test)]
pub(crate) fn tiny_dflash2_model_for_stepper_test(
    target: &super::config::Qwen3_5Config,
) -> Result<DFlash2Model> {
    let hidden = target.hidden_size as usize;
    let head_dim = target.head_dim as usize;
    let config = DFlash2Config {
        block_size: 2,
        mask_token_id: 0,
        target_layers: vec![1, 3],
        target_num_layers: target.num_layers as usize,
        hidden_size: hidden,
        intermediate_size: hidden * 2,
        num_hidden_layers: 1,
        num_attention_heads: target.num_heads as usize,
        num_key_value_heads: target.num_kv_heads as usize,
        head_dim,
        vocab_size: target.vocab_size as usize,
        rms_norm_eps: target.rms_norm_eps,
        max_position_embeddings: target.max_position_embeddings as usize,
        sliding_window: 8,
        conv_group_size: head_dim,
        conv_kernel_size: 2,
        selector_rank: 8,
        selector_top_k: 4,
        rope_theta: target.rope_theta,
    };
    if config
        .target_layers
        .iter()
        .any(|&layer| layer >= config.target_num_layers)
        || !hidden.is_multiple_of(config.conv_group_size)
        || config.vocab_size < config.selector_top_k
    {
        return Err(Error::from_reason(
            "tiny DFlash2 stepper fixture is incompatible with the target config",
        ));
    }

    let linear = |input: usize, output: usize| -> Result<LinearProj> {
        let mut layer = Linear::new(input as u32, output as u32, Some(false))?;
        layer.set_weight(&MxArray::random_normal(
            &[output as i64, input as i64],
            0.0,
            0.02,
            Some(DType::BFloat16),
        )?)?;
        Ok(LinearProj::Standard(layer))
    };
    let norm = |size: usize| -> Result<RMSNorm> {
        let mut layer = RMSNorm::new(size as u32, Some(config.rms_norm_eps))?;
        layer.set_weight(&MxArray::ones(&[size as i64], Some(DType::BFloat16))?)?;
        Ok(layer)
    };
    let embedding = |rows: usize, width: usize| -> Result<Embedding> {
        let mut layer = Embedding::new(rows as u32, width as u32)?;
        layer.load_weight(&MxArray::random_normal(
            &[rows as i64, width as i64],
            0.0,
            0.02,
            Some(DType::BFloat16),
        )?)?;
        Ok(layer)
    };
    let grouped_conv = || -> Result<GroupedDynamicCausalConv> {
        Ok(GroupedDynamicCausalConv {
            base_kernel: MxArray::random_normal(
                &[2, config.conv_kernel_size as i64, hidden as i64],
                0.0,
                0.02,
                Some(DType::BFloat16),
            )?,
            kernel_projection: linear(
                hidden,
                2 * config.conv_kernel_size * (hidden / config.conv_group_size),
            )?,
            kernel_size: config.conv_kernel_size,
            group_size: config.conv_group_size,
        })
    };
    let attention = DFlash2Attention {
        q_proj: linear(hidden, config.num_attention_heads * head_dim)?,
        k_proj: linear(hidden, config.num_key_value_heads * head_dim)?,
        v_proj: linear(hidden, config.num_key_value_heads * head_dim)?,
        o_proj: linear(config.num_attention_heads * head_dim, hidden)?,
        qkv_proj: None,
        q_norm: norm(head_dim)?,
        k_norm: norm(head_dim)?,
        rope: RoPE::new(head_dim as i32, Some(false), Some(config.rope_theta), None),
        num_heads: config.num_attention_heads as i64,
        num_kv_heads: config.num_key_value_heads as i64,
        head_dim: head_dim as i64,
        sliding_window: config.sliding_window as i64,
    };
    let layer = DFlash2Layer {
        attention,
        mlp: DFlash2Mlp {
            gate_proj: linear(hidden, config.intermediate_size)?,
            up_proj: linear(hidden, config.intermediate_size)?,
            down_proj: linear(config.intermediate_size, hidden)?,
            gate_up: None,
        },
        input_norm: norm(hidden)?,
        post_attention_norm: norm(hidden)?,
        attention_conv: grouped_conv()?,
        mlp_conv: grouped_conv()?,
    };
    Ok(DFlash2Model {
        fc: linear(hidden * config.target_layers.len(), hidden)?,
        hidden_norm: norm(hidden)?,
        norm: norm(hidden)?,
        selector: CandidateSelector {
            predecessor_codebook: embedding(config.vocab_size, config.selector_rank)?,
            successor_codebook: embedding(config.vocab_size, config.selector_rank)?,
            hidden_projection: linear(hidden, config.selector_rank)?,
            top_k: config.selector_top_k,
            rank: config.selector_rank,
            vocab_size: config.vocab_size,
        },
        layers: vec![layer],
        config,
        weight_bytes: 0,
        context_kv: None,
    })
}

fn required(params: &mut HashMap<String, MxArray>, key: &str, shape: &[i64]) -> Result<MxArray> {
    let value = params
        .remove(key)
        .ok_or_else(|| Error::from_reason(format!("DFlash2 checkpoint is missing '{key}'")))?;
    validate_tensor(&value, key, shape)?;
    Ok(value)
}

fn validate_tensor(value: &MxArray, key: &str, shape: &[i64]) -> Result<()> {
    if value.shape()?.as_ref() != shape {
        return Err(Error::from_reason(format!(
            "DFlash2 tensor '{key}' has shape {:?}, expected {shape:?}",
            value.shape()?.as_ref()
        )));
    }
    let actual = value.dtype()?;
    if !matches!(actual, DType::Float16 | DType::BFloat16 | DType::Float32) {
        return Err(Error::from_reason(format!(
            "DFlash2 tensor '{key}' has dtype {actual:?}, expected floating point"
        )));
    }
    Ok(())
}

/// The published DFlash2 companion ships bf16. Its dense projections load as
/// affine Q4/group64: against bf16 and Q8 it kept teacher-forced acceptance
/// within noise and gave the lowest decode time per committed token. On a
/// Metal host the packed arrays are then tiled into the K-quant `a4g64@t64`
/// contract (same codes, same bf16 scale and bias per group), so a decode
/// block's 8 rows take the tensor-op `qmm_m8_nax_t64` kernel instead of
/// MLX's per-row affine `qmv` (which read the weights once per row: ~2.2x
/// the bandwidth floor at M = 8). The draft reuses the target output head.
/// A change of draft precision moves proposals and verify grouping, so it
/// can change the transcript even though the target verifies every emitted
/// token.
const DRAFT_GROUP_SIZE: i32 = 64;
const DRAFT_BITS: i32 = 4;

/// Affine-quantize a floating `[out, in]` weight, returning MLX's packed
/// `(weight, scales, biases)` triple consumed by `quantized_matmul`.
fn quantize_affine(
    weight: &MxArray,
    group_size: i32,
    bits: i32,
) -> Result<(MxArray, MxArray, MxArray)> {
    let mut out_q = std::ptr::null_mut();
    let mut out_s = std::ptr::null_mut();
    let mut out_b = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_quantize(
            weight.as_raw_ptr(),
            group_size,
            bits,
            c"affine".as_ptr(),
            &mut out_q,
            &mut out_s,
            &mut out_b,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_quantize(affine) failed on a DFlash2 draft weight",
        ));
    }
    let packed = MxArray::from_handle(out_q, "dflash2 quantized weight")?;
    let scales = MxArray::from_handle(out_s, "dflash2 quantization scales")?;
    let biases = MxArray::from_handle(out_b, "dflash2 quantization biases")?;
    // mlx_quantize outputs are lazy: left unevaluated they pin the dense bf16
    // source in the graph until the first draft forward, so residency sizing
    // would run against ~3.6 GB of still-live dense weights and the first
    // proposal would pay the whole quantize cost. Evaluating here releases
    // each dense source when the caller drops it.
    packed.eval();
    scales.eval();
    biases.eval();
    Ok((packed, scales, biases))
}

/// Quantize a floating `[out, in]` weight to affine Q4/group64 as a
/// `QuantizedLinear`, tiled into the `a4g64@t64` K-quant contract when this
/// host runs the `_t64` kernels ([`QuantizedLinear::tile_kquant_layout`];
/// the row-major arrays are released). The shape must be tileable
/// (`N % 64 == 0`, `K % 256 == 0`) on every host: an odd draft geometry
/// fails here rather than silently taking a slower layout.
fn quantize_draft_weight(weight: &MxArray, name: &str) -> Result<QuantizedLinear> {
    use crate::models::quant_dispatch::{kquant_tileable, kquant_tiled_enabled};
    let shape = weight.shape()?;
    let (rows, k) = (shape[0], shape[1]);
    if !kquant_tileable(rows, k) {
        return Err(Error::from_reason(format!(
            "DFlash2 draft projection '{name}' is [{rows}, {k}]; the a{DRAFT_BITS}g{DRAFT_GROUP_SIZE} \
             Tiled64 layout needs N % 64 == 0 and K % 256 == 0"
        )));
    }
    let (packed, scales, biases) = quantize_affine(weight, DRAFT_GROUP_SIZE, DRAFT_BITS)?;
    let mut linear = QuantizedLinear::new(
        packed,
        scales,
        Some(biases),
        None,
        DRAFT_GROUP_SIZE,
        DRAFT_BITS,
        "affine".to_string(),
    );
    if kquant_tiled_enabled() && !linear.tile_kquant_layout()? {
        return Err(Error::from_reason(format!(
            "DFlash2 draft projection '{name}' [{rows}, {k}] did not tile"
        )));
    }
    Ok(linear)
}

/// Builds a draft projection as affine Q4/group64. `savings` accumulates
/// `dense − resident` bytes so the loader can report true residency rather
/// than the bf16 file size.
fn draft_linear(
    params: &mut HashMap<String, MxArray>,
    prefix: &str,
    input: usize,
    output: usize,
    savings: &mut u64,
) -> Result<LinearProj> {
    let key = format!("{prefix}.weight");
    let weight = required(params, &key, &[output as i64, input as i64])?;
    let linear = quantize_draft_weight(&weight, &key)?;
    let resident = linear.get_weight().nbytes() as u64
        + linear.get_scales().nbytes() as u64
        + linear.get_biases().map_or(0, |b| b.nbytes() as u64);
    *savings += (weight.nbytes() as u64).saturating_sub(resident);
    Ok(LinearProj::Quantized(linear))
}

fn dense_linear(
    params: &mut HashMap<String, MxArray>,
    prefix: &str,
    input: usize,
    output: usize,
) -> Result<LinearProj> {
    let weight = required(
        params,
        &format!("{prefix}.weight"),
        &[output as i64, input as i64],
    )?;
    Ok(LinearProj::Standard(Linear::from_weights(&weight, None)?))
}

fn norm(
    params: &mut HashMap<String, MxArray>,
    key: &str,
    size: usize,
    eps: f64,
) -> Result<RMSNorm> {
    RMSNorm::from_weight(&required(params, key, &[size as i64])?, Some(eps))
}

fn codebook(
    params: &mut HashMap<String, MxArray>,
    key: &str,
    vocab: usize,
    rank: usize,
) -> Result<Embedding> {
    let weight = required(params, key, &[vocab as i64, rank as i64])?;
    let mut embedding = Embedding::new_uninitialized(vocab as u32, rank as u32)?;
    embedding.load_weight(&weight)?;
    Ok(embedding)
}

fn grouped_conv(
    params: &mut HashMap<String, MxArray>,
    base: &str,
    name: &str,
    config: &DFlash2Config,
    savings: &mut u64,
) -> Result<GroupedDynamicCausalConv> {
    let hidden = config.hidden_size;
    let groups = hidden / config.conv_group_size;
    Ok(GroupedDynamicCausalConv {
        base_kernel: required(
            params,
            &format!("{base}.{name}.base_kernel"),
            &[2, config.conv_kernel_size as i64, hidden as i64],
        )?,
        kernel_projection: draft_linear(
            params,
            &format!("{base}.{name}.kernel_projection"),
            hidden,
            2 * config.conv_kernel_size * groups,
            savings,
        )?,
        kernel_size: config.conv_kernel_size,
        group_size: config.conv_group_size,
    })
}

fn resolve_safetensors(path: &Path) -> Result<PathBuf> {
    if path.join("model.safetensors").is_file() {
        return Ok(path.join("model.safetensors"));
    }
    Err(Error::from_reason(format!(
        "DFlash2 checkpoint {} is missing model.safetensors",
        path.display()
    )))
}

fn expected_tensor_shapes(config: &DFlash2Config) -> Vec<(String, Vec<i64>)> {
    let hidden = config.hidden_size as i64;
    let intermediate = config.intermediate_size as i64;
    let groups = config.hidden_size / config.conv_group_size;
    let mut expected = vec![
        (
            "fc.weight".to_string(),
            vec![hidden, hidden * config.target_layers.len() as i64],
        ),
        ("hidden_norm.weight".to_string(), vec![hidden]),
        ("norm.weight".to_string(), vec![hidden]),
    ];
    for index in 0..config.num_hidden_layers {
        let base = format!("layers.{index}");
        let attention = format!("{base}.self_attn");
        expected.extend([
            (
                format!("{attention}.q_proj.weight"),
                vec![
                    (config.num_attention_heads * config.head_dim) as i64,
                    hidden,
                ],
            ),
            (
                format!("{attention}.k_proj.weight"),
                vec![
                    (config.num_key_value_heads * config.head_dim) as i64,
                    hidden,
                ],
            ),
            (
                format!("{attention}.v_proj.weight"),
                vec![
                    (config.num_key_value_heads * config.head_dim) as i64,
                    hidden,
                ],
            ),
            (
                format!("{attention}.o_proj.weight"),
                vec![
                    hidden,
                    (config.num_attention_heads * config.head_dim) as i64,
                ],
            ),
            (
                format!("{attention}.q_norm.weight"),
                vec![config.head_dim as i64],
            ),
            (
                format!("{attention}.k_norm.weight"),
                vec![config.head_dim as i64],
            ),
            (
                format!("{base}.mlp.gate_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{base}.mlp.up_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{base}.mlp.down_proj.weight"),
                vec![hidden, intermediate],
            ),
            (format!("{base}.input_layernorm.weight"), vec![hidden]),
            (
                format!("{base}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
            (
                format!("{base}.attention_conv.base_kernel"),
                vec![2, config.conv_kernel_size as i64, hidden],
            ),
            (
                format!("{base}.attention_conv.kernel_projection.weight"),
                vec![(2 * config.conv_kernel_size * groups) as i64, hidden],
            ),
            (
                format!("{base}.mlp_conv.base_kernel"),
                vec![2, config.conv_kernel_size as i64, hidden],
            ),
            (
                format!("{base}.mlp_conv.kernel_projection.weight"),
                vec![(2 * config.conv_kernel_size * groups) as i64, hidden],
            ),
        ]);
    }
    expected.extend([
        (
            "candidate_selector.predecessor_codebook".to_string(),
            vec![config.vocab_size as i64, config.selector_rank as i64],
        ),
        (
            "candidate_selector.successor_codebook".to_string(),
            vec![config.vocab_size as i64, config.selector_rank as i64],
        ),
        (
            "candidate_selector.hidden_projection.weight".to_string(),
            vec![config.selector_rank as i64, hidden],
        ),
    ]);
    expected
}

fn validate_tensor_inventory(
    params: &HashMap<String, MxArray>,
    config: &DFlash2Config,
) -> Result<()> {
    let expected = expected_tensor_shapes(config);
    let expected_names = expected
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<std::collections::HashSet<_>>();
    let mut missing = Vec::new();
    for (name, shape) in &expected {
        let Some(array) = params.get(name) else {
            missing.push(name.clone());
            continue;
        };
        validate_tensor(array, name, shape)?;
    }
    let mut unexpected = params
        .keys()
        .filter(|name| !expected_names.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    unexpected.sort();
    if !missing.is_empty() || !unexpected.is_empty() {
        missing.sort();
        return Err(Error::from_reason(format!(
            "DFlash2 tensor inventory mismatch: missing={missing:?}, unexpected={unexpected:?}"
        )));
    }
    Ok(())
}

pub(crate) fn load_dflash2(path: &Path) -> Result<(DFlash2Model, u64)> {
    if !path.is_dir() {
        return Err(Error::from_reason(format!(
            "DFlash2 path is not a directory: {}",
            path.display()
        )));
    }
    let config_data = fs::read_to_string(path.join("config.json")).map_err(|error| {
        Error::from_reason(format!("Failed to read DFlash2 config.json: {error}"))
    })?;
    let raw: RawConfig = serde_json::from_str(&config_data).map_err(|error| {
        Error::from_reason(format!("Failed to parse DFlash2 config.json: {error}"))
    })?;
    let config = DFlash2Config::from_raw(raw)?;
    let tensor_path = resolve_safetensors(path)?;
    crate::engine::persistence::prewarm_checkpoint_pages(path);
    let mut params = load_safetensors_lazy(&tensor_path)?;
    // Structural preflight precedes the first GPU evaluation. A descriptor-
    // valid but incomplete companion must fail before any target or draft
    // state can be mutated.
    validate_tensor_inventory(&params, &config)?;
    let weight_bytes = params.values().fold(0u64, |total, array| {
        total.saturating_add(array.nbytes() as u64)
    });
    let arrays = params.values().collect::<Vec<_>>();
    let _resident = crate::array::memory::materialize_weights(&arrays)?;

    let hidden = config.hidden_size;
    let mut savings = 0u64;
    let fc = draft_linear(
        &mut params,
        "fc",
        hidden * config.target_layers.len(),
        hidden,
        &mut savings,
    )?;
    let hidden_norm = norm(
        &mut params,
        "hidden_norm.weight",
        hidden,
        config.rms_norm_eps,
    )?;
    let final_norm = norm(&mut params, "norm.weight", hidden, config.rms_norm_eps)?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for index in 0..config.num_hidden_layers {
        let base = format!("layers.{index}");
        let attention_base = format!("{base}.self_attn");
        let attention = DFlash2Attention {
            q_proj: draft_linear(
                &mut params,
                &format!("{attention_base}.q_proj"),
                hidden,
                config.num_attention_heads * config.head_dim,
                &mut savings,
            )?,
            k_proj: draft_linear(
                &mut params,
                &format!("{attention_base}.k_proj"),
                hidden,
                config.num_key_value_heads * config.head_dim,
                &mut savings,
            )?,
            v_proj: draft_linear(
                &mut params,
                &format!("{attention_base}.v_proj"),
                hidden,
                config.num_key_value_heads * config.head_dim,
                &mut savings,
            )?,
            o_proj: draft_linear(
                &mut params,
                &format!("{attention_base}.o_proj"),
                config.num_attention_heads * config.head_dim,
                hidden,
                &mut savings,
            )?,
            qkv_proj: None,
            q_norm: norm(
                &mut params,
                &format!("{attention_base}.q_norm.weight"),
                config.head_dim,
                config.rms_norm_eps,
            )?,
            k_norm: norm(
                &mut params,
                &format!("{attention_base}.k_norm.weight"),
                config.head_dim,
                config.rms_norm_eps,
            )?,
            rope: RoPE::new(
                config.head_dim as i32,
                Some(false),
                Some(config.rope_theta),
                None,
            ),
            num_heads: config.num_attention_heads as i64,
            num_kv_heads: config.num_key_value_heads as i64,
            head_dim: config.head_dim as i64,
            sliding_window: config.sliding_window as i64,
        };
        let mlp_base = format!("{base}.mlp");
        let mlp = DFlash2Mlp {
            gate_proj: draft_linear(
                &mut params,
                &format!("{mlp_base}.gate_proj"),
                hidden,
                config.intermediate_size,
                &mut savings,
            )?,
            up_proj: draft_linear(
                &mut params,
                &format!("{mlp_base}.up_proj"),
                hidden,
                config.intermediate_size,
                &mut savings,
            )?,
            down_proj: draft_linear(
                &mut params,
                &format!("{mlp_base}.down_proj"),
                config.intermediate_size,
                hidden,
                &mut savings,
            )?,
            gate_up: None,
        };
        let input_norm = norm(
            &mut params,
            &format!("{base}.input_layernorm.weight"),
            hidden,
            config.rms_norm_eps,
        )?;
        let post_attention_norm = norm(
            &mut params,
            &format!("{base}.post_attention_layernorm.weight"),
            hidden,
            config.rms_norm_eps,
        )?;
        let attention_conv =
            grouped_conv(&mut params, &base, "attention_conv", &config, &mut savings)?;
        let mlp_conv = grouped_conv(&mut params, &base, "mlp_conv", &config, &mut savings)?;
        layers.push(DFlash2Layer {
            attention,
            mlp,
            input_norm,
            post_attention_norm,
            attention_conv,
            mlp_conv,
        });
    }
    let selector = CandidateSelector {
        predecessor_codebook: codebook(
            &mut params,
            "candidate_selector.predecessor_codebook",
            config.vocab_size,
            config.selector_rank,
        )?,
        successor_codebook: codebook(
            &mut params,
            "candidate_selector.successor_codebook",
            config.vocab_size,
            config.selector_rank,
        )?,
        // Keep the published selector projection at its original precision.
        hidden_projection: dense_linear(
            &mut params,
            "candidate_selector.hidden_projection",
            hidden,
            config.selector_rank,
        )?,
        top_k: config.selector_top_k,
        rank: config.selector_rank,
        vocab_size: config.vocab_size,
    };
    if !params.is_empty() {
        let mut unexpected = params.keys().cloned().collect::<Vec<_>>();
        unexpected.sort();
        return Err(Error::from_reason(format!(
            "DFlash2 checkpoint contains unexpected tensors: {unexpected:?}"
        )));
    }
    let weight_bytes = weight_bytes.saturating_sub(savings);
    let mut model = DFlash2Model {
        config,
        weight_bytes,
        fc,
        hidden_norm,
        layers,
        norm: final_norm,
        selector,
        context_kv: None,
    };
    model.merge_projections()?;
    let weight_bytes = model.weight_bytes;
    Ok((model, weight_bytes))
}

#[cfg(test)]
mod tests {
    use super::{
        CandidateSelector, DFlash2Config, DFlash2ContextCache, SelectorPath,
        normalized_selector_probs, parallel_query_ids,
    };
    use crate::array::{DType, MxArray};
    use crate::models::quantized_linear::LinearProj;
    use crate::nn::{Embedding, Linear};

    /// The smallest geometry whose every draft projection is tileable
    /// (`N % 64 == 0`, `K % 256 == 0`): o_proj's K is `heads * head_dim`.
    fn checkpoint_inventory_config() -> DFlash2Config {
        DFlash2Config {
            block_size: 7,
            mask_token_id: 0,
            target_layers: vec![0, 1],
            target_num_layers: 2,
            hidden_size: 256,
            intermediate_size: 256,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: 64,
            vocab_size: 32,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 128,
            sliding_window: 8,
            conv_group_size: 16,
            conv_kernel_size: 2,
            selector_rank: 8,
            selector_top_k: 4,
            rope_theta: 10_000.0,
        }
    }

    fn checkpoint_inventory() -> std::collections::HashMap<String, MxArray> {
        super::expected_tensor_shapes(&checkpoint_inventory_config())
            .into_iter()
            .map(|(name, shape)| {
                let array = MxArray::zeros(&shape, Some(DType::BFloat16)).unwrap();
                (name, array)
            })
            .collect()
    }

    #[test]
    fn dense_draft_inventory_rejects_malformed_tensors_before_evaluation() {
        let config = checkpoint_inventory_config();
        let mut params = checkpoint_inventory();
        super::validate_tensor_inventory(&params, &config).unwrap();
        for (name, shape) in super::expected_tensor_shapes(&config) {
            let saved = params.remove(&name).unwrap();
            assert!(
                super::validate_tensor_inventory(&params, &config)
                    .unwrap_err()
                    .to_string()
                    .contains(&name)
            );
            params.insert(
                name.clone(),
                MxArray::zeros(&[1], Some(DType::BFloat16)).unwrap(),
            );
            assert!(
                super::validate_tensor_inventory(&params, &config)
                    .unwrap_err()
                    .to_string()
                    .contains("shape")
            );
            params.insert(
                name.clone(),
                MxArray::zeros(&shape, Some(DType::Uint32)).unwrap(),
            );
            assert!(
                super::validate_tensor_inventory(&params, &config)
                    .unwrap_err()
                    .to_string()
                    .contains("dtype")
            );
            params.insert(name, saved);
        }
        params.insert(
            "fc.scales".into(),
            MxArray::zeros(&[256, 2], Some(DType::BFloat16)).unwrap(),
        );
        assert!(
            super::validate_tensor_inventory(&params, &config)
                .unwrap_err()
                .to_string()
                .contains("fc.scales")
        );
    }

    #[test]
    fn selector_linear_keeps_checkpoint_precision() {
        let mut params = checkpoint_inventory();
        let projection = super::dense_linear(&mut params, "fc", 512, 256).unwrap();
        assert!(matches!(projection, LinearProj::Standard(_)));
        assert_eq!(projection.get_weight().dtype().unwrap(), DType::BFloat16);
    }

    /// The mode a draft projection carries on this host: the affine arrays
    /// tiled into the K-quant contract where the `_t64` kernels run, MLX's
    /// own affine route elsewhere.
    fn draft_mode() -> &'static str {
        if crate::models::quant_dispatch::kquant_tiled_enabled() {
            "a4g64@t64"
        } else {
            "affine"
        }
    }

    /// Asserts an affine Q4/group64 projection of a bf16 `[rows, cols]`
    /// weight: 8 codes per u32 and one bf16 scale/offset pair per 64 inputs
    /// (the Tiled64 permutation keeps the 2-D shapes).
    fn assert_q4_group64(projection: &LinearProj, rows: i64, cols: i64, name: &str) {
        let LinearProj::Quantized(packed) = projection else {
            panic!("{name}: draft projection must load quantized");
        };
        assert_eq!(packed.mode(), draft_mode(), "{name}");
        assert_eq!(packed.bits(), 4, "{name}");
        assert_eq!(
            packed.get_weight().dtype().unwrap(),
            DType::Uint32,
            "{name}"
        );
        assert_eq!(
            packed.get_weight().shape().unwrap().as_ref(),
            &[rows, cols / 8],
            "{name}: packed codes"
        );
        assert_eq!(
            packed.get_scales().dtype().unwrap(),
            DType::BFloat16,
            "{name}"
        );
        assert_eq!(
            packed.get_scales().shape().unwrap().as_ref(),
            &[rows, cols / 64],
            "{name}: scales"
        );
        let biases = packed.get_biases().expect("affine Q4 has offsets");
        assert_eq!(biases.dtype().unwrap(), DType::BFloat16, "{name}");
        assert_eq!(
            biases.shape().unwrap().as_ref(),
            &[rows, cols / 64],
            "{name}: offsets"
        );
    }

    /// `draft_linear` keeps the affine Q4/group64 numerics through the
    /// tiled contract: integer endpoints spanning exactly 15 steps decode
    /// exactly, in rows of either sign (stored offsets), on every route a
    /// decode block reaches (M = 1, 8 and a prefill height).
    #[test]
    fn draft_linear_is_q4_group64_with_exact_endpoints_and_residency() {
        let (rows, cols) = (64i64, 256i64);
        let values = (0..rows * cols)
            .map(|i| {
                let (row, col) = (i / cols, i % cols);
                if col % 2 == 0 {
                    0.0
                } else if row % 2 == 0 {
                    15.0
                } else {
                    -15.0
                }
            })
            .collect::<Vec<_>>();
        let weight = MxArray::from_float32(&values, &[rows, cols])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let mut params = std::collections::HashMap::from([("fc.weight".into(), weight.clone())]);
        let mut savings = 0;
        let projection = super::draft_linear(
            &mut params,
            "fc",
            cols as usize,
            rows as usize,
            &mut savings,
        )
        .unwrap();
        assert_q4_group64(&projection, rows, cols, "fc");
        let LinearProj::Quantized(packed) = &projection else {
            unreachable!();
        };
        let resident = packed.get_weight().nbytes()
            + packed.get_scales().nbytes()
            + packed.get_biases().map_or(0, |biases| biases.nbytes());
        // Each row owns 128 packed bytes and four bf16 scale/offset pairs.
        assert_eq!(resident as i64, rows * (cols / 2 + cols / 64 * 4));
        assert_eq!(savings, weight.nbytes() as u64 - resident as u64);
        assert!(params.is_empty());

        // Every row sums its 128 non-zero entries: +-1920, exact in bf16.
        let want = (0..rows)
            .map(|row| if row % 2 == 0 { 1920.0 } else { -1920.0 })
            .collect::<Vec<f32>>();
        for m in [1i64, 8, 87] {
            let input = MxArray::ones(&[1, m, cols], Some(DType::BFloat16)).unwrap();
            let output = projection.forward(&input).unwrap().to_float32().unwrap();
            assert_eq!(output.len() as i64, m * rows, "M={m}");
            for (row, got) in output.chunks_exact(rows as usize).enumerate() {
                assert_eq!(got, want.as_slice(), "M={m} input row {row}");
            }
        }

        // An untileable geometry fails loud rather than loading slower.
        let odd = MxArray::zeros(&[2, 128], Some(DType::BFloat16)).unwrap();
        let Err(err) = super::quantize_draft_weight(&odd, "odd") else {
            panic!("an untileable draft weight must be refused");
        };
        assert!(err.to_string().contains("N % 64 == 0"), "{err}");
    }

    #[test]
    fn load_quantizes_every_draft_projection_and_keeps_the_selector_dense() {
        let config = checkpoint_inventory_config();
        let dir = std::env::temp_dir().join(format!(
            "mlx-dflash2-load-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({
                "architectures": ["DFlash2DraftModel"],
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_target_layers": config.target_num_layers,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "head_dim": config.head_dim,
                "vocab_size": config.vocab_size,
                "rms_norm_eps": config.rms_norm_eps,
                "max_position_embeddings": config.max_position_embeddings,
                "sliding_window": config.sliding_window,
                "layer_types": ["sliding_attention"],
                "rope_theta": config.rope_theta,
                "dflash_config": {
                    "block_size": config.block_size + 1,
                    "conv_group_size": config.conv_group_size,
                    "conv_kernel_size": config.conv_kernel_size,
                    "mask_token_id": config.mask_token_id,
                    "selector_rank": config.selector_rank,
                    "selector_top_k": config.selector_top_k,
                    "target_layer_ids": config.target_layers,
                },
            })
            .to_string(),
        )
        .unwrap();
        let shapes = super::expected_tensor_shapes(&config);
        let mut tensors = shapes
            .iter()
            .map(|(name, shape)| {
                let array =
                    MxArray::random_normal(shape, 0.0, 0.02, Some(DType::BFloat16)).unwrap();
                (name.clone(), array)
            })
            .collect::<std::collections::HashMap<_, _>>();
        crate::utils::safetensors::save_safetensors(
            dir.join("model.safetensors"),
            &mut tensors,
            None,
        )
        .unwrap();
        let loaded = super::load_dflash2(&dir);
        std::fs::remove_dir_all(&dir).unwrap();
        let (model, bytes) = loaded.unwrap();

        let projection = |name: &str| -> bool {
            name == "fc.weight"
                || name.ends_with("_proj.weight")
                || name.ends_with(".kernel_projection.weight")
        };
        let packed_bytes = |elements: u64| elements / 2 + elements / 64 * 4;
        let mut expected_bytes = shapes
            .iter()
            .map(|(name, shape)| {
                let elements = shape.iter().product::<i64>() as u64;
                if projection(name) {
                    // u32 codes at 4 bits plus bf16 scale and offset per 64.
                    packed_bytes(elements)
                } else {
                    elements * 2
                }
            })
            .sum::<u64>();
        // The merges follow the tiled contract: on a Metal host every
        // projection tiles, so q|k|v, gate|up and the cross-layer k|v (a
        // second resident copy of k/v) all exist.
        let tiled = crate::models::quant_dispatch::kquant_tiled_enabled();
        assert_eq!(model.context_kv.is_some(), tiled);
        assert_eq!(model.layers[0].attention.qkv_proj.is_some(), tiled);
        assert_eq!(model.layers[0].mlp.gate_up.is_some(), tiled);
        if model.context_kv.is_some() {
            let kv = (config.num_key_value_heads * config.head_dim * config.hidden_size) as u64;
            expected_bytes += 2 * packed_bytes(kv) * config.num_hidden_layers as u64;
        }
        assert_eq!(bytes, expected_bytes);
        assert_eq!(model.weight_bytes, expected_bytes);

        let hidden = config.hidden_size as i64;
        let intermediate = config.intermediate_size as i64;
        let attention = (config.num_attention_heads * config.head_dim) as i64;
        let kv = (config.num_key_value_heads * config.head_dim) as i64;
        let conv =
            (2 * config.conv_kernel_size * (config.hidden_size / config.conv_group_size)) as i64;
        let taps = config.target_layers.len() as i64;
        assert_q4_group64(&model.fc, hidden, hidden * taps, "fc");
        assert_eq!(model.layers.len(), 1);
        let layer = &model.layers[0];
        for (name, projection, rows, cols) in [
            ("q_proj", &layer.attention.q_proj, attention, hidden),
            ("k_proj", &layer.attention.k_proj, kv, hidden),
            ("v_proj", &layer.attention.v_proj, kv, hidden),
            ("o_proj", &layer.attention.o_proj, hidden, attention),
            ("gate_proj", &layer.mlp.gate_proj, intermediate, hidden),
            ("up_proj", &layer.mlp.up_proj, intermediate, hidden),
            ("down_proj", &layer.mlp.down_proj, hidden, intermediate),
            (
                "attention_conv",
                &layer.attention_conv.kernel_projection,
                conv,
                hidden,
            ),
            ("mlp_conv", &layer.mlp_conv.kernel_projection, conv, hidden),
        ] {
            assert_q4_group64(projection, rows, cols, name);
        }
        let selector = &model.selector.hidden_projection;
        assert!(matches!(selector, LinearProj::Standard(_)));
        assert_eq!(selector.get_weight().dtype().unwrap(), DType::BFloat16);
    }

    fn test_selector(vocab: usize, top_k: usize, rank: usize, hidden: usize) -> CandidateSelector {
        let normal = |rows: usize, cols: usize| {
            MxArray::random_normal(&[rows as i64, cols as i64], 0.0, 0.5, Some(DType::Float32))
                .unwrap()
        };
        CandidateSelector {
            predecessor_codebook: Embedding::from_weight(&normal(vocab, rank)).unwrap(),
            successor_codebook: Embedding::from_weight(&normal(vocab, rank)).unwrap(),
            hidden_projection: LinearProj::Standard(
                Linear::from_weights(&normal(rank, hidden), None).unwrap(),
            ),
            top_k,
            rank,
            vocab_size: vocab,
        }
    }

    fn direct_fused_greedy(candidates: &MxArray, scores: &MxArray) -> Option<Vec<i32>> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return None;
        }
        let mut path = std::ptr::null_mut();
        assert!(unsafe {
            mlx_sys::mlx_dflash2_greedy_path(
                candidates.as_raw_ptr(),
                scores.as_raw_ptr(),
                &mut path,
            )
        });
        assert!(!path.is_null());
        let path = MxArray::from_handle(path, "test dflash2 greedy path").unwrap();
        Some(path.to_int32().unwrap().as_ref().to_vec())
    }

    fn lazy_greedy(
        selector: &CandidateSelector,
        candidates: &MxArray,
        scores: &MxArray,
        length: usize,
    ) -> Vec<i32> {
        selector
            .lazy_greedy_path(candidates, scores, length)
            .unwrap()
            .to_int32()
            .unwrap()
            .as_ref()
            .to_vec()
    }

    #[test]
    fn fused_greedy_path_matches_lazy_random_and_branching_walks() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            eprintln!("skipping: fused DFlash2 greedy path requires Metal");
            return;
        }
        for top_k in [4usize, 16] {
            let selector = test_selector(32, top_k, 4, 4);
            for length in 1usize..=8 {
                let candidate_values = (0..length * top_k)
                    .map(|i| 10_000 + i as i32)
                    .collect::<Vec<_>>();
                let candidates =
                    MxArray::from_int32(&candidate_values, &[1, length as i64, top_k as i64])
                        .unwrap();

                // Finite random scores cover arbitrary predecessor-dependent
                // routes. Exact output indices, not score tolerances, are the
                // contract under test.
                let random_scores = MxArray::random_normal(
                    &[length as i64, top_k as i64, top_k as i64],
                    0.0,
                    1.0,
                    Some(DType::Float32),
                )
                .unwrap();
                assert_eq!(
                    direct_fused_greedy(&candidates, &random_scores).unwrap(),
                    lazy_greedy(&selector, &candidates, &random_scores, length),
                    "random L={length} K={top_k}",
                );

                // Each predecessor row deliberately has a different winner;
                // a kernel that forgets to feed the prior selection into the
                // next position fails this expected path.
                let mut branching = vec![-100.0f32; length * top_k * top_k];
                for position in 0..length {
                    for predecessor in 0..top_k {
                        let winner = (predecessor * 3 + position * 5 + 1) % top_k;
                        let row = (position * top_k + predecessor) * top_k;
                        branching[row + winner] = 100.0 + position as f32;
                    }
                }
                let scores =
                    MxArray::from_float32(&branching, &[length as i64, top_k as i64, top_k as i64])
                        .unwrap();
                let mut predecessor = 0usize;
                let mut expected = Vec::with_capacity(length);
                for position in 0..length {
                    predecessor = (predecessor * 3 + position * 5 + 1) % top_k;
                    expected.push(candidate_values[position * top_k + predecessor]);
                }
                assert_eq!(
                    direct_fused_greedy(&candidates, &scores).unwrap(),
                    expected,
                    "branching L={length} K={top_k}",
                );
            }
        }
    }

    #[test]
    fn fused_greedy_path_matches_lazy_ties_infinities_signed_zero_and_nan() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            eprintln!("skipping: fused DFlash2 greedy path requires Metal");
            return;
        }
        for top_k in [4usize, 16] {
            let length = 12usize;
            let selector = test_selector(32, top_k, 4, 4);
            let candidate_values = (0..length * top_k)
                .map(|i| 20_000 + i as i32)
                .collect::<Vec<_>>();
            let candidates =
                MxArray::from_int32(&candidate_values, &[1, length as i64, top_k as i64]).unwrap();
            let mut values = vec![f32::NEG_INFINITY; length * top_k * top_k];
            for position in 0..length {
                for predecessor in 0..top_k {
                    let row = (position * top_k + predecessor) * top_k;
                    match position % 7 {
                        0 => {
                            values[row + 1] = 7.0;
                            values[row + top_k - 2] = 7.0; // last equal max wins
                        }
                        1 => {
                            // IEEE `>` treats both zero signs as equal, so a
                            // trailing -0.0 must beat an earlier +0.0 under
                            // the reversed last-maximum walk. `total_cmp`
                            // would choose the +0.0 and is deliberately not
                            // the device argmax contract exercised here.
                            values[row..row + top_k].fill(-0.0);
                            values[row + 1] = 0.0;
                        }
                        2 => {
                            // A NaN beats every number, and the reverse walk
                            // meets the last NaN slot first.
                            values[row..row + top_k].fill(f32::NAN);
                            values[row] = 3.0;
                            values[row + top_k / 2] = 3.0;
                        }
                        3 => values[row..row + top_k].fill(f32::NAN),
                        4 => {
                            values[row + 1] = f32::INFINITY;
                            values[row + top_k - 2] = f32::INFINITY;
                        }
                        5 => {
                            // The walk meets 9.0 before the one NaN; the NaN
                            // still wins.
                            values[row + 1] = f32::NAN;
                            values[row + top_k - 1] = 9.0;
                        }
                        _ => {
                            // Leave the row entirely -inf: no value exceeds
                            // the initial -inf, so the reverse walk retains
                            // the final slot.
                        }
                    }
                }
            }
            let scores =
                MxArray::from_float32(&values, &[length as i64, top_k as i64, top_k as i64])
                    .unwrap();
            let expected_slots = [
                top_k - 2,
                top_k - 1,
                top_k - 1,
                top_k - 1,
                top_k - 2,
                1,
                top_k - 1,
            ]
            .into_iter()
            .cycle()
            .take(length)
            .collect::<Vec<_>>();
            let expected = expected_slots
                .iter()
                .enumerate()
                .map(|(position, &slot)| candidate_values[position * top_k + slot])
                .collect::<Vec<_>>();
            let fused = direct_fused_greedy(&candidates, &scores).unwrap();
            assert_eq!(fused, expected, "special values K={top_k}");
            assert_eq!(
                fused,
                lazy_greedy(&selector, &candidates, &scores, length),
                "device argmax semantics K={top_k}",
            );
        }
    }

    #[test]
    fn fused_greedy_path_accepts_noncontiguous_views() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            eprintln!("skipping: fused DFlash2 greedy path requires Metal");
            return;
        }
        let (length, top_k) = (7usize, 16usize);
        let selector = test_selector(32, top_k, 4, 4);
        let candidates = MxArray::from_int32(
            &(0..length * top_k).map(|i| i as i32).collect::<Vec<_>>(),
            &[1, top_k as i64, length as i64],
        )
        .unwrap()
        .transpose(Some(&[0, 2, 1]))
        .unwrap();
        let scores = MxArray::random_normal(
            &[top_k as i64, length as i64, top_k as i64],
            0.0,
            1.0,
            Some(DType::Float32),
        )
        .unwrap()
        .transpose(Some(&[1, 0, 2]))
        .unwrap();
        assert_eq!(
            direct_fused_greedy(&candidates, &scores).unwrap(),
            lazy_greedy(&selector, &candidates, &scores, length),
        );
    }

    #[test]
    fn fused_greedy_path_ffi_rejects_malformed_inputs() {
        let good_candidates = MxArray::from_int32(&[0; 8], &[1, 2, 4]).unwrap();
        let good_scores = MxArray::from_float32(&[0.0; 32], &[2, 4, 4]).unwrap();
        let bad_candidate_dtype = MxArray::from_float32(&[0.0; 8], &[1, 2, 4]).unwrap();
        let bad_score_dtype = good_scores.astype(DType::BFloat16).unwrap();
        let wrong_length = MxArray::from_float32(&[0.0; 16], &[1, 4, 4]).unwrap();
        let nonsquare = MxArray::from_float32(&[0.0; 40], &[2, 4, 5]).unwrap();
        for (candidates, scores) in [
            (&bad_candidate_dtype, &good_scores),
            (&good_candidates, &bad_score_dtype),
            (&good_candidates, &wrong_length),
            (&good_candidates, &nonsquare),
        ] {
            let mut path = std::ptr::null_mut();
            assert!(!unsafe {
                mlx_sys::mlx_dflash2_greedy_path(
                    candidates.as_raw_ptr(),
                    scores.as_raw_ptr(),
                    &mut path,
                )
            });
            assert!(path.is_null());
        }
        assert!(!unsafe {
            mlx_sys::mlx_dflash2_greedy_path(
                good_candidates.as_raw_ptr(),
                good_scores.as_raw_ptr(),
                std::ptr::null_mut(),
            )
        });
    }

    /// Diagnostic only: isolates the terminal device-resident predecessor walk
    /// at the production DFlash2 shape. Each timed iteration constructs a fresh
    /// graph, evaluates it, and reads the path; shared inputs are materialized
    /// before warmup. Alternating batches limit order/thermal bias. This prints
    /// observations and deliberately asserts no timing threshold.
    #[test]
    #[ignore = "diagnostic Metal timing; run manually on an idle machine"]
    fn benchmark_fused_greedy_path_vs_lazy_graph() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            eprintln!("skipping: DFlash2 greedy-path benchmark requires Metal");
            return;
        }
        const LENGTH: usize = 7;
        const TOP_K: usize = 16;
        const BATCH_ITERS: usize = 50;
        const SAMPLES: usize = 12;
        let selector = test_selector(32, TOP_K, 4, 4);
        let candidates = MxArray::from_int32(
            &(0..LENGTH * TOP_K).map(|i| i as i32).collect::<Vec<_>>(),
            &[1, LENGTH as i64, TOP_K as i64],
        )
        .unwrap();
        let scores = MxArray::random_normal(
            &[LENGTH as i64, TOP_K as i64, TOP_K as i64],
            0.0,
            1.0,
            Some(DType::Float32),
        )
        .unwrap();
        MxArray::eval_arrays(&[&candidates, &scores]).unwrap();

        let expected = lazy_greedy(&selector, &candidates, &scores, LENGTH);
        assert_eq!(direct_fused_greedy(&candidates, &scores).unwrap(), expected);
        for _ in 0..3 {
            assert_eq!(
                lazy_greedy(&selector, &candidates, &scores, LENGTH),
                expected
            );
            assert_eq!(direct_fused_greedy(&candidates, &scores).unwrap(), expected);
        }

        let measure = |fused: bool| {
            let start = std::time::Instant::now();
            for _ in 0..BATCH_ITERS {
                let path = if fused {
                    direct_fused_greedy(&candidates, &scores).unwrap()
                } else {
                    lazy_greedy(&selector, &candidates, &scores, LENGTH)
                };
                std::hint::black_box(path);
            }
            start.elapsed().as_secs_f64() * 1e6 / BATCH_ITERS as f64
        };
        let (mut lazy_us, mut fused_us) =
            (Vec::with_capacity(SAMPLES), Vec::with_capacity(SAMPLES));
        for sample in 0..SAMPLES {
            let (lazy, fused) = if sample % 2 == 0 {
                (measure(false), measure(true))
            } else {
                let fused = measure(true);
                (measure(false), fused)
            };
            lazy_us.push(lazy);
            fused_us.push(fused);
            eprintln!("[greedy-path] sample={sample} lazy={lazy:.2}us fused={fused:.2}us");
        }
        eprintln!("[greedy-path] lazy_us={lazy_us:?} fused_us={fused_us:?}");
    }

    #[test]
    fn selector_device_walk_matches_host_greedy_walk() {
        let (vocab, top_k, rank, hidden, len) = (32usize, 4usize, 8usize, 8usize, 6i64);
        let selector = test_selector(vocab, top_k, rank, hidden);
        let hidden =
            MxArray::random_normal(&[1, len, hidden as i64], 0.0, 1.0, Some(DType::Float32))
                .unwrap();
        let logits =
            MxArray::random_normal(&[1, len, vocab as i64], 0.0, 1.0, Some(DType::Float32))
                .unwrap();

        let (host_path, _) = selector
            .select(&hidden, &logits, 3, 0.0, false, &mut rand::rng())
            .expect("host select");
        let SelectorPath::Host(host_ids) = host_path else {
            panic!("host walk must return host ids");
        };
        let (device_path, sparse) = selector
            .select(&hidden, &logits, 3, 0.0, true, &mut rand::rng())
            .expect("device select");
        assert!(sparse.is_empty(), "greedy device walk emits no sparse rows");
        let SelectorPath::Device(ids) = device_path else {
            panic!("greedy device walk must return device ids");
        };
        let device_ids: Vec<i32> = ids.to_int32().unwrap().as_ref().to_vec();
        assert_eq!(device_ids, host_ids);
    }

    /// Zeroing the hidden projection zeroes the edge term, so every top-k
    /// score row degenerates to the unary logits and duplicate maxima become
    /// EXACT f32 ties. The host walk's `max_by` keeps the last maximum; the
    /// device walk must agree (reversed argmax), or the greedy proposal path
    /// would silently diverge on ties.
    #[test]
    fn selector_device_walk_matches_host_last_max_on_ties() {
        let (vocab, top_k, rank, hidden, len) = (16usize, 4usize, 4usize, 4usize, 4i64);
        let mut selector = test_selector(vocab, top_k, rank, hidden);
        selector.hidden_projection = LinearProj::Standard(
            Linear::from_weights(
                &MxArray::zeros(&[rank as i64, hidden as i64], Some(DType::Float32)).unwrap(),
                None,
            )
            .unwrap(),
        );
        let hidden = MxArray::zeros(&[1, len, hidden as i64], Some(DType::Float32)).unwrap();
        // Two equal maxima inside the top-4 of every row: ties at different
        // candidate slots always discriminate first- vs last-max policies.
        let row: Vec<f32> = (0..vocab)
            .map(|i| {
                if i == 5 || i == 9 {
                    2.0
                } else {
                    i as f32 * 0.1
                }
            })
            .collect();
        let flat = row.repeat(len as usize);
        let logits = MxArray::from_float32(&flat, &[1, len, vocab as i64]).unwrap();

        let (host_path, _) = selector
            .select(&hidden, &logits, 0, 0.0, false, &mut rand::rng())
            .unwrap();
        let SelectorPath::Host(host_ids) = host_path else {
            panic!("host walk must return host ids");
        };
        let (device_path, _) = selector
            .select(&hidden, &logits, 0, 0.0, true, &mut rand::rng())
            .unwrap();
        let SelectorPath::Device(ids) = device_path else {
            panic!("greedy device walk must return device ids");
        };
        let device_ids: Vec<i32> = ids.to_int32().unwrap().as_ref().to_vec();
        assert_eq!(device_ids, host_ids);
        assert!(
            device_ids.iter().all(|&id| id == 5 || id == 9),
            "tie rows must select one of the tied maxima, got {device_ids:?}"
        );
    }

    #[test]
    fn fused_topk16_rejects_empty_and_undersized_shapes() -> napi::Result<()> {
        // Rejected before shader construction or evaluation, including zero
        // vocabulary (which must not reach the row-count division).
        for shape in [[1, 2, 0], [1, 0, 16], [1, 2, 15]] {
            let logits = MxArray::zeros(&shape, Some(DType::Float32))?;
            let mut ids = logits.as_raw_ptr();
            let mut values = logits.as_raw_ptr();
            assert!(!unsafe {
                mlx_sys::mlx_dflash2_topk16(logits.as_raw_ptr(), &mut ids, &mut values)
            });
            assert!(ids.is_null() && values.is_null());
        }
        Ok(())
    }

    /// The sharded top-16 kernel must return exactly the argpartition
    /// candidate SET per row — same ids, matching logits values, ascending
    /// value order. Called through the FFI directly so the test can never
    /// pass vacuously on the fallback path.
    #[test]
    fn fused_topk16_matches_argpartition_candidate_set() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            eprintln!("SKIP fused topk16 parity: Metal backend unavailable");
            return;
        }
        // 248320 is the vocabulary in both the target GGUF text config and
        // its DFlash2 companion. Two rows bound each fixture below 2 MiB of
        // host logits while covering the production BF16 specialization.
        const ROWS: usize = 2;
        for (dtype, vocab) in [
            (DType::Float32, 8192_i64),
            (DType::BFloat16, 248320),
            (DType::Float16, 248320),
        ] {
            let logits = if dtype == DType::Float32 {
                MxArray::random_normal(&[1, ROWS as i64, vocab], 0.0, 1.0, Some(dtype)).unwrap()
            } else {
                let mut data = (0..ROWS * vocab as usize)
                    .map(|i| -1.0 - (i % 97) as f32)
                    .collect::<Vec<_>>();
                // Distinct exactly representable winners avoid BF16/F16
                // cutoff ties and exercise every scan shard.
                for row in 0..ROWS {
                    for rank in 0..16usize {
                        let token = rank * (vocab as usize / 16) + row * 13 + 5;
                        data[row * vocab as usize + token] = (rank + 1) as f32;
                    }
                }
                MxArray::from_float32(&data, &[1, ROWS as i64, vocab])
                    .unwrap()
                    .astype(dtype)
                    .unwrap()
            };
            let mut ids = std::ptr::null_mut();
            let mut values = std::ptr::null_mut();
            let status = unsafe {
                mlx_sys::mlx_dflash2_topk16_test(logits.as_raw_ptr(), &mut ids, &mut values)
            };
            if status == 0 {
                assert!(ids.is_null() && values.is_null());
                eprintln!(
                    "SKIP fused topk16 parity dtype={dtype:?} vocab={vocab}: valid fixture \
                     is unsupported by pipeline SIMD/threadgroup/shared-memory capabilities"
                );
                continue;
            }
            assert_eq!(
                status, 1,
                "fused topk16 construction/preflight failed for dtype={dtype:?} vocab={vocab}; \
                 see native error"
            );
            let ids = MxArray::from_handle(ids, "topk16 ids").unwrap();
            let values = MxArray::from_handle(values, "topk16 values").unwrap();
            let ids: Vec<i32> = ids.to_int32().unwrap().as_ref().to_vec();
            let values: Vec<f32> = values.to_float32().unwrap().as_ref().to_vec();
            let logits_f: Vec<f32> = logits
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .as_ref()
                .to_vec();

            let reference = logits
                .argpartition(-16, Some(-1))
                .unwrap()
                .slice_axis(2, vocab - 16, vocab)
                .unwrap();
            let ref_ids: Vec<i32> = reference.to_int32().unwrap().as_ref().to_vec();

            for row in 0..ROWS {
                let mine = &ids[row * 16..(row + 1) * 16];
                let mine_v = &values[row * 16..(row + 1) * 16];
                let theirs = &ref_ids[row * 16..(row + 1) * 16];
                let mut a = mine.to_vec();
                let mut b = theirs.to_vec();
                a.sort_unstable();
                b.sort_unstable();
                assert_eq!(
                    a, b,
                    "dtype={dtype:?} vocab={vocab} row={row}: candidate sets differ"
                );
                assert!(
                    mine_v.windows(2).all(|w| w[0] <= w[1]),
                    "dtype={dtype:?} vocab={vocab} row={row}: values not ascending: {mine_v:?}"
                );
                for (i, (&id, &v)) in mine.iter().zip(mine_v.iter()).enumerate() {
                    assert!(
                        id >= 0 && (id as i64) < vocab,
                        "dtype={dtype:?} vocab={vocab} row={row} slot={i}: bad id {id}"
                    );
                    let expected = logits_f[row * vocab as usize + id as usize];
                    assert_eq!(v, expected, "row {row} slot {i}: value mismatch");
                }
            }
            eprintln!(
                "PASS fused topk16 parity: native scan+merge ran for {ROWS} rows, \
                 dtype={dtype:?}, vocab={vocab}"
            );
        }
    }

    /// `record_only` advances logical length and provenance identically to
    /// `append` — just without the (dead) K/V write. The `logical_len ==
    /// token_history.len()` invariant is what `append` validates on the next
    /// retained segment.
    #[test]
    fn record_only_advances_context_without_kv() {
        let config = DFlash2Config {
            block_size: 7,
            mask_token_id: 0,
            target_layers: vec![0],
            target_num_layers: 1,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            vocab_size: 32,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 128,
            sliding_window: 8,
            conv_group_size: 1,
            conv_kernel_size: 2,
            selector_rank: 4,
            selector_top_k: 4,
            rope_theta: 10_000.0,
        };
        let mut context = DFlash2ContextCache::new(&config);
        context.record_only(&[10, 11, 12]).unwrap();
        context.record_only(&[13, 14]).unwrap();
        assert_eq!(context.logical_len(), 5);
        assert_eq!(context.token_history(), &[10, 11, 12, 13, 14]);
    }

    #[test]
    fn block_has_one_anchor_and_mask_rows() {
        assert_eq!(
            parallel_query_ids(7, 248_070, 3),
            vec![7, 248_070, 248_070, 248_070]
        );
    }

    #[test]
    fn selector_softmax_is_normalized_and_temperature_scaled() {
        let probs = normalized_selector_probs(&[0.0, 1.0, 2.0], 0.5).expect("selector probs");
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn fused_convolve_matches_elementwise_chain() {
        use super::GroupedDynamicCausalConv;
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            eprintln!("skipping: fused conv requires a Metal back-end");
            return;
        }
        let (hidden_size, group_size, kernel_size) = (64usize, 16i64, 2usize);
        let groups = hidden_size / group_size as usize;
        let conv = GroupedDynamicCausalConv {
            base_kernel: MxArray::random_normal(
                &[2, kernel_size as i64, hidden_size as i64],
                0.0,
                0.5,
                Some(DType::BFloat16),
            )
            .expect("base"),
            kernel_projection: LinearProj::Standard(
                Linear::from_weights(
                    &MxArray::zeros(
                        &[2 * kernel_size as i64 * groups as i64, hidden_size as i64],
                        Some(DType::BFloat16),
                    )
                    .expect("proj weight"),
                    None,
                )
                .expect("proj"),
            ),
            kernel_size,
            group_size: group_size as usize,
        };
        for length in [1i64, 5, 33] {
            let hidden = MxArray::random_normal(
                &[1, length, hidden_size as i64],
                0.0,
                1.0,
                Some(DType::BFloat16),
            )
            .expect("hidden");
            let dynamic = MxArray::random_normal(
                &[1, length, 2, kernel_size as i64, groups as i64],
                0.0,
                0.5,
                Some(DType::BFloat16),
            )
            .expect("dynamic");
            for side in [0usize, 1] {
                let fused = conv
                    .fused_convolve(&hidden, &dynamic, side)
                    .expect("fused conv must accept well-formed bf16 inputs");
                let reference = conv
                    .convolve_elementwise(&hidden, &dynamic, side)
                    .expect("reference conv");
                let max_abs = fused
                    .sub(&reference)
                    .and_then(|d| d.abs())
                    .and_then(|d| d.astype(DType::Float32))
                    .and_then(|d| d.reshape(&[-1]))
                    .and_then(|d| d.max(Some(&[0]), Some(false)))
                    .expect("diff");
                max_abs.eval();
                assert_eq!(
                    max_abs.item_at_float32(0).expect("item"),
                    0.0,
                    "fused conv diverged at L={length} side={side}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires the external 3.6 GB DFlash2 checkpoint"]
    fn loads_real_qwen38_dflash2_checkpoint_strictly() {
        let path = std::env::var("MLX_TEST_QWEN38_DFLASH2_PATH")
            .expect("set MLX_TEST_QWEN38_DFLASH2_PATH");
        let (model, bytes) = super::load_dflash2(std::path::Path::new(&path))
            .expect("real DFlash2 checkpoint must load");
        assert_eq!(model.config.block_size, 7);
        assert_eq!(model.config.target_layers, vec![5, 19, 33, 47, 61]);
        // Q4/group64 projections: ~1.27 GB resident versus 3.85 GB bf16
        // and ~2.17 GB at Q8/group64, plus the cross-layer k|v copy.
        assert!(
            bytes > 1_000_000_000 && bytes < 1_500_000_000,
            "resident draft bytes {bytes}"
        );
        // Every projection of the published geometry tiles (hidden 5120,
        // 32/8 heads of 128, intermediate 17408, fc K = 25600, conv N =
        // 1280) and the merges follow.
        let tiled = crate::models::quant_dispatch::kquant_tiled_enabled();
        assert_q4_group64(&model.fc, 5120, 25600, "fc");
        for layer in &model.layers {
            assert_eq!(layer.attention.qkv_proj.is_some(), tiled, "q|k|v merge");
            assert_eq!(layer.mlp.gate_up.is_some(), tiled, "gate|up merge");
            assert_q4_group64(&layer.attention.o_proj, 5120, 4096, "o_proj");
            assert_q4_group64(&layer.mlp.down_proj, 5120, 17408, "down_proj");
            assert_q4_group64(
                &layer.attention_conv.kernel_projection,
                1280,
                5120,
                "attention_conv",
            );
        }
        assert_eq!(model.context_kv.is_some(), tiled, "cross-layer k|v merge");
    }

    fn three_layer_tiny_draft() -> super::DFlash2Model {
        let target = super::super::config::Qwen3_5Config {
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
            vocab_size: 32,
            hidden_size: 64,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 256,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
            linear_num_value_heads: 2,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            partial_rotary_factor: 0.25,
            rope_theta: 10_000.0,
            paged_cache_memory_mb: None,
            paged_cache_initial_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: Some(false),
            persist_paged_cache: None,
            n_mtp_layers: 0,
        };
        unsafe { mlx_sys::mlx_seed(0xDF1A_5A11) };
        let mut model = super::tiny_dflash2_model_for_stepper_test(&target).unwrap();
        for _ in 0..2 {
            let extra = super::tiny_dflash2_model_for_stepper_test(&target).unwrap();
            model.layers.extend(extra.layers);
        }
        model.config.num_hidden_layers = model.layers.len();
        // Non-unit norm weights so a skipped or misplaced norm changes bits.
        for layer in &mut model.layers {
            for norm in [&mut layer.input_norm, &mut layer.post_attention_norm] {
                norm.set_weight(
                    &MxArray::random_normal(&[64], 1.0, 0.2, Some(DType::BFloat16)).unwrap(),
                )
                .unwrap();
            }
        }
        model
            .norm
            .set_weight(&MxArray::random_normal(&[64], 1.0, 0.2, Some(DType::BFloat16)).unwrap())
            .unwrap();
        model
    }

    /// The draft forward as it was before residual sums moved into the next
    /// norm and before the sliding mask was shared: plain adds, separate
    /// norms and a mask built per layer.
    fn reference_forward_hidden(
        model: &super::DFlash2Model,
        embedding: &Embedding,
        block_ids: &MxArray,
        query_base: i32,
        context: &super::DFlash2ContextCache,
    ) -> MxArray {
        let mut hidden = embedding.forward(block_ids).unwrap();
        for (index, layer) in model.layers.iter().enumerate() {
            let cached = context.layers[index].view();
            let live_len = cached
                .as_ref()
                .map(|(keys, _)| keys.shape_at(2).unwrap())
                .unwrap_or(0) as i32;
            let context_base = context.logical_len.saturating_sub(live_len);
            let residual = hidden.clone();
            let (prepared, dynamic) = layer
                .attention_conv
                .prepare(&layer.input_norm.forward(&hidden).unwrap())
                .unwrap();
            let mut fresh = None;
            // The pre-window path: the block's K/V concatenated after a copy
            // of the context, the three projections and the four-op
            // norm/rope chain.
            let attention = &layer.attention;
            let (batch, seq) = (prepared.shape_at(0).unwrap(), prepared.shape_at(1).unwrap());
            let queries = attention
                .q_proj
                .forward(&prepared)
                .unwrap()
                .reshape(&[batch, seq, attention.num_heads, attention.head_dim])
                .unwrap();
            let queries = attention
                .rope
                .forward(
                    &attention
                        .q_norm
                        .forward(&queries)
                        .unwrap()
                        .transpose(Some(&[0, 2, 1, 3]))
                        .unwrap(),
                    Some(query_base),
                )
                .unwrap();
            let (block_keys, block_values) =
                attention.project_context(&prepared, query_base).unwrap();
            let (keys, values, key_base) = match &cached {
                Some((context_keys, context_values)) => (
                    MxArray::concatenate(context_keys, &block_keys, 2).unwrap(),
                    MxArray::concatenate(context_values, &block_values, 2).unwrap(),
                    context_base,
                ),
                None => (block_keys, block_values, query_base),
            };
            let attention = attention
                .attend(&queries, &keys, &values, key_base, query_base, &mut fresh)
                .unwrap();
            hidden = residual
                .add(&layer.attention_conv.finish(&attention, &dynamic).unwrap())
                .unwrap();
            let (prepared, dynamic) = layer
                .mlp_conv
                .prepare(&layer.post_attention_norm.forward(&hidden).unwrap())
                .unwrap();
            hidden = hidden
                .add(
                    &layer
                        .mlp_conv
                        .finish(&layer.mlp.forward(&prepared, 8).unwrap(), &dynamic)
                        .unwrap(),
                )
                .unwrap();
        }
        model.norm.forward(&hidden).unwrap()
    }

    /// A draft at the published Qwen3.8 attention geometry (hidden 5120,
    /// 32/8 heads of 128, affine Q4/g64 as `draft_linear` loads it) with
    /// random weights.
    fn quantized_draft_model(layers: usize, intermediate: usize) -> super::DFlash2Model {
        use super::{DFlash2Attention, DFlash2Layer, DFlash2Mlp, GroupedDynamicCausalConv};
        use crate::nn::{RMSNorm, RoPE};
        let (hidden, heads, kv_heads, head_dim) = (5120usize, 32i64, 8i64, 128i64);
        let eps = 1e-6;
        let quantized = |input: usize, output: usize| -> LinearProj {
            let weight = MxArray::random_normal(
                &[output as i64, input as i64],
                0.0,
                0.02,
                Some(DType::BFloat16),
            )
            .unwrap();
            LinearProj::Quantized(super::quantize_draft_weight(&weight, "test").unwrap())
        };
        let norm = |size: i64| {
            RMSNorm::from_weight(
                &MxArray::random_normal(&[size], 1.0, 0.2, Some(DType::BFloat16)).unwrap(),
                Some(eps),
            )
            .unwrap()
        };
        let conv = || GroupedDynamicCausalConv {
            base_kernel: MxArray::random_normal(
                &[2, 2, hidden as i64],
                0.0,
                0.02,
                Some(DType::BFloat16),
            )
            .unwrap(),
            kernel_projection: quantized(hidden, 4 * (hidden / 16)),
            kernel_size: 2,
            group_size: 16,
        };
        let config = super::DFlash2Config {
            block_size: 7,
            mask_token_id: 0,
            target_layers: vec![1, 3],
            target_num_layers: 4,
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_hidden_layers: layers,
            num_attention_heads: heads as usize,
            num_key_value_heads: kv_heads as usize,
            head_dim: head_dim as usize,
            vocab_size: 64,
            rms_norm_eps: eps,
            max_position_embeddings: 262_144,
            sliding_window: 2048,
            conv_group_size: 16,
            conv_kernel_size: 2,
            selector_rank: 8,
            selector_top_k: 4,
            rope_theta: 10_000_000.0,
        };
        super::DFlash2Model {
            fc: quantized(hidden * 2, hidden),
            hidden_norm: norm(hidden as i64),
            norm: norm(hidden as i64),
            selector: test_selector(64, 4, 8, hidden),
            layers: (0..layers)
                .map(|_| DFlash2Layer {
                    attention: DFlash2Attention {
                        q_proj: quantized(hidden, (heads * head_dim) as usize),
                        k_proj: quantized(hidden, (kv_heads * head_dim) as usize),
                        v_proj: quantized(hidden, (kv_heads * head_dim) as usize),
                        o_proj: quantized((heads * head_dim) as usize, hidden),
                        qkv_proj: None,
                        q_norm: norm(head_dim),
                        k_norm: norm(head_dim),
                        rope: RoPE::new(
                            head_dim as i32,
                            Some(false),
                            Some(config.rope_theta),
                            None,
                        ),
                        num_heads: heads,
                        num_kv_heads: kv_heads,
                        head_dim,
                        sliding_window: config.sliding_window as i64,
                    },
                    mlp: DFlash2Mlp {
                        gate_proj: quantized(hidden, intermediate),
                        up_proj: quantized(hidden, intermediate),
                        down_proj: quantized(intermediate, hidden),
                        gate_up: None,
                    },
                    input_norm: norm(hidden as i64),
                    post_attention_norm: norm(hidden as i64),
                    attention_conv: conv(),
                    mlp_conv: conv(),
                })
                .collect(),
            config,
            weight_bytes: 0,
            context_kv: None,
        }
    }

    fn bits(array: &MxArray) -> Vec<u16> {
        array.eval();
        array.to_uint16_native().unwrap()
    }

    /// The merge gate admits K-quant projections (a tiled one only at whole
    /// tile widths) and declines MLX's affine route and dense projections.
    #[test]
    fn merge_rows_gate_follows_the_packed_contract() {
        use super::kquant_rows_merge;
        use crate::models::quantized_linear::QuantizedLinear;
        let packed = |mode: &str| {
            QuantizedLinear::new(
                MxArray::zeros(&[64, 32], Some(DType::Uint32)).unwrap(),
                MxArray::zeros(&[64, 4], Some(DType::BFloat16)).unwrap(),
                Some(MxArray::zeros(&[64, 4], Some(DType::BFloat16)).unwrap()),
                None,
                64,
                4,
                mode.to_string(),
            )
        };
        assert!(kquant_rows_merge(&packed("a4g64@t64"), &[64, 1024]));
        assert!(!kquant_rows_merge(&packed("a4g64@t64"), &[64, 96]));
        assert!(kquant_rows_merge(&packed("q4k@t64"), &[128]));
        assert!(!kquant_rows_merge(&packed("q4k"), &[64]));
        assert!(!kquant_rows_merge(&packed("affine"), &[64]));
        assert!(!kquant_rows_merge(&packed("mxfp4"), &[64]));
    }

    /// The flat window holds exactly the newest `window` appended rows in
    /// order through growth, reallocation and over-window appends, and
    /// `with_block` presents the window followed by the block, bit for bit
    /// like a concatenation — while the block rows it parks past the window
    /// never leak into later views.
    #[test]
    fn draft_kv_window_matches_rolling_concatenation() {
        let (heads, dim, window) = (2i64, 16i64, 20i64);
        let rows = |t: i64| {
            (
                MxArray::random_normal(&[1, heads, t, dim], 0.0, 1.0, Some(DType::BFloat16))
                    .unwrap(),
                MxArray::random_normal(&[1, t, heads, dim], 0.0, 1.0, Some(DType::BFloat16))
                    .unwrap()
                    .transpose(Some(&[0, 2, 1, 3]))
                    .unwrap(),
            )
        };
        let mut live = super::DraftKvWindow::new(window, 3);
        let mut all_keys: Option<MxArray> = None;
        let mut all_values: Option<MxArray> = None;
        for (appends, t) in [
            1i64, 3, 8, 1, 7, 5, 2, 25, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        ]
        .into_iter()
        .enumerate()
        {
            let (block_keys, block_values) = rows(t);
            let (keys, values) = live.with_block(&block_keys, &block_values).unwrap();
            let (want_keys, want_values) = match (&all_keys, &all_values) {
                (Some(k), Some(v)) => {
                    let n = k.shape_at(2).unwrap();
                    let (k, v) = (
                        k.slice_axis(2, (n - window).max(0), n).unwrap(),
                        v.slice_axis(2, (n - window).max(0), n).unwrap(),
                    );
                    (
                        MxArray::concatenate(&k, &block_keys, 2).unwrap(),
                        MxArray::concatenate(&v, &block_values, 2).unwrap(),
                    )
                }
                _ => (block_keys.clone(), block_values.clone()),
            };
            assert_eq!(
                bits(&keys),
                bits(&want_keys),
                "append {appends}: block view keys"
            );
            assert_eq!(
                bits(&values),
                bits(&want_values),
                "append {appends}: block view values"
            );
            let (context_keys, context_values) = rows(t);
            live.append(&context_keys, &context_values).unwrap();
            all_keys = Some(match all_keys {
                Some(k) => MxArray::concatenate(&k, &context_keys, 2).unwrap(),
                None => context_keys,
            });
            all_values = Some(match all_values {
                Some(v) => MxArray::concatenate(&v, &context_values, 2).unwrap(),
                None => context_values,
            });
            let n = all_keys.as_ref().unwrap().shape_at(2).unwrap();
            let (keys, values) = live.view().unwrap();
            assert_eq!(live.len(), n.min(window));
            assert_eq!(
                bits(&keys),
                bits(
                    &all_keys
                        .as_ref()
                        .unwrap()
                        .slice_axis(2, (n - window).max(0), n)
                        .unwrap()
                ),
                "append {appends}: window keys"
            );
            assert_eq!(
                bits(&values),
                bits(
                    &all_values
                        .as_ref()
                        .unwrap()
                        .slice_axis(2, (n - window).max(0), n)
                        .unwrap()
                ),
                "append {appends}: window values"
            );
        }
    }

    /// bf16 bit patterns `got` against `want`: identical when `exact`, else
    /// (the M = 8 heights, where the merged width changes the tensor-op
    /// split-K and so the fp32 summation order) every element within 2^-6
    /// of the tensor's largest magnitude and at most 5% of them differing
    /// at all — last-bit flips, not a wrong row or slice.
    fn assert_bits_match(got: &[u16], want: &[u16], exact: bool, label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        if exact {
            assert!(got == want, "{label}: bits differ");
            return;
        }
        let f = |b: u16| half::bf16::from_bits(b).to_f32();
        let scale = want.iter().map(|&w| f(w).abs()).fold(0f32, f32::max);
        let worst = got
            .iter()
            .zip(want)
            .map(|(&g, &w)| (f(g) - f(w)).abs())
            .fold(0f32, f32::max);
        let differing = got.iter().zip(want).filter(|(g, w)| g != w).count();
        assert!(
            worst <= scale / 64.0 && differing * 20 <= got.len(),
            "{label}: max |diff| {worst} of scale {scale}, {differing}/{} elements differ",
            got.len()
        );
    }

    /// The merged `q|k|v`, `gate|up` and cross-layer `k|v` matmuls, the
    /// fused q/k norm + RoPE kernel and the one RoPE over all layers' keys
    /// must reproduce the separate projections and the four-op chain at the
    /// real head geometry. Bit for bit at 1..=7 rows (the tiled `qmv_t64` /
    /// `qmv_wide_t64` kernels reduce each row the same way at any width) and
    /// at a prefill-high 87 rows (the merge is bypassed: the GEMM split-K
    /// follows N). At 8 rows the `qmm_m8_nax_t64` split-K count follows N
    /// too (`qmm_m8_nax_splits`: k/v at N = 1024 split 8 ways, the merged
    /// q|k|v at 6144 4 ways), so the merged widths round the fp32 partial
    /// sums differently in a few last bits: `assert_bits_match` bounds it.
    #[test]
    fn merged_draft_projections_match_separate_bits() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            eprintln!("skipping: Metal only");
            return;
        }
        let mut model = quantized_draft_model(3, 4096);
        let heights = [1i64, 2, 3, 4, 5, 6, 7, 8, 87];
        let inputs = heights
            .iter()
            .map(|&t| {
                MxArray::random_normal(&[1, t, 5120], 0.0, 1.0, Some(DType::BFloat16)).unwrap()
            })
            .collect::<Vec<_>>();
        let base = 1234;
        // References from the unmerged model: the four-op norm/rope chain,
        // separate matmuls, per-layer context projections.
        let mut want = Vec::new();
        for x in &inputs {
            let seq = x.shape_at(1).unwrap();
            let mut per_layer = Vec::new();
            for layer in &model.layers {
                let attention = &layer.attention;
                let queries = attention
                    .q_proj
                    .forward(x)
                    .unwrap()
                    .reshape(&[1, seq, attention.num_heads, attention.head_dim])
                    .unwrap();
                let queries = attention
                    .rope
                    .forward(
                        &attention
                            .q_norm
                            .forward(&queries)
                            .unwrap()
                            .transpose(Some(&[0, 2, 1, 3]))
                            .unwrap(),
                        Some(base),
                    )
                    .unwrap();
                let (keys, values) = attention.project_context(x, base).unwrap();
                per_layer.push((
                    bits(&queries),
                    bits(&keys),
                    bits(&values),
                    bits(&layer.mlp.forward(x, 0).unwrap()),
                ));
            }
            let context = model
                .project_context(x, base)
                .unwrap()
                .iter()
                .map(|(k, v)| (bits(k), bits(v)))
                .collect::<Vec<_>>();
            want.push((per_layer, context));
        }
        model.merge_projections().unwrap();
        assert!(model.context_kv.is_some(), "cross-layer k|v merge");
        let probe = |heads: i64| {
            MxArray::random_normal(&[1, 8, heads, 128], 0.0, 1.0, Some(DType::BFloat16)).unwrap()
        };
        assert!(
            model.layers[0]
                .attention
                .fused_qk_norm_rope(&probe(32), &probe(8), base)
                .is_some(),
            "the fused norm/rope kernel must serve the draft geometry"
        );
        for (x, (per_layer, context)) in inputs.iter().zip(&want) {
            let seq = x.shape_at(1).unwrap();
            let exact = seq != model.merged_rows();
            for (index, layer) in model.layers.iter().enumerate() {
                assert!(layer.attention.qkv_proj.is_some(), "q|k|v merge");
                assert!(layer.mlp.gate_up.is_some(), "gate|up merge");
                let (queries, keys, values) = layer
                    .attention
                    .project_block(x, base, model.merged_rows())
                    .unwrap();
                let (want_q, want_k, want_v, want_mlp) = &per_layer[index];
                let label = |what: &str| format!("T={seq} layer {index}: {what}");
                assert_bits_match(&bits(&queries), want_q, exact, &label("queries"));
                assert_bits_match(&bits(&keys), want_k, exact, &label("block keys"));
                assert_bits_match(&bits(&values), want_v, exact, &label("block values"));
                assert_bits_match(
                    &bits(&layer.mlp.forward(x, model.merged_rows()).unwrap()),
                    want_mlp,
                    exact,
                    &label("mlp"),
                );
            }
            let got = model.project_context(x, base).unwrap();
            assert_eq!(got.len(), context.len());
            for (index, ((k, v), (want_k, want_v))) in got.iter().zip(context).enumerate() {
                let label = |what: &str| format!("T={seq} layer {index}: {what}");
                assert_bits_match(&bits(k), want_k, exact, &label("context keys"));
                assert_bits_match(&bits(v), want_v, exact, &label("context values"));
            }
        }
    }

    /// `forward_hidden` fuses each residual sum into the next norm and shares
    /// one sliding mask across layers; with and without a mask (context
    /// shorter and longer than the window) it must equal the reference bits.
    #[test]
    fn forward_hidden_fused_residuals_and_shared_mask_are_bit_identical() {
        let model = three_layer_tiny_draft();
        let mut embedding = Embedding::new(32, 64).unwrap();
        embedding
            .load_weight(
                &MxArray::random_normal(&[32, 64], 0.0, 0.5, Some(DType::BFloat16)).unwrap(),
            )
            .unwrap();
        let mut cases = 0;
        for context_rows in [3usize, 12, 30] {
            let mut context = super::DFlash2ContextCache::new(&model.config);
            let fused = MxArray::random_normal(
                &[1, context_rows as i64, 64],
                0.0,
                1.0,
                Some(DType::BFloat16),
            )
            .unwrap();
            let tokens: Vec<u32> = (0..context_rows as u32).map(|t| 3 + t % 20).collect();
            context.append(&model, &fused, 0, &tokens).unwrap();
            let base = context.logical_len;
            let block = MxArray::from_int32(&[7, 0, 0], &[1, 3]).unwrap();
            let got = model
                .forward_hidden(&embedding, &block, base, &mut context)
                .unwrap();
            let want = reference_forward_hidden(&model, &embedding, &block, base, &context);
            got.eval();
            want.eval();
            assert_eq!(
                got.to_uint16_native().unwrap(),
                want.to_uint16_native().unwrap(),
                "context_rows={context_rows}"
            );
            cases += 1;
        }
        assert_eq!(cases, 3);
    }
}
