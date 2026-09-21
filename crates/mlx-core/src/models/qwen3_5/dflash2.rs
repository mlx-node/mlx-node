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
use crate::models::gemma4::layer_cache::Gemma4LayerCache;
use crate::models::quantized_linear::{LinearProj, QuantizedLinear};
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

struct DFlash2Attention {
    q_proj: LinearProj,
    k_proj: LinearProj,
    v_proj: LinearProj,
    o_proj: LinearProj,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    rope: RoPE,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    sliding_window: i64,
}

impl DFlash2Attention {
    fn project_context(&self, x: &MxArray, base: i32) -> Result<(MxArray, MxArray)> {
        let batch = x.shape_at(0)?;
        let seq = x.shape_at(1)?;
        let keys =
            self.k_proj
                .forward(x)?
                .reshape(&[batch, seq, self.num_kv_heads, self.head_dim])?;
        let keys = self.k_norm.forward(&keys)?.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = self.rope.forward(&keys, Some(base))?;
        let values = self
            .v_proj
            .forward(x)?
            .reshape(&[batch, seq, self.num_kv_heads, self.head_dim])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        Ok((keys, values))
    }

    fn forward(
        &self,
        x: &MxArray,
        context: Option<&(MxArray, MxArray)>,
        context_base: i32,
        query_base: i32,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq = x.shape_at(1)?;
        let queries =
            self.q_proj
                .forward(x)?
                .reshape(&[batch, seq, self.num_heads, self.head_dim])?;
        let queries = self
            .q_norm
            .forward(&queries)?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let queries = self.rope.forward(&queries, Some(query_base))?;
        let (block_keys, block_values) = self.project_context(x, query_base)?;
        let (keys, values, key_base) = match context {
            Some((context_keys, context_values)) => (
                MxArray::concatenate(context_keys, &block_keys, 2)?,
                MxArray::concatenate(context_values, &block_values, 2)?,
                context_base,
            ),
            None => (block_keys, block_values, query_base),
        };
        let key_len = keys.shape_at(2)?;
        let mask =
            non_causal_sliding_mask(query_base, seq, key_base, key_len, self.sliding_window)?;
        let attended = scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
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
}

impl DFlash2Mlp {
    fn forward(&self, hidden: &MxArray) -> Result<MxArray> {
        let gated = Activations::swiglu_compiled(
            &self.gate_proj.forward(hidden)?,
            &self.up_proj.forward(hidden)?,
        )?;
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
        if std::env::var_os("MLX_DFLASH2_CONV_ELEMENTWISE").is_some() {
            return None;
        }
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

impl DFlash2Layer {
    fn forward(
        &self,
        hidden: &MxArray,
        context: Option<&(MxArray, MxArray)>,
        context_base: i32,
        query_base: i32,
    ) -> Result<MxArray> {
        let residual = hidden;
        let (prepared, dynamic) = self
            .attention_conv
            .prepare(&self.input_norm.forward(hidden)?)?;
        let attention = self
            .attention
            .forward(&prepared, context, context_base, query_base)?;
        let hidden = residual.add(&self.attention_conv.finish(&attention, &dynamic)?)?;
        let (prepared, dynamic) = self
            .mlp_conv
            .prepare(&self.post_attention_norm.forward(&hidden)?)?;
        hidden.add(
            &self
                .mlp_conv
                .finish(&self.mlp.forward(&prepared)?, &dynamic)?,
        )
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
        if self.top_k != 16
            || std::env::var_os("MLX_DISABLE_DFLASH2_TOPK16").is_some()
            || !unsafe { sys::mlx_metal_is_available() }
        {
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
            // Device walk: per position, gather the score row addressed by
            // the running predecessor index, argmax it, gather that column's
            // candidate token. ~4 lazy ops per position, zero host reads —
            // the path stays a graph node the verify block consumes directly.
            let mut predecessor = MxArray::from_int32(&[0], &[1])?;
            // The host walk's `max_by` keeps the LAST maximum on score ties;
            // argmax returns the first. Argmax over the reversed row and
            // un-reverse the index so both walks agree bit-for-bit. (For NaN
            // scores the walks can still diverge — `total_cmp` ranks NaN
            // above +inf while argmax skips NaN — but NaN selector scores
            // are already-corrupt upstream state.)
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
                // candidates: [1, L, K] — gather slot `selected` of row
                // `position` (the host walk's `candidate_ids[position*K + sel]`).
                let cand_i = candidates.slice_axis(1, position as i64, position as i64 + 1)?;
                tokens.push(cand_i.take(&selected, 2)?.reshape(&[1])?); // [1]
                predecessor = selected;
            }
            let path = MxArray::concatenate_many(tokens.iter().collect(), Some(0))?;
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

pub(crate) struct DFlash2ContextCache {
    layers: Vec<Gemma4LayerCache>,
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
                .map(|_| Gemma4LayerCache::new_sliding(config.sliding_window as i32 - 1))
                .collect(),
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
        for (layer, cache) in model.layers.iter().zip(self.layers.iter_mut()) {
            let (keys, values) = layer.attention.project_context(fused_context, base)?;
            let _ = cache.update_and_fetch(&keys, &values)?;
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

    pub(crate) fn eval(&self) -> Result<()> {
        let mut arrays = Vec::new();
        for cache in &self.layers {
            cache.collect_cache_arrays(&mut arrays);
        }
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

    fn forward_hidden(
        &self,
        target_embedding: &Embedding,
        block_ids: &MxArray,
        query_base: i32,
        context: &DFlash2ContextCache,
    ) -> Result<MxArray> {
        let mut hidden = target_embedding.forward(block_ids)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let cached = context.layers[index].get_cached_kv();
            let live_len = cached
                .as_ref()
                .map(|(keys, _)| keys.shape_at(2))
                .transpose()?
                .unwrap_or(0) as i32;
            let context_base = context.logical_len.saturating_sub(live_len);
            hidden = layer.forward(&hidden, cached.as_ref(), context_base, query_base)?;
        }
        self.norm.forward(&hidden)
    }

    pub(crate) fn propose<R: Rng + ?Sized>(
        &self,
        target_embedding: &Embedding,
        target_lm_head: Option<&LinearProj>,
        context: &DFlash2ContextCache,
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
        let hidden = self.forward_hidden(target_embedding, &block, context.logical_len, context)?;
        let hidden = hidden.slice_axis(1, 1, max_len as i64 + 1)?;
        let logits = match target_lm_head {
            Some(head) => head.forward(&hidden)?,
            None => target_embedding.as_linear(&hidden)?,
        };
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

fn required(params: &mut HashMap<String, MxArray>, key: &str, shape: &[i64]) -> Result<MxArray> {
    let value = params
        .remove(key)
        .ok_or_else(|| Error::from_reason(format!("DFlash2 checkpoint is missing '{key}'")))?;
    if value.shape()?.as_ref() != shape {
        return Err(Error::from_reason(format!(
            "DFlash2 tensor '{key}' has shape {:?}, expected {shape:?}",
            value.shape()?.as_ref()
        )));
    }
    if !matches!(
        value.dtype()?,
        DType::Float16 | DType::BFloat16 | DType::Float32
    ) {
        return Err(Error::from_reason(format!(
            "DFlash2 tensor '{key}' must be floating point"
        )));
    }
    Ok(value)
}

/// Load-time quantization for the dense DFlash2 companion. The published
/// checkpoints ship bf16 only; affine-quantizing the draft projections cuts
/// ~2.6 GB of per-cycle weight streaming on this model. Draft numerics affect
/// only the proposal acceptance rate — the target verifies every emitted
/// token, so output correctness is preserved regardless of draft precision.
///
/// `MLX_DFLASH2_DRAFT_QUANT`: `off`/`bf16` (default — bit-exact draft),
/// `q4` (affine 4-bit, group 64), `q8` (affine 8-bit, group 32).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DraftQuantization {
    Off,
    Affine4x64,
    Affine8x32,
}

impl DraftQuantization {
    fn params(self) -> Option<(i32, i32)> {
        match self {
            DraftQuantization::Off => None,
            DraftQuantization::Affine4x64 => Some((64, 4)),
            DraftQuantization::Affine8x32 => Some((32, 8)),
        }
    }
}

fn draft_quantization() -> DraftQuantization {
    static QUANT: std::sync::OnceLock<DraftQuantization> = std::sync::OnceLock::new();
    *QUANT.get_or_init(|| {
        let value = std::env::var("MLX_DFLASH2_DRAFT_QUANT")
            .unwrap_or_default()
            .to_ascii_lowercase();
        match value.as_str() {
            "q4" | "4" | "affine4" | "int4" => DraftQuantization::Affine4x64,
            "q8" | "8" | "affine8" | "int8" => DraftQuantization::Affine8x32,
            "" | "off" | "0" | "none" | "bf16" | "false" => DraftQuantization::Off,
            other => {
                eprintln!(
                    "warning: unrecognized MLX_DFLASH2_DRAFT_QUANT='{other}', expected q4|q8|off; using off"
                );
                DraftQuantization::Off
            }
        }
    })
}

/// Draft-only clone of the target LM head at [`draft_quantization`]
/// precision. The draft runs the head every cycle only to feed the selector's
/// top-k, so proposal logits do not need the target's q6k bits — a q4 copy
/// streams ~0.56 GB less per cycle. The verify head stays the target's
/// original projection, so emitted tokens are unaffected.
///
/// Returns the clone plus its resident byte count so the caller can fold the
/// extra allocation into model-residency accounting — the clone is a fresh
/// packed copy, not a view of the target head.
pub(crate) fn build_draft_lm_head(
    target: Option<&LinearProj>,
) -> Result<Option<(LinearProj, u64)>> {
    let Some((group_size, bits)) = draft_quantization().params() else {
        return Ok(None);
    };
    let Some(proj) = target else {
        return Ok(None);
    };
    let (dense, bias) = match proj {
        LinearProj::Standard(l) => (l.get_weight().astype(DType::BFloat16)?, l.get_bias()),
        // A Hadamard projection transforms the input at forward time;
        // dense_weight_bf16 returns the stored (rotated-space) weight, so a
        // plain affine clone would multiply untransformed hidden states and
        // produce invalid proposal logits. The loader already rejects
        // prism+draft checkpoints, but if that ever changes, skip the clone —
        // proposals then fall back to the target head (slower, still correct).
        LinearProj::Quantized(ql) if ql.has_hadamard() => return Ok(None),
        // fp8_e4m3/sym8 heads keep crate-specific storage that the generic
        // dequantizer cannot read — dense_weight_bf16 dispatches to the
        // retained/manual reconstruction for those modes.
        LinearProj::Quantized(ql) => (ql.dense_weight_bf16()?, ql.additive_bias().cloned()),
    };
    let (packed, scales, biases) = quantize_affine(&dense, group_size, bits)?;
    // The additive bias clone shares the target head's storage — already
    // counted in the params fold — so only the fresh packed arrays count here.
    let resident = packed.nbytes() as u64 + scales.nbytes() as u64 + biases.nbytes() as u64;
    Ok(Some((
        LinearProj::Quantized(QuantizedLinear::new(
            packed,
            scales,
            Some(biases),
            bias,
            group_size,
            bits,
            "affine".to_string(),
        )),
        resident,
    )))
}

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

/// Builds a draft projection, honoring [`draft_quantization`]. `savings`
/// accumulates `dense − resident` bytes so the loader can report true
/// residency rather than the bf16 file size.
fn linear(
    params: &mut HashMap<String, MxArray>,
    prefix: &str,
    input: usize,
    output: usize,
    quant: DraftQuantization,
    savings: &mut u64,
) -> Result<LinearProj> {
    let weight = required(
        params,
        &format!("{prefix}.weight"),
        &[output as i64, input as i64],
    )?;
    if let Some((group_size, bits)) = quant.params() {
        let (packed, scales, biases) = quantize_affine(&weight, group_size, bits)?;
        let resident = packed.nbytes() as u64 + scales.nbytes() as u64 + biases.nbytes() as u64;
        *savings += (weight.nbytes() as u64).saturating_sub(resident);
        return Ok(LinearProj::Quantized(QuantizedLinear::new(
            packed,
            scales,
            Some(biases),
            None,
            group_size,
            bits,
            "affine".to_string(),
        )));
    }
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
    quant: DraftQuantization,
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
        kernel_projection: linear(
            params,
            &format!("{base}.{name}.kernel_projection"),
            hidden,
            2 * config.conv_kernel_size * groups,
            quant,
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
        if array.shape()?.as_ref() != shape.as_slice() {
            return Err(Error::from_reason(format!(
                "DFlash2 tensor '{name}' has shape {:?}, expected {shape:?}",
                array.shape()?.as_ref()
            )));
        }
        if !matches!(
            array.dtype()?,
            DType::Float16 | DType::BFloat16 | DType::Float32
        ) {
            return Err(Error::from_reason(format!(
                "DFlash2 tensor '{name}' must be floating point"
            )));
        }
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
    let quant = draft_quantization();
    let mut savings = 0u64;
    let fc = linear(
        &mut params,
        "fc",
        hidden * config.target_layers.len(),
        hidden,
        quant,
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
            q_proj: linear(
                &mut params,
                &format!("{attention_base}.q_proj"),
                hidden,
                config.num_attention_heads * config.head_dim,
                quant,
                &mut savings,
            )?,
            k_proj: linear(
                &mut params,
                &format!("{attention_base}.k_proj"),
                hidden,
                config.num_key_value_heads * config.head_dim,
                quant,
                &mut savings,
            )?,
            v_proj: linear(
                &mut params,
                &format!("{attention_base}.v_proj"),
                hidden,
                config.num_key_value_heads * config.head_dim,
                quant,
                &mut savings,
            )?,
            o_proj: linear(
                &mut params,
                &format!("{attention_base}.o_proj"),
                config.num_attention_heads * config.head_dim,
                hidden,
                quant,
                &mut savings,
            )?,
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
            gate_proj: linear(
                &mut params,
                &format!("{mlp_base}.gate_proj"),
                hidden,
                config.intermediate_size,
                quant,
                &mut savings,
            )?,
            up_proj: linear(
                &mut params,
                &format!("{mlp_base}.up_proj"),
                hidden,
                config.intermediate_size,
                quant,
                &mut savings,
            )?,
            down_proj: linear(
                &mut params,
                &format!("{mlp_base}.down_proj"),
                config.intermediate_size,
                hidden,
                quant,
                &mut savings,
            )?,
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
        let attention_conv = grouped_conv(
            &mut params,
            &base,
            "attention_conv",
            &config,
            quant,
            &mut savings,
        )?;
        let mlp_conv = grouped_conv(&mut params, &base, "mlp_conv", &config, quant, &mut savings)?;
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
        // The selector's hidden projection stays dense: at 2.6 MB it is not
        // worth risking score fidelity for the predecessor walk.
        hidden_projection: linear(
            &mut params,
            "candidate_selector.hidden_projection",
            hidden,
            config.selector_rank,
            DraftQuantization::Off,
            &mut savings,
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
    if quant != DraftQuantization::Off {
        eprintln!(
            "dflash2 draft quant {quant:?}: resident {:.2} GiB (saved {:.2} GiB)",
            weight_bytes as f64 / (1 << 30) as f64,
            savings as f64 / (1 << 30) as f64,
        );
    }
    Ok((
        DFlash2Model {
            config,
            weight_bytes,
            fc,
            hidden_norm,
            layers,
            norm: final_norm,
            selector,
        },
        weight_bytes,
    ))
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

    /// The sharded top-16 kernel must return exactly the argpartition
    /// candidate SET per row — same ids, matching logits values, ascending
    /// value order. Called through the FFI directly so the test can never
    /// pass vacuously on the fallback path.
    #[test]
    fn fused_topk16_matches_argpartition_candidate_set() {
        let vocab: i64 = 8192;
        let logits =
            MxArray::random_normal(&[1, 5, vocab], 0.0, 1.0, Some(DType::Float32)).unwrap();
        let mut ids = std::ptr::null_mut();
        let mut values = std::ptr::null_mut();
        if !unsafe { mlx_sys::mlx_dflash2_topk16(logits.as_raw_ptr(), &mut ids, &mut values) } {
            return; // no Metal backend
        }
        let ids = MxArray::from_handle(ids, "topk16 ids").unwrap();
        let values = MxArray::from_handle(values, "topk16 values").unwrap();
        let ids: Vec<i32> = ids.to_int32().unwrap().as_ref().to_vec();
        let values: Vec<f32> = values.to_float32().unwrap().as_ref().to_vec();
        let logits_f: Vec<f32> = logits.to_float32().unwrap().as_ref().to_vec();

        let reference = logits
            .argpartition(-16, Some(-1))
            .unwrap()
            .slice_axis(2, vocab - 16, vocab)
            .unwrap();
        let ref_ids: Vec<i32> = reference.to_int32().unwrap().as_ref().to_vec();

        for row in 0..5usize {
            let mine = &ids[row * 16..(row + 1) * 16];
            let mine_v = &values[row * 16..(row + 1) * 16];
            let theirs = &ref_ids[row * 16..(row + 1) * 16];
            // Same candidate set (tie order may differ between the two
            // orderings — compare sorted).
            let mut a = mine.to_vec();
            let mut b = theirs.to_vec();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "row {row}: candidate sets differ");
            // Ascending order + values equal the logits at those indices.
            assert!(
                mine_v.windows(2).all(|w| w[0] <= w[1]),
                "row {row}: values not ascending: {mine_v:?}"
            );
            for (i, (&id, &v)) in mine.iter().zip(mine_v.iter()).enumerate() {
                assert!(
                    id >= 0 && (id as i64) < vocab,
                    "row {row} slot {i}: bad id {id}"
                );
                let expected = logits_f[row * vocab as usize + id as usize];
                assert_eq!(v, expected, "row {row} slot {i}: value mismatch");
            }
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
        match super::draft_quantization() {
            super::DraftQuantization::Off => assert!(bytes > 3_000_000_000),
            _ => assert!(bytes > 500_000_000 && bytes < 3_000_000_000),
        }
    }
}
