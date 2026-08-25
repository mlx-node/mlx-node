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

use napi::bindgen_prelude::*;
use rand::{Rng, RngExt};
use serde::Deserialize;

use crate::array::attention::scaled_dot_product_attention;
use crate::array::{DType, MxArray};
use crate::models::gemma4::layer_cache::Gemma4LayerCache;
use crate::models::qwen3_5::quantized_linear::LinearProj;
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
    sliding_window: usize,
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
        let gated = Activations::silu(&self.gate_proj.forward(hidden)?)?;
        self.down_proj
            .forward(&gated.mul(&self.up_proj.forward(hidden)?)?)
    }
}

struct GroupedDynamicCausalConv {
    base_kernel: MxArray,
    kernel_projection: LinearProj,
    kernel_size: usize,
    group_size: usize,
}

impl GroupedDynamicCausalConv {
    fn convolve(&self, hidden: &MxArray, dynamic: &MxArray, side: usize) -> Result<MxArray> {
        let batch = hidden.shape_at(0)?;
        let length = hidden.shape_at(1)?;
        let hidden_size = hidden.shape_at(2)?;
        let groups = hidden_size / self.group_size as i64;
        let blocks = hidden.reshape(&[batch, length, groups, self.group_size as i64])?;
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
        let dynamic = self.kernel_projection.forward(hidden)?.reshape(&[
            batch,
            length,
            2,
            self.kernel_size as i64,
            groups,
        ])?;
        let before = dynamic.slice_axis(2, 0, 1)?.squeeze(Some(&[2]))?;
        let after = dynamic.slice_axis(2, 1, 2)?.squeeze(Some(&[2]))?;
        Ok((self.convolve(hidden, &before, 0)?, after))
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

impl CandidateSelector {
    fn select<R: Rng + ?Sized>(
        &self,
        hidden: &MxArray,
        logits: &MxArray,
        anchor: u32,
        temperature: f64,
        rng: &mut R,
    ) -> Result<(Vec<i32>, Vec<SparseDistribution>)> {
        let length = hidden.shape_at(1)? as usize;
        let candidates = logits
            .argpartition(-(self.top_k as i32), Some(-1))?
            .slice_axis(
                2,
                self.vocab_size as i64 - self.top_k as i64,
                self.vocab_size as i64,
            )?;
        let unary = logits.take_along_axis(&candidates, -1)?;
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
        MxArray::eval_arrays(&[&candidates, &scores])?;
        let candidate_ids = candidates.to_int32()?;
        let edge_scores = scores.to_float32()?;

        let greedy = is_greedy_temperature(temperature);
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
        Ok((path, rows))
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
        rng: &mut R,
    ) -> Result<(Vec<i32>, Vec<SparseDistribution>)> {
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
        let block = MxArray::from_int32(&ids, &[1, ids.len() as i64])?;
        let hidden = self.forward_hidden(target_embedding, &block, context.logical_len, context)?;
        let hidden = hidden.slice_axis(1, 1, max_len as i64 + 1)?;
        let logits = match target_lm_head {
            Some(head) => head.forward(&hidden)?,
            None => target_embedding.as_linear(&hidden)?,
        };
        self.selector
            .select(&hidden, &logits, anchor, temperature, rng)
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

fn linear(
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
    let fc = linear(
        &mut params,
        "fc",
        hidden * config.target_layers.len(),
        hidden,
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
            )?,
            k_proj: linear(
                &mut params,
                &format!("{attention_base}.k_proj"),
                hidden,
                config.num_key_value_heads * config.head_dim,
            )?,
            v_proj: linear(
                &mut params,
                &format!("{attention_base}.v_proj"),
                hidden,
                config.num_key_value_heads * config.head_dim,
            )?,
            o_proj: linear(
                &mut params,
                &format!("{attention_base}.o_proj"),
                config.num_attention_heads * config.head_dim,
                hidden,
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
            )?,
            up_proj: linear(
                &mut params,
                &format!("{mlp_base}.up_proj"),
                hidden,
                config.intermediate_size,
            )?,
            down_proj: linear(
                &mut params,
                &format!("{mlp_base}.down_proj"),
                config.intermediate_size,
                hidden,
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
        let attention_conv = grouped_conv(&mut params, &base, "attention_conv", &config)?;
        let mlp_conv = grouped_conv(&mut params, &base, "mlp_conv", &config)?;
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
        hidden_projection: linear(
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
    use super::{normalized_selector_probs, parallel_query_ids};

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
    #[ignore = "requires the external 3.6 GB DFlash2 checkpoint"]
    fn loads_real_qwen38_dflash2_checkpoint_strictly() {
        let path = std::env::var("MLX_TEST_QWEN38_DFLASH2_PATH")
            .expect("set MLX_TEST_QWEN38_DFLASH2_PATH");
        let (model, bytes) = super::load_dflash2(std::path::Path::new(&path))
            .expect("real DFlash2 checkpoint must load");
        assert_eq!(model.config.block_size, 7);
        assert_eq!(model.config.target_layers, vec![5, 19, 33, 47, 61]);
        assert!(bytes > 3_000_000_000);
    }
}
