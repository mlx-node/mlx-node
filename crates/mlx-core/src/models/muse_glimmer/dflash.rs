//! Meta DFlash companion for Muse-Glimmer.
//!
//! The drafter consumes five post-layer target residuals, projects them into a
//! context stream, and predicts a 16-token block in one non-causal forward.
//! It shares the target token embedding and LM head.

use napi::bindgen_prelude::*;
use rand::Rng;

use crate::array::attention::scaled_dot_product_attention;
use crate::array::{DType, MxArray};
use crate::models::gemma4::layer_cache::Gemma4LayerCache;
use crate::models::gemma4::quantized_linear::LinearProj;
use crate::nn::{Embedding, RMSNorm, RoPE};
use crate::sampling::{SamplingConfig, is_greedy_temperature, sampling_distribution};

use super::config::{MuseGlimmerDFlashConfig, MuseGlimmerTextConfig};
use super::mlp::MuseGlimmerMlp;

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
    let distance = queries.sub(&keys)?;
    let window = MxArray::scalar_int(window as i32)?;
    Ok(Some(
        distance
            .less(&window)?
            .reshape(&[1, 1, query_len, key_len])?,
    ))
}

pub(crate) struct DFlashAttention {
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

impl DFlashAttention {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        config: &MuseGlimmerDFlashConfig,
        q_proj: LinearProj,
        k_proj: LinearProj,
        v_proj: LinearProj,
        o_proj: LinearProj,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            // The official DFlash GGUF intentionally keeps HF's half-split
            // Q/K rows; unlike the target GGUF it is not unpermuted.
            rope: RoPE::new(
                config.head_dim as i32,
                Some(false),
                Some(config.rope_theta as f64),
                None,
            ),
            num_heads: config.num_attention_heads as i64,
            num_kv_heads: config.num_key_value_heads as i64,
            head_dim: config.head_dim as i64,
            sliding_window: config.sliding_window as i64,
        }
    }

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

pub(crate) struct DFlashLayer {
    attention: DFlashAttention,
    mlp: MuseGlimmerMlp,
    input_norm: RMSNorm,
    post_attention_norm: RMSNorm,
}

impl DFlashLayer {
    pub(crate) fn new(
        attention: DFlashAttention,
        mlp: MuseGlimmerMlp,
        input_norm: RMSNorm,
        post_attention_norm: RMSNorm,
    ) -> Self {
        Self {
            attention,
            mlp,
            input_norm,
            post_attention_norm,
        }
    }

    fn forward(
        &self,
        x: &MxArray,
        context: Option<&(MxArray, MxArray)>,
        context_base: i32,
        query_base: i32,
    ) -> Result<MxArray> {
        let attention = self.attention.forward(
            &self.input_norm.forward(x)?,
            context,
            context_base,
            query_base,
        )?;
        let hidden = x.add(&attention)?;
        let mlp = self
            .mlp
            .forward(&self.post_attention_norm.forward(&hidden)?)?;
        hidden.add(&mlp)
    }
}

pub(crate) struct DFlashContextCache {
    layers: Vec<Gemma4LayerCache>,
    logical_len: i32,
}

impl DFlashContextCache {
    pub(crate) fn new_at(config: &MuseGlimmerDFlashConfig, logical_len: i32) -> Self {
        Self {
            layers: (0..config.num_hidden_layers)
                .map(|_| Gemma4LayerCache::new_sliding(config.sliding_window as i32))
                .collect(),
            logical_len,
        }
    }

    pub(crate) fn append(
        &mut self,
        model: &DFlashModel,
        fused_context: &MxArray,
        base: i32,
    ) -> Result<()> {
        if base != self.logical_len {
            return Err(Error::from_reason(format!(
                "Muse-Glimmer DFlash context append starts at {base}, expected {}",
                self.logical_len
            )));
        }
        let rows = fused_context.shape_at(1)? as i32;
        for (layer, cache) in model.layers.iter().zip(self.layers.iter_mut()) {
            let (keys, values) = layer.attention.project_context(fused_context, base)?;
            let _ = cache.update_and_fetch(&keys, &values)?;
        }
        self.logical_len = self.logical_len.saturating_add(rows);
        Ok(())
    }

    pub(crate) fn logical_len(&self) -> i32 {
        self.logical_len
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

pub(crate) struct DFlashModel {
    pub(crate) config: MuseGlimmerDFlashConfig,
    fc: LinearProj,
    hidden_norm: RMSNorm,
    layers: Vec<DFlashLayer>,
    norm: RMSNorm,
}

impl DFlashModel {
    pub(crate) fn from_loaded(
        config: MuseGlimmerDFlashConfig,
        fc: LinearProj,
        hidden_norm: RMSNorm,
        layers: Vec<DFlashLayer>,
        norm: RMSNorm,
    ) -> Self {
        Self {
            config,
            fc,
            hidden_norm,
            layers,
            norm,
        }
    }

    pub(crate) fn fuse_context(&self, taps: &[MxArray]) -> Result<MxArray> {
        if taps.len() != self.config.target_layers.len() || taps.is_empty() {
            return Err(Error::from_reason(format!(
                "Muse-Glimmer DFlash expects {} target taps, got {}",
                self.config.target_layers.len(),
                taps.len()
            )));
        }
        let refs = taps.iter().collect::<Vec<_>>();
        let concatenated = MxArray::concatenate_many(refs, Some(2))?;
        self.hidden_norm.forward(&self.fc.forward(&concatenated)?)
    }

    pub(crate) fn forward_block(
        &self,
        target_embedding: &Embedding,
        target_lm_head: Option<&LinearProj>,
        target_config: &MuseGlimmerTextConfig,
        block_ids: &MxArray,
        query_base: i32,
        context: &DFlashContextCache,
    ) -> Result<MxArray> {
        let mut hidden = target_embedding.forward(block_ids)?;
        let context_len = context.logical_len;
        for (index, layer) in self.layers.iter().enumerate() {
            let cached = context.layers[index].get_cached_kv();
            let live_len = cached
                .as_ref()
                .map(|(keys, _)| keys.shape_at(2))
                .transpose()?
                .unwrap_or(0) as i32;
            let context_base = context_len.saturating_sub(live_len);
            hidden = layer.forward(&hidden, cached.as_ref(), context_base, query_base)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let logits = match target_lm_head {
            Some(lm_head) => lm_head.forward(&hidden)?,
            None => target_embedding.as_linear(&hidden)?,
        }
        .mul_scalar(target_config.output_multiplier as f64)?;
        let cap = target_config.final_logit_softcapping as f64;
        logits.div_scalar(cap)?.tanh()?.mul_scalar(cap)
    }

    pub(crate) fn propose<R: Rng + ?Sized>(
        &self,
        target_embedding: &Embedding,
        target_lm_head: Option<&LinearProj>,
        target_config: &MuseGlimmerTextConfig,
        context: &DFlashContextCache,
        anchor: u32,
        max_len: usize,
        sampling: &SamplingConfig,
        rng: &mut R,
    ) -> Result<(Vec<i32>, Vec<MxArray>)> {
        // DFlash is parallel infill, not an autoregressive draft. The query
        // contains one bonus/anchor token followed by K mask tokens, and only
        // the K mask rows are projected to draft distributions. Sampling the
        // anchor row shifts every proposal by one: position zero can still
        // look plausible, but every later acceptance collapses to zero.
        let ids = parallel_query_ids(anchor, self.config.mask_token_id, max_len);
        let block = MxArray::from_int32(&ids, &[1, ids.len() as i64])?;
        let logits = self.forward_block(
            target_embedding,
            target_lm_head,
            target_config,
            &block,
            context.logical_len,
            context,
        )?;
        let greedy = is_greedy_temperature(sampling.temperature.unwrap_or(1.0));
        let vocab = target_config.vocab_size as i64;
        let mut draft_ids = Vec::with_capacity(max_len);
        let mut distributions = Vec::with_capacity(if greedy { 0 } else { max_len });
        for position in 0..max_len {
            let query_row = position as i64 + 1;
            let row = logits
                .slice_axis(1, query_row, query_row + 1)?
                .reshape(&[vocab])?;
            if greedy {
                draft_ids.push(row.argmax(0, Some(false))?.astype(DType::Int32)?);
            } else {
                let distribution =
                    sampling_distribution(&row, Some(*sampling))?.astype(DType::Float32)?;
                draft_ids.push(crate::sampling::sample_dense_distribution_array(
                    &distribution,
                    rng,
                )?);
                distributions.push(distribution);
            }
        }
        Ok((
            crate::sampling::materialize_draft_tokens(&draft_ids)?,
            distributions,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::parallel_query_ids;

    #[test]
    fn block_sixteen_has_one_anchor_and_sixteen_sampled_mask_rows() {
        let ids = parallel_query_ids(7, 201_818, 16);
        assert_eq!(ids.len(), 17);
        assert_eq!(ids[0], 7);
        assert!(ids[1..].iter().all(|&id| id == 201_818));
    }
}
