//! External DFlash2 drafter for dense and MoE Qwen3.5/3.8 targets.
//!
//! DFlash2 consumes the checkpoint-selected target residual streams, runs its
//! parallel draft transformer, and chooses a Markov path through the top-16
//! token candidates at each position. The drafter shares the target embedding
//! and language-model head; its checkpoint therefore contains only the draft
//! transformer, two-tap grouped dynamic convolutions, and selector codebooks.

mod cache;
mod compiled;
mod config;
#[cfg(test)]
mod flash_next_tests;
mod layers;
mod loader;
mod selector;
mod spanned_head;
#[cfg(test)]
mod test_helpers;
#[cfg(test)]
mod tests;

use std::sync::atomic::Ordering;

use mlx_sys as sys;
use napi::bindgen_prelude::*;
use rand::Rng;

pub(crate) use cache::DFlash2ContextCache;
use cache::kquant_rows_merge;
#[cfg(test)]
use cache::store_window_tail;
#[cfg(test)]
pub(crate) use cache::{DraftKvWindow, RING_ALLOCATIONS};
use compiled::CompiledDraft;
pub(crate) use config::DFlash2Config;
use config::DraftExecutionPolicy;
#[cfg(test)]
use layers::{DFlash2Attention, DFlash2Mlp, GroupedDynamicCausalConv};
use layers::{DFlash2Layer, DraftLayerOutput, DraftResidual};
#[cfg(test)]
use loader::{
    dense_linear, draft_linear, expected_tensor_shapes, quantize_draft_weight,
    validate_tensor_inventory,
};
pub(crate) use loader::{dflash2_file_bytes, load_dflash2, read_dflash2_config};
use selector::CandidateSelector;
pub(crate) use selector::SelectorPath;
#[cfg(test)]
pub(crate) use selector::device_sampled_path;
#[cfg(test)]
use selector::normalized_selector_probs;
use spanned_head::cuda_spanned_head_enabled;
#[cfg(test)]
pub(crate) use test_helpers::{
    dflash2_config_with_context, tiny_dflash2_config, tiny_dflash2_model_for_stepper_test,
    tiny_dflash2_model_with_block, write_test_checkpoint,
};

use crate::array::{DType, MxArray};
#[cfg(test)]
use crate::engine::dspark_turn::non_causal_sliding_mask;
use crate::engine::dspark_turn::parallel_query_ids;
use crate::models::quantized_linear::{LinearProj, QuantizedLinear};
use crate::nn::{Embedding, RMSNorm};
use crate::sampling::{SparseDistribution, is_greedy_temperature};

/// Proposals one DFlash2 cycle drafts (Splash `SPLASH_DRAFT_PROPOSAL_TOKENS`).
pub(crate) const DFLASH2_DRAFT_PROPOSALS: usize = 7;

/// Rows of the one verify graph: the anchor and the proposals (Splash
/// `SPLASH_TARGET_VERIFY_ROWS`).
pub(crate) const DFLASH2_VERIFY_ROWS: usize = DFLASH2_DRAFT_PROPOSALS + 1;

/// What a draft must match in its target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DFlash2TargetShape {
    pub(crate) hidden_size: usize,
    pub(crate) vocab_size: usize,
    pub(crate) valid_vocab_size: usize,
    pub(crate) num_layers: usize,
    pub(crate) final_logit_softcapping: Option<f32>,
    pub(crate) reads_layer_inputs: bool,
}

impl DFlash2TargetShape {
    pub(crate) const fn new(hidden_size: usize, vocab_size: usize, num_layers: usize) -> Self {
        Self {
            hidden_size,
            vocab_size,
            valid_vocab_size: vocab_size,
            num_layers,
            final_logit_softcapping: None,
            reads_layer_inputs: false,
        }
    }

    /// Number of target layer indices a draft may tap (`num_layers`, or
    /// `num_layers + 1` when the target reads pre-layer inputs and exposes
    /// the post-final-norm residual at index `num_layers`).
    pub(crate) const fn readable_layers(self) -> usize {
        self.num_layers + self.reads_layer_inputs as usize
    }
}

impl From<&super::config::Qwen3_5Config> for DFlash2TargetShape {
    fn from(target: &super::config::Qwen3_5Config) -> Self {
        Self::new(
            target.hidden_size as usize,
            target.vocab_size as usize,
            target.num_layers as usize,
        )
    }
}

/// The target's token table and vocabulary head as the draft reads them
/// (Splash `DFlashDraft::addDecode` takes `vocabularyProjection`).
pub(crate) trait DFlash2TargetView {
    /// `[1, rows, hidden]`: the token rows of `block`, whose host ids are
    /// `ids`.
    fn embed(&mut self, block: &MxArray, ids: &[i32]) -> Result<MxArray>;
    /// The head's logits of the draft's normed hidden rows.
    fn logits(&mut self, hidden: &MxArray) -> Result<MxArray>;
    /// The table and head as arrays a compiled draft may capture; `None`
    /// keeps every proposal eager unless [`Self::target_lm_head_qlinear`]
    /// provides a resident quantized head for embedded-input compilation or
    /// the compiled draft body runs on pre-embedded rows.
    fn traced(&self) -> Option<(&Embedding, Option<&LinearProj>)>;
    /// Resident quantized `lm_head` projection when [`Self::traced`] is `None`
    /// (e.g., external row-table embeddings), enabling CUDA spanned draft
    /// heads and compiled embedded-input draft graphs.
    fn target_lm_head_qlinear(&self) -> Option<&QuantizedLinear> {
        None
    }
    /// Optional cap on how many leading proposal positions from the full
    /// bidirectional draft block need `lm_head` + selector output (`None` =
    /// all `max_len` positions). Exact because `CandidateSelector` is strictly
    /// left-to-right causal across positions.
    fn proposal_head_limit(&self) -> Option<usize> {
        None
    }
    /// When `true`, [`Self::logits`] is invoked on ONLY the `head_len`
    /// proposal rows `[1, head_len, hidden]` instead of the full `1 + head_len`
    /// anchor+proposal block (used by targets whose `lm_head` kernel natively
    /// projects `M = 7` without wasted anchor compute, e.g. Inkling-2-Mini).
    fn project_proposals_only(&self) -> bool {
        false
    }
}

/// A Qwen3.5 target: its `Embedding` and its head, or the tied table.
pub(crate) struct DenseTargetView<'a> {
    pub(crate) embedding: &'a Embedding,
    pub(crate) lm_head: Option<&'a LinearProj>,
}

impl DFlash2TargetView for DenseTargetView<'_> {
    fn embed(&mut self, block: &MxArray, _ids: &[i32]) -> Result<MxArray> {
        self.embedding.forward(block)
    }

    fn logits(&mut self, hidden: &MxArray) -> Result<MxArray> {
        match self.lm_head {
            Some(head) => head.forward_logits(hidden),
            None => self.embedding.as_linear(hidden),
        }
    }

    fn traced(&self) -> Option<(&Embedding, Option<&LinearProj>)> {
        Some((self.embedding, self.lm_head))
    }
}

pub(crate) struct DFlash2Model {
    execution_policy: DraftExecutionPolicy,
    compiled: CompiledDraft,
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
    /// Optional `[1, 1, hidden]` BF16 mask-token row from `mask_embedding.safetensors`
    /// overriding query rows `1..` after target embedding lookup.
    mask_embedding: Option<MxArray>,
}

impl DFlash2Model {
    /// Dense CUDA uses the same compiled whole-block draft as the supported
    /// MoE profile. Metal retains the captured dense policy unchanged.
    pub(crate) fn configure_dense_execution(&mut self) {
        let cuda = crate::accelerator::cuda_ops_available();
        self.invalidate_compiled();
        self.execution_policy = self.execution_policy.with_dense_cuda_profile(cuda);
    }

    /// Called by the Inkling-2-Mini loader after `validate_target`: binds the
    /// target's unpadded active vocabulary (`200058` of `201024`) onto the
    /// selector and enables Metal compiled draft body + NAX attention when the
    /// draft geometry and device support them.
    pub(crate) fn configure_inkling2_execution(&mut self, target: DFlash2TargetShape) {
        self.selector.active_vocab_size = target.valid_vocab_size.min(self.selector.vocab_size);
        let supported = self.config.supports_inkling2_execution_defaults();
        let metal = unsafe { sys::mlx_metal_is_available() };
        let nax = supported && metal && unsafe { sys::mlx_metal_is_nax_available() };
        self.invalidate_compiled();
        self.execution_policy = self
            .execution_policy
            .with_inkling2_metal_profile(supported && metal, nax);
        for layer in &mut self.layers {
            layer.attention.tensor_attention_enabled = self.execution_policy.tensor_attention();
        }
    }

    pub(super) fn apply_mask_embedding(&self, embedded: MxArray) -> Result<MxArray> {
        let Some(ref mask_emb) = self.mask_embedding else {
            return Ok(embedded);
        };
        let rows = embedded.shape_at(1)?;
        if rows <= 1 {
            return Ok(embedded);
        }
        let anchor_row = embedded.slice_axis(1, 0, 1)?;
        let hidden = embedded.shape_at(2)?;
        let dtype = embedded.dtype()?;
        let mask_row = if mask_emb.dtype()? == dtype {
            mask_emb.clone()
        } else {
            mask_emb.astype(dtype)?
        };
        let repeated = mask_row.broadcast_to(&[1, rows - 1, hidden])?;
        MxArray::concatenate(&anchor_row, &repeated, 1)
    }

    /// Called only by the MoE loader after target validation and before the
    /// companion is published. Dense loaders retain the captured unset-off
    /// policy. Runtime hardware, dtype, shape and gradient guards still apply.
    pub(crate) fn configure_moe_execution(
        &mut self,
        target: &crate::models::qwen3_5_moe::Qwen3_5MoeConfig,
    ) {
        let target_profile = [
            target.hidden_size,
            target.num_layers,
            target.num_experts,
            target.num_experts_per_tok,
            target
                .moe_intermediate_size
                .unwrap_or(target.intermediate_size),
        ];
        let supported = self.config.supports_moe_execution_defaults(target_profile);
        // Also safe if a future internal caller changes policy after tracing:
        // erase captured graphs without clearing a prior fail-closed latch.
        self.invalidate_compiled();
        self.execution_policy = self.execution_policy.with_moe_profile(supported);
        for layer in &mut self.layers {
            layer.attention.tensor_attention_enabled = self.execution_policy.tensor_attention();
        }
    }

    pub(crate) fn validate_target(&self, target: impl Into<DFlash2TargetShape>) -> Result<()> {
        self.config.validate_target(target.into())
    }

    /// Content-bound fingerprint of the loaded, materialized fusion weights
    /// (`self.fc` + `self.hidden_norm`) and the config parameters that govern
    /// [`crate::models::qwen3_5::draft_context::DraftContextWindow`].
    ///
    /// Computed strictly from in-memory tensors so persistent SSD caches
    /// cannot suffer TOCTOU races or miss same-size/same-mtime weight edits.
    pub(crate) fn context_fusion_fingerprint(&self) -> Result<Vec<u8>> {
        fn push_array_bytes(
            components: &mut Vec<Vec<u8>>,
            tag: &[u8],
            arr: &MxArray,
        ) -> Result<()> {
            let dtype = arr.dtype()?;
            let shape = arr.shape()?;
            let mut meta = Vec::with_capacity(tag.len() + 12 + shape.len() * 8);
            meta.extend_from_slice(tag);
            meta.extend_from_slice(&(dtype as i32).to_le_bytes());
            meta.extend_from_slice(&(shape.len() as u64).to_le_bytes());
            for &dim in shape.as_ref() {
                meta.extend_from_slice(&dim.to_le_bytes());
            }
            components.push(meta);

            let raw = match dtype {
                DType::Uint8 => arr.to_uint8()?,
                DType::Int8 => arr.to_int8()?.into_iter().map(|v| v as u8).collect(),
                DType::Float16 | DType::BFloat16 => {
                    let vals = arr.to_uint16_native()?;
                    let mut out = Vec::with_capacity(vals.len() * 2);
                    for v in vals {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    out
                }
                DType::Uint32 => {
                    let vals = arr.to_uint32()?;
                    let mut out = Vec::with_capacity(vals.len() * 4);
                    for &v in vals.as_ref() {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    out
                }
                DType::Int32 => {
                    let vals = arr.to_int32()?;
                    let mut out = Vec::with_capacity(vals.len() * 4);
                    for &v in vals.as_ref() {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    out
                }
                DType::Float32 => {
                    let vals = arr.to_float32()?;
                    let mut out = Vec::with_capacity(vals.len() * 4);
                    for &v in vals.as_ref() {
                        out.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                    out
                }
            };
            components.push(raw);
            Ok(())
        }

        let mut components: Vec<Vec<u8>> = Vec::with_capacity(16);
        components.push(b"dflash2-fusion-weights:v1".to_vec());

        // Canonical config bytes governing `DraftContextWindow` tap extraction and fusion.
        let mut cfg_bytes = Vec::with_capacity(64 + self.config.target_layers.len() * 8);
        cfg_bytes.extend_from_slice(&(self.config.hidden_size as u64).to_le_bytes());
        cfg_bytes.extend_from_slice(&(self.config.sliding_window as u64).to_le_bytes());
        cfg_bytes.extend_from_slice(&(self.config.block_size as u64).to_le_bytes());
        cfg_bytes.extend_from_slice(&(self.config.target_num_layers as u64).to_le_bytes());
        cfg_bytes.extend_from_slice(&self.config.rms_norm_eps.to_bits().to_le_bytes());
        cfg_bytes.push(match self.config.capture_final_norm {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        });
        match self.config.embedding_normalization.as_deref() {
            None => cfg_bytes.push(0),
            Some(mode) => {
                cfg_bytes.push(1);
                cfg_bytes.extend_from_slice(&(mode.len() as u64).to_le_bytes());
                cfg_bytes.extend_from_slice(mode.as_bytes());
            }
        }
        cfg_bytes.extend_from_slice(&(self.config.target_layers.len() as u64).to_le_bytes());
        for &layer in &self.config.target_layers {
            cfg_bytes.extend_from_slice(&(layer as u64).to_le_bytes());
        }
        components.push(cfg_bytes);

        // `self.fc` projection metadata and exact materialized tensor bytes.
        match &self.fc {
            LinearProj::Quantized(q) => {
                let mut fc_meta = Vec::with_capacity(32 + q.mode().len());
                fc_meta.extend_from_slice(b"fc:quantized");
                fc_meta.extend_from_slice(&q.bits().to_le_bytes());
                fc_meta.extend_from_slice(&q.group_size().to_le_bytes());
                fc_meta.extend_from_slice(&(q.mode().len() as u64).to_le_bytes());
                fc_meta.extend_from_slice(q.mode().as_bytes());
                components.push(fc_meta);
                push_array_bytes(&mut components, b"fc.weight", q.get_weight())?;
                push_array_bytes(&mut components, b"fc.scales", q.get_scales())?;
                if let Some(biases) = q.get_biases() {
                    components.push(vec![1u8]);
                    push_array_bytes(&mut components, b"fc.biases", biases)?;
                } else {
                    components.push(vec![0u8]);
                }
            }
            LinearProj::Standard(lin) => {
                components.push(b"fc:standard".to_vec());
                let w = lin.get_weight();
                push_array_bytes(&mut components, b"fc.weight", &w)?;
                if let Some(b) = lin.get_bias() {
                    components.push(vec![1u8]);
                    push_array_bytes(&mut components, b"fc.bias", &b)?;
                } else {
                    components.push(vec![0u8]);
                }
            }
        }

        // `self.hidden_norm` policy, epsilon, and exact materialized scale bytes.
        let mut norm_meta = Vec::with_capacity(16);
        norm_meta.extend_from_slice(b"hidden_norm");
        norm_meta.extend_from_slice(&self.hidden_norm.eps_f32().to_bits().to_le_bytes());
        norm_meta.push(u8::from(self.hidden_norm.is_precise_f32()));
        components.push(norm_meta);
        push_array_bytes(
            &mut components,
            b"hidden_norm.weight",
            self.hidden_norm.weight(),
        )?;

        let fp = mlx_paged_attn::ColdCacheFingerprint::from_components(
            components.iter().map(Vec::as_slice),
        );
        Ok(fp.as_bytes().to_vec())
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
    /// The merged and separate projections are not bit-equal in general: the
    /// kernel tier and its split-K follow both the row count M and the width
    /// N, so at the published Qwen3.8 draft geometry on an M5 Max the merged
    /// context `k|v` differs from the per-layer projections at 1 and at 8
    /// rows. A draft K/V row's bits depend on the route and width it was
    /// projected at; only the same (route, width) reproduces it. Per-layer
    /// merges turn the originals into row views of the merged buffer (no
    /// extra bytes); the cross-layer `k|v` merge is an extra resident copy,
    /// counted in `weight_bytes`.
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
        self.project_context_rows(fused_context, base, None)
    }

    fn project_context_rows(
        &self,
        fused_context: &MxArray,
        base: i32,
        retained_rows: Option<i64>,
    ) -> Result<Vec<(MxArray, MxArray)>> {
        let Some(context_kv) = self.context_kv.as_ref().filter(|_| {
            fused_context
                .shape_at(1)
                .is_ok_and(|rows| rows <= self.merged_rows())
        }) else {
            return self
                .layers
                .iter()
                .map(|layer| {
                    if retained_rows.is_some() {
                        layer
                            .attention
                            .project_context_rows(fused_context, base, retained_rows)
                    } else {
                        layer.attention.project_context(fused_context, base)
                    }
                })
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
            let key_prefix = retained_rows
                .map(|rows| pair[0].slice_axis(1, 0, rows))
                .transpose()?;
            let key_rows = key_prefix.as_ref().unwrap_or(&pair[0]);
            let normed = attention.k_norm.forward(&key_rows.reshape(&[
                key_rows.shape_at(0)?,
                key_rows.shape_at(1)?,
                attention.num_kv_heads,
                attention.head_dim,
            ])?)?;
            keys.push(normed.transpose(Some(&[0, 2, 1, 3]))?);
            let value_prefix = retained_rows
                .map(|rows| pair[1].slice_axis(1, 0, rows))
                .transpose()?;
            values.push(attention.finish_values(value_prefix.as_ref().unwrap_or(&pair[1]))?);
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

    #[cfg(test)]
    fn forward_hidden(
        &self,
        target_embedding: &Embedding,
        block_ids: &MxArray,
        query_base: i32,
        context: &mut DFlash2ContextCache,
    ) -> Result<MxArray> {
        let embedded = target_embedding.forward(block_ids)?;
        self.forward_embedded(&embedded, query_base, context)
    }

    /// The draft layers and final norm over a block's embedded rows
    /// `[1, rows, hidden]`.
    fn forward_embedded(
        &self,
        embedded: &MxArray,
        query_base: i32,
        context: &mut DFlash2ContextCache,
    ) -> Result<MxArray> {
        let mut pending: Option<DraftLayerOutput> = None;
        let DFlash2ContextCache {
            layers: windows,
            mask_memo,
            ..
        } = context;
        // Every layer's window holds the same rows, so one mask serves all.
        let mask = match windows.first() {
            Some(window) => window.block_mask(
                mask_memo,
                query_base,
                embedded.shape_at(1)?,
                self.config.sliding_window as i64,
            )?,
            None => None,
        };
        for (layer, window) in self.layers.iter().zip(windows.iter_mut()) {
            let input = match &pending {
                Some(output) => output.as_input(),
                None => DraftResidual::Hidden(embedded),
            };
            let next =
                layer.forward(input, window, query_base, mask.as_ref(), self.merged_rows())?;
            pending = Some(next);
        }
        match &pending {
            Some(output) => Ok(output.hidden_and_normalized(&self.norm)?.1),
            None => self.norm.forward(embedded),
        }
    }

    /// Run DFlash2 draft + selector and return the unevaluated device-resident
    /// candidate IDs (`[1, L, K]` Int32) and predecessor-conditioned transition
    /// scores (`[L, K, K]` Float32) without host sync.
    ///
    /// On the first compiled-eligible cycle, performs a one-time eval before
    /// setting `compiled.prepared = true` (mirroring [`Self::propose_with`]).
    pub(crate) fn propose_tree_lattice_arrays(
        &self,
        target_embedding: &Embedding,
        target_lm_head: Option<&LinearProj>,
        context: &mut DFlash2ContextCache,
        anchor: u32,
        max_len: usize,
    ) -> Result<(MxArray, MxArray)> {
        if max_len == 0 {
            return Err(Error::from_reason(
                "DFlash2 propose_tree_lattice_arrays requires max_len > 0",
            ));
        }
        context.observe_tokens(&[anchor]);
        let ids = parallel_query_ids(anchor, self.config.mask_token_id, max_len);
        let block = MxArray::from_int32(&ids, &[1, ids.len() as i64])?;
        let use_spanned_head = crate::accelerator::cuda_ops_available()
            && cuda_spanned_head_enabled()
            && !context.wide_vocab();
        let spanned_heads = if use_spanned_head {
            self.spanned_draft_heads(target_lm_head)?
        } else {
            None
        };
        let compile = max_len == self.config.block_size
            && unsafe { sys::mlx_default_device() } == 1
            && self.execution_policy.compile_draft();
        let compiled_pair = if compile && self.compiled.prepared.load(Ordering::Relaxed) {
            match self.propose_tree_lattice_compiled(
                target_embedding,
                target_lm_head,
                spanned_heads.as_deref(),
                context,
                &block,
            ) {
                Ok(pair) => pair,
                Err(error) => {
                    self.compiled.disabled.store(true, Ordering::Relaxed);
                    return Err(error);
                }
            }
        } else {
            None
        };
        let (candidates, scores) = if let Some(pair) = compiled_pair {
            pair
        } else {
            let query_base = context.logical_len;
            let embedded = self.apply_mask_embedding(target_embedding.forward(&block)?)?;
            let hidden = self.forward_embedded(&embedded, query_base, context)?;
            let rows = max_len as i64 + 1;
            let (draft_hidden, draft_logits, is_spanned) =
                if let Some(spanned) = spanned_heads.as_deref() {
                    let draft_hidden = hidden.slice_axis(1, 1, rows)?;
                    let draft_logits = spanned.forward_logits(&draft_hidden)?;
                    (draft_hidden, draft_logits, true)
                } else {
                    let logits = match target_lm_head {
                        Some(head) => head.forward_logits(&hidden)?,
                        None => target_embedding.as_linear(&hidden)?,
                    };
                    let draft_hidden = hidden.slice_axis(1, 1, rows)?;
                    let draft_logits = logits.slice_axis(1, 1, rows)?;
                    (draft_hidden, draft_logits, false)
                };
            let anchor_ids = MxArray::from_int32(&[anchor as i32], &[1])?;
            self.selector.select_lattice_arrays(
                &draft_hidden,
                &draft_logits,
                &anchor_ids,
                is_spanned,
            )?
        };
        if compile && !self.compiled.prepared.load(Ordering::Relaxed) {
            MxArray::eval_arrays(&[&candidates, &scores])?;
            self.compiled.prepared.store(true, Ordering::Relaxed);
        }
        Ok((candidates, scores))
    }

    /// Run DFlash2 draft + selector over a [`DFlash2TargetView`] (such as
    /// `qwen4_exp`'s external embedding row store + resident quantized
    /// `lm_head`, or `inkling2`'s `embed_norm` + custom Metal `lm_head`) and
    /// return unevaluated `(candidates [1, L, K] Int32, scores [L, K, K] Float32)`.
    ///
    /// Always runs at full `max_len == self.config.block_size` (`7`, `rows = 8`)
    /// so the compiled embedded lattice / body graph never retraces.
    pub(crate) fn propose_tree_lattice_arrays_embedded(
        &self,
        target: &mut dyn DFlash2TargetView,
        context: &mut DFlash2ContextCache,
        anchor: u32,
        max_len: usize,
    ) -> Result<(MxArray, MxArray)> {
        if max_len == 0 {
            return Err(Error::from_reason(
                "DFlash2 propose_tree_lattice_arrays_embedded requires max_len > 0",
            ));
        }
        context.observe_tokens(&[anchor]);
        let ids = parallel_query_ids(anchor, self.config.mask_token_id, max_len);
        let block = MxArray::from_int32(&ids, &[1, ids.len() as i64])?;
        let use_spanned_head = crate::accelerator::cuda_ops_available()
            && cuda_spanned_head_enabled()
            && !context.wide_vocab();
        let spanned_heads = if use_spanned_head {
            if let Some(ql) = target.target_lm_head_qlinear() {
                self.spanned_draft_heads_ql(ql)?
            } else if let Some((_, head)) = target.traced() {
                self.spanned_draft_heads(head)?
            } else {
                None
            }
        } else {
            None
        };
        let gpu_compile = max_len == self.config.block_size
            && unsafe { sys::mlx_default_device() } == 1
            && self.execution_policy.compile_draft();
        let compile_cuda_embedded =
            crate::accelerator::cuda_ops_available() && target.target_lm_head_qlinear().is_some();
        let compile_body = !compile_cuda_embedded && target.traced().is_none() && gpu_compile;
        let compile = (compile_cuda_embedded || compile_body) && gpu_compile;
        let embedded = self.apply_mask_embedding(target.embed(&block, &ids)?)?;
        let compiled_pair = if compile_cuda_embedded
            && gpu_compile
            && self.compiled.prepared.load(Ordering::Relaxed)
        {
            if let Some(ql) = target.target_lm_head_qlinear() {
                match self.propose_tree_lattice_compiled_embedded(
                    &embedded,
                    anchor,
                    ql,
                    spanned_heads.as_deref(),
                    context,
                ) {
                    Ok(pair) => pair,
                    Err(error) => {
                        self.compiled.disabled.store(true, Ordering::Relaxed);
                        return Err(error);
                    }
                }
            } else {
                None
            }
        } else {
            None
        };
        let (candidates, scores) = if let Some(pair) = compiled_pair {
            pair
        } else {
            let query_base = context.logical_len;
            let compiled_hidden = if compile_body && self.compiled.prepared.load(Ordering::Relaxed)
            {
                match self.forward_embedded_compiled(&embedded, context) {
                    Ok(hidden) => hidden,
                    Err(error) => {
                        self.compiled.disabled.store(true, Ordering::Relaxed);
                        return Err(error);
                    }
                }
            } else {
                None
            };
            let hidden = match compiled_hidden {
                Some(h) => h,
                None => self.forward_embedded(&embedded, query_base, context)?,
            };
            let rows = max_len as i64 + 1;
            let draft_hidden = hidden.slice_axis(1, 1, rows)?;
            let (draft_logits, is_spanned) = if let Some(spanned) = spanned_heads.as_deref() {
                (spanned.forward_logits(&draft_hidden)?, true)
            } else if target.project_proposals_only() {
                (target.logits(&draft_hidden)?, false)
            } else {
                // Match `propose_tree_lattice_compiled_embedded`: run the full-vocab
                // head on the whole block (anchor + proposals, M=8) so Metal keeps its
                // K-quant M = 8 NAX route and eager/compiled stay bit-identical.
                (target.logits(&hidden)?.slice_axis(1, 1, rows)?, false)
            };
            let anchor_ids = MxArray::from_int32(&[anchor as i32], &[1])?;
            self.selector.select_lattice_arrays(
                &draft_hidden,
                &draft_logits,
                &anchor_ids,
                is_spanned,
            )?
        };
        if compile && !self.compiled.prepared.load(Ordering::Relaxed) {
            MxArray::eval_arrays(&[&candidates, &scores])?;
            self.compiled.prepared.store(true, Ordering::Relaxed);
        }
        Ok((candidates, scores))
    }

    /// Evaluate and read back device-resident `(candidates, scores)` into a
    /// host [`DraftLattice`].
    pub(crate) fn lattice_from_arrays(
        &self,
        candidates: &MxArray,
        scores: &MxArray,
        max_len: usize,
    ) -> Result<super::verify_tree::DraftLattice> {
        if max_len == 0 {
            return Ok(super::verify_tree::DraftLattice::empty());
        }
        MxArray::eval_arrays(&[candidates, scores])?;
        let cand_i32 = candidates.to_int32()?;
        let scores_f32 = scores.to_float32()?;
        let top_k = self.selector.top_k;
        let expected_cands = max_len.saturating_mul(top_k);
        let expected_scores = expected_cands.saturating_mul(top_k);
        if cand_i32.len() != expected_cands || scores_f32.len() != expected_scores {
            return Err(Error::from_reason(format!(
                "DFlash2 tree lattice shape mismatch: got {} candidates / {} scores, expected {expected_cands} / {expected_scores}",
                cand_i32.len(),
                scores_f32.len()
            )));
        }
        let cand_u32 = cand_i32
            .iter()
            .map(|&id| {
                u32::try_from(id).map_err(|_| {
                    Error::from_reason(format!(
                        "DFlash2 tree lattice produced negative candidate id {id}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(super::verify_tree::DraftLattice::from_selector_scores(
            &cand_u32,
            &scores_f32,
            max_len,
            top_k,
        ))
    }

    /// Run DFlash2 draft + selector and materialize the full `[L, K, K]`
    /// conditional candidate lattice (`~7.6 KB` readback for `L=7, K=16`) for
    /// speculative verification tree construction.
    #[allow(dead_code)]
    pub(crate) fn propose_tree_lattice(
        &self,
        target_embedding: &Embedding,
        target_lm_head: Option<&LinearProj>,
        context: &mut DFlash2ContextCache,
        anchor: u32,
        max_len: usize,
    ) -> Result<super::verify_tree::DraftLattice> {
        if max_len == 0 {
            return Ok(super::verify_tree::DraftLattice::empty());
        }
        let (candidates, scores) = self.propose_tree_lattice_arrays(
            target_embedding,
            target_lm_head,
            context,
            anchor,
            max_len,
        )?;
        self.lattice_from_arrays(&candidates, &scores, max_len)
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
        confidence: Option<&mut Option<MxArray>>,
        rng: &mut R,
    ) -> Result<(SelectorPath, Vec<SparseDistribution>)> {
        self.propose_with(
            &mut DenseTargetView {
                embedding: target_embedding,
                lm_head: target_lm_head,
            },
            context,
            anchor,
            max_len,
            temperature,
            device_path,
            confidence,
            rng,
        )
    }

    /// [`Self::propose`] over any target's token table and head.
    pub(crate) fn propose_with<R: Rng + ?Sized>(
        &self,
        target: &mut dyn DFlash2TargetView,
        context: &mut DFlash2ContextCache,
        anchor: u32,
        max_len: usize,
        temperature: f64,
        device_path: bool,
        confidence: Option<&mut Option<MxArray>>,
        rng: &mut R,
    ) -> Result<(SelectorPath, Vec<SparseDistribution>)> {
        // Observe the current anchor (prefill first token or previous boundary
        // token) before deciding narrow vs full draft head.
        context.observe_tokens(&[anchor]);
        let ids = parallel_query_ids(anchor, self.config.mask_token_id, max_len);
        let phase_time = std::env::var("MLX_DFLASH2_PHASE_TIME").is_ok();
        let t0 = std::time::Instant::now();
        let block = MxArray::from_int32(&ids, &[1, ids.len() as i64])?;
        let head_len = target
            .proposal_head_limit()
            .map_or(max_len, |limit| limit.clamp(1, max_len));
        let use_spanned_head = crate::accelerator::cuda_ops_available()
            && cuda_spanned_head_enabled()
            && !context.wide_vocab();
        let has_traced = target.traced().is_some();
        let has_target_ql = !has_traced && target.target_lm_head_qlinear().is_some();
        let spanned_heads = if use_spanned_head {
            if let Some((_, head)) = target.traced() {
                self.spanned_draft_heads(head)?
            } else if let Some(ql) = target.target_lm_head_qlinear() {
                self.spanned_draft_heads_ql(ql)?
            } else {
                None
            }
        } else {
            None
        };
        let gpu_compile = max_len == self.config.block_size
            && unsafe { sys::mlx_default_device() } == 1
            && self.execution_policy.compile_draft();
        let compile_embedded = has_target_ql && crate::accelerator::cuda_ops_available();
        let compile_full = device_path
            && (has_traced || compile_embedded)
            && is_greedy_temperature(temperature)
            && gpu_compile;
        let compile_body = !has_traced && !compile_embedded && gpu_compile;
        let compile = compile_full || compile_body;
        // Only full proposals compile: a budget-limited tail would otherwise
        // pay for a new graph used once. The first full proposal runs eager
        // and is explicitly evaluated below before prepared becomes true, so
        // even a failed verification leaves materialized captured weights.
        let mut pre_embedded: Option<MxArray> = None;
        if compile_full && self.compiled.prepared.load(Ordering::Relaxed) {
            if let Some((target_embedding, target_lm_head)) = target.traced() {
                match self.propose_compiled(
                    target_embedding,
                    target_lm_head,
                    spanned_heads.as_deref(),
                    context,
                    &block,
                    confidence.is_some(),
                ) {
                    Ok(Some((path, probabilities))) => {
                        if let Some(out) = confidence {
                            *out = probabilities;
                        }
                        return Ok((SelectorPath::Device(path), Vec::new()));
                    }
                    Ok(None) => {}
                    Err(error) => {
                        self.compiled.disabled.store(true, Ordering::Relaxed);
                        return Err(error);
                    }
                }
            } else if compile_embedded {
                let embedded = self.apply_mask_embedding(target.embed(&block, &ids)?)?;
                let compiled_res = if let Some(ql) = target.target_lm_head_qlinear() {
                    self.propose_compiled_embedded(
                        &embedded,
                        anchor,
                        ql,
                        spanned_heads.as_deref(),
                        context,
                        head_len as i64,
                        confidence.is_some(),
                    )
                } else {
                    Ok(None)
                };
                match compiled_res {
                    Ok(Some((path, probabilities))) => {
                        if let Some(out) = confidence {
                            *out = probabilities;
                        }
                        return Ok((SelectorPath::Device(path), Vec::new()));
                    }
                    Ok(None) => {
                        pre_embedded = Some(embedded);
                    }
                    Err(error) => {
                        self.compiled.disabled.store(true, Ordering::Relaxed);
                        return Err(error);
                    }
                }
            }
        }
        let query_base = context.logical_len;
        let embedded = match pre_embedded {
            Some(e) => e,
            None => self.apply_mask_embedding(target.embed(&block, &ids)?)?,
        };
        let compiled_hidden = if compile_body && self.compiled.prepared.load(Ordering::Relaxed) {
            match self.forward_embedded_compiled(&embedded, context) {
                Ok(h) => h,
                Err(error) => {
                    self.compiled.disabled.store(true, Ordering::Relaxed);
                    return Err(error);
                }
            }
        } else {
            None
        };
        let hidden = match compiled_hidden {
            Some(h) => h,
            None => self.forward_embedded(&embedded, query_base, context)?,
        };
        let rows = head_len as i64 + 1;
        let (hidden, logits, is_spanned) = if let Some(spanned) = spanned_heads.as_deref() {
            let draft_hidden = hidden.slice_axis(1, 1, rows)?;
            let draft_logits = spanned.forward_logits(&draft_hidden)?;
            (draft_hidden, draft_logits, true)
        } else if target.project_proposals_only()
            || (head_len < max_len && crate::accelerator::cuda_ops_available())
        {
            let draft_hidden = hidden.slice_axis(1, 1, rows)?;
            let draft_logits = target.logits(&draft_hidden)?;
            (draft_hidden, draft_logits, false)
        } else {
            // The head runs on the whole block (anchor + proposals, 8 rows), the
            // height its K-quant M = 8 route serves, and the anchor row's logits
            // are dropped: at 7 rows the head took the per-row kernel and cost
            // more than at 8.
            let logits = target.logits(&hidden)?;
            let draft_hidden = hidden.slice_axis(1, 1, rows)?;
            let draft_logits = logits.slice_axis(1, 1, rows)?;
            (draft_hidden, draft_logits, false)
        };
        let out = self.selector.select_spanned(
            &hidden,
            &logits,
            anchor,
            temperature,
            device_path,
            is_spanned,
            confidence,
            rng,
        )?;
        if compile && !self.compiled.prepared.load(Ordering::Relaxed) {
            // This one-time boundary also makes direct Rust callers safe when
            // they propose again before consuming the previous path.
            match &out.0 {
                SelectorPath::Device(path) => {
                    MxArray::eval_arrays(&[path])?;
                    self.compiled.prepared.store(true, Ordering::Relaxed);
                }
                SelectorPath::Sampled { ids, .. } => {
                    MxArray::eval_arrays(&[ids])?;
                    self.compiled.prepared.store(true, Ordering::Relaxed);
                }
                SelectorPath::Host(_) if compile_body => {
                    self.compiled.prepared.store(true, Ordering::Relaxed);
                }
                SelectorPath::Host(_) => {}
            }
        }
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
