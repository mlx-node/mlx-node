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

/// Full target contract validated against a loaded DFlash2 companion: core
/// geometry plus optional unpadded valid vocabulary size, post-correction
/// logit softcapping, and whether `layer == num_layers` (final-norm tap) is
/// supported by the target (e.g. Inkling-2-Mini).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DFlash2TargetContract {
    pub(crate) hidden_size: usize,
    pub(crate) vocab_size: usize,
    pub(crate) num_layers: usize,
    pub(crate) valid_vocab_size: Option<usize>,
    pub(crate) final_logit_softcapping: Option<f32>,
    pub(crate) allow_final_norm_tap: bool,
}

impl From<DFlash2TargetShape> for DFlash2TargetContract {
    fn from(shape: DFlash2TargetShape) -> Self {
        Self {
            hidden_size: shape.hidden_size,
            vocab_size: shape.vocab_size,
            num_layers: shape.num_layers,
            valid_vocab_size: (shape.valid_vocab_size != shape.vocab_size)
                .then_some(shape.valid_vocab_size),
            final_logit_softcapping: shape.final_logit_softcapping,
            allow_final_norm_tap: shape.reads_layer_inputs,
        }
    }
}

impl From<DFlash2TargetContract> for DFlash2TargetShape {
    fn from(contract: DFlash2TargetContract) -> Self {
        Self {
            hidden_size: contract.hidden_size,
            vocab_size: contract.vocab_size,
            valid_vocab_size: contract.valid_vocab_size.unwrap_or(contract.vocab_size),
            num_layers: contract.num_layers,
            final_logit_softcapping: contract.final_logit_softcapping,
            reads_layer_inputs: contract.allow_final_norm_tap,
        }
    }
}

impl From<&super::config::Qwen3_5Config> for DFlash2TargetContract {
    fn from(target: &super::config::Qwen3_5Config) -> Self {
        DFlash2TargetShape::from(target).into()
    }
}

/// Borrowed resident target `lm_head` passed into compiled embedded-input draft
/// graphs, supporting both `LinearProj` (`Standard` BF16 or `Quantized`) and
/// legacy `&QuantizedLinear` views (`qwen4_exp`).
#[derive(Clone, Copy)]
pub(super) enum EmbeddedLmHead<'a> {
    Proj(&'a LinearProj),
    Quantized(&'a QuantizedLinear),
}

impl<'a> EmbeddedLmHead<'a> {
    fn from_target(target: &'a dyn DFlash2TargetView) -> Option<Self> {
        if let Some(proj) = target.target_lm_head_proj() {
            Some(Self::Proj(proj))
        } else {
            target.target_lm_head_qlinear().map(Self::Quantized)
        }
    }

    fn as_quantized(self) -> Option<&'a QuantizedLinear> {
        match self {
            Self::Proj(proj) => proj.as_quantized(),
            Self::Quantized(ql) => Some(ql),
        }
    }

    pub(super) fn forward_logits(self, hidden: &MxArray) -> Result<MxArray> {
        match self {
            Self::Proj(proj) => proj.forward_logits(hidden),
            Self::Quantized(ql) => ql.forward(hidden),
        }
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
    /// keeps every proposal eager unless [`Self::target_lm_head_proj`] or
    /// [`Self::target_lm_head_qlinear`] provides a resident head for
    /// embedded-input compilation or the compiled draft body runs on
    /// pre-embedded rows.
    fn traced(&self) -> Option<(&Embedding, Option<&LinearProj>)>;
    /// Resident `lm_head` projection (`Standard` BF16 or `Quantized`) when
    /// [`Self::traced`] is `None`, enabling compiled embedded-input draft
    /// graphs for both GGUF quantized and SafeTensors dense targets.
    fn target_lm_head_proj(&self) -> Option<&LinearProj> {
        None
    }
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

/// One lane in a batched multi-lane [`DFlash2Model::propose_with_batched`] wave.
pub(crate) struct DFlash2BatchProposalLane<'a> {
    pub(crate) context: &'a mut DFlash2ContextCache,
    pub(crate) anchor: u32,
    pub(crate) temperature: f64,
    pub(crate) device_path: bool,
    pub(crate) want_confidence: bool,
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
    /// Optional `[hidden_size]` BF16 mask-token embedding shipped by a DFlash2
    /// checkpoint (`mask_embedding` in `model.safetensors` or sidecar
    /// `mask_embedding.safetensors`), spliced onto rows `1..rows` of each block.
    mask_embedding: Option<MxArray>,
    /// Every layer's `k|v` context projection row-merged (`k0|v0|k1|v1|…`):
    /// the accepted rows' draft K/V for all layers in one quantized matmul.
    /// `None` when the packed formats do not merge (per-layer projections).
    context_kv: Option<LinearProj>,
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
    /// target's unpadded active vocabulary (`200058` of `201024`) and softcap
    /// onto the selector and enables Metal compiled draft body + NAX attention
    /// (or compiled CUDA execution on CUDA hosts) when supported.
    pub(crate) fn configure_inkling2_execution(&mut self, target: DFlash2TargetShape) {
        let valid_vocab = target.valid_vocab_size.min(self.selector.vocab_size);
        if valid_vocab < self.config.vocab_size && self.config.valid_vocab_size.is_none() {
            self.config.valid_vocab_size = Some(valid_vocab);
        }
        self.selector.valid_vocab_size = Some(valid_vocab);
        if let Some(cap) = target.final_logit_softcapping.filter(|&c| c > 0.0) {
            if self.config.final_logit_softcapping.is_none() {
                self.config.final_logit_softcapping = Some(cap);
            }
            if self.config.selector_softcap_mode.is_none() {
                self.config.selector_softcap_mode = Some("after_correction".to_string());
            }
            self.selector.post_softcap = self.config.post_edge_softcap();
        }
        let supported = self.config.supports_inkling2_execution_defaults();
        let metal = unsafe { sys::mlx_metal_is_available() };
        let cuda = crate::accelerator::cuda_ops_available();
        let nax = supported && metal && unsafe { sys::mlx_metal_is_nax_available() };
        self.invalidate_compiled();
        self.execution_policy = self
            .execution_policy
            .with_inkling2_metal_profile(supported && metal, nax)
            .with_dense_cuda_profile(cuda);
        for layer in &mut self.layers {
            layer.attention.tensor_attention_enabled = self.execution_policy.tensor_attention();
        }
    }

    pub(super) fn apply_mask_embedding(&self, embedded: MxArray) -> Result<MxArray> {
        if self.mask_embedding.is_none() {
            return Ok(embedded);
        }
        self.splice_mask_embedding(&embedded)
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
        self.config.validate_target(target)
    }

    /// Validate `target` and apply target-supplied contract parameters (such as
    /// `valid_vocab_size` or `final_logit_softcapping` when a checkpoint's
    /// `config.json` omits them, e.g. `incoai/Inkling-2-Mini-DFlash2`).
    #[allow(dead_code)]
    pub(crate) fn configure_target_contract(
        &mut self,
        target: impl Into<DFlash2TargetShape>,
    ) -> Result<()> {
        let shape = target.into();
        self.config.validate_target(shape)?;
        self.invalidate_compiled();
        let valid_vocab = shape.valid_vocab_size.min(self.selector.vocab_size);
        if valid_vocab < self.config.vocab_size || self.config.valid_vocab_size.is_some() {
            self.config.valid_vocab_size = Some(valid_vocab);
            self.selector.valid_vocab_size = Some(valid_vocab);
        }
        if let Some(cap) = shape.final_logit_softcapping.filter(|&c| c > 0.0) {
            if self.config.final_logit_softcapping.is_none() {
                self.config.final_logit_softcapping = Some(cap);
            }
            if self.config.selector_softcap_mode.is_none() {
                self.config.selector_softcap_mode = Some("after_correction".to_string());
            }
            self.selector.post_softcap = self.config.post_edge_softcap();
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn mask_embedding(&self) -> Option<&MxArray> {
        self.mask_embedding.as_ref()
    }

    /// Build `[B, 1 + mask_rows, hidden]` block embeddings from a `[B, 1, hidden]`
    /// anchor embedding and this draft's optional `mask_embedding`. Returns
    /// `Ok(None)` when the checkpoint does not ship a dedicated `mask_embedding`.
    pub(crate) fn build_block_with_mask_embedding(
        &self,
        anchor_embedded: &MxArray,
        mask_rows: i64,
    ) -> Result<Option<MxArray>> {
        let Some(mask) = self.mask_embedding.as_ref() else {
            return Ok(None);
        };
        if mask_rows <= 0 {
            return Ok(Some(anchor_embedded.clone()));
        }
        let batch = anchor_embedded.shape_at(0)?;
        let hidden = anchor_embedded.shape_at(2)?;
        let dtype = anchor_embedded.dtype()?;
        let mask = if mask.dtype()? == dtype {
            mask.clone()
        } else {
            mask.astype(dtype)?
        };
        let repeated = mask
            .reshape(&[1, 1, hidden])?
            .broadcast_to(&[batch, mask_rows, hidden])?;
        Ok(Some(MxArray::concatenate(anchor_embedded, &repeated, 1)?))
    }

    /// If this checkpoint ships a dedicated `mask_embedding`, replace rows
    /// `1..rows` of `[B, rows, hidden]` block embeddings with it; otherwise
    /// return `embedded` unchanged.
    pub(crate) fn splice_mask_embedding(&self, embedded: &MxArray) -> Result<MxArray> {
        if self.mask_embedding.is_none() {
            return Ok(embedded.clone());
        }
        let rows = embedded.shape_at(1)?;
        if rows <= 1 {
            return Ok(embedded.clone());
        }
        let anchor_row = embedded.slice_axis(1, 0, 1)?;
        Ok(self
            .build_block_with_mask_embedding(&anchor_row, rows - 1)?
            .unwrap_or_else(|| embedded.clone()))
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

    /// Batched context K/V projection across `N` lanes whose kept fused context
    /// rows are concatenated along axis 1 into `fused_all` (`[1, total_keep, H]`).
    ///
    /// Runs ONE merged `context_kv` matmul across all layers when available and
    /// `total_keep <= 32` (Metal 4 NAX decode tier), else per-layer `k_proj` /
    /// `v_proj` over `[1, total_keep, H]`, applies `k_norm` once per layer, then
    /// slices each lane's `(base_i, keep_i)` segment and applies RoPE at `base_i`.
    pub(super) fn project_context_batched(
        &self,
        fused_all: &MxArray,
        segments: &[(i32, usize)],
    ) -> Result<Vec<Vec<(MxArray, MxArray)>>> {
        let num_lanes = segments.len();
        let num_layers = self.layers.len();
        if num_lanes == 0 {
            return Ok(Vec::new());
        }
        if num_layers == 0 {
            return Ok(vec![Vec::new(); num_lanes]);
        }
        let total_keep = fused_all.shape_at(1)?;
        let expected_total: i64 = segments.iter().map(|&(_, keep)| keep as i64).sum();
        if fused_all.shape_at(0)? != 1 || total_keep != expected_total {
            return Err(Error::from_reason(format!(
                "DFlash2 project_context_batched shape {:?} does not match [1, {expected_total}, _]",
                fused_all.shape()?.as_ref()
            )));
        }

        let mut normed_keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        let use_merged_kv = self.context_kv.is_some() && total_keep <= 32;
        if use_merged_kv && let Some(context_kv) = self.context_kv.as_ref() {
            let first = &self.layers[0];
            let kv_width = first.attention.num_kv_heads * first.attention.head_dim;
            let bounds = (1..2 * num_layers as i64)
                .map(|i| i * kv_width)
                .collect::<Vec<_>>();
            let parts = context_kv.forward(fused_all)?.split_sections(&bounds, -1)?;
            for (layer, pair) in self.layers.iter().zip(parts.as_chunks::<2>().0) {
                let attention = &layer.attention;
                let k_normed = attention
                    .k_norm
                    .forward(&pair[0].reshape(&[
                        1,
                        total_keep,
                        attention.num_kv_heads,
                        attention.head_dim,
                    ])?)?
                    .transpose(Some(&[0, 2, 1, 3]))?;
                normed_keys.push(k_normed);
                values.push(attention.finish_values(&pair[1])?);
            }
        } else {
            for layer in &self.layers {
                let attention = &layer.attention;
                let k_raw = attention.k_proj.forward(fused_all)?;
                let k_normed = attention
                    .k_norm
                    .forward(&k_raw.reshape(&[
                        1,
                        total_keep,
                        attention.num_kv_heads,
                        attention.head_dim,
                    ])?)?
                    .transpose(Some(&[0, 2, 1, 3]))?;
                let v_raw = attention.v_proj.forward(fused_all)?;
                normed_keys.push(k_normed);
                values.push(attention.finish_values(&v_raw)?);
            }
        }

        let mut per_lane = Vec::with_capacity(num_lanes);
        let mut off = 0i64;
        for &(base, keep) in segments {
            let end = off + keep as i64;
            let lane_kv = if use_merged_kv && num_layers > 1 {
                let first = &self.layers[0].attention;
                let mut lane_keys = Vec::with_capacity(num_layers);
                let mut lane_vals = Vec::with_capacity(num_layers);
                for layer_idx in 0..num_layers {
                    lane_keys.push(normed_keys[layer_idx].slice_axis(2, off, end)?);
                    lane_vals.push(values[layer_idx].slice_axis(2, off, end)?);
                }
                let roped = first.rope.forward(
                    &MxArray::concatenate_many(lane_keys.iter().collect(), Some(1))?,
                    Some(base),
                )?;
                let head_bounds = (1..num_layers as i64)
                    .map(|i| i * first.num_kv_heads)
                    .collect::<Vec<_>>();
                roped
                    .split_sections(&head_bounds, 1)?
                    .into_iter()
                    .zip(lane_vals)
                    .collect()
            } else {
                let mut lane_kv = Vec::with_capacity(num_layers);
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let k_i = normed_keys[layer_idx].slice_axis(2, off, end)?;
                    let v_i = values[layer_idx].slice_axis(2, off, end)?;
                    let k_roped = layer.attention.rope.forward(&k_i, Some(base))?;
                    lane_kv.push((k_roped, v_i));
                }
                lane_kv
            };
            per_lane.push(lane_kv);
            off = end;
        }
        Ok(per_lane)
    }

    #[cfg(test)]
    fn forward_hidden(
        &self,
        target_embedding: &Embedding,
        block_ids: &MxArray,
        query_base: i32,
        context: &mut DFlash2ContextCache,
    ) -> Result<MxArray> {
        let embedded = self.splice_mask_embedding(&target_embedding.forward(block_ids)?)?;
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

    /// Batched multi-lane draft transformer and final norm over `[N, rows, hidden]`
    /// embedded blocks with independent per-lane [`DFlash2ContextCache`]s.
    fn forward_embedded_batched(
        &self,
        embedded: &MxArray,
        contexts: &mut [&mut DFlash2ContextCache],
    ) -> Result<MxArray> {
        let n = contexts.len();
        if embedded.shape_at(0)? as usize != n {
            return Err(Error::from_reason(format!(
                "DFlash2 forward_embedded_batched batch {} != contexts len {n}",
                embedded.shape_at(0)?
            )));
        }
        let num_layers = self.layers.len();
        let seq_len = embedded.shape_at(1)?;
        let mut query_bases = Vec::with_capacity(n);
        let mut masks = Vec::with_capacity(n);
        for ctx in contexts.iter_mut() {
            if ctx.layers.len() != num_layers {
                return Err(Error::from_reason(format!(
                    "DFlash2 context layer count {} != model layers {num_layers}",
                    ctx.layers.len()
                )));
            }
            let query_base = ctx.logical_len;
            query_bases.push(query_base);
            let DFlash2ContextCache {
                layers: windows,
                mask_memo,
                ..
            } = &mut **ctx;
            let mask = match windows.first() {
                Some(window) => window.block_mask(
                    mask_memo,
                    query_base,
                    seq_len,
                    self.config.sliding_window as i64,
                )?,
                None => None,
            };
            masks.push(mask);
        }
        let offsets = MxArray::from_int32(&query_bases, &[n as i64])?;
        let mut pending: Option<DraftLayerOutput> = None;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut layer_windows = contexts
                .iter_mut()
                .map(|ctx| &mut ctx.layers[layer_idx])
                .collect::<Vec<_>>();
            let input = match &pending {
                Some(output) => output.as_input(),
                None => DraftResidual::Hidden(embedded),
            };
            let next = layer.forward_batched(
                input,
                &mut layer_windows,
                &offsets,
                &masks,
                self.merged_rows(),
            )?;
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
        context.observe_token(anchor);
        let ids = parallel_query_ids(anchor, self.config.mask_token_id, max_len);
        let block = MxArray::from_int32(&ids, &[1, ids.len() as i64])?;
        let draft_head = self.resolve_draft_head(
            target_lm_head.map(EmbeddedLmHead::Proj),
            context.wide_vocab(),
        )?;
        let compile = max_len == self.config.block_size
            && unsafe { sys::mlx_default_device() } == 1
            && self.execution_policy.compile_draft();
        let compiled_pair = if compile && self.compiled.prepared.load(Ordering::Relaxed) {
            match self.propose_tree_lattice_compiled(
                target_embedding,
                target_lm_head,
                draft_head.as_ref(),
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
            let draft_hidden = hidden.slice_axis(1, 1, rows)?;
            let (draft_logits, spanned_layout) = if let Some(head) = draft_head.as_ref() {
                (head.forward_logits(&draft_hidden)?, head.spanned_layout())
            } else if crate::accelerator::cuda_ops_available() {
                let draft_logits = match target_lm_head {
                    Some(head) => head.forward_logits(&draft_hidden)?,
                    None => target_embedding.as_linear(&draft_hidden)?,
                };
                (draft_logits, None)
            } else {
                let logits = match target_lm_head {
                    Some(head) => head.forward_logits(&hidden)?,
                    None => target_embedding.as_linear(&hidden)?,
                };
                (logits.slice_axis(1, 1, rows)?, None)
            };
            let anchor_ids = MxArray::from_int32(&[anchor as i32], &[1])?;
            self.selector.select_lattice_arrays(
                &draft_hidden,
                &draft_logits,
                &anchor_ids,
                spanned_layout,
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
        context.observe_token(anchor);
        let ids = parallel_query_ids(anchor, self.config.mask_token_id, max_len);
        let block = MxArray::from_int32(&ids, &[1, ids.len() as i64])?;
        let embedded_head = EmbeddedLmHead::from_target(target);
        let has_embedded_head = embedded_head.is_some();
        let draft_head_source = embedded_head.or_else(|| {
            target
                .traced()
                .and_then(|(_, head)| head.map(EmbeddedLmHead::Proj))
        });
        let draft_head = self.resolve_draft_head(draft_head_source, context.wide_vocab())?;
        let gpu_compile = max_len == self.config.block_size
            && unsafe { sys::mlx_default_device() } == 1
            && self.execution_policy.compile_draft();
        let compile_cuda_embedded = crate::accelerator::cuda_ops_available() && has_embedded_head;
        let compile_body = !compile_cuda_embedded && target.traced().is_none() && gpu_compile;
        let compile = (compile_cuda_embedded || compile_body) && gpu_compile;
        let embedded = self.apply_mask_embedding(target.embed(&block, &ids)?)?;
        let compiled_pair = if compile_cuda_embedded
            && gpu_compile
            && self.compiled.prepared.load(Ordering::Relaxed)
        {
            if let Some(head) = EmbeddedLmHead::from_target(target) {
                match self.propose_tree_lattice_compiled_embedded(
                    &embedded,
                    anchor,
                    head,
                    draft_head.as_ref(),
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
            let (draft_logits, spanned_layout) = if let Some(head) = draft_head.as_ref() {
                (head.forward_logits(&draft_hidden)?, head.spanned_layout())
            } else if target.project_proposals_only() || crate::accelerator::cuda_ops_available() {
                // Project only the `block_size` proposal rows (skipping unused anchor
                // row 0) when the target requests proposal-only logits (e.g. Metal
                // Inkling-2-Mini) or on CUDA (matching `propose_tree_lattice_compiled_embedded`
                // and hitting M=7 directly without computing or slicing row 0).
                (target.logits(&draft_hidden)?, None)
            } else {
                // Match `propose_tree_lattice_compiled_embedded`: run the full-vocab
                // head on the whole block (anchor + proposals, M=8) so Metal keeps its
                // K-quant M = 8 NAX route and eager/compiled stay bit-identical.
                (target.logits(&hidden)?.slice_axis(1, 1, rows)?, None)
            };
            let anchor_ids = MxArray::from_int32(&[anchor as i32], &[1])?;
            self.selector.select_lattice_arrays(
                &draft_hidden,
                &draft_logits,
                &anchor_ids,
                spanned_layout,
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
        context.observe_token(anchor);
        context.clear_pre_propose_handles();
        let ids = parallel_query_ids(anchor, self.config.mask_token_id, max_len);
        let phase_time = std::env::var("MLX_DFLASH2_PHASE_TIME").is_ok();
        let t0 = std::time::Instant::now();
        let block = MxArray::from_int32(&ids, &[1, ids.len() as i64])?;
        let head_len = target
            .proposal_head_limit()
            .map_or(max_len, |limit| limit.clamp(1, max_len));
        let has_traced = target.traced().is_some();
        let has_embedded_head = !has_traced && EmbeddedLmHead::from_target(target).is_some();
        let draft_head_source = target
            .traced()
            .and_then(|(_, head)| head.map(EmbeddedLmHead::Proj))
            .or_else(|| EmbeddedLmHead::from_target(target));
        let draft_head = self.resolve_draft_head(draft_head_source, context.wide_vocab())?;
        let gpu_compile = max_len == self.config.block_size
            && unsafe { sys::mlx_default_device() } == 1
            && self.execution_policy.compile_draft();
        let compile_embedded = has_embedded_head && crate::accelerator::cuda_ops_available();
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
                    draft_head.as_ref(),
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
                let compiled_res = if let Some(head) = EmbeddedLmHead::from_target(target) {
                    self.propose_compiled_embedded(
                        &embedded,
                        anchor,
                        head,
                        draft_head.as_ref(),
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
        let draft_hidden = hidden.slice_axis(1, 1, rows)?;
        let (hidden, logits, spanned_layout) = if let Some(head) = draft_head.as_ref() {
            let draft_logits = head.forward_logits(&draft_hidden)?;
            (draft_hidden, draft_logits, head.spanned_layout())
        } else if target.project_proposals_only() || crate::accelerator::cuda_ops_available() {
            // Project only the proposal rows (skipping unused anchor row 0) when
            // the target requests proposal-only logits or on CUDA (keeping eager
            // and compiled bit-identical without projecting row 0).
            let draft_logits = target.logits(&draft_hidden)?;
            (draft_hidden, draft_logits, None)
        } else {
            // The head runs on the whole block (anchor + proposals, 8 rows), the
            // height its K-quant M = 8 route serves, and the anchor row's logits
            // are dropped: at 7 rows the head took the per-row kernel and cost
            // more than at 8.
            let logits = target.logits(&hidden)?;
            let draft_logits = logits.slice_axis(1, 1, rows)?;
            (draft_hidden, draft_logits, None)
        };
        let out = self.selector.select_spanned(
            &hidden,
            &logits,
            anchor,
            temperature,
            device_path,
            spanned_layout,
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

    /// Validate that `cache` can accept `keep` new context rows for this model
    /// before entering a multi-lane batched append (so one malformed lane cannot
    /// reject healthy peers during `DFlash2ContextCache::append_batched`'s pre-check).
    pub(crate) fn validate_context_for_append(
        &self,
        cache: &DFlash2ContextCache,
        keep: usize,
    ) -> Result<()> {
        if keep == 0 {
            return Err(Error::from_reason(
                "DFlash2ContextCache::append_batched requires non-empty token_ids per lane",
            ));
        }
        if usize::try_from(cache.logical_len).ok() != Some(cache.token_history.len()) {
            return Err(Error::from_reason(format!(
                "DFlash2 context provenance length {} does not match logical length {}",
                cache.token_history.len(),
                cache.logical_len
            )));
        }
        if cache.layers.len() != self.layers.len() {
            return Err(Error::from_reason(format!(
                "DFlash2 context layer count {} != model layers {}",
                cache.layers.len(),
                self.layers.len()
            )));
        }
        let keep_i32 = i32::try_from(keep).map_err(|_| {
            Error::from_reason(format!(
                "DFlash2 context append row count {keep} exceeds i32"
            ))
        })?;
        cache
            .logical_len
            .checked_add(keep_i32)
            .ok_or_else(|| Error::from_reason("DFlash2 context length overflow"))?;
        Ok(())
    }

    /// Snapshot each layer's ring `end` cursor so a batched-append error handler
    /// can distinguish untouched lanes from a partially mutated failing lane.
    pub(crate) fn context_window_ends(cache: &DFlash2ContextCache) -> Vec<i64> {
        cache.layers.iter().map(|window| window.end).collect()
    }

    /// Test-only fault injector: replaces `cache.layers[layer].keys` with a 1×1×1×1
    /// buffer so the next `append` to that layer fails deterministically inside
    /// `DraftKvWindow::store_rows` after earlier layers have already mutated.
    #[cfg(test)]
    pub(crate) fn corrupt_window_buffer_for_test(cache: &mut DFlash2ContextCache, layer: usize) {
        let idx = layer.min(cache.layers.len().saturating_sub(1));
        if let Some(window) = cache.layers.get_mut(idx) {
            window.keys = Some(MxArray::zeros(&[1, 1, 1, 1], Some(DType::BFloat16)).unwrap());
        }
    }

    /// Batched multi-lane proposal over a [`DFlash2TargetView`]: runs ONE
    /// `[N, block_len + 1]` embed, ONE batched 5-layer draft pass (with exact
    /// per-lane KV windows & RoPE offsets), ONE `[N, head_len, valid_vocab]`
    /// `lm_head` matmul, and per-lane candidate selection returning one `Result`
    /// per lane so a fault on one lane never aborts healthy peers.
    ///
    /// Keeps `N == 1` on [`Self::propose_with`] (preserving the compiled solo
    /// graph) and falls back per-lane on CUDA `wide_vocab` mismatch or if the
    /// shared batched forward stage errors.
    pub(crate) fn propose_with_batched(
        &self,
        target: &mut dyn DFlash2TargetView,
        lanes: &mut [DFlash2BatchProposalLane<'_>],
        rngs: &mut [&mut dyn Rng],
        block_len: usize,
    ) -> Vec<Result<(SelectorPath, Vec<SparseDistribution>, Option<MxArray>)>> {
        if lanes.len() != rngs.len() {
            let reason = format!(
                "DFlash2 propose_with_batched lanes ({}) != rngs ({})",
                lanes.len(),
                rngs.len()
            );
            return (0..lanes.len())
                .map(|_| Err(Error::from_reason(reason.clone())))
                .collect();
        }
        if block_len == 0 {
            return (0..lanes.len())
                .map(|_| {
                    Err(Error::from_reason(
                        "DFlash2 propose_with_batched requires block_len > 0",
                    ))
                })
                .collect();
        }
        if lanes.is_empty() {
            return Vec::new();
        }
        let run_per_lane = |target: &mut dyn DFlash2TargetView,
                            lanes: &mut [DFlash2BatchProposalLane<'_>],
                            rngs: &mut [&mut dyn Rng]| {
            lanes
                .iter_mut()
                .zip(rngs.iter_mut())
                .map(|(lane, rng)| {
                    let mut conf = None;
                    let (path, sparse) = self.propose_with(
                        target,
                        lane.context,
                        lane.anchor,
                        block_len,
                        lane.temperature,
                        lane.device_path,
                        lane.want_confidence.then_some(&mut conf),
                        *rng,
                    )?;
                    Ok((path, sparse, conf))
                })
                .collect::<Vec<_>>()
        };
        if lanes.len() == 1 {
            return run_per_lane(target, lanes, rngs);
        }

        for lane in lanes.iter_mut() {
            lane.context.observe_token(lane.anchor);
        }

        // On CUDA, lanes with mismatched `wide_vocab` select different spanned vs
        // full draft heads, so run each lane independently through `propose_with`.
        if crate::accelerator::cuda_ops_available()
            && lanes
                .windows(2)
                .any(|w| w[0].context.wide_vocab() != w[1].context.wide_vocab())
        {
            return run_per_lane(target, lanes, rngs);
        }

        // Snapshot committed ring handles before `forward_embedded_batched` mutates
        // `DraftKvWindow` via `with_block` so a failed build or deferred eval can
        // restore clean committed buffers before retrying solo.
        for lane in lanes.iter_mut() {
            lane.context.save_pre_propose_handles();
        }

        let shared =
            (|| -> Result<(MxArray, MxArray, Option<spanned_head::SpannedVocabLayout>)> {
                let head_len = target
                    .proposal_head_limit()
                    .map_or(block_len, |limit| limit.clamp(1, block_len));
                let draft_head_source = target
                    .traced()
                    .and_then(|(_, head)| head.map(EmbeddedLmHead::Proj))
                    .or_else(|| EmbeddedLmHead::from_target(target));
                let draft_head =
                    self.resolve_draft_head(draft_head_source, lanes[0].context.wide_vocab())?;

                let n = lanes.len();
                let seq_len = block_len + 1;
                let mut all_ids = Vec::with_capacity(n * seq_len);
                for lane in lanes.iter() {
                    all_ids.extend_from_slice(&parallel_query_ids(
                        lane.anchor,
                        self.config.mask_token_id,
                        block_len,
                    ));
                }
                let stacked_block = MxArray::from_int32(&all_ids, &[n as i64, seq_len as i64])?;
                let embedded =
                    self.apply_mask_embedding(target.embed(&stacked_block, &all_ids)?)?;

                let mut contexts = lanes
                    .iter_mut()
                    .map(|lane| &mut *lane.context)
                    .collect::<Vec<_>>();
                let hidden = self.forward_embedded_batched(&embedded, &mut contexts)?;

                let rows = head_len as i64 + 1;
                let draft_hidden = hidden.slice_axis(1, 1, rows)?;
                let (draft_logits, spanned_layout) = if let Some(head) = draft_head.as_ref() {
                    (head.forward_logits(&draft_hidden)?, head.spanned_layout())
                } else if target.project_proposals_only()
                    || crate::accelerator::cuda_ops_available()
                {
                    (target.logits(&draft_hidden)?, None)
                } else {
                    let logits = target.logits(&hidden)?;
                    (logits.slice_axis(1, 1, rows)?, None)
                };
                Ok((draft_hidden, draft_logits, spanned_layout))
            })();

        let (draft_hidden, draft_logits, spanned_layout) = match shared {
            Ok(tuple) => tuple,
            Err(_) => {
                for lane in lanes.iter_mut() {
                    lane.context.restore_pre_propose_handles();
                }
                return run_per_lane(target, lanes, rngs);
            }
        };

        let mut out = Vec::with_capacity(lanes.len());
        for (i, (lane, rng)) in lanes.iter().zip(rngs.iter_mut()).enumerate() {
            let bi = i as i64;
            let lane_res = (|| {
                let hidden_i = draft_hidden.slice_axis(0, bi, bi + 1)?;
                let logits_i = draft_logits.slice_axis(0, bi, bi + 1)?;
                let mut conf = None;
                let (path, sparse) = self.selector.select_spanned(
                    &hidden_i,
                    &logits_i,
                    lane.anchor,
                    lane.temperature,
                    lane.device_path,
                    spanned_layout,
                    lane.want_confidence.then_some(&mut conf),
                    *rng,
                )?;
                Ok((path, sparse, conf))
            })();
            out.push(lane_res);
        }
        if lanes.len() > 1 && out.iter().all(Result::is_err) {
            drop(out);
            drop((draft_hidden, draft_logits));
            for lane in lanes.iter_mut() {
                lane.context.restore_pre_propose_handles();
            }
            return run_per_lane(target, lanes, rngs);
        }
        out
    }
}

#[cfg(test)]
mod batched_tests {
    use super::*;
    use crate::nn::Linear;

    fn det_bf16(shape: &[i64], seed: u64, scale: f32) -> MxArray {
        let len = shape.iter().product::<i64>() as usize;
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut vals = Vec::with_capacity(len);
        for _ in 0..len {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32) - 1.0;
            vals.push(u * scale);
        }
        MxArray::from_float32(&vals, shape)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    fn build_two_layer_dense_dflash2(with_mask_embedding: bool) -> (DFlash2Model, Embedding) {
        let config = DFlash2Config {
            block_size: 7,
            mask_token_id: 0,
            target_layers: vec![0, 1],
            target_num_layers: 2,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 32,
            vocab_size: 32,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 128,
            sliding_window: 12,
            conv_group_size: 16,
            conv_kernel_size: 2,
            selector_rank: 8,
            selector_top_k: 4,
            rope_theta: 10_000.0,
            final_logit_softcapping: None,
            selector_softcap_mode: None,
            valid_vocab_size: None,
            capture_final_norm: None,
            embedding_normalization: None,
        };
        let seed = std::cell::Cell::new(100u64);
        let next_seed = || {
            let next = seed.get() + 1;
            seed.set(next);
            next
        };
        let mk_linear = |in_f: usize, out_f: usize| -> LinearProj {
            let w = det_bf16(&[out_f as i64, in_f as i64], next_seed(), 0.25);
            LinearProj::Standard(Linear::from_weights(&w, None).unwrap())
        };
        let mk_norm = |size: usize| -> RMSNorm {
            let w = MxArray::ones(&[size as i64], Some(DType::BFloat16)).unwrap();
            RMSNorm::from_weight(&w, Some(config.rms_norm_eps)).unwrap()
        };
        let mk_embed = |rows: usize, width: usize| -> Embedding {
            let w = det_bf16(&[rows as i64, width as i64], next_seed(), 0.4);
            Embedding::from_weight(&w).unwrap()
        };
        let mk_conv = || -> GroupedDynamicCausalConv {
            let groups = config.hidden_size / config.conv_group_size;
            GroupedDynamicCausalConv {
                base_kernel: det_bf16(
                    &[2, config.conv_kernel_size as i64, config.hidden_size as i64],
                    next_seed(),
                    0.2,
                ),
                kernel_projection: mk_linear(
                    config.hidden_size,
                    2 * config.conv_kernel_size * groups,
                ),
                kernel_size: config.conv_kernel_size,
                group_size: config.conv_group_size,
            }
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            let attention = DFlash2Attention {
                tensor_attention_enabled: false,
                q_proj: mk_linear(
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim,
                ),
                k_proj: mk_linear(
                    config.hidden_size,
                    config.num_key_value_heads * config.head_dim,
                ),
                v_proj: mk_linear(
                    config.hidden_size,
                    config.num_key_value_heads * config.head_dim,
                ),
                o_proj: mk_linear(
                    config.num_attention_heads * config.head_dim,
                    config.hidden_size,
                ),
                qkv_proj: None,
                q_norm: mk_norm(config.head_dim),
                k_norm: mk_norm(config.head_dim),
                rope: crate::nn::RoPE::new(
                    config.head_dim as i32,
                    Some(false),
                    Some(config.rope_theta),
                    None,
                ),
                num_heads: config.num_attention_heads as i64,
                num_kv_heads: config.num_key_value_heads as i64,
                head_dim: config.head_dim as i64,
            };
            layers.push(DFlash2Layer {
                attention,
                mlp: DFlash2Mlp {
                    gate_proj: mk_linear(config.hidden_size, config.intermediate_size),
                    up_proj: mk_linear(config.hidden_size, config.intermediate_size),
                    down_proj: mk_linear(config.intermediate_size, config.hidden_size),
                    gate_up: None,
                },
                input_norm: mk_norm(config.hidden_size),
                post_attention_norm: mk_norm(config.hidden_size),
                attention_conv: mk_conv(),
                mlp_conv: mk_conv(),
            });
        }
        let target_embed = mk_embed(config.vocab_size, config.hidden_size);
        let mask_embedding =
            with_mask_embedding.then(|| det_bf16(&[config.hidden_size as i64], next_seed(), 0.3));
        let model = DFlash2Model {
            execution_policy: DraftExecutionPolicy::capture(),
            compiled: CompiledDraft::default(),
            fc: mk_linear(
                config.hidden_size * config.target_layers.len(),
                config.hidden_size,
            ),
            hidden_norm: mk_norm(config.hidden_size),
            layers,
            norm: mk_norm(config.hidden_size),
            selector: CandidateSelector {
                predecessor_codebook: mk_embed(config.vocab_size, config.selector_rank),
                successor_codebook: mk_embed(config.vocab_size, config.selector_rank),
                hidden_projection: mk_linear(config.hidden_size, config.selector_rank),
                top_k: config.selector_top_k,
                rank: config.selector_rank,
                vocab_size: config.vocab_size,
                valid_vocab_size: None,
                post_softcap: None,
            },
            mask_embedding,
            context_kv: None,
            config,
            weight_bytes: 0,
        };
        (model, target_embed)
    }

    fn path_to_ids(path: SelectorPath) -> Vec<i32> {
        match path {
            SelectorPath::Host(ids) => ids,
            SelectorPath::Device(arr) => arr.to_int32().unwrap().to_vec(),
            SelectorPath::Sampled { ids, .. } => ids.to_int32().unwrap().to_vec(),
        }
    }

    fn assert_close_f32(a: &MxArray, b: &MxArray, tol: f32, label: &str) {
        assert_eq!(
            a.shape().unwrap().as_ref(),
            b.shape().unwrap().as_ref(),
            "{label}: shape"
        );
        let av = a.astype(DType::Float32).unwrap().to_float32().unwrap();
        let bv = b.astype(DType::Float32).unwrap().to_float32().unwrap();
        let mut max_diff = 0.0f32;
        for (&x, &y) in av.iter().zip(bv.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(
            max_diff <= tol,
            "{label}: max_diff={max_diff} exceeds tol={tol}"
        );
    }

    #[test]
    fn batched_dflash2_propose_and_tap_append_match_sequential_n2_and_n3() {
        let (model, target_embed) = build_two_layer_dense_dflash2(true);
        let h = model.config.hidden_size as i64;
        let block_len = model.config.block_size;

        for (n_lanes, init_lens, keeps, anchors1, anchors2) in [
            (
                2usize,
                vec![3usize, 9],
                vec![4usize, 5],
                vec![5u32, 11],
                vec![7u32, 19],
            ),
            (
                3usize,
                vec![0usize, 5, 13],
                vec![2usize, 8, 4],
                vec![2u32, 14, 27],
                vec![9u32, 3, 21],
            ),
        ] {
            let mut seq_caches = Vec::with_capacity(n_lanes);
            let mut bat_caches = Vec::with_capacity(n_lanes);

            // Seed each lane with its own distinct initial context length (covering empty,
            // linear, and wrapped sliding windows simultaneously!).
            for (lane_idx, &init_len) in init_lens.iter().enumerate() {
                let mut sc = DFlash2ContextCache::new(&model.config);
                let mut bc = DFlash2ContextCache::new(&model.config);
                if init_len > 0 {
                    let tokens: Vec<u32> = (0..init_len)
                        .map(|t| ((lane_idx * 7 + t + 1) % model.config.vocab_size) as u32)
                        .collect();
                    let t0 = det_bf16(&[1, init_len as i64, h], 1000 + lane_idx as u64 * 10, 0.3);
                    let t1 = det_bf16(&[1, init_len as i64, h], 1001 + lane_idx as u64 * 10, 0.3);
                    let fused = model.fuse_context(&[t0, t1]).unwrap();
                    sc.append(&model, &fused, 0, &tokens).unwrap();
                    bc.append(&model, &fused, 0, &tokens).unwrap();
                }
                seq_caches.push(sc);
                bat_caches.push(bc);
            }

            let mut view = DenseTargetView {
                embedding: &target_embed,
                lm_head: None,
            };

            // Cycle 1: compare batched propose vs sequential propose (both device_path=true + confidence
            // and device_path=false!).
            for device_path in [true, false] {
                let mut seq_ids = Vec::with_capacity(n_lanes);
                let mut seq_confs = Vec::with_capacity(n_lanes);
                for (i, sc) in seq_caches.iter_mut().enumerate() {
                    let mut rng = rand::rng();
                    let mut conf = None;
                    let (path, _) = model
                        .propose_with(
                            &mut view,
                            sc,
                            anchors1[i],
                            block_len,
                            0.0,
                            device_path,
                            device_path.then_some(&mut conf),
                            &mut rng,
                        )
                        .unwrap();
                    seq_ids.push(path_to_ids(path));
                    seq_confs.push(conf);
                }

                let mut rngs_owned: Vec<_> = (0..n_lanes).map(|_| rand::rng()).collect();
                let mut rng_refs: Vec<&mut dyn Rng> =
                    rngs_owned.iter_mut().map(|r| r as &mut dyn Rng).collect();
                let mut batch_lanes: Vec<DFlash2BatchProposalLane<'_>> = bat_caches
                    .iter_mut()
                    .enumerate()
                    .map(|(i, ctx)| DFlash2BatchProposalLane {
                        context: ctx,
                        anchor: anchors1[i],
                        temperature: 0.0,
                        device_path,
                        want_confidence: device_path,
                    })
                    .collect();
                let bat_out = model.propose_with_batched(
                    &mut view,
                    &mut batch_lanes,
                    &mut rng_refs,
                    block_len,
                );

                for (i, res) in bat_out.into_iter().enumerate() {
                    let (path, _, conf) = res.unwrap();
                    assert_eq!(
                        path_to_ids(path),
                        seq_ids[i],
                        "cycle 1 N={n_lanes} lane={i} device_path={device_path} draft ids mismatch"
                    );
                    if let (Some(sc), Some(bc)) = (&seq_confs[i], &conf) {
                        assert_close_f32(
                            sc,
                            bc,
                            2e-2,
                            &format!("cycle 1 N={n_lanes} lane={i} confidence"),
                        );
                    }
                }
            }

            // Batched tap append with DIFFERENT keep lengths per lane!
            let mut per_lane_taps: Vec<[MxArray; 2]> = Vec::with_capacity(n_lanes);
            let mut per_lane_tokens: Vec<Vec<u32>> = Vec::with_capacity(n_lanes);
            for (i, &keep) in keeps.iter().enumerate() {
                let t0 = det_bf16(&[1, keep as i64, h], 2000 + i as u64 * 10, 0.35);
                let t1 = det_bf16(&[1, keep as i64, h], 2001 + i as u64 * 10, 0.35);
                let tokens: Vec<u32> = (0..keep)
                    .map(|t| ((i * 11 + t + 3) % model.config.vocab_size) as u32)
                    .collect();
                let fused = model.fuse_context(&[t0.clone(), t1.clone()]).unwrap();
                let base = seq_caches[i].logical_len();
                seq_caches[i].append(&model, &fused, base, &tokens).unwrap();
                per_lane_taps.push([t0, t1]);
                per_lane_tokens.push(tokens);
            }

            let t0_refs: Vec<&MxArray> = per_lane_taps.iter().map(|t| &t[0]).collect();
            let t1_refs: Vec<&MxArray> = per_lane_taps.iter().map(|t| &t[1]).collect();
            let stacked_t0 = MxArray::concatenate_many(t0_refs, Some(1)).unwrap();
            let stacked_t1 = MxArray::concatenate_many(t1_refs, Some(1)).unwrap();
            let fused_all = model.fuse_context(&[stacked_t0, stacked_t1]).unwrap();
            {
                let mut append_lanes: Vec<(&mut DFlash2ContextCache, &[u32])> = bat_caches
                    .iter_mut()
                    .zip(per_lane_tokens.iter())
                    .map(|(c, toks)| (c, toks.as_slice()))
                    .collect();
                DFlash2ContextCache::append_batched(&model, &fused_all, &mut append_lanes).unwrap();
            }

            // Verify every lane's logical_len, provenance, wide_vocab, and KV window contents!
            for i in 0..n_lanes {
                assert_eq!(bat_caches[i].logical_len(), seq_caches[i].logical_len());
                assert_eq!(bat_caches[i].token_history(), seq_caches[i].token_history());
                assert_eq!(bat_caches[i].wide_vocab(), seq_caches[i].wide_vocab());
                let seq_views = seq_caches[i].window_views();
                let bat_views = bat_caches[i].window_views();
                assert_eq!(seq_views.len(), bat_views.len());
                for (l, (sv, bv)) in seq_views.into_iter().zip(bat_views).enumerate() {
                    let (sk, sv_arr) = sv.unwrap();
                    let (bk, bv_arr) = bv.unwrap();
                    assert_close_f32(
                        &sk,
                        &bk,
                        2e-2,
                        &format!("N={n_lanes} lane={i} layer={l} K window"),
                    );
                    assert_close_f32(
                        &sv_arr,
                        &bv_arr,
                        2e-2,
                        &format!("N={n_lanes} lane={i} layer={l} V window"),
                    );
                }
            }

            // Cycle 2: second batched propose AFTER batched tap append (with wrapped rings)!
            let mut seq_ids2 = Vec::with_capacity(n_lanes);
            for (i, sc) in seq_caches.iter_mut().enumerate() {
                let mut rng = rand::rng();
                let (path, _) = model
                    .propose_with(
                        &mut view,
                        sc,
                        anchors2[i],
                        block_len,
                        0.0,
                        true,
                        None,
                        &mut rng,
                    )
                    .unwrap();
                seq_ids2.push(path_to_ids(path));
            }
            let mut rngs_owned: Vec<_> = (0..n_lanes).map(|_| rand::rng()).collect();
            let mut rng_refs: Vec<&mut dyn Rng> =
                rngs_owned.iter_mut().map(|r| r as &mut dyn Rng).collect();
            let mut batch_lanes: Vec<DFlash2BatchProposalLane<'_>> = bat_caches
                .iter_mut()
                .enumerate()
                .map(|(i, ctx)| DFlash2BatchProposalLane {
                    context: ctx,
                    anchor: anchors2[i],
                    temperature: 0.0,
                    device_path: true,
                    want_confidence: false,
                })
                .collect();
            let bat_out2 =
                model.propose_with_batched(&mut view, &mut batch_lanes, &mut rng_refs, block_len);
            for (i, res) in bat_out2.into_iter().enumerate() {
                let (path, _, _) = res.unwrap();
                assert_eq!(
                    path_to_ids(path),
                    seq_ids2[i],
                    "cycle 2 N={n_lanes} lane={i} draft ids mismatch after batched tap append"
                );
            }
        }
    }

    #[test]
    fn batched_dflash2_quantized_merged_projections_n2_and_n3() {
        let dir = std::env::temp_dir().join(format!(
            "mlx-dflash2-batched-quant-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let cfg = serde_json::json!({
            "architectures": ["DFlash2DraftModel"],
            "hidden_size": 256,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_target_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "vocab_size": 32,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 128,
            "sliding_window": 12,
            "layer_types": ["sliding_attention", "sliding_attention"],
            "rope_theta": 10_000.0,
            "dflash_config": {
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 0,
                "selector_rank": 8,
                "selector_top_k": 4,
                "target_layer_ids": [0, 1],
            },
        });
        write_test_checkpoint(&dir, &cfg).unwrap();
        let loaded = load_dflash2(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        let (model, _) = loaded.unwrap();

        let h = model.config.hidden_size as i64;
        let target_embed = Embedding::from_weight(&det_bf16(&[32, h], 777, 0.3)).unwrap();
        let mut view = DenseTargetView {
            embedding: &target_embed,
            lm_head: None,
        };

        for (n_lanes, keeps, anchors) in [
            (2usize, vec![3usize, 5], vec![4u32, 12]),
            (3usize, vec![2usize, 6, 4], vec![1u32, 15, 29]),
        ] {
            let mut seq_caches: Vec<_> = (0..n_lanes)
                .map(|_| DFlash2ContextCache::new(&model.config))
                .collect();
            let mut bat_caches: Vec<_> = (0..n_lanes)
                .map(|_| DFlash2ContextCache::new(&model.config))
                .collect();

            // Different initial bases via sequential prefill on lane 1..N so every lane has a
            // distinct query_base when entering batched append & batched propose.
            for i in 0..n_lanes {
                let init_len = (i + 1) * 3;
                let t0 = det_bf16(&[1, init_len as i64, h], 3000 + i as u64 * 10, 0.25);
                let t1 = det_bf16(&[1, init_len as i64, h], 3001 + i as u64 * 10, 0.25);
                let fused = model.fuse_context(&[t0, t1]).unwrap();
                let toks: Vec<u32> = (0..init_len).map(|k| ((i + k) % 32) as u32).collect();
                seq_caches[i].append(&model, &fused, 0, &toks).unwrap();
                bat_caches[i].append(&model, &fused, 0, &toks).unwrap();
            }

            // Batched append through merged `context_kv` (when tiled/CUDA) vs sequential append.
            let mut per_lane_taps = Vec::with_capacity(n_lanes);
            let mut per_lane_toks = Vec::with_capacity(n_lanes);
            for (i, &keep) in keeps.iter().enumerate() {
                let t0 = det_bf16(&[1, keep as i64, h], 4000 + i as u64 * 10, 0.25);
                let t1 = det_bf16(&[1, keep as i64, h], 4001 + i as u64 * 10, 0.25);
                let toks: Vec<u32> = (0..keep).map(|k| ((i * 5 + k + 1) % 32) as u32).collect();
                let fused = model.fuse_context(&[t0.clone(), t1.clone()]).unwrap();
                let base = seq_caches[i].logical_len();
                seq_caches[i].append(&model, &fused, base, &toks).unwrap();
                per_lane_taps.push([t0, t1]);
                per_lane_toks.push(toks);
            }
            let t0_refs: Vec<&MxArray> = per_lane_taps.iter().map(|t| &t[0]).collect();
            let t1_refs: Vec<&MxArray> = per_lane_taps.iter().map(|t| &t[1]).collect();
            let fused_all = model
                .fuse_context(&[
                    MxArray::concatenate_many(t0_refs, Some(1)).unwrap(),
                    MxArray::concatenate_many(t1_refs, Some(1)).unwrap(),
                ])
                .unwrap();
            {
                let mut lanes: Vec<(&mut DFlash2ContextCache, &[u32])> = bat_caches
                    .iter_mut()
                    .zip(per_lane_toks.iter())
                    .map(|(c, t)| (c, t.as_slice()))
                    .collect();
                DFlash2ContextCache::append_batched(&model, &fused_all, &mut lanes).unwrap();
            }

            for i in 0..n_lanes {
                assert_eq!(bat_caches[i].logical_len(), seq_caches[i].logical_len());
                assert_eq!(bat_caches[i].token_history(), seq_caches[i].token_history());
                for (l, (sv, bv)) in seq_caches[i]
                    .window_views()
                    .into_iter()
                    .zip(bat_caches[i].window_views())
                    .enumerate()
                {
                    let (sk, sv_arr) = sv.unwrap();
                    let (bk, bv_arr) = bv.unwrap();
                    assert_close_f32(
                        &sk,
                        &bk,
                        5e-2,
                        &format!("quant N={n_lanes} lane={i} layer={l} K window"),
                    );
                    assert_close_f32(
                        &sv_arr,
                        &bv_arr,
                        5e-2,
                        &format!("quant N={n_lanes} lane={i} layer={l} V window"),
                    );
                }
            }

            // Batched propose through merged `qkv_proj` + `gate_up` across N lanes.
            let mut rngs_owned: Vec<_> = (0..n_lanes).map(|_| rand::rng()).collect();
            let mut rng_refs: Vec<&mut dyn Rng> =
                rngs_owned.iter_mut().map(|r| r as &mut dyn Rng).collect();
            let mut batch_lanes: Vec<DFlash2BatchProposalLane<'_>> = bat_caches
                .iter_mut()
                .enumerate()
                .map(|(i, ctx)| DFlash2BatchProposalLane {
                    context: ctx,
                    anchor: anchors[i],
                    temperature: 0.0,
                    device_path: true,
                    want_confidence: false,
                })
                .collect();
            let bat_out = model.propose_with_batched(
                &mut view,
                &mut batch_lanes,
                &mut rng_refs,
                model.config.block_size,
            );
            assert_eq!(bat_out.len(), n_lanes);
            for res in bat_out {
                let (path, _, _) = res.unwrap();
                assert_eq!(path_to_ids(path).len(), model.config.block_size);
            }
        }
    }

    #[test]
    fn batched_dflash2_restore_pre_propose_handles_recovers_committed_ring_for_solo_retry() {
        let (model, target_embed) = build_two_layer_dense_dflash2(true);
        let h = model.config.hidden_size as i64;
        let block_len = model.config.block_size;
        let init_lens = [4usize, 13usize];
        let anchors = [6u32, 17u32];

        let mut seq_caches = Vec::with_capacity(init_lens.len());
        let mut bat_caches = Vec::with_capacity(init_lens.len());
        for (lane_idx, &init_len) in init_lens.iter().enumerate() {
            let mut sc = DFlash2ContextCache::new(&model.config);
            let mut bc = DFlash2ContextCache::new(&model.config);
            let tokens: Vec<u32> = (0..init_len)
                .map(|t| ((lane_idx * 9 + t + 2) % model.config.vocab_size) as u32)
                .collect();
            let t0 = det_bf16(&[1, init_len as i64, h], 6100 + lane_idx as u64 * 10, 0.3);
            let t1 = det_bf16(&[1, init_len as i64, h], 6101 + lane_idx as u64 * 10, 0.3);
            let fused = model.fuse_context(&[t0, t1]).unwrap();
            sc.append(&model, &fused, 0, &tokens).unwrap();
            bc.append(&model, &fused, 0, &tokens).unwrap();
            sc.eval().unwrap();
            bc.eval().unwrap();
            seq_caches.push(sc);
            bat_caches.push(bc);
        }

        let pre_ptrs: Vec<Vec<(usize, usize)>> = bat_caches
            .iter()
            .map(|cache| {
                cache
                    .layers
                    .iter()
                    .map(|window| {
                        (
                            window.keys.as_ref().expect("committed keys").as_raw_ptr() as usize,
                            window
                                .values
                                .as_ref()
                                .expect("committed values")
                                .as_raw_ptr() as usize,
                        )
                    })
                    .collect()
            })
            .collect();

        let mut view = DenseTargetView {
            embedding: &target_embed,
            lm_head: None,
        };

        // Compute bit-exact solo references before batched mutation.
        let mut seq_ids = Vec::with_capacity(init_lens.len());
        for (i, sc) in seq_caches.iter_mut().enumerate() {
            let mut rng = rand::rng();
            let (path, _) = model
                .propose_with(
                    &mut view, sc, anchors[i], block_len, 0.0, true, None, &mut rng,
                )
                .unwrap();
            seq_ids.push(path_to_ids(path));
        }

        // Run batched proposal (device_path=true leaves lazy graph attached to ring handles),
        // then drop the lazy batch outputs as if deferred GPU evaluation failed.
        {
            let mut rngs_owned: Vec<_> = (0..init_lens.len()).map(|_| rand::rng()).collect();
            let mut rng_refs: Vec<&mut dyn Rng> =
                rngs_owned.iter_mut().map(|r| r as &mut dyn Rng).collect();
            let mut batch_lanes: Vec<DFlash2BatchProposalLane<'_>> = bat_caches
                .iter_mut()
                .enumerate()
                .map(|(i, ctx)| DFlash2BatchProposalLane {
                    context: ctx,
                    anchor: anchors[i],
                    temperature: 0.0,
                    device_path: true,
                    want_confidence: false,
                })
                .collect();
            let bat_out =
                model.propose_with_batched(&mut view, &mut batch_lanes, &mut rng_refs, block_len);
            assert_eq!(bat_out.len(), init_lens.len());
            for res in bat_out {
                assert!(res.is_ok(), "lazy batched proposal build should succeed");
            }
        }

        // `with_block` must have replaced every layer's committed ring handle while keeping
        // the pre-proposal snapshot available for restoration.
        for (i, cache) in bat_caches.iter().enumerate() {
            assert!(
                cache.pre_propose_handles.is_some(),
                "lane {i} must retain pre_propose_handles after propose_with_batched"
            );
            for (l, window) in cache.layers.iter().enumerate() {
                let mutated = (
                    window.keys.as_ref().expect("mutated keys").as_raw_ptr() as usize,
                    window.values.as_ref().expect("mutated values").as_raw_ptr() as usize,
                );
                assert_ne!(
                    mutated, pre_ptrs[i][l],
                    "lane {i} layer {l} ring handles must mutate during with_block"
                );
            }
        }

        // Restoring pre-proposal handles recovers exact committed ring handle identity,
        // and solo retry matches solo reference output bit-for-bit.
        for (i, cache) in bat_caches.iter_mut().enumerate() {
            cache.restore_pre_propose_handles();
            assert!(
                cache.pre_propose_handles.is_none(),
                "restore_pre_propose_handles must consume the snapshot"
            );
            for (l, window) in cache.layers.iter().enumerate() {
                let restored = (
                    window.keys.as_ref().expect("restored keys").as_raw_ptr() as usize,
                    window
                        .values
                        .as_ref()
                        .expect("restored values")
                        .as_raw_ptr() as usize,
                );
                assert_eq!(
                    restored, pre_ptrs[i][l],
                    "lane {i} layer {l} must restore exact pre-proposal raw handles"
                );
            }
            let mut rng = rand::rng();
            let (path, _) = model
                .propose_with(
                    &mut view, cache, anchors[i], block_len, 0.0, true, None, &mut rng,
                )
                .unwrap();
            assert_eq!(
                path_to_ids(path),
                seq_ids[i],
                "solo retry after restore must match solo reference bit-for-bit on lane {i}"
            );
        }

        // Mutating lifecycle methods must clear any saved snapshot so stale handles cannot linger.
        let cache = &mut bat_caches[0];
        cache.save_pre_propose_handles();
        assert!(cache.pre_propose_handles.is_some());
        let t0 = det_bf16(&[1, 1, h], 6900, 0.3);
        let t1 = det_bf16(&[1, 1, h], 6901, 0.3);
        let fused = model.fuse_context(&[t0, t1]).unwrap();
        let base = cache.logical_len();
        cache.append(&model, &fused, base, &[3]).unwrap();
        assert!(cache.pre_propose_handles.is_none());

        cache.save_pre_propose_handles();
        assert!(cache.pre_propose_handles.is_some());
        cache.record_only(&[4]).unwrap();
        assert!(cache.pre_propose_handles.is_none());

        cache.save_pre_propose_handles();
        assert!(cache.pre_propose_handles.is_some());
        cache.reset();
        assert!(cache.pre_propose_handles.is_none());
    }
}
