use crate::array::{DType, MxArray};
use crate::nn::RMSNorm;
use crate::transformer::MLP;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use crate::transformer::paged_kv_cache_adapter::SeqId;
use napi::bindgen_prelude::*;

use super::attention::{AttentionMask, PagedPass, PagedWaveLane, Qwen3_5Attention, RopeMode};
use super::config::{Qwen3_5Config, Qwen35MoeParams};
use super::gated_delta_net::GatedDeltaNet;
use super::gdn_blob::{GdnWaveCells, gdn_wave_layer};
use super::layer_cache::Qwen3_5LayerCache;
use super::sparse_moe::SparseMoeBlock;
use crate::models::quantized_linear::{MLPVariant, QuantizedLinear};

/// Per-layer routing kind for Qwen3.5's paged dispatch.
///
/// Mirrors `Lfm2LayerKind` / `Gemma4LayerKind` but reflects Qwen3.5's
/// hybrid layout (linear attention via GatedDeltaNet + full attention).
/// Only full-attention layers route through the paged adapter; linear
/// (GDN) layers continue to use `Qwen3_5LayerCache::Linear(ArraysCache)`
/// with no cross-request prefix reuse — vLLM's `MambaManager`-style "no
/// prefix reuse for recurrent layers" stance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen3_5LayerKind {
    /// GDN linear-attention layer that stays on the existing flat
    /// `Qwen3_5LayerCache::Linear` path.
    Linear,
    /// Full-attention layer routed through the paged adapter.
    /// `paged_idx` is the FULL-ATTENTION ORDINAL into the adapter's
    /// `LayerKVPool` (NOT the absolute decoder index). The pool is
    /// sized for `Qwen3_5Config::full_attention_layer_count()` slots.
    FullAttentionPaged { paged_idx: u32 },
}

/// Attention type for a decoder layer.
///
/// GDN carries native split packed-projection state and is the common hot path
/// in this hybrid model. Keep it inline rather than adding a pointer chase to
/// every linear-attention forward merely to equalize enum variant sizes.
#[allow(clippy::large_enum_variant)]
pub enum AttentionType {
    Linear(GatedDeltaNet),
    Full(Qwen3_5Attention),
}

/// Build the per-layer routing list for Qwen3.5 (dense or MoE).
///
/// Walks `is_linear_layer(i)` to assign:
/// * `Linear` for GDN layers,
/// * `FullAttentionPaged { paged_idx }` for full-attention layers
///   where `paged_idx` is the ordinal index counting only
///   full-attention layers (i.e. the index into the paged adapter's
///   `LayerKVPool`).
pub(crate) fn compute_layer_kinds(
    num_layers: usize,
    is_linear: impl Fn(usize) -> bool,
) -> Vec<Qwen3_5LayerKind> {
    let mut kinds = Vec::with_capacity(num_layers);
    let mut paged_idx: u32 = 0;
    for i in 0..num_layers {
        if is_linear(i) {
            kinds.push(Qwen3_5LayerKind::Linear);
        } else {
            kinds.push(Qwen3_5LayerKind::FullAttentionPaged { paged_idx });
            paged_idx += 1;
        }
    }
    kinds
}

/// Input to a decoder layer when cross-layer `AddRMSNorm` fusion is active:
/// either a materialized hidden state (layer 0 after embedding) or a deferred
/// `(h, delta)` pair from the previous layer's MLP whose sum is fused with
/// this layer's `input_layernorm` (or `final_norm` after the last layer).
pub(crate) enum LayerResidual {
    Hidden(MxArray),
    Pending { h: MxArray, delta: MxArray },
}

enum ResolvedInputNorm {
    Normed(MxArray),
    PrequantFull {
        x_i8: MxArray,
        s_x: MxArray,
    },
    PrequantLinear {
        normed: MxArray,
        x_i8: MxArray,
        s_x: MxArray,
    },
}

impl LayerResidual {
    pub(crate) fn seq_len(&self) -> Result<i64> {
        match self {
            Self::Hidden(x) => x.shape_at(1),
            Self::Pending { h, .. } => h.shape_at(1),
        }
    }

    pub(crate) fn into_hidden(self) -> Result<MxArray> {
        match self {
            Self::Hidden(x) => Ok(x),
            Self::Pending { h, delta } => h.add(&delta),
        }
    }
}

/// MLP type for a Qwen3.5 decoder layer (dense MLP or sparse MoE block).
///
/// The `Dense` variant carries an inline `MLPVariant`; `MoE` is boxed.
/// Constructed once per layer at load, never in a hot loop.
#[allow(clippy::large_enum_variant)]
pub enum MLPType {
    Dense(MLPVariant),
    MoE(Box<SparseMoeBlock>),
}

impl MLPType {
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::MoE(moe) => moe.forward(x),
        }
    }

    pub fn forward_with_budget(
        &self,
        x: &MxArray,
        row_budget: Option<&MxArray>,
    ) -> Result<MxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::MoE(moe) => moe.forward_with_budget(x, row_budget),
        }
    }

    pub(crate) fn forward_store_residual_with_int8(
        &self,
        x: &MxArray,
        residual: &MxArray,
        int8_act: bool,
    ) -> Result<MxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward_store_residual_with_int8(x, residual, int8_act),
            Self::MoE(moe) => residual.add(&moe.forward(x)?),
        }
    }

    pub(crate) fn forward_with_int8(&self, x: &MxArray, int8_act: bool) -> Result<MxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward_with_int8(x, int8_act),
            Self::MoE(moe) => moe.forward(x),
        }
    }

    pub(crate) fn forward_verify_with_budget(
        &self,
        x: &MxArray,
        residual: &MxArray,
        row_budget: Option<&MxArray>,
    ) -> Result<MxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward_verify(x, residual),
            Self::MoE(moe) => residual.add(&moe.forward_with_budget(x, row_budget)?),
        }
    }

    pub(crate) fn forward_verify_lanes(&self, x: &MxArray, residual: &MxArray) -> Result<MxArray> {
        match self {
            Self::Dense(mlp) => mlp.forward_verify_lanes(x, residual),
            Self::MoE(moe) => residual.add(&moe.forward(x)?),
        }
    }

    pub fn is_quantized(&self) -> bool {
        match self {
            Self::Dense(mlp) => mlp.is_quantized(),
            Self::MoE(moe) => moe.is_quantized(),
        }
    }

    pub fn finalize_gate_up(&mut self) -> Result<()> {
        match self {
            Self::Dense(mlp) => mlp.finalize_gate_up(),
            Self::MoE(_) => Ok(()),
        }
    }

    pub fn as_dense(&self) -> Option<&MLPVariant> {
        match self {
            Self::Dense(mlp) => Some(mlp),
            Self::MoE(_) => None,
        }
    }

    pub fn as_dense_mut(&mut self) -> Option<&mut MLPVariant> {
        match self {
            Self::Dense(mlp) => Some(mlp),
            Self::MoE(_) => None,
        }
    }

    pub fn as_moe(&self) -> Option<&SparseMoeBlock> {
        match self {
            Self::MoE(moe) => Some(moe),
            Self::Dense(_) => None,
        }
    }

    pub fn as_moe_mut(&mut self) -> Option<&mut SparseMoeBlock> {
        match self {
            Self::MoE(moe) => Some(moe),
            Self::Dense(_) => None,
        }
    }

    pub(crate) fn attach_planes(
        &mut self,
        array: MxArray,
        tensor: crate::utils::gguf_planes::PlanesTensor,
    ) -> Result<()> {
        match self {
            Self::Dense(mlp) => mlp.attach_planes(array, tensor),
            Self::MoE(_) => Err(Error::from_reason(format!(
                "planes {} cannot attach to an MoE MLP",
                tensor.key
            ))),
        }
    }

    pub(crate) fn planes(&self) -> Vec<&crate::models::quantized_linear::AttachedPlanes> {
        match self {
            Self::Dense(mlp) => mlp.planes(),
            Self::MoE(_) => Vec::new(),
        }
    }

    pub(crate) fn detach_planes(&mut self) -> Vec<crate::models::quantized_linear::AttachedPlanes> {
        match self {
            Self::Dense(mlp) => mlp.detach_planes(),
            Self::MoE(_) => Vec::new(),
        }
    }

    pub(crate) fn clear_cuda_dense_prefill(&self) {
        match self {
            Self::Dense(mlp) => mlp.clear_cuda_dense_prefill(),
            Self::MoE(_) => {}
        }
    }

    pub(crate) fn can_fuse_norm_sym8_cuda(&self) -> Option<Option<&MxArray>> {
        match self {
            Self::Dense(mlp) => mlp.can_fuse_norm_sym8_cuda(),
            Self::MoE(_) => None,
        }
    }

    pub(crate) fn forward_prequantized_sym8_cuda(
        &self,
        x_i8: &MxArray,
        s_x: &MxArray,
        prefix_shape: &[i64],
        out_dtype: DType,
    ) -> Result<MxArray> {
        match self {
            Self::Dense(mlp) => {
                mlp.forward_prequantized_sym8_cuda(x_i8, s_x, prefix_shape, out_dtype)
            }
            Self::MoE(_) => Err(Error::from_reason(
                "MLPType::forward_prequantized_sym8_cuda requires a dense MLP",
            )),
        }
    }

    pub fn wave_ready(&self, hidden: i64) -> Result<bool> {
        match self {
            Self::Dense(mlp) => mlp.wave_ready(hidden),
            Self::MoE(_) => Ok(false),
        }
    }

    #[cfg(test)]
    pub(crate) fn has_hadamard(&self) -> Option<(bool, bool, bool)> {
        match self {
            Self::Dense(mlp) => mlp.has_hadamard(),
            Self::MoE(_) => None,
        }
    }
}

/// A single decoder layer in the Qwen3.5 model (dense or MoE).
///
/// Each layer has:
/// - Either linear attention (GatedDeltaNet) or full attention (Qwen3NextAttention)
/// - Either a dense MLP (standard or quantized) or a sparse MoE block
/// - Pre-norm architecture with residual connections
pub struct DecoderLayer {
    pub attn: AttentionType,
    pub mlp: MLPType,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    pub(crate) fn clear_cuda_dense_prefill(&self) {
        match &self.attn {
            AttentionType::Linear(gdn) => gdn.clear_cuda_dense_prefill(),
            AttentionType::Full(attention) => attention.clear_cuda_dense_prefill(),
        }
        self.mlp.clear_cuda_dense_prefill();
    }

    /// `h = x + res` + `normed = norm(h)` — one Metal dispatch via the fused
    /// add_rmsnorm kernel when the contract holds, else the separate ops.
    /// The fused kernel is bit-identical to `x.add(res)` + `norm.forward(h)`
    /// (same element mapping as `rms_single_row`/`rms_looped`), so this is
    /// safe on every path — eager, paged, prefill and compiled verify alike.
    pub(crate) fn add_residual_norm(
        norm: &RMSNorm,
        x: &MxArray,
        res: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        if let Some(pair) = norm.forward_residual_add(x, res) {
            return Ok(pair);
        }
        let h = x.add(res)?;
        let normed = norm.forward(&h)?;
        Ok((h, normed))
    }

    fn try_sym8_norm_hadamard_quant(
        norm: &RMSNorm,
        h: &MxArray,
        delta: Option<&MxArray>,
        signs: Option<&MxArray>,
        write_normed: bool,
    ) -> Result<Option<crate::kernels::sym8::Sym8NormHadQuantOutput>> {
        let w_is_f32 = if norm.is_precise_f32() {
            true
        } else if norm.is_native_policy() {
            false
        } else {
            return Ok(None);
        };
        crate::kernels::sym8::add_rmsnorm_hadamard_quant(
            h,
            delta,
            norm.weight(),
            norm.eps_f32(),
            signs,
            write_normed,
            w_is_f32,
        )
    }

    /// Resolve `input` through `input_layernorm`, fusing `(optional residual add) +
    /// RMSNorm + optional Hadamard-1024 + INT8 quant` into one CUDA kernel when
    /// the mixer's input projections are `Sym8`-eligible.
    fn resolve_input_norm(&self, input: LayerResidual) -> Result<(MxArray, ResolvedInputNorm)> {
        if crate::accelerator::cuda_sym8_available() {
            let (h_ref, delta_ref) = match &input {
                LayerResidual::Hidden(x) => (x, None),
                LayerResidual::Pending { h, delta } => (h, Some(delta)),
            };
            if h_ref.ndim()? == 3 {
                match &self.attn {
                    AttentionType::Full(attn) => {
                        if let Some(signs) = attn.can_fuse_norm_sym8_cuda()?
                            && let Some(out) = Self::try_sym8_norm_hadamard_quant(
                                &self.input_layernorm,
                                h_ref,
                                delta_ref,
                                signs,
                                false,
                            )?
                        {
                            let completed_x = match out.h_out {
                                Some(h_out) => h_out,
                                None => match input {
                                    LayerResidual::Hidden(x) => x,
                                    LayerResidual::Pending { .. } => {
                                        return Err(Error::from_reason(
                                            "sym8 fused input norm missing residual output",
                                        ));
                                    }
                                },
                            };
                            return Ok((
                                completed_x,
                                ResolvedInputNorm::PrequantFull {
                                    x_i8: out.x_i8,
                                    s_x: out.s_x,
                                },
                            ));
                        }
                    }
                    AttentionType::Linear(gdn) => {
                        if let Some(signs) = gdn.can_fuse_norm_sym8_cuda()
                            && let Some(out) = Self::try_sym8_norm_hadamard_quant(
                                &self.input_layernorm,
                                h_ref,
                                delta_ref,
                                signs,
                                true,
                            )?
                        {
                            let normed = out.normed_out.ok_or_else(|| {
                                Error::from_reason(
                                    "sym8 fused GDN input norm missing normed output",
                                )
                            })?;
                            let completed_x = match out.h_out {
                                Some(h_out) => h_out,
                                None => match input {
                                    LayerResidual::Hidden(x) => x,
                                    LayerResidual::Pending { .. } => {
                                        return Err(Error::from_reason(
                                            "sym8 fused input norm missing residual output",
                                        ));
                                    }
                                },
                            };
                            return Ok((
                                completed_x,
                                ResolvedInputNorm::PrequantLinear {
                                    normed,
                                    x_i8: out.x_i8,
                                    s_x: out.s_x,
                                },
                            ));
                        }
                    }
                }
            }
        }
        let (x, normed) = match input {
            LayerResidual::Hidden(x) => {
                let normed = self.input_layernorm.forward(&x)?;
                (x, normed)
            }
            LayerResidual::Pending { h, delta } => {
                Self::add_residual_norm(&self.input_layernorm, &h, &delta)?
            }
        };
        Ok((x, ResolvedInputNorm::Normed(normed)))
    }

    /// Compute `(h = x + attn_out, mlp_out)` fusing `Add + post_attention_layernorm +
    /// optional Hadamard-1024 + INT8 quant` into one CUDA kernel on `Sym8` MLPs.
    fn post_attention_mlp_pending(
        &self,
        x: &MxArray,
        attn_out: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        self.post_attention_mlp_pending_with_budget(x, attn_out, None)
    }

    /// Variant of [`Self::post_attention_mlp_pending`] forwarding `row_budget`
    /// to MoE MLPs (dense MLPs ignore `row_budget` and keep CUDA `Sym8` fusion).
    fn post_attention_mlp_pending_with_budget(
        &self,
        x: &MxArray,
        attn_out: &MxArray,
        row_budget: Option<&MxArray>,
    ) -> Result<(MxArray, MxArray)> {
        if let Some(signs) = self.mlp.can_fuse_norm_sym8_cuda()
            && let Some(out) = Self::try_sym8_norm_hadamard_quant(
                &self.post_attention_layernorm,
                x,
                Some(attn_out),
                signs,
                false,
            )?
        {
            let h = out.h_out.ok_or_else(|| {
                Error::from_reason("sym8 fused post-attention norm missing residual output")
            })?;
            let shape = h.shape()?;
            let prefix_shape = &shape[..shape.len() - 1];
            let mlp_out = self.mlp.forward_prequantized_sym8_cuda(
                &out.x_i8,
                &out.s_x,
                prefix_shape,
                DType::BFloat16,
            )?;
            return Ok((h, mlp_out));
        }
        let (h, normed) = Self::add_residual_norm(&self.post_attention_layernorm, x, attn_out)?;
        let mlp_out = self.mlp.forward_with_budget(&normed, row_budget)?;
        Ok((h, mlp_out))
    }

    /// Whether this layer uses linear attention (derived from attention type).
    pub fn is_linear(&self) -> bool {
        matches!(self.attn, AttentionType::Linear(_))
    }

    /// Prefill tail pruning for the final layer when it is a FullAttention layer
    /// past the last tap, preserving cross-layer `AddRMSNorm` fusion: resolves
    /// `input` through `input_layernorm` (returning the completed previous
    /// layer hidden state as the first tuple element), updates the KV cache for
    /// all rows, and either skips Q/SDPA/O/MLP (`keep_last_row == false`) or
    /// computes them only on the last row, returning `LayerResidual::Pending`
    /// so `final_norm` fuses the last MLP residual too.
    pub(crate) fn forward_final_prefill_pruned_pending(
        &mut self,
        input: &LayerResidual,
        cache: Option<&mut Qwen3_5LayerCache>,
        keep_last_row: bool,
    ) -> Result<Option<(MxArray, LayerResidual)>> {
        let AttentionType::Full(attn) = &self.attn else {
            return Ok(None);
        };
        let kv_cache = match cache {
            Some(Qwen3_5LayerCache::FullAttention(kv)) => Some(kv),
            None => None,
            Some(Qwen3_5LayerCache::Linear(_)) => return Ok(None),
        };
        let seq_len = input.seq_len()?;
        if seq_len <= 1 {
            return Ok(None);
        }
        let (x, normed) = match input {
            LayerResidual::Hidden(x) => (x.clone(), self.input_layernorm.forward(x)?),
            LayerResidual::Pending { h, delta } => {
                Self::add_residual_norm(&self.input_layernorm, h, delta)?
            }
        };
        if !keep_last_row {
            attn.forward_kv_only(&normed, kv_cache)?;
            let last = x.slice_axis(1, seq_len - 1, seq_len)?;
            return Ok(Some((x, LayerResidual::Hidden(last))));
        }
        let attn_last = attn.forward_last_row_prefill(&normed, kv_cache)?;
        let x_last = x.slice_axis(1, seq_len - 1, seq_len)?;
        let (h, mlp_out) = self.post_attention_mlp_pending(&x_last, &attn_last)?;
        Ok(Some((x, LayerResidual::Pending { h, delta: mlp_out })))
    }

    /// Set the rotary position embedding mode for full-attention layers.
    /// Linear (GDN) layers ignore this setting.
    #[allow(dead_code)]
    pub(crate) fn set_rope_mode(&mut self, mode: RopeMode) {
        if let AttentionType::Full(attn) = &mut self.attn {
            attn.set_rope_mode(mode);
        }
    }

    /// Batched flat-cache forward pass with separate full-attention and linear
    /// masks plus per-row valid lengths for right-padded GDN `conv_state`
    /// updates.
    ///
    /// Routes `attn_mask` + `position_ids` to `Qwen3_5Attention::forward`, and
    /// `linear_mask` + `linear_lengths` to
    /// `GatedDeltaNet::forward_batched_masked`.
    #[allow(dead_code)]
    pub(crate) fn forward_batched_flat(
        &self,
        x: &MxArray,
        attn_mask: Option<&MxArray>,
        linear_mask: Option<&MxArray>,
        linear_lengths: Option<&[i32]>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
    ) -> Result<MxArray> {
        self.forward_batched_flat_with_mask(
            x,
            attn_mask.map(AttentionMask::Full),
            linear_mask,
            linear_lengths,
            cache,
            position_ids,
        )
    }

    /// Variant of [`Self::forward_batched_flat`] accepting an explicit
    /// [`AttentionMask`] mode for full-attention layers.
    pub(crate) fn forward_batched_flat_with_mask(
        &self,
        x: &MxArray,
        attn_mask: Option<AttentionMask<'_>>,
        linear_mask: Option<&MxArray>,
        linear_lengths: Option<&[i32]>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
    ) -> Result<MxArray> {
        let (_, resolved) = self.resolve_input_norm(LayerResidual::Hidden(x.clone()))?;
        let attn_out = match (&self.attn, resolved) {
            (
                AttentionType::Linear(gdn),
                ResolvedInputNorm::PrequantLinear { normed, x_i8, s_x },
            ) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward_batched_masked_prequant_sym8(
                    &normed,
                    (&x_i8, &s_x),
                    linear_mask,
                    linear_lengths,
                    ac,
                )?
            }
            (AttentionType::Linear(gdn), ResolvedInputNorm::Normed(normed)) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward_batched_masked(&normed, linear_mask, linear_lengths, ac)?
            }
            (AttentionType::Full(attn), ResolvedInputNorm::PrequantFull { x_i8, s_x }) => {
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward_with_mask_prequant_sym8(&x_i8, &s_x, attn_mask, kvc, position_ids)?
            }
            (AttentionType::Full(attn), ResolvedInputNorm::Normed(normed)) => {
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward_with_mask(&normed, attn_mask, kvc, position_ids)?
            }
            _ => unreachable!("resolve_input_norm variant always matches self.attn"),
        };

        let (h, mlp_out) = self.post_attention_mlp_pending(x, &attn_out)?;
        h.add(&mlp_out)
    }

    /// Cross-layer `AddRMSNorm` variant of [`Self::forward_batched_flat`]:
    /// resolves `input` through `input_layernorm` (fusing the previous layer's
    /// MLP residual add into `input_layernorm` when `input` is
    /// [`LayerResidual::Pending`]), runs the batched mixer and MLP, and returns
    /// [`LayerResidual::Pending`] so the next layer's `input_layernorm` (or
    /// `final_norm` after the last layer) fuses this layer's MLP residual too.
    #[allow(dead_code)]
    pub(crate) fn forward_batched_flat_pending(
        &self,
        input: LayerResidual,
        attn_mask: Option<&MxArray>,
        linear_mask: Option<&MxArray>,
        linear_lengths: Option<&[i32]>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
    ) -> Result<LayerResidual> {
        self.forward_batched_flat_pending_with_mask(
            input,
            attn_mask.map(AttentionMask::Full),
            linear_mask,
            linear_lengths,
            cache,
            position_ids,
        )
    }

    /// Variant of [`Self::forward_batched_flat_pending`] accepting an explicit
    /// [`AttentionMask`] mode for full-attention layers.
    pub(crate) fn forward_batched_flat_pending_with_mask(
        &self,
        input: LayerResidual,
        attn_mask: Option<AttentionMask<'_>>,
        linear_mask: Option<&MxArray>,
        linear_lengths: Option<&[i32]>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
    ) -> Result<LayerResidual> {
        let (x, resolved) = self.resolve_input_norm(input)?;
        let attn_out = match (&self.attn, resolved) {
            (
                AttentionType::Linear(gdn),
                ResolvedInputNorm::PrequantLinear { normed, x_i8, s_x },
            ) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward_batched_masked_prequant_sym8(
                    &normed,
                    (&x_i8, &s_x),
                    linear_mask,
                    linear_lengths,
                    ac,
                )?
            }
            (AttentionType::Linear(gdn), ResolvedInputNorm::Normed(normed)) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward_batched_masked(&normed, linear_mask, linear_lengths, ac)?
            }
            (AttentionType::Full(attn), ResolvedInputNorm::PrequantFull { x_i8, s_x }) => {
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward_with_mask_prequant_sym8(&x_i8, &s_x, attn_mask, kvc, position_ids)?
            }
            (AttentionType::Full(attn), ResolvedInputNorm::Normed(normed)) => {
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward_with_mask(&normed, attn_mask, kvc, position_ids)?
            }
            _ => unreachable!("resolve_input_norm variant always matches self.attn"),
        };

        let (h, mlp_out) = self.post_attention_mlp_pending(&x, &attn_out)?;
        Ok(LayerResidual::Pending { h, delta: mlp_out })
    }
    pub fn new(config: &Qwen3_5Config, layer_idx: usize) -> Result<Self> {
        Self::new_with_moe(config, None, layer_idx)
    }

    pub(crate) fn new_with_moe(
        config: &Qwen3_5Config,
        moe: Option<&Qwen35MoeParams>,
        layer_idx: usize,
    ) -> Result<Self> {
        let is_linear = config.is_linear_layer(layer_idx);

        let attn = if is_linear {
            AttentionType::Linear(GatedDeltaNet::new(config)?)
        } else {
            AttentionType::Full(Qwen3_5Attention::new(config)?)
        };

        let mlp = match moe {
            Some(moe_params) if moe_params.is_moe_layer(layer_idx) => MLPType::MoE(Box::new(
                SparseMoeBlock::new_with_params(config, moe_params)?,
            )),
            _ => MLPType::Dense(MLPVariant::Standard(MLP::new(
                config.hidden_size as u32,
                config.intermediate_size as u32,
            )?)),
        };

        let input_layernorm = RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;
        let post_attention_layernorm =
            RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward_mlp_decode_rows(&self, x: &MxArray) -> Result<MxArray> {
        let batch = usize::try_from(x.shape_at(0)?)
            .map_err(|_| Error::from_reason("Qwen3.5 decode MLP received a negative batch"))?;
        if batch == 0 {
            return Err(Error::from_reason(
                "Qwen3.5 decode MLP requires at least one row",
            ));
        }
        let rows = (0..batch)
            .map(|row| {
                let row = x.slice_axis(0, row as i64, row as i64 + 1)?;
                self.mlp.forward(&row)
            })
            .collect::<Result<Vec<_>>>()?;
        MxArray::concatenate_many(rows.iter().collect(), Some(0))
    }

    /// Post-attention half of the layer whose expert tail routes only the rows
    /// of `row_budget` (Splash `budgetsVerifyRows`). Dense MLPs ignore
    /// `row_budget` and take the fused CUDA `Sym8` path when eligible.
    fn finish_layer_with_budget(
        &self,
        x: &MxArray,
        attn_out: MxArray,
        row_budget: Option<&MxArray>,
    ) -> Result<MxArray> {
        let (h, mlp_out) = self.post_attention_mlp_pending_with_budget(x, &attn_out, row_budget)?;
        h.add(&mlp_out)
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input [B, T, hidden_size]
    /// * `mask` - Attention mask (causal)
    /// * `cache` - Optional layer cache
    /// * `position_ids` - Optional [3, B, T] M-RoPE positions (VLM only, full attention layers)
    pub fn forward(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
    ) -> Result<MxArray> {
        self.forward_inner(x, mask, cache, position_ids, use_kernel)
    }

    fn forward_inner(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
    ) -> Result<MxArray> {
        self.forward_inner_with_tape(x, mask, cache, position_ids, use_kernel, None)
    }

    /// Like [`DecoderLayer::forward`], but threads an eager-MTP tape sink into
    /// the GDN (`Linear`) sublayer so the verify forward can record the
    /// per-step recurrence inputs for the rollback replay. Full-attention
    /// layers ignore the sink (it stays `None` for their tape slot). When
    /// `tape_sink` is `None`, behavior is byte-identical to `forward`.
    pub(crate) fn forward_with_tape(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
    ) -> Result<MxArray> {
        self.forward_inner_with_tape(x, mask, cache, position_ids, use_kernel, tape_sink)
    }

    fn forward_inner_with_tape(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
    ) -> Result<MxArray> {
        self.forward_with_tape_budget(x, mask, cache, position_ids, use_kernel, tape_sink, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_tape_budget(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
        row_budget: Option<&MxArray>,
    ) -> Result<MxArray> {
        // Pre-norm + attention (always on full T for cache fidelity)
        let (_, resolved) = self.resolve_input_norm(LayerResidual::Hidden(x.clone()))?;
        let attn_out = match (&self.attn, resolved) {
            (
                AttentionType::Linear(gdn),
                ResolvedInputNorm::PrequantLinear { normed, x_i8, s_x },
            ) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward_with_tape_deferred_tree_prequant_sym8(
                    &normed,
                    (&x_i8, &s_x),
                    mask,
                    ac,
                    use_kernel,
                    tape_sink,
                    None,
                    None,
                )?
            }
            (AttentionType::Linear(gdn), ResolvedInputNorm::Normed(normed)) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward_with_tape(&normed, mask, ac, use_kernel, tape_sink)?
            }
            (AttentionType::Full(attn), ResolvedInputNorm::PrequantFull { x_i8, s_x }) => {
                let _ = tape_sink;
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward_tree_prequant_sym8(&x_i8, &s_x, mask, kvc, position_ids, None)?
            }
            (AttentionType::Full(attn), ResolvedInputNorm::Normed(normed)) => {
                // Full-attention layers do not record a GDN tape.
                let _ = tape_sink;
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward(&normed, mask, kvc, position_ids)?
            }
            _ => unreachable!("resolve_input_norm variant always matches self.attn"),
        };
        // Residual + post-attention norm + MLP (fused on CUDA Sym8 when eligible; MoE honors row_budget)
        self.finish_layer_with_budget(x, attn_out, row_budget)
    }

    /// Cross-layer `AddRMSNorm` variant of [`Self::forward_inner_with_tape`]:
    /// resolves `input` through `input_layernorm` (fusing the previous layer's
    /// MLP residual add into `input_layernorm` when `input` is
    /// [`LayerResidual::Pending`]), runs the mixer (with optional CUDA deferred
    /// GDN fold) and MLP, and returns `(completed_x, LayerResidual::Pending { h, delta })`.
    /// `completed_x` is the previous layer's exact post-MLP hidden state.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn forward_inner_with_tape_pending(
        &mut self,
        input: LayerResidual,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
        cuda_deferred: Option<Option<&super::layer_cache::CudaPendingGdn>>,
    ) -> Result<(MxArray, LayerResidual)> {
        self.forward_inner_with_tape_pending_tree(
            input,
            mask,
            cache,
            position_ids,
            use_kernel,
            tape_sink,
            cuda_deferred,
            None,
        )
    }

    /// Tree-verification variant of [`Self::forward_inner_with_tape_pending`]:
    /// threads optional [`super::verify_tree::TreeDeviceViews`] into both
    /// linear (GDN) and full-attention layers while preserving cross-layer
    /// `AddRMSNorm` fusion (`tree_len <= 16`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_inner_with_tape_pending_tree(
        &mut self,
        input: LayerResidual,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
        cuda_deferred: Option<Option<&super::layer_cache::CudaPendingGdn>>,
        tree: Option<&super::verify_tree::TreeDeviceViews>,
    ) -> Result<(MxArray, LayerResidual)> {
        let (x, resolved) = self.resolve_input_norm(input)?;
        let attn_out = match (&self.attn, resolved) {
            (
                AttentionType::Linear(gdn),
                ResolvedInputNorm::PrequantLinear { normed, x_i8, s_x },
            ) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward_with_tape_deferred_tree_prequant_sym8(
                    &normed,
                    (&x_i8, &s_x),
                    mask,
                    ac,
                    use_kernel,
                    tape_sink,
                    cuda_deferred,
                    tree,
                )?
            }
            (AttentionType::Linear(gdn), ResolvedInputNorm::Normed(normed)) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward_with_tape_deferred_tree(
                    &normed,
                    mask,
                    ac,
                    use_kernel,
                    tape_sink,
                    cuda_deferred,
                    tree,
                )?
            }
            (AttentionType::Full(attn), ResolvedInputNorm::PrequantFull { x_i8, s_x }) => {
                let _ = tape_sink;
                let _ = cuda_deferred;
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward_tree_prequant_sym8(&x_i8, &s_x, mask, kvc, position_ids, tree)?
            }
            (AttentionType::Full(attn), ResolvedInputNorm::Normed(normed)) => {
                let _ = tape_sink;
                let _ = cuda_deferred;
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward_tree(&normed, mask, kvc, position_ids, tree)?
            }
            _ => unreachable!("resolve_input_norm variant always matches self.attn"),
        };
        let (h, mlp_out) = self.post_attention_mlp_pending(&x, &attn_out)?;
        Ok((x, LayerResidual::Pending { h, delta: mlp_out }))
    }

    /// Forward pass with paged-or-flat dispatch for Qwen3.5.
    ///
    /// Branches on `kind`:
    /// * `Linear` — calls the existing flat `forward(...)` over the
    ///   per-layer `Qwen3_5LayerCache::Linear(ArraysCache)` slot. GDN
    ///   layers do NOT participate in cross-request prefix reuse on
    ///   this path (mirrors LFM2's conv-layer treatment and vLLM's
    ///   `MambaManager` default behavior).
    /// * `FullAttentionPaged { paged_idx }` — pre-norms via
    ///   `input_layernorm`, calls `Qwen3_5Attention::forward_paged` to
    ///   write K/V into the pool and compute attention, then runs the
    ///   shared MLP/residual tail.
    ///
    /// `flat_cache` is required for the `Linear` branch (it must point
    /// to the per-layer `Qwen3_5LayerCache::Linear` slot). For the
    /// `FullAttentionPaged` branch it is ignored — the adapter owns
    /// K/V.
    ///
    /// `use_kernel` is forwarded only to the flat `forward(...)` path.
    /// `position_ids` is forwarded to both: the flat `forward(...)` and the
    /// `FullAttentionPaged` branch's `forward_paged`, which applies 3-row
    /// M-RoPE over those positions for an image-bearing prefill and the
    /// scalar partial-RoPE when they are `None` (text-only).
    ///
    /// **Layer-kind / operator coherence**: an attention-type
    /// mismatch (e.g. `FullAttentionPaged` on a Linear operator)
    /// returns a descriptive error rather than panicking — matches
    /// the LFM2 dispatch pattern.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged_or_flat(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        first_logical_position: u32,
        pass: PagedPass,
        mask: Option<&MxArray>,
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        rope_position_offset: i32,
        mrope_cache: &mut Option<(MxArray, MxArray)>,
    ) -> Result<MxArray> {
        let int8_act = pass == PagedPass::Prefill
            && adapter.activations() == crate::transformer::Activations::Auto;
        match kind {
            Qwen3_5LayerKind::Linear => {
                // Linear (GDN) layer takes the flat path (opting into INT8
                // prefill projections when `Activations::Auto` on prefill).
                let _ = first_logical_position;
                let _ = rope_position_offset;
                if !matches!(self.attn, AttentionType::Linear(_)) {
                    return Err(Error::from_reason(
                        "Qwen3_5DecoderLayer::forward_paged_or_flat: kind=Linear applied to a \
                         FullAttention operator",
                    ));
                }
                if !int8_act {
                    return self.forward(x, mask, flat_cache, position_ids, use_kernel);
                }
                let AttentionType::Linear(gdn) = &self.attn else {
                    unreachable!("checked Linear above");
                };
                let normed = self.input_layernorm.forward(x)?;
                let ac = flat_cache.and_then(|c| c.as_arrays_cache_mut());
                let attn_out =
                    gdn.forward_with_tape_int8(&normed, mask, ac, use_kernel, None, true)?;
                let (h, normed) =
                    Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?;
                h.add(&self.mlp.forward_with_int8(&normed, true)?)
            }
            Qwen3_5LayerKind::FullAttentionPaged { paged_idx } => {
                let _ = flat_cache; // adapter owns K/V for paged layers
                let _ = use_kernel;
                let _ = mask; // paged path uses internal causal mask
                let attn = match &self.attn {
                    AttentionType::Full(a) => a,
                    AttentionType::Linear(_) => {
                        return Err(Error::from_reason(
                            "Qwen3_5DecoderLayer::forward_paged_or_flat: kind=FullAttentionPaged \
                             applied to a Linear (GDN) operator",
                        ));
                    }
                };
                // Pre-norm + paged attention. `position_ids` carries M-RoPE
                // positions for the image-bearing prefill; `None` keeps the
                // scalar-offset text path.
                let normed = self.input_layernorm.forward(x)?;
                let attn_out = attn.forward_paged(
                    &normed,
                    adapter,
                    paged_idx,
                    first_logical_position,
                    pass,
                    position_ids,
                    rope_position_offset,
                    mrope_cache,
                )?;
                // Residual + post-attention norm (fused when eligible).
                let (h, normed) =
                    Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?;
                h.add(&self.mlp.forward_with_int8(&normed, int8_act)?)
            }
        }
    }

    /// [`Self::forward_paged_or_flat`] for a paged prefill chunk
    /// (`PagedPass::Prefill`, no mask, kernels on): the mixer output
    /// projection adds the residual in its GEMM store when its shape takes
    /// the epilogue GEMM (else a separate add + post-attention norm), and
    /// the MLP runs its SwiGLU and residual add in the GEMM stores where they
    /// apply. The bits of [`Self::forward_paged_or_flat`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged_prefill(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        first_logical_position: u32,
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        rope_position_offset: i32,
        mrope_cache: &mut Option<(MxArray, MxArray)>,
    ) -> Result<MxArray> {
        /// The mixer's output: with the residual, or the delta to add.
        enum Mixed {
            Hidden(MxArray),
            Delta(MxArray),
        }
        let int8_act = adapter.activations() == crate::transformer::Activations::Auto;
        let rows = x.size()? as i64 / x.shape_at(x.ndim()?.saturating_sub(1))?.max(1);
        let normed = self.input_layernorm.forward(x)?;
        let mixed = match (kind, &mut self.attn) {
            (Qwen3_5LayerKind::Linear, AttentionType::Linear(gdn)) => {
                let ac = flat_cache.and_then(|c| c.as_arrays_cache_mut());
                if gdn.out_proj_store_route(rows)? {
                    Mixed::Hidden(gdn.forward_store_residual_int8(&normed, ac, true, x, int8_act)?)
                } else {
                    Mixed::Delta(
                        gdn.forward_with_tape_int8(&normed, None, ac, true, None, int8_act)?,
                    )
                }
            }
            (Qwen3_5LayerKind::FullAttentionPaged { paged_idx }, AttentionType::Full(attn)) => {
                if attn.o_proj_store_route(rows)? {
                    Mixed::Hidden(attn.forward_paged_store_residual(
                        &normed,
                        adapter,
                        paged_idx,
                        first_logical_position,
                        PagedPass::Prefill,
                        position_ids,
                        rope_position_offset,
                        mrope_cache,
                        x,
                    )?)
                } else {
                    Mixed::Delta(attn.forward_paged(
                        &normed,
                        adapter,
                        paged_idx,
                        first_logical_position,
                        PagedPass::Prefill,
                        position_ids,
                        rope_position_offset,
                        mrope_cache,
                    )?)
                }
            }
            (Qwen3_5LayerKind::Linear, AttentionType::Full(_)) => {
                return Err(Error::from_reason(
                    "Qwen3_5DecoderLayer::forward_paged_prefill: kind=Linear applied to a \
                     FullAttention operator",
                ));
            }
            (Qwen3_5LayerKind::FullAttentionPaged { .. }, AttentionType::Linear(_)) => {
                return Err(Error::from_reason(
                    "Qwen3_5DecoderLayer::forward_paged_prefill: kind=FullAttentionPaged \
                     applied to a Linear (GDN) operator",
                ));
            }
        };
        let (h, normed) = match mixed {
            Mixed::Hidden(h) => {
                let normed = self.post_attention_layernorm.forward(&h)?;
                (h, normed)
            }
            Mixed::Delta(attn_out) => {
                Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?
            }
        };
        self.mlp
            .forward_store_residual_with_int8(&normed, &h, int8_act)
    }

    /// A text prefill chunk through a full-attention layer whose output no
    /// one reads: input norm and the attention's K/V store, no output.
    pub(crate) fn store_paged_kv(
        &self,
        x: &MxArray,
        paged_idx: u32,
        adapter: &mut PagedKVCacheAdapter,
        first_logical_position: u32,
        rope_position_offset: i32,
    ) -> Result<()> {
        let AttentionType::Full(attn) = &self.attn else {
            return Err(Error::from_reason(
                "Qwen3_5DecoderLayer::store_paged_kv: a Linear (GDN) operator stores no K/V",
            ));
        };
        let normed = self.input_layernorm.forward(x)?;
        attn.store_paged_kv(
            &normed,
            adapter,
            paged_idx,
            first_logical_position,
            rope_position_offset,
        )
    }

    /// One uniform decode forward over independent recurrent-state rows.
    ///
    /// When `preserve_singleton_projection_graphs` is `true` (used by row-exact
    /// MoE loads and MoE scheduled MTP verify), every projection, GDN layer,
    /// and MLP/MoE block runs on each row's own graph.
    pub(crate) fn forward_paged_batched(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        rows: &[(SeqId, u32)],
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        preserve_singleton_projection_graphs: bool,
    ) -> Result<MxArray> {
        self.forward_paged_batched_with_tape(
            x,
            kind,
            adapter,
            rows,
            flat_cache,
            preserve_singleton_projection_graphs,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged_batched_with_tape(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        rows: &[(SeqId, u32)],
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        preserve_singleton_projection_graphs: bool,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
    ) -> Result<MxArray> {
        match kind {
            Qwen3_5LayerKind::Linear => {
                if !matches!(self.attn, AttentionType::Linear(_)) {
                    return Err(Error::from_reason(
                        "Qwen3_5DecoderLayer::forward_paged_batched: Linear kind/operator mismatch",
                    ));
                }
                if !preserve_singleton_projection_graphs {
                    return self.forward_with_tape(x, None, flat_cache, None, true, tape_sink);
                }
                let normed = self.input_layernorm.forward(x)?;
                let attn_out = match &mut self.attn {
                    AttentionType::Linear(gdn) => {
                        let cache = flat_cache
                            .and_then(|cache| cache.as_arrays_cache_mut())
                            .ok_or_else(|| {
                                Error::from_reason(
                                    "Qwen3.5 batched GDN layer has no recurrent cache",
                                )
                            })?;
                        let capture = tape_sink.is_some();
                        let mut tapes = Vec::new();
                        let (output, next_cache) = super::arrays_cache::forward_rows_independently(
                            &normed,
                            cache,
                            |row, row_cache| {
                                let mut tape = None;
                                let output = gdn.forward_with_tape(
                                    row,
                                    None,
                                    Some(row_cache),
                                    true,
                                    capture.then_some(&mut tape),
                                )?;
                                if capture {
                                    tapes.push(tape.ok_or_else(|| {
                                        Error::from_reason(
                                            "GDN verify did not record an owner tape",
                                        )
                                    })?);
                                }
                                Ok(output)
                            },
                        )?;
                        if let Some(sink) = tape_sink {
                            *sink = Some(super::gated_delta_net::GdnLayerTape::stack_rows(&tapes)?);
                        }
                        *cache = next_cache;
                        output
                    }
                    AttentionType::Full(_) => {
                        return Err(Error::from_reason(
                            "Qwen3_5DecoderLayer::forward_paged_batched: Linear operator disappeared",
                        ));
                    }
                };
                let (h, normed) =
                    Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?;
                h.add(&self.forward_mlp_decode_rows(&normed)?)
            }
            Qwen3_5LayerKind::FullAttentionPaged { paged_idx } => {
                if let Some(tape) = tape_sink {
                    *tape = None;
                }
                let attn = match &self.attn {
                    AttentionType::Full(attn) => attn,
                    AttentionType::Linear(_) => {
                        return Err(Error::from_reason(
                            "Qwen3_5DecoderLayer::forward_paged_batched: FullAttention kind/operator mismatch",
                        ));
                    }
                };
                let normed = self.input_layernorm.forward(x)?;
                let attn_out = attn.forward_paged_batched(
                    &normed,
                    adapter,
                    paged_idx,
                    rows,
                    preserve_singleton_projection_graphs,
                )?;
                let (h, normed) =
                    Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?;
                let mlp_out = if preserve_singleton_projection_graphs {
                    self.forward_mlp_decode_rows(&normed)?
                } else {
                    self.mlp.forward(&normed)?
                };
                h.add(&mlp_out)
            }
        }
    }

    /// Tape-recording variant of [`Self::forward_paged_or_flat`] for the eager
    /// paged MTP verify forward.
    ///
    /// Identical to `forward_paged_or_flat` except the `Linear` (GDN) branch
    /// records a per-layer [`GdnLayerTape`] into `tape_sink` (the rollback
    /// replay keystone), exactly like the flat
    /// [`Self::forward_with_tape`]. The `FullAttentionPaged` branch never
    /// records a tape (full-attention K/V lives in the paged pool and is
    /// rewound by `PagedKVCacheAdapter::rollback_last_tokens`); it drains
    /// `tape_sink` to `None` so the absolute-layer-indexed tape stays `None`
    /// for full-attention slots, matching `forward_inner_with_tape`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged_or_flat_with_tape(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        first_logical_position: u32,
        pass: PagedPass,
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
        rope_position_offset: i32,
        row_budget: Option<&MxArray>,
    ) -> Result<MxArray> {
        match kind {
            Qwen3_5LayerKind::Linear => {
                let _ = adapter;
                let _ = first_logical_position;
                let _ = pass;
                let _ = rope_position_offset;
                if !matches!(self.attn, AttentionType::Linear(_)) {
                    return Err(Error::from_reason(
                        "Qwen3_5DecoderLayer::forward_paged_or_flat_with_tape: kind=Linear applied \
                         to a FullAttention operator",
                    ));
                }
                self.forward_with_tape_budget(
                    x, None, flat_cache, None, true, tape_sink, row_budget,
                )
            }
            Qwen3_5LayerKind::FullAttentionPaged { paged_idx } => {
                let _ = flat_cache; // adapter owns K/V for paged layers
                // Full-attention layers do not record a GDN tape.
                if let Some(slot) = tape_sink {
                    *slot = None;
                }
                let attn = match &self.attn {
                    AttentionType::Full(a) => a,
                    AttentionType::Linear(_) => {
                        return Err(Error::from_reason(
                            "Qwen3_5DecoderLayer::forward_paged_or_flat_with_tape: \
                             kind=FullAttentionPaged applied to a Linear (GDN) operator",
                        ));
                    }
                };
                let normed = self.input_layernorm.forward(x)?;
                // MTP tape forwards always carry M-RoPE positions as `None`
                // (the K+1 verify ids are re-embedded, not image features), so
                // RoPE takes the scalar-offset path. `rope_position_offset`
                // still carries any cross-turn delta: a text turn that
                // warm-continues an image prefill runs paged MTP and must
                // rotate at the compressed position, not the physical slot.
                let attn_out = attn.forward_paged(
                    &normed,
                    adapter,
                    paged_idx,
                    first_logical_position,
                    pass,
                    None,
                    rope_position_offset,
                    &mut None,
                )?;
                self.finish_layer_with_budget(x, attn_out, row_budget)
            }
        }
    }

    /// [`Self::forward_paged_or_flat_with_tape`] for a DFlash2 verify
    /// (`PagedPass::Verify`): a mixer output projection with planes adds the
    /// residual in its `_r` store and the post-attention norm runs on that
    /// hidden state alone; the MLP runs [`MLPVariant::forward_verify`] (or
    /// budgeted MoE forward). A projection without planes runs as in
    /// [`Self::forward_paged_or_flat_with_tape`], with its bits.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_verify_with_tape(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        first_logical_position: u32,
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
        rope_position_offset: i32,
        row_budget: Option<&MxArray>,
    ) -> Result<MxArray> {
        /// The mixer's output: with the residual, or the delta to add.
        enum Mixed {
            Hidden(MxArray),
            Delta(MxArray),
        }
        let rows = x.size()? as i64 / x.shape_at(x.ndim()?.saturating_sub(1))?.max(1);
        let mixed = match (kind, &mut self.attn) {
            (Qwen3_5LayerKind::Linear, AttentionType::Linear(gdn)) => {
                let normed = self.input_layernorm.forward(x)?;
                let ac = flat_cache.and_then(|c| c.as_arrays_cache_mut());
                if gdn.out_proj_verify_route(rows) {
                    Mixed::Hidden(gdn.forward_verify_residual_with_tape(&normed, ac, tape_sink, x)?)
                } else {
                    Mixed::Delta(gdn.forward_with_tape(&normed, None, ac, true, tape_sink)?)
                }
            }
            (Qwen3_5LayerKind::FullAttentionPaged { paged_idx }, AttentionType::Full(attn)) => {
                // Full-attention layers do not record a GDN tape.
                if let Some(slot) = tape_sink {
                    *slot = None;
                }
                let normed = self.input_layernorm.forward(x)?;
                if attn.o_proj_verify_route(rows) {
                    Mixed::Hidden(attn.forward_paged_verify_residual(
                        &normed,
                        adapter,
                        paged_idx,
                        first_logical_position,
                        rope_position_offset,
                        x,
                    )?)
                } else {
                    Mixed::Delta(attn.forward_paged(
                        &normed,
                        adapter,
                        paged_idx,
                        first_logical_position,
                        PagedPass::Verify,
                        None,
                        rope_position_offset,
                        &mut None,
                    )?)
                }
            }
            (Qwen3_5LayerKind::Linear, AttentionType::Full(_)) => {
                return Err(Error::from_reason(
                    "Qwen3_5DecoderLayer::forward_verify_with_tape: kind=Linear applied to a \
                     FullAttention operator",
                ));
            }
            (Qwen3_5LayerKind::FullAttentionPaged { .. }, AttentionType::Linear(_)) => {
                return Err(Error::from_reason(
                    "Qwen3_5DecoderLayer::forward_verify_with_tape: kind=FullAttentionPaged \
                     applied to a Linear (GDN) operator",
                ));
            }
        };
        let (h, normed) = match mixed {
            Mixed::Hidden(h) => {
                let normed = self.post_attention_layernorm.forward(&h)?;
                (h, normed)
            }
            Mixed::Delta(attn_out) => {
                Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?
            }
        };
        self.mlp.forward_verify_with_budget(&normed, &h, row_budget)
    }

    /// One layer of a verify wave, `x` `[N, T, H]` (lane `i` = row `i`):
    /// norms, residuals and SwiGLU stacked, the mixer through
    /// [`gdn_wave_layer`] (blob row `row`) or
    /// [`Qwen3_5Attention::forward_paged_wave_residual`], the MLP through
    /// [`MLPVariant::forward_verify_lanes`]. A mixer output projection with
    /// planes adds the residual in its `_r` store over the lanes and the
    /// post-attention norm runs on that hidden state alone. Each lane's rows
    /// carry the bits [`Self::forward_verify_with_tape`] gives that lane
    /// alone.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_wave(
        &self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        layer_idx: usize,
        row: usize,
        adapter: &mut PagedKVCacheAdapter,
        attn_lanes: &[PagedWaveLane],
        gdn_lanes: &mut [GdnWaveCells<'_>],
    ) -> Result<MxArray> {
        let normed = self.input_layernorm.forward(x)?;
        let lanes = x.shape_at(0)?;
        let (residual, mixed) = match (kind, &self.attn) {
            (Qwen3_5LayerKind::Linear, AttentionType::Linear(gdn)) => {
                let residual = gdn.out_proj_verify_lanes_route(lanes).then_some(x);
                let out = gdn_wave_layer(gdn, &normed, layer_idx, row, gdn_lanes, residual)?;
                (residual, out)
            }
            (Qwen3_5LayerKind::FullAttentionPaged { paged_idx }, AttentionType::Full(attn)) => {
                let residual = attn.o_proj_verify_lanes_route(lanes).then_some(x);
                let out = attn.forward_paged_wave_residual(
                    &normed, adapter, paged_idx, attn_lanes, residual,
                )?;
                (residual, out)
            }
            _ => {
                return Err(Error::from_reason(format!(
                    "Qwen3_5DecoderLayer::forward_wave: layer {layer_idx} kind/operator mismatch"
                )));
            }
        };
        let (h, normed) = match residual {
            Some(_) => {
                let normed = self.post_attention_layernorm.forward(&mixed)?;
                (mixed, normed)
            }
            None => Self::add_residual_norm(&self.post_attention_layernorm, x, &mixed)?,
        };
        self.mlp.forward_verify_lanes(&normed, &h)
    }

    // ========== Weight accessors ==========

    pub fn set_input_layernorm_weight(&mut self, w: &MxArray, compute_dtype: DType) -> Result<()> {
        self.input_layernorm
            .set_weight(&super::sidecar_to_compute_dtype(w, compute_dtype)?)
    }

    pub fn set_post_attention_layernorm_weight(
        &mut self,
        w: &MxArray,
        compute_dtype: DType,
    ) -> Result<()> {
        self.post_attention_layernorm
            .set_weight(&super::sidecar_to_compute_dtype(w, compute_dtype)?)
    }

    /// Set `input_layernorm` preserving the raw tensor dtype (MoE parity).
    pub fn set_input_layernorm_weight_raw(&mut self, w: &MxArray) -> Result<()> {
        self.input_layernorm.set_weight(w)
    }

    /// Set `post_attention_layernorm` preserving the raw tensor dtype (MoE parity).
    pub fn set_post_attention_layernorm_weight_raw(&mut self, w: &MxArray) -> Result<()> {
        self.post_attention_layernorm.set_weight(w)
    }

    /// Enable Float32 RMSNorm evaluation on `input_layernorm`, `post_attention_layernorm`,
    /// and (for full-attention layers) `q_norm` / `k_norm`.
    pub(crate) fn set_precise_f32_norms(&mut self) {
        self.input_layernorm.set_precise_f32();
        self.post_attention_layernorm.set_precise_f32();
        if let AttentionType::Full(attn) = &mut self.attn {
            attn.set_precise_f32_norms();
        }
    }

    // ========== Weight getters (for training parameter extraction) ==========

    pub fn get_input_layernorm_weight(&self) -> MxArray {
        self.input_layernorm.get_weight()
    }

    pub fn get_post_attention_layernorm_weight(&self) -> MxArray {
        self.post_attention_layernorm.get_weight()
    }

    /// Replace the dense MLP with a quantized version. Attempts the packed
    /// gate|up row-merge (`finalize_gate_up`); incompatible pairs simply stay
    /// unmerged, but an internal merge error fails the load loud.
    pub fn set_quantized_dense_mlp(
        &mut self,
        gate_proj: QuantizedLinear,
        up_proj: QuantizedLinear,
        down_proj: QuantizedLinear,
    ) -> Result<()> {
        self.set_quantized_dense_mlp_with_merge(gate_proj, up_proj, down_proj, true)
    }

    /// Like [`Self::set_quantized_dense_mlp`], with `merge_gate_up=false` to
    /// skip the GPU `concat_rows` merge when preserving host-backed overflow
    /// projections on discrete CUDA GPUs.
    pub(crate) fn set_quantized_dense_mlp_with_merge(
        &mut self,
        gate_proj: QuantizedLinear,
        up_proj: QuantizedLinear,
        down_proj: QuantizedLinear,
        merge_gate_up: bool,
    ) -> Result<()> {
        self.mlp = MLPType::Dense(MLPVariant::Quantized {
            gate_proj,
            up_proj,
            down_proj,
            gate_up: None,
        });
        if merge_gate_up {
            self.mlp.finalize_gate_up()?;
        }
        Ok(())
    }

    /// Whether any main-model projection in this decoder layer is quantized.
    pub fn is_quantized(&self) -> bool {
        let attention = match &self.attn {
            AttentionType::Linear(gdn) => gdn.is_quantized(),
            AttentionType::Full(attn) => attn.is_quantized(),
        };
        attention || self.mlp.is_quantized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_layer_kinds_dense_qwen35_default_interval() {
        // Default `full_attention_interval = 4` makes layer 3, 7, 11,
        // ... full-attention; everything else is GDN.
        let n = 12usize;
        let kinds = compute_layer_kinds(n, |i| (i + 1) % 4 != 0);

        let mut paged_idx = 0u32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let expected_full = (i + 1).is_multiple_of(4);
            match kinds[i] {
                Qwen3_5LayerKind::Linear => {
                    assert!(!expected_full, "layer {i} should be Linear");
                }
                Qwen3_5LayerKind::FullAttentionPaged { paged_idx: pidx } => {
                    assert!(expected_full, "layer {i} should be FullAttentionPaged");
                    assert_eq!(pidx, paged_idx, "paged_idx mismatch at layer {i}");
                    paged_idx += 1;
                }
            }
        }
        // 12/4 = 3 full attention layers expected
        assert_eq!(paged_idx, 3);
    }

    #[test]
    fn test_compute_layer_kinds_all_linear() {
        // full_attention_interval <= 0 → all linear
        let n = 5usize;
        let kinds = compute_layer_kinds(n, |_| true);
        for k in &kinds {
            assert!(matches!(k, Qwen3_5LayerKind::Linear));
        }
    }

    #[test]
    fn test_compute_layer_kinds_all_full_attention() {
        // is_linear always false → all paged
        let n = 4usize;
        let kinds = compute_layer_kinds(n, |_| false);
        for (i, k) in kinds.iter().enumerate() {
            match k {
                Qwen3_5LayerKind::FullAttentionPaged { paged_idx } => {
                    assert_eq!(*paged_idx, i as u32);
                }
                _ => panic!("expected FullAttentionPaged at {i}"),
            }
        }
    }

    #[test]
    fn forward_batched_flat_routes_linear_and_full_masks() {
        let cfg = Qwen3_5Config {
            rope_scaling: None,
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
            vocab_size: 32,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 128,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            partial_rotary_factor: 0.5,
            rope_theta: 10_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            n_mtp_layers: 0,
        };
        let mut linear_layer = DecoderLayer::new(&cfg, 0).unwrap();
        let mut full_layer = DecoderLayer::new(&cfg, 1).unwrap();
        assert!(linear_layer.is_linear());
        assert!(!full_layer.is_linear());

        // set_rope_mode is a no-op on linear layers and updates full layers.
        linear_layer.set_rope_mode(RopeMode::Temporal);
        full_layer.set_rope_mode(RopeMode::Temporal);

        let x_data: Vec<f32> = (0..(2 * 4 * 64))
            .map(|i| ((i as f32) * 0.031).sin() * 0.2)
            .collect();
        let x = MxArray::from_float32(&x_data, &[2, 4, 64]).unwrap();
        let linear_mask = MxArray::from_int32(&[1, 1, 1, 0, 1, 1, 1, 1], &[2, 4])
            .unwrap()
            .greater(&MxArray::scalar_int(0).unwrap())
            .unwrap();

        let mut linear_cache = Qwen3_5LayerCache::new_linear();
        let mut full_cache = Qwen3_5LayerCache::new_full_attention();
        let out_lin = linear_layer
            .forward_batched_flat(
                &x,
                None,
                Some(&linear_mask),
                Some(&[3, 4]),
                Some(&mut linear_cache),
                None,
            )
            .unwrap();
        let out_full = full_layer
            .forward_batched_flat(&out_lin, None, None, None, Some(&mut full_cache), None)
            .unwrap();
        assert_eq!(out_full.shape().unwrap().as_ref(), &[2, 4, 64]);
        assert_eq!(full_cache.as_kv_cache_mut().unwrap().get_offset(), 4);
    }
}
