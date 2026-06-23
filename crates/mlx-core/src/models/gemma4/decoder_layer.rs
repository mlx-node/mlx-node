use crate::array::MxArray;
use crate::nn::activations::Activations;
use crate::nn::{Linear, RMSNorm};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use napi::bindgen_prelude::*;

use super::attention::Gemma4Attention;
use super::config::Gemma4Config;
use super::diagnostic;
use super::layer_cache::Gemma4LayerCache;
use super::mlp::GemmaMLP;
use super::moe::{Gemma4MoE, Gemma4Router};
use super::quantized_linear::{Gemma4MLPVariant, QuantizedLinear, QuantizedSwitchLinear};

/// Per-layer routing kind for Gemma4's paged dispatch.
///
/// Mirrors `Lfm2LayerKind` but reflects Gemma4's hybrid attention layout
/// (sliding-window + global) plus optional KV-shared layers (last
/// `num_kv_shared_layers` reuse K/V from an earlier non-shared anchor of
/// the same attention type).
///
/// Indexing rules:
/// * `paged_idx` is the GLOBAL-LAYER ORDINAL into the paged adapter's
///   `LayerKVPool` (NOT the absolute decoder index). The pool is sized
///   for physical non-shared `full_attention` layers in `config.layer_types`,
///   so global owner layers are numbered 0..N_global in their original order.
///   KV-shared global layers reuse their anchor's ordinal.
/// * `anchor_layer_idx` (in `SharedOnSliding`) is the ABSOLUTE decoder
///   index of the anchor whose flat `Gemma4LayerCache::Sliding` slot
///   feeds the shared layer's K/V via `take_stashed_kv`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Gemma4LayerKind {
    /// Sliding (local) attention layer — stays on the existing flat
    /// `RotatingKVCache` path even when the paged adapter is enabled.
    /// Sliding's window-trimmed semantics don't map cleanly onto the
    /// adapter's contiguous-block layout.
    Sliding,
    /// Global (full) attention layer routed through the paged adapter.
    /// `paged_idx` is the global-layer ordinal — see the type rustdoc.
    GlobalPaged { paged_idx: u32 },
    /// KV-shared layer whose anchor is a global layer that goes through
    /// the paged adapter. The shared layer reads K/V via
    /// `adapter.read_kv_range(anchor_paged_idx, 0, total_ctx)`.
    SharedOnGlobal { anchor_paged_idx: u32 },
    /// KV-shared layer whose anchor is a sliding (flat-cache) layer.
    /// The shared layer pulls K/V from `caches[anchor_layer_idx]`'s
    /// stash (mirrors the existing flat-path KV-sharing flow).
    SharedOnSliding { anchor_layer_idx: u32 },
}

/// Inputs threaded into `forward_paged_or_flat` for KV-shared layers.
///
/// `cache_offset` is the RoPE offset for the queries — for
/// `SharedOnGlobal` it equals the anchor's logical position when the
/// anchor processed the same chunk; for `SharedOnSliding` it is the
/// anchor's pre-update offset (the flat-path's
/// `caches[anchor_layer_idx].get_offset() - seq_len`).
///
/// `total_ctx` is only consumed by `SharedOnGlobal` (the K/V token count
/// to read from the anchor's paged slot). `keys` / `values` are only
/// consumed by `SharedOnSliding` (anchor stash).
pub(crate) struct SharedKvInputs<'a> {
    pub(crate) cache_offset: i32,
    pub(crate) total_ctx: u32,
    pub(crate) keys: Option<&'a MxArray>,
    pub(crate) values: Option<&'a MxArray>,
}

/// A single decoder layer in the Gemma4 model.
///
/// Gemma4 uses 4 norms per layer (vs 2 in Qwen3.5):
///   - input_layernorm (pre-attention)
///   - post_attention_layernorm (post-attention, before residual)
///   - pre_feedforward_layernorm (pre-FFN)
///   - post_feedforward_layernorm (post-FFN, before residual)
///
/// Forward:
///   r = self_attn(input_layernorm(x))
///   h = x + post_attention_layernorm(r)
///   r = mlp(pre_feedforward_layernorm(h))
///   out = h + post_feedforward_layernorm(r)
///   out = out * layer_scalar
///   out = out + ple_projection(gelu(ple_gate(out)) * per_layer_input)  (if PLE)
pub struct Gemma4DecoderLayer {
    pub self_attn: Gemma4Attention,
    pub mlp: Gemma4MLPVariant,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    pre_feedforward_layernorm: RMSNorm,
    post_feedforward_layernorm: RMSNorm,

    /// Per-layer scalar multiplier. Applied to the layer output.
    /// Values range from ~0.018 to ~0.878 depending on layer depth.
    layer_scalar: Option<MxArray>,

    // PLE (Per-Layer Embeddings) per-layer components.
    // Only present when config.per_layer_input_embeds is true.
    per_layer_input_gate: Option<Linear>,
    per_layer_projection: Option<Linear>,
    post_per_layer_input_norm: Option<RMSNorm>,

    // MoE components (None for dense-only models like E2B).
    // When present, MoE runs in parallel with the dense MLP:
    //   hidden_states_1 = post_feedforward_layernorm_1(mlp_out)
    //   hidden_states_2 = post_feedforward_layernorm_2(moe(pre_feedforward_layernorm_2(residual), router(residual)))
    //   combined = hidden_states_1 + hidden_states_2
    router: Option<Gemma4Router>,
    moe: Option<Gemma4MoE>,
    pre_feedforward_layernorm_2: Option<RMSNorm>,
    post_feedforward_layernorm_1: Option<RMSNorm>,
    post_feedforward_layernorm_2: Option<RMSNorm>,
}

impl Gemma4DecoderLayer {
    pub fn new(config: &Gemma4Config, layer_idx: usize) -> Result<Self> {
        let self_attn = Gemma4Attention::new(config, layer_idx)?;

        let intermediate = config.effective_intermediate_size(layer_idx);
        let mlp = Gemma4MLPVariant::Standard(GemmaMLP::new(
            config.hidden_size as u32,
            intermediate as u32,
        )?);

        let eps = Some(config.rms_norm_eps);
        let h = config.hidden_size as u32;
        let input_layernorm = RMSNorm::new(h, eps)?;
        let post_attention_layernorm = RMSNorm::new(h, eps)?;
        let pre_feedforward_layernorm = RMSNorm::new(h, eps)?;
        let post_feedforward_layernorm = RMSNorm::new(h, eps)?;

        // PLE per-layer components (only when per_layer_input_embeds is true)
        let ple_dim = config.ple_dim();
        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) =
            if config.per_layer_input_embeds && ple_dim > 0 {
                (
                    Some(Linear::new(h, ple_dim as u32, Some(false))?),
                    Some(Linear::new(ple_dim as u32, h, Some(false))?),
                    Some(RMSNorm::new(h, eps)?),
                )
            } else {
                (None, None, None)
            };

        // MoE components (only when enable_moe_block is true)
        let (
            router,
            moe,
            pre_feedforward_layernorm_2,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
        ) = if config.enable_moe_block {
            let num_experts = config.num_experts.unwrap_or(128) as u32;
            let top_k = config.top_k_experts.unwrap_or(8) as u32;
            let moe_inter = config.moe_intermediate_size.unwrap_or(704) as u32;
            (
                Some(Gemma4Router::new(
                    h,
                    num_experts,
                    top_k,
                    config.rms_norm_eps,
                )?),
                Some(Gemma4MoE::new(h, num_experts, top_k, moe_inter)?),
                Some(RMSNorm::new(h, eps)?),
                Some(RMSNorm::new(h, eps)?),
                Some(RMSNorm::new(h, eps)?),
            )
        } else {
            (None, None, None, None, None)
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            layer_scalar: None,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            router,
            moe,
            pre_feedforward_layernorm_2,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
        })
    }

    /// Forward pass with 4-norm residual pattern.
    ///
    /// `per_layer_input`: optional PLE input for this layer, shape [B, T, ple_dim].
    /// Computed at the model level and sliced per-layer before calling this method.
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Gemma4LayerCache>,
        per_layer_input: Option<&MxArray>,
        needs_stash: bool,
    ) -> Result<MxArray> {
        // Pre-norm + attention
        let normed = self.input_layernorm.forward(x)?;
        diagnostic::dump_norm_current_layer("hin", &normed, None);
        let attn_out = self.self_attn.forward(&normed, mask, cache, needs_stash)?;
        diagnostic::dump_norm_current_layer("attn", &attn_out, None);

        let out = self.apply_ffn_ple_scalar(x, &attn_out, per_layer_input)?;
        diagnostic::dump_norm_current_layer("hout", &out, None);
        Ok(out)
    }

    /// Forward pass for KV-shared layers.
    ///
    /// Same 4-norm residual structure as `forward`, but uses `forward_shared` on
    /// attention (only computes Q; K/V come from the anchor layer's cache).
    ///
    /// # Arguments
    /// * `x` - Input hidden states [B, T, hidden_size]
    /// * `mask` - Attention mask (may be adjusted for anchor's sequence length)
    /// * `shared_keys` - [B, H_kv, T_anchor, D] from anchor layer's cache
    /// * `shared_values` - [B, H_kv, T_anchor, D] from anchor layer's cache
    /// * `cache_offset` - RoPE offset for queries (from anchor cache)
    /// * `per_layer_input` - Optional PLE input for this layer
    pub fn forward_shared(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        shared_keys: &MxArray,
        shared_values: &MxArray,
        cache_offset: i32,
        per_layer_input: Option<&MxArray>,
    ) -> Result<MxArray> {
        // Pre-norm + shared attention (Q-only, reuses anchor K/V)
        let normed = self.input_layernorm.forward(x)?;
        diagnostic::dump_norm_current_layer("hin", &normed, None);
        let extra = format!(
            ",branch=shared_flat,cache_offset={cache_offset},shared_keys_T={}",
            shared_keys.shape_at(2).unwrap_or(-1)
        );
        diagnostic::dump_norm_current_layer("shared_keys", shared_keys, Some(&extra));
        let attn_out = self.self_attn.forward_shared(
            &normed,
            mask,
            shared_keys,
            shared_values,
            cache_offset,
        )?;
        diagnostic::dump_norm_current_layer("attn", &attn_out, None);

        let out = self.apply_ffn_ple_scalar(x, &attn_out, per_layer_input)?;
        diagnostic::dump_norm_current_layer("hout", &out, None);
        Ok(out)
    }

    /// Forward pass with paged-or-flat dispatch.
    ///
    /// Branches on `kind`:
    /// * `Sliding` — calls the existing flat `forward(...)` over the
    ///   per-layer `Gemma4LayerCache::Sliding` slot. Sliding layers
    ///   stay on the flat `RotatingKVCache` path even when the paged
    ///   adapter is enabled (window-trimmed semantics don't fit the
    ///   pool layout).
    /// * `GlobalPaged { paged_idx }` — pre-norms the input via
    ///   `input_layernorm`, calls `self_attn.forward_paged` to write
    ///   K/V into the pool and compute attention, then runs the
    ///   shared `apply_ffn_ple_scalar` tail.
    /// * `SharedOnGlobal { anchor_paged_idx }` — Q-only forward with
    ///   K/V read directly from the anchor's paged slot via
    ///   `adapter.read_kv_range(...)`. The anchor must have already
    ///   processed the same chunk in this turn so its slot is up to
    ///   date. The caller carries `anchor_total_ctx` (= number of K/V
    ///   tokens in the anchor's slot) and `anchor_cache_offset`
    ///   (= RoPE offset matching the anchor's queries).
    /// * `SharedOnSliding { anchor_layer_idx }` — Q-only forward with
    ///   K/V read from the anchor's flat-cache stash (mirrors the flat
    ///   path's `forward_shared`). The caller pulls `(shared_keys,
    ///   shared_values, cache_offset)` from
    ///   `caches[anchor_layer_idx].take_stashed_kv()` ahead of time
    ///   and passes them in via the `shared_kv` parameter.
    ///
    /// `flat_cache` is required for the `Sliding` branch (it's the
    /// per-layer flat cache slot). For `GlobalPaged` and the shared
    /// kinds it is ignored (the adapter / anchor stash own K/V).
    ///
    /// `needs_stash` only matters for `Sliding` (mirrors the flat
    /// `forward(...)` call signature).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged_or_flat(
        &self,
        x: &MxArray,
        kind: Gemma4LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
        mask: Option<&MxArray>,
        flat_cache: Option<&mut Gemma4LayerCache>,
        per_layer_input: Option<&MxArray>,
        needs_stash: bool,
        shared_kv: Option<SharedKvInputs<'_>>,
    ) -> Result<MxArray> {
        match kind {
            Gemma4LayerKind::Sliding => {
                // Flat path: existing forward(...).
                let _ = shared_kv;
                self.forward(x, mask, flat_cache, per_layer_input, needs_stash)
            }
            Gemma4LayerKind::GlobalPaged { paged_idx } => {
                let _ = flat_cache; // adapter owns K/V for global layers
                let _ = needs_stash;
                let _ = shared_kv;
                // `mask` is normally None for global layers (the paged path uses
                // an internal causal mask). During a unified-vision prefill the
                // caller passes the explicit bidirectional-overlay keep-mask,
                // which `forward_paged` applies in the fresh-prefill branch.
                let normed = self.input_layernorm.forward(x)?;
                diagnostic::dump_norm_current_layer("hin", &normed, None);
                let attn_out = self.self_attn.forward_paged(
                    &normed,
                    adapter,
                    paged_idx,
                    first_logical_position,
                    cached_prefix_len,
                    is_prefill,
                    mask,
                )?;
                diagnostic::dump_norm_current_layer(
                    "attn",
                    &attn_out,
                    Some(&format!(
                        ",branch=global_paged,paged_idx={paged_idx},first_pos={first_logical_position}"
                    )),
                );
                let out = self.apply_ffn_ple_scalar(x, &attn_out, per_layer_input)?;
                diagnostic::dump_norm_current_layer("hout", &out, None);
                Ok(out)
            }
            Gemma4LayerKind::SharedOnGlobal { anchor_paged_idx } => {
                let _ = flat_cache;
                let _ = mask;
                let _ = needs_stash;
                let inputs = shared_kv.ok_or_else(|| {
                    Error::from_reason(
                        "forward_paged_or_flat: SharedOnGlobal requires shared_kv with \
                         total_ctx + cache_offset",
                    )
                })?;
                let normed = self.input_layernorm.forward(x)?;
                diagnostic::dump_norm_current_layer("hin", &normed, None);
                // Pre-dump the donor's K from the pool so we can compare
                // it against the flat path's `shared_keys` stash. This
                // requires reading the pool twice (the inner SDPA call
                // also re-reads), but the cost is gated on
                // MLX_DEBUG_GEMMA4_DUMP=1 and only fires when the env
                // var is set.
                if diagnostic::dump_enabled()
                    && let Ok((peeked_k, _peeked_v)) =
                        adapter.read_kv_range(anchor_paged_idx, 0, inputs.total_ctx)
                {
                    let extra = format!(
                        ",branch=shared_paged,anchor_paged_idx={anchor_paged_idx},cache_offset={},total_ctx={}",
                        inputs.cache_offset, inputs.total_ctx,
                    );
                    diagnostic::dump_norm_current_layer("shared_keys", &peeked_k, Some(&extra));
                }
                let extra = format!(
                    ",branch=shared_paged,anchor_paged_idx={anchor_paged_idx},cache_offset={},total_ctx={}",
                    inputs.cache_offset, inputs.total_ctx,
                );
                let attn_out = self.self_attn.forward_paged_shared(
                    &normed,
                    adapter,
                    anchor_paged_idx,
                    inputs.cache_offset,
                    inputs.total_ctx,
                    is_prefill,
                )?;
                diagnostic::dump_norm_current_layer("attn", &attn_out, Some(&extra));
                let out = self.apply_ffn_ple_scalar(x, &attn_out, per_layer_input)?;
                diagnostic::dump_norm_current_layer("hout", &out, None);
                Ok(out)
            }
            Gemma4LayerKind::SharedOnSliding { .. } => {
                let _ = flat_cache;
                let _ = needs_stash;
                let _ = adapter;
                let _ = first_logical_position;
                let _ = cached_prefix_len;
                let _ = is_prefill;
                let inputs = shared_kv.ok_or_else(|| {
                    Error::from_reason(
                        "forward_paged_or_flat: SharedOnSliding requires shared_kv with \
                         (keys, values, cache_offset)",
                    )
                })?;
                let keys = inputs
                    .keys
                    .ok_or_else(|| Error::from_reason("SharedOnSliding requires shared_kv.keys"))?;
                let values = inputs.values.ok_or_else(|| {
                    Error::from_reason("SharedOnSliding requires shared_kv.values")
                })?;
                // Delegate to flat `forward_shared` (which already dumps).
                self.forward_shared(x, mask, keys, values, inputs.cache_offset, per_layer_input)
            }
        }
    }

    /// Shared tail after attention: post-attention norm, FFN+MoE, PLE, layer scalar.
    fn apply_ffn_ple_scalar(
        &self,
        x: &MxArray,
        attn_out: &MxArray,
        per_layer_input: Option<&MxArray>,
    ) -> Result<MxArray> {
        // Post-attention norm + residual
        let attn_normed = self.post_attention_layernorm.forward(attn_out)?;
        let h = x.add(&attn_normed)?;

        // Pre-FFN norm + MLP
        let ffn_normed = self.pre_feedforward_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&ffn_normed)?;

        // MoE branch (parallel to dense MLP)
        let combined_mlp_out = if let (Some(router), Some(moe), Some(pf1), Some(pf2), Some(pff2)) = (
            &self.router,
            &self.moe,
            &self.post_feedforward_layernorm_1,
            &self.post_feedforward_layernorm_2,
            &self.pre_feedforward_layernorm_2,
        ) {
            // Dense MLP branch: normalize output
            let hidden_states_1 = pf1.forward(&mlp_out)?;

            // MoE branch: router and experts see the RESIDUAL (pre-MLP state), not MLP output
            let (top_k_indices, top_k_weights) = router.forward(&h)?;
            let hidden_states_2 = pff2.forward(&h)?;
            let hidden_states_2 = moe.forward(&hidden_states_2, &top_k_indices, &top_k_weights)?;
            let hidden_states_2 = pf2.forward(&hidden_states_2)?;

            // Combine dense MLP and MoE outputs
            hidden_states_1.add(&hidden_states_2)?
        } else {
            mlp_out
        };

        // Post-FFN norm + residual
        let mlp_normed = self.post_feedforward_layernorm.forward(&combined_mlp_out)?;
        let mut out = h.add(&mlp_normed)?;

        // Apply PLE (per-layer embeddings) BEFORE layer_scalar (matches HF order)
        if let (Some(gate_proj), Some(proj), Some(norm), Some(ple_input)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            let residual = out.clone();
            let gated = gate_proj.forward(&residual)?;
            let gated = Activations::gelu(&gated)?;
            let gated = gated.mul(ple_input)?;
            let gated = proj.forward(&gated)?;
            let gated = norm.forward(&gated)?;
            out = residual.add(&gated)?;
        }

        // Apply layer scalar LAST (multiplies entire layer output including PLE)
        if let Some(ref scalar) = self.layer_scalar {
            out = out.mul(scalar)?;
        }

        Ok(out)
    }

    // ========== Test-only norm weight getters ==========

    #[cfg(test)]
    pub(crate) fn input_layernorm_weight(&self) -> MxArray {
        self.input_layernorm.get_weight()
    }
    #[cfg(test)]
    pub(crate) fn post_attention_layernorm_weight(&self) -> MxArray {
        self.post_attention_layernorm.get_weight()
    }
    #[cfg(test)]
    pub(crate) fn pre_feedforward_layernorm_weight(&self) -> MxArray {
        self.pre_feedforward_layernorm.get_weight()
    }
    #[cfg(test)]
    pub(crate) fn post_feedforward_layernorm_weight(&self) -> MxArray {
        self.post_feedforward_layernorm.get_weight()
    }

    // ========== Norm weight setters ==========

    pub fn set_input_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.input_layernorm.set_weight(w)
    }

    pub fn set_post_attention_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.post_attention_layernorm.set_weight(w)
    }

    pub fn set_pre_feedforward_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.pre_feedforward_layernorm.set_weight(w)
    }

    pub fn set_post_feedforward_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.post_feedforward_layernorm.set_weight(w)
    }

    /// Replace the dense MLP with a quantized version.
    pub fn set_quantized_dense_mlp(
        &mut self,
        gate_proj: QuantizedLinear,
        up_proj: QuantizedLinear,
        down_proj: QuantizedLinear,
    ) {
        self.mlp = Gemma4MLPVariant::Quantized {
            gate_proj,
            up_proj,
            down_proj,
        };
    }

    // ========== Layer scalar setter ==========

    pub fn set_layer_scalar(&mut self, scalar: &MxArray) -> Result<()> {
        self.layer_scalar = Some(scalar.clone());
        Ok(())
    }

    // ========== PLE per-layer weight setters ==========

    pub fn set_per_layer_input_gate_weight(&mut self, w: &MxArray) -> Result<()> {
        if let Some(ref mut gate) = self.per_layer_input_gate {
            gate.set_weight(w)
        } else {
            Err(Error::from_reason(
                "per_layer_input_gate not initialized (PLE not enabled)",
            ))
        }
    }

    pub fn set_per_layer_projection_weight(&mut self, w: &MxArray) -> Result<()> {
        if let Some(ref mut proj) = self.per_layer_projection {
            proj.set_weight(w)
        } else {
            Err(Error::from_reason(
                "per_layer_projection not initialized (PLE not enabled)",
            ))
        }
    }

    pub fn set_post_per_layer_input_norm_weight(&mut self, w: &MxArray) -> Result<()> {
        if let Some(ref mut norm) = self.post_per_layer_input_norm {
            norm.set_weight(w)
        } else {
            Err(Error::from_reason(
                "post_per_layer_input_norm not initialized (PLE not enabled)",
            ))
        }
    }

    /// Returns true if this layer has PLE components.
    pub fn has_ple(&self) -> bool {
        self.per_layer_input_gate.is_some()
    }

    /// Returns true if this layer has MoE components.
    pub fn has_moe(&self) -> bool {
        self.router.is_some()
    }

    // ========== MoE weight setters ==========

    pub fn set_router_scale(&mut self, w: &MxArray) -> Result<()> {
        if let Some(ref mut router) = self.router {
            router.set_scale(w)
        } else {
            Err(Error::from_reason(
                "Router not initialized (MoE not enabled)",
            ))
        }
    }

    pub fn set_router_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        if let Some(ref mut router) = self.router {
            router.set_proj_weight(w)
        } else {
            Err(Error::from_reason(
                "Router not initialized (MoE not enabled)",
            ))
        }
    }

    pub fn set_router_proj_quantized(
        &mut self,
        weight: &MxArray,
        scales: &MxArray,
        biases: Option<&MxArray>,
        group_size: i32,
        bits: i32,
    ) -> Result<()> {
        if let Some(ref mut router) = self.router {
            router.set_proj_quantized(weight, scales, biases, group_size, bits)
        } else {
            Err(Error::from_reason(
                "Router not initialized (MoE not enabled)",
            ))
        }
    }

    pub fn set_moe_gate_up_proj(&mut self, w: &MxArray) -> Result<()> {
        if let Some(ref mut moe) = self.moe {
            moe.set_gate_up_proj(w)
        } else {
            Err(Error::from_reason("MoE not initialized"))
        }
    }

    pub fn set_moe_gate_up_proj_quantized(&mut self, qsl: QuantizedSwitchLinear) -> Result<()> {
        if let Some(ref mut moe) = self.moe {
            moe.set_gate_up_proj_quantized(qsl);
            Ok(())
        } else {
            Err(Error::from_reason("MoE not initialized"))
        }
    }

    pub fn set_moe_down_proj(&mut self, w: &MxArray) -> Result<()> {
        if let Some(ref mut moe) = self.moe {
            moe.set_down_proj(w)
        } else {
            Err(Error::from_reason("MoE not initialized"))
        }
    }

    pub fn set_moe_down_proj_quantized(&mut self, qsl: QuantizedSwitchLinear) -> Result<()> {
        if let Some(ref mut moe) = self.moe {
            moe.set_down_proj_quantized(qsl);
            Ok(())
        } else {
            Err(Error::from_reason("MoE not initialized"))
        }
    }

    pub fn set_moe_per_expert_scale(&mut self, w: &MxArray) -> Result<()> {
        if let Some(ref mut router) = self.router {
            router.set_per_expert_scale(w)
        } else {
            Err(Error::from_reason(
                "Router not initialized (MoE not enabled)",
            ))
        }
    }

    pub fn set_pre_feedforward_layernorm_2_weight(&mut self, w: &MxArray) -> Result<()> {
        if let Some(ref mut norm) = self.pre_feedforward_layernorm_2 {
            norm.set_weight(w)
        } else {
            Err(Error::from_reason(
                "pre_feedforward_layernorm_2 not initialized (MoE not enabled)",
            ))
        }
    }

    pub fn set_post_feedforward_layernorm_1_weight(&mut self, w: &MxArray) -> Result<()> {
        if let Some(ref mut norm) = self.post_feedforward_layernorm_1 {
            norm.set_weight(w)
        } else {
            Err(Error::from_reason(
                "post_feedforward_layernorm_1 not initialized (MoE not enabled)",
            ))
        }
    }

    pub fn set_post_feedforward_layernorm_2_weight(&mut self, w: &MxArray) -> Result<()> {
        if let Some(ref mut norm) = self.post_feedforward_layernorm_2 {
            norm.set_weight(w)
        } else {
            Err(Error::from_reason(
                "post_feedforward_layernorm_2 not initialized (MoE not enabled)",
            ))
        }
    }
}
