use std::sync::OnceLock;

use crate::array::MxArray;
use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::mask::create_causal_mask;
use crate::models::qwen3_5_moe::quantized_linear::LinearProj;
use crate::nn::{Linear, RMSNorm, RoPE};
use crate::transformer::KVCache;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use napi::bindgen_prelude::*;

/// When enabled (default), paged decode writes K/V into the pool with the
/// graph-native, lazily-scheduled `update_keys_values_native` so the write
/// feeds the same-step attention read through MLX graph dependencies — no
/// per-layer host sync. When disabled, the synchronous `update_keys_values`
/// (a raw Metal write outside the graph scheduler) is used instead. The sync
/// path is also taken automatically when the native write fails.
fn native_kv_write_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled_or_default("MLX_LFM2_NATIVE_KV_WRITE", true)
    })
}

/// When enabled (default), paged decode gathers historical K/V with the
/// graph-native `gather_kv_for_decode_graph`, which consumes the lazy pool
/// arrays via graph dependencies (no per-layer host eval). When disabled, the
/// synchronous `gather_kv_for_decode` (which forces a pending-write eval and
/// reads the pool outside the graph) is used. The sync path is also taken
/// automatically when the graph gather is unavailable for the inputs.
fn graph_decode_gather_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled_or_default("MLX_LFM2_GRAPH_DECODE_GATHER", true)
    })
}

/// When enabled (opt-in; default OFF), cache-hit prefill (`cached_prefix_len > 0`,
/// i.e. every multi-turn chat continuation) first tries the MLX graph-native
/// paged-attention bridge (`PagedKVCacheAdapter::gather_kv_for_prefill_chunk`),
/// which reads the K/V pool through MLX graph dependencies with no forced host
/// sync. When disabled (the default), or when the bridge is unavailable for the
/// inputs (non-Metal backend, batch > 1, an unsupported cache dtype, or an
/// oversized auxiliary buffer), the synchronous `read_kv_range` path is used
/// instead — a `[0, total_ctx)` host read that forces a per-layer pool eval via
/// `eval_pending_pool_write_for_layer`.
///
/// The bridge reads the SAME physical KV bytes as `read_kv_range`; only the
/// attention kernel differs (fused paged-attn vs explicit-mask SDPA), the
/// accepted ~1-ULP class already shipped default-on for paged DECODE. It is held
/// opt-in here — unlike `qwen3_5`/`gemma4`, which default it on — only because
/// the divergence has no green automated parity gate on the one available LFM2
/// checkpoint (greedy decode there is repeat-loop-degenerate, so byte-identical
/// text parity is an unreliable oracle). Flip to default-on once a gemma4-style
/// paged-vs-flat gate exists on a stable checkpoint.
fn paged_prefill_paged_attention_enabled() -> bool {
    // Without a Metal backend (CUDA/Linux build) the C++ paged-attention
    // kernel throws, so a cache-hit prefill must NOT dispatch it. Hard-close
    // the path here so reuse-turn prefills stay on the device-agnostic SDPA
    // fallback (`read_kv_range` + explicit mask).
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return false;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled_or_default(
            "MLX_LFM2_PAGED_PREFILL_PAGED_ATTENTION",
            false,
        )
    })
}

/// LFM2 multi-head attention with QK RMSNorm and RoPE.
///
/// Follows `lfm2.py:53-109` (Attention class).
///
/// Key features:
/// - GQA: 32 query heads, 8 KV heads (head_dim=64)
/// - Per-head RMSNorm on Q and K (not V)
/// - Standard RoPE (neox-style, base=1M)
/// - No bias on any projection
pub struct Lfm2Attention {
    pub(crate) q_proj: LinearProj,
    pub(crate) k_proj: LinearProj,
    pub(crate) v_proj: LinearProj,
    pub(crate) out_proj: LinearProj,
    pub(crate) q_layernorm: RMSNorm,
    pub(crate) k_layernorm: RMSNorm,
    rope: RoPE,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f64,
}

impl Lfm2Attention {
    /// Create a new LFM2 attention layer.
    pub fn new(
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        norm_eps: f64,
        rope_theta: f64,
    ) -> Result<Self> {
        let h = hidden_size as u32;
        let q_dim = (num_heads * head_dim) as u32;
        let kv_dim = (num_kv_heads * head_dim) as u32;

        let q_proj = LinearProj::Standard(Linear::new(h, q_dim, Some(false))?);
        let k_proj = LinearProj::Standard(Linear::new(h, kv_dim, Some(false))?);
        let v_proj = LinearProj::Standard(Linear::new(h, kv_dim, Some(false))?);
        let out_proj = LinearProj::Standard(Linear::new(q_dim, h, Some(false))?);

        let q_layernorm = RMSNorm::new(head_dim as u32, Some(norm_eps))?;
        let k_layernorm = RMSNorm::new(head_dim as u32, Some(norm_eps))?;

        let rope = RoPE::new(
            head_dim,
            Some(false), // traditional=False (neox-style)
            Some(rope_theta),
            None,
        );

        let scale = (head_dim as f64).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_layernorm,
            k_layernorm,
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input [B, T, hidden_size]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional KVCache for incremental decoding
    ///
    /// # Returns
    /// Output tensor [B, T, hidden_size]
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // Q/K/V projections
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape to [B, T, num_heads, head_dim] and apply per-head layernorm
        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let queries = self.q_layernorm.forward(&queries)?;
        // Transpose to [B, num_heads, T, head_dim]
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;

        let keys = keys.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let keys = self.k_layernorm.forward(&keys)?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;

        // V: reshape + transpose (no layernorm on V)
        let values = values.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        // Apply RoPE with cache offset
        let offset = cache.as_ref().map_or(0, |c| c.get_offset());
        let queries = self.rope.forward(&queries, Some(offset))?;
        let keys = self.rope.forward(&keys, Some(offset))?;

        // Update KV cache
        let (keys, values) = if let Some(c) = cache {
            c.update_and_fetch(&keys, &values)?
        } else {
            (keys, values)
        };

        // Scaled dot-product attention
        let output = if let Some(m) = mask {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale, Some(m))?
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(&queries, &keys, &values, self.scale)?
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale, None)?
        };

        // Transpose back [B, H, T, D] -> [B, T, H*D]
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Output projection
        self.out_proj.forward(&output)
    }

    /// Forward pass driven by `PagedKVCacheAdapter` for full-attention
    /// LFM2 layers.
    ///
    /// Mirrors `TransformerBlock::forward_paged_adapter` (Qwen3) but
    /// adapted for LFM2's attention layout (Q/K layernorm AFTER reshape,
    /// V has no layernorm, no Q gating). The decoder layer's
    /// pre-attention `operator_norm` is applied OUTSIDE this method to
    /// match the existing flat-path call site, so `x` here is already
    /// pre-normalized.
    ///
    /// Caller responsibilities (mirrors Qwen3 helper contract):
    /// 1. `adapter.record_tokens(&[suffix])` BEFORE this call so the
    ///    cursor is advanced by the chunk; `update_keys_values` enforces
    ///    alignment.
    /// 2. `attn_layer_idx` is the ATTENTION-LAYER ORDINAL into the
    ///    adapter's `LayerKVPool`, NOT the absolute decoder index. The
    ///    pool is sized for `config.full_attn_idxs().len()` slots.
    ///
    /// Output: `[1, seq_len, hidden_size]` so the residual `h = x + r`
    /// in the decoder layer stays the same.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // 1. Q/K/V projections
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // 2. Reshape to [B, T, H, D] and apply per-head layernorm to Q/K
        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let queries = self.q_layernorm.forward(&queries)?;
        let queries_bhtd = queries.transpose(Some(&[0, 2, 1, 3]))?;

        let keys = keys.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let keys = self.k_layernorm.forward(&keys)?;
        let keys_bhtd = keys.transpose(Some(&[0, 2, 1, 3]))?;

        let values = values.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values_bhtd = values.transpose(Some(&[0, 2, 1, 3]))?;

        // 3. Apply RoPE using `first_logical_position` (the adapter's
        //    pre-record offset).
        let rope_offset = first_logical_position as i32;
        let queries_bhtd = self.rope.forward(&queries_bhtd, Some(rope_offset))?;
        let keys_bhtd = self.rope.forward(&keys_bhtd, Some(rope_offset))?;

        // 4. Convert K/V to the paged layout `[num_tokens, n_kv_heads,
        //    head_dim]` expected by `update_keys_values`. Currently
        //    batch=1 so `num_tokens = batch * seq_len = seq_len`.
        //    [B, H_kv, T, D] -> [B, T, H_kv, D] -> [B*T, H_kv, D]
        let keys_paged = keys_bhtd.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            batch * seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values_paged = values_bhtd.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            batch * seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;

        // Prefer the graph-native lazy write so the same-step attention read
        // depends on it through MLX's graph (no per-layer host sync). Fall
        // back to the synchronous write if it is disabled or the native
        // kernel could not place the K/V (a failed native write leaves the
        // pool untouched, so the sync write below is not a double-write).
        let native_written = native_kv_write_enabled()
            && adapter
                .update_keys_values_native(
                    attn_layer_idx,
                    &keys_paged,
                    &values_paged,
                    first_logical_position,
                )
                .is_ok();
        if !native_written {
            adapter
                .update_keys_values(
                    attn_layer_idx,
                    &keys_paged,
                    &values_paged,
                    first_logical_position,
                )
                .map_err(napi::Error::from_reason)?;
        }

        // 5. Compute attention output.
        let attn_bhtd = if is_prefill {
            if cached_prefix_len == 0 {
                // Fresh prefill: SDPA over in-flight Q/K/V with internal
                // causal mask.
                if seq_len > 1 {
                    scaled_dot_product_attention_causal(
                        &queries_bhtd,
                        &keys_bhtd,
                        &values_bhtd,
                        self.scale,
                    )?
                } else {
                    scaled_dot_product_attention(
                        &queries_bhtd,
                        &keys_bhtd,
                        &values_bhtd,
                        self.scale,
                        None,
                    )?
                }
            } else {
                // Cache-hit prefill: the suffix was just written above.
                // Prefer the MLX graph-native paged-attention bridge (reads
                // the pool through graph dependencies, no per-layer host
                // sync) over `read_kv_range`, which forces
                // `eval_pending_pool_write_for_layer` on every call. Mirrors
                // the qwen3_5 / gemma4 `forward_paged` cache-hit branch.
                let total_ctx = cached_prefix_len + (seq_len as u32);
                let maybe_paged_attn = if batch == 1 && paged_prefill_paged_attention_enabled() {
                    // [B, H, T, D] -> [H, T, D] -> [T, H, D], matching
                    // `PagedKVCacheAdapter::gather_kv_for_prefill_chunk`.
                    let queries_paged = queries_bhtd
                        .squeeze(Some(&[0]))?
                        .transpose(Some(&[1, 0, 2]))?;
                    match adapter.gather_kv_for_prefill_chunk(
                        attn_layer_idx,
                        &queries_paged,
                        cached_prefix_len,
                        self.scale as f32,
                    ) {
                        Ok(attn_t_h_d) => {
                            let target_dtype = x.dtype()?;
                            let attn_t_h_d = attn_t_h_d.astype(target_dtype)?;
                            // [T, H, D] -> [H, T, D] -> [B, H, T, D]
                            let attn = attn_t_h_d.transpose(Some(&[1, 0, 2]))?.reshape(&[
                                batch,
                                self.num_heads as i64,
                                seq_len,
                                self.head_dim as i64,
                            ])?;
                            Some(attn)
                        }
                        Err(_) => None,
                    }
                } else {
                    None
                };

                match maybe_paged_attn {
                    Some(attn) => attn,
                    None => {
                        let (k_full, v_full) = adapter
                            .read_kv_range(attn_layer_idx, 0, total_ctx)
                            .map_err(napi::Error::from_reason)?;
                        let mask = create_causal_mask(
                            seq_len as i32,
                            Some(cached_prefix_len as i32),
                            None,
                        )?;
                        scaled_dot_product_attention(
                            &queries_bhtd,
                            &k_full,
                            &v_full,
                            self.scale,
                            Some(&mask),
                        )?
                    }
                }
            }
        } else {
            // Decode: gather full historical K/V via the paged kernel. Both
            // gather variants expect `[1, num_query_heads, head_size]`
            // queries, so reshape from [1, H, 1, D].
            let queries_3d = queries_bhtd.squeeze(Some(&[2]))?.reshape(&[
                1,
                self.num_heads as i64,
                self.head_dim as i64,
            ])?;
            // Prefer the graph-native gather (reads the lazy pool arrays
            // through graph dependencies — no per-layer host eval). Fall back
            // to the synchronous gather when it is disabled or unavailable for
            // these inputs (e.g. a query/cache dtype it cannot serve).
            let attn_3d = if graph_decode_gather_enabled() {
                match adapter.gather_kv_for_decode_graph(
                    attn_layer_idx,
                    &queries_3d,
                    self.scale as f32,
                    /* softcap */ 1.0,
                ) {
                    Ok(attn_3d) => attn_3d,
                    Err(_) => adapter
                        .gather_kv_for_decode(
                            attn_layer_idx,
                            &queries_3d,
                            self.scale as f32,
                            /* softcap */ 1.0,
                        )
                        .map_err(napi::Error::from_reason)?,
                }
            } else {
                adapter
                    .gather_kv_for_decode(
                        attn_layer_idx,
                        &queries_3d,
                        self.scale as f32,
                        /* softcap */ 1.0,
                    )
                    .map_err(napi::Error::from_reason)?
            };
            // Cast back to x's dtype so the residual stays homogeneous.
            let target_dtype = x.dtype()?;
            let attn_3d = attn_3d.astype(target_dtype)?;
            // Reshape [1, H, D] -> [1, H, 1, D] for the standard
            // [B,H,T,D] tail.
            attn_3d.reshape(&[1, self.num_heads as i64, 1, self.head_dim as i64])?
        };

        // 6. Output: [B, H, T, D] -> [B, T, H*D] -> projection.
        let output = attn_bhtd.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;
        self.out_proj.forward(&output)
    }

    // ========== Weight setters ==========
    //
    // NOTE: q/k/v/out_proj weights are loaded via the `*_proj_mut()` accessors
    // below, which expose the mode-aware `LinearProj`. The persistence layer
    // either installs a `QuantizedLinear` backend (any of affine / mxfp4 /
    // mxfp8 / nvfp4) via `LinearProj::set_quantized`, or sets a dense bf16
    // weight via `LinearProj::set_weight`. The `forward` path dispatches
    // quantized vs dense transparently. q/k_layernorm are never quantized.

    pub fn set_q_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.q_layernorm.set_weight(w)
    }

    pub fn set_k_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.k_layernorm.set_weight(w)
    }

    // ========== Mutable projection accessors ==========
    //
    // Expose the mode-aware `LinearProj`s so the persistence layer can install
    // a quantized backend (affine / mxfp4 / mxfp8 / nvfp4) via
    // `set_quantized`, or a plain bf16 weight via `set_weight`, uniformly for a
    // fully quantized checkpoint. The `forward` path dispatches quantized vs
    // dense transparently.

    pub fn q_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.q_proj
    }

    pub fn k_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.k_proj
    }

    pub fn v_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.v_proj
    }

    pub fn out_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.out_proj
    }
}
