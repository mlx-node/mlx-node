use crate::array::mask::create_causal_mask;
use crate::array::{MxArray, scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::nn::RMSNorm;
use crate::transformer::attention::Attention;
use crate::transformer::kv_cache::KVCache;
use crate::transformer::mlp::MLP;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::ptr;

/// Transformer block combining self-attention and MLP with pre-normalization.
///
/// Architecture (Qwen3/Llama style):
/// 1. x = x + self_attn(norm(x))  # Pre-norm + residual
/// 2. x = x + mlp(norm(x))        # Pre-norm + residual
pub struct TransformerBlock {
    pub(crate) self_attn: Attention,
    pub(crate) mlp: MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    // Config for fused forward
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    attn_scale: f32,
    rope_base: f32,
    rope_dims: i32,
    norm_eps: f32,
    use_qk_norm: bool,
}

impl TransformerBlock {
    /// Creates a new transformer block.
    ///
    /// # Arguments
    /// * `hidden_size` - Model dimension
    /// * `num_heads` - Number of attention heads
    /// * `num_kv_heads` - Number of key/value heads (for GQA)
    /// * `intermediate_size` - FFN hidden dimension
    /// * `rms_norm_eps` - Epsilon for RMSNorm
    /// * `rope_theta` - RoPE base frequency (optional)
    /// * `use_qk_norm` - Whether to use QK normalization (optional)
    /// * `head_dim` - Dimension per head (optional)
    pub fn new(
        hidden_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        intermediate_size: u32,
        rms_norm_eps: f64,
        rope_theta: Option<f64>,
        use_qk_norm: Option<bool>,
        head_dim: Option<u32>,
    ) -> Result<Self> {
        let head_dim_val = head_dim.unwrap_or(hidden_size / num_heads);
        let rope_theta_val = rope_theta.unwrap_or(10000.0);
        let use_qk_norm_val = use_qk_norm.unwrap_or(false);

        let self_attn = Attention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            Some(head_dim_val),
            Some(rope_theta_val),
            Some(use_qk_norm_val),
            Some(rms_norm_eps),
        )?;

        let mlp = MLP::new(hidden_size, intermediate_size)?;

        let input_layernorm = RMSNorm::new(hidden_size, Some(rms_norm_eps))?;
        let post_attention_layernorm = RMSNorm::new(hidden_size, Some(rms_norm_eps))?;

        // Store config for fused forward
        let attn_scale = 1.0 / (head_dim_val as f64).sqrt();

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            n_heads: num_heads as i32,
            n_kv_heads: num_kv_heads as i32,
            head_dim: head_dim_val as i32,
            attn_scale: attn_scale as f32,
            rope_base: rope_theta_val as f32,
            rope_dims: head_dim_val as i32,
            norm_eps: rms_norm_eps as f32,
            use_qk_norm: use_qk_norm_val,
        })
    }

    /// Forward pass through transformer block.
    ///
    /// # Arguments
    /// * `x` - Input tensor, shape: (batch, seq_len, hidden_size)
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional KV cache for incremental generation
    ///
    /// # Returns
    /// Output tensor, shape: (batch, seq_len, hidden_size)
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        // For cached inference, use component-based forward (cache handling is complex)
        // For non-cached (training/prefill), use fused C++ implementation
        if cache.is_some() {
            return self.forward_with_cache(x, mask, cache);
        }

        // Use fused C++ implementation for non-cached forward
        // This reduces ~40 FFI calls to 1 per block
        let input_norm_w = self.input_layernorm.get_weight();
        let post_attn_norm_w = self.post_attention_layernorm.get_weight();
        let w_q = self.self_attn.get_q_proj_weight();
        let w_k = self.self_attn.get_k_proj_weight();
        let w_v = self.self_attn.get_v_proj_weight();
        let w_o = self.self_attn.get_o_proj_weight();
        let q_norm_w = self.self_attn.get_q_norm_weight();
        let k_norm_w = self.self_attn.get_k_norm_weight();
        let w_gate = self.mlp.get_gate_proj_weight();
        let w_up = self.mlp.get_up_proj_weight();
        let w_down = self.mlp.get_down_proj_weight();

        // Determine if we should use causal masking
        let use_causal = mask.is_none(); // Use causal if no explicit mask provided

        let handle = unsafe {
            sys::mlx_fused_transformer_block_forward(
                x.handle.0,
                input_norm_w.handle.0,
                post_attn_norm_w.handle.0,
                w_q.handle.0,
                w_k.handle.0,
                w_v.handle.0,
                w_o.handle.0,
                q_norm_w.map(|w| w.handle.0).unwrap_or(ptr::null_mut()),
                k_norm_w.map(|w| w.handle.0).unwrap_or(ptr::null_mut()),
                w_gate.handle.0,
                w_up.handle.0,
                w_down.handle.0,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.attn_scale,
                self.rope_base,
                self.rope_dims,
                self.norm_eps,
                self.norm_eps, // qk_norm_eps (same as norm_eps for Qwen3)
                use_causal,
                0, // rope_offset for non-cached
            )
        };

        MxArray::from_handle(handle, "fused_transformer_block_forward")
    }

    /// Forward pass with KV cache (component-based for complex cache handling)
    fn forward_with_cache(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        // 1. Self-attention with pre-norm and residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        let h = x.add(&attn_out)?; // Residual connection

        // 2. MLP with pre-norm and residual (uses fused MLP internally)
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let out = h.add(&mlp_out)?; // Residual connection

        Ok(out)
    }

    /// Forward pass driven by `PagedKVCacheAdapter` (vLLM-style block-paged
    /// KV with refcounted prefix reuse).
    ///
    /// This is the read-side complement to `Qwen3Inner::paged_adapter`, used
    /// when `Qwen3Config::use_block_paged_cache` is opt-in true. It is
    /// intentionally separate from `forward_paged_metal` (which talks to the
    /// older `PagedKVCache` + `ContinuousBatchingScheduler` path); this entry
    /// point is the one we plan to wire `chat_sync_core` through, and it
    /// tracks per-request block-table semantics rather than batched
    /// continuous-batching semantics.
    ///
    /// # Arguments
    /// * `x` — input tensor.
    ///   - For prefill: `[1, seq_len, hidden_size]` (the suffix tokens this
    ///     layer needs to attend; a cached prefix already lives in the pool
    ///     and the adapter's block_table covers the full request).
    ///   - For decode: `[1, 1, hidden_size]` (single new token).
    /// * `adapter` — the active per-request adapter. Caller must already
    ///   have run `reset_for_new_request`, optionally `find_cached_prefix`,
    ///   `allocate_suffix_blocks`, and `record_tokens` for **this chunk**
    ///   *before* calling this method (see `update_keys_values` alignment
    ///   contract).
    /// * `layer_idx` — index of this layer (must match the order layers were
    ///   constructed in `Qwen3Inner::new`).
    /// * `first_logical_position` — logical token index in the request where
    ///   this chunk's first token lives. For a fresh prefill that's
    ///   `cached_token_count` (the cache hit boundary); for decode it's
    ///   `current_token_count - 1`. Forwarded verbatim to
    ///   `update_keys_values` for the alignment check.
    /// * `is_prefill` — selects the attention path. Prefill runs standard
    ///   causal SDPA on the in-flight Q/K/V (the paged Metal kernel is
    ///   decode-only — single-token Q per sequence — so we do not use it
    ///   here, even though we still write through to the pool). Decode
    ///   goes through `gather_kv_for_decode`, which pulls historical K/V
    ///   from the paged buffers and runs the paged-attention Metal kernel.
    /// * `cached_prefix_len` — number of logical tokens already covered by
    ///   the prefix cache when `is_prefill` is true. When > 0 the suffix
    ///   tokens must attend over the FULL `[0, first_logical_position +
    ///   num_suffix_tokens)` context, so `read_kv_range` materializes the
    ///   cached + just-written K/V back to MxArrays for SDPA. Ignored in
    ///   the decode path.
    /// * `_positions`, `_num_query_heads`, `_num_seqs`, `_seq_len` — kept on
    ///   the signature for symmetry with `forward_paged_metal` and likely
    ///   future multi-sequence batching, but unused in the current
    ///   single-sequence path.
    ///
    /// # Returns
    /// Output tensor with the same shape as `x`.
    ///
    /// # Caveats / unwired pieces
    ///
    /// - **Single-sequence only.** `gather_kv_for_decode` and the adapter's
    ///   block_table are currently scoped to one request. Continuous
    ///   batching is the legacy `forward_paged_metal` path's responsibility.
    /// - **Cache-hit prefill via host-side gather.** When `cached_prefix_len > 0`
    ///   we `read_kv_range` the full context K/V back to MxArrays and run
    ///   causal SDPA over them. That gather is host-side (slow but correct).
    ///   The on-device zero-copy fast-path (TODO in the adapter) will replace
    ///   the read with a Metal kernel that attends in-place against the paged
    ///   buffers.
    /// - **Decode output dtype.** `gather_kv_for_decode` currently returns
    ///   Float32 (host-roundtrip). We `astype` it back to `x`'s dtype
    ///   before output projection / residual so the rest of the block stays
    ///   in BF16. The on-device zero-copy fast-path (TODO in the adapter)
    ///   will remove this cast.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_paged_adapter(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        layer_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        _num_query_heads: u32,
        _positions: &MxArray,
        _num_seqs: i64,
        _seq_len: i64,
        is_prefill: bool,
    ) -> Result<MxArray> {
        // 1. Input layer norm.
        let normed = self.input_layernorm.forward(x)?;

        // 2. Compute Q/K/V with RoPE applied. Use `compute_qkv` with the
        //    request's logical position as the RoPE offset — this matches
        //    `forward_with_cache`'s use of `cache.get_offset()`. For
        //    prefill the offset is the first cached/new position; for
        //    decode it's the position of the single new token.
        //
        //    `compute_qkv` returns Q/K/V in the *paged* layout
        //    `[num_tokens, num_heads, head_dim]` (see Attention::compute_qkv
        //    docstring) — exactly what `update_keys_values` and SDPA
        //    expect. For prefill we still need to reshape into 4D
        //    (batch=1, num_heads, seq_len, head_dim) for SDPA; for decode
        //    we feed `gather_kv_for_decode` the 3D queries directly.
        let qkv = self
            .self_attn
            .compute_qkv(&normed, first_logical_position as i32)?;

        // 3. Always write the new K/V chunk through to the paged pool so
        //    future tokens (or future requests sharing this prefix) can
        //    read them back via gather. Note: the adapter's
        //    `record_tokens` MUST already have advanced the cursor by the
        //    chunk size before this call (see method docstring).
        adapter
            .update_keys_values(layer_idx, &qkv.keys, &qkv.values, first_logical_position)
            .map_err(napi::Error::from_reason)?;

        // 4. Compute attention output.
        let attn_out_paged = if is_prefill {
            // Prefill: reshape Q into SDPA-friendly 4D layout. K/V depend
            // on whether we have a cached prefix:
            //   - `cached_prefix_len == 0`: SDPA over the in-flight Q/K/V
            //     directly (no read back required).
            //   - `cached_prefix_len > 0`: read the FULL `[0, total_ctx)`
            //     K/V back from the pool — cached prefix already lives
            //     there, and `update_keys_values` (called above) just
            //     wrote the suffix. SDPA then attends Q over total_ctx.
            let num_tokens = qkv.queries.shape_at(0)?;
            let n_heads = qkv.queries.shape_at(1)?;
            let n_kv_heads = qkv.keys.shape_at(1)?;
            let head_dim = qkv.queries.shape_at(2)?;

            // Q: [num_tokens, n_heads, head_dim] -> [1, n_heads, num_tokens, head_dim]
            let q_4d = qkv
                .queries
                .reshape(&[1, num_tokens, n_heads, head_dim])?
                .transpose(Some(&[0, 2, 1, 3]))?;

            let scale = self.self_attn.get_scale();

            let attn_4d = if cached_prefix_len == 0 {
                // No cached prefix — SDPA over the in-flight Q/K/V with
                // MLX's optimized internal causal mask (Q seq == K seq).
                let k_4d = qkv
                    .keys
                    .reshape(&[1, num_tokens, n_kv_heads, head_dim])?
                    .transpose(Some(&[0, 2, 1, 3]))?;
                let v_4d = qkv
                    .values
                    .reshape(&[1, num_tokens, n_kv_heads, head_dim])?
                    .transpose(Some(&[0, 2, 1, 3]))?;
                scaled_dot_product_attention_causal(&q_4d, &k_4d, &v_4d, scale)?
            } else {
                // Cache hit: pull total_ctx K/V back from the pool. The
                // suffix was already written via `update_keys_values`
                // above, so the pool's `[0, total_ctx)` covers the full
                // attention context.
                let total_ctx = cached_prefix_len + (num_tokens as u32);
                let (k_4d, v_4d) = adapter
                    .read_kv_range(layer_idx, 0, total_ctx)
                    .map_err(napi::Error::from_reason)?;
                // `read_kv_range` returns `[1, n_kv_heads, total_ctx, head_dim]`.

                // Build an explicit causal mask of shape
                // `[num_tokens, total_ctx]` where row i (suffix token i,
                // logical position cached_prefix_len + i) can attend to
                // columns 0..=cached_prefix_len + i. `create_causal_mask(
                // seq_len=num_tokens, offset=cached_prefix_len)` does
                // exactly that. The mask broadcasts over batch / heads via
                // SDPA's mask handling.
                let mask =
                    create_causal_mask(num_tokens as i32, Some(cached_prefix_len as i32), None)?;
                scaled_dot_product_attention(&q_4d, &k_4d, &v_4d, scale, Some(&mask))?
            };

            // [1, n_heads, num_tokens, head_dim] -> [num_tokens, n_heads, head_dim]
            // for `output_projection`'s expected layout.
            attn_4d
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[num_tokens, n_heads, head_dim])?
        } else {
            // Decode: K/V at the new position has just been written to the
            // pool by `update_keys_values` above. Dispatch the
            // `gather_kv_for_decode` Metal kernel directly against the
            // on-GPU paged buffers — avoids the per-step host roundtrip
            // (~57 MB per layer per K/V on long contexts) that
            // `read_kv_range` performs and that was driving a ~40 GB
            // memory regression in long-context decode (see Fix #2 spec).
            //
            // The kernel returns the query/io dtype. Cast back to x's dtype
            // so the rest of the block stays homogeneous. This is a
            // zero-extra-host-buffer path — the K/V are not materialized as
            // MxArrays at all.
            //
            // `gather_kv_for_decode` expects queries shape
            // `[1, num_query_heads, head_size]` (3-D). `qkv.queries` from
            // `compute_qkv` is already `[num_tokens=1, n_heads, head_dim]`
            // for the single-token decode chunk, so it's a direct fit.
            let scale = self.self_attn.get_scale();

            let attn_3d = adapter
                .gather_kv_for_decode(
                    layer_idx,
                    &qkv.queries,
                    scale as f32,
                    /* softcap */ 1.0,
                )
                .map_err(napi::Error::from_reason)?;

            // Cast back to x's dtype.
            // `gather_kv_for_decode` returns `[1, n_heads, head_dim]`, which
            // is exactly the layout `output_projection` expects for
            // `batch * seq_len = 1`.
            let target_dtype = x.dtype()?;
            attn_3d.astype(target_dtype)?
        };

        // 5. Output projection + residual + MLP, identical to the other
        //    forward variants.
        let num_seqs = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;
        let attn_out = self
            .self_attn
            .output_projection(&attn_out_paged, num_seqs, seq_len)?;
        let h = x.add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let out = h.add(&mlp_out)?;

        Ok(out)
    }

    // Norm weight getters/setters for parameter management

    pub fn get_input_layernorm_weight(&self) -> MxArray {
        self.input_layernorm.get_weight()
    }

    pub fn get_post_attention_layernorm_weight(&self) -> MxArray {
        self.post_attention_layernorm.get_weight()
    }

    pub fn set_input_layernorm_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.input_layernorm.set_weight(weight)?;
        Ok(())
    }

    pub fn set_post_attention_layernorm_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.post_attention_layernorm.set_weight(weight)?;
        Ok(())
    }
}

impl Clone for TransformerBlock {
    fn clone(&self) -> Self {
        Self {
            self_attn: self.self_attn.clone(),
            mlp: self.mlp.clone(),
            input_layernorm: self.input_layernorm.clone(),
            post_attention_layernorm: self.post_attention_layernorm.clone(),
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            attn_scale: self.attn_scale,
            rope_base: self.rope_base,
            rope_dims: self.rope_dims,
            norm_eps: self.norm_eps,
            use_qk_norm: self.use_qk_norm,
        }
    }
}
