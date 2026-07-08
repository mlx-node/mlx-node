#![allow(non_camel_case_types)]

#[repr(C)]
#[derive(Debug)]
pub struct mlx_array {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mlx_stream {
    pub index: i32,
    pub device_type: i32, // 0 = CPU, 1 = GPU
}

unsafe extern "C-unwind" {
    pub fn mlx_seed(seed: u64);
    pub fn mlx_array_from_int32(data: *const i32, shape: *const i64, ndim: usize)
    -> *mut mlx_array;
    pub fn mlx_array_from_int64(data: *const i64, shape: *const i64, ndim: usize)
    -> *mut mlx_array;
    pub fn mlx_array_from_uint32(
        data: *const u32,
        shape: *const i64,
        ndim: usize,
    ) -> *mut mlx_array;
    pub fn mlx_array_from_uint8(data: *const u8, shape: *const i64, ndim: usize) -> *mut mlx_array;
    pub fn mlx_array_from_int8(data: *const i8, shape: *const i64, ndim: usize) -> *mut mlx_array;
    pub fn mlx_array_from_float32(
        data: *const f32,
        shape: *const i64,
        ndim: usize,
    ) -> *mut mlx_array;
    pub fn mlx_array_from_bfloat16(
        data: *const u16,
        shape: *const i64,
        ndim: usize,
    ) -> *mut mlx_array;
    pub fn mlx_array_from_float16(
        data: *const u16,
        shape: *const i64,
        ndim: usize,
    ) -> *mut mlx_array;
    pub fn mlx_from_fp8(handle: *mut mlx_array, target_dtype: i32) -> *mut mlx_array;
    pub fn mlx_to_fp8(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_scalar_float(value: f64) -> *mut mlx_array;
    pub fn mlx_array_scalar_int(value: i32) -> *mut mlx_array;
    pub fn mlx_array_zeros(shape: *const i64, ndim: usize, dtype: i32) -> *mut mlx_array;
    pub fn mlx_array_ones(shape: *const i64, ndim: usize, dtype: i32) -> *mut mlx_array;
    pub fn mlx_array_full(
        shape: *const i64,
        ndim: usize,
        value_handle: *mut mlx_array,
        dtype: i32,
        has_dtype: bool,
    ) -> *mut mlx_array;
    pub fn mlx_array_reshape(
        handle: *mut mlx_array,
        shape: *const i64,
        ndim: usize,
    ) -> *mut mlx_array;
    pub fn mlx_array_astype(handle: *mut mlx_array, dtype: i32) -> *mut mlx_array;
    pub fn mlx_array_copy(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_log_softmax(handle: *mut mlx_array, axis: i32) -> *mut mlx_array;
    pub fn mlx_array_logsumexp(
        handle: *mut mlx_array,
        axes: *const i32,
        axes_len: usize,
        keepdims: bool,
    ) -> *mut mlx_array;
    pub fn mlx_array_softmax(handle: *mut mlx_array, axis: i32) -> *mut mlx_array;
    pub fn mlx_array_softmax_precise(handle: *mut mlx_array, axis: i32) -> *mut mlx_array;
    pub fn mlx_array_sigmoid(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_exp(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_log(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_sum(
        handle: *mut mlx_array,
        axes: *const i32,
        axes_len: usize,
        keepdims: bool,
    ) -> *mut mlx_array;
    pub fn mlx_array_mean(
        handle: *mut mlx_array,
        axes: *const i32,
        axes_len: usize,
        keepdims: bool,
    ) -> *mut mlx_array;
    pub fn mlx_array_stack(handles: *const *mut mlx_array, len: usize, axis: i32)
    -> *mut mlx_array;
    pub fn mlx_array_clip(handle: *mut mlx_array, lo: f64, hi: f64) -> *mut mlx_array;
    pub fn mlx_array_minimum(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_maximum(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_add(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_sub(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_mul(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_div(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_add_scalar(handle: *mut mlx_array, value: f64) -> *mut mlx_array;
    pub fn mlx_array_mul_scalar(handle: *mut mlx_array, value: f64) -> *mut mlx_array;
    pub fn mlx_array_sub_scalar(handle: *mut mlx_array, value: f64) -> *mut mlx_array;
    pub fn mlx_array_div_scalar(handle: *mut mlx_array, value: f64) -> *mut mlx_array;
    pub fn mlx_array_matmul(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    // Fused addmm: D = beta * C + alpha * (A @ B)
    pub fn mlx_array_addmm(
        c: *mut mlx_array,
        a: *mut mlx_array,
        b: *mut mlx_array,
        alpha: f32,
        beta: f32,
    ) -> *mut mlx_array;

    // Fused SwiGLU MLP forward: output = down(silu(gate(x)) * up(x))
    // Weights are [out_features, in_features], transposed internally
    pub fn mlx_swiglu_mlp_forward(
        x: *mut mlx_array,
        w_gate: *mut mlx_array,
        w_up: *mut mlx_array,
        w_down: *mut mlx_array,
    ) -> *mut mlx_array;

    // E39: SwiGLU MLP with pre-stacked + pre-transposed weights
    //   w_gate_up_t: [hidden, 2*intermediate]
    //   w_down_t:    [intermediate, hidden]
    pub fn mlx_swiglu_mlp_forward_stacked(
        x: *mut mlx_array,
        w_gate_up_t: *mut mlx_array,
        w_down_t: *mut mlx_array,
    ) -> *mut mlx_array;

    // Fused Transformer Block forward (without KV cache)
    pub fn mlx_fused_transformer_block_forward(
        x: *mut mlx_array,
        input_norm_w: *mut mlx_array,
        post_attn_norm_w: *mut mlx_array,
        w_q: *mut mlx_array,
        w_k: *mut mlx_array,
        w_v: *mut mlx_array,
        w_o: *mut mlx_array,
        q_norm_w: *mut mlx_array,
        k_norm_w: *mut mlx_array,
        w_gate: *mut mlx_array,
        w_up: *mut mlx_array,
        w_down: *mut mlx_array,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        attn_scale: f32,
        rope_base: f32,
        rope_dims: i32,
        norm_eps: f32,
        qk_norm_eps: f32,
        use_causal: bool,
        rope_offset: i32,
    ) -> *mut mlx_array;

    // Fused Q/K/V projection with RoPE for cached attention
    pub fn mlx_fused_attention_qkv(
        x: *mut mlx_array,
        w_q: *mut mlx_array,
        w_k: *mut mlx_array,
        w_v: *mut mlx_array,
        q_norm_w: *mut mlx_array, // Can be null
        k_norm_w: *mut mlx_array, // Can be null
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rope_base: f32,
        rope_dims: i32,
        qk_norm_eps: f32,
        rope_offset: i32,
        q_out: *mut *mut mlx_array,
        k_out: *mut *mut mlx_array,
        v_out: *mut *mut mlx_array,
    );

    // Fused SDPA + output projection for cached attention
    pub fn mlx_fused_attention_output(
        q: *mut mlx_array,
        k: *mut mlx_array,
        v: *mut mlx_array,
        w_o: *mut mlx_array,
        n_heads: i32,
        head_dim: i32,
        attn_scale: f32,
        use_causal: bool,
    ) -> *mut mlx_array;

    pub fn mlx_array_transpose(
        handle: *mut mlx_array,
        axes: *const i32,
        axes_len: usize,
    ) -> *mut mlx_array;
    pub fn mlx_array_take(
        handle: *mut mlx_array,
        indices: *mut mlx_array,
        axis: i32,
    ) -> *mut mlx_array;
    pub fn mlx_array_take_along_axis(
        handle: *mut mlx_array,
        indices: *mut mlx_array,
        axis: i32,
    ) -> *mut mlx_array;
    pub fn mlx_array_put_along_axis(
        handle: *mut mlx_array,
        indices: *mut mlx_array,
        values: *mut mlx_array,
        axis: i32,
    ) -> *mut mlx_array;
    pub fn mlx_array_arange(start: f64, stop: f64, step: f64, dtype: i32) -> *mut mlx_array;
    pub fn mlx_array_linspace(
        start: f64,
        stop: f64,
        num: i32,
        dtype: i32,
        has_dtype: bool,
    ) -> *mut mlx_array;
    pub fn mlx_array_eye(n: i32, m: i32, k: i32, dtype: i32, has_dtype: bool) -> *mut mlx_array;
    pub fn mlx_array_slice(
        handle: *mut mlx_array,
        starts: *const i64,
        stops: *const i64,
        ndim: usize,
    ) -> *mut mlx_array;
    // Optimized slice assignment functions - no shape allocation
    pub fn mlx_array_slice_assign_axis(
        src_handle: *mut mlx_array,
        update_handle: *mut mlx_array,
        axis: usize,
        start: i64,
        end: i64,
    ) -> *mut mlx_array;
    pub fn mlx_array_slice_assign_axis_inplace(
        src_handle: *mut mlx_array,
        update_handle: *mut mlx_array,
        axis: usize,
        start: i64,
        end: i64,
    );
    // Optimized slice along a single axis - no shape allocation
    pub fn mlx_array_slice_axis(
        src_handle: *mut mlx_array,
        axis: usize,
        start: i64,
        end: i64,
    ) -> *mut mlx_array;
    pub fn mlx_array_concatenate(
        handles: *const *mut mlx_array,
        len: usize,
        axis: i32,
    ) -> *mut mlx_array;
    pub fn mlx_array_sort(handle: *mut mlx_array, axis: i32, has_axis: bool) -> *mut mlx_array;
    pub fn mlx_array_argsort(handle: *mut mlx_array, axis: i32, has_axis: bool) -> *mut mlx_array;
    pub fn mlx_array_partition(
        handle: *mut mlx_array,
        kth: i32,
        axis: i32,
        has_axis: bool,
    ) -> *mut mlx_array;
    pub fn mlx_array_argpartition(
        handle: *mut mlx_array,
        kth: i32,
        axis: i32,
        has_axis: bool,
    ) -> *mut mlx_array;
    pub fn mlx_array_eval(handle: *mut mlx_array);
    pub fn mlx_async_eval(handles: *mut *mut mlx_array, count: usize);
    pub fn mlx_eval(handles: *mut *mut mlx_array, count: usize) -> bool;
    pub fn mlx_array_size(handle: *mut mlx_array) -> usize;
    pub fn mlx_array_ndim(handle: *mut mlx_array) -> usize;
    pub fn mlx_array_shape(handle: *mut mlx_array, out: *mut i64);
    pub fn mlx_array_shape_at(handle: *mut mlx_array, axis: usize) -> i64;
    pub fn mlx_array_get_batch_seq_len(
        handle: *mut mlx_array,
        batch: *mut i64,
        seq_len: *mut i64,
    ) -> bool;
    pub fn mlx_array_get_batch_seq_hidden(
        handle: *mut mlx_array,
        batch: *mut i64,
        seq_len: *mut i64,
        hidden: *mut i64,
    ) -> bool;
    pub fn mlx_array_item_at_int32(handle: *mut mlx_array, index: usize, out: *mut i32) -> bool;
    pub fn mlx_array_item_at_uint32(handle: *mut mlx_array, index: usize, out: *mut u32) -> bool;
    pub fn mlx_array_item_at_float32(handle: *mut mlx_array, index: usize, out: *mut f32) -> bool;
    pub fn mlx_array_dtype(handle: *mut mlx_array) -> i32;
    pub fn mlx_array_to_float32(handle: *mut mlx_array, out: *mut f32, len: usize) -> bool;
    pub fn mlx_array_to_int32(handle: *mut mlx_array, out: *mut i32, len: usize) -> bool;
    pub fn mlx_array_to_uint32(handle: *mut mlx_array, out: *mut u32, len: usize) -> bool;
    pub fn mlx_array_to_uint8(handle: *mut mlx_array, out: *mut u8, len: usize) -> bool;
    pub fn mlx_array_to_int8(handle: *mut mlx_array, out: *mut i8, len: usize) -> bool;
    pub fn mlx_array_to_uint16(handle: *mut mlx_array, out: *mut u16, len: usize) -> bool;
    /// Read a raw byte range straight from a safetensors file into `out`,
    /// bypassing MLX (no array constructed). `file_offset` is the absolute byte
    /// offset of the tensor data within the file; `out_len` is the tensor byte
    /// length. The read is chunked internally. Returns false on any I/O error,
    /// short read, or out-of-range request.
    pub fn mlx_safetensor_read_raw(
        file_path: *const std::os::raw::c_char,
        file_offset: u64,
        out: *mut u8,
        out_len: usize,
    ) -> bool;
    pub fn mlx_array_delete(arr: *mut mlx_array);
    pub fn mlx_synchronize();
    pub fn mlx_clear_cache();
    pub fn mlx_compile_clear_cache() -> bool;
    pub fn mlx_stop_gradient(a: *mut mlx_array) -> *mut mlx_array;

    // Random number generation
    pub fn mlx_array_random_uniform(
        shape: *const i64,
        ndim: usize,
        low: f32,
        high: f32,
        dtype: i32,
    ) -> *mut mlx_array;
    pub fn mlx_array_random_normal(
        shape: *const i64,
        ndim: usize,
        mean: f32,
        std: f32,
        dtype: i32,
    ) -> *mut mlx_array;
    pub fn mlx_array_random_bernoulli(shape: *const i64, ndim: usize, prob: f32) -> *mut mlx_array;
    pub fn mlx_array_randint(shape: *const i64, ndim: usize, low: i32, high: i32)
    -> *mut mlx_array;
    pub fn mlx_array_categorical(handle: *mut mlx_array, axis: i32) -> *mut mlx_array;

    // Gradient computation (callback-based - this is the MLX-native approach)
    pub fn mlx_compute_gradients(
        loss_fn: LossFunctionPtr,
        context: *mut std::os::raw::c_void,
        input_handles: *const *mut mlx_array,
        input_count: usize,
        output_handles: *mut *mut mlx_array,
    ) -> usize;

    pub fn mlx_value_and_gradients(
        loss_fn: LossFunctionPtr,
        context: *mut std::os::raw::c_void,
        input_handles: *const *mut mlx_array,
        input_count: usize,
        loss_handle: *mut *mut mlx_array,
        grad_handles: *mut *mut mlx_array,
    ) -> usize;

    // Gradient checkpointing
    pub fn mlx_checkpoint_apply(
        layer_fn: LayerFunctionPtr,
        context: *mut std::os::raw::c_void,
        input_handles: *const *mut mlx_array,
        input_count: usize,
        output_handles: *mut *mut mlx_array,
        max_outputs: usize,
    ) -> usize;

    // Comparison operations
    pub fn mlx_array_equal(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_not_equal(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_less(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_less_equal(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_greater(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_greater_equal(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;

    // Logical operations
    pub fn mlx_array_logical_and(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_logical_or(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_logical_not(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_where(
        condition: *mut mlx_array,
        x: *mut mlx_array,
        y: *mut mlx_array,
    ) -> *mut mlx_array;

    // Advanced reduction operations
    pub fn mlx_array_argmax(handle: *mut mlx_array, axis: i32, keepdims: bool) -> *mut mlx_array;
    pub fn mlx_array_argmin(handle: *mut mlx_array, axis: i32, keepdims: bool) -> *mut mlx_array;
    pub fn mlx_array_max(
        handle: *mut mlx_array,
        axes: *const i32,
        axes_len: usize,
        keepdims: bool,
    ) -> *mut mlx_array;
    pub fn mlx_array_min(
        handle: *mut mlx_array,
        axes: *const i32,
        axes_len: usize,
        keepdims: bool,
    ) -> *mut mlx_array;
    pub fn mlx_array_prod(
        handle: *mut mlx_array,
        axes: *const i32,
        axes_len: usize,
        keepdims: bool,
    ) -> *mut mlx_array;
    pub fn mlx_array_var(
        handle: *mut mlx_array,
        axes: *const i32,
        axes_len: usize,
        keepdims: bool,
        ddof: i32,
    ) -> *mut mlx_array;
    pub fn mlx_array_std(
        handle: *mut mlx_array,
        axes: *const i32,
        axes_len: usize,
        keepdims: bool,
        ddof: i32,
    ) -> *mut mlx_array;
    pub fn mlx_array_cumsum(handle: *mut mlx_array, axis: i32) -> *mut mlx_array;
    pub fn mlx_array_cumprod(handle: *mut mlx_array, axis: i32) -> *mut mlx_array;

    // Array manipulation operations
    pub fn mlx_array_pad(
        handle: *mut mlx_array,
        pad_width: *const i32,
        ndim: usize,
        constant_value: f32,
    ) -> *mut mlx_array;
    pub fn mlx_array_roll(handle: *mut mlx_array, shift: i32, axis: i32) -> *mut mlx_array;
    pub fn mlx_array_split(
        handle: *mut mlx_array,
        indices_or_sections: i32,
        axis: i32,
    ) -> *mut mlx_array;
    pub fn mlx_array_split_multi(
        handle: *mut mlx_array,
        indices_or_sections: i32,
        axis: i32,
        out_handles: *mut u64,
        max_outputs: usize,
    ) -> usize;
    pub fn mlx_array_tile(
        handle: *mut mlx_array,
        reps: *const i32,
        reps_len: usize,
    ) -> *mut mlx_array;
    pub fn mlx_array_repeat(handle: *mut mlx_array, repeats: i32, axis: i32) -> *mut mlx_array;
    pub fn mlx_array_squeeze(
        handle: *mut mlx_array,
        axes: *const i32,
        axes_len: usize,
    ) -> *mut mlx_array;
    pub fn mlx_array_expand_dims(handle: *mut mlx_array, axis: i32) -> *mut mlx_array;
    pub fn mlx_array_broadcast_to(
        handle: *mut mlx_array,
        shape: *const i64,
        ndim: usize,
    ) -> *mut mlx_array;

    // Additional math operations
    pub fn mlx_array_abs(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_negative(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_sign(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_sqrt(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_square(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_power(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_sin(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_cos(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_tan(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_sinh(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_cosh(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_tanh(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_erf(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_floor(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_ceil(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_round(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_floor_divide(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_remainder(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_reciprocal(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_arcsin(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_arccos(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_arctan(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_log10(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_log2(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_log1p(handle: *mut mlx_array) -> *mut mlx_array;

    // NaN/Inf checking operations (GPU-native)
    pub fn mlx_array_isnan(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_isinf(handle: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_isfinite(handle: *mut mlx_array) -> *mut mlx_array;

    // Create scalar with specific dtype (no AsType node)
    pub fn mlx_array_scalar_float_dtype(value: f64, dtype: i32) -> *mut mlx_array;

    // Compiled GELU approximate (fused kernel, matches Python nn.gelu_approx)
    pub fn mlx_gelu_approx(handle: *mut mlx_array) -> *mut mlx_array;
    // Compiled GeGLU: gelu_approx(gate) * up (fused kernel)
    pub fn mlx_geglu(gate: *mut mlx_array, up: *mut mlx_array) -> *mut mlx_array;
    // Compiled logit softcap: tanh(x / softcap) * softcap (fused kernel)
    pub fn mlx_logit_softcap(x: *mut mlx_array, softcap: *mut mlx_array) -> *mut mlx_array;

    // Fast operations (mlx::fast namespace)
    pub fn mlx_fast_rope(
        handle: *mut mlx_array,
        dims: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: i32,
    ) -> *mut mlx_array;
    /// fast::rope with array offset and optional precomputed freqs.
    /// When freqs is non-null, base is ignored (pass 0.0).
    /// freqs must be 1-D with shape [dims/2].
    pub fn mlx_fast_rope_with_freqs(
        handle: *mut mlx_array,
        dims: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: *mut mlx_array,
        freqs: *mut mlx_array,
    ) -> *mut mlx_array;
    pub fn mlx_fast_scaled_dot_product_attention(
        queries: *mut mlx_array,
        keys: *mut mlx_array,
        values: *mut mlx_array,
        scale: f32,
        mask_mode: *const std::os::raw::c_char,
        mask: *mut mlx_array,
        has_mask: bool,
    ) -> *mut mlx_array;
    pub fn mlx_fast_rms_norm(
        x: *mut mlx_array,
        weight: *mut mlx_array, // nullable
        eps: f32,
    ) -> *mut mlx_array;
    pub fn mlx_fast_layer_norm(
        x: *mut mlx_array,
        weight: *mut mlx_array, // nullable
        bias: *mut mlx_array,   // nullable
        eps: f32,
    ) -> *mut mlx_array;
    pub fn mlx_compiled_sample_full(
        logits: *mut mlx_array,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        sampler_mode: i32,
    ) -> *mut mlx_array;

    /// Return the normalized probability distribution the compiled sampler
    /// (`mlx_compiled_sample_full`) draws from, using the SAME filter chain and
    /// `sampler_mode`. Output is `softmax(filtered_logits * inv_temp)` over the
    /// last axis (filtered-out tokens are exactly 0). Stochastic MTP acceptance
    /// consumes this as the proposal density `q` (draft logits) and target
    /// density `p` (verify logits) so accept/reject + residual resampling match
    /// the draw distribution by construction. At temperature == 0 it returns a
    /// one-hot argmax distribution (callers must not rely on it at T=0 — the
    /// accept path takes its argmax-only shortcut and ignores q/p).
    pub fn mlx_compiled_sampling_distribution(
        logits: *mut mlx_array,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        sampler_mode: i32,
    ) -> *mut mlx_array;

    /// Compiled sampling using mlx::core::compile for the categorical step
    /// This matches mlx-lm's @partial(mx.compile, ...) approach
    pub fn mlx_compiled_sample_and_logprobs(
        logits: *mut mlx_array,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        sampler_mode: i32,
        out_token: *mut *mut mlx_array,
        out_logprobs: *mut *mut mlx_array,
    );

    // Stream operations
    pub fn mlx_default_stream(device_type: i32) -> mlx_stream;
    pub fn mlx_new_stream(device_type: i32) -> mlx_stream;
    pub fn mlx_set_default_stream(stream: mlx_stream);
    pub fn mlx_stream_synchronize(stream: mlx_stream);
    pub fn mlx_default_device() -> i32;
    pub fn mlx_set_default_device(device_type: i32);

    // Metal operations (for memory management).
    //
    // Fallible-FFI contract: every memory accessor below returns `i32`
    // (0 = success, -1 = caught C++ exception) and writes its measurement
    // through a caller-supplied out-pointer. This replaces the prior
    // "ambiguous sentinel `0`" contract: a real measurement of zero bytes
    // is now distinguishable from a caught exception. See
    // `crates/mlx-sys/src/mlx_stream.cpp` and `mlx_paged_profile.cpp`
    // for the C++-side rationale (lazy `metal::allocator()` construction
    // throws on no-Metal hosts; without the catch-all the unwind crosses
    // the FFI boundary and aborts the process). Pass null `out_*` to
    // ignore the value.
    pub fn mlx_metal_is_available() -> bool;
    pub fn mlx_metal_device_info() -> *const std::os::raw::c_char;
    pub fn mlx_set_wired_limit(limit: u64, out_old_limit: *mut u64) -> i32;
    pub fn mlx_get_peak_memory(out_value: *mut u64) -> i32;
    pub fn mlx_get_active_memory(out_value: *mut u64) -> i32;
    pub fn mlx_get_cache_memory(out_value: *mut u64) -> i32;
    pub fn mlx_reset_peak_memory() -> i32;
    pub fn mlx_set_memory_limit(limit: u64, out_old_limit: *mut u64) -> i32;
    pub fn mlx_get_memory_limit(out_value: *mut u64) -> i32;
    pub fn mlx_set_cache_limit(limit: u64, out_old_limit: *mut u64) -> i32;
    pub fn mlx_array_nbytes(handle: *mut mlx_array) -> usize;

    // Total physical system memory in bytes (Apple Silicon: unified memory
    // shared with the GPU). On macOS this matches `device_info()["memory_size"]`
    // and is sourced from `sysctlbyname("hw.memsize")` so it works
    // regardless of Metal availability. Returns 0 on success (with value
    // written to `out_value`), -1 if sysctl fails / non-macOS host. The
    // Rust profile.rs auto-sizer surfaces ProfileError on -1.
    pub fn mlx_total_system_memory(out_value: *mut u64) -> i32;

    // GPU-visible working-set bound (`MTLDevice
    // recommendedMaxWorkingSetSize`). Returns 0 on success (writes value
    // through `out_value`); returns -1 if Metal unavailable, device_info
    // is missing the entry, or an exception is caught.
    pub fn mlx_max_recommended_working_set_size(out_value: *mut u64) -> i32;

    // Wrap an existing MTL::Buffer (`void*` MTLBuffer pointer — the same
    // shape `mlx_array_get_metal_buffer` returns) as an MLX `array` view.
    // Zero-copy: the returned array retains the underlying MTL::Buffer and
    // releases that retain when the array is dropped, so the view survives the
    // original Rust buffer holder.
    //
    // - `metal_buffer_ptr`: MTL::Buffer* as `void*`
    // - `dims`/`ndim`: view shape (caller validates element-count vs
    //   buffer length)
    // - `dtype_code`: BridgeDType (FLOAT16/BFLOAT16/UINT8/etc.)
    // Returns null on unsupported platforms / null buffer / invalid args.
    pub fn mlx_array_from_metal_buffer_view(
        metal_buffer_ptr: *mut std::ffi::c_void,
        dims: *const i64,
        ndim: usize,
        dtype_code: i32,
    ) -> *mut mlx_array;

    // Fused forward step - single FFI call for entire forward pass
    // This reduces FFI overhead from ~300 calls to 1 call per token
    // Uses array offsets for batched generation with proper per-sequence RoPE positions.
    pub fn mlx_qwen3_forward_step(
        // Input
        input_ids: *mut mlx_array, // [batch, seq_len]
        // Model weights
        embedding_weight: *mut mlx_array,     // [vocab, hidden]
        layer_weights: *const *mut mlx_array, // [num_layers * 11]
        num_layers: i32,
        final_norm_weight: *mut mlx_array, // [hidden]
        lm_head_weight: *mut mlx_array,    // null if tied
        tie_word_embeddings: bool,
        // Model config
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_theta: f32,
        norm_eps: f32,
        // KV cache inputs (null for prefill without cache)
        kv_keys_in: *const *mut mlx_array,   // [num_layers] or null
        kv_values_in: *const *mut mlx_array, // [num_layers] or null
        cache_idx_in: i32,                   // Shared cache write position
        // Array offsets for batched generation
        rope_offsets: *mut mlx_array, // [batch] - per-sequence RoPE offsets
        left_padding: *mut mlx_array, // [batch] - left padding amounts
        // Outputs
        out_logits: *mut *mut mlx_array,    // [batch, seq_len, vocab]
        out_kv_keys: *mut *mut mlx_array,   // [num_layers]
        out_kv_values: *mut *mut mlx_array, // [num_layers]
        out_cache_idx: *mut i32,            // Updated write position
    );

}

// ============================================================================
// Metal Buffer Extraction FFI
// ============================================================================
//
// These functions extract Metal buffer pointers from MLX arrays for use
// with external Metal kernel dispatch (e.g., Rust metal crate).
//
// IMPORTANT: Only valid when Metal backend is available (macOS with GPU).
// On CPU-only builds or non-macOS platforms, buffer pointers are NOT MTLBuffer*.
//
// Note: mlx_metal_is_available() is already declared earlier in this file.

unsafe extern "C-unwind" {
    /// Get the raw Metal buffer pointer from an MLX array
    /// Returns the MTLBuffer* as a void* for FFI compatibility
    /// Returns nullptr if:
    ///   - handle is null
    ///   - Metal/GPU is not available (buffer would not be MTLBuffer*)
    ///   - array has no data
    pub fn mlx_array_get_metal_buffer(handle: *mut mlx_array) -> *mut std::ffi::c_void;

    /// Get the byte offset into the Metal buffer for this array
    /// This is needed for sliced/strided arrays that share a buffer
    /// Note: Returns bytes (MLX's offset() is already in bytes)
    pub fn mlx_array_get_buffer_offset(handle: *mut mlx_array) -> usize;

    /// Get the data size of the array in number of ELEMENTS (not bytes)
    /// To get bytes, multiply by itemsize from mlx_array_get_itemsize()
    pub fn mlx_array_get_data_size(handle: *mut mlx_array) -> usize;

    /// Get the item size in bytes for the array's dtype
    pub fn mlx_array_get_itemsize(handle: *mut mlx_array) -> usize;

    /// Synchronize - ensure all MLX operations are complete
    /// Call this before dispatching external Metal kernels
    pub fn mlx_metal_synchronize();

}

// ================================================================================
// Paged ops test helpers (defined in `mlx_paged_ops.cpp`)
//
// These exist solely so the unit tests in
// `crates/mlx-paged-attn/tests/paged_ops_smoke.rs` can exercise the
// C++ `PagedKVWrite` / `PagedAttention` primitives' `is_equivalent`
// and `vjp` semantics without standing up a separate C++ test runner.
// ================================================================================

unsafe extern "C-unwind" {
    /// Emit the MLX C++ `paged_attention(...)` Custom primitive and return the
    /// lazy on-device output array. Returns null for bridge/factory validation
    /// errors so callers can fall back to a conservative attention path. GPU
    /// dispatch errors still occur later when MLX evaluates the returned array.
    #[allow(clippy::too_many_arguments)]
    pub fn mlx_paged_attention_forward(
        q: *mut mlx_array,
        k_pool: *mut mlx_array,
        v_pool: *mut mlx_array,
        block_table: *mut mlx_array,
        seq_lens: *mut mlx_array,
        k_scale: *mut mlx_array,
        v_scale: *mut mlx_array,
        scale: f32,
        softcap: f32,
        sliding_window: i32,
        block_size: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_size: i32,
        kv_dtype: u8,
    ) -> *mut mlx_array;

    /// Emit the MLX C++ `paged_kv_write(...)` Custom primitive and return the
    /// lazy K/V pool output arrays through `out_k_pool` / `out_v_pool`.
    /// Returns false for bridge/factory validation errors.
    #[allow(clippy::too_many_arguments)]
    pub fn mlx_paged_kv_write_forward(
        k_pool: *mut mlx_array,
        v_pool: *mut mlx_array,
        new_k: *mut mlx_array,
        new_v: *mut mlx_array,
        slot_mapping: *mut mlx_array,
        k_scale: *mut mlx_array,
        v_scale: *mut mlx_array,
        block_size: i32,
        num_kv_heads: i32,
        head_size: i32,
        kv_dtype: u8,
        out_k_pool: *mut *mut mlx_array,
        out_v_pool: *mut *mut mlx_array,
    ) -> bool;

    /// Compare two `PagedKVWrite` primitives via `is_equivalent`.
    /// Returns `true` iff both are equivalent (same scalar state).
    pub fn mlx_paged_kv_write_is_equivalent(
        block_size_lhs: i32,
        num_kv_heads_lhs: i32,
        head_size_lhs: i32,
        x_pack_lhs: i32,
        kv_dtype_lhs: u8,
        block_size_rhs: i32,
        num_kv_heads_rhs: i32,
        head_size_rhs: i32,
        x_pack_rhs: i32,
        kv_dtype_rhs: u8,
    ) -> bool;

    /// Returns 1 iff `PagedKVWrite::vjp` throws `std::runtime_error`,
    /// 0 otherwise. The test asserts on the value `1`.
    pub fn mlx_paged_kv_write_vjp_throws() -> i32;

    /// Returns 1 iff `PagedAttention::vjp` throws `std::runtime_error`,
    /// 0 otherwise.
    pub fn mlx_paged_attention_vjp_throws() -> i32;

    /// Compare two `PagedAttention` primitives via `is_equivalent`.
    #[allow(clippy::too_many_arguments)]
    pub fn mlx_paged_attention_is_equivalent(
        scale_lhs: f32,
        softcap_lhs: f32,
        block_size_lhs: i32,
        num_q_heads_lhs: i32,
        num_kv_heads_lhs: i32,
        head_size_lhs: i32,
        sliding_window_lhs: i32,
        kv_dtype_lhs: u8,
        scale_rhs: f32,
        softcap_rhs: f32,
        block_size_rhs: i32,
        num_q_heads_rhs: i32,
        num_kv_heads_rhs: i32,
        head_size_rhs: i32,
        sliding_window_rhs: i32,
        kv_dtype_rhs: u8,
    ) -> bool;

    /// Verify `PagedAttention::output_shapes` reports
    /// `{q_num_tokens, num_q_heads, head_size}` from the primitive's
    /// scalar state (NOT from q's trailing dims). Writes the resulting
    /// shape (3 elements) to `out_shape` and returns the number of
    /// dimensions on the returned shape.
    #[allow(clippy::too_many_arguments)]
    pub fn mlx_paged_attention_test_output_shapes(
        q_num_tokens: i32,
        q_dim1_actual: i32,
        q_dim2_actual: i32,
        scale: f32,
        softcap: f32,
        block_size: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_size: i32,
        sliding_window: i32,
        kv_dtype_raw: u8,
        out_shape: *mut i32,
    ) -> i32;

    /// Returns 1 iff `paged_attention(...)` (the public factory)
    /// throws `std::invalid_argument` when called with
    /// `sliding_window=512`, 0 otherwise.
    pub fn mlx_paged_attention_factory_rejects_sliding_window() -> i32;

    /// Returns 1 iff `paged_attention(...)` throws when q's trailing
    /// dims disagree with the primitive's scalar state.
    pub fn mlx_paged_attention_factory_rejects_q_shape_mismatch() -> i32;

    /// Returns 1 iff `paged_kv_write(...)` throws when the K-pool's
    /// interior dims disagree with the primitive's scalar state.
    pub fn mlx_paged_kv_write_factory_rejects_pool_shape_mismatch() -> i32;

    // =============================================================================
    // Negative-validation FFI declarations.
    //
    // Each helper constructs `paged_attention` / `paged_kv_write` factory
    // inputs that are well-formed EXCEPT for one specific dim or dtype,
    // calls the factory, and returns 1 iff `std::invalid_argument` was
    // thrown. The Rust unit tests assert the value `1`.
    // =============================================================================

    /// q rank != 3 must be rejected.
    pub fn mlx_paged_attention_factory_rejects_q_rank_not_3() -> i32;

    /// block_table.shape(0) != q.shape(0) must be rejected.
    pub fn mlx_paged_attention_factory_rejects_block_table_batch_mismatch() -> i32;

    /// block_table dtype != int32 must be rejected.
    pub fn mlx_paged_attention_factory_rejects_block_table_dtype() -> i32;

    /// seq_lens.shape(0) != q.shape(0) must be rejected.
    pub fn mlx_paged_attention_factory_rejects_seq_lens_batch_mismatch() -> i32;

    /// k_pool.shape(2) != head_size / x_pack must be rejected.
    pub fn mlx_paged_attention_factory_rejects_k_pool_inner_dim() -> i32;

    /// k_pool.shape(4) != x_pack must be rejected.
    pub fn mlx_paged_attention_factory_rejects_k_pool_x_pack() -> i32;

    /// v_pool.shape(2) != head_size must be rejected.
    pub fn mlx_paged_attention_factory_rejects_v_pool_head_dim() -> i32;

    /// k_pool.shape(0) != v_pool.shape(0) must be rejected (num_blocks mismatch).
    pub fn mlx_paged_attention_factory_rejects_num_blocks_mismatch() -> i32;

    /// slot_mapping rank != 1 must be rejected.
    pub fn mlx_paged_kv_write_factory_rejects_slot_mapping_rank() -> i32;

    /// slot_mapping dtype != int64 must be rejected.
    pub fn mlx_paged_kv_write_factory_rejects_slot_mapping_dtype() -> i32;

    /// slot_mapping length != new_k.shape(0) must be rejected.
    pub fn mlx_paged_kv_write_factory_rejects_slot_mapping_length() -> i32;

    /// slot_mapping with a max value >= num_blocks * block_size must be
    /// rejected (safety check; eval-based bounds verification).
    pub fn mlx_paged_kv_write_factory_rejects_slot_mapping_out_of_range() -> i32;

    /// Assert the factory-side slot_mapping bounds guard's
    /// `std::invalid_argument` message contains the `[runtime]` marker
    /// (matching the eval_gpu-side guard for the same data-dependent
    /// property). Return-code contract (see C++ helper): 1 = passes
    /// (threw + marker present), 0 = no throw, -1 = wrong exception class
    /// / setup error, -2 = threw but `[runtime]` marker missing.
    pub fn mlx_paged_kv_write_factory_slot_mapping_out_of_range_runtime_marker() -> i32;

    // =============================================================================
    // dtype-mismatch FFI declarations.
    //
    // Each helper constructs `paged_attention` / `paged_kv_write` factory
    // inputs that are well-formed EXCEPT for one specific dtype slot
    // (q, k_pool, v_pool, new_k, or new_v) that disagrees with the
    // dtype implied by `kv_dtype`. The factory MUST reject by throwing
    // `std::invalid_argument`. Returns 1 on rejection, 0 otherwise.
    // =============================================================================

    /// k_pool dtype != bfloat16 must be rejected for kv_dtype=Bf16.
    pub fn mlx_paged_kv_write_factory_rejects_k_pool_dtype_bf16() -> i32;

    /// v_pool dtype != bfloat16 must be rejected for kv_dtype=Bf16.
    pub fn mlx_paged_kv_write_factory_rejects_v_pool_dtype_bf16() -> i32;

    /// new_k dtype != bfloat16 must be rejected for kv_dtype=Bf16.
    pub fn mlx_paged_kv_write_factory_rejects_new_k_dtype_bf16() -> i32;

    /// new_v dtype != bfloat16 must be rejected for kv_dtype=Bf16.
    pub fn mlx_paged_kv_write_factory_rejects_new_v_dtype_bf16() -> i32;

    /// k_pool dtype != uint8 must be rejected for kv_dtype=Fp8.
    pub fn mlx_paged_kv_write_factory_rejects_k_pool_dtype_fp8() -> i32;

    /// new_k dtype != bfloat16 must be rejected for kv_dtype=Fp8
    /// (FP8 io dtype is bfloat16).
    pub fn mlx_paged_kv_write_factory_rejects_new_k_dtype_fp8() -> i32;

    /// q dtype != bfloat16 must be rejected for kv_dtype=Bf16.
    pub fn mlx_paged_attention_factory_rejects_q_dtype_bf16() -> i32;

    /// q dtype != bfloat16 must be rejected for kv_dtype=Fp8 (FP8 io
    /// dtype is bfloat16).
    pub fn mlx_paged_attention_factory_rejects_q_dtype_fp8() -> i32;

    /// k_pool dtype != bfloat16 must be rejected for kv_dtype=Bf16.
    pub fn mlx_paged_attention_factory_rejects_k_pool_dtype_bf16() -> i32;

    /// k_pool dtype != uint8 must be rejected for kv_dtype=Fp8.
    pub fn mlx_paged_attention_factory_rejects_k_pool_dtype_fp8() -> i32;

    // =============================================================================
    // GQA head-group divisibility.
    //
    // The Metal kernel computes `num_queries_per_kv = num_heads /
    // num_kv_heads` and then `kv_head_idx = head_idx /
    // num_queries_per_kv`. A num_kv_heads of 0, num_q_heads <
    // num_kv_heads, or non-divisible grouping triggers division by
    // zero or out-of-pool K/V reads. The factory must reject all three.
    // Each helper returns 1 iff `std::invalid_argument` was thrown.
    // =============================================================================

    /// num_kv_heads = 0 must be rejected.
    pub fn mlx_paged_attention_factory_rejects_zero_kv_heads() -> i32;

    /// num_q_heads (2) < num_kv_heads (4) must be rejected (kernel
    /// would divide by zero on `kv_head_idx = head_idx /
    /// num_queries_per_kv` because `num_queries_per_kv = 0`).
    pub fn mlx_paged_attention_factory_rejects_q_heads_less_than_kv_heads() -> i32;

    /// num_q_heads (6) not divisible by num_kv_heads (4) must be
    /// rejected (later heads would compute kv_head_idx outside the
    /// KV-head pool dimension).
    pub fn mlx_paged_attention_factory_rejects_indivisible_grouping() -> i32;

    /// block_size = 0 must be rejected. Without this check the pool
    /// shape equality accepts a zero-sized pool block dim when
    /// block_size=0, and `eval_gpu`'s bounds check then divides by
    /// zero in host code (`(s + block_size - 1) / block_size`).
    pub fn mlx_paged_attention_factory_rejects_zero_block_size() -> i32;

    /// head_size = 0 must be rejected. The Metal kernel uses head_size
    /// as a grid extent and indexing stride; a zero-sized inner dim
    /// would set up a degenerate Metal launch. Mirrors
    /// `paged_kv_write`'s identical check for symmetry.
    pub fn mlx_paged_attention_factory_rejects_zero_head_size() -> i32;

    /// Compile a `paged_kv_write`-emitting function, call it once with a
    /// valid slot_mapping (cache miss, factory check passes), then call
    /// it again with an out-of-range slot_mapping. The cache HIT bypasses
    /// the factory; `PagedKVWrite::eval_gpu`'s own bounds check MUST
    /// throw `std::invalid_argument` on the second eval.
    ///
    /// Return codes:
    ///   1   — second-call eval threw `std::invalid_argument` (fix
    ///         working).
    ///   0   — second-call eval did NOT throw (regression).
    ///  -1   — internal/setup error.
    ///  -3   — Metal not available; eval-based verification skipped.
    pub fn mlx_paged_kv_write_compile_cached_oob_throws() -> i32;

    // =============================================================================
    // PagedAttention::eval_gpu must runtime-bounds-check `seq_lens` and
    // `block_table` contents.
    //
    // Each helper builds REAL data-backed arrays with exactly one bad
    // input value, calls `paged_attention(...)`, and evals the result.
    // Returns 1 if eval throws `std::invalid_argument`, 0 if no throw,
    // -1 on setup error, -3 if Metal is unavailable.
    // =============================================================================

    /// seq_lens with a negative entry must be rejected by eval_gpu.
    pub fn mlx_paged_attention_eval_gpu_rejects_negative_seq_len() -> i32;

    /// seq_lens larger than `max_blocks_per_seq * block_size` must be
    /// rejected by eval_gpu.
    pub fn mlx_paged_attention_eval_gpu_rejects_oversized_seq_len() -> i32;

    /// block_table with a negative entry (within the row's used region)
    /// must be rejected by eval_gpu.
    pub fn mlx_paged_attention_eval_gpu_rejects_negative_block_id() -> i32;

    /// block_table with an entry == num_blocks (one past valid) must be
    /// rejected by eval_gpu.
    pub fn mlx_paged_attention_eval_gpu_rejects_oob_block_id() -> i32;

    /// Compile a `paged_attention`-emitting function, call it with
    /// valid inputs (cache miss → factory + eval_gpu pass), then call
    /// it again with an out-of-range block id (cache HIT bypasses the
    /// factory). The eval_gpu runtime bounds check MUST throw on the
    /// second eval.
    ///
    /// Return codes:
    ///   1   — second-call eval threw `std::invalid_argument` (fix
    ///         working — the compile-cached path is bounds-checked).
    ///   0   — second-call eval did NOT throw (regression).
    ///  -1   — internal/setup error (first call failed unexpectedly).
    ///  -3   — Metal not available; eval-based verification skipped.
    pub fn mlx_paged_attention_compile_cached_oob_throws() -> i32;

    /// Reset the compile-trace counter to 0 before exercising the
    /// `mlx_paged_kv_write_compile_trace_smoke` helper.
    pub fn mlx_paged_kv_write_trace_count_reset();

    /// Read the compile-trace counter. Each cache-miss inside the
    /// MLX compile cache increments it once via the trace function.
    pub fn mlx_paged_kv_write_trace_count_get() -> i32;

    /// Build a `mlx::core::compile`-wrapped function around an
    /// internal trace function that emits a `paged_kv_write`
    /// primitive, call it twice with REAL data-backed inputs that
    /// share shapes/dtypes but differ in contents, and return the
    /// number of times the inner trace ran.
    ///
    /// Beyond counting traces, the helper EVALS each call's outputs
    /// and inspects the second call's K-pool slots after eval. This
    /// proves both that the cache HIT (counter==1) AND that runtime
    /// contents flow through `compile_replace` correctly (the second
    /// call's K bytes appear at the second call's slot positions, not
    /// the first call's).
    ///
    /// Return codes:
    ///   `count` (>=0) — trace counter at end (1 on success).
    ///   -1            — internal/setup error.
    ///   -2            — second-call slots did NOT contain second-call
    ///                   K values (compile_replace runtime-thread bug).
    ///   -3            — Metal not available; eval-based verification
    ///                   skipped (trace-count check still ran).
    pub fn mlx_paged_kv_write_compile_trace_smoke(num_tokens: i32) -> i32;

    // =============================================================================
    // Factory must reject non-row-contiguous or nonzero-offset views for
    // ALL inputs.
    //
    // Each helper builds a real data-backed array, applies `slice` /
    // `transpose` to produce a non-row-contiguous or nonzero-offset
    // view, then calls the public factory. Returns 1 if
    // `std::invalid_argument` is thrown, 0 otherwise. Returns -3 if
    // Metal is unavailable (the slice/transpose eval needs it).
    // =============================================================================

    /// k_pool sliced along axis 0 (`pool[1:5]`) must be rejected by the
    /// `paged_kv_write` factory.
    pub fn mlx_paged_kv_write_factory_rejects_non_contiguous_k_pool() -> i32;

    /// q transposed from `[64, 8, 1]` → `[1, 8, 64]` must be rejected
    /// by the `paged_attention` factory (right shape, wrong stride
    /// order — row_contiguous == false).
    pub fn mlx_paged_attention_factory_rejects_non_contiguous_q() -> i32;

    /// block_table sliced along axis 0 (`bt[1:3]`) must be rejected by
    /// the `paged_attention` factory.
    pub fn mlx_paged_attention_factory_rejects_non_contiguous_block_table() -> i32;

    /// seq_lens sliced as `seq_lens[1:]` (nonzero offset) must be
    /// rejected by the `paged_attention` factory.
    pub fn mlx_paged_attention_factory_rejects_non_contiguous_seq_lens() -> i32;

    /// Production FFI bridge must reject non-contiguous metadata instead
    /// of hiding it behind lazy `contiguous(...)` metadata copies.
    pub fn mlx_paged_attention_forward_rejects_non_contiguous_metadata() -> i32;

    /// Production FFI bridge must accept already-materialized metadata and
    /// evaluate the returned lazy paged-attention output successfully.
    pub fn mlx_paged_attention_forward_eval_accepts_materialized_metadata() -> i32;

    /// Production FFI bridge must emit and evaluate lazy paged-kv-write pool
    /// outputs without forcing Rust-side Metal buffer extraction.
    pub fn mlx_paged_kv_write_forward_eval_smoke() -> i32;

    // =============================================================================
    // PagedKVWrite::eval_gpu and PagedAttention::eval_gpu must mirror the
    // row-contiguous / zero-offset check that the factories already
    // perform. The compile cache key only compares rank/shape/dtype, so a
    // graph first traced with contiguous inputs can be replayed via
    // `compile_replace` with a same-shape sliced/transposed view that
    // bypasses the factory's check entirely.
    //
    // Each helper compiles a function emitting the relevant primitive,
    // calls it once with contiguous inputs (cache miss; factory +
    // eval_gpu both pass), then calls it again with the SAME shapes and
    // dtypes but with one input substituted by a non-row-contiguous /
    // nonzero-offset view. The mirrored eval_gpu check MUST throw on
    // the second eval.
    //
    // Return codes:
    //   1   — second-call eval threw `std::invalid_argument` (fix
    //         working — the compile-cached path is contiguity-checked).
    //   0   — second-call eval did NOT throw (regression — a malformed
    //         view reached the kernel).
    //  -1   — internal/setup error (first call failed unexpectedly).
    //  -3   — Metal not available; eval-based verification skipped
    //         (slice/transpose materialization needs Metal).
    // =============================================================================

    /// Compile a `paged_kv_write`-emitting function, call it once with
    /// contiguous inputs, then call it again with `new_k` substituted
    /// by a transposed (non-row-contiguous) view. The mirrored check
    /// inside `PagedKVWrite::eval_gpu` MUST throw on the second eval.
    pub fn mlx_paged_kv_write_compile_cached_non_contiguous_throws() -> i32;

    /// Compile a `paged_attention`-emitting function, call it once with
    /// contiguous inputs, then call it again with `block_table`
    /// substituted by a sliced (nonzero-offset) view. The mirrored
    /// check inside `PagedAttention::eval_gpu` MUST throw on the second
    /// eval.
    pub fn mlx_paged_attention_compile_cached_non_contiguous_throws() -> i32;

    // Defense-in-depth tests: directly construct the primitive with
    // deliberately bad scalar state and dispatch `eval_gpu` via `eval()`.
    // The factory is never invoked, so the throw must come from the
    // validator inside `eval_gpu` itself — proving that compile-cache
    // replay (which bypasses the factory) can never bypass a check the
    // helper performs. Every helper distinguishes graph construction from
    // eval, AND verifies the exception message contains the eval_gpu
    // validator context tag. Return codes:
    //   *  1 — eval_gpu validator threw `std::invalid_argument` with
    //          the expected context tag (PASS).
    //   *  0 — eval did not throw (bad inputs accepted; FAIL).
    //   *  2 — graph construction (`make_arrays` / `array(...)`)
    //          threw before eval (internal helper bug; FAIL).
    //   * -1 — non-`std::invalid_argument` exception (FAIL).
    //   * -2 — eval threw `std::invalid_argument` but message did
    //          NOT contain the eval_gpu context tag — throw site is
    //          NOT the validator (FAIL).
    pub fn mlx_paged_kv_write_eval_gpu_rejects_zero_kv_heads() -> i32;
    pub fn mlx_paged_kv_write_eval_gpu_rejects_zero_block_size() -> i32;
    pub fn mlx_paged_kv_write_eval_gpu_rejects_zero_head_size() -> i32;
    pub fn mlx_paged_kv_write_eval_gpu_rejects_x_pack_dtype_mismatch() -> i32;
    /// Same scenario as `..._rejects_zero_block_size`, but with
    /// `slot_mapping={-1,-1}` so the runtime bounds guard CANNOT fire —
    /// only the scalar validator can throw. Proves the `[validator]`
    /// marker is on the scalar reject and not on a runtime guard
    /// masquerading as one.
    pub fn mlx_paged_kv_write_eval_gpu_validator_proof_zero_block_size() -> i32;
    pub fn mlx_paged_attention_eval_gpu_rejects_zero_kv_heads() -> i32;
    pub fn mlx_paged_attention_eval_gpu_rejects_indivisible_grouping() -> i32;
    pub fn mlx_paged_attention_eval_gpu_rejects_sliding_window() -> i32;
    pub fn mlx_paged_attention_eval_gpu_rejects_zero_block_size() -> i32;

    /// Stress test (mixed paged + non-paged ops, correctness +
    /// determinism + V1/V2 coverage).
    ///
    /// Builds a small graph mixing `paged_kv_write` + non-paged
    /// `add` + `paged_attention` + `add`, runs it `iterations` times
    /// with identical inputs, and asserts:
    ///
    /// - every run is byte-equal to a synchronous reference output
    ///   computed with explicit `eval()` between write and read
    ///   (proves the encoder fence honors the write→read dep);
    /// - every run differs from a no-write baseline output computed
    ///   against a zero pool (proves the write actually landed before
    ///   the read — guards against a deterministically stale read).
    ///
    /// `seq_len` selects the V1 (no partitioning) vs V2 (with
    /// partitioning + reduce) kernel path: `max_context_len <= 512`
    /// picks V1, `> 512` picks V2.
    ///
    /// Returns:
    ///
    /// - `0` — success
    /// - `-1` — internal/setup error
    /// - `-2` — run diverged from synchronous reference (race detected)
    /// - `-3` — Metal not available; test skipped
    /// - `-4` — run matched no-write baseline (write didn't land)
    pub fn mlx_paged_phase2_stress_mixed_graph_v(iterations: i32, seq_len: i32) -> i32;

    /// Backward-compatible default-V1 wrapper around
    /// `mlx_paged_phase2_stress_mixed_graph_v` with `seq_len=8`. Same
    /// return-code contract.
    pub fn mlx_paged_phase2_stress_mixed_graph(iterations: i32) -> i32;

    /// Compare two `PagedAttentionVarlen` primitives via `is_equivalent`.
    /// Returns 1 iff equivalent, 0 otherwise.
    #[allow(clippy::too_many_arguments)]
    pub fn mlx_paged_attention_varlen_is_equivalent(
        scale_lhs: f32,
        softcap_lhs: f32,
        block_size_lhs: i32,
        num_q_heads_lhs: i32,
        num_kv_heads_lhs: i32,
        head_size_lhs: i32,
        sliding_window_lhs: i32,
        kv_dtype_lhs: u8,
        scale_rhs: f32,
        softcap_rhs: f32,
        block_size_rhs: i32,
        num_q_heads_rhs: i32,
        num_kv_heads_rhs: i32,
        head_size_rhs: i32,
        sliding_window_rhs: i32,
        kv_dtype_rhs: u8,
    ) -> i32;

    /// Returns 1 iff `PagedAttentionVarlen::vjp` throws
    /// `std::runtime_error`.
    pub fn mlx_paged_attention_varlen_vjp_throws() -> i32;

    /// Returns 1 iff the public `paged_attention_varlen(...)` factory
    /// accepts well-formed tracer inputs and produces an output array of
    /// the expected shape. -1 on shape mismatch, 0 on factory exception.
    pub fn mlx_paged_attention_varlen_factory_accepts_wellformed() -> i32;

    /// Returns 1 iff the factory rejects a `cu_seqlens_q` array whose
    /// length is not `num_seqs + 1`.
    pub fn mlx_paged_attention_varlen_factory_rejects_cu_seqlens_len() -> i32;

    /// Returns 1 iff the factory rejects a `cu_seqlens_q` array whose
    /// dtype is not int32.
    pub fn mlx_paged_attention_varlen_factory_rejects_cu_seqlens_dtype() -> i32;

    /// Returns 1 iff `paged_attention_varlen(...)` composes inside
    /// `mlx::core::compile(...)` and returns an output of the expected
    /// shape. -1 on any thrown exception or shape mismatch.
    pub fn mlx_paged_attention_varlen_compile_trace_smoke() -> i32;
}

// ================================================================================
// Quantization Operations (for QuantizedKVCache)
// ================================================================================

unsafe extern "C-unwind" {
    /// Quantize a matrix along its last axis.
    /// Mode: "affine" (returns 3 arrays: weight, scales, biases),
    ///       "mxfp4" / "mxfp8" / "nvfp4" (returns 2 arrays: weight, scales).
    pub fn mlx_quantize(
        w: *mut mlx_array,
        group_size: i32,
        bits: i32,
        mode: *const std::os::raw::c_char,
        out_quantized: *mut *mut mlx_array,
        out_scales: *mut *mut mlx_array,
        out_biases: *mut *mut mlx_array,
    ) -> bool;

    /// Dequantize a matrix that was quantized with mlx_quantize.
    /// Mode must match the mode used during quantization.
    pub fn mlx_dequantize(
        quantized: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array, // nullable
        group_size: i32,
        bits: i32,
        out_dtype: i32, // -1 for input dtype
        mode: *const std::os::raw::c_char,
    ) -> *mut mlx_array;

    // 2D Convolution using MLX native conv2d
    pub fn mlx_conv2d(
        input: *mut mlx_array,
        weight: *mut mlx_array,
        stride_h: i32,
        stride_w: i32,
        padding_h: i32,
        padding_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        groups: i32,
    ) -> *mut mlx_array;

    // 2D Transposed Convolution using MLX native conv_transpose2d
    pub fn mlx_conv_transpose2d(
        input: *mut mlx_array,
        weight: *mut mlx_array,
        stride_h: i32,
        stride_w: i32,
        padding_h: i32,
        padding_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        groups: i32,
    ) -> *mut mlx_array;

    /// Fused PaddleOCR-VL forward pass - entire transformer forward in one FFI call.
    /// Uses mRoPE (multimodal rotary position embedding) instead of standard RoPE.
    /// 9 weights per layer: [input_norm, post_attn_norm, q, k, v, o, gate, up, down]
    pub fn mlx_paddleocr_vl_forward_step(
        input_embeds: *mut mlx_array,         // [batch, seq_len, hidden_size]
        layer_weights: *const *mut mlx_array, // [num_layers * 9]
        num_layers: i32,
        final_norm_weight: *mut mlx_array, // [hidden_size]
        lm_head_weight: *mut mlx_array,    // [vocab_size, hidden_size]
        inv_freq: *mut mlx_array,          // [1, 1, half_dim, 1]
        position_ids: *mut mlx_array,      // [3, batch, seq_len]
        mrope_section: *const i32,         // [3] e.g. {16, 24, 24}
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        norm_eps: f32,
        kv_keys_in: *const *mut mlx_array,   // [num_layers] or null
        kv_values_in: *const *mut mlx_array, // [num_layers] or null
        cache_idx_in: i32,
        out_logits: *mut *mut mlx_array,
        out_kv_keys: *mut *mut mlx_array,   // [num_layers]
        out_kv_values: *mut *mut mlx_array, // [num_layers]
        out_cache_idx: *mut i32,
    );

    /// Batched PaddleOCR-VL forward pass with left-padding-aware attention masking.
    /// Like mlx_paddleocr_vl_forward_step but supports batch > 1 during decode.
    pub fn mlx_paddleocr_vl_forward_step_batched(
        input_embeds: *mut mlx_array,         // [batch, seq_len, hidden_size]
        layer_weights: *const *mut mlx_array, // [num_layers * 9]
        num_layers: i32,
        final_norm_weight: *mut mlx_array,
        lm_head_weight: *mut mlx_array,
        inv_freq: *mut mlx_array,
        position_ids: *mut mlx_array, // [3, batch, seq_len]
        mrope_section: *const i32,
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        norm_eps: f32,
        left_padding: *mut mlx_array, // [batch] - left padding amounts
        kv_keys_in: *const *mut mlx_array,
        kv_values_in: *const *mut mlx_array,
        cache_idx_in: i32,
        out_logits: *mut *mut mlx_array,
        out_kv_keys: *mut *mut mlx_array,
        out_kv_values: *mut *mut mlx_array,
        out_cache_idx: *mut i32,
    );

    // ============================================
    // Conv1d
    // ============================================
    pub fn mlx_conv1d(
        input: *mut mlx_array,
        weight: *mut mlx_array,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> *mut mlx_array;

    // ============================================
    // Gather MM (for MoE / SwitchLinear)
    // ============================================
    pub fn mlx_gather_mm(
        a: *mut mlx_array,
        b: *mut mlx_array,
        lhs_indices: *mut mlx_array, // nullable
        rhs_indices: *mut mlx_array, // nullable
        sorted_indices: bool,
    ) -> *mut mlx_array;

    // ============================================
    // Quantized Matmul (for QuantizedLinear)
    // ============================================
    pub fn mlx_quantized_matmul(
        x: *mut mlx_array,
        w: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array, // nullable
        transpose: bool,
        group_size: i32,
        bits: i32,
        mode: *const std::os::raw::c_char,
    ) -> *mut mlx_array;

    // ============================================
    // Gather QMM (for QuantizedSwitchLinear / MoE)
    // ============================================
    pub fn mlx_gather_qmm(
        x: *mut mlx_array,
        w: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array,      // nullable
        lhs_indices: *mut mlx_array, // nullable
        rhs_indices: *mut mlx_array, // nullable
        transpose: bool,
        group_size: i32,
        bits: i32,
        mode: *const std::os::raw::c_char,
        sorted_indices: bool,
    ) -> *mut mlx_array;

    // Gated Delta Recurrence Metal Kernel
    pub fn mlx_gated_delta_kernel(
        q: *mut mlx_array,
        k: *mut mlx_array,
        v: *mut mlx_array,
        g: *mut mlx_array,
        beta: *mut mlx_array,
        state: *mut mlx_array,
        mask: *mut mlx_array, // nullptr if no mask
        out_y: *mut *mut mlx_array,
        out_state: *mut *mut mlx_array,
    ) -> bool;

    // Fused compute_g: g = exp(-exp(A_log) * softplus(a + dt_bias))
    pub fn mlx_fused_compute_g(
        a_log: *mut mlx_array,
        a: *mut mlx_array,
        dt_bias: *mut mlx_array,
    ) -> *mut mlx_array;

    // Chunked gated delta recurrence for prefill (BT=32 tokens per chunk)
    pub fn mlx_gated_delta_chunked(
        q: *mut mlx_array,
        k: *mut mlx_array,
        v: *mut mlx_array,
        g: *mut mlx_array,
        beta: *mut mlx_array,
        state: *mut mlx_array,
        out_y: *mut *mut mlx_array,
        out_state: *mut *mut mlx_array,
    ) -> bool;

    // GPU architecture generation (M1=13, M2=14, M3=15, M4=16, M5=17)
    pub fn mlx_gpu_architecture_gen() -> i32;

    // NA (Neural Accelerator) int8 W8A8 prefill GEMM primitives.
    //
    // All three gate internally on M5+ (gpu gen>=17) AND K % 16 == 0, returning
    // false on unsupported/failure so the Rust caller can fall back to bf16.
    // int8 lives entirely C++-side (Rust has no Int8 DType): the int8 GEMM takes
    // bf16/f32 arrays holding integer values in [-127,127]; the W8A8 ops hold the
    // int8 weight as an opaque mlx_array Rust never introspects.

    // int8 x @ w^T -> int32 [M,N]. x,w are bf16/f32 INTEGER-valued arrays; w is
    // [N,K] (rows = output channels). out_i32 is int32 [M,N] on success.
    pub fn mlx_matmul_int8(
        x: *mut mlx_array,
        w: *mut mlx_array,
        out_i32: *mut *mut mlx_array,
    ) -> bool;

    // Per-output-channel symmetric int8 weight quant (load-time, runs once).
    // w_bf16: [N,K] -> out_w_i8: opaque int8 PRE-TRANSPOSED to [K,N] kernel
    // layout (Stage 4b — hoists the per-forward transpose to load), out_s_w:
    // f32 [N] indexing the output channel N.
    pub fn mlx_quantize_weight_int8(
        w: *mut mlx_array,
        out_w_i8: *mut *mut mlx_array,
        out_s_w: *mut *mut mlx_array,
    ) -> bool;

    // CONVERT-time sym8 quantizer (checkpoint layout). Same per-output-channel
    // symmetric math as mlx_quantize_weight_int8 but WITHOUT the [K,N] kernel
    // transpose: returns the STORABLE int8 [N,K] weight (source orientation)
    // plus f32 [N] scales. Dequant: w[n,k] ≈ scales[n] * q[n,k]. 2D-only.
    pub fn mlx_sym8_quantize_store(
        w: *mut mlx_array,
        out_q: *mut *mut mlx_array,
        out_scales: *mut *mut mlx_array,
    ) -> bool;

    // LOAD-time sym8 kernel-operand builder: stored checkpoint int8 [N,K]
    // weight -> contiguous [K,N] int8 kernel operand (the exact
    // transpose+contiguous tail of mlx_quantize_weight_int8, requant-free).
    // Fail-loud on non-2D / non-int8 input. Evals before returning so the
    // transpose copy happens ONCE at load.
    pub fn mlx_sym8_kernel_operand(w: *mut mlx_array, out_w_kn: *mut *mut mlx_array) -> bool;

    // W8A8 linear: per-token int8 act quant + int8 GEMM + rescale -> bf16 [M,N].
    // x_bf16: [M,K], w_i8: [K,N] int8 (opaque, pre-transposed at load), s_w:
    // f32 [N]. Returns a LAZY array (Stage 4b — no force-eval); caller evals at
    // end of forward.
    pub fn mlx_w8a8_linear(
        x: *mut mlx_array,
        w_i8: *mut mlx_array,
        s_w: *mut mlx_array,
        out_bf16: *mut *mut mlx_array,
    ) -> bool;

    // sym8 DECODE matvec (QMV): per-token int8 act quant + int8 MATVEC + rescale
    // -> bf16 [M,N] = x @ w^T. The small-M (decode, M=1..~16) analogue of
    // mlx_w8a8_linear — reuses the SAME activation int8 quant + the SAME [K,N]
    // pre-transposed weight / f32 [N] s_w, but runs a dedicated BW-bound matvec
    // (one thread per output column) instead of the 128x64 prefill tile (which
    // wastes 127/128 rows at M=1). x_bf16: [M,K], w_i8: [K,N] int8 (opaque,
    // pre-transposed at load), s_w: f32 [N]. Returns a LAZY array; caller evals
    // at end of forward. Returns false (Rust falls back) on gen<17 / K%16!=0.
    pub fn mlx_int8_qmv(
        x: *mut mlx_array,
        w_i8: *mut mlx_array,
        s_w: *mut mlx_array,
        out_bf16: *mut *mut mlx_array,
    ) -> bool;

    // W8A16 sym8 DECODE matvec (QMV): bf16 activations read DIRECTLY — NO
    // activation quant (single kernel pass, f32 accumulate, activation-exact).
    // y[m,n] = bf16(s_w[n] * sum_k x[m,k]*w[n,k]). Takes BOTH weight
    // orientations: w_kn [K,N] (the GEMM operand — consumed by the 2D-block
    // fallback under INT8_QMV16_SG=0 and the INT8_QMV_W8A16=0 W8A8 reroute)
    // and w_nk [N,K] int8 (the CHECKPOINT tensor — consumed by the DEFAULT
    // simd_sum-style kernel, which streams [N,K] row-major like MLX's affine
    // qmv; buffer-shared with the stored checkpoint, not an extra copy).
    // This is the PRODUCTION sym8 decode op (M<=2). Returns a LAZY array;
    // caller evals at end of forward. false on gen<17 / K%16!=0.
    pub fn mlx_int8_qmv_w8a16(
        x: *mut mlx_array,
        w_kn: *mut mlx_array,
        w_nk: *mut mlx_array,
        s_w: *mut mlx_array,
        out_bf16: *mut *mut mlx_array,
    ) -> bool;

    // MEASUREMENT ONLY (de-risk microbench). Affine-group W8A8 linear directly
    // on the model's EXACT affine packed weight (no re-quant). Computes
    //   y[m,n] = s_x[m] * sum_g ( scale[n,g]*P[m,n,g] + bias[n,g]*S[m,g] )
    // with per-token int8 activation quant (identical to the symmetric path).
    //   x:        [M,K] bf16
    //   packed_w: [N, K/4] uint32 affine 8-bit packed weight
    //   scales:   [N, K/group_size] f32
    //   biases:   [N, K/group_size] f32
    // Returns bf16 [M,N]. Returns false (Rust falls back) when gen<17, bits!=8,
    // or K % group_size != 0. NOT a production op.
    pub fn mlx_affine_w8a8_linear(
        x: *mut mlx_array,
        packed_w: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array,
        group_size: i32,
        bits: i32,
        out: *mut *mut mlx_array,
    ) -> bool;

    // MEASUREMENT ONLY (de-risk microbench). LOAD-TIME prepare for the TILED
    // affine-group W8A8 GEMM (runs once per quantized linear). Unpacks the
    // affine packed weight into the SIGNED int8 [K,N] kernel operand the tiled
    // matmul2d wants, keeps the f32 scale, and precomputes
    // bias_adj = 128*scale + bias.
    //   packed_w: [N, K/4] uint32 affine 8-bit packed weight
    //   scales:   [N, K/group_size] f32
    //   biases:   [N, K/group_size] f32
    //   -> out_q_s:   opaque int8 [K,N] (q - 128, kernel operand)
    //      out_scale: f32 [N, K/group_size] (scale kept)
    //      out_badj:  f32 [N, K/group_size] (= 128*scale + bias)
    // Returns false (Rust falls back) when gen<17, bits!=8, or K%group_size!=0.
    pub fn mlx_affine_w8a8_prepare(
        packed_w: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array,
        group_size: i32,
        bits: i32,
        out_q_s: *mut *mut mlx_array,
        out_scale: *mut *mut mlx_array,
        out_badj: *mut *mut mlx_array,
    ) -> bool;

    // MEASUREMENT ONLY (de-risk microbench). Per-FORWARD prepared linear (the
    // TIMED hot path). Per-token int8 act quant + per-group act-sum S, then the
    // TILED grouped matmul2d GEMM.
    //   x:     [M,K] bf16
    //   q_s:   [K,N] int8 (signed, from mlx_affine_w8a8_prepare)
    //   scale: [N, K/group_size] f32 (kept)
    //   badj:  [N, K/group_size] f32 (= 128*scale + bias)
    //   -> out: bf16 [M,N] (LAZY). Returns false on gen<17, K%group_size!=0.
    pub fn mlx_affine_w8a8_linear_prepared(
        x: *mut mlx_array,
        q_s: *mut mlx_array,
        scale: *mut mlx_array,
        badj: *mut mlx_array,
        group_size: i32,
        out: *mut *mut mlx_array,
    ) -> bool;

    // MEASUREMENT ONLY (profiler/test scope). Pure int8 GEMM with a
    // PRE-TRANSPOSED [K,N] weight — isolates the kernel from the per-call
    // int8_weight_to_kn transpose. x: [M,K] bf16/f32 int-valued (cast to int8),
    // w_kn: [K,N] int8 (opaque, from mlx_quantize_weight_int8) used directly.
    // out_i32: int32 [M,N] = x @ w^T. NOT a production op.
    pub fn mlx_int8_gemm_pretransposed(
        x: *mut mlx_array,
        w_kn: *mut mlx_array,
        out_i32: *mut *mut mlx_array,
    ) -> bool;

    // MEASUREMENT ONLY. Same as above but mode::multiply + init_value=nullopt →
    // skips MLX's per-call full-output zero fill. Isolates the fill cost. NOT a
    // production op.
    pub fn mlx_int8_gemm_pretransposed_nofill(
        x: *mut mlx_array,
        w_kn: *mut mlx_array,
        out_i32: *mut *mut mlx_array,
    ) -> bool;

    // MEASUREMENT ONLY (parity test scope). FUSED v1 activation-quant kernel.
    // x_bf16 [M,K] -> out_i8_as_i32 = int32([M,K] int8 quant) (Rust has no Int8
    // dtype, so widened to int32 for readback), out_s_x = f32 [M,1].
    pub fn mlx_int8_act_quant_fused(
        x: *mut mlx_array,
        out_i8_as_i32: *mut *mut mlx_array,
        out_s_x: *mut *mut mlx_array,
    ) -> bool;

    // MEASUREMENT ONLY. The LAZY activation-quant chain (parity reference for the
    // fused kernel — must be bit-identical). Same I/O as the fused FFI.
    pub fn mlx_int8_act_quant_lazy(
        x: *mut mlx_array,
        out_i8_as_i32: *mut *mut mlx_array,
        out_s_x: *mut *mut mlx_array,
    ) -> bool;

    // MEASUREMENT ONLY (parity test scope). FUSED v1 rescale kernel.
    // acc [M,N] int32, s_x [M,1] f32, s_w [N] f32 -> y bf16 [M,N].
    pub fn mlx_int8_rescale_fused(
        acc: *mut mlx_array,
        s_x: *mut mlx_array,
        s_w: *mut mlx_array,
        out_bf16: *mut *mut mlx_array,
    ) -> bool;

    // MEASUREMENT ONLY. The LAZY rescale (parity reference for the fused kernel —
    // must match to bf16 eps). Same I/O as the fused FFI.
    pub fn mlx_int8_rescale_lazy(
        acc: *mut mlx_array,
        s_x: *mut mlx_array,
        s_w: *mut mlx_array,
        out_bf16: *mut *mut mlx_array,
    ) -> bool;

    // Fused GDN gating: beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)
    pub fn mlx_fused_gdn_gating(
        b: *mut mlx_array,
        a: *mut mlx_array,
        a_log: *mut mlx_array,
        dt_bias: *mut mlx_array,
        num_heads: i32,
        total_elements: i32,
        out_beta: *mut *mut mlx_array,
        out_g: *mut *mut mlx_array,
    ) -> bool;

    /// Load safetensors file using MLX's lazy loading (data read on eval, not upfront).
    /// Calls `callback` for each tensor with (name, name_len, array_handle, ctx).
    /// Returns number of tensors loaded, or -1 on error.
    pub fn mlx_load_safetensors(
        path: *const std::os::raw::c_char,
        callback: unsafe extern "C-unwind" fn(
            name: *const std::os::raw::c_char,
            name_len: usize,
            handle: *mut mlx_array,
            ctx: *mut std::os::raw::c_void,
        ),
        ctx: *mut std::os::raw::c_void,
    ) -> i32;
}

// Gradient computation types
pub type LossFunctionPtr = extern "C-unwind" fn(
    inputs: *const *mut mlx_array,
    input_count: usize,
    context: *mut std::os::raw::c_void,
) -> *mut mlx_array;

// Checkpoint layer function type: takes inputs, writes outputs, returns count
pub type LayerFunctionPtr = extern "C-unwind" fn(
    inputs: *const *mut mlx_array,
    input_count: usize,
    outputs: *mut *mut mlx_array,
    max_outputs: usize,
    context: *mut std::os::raw::c_void,
) -> usize;
