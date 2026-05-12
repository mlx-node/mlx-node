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
    pub fn mlx_array_to_uint16(handle: *mut mlx_array, out: *mut u16, len: usize) -> bool;
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
    ) -> *mut mlx_array;

    /// Compiled sampling using mlx::core::compile for the categorical step
    /// This matches mlx-lm's @partial(mx.compile, ...) approach
    pub fn mlx_compiled_sample_and_logprobs(
        logits: *mut mlx_array,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        out_token: *mut *mut mlx_array,
        out_logprobs: *mut *mut mlx_array,
    );

    // Stream operations
    pub fn mlx_default_stream(device_type: i32) -> mlx_stream;
    pub fn mlx_new_stream(device_type: i32) -> mlx_stream;
    pub fn mlx_set_default_stream(stream: mlx_stream);
    pub fn mlx_stream_synchronize(stream: mlx_stream);

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
// Paged ops Phase 1 test helpers (defined in `mlx_paged_ops.cpp`)
//
// These exist solely so the unit tests in
// `crates/mlx-paged-attn/tests/paged_ops_smoke.rs` can exercise the
// C++ `PagedKVWrite` / `PagedAttention` primitives' `is_equivalent`
// and `vjp` semantics without standing up a separate C++ test runner.
// Phase 2 may delete them.
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
    // Phase 1 review-round-3 negative-validation FFI declarations.
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
    /// rejected (Phase 1 safety check; eval-based bounds verification).
    pub fn mlx_paged_kv_write_factory_rejects_slot_mapping_out_of_range() -> i32;

    /// Round-13 finding: assert the factory-side slot_mapping bounds
    /// guard's `std::invalid_argument` message contains the `[runtime]`
    /// marker (matching the same marker used by the eval_gpu-side guard
    /// for the same data-dependent property). See the C++ helper for
    /// the full return-code contract: 1 = passes (threw + marker
    /// present), 0 = no throw, -1 = wrong exception class / setup
    /// error, -2 = threw but `[runtime]` marker missing.
    pub fn mlx_paged_kv_write_factory_slot_mapping_out_of_range_runtime_marker() -> i32;

    // =============================================================================
    // Phase 1 review-round-4 dtype-mismatch FFI declarations (finding B+C).
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
    /// (Phase 1 contract: FP8 io dtype is bfloat16).
    pub fn mlx_paged_kv_write_factory_rejects_new_k_dtype_fp8() -> i32;

    /// q dtype != bfloat16 must be rejected for kv_dtype=Bf16.
    pub fn mlx_paged_attention_factory_rejects_q_dtype_bf16() -> i32;

    /// q dtype != bfloat16 must be rejected for kv_dtype=Fp8 (Phase 1
    /// contract: FP8 io dtype is bfloat16).
    pub fn mlx_paged_attention_factory_rejects_q_dtype_fp8() -> i32;

    /// k_pool dtype != bfloat16 must be rejected for kv_dtype=Bf16.
    pub fn mlx_paged_attention_factory_rejects_k_pool_dtype_bf16() -> i32;

    /// k_pool dtype != uint8 must be rejected for kv_dtype=Fp8.
    pub fn mlx_paged_attention_factory_rejects_k_pool_dtype_fp8() -> i32;

    // =============================================================================
    // Phase 1 review-round-6 finding: GQA head-group divisibility.
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
    // Phase 1 review-round-5 finding: PagedAttention::eval_gpu must
    // runtime-bounds-check `seq_lens` and `block_table` contents.
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
    // Phase 1 review-round-8 finding: factory must reject
    // non-row-contiguous or nonzero-offset views for ALL inputs.
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
    // Phase 1 review-round-9 finding: PagedKVWrite::eval_gpu and
    // PagedAttention::eval_gpu must mirror the row-contiguous /
    // zero-offset check that the factories already perform. The compile
    // cache key only compares rank/shape/dtype, so a graph first traced
    // with contiguous inputs can be replayed via `compile_replace` with
    // a same-shape sliced/transposed view that bypasses the factory's
    // check entirely.
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

    // Round-10 defense-in-depth tests: directly construct the
    // primitive with deliberately bad scalar state and dispatch
    // `eval_gpu` via `eval()`. The factory is never invoked, so the
    // throw must come from the validator inside `eval_gpu` itself —
    // proving that compile-cache replay (which bypasses the factory)
    // can never bypass a check the helper performs.
    //
    // Round-11 tightening: every helper distinguishes graph
    // construction from eval, AND verifies the exception message
    // contains the eval_gpu validator context tag. Return codes:
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
    /// Round-12 regression: same scenario as
    /// `..._rejects_zero_block_size`, but with `slot_mapping={-1,-1}`
    /// so the runtime bounds guard CANNOT fire — only the scalar
    /// validator can throw. Proves the `[validator]` marker is on the
    /// scalar reject and not on a runtime guard masquerading as one.
    pub fn mlx_paged_kv_write_eval_gpu_validator_proof_zero_block_size() -> i32;
    pub fn mlx_paged_attention_eval_gpu_rejects_zero_kv_heads() -> i32;
    pub fn mlx_paged_attention_eval_gpu_rejects_indivisible_grouping() -> i32;
    pub fn mlx_paged_attention_eval_gpu_rejects_sliding_window() -> i32;
    pub fn mlx_paged_attention_eval_gpu_rejects_zero_block_size() -> i32;

    /// Phase 2 stress test (mixed paged + non-paged ops, correctness +
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
}

// ================================================================================
// Quantization Operations (for QuantizedKVCache)
// ================================================================================

unsafe extern "C-unwind" {
    /// Quantize a matrix along its last axis.
    /// Mode: "affine" (returns 3 arrays), "mxfp4"/"mxfp8" (returns 2 arrays, biases=nullptr).
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

    // ============================================
    // Qwen3.5 Fused Forward Pass
    // ============================================

    /// Store a model weight by name (called once per weight during model load)
    pub fn mlx_store_weight(name: *const std::os::raw::c_char, weight: *mut mlx_array);

    /// Clear all stored weights (called on model destruction)
    pub fn mlx_clear_weights();

    /// Get the number of stored weights (for debugging)
    pub fn mlx_weight_count() -> usize;

    /// Set the active model ID (called after all weights are stored).
    /// Inference checks this against its own model_id to avoid cross-model contamination.
    pub fn mlx_set_model_id(id: u64);

    /// Get the active model ID. Returns 0 if no model has registered weights.
    pub fn mlx_qwen35_get_model_id() -> u64;

    /// Initialize compiled forward pass from post-prefill caches.
    /// Call once after prefill, before decode loop.
    /// cache_arrays: [num_layers * 2] non-null pointers to prefill cache arrays.
    pub fn mlx_qwen35_compiled_init_from_prefill(
        num_layers: i32,
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_theta: f32,
        rope_dims: i32,
        rms_norm_eps: f32,
        full_attention_interval: i32,
        linear_num_k_heads: i32,
        linear_num_v_heads: i32,
        linear_key_head_dim: i32,
        linear_value_head_dim: i32,
        linear_conv_kernel_dim: i32,
        tie_word_embeddings: i32,
        max_kv_len: i32,
        batch_size: i32,
        cache_arrays: *mut *mut mlx_array,
        prefill_offset: i32,
    );

    /// Compiled single-token decode step.
    /// Runs the full 64-layer forward pass with graph caching (mlx::core::compile).
    /// output_logits receives heap-allocated array; cache_offset_out receives new offset.
    pub fn mlx_qwen35_forward_compiled(
        input_ids: *mut mlx_array,
        embedding_weight: *mut mlx_array,
        output_logits: *mut *mut mlx_array,
        cache_offset_out: *mut i32,
    );

    /// Eval next_token and all compiled cache arrays to prevent graph accumulation.
    pub fn mlx_qwen35_eval_token_and_compiled_caches(next_token: *mut mlx_array);

    /// Adjust the compiled offset by delta (for VLM rope_deltas).
    pub fn mlx_qwen35_compiled_adjust_offset(delta: i32);

    /// Reset compiled state (call on model reset / new conversation).
    pub fn mlx_qwen35_compiled_reset();

    /// Export compiled caches for PromptCache reuse.
    pub fn mlx_qwen35_export_caches(out_ptrs: *mut *mut mlx_array, max_count: i32) -> i32;

    /// Get current compiled cache offset (tokens processed).
    pub fn mlx_qwen35_get_cache_offset() -> i32;

    /// Export paged dense linear-attention caches for live-session continuation.
    /// Full-attention K/V stays in the Rust paged adapter pools; this returns
    /// the paged graph's per-layer `(conv_state, recurrent_state)` slots so
    /// Rust can seed the next turn without replaying the cached prefix through
    /// GDN. Returns number of arrays exported, or 0 if paged state is not
    /// initialized.
    pub fn mlx_qwen35_export_paged_linear_caches(
        out_ptrs: *mut *mut mlx_array,
        max_count: i32,
    ) -> i32;

    /// Get current paged dense cache offset (tokens processed by the compiled
    /// paged decode graph).
    pub fn mlx_qwen35_get_paged_cache_offset() -> i32;

    // ============================================
    // Phase 5 piece 1: paged Dense forward (coexists with the flat
    // compiled path). The Rust dispatcher decides per-turn which graph
    // to run; `mlx_qwen35_compiled_reset` wipes BOTH graphs' state.
    // ============================================

    /// Initialize the paged Dense forward graph from per-layer pool /
    /// scale handles. See the C++ docstring on `mlx_qwen35_init_paged`
    /// for the full layout contract. Phase 5 piece 1 hard-codes
    /// `block_size = 16`, `kv_dtype = Bf16`, `x_pack = 8`,
    /// `sliding_window = 0`.
    ///
    /// `k_pool_handles`, `v_pool_handles`, `k_scale_handles`,
    /// `v_scale_handles` are arrays of `num_layers` `mlx_array*` each.
    /// Linear-layer slots may be null (placeholders are stored).
    /// `linear_cache_arrays` is a `2 * num_layers` array of
    /// `(conv_state, recurrent_state)` pairs; full-attn slots are
    /// ignored. Pass null for the entire array to skip seeding.
    ///
    /// Returns `0` on success, `-1` on failure (e.g. missing pool/scale
    /// handles, exception during graph build). On failure the C++ side
    /// clears `g_dense_paged_inited` and emits a stderr diagnostic.
    /// The Rust caller MUST inspect the return value and fall back to
    /// the pure-Rust paged path on `-1`.
    pub fn mlx_qwen35_init_paged(
        num_layers: i32,
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_theta: f32,
        rope_dims: i32,
        rms_norm_eps: f32,
        full_attention_interval: i32,
        linear_num_k_heads: i32,
        linear_num_v_heads: i32,
        linear_key_head_dim: i32,
        linear_value_head_dim: i32,
        linear_conv_kernel_dim: i32,
        tie_word_embeddings: i32,
        max_kv_len: i32,
        batch_size: i32,
        k_pool_handles: *mut *mut mlx_array,
        v_pool_handles: *mut *mut mlx_array,
        k_scale_handles: *mut *mut mlx_array,
        v_scale_handles: *mut *mut mlx_array,
        linear_cache_arrays: *mut *mut mlx_array,
        prefill_offset: i32,
    ) -> i32;

    /// Single-token paged Dense decode step. Sets `*output_logits` to a
    /// heap-allocated `mlx_array*` (caller owns) on success, or
    /// `nullptr` on error / when `mlx_qwen35_init_paged` hasn't been
    /// called. `cache_offset_out` receives the post-step offset.
    ///
    /// **Phase 5 piece 1 contract — decode-only.** `input_ids` MUST
    /// have exactly one element and `slot_mapping` MUST be `[1]`.
    /// Multi-token / chunked prefill is reserved for later phases. The
    /// contract is enforced on the C++ side: violating it returns null
    /// logits and writes a stderr diagnostic, leaving global state
    /// untouched so the caller can fall back to the flat path.
    pub fn mlx_qwen35_forward_paged(
        input_ids: *mut mlx_array,
        embedding_weight: *mut mlx_array,
        offset_arr: *mut mlx_array,
        block_table: *mut mlx_array,
        slot_mapping: *mut mlx_array,
        num_valid_tokens: *mut mlx_array,
        num_valid_blocks: *mut mlx_array,
        seq_lens: *mut mlx_array,
        output_logits: *mut *mut mlx_array,
        cache_offset_out: *mut i32,
    );

    // ============================================
    // Qwen3.5 VLM Prefill
    // ============================================

    // VLM prefill
    pub fn mlx_qwen35_vlm_prefill(
        inputs_embeds: *mut mlx_array,
        position_ids: *mut mlx_array,
        num_layers: i32,
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_theta: f32,
        rope_dims: i32,
        rms_norm_eps: f32,
        full_attention_interval: i32,
        linear_num_k_heads: i32,
        linear_num_v_heads: i32,
        linear_key_head_dim: i32,
        linear_value_head_dim: i32,
        linear_conv_kernel_dim: i32,
        tie_word_embeddings: i32,
        max_kv_len: i32,
        batch_size: i32,
        mrope_section: *const i32,
        rope_deltas: i32,
        output_logits: *mut *mut mlx_array,
    );

    pub fn mlx_qwen35_vlm_cache_count() -> i32;
    pub fn mlx_qwen35_vlm_get_cache(index: i32) -> *mut mlx_array;
    pub fn mlx_qwen35_vlm_reset();

    // ============================================
    // Qwen3.5 MoE Forward Pass (non-compiled)
    // ============================================

    /// Initialize MoE forward pass from post-prefill caches.
    pub fn mlx_qwen35_moe_init_from_prefill(
        num_layers: i32,
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_theta: f32,
        rope_dims: i32,
        rms_norm_eps: f32,
        full_attention_interval: i32,
        linear_num_k_heads: i32,
        linear_num_v_heads: i32,
        linear_key_head_dim: i32,
        linear_value_head_dim: i32,
        linear_conv_kernel_dim: i32,
        tie_word_embeddings: i32,
        max_kv_len: i32,
        batch_size: i32,
        num_experts: i32,
        num_experts_per_tok: i32,
        norm_topk_prob: i32,
        decoder_sparse_step: i32,
        mlp_only_layers: *const i32,
        mlp_only_layers_len: i32,
        cache_arrays: *mut *mut mlx_array,
        prefill_offset: i32,
    );

    /// MoE single-token decode step.
    pub fn mlx_qwen35_moe_forward(
        input_ids: *mut mlx_array,
        embedding_weight: *mut mlx_array,
        output_logits: *mut *mut mlx_array,
        cache_offset_out: *mut i32,
    );

    /// Eval next_token and all MoE cache arrays to prevent graph accumulation.
    pub fn mlx_qwen35_moe_eval_token_and_caches(next_token: *mut mlx_array);

    /// Reset MoE state.
    pub fn mlx_qwen35_moe_reset();

    /// Export MoE caches for PromptCache reuse.
    /// Copies cache arrays to caller-provided output pointers.
    /// Returns number of arrays exported, or 0 if not initialized.
    pub fn mlx_qwen35_moe_export_caches(out_ptrs: *mut *mut mlx_array, max_count: i32) -> i32;

    /// Export paged MoE linear-attention caches for live-session continuation.
    /// Full-attention K/V stays in the Rust paged adapter pools; this returns
    /// the paged graph's per-layer `(conv_state, recurrent_state)` slots so
    /// Rust can seed the next turn without replaying the cached prefix through
    /// GDN. Returns number of arrays exported, or 0 if paged state is not
    /// initialized.
    pub fn mlx_qwen35_moe_export_paged_linear_caches(
        out_ptrs: *mut *mut mlx_array,
        max_count: i32,
    ) -> i32;

    /// Get current paged MoE cache offset (tokens processed by the compiled
    /// paged decode graph).
    pub fn mlx_qwen35_moe_get_paged_cache_offset() -> i32;

    /// Get current MoE cache offset (tokens processed).
    pub fn mlx_qwen35_moe_get_cache_offset() -> i32;

    /// Adjust MoE cache offset by delta (for VLM M-RoPE position correction).
    pub fn mlx_qwen35_moe_adjust_offset(delta: i32);

    // ============================================
    // Phase 4 piece 1: paged MoE forward (coexists with the flat path).
    //
    // The Rust caller migration lands in piece 2; piece 3 deletes the
    // legacy flat FFI above.
    // ============================================

    /// Initialize the paged MoE forward graph from per-layer pool / scale
    /// handles. See the C++ docstring on `mlx_qwen35_moe_init_paged` for
    /// the full layout contract. Phase 4 piece 1 hard-codes
    /// `block_size = 16`, `kv_dtype = Bf16`, `x_pack = 8`,
    /// `sliding_window = 0`.
    ///
    /// `k_pool_handles`, `v_pool_handles`, `k_scale_handles`,
    /// `v_scale_handles` are arrays of `num_layers` `mlx_array*` each.
    /// Linear-layer slots may be null (placeholders are stored).
    /// `linear_cache_arrays` is a `2 * num_layers` array of
    /// `(conv_state, recurrent_state)` pairs; full-attn slots are
    /// ignored. Pass null for the entire array to skip seeding.
    ///
    /// Returns `0` on success, `-1` on failure (e.g. missing pool/scale
    /// handles, exception during graph build). On failure the C++ side
    /// clears `g_paged_inited` and emits a stderr diagnostic. The Rust
    /// caller MUST inspect the return value and fall back to the pure-Rust
    /// paged path on `-1`; entering the compiled paged decode after init
    /// failure dispatches against uninitialized globals.
    pub fn mlx_qwen35_moe_init_paged(
        num_layers: i32,
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_theta: f32,
        rope_dims: i32,
        rms_norm_eps: f32,
        full_attention_interval: i32,
        linear_num_k_heads: i32,
        linear_num_v_heads: i32,
        linear_key_head_dim: i32,
        linear_value_head_dim: i32,
        linear_conv_kernel_dim: i32,
        tie_word_embeddings: i32,
        max_kv_len: i32,
        batch_size: i32,
        num_experts: i32,
        num_experts_per_tok: i32,
        norm_topk_prob: i32,
        decoder_sparse_step: i32,
        mlp_only_layers: *const i32,
        mlp_only_layers_len: i32,
        k_pool_handles: *mut *mut mlx_array,
        v_pool_handles: *mut *mut mlx_array,
        k_scale_handles: *mut *mut mlx_array,
        v_scale_handles: *mut *mut mlx_array,
        linear_cache_arrays: *mut *mut mlx_array,
        prefill_offset: i32,
    ) -> i32;

    /// Single-token paged decode step. Sets `*output_logits` to a heap-
    /// allocated `mlx_array*` (caller owns) on success, or `nullptr` on
    /// error / when `mlx_qwen35_moe_init_paged` hasn't been called.
    /// `cache_offset_out` receives the post-step offset.
    ///
    /// **Phase 4 piece 1 contract — decode-only.** `input_ids` MUST
    /// have exactly one element and `slot_mapping` MUST be `[1]`.
    /// Multi-token / chunked prefill is reserved for piece 2. The
    /// contract is enforced on the C++ side: violating it returns null
    /// logits and writes a stderr diagnostic, leaving global state
    /// untouched so the caller can fall back to the flat path.
    pub fn mlx_qwen35_moe_forward_paged(
        input_ids: *mut mlx_array,
        embedding_weight: *mut mlx_array,
        offset_arr: *mut mlx_array,
        block_table: *mut mlx_array,
        slot_mapping: *mut mlx_array,
        num_valid_tokens: *mut mlx_array,
        num_valid_blocks: *mut mlx_array,
        seq_lens: *mut mlx_array,
        output_logits: *mut *mut mlx_array,
        cache_offset_out: *mut i32,
    );

    /// Phase 4 piece 1 test helper — builds the `attn_for_compile_paged`
    /// graph in isolation against synthetic weights, force-evaluates the
    /// output (so paged_kv_write + paged_attention actually dispatch on
    /// the Metal queue), and cleans up the synthetic weights.
    ///
    /// Returns 0 on success, non-zero on failure (exception caught and
    /// stderr diagnostic written). Used by
    /// `crates/mlx-paged-attn/tests/qwen3_5_moe_paged_smoke.rs` to
    /// guarantee the paged graph itself is exercised — the existing
    /// `forward_paged` smoke test fails inside the embedding/LM-head
    /// lookups before reaching the paged attention graph.
    ///
    /// IMPORTANT: this helper writes to the global weight map and
    /// clears its own additions on exit. Callers MUST also invoke
    /// `mlx_clear_weights()` before/after to avoid contaminating other
    /// model state.
    pub fn mlx_qwen35_moe_trace_paged_attn_helper() -> i32;

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
