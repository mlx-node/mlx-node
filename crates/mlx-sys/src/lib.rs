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

unsafe extern "C" {
    pub fn mlx_version() -> *const std::os::raw::c_char;
    pub fn mlx_seed(seed: u64);
    pub fn mlx_array_from_int32(data: *const i32, shape: *const i64, ndim: usize)
    -> *mut mlx_array;
    pub fn mlx_array_from_uint32(
        data: *const u32,
        shape: *const i64,
        ndim: usize,
    ) -> *mut mlx_array;
    pub fn mlx_array_from_float32(
        data: *const f32,
        shape: *const i64,
        ndim: usize,
    ) -> *mut mlx_array;
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
    pub fn mlx_array_logsumexp(handle: *mut mlx_array, axis: i32, keepdims: bool)
    -> *mut mlx_array;
    pub fn mlx_array_softmax(handle: *mut mlx_array, axis: i32) -> *mut mlx_array;
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

    // Fused Multi-Head Attention forward (without KV cache)
    pub fn mlx_fused_attention_forward(
        x: *mut mlx_array,
        w_q: *mut mlx_array,
        w_k: *mut mlx_array,
        w_v: *mut mlx_array,
        w_o: *mut mlx_array,
        q_norm_w: *mut mlx_array, // Can be null
        k_norm_w: *mut mlx_array, // Can be null
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        rope_base: f32,
        rope_dims: i32,
        qk_norm_eps: f32,
        use_causal: bool,
        rope_offset: i32,
    ) -> *mut mlx_array;

    // Fused Multi-Head Attention forward with KV cache
    pub fn mlx_fused_attention_forward_cached(
        x: *mut mlx_array,
        w_q: *mut mlx_array,
        w_k: *mut mlx_array,
        w_v: *mut mlx_array,
        w_o: *mut mlx_array,
        q_norm_w: *mut mlx_array,
        k_norm_w: *mut mlx_array,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        rope_base: f32,
        rope_dims: i32,
        qk_norm_eps: f32,
        use_causal: bool,
        cached_keys: *mut *mut mlx_array,
        cached_values: *mut *mut mlx_array,
        cache_offset: i32,
        output: *mut *mut mlx_array,
    );

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

    // Fused Transformer Block forward with KV cache
    pub fn mlx_fused_transformer_block_forward_cached(
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
        cached_keys: *mut *mut mlx_array,
        cached_values: *mut *mut mlx_array,
        cache_offset: i32,
        output: *mut *mut mlx_array,
    );

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
    pub fn mlx_array_slice_update(
        src_handle: *mut mlx_array,
        update_handle: *mut mlx_array,
        starts: *const i64,
        stops: *const i64,
        ndim: usize,
    ) -> *mut mlx_array;
    pub fn mlx_array_slice_update_inplace(
        src_handle: *mut mlx_array,
        update_handle: *mut mlx_array,
        starts: *const i64,
        stops: *const i64,
        ndim: usize,
    );
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
    pub fn mlx_array_scatter(
        src_handle: *mut mlx_array,
        indices_handle: *mut mlx_array,
        updates_handle: *mut mlx_array,
        axis: i32,
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
    pub fn mlx_array_to_float32_noeval(handle: *mut mlx_array, out: *mut f32, len: usize) -> bool;
    pub fn mlx_array_to_int32(handle: *mut mlx_array, out: *mut i32, len: usize) -> bool;
    pub fn mlx_array_to_int32_noeval(handle: *mut mlx_array, out: *mut i32, len: usize) -> bool;
    pub fn mlx_array_to_uint32(handle: *mut mlx_array, out: *mut u32, len: usize) -> bool;
    pub fn mlx_array_delete(arr: *mut mlx_array);
    pub fn mlx_synchronize();
    pub fn mlx_clear_cache();
    pub fn mlx_compiled_categorical_sample(
        logits: *mut mlx_array,
        temperature: f32,
    ) -> *mut mlx_array;
    pub fn mlx_compiled_top_k(logprobs: *mut mlx_array, k: i32) -> *mut mlx_array;
    pub fn mlx_compiled_top_p(logprobs: *mut mlx_array, p: f32) -> *mut mlx_array;
    pub fn mlx_compiled_min_p(
        logprobs: *mut mlx_array,
        min_p: f32,
        min_tokens_to_keep: i32,
    ) -> *mut mlx_array;

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

    // Fast operations (mlx::fast namespace)
    pub fn mlx_fast_rope(
        handle: *mut mlx_array,
        dims: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: i32,
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
    pub fn mlx_compiled_apply_temperature(
        logits: *mut mlx_array,
        temperature: f32,
    ) -> *mut mlx_array;
    pub fn mlx_compiled_sample_full(
        logits: *mut mlx_array,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
    ) -> *mut mlx_array;

    /// Optimized sampling that returns BOTH token and logprobs
    /// This eliminates redundant logprobs computation by computing once and returning both.
    pub fn mlx_sample_and_logprobs(
        logits: *mut mlx_array,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        out_token: *mut *mut mlx_array,
        out_logprobs: *mut *mut mlx_array,
    );

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

    // Metal operations (for memory management)
    pub fn mlx_metal_is_available() -> bool;
    pub fn mlx_metal_device_info() -> *const std::os::raw::c_char;
    pub fn mlx_set_wired_limit(limit: usize) -> usize;
    pub fn mlx_get_wired_limit() -> usize;
    pub fn mlx_get_peak_memory() -> usize;
    pub fn mlx_get_active_memory() -> usize;
    pub fn mlx_get_cache_memory() -> usize;
    pub fn mlx_reset_peak_memory();
    pub fn mlx_set_memory_limit(limit: usize) -> usize;
    pub fn mlx_get_memory_limit() -> usize;
    pub fn mlx_array_nbytes(handle: *mut mlx_array) -> usize;

    // Fused generation loop - entire generation in one FFI call
    // This matches mlx-lm's async pipelining pattern for maximum performance
    pub fn mlx_qwen3_generate(
        // Input
        input_ids: *mut mlx_array, // [1, prompt_len]
        // Model weights
        embedding_weight: *mut mlx_array,     // [vocab, hidden]
        layer_weights: *const *mut mlx_array, // [num_layers * 11] weights per layer
        num_layers: i32,
        final_norm_weight: *mut mlx_array, // [hidden]
        lm_head_weight: *mut mlx_array,    // [vocab, hidden] or null if tied
        tie_word_embeddings: bool,
        // Model config
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_theta: f32,
        norm_eps: f32,
        // Generation config
        max_new_tokens: i32,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        repetition_penalty: f32,
        repetition_context_size: i32,
        eos_token_id: i32,
        // Outputs (caller allocates)
        out_tokens: *mut i32,   // [max_new_tokens]
        out_logprobs: *mut f32, // [max_new_tokens]
        out_num_tokens: *mut i32,
        out_finish_reason: *mut i32, // 0=length, 1=eos
    );

    // Fused forward step - single FFI call for entire forward pass
    // This reduces FFI overhead from ~300 calls to 1 call per token
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
        cache_offsets_in: *const i32,        // [num_layers] or null
        cache_capacities_in: *const i32,     // [num_layers] or null (pre-allocated buffer sizes)
        // Outputs
        out_logits: *mut *mut mlx_array,    // [batch, seq_len, vocab]
        out_kv_keys: *mut *mut mlx_array,   // [num_layers]
        out_kv_values: *mut *mut mlx_array, // [num_layers]
        out_cache_offsets: *mut i32,        // [num_layers]
        out_cache_capacities: *mut i32,     // [num_layers] (updated pre-allocated sizes)
    );
}

// Gradient computation types
pub type LossFunctionPtr = extern "C" fn(
    inputs: *const *mut mlx_array,
    input_count: usize,
    context: *mut std::os::raw::c_void,
) -> *mut mlx_array;
