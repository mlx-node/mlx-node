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
    pub fn mlx_version() -> *const std::os::raw::c_char;
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
    pub fn mlx_eval(handles: *mut *mut mlx_array, count: usize);
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
    pub fn mlx_array_to_uint8(handle: *mut mlx_array, out: *mut u8, len: usize) -> bool;
    pub fn mlx_array_to_uint16(handle: *mut mlx_array, out: *mut u16, len: usize) -> bool;
    pub fn mlx_array_delete(arr: *mut mlx_array);
    pub fn mlx_synchronize();
    pub fn mlx_clear_cache();
    pub fn mlx_compile_clear_cache() -> bool;
    pub fn mlx_stop_gradient(a: *mut mlx_array) -> *mut mlx_array;
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
    pub fn mlx_set_cache_limit(limit: usize) -> usize;
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

    // Batched forward step - true batch generation with array RoPE offsets
    // Enables parallel batch generation with left-padded variable-length sequences
    pub fn mlx_qwen3_forward_step_batched(
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
        // Batched RoPE offsets (key difference from scalar version)
        rope_offsets: *mut mlx_array, // [batch] - per-sequence offsets
        // Left padding info for attention mask
        left_padding: *mut mlx_array, // [batch] - left padding amounts
        // KV cache inputs (shared across batch, indexed by cache_idx)
        kv_keys_in: *const *mut mlx_array,   // [num_layers] or null
        kv_values_in: *const *mut mlx_array, // [num_layers] or null
        cache_idx_in: i32,                   // Current write position (shared)
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
    pub fn mlx_qwen35_store_weight(name: *const std::os::raw::c_char, weight: *mut mlx_array);

    /// Clear all stored weights (called on model destruction)
    pub fn mlx_qwen35_clear_weights();

    /// Get the number of stored weights (for debugging)
    pub fn mlx_qwen35_weight_count() -> usize;

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

    /// Synchronously eval all compiled cache arrays (for training decode loop).
    pub fn mlx_qwen35_sync_eval_compiled_caches();

    /// Adjust the compiled offset by delta (for VLM rope_deltas).
    pub fn mlx_qwen35_compiled_adjust_offset(delta: i32);

    /// Reset compiled state (call on model reset / new conversation).
    pub fn mlx_qwen35_compiled_reset();

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
    pub fn mlx_qwen35_vlm_get_offset() -> i32;
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

    /// Synchronously eval all MoE cache arrays (for training decode loop).
    pub fn mlx_qwen35_moe_sync_eval_caches();

    /// Reset MoE state.
    pub fn mlx_qwen35_moe_reset();

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
