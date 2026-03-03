#include "mlx_common.h"

// ============================================================================
// FUSED QWEN3 GENERATION
// ============================================================================
// This function implements the ENTIRE generation loop in C++, eliminating
// FFI overhead and matching mlx-lm's async pipelining pattern.

// KV cache chunk size for pre-allocation (matches mlx-lm step=256)
constexpr int KV_CACHE_CHUNK_SIZE = 256;

// Forward declaration for batched causal mask (defined later in the file)
static array create_batched_causal_mask(
    const array& left_padding,
    int seq_len,
    int kv_len,
    int cache_idx);

// Helper: Apply one transformer block with OPTIMIZED KV cache
// Uses pre-allocated buffers with slice_update for O(N) total instead of O(N²)
/// Transformer block forward with array RoPE offsets for batched generation.
///
/// Unlike the scalar offset version, this uses per-sequence offsets for RoPE
/// and creates a proper batched causal mask respecting left-padding.
/// This enables correct generation when group_size > 1.
static array transformer_block_forward_cached(
    const array& x,
    // Weights for this layer
    const array& input_norm_w,
    const array& post_attn_norm_w,
    const array& w_q, const array& w_k, const array& w_v, const array& w_o,
    const array* q_norm_w, const array* k_norm_w,
    const array& w_gate, const array& w_up, const array& w_down,
    // Config
    int n_heads, int n_kv_heads, int head_dim,
    float attn_scale, float rope_base, int rope_dims,
    float norm_eps, float qk_norm_eps,
    // KV cache (in/out) - shared across batch
    std::optional<array>& cached_keys, std::optional<array>& cached_values,
    int& cache_idx,  // Current write position in cache (shared)
    // Batched offsets for RoPE
    const array& rope_offsets,  // [batch] - per-sequence RoPE offsets
    // Left padding info for mask
    const array& left_padding   // [batch] - left padding amounts
) {
  int batch = static_cast<int>(x.shape()[0]);
  int seq_len = static_cast<int>(x.shape()[1]);

  // === Self-Attention ===
  auto normed = fast::rms_norm(x, std::optional<array>(input_norm_w), norm_eps, {});

  auto w_q_t = transpose(w_q, {1, 0});
  auto w_k_t = transpose(w_k, {1, 0});
  auto w_v_t = transpose(w_v, {1, 0});
  auto w_o_t = transpose(w_o, {1, 0});

  auto queries = matmul(normed, w_q_t);
  auto keys = matmul(normed, w_k_t);
  auto values = matmul(normed, w_v_t);

  queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
  keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
  values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

  // QK normalization (optional)
  if (q_norm_w) {
    queries = fast::rms_norm(queries, std::optional<array>(*q_norm_w), qk_norm_eps, {});
  }
  if (k_norm_w) {
    keys = fast::rms_norm(keys, std::optional<array>(*k_norm_w), qk_norm_eps, {});
  }

  queries = transpose(queries, {0, 2, 1, 3});
  keys = transpose(keys, {0, 2, 1, 3});
  values = transpose(values, {0, 2, 1, 3});
  // Shape: [batch, n_kv_heads, seq_len, head_dim]

  // RoPE with per-sequence position offsets
  // Each batch element may have processed different numbers of real tokens due to left-padding.
  // rope_offsets[b] = number of real tokens processed by batch element b = cache_idx - left_padding[b]
  //
  // For example, with cache_idx=14:
  // - Batch 0 (left_padding=0): rope_offset = 14 (14 real tokens)
  // - Batch 2 (left_padding=2): rope_offset = 12 (12 real tokens)
  //
  // We use the array overload of fast::rope to apply different offsets per batch element.
  bool traditional = false;
  float rope_scale = 1.0f;
  queries = fast::rope(queries, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offsets, std::nullopt, {});
  keys = fast::rope(keys, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offsets, std::nullopt, {});

  // === KV Cache Update (shared buffer, batch write) ===
  int prev_cache_idx = cache_idx;  // Save for mask computation
  int new_idx = cache_idx + seq_len;

  if (!cached_keys.has_value()) {
    // First call: allocate initial buffer
    int initial_capacity = ((seq_len + KV_CACHE_CHUNK_SIZE - 1) / KV_CACHE_CHUNK_SIZE) * KV_CACHE_CHUNK_SIZE;
    initial_capacity = std::max(initial_capacity, KV_CACHE_CHUNK_SIZE);

    auto buffer_shape = Shape{batch, n_kv_heads, initial_capacity, head_dim};
    cached_keys = zeros(buffer_shape, keys.dtype());
    cached_values = zeros(buffer_shape, values.dtype());

    // Insert initial keys/values at cache_idx
    cached_keys = slice_update(*cached_keys, keys, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
    cached_values = slice_update(*cached_values, values, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
  } else {
    // Subsequent calls: check if we need to expand buffer
    int current_capacity = static_cast<int>(cached_keys->shape()[2]);
    if (new_idx > current_capacity) {
      int new_capacity = ((new_idx + KV_CACHE_CHUNK_SIZE - 1) / KV_CACHE_CHUNK_SIZE) * KV_CACHE_CHUNK_SIZE;
      auto new_shape = Shape{batch, n_kv_heads, new_capacity, head_dim};

      auto new_k_buffer = zeros(new_shape, cached_keys->dtype());
      auto new_v_buffer = zeros(new_shape, cached_values->dtype());

      // Copy existing data
      new_k_buffer = slice_update(new_k_buffer, *cached_keys, {0, 0, 0, 0}, cached_keys->shape());
      new_v_buffer = slice_update(new_v_buffer, *cached_values, {0, 0, 0, 0}, cached_values->shape());

      cached_keys = new_k_buffer;
      cached_values = new_v_buffer;
    }

    // Insert new keys/values
    cached_keys = slice_update(*cached_keys, keys, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
    cached_values = slice_update(*cached_values, values, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
  }

  // Get valid keys/values up to new_idx
  auto keys_valid = slice(*cached_keys, {0, 0, 0, 0}, {batch, n_kv_heads, new_idx, head_dim});
  auto values_valid = slice(*cached_values, {0, 0, 0, 0}, {batch, n_kv_heads, new_idx, head_dim});

  // Update cache_idx for next call
  cache_idx = new_idx;

  // Compute attention output with proper left-padding aware mask
  // Create mask that respects left-padding (prevents attending to padding tokens)
  auto mask = create_batched_causal_mask(left_padding, seq_len, new_idx, prev_cache_idx);

  // MLX SDPA: mask where true = attend (keep), false = mask out
  // create_batched_causal_mask returns true = attend, so pass directly
  // NOTE: Do NOT call eval() here - this is called per-layer and would destroy pipelining
  auto attn_output = fast::scaled_dot_product_attention(queries, keys_valid, values_valid, attn_scale, "", mask, std::nullopt, {});

  attn_output = transpose(attn_output, {0, 2, 1, 3});
  attn_output = reshape(attn_output, {batch, seq_len, n_heads * head_dim});
  attn_output = matmul(attn_output, w_o_t);

  auto h = x + attn_output;

  // === MLP ===
  auto mlp_input = fast::rms_norm(h, std::optional<array>(post_attn_norm_w), norm_eps, {});
  auto w_gate_t = transpose(w_gate, {1, 0});
  auto w_up_t = transpose(w_up, {1, 0});
  auto w_down_t = transpose(w_down, {1, 0});

  auto gate = matmul(mlp_input, w_gate_t);
  auto up = matmul(mlp_input, w_up_t);
  auto activated = mlx::core::sigmoid(gate) * gate * up;  // SiLU(gate) * up
  auto mlp_output = matmul(activated, w_down_t);

  auto output = h + mlp_output;
  // NOTE: Do NOT call eval() or synchronize() here - it would be called 28 times per forward!
  // MLX uses lazy evaluation and syncs are handled at appropriate boundaries
  return output;
}

// Helper: Run forward through all layers and get logits
// Uses array offsets for batched generation with proper per-sequence RoPE positions.
static array forward_all_layers(
    const array& input_ids,
    const array& embedding_weight,
    mlx_array** layer_weights,
    int num_layers,
    const array& final_norm_w,
    const array* lm_head_w,
    bool tie_word_embeddings,
    // Config
    int hidden_size, int n_heads, int n_kv_heads, int head_dim,
    float attn_scale, float rope_base, float norm_eps,
    // KV caches (use optional since array has no default ctor)
    std::vector<std::optional<array>>& kv_keys,
    std::vector<std::optional<array>>& kv_values,
    int& cache_idx,  // Single shared cache index (updated on return)
    // Array offsets for batched generation
    const array& rope_offsets,  // [batch] - per-sequence RoPE offsets
    const array& left_padding   // [batch] - left padding amounts
) {

  // Embedding lookup
  auto hidden = take(embedding_weight, input_ids, 0);

  // Initialize per-layer cache indices (all start at same value, advance together)
  std::vector<int> cache_indices(num_layers, cache_idx);

  // Process each layer
  for (int i = 0; i < num_layers; i++) {
    // Extract weights for this layer (11 weights per layer)
    int base = i * 11;
    auto& input_norm_w = *reinterpret_cast<array*>(layer_weights[base + 0]);
    auto& post_attn_norm_w = *reinterpret_cast<array*>(layer_weights[base + 1]);
    auto& w_q = *reinterpret_cast<array*>(layer_weights[base + 2]);
    auto& w_k = *reinterpret_cast<array*>(layer_weights[base + 3]);
    auto& w_v = *reinterpret_cast<array*>(layer_weights[base + 4]);
    auto& w_o = *reinterpret_cast<array*>(layer_weights[base + 5]);
    array* q_norm_w = layer_weights[base + 6] ? reinterpret_cast<array*>(layer_weights[base + 6]) : nullptr;
    array* k_norm_w = layer_weights[base + 7] ? reinterpret_cast<array*>(layer_weights[base + 7]) : nullptr;
    auto& w_gate = *reinterpret_cast<array*>(layer_weights[base + 8]);
    auto& w_up = *reinterpret_cast<array*>(layer_weights[base + 9]);
    auto& w_down = *reinterpret_cast<array*>(layer_weights[base + 10]);

    // Each layer has its own cache index (all start at same value, advance independently)
    hidden = transformer_block_forward_cached(
        hidden,
        input_norm_w, post_attn_norm_w,
        w_q, w_k, w_v, w_o, q_norm_w, k_norm_w,
        w_gate, w_up, w_down,
        n_heads, n_kv_heads, head_dim,
        attn_scale, rope_base, head_dim, // rope_dims = head_dim
        norm_eps, norm_eps, // qk_norm_eps = norm_eps
        kv_keys[i], kv_values[i], cache_indices[i],
        rope_offsets, left_padding);
  }

  // Update output cache_idx (all layers should have advanced by same amount)
  cache_idx = cache_indices[0];

  // Final normalization
  hidden = fast::rms_norm(hidden, std::optional<array>(final_norm_w), norm_eps, {});

  // LM head
  // NOTE: Do NOT call eval() or synchronize() here - MLX uses lazy evaluation
  auto logits = tie_word_embeddings
    ? matmul(hidden, transpose(embedding_weight, {1, 0}))
    : matmul(hidden, transpose(*lm_head_w, {1, 0}));
  return logits;
}

// Helper: Sample from logprobs with full filtering support
// Uses compiled kernels for performance, matches mlx_compiled_sample_and_logprobs
static array sample_with_filters(
    const array& logprobs_in,
    float temperature,
    int top_k,
    float top_p,
    float min_p) {

  // Greedy fast path
  if (temperature == 0.0f) {
    return argmax(logprobs_in, -1);
  }

  // Fast path: no filters enabled - use compiled sampler directly
  bool needs_filters = (top_k > 0) || (top_p > 0.0f && top_p < 1.0f) || (min_p > 0.0f);
  if (!needs_filters) {
    static auto compiled_sampler = mlx::core::compile([](const std::vector<array>& inputs) {
      auto lp = inputs[0];
      auto temp_scalar = inputs[1];
      auto scaled = multiply(lp, temp_scalar);
      return std::vector<array>{mlx::core::random::categorical(scaled, -1)};
    });
    auto temp_array = array(1.0f / temperature);
    auto results = compiled_sampler({logprobs_in, temp_array});
    return results[0];
  }

  auto logprobs = logprobs_in;

  // Apply top_k filter
  if (top_k > 0) {
    int vocab_size = logprobs.shape().back();
    if (top_k < vocab_size) {
      auto neg_logprobs = negative(logprobs);
      auto partitioned_indices = argpartition(neg_logprobs, top_k - 1, -1);
      auto shape = partitioned_indices.shape();
      mlx::core::Shape starts(shape.size(), 0);
      mlx::core::Shape ends(shape.begin(), shape.end());
      starts[starts.size() - 1] = top_k;
      auto mask_idx = slice(partitioned_indices, starts, ends);
      auto neg_inf = array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
      logprobs = put_along_axis(logprobs, mask_idx, neg_inf, -1);
    }
  }

  // Apply top_p (nucleus) filter
  if (top_p > 0.0f && top_p < 1.0f) {
    auto probs = exp(logprobs);
    auto sorted_indices = argsort(logprobs, -1);
    auto sorted_probs = take_along_axis(probs, sorted_indices, -1);
    auto cumulative_probs = cumsum(sorted_probs, -1);
    auto shape = sorted_indices.shape();
    int last_dim = shape[shape.size() - 1];
    auto arange_vals = arange(0, last_dim, sorted_indices.dtype());
    auto zeros_arr = zeros_like(sorted_indices);
    auto inverse_indices = put_along_axis(zeros_arr, sorted_indices, arange_vals, -1);
    cumulative_probs = take_along_axis(cumulative_probs, inverse_indices, -1);
    // Subtract epsilon for numerical stability (consistent with Rust path)
    constexpr float EPSILON = 1e-7f;
    auto threshold = array((1.0f - top_p) - EPSILON);
    auto mask = greater(cumulative_probs, threshold);
    auto neg_inf = array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
    logprobs = mlx::core::where(mask, logprobs, neg_inf);
  }

  // Apply min_p filter
  if (min_p > 0.0f) {
    auto sorted_indices = argsort(negative(logprobs), -1);
    auto sorted_logprobs = take_along_axis(logprobs, sorted_indices, -1);
    auto shape = sorted_logprobs.shape();
    mlx::core::Shape starts(shape.size(), 0);
    mlx::core::Shape ends(shape.begin(), shape.end());
    ends[ends.size() - 1] = 1;
    auto max_logprob = slice(sorted_logprobs, starts, ends);
    auto threshold = max_logprob + log(array(min_p));
    auto mask = less(logprobs, threshold);
    auto neg_inf = array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
    logprobs = mlx::core::where(mask, neg_inf, logprobs);
  }

  // Use compiled sampler for the actual sampling
  static auto compiled_sampler = mlx::core::compile([](const std::vector<array>& inputs) {
    auto lp = inputs[0];
    auto temp_scalar = inputs[1];
    auto scaled = multiply(lp, temp_scalar);
    return std::vector<array>{mlx::core::random::categorical(scaled, -1)};
  });
  auto temp_array = array(1.0f / temperature);
  auto results = compiled_sampler({logprobs, temp_array});
  return results[0];
}

// ============================================================================
// BATCHED FORWARD STEP - True batch generation with array RoPE offsets
// ============================================================================
// This function performs batched forward passes with per-sequence RoPE offsets,
// enabling true parallel batch generation with left-padded sequences.
// Uses BatchKVCache semantics where each sequence can have different offsets.

/// Create a batched causal attention mask that respects left-padding.
///
/// Handles both prefill (seq_len > 1) and decode (seq_len = 1) phases:
/// - Prefill: Creates triangular causal mask where each query attends to positions [left_pad..query_pos]
/// - Decode: Creates single-row mask where query attends to all valid cached positions
///
/// IMPORTANT: Padding positions (q_pos < left_padding) get a special mask that allows
/// them to attend only to position 0 (themselves or other padding), preventing NaN from
/// softmax(all -inf). This doesn't affect output quality since padding positions are ignored.
///
/// Returns: [batch, 1, seq_len, kv_len] boolean mask (true = attend, false = mask)
static array create_batched_causal_mask(
    const array& left_padding,  // [batch] - padding for each sequence
    int seq_len,                // Current query sequence length
    int kv_len,                 // Total KV cache length (cache_idx + seq_len)
    int cache_idx               // Current cache write position (where new tokens start)
) {
    int batch = static_cast<int>(left_padding.shape()[0]);

    // Create KV position indices: [1, 1, 1, kv_len]
    auto kv_positions = arange(0, kv_len, mlx::core::int32);
    kv_positions = reshape(kv_positions, {1, 1, 1, kv_len});

    // Create query positions: [1, 1, seq_len, 1]
    // For prefill (cache_idx=0): query positions are [0, 1, ..., seq_len-1]
    // For decode (seq_len=1): query position is [cache_idx]
    auto q_positions = arange(cache_idx, cache_idx + seq_len, mlx::core::int32);
    q_positions = reshape(q_positions, {1, 1, seq_len, 1});

    // Left padding expanded: [batch, 1, 1, 1]
    auto left_pad = reshape(left_padding, {batch, 1, 1, 1});

    // Mask out positions before left_padding (padded positions)
    // valid_kv = kv_positions >= left_padding
    auto valid_kv = greater_equal(kv_positions, left_pad);

    // Causal constraint: each query can attend to KV positions <= its own position
    // This creates a triangular mask for prefill and a single-row mask for decode
    auto causal_mask = less_equal(kv_positions, q_positions);

    // Combined mask: both valid (not padded) AND causal
    auto mask = logical_and(valid_kv, causal_mask);

    // CRITICAL: For padding query positions (q_pos < left_padding), the mask would be all-false,
    // causing softmax(all -inf) = NaN. To prevent this, let padding queries attend to position 0.
    // This produces a valid softmax output for padding positions (which are ignored anyway).
    //
    // is_padding_query[b, 1, q, 1] = (q_positions < left_pad[b])
    // For q_pos < left_pad: allow attending to kv_pos=0 to avoid NaN
    auto is_padding_query = less(q_positions, left_pad);  // [batch, 1, seq_len, 1]
    auto kv_pos_zero = equal(kv_positions, array(0));     // [1, 1, 1, kv_len] - true only at pos 0
    auto padding_fallback = logical_and(is_padding_query, kv_pos_zero);  // [batch, 1, seq_len, kv_len]
    mask = logical_or(mask, padding_fallback);

    // Broadcast to [batch, 1, seq_len, kv_len] for SDPA
    // The batch dimension broadcasts from left_pad, seq_len from q_positions
    mask = broadcast_to(mask, {batch, 1, seq_len, kv_len});

    return mask;
}

/// Batched transformer block forward with array RoPE offsets.
///
/// Unlike the scalar offset version, this uses per-sequence offsets for RoPE
/// and creates a proper batched causal mask respecting left-padding.
static array transformer_block_forward_batched(
    const array& x,
    // Weights for this layer
    const array& input_norm_w,
    const array& post_attn_norm_w,
    const array& w_q, const array& w_k, const array& w_v, const array& w_o,
    const array* q_norm_w, const array* k_norm_w,
    const array& w_gate, const array& w_up, const array& w_down,
    // Config
    int n_heads, int n_kv_heads, int head_dim,
    float attn_scale, float rope_base, int rope_dims,
    float norm_eps, float qk_norm_eps,
    // KV cache (in/out) - shared across batch
    std::optional<array>& cached_keys, std::optional<array>& cached_values,
    int& cache_idx,  // Current write position in cache
    // Batched offsets for RoPE
    const array& rope_offsets,  // [batch] - per-sequence RoPE offsets
    // Left padding info for mask
    const array& left_padding   // [batch] - left padding amounts
) {
    int batch = static_cast<int>(x.shape()[0]);
    int seq_len = static_cast<int>(x.shape()[1]);

    // === Self-Attention ===
    auto normed = fast::rms_norm(x, std::optional<array>(input_norm_w), norm_eps, {});

    auto w_q_t = transpose(w_q, {1, 0});
    auto w_k_t = transpose(w_k, {1, 0});
    auto w_v_t = transpose(w_v, {1, 0});
    auto w_o_t = transpose(w_o, {1, 0});

    auto queries = matmul(normed, w_q_t);
    auto keys = matmul(normed, w_k_t);
    auto values = matmul(normed, w_v_t);

    queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
    keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
    values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

    // QK normalization (optional)
    if (q_norm_w) {
        queries = fast::rms_norm(queries, std::optional<array>(*q_norm_w), qk_norm_eps, {});
    }
    if (k_norm_w) {
        keys = fast::rms_norm(keys, std::optional<array>(*k_norm_w), qk_norm_eps, {});
    }

    queries = transpose(queries, {0, 2, 1, 3});
    keys = transpose(keys, {0, 2, 1, 3});
    values = transpose(values, {0, 2, 1, 3});
    // Shape: [batch, n_kv_heads, seq_len, head_dim]

    // RoPE with per-sequence position offsets
    // Each batch element may have processed different numbers of real tokens due to left-padding.
    // rope_offsets[b] = number of real tokens processed by batch element b = cache_idx - left_padding[b]
    //
    // For example, with cache_idx=14:
    // - Batch 0 (left_padding=0): rope_offset = 14 (14 real tokens)
    // - Batch 2 (left_padding=2): rope_offset = 12 (12 real tokens)
    //
    // We use the array overload of fast::rope to apply different offsets per batch element.
    bool traditional = false;
    float rope_scale = 1.0f;
    queries = fast::rope(queries, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offsets, std::nullopt, {});
    keys = fast::rope(keys, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offsets, std::nullopt, {});

    // === KV Cache Update (shared buffer, batch write) ===
    int prev_cache_idx = cache_idx;  // Save for mask computation
    int new_idx = cache_idx + seq_len;

    if (!cached_keys.has_value()) {
        // First call: allocate initial buffer
        int initial_capacity = ((seq_len + KV_CACHE_CHUNK_SIZE - 1) / KV_CACHE_CHUNK_SIZE) * KV_CACHE_CHUNK_SIZE;
        initial_capacity = std::max(initial_capacity, KV_CACHE_CHUNK_SIZE);

        auto buffer_shape = Shape{batch, n_kv_heads, initial_capacity, head_dim};
        cached_keys = zeros(buffer_shape, keys.dtype());
        cached_values = zeros(buffer_shape, values.dtype());

        // Insert initial keys/values at cache_idx
        cached_keys = slice_update(*cached_keys, keys, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
        cached_values = slice_update(*cached_values, values, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
    } else {
        // Subsequent calls: check if we need to expand buffer
        int current_capacity = static_cast<int>(cached_keys->shape()[2]);
        if (new_idx > current_capacity) {
            int new_capacity = ((new_idx + KV_CACHE_CHUNK_SIZE - 1) / KV_CACHE_CHUNK_SIZE) * KV_CACHE_CHUNK_SIZE;
            auto new_shape = Shape{batch, n_kv_heads, new_capacity, head_dim};

            auto new_k_buffer = zeros(new_shape, cached_keys->dtype());
            auto new_v_buffer = zeros(new_shape, cached_values->dtype());

            // Copy existing data
            new_k_buffer = slice_update(new_k_buffer, *cached_keys, {0, 0, 0, 0}, cached_keys->shape());
            new_v_buffer = slice_update(new_v_buffer, *cached_values, {0, 0, 0, 0}, cached_values->shape());

            cached_keys = new_k_buffer;
            cached_values = new_v_buffer;
        }

        // Insert new keys/values
        cached_keys = slice_update(*cached_keys, keys, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
        cached_values = slice_update(*cached_values, values, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
    }

    // Get valid keys/values up to new_idx
    auto keys_valid = slice(*cached_keys, {0, 0, 0, 0}, {batch, n_kv_heads, new_idx, head_dim});
    auto values_valid = slice(*cached_values, {0, 0, 0, 0}, {batch, n_kv_heads, new_idx, head_dim});

    // Update cache_idx for next call
    cache_idx = new_idx;

    // Compute attention output with proper left-padding aware mask
    // Create mask that respects left-padding (prevents attending to padding tokens)
    auto mask = create_batched_causal_mask(left_padding, seq_len, new_idx, prev_cache_idx);

    // MLX SDPA: mask where true = attend (keep), false = mask out
    // create_batched_causal_mask returns true = attend, so pass directly
    // NOTE: Do NOT call eval() here - this is called per-layer and would destroy pipelining
    auto attn_output = fast::scaled_dot_product_attention(queries, keys_valid, values_valid, attn_scale, "", mask, std::nullopt, {});

    attn_output = transpose(attn_output, {0, 2, 1, 3});
    attn_output = reshape(attn_output, {batch, seq_len, n_heads * head_dim});
    attn_output = matmul(attn_output, w_o_t);

    auto h = x + attn_output;

    // === MLP ===
    auto mlp_input = fast::rms_norm(h, std::optional<array>(post_attn_norm_w), norm_eps, {});
    auto w_gate_t = transpose(w_gate, {1, 0});
    auto w_up_t = transpose(w_up, {1, 0});
    auto w_down_t = transpose(w_down, {1, 0});

    auto gate = matmul(mlp_input, w_gate_t);
    auto up = matmul(mlp_input, w_up_t);
    auto activated = mlx::core::sigmoid(gate) * gate * up;  // SiLU(gate) * up
    auto mlp_output = matmul(activated, w_down_t);

    // NOTE: Do NOT call eval() or synchronize() here - called 28 times per forward!
    auto output = h + mlp_output;
    return output;
}

// =============================================================================
// Fused Multimodal Rotary Position Embedding (mRoPE)
// =============================================================================
// Replaces ~38 individual graph ops per call with a single fused operation.
// Called once per transformer layer (28x per decode step), saving ~1000 graph nodes.
//
// Equivalent Python:
//   mrope_section = cumsum(mrope_section * 2)[:-1]
//   cos = concat([m[i%3] for i,m in enumerate(split(cos, mrope_section, -1))], -1)
//   sin = concat([m[i%3] for i,m in enumerate(split(sin, mrope_section, -1))], -1)
//   q_embed = q * cos + rotate_half(q) * sin
//   k_embed = k * cos + rotate_half(k) * sin

// Helper: rotate_half - split last dim in half, negate second half, concatenate reversed
static array rotate_half_impl(const array& x) {
    int ndim = x.ndim();
    ShapeElem last_dim = x.shape(ndim - 1);
    ShapeElem half = last_dim / 2;

    // Build start/end vectors for slice along last axis only
    // slice(array, start, end) where start/end have one entry per dimension
    Shape start_x1(ndim, 0);
    Shape end_x1(x.shape().begin(), x.shape().end());
    end_x1[ndim - 1] = half;

    Shape start_x2(ndim, 0);
    start_x2[ndim - 1] = half;
    Shape end_x2(x.shape().begin(), x.shape().end());

    auto x1 = slice(x, start_x1, end_x1);
    auto x2 = slice(x, start_x2, end_x2);

    // Negate x2 and concatenate: [-x2, x1]
    auto neg_x2 = -x2;
    return concatenate({neg_x2, x1}, ndim - 1);
}

// ============================================================================
// Compiled SwiGLU activation
// ============================================================================
// Python's nn.silu uses @mx.compile to fuse sigmoid(x)*x into a single Metal
// kernel. We do the same here for the full SwiGLU: silu(gate) * up.
// This avoids 2 extra memory round-trips per layer (18 layers = 36 saved).

static std::vector<array> swiglu_impl(const std::vector<array>& inputs) {
    const auto& gate = inputs[0];
    const auto& up = inputs[1];
    return {mlx::core::sigmoid(gate) * gate * up};
}

static auto& compiled_swiglu() {
    static auto fn = mlx::core::compile(swiglu_impl, /*shapeless=*/true);
    return fn;
}

// ============================================================================
// PaddleOCR-VL Fused Forward Pass
// ============================================================================
// Implements the full ERNIE language model forward pass in C++ to eliminate
// ~1400 NAPI FFI calls per decode step, reducing them to ~6.
// Uses mRoPE (multimodal rotary position embedding) instead of standard RoPE.

// Helper: compute mRoPE cos/sin from inv_freq and position_ids
// inv_freq: [1, 1, half_dim, 1], position_ids: [3, batch, seq_len]
// Returns (cos, sin) each [3, batch, seq_len, head_dim]
static std::pair<array, array> compute_mrope_cos_sin(
    const array& inv_freq,
    const array& position_ids,
    int head_dim
) {
    int batch = static_cast<int>(position_ids.shape(1));
    int seq_len = static_cast<int>(position_ids.shape(2));
    int half_dim = head_dim / 2;

    // Broadcast inv_freq to [3, batch, half_dim, 1]
    auto inv_freq_expanded = broadcast_to(inv_freq, {3, batch, half_dim, 1});

    // position_ids: [3, batch, seq_len] -> [3, batch, 1, seq_len]
    auto pos_expanded = astype(reshape(position_ids, {3, batch, 1, seq_len}), mlx::core::float32);

    // freqs = inv_freq @ position_ids -> [3, batch, half_dim, seq_len]
    auto freqs = matmul(inv_freq_expanded, pos_expanded);

    // Transpose to [3, batch, seq_len, half_dim]
    freqs = transpose(freqs, {0, 1, 3, 2});

    // Double the freqs: [3, batch, seq_len, head_dim]
    auto emb = concatenate({freqs, freqs}, -1);

    return {cos(emb), sin(emb)};
}

// Pre-compute sectioned mRoPE cos/sin from raw cos/sin [3, batch, seq_len, head_dim]
// Returns cos_final, sin_final: [batch, 1, seq_len, head_dim] ready for direct multiply
static std::pair<array, array> compute_mrope_sectioned(
    const array& cos_in,
    const array& sin_in,
    const int* mrope_section
) {
    int b0 = mrope_section[0] * 2;
    int b1 = b0 + mrope_section[1] * 2;

    int batch = static_cast<int>(cos_in.shape(1));
    int seq_len = static_cast<int>(cos_in.shape(2));
    int head_dim_full = static_cast<int>(cos_in.shape(3));

    Shape split_indices = {(ShapeElem)b0, (ShapeElem)b1};

    auto cos_parts = split(cos_in, split_indices, -1);
    auto sin_parts = split(sin_in, split_indices, -1);

    std::vector<array> cos_selected, sin_selected;
    for (int i = 0; i < 3; i++) {
        ShapeElem sec_dim = cos_parts[i].shape(3);
        auto ci = slice(cos_parts[i],
                       {(ShapeElem)i, 0, 0, 0},
                       {(ShapeElem)(i + 1), (ShapeElem)batch, (ShapeElem)seq_len, sec_dim});
        auto si = slice(sin_parts[i],
                       {(ShapeElem)i, 0, 0, 0},
                       {(ShapeElem)(i + 1), (ShapeElem)batch, (ShapeElem)seq_len, sec_dim});
        cos_selected.push_back(squeeze(ci, 0));
        sin_selected.push_back(squeeze(si, 0));
    }

    auto cos_final = reshape(concatenate(cos_selected, -1), {batch, 1, seq_len, head_dim_full});
    auto sin_final = reshape(concatenate(sin_selected, -1), {batch, 1, seq_len, head_dim_full});

    return {cos_final, sin_final};
}

// Apply pre-computed mRoPE cos/sin to Q and K
// q, k: [batch, n_heads, seq_len, head_dim]
// cos_final, sin_final: [batch, 1, seq_len, head_dim] (already sectioned)
static std::pair<array, array> apply_mrope_paddleocr(
    const array& q,
    const array& k,
    const array& cos_final,
    const array& sin_final
) {
    auto q_embed = q * cos_final + rotate_half_impl(q) * sin_final;
    auto k_embed = k * cos_final + rotate_half_impl(k) * sin_final;
    return {q_embed, k_embed};
}

// Helper: single PaddleOCR-VL transformer block forward with KV cache
// Takes pre-sectioned cos/sin (no per-layer splitting) and pre-transposed weights
static array paddleocr_vl_block_forward_cached(
    const array& x,
    const array& input_norm_w,
    const array& post_attn_norm_w,
    // Pre-transposed weight matrices [out, in] -> [in, out]
    const array& w_q_t, const array& w_k_t, const array& w_v_t, const array& w_o_t,
    const array& w_gate_t, const array& w_up_t, const array& w_down_t,
    int n_heads, int n_kv_heads, int head_dim,
    float attn_scale, float norm_eps,
    // Pre-sectioned mRoPE: [batch, 1, seq_len, head_dim]
    const array& cos_final, const array& sin_final,
    std::optional<array>& cached_keys, std::optional<array>& cached_values,
    int& cache_idx
) {
    int batch = static_cast<int>(x.shape(0));
    int seq_len = static_cast<int>(x.shape(1));

    // Self-Attention
    auto normed = fast::rms_norm(x, std::optional<array>(input_norm_w), norm_eps, {});

    auto queries = matmul(normed, w_q_t);
    auto keys = matmul(normed, w_k_t);
    auto values = matmul(normed, w_v_t);

    // Reshape + transpose to [batch, n_heads, seq_len, head_dim]
    queries = transpose(reshape(queries, {batch, seq_len, n_heads, head_dim}), {0, 2, 1, 3});
    keys = transpose(reshape(keys, {batch, seq_len, n_kv_heads, head_dim}), {0, 2, 1, 3});
    values = transpose(reshape(values, {batch, seq_len, n_kv_heads, head_dim}), {0, 2, 1, 3});

    // Apply pre-sectioned mRoPE (just multiply, no splitting)
    auto [q_rotated, k_rotated] = apply_mrope_paddleocr(queries, keys, cos_final, sin_final);

    // KV Cache Update
    int new_idx = cache_idx + seq_len;

    if (!cached_keys.has_value()) {
        int initial_capacity = ((seq_len + KV_CACHE_CHUNK_SIZE - 1) / KV_CACHE_CHUNK_SIZE) * KV_CACHE_CHUNK_SIZE;
        initial_capacity = std::max(initial_capacity, KV_CACHE_CHUNK_SIZE);

        auto buffer_shape = Shape{batch, n_kv_heads, initial_capacity, head_dim};
        cached_keys = zeros(buffer_shape, k_rotated.dtype());
        cached_values = zeros(buffer_shape, values.dtype());

        cached_keys = slice_update(*cached_keys, k_rotated, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
        cached_values = slice_update(*cached_values, values, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
    } else {
        int current_capacity = static_cast<int>(cached_keys->shape()[2]);
        if (new_idx > current_capacity) {
            int new_capacity = ((new_idx + KV_CACHE_CHUNK_SIZE - 1) / KV_CACHE_CHUNK_SIZE) * KV_CACHE_CHUNK_SIZE;
            auto new_shape = Shape{batch, n_kv_heads, new_capacity, head_dim};
            auto new_k_buffer = zeros(new_shape, cached_keys->dtype());
            auto new_v_buffer = zeros(new_shape, cached_values->dtype());
            new_k_buffer = slice_update(new_k_buffer, *cached_keys, {0, 0, 0, 0}, cached_keys->shape());
            new_v_buffer = slice_update(new_v_buffer, *cached_values, {0, 0, 0, 0}, cached_values->shape());
            cached_keys = new_k_buffer;
            cached_values = new_v_buffer;
        }
        cached_keys = slice_update(*cached_keys, k_rotated, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
        cached_values = slice_update(*cached_values, values, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
    }

    auto keys_valid = slice(*cached_keys, {0, 0, 0, 0}, {batch, n_kv_heads, new_idx, head_dim});
    auto values_valid = slice(*cached_values, {0, 0, 0, 0}, {batch, n_kv_heads, new_idx, head_dim});
    cache_idx = new_idx;

    // SDPA: causal for prefill, no mask for decode
    auto attn_output = (seq_len > 1)
        ? fast::scaled_dot_product_attention(q_rotated, keys_valid, values_valid, attn_scale, "causal", std::nullopt, std::nullopt, {})
        : fast::scaled_dot_product_attention(q_rotated, keys_valid, values_valid, attn_scale, "", std::nullopt, std::nullopt, {});

    // Output projection + residual
    attn_output = transpose(attn_output, {0, 2, 1, 3});
    attn_output = reshape(attn_output, {batch, seq_len, n_heads * head_dim});
    attn_output = matmul(attn_output, w_o_t);
    auto h = x + attn_output;

    // MLP (SwiGLU) + residual
    auto mlp_input = fast::rms_norm(h, std::optional<array>(post_attn_norm_w), norm_eps, {});

    auto gate = matmul(mlp_input, w_gate_t);
    auto up = matmul(mlp_input, w_up_t);
    auto activated = compiled_swiglu()({gate, up})[0];
    auto mlp_output = matmul(activated, w_down_t);

    return h + mlp_output;
}

// Helper: single PaddleOCR-VL transformer block forward with KV cache + batched mask
// Like paddleocr_vl_block_forward_cached but uses create_batched_causal_mask for
// left-padding-aware attention masking needed during batched decode.
static array paddleocr_vl_block_forward_batched(
    const array& x,
    const array& input_norm_w,
    const array& post_attn_norm_w,
    const array& w_q_t, const array& w_k_t, const array& w_v_t, const array& w_o_t,
    const array& w_gate_t, const array& w_up_t, const array& w_down_t,
    int n_heads, int n_kv_heads, int head_dim,
    float attn_scale, float norm_eps,
    const array& cos_final, const array& sin_final,
    std::optional<array>& cached_keys, std::optional<array>& cached_values,
    int& cache_idx,
    const array& left_padding
) {
    int batch = static_cast<int>(x.shape(0));
    int seq_len = static_cast<int>(x.shape(1));

    // Self-Attention
    auto normed = fast::rms_norm(x, std::optional<array>(input_norm_w), norm_eps, {});

    auto queries = matmul(normed, w_q_t);
    auto keys = matmul(normed, w_k_t);
    auto values = matmul(normed, w_v_t);

    queries = transpose(reshape(queries, {batch, seq_len, n_heads, head_dim}), {0, 2, 1, 3});
    keys = transpose(reshape(keys, {batch, seq_len, n_kv_heads, head_dim}), {0, 2, 1, 3});
    values = transpose(reshape(values, {batch, seq_len, n_kv_heads, head_dim}), {0, 2, 1, 3});

    // Apply pre-sectioned mRoPE
    auto [q_rotated, k_rotated] = apply_mrope_paddleocr(queries, keys, cos_final, sin_final);

    // KV Cache Update
    int prev_cache_idx = cache_idx;
    int new_idx = cache_idx + seq_len;

    if (!cached_keys.has_value()) {
        int initial_capacity = ((seq_len + KV_CACHE_CHUNK_SIZE - 1) / KV_CACHE_CHUNK_SIZE) * KV_CACHE_CHUNK_SIZE;
        initial_capacity = std::max(initial_capacity, KV_CACHE_CHUNK_SIZE);

        auto buffer_shape = Shape{batch, n_kv_heads, initial_capacity, head_dim};
        cached_keys = zeros(buffer_shape, k_rotated.dtype());
        cached_values = zeros(buffer_shape, values.dtype());

        cached_keys = slice_update(*cached_keys, k_rotated, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
        cached_values = slice_update(*cached_values, values, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
    } else {
        int current_capacity = static_cast<int>(cached_keys->shape()[2]);
        if (new_idx > current_capacity) {
            int new_capacity = ((new_idx + KV_CACHE_CHUNK_SIZE - 1) / KV_CACHE_CHUNK_SIZE) * KV_CACHE_CHUNK_SIZE;
            auto new_shape = Shape{batch, n_kv_heads, new_capacity, head_dim};
            auto new_k_buffer = zeros(new_shape, cached_keys->dtype());
            auto new_v_buffer = zeros(new_shape, cached_values->dtype());
            new_k_buffer = slice_update(new_k_buffer, *cached_keys, {0, 0, 0, 0}, cached_keys->shape());
            new_v_buffer = slice_update(new_v_buffer, *cached_values, {0, 0, 0, 0}, cached_values->shape());
            cached_keys = new_k_buffer;
            cached_values = new_v_buffer;
        }
        cached_keys = slice_update(*cached_keys, k_rotated, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
        cached_values = slice_update(*cached_values, values, {0, 0, cache_idx, 0}, {batch, n_kv_heads, new_idx, head_dim});
    }

    auto keys_valid = slice(*cached_keys, {0, 0, 0, 0}, {batch, n_kv_heads, new_idx, head_dim});
    auto values_valid = slice(*cached_values, {0, 0, 0, 0}, {batch, n_kv_heads, new_idx, head_dim});
    cache_idx = new_idx;

    // Create left-padding-aware attention mask
    auto mask = create_batched_causal_mask(left_padding, seq_len, new_idx, prev_cache_idx);
    auto attn_output = fast::scaled_dot_product_attention(q_rotated, keys_valid, values_valid, attn_scale, "", mask, std::nullopt, {});

    // Output projection + residual
    attn_output = transpose(attn_output, {0, 2, 1, 3});
    attn_output = reshape(attn_output, {batch, seq_len, n_heads * head_dim});
    attn_output = matmul(attn_output, w_o_t);
    auto h = x + attn_output;

    // MLP (SwiGLU) + residual
    auto mlp_input = fast::rms_norm(h, std::optional<array>(post_attn_norm_w), norm_eps, {});
    auto gate = matmul(mlp_input, w_gate_t);
    auto up = matmul(mlp_input, w_up_t);
    auto activated = compiled_swiglu()({gate, up})[0];
    auto mlp_output = matmul(activated, w_down_t);

    return h + mlp_output;
}

// ============================================================================
// Public FFI functions
// ============================================================================

extern "C" {

// Main fused generation function
// Implements entire generation loop in C++ with async pipelining
void mlx_qwen3_generate(
    // Input prompt
    mlx_array* input_ids_handle,

    // Model weights
    mlx_array* embedding_weight_handle,
    mlx_array** layer_weights,  // [num_layers * 11] weights
    int num_layers,
    mlx_array* final_norm_weight_handle,
    mlx_array* lm_head_weight_handle,  // Can be null if tied
    bool tie_word_embeddings,

    // Model config
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float rope_theta,
    float norm_eps,

    // Generation config
    int max_new_tokens,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    float repetition_penalty,
    int repetition_context_size,
    int eos_token_id,

    // Outputs (caller allocates)
    int32_t* out_tokens,
    float* out_logprobs,
    int* out_num_tokens,
    int* out_finish_reason) {

  auto input_ids = *reinterpret_cast<array*>(input_ids_handle);
  auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_handle);
  auto& final_norm_w = *reinterpret_cast<array*>(final_norm_weight_handle);
  array* lm_head_w = lm_head_weight_handle ? reinterpret_cast<array*>(lm_head_weight_handle) : nullptr;

  float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Set wired limit to max recommended (matches mlx-lm wired_limit context manager)
  // This keeps model weights in fast GPU memory
  size_t old_wired_limit = 0;
  if (mlx::core::metal::is_available()) {
    auto& info = mlx::core::gpu::device_info();
    size_t max_rec = std::get<size_t>(info.at("max_recommended_working_set_size"));
    old_wired_limit = mlx::core::set_wired_limit(max_rec);
  }

  // Create dedicated generation stream (matches mlx-lm line 216)
  auto generation_stream = new_stream(default_device());

  // Initialize KV caches for all layers (use optional since array has no default ctor)
  std::vector<std::optional<array>> kv_keys(num_layers);
  std::vector<std::optional<array>> kv_values(num_layers);
  int cache_idx = 0;  // Shared cache write position

  // For single-sequence generation: batch=1, no left padding
  array rope_offsets = array({0}, {1}, mlx::core::int32);
  array left_padding = array({0}, {1}, mlx::core::int32);

  // Track recent tokens for repetition penalty
  std::vector<int32_t> recent_tokens;
  auto input_flat = flatten(input_ids);
  eval(input_flat);
  auto input_data = input_flat.data<int32_t>();
  for (size_t i = 0; i < input_flat.size(); i++) {
    recent_tokens.push_back(input_data[i]);
  }

  // === Prefill: Process entire prompt ===
  // Initialize y and logprobs_arr inside the lambda to avoid default construction
  auto prefill_result = [&]() -> std::pair<array, array> {
    StreamContext ctx(generation_stream);
    auto logits = forward_all_layers(
        input_ids, embedding_weight, layer_weights, num_layers,
        final_norm_w, lm_head_w, tie_word_embeddings,
        hidden_size, num_heads, num_kv_heads, head_dim,
        attn_scale, rope_theta, norm_eps,
        kv_keys, kv_values, cache_idx,
        rope_offsets, left_padding);
    // Update rope_offsets after prefill
    rope_offsets = array({cache_idx}, {1}, mlx::core::int32);

    // Extract last position logits [batch, seq, vocab] -> [vocab]
    int seq_len = static_cast<int>(logits.shape()[1]);
    logits = slice(logits, {0, seq_len - 1, 0}, {1, seq_len, logits.shape()[2]});
    logits = squeeze(logits, {0, 1});

    // Compute logprobs
    auto lp = logits - logsumexp(logits, -1, true);

    // Sample first token with full filtering
    auto tok = sample_with_filters(lp, temperature, top_k, top_p, min_p);
    return {tok, lp};
  }();
  auto y = prefill_result.first;
  auto logprobs_arr = prefill_result.second;
  async_eval({y, logprobs_arr});

  // === Generation loop with async pipelining ===
  std::optional<array> next_y, next_logprobs;

  for (int n = 0; n < max_new_tokens; n++) {
    // Schedule NEXT token computation (while we process current)
    if (n + 1 < max_new_tokens) {
      StreamContext ctx(generation_stream);

      // Reshape current token for next forward pass
      auto next_input = reshape(y, {1, 1});

      auto logits = forward_all_layers(
          next_input, embedding_weight, layer_weights, num_layers,
          final_norm_w, lm_head_w, tie_word_embeddings,
          hidden_size, num_heads, num_kv_heads, head_dim,
          attn_scale, rope_theta, norm_eps,
          kv_keys, kv_values, cache_idx,
          rope_offsets, left_padding);
      // Increment rope offset for next iteration
      rope_offsets = rope_offsets + array(1, mlx::core::int32);

      // Extract logits (already [1, 1, vocab] -> squeeze to [vocab])
      logits = squeeze(logits, {0, 1});

      // Apply repetition penalty if enabled
      if (repetition_penalty != 1.0f && !recent_tokens.empty()) {
        size_t ctx_start = recent_tokens.size() > static_cast<size_t>(repetition_context_size)
            ? recent_tokens.size() - repetition_context_size : 0;
        for (size_t i = ctx_start; i < recent_tokens.size(); i++) {
          int32_t tok = recent_tokens[i];
          auto tok_logit = slice(logits, {tok}, {tok + 1});
          auto updated = where(tok_logit < array(0.0f), tok_logit * array(repetition_penalty), tok_logit / array(repetition_penalty));
          logits = scatter(logits, array({tok}), squeeze(updated), 0);
        }
      }

      next_logprobs = logits - logsumexp(logits, -1, true);
      next_y = sample_with_filters(*next_logprobs, temperature, top_k, top_p, min_p);
    }
    if (next_y.has_value() && next_logprobs.has_value()) {
      async_eval({*next_y, *next_logprobs});
    }

    // Sync first token only
    if (n == 0) {
      eval(y);
    }

    // Extract CURRENT token (overlaps with NEXT computation on GPU)
    int32_t token = y.item<int32_t>();
    // Get logprob at the token index
    auto lp_arr = slice(logprobs_arr, {token}, {token + 1});
    eval(lp_arr);
    float lp = lp_arr.item<float>();

    out_tokens[n] = token;
    out_logprobs[n] = lp;
    recent_tokens.push_back(token);

    // Check EOS
    if (token == eos_token_id) {
      *out_num_tokens = n + 1;
      *out_finish_reason = 1;  // eos
      // Restore wired limit before returning
      if (mlx::core::metal::is_available()) {
        mlx::core::synchronize(generation_stream);
        mlx::core::set_wired_limit(old_wired_limit);
      }
      return;
    }

    // Clear cache periodically (matches mlx-lm line 456)
    if (n % 256 == 0 && n > 0) {
      clear_cache();
    }

    // Advance to next token
    if (next_y.has_value()) {
      y = *next_y;
      logprobs_arr = *next_logprobs;
    }
  }

  *out_num_tokens = max_new_tokens;
  *out_finish_reason = 0;  // length

  // Restore wired limit before returning
  if (mlx::core::metal::is_available()) {
    mlx::core::synchronize(generation_stream);
    mlx::core::set_wired_limit(old_wired_limit);
  }
}

// ============================================================================
// FUSED FORWARD STEP - Single FFI call per token
// ============================================================================
// This function performs ONE forward pass (embedding -> all layers -> logits)
// in a single C++ call, eliminating FFI overhead from the hot path.
//
// For a model with 28 layers, this reduces FFI calls from ~300 to 1 per token!

void mlx_qwen3_forward_step(
    // Input
    mlx_array* input_ids_handle,        // [batch, seq_len]

    // Model weights
    mlx_array* embedding_weight_handle, // [vocab, hidden]
    mlx_array* const* layer_weights,    // [num_layers * 11]
    int num_layers,
    mlx_array* final_norm_weight_handle,
    mlx_array* lm_head_weight_handle,   // null if tied
    bool tie_word_embeddings,

    // Model config
    int hidden_size, int num_heads, int num_kv_heads, int head_dim,
    float rope_theta, float norm_eps,

    // KV cache inputs (null for prefill without cache)
    mlx_array* const* kv_keys_in,       // [num_layers] or null
    mlx_array* const* kv_values_in,     // [num_layers] or null
    int cache_idx_in,                   // Shared cache write position

    // Array offsets for batched generation
    mlx_array* rope_offsets_handle,     // [batch] - per-sequence RoPE offsets
    mlx_array* left_padding_handle,     // [batch] - left padding amounts

    // Outputs (caller must free)
    mlx_array** out_logits,             // [batch, seq_len, vocab]
    mlx_array** out_kv_keys,            // [num_layers] new key arrays
    mlx_array** out_kv_values,          // [num_layers] new value arrays
    int* out_cache_idx                  // Updated write position
) {
    auto& input_ids = *reinterpret_cast<array*>(input_ids_handle);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_handle);
    auto& final_norm_w = *reinterpret_cast<array*>(final_norm_weight_handle);
    auto& rope_offsets = *reinterpret_cast<array*>(rope_offsets_handle);
    auto& left_padding = *reinterpret_cast<array*>(left_padding_handle);
    array* lm_head_w = lm_head_weight_handle ? reinterpret_cast<array*>(lm_head_weight_handle) : nullptr;

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int rope_dims = head_dim;

    int batch = static_cast<int>(input_ids.shape()[0]);
    int seq_len = static_cast<int>(input_ids.shape()[1]);

    // Initialize KV cache state with per-layer cache indices
    std::vector<std::optional<array>> kv_keys(num_layers);
    std::vector<std::optional<array>> kv_values(num_layers);
    std::vector<int> cache_indices(num_layers, cache_idx_in);

    if (kv_keys_in != nullptr && kv_values_in != nullptr) {
        for (int i = 0; i < num_layers; i++) {
            if (kv_keys_in[i] != nullptr) {
                kv_keys[i] = *reinterpret_cast<array*>(kv_keys_in[i]);
            }
            if (kv_values_in[i] != nullptr) {
                kv_values[i] = *reinterpret_cast<array*>(kv_values_in[i]);
            }
        }
    }

    // Embedding lookup
    auto hidden = take(embedding_weight, input_ids, 0);

    // Process each layer with array offsets
    for (int i = 0; i < num_layers; i++) {
        // Extract weights for this layer (11 weights per layer)
        int base = i * 11;
        auto& input_norm_w = *reinterpret_cast<array*>(layer_weights[base + 0]);
        auto& post_attn_norm_w = *reinterpret_cast<array*>(layer_weights[base + 1]);
        auto& w_q = *reinterpret_cast<array*>(layer_weights[base + 2]);
        auto& w_k = *reinterpret_cast<array*>(layer_weights[base + 3]);
        auto& w_v = *reinterpret_cast<array*>(layer_weights[base + 4]);
        auto& w_o = *reinterpret_cast<array*>(layer_weights[base + 5]);
        array* q_norm_w = layer_weights[base + 6] ? reinterpret_cast<array*>(layer_weights[base + 6]) : nullptr;
        array* k_norm_w = layer_weights[base + 7] ? reinterpret_cast<array*>(layer_weights[base + 7]) : nullptr;
        auto& w_gate = *reinterpret_cast<array*>(layer_weights[base + 8]);
        auto& w_up = *reinterpret_cast<array*>(layer_weights[base + 9]);
        auto& w_down = *reinterpret_cast<array*>(layer_weights[base + 10]);

        // Each layer has its own cache index (all start at same value, advance independently)
        hidden = transformer_block_forward_cached(
            hidden,
            input_norm_w, post_attn_norm_w,
            w_q, w_k, w_v, w_o, q_norm_w, k_norm_w,
            w_gate, w_up, w_down,
            num_heads, num_kv_heads, head_dim,
            attn_scale, rope_theta, rope_dims,
            norm_eps, norm_eps,
            kv_keys[i], kv_values[i], cache_indices[i],
            rope_offsets, left_padding);
    }

    // Final normalization
    hidden = fast::rms_norm(hidden, std::optional<array>(final_norm_w), norm_eps, {});

    // LM head and store output
    // NOTE: Do NOT call eval() or synchronize() here - MLX uses lazy evaluation
    // and the Rust code handles synchronization when needed (e.g., for sampling)
    if (tie_word_embeddings) {
        auto logits = matmul(hidden, transpose(embedding_weight, {1, 0}));
        *out_logits = reinterpret_cast<mlx_array*>(new array(std::move(logits)));
    } else {
        auto logits = matmul(hidden, transpose(*lm_head_w, {1, 0}));
        *out_logits = reinterpret_cast<mlx_array*>(new array(std::move(logits)));
    }

    // Output KV caches
    for (int i = 0; i < num_layers; i++) {
        if (kv_keys[i].has_value()) {
            out_kv_keys[i] = reinterpret_cast<mlx_array*>(new array(std::move(*kv_keys[i])));
        } else {
            out_kv_keys[i] = nullptr;
        }
        if (kv_values[i].has_value()) {
            out_kv_values[i] = reinterpret_cast<mlx_array*>(new array(std::move(*kv_values[i])));
        } else {
            out_kv_values[i] = nullptr;
        }
    }

    // Output updated cache index (all layers should have same value)
    *out_cache_idx = cache_indices[0];
}

/// Batched forward step for true parallel batch generation.
///
/// Unlike mlx_qwen3_forward_step which uses scalar cache_offset,
/// this function accepts an array of per-sequence offsets enabling
/// true batch generation with left-padded variable-length sequences.
void mlx_qwen3_forward_step_batched(
    // Input
    mlx_array* input_ids_handle,        // [batch, seq_len]

    // Model weights
    mlx_array* embedding_weight_handle, // [vocab, hidden]
    mlx_array* const* layer_weights,    // [num_layers * 11]
    int num_layers,
    mlx_array* final_norm_weight_handle,
    mlx_array* lm_head_weight_handle,   // null if tied
    bool tie_word_embeddings,

    // Model config
    int hidden_size, int num_heads, int num_kv_heads, int head_dim,
    float rope_theta, float norm_eps,

    // Batched RoPE offsets (key difference from scalar version)
    mlx_array* rope_offsets_handle,     // [batch] - per-sequence offsets

    // Left padding info for attention mask
    mlx_array* left_padding_handle,     // [batch] - left padding amounts

    // KV cache inputs (shared across batch, indexed by cache_idx)
    mlx_array* const* kv_keys_in,       // [num_layers] or null
    mlx_array* const* kv_values_in,     // [num_layers] or null
    int cache_idx_in,                   // Current write position (shared)

    // Outputs (caller must free)
    mlx_array** out_logits,             // [batch, seq_len, vocab]
    mlx_array** out_kv_keys,            // [num_layers] new key arrays
    mlx_array** out_kv_values,          // [num_layers] new value arrays
    int* out_cache_idx                  // Updated write position
) {
    auto& input_ids = *reinterpret_cast<array*>(input_ids_handle);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_handle);
    auto& final_norm_w = *reinterpret_cast<array*>(final_norm_weight_handle);
    auto& rope_offsets = *reinterpret_cast<array*>(rope_offsets_handle);
    auto& left_padding = *reinterpret_cast<array*>(left_padding_handle);
    array* lm_head_w = lm_head_weight_handle ? reinterpret_cast<array*>(lm_head_weight_handle) : nullptr;

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int rope_dims = head_dim;

    int batch = static_cast<int>(input_ids.shape()[0]);
    int seq_len = static_cast<int>(input_ids.shape()[1]);

    // Initialize KV cache state with per-layer cache indices
    std::vector<std::optional<array>> kv_keys(num_layers);
    std::vector<std::optional<array>> kv_values(num_layers);
    std::vector<int> cache_indices(num_layers, cache_idx_in);

    if (kv_keys_in != nullptr && kv_values_in != nullptr) {
        for (int i = 0; i < num_layers; i++) {
            if (kv_keys_in[i] != nullptr) {
                kv_keys[i] = *reinterpret_cast<array*>(kv_keys_in[i]);
            }
            if (kv_values_in[i] != nullptr) {
                kv_values[i] = *reinterpret_cast<array*>(kv_values_in[i]);
            }
        }
    }

    // Embedding lookup
    auto hidden = take(embedding_weight, input_ids, 0);

    // Process each layer with batched offsets
    for (int i = 0; i < num_layers; i++) {
        // Extract weights for this layer (11 weights per layer)
        int base = i * 11;
        auto& input_norm_w = *reinterpret_cast<array*>(layer_weights[base + 0]);
        auto& post_attn_norm_w = *reinterpret_cast<array*>(layer_weights[base + 1]);
        auto& w_q = *reinterpret_cast<array*>(layer_weights[base + 2]);
        auto& w_k = *reinterpret_cast<array*>(layer_weights[base + 3]);
        auto& w_v = *reinterpret_cast<array*>(layer_weights[base + 4]);
        auto& w_o = *reinterpret_cast<array*>(layer_weights[base + 5]);
        array* q_norm_w = layer_weights[base + 6] ? reinterpret_cast<array*>(layer_weights[base + 6]) : nullptr;
        array* k_norm_w = layer_weights[base + 7] ? reinterpret_cast<array*>(layer_weights[base + 7]) : nullptr;
        auto& w_gate = *reinterpret_cast<array*>(layer_weights[base + 8]);
        auto& w_up = *reinterpret_cast<array*>(layer_weights[base + 9]);
        auto& w_down = *reinterpret_cast<array*>(layer_weights[base + 10]);

        // Each layer has its own cache index (all start at same value, advance independently)
        hidden = transformer_block_forward_batched(
            hidden,
            input_norm_w, post_attn_norm_w,
            w_q, w_k, w_v, w_o, q_norm_w, k_norm_w,
            w_gate, w_up, w_down,
            num_heads, num_kv_heads, head_dim,
            attn_scale, rope_theta, rope_dims,
            norm_eps, norm_eps,
            kv_keys[i], kv_values[i], cache_indices[i],
            rope_offsets, left_padding);
    }

    // Final normalization
    hidden = fast::rms_norm(hidden, std::optional<array>(final_norm_w), norm_eps, {});

    // LM head and store output
    // NOTE: Do NOT call eval() or synchronize() here - MLX uses lazy evaluation
    // and the Rust code handles synchronization when needed (e.g., for sampling)
    if (tie_word_embeddings) {
        auto logits = matmul(hidden, transpose(embedding_weight, {1, 0}));
        *out_logits = reinterpret_cast<mlx_array*>(new array(std::move(logits)));
    } else {
        auto logits = matmul(hidden, transpose(*lm_head_w, {1, 0}));
        *out_logits = reinterpret_cast<mlx_array*>(new array(std::move(logits)));
    }

    // Output KV caches
    for (int i = 0; i < num_layers; i++) {
        if (kv_keys[i].has_value()) {
            out_kv_keys[i] = reinterpret_cast<mlx_array*>(new array(std::move(*kv_keys[i])));
        } else {
            out_kv_keys[i] = nullptr;
        }
        if (kv_values[i].has_value()) {
            out_kv_values[i] = reinterpret_cast<mlx_array*>(new array(std::move(*kv_values[i])));
        } else {
            out_kv_values[i] = nullptr;
        }
    }

    // All layers should have the same final cache index
    *out_cache_idx = cache_indices[0];
}

// ========================================================================
//
// These functions extract Metal buffer pointers from MLX arrays for use
// with external Metal kernel dispatch (e.g., from Rust metal crate).
//
// The extracted pointers are only valid after eval() and before any
// MLX operations that could reallocate the buffer.
//
// IMPORTANT: Only valid when Metal backend is available. On CPU-only
// builds or when GPU is unavailable, buffer pointers are NOT MTLBuffer*.
//
// Note: mlx_metal_is_available() is already defined earlier in this file.

/// Get the raw Metal buffer pointer from an MLX array
/// Returns the MTLBuffer* as a void* for FFI compatibility
/// Returns nullptr if:
///   - handle is null
///   - Metal backend is not available (buffer would not be MTLBuffer*)
///   - array has no data
void* mlx_array_get_metal_buffer(mlx_array* handle) {
    if (!handle) return nullptr;

    // Use Metal-specific availability check (not generic GPU)
    // This ensures we only return pointers when using Metal backend,
    // not when CUDA or other GPU backends might be in use
    if (!mlx::core::metal::is_available()) return nullptr;

    auto& arr = *reinterpret_cast<array*>(handle);

    // Ensure array is evaluated
    eval(arr);

    // Check if array has data
    if (arr.data_size() == 0) return nullptr;

    // When Metal backend is available, all MLX buffers use MTLBuffer
    // (Metal uses unified memory architecture on Apple Silicon)
    return const_cast<void*>(arr.buffer().ptr());
}

/// Get the byte offset into the Metal buffer for this array
/// This is needed for sliced/strided arrays that share a buffer
/// Note: offset() already returns bytes (used with char* in MLX internals)
size_t mlx_array_get_buffer_offset(mlx_array* handle) {
    if (!handle) return 0;
    auto& arr = *reinterpret_cast<array*>(handle);

    // eval not strictly needed for offset, but ensure consistency
    eval(arr);

    // offset() returns byte offset (see array.h line 374: char* + offset)
    return arr.data_size() > 0 ? static_cast<size_t>(arr.offset()) : 0;
}

/// Get the data size of the array in number of elements (NOT bytes)
/// To get bytes, multiply by itemsize
size_t mlx_array_get_data_size(mlx_array* handle) {
    if (!handle) return 0;
    auto& arr = *reinterpret_cast<array*>(handle);
    return arr.data_size();
}

/// Get the item size in bytes for the array's dtype
size_t mlx_array_get_itemsize(mlx_array* handle) {
    if (!handle) return 0;
    auto& arr = *reinterpret_cast<array*>(handle);
    return arr.itemsize();
}

/// Synchronize - ensure all MLX operations are complete
/// Call this before dispatching external Metal kernels
void mlx_metal_synchronize() {
    mlx::core::synchronize();
}

// ================================================================================
// Quantization Operations (for QuantizedKVCache)
// ================================================================================

/// Quantize a matrix along its last axis using affine quantization.
/// Returns a QuantizeResult struct with pointers to quantized weights, scales, and biases.
/// The caller is responsible for freeing all returned arrays.
///
/// @param w          Input array to quantize
/// @param group_size Number of elements per quantization group (default: 64)
/// @param bits       Number of bits for quantization (4 or 8, default: 4)
/// @param out_quantized  Output: quantized weights (packed uint8/uint32)
/// @param out_scales     Output: per-group scales
/// @param out_biases     Output: per-group biases (zero points)
/// @return true on success, false on error
bool mlx_quantize(
    mlx_array* w,
    int32_t group_size,
    int32_t bits,
    const char* mode,
    mlx_array** out_quantized,
    mlx_array** out_scales,
    mlx_array** out_biases
) {
    if (!w || !out_quantized || !out_scales || !out_biases) {
        return false;
    }

    try {
        auto& w_arr = *reinterpret_cast<array*>(w);

        std::optional<int> gs = group_size > 0 ? std::optional<int>(group_size) : std::nullopt;
        std::optional<int> b = bits > 0 ? std::optional<int>(bits) : std::nullopt;
        std::string mode_str = (mode && mode[0]) ? std::string(mode) : "affine";

        auto result = mlx::core::quantize(w_arr, gs, b, mode_str);

        // FP modes (mxfp4, mxfp8) return 2 arrays [quantized, scales], affine returns 3 [quantized, scales, biases]
        if (result.size() == 2) {
            *out_quantized = reinterpret_cast<mlx_array*>(new array(std::move(result[0])));
            *out_scales = reinterpret_cast<mlx_array*>(new array(std::move(result[1])));
            *out_biases = nullptr;
        } else if (result.size() == 3) {
            *out_quantized = reinterpret_cast<mlx_array*>(new array(std::move(result[0])));
            *out_scales = reinterpret_cast<mlx_array*>(new array(std::move(result[1])));
            *out_biases = reinterpret_cast<mlx_array*>(new array(std::move(result[2])));
        } else {
            std::cerr << "[MLX] quantize returned unexpected number of arrays: " << result.size() << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[MLX] Exception in mlx_quantize: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "[MLX] Unknown exception in mlx_quantize" << std::endl;
        return false;
    }
}

/// Dequantize a matrix that was quantized with mlx_quantize.
/// Reconstructs the original values using: value = quantized * scale + bias
///
/// @param quantized  Quantized weights from mlx_quantize
/// @param scales     Per-group scales from mlx_quantize
/// @param biases     Per-group biases from mlx_quantize (nullable for symmetric quant)
/// @param group_size Number of elements per quantization group (must match quantize)
/// @param bits       Number of bits (must match quantize)
/// @param out_dtype  Output dtype (0=float32, 5=bfloat16, 6=float16), -1 for input dtype
/// @return Dequantized array, or nullptr on error
mlx_array* mlx_dequantize(
    mlx_array* quantized,
    mlx_array* scales,
    mlx_array* biases,
    int32_t group_size,
    int32_t bits,
    int32_t out_dtype,
    const char* mode
) {
    if (!quantized || !scales) {
        return nullptr;
    }

    try {
        auto& q_arr = *reinterpret_cast<array*>(quantized);
        auto& s_arr = *reinterpret_cast<array*>(scales);

        std::optional<array> b_opt = std::nullopt;
        if (biases) {
            b_opt = *reinterpret_cast<array*>(biases);
        }

        std::optional<int> gs = group_size > 0 ? std::optional<int>(group_size) : std::nullopt;
        std::optional<int> b = bits > 0 ? std::optional<int>(bits) : std::nullopt;
        std::optional<mlx::core::Dtype> dtype = std::nullopt;
        std::string mode_str = (mode && mode[0]) ? std::string(mode) : "affine";

        if (out_dtype >= 0) {
            dtype = to_mlx_dtype(out_dtype);
        }

        auto result = mlx::core::dequantize(q_arr, s_arr, b_opt, gs, b, mode_str, std::nullopt, dtype);

        return reinterpret_cast<mlx_array*>(new array(std::move(result)));
    } catch (const std::exception& e) {
        std::cerr << "[MLX] Exception in mlx_dequantize: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "[MLX] Unknown exception in mlx_dequantize" << std::endl;
        return nullptr;
    }
}

// Main PaddleOCR-VL forward step: embedding -> mRoPE cos/sin -> 18 layers -> norm -> LM head
void mlx_paddleocr_vl_forward_step(
    mlx_array* input_embeds_handle,
    mlx_array* const* layer_weights,
    int num_layers,
    mlx_array* final_norm_weight_handle,
    mlx_array* lm_head_weight_handle,
    mlx_array* inv_freq_handle,
    mlx_array* position_ids_handle,
    const int* mrope_section,
    int hidden_size, int num_heads, int num_kv_heads, int head_dim,
    float norm_eps,
    mlx_array* const* kv_keys_in,
    mlx_array* const* kv_values_in,
    int cache_idx_in,
    mlx_array** out_logits,
    mlx_array** out_kv_keys,
    mlx_array** out_kv_values,
    int* out_cache_idx
) {
    auto& input_embeds = *reinterpret_cast<array*>(input_embeds_handle);
    auto& final_norm_w = *reinterpret_cast<array*>(final_norm_weight_handle);
    auto& lm_head_w = *reinterpret_cast<array*>(lm_head_weight_handle);
    auto& inv_freq = *reinterpret_cast<array*>(inv_freq_handle);
    auto& position_ids = *reinterpret_cast<array*>(position_ids_handle);

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Compute mRoPE cos/sin once, then pre-section for all layers (saves ~306 ops)
    auto [mrope_cos, mrope_sin] = compute_mrope_cos_sin(inv_freq, position_ids, head_dim);
    auto [cos_final, sin_final] = compute_mrope_sectioned(mrope_cos, mrope_sin, mrope_section);

    // Cast cos/sin to match input dtype (e.g. float16/bfloat16) to avoid promoting
    // all downstream computations (Q*cos, K*cos, KV cache, SDPA) to float32.
    // Python does: cos.astype(x.dtype), sin.astype(x.dtype)
    cos_final = astype(cos_final, input_embeds.dtype());
    sin_final = astype(sin_final, input_embeds.dtype());

    // Pre-transpose all layer weights once (saves 7 transposes x (num_layers-1) = ~119 ops)
    struct LayerWeightsT {
        array norm_in, norm_post;
        array q_t, k_t, v_t, o_t, gate_t, up_t, down_t;
    };
    std::vector<LayerWeightsT> layer_w;
    layer_w.reserve(num_layers);
    for (int i = 0; i < num_layers; i++) {
        int base = i * 9;
        auto& w_q = *reinterpret_cast<array*>(layer_weights[base + 2]);
        auto& w_k = *reinterpret_cast<array*>(layer_weights[base + 3]);
        auto& w_v = *reinterpret_cast<array*>(layer_weights[base + 4]);
        auto& w_o = *reinterpret_cast<array*>(layer_weights[base + 5]);
        auto& w_gate = *reinterpret_cast<array*>(layer_weights[base + 6]);
        auto& w_up = *reinterpret_cast<array*>(layer_weights[base + 7]);
        auto& w_down = *reinterpret_cast<array*>(layer_weights[base + 8]);
        layer_w.push_back({
            *reinterpret_cast<array*>(layer_weights[base + 0]),
            *reinterpret_cast<array*>(layer_weights[base + 1]),
            transpose(w_q, {1, 0}), transpose(w_k, {1, 0}),
            transpose(w_v, {1, 0}), transpose(w_o, {1, 0}),
            transpose(w_gate, {1, 0}), transpose(w_up, {1, 0}),
            transpose(w_down, {1, 0})
        });
    }

    // Initialize per-layer KV cache state
    std::vector<std::optional<array>> kv_keys(num_layers);
    std::vector<std::optional<array>> kv_values(num_layers);
    std::vector<int> cache_indices(num_layers, cache_idx_in);

    if (kv_keys_in != nullptr && kv_values_in != nullptr) {
        for (int i = 0; i < num_layers; i++) {
            if (kv_keys_in[i] != nullptr) kv_keys[i] = *reinterpret_cast<array*>(kv_keys_in[i]);
            if (kv_values_in[i] != nullptr) kv_values[i] = *reinterpret_cast<array*>(kv_values_in[i]);
        }
    }

    // Forward through all layers
    auto hidden = input_embeds;
    for (int i = 0; i < num_layers; i++) {
        auto& lw = layer_w[i];
        hidden = paddleocr_vl_block_forward_cached(
            hidden, lw.norm_in, lw.norm_post,
            lw.q_t, lw.k_t, lw.v_t, lw.o_t, lw.gate_t, lw.up_t, lw.down_t,
            num_heads, num_kv_heads, head_dim,
            attn_scale, norm_eps,
            cos_final, sin_final,
            kv_keys[i], kv_values[i], cache_indices[i]);
    }

    // Final norm + LM head
    hidden = fast::rms_norm(hidden, std::optional<array>(final_norm_w), norm_eps, {});
    auto logits = matmul(hidden, transpose(lm_head_w, {1, 0}));

    *out_logits = reinterpret_cast<mlx_array*>(new array(std::move(logits)));

    // Output KV caches
    for (int i = 0; i < num_layers; i++) {
        out_kv_keys[i] = kv_keys[i].has_value() ? reinterpret_cast<mlx_array*>(new array(std::move(*kv_keys[i]))) : nullptr;
        out_kv_values[i] = kv_values[i].has_value() ? reinterpret_cast<mlx_array*>(new array(std::move(*kv_values[i]))) : nullptr;
    }
    *out_cache_idx = cache_indices[0];
}

// Batched PaddleOCR-VL forward step: input_embeds -> mRoPE -> layers -> norm -> LM head
// Like mlx_paddleocr_vl_forward_step but with left_padding for batched decode.
void mlx_paddleocr_vl_forward_step_batched(
    mlx_array* input_embeds_handle,
    mlx_array* const* layer_weights,
    int num_layers,
    mlx_array* final_norm_weight_handle,
    mlx_array* lm_head_weight_handle,
    mlx_array* inv_freq_handle,
    mlx_array* position_ids_handle,
    const int* mrope_section,
    int hidden_size, int num_heads, int num_kv_heads, int head_dim,
    float norm_eps,
    mlx_array* left_padding_handle,
    mlx_array* const* kv_keys_in,
    mlx_array* const* kv_values_in,
    int cache_idx_in,
    mlx_array** out_logits,
    mlx_array** out_kv_keys,
    mlx_array** out_kv_values,
    int* out_cache_idx
) {
    auto& input_embeds = *reinterpret_cast<array*>(input_embeds_handle);
    auto& final_norm_w = *reinterpret_cast<array*>(final_norm_weight_handle);
    auto& lm_head_w = *reinterpret_cast<array*>(lm_head_weight_handle);
    auto& inv_freq = *reinterpret_cast<array*>(inv_freq_handle);
    auto& position_ids = *reinterpret_cast<array*>(position_ids_handle);
    auto& left_padding = *reinterpret_cast<array*>(left_padding_handle);

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Compute mRoPE cos/sin once, then pre-section for all layers
    auto [mrope_cos, mrope_sin] = compute_mrope_cos_sin(inv_freq, position_ids, head_dim);
    auto [cos_final, sin_final] = compute_mrope_sectioned(mrope_cos, mrope_sin, mrope_section);

    cos_final = astype(cos_final, input_embeds.dtype());
    sin_final = astype(sin_final, input_embeds.dtype());

    // Pre-transpose all layer weights once
    struct LayerWeightsT {
        array norm_in, norm_post;
        array q_t, k_t, v_t, o_t, gate_t, up_t, down_t;
    };
    std::vector<LayerWeightsT> layer_w;
    layer_w.reserve(num_layers);
    for (int i = 0; i < num_layers; i++) {
        int base = i * 9;
        auto& w_q = *reinterpret_cast<array*>(layer_weights[base + 2]);
        auto& w_k = *reinterpret_cast<array*>(layer_weights[base + 3]);
        auto& w_v = *reinterpret_cast<array*>(layer_weights[base + 4]);
        auto& w_o = *reinterpret_cast<array*>(layer_weights[base + 5]);
        auto& w_gate = *reinterpret_cast<array*>(layer_weights[base + 6]);
        auto& w_up = *reinterpret_cast<array*>(layer_weights[base + 7]);
        auto& w_down = *reinterpret_cast<array*>(layer_weights[base + 8]);
        layer_w.push_back({
            *reinterpret_cast<array*>(layer_weights[base + 0]),
            *reinterpret_cast<array*>(layer_weights[base + 1]),
            transpose(w_q, {1, 0}), transpose(w_k, {1, 0}),
            transpose(w_v, {1, 0}), transpose(w_o, {1, 0}),
            transpose(w_gate, {1, 0}), transpose(w_up, {1, 0}),
            transpose(w_down, {1, 0})
        });
    }

    // Initialize per-layer KV cache state
    std::vector<std::optional<array>> kv_keys(num_layers);
    std::vector<std::optional<array>> kv_values(num_layers);
    std::vector<int> cache_indices(num_layers, cache_idx_in);

    if (kv_keys_in != nullptr && kv_values_in != nullptr) {
        for (int i = 0; i < num_layers; i++) {
            if (kv_keys_in[i] != nullptr) kv_keys[i] = *reinterpret_cast<array*>(kv_keys_in[i]);
            if (kv_values_in[i] != nullptr) kv_values[i] = *reinterpret_cast<array*>(kv_values_in[i]);
        }
    }

    // Forward through all layers with batched mask
    auto hidden = input_embeds;
    for (int i = 0; i < num_layers; i++) {
        auto& lw = layer_w[i];
        hidden = paddleocr_vl_block_forward_batched(
            hidden, lw.norm_in, lw.norm_post,
            lw.q_t, lw.k_t, lw.v_t, lw.o_t, lw.gate_t, lw.up_t, lw.down_t,
            num_heads, num_kv_heads, head_dim,
            attn_scale, norm_eps,
            cos_final, sin_final,
            kv_keys[i], kv_values[i], cache_indices[i],
            left_padding);
    }

    // Final norm + LM head
    hidden = fast::rms_norm(hidden, std::optional<array>(final_norm_w), norm_eps, {});
    auto logits = matmul(hidden, transpose(lm_head_w, {1, 0}));

    *out_logits = reinterpret_cast<mlx_array*>(new array(std::move(logits)));

    for (int i = 0; i < num_layers; i++) {
        out_kv_keys[i] = kv_keys[i].has_value() ? reinterpret_cast<mlx_array*>(new array(std::move(*kv_keys[i]))) : nullptr;
        out_kv_values[i] = kv_values[i].has_value() ? reinterpret_cast<mlx_array*>(new array(std::move(*kv_values[i]))) : nullptr;
    }
    *out_cache_idx = cache_indices[0];
}

mlx_array* mlx_conv2d(
    mlx_array* input,
    mlx_array* weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    auto inp = reinterpret_cast<mlx::core::array*>(input);
    auto wt = reinterpret_cast<mlx::core::array*>(weight);
    mlx::core::array result = mlx::core::conv2d(
        *inp, *wt,
        {stride_h, stride_w},
        {padding_h, padding_w},
        {dilation_h, dilation_w},
        groups
    );
    return reinterpret_cast<mlx_array*>(new mlx::core::array(std::move(result)));
}

mlx_array* mlx_conv_transpose2d(
    mlx_array* input,
    mlx_array* weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    auto inp = reinterpret_cast<mlx::core::array*>(input);
    auto wt = reinterpret_cast<mlx::core::array*>(weight);

    // Output padding is hardcoded to (0, 0) since current usage (DBHead) requires
    // exact upsampling with kernel=2, stride=2, padding=0:
    //   output_h = (input_h - 1) * stride_h - 2*padding_h + kernel_h + output_padding_h
    //            = (input_h - 1) * 2 - 0 + 2 + 0 = 2 * input_h (exact 2x)
    // For future use cases requiring non-zero output_padding, expose as parameter.
    mlx::core::array result = mlx::core::conv_transpose2d(
        *inp, *wt,
        std::pair<int,int>{stride_h, stride_w},
        std::pair<int,int>{padding_h, padding_w},
        std::pair<int,int>{dilation_h, dilation_w},
        std::pair<int,int>{0, 0},
        groups
    );
    return reinterpret_cast<mlx_array*>(new mlx::core::array(std::move(result)));
}

// ============================================
// Conv1d
// ============================================

mlx_array* mlx_conv1d(
    mlx_array* input,
    mlx_array* weight,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    try {
        auto inp = reinterpret_cast<mlx::core::array*>(input);
        auto wt = reinterpret_cast<mlx::core::array*>(weight);
        mlx::core::array result = mlx::core::conv1d(
            *inp, *wt, stride, padding, dilation, groups
        );
        return reinterpret_cast<mlx_array*>(new mlx::core::array(std::move(result)));
    } catch (const std::exception& e) {
        std::cerr << "mlx_conv1d error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================
// Gather MM (for MoE / SwitchLinear)
// ============================================

mlx_array* mlx_gather_mm(
    mlx_array* a,
    mlx_array* b,
    mlx_array* lhs_indices,
    mlx_array* rhs_indices,
    bool sorted_indices
) {
    try {
        auto a_arr = reinterpret_cast<mlx::core::array*>(a);
        auto b_arr = reinterpret_cast<mlx::core::array*>(b);

        // Build optional index arrays - nullptr means no indexing on that side
        std::optional<mlx::core::array> lhs_opt = std::nullopt;
        std::optional<mlx::core::array> rhs_opt = std::nullopt;
        if (lhs_indices != nullptr) {
            lhs_opt = *reinterpret_cast<mlx::core::array*>(lhs_indices);
        }
        if (rhs_indices != nullptr) {
            rhs_opt = *reinterpret_cast<mlx::core::array*>(rhs_indices);
        }

        mlx::core::array result = mlx::core::gather_mm(
            *a_arr, *b_arr, lhs_opt, rhs_opt, sorted_indices
        );
        return reinterpret_cast<mlx_array*>(new mlx::core::array(std::move(result)));
    } catch (const std::exception& e) {
        std::cerr << "mlx_gather_mm error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================
// Quantized Matmul (for QuantizedLinear)
// ============================================

mlx_array* mlx_quantized_matmul(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* biases,        // nullable
    bool transpose,
    int group_size,
    int bits,
    const char* mode          // "affine" or "none"
) {
    try {
        auto x_arr = reinterpret_cast<mlx::core::array*>(x);
        auto w_arr = reinterpret_cast<mlx::core::array*>(w);
        auto scales_arr = reinterpret_cast<mlx::core::array*>(scales);

        std::optional<mlx::core::array> biases_opt = std::nullopt;
        if (biases != nullptr) {
            biases_opt = *reinterpret_cast<mlx::core::array*>(biases);
        }

        std::string mode_str(mode ? mode : "affine");

        mlx::core::array result = mlx::core::quantized_matmul(
            *x_arr, *w_arr, *scales_arr, biases_opt,
            transpose,
            std::optional<int>(group_size),
            std::optional<int>(bits),
            mode_str
        );
        return reinterpret_cast<mlx_array*>(new mlx::core::array(std::move(result)));
    } catch (const std::exception& e) {
        std::cerr << "mlx_quantized_matmul error: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================
// Gather QMM (for QuantizedSwitchLinear / MoE)
// ============================================

mlx_array* mlx_gather_qmm(
    mlx_array* x,
    mlx_array* w,
    mlx_array* scales,
    mlx_array* biases,        // nullable
    mlx_array* lhs_indices,   // nullable
    mlx_array* rhs_indices,   // nullable
    bool transpose,
    int group_size,
    int bits,
    const char* mode,         // "affine" or "none"
    bool sorted_indices
) {
    try {
        auto x_arr = reinterpret_cast<mlx::core::array*>(x);
        auto w_arr = reinterpret_cast<mlx::core::array*>(w);
        auto scales_arr = reinterpret_cast<mlx::core::array*>(scales);

        std::optional<mlx::core::array> biases_opt = std::nullopt;
        if (biases != nullptr) {
            biases_opt = *reinterpret_cast<mlx::core::array*>(biases);
        }

        std::optional<mlx::core::array> lhs_opt = std::nullopt;
        std::optional<mlx::core::array> rhs_opt = std::nullopt;
        if (lhs_indices != nullptr) {
            lhs_opt = *reinterpret_cast<mlx::core::array*>(lhs_indices);
        }
        if (rhs_indices != nullptr) {
            rhs_opt = *reinterpret_cast<mlx::core::array*>(rhs_indices);
        }

        std::string mode_str(mode ? mode : "affine");

        mlx::core::array result = mlx::core::gather_qmm(
            *x_arr, *w_arr, *scales_arr, biases_opt,
            lhs_opt, rhs_opt,
            transpose,
            std::optional<int>(group_size),
            std::optional<int>(bits),
            mode_str,
            sorted_indices
        );
        return reinterpret_cast<mlx_array*>(new mlx::core::array(std::move(result)));
    } catch (const std::exception& e) {
        std::cerr << "mlx_gather_qmm error: " << e.what() << std::endl;
        return nullptr;
    }
}

}  // extern "C"
