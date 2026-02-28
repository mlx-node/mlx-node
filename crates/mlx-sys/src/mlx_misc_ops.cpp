#include "mlx_common.h"

extern "C" {

// Synchronize with the default stream to ensure all operations complete
void mlx_synchronize() {
  try {
    mlx::core::synchronize();
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in synchronize: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in synchronize" << std::endl;
  }
}

// Clear the memory cache to prevent memory buildup
void mlx_clear_cache() {
  try {
    mlx::core::clear_cache();
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in clear_cache: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in clear_cache" << std::endl;
  }
}

// Clear the compiler cache (traced computation graphs)
// This releases memory from compiled/traced functions
// Returns true on success, false on failure
bool mlx_compile_clear_cache() {
  try {
    mlx::core::detail::compile_clear_cache();
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in compile_clear_cache: " << e.what() << std::endl;
    return false;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in compile_clear_cache" << std::endl;
    return false;
  }
}

// Compiled categorical sampling function (like MLX-LM's categorical_sampling)
// This is compiled once and reused for all sampling calls
mlx_array* mlx_compiled_categorical_sample(mlx_array* logits_handle, float temperature) {
  // Define the sampling function to compile
  static auto compiled_sampler = mlx::core::compile([](const std::vector<array>& inputs) {
    // inputs[0] = logits
    // inputs[1] = temperature (as a scalar array)
    auto logits = inputs[0];
    auto temp_scalar = inputs[1];

    // Scale logits by 1/temperature: logits * (1 / temp)
    auto scaled_logits = mlx::core::multiply(logits, temp_scalar);

    // Sample from categorical distribution
    auto sampled = mlx::core::random::categorical(scaled_logits, -1);

    return std::vector<array>{sampled};
  });

  // Convert inputs
  auto logits = *reinterpret_cast<array*>(logits_handle);
  auto temp_array = mlx::core::array(1.0f / temperature); // Create 1/temp as array

  // Call compiled function
  auto result = compiled_sampler({logits, temp_array});

  // Return result
  return reinterpret_cast<mlx_array*>(new array(std::move(result[0])));
}

// Top-k sampling (simplified - no compilation for now)
mlx_array* mlx_compiled_top_k(mlx_array* logprobs_handle, int top_k) {
  auto logprobs = *reinterpret_cast<array*>(logprobs_handle);

  // Partition to find top-k
  auto neg_logprobs = mlx::core::negative(logprobs);
  auto partitioned_indices = mlx::core::argpartition(neg_logprobs, top_k - 1, -1);

  // Get indices to mask (everything after top-k)
  auto shape = partitioned_indices.shape();
  mlx::core::Shape starts(shape.size(), 0);
  mlx::core::Shape ends(shape.begin(), shape.end());
  starts[starts.size() - 1] = top_k;

  auto mask_idx = mlx::core::slice(partitioned_indices, starts, ends);

  // Create -inf array and scatter at mask positions
  auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
  auto mask_shape = mask_idx.shape();
  auto broadcasted_neg_inf = mlx::core::broadcast_to(neg_inf, mask_shape);

  std::vector<array> indices_vec = {mask_idx};
  std::vector<int> axes_vec = {-1};
  auto masked_logprobs = mlx::core::scatter(logprobs, indices_vec, broadcasted_neg_inf, axes_vec);

  return reinterpret_cast<mlx_array*>(new array(std::move(masked_logprobs)));
}

// Top-p sampling (simplified - no compilation for now)
mlx_array* mlx_compiled_top_p(mlx_array* logprobs_handle, float top_p) {
  auto logprobs = *reinterpret_cast<array*>(logprobs_handle);

  // Convert to probabilities
  auto probs = mlx::core::exp(logprobs);

  // Sort in ascending order
  auto sorted_indices = mlx::core::argsort(logprobs, -1);
  auto sorted_probs = mlx::core::take_along_axis(probs, sorted_indices, -1);

  // Compute cumulative sum
  auto cumulative_probs = mlx::core::cumsum(sorted_probs, -1);

  // Rearrange cumulative probs back to original order
  auto shape = sorted_indices.shape();
  auto arange_vals = mlx::core::arange(0, shape[shape.size() - 1], sorted_indices.dtype());
  auto zeros = mlx::core::zeros_like(sorted_indices);

  std::vector<array> inv_indices_vec = {sorted_indices};
  std::vector<int> inv_axes_vec = {-1};
  auto inverse_indices = mlx::core::scatter(zeros, inv_indices_vec, arange_vals, inv_axes_vec);

  cumulative_probs = mlx::core::take_along_axis(cumulative_probs, inverse_indices, -1);

  // Select tokens with cumulative probs below threshold
  // Subtract epsilon for numerical stability (consistent with Rust path)
  // This prevents floating-point precision from excluding tokens at exact boundaries
  constexpr float EPSILON = 1e-7f;  // Use 1e-7 for float32 (vs 1e-10 in Rust for f64)
  auto threshold = mlx::core::array((1.0f - top_p) - EPSILON);
  auto mask = mlx::core::greater(cumulative_probs, threshold);

  auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
  auto result = mlx::core::where(mask, neg_inf, logprobs);

  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// Min-p sampling (simplified - no compilation for now)
mlx_array* mlx_compiled_min_p(mlx_array* logprobs_handle, float min_p, int min_tokens_to_keep) {
  auto logprobs = *reinterpret_cast<array*>(logprobs_handle);

  // Sort in descending order
  auto sorted_indices = mlx::core::argsort(mlx::core::negative(logprobs), -1);
  auto sorted_logprobs = mlx::core::take_along_axis(logprobs, sorted_indices, -1);

  // Get top logprob
  auto shape = sorted_logprobs.shape();
  mlx::core::Shape starts(shape.size(), 0);
  mlx::core::Shape ends(shape.begin(), shape.end());
  ends[ends.size() - 1] = 1;
  auto top_logprobs = mlx::core::slice(sorted_logprobs, starts, ends);

  // Calculate min_p threshold
  auto log_min_p = mlx::core::array(std::log(min_p));
  auto scaled_min_p = mlx::core::add(top_logprobs, log_min_p);

  // Mask tokens below threshold
  auto tokens_to_remove = mlx::core::less(sorted_logprobs, scaled_min_p);

  // Keep at least min_tokens_to_keep tokens
  if (min_tokens_to_keep > 0) {
    auto keep_shape = tokens_to_remove.shape();
    mlx::core::Shape keep_starts(keep_shape.size(), 0);
    mlx::core::Shape keep_ends(keep_shape.begin(), keep_shape.end());
    keep_ends[keep_ends.size() - 1] = min_tokens_to_keep;

    auto false_vals = mlx::core::zeros(keep_ends, tokens_to_remove.dtype());

    std::vector<array> keep_indices_vec = {mlx::core::arange(0, min_tokens_to_keep, mlx::core::int32)};
    std::vector<int> keep_axes_vec = {-1};
    tokens_to_remove = mlx::core::scatter(tokens_to_remove, keep_indices_vec, false_vals, keep_axes_vec);
  }

  // Apply mask
  auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
  auto selected_logprobs = mlx::core::where(tokens_to_remove, neg_inf, sorted_logprobs);

  // Rearrange back to original order
  auto zeros = mlx::core::zeros_like(sorted_indices);
  auto arange_vals = mlx::core::arange(0, shape[shape.size() - 1], sorted_indices.dtype());

  std::vector<array> inv_indices_vec = {sorted_indices};
  std::vector<int> inv_axes_vec = {-1};
  auto inverse_indices = mlx::core::scatter(zeros, inv_indices_vec, arange_vals, inv_axes_vec);

  auto result = mlx::core::take_along_axis(selected_logprobs, inverse_indices, -1);

  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// Temperature application for compiled sampling
mlx_array* mlx_compiled_apply_temperature(mlx_array* logits_handle, float temperature) {
  auto logits = reinterpret_cast<array*>(logits_handle);
  auto result = *logits / temperature;
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// ============================================================================
// Compiled sampling filters — each filter compiled separately (matches Python mlx-lm)
// ============================================================================
// Each filter produces a single Compiled primitive instead of ~30 individual ops.
// This reduces the eval tape by ~2,000 nodes per decode step.

// Top_k filter — NOT compiled because argpartition kth= requires concrete int.
// Applied inline with minimal ops. Less impactful since top_k is rarely used
// in typical sampling (temperature + top_p is the standard).
static array apply_top_k_filter(const array& logprobs, int k) {
  int vocab_size = logprobs.shape().back();
  if (k >= vocab_size) return logprobs;
  auto neg_logprobs = mlx::core::negative(logprobs);
  auto partitioned_indices = mlx::core::argpartition(neg_logprobs, k - 1, -1);
  auto shape = partitioned_indices.shape();
  mlx::core::Shape starts(shape.size(), 0);
  mlx::core::Shape ends(shape.begin(), shape.end());
  starts[starts.size() - 1] = k;
  auto mask_idx = mlx::core::slice(partitioned_indices, starts, ends);
  auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
  return mlx::core::put_along_axis(logprobs, mask_idx, neg_inf, -1);
}

// Compiled top_p: one fused kernel (matches Python @mx.compile apply_top_p)
static auto& compiled_top_p_fn() {
  static auto fn = mlx::core::compile([](const std::vector<array>& inputs) {
    auto logprobs = inputs[0];
    auto top_p_arr = inputs[1];

    auto probs = mlx::core::exp(logprobs);
    auto sorted_indices = mlx::core::argsort(logprobs, -1);
    auto sorted_probs = mlx::core::take_along_axis(probs, sorted_indices, -1);
    auto cumulative_probs = mlx::core::cumsum(sorted_probs, -1);

    auto shape = sorted_indices.shape();
    int last_dim = shape[shape.size() - 1];
    auto arange_vals = mlx::core::arange(0, last_dim, sorted_indices.dtype());
    auto zeros = mlx::core::zeros_like(sorted_indices);
    auto inverse_indices = mlx::core::put_along_axis(zeros, sorted_indices, arange_vals, -1);
    cumulative_probs = mlx::core::take_along_axis(cumulative_probs, inverse_indices, -1);

    auto threshold = mlx::core::subtract(
        mlx::core::array(1.0f),
        mlx::core::subtract(top_p_arr, mlx::core::array(1e-7f)));
    auto mask = mlx::core::greater(cumulative_probs, threshold);
    auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
    return std::vector<array>{mlx::core::where(mask, logprobs, neg_inf)};
  });
  return fn;
}

// Compiled min_p: one fused kernel (matches Python @mx.compile apply_min_p)
static auto& compiled_min_p_fn() {
  static auto fn = mlx::core::compile([](const std::vector<array>& inputs) {
    auto logprobs = inputs[0];
    auto min_p_arr = inputs[1];

    auto sorted_indices = mlx::core::argsort(mlx::core::negative(logprobs), -1);
    auto sorted_logprobs = mlx::core::take_along_axis(logprobs, sorted_indices, -1);

    auto shape = sorted_logprobs.shape();
    mlx::core::Shape starts(shape.size(), 0);
    mlx::core::Shape ends(shape.begin(), shape.end());
    ends[ends.size() - 1] = 1;
    auto top_logprobs = mlx::core::slice(sorted_logprobs, starts, ends);

    auto log_min_p = mlx::core::log(min_p_arr);
    auto scaled_min_p = mlx::core::add(top_logprobs, log_min_p);
    auto tokens_to_remove = mlx::core::less(sorted_logprobs, scaled_min_p);

    // Keep at least 1 token
    mlx::core::Shape first_starts(shape.size(), 0);
    mlx::core::Shape first_ends(shape.begin(), shape.end());
    first_ends[first_ends.size() - 1] = 1;
    auto keep_first = mlx::core::zeros_like(mlx::core::slice(tokens_to_remove, first_starts, first_ends));
    auto keep_indices = mlx::core::arange(0, 1, mlx::core::int32);
    tokens_to_remove = mlx::core::put_along_axis(tokens_to_remove, keep_indices, keep_first, -1);

    auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
    auto selected_logprobs = mlx::core::where(tokens_to_remove, neg_inf, sorted_logprobs);

    int last_dim = shape[shape.size() - 1];
    auto zeros = mlx::core::zeros_like(sorted_indices);
    auto arange_vals = mlx::core::arange(0, last_dim, sorted_indices.dtype());
    auto inverse_indices = mlx::core::put_along_axis(zeros, sorted_indices, arange_vals, -1);
    return std::vector<array>{mlx::core::take_along_axis(selected_logprobs, inverse_indices, -1)};
  });
  return fn;
}

// Compiled categorical sampling with temperature
static auto& compiled_categorical_fn() {
  static auto fn = mlx::core::compile([](const std::vector<array>& inputs) {
    auto logprobs = inputs[0];
    auto inv_temp = inputs[1];
    auto scaled = mlx::core::multiply(logprobs, inv_temp);
    return std::vector<array>{mlx::core::random::categorical(scaled, -1)};
  });
  return fn;
}

// ============================================================================
// Main compiled sampling function — uses compiled filters
// ============================================================================
mlx_array* mlx_compiled_sample_full(
    mlx_array* logits_handle,
    float temperature,
    int top_k,
    float top_p,
    float min_p
) {
  auto logits = *reinterpret_cast<array*>(logits_handle);

  // Fast path: temperature == 0 means argmax (greedy)
  if (temperature == 0.0f) {
    auto result = mlx::core::argmax(logits, -1);
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  }

  bool needs_filters = (top_k > 0) || (top_p > 0.0f && top_p < 1.0f) || (min_p > 0.0f);

  // Fast path: no filters — compiled categorical only
  if (!needs_filters) {
    auto inv_temp = mlx::core::array(1.0f / temperature);
    auto results = compiled_categorical_fn()({logits, inv_temp});
    return reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
  }

  // Convert logits to logprobs: logprobs = logits - logsumexp(logits)
  auto logsumexp = mlx::core::logsumexp(logits, std::vector<int>{-1}, true);
  auto logprobs = mlx::core::subtract(logits, logsumexp);

  // Apply filters (each compiled into minimal graph nodes)
  if (top_k > 0) {
    logprobs = apply_top_k_filter(logprobs, top_k);
  }

  if (top_p > 0.0f && top_p < 1.0f) {
    logprobs = compiled_top_p_fn()({logprobs, mlx::core::array(top_p)})[0];
  }

  if (min_p > 0.0f) {
    logprobs = compiled_min_p_fn()({logprobs, mlx::core::array(min_p)})[0];
  }

  // Apply temperature and sample — compiled categorical
  auto inv_temp = mlx::core::array(1.0f / temperature);
  auto results = compiled_categorical_fn()({logprobs, inv_temp});
  return reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
}

// ============================================================================
// Optimized sampling that returns BOTH token and logprobs (eliminates redundant computation)
// ============================================================================
// Key insight from mlx-lm: compute logprobs ONCE and use for both:
// 1. Sampling (with filters applied)
// 2. Return value (original, unfiltered)
// This eliminates the redundant logsumexp computation in Rust.

void mlx_sample_and_logprobs(
    mlx_array* logits_handle,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    mlx_array** out_token,
    mlx_array** out_logprobs
) {
  auto logits = *reinterpret_cast<array*>(logits_handle);

  // Always compute logprobs (needed for return and potentially for filters)
  auto logsumexp_val = mlx::core::logsumexp(logits, std::vector<int>{-1}, true);
  auto logprobs = mlx::core::subtract(logits, logsumexp_val);

  // Fast path: temperature == 0 means argmax (greedy)
  if (temperature == 0.0f) {
    auto result = mlx::core::argmax(logits, -1);
    *out_token = reinterpret_cast<mlx_array*>(new array(std::move(result)));
    *out_logprobs = reinterpret_cast<mlx_array*>(new array(std::move(logprobs)));
    return;
  }

  // Fast path: no filters, just temperature sampling
  bool needs_filters = (top_k > 0) || (top_p > 0.0f && top_p < 1.0f) || (min_p > 0.0f);
  if (!needs_filters) {
    auto inv_temp = mlx::core::array(1.0f / temperature);
    auto scaled = mlx::core::multiply(logprobs, inv_temp);
    auto result = mlx::core::random::categorical(scaled, -1);
    *out_token = reinterpret_cast<mlx_array*>(new array(std::move(result)));
    *out_logprobs = reinterpret_cast<mlx_array*>(new array(std::move(logprobs)));
    return;
  }

  // Keep original logprobs for return (CoW - no extra memory until modified)
  auto original_logprobs = logprobs;

  // Apply top_k filter
  if (top_k > 0) {
    int vocab_size = logprobs.shape().back();
    if (top_k < vocab_size) {
      auto neg_logprobs = mlx::core::negative(logprobs);
      auto partitioned_indices = mlx::core::argpartition(neg_logprobs, top_k - 1, -1);
      auto shape = partitioned_indices.shape();
      mlx::core::Shape starts(shape.size(), 0);
      mlx::core::Shape ends(shape.begin(), shape.end());
      starts[starts.size() - 1] = top_k;
      auto mask_idx = mlx::core::slice(partitioned_indices, starts, ends);
      auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
      logprobs = mlx::core::put_along_axis(logprobs, mask_idx, neg_inf, -1);
    }
  }

  // Apply top_p filter
  if (top_p > 0.0f && top_p < 1.0f) {
    auto probs = mlx::core::exp(logprobs);
    auto sorted_indices = mlx::core::argsort(logprobs, -1);
    auto sorted_probs = mlx::core::take_along_axis(probs, sorted_indices, -1);
    auto cumulative_probs = mlx::core::cumsum(sorted_probs, -1);
    auto shape = sorted_indices.shape();
    int last_dim = shape[shape.size() - 1];
    auto arange_vals = mlx::core::arange(0, last_dim, sorted_indices.dtype());
    auto zeros = mlx::core::zeros_like(sorted_indices);
    auto inverse_indices = mlx::core::put_along_axis(zeros, sorted_indices, arange_vals, -1);
    cumulative_probs = mlx::core::take_along_axis(cumulative_probs, inverse_indices, -1);
    // Subtract epsilon for numerical stability (consistent with Rust path)
    constexpr float EPSILON = 1e-7f;
    auto threshold = mlx::core::array((1.0f - top_p) - EPSILON);
    auto mask = mlx::core::greater(cumulative_probs, threshold);
    auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
    logprobs = mlx::core::where(mask, logprobs, neg_inf);
  }

  // Apply min_p filter
  if (min_p > 0.0f) {
    auto sorted_indices = mlx::core::argsort(mlx::core::negative(logprobs), -1);
    auto sorted_logprobs = mlx::core::take_along_axis(logprobs, sorted_indices, -1);
    auto shape = sorted_logprobs.shape();
    mlx::core::Shape starts(shape.size(), 0);
    mlx::core::Shape ends(shape.begin(), shape.end());
    ends[ends.size() - 1] = 1;
    auto top_logprobs = mlx::core::slice(sorted_logprobs, starts, ends);
    auto log_min_p = mlx::core::array(std::log(min_p));
    auto scaled_min_p = mlx::core::add(top_logprobs, log_min_p);
    auto tokens_to_remove = mlx::core::less(sorted_logprobs, scaled_min_p);
    mlx::core::Shape first_starts(shape.size(), 0);
    mlx::core::Shape first_ends(shape.begin(), shape.end());
    first_ends[first_ends.size() - 1] = 1;
    auto keep_first = mlx::core::zeros_like(mlx::core::slice(tokens_to_remove, first_starts, first_ends));
    auto keep_indices = mlx::core::arange(0, 1, mlx::core::int32);
    tokens_to_remove = mlx::core::put_along_axis(tokens_to_remove, keep_indices, keep_first, -1);
    auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
    auto selected_logprobs = mlx::core::where(tokens_to_remove, neg_inf, sorted_logprobs);
    int last_dim = shape[shape.size() - 1];
    auto zeros = mlx::core::zeros_like(sorted_indices);
    auto arange_vals = mlx::core::arange(0, last_dim, sorted_indices.dtype());
    auto inverse_indices = mlx::core::put_along_axis(zeros, sorted_indices, arange_vals, -1);
    logprobs = mlx::core::take_along_axis(selected_logprobs, inverse_indices, -1);
  }

  // Sample from filtered logprobs
  auto inv_temp = mlx::core::array(1.0f / temperature);
  auto scaled = mlx::core::multiply(logprobs, inv_temp);
  auto result = mlx::core::random::categorical(scaled, -1);

  // Return token and ORIGINAL (unfiltered) logprobs
  *out_token = reinterpret_cast<mlx_array*>(new array(std::move(result)));
  *out_logprobs = reinterpret_cast<mlx_array*>(new array(std::move(original_logprobs)));
}

// ============================================================================
// Compiled Sampling with Logprobs (uses existing mlx_compiled_categorical_sample)
// ============================================================================
// Uses the already-compiled categorical sampler for the final sampling step.

void mlx_compiled_sample_and_logprobs(
    mlx_array* logits_handle,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    mlx_array** out_token,
    mlx_array** out_logprobs
) {
  auto logits = *reinterpret_cast<array*>(logits_handle);

  // Compute logprobs once
  auto logsumexp_val = mlx::core::logsumexp(logits, std::vector<int>{-1}, true);
  auto logprobs = mlx::core::subtract(logits, logsumexp_val);

  // Greedy fast path
  if (temperature == 0.0f) {
    auto result = mlx::core::argmax(logits, -1);
    *out_token = reinterpret_cast<mlx_array*>(new array(std::move(result)));
    *out_logprobs = reinterpret_cast<mlx_array*>(new array(std::move(logprobs)));
    return;
  }

  // No filters - use compiled categorical directly
  bool needs_filters = (top_k > 0) || (top_p > 0.0f && top_p < 1.0f) || (min_p > 0.0f);
  if (!needs_filters) {
    // Use the same compiled sampler pattern as mlx_compiled_categorical_sample
    static auto compiled_sampler = mlx::core::compile([](const std::vector<array>& inputs) {
      auto lp = inputs[0];
      auto temp_scalar = inputs[1];
      auto scaled = mlx::core::multiply(lp, temp_scalar);
      return std::vector<array>{mlx::core::random::categorical(scaled, -1)};
    });
    auto temp_array = mlx::core::array(1.0f / temperature);
    auto results = compiled_sampler({logprobs, temp_array});
    *out_token = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
    *out_logprobs = reinterpret_cast<mlx_array*>(new array(std::move(logprobs)));
    return;
  }

  // Keep original for return
  auto original_logprobs = logprobs;

  // Apply filters (same as mlx_sample_and_logprobs)
  if (top_k > 0) {
    int vocab_size = logprobs.shape().back();
    if (top_k < vocab_size) {
      auto neg_logprobs = mlx::core::negative(logprobs);
      auto partitioned_indices = mlx::core::argpartition(neg_logprobs, top_k - 1, -1);
      auto shape = partitioned_indices.shape();
      mlx::core::Shape starts(shape.size(), 0);
      mlx::core::Shape ends(shape.begin(), shape.end());
      starts[starts.size() - 1] = top_k;
      auto mask_idx = mlx::core::slice(partitioned_indices, starts, ends);
      auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
      logprobs = mlx::core::put_along_axis(logprobs, mask_idx, neg_inf, -1);
    }
  }

  if (top_p > 0.0f && top_p < 1.0f) {
    auto probs = mlx::core::exp(logprobs);
    auto sorted_indices = mlx::core::argsort(logprobs, -1);
    auto sorted_probs = mlx::core::take_along_axis(probs, sorted_indices, -1);
    auto cumulative_probs = mlx::core::cumsum(sorted_probs, -1);
    auto shape = sorted_indices.shape();
    int last_dim = shape[shape.size() - 1];
    auto arange_vals = mlx::core::arange(0, last_dim, sorted_indices.dtype());
    auto zeros = mlx::core::zeros_like(sorted_indices);
    auto inverse_indices = mlx::core::put_along_axis(zeros, sorted_indices, arange_vals, -1);
    cumulative_probs = mlx::core::take_along_axis(cumulative_probs, inverse_indices, -1);
    // Subtract epsilon for numerical stability (consistent with Rust path)
    constexpr float EPSILON = 1e-7f;
    auto threshold = mlx::core::array((1.0f - top_p) - EPSILON);
    auto mask = mlx::core::greater(cumulative_probs, threshold);
    auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
    logprobs = mlx::core::where(mask, logprobs, neg_inf);
  }

  if (min_p > 0.0f) {
    auto sorted_indices = mlx::core::argsort(mlx::core::negative(logprobs), -1);
    auto sorted_logprobs = mlx::core::take_along_axis(logprobs, sorted_indices, -1);
    auto shape = sorted_logprobs.shape();
    mlx::core::Shape starts(shape.size(), 0);
    mlx::core::Shape ends(shape.begin(), shape.end());
    ends[ends.size() - 1] = 1;
    auto top_logprobs = mlx::core::slice(sorted_logprobs, starts, ends);
    auto log_min_p = mlx::core::array(std::log(min_p));
    auto scaled_min_p = mlx::core::add(top_logprobs, log_min_p);
    auto tokens_to_remove = mlx::core::less(sorted_logprobs, scaled_min_p);
    mlx::core::Shape first_starts(shape.size(), 0);
    mlx::core::Shape first_ends(shape.begin(), shape.end());
    first_ends[first_ends.size() - 1] = 1;
    auto keep_first = mlx::core::zeros_like(mlx::core::slice(tokens_to_remove, first_starts, first_ends));
    auto keep_indices = mlx::core::arange(0, 1, mlx::core::int32);
    tokens_to_remove = mlx::core::put_along_axis(tokens_to_remove, keep_indices, keep_first, -1);
    auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
    auto selected_logprobs = mlx::core::where(tokens_to_remove, neg_inf, sorted_logprobs);
    int last_dim = shape[shape.size() - 1];
    auto zeros = mlx::core::zeros_like(sorted_indices);
    auto arange_vals = mlx::core::arange(0, last_dim, sorted_indices.dtype());
    auto inverse_indices = mlx::core::put_along_axis(zeros, sorted_indices, arange_vals, -1);
    logprobs = mlx::core::take_along_axis(selected_logprobs, inverse_indices, -1);
  }

  // Use compiled categorical sampler at the end (reuse the same static compiled function)
  static auto compiled_sampler_filtered = mlx::core::compile([](const std::vector<array>& inputs) {
    auto lp = inputs[0];
    auto temp_scalar = inputs[1];
    auto scaled = mlx::core::multiply(lp, temp_scalar);
    return std::vector<array>{mlx::core::random::categorical(scaled, -1)};
  });
  auto temp_array = mlx::core::array(1.0f / temperature);
  auto results = compiled_sampler_filtered({logprobs, temp_array});

  *out_token = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
  *out_logprobs = reinterpret_cast<mlx_array*>(new array(std::move(original_logprobs)));
}

}  // extern "C"
