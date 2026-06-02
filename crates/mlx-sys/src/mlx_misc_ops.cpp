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

// Temperatures in `[0, GREEDY_TEMPERATURE_EPS]` are treated as greedy (argmax)
// by every sampler below. This MUST match the Rust MTP accept gates in
// chat_common.rs / sampling.rs (`temperature <= 1e-6` == greedy): keeping the
// threshold identical makes the draft draw, target/bonus draw,
// sampling_distribution, the AR sampler, and the accept gates byte-consistent
// for tiny T (otherwise a tiny nonzero T would draw stochastically but accept
// greedily, or AR would diverge from MTP).
constexpr float GREEDY_TEMPERATURE_EPS = 1e-6f;

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

// MTPLX fast sparse top-p rule: after temperature scaling and top-k selection,
// keep a token while the cumulative probability mass before it is below top_p.
static auto& compiled_mtplx_top_p_fn() {
  static auto fn = mlx::core::compile([](const std::vector<array>& inputs) {
    auto logprobs = inputs[0];
    auto top_p_arr = inputs[1];

    auto probs = mlx::core::exp(logprobs);
    auto sorted_indices = mlx::core::argsort(mlx::core::negative(logprobs), -1);
    auto sorted_logprobs = mlx::core::take_along_axis(logprobs, sorted_indices, -1);
    auto sorted_probs = mlx::core::take_along_axis(probs, sorted_indices, -1);
    auto cumulative_before = mlx::core::subtract(
        mlx::core::cumsum(sorted_probs, -1),
        sorted_probs);
    auto keep_sorted = mlx::core::less(cumulative_before, top_p_arr);
    auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
    auto selected_logprobs = mlx::core::where(keep_sorted, sorted_logprobs, neg_inf);

    auto shape = sorted_indices.shape();
    int last_dim = shape[shape.size() - 1];
    auto arange_vals = mlx::core::arange(0, last_dim, sorted_indices.dtype());
    auto zeros = mlx::core::zeros_like(sorted_indices);
    auto inverse_indices = mlx::core::put_along_axis(zeros, sorted_indices, arange_vals, -1);
    return std::vector<array>{mlx::core::take_along_axis(selected_logprobs, inverse_indices, -1)};
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

// Categorical sampling with temperature — NOT compiled.
//
// The C++ compile() API does not support random state management
// (inputs/outputs=mx.random.state is a Python-only feature).
// Compiling random::categorical would capture the random key at trace time
// and reuse it on every call, corrupting MLX's global random state.
static array categorical_with_temp(const array& logprobs, const array& inv_temp) {
  auto scaled = mlx::core::multiply(logprobs, inv_temp);
  return mlx::core::random::categorical(scaled, -1);
}

// ============================================================================
// Shared filter chain — the SINGLE source of truth for the distribution the
// compiled sampler draws from.
// ============================================================================
//
// `mlx_compiled_sample_full` (the categorical draw) and
// `mlx_compiled_sampling_distribution` (the normalized proposal/target
// density consumed by stochastic MTP acceptance) MUST sample from the exact
// same distribution. Factoring the filter chain here makes them impossible to
// drift: both call this helper, then either draw via `categorical_with_temp`
// or normalize via `softmax`.
//
// The returned pair is `(filtered_logits, inv_temp)`; the distribution the
// sampler draws from is `softmax(filtered_logits * inv_temp)` along the last
// axis (this is exactly what `random::categorical` evaluates after the
// `multiply` in `categorical_with_temp`).
//
// `needs_filters` mirrors the caller's fast-path gate. When false the caller
// short-circuits to the no-filter path (raw logits + 1/temperature); this
// helper is only invoked on the filtered path so the two callers stay
// byte-identical to their respective fast paths.
struct CompiledSamplerLogits {
  array filtered_logits;
  array inv_temp;
};

static CompiledSamplerLogits compiled_sampler_logits(
    const array& logits,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    int sampler_mode
) {
  // Default mode filters on raw model probabilities, then applies temperature
  // only for the final categorical draw. MTPLX parity mode filters on the
  // temperature-scaled distribution, matching its sparse MTP proposal path.
  bool temperature_first = sampler_mode == 1;
  auto sampler_logits = logits;
  if (temperature_first) {
    sampler_logits = mlx::core::multiply(logits, mlx::core::array(1.0f / temperature));
  }

  // Convert sampler logits to logprobs: logprobs = logits - logsumexp(logits)
  auto logsumexp = mlx::core::logsumexp(sampler_logits, std::vector<int>{-1}, true);
  auto logprobs = mlx::core::subtract(sampler_logits, logsumexp);

  // Apply filters (each compiled into minimal graph nodes). MTPLX parity uses
  // temperature-scaled top-k support plus its fast sparse top-p keep rule. The
  // default keeps the existing mlx-node top-k then top-p behavior.
  if (top_k > 0) {
    logprobs = apply_top_k_filter(logprobs, top_k);
  }
  if (top_p > 0.0f && top_p < 1.0f) {
    auto top_p_arr = mlx::core::array(top_p);
    // MTPLX parity uses its fast sparse top-p keep rule; default keeps the
    // existing mlx-node top-p behavior.
    logprobs = temperature_first
                   ? compiled_mtplx_top_p_fn()({logprobs, top_p_arr})[0]
                   : compiled_top_p_fn()({logprobs, top_p_arr})[0];
  }

  if (min_p > 0.0f) {
    logprobs = compiled_min_p_fn()({logprobs, mlx::core::array(min_p)})[0];
  }

  // In MTPLX parity mode `logprobs` is already temperature-scaled, so the
  // final scale is 1.0.
  auto inv_temp = mlx::core::array(temperature_first ? 1.0f : (1.0f / temperature));
  return CompiledSamplerLogits{std::move(logprobs), std::move(inv_temp)};
}

// ============================================================================
// Main compiled sampling function — uses compiled filters
// ============================================================================
mlx_array* mlx_compiled_sample_full(
    mlx_array* logits_handle,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    int sampler_mode
) {
  auto logits = *reinterpret_cast<array*>(logits_handle);

  // Fast path: tiny temperature means argmax (greedy). Treat the whole
  // [0, GREEDY_TEMPERATURE_EPS] band as greedy so the draft draw matches the
  // Rust accept gate (temperature <= 1e-6).
  if (temperature <= GREEDY_TEMPERATURE_EPS) {
    auto result = mlx::core::argmax(logits, -1);
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  }

  bool needs_filters = (top_k > 0) || (top_p > 0.0f && top_p < 1.0f) || (min_p > 0.0f);

  // Fast path: no filters — categorical only
  if (!needs_filters) {
    auto inv_temp = mlx::core::array(1.0f / temperature);
    auto result = categorical_with_temp(logits, inv_temp);
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  }

  auto sampler =
      compiled_sampler_logits(logits, temperature, top_k, top_p, min_p, sampler_mode);
  // Apply temperature and sample — uncompiled categorical.
  auto result = categorical_with_temp(sampler.filtered_logits, sampler.inv_temp);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// ============================================================================
// Compiled sampling distribution — the normalized probability vector the
// compiled sampler draws from.
// ============================================================================
//
// Returns `softmax(filtered_logits * inv_temp)` over the last axis, i.e. the
// EXACT distribution `mlx_compiled_sample_full` samples its token from, using
// the SAME `sampler_mode` filter ordering. Stochastic MTP acceptance consumes
// this as the proposal density `q` (draft logits) and the target density `p`
// (verify logits) so the probability-ratio accept/reject and residual resample
// match the draw distribution by construction — preserving Leviathan-Chen
// exactness for arbitrary temperature/top_k/top_p/min_p and both parity modes.
//
// Filtered-out tokens (`-inf` logits) softmax to exactly 0, matching the
// categorical draw's support. The output dtype follows `softmax` on the input
// dtype; callers cast to f32 for the accept math.
//
// At temperature == 0 the sampler is argmax-only and accept/reject ignores the
// proposal density, so callers MUST NOT call this at T=0; this function still
// returns a defined one-hot argmax distribution (so a stray call is benign).
mlx_array* mlx_compiled_sampling_distribution(
    mlx_array* logits_handle,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    int sampler_mode
) {
  auto logits = *reinterpret_cast<array*>(logits_handle);

  // Tiny T (greedy): argmax-only sampler. Return a one-hot distribution so a
  // stray caller sees a defined, normalized vector (accept/reject ignores q/p
  // when greedy). The whole [0, GREEDY_TEMPERATURE_EPS] band is greedy to match
  // the Rust accept gate (temperature <= 1e-6).
  //
  // Build the one-hot in int32 index space (NOT the logits dtype): for bf16/f16
  // logits with a large vocab, integer token IDs (e.g. ~150k) are not exactly
  // representable, so a float `arange == idx` comparison could place the `1.0`
  // at the wrong position and shift `argmax(p_target)`. Comparing int32 indices
  // is exact; only the final 0/1 mask is cast to the logits dtype.
  if (temperature <= GREEDY_TEMPERATURE_EPS) {
    auto idx = mlx::core::argmax(logits, -1, true);
    int vocab = logits.shape().back();
    auto positions = mlx::core::arange(0, vocab, mlx::core::int32);
    auto one_hot = mlx::core::equal(positions, mlx::core::astype(idx, mlx::core::int32));
    auto result = mlx::core::astype(one_hot, logits.dtype());
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  }

  bool needs_filters = (top_k > 0) || (top_p > 0.0f && top_p < 1.0f) || (min_p > 0.0f);

  // Fast path: no filters — softmax(logits / temperature), matching the
  // no-filter `categorical_with_temp(logits, 1/temperature)` draw.
  if (!needs_filters) {
    auto scaled = mlx::core::multiply(logits, mlx::core::array(1.0f / temperature));
    auto result = mlx::core::softmax(scaled, std::vector<int>{-1}, true);
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  }

  auto sampler =
      compiled_sampler_logits(logits, temperature, top_k, top_p, min_p, sampler_mode);
  // The categorical draw is over `softmax(filtered_logits * inv_temp)`; return
  // that normalized distribution instead of drawing from it. `precise=true`
  // softmax: the draw distribution is analytically softmax, and precise f32
  // probabilities matter for the `min(1, p/q)` accept ratio downstream.
  auto scaled = mlx::core::multiply(sampler.filtered_logits, sampler.inv_temp);
  auto result = mlx::core::softmax(scaled, std::vector<int>{-1}, true);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// ============================================================================
// Compiled Sampling with Logprobs
// ============================================================================
// Uses an inline compiled categorical sampler for the final sampling step.

void mlx_compiled_sample_and_logprobs(
    mlx_array* logits_handle,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    int sampler_mode,
    mlx_array** out_token,
    mlx_array** out_logprobs
) {
  auto logits = *reinterpret_cast<array*>(logits_handle);

  // Compute logprobs once
  auto logsumexp_val = mlx::core::logsumexp(logits, std::vector<int>{-1}, true);
  auto logprobs = mlx::core::subtract(logits, logsumexp_val);

  // Greedy fast path. Treat the whole [0, GREEDY_TEMPERATURE_EPS] band as
  // greedy so the AR sampler stays byte-identical with the MTP samplers and the
  // Rust accept gate (temperature <= 1e-6) at tiny T.
  if (temperature <= GREEDY_TEMPERATURE_EPS) {
    auto result = mlx::core::argmax(logits, -1);
    *out_token = reinterpret_cast<mlx_array*>(new array(std::move(result)));
    *out_logprobs = reinterpret_cast<mlx_array*>(new array(std::move(logprobs)));
    return;
  }

  // No filters - use compiled categorical directly
  bool needs_filters = (top_k > 0) || (top_p > 0.0f && top_p < 1.0f) || (min_p > 0.0f);
  if (!needs_filters) {
    // Inline compiled sampler: scale logprobs by 1/temp then categorical
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

  bool temperature_first = sampler_mode == 1;
  auto sampler_logprobs = logprobs;
  if (temperature_first) {
    auto scaled_logits = mlx::core::multiply(logits, mlx::core::array(1.0f / temperature));
    auto scaled_logsumexp = mlx::core::logsumexp(scaled_logits, std::vector<int>{-1}, true);
    sampler_logprobs = mlx::core::subtract(scaled_logits, scaled_logsumexp);
  }

  // Apply filters. MTPLX parity uses temperature-scaled top-k support plus its
  // fast sparse top-p keep rule; default mode preserves the existing ordering.
  logprobs = sampler_logprobs;
  auto apply_top_p_inline = [&logprobs](float p) {
    if (p <= 0.0f || p >= 1.0f) return;
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
    auto threshold = mlx::core::array((1.0f - p) - EPSILON);
    auto mask = mlx::core::greater(cumulative_probs, threshold);
    auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
    logprobs = mlx::core::where(mask, logprobs, neg_inf);
  };
  auto apply_mtplx_top_p_inline = [&logprobs](float p) {
    if (p <= 0.0f || p >= 1.0f) return;
    logprobs = compiled_mtplx_top_p_fn()({logprobs, mlx::core::array(p)})[0];
  };

  if (top_k > 0) {
    logprobs = apply_top_k_filter(logprobs, top_k);
  }
  if (temperature_first) {
    apply_mtplx_top_p_inline(top_p);
  } else {
    apply_top_p_inline(top_p);
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
  auto temp_array = mlx::core::array(temperature_first ? 1.0f : (1.0f / temperature));
  auto results = compiled_sampler_filtered({logprobs, temp_array});

  *out_token = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
  *out_logprobs = reinterpret_cast<mlx_array*>(new array(std::move(original_logprobs)));
}

// Stop gradient: detach tensor from computation graph
mlx_array* mlx_stop_gradient(mlx_array* a) {
    try {
        auto arr = reinterpret_cast<mlx::core::array*>(a);
        auto result = mlx::core::stop_gradient(*arr);
        return reinterpret_cast<mlx_array*>(new mlx::core::array(std::move(result)));
    } catch (const std::exception& e) {
        std::cerr << "mlx_stop_gradient error: " << e.what() << std::endl;
        return nullptr;
    }
}

}  // extern "C"
