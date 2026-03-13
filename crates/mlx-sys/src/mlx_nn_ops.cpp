#include "mlx_common.h"

extern "C" {

mlx_array* mlx_array_transpose(mlx_array* handle,
                               const int32_t* axes,
                               size_t axes_len) {
  auto arr = reinterpret_cast<array*>(handle);
  std::vector<int> perm;
  if (axes && axes_len > 0) {
    perm = make_axes(axes, axes_len);
  }
  // When no axes provided, transpose should reverse all dimensions
  array result = perm.empty() ? transpose(*arr) : transpose(*arr, perm);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

void mlx_array_eval(mlx_array* handle) {
  try {
    auto arr = reinterpret_cast<array*>(handle);
    if (arr) {
      arr->eval();
    }
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in array_eval: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in array_eval" << std::endl;
  }
}

void mlx_async_eval(mlx_array** handles, size_t count) {
  try {
    std::vector<array> arrays;
    arrays.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (handles[i]) {
        arrays.push_back(*reinterpret_cast<array*>(handles[i]));
      }
    }
    mlx::core::async_eval(std::move(arrays));
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in async_eval: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in async_eval" << std::endl;
  }
}

size_t mlx_array_size(mlx_array* handle) {
  if (!handle) return 0;
  auto arr = reinterpret_cast<array*>(handle);
  return arr->size();
}

size_t mlx_array_ndim(mlx_array* handle) {
  if (!handle) return 0;
  auto arr = reinterpret_cast<array*>(handle);
  return arr->ndim();
}

void mlx_array_shape(mlx_array* handle, int64_t* out) {
  auto arr = reinterpret_cast<array*>(handle);
  if (!arr || !out) {
    return;
  }
  const Shape& shape = arr->shape();
  for (size_t i = 0; i < shape.size(); ++i) {
    out[i] = shape[i];
  }
}

int64_t mlx_array_shape_at(mlx_array* handle, size_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  if (!arr) {
    return -1;
  }
  const Shape& shape = arr->shape();
  if (axis >= shape.size()) {
    return -1;
  }
  return shape[axis];
}

// Get batch and sequence length for 2D arrays (common pattern in transformers)
// Returns true on success, false if not 2D array
bool mlx_array_get_batch_seq_len(mlx_array* handle, int64_t* batch, int64_t* seq_len) {
  auto arr = reinterpret_cast<array*>(handle);
  if (!arr || !batch || !seq_len) {
    return false;
  }
  const Shape& shape = arr->shape();
  if (shape.size() != 2) {
    return false;
  }
  *batch = shape[0];
  *seq_len = shape[1];
  return true;
}

// Get batch, sequence length, and hidden size for 3D arrays (common pattern in transformers)
// Returns true on success, false if not 3D array
bool mlx_array_get_batch_seq_hidden(mlx_array* handle, int64_t* batch, int64_t* seq_len, int64_t* hidden) {
  auto arr = reinterpret_cast<array*>(handle);
  if (!arr || !batch || !seq_len || !hidden) {
    return false;
  }
  const Shape& shape = arr->shape();
  if (shape.size() != 3) {
    return false;
  }
  *batch = shape[0];
  *seq_len = shape[1];
  *hidden = shape[2];
  return true;
}

} // extern "C" — temporarily close for C++ template

// Read a scalar element from an evaluated array at the given index, casting to
// the requested output type entirely on CPU.  Never creates a GPU astype+eval
// -- that was the root cause of a 4.4 ms per-step stall in decode loops.
namespace {
template <typename Out>
Out read_scalar(const array& arr, size_t index) {
  switch (arr.dtype()) {
    case mlx::core::bool_:    return static_cast<Out>(arr.data<bool>()[index]);
    case mlx::core::uint8:    return static_cast<Out>(arr.data<uint8_t>()[index]);
    case mlx::core::uint16:   return static_cast<Out>(arr.data<uint16_t>()[index]);
    case mlx::core::uint32:   return static_cast<Out>(arr.data<uint32_t>()[index]);
    case mlx::core::int8:     return static_cast<Out>(arr.data<int8_t>()[index]);
    case mlx::core::int16:    return static_cast<Out>(arr.data<int16_t>()[index]);
    case mlx::core::int32:    return static_cast<Out>(arr.data<int32_t>()[index]);
    case mlx::core::int64:    return static_cast<Out>(arr.data<int64_t>()[index]);
    case mlx::core::uint64:   return static_cast<Out>(arr.data<uint64_t>()[index]);
    case mlx::core::float16:
      return static_cast<Out>(static_cast<float>(arr.data<mlx::core::float16_t>()[index]));
    case mlx::core::bfloat16:
      return static_cast<Out>(static_cast<float>(arr.data<mlx::core::bfloat16_t>()[index]));
    case mlx::core::float32:  return static_cast<Out>(arr.data<float>()[index]);
    default:                  return Out{};
  }
}
} // namespace

extern "C" {

bool mlx_array_item_at_float32(mlx_array* handle, size_t index, float* out) {
  if (!handle || !out) return false;
  auto arr = reinterpret_cast<array*>(handle);
  if (index >= arr->size()) return false;
  *out = read_scalar<float>(*arr, index);
  return true;
}

bool mlx_array_item_at_int32(mlx_array* handle, size_t index, int32_t* out) {
  if (!handle || !out) return false;
  auto arr = reinterpret_cast<array*>(handle);
  if (index >= arr->size()) return false;
  *out = read_scalar<int32_t>(*arr, index);
  return true;
}

bool mlx_array_item_at_uint32(mlx_array* handle, size_t index, uint32_t* out) {
  if (!handle || !out) return false;
  auto arr = reinterpret_cast<array*>(handle);
  if (index >= arr->size()) return false;
  *out = read_scalar<uint32_t>(*arr, index);
  return true;
}

int32_t mlx_array_dtype(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  if (!arr) {
    return -1;
  }
  return from_mlx_dtype(arr->dtype());
}

bool mlx_array_to_float32(mlx_array* handle, float* out, size_t len) {
  if (!out) {
    return false;
  }
  auto arr = reinterpret_cast<array*>(handle);
  if (!arr) {
    return false;
  }
  return copy_to_buffer(*arr, out, len);
}

bool mlx_array_to_float32_noeval(mlx_array* handle, float* out, size_t len) {
  if (!out) {
    return false;
  }
  auto arr = reinterpret_cast<array*>(handle);
  if (!arr) {
    return false;
  }
  return copy_to_buffer_noeval(*arr, out, len);
}

bool mlx_array_to_int32(mlx_array* handle, int32_t* out, size_t len) {
  if (!out) {
    return false;
  }
  auto arr = reinterpret_cast<array*>(handle);
  if (!arr) {
    return false;
  }
  return copy_to_buffer(*arr, out, len);
}

bool mlx_array_to_int32_noeval(mlx_array* handle, int32_t* out, size_t len) {
  if (!out) {
    return false;
  }
  auto arr = reinterpret_cast<array*>(handle);
  if (!arr) {
    return false;
  }
  return copy_to_buffer_noeval(*arr, out, len);
}

bool mlx_array_to_uint32(mlx_array* handle, uint32_t* out, size_t len) {
  if (!out) {
    return false;
  }
  auto arr = reinterpret_cast<array*>(handle);
  if (!arr) {
    return false;
  }
  return copy_to_buffer(*arr, out, len);
}

// Extract raw uint16 values from bf16/f16 arrays without f32 conversion.
// The array must already be bfloat16 or float16 dtype — no type casting is done.
bool mlx_array_to_uint16(mlx_array* handle, uint16_t* out, size_t len) {
  if (!out) {
    return false;
  }
  auto arr = reinterpret_cast<array*>(handle);
  if (!arr) {
    return false;
  }
  try {
    // Force materialization
    auto zeros_arr = zeros(arr->shape(), arr->dtype());
    auto materialized = add(*arr, zeros_arr);
    materialized.eval();

    auto flat = flatten(materialized);
    flat.eval();

    if (flat.size() != len) {
      return false;
    }

    // Both bfloat16_t and float16_t are 16-bit types with same memory layout as uint16_t
    auto dtype = flat.dtype();
    if (dtype == mlx::core::bfloat16) {
      const auto* data = flat.data<mlx::core::bfloat16_t>();
      std::memcpy(out, data, len * sizeof(uint16_t));
    } else if (dtype == mlx::core::float16) {
      const auto* data = flat.data<mlx::core::float16_t>();
      std::memcpy(out, data, len * sizeof(uint16_t));
    } else {
      return false; // Only bf16/f16 supported
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] copy_to_buffer(uint16): " << e.what() << std::endl;
    return false;
  }
}

void mlx_array_delete(mlx_array* arr) {
  try {
    delete reinterpret_cast<array*>(arr);
  } catch (const std::exception& e) {
    // Log but don't propagate - destructor exceptions are fatal to Rust FFI
    std::cerr << "[MLX] Exception during array delete: " << e.what() << std::endl;
  } catch (...) {
    // Catch all other exceptions to prevent propagation to Rust
    std::cerr << "[MLX] Unknown exception during array delete" << std::endl;
  }
}

// Random number generation functions
mlx_array* mlx_array_random_uniform(const int64_t* shape,
                                    size_t ndim,
                                    float low,
                                    float high,
                                    int32_t dtype) {
  Shape target_shape = make_shape(shape, ndim);
  array arr =
      mlx::core::random::uniform(low, high, target_shape, to_mlx_dtype(dtype));
  return reinterpret_cast<mlx_array*>(new array(std::move(arr)));
}

mlx_array* mlx_array_random_normal(const int64_t* shape,
                                   size_t ndim,
                                   float mean,
                                   float std,
                                   int32_t dtype) {
  Shape target_shape = make_shape(shape, ndim);
  array arr =
      mlx::core::random::normal(target_shape, to_mlx_dtype(dtype), mean, std);
  return reinterpret_cast<mlx_array*>(new array(std::move(arr)));
}

mlx_array* mlx_array_random_bernoulli(const int64_t* shape,
                                      size_t ndim,
                                      float prob) {
  Shape target_shape = make_shape(shape, ndim);
  array arr = mlx::core::random::bernoulli(prob, target_shape);
  return reinterpret_cast<mlx_array*>(new array(std::move(arr)));
}

mlx_array* mlx_array_randint(const int64_t* shape,
                             size_t ndim,
                             int32_t low,
                             int32_t high) {
  Shape target_shape = make_shape(shape, ndim);
  array arr = mlx::core::random::randint(low, high, target_shape);
  return reinterpret_cast<mlx_array*>(new array(std::move(arr)));
}

mlx_array* mlx_array_categorical(mlx_array* logits_handle, int32_t axis) {
  auto logits_arr = reinterpret_cast<array*>(logits_handle);
  array result = mlx::core::random::categorical(*logits_arr, axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// Comparison operations
mlx_array* mlx_array_equal(mlx_array* lhs, mlx_array* rhs) {
  auto lhs_arr = reinterpret_cast<array*>(lhs);
  auto rhs_arr = reinterpret_cast<array*>(rhs);
  if (!lhs_arr || !rhs_arr) {
    return 0;
  }
  array result = mlx::core::equal(*lhs_arr, *rhs_arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_not_equal(mlx_array* lhs, mlx_array* rhs) {
  auto lhs_arr = reinterpret_cast<array*>(lhs);
  auto rhs_arr = reinterpret_cast<array*>(rhs);
  if (!lhs_arr || !rhs_arr) {
    return 0;
  }
  array result = mlx::core::not_equal(*lhs_arr, *rhs_arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_less(mlx_array* lhs, mlx_array* rhs) {
  auto lhs_arr = reinterpret_cast<array*>(lhs);
  auto rhs_arr = reinterpret_cast<array*>(rhs);
  if (!lhs_arr || !rhs_arr) {
    return 0;
  }
  array result = mlx::core::less(*lhs_arr, *rhs_arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_less_equal(mlx_array* lhs, mlx_array* rhs) {
  auto lhs_arr = reinterpret_cast<array*>(lhs);
  auto rhs_arr = reinterpret_cast<array*>(rhs);
  if (!lhs_arr || !rhs_arr) {
    return 0;
  }
  array result = mlx::core::less_equal(*lhs_arr, *rhs_arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_greater(mlx_array* lhs, mlx_array* rhs) {
  auto lhs_arr = reinterpret_cast<array*>(lhs);
  auto rhs_arr = reinterpret_cast<array*>(rhs);
  if (!lhs_arr || !rhs_arr) {
    return 0;
  }
  array result = mlx::core::greater(*lhs_arr, *rhs_arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_greater_equal(mlx_array* lhs, mlx_array* rhs) {
  auto lhs_arr = reinterpret_cast<array*>(lhs);
  auto rhs_arr = reinterpret_cast<array*>(rhs);
  if (!lhs_arr || !rhs_arr) {
    return 0;
  }
  array result = mlx::core::greater_equal(*lhs_arr, *rhs_arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// Logical operations
mlx_array* mlx_array_logical_and(mlx_array* lhs, mlx_array* rhs) {
  auto lhs_arr = reinterpret_cast<array*>(lhs);
  auto rhs_arr = reinterpret_cast<array*>(rhs);
  if (!lhs_arr || !rhs_arr) {
    return 0;
  }
  array result = mlx::core::logical_and(*lhs_arr, *rhs_arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_logical_or(mlx_array* lhs, mlx_array* rhs) {
  auto lhs_arr = reinterpret_cast<array*>(lhs);
  auto rhs_arr = reinterpret_cast<array*>(rhs);
  if (!lhs_arr || !rhs_arr) {
    return 0;
  }
  array result = mlx::core::logical_or(*lhs_arr, *rhs_arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_logical_not(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::logical_not(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_where(mlx_array* condition, mlx_array* x, mlx_array* y) {
  auto cond_arr = reinterpret_cast<array*>(condition);
  auto x_arr = reinterpret_cast<array*>(x);
  auto y_arr = reinterpret_cast<array*>(y);
  array result = mlx::core::where(*cond_arr, *x_arr, *y_arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// Advanced reduction operations
mlx_array* mlx_array_argmax(mlx_array* handle, int32_t axis, bool keepdims) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::argmax(*arr, axis, keepdims);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_argmin(mlx_array* handle, int32_t axis, bool keepdims) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::argmin(*arr, axis, keepdims);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_max(mlx_array* handle,
                         const int32_t* axes,
                         size_t axes_len,
                         bool keepdims) {
  auto arr = reinterpret_cast<array*>(handle);
  array result =
      (axes_len == 0)
          ? mlx::core::max(*arr, keepdims)
          : mlx::core::max(*arr, make_axes(axes, axes_len), keepdims);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_min(mlx_array* handle,
                         const int32_t* axes,
                         size_t axes_len,
                         bool keepdims) {
  auto arr = reinterpret_cast<array*>(handle);
  array result =
      (axes_len == 0)
          ? mlx::core::min(*arr, keepdims)
          : mlx::core::min(*arr, make_axes(axes, axes_len), keepdims);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_prod(mlx_array* handle,
                          const int32_t* axes,
                          size_t axes_len,
                          bool keepdims) {
  auto arr = reinterpret_cast<array*>(handle);
  array result =
      (axes_len == 0)
          ? mlx::core::prod(*arr, keepdims)
          : mlx::core::prod(*arr, make_axes(axes, axes_len), keepdims);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_var(mlx_array* handle,
                         const int32_t* axes,
                         size_t axes_len,
                         bool keepdims,
                         int32_t ddof) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = (axes_len == 0)
                     ? mlx::core::var(*arr, keepdims, ddof)
                     : mlx::core::var(*arr, make_axes(axes, axes_len), keepdims, ddof);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_std(mlx_array* handle,
                         const int32_t* axes,
                         size_t axes_len,
                         bool keepdims,
                         int32_t ddof) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = (axes_len == 0)
                     ? mlx::core::std(*arr, keepdims, ddof)
                     : mlx::core::std(*arr, make_axes(axes, axes_len), keepdims, ddof);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_cumsum(mlx_array* handle, int32_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::cumsum(*arr, axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_cumprod(mlx_array* handle, int32_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::cumprod(*arr, axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// Array manipulation operations
mlx_array* mlx_array_pad(mlx_array* handle,
                         const int32_t* pad_width,
                         size_t ndim,
                         float constant_value) {
  auto arr = reinterpret_cast<array*>(handle);
  std::vector<std::pair<int, int>> pad_pairs;
  pad_pairs.reserve(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    pad_pairs.push_back({pad_width[i * 2], pad_width[i * 2 + 1]});
  }
  array result = mlx::core::pad(*arr, pad_pairs, array(constant_value));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_roll(mlx_array* handle, int32_t shift, int32_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::roll(*arr, shift, axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// Returns the number of splits, and fills the output array with handles
size_t mlx_array_split_multi(mlx_array* handle,
                             int32_t indices_or_sections,
                             int32_t axis,
                             uint64_t* out_handles,
                             size_t max_outputs) {
  if (!handle || !out_handles) return 0;
  auto arr = reinterpret_cast<array*>(handle);
  auto splits = mlx::core::split(*arr, indices_or_sections, axis);
  size_t count = std::min(splits.size(), max_outputs);
  for (size_t i = 0; i < count; ++i) {
    out_handles[i] =
        reinterpret_cast<uint64_t>(new array(std::move(splits[i])));
  }
  return count;
}

// Keep the old single-output version for backwards compatibility
mlx_array* mlx_array_split(mlx_array* handle,
                           int32_t indices_or_sections,
                           int32_t axis) {
  // Note: This is a simplified version that returns the first split
  // In a full implementation, we'd need to return multiple handles
  auto arr = reinterpret_cast<array*>(handle);
  auto splits = mlx::core::split(*arr, indices_or_sections, axis);
  if (splits.size() > 0) {
    return reinterpret_cast<mlx_array*>(new array(std::move(splits[0])));
  }
  return nullptr;
}

mlx_array* mlx_array_tile(mlx_array* handle,
                          const int32_t* reps,
                          size_t reps_len) {
  auto arr = reinterpret_cast<array*>(handle);
  std::vector<int> target_reps = make_axes(reps, reps_len);
  array result = mlx::core::tile(*arr, target_reps);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_repeat(mlx_array* handle, int32_t repeats, int32_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::repeat(*arr, repeats, axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_squeeze(mlx_array* handle,
                             const int32_t* axes,
                             size_t axes_len) {
  auto arr = reinterpret_cast<array*>(handle);
  if (axes_len == 0) {
    array result = mlx::core::squeeze(*arr);
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } else {
    std::vector<int> target_axes = make_axes(axes, axes_len);
    array result = mlx::core::squeeze(*arr, target_axes);
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  }
}

mlx_array* mlx_array_expand_dims(mlx_array* handle, int32_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::expand_dims(*arr, axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_broadcast_to(mlx_array* handle,
                                  const int64_t* shape,
                                  size_t ndim) {
  auto arr = reinterpret_cast<array*>(handle);
  Shape target_shape = make_shape(shape, ndim);
  array result = mlx::core::broadcast_to(*arr, target_shape);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// Additional math operations
mlx_array* mlx_array_abs(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::abs(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_negative(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::negative(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_sign(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::sign(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_sqrt(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::sqrt(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_square(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::square(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_power(mlx_array* lhs, mlx_array* rhs) {
  auto lhs_arr = reinterpret_cast<array*>(lhs);
  auto rhs_arr = reinterpret_cast<array*>(rhs);
  if (!lhs_arr || !rhs_arr) {
    return 0;
  }
  array result = mlx::core::power(*lhs_arr, *rhs_arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_sin(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::sin(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_cos(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::cos(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_tan(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::tan(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_sinh(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::sinh(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_cosh(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::cosh(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_tanh(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::tanh(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_floor(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::floor(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_ceil(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::ceil(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_round(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::round(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_floor_divide(mlx_array* lhs, mlx_array* rhs) {
  auto a = reinterpret_cast<array*>(lhs);
  auto b = reinterpret_cast<array*>(rhs);
  array result = floor_divide(*a, *b);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_remainder(mlx_array* lhs, mlx_array* rhs) {
  auto a = reinterpret_cast<array*>(lhs);
  auto b = reinterpret_cast<array*>(rhs);
  array result = remainder(*a, *b);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_reciprocal(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = reciprocal(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_arcsin(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = arcsin(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_arccos(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = arccos(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_arctan(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = arctan(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_log10(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = log10(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_log2(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = log2(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_log1p(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = log1p(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// NaN/Inf checking operations (GPU-native)
mlx_array* mlx_array_isnan(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::isnan(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_isinf(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = mlx::core::isinf(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_isfinite(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  // isfinite = !isnan && !isinf
  array nan_mask = mlx::core::isnan(*arr);
  array inf_mask = mlx::core::isinf(*arr);
  array bad_mask = mlx::core::logical_or(nan_mask, inf_mask);
  array result = mlx::core::logical_not(bad_mask);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// Fast operations
mlx_array* mlx_fast_rope(mlx_array* handle,
                         int32_t dims,
                         bool traditional,
                         float base,
                         float scale,
                         int32_t offset) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = fast::rope(*arr, dims, traditional, std::optional<float>(base),
                            scale, offset, std::nullopt);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_fast_scaled_dot_product_attention(mlx_array* queries,
                                                 mlx_array* keys,
                                                 mlx_array* values,
                                                 float scale,
                                                 const char* mask_mode_str,
                                                 mlx_array* mask,
                                                 bool has_mask) {
  auto q = reinterpret_cast<array*>(queries);
  auto k = reinterpret_cast<array*>(keys);
  auto v = reinterpret_cast<array*>(values);
  // Convert C string to std::string, default to empty if null
  std::string mask_mode = mask_mode_str ? std::string(mask_mode_str) : "";

  std::optional<array> mask_arr = std::nullopt;

  // If mask_mode is "causal", don't use mask (MLX handles it internally)
  // Otherwise, if has_mask is true, use the mask array
  if (mask_mode != "causal" && has_mask) {
    auto m = reinterpret_cast<array*>(mask);
    if (m) {
      mask_arr = *m;
    }
  }

  array result = fast::scaled_dot_product_attention(
      *q, *k, *v, scale, mask_mode, mask_arr, std::nullopt);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_fast_rms_norm(mlx_array* x,
                              mlx_array* weight,
                              float eps) {
  auto x_arr = reinterpret_cast<array*>(x);
  std::optional<array> weight_opt = weight ?
      std::optional(*reinterpret_cast<array*>(weight)) : std::nullopt;
  // Use default stream (empty braces)
  array result = fast::rms_norm(*x_arr, weight_opt, eps, {});
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_fast_layer_norm(mlx_array* x,
                                mlx_array* weight,
                                mlx_array* bias,
                                float eps) {
  auto x_arr = reinterpret_cast<array*>(x);
  std::optional<array> weight_opt = weight ?
      std::optional(*reinterpret_cast<array*>(weight)) : std::nullopt;
  std::optional<array> bias_opt = bias ?
      std::optional(*reinterpret_cast<array*>(bias)) : std::nullopt;
  // Use default stream (empty braces)
  array result = fast::layer_norm(*x_arr, weight_opt, bias_opt, eps, {});
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// ── SafeTensors lazy loading ─────────────────────────────────────────────────

/// Load safetensors file using MLX's lazy loading (data read on eval, not upfront).
/// Calls `callback` for each tensor with (name, name_len, array_handle).
/// Returns number of tensors loaded, or -1 on error.
int32_t mlx_load_safetensors(
    const char* path,
    void (*callback)(const char* name, size_t name_len, mlx_array* handle, void* ctx),
    void* ctx
) {
    try {
        auto [tensors, metadata] = mlx::core::load_safetensors(std::string(path));
        int32_t count = 0;
        for (auto& [name, arr] : tensors) {
            auto* handle = reinterpret_cast<mlx_array*>(new array(std::move(arr)));
            callback(name.c_str(), name.size(), handle, ctx);
            count++;
        }
        return count;
    } catch (const std::exception& e) {
        std::cerr << "mlx_load_safetensors error: " << e.what() << std::endl;
        return -1;
    }
}

}  // extern "C"
