#include "mlx_common.h"

extern "C" {

const char* mlx_version() {
  return mlx::core::version();
}

void mlx_seed(uint64_t seed) {
  mlx::core::random::seed(seed);
}

mlx_array* mlx_array_from_int32(const int32_t* data,
                                const int64_t* shape,
                                size_t ndim) {
  Shape target_shape = make_shape(shape, ndim);
  auto arr = new mlx::core::array(data, target_shape, mlx::core::int32);
  return reinterpret_cast<mlx_array*>(arr);
}

mlx_array* mlx_array_from_int64(const int64_t* data,
                                const int64_t* shape,
                                size_t ndim) {
  Shape target_shape = make_shape(shape, ndim);
  auto arr = new mlx::core::array(data, target_shape, mlx::core::int64);
  return reinterpret_cast<mlx_array*>(arr);
}

mlx_array* mlx_array_from_uint32(const uint32_t* data,
                                 const int64_t* shape,
                                 size_t ndim) {
  Shape target_shape = make_shape(shape, ndim);
  auto arr = new array(data, target_shape, mlx::core::uint32);
  return reinterpret_cast<mlx_array*>(arr);
}

mlx_array* mlx_array_from_float32(const float* data,
                                  const int64_t* shape,
                                  size_t ndim) {
  Shape target_shape = make_shape(shape, ndim);
  auto arr = new array(data, target_shape, mlx::core::float32);
  return reinterpret_cast<mlx_array*>(arr);
}

// Create array from bfloat16 raw bytes (uint16 representation)
// This enables zero-copy loading of bf16 weights from safetensors
mlx_array* mlx_array_from_bfloat16(const uint16_t* data,
                                   const int64_t* shape,
                                   size_t ndim) {
  Shape target_shape = make_shape(shape, ndim);
  // bfloat16_t has the same memory layout as uint16_t (just the bits_ field)
  // so we can safely reinterpret_cast
  auto bf16_data = reinterpret_cast<const mlx::core::bfloat16_t*>(data);
  auto arr = new array(bf16_data, target_shape, mlx::core::bfloat16);
  return reinterpret_cast<mlx_array*>(arr);
}

// Create array from float16 raw bytes (uint16 representation)
// This enables zero-copy loading of f16 weights from safetensors
mlx_array* mlx_array_from_float16(const uint16_t* data,
                                  const int64_t* shape,
                                  size_t ndim) {
  Shape target_shape = make_shape(shape, ndim);
  // float16_t has the same memory layout as uint16_t
  auto f16_data = reinterpret_cast<const mlx::core::float16_t*>(data);
  auto arr = new array(f16_data, target_shape, mlx::core::float16);
  return reinterpret_cast<mlx_array*>(arr);
}

// Create array from uint8 raw bytes
// Used for loading FP8 E4M3 weights (1 byte per element)
mlx_array* mlx_array_from_uint8(const uint8_t* data,
                                const int64_t* shape,
                                size_t ndim) {
  try {
    Shape target_shape = make_shape(shape, ndim);
    auto arr = new array(data, target_shape, mlx::core::uint8);
    return reinterpret_cast<mlx_array*>(arr);
  } catch (const std::exception& e) {
    std::cerr << "[MLX] mlx_array_from_uint8: " << e.what() << std::endl;
    return nullptr;
  }
}

// Convert FP8 E4M3 array to target dtype using MLX's from_fp8
// Input must be a uint8 array containing FP8 E4M3 encoded values
// target_dtype: 0=float32, 2=float16, 3=bfloat16
mlx_array* mlx_from_fp8(mlx_array* handle, int32_t target_dtype) {
  if (!handle) {
    std::cerr << "[MLX] mlx_from_fp8: null handle" << std::endl;
    return nullptr;
  }
  try {
    auto& arr = *reinterpret_cast<array*>(handle);
    auto dtype = to_mlx_dtype(target_dtype);
    auto result = mlx::core::from_fp8(arr, dtype);
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  } catch (const std::exception& e) {
    std::cerr << "[MLX] mlx_from_fp8: " << e.what() << std::endl;
    return nullptr;
  }
}

mlx_array* mlx_array_scalar_float(double value) {
  auto arr = new array(static_cast<float>(value));
  return reinterpret_cast<mlx_array*>(arr);
}

// Create a scalar with a specific dtype (no AsType node) — matches Python's array(val, dtype)
mlx_array* mlx_array_scalar_float_dtype(double value, int32_t dtype) {
  auto dt = to_mlx_dtype(dtype);
  auto arr = new array(static_cast<float>(value), dt);
  return reinterpret_cast<mlx_array*>(arr);
}

mlx_array* mlx_array_scalar_int(int32_t value) {
  auto arr = new array(value);
  return reinterpret_cast<mlx_array*>(arr);
}

mlx_array* mlx_array_zeros(const int64_t* shape, size_t ndim, int32_t dtype) {
  Shape target_shape = make_shape(shape, ndim);
  auto arr = new array(zeros(target_shape, to_mlx_dtype(dtype)));
  return reinterpret_cast<mlx_array*>(arr);
}

mlx_array* mlx_array_ones(const int64_t* shape, size_t ndim, int32_t dtype) {
  Shape target_shape = make_shape(shape, ndim);
  auto arr = new array(ones(target_shape, to_mlx_dtype(dtype)));
  return reinterpret_cast<mlx_array*>(arr);
}

mlx_array* mlx_array_full(const int64_t* shape,
                          size_t ndim,
                          mlx_array* value_handle,
                          int32_t dtype,
                          bool has_dtype) {
  auto value = reinterpret_cast<array*>(value_handle);
  Shape target_shape = make_shape(shape, ndim);
  array result =
      has_dtype ? full(std::move(target_shape), *value, to_mlx_dtype(dtype))
                : full(std::move(target_shape), *value);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_reshape(mlx_array* handle,
                             const int64_t* shape,
                             size_t ndim) {
  auto arr = reinterpret_cast<mlx::core::array*>(handle);
  Shape target_shape = make_shape(shape, ndim);
  array result = reshape(*arr, std::move(target_shape));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_astype(mlx_array* handle, int32_t dtype) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = astype(*arr, to_mlx_dtype(dtype));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_copy(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = copy(*arr);
  // Keep lazy - let caller decide when to evaluate (matches Python MLX behavior)
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_log_softmax(mlx_array* handle, int32_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  std::vector<int> axes{axis};
  array lse = logsumexp(*arr, axes, true);
  array result = *arr - lse;
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_logsumexp(mlx_array* handle,
                               const int32_t* axes,
                               size_t axes_len,
                               bool keepdims) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = (axes_len == 0)
                     ? logsumexp(*arr, keepdims)
                     : logsumexp(*arr, make_axes(axes, axes_len), keepdims);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_softmax(mlx_array* handle, int32_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = softmax(*arr, axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_softmax_precise(mlx_array* handle, int32_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = softmax(*arr, axis, /*precise=*/true);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_sigmoid(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = sigmoid(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_exp(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = exp(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_log(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = log(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_sum(mlx_array* handle,
                         const int32_t* axes,
                         size_t axes_len,
                         bool keepdims) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = (axes_len == 0)
                     ? sum(*arr, keepdims)
                     : sum(*arr, make_axes(axes, axes_len), keepdims);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_mean(mlx_array* handle,
                          const int32_t* axes,
                          size_t axes_len,
                          bool keepdims) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = (axes_len == 0)
                     ? mean(*arr, keepdims)
                     : mean(*arr, make_axes(axes, axes_len), keepdims);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_stack(mlx_array* const* handles,
                           size_t len,
                           int32_t axis) {
  std::vector<array> inputs;
  inputs.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    auto arr = reinterpret_cast<array*>(handles[i]);
    inputs.push_back(*arr);
  }
  array result = stack(inputs, axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_clip(mlx_array* handle, double lo, double hi) {
  auto arr = reinterpret_cast<array*>(handle);
  std::optional<array> lower;
  std::optional<array> upper;
  // Cast bounds to input dtype to avoid f32 promotion with bf16/f16 inputs
  if (std::isfinite(lo)) {
    lower = astype(array(static_cast<float>(lo)), arr->dtype());
  }
  if (std::isfinite(hi)) {
    upper = astype(array(static_cast<float>(hi)), arr->dtype());
  }
  array result = clip(*arr, lower, upper);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_minimum(mlx_array* lhs, mlx_array* rhs) {
  auto a = reinterpret_cast<array*>(lhs);
  auto b = reinterpret_cast<array*>(rhs);
  array result = minimum(*a, *b);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_maximum(mlx_array* lhs, mlx_array* rhs) {
  auto a = reinterpret_cast<array*>(lhs);
  auto b = reinterpret_cast<array*>(rhs);
  array result = maximum(*a, *b);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_add(mlx_array* lhs, mlx_array* rhs) {
  auto a = reinterpret_cast<array*>(lhs);
  auto b = reinterpret_cast<array*>(rhs);
  array result = *a + *b;
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_sub(mlx_array* lhs, mlx_array* rhs) {
  auto a = reinterpret_cast<array*>(lhs);
  auto b = reinterpret_cast<array*>(rhs);
  array result = *a - *b;
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_mul(mlx_array* lhs, mlx_array* rhs) {
  auto a = reinterpret_cast<array*>(lhs);
  auto b = reinterpret_cast<array*>(rhs);
  array result = *a * *b;
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_div(mlx_array* lhs, mlx_array* rhs) {
  auto a = reinterpret_cast<array*>(lhs);
  auto b = reinterpret_cast<array*>(rhs);
  array result = *a / *b;
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_add_scalar(mlx_array* handle, double value) {
  auto arr = reinterpret_cast<array*>(handle);
  // Create scalar directly in target dtype (no AsType node) — matches Python's to_array()
  auto dt = mlx::core::issubdtype(arr->dtype(), mlx::core::floating) ? arr->dtype() : mlx::core::float32;
  array scalar(static_cast<float>(value), dt);
  array result = *arr + scalar;
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_mul_scalar(mlx_array* handle, double value) {
  auto arr = reinterpret_cast<array*>(handle);
  auto dt = mlx::core::issubdtype(arr->dtype(), mlx::core::floating) ? arr->dtype() : mlx::core::float32;
  array scalar(static_cast<float>(value), dt);
  array result = *arr * scalar;
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_sub_scalar(mlx_array* handle, double value) {
  auto arr = reinterpret_cast<array*>(handle);
  auto dt = mlx::core::issubdtype(arr->dtype(), mlx::core::floating) ? arr->dtype() : mlx::core::float32;
  array scalar(static_cast<float>(value), dt);
  array result = *arr - scalar;
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_div_scalar(mlx_array* handle, double value) {
  auto arr = reinterpret_cast<array*>(handle);
  auto dt = mlx::core::issubdtype(arr->dtype(), mlx::core::floating) ? arr->dtype() : mlx::core::float32;
  array scalar(static_cast<float>(value), dt);
  array result = *arr / scalar;
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_take(mlx_array* handle,
                          mlx_array* indices_handle,
                          int32_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  auto idx = reinterpret_cast<array*>(indices_handle);
  if (!arr || !idx) {
    return 0;
  }
  array result = take(*arr, *idx, axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_take_along_axis(mlx_array* handle,
                                     mlx_array* indices_handle,
                                     int32_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  auto idx = reinterpret_cast<array*>(indices_handle);
  if (!arr || !idx) {
    return 0;
  }
  array result = take_along_axis(*arr, *idx, axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// Put values into array at specified indices along an axis
// This is simpler than scatter and matches the Python put_along_axis API
mlx_array* mlx_array_put_along_axis(mlx_array* handle,
                                     mlx_array* indices_handle,
                                     mlx_array* values_handle,
                                     int32_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  auto indices = reinterpret_cast<array*>(indices_handle);
  auto values = reinterpret_cast<array*>(values_handle);
  if (!arr || !indices || !values) {
    return 0;
  }

  array result = mlx::core::put_along_axis(*arr, *indices, *values, axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_arange(double start,
                            double stop,
                            double step,
                            int32_t dtype) {
  double actual_step = (std::abs(step) < 1e-12) ? 1.0 : step;
  array result = arange(start, stop, actual_step, to_mlx_dtype(dtype));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_linspace(double start,
                              double stop,
                              int32_t num,
                              int32_t dtype,
                              bool has_dtype) {
  array result = has_dtype ? linspace(start, stop, num, to_mlx_dtype(dtype))
                           : linspace(start, stop, num);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_eye(int32_t n,
                         int32_t m,
                         int32_t k,
                         int32_t dtype,
                         bool has_dtype) {
  array result = has_dtype ? eye(n, m, k, to_mlx_dtype(dtype)) : eye(n, m, k);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_slice(mlx_array* handle,
                           const int64_t* starts,
                           const int64_t* stops,
                           size_t ndim) {
  auto arr = reinterpret_cast<array*>(handle);
  Shape start_shape = make_shape(starts, ndim);
  Shape stop_shape = make_shape(stops, ndim);
  array result = slice(*arr, std::move(start_shape), std::move(stop_shape));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_slice_update(mlx_array* src_handle,
                                   mlx_array* update_handle,
                                   const int64_t* starts,
                                   const int64_t* stops,
                                   size_t ndim) {
  auto src = reinterpret_cast<array*>(src_handle);
  auto update = reinterpret_cast<array*>(update_handle);
  Shape start_shape = make_shape(starts, ndim);
  Shape stop_shape = make_shape(stops, ndim);
  array result = slice_update(*src, *update, std::move(start_shape), std::move(stop_shape));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// In-place slice update - modifies src directly instead of creating new array
// This matches Python's behavior: self.keys[..., prev:offset, :] = keys
void mlx_array_slice_update_inplace(mlx_array* src_handle,
                                     mlx_array* update_handle,
                                     const int64_t* starts,
                                     const int64_t* stops,
                                     size_t ndim) {
  auto src = reinterpret_cast<array*>(src_handle);
  auto update = reinterpret_cast<array*>(update_handle);
  Shape start_shape = make_shape(starts, ndim);
  Shape stop_shape = make_shape(stops, ndim);
  array result = slice_update(*src, *update, std::move(start_shape), std::move(stop_shape));
  // Use overwrite_descriptor to modify src in-place (no new allocation!)
  src->overwrite_descriptor(result);
}

// Optimized slice assignment along a single axis - no allocation for shape access
// Returns new array with the slice updated
mlx_array* mlx_array_slice_assign_axis(mlx_array* src_handle,
                                        mlx_array* update_handle,
                                        size_t axis,
                                        int64_t start,
                                        int64_t end) {
  auto src = reinterpret_cast<array*>(src_handle);
  auto update = reinterpret_cast<array*>(update_handle);

  // Access shape directly without allocation
  const Shape& shape = src->shape();
  size_t ndim = shape.size();

  // Build start and stop shapes
  Shape start_shape(ndim, 0);
  Shape stop_shape = shape;

  start_shape[axis] = start;
  stop_shape[axis] = end;

  // Perform slice update and return new array
  array result = slice_update(*src, *update, std::move(start_shape), std::move(stop_shape));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// Optimized in-place slice assignment along a single axis - no allocation
// Modifies src directly
void mlx_array_slice_assign_axis_inplace(mlx_array* src_handle,
                                          mlx_array* update_handle,
                                          size_t axis,
                                          int64_t start,
                                          int64_t end) {
  auto src = reinterpret_cast<array*>(src_handle);
  auto update = reinterpret_cast<array*>(update_handle);

  // Access shape directly without allocation
  const Shape& shape = src->shape();
  size_t ndim = shape.size();

  // Build start and stop shapes
  Shape start_shape(ndim, 0);
  Shape stop_shape = shape;

  start_shape[axis] = start;
  stop_shape[axis] = end;

  // Perform slice update and modify src in-place
  array result = slice_update(*src, *update, std::move(start_shape), std::move(stop_shape));
  src->overwrite_descriptor(result);
}

// Optimized slice along a single axis - no allocation for shape access
// Returns sliced array
mlx_array* mlx_array_slice_axis(mlx_array* src_handle,
                                 size_t axis,
                                 int64_t start,
                                 int64_t end) {
  auto src = reinterpret_cast<array*>(src_handle);

  // Access shape directly without allocation
  const Shape& shape = src->shape();
  size_t ndim = shape.size();

  // Build start and stop shapes
  Shape start_shape(ndim, 0);
  Shape stop_shape = shape;

  start_shape[axis] = start;
  stop_shape[axis] = end;

  // Perform slice and return new array
  array result = slice(*src, std::move(start_shape), std::move(stop_shape));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_scatter(mlx_array* src_handle,
                             mlx_array* indices_handle,
                             mlx_array* updates_handle,
                             int32_t axis) {
  auto src = reinterpret_cast<array*>(src_handle);
  auto indices = reinterpret_cast<array*>(indices_handle);
  auto updates = reinterpret_cast<array*>(updates_handle);
  array result = scatter(*src, *indices, *updates, axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_concatenate(mlx_array* const* handles,
                                 size_t len,
                                 int32_t axis) {
  std::vector<array> inputs;
  inputs.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    auto arr = reinterpret_cast<array*>(handles[i]);
    inputs.push_back(*arr);
  }
  array result = concatenate(std::move(inputs), axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_sort(mlx_array* handle, int32_t axis, bool has_axis) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = has_axis ? sort(*arr, axis) : sort(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_argsort(mlx_array* handle, int32_t axis, bool has_axis) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = has_axis ? argsort(*arr, axis) : argsort(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_partition(mlx_array* handle,
                               int32_t kth,
                               int32_t axis,
                               bool has_axis) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = has_axis ? partition(*arr, kth, axis) : partition(*arr, kth);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_argpartition(mlx_array* handle,
                                  int32_t kth,
                                  int32_t axis,
                                  bool has_axis) {
  auto arr = reinterpret_cast<array*>(handle);
  array result =
      has_axis ? argpartition(*arr, kth, axis) : argpartition(*arr, kth);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_matmul(mlx_array* lhs, mlx_array* rhs) {
  auto a = reinterpret_cast<array*>(lhs);
  auto b = reinterpret_cast<array*>(rhs);
  array result = matmul(*a, *b);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// Extract uint8 data from a uint8 array (used for MXFP8 scales in SafeTensors writer)
bool mlx_array_to_uint8(mlx_array* handle, uint8_t* out, size_t len) {
  if (!out) {
    return false;
  }
  auto arr = reinterpret_cast<array*>(handle);
  if (!arr) {
    return false;
  }
  try {
    auto flat = flatten(*arr);
    flat.eval();

    if (flat.size() != len) {
      return false;
    }

    if (flat.dtype() != mlx::core::uint8) {
      return false;
    }

    const auto* data = flat.data<uint8_t>();
    std::memcpy(out, data, len * sizeof(uint8_t));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[MLX] mlx_array_to_uint8: " << e.what() << std::endl;
    return false;
  }
}

// Compute D = beta * C + alpha * (A @ B)
// This is a fused operation that's more efficient than separate matmul and add
mlx_array* mlx_array_addmm(mlx_array* c_handle, mlx_array* a_handle, mlx_array* b_handle,
                           float alpha, float beta) {
  auto c = reinterpret_cast<array*>(c_handle);
  auto a = reinterpret_cast<array*>(a_handle);
  auto b = reinterpret_cast<array*>(b_handle);
  array result = addmm(*c, *a, *b, alpha, beta);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

}  // extern "C"
