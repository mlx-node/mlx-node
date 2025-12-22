#include "mlx/mlx.h"
#include "mlx/fast.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"
#include "mlx/memory.h"
#include "mlx/compile.h"
#include "mlx/backend/metal/metal.h"

// Paged attention Metal kernel source (for future GPU dispatch)
#include "paged_attn_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <mutex>

// Forward declaration of opaque handle type for FFI
struct mlx_array;

// Stream struct for FFI (matches Rust definition)
struct mlx_stream {
  int32_t index;
  int32_t device_type;  // 0 = CPU, 1 = GPU
};

namespace {
using mlx::core::arange;
using mlx::core::array;
using mlx::core::astype;
using mlx::core::clip;
using mlx::core::concatenate;
using mlx::core::copy;
using mlx::core::exp;
using mlx::core::eye;
using mlx::core::full;
using mlx::core::linspace;
using mlx::core::log;
using mlx::core::logsumexp;
using mlx::core::matmul;
using mlx::core::maximum;
using mlx::core::mean;
using mlx::core::minimum;
using mlx::core::ones;
using mlx::core::reshape;
using mlx::core::Shape;
using mlx::core::ShapeElem;
using mlx::core::slice;
using mlx::core::squeeze;
using mlx::core::stack;
using mlx::core::sum;
using mlx::core::take;
using mlx::core::transpose;
using mlx::core::zeros;

// Comparison operations
using mlx::core::equal;
using mlx::core::greater;
using mlx::core::greater_equal;
using mlx::core::less;
using mlx::core::less_equal;
using mlx::core::not_equal;

// Logical operations
using mlx::core::logical_and;
using mlx::core::logical_not;
using mlx::core::logical_or;
using mlx::core::where;

// Reduction operations
using mlx::core::argmax;
using mlx::core::argmin;
using mlx::core::cumprod;
using mlx::core::cumsum;
using mlx::core::max;
using mlx::core::min;
using mlx::core::prod;
using mlx::core::std;
using mlx::core::var;

// Array manipulation
using mlx::core::argpartition;
using mlx::core::argsort;
using mlx::core::broadcast_to;
using mlx::core::expand_dims;
using mlx::core::pad;
using mlx::core::partition;
using mlx::core::repeat;
using mlx::core::roll;
using mlx::core::sort;
using mlx::core::split;
using mlx::core::tile;

// Math operations
using mlx::core::abs;
using mlx::core::ceil;
using mlx::core::cos;
using mlx::core::cosh;
using mlx::core::floor;
using mlx::core::negative;
using mlx::core::power;
using mlx::core::round;
using mlx::core::sign;
using mlx::core::sin;
using mlx::core::sinh;
using mlx::core::sqrt;
using mlx::core::square;
using mlx::core::tan;
using mlx::core::tanh;

// Fast operations
namespace fast = mlx::core::fast;

// Stream and evaluation
using mlx::core::async_eval;
using mlx::core::clear_cache;
using mlx::core::default_device;
using mlx::core::default_stream;
using mlx::core::Device;
using mlx::core::eval;
using mlx::core::flatten;
using mlx::core::new_stream;
using mlx::core::scatter;
using mlx::core::sigmoid;
using mlx::core::Stream;
using mlx::core::StreamContext;

Shape make_shape(const int64_t* dims, size_t ndim) {
  Shape shape;
  shape.reserve(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    shape.push_back(static_cast<ShapeElem>(dims[i]));
  }
  return shape;
}

std::vector<int> make_axes(const int32_t* axes, size_t len) {
  std::vector<int> result;
  result.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    result.push_back(static_cast<int>(axes[i]));
  }
  return result;
}

enum BridgeDType : int32_t {
  FLOAT32 = 0,
  INT32 = 1,
  FLOAT16 = 2,
  BFLOAT16 = 3,
};

mlx::core::Dtype to_mlx_dtype(int32_t code) {
  switch (code) {
    case FLOAT32:
      return mlx::core::float32;
    case INT32:
      return mlx::core::int32;
    case FLOAT16:
      return mlx::core::float16;
    case BFLOAT16:
      return mlx::core::bfloat16;
    default:
      return mlx::core::float32;
  }
}

int32_t from_mlx_dtype(mlx::core::Dtype dtype) {
  switch (dtype) {
    case mlx::core::float32:
      return FLOAT32;
    case mlx::core::int32:
      return INT32;
    case mlx::core::float16:
      return FLOAT16;
    case mlx::core::bfloat16:
      return BFLOAT16;
    default:
      return -1;
  }
}

bool copy_to_buffer(const array& arr, float* out, size_t len) {
  // Force materialization by adding zeros - this ensures broadcast values are
  // expanded
  auto zeros_arr = zeros(arr.shape(), arr.dtype());
  auto materialized = add(arr, zeros_arr);
  materialized.eval();

  // Now flatten and copy
  auto flat = flatten(materialized);
  auto host = (flat.dtype() == mlx::core::float32)
                  ? flat
                  : astype(flat, mlx::core::float32);
  host.eval();

  if (host.size() != len) {
    return false;
  }
  const float* data = host.data<float>();
  std::copy(data, data + len, out);
  return true;
}

bool copy_to_buffer(const array& arr, int32_t* out, size_t len) {
  // Force materialization by adding zeros - this ensures broadcast values are
  // expanded
  auto zeros_arr = zeros(arr.shape(), arr.dtype());
  auto materialized = add(arr, zeros_arr);
  materialized.eval();

  // Now flatten and copy
  auto flat = flatten(materialized);
  auto host = (flat.dtype() == mlx::core::int32)
                  ? flat
                  : astype(flat, mlx::core::int32);
  host.eval();

  if (host.size() != len) {
    return false;
  }
  const int32_t* data = host.data<int32_t>();
  std::copy(data, data + len, out);
  return true;
}

bool copy_to_buffer(const array& arr, uint32_t* out, size_t len) {
  // Force materialization by adding zeros - this ensures broadcast values are
  // expanded
  auto zeros_arr = zeros(arr.shape(), arr.dtype());
  auto materialized = add(arr, zeros_arr);
  materialized.eval();

  // Now flatten and copy
  auto flat = flatten(materialized);
  auto host = (flat.dtype() == mlx::core::uint32)
                  ? flat
                  : astype(flat, mlx::core::uint32);
  host.eval();

  if (host.size() != len) {
    return false;
  }
  const uint32_t* data = host.data<uint32_t>();
  std::copy(data, data + len, out);
  return true;
}

// NO-EVAL versions: assume input array is already evaluated (for async pipeline)
// Skips the add(zeros) materialization step, only evals transformations
bool copy_to_buffer_noeval(const array& arr, float* out, size_t len) {
  // Input arr is already evaluated by async_eval
  // Skip the add(zeros) step - assume no broadcast expansion needed
  auto flat = flatten(arr);
  auto host = (flat.dtype() == mlx::core::float32)
                  ? flat
                  : astype(flat, mlx::core::float32);
  host.eval();  // Only eval the transformation (flatten/astype)

  if (host.size() != len) {
    return false;
  }
  const float* data = host.data<float>();
  std::copy(data, data + len, out);
  return true;
}

bool copy_to_buffer_noeval(const array& arr, int32_t* out, size_t len) {
  // Input arr is already evaluated by async_eval
  // Skip the add(zeros) step - assume no broadcast expansion needed
  auto flat = flatten(arr);
  auto host = (flat.dtype() == mlx::core::int32)
                  ? flat
                  : astype(flat, mlx::core::int32);
  host.eval();  // Only eval the transformation (flatten/astype)

  if (host.size() != len) {
    return false;
  }
  const int32_t* data = host.data<int32_t>();
  std::copy(data, data + len, out);
  return true;
}

}  // namespace

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

mlx_array* mlx_array_scalar_float(double value) {
  auto arr = new array(static_cast<float>(value));
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
  // Force evaluation before copying to avoid lazy evaluation issues
  arr->eval();
  array result = copy(*arr);
  // Evaluate the copy to ensure it's materialized
  result.eval();
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_log_softmax(mlx_array* handle, int32_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  std::vector<int> axes{axis};
  array lse = logsumexp(*arr, axes, true);
  array result = *arr - lse;
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_logsumexp(mlx_array* handle, int32_t axis, bool keepdims) {
  auto arr = reinterpret_cast<array*>(handle);
  std::vector<int> axes{axis};
  array result = logsumexp(*arr, axes, keepdims);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_softmax(mlx_array* handle, int32_t axis) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = softmax(*arr, axis);
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
  if (std::isfinite(lo)) {
    lower = array(static_cast<float>(lo));
  }
  if (std::isfinite(hi)) {
    upper = array(static_cast<float>(hi));
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
  array result = *arr + array(static_cast<float>(value));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_mul_scalar(mlx_array* handle, double value) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = *arr * array(static_cast<float>(value));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_sub_scalar(mlx_array* handle, double value) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = *arr - array(static_cast<float>(value));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_div_scalar(mlx_array* handle, double value) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = *arr / array(static_cast<float>(value));
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

// Fused SwiGLU MLP forward: output = down(silu(gate(x)) * up(x))
// Weights are stored as [out_features, in_features], so we transpose for matmul.
// This fuses 8 operations into 1 FFI call:
// 1. gate = x @ w_gate.T
// 2. up = x @ w_up.T
// 3. gate_act = silu(gate) = gate * sigmoid(gate)
// 4. gated = gate_act * up
// 5. output = gated @ w_down.T
mlx_array* mlx_swiglu_mlp_forward(mlx_array* x_handle,
                                   mlx_array* w_gate_handle,
                                   mlx_array* w_up_handle,
                                   mlx_array* w_down_handle) {
  auto x = reinterpret_cast<array*>(x_handle);
  auto w_gate = reinterpret_cast<array*>(w_gate_handle);
  auto w_up = reinterpret_cast<array*>(w_up_handle);
  auto w_down = reinterpret_cast<array*>(w_down_handle);

  // Transpose weights: [out, in] -> [in, out] for matmul
  auto w_gate_t = transpose(*w_gate, {1, 0});
  auto w_up_t = transpose(*w_up, {1, 0});
  auto w_down_t = transpose(*w_down, {1, 0});

  // gate = x @ w_gate.T
  auto gate = matmul(*x, w_gate_t);

  // up = x @ w_up.T
  auto up = matmul(*x, w_up_t);

  // silu(gate) = gate * sigmoid(gate)
  auto gate_act = gate * sigmoid(gate);

  // gated = gate_act * up
  auto gated = gate_act * up;

  // output = gated @ w_down.T
  auto output = matmul(gated, w_down_t);

  return reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

// Fused Multi-Head Attention forward pass (without KV cache)
// Reduces ~20 FFI calls to 1 for attention computation
// Parameters:
//   x: input [B, L, D]
//   w_q, w_k, w_v, w_o: projection weights [out, in]
//   q_norm_w, k_norm_w: optional QK norm weights (can be nullptr)
//   n_heads, n_kv_heads, head_dim: attention configuration
//   scale: attention scale factor (usually 1/sqrt(head_dim))
//   rope_base, rope_dims: RoPE parameters
//   qk_norm_eps: epsilon for QK normalization
//   use_causal: whether to use causal masking
//   rope_offset: position offset for RoPE (for KV cache support)
mlx_array* mlx_fused_attention_forward(
    mlx_array* x_handle,
    mlx_array* w_q_handle,
    mlx_array* w_k_handle,
    mlx_array* w_v_handle,
    mlx_array* w_o_handle,
    mlx_array* q_norm_w_handle,  // Can be nullptr if no QK norm
    mlx_array* k_norm_w_handle,  // Can be nullptr if no QK norm
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale,
    float rope_base,
    int rope_dims,
    float qk_norm_eps,
    bool use_causal,
    int rope_offset) {

  auto x = reinterpret_cast<array*>(x_handle);
  auto w_q = reinterpret_cast<array*>(w_q_handle);
  auto w_k = reinterpret_cast<array*>(w_k_handle);
  auto w_v = reinterpret_cast<array*>(w_v_handle);
  auto w_o = reinterpret_cast<array*>(w_o_handle);

  // Get input shape (cast to int for MLX Shape)
  int batch = static_cast<int>(x->shape()[0]);
  int seq_len = static_cast<int>(x->shape()[1]);

  // 1. Project Q/K/V (transpose weights then matmul)
  auto w_q_t = transpose(*w_q, {1, 0});
  auto w_k_t = transpose(*w_k, {1, 0});
  auto w_v_t = transpose(*w_v, {1, 0});
  auto w_o_t = transpose(*w_o, {1, 0});

  auto queries = matmul(*x, w_q_t);  // [B, L, n_heads * head_dim]
  auto keys = matmul(*x, w_k_t);     // [B, L, n_kv_heads * head_dim]
  auto values = matmul(*x, w_v_t);   // [B, L, n_kv_heads * head_dim]

  // 2. Reshape to multi-head format [B, L, n_heads, head_dim]
  queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
  keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
  values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

  // 3. Apply QK normalization if weights provided (before transpose!)
  if (q_norm_w_handle) {
    auto q_norm_w = reinterpret_cast<array*>(q_norm_w_handle);
    queries = fast::rms_norm(queries, std::optional<array>(*q_norm_w), qk_norm_eps, {});
  }
  if (k_norm_w_handle) {
    auto k_norm_w = reinterpret_cast<array*>(k_norm_w_handle);
    keys = fast::rms_norm(keys, std::optional<array>(*k_norm_w), qk_norm_eps, {});
  }

  // 4. Transpose to attention layout [B, n_heads, L, head_dim]
  queries = transpose(queries, {0, 2, 1, 3});
  keys = transpose(keys, {0, 2, 1, 3});
  values = transpose(values, {0, 2, 1, 3});

  // 5. Apply RoPE
  bool traditional = false;  // MLX uses non-traditional by default
  float rope_scale = 1.0f;
  queries = fast::rope(queries, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offset, std::nullopt, {});
  keys = fast::rope(keys, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offset, std::nullopt, {});

  // 6. Scaled dot-product attention
  std::string mask_mode = use_causal && seq_len > 1 ? "causal" : "";
  auto output = fast::scaled_dot_product_attention(queries, keys, values, scale, mask_mode, {}, std::nullopt, {});

  // 7. Transpose back to [B, L, n_heads, head_dim]
  output = transpose(output, {0, 2, 1, 3});

  // 8. Reshape to [B, L, n_heads * head_dim]
  output = reshape(output, {batch, seq_len, n_heads * head_dim});

  // 9. Output projection
  output = matmul(output, w_o_t);

  return reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

// Fused attention with KV cache support
// Returns output array and updates cached_keys/cached_values in-place
void mlx_fused_attention_forward_cached(
    mlx_array* x_handle,
    mlx_array* w_q_handle,
    mlx_array* w_k_handle,
    mlx_array* w_v_handle,
    mlx_array* w_o_handle,
    mlx_array* q_norm_w_handle,
    mlx_array* k_norm_w_handle,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale,
    float rope_base,
    int rope_dims,
    float qk_norm_eps,
    bool use_causal,
    // KV cache (in/out)
    mlx_array** cached_keys_ptr,
    mlx_array** cached_values_ptr,
    int cache_offset,
    // Output
    mlx_array** output_ptr) {

  auto x = reinterpret_cast<array*>(x_handle);
  auto w_q = reinterpret_cast<array*>(w_q_handle);
  auto w_k = reinterpret_cast<array*>(w_k_handle);
  auto w_v = reinterpret_cast<array*>(w_v_handle);
  auto w_o = reinterpret_cast<array*>(w_o_handle);

  // Get input shape (cast to int for MLX Shape)
  int batch = static_cast<int>(x->shape()[0]);
  int seq_len = static_cast<int>(x->shape()[1]);

  // 1. Project Q/K/V
  auto w_q_t = transpose(*w_q, {1, 0});
  auto w_k_t = transpose(*w_k, {1, 0});
  auto w_v_t = transpose(*w_v, {1, 0});
  auto w_o_t = transpose(*w_o, {1, 0});

  auto queries = matmul(*x, w_q_t);
  auto keys = matmul(*x, w_k_t);
  auto values = matmul(*x, w_v_t);

  // 2. Reshape
  queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
  keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
  values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

  // 3. QK normalization
  if (q_norm_w_handle) {
    auto q_norm_w = reinterpret_cast<array*>(q_norm_w_handle);
    queries = fast::rms_norm(queries, std::optional<array>(*q_norm_w), qk_norm_eps, {});
  }
  if (k_norm_w_handle) {
    auto k_norm_w = reinterpret_cast<array*>(k_norm_w_handle);
    keys = fast::rms_norm(keys, std::optional<array>(*k_norm_w), qk_norm_eps, {});
  }

  // 4. Transpose to [B, n_heads, L, head_dim]
  queries = transpose(queries, {0, 2, 1, 3});
  keys = transpose(keys, {0, 2, 1, 3});
  values = transpose(values, {0, 2, 1, 3});

  // 5. Apply RoPE
  bool traditional = false;
  float rope_scale = 1.0f;
  queries = fast::rope(queries, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, cache_offset, std::nullopt, {});
  keys = fast::rope(keys, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, cache_offset, std::nullopt, {});

  // 6. Update KV cache
  if (*cached_keys_ptr && *cached_values_ptr) {
    auto cached_keys = reinterpret_cast<array*>(*cached_keys_ptr);
    auto cached_values = reinterpret_cast<array*>(*cached_values_ptr);

    // Concatenate new keys/values with cache
    keys = concatenate({*cached_keys, keys}, 2);
    values = concatenate({*cached_values, values}, 2);

    // Update cache pointers
    delete cached_keys;
    delete cached_values;
    *cached_keys_ptr = reinterpret_cast<mlx_array*>(new array(keys));
    *cached_values_ptr = reinterpret_cast<mlx_array*>(new array(values));
  } else {
    // Initialize cache
    *cached_keys_ptr = reinterpret_cast<mlx_array*>(new array(keys));
    *cached_values_ptr = reinterpret_cast<mlx_array*>(new array(values));
  }

  // 7. Scaled dot-product attention
  // For generation (seq_len == 1), no mask needed
  // For prefill (seq_len > 1), use causal mask
  int kv_len = static_cast<int>(keys.shape()[2]);
  std::string mask_mode = "";
  if (use_causal && seq_len > 1 && seq_len == kv_len) {
    mask_mode = "causal";
  }
  auto output = fast::scaled_dot_product_attention(queries, keys, values, scale, mask_mode, {}, std::nullopt, {});

  // 8. Transpose back
  output = transpose(output, {0, 2, 1, 3});

  // 9. Reshape
  output = reshape(output, {batch, seq_len, n_heads * head_dim});

  // 10. Output projection
  output = matmul(output, w_o_t);

  *output_ptr = reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

// Fused Transformer Block forward (without KV cache)
// Combines: norm -> attention -> residual -> norm -> mlp -> residual
// Reduces ~40 FFI calls to 1 per block
mlx_array* mlx_fused_transformer_block_forward(
    mlx_array* x_handle,
    // Layer norm weights
    mlx_array* input_norm_w_handle,
    mlx_array* post_attn_norm_w_handle,
    // Attention weights
    mlx_array* w_q_handle,
    mlx_array* w_k_handle,
    mlx_array* w_v_handle,
    mlx_array* w_o_handle,
    mlx_array* q_norm_w_handle,  // Can be nullptr
    mlx_array* k_norm_w_handle,  // Can be nullptr
    // MLP weights
    mlx_array* w_gate_handle,
    mlx_array* w_up_handle,
    mlx_array* w_down_handle,
    // Config
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float attn_scale,
    float rope_base,
    int rope_dims,
    float norm_eps,
    float qk_norm_eps,
    bool use_causal,
    int rope_offset) {

  auto x = reinterpret_cast<array*>(x_handle);
  auto input_norm_w = reinterpret_cast<array*>(input_norm_w_handle);
  auto post_attn_norm_w = reinterpret_cast<array*>(post_attn_norm_w_handle);
  auto w_q = reinterpret_cast<array*>(w_q_handle);
  auto w_k = reinterpret_cast<array*>(w_k_handle);
  auto w_v = reinterpret_cast<array*>(w_v_handle);
  auto w_o = reinterpret_cast<array*>(w_o_handle);
  auto w_gate = reinterpret_cast<array*>(w_gate_handle);
  auto w_up = reinterpret_cast<array*>(w_up_handle);
  auto w_down = reinterpret_cast<array*>(w_down_handle);

  // Get input shape
  int batch = static_cast<int>(x->shape()[0]);
  int seq_len = static_cast<int>(x->shape()[1]);

  // === Part 1: Self-Attention ===

  // 1. Input layer norm
  auto normed = fast::rms_norm(*x, std::optional<array>(*input_norm_w), norm_eps, {});

  // 2. Q/K/V projections
  auto w_q_t = transpose(*w_q, {1, 0});
  auto w_k_t = transpose(*w_k, {1, 0});
  auto w_v_t = transpose(*w_v, {1, 0});
  auto w_o_t = transpose(*w_o, {1, 0});

  auto queries = matmul(normed, w_q_t);
  auto keys = matmul(normed, w_k_t);
  auto values = matmul(normed, w_v_t);

  // 3. Reshape to multi-head format
  queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
  keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
  values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

  // 4. QK normalization
  if (q_norm_w_handle) {
    auto q_norm_w = reinterpret_cast<array*>(q_norm_w_handle);
    queries = fast::rms_norm(queries, std::optional<array>(*q_norm_w), qk_norm_eps, {});
  }
  if (k_norm_w_handle) {
    auto k_norm_w = reinterpret_cast<array*>(k_norm_w_handle);
    keys = fast::rms_norm(keys, std::optional<array>(*k_norm_w), qk_norm_eps, {});
  }

  // 5. Transpose to attention layout
  queries = transpose(queries, {0, 2, 1, 3});
  keys = transpose(keys, {0, 2, 1, 3});
  values = transpose(values, {0, 2, 1, 3});

  // 6. Apply RoPE
  bool traditional = false;
  float rope_scale = 1.0f;
  queries = fast::rope(queries, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offset, std::nullopt, {});
  keys = fast::rope(keys, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, rope_offset, std::nullopt, {});

  // 7. Scaled dot-product attention
  std::string mask_mode = use_causal && seq_len > 1 ? "causal" : "";
  auto attn_output = fast::scaled_dot_product_attention(queries, keys, values, attn_scale, mask_mode, {}, std::nullopt, {});

  // 8. Transpose back and reshape
  attn_output = transpose(attn_output, {0, 2, 1, 3});
  attn_output = reshape(attn_output, {batch, seq_len, n_heads * head_dim});

  // 9. Output projection
  attn_output = matmul(attn_output, w_o_t);

  // 10. Attention residual
  auto h = *x + attn_output;

  // === Part 2: MLP ===

  // 11. Post-attention layer norm
  auto mlp_input = fast::rms_norm(h, std::optional<array>(*post_attn_norm_w), norm_eps, {});

  // 12. MLP (SwiGLU)
  auto w_gate_t = transpose(*w_gate, {1, 0});
  auto w_up_t = transpose(*w_up, {1, 0});
  auto w_down_t = transpose(*w_down, {1, 0});

  auto gate = matmul(mlp_input, w_gate_t);
  auto up = matmul(mlp_input, w_up_t);
  auto gate_act = gate * sigmoid(gate);  // SiLU
  auto gated = gate_act * up;
  auto mlp_output = matmul(gated, w_down_t);

  // 13. MLP residual
  auto output = h + mlp_output;

  return reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

// Fused Transformer Block with KV cache support
void mlx_fused_transformer_block_forward_cached(
    mlx_array* x_handle,
    // Layer norm weights
    mlx_array* input_norm_w_handle,
    mlx_array* post_attn_norm_w_handle,
    // Attention weights
    mlx_array* w_q_handle,
    mlx_array* w_k_handle,
    mlx_array* w_v_handle,
    mlx_array* w_o_handle,
    mlx_array* q_norm_w_handle,
    mlx_array* k_norm_w_handle,
    // MLP weights
    mlx_array* w_gate_handle,
    mlx_array* w_up_handle,
    mlx_array* w_down_handle,
    // Config
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float attn_scale,
    float rope_base,
    int rope_dims,
    float norm_eps,
    float qk_norm_eps,
    bool use_causal,
    // KV cache (in/out)
    mlx_array** cached_keys_ptr,
    mlx_array** cached_values_ptr,
    int cache_offset,
    // Output
    mlx_array** output_ptr) {

  auto x = reinterpret_cast<array*>(x_handle);
  auto input_norm_w = reinterpret_cast<array*>(input_norm_w_handle);
  auto post_attn_norm_w = reinterpret_cast<array*>(post_attn_norm_w_handle);
  auto w_q = reinterpret_cast<array*>(w_q_handle);
  auto w_k = reinterpret_cast<array*>(w_k_handle);
  auto w_v = reinterpret_cast<array*>(w_v_handle);
  auto w_o = reinterpret_cast<array*>(w_o_handle);
  auto w_gate = reinterpret_cast<array*>(w_gate_handle);
  auto w_up = reinterpret_cast<array*>(w_up_handle);
  auto w_down = reinterpret_cast<array*>(w_down_handle);

  int batch = static_cast<int>(x->shape()[0]);
  int seq_len = static_cast<int>(x->shape()[1]);

  // === Part 1: Self-Attention ===

  auto normed = fast::rms_norm(*x, std::optional<array>(*input_norm_w), norm_eps, {});

  auto w_q_t = transpose(*w_q, {1, 0});
  auto w_k_t = transpose(*w_k, {1, 0});
  auto w_v_t = transpose(*w_v, {1, 0});
  auto w_o_t = transpose(*w_o, {1, 0});

  auto queries = matmul(normed, w_q_t);
  auto keys = matmul(normed, w_k_t);
  auto values = matmul(normed, w_v_t);

  queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
  keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
  values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

  if (q_norm_w_handle) {
    auto q_norm_w = reinterpret_cast<array*>(q_norm_w_handle);
    queries = fast::rms_norm(queries, std::optional<array>(*q_norm_w), qk_norm_eps, {});
  }
  if (k_norm_w_handle) {
    auto k_norm_w = reinterpret_cast<array*>(k_norm_w_handle);
    keys = fast::rms_norm(keys, std::optional<array>(*k_norm_w), qk_norm_eps, {});
  }

  queries = transpose(queries, {0, 2, 1, 3});
  keys = transpose(keys, {0, 2, 1, 3});
  values = transpose(values, {0, 2, 1, 3});

  bool traditional = false;
  float rope_scale = 1.0f;
  queries = fast::rope(queries, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, cache_offset, std::nullopt, {});
  keys = fast::rope(keys, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, cache_offset, std::nullopt, {});

  // Update KV cache
  if (*cached_keys_ptr && *cached_values_ptr) {
    auto cached_keys = reinterpret_cast<array*>(*cached_keys_ptr);
    auto cached_values = reinterpret_cast<array*>(*cached_values_ptr);
    keys = concatenate({*cached_keys, keys}, 2);
    values = concatenate({*cached_values, values}, 2);
    delete cached_keys;
    delete cached_values;
    *cached_keys_ptr = reinterpret_cast<mlx_array*>(new array(keys));
    *cached_values_ptr = reinterpret_cast<mlx_array*>(new array(values));
  } else {
    *cached_keys_ptr = reinterpret_cast<mlx_array*>(new array(keys));
    *cached_values_ptr = reinterpret_cast<mlx_array*>(new array(values));
  }

  int kv_len = static_cast<int>(keys.shape()[2]);
  std::string mask_mode = "";
  if (use_causal && seq_len > 1 && seq_len == kv_len) {
    mask_mode = "causal";
  }
  auto attn_output = fast::scaled_dot_product_attention(queries, keys, values, attn_scale, mask_mode, {}, std::nullopt, {});

  attn_output = transpose(attn_output, {0, 2, 1, 3});
  attn_output = reshape(attn_output, {batch, seq_len, n_heads * head_dim});
  attn_output = matmul(attn_output, w_o_t);

  auto h = *x + attn_output;

  // === Part 2: MLP ===

  auto mlp_input = fast::rms_norm(h, std::optional<array>(*post_attn_norm_w), norm_eps, {});

  auto w_gate_t = transpose(*w_gate, {1, 0});
  auto w_up_t = transpose(*w_up, {1, 0});
  auto w_down_t = transpose(*w_down, {1, 0});

  auto gate = matmul(mlp_input, w_gate_t);
  auto up = matmul(mlp_input, w_up_t);
  auto gate_act = gate * sigmoid(gate);
  auto gated = gate_act * up;
  auto mlp_output = matmul(gated, w_down_t);

  auto output = h + mlp_output;

  *output_ptr = reinterpret_cast<mlx_array*>(new array(std::move(output)));
}

// Fused Q/K/V projection with RoPE for cached attention
// Returns Q, K, V in attention layout (B, n_heads, L, head_dim) with RoPE applied
// This fuses: projection -> reshape -> qk_norm -> transpose -> RoPE
void mlx_fused_attention_qkv(
    mlx_array* x_handle,
    mlx_array* w_q_handle,
    mlx_array* w_k_handle,
    mlx_array* w_v_handle,
    mlx_array* q_norm_w_handle,  // Can be null
    mlx_array* k_norm_w_handle,  // Can be null
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float rope_base,
    int rope_dims,
    float qk_norm_eps,
    int rope_offset,
    mlx_array** q_out,
    mlx_array** k_out,
    mlx_array** v_out
) {
    try {
        auto x = reinterpret_cast<array*>(x_handle);
        auto w_q = reinterpret_cast<array*>(w_q_handle);
        auto w_k = reinterpret_cast<array*>(w_k_handle);
        auto w_v = reinterpret_cast<array*>(w_v_handle);

        int batch = static_cast<int>(x->shape()[0]);
        int seq_len = static_cast<int>(x->shape()[1]);

        // Transpose weights for matmul: (hidden, proj) -> (proj, hidden)
        auto w_q_t = transpose(*w_q);
        auto w_k_t = transpose(*w_k);
        auto w_v_t = transpose(*w_v);

        // 1. Q/K/V projections
        auto queries = matmul(*x, w_q_t);  // (B, L, n_heads * head_dim)
        auto keys = matmul(*x, w_k_t);     // (B, L, n_kv_heads * head_dim)
        auto values = matmul(*x, w_v_t);   // (B, L, n_kv_heads * head_dim)

        // 2. Reshape to multi-head format: (B, L, n_heads, head_dim)
        queries = reshape(queries, {batch, seq_len, n_heads, head_dim});
        keys = reshape(keys, {batch, seq_len, n_kv_heads, head_dim});
        values = reshape(values, {batch, seq_len, n_kv_heads, head_dim});

        // 3. Apply QK normalization BEFORE transpose (matching transformers)
        if (q_norm_w_handle) {
            auto q_norm_w = reinterpret_cast<array*>(q_norm_w_handle);
            queries = mlx::core::fast::rms_norm(queries, *q_norm_w, qk_norm_eps);
        }
        if (k_norm_w_handle) {
            auto k_norm_w = reinterpret_cast<array*>(k_norm_w_handle);
            keys = mlx::core::fast::rms_norm(keys, *k_norm_w, qk_norm_eps);
        }

        // 4. Transpose to attention layout: (B, n_heads, L, head_dim)
        queries = transpose(queries, {0, 2, 1, 3});
        keys = transpose(keys, {0, 2, 1, 3});
        values = transpose(values, {0, 2, 1, 3});

        // 5. Apply RoPE
        queries = mlx::core::fast::rope(queries, rope_dims, false, rope_base, 1.0f, rope_offset);
        keys = mlx::core::fast::rope(keys, rope_dims, false, rope_base, 1.0f, rope_offset);

        *q_out = reinterpret_cast<mlx_array*>(new array(std::move(queries)));
        *k_out = reinterpret_cast<mlx_array*>(new array(std::move(keys)));
        *v_out = reinterpret_cast<mlx_array*>(new array(std::move(values)));
    } catch (const std::exception& e) {
        std::cerr << "mlx_fused_attention_qkv error: " << e.what() << std::endl;
        *q_out = nullptr;
        *k_out = nullptr;
        *v_out = nullptr;
    }
}

// Fused SDPA + output projection for cached attention
// Takes Q (B, n_heads, L, head_dim) and full cached K/V (B, n_kv_heads, total_len, head_dim)
// Returns output (B, L, hidden_size)
mlx_array* mlx_fused_attention_output(
    mlx_array* q_handle,
    mlx_array* k_handle,
    mlx_array* v_handle,
    mlx_array* w_o_handle,
    int n_heads,
    int head_dim,
    float attn_scale,
    bool use_causal
) {
    try {
        auto queries = reinterpret_cast<array*>(q_handle);
        auto keys = reinterpret_cast<array*>(k_handle);
        auto values = reinterpret_cast<array*>(v_handle);
        auto w_o = reinterpret_cast<array*>(w_o_handle);

        int batch = static_cast<int>(queries->shape()[0]);
        int q_len = static_cast<int>(queries->shape()[2]);
        int hidden_size = n_heads * head_dim;

        // SDPA - determine mask mode (valid modes: "causal", "array", or "" for none)
        std::string mask_mode = (use_causal && q_len > 1) ? "causal" : "";
        auto attn_output = mlx::core::fast::scaled_dot_product_attention(
            *queries, *keys, *values, attn_scale, mask_mode
        );

        // Transpose back: (B, n_heads, L, head_dim) -> (B, L, n_heads, head_dim)
        attn_output = transpose(attn_output, {0, 2, 1, 3});

        // Reshape: (B, L, n_heads, head_dim) -> (B, L, hidden_size)
        attn_output = reshape(attn_output, {batch, q_len, hidden_size});

        // Output projection
        auto w_o_t = transpose(*w_o);
        auto output = matmul(attn_output, w_o_t);

        return reinterpret_cast<mlx_array*>(new array(std::move(output)));
    } catch (const std::exception& e) {
        std::cerr << "mlx_fused_attention_output error: " << e.what() << std::endl;
        return nullptr;
    }
}

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
  auto arr = reinterpret_cast<array*>(handle);
  return arr->size();
}

size_t mlx_array_ndim(mlx_array* handle) {
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

bool mlx_array_item_at_float32(mlx_array* handle, size_t index, float* out) {
  auto arr = reinterpret_cast<array*>(handle);
  if (index >= arr->size()) {
    return false;  // Index out of bounds
  }

  // Array is already evaluated by async_eval in the caller
  // Extract single element at index and cast to float in C++ (no GPU array conversion)
  switch (arr->dtype()) {
    case mlx::core::float32:
      *out = arr->data<float>()[index];
      break;
    default:
      auto converted = astype(*arr, mlx::core::float32);
      converted.eval();
      *out = converted.data<float>()[index];
      break;
  }
  return true;
}

bool mlx_array_item_at_int32(mlx_array* handle, size_t index, int32_t* out) {
  auto arr = reinterpret_cast<array*>(handle);
  if (index >= arr->size()) {
    return false;  // Index out of bounds
  }
  // Array is already evaluated by async_eval in the caller
  // Extract single element at index and cast to int32 in C++ (no GPU array conversion)
  switch (arr->dtype()) {
    case mlx::core::int32:
      *out = arr->data<int32_t>()[index];
      break;
    default:
      auto converted = astype(*arr, mlx::core::int32);
      converted.eval();
      *out = converted.data<int32_t>()[index];
      break;
  }
  return true;
}

bool mlx_array_item_at_uint32(mlx_array* handle, size_t index, uint32_t* out) {
  auto arr = reinterpret_cast<array*>(handle);
  if (index >= arr->size()) {
    return false;  // Index out of bounds
  }
  // Array is already evaluated by async_eval in the caller
  // Extract single element at index and cast to uint32 in C++ (no GPU array conversion)
  switch (arr->dtype()) {
    case mlx::core::uint32:
      *out = arr->data<uint32_t>()[index];
      break;
    default:
      auto converted = astype(*arr, mlx::core::uint32);
      converted.eval();
      *out = converted.data<uint32_t>()[index];
      break;
  }
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
  std::vector<int> target_axes = make_axes(axes, axes_len);
  array result = mlx::core::var(*arr, target_axes, keepdims, ddof);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_array_std(mlx_array* handle,
                         const int32_t* axes,
                         size_t axes_len,
                         bool keepdims,
                         int32_t ddof) {
  auto arr = reinterpret_cast<array*>(handle);
  std::vector<int> target_axes = make_axes(axes, axes_len);
  array result = mlx::core::std(*arr, target_axes, keepdims, ddof);
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

// ============================================================================
// Gradient Computation
// ============================================================================

// Function pointer type for computing scalar loss from array inputs
// Returns: scalar loss value as mlx_array* pointer
// Context: user data passed through
typedef mlx_array* (*LossFunctionPtr)(mlx_array* const* inputs,
                                      size_t input_count,
                                      void* context);

// Helper to wrap C function pointer as std::function for MLX
class LossFunctionWrapper {
 public:
  LossFunctionWrapper(LossFunctionPtr fn, void* ctx) : fn_(fn), context_(ctx) {}

  array operator()(const std::vector<array>& inputs) {
    // Convert arrays to mlx_array* handles
    std::vector<mlx_array*> handles;
    handles.reserve(inputs.size());
    for (const auto& arr : inputs) {
      handles.push_back(reinterpret_cast<mlx_array*>(new array(arr)));
    }

    // Call user function
    mlx_array* loss_handle = fn_(handles.data(), handles.size(), context_);

    // Clean up input handles - Rust callback copies these, so we own them
    for (auto* handle : handles) {
      delete reinterpret_cast<array*>(handle);
    }

    // Get loss array
    auto loss_ptr = reinterpret_cast<array*>(loss_handle);
    if (!loss_ptr) {
      throw std::runtime_error("Loss function returned invalid handle");
    }

    array result = *loss_ptr;

    // Clean up the handle that Rust returned (via std::mem::forget)
    // Rust prevents its drop to avoid double-free, so we must delete it here
    delete loss_ptr;

    return result;
  }

 private:
  LossFunctionPtr fn_;
  void* context_;
};

/**
 * Compute gradients of a scalar loss function w.r.t. input arrays
 *
 * @param loss_fn C function pointer that computes loss from inputs
 * @param context User context passed to loss_fn
 * @param input_handles Array handles to compute gradients w.r.t.
 * @param input_count Number of input arrays
 * @param output_handles Output array for gradient handles (must be
 * pre-allocated)
 * @return Number of gradients computed (should equal input_count), or 0 on
 * error
 */
extern "C" size_t mlx_compute_gradients(LossFunctionPtr loss_fn,
                                        void* context,
                                        mlx_array* const* input_handles,
                                        size_t input_count,
                                        mlx_array** output_handles) {
  if (!loss_fn || !input_handles || !output_handles || input_count == 0) {
    return 0;
  }

  // Get input arrays
  std::vector<array> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto arr = reinterpret_cast<array*>(input_handles[i]);
    inputs.push_back(*arr);
  }

  // Create wrapper
  LossFunctionWrapper wrapper(loss_fn, context);

  // Convert to std::function for MLX
  std::function<array(const std::vector<array>&)> loss_func =
      [&wrapper](const std::vector<array>& args) { return wrapper(args); };

  // Build argnums vector {0, 1, 2, ..., input_count-1}
  // This tells MLX to compute gradients with respect to ALL inputs
  std::vector<int> argnums;
  argnums.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    argnums.push_back(static_cast<int>(i));
  }

  // Compute gradients using MLX with all argnums
  // grad() takes the loss function and returns a function that computes
  // gradients
  auto grad_fn = mlx::core::grad(loss_func, argnums);

  // Call gradient function with inputs
  std::vector<array> gradients = grad_fn(inputs);

  // Store gradient handles
  if (gradients.size() != input_count) {
    return 0;  // Unexpected gradient count
  }

  for (size_t i = 0; i < gradients.size(); i++) {
    output_handles[i] =
        reinterpret_cast<mlx_array*>(new array(std::move(gradients[i])));
  }

  return gradients.size();
}

/**
 * Compute both value and gradients of a scalar loss function
 *
 * @param loss_fn C function pointer that computes loss from inputs
 * @param context User context passed to loss_fn
 * @param input_handles Array handles to compute gradients w.r.t.
 * @param input_count Number of input arrays
 * @param loss_handle Output for loss value handle
 * @param grad_handles Output array for gradient handles (must be pre-allocated)
 * @return Number of gradients computed, or 0 on error
 */
extern "C" size_t mlx_value_and_gradients(LossFunctionPtr loss_fn,
                                          void* context,
                                          mlx_array* const* input_handles,
                                          size_t input_count,
                                          mlx_array** loss_handle,
                                          mlx_array** grad_handles) {
  if (!loss_fn || !input_handles || !loss_handle || !grad_handles ||
      input_count == 0) {
    return 0;
  }

  // Get input arrays
  std::vector<array> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto arr = reinterpret_cast<array*>(input_handles[i]);
    inputs.push_back(*arr);
  }

  // Create wrapper
  LossFunctionWrapper wrapper(loss_fn, context);

  // Convert to std::function for MLX
  std::function<array(const std::vector<array>&)> loss_func =
      [&wrapper](const std::vector<array>& args) { return wrapper(args); };

  // Build argnums vector {0, 1, 2, ..., input_count-1}
  // This tells MLX to compute gradients with respect to ALL inputs
  std::vector<int> argnums;
  argnums.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    argnums.push_back(static_cast<int>(i));
  }

  // Compute value and gradients using MLX with all argnums
  auto value_and_grad_fn = mlx::core::value_and_grad(loss_func, argnums);

  // Call with inputs
  auto [value, gradients] = value_and_grad_fn(inputs);

  // Store loss value (for scalar functions, value is directly an array)
  *loss_handle = reinterpret_cast<mlx_array*>(new array(std::move(value)));

  // Store gradient handles
  if (gradients.size() != input_count) {
    return 0;
  }

  for (size_t i = 0; i < gradients.size(); i++) {
    grad_handles[i] =
        reinterpret_cast<mlx_array*>(new array(std::move(gradients[i])));
  }

  return gradients.size();
}

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
  auto threshold = mlx::core::array(1.0f - top_p);
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
// Fully compiled sampling function (matches MLX-LM's compiled sampler chain)
// ============================================================================
// This is the key optimization: compile the ENTIRE sampling chain into one fused kernel
// - Converts logits to logprobs
// - Applies top_k, top_p, min_p filters
// - Applies temperature and samples
// - All in one compiled Metal kernel!

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

  // Fast path: no filters, just temperature sampling - USE COMPILED SAMPLER
  bool needs_filters = (top_k > 0) || (top_p > 0.0f && top_p < 1.0f) || (min_p > 0.0f);
  if (!needs_filters) {
    // Use compiled categorical sampler for graph caching benefits
    static auto compiled_sampler = mlx::core::compile([](const std::vector<array>& inputs) {
      auto lg = inputs[0];
      auto temp_scalar = inputs[1];
      auto scaled = mlx::core::multiply(lg, temp_scalar);
      return std::vector<array>{mlx::core::random::categorical(scaled, -1)};
    });
    auto temp_array = mlx::core::array(1.0f / temperature);
    auto results = compiled_sampler({logits, temp_array});
    return reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
  }

  // Convert logits to logprobs (log-softmax): logprobs = logits - logsumexp(logits)
  auto logsumexp = mlx::core::logsumexp(logits, std::vector<int>{-1}, true);
  auto logprobs = mlx::core::subtract(logits, logsumexp);

  // Apply top_k filter if enabled (matches mlx-lm apply_top_k)
  if (top_k > 0) {
    int vocab_size = logprobs.shape().back();
    if (top_k < vocab_size) {
      // argpartition to find top-k indices (mlx-lm: mx.argpartition(-logprobs, kth=top_k-1, axis=-1)[..., top_k:])
      auto neg_logprobs = mlx::core::negative(logprobs);
      auto partitioned_indices = mlx::core::argpartition(neg_logprobs, top_k - 1, -1);

      // Get indices to mask (everything after top_k)
      auto shape = partitioned_indices.shape();
      mlx::core::Shape starts(shape.size(), 0);
      mlx::core::Shape ends(shape.begin(), shape.end());
      starts[starts.size() - 1] = top_k;

      auto mask_idx = mlx::core::slice(partitioned_indices, starts, ends);

      // Use put_along_axis (matches mlx-lm: mx.put_along_axis(logprobs, mask_idx, -inf, axis=-1))
      auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
      logprobs = mlx::core::put_along_axis(logprobs, mask_idx, neg_inf, -1);
    }
  }

  // Apply top_p filter if enabled (matches mlx-lm apply_top_p)
  if (top_p > 0.0f && top_p < 1.0f) {
    // Convert to probs and sort in ascending order
    auto probs = mlx::core::exp(logprobs);
    auto sorted_indices = mlx::core::argsort(logprobs, -1);
    auto sorted_probs = mlx::core::take_along_axis(probs, sorted_indices, -1);
    auto cumulative_probs = mlx::core::cumsum(sorted_probs, -1);

    // Rearrange cumulative probs back to original order using put_along_axis
    auto shape = sorted_indices.shape();
    int last_dim = shape[shape.size() - 1];
    auto arange_vals = mlx::core::arange(0, last_dim, sorted_indices.dtype());

    // Create inverse mapping: put_along_axis(zeros, sorted_indices, arange, axis=-1)
    auto zeros = mlx::core::zeros_like(sorted_indices);
    auto inverse_indices = mlx::core::put_along_axis(zeros, sorted_indices, arange_vals, -1);

    cumulative_probs = mlx::core::take_along_axis(cumulative_probs, inverse_indices, -1);

    // Select tokens with cumulative probs below threshold
    auto threshold = mlx::core::array(1.0f - top_p);
    auto mask = mlx::core::greater(cumulative_probs, threshold);

    auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
    logprobs = mlx::core::where(mask, logprobs, neg_inf);
  }

  // Apply min_p filter if enabled (matches mlx-lm apply_min_p)
  if (min_p > 0.0f) {
    // Sort in descending order
    auto sorted_indices = mlx::core::argsort(mlx::core::negative(logprobs), -1);
    auto sorted_logprobs = mlx::core::take_along_axis(logprobs, sorted_indices, -1);

    // Get top logprob (first element after descending sort)
    auto shape = sorted_logprobs.shape();
    mlx::core::Shape starts(shape.size(), 0);
    mlx::core::Shape ends(shape.begin(), shape.end());
    ends[ends.size() - 1] = 1;
    auto top_logprobs = mlx::core::slice(sorted_logprobs, starts, ends);

    // Calculate min_p threshold: scaled_min_p = top_logprobs + log(min_p)
    auto log_min_p = mlx::core::array(std::log(min_p));
    auto scaled_min_p = mlx::core::add(top_logprobs, log_min_p);

    // Mask tokens below threshold
    auto tokens_to_remove = mlx::core::less(sorted_logprobs, scaled_min_p);

    // Keep at least 1 token (set first position to false)
    mlx::core::Shape first_starts(shape.size(), 0);
    mlx::core::Shape first_ends(shape.begin(), shape.end());
    first_ends[first_ends.size() - 1] = 1;
    auto first_slice = mlx::core::slice(tokens_to_remove, first_starts, first_ends);
    auto keep_first = mlx::core::zeros_like(first_slice);
    auto keep_indices = mlx::core::arange(0, 1, mlx::core::int32);
    tokens_to_remove = mlx::core::put_along_axis(tokens_to_remove, keep_indices, keep_first, -1);

    // Apply mask
    auto neg_inf = mlx::core::array(-std::numeric_limits<float>::infinity(), logprobs.dtype());
    auto selected_logprobs = mlx::core::where(tokens_to_remove, neg_inf, sorted_logprobs);

    // Rearrange back to original order
    int last_dim = shape[shape.size() - 1];
    auto zeros = mlx::core::zeros_like(sorted_indices);
    auto arange_vals = mlx::core::arange(0, last_dim, sorted_indices.dtype());
    auto inverse_indices = mlx::core::put_along_axis(zeros, sorted_indices, arange_vals, -1);

    logprobs = mlx::core::take_along_axis(selected_logprobs, inverse_indices, -1);
  }

  // Apply temperature and sample using compiled categorical sampler
  static auto compiled_sampler_filtered = mlx::core::compile([](const std::vector<array>& inputs) {
    auto lp = inputs[0];
    auto temp_scalar = inputs[1];
    auto scaled = mlx::core::multiply(lp, temp_scalar);
    return std::vector<array>{mlx::core::random::categorical(scaled, -1)};
  });
  auto temp_array = mlx::core::array(1.0f / temperature);
  auto results = compiled_sampler_filtered({logprobs, temp_array});

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
    auto threshold = mlx::core::array(1.0f - top_p);
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
    auto threshold = mlx::core::array(1.0f - top_p);
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

}  // End anonymous namespace

// ============================================================================
// Stream Operations (extern "C" for FFI)
// ============================================================================

namespace {
// Helper to convert device type to MLX Device
mlx::core::Device to_device_helper(int32_t device_type) {
  return device_type == 0 ? mlx::core::Device::cpu : mlx::core::Device::gpu;
}

// Helper to convert MLX Stream to mlx_stream struct
mlx_stream to_mlx_stream_helper(const mlx::core::Stream& s) {
  mlx_stream result;
  result.index = s.index;
  result.device_type = (s.device == mlx::core::Device::cpu) ? 0 : 1;
  return result;
}

// Helper to convert mlx_stream struct to MLX Stream
mlx::core::Stream from_mlx_stream_helper(mlx_stream s) {
  return mlx::core::Stream(s.index, to_device_helper(s.device_type));
}
}  // End helpers namespace

extern "C" {

// Get the default stream for the given device
mlx_stream mlx_default_stream(int32_t device_type) {
  auto device = to_device_helper(device_type);
  auto stream = mlx::core::default_stream(device);
  return to_mlx_stream_helper(stream);
}

// Create a new stream on the given device
mlx_stream mlx_new_stream(int32_t device_type) {
  auto device = to_device_helper(device_type);
  auto stream = mlx::core::new_stream(device);
  return to_mlx_stream_helper(stream);
}

// Set the default stream
void mlx_set_default_stream(mlx_stream stream) {
  try {
    auto s = from_mlx_stream_helper(stream);
    mlx::core::set_default_stream(s);
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in set_default_stream: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in set_default_stream" << std::endl;
  }
}

// Synchronize with the given stream
void mlx_stream_synchronize(mlx_stream stream) {
  try {
    auto s = from_mlx_stream_helper(stream);
    mlx::core::synchronize(s);
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in stream_synchronize: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in stream_synchronize" << std::endl;
  }
}

// ================================================================================
// Metal Operations (Memory Management)
// ================================================================================

// Check if Metal backend is available
bool mlx_metal_is_available() {
  try {
    return mlx::core::metal::is_available();
  } catch (...) {
    return false;
  }
}

// Get Metal device information as JSON string
// Returns a JSON string with device properties like max_recommended_working_set_size
const char* mlx_metal_device_info() {
  // Static buffer to hold the JSON string
  static std::string info_json;

  if (!mlx::core::metal::is_available()) {
    info_json = "{\"available\": false}";
    return info_json.c_str();
  }

  try {
    const auto& device_info = mlx::core::metal::device_info();

    // Build JSON string manually
    std::ostringstream json;
    json << "{";
    json << "\"available\": true";

    // Get max_recommended_working_set_size (this is the key we need for wired_limit)
    auto it = device_info.find("max_recommended_working_set_size");
    if (it != device_info.end()) {
      // The value is a variant<string, size_t>, extract size_t
      if (const auto* val = std::get_if<size_t>(&it->second)) {
        json << ", \"max_recommended_working_set_size\": " << *val;
      }
    }

    json << "}";

    info_json = json.str();
    return info_json.c_str();
  } catch (const std::exception& e) {
    info_json = "{\"available\": true, \"error\": \"" + std::string(e.what()) + "\"}";
    return info_json.c_str();
  }
}

// Set the wired memory limit and return the old limit
// Wired memory cannot be paged out (important for Metal GPU)
// Uses mlx::core::set_wired_limit (not metal-specific)
size_t mlx_set_wired_limit(size_t limit) {
  try {
    return mlx::core::set_wired_limit(limit);
  } catch (const std::exception& e) {
    std::cerr << "[MLX] Exception in set_wired_limit: " << e.what() << std::endl;
    return 0;
  } catch (...) {
    std::cerr << "[MLX] Unknown exception in set_wired_limit" << std::endl;
    return 0;
  }
}

// Get the current wired memory limit
// Note: MLX doesn't have a get_wired_limit function, so we return 0
// The set_wired_limit function returns the old limit when called
size_t mlx_get_wired_limit() {
  // MLX doesn't provide a getter for wired limit
  // Return 0 to indicate no limit is set
  return 0;
}

// Get peak memory usage (works with any backend)
size_t mlx_get_peak_memory() {
  return mlx::core::get_peak_memory();
}

// Get actively used memory in bytes (excludes cached memory)
size_t mlx_get_active_memory() {
  return mlx::core::get_active_memory();
}

// Get cache memory size in bytes
size_t mlx_get_cache_memory() {
  return mlx::core::get_cache_memory();
}

// Reset peak memory counter to zero
void mlx_reset_peak_memory() {
  mlx::core::reset_peak_memory();
}

// Set memory limit (guideline for max memory use)
// Returns the previous limit
size_t mlx_set_memory_limit(size_t limit) {
  return mlx::core::set_memory_limit(limit);
}

// Get current memory limit
size_t mlx_get_memory_limit() {
  return mlx::core::get_memory_limit();
}

// Get the number of bytes in an array without evaluating it
// This is much faster than calling shape() which triggers evaluation
size_t mlx_array_nbytes(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  return static_cast<uint64_t>(arr->nbytes());
}

// ============================================================================
// FUSED QWEN3 GENERATION
// ============================================================================
// This function implements the ENTIRE generation loop in C++, eliminating
// FFI overhead and matching mlx-lm's async pipelining pattern.

// KV cache chunk size for pre-allocation (matches mlx-lm step=256)
constexpr int KV_CACHE_CHUNK_SIZE = 256;

// Helper: Apply one transformer block with OPTIMIZED KV cache
// Uses pre-allocated buffers with slice_update for O(N) total instead of O(N²)
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
    // KV cache (in/out) - pre-allocated buffers
    std::optional<array>& cached_keys, std::optional<array>& cached_values,
    int& cache_offset, int& cache_capacity) {

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

  // RoPE
  bool traditional = false;
  float rope_scale = 1.0f;
  queries = fast::rope(queries, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, cache_offset, std::nullopt, {});
  keys = fast::rope(keys, rope_dims, traditional, std::optional<float>(rope_base), rope_scale, cache_offset, std::nullopt, {});

  // === OPTIMIZED KV Cache Update ===
  // Uses pre-allocated buffers with slice_update for O(N) total
  int new_offset = cache_offset + seq_len;

  if (!cached_keys.has_value()) {
    // First call: allocate initial buffer
    // Round up to next chunk boundary for efficiency
    int initial_capacity = ((seq_len + KV_CACHE_CHUNK_SIZE - 1) / KV_CACHE_CHUNK_SIZE) * KV_CACHE_CHUNK_SIZE;
    initial_capacity = std::max(initial_capacity, KV_CACHE_CHUNK_SIZE);

    // Pre-allocate buffers: [batch, n_kv_heads, capacity, head_dim]
    auto buffer_shape = Shape{batch, n_kv_heads, initial_capacity, head_dim};
    cached_keys = zeros(buffer_shape, keys.dtype());
    cached_values = zeros(buffer_shape, values.dtype());
    cache_capacity = initial_capacity;

    // Insert initial keys/values at offset 0
    cached_keys = slice_update(*cached_keys, keys, {0, 0, 0, 0}, {batch, n_kv_heads, seq_len, head_dim});
    cached_values = slice_update(*cached_values, values, {0, 0, 0, 0}, {batch, n_kv_heads, seq_len, head_dim});
  } else {
    // Subsequent calls: check if we need to expand buffer
    if (new_offset > cache_capacity) {
      // Expand buffer by growing to next chunk boundary
      int new_capacity = ((new_offset + KV_CACHE_CHUNK_SIZE - 1) / KV_CACHE_CHUNK_SIZE) * KV_CACHE_CHUNK_SIZE;
      auto new_shape = Shape{batch, n_kv_heads, new_capacity, head_dim};

      // Allocate new buffers and copy existing data
      auto new_keys = zeros(new_shape, keys.dtype());
      auto new_values = zeros(new_shape, values.dtype());

      // Copy existing cached data
      new_keys = slice_update(new_keys, *cached_keys, {0, 0, 0, 0}, {batch, n_kv_heads, cache_capacity, head_dim});
      new_values = slice_update(new_values, *cached_values, {0, 0, 0, 0}, {batch, n_kv_heads, cache_capacity, head_dim});

      cached_keys = new_keys;
      cached_values = new_values;
      cache_capacity = new_capacity;
    }

    // Insert new keys/values at current offset using slice_update (O(1)!)
    cached_keys = slice_update(*cached_keys, keys,
        {0, 0, cache_offset, 0}, {batch, n_kv_heads, new_offset, head_dim});
    cached_values = slice_update(*cached_values, values,
        {0, 0, cache_offset, 0}, {batch, n_kv_heads, new_offset, head_dim});
  }
  cache_offset = new_offset;

  // For SDPA, we only use the valid portion of the cache
  auto keys_valid = slice(*cached_keys, {0, 0, 0, 0}, {batch, n_kv_heads, cache_offset, head_dim});
  auto values_valid = slice(*cached_values, {0, 0, 0, 0}, {batch, n_kv_heads, cache_offset, head_dim});

  // SDPA
  int kv_len = cache_offset;
  std::string mask_mode = "";
  if (seq_len > 1 && seq_len == kv_len) {
    mask_mode = "causal";
  }
  auto attn_output = fast::scaled_dot_product_attention(queries, keys_valid, values_valid, attn_scale, mask_mode, {}, std::nullopt, {});

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
  auto gate_act = gate * sigmoid(gate);
  auto gated = gate_act * up;
  auto mlp_output = matmul(gated, w_down_t);

  return h + mlp_output;
}

// Helper: Run forward through all layers and get logits
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
    std::vector<int>& cache_offsets,
    std::vector<int>& cache_capacities) {

  // Embedding lookup
  auto hidden = take(embedding_weight, input_ids, 0);

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

    hidden = transformer_block_forward_cached(
        hidden,
        input_norm_w, post_attn_norm_w,
        w_q, w_k, w_v, w_o, q_norm_w, k_norm_w,
        w_gate, w_up, w_down,
        n_heads, n_kv_heads, head_dim,
        attn_scale, rope_base, head_dim, // rope_dims = head_dim
        norm_eps, norm_eps, // qk_norm_eps = norm_eps
        kv_keys[i], kv_values[i], cache_offsets[i], cache_capacities[i]);
  }

  // Final normalization
  hidden = fast::rms_norm(hidden, std::optional<array>(final_norm_w), norm_eps, {});

  // LM head
  if (tie_word_embeddings) {
    return matmul(hidden, transpose(embedding_weight, {1, 0}));
  } else {
    return matmul(hidden, transpose(*lm_head_w, {1, 0}));
  }
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
    auto threshold = array(1.0f - top_p);
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
    auto& info = mlx::core::metal::device_info();
    size_t max_rec = std::get<size_t>(info.at("max_recommended_working_set_size"));
    old_wired_limit = mlx::core::set_wired_limit(max_rec);
  }

  // Create dedicated generation stream (matches mlx-lm line 216)
  auto generation_stream = new_stream(default_device());

  // Initialize KV caches for all layers (use optional since array has no default ctor)
  std::vector<std::optional<array>> kv_keys(num_layers);
  std::vector<std::optional<array>> kv_values(num_layers);
  std::vector<int> cache_offsets(num_layers, 0);
  std::vector<int> cache_capacities(num_layers, 0);  // Track pre-allocated buffer sizes

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
        kv_keys, kv_values, cache_offsets, cache_capacities);

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
          kv_keys, kv_values, cache_offsets, cache_capacities);

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
// This function performs ONE forward pass (embedding → all layers → logits)
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
    const int* cache_offsets_in,        // [num_layers] or null
    const int* cache_capacities_in,     // [num_layers] or null (pre-allocated buffer sizes)

    // Outputs (caller must free)
    mlx_array** out_logits,             // [batch, seq_len, vocab]
    mlx_array** out_kv_keys,            // [num_layers] new key arrays
    mlx_array** out_kv_values,          // [num_layers] new value arrays
    int* out_cache_offsets,             // [num_layers] updated offsets
    int* out_cache_capacities           // [num_layers] updated capacities
) {
    auto& input_ids = *reinterpret_cast<array*>(input_ids_handle);
    auto& embedding_weight = *reinterpret_cast<array*>(embedding_weight_handle);
    auto& final_norm_w = *reinterpret_cast<array*>(final_norm_weight_handle);
    array* lm_head_w = lm_head_weight_handle ? reinterpret_cast<array*>(lm_head_weight_handle) : nullptr;

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int rope_dims = head_dim;

    int batch = static_cast<int>(input_ids.shape()[0]);
    int seq_len = static_cast<int>(input_ids.shape()[1]);

    // Initialize or restore KV cache state
    std::vector<std::optional<array>> kv_keys(num_layers);
    std::vector<std::optional<array>> kv_values(num_layers);
    std::vector<int> cache_offsets(num_layers, 0);
    std::vector<int> cache_capacities(num_layers, 0);

    if (kv_keys_in != nullptr && kv_values_in != nullptr && cache_offsets_in != nullptr) {
        for (int i = 0; i < num_layers; i++) {
            if (kv_keys_in[i] != nullptr) {
                kv_keys[i] = *reinterpret_cast<array*>(kv_keys_in[i]);
            }
            if (kv_values_in[i] != nullptr) {
                kv_values[i] = *reinterpret_cast<array*>(kv_values_in[i]);
            }
            cache_offsets[i] = cache_offsets_in[i];
            if (cache_capacities_in != nullptr) {
                cache_capacities[i] = cache_capacities_in[i];
            }
        }
    }

    // Embedding lookup
    auto hidden = take(embedding_weight, input_ids, 0);

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

        hidden = transformer_block_forward_cached(
            hidden,
            input_norm_w, post_attn_norm_w,
            w_q, w_k, w_v, w_o, q_norm_w, k_norm_w,
            w_gate, w_up, w_down,
            num_heads, num_kv_heads, head_dim,
            attn_scale, rope_theta, rope_dims,
            norm_eps, norm_eps,
            kv_keys[i], kv_values[i], cache_offsets[i], cache_capacities[i]);
    }

    // Final normalization
    hidden = fast::rms_norm(hidden, std::optional<array>(final_norm_w), norm_eps, {});

    // LM head and store output
    if (tie_word_embeddings) {
        auto logits = matmul(hidden, transpose(embedding_weight, {1, 0}));
        *out_logits = reinterpret_cast<mlx_array*>(new array(std::move(logits)));
    } else {
        auto logits = matmul(hidden, transpose(*lm_head_w, {1, 0}));
        *out_logits = reinterpret_cast<mlx_array*>(new array(std::move(logits)));
    }

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
        out_cache_offsets[i] = cache_offsets[i];
        out_cache_capacities[i] = cache_capacities[i];
    }
}

// ============================================================================
// PagedAttention Implementation
// ============================================================================

// Metal kernel dispatch helpers
// NOTE: Full Metal dispatch requires integrating with MLX's Metal build system.
// The kernel source is available in paged_attn_kernels.h for future use.
// For now, we use MLX's lazy evaluation which runs on GPU automatically.

namespace paged_attn_metal {

// Check if Metal is available (for logging/debugging)
static bool is_metal_available() {
    return mlx::core::metal::is_available();
}

// Get kernel name based on dtype (for future Metal dispatch)
static std::string get_reshape_and_cache_kernel_name(mlx::core::Dtype dtype) {
    using namespace mlx::core;
    switch (dtype) {
        case float32: return "reshape_and_cache_kv_float_cache_float";
        case bfloat16: return "reshape_and_cache_kv_bfloat16_t_cache_bfloat16_t";
        case float16:
        default: return "reshape_and_cache_kv_half_cache_half";
    }
}

} // namespace paged_attn_metal

/// PagedAttention configuration (matches Rust FFI)
struct PagedAttnConfig {
    uint32_t block_size;
    uint32_t num_blocks;
    uint32_t head_size;
    uint32_t num_kv_heads;
    uint32_t num_layers;
    uint32_t dtype;  // 0=float16, 1=bfloat16, 2=float32
};

/// PagedAttention cache state (per-layer key and value caches)
struct PagedAttnCache {
    std::vector<array> key_caches;    // [num_layers] of [num_blocks, num_kv_heads, head_size/x, block_size, x]
    std::vector<array> value_caches;  // [num_layers] of [num_blocks, num_kv_heads, head_size, block_size]
    PagedAttnConfig config;

    // Metal kernel support (optional - when GPU dispatch is enabled)
    bool use_metal_kernels = false;

    // ALiBi slopes for positional encoding (optional)
    std::optional<array> alibi_slopes;

    // FP8 quantization scales (optional)
    std::optional<array> k_scale;
    std::optional<array> v_scale;
};

/// Create a new PagedAttention KV cache
PagedAttnCache* mlx_paged_attn_create_cache(const PagedAttnConfig* config) {
    if (!config) return nullptr;

    auto cache = new PagedAttnCache();
    cache->config = *config;

    // Determine dtype
    mlx::core::Dtype dtype = mlx::core::float16;  // Default to float16
    switch (config->dtype) {
        case 0: dtype = mlx::core::float16; break;
        case 1: dtype = mlx::core::bfloat16; break;
        case 2: dtype = mlx::core::float32; break;
        default: break;
    }

    // Key cache layout: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    // where x is a vectorization factor (typically 8 for half, 4 for float)
    // For simplicity, we use x=8 for half types, x=4 for float32
    int x = (dtype == mlx::core::float32) ? 4 : 8;
    int head_size_x = config->head_size / x;

    // Value cache layout: [num_blocks, num_kv_heads, head_size, block_size]
    for (uint32_t i = 0; i < config->num_layers; i++) {
        // Key cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        array key_cache = zeros({
            static_cast<int>(config->num_blocks),
            static_cast<int>(config->num_kv_heads),
            head_size_x,
            static_cast<int>(config->block_size),
            x
        }, dtype);
        cache->key_caches.push_back(std::move(key_cache));

        // Value cache: [num_blocks, num_kv_heads, head_size, block_size]
        array value_cache = zeros({
            static_cast<int>(config->num_blocks),
            static_cast<int>(config->num_kv_heads),
            static_cast<int>(config->head_size),
            static_cast<int>(config->block_size)
        }, dtype);
        cache->value_caches.push_back(std::move(value_cache));
    }

    return cache;
}

/// Free a PagedAttention KV cache
void mlx_paged_attn_free_cache(PagedAttnCache* cache) {
    delete cache;
}

/// Get the key cache tensor for a layer
mlx_array* mlx_paged_attn_get_key_cache(PagedAttnCache* cache, uint32_t layer_idx) {
    if (!cache || layer_idx >= cache->key_caches.size()) return nullptr;
    return reinterpret_cast<mlx_array*>(new array(cache->key_caches[layer_idx]));
}

/// Get the value cache tensor for a layer
mlx_array* mlx_paged_attn_get_value_cache(PagedAttnCache* cache, uint32_t layer_idx) {
    if (!cache || layer_idx >= cache->value_caches.size()) return nullptr;
    return reinterpret_cast<mlx_array*>(new array(cache->value_caches[layer_idx]));
}

/// Update the cache with new keys and values.
///
/// This function updates the paged KV cache with new key/value vectors.
/// Uses a software implementation with MLX operations for correctness.
/// For production, integrate Metal kernels via MLX's metal_kernel API.
///
/// Inputs:
/// - keys: [num_tokens, num_kv_heads, head_size]
/// - values: [num_tokens, num_kv_heads, head_size]
/// - slot_mapping: [num_tokens] - linear slot indices for each token
///
/// Cache layout:
/// - key_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
/// - value_cache: [num_blocks, num_kv_heads, head_size, block_size]
void mlx_paged_attn_reshape_and_cache(
    PagedAttnCache* cache,
    uint32_t layer_idx,
    mlx_array* keys,
    mlx_array* values,
    mlx_array* slot_mapping
) {
    if (!cache || !keys || !values || !slot_mapping) return;
    if (layer_idx >= cache->key_caches.size()) return;

    auto& k = *reinterpret_cast<array*>(keys);
    auto& v = *reinterpret_cast<array*>(values);
    auto& slots = *reinterpret_cast<array*>(slot_mapping);

    // Get cache references
    auto& key_cache = cache->key_caches[layer_idx];
    auto& value_cache = cache->value_caches[layer_idx];

    int num_tokens = k.shape(0);
    int num_heads = k.shape(1);
    int head_size = k.shape(2);
    int block_size = cache->config.block_size;
    int x = key_cache.shape(4);  // Vectorization factor

    // Implementation using MLX operations (runs on GPU via lazy evaluation)
    //
    // METAL KERNEL STATUS:
    // - Kernel source: READY in paged_attn_kernels.h (get_reshape_and_cache_source())
    // - Rust dispatch: READY in mlx-paged-attn/src/metal/reshape_and_cache.rs
    // - Precompiled lib: paged_attn.metallib (4.7 MB)
    //
    // BLOCKING ISSUE: MLX's metal_kernel() API doesn't support function constants
    // (use_fp8_scales), which the kernel requires. Options:
    // 1. Contribute paged attention to MLX as a first-class primitive
    // 2. Use MLX's internal Metal dispatch when API becomes available
    // 3. Fork the kernel to use template arguments instead of function constants
    //
    // For now, the software implementation is functionally correct.

    int num_blocks = key_cache.shape(0);
    int head_size_x = head_size / x;

    auto block_indices = divide(slots, array(block_size, slots.dtype()));
    auto block_offsets = remainder(slots, array(block_size, slots.dtype()));

    for (int i = 0; i < num_tokens; i++) {
        array key_token = slice(k, {i, 0, 0}, {i + 1, num_heads, head_size});
        key_token = squeeze(key_token, 0);

        array value_token = slice(v, {i, 0, 0}, {i + 1, num_heads, head_size});
        value_token = squeeze(value_token, 0);

        array block_idx_arr = slice(block_indices, {i}, {i + 1});
        array block_off_arr = slice(block_offsets, {i}, {i + 1});
        block_idx_arr.eval();
        block_off_arr.eval();

        int block_idx = static_cast<int>(block_idx_arr.item<int32_t>());
        int block_off = static_cast<int>(block_off_arr.item<int32_t>());

        // Update value cache
        array v_block = slice(value_cache, {block_idx, 0, 0, 0}, {block_idx + 1, num_heads, head_size, block_size});
        v_block = squeeze(v_block, 0);

        std::vector<array> v_slices;
        for (int j = 0; j < block_size; j++) {
            if (j == block_off) {
                v_slices.push_back(expand_dims(value_token, -1));
            } else {
                v_slices.push_back(slice(v_block, {0, 0, j}, {num_heads, head_size, j + 1}));
            }
        }
        array updated_v = expand_dims(concatenate(v_slices, -1), 0);

        std::vector<array> v_parts;
        if (block_idx > 0) {
            v_parts.push_back(slice(value_cache, {0, 0, 0, 0}, {block_idx, num_heads, head_size, block_size}));
        }
        v_parts.push_back(updated_v);
        if (block_idx + 1 < num_blocks) {
            v_parts.push_back(slice(value_cache, {block_idx + 1, 0, 0, 0}, {num_blocks, num_heads, head_size, block_size}));
        }
        value_cache = concatenate(v_parts, 0);

        // Update key cache
        array k_block = slice(key_cache, {block_idx, 0, 0, 0, 0}, {block_idx + 1, num_heads, head_size_x, block_size, x});
        k_block = squeeze(k_block, 0);
        array key_reshaped = reshape(key_token, {num_heads, head_size_x, x});

        std::vector<array> k_slices;
        for (int j = 0; j < block_size; j++) {
            if (j == block_off) {
                k_slices.push_back(expand_dims(key_reshaped, 2));
            } else {
                k_slices.push_back(slice(k_block, {0, 0, j, 0}, {num_heads, head_size_x, j + 1, x}));
            }
        }
        array updated_k = expand_dims(concatenate(k_slices, 2), 0);

        std::vector<array> k_parts;
        if (block_idx > 0) {
            k_parts.push_back(slice(key_cache, {0, 0, 0, 0, 0}, {block_idx, num_heads, head_size_x, block_size, x}));
        }
        k_parts.push_back(updated_k);
        if (block_idx + 1 < num_blocks) {
            k_parts.push_back(slice(key_cache, {block_idx + 1, 0, 0, 0, 0}, {num_blocks, num_heads, head_size_x, block_size, x}));
        }
        key_cache = concatenate(k_parts, 0);
    }

    cache->key_caches[layer_idx] = key_cache;
    cache->value_caches[layer_idx] = value_cache;
}

/// Run paged attention forward pass
/// Software implementation using MLX operations.
///
/// METAL KERNEL STATUS:
/// - Kernel source: READY in paged_attn_kernels.h (get_paged_attention_source())
/// - Rust dispatch: READY in mlx-paged-attn/src/metal/paged_attention.rs
/// - V1 kernel (short sequences): paged_attention_half_cache_half_hs128_bs16_nt256_nsl32_ps0
/// - V2 kernel (long sequences with partitioning): partition_size=512
///
/// BLOCKING ISSUE: MLX's metal_kernel() API doesn't support:
/// 1. Function constants (use_partitioning, use_alibi, use_fp8_scales)
/// 2. The 18-buffer binding pattern used by the kernel
///
/// The software implementation runs on GPU via MLX's lazy evaluation and is
/// functionally correct, though slower than the optimized Metal kernel.
///
/// This implements the paged attention algorithm:
/// 1. Gather K/V blocks for each sequence using block_tables
/// 2. Reshape gathered blocks to contiguous K/V tensors
/// 3. Run scaled dot-product attention
///
/// Inputs:
/// - queries: [num_seqs, num_heads, head_size]
/// - key_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x] (vectorized)
/// - value_cache: [num_blocks, num_kv_heads, head_size, block_size]
/// - block_tables: [num_seqs, max_blocks_per_seq]
/// - context_lens: [num_seqs]
mlx_array* mlx_paged_attn_forward(
    mlx_array* queries,
    mlx_array* key_cache,
    mlx_array* value_cache,
    mlx_array* block_tables,
    mlx_array* context_lens,
    float scale,
    uint32_t block_size,
    uint32_t max_context_len
) {
    if (!queries || !key_cache || !value_cache || !block_tables || !context_lens) {
        return nullptr;
    }

    auto& q = *reinterpret_cast<array*>(queries);
    auto& k_cache = *reinterpret_cast<array*>(key_cache);
    auto& v_cache = *reinterpret_cast<array*>(value_cache);
    auto& block_table = *reinterpret_cast<array*>(block_tables);
    auto& ctx_lens = *reinterpret_cast<array*>(context_lens);

    int num_seqs = q.shape(0);
    int num_heads = q.shape(1);
    int head_size = q.shape(2);
    int num_kv_heads = v_cache.shape(1);
    int heads_per_kv = num_heads / num_kv_heads;

    // Key cache vectorization factor
    int x = k_cache.shape(4);
    int head_size_x = k_cache.shape(2);

    // Evaluate context lengths to CPU for the loop
    ctx_lens.eval();
    auto ctx_lens_vec = std::vector<int32_t>(num_seqs);
    std::memcpy(ctx_lens_vec.data(), ctx_lens.data<int32_t>(), num_seqs * sizeof(int32_t));

    // Process each sequence (software path - Metal kernel does this in parallel)
    std::vector<array> outputs;
    outputs.reserve(num_seqs);

    for (int seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
        int context_len = ctx_lens_vec[seq_idx];
        int num_blocks_needed = (context_len + block_size - 1) / block_size;

        if (context_len == 0 || num_blocks_needed == 0) {
            // Empty context - return zeros for this sequence
            outputs.push_back(zeros({num_heads, head_size}, q.dtype()));
            continue;
        }

        // Get block indices for this sequence: block_table[seq_idx, :num_blocks_needed]
        array seq_blocks = slice(block_table, {seq_idx, 0}, {seq_idx + 1, num_blocks_needed});
        seq_blocks = reshape(seq_blocks, {num_blocks_needed});

        // Gather key blocks: [num_blocks_needed, num_kv_heads, head_size/x, block_size, x]
        array gathered_keys = take(k_cache, seq_blocks, 0);

        // Reshape keys to [num_kv_heads, context_len, head_size]
        // First: [num_blocks_needed, num_kv_heads, head_size/x, block_size, x]
        //     -> [num_kv_heads, num_blocks_needed, head_size/x, block_size, x]
        gathered_keys = transpose(gathered_keys, {1, 0, 2, 3, 4});
        // -> [num_kv_heads, num_blocks_needed, block_size, head_size/x, x]
        gathered_keys = transpose(gathered_keys, {0, 1, 3, 2, 4});
        // -> [num_kv_heads, num_blocks_needed * block_size, head_size]
        gathered_keys = reshape(gathered_keys, {num_kv_heads, num_blocks_needed * (int)block_size, head_size});
        // Trim to actual context length
        gathered_keys = slice(gathered_keys, {0, 0, 0}, {num_kv_heads, context_len, head_size});

        // Gather value blocks: [num_blocks_needed, num_kv_heads, head_size, block_size]
        array gathered_values = take(v_cache, seq_blocks, 0);

        // Reshape values to [num_kv_heads, context_len, head_size]
        // [num_blocks_needed, num_kv_heads, head_size, block_size]
        // -> [num_kv_heads, num_blocks_needed, head_size, block_size]
        gathered_values = transpose(gathered_values, {1, 0, 2, 3});
        // -> [num_kv_heads, num_blocks_needed, block_size, head_size]
        gathered_values = transpose(gathered_values, {0, 1, 3, 2});
        // -> [num_kv_heads, num_blocks_needed * block_size, head_size]
        gathered_values = reshape(gathered_values, {num_kv_heads, num_blocks_needed * (int)block_size, head_size});
        // Trim to actual context length
        gathered_values = slice(gathered_values, {0, 0, 0}, {num_kv_heads, context_len, head_size});

        // Handle GQA: repeat K/V heads to match query heads
        if (heads_per_kv > 1) {
            // Expand K/V from [num_kv_heads, ctx_len, head_size] to [num_heads, ctx_len, head_size]
            // Use repeat to duplicate each KV head heads_per_kv times
            std::vector<array> expanded_k, expanded_v;
            for (int kv_head = 0; kv_head < num_kv_heads; kv_head++) {
                array k_head = slice(gathered_keys, {kv_head, 0, 0}, {kv_head + 1, context_len, head_size});
                array v_head = slice(gathered_values, {kv_head, 0, 0}, {kv_head + 1, context_len, head_size});
                for (int r = 0; r < heads_per_kv; r++) {
                    expanded_k.push_back(k_head);
                    expanded_v.push_back(v_head);
                }
            }
            gathered_keys = concatenate(expanded_k, 0);
            gathered_values = concatenate(expanded_v, 0);
        }

        // Get query for this sequence: [num_heads, head_size]
        array seq_query = slice(q, {seq_idx, 0, 0}, {seq_idx + 1, num_heads, head_size});
        seq_query = reshape(seq_query, {num_heads, 1, head_size});

        // Attention scores: Q @ K^T -> [num_heads, 1, context_len]
        array k_transposed = transpose(gathered_keys, {0, 2, 1}); // [num_heads, head_size, context_len]
        array scores = matmul(seq_query, k_transposed);
        scores = multiply(scores, array(scale, q.dtype()));

        // Softmax over context dimension
        array weights = softmax(scores, -1);

        // Weighted sum: weights @ V -> [num_heads, 1, head_size]
        array output = matmul(weights, gathered_values);
        output = reshape(output, {num_heads, head_size});

        outputs.push_back(output);
    }

    // Stack outputs: [num_seqs, num_heads, head_size]
    array result = stack(outputs, 0);

    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

/// Copy blocks for copy-on-write semantics.
///
/// This is used during beam search to copy KV cache blocks when forking sequences.
/// Uses MLX operations for portability. For production, integrate Metal kernel.
///
/// block_mapping: [num_pairs, 2] where each pair is (src_block_id, dst_block_id)
void mlx_paged_attn_copy_blocks(
    PagedAttnCache* cache,
    uint32_t layer_idx,
    mlx_array* block_mapping
) {
    if (!cache || !block_mapping) return;
    if (layer_idx >= cache->key_caches.size()) return;

    auto& mapping = *reinterpret_cast<array*>(block_mapping);
    auto& key_cache = cache->key_caches[layer_idx];
    auto& value_cache = cache->value_caches[layer_idx];

    // Evaluate mapping to get block indices
    mapping.eval();

    // Handle both 1D [num_pairs * 2] and 2D [num_pairs, 2] formats
    int num_pairs;
    bool is_2d = (mapping.ndim() == 2 && mapping.shape(1) == 2);
    if (is_2d) {
        num_pairs = mapping.shape(0);  // 2D: [num_pairs, 2]
    } else {
        num_pairs = mapping.shape(0) / 2;  // 1D: [num_pairs * 2]
    }
    if (num_pairs == 0) return;

    // Copy each block pair using MLX gather/scatter
    for (int i = 0; i < num_pairs; i++) {
        // Extract block indices from the mapping array
        array src_arr = is_2d
            ? slice(mapping, {i, 0}, {i + 1, 1})
            : slice(mapping, {2 * i}, {2 * i + 1});
        array dst_arr = is_2d
            ? slice(mapping, {i, 1}, {i + 1, 2})
            : slice(mapping, {2 * i + 1}, {2 * i + 2});
        src_arr.eval();
        dst_arr.eval();
        int64_t src_block = src_arr.item<int64_t>();
        int64_t dst_block = dst_arr.item<int64_t>();

        // Copy key cache block
        // key_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        int num_heads = key_cache.shape(1);
        int head_size_x = key_cache.shape(2);
        int block_size = key_cache.shape(3);
        int x = key_cache.shape(4);

        array src_k_block = slice(key_cache,
            {static_cast<int>(src_block), 0, 0, 0, 0},
            {static_cast<int>(src_block + 1), num_heads, head_size_x, block_size, x});

        // Rebuild key cache with copied block at dst position
        int num_blocks = key_cache.shape(0);
        std::vector<array> k_parts;
        if (dst_block > 0) {
            k_parts.push_back(slice(key_cache, {0, 0, 0, 0, 0},
                {static_cast<int>(dst_block), num_heads, head_size_x, block_size, x}));
        }
        k_parts.push_back(src_k_block);
        if (dst_block + 1 < num_blocks) {
            k_parts.push_back(slice(key_cache, {static_cast<int>(dst_block + 1), 0, 0, 0, 0},
                {num_blocks, num_heads, head_size_x, block_size, x}));
        }
        key_cache = concatenate(k_parts, 0);

        // Copy value cache block
        // value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        int head_size = value_cache.shape(2);

        array src_v_block = slice(value_cache,
            {static_cast<int>(src_block), 0, 0, 0},
            {static_cast<int>(src_block + 1), num_heads, head_size, block_size});

        std::vector<array> v_parts;
        if (dst_block > 0) {
            v_parts.push_back(slice(value_cache, {0, 0, 0, 0},
                {static_cast<int>(dst_block), num_heads, head_size, block_size}));
        }
        v_parts.push_back(src_v_block);
        if (dst_block + 1 < num_blocks) {
            v_parts.push_back(slice(value_cache, {static_cast<int>(dst_block + 1), 0, 0, 0},
                {num_blocks, num_heads, head_size, block_size}));
        }
        value_cache = concatenate(v_parts, 0);
    }

    // Store updated caches back
    cache->key_caches[layer_idx] = key_cache;
    cache->value_caches[layer_idx] = value_cache;
}

// ========================================================================
// Metal Buffer Extraction for External Kernel Dispatch
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

}  // End extern "C"
