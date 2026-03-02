#pragma once

#include "mlx/mlx.h"
#include "mlx/fast.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"
#include "mlx/memory.h"
#include "mlx/compile.h"
#include "mlx/compile_impl.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/gpu/device_info.h"

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
using mlx::core::add;
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
  UINT32 = 4,
  UINT8 = 5,
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
    case UINT32:
      return mlx::core::uint32;
    case UINT8:
      return mlx::core::uint8;
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
    case mlx::core::uint32:
      return UINT32;
    case mlx::core::uint8:
      return UINT8;
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
