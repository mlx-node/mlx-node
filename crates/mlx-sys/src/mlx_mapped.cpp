// Arrays whose storage is a caller-owned mapping (a read-only mmap of a
// weight pack). The buffer is `newBufferWithBytesNoCopy` over the mapping;
// MLX never copies the bytes. `array(void*, ...)` is NOT used here: it falls
// back to malloc + memcpy when the driver refuses the no-copy buffer, which
// would silently turn a 72 GiB file-backed pack into anonymous memory.

#include "mlx_common.h"

#include <cstdio>

extern "C" {

// Returns null (and never an array over copied bytes) when the driver refuses
// a no-copy buffer over `ptr` or hands back one whose contents pointer is not
// `ptr`. `release_fn(ctx)` runs once, after MLX released the buffer, when the
// last array sharing the storage dies; on a null return it does not run.
mlx_array* mlx_array_from_mapped(void* ptr,
                                 size_t bytes,
                                 const int64_t* shape,
                                 size_t ndim,
                                 int32_t dtype_code,
                                 void (*release_fn)(void*),
                                 void* ctx) {
  try {
    if (ptr == nullptr || bytes == 0) return nullptr;
    Shape target_shape = make_shape(shape, ndim);
    auto dtype = to_mlx_dtype(dtype_code);
    size_t numel = 1;
    for (auto d : target_shape) {
      if (d < 0) return nullptr;
      numel *= static_cast<size_t>(d);
    }
    if (numel * dtype.size() != bytes) return nullptr;

    // Armed only once the array owns the buffer: a throw while building it
    // still runs this deleter, and the caller releases ctx on a null return.
    auto armed = std::make_shared<bool>(false);
    mlx::core::allocator::Deleter deleter =
        [release_fn, ctx, armed](mlx::core::allocator::Buffer b) {
          mlx::core::allocator::release(b);
          if (*armed && release_fn) release_fn(ctx);
        };
    auto buffer = mlx::core::allocator::make_buffer(ptr, bytes);
    if (buffer.ptr() == nullptr) return nullptr;
    if (buffer.raw_ptr() != ptr) {
      mlx::core::allocator::release(buffer);
      return nullptr;
    }
    mlx::core::allocator::Data data(buffer, std::move(deleter));
    auto arr = new array(std::move(data), std::move(target_shape), dtype);
    *armed = true;
    return reinterpret_cast<mlx_array*>(arr);
  } catch (const std::exception& e) {
    std::cerr << "[MLX] array_from_mapped: " << e.what() << std::endl;
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

// Storage pointer of an array that already holds data (a mapped array, or any
// evaluated array); null for an unevaluated array.
const void* mlx_array_data_ptr(mlx_array* handle) {
  if (handle == nullptr) return nullptr;
  auto* arr = reinterpret_cast<array*>(handle);
  if (!arr->is_available() || arr->buffer().ptr() == nullptr) return nullptr;
  return arr->data<void>();
}

}  // extern "C"
