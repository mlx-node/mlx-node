// Metal-specific array allocation, zero-copy mmap views, residency sets, and
// purgeable-state helpers.

#include "mlx_common.h"

#ifdef MLX_NODE_METAL_ENABLED
#include "mlx/backend/metal/device.h"
#include <mach/vm_page_size.h>
#endif

#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

#ifdef MLX_NODE_METAL_ENABLED
using mlx::core::array;
using mlx::core::Device;

constexpr uint64_t kMaxBytesPerResidencySet = 10ULL << 30;

struct MmapResidencySets {
  std::vector<MTL::ResidencySet*> sets;

  ~MmapResidencySets() {
    if (sets.empty()) return;
    if (__builtin_available(macOS 15, iOS 18, *)) {
      auto pool = mlx::core::metal::new_scoped_memory_pool();
      for (MTL::ResidencySet* rset : sets) {
        if (!rset) continue;
        rset->endResidency();
        rset->removeAllAllocations();
        rset->commit();
        rset->release();
      }
    }
    sets.clear();
  }
};

bool known_dtype_code(int32_t code) {
  return code >= 0 && code <= 6;
}

mlx_array* wrap(array a) {
  return reinterpret_cast<mlx_array*>(new array(std::move(a)));
}
#endif

}  // namespace

extern "C" {

// A native Metal allocation with a CPU-visible pointer and a lease invoked
// after the last array sharing it dies. Null on unsupported builds or failure;
// `release_fn(ctx)` is not invoked on a null return.
mlx_array* mlx_array_alloc_purgeable(const int64_t* shape,
                                     size_t ndim,
                                     int32_t dtype_code,
                                     void** out_data,
                                     void (*release_fn)(void*),
                                     void* ctx) {
  if (out_data) *out_data = nullptr;
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!out_data || (ndim > 0 && !shape) || !known_dtype_code(dtype_code)) {
      return nullptr;
    }
    auto dtype = to_mlx_dtype(dtype_code);
    mlx::core::Shape target;
    target.reserve(ndim);
    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
      if (shape[i] <= 0 ||
          shape[i] > std::numeric_limits<mlx::core::ShapeElem>::max()) {
        return nullptr;
      }
      const size_t d = static_cast<size_t>(shape[i]);
      if (numel > std::numeric_limits<size_t>::max() / d) return nullptr;
      numel *= d;
      target.push_back(static_cast<mlx::core::ShapeElem>(shape[i]));
    }
    if (numel > std::numeric_limits<size_t>::max() / dtype.size()) {
      return nullptr;
    }
    const size_t bytes = numel * dtype.size();
    auto armed = std::make_shared<bool>(false);
    mlx::core::allocator::Deleter deleter =
        [release_fn, ctx, armed](mlx::core::allocator::Buffer b) {
          if (b.ptr()) {
            static_cast<MTL::Buffer*>(b.ptr())->setPurgeableState(
                MTL::PurgeableStateNonVolatile);
          }
          mlx::core::allocator::allocator().free(b);
          if (*armed && release_fn) release_fn(ctx);
        };
    auto buffer = mlx::core::allocator::allocator().malloc(bytes);
    if (buffer.ptr() == nullptr) return nullptr;
    mlx::core::allocator::Data data_owner(buffer, std::move(deleter));
    array a(std::move(data_owner), std::move(target), dtype);
    void* data = a.data<void>();
    if (data == nullptr) return nullptr;
    *out_data = data;
    auto handle = wrap(std::move(a));
    *armed = true;
    return handle;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[MLX] array_alloc_purgeable: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
#else
  (void)shape;
  (void)ndim;
  (void)dtype_code;
  (void)release_fn;
  (void)ctx;
  return nullptr;
#endif
}

// Zero-copy read-only uint8 [n_pages, page_size] array over a page-aligned
// PROT_READ MAP_SHARED mmap slice. Bypasses MetalAllocator::make_buffer so
// active_memory_ and residency_sets_ stay untouched (0 B). Returns nullptr on
// invalid alignment, length > maxBufferLength, non-Metal builds, or allocation
// failure; `release_fn(ctx)` is never invoked on a nullptr return.
mlx_array* mlx_array_from_mmap_readonly(void* aligned_ptr,
                                        size_t aligned_len,
                                        void (*release_fn)(void*),
                                        void* ctx) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!aligned_ptr || aligned_len == 0 || !mlx::core::metal::is_available()) {
      return nullptr;
    }
    const size_t page = static_cast<size_t>(vm_page_size);
    if (page == 0 || (reinterpret_cast<uintptr_t>(aligned_ptr) % page) != 0 ||
        (aligned_len % page) != 0) {
      return nullptr;
    }
    const size_t n_pages = aligned_len / page;
    if (n_pages > static_cast<size_t>(
                      std::numeric_limits<mlx::core::ShapeElem>::max()) ||
        page > static_cast<size_t>(
                   std::numeric_limits<mlx::core::ShapeElem>::max())) {
      return nullptr;
    }
    auto* mtl_dev = mlx::core::metal::device(Device::gpu).mtl_device();
    if (!mtl_dev ||
        aligned_len > static_cast<size_t>(mtl_dev->maxBufferLength())) {
      return nullptr;
    }
    constexpr MTL::ResourceOptions opts =
        MTL::ResourceStorageModeShared |
        MTL::ResourceHazardTrackingModeUntracked;
    MTL::Buffer* mtl_buf = nullptr;
    {
      auto pool = mlx::core::metal::new_scoped_memory_pool();
      mtl_buf = mtl_dev->newBuffer(aligned_ptr, aligned_len, opts, nullptr);
    }
    if (!mtl_buf) return nullptr;

    struct MtlBufGuard {
      MTL::Buffer* buf = nullptr;
      ~MtlBufGuard() {
        if (buf) {
          auto pool = mlx::core::metal::new_scoped_memory_pool();
          buf->release();
        }
      }
    } guard{mtl_buf};

    auto armed = std::make_shared<bool>(false);
    mlx::core::allocator::Deleter deleter =
        [mtl_buf, release_fn, ctx, armed](mlx::core::allocator::Buffer) {
          {
            auto pool = mlx::core::metal::new_scoped_memory_pool();
            mtl_buf->release();
          }
          if (*armed && release_fn) release_fn(ctx);
        };
    mlx::core::allocator::Data data_owner(
        mlx::core::allocator::Buffer{static_cast<void*>(mtl_buf)},
        std::move(deleter));
    guard.buf = nullptr;
    mlx::core::Shape shape{static_cast<mlx::core::ShapeElem>(n_pages),
                           static_cast<mlx::core::ShapeElem>(page)};
    array a(std::move(data_owner), std::move(shape), mlx::core::uint8);
    if (a.data<void>() == nullptr) return nullptr;
    auto* handle = wrap(std::move(a));
    *armed = true;
    return handle;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[MLX] array_from_mmap_readonly: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
#else
  (void)aligned_ptr;
  (void)aligned_len;
  (void)release_fn;
  (void)ctx;
  return nullptr;
#endif
}

// Standing Metal residency sets (capped at 10 GiB each so macOS does not drop
// one monolithic mapping under memory pressure) over existing array buffers.
// Caller must destroy the handle before releasing the backing buffers.
void* mlx_metal_residency_set_create(mlx_array* const* arrays, size_t count) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!arrays || count == 0 || !mlx::core::metal::is_available()) {
      return nullptr;
    }
    if (__builtin_available(macOS 15, iOS 18, *)) {
      auto* mtl_dev = mlx::core::metal::device(Device::gpu).mtl_device();
      if (!mtl_dev || !mtl_dev->supportsFamily(MTL::GPUFamilyMetal3)) {
        return nullptr;
      }
      std::vector<std::vector<MTL::Allocation*>> buckets;
      uint64_t set_bytes = 0;
      for (size_t i = 0; i < count; ++i) {
        if (!arrays[i]) continue;
        auto* arr = reinterpret_cast<array*>(arrays[i]);
        void* buf_ptr = arr->buffer().ptr();
        if (!buf_ptr) continue;
        auto* mtl_buf = static_cast<MTL::Buffer*>(buf_ptr);
        const uint64_t alloc_bytes = static_cast<uint64_t>(mtl_buf->length());
        if (buckets.empty() ||
            (set_bytes > 0 &&
             (alloc_bytes > kMaxBytesPerResidencySet ||
              set_bytes > kMaxBytesPerResidencySet - alloc_bytes))) {
          buckets.emplace_back();
          set_bytes = 0;
        }
        buckets.back().push_back(static_cast<MTL::Allocation*>(buf_ptr));
        set_bytes += alloc_bytes;
      }
      if (buckets.empty()) {
        return nullptr;
      }
      auto pool = mlx::core::metal::new_scoped_memory_pool();
      auto holder = std::make_unique<MmapResidencySets>();
      holder->sets.reserve(buckets.size());
      for (const auto& bucket : buckets) {
        auto* desc = MTL::ResidencySetDescriptor::alloc()->init()->autorelease();
        desc->setInitialCapacity(static_cast<NS::UInteger>(bucket.size()));
        NS::Error* error = nullptr;
        MTL::ResidencySet* rset = mtl_dev->newResidencySet(desc, &error);
        if (!rset) {
          if (error && error->localizedDescription()) {
            std::fprintf(stderr, "[MLX] metal_residency_set_create: %s\n",
                         error->localizedDescription()->utf8String());
          }
          return nullptr;
        }
        for (MTL::Allocation* alloc : bucket) {
          rset->addAllocation(alloc);
        }
        rset->commit();
        rset->requestResidency();
        holder->sets.push_back(rset);
      }
      return static_cast<void*>(holder.release());
    }
    return nullptr;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[MLX] metal_residency_set_create: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
#else
  (void)arrays;
  (void)count;
  return nullptr;
#endif
}

void mlx_metal_residency_set_request(void* handle) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!handle) return;
    if (__builtin_available(macOS 15, iOS 18, *)) {
      auto pool = mlx::core::metal::new_scoped_memory_pool();
      auto* holder = static_cast<MmapResidencySets*>(handle);
      for (MTL::ResidencySet* rset : holder->sets) {
        if (rset) rset->requestResidency();
      }
    }
  } catch (...) {
  }
#else
  (void)handle;
#endif
}

void mlx_metal_residency_set_destroy(void* handle) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!handle) return;
    delete static_cast<MmapResidencySets*>(handle);
  } catch (...) {
  }
#else
  (void)handle;
#endif
}

// Set purgeability on the array's existing Metal allocation. Returns the
// previous state (2 non-volatile, 3 volatile, 4 empty), or -1 on unsupported
// builds and invalid handles/states. Never evaluates or replaces the array.
int32_t mlx_array_set_purgeable_state(mlx_array* handle, int32_t state) {
#ifdef MLX_NODE_METAL_ENABLED
  try {
    if (!handle || state < MTL::PurgeableStateNonVolatile ||
        state > MTL::PurgeableStateEmpty) {
      return -1;
    }
    auto& a = *reinterpret_cast<array*>(handle);
    if (a.buffer().ptr() == nullptr) return -1;
    auto* buffer = static_cast<MTL::Buffer*>(a.buffer().ptr());
    return static_cast<int32_t>(buffer->setPurgeableState(
        static_cast<MTL::PurgeableState>(state)));
  } catch (...) {
    return -1;
  }
#else
  (void)handle;
  (void)state;
  return -1;
#endif
}

}  // extern "C"
