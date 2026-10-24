#pragma once
// Splash split_reduce.h (Apache-2.0), already used by the staged kernels.
// Shared without arithmetic changes by SG8 and M8 NAX split kernels.
#include <metal_stdlib>
using namespace metal;
namespace kq_split {
// True in every thread of the threadgroup that arrives last at `counter`.
METAL_FUNC bool split_arrive_last(
    device atomic_uint* counter,
    uint splits,
    uint thread_index,
    threadgroup uint* arrival) {
  threadgroup_barrier(mem_flags::mem_device);
  if (thread_index == 0) {
    atomic_thread_fence(
        mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    atomic_thread_fence(
        mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  return *arrival == splits - 1;
}

METAL_FUNC void split_release(device atomic_uint* counter, uint thread_index) {
  if (thread_index == 0) {
    atomic_store_explicit(counter, 0u, memory_order_relaxed);
  }
}

} // namespace kq_split
