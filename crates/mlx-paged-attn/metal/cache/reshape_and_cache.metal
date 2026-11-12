#include "../utils.metal"
#include <metal_stdlib>

using namespace metal;

template <typename KV_T, typename CACHE_T>
inline CACHE_T to_cache(KV_T v) = delete;

template <> inline float to_cache<float, float>(float v) { return v; }

template <> inline bfloat16_t to_cache<bfloat16_t, bfloat16_t>(bfloat16_t v) {
  return v;
}

template <> inline half to_cache<half, half>(half v) { return v; }

// ValuesTokenMajor (the `_vt` instance): the caller passes x = head_size,
// which makes K token-major, and V takes K's index: both
// [num_blocks, num_heads, block_size, head_size] (Splash
// `splash_kv_key_element`; `splash_kv_value_index` under `ValuesTokenMajor`).
template <typename KV_T, typename CACHE_T, bool ValuesTokenMajor>
[[kernel]] void reshape_and_cache(
    const device KV_T *__restrict__ key
    [[buffer(0)]], // [num_tokens, num_heads, head_size]
    const device KV_T *__restrict__ value
    [[buffer(1)]], // [num_tokens, num_heads, head_size]
    device CACHE_T *__restrict__ key_cache
    [[buffer(2)]], // [num_blocks, num_heads, head_size/x, block_size, x]
    device CACHE_T *__restrict__ value_cache
    [[buffer(3)]], // [num_blocks, num_heads, head_size, block_size]
    const device int64_t *__restrict__ slot_mapping
    [[buffer(4)]], // [num_tokens]
    device const int &key_stride [[buffer(5)]],
    device const int &value_stride [[buffer(6)]],
    device const int &num_heads [[buffer(7)]],
    device const int &head_size [[buffer(8)]],
    device const int &block_size [[buffer(9)]],
    device const int &x [[buffer(10)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  const int64_t token_idx = gid;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = tid; i < n; i += threads_per_threadgroup) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int64_t tgt_value_idx =
        ValuesTokenMajor
            ? tgt_key_idx
            : block_idx * num_heads * head_size * block_size +
                  head_idx * head_size * block_size +
                  head_offset * block_size + block_offset;

    key_cache[tgt_key_idx] = to_cache<KV_T, CACHE_T>(key[src_key_idx]);
    value_cache[tgt_value_idx] = to_cache<KV_T, CACHE_T>(value[src_value_idx]);
  }
}

#define instantiate_reshape_and_cache(kv_type, cache_type, vt, suffix)        \
  template [[host_name("reshape_and_cache_kv_" #kv_type                        \
                       "_cache_" #cache_type suffix)]] [[kernel]] void         \
  reshape_and_cache<kv_type, cache_type, vt>(                                  \
      const device kv_type *__restrict__ key [[buffer(0)]],                    \
      const device kv_type *__restrict__ value [[buffer(1)]],                  \
      device cache_type *__restrict__ key_cache [[buffer(2)]],                 \
      device cache_type *__restrict__ value_cache [[buffer(3)]],               \
      const device int64_t *__restrict__ slot_mapping [[buffer(4)]],           \
      device const int &key_stride [[buffer(5)]],                              \
      device const int &value_stride [[buffer(6)]],                            \
      device const int &num_heads [[buffer(7)]],                               \
      device const int &head_size [[buffer(8)]],                               \
      device const int &block_size [[buffer(9)]],                              \
      device const int &x [[buffer(10)]],                                      \
      uint gid [[threadgroup_position_in_grid]],                               \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint threads_per_threadgroup [[threads_per_threadgroup]]);

instantiate_reshape_and_cache(float, float, false, "");
instantiate_reshape_and_cache(bfloat16_t, bfloat16_t, false, "");
instantiate_reshape_and_cache(half, half, false, "");
instantiate_reshape_and_cache(bfloat16_t, bfloat16_t, true, "_vt");
