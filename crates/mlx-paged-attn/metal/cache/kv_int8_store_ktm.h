// Int8 paged store for the keys-token-major page (Splash dense
// `kv::Layout`, `splash_store_kv_row` in `paged_store_row.h`; Splash 1.6.1,
// Apache-2.0): K rows token-major as the `_vt` instances write them
// (`splash_kv_key_element`), V rows dimension-major as the default instances
// write them (`splash_kv_value_element`). Math and dispatch are those of
// `reshape_and_cache_int8`. Included at the end of kv_int8_store.metal so it
// keeps that file's `-fno-fast-math`.
//
//   key_cache     [num_blocks, num_heads, block_size, HeadDim] char
//   value_cache   [num_blocks, num_heads, HeadDim, block_size] char
//   key_scales    [num_blocks, num_heads, block_size] f32
//   value_scales  (Splash `splash_q8_scale_index`)
template <uint HeadDim>
[[kernel]] void reshape_and_cache_int8_ktm(
    const device bfloat16_t *__restrict__ key [[buffer(0)]],
    const device bfloat16_t *__restrict__ value [[buffer(1)]],
    device char *__restrict__ key_cache [[buffer(2)]],
    device char *__restrict__ value_cache [[buffer(3)]],
    const device int64_t *__restrict__ slot_mapping [[buffer(4)]],
    device float *__restrict__ key_scales [[buffer(5)]],
    device float *__restrict__ value_scales [[buffer(6)]],
    constant int &key_stride [[buffer(7)]],
    constant int &value_stride [[buffer(8)]],
    constant int &num_heads [[buffer(9)]],
    constant int &block_size [[buffer(10)]],
    constant int &num_tokens [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint dimension [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[HeadDim / 32];

  const uint rows = uint(num_tokens) * uint(num_heads);
  const bool value_tensor = group >= rows;
  const uint row = value_tensor ? group - rows : group;
  const uint token = row % uint(num_tokens);
  const uint head = row / uint(num_tokens);

  // Every thread of the group shares the token, so the whole group leaves
  // before any barrier.
  const int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  const ulong block = ulong(slot / block_size);
  const ulong page_token = ulong(slot % block_size);
  const ulong block_head = block * ulong(num_heads) + head;

  const float value_f =
      value_tensor
          ? float(value[ulong(token) * ulong(value_stride) + head * HeadDim + dimension])
          : float(key[ulong(token) * ulong(key_stride) + head * HeadDim + dimension]);

  float local_maximum = simd_max(abs(value_f));
  if (simd_lane == 0) {
    maxima[simd_group] = local_maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    for (uint offset = HeadDim / 64; offset != 0; offset >>= 1) {
      if (simd_lane < offset) {
        maxima[simd_lane] = max(maxima[simd_lane], maxima[simd_lane + offset]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float maximum = maxima[0];
  const float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
  const int quantized =
      maximum == 0.0f ? 0 : clamp(int(rint(value_f * 127.0f / maximum)), -127, 127);

  const ulong scale_index = block_head * ulong(block_size) + page_token;
  if (value_tensor) {
    value_cache[(block_head * HeadDim + dimension) * ulong(block_size) + page_token] =
        char(quantized);
    if (dimension == 0) {
      value_scales[scale_index] = scale;
    }
  } else {
    key_cache[scale_index * HeadDim + dimension] = char(quantized);
    if (dimension == 0) {
      key_scales[scale_index] = scale;
    }
  }
}

#define instantiate_reshape_and_cache_int8_ktm(head_dim)                      \
  template [[host_name("reshape_and_cache_int8_kv_bfloat16_t_hd" #head_dim     \
                       "_ktm")]]                                               \
  [[kernel]] void reshape_and_cache_int8_ktm<head_dim>(                        \
      const device bfloat16_t *__restrict__ key [[buffer(0)]],                 \
      const device bfloat16_t *__restrict__ value [[buffer(1)]],               \
      device char *__restrict__ key_cache [[buffer(2)]],                       \
      device char *__restrict__ value_cache [[buffer(3)]],                     \
      const device int64_t *__restrict__ slot_mapping [[buffer(4)]],           \
      device float *__restrict__ key_scales [[buffer(5)]],                     \
      device float *__restrict__ value_scales [[buffer(6)]],                   \
      constant int &key_stride [[buffer(7)]],                                  \
      constant int &value_stride [[buffer(8)]],                                \
      constant int &num_heads [[buffer(9)]],                                   \
      constant int &block_size [[buffer(10)]],                                 \
      constant int &num_tokens [[buffer(11)]],                                 \
      uint group [[threadgroup_position_in_grid]],                             \
      uint dimension [[thread_position_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]);

instantiate_reshape_and_cache_int8_ktm(128);
instantiate_reshape_and_cache_int8_ktm(256);
