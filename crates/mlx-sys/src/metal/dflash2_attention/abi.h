#pragma once
#ifdef __METAL_VERSION__
using DFlashStride = metal::ulong;
#else
#include <cstdint>
using DFlashStride = uint64_t;
#endif
struct DraftAttentionParams {
  DFlashStride query_head_stride;
  DFlashStride query_row_stride;
  DFlashStride key_head_stride;
  DFlashStride key_row_stride;
  DFlashStride value_head_stride;
  DFlashStride value_row_stride;
  unsigned key_rows;
  unsigned has_mask;
  float scale;
  unsigned reserved;
};
