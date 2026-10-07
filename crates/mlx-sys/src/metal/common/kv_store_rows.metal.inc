// In-place KV row store (Splash `verify_attention_*_store`,
// runtime/metal/kernels/common/paged_store_row.h, flat-cache form): ONE
// dispatch writes new K and V rows (and, for int8 caches, their scales) into
// the cache buffers at a row offset. A bit copy: the rows arrive in the
// cache's element format (BF16, or int8 + f32 scales quantized by the verify
// graph), so the result equals the slice_update per tensor it replaces.
//
// Tensor i: src `[B, H, T, D_i]` with arbitrary strides, dst `[B, H, cap,
// D_i]` with its own strides; scales are D = 1. Grid (threads):
// (max_i elems_i, tensors); threadgroup (256, 1, 1).
//
// Two entry points share the addressing: `mlx_node_kv_store_rows` binds up
// to four tensors (one layer), `mlx_node_kv_store_rows_table` reads any
// number of (dst, src) GPU addresses from a table so every layer's store is
// one dispatch (the buffers are resident through MLX's residency sets).
R"(
#define KV_STORE_MAX_TENSORS 4

struct KvStoreParams {
  int tensors;
  int rows;      // T
  int offset;    // first destination row
  int batch;
  int heads;
  int dims[KV_STORE_MAX_TENSORS];
  int itemsize[KV_STORE_MAX_TENSORS];
  uint elems[KV_STORE_MAX_TENSORS];
  long src_strides[KV_STORE_MAX_TENSORS][4];  // (b, h, t, d) in elements
  long dst_strides[KV_STORE_MAX_TENSORS][4];
};

struct KvStoreTensor {
  ulong dst;             // GPU address of the cache buffer (+ array offset)
  ulong src;             // GPU address of the row block
  long src_strides[4];   // (b, h, t, d) in elements
  long dst_strides[4];
  uint elems;
  int itemsize;
  int dims;
  int rows;
  int heads;
  int offset;
};

template <typename SrcStrides, typename DstStrides>
METAL_FUNC void kv_store_copy(device uchar* dst, const device uchar* src,
                              uint e, int itemsize, int d_n, int rows,
                              int heads, int offset, SrcStrides src_strides,
                              DstStrides dst_strides) {
  const uint dn = uint(d_n);
  const uint d = e % dn;
  const uint t = (e / dn) % uint(rows);
  const uint h = (e / (dn * uint(rows))) % uint(heads);
  const uint b = e / (dn * uint(rows) * uint(heads));
  const long s = long(b) * src_strides[0] + long(h) * src_strides[1] +
      long(t) * src_strides[2] + long(d) * src_strides[3];
  const long o = long(b) * dst_strides[0] + long(h) * dst_strides[1] +
      long(offset + int(t)) * dst_strides[2] + long(d) * dst_strides[3];
  switch (itemsize) {
    case 1:
      dst[o] = src[s];
      break;
    case 2:
      reinterpret_cast<device ushort*>(dst)[o] =
          reinterpret_cast<const device ushort*>(src)[s];
      break;
    default:
      reinterpret_cast<device uint*>(dst)[o] =
          reinterpret_cast<const device uint*>(src)[s];
      break;
  }
}

[[kernel]] void mlx_node_kv_store_rows(
    device uchar* dst0 [[buffer(0)]], device uchar* dst1 [[buffer(1)]],
    device uchar* dst2 [[buffer(2)]], device uchar* dst3 [[buffer(3)]],
    const device uchar* src0 [[buffer(4)]], const device uchar* src1 [[buffer(5)]],
    const device uchar* src2 [[buffer(6)]], const device uchar* src3 [[buffer(7)]],
    constant KvStoreParams& p [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.y;
  const uint e = gid.x;
  if (i >= uint(p.tensors) || e >= p.elems[i]) {
    return;
  }
  device uchar* dst;
  const device uchar* src;
  switch (i) {
    case 0: dst = dst0; src = src0; break;
    case 1: dst = dst1; src = src1; break;
    case 2: dst = dst2; src = src2; break;
    default: dst = dst3; src = src3; break;
  }
  kv_store_copy(dst, src, e, p.itemsize[i], p.dims[i], p.rows, p.heads,
                p.offset, p.src_strides[i], p.dst_strides[i]);
}

[[kernel]] void mlx_node_kv_store_rows_table(
    const device KvStoreTensor* table [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]) {
  const device KvStoreTensor& p = table[gid.y];
  const uint e = gid.x;
  if (e >= p.elems) {
    return;
  }
  kv_store_copy(reinterpret_cast<device uchar*>(p.dst),
                reinterpret_cast<const device uchar*>(p.src), e, p.itemsize,
                p.dims, p.rows, p.heads, p.offset, p.src_strides,
                p.dst_strides);
}
)"
