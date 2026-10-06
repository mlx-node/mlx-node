// In-place KV row store (Splash `verify_attention_*_store`,
// runtime/metal/kernels/common/paged_store_row.h, flat-cache form): ONE
// dispatch writes a layer's new K and V rows (and, for int8 caches, their
// scales) into the cache buffers at `offset`. A bit copy: the rows arrive in
// the cache's element format (BF16, or int8 + f32 scales quantized by the
// verify graph), so the result equals the slice_update per tensor it
// replaces.
//
// Tensor i: src `[B, H, T, D_i]` with arbitrary strides, dst `[B, H, cap,
// D_i]` with its own strides; scales are D = 1. Grid (threads):
// (max_i elems_i, tensors); threadgroup (256, 1, 1).
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
  const uint d_n = uint(p.dims[i]);
  const uint d = e % d_n;
  const uint t = (e / d_n) % uint(p.rows);
  const uint h = (e / (d_n * uint(p.rows))) % uint(p.heads);
  const uint b = e / (d_n * uint(p.rows) * uint(p.heads));
  const long s = long(b) * p.src_strides[i][0] + long(h) * p.src_strides[i][1] +
      long(t) * p.src_strides[i][2] + long(d) * p.src_strides[i][3];
  const long o = long(b) * p.dst_strides[i][0] + long(h) * p.dst_strides[i][1] +
      long(p.offset + int(t)) * p.dst_strides[i][2] + long(d) * p.dst_strides[i][3];
  switch (p.itemsize[i]) {
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
)"
