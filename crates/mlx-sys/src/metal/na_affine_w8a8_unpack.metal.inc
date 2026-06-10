// Affine-W8A8 LOAD-TIME weight unpack, for fast::metal_kernel JIT.
//
// Unpacks the MLX affine 8-bit packed weight into the SIGNED int8 [K,N] kernel
// operand the tiled matmul2d consumes. Done ONCE at load (mlx_affine_w8a8_prepare).
//
//   wq  : const device uint32_t* [N, K/4]  packed affine 8-bit weight. 4 uint8
//         per word, little-endian; uint8 q[n,k] = byte (k%4) of wq[n*(K/4)+k/4].
//   N,K : const constant int32_t&
//   q_s : device int8_t*         [K,N]      output, contiguous [K,N] with
//         q_s[k*N + n] = (int8_t)(q[n,k] - 128). This is the SAME column-major
//         (N,K) operand the symmetric kernel calls b_kn: the matmul2d's
//         tensor_inline (N,K) view reads q_s[n + k*N] = q[n,k]-128.
//
// q in [0,255] - 128 -> q_s in [-128,127], exactly representable as int8_t.
//
// One thread per (n,k) output element: grid.x = N, grid.y = K.
R"(
  const uint n = thread_position_in_grid.x;   // output channel / column
  const uint k = thread_position_in_grid.y;   // K index / row
  if (n >= (uint)N || k >= (uint)K) {
    return;
  }
  const uint k_packed = (uint)K / 4u;
  const uint word = wq[(size_t)n * (size_t)k_packed + (size_t)(k >> 2)];
  const uint q = (word >> ((k & 3u) * 8u)) & 0xFFu;   // [0,255]
  const int q_s_val = (int)q - 128;                   // [-128,127]
  q_s[(size_t)k * (size_t)N + (size_t)n] = (int8_t)q_s_val;
)"
