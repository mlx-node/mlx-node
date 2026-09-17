// Q4/group32 decode. Same lane assignment and FP32 reduction as MLX's
// affine_qmv_fast, with BF16 activation/output and independently typed sidecars.
// FP16 values widen in registers; no tensor conversion or dense weight exists.
R"(
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint row0 = threadgroup_position_in_grid.x * 8 + sg * 4;
  const int groups = K / 32;
  float result[4] = {0};
  for (int k = 0; k < K; k += 512) {
    const int base = k + lane * 16;
    // A group32 row can end halfway through the final 512-value tile.
    // Inactive lanes contribute zero to the final SIMD reduction.
    if (base >= K) continue;
    float xt[16];
    float sum = 0;
    for (int i = 0; i < 16; i += 4) {
      const float a = float(X[base + i]);
      const float b = float(X[base + i + 1]);
      const float c = float(X[base + i + 2]);
      const float d = float(X[base + i + 3]);
      sum += a + b + c + d;
      xt[i] = a;
      xt[i + 1] = b / 16.0f;
      xt[i + 2] = c / 256.0f;
      xt[i + 3] = d / 4096.0f;
    }
    for (int r = 0; r < 4; r++) {
      const uint row = row0 + r;
      const device ushort* wp = reinterpret_cast<const device ushort*>(W)
          + row * (K / 4) + base / 4;
      float dot = 0;
      for (int i = 0; i < 4; i++) {
        const ushort q = wp[i];
        dot += xt[4*i] * (q & 0x000f) + xt[4*i+1] * (q & 0x00f0)
             + xt[4*i+2] * (q & 0x0f00) + xt[4*i+3] * (q & 0xf000);
      }
      const int g = row * groups + base / 32;
      const float scale = float(S[g]);
      const float bias = SYMMETRIC
          ? (HALF_SCALES ? float(half(-8.0f * scale)) : -8.0f * scale)
          : float(B[g]);
      result[r] += scale * dot + sum * bias;
    }
  }
  for (int r = 0; r < 4; r++) {
    result[r] = simd_sum(result[r]);
    if (lane == 0) Y[row0 + r] = bfloat16_t(result[r]);
  }
)"
