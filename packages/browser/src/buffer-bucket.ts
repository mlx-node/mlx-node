/**
 * Buffer size bucketing — SHARED by gpu-worker and webgpu-bridge-stub.
 *
 * Exact-size keying misses when requests differ by a few bytes (e.g., 513 vs 512).
 * Bucketing coalesces near-size requests into shared pool bins so they can
 * reuse each other's buffers.
 *
 * Scheme:
 *   - size <= 4096:  round up to next multiple of 256  (waste <= 256 B)
 *   - size >  4096:  round up to next power of two     (waste <= 2x)
 *
 * Rationale:
 *   256-B granule caps small-buffer waste at 256 B (tight for uniforms, tiny
 *   params). Power-of-two for large buffers caps waste at 2x, which is fine
 *   on M3's 18 GB unified memory and gives broad coalescing across nearby
 *   allocation sizes (e.g., KV cache rows of various seq lengths).
 *
 * INVARIANT: gpu-worker and webgpu-bridge-stub MUST use this same function
 * so that a release in one bucket always matches a create lookup in the
 * same bucket. Diverging implementations would cause pool lookups to miss
 * buffers that are already parked — or worse, return a physically-undersized
 * buffer from a bucket with a larger logical request.
 *
 * Correctness: pure function, no side effects. Always returns size >= input,
 * always a multiple of 4 (WebGPU minimum alignment). For size <= 0 returns 0
 * — callers already guard against pooling empty buffers.
 */
export function roundUpBucket(size: number): number {
  if (size <= 0) return 0;
  if (size <= 4096) {
    // Round up to next multiple of 256.
    return (size + 255) & ~255;
  }
  // Round up to next power of two. For 32-bit sizes this is safe; WebGPU
  // buffer size limits bound us well below 2^31. Math.clz32 returns the
  // leading-zero count of size-1 when viewed as u32.
  const v = size - 1;
  const shift = 32 - Math.clz32(v);
  return 1 << shift;
}
