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
 *
 * 64-bit-safety: buffers CAN exceed 2^31 bytes (KV caches, large weight
 * tensors, > 4 GiB allocations). The large-size branch uses non-bitwise
 * arithmetic (iterative doubling starting at 8192) so it is correct for all
 * finite JS integer sizes up to Number.MAX_SAFE_INTEGER (2^53). Bitwise
 * operators must NOT be used on values that may exceed 2^31, since JS
 * coerces them to int32 and silently truncates.
 */
export function roundUpBucket(size: number): number {
  if (size <= 0) return 0;
  if (size <= 4096) {
    // Round up to next multiple of 256. `(size + 255) & ~255` is safe here
    // because size <= 4096 is well within int32 range.
    return (size + 255) & ~255;
  }
  // Round up to next power of two via iterative doubling. 4096 is the
  // small-branch ceiling, so the first large bucket is 8192. Max ~41
  // iterations to reach 2^53 — negligible cost vs. a GPU RPC. Uses plain
  // multiplication to avoid the 32-bit overflow that `1 << shift` would
  // incur for shifts >= 31 (e.g., a 2 GiB buffer would wrap to 1 byte).
  let bucket = 8192;
  while (bucket < size) {
    bucket *= 2;
  }
  return bucket;
}
