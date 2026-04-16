/**
 * Buffer size keying — SHARED by gpu-worker and webgpu-bridge-stub.
 *
 * History:
 *   Task D (commits 9ec0e28, 573c34d, 7c4f451, 3cdf682) introduced 256B /
 *   PO2 bucketing under the hypothesis that near-size requests would share
 *   pool bins and coalesce into shared buffers. Empirical data from the
 *   `[profile] bridge-hist` / `[profile] gpu-hist` instrumentation (commit
 *   6f15a3f) falsified that hypothesis: the MLX allocator emits a small
 *   number of EXACT sizes (driven by `mx::compile`'s deterministic kernel
 *   shapes), so exact-size keying would hit just as often while avoiding
 *   the per-miss over-allocation. Worse, the small-size bucket `bucket=256`
 *   combined four distinct workloads (size=8, 32, 64, 256) with only
 *   ~7% hit rate, meaning bucketing ACTIVELY HURT reuse on the hot tiny-
 *   uniform stream while wasting up to 32x memory per allocation.
 *
 *   Measurements on 2026-04-16 Qwen3.5-0.8B decode (M3):
 *   - Bucketed (3cdf682):    18.6–18.8 tok/s, pool hit 63.6% / gpu-pool 45.8%
 *   - Baseline (pre-Task-D): 20.9 tok/s,      pool hit 62.4%
 *
 *   Decision: revert to exact-size keying. The bucket module is retained as
 *   the shared source of truth so both pools stay in agreement on keying;
 *   if a future workload shift ever warrants bucketing, this is the single
 *   place to change.
 *
 * Semantics:
 *   - size <= 0: returns 0 (callers already guard against pooling empty).
 *   - size >  0: returns size unchanged (identity).
 *
 * INVARIANT: gpu-worker and webgpu-bridge-stub MUST share this function so
 * that a release in bucket B always matches a create lookup in bucket B.
 * Diverging implementations would cause pool lookups to miss buffers that
 * are already parked.
 *
 * Correctness: pure function, no side effects. Always returns size >= input.
 */
export function roundUpBucket(size: number): number {
  if (size <= 0) return 0;
  return size;
}
