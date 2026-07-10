/** Combine an abort signal with an optional second one: the result aborts when
 *  EITHER source does. Used to bind a /jspace readout's cancellation to BOTH the
 *  component lifetime AND the worker generation serving it, so a worker teardown
 *  (a post-ready worker/GPU error → model retry, which terminates + replaces the
 *  worker while /jspace stays mounted) rejects the in-flight readout promptly
 *  instead of stranding the single-flight queue until the 60 s client timeout. */
export function composeAbort(base: AbortSignal, extra?: AbortSignal): AbortSignal {
  return extra ? AbortSignal.any([base, extra]) : base;
}
