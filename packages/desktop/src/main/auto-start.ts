/**
 * Whether launch should fork INFERENCE at all — the one startup decision that
 * must not be learned from a crash.
 *
 * On a machine with no downloaded models the sidecar's `createHost` rejects
 * with `NoModelsDiscoveredError` and exits `EXIT_STARTUP_FAILED`. Before this
 * gate, `autoStartInference` (default ON) sent every fresh install through
 * that failure on every launch — pointless even with the supervisor's
 * no-restart rule, and the tray's first impression was a red ✕ next to the
 * onboarding screen that exists to fix exactly this state.
 *
 * Fail-open on discovery ERROR (modelCount `null`): an unreadable models dir
 * is not the same fact as an empty one, and a hiccup in this pre-flight must
 * not silently disable auto-start — the sidecar's own discovery is
 * authoritative and now fails exactly once with a clear reason.
 */
export interface AutoStartDecision {
  start: boolean;
  /** One line, for the launch log. */
  reason: string;
}

export function decideAutoStart(input: { enabled: boolean; modelCount: number | null }): AutoStartDecision {
  if (!input.enabled) return { start: false, reason: 'disabled in settings' };
  if (input.modelCount === 0) return { start: false, reason: 'no local models yet' };
  if (input.modelCount === null) return { start: true, reason: 'model discovery failed; trying anyway' };
  return { start: true, reason: `${input.modelCount} local model(s)` };
}
