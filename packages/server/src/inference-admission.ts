import type { ModelWorkCoordinator } from './model-work-coordinator.js';
import type { ModelRegistry } from './registry.js';
import type { PreDispatchAdmission, SessionRegistry } from './session-registry.js';

/**
 * The session registry `name` resolves to with no model work: its own, or the
 * loaded model's once `bindLoadedModel` binds `name` to it. `undefined` sends
 * the request through `resolveModel` under the writer slot, which waits for
 * every running turn.
 */
export function residentSessionRegistry(
  registry: ModelRegistry,
  name: string,
  bindLoadedModel: ((name: string) => boolean) | undefined,
  modelWorkCoordinator: ModelWorkCoordinator | undefined,
): SessionRegistry | undefined {
  const resident = registry.getSessionRegistry(name);
  if (resident !== undefined || bindLoadedModel === undefined) return resident;
  // A held or queued writer may swap the model out after this request takes its reader slot.
  if (
    modelWorkCoordinator !== undefined &&
    (modelWorkCoordinator.writerActive || modelWorkCoordinator.waitingWriters > 0)
  )
    return undefined;
  return bindLoadedModel(name) ? registry.getSessionRegistry(name) : undefined;
}

export function withAdmissionControlledInference<T>(
  sessionReg: SessionRegistry,
  modelWorkCoordinator: ModelWorkCoordinator | undefined,
  // Pre-dispatch permit handed off ATOMICALLY as this call's admission
  // (the selected admission lane consumes it instead of charging
  // `queuedCount` a second time). See `beginPreDispatchAdmission`. Placed BEFORE `fn`
  // so call sites keep the trailing-closure layout.
  permit: PreDispatchAdmission | undefined,
  fn: () => Promise<T>,
): Promise<T> {
  const run = () => (modelWorkCoordinator ? modelWorkCoordinator.withInference(fn) : fn());
  return sessionReg.concurrentAdmissionLimit > 1
    ? sessionReg.withAdmission(run, permit)
    : sessionReg.withExclusive(run, permit);
}
