import type { ModelWorkCoordinator } from './model-work-coordinator.js';
import type { PreDispatchAdmission, SessionRegistry } from './session-registry.js';

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
