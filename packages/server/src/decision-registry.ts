import type { DecisionModel } from '@mlx-node/lm';

import type { ModelAdmissionLane } from './model-work-coordinator.js';
import type { PreDispatchAdmission } from './session-registry.js';

export class DecisionQueueFullError extends Error {}
interface Binding extends ModelAdmissionLane {
  model: DecisionModel;
  canonicalName: string;
  pending: number;
  tail: Promise<void>;
}
export interface DecisionEntry {
  id: string;
  created: number;
  binding: Binding;
}

/** Identity-based serial admission; aliases and re-registration share one queue. */
export class DecisionRegistry {
  private readonly entries = new Map<string, DecisionEntry>();
  private readonly bindings = new WeakMap<DecisionModel, Binding>();
  private readonly permits = new WeakMap<
    PreDispatchAdmission,
    { binding: Binding; released: boolean; consumed: boolean }
  >();
  constructor(private readonly maxQueueDepth?: number) {}
  private reserve(binding: Binding): PreDispatchAdmission {
    if (this.maxQueueDepth !== undefined && binding.pending >= this.maxQueueDepth + 1)
      throw new DecisionQueueFullError('CLEF request queue is full');
    binding.pending++;
    const state = { binding, released: false, consumed: false };
    const permit = {
      release: () => {
        if (!state.released && !state.consumed) {
          state.released = true;
          binding.pending--;
        }
      },
    };
    this.permits.set(permit, state);
    return permit;
  }
  lane(name: string): ModelAdmissionLane | undefined {
    return this.entries.get(name)?.binding;
  }
  register(name: string, model: DecisionModel): void {
    let binding = this.bindings.get(model);
    if (!binding) {
      const created: Binding = {
        model,
        canonicalName: name,
        pending: 0,
        tail: Promise.resolve(),
        beginPreDispatchAdmission: () => this.reserve(created),
      };
      binding = created;
      this.bindings.set(model, binding);
    }
    this.entries.set(name, { id: name, created: Math.floor(Date.now() / 1000), binding });
  }
  get(name: string): DecisionModel | undefined {
    return this.entries.get(name)?.binding.model;
  }
  unregister(name: string): boolean {
    return this.entries.delete(name);
  }
  list(): { id: string; object: string; created: number; owned_by: string }[] {
    return [...this.entries.values()].map((e) => ({
      id: e.id,
      object: 'model',
      created: e.created,
      owned_by: 'mlx-node',
    }));
  }
  acquire(
    name: string,
    permit?: PreDispatchAdmission,
  ):
    | {
        model: DecisionModel;
        canonicalName: string;
        run<T>(fn: () => Promise<T>, signal?: AbortSignal): Promise<T>;
        release(): void;
        current(): boolean;
      }
    | undefined {
    const entry = this.entries.get(name);
    if (!entry) return;
    const binding = entry.binding;
    // Count the running request plus waiters, including those waiting on the
    // process reader lock. Reserve synchronously before the first await.
    const admission = permit ?? this.reserve(binding);
    const state = this.permits.get(admission);
    if (!state || state.binding !== binding || state.released || state.consumed)
      throw new Error('Invalid CLEF admission permit');
    state.consumed = true;
    let released = false;
    let used = false;
    return {
      model: binding.model,
      canonicalName: binding.canonicalName,
      current: () => this.entries.get(name)?.binding === binding,
      release: () => {
        if (!released) {
          released = true;
          binding.pending--;
        }
      },
      async run<T>(fn: () => Promise<T>, signal?: AbortSignal): Promise<T> {
        if (used || released) throw new Error('Decision admission already consumed');
        used = true;
        const previous = binding.tail;
        let finish!: () => void;
        binding.tail = new Promise<void>((resolve) => {
          finish = resolve;
        });
        let onAbort: (() => void) | undefined;
        try {
          if (signal) {
            signal.throwIfAborted();
            const aborted = new Promise<never>((_, reject) => {
              onAbort = () => reject(signal.reason ?? new Error('Request aborted'));
              signal.addEventListener('abort', onAbort, { once: true });
            });
            await Promise.race([previous, aborted]);
          } else await previous;
          signal?.throwIfAborted();
          return await fn();
        } finally {
          if (onAbort) signal?.removeEventListener('abort', onAbort);
          // A cancelled waiter must not let its successor overtake the
          // still-running predecessor.
          void previous.then(finish);
        }
      },
    };
  }
}
