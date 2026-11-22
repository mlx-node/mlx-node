import type { DecisionModel, ExtractionModel } from '@mlx-node/lm';

import type { ModelAdmissionLane } from './model-work-coordinator.js';
import { noteQueueRejection } from './queue-rejections.js';
import type { PreDispatchAdmission } from './session-registry.js';
import { notifyTurnEdge, type LiveTurn, type ModelIdentity } from './turn-trace.js';

export type SystemOneModel = DecisionModel | ExtractionModel;

export class DecisionQueueFullError extends Error {}
interface Binding extends ModelAdmissionLane {
  model: SystemOneModel;
  canonicalName: string;
  identity: ModelIdentity | undefined;
  pending: number;
  active: number;
  begun: number;
  lastSettleAt: number;
  tail: Promise<void>;
}
export interface DecisionEntry {
  id: string;
  created: number;
  binding: Binding;
}

export interface ResidentDecisionBinding {
  names: string[];
  model: SystemOneModel;
  kind: 'decision';
  sessions: {
    liveTurns(): LiveTurn[];
    turnStats(): { live: number; begun: number; lastSettleAt: number };
    readonly activeDispatches: number;
    readonly queueDepth: number;
    readonly queueDepthLimit: number | undefined;
    readonly concurrentAdmissionLimit: number;
    readonly identity: ModelIdentity | undefined;
    readonly servedWindowTokens: number | undefined;
    tryIdleProbe<T>(call: () => Promise<T>): Promise<T> | null;
  };
}

/** Identity-based serial admission; aliases and re-registration share one queue. */
export class DecisionRegistry {
  private readonly entries = new Map<string, DecisionEntry>();
  private readonly bindings = new WeakMap<SystemOneModel, Binding>();
  private readonly permits = new WeakMap<
    PreDispatchAdmission,
    { binding: Binding; released: boolean; consumed: boolean }
  >();
  constructor(private readonly maxQueueDepth?: number) {}
  private reserve(binding: Binding): PreDispatchAdmission {
    if (this.maxQueueDepth !== undefined && binding.pending >= this.maxQueueDepth + 1) {
      noteQueueRejection('decision');
      throw new DecisionQueueFullError('CLEF request queue is full');
    }
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
  register(name: string, model: SystemOneModel, opts?: { identity?: ModelIdentity }): void {
    let binding = this.bindings.get(model);
    if (!binding) {
      const created: Binding = {
        model,
        canonicalName: name,
        identity: opts?.identity,
        pending: 0,
        active: 0,
        begun: 0,
        lastSettleAt: 0,
        tail: Promise.resolve(),
        beginPreDispatchAdmission: () => this.reserve(created),
      };
      binding = created;
      this.bindings.set(model, binding);
    } else if (binding.identity === undefined && opts?.identity !== undefined) {
      binding.identity = opts.identity;
    }
    this.entries.set(name, { id: name, created: Math.floor(Date.now() / 1000), binding });
  }
  get(name: string): SystemOneModel | undefined {
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
  /** One resident entry per distinct decision model object, with all registered names pointing to it. */
  listResidentBindings(): ResidentDecisionBinding[] {
    const grouped = new Map<Binding, string[]>();
    for (const [name, entry] of this.entries) {
      const list = grouped.get(entry.binding);
      if (list === undefined) grouped.set(entry.binding, [name]);
      else list.push(name);
    }
    const maxQueueDepth = this.maxQueueDepth;
    const out: ResidentDecisionBinding[] = [];
    for (const [binding, names] of grouped) {
      out.push({
        names,
        model: binding.model,
        kind: 'decision',
        sessions: {
          liveTurns: () => [],
          turnStats: () => ({ live: binding.active, begun: binding.begun, lastSettleAt: binding.lastSettleAt }),
          get activeDispatches() {
            return binding.active;
          },
          get queueDepth() {
            return Math.max(0, binding.pending - binding.active);
          },
          queueDepthLimit: maxQueueDepth,
          concurrentAdmissionLimit: 1,
          identity: binding.identity,
          servedWindowTokens: binding.identity?.contextWindow,
          tryIdleProbe: () => null,
        },
      });
    }
    return out;
  }
  acquire(
    name: string,
    permit?: PreDispatchAdmission,
  ):
    | {
        model: SystemOneModel;
        canonicalName: string;
        identity: ModelIdentity | undefined;
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
      identity: binding.identity,
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
        let activeIncremented = false;
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
          binding.active += 1;
          binding.begun += 1;
          activeIncremented = true;
          notifyTurnEdge();
          return await fn();
        } finally {
          if (activeIncremented) {
            binding.active = Math.max(0, binding.active - 1);
            binding.lastSettleAt = Date.now();
            notifyTurnEdge();
          }
          if (onAbort) signal?.removeEventListener('abort', onAbort);
          // A cancelled waiter must not let its successor overtake the
          // still-running predecessor.
          void previous.then(finish);
        }
      },
    };
  }
}
