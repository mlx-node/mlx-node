import { rerankCapabilities } from '@mlx-node/rerank/catalog';
import { RerankError } from '@mlx-node/rerank/protocol';
import type { RerankerModel } from '@mlx-node/rerank/types';

import type { PublicModelEntry } from './handler.js';
import type { ModelAdmissionLane, ModelWorkCoordinator } from './model-work-coordinator.js';
import type { PreDispatchAdmission } from './session-registry.js';
interface Binding extends ModelAdmissionLane {
  model: RerankerModel;
  pending: number;
  tail: Promise<void>;
}
type Coordinator = Pick<ModelWorkCoordinator, 'bindRequestLoadAdmissions' | 'unbindRequestLoadAdmissions'>;
/** Each model identity owns one bounded FIFO, including every alias. */
export class RerankRegistry {
  private readonly entries = new Map<string, Binding>();
  private readonly bindings = new WeakMap<RerankerModel, Binding>();
  private readonly permits = new WeakMap<
    PreDispatchAdmission,
    { binding: Binding; consumed: boolean; released: boolean }
  >();
  private coordinator?: Coordinator;
  constructor(private readonly maxQueueDepth?: number) {}
  setCoordinator(value?: Coordinator): void {
    if (value === this.coordinator) return;
    for (const [name, binding] of this.entries) {
      this.coordinator?.unbindRequestLoadAdmissions(name, binding);
      value?.bindRequestLoadAdmissions(name, binding);
    }
    this.coordinator = value;
  }
  register(name: string, model: RerankerModel): void {
    let binding = this.bindings.get(model);
    if (!binding) {
      const created: Binding = {
        model,
        pending: 0,
        tail: Promise.resolve(),
        beginPreDispatchAdmission: () => this.reserve(created),
      };
      binding = created;
      this.bindings.set(model, binding);
    }
    if (this.entries.get(name) === binding) return;
    this.unregister(name);
    this.entries.set(name, binding);
    this.coordinator?.bindRequestLoadAdmissions(name, binding);
  }
  unregister(name: string): void {
    const binding = this.entries.get(name);
    if (binding) this.coordinator?.unbindRequestLoadAdmissions(name, binding);
    this.entries.delete(name);
  }
  get(name: string): RerankerModel | undefined {
    return this.entries.get(name)?.model;
  }
  lane(name: string): ModelAdmissionLane | undefined {
    return this.entries.get(name);
  }
  get pending(): number {
    return [...new Set(this.entries.values())].reduce((n, b) => n + b.pending, 0);
  }
  get queueDepth(): number {
    return Math.max(0, ...[...new Set(this.entries.values())].map((b) => b.pending - 1));
  }
  list(): PublicModelEntry[] {
    return [...this.entries.keys()].map((id) => ({
      id,
      object: 'model',
      created: 0,
      owned_by: 'mlx-node',
      modality: 'text',
      task: 'rerank',
      model_family: this.entries.get(id)!.model.info.family,
      capabilities: rerankCapabilities,
    }));
  }
  async disposeAll(): Promise<void> {
    const models = new Set([...this.entries.values()].map((b) => b.model));
    for (const name of this.entries.keys()) this.unregister(name);
    const results = await Promise.allSettled([...models].map((model) => model.dispose()));
    const failures = results.filter((r) => r.status === 'rejected').map((r) => r.reason);
    if (failures.length) throw new AggregateError(failures, 'Reranker disposal failed');
  }
  private reserve(binding: Binding): PreDispatchAdmission {
    if (this.maxQueueDepth !== undefined && binding.pending >= this.maxQueueDepth + 1)
      throw new RerankError('queue_full', 'Rerank queue is full', 429);
    binding.pending++;
    const state = { binding, consumed: false, released: false };
    const permit = {
      release: () => {
        if (!state.consumed && !state.released) {
          state.released = true;
          binding.pending--;
        }
      },
    };
    this.permits.set(permit, state);
    return permit;
  }
  acquire(name: string, permit?: PreDispatchAdmission) {
    const binding = this.entries.get(name);
    if (!binding) return undefined;
    const admission = permit ?? this.reserve(binding);
    const state = this.permits.get(admission);
    if (!state || state.binding !== binding || state.consumed || state.released)
      throw new Error('Invalid rerank admission permit');
    state.consumed = true;
    let released = false;
    let used = false;
    return {
      model: binding.model,
      current: () => this.entries.get(name) === binding,
      release: () => {
        if (!released) {
          released = true;
          binding.pending--;
        }
      },
      async run<T>(fn: () => Promise<T>, signal?: AbortSignal): Promise<T> {
        if (released || used) throw new Error('Rerank admission already consumed');
        used = true;
        const previous = binding.tail;
        let finish!: () => void;
        binding.tail = new Promise<void>((resolve) => {
          finish = resolve;
        });
        let abort: (() => void) | undefined;
        try {
          signal?.throwIfAborted();
          if (signal)
            await Promise.race([
              previous,
              new Promise<never>((_, reject) => {
                abort = () => reject(signal.reason);
                signal.addEventListener('abort', abort, { once: true });
              }),
            ]);
          else await previous;
          signal?.throwIfAborted();
          return await fn();
        } finally {
          if (abort) signal?.removeEventListener('abort', abort);
          // Cancelled waiters cannot let successors overtake a running predecessor.
          void previous.then(finish);
        }
      },
    };
  }
}
