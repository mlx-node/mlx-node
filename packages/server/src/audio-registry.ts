import type { Qwen3AsrResult, Qwen3AsrStreamOptions, Qwen3AsrTranscribeOptions } from '@mlx-node/core';
import type { TtsModel } from '@mlx-node/tts';

import type { PublicModelEntry } from './handler.js';

export interface AsrStream {
  feed(samples: Float32Array): Promise<Qwen3AsrResult | null | undefined>;
  finish(): Promise<Qwen3AsrResult>;
  cancel(): void;
  dispose(): Promise<unknown>;
}
export interface AsrModel {
  startTranscription(
    samples: Float32Array,
    options?: Qwen3AsrTranscribeOptions,
  ): { cancel(): void; result(): Promise<Qwen3AsrResult> };
  createStream(options?: Qwen3AsrStreamOptions): Promise<AsrStream>;
  dispose(): Promise<unknown>;
}
export type AudioModel = { kind: 'asr'; model: AsrModel } | { kind: 'tts'; model: TtsModel };

import type { ModelAdmissionLane } from './model-work-coordinator.js';
import { noteQueueRejection } from './queue-rejections.js';
import type { PreDispatchAdmission } from './session-registry.js';

export class AudioQueueFullError extends Error {}
interface Binding extends ModelAdmissionLane {
  model: AudioModel;
  canonicalName: string;
  pending: number;
  tail: Promise<void>;
}
export interface AudioEntry {
  id: string;
  created: number;
  binding: Binding;
}

/** Identity-based serial admission; aliases and re-registration share one queue. */
export class AudioRegistry {
  private readonly entries = new Map<string, AudioEntry>();
  private readonly unregisterListeners = new Set<(model: AudioModel) => void>();
  private readonly bindings = new WeakMap<object, Binding>();
  private readonly permits = new WeakMap<
    PreDispatchAdmission,
    { binding: Binding; released: boolean; consumed: boolean }
  >();
  private coordinator?: Pick<
    import('./model-work-coordinator.js').ModelWorkCoordinator,
    'bindRequestLoadAdmissions' | 'unbindRequestLoadAdmissions'
  >;
  constructor(private readonly maxQueueDepth?: number) {}
  setCoordinator(coordinator: typeof this.coordinator): void {
    if (this.coordinator === coordinator) return;
    for (const [name, entry] of this.entries) {
      this.coordinator?.unbindRequestLoadAdmissions(name, entry.binding);
      coordinator?.bindRequestLoadAdmissions(name, entry.binding);
    }
    this.coordinator = coordinator;
  }
  get pending(): number {
    return [...new Set([...this.entries.values()].map((e) => e.binding))].reduce((n, b) => n + b.pending, 0);
  }
  get queueDepth(): number {
    return Math.max(0, ...[...this.entries.values()].map((e) => Math.max(0, e.binding.pending - 1)));
  }
  async disposeAll(kind?: AudioModel['kind']): Promise<void> {
    const models = new Set(
      [...this.entries.values()].filter((e) => !kind || e.binding.model.kind === kind).map((e) => e.binding.model),
    );
    for (const [name, entry] of this.entries) if (models.has(entry.binding.model)) this.unregister(name);
    const results = await Promise.allSettled([...models].map((m) => m.model.dispose()));
    const failures = results.filter((r) => r.status === 'rejected');
    if (failures.length)
      throw new AggregateError(
        failures.map((r) => (r as PromiseRejectedResult).reason),
        'Audio model disposal failed',
      );
  }
  private reserve(binding: Binding): PreDispatchAdmission {
    if (this.maxQueueDepth !== undefined && binding.pending >= this.maxQueueDepth + 1) {
      noteQueueRejection('audio');
      throw new AudioQueueFullError('Audio request queue is full');
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
  register(name: string, model: AudioModel): void {
    let binding = this.bindings.get(model.model);
    if (!binding) {
      const created: Binding = {
        model,
        canonicalName: name,
        pending: 0,
        tail: Promise.resolve(),
        beginPreDispatchAdmission: () => this.reserve(created),
      };
      binding = created;
      this.bindings.set(model.model, binding);
    }
    if (this.entries.get(name)?.binding === binding) return;
    this.unregister(name);
    this.entries.set(name, { id: name, created: Math.floor(Date.now() / 1000), binding });
    this.coordinator?.bindRequestLoadAdmissions(name, binding);
  }
  get(name: string): AudioModel | undefined {
    return this.entries.get(name)?.binding.model;
  }
  unregister(name: string): boolean {
    const binding = this.entries.get(name)?.binding;
    if (binding) this.coordinator?.unbindRequestLoadAdmissions(name, binding);
    const removed = this.entries.delete(name);
    if (binding && ![...this.entries.values()].some((entry) => entry.binding === binding))
      for (const listener of this.unregisterListeners) listener(binding.model);
    return removed;
  }
  onUnregister(listener: (model: AudioModel) => void): () => void {
    this.unregisterListeners.add(listener);
    return () => this.unregisterListeners.delete(listener);
  }
  nameForModel(model: TtsModel): string | undefined {
    return [...this.entries.values()].find((entry) => entry.binding.model.model === model)?.id;
  }
  list(): PublicModelEntry[] {
    return [...this.entries.values()].map((e) => ({
      id: e.id,
      object: 'model',
      created: e.created,
      owned_by: 'mlx-node',
      modality: 'audio',
      task: e.binding.model.kind,
      capabilities:
        e.binding.model.kind === 'tts' ? ttsCapabilities(e.binding.model.model.capabilities) : asrCapabilities,
    }));
  }
  acquire(
    name: string,
    permit?: PreDispatchAdmission,
  ):
    | {
        model: AudioModel;
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
      throw new Error('Invalid Audio admission permit');
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
        if (used || released) throw new Error('Audio admission already consumed');
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

export const asrCapabilities = {
  transcription: true,
  streaming: 'revision',
  input_formats: ['wav'],
  native_sample_rate: 16000,
};
export function ttsCapabilities(c: TtsModel['capabilities']): Record<string, unknown> {
  return {
    family: c.family,
    variant: c.variant,
    sample_rate: c.sampleRate,
    channels: c.channels,
    voices: c.voices,
    languages: c.languages,
    conditioning: c.conditioning,
    voice_cloning: c.voiceCloning,
    text_streaming: c.textStreaming,
    audio_streaming: c.audioStreaming,
    http_speech: c.conditioning.some((condition) => condition.voice !== 'reference' || c.voiceCloning),
    http_voice_cloning: c.voiceCloning,
  };
}
