import { RerankError } from '@mlx-node/rerank/protocol';

import type { PublicModelEntry } from '../handler.js';
import type { ModelRegistry } from '../registry.js';
import type { SwapController } from './swap.js';
export interface CapabilityAdapter {
  kind: 'chat' | 'asr' | 'tts' | 'rerank';
  listModels(): PublicModelEntry[];
  hasResident(name: string): boolean;
  resolveModel(name: string): Promise<void>;
  disposeAll(): Promise<void>;
}
/** Called under ModelWorkCoordinator's writer admission, shared by every task. */
export function withCapabilityModels(
  adapters: CapabilityAdapter[],
  residency: 'capability' | 'single',
): SwapController {
  const byName = new Map<string, CapabilityAdapter>();
  for (const adapter of adapters)
    for (const entry of adapter.listModels()) {
      if (byName.has(entry.id)) throw new Error(`Duplicate model name: ${entry.id}`);
      byName.set(entry.id, adapter);
    }
  const chat = adapters.find((a) => a.kind === 'chat' && a.listModels().length > 0);
  return {
    async resolveModel(name) {
      const adapter = byName.get(name) ?? chat;
      if (!adapter) throw new RerankError('model_not_found', `Model ${name} not found`, 404);
      if (adapter.hasResident(name)) return;
      if (residency === 'single') for (const other of adapters) if (other !== adapter) await other.disposeAll();
      await adapter.resolveModel(name);
    },
    listModels: () => adapters.flatMap((a) => a.listModels()).sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)),
  };
}
export function chatCapability(chat: SwapController, registry: ModelRegistry): CapabilityAdapter {
  return {
    kind: 'chat',
    listModels: () => chat.listModels(),
    hasResident: (name) => !!registry.getAny(name),
    resolveModel: (name) => chat.resolveModel(name),
    async disposeAll() {
      if (chat.disposeAll) {
        await chat.disposeAll();
        return;
      }
      const sessions = registry.listSessionRegistries();
      for (const item of registry.list()) if (registry.getAny(item.id)) registry.unregister(item.id);
      await Promise.all(sessions.map((s) => s.flushPendingDisposals()));
    },
  };
}
