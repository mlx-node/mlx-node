import type { ExtensionAPI, ExtensionContext, InlineExtension } from '@earendil-works/pi-coding-agent';
import { describe, expect, it, vi } from 'vite-plus/test';

import { createMlxProviderExtension } from '../src/provider/index.js';
import type { MlxModelHost } from '../src/provider/model-host.js';

function loadExtension(): {
  handlers: Map<string, (event: never, ctx: ExtensionContext) => void>;
  registerProvider: ReturnType<typeof vi.fn>;
} {
  const handlers = new Map<string, (event: never, ctx: ExtensionContext) => void>();
  const registerProvider = vi.fn();
  const pi = {
    registerProvider,
    on(event: string, handler: (event: never, ctx: ExtensionContext) => void): void {
      handlers.set(event, handler);
    },
  } as unknown as ExtensionAPI;
  const extension: InlineExtension = createMlxProviderExtension([], {} as MlxModelHost);
  if (typeof extension === 'function') throw new Error('expected a named extension');
  void extension.factory(pi);
  return { handlers, registerProvider };
}

describe('createMlxProviderExtension', () => {
  it('registers the provider and all performance-status lifecycle handlers', () => {
    const { handlers, registerProvider } = loadExtension();

    expect(registerProvider).toHaveBeenCalledOnce();
    expect([...handlers.keys()]).toEqual(['message_end', 'turn_start', 'model_select', 'session_shutdown']);
  });

  it.each(['turn_start', 'model_select', 'session_shutdown'])('clears stale TUI performance on %s', (event) => {
    const { handlers } = loadExtension();
    const setStatus = vi.fn();
    const ctx = { mode: 'tui', ui: { setStatus } } as unknown as ExtensionContext;

    handlers.get(event)!({} as never, ctx);

    expect(setStatus).toHaveBeenCalledWith('mlx-performance', undefined);
  });
});
