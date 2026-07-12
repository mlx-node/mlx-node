import type {
  ExtensionAPI,
  ExtensionContext,
  SessionInfoChangedEvent,
  SessionShutdownEvent,
  SessionStartEvent,
} from '@earendil-works/pi-coding-agent';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { createTerminalTitleExtension } from '../src/extensions/terminal-title.js';

type Handler<Event> = (event: Event, ctx: ExtensionContext) => Promise<void> | void;

function loadExtension(sessionName?: string): {
  sessionStart: Handler<SessionStartEvent>;
  sessionInfoChanged: Handler<SessionInfoChangedEvent>;
  modelSelect: Handler<{ type: 'model_select' }>;
  sessionShutdown: Handler<SessionShutdownEvent>;
} {
  const handlers = new Map<string, unknown>();
  const pi = {
    getSessionName: () => sessionName,
    on(event: string, handler: unknown): void {
      handlers.set(event, handler);
    },
  } as unknown as ExtensionAPI;

  const extension = createTerminalTitleExtension();
  expect(typeof extension).toBe('object');
  if (typeof extension === 'function') throw new Error('expected a named extension');
  expect(extension.name).toBe('mlx-terminal-title');
  void extension.factory(pi);

  return {
    sessionStart: handlers.get('session_start') as Handler<SessionStartEvent>,
    sessionInfoChanged: handlers.get('session_info_changed') as Handler<SessionInfoChangedEvent>,
    modelSelect: handlers.get('model_select') as Handler<{ type: 'model_select' }>,
    sessionShutdown: handlers.get('session_shutdown') as Handler<SessionShutdownEvent>,
  };
}

function makeContext(
  mode: ExtensionContext['mode'] = 'tui',
  modelId?: string,
): { ctx: ExtensionContext; titles: string[] } {
  const titles: string[] = [];
  const ctx = {
    mode,
    cwd: '/Users/brooklyn/workspace/github/Image',
    model: modelId === undefined ? undefined : { id: modelId },
    ui: {
      setTitle(title: string): void {
        titles.push(title);
      },
    },
  } as unknown as ExtensionContext;
  return { ctx, titles };
}

describe('createTerminalTitleExtension', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('automatically replaces pi branding and includes the selected model after session startup', () => {
    const { sessionStart } = loadExtension();
    const { ctx, titles } = makeContext('tui', 'qwen3.5-35b-a3b');

    void sessionStart({ type: 'session_start', reason: 'startup' }, ctx);
    expect(titles).toEqual([]);
    vi.runAllTimers();

    expect(titles).toEqual(['mlx - qwen3.5-35b-a3b - Image']);
  });

  it('includes the current session name', () => {
    const { sessionInfoChanged } = loadExtension('Qwen work');
    const { ctx, titles } = makeContext();

    void sessionInfoChanged({ type: 'session_info_changed', name: 'Qwen work' }, ctx);

    expect(titles).toEqual(['mlx - Qwen work - Image']);
  });

  it('updates automatically when the selected model changes', () => {
    const { modelSelect } = loadExtension();
    const { ctx, titles } = makeContext('tui', 'qwen3.6-27b-mxfp4');

    void modelSelect({ type: 'model_select' }, ctx);

    expect(titles).toEqual(['mlx - qwen3.6-27b-mxfp4 - Image']);
  });

  it('does not emit terminal control updates outside TUI mode', () => {
    const { sessionStart, sessionInfoChanged } = loadExtension();
    const { ctx, titles } = makeContext('print');

    void sessionStart({ type: 'session_start', reason: 'startup' }, ctx);
    void sessionInfoChanged({ type: 'session_info_changed', name: undefined }, ctx);
    vi.runAllTimers();

    expect(titles).toEqual([]);
  });

  it('cancels a pending startup update during shutdown', () => {
    const { sessionStart, sessionShutdown } = loadExtension();
    const { ctx, titles } = makeContext();

    void sessionStart({ type: 'session_start', reason: 'startup' }, ctx);
    void sessionShutdown({ type: 'session_shutdown', reason: 'quit' }, ctx);
    vi.runAllTimers();

    expect(titles).toEqual([]);
  });
});
