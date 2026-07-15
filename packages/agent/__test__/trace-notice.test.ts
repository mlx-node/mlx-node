import type {
  ExtensionAPI,
  ExtensionContext,
  InlineExtension,
  SessionShutdownEvent,
  SessionStartEvent,
} from '@earendil-works/pi-coding-agent';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { createTraceNoticeExtension } from '../src/extensions/trace-notice.js';

type SessionStartHandler = (event: SessionStartEvent, ctx: ExtensionContext) => Promise<void> | void;
type SessionShutdownHandler = (event: SessionShutdownEvent, ctx: ExtensionContext) => Promise<void> | void;

function captureHandlers(extension: InlineExtension): {
  sessionStart: SessionStartHandler;
  sessionShutdown: SessionShutdownHandler;
} {
  let sessionStart: SessionStartHandler | undefined;
  let sessionShutdown: SessionShutdownHandler | undefined;
  const pi = {
    on(event: string, handler: unknown): void {
      if (event === 'session_start') sessionStart = handler as SessionStartHandler;
      if (event === 'session_shutdown') sessionShutdown = handler as SessionShutdownHandler;
    },
  } as unknown as ExtensionAPI;

  expect(typeof extension).toBe('object');
  if (typeof extension === 'function') throw new Error('expected a named extension');
  expect(extension.name).toBe('mlx-trace-notice');
  void extension.factory(pi);
  expect(sessionStart).toBeDefined();
  expect(sessionShutdown).toBeDefined();
  return { sessionStart: sessionStart!, sessionShutdown: sessionShutdown! };
}

function context(mode: ExtensionContext['mode'], notify: ReturnType<typeof vi.fn>): ExtensionContext {
  return { mode, ui: { notify } } as unknown as ExtensionContext;
}

describe('createTraceNoticeExtension', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('announces the exact inference-log path after TUI session startup', () => {
    const logFile = '/Users/test/.mlx-node/logs/agent/run/inference.log';
    const notify = vi.fn();
    const { sessionStart } = captureHandlers(createTraceNoticeExtension(logFile));

    void sessionStart({ type: 'session_start', reason: 'startup' }, context('tui', notify));
    expect(notify).not.toHaveBeenCalled();
    vi.runAllTimers();

    expect(notify).toHaveBeenCalledOnce();
    expect(notify).toHaveBeenCalledWith(`mlx agent: inference log ${logFile}`, 'info');
  });

  it.each(['print', 'json', 'rpc'] as const)('does not duplicate the CLI announcement in %s mode', (mode) => {
    const notify = vi.fn();
    const { sessionStart } = captureHandlers(createTraceNoticeExtension('/tmp/inference.log'));

    void sessionStart({ type: 'session_start', reason: 'startup' }, context(mode, notify));
    vi.runAllTimers();

    expect(notify).not.toHaveBeenCalled();
  });

  it('cancels a pending notice during shutdown', () => {
    const notify = vi.fn();
    const ctx = context('tui', notify);
    const { sessionStart, sessionShutdown } = captureHandlers(createTraceNoticeExtension('/tmp/inference.log'));

    void sessionStart({ type: 'session_start', reason: 'startup' }, ctx);
    void sessionShutdown({ type: 'session_shutdown', reason: 'quit' }, ctx);
    vi.runAllTimers();

    expect(notify).not.toHaveBeenCalled();
  });
});
