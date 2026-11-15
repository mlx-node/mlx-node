/**
 * What the CONTROL PANEL process tells MAIN: the pong (one synchronous function over two in-memory lookups), the
 * worker restart note (never an exit), and the closing note.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { createOpenCalls } from '@mlx-node/dashboard';
import { describe, expect, it } from 'vite-plus/test';

import {
  closingOf,
  firstClosing,
  onRuntimeLifecycle,
  pongOf,
  type PongRuntime,
} from '../src/control-panel/lifecycle.js';
import { readPanelMessage, type PanelMessage } from '../src/control-panel/liveness.js';

const CONV = '019a1f2e-7b3c-7d4e-8f50-61728394a5b6';

function stub(busy: string[], closing: string[] = busy): PongRuntime & { calls: number } {
  const runtime = {
    calls: 0,
    busyConversations: () => busy,
    closingConversations: () => closing,
    workerQueue: () => ({ wq: 2, wbusyMs: 41_000, wroute: 'GET /api/sessions/:id' }),
    call: () => {
      runtime.calls += 1;
      return Promise.resolve({ ok: true, status: 200, body: null });
    },
  };
  return runtime;
}

describe('the pong', () => {
  it('is synchronous, reads only the runtime’s in-memory lookups, and names the oldest slow call by pattern', () => {
    let now = 1_000;
    const calls = createOpenCalls(() => now);
    const close = calls.open({ method: 'GET', path: '/api/sessions/019a1f2e-7b3c-7d4e-8f50-61728394a5b6?x=1' });
    now += 6_000;
    const runtime = stub([CONV]);
    const pong = pongOf(4, runtime, calls, now);
    expect(pong).not.toBeInstanceOf(Promise);
    expect(runtime.calls).toBe(0);
    expect(pong).toEqual({
      type: 'mlx-pong',
      n: 4,
      open: 1,
      convs: [CONV],
      slow: {
        call: 1,
        route: 'GET /api/sessions/:id',
        thread: 'worker',
        ageMs: 6_000,
        wq: 2,
        wbusyMs: 41_000,
        wroute: 'GET /api/sessions/:id',
      },
    });
    expect(readPanelMessage(pong)).toEqual(pong);
    close();
    expect(pongOf(5, runtime, calls, now)).toMatchObject({ slow: null });
  });

  it('answers before the runtime exists', () => {
    expect(pongOf(1, null, createOpenCalls(), 0)).toEqual({ type: 'mlx-pong', n: 1, open: 0, convs: [], slow: null });
  });

  it('carries at most 8 conversations', () => {
    const many = Array.from({ length: 12 }, () => CONV);
    const pong = pongOf(1, stub(many), createOpenCalls(), 0);
    expect(pong.type === 'mlx-pong' ? [pong.open, pong.convs.length] : null).toEqual([12, 8]);
  });
});

describe('a database worker event', () => {
  it('is told to MAIN and never ends the process', () => {
    const posted: PanelMessage[] = [];
    const logged: string[] = [];
    const exit = process.exit.bind(process);
    let exits = 0;
    process.exit = ((): never => {
      exits += 1;
      throw new Error('exit');
    }) as typeof process.exit;
    try {
      onRuntimeLifecycle(
        { type: 'worker-restarted', reason: 'no-ack', route: 'GET /api/chat/conversations', ageMs: 64_000, attempt: 1 },
        (message) => posted.push(message),
        () => 1,
        (line) => logged.push(line),
      );
      onRuntimeLifecycle(
        { type: 'worker-gave-up', reason: 'exit', route: null, ageMs: null, attempt: 6 },
        (message) => posted.push(message),
        () => 0,
        (line) => logged.push(line),
      );
      onRuntimeLifecycle(
        { type: 'db-closed' },
        (message) => posted.push(message),
        () => 0,
        (line) => logged.push(line),
      );
    } finally {
      process.exit = exit;
    }
    expect(exits).toBe(0);
    expect(posted).toEqual([
      {
        type: 'mlx-worker',
        reason: 'no-ack',
        route: 'GET /api/chat/conversations',
        ageMs: 64_000,
        attempt: 1,
        open: 1,
      },
      { type: 'mlx-worker', reason: 'gave-up', route: null, ageMs: null, attempt: 6, open: 0 },
    ]);
    expect(logged).toHaveLength(2);
  });

  it('the entry has no exit path for a worker event', () => {
    const entry = readFileSync(fileURLToPath(new URL('../src/control-panel/index.ts', import.meta.url)), 'utf8');
    expect(entry).not.toContain('worker-down');
    expect(entry).not.toContain('teardown(70)');
    expect(entry).toContain('onRuntimeLifecycle(event, post');
  });
});

describe('the closing note', () => {
  it('lists the turns the teardown is about to cancel', () => {
    expect(closingOf(stub([CONV]))).toEqual({ type: 'mlx-closing', convs: [CONV] });
    expect(closingOf(stub([CONV], []))).toEqual({ type: 'mlx-closing', convs: [] });
    expect(closingOf(null)).toEqual({ type: 'mlx-closing', convs: [] });
  });

  it('goes once: a second signal sends no note to replace the first list', () => {
    const closing = firstClosing();
    expect(closing(stub([CONV]))).toEqual({ type: 'mlx-closing', convs: [CONV] });
    expect(closing(stub([]))).toBeNull();
  });
});
