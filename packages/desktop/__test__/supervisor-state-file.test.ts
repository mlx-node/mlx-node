import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { mkdir, rename, unlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  createSupervisorStateWriter,
  projectSupervisorState,
  type SupervisorStateFile,
  type SupervisorStateWriterIo,
} from '../src/main/supervisor-state-file.js';
import { MAX_NATIVE_ERRORS } from '../src/main/supervisor/index.js';
import type { SupervisorSnapshot } from '../src/main/supervisor/types.js';

function snapshot(overrides: Partial<SupervisorSnapshot> = {}): SupervisorSnapshot {
  return {
    state: 'restarting',
    pid: 4242,
    url: 'http://127.0.0.1:5000/?token=secret-token',
    generation: 3,
    consecutiveCrashes: 2,
    lastExit: {
      verdict: 'crash',
      reason: 'killed by SIGABRT while loading /Users/someone/models/private',
      code: null,
      signal: 'SIGABRT',
      atMs: 1000,
      stderrTail: ['libc++abi: terminating due to uncaught exception: secret detail'],
    },
    health: { status: 'loading', models: { resident: ['private-model'] } },
    nativeErrors: [{ context: 'array_eval', detail: 'std::runtime_error: secret detail', observedAtMs: 900 }],
    traceFile: '/tmp/trace-secret.log',
    ...overrides,
  };
}

describe('supervisor state file', () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-supervisor-state-'));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it('projects a snapshot without reason, stderr or native detail', () => {
    const file = projectSupervisorState(snapshot(), 77, 5000);
    expect(file).toEqual({
      version: 1,
      writtenAt: 5000,
      mainPid: 77,
      state: 'restarting',
      pid: 4242,
      generation: 3,
      consecutiveCrashes: 2,
      lastExit: { verdict: 'crash', code: null, signal: 'SIGABRT', atMs: 1000 },
      healthStatus: 'loading',
      nativeErrorCount: 1,
      nativeErrorsCapped: false,
      lastNativeErrorContext: 'array_eval',
      lastNativeErrorAt: 900,
    } satisfies SupervisorStateFile);
    const text = JSON.stringify(file);
    for (const leaked of ['secret', 'private', 'token', 'trace', 'http']) expect(text).not.toContain(leaked);
  });

  it('rejects a native error context outside the safe charset', () => {
    for (const context of ['Array Eval', 'array-eval', 'a'.repeat(33), '', 'array_eval\n']) {
      const file = projectSupervisorState(
        snapshot({ nativeErrors: [{ context, detail: 'x', observedAtMs: 1 }] }),
        1,
        1,
      );
      expect(file.lastNativeErrorContext, JSON.stringify(context)).toBeNull();
      expect(file.nativeErrorCount).toBe(1);
    }
    expect(projectSupervisorState(snapshot({ nativeErrors: [] }), 1, 1)).toMatchObject({
      nativeErrorCount: 0,
      lastNativeErrorContext: null,
      lastNativeErrorAt: null,
    });
  });

  it('marks the native error count as capped at 20', () => {
    expect(MAX_NATIVE_ERRORS).toBe(20);
    const errors = Array.from({ length: MAX_NATIVE_ERRORS }, (_, i) => ({
      context: 'array_eval',
      detail: 'd',
      observedAtMs: i,
    }));
    expect(projectSupervisorState(snapshot({ nativeErrors: errors }), 1, 1)).toMatchObject({
      nativeErrorCount: 20,
      nativeErrorsCapped: true,
      lastNativeErrorAt: 19,
    });
    expect(projectSupervisorState(snapshot({ nativeErrors: errors.slice(1) }), 1, 1).nativeErrorsCapped).toBe(false);
  });

  it('coalesces writes to the latest state', async () => {
    const path = join(dir, 'desktop', 'supervisor.json');
    let release!: () => void;
    const firstWrite = new Promise<void>((resolve) => {
      release = resolve;
    });
    const written: string[] = [];
    const io: SupervisorStateWriterIo = {
      mkdir,
      writeFile: vi.fn(async (target: string, data: string) => {
        if (written.length === 0) {
          written.push(data);
          await firstWrite;
        } else written.push(data);
        await writeFile(target, data);
      }),
      rename,
      unlink,
    };
    const writer = createSupervisorStateWriter(path, io);
    const states = [1, 2, 3, 4].map((n) => projectSupervisorState(snapshot({ generation: n }), 1, n));
    for (const state of states) writer.write(state);
    release();
    for (let i = 0; i < 50 && written.length < 2; i++) await new Promise((r) => setImmediate(r));
    for (let i = 0; i < 50; i++) await new Promise((r) => setImmediate(r));

    // The first write was already in flight; of the three queued behind it only the newest is written.
    expect(written.map((data) => (JSON.parse(data) as SupervisorStateFile).generation)).toEqual([1, 4]);
    expect((JSON.parse(readFileSync(path, 'utf8')) as SupervisorStateFile).generation).toBe(4);
    expect(existsSync(`${path}.tmp`)).toBe(false);

    await writer.dispose();
    expect(existsSync(path)).toBe(false);
    writer.write(states[0]!);
    for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r));
    expect(existsSync(path)).toBe(false);
  });
});
