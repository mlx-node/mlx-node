import { EventEmitter } from 'node:events';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  collectAgentUpdateResult,
  createAgentUpdater,
  presentAgentUpdate,
  type AgentUpdateCheck,
  type AgentUpdateChild,
} from '../src/main/agent-updates.js';

function harness() {
  const run = vi.fn<(args: string[]) => Promise<AgentUpdateCheck>>();
  const onChange = vi.fn();
  const report = vi.fn();
  const updater = createAgentUpdater({ run, onChange, report });
  return { run, onChange, report, updater };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('presentAgentUpdate', () => {
  it('labels every status', () => {
    expect(presentAgentUpdate('idle').label).toBe('Check for MLX Agent Update…');
    expect(presentAgentUpdate('checking').enabled).toBe(false);
    expect(presentAgentUpdate('available', 'mlx-agent-v0.3.0').label).toBe('Install MLX Agent 0.3.0…');
    expect(presentAgentUpdate('available', null).label).toBe('Install MLX Agent Update…');
    expect(presentAgentUpdate('installing').label).toBe('Downloading MLX Agent Update…');
    expect(presentAgentUpdate('current').label).toContain('Up to Date');
    expect(presentAgentUpdate('error').label).toContain('Failed');
  });
});

describe('agent updater', () => {
  it('checks on start and reports an available update with its tag', async () => {
    const { run, updater } = harness();
    run.mockResolvedValue({ updateAvailable: true, latest: 'mlx-agent-v0.3.0' });
    updater.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(run).toHaveBeenCalledWith(['--check']);
    expect(updater.status()).toBe('available');
    expect(updater.latest()).toBe('mlx-agent-v0.3.0');
  });

  it('lands on current when nothing is newer', async () => {
    const { run, updater } = harness();
    run.mockResolvedValue({ updateAvailable: false, latest: 'mlx-agent-v0.2.0' });
    updater.check();
    await vi.advanceTimersByTimeAsync(0);
    expect(updater.status()).toBe('current');
  });

  it('installs only from available and returns to current', async () => {
    const { run, updater } = harness();
    run.mockResolvedValue({ updateAvailable: true, latest: 'mlx-agent-v0.3.0' });
    updater.check();
    await vi.advanceTimersByTimeAsync(0);
    // Older CLIs print human text only → the parsed result is {}.
    run.mockResolvedValue({});
    updater.install();
    expect(updater.status()).toBe('installing');
    await vi.advanceTimersByTimeAsync(0);
    expect(run).toHaveBeenLastCalledWith([]);
    expect(updater.status()).toBe('current');
    // A second install from current is a no-op — the menu routes it to check.
    updater.install();
    expect(run).toHaveBeenCalledTimes(2);
  });

  it('stays available when the install reports a fallback below latest', async () => {
    // A CLI that fell back to an older candidate ends its piped stdout with
    // updateAvailable: true — claiming 'current' would mask the shortfall
    // until the next periodic check.
    const { run, updater } = harness();
    run.mockResolvedValue({ updateAvailable: true, latest: 'mlx-agent-v0.3.0' });
    updater.check();
    await vi.advanceTimersByTimeAsync(0);
    run.mockResolvedValue({ updateAvailable: true, latest: 'mlx-agent-v0.3.0', current: 'mlx-agent-v0.2.0' });
    updater.install();
    await vi.advanceTimersByTimeAsync(0);
    expect(updater.status()).toBe('available');
    expect(updater.latest()).toBe('mlx-agent-v0.3.0');
  });

  it('serialises concurrent checks and installs', async () => {
    const { run, updater } = harness();
    const pending = Promise.withResolvers<AgentUpdateCheck>();
    run.mockReturnValue(pending.promise);
    updater.check();
    updater.check();
    updater.install();
    expect(run).toHaveBeenCalledTimes(1);
    pending.resolve({ updateAvailable: true, latest: 'mlx-agent-v0.3.0' });
    await vi.advanceTimersByTimeAsync(0);
    expect(updater.status()).toBe('available');
  });

  it('reports failures and permits retry', async () => {
    const { run, updater, report } = harness();
    const failure = new Error('mirror unreachable');
    run.mockRejectedValueOnce(failure);
    updater.check();
    await vi.advanceTimersByTimeAsync(0);
    expect(report).toHaveBeenCalledWith(failure);
    expect(updater.status()).toBe('error');
    run.mockResolvedValue({ updateAvailable: false });
    updater.check();
    await vi.advanceTimersByTimeAsync(0);
    expect(updater.status()).toBe('current');
  });

  it('stop freezes the interval and silences onChange', async () => {
    const { run, updater, onChange } = harness();
    run.mockResolvedValue({ updateAvailable: false });
    updater.start();
    await vi.advanceTimersByTimeAsync(0);
    updater.stop();
    onChange.mockClear();
    await vi.advanceTimersByTimeAsync(6 * 60 * 60 * 1000);
    expect(run).toHaveBeenCalledTimes(1);
    expect(onChange).not.toHaveBeenCalled();
  });
});

/**
 * Node can fire `exit` while piped `data` is still buffered, so the parse
 * must wait for stream close: an early parse sees an empty status (silent
 * "current") or a truncated JSON line (throw inside the event callback —
 * an uncaught exception, not a rejection).
 */
describe('collectAgentUpdateResult', () => {
  function fakeChild() {
    const child = new EventEmitter();
    const stdout = new EventEmitter();
    const stderr = new EventEmitter();
    const tap = Object.assign(child, { stdout, stderr }) as unknown as AgentUpdateChild & EventEmitter & {
      stdout: EventEmitter;
      stderr: EventEmitter;
    };
    return tap;
  }

  it('parses the status after exit when stdout still has data buffered', async () => {
    const child = fakeChild();
    const pending = collectAgentUpdateResult(child, true);
    // exit first, status after — the broken order this guards.
    child.emit('exit', 0);
    child.stdout.emit('data', '{"updateAvailable":true,"latest":"mlx-agent-v0.3.0"}');
    child.stderr.emit('close');
    child.stdout.emit('close');
    await expect(pending).resolves.toEqual({ updateAvailable: true, latest: 'mlx-agent-v0.3.0' });
  });

  it('rejects a --check that exits 0 with no status line', async () => {
    const child = fakeChild();
    const pending = collectAgentUpdateResult(child, true);
    child.emit('exit', 0);
    child.stdout.emit('close');
    child.stderr.emit('close');
    await expect(pending).rejects.toThrow('without a status line');
  });

  it('rejects malformed status JSON instead of throwing inside the event callback', async () => {
    const child = fakeChild();
    const pending = collectAgentUpdateResult(child, true);
    child.emit('exit', 0);
    child.stdout.emit('data', '{"updateAvailable":');
    child.stdout.emit('close');
    child.stderr.emit('close');
    await expect(pending).rejects.toThrow('malformed status');
  });

  it('resolves {} for an install whose output is human text only', async () => {
    const child = fakeChild();
    const pending = collectAgentUpdateResult(child, false);
    child.stdout.emit('data', 'mlx agent: updated to mlx-agent-v0.3.0 → /m/mlx-agent\n');
    child.stdout.emit('close');
    child.stderr.emit('close');
    child.emit('exit', 0);
    await expect(pending).resolves.toEqual({});
  });

  it('rejects a non-zero exit with the captured stderr', async () => {
    const child = fakeChild();
    const pending = collectAgentUpdateResult(child, true);
    child.stderr.emit('data', 'checksum mismatch');
    child.stderr.emit('close');
    child.stdout.emit('close');
    child.emit('exit', 1);
    await expect(pending).rejects.toThrow('exited 1: checksum mismatch');
  });
});
