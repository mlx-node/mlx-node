import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { createAgentUpdater, presentAgentUpdate, type AgentUpdateCheck } from '../src/main/agent-updates.js';

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
    updater.install();
    expect(updater.status()).toBe('installing');
    run.mockResolvedValue({});
    await vi.advanceTimersByTimeAsync(0);
    expect(run).toHaveBeenLastCalledWith([]);
    expect(updater.status()).toBe('current');
    // A second install from current is a no-op — the menu routes it to check.
    updater.install();
    expect(run).toHaveBeenCalledTimes(2);
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
