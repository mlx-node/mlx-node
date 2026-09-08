import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { CI_POLL_INTERVAL_MS, CI_WAIT_TIMEOUT_MS, waitForReleaseCI } from '../scripts/release-ci.js';

const commit = 'a'.repeat(40);
const successful = {
  id: 10,
  head_sha: commit,
  head_branch: 'main',
  event: 'push',
  path: '.github/workflows/ci.yml',
  status: 'completed',
  conclusion: 'success',
  html_url: 'https://github.com/mlx-node/mlx-node/actions/runs/10',
};
const response = (...runs: object[]): string => JSON.stringify({ workflow_runs: runs });

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('release CI gate', () => {
  it('waits for the exact main-push workflow run to appear and finish', async () => {
    const github = vi
      .fn()
      .mockReturnValueOnce(
        response(
          { ...successful, head_sha: 'b'.repeat(40) },
          { ...successful, event: 'pull_request' },
          { ...successful, head_branch: 'feature' },
          { ...successful, path: '.github/workflows/desktop-release.yml' },
        ),
      )
      .mockReturnValueOnce(response({ ...successful, status: 'in_progress', conclusion: null }))
      .mockReturnValue(response(successful));
    const done = vi.fn();
    const waiting = waitForReleaseCI(commit, github).then(done);
    expect(done).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(CI_POLL_INTERVAL_MS);
    expect(done).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(CI_POLL_INTERVAL_MS);
    await waiting;
    expect(done).toHaveBeenCalledTimes(1);
    expect(github).toHaveBeenCalledWith([
      'api',
      `repos/{owner}/{repo}/actions/workflows/ci.yml/runs?head_sha=${commit}&branch=main&event=push&per_page=100`,
    ]);
  });

  it.each(['failure', 'cancelled', 'timed_out', 'action_required', 'skipped', 'neutral', null])(
    'rejects a completed run with conclusion %s',
    async (conclusion) => {
      const github = vi.fn(() => response({ ...successful, conclusion }));
      await expect(waitForReleaseCI(commit, github)).rejects.toThrow('Release blocked');
      expect(vi.getTimerCount()).toBe(0);
    },
  );

  it('does not reuse an older success when the newest run failed', async () => {
    const github = vi.fn(() => response(successful, { ...successful, id: 11, conclusion: 'failure' }));
    await expect(waitForReleaseCI(commit, github)).rejects.toThrow('Release blocked');
  });

  it('waits for a rerun even if its previous attempt succeeded', async () => {
    const github = vi
      .fn()
      .mockReturnValueOnce(response({ ...successful, status: 'queued' }))
      .mockReturnValue(response(successful));
    const done = vi.fn();
    const waiting = waitForReleaseCI(commit, github).then(done);
    expect(done).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(CI_POLL_INTERVAL_MS);
    await waiting;
    expect(done).toHaveBeenCalledTimes(1);
  });

  it.each(['missing', 'pending'])('times out when CI remains %s', async (state) => {
    const github = vi.fn(() => (state === 'missing' ? response() : response({ ...successful, status: 'in_progress' })));
    const rejection = expect(waitForReleaseCI(commit, github)).rejects.toThrow('release remains a draft');
    await vi.advanceTimersByTimeAsync(CI_WAIT_TIMEOUT_MS);
    await rejection;
    expect(vi.getTimerCount()).toBe(0);
  });

  it('fails closed on API errors', async () => {
    const github = vi.fn(() => {
      throw new Error('API unavailable');
    });
    await expect(waitForReleaseCI(commit, github)).rejects.toThrow('API unavailable');
  });

  it('requires the full tagged commit SHA', async () => {
    const github = vi.fn();
    await expect(waitForReleaseCI('main', github)).rejects.toThrow('full commit SHA');
    expect(github).not.toHaveBeenCalled();
  });
});
