/**
 * Transient-failure retry for `mlx download model`.
 *
 * The regression these guard: a checkpoint is tens of files and tens of
 * gigabytes pulled from a CDN, and the downloader had NO retry at all — one
 * 5xx anywhere in the set aborted the whole download and discarded every
 * completed file. CI hit exactly that on the gemma4 e2e leg:
 *
 *   HubApiError: Api error with status 500
 *     data: { message: 'Key service error: Timeout occurred while creating a new object' }
 *
 * Driven on fake timers: the real backoff sleeps 1s/2s/4s, so an exhaustion
 * test would otherwise cost 7 seconds of wall clock to assert arithmetic.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vite-plus/test';

import { isRetriableFetchError, withRetries } from '../src/commands/download-model.js';

/** A `HubApiError`-shaped failure: what `@huggingface/hub` actually throws. */
function hubError(statusCode: number): Error {
  return Object.assign(new Error(`Api error with status ${statusCode}`), { statusCode });
}

describe('isRetriableFetchError', () => {
  it('retries what the server says is its own fault, and a rate limit', () => {
    for (const status of [500, 502, 503, 504, 429]) {
      expect([status, isRetriableFetchError(hubError(status))]).toEqual([status, true]);
    }
  });

  it('does NOT retry a settled answer', () => {
    // Repeating these only delays the message the user needs: 401/403 is a
    // missing token or no access to a gated repo, 404 is the wrong repo or
    // revision. None of them changes on a second ask.
    for (const status of [400, 401, 403, 404, 416]) {
      expect([status, isRetriableFetchError(hubError(status))]).toEqual([status, false]);
    }
  });

  it('retries a transport failure, which carries no status at all', () => {
    // Socket reset, DNS, TLS, timeout — `fetch` rejects with a plain Error.
    expect(isRetriableFetchError(new Error('fetch failed'))).toBe(true);
    expect(isRetriableFetchError(undefined)).toBe(false);
    expect(isRetriableFetchError(null)).toBe(false);
  });
});

describe('withRetries', () => {
  let warn: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.useFakeTimers();
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.useRealTimers();
    warn.mockRestore();
  });

  it('returns the first success without sleeping', async () => {
    const attempt = vi.fn(async () => 'ok');
    await expect(withRetries('f', attempt)).resolves.toBe('ok');
    expect(attempt).toHaveBeenCalledTimes(1);
    expect(warn).not.toHaveBeenCalled();
  });

  it('survives a transient 500 and returns the retry’s result', async () => {
    // The CI failure, in one test: the first pull 500s, the second lands.
    const attempt = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(hubError(500))
      .mockResolvedValueOnce('/cache/blob');
    const settled = withRetries('model.safetensors', attempt);
    await vi.runAllTimersAsync();
    await expect(settled).resolves.toBe('/cache/blob');
    expect(attempt).toHaveBeenCalledTimes(2);
  });

  it('gives up after a bounded number of attempts rather than looping forever', async () => {
    const attempt = vi.fn(async () => {
      throw hubError(503);
    });
    const settled = withRetries('model.safetensors', attempt);
    const assertion = expect(settled).rejects.toMatchObject({ statusCode: 503 });
    await vi.runAllTimersAsync();
    await assertion;
    // Bounded: the last failure is rethrown, not swallowed into a hang.
    expect(attempt).toHaveBeenCalledTimes(4);
  });

  it('fails a permanent error on the FIRST attempt, with no backoff', async () => {
    // The over-correction guard. A retry loop that cannot tell 403 from 500
    // turns "you have no access to this repo" into the same message three
    // sleeps later, and hides it behind retry chatter.
    const attempt = vi.fn(async () => {
      throw hubError(403);
    });
    const settled = withRetries('model.safetensors', attempt);
    // Drain any backoff this SHOULD not have scheduled. Correct code rejects
    // before a timer exists, so this is a no-op; a version that retries 403
    // then fails the count below in milliseconds instead of hanging until the
    // suite timeout, which is the difference between a diagnosis and a stall.
    void vi.runAllTimersAsync();
    await expect(settled).rejects.toMatchObject({ statusCode: 403 });
    expect(attempt).toHaveBeenCalledTimes(1);
    expect(warn).not.toHaveBeenCalled();
  });
});
