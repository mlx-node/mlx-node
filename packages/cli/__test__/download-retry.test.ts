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
  it('retries what the server says is its own fault, a rate limit, and a request timeout', () => {
    // 408 rides with 429/5xx: RFC 9110 §15.5.9 makes it transient by
    // definition, and nothing in @huggingface/hub@2.13.2 retries it for us.
    for (const status of [500, 502, 503, 504, 429, 408]) {
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

  it('does NOT retry a local filesystem refusal', () => {
    // `downloadFileToCacheDir` mkdirs, streams to `<blob>.incomplete`, renames
    // and symlinks, wrapping none of it — so a full or unwritable cache volume
    // surfaces here as a plain Error carrying a Node errno and NO status.
    // Retrying is worse than useless: `.incomplete` is reopened with 'w' and
    // the GET carries no Range header, so each attempt re-downloads the shard
    // from byte 0 into the same full disk before reporting the real error.
    for (const [code, syscall] of [
      ['ENOSPC', 'write'],
      ['EDQUOT', 'write'],
      ['EACCES', 'mkdir'],
      ['EPERM', 'rename'],
      ['EROFS', 'mkdir'],
    ] as const) {
      const err = Object.assign(new Error(`${code}: failed, ${syscall}`), { code, syscall, errno: -1 });
      expect([code, isRetriableFetchError(err)]).toEqual([code, false]);
    }
  });

  it('still retries an errno that clears on its own', () => {
    // The over-correction guard on the list above: "too many open files" is a
    // transient local condition, not a refusal.
    const err = Object.assign(new Error('EMFILE: too many open files'), { code: 'EMFILE' });
    expect(isRetriableFetchError(err)).toBe(true);
  });

  it('retries a content-GET failure that arrives as a bare string', () => {
    // `WebBlob.stream()` / `XetBlob.stream()` abort their writable with
    // `error.message`, not the error — so a failure on the shard download
    // itself reaches us with no type (measured: `typeof e === 'string'`,
    // `e instanceof Error === false`). Reading only `Error` would refuse to
    // retry the single case this whole retry exists for.
    expect(isRetriableFetchError('fetch failed')).toBe(true);
    expect(isRetriableFetchError('Api error with status 503. URL: https://hf.co/x')).toBe(true);
    expect(isRetriableFetchError('Api error with status 500. URL: https://hf.co/x')).toBe(true);
    expect(isRetriableFetchError('Api error with status 408. URL: https://hf.co/x')).toBe(true);
    // …and the status is still honoured through the text, so a permanent
    // answer stays permanent even after being flattened.
    expect(isRetriableFetchError('Api error with status 403. URL: https://hf.co/x')).toBe(false);
    expect(isRetriableFetchError('Api error with status 404. URL: https://hf.co/x')).toBe(false);
  });

  it('cannot honour a status the hub client deleted, and retries instead', () => {
    // Documents a REAL looseness rather than an aspiration. `createApiError`
    // (@huggingface/hub@2.13.2, src/error.ts:15-26) writes the "Api error with
    // status N" prefix and then, when the body is `application/json`, replaces
    // the ENTIRE message with `json.error || json.message`, keeping only the
    // `. URL: …` trailer. Combined with WebBlob/XetBlob aborting on
    // `error.message` (a bare string), a JSON-bodied 403 on the content GET
    // arrives with no status anywhere and takes the default-retry branch.
    //
    // Not fixable without pattern-matching server prose — the allowlist this
    // predicate exists to avoid — and it errs toward retrying, which is the
    // safe direction. Pinned so it is a known cost, not a surprise.
    expect(isRetriableFetchError('Invalid credentials in Authorization header. URL: https://hf.co/x')).toBe(true);
    // The same status DOES stay permanent on the error-OBJECT path, which
    // keeps `statusCode` — so this is a property of the flattening, not of the
    // predicate's view of 403.
    expect(isRetriableFetchError(hubError(403))).toBe(false);
  });

  it('does NOT retry a payload the chunk reader refuses', () => {
    // `XetBlob` throws these from inside the chunk loop (XetBlob.ts:387,395)
    // after part of the shard has streamed. They carry no status and no errno,
    // so before this they took the default-retry branch and cost three more
    // full multi-GB transfers to fail identically on the same bytes.
    for (const text of ['Unsupported chunk version 3', 'Unsupported compression scheme ByteGroupingLZ4x']) {
      expect([text, isRetriableFetchError(text)]).toEqual([text, false]);
      expect([text, isRetriableFetchError(new Error(text))]).toEqual([text, false]);
    }
    // The over-correction guard: a TRUNCATED transfer looks similar and is the
    // transient case this retry exists for. It must stay retriable.
    expect(isRetriableFetchError('Failed to fetch all data for term abc123, fetched 4 bytes out of 99')).toBe(true);
  });

  it('retries transport failures that are not spelled "fetch failed"', () => {
    // Naming known network errors is a trap: `fetch` rejects `TypeError: fetch
    // failed` on connect/DNS but `TypeError: terminated` on a mid-body socket
    // reset — the dominant multi-GB shard failure — and an `AbortSignal.timeout`
    // rejection is a DOMException whose `code` is a NUMBER, not an errno.
    expect(isRetriableFetchError(new TypeError('terminated'))).toBe(true);
    expect(isRetriableFetchError(new DOMException('aborted due to timeout', 'TimeoutError'))).toBe(true);
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

  it('waits exactly 1s, then 2s, then 4s — not merely "some" delay', async () => {
    // `runAllTimersAsync` drains ANY finite delay, so every other test in this
    // file passes with `RETRY_BASE_MS = 0` (measured). That leaves the 1/2/4s
    // schedule — the thing that stops a retry storm from amplifying the CDN
    // failure it is reacting to — completely unguarded. This walks the clock
    // instead, asserting no attempt fires one millisecond early.
    const attempt = vi.fn(async () => {
      throw hubError(503);
    });
    const settled = withRetries('model.safetensors', attempt);
    const assertion = expect(settled).rejects.toMatchObject({ statusCode: 503 });

    // Attempt 1 runs immediately, with no timer scheduled before it.
    await vi.advanceTimersByTimeAsync(0);
    expect(attempt).toHaveBeenCalledTimes(1);

    for (const [n, waitMs] of [
      [2, 1_000],
      [3, 2_000],
      [4, 4_000],
    ] as const) {
      await vi.advanceTimersByTimeAsync(waitMs - 1);
      expect([n, attempt.mock.calls.length]).toEqual([n, n - 1]);
      await vi.advanceTimersByTimeAsync(1);
      expect([n, attempt.mock.calls.length]).toEqual([n, n]);
    }

    await assertion;
    // The pause is explained to the user with the real wait, not a guess.
    expect(warn).toHaveBeenCalledTimes(3);
    expect(warn.mock.calls.map((c: unknown[]) => String(c[0]))).toEqual([
      expect.stringContaining('retry 1/3 in 1000ms'),
      expect.stringContaining('retry 2/3 in 2000ms'),
      expect.stringContaining('retry 3/3 in 4000ms'),
    ]);
  });

  it('names the failure in the retry line, including a bare-string one', async () => {
    // Regression: `reason` read `.message` off the error, which is `undefined`
    // on a string — so the content-GET failures this whole retry exists for
    // logged `failed (undefined)` and hid their own cause.
    const attempt = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce('Api error with status 500. URL: https://hf.co/x')
      .mockResolvedValueOnce('/cache/blob');
    const settled = withRetries('model-00001-of-00009.safetensors', attempt);
    await vi.runAllTimersAsync();
    await expect(settled).resolves.toBe('/cache/blob');
    const line = String(warn.mock.calls[0][0]);
    expect(line).toContain('Api error with status 500');
    expect(line).not.toContain('undefined');
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

  it('does not sleep three times before reporting a full disk', async () => {
    // The user-visible cost of getting the predicate wrong: 7s of backoff and
    // three more doomed multi-GB downloads before the real cause is printed.
    const attempt = vi.fn(async () => {
      throw Object.assign(new Error('ENOSPC: no space left on device, write'), { code: 'ENOSPC' });
    });
    const settled = withRetries('model-00001-of-00009.safetensors', attempt);
    void vi.runAllTimersAsync();
    await expect(settled).rejects.toMatchObject({ code: 'ENOSPC' });
    expect(attempt).toHaveBeenCalledTimes(1);
    expect(warn).not.toHaveBeenCalled();
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
