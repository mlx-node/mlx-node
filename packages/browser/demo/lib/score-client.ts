// Main-thread bridge for the speculative-decoding scoreTokens hook.
//
// The MLX worker (packages/browser/src/mlx-worker.ts) accepts a
// `scoreTokens` message and replies with either `scoreTokensResult` or
// `scoreTokensError`, correlated by a caller-supplied id. This helper wraps
// that protocol in a Promise so chapter components can `await` a single call
// without reasoning about the worker's chat/stream message surface.
//
// The protocol mirrors `runForInspector` in inspector-client.ts: we attach a
// scoped `message` listener via `addEventListener` (the worker's main message
// handler is assigned through `worker.onmessage`, so listeners installed here
// run alongside without disturbing it), filter on id, then resolve / reject +
// unregister.

import {
  SCORE_TOKENS_ERROR_TYPE,
  SCORE_TOKENS_REQUEST_TYPE,
  SCORE_TOKENS_RESULT_TYPE,
  type ScoreTokensRequest,
  type ScoreTokensResponse,
  type ScoreTokensRun,
} from '../../src/inspector-types';

const DEFAULT_TIMEOUT_MS = 60_000;

function makeAbortError(): DOMException {
  return new DOMException('scoreTokens run aborted', 'AbortError');
}

function nextScoreId(): string {
  const cryptoObj = globalThis.crypto as { randomUUID?: () => string } | undefined;
  if (cryptoObj?.randomUUID) {
    return cryptoObj.randomUUID();
  }
  return `score-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export type ScoreClientOptions = {
  /** Override the request timeout (ms). Defaults to 60s. */
  timeoutMs?: number;
  /**
   * Cancel the in-flight call early. Rejecting on the signal short-circuits
   * the 60s timeout — important because callers (e.g. the load-model effect)
   * terminate the worker on cleanup, after which no reply will ever arrive.
   *
   * Aborting rejects the returned Promise with an `AbortError` DOMException.
   * If the signal is already aborted at call time, we reject synchronously
   * without ever posting the request.
   */
  signal?: AbortSignal;
};

/**
 * Send a `scoreTokens` request to the MLX worker and await the result: ONE
 * forward pass over `[prefixIds + draftIds]` returning per-position top-K
 * logits for the D draft positions. The accept rule is the caller's:
 * `accept[i] ⇔ positions[i].argmaxTokenId === positions[i].draftTokenId`.
 *
 * Rejects with a clear message if the worker reports an error, if no reply
 * arrives within the timeout, if the abort signal fires, or if `worker` is
 * null.
 */
export function scoreTokens(
  worker: Worker | null,
  req: Omit<ScoreTokensRequest, 'type' | 'id'>,
  options?: ScoreClientOptions,
): Promise<ScoreTokensRun> {
  if (!worker) {
    return Promise.reject(new Error('MLX worker is not available'));
  }
  const signal = options?.signal;
  if (signal?.aborted) {
    return Promise.reject(makeAbortError());
  }
  const id = nextScoreId();
  const timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  return new Promise<ScoreTokensRun>((resolve, reject) => {
    let settled = false;
    let timeoutHandle: ReturnType<typeof setTimeout> | null = null;

    const cleanup = () => {
      worker.removeEventListener('message', onMessage);
      if (timeoutHandle != null) {
        clearTimeout(timeoutHandle);
        timeoutHandle = null;
      }
      if (signal) {
        signal.removeEventListener('abort', onAbort);
      }
    };

    const settleResolve = (value: ScoreTokensRun) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(value);
    };

    const settleReject = (err: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err);
    };

    const onMessage = (event: MessageEvent) => {
      const msg = event.data as ScoreTokensResponse | undefined;
      if (!msg || typeof msg !== 'object') return;
      if (msg.type === SCORE_TOKENS_RESULT_TYPE && msg.id === id) {
        settleResolve(msg.result);
      } else if (msg.type === SCORE_TOKENS_ERROR_TYPE && msg.id === id) {
        settleReject(new Error(msg.error || 'scoreTokens run failed'));
      }
    };

    const onAbort = () => {
      settleReject(makeAbortError());
    };

    worker.addEventListener('message', onMessage);
    if (signal) {
      signal.addEventListener('abort', onAbort);
    }

    timeoutHandle = setTimeout(() => {
      settleReject(new Error(`scoreTokens run timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    const request: ScoreTokensRequest = {
      type: SCORE_TOKENS_REQUEST_TYPE,
      id,
      prefixIds: req.prefixIds,
      draftIds: req.draftIds,
      topK: req.topK,
    };

    try {
      worker.postMessage(request);
    } catch (err) {
      settleReject(err instanceof Error ? err : new Error(String(err)));
    }
  });
}
