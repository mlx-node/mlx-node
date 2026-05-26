// Main-thread bridge for the Tokenization chapter (chapter 1).
//
// The MLX worker (packages/browser/src/mlx-worker.ts) accepts a `tokenize`
// message and replies with either `tokenizeResult` or `tokenizeError`,
// correlated by a caller-supplied id. This helper wraps that protocol in a
// Promise so the chapter widget can `await` a single call without reasoning
// about the worker's chat/stream surface.
//
// The shape closely mirrors `inspector-client.ts`. We attach a scoped
// `message` listener via `addEventListener` (the worker's main message
// handler is assigned through `worker.onmessage`, so listeners installed here
// run alongside without disturbing it), filter on id, then resolve / reject +
// unregister.

import {
  TOKENIZE_ERROR_TYPE,
  TOKENIZE_REQUEST_TYPE,
  TOKENIZE_RESULT_TYPE,
  type TokenInfo,
  type TokenizeRequest,
  type TokenizeResult,
} from "../../src/inspector-types";

const DEFAULT_TIMEOUT_MS = 60_000;

function makeAbortError(): DOMException {
  return new DOMException("Tokenize aborted", "AbortError");
}

function nextTokenizeId(): string {
  const cryptoObj = globalThis.crypto as
    | { randomUUID?: () => string }
    | undefined;
  if (cryptoObj?.randomUUID) {
    return cryptoObj.randomUUID();
  }
  return `tokenize-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export type TokenizerClientOptions = {
  /** Override the request timeout (ms). Defaults to 60s. */
  timeoutMs?: number;
  /**
   * Cancel the in-flight call early. Rejecting on the signal short-circuits
   * the 60s timeout — important when the worker is terminated (model swap /
   * reload), after which no reply will ever arrive.
   *
   * Aborting rejects the returned Promise with an `AbortError` DOMException.
   * If the signal is already aborted at call time, we reject synchronously
   * without ever posting the request.
   */
  signal?: AbortSignal;
};

/**
 * Send a `tokenize` request to the MLX worker and await the resulting
 * `TokenInfo[]`. Rejects with a clear message if the worker reports an
 * error (e.g. "Model not loaded"), if no reply arrives within the timeout,
 * if the abort signal fires, or if `worker` is null.
 */
export function tokenize(
  worker: Worker | null,
  prompt: string,
  options?: TokenizerClientOptions,
): Promise<TokenInfo[]> {
  if (!worker) {
    return Promise.reject(new Error("MLX worker is not available"));
  }
  const signal = options?.signal;
  if (signal?.aborted) {
    return Promise.reject(makeAbortError());
  }
  const id = nextTokenizeId();
  const timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  return new Promise<TokenInfo[]>((resolve, reject) => {
    let settled = false;
    let timeoutHandle: ReturnType<typeof setTimeout> | null = null;

    const cleanup = () => {
      worker.removeEventListener("message", onMessage);
      if (timeoutHandle != null) {
        clearTimeout(timeoutHandle);
        timeoutHandle = null;
      }
      if (signal) {
        signal.removeEventListener("abort", onAbort);
      }
    };

    const settleResolve = (value: TokenInfo[]) => {
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
      const msg = event.data as TokenizeResult | undefined;
      if (!msg || typeof msg !== "object") return;
      if (msg.type === TOKENIZE_RESULT_TYPE && msg.id === id) {
        settleResolve(msg.tokens);
      } else if (msg.type === TOKENIZE_ERROR_TYPE && msg.id === id) {
        settleReject(new Error(msg.error || "Tokenize failed"));
      }
    };

    const onAbort = () => {
      settleReject(makeAbortError());
    };

    worker.addEventListener("message", onMessage);
    if (signal) {
      signal.addEventListener("abort", onAbort);
    }

    timeoutHandle = setTimeout(() => {
      settleReject(new Error(`Tokenize timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    const request: TokenizeRequest = {
      type: TOKENIZE_REQUEST_TYPE,
      id,
      prompt,
    };

    try {
      worker.postMessage(request);
    } catch (err) {
      settleReject(err instanceof Error ? err : new Error(String(err)));
    }
  });
}
