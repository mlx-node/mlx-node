// Main-thread bridge for the Embeddings chapter (chapter 2).
//
// The MLX worker (packages/browser/src/mlx-worker.ts) accepts an `embed`
// message and replies with either `embedResult` or `embedError`, correlated
// by a caller-supplied id. This helper wraps that protocol in a Promise so
// the chapter widget can `await` a single call without reasoning about the
// worker's chat/stream surface.
//
// The shape mirrors `tokenizer-client.ts`. We attach a scoped `message`
// listener via `addEventListener` (the worker's main message handler is
// assigned through `worker.onmessage`, so listeners installed here run
// alongside without disturbing it), filter on id, then resolve / reject +
// unregister.

import {
  EMBED_ERROR_TYPE,
  EMBED_REQUEST_TYPE,
  EMBED_RESULT_TYPE,
  type EmbedRequest,
  type EmbedResult,
} from "../../src/inspector-types";

const DEFAULT_TIMEOUT_MS = 60_000;

function makeAbortError(): DOMException {
  return new DOMException("Embed aborted", "AbortError");
}

function nextEmbedId(): string {
  const cryptoObj = globalThis.crypto as
    | { randomUUID?: () => string }
    | undefined;
  if (cryptoObj?.randomUUID) {
    return cryptoObj.randomUUID();
  }
  return `embed-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export type EmbedClientOptions = {
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

export type EmbedClientResult = {
  embeddings: Float32Array;
  hiddenDim: number;
};

/**
 * Send an `embed` request to the MLX worker and await the resulting
 * `[tokenIds.length, hiddenDim]` flat Float32Array. Rejects with a clear
 * message if the worker reports an error (e.g. "Model not loaded"), if no
 * reply arrives within the timeout, if the abort signal fires, or if
 * `worker` is null.
 */
export function embed(
  worker: Worker | null,
  tokenIds: number[],
  options?: EmbedClientOptions,
): Promise<EmbedClientResult> {
  if (!worker) {
    return Promise.reject(new Error("MLX worker is not available"));
  }
  const signal = options?.signal;
  if (signal?.aborted) {
    return Promise.reject(makeAbortError());
  }
  const id = nextEmbedId();
  const timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  return new Promise<EmbedClientResult>((resolve, reject) => {
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

    const settleResolve = (value: EmbedClientResult) => {
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
      const msg = event.data as EmbedResult | undefined;
      if (!msg || typeof msg !== "object") return;
      if (msg.type === EMBED_RESULT_TYPE && msg.id === id) {
        settleResolve({ embeddings: msg.embeddings, hiddenDim: msg.hiddenDim });
      } else if (msg.type === EMBED_ERROR_TYPE && msg.id === id) {
        settleReject(new Error(msg.error || "Embed failed"));
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
      settleReject(new Error(`Embed timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    const request: EmbedRequest = {
      type: EMBED_REQUEST_TYPE,
      id,
      tokenIds,
    };

    try {
      worker.postMessage(request);
    } catch (err) {
      settleReject(err instanceof Error ? err : new Error(String(err)));
    }
  });
}
