// Main-thread bridge for the interpretability lensReadout hook.
//
// The MLX worker (packages/browser/src/mlx-worker.ts) accepts a
// `lensReadout` message and replies with either `lensReadoutResult` or
// `lensReadoutError`, correlated by a caller-supplied id. This helper wraps
// that protocol in a Promise so chapter components can `await` a single call
// without reasoning about the worker's chat/stream message surface.
//
// The protocol mirrors `scoreTokens` in score-client.ts: we attach a scoped
// `message` listener via `addEventListener` (the worker's main message handler
// is assigned through `worker.onmessage`, so listeners installed here run
// alongside without disturbing it), filter on id, then resolve / reject +
// unregister.

import {
  LENS_READOUT_ERROR_TYPE,
  LENS_READOUT_REQUEST_TYPE,
  LENS_READOUT_RESULT_TYPE,
  LOAD_LENS_PACK_ERROR_TYPE,
  LOAD_LENS_PACK_REQUEST_TYPE,
  LOAD_LENS_PACK_RESULT_TYPE,
  type LensReadoutRequest,
  type LensReadoutResponse,
  type LensReadoutRun,
  type LoadLensPackRequest,
  type LoadLensPackResponse,
} from '../../src/inspector-types';

const DEFAULT_TIMEOUT_MS = 60_000;

function makeAbortError(): DOMException {
  return new DOMException('lensReadout run aborted', 'AbortError');
}

function nextLensId(): string {
  const cryptoObj = globalThis.crypto as { randomUUID?: () => string } | undefined;
  if (cryptoObj?.randomUUID) {
    return cryptoObj.randomUUID();
  }
  return `lens-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export type LensClientOptions = {
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
 * Send a `lensReadout` request to the MLX worker and await the result: ONE
 * forward pass over `promptIds` returning a `layers × positions` grid of
 * per-cell top-K readouts plus per-pinned-token full-vocab rank tracks.
 * `useJacobian` is forwarded verbatim — `false` is the plain logit lens (the
 * browser ships no pack); a `true` with a non-final layer and no pack surfaces
 * the backend's naming error, never a silent downgrade.
 *
 * Rejects with a clear message if the worker reports an error, if no reply
 * arrives within the timeout, if the abort signal fires, or if `worker` is
 * null.
 */
export function lensReadout(
  worker: Worker | null,
  req: Omit<LensReadoutRequest, 'type' | 'id'>,
  options?: LensClientOptions,
): Promise<LensReadoutRun> {
  if (!worker) {
    return Promise.reject(new Error('MLX worker is not available'));
  }
  const signal = options?.signal;
  if (signal?.aborted) {
    return Promise.reject(makeAbortError());
  }
  const id = nextLensId();
  const timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  return new Promise<LensReadoutRun>((resolve, reject) => {
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

    const settleResolve = (value: LensReadoutRun) => {
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
      const msg = event.data as LensReadoutResponse | undefined;
      if (!msg || typeof msg !== 'object') return;
      if (msg.type === LENS_READOUT_RESULT_TYPE && msg.id === id) {
        settleResolve(msg.result);
      } else if (msg.type === LENS_READOUT_ERROR_TYPE && msg.id === id) {
        settleReject(new Error(msg.error || 'lensReadout run failed'));
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
      settleReject(new Error(`lensReadout run timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    const request: LensReadoutRequest = {
      type: LENS_READOUT_REQUEST_TYPE,
      id,
      promptIds: req.promptIds,
      layers: req.layers,
      topK: req.topK,
      pinnedIds: req.pinnedIds,
      useJacobian: req.useJacobian,
      noTile: req.noTile,
    };

    try {
      worker.postMessage(request);
    } catch (err) {
      settleReject(err instanceof Error ? err : new Error(String(err)));
    }
  });
}

/** Result of a `loadLensPack` call: how many Jacobians the backend loaded, and
 *  whether the pack was already resident (idempotent load-once → a repeat call
 *  resolves immediately with `loaded: 0, alreadyLoaded: true`). */
export type LoadLensPackResult = { loaded: number; alreadyLoaded: boolean };

// The pack is ~46 MiB (23 Jacobians): a cold fetch + GPU upload can exceed the
// 60s readout budget, so default higher. Still overridable / abortable.
const DEFAULT_LOAD_PACK_TIMEOUT_MS = 180_000;

/**
 * Send a `loadLensPack` request to the MLX worker: fetch the shipped f16 J-lens
 * pack from the same model base as the weights, upload its tensors to WebGPU,
 * and populate the model thread's `lens_pack` so a subsequent
 * `lensReadout({ useJacobian: true })` applies the fitted Jacobians instead of
 * erroring. Idempotent — a second call after a successful load resolves
 * immediately (`alreadyLoaded: true`). Mirrors {@link lensReadout}'s protocol:
 * scoped `message` listener, id correlation, timeout + abort.
 *
 * Rejects with a clear message if the worker reports an error (e.g. the pack is
 * missing / a 404), if no reply arrives within the timeout, if the abort signal
 * fires, or if `worker` is null.
 */
export function loadLensPack(worker: Worker | null, options?: LensClientOptions): Promise<LoadLensPackResult> {
  if (!worker) {
    return Promise.reject(new Error('MLX worker is not available'));
  }
  const signal = options?.signal;
  if (signal?.aborted) {
    return Promise.reject(makeAbortError());
  }
  const id = nextLensId();
  const timeoutMs = options?.timeoutMs ?? DEFAULT_LOAD_PACK_TIMEOUT_MS;

  return new Promise<LoadLensPackResult>((resolve, reject) => {
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

    const settleResolve = (value: LoadLensPackResult) => {
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
      const msg = event.data as LoadLensPackResponse | undefined;
      if (!msg || typeof msg !== 'object') return;
      if (msg.type === LOAD_LENS_PACK_RESULT_TYPE && msg.id === id) {
        settleResolve({ loaded: msg.loaded, alreadyLoaded: msg.alreadyLoaded });
      } else if (msg.type === LOAD_LENS_PACK_ERROR_TYPE && msg.id === id) {
        settleReject(new Error(msg.error || 'loadLensPack failed'));
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
      settleReject(new Error(`loadLensPack timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    const request: LoadLensPackRequest = {
      type: LOAD_LENS_PACK_REQUEST_TYPE,
      id,
    };

    try {
      worker.postMessage(request);
    } catch (err) {
      settleReject(err instanceof Error ? err : new Error(String(err)));
    }
  });
}
