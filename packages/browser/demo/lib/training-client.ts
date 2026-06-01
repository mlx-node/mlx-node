// Main-thread bridge for the Education App "Training" playground (Chapter 13).
//
// The MLX worker (packages/browser/src/mlx-worker.ts) accepts `trainInit`,
// `trainStep`, `trainSample`, and `trainReset` messages and replies with a
// matching `*Result` / `*Error` message correlated by a caller-supplied id.
// This helper wraps that protocol in Promises so the playground widget can
// `await` each call without reasoning about the worker's message surface.
//
// The pattern mirrors inspector-client.ts exactly: attach a scoped `message`
// listener via `addEventListener` (the worker's main handler is assigned
// through `worker.onmessage`, so listeners installed here run alongside without
// disturbing it), filter on the correlation id, then resolve / reject and
// unregister. Like the inspector client, each call accepts an AbortSignal so a
// caller that tears down the worker rejects in-flight Promises within a tick
// instead of waiting out the timeout.

import {
  SCATTER_SELF_TEST_ERROR_TYPE,
  SCATTER_SELF_TEST_REQUEST_TYPE,
  SCATTER_SELF_TEST_RESULT_TYPE,
  TRAIN_INIT_ERROR_TYPE,
  TRAIN_INIT_REQUEST_TYPE,
  TRAIN_INIT_RESULT_TYPE,
  TRAIN_RESET_ERROR_TYPE,
  TRAIN_RESET_REQUEST_TYPE,
  TRAIN_RESET_RESULT_TYPE,
  TRAIN_SAMPLE_ERROR_TYPE,
  TRAIN_SAMPLE_REQUEST_TYPE,
  TRAIN_SAMPLE_RESULT_TYPE,
  TRAIN_STEP_ERROR_TYPE,
  TRAIN_STEP_REQUEST_TYPE,
  TRAIN_STEP_RESULT_TYPE,
  type TinyTrainerConfig,
  type TrainingRequest,
} from '../../src/training-types';

// A single train step is fast (tiny model, short sequence), but the very first
// trainInit kicks off weight init + an MLX seed call, and the device may still
// be warming. 30s is generous headroom; per-step calls settle in well under it.
const DEFAULT_TIMEOUT_MS = 30_000;

export type { TinyTrainerConfig };

export type TrainingClientOptions = {
  /** Override the request timeout (ms). Defaults to 30s. */
  timeoutMs?: number;
  /**
   * Cancel the in-flight call early. Aborting rejects the returned Promise
   * with an `AbortError` DOMException. If the signal is already aborted at
   * call time we reject synchronously without posting the request — important
   * because the worker may be torn down on cleanup, after which no reply ever
   * arrives.
   */
  signal?: AbortSignal;
};

function makeAbortError(): DOMException {
  return new DOMException('Training run aborted', 'AbortError');
}

function nextTrainingId(): string {
  const cryptoObj = globalThis.crypto as { randomUUID?: () => string } | undefined;
  if (cryptoObj?.randomUUID) {
    return cryptoObj.randomUUID();
  }
  return `training-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Post one training request and await the correlated reply.
 *
 * `resultType` / `errorType` are the worker reply `type` strings for this
 * request; `extract` pulls the resolved value out of the (validated) result
 * message. Rejects on the error reply, on timeout, on abort, or if `worker`
 * is null.
 */
function request<TResult, TReq extends Omit<TrainingRequest, 'id'>>(
  worker: Worker | null,
  message: TReq,
  resultType: string,
  errorType: string,
  extract: (msg: Record<string, unknown>) => TResult,
  options?: TrainingClientOptions,
): Promise<TResult> {
  if (!worker) {
    return Promise.reject(new Error('MLX worker is not available'));
  }
  const signal = options?.signal;
  if (signal?.aborted) {
    return Promise.reject(makeAbortError());
  }
  const id = nextTrainingId();
  const timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  return new Promise<TResult>((resolve, reject) => {
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

    const settleResolve = (value: TResult) => {
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
      const msg = event.data as Record<string, unknown> | undefined;
      if (!msg || typeof msg !== 'object') return;
      if (msg.id !== id) return;
      if (msg.type === resultType) {
        settleResolve(extract(msg));
      } else if (msg.type === errorType) {
        const error = typeof msg.error === 'string' ? msg.error : 'Training request failed';
        settleReject(new Error(error));
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
      settleReject(new Error(`Training request "${message.type}" timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    try {
      worker.postMessage({ ...message, id });
    } catch (err) {
      settleReject(err instanceof Error ? err : new Error(String(err)));
    }
  });
}

/** Construct (or replace) the tiny trainer in the worker. */
export function trainInit(
  worker: Worker | null,
  config: TinyTrainerConfig,
  seed: number,
  learningRate?: number,
  options?: TrainingClientOptions,
): Promise<void> {
  return request(
    worker,
    {
      type: TRAIN_INIT_REQUEST_TYPE,
      config,
      seed,
      ...(learningRate !== undefined ? { learningRate } : {}),
    },
    TRAIN_INIT_RESULT_TYPE,
    TRAIN_INIT_ERROR_TYPE,
    () => undefined,
    options,
  );
}

/** Run one teacher-forced training step; resolves with the new step + loss. */
export function trainStep(
  worker: Worker | null,
  tokenIds: number[],
  options?: TrainingClientOptions,
): Promise<{ step: number; loss: number }> {
  return request(
    worker,
    { type: TRAIN_STEP_REQUEST_TYPE, tokenIds },
    TRAIN_STEP_RESULT_TYPE,
    TRAIN_STEP_ERROR_TYPE,
    (msg) => ({ step: Number(msg.step), loss: Number(msg.loss) }),
    options,
  );
}

/** Autoregressively sample `n` token ids continuing `promptIds`. */
export function trainSample(
  worker: Worker | null,
  promptIds: number[],
  n: number,
  temperature: number,
  options?: TrainingClientOptions,
): Promise<{ ids: number[] }> {
  return request(
    worker,
    { type: TRAIN_SAMPLE_REQUEST_TYPE, promptIds, n, temperature },
    TRAIN_SAMPLE_RESULT_TYPE,
    TRAIN_SAMPLE_ERROR_TYPE,
    (msg) => ({ ids: Array.isArray(msg.ids) ? (msg.ids as number[]) : [] }),
    options,
  );
}

/** Re-initialize the trainer's weights + optimizer from `seed`. */
export function trainReset(worker: Worker | null, seed: number, options?: TrainingClientOptions): Promise<void> {
  return request(
    worker,
    { type: TRAIN_RESET_REQUEST_TYPE, seed },
    TRAIN_RESET_RESULT_TYPE,
    TRAIN_RESET_ERROR_TYPE,
    () => undefined,
    options,
  );
}

/**
 * Run the WebGPU scatter_add kernel-correctness self-test on the worker and
 * resolve with its raw length-5 result. A correct accumulating kernel yields
 * [0,0,0,3,0]; a broken last-write-wins kernel yields [0,0,0,1,0]. Needs only
 * the device up (no trainer / big model). Same id-correlation, abort, and
 * timeout behavior as the train* calls.
 */
export function scatterSelfTest(worker: Worker | null, options?: TrainingClientOptions): Promise<number[]> {
  return request(
    worker,
    { type: SCATTER_SELF_TEST_REQUEST_TYPE },
    SCATTER_SELF_TEST_RESULT_TYPE,
    SCATTER_SELF_TEST_ERROR_TYPE,
    (msg) => (Array.isArray(msg.result) ? (msg.result as number[]) : []),
    options,
  );
}
