/**
 * Stage-0 real-model concurrency regression suite.
 *
 * Enable with an already-converted Qwen3 checkpoint:
 *
 *   QWEN3_STAGE0_MODEL_PATH=/abs/path/to/qwen3-mlx-bf16 \
 *   MLX_PAGED_PREFILL_CHUNK_SIZE=512 \
 *     yarn test __test__/server/concurrent-stage0-e2e.test.ts
 *
 * The suite is intentionally env-gated, sequential, and single-resident:
 * every case shares ONE Qwen3 instance and ONE maxQueueDepthPerModel=1 server
 * so a model-e2e runner never loads competing copies onto its single Metal
 * GPU. Unset QWEN3_STAGE0_MODEL_PATH skips cleanly. A set but invalid path
 * fails in beforeAll rather than silently deleting CI coverage.
 *
 * Named regression mutations (all must make their matching oracle fail):
 *
 * - H1a ASYNC_RESET: restore resetCaches' synchronous NAPI wait. The 50 ms
 *   tick probe then observes a >=500 ms event-loop gap while reset is queued.
 * - H1b NO_PREFILL_CHECKPOINTS: run with
 *   MLX_PAGED_PREFILL_CHUNK_SIZE=0. The aborted long prefill cannot stop at a
 *   chunk boundary, so reset + its deterministic successor miss the deadline.
 * - H2 DROP_NONSTREAM_CANCEL: hide the model's three
 *   chatSession*Cancellable methods in instrumentModel. The destroyed holder
 *   keeps the successor visibly queued and it misses the completion deadline.
 * - H3 CAP_TWO: construct the server with maxQueueDepthPerModel=2. The third
 *   request is admitted instead of returning the exact queue_full 429.
 * - H4 WRITE_TRUE: return the delegate write() result from the forced-false
 *   hook. The endpoint consumes more real model events instead of parking.
 *
 * H4 deliberately does NOT assert a fixed RSS delta. Allocator reuse, native
 * StreamTx buffering, and runner variance make that signal unobservable and
 * flaky at this layer. Task 5's deterministic 4,096-event mock mutation gate
 * proves the memory-relevant bound; this real-model tier proves that the same
 * HTTP drain gate parks genuine Qwen3 event consumption and unwinds on close.
 */

import { existsSync, readFileSync, statSync } from 'node:fs';
import { request as httpRequest, type ClientRequest, type IncomingMessage, type ServerResponse } from 'node:http';
import type { AddressInfo, Socket } from 'node:net';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';

import { loadModel, type ChatStreamEvent, type LoadableModel, type SessionCapableModel } from '@mlx-node/lm';
import { createServer, type ServerInstance } from '@mlx-node/server';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

const MODEL_PATH = process.env.QWEN3_STAGE0_MODEL_PATH;
const MODEL_NAME = 'stage0-qwen3';
// This checkpoint's CI-sized paged pool reports 18,720 prompt-token capacity;
// stay below it while remaining comfortably above the plan's 8k+ threshold.
const LONG_PROMPT = `${'x '.repeat(16_000)}\nAnswer with the single word done.`;
const CONTROL_BODY = {
  model: MODEL_NAME,
  input: 'Answer with exactly the single word READY and nothing else.',
  temperature: 0,
  max_output_tokens: 32,
} as const;
const H1_COMPLETION_DEADLINE_MS = 10_000;
// Repeating the same real prefill cancellation amplifies the checkpoint
// regression without tightening the wall-clock ceiling: healthy runs do only
// one 512-token chunk per round, while NO_PREFILL_CHECKPOINTS must process the
// full 16k prompt twelve times and reliably crosses the generous 10s deadline.
const H1_ABORT_ROUNDS = 12;
const H2_COMPLETION_DEADLINE_MS = 10_000;

type JsonHttpResult = {
  status: number;
  headers: IncomingMessage['headers'];
  body: Record<string, unknown>;
};

type StartedJsonRequest = {
  request: ClientRequest;
  result: Promise<JsonHttpResult>;
};

type StartedSseRequest = {
  request: ClientRequest;
  response: Promise<IncomingMessage>;
};

type InstrumentedModel = {
  model: SessionCapableModel;
  streamStarts: () => number;
  streamEvents: () => number;
  streamCloses: () => number;
  nonStreamingDispatches: () => number;
  nonStreamingCancellations: () => number;
};

const trackedSockets = new Set<Socket>();

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function withTimeout<T>(promise: Promise<T>, label: string, timeoutMs: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} (>${timeoutMs}ms)`)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

async function waitUntil(predicate: () => boolean, label: string, timeoutMs = 5_000): Promise<void> {
  const deadline = performance.now() + timeoutMs;
  while (performance.now() < deadline) {
    if (predicate()) return;
    await delay(10);
  }
  throw new Error(`${label} (>${timeoutMs}ms)`);
}

function trackClientSocket(req: ClientRequest): void {
  req.once('socket', (socket) => {
    trackedSockets.add(socket);
    socket.once('close', () => trackedSockets.delete(socket));
  });
}

function destroyTrackedSockets(): void {
  for (const socket of trackedSockets) socket.destroy();
}

function requestOptions(instance: ServerInstance, body: string, extraHeaders?: Record<string, string>) {
  const { port } = instance.server.address() as AddressInfo;
  return {
    host: '127.0.0.1',
    port,
    path: '/v1/responses',
    method: 'POST',
    agent: false,
    headers: {
      connection: 'close',
      'content-type': 'application/json',
      'content-length': String(Buffer.byteLength(body)),
      ...extraHeaders,
    },
  } as const;
}

function startJsonRequest(instance: ServerInstance, body: object): StartedJsonRequest {
  const payload = JSON.stringify(body);
  let req!: ClientRequest;
  const result = new Promise<JsonHttpResult>((resolve, reject) => {
    req = httpRequest(requestOptions(instance, payload), (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (chunk: Buffer | string) => chunks.push(Buffer.from(chunk)));
      res.once('error', reject);
      res.once('end', () => {
        const raw = Buffer.concat(chunks).toString('utf8');
        let parsed: Record<string, unknown>;
        try {
          parsed = JSON.parse(raw) as Record<string, unknown>;
        } catch (error) {
          reject(
            new Error(`HTTP ${res.statusCode ?? 0} returned non-JSON body ${JSON.stringify(raw)}: ${String(error)}`),
          );
          return;
        }
        resolve({ status: res.statusCode ?? 0, headers: res.headers, body: parsed });
      });
    });
    req.once('error', reject);
    trackClientSocket(req);
    req.end(payload);
  });
  return { request: req, result };
}

async function postJson(
  instance: ServerInstance,
  body: object,
  label: string,
  timeoutMs = 20_000,
): Promise<JsonHttpResult> {
  const started = startJsonRequest(instance, body);
  try {
    return await withTimeout(started.result, label, timeoutMs);
  } catch (error) {
    started.request.destroy();
    throw error;
  }
}

function startSseRequest(
  instance: ServerInstance,
  body: object,
  extraHeaders?: Record<string, string>,
): StartedSseRequest {
  const payload = JSON.stringify(body);
  let req!: ClientRequest;
  const response = new Promise<IncomingMessage>((resolve, reject) => {
    req = httpRequest(requestOptions(instance, payload, extraHeaders), (res) => {
      res.pause();
      resolve(res);
    });
    req.once('error', reject);
    trackClientSocket(req);
    req.end(payload);
  });
  return { request: req, response };
}

function outputText(result: JsonHttpResult): string {
  expect(result.status).toBe(200);
  // A deterministic control that consumes its explicit token budget is a
  // successful Responses API result with status=incomplete/reason=max tokens.
  // The parity oracle is the byte-exact output_text, not whether Qwen emitted
  // EOS inside this deliberately small budget.
  expect(['completed', 'incomplete']).toContain(result.body.status);
  expect(typeof result.body.output_text).toBe('string');
  return result.body.output_text as string;
}

function isSessionModel(model: LoadableModel): model is LoadableModel & SessionCapableModel {
  const candidate = model as Partial<SessionCapableModel>;
  return (
    typeof candidate.chatSessionStart === 'function' &&
    typeof candidate.chatStreamSessionStart === 'function' &&
    typeof candidate.resetCaches === 'function'
  );
}

function instrumentModel(target: SessionCapableModel): InstrumentedModel {
  const streamMethods = new Set<PropertyKey>([
    'chatStreamSessionStart',
    'chatStreamSessionContinue',
    'chatStreamSessionContinueTool',
  ]);
  const nonStreamingMethods = new Set<PropertyKey>([
    'chatSessionStart',
    'chatSessionContinue',
    'chatSessionContinueTool',
    'chatSessionStartCancellable',
    'chatSessionContinueCancellable',
    'chatSessionContinueToolCancellable',
  ]);
  let streamStarts = 0;
  let streamEvents = 0;
  let streamCloses = 0;
  let nonStreamingDispatches = 0;
  let nonStreamingCancellations = 0;

  const proxy = new Proxy(target, {
    get(model, property) {
      const value = Reflect.get(model, property, model) as unknown;
      if (streamMethods.has(property) && typeof value === 'function') {
        return async function* (...args: unknown[]): AsyncGenerator<ChatStreamEvent> {
          streamStarts += 1;
          try {
            const stream = (value as (...callArgs: unknown[]) => AsyncGenerator<ChatStreamEvent>).apply(model, args);
            for await (const event of stream) {
              // Count only events actually pulled through the real Qwen3
              // generator by the endpoint. Native callback queue production
              // is intentionally outside Task 5's HTTP-side bound.
              streamEvents += 1;
              yield event;
            }
          } finally {
            streamCloses += 1;
          }
        };
      }
      if (nonStreamingMethods.has(property) && typeof value === 'function') {
        return async (...args: unknown[]) => {
          // The wrapper is entered from inside SessionRegistry.withExclusive,
          // immediately before the native call. Mark here (before awaiting the
          // async NAPI method) so a very fast 0.6B turn cannot finish between
          // native handle creation and the test's 10 ms readiness poll.
          nonStreamingDispatches += 1;
          const result = await (value as (...callArgs: unknown[]) => Promise<unknown>).apply(model, args);
          if (String(property).endsWith('Cancellable')) {
            const call = result as {
              handle: unknown;
              result: () => Promise<unknown>;
            };
            const awaitResult = call.result.bind(call);
            return {
              handle: call.handle,
              result: async () => {
                try {
                  return await awaitResult();
                } catch (error) {
                  if (String(error).includes('chat session cancelled')) {
                    nonStreamingCancellations += 1;
                  }
                  throw error;
                }
              },
            };
          }
          return result;
        };
      }
      return typeof value === 'function' ? value.bind(model) : value;
    },
  }) as SessionCapableModel;

  return {
    model: proxy,
    streamStarts: () => streamStarts,
    streamEvents: () => streamEvents,
    streamCloses: () => streamCloses,
    nonStreamingDispatches: () => nonStreamingDispatches,
    nonStreamingCancellations: () => nonStreamingCancellations,
  };
}

function startTickProbe(intervalMs = 50): { stop: () => number } {
  let last = performance.now();
  let maxGap = 0;
  const timer = setInterval(() => {
    const now = performance.now();
    maxGap = Math.max(maxGap, now - last);
    last = now;
  }, intervalMs);
  return {
    stop: () => {
      clearInterval(timer);
      maxGap = Math.max(maxGap, performance.now() - last);
      return maxGap;
    },
  };
}

const stage0Describe = MODEL_PATH ? describe.sequential : describe.skip;

stage0Describe('Stage-0 real-model concurrency hazards', () => {
  let instance: ServerInstance;
  let nativeModel: SessionCapableModel;
  let instrumented: InstrumentedModel;
  let forcedFalseWrites = 0;

  const sessionRegistry = () => {
    const registry = instance.registry.getSessionRegistry(MODEL_NAME);
    if (registry === undefined) throw new Error(`missing SessionRegistry for ${MODEL_NAME}`);
    return registry;
  };

  const waitForAdmissionDrain = (label: string, timeoutMs = 15_000) =>
    waitUntil(
      () =>
        sessionRegistry().queueDepth === 0 &&
        sessionRegistry().preDispatchAdmitCount === 0 &&
        instance.health().work.inFlight === 0,
      label,
      timeoutMs,
    );

  beforeAll(async () => {
    if (MODEL_PATH === undefined) return;
    if (!existsSync(MODEL_PATH) || !statSync(MODEL_PATH).isDirectory()) {
      throw new Error(`QWEN3_STAGE0_MODEL_PATH is set but is not a directory: ${MODEL_PATH}`);
    }
    if (!existsSync(join(MODEL_PATH, 'config.json'))) {
      throw new Error(`QWEN3_STAGE0_MODEL_PATH has no config.json: ${MODEL_PATH}`);
    }
    let modelType: unknown;
    try {
      modelType = (JSON.parse(readFileSync(join(MODEL_PATH, 'config.json'), 'utf8')) as { model_type?: unknown })
        .model_type;
    } catch (error) {
      throw new Error(`QWEN3_STAGE0_MODEL_PATH has an invalid config.json: ${MODEL_PATH}: ${String(error)}`);
    }
    if (modelType !== 'qwen3') {
      throw new Error(
        `QWEN3_STAGE0_MODEL_PATH must identify a Qwen3 checkpoint; config model_type=${String(modelType)}`,
      );
    }

    instance = await createServer({
      port: 0,
      host: '127.0.0.1',
      disableStore: true,
      idleClearCacheMs: 0,
      maxQueueDepthPerModel: 1,
    });
    await instance.loadModel({
      name: MODEL_NAME,
      load: async () => {
        const loaded = await loadModel(MODEL_PATH);
        if (!isSessionModel(loaded)) {
          throw new Error(`QWEN3_STAGE0_MODEL_PATH did not load a session-capable chat model: ${MODEL_PATH}`);
        }
        nativeModel = loaded;
        instrumented = instrumentModel(loaded);
        return instrumented.model;
      },
    });

    // H4's per-response transport mutation. The normal server request
    // listener was installed by createServer; prepend makes this wrapper
    // run first, while the opt-in header confines it to exactly one test
    // request. Delegate the bytes, but report one false return and never
    // synthesize drain: the handler must park until close settles its helper.
    instance.server.prependListener('request', (req, res) => {
      if (req.headers['x-stage0-force-backpressure'] !== '1') return;
      const originalWrite = res.write.bind(res);
      let forced = false;
      res.write = ((...args: Parameters<ServerResponse['write']>) => {
        const ok = originalWrite(...args);
        if (!forced) {
          forced = true;
          forcedFalseWrites += 1;
          return false;
        }
        return ok;
      }) as ServerResponse['write'];
    });
  }, 120_000);

  afterAll(async () => {
    destroyTrackedSockets();
    if (instance !== undefined) {
      await instance.close({ timeoutMs: 10_000 });
    }
  }, 30_000);

  it('H1 keeps 50ms ticks alive and cancels long prefill before reset + greedy successor', async () => {
    await nativeModel.resetCaches();
    const control = await postJson(instance, CONTROL_BODY, 'H1 greedy control did not complete');
    const expected = outputText(control);
    await nativeModel.resetCaches();

    const baselineStreamStarts = instrumented.streamStarts();
    const baselineStreamCloses = instrumented.streamCloses();
    const tickProbe = startTickProbe();
    const holders: StartedSseRequest[] = [];
    const holderResponses: IncomingMessage[] = [];
    const holderOutcomes: Promise<unknown>[] = [];
    let stopHazardLoop = false;
    // Give the interval one ordinary sample before entering the hazard window.
    await delay(75);

    try {
      const hazardStarted = performance.now();
      const successor = await withTimeout(
        (async () => {
          for (let round = 0; round < H1_ABORT_ROUNDS; round += 1) {
            const holder = startSseRequest(instance, {
              model: MODEL_NAME,
              input: LONG_PROMPT,
              stream: true,
              temperature: 0,
              max_output_tokens: 64,
            });
            holders.push(holder);
            holderOutcomes.push(holder.response.catch((error: unknown) => error));
            const response = await withTimeout(
              holder.response,
              `H1 streaming holder ${round + 1} never returned SSE headers`,
              5_000,
            );
            holderResponses.push(response);
            await waitUntil(
              () => instrumented.streamStarts() > baselineStreamStarts + round,
              `H1 real stream ${round + 1} never entered native dispatch`,
            );
            await delay(25);
            // The handle is now in flight. Abort before the first chunk can
            // cross the HTTP iterator, then enqueue resetCaches DIRECTLY on
            // the model. Awaiting it also keeps rounds strictly serial.
            response.socket.destroy();
            holder.request.destroy();
            await nativeModel.resetCaches();
            if (stopHazardLoop) {
              throw new Error('H1 hazard loop stopped after the outer completion deadline');
            }
            await waitUntil(
              () => instrumented.streamCloses() > baselineStreamCloses + round,
              `H1 aborted stream ${round + 1} did not unwind`,
            );
          }
          return await postJson(
            instance,
            CONTROL_BODY,
            'H1 greedy successor did not complete',
            H1_COMPLETION_DEADLINE_MS,
          );
        })(),
        'H1 aborted prefill did not release reset + successor within its generous deadline',
        H1_COMPLETION_DEADLINE_MS,
      );
      expect(outputText(successor)).toBe(expected);
      expect(performance.now() - hazardStarted).toBeLessThan(H1_COMPLETION_DEADLINE_MS);
      await waitForAdmissionDrain('H1 queue/admission/in-flight state did not drain');
      // Let a post-hazard interval callback record any delayed tick before
      // reading the maximum.
      await delay(75);
      expect(tickProbe.stop()).toBeLessThan(500);
    } catch (error) {
      // Promise.race cannot cancel an in-progress async loop. Latch this so a
      // timed-out native reset may settle, but cannot launch another holder
      // after the test has begun teardown.
      stopHazardLoop = true;
      throw error;
    } finally {
      tickProbe.stop();
      for (const response of holderResponses) response.socket.destroy();
      for (const holder of holders) holder.request.destroy();
      await Promise.all(holderOutcomes);
    }
  }, 45_000);

  it('H2 cancels a destroyed non-streaming raw socket and releases its visibly queued successor', async () => {
    await nativeModel.resetCaches();
    const control = await postJson(instance, CONTROL_BODY, 'H2 greedy control did not complete');
    const expected = outputText(control);
    await nativeModel.resetCaches();

    const baselineDispatches = instrumented.nonStreamingDispatches();
    const baselineCancellations = instrumented.nonStreamingCancellations();
    const holder = startJsonRequest(instance, {
      model: MODEL_NAME,
      input: LONG_PROMPT,
      temperature: 0,
      max_output_tokens: 128,
    });
    let holderSettled = false;
    let holderFinal: unknown;
    const holderOutcome = holder.result
      .catch((error: unknown) => error)
      .then((outcome) => {
        holderFinal = outcome;
        return outcome;
      })
      .finally(() => {
        holderSettled = true;
      });
    let successor: StartedJsonRequest | undefined;

    try {
      await waitUntil(
        () => instrumented.nonStreamingDispatches() > baselineDispatches,
        'H2 holder never entered cancellable native dispatch',
      );
      successor = startJsonRequest(instance, CONTROL_BODY);
      let successorSettled = false;
      void successor.result.then(
        () => {
          successorSettled = true;
        },
        () => {
          successorSettled = true;
        },
      );
      await waitUntil(
        () => sessionRegistry().queueDepth === 1 || holderSettled || successorSettled,
        'H2 successor never reached a stable queue/terminal state',
      );
      if (sessionRegistry().queueDepth !== 1) {
        throw new Error(
          `H2 successor was never visibly queued: holderSettled=${holderSettled} ` +
            `successorSettled=${successorSettled} preDispatch=${sessionRegistry().preDispatchAdmitCount} ` +
            `holder=${JSON.stringify(holderFinal)}`,
        );
      }

      holder.request.destroy();
      const result = await withTimeout(
        successor.result,
        'H2 successor stayed queued after the holder socket was destroyed',
        H2_COMPLETION_DEADLINE_MS,
      );
      expect(outputText(result)).toBe(expected);
      await holderOutcome;
      expect(instrumented.nonStreamingCancellations()).toBe(baselineCancellations + 1);
      await waitForAdmissionDrain('H2 queue/admission/in-flight state did not drain');
    } finally {
      holder.request.destroy();
      successor?.request.destroy();
    }
  }, 45_000);

  it('H3 returns the exact real-HTTP queue_full 429 for holder + waiter + third', async () => {
    await nativeModel.resetCaches();
    const baselineDispatches = instrumented.nonStreamingDispatches();
    const holder = startJsonRequest(instance, {
      model: MODEL_NAME,
      input: LONG_PROMPT,
      temperature: 0,
      max_output_tokens: 128,
    });
    let holderSettled = false;
    let holderFinal: unknown;
    const holderOutcome = holder.result
      .catch((error: unknown) => error)
      .then((outcome) => {
        holderFinal = outcome;
        return outcome;
      })
      .finally(() => {
        holderSettled = true;
      });
    let waiter: StartedJsonRequest | undefined;
    let waiterOutcome: Promise<unknown> | undefined;

    try {
      await waitUntil(
        () => instrumented.nonStreamingDispatches() > baselineDispatches,
        'H3 holder never entered native dispatch',
      );
      waiter = startJsonRequest(instance, CONTROL_BODY);
      let waiterSettled = false;
      waiterOutcome = waiter.result
        .catch((error: unknown) => error)
        .finally(() => {
          waiterSettled = true;
        });
      await waitUntil(
        () => sessionRegistry().queueDepth === 1 || holderSettled || waiterSettled,
        'H3 waiter never reached a stable queue/terminal state',
      );
      if (sessionRegistry().queueDepth !== 1) {
        throw new Error(
          `H3 waiter was never visibly queued: holderSettled=${holderSettled} ` +
            `waiterSettled=${waiterSettled} preDispatch=${sessionRegistry().preDispatchAdmitCount} ` +
            `holder=${JSON.stringify(holderFinal)}`,
        );
      }

      const overflow = await postJson(instance, CONTROL_BODY, 'H3 overflow request did not return promptly', 2_000);
      expect(overflow.status).toBe(429);
      expect(overflow.headers['retry-after']).toBe('1');
      const error = overflow.body.error as Record<string, unknown> | undefined;
      expect(error?.type).toBe('rate_limit_error');
      expect(error?.code).toBe('queue_full');
    } finally {
      // Destroy the queued socket first so its pre-dispatch abort is visible
      // before releasing the holder. No queued cleanup request may dispatch.
      waiter?.request.destroy();
      holder.request.destroy();
      await Promise.all([holderOutcome, waiterOutcome]);
      await waitForAdmissionDrain('H3 cleanup did not drain holder/waiter admission state');
    }
  }, 45_000);

  it('H4 parks real Qwen3 event consumption on forced write(false) and close unwinds it', async () => {
    await nativeModel.resetCaches();
    const baselineWrites = forcedFalseWrites;
    const baselineEvents = instrumented.streamEvents();
    const baselineCloses = instrumented.streamCloses();
    const stream = startSseRequest(
      instance,
      {
        model: MODEL_NAME,
        input: 'Count upward forever, one integer per token.',
        stream: true,
        temperature: 0,
        max_output_tokens: 64,
      },
      { 'x-stage0-force-backpressure': '1' },
    );
    const streamOutcome = stream.response.catch((error: unknown) => error);
    let response: IncomingMessage | undefined;

    try {
      response = await withTimeout(stream.response, 'H4 SSE headers were never returned', 5_000);
      await waitUntil(() => forcedFalseWrites === baselineWrites + 1, 'H4 hook never forced write(false)');
      await waitUntil(
        () => instrumented.streamEvents() > baselineEvents,
        'H4 did not consume the first genuine Qwen3 event',
        10_000,
      );
      const parkedAt = instrumented.streamEvents();
      expect(parkedAt).toBe(baselineEvents + 1);
      await delay(300);
      expect(instrumented.streamEvents()).toBe(parkedAt);
      expect(instrumented.streamCloses()).toBe(baselineCloses);

      response.socket.destroy();
      stream.request.destroy();
      await waitUntil(
        () => instrumented.streamCloses() > baselineCloses,
        'H4 close did not unwind the parked real-model generator',
        10_000,
      );
      await waitForAdmissionDrain('H4 close did not drain queue/admission/in-flight state');
    } finally {
      response?.socket.destroy();
      stream.request.destroy();
      await streamOutcome;
    }
  }, 30_000);
});
