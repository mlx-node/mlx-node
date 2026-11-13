/**
 * A fake inference engine on 127.0.0.1: it records every request and answers the model select call
 * (`POST /v1/messages/count_tokens`) and the unload route (`POST /v1/models/unload`) as a script says. An answer can
 * wait on a promise, so a test holds a "load" open as long as it needs. Nothing loads a model.
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';

export interface EngineRequest {
  method: string;
  path: string;
  headers: Record<string, string>;
  body: string;
  /** The client closed the socket before the answer was sent. */
  closedEarly: boolean;
  /** The answer was sent. */
  answered: boolean;
}

export interface EngineAnswer {
  status?: number;
  json?: unknown;
  /** Answer only once this settles. */
  until?: Promise<unknown>;
  /** Close the socket with no answer. */
  drop?: boolean;
}

export type EngineScript = (request: EngineRequest) => EngineAnswer;

export interface FakeEngine {
  url: string;
  requests: EngineRequest[];
  /** The select calls, in order. */
  selects(): EngineRequest[];
  /** Replaces the script; the default answers 200 to both routes. */
  script(next: EngineScript): void;
  close(): Promise<void>;
}

export const SELECT_PATH = '/v1/messages/count_tokens';
export const UNLOAD_PATH = '/v1/models/unload';

export const DEFAULT_ENGINE_SCRIPT: EngineScript = (request) => {
  if (request.method === 'POST' && request.path === SELECT_PATH) return { json: { input_tokens: 1 } };
  if (request.method === 'POST' && request.path === UNLOAD_PATH) return { json: { unloaded: [] } };
  return { status: 404, json: { error: 'not found' } };
};

function headersOf(message: IncomingMessage): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [name, value] of Object.entries(message.headers)) {
    if (typeof value === 'string') out[name] = value;
    else if (Array.isArray(value)) out[name] = value.join(', ');
  }
  return out;
}

function answer(request: EngineRequest, message: IncomingMessage, response: ServerResponse, a: EngineAnswer): void {
  if (request.closedEarly) return;
  if (a.drop === true) {
    message.socket.destroy();
    return;
  }
  request.answered = true;
  response.writeHead(a.status ?? 200, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(a.json ?? {}));
}

export async function startFakeEngine(script: EngineScript = DEFAULT_ENGINE_SCRIPT): Promise<FakeEngine> {
  const requests: EngineRequest[] = [];
  let current = script;
  const server: Server = createServer((message, response) => {
    const chunks: Buffer[] = [];
    message.on('data', (chunk: Buffer) => chunks.push(chunk));
    message.on('end', () => {
      const url = new URL(message.url ?? '/', 'http://127.0.0.1');
      const request: EngineRequest = {
        method: message.method ?? 'GET',
        path: url.pathname,
        headers: headersOf(message),
        body: Buffer.concat(chunks).toString('utf8'),
        closedEarly: false,
        answered: false,
      };
      requests.push(request);
      response.on('close', () => {
        if (!request.answered) request.closedEarly = true;
      });
      const a = current(request);
      if (a.until === undefined) answer(request, message, response, a);
      else
        void a.until.then(
          () => answer(request, message, response, a),
          () => answer(request, message, response, a),
        );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    selects: () => requests.filter((r) => r.path === SELECT_PATH),
    script: (next) => {
      current = next;
    },
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

/** A promise and its release, to hold an answer open. */
export function gate(): { promise: Promise<void>; open: () => void } {
  let open = (): void => {};
  const promise = new Promise<void>((resolve) => {
    open = resolve;
  });
  return { promise, open };
}

/** How long the dev harness engine takes to "load" a model that is not in memory. */
export const HARNESS_LOAD_MS = 4_000;

/** What the dev harness engine holds: the load it runs and the model in memory. */
export interface HarnessEngineState {
  loading: { model: string; startedAt: number } | null;
  resident: string | null;
}

export interface HarnessEngineOptions {
  /** Default {@link HARNESS_LOAD_MS}. */
  loadMs?: number;
  /** Called on each change of {@link HarnessEngineState}. */
  onState?: (state: HarnessEngineState) => void;
  /** Every unload answers 409, as an engine with a running answer does. */
  unloadBusy?: boolean;
}

/**
 * The dev harness's engine: a select of the model in memory answers at once; any other model "loads" for `loadMs`
 * and then is the one in memory. Loads run one at a time, in the order they were asked for, as the engine's write
 * slot does. An unload empties it, and answers 409 while a load runs.
 */
export function startHarnessEngine(options: HarnessEngineOptions = {}): Promise<FakeEngine> {
  const loadMs = options.loadMs ?? HARNESS_LOAD_MS;
  const state: HarnessEngineState = { loading: null, resident: null };
  let queue: Promise<void> = Promise.resolve();
  const report = (): void => options.onState?.({ loading: state.loading, resident: state.resident });
  return startFakeEngine((request) => {
    if (request.method === 'POST' && request.path === SELECT_PATH) {
      let model: unknown = null;
      try {
        model = (JSON.parse(request.body) as { model?: unknown }).model;
      } catch {
        return { status: 400, json: {} };
      }
      if (typeof model !== 'string') return { status: 400, json: {} };
      if (model === state.resident && state.loading === null) return { json: { input_tokens: 1 } };
      const name = model;
      queue = queue.then(async () => {
        if (name === state.resident) return;
        state.loading = { model: name, startedAt: Date.now() };
        report();
        await new Promise<void>((resolve) => setTimeout(resolve, loadMs));
        state.loading = null;
        state.resident = name;
        report();
      });
      return { until: queue, json: { input_tokens: 1 } };
    }
    if (request.method === 'POST' && request.path === UNLOAD_PATH) {
      if (options.unloadBusy === true || state.loading !== null) return { status: 409, json: {} };
      const unloaded = state.resident === null ? [] : [state.resident];
      state.resident = null;
      report();
      return { json: { unloaded } };
    }
    return { status: 404, json: { error: 'not found' } };
  });
}
