/**
 * JSON-RPC 2.0 peer for ACP. Our request ids are integers from 1; the agent's requests use its own id space. A
 * message with `method` and `id` is an agent request, with `id` and no `method` a reply to us, with `method` only a
 * notification. Error replies keep only the numeric code, the agent's fixed `error_kind` and a `kind` from
 * {@link RPC_KINDS}: message and data text from the agent never leave this module.
 */

import { encodeFrame } from './framing.js';

/** Per-method reply deadlines. `session/prompt` has none: Chat's turn cap bounds it, a Work turn has no limit. */
export const ACP_TIMEOUTS = {
  initialize: 10_000,
  'session/new': 15_000,
  'session/load': 30_000,
  '_x.ai/session/info': 5_000,
  'session/set_model': 10_000,
  '_x.ai/session/rename': 10_000,
  '_x.ai/session/delete': 10_000,
  '_x.ai/workflows/stop': 10_000,
} as const;

/** The 100th line that is not JSON is a protocol error. */
export const MAX_GARBAGE_LINES = 100;

export const METHOD_NOT_FOUND = -32601;
export const INTERNAL_ERROR = -32603;

/** The values of `error.data.kind` the client keeps: the end of a manual compaction. */
export const RPC_KINDS = ['compact_failed', 'compact_cancelled'] as const;
export type RpcKind = (typeof RPC_KINDS)[number];

function rpcKind(value: unknown): RpcKind | null {
  for (const kind of RPC_KINDS) if (value === kind) return kind;
  return null;
}

export class AcpRpcError extends Error {
  readonly code: number;
  /** `error.data.error_kind` when it is a string; the only part of the agent's error text we keep. */
  readonly errorKind: string | null;
  /** `error.data.kind` when it is one of {@link RPC_KINDS}. */
  readonly kind: RpcKind | null;

  constructor(code: number, errorKind: string | null, kind: RpcKind | null = null) {
    super(`agent error ${code}`);
    this.code = code;
    this.errorKind = errorKind;
    this.kind = kind;
    this.name = 'AcpRpcError';
  }
}

/** A soft deadline passed. The client stays open; `late` settles with the reply when it comes. */
export class AcpSoftTimeoutError extends Error {
  readonly method: string;
  readonly late: Promise<unknown>;

  constructor(method: string, late: Promise<unknown>) {
    super(`agent did not answer ${method} in time`);
    this.method = method;
    this.late = late;
    this.name = 'AcpSoftTimeoutError';
  }
}

export class AcpTimeoutError extends Error {
  readonly method: string;

  constructor(method: string) {
    super(`agent did not answer ${method}`);
    this.method = method;
    this.name = 'AcpTimeoutError';
  }
}

export class AcpClosedError extends Error {
  constructor() {
    super('agent connection closed');
    this.name = 'AcpClosedError';
  }
}

export interface AcpTransport {
  write(frame: string): void;
}

/** `unreadable`: handling one agent line threw (for example a payload nested too deep to serialise). */
export type ProtocolErrorReason = 'garbage' | 'timeout' | 'framing' | 'write-queue' | 'unreadable';

export interface AcpClientHandlers {
  onNotification?: (method: string, params: unknown) => void;
  /** An agent request. Resolve `{ result }` to answer it, or null for "method not found". */
  onRequest?: (method: string, params: unknown) => Promise<{ result: unknown } | null>;
  onProtocolError?: (reason: ProtocolErrorReason) => void;
}

interface Pending {
  method: string;
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout> | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isRequestId(value: unknown): value is number | string {
  return (typeof value === 'number' && Number.isFinite(value)) || typeof value === 'string';
}

export class AcpClient {
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();
  private garbage = 0;
  private closed = false;
  private readonly transport: AcpTransport;
  private readonly handlers: AcpClientHandlers;

  constructor(transport: AcpTransport, handlers: AcpClientHandlers = {}) {
    this.transport = transport;
    this.handlers = handlers;
  }

  get garbageLines(): number {
    return this.garbage;
  }

  get isClosed(): boolean {
    return this.closed;
  }

  /** One line from the agent's stdout. */
  receive(line: string): void {
    if (this.closed) return;
    let blank = true;
    for (let i = 0; i < line.length && blank; i++) {
      const c = line.charCodeAt(i);
      if (c !== 32 && c !== 9 && c !== 13) blank = false;
    }
    if (blank) return;
    let message: unknown;
    try {
      message = JSON.parse(line);
    } catch {
      message = undefined;
    }
    if (!isRecord(message)) {
      this.garbage += 1;
      if (this.garbage >= MAX_GARBAGE_LINES) this.handlers.onProtocolError?.('garbage');
      return;
    }
    try {
      this.dispatch(message);
    } catch {
      this.handlers.onProtocolError?.('unreadable');
    }
  }

  private dispatch(message: Record<string, unknown>): void {
    const method = message.method;
    const hasId = 'id' in message && isRequestId(message.id);
    if (typeof method === 'string' && hasId) {
      this.answerRequest(message.id as number | string, method, message.params);
      return;
    }
    if (typeof method === 'string') {
      this.handlers.onNotification?.(method, message.params);
      return;
    }
    if (hasId && typeof message.id === 'number') this.settle(message.id, message);
  }

  request(method: string, params: unknown, timeoutMs: number | null): Promise<unknown> {
    return this.start(method, params, timeoutMs).reply;
  }

  /** {@link request} with the request's id, for {@link isPending}. */
  start(method: string, params: unknown, timeoutMs: number | null): { id: number; reply: Promise<unknown> } {
    const id = this.nextId++;
    if (this.closed) return { id, reply: Promise.reject(new AcpClosedError()) };
    const reply = new Promise<unknown>((resolve, reject) => {
      const entry: Pending = { method, resolve, reject, timer: null };
      if (timeoutMs !== null) entry.timer = this.deadline(id, entry, timeoutMs);
      this.pending.set(id, entry);
      try {
        this.transport.write(encodeFrame({ jsonrpc: '2.0', id, method, params }));
      } catch (error) {
        this.pending.delete(id);
        if (entry.timer !== null) clearTimeout(entry.timer);
        reject(error instanceof Error ? error : new AcpClosedError());
      }
    });
    return { id, reply };
  }

  /**
   * The soft call: {@link request} with a deadline that never closes the client. A missed {@link ACP_TIMEOUTS}
   * deadline is a protocol error that stops the agent with every run and command it holds; past `ms` this rejects
   * with {@link AcpSoftTimeoutError} instead and the request stays open. Every call added that is not a turn uses it.
   */
  requestSoft(method: string, params: unknown, ms: number): Promise<unknown> {
    const { reply } = this.start(method, params, null);
    return new Promise<unknown>((resolve, reject) => {
      const timer = setTimeout(() => reject(new AcpSoftTimeoutError(method, reply)), ms);
      reply.then(
        (value) => {
          clearTimeout(timer);
          resolve(value);
        },
        (error: unknown) => {
          clearTimeout(timer);
          reject(error);
        },
      );
    });
  }

  /**
   * Gives request `id`, started with no deadline, the deadline of {@link request} from now: a miss is the same protocol
   * error. Nothing for a request that has a deadline or a reply.
   */
  harden(id: number, timeoutMs: number): void {
    const entry = this.pending.get(id);
    if (entry === undefined || entry.timer !== null) return;
    entry.timer = this.deadline(id, entry, timeoutMs);
  }

  private deadline(id: number, entry: Pending, timeoutMs: number): ReturnType<typeof setTimeout> {
    return setTimeout(() => {
      this.pending.delete(id);
      entry.reject(new AcpTimeoutError(entry.method));
      this.handlers.onProtocolError?.('timeout');
    }, timeoutMs);
  }

  /** Whether request `id` has no reply yet. False from the line that carries its reply on, before any promise job. */
  isPending(id: number): boolean {
    return this.pending.has(id);
  }

  notify(method: string, params: unknown): void {
    if (this.closed) return;
    this.transport.write(encodeFrame({ jsonrpc: '2.0', method, params }));
  }

  /** Rejects every pending request with {@link AcpClosedError}; later calls fail at once. */
  close(): void {
    if (this.closed) return;
    this.closed = true;
    for (const entry of this.pending.values()) {
      if (entry.timer !== null) clearTimeout(entry.timer);
      entry.reject(new AcpClosedError());
    }
    this.pending.clear();
  }

  private settle(id: number, message: Record<string, unknown>): void {
    const entry = this.pending.get(id);
    if (entry === undefined) return;
    this.pending.delete(id);
    if (entry.timer !== null) clearTimeout(entry.timer);
    if ('error' in message && message.error !== undefined && message.error !== null) {
      const error = isRecord(message.error) ? message.error : {};
      const code = typeof error.code === 'number' ? error.code : INTERNAL_ERROR;
      const data = isRecord(error.data) ? error.data : {};
      entry.reject(
        new AcpRpcError(code, typeof data.error_kind === 'string' ? data.error_kind : null, rpcKind(data.kind)),
      );
      return;
    }
    entry.resolve(message.result);
  }

  private answerRequest(id: number | string, method: string, params: unknown): void {
    const reply = (body: Record<string, unknown>): void => {
      if (this.closed) return;
      try {
        this.transport.write(encodeFrame({ jsonrpc: '2.0', id, ...body }));
      } catch {
        // The process is gone; its exit handler ends everything that waited on it.
      }
    };
    const handler = this.handlers.onRequest;
    if (handler === undefined) {
      reply({ error: { code: METHOD_NOT_FOUND, message: 'Method not found' } });
      return;
    }
    let answered: Promise<{ result: unknown } | null>;
    try {
      answered = handler(method, params);
    } catch (error) {
      answered = Promise.reject(error);
    }
    answered.then(
      (answer) =>
        reply(
          answer === null
            ? { error: { code: METHOD_NOT_FOUND, message: 'Method not found' } }
            : { result: answer.result },
        ),
      () => reply({ error: { code: INTERNAL_ERROR, message: 'Internal error' } }),
    );
  }
}
