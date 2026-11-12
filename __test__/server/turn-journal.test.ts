/**
 * The desktop sidecar's turn journal and quarantine (Splash `RuntimeCall.started`,
 * `Quarantine.refuses`, `request_fingerprint`) at the two chat endpoints.
 */

import { EventEmitter } from 'node:events';
import { mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { createServer, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { ChatConfig, ChatMessage } from '@mlx-node/core';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { handleCreateMessage } from '../../packages/server/src/endpoints/messages.js';
import { handleCreateResponse } from '../../packages/server/src/endpoints/responses.js';
import { createHandler } from '../../packages/server/src/handler.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';
import {
  QUARANTINE_MESSAGE,
  rawRequestFingerprint,
  requestFingerprint,
  TurnJournal,
} from '../../packages/server/src/turn-journal.js';

const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function journalDir(): string {
  const dir = mkdtempSync(join(tmpdir(), 'mlx-turn-journal-'));
  dirs.push(dir);
  return dir;
}

function createMockRes(): {
  res: ServerResponse;
  disconnect: () => void;
  getBody: () => string;
  getStatus: () => number;
} {
  const emitter = new EventEmitter();
  const res = emitter as unknown as ServerResponse & {
    writableEnded: boolean;
    writableFinished: boolean;
    headersSent: boolean;
    destroyed: boolean;
  };
  let body = '';
  let status = 200;
  let socketDestroyed = false;
  const socket = {
    get destroyed() {
      return socketDestroyed;
    },
    once: (event: string, cb: () => void): void => {
      emitter.once(event, cb);
    },
    on: (event: string, cb: () => void): void => {
      emitter.on(event, cb);
    },
    removeListener: (event: string, cb: () => void): void => {
      emitter.removeListener(event, cb);
    },
  };
  res.writableEnded = false;
  res.writableFinished = false;
  res.headersSent = false;
  res.destroyed = false;
  Object.defineProperty(res, 'socket', { value: socket, configurable: true });
  const takeCallback = (args: unknown[]): ((err?: Error | null) => void) | undefined =>
    args.find((a): a is (err?: Error | null) => void => typeof a === 'function');
  res.writeHead = ((code: number) => {
    status = code;
    res.headersSent = true;
    return res;
  }) as unknown as ServerResponse['writeHead'];
  res.setHeader = (() => res) as unknown as ServerResponse['setHeader'];
  res.getHeader = (() => undefined) as unknown as ServerResponse['getHeader'];
  res.removeHeader = (() => {}) as ServerResponse['removeHeader'];
  res.write = ((chunk: unknown, ...rest: unknown[]) => {
    if (typeof chunk === 'string') body += chunk;
    const cb = takeCallback(rest);
    if (cb) queueMicrotask(() => cb(null));
    return true;
  }) as ServerResponse['write'];
  res.end = ((chunk: unknown, ...rest: unknown[]) => {
    if (typeof chunk === 'string') body += chunk;
    res.headersSent = true;
    res.writableEnded = true;
    const cb = takeCallback(typeof chunk === 'function' ? [chunk] : rest);
    if (cb) queueMicrotask(() => cb(null));
    return res;
  }) as ServerResponse['end'];
  res.destroy = (() => {
    socketDestroyed = true;
    res.destroyed = true;
    emitter.emit('close');
    return res;
  }) as ServerResponse['destroy'];
  return {
    res,
    disconnect(): void {
      socketDestroyed = true;
      res.destroyed = true;
      emitter.emit('close');
    },
    getBody: () => body,
    getStatus: () => status,
  };
}

const terminalChunk = {
  done: true,
  text: 'ok',
  finishReason: 'stop',
  toolCalls: [],
  thinking: null,
  thinkingEnabled: false,
  numTokens: 1,
  promptTokens: 1,
  reasoningTokens: 0,
  rawText: 'ok',
};

/** What the journal dir held each time the model started a turn. */
function createJournalModel(dir: string, opts: { fail?: boolean; hold?: Promise<void> } = {}) {
  const seen: string[][] = [];
  const look = (): void => {
    seen.push(readdirSync(dir).map((name) => readFileSync(join(dir, name), 'utf8')));
  };
  async function* stream() {
    look();
    if (opts.fail === true) throw new Error('native turn failed');
    yield terminalChunk;
  }
  const model = {
    chatSessionStart: vi.fn(async () => {
      look();
      await opts.hold;
      if (opts.fail === true) throw new Error('native turn failed');
      return { ...terminalChunk, cachedTokens: 0, performance: undefined };
    }),
    chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path not expected')),
    chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path not expected')),
    chatStreamSessionStart: vi.fn(() => stream()),
    chatStreamSessionContinue: vi.fn(() => stream()),
    chatStreamSessionContinueTool: vi.fn(() => stream()),
    resetCaches: vi.fn().mockResolvedValue(undefined),
  };
  return { model, seen };
}

async function settle(): Promise<void> {
  for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r));
}

type Endpoint = 'messages' | 'responses';

function send(
  endpoint: Endpoint,
  registry: ModelRegistry,
  journal: TurnJournal,
  text: string,
  stream: boolean,
  res: ServerResponse,
  resolveModel?: (name: string) => Promise<void>,
): Promise<void> {
  if (endpoint === 'messages') {
    return handleCreateMessage(
      res,
      { model: 'm', messages: [{ role: 'user', content: text }], max_tokens: 8, ...(stream ? { stream } : {}) },
      registry,
      undefined,
      undefined,
      resolveModel,
      undefined,
      undefined,
      journal,
    );
  }
  return handleCreateResponse(
    res,
    { model: 'm', input: text, store: false, ...(stream ? { stream } : {}) },
    registry,
    null,
    undefined,
    undefined,
    undefined,
    undefined,
    resolveModel,
    journal,
  );
}

const ENDPOINTS: readonly Endpoint[] = ['messages', 'responses'];

describe('the turn journal at the chat endpoints', () => {
  for (const endpoint of ENDPOINTS) {
    it(`a turn is journaled from dispatch until it ends (/v1/${endpoint})`, async () => {
      for (const [stream, fail] of [
        [false, false],
        [true, false],
        [false, true],
        [true, true],
      ] as const) {
        const dir = journalDir();
        const journal = new TurnJournal({ dir, quarantined: new Set() });
        const begin = vi.spyOn(journal, 'begin');
        const { model, seen } = createJournalModel(dir, { fail });
        const registry = new ModelRegistry();
        registry.register('m', model as never);

        await send(endpoint, registry, journal, 'hello', stream, createMockRes().res);

        expect(begin).toHaveBeenCalledTimes(1);
        const fingerprint = begin.mock.calls[0][0];
        expect(fingerprint.length).toBe(64);
        expect(seen).toEqual([[fingerprint]]);
        expect(readdirSync(dir)).toEqual([]);
      }
    });

    it(`a request aborted while queued is never journaled (/v1/${endpoint})`, async () => {
      let release!: () => void;
      const hold = new Promise<void>((r) => {
        release = r;
      });
      const dir = journalDir();
      const journal = new TurnJournal({ dir, quarantined: new Set() });
      const begin = vi.spyOn(journal, 'begin');
      const { model } = createJournalModel(dir, { hold });
      const registry = new ModelRegistry();
      registry.register('m', model as never);

      const holderDone = send(endpoint, registry, journal, 'A', false, createMockRes().res);
      await settle();
      expect(begin).toHaveBeenCalledTimes(1);

      const queued = createMockRes();
      const queuedDone = send(endpoint, registry, journal, 'B', true, queued.res);
      await settle();
      queued.disconnect();
      release();
      await holderDone;
      await queuedDone;

      expect(begin).toHaveBeenCalledTimes(1);
      expect(model.chatStreamSessionStart).not.toHaveBeenCalled();
      expect(readdirSync(dir)).toEqual([]);
    });
  }

  it('a quarantined request gets 400 request_quarantined and never reaches the model', async () => {
    for (const endpoint of ENDPOINTS) {
      const dir = journalDir();
      const first = new TurnJournal({ dir, quarantined: new Set() });
      const begin = vi.spyOn(first, 'begin');
      const { model } = createJournalModel(dir);
      const registry = new ModelRegistry();
      registry.register('m', model as never);
      await send(endpoint, registry, first, 'crashes the model', false, createMockRes().res);
      const fingerprint = begin.mock.calls[0][0];
      const dispatched = model.chatSessionStart.mock.calls.length;

      const quarantined = new TurnJournal({ dir, quarantined: new Set([fingerprint]) });
      const refusedBegin = vi.spyOn(quarantined, 'begin');
      const refused = createMockRes();
      await send(endpoint, registry, quarantined, 'crashes the model', false, refused.res);

      expect(refused.getStatus()).toBe(400);
      const body = JSON.parse(refused.getBody());
      if (endpoint === 'messages') {
        expect(body).toEqual({ type: 'error', error: { type: 'invalid_request_error', message: QUARANTINE_MESSAGE } });
      } else {
        expect(body.error).toMatchObject({
          type: 'invalid_request_error',
          code: 'request_quarantined',
          message: QUARANTINE_MESSAGE,
        });
      }
      expect(model.chatSessionStart.mock.calls.length).toBe(dispatched);
      expect(refusedBegin).not.toHaveBeenCalled();

      const other = createMockRes();
      await send(endpoint, registry, quarantined, 'crashes the model!', false, other.res);
      expect(other.getStatus()).toBe(200);
    }
  });

  it('a quarantined request is refused before any model load', async () => {
    for (const endpoint of ENDPOINTS) {
      const dir = journalDir();
      const first = new TurnJournal({ dir, quarantined: new Set() });
      const begin = vi.spyOn(first, 'begin');
      const { model } = createJournalModel(dir);
      const resident = new ModelRegistry();
      resident.register('m', model as never);
      await send(endpoint, resident, first, 'crashes the model', false, createMockRes().res);
      const fingerprint = begin.mock.calls[0][0];

      const quarantined = new TurnJournal({ dir, quarantined: new Set([fingerprint]) });
      const cold = new ModelRegistry();
      const resolveModel = vi.fn(async (name: string) => {
        cold.register(name, model as never);
      });
      const refused = createMockRes();
      await send(endpoint, cold, quarantined, 'crashes the model', false, refused.res, resolveModel);
      expect(refused.getStatus()).toBe(400);
      expect(resolveModel).not.toHaveBeenCalled();

      const other = createMockRes();
      await send(endpoint, cold, quarantined, 'crashes the model!', false, other.res, resolveModel);
      expect(other.getStatus()).toBe(200);
      expect(resolveModel).toHaveBeenCalledTimes(1);
    }
  });
});

describe('requestFingerprint (Splash request_fingerprint)', () => {
  const messages: ChatMessage[] = [
    { role: 'system', content: 'be brief' },
    { role: 'user', content: 'look', images: [new Uint8Array([1, 2, 3])] },
  ];
  const config: ChatConfig = { maxNewTokens: 8, temperature: 0.5, cacheSalt: 's' };
  const base = requestFingerprint('m', messages, config);

  it('the fingerprint ignores identity and reporting and keeps the content', () => {
    let hex = true;
    for (const c of base) hex &&= (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    expect(base.length).toBe(64);
    expect(hex).toBe(true);

    const same = (m: readonly ChatMessage[], c: ChatConfig, model = 'm'): boolean =>
      requestFingerprint(model, m, c) === base;
    expect(same(messages, { ...config, cacheOwnerId: 'o1', cacheRootOwnerId: 'r1', reportPerformance: true })).toBe(
      true,
    );
    expect(same(messages, { ...config, modelAlias: 'alias' } as ChatConfig)).toBe(true);
    expect(same(messages, { cacheSalt: 's', temperature: 0.5, maxNewTokens: 8 })).toBe(true);
    expect(same(messages, { ...config, topK: undefined })).toBe(true);

    expect(same(messages, { ...config, temperature: 0.6 })).toBe(false);
    expect(same(messages, { ...config, cacheSalt: 't' })).toBe(false);
    expect(same(messages, { ...config, tools: [] })).toBe(false);
    expect(same(messages, config, 'n')).toBe(false);
    expect(same([messages[0], { ...messages[1], content: 'look!' }], config)).toBe(false);
    // Image content, not its size (Splash `ImageSpan` digests).
    expect(same([messages[0], { ...messages[1], images: [new Uint8Array([1, 2, 4])] }], config)).toBe(false);
    expect(same([messages[0], { ...messages[1], images: [new Uint8Array([1, 2, 3, 4])] }], config)).toBe(false);
    expect(same([messages[0], { ...messages[1], images: [new Uint8Array([1, 2, 3])] }], config)).toBe(true);

    // A continuation's stored chain, by its id.
    const continued = requestFingerprint('m', messages, config, 'resp_a');
    expect(continued).not.toBe(base);
    expect(requestFingerprint('m', messages, config, 'resp_b')).not.toBe(continued);
    expect(requestFingerprint('m', messages, config, 'resp_a')).toBe(continued);
  });

  it('a raw route body is keyed by its route', () => {
    expect(rawRequestFingerprint('/v1/systemone', '{}')).not.toBe(rawRequestFingerprint('/v1/other', '{}'));
    expect(rawRequestFingerprint('/v1/systemone', '{}')).toBe(rawRequestFingerprint('/v1/systemone', '{}'));
  });
});

describe('the turn journal at /v1/systemone', () => {
  it('a decision is journaled only while it holds the lane', async () => {
    const dir = journalDir();
    const journal = new TurnJournal({ dir, quarantined: new Set() });
    let release!: () => void;
    const hold = new Promise<void>((r) => {
      release = r;
    });
    const seen: string[][] = [];
    const decideRaw = vi.fn(async (raw: string) => {
      seen.push(readdirSync(dir).map((name) => readFileSync(join(dir, name), 'utf8')));
      if (raw.includes('first')) await hold;
      return { answers: { flag: { type: 'noul' as const, noul: 0.5 } }, usage: { input_tokens: 1, output_tokens: 0 } };
    });
    const registry = new ModelRegistry();
    registry.registerDecision('flash', { kind: 'decision', decideRaw } as never);
    const server = createServer(createHandler(registry, { store: null, turnJournal: journal }));
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    try {
      const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1/systemone`;
      const body = (state: string): string =>
        JSON.stringify({ model: 'flash', state, questions: { flag: { type: 'noul' } } });
      const first = fetch(url, { method: 'POST', body: body('first') });
      while (decideRaw.mock.calls.length === 0) await settle();
      const second = fetch(url, { method: 'POST', body: body('second') });
      await new Promise((r) => setTimeout(r, 50));
      // The queued second decision is not started yet: one file, the first's.
      expect(readdirSync(dir).length).toBe(1);
      release();
      expect((await first).status).toBe(200);
      expect((await second).status).toBe(200);
      expect(seen).toEqual([
        [rawRequestFingerprint('/v1/systemone', body('first'))],
        [rawRequestFingerprint('/v1/systemone', body('second'))],
      ]);
      expect(readdirSync(dir)).toEqual([]);
    } finally {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });
});
