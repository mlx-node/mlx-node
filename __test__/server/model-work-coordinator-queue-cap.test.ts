import type { ServerResponse } from 'node:http';
import { Writable } from 'node:stream';

import type { ChatResult } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

import { handleCreateMessage } from '../../packages/server/src/endpoints/messages.js';
import { handleCreateResponse } from '../../packages/server/src/endpoints/responses.js';
import { ModelWorkCoordinator } from '../../packages/server/src/model-work-coordinator.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

const tick = () => new Promise<void>((resolve) => setImmediate(resolve));
const timeout = (ms: number) => new Promise<'timeout'>((resolve) => setTimeout(() => resolve('timeout'), ms));

function makeChatResult(overrides: Partial<ChatResult> = {}): ChatResult {
  return {
    text: 'ok',
    toolCalls: [],
    numTokens: 1,
    promptTokens: 1,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: 'ok',
    cachedTokens: 0,
    ...overrides,
  };
}

function createModel(): SessionCapableModel {
  return {
    chatSessionStart: vi.fn(async () => makeChatResult()),
    chatSessionContinue: vi.fn(async () => makeChatResult()),
    chatSessionContinueTool: vi.fn(async () => makeChatResult()),
    chatStreamSessionStart: vi.fn(),
    chatStreamSessionContinue: vi.fn(),
    chatStreamSessionContinueTool: vi.fn(),
    resetCaches: vi.fn(),
  } as unknown as SessionCapableModel;
}

function createMockRes(): {
  res: ServerResponse;
  getStatus: () => number;
  getBody: () => string;
  getHeaders: () => Record<string, string | string[]>;
  waitForEnd: () => Promise<void>;
} {
  let status = 200;
  let body = '';
  const headers: Record<string, string | string[]> = {};
  let endResolve!: () => void;
  const endPromise = new Promise<void>((resolve) => {
    endResolve = resolve;
  });

  const writable = new Writable({
    write(chunk: Uint8Array | string, _encoding: string, callback: () => void) {
      body += chunk.toString();
      callback();
    },
  }) as Writable & {
    headersSent: boolean;
    writeHead: (statusCode: number, headers?: Record<string, string>) => Writable;
    setHeader: (name: string, value: string) => void;
  };

  writable.headersSent = false;
  writable.writeHead = (statusCode: number, responseHeaders?: Record<string, string>) => {
    status = statusCode;
    if (responseHeaders) {
      for (const [key, value] of Object.entries(responseHeaders)) {
        headers[key.toLowerCase()] = value;
      }
    }
    writable.headersSent = true;
    return writable;
  };
  writable.setHeader = (name: string, value: string) => {
    headers[name.toLowerCase()] = value;
  };

  const originalEnd = writable.end.bind(writable);
  writable.end = (chunkArg?: unknown, encodingArg?: unknown, cbArg?: unknown) => {
    let chunk: string | Uint8Array | undefined;
    let cb: ((err?: Error | null) => void) | undefined;
    if (typeof chunkArg === 'function') {
      cb = chunkArg as (err?: Error | null) => void;
    } else {
      chunk = chunkArg as string | Uint8Array | undefined;
      if (typeof encodingArg === 'function') {
        cb = encodingArg as (err?: Error | null) => void;
      } else if (typeof cbArg === 'function') {
        cb = cbArg as (err?: Error | null) => void;
      }
    }
    if (chunk != null) body += chunk.toString();
    writable.headersSent = true;
    originalEnd(undefined, (err?: Error | null) => {
      if (cb) cb(err ?? null);
      endResolve();
    });
    return writable;
  };

  return {
    res: writable as unknown as ServerResponse,
    getStatus: () => status,
    getBody: () => body,
    getHeaders: () => headers,
    waitForEnd: () => endPromise,
  };
}

async function waitForQueueDepth(registry: ModelRegistry, modelName: string, expected: number): Promise<void> {
  for (let i = 0; i < 10; i += 1) {
    if (registry.getSessionRegistry(modelName)?.queueDepth === expected) return;
    await tick();
  }
  expect(registry.getSessionRegistry(modelName)?.queueDepth).toBe(expected);
}

describe('ModelWorkCoordinator + SessionRegistry queue cap', () => {
  it('POST /v1/messages enforces the per-model queue cap before waiting behind a model-load writer', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    registry.register('cap-model', createModel());
    const coordinator = new ModelWorkCoordinator();
    const writerStarted = deferred();
    const releaseWriter = deferred();
    let requestA: Promise<void> | undefined;
    let requestB: Promise<void> | undefined;
    const writer = coordinator.withModelLoad(async () => {
      writerStarted.resolve(undefined);
      await releaseWriter.promise;
    });

    await writerStarted.promise;

    try {
      const mockA = createMockRes();
      requestA = handleCreateMessage(
        mockA.res,
        {
          model: 'cap-model',
          messages: [{ role: 'user', content: 'A' }],
          max_tokens: 16,
        },
        registry,
        undefined,
        null,
        undefined,
        coordinator,
      );
      await tick();

      const mockB = createMockRes();
      requestB = handleCreateMessage(
        mockB.res,
        {
          model: 'cap-model',
          messages: [{ role: 'user', content: 'B' }],
          max_tokens: 16,
        },
        registry,
        undefined,
        null,
        undefined,
        coordinator,
      );
      await waitForQueueDepth(registry, 'cap-model', 1);

      const mockC = createMockRes();
      const requestC = handleCreateMessage(
        mockC.res,
        {
          model: 'cap-model',
          messages: [{ role: 'user', content: 'C' }],
          max_tokens: 16,
        },
        registry,
        undefined,
        null,
        undefined,
        coordinator,
      );
      const cOutcome = await Promise.race([requestC.then(() => 'done' as const), timeout(50)]);

      expect(cOutcome).toBe('done');
      await mockC.waitForEnd();
      expect(mockC.getStatus()).toBe(429);
      expect(mockC.getHeaders()['retry-after']).toBe('1');
      const parsed = JSON.parse(mockC.getBody());
      expect(parsed.type).toBe('error');
      expect(parsed.error.type).toBe('rate_limit_error');
      expect(parsed.error.message).toContain('Model queue full');
    } finally {
      releaseWriter.resolve(undefined);
      await writer;
      if (requestA) await requestA;
      if (requestB) await requestB;
    }
  });

  it('POST /v1/responses enforces the per-model queue cap before waiting behind a model-load writer', async () => {
    const registry = new ModelRegistry({ maxQueueDepth: 1 });
    registry.register('cap-model', createModel());
    const coordinator = new ModelWorkCoordinator();
    const writerStarted = deferred();
    const releaseWriter = deferred();
    let requestA: Promise<void> | undefined;
    let requestB: Promise<void> | undefined;
    const writer = coordinator.withModelLoad(async () => {
      writerStarted.resolve(undefined);
      await releaseWriter.promise;
    });

    await writerStarted.promise;

    try {
      const mockA = createMockRes();
      requestA = handleCreateResponse(
        mockA.res,
        { model: 'cap-model', input: 'A' },
        registry,
        null,
        undefined,
        undefined,
        null,
        coordinator,
      );
      await tick();

      const mockB = createMockRes();
      requestB = handleCreateResponse(
        mockB.res,
        { model: 'cap-model', input: 'B' },
        registry,
        null,
        undefined,
        undefined,
        null,
        coordinator,
      );
      await waitForQueueDepth(registry, 'cap-model', 1);

      const mockC = createMockRes();
      const requestC = handleCreateResponse(
        mockC.res,
        { model: 'cap-model', input: 'C' },
        registry,
        null,
        undefined,
        undefined,
        null,
        coordinator,
      );
      const cOutcome = await Promise.race([requestC.then(() => 'done' as const), timeout(50)]);

      expect(cOutcome).toBe('done');
      await mockC.waitForEnd();
      expect(mockC.getStatus()).toBe(429);
      expect(mockC.getHeaders()['retry-after']).toBe('1');
      const parsed = JSON.parse(mockC.getBody());
      expect(parsed.error.type).toBe('rate_limit_error');
      expect(parsed.error.code).toBe('queue_full');
      expect(parsed.error.message).toContain('Model queue full');
    } finally {
      releaseWriter.resolve(undefined);
      await writer;
      if (requestA) await requestA;
      if (requestB) await requestB;
    }
  });
});
