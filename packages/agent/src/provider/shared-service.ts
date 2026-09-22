/** Native-free listener: bind election happens before importing the model runtime. */
import { randomBytes } from 'node:crypto';
import { once } from 'node:events';
import { mkdir, rename, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { join } from 'node:path';

import {
  SHARED_IDLE_MS,
  SHARED_PROTOCOL,
  SHARED_REQUEST_LIMIT,
  type SharedEndpoint,
  type SharedFrame,
  type SharedRequest,
} from './shared-protocol.js';

export interface SharedBackend {
  stream(request: SharedRequest, signal: AbortSignal): AsyncIterable<SharedFrame>;
  busy(): boolean;
  close(): Promise<void>;
}

export async function startSharedService(options: {
  directory: string;
  port: number;
  loadBackend: () => Promise<SharedBackend>;
  idleMs?: number;
}): Promise<{ endpoint: SharedEndpoint; close(): Promise<void> }> {
  const token = randomBytes(32).toString('hex');
  let backend: Promise<SharedBackend> | undefined;
  let resolvedBackend: SharedBackend | undefined;
  let active = 0;
  let lastActivity = Date.now();
  let closing: Promise<void> | undefined;
  let requestsDrained: (() => void) | undefined;
  const server = createServer(async (request, response) => {
    if (request.headers.authorization !== `Bearer ${token}`) {
      response.writeHead(401).end();
      return;
    }
    if (request.method === 'GET' && request.url === '/health') {
      if (closing) {
        response.writeHead(503).end();
        return;
      }
      response.setHeader('content-type', 'application/json');
      response.end(JSON.stringify({ protocol: SHARED_PROTOCOL, pid: process.pid }));
      return;
    }
    if (request.method !== 'POST' || request.url !== '/stream') {
      response.writeHead(404).end();
      return;
    }
    if (active >= SHARED_REQUEST_LIMIT || closing) {
      response.writeHead(429).end('Shared inference queue is full.');
      return;
    }
    active++;
    lastActivity = Date.now();
    const controller = new AbortController();
    response.on('close', () => controller.abort());
    try {
      const chunks: Buffer[] = [];
      let bytes = 0;
      for await (const chunk of request) {
        bytes += chunk.length;
        if (bytes > 32 * 1024 * 1024) throw new Error('Delegate inference request exceeds 32 MiB.');
        chunks.push(chunk);
      }
      controller.signal.throwIfAborted();
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as SharedRequest;
      if (
        !body.profile?.discovered?.path ||
        body.model?.id !== body.profile.discovered.name ||
        !body.context?.messages
      ) {
        throw new Error('Invalid delegate inference request.');
      }
      // A rejected import stays rejected: never quietly fall back to per-client inference.
      backend ??= options.loadBackend().then((value) => (resolvedBackend = value));
      const engine = await backend;
      response.setHeader('content-type', 'application/x-ndjson');
      for await (const frame of engine.stream(body, controller.signal)) {
        if (controller.signal.aborted) break;
        if (!response.write(`${JSON.stringify(frame)}\n`)) {
          await once(response, 'drain', { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(30_000)]) });
        }
      }
    } catch (error) {
      controller.abort();
      if (!response.destroyed) {
        response.end(`${JSON.stringify({ error: error instanceof Error ? error.message : String(error) })}\n`);
      }
    } finally {
      response.end();
      active--;
      lastActivity = Date.now();
      if (active === 0) requestsDrained?.();
    }
  });
  server.requestTimeout = 60_000;
  server.headersTimeout = 10_000;
  // The kernel is the election lock. Losers exit before loadBackend or any weights.
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(options.port, '127.0.0.1', () => {
      server.off('error', reject);
      resolve();
    });
  });
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Shared inference listener has no port.');
  const endpoint: SharedEndpoint = { protocol: SHARED_PROTOCOL, pid: process.pid, port: address.port, token };
  try {
    await mkdir(options.directory, { recursive: true, mode: 0o700 });
    const temporary = join(options.directory, `endpoint-${process.pid}.json`);
    await writeFile(temporary, JSON.stringify(endpoint), { mode: 0o600 });
    await rename(temporary, join(options.directory, 'endpoint.json'));
  } catch (error) {
    server.close();
    throw error;
  }
  const close = (): Promise<void> => {
    closing ??= (async () => {
      clearInterval(timer);
      if (active > 0)
        await new Promise<void>((resolve) => {
          requestsDrained = resolve;
        });
      // Retain the kernel election lock until native work and resident disposal
      // finish. Unbinding earlier would let a replacement load a second model.
      await resolvedBackend?.close();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    })();
    return closing;
  };
  const idleMs = options.idleMs ?? SHARED_IDLE_MS;
  const timer = setInterval(
    () => {
      if (active === 0 && !resolvedBackend?.busy() && Date.now() - lastActivity >= idleMs) {
        void close().catch((error: unknown) => console.error(error));
      }
    },
    Math.min(idleMs, 1000),
  );
  timer.unref();
  return { endpoint, close };
}
