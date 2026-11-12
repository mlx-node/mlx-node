import { createServer as createHttpServer } from 'node:http';
import { connect } from 'node:net';

import type { SessionCapableModel } from '@mlx-node/lm';
import { createHandler, createServer, ModelRegistry, type ServerInstance } from '@mlx-node/server';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

/**
 * `POST /v1/memory/release` over HTTP, the cases of Splash
 * `test_memory_release.py`: a request takes no fields, a 409 says why the
 * weights stay, and the route needs the API key.
 */
const RELEASED = {
  idle_release_seconds: 600,
  released: true,
  restoring: false,
  restores: 0,
  restore_failures: 0,
};

let servers: ServerInstance[] = [];
afterEach(async () => {
  const open = servers;
  servers = [];
  for (const server of open) await server.close({ timeoutMs: 100 });
});

async function start(over: Parameters<typeof createServer>[0] = {}): Promise<{
  server: ServerInstance;
  calls: { releases: number };
}> {
  const calls = { releases: 0 };
  const server = await createServer({
    port: 0,
    host: '127.0.0.1',
    disableStore: true,
    releaseWeights: async () => {
      calls.releases += 1;
    },
    ...over,
  });
  servers.push(server);
  return { server, calls };
}

function portOf(server: { address(): unknown }): number {
  const address = server.address();
  return address !== null && typeof address === 'object' ? (address as { port: number }).port : 0;
}

/** Exactly these request bytes on a fresh socket; the status, headers and JSON body. */
async function raw(
  port: number,
  method: string,
  headers: Record<string, string>,
  body = '',
): Promise<{ status: number; headers: Record<string, string>; json: any }> {
  const lines = [`${method} /v1/memory/release HTTP/1.1`, 'Host: 127.0.0.1', 'Connection: close'];
  for (const [name, value] of Object.entries(headers)) lines.push(`${name}: ${value}`);
  const text = await new Promise<string>((resolve, reject) => {
    const socket = connect(port, '127.0.0.1');
    let data = '';
    socket.setEncoding('utf-8');
    socket.on('data', (chunk: string) => (data += chunk));
    socket.on('end', () => resolve(data));
    socket.on('error', reject);
    socket.write(`${lines.join('\r\n')}\r\n\r\n${body}`);
  });
  const split = text.indexOf('\r\n\r\n');
  const head = text.slice(0, split).split('\r\n');
  const status = Number(head[0]!.split(' ')[1]);
  const parsed: Record<string, string> = {};
  for (const line of head.slice(1)) {
    const colon = line.indexOf(':');
    parsed[line.slice(0, colon).toLowerCase()] = line.slice(colon + 1).trim();
  }
  let payload = text.slice(split + 4);
  if (parsed['transfer-encoding'] === 'chunked') {
    const lineEnd = payload.indexOf('\r\n');
    const size = Number.parseInt(payload.slice(0, lineEnd), 16);
    payload = payload.slice(lineEnd + 2, lineEnd + 2 + size);
  }
  return { status, headers: parsed, json: payload.length > 0 ? JSON.parse(payload) : null };
}

describe('POST /v1/memory/release', () => {
  it('releases for a request without fields', async () => {
    const { server, calls } = await start();
    const port = portOf(server.server);
    for (const [headers, body] of [
      [{}, ''],
      [{ 'Content-Length': '0' }, ''],
      [{ 'Content-Type': 'application/json', 'Content-Length': '2' }, '{}'],
      [{ 'Content-Type': 'application/json', 'Content-Length': '10' }, '{"x":null}'],
    ] as const) {
      const answer = await raw(port, 'POST', headers, body);
      expect([answer.status, answer.json]).toEqual([200, { released: true, weights: RELEASED }]);
    }
    expect(calls.releases).toBe(4);
  });

  it('refuses a body with fields or not JSON before releasing', async () => {
    const { server, calls } = await start();
    const port = portOf(server.server);
    for (const [headers, body, status, message] of [
      [{ 'Content-Type': 'application/json', 'Content-Length': '7' }, '{"x":1}', 400, 'this endpoint takes no fields'],
      [{ 'Content-Type': 'application/json', 'Content-Length': '2' }, '[]', 400, 'request body must be an object'],
      [{ 'Content-Type': 'text/plain', 'Content-Length': '2' }, '{}', 415, 'Content-Type must be application/json'],
      [
        { 'Content-Type': 'application/json', 'Transfer-Encoding': 'chunked' },
        '2\r\n{}\r\n0\r\n\r\n',
        400,
        'transfer encoding is not supported',
      ],
    ] as const) {
      const answer = await raw(port, 'POST', headers, body);
      expect([answer.status, answer.json.error.message]).toEqual([status, message]);
    }
    expect(calls.releases).toBe(0);
  });

  // Mutation M13: answer 200 while busy → red.
  it('answers 409 engine_busy while a request is in service or a load runs', async () => {
    const { server, calls } = await start();
    const port = portOf(server.server);

    server.weights!.beginRequest();
    let answer = await raw(port, 'POST', {});
    expect([answer.status, answer.json.error.code]).toEqual([409, 'engine_busy']);
    expect(answer.json.error.message).toContain('serving requests');
    server.weights!.endRequest();

    let finish!: () => void;
    const loading = server.modelWork.withModelLoad(() => new Promise<void>((resolve) => (finish = resolve)));
    answer = await raw(port, 'POST', {});
    expect([answer.status, answer.json.error.code]).toEqual([409, 'engine_busy']);
    finish();
    await loading;
    expect(calls.releases).toBe(0);

    // Released, then a reload runs: the weights are restoring.
    expect((await raw(port, 'POST', {})).status).toBe(200);
    const reload = server.withSuspendedDrains(() => new Promise<void>((resolve) => (finish = resolve)));
    answer = await raw(port, 'POST', {});
    expect([answer.status, answer.json.error.code]).toEqual([409, 'engine_busy']);
    expect(answer.json.error.message).toContain('restoring its weights');
    expect(server.health().weights).toMatchObject({ released: true, restoring: true });
    finish();
    await reload;
    expect(calls.releases).toBe(1);
  });

  // Mutation M17: a response persist's retain is not counted as busy → red.
  it('answers 409 and keeps the model while a response persist retains it', async () => {
    const dispose = vi.fn(async () => undefined);
    const model = { chatSessionStart: vi.fn(), resetCaches: vi.fn(), dispose } as unknown as SessionCapableModel;
    let instance!: ServerInstance;
    const { server } = await start({
      releaseWeights: async () => {
        for (const entry of instance.registry.list()) instance.registry.unregister(entry.id, { dispose: true });
        await instance.registry.settleModelDisposals();
      },
    });
    instance = server;
    const port = portOf(server.server);
    server.registry.register('model-a', model);
    server.registry.retainBinding(model);

    let answer = await raw(port, 'POST', {});
    expect([answer.status, answer.json.error.code]).toEqual([409, 'engine_busy']);
    expect(server.registry.get('model-a')).toBeDefined();
    expect(dispose).not.toHaveBeenCalled();
    expect(server.health().weights).toMatchObject({ released: false });

    server.registry.releaseBinding(model);
    answer = await raw(port, 'POST', {});
    expect([answer.status, answer.json.released]).toEqual([200, true]);
    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it('answers 405 to another method and 404 when nothing can be released', async () => {
    const { server } = await start();
    expect((await raw(portOf(server.server), 'GET', {})).status).toBe(405);
    const bare = await createServer({ port: 0, host: '127.0.0.1', disableStore: true });
    servers.push(bare);
    expect((await raw(portOf(bare.server), 'POST', {})).status).toBe(404);
  });

  it('releases with --idle-release off, whose seconds are null', async () => {
    const { server, calls } = await start({ idleReleaseMs: Infinity });
    const answer = await raw(portOf(server.server), 'POST', {});
    expect([answer.status, answer.json]).toEqual([
      200,
      { released: true, weights: { ...RELEASED, idle_release_seconds: null } },
    ]);
    expect(calls.releases).toBe(1);
  });

  it('needs the API key', async () => {
    const { server, calls } = await start({ authToken: 'test-server-key' });
    const port = portOf(server.server);
    expect((await raw(port, 'POST', {})).status).toBe(401);
    expect((await raw(port, 'POST', { Authorization: 'Bearer test-server-key' })).status).toBe(200);
    expect(calls.releases).toBe(1);
  });

  it('answers 503 server_shutdown while the server closes', async () => {
    const http = createHttpServer(
      createHandler(new ModelRegistry(), { releaseMemory: async () => ({ kind: 'closing' }) }),
    );
    await new Promise<void>((resolve) => http.listen(0, '127.0.0.1', () => resolve()));
    try {
      const answer = await raw(portOf(http), 'POST', {});
      expect([answer.status, answer.json.error.code, answer.json.error.message]).toEqual([
        503,
        'server_shutdown',
        'server is shutting down',
      ]);
      expect(answer.headers['retry-after']).toBe('1');
    } finally {
      await new Promise<void>((resolve) => http.close(() => resolve()));
    }
  });
});
