import { mkdtempSync, readFileSync } from 'node:fs';
import { createServer, request as httpRequest } from 'node:http';
import type { Server } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { attachLogger, resolveLogDir } from '../../../packages/server/src/host/logger.js';

function makeTmpDir(): string {
  return mkdtempSync(join(tmpdir(), 'mlx-log-test-'));
}

function pickFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const probe = createServer();
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const addr = probe.address();
      if (addr && typeof addr === 'object') {
        const port = addr.port;
        probe.close(() => resolve(port));
      } else {
        probe.close(() => reject(new Error('no port')));
      }
    });
  });
}

function closeServer(srv: Server): Promise<void> {
  return new Promise((resolve) => srv.close(() => resolve()));
}

function postJson(port: number, body: unknown): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(body);
    const req = httpRequest(
      {
        host: '127.0.0.1',
        port,
        method: 'POST',
        path: '/echo',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
      },
      (res) => {
        let out = '';
        res.on('data', (c: Buffer) => {
          out += c.toString('utf8');
        });
        res.on('end', () => resolve({ status: res.statusCode ?? 0, body: out }));
      },
    );
    req.on('error', reject);
    req.write(payload);
    req.end();
  });
}

describe('resolveLogDir', () => {
  const prevEnv = process.env.MLX_LOG_DIR;
  afterEach(() => {
    if (prevEnv === undefined) delete process.env.MLX_LOG_DIR;
    else process.env.MLX_LOG_DIR = prevEnv;
  });

  it('prefers explicit over env over default', () => {
    process.env.MLX_LOG_DIR = '/from/env';
    expect(resolveLogDir('/from/flag', '/home')).toBe('/from/flag');
    expect(resolveLogDir(undefined, '/home')).toBe('/from/env');
  });

  it('falls back to a timestamped dir under mlxNodeHome/logs', () => {
    delete process.env.MLX_LOG_DIR;
    const got = resolveLogDir(undefined, '/home/me/.mlx-node');
    expect(got.startsWith('/home/me/.mlx-node/logs/')).toBe(true);
    expect(got.length).toBeGreaterThan('/home/me/.mlx-node/logs/'.length);
  });
});

describe('attachLogger', () => {
  it('captures one NDJSON line per request with req+res bodies', async () => {
    const logDir = makeTmpDir();
    const port = await pickFreePort();

    const srv = createServer((_req, res) => {
      res.setHeader('content-type', 'application/json');
      res.writeHead(200);
      res.write('{"hello":');
      res.end('"world"}');
    });
    const logger = attachLogger(srv, logDir);
    await new Promise<void>((resolve) => srv.listen(port, '127.0.0.1', resolve));

    const reply = await postJson(port, { ping: 1 });
    expect(reply.status).toBe(200);
    expect(reply.body).toBe('{"hello":"world"}');

    await logger.close();
    await closeServer(srv);

    const lines = readFileSync(join(logDir, 'requests.ndjson'), 'utf-8').trim().split('\n');
    expect(lines).toHaveLength(1);
    const row = JSON.parse(lines[0]) as {
      method: string;
      path: string;
      status: number;
      reqBody: string;
      resBody: string;
      elapsedMs: number;
    };
    expect(row.method).toBe('POST');
    expect(row.path).toBe('/echo');
    expect(row.status).toBe(200);
    expect(row.reqBody).toBe('{"ping":1}');
    expect(row.resBody).toBe('{"hello":"world"}');
    expect(row.elapsedMs).toBeGreaterThanOrEqual(0);

    const pretty = readFileSync(join(logDir, 'session.log'), 'utf-8');
    expect(pretty).toContain('POST /echo');
    expect(pretty).toContain('200 POST /echo');
  });

  it('captures streamed chunks in order (SSE-style)', async () => {
    const logDir = makeTmpDir();
    const port = await pickFreePort();

    const srv = createServer((_req, res) => {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.write('data: a\n\n');
      res.write('data: b\n\n');
      res.end('data: [DONE]\n\n');
    });
    const logger = attachLogger(srv, logDir);
    await new Promise<void>((resolve) => srv.listen(port, '127.0.0.1', resolve));

    await postJson(port, { stream: true });
    await logger.close();
    await closeServer(srv);

    const row = JSON.parse(readFileSync(join(logDir, 'requests.ndjson'), 'utf-8').trim()) as { resBody: string };
    expect(row.resBody).toBe('data: a\n\ndata: b\n\ndata: [DONE]\n\n');
  });

  it('logs total first-token latency separately from native ttft', async () => {
    const logDir = makeTmpDir();
    const port = await pickFreePort();

    const srv = createServer((_req, res) => {
      res.setHeader('content-type', 'application/json');
      res.end(
        JSON.stringify({
          model: 'gemma-test',
          stop_reason: 'end_turn',
          usage: {
            input_tokens: 20,
            output_tokens: 5,
            time_to_first_token_ms: 250,
            prefill_tokens_per_second: 20,
            decode_tokens_per_second: 40,
            server_inference_elapsed_ms: 350,
            server_total_time_to_first_token_ms: 312,
            server_model_resolve_ms: 57,
            server_queue_ms: 5,
            server_pre_inference_ms: 62,
          },
        }),
      );
    });
    const logger = attachLogger(srv, logDir);
    await new Promise<void>((resolve) => srv.listen(port, '127.0.0.1', resolve));

    await postJson(port, { model: 'gemma-test' });
    await logger.close();
    await closeServer(srv);

    const pretty = readFileSync(join(logDir, 'session.log'), 'utf-8');
    expect(pretty).toContain('perf(ttfb=312ms ttft=250ms prefill=20.00/s decode=40.00/s infer=350ms)');
    expect(pretty).toContain('server(resolve=57ms queue=5ms pre=62ms)');
  });

  it('surfaces load_wait and load_owner in the server(...) summary when present', async () => {
    // Observability split: a follower request that arrives during a peer's
    // cold load gets `load_wait_ms` ≈ cold load latency and
    // `load_owner=false`. Without surfacing the split, the follower's
    // `resolve_ms` would be indistinguishable from a request that drove
    // the load itself.
    const logDir = makeTmpDir();
    const port = await pickFreePort();

    const srv = createServer((_req, res) => {
      res.setHeader('content-type', 'application/json');
      res.end(
        JSON.stringify({
          model: 'gemma-test',
          stop_reason: 'end_turn',
          usage: {
            input_tokens: 20,
            output_tokens: 5,
            server_model_resolve_ms: 60_300,
            server_load_wait_ms: 60_290,
            server_load_owner: false,
            server_queue_ms: 1,
            server_pre_inference_ms: 60_302,
          },
        }),
      );
    });
    const logger = attachLogger(srv, logDir);
    await new Promise<void>((resolve) => srv.listen(port, '127.0.0.1', resolve));

    await postJson(port, { model: 'gemma-test' });
    await logger.close();
    await closeServer(srv);

    const pretty = readFileSync(join(logDir, 'session.log'), 'utf-8');
    expect(pretty).toContain('server(resolve=60300ms load_wait=60290ms load_owner=false queue=1ms pre=60302ms)');
  });

  it('redacts credential headers, keeping the header names', async () => {
    // `requests.ndjson` is what users paste into bug reports, and every client
    // of a protected host presents a live secret on every turn — the per-launch
    // token `mlx launch claude` generates, or `MLX_SERVER_AUTH_TOKEN`. The
    // NAMES stay: "presented a token and it was wrong" and "presented nothing"
    // are different bugs, and a log that drops the header cannot tell them
    // apart.
    const logDir = makeTmpDir();
    const port = await pickFreePort();

    const srv = createServer((_req, res) => {
      res.writeHead(200).end('{}');
    });
    const logger = attachLogger(srv, logDir);
    await new Promise<void>((resolve) => srv.listen(port, '127.0.0.1', resolve));

    await new Promise<void>((resolve, reject) => {
      const req = httpRequest(
        {
          host: '127.0.0.1',
          port,
          method: 'GET',
          path: '/v1/models',
          headers: {
            authorization: 'Bearer the-live-bearer-token',
            'x-api-key': 'sk-ant-the-users-own-key',
            'proxy-authorization': 'Basic the-proxy-secret',
            cookie: 'session=the-session-cookie',
            'anthropic-version': '2023-06-01',
          },
        },
        (res) => {
          res.resume();
          res.on('end', () => resolve());
        },
      );
      req.on('error', reject);
      req.end();
    });

    await logger.close();
    await closeServer(srv);

    const raw = readFileSync(join(logDir, 'requests.ndjson'), 'utf-8');
    for (const secret of [
      'the-live-bearer-token',
      'sk-ant-the-users-own-key',
      'the-proxy-secret',
      'the-session-cookie',
    ]) {
      expect(raw, secret).not.toContain(secret);
    }
    const row = JSON.parse(raw.trim()) as { reqHeaders: Record<string, string> };
    expect(row.reqHeaders.authorization).toBe('[redacted]');
    expect(row.reqHeaders['x-api-key']).toBe('[redacted]');
    expect(row.reqHeaders['proxy-authorization']).toBe('[redacted]');
    expect(row.reqHeaders.cookie).toBe('[redacted]');
    // Everything else still lands verbatim — the log's whole job.
    expect(row.reqHeaders['anthropic-version']).toBe('2023-06-01');
  });

  it('leaves a request with no credentials untouched', async () => {
    const logDir = makeTmpDir();
    const port = await pickFreePort();

    const srv = createServer((_req, res) => {
      res.writeHead(200).end('{}');
    });
    const logger = attachLogger(srv, logDir);
    await new Promise<void>((resolve) => srv.listen(port, '127.0.0.1', resolve));

    await postJson(port, { ping: 1 });
    await logger.close();
    await closeServer(srv);

    const row = JSON.parse(readFileSync(join(logDir, 'requests.ndjson'), 'utf-8').trim()) as {
      reqHeaders: Record<string, string>;
    };
    // A blanket `'[redacted]'` for absent headers would invent a credential
    // that was never presented.
    expect('authorization' in row.reqHeaders).toBe(false);
    expect('x-api-key' in row.reqHeaders).toBe(false);
    expect(row.reqHeaders['content-type']).toBe('application/json');
  });

  /*
   * A request that finishes after `close()` has ended the log streams writes
   * from its own `finish` handler. `write()` after `end()` reports
   * `ERR_STREAM_WRITE_AFTER_END` asynchronously, as an `error` event — the
   * `try`/`catch` around each write cannot see it, and with no listener the
   * event is fatal. The host closes the server first so this is rare, but the
   * forced-close path still emits `res.on('close')` a tick later, so the
   * listener is the thing that has to hold.
   *
   * The window is `writableEnded && !destroyed`. Once `end()` finishes
   * flushing, `destroyed` is set and Node drops the error silently, which is
   * why this deliberately does NOT await `close()` — awaiting it would test
   * the harmless case and pass with the listeners removed.
   */
  it('survives a request completing while close() is still flushing', async () => {
    const logDir = makeTmpDir();
    const port = await pickFreePort();

    let release: (() => void) | null = null;
    const srv = createServer((req, res) => {
      if (req.url === '/big') {
        // Captured verbatim into `requests.ndjson`, so this response is what
        // puts megabytes into the stream's queue and holds `end()` flushing
        // long enough for the second request's write to land inside the
        // window. Without it `end()` completes first, the stream is
        // `destroyed`, and Node discards the error instead of throwing —
        // which would make this test pass with the listeners removed.
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        const chunk = `data: ${'x'.repeat(64 * 1024)}\n\n`;
        for (let i = 0; i < 128; i++) res.write(chunk);
        res.end();
        return;
      }
      release = () => res.writeHead(200).end('{}');
    });
    const logger = attachLogger(srv, logDir);

    // Registered after `attachLogger`, so for `/big` the logger's own `finish`
    // listener — and its multi-MB write — has already run when this fires.
    srv.on('request', (req, res) => {
      if (req.url !== '/big') return;
      res.on('finish', () => {
        void logger.close();
        release?.();
      });
    });
    await new Promise<void>((resolve) => srv.listen(port, '127.0.0.1', resolve));

    const inFlight = postJson(port, { ping: 1 });
    while (release === null) await new Promise((r) => setTimeout(r, 5));

    // An uncaught `error` would end the whole run rather than fail this
    // assertion, so it is captured instead: the handler makes the
    // would-be-fatal event observable without letting it be fatal.
    const fatal: string[] = [];
    const onUncaught = (err: Error): void => {
      fatal.push((err as NodeJS.ErrnoException).code ?? err.message);
    };
    process.on('uncaughtException', onUncaught);
    try {
      await new Promise<void>((resolve, reject) => {
        const req = httpRequest({ host: '127.0.0.1', port, method: 'GET', path: '/big' }, (res) => {
          res.on('data', () => {});
          res.on('end', () => resolve());
        });
        req.on('error', reject);
        req.end();
      });
      await inFlight;
      // Let the streams' async error events land.
      await new Promise((r) => setTimeout(r, 200));
    } finally {
      process.off('uncaughtException', onUncaught);
    }
    await closeServer(srv);

    expect(fatal).toEqual([]);
  });
});
