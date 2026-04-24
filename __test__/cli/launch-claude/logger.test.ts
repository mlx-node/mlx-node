import { mkdtempSync, readFileSync } from 'node:fs';
import { createServer, request as httpRequest } from 'node:http';
import type { Server } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { attachLogger, resolveLogDir } from '../../../packages/cli/src/commands/launch-claude/logger.js';

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
});
