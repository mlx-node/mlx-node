/**
 * Request/response logger for `mlx launch claude --verbose`.
 *
 * Each HTTP turn is written as one line of newline-delimited JSON to
 * `requests.ndjson`. Streaming responses capture every chunk written
 * to the socket so SSE events land verbatim — enough to audit cache
 * hits (`x-session-cache` header), tool-call round-trips, and the
 * model's token-level output post-hoc.
 *
 * `session.log` is the human-readable companion: one line per request
 * arrival and completion, for `tail -f` during a live session.
 */

import { Buffer } from 'node:buffer';
import { createWriteStream, mkdirSync } from 'node:fs';
import type { IncomingMessage, Server, ServerResponse } from 'node:http';
import { join } from 'node:path';

export interface Logger {
  /** Absolute log directory in use. */
  readonly logDir: string;
  /** Flush and close the underlying streams. Safe to call multiple times. */
  close(): Promise<void>;
}

/**
 * Attach request/response capture to `server`. The caller is responsible
 * for calling `close()` before `server.close()` completes so the tail of
 * the streams is flushed to disk.
 */
export function attachLogger(server: Server, logDir: string): Logger {
  mkdirSync(logDir, { recursive: true });

  const reqLog = createWriteStream(join(logDir, 'requests.ndjson'), { flags: 'a' });
  const pretty = createWriteStream(join(logDir, 'session.log'), { flags: 'a' });

  const writePretty = (line: string): void => {
    try {
      pretty.write(`${new Date().toISOString()} ${line}\n`);
    } catch {
      /* never let logging failure break serving */
    }
  };

  writePretty(`[logging] writing to ${logDir}`);
  writePretty(`[logging]   requests.ndjson  — one JSON line per HTTP turn (full body in/out, SSE chunks)`);
  writePretty(`[logging]   session.log      — human-readable chronological trace`);

  // Node's http.Server multicasts request events — our listener fires
  // alongside the createServer handler. Dedupe in case the same
  // request ever re-enters (paranoia; it shouldn't).
  const seen = new WeakSet<IncomingMessage>();

  // Prepend so our listener runs BEFORE the main handler. This is
  // what lets the write/end wrappers land before any synchronous
  // response path writes — a sync handler (easy to hit in tests)
  // would otherwise miss the wrap entirely.
  server.prependListener('request', (req: IncomingMessage, res: ServerResponse) => {
    if (seen.has(req)) return;
    seen.add(req);

    const start = Date.now();
    const rid = `${start.toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

    writePretty(`[req ${rid}] ${req.method ?? '?'} ${req.url ?? '?'}`);

    let reqBody = '';
    req.on('data', (chunk: Buffer) => {
      reqBody += chunk.toString('utf8');
    });

    // Wrap res.write / res.end to capture streamed chunks (SSE + regular).
    // Preserve original method semantics — we only observe, never transform.
    const chunks: string[] = [];
    const origWrite = res.write.bind(res);
    const origEnd = res.end.bind(res);

    // biome-ignore lint/suspicious/noExplicitAny: capture wrapper
    (res as any).write = (chunk: unknown, ...rest: unknown[]): boolean => {
      if (chunk != null) {
        try {
          chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk as Uint8Array).toString('utf8'));
        } catch {
          /* ignore */
        }
      }
      return (origWrite as (...a: unknown[]) => boolean)(chunk, ...rest);
    };

    // biome-ignore lint/suspicious/noExplicitAny: capture wrapper
    (res as any).end = (chunk: unknown, ...rest: unknown[]): ServerResponse => {
      if (chunk != null) {
        try {
          chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk as Uint8Array).toString('utf8'));
        } catch {
          /* ignore */
        }
      }
      return (origEnd as (...a: unknown[]) => ServerResponse)(chunk, ...rest);
    };

    // Write the NDJSON entry when BOTH the request body has been fully
    // consumed (`req.on('end')`) AND the response has fully flushed
    // (`res.on('finish')`). Necessary because a synchronous handler can
    // call `res.end()` before our `req.on('data')` listener has seen
    // any chunks — the stream was set flowing on `.on('data', ...)`
    // but the chunks deliver in a later tick.
    let reqDone = false;
    let resDone = false;
    let emitted = false;
    const tryEmit = (): void => {
      if (emitted || !reqDone || !resDone) return;
      emitted = true;
      const resBody = chunks.join('');
      const entry = {
        rid,
        t: new Date(start).toISOString(),
        method: req.method,
        path: req.url,
        reqHeaders: req.headers,
        reqBody: reqBody || null,
        status: res.statusCode,
        resHeaders: res.getHeaders(),
        resBody,
        elapsedMs: Date.now() - start,
      };
      try {
        reqLog.write(`${JSON.stringify(entry)}\n`);
      } catch {
        /* never block the response */
      }
      writePretty(
        `[req ${rid}] ${res.statusCode} ${req.method ?? '?'} ${req.url ?? '?'} ${entry.elapsedMs}ms ${resBody.length}B`,
      );
    };
    req.on('end', () => {
      reqDone = true;
      tryEmit();
    });
    req.on('close', () => {
      // Client may drop the body mid-flight; emit whatever we have.
      reqDone = true;
      tryEmit();
    });
    res.on('finish', () => {
      resDone = true;
      tryEmit();
    });
    res.on('close', () => {
      if (!res.writableEnded) {
        writePretty(`[req ${rid}] aborted (client close) after ${Date.now() - start}ms`);
      }
      // Abort path: treat as done so we still emit what we captured.
      resDone = true;
      tryEmit();
    });
  });

  let closed = false;
  return {
    logDir,
    async close(): Promise<void> {
      if (closed) return;
      closed = true;
      await Promise.all([
        new Promise<void>((resolve) => reqLog.end(resolve)),
        new Promise<void>((resolve) => pretty.end(resolve)),
      ]);
    },
  };
}

/**
 * Resolve the log directory for a verbose launch.
 *
 * Order: explicit `--log-dir` > `MLX_LOG_DIR` env > a fresh timestamped
 * directory under `<mlxNodeHome>/logs/`. The timestamped default gives
 * each launch its own dir so concurrent / sequential runs don't
 * interleave into one file.
 */
export function resolveLogDir(explicit: string | undefined, mlxNodeHome: string): string {
  if (explicit) return explicit;
  const envDir = process.env.MLX_LOG_DIR;
  if (envDir) return envDir;
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  return join(mlxNodeHome, 'logs', stamp);
}
