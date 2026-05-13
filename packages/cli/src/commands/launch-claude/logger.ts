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

interface UsageSummary {
  input_tokens?: number;
  cache_read_input_tokens?: number;
  input_tokens_details?: { cached_tokens?: number };
  output_tokens?: number;
  time_to_first_token_ms?: number;
  prefill_tokens_per_second?: number;
  decode_tokens_per_second?: number;
  server_inference_elapsed_ms?: number;
  server_total_time_to_first_token_ms?: number;
  prefill_input_tokens?: number;
  cached_prefix_tokens?: number;
  server_model_resolve_ms?: number;
  server_load_wait_ms?: number;
  server_load_owner?: boolean;
  server_queue_ms?: number;
  server_pre_inference_ms?: number;
  server_paged_prefill_chunk_size?: number;
  server_paged_prefill_eval_interval?: number;
  server_paged_decode_cache_clear_interval?: number;
}

function parseJsonObject(value: string | null): Record<string, unknown> | null {
  if (!value) return null;
  try {
    const parsed: unknown = JSON.parse(value);
    return parsed != null && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

function asUsage(value: unknown): UsageSummary | undefined {
  return value != null && typeof value === 'object' ? (value as UsageSummary) : undefined;
}

function fmtMs(ms: number | undefined): string | undefined {
  return typeof ms === 'number' && Number.isFinite(ms) ? `${Math.round(ms)}ms` : undefined;
}

function fmtRate(rate: number | undefined): string | undefined {
  return typeof rate === 'number' && Number.isFinite(rate) ? `${rate.toFixed(2)}/s` : undefined;
}

function addMs(left: number | undefined, right: number | undefined): number | undefined {
  return typeof left === 'number' && Number.isFinite(left) && typeof right === 'number' && Number.isFinite(right)
    ? left + right
    : undefined;
}

function extractUsageSummary(resBody: string): { model?: string; usage?: UsageSummary; stop?: string } {
  let model: string | undefined;
  let usage: UsageSummary | undefined;
  let stop: string | undefined;

  const trimmed = resBody.trimStart();
  if (trimmed.startsWith('{')) {
    const json = parseJsonObject(trimmed);
    if (json) {
      const response =
        json.response != null && typeof json.response === 'object' ? (json.response as Record<string, unknown>) : json;
      model = typeof response.model === 'string' ? response.model : undefined;
      usage = asUsage(response.usage);
      stop = typeof response.stop_reason === 'string' ? response.stop_reason : undefined;
      return { model, usage, stop };
    }
  }

  for (const line of resBody.split('\n')) {
    if (!line.startsWith('data: ')) continue;
    const payload = line.slice(6);
    if (payload === '[DONE]') continue;
    const event = parseJsonObject(payload);
    if (!event) continue;
    if (event.type === 'message_start' && event.message != null && typeof event.message === 'object') {
      const message = event.message as Record<string, unknown>;
      if (typeof message.model === 'string') model = message.model;
    }
    if (event.type === 'message_delta') {
      usage = asUsage(event.usage) ?? usage;
      const delta =
        event.delta != null && typeof event.delta === 'object' ? (event.delta as Record<string, unknown>) : null;
      if (typeof delta?.stop_reason === 'string') stop = delta.stop_reason;
    }
    if (
      (event.type === 'response.completed' || event.type === 'response.failed') &&
      event.response != null &&
      typeof event.response === 'object'
    ) {
      const response = event.response as Record<string, unknown>;
      if (typeof response.model === 'string') model = response.model;
      usage = asUsage(response.usage) ?? usage;
      if (typeof response.status === 'string') stop = response.status;
    }
  }

  return { model, usage, stop };
}

function buildTimingSummary(reqBody: string, resBody: string): string {
  const request = parseJsonObject(reqBody);
  const response = extractUsageSummary(resBody);
  const model = typeof request?.model === 'string' ? request.model : response.model;
  const usage = response.usage;
  if (!usage && !model) return '';

  const parts: string[] = [];
  if (model) parts.push(`model=${model}`);
  if (usage) {
    const cachedTokens =
      usage.cache_read_input_tokens ?? usage.cached_prefix_tokens ?? usage.input_tokens_details?.cached_tokens;
    const tokenParts = [
      typeof usage.input_tokens === 'number' ? `in=${usage.input_tokens}` : undefined,
      typeof cachedTokens === 'number' ? `cache=${cachedTokens}` : undefined,
      typeof usage.prefill_input_tokens === 'number' ? `prefill=${usage.prefill_input_tokens}` : undefined,
      typeof usage.output_tokens === 'number' ? `out=${usage.output_tokens}` : undefined,
    ].filter((part): part is string => part != null);
    if (tokenParts.length > 0) parts.push(`tok(${tokenParts.join(' ')})`);

    const totalFirstTokenMs =
      usage.server_total_time_to_first_token_ms ?? addMs(usage.server_pre_inference_ms, usage.time_to_first_token_ms);
    const timingParts = [
      fmtMs(totalFirstTokenMs) ? `ttfb=${fmtMs(totalFirstTokenMs)}` : undefined,
      fmtMs(usage.time_to_first_token_ms) ? `ttft=${fmtMs(usage.time_to_first_token_ms)}` : undefined,
      fmtRate(usage.prefill_tokens_per_second) ? `prefill=${fmtRate(usage.prefill_tokens_per_second)}` : undefined,
      fmtRate(usage.decode_tokens_per_second) ? `decode=${fmtRate(usage.decode_tokens_per_second)}` : undefined,
      fmtMs(usage.server_inference_elapsed_ms) ? `infer=${fmtMs(usage.server_inference_elapsed_ms)}` : undefined,
    ].filter((part): part is string => part != null);
    if (timingParts.length > 0) parts.push(`perf(${timingParts.join(' ')})`);

    const serverParts = [
      fmtMs(usage.server_model_resolve_ms) ? `resolve=${fmtMs(usage.server_model_resolve_ms)}` : undefined,
      fmtMs(usage.server_load_wait_ms) ? `load_wait=${fmtMs(usage.server_load_wait_ms)}` : undefined,
      typeof usage.server_load_owner === 'boolean' ? `load_owner=${usage.server_load_owner}` : undefined,
      fmtMs(usage.server_queue_ms) ? `queue=${fmtMs(usage.server_queue_ms)}` : undefined,
      fmtMs(usage.server_pre_inference_ms) ? `pre=${fmtMs(usage.server_pre_inference_ms)}` : undefined,
    ].filter((part): part is string => part != null);
    if (serverParts.length > 0) parts.push(`server(${serverParts.join(' ')})`);

    const tuningParts = [
      typeof usage.server_paged_prefill_chunk_size === 'number'
        ? `prefill_chunk=${usage.server_paged_prefill_chunk_size}`
        : undefined,
      typeof usage.server_paged_prefill_eval_interval === 'number'
        ? `prefill_eval=${usage.server_paged_prefill_eval_interval}`
        : undefined,
      typeof usage.server_paged_decode_cache_clear_interval === 'number'
        ? `decode_clear=${usage.server_paged_decode_cache_clear_interval}`
        : undefined,
    ].filter((part): part is string => part != null);
    if (tuningParts.length > 0) parts.push(`tune(${tuningParts.join(' ')})`);
  }
  if (response.stop) parts.push(`stop=${response.stop}`);

  return parts.length > 0 ? ` ${parts.join(' ')}` : '';
}

function buildRequestBodySummary(reqBody: string): string {
  const request = parseJsonObject(reqBody);
  if (!request) return '';

  const parts: string[] = [];
  if (typeof request.model === 'string') parts.push(`model=${request.model}`);
  if (typeof request.max_tokens === 'number') parts.push(`max_tokens=${request.max_tokens}`);
  if (typeof request.stream === 'boolean') parts.push(`stream=${request.stream}`);
  if (Array.isArray(request.messages)) parts.push(`messages=${request.messages.length}`);
  if (Array.isArray(request.tools)) parts.push(`tools=${request.tools.length}`);
  if (typeof request.system === 'string') {
    parts.push(`system=string`);
  } else if (Array.isArray(request.system)) {
    parts.push(`system=blocks:${request.system.length}`);
  }

  return parts.length > 0 ? ` ${parts.join(' ')}` : '';
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
  if (process.env.MLX_INFERENCE_TRACE_FILE) {
    writePretty(`[logging]   inference trace — ${process.env.MLX_INFERENCE_TRACE_FILE}`);
  }

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
    let requestBodyLogged = false;
    const logRequestBody = (phase: 'end' | 'close'): void => {
      if (requestBodyLogged) return;
      requestBodyLogged = true;
      writePretty(
        `[req ${rid}] request_body_${phase} ${Buffer.byteLength(reqBody, 'utf8')}B${buildRequestBodySummary(reqBody)}`,
      );
    };
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
      const timingSummary = buildTimingSummary(reqBody, resBody);
      writePretty(
        `[req ${rid}] ${res.statusCode} ${req.method ?? '?'} ${req.url ?? '?'} ${entry.elapsedMs}ms ${resBody.length}B${timingSummary}`,
      );
    };
    req.on('end', () => {
      reqDone = true;
      logRequestBody('end');
      tryEmit();
    });
    req.on('close', () => {
      // Client may drop the body mid-flight; emit whatever we have.
      reqDone = true;
      logRequestBody('close');
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
