/** Path-based router for /v1/* endpoints. */

import type { IncomingMessage, ServerResponse } from 'node:http';

import type { ResponseStore } from '@mlx-node/core';

import type { AudioService } from './audio-service.js';
import { handleMemoryRelease, type MemoryReleaseAnswer } from './endpoints/memory-release.js';
import { handleCountMessageTokens } from './endpoints/messages-count-tokens.js';
import { handleCreateMessage } from './endpoints/messages.js';
import { handleListModels, handleUnloadModels, listedFamily } from './endpoints/models.js';
import { handleCreateResponse } from './endpoints/responses.js';
import { handleSystemOne } from './endpoints/systemone.js';
import {
  sendAnthropicBadRequest,
  sendAnthropicMethodNotAllowed,
  sendBadRequest,
  sendMethodNotAllowed,
  sendNotFound,
  sendRequestQuarantined,
} from './errors.js';
import type { PublicModelEntry } from './handler.js';
import { toMinimalHealth, type ServerHealth } from './health.js';
import type { IdleSweeper } from './idle-sweeper.js';
import type { ModelWorkCoordinator } from './model-work-coordinator.js';
import type { ModelRegistry } from './registry.js';
import { ANTHROPIC_MAX_BODY_BYTES } from './request-validation.js';
import type { RerankService } from './rerank-service.js';
import { stringifyStrictJson } from './strict-json.js';
import { QUARANTINE_MESSAGE, rawRequestFingerprint, type TurnJournal } from './turn-journal.js';
import type { AnthropicCountTokensRequest, AnthropicMessagesRequest } from './types-anthropic.js';
import type { ResponsesAPIRequest } from './types.js';

/** Max request body size (10 MB). */
const MAX_BODY_BYTES = 10 * 1024 * 1024;
/** Max request body size for /v1/systemone (16 MiB — accommodates Clairic embedded media). */
const MAX_SYSTEMONE_BODY_BYTES = 16 * 1024 * 1024;
/**
 * Bounded discard window after a request crosses `maxBytes`.
 *
 * Closing a socket immediately while unread body bytes still sit in the kernel
 * receive buffer triggers a TCP RST on macOS/Linux that races with the 400 JSON
 * response and surfaces as `ECONNRESET` in `fetch()` / pooled HTTP clients.
 * Discarding a small tail (with 0 bytes buffered) lets typical oversized uploads
 * read the 400 + `Connection: close` cleanly, while huge or stalled uploads still
 * cut off promptly.
 */
const MAX_OVERSIZE_DRAIN_BYTES = 4 * 1024 * 1024;
const MAX_OVERSIZE_DRAIN_MS = 500;

/**
 * The request's pathname, parsed against a CONSTANT base.
 *
 * Never against `Host`. That header is attacker-controlled text on every
 * request — `Host: [` makes `new URL()` throw `ERR_INVALID_URL`, and a throw
 * from an async request listener is an unhandled rejection, which under Node's
 * default `--unhandled-rejections=throw` takes the whole process down. One
 * malformed byte from any client that can reach the socket was enough to end
 * inference. A pathname does not depend on the authority anyway, so a fixed
 * base is both safer and equivalent: an absolute-form request URI
 * (`GET http://host/v1/models HTTP/1.1`, legal in HTTP/1.1) still wins over
 * the base and yields the same path it always did.
 *
 * `req.url` itself is guarded too, for the same reason rather than a known
 * input: this function's contract is that no request can make it throw.
 */
export function requestPathname(req: IncomingMessage): string {
  const raw = req.url ?? '/';
  try {
    return new URL(raw, 'http://localhost').pathname;
  } catch {
    const cut = raw.search(/[?#]/);
    const path = cut === -1 ? raw : raw.slice(0, cut);
    return path.startsWith('/') ? path : '/';
  }
}

function readBody(req: IncomingMessage, res: ServerResponse, maxBytes: number = MAX_BODY_BYTES): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let totalBytes = 0;
    let draining = false;
    let drainedBytes = 0;
    let drainTimer: ReturnType<typeof setTimeout> | null = null;
    let settled = false;

    const cleanup = () => {
      req.off('data', onData);
      req.off('end', onEnd);
      req.off('error', onError);
      req.off('aborted', onAborted);
      if (drainTimer !== null) {
        clearTimeout(drainTimer);
        drainTimer = null;
      }
    };

    const failTooLarge = (pause: boolean) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (pause) req.pause();
      // Always close after rejecting an oversized body so unread bytes (when
      // cut off early) can never poison a keep-alive connection.
      res.setHeader('Connection', 'close');
      reject(new Error('Request body too large'));
    };

    const onData = (chunk: Buffer) => {
      if (draining) {
        drainedBytes += chunk.length;
        if (drainedBytes >= MAX_OVERSIZE_DRAIN_BYTES) {
          failTooLarge(true);
        }
        return;
      }
      totalBytes += chunk.length;
      if (totalBytes > maxBytes) {
        chunks.length = 0;
        draining = true;
        drainedBytes = totalBytes - maxBytes;
        if (drainedBytes >= MAX_OVERSIZE_DRAIN_BYTES) {
          failTooLarge(true);
          return;
        }
        drainTimer = setTimeout(() => failTooLarge(true), MAX_OVERSIZE_DRAIN_MS);
        drainTimer.unref?.();
        return;
      }
      chunks.push(chunk);
    };
    const onEnd = () => {
      if (draining) {
        failTooLarge(false);
        return;
      }
      if (settled) return;
      settled = true;
      cleanup();
      resolve(Buffer.concat(chunks).toString('utf-8'));
    };
    const onAborted = () => {
      if (draining) {
        failTooLarge(true);
        return;
      }
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error('Request aborted'));
    };
    const onError = (err: Error) => {
      if (draining) {
        failTooLarge(true);
        return;
      }
      if (settled) return;
      settled = true;
      cleanup();
      reject(err);
    };
    req.on('data', onData);
    req.on('end', onEnd);
    req.on('error', onError);
    req.on('aborted', onAborted);
  });
}

/**
 * Trailing options bag. Added as an object rather than two more positional
 * parameters — `routeRequest` already carries nine, and the two knobs here
 * are unrelated to each other.
 */
export interface RouteExtras {
  /** Builds the `/health` body. Omitted ⇒ the legacy constant `{ status: 'ok' }`. */
  health?: () => ServerHealth;
  /**
   * Whether the caller presented a valid token. Only consulted by `/health`,
   * which is the one route reachable without one. `true` when no token is
   * configured at all, so an unprotected server keeps serving the full body.
   */
  authenticated?: boolean;
  audio?: AudioService;
  rerank?: RerankService;
  /** Thinking budget of a Messages thinking request that sets none. */
  thinkingTokenBudget?: number;
  /** The cycle guard of a Messages request. */
  thinkingCycleGuard?: boolean;
  /** The end text of a Messages request's cut thought. */
  thinkingEndText?: string;
  /**
   * `/v1/models` lists each family's thinking levels, and `/v1/messages` and `/v1/messages/count_tokens` map
   * `extra_body.reasoning_effort` through them.
   */
  thinkingLevels?: boolean;
  /** The desktop supervisor's turn journal and quarantine. */
  turnJournal?: TurnJournal;
  /** `POST /v1/memory/release`; omitted ⇒ the route is not served. */
  releaseMemory?: () => Promise<MemoryReleaseAnswer>;
  /** Backs `POST /v1/models/unload`; the route answers 404 without it. */
  unloadModels?: () => Promise<string[]>;
  /** See `ServerConfig.bindLoadedModel`. */
  bindLoadedModel?: (name: string) => boolean;
}

export async function routeRequest(
  req: IncomingMessage,
  res: ServerResponse,
  registry: ModelRegistry,
  store: ResponseStore | null,
  responseRetentionSec?: number,
  idleSweeper?: IdleSweeper | null,
  resolveModel?: (name: string) => Promise<void>,
  listModels?: () => PublicModelEntry[],
  modelWorkCoordinator?: ModelWorkCoordinator,
  extras?: RouteExtras,
): Promise<void> {
  const path = requestPathname(req);
  const levelsFamily =
    extras?.thinkingLevels === true ? (name: string) => listedFamily(name, registry, listModels) : undefined;
  const rejectAudio = (body: unknown): boolean => {
    const name = body && typeof body === 'object' && 'model' in body ? body.model : undefined;
    if (
      typeof name !== 'string' ||
      !(
        registry.audio.get(name) ||
        registry.rerank.get(name) ||
        listModels?.().some((e) => e.id === name && (e.modality === 'audio' || e.task === 'rerank'))
      )
    )
      return false;
    if (path.startsWith('/v1/messages')) sendAnthropicBadRequest(res, 'Model requires its task-specific endpoint');
    else {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({
          error: {
            type: 'invalid_request_error',
            code: 'model_capability_mismatch',
            param: 'model',
            message: 'Model requires its task-specific endpoint',
          },
        }),
      );
    }
    return true;
  };

  if (path === '/v1/rerank' && extras?.rerank) {
    await extras.rerank.handle(req, res);
    return;
  }

  if (path.startsWith('/v1/audio/') && extras?.audio) {
    await extras.audio.handle(req, res, path);
    return;
  }

  if (path === '/v1/systemone') {
    if (req.method !== 'POST') {
      sendMethodNotAllowed(res, 'POST');
      return;
    }
    let raw: string;
    try {
      raw = await readBody(req, res, MAX_SYSTEMONE_BODY_BYTES);
    } catch (error) {
      sendBadRequest(res, error instanceof Error ? error.message : 'Invalid request body');
      return;
    }
    try {
      if (rejectAudio(JSON.parse(raw))) return;
    } catch {
      /* The endpoint reports malformed JSON. */
    }
    const journal = extras?.turnJournal;
    const fingerprint = journal === undefined ? null : rawRequestFingerprint(path, raw);
    if (fingerprint !== null && journal?.refuses(fingerprint) === true) {
      sendRequestQuarantined(res, QUARANTINE_MESSAGE);
      return;
    }
    await handleSystemOne(
      req,
      res,
      raw,
      registry,
      idleSweeper,
      resolveModel,
      modelWorkCoordinator,
      fingerprint === null || journal === undefined ? undefined : () => journal.begin(fingerprint),
    );
    return;
  }

  if (path === '/v1/memory/release' && extras?.releaseMemory) {
    // No inference request: it never moves the idle sweeper or the weights' clock.
    if (req.method !== 'POST') {
      sendMethodNotAllowed(res, 'POST');
      return;
    }
    await handleMemoryRelease(req, res, extras.releaseMemory);
    return;
  }

  if (path === '/v1/models') {
    if (req.method !== 'GET') {
      sendMethodNotAllowed(res, 'GET');
      return;
    }
    handleListModels(res, registry, listModels, extras?.thinkingLevels === true);
    return;
  }

  if (path === '/v1/models/unload' && extras?.unloadModels && modelWorkCoordinator) {
    if (req.method !== 'POST') {
      sendMethodNotAllowed(res, 'POST');
      return;
    }
    await handleUnloadModels(res, extras.unloadModels, registry, modelWorkCoordinator, idleSweeper);
    return;
  }

  if (path === '/v1/responses') {
    if (req.method !== 'POST') {
      sendMethodNotAllowed(res, 'POST');
      return;
    }

    let body: ResponsesAPIRequest;
    try {
      const raw = await readBody(req, res);
      body = JSON.parse(raw) as ResponsesAPIRequest;
    } catch (err) {
      const msg =
        err instanceof Error && err.message === 'Request body too large' ? err.message : 'Invalid JSON in request body';
      sendBadRequest(res, msg);
      return;
    }

    if (rejectAudio(body)) return;
    await handleCreateResponse(
      res,
      body,
      registry,
      store,
      req,
      responseRetentionSec,
      idleSweeper,
      modelWorkCoordinator,
      resolveModel,
      extras?.turnJournal,
      extras?.bindLoadedModel,
    );
    return;
  }

  if (path === '/v1/messages/count_tokens') {
    if (req.method !== 'POST') {
      sendAnthropicMethodNotAllowed(res, 'POST');
      return;
    }

    let body: AnthropicCountTokensRequest;
    try {
      const raw = await readBody(req, res, ANTHROPIC_MAX_BODY_BYTES);
      body = JSON.parse(raw) as AnthropicCountTokensRequest;
    } catch (err) {
      const msg =
        err instanceof Error && err.message === 'Request body too large' ? err.message : 'Invalid JSON in request body';
      sendAnthropicBadRequest(res, msg);
      return;
    }

    if (rejectAudio(body)) return;
    await handleCountMessageTokens(
      res,
      body,
      registry,
      idleSweeper,
      resolveModel,
      modelWorkCoordinator,
      extras?.bindLoadedModel,
      levelsFamily,
    );
    return;
  }

  if (path === '/v1/messages') {
    if (req.method !== 'POST') {
      sendAnthropicMethodNotAllowed(res, 'POST');
      return;
    }

    let body: AnthropicMessagesRequest;
    try {
      const raw = await readBody(req, res, ANTHROPIC_MAX_BODY_BYTES);
      body = JSON.parse(raw) as AnthropicMessagesRequest;
    } catch (err) {
      const msg =
        err instanceof Error && err.message === 'Request body too large' ? err.message : 'Invalid JSON in request body';
      sendAnthropicBadRequest(res, msg);
      return;
    }

    if (rejectAudio(body)) return;
    await handleCreateMessage(
      res,
      body,
      registry,
      req,
      idleSweeper,
      resolveModel,
      modelWorkCoordinator,
      {
        tokenBudget: extras?.thinkingTokenBudget,
        cycleGuard: extras?.thinkingCycleGuard,
        endText: extras?.thinkingEndText,
        ...(levelsFamily !== undefined ? { levelsFamily } : {}),
      },
      extras?.turnJournal,
      extras?.bindLoadedModel,
    );
    return;
  }

  if (path === '/health' || path === '/v1/health') {
    // Deliberately NOT bracketed by `idleSweeper.beginRequest/endRequest`
    // and free of any native call: a supervisor polling on an interval must
    // not keep pushing the drain timer out, nor touch the MLX allocator.
    // Every field is read from plain JavaScript state.
    const health = extras?.health?.();
    if (health === undefined) {
      // No reporter wired (a bare `createHandler` mounted by hand): keep the
      // historical constant so existing consumers are unaffected.
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(stringifyStrictJson({ status: 'ok' }));
      return;
    }
    // Unauthenticated pollers get liveness only. `models.resident` leaks
    // project names and local paths, so it stays behind the token.
    const body = extras?.authenticated === false ? toMinimalHealth(health) : health;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(stringifyStrictJson(body));
    return;
  }

  // Liveness probe at `/`. Claude Code issues `HEAD /` before its first
  // request; respond 200 so the probe doesn't leave a 404 in the logs.
  if (path === '/') {
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      sendMethodNotAllowed(res, 'GET, HEAD');
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(req.method === 'HEAD' ? undefined : stringifyStrictJson({ service: 'mlx-node' }));
    return;
  }

  sendNotFound(res, `No route matches ${req.method} ${path}`);
}
