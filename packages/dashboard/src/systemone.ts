/**
 * System One (Clairic + CLEF) async job runner and history store for the Control Panel.
 *
 * Talks to the local desktop inference sidecar over loopback `node:http` (`POST /v1/systemone` and
 * `POST /v1/models/unload`) using `x-api-key` from `~/.mlx-node/desktop/inference.json`. Stays 100%
 * addon-free (`packages/desktop/__test__/child-entries.test.ts`): only pure type/validation subpaths
 * from `@mlx-node/lm` are imported.
 */

import { randomUUID } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { chmod, mkdir, rename, unlink, writeFile } from 'node:fs/promises';
import { request as httpRequest } from 'node:http';
import { dirname } from 'node:path';

import { validateSystemOneClairicEnvelope } from '@mlx-node/lm/clairic-types';
import { validateClefRequest } from '@mlx-node/lm/clef-types';
import { discoverLocalChatModels } from '@mlx-node/lm/model-discovery';

import { ApiError } from './api/errors.js';
import { safeReader, type EndpointReader, type InferenceEndpoint } from './chat/agent/endpoint.js';
import { engineUrl, MODEL_SELECT_MAX_MS, UNLOAD_MAX_MS, unloadModels } from './chat/agent/select-model.js';
import type { RuntimeLive } from './live.js';
import { systemOneHistoryPath } from './paths.js';

/** Maximum payload size accepted and forwarded to `/v1/systemone` (16 MiB). */
export const SYSTEMONE_MAX_BODY_BYTES = 16 * 1024 * 1024;
/** Maximum response body read from `/v1/systemone` (16 MiB). */
const SYSTEMONE_MAX_RESPONSE_BYTES = 16 * 1024 * 1024;
/** Number of completed runs retained in memory and on disk. */
export const SYSTEMONE_HISTORY_LIMIT = 30;
/** Base64 / data-URI payloads larger than this are stripped before persisting history. */
export const SYSTEMONE_HISTORY_MEDIA_LIMIT_BYTES = 256 * 1024;
/** Overall persisted history file cap (8 MiB). */
const SYSTEMONE_HISTORY_FILE_MAX_BYTES = 8 * 1024 * 1024;

export interface SystemOneModelInfo {
  id: string;
  name: string;
  modelType: 'clairic' | 'clef';
  format: string;
  /** The build discovery names the model by (`Q8_0`), the catalog's build id; `null` for a model only the running endpoint lists. */
  variant: string | null;
  supportsImages: boolean;
  contextWindow: number;
  resident: boolean;
  path: string;
}

export interface SystemOneHistoryEntry {
  id: string;
  model: string;
  mode: 'extract' | 'decide';
  createdAt: number;
  elapsedMs: number;
  status: 'done' | 'error' | 'cancelled';
  fieldCount: number;
  inputTokens: number;
  outputTokens: number;
  request: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: string;
}

export interface SystemOneJobSnapshot {
  id: string;
  status: 'running' | 'done' | 'error' | 'cancelled';
  model: string;
  mode: 'extract' | 'decide';
  startedAt: number;
  elapsedMs: number;
  result?: Record<string, unknown>;
  error?: string;
}

export interface SystemOneState {
  sidecarRunning: boolean;
  models: SystemOneModelInfo[];
  activeJobId: string | null;
  history: SystemOneHistoryEntry[];
}

export interface SystemOneRunInput {
  model: string;
  payload: Record<string, unknown>;
  mode?: 'extract' | 'decide';
  /** When true, execute the job without recording it in persisted run history (e.g. auto-detect fields). */
  skipHistory?: boolean;
}

export interface SystemOneRunResponse {
  jobId: string;
}

export interface SystemOneServiceOptions {
  modelsDir: string;
  readEndpoint: EndpointReader;
  live?: { readNow(): Promise<RuntimeLive> };
  historyPath?: string;
  requestTimeoutMs?: number;
  unloadMaxMs?: number;
  now?: () => number;
}

interface LiveJob {
  snapshot: SystemOneJobSnapshot;
  controller: AbortController | null;
  skipHistory: boolean;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isDecisionEndpointModel(model: InferenceEndpoint['models'][number]): boolean {
  return model.kind === 'decision' || model.modelType === 'clairic' || model.modelType === 'clef';
}

function sanitizeMediaItem(item: unknown): unknown {
  if (typeof item === 'string') {
    const bytes = Buffer.byteLength(item, 'utf8');
    if (bytes > SYSTEMONE_HISTORY_MEDIA_LIMIT_BYTES) {
      return `[omitted media payload: ${bytes} bytes]`;
    }
    return item;
  }
  if (isPlainObject(item) && typeof item.base64 === 'string') {
    const bytes = Buffer.byteLength(item.base64, 'utf8');
    if (bytes > SYSTEMONE_HISTORY_MEDIA_LIMIT_BYTES) {
      const { base64: _dropped, ...rest } = item;
      return { ...rest, omitted: true, bytes };
    }
  }
  return item;
}

/** Strip huge base64 image/PDF blobs from a stored request while preserving text, schema, and small media. */
export function sanitizeRequestForHistory(request: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(request)) {
    if ((key === 'images' || key === 'documents') && Array.isArray(value)) {
      out[key] = value.map(sanitizeMediaItem);
    } else if (typeof value === 'string' && Buffer.byteLength(value, 'utf8') > SYSTEMONE_HISTORY_MEDIA_LIMIT_BYTES) {
      // Preserve text state up to limit; drop oversized opaque strings.
      out[key] = `${value.slice(0, 4096)}… [truncated ${Buffer.byteLength(value, 'utf8')} bytes]`;
    } else {
      out[key] = value;
    }
  }
  return out;
}

function countFields(
  payload: Record<string, unknown>,
  mode: 'extract' | 'decide',
  result?: Record<string, unknown>,
): number {
  const container = mode === 'decide' ? payload.questions : payload.fields;
  if (isPlainObject(container)) return Object.keys(container).length;
  if (result !== undefined && isPlainObject(result.answers)) return Object.keys(result.answers).length;
  return 0;
}

function extractTokens(result: Record<string, unknown> | undefined): { inputTokens: number; outputTokens: number } {
  if (result === undefined || !isPlainObject(result.usage)) return { inputTokens: 0, outputTokens: 0 };
  const usage = result.usage;
  const input =
    typeof usage.input_tokens === 'number' && Number.isFinite(usage.input_tokens)
      ? Math.max(0, Math.floor(usage.input_tokens))
      : 0;
  const output =
    typeof usage.output_tokens === 'number' && Number.isFinite(usage.output_tokens)
      ? Math.max(0, Math.floor(usage.output_tokens))
      : 0;
  return { inputTokens: input, outputTokens: output };
}

function extractErrorMessage(body: string, status: number): string {
  try {
    const parsed = JSON.parse(body) as unknown;
    if (isPlainObject(parsed)) {
      if (
        isPlainObject(parsed.error) &&
        typeof parsed.error.message === 'string' &&
        parsed.error.message.trim() !== ''
      ) {
        return parsed.error.message;
      }
      if (typeof parsed.message === 'string' && parsed.message.trim() !== '') {
        return parsed.message;
      }
    }
  } catch {
    // Non-JSON error body.
  }
  const trimmed = body.trim();
  return trimmed.length > 0 && trimmed.length <= 400 ? trimmed : `System One request failed (HTTP ${status})`;
}

function isHistoryEntry(value: unknown): value is SystemOneHistoryEntry {
  if (!isPlainObject(value)) return false;
  if (typeof value.id !== 'string' || value.id === '') return false;
  if (typeof value.model !== 'string' || value.model === '') return false;
  if (value.mode !== 'extract' && value.mode !== 'decide') return false;
  if (typeof value.createdAt !== 'number' || !Number.isFinite(value.createdAt)) return false;
  if (typeof value.elapsedMs !== 'number' || !Number.isFinite(value.elapsedMs)) return false;
  if (value.status !== 'done' && value.status !== 'error' && value.status !== 'cancelled') return false;
  if (typeof value.fieldCount !== 'number' || !Number.isFinite(value.fieldCount)) return false;
  if (typeof value.inputTokens !== 'number' || !Number.isFinite(value.inputTokens)) return false;
  if (typeof value.outputTokens !== 'number' || !Number.isFinite(value.outputTokens)) return false;
  if (!isPlainObject(value.request)) return false;
  if (value.result !== undefined && !isPlainObject(value.result)) return false;
  if (value.error !== undefined && typeof value.error !== 'string') return false;
  return true;
}

function readHistorySync(path: string): SystemOneHistoryEntry[] {
  try {
    const raw = readFileSync(path, 'utf8');
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isHistoryEntry).slice(0, SYSTEMONE_HISTORY_LIMIT);
  } catch {
    return [];
  }
}

type HttpPostResult =
  | { kind: 'response'; status: number; body: string }
  | { kind: 'aborted' }
  | { kind: 'timeout' }
  | { kind: 'error'; message: string };

function postSystemOne(
  url: URL,
  token: string,
  payload: string,
  options: { timeoutMs: number; signal: AbortSignal },
): Promise<HttpPostResult> {
  if (options.signal.aborted) return Promise.resolve({ kind: 'aborted' });
  return new Promise<HttpPostResult>((resolve) => {
    let settled = false;
    const req = httpRequest(url, {
      method: 'POST',
      agent: false,
      headers: {
        'x-api-key': token,
        'content-type': 'application/json',
        'content-length': Buffer.byteLength(payload, 'utf8'),
      },
    });
    const finish = (result: HttpPostResult): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      options.signal.removeEventListener('abort', onAbort);
      if (result.kind !== 'response') req.destroy();
      resolve(result);
    };
    const onAbort = (): void => finish({ kind: 'aborted' });
    const timer = setTimeout(() => finish({ kind: 'timeout' }), options.timeoutMs);
    timer.unref?.();
    options.signal.addEventListener('abort', onAbort, { once: true });
    req.on('response', (res) => {
      const status = res.statusCode ?? 0;
      const chunks: Buffer[] = [];
      let total = 0;
      let overflow = false;
      res.on('data', (chunk: Buffer) => {
        total += chunk.length;
        if (total <= SYSTEMONE_MAX_RESPONSE_BYTES) chunks.push(chunk);
        else overflow = true;
      });
      res.on('error', (err: Error) => {
        if (options.signal.aborted) finish({ kind: 'aborted' });
        else finish({ kind: 'error', message: err.message || 'Response stream failed' });
      });
      const complete = (): void => {
        if (options.signal.aborted) {
          finish({ kind: 'aborted' });
          return;
        }
        if (overflow) {
          finish({ kind: 'error', message: 'System One response exceeded 16 MiB limit' });
          return;
        }
        finish({ kind: 'response', status, body: Buffer.concat(chunks).toString('utf8') });
      };
      res.on('end', complete);
      res.on('close', complete);
    });
    req.on('error', (err: Error) => {
      if (options.signal.aborted) finish({ kind: 'aborted' });
      else finish({ kind: 'error', message: err.message || 'Could not reach inference engine' });
    });
    req.end(payload);
  });
}

export class SystemOneService {
  private readonly modelsDir: string;
  private readonly readEndpoint: EndpointReader;
  private readonly live: { readNow(): Promise<RuntimeLive> } | undefined;
  private readonly historyPath: string;
  private readonly requestTimeoutMs: number;
  private readonly unloadMaxMs: number;
  private readonly now: () => number;

  private readonly jobs = new Map<string, LiveJob>();
  private readonly subscribers = new Map<string, Set<(snapshot: SystemOneJobSnapshot) => void>>();
  private activeJobId: string | null = null;
  private history: SystemOneHistoryEntry[];
  private historyWrite: Promise<void> = Promise.resolve();
  private closed = false;

  constructor(options: SystemOneServiceOptions) {
    this.modelsDir = options.modelsDir;
    this.readEndpoint = safeReader(options.readEndpoint);
    this.live = options.live;
    this.historyPath = options.historyPath ?? systemOneHistoryPath();
    this.requestTimeoutMs = options.requestTimeoutMs ?? MODEL_SELECT_MAX_MS;
    this.unloadMaxMs = options.unloadMaxMs ?? UNLOAD_MAX_MS;
    this.now = options.now ?? Date.now;
    this.history = readHistorySync(this.historyPath);
  }

  async state(): Promise<SystemOneState> {
    const [endpoint, liveSnapshot, discovered] = await Promise.all([
      this.readEndpoint(),
      this.live?.readNow().catch(() => null) ?? Promise.resolve(null),
      discoverLocalChatModels(this.modelsDir, { includeDecisions: true }).catch(() => []),
    ]);
    const residentNames = new Set<string>();
    if (endpoint !== null && liveSnapshot !== null) {
      for (const proc of liveSnapshot.processes) {
        if (proc.status !== 'running' || proc.pid !== endpoint.pid) continue;
        for (const model of proc.models) residentNames.add(model.name);
      }
    }

    const byName = new Map<string, SystemOneModelInfo>();
    for (const model of discovered) {
      if (model.modelType !== 'clairic' && model.modelType !== 'clef') continue;
      byName.set(model.name, {
        id: model.name,
        name: model.name,
        modelType: model.modelType,
        format: model.format,
        variant: model.variant,
        supportsImages: model.supportsImages,
        contextWindow: model.contextWindow,
        resident: residentNames.has(model.name),
        path: model.path,
      });
    }
    if (endpoint !== null) {
      for (const model of endpoint.models) {
        if (!isDecisionEndpointModel(model)) continue;
        const existing = byName.get(model.name);
        if (existing !== undefined) {
          existing.resident = residentNames.has(model.name);
          continue;
        }
        const modelType: 'clairic' | 'clef' = model.modelType === 'clef' ? 'clef' : 'clairic';
        byName.set(model.name, {
          id: model.name,
          name: model.name,
          modelType,
          format: model.format,
          variant: null,
          supportsImages: modelType === 'clairic',
          contextWindow: modelType === 'clairic' ? 65536 : 16384,
          resident: residentNames.has(model.name),
          path: model.path,
        });
      }
    }

    const models = [...byName.values()].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    return {
      sidecarRunning: endpoint !== null,
      models,
      activeJobId: this.activeJobId,
      history: this.history.map((entry) => ({ ...entry })),
    };
  }

  async run(input: SystemOneRunInput): Promise<SystemOneRunResponse> {
    if (this.closed) throw ApiError.unavailable('Control Panel is shutting down');
    const model = typeof input.model === 'string' ? input.model.trim() : '';
    if (model === '') throw ApiError.badRequest('model is required');
    if (!isPlainObject(input.payload)) throw ApiError.badRequest('payload must be a JSON object');

    const payload = input.payload;
    const mode: 'extract' | 'decide' =
      input.mode === 'extract' || input.mode === 'decide'
        ? input.mode
        : Object.hasOwn(payload, 'questions')
          ? 'decide'
          : 'extract';

    const wireRequest: Record<string, unknown> = { ...payload, model };
    try {
      if (mode === 'decide') validateClefRequest(wireRequest);
      else validateSystemOneClairicEnvelope(wireRequest);
    } catch (error) {
      throw ApiError.badRequest(error instanceof Error ? error.message : 'Invalid System One request');
    }

    const raw = JSON.stringify(wireRequest);
    if (Buffer.byteLength(raw, 'utf8') > SYSTEMONE_MAX_BODY_BYTES) {
      throw ApiError.badRequest('Request body exceeds 16 MiB limit');
    }

    const endpoint = await this.readEndpoint();
    if (endpoint === null) throw ApiError.unavailable('Inference engine is not running');
    const url = engineUrl(endpoint.url, '/v1/systemone');
    if (url === null) throw ApiError.unavailable('Invalid inference engine endpoint');

    const jobId = randomUUID();
    const startedAt = this.now();
    const controller = new AbortController();
    const snapshot: SystemOneJobSnapshot = {
      id: jobId,
      status: 'running',
      model,
      mode,
      startedAt,
      elapsedMs: 0,
    };
    const job: LiveJob = { snapshot, controller, skipHistory: input.skipHistory === true };
    this.jobs.set(jobId, job);
    this.activeJobId = jobId;
    this.emit(jobId, snapshot);

    void this.executeJob(job, url, endpoint.token, raw, wireRequest);
    return { jobId };
  }

  job(id: string): SystemOneJobSnapshot {
    const live = this.jobs.get(id);
    if (live !== undefined) {
      if (live.snapshot.status === 'running') {
        return { ...live.snapshot, elapsedMs: Math.max(0, this.now() - live.snapshot.startedAt) };
      }
      return { ...live.snapshot };
    }
    const historical = this.history.find((entry) => entry.id === id);
    if (historical !== undefined) {
      return {
        id: historical.id,
        status: historical.status,
        model: historical.model,
        mode: historical.mode,
        startedAt: historical.createdAt,
        elapsedMs: historical.elapsedMs,
        ...(historical.result !== undefined ? { result: historical.result } : {}),
        ...(historical.error !== undefined ? { error: historical.error } : {}),
      };
    }
    throw ApiError.notFound(`No such System One job "${id}"`);
  }

  cancel(id: string): { ok: true } {
    const live = this.jobs.get(id);
    if (live === undefined) {
      if (this.history.some((entry) => entry.id === id)) return { ok: true };
      throw ApiError.notFound(`No such System One job "${id}"`);
    }
    if (live.snapshot.status === 'running' && live.controller !== null && !live.controller.signal.aborted) {
      live.controller.abort();
    }
    return { ok: true };
  }

  async clearHistory(): Promise<{ ok: true }> {
    await this.persistHistorySnapshot([], true);
    this.history = [];
    for (const [id, live] of this.jobs) {
      if (live.snapshot.status !== 'running') {
        this.jobs.delete(id);
        this.subscribers.delete(id);
      }
    }
    return { ok: true };
  }

  async unload(): Promise<{ unloaded: string[] }> {
    if (this.closed) throw ApiError.unavailable('Control Panel is shutting down');
    const endpoint = await this.readEndpoint();
    if (endpoint === null) throw ApiError.unavailable('Inference engine is not running');
    const answer = await unloadModels(endpoint, { maxMs: this.unloadMaxMs });
    if (answer === 'busy') {
      throw ApiError.conflict('The engine is busy: an inference request or model load is running.');
    }
    if (typeof answer === 'string') {
      throw ApiError.unavailable('The inference engine could not unload models.');
    }
    return answer;
  }

  watch(jobId: string, listener: (snapshot: SystemOneJobSnapshot) => void): () => void {
    let set = this.subscribers.get(jobId);
    if (set === undefined) {
      set = new Set();
      this.subscribers.set(jobId, set);
    }
    set.add(listener);
    try {
      const current = this.job(jobId);
      listener(current);
    } catch {
      // Unknown job id: stay attached in case caller subscribed just before start, or ignore.
    }
    return () => {
      const listeners = this.subscribers.get(jobId);
      if (listeners === undefined) return;
      listeners.delete(listener);
      if (listeners.size === 0) this.subscribers.delete(jobId);
    };
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    for (const live of this.jobs.values()) {
      if (live.snapshot.status === 'running') live.controller?.abort();
    }
    this.subscribers.clear();
    await this.historyWrite;
  }

  private async executeJob(
    job: LiveJob,
    url: URL,
    token: string,
    raw: string,
    wireRequest: Record<string, unknown>,
  ): Promise<void> {
    const { snapshot, controller } = job;
    const response = await postSystemOne(url, token, raw, {
      timeoutMs: this.requestTimeoutMs,
      signal: controller!.signal,
    });
    const elapsedMs = Math.max(0, this.now() - snapshot.startedAt);
    job.controller = null;

    let finalSnapshot: SystemOneJobSnapshot;
    if (response.kind === 'aborted') {
      finalSnapshot = { ...snapshot, status: 'cancelled', elapsedMs };
    } else if (response.kind === 'timeout') {
      finalSnapshot = { ...snapshot, status: 'error', elapsedMs, error: 'System One request timed out' };
    } else if (response.kind === 'error') {
      finalSnapshot = { ...snapshot, status: 'error', elapsedMs, error: response.message };
    } else if (response.status >= 200 && response.status < 300) {
      try {
        const parsed = JSON.parse(response.body) as unknown;
        if (!isPlainObject(parsed)) throw new Error('Invalid JSON response from inference engine');
        finalSnapshot = { ...snapshot, status: 'done', elapsedMs, result: parsed };
      } catch (err) {
        finalSnapshot = {
          ...snapshot,
          status: 'error',
          elapsedMs,
          error: err instanceof Error ? err.message : 'Invalid response from inference engine',
        };
      }
    } else {
      finalSnapshot = {
        ...snapshot,
        status: 'error',
        elapsedMs,
        error: extractErrorMessage(response.body, response.status),
      };
    }

    job.snapshot = finalSnapshot;
    if (this.activeJobId === snapshot.id) {
      this.activeJobId = null;
      for (const candidate of this.jobs.values()) {
        if (candidate.snapshot.status === 'running') {
          this.activeJobId = candidate.snapshot.id;
          break;
        }
      }
    }

    if (!job.skipHistory) {
      const { inputTokens, outputTokens } = extractTokens(finalSnapshot.result);
      const entry: SystemOneHistoryEntry = {
        id: finalSnapshot.id,
        model: finalSnapshot.model,
        mode: finalSnapshot.mode,
        createdAt: finalSnapshot.startedAt,
        elapsedMs: finalSnapshot.elapsedMs,
        status: finalSnapshot.status === 'running' ? 'error' : finalSnapshot.status,
        fieldCount: countFields(wireRequest, finalSnapshot.mode, finalSnapshot.result),
        inputTokens,
        outputTokens,
        request: sanitizeRequestForHistory(wireRequest),
        ...(finalSnapshot.result !== undefined ? { result: finalSnapshot.result } : {}),
        ...(finalSnapshot.error !== undefined ? { error: finalSnapshot.error } : {}),
      };
      this.history = [entry, ...this.history.filter((h) => h.id !== entry.id)].slice(0, SYSTEMONE_HISTORY_LIMIT);
    }
    this.emit(finalSnapshot.id, finalSnapshot);
    this.pruneFinishedJobs();
    if (!job.skipHistory) {
      void this.persistHistory();
    }
  }

  private pruneFinishedJobs(): void {
    const keep = new Set(this.history.map((h) => h.id));
    const ephemeralFinished: string[] = [];
    for (const [id, live] of this.jobs) {
      if (live.snapshot.status === 'running' || keep.has(id)) continue;
      if (live.skipHistory) {
        ephemeralFinished.push(id);
      } else {
        this.jobs.delete(id);
      }
    }
    // Keep the last 8 finished ephemeral jobs briefly for late polling / reconnect checks.
    const dropCount = Math.max(0, ephemeralFinished.length - 8);
    for (let i = 0; i < dropCount; i += 1) {
      this.jobs.delete(ephemeralFinished[i]!);
    }
  }

  private emit(jobId: string, snapshot: SystemOneJobSnapshot): void {
    const listeners = this.subscribers.get(jobId);
    if (listeners === undefined) return;
    const copy = { ...snapshot };
    for (const listener of Array.from(listeners)) {
      try {
        listener(copy);
      } catch {
        // A broken subscriber must not interrupt other subscribers.
      }
    }
  }

  private persistHistory(): Promise<void> {
    const snapshot = this.history.map((entry) => ({ ...entry }));
    return this.persistHistorySnapshot(snapshot, false);
  }

  private persistHistorySnapshot(snapshot: SystemOneHistoryEntry[], strict: boolean): Promise<void> {
    const file = this.historyPath;
    const write = async (): Promise<void> => {
      try {
        const dir = dirname(file);
        await mkdir(dir, { recursive: true, mode: 0o700 });
        await chmod(dir, 0o700).catch(() => {});
        const trimmed = [...snapshot];
        let json = JSON.stringify(trimmed);
        while (trimmed.length > 1 && Buffer.byteLength(json, 'utf8') > SYSTEMONE_HISTORY_FILE_MAX_BYTES) {
          trimmed.pop();
          json = JSON.stringify(trimmed);
        }
        const temp = `${file}.${randomUUID()}.tmp`;
        try {
          await writeFile(temp, json, { mode: 0o600, flag: 'wx' });
          await rename(temp, file);
        } finally {
          await unlink(temp).catch(() => {});
        }
      } catch (err) {
        if (strict) {
          throw ApiError.internal(
            `Could not clear System One history: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        // Background history persistence is best-effort; never crash the runtime on disk failure.
      }
    };
    const next = this.historyWrite.then(write, write);
    this.historyWrite = next.catch(() => {});
    return next;
  }
}
