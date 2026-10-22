import type { IncomingMessage, ServerResponse } from 'node:http';

import { parseRerankBody, RerankError, rerankError, rerankWireResponse } from '@mlx-node/rerank/protocol';
import type { RerankRequest, RerankResponse } from '@mlx-node/rerank/types';

import type { PublicModelEntry } from './handler.js';
import type { IdleSweeper } from './idle-sweeper.js';
import { ModelLoadQueueFullError, type ModelWorkCoordinator } from './model-work-coordinator.js';
import type { ModelRegistry } from './registry.js';
export interface RerankServiceOptions {
  maxBodyBytes?: number;
  maxDocuments?: number;
  timeoutMs?: number;
  maxConcurrentRequests?: number;
}
/** Validate before opening server resources; also used by standalone handlers. */
export function resolveRerankLimits(options: RerankServiceOptions = {}): Required<RerankServiceOptions> {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw new Error('Invalid rerank service options');
  for (const key of Object.keys(options))
    if (!['maxBodyBytes', 'maxDocuments', 'timeoutMs', 'maxConcurrentRequests'].includes(key))
      throw new Error(`Unknown rerank service option: ${key}`);
  const limits = {
    maxBodyBytes: 16 * 1024 * 1024,
    maxDocuments: 1024,
    timeoutMs: 300_000,
    maxConcurrentRequests: 17,
    ...options,
  };
  for (const [key, value] of Object.entries(limits))
    if (!Number.isSafeInteger(value) || value < 1) throw new Error(`Invalid rerank ${key}`);
  // Node clamps larger delays to 1 ms, which would immediately expire every request.
  if (limits.timeoutMs > 2 ** 31 - 1) throw new Error('Rerank timeoutMs exceeds the Node timer range');
  return limits;
}
interface Dependencies {
  registry: ModelRegistry;
  coordinator: ModelWorkCoordinator;
  idleSweeper?: IdleSweeper | null;
  resolveModel?: (name: string) => Promise<void>;
  listModels?: () => PublicModelEntry[];
}
export interface RerankMetrics {
  task: 'rerank';
  model?: string;
  state: 'running' | 'completed' | 'error' | 'cancelled';
  errorCode?: string;
  modelLoadMs?: number;
  queueWaitMs?: number;
  computeMs?: number;
  totalMs?: number;
  profile?: string;
  quantized?: boolean;
  weightFormats?: readonly string[];
  documents?: number;
  blocks?: number;
  batches?: number;
  paddingTokens?: number;
  promptTokens?: number;
  truncatedDocuments?: number;
}
const metrics = new WeakMap<ServerResponse, RerankMetrics>();
export function rerankMetrics(res: ServerResponse): RerankMetrics | undefined {
  return metrics.get(res);
}
export class RerankService {
  private readonly limits: Required<RerankServiceOptions>;
  private readonly active = new Map<AbortController, Promise<void>>();
  private closing = false;
  private closePromise?: Promise<void>;
  constructor(
    private readonly deps: Dependencies,
    options: RerankServiceOptions = {},
  ) {
    this.limits = resolveRerankLimits(options);
    deps.registry.setModelLoadAdmissionCoordinator(deps.coordinator);
  }
  async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    if (req.method !== 'POST') {
      res.writeHead(405, { Allow: 'POST' });
      res.end();
      return;
    }
    if (this.closing) {
      this.error(res, new RerankError('server_shutdown', 'Rerank service is closing', 503));
      return;
    }
    if (this.active.size >= this.limits.maxConcurrentRequests) {
      this.error(res, new RerankError('queue_full', 'Too many rerank requests', 429));
      return;
    }
    const controller = new AbortController();
    const { signal } = controller;
    const started = performance.now();
    const record: RerankMetrics = { task: 'rerank', state: 'running' };
    metrics.set(res, record);
    const disconnect = () => {
      if (!res.writableFinished) controller.abort(new RerankError('cancelled', 'Client disconnected', 499));
    };
    req.on('aborted', disconnect);
    res.on('close', disconnect);
    res.on('error', disconnect);
    const timer = setTimeout(
      () => controller.abort(new RerankError('deadline_exceeded', 'Rerank deadline exceeded', 504)),
      this.limits.timeoutMs,
    );
    const operation = Promise.resolve().then(async () => {
      try {
        const { model, request } = parseRerankBody(await readBody(req, signal, this.limits.maxBodyBytes));
        if (request.documents.length > this.limits.maxDocuments)
          throw new RerankError('input_too_large', 'Too many documents', 413);
        record.model = model;
        record.documents = request.documents.length;
        const result = await this.run(model, request, signal, record);
        signal.throwIfAborted();
        record.state = 'completed';
        if (result.meta.profile === 'xlmr-cross-encoder-v1') {
          record.batches = result.meta.batches;
          record.paddingTokens = result.meta.paddingTokens;
        } else record.blocks = result.meta.blocks;
        record.promptTokens = result.usage.promptTokens;
        record.truncatedDocuments = result.meta.truncatedDocumentIndices.length;
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(rerankWireResponse(model, result)));
      } catch (error) {
        const failure = signal.aborted ? signal.reason : error;
        record.state = signal.aborted ? 'cancelled' : 'error';
        const converted =
          failure instanceof ModelLoadQueueFullError
            ? new RerankError('queue_full', failure.message, 429)
            : rerankError(failure);
        record.errorCode = converted.code;
        this.error(res, converted);
      } finally {
        record.totalMs = performance.now() - started;
        clearTimeout(timer);
        req.off('aborted', disconnect);
        res.off('close', disconnect);
        res.off('error', disconnect);
        if (!req.complete) {
          if (res.writableFinished || res.destroyed) req.destroy();
          else res.once('finish', () => req.destroy());
        }
        this.active.delete(controller);
      }
    });
    this.active.set(controller, operation);
    await operation;
  }
  private async run(
    name: string,
    request: RerankRequest,
    signal: AbortSignal,
    record: RerankMetrics,
  ): Promise<RerankResponse> {
    const { registry, coordinator, resolveModel, idleSweeper } = this.deps;
    const catalog: PublicModelEntry[] =
      this.deps.listModels?.() ?? registry.list().map((entry) => ({ ...entry, object: 'model' as const }));
    const entry = catalog.find((e) => e.id === name);
    if (entry && entry.task !== 'rerank')
      throw new RerankError('model_capability_mismatch', 'Model does not support rerank');
    if (!entry && !registry.rerank.get(name)) throw new RerankError('model_not_found', `Model ${name} not found`, 404);
    const admission = coordinator.beginRequestLoadAdmission(name);
    let lease: ReturnType<ModelRegistry['rerank']['acquire']>;
    const queuedAt = performance.now();
    idleSweeper?.beginRequest();
    try {
      signal.throwIfAborted();
      if (!registry.rerank.get(name) && resolveModel) {
        const time = performance.now();
        await coordinator.withModelLoad(() => resolveModel(name), name, signal);
        record.modelLoadMs = performance.now() - time;
      }
      signal.throwIfAborted();
      const lane = registry.rerank.lane(name);
      // The catalog check already established this model exists. A queued
      // writer may have replaced its binding before this continuation resumed.
      if (!lane) throw new RerankError('model_unavailable', 'Reranker is unavailable', 503);
      lease = registry.rerank.acquire(name, admission.transferToResident(lane));
      if (!lease) throw new RerankError('model_unavailable', 'Reranker changed while queued', 503);
      const dispatch = lease;
      return await dispatch.run(
        () =>
          coordinator.withInference(async () => {
            signal.throwIfAborted();
            if (!dispatch.current()) throw new RerankError('model_unavailable', 'Reranker changed while queued', 503);
            record.queueWaitMs = performance.now() - queuedAt - (record.modelLoadMs ?? 0);
            record.profile = dispatch.model.info.profile;
            record.quantized = dispatch.model.info.quantized;
            record.weightFormats = dispatch.model.info.weightFormats;
            const time = performance.now();
            try {
              return await dispatch.model.rerank({ ...request, signal });
            } finally {
              record.computeMs = performance.now() - time;
            }
          }, signal),
        signal,
      );
    } finally {
      lease?.release();
      admission.release();
      idleSweeper?.endRequest();
    }
  }
  close(): Promise<void> {
    this.closing = true;
    return (this.closePromise ??= (async () => {
      for (const controller of this.active.keys())
        controller.abort(new RerankError('server_shutdown', 'Rerank service is closing', 503));
      await Promise.allSettled(this.active.values());
      if (this.deps.registry.rerank.list().length)
        await this.deps.coordinator.withModelLoad(() => this.deps.registry.rerank.disposeAll(), 'rerank shutdown');
    })());
  }
  private error(res: ServerResponse, error: RerankError): void {
    if (res.destroyed || res.writableEnded) return;
    if (error.status === 429 || error.status === 503) res.setHeader('Retry-After', '1');
    res.writeHead(error.status, { 'Content-Type': 'application/json', Connection: 'close' });
    res.end(
      JSON.stringify({
        error: {
          code: error.code,
          message: error.message,
          type: error.status < 500 ? 'invalid_request_error' : 'server_error',
        },
      }),
    );
  }
}
function readBody(req: IncomingMessage, signal: AbortSignal, maxBytes: number): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let bytes = 0;
    const cleanup = () => {
      req.off('data', data);
      req.off('end', end);
      req.off('error', fail);
      signal.removeEventListener('abort', abort);
    };
    const fail = (error: unknown) => {
      cleanup();
      req.pause();
      reject(error);
    };
    const abort = () => fail(signal.reason);
    const data = (chunk: Buffer) => {
      bytes += chunk.length;
      if (bytes > maxBytes) fail(new RerankError('input_too_large', 'Request body too large', 413));
      else chunks.push(chunk);
    };
    const end = () => {
      cleanup();
      try {
        resolve(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks))));
      } catch {
        reject(new RerankError('invalid_request', 'Invalid JSON request'));
      }
    };
    req.on('data', data);
    req.on('end', end);
    req.on('error', fail);
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
  });
}
