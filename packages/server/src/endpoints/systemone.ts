import { randomUUID } from 'node:crypto';
import type { IncomingMessage, ServerResponse } from 'node:http';

import { getMemorySnapshot } from '@mlx-node/core';
import { validateSystemOneClairicEnvelope } from '@mlx-node/lm/clairic-types';
import { validateClefRequest } from '@mlx-node/lm/clef-types';

import { DecisionQueueFullError } from '../decision-registry.js';
import { sendBadRequest, sendInternalError, sendNotFound, sendRateLimit } from '../errors.js';
import type { IdleSweeper } from '../idle-sweeper.js';
import { MetricsTrace } from '../metrics-trace.js';
import { ModelLoadQueueFullError, type ModelWorkCoordinator } from '../model-work-coordinator.js';
import type { ModelRegistry } from '../registry.js';
import { stringifyStrictJson } from '../strict-json.js';
import { recordExternalTurn } from '../turn-trace.js';

function safeMemorySnapshot(): { activeBytes: number; peakBytes: number } | undefined {
  try {
    const snap = getMemorySnapshot();
    return { activeBytes: snap.activeBytes, peakBytes: snap.peakBytes };
  } catch {
    return undefined;
  }
}

function countMediaInputs(record: Record<string, unknown>): number {
  const images = Array.isArray(record.images) ? record.images.length : 0;
  const documents = Array.isArray(record.documents) ? record.documents.length : 0;
  return images + documents;
}

export async function handleSystemOne(
  req: IncomingMessage,
  res: ServerResponse,
  raw: string,
  registry: ModelRegistry,
  idleSweeper?: IdleSweeper | null,
  resolveModel?: (name: string) => Promise<void>,
  coordinator?: ModelWorkCoordinator,
  /** Journals the turn once it holds the decision lane; returns its end. */
  beginTurn?: () => () => void,
  /** Injectable metrics trace sink for unit tests. */
  traceSink?: Pick<MetricsTrace, 'record'>,
): Promise<void> {
  // Bind BEFORE admission so a registerDecision inside resolveModel
  // synchronously transfers every outstanding cold permit, matching the
  // Messages/Responses handlers.
  if (coordinator) {
    registry.setModelLoadAdmissionCoordinator(coordinator);
  }
  const requestId = randomUUID();
  const requestStartMs = Date.now();
  res.setHeader('x-typesafe-request-id', requestId);
  let modelName: string;
  let mode: 'decide' | 'extract';
  let imageCount = 0;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('Request body must be a JSON object');
    }
    const record = parsed as Record<string, unknown>;
    if (typeof record.model !== 'string' || !record.model.trim()) {
      throw new Error('model is required');
    }
    const hasQuestions = Object.hasOwn(record, 'questions');
    const hasFields = Object.hasOwn(record, 'fields');
    if (hasQuestions && hasFields) {
      throw new Error('Request cannot specify both questions and fields');
    }
    if (!hasQuestions && !hasFields) {
      throw new Error('Request must specify either questions or fields');
    }
    if (hasQuestions) {
      validateClefRequest(parsed);
      mode = 'decide';
    } else {
      validateSystemOneClairicEnvelope(parsed);
      mode = 'extract';
      imageCount = countMediaInputs(record);
    }
    modelName = record.model;
  } catch (error) {
    sendBadRequest(res, (error as Error).message);
    return;
  }
  const abort = new AbortController();
  const onClose = () => abort.abort();
  res.once('close', onClose);
  req.once('aborted', onClose);
  if (req.aborted || res.destroyed) abort.abort();
  let loadAdmission: ReturnType<ModelWorkCoordinator['beginRequestLoadAdmission']> | undefined;
  let lease: ReturnType<ModelRegistry['decisions']['acquire']>;
  let resolveMs: number | undefined;
  idleSweeper?.beginRequest();
  try {
    abort.signal.throwIfAborted();
    // A resident chat model can never serve System One: skip the load queue
    // entirely and let the 400 below answer without taking a writer slot.
    if (!registry.decisions.get(modelName) && !registry.get(modelName) && resolveModel) {
      loadAdmission = coordinator?.beginRequestLoadAdmission(modelName);
      const resolveStart = Date.now();
      const resolve = () => resolveModel(modelName);
      if (coordinator) await coordinator.withModelLoad(resolve, modelName, abort.signal);
      else await resolve();
      resolveMs = Math.max(0, Date.now() - resolveStart);
    }
    abort.signal.throwIfAborted();
    const lane = registry.decisions.lane(modelName);
    const permit = lane ? loadAdmission?.transferToResident(lane) : undefined;
    lease = registry.decisions.acquire(modelName, permit);
    if (!lease) {
      if (registry.get(modelName)) {
        sendBadRequest(
          res,
          mode === 'extract'
            ? `Model "${modelName}" does not support extraction`
            : `Model "${modelName}" does not support decisions`,
        );
      } else {
        sendNotFound(res, `Model "${modelName}" not found`);
      }
      return;
    }
    const dispatch = lease;
    if (mode === 'decide' && !('decideRaw' in dispatch.model)) {
      sendBadRequest(res, `Model "${modelName}" does not support decisions`);
      return;
    }
    if (mode === 'extract' && !('extractRaw' in dispatch.model)) {
      sendBadRequest(res, `Model "${modelName}" does not support extraction`);
      return;
    }
    const run = () =>
      dispatch.run(async () => {
        if (!dispatch.current()) throw new Error('Model binding changed while waiting; retry the request');
        const endTurn = beginTurn?.();
        const dispatchStartMs = Date.now();
        const preDispatchMs = Math.max(0, dispatchStartMs - requestStartMs);
        let traceEmitted = false;
        const emitTrace = (
          outcome: 'ok' | 'error' | 'cancelled',
          usage?: { input_tokens?: number; output_tokens?: number },
        ): void => {
          if (traceEmitted) return;
          traceEmitted = true;
          try {
            const sink = traceSink ?? new MetricsTrace();
            const mem = safeMemorySnapshot();
            const promptTokens =
              typeof usage?.input_tokens === 'number' && Number.isFinite(usage.input_tokens)
                ? Math.max(0, Math.floor(usage.input_tokens))
                : undefined;
            const outputTokens =
              typeof usage?.output_tokens === 'number' && Number.isFinite(usage.output_tokens)
                ? Math.max(0, Math.floor(usage.output_tokens))
                : undefined;
            sink.record({
              traceId: requestId,
              ts: Date.now(),
              model: modelName,
              modelId: dispatch.canonicalName,
              modelType: dispatch.identity?.modelType ?? (mode === 'extract' ? 'clairic' : 'clef'),
              outcome,
              ...(outcome === 'error' ? { errorKind: 'exception' as const } : {}),
              endpoint: 'systemone',
              stream: false,
              dispatch: 'start',
              finishReason: outcome === 'ok' ? 'stop' : outcome === 'cancelled' ? 'cancelled' : 'error',
              durationMs: Math.max(0, Date.now() - dispatchStartMs),
              preDispatchMs,
              ...(resolveMs !== undefined ? { resolveMs } : {}),
              ...(promptTokens !== undefined ? { promptTokens } : {}),
              ...(outputTokens !== undefined ? { outputTokens } : {}),
              ...(dispatch.identity?.contextWindow ? { ctxWindow: dispatch.identity.contextWindow } : {}),
              toolCount: 0,
              imageCount,
              concurrency: 1,
              concurrencyPeak: 1,
              ...(mem !== undefined ? { memActiveBytes: mem.activeBytes, memPeakBytes: mem.peakBytes } : {}),
            });
            recordExternalTurn(outcome, outputTokens ?? 0);
          } catch {
            // Telemetry is best-effort and must never fail a System One request.
          }
        };
        try {
          const result =
            mode === 'extract'
              ? await (dispatch.model as Extract<typeof dispatch.model, { extractRaw: unknown }>).extractRaw(raw, {
                  signal: abort.signal,
                })
              : await (dispatch.model as Extract<typeof dispatch.model, { decideRaw: unknown }>).decideRaw(raw, {
                  signal: abort.signal,
                });
          if (abort.signal.aborted) {
            emitTrace('cancelled', result.usage);
            abort.signal.throwIfAborted();
          }
          emitTrace('ok', result.usage);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(stringifyStrictJson({ model: dispatch.canonicalName, ...result }));
        } catch (error) {
          emitTrace(abort.signal.aborted ? 'cancelled' : 'error');
          throw error;
        } finally {
          endTurn?.();
        }
      }, abort.signal);
    if (coordinator) await coordinator.withInference(run, abort.signal);
    else await run();
  } catch (error) {
    if (abort.signal.aborted || res.destroyed) return;
    if (
      error instanceof DecisionQueueFullError ||
      error instanceof ModelLoadQueueFullError ||
      (error instanceof Error && error.message.includes('binding changed while waiting'))
    )
      sendRateLimit(res, error.message);
    else if (
      error instanceof Error &&
      (error.message.includes('Invalid CLEF request:') || error.message.includes('Invalid Clairic request:'))
    )
      sendBadRequest(res, error.message);
    else sendInternalError(res, error instanceof Error ? error.message : 'System One inference failed');
  } finally {
    lease?.release();
    loadAdmission?.release();
    idleSweeper?.endRequest();
    res.removeListener('close', onClose);
    req.removeListener('aborted', onClose);
  }
}
