import { createReadStream } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';

import type { Qwen3AsrResult } from '@mlx-node/core';
import {
  InvalidReferenceAudioError,
  InvalidWavError,
  NON_STREAMING_CHUNK_DURATION_MS,
  WavReader,
  WavWriter,
  changeAudioSpeed,
} from '@mlx-node/tts';
import type { TtsModel, TtsOptions } from '@mlx-node/tts';

import {
  AudioError,
  AudioTempStore,
  bad,
  event,
  ndjson,
  object,
  readJson,
  string,
  uploadWav,
  write,
} from './audio-io.js';
import { beginAudioMetrics, audioMetrics, type AudioRequestMetrics } from './audio-metrics.js';
import { AudioQueueFullError, type AudioModel } from './audio-registry.js';
import {
  speechRequest,
  textEvents,
  transcriptionRequest,
  validateVoice,
  textLength,
  type SpeechRequest,
} from './audio-validation.js';
import { AudioVoices, type AudioVoiceRecord } from './audio-voices.js';
import type { PublicModelEntry } from './handler.js';
import type { IdleSweeper } from './idle-sweeper.js';
import { ModelLoadQueueFullError, type ModelWorkCoordinator } from './model-work-coordinator.js';
import type { ModelRegistry } from './registry.js';
import { beginSSE, endSSE } from './streaming.js';

export interface AudioServiceOptions {
  maxUploadBytes?: number;
  maxTempBytes?: number;
  maxConcurrentRequests?: number;
  inputIdleMs?: number;
  outputDrainMs?: number;
  maxVoiceRecords?: number;
  maxReferenceSeconds?: number;
}
export interface AudioServiceDependencies {
  registry: ModelRegistry;
  coordinator: ModelWorkCoordinator;
  idleSweeper?: IdleSweeper | null;
  resolveModel?: (name: string) => Promise<void>;
  listModels?: () => PublicModelEntry[];
}
export const AUDIO_EXPOSE_HEADERS =
  'X-Sample-Rate, X-Channels, X-Audio-Seconds, X-First-Pcm-Ms, X-Realtime-Factor, X-Finish-Reason, X-Should-Retry, Retry-After';

/** Owns request lifetimes, never native inference details or HTTP parsing policy. */
export class AudioService {
  private readonly temp: AudioTempStore;
  private readonly voices: AudioVoices;
  private readonly active = new Map<AbortController, Promise<void>>();
  private closing = false;
  private closePromise?: Promise<void>;
  private readonly limits: Required<AudioServiceOptions>;
  constructor(
    private readonly deps: AudioServiceDependencies,
    options: AudioServiceOptions = {},
  ) {
    this.limits = {
      maxUploadBytes: 25 * 1024 * 1024,
      maxTempBytes: 512 * 1024 * 1024,
      maxConcurrentRequests: 17,
      inputIdleMs: 30_000,
      outputDrainMs: 30_000,
      maxVoiceRecords: 128,
      maxReferenceSeconds: 30,
      ...options,
    };
    for (const [name, value] of Object.entries(this.limits))
      if (!Number.isSafeInteger(value) || value < 1) throw new Error(`Invalid audio ${name}`);
    this.temp = new AudioTempStore(this.limits.maxTempBytes);
    this.voices = new AudioVoices(deps.registry.audio, this.limits.maxVoiceRecords);
    deps.registry.setModelLoadAdmissionCoordinator(deps.coordinator);
  }
  private catalog(): PublicModelEntry[] {
    return (
      this.deps.listModels?.() ?? this.deps.registry.list().map((entry) => ({ ...entry, object: 'model' as const }))
    );
  }
  async handle(req: IncomingMessage, res: ServerResponse, path: string): Promise<void> {
    if (path === '/v1/audio/voices' && req.method === 'GET') {
      const model = new URL(req.url ?? '/', 'http://localhost').searchParams.get('model');
      const catalog = this.catalog();
      if (model && (catalog.some((e) => e.id === model && e.task !== 'tts') || this.deps.registry.getAny(model))) {
        this.error(res, new AudioError(400, 'model_capability_mismatch', 'Model does not support TTS voices', 'model'));
        return;
      }
      const entries = catalog.filter((e) => e.task === 'tts' && (!model || e.id === model));
      if (model && !entries.length) {
        this.error(res, new AudioError(404, 'model_not_found', 'TTS model not found', 'model'));
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ object: 'list', data: entries.map((e) => ({ model: e.id, ...e.capabilities })) }));
      return;
    }
    const voiceId = /^\/v1\/audio\/voices\/([^/]+)$/.exec(path)?.[1];
    const voiceRequest = path === '/v1/audio/voices' || voiceId !== undefined;
    if (!voiceRequest && !['/v1/audio/speech', '/v1/audio/speech/stream', '/v1/audio/transcriptions'].includes(path)) {
      this.error(res, new AudioError(404, 'not_found', 'Audio endpoint not found'));
      return;
    }
    const methods = voiceId ? ['GET', 'DELETE'] : path === '/v1/audio/voices' ? ['GET', 'POST'] : ['POST'];
    if (!methods.includes(req.method ?? '')) {
      res.setHeader('Allow', methods.join(', '));
      this.error(res, new AudioError(405, 'method_not_allowed', `Use ${methods.join(' or ')}`));
      return;
    }
    if (this.closing || this.active.size >= this.limits.maxConcurrentRequests) {
      this.error(
        res,
        new AudioError(this.closing ? 503 : 429, 'audio_capacity_exceeded', 'Audio service is at capacity'),
      );
      return;
    }
    const controller = new AbortController();
    const metrics = beginAudioMetrics(
      res,
      voiceRequest ? 'voice' : path === '/v1/audio/transcriptions' ? 'asr' : 'tts',
    );
    const started = performance.now();
    // The logger records on 'finish'; stamp first if the flush beats the handler's exit.
    const stampTotal = () => {
      metrics.totalMs ??= performance.now() - started;
    };
    res.prependOnceListener('finish', stampTotal);
    let inferenceDispatched = false;
    // Handing work to the SDK is the conservative retry boundary, including
    // synchronous start failures and failures before the first output chunk.
    const onDispatch = () => {
      inferenceDispatched = true;
    };
    const disconnected = () => controller.abort(new Error('Audio client disconnected'));
    const requestClosed = () => {
      if (!req.complete) disconnected();
    };
    const responseClosed = () => {
      if (!res.writableFinished) disconnected();
    };
    req.once('aborted', disconnected);
    req.once('close', requestClosed);
    req.once('error', disconnected);
    res.once('close', responseClosed);
    res.once('error', disconnected);
    if (req.aborted || res.destroyed) disconnected();
    // Defer dispatch until the lifecycle has been entered into active.
    const operation = Promise.resolve().then(async () => {
      try {
        controller.signal.throwIfAborted();
        if (voiceRequest) {
          if (voiceId) await this.voiceItem(req, res, voiceId, controller.signal, onDispatch);
          else await this.createVoice(req, res, controller.signal, onDispatch);
        } else if (path === '/v1/audio/transcriptions') await this.transcribe(req, res, controller.signal, onDispatch);
        else await this.speech(req, res, controller.signal, started, path.endsWith('/stream'), onDispatch);
      } catch (error) {
        // Inference can reject with its own cancellation error. The request's
        // abort reason distinguishes host shutdown from a disconnected client.
        const failure = controller.signal.aborted ? controller.signal.reason : error;
        metrics.state = controller.signal.aborted ? 'cancelled' : 'error';
        metrics.errorCode = failure instanceof AudioError ? failure.code : 'audio_inference_failed';
        // transcribe/speech have already acknowledged native exit and cleaned
        // their resources. A live client must now receive a terminal response,
        // including when shutdown interrupted a fully uploaded request.
        if (!res.destroyed && !res.writableEnded && (!controller.signal.aborted || failure instanceof AudioError)) {
          if (controller.signal.aborted) {
            // SSE may already have advertised keep-alive before shutdown.
            // Close its transport after the terminal frame has flushed too.
            res.shouldKeepAlive = false;
            const socket = res.socket;
            res.once('finish', () => socket?.end());
          }
          this.error(res, failure, inferenceDispatched);
        }
      } finally {
        if (metrics.state === 'running') metrics.state = controller.signal.aborted ? 'cancelled' : 'completed';
        stampTotal();
        res.off('finish', stampTotal);
        req.off('aborted', disconnected);
        req.off('close', requestClosed);
        req.off('error', disconnected);
        res.off('close', responseClosed);
        res.off('error', disconnected);
        if (!req.complete) {
          // Close only after an early error response is flushed; never drain
          // an unlimited rejected upload just to keep its connection alive.
          if (res.writableFinished || res.destroyed) req.destroy();
          else res.once('finish', () => req.destroy());
        }
        this.active.delete(controller);
      }
    });
    this.active.set(controller, operation);
    await operation;
  }
  close(): Promise<void> {
    return (this.closePromise ??= (async () => {
      this.closing = true;
      for (const c of this.active.keys())
        c.abort(new AudioError(503, 'server_shutdown', 'Audio service is shutting down'));
      await Promise.allSettled(this.active.values());
      try {
        if (this.deps.registry.audio.list().length)
          await this.deps.coordinator.withModelLoad(() => this.deps.registry.audio.disposeAll(), 'audio shutdown');
      } finally {
        try {
          await this.voices.close();
        } finally {
          await this.temp.close();
        }
      }
    })());
  }
  private error(res: ServerResponse, error: unknown, inferenceDispatched = false): void {
    const metrics = audioMetrics(res);
    let e = error instanceof AudioError ? error : undefined;
    if (error instanceof AudioQueueFullError || error instanceof ModelLoadQueueFullError)
      e = new AudioError(429, 'queue_full', error.message);
    if (!e && error instanceof Error && /Unsupported Qwen3-ASR language/.test(error.message))
      e = new AudioError(400, 'unsupported_language', error.message, 'language');
    if (!e && error instanceof Error && /ASR input plus output budget/.test(error.message))
      e = new AudioError(400, 'context_length_exceeded', error.message);
    if (!e && error instanceof InvalidWavError)
      e = new AudioError(400, 'invalid_audio', error.message, metrics?.task === 'voice' ? 'audio_sample' : 'file');
    if (!e && error instanceof InvalidReferenceAudioError)
      e = new AudioError(400, 'invalid_audio', error.message, 'audio_sample');
    e ??= new AudioError(
      500,
      'audio_inference_failed',
      error instanceof Error ? error.message : 'Audio inference failed',
    );
    if (metrics) metrics.errorCode = e.code;
    const body = {
      error: {
        message: e.message,
        type: e.status === 429 ? 'rate_limit_error' : e.status < 500 ? 'invalid_request_error' : 'server_error',
        code: e.code,
        param: e.param,
      },
    };
    if (res.headersSent) {
      if (String(res.getHeader('Content-Type')).startsWith('text/event-stream')) {
        // Error frames are small; a still-blocked output is terminated, not
        // allowed to accumulate more terminal events.
        if (res.writableNeedDrain) res.destroy();
        else {
          res.write(event('error', body));
          endSSE(res);
        }
      } else res.destroy();
      return;
    }
    if (inferenceDispatched) {
      // Nonstandard hint honored by the official Node/Python SDKs. Keep the
      // real error status; this cannot prevent retries after connection loss.
      res.setHeader('X-Should-Retry', 'false');
      res.removeHeader('Retry-After');
    } else if (e.status === 429 || e.status === 503) res.setHeader('Retry-After', '1');
    res.writeHead(e.status, { 'Content-Type': 'application/json', Connection: 'close' });
    res.end(JSON.stringify(body));
  }
  private async withModel<T>(
    name: string,
    kind: AudioModel['kind'],
    signal: AbortSignal,
    metrics: AudioRequestMetrics,
    run: (model: AudioModel) => Promise<T>,
    expectedModel?: TtsModel,
  ): Promise<T> {
    const { registry, coordinator, resolveModel, idleSweeper } = this.deps;
    metrics.model = name;
    const queuedAt = performance.now();
    const entry = this.catalog().find((e) => e.id === name);
    const resident = registry.audio.get(name);
    if (expectedModel && resident?.model !== expectedModel)
      throw new AudioError(410, 'voice_unavailable', 'The voice model instance was unloaded', 'voice');
    if (entry ? entry.task !== kind : registry.getAny(name) !== undefined)
      throw new AudioError(400, 'model_capability_mismatch', `Model does not support ${kind}`, 'model');
    if (!entry && !resident) throw new AudioError(404, 'model_not_found', `Model ${name} not found`, 'model');
    const admission = coordinator.beginRequestLoadAdmission(name);
    let lease: ReturnType<ModelRegistry['audio']['acquire']>;
    idleSweeper?.beginRequest();
    try {
      signal.throwIfAborted();
      if (!registry.audio.get(name) && resolveModel) {
        await coordinator.withModelLoad(
          async () => {
            const started = performance.now();
            try {
              await resolveModel(name);
            } finally {
              metrics.modelLoadMs = performance.now() - started;
            }
          },
          name,
          signal,
        );
      }
      signal.throwIfAborted();
      const lane = registry.audio.lane(name);
      if (!lane) throw new AudioError(404, 'model_not_found', `Audio model ${name} is unavailable`, 'model');
      lease = registry.audio.acquire(name, admission.transferToResident(lane));
      const dispatch = lease!;
      return await dispatch.run(
        () =>
          coordinator.withInference(async () => {
            signal.throwIfAborted();
            if (expectedModel && (!dispatch.current() || dispatch.model.model !== expectedModel))
              throw new AudioError(410, 'voice_unavailable', 'The voice model instance was unloaded', 'voice');
            if (!dispatch.current())
              throw new AudioError(429, 'model_changed', 'Model changed while queued; retry the request');
            if (dispatch.model.kind !== kind)
              throw new AudioError(400, 'model_capability_mismatch', `Model does not support ${kind}`, 'model');
            metrics.queueWaitMs = performance.now() - queuedAt - (metrics.modelLoadMs ?? 0);
            return run(dispatch.model);
          }, signal),
        signal,
      );
    } finally {
      lease?.release();
      admission.release();
      idleSweeper?.endRequest();
    }
  }
  private async createVoice(
    req: IncomingMessage,
    res: ServerResponse,
    signal: AbortSignal,
    onDispatch: () => void,
  ): Promise<void> {
    const file = await this.temp.create();
    let reader: WavReader | undefined;
    try {
      const metrics = audioMetrics(res)!;
      const started = performance.now();
      const fields = await uploadWav(
        req,
        signal,
        file,
        this.limits.maxUploadBytes,
        this.limits.inputIdleMs,
        'audio_sample',
      );
      metrics.uploadMs = performance.now() - started;
      metrics.tempBytes = file.bytesReserved;
      for (const key of Object.keys(fields))
        if (!['model', 'name', 'transcript', 'type'].includes(key)) bad(`Unsupported voice parameter: ${key}`, key);
      const model = string(fields.model, 'model')!;
      const name = string(fields.name, 'name')!;
      const transcript = string(fields.transcript, 'transcript')!;
      if (textLength(name) > 256) bad('name exceeds 256 characters', 'name');
      if (textLength(transcript) > 4096) bad('transcript exceeds 4096 characters', 'transcript');
      if (fields.type !== undefined && fields.type !== 'audio_sample') bad('Expected type audio_sample', 'type');
      // Reject a known capability mismatch before a lazy load can evict an
      // existing model and its prepared voices. Recheck after loading below.
      if (this.catalog().find((entry) => entry.id === model)?.capabilities?.voice_cloning === false)
        throw new AudioError(
          400,
          'model_capability_mismatch',
          'Model does not support reference voice cloning',
          'model',
        );
      const wavStarted = performance.now();
      reader = await WavReader.open(file.path);
      metrics.wavReadMs = performance.now() - wavStarted;
      if (!reader.frames || reader.frames / reader.sampleRate > this.limits.maxReferenceSeconds)
        bad(`Reference audio must be nonempty and at most ${this.limits.maxReferenceSeconds} seconds`, 'audio_sample');
      const audio = reader;
      const result = await this.withModel(model, 'tts', signal, metrics, async (binding) => {
        if (binding.kind !== 'tts' || !binding.model.capabilities.voiceCloning)
          throw new AudioError(
            400,
            'model_capability_mismatch',
            'Model does not support reference voice cloning',
            'model',
          );
        const samples = await readSamples(audio, signal, metrics);
        const reservation = await this.voices.reserve(binding.model);
        try {
          signal.throwIfAborted();
          onDispatch();
          const prepareStarted = performance.now();
          const prepared = await binding.model.prepareVoice({
            audio: { samples, sampleRate: audio.sampleRate, channels: 1 },
            transcript,
            signal,
          });
          metrics.computeMs = performance.now() - prepareStarted;
          let record: AudioVoiceRecord | undefined;
          try {
            signal.throwIfAborted();
            record = await this.voices.add(name, model, binding.model, prepared, reservation);
            // Return the upload quota before exposing the new handle. Failure or
            // cancellation rolls back preparation under the same execution lease.
            await audio.close();
            reader = undefined;
            await file.dispose();
            signal.throwIfAborted();
            const description = await this.voices.describe(record);
            signal.throwIfAborted();
            return description;
          } catch (error) {
            if (record) await this.voices.release(record);
            else await prepared.dispose();
            throw error;
          }
        } finally {
          this.voices.unreserve(reservation);
        }
      });
      res.writeHead(200, { 'Content-Type': 'application/json', Location: `/v1/audio/voices/${result.id}` });
      res.end(JSON.stringify(result));
    } finally {
      try {
        await reader?.close();
      } finally {
        await file.dispose();
      }
    }
  }
  private async voiceItem(
    req: IncomingMessage,
    res: ServerResponse,
    id: string,
    signal: AbortSignal,
    onDispatch: () => void,
  ): Promise<void> {
    const record = await this.voices.record(id);
    const run = async () => {
      if (req.method === 'DELETE') {
        onDispatch();
        await this.voices.release(record);
        return { id, object: 'audio.voice.deleted', deleted: true };
      }
      return this.voices.describe(record);
    };
    let result;
    for (;;) {
      signal.throwIfAborted();
      const runtime = this.voices.binding(record);
      const name = this.voices.modelName(record);
      if (!runtime || !name) {
        result = await run();
        break;
      }
      try {
        result = await this.withModel(name, 'tts', signal, audioMetrics(res)!, run, runtime.model);
        break;
      } catch (error) {
        // An unload revokes bindings synchronously. Query/release never reload
        // a checkpoint solely to inspect a stale ephemeral voice. If only the
        // selected alias vanished, acquire the surviving alias's lease first.
        if (!(error instanceof AudioError) || error.code !== 'voice_unavailable') throw error;
      }
    }
    signal.throwIfAborted();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(result));
  }
  private async transcribe(
    req: IncomingMessage,
    res: ServerResponse,
    signal: AbortSignal,
    onDispatch: () => void,
  ): Promise<void> {
    const file = await this.temp.create();
    let reader: WavReader | undefined;
    let finishResponse: (() => void | Promise<void>) | undefined;
    try {
      const metrics = audioMetrics(res)!;
      const uploadStarted = performance.now();
      const fields = await uploadWav(req, signal, file, this.limits.maxUploadBytes, this.limits.inputIdleMs);
      metrics.uploadMs = performance.now() - uploadStarted;
      metrics.tempBytes = file.bytesReserved;
      const request = transcriptionRequest(fields);
      const wavStarted = performance.now();
      reader = await WavReader.open(file.path);
      metrics.wavReadMs = performance.now() - wavStarted;
      if (!reader.frames) bad('WAV contains no audio', 'file');
      // A tiny source rate must not expand a small upload into unbounded
      // resampled audio or feature tensors.
      if (reader.frames / reader.sampleRate > 1200) bad('Audio exceeds the 20 minute duration limit', 'file');
      request.options.sampleRate = reader.sampleRate;
      const audio = reader;
      await this.withModel(request.model, 'asr', signal, metrics, async (binding) => {
        if (binding.kind !== 'asr') throw new Error('Invalid ASR binding');
        if (!request.stream) {
          const samples = await readSamples(audio, signal, metrics);
          onDispatch();
          const task = binding.model.startTranscription(samples, request.options);
          const abort = () => task.cancel();
          signal.addEventListener('abort', abort, { once: true });
          try {
            if (signal.aborted) task.cancel();
            const result = await task.result();
            addAsrStages(metrics, result);
            Object.assign(metrics, {
              audioSeconds: result.audioSeconds,
              finishReason: result.reachedMaxTokens ? 'length' : 'stop',
            });
            signal.throwIfAborted();
            finishResponse = () => {
              signal.throwIfAborted();
              res.setHeader('X-Finish-Reason', result.reachedMaxTokens ? 'length' : 'stop');
              res.writeHead(200, {
                'Content-Type': request.format === 'text' ? 'text/plain; charset=utf-8' : 'application/json',
              });
              res.end(
                request.format === 'text'
                  ? result.text
                  : JSON.stringify({
                      text: result.text,
                      language: result.language?.toLowerCase(),
                      duration: result.audioSeconds,
                      finish_reason: result.reachedMaxTokens ? 'length' : 'stop',
                    }),
              );
            };
          } finally {
            signal.removeEventListener('abort', abort);
          }
        } else {
          onDispatch();
          const stream = await binding.model.createStream(request.options);
          let budgetExhausted = false;
          const abort = () => stream.cancel();
          signal.addEventListener('abort', abort, { once: true });
          const emit = async (result: Qwen3AsrResult) => {
            budgetExhausted ||= result.reachedMaxTokens;
            // A token frontier can split a UTF-8 character. Decoding that
            // prefix alone may contain a replacement character; preserve the
            // canonical full hypothesis instead of duplicating it on screen.
            const stable = result.isFinal
              ? result.text
              : result.text.startsWith(result.stableText)
                ? result.stableText
                : '';
            const provisional = result.text.slice(stable.length);
            addAsrStages(metrics, result);
            metrics.audioSeconds = result.audioSeconds;
            if (result.isFinal) metrics.finishReason = budgetExhausted ? 'length' : 'stop';
            signal.throwIfAborted();
            if (!res.headersSent) {
              res.setHeader('X-Accel-Buffering', 'no');
              beginSSE(res);
            }
            await write(
              res,
              event('transcript', {
                stable,
                provisional,
                revision: result.revision,
                is_final: result.isFinal,
                language: result.language?.toLowerCase(),
                duration: result.audioSeconds,
                reached_max_tokens: result.isFinal ? budgetExhausted : result.reachedMaxTokens,
                finish_reason: result.isFinal ? (budgetExhausted ? 'length' : 'stop') : null,
              }),
              signal,
              this.limits.outputDrainMs,
            );
          };
          try {
            if (signal.aborted) stream.cancel();
            const frames = Math.max(1, Math.round((request.options.chunkSeconds ?? 2) * audio.sampleRate));
            for await (const chunk of audio.chunks(frames, true)) {
              signal.throwIfAborted();
              const result = await stream.feed(chunk);
              if (result) await emit(result);
            }
            const final = await stream.finish();
            finishResponse = async () => {
              await emit(final);
              endSSE(res);
            };
          } finally {
            signal.removeEventListener('abort', abort);
            await stream.dispose();
          }
        }
      });
    } finally {
      try {
        await reader?.close();
      } finally {
        await file.dispose();
      }
    }
    // The terminal response is an ownership boundary: a client starting its
    // next upload immediately must not race the previous file's quota refund.
    await finishResponse?.();
  }
  private async speech(
    req: IncomingMessage,
    res: ServerResponse,
    signal: AbortSignal,
    started: number,
    inputStream: boolean,
    onDispatch: () => void,
  ): Promise<void> {
    const type = req.headers['content-type'] ?? '';
    if (!new RegExp(inputStream ? '^application/x-ndjson(?:;|$)' : '^application/json(?:;|$)', 'i').test(type))
      throw new AudioError(
        415,
        'unsupported_media_type',
        inputStream ? 'Expected application/x-ndjson' : 'Expected application/json',
      );
    const inputAbort = new AbortController();
    const inputSignal = AbortSignal.any([signal, inputAbort.signal]);
    let records: AsyncGenerator<unknown> | undefined;
    let body: Record<string, unknown>;
    if (inputStream) {
      records = ndjson(req, inputSignal, this.limits.inputIdleMs);
      const first = await records.next();
      body = object(first.value);
      if (body.type !== 'config') bad('First NDJSON record must be config');
    } else body = await readJson(req, signal, this.limits.inputIdleMs);
    const request = speechRequest(body, inputStream);
    if (
      !this.catalog().some((entry) => entry.id === request.model) &&
      !this.deps.registry.audio.get(request.model) &&
      !this.deps.registry.getAny(request.model)
    )
      throw new AudioError(404, 'model_not_found', `Model ${request.model} not found`, 'model');
    const selectedVoice = request.options.voice;
    const voiceRecord =
      typeof selectedVoice !== 'string' && 'id' in selectedVoice
        ? await this.voices.record(selectedVoice.id)
        : undefined;
    const voiceBinding = voiceRecord ? this.voices.binding(voiceRecord) : undefined;
    if (voiceRecord && !voiceBinding)
      throw new AudioError(
        410,
        'voice_unavailable',
        'Voice was released or evicted; upload the reference again',
        'voice',
      );
    if (voiceBinding && this.deps.registry.audio.get(request.model)?.model !== voiceBinding.model)
      throw new AudioError(400, 'voice_model_mismatch', 'Voice belongs to a different model instance', 'voice');
    const file = request.format === 'wav' ? await this.temp.create() : undefined;
    try {
      const metrics = audioMetrics(res)!;
      await this.withModel(
        request.model,
        'tts',
        signal,
        metrics,
        async (binding) => {
          if (binding.kind !== 'tts') throw new Error('Invalid TTS binding');
          const voice =
            typeof selectedVoice !== 'string' && 'id' in selectedVoice
              ? await this.voices.resolve(voiceRecord!, binding.model)
              : selectedVoice;
          const options: TtsOptions = { ...request.options, voice };
          validateVoice(binding.model, options);
          const input = records ? textEvents(records, binding.model, options) : request.input!;
          onDispatch();
          const chunkDurationMs = options.chunkDurationMs ?? (file ? NON_STREAMING_CHUNK_DURATION_MS : undefined);
          const suppressBootstrapSilence = file ? false : options.suppressBootstrapSilence;
          const source = binding.model.synthesizeStream(input, {
            ...options,
            chunkDurationMs,
            suppressBootstrapSilence,
            signal,
          });
          void source.completed.catch(() => {});
          const processed = changeAudioSpeed(source, request.speed);
          let writer: WavWriter | undefined;
          let frames = 0;
          let firstPcmMs: number | undefined;
          const caps = binding.model.capabilities;
          try {
            const { audioEncodePcm16 } = await import('@mlx-node/core');
            for await (const chunk of processed) {
              signal.throwIfAborted();
              if (!chunk.samples.length) continue;
              firstPcmMs ??= performance.now() - started;
              metrics.firstPcmMs = firstPcmMs;
              frames += chunk.samples.length / caps.channels;
              if (file) {
                if (!writer) {
                  file.reserve(44);
                  writer = await WavWriter.open(file.path, caps.sampleRate, caps.channels);
                }
                file.reserve(chunk.samples.length * 2);
                metrics.tempBytes = file.bytesReserved;
                await writer.write(chunk);
              } else {
                if (!res.headersSent) this.beginSpeech(res, request, caps, firstPcmMs);
                const pcm = audioEncodePcm16(chunk.samples);
                await write(
                  res,
                  request.sse
                    ? event('speech.audio.delta', {
                        audio: pcm.toString('base64'),
                        response_format: 'pcm',
                        sample_rate: caps.sampleRate,
                        channels: caps.channels,
                      })
                    : pcm,
                  signal,
                  this.limits.outputDrainMs,
                );
              }
            }
            const stats = await source.completed;
            signal.throwIfAborted();
            if (firstPcmMs === undefined) throw new AudioError(500, 'empty_audio', 'Model generated no audio');
            const seconds = frames / caps.sampleRate;
            const rtf = (stats.synthesisMs + processed.stats.processingMs) / 1000 / seconds;
            Object.assign(metrics, {
              computeMs: stats.synthesisMs + processed.stats.processingMs,
              firstPcmMs,
              pcmFrames: frames,
              audioSeconds: seconds,
              realtimeFactor: rtf,
              finishReason: stats.finishReason,
              inputWaitMs: stats.inputWaitMs,
              consumerWaitMs: stats.consumerWaitMs,
            });
            const final = {
              audio_seconds: seconds,
              first_pcm_ms: firstPcmMs,
              realtime_factor: rtf,
              finish_reason: stats.finishReason,
              input_wait_ms: stats.inputWaitMs,
              consumer_wait_ms: stats.consumerWaitMs,
            };
            if (file) {
              await writer!.close();
              writer = undefined;
              this.speechHeaders(res, caps, firstPcmMs);
              res.setHeader('X-Audio-Seconds', seconds);
              res.setHeader('X-Realtime-Factor', rtf);
              res.setHeader('X-Finish-Reason', stats.finishReason);
            } else if (request.sse) {
              await write(res, event('speech.audio.done', { ...final }), signal, this.limits.outputDrainMs);
              endSSE(res);
            } else {
              res.addTrailers({
                'X-Audio-Seconds': String(seconds),
                'X-Realtime-Factor': String(rtf),
                'X-Finish-Reason': stats.finishReason,
              });
              res.end();
            }
          } finally {
            inputAbort.abort(new Error('Speech input closed'));
            source.cancel();
            try {
              await source.completed.catch(() => {});
            } finally {
              await writer?.close();
            }
          }
        },
        voiceBinding?.model,
      );
      if (file) {
        // File delivery is outside the inference lease.
        signal.throwIfAborted();
        res.writeHead(200, { 'Content-Type': 'audio/wav', 'Content-Length': (await file.file.stat()).size });
        // A Content-Length response completes at its last byte, before end().
        // Hold one read block until the spool's quota has been returned so a
        // client's immediate next request cannot race asynchronous cleanup.
        let tail: Buffer | undefined;
        for await (const bytes of createReadStream(file.path)) {
          if (tail) await write(res, tail, signal, this.limits.outputDrainMs);
          tail = bytes as Buffer;
        }
        await file.dispose();
        if (tail) await write(res, tail, signal, this.limits.outputDrainMs);
        res.end();
      }
    } finally {
      inputAbort.abort(new Error('Speech input closed'));
      try {
        await records?.return(undefined);
      } finally {
        await file?.dispose();
      }
    }
  }
  private speechHeaders(res: ServerResponse, caps: TtsModel['capabilities'], firstPcmMs: number): void {
    res.setHeader('X-Sample-Rate', caps.sampleRate);
    res.setHeader('X-Channels', caps.channels);
    res.setHeader('X-First-Pcm-Ms', firstPcmMs);
  }
  private beginSpeech(
    res: ServerResponse,
    request: SpeechRequest,
    caps: TtsModel['capabilities'],
    firstPcmMs: number,
  ): void {
    this.speechHeaders(res, caps, firstPcmMs);
    res.setHeader('X-Accel-Buffering', 'no');
    if (request.sse) beginSSE(res);
    else
      res.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        Trailer: 'X-Audio-Seconds, X-Realtime-Factor, X-Finish-Reason',
      });
  }
}

/** Decodes the whole WAV as mono and adds the time to `wavReadMs`. */
async function readSamples(audio: WavReader, signal: AbortSignal, metrics: AudioRequestMetrics): Promise<Float32Array> {
  const started = performance.now();
  const samples = new Float32Array(audio.frames);
  let offset = 0;
  for await (const chunk of audio.chunks(16384, true)) {
    signal.throwIfAborted();
    samples.set(chunk, offset);
    offset += chunk.length;
  }
  signal.throwIfAborted();
  metrics.wavReadMs = (metrics.wavReadMs ?? 0) + performance.now() - started;
  return samples;
}
/** Stage times add up over streaming revisions; a stream's tokenIds is its whole transcript so far. */
function addAsrStages(metrics: AudioRequestMetrics, result: Qwen3AsrResult): void {
  metrics.computeMs = (metrics.computeMs ?? 0) + result.totalMs;
  metrics.featureMs = (metrics.featureMs ?? 0) + result.featureMs;
  metrics.prefillMs = (metrics.prefillMs ?? 0) + result.prefillMs;
  metrics.decodeMs = (metrics.decodeMs ?? 0) + result.decodeMs;
  metrics.outputTokens = result.tokenIds.length;
}
