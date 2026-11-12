import { request as httpRequest } from 'node:http';
import type { AddressInfo } from 'node:net';

import { audioEncodePcm16, audioWavHeader } from '@mlx-node/core';
import type { Qwen3AsrResult } from '@mlx-node/core';
import { InvalidWavError, WavReader } from '@mlx-node/tts';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { AudioQueueFullError, AudioRegistry, type AsrModel } from '../../packages/server/src/audio-registry.js';
import type { AudioServiceOptions } from '../../packages/server/src/audio-service.js';
import { queueRejections } from '../../packages/server/src/queue-rejections.js';
import { createServer, type ServerInstance } from '../../packages/server/src/server.js';
import { createTtsModel, type TtsBackend } from '../../packages/tts/src/model.js';

const servers: ServerInstance[] = [];
afterEach(async () => {
  for (const s of servers.splice(0)) await s.close({ timeoutMs: 100 });
});
async function start(audio?: AudioServiceOptions, authToken?: string) {
  const server = await createServer({
    port: 0,
    host: '127.0.0.1',
    disableStore: true,
    idleClearCacheMs: 0,
    audio,
    authToken,
  });
  servers.push(server);
  return { server, url: `http://127.0.0.1:${(server.server.address() as AddressInfo).port}` };
}
function tts(rate = 24000) {
  const submitted: string[] = [];
  const backend: TtsBackend = {
    capabilities: {
      family: 'test',
      variant: 'custom_voice',
      sampleRate: rate,
      channels: 1,
      voices: ['voice'],
      languages: ['english'],
      voiceCloning: false,
      conditioning: [{ voice: 'preset', instruct: 'supported' }],
      textStreaming: 'segmented',
      audioStreaming: true,
    },
    prepareVoice: vi.fn(),
    releaseVoice: vi.fn(),
    dispose: vi.fn(async () => {}),
    start(text) {
      submitted.push(text);
      let at = 0;
      return {
        next: async () =>
          at++ === 0
            ? { samples: new Float32Array([-0.5, 0, 0.5]), finished: false }
            : { samples: new Float32Array(), finished: true, finishReason: 'eos', synthesisMs: 1 },
        cancel: vi.fn(),
        waitFinished: async () => {},
      };
    },
  };
  return { model: createTtsModel(backend), submitted, backend };
}
function result(text: string, revision = 0, isFinal = true): Qwen3AsrResult {
  return {
    text,
    stableText: isFinal ? text : text.slice(0, 1),
    provisionalText: isFinal ? '' : text.slice(1),
    revision,
    isFinal,
    tokenIds: [],
    reachedMaxTokens: false,
    audioSeconds: 1,
    segmentAudioSeconds: 1,
    featureMs: 0,
    prefillMs: 0,
    decodeMs: 1,
    totalMs: 1,
    tokensPerSecond: 1,
    realTimeFactor: 0.001,
  };
}
function asr() {
  const feeds: Float32Array[] = [];
  const options: unknown[] = [];
  const dispose = vi.fn(async () => {});
  const model: AsrModel = {
    dispose,
    startTranscription: vi.fn((samples, opts) => {
      feeds.push(samples);
      options.push(opts);
      return { cancel: vi.fn(), result: async () => result('whole') };
    }),
    createStream: vi.fn(async (opts) => {
      options.push(opts);
      let revision = 0;
      return {
        feed: async (chunk: Float32Array) => {
          feeds.push(chunk);
          return result(++revision === 1 ? 'old' : 'revised', revision, false);
        },
        finish: async () => result('final', ++revision),
        cancel: vi.fn(),
        dispose: vi.fn(async () => {}),
        startCapture: vi.fn(),
      };
    }),
  };
  return { model, feeds, options, dispose };
}
function wav(rate = 48000, seconds = 0.01, channels = 1): Buffer<ArrayBuffer> {
  const samples = new Float32Array(Math.round(rate * seconds) * channels).fill(0.25);
  const pcm = audioEncodePcm16(samples);
  return Buffer.concat([audioWavHeader(rate, channels, pcm.length), pcm]);
}
async function speech(url: string, extra: Record<string, unknown> = {}) {
  return fetch(`${url}/v1/audio/speech`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'tts', voice: 'voice', input: 'Hello.', ...extra }),
  });
}
function binaryRequest(url: string, body: string, type = 'application/json') {
  return new Promise<{ bytes: Buffer; headers: Record<string, unknown>; trailers: Record<string, unknown> }>(
    (resolve, reject) => {
      const req = httpRequest(url, { method: 'POST', headers: { 'Content-Type': type } }, (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (c) => chunks.push(c));
        res.on('error', reject);
        res.on('end', () => resolve({ bytes: Buffer.concat(chunks), headers: res.headers, trailers: res.trailers }));
      });
      req.on('error', reject);
      req.end(body);
    },
  );
}

describe('audio HTTP contract', () => {
  it('keeps authentication and rejects unsupported formats before inference', async () => {
    const { server, url } = await start();
    const fake = tts();
    server.registry.audio.register('tts', { kind: 'tts', model: fake.model });
    for (const format of ['mp3', 'opus', 'aac', 'flac']) {
      const res = await speech(url, { response_format: format });
      expect(res.status).toBe(400);
      expect((await res.json()).error.code).toBe('unsupported_response_format');
    }
    expect(fake.submitted).toEqual([]);
    const secure = await start(undefined, 'secret');
    expect((await fetch(`${secure.url}/health`)).status).toBe(200);
    expect((await fetch(`${secure.url}/v1/audio/voices`)).status).toBe(401);
    expect((await fetch(`${secure.url}/v1/audio/voices`, { headers: { 'x-api-key': 'secret' } })).status).toBe(200);
  });
  it('lists native capabilities and returns correct complete WAV and PCM trailers', async () => {
    const { server, url } = await start();
    const fake = tts(44100);
    server.registry.audio.register('tts', { kind: 'tts', model: fake.model });
    const list = await (await fetch(`${url}/v1/models`)).json();
    expect(list.data[0]).toMatchObject({ modality: 'audio', task: 'tts', capabilities: { sample_rate: 44100 } });
    const voices = await (await fetch(`${url}/v1/audio/voices?model=tts`)).json();
    expect(voices.data[0]).toMatchObject({ voices: ['voice'], sample_rate: 44100 });
    const res = await speech(url);
    expect(res.status).toBe(200);
    expect(res.headers.get('x-sample-rate')).toBe('44100');
    const bytes = Buffer.from(await res.arrayBuffer());
    expect(bytes.toString('ascii', 0, 4)).toBe('RIFF');
    expect(bytes.readUInt32LE(4)).toBe(bytes.length - 8);
    expect(Number(res.headers.get('x-audio-seconds'))).toBeCloseTo(3 / 44100);
    const pcm = await binaryRequest(
      `${url}/v1/audio/speech`,
      JSON.stringify({ model: 'tts', voice: 'voice', input: 'Hello.', response_format: 'pcm' }),
    );
    expect(pcm.bytes).toEqual(bytes.subarray(44));
    expect(pcm.trailers['x-finish-reason']).toBe('eos');
    expect(Number(pcm.trailers['x-audio-seconds'])).toBeCloseTo(3 / 44100);
  });
  it('returns PCM SSE with metadata and one completion event', async () => {
    const { server, url } = await start();
    server.registry.audio.register('tts', { kind: 'tts', model: tts().model });
    const res = await speech(url, { response_format: 'pcm', stream_format: 'sse' });
    const body = await res.text();
    const events = body
      .split('\n')
      .filter((l) => l.startsWith('data: '))
      .map((l) => JSON.parse(l.slice(6)));
    expect(events.map((e) => e.type)).toEqual(['speech.audio.delta', 'speech.audio.done']);
    expect(events[0]).toMatchObject({ sample_rate: 24000, channels: 1, response_format: 'pcm' });
    expect(Buffer.from(events[0].audio, 'base64').length).toBe(6);
    expect(events[1].audio_seconds).toBeCloseTo(3 / 24000);
  });
  it('synthesizes WAV with whole-utterance codec chunks and keeps streaming chunks for PCM', async () => {
    const { server, url } = await start();
    const fake = tts();
    const backendStart = vi.spyOn(fake.backend, 'start');
    server.registry.audio.register('tts', { kind: 'tts', model: fake.model });
    for (const extra of [{}, { response_format: 'pcm' }, { stream: { chunk_duration_ms: 200 } }]) {
      const res = await speech(url, extra);
      expect(res.status).toBe(200);
      await res.arrayBuffer();
    }
    expect(backendStart.mock.calls.map(([, options]) => options.chunkDurationMs)).toEqual([640, 160, 200]);
  });
  it('offers the bootstrap-silence drop to PCM and SSE but never to WAV, with a per-request opt-out', async () => {
    const { server, url } = await start();
    const fake = tts();
    const backendStart = vi.spyOn(fake.backend, 'start');
    server.registry.audio.register('tts', { kind: 'tts', model: fake.model });
    for (const extra of [
      {},
      { suppress_bootstrap_silence: true },
      { response_format: 'pcm' },
      { response_format: 'pcm', stream_format: 'sse' },
      { response_format: 'pcm', suppress_bootstrap_silence: false },
      { response_format: 'pcm', suppress_bootstrap_silence: true },
    ]) {
      const res = await speech(url, extra);
      expect(res.status).toBe(200);
      await res.arrayBuffer();
    }
    expect(backendStart.mock.calls.map(([, options]) => options.suppressBootstrapSilence)).toEqual([
      false,
      false,
      true,
      true,
      false,
      true,
    ]);
    for (const value of ['false', 0, null, {}]) {
      const res = await speech(url, { response_format: 'pcm', suppress_bootstrap_silence: value });
      expect(res.status).toBe(400);
      expect((await res.json()).error).toMatchObject({ param: 'suppress_bootstrap_silence' });
    }
    expect(backendStart).toHaveBeenCalledTimes(6);
  });
  it('applies the NDJSON config opt-out to every segment', async () => {
    const { server, url } = await start();
    const fake = tts();
    const backendStart = vi.spyOn(fake.backend, 'start');
    server.registry.audio.register('tts', { kind: 'tts', model: fake.model });
    for (const config of [{}, { suppress_bootstrap_silence: false }]) {
      const records = [
        { type: 'config', model: 'tts', voice: 'voice', ...config },
        { type: 'text', text: '第一句。' },
        { type: 'text', text: '第二句。' },
      ];
      const res = await binaryRequest(
        `${url}/v1/audio/speech/stream`,
        records.map((record) => JSON.stringify(record) + '\n').join(''),
        'application/x-ndjson',
      );
      expect(res.trailers['x-finish-reason']).toBe('eos');
    }
    expect(backendStart.mock.calls.map(([text, options]) => [text, options.suppressBootstrapSilence])).toEqual([
      ['第一句。', true],
      ['第二句。', true],
      ['第一句。', false],
      ['第二句。', false],
    ]);
  });
  it.each([16000, 22050, 24000, 44100, 48000, 96000])(
    'passes %i Hz source metadata and downmixes without server resampling',
    async (rate) => {
      const { server, url } = await start();
      const fake = asr();
      server.registry.audio.register('asr', { kind: 'asr', model: fake.model });
      const bytes = wav(rate, 0.01, 2);
      const res = await fetch(`${url}/v1/audio/transcriptions?model=asr`, {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body: bytes,
      });
      expect(res.status).toBe(200);
      expect(await res.json()).toMatchObject({ text: 'whole', finish_reason: 'stop' });
      expect(fake.options[0]).toMatchObject({ sampleRate: rate });
      expect(fake.options[0]).not.toHaveProperty('maxTokens');
      expect(fake.feeds[0].length).toBe(Math.round(rate * 0.01));
      expect(fake.feeds[0][0]).toBe(0.25);
    },
  );
  it('leaves the one-shot output budget to the model unless sampling.max_tokens sets it', async () => {
    const { server, url } = await start();
    const fake = asr();
    server.registry.audio.register('asr', { kind: 'asr', model: fake.model });
    const post = (query: string) =>
      fetch(`${url}/v1/audio/transcriptions?model=asr${query}`, {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body: wav(16000, 0.01),
      });
    const sampling = (maxTokens: number) =>
      `&sampling=${encodeURIComponent(JSON.stringify({ max_tokens: maxTokens }))}`;
    for (const query of ['', sampling(16), sampling(16384), '&stream=true']) {
      const res = await post(query);
      expect(res.status).toBe(200);
      await res.text();
    }
    expect(fake.options[0]).not.toHaveProperty('maxTokens');
    expect(fake.options[1]).toMatchObject({ maxTokens: 16 });
    expect(fake.options[2]).toMatchObject({ maxTokens: 16384 });
    expect(fake.options[3]).toMatchObject({ maxTokens: 32 });
  });
  it('extracts multipart file bytes with model fields after the file and matches raw input', async () => {
    const { server, url } = await start();
    const fake = asr();
    server.registry.audio.register('asr', { kind: 'asr', model: fake.model });
    const form = new FormData();
    form.set('file', new Blob([wav()]), 'input.wav');
    form.set('model', 'asr');
    form.set('language', 'en');
    const res = await fetch(`${url}/v1/audio/transcriptions`, { method: 'POST', body: form });
    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({ text: 'whole' });
    expect(fake.options[0]).toMatchObject({ language: 'en', sampleRate: 48000 });
    expect(fake.feeds[0][0]).toBe(0.25);
  });
  it('feeds one rolling interval at a time and returns revisions, never append-only deltas', async () => {
    const { server, url } = await start();
    const fake = asr();
    server.registry.audio.register('asr', { kind: 'asr', model: fake.model });
    const res = await fetch(`${url}/v1/audio/transcriptions?model=asr&stream=true`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: wav(16000, 4.1),
    });
    const events = (await res.text())
      .split('\n')
      .filter((l) => l.startsWith('data: '))
      .map((l) => JSON.parse(l.slice(6)));
    expect(fake.feeds.map((x) => x.length)).toEqual([32000, 32000, 1600]);
    expect(events.every((e) => e.type === 'transcript')).toBe(true);
    expect(events[1].stable + events[1].provisional).toBe('revised');
    expect(events.at(-1)).toMatchObject({ stable: 'final', provisional: '', is_final: true });
  });
  it.each([
    { language: 'English', expected: 'english' },
    { language: 'Chinese', expected: 'chinese' },
    { language: 'english', expected: 'english' },
    { language: undefined, expected: undefined },
  ])(
    'normalizes detected language $language in JSON and every SSE revision without mutating SDK results',
    async ({ language, expected }) => {
      const { server, url } = await start();
      const final = Object.freeze({ ...result('final', 2), language });
      const revision = Object.freeze({ ...result('draft', 1, false), language });
      const fake = asr();
      fake.model.startTranscription = vi.fn(() => ({ cancel: vi.fn(), result: async () => final }));
      fake.model.createStream = vi.fn(async () => ({
        feed: async () => revision,
        finish: async () => final,
        cancel: vi.fn(),
        dispose: vi.fn(async () => {}),
        startCapture: vi.fn(),
      }));
      server.registry.audio.register('asr', { kind: 'asr', model: fake.model });
      for (const stream of [false, true]) {
        const res = await fetch(`${url}/v1/audio/transcriptions?model=asr&stream=${stream}`, {
          method: 'POST',
          headers: { 'Content-Type': 'audio/wav' },
          body: wav(),
        });
        expect(res.status).toBe(200);
        const responses = stream
          ? (await res.text())
              .split('\n')
              .filter((line) => line.startsWith('data: '))
              .map((line) => JSON.parse(line.slice(6)))
          : [await res.json()];
        expect(responses).toHaveLength(stream ? 2 : 1);
        for (const response of responses) {
          if (language === undefined) expect(response).not.toHaveProperty('language');
          else expect(response.language).toBe(expected);
        }
      }
      expect(final.language).toBe(language);
      expect(revision.language).toBe(language);
    },
  );
  it('receives audio before NDJSON request EOF, then accepts more text on the same request', async () => {
    const { server, url } = await start();
    const fake = tts();
    server.registry.audio.register('tts', { kind: 'tts', model: fake.model });
    await new Promise<void>((resolve, reject) => {
      let requestEnded = false;
      const req = httpRequest(
        `${url}/v1/audio/speech/stream`,
        { method: 'POST', headers: { 'Content-Type': 'application/x-ndjson' } },
        (res) => {
          expect(res.statusCode).toBe(200);
          let chunks = 0;
          res.on('data', () => {
            if (chunks++ === 0) {
              expect(requestEnded).toBe(false);
              req.end(JSON.stringify({ type: 'text', text: '第二句。' }) + '\n');
              requestEnded = true;
            }
          });
          res.on('error', reject);
          res.on('end', () => {
            expect(chunks).toBeGreaterThan(0);
            resolve();
          });
        },
      );
      req.setTimeout(3000, () => req.destroy(new Error('Duplex request timed out')));
      req.on('error', reject);
      req.write(JSON.stringify({ type: 'config', model: 'tts', voice: 'voice' }) + '\n');
      req.write(JSON.stringify({ type: 'text', text: '第一句。' }) + '\n');
    });
    expect(fake.submitted.join('')).toBe('第一句。第二句。');
  });
  it('rejects oversized and malformed uploads without dispatching ASR', async () => {
    const { server, url } = await start({ maxUploadBytes: 60 });
    const fake = asr();
    server.registry.audio.register('asr', { kind: 'asr', model: fake.model });
    const res = await fetch(`${url}/v1/audio/transcriptions?model=asr`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: wav(),
    });
    expect(res.status).toBe(413);
    expect(fake.feeds).toEqual([]);
    const bad = await fetch(`${url}/v1/audio/transcriptions?model=asr`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: 'not wav',
    });
    expect(bad.status).toBe(400);
    expect((await bad.json()).error).toMatchObject({ code: 'invalid_audio', param: 'file' });
    expect(fake.feeds).toEqual([]);
  });
  it('counts Unicode characters, rejects text over 4096 and wrong modality', async () => {
    const { server, url } = await start();
    const fake = tts();
    server.registry.audio.register('tts', { kind: 'tts', model: fake.model });
    expect((await speech(url, { input: '😀'.repeat(4097) })).status).toBe(400);
    server.registry.audio.register('asr', { kind: 'asr', model: asr().model });
    expect((await speech(url, { model: 'asr' })).status).toBe(400);
    expect((await speech(url, { model: 'unknown' })).status).toBe(404);
  });
});

it('classifies WAV failures by type, not words in an inference error', async () => {
  const { server, url } = await start();
  const fake = asr();
  fake.model.startTranscription = () => ({
    cancel() {},
    result: async () => {
      throw new Error('WAV backend unavailable');
    },
  });
  server.registry.audio.register('asr', { kind: 'asr', model: fake.model });
  const upload = () =>
    fetch(`${url}/v1/audio/transcriptions?model=asr`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: wav(),
    });
  const failed = await upload();
  expect(failed.status).toBe(500);
  expect((await failed.json()).error.code).toBe('audio_inference_failed');
  const open = vi.spyOn(WavReader, 'open').mockRejectedValueOnce(new InvalidWavError('Malformed container'));
  try {
    const invalid = await upload();
    expect(invalid.status).toBe(400);
    expect((await invalid.json()).error).toMatchObject({ code: 'invalid_audio', param: 'file' });
  } finally {
    open.mockRestore();
  }
});

it('preserves a rolling budget exhaustion in the final revision and rejects malformed multipart', async () => {
  const { server, url } = await start();
  const fake = asr();
  fake.model.createStream = async () => ({
    feed: async () => ({ ...result('partial', 1, false), reachedMaxTokens: true }),
    finish: async () => result('final', 2),
    cancel() {},
    dispose: async () => {},
  });
  server.registry.audio.register('asr', { kind: 'asr', model: fake.model });
  const response = await fetch(`${url}/v1/audio/transcriptions?model=asr&stream=true`, {
    method: 'POST',
    headers: { 'Content-Type': 'audio/wav' },
    body: wav(),
  });
  const events = (await response.text())
    .split('\n')
    .filter((line) => line.startsWith('data: '))
    .map((line) => JSON.parse(line.slice(6)));
  expect(events.at(-1)).toMatchObject({
    is_final: true,
    provisional: '',
    reached_max_tokens: true,
    finish_reason: 'length',
  });
  const malformed = await fetch(`${url}/v1/audio/transcriptions`, {
    method: 'POST',
    headers: { 'Content-Type': 'multipart/form-data; boundary=bad' },
    body: '--bad\r\nContent-Disposition: form-data; name="model"\r\n\r\nasr',
  });
  expect(malformed.status).toBe(400);
});

it('refunds upload quota after both inference errors and successful requests', async () => {
  const { server, url } = await start({ maxTempBytes: 128 });
  const fake = asr();
  let fail = true;
  fake.model.startTranscription = () => ({
    cancel() {},
    result: async () => {
      if (fail) {
        fail = false;
        throw new Error('backend failure');
      }
      return result('recovered');
    },
  });
  server.registry.audio.register('asr', { kind: 'asr', model: fake.model });
  const bytes = wav(16000, 0.001); // 76 bytes; two leaked uploads would exceed the quota.
  for (const status of [500, 200, 200]) {
    const res = await fetch(`${url}/v1/audio/transcriptions?model=asr`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: bytes,
    });
    expect(res.status).toBe(status);
    await res.text();
  }
});

it('keeps a Unicode revision coherent when the native token frontier splits a character', async () => {
  const { server, url } = await start();
  const fake = asr();
  fake.model.createStream = async () => ({
    feed: async () => ({ ...result('你好世界', 1, false), stableText: '你好\uFFFD', provisionalText: '你好世界' }),
    finish: async () => result('你好世界', 2),
    cancel() {},
    dispose: async () => {},
  });
  server.registry.audio.register('asr', { kind: 'asr', model: fake.model });
  const res = await fetch(`${url}/v1/audio/transcriptions?model=asr&stream=true`, {
    method: 'POST',
    headers: { 'Content-Type': 'audio/wav' },
    body: wav(),
  });
  const revisions = (await res.text())
    .split('\n')
    .filter((line) => line.startsWith('data: '))
    .map((line) => JSON.parse(line.slice(6)));
  expect(revisions).toHaveLength(2);
  for (const revision of revisions) expect(revision.stable + revision.provisional).toBe('你好世界');
});

it('releases a complete WAV spool before the client can begin the next synthesis', async () => {
  const { server, url } = await start({ maxTempBytes: 64 });
  server.registry.audio.register('tts', { kind: 'tts', model: tts().model });
  for (let i = 0; i < 3; i++) {
    const res = await speech(url);
    expect(res.status).toBe(200);
    expect((await res.arrayBuffer()).byteLength).toBe(50);
  }
});

describe('audio admission', () => {
  it('counts a rejection when the queue is full', () => {
    const registry = new AudioRegistry(0);
    registry.register('asr', { kind: 'asr', model: asr().model });
    const lane = registry.lane('asr')!;
    const first = lane.beginPreDispatchAdmission();
    const before = queueRejections();
    expect(() => lane.beginPreDispatchAdmission()).toThrow(AudioQueueFullError);
    expect(queueRejections()).toBe(before + 1);
    first.release();
    expect(() => lane.beginPreDispatchAdmission().release()).not.toThrow();
    expect(queueRejections()).toBe(before + 1);
  });
});
