import { describe, expect, it, vi } from 'vite-plus/test';

import { createTtsModel } from '../src/model.js';
import type { BackendOptions } from '../src/model.js';
import { loadQwen3Backend } from '../src/qwen3.js';
import type { TtsCapabilities } from '../src/types.js';

const mock = vi.hoisted(() => ({
  capabilities: {
    family: 'qwen3_tts',
    variant: 'custom_voice',
    sampleRate: 24000,
    channels: 1,
    voices: ['vivian'],
    languages: ['chinese', 'english'],
    voiceCloning: false,
    conditioning: [{ voice: 'preset', instruct: 'supported' }],
    textStreaming: 'segmented',
    audioStreaming: true,
    samplesPerFrame: 1920, // 80 ms frames at 24 kHz
  } as TtsCapabilities & { samplesPerFrame: number },
  loadArgs: [] as [string, string][],
  startArgs: [] as [string, Record<string, unknown>][],
}));
vi.mock('@mlx-node/core', () => ({
  TtsNativeModel: {
    async load(path: string, optionsJson: string) {
      mock.loadArgs.push([path, optionsJson]);
      return {
        metadata: JSON.stringify(mock.capabilities),
        start(text: string, optionsJson: string) {
          mock.startArgs.push([text, JSON.parse(optionsJson)]);
          return {
            next: async () => ({ samples: new Float32Array(), finished: true, finishReason: 'eos' }),
            cancel: () => {},
            waitFinished: async () => {},
          };
        },
        prepareVoice: async () => 'prepared',
        releaseVoice: async () => {},
        dispose: async () => {},
      };
    },
  },
}));

const options = (overrides: Partial<BackendOptions> = {}): BackendOptions => ({
  voice: 'vivian',
  language: 'chinese',
  chunkDurationMs: 160,
  audioBufferSeconds: 1,
  ...overrides,
});

describe('Qwen3 backend adapter', () => {
  it('passes instruction-cache options through to the native loader', async () => {
    const backend = await loadQwen3Backend('/model', {
      instructionCache: { enabled: true, maxBytes: 1024, maxEntries: 4 },
    });
    expect(mock.loadArgs).toEqual([
      ['/model', JSON.stringify({ instruction_cache: { enabled: true, max_bytes: 1024, max_entries: 4 } })],
    ]);
    const { samplesPerFrame, ...rest } = mock.capabilities;
    expect(samplesPerFrame).toBe(1920);
    expect(backend.capabilities).toEqual(rest);
  });
  it('rejects malformed instruction-cache options before loading', async () => {
    const before = mock.loadArgs.length;
    await expect(loadQwen3Backend('/model', { instructionCache: { enabled: true, maxBytes: -1 } })).rejects.toThrow(
      RangeError,
    );
    await expect(
      loadQwen3Backend('/model', { instructionCache: { enabled: 'yes' as unknown as boolean } }),
    ).rejects.toThrow(TypeError);
    expect(mock.loadArgs.length).toBe(before);
  });
  it('maps SDK options to the native start JSON', async () => {
    const backend = await loadQwen3Backend('/model');
    backend.start('你好。', options({ instruct: '慢速读。', seed: 534, topP: 0.9, repetitionPenalty: 1.1 }));
    expect(mock.startArgs.at(-1)).toEqual([
      '你好。',
      {
        voice: 'vivian',
        voice_description: undefined,
        instruct: '慢速读。',
        prepared_voice_id: undefined,
        language: 'chinese',
        temperature: undefined,
        top_k: undefined,
        top_p: 0.9,
        repetition_penalty: 1.1,
        seed: 534,
        chunk_frames: 2, // 160 ms / 80 ms frames
        buffer_chunks: 6, // 1 s buffer / 160 ms chunks
        max_frames: 1500, // default 120 s / 80 ms frames
      },
    ]);
  });
  it('derives chunk and capacity frames from the codec frame duration', async () => {
    const backend = await loadQwen3Backend('/model');
    backend.start('x', options({ chunkDurationMs: 100, audioBufferSeconds: 0.5, maxDurationSeconds: 1 }));
    expect(mock.startArgs.at(-1)?.[1]).toMatchObject({ chunk_frames: 2, buffer_chunks: 3, max_frames: 13 });
    backend.start('x', options({ chunkDurationMs: 1 }));
    expect(mock.startArgs.at(-1)?.[1]).toMatchObject({ chunk_frames: 1, buffer_chunks: 12 });
  });
  it('rejects buffers above the native capacity before generation', async () => {
    const backend = await loadQwen3Backend('/model');
    const before = mock.startArgs.length;
    expect(() => backend.start('x', options({ audioBufferSeconds: 700 }))).toThrow(RangeError);
    expect(() => backend.start('x', options({ audioBufferSeconds: 700 }))).toThrow(/buffer capacity \(4096 chunks\)/);
    expect(mock.startArgs.length).toBe(before);
  });
  it('wires model defaults through the stream backend', async () => {
    const backend = await loadQwen3Backend('/model');
    const model = createTtsModel(backend);
    await model.synthesize('你好。', { voice: 'vivian', language: 'chinese', seed: 1 });
    expect(mock.startArgs.at(-1)?.[1]).toMatchObject({
      voice: 'vivian',
      language: 'chinese',
      seed: 1,
      chunk_frames: 2,
      buffer_chunks: 6,
    });
    await model.dispose();
  });
});
