import { describe, expect, it, vi } from 'vite-plus/test';

import { createTtsModel } from '../src/model.js';
import { changeAudioSpeed } from '../src/speed.js';
import type { AudioChunk } from '../src/types.js';

const native = vi.hoisted(() => ({ closed: 0, finished: 0, created: 0 }));
vi.mock('@mlx-node/core', () => ({
  PcmTempo: class {
    constructor() {
      native.created++;
    }
    write(samples: Float32Array) {
      return samples.slice(0, 1);
    }
    finish() {
      native.finished++;
      return new Float32Array([9]);
    }
    close() {
      native.closed++;
    }
  },
}));
const chunk = (startSample: number, segmentIndex = 0): AudioChunk => ({
  samples: new Float32Array([1, 2, 3]),
  sampleRate: 24000,
  channels: 1,
  startSample,
  segmentIndex,
});
async function* source() {
  yield chunk(0);
  yield chunk(3, 1);
}
describe('independent audio speed transform', () => {
  it('releases a TTS model when closed before the first pull', async () => {
    const model = createTtsModel({
      capabilities: {
        family: 'test',
        variant: 'test',
        sampleRate: 24000,
        channels: 1,
        voices: ['voice'],
        languages: ['english'],
        voiceCloning: false,
        conditioning: [{ voice: 'preset', instruct: 'unsupported' }],
        textStreaming: 'segmented',
        audioStreaming: true,
      },
      prepareVoice: async () => '',
      releaseVoice: async () => {},
      dispose: async () => {},
      start() {
        throw new Error('Must not generate audio');
      },
    });
    const source = model.synthesizeStream('Hello.', { voice: 'voice' });
    await changeAudioSpeed(source, 1.15)[Symbol.asyncIterator]().return?.();
    await expect(source.completed).rejects.toMatchObject({ name: 'AbortError' });
    const next = model.synthesizeStream('Reusable.', { voice: 'voice' });
    next.cancel();
    await model.dispose();
  });
  it('claims once and closes an unstarted source on return or throw', async () => {
    for (const operation of ['return', 'throw'] as const) {
      const returned = vi.fn(async () => ({ done: true as const, value: undefined }));
      const input = { [Symbol.asyncIterator]: () => ({ next: vi.fn(), return: returned }) };
      const stream = changeAudioSpeed(input, 1.25);
      const iterator = stream[Symbol.asyncIterator]();
      expect(() => stream[Symbol.asyncIterator]()).toThrow('only be consumed once');
      if (operation === 'return') await iterator.return?.();
      else await expect(iterator.throw?.(new Error('stop'))).rejects.toThrow('stop');
      expect(returned).toHaveBeenCalledTimes(1);
    }
  });
  it('flushes each segment and assigns contiguous transformed frame offsets', async () => {
    native.closed = native.finished = native.created = 0;
    const stream = changeAudioSpeed(source(), 1.25);
    const output = [];
    for await (const part of stream) output.push(part);
    expect(output.map((x) => [x.startSample, x.segmentIndex, [...x.samples]])).toEqual([
      [0, 0, [1]],
      [1, 0, [9]],
      [2, 1, [1]],
      [3, 1, [9]],
    ]);
    expect(stream.stats).toMatchObject({ inputFrames: 6, outputFrames: 4 });
    expect(native).toEqual({ created: 2, finished: 2, closed: 2 });
  });
  it('passes speed one through without native state or PCM changes', async () => {
    const original = chunk(0);
    async function* input() {
      yield original;
    }
    const created = native.created;
    const stream = changeAudioSpeed(input(), 1);
    for await (const part of stream) expect(part.samples).toBe(original.samples);
    expect(native.created).toBe(created);
    expect(stream.stats).toEqual({ inputFrames: 3, outputFrames: 3, processingMs: 0 });
  });
  it('closes the source and processor without flushing on early return', async () => {
    let returned = false;
    async function* input() {
      try {
        yield chunk(0);
        yield chunk(3);
      } finally {
        returned = true;
      }
    }
    const closed = native.closed;
    const finished = native.finished;
    for await (const _ of changeAudioSpeed(input(), 1.25)) break;
    expect(returned).toBe(true);
    expect(native.closed).toBe(closed + 1);
    expect(native.finished).toBe(finished);
  });
  it('rejects discontinuities and releases resources on source failure', async () => {
    for (const invalid of [chunk(8), { ...chunk(0), channels: 2 }, { ...chunk(0), samples: new Float32Array([NaN]) }]) {
      async function* input() {
        yield invalid;
      }
      await expect(changeAudioSpeed(input(), 1.25)[Symbol.asyncIterator]().next()).rejects.toThrow();
    }
    async function* broken() {
      yield chunk(0);
      throw new Error('input failed');
    }
    const closed = native.closed;
    const stream = changeAudioSpeed(broken(), 1.25)[Symbol.asyncIterator]();
    await stream.next();
    await expect(stream.next()).rejects.toThrow('input failed');
    expect(native.closed).toBe(closed + 1);
  });
});
