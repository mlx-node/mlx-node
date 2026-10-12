import { setFlagsFromString } from 'node:v8';
import { runInNewContext } from 'node:vm';

import { describe, it, expect, vi } from 'vite-plus/test';

import { createTtsModel, type TtsBackend } from '../src/model.js';
import { BoundedQueue } from '../src/queue.js';
import { TextSegmenter } from '../src/segmenter.js';
import type { TtsStream } from '../src/types.js';

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function fake() {
  const submitted: string[] = [];
  let cancelled = 0;
  const backend: TtsBackend = {
    capabilities: {
      family: 'test',
      variant: 'test',
      sampleRate: 10,
      channels: 1,
      voices: ['voice'],
      languages: ['english'],
      voiceCloning: true,
      conditioning: [
        { voice: 'preset', instruct: 'supported' },
        { voice: 'reference', instruct: 'unsupported' },
      ],
      textStreaming: 'segmented',
      audioStreaming: true,
    },
    prepareVoice: vi.fn(async () => 'prepared'),
    releaseVoice: vi.fn(async () => {}),
    dispose: vi.fn(async () => {}),
    start(text) {
      submitted.push(text);
      let at = 0;
      return {
        next: async () =>
          at++ < 2
            ? { samples: new Float32Array([1, 2]), finished: false }
            : { samples: new Float32Array(), finished: true, finishReason: 'eos', synthesisMs: 10 },
        cancel: () => {
          cancelled++;
        },
        waitFinished: async () => {},
      };
    },
  };
  return { model: createTtsModel(backend), submitted, backend, cancelled: () => cancelled };
}

describe('text commits', () => {
  it('preserves every UTF-16 code unit through arbitrary input chunks', () => {
    const text =
      '你好！👨‍👩‍👦 café e\u0301。 A price is 12.50 dollars. This is a long sentence with words and clauses, followed by a tail';
    for (let size = 1; size <= 17; size++) {
      const segmenter = new TextSegmenter(16);
      const parts: string[] = [];
      for (let i = 0; i < text.length; i += size) parts.push(...segmenter.push(text.slice(i, i + size)));
      parts.push(...segmenter.push('', true));
      expect(parts.join('')).toBe(text);
    }
  });
  it('flushes complete Chinese sentences before input end and preserves a trailing decimal', () => {
    const s = new TextSegmenter();
    expect([...s.push('你好。')]).toEqual(['你好。']);
    expect([...s.push('It costs 12.')]).toEqual([]);
    expect([...s.push('50 dollars. ', true)].join('')).toBe('It costs 12.50 dollars. ');
  });
  it('never exceeds configured graphemes on a complete string', () => {
    const text = '👨‍👩‍👦🙂'.repeat(40);
    const s = new TextSegmenter(7);
    const parts = [...s.push(text, true)];
    const g = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
    expect(parts.join('')).toBe(text);
    for (const p of parts) expect([...g.segment(p)].length).toBeLessThanOrEqual(7);
  });
  it('keeps combining marks attached at clause and whitespace boundaries', () => {
    const graphemes = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
    for (const text of ['a,\u0301bc', 'a \u0301bc', 'a，\uFE00bc', 'a:\u0301bc']) {
      const expectedBoundaries = new Set([...graphemes.segment(text)].map((part) => part.index));
      expectedBoundaries.add(text.length);
      const parts = [...new TextSegmenter(2).push(text, true)];
      let offset = 0;
      for (const part of parts) {
        offset += part.length;
        expect(expectedBoundaries.has(offset)).toBe(true);
        expect([...graphemes.segment(part)].length).toBeLessThanOrEqual(2);
      }
      expect(parts.join('')).toBe(text);
    }
  });
});
describe('session lifecycle', () => {
  it('rejects invalid sampling options before starting the backend', async () => {
    const { model, submitted } = fake();
    for (const options of [
      { temperature: NaN },
      { temperature: Infinity },
      { topP: NaN },
      { topP: 1.1 },
      { topK: 1.5 },
      { topK: -1 },
      { repetitionPenalty: 0 },
    ])
      expect(() => model.synthesizeStream('Hello.', { voice: 'voice', ...options })).toThrow();
    expect(submitted).toEqual([]);
    await model.dispose();
  });
  it('starts synthesis before the producer ends, and cancellation wakes a paused input', async () => {
    const { model, submitted } = fake();
    let returned = false;
    const input = {
      [Symbol.asyncIterator]() {
        let n = 0;
        return {
          next: () =>
            n++ === 0
              ? Promise.resolve({ done: false as const, value: '你好。' })
              : new Promise<IteratorResult<string>>(() => {}),
          return: async () => {
            returned = true;
            return { done: true as const, value: undefined };
          },
        };
      },
    };
    const stream = model.synthesizeStream(input, { voice: 'voice' });
    const iterator = stream[Symbol.asyncIterator]();
    expect((await iterator.next()).value?.samples.length).toBe(2);
    expect(submitted).toEqual(['你好。']);
    expect(() => model.synthesizeStream('busy', { voice: 'voice' })).toThrow('busy');
    stream.cancel();
    await expect(stream.completed).rejects.toMatchObject({ name: 'AbortError' });
    expect(returned).toBe(true);
    expect((await model.synthesize('Reusable.', { voice: 'voice' })).stats.segments).toBe(1);
    await model.dispose();
  });
  it('cancels on iterator break and releases the native producer before reuse', async () => {
    const { model, cancelled } = fake();
    const stream = model.synthesizeStream('Hi.', { voice: 'voice' });
    for await (const _ of stream) break;
    await expect(stream.completed).rejects.toMatchObject({ name: 'AbortError' });
    expect(cancelled()).toBe(1);
    expect((await model.synthesize('Again.', { voice: 'voice' })).samples.length).toBe(4);
    await model.dispose();
  });
  it('finishes cancellation even when the input iterator return throws synchronously', async () => {
    const { model } = fake();
    const input: AsyncIterable<string> = {
      [Symbol.asyncIterator]() {
        let first = true;
        return {
          next() {
            if (first) {
              first = false;
              return Promise.resolve({ done: false as const, value: 'Hi!' });
            }
            return new Promise<IteratorResult<string>>(() => {});
          },
          return() {
            throw new Error('return exploded');
          },
        };
      },
    };
    const stream = model.synthesizeStream(input, { voice: 'voice' });
    await stream[Symbol.asyncIterator]().next();
    stream.cancel();
    await expect(stream.completed).rejects.toMatchObject({ name: 'AbortError' });
    await model.dispose();
  });
  it('cancel before iteration and disposal are idempotent', async () => {
    const { model, submitted, backend } = fake();
    const stream = model.synthesizeStream('Never.', { voice: 'voice' });
    stream.cancel();
    await expect(stream.completed).rejects.toMatchObject({ name: 'AbortError' });
    expect(submitted).toEqual([]);
    expect(await stream[Symbol.asyncIterator]().next()).toMatchObject({ done: true });
    await Promise.all([model.dispose(), model.dispose()]);
    expect(backend.dispose).toHaveBeenCalledTimes(1);
    expect(() => model.synthesizeStream('After.', { voice: 'voice' })).toThrow('disposed');
  });
  it('removes its original abort listener when the caller reuses the options object', async () => {
    const { model } = fake();
    const controller = new AbortController();
    const remove = vi.spyOn(controller.signal, 'removeEventListener');
    const options = { voice: 'voice', signal: controller.signal };
    const stream = model.synthesizeStream('Hello.', options);
    options.signal = new AbortController().signal;
    for await (const _ of stream) {
      // Drain normally to exercise successful completion cleanup.
    }
    await stream.completed;
    expect(remove).toHaveBeenCalledExactlyOnceWith('abort', expect.any(Function));
    await model.dispose();
  });
  it('cancels the stream when a caller AbortSignal fires', async () => {
    const { model, cancelled } = fake();
    const controller = new AbortController();
    const stream = model.synthesizeStream('Hi.', { voice: 'voice', signal: controller.signal });
    const iterator = stream[Symbol.asyncIterator]();
    expect((await iterator.next()).value?.samples.length).toBe(2);
    controller.abort();
    // Abort delivers a generator return: the pull resolves done while
    // completed still rejects with AbortError.
    expect(await iterator.next()).toMatchObject({ done: true });
    await expect(stream.completed).rejects.toMatchObject({ name: 'AbortError' });
    expect(cancelled()).toBe(1);
    await model.synthesize('Reusable.', { voice: 'voice' });
    await model.dispose();
  });
  // Collection timing and FinalizationRegistry callback delivery are not
  // deterministic (conservative stack roots, loaded CI machines), so the
  // retry budget absorbs the occasional run where GC never reclaims in time.
  it('frees the busy slot on GC for an abandoned stream whose signal stays alive', { retry: 2 }, async () => {
    const { model, submitted } = fake();
    // Node exposes gc() only under --expose-gc; setting the flag before a new
    // vm context is created yields a gc that collects this process's heap.
    setFlagsFromString('--expose_gc');
    const gc = runInNewContext('gc') as () => void;
    const collected = { freed: false };
    const registry = new FinalizationRegistry(() => {
      collected.freed = true;
    });
    const controller = new AbortController();
    const abandon = () => {
      const stream = model.synthesizeStream('Abandoned.', { voice: 'voice', signal: controller.signal });
      registry.register(stream, 0);
    };
    abandon();
    expect(submitted).toEqual([]);
    expect(() => model.synthesizeStream('Busy.', { voice: 'voice' })).toThrow('busy');
    for (let i = 0; !collected.freed && i < 12; i++) {
      // Churn clears conservative stack roots that may still point at the stream.
      let acc = 0;
      for (let n = 0; n < 1e5; n++) acc = (acc + n) % 7;
      gc();
      gc();
      await new Promise((resolve) => setImmediate(resolve));
    }
    expect(collected.freed).toBe(true);
    // The live signal can no longer pin the stream, so a new request proceeds.
    expect((await model.synthesize('After.', { voice: 'voice' })).samples.length).toBe(4);
    await model.dispose();
  });
  it('rejects completed with AbortError when cancel races the natural queue close', async () => {
    const { model, backend } = fake();
    let stream: TtsStream | undefined;
    vi.spyOn(backend, 'start').mockImplementationOnce(() => ({
      next: async () => ({
        samples: new Float32Array([1, 2]),
        finished: true,
        finishReason: 'eos',
        synthesisMs: 10,
      }),
      cancel: () => {},
      // Awaits queue.shift() resume after the terminal packet: abort before
      // waitFinished resolves so the close is not a natural end.
      waitFinished: async () => stream?.cancel(),
    }));
    stream = model.synthesizeStream('Hi.', { voice: 'voice' });
    const iterator = stream[Symbol.asyncIterator]();
    // The only chunk carries the terminal flag; the second pull enters the
    // waitFinished/queue-close window where the cancel lands.
    expect((await iterator.next()).value?.samples.length).toBe(2);
    await expect(iterator.next()).rejects.toMatchObject({ name: 'AbortError' });
    await expect(stream.completed).rejects.toMatchObject({ name: 'AbortError' });
    await model.dispose();
  });
  it('return before the first pull settles completion and releases the model', async () => {
    const { model, submitted } = fake();
    const stream = model.synthesizeStream('Never.', { voice: 'voice' });
    const iterator = stream[Symbol.asyncIterator]();
    await iterator.return!();
    await expect(stream.completed).rejects.toMatchObject({ name: 'AbortError' });
    expect(await iterator.next()).toMatchObject({ done: true });
    expect(submitted).toEqual([]);
    await model.synthesize('Again.', { voice: 'voice' });
    await model.dispose();
  });
  it('maintains sample positions across segments and awaits completion', async () => {
    const { model } = fake();
    const stream = model.synthesizeStream('你好。再见。', { voice: 'voice' });
    const positions = [];
    for await (const c of stream) positions.push([c.startSample, c.segmentIndex]);
    expect(positions).toEqual([
      [0, 0],
      [2, 0],
      [4, 1],
      [6, 1],
    ]);
    expect(await stream.completed).toMatchObject({
      audioSeconds: 0.8,
      segments: 2,
      finishReason: 'eos',
      synthesisMs: 20,
    });
    await model.dispose();
  });
  it('scopes prepared voices to their model and reuses them', async () => {
    const a = fake(),
      b = fake();
    const voice = await a.model.prepareVoice({
      audio: { samples: new Float32Array([1, -1, 0, 0]), sampleRate: 10, channels: 2 },
      transcript: 'Hello',
    });
    expect(a.backend.prepareVoice).toHaveBeenCalledWith(new Float32Array([0, 0]), 10, 'Hello');
    expect(() => b.model.synthesizeStream('Hello', { voice })).toThrow('does not belong');
    await a.model.synthesize('One.', { voice });
    await a.model.synthesize('Two.', { voice });
    expect(a.backend.prepareVoice).toHaveBeenCalledTimes(1);
    await a.model.dispose();
    await b.model.dispose();
  });
  it('releases a busy voice once the native queue drains, then releases it once', async () => {
    const { model, backend } = fake();
    const voice = await model.prepareVoice({
      audio: { samples: new Float32Array([0, 1]), sampleRate: 10 },
      transcript: 'Hello',
    });
    const stream = model.synthesizeStream('Hello.', { voice });
    await voice.dispose();
    expect(backend.releaseVoice).toHaveBeenCalledExactlyOnceWith(voice.id);
    stream.cancel();
    await expect(stream.completed).rejects.toMatchObject({ name: 'AbortError' });
    expect(() => model.synthesizeStream('Released.', { voice })).toThrow('does not belong');
    await Promise.all([voice.dispose(), voice.dispose()]);
    expect(backend.releaseVoice).toHaveBeenCalledExactlyOnceWith(voice.id);
    expect(() => model.synthesizeStream('Released.', { voice })).toThrow('does not belong');
    await voice.dispose();
    expect(backend.releaseVoice).toHaveBeenCalledTimes(1);
    await model.dispose();
  });
  it('reserves the model while a voice is being released', async () => {
    const { model, backend } = fake();
    const reference = { audio: { samples: new Float32Array([0, 1]), sampleRate: 10 }, transcript: 'Hello' };
    vi.spyOn(backend, 'prepareVoice').mockResolvedValueOnce('first').mockResolvedValueOnce('second');
    const first = await model.prepareVoice(reference);
    const second = await model.prepareVoice(reference);
    const released = deferred<void>();
    vi.spyOn(backend, 'releaseVoice').mockImplementationOnce(() => released.promise);
    const release = first.dispose();
    expect(() => model.synthesizeStream('Busy.', { voice: 'voice' })).toThrow('busy');
    expect(() => model.prepareVoice(reference)).toThrow('busy');
    // A second release joins the in-flight one instead of busy-rejecting.
    const secondRelease = second.dispose();
    released.resolve();
    await Promise.all([release, secondRelease]);
    expect(backend.releaseVoice).toHaveBeenNthCalledWith(1, 'first');
    expect(backend.releaseVoice).toHaveBeenNthCalledWith(2, 'second');
    expect(() => model.synthesizeStream('Released.', { voice: second })).toThrow('does not belong');
    await model.synthesize('Model is free again.', { voice: 'voice' });
    await model.dispose();
  });
  it('releases a voice during preparation once the native queue drains', async () => {
    const { model, backend } = fake();
    const reference = { audio: { samples: new Float32Array([0, 1]), sampleRate: 10 }, transcript: 'Hello' };
    const voice = await model.prepareVoice(reference);
    const preparation = deferred<string>();
    vi.spyOn(backend, 'prepareVoice').mockImplementationOnce(() => preparation.promise);
    const pending = model.prepareVoice(reference);
    await voice.dispose();
    expect(backend.releaseVoice).toHaveBeenCalledExactlyOnceWith(voice.id);
    preparation.resolve('second');
    await pending;
    expect(() => model.synthesizeStream('Released.', { voice })).toThrow('does not belong');
    await model.dispose();
  });
  it('allows retrying failed voice release and does not call a disposed backend', async () => {
    const { model, backend } = fake();
    const voice = await model.prepareVoice({
      audio: { samples: new Float32Array([0, 1]), sampleRate: 10 },
      transcript: 'Hello',
    });
    const error = new Error('release failed');
    vi.spyOn(backend, 'releaseVoice').mockRejectedValueOnce(error);
    await expect(voice.dispose()).rejects.toBe(error);
    await model.synthesize('Still owned.', { voice });
    await voice.dispose();
    expect(backend.releaseVoice).toHaveBeenCalledTimes(2);
    const other = await model.prepareVoice({
      audio: { samples: new Float32Array([0, 1]), sampleRate: 10 },
      transcript: 'Hello',
    });
    await model.dispose();
    await Promise.all([voice.dispose(), other.dispose()]);
    expect(backend.releaseVoice).toHaveBeenCalledTimes(2);
  });
  it('starts backend shutdown before waiting for an outstanding voice preparation', async () => {
    const { model, backend } = fake();
    const preparation = deferred<string>();
    const error = new Error('preparation cancelled');
    vi.spyOn(backend, 'prepareVoice').mockImplementationOnce(() => preparation.promise);
    vi.spyOn(backend, 'dispose').mockImplementationOnce(async () => preparation.reject(error));
    const prepared = model.prepareVoice({
      audio: { samples: new Float32Array([0, 1]), sampleRate: 10 },
      transcript: 'Hello',
    });
    const rejected = expect(prepared).rejects.toBe(error);
    await model.dispose();
    await rejected;
    expect(backend.dispose).toHaveBeenCalledTimes(1);
    expect(() => model.synthesizeStream('After shutdown.', { voice: 'voice' })).toThrow('disposed');
  });
  it('waits for an in-flight voice release during model disposal', async () => {
    const { model, backend } = fake();
    const voice = await model.prepareVoice({
      audio: { samples: new Float32Array([0, 1]), sampleRate: 10 },
      transcript: 'Hello',
    });
    const released = deferred<void>();
    vi.spyOn(backend, 'releaseVoice').mockImplementationOnce(() => released.promise);
    const release = voice.dispose();
    const dispose = model.dispose();
    let disposed = false;
    void dispose.then(() => {
      disposed = true;
    });
    await Promise.resolve();
    expect(backend.dispose).toHaveBeenCalledTimes(1);
    expect(disposed).toBe(false);
    released.resolve();
    await Promise.all([release, dispose]);
    await voice.dispose();
    expect(backend.releaseVoice).toHaveBeenCalledTimes(1);
  });
  it('propagates an input error to iterator and completion', async () => {
    const { model } = fake();
    const input: AsyncIterable<string> = {
      [Symbol.asyncIterator]: () => ({
        next: async () => {
          throw new Error('source failed');
        },
      }),
    };
    const stream = model.synthesizeStream(input, { voice: 'voice' });
    await expect(stream[Symbol.asyncIterator]().next()).rejects.toThrow('source failed');
    await expect(stream.completed).rejects.toThrow('source failed');
    await model.dispose();
  });
  it.each([0, '', false, null, undefined])('preserves the exact input rejection value %s', async (error) => {
    const { model } = fake();
    const input: AsyncIterable<string> = {
      [Symbol.asyncIterator]: () => ({ next: () => Promise.reject(error) }),
    };
    const stream = model.synthesizeStream(input, { voice: 'voice' });
    await expect(stream[Symbol.asyncIterator]().next()).rejects.toBe(error);
    await expect(stream.completed).rejects.toBe(error);
    await model.synthesize('Reusable.', { voice: 'voice' });
    await model.dispose();
  });
  it('preserves an input failure when cancelled while the consumer is paused', async () => {
    const { model, cancelled } = fake();
    const pendingInput = deferred<IteratorResult<string>>();
    const error = new Error('input failed while paused');
    const input: AsyncIterable<string> = {
      [Symbol.asyncIterator]() {
        let first = true;
        return {
          next() {
            if (first) {
              first = false;
              return Promise.resolve({ done: false as const, value: 'Hello!' });
            }
            return pendingInput.promise;
          },
        };
      },
    };
    const stream = model.synthesizeStream(input, { voice: 'voice' });
    await stream[Symbol.asyncIterator]().next();
    pendingInput.reject(error);
    await vi.waitFor(() => expect(cancelled()).toBe(1));
    stream.cancel();
    await expect(stream.completed).rejects.toBe(error);
    await model.dispose();
  });
  it('bounds producer progress and wakes blocked writers on close', async () => {
    const controller = new AbortController();
    const q = new BoundedQueue<number>(1, controller.signal);
    await q.push(1);
    let pushed = false;
    const second = q.push(2).then(() => {
      pushed = true;
    });
    await Promise.resolve();
    expect(pushed).toBe(false);
    expect(await q.shift()).toBe(1);
    await second;
    expect(await q.shift()).toBe(2);
    await q.push(3);
    const pending = q.push(4);
    q.close(new Error('closed'));
    await expect(pending).rejects.toThrow('closed');
  });
});

describe('instruction commits', () => {
  async function* events(values: import('../src/types.js').TtsInputEvent[]) {
    yield* values;
  }
  it('commits pending text under the old instruction and freezes queued segments', async () => {
    const { model, backend } = fake();
    const start = vi.spyOn(backend, 'start');
    await model.synthesize(
      events([
        '第一句',
        { type: 'instruct', value: 'excited' },
        { type: 'text', text: '第二句。第三句' },
        { type: 'instruct', value: null },
        '第四句。',
      ]),
      { voice: 'voice', instruct: 'calm', queuedSegments: 1 },
    );
    expect(start.mock.calls.map(([text, options]) => [text, options.instruct])).toEqual([
      ['第一句', 'calm'],
      ['第二句。', 'excited'],
      ['第三句', 'excited'],
      ['第四句。', undefined],
    ]);
    await model.dispose();
  });
  it('does not split a sentence when the effective instruction is unchanged', async () => {
    const { model, backend } = fake();
    const start = vi.spyOn(backend, 'start');
    await model.synthesize(
      events([
        'Hello',
        { type: 'instruct', value: 'calm' },
        ' world',
        { type: 'instruct', value: null },
        { type: 'instruct', value: '  ' },
        { type: 'instruct', value: 'excited' },
        { type: 'flush' },
        'Bye.',
      ]),
      { voice: 'voice', instruct: 'calm' },
    );
    expect(start.mock.calls.map(([text, options]) => [text, options.instruct])).toEqual([
      ['Hello world', 'calm'],
      ['Bye.', 'excited'],
    ]);
    await model.dispose();
  });
  it('rejects unsupported reference instructions before any generation', async () => {
    const { model, submitted } = fake();
    const voice = await model.prepareVoice({
      audio: { samples: new Float32Array([0, 1]), sampleRate: 10 },
      transcript: 'ref',
    });
    expect(() => model.synthesizeStream('Hello', { voice, instruct: 'calm' })).toThrow('unsupported');
    await expect(model.synthesize(events([{ type: 'instruct', value: 'calm' }, 'Hello']), { voice })).rejects.toThrow(
      'unsupported',
    );
    expect(submitted).toEqual([]);
    await model.synthesize('Again.', { voice });
    await model.dispose();
  });
  it('keeps the VoiceDesign description when a style is cleared', async () => {
    const { backend } = fake();
    backend.capabilities.conditioning = [{ voice: 'description', instruct: 'supported' }];
    const model = createTtsModel(backend);
    const start = vi.spyOn(backend, 'start');
    const voice = { type: 'description' as const, description: 'A low voice.' };
    await model.synthesize(events(['One.', { type: 'instruct', value: null }, 'Two.']), { voice, instruct: 'calm' });
    expect(start.mock.calls.map(([, o]) => [o.voiceDescription, o.instruct])).toEqual([
      ['A low voice.', 'calm'],
      ['A low voice.', undefined],
    ]);
    expect(() => model.synthesizeStream('Hi', { voice: { type: 'description', description: ' ' } })).toThrow(
      'nonempty',
    );
    expect(() => model.synthesizeStream('Hi', { voice: 'voice' })).toThrow('Unsupported TTS voice mode');
    await model.dispose();
  });
  it('cancels while a control event is blocked behind queued text and permits reuse', async () => {
    const { model, submitted } = fake();
    const stream = model.synthesizeStream(
      events(['One. Two. Three. Four', { type: 'instruct', value: 'quiet' }, 'Never.']),
      { voice: 'voice', instruct: 'calm', queuedSegments: 1 },
    );
    const iterator = stream[Symbol.asyncIterator]();
    await iterator.next();
    stream.cancel();
    await expect(stream.completed).rejects.toMatchObject({ name: 'AbortError' });
    expect(submitted.join('')).not.toContain('Never');
    await model.synthesize('Reusable.', { voice: 'voice' });
    await model.dispose();
  });
  it('invalid control events fail without leaking a busy session', async () => {
    const { model } = fake();
    for (const value of [42, undefined, {}]) {
      const input = events([{ type: 'instruct', value } as unknown as import('../src/types.js').TtsInputEvent]);
      await expect(model.synthesize(input, { voice: 'voice' })).rejects.toThrow('Invalid TTS input event');
    }
    await model.synthesize('Again.', { voice: 'voice' });
    await model.dispose();
  });
});
