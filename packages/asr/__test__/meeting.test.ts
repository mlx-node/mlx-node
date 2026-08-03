import type {
  Qwen3AsrCaptureOptions,
  Qwen3AsrCaptureStats,
  Qwen3AsrModel,
  Qwen3AsrResult,
  Qwen3AsrStream,
} from '@mlx-node/core';
import { describe, expect, it } from 'vite-plus/test';

import { startMeetingTranscription } from '../src/index.js';

const RESULT = { text: 'hello' } as Qwen3AsrResult;
const STATS = { capturedFrames: 480, droppedFrames: 0 } as Qwen3AsrCaptureStats;

class FakeCapture {
  readonly deviceName = 'Fake device';
  readonly sampleRate = 48_000;
  readonly channels = 1;
  readonly source = 'microphone';
  pauses = 0;
  resumes = 0;
  stops = 0;

  pause(): void {
    this.pauses += 1;
  }

  resume(): void {
    this.resumes += 1;
  }

  async stop(): Promise<Qwen3AsrCaptureStats> {
    this.stops += 1;
    return STATS;
  }
}

class FakeStream {
  readonly capture = new FakeCapture();
  captureOptions: Qwen3AsrCaptureOptions | undefined;
  callback: ((error: Error | null, result: Qwen3AsrResult) => void) | undefined;
  finishes = 0;
  failStart = false;

  startCapture(
    options: Qwen3AsrCaptureOptions | undefined,
    callback: (error: Error | null, result: Qwen3AsrResult) => void,
  ): FakeCapture {
    if (this.failStart) throw new Error('capture failed');
    this.captureOptions = options;
    this.callback = callback;
    return this.capture;
  }

  async finish(): Promise<Qwen3AsrResult> {
    this.finishes += 1;
    return RESULT;
  }
}

function fakeModel(failSecondCapture = false): { model: Qwen3AsrModel; streams: FakeStream[] } {
  const streams: FakeStream[] = [];
  const model = {
    async createStream(): Promise<Qwen3AsrStream> {
      const stream = new FakeStream();
      stream.failStart = failSecondCapture && streams.length === 1;
      streams.push(stream);
      return stream as unknown as Qwen3AsrStream;
    },
  } as Qwen3AsrModel;
  return { model, streams };
}

describe('startMeetingTranscription', () => {
  it('starts source-tagged system and microphone tracks and stops them once', async () => {
    const { model, streams } = fakeModel();
    const events: string[] = [];
    const meeting = await startMeetingTranscription(model, {
      onResult: ({ source, result }) => events.push(`${source}:${result.text}`),
    });

    expect(streams).toHaveLength(2);
    expect(streams[0].captureOptions?.source).toBe('systemAudio');
    expect(streams[1].captureOptions?.source).toBe('microphone');

    streams[0].callback?.(null, RESULT);
    streams[1].callback?.(null, RESULT);
    expect(events).toStrictEqual(['systemAudio:hello', 'microphone:hello']);

    meeting.pause();
    meeting.resume();
    expect(streams.map((stream) => [stream.capture.pauses, stream.capture.resumes])).toStrictEqual([
      [1, 1],
      [1, 1],
    ]);

    const firstStop = meeting.stop();
    const secondStop = meeting.stop();
    expect(secondStop).toBe(firstStop);
    const final = await firstStop;
    expect(final.microphone?.result).toBe(RESULT);
    expect(final.systemAudio?.result).toBe(RESULT);
    expect(streams.map((stream) => [stream.capture.stops, stream.finishes])).toStrictEqual([
      [1, 1],
      [1, 1],
    ]);
  });

  it('cleans up the first track if the second source fails to start', async () => {
    const { model, streams } = fakeModel(true);

    await expect(
      startMeetingTranscription(model, {
        onResult() {},
      }),
    ).rejects.toThrow('capture failed');
    expect(streams[0].capture.stops).toBe(1);
    expect(streams[0].finishes).toBe(1);
  });

  it('rejects a meeting with no enabled source', async () => {
    const { model, streams } = fakeModel();

    await expect(
      startMeetingTranscription(model, {
        microphone: false,
        systemAudio: false,
        onResult() {},
      }),
    ).rejects.toThrow('At least one meeting audio source must be enabled');
    expect(streams).toHaveLength(0);
  });
});
