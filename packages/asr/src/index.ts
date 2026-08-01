/**
 * Qwen3-ASR inference and low-latency microphone capture on Apple Silicon.
 */
import {
  Qwen3AsrCapture,
  Qwen3AsrModel,
  Qwen3AsrStream,
  qwen3AsrInputDevices,
  type Qwen3AsrCaptureOptions,
  type Qwen3AsrCaptureStats,
  type Qwen3AsrInputDevice,
  type Qwen3AsrResult,
  type Qwen3AsrStreamOptions,
  type Qwen3AsrTranscribeOptions,
} from '@mlx-node/core';

export {
  Qwen3AsrCapture,
  Qwen3AsrModel,
  Qwen3AsrStream,
  qwen3AsrInputDevices,
  type Qwen3AsrCaptureOptions,
  type Qwen3AsrCaptureStats,
  type Qwen3AsrInputDevice,
  type Qwen3AsrResult,
  type Qwen3AsrStreamOptions,
  type Qwen3AsrTranscribeOptions,
};

export interface Qwen3AsrRealtimeOptions {
  /** Rolling decode cadence, language, and prompting options. */
  stream?: Qwen3AsrStreamOptions;
  /** CPAL device and callback-ring options. */
  capture?: Qwen3AsrCaptureOptions;
  /** Called for every rolling transcription revision. */
  onResult: (result: Qwen3AsrResult) => void;
  /** Called for asynchronous CPAL or model-worker errors. */
  onError?: (error: Error) => void;
}

export interface Qwen3AsrRealtimeFinal {
  result: Qwen3AsrResult;
  capture: Qwen3AsrCaptureStats;
}

/**
 * Owns one model stream and one CPAL input stream. Call `stop()` to drain the
 * lock-free capture ring and receive the final, non-provisional transcript.
 */
export class Qwen3AsrRealtimeSession {
  readonly stream: Qwen3AsrStream;
  readonly capture: Qwen3AsrCapture;

  #stopPromise: Promise<Qwen3AsrRealtimeFinal> | undefined;
  private readonly getLastError: () => Error | undefined;

  private constructor(stream: Qwen3AsrStream, capture: Qwen3AsrCapture, getLastError: () => Error | undefined) {
    this.stream = stream;
    this.capture = capture;
    this.getLastError = getLastError;
  }

  static async start(model: Qwen3AsrModel, options: Qwen3AsrRealtimeOptions): Promise<Qwen3AsrRealtimeSession> {
    const stream = await model.createStream(options.stream);
    let lastError: Error | undefined;
    let capture: Qwen3AsrCapture;
    try {
      capture = stream.startCapture(options.capture, (error, result) => {
        if (error) {
          lastError = error;
          options.onError?.(error);
          return;
        }
        options.onResult(result);
      });
    } catch (error) {
      await stream.finish().catch(() => undefined);
      throw error;
    }
    return new Qwen3AsrRealtimeSession(stream, capture, () => lastError);
  }

  get deviceName(): string {
    return this.capture.deviceName;
  }

  get sampleRate(): number {
    return this.capture.sampleRate;
  }

  get lastError(): Error | undefined {
    return this.getLastError();
  }

  pause(): void {
    this.capture.pause();
  }

  resume(): void {
    this.capture.resume();
  }

  stop(): Promise<Qwen3AsrRealtimeFinal> {
    this.#stopPromise ??= (async () => {
      let capture: Qwen3AsrCaptureStats;
      try {
        capture = await this.capture.stop();
      } catch (error) {
        await this.stream.finish().catch(() => undefined);
        throw error;
      }
      const result = await this.stream.finish();
      const error = this.lastError;
      if (error) throw error;
      return { result, capture };
    })();
    return this.#stopPromise;
  }
}

export function startRealtimeTranscription(
  model: Qwen3AsrModel,
  options: Qwen3AsrRealtimeOptions,
): Promise<Qwen3AsrRealtimeSession> {
  return Qwen3AsrRealtimeSession.start(model, options);
}
