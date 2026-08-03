/**
 * Qwen3-ASR inference and low-latency Core Audio capture on Apple Silicon.
 */
import {
  Qwen3AsrCapture,
  Qwen3AsrCaptureSource,
  Qwen3AsrModel,
  Qwen3AsrStream,
  qwen3AsrAudioDevices,
  qwen3AsrInputDevices,
  type Qwen3AsrAudioDevice,
  type Qwen3AsrCaptureOptions,
  type Qwen3AsrCaptureStats,
  type Qwen3AsrInputDevice,
  type Qwen3AsrResult,
  type Qwen3AsrStreamOptions,
  type Qwen3AsrTranscribeOptions,
} from '@mlx-node/core';

export {
  Qwen3AsrCapture,
  Qwen3AsrCaptureSource,
  Qwen3AsrModel,
  Qwen3AsrStream,
  qwen3AsrAudioDevices,
  qwen3AsrInputDevices,
  type Qwen3AsrAudioDevice,
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
  /** Core Audio source, device, application filter, and callback-ring options. */
  capture?: Qwen3AsrCaptureOptions;
  /** Called for every rolling transcription revision. */
  onResult: (result: Qwen3AsrResult) => void;
  /** Called for asynchronous capture or model-worker errors. */
  onError?: (error: Error) => void;
}

export interface Qwen3AsrRealtimeFinal {
  result: Qwen3AsrResult;
  capture: Qwen3AsrCaptureStats;
}

/**
 * Owns one model stream and one Core Audio source. Call `stop()` to drain the
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

  get source(): Qwen3AsrCaptureSource {
    return this.capture.source;
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

type CaptureTimingOptions = Pick<Qwen3AsrCaptureOptions, 'feedMilliseconds' | 'ringSeconds'>;

export interface Qwen3AsrMicrophoneOptions extends CaptureTimingOptions {
  /** Stable input-device UID from `qwen3AsrAudioDevices()`. */
  deviceId?: string;
  /** Input-device name. Prefer `deviceId` when persisting a selection. */
  deviceName?: string;
}

export interface Qwen3AsrSystemAudioOptions extends CaptureTimingOptions {
  /** Stable output-device UID from `qwen3AsrAudioDevices()`. */
  deviceId?: string;
  /** Output-device name. Prefer `deviceId` when persisting a selection. */
  deviceName?: string;
  /** Capture only these applications. Omit to capture all system output. */
  applicationBundleIds?: string[];
}

export type Qwen3AsrMeetingSource = 'microphone' | 'systemAudio';

export interface Qwen3AsrMeetingResultEvent {
  source: Qwen3AsrMeetingSource;
  result: Qwen3AsrResult;
}

export interface Qwen3AsrMeetingErrorEvent {
  source: Qwen3AsrMeetingSource;
  error: Error;
}

export interface Qwen3AsrMeetingOptions {
  /** Shared rolling decode cadence, language, and prompting options. */
  stream?: Qwen3AsrStreamOptions;
  /** Microphone capture options. `false` disables this track. Default enabled. */
  microphone?: false | Qwen3AsrMicrophoneOptions;
  /** System/output audio options. `false` disables this track. Default enabled. */
  systemAudio?: false | Qwen3AsrSystemAudioOptions;
  /** Called for every rolling revision, tagged with its audio source. */
  onResult: (event: Qwen3AsrMeetingResultEvent) => void;
  /** Called for asynchronous capture or model-worker errors. */
  onError?: (event: Qwen3AsrMeetingErrorEvent) => void;
}

export interface Qwen3AsrMeetingFinal {
  microphone?: Qwen3AsrRealtimeFinal;
  systemAudio?: Qwen3AsrRealtimeFinal;
}

/**
 * A meeting owns one independently clocked transcription track per enabled
 * source. Results stay source-tagged so speaker-mic audio and remote/system
 * audio are never silently mixed or ordered by unrelated device clocks.
 */
export class Qwen3AsrMeetingSession {
  readonly microphone?: Qwen3AsrRealtimeSession;
  readonly systemAudio?: Qwen3AsrRealtimeSession;

  #stopPromise: Promise<Qwen3AsrMeetingFinal> | undefined;

  private constructor(tracks: { microphone?: Qwen3AsrRealtimeSession; systemAudio?: Qwen3AsrRealtimeSession }) {
    this.microphone = tracks.microphone;
    this.systemAudio = tracks.systemAudio;
  }

  static async start(model: Qwen3AsrModel, options: Qwen3AsrMeetingOptions): Promise<Qwen3AsrMeetingSession> {
    if (options.microphone === false && options.systemAudio === false) {
      throw new Error('At least one meeting audio source must be enabled');
    }

    const tracks: {
      microphone?: Qwen3AsrRealtimeSession;
      systemAudio?: Qwen3AsrRealtimeSession;
    } = {};
    const started: Qwen3AsrRealtimeSession[] = [];

    const startTrack = async (
      source: Qwen3AsrMeetingSource,
      capture: Qwen3AsrCaptureOptions,
    ): Promise<Qwen3AsrRealtimeSession> => {
      const session = await Qwen3AsrRealtimeSession.start(model, {
        stream: options.stream,
        capture,
        onResult: (result) => options.onResult({ source, result }),
        onError: (error) => options.onError?.({ source, error }),
      });
      started.push(session);
      return session;
    };

    try {
      // Ask for system-audio permission before opening the microphone. This
      // avoids leaving a live mic running while the system permission sheet is
      // waiting for a response.
      if (options.systemAudio !== false) {
        tracks.systemAudio = await startTrack('systemAudio', {
          ...options.systemAudio,
          source: 'systemAudio' as Qwen3AsrCaptureSource,
        });
      }
      if (options.microphone !== false) {
        tracks.microphone = await startTrack('microphone', {
          ...options.microphone,
          source: 'microphone' as Qwen3AsrCaptureSource,
        });
      }
    } catch (error) {
      await Promise.allSettled(started.map((session) => session.stop()));
      throw error;
    }

    return new Qwen3AsrMeetingSession(tracks);
  }

  pause(): void {
    this.forEachTrack((track) => track.pause());
  }

  resume(): void {
    this.forEachTrack((track) => track.resume());
  }

  stop(): Promise<Qwen3AsrMeetingFinal> {
    this.#stopPromise ??= (async () => {
      const microphone = this.microphone?.stop();
      const systemAudio = this.systemAudio?.stop();
      const settled = await Promise.allSettled(
        [microphone, systemAudio].filter((promise): promise is Promise<Qwen3AsrRealtimeFinal> => promise !== undefined),
      );
      const failure = settled.find((result): result is PromiseRejectedResult => result.status === 'rejected');
      if (failure) throw failure.reason;

      const final: Qwen3AsrMeetingFinal = {};
      let index = 0;
      if (microphone) final.microphone = (settled[index++] as PromiseFulfilledResult<Qwen3AsrRealtimeFinal>).value;
      if (systemAudio) final.systemAudio = (settled[index] as PromiseFulfilledResult<Qwen3AsrRealtimeFinal>).value;
      return final;
    })();
    return this.#stopPromise;
  }

  private forEachTrack(action: (track: Qwen3AsrRealtimeSession) => void): void {
    let firstError: unknown;
    for (const track of [this.microphone, this.systemAudio]) {
      if (!track) continue;
      try {
        action(track);
      } catch (error) {
        firstError ??= error;
      }
    }
    if (firstError) throw firstError;
  }
}

/** Start microphone and system-audio transcription as one meeting session. */
export function startMeetingTranscription(
  model: Qwen3AsrModel,
  options: Qwen3AsrMeetingOptions,
): Promise<Qwen3AsrMeetingSession> {
  return Qwen3AsrMeetingSession.start(model, options);
}
