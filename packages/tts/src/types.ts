export interface AudioChunk {
  samples: Float32Array;
  sampleRate: number;
  channels: number;
  /** Frame offset across the entire output, independent of segment boundaries. */
  startSample: number;
  segmentIndex: number;
}
export interface AudioData {
  samples: Float32Array;
  sampleRate: number;
  channels?: number;
}
export type TtsVoiceMode = 'preset' | 'reference' | 'description';
export interface TtsConditioningCapability {
  voice: TtsVoiceMode;
  instruct: 'supported' | 'experimental' | 'unsupported';
}
export interface TtsCapabilities {
  family: string;
  variant: string;
  sampleRate: number;
  channels: number;
  voices: readonly string[];
  languages: readonly string[];
  voiceCloning: boolean;
  conditioning: readonly TtsConditioningCapability[];
  textStreaming: 'segmented';
  audioStreaming: boolean;
}
export type TtsInputEvent =
  | string
  | { type: 'text'; text: string }
  | { type: 'flush' }
  | { type: 'instruct'; value: string | null };
export type TtsInput = string | AsyncIterable<TtsInputEvent>;
export type TtsVoice = string | PreparedVoice | { type: 'description'; description: string };
export interface PreparedVoice {
  readonly id: string;
  /**
   * Release this voice. Repeated disposal is safe. Releases may be requested
   * while the model is busy; they are queued behind the active operation.
   */
  dispose(): Promise<void>;
}
export interface TtsLoadOptions {
  instructionCache?: {
    /** Experimental, default false: split-prefix rounding can affect speech content. */
    enabled?: boolean;
    maxBytes?: number;
    maxEntries?: number;
  };
}
export interface TtsOptions {
  voice: TtsVoice;
  /** Natural-language delivery instructions; support depends on the voice mode. */
  instruct?: string;
  language?: string;
  signal?: AbortSignal;
  seed?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  /** Per-segment generation limit in seconds (default 120); truncation reports `finishReason: 'length'`. */
  maxDurationSeconds?: number;
  chunkDurationMs?: number;
  audioBufferSeconds?: number;
  maxSegmentGraphemes?: number;
  queuedSegments?: number;
}
export interface TtsStats {
  audioSeconds: number;
  synthesisMs: number;
  wallMs: number;
  /** Time waiting for the next committed text segment. */
  inputWaitMs: number;
  /** Time the iterator was suspended while the consumer handled PCM. */
  consumerWaitMs: number;
  /** Synthesis seconds / generated audio seconds; below one is faster than realtime. */
  realTimeFactor: number | null;
  firstPcmMs: number | null;
  segments: number;
  finishReason: 'eos' | 'length';
}
export interface TtsStream extends AsyncIterable<AudioChunk> {
  readonly completed: Promise<TtsStats>;
  cancel(): void;
}
export interface TtsAudio extends AudioData {
  channels: number;
  stats: TtsStats;
}
export interface TtsModel {
  readonly capabilities: TtsCapabilities;
  prepareVoice(input: { audio: AudioData; transcript: string }): Promise<PreparedVoice>;
  synthesize(input: TtsInput, options: TtsOptions): Promise<TtsAudio>;
  synthesizeStream(input: TtsInput, options: TtsOptions): TtsStream;
  dispose(): Promise<void>;
}
