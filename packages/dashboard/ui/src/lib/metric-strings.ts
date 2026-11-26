/**
 * Every metric word of the app and its one-sentence definition. A surface imports a name from here and never types
 * the word itself. `{x}` is filled by `fill` in `chat-strings.ts`.
 */

export type MetricWord = {
  readonly label: string;
  readonly unit?: string;
  readonly tip: string;
};

/** Seconds of live data behind `Output rate` and `Prefill speed`. */
export const RATE_WINDOW_SECONDS = 10;

/** Seconds an idle session row stays in the live lists without a new call. */
export const LANE_HOLD_SECONDS = 10;

export const METRIC = {
  outputRate: {
    label: 'Output rate',
    unit: 'tok/s',
    tip: 'Tokens out per second while any lane is decoding, all lanes together. Last 10 s.',
  },
  decodeSpeed: {
    label: 'Decode speed',
    unit: 'tok/s',
    tip: 'Output tokens per second of decode time. The first token is not counted.',
  },
  prefillSpeed: {
    label: 'Prefill speed',
    unit: 'tok/s',
    tip: 'New prompt tokens per second of prefill. Cached tokens and prefills under 256 tokens are not counted.',
  },
  firstToken: {
    label: 'First token',
    tip: 'Start of the request to its first token: waiting, preparing and prefill.',
  },
  laneSpeed: {
    label: 'Speed',
    unit: 'tok/s',
    tip: 'This lane: prefill speed while reading, decode speed while writing.',
  },
  prefillTime: { label: 'Prefill time', tip: 'Time spent reading the prompt.' },
  decodeTime: { label: 'Decode time', tip: 'Time spent writing the answer after the first token.' },
  queueWait: { label: 'Queue wait', tip: 'Time the server held the request before the engine took it.' },
  modelBusy: { label: 'Model busy', tip: 'The model was working on something else.' },
  laneWait: { label: 'Lane wait', tip: 'No free lane or not enough memory yet.' },
  preparing: { label: 'Preparing', tip: 'Building the prompt and restoring cached tokens.' },
  running: { label: 'Running', tip: 'A turn that does not stream: its progress is not reported until it ends.' },
  idle: {
    label: 'Idle',
    tip: `The session's last call ended. An agent calls again after its tool runs, else the row leaves in ${String(LANE_HOLD_SECONDS)} s.`,
  },
  timeByPhase: {
    label: 'Time by phase',
    tip: 'Mean lanes in each state. One lane busy the whole time counts as 1.',
  },
  inFlight: { label: 'In flight', tip: 'Requests the engine has taken and not finished.' },
  queued: { label: 'Queued', tip: 'Requests the server holds before the engine takes them.' },
  waiting: {
    label: 'Waiting',
    tip: 'Taken by the engine, not started: the model is busy or no lane is free.',
  },
} as const satisfies Readonly<Record<string, MetricWord>>;

/** The four colored bands of a time-by-phase chart, bottom to top. */
export const BANDS = {
  decode: 'Decode',
  prefill: 'Prefill',
  other: 'Other',
  waiting: 'Waiting',
} as const;

/** An idle lane's row text: what its last call did. */
export const IDLE_LANE = {
  lastCallOut: 'last call · {tokens} out',
  lastCallEnded: 'last call ended',
} as const;

/** What `Waiting` is waiting for, in the lane's row text. */
export const WAIT_REASON = {
  modelBusy: 'model busy',
  noFreeLane: 'no free lane',
} as const;

/** The state badge of the engine, from the newest snapshot. */
export const ENGINE_STATE = {
  unresponsive: 'Not responding',
  loading: 'Loading',
  decode: 'Decode',
  prefill: 'Prefill',
  prefillDecode: 'Prefill + decode',
  working: 'Working',
  waiting: 'Waiting',
  queued: 'Queued',
  idle: 'Idle',
} as const;

/** Lane counts beside the state badge. */
export const LANES = {
  one: '1 lane',
  many: '{n} lanes',
  decoding: '{n} decode',
  prefilling: '{n} prefill',
  waiting: '{n} waiting',
  other: '{n} other',
  /** Name and unit of the stacked lanes under the live line. */
  byPhase: 'Lanes by phase',
  unit: 'lanes',
} as const;

/** Fact labels in the hover panels of the Engine "Now" ticker. */
export const NOW_FACTS = {
  lanesDecoding: 'Lanes decoding',
  lanesInPrefill: 'Lanes in prefill',
  window: 'Window',
  cachedTokens: 'Cached tokens',
  notCounted: 'not counted',
} as const;

/** Prefill progress words. */
export const PREFILL_PROGRESS = {
  /** `done` of `total` computed tokens. */
  tokensOf: '{done} of {total} tokens',
  /** One-shot prefill: no count yet, only the total and the time so far. */
  tokensElapsed: '{total} tokens · {elapsed}',
  cached: '{n} cached',
  restoring: 'restoring {n} cached tokens',
} as const;

/** Window wording and the empty value. */
export const WINDOW = {
  last10s: 'last 10 s',
  none: '—',
} as const;

/** The Overview engine panel: the words after its two speeds and the name of its activity strip. */
export const ENGINE_PANEL = {
  /** After the Output rate: "64.2 tok/s out". */
  out: 'out',
  /** After the Prefill speed: "1,135 tok/s prefill". */
  prefill: 'prefill',
  /** `{shares}`: each band's lane time over the observed part of the last 2 minutes, then idle. */
  activity: 'Engine activity, last 2 minutes: {shares}',
  activityShare: '{band} {pct}',
  idle: 'idle',
  activityNone: 'Engine activity, last 2 minutes: nothing recorded yet',
} as const;
