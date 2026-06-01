// Canonical data contract for the Education App "Training" playground (Chapter
// 13 — Training). Mirrors inspector-types.ts: the worker (`mlx-worker.ts`) and
// the main-thread bridge (`demo/lib/training-client.ts`) both read/write these
// `type` fields on `postMessage` payloads, so exporting them here keeps a
// single source of truth and the two ends can't drift on a typo.
//
// Unlike the inspector hooks, the Training playground does NOT need the big
// 1.6 GB Qwen model — it only needs the WASM runtime + WebGPU device up. It
// constructs its own tiny char-level transformer in the worker via the
// `TinyTrainer` NAPI class and trains it from scratch.

// -----------------------------------------------------------------------------
// Device-only init (bring the WASM runtime + WebGPU device up, skip the model).
// -----------------------------------------------------------------------------
//
// The normal `init` message fetches and loads the full model after the device
// is up. For the Training chapter we want the device without the model: the
// worker runs `handleInit` through the device-up milestone and then, when
// `deviceOnly === true`, posts `deviceReady` and returns BEFORE the model
// fetch. The full-model init path is unchanged.

export type DeviceReadyMessage = {
  type: 'deviceReady';
};

export const DEVICE_READY_TYPE = 'deviceReady' as const;

// -----------------------------------------------------------------------------
// Tiny trainer config (Qwen3-shaped). Field names are the camelCase JS view of
// the Rust `Qwen3Config` struct (`crates/mlx-core/src/models/qwen3/config.rs`).
// Every required field must be present; the worker passes this object straight
// into `new mlxExports.TinyTrainer(config, seed, learningRate)`.
// -----------------------------------------------------------------------------

export type TinyTrainerConfig = {
  vocabSize: number;
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  numKvHeads: number;
  intermediateSize: number;
  rmsNormEps: number;
  ropeTheta: number;
  maxPositionEmbeddings: number;
  headDim: number;
  useQkNorm: boolean;
  tieWordEmbeddings: boolean;
  padTokenId: number;
  eosTokenId: number;
  bosTokenId: number;
};

// -----------------------------------------------------------------------------
// Worker-bridge protocol. Every request carries a caller-supplied correlation
// id; the worker echoes it on the matching `*Result` / `*Error` reply.
// -----------------------------------------------------------------------------

export type TrainInitRequest = {
  type: 'trainInit';
  id: string;
  config: TinyTrainerConfig;
  seed: number;
  /** AdamW learning rate. Optional — the trainer defaults to ~3e-3. */
  learningRate?: number;
};

export type TrainStepRequest = {
  type: 'trainStep';
  id: string;
  /** Token ids for one teacher-forced step. Must contain >= 2 ids. */
  tokenIds: number[];
};

export type TrainSampleRequest = {
  type: 'trainSample';
  id: string;
  /** Non-empty prompt ids to continue. */
  promptIds: number[];
  /** Number of new tokens to generate. */
  n: number;
  /** Sampling temperature; <= 0 is greedy. */
  temperature: number;
};

export type TrainResetRequest = {
  type: 'trainReset';
  id: string;
  /** Seed for the re-initialized weights. */
  seed: number;
};

// Direct kernel-correctness probe for the WebGPU scatter_add op. The loss
// curve alone can't prove the gradient scatter accumulates colliding indices
// (a last-write-wins bug still trains, just wrong), so the playground runs
// this once and surfaces the invariant. The worker calls the top-level
// `webgpuScatterAddSelftest()` WASM export, which returns a length-5
// `number[]`: a correct accumulating kernel yields [0,0,0,3,0] (row 3 summed
// three colliding +1 updates); a broken last-write-wins kernel yields
// [0,0,0,1,0].
export type ScatterSelfTestRequest = {
  type: 'scatterSelfTest';
  id: string;
};

export type TrainingRequest =
  | TrainInitRequest
  | TrainStepRequest
  | TrainSampleRequest
  | TrainResetRequest
  | ScatterSelfTestRequest;

export type TrainInitResult =
  | { type: 'trainInitResult'; id: string }
  | { type: 'trainInitError'; id: string; error: string };

export type TrainStepResult =
  | { type: 'trainStepResult'; id: string; step: number; loss: number }
  | { type: 'trainStepError'; id: string; error: string };

export type TrainSampleResult =
  | { type: 'trainSampleResult'; id: string; ids: number[] }
  | { type: 'trainSampleError'; id: string; error: string };

export type TrainResetResult =
  | { type: 'trainResetResult'; id: string }
  | { type: 'trainResetError'; id: string; error: string };

export type ScatterSelfTestResult =
  | { type: 'scatterSelfTestResult'; id: string; result: number[] }
  | { type: 'scatterSelfTestError'; id: string; error: string };

// -----------------------------------------------------------------------------
// Protocol string constants (the runtime values both ends compare against).
// -----------------------------------------------------------------------------

export const TRAIN_INIT_REQUEST_TYPE = 'trainInit' as const;
export const TRAIN_INIT_RESULT_TYPE = 'trainInitResult' as const;
export const TRAIN_INIT_ERROR_TYPE = 'trainInitError' as const;

export const TRAIN_STEP_REQUEST_TYPE = 'trainStep' as const;
export const TRAIN_STEP_RESULT_TYPE = 'trainStepResult' as const;
export const TRAIN_STEP_ERROR_TYPE = 'trainStepError' as const;

export const TRAIN_SAMPLE_REQUEST_TYPE = 'trainSample' as const;
export const TRAIN_SAMPLE_RESULT_TYPE = 'trainSampleResult' as const;
export const TRAIN_SAMPLE_ERROR_TYPE = 'trainSampleError' as const;

export const TRAIN_RESET_REQUEST_TYPE = 'trainReset' as const;
export const TRAIN_RESET_RESULT_TYPE = 'trainResetResult' as const;
export const TRAIN_RESET_ERROR_TYPE = 'trainResetError' as const;

export const SCATTER_SELF_TEST_REQUEST_TYPE = 'scatterSelfTest' as const;
export const SCATTER_SELF_TEST_RESULT_TYPE = 'scatterSelfTestResult' as const;
export const SCATTER_SELF_TEST_ERROR_TYPE = 'scatterSelfTestError' as const;
