/**
 * Native half of the family registry: one loader binding per
 * `MODEL_FAMILY_DATA` row, plus `detectModelType` (filesystem + GGUF).
 */

import { readFile } from 'node:fs/promises';
import { dirname, extname, join } from 'node:path';

import {
  Gemma4Model as NativeGemma4Model,
  ggufArchitecture,
  HarrierModel,
  Lfm2Model as NativeLfm2Model,
  MuseGlimmerModel as NativeMuseGlimmerModel,
  NemotronHModel as NativeNemotronHModel,
  QianfanOCRModel,
  Qwen3Model as NativeQwen3Model,
  Qwen35Model as NativeQwen35Model,
  Qwen35MoeModel as NativeQwen35MoeModel,
} from '@mlx-node/core';

import { ChatSession, type SessionCapableModel } from '../chat-session.js';
import {
  familyDataFor,
  MalformedModelConfigError,
  matchFamily,
  MODEL_FAMILY_DATA,
  UnsupportedModelTypeError,
  type ModelType,
  type TrainableFamilyId,
} from '../family-data.js';
import {
  Gemma4Model,
  Lfm2Model,
  MuseGlimmerModel,
  NemotronHModel,
  Qwen3Model,
  Qwen35Model,
  Qwen35MoeModel,
} from '../stream.js';

/** Optional settings for {@link loadModel} / {@link loadSession}. */
export interface LoadModelOptions {
  /**
   * Directory of an external draft checkpoint (config.json +
   * model.safetensors) loaded alongside the target for speculative decoding.
   * Gemma4 accepts either a DSpark draft or a Google gemma-4 assistant draft
   * (`google/gemma-4-*-it-assistant`); the variant is auto-detected from
   * the draft's config.json (`model_type` `gemma4_assistant` /
   * `gemma4_unified_assistant` → assistant, `architectures` containing
   * `Gemma4DSparkModel` → DSpark). When omitted, Gemma4 automatically loads
   * an embedded draft from `<modelPath>/draft/` when present. Draft decoding
   * runs on the flat KV-cache path, so the target checkpoint must not
   * explicitly enable `use_block_paged_cache`.
   *
   * Dense `qwen3_5` accepts a z-lab `DFlash2DraftModel` companion such as
   * `z-lab/Qwen3.8-27B-DFlash2`. It shares the Qwen3.8 target embedding and
   * LM head, validates all companion tensors at load time, and takes
   * precedence over an inline target MTP head. Other model families reject
   * this option.
   */
  draftModelPath?: string;
}

type NativeModelClass = abstract new (...args: never[]) => object;

interface LoaderBinding {
  readonly load: (modelPath: string, options?: LoadModelOptions) => Promise<unknown>;
  /**
   * Native `@mlx-node/core` class behind this family. The public
   * `LoadableModel` / `TrainableModel` unions derive from these classes —
   * NOT from the streaming-wrapper types the loaders return — so native
   * instances stay assignable and trainers can pass them directly to the
   * Rust engine factory methods without type conflicts. Loaded wrapper
   * instances are runtime subclasses of their native class, so
   * `instanceof` narrowing against these classes still works on
   * `loadModel` results.
   */
  readonly nativeModelClass: NativeModelClass;
}

/**
 * Native half of the family registry: one loader + native class per
 * `MODEL_FAMILY_DATA` row (the native-free half in `../family-data.ts`).
 * `satisfies Record<ModelType, LoaderBinding>` makes the zip exhaustive both
 * ways — a data row without a binding, or a binding without a row, fails to
 * compile.
 */
const LOADER_BINDINGS = {
  gemma4: {
    load: (modelPath: string, options?: LoadModelOptions) =>
      Gemma4Model.load(
        modelPath,
        options?.draftModelPath === undefined ? null : { draftModelPath: options.draftModelPath },
      ),
    nativeModelClass: NativeGemma4Model,
  },
  muse_glimmer: {
    load: (modelPath: string) => MuseGlimmerModel.load(modelPath),
    nativeModelClass: NativeMuseGlimmerModel,
  },
  harrier: {
    load: (modelPath: string) => HarrierModel.load(modelPath),
    nativeModelClass: HarrierModel,
  },
  qwen3: {
    load: (modelPath: string) => Qwen3Model.load(modelPath),
    nativeModelClass: NativeQwen3Model,
  },
  qwen3_5: {
    load: (modelPath: string, options?: LoadModelOptions) =>
      Qwen35Model.load(
        modelPath,
        options?.draftModelPath === undefined ? null : { draftModelPath: options.draftModelPath },
      ),
    nativeModelClass: NativeQwen35Model,
  },
  qwen3_5_moe: {
    load: (modelPath: string) => Qwen35MoeModel.load(modelPath),
    nativeModelClass: NativeQwen35MoeModel,
  },
  lfm2: {
    load: (modelPath: string) => Lfm2Model.load(modelPath),
    nativeModelClass: NativeLfm2Model,
  },
  lfm2_moe: {
    load: (modelPath: string) => Lfm2Model.load(modelPath),
    nativeModelClass: NativeLfm2Model,
  },
  nemotron_h: {
    load: (modelPath: string) => NemotronHModel.load(modelPath),
    nativeModelClass: NativeNemotronHModel,
  },
  internvl_chat: {
    load: (modelPath: string) => QianfanOCRModel.load(modelPath),
    nativeModelClass: QianfanOCRModel,
  },
  'qianfan-ocr': {
    load: (modelPath: string) => QianfanOCRModel.load(modelPath),
    nativeModelClass: QianfanOCRModel,
  },
} as const satisfies Record<ModelType, LoaderBinding>;

type LoaderBindings = typeof LOADER_BINDINGS;

export type { ModelType };

/**
 * Union of the native `@mlx-node/core` model classes across every registered
 * family — the public contract of {@link loadModel}. At runtime the chat
 * families resolve to streaming-wrapper subclasses of these classes
 * (AsyncGenerator `chatStream*` overrides), but the public type names the
 * native classes so downstream code can pass instances directly to Rust
 * engine factory methods without type conflicts.
 */
export type LoadableModel = InstanceType<LoaderBindings[ModelType]['nativeModelClass']>;

/**
 * Union accepted by trainer APIs: registered wrapper results plus their native
 * FFI instances. Both sides derive from the same trainable family ids.
 */
export type TrainableModel =
  | Awaited<ReturnType<LoaderBindings[TrainableFamilyId]['load']>>
  | InstanceType<LoaderBindings[TrainableFamilyId]['nativeModelClass']>;

// Only families whose native `load(path)` accepts a GGUF file carry a
// `ggufArchitectures` row entry (see family-data.ts).
const GGUF_ARCHITECTURE_MODEL_TYPES = new Map<string, ModelType>(
  MODEL_FAMILY_DATA.flatMap((row) =>
    'ggufArchitectures' in row ? row.ggufArchitectures.map((architecture) => [architecture, row.id] as const) : [],
  ),
);

function requireFamilyData(modelType: ModelType) {
  const family = familyDataFor(modelType);
  if (family === undefined) {
    throw new Error(`Internal error: missing model family descriptor for "${modelType}"`);
  }
  return family;
}

/**
 * Dispatch a load through the registry, validating draft-capable families.
 * `draftModelPath` reaches only gemma4 and dense qwen3_5; every other family
 * rejects it loudly instead of silently ignoring speculative-decode intent.
 */
function dispatchLoad(
  modelType: ModelType,
  modelPath: string,
  options: LoadModelOptions | undefined,
): Promise<unknown> {
  if (options?.draftModelPath !== undefined && requireFamilyData(modelType).acceptsDraftModel !== true) {
    throw new Error(
      `draftModelPath (speculative-decoding draft) is only supported by gemma4 and qwen3_5 models; ` +
        `${modelPath} has model_type "${modelType}"`,
    );
  }
  const binding: LoaderBinding = LOADER_BINDINGS[modelType];
  return binding.load(modelPath, options);
}

/**
 * Load a model from disk, auto-detecting architecture from config.json.
 *
 * Supports both language models (Qwen3, Qwen3.5) and vision-language models
 * (Qianfan-OCR / InternVL). Use `instanceof` to narrow the returned type.
 *
 * `options.draftModelPath` attaches an external draft checkpoint for
 * speculative decoding — gemma4 and dense qwen3_5 only; every other family
 * rejects it.
 * Without the option, Gemma4 loads `<modelPath>/draft/` automatically when
 * that embedded checkpoint is present.
 */
export async function loadModel(modelPath: string, options?: LoadModelOptions): Promise<LoadableModel> {
  const modelType = await detectModelType(modelPath);
  return dispatchLoad(modelType, modelPath, options) as Promise<LoadableModel>;
}

/**
 * Load a model and wrap it in a {@link ChatSession} for multi-turn chat.
 *
 * Convenience around `loadModel()` + `new ChatSession(model)` for the
 * common case where a caller just wants an ergonomic session handle.
 *
 * Rejects models that cannot be driven by a `ChatSession`:
 *   - Embedding models (`HarrierModel`) have no chat surface.
 *   - The native `QianfanOCRModel` exposes callback-based streaming
 *     methods that do not structurally satisfy `SessionCapableModel`'s
 *     `AsyncGenerator` overloads. The VLM AsyncGenerator wrapper lives
 *     in `@mlx-node/vlm` (importing it here would create a circular
 *     package dependency), so callers who want a Qianfan-OCR session
 *     must import `QianfanOCRModel` from `@mlx-node/vlm` and construct
 *     `new ChatSession(model)` directly.
 *
 * `options.draftModelPath` attaches an external draft checkpoint for
 * speculative decoding — gemma4 and dense qwen3_5 only; every other family
 * rejects it.
 * Without the option, Gemma4 loads `<modelPath>/draft/` automatically when
 * that embedded checkpoint is present.
 * The resulting session auto-enables the speculative path when the model
 * reports `hasMtpWeights()` AND does not opt out of the auto-default; pass
 * `enableMtp: false` per call to suppress it, or `enableMtp: true` to force
 * it on a family that opts out. NemotronH opts out (`mtpAutoEnabled()`
 * returns false) because forcing MTP moves the turn into the exclusive lane
 * and out of continuous batching; see `ChatSession.mtpAutoDefaultAllowed`.
 */
export async function loadSession(
  modelPath: string,
  options?: LoadModelOptions,
): Promise<ChatSession<SessionCapableModel>> {
  const modelType = await detectModelType(modelPath);
  const kind = requireFamilyData(modelType).kind;
  if (kind === 'embedding') {
    throw new Error('loadSession: embedding models (Harrier) cannot be wrapped in a ChatSession');
  }
  if (kind === 'vlm') {
    throw new Error(
      'loadSession: Qianfan-OCR / InternVL session support lives in @mlx-node/vlm. Import QianfanOCRModel from @mlx-node/vlm and construct ChatSession(model) directly.',
    );
  }
  const m = await dispatchLoad(modelType, modelPath, options);
  return new ChatSession(m as unknown as SessionCapableModel);
}

export async function detectModelType(modelPath: string): Promise<ModelType> {
  const isGguf = extname(modelPath).toLowerCase() === '.gguf';
  const configPath = isGguf ? join(dirname(modelPath), 'config.json') : join(modelPath, 'config.json');
  let raw: string;
  try {
    raw = await readFile(configPath, 'utf-8');
  } catch (e) {
    if (isGguf && typeof e === 'object' && e !== null && 'code' in e && e.code === 'ENOENT') {
      const architecture = ggufArchitecture(modelPath);
      const modelType = GGUF_ARCHITECTURE_MODEL_TYPES.get(architecture);
      if (modelType === undefined) {
        throw new Error(`Unsupported GGUF architecture "${architecture}" in ${modelPath}`);
      }
      return modelType;
    }
    throw new Error(`Cannot detect model type: config.json not found in ${modelPath}`);
  }

  try {
    return matchFamily(modelPath, JSON.parse(raw));
  } catch (e) {
    if (e instanceof UnsupportedModelTypeError || e instanceof MalformedModelConfigError) throw e;
    throw new Error(`Cannot detect model type: config.json not found in ${modelPath}`);
  }
}
