/**
 * Model loading utilities for Qwen3 models
 *
 * Handles loading pretrained weights from MLX format or converting from HuggingFace.
 */

import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import {
  Gemma4Model as NativeGemma4Model,
  HarrierModel,
  Lfm2Model as NativeLfm2Model,
  QianfanOCRModel,
  Qwen3Model as NativeQwen3Model,
  Qwen35Model as NativeQwen35Model,
  Qwen35MoeModel as NativeQwen35MoeModel,
} from '@mlx-node/core';

import { ChatSession, type SessionCapableModel } from '../chat-session.js';
import { Gemma4Model, Lfm2Model, Qwen3Model, Qwen35Model, Qwen35MoeModel } from '../stream.js';

/** Optional settings for {@link loadModel} / {@link loadSession}. */
export interface LoadModelOptions {
  /**
   * Gemma4 only: directory of an external draft checkpoint (config.json +
   * model.safetensors) loaded alongside the target model for speculative
   * decoding (forwarded as `Gemma4LoadOptions.draftModelPath`). Accepts
   * either a DSpark draft or a Google gemma-4 assistant draft
   * (`google/gemma-4-*-it-assistant`); the variant is auto-detected from
   * the draft's config.json (`model_type` `gemma4_assistant` /
   * `gemma4_unified_assistant` → assistant, `architectures` containing
   * `Gemma4DSparkModel` → DSpark). Draft decoding runs on the flat
   * KV-cache path, so the target checkpoint must not explicitly enable
   * `use_block_paged_cache`. Setting this for any other model family is a
   * hard error — no other loader accepts a draft model.
   */
  draftModelPath?: string;
}

type ModelKind = 'trainable' | 'loadable' | 'embedding' | 'vlm';

interface NormalizedModelConfig {
  readonly usesDefaultModelType: boolean;
  readonly rawModelType: string | undefined;
  readonly rawModelTypeLabel: string;
  readonly architectures: ReadonlySet<string>;
}

interface ModelConfigMatchContext extends NormalizedModelConfig {
  readonly modelType: string | undefined;
}

interface ModelConfigMatcher {
  /** Exact raw `config.json` model_type values owned by this family. */
  readonly rawModelTypes: readonly string[];
  /** Optional higher-priority architecture probe for shared or absent model_type values. */
  readonly architectureProbe?: (config: ModelConfigMatchContext) => boolean;
}

type NativeModelClass = abstract new (...args: never[]) => object;

interface ModelFamilyDescriptor {
  readonly modelType: string;
  readonly kind: ModelKind;
  readonly match: ModelConfigMatcher;
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
  readonly acceptsDraftModel?: true;
  /** Backward-compatible fallback when config.json omits model_type or sets it to null. */
  readonly defaultForNullishModelType?: true;
}

/**
 * Ordered source of truth for every supported model family. Each entry owns
 * its canonical `ModelType`, raw config aliases / architecture probes, loader,
 * and `ChatSession` eligibility:
 *
 *   - `'trainable'` — GRPO/SFT-capable LM (Qwen3 family); chat-capable.
 *   - `'loadable'`  — chat-capable LM with no trainer engine (Gemma4, LFM2).
 *   - `'embedding'` — no chat surface (Harrier); rejected by `loadSession`.
 *   - `'vlm'`       — VLM whose AsyncGenerator wrapper lives in
 *                     `@mlx-node/vlm` (importing it here would create a
 *                     circular package dependency), so `loadSession`
 *                     rejects it and routes callers to `@mlx-node/vlm`.
 *
 * A base family is selected from an explicit alias or the single declarative
 * nullish-model_type default, then architecture probes refine it in declaration
 * order. Gemma's unified architecture is authoritative (matching the native
 * loader); Harrier refines a Qwen3 base. Adding a family means adding one
 * descriptor here, without a second normalization or dispatch branch.
 */
const MODEL_FAMILY_REGISTRY = [
  {
    modelType: 'gemma4',
    kind: 'loadable',
    match: {
      rawModelTypes: ['gemma4', 'gemma4_text', 'gemma4_unified'],
      architectureProbe: ({ architectures }) => architectures.has('Gemma4UnifiedForConditionalGeneration'),
    },
    load: (modelPath: string, options?: LoadModelOptions) =>
      Gemma4Model.load(
        modelPath,
        options?.draftModelPath === undefined ? null : { draftModelPath: options.draftModelPath },
      ),
    nativeModelClass: NativeGemma4Model,
    acceptsDraftModel: true,
  },
  {
    modelType: 'harrier',
    kind: 'embedding',
    match: {
      rawModelTypes: ['harrier'],
      architectureProbe: ({ modelType, architectures }) =>
        modelType === 'qwen3' && architectures.has('Qwen3Model') && !architectures.has('Qwen3ForCausalLM'),
    },
    load: (modelPath: string) => HarrierModel.load(modelPath),
    nativeModelClass: HarrierModel,
  },
  {
    modelType: 'qwen3',
    kind: 'trainable',
    match: { rawModelTypes: ['qwen3'] },
    load: (modelPath: string) => Qwen3Model.load(modelPath),
    nativeModelClass: NativeQwen3Model,
    defaultForNullishModelType: true,
  },
  {
    modelType: 'qwen3_5',
    kind: 'trainable',
    match: { rawModelTypes: ['qwen3_5'] },
    load: (modelPath: string) => Qwen35Model.load(modelPath),
    nativeModelClass: NativeQwen35Model,
  },
  {
    modelType: 'qwen3_5_moe',
    kind: 'trainable',
    match: { rawModelTypes: ['qwen3_5_moe'] },
    load: (modelPath: string) => Qwen35MoeModel.load(modelPath),
    nativeModelClass: NativeQwen35MoeModel,
  },
  {
    modelType: 'lfm2',
    kind: 'loadable',
    match: { rawModelTypes: ['lfm2'] },
    load: (modelPath: string) => Lfm2Model.load(modelPath),
    nativeModelClass: NativeLfm2Model,
  },
  {
    modelType: 'lfm2_moe',
    kind: 'loadable',
    match: { rawModelTypes: ['lfm2_moe'] },
    load: (modelPath: string) => Lfm2Model.load(modelPath),
    nativeModelClass: NativeLfm2Model,
  },
  {
    modelType: 'internvl_chat',
    kind: 'vlm',
    match: { rawModelTypes: ['internvl_chat'] },
    load: (modelPath: string) => QianfanOCRModel.load(modelPath),
    nativeModelClass: QianfanOCRModel,
  },
  {
    modelType: 'qianfan-ocr',
    kind: 'vlm',
    match: { rawModelTypes: ['qianfan-ocr'] },
    load: (modelPath: string) => QianfanOCRModel.load(modelPath),
    nativeModelClass: QianfanOCRModel,
  },
] as const satisfies readonly ModelFamilyDescriptor[];

export type ModelType = (typeof MODEL_FAMILY_REGISTRY)[number]['modelType'];

type RegisteredModelFamily = (typeof MODEL_FAMILY_REGISTRY)[number];
type RegisteredTrainableFamily = Extract<RegisteredModelFamily, { readonly kind: 'trainable' }>;

/**
 * Union of the native `@mlx-node/core` model classes across every registered
 * family — the public contract of {@link loadModel}. At runtime the chat
 * families resolve to streaming-wrapper subclasses of these classes
 * (AsyncGenerator `chatStream*` overrides), but the public type names the
 * native classes so downstream code can pass instances directly to Rust
 * engine factory methods without type conflicts.
 */
export type LoadableModel = InstanceType<RegisteredModelFamily['nativeModelClass']>;

/**
 * Union accepted by trainer APIs: registered wrapper results plus their native
 * FFI instances. Both sides derive from the same trainable registry rows.
 */
export type TrainableModel =
  | Awaited<ReturnType<RegisteredTrainableFamily['load']>>
  | InstanceType<RegisteredTrainableFamily['nativeModelClass']>;

interface ModelFamilyIndex<Family extends ModelFamilyDescriptor> {
  readonly byModelType: ReadonlyMap<string, Family>;
  readonly byRawModelType: ReadonlyMap<string, Family>;
  readonly defaultForNullishModelType: Family;
}

function buildModelFamilyIndex<const Family extends ModelFamilyDescriptor>(
  registry: readonly Family[],
): ModelFamilyIndex<Family> {
  const byModelType = new Map<string, Family>();
  const byRawModelType = new Map<string, Family>();
  let defaultForNullishModelType: Family | undefined;

  for (const family of registry) {
    const previousFamily = byModelType.get(family.modelType);
    if (previousFamily !== undefined) {
      throw new Error(`Duplicate canonical model type "${family.modelType}" in model family registry`);
    }
    byModelType.set(family.modelType, family);

    for (const rawModelType of family.match.rawModelTypes) {
      const previous = byRawModelType.get(rawModelType);
      if (previous !== undefined) {
        throw new Error(
          `Duplicate model_type alias "${rawModelType}" for "${previous.modelType}" and "${family.modelType}"`,
        );
      }
      byRawModelType.set(rawModelType, family);
    }

    if (family.defaultForNullishModelType === true) {
      if (defaultForNullishModelType !== undefined) {
        throw new Error(
          `Duplicate nullish-model_type defaults for "${defaultForNullishModelType.modelType}" and "${family.modelType}"`,
        );
      }
      defaultForNullishModelType = family;
    }
  }

  if (defaultForNullishModelType === undefined) {
    throw new Error('Model family registry must declare exactly one nullish-model_type default');
  }
  return { byModelType, byRawModelType, defaultForNullishModelType };
}

const MODEL_FAMILY_INDEX = buildModelFamilyIndex(MODEL_FAMILY_REGISTRY);

function findFamily(modelType: ModelType): ModelFamilyDescriptor {
  const family = MODEL_FAMILY_INDEX.byModelType.get(modelType);
  if (family === undefined) {
    throw new Error(`Internal error: missing model family descriptor for "${modelType}"`);
  }
  return family;
}

function matchesArchitectureProbe(family: ModelFamilyDescriptor, config: ModelConfigMatchContext): boolean {
  return family.match.architectureProbe?.(config) === true;
}

class MalformedModelConfigError extends Error {
  constructor(modelPath: string, reason: string) {
    super(`Malformed config.json in ${modelPath}: ${reason}`);
    this.name = 'MalformedModelConfigError';
  }
}

/**
 * Fail-closed validation: a config.json whose root is not a plain object,
 * or whose `architectures` is neither an array nor a string, is rejected
 * instead of coerced (coercion would fall through to the qwen3
 * nullish-model_type default and silently misroute the checkpoint).
 * Blessed lenient shapes stay accepted: `{}` root (qwen3 default),
 * missing/`null` `architectures` (empty set), bare-string `architectures`
 * (single-element set), and non-string array entries (filtered out).
 */
function normalizeConfig(modelPath: string, config: unknown): NormalizedModelConfig {
  if (typeof config !== 'object' || config === null || Array.isArray(config)) {
    throw new MalformedModelConfigError(modelPath, 'root must be a JSON object');
  }
  const object = config as Record<string, unknown>;
  const hasModelType = Object.hasOwn(object, 'model_type');
  const rawModelTypeValue = hasModelType ? object.model_type : undefined;
  const usesDefaultModelType = !hasModelType || rawModelTypeValue === null;
  const rawModelType = typeof rawModelTypeValue === 'string' ? rawModelTypeValue : undefined;
  const rawModelTypeLabel = hasModelType ? String(rawModelTypeValue) : '<missing>';
  const rawArchitectures = 'architectures' in object ? object.architectures : undefined;
  if (
    rawArchitectures !== undefined &&
    rawArchitectures !== null &&
    !Array.isArray(rawArchitectures) &&
    typeof rawArchitectures !== 'string'
  ) {
    throw new MalformedModelConfigError(modelPath, '"architectures" must be an array or a string');
  }
  const architectures = Array.isArray(rawArchitectures)
    ? rawArchitectures.filter((architecture): architecture is string => typeof architecture === 'string')
    : typeof rawArchitectures === 'string'
      ? [rawArchitectures]
      : [];

  return { usesDefaultModelType, rawModelType, rawModelTypeLabel, architectures: new Set(architectures) };
}

class UnsupportedModelTypeError extends Error {
  constructor(modelPath: string, rawModelTypeLabel: string) {
    super(`Unsupported model_type "${rawModelTypeLabel}" in ${modelPath}/config.json`);
    this.name = 'UnsupportedModelTypeError';
  }
}

/**
 * Dispatch a load through the registry, validating gemma4-only options.
 * `draftModelPath` reaches ONLY the gemma4 row; every other family rejects
 * it loudly instead of silently ignoring a caller's speculative-decode
 * intent.
 */
function dispatchLoad(
  modelType: ModelType,
  modelPath: string,
  options: LoadModelOptions | undefined,
): Promise<unknown> {
  const family = findFamily(modelType);
  if (options?.draftModelPath !== undefined && family.acceptsDraftModel !== true) {
    throw new Error(
      `draftModelPath (speculative-decoding draft) is only supported by gemma4 models; ` +
        `${modelPath} has model_type "${modelType}"`,
    );
  }
  return family.load(modelPath, options);
}

/**
 * Load a model from disk, auto-detecting architecture from config.json.
 *
 * Supports both language models (Qwen3, Qwen3.5) and vision-language models
 * (Qianfan-OCR / InternVL). Use `instanceof` to narrow the returned type.
 *
 * `options.draftModelPath` attaches an external draft checkpoint (DSpark or
 * Google gemma-4 assistant, auto-detected from the draft's config.json) for
 * speculative decoding — gemma4 only; any other detected family rejects it.
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
 * `options.draftModelPath` attaches an external draft checkpoint (DSpark or
 * Google gemma-4 assistant, auto-detected from the draft's config.json) for
 * speculative decoding — gemma4 only; any other detected family rejects it.
 * The resulting session auto-enables the speculative path (the model
 * reports `hasMtpWeights()`); pass `enableMtp: false` per call to opt out.
 */
export async function loadSession(
  modelPath: string,
  options?: LoadModelOptions,
): Promise<ChatSession<SessionCapableModel>> {
  const modelType = await detectModelType(modelPath);
  const kind = findFamily(modelType).kind;
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
  try {
    const raw = await readFile(join(modelPath, 'config.json'), 'utf-8');
    const config = normalizeConfig(modelPath, JSON.parse(raw));
    const baseFamily = config.usesDefaultModelType
      ? MODEL_FAMILY_INDEX.defaultForNullishModelType
      : config.rawModelType === undefined
        ? undefined
        : MODEL_FAMILY_INDEX.byRawModelType.get(config.rawModelType);
    const matchContext: ModelConfigMatchContext = { ...config, modelType: baseFamily?.modelType };
    const family =
      MODEL_FAMILY_REGISTRY.find((candidate) => matchesArchitectureProbe(candidate, matchContext)) ?? baseFamily;
    if (family === undefined) throw new UnsupportedModelTypeError(modelPath, config.rawModelTypeLabel);
    return family.modelType;
  } catch (e) {
    if (e instanceof UnsupportedModelTypeError || e instanceof MalformedModelConfigError) throw e;
    throw new Error(`Cannot detect model type: config.json not found in ${modelPath}`);
  }
}
