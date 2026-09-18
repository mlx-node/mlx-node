/**
 * Native-free per-family registration rows plus the pure `matchFamily`
 * detection they drive.
 *
 * This module must stay free of runtime imports (`import type` only): it is
 * re-exported through the native-free `@mlx-node/agent/catalog` subpath to the
 * dashboard viewer process, which must never dlopen the Metal addon.
 * `packages/agent/__test__/catalog-native-free.test.ts` gates that contract.
 */

import type { ChatConfig } from '@mlx-node/core';

export type ModelFamilyKind = 'trainable' | 'loadable' | 'embedding' | 'vlm';

export interface NormalizedModelConfig {
  readonly usesDefaultModelType: boolean;
  readonly rawModelType: string | undefined;
  readonly rawModelTypeLabel: string;
  readonly architectures: ReadonlySet<string>;
}

export interface ModelConfigMatchContext extends NormalizedModelConfig {
  readonly modelType: string | undefined;
}

export interface ModelConfigMatcher {
  /** Exact raw `config.json` model_type values owned by this family. */
  readonly rawModelTypes: readonly string[];
  /** Optional higher-priority architecture probe for shared or absent model_type values. */
  readonly architectureProbe?: (config: ModelConfigMatchContext) => boolean;
}

/**
 * Structural mirror of pi's `ProviderModelConfig['thinkingLevelMap']`. Declared
 * here because pi types are agent-only; `packages/agent/src/provider/models.ts`
 * pins assignability with a `satisfies` check at its use site.
 */
export interface FamilyThinkingLevelMap {
  readonly off?: string | null;
  readonly minimal?: string | null;
  readonly low?: string | null;
  readonly medium?: string | null;
  readonly high?: string | null;
  readonly xhigh?: string | null;
  readonly max?: string | null;
}

export interface FamilyTraits {
  /**
   * Whether the family emits `<think>` reasoning (drives pi's thinking
   * levels). Gemma4 routes its `<|channel>thought` protocol through its
   * family stream parser rather than the generic `<think>` tracker, but the
   * user-facing level still controls the prompt's `<|think|>` capability.
   */
  readonly reasoning: boolean;
  /**
   * Optional family-specific projection of Pi's thinking controls. Gemma4's
   * prompt protocol has two modes rather than four distinct effort levels:
   * minimal disables `<|think|>` and high enables it.
   */
  readonly thinkingLevelMap?: FamilyThinkingLevelMap;
  /**
   * Context-window fallback when `config.json` carries no
   * `max_position_embeddings` at either nesting level.
   *
   * A family's VOCAB size is never the right value here — the two are
   * unrelated numbers that happen to collide on some checkpoints
   * (nemotron_h is 131072 vocab / 1048576 context).
   */
  readonly fallbackContextWindow: number;
}

/**
 * Sampling defaults from Unsloth's Qwen3.6 guide:
 * https://unsloth.ai/docs/models/qwen3.6#recommended-settings
 *
 * All modes pin `top_k = 20` and `min_p = 0.0`; they differ in
 * `temperature`, `top_p`, and `presence_penalty`.
 *
 * Deliberately no `maxConsecutiveTokens` / `maxNgramRepeats` / `ngramSize`: the
 * native anti-repetition cutoff is off by default (vLLM-aligned), and a client
 * can still opt in per request.
 */
export const QWEN_SAMPLING_DEFAULTS = {
  /** Thinking mode for precise coding tasks. */
  thinkingCoding: {
    temperature: 0.6,
    topP: 0.95,
    topK: 20,
    minP: 0.0,
    presencePenalty: 0.0,
    repetitionPenalty: 1.0,
  } satisfies ChatConfig,

  /** Thinking mode for general tasks. */
  thinkingGeneral: {
    temperature: 1.0,
    topP: 0.95,
    topK: 20,
    minP: 0.0,
    presencePenalty: 1.5,
    repetitionPenalty: 1.0,
  } satisfies ChatConfig,

  /** Instruct (non-thinking) for general tasks. */
  instructGeneral: {
    temperature: 0.7,
    topP: 0.8,
    topK: 20,
    minP: 0.0,
    presencePenalty: 1.5,
    repetitionPenalty: 1.0,
  } satisfies ChatConfig,

  /** Instruct (non-thinking) for reasoning tasks. */
  instructReasoning: {
    temperature: 1.0,
    topP: 0.95,
    topK: 20,
    minP: 0.0,
    presencePenalty: 1.5,
    repetitionPenalty: 1.0,
  } satisfies ChatConfig,
} as const;

/** Sampling defaults for Gemma4 Instruct. */
export const GEMMA4_SAMPLING_DEFAULTS: ChatConfig = {
  temperature: 0.7,
  topP: 0.95,
  topK: 64,
  minP: 0.0,
  presencePenalty: 0.0,
  repetitionPenalty: 1.0,
};

/** Sampling defaults from Meta's Muse-Glimmer release recipe. */
export const MUSE_GLIMMER_SAMPLING_DEFAULTS: ChatConfig = {
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  minP: 0.0,
  presencePenalty: 0.0,
  repetitionPenalty: 1.0,
};

/** Sampling defaults from NVIDIA's Nemotron 3.5 Lightning release recipe. */
export const NEMOTRON_SAMPLING_DEFAULTS: ChatConfig = {
  temperature: 1.0,
  topP: 0.95,
  topK: 20,
  minP: 0.0,
  presencePenalty: 0.0,
  repetitionPenalty: 1.0,
};

/** Sampling defaults for LFM2.5 Thinking. */
export const LFM2_SAMPLING_DEFAULTS: ChatConfig = {
  temperature: 0.05,
  topP: 1.0,
  topK: 50,
  minP: 0.0,
  presencePenalty: 0.0,
  repetitionPenalty: 1.05,
};

/**
 * Sampling defaults from IFM's K2-Horizon model card
 * (temperature 1.0 / top_p 0.95; the card publishes no top_k).
 */
export const K2_SAMPLING_DEFAULTS: ChatConfig = {
  temperature: 1.0,
  topP: 0.95,
  minP: 0.0,
  presencePenalty: 0.0,
  repetitionPenalty: 1.0,
};

/**
 * Sampling defaults + per-model output cap. A per-request client value still
 * wins: `ChatSession.mergeConfig` treats per-call config as an overlay on top
 * of `defaultConfig`.
 */
export interface LaunchPreset {
  sampling: ChatConfig;
  maxOutputTokens: number;
}

/**
 * Structural mirror of `StreamingModelOptions` in `stream.ts` — the options
 * literal a family's `makeStreamingModel` wrapper passes. Declared here so
 * the registry row owns the value; `stream.ts` pins field-for-field
 * assignability against the factory's own interface in its compile-time
 * conformance block (the same pattern {@link FamilyThinkingLevelMap} uses
 * for pi's thinking-level map).
 */
export interface FamilyStreamOpts {
  /**
   * `static load` records the on-disk model path so the wrapper can serve
   * `applyChatTemplate` from a lazily constructed tokenizer.
   */
  readonly recordModelPath: boolean;
  /**
   * Attach the factory's path-backed `applyChatTemplate`; defaults to
   * `recordModelPath`. Qwen3 records its path but keeps the native
   * tokenizer-backed implementation, hence `applyTemplate: false`.
   */
  readonly applyTemplate?: boolean;
  /**
   * Model-specific ordering for structured multimodal content parts. The
   * tokenizer applies this policy after sanitization while the checkpoint
   * Jinja template continues to own all role and wire-format tokens.
   */
  readonly templateContentPolicy?: {
    readonly order: 'textThenMedia' | 'imagesThenText';
    /**
     * When sanitized text already contains this model-owned placeholder, keep
     * the message structured but do not synthesize additional image parts.
     */
    readonly existingImagePlaceholder?: string;
  };
  /**
   * Preserve the native raw assistant bytes in session history. LFM2's
   * checkpoint template consumes reasoning inside `message.content` and does
   * not read the structured `reasoning_content` field used by Qwen/Gemma.
   */
  readonly replayAssistantRawText?: boolean;
}

interface ModelFamilyDataBase {
  /** Canonical `ModelType` id. */
  readonly id: string;
  readonly match: ModelConfigMatcher;
  /** Backward-compatible fallback when config.json omits model_type or sets it to null. */
  readonly defaultForNullishModelType?: true;
  readonly acceptsDraftModel?: true;
  /**
   * Native `load` accepts a matching auxiliary HF checkpoint directory — the
   * original checkpoint supplying qwen4_exp's MTP and vision weights that a
   * GGUF omits. Gates `LoadModelOptions.auxiliaryModelPath` in
   * `models/model-loader.ts` the same way `acceptsDraftModel` gates
   * `draftModelPath`.
   */
  readonly acceptsAuxiliaryModel?: true;
  /**
   * GGUF `general.architecture` values whose native `load(path)` accepts a
   * direct GGUF file.
   */
  readonly ggufArchitectures?: readonly string[];
  /** Config markers for optimistic discovery; the loaded model is authoritative. */
  readonly visionConfigKeys?: readonly string[];
  /** Native GGUF discovery policy, shared by file and directory scans. */
  readonly ggufDiscovery?: {
    readonly variants: 'all' | 'qwen35-xl';
    readonly requiresAssets?: true;
    readonly split?: true;
  };
  /**
   * The options literal this family's `makeStreamingModel` wrapper passes —
   * `stream.ts` builds each chat family's class from `FAMILY_ROWS[id].streamOpts`
   * and `@mlx-node/vlm` reads the `qianfan-ocr` row the same way. Optional:
   * families without a streaming wrapper carry none (`harrier` is
   * embedding-only; `internvl_chat` loads the unwrapped native class), and
   * `lfm2_moe` shares `lfm2`'s wrapper, so the literal lives once, on `lfm2`.
   */
  readonly streamOpts?: FamilyStreamOpts;
  /**
   * This family's paged prefix blocks restore soundly from the SSD cold
   * tier. `COLD_TIER_RESTORE_FAMILIES` (`packages/agent/src/cold-tier.ts`)
   * derives its set from these flags; the native `COLD_RESTORE_FAMILIES`
   * (`crates/mlx-core/src/cold_tier.rs`) remains the authority —
   * `cold-tier-families.test.ts` pins parity. Widening is authorized by the
   * family's native restart-parity gate alone; see cold-tier.ts.
   */
  readonly coldRestoreEligible?: true;
}

/**
 * A chat-capable family MUST declare `traits` and a `launchPreset` — the
 * compile-time completeness gate. One preset serves every surface:
 * `@mlx-node/server` discovery, `mlx launch claude` and `mlx agent`. A chat
 * family is therefore reachable from all of them or from none, never from one
 * and not another.
 */
interface ChatFamilyData extends ModelFamilyDataBase {
  readonly kind: 'trainable' | 'loadable';
  readonly traits: FamilyTraits;
  readonly launchPreset: LaunchPreset;
}

interface NonGenerativeFamilyData extends ModelFamilyDataBase {
  readonly kind: 'embedding' | 'vlm';
  readonly traits?: undefined;
  readonly launchPreset?: undefined;
}

export type ModelFamilyData = ChatFamilyData | NonGenerativeFamilyData;

/**
 * Ordered source of truth for every supported model family's registration
 * data. Each entry owns its canonical `ModelType`, raw config aliases /
 * architecture probes, and `ChatSession` eligibility via `kind`:
 *
 *   - `'trainable'` — GRPO/SFT-capable LM (Qwen3 family); chat-capable.
 *   - `'loadable'`  — chat-capable LM with no trainer engine (Gemma4, LFM2).
 *   - `'embedding'` — no chat surface (Harrier); rejected by `loadSession`.
 *   - `'vlm'`       — VLM whose AsyncGenerator wrapper lives in
 *                     `@mlx-node/vlm` (importing it here would create a
 *                     circular package dependency), so `loadSession`
 *                     rejects it and routes callers to `@mlx-node/vlm`.
 *
 * ORDER IS LOAD-BEARING: a base family is selected from an explicit alias or
 * the single declarative nullish-model_type default, then architecture probes
 * refine it in declaration order. Gemma's unified architecture is
 * authoritative (matching the native loader); Harrier refines a Qwen3 base.
 * Adding a family means adding one data row here plus one loader binding in
 * `models/model-loader.ts`; the row type and the family-completeness test
 * enumerate everything else.
 */
export const MODEL_FAMILY_DATA = [
  {
    id: 'gemma4',
    visionConfigKeys: ['vision_config', 'unified_vision_config'],
    kind: 'loadable',
    ggufArchitectures: ['gemma4'],
    ggufDiscovery: { variants: 'all', requiresAssets: true },
    match: {
      rawModelTypes: ['gemma4', 'gemma4_text', 'gemma4_unified'],
      architectureProbe: ({ architectures }) => architectures.has('Gemma4UnifiedForConditionalGeneration'),
    },
    acceptsDraftModel: true,
    coldRestoreEligible: true,
    streamOpts: { recordModelPath: true },
    traits: {
      reasoning: true,
      thinkingLevelMap: {
        minimal: 'minimal',
        low: null,
        medium: null,
        high: 'high',
      },
      fallbackContextWindow: 131072,
    },
    launchPreset: {
      sampling: GEMMA4_SAMPLING_DEFAULTS,
      maxOutputTokens: 16384,
    },
  },
  {
    id: 'muse_glimmer',
    kind: 'loadable',
    ggufArchitectures: ['muse-glimmer'],
    ggufDiscovery: { variants: 'all', requiresAssets: true },
    match: {
      rawModelTypes: ['muse_glimmer', 'muse_glimmer_text'],
      architectureProbe: ({ architectures }) => architectures.has('MuseGlimmerForConditionalGeneration'),
    },
    coldRestoreEligible: true,
    streamOpts: { recordModelPath: true },
    traits: {
      reasoning: true,
      fallbackContextWindow: 131072,
    },
    launchPreset: {
      sampling: MUSE_GLIMMER_SAMPLING_DEFAULTS,
      maxOutputTokens: 16384,
    },
  },
  {
    id: 'harrier',
    kind: 'embedding',
    match: {
      rawModelTypes: ['harrier'],
      architectureProbe: ({ modelType, architectures }) =>
        modelType === 'qwen3' && architectures.has('Qwen3Model') && !architectures.has('Qwen3ForCausalLM'),
    },
  },
  {
    id: 'qwen3',
    kind: 'trainable',
    match: { rawModelTypes: ['qwen3'] },
    defaultForNullishModelType: true,
    coldRestoreEligible: true,
    /**
     * Records its model path (so prototype-set + path-recording match the
     * other families) but does not install the factory's path-backed
     * `applyChatTemplate`; it retains the native tokenizer-backed method.
     */
    streamOpts: { recordModelPath: true, applyTemplate: false },
    traits: { reasoning: true, fallbackContextWindow: 40960 },
    launchPreset: {
      sampling: QWEN_SAMPLING_DEFAULTS.thinkingCoding,
      maxOutputTokens: 38912,
    },
  },
  {
    id: 'qwen3_5',
    visionConfigKeys: ['vision_config'],
    kind: 'trainable',
    match: { rawModelTypes: ['qwen3_5'] },
    acceptsDraftModel: true,
    coldRestoreEligible: true,
    streamOpts: { recordModelPath: true },
    ggufArchitectures: ['qwen35'],
    ggufDiscovery: { variants: 'qwen35-xl' },
    traits: { reasoning: true, fallbackContextWindow: 262144 },
    launchPreset: {
      sampling: QWEN_SAMPLING_DEFAULTS.thinkingCoding,
      maxOutputTokens: 81920,
    },
  },
  {
    id: 'qwen3_5_moe',
    visionConfigKeys: ['vision_config'],
    kind: 'trainable',
    ggufArchitectures: ['qwen35moe'],
    ggufDiscovery: { variants: 'qwen35-xl' },
    match: { rawModelTypes: ['qwen3_5_moe'] },
    coldRestoreEligible: true,
    streamOpts: { recordModelPath: true },
    traits: { reasoning: true, fallbackContextWindow: 262144 },
    launchPreset: {
      sampling: QWEN_SAMPLING_DEFAULTS.thinkingCoding,
      maxOutputTokens: 81920,
    },
  },
  {
    id: 'qwen4_exp',
    visionConfigKeys: ['vision_config'],
    kind: 'loadable',
    match: { rawModelTypes: ['qwen4_exp', 'qwen4_exp_text'] },
    acceptsAuxiliaryModel: true,
    streamOpts: { recordModelPath: true },
    ggufArchitectures: ['qwen4exp'],
    ggufDiscovery: { variants: 'all', split: true },
    traits: { reasoning: true, fallbackContextWindow: 262144 },
    launchPreset: { sampling: QWEN_SAMPLING_DEFAULTS.thinkingCoding, maxOutputTokens: 8192 },
  },
  {
    id: 'lfm2',
    kind: 'loadable',
    match: { rawModelTypes: ['lfm2'] },
    coldRestoreEligible: true,
    /**
     * `replayAssistantRawText` also serves `lfm2_moe`: both families load
     * through the shared `Lfm2Model` wrapper, so the flag lives once, here.
     */
    streamOpts: { recordModelPath: true, replayAssistantRawText: true },
    traits: { reasoning: true, fallbackContextWindow: 128000 },
    launchPreset: {
      sampling: LFM2_SAMPLING_DEFAULTS,
      maxOutputTokens: 8192,
    },
  },
  {
    id: 'lfm2_moe',
    kind: 'loadable',
    match: { rawModelTypes: ['lfm2_moe'] },
    coldRestoreEligible: true,
    traits: { reasoning: true, fallbackContextWindow: 128000 },
    /**
     * LFM2.5-8B-A1B: LiquidAI's MoE card recommends temperature 0.2 / top_k 80
     * — deliberately NOT the dense `lfm2` values (0.05 / 50).
     */
    launchPreset: {
      sampling: {
        temperature: 0.2,
        topP: 1.0,
        topK: 80,
        minP: 0.0,
        presencePenalty: 0.0,
        repetitionPenalty: 1.05,
      },
      maxOutputTokens: 8192,
    },
  },
  {
    id: 'nemotron_h',
    kind: 'loadable',
    match: {
      rawModelTypes: ['nemotron_h'],
      architectureProbe: ({ architectures }) => architectures.has('NemotronHForCausalLM'),
    },
    streamOpts: { recordModelPath: true },
    traits: {
      reasoning: true,
      fallbackContextWindow: 1048576,
    },
    launchPreset: {
      sampling: NEMOTRON_SAMPLING_DEFAULTS,
      maxOutputTokens: 32768,
    },
  },
  {
    id: 'k2_horizon',
    kind: 'loadable',
    match: {
      rawModelTypes: ['k2_horizon'],
      architectureProbe: ({ architectures }) => architectures.has('K2HorizonForCausalLM'),
    },
    streamOpts: { recordModelPath: true },
    traits: {
      reasoning: true,
      /**
       * K2 always thinks — its template raises on `reasoning_effort="none"`
       * and offers exactly three levels (high/medium/low, rendered as
       * `<ifm|think>` / `<ifm|think_fast>` / `<ifm|think_faster>`). The map
       * below hides xhigh/max from pi (null) and projects minimal→low; the
       * native side applies the same projection for direct API callers
       * (none/minimal→low, xhigh/max→high — `normalize_k2_effort`).
       */
      thinkingLevelMap: {
        minimal: 'low',
        low: 'low',
        medium: 'medium',
        high: 'high',
        xhigh: null,
        max: null,
      },
      fallbackContextWindow: 524288,
    },
    launchPreset: {
      sampling: K2_SAMPLING_DEFAULTS,
      maxOutputTokens: 32768,
    },
  },
  {
    id: 'internvl_chat',
    kind: 'vlm',
    match: { rawModelTypes: ['internvl_chat'] },
  },
  {
    id: 'qianfan-ocr',
    kind: 'vlm',
    match: { rawModelTypes: ['qianfan-ocr'] },
    /** Consumed by the `QianfanOCRModel` wrapper in `@mlx-node/vlm`. */
    streamOpts: {
      recordModelPath: true,
      templateContentPolicy: {
        order: 'imagesThenText',
        existingImagePlaceholder: '<image>',
      },
    },
  },
] as const satisfies readonly ModelFamilyData[];

type FamilyDataRow = (typeof MODEL_FAMILY_DATA)[number];

export type ModelType = FamilyDataRow['id'];

type ChatFamilyRow = Extract<FamilyDataRow, { readonly kind: 'trainable' | 'loadable' }>;

export type ChatFamilyId = ChatFamilyRow['id'];

export type TrainableFamilyId = Extract<FamilyDataRow, { readonly kind: 'trainable' }>['id'];

/**
 * The registry rows keyed by canonical id, with each row's literal type
 * preserved — `FAMILY_ROWS.qwen3_5.streamOpts` still knows `recordModelPath`
 * is `true`, which `makeStreamingModel`'s `ResolvedApplyTemplate` needs (a
 * `ModelFamilyData`-typed lookup such as {@link familyDataFor} would widen
 * the literals to `boolean` and break the per-family instance surface).
 */
type FamilyRowById = { [R in FamilyDataRow as R['id']]: R };

export const FAMILY_ROWS = Object.fromEntries(MODEL_FAMILY_DATA.map((row) => [row.id, row] as const)) as FamilyRowById;

/**
 * Every chat-capable family (kind trainable | loadable), in registry order —
 * the default set the paged-config override manager forces onto the block-paged
 * path. Derived, so a new chat family can never be forgotten.
 */
export const CHAT_FAMILY_IDS: readonly ChatFamilyId[] = MODEL_FAMILY_DATA.filter(
  (row): row is ChatFamilyRow => row.kind === 'trainable' || row.kind === 'loadable',
).map((row) => row.id);

/** Detection results that cannot back a chat endpoint (kind embedding | vlm). */
export const NON_GENERATIVE_FAMILY_IDS: ReadonlySet<ModelType> = new Set<ModelType>(
  MODEL_FAMILY_DATA.filter((row) => row.kind === 'embedding' || row.kind === 'vlm').map((row) => row.id),
);

interface FamilyDataIndex {
  readonly byId: ReadonlyMap<string, ModelFamilyData>;
  readonly byRawModelType: ReadonlyMap<string, ModelFamilyData>;
  readonly defaultForNullishModelType: ModelFamilyData;
}

function buildFamilyDataIndex(rows: readonly ModelFamilyData[]): FamilyDataIndex {
  const byId = new Map<string, ModelFamilyData>();
  const byRawModelType = new Map<string, ModelFamilyData>();
  let defaultForNullishModelType: ModelFamilyData | undefined;

  for (const family of rows) {
    const previousFamily = byId.get(family.id);
    if (previousFamily !== undefined) {
      throw new Error(`Duplicate canonical model type "${family.id}" in model family registry`);
    }
    byId.set(family.id, family);

    for (const rawModelType of family.match.rawModelTypes) {
      const previous = byRawModelType.get(rawModelType);
      if (previous !== undefined) {
        throw new Error(`Duplicate model_type alias "${rawModelType}" for "${previous.id}" and "${family.id}"`);
      }
      byRawModelType.set(rawModelType, family);
    }

    if (family.defaultForNullishModelType === true) {
      if (defaultForNullishModelType !== undefined) {
        throw new Error(
          `Duplicate nullish-model_type defaults for "${defaultForNullishModelType.id}" and "${family.id}"`,
        );
      }
      defaultForNullishModelType = family;
    }
  }

  if (defaultForNullishModelType === undefined) {
    throw new Error('Model family registry must declare exactly one nullish-model_type default');
  }
  return { byId, byRawModelType, defaultForNullishModelType };
}

const FAMILY_DATA_INDEX = buildFamilyDataIndex(MODEL_FAMILY_DATA);

/** Registration data for a canonical family id, or `undefined` for a foreign string. */
export function familyDataFor(modelType: string): ModelFamilyData | undefined {
  return FAMILY_DATA_INDEX.byId.get(modelType);
}

function chatFamilyDataFor(modelType: string): ChatFamilyData | undefined {
  const row = FAMILY_DATA_INDEX.byId.get(modelType);
  return row !== undefined && (row.kind === 'trainable' || row.kind === 'loadable') ? row : undefined;
}

/** Canonical family id owning a raw `config.json` model_type alias. */
export function rawModelTypeToCanonical(rawModelType: string): ModelType | undefined {
  return FAMILY_DATA_INDEX.byRawModelType.get(rawModelType)?.id as ModelType | undefined;
}

/** Agent discovery traits for a chat-capable family. */
export function familyTraitsFor(modelType: string): FamilyTraits | undefined {
  return chatFamilyDataFor(modelType)?.traits;
}

/**
 * Launch preset for every surface that serves a chat family:
 * `@mlx-node/server` discovery, `mlx launch claude` and `mlx agent`.
 * `undefined` only for a non-generative family or an unknown type.
 */
export function launchPresetFor(modelType: string): LaunchPreset | undefined {
  return chatFamilyDataFor(modelType)?.launchPreset;
}

export class MalformedModelConfigError extends Error {
  constructor(modelPath: string, reason: string) {
    super(`Malformed config.json in ${modelPath}: ${reason}`);
    this.name = 'MalformedModelConfigError';
  }
}

export class UnsupportedModelTypeError extends Error {
  constructor(modelPath: string, rawModelTypeLabel: string) {
    super(`Unsupported model_type "${rawModelTypeLabel}" in ${modelPath}/config.json`);
    this.name = 'UnsupportedModelTypeError';
  }
}

/**
 * Fail-closed validation: a config.json whose root is not a plain object,
 * or whose `architectures` is neither an array nor a string, is rejected
 * instead of coerced (coercion would fall through to the qwen3
 * nullish-model_type default and silently misroute the checkpoint).
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

/**
 * Pure family detection over a parsed `config.json`: alias (or the qwen3
 * nullish default) picks a base family, then architecture probes refine it in
 * registry declaration order. Throws {@link MalformedModelConfigError} /
 * {@link UnsupportedModelTypeError} with `modelPath` naming the checkpoint.
 * The filesystem/GGUF half lives in `detectModelType`
 * (`models/model-loader.ts`); native-free consumers (dashboard labels via
 * `@mlx-node/agent/catalog`) call this directly.
 */
export function matchFamily(modelPath: string, parsedConfig: unknown): ModelType {
  const config = normalizeConfig(modelPath, parsedConfig);
  const rows: readonly ModelFamilyData[] = MODEL_FAMILY_DATA;
  const baseFamily = config.usesDefaultModelType
    ? FAMILY_DATA_INDEX.defaultForNullishModelType
    : config.rawModelType === undefined
      ? undefined
      : FAMILY_DATA_INDEX.byRawModelType.get(config.rawModelType);
  const matchContext: ModelConfigMatchContext = { ...config, modelType: baseFamily?.id };
  const family = rows.find((candidate) => candidate.match.architectureProbe?.(matchContext) === true) ?? baseFamily;
  if (family === undefined) throw new UnsupportedModelTypeError(modelPath, config.rawModelTypeLabel);
  return family.id as ModelType;
}
