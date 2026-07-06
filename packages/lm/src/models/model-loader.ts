/**
 * Model loading utilities for Qwen3 models
 *
 * Handles loading pretrained weights from MLX format or converting from HuggingFace.
 */

import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { HarrierModel, QianfanOCRModel } from '@mlx-node/core';

import { ChatSession, type SessionCapableModel } from '../chat-session.js';
import type { LoadableModel } from '../interfaces.js';
import { Gemma4Model, Lfm2Model, Qwen3Model, Qwen35Model, Qwen35MoeModel } from '../stream.js';

/**
 * Single source of truth for every supported `model_type`. Each entry
 * pairs the loader (which native/wrapper class to instantiate) with a
 * `kind` that drives `ChatSession` eligibility:
 *
 *   - `'trainable'` — GRPO/SFT-capable LM (Qwen3 family); chat-capable.
 *   - `'loadable'`  — chat-capable LM with no trainer engine (Gemma4, LFM2).
 *   - `'embedding'` — no chat surface (Harrier); rejected by `loadSession`.
 *   - `'vlm'`       — VLM whose AsyncGenerator wrapper lives in
 *                     `@mlx-node/vlm` (importing it here would create a
 *                     circular package dependency), so `loadSession`
 *                     rejects it and routes callers to `@mlx-node/vlm`.
 *
 * The `ModelType` union, the supported-type set, the `loadModel`
 * dispatch, and the `loadSession` rejection rules are all derived from
 * this table — adding a family means adding one row here.
 */
const MODEL_REGISTRY = {
  qwen3: { load: (p: string) => Qwen3Model.load(p), kind: 'trainable' },
  qwen3_5: { load: (p: string) => Qwen35Model.load(p), kind: 'trainable' },
  qwen3_5_moe: { load: (p: string) => Qwen35MoeModel.load(p), kind: 'trainable' },
  gemma4: {
    load: (p: string, o?: LoadModelOptions) =>
      Gemma4Model.load(p, o?.draftModelPath === undefined ? null : { draftModelPath: o.draftModelPath }),
    kind: 'loadable',
  },
  lfm2: { load: (p: string) => Lfm2Model.load(p), kind: 'loadable' },
  lfm2_moe: { load: (p: string) => Lfm2Model.load(p), kind: 'loadable' },
  harrier: { load: (p: string) => HarrierModel.load(p), kind: 'embedding' },
  internvl_chat: { load: (p: string) => QianfanOCRModel.load(p), kind: 'vlm' },
  'qianfan-ocr': { load: (p: string) => QianfanOCRModel.load(p), kind: 'vlm' },
} as const satisfies Record<string, { load: (p: string) => Promise<unknown>; kind: string }>;

export type ModelType = keyof typeof MODEL_REGISTRY;

const SUPPORTED_MODEL_TYPES = new Set<ModelType>(Object.keys(MODEL_REGISTRY) as ModelType[]);

/** Optional settings for {@link loadModel} / {@link loadSession}. */
export interface LoadModelOptions {
  /**
   * Gemma4 only: directory of a DSpark draft checkpoint (config.json +
   * model.safetensors) loaded alongside the target model for speculative
   * decoding (forwarded as `Gemma4LoadOptions.draftModelPath`). DSpark runs
   * on the flat KV-cache path, so the target checkpoint must not explicitly
   * enable `use_block_paged_cache`. Setting this for any other model family
   * is a hard error — no other loader accepts a draft model.
   */
  draftModelPath?: string;
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
  if (options?.draftModelPath !== undefined && modelType !== 'gemma4') {
    throw new Error(
      `draftModelPath (DSpark speculative decoding) is only supported by gemma4 models; ` +
        `${modelPath} has model_type "${modelType}"`,
    );
  }
  return modelType === 'gemma4'
    ? MODEL_REGISTRY.gemma4.load(modelPath, options)
    : MODEL_REGISTRY[modelType].load(modelPath);
}

/**
 * Load a model from disk, auto-detecting architecture from config.json.
 *
 * Supports both language models (Qwen3, Qwen3.5) and vision-language models
 * (Qianfan-OCR / InternVL). Use `instanceof` to narrow the returned type.
 *
 * `options.draftModelPath` attaches a DSpark draft checkpoint for
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
 * `options.draftModelPath` attaches a DSpark draft checkpoint for
 * speculative decoding — gemma4 only; any other detected family rejects it.
 * The resulting session auto-enables the speculative path (the model
 * reports `hasMtpWeights()`); pass `enableMtp: false` per call to opt out.
 */
export async function loadSession(
  modelPath: string,
  options?: LoadModelOptions,
): Promise<ChatSession<SessionCapableModel>> {
  const modelType = await detectModelType(modelPath);
  const kind = MODEL_REGISTRY[modelType].kind;
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
    const config = JSON.parse(raw);
    const rawModelType: string = config.model_type ?? 'qwen3';

    // Normalize Gemma 4 text-family variants onto the single `gemma4` registry
    // key: `gemma4_text` (31B/E2B/26B) and `gemma4_unified` (12B multimodal,
    // loaded text-only) both drive the shared `gemma4` decoder.
    let modelType: ModelType = (
      rawModelType === 'gemma4_text' || rawModelType === 'gemma4_unified' ? 'gemma4' : rawModelType
    ) as ModelType;

    // Detect embedding models: Qwen3 backbone with base architecture (no ForCausalLM)
    if (modelType === 'qwen3') {
      const architectures: string[] = config.architectures ?? [];
      if (architectures.includes('Qwen3Model') && !architectures.includes('Qwen3ForCausalLM')) {
        modelType = 'harrier';
      }
    }

    // Route architecture-only unified Gemma 4 checkpoints to `gemma4` even when
    // `model_type` is absent (and thus qwen3-defaulted above). The native loader
    // (gemma4/persistence.rs parse_config) flags `is_unified` on EITHER
    // `model_type == "gemma4_unified"` OR this architecture; mirror that here so
    // a unified checkpoint carrying only `architectures` is not misrouted to
    // Qwen3Model.
    const architectures: string[] = config.architectures ?? [];
    if (architectures.includes('Gemma4UnifiedForConditionalGeneration')) {
      modelType = 'gemma4';
    }

    if (!SUPPORTED_MODEL_TYPES.has(modelType)) {
      throw new Error(`Unsupported model_type "${modelType}" in ${modelPath}/config.json`);
    }
    return modelType;
  } catch (e) {
    if (e instanceof Error && e.message.startsWith('Unsupported model_type')) throw e;
    throw new Error(`Cannot detect model type: config.json not found in ${modelPath}`);
  }
}
