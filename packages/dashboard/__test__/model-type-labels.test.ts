/**
 * Pins the dashboard's family label against the lm registry's detection
 * decisions (`MODEL_FAMILY_REGISTRY` in
 * `packages/lm/src/models/model-loader.ts`): every raw alias labels its
 * canonical family id, the architecture probes fire in registry declaration
 * order (gemma4 unified, muse, harrier, nemotron), and shapes the loader
 * rejects (unknown model_type, malformed architectures) keep the dashboard's
 * raw-string fallback instead of throwing. The fixtures are deliberately a
 * hand-held mirror of the registry rows: the dashboard must produce these
 * labels no matter how its detector is implemented.
 */

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import { discoverLocalModels } from '../src/models.js';

const UNIFIED_GEMMA = 'Gemma4UnifiedForConditionalGeneration';

const LABEL_CASES = [
  // Raw config aliases → canonical family id, one row per registry alias.
  ['alias-gemma4', { model_type: 'gemma4' }, 'gemma4'],
  ['alias-gemma4-text', { model_type: 'gemma4_text' }, 'gemma4'],
  ['alias-gemma4-unified', { model_type: 'gemma4_unified' }, 'gemma4'],
  ['alias-muse-glimmer', { model_type: 'muse_glimmer' }, 'muse_glimmer'],
  ['alias-muse-glimmer-text', { model_type: 'muse_glimmer_text' }, 'muse_glimmer'],
  ['alias-harrier', { model_type: 'harrier' }, 'harrier'],
  ['alias-qwen3', { model_type: 'qwen3' }, 'qwen3'],
  ['alias-qwen3-5', { model_type: 'qwen3_5' }, 'qwen3_5'],
  ['alias-qwen3-5-moe', { model_type: 'qwen3_5_moe' }, 'qwen3_5_moe'],
  ['alias-lfm2', { model_type: 'lfm2' }, 'lfm2'],
  ['alias-lfm2-moe', { model_type: 'lfm2_moe' }, 'lfm2_moe'],
  ['alias-nemotron-h', { model_type: 'nemotron_h' }, 'nemotron_h'],
  ['alias-internvl-chat', { model_type: 'internvl_chat' }, 'internvl_chat'],
  ['alias-qianfan-ocr', { model_type: 'qianfan-ocr' }, 'qianfan-ocr'],

  // Architecture probes, in registry declaration order and precedence.
  ['probe-gemma-missing-type', { architectures: [UNIFIED_GEMMA] }, 'gemma4'],
  ['probe-gemma-null-type', { model_type: null, architectures: [UNIFIED_GEMMA] }, 'gemma4'],
  ['probe-gemma-unknown-type', { model_type: 'llama', architectures: [UNIFIED_GEMMA] }, 'gemma4'],
  ['probe-gemma-other-family-type', { model_type: 'lfm2', architectures: [UNIFIED_GEMMA] }, 'gemma4'],
  ['probe-gemma-string-arch', { architectures: UNIFIED_GEMMA }, 'gemma4'],
  ['probe-gemma-over-harrier', { model_type: 'qwen3', architectures: ['Qwen3Model', UNIFIED_GEMMA] }, 'gemma4'],
  ['probe-muse-arch', { model_type: 'unknown', architectures: ['MuseGlimmerForConditionalGeneration'] }, 'muse_glimmer'],
  ['probe-harrier', { model_type: 'qwen3', architectures: ['Qwen3Model'] }, 'harrier'],
  ['probe-harrier-causal-lm-negative', { model_type: 'qwen3', architectures: ['Qwen3Model', 'Qwen3ForCausalLM'] }, 'qwen3'],
  ['probe-harrier-default-base', { architectures: ['Qwen3Model'] }, 'harrier'],
  ['probe-harrier-non-string-entries', { model_type: 'qwen3', architectures: ['Qwen3Model', 42] }, 'harrier'],
  ['probe-nemotron-arch', { model_type: 'nemotron', architectures: ['NemotronHForCausalLM'] }, 'nemotron_h'],

  // Loader defaults and the dashboard's raw-string fallback for shapes the
  // loader fails closed on (a viewer must label, never throw).
  ['fallback-empty-config', {}, 'qwen3'],
  ['fallback-null-type', { model_type: null }, 'qwen3'],
  ['fallback-unknown-type', { model_type: 'llama' }, 'llama'],
  ['fallback-non-string-type', { model_type: 42 }, 'qwen3'],
  ['fallback-null-arch', { model_type: 'lfm2', architectures: null }, 'lfm2'],
  ['fallback-unrecognized-arch', { model_type: 'lfm2', architectures: ['LlamaForCausalLM'] }, 'lfm2'],
  ['fallback-malformed-arch-known-alias', { model_type: 'muse_glimmer_text', architectures: 42 }, 'muse_glimmer_text'],
  ['fallback-malformed-arch-unknown-type', { model_type: 'llama', architectures: {} }, 'llama'],
] as const satisfies readonly (readonly [string, Record<string, unknown>, string])[];

let modelsDir: string;
let labelByName: Map<string, string>;

beforeAll(() => {
  modelsDir = mkdtempSync(join(tmpdir(), 'dash-model-type-labels-'));
  for (const [name, config] of LABEL_CASES) {
    const dir = join(modelsDir, name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'config.json'), JSON.stringify(config));
  }
  const { models } = discoverLocalModels(modelsDir);
  labelByName = new Map(models.map((model) => [model.name, model.modelType]));
});

afterAll(() => {
  rmSync(modelsDir, { recursive: true, force: true });
});

describe('dashboard model-type labels', () => {
  it('labels every fixture', () => {
    expect(labelByName.size).toBe(LABEL_CASES.length);
  });

  it.each(LABEL_CASES)('labels %s', (name, _config, expected) => {
    expect(labelByName.get(name)).toBe(expected);
  });
});
