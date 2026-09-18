/**
 * Pins the dashboard's family label against the lm registry's own detection
 * (`matchFamily` over `MODEL_FAMILY_DATA` in `packages/lm/src/family-data.ts`).
 * Alias fixtures are derived from the registry rows themselves so a new family
 * is covered automatically (the alias→family map's literal pin lives in
 * `model-loader-registry.test.ts`); the architecture-probe and fallback cases
 * below stay hand-written because they pin probe order (gemma4 unified, muse,
 * harrier, nemotron) and the fail-closed shapes the loader must not throw on.
 */

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { MODEL_FAMILY_DATA } from '@mlx-node/lm/family-data';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import { discoverLocalModels } from '../src/models.js';

const UNIFIED_GEMMA = 'Gemma4UnifiedForConditionalGeneration';

// Raw config aliases → canonical family id, one row per registry alias.
const ALIAS_CASES = MODEL_FAMILY_DATA.flatMap((row) =>
  row.match.rawModelTypes.map((raw) => [`alias-${raw}`, { model_type: raw }, row.id] as const),
);

const LABEL_CASES = [
  ...ALIAS_CASES,

  // Architecture probes, in registry declaration order and precedence.
  ['probe-gemma-missing-type', { architectures: [UNIFIED_GEMMA] }, 'gemma4'],
  ['probe-gemma-null-type', { model_type: null, architectures: [UNIFIED_GEMMA] }, 'gemma4'],
  ['probe-gemma-unknown-type', { model_type: 'llama', architectures: [UNIFIED_GEMMA] }, 'gemma4'],
  ['probe-gemma-other-family-type', { model_type: 'lfm2', architectures: [UNIFIED_GEMMA] }, 'gemma4'],
  ['probe-gemma-string-arch', { architectures: UNIFIED_GEMMA }, 'gemma4'],
  ['probe-gemma-over-harrier', { model_type: 'qwen3', architectures: ['Qwen3Model', UNIFIED_GEMMA] }, 'gemma4'],
  [
    'probe-muse-arch',
    { model_type: 'unknown', architectures: ['MuseGlimmerForConditionalGeneration'] },
    'muse_glimmer',
  ],
  ['probe-harrier', { model_type: 'qwen3', architectures: ['Qwen3Model'] }, 'harrier'],
  [
    'probe-harrier-causal-lm-negative',
    { model_type: 'qwen3', architectures: ['Qwen3Model', 'Qwen3ForCausalLM'] },
    'qwen3',
  ],
  ['probe-harrier-default-base', { architectures: ['Qwen3Model'] }, 'harrier'],
  ['probe-harrier-non-string-entries', { model_type: 'qwen3', architectures: ['Qwen3Model', 42] }, 'harrier'],
  ['probe-nemotron-arch', { model_type: 'nemotron', architectures: ['NemotronHForCausalLM'] }, 'nemotron_h'],
  ['probe-k2-arch', { model_type: 'unknown', architectures: ['K2HorizonForCausalLM'] }, 'k2_horizon'],

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
