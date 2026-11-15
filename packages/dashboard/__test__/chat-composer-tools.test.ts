/**
 * @vitest-environment happy-dom
 */

import { describe, expect, it, vi } from 'vite-plus/test';

import { CHAT } from '../ui/src/lib/chat-strings.js';
import type { ChatHostState } from '../ui/src/lib/chat-types.js';
import { modelBases, startModel } from '../ui/src/pages/chat/composer-tools.js';
import { pickedVariants, storeVariant } from '../ui/src/pages/chat/drafts.js';
import { hostModel, READY_CHAT_HOST } from './render.js';

const MODELS = ['alpha', 'bravo', 'charlie', 'delta'].map((id) => hostModel(id, 32768));

function host(over: Partial<ChatHostState>): ChatHostState {
  return { ...READY_CHAT_HOST, models: MODELS, resident: [], loading: [], defaultModelId: 'alpha', ...over };
}

describe('startModel', () => {
  it('starts a new conversation on the model in memory, before the stored choice', () => {
    const state = host({ resident: [{ modelId: 'charlie', sizeBytes: null }] });
    expect(startModel(state, 'bravo')).toBe('charlie');
    expect(startModel(state, null)).toBe('charlie');
  });

  it('with no model in memory: the model in load, then the stored choice, then the default, then the first', () => {
    expect(
      startModel(
        host({ loading: [{ modelId: 'delta', startedAt: 1, phase: 'weights', stepsDone: 0, steps: 0 }] }),
        'bravo',
      ),
    ).toBe('delta');
    expect(startModel(host({}), 'bravo')).toBe('bravo');
    expect(startModel(host({ defaultModelId: 'charlie' }), 'gone')).toBe('charlie');
    expect(startModel(host({ defaultModelId: null }), null)).toBe('alpha');
  });

  it('ignores a model in memory that the engine does not serve to the app', () => {
    const state = host({ resident: [{ modelId: 'other', sizeBytes: null }] });
    expect(startModel(state, 'bravo')).toBe('bravo');
    expect(startModel(null, 'bravo')).toBeNull();
    expect(startModel(host({ models: [] }), 'bravo')).toBeNull();
  });
});

const GB = 1024 ** 3;

/** A served model of `base`; size in GB. */
function build(
  id: string,
  base: string,
  variant: string,
  sizeGb: number | null,
  over: Partial<ChatHostState['models'][number]> = {},
): ChatHostState['models'][number] {
  return {
    id,
    base,
    variant,
    format: id.includes('gguf') || variant.startsWith('UD-') ? 'gguf' : 'mlx',
    contextTokens: 262144,
    sizeBytes: sizeGb === null ? null : Math.round(sizeGb * GB),
    ...over,
  };
}

const XL = build('Qwen3.8-27B-UD-Q4_K_XL', 'Qwen3.8-27B', 'UD-Q4_K_XL', 17.6);
const M = build('Qwen3.8-27B-UD-Q4_K_M', 'Qwen3.8-27B', 'UD-Q4_K_M', 16.5);
const BF16 = build('qwen3.8-27b-bf16', 'Qwen3.8-27B', 'BF16', 55.6);
const AGENTS = build('agents-a1', 'agents-a1', 'BF16', 70.2);
const FLASH = build('Qwen3.8-Flash-Next-UD-Q4_K_XL', 'Qwen3.8-Flash-Next', 'UD-Q4_K_XL', 111.3);
const ZETA = build('zeta-q4', 'zeta', 'Q4_0', null);
const ZETA8 = build('zeta-q8', 'zeta', 'Q8_0', 9);
const VARIANTS = [FLASH, BF16, AGENTS, XL, M, ZETA, ZETA8];

function bases(
  over: Partial<ChatHostState>,
  resident: string[] = [],
  value: string | null = null,
  picked: Record<string, string> = {},
  mode: 'chat' | 'work' = 'chat',
) {
  return modelBases(host({ models: VARIANTS, ...over }), new Set(resident), mode, value, picked);
}

describe('modelBases', () => {
  it('groups by base: one row per base, its quantizations smallest first, unknown sizes last', () => {
    const rows = bases({});
    expect(rows.map((row) => row.base)).toEqual(['agents-a1', 'Qwen3.8-27B', 'Qwen3.8-Flash-Next', 'zeta']);
    expect(rows[1]!.variants.map((v) => v.id)).toEqual([M.id, XL.id, BF16.id]);
    expect(rows[1]!.variants.map((v) => v.label)).toEqual(['UD-Q4_K_M', 'UD-Q4_K_XL', 'BF16']);
    expect(rows[3]!.variants.map((v) => v.id)).toEqual([ZETA8.id, ZETA.id]);
  });

  it('in memory first, then in load, then the rest, in both lists; a base takes its best state', () => {
    const rows = bases({ loading: [{ modelId: FLASH.id, startedAt: 1, phase: 'weights', stepsDone: 0, steps: 0 }] }, [
      BF16.id,
    ]);
    expect(rows.map((row) => [row.base, row.state])).toEqual([
      ['Qwen3.8-27B', 'loaded'],
      ['Qwen3.8-Flash-Next', 'loading'],
      ['agents-a1', 'idle'],
      ['zeta', 'idle'],
    ]);
    expect(rows[0]!.variants.map((v) => [v.label, v.state])).toEqual([
      ['BF16', 'loaded'],
      ['UD-Q4_K_M', 'idle'],
      ['UD-Q4_K_XL', 'idle'],
    ]);
  });

  it('a base with a loaded and a loading variant says loading and names the one in load', () => {
    const rows = bases({ loading: [{ modelId: M.id, startedAt: 1, phase: 'weights', stepsDone: 0, steps: 0 }] }, [
      XL.id,
    ]);
    const qwen = rows[0]!;
    expect(qwen.base).toBe('Qwen3.8-27B');
    expect(qwen.state).toBe('loading');
    expect(qwen.stateVariant).toBe('UD-Q4_K_M');
    expect(qwen.variants.map((v) => [v.label, v.state])).toEqual([
      ['UD-Q4_K_XL', 'loaded'],
      ['UD-Q4_K_M', 'loading'],
      ['BF16', 'idle'],
    ]);
  });

  it('facts: the count before the context on a base with 2+; context, format and size on a base with one', () => {
    const rows = bases({}, [FLASH.id]);
    const flash = rows[0]!;
    expect(flash.base).toBe('Qwen3.8-Flash-Next');
    expect(flash.state).toBe('loaded');
    expect(flash.stateVariant).toBeUndefined();
    expect(flash.facts).toEqual(['256K context', 'GGUF', '111 GB']);
    const qwen = rows.find((row) => row.base === 'Qwen3.8-27B')!;
    expect(qwen.facts).toEqual(['3 quantizations', '256K context']);
    expect(qwen.variants[0]!.facts).toEqual(['GGUF', '16.5 GB']);
    expect(qwen.variants[2]!.facts).toEqual(['MLX', '55.6 GB']);
    expect(rows.find((row) => row.base === 'zeta')!.variants[1]!.facts).toEqual(['MLX']);
  });

  it('contexts that differ inside a base: "up to" on the base, the context on each quantization', () => {
    const models = [build('a-q4', 'a', 'Q4_0', 1, { contextTokens: 40960 }), build('a-q8', 'a', 'Q8_0', 2)];
    const [row] = modelBases(host({ models }), new Set(), 'chat', null, {});
    expect(row!.facts).toEqual(['2 quantizations', 'up to 256K context']);
    expect(row!.variants.map((v) => v.facts)).toEqual([
      ['MLX', '1.0 GB', '40K context'],
      ['MLX', '2.0 GB', '256K context'],
    ]);
  });

  it('Work: a small variant warns; the base warns only when all of its variants are small', () => {
    const models = [
      build('a-q4', 'a', 'Q4_0', 1, { contextTokens: 16384 }),
      build('a-q8', 'a', 'Q8_0', 2),
      build('b-q4', 'b', 'Q4_0', 1, { contextTokens: 16384 }),
    ];
    const work = modelBases(host({ models }), new Set(), 'work', null, {});
    expect(work[0]!.warning).toBeUndefined();
    expect(work[0]!.variants.map((v) => v.warning)).toEqual([CHAT.modelSmall, undefined]);
    expect(work[1]!.warning).toBe(CHAT.modelSmall);
    const chat = modelBases(host({ models }), new Set(), 'chat', null, {});
    expect(chat.every((row) => row.warning === undefined)).toBe(true);
  });

  it('preselect: the conversation model, then in memory, then in load, then the last picked, then the smallest', () => {
    const qwen = (rows: ReturnType<typeof bases>) => rows.find((row) => row.base === 'Qwen3.8-27B')!.preselect;
    const loading = {
      loading: [{ modelId: BF16.id, startedAt: 1, phase: 'weights' as const, stepsDone: 0, steps: 0 }],
    };
    const picked = { 'qwen3.8-27b': XL.id };
    expect(qwen(bases(loading, [M.id], BF16.id, picked))).toBe(BF16.id);
    expect(qwen(bases(loading, [M.id], AGENTS.id, picked))).toBe(M.id);
    expect(qwen(bases(loading, [], AGENTS.id, picked))).toBe(BF16.id);
    expect(qwen(bases({}, [], null, picked))).toBe(XL.id);
    expect(qwen(bases({}, [], null, {}))).toBe(M.id);
    expect(qwen(bases({}, [], null, { 'qwen3.8-27b': 'gone-id' }))).toBe(M.id);
    expect(qwen(bases({}, [], null, { 'qwen3.8-27b': AGENTS.id }))).toBe(M.id);
  });

  it('the chat is on M while memory holds XL: M is preselected', () => {
    expect(bases({}, [XL.id], M.id).find((row) => row.base === 'Qwen3.8-27B')!.preselect).toBe(M.id);
  });
});

describe('the last model picked per base', () => {
  const KEY = 'mlx-node:chat-variant:v1';

  it('is one map for both modes, keyed by the base in lower case', () => {
    localStorage.removeItem(KEY);
    expect(pickedVariants()).toEqual({});
    storeVariant('Qwen3.8-27B', XL.id);
    storeVariant('agents-a1', AGENTS.id);
    expect(JSON.parse(localStorage.getItem(KEY)!)).toEqual({ 'qwen3.8-27b': XL.id, 'agents-a1': AGENTS.id });
    expect(pickedVariants()).toEqual({ 'qwen3.8-27b': XL.id, 'agents-a1': AGENTS.id });
    localStorage.setItem(KEY, '[1,2]');
    expect(pickedVariants()).toEqual({});
    localStorage.setItem(KEY, '{"a":3,"b":"b-q4"}');
    expect(pickedVariants()).toEqual({ b: 'b-q4' });
    localStorage.removeItem(KEY);
  });

  it('storage that throws reads as empty and a write is dropped; the smallest is preselected', () => {
    const blocked = (): never => {
      throw new Error('blocked');
    };
    vi.stubGlobal('localStorage', { getItem: blocked, setItem: blocked, removeItem: blocked });
    try {
      expect(() => storeVariant('Qwen3.8-27B', XL.id)).not.toThrow();
      expect(pickedVariants()).toEqual({});
      expect(bases({}, [], null, pickedVariants()).find((row) => row.base === 'Qwen3.8-27B')!.preselect).toBe(M.id);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
