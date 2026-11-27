import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vite-plus/test';

import { CHAT_FAMILY_IDS, familyTraitsFor, THINKING_LEVELS } from '../../packages/lm/src/family-data.js';
import { AGENT_HOST_THINKING } from '../../packages/server/src/host/agent-thinking.js';
import {
  levelsFor,
  reasoningEffortsFor,
  thinkingFor,
  type LevelSetting,
} from '../../packages/server/src/host/thinking-levels.js';

/** The engine table: levels shown, and what each non-default level sets. */
const BUDGET_FAMILY = {
  levels: ['none', 'low', 'medium', 'high'],
  settings: {
    none: { reasoningEffort: 'none' },
    low: { thinkingTokenBudget: 1024 },
    medium: { thinkingTokenBudget: 4096 },
  },
};
/** A family with a checkpoint template that reads `reasoning_effort`: each budget goes with its level's word. */
const BUDGET_EFFORT_FAMILY = {
  levels: ['none', 'low', 'medium', 'high'],
  settings: {
    none: { reasoningEffort: 'none' },
    low: { reasoningEffort: 'low', thinkingTokenBudget: 1024 },
    medium: { reasoningEffort: 'medium', thinkingTokenBudget: 4096 },
  },
};
const ALWAYS_ON = {
  levels: ['low', 'high'],
  settings: { low: { reasoningEffort: 'low', thinkingTokenBudget: 256 } },
};
const TABLE: Record<string, { levels: string[]; settings: Record<string, LevelSetting> } | null> = {
  qwen3: BUDGET_FAMILY,
  qwen3_5: BUDGET_EFFORT_FAMILY,
  qwen3_5_moe: BUDGET_FAMILY,
  qwen4_exp: BUDGET_EFFORT_FAMILY,
  nemotron_h: BUDGET_FAMILY,
  deepseek_v41: BUDGET_EFFORT_FAMILY,
  hy_v4_flash: {
    levels: ['none', 'low', 'high', 'max'],
    settings: { none: { reasoningEffort: 'none' }, low: { reasoningEffort: 'low' }, max: { reasoningEffort: 'max' } },
  },
  inkling2: {
    levels: ['none', 'low', 'medium', 'high'],
    settings: {
      none: { reasoningEffort: 'none' },
      low: { reasoningEffort: 'low' },
      medium: { reasoningEffort: 'medium' },
    },
  },
  k2_horizon: {
    levels: ['low', 'medium', 'high'],
    settings: { low: { reasoningEffort: 'low', thinkingTokenBudget: 256 }, medium: { reasoningEffort: 'medium' } },
  },
  lfm2: ALWAYS_ON,
  lfm2_moe: ALWAYS_ON,
  muse_glimmer: ALWAYS_ON,
  gemma4: null,
};

/** The values the agent's `ReasoningEffort` parses (`xai-grok-sampling-types/src/types.rs`). */
const AGENT_EFFORTS = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'];

describe('thinking levels: the family table', () => {
  it('every chat family has a row, and the table names exactly the chat families', () => {
    for (const id of CHAT_FAMILY_IDS) expect(familyTraitsFor(id)?.thinkingLevels, id).toBeDefined();
    expect(Object.keys(TABLE).sort()).toEqual([...CHAT_FAMILY_IDS].sort());
  });

  it.each(Object.keys(TABLE))('%s: the levels, the default High, and what each level sets', (family) => {
    const want = TABLE[family]!;
    if (want === null) {
      expect(levelsFor(family)).toBeNull();
      return;
    }
    expect(levelsFor(family)).toEqual({ levels: want.levels, default: 'high' });
    for (const level of want.levels) {
      expect(thinkingFor(family, level), `${family} ${level}`).toEqual(level === 'high' ? {} : want.settings[level]);
    }
  });

  it('the default level of every family sets nothing, so its request runs as one that names no level', () => {
    for (const id of CHAT_FAMILY_IDS) {
      const levels = levelsFor(id);
      if (levels !== null) expect(thinkingFor(id, levels.default), id).toEqual({});
    }
  });

  // inkling2 defaults to 0.9 (`DEFAULT_REASONING_EFFORT`), k2_horizon's template to `default('high')`.
  it('inkling2 and k2_horizon run at High when a request names no level', () => {
    expect(levelsFor('inkling2')!.default).toBe('high');
    expect(levelsFor('k2_horizon')!.default).toBe('high');
  });

  it('no level thinks past the agent hosts budget', () => {
    for (const id of CHAT_FAMILY_IDS) {
      for (const level of levelsFor(id)?.levels ?? []) {
        const budget = thinkingFor(id, level)!.thinkingTokenBudget;
        if (budget !== undefined) expect(budget, `${id} ${level}`).toBeLessThanOrEqual(AGENT_HOST_THINKING.tokenBudget);
      }
    }
  });

  it('gemma4 offers none: no levels, no /v1/models efforts, every value maps as without levels', () => {
    expect(levelsFor('gemma4')).toBeNull();
    expect(reasoningEffortsFor('gemma4')).toBeUndefined();
    for (const value of THINKING_LEVELS) expect(thinkingFor('gemma4', value)).toBeNull();
  });

  it('a value the family does not list, a foreign word and a non-chat family give no setting', () => {
    expect(thinkingFor('qwen3_5', 'max')).toBeNull();
    expect(thinkingFor('qwen3_5', 'xhigh')).toBeNull();
    expect(thinkingFor('hy_v4_flash', 'medium')).toBeNull();
    expect(thinkingFor('lfm2', 'none')).toBeNull();
    for (const word of ['minimal', 'bogus', '__proto__', 'toString', 'constructor', '']) {
      expect(thinkingFor('qwen3_5', word), word).toBeNull();
    }
    expect(levelsFor('harrier')).toBeNull();
    expect(levelsFor('clef')).toBeNull();
    expect(levelsFor('jina_reranker')).toBeNull();
    expect(levelsFor('not-a-family')).toBeNull();
    expect(thinkingFor('not-a-family', 'low')).toBeNull();
  });

  it('the /v1/models rows: labels, one default, values the agent parses, never minimal', () => {
    expect(reasoningEffortsFor('qwen3_5')).toEqual([
      { value: 'none', label: 'Off' },
      { value: 'low', label: 'Low' },
      { value: 'medium', label: 'Medium' },
      { value: 'high', label: 'High', default: true },
    ]);
    expect(reasoningEffortsFor('hy_v4_flash')!.map((row) => row.label)).toEqual(['Off', 'Low', 'High', 'Max']);
    for (const id of CHAT_FAMILY_IDS) {
      const rows = reasoningEffortsFor(id);
      if (rows === undefined) continue;
      expect(
        rows.filter((row) => row.default === true),
        id,
      ).toHaveLength(1);
      for (const row of rows) {
        expect(AGENT_EFFORTS).toContain(row.value);
        expect(row.value).not.toBe('minimal');
      }
    }
  });
});

/** Checkpoint templates kept in the repo (`crates/mlx-core/src/tokenizer/fixtures`) and the family that loads each. */
const TEMPLATES = [
  { family: 'qwen3_5', file: 'qwen3.8-27b/chat_template.jinja', readsEffort: true },
  { family: 'qwen3_5', file: 'qwen3.6-27b/chat_template.jinja', readsEffort: false },
  { family: 'deepseek_v41', file: 'hf/deepseek-v4.1-06cada875e3a.jinja', readsEffort: true },
  { family: 'hy_v4_flash', file: 'hf/hy-v4-flash-09a8ba8fb355/chat_template.jinja', readsEffort: true },
  { family: 'nemotron_h', file: 'nemotron-3.5-lightning-58933db77d30.jinja', readsEffort: false },
] as const;

const template = (file: string): string =>
  readFileSync(new URL(`../../crates/mlx-core/src/tokenizer/fixtures/${file}`, import.meta.url), 'utf8');

const names = (text: string, word: string): boolean => text.includes(`'${word}'`) || text.includes(`"${word}"`);

describe('thinking levels: the checkpoint templates', () => {
  it('a template that reads reasoning_effort gets a word it names with every level that sets one or a budget', () => {
    for (const { family, file, readsEffort } of TEMPLATES) {
      const text = template(file);
      expect(text.includes('reasoning_effort'), file).toBe(readsEffort);
      if (!readsEffort) continue;
      const levels = levelsFor(family)!;
      for (const level of levels.levels) {
        if (level === levels.default || level === 'none') continue;
        const setting = thinkingFor(family, level)!;
        expect(setting.reasoningEffort, `${file} ${level}`).toBeDefined();
        expect(names(text, setting.reasoningEffort!), `${file} ${level}`).toBe(true);
      }
    }
  });

  // Qwen3.8 reads `reasoning_effort|default('xhigh')` (the High word), so a budget sent alone would run there.
  it('Qwen3.8 Low and Medium send the template words with their budgets; High sends nothing and runs at xhigh', () => {
    const text = template('qwen3.8-27b/chat_template.jinja');
    expect(text.includes("reasoning_effort|default('xhigh')")).toBe(true);
    expect(thinkingFor('qwen3_5', 'low')).toEqual({ reasoningEffort: 'low', thinkingTokenBudget: 1024 });
    expect(thinkingFor('qwen3_5', 'medium')).toEqual({ reasoningEffort: 'medium', thinkingTokenBudget: 4096 });
    expect(thinkingFor('qwen3_5', 'high')).toEqual({});
  });

  it('a family none of whose templates reads an effort keeps budgets alone', () => {
    expect(thinkingFor('nemotron_h', 'low')).toEqual({ thinkingTokenBudget: 1024 });
    expect(thinkingFor('nemotron_h', 'medium')).toEqual({ thinkingTokenBudget: 4096 });
  });
});
