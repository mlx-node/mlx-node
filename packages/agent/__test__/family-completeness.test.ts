/**
 * Runtime backstop behind the family-data row type's compile-time gate: every
 * chat family must carry traits and an agent-servable preset, the paged
 * override manager's default policy must cover it, and the chat /
 * non-generative split must partition the whole registry. Survives type-level
 * erosion (an `as any` on a row) and guards the derived helpers themselves.
 */

import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  agentLaunchPresetFor,
  CHAT_FAMILY_IDS,
  familyTraitsFor,
  MODEL_FAMILY_DATA,
  NON_GENERATIVE_FAMILY_IDS,
  PagedConfigOverrideManager,
  serverLaunchPresetFor,
} from '@mlx-node/lm';
import { describe, expect, it } from 'vite-plus/test';

describe('family-data completeness', () => {
  it.each([...CHAT_FAMILY_IDS])('chat family %s has traits and an agent launch preset', (id) => {
    expect(familyTraitsFor(id)).toBeDefined();
    expect(agentLaunchPresetFor(id)).toBeDefined();
    expect(familyTraitsFor(id)!.fallbackContextWindow).toBeGreaterThan(0);
    expect(agentLaunchPresetFor(id)!.maxOutputTokens).toBeGreaterThan(0);
  });

  it('preserves the lfm2_moe preset split: agent-served, skipped by server discovery', () => {
    expect(serverLaunchPresetFor('lfm2_moe')).toBeUndefined();
    const agentPreset = agentLaunchPresetFor('lfm2_moe');
    expect(agentPreset).toBeDefined();
    // LiquidAI's MoE card values — NOT the dense lfm2 preset.
    expect(agentPreset!.sampling.temperature).toBe(0.2);
    expect(agentPreset!.sampling.topK).toBe(80);
  });

  it('forces the paged overlay by default for every chat family', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-family-completeness-'));
    const manager = new PagedConfigOverrideManager();
    try {
      for (const id of CHAT_FAMILY_IDS) {
        const dir = join(root, id);
        await mkdir(dir, { recursive: true });
        await writeFile(join(dir, 'config.json'), JSON.stringify({ model_type: id, use_block_paged_cache: false }));
        const resolved = await manager.resolve(dir, id);
        expect(resolved, id).not.toBe(dir);
        const config = JSON.parse(await readFile(join(resolved, 'config.json'), 'utf-8')) as Record<string, unknown>;
        expect(config.use_block_paged_cache, id).toBe(true);
      }
    } finally {
      await manager.cleanup();
      await rm(root, { recursive: true, force: true });
    }
  });

  it('splits chat and non-generative families disjointly over the whole registry', () => {
    const chat = new Set<string>(CHAT_FAMILY_IDS);
    for (const id of NON_GENERATIVE_FAMILY_IDS) {
      expect(chat.has(id), id).toBe(false);
    }
    const union = [...chat, ...NON_GENERATIVE_FAMILY_IDS].sort();
    expect(union).toEqual(MODEL_FAMILY_DATA.map((row) => row.id).sort());
  });

  it('keeps the non-generative set at the embedding/vlm families', () => {
    expect([...NON_GENERATIVE_FAMILY_IDS].sort()).toEqual(['harrier', 'internvl_chat', 'qianfan-ocr']);
  });
});
