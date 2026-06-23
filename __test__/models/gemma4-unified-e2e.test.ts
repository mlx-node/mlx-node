import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

import { ChatSession, loadSession } from '@mlx-node/lm';
import { beforeAll, describe, expect, it } from 'vite-plus/test';

/**
 * End-to-end coherence test for the unified Gemma 4 12B checkpoint
 * (`gemma4_unified`, `Gemma4UnifiedForConditionalGeneration`) loaded as a
 * TEXT model.
 *
 * Exercises the public `loadSession` + `ChatSession.sendStream` greedy path
 * against the real 24GB checkpoint. It guards the unified→`gemma4` dispatch,
 * the `is_unified` config detection, and the load-time handling of the
 * multimodal embedder weights — the unified checkpoint KEEPS `vision_embedder.*`
 * / `embed_vision.*` / `embed_audio.*` (validated, then loaded into the
 * embedders) while text decode ignores them. A regression in any of them either
 * fails to load or collapses greedy decode to a degenerate repeat, which the
 * assertions below reject.
 *
 * Presence-gated: the suite runs only when the checkpoint directory is found,
 * so CI without the (multi-gigabyte) weights auto-skips.
 *
 * To run locally:
 *   vp test __test__/models/gemma4-unified-e2e.test.ts
 *   (or set GEMMA4_UNIFIED_MODEL_PATH to a checkpoint dir elsewhere).
 */

/** True for a directory that looks like a loadable gemma4_unified checkpoint. */
function isUnifiedCheckpoint(dir: string): boolean {
  const cfg = resolve(dir, 'config.json');
  if (!existsSync(cfg)) return false;
  const hasWeights =
    existsSync(resolve(dir, 'model.safetensors')) || existsSync(resolve(dir, 'model.safetensors.index.json'));
  return hasWeights;
}

/** Checkpoint dir, or null when none is available. */
function findModelPath(): string | null {
  const env = process.env.GEMMA4_UNIFIED_MODEL_PATH;
  if (env && isUnifiedCheckpoint(env)) return env;
  const fallback = resolve(process.cwd(), '.cache/models/gemma-4-12b-it');
  if (isUnifiedCheckpoint(fallback)) return fallback;
  return null;
}

/**
 * Fraction of the whitespace-stripped text covered by its most-repeated short
 * n-gram (n in 1..4). A degenerate decode that loops a token/subword collapses
 * to ~1.0 here; legitimate prose stays well below 0.5.
 */
function maxNgramDominance(text: string): number {
  const s = text.replace(/\s+/g, '');
  if (s.length < 12) return 0;
  let worst = 0;
  for (let n = 1; n <= 4; n++) {
    const counts = new Map<string, number>();
    for (let i = 0; i + n <= s.length; i++) {
      const g = s.slice(i, i + n);
      counts.set(g, (counts.get(g) ?? 0) + 1);
    }
    const maxCount = Math.max(...counts.values());
    worst = Math.max(worst, (maxCount * n) / s.length);
  }
  return worst;
}

/** Reject a degenerate decode; a healthy one is varied and terminates cleanly. */
function assertCoherent(text: string, finishReason: string, numTokens: number): void {
  const trimmed = text.trim();
  expect(trimmed.length).toBeGreaterThan(0);
  expect(['stop', 'length']).toContain(finishReason);
  expect(numTokens).toBeGreaterThan(0);

  const words = trimmed.split(/\s+/).filter(Boolean);
  if (words.length > 3) {
    const counts = new Map<string, number>();
    for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
    expect(counts.size).toBeGreaterThan(1);
    const maxRepeat = Math.max(...counts.values());
    expect(maxRepeat).toBeLessThan(words.length * 0.8);
  }
  expect(maxNgramDominance(trimmed)).toBeLessThan(0.8);
}

const modelPath = findModelPath();
const modelExists = modelPath !== null;

describe.skipIf(!modelExists)('Gemma 4 unified 12B — text-only end-to-end decode', () => {
  let session: ChatSession;

  beforeAll(async () => {
    if (!modelExists || !modelPath) return;
    session = await loadSession(modelPath);
  }, 300_000);

  /** Greedy (T=0) turn on a reset session; returns the streamed text. */
  async function runTurn(
    prompt: string,
    maxNewTokens = 64,
  ): Promise<{ text: string; finishReason: string; numTokens: number }> {
    let text = '';
    let finishReason = 'unknown';
    let numTokens = 0;
    for await (const event of session.sendStream(prompt, {
      config: { maxNewTokens, temperature: 0, reportPerformance: false },
    })) {
      if (event.done) {
        finishReason = event.finishReason;
        numTokens = event.numTokens;
      } else {
        text += event.text;
      }
    }
    await session.reset();
    return { text, finishReason, numTokens };
  }

  it('answers a factual question (capital of France -> Paris)', async () => {
    const r = await runTurn('What is the capital of France?', 24);
    assertCoherent(r.text, r.finishReason, r.numTokens);
    expect(r.text.toLowerCase()).toContain('paris');
  });

  it('produces a coherent open-ended sentence about the ocean', async () => {
    const r = await runTurn('Write one sentence about the ocean.', 64);
    assertCoherent(r.text, r.finishReason, r.numTokens);
    expect(r.text.toLowerCase()).toContain('ocean');
  });
});
