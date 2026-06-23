import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';
import { beforeAll, describe, expect, it } from 'vite-plus/test';

/**
 * End-to-end coherence test for the Gemma 4 E2B QAT (wNa8o8) checkpoint
 * (`google/gemma-4-E2B-it-qat-mobile-transformers`).
 *
 * Exercises the full public path that production traffic hits — `loadModel`
 * + `ChatSession.sendStream` greedy decode — against a converted MLX checkpoint
 * of the source wNa8o8 model. It guards the wNa8o8 importer + quantized lm_head
 * + KV-share-aware loader end-to-end: a regression in any of them (or in the
 * fused SDPA Metal kernel for head_dim=256/512) collapses greedy decode to a
 * degenerate single-token spew, which the assertions below reject.
 *
 * Opt-in / presence-gated (mirrors the `QWEN3_PAGED_PARITY_MODEL_PATH` Rust
 * `#[ignore]` convention): the suite runs only when a converted checkpoint is
 * found, so it never blocks CI on machines without the weights.
 *
 * To run locally:
 *   1. Convert the source checkpoint (one-time, ~90s):
 *        mlx convert \
 *          -i ~/.mlx-node/models/gemma-4-e2b-it-qat-mobile-transformers \
 *          -o .cache/models/gemma-4-e2b-qat-mlx \
 *          -m gemma4
 *   2. Run the suite:
 *        vp test __test__/models/gemma4-qat-e2e.test.ts
 *      (or set GEMMA4_QAT_MODEL_PATH to a converted checkpoint elsewhere).
 *
 * NOTE: a worktree-local `yarn build:native` can ship a broken `mlx.metallib`
 * whose fused SDPA kernel is wrong for head_dim=256/512 → garbage decode that
 * fails these assertions. CI builds a correct metallib; locally, swap the main
 * checkout's `packages/core/mlx.metallib` in after a worktree rebuild.
 */

/**
 * True only for a *converted* MLX checkpoint this runtime can actually load.
 *
 * Gating on mere directory existence is unsafe: a stale env var, the raw
 * wNa8o8 source dir (which also has config.json + model.safetensors), or a
 * partial cache restore would flip `skipIf` off and crash `beforeAll`'s
 * `loadModel` instead of skipping. We require all of:
 *   - a parseable config.json,
 *   - the top-level `quantization` block our converter writes (the source dir
 *     has only `quantization_config`, so this excludes it), and
 *   - the weights file (excludes a config-only partial restore).
 */
function isConvertedCheckpoint(dir: string): boolean {
  const cfgPath = resolve(dir, 'config.json');
  if (!existsSync(cfgPath)) return false;
  const hasWeights =
    existsSync(resolve(dir, 'model.safetensors')) || existsSync(resolve(dir, 'model.safetensors.index.json'));
  if (!hasWeights) return false;
  try {
    const cfg = JSON.parse(readFileSync(cfgPath, 'utf-8')) as { quantization?: unknown };
    return cfg.quantization != null;
  } catch {
    return false;
  }
}

/** Converted MLX checkpoint dir, or null when none is available. */
function findModelPath(): string | null {
  const env = process.env.GEMMA4_QAT_MODEL_PATH;
  if (env && isConvertedCheckpoint(env)) return env;
  // Canonical local convert target (see header). This dir is specific to this
  // checkpoint and gitignored, so it never coincidentally triggers in CI.
  const fallback = resolve(process.cwd(), '.cache/models/gemma-4-e2b-qat-mlx');
  if (isConvertedCheckpoint(fallback)) return fallback;
  return null;
}

/**
 * Fraction of the whitespace-stripped text covered by its most-repeated short
 * n-gram (n ∈ 1..4). A degenerate decode that loops a token/subword *without*
 * spaces ("ParisParis…", "10101010…") collapses to ~1.0 here; legitimate prose
 * stays well below 0.5. Returns 0 for short strings where the ratio is noise.
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

/**
 * Reject a degenerate decode. A broken SDPA / quant path collapses greedy
 * decode to a repeated unit; a healthy one produces varied text that
 * terminates cleanly. Guards both space-separated word loops and no-space
 * subword loops so the targeted single-token-spew regression can't slip
 * through on a whitespace technicality.
 */
function assertCoherent(text: string, finishReason: string, numTokens: number): void {
  const trimmed = text.trim();
  expect(trimmed.length).toBeGreaterThan(0);
  expect(['stop', 'length']).toContain(finishReason);
  expect(numTokens).toBeGreaterThan(0);

  const words = trimmed.split(/\s+/).filter(Boolean);
  if (words.length > 3) {
    // At least two distinct tokens, and no single token dominating the answer
    // (>80% of a long decode = a repetition loop).
    const counts = new Map<string, number>();
    for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
    expect(counts.size).toBeGreaterThan(1);
    const maxRepeat = Math.max(...counts.values());
    expect(maxRepeat).toBeLessThan(words.length * 0.8);
  }
  // Whitespace-independent repetition guard (catches no-space subword loops).
  expect(maxNgramDominance(trimmed)).toBeLessThan(0.8);
}

const modelPath = findModelPath();
const modelExists = modelPath !== null;
const imagePath = resolve(process.cwd(), 'examples/ocr.png');
const imageExists = existsSync(imagePath);

describe.skipIf(!modelExists)('Gemma 4 E2B QAT (wNa8o8) — end-to-end decode', () => {
  let session: ChatSession;

  beforeAll(async () => {
    if (!modelExists || !modelPath) return;
    const model = await loadModel(modelPath);
    session = new ChatSession(model as unknown as SessionCapableModel, {
      system: 'You are a helpful assistant. Be concise.',
    });
  }, 120_000);

  /** Greedy (T=0) turn on a reset session; returns the streamed text. */
  async function runTurn(
    prompt: string,
    images: Uint8Array[] | undefined,
    maxNewTokens = 80,
  ): Promise<{ text: string; finishReason: string; numTokens: number }> {
    let text = '';
    let finishReason = 'unknown';
    let numTokens = 0;
    for await (const event of session.sendStream(prompt, {
      ...(images !== undefined && { images }),
      config: { maxNewTokens, temperature: 0, reportPerformance: false },
    })) {
      if (event.done) {
        finishReason = event.finishReason;
        numTokens = event.numTokens;
      } else {
        text += event.text;
      }
    }
    // Reset between turns so each greedy decode is independent.
    await session.reset();
    return { text, finishReason, numTokens };
  }

  it('answers a factual question (capital of France → Paris)', async () => {
    const r = await runTurn('What is the capital of France?', undefined, 16);
    assertCoherent(r.text, r.finishReason, r.numTokens);
    expect(r.text.toLowerCase()).toContain('paris');
  });

  it('counts deterministically from 1 to 10', async () => {
    const r = await runTurn('Count from 1 to 10.', undefined, 48);
    assertCoherent(r.text, r.finishReason, r.numTokens);
    // The full sequence must appear; a degenerate decode never reaches "10".
    for (const n of ['1', '2', '5', '9', '10']) {
      expect(r.text).toContain(n);
    }
  });

  it('produces a coherent open-ended sentence', async () => {
    const r = await runTurn('Write one sentence about the ocean.', undefined, 80);
    assertCoherent(r.text, r.finishReason, r.numTokens);
    expect(r.text.toLowerCase()).toContain('ocean');
  });

  it.runIf(imageExists)('produces a coherent caption for a document image', async () => {
    const buf = readFileSync(imagePath);
    const bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    const r = await runTurn('Describe this image.', [bytes], 96);
    assertCoherent(r.text, r.finishReason, r.numTokens);
    // A working vision path yields a multi-word description, not one token.
    expect(r.text.trim().split(/\s+/).filter(Boolean).length).toBeGreaterThan(3);
  });
});
