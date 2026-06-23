import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';
import { beforeAll, describe, expect, it } from 'vite-plus/test';

/**
 * Structural / smoke test for the unified Gemma 4 12B (`gemma4_unified`)
 * ENCODER-FREE vision path — image -> patches -> VisionEmbedder -> embed_vision
 * -> masked_scatter into the token stream -> greedy decode.
 *
 * This exercises the public `loadModel` + `ChatSession.sendStream` image turn
 * against the real 24GB checkpoint. It guards the unified vision config + the
 * unified image processor (model_patch_size=48 patchify + xy position ids) +
 * the `vision_embedder.*` / `embed_vision.*` weight load + the
 * `mask_count == feature_count` assertion (correct soft-token count) inside the
 * merge. A regression in any of those either fails to load, panics on the
 * count mismatch, or collapses greedy decode to a degenerate repeat — all of
 * which the assertions below reject.
 *
 * This is a Phase-2a coherence/structural check using the EXISTING causal
 * vision prefill; exact mlx-vlm golden parity (the bidirectional-attention
 * overlay + PIL-bicubic resize match) is a separate Phase-2b concern. The
 * resize filter here is the `image` crate's CatmullRom (bicubic-equivalent);
 * any pixel-level divergence from PIL bicubic is a 2b parity item, not a 2a
 * blocker.
 *
 * Presence-gated on BOTH the checkpoint dir and the fixture image, so CI
 * without the (multi-gigabyte) weights or the image auto-skips.
 *
 * NOTE: a worktree-local `yarn build:native` can ship a broken `mlx.metallib`
 * (~63KB smaller) whose fused SDPA kernel is wrong for head_dim=256/512 ->
 * garbage decode. CI builds a correct metallib; locally, swap the main
 * checkout's `packages/core/mlx.metallib` in after a worktree rebuild.
 *
 * To run locally:
 *   vp test __test__/models/gemma4-unified-vision-e2e.test.ts
 *   (or set GEMMA4_UNIFIED_MODEL_PATH to a checkpoint dir elsewhere).
 */

function isUnifiedCheckpoint(dir: string): boolean {
  const cfg = resolve(dir, 'config.json');
  if (!existsSync(cfg)) return false;
  const hasWeights =
    existsSync(resolve(dir, 'model.safetensors')) || existsSync(resolve(dir, 'model.safetensors.index.json'));
  return hasWeights;
}

function findModelPath(): string | null {
  const env = process.env.GEMMA4_UNIFIED_MODEL_PATH;
  if (env && isUnifiedCheckpoint(env)) return env;
  const fallback = resolve(process.cwd(), '.cache/models/gemma-4-12b-it');
  if (isUnifiedCheckpoint(fallback)) return fallback;
  return null;
}

/**
 * Fraction of the whitespace-stripped text covered by its most-repeated short
 * n-gram (n in 1..4). A degenerate decode collapses to ~1.0; prose stays low.
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
const imagePath = resolve(process.cwd(), 'examples/ocr.png');
const imageExists = existsSync(imagePath);

describe.skipIf(!modelExists || !imageExists)('Gemma 4 unified 12B — vision image turn (encoder-free)', () => {
  let session: ChatSession;

  beforeAll(async () => {
    if (!modelExists || !modelPath) return;
    const model = await loadModel(modelPath);
    session = new ChatSession(model as unknown as SessionCapableModel, {
      system: 'You are a helpful assistant. Be concise.',
    });
  }, 300_000);

  /** Greedy (T=0) image turn on a reset session; returns the streamed text. */
  async function runImageTurn(
    prompt: string,
    images: Uint8Array[],
    maxNewTokens = 64,
  ): Promise<{ text: string; finishReason: string; numTokens: number }> {
    let text = '';
    let finishReason = 'unknown';
    let numTokens = 0;
    for await (const event of session.sendStream(prompt, {
      images,
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

  it('runs an image turn end-to-end and produces a coherent, non-degenerate description', async () => {
    const buf = readFileSync(imagePath);
    const bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    // A mask_count != feature_count regression throws here (rejects the turn);
    // a broken decode collapses to a repeat that assertCoherent rejects.
    const r = await runImageTurn('Describe this image.', [bytes], 96);
    assertCoherent(r.text, r.finishReason, r.numTokens);
    // A working vision path yields a multi-word description, not one token.
    expect(r.text.trim().split(/\s+/).filter(Boolean).length).toBeGreaterThan(3);
  });
});
