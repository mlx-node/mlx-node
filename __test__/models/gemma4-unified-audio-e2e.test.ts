import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';
import { beforeAll, describe, expect, it } from 'vite-plus/test';

/**
 * Structural / smoke test for the unified Gemma 4 12B (`gemma4_unified`)
 * ENCODER-FREE AUDIO path — WAV -> mono 16 kHz f32 PCM (`decode_wav_to_pcm`)
 * -> `[n_frames, 640]` raw windows (`frames_from_pcm`) -> `embed_audio`
 * projection -> `masked_scatter` at the `<|audio|>` (258881) positions into the
 * scaled token stream -> greedy decode.
 *
 * This exercises the public `loadModel` + `ChatSession.sendStream({ audio })`
 * turn against the real 24GB checkpoint. It guards the WAV decode, the audio
 * token expansion (`boa + audio×n_frames + eoa`), the `embed_audio.*` weight
 * load, and the `mask_count == feature_count` assertion (audio frame count) in
 * the merge. A regression in any of those either fails to load, panics on the
 * count mismatch, or collapses greedy decode to a degenerate repeat — all of
 * which the assertions below reject.
 *
 * Audio turns are CAUSAL (no bidirectional overlay — the overlay is disabled
 * when audio tokens are present; see `vision_overlay_active`).
 *
 * Presence-gated on BOTH the checkpoint dir and the fixture audio, so CI
 * without the (multi-gigabyte) weights or the clip auto-skips.
 *
 * NOTE: a worktree-local `yarn build:native` can ship a broken `mlx.metallib`
 * (~63KB smaller) whose fused SDPA kernel is wrong for head_dim=256/512 ->
 * garbage decode. CI builds a correct metallib; locally, swap the main
 * checkout's `packages/core/mlx.metallib` in (md5 23044b4f...) after a worktree
 * rebuild.
 *
 * To run locally:
 *   vp test __test__/models/gemma4-unified-audio-e2e.test.ts
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
const audioPath = resolve(process.cwd(), 'examples/audio-ask-16k.wav');
const audioExists = existsSync(audioPath);
const imagePath = resolve(process.cwd(), 'examples/ocr.png');
const imageExists = existsSync(imagePath);

describe.skipIf(!modelExists || !audioExists)('Gemma 4 unified 12B — audio turn (encoder-free)', () => {
  let session: ChatSession;

  beforeAll(async () => {
    if (!modelExists || !modelPath) return;
    const model = await loadModel(modelPath);
    session = new ChatSession(model as unknown as SessionCapableModel, {
      system: 'You are a helpful assistant. Be concise.',
    });
  }, 300_000);

  /** Greedy (T=0) audio turn on a reset session; returns the streamed text. */
  async function runAudioTurn(
    prompt: string,
    audio: Uint8Array[],
    maxNewTokens = 96,
  ): Promise<{ text: string; finishReason: string; numTokens: number }> {
    let text = '';
    let finishReason = 'unknown';
    let numTokens = 0;
    for await (const event of session.sendStream(prompt, {
      audio,
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

  it('runs an audio turn end-to-end and produces a coherent, non-degenerate response', async () => {
    const buf = readFileSync(audioPath);
    const bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    // A mask_count != feature_count regression throws here (rejects the turn);
    // a broken decode collapses to a repeat that assertCoherent rejects.
    const r = await runAudioTurn('Transcribe or describe this audio.', [bytes], 96);
    // eslint-disable-next-line no-console
    console.log('[gemma4-audio-e2e] model response:', JSON.stringify(r.text.trim()));
    assertCoherent(r.text, r.finishReason, r.numTokens);
    // A working audio path yields a multi-word response, not one token.
    expect(r.text.trim().split(/\s+/).filter(Boolean).length).toBeGreaterThan(3);
  });

  // Mixed image+audio turn: proves the combined turn builds BOTH scatters and
  // carries both token runs, and runs causally (the bidirectional vision
  // overlay is disabled when audio is present — see `vision_overlay_active`).
  it.skipIf(!imageExists)('runs a combined image+audio turn causally and coherently', async () => {
    const abuf = readFileSync(audioPath);
    const audio = new Uint8Array(abuf.buffer, abuf.byteOffset, abuf.byteLength);
    const ibuf = readFileSync(imagePath);
    const image = new Uint8Array(ibuf.buffer, ibuf.byteOffset, ibuf.byteLength);

    let text = '';
    let finishReason = 'unknown';
    let numTokens = 0;
    for await (const event of session.sendStream('Use the image and the audio to answer briefly.', {
      images: [image],
      audio: [audio],
      config: { maxNewTokens: 96, temperature: 0, reportPerformance: false },
    })) {
      if (event.done) {
        finishReason = event.finishReason;
        numTokens = event.numTokens;
      } else {
        text += event.text;
      }
    }
    await session.reset();
    // eslint-disable-next-line no-console
    console.log('[gemma4-audio-e2e] mixed image+audio response:', JSON.stringify(text.trim()));
    assertCoherent(text, finishReason, numTokens);
  });
});
