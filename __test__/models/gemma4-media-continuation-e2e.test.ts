import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';
import { beforeAll, describe, expect, it } from 'vite-plus/test';

/**
 * Media -> text continuation parity (Gemma 4).
 *
 * A text follow-up may warm-continue only after a PURE-IMAGE turn. The vision
 * finalizer keeps the global paged KV live and arms
 * `media_session_continuable` only when the physical sliding-anchor checkpoint
 * stored successfully. Image identity is part of every affected paged block
 * key, so a positive `cachedTokens` value proves reuse of the exact ordered
 * image lineage rather than a token-only placeholder match.
 *
 * KV sharing does not make an image turn ineligible. E2B's
 * `SharedOnSliding` alias slots intentionally own no private K/V and are skipped
 * by checkpoint readiness/storage; their physical non-shared sliding anchors
 * carry the state for both layers. Both standard SigLIP images (E2B) and unified
 * bidirectional-vision images therefore warm-continue on faithful live K/V.
 *
 * Audio and mixed-media identity is not represented in the per-block keys yet.
 * Those turns deliberately use `skip_lookup` with a zero cache-hit ceiling, do
 * not publish reusable prefix checkpoints, and leave the continuation marker
 * off. The native text delta then returns the typed restart error; `ChatSession`
 * catches it before any token is emitted and cold-replays the complete history.
 *
 * ## What this file asserts
 *
 *  - NON-UNIFIED and UNIFIED image: `cachedTokens > 0` plus a coherent answer.
 *    E2B additionally pins its known byte-exact warm-vs-cold golden. Unified
 *    image does not require byte equality: its cache-hit and full-prefill
 *    kernels can differ by a deterministic ~1-ULP BF16 reduction-order drift
 *    that flips a near-tie argmax while preserving a coherent answer.
 *  - AUDIO: `cachedTokens == 0`, proving the follow-up used the deliberate cold
 *    replay, plus exact FINAL-answer parity against a direct cold replay of the
 *    same history after resetting the loaded model.
 *
 * Greedy T=0 throughout. Presence-gated on the converted checkpoints + fixtures,
 * mirroring the existing gemma4 e2e tests.
 *
 * NOTE: a worktree-local `yarn build:native` can ship a broken `mlx.metallib`
 * (~63KB smaller) whose fused SDPA kernel is wrong for head_dim=256/512 ->
 * garbage decode. CI builds a correct metallib; locally, swap the main
 * checkout's `packages/core/mlx.metallib` (md5 23044b4f...) in after a worktree
 * rebuild, or these parity assertions can flake on a broken kernel.
 */

interface UserMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  images?: Uint8Array[];
  audio?: Uint8Array[];
}

function hasWeights(dir: string): boolean {
  return existsSync(resolve(dir, 'model.safetensors')) || existsSync(resolve(dir, 'model.safetensors.index.json'));
}

function findFirst(dirs: string[]): string | null {
  for (const d of dirs) {
    const abs = resolve(process.cwd(), d);
    if (existsSync(resolve(abs, 'config.json')) && hasWeights(abs)) return abs;
  }
  return null;
}

const SYSTEM = 'You are a helpful assistant. Be concise.';

const unifiedModelPath =
  process.env.GEMMA4_UNIFIED_MODEL_PATH && hasWeights(process.env.GEMMA4_UNIFIED_MODEL_PATH)
    ? process.env.GEMMA4_UNIFIED_MODEL_PATH
    : findFirst(['.cache/models/gemma-4-12b-it']);

const imageModelPath =
  process.env.GEMMA4_NONUNIFIED_MODEL_PATH && hasWeights(process.env.GEMMA4_NONUNIFIED_MODEL_PATH)
    ? process.env.GEMMA4_NONUNIFIED_MODEL_PATH
    : findFirst(['.cache/models/gemma-4-e2b-it', '.cache/models/gemma-4-e2b-it-mlx']);

const audioPath = resolve(process.cwd(), 'examples/audio-ask-16k.wav');
const audioExists = existsSync(audioPath);
const imagePath = resolve(process.cwd(), 'examples/ocr.png');
const imageExists = existsSync(imagePath);

function readBytes(p: string): Uint8Array {
  const buf = readFileSync(p);
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

/** Run one streamed turn on a ChatSession; returns the accumulated reply. */
async function streamTurn(
  session: ChatSession,
  prompt: string,
  opts: { images?: Uint8Array[]; audio?: Uint8Array[] } = {},
  maxNewTokens = 48,
): Promise<{ rawText: string; parsedText: string; numTokens: number; finishReason: string; cachedTokens?: number }> {
  let rawText = '';
  let parsedText = '';
  let numTokens = 0;
  let finishReason = 'unknown';
  let cachedTokens: number | undefined;
  for await (const event of session.sendStream(prompt, {
    ...(opts.images && { images: opts.images }),
    ...(opts.audio && { audio: opts.audio }),
    config: { maxNewTokens, temperature: 0, reportPerformance: false },
  })) {
    if (event.done) {
      finishReason = event.finishReason;
      numTokens = event.numTokens;
      rawText = event.rawText;
      cachedTokens = event.cachedTokens;
    } else if (event.isReasoning !== true) {
      parsedText += event.text;
    }
  }
  return { rawText, parsedText, numTokens, finishReason, cachedTokens };
}

/**
 * COLD reference: an already-reset model + manually-built media-bearing history
 * driven through native `chatSessionStart`, cold-prefilling the WHOLE
 * conversation (system + media turn-1 + turn-1 reply + the text delta). Reusing
 * the loaded model avoids holding a second multi-gigabyte checkpoint. Returns
 * turn-2's reply. `assistantTurn1` must be the parsed assistant content committed
 * by `ChatSession`; the Gemma template strips prior reasoning on both paths.
 */
async function coldReplayTurn2(
  model: SessionCapableModel,
  media: { images?: Uint8Array[]; audio?: Uint8Array[] },
  prompt1: string,
  assistantTurn1: string,
  prompt2: string,
  maxNewTokens: number,
): Promise<{ rawText: string; text: string; numTokens: number }> {
  const userTurn1: UserMessage = { role: 'user', content: prompt1 };
  if (media.images) userTurn1.images = media.images;
  if (media.audio) userTurn1.audio = media.audio;
  const history: UserMessage[] = [
    { role: 'system', content: SYSTEM },
    userTurn1,
    { role: 'assistant', content: assistantTurn1 },
    { role: 'user', content: prompt2 },
  ];
  const r = await model.chatSessionStart(history, {
    maxNewTokens,
    temperature: 0,
    reportPerformance: false,
  });
  return { rawText: r.rawText, text: r.text, numTokens: r.numTokens };
}

// -- NON-UNIFIED image continuation (e2b, KV-shared): alias slots are skipped
//    while their physical sliding anchors checkpoint the image-feature K/V, so
//    the media->text follow-up warm-continues on the exact image lineage. --
describe.skipIf(!imageModelPath || !imageExists)(
  'Gemma 4 — WARM non-unified image->text continuation parity (KV-shared)',
  () => {
    let model: SessionCapableModel;

    beforeAll(async () => {
      if (!imageModelPath) return;
      model = (await loadModel(imageModelPath)) as unknown as SessionCapableModel;
    }, 300_000);

    it('warm text delta reuses the exact image lineage and matches the cold final answer', async () => {
      const images = [readBytes(imagePath)];
      const prompt1 = 'Describe this image.';
      const prompt2 = 'What is the main color?';
      const maxNew = 40;

      const session = new ChatSession(model, { system: SYSTEM });
      const turn1 = await streamTurn(session, prompt1, { images }, 48);
      expect(turn1.rawText.length).toBeGreaterThan(0);

      // E2B is KV-shared (`num_kv_shared_layers=20`), but only the alias slots
      // lack private K/V. Their physical sliding anchors store the checkpoint;
      // the exact image hash + ordered positions keep the live global blocks on
      // the same image-aware lineage for this delta.
      const warm = await streamTurn(session, prompt2, {}, maxNew);
      await session.reset();

      const cold = await coldReplayTurn2(model, { images }, prompt1, turn1.parsedText, prompt2, maxNew);

      // eslint-disable-next-line no-console
      console.log('[gemma4-cont-image] warm:', JSON.stringify(warm.parsedText), 'cold:', JSON.stringify(cold.text));
      // The delta path reports the full prior history reused by construction.
      // The image-aware block keys and exact-position gate make this a stronger
      // signal than a token-only prefix hit.
      expect(warm.cachedTokens ?? 0).toBeGreaterThan(0);
      expect(warm.parsedText.trim()).toBe(cold.text.trim());
      expect(warm.rawText).toBe(cold.rawText);
      expect(warm.numTokens).toBe(cold.numTokens);
      expect(warm.finishReason === 'stop' || warm.finishReason === 'length').toBe(true);
      expect(warm.numTokens).toBeGreaterThan(0);
      const words = warm.parsedText.trim().split(/\s+/).filter(Boolean);
      expect(words.length).toBeGreaterThan(3);
      // No degenerate single-token loop.
      const counts = new Map<string, number>();
      for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
      expect(Math.max(...counts.values())).toBeLessThan(words.length * 0.8);
    });
  },
);

// -- AUDIO continuation: audio identity is not part of paged block keys, so the
//    follow-up cold-replays the full history and must match a direct cold
//    reference at the final-answer boundary. --
describe.skipIf(!unifiedModelPath || !audioExists)('Gemma 4 — COLD audio->text continuation parity', () => {
  let model: SessionCapableModel;

  beforeAll(async () => {
    if (!unifiedModelPath) return;
    model = (await loadModel(unifiedModelPath)) as unknown as SessionCapableModel;
  }, 300_000);

  it('text follow-up cold-replays full history and matches a direct cold replay', async () => {
    const audio = [readBytes(audioPath)];
    const prompt1 = 'Briefly describe this audio.';
    const prompt2 = 'In one word, what language is it?';
    const maxNew = 40;

    const session = new ChatSession(model, { system: SYSTEM });
    const turn1 = await streamTurn(session, prompt1, { audio }, 40);
    expect(turn1.rawText.length).toBeGreaterThan(0);
    const replayed = await streamTurn(session, prompt2, {}, maxNew);
    await session.reset();

    // The native media guard rejected the delta before emitting tokens and the
    // TS session replayed full history. Audio/mixed preparation deliberately
    // disables cache lookup, so the replay must report zero cached tokens.
    expect(replayed.cachedTokens ?? 0).toBe(0);

    const cold = await coldReplayTurn2(model, { audio }, prompt1, turn1.parsedText, prompt2, maxNew);

    // eslint-disable-next-line no-console
    console.log(
      '[gemma4-cont-audio] replayed:',
      JSON.stringify(replayed.parsedText),
      'cold:',
      JSON.stringify(cold.text),
    );
    // The ChatSession replay and the direct cold replay render the same history;
    // greedy decoding must therefore agree at the parsed final-answer boundary.
    expect(replayed.parsedText.trim()).toBe(cold.text.trim());
    expect(replayed.parsedText.trim().length).toBeGreaterThan(0);
  });
});

// -- UNIFIED image continuation: warm continues. The 12B is non-KV-shared
//    (`num_kv_shared_layers=0`), so every sliding layer stores real K/V -> the
//    finalize stored/live gate arms the marker and the text delta hits
//    `state="live"`. The warm delta routes through the GENERIC causal text path
//    (no bidirectional overlay — the overlay only makes the IMAGE span
//    bidirectional during prefill, a no-op over text queries in both the warm
//    and cold paths; control-flow verified), so it is numerically faithful.
//
//    Strict warm==cold byte parity is NOT asserted here (it does not hold). The
//    only warm-vs-cold difference is a deterministic ~1-ULP BF16
//    cache-hit-kernel reduction-order drift (the documented
//    `paged_decode_long_context_1ulp` class): the warm decode reads the live
//    image-span KV through the paged cache-hit kernel while the cold decode runs
//    a flat full-prompt prefill, and the tiny reduction-order delta flips one
//    early near-tie argmax that cascades into a different-but-coherent tail. On
//    this prompt the warm/cold final answers share the prefix "...the main
//    subject of the image" then diverge on a single token (" provided" vs "."),
//    both coherently analyzing the same image ("Trunch Parish Council"). So the
//    unified block asserts the FAITHFUL-warm contract (cachedTokens>0 + the same
//    coherence checks the e2b warm block uses) rather than byte parity. Exact
//    final-answer equality is asserted only for audio's cold-vs-cold replay. --
describe.skipIf(!unifiedModelPath || !imageExists)('Gemma 4 — WARM unified image->text continuation', () => {
  let model: SessionCapableModel;

  beforeAll(async () => {
    if (!unifiedModelPath) return;
    model = (await loadModel(unifiedModelPath)) as unknown as SessionCapableModel;
  }, 300_000);

  it('warm text delta after a unified image turn continues on live KV and answers coherently', async () => {
    const images = [readBytes(imagePath)];
    const prompt1 = 'Describe this image.';
    const prompt2 = 'What is the main subject?';
    const maxNew = 40;

    const session = new ChatSession(model, { system: SYSTEM });
    const turn1 = await streamTurn(session, prompt1, { images }, 48);
    expect(turn1.rawText.length).toBeGreaterThan(0);
    const warm = await streamTurn(session, prompt2, {}, maxNew);
    await session.reset();

    // eslint-disable-next-line no-console
    console.log('[gemma4-cont-unified] warm:', JSON.stringify(warm.parsedText));
    // The warm delta continued on the live unified-image KV (did NOT cold-restart).
    expect(warm.cachedTokens ?? 0).toBeGreaterThan(0);
    // Coherence (byte parity is ill-posed under the ~1-ULP cache-hit-kernel drift
    // above): a coherent, non-degenerate answer with a clean finish.
    expect(warm.finishReason === 'stop' || warm.finishReason === 'length').toBe(true);
    expect(warm.numTokens).toBeGreaterThan(0);
    const words = warm.parsedText.trim().split(/\s+/).filter(Boolean);
    expect(words.length).toBeGreaterThan(3);
    // No degenerate single-token loop.
    const counts = new Map<string, number>();
    for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
    expect(Math.max(...counts.values())).toBeLessThan(words.length * 0.8);
  });
});
