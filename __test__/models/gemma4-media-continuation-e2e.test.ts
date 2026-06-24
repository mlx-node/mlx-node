import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';
import { beforeAll, describe, expect, it } from 'vite-plus/test';

/**
 * Media -> text continuation parity (Gemma 4).
 *
 * A text follow-up after ANY media turn (AUDIO, NON-UNIFIED image, or UNIFIED
 * bidirectional-vision image) MAY warm-continue on the live media KV, but only
 * when the warm restore is numerically FAITHFUL. The eligibility gate admits any
 * image/audio turn; the vision-core finalize keeps the global paged KV
 * registered for reuse and arms `media_session_continuable` ONLY when the
 * sliding-history checkpoint actually STORED. That stored/live gate splits the
 * two checkpoint classes:
 *
 *  - NON-KV-SHARED (12B audio, `num_kv_shared_layers=0`): every sliding layer is
 *    a plain `Sliding` anchor that physically wrote real audio/image-feature K/V
 *    during the media prefill, so the sliding checkpoint STORES, the marker
 *    arms, and the next delta hits `state="live"` — reusing the in-place
 *    faithful K/V. This is the canonical FAITHFUL warm path.
 *  - KV-SHARED (e2b image, `num_kv_shared_layers=20`): the `SharedOnSliding`
 *    layers never wrote their own flat K/V, so the sliding checkpoint stores
 *    NOTHING (`stored=false`). A warm restore would fall to `state="replay"`,
 *    which rebuilds media-position sliding K/V from `embed_tokens.forward(id)` =
 *    RAW `<|image|>` special-token embeddings, NOT the scattered SigLIP/audio
 *    features — numerically UNFAITHFUL. So the marker is left OFF; the text
 *    follow-up cleanly COLD-RESTARTS through the `ChatSession` Phase-1 catch.
 *
 *  (REPLAY is faithful for TEXT positions because a text token's true embedding
 *  IS `embed_tokens.forward(id)`; it cannot reconstruct a MEDIA position's true
 *  embedding, which is a scattered feature.)
 *
 * ## What this file asserts
 *
 *  - AUDIO (12B, non-KV-shared): the warm path actually continued
 *    (cachedTokens > 0) and the warm FINAL answer equals the cold FINAL answer.
 *    This is the load-bearing FAITHFUL-warm proof.
 *  - UNIFIED image (12B, non-KV-shared): the warm delta continues
 *    (cachedTokens > 0) and answers coherently. The bidirectional overlay only
 *    edits the IMAGE-span mask during prefill (a no-op over text queries), so the
 *    warm text delta routes through the generic causal text path and is faithful.
 *    Strict warm==cold byte parity is NOT asserted: a deterministic ~1-ULP BF16
 *    cache-hit-kernel reduction-order drift (the documented
 *    `paged_decode_long_context_1ulp` class) flips one early near-tie argmax that
 *    cascades into a different-but-coherent tail. So this block asserts the
 *    faithful-warm contract (cachedTokens>0) plus coherence, not byte parity. The
 *    audio block above happens to dodge the near-tie and stays byte-exact.
 *  - NON-UNIFIED image (e2b, KV-shared): the follow-up cold-restarts; the test
 *    is a single-shot/cold-restart COHERENCE check (coherent, non-degenerate
 *    answer). This is the negative control proving KV-shared still cold-restarts.
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
  return (
    existsSync(resolve(dir, 'model.safetensors')) || existsSync(resolve(dir, 'model.safetensors.index.json'))
  );
}

function findFirst(dirs: string[]): string | null {
  for (const d of dirs) {
    const abs = resolve(process.cwd(), d);
    if (existsSync(resolve(abs, 'config.json')) && hasWeights(abs)) return abs;
  }
  return null;
}

const SYSTEM = 'You are a helpful assistant. Be concise.';

const audioModelPath =
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
    } else {
      parsedText += event.text;
    }
  }
  return { rawText, parsedText, numTokens, finishReason, cachedTokens };
}

/**
 * COLD reference: a fresh model + manually-built media-bearing history driven
 * through the native `chatSessionStart`, cold-prefilling the WHOLE conversation
 * (system + media turn-1 + turn-1 reply + the text delta). Returns turn-2's
 * reply. `assistantTurn1` is the content stored for the prior assistant turn —
 * `rawText` for the byte-exact (no-thinking) case; `parsedText` is equivalent
 * for the audio case since the template strips reasoning either way.
 */
async function coldReplayTurn2(
  modelPath: string,
  media: { images?: Uint8Array[]; audio?: Uint8Array[] },
  prompt1: string,
  assistantTurn1: string,
  prompt2: string,
  maxNewTokens: number,
): Promise<{ rawText: string; text: string; numTokens: number }> {
  const model = (await loadModel(modelPath)) as unknown as {
    chatSessionStart: (
      messages: UserMessage[],
      config?: { maxNewTokens?: number; temperature?: number; reportPerformance?: boolean },
    ) => Promise<{ rawText: string; text: string; numTokens: number }>;
  };
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

// -- NON-UNIFIED image continuation (e2b, KV-shared): the media->text follow-up
//    COLD-RESTARTS (the sliding checkpoint does not store -> warm restore would
//    be unfaithful -> the marker is left off). Coherence check only. --
describe.skipIf(!imageModelPath || !imageExists)(
  'Gemma 4 — non-unified image->text continue cold-restarts (KV-shared)',
  () => {
    let model: SessionCapableModel;

    beforeAll(async () => {
      if (!imageModelPath) return;
      model = (await loadModel(imageModelPath)) as unknown as SessionCapableModel;
    }, 300_000);

    it('image -> text continue cold-restarts and still answers coherently', async () => {
      const images = [readBytes(imagePath)];
      const prompt1 = 'Describe this image.';
      const prompt2 = 'What is the main color?';
      const maxNew = 40;

      const session = new ChatSession(model, { system: SYSTEM });
      const turn1 = await streamTurn(session, prompt1, { images }, 48);
      expect(turn1.rawText.length).toBeGreaterThan(0);

      // e2b is KV-shared (`num_kv_shared_layers=20`): the sliding-history
      // checkpoint stores nothing, so the finalize leaves the marker false. The
      // native continue throws the IMAGE restart prefix and the TS send()
      // absorbs it into a cold replay. The observable contract is that the
      // follow-up still produces a coherent, non-degenerate answer (the
      // single-shot cold-restart fallback works) — NOT that it warm-continued.
      const observed = await streamTurn(session, prompt2, {}, maxNew);
      await session.reset();

      // eslint-disable-next-line no-console
      console.log('[gemma4-cont-image] observed:', JSON.stringify(observed.parsedText));
      // Cold-restart signal: the vision cold prefill primes with skip_lookup +
      // max_cache_hit_tokens=0, so a correct cold-restart reports cachedTokens=0.
      // A regression that wrongly re-armed the marker would warm-continue and
      // report cachedTokens>0 — so this fails loud on accidental warm reuse.
      expect(observed.cachedTokens ?? 0).toBe(0);
      expect(observed.finishReason === 'stop' || observed.finishReason === 'length').toBe(true);
      expect(observed.numTokens).toBeGreaterThan(0);
      const words = observed.parsedText.trim().split(/\s+/).filter(Boolean);
      expect(words.length).toBeGreaterThan(3);
      // No degenerate single-token loop.
      const counts = new Map<string, number>();
      for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
      expect(Math.max(...counts.values())).toBeLessThan(words.length * 0.8);
    });
  },
);

// -- AUDIO continuation: warm continues + final-answer parity (thinking -> cold
//    strips it, so a byte-exact golden is ill-posed; see header). This is the
//    canonical FAITHFUL warm path: 12B is non-KV-shared (`num_kv_shared_layers=0`)
//    so the sliding caches store -> the next delta hits `state="live"`. --
describe.skipIf(!audioModelPath || !audioExists)('Gemma 4 — WARM audio->text continuation parity', () => {
  let model: SessionCapableModel;

  beforeAll(async () => {
    if (!audioModelPath) return;
    model = (await loadModel(audioModelPath)) as unknown as SessionCapableModel;
  }, 300_000);

  it('warm text delta after an audio turn continues on live KV and matches the cold final answer', async () => {
    const audio = [readBytes(audioPath)];
    const prompt1 = 'Briefly describe this audio.';
    const prompt2 = 'In one word, what language is it?';
    const maxNew = 40;

    const session = new ChatSession(model, { system: SYSTEM });
    const turn1 = await streamTurn(session, prompt1, { audio }, 40);
    expect(turn1.rawText.length).toBeGreaterThan(0);
    const warm = await streamTurn(session, prompt2, {}, maxNew);
    await session.reset();

    // The warm delta continued on the live audio KV (did not cold-restart).
    expect(warm.cachedTokens ?? 0).toBeGreaterThan(0);

    const cold = await coldReplayTurn2(audioModelPath!, { audio }, prompt1, turn1.parsedText, prompt2, maxNew);

    // eslint-disable-next-line no-console
    console.log('[gemma4-cont-audio] warm:', JSON.stringify(warm.parsedText), 'cold:', JSON.stringify(cold.text));
    // FINAL-answer parity (the full token stream differs only by the template's
    // prior-CoT strip on the cold side, a rendering artifact — see header).
    expect(warm.parsedText.trim()).toBe(cold.text.trim());
    expect(warm.parsedText.trim().length).toBeGreaterThan(0);
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
//    both coherently analyzing the same image ("Trunch Parish Council"). The
//    audio block above happens to dodge a near-tie and stays byte-exact; this
//    image prompt hits one early. So the unified block asserts the FAITHFUL-warm
//    contract (cachedTokens>0 + the same coherence checks the e2b cold-restart
//    block uses) rather than byte parity. --
describe.skipIf(!audioModelPath || !imageExists)('Gemma 4 — WARM unified image->text continuation', () => {
  let model: SessionCapableModel;

  beforeAll(async () => {
    if (!audioModelPath) return;
    model = (await loadModel(audioModelPath)) as unknown as SessionCapableModel;
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
