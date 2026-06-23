import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';
import { beforeAll, describe, expect, it } from 'vite-plus/test';

/**
 * Phase-2 WARM media -> text continuation golden parity (Gemma 4).
 *
 * A text follow-up after a pure-causal media turn (AUDIO, or a NON-UNIFIED
 * image) warm-continues on the live media KV: the vision-core finalize keeps
 * the global paged KV registered for reuse and arms `media_session_continuable`,
 * so the native `chatSessionContinue` succeeds instead of throwing the
 * media-held restart prefix. The next delta restores the prefix via REPLAY
 * (re-walking the matched prefix over the live content-addressed global KV) —
 * exactly the mechanism a TEXT warm-continue uses on the same checkpoint. (On
 * KV-shared checkpoints like e2b, the sliding-history fast-path checkpoint is a
 * structural no-op for both text and vision; continuation rides the replay
 * path, which is byte-exact, just not free.)
 *
 * ## What is byte-exact vs not, and why
 *
 *  - NON-UNIFIED image (e2b): turn-1 emits NO `<|channel>thought…<channel|>`
 *    block, so re-rendering turn-1 through the chat template is lossless. The
 *    WARM 2-turn run is therefore TOKEN-EXACT (identical `rawText` +
 *    `numTokens`) to a COLD full-prompt replay. This is the load-bearing R1
 *    proof: warm == cold byte-for-byte.
 *  - AUDIO (unified 12B): turn-1 emits a `<|channel>thought…<channel|>` block.
 *    The WARM KV holds that raw block; a COLD replay through `chatSessionStart`
 *    re-renders turn-1 via jinja, whose `strip_thinking` macro DROPS prior-turn
 *    reasoning by design. So the cold turn-2 sees a shorter prefix and re-derives
 *    a longer reasoning trace — a TEMPLATE-RENDERING artifact, not a warm-path
 *    bug (both reach the same FINAL answer). There is no raw-token-prefill API
 *    and no strip-disable, so a token-exact warm==cold golden is ill-posed for a
 *    thinking checkpoint. The audio case therefore asserts (a) the warm path
 *    actually continued (cachedTokens > 0) and (b) the warm FINAL answer equals
 *    the cold FINAL answer. The shared warm-path code is the SAME as the image
 *    case (gated only on has_audio vs has_image), so the image byte-exactness
 *    transitively covers the audio numerics.
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

// -- NON-UNIFIED image continuation: BYTE-EXACT golden (no thinking -> faithful) --
describe.skipIf(!imageModelPath || !imageExists)(
  'Gemma 4 — WARM non-unified image->text continuation parity (byte-exact)',
  () => {
    let model: SessionCapableModel;

    beforeAll(async () => {
      if (!imageModelPath) return;
      model = (await loadModel(imageModelPath)) as unknown as SessionCapableModel;
    }, 300_000);

    it('warm text delta after a non-unified image turn == cold full-prompt replay (token-exact)', async () => {
      const images = [readBytes(imagePath)];
      const prompt1 = 'Describe this image.';
      const prompt2 = 'What is the main color?';
      const maxNew = 40;

      const session = new ChatSession(model, { system: SYSTEM });
      const turn1 = await streamTurn(session, prompt1, { images }, 48);
      expect(turn1.rawText.length).toBeGreaterThan(0);
      // Faithfulness precondition: a thinking block would make the cold replay
      // (which strips prior reasoning) diverge. e2b image turns are clean.
      expect(turn1.rawText.includes('<|channel>')).toBe(false);

      const warm = await streamTurn(session, prompt2, {}, maxNew);
      await session.reset();

      // The warm delta must have actually continued (reused the media KV).
      expect(warm.cachedTokens ?? 0).toBeGreaterThan(0);

      const cold = await coldReplayTurn2(imageModelPath!, { images }, prompt1, turn1.rawText, prompt2, maxNew);

      // eslint-disable-next-line no-console
      console.log('[gemma4-cont-image] warm:', JSON.stringify(warm.rawText), 'cold:', JSON.stringify(cold.rawText));
      // Byte-exact: identical rawText + numTokens == identical generated ids.
      expect(warm.rawText).toBe(cold.rawText);
      expect(warm.numTokens).toBe(cold.numTokens);
    });
  },
);

// -- AUDIO continuation: warm continues + final-answer parity (thinking -> cold
//    strips it, so a byte-exact golden is ill-posed; see header). --
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

// -- UNIFIED image stays single-shot in Phase 2: the warm continue is NOT armed;
//    the TS send() transparently falls back to cold replay (Phase 3 enables warm). --
describe.skipIf(!audioModelPath || !imageExists)('Gemma 4 — UNIFIED image continue stays single-shot (Phase 2)', () => {
  let model: SessionCapableModel;

  beforeAll(async () => {
    if (!audioModelPath) return;
    model = (await loadModel(audioModelPath)) as unknown as SessionCapableModel;
  }, 300_000);

  it('unified image -> text continue stays single-shot and still answers coherently', async () => {
    const images = [readBytes(imagePath)];
    const prompt1 = 'Describe this image.';
    const prompt2 = 'What is the main subject?';
    const maxNew = 40;

    const session = new ChatSession(model, { system: SYSTEM });
    const turn1 = await streamTurn(session, prompt1, { images }, 48);
    expect(turn1.rawText.length).toBeGreaterThan(0);

    // The unified path keeps the media turn single-shot in Phase 2: the marker
    // stays false, the native continue throws the IMAGE restart prefix, and the
    // TS send() absorbs it into a cold replay. The observable contract is that
    // the follow-up still produces a coherent, non-degenerate answer (the
    // single-shot fallback works) — NOT that it warm-continued. A byte-exact
    // comparison against a hand-built cold reference is ill-posed here for the
    // same template prior-CoT strip reason as the audio case (unified 12B turn-1
    // emits a <|channel>thought…<channel|> block); Phase 3 will add the warm
    // path under its own byte-exact golden.
    const observed = await streamTurn(session, prompt2, {}, maxNew);
    await session.reset();

    // eslint-disable-next-line no-console
    console.log('[gemma4-cont-unified] observed:', JSON.stringify(observed.parsedText));
    expect(observed.finishReason === 'stop' || observed.finishReason === 'length').toBe(true);
    expect(observed.numTokens).toBeGreaterThan(0);
    const words = observed.parsedText.trim().split(/\s+/).filter(Boolean);
    expect(words.length).toBeGreaterThan(3);
    // No degenerate single-token loop.
    const counts = new Map<string, number>();
    for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
    expect(Math.max(...counts.values())).toBeLessThan(words.length * 0.8);
  });
});
