import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';
import { beforeAll, describe, expect, it } from 'vite-plus/test';

/**
 * Media -> text continuation parity (Gemma 4).
 *
 * Every follow-up re-renders the complete structured transcript through the
 * checkpoint-provided chat template. Native code reuses the live media KV only
 * when that rendered token stream exactly extends the cached token history.
 *
 * KV sharing does not make an image turn ineligible. E2B's
 * `SharedOnSliding` alias slots intentionally own no private K/V and are skipped
 * by checkpoint readiness/storage; their physical non-shared sliding anchors
 * carry the state for both layers. Both standard SigLIP images (E2B) and unified
 * bidirectional-vision images may therefore reuse faithful live K/V when the
 * template-rendered transcript is also an exact token-prefix extension.
 *
 * Audio and mixed-media identity is not represented in the per-block keys yet.
 * Those turns deliberately use `skip_lookup` with a zero cache-hit ceiling, do
 * not publish reusable prefix checkpoints, and leave the continuation marker
 * off. Their finalizer releases every KV group while KEEPING the committed
 * token history and the media keys, so the follow-up stays inside the same
 * session and cold-prefills the complete template-rendered conversation.
 *
 * A stock Gemma template drops a prior plain assistant turn's thought when it
 * re-renders, which would leave turn 2 a non-extension of the committed KV. A
 * stateful session renders with `preserve_thinking`, so the re-render does
 * reproduce that thought and the follow-up IS an exact token-prefix extension
 * of the live media KV. One-shot renders are untouched and stay byte-identical
 * to transformers / mlx-lm / mlx-vlm / vLLM.
 *
 * ## What this file asserts
 *
 *  - NON-UNIFIED and UNIFIED image: the follow-up reuses the ENTIRE committed
 *    turn-1 history and nothing less, plus a coherent answer. E2B additionally
 *    pins byte-exact equality against a direct cold replay of the same
 *    structured history: warm media-KV reuse and a cold prefill of the same
 *    conversation must decode to the same bytes.
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
  reasoningContent?: string;
  thinkingEnabled?: boolean;
  images?: Uint8Array[];
  audio?: Uint8Array[];
}

/**
 * The assistant turn exactly as `ChatSession` commits it: parsed content, the
 * reasoning it captured, and the thinking mode the turn ran under. A cold
 * reference that drops any of the three renders a DIFFERENT conversation than
 * the live session replays, so it can no longer be a parity reference.
 */
interface CommittedAssistantTurn {
  content: string;
  reasoningContent: string | null;
  thinkingEnabled: boolean;
}

function hasWeights(dir: string): boolean {
  return existsSync(resolve(dir, 'model.safetensors')) || existsSync(resolve(dir, 'model.safetensors.index.json'));
}

function hasChatTemplate(dir: string): boolean {
  if (existsSync(resolve(dir, 'chat_template.jinja'))) return true;
  const tokenizerConfig = resolve(dir, 'tokenizer_config.json');
  if (!existsSync(tokenizerConfig)) return false;
  try {
    const parsed = JSON.parse(readFileSync(tokenizerConfig, 'utf8')) as { chat_template?: unknown };
    return typeof parsed.chat_template === 'string' && parsed.chat_template.length > 0;
  } catch {
    return false;
  }
}

function findFirst(dirs: string[]): string | null {
  for (const d of dirs) {
    const abs = resolve(process.cwd(), d);
    if (existsSync(resolve(abs, 'config.json')) && hasWeights(abs) && hasChatTemplate(abs)) return abs;
  }
  return null;
}

const SYSTEM = 'You are a helpful assistant. Be concise.';

const unifiedModelPath =
  process.env.GEMMA4_UNIFIED_MODEL_PATH &&
  hasWeights(process.env.GEMMA4_UNIFIED_MODEL_PATH) &&
  hasChatTemplate(process.env.GEMMA4_UNIFIED_MODEL_PATH)
    ? process.env.GEMMA4_UNIFIED_MODEL_PATH
    : findFirst(['.cache/models/gemma-4-12b-it']);

const imageModelPath =
  process.env.GEMMA4_NONUNIFIED_MODEL_PATH &&
  hasWeights(process.env.GEMMA4_NONUNIFIED_MODEL_PATH) &&
  hasChatTemplate(process.env.GEMMA4_NONUNIFIED_MODEL_PATH)
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

interface StreamedTurn {
  rawText: string;
  parsedText: string;
  numTokens: number;
  promptTokens: number;
  finishReason: string;
  cachedTokens?: number;
  thinking: string | null;
  thinkingEnabled: boolean;
}

/** Run one streamed turn on a ChatSession; returns the accumulated reply. */
async function streamTurn(
  session: ChatSession,
  prompt: string,
  opts: { images?: Uint8Array[]; audio?: Uint8Array[] } = {},
  maxNewTokens = 48,
): Promise<StreamedTurn> {
  let rawText = '';
  let parsedText = '';
  let numTokens = 0;
  let promptTokens = 0;
  let finishReason = 'unknown';
  let cachedTokens: number | undefined;
  let thinking: string | null = null;
  let thinkingEnabled = false;
  for await (const event of session.sendStream(prompt, {
    ...(opts.images && { images: opts.images }),
    ...(opts.audio && { audio: opts.audio }),
    config: { maxNewTokens, temperature: 0, reportPerformance: false },
  })) {
    if (event.done) {
      finishReason = event.finishReason;
      numTokens = event.numTokens;
      promptTokens = event.promptTokens;
      rawText = event.rawText;
      cachedTokens = event.cachedTokens;
      thinking = event.thinking;
      thinkingEnabled = event.thinkingEnabled;
    } else if (event.isReasoning !== true) {
      parsedText += event.text;
    }
  }
  return { rawText, parsedText, numTokens, promptTokens, finishReason, cachedTokens, thinking, thinkingEnabled };
}

/** The assistant message `ChatSession` commits for a finished streamed turn. */
function committedAssistantTurn(turn: StreamedTurn): CommittedAssistantTurn {
  return {
    content: turn.parsedText,
    reasoningContent: turn.thinking,
    thinkingEnabled: turn.thinkingEnabled,
  };
}

/**
 * Length of the token history a media turn commits.
 *
 * `finalize_vision_turn_media_state` keeps every token on a `"length"` finish
 * and drops the terminal token otherwise, mirroring the text path's
 * `save_paged_history`. A follow-up that warm-continues this session must
 * report exactly this many reused tokens — no more (it would be reading past
 * the commit) and no less (it would be re-prefilling proven-identical
 * prefix).
 */
function committedHistoryLength(turn: StreamedTurn): number {
  return turn.promptTokens + turn.numTokens - (turn.finishReason === 'length' ? 0 : 1);
}

/**
 * COLD reference: an already-reset model + manually-built media-bearing history
 * driven through native `chatSessionStart`, cold-prefilling the WHOLE
 * conversation (system + media turn-1 + turn-1 reply + the text delta). Reusing
 * the loaded model avoids holding a second multi-gigabyte checkpoint. Returns
 * turn-2's reply.
 *
 * `assistantTurn1` must be the message `ChatSession` committed — content,
 * captured reasoning and thinking mode alike. A session re-render preserves
 * that reasoning, so a reference that passes only the content describes a
 * SHORTER conversation and is not a parity reference at all.
 */
async function coldReplayTurn2(
  model: SessionCapableModel,
  media: { images?: Uint8Array[]; audio?: Uint8Array[] },
  prompt1: string,
  assistantTurn1: CommittedAssistantTurn,
  prompt2: string,
  maxNewTokens: number,
): Promise<{ rawText: string; text: string; numTokens: number; promptTokens: number; cachedTokens: number }> {
  const userTurn1: UserMessage = { role: 'user', content: prompt1 };
  if (media.images) userTurn1.images = media.images;
  if (media.audio) userTurn1.audio = media.audio;
  const assistant: UserMessage = {
    role: 'assistant',
    content: assistantTurn1.content,
    thinkingEnabled: assistantTurn1.thinkingEnabled,
  };
  if (assistantTurn1.reasoningContent != null) {
    assistant.reasoningContent = assistantTurn1.reasoningContent;
  }
  const history: UserMessage[] = [
    { role: 'system', content: SYSTEM },
    userTurn1,
    assistant,
    { role: 'user', content: prompt2 },
  ];
  const r = await model.chatSessionStart(history, {
    maxNewTokens,
    temperature: 0,
    reportPerformance: false,
  });
  return {
    rawText: r.rawText,
    text: r.text,
    numTokens: r.numTokens,
    promptTokens: r.promptTokens,
    cachedTokens: r.cachedTokens,
  };
}

// -- NON-UNIFIED image continuation (e2b, KV-shared). A checkpoint without a
//    model-provided template is intentionally excluded: production fails closed
//    for that checkpoint instead of manufacturing a Gemma prompt in Rust. --
describe.skipIf(!imageModelPath || !imageExists)(
  'Gemma 4 — template-driven non-unified image->text continuation parity (KV-shared)',
  () => {
    let model: SessionCapableModel;

    beforeAll(async () => {
      if (!imageModelPath) return;
      model = (await loadModel(imageModelPath)) as unknown as SessionCapableModel;
    }, 300_000);

    it('re-rendered text follow-up matches the cold final answer', async () => {
      const images = [readBytes(imagePath)];
      const prompt1 = 'Describe this image.';
      const prompt2 = 'What is the main color?';
      const maxNew = 40;

      const session = new ChatSession(model, { system: SYSTEM });
      const turn1 = await streamTurn(session, prompt1, { images }, 48);
      expect(turn1.rawText.length).toBeGreaterThan(0);

      const replayed = await streamTurn(session, prompt2, {}, maxNew);
      await session.reset();

      const cold = await coldReplayTurn2(model, { images }, prompt1, committedAssistantTurn(turn1), prompt2, maxNew);

      // eslint-disable-next-line no-console
      console.log(
        '[gemma4-cont-image] replayed:',
        JSON.stringify(replayed.parsedText),
        'cold:',
        JSON.stringify(cold.text),
      );
      // The re-render is an exact token-prefix extension of the live media KV,
      // so the follow-up reuses the whole committed turn-1 history — not a
      // block-aligned fraction of it.
      expect(replayed.cachedTokens).toBe(committedHistoryLength(turn1));
      // Both paths rendered the same conversation; the cold reference prefills
      // every token the warm turn reused.
      expect(cold.promptTokens).toBe(replayed.promptTokens);
      expect(cold.cachedTokens).toBe(0);
      expect(replayed.parsedText.trim()).toBe(cold.text.trim());
      expect(replayed.rawText).toBe(cold.rawText);
      expect(replayed.numTokens).toBe(cold.numTokens);
      expect(replayed.finishReason === 'stop' || replayed.finishReason === 'length').toBe(true);
      expect(replayed.numTokens).toBeGreaterThan(0);
      const words = replayed.parsedText.trim().split(/\s+/).filter(Boolean);
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

    // Audio/mixed preparation deliberately disables cache lookup, so the
    // complete template-rendered follow-up must report zero cached tokens —
    // and must still be SERVED: the turn-1 finalizer dropped the KV but kept
    // the session, so this is a cold replay inside the same session, never an
    // "initialized cache owner" rejection.
    expect(replayed.cachedTokens ?? 0).toBe(0);
    expect(replayed.finishReason === 'stop' || replayed.finishReason === 'length').toBe(true);

    const cold = await coldReplayTurn2(model, { audio }, prompt1, committedAssistantTurn(turn1), prompt2, maxNew);

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

// -- UNIFIED image continuation. The live image checkpoint remains available,
//    and a template-rendered turn may use it only after an exact token-prefix
//    match. A session render preserves turn-1 reasoning, so the follow-up is
//    that exact extension and continues the live bidirectional-vision KV. --
describe.skipIf(!unifiedModelPath || !imageExists)('Gemma 4 — template-driven unified image->text continuation', () => {
  let model: SessionCapableModel;

  beforeAll(async () => {
    if (!unifiedModelPath) return;
    model = (await loadModel(unifiedModelPath)) as unknown as SessionCapableModel;
  }, 300_000);

  it('continues the live media KV when the template-rendered history extends it exactly', async () => {
    const images = [readBytes(imagePath)];
    const prompt1 = 'Describe this image.';
    const prompt2 = 'What is the main subject?';
    const maxNew = 40;

    const session = new ChatSession(model, { system: SYSTEM });
    const turn1 = await streamTurn(session, prompt1, { images }, 48);
    expect(turn1.rawText.length).toBeGreaterThan(0);
    const replayed = await streamTurn(session, prompt2, {}, maxNew);
    await session.reset();

    // eslint-disable-next-line no-console
    console.log('[gemma4-cont-unified] replayed:', JSON.stringify(replayed.parsedText));
    expect(replayed.cachedTokens).toBe(committedHistoryLength(turn1));
    expect(replayed.finishReason === 'stop' || replayed.finishReason === 'length').toBe(true);
    expect(replayed.numTokens).toBeGreaterThan(0);
    const words = replayed.parsedText.trim().split(/\s+/).filter(Boolean);
    expect(words.length).toBeGreaterThan(3);
    // No degenerate single-token loop.
    const counts = new Map<string, number>();
    for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
    expect(Math.max(...counts.values())).toBeLessThan(words.length * 0.8);
  });
});
