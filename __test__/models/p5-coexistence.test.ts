/**
 * P5 two-model coexistence (G3) — end-to-end with REAL models.
 *
 * P5 replaced a single process-global active-model-id gate over the
 * compiled-decode weight store with a per-model-id slot map. Before P5,
 * loading a 2nd compiled model DEMOTED the 1st to the eager Rust path
 * because the compiled `g_weights` store only held one model's weights at
 * a time. This test proves the headline feature: two compiled models
 * resident in one process, both run correctly, and the first is NOT
 * demoted when the second loads.
 *
 * THE OBSERVABLE — the compiled decode branch sets a `DecodeProfiler`
 * label "chat_compiled" (crates/mlx-core/src/models/qwen3_5/model.rs:2218);
 * the eager Rust fallback sets "chat_rust" (:2609). The per-model-id gate
 * that selects between them is
 * `use_compiled = mlx_qwen35_model_has_weights(model_id) != 0`
 * (model.rs:1509 — the non-streaming chatSessionContinue path). The label
 * is exposed to TS as `getProfilingData().generations[].label`
 * (packages/core/index.d.cts:3105 / :3172) and is ONLY recorded while
 * `setProfilingEnabled(true)`. With the OLD pre-P5 code, model A's 2nd
 * turn (after B loads) read "chat_rust" (demoted); post-P5 it must read
 * "chat_compiled". THAT is the spine of this test.
 *
 * This is a real-model gated test: it self-skips when the checkpoints are
 * absent (e.g. CI without the model volume) so it runs locally and no-ops
 * elsewhere.
 */

import { existsSync } from 'node:fs';

import {
  getProfilingData,
  resetProfilingData,
  setProfilingEnabled,
  type GenerationProfile,
} from '@mlx-node/core';
import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

// Compiled-decode profiler label for the NON-STREAMING send() path
// (crates/mlx-core/src/models/qwen3_5/model.rs:2218). The eager fallback
// would be "chat_rust" (:2609) — pre-P5 that is what A's 2nd turn read.
const COMPILED_LABEL = 'chat_compiled';

// Checkpoints (all dense qwen3_5 bf16 + one lfm2 bf16). Dense A has a
// repo-local fallback under .cache/models.
const DENSE_A_CANDIDATES = [
  '/Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16',
  `${process.cwd()}/.cache/models/qwen3.5-0.8b-mlx-bf16`,
];
const DENSE_B_PATH = '/Volumes/P4510/models/Qwen3.5-4B-mlx';
const LFM2_PATH = '/Volumes/P4510/models/lfm2.5-1.2b-thinking-mlx';

function firstExisting(paths: string[]): string | undefined {
  return paths.find((p) => existsSync(p));
}

const denseAPath = firstExisting(DENSE_A_CANDIDATES);
const denseBPresent = existsSync(DENSE_B_PATH);
const lfm2Present = existsSync(LFM2_PATH);

// Deterministic greedy turn config. temperature:0 => pure argmax (no
// seed); the two repetition early-stops are disabled
// (maxConsecutiveTokens:0 / maxNgramRepeats:0 — engine/params.rs:257-258)
// so token count + finishReason stay stable across the A-before-B and
// A-after-B turns; reasoningEffort:'none' + enableMtp:false keep the
// path on the plain single-token compiled decode loop.
const TURN_CONFIG = {
  maxNewTokens: 24,
  temperature: 0,
  reasoningEffort: 'none',
  enableMtp: false,
  maxConsecutiveTokens: 0,
  maxNgramRepeats: 0,
  reuseCache: false,
} as const;

// lfm2's template ignores enable_thinking, so additionally pin the
// thinking budget to 0.
const LFM2_TURN_CONFIG = { ...TURN_CONFIG, thinkingTokenBudget: 0 } as const;

const PROMPT = 'List three colors. Reply with just the names.';

interface TurnObservation {
  text: string;
  rawText: string;
  finishReason: string;
  label: string;
  modelType: string;
}

/**
 * Read the newly-appended generation. `getProfilingData().generations`
 * is a cumulative global store (crates/mlx-core/src/profiling.rs:287 —
 * push_generation appends), so the just-finished turn is the LAST entry.
 * We assert the array actually grew to be sure we're reading this turn's
 * record and not a stale one.
 */
function lastGeneration(beforeLen: number): GenerationProfile {
  const gens = getProfilingData().generations;
  expect(gens.length).toBe(beforeLen + 1);
  return gens[gens.length - 1]!;
}

/**
 * Run ONE measured turn as a fresh turn-0 send.
 *
 * Each measured turn calls `session.reset()` first (chat-session.ts:831 —
 * `model.resetCaches()` + `turnCount = 0`) so the prompt is always the
 * single-turn PROMPT, never a multi-turn continuation. Without this, a
 * second `send()` on the same session takes the delta `chatSessionContinue`
 * path conditioned on turn-1 history, producing a DIFFERENT (longer)
 * prompt — which makes the turn-to-turn output legitimately diverge for
 * reasons unrelated to P5. Resetting keeps every turn an apples-to-apples
 * turn-0 run of the identical prompt, which is what "run A again" must mean
 * for the no-demotion comparison. Crucially `resetCaches()` only clears the
 * KV cache; it does NOT touch the per-model-id compiled weight slot, so the
 * compiled-path gate (`mlx_qwen35_model_has_weights(model_id)`) is
 * unaffected — exactly the residency this test is probing.
 */
async function runTurn(
  session: ChatSession,
  config: Record<string, unknown>,
): Promise<TurnObservation> {
  await session.reset();
  const beforeLen = getProfilingData().generations.length;
  const result = await session.send(PROMPT, { config });
  const gen = lastGeneration(beforeLen);
  return {
    text: result.text,
    rawText: result.rawText,
    finishReason: result.finishReason,
    label: gen.label,
    modelType: gen.modelType,
  };
}

const denseGateReady = Boolean(denseAPath) && denseBPresent;

describe.sequential('P5 two-model coexistence (G3)', () => {
  beforeAll(() => {
    resetProfilingData();
    setProfilingEnabled(true);
  });

  afterAll(() => {
    setProfilingEnabled(false);
    resetProfilingData();
  });

  // ── G3a: NO-DEMOTION (the headline), two compiled DENSE models ──────
  it.skipIf(!denseGateReady)(
    'keeps model A on the compiled path after model B loads (no demotion)',
    async () => {
      const sessionA = new ChatSession(
        (await loadModel(denseAPath!)) as unknown as SessionCapableModel,
      );

      // 1. A's first turn — must be compiled.
      const a1 = await runTurn(sessionA, { ...TURN_CONFIG });

      // 2. Load B (now two compiled dense models resident), run on B.
      const sessionB = new ChatSession(
        (await loadModel(DENSE_B_PATH)) as unknown as SessionCapableModel,
      );
      const b1 = await runTurn(sessionB, { ...TURN_CONFIG });

      // 3. THE PROOF: run A AGAIN while B is resident.
      const a2 = await runTurn(sessionA, { ...TURN_CONFIG });

      // 4. Run B again for symmetry.
      const b2 = await runTurn(sessionB, { ...TURN_CONFIG });

      // Diagnostics in case an assertion fails — print verbatim labels.
      const observed = { a1, b1, a2, b2 };
      // eslint-disable-next-line no-console
      console.log('[G3a] observed labels:', JSON.stringify(observed, null, 2));

      // Every turn took the compiled path.
      expect(a1.label).toBe(COMPILED_LABEL);
      expect(b1.label).toBe(COMPILED_LABEL);
      // THE CRITICAL ASSERTION: pre-P5 this was the eager label.
      expect(a2.label).toBe(COMPILED_LABEL);
      expect(b2.label).toBe(COMPILED_LABEL);

      // A is unperturbed by B's load (output stable turn-to-turn).
      expect(a2.text).toBe(a1.text);
      expect(a2.rawText).toBe(a1.rawText);
      expect(b2.text).toBe(b1.text);
      expect(b2.rawText).toBe(b1.rawText);

      // Sanity guard: the two models are genuinely different, so the
      // equality checks above are not trivially comparing identical
      // strings.
      expect(a1.text).not.toBe(b1.text);

      // Greedy + early-stops-off => length-bounded completions.
      expect(a1.finishReason).toBe('length');
      expect(b1.finishReason).toBe('length');
      expect(a2.finishReason).toBe('length');
      expect(b2.finishReason).toBe('length');
    },
  );

  // ── G3b: FAMILY-MAP INDEPENDENCE (dense + lfm2 coexist) ─────────────
  const familyGateReady = Boolean(denseAPath) && lfm2Present;
  it.skipIf(!familyGateReady)(
    'keeps dense on the compiled path after a different family (lfm2) loads',
    async () => {
      const denseSession = new ChatSession(
        (await loadModel(denseAPath!)) as unknown as SessionCapableModel,
      );

      // 1. Dense first turn — must be compiled.
      const d1 = await runTurn(denseSession, { ...TURN_CONFIG });

      // 2. Load lfm2 (a different family slot map) and run it — just
      //    confirm it produces non-empty output without error. We do NOT
      //    assert lfm2's label string.
      const lfm2Session = new ChatSession(
        (await loadModel(LFM2_PATH)) as unknown as SessionCapableModel,
      );
      const lfmBeforeLen = getProfilingData().generations.length;
      const lfmResult = await lfm2Session.send(PROMPT, { config: { ...LFM2_TURN_CONFIG } });
      // The generation store still grew (lfm2 recorded a profile of its
      // own), confirming the lfm2 turn ran end-to-end.
      expect(getProfilingData().generations.length).toBe(lfmBeforeLen + 1);
      expect(typeof lfmResult.text).toBe('string');
      expect(lfmResult.text.length + lfmResult.rawText.length).toBeGreaterThan(0);

      // 3. Run dense AGAIN — must STILL be compiled + stable output.
      const d2 = await runTurn(denseSession, { ...TURN_CONFIG });

      // eslint-disable-next-line no-console
      console.log(
        '[G3b] dense labels:',
        JSON.stringify({ d1: d1.label, d2: d2.label, lfm2ModelType: undefined }, null, 2),
      );

      expect(d1.label).toBe(COMPILED_LABEL);
      // The proof for G3b: a different family loading did not demote dense.
      expect(d2.label).toBe(COMPILED_LABEL);
      expect(d2.text).toBe(d1.text);
      expect(d2.rawText).toBe(d1.rawText);
      expect(d1.finishReason).toBe('length');
      expect(d2.finishReason).toBe('length');
    },
  );
});
