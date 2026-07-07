import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

import { ChatSession, loadModel, loadSession, type SessionCapableModel } from '@mlx-node/lm';
import { beforeAll, describe, expect, it } from 'vite-plus/test';

/**
 * End-to-end test for Gemma4 assistant-checkpoint speculative decoding
 * (google/gemma-4-12B-it-assistant external draft) through the public TS
 * surface: `loadSession(modelPath, { draftModelPath })`.
 *
 * PRIMARY ORACLE (mirrors `crates/mlx-core/tests/gemma4_assistant.rs`): the
 * assistant draft is LOSSLESS at T=0 — the greedy turn from the
 * draft-attached session must byte-match the greedy turn from a plain
 * no-draft session, and the draft-attached turn must actually run
 * speculative cycles (`performance.mtpCycles > 0`, not a silent AR
 * fallback). The no-draft model also pins the `ChatSession` auto-default:
 * without a draft, `hasMtpWeights()` is `false`, so no MTP config is
 * injected and the turn reports no cycles.
 *
 * Env-gated (the Rust suite's convention — both required, skipped otherwise):
 *   MLX_TEST_GEMMA4_MODEL_PATH     — bf16 Gemma-4-12B-IT checkpoint dir
 *   MLX_TEST_GEMMA4_ASSISTANT_PATH — gemma-4-12B-it-assistant draft dir
 *
 * To run locally:
 *   MLX_TEST_GEMMA4_MODEL_PATH=.cache/models/gemma-4-12b-it \
 *   MLX_TEST_GEMMA4_ASSISTANT_PATH=.cache/models/gemma-4-12B-it-assistant \
 *     vp test __test__/models/gemma4-assistant-e2e.test.ts --run
 *
 * The fixture prompt is tie-screened (see the Rust suite's module doc): a
 * constrained generation whose greedy top-2 logit gaps sit far above bf16
 * kernel noise, so the byte-equal oracle stays strict without near-tie
 * flakes.
 */

/** True for a directory that looks like a loadable checkpoint. */
function isCheckpointDir(dir: string): boolean {
  if (!existsSync(resolve(dir, 'config.json'))) return false;
  return existsSync(resolve(dir, 'model.safetensors')) || existsSync(resolve(dir, 'model.safetensors.index.json'));
}

const modelPath = process.env.MLX_TEST_GEMMA4_MODEL_PATH ?? '';
const draftPath = process.env.MLX_TEST_GEMMA4_ASSISTANT_PATH ?? '';
const gated = modelPath !== '' && draftPath !== '';

describe.skipIf(!gated)('Gemma4 assistant-draft speculative decoding — TS e2e (greedy matches no-draft)', () => {
  let arSession: ChatSession;
  let assistantSession: ChatSession;

  beforeAll(async () => {
    if (!gated) return;
    // Fail loudly on a stale env var (the Rust suite asserts the same) —
    // a skip here would silently stop covering the assistant load path.
    expect(isCheckpointDir(modelPath), `MLX_TEST_GEMMA4_MODEL_PATH is not a checkpoint dir: ${modelPath}`).toBe(true);
    expect(isCheckpointDir(draftPath), `MLX_TEST_GEMMA4_ASSISTANT_PATH is not a checkpoint dir: ${draftPath}`).toBe(
      true,
    );
    // Plain model loaded via `loadModel` so the no-draft `hasMtpWeights()`
    // contract is pinned directly on the model handle; the draft-attached
    // session goes through `loadSession` — the exact public surface the
    // zero-TS-change claim is about.
    const plainModel = await loadModel(modelPath);
    expect((plainModel as unknown as { hasMtpWeights(): boolean }).hasMtpWeights()).toBe(false);
    arSession = new ChatSession(plainModel as unknown as SessionCapableModel);
    assistantSession = await loadSession(modelPath, { draftModelPath: draftPath });
  }, 900_000);

  // Tie-screened fixture shared with the Rust primary oracle
  // (`assistant_greedy_matches_ar_multi_cycle`): a constrained numbered
  // recipe holds 200 greedy tokens without a near-tie.
  const PROMPT = 'Give a simple recipe for pancakes with numbered steps.';

  async function greedyTurn(session: ChatSession) {
    return session.send(PROMPT, {
      config: { maxNewTokens: 200, temperature: 0, reportPerformance: true },
    });
  }

  it('greedy assistant send matches the no-draft session and runs speculative cycles', async () => {
    const ar = await greedyTurn(arSession);
    const assistant = await greedyTurn(assistantSession);

    // Coherence guard: byte-equality alone would also pass if BOTH runs
    // produced identical garbage (e.g. a corrupt metallib); a greedy
    // pancake recipe always names its subject.
    expect(ar.text.toLowerCase()).toContain('pancake');
    expect(ar.text.trim().length).toBeGreaterThan(50);

    // T=0 losslessness: byte-equal output against the no-draft session.
    expect(assistant.text).toBe(ar.text);
    expect(assistant.rawText).toBe(ar.rawText);
    expect(assistant.finishReason).toBe(ar.finishReason);
    expect(assistant.numTokens).toBe(ar.numTokens);

    // The draft-attached turn really ran speculative cycles (no silent AR
    // fallback) and filled the speculative stats.
    expect(assistant.performance?.mtpCycles ?? 0).toBeGreaterThan(0);
    expect(assistant.performance?.mtpMeanAcceptedTokensTotal ?? 0).toBeGreaterThan(0);

    // The no-draft session stayed plain AR: `hasMtpWeights()` is false
    // there (asserted at load), so the ChatSession MTP auto-default never
    // kicked in.
    expect(ar.performance?.mtpCycles ?? 0).toBe(0);
  }, 600_000);
});
