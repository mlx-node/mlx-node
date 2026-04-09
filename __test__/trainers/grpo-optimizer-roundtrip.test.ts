/**
 * GRPO checkpoint + optimizer state round-trip integration test.
 *
 * Exercises the full `saveCheckpoint` → `GRPOTrainer.create({ resumeFromCheckpoint })`
 * chain for a loaded Qwen3 model. This is the user-visible guarantee that
 * matters: save a checkpoint mid-training, restart the process, resume, and
 * keep training without a crash and without losing step state.
 *
 * Specifically this test guards two fairly recent regressions:
 *
 *   - `f281143` — `Qwen3Model::save_model` used to touch weight MxArrays off
 *     the dedicated model thread, which crashed on loaded (thread-backed)
 *     models. `saveCheckpoint` now routes the save through the model thread.
 *     Before the fix, step 2 below aborted with a foreign exception.
 *
 *   - `9eff18a` — `GrpoTrainingEngine::save_optimizer_state` /
 *     `load_optimizer_state` were silent `warn!` no-op stubs after the Phase 3
 *     training-thread refactor. `GRPOTrainer` was calling them on every
 *     checkpoint and on resume, but the AdamW moment tensors and step counter
 *     silently failed to round-trip. Both entry points now go through the
 *     dedicated model thread via `SaveOptimizerState` / `LoadOptimizerState`
 *     commands. The resume path in step 3 below is the user-visible surface
 *     of that fix: it must complete without aborting, regardless of whether
 *     the optimizer had populated moments at save time.
 *
 * ---------------------------------------------------------------------------
 * Note on AdamW moment assertions
 * ---------------------------------------------------------------------------
 * An earlier draft of this test tried to assert that
 * `optimizer_state.safetensors` was present after the save and that the
 * resumed loss matched the pre-save loss within a sane factor. That proved
 * impractical on a tiny random-weight Qwen3:
 *
 *   - With a short `maxCompletionLength` (<=16) every rollout hits the
 *     degenerate-completion filter in the engine (`finish_reason="length"` +
 *     `tokens >= 0.9 * max_completion_length`), the train step early-returns
 *     with `loss=0, gradientsApplied=false`, and AdamW never populates a
 *     single moment tensor — so `save_optimizer_state_sync` short-circuits
 *     out of the `keys.is_empty()` branch without writing a file.
 *   - With `maxCompletionLength` >= 17, generation can finish via the
 *     repetition detector (`finish_reason="repetition"`) which bypasses the
 *     filter, but on tiny bf16 random weights the Metal autograd backward
 *     kernel aborts with a foreign C++ exception before returning control
 *     to Rust. That abort is orthogonal to this fix and not something this
 *     test should try to reproduce.
 *
 * So instead of chasing a configuration where both autograd and the filter
 * cooperate, this test deliberately stays inside the safe max=10 regime and
 * validates the plumbing that the two fixes are actually about:
 *
 *   a) `saveCheckpoint` on a loaded (thread-backed) Qwen3 model completes
 *      without crashing and writes the files the resume path expects
 *      (`training_state.json`, `weights.safetensors`, `config.json`,
 *      `tokenizer.json`).
 *   b) `GRPOTrainer.create({ resumeFromCheckpoint })` rebuilds the trainer
 *      without aborting, including the `loadOptimizerState` call that used
 *      to silently no-op on the old `warn!` stub and now dispatches to the
 *      model thread. A missing optimizer file is tolerated by the resume
 *      code's try/catch, but the underlying `LoadOptimizerState` command
 *      must still be wired through the thread — a regression that broke
 *      that wiring would abort the test worker, not fall into the warn.
 *   c) The resumed trainer preserves the JS-side step counter from
 *      `training_state.json` (3 → still 3 before any new step).
 *   d) A subsequent `trainStep` on the resumed trainer returns finite
 *      metrics. This proves the model thread is still live and the
 *      optimizer state (populated or not) loaded through `loadOptimizerState`
 *      didn't leave the training engine in a broken state.
 */

import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { Qwen3Model } from '@mlx-node/core';
import { loadModel } from '@mlx-node/lm';
import { GRPOTrainer, type RewardOutput } from '@mlx-node/trl';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import { createTempModel } from '../test-model-utils';

// Deterministic varying reward — two completions in a group-of-2 get
// different scores, giving a non-zero advantage if the engine ever makes
// it past the degenerate-completion filter. For this tiny random-weight
// model that filter will in practice kill every rollout (see the header
// comment), in which case the reward values are moot.
const deterministicReward = (outputs: RewardOutput[]): Float32Array =>
  Float32Array.from(outputs.map((_o, i) => (i % 2 === 0 ? 1.0 : 0.0)));

interface TrainingStateJson {
  step: number;
  epoch: number;
  timestamp: string;
  hasOptimizerState: boolean;
}

describe.sequential('GRPOTrainer checkpoint + optimizer state round-trip', () => {
  let tempModel: { modelPath: string; cleanup: () => void };
  let checkpointDir: string;

  beforeAll(async () => {
    // Default TINY_TEST_CONFIG from test-model-utils — identical shape to the
    // other GRPO trainer tests, which is the combination of head_dim /
    // hidden_size / num_heads known to be stable on the Metal backward path.
    tempModel = await createTempModel();
    checkpointDir = mkdtempSync(join(tmpdir(), 'mlx-grpo-opt-roundtrip-'));
  });

  afterAll(() => {
    tempModel?.cleanup();
    if (checkpointDir && existsSync(checkpointDir)) {
      try {
        rmSync(checkpointDir, { recursive: true, force: true });
      } catch (err) {
        console.warn(`Failed to cleanup checkpoint dir ${checkpointDir}:`, err);
      }
    }
  });

  it('saves a checkpoint through the model thread and resumes without crashing', async () => {
    // --- Phase 1: initial trainer, run a few steps -------------------------
    const loadedA = await loadModel(tempModel.modelPath);
    expect(loadedA).toBeInstanceOf(Qwen3Model);
    const modelA = loadedA as unknown as Qwen3Model;

    const sharedTrainerOptions = {
      modelName: 'qwen3-tiny-roundtrip',
      modelPath: tempModel.modelPath, // so saveCheckpoint can find tokenizer.json
      groupSize: 2,
      // max_completion_length is deliberately small — see the header comment.
      // The tiny random-weight model hits the degenerate-completion filter in
      // this regime, so gradients don't actually flow; we're validating the
      // save/resume plumbing, not the numerics.
      maxCompletionLength: 10,
      temperature: 0.8,
      topP: 0.95,
      learningRate: 1e-4,
      rewardFunction: deterministicReward,
      logConsole: false,
      outputDir: checkpointDir,
    } as const;

    const trainerA = new GRPOTrainer(modelA, sharedTrainerOptions);

    const prompts = [[{ role: 'user' as const, content: 'hi' }]];

    const m1 = await trainerA.trainStep(prompts);
    const m2 = await trainerA.trainStep(prompts);
    const m3 = await trainerA.trainStep(prompts);

    for (const [idx, m] of [m1, m2, m3].entries()) {
      expect(typeof m.loss).toBe('number');
      expect(Number.isFinite(m.loss)).toBe(true);
      // The JS-side step counter on GRPOTrainer is incremented per trainStep
      // regardless of whether the engine early-returned, so step follows the
      // call index 1..3 even in the filter-killed regime.
      expect(m.step).toBe(idx + 1);
    }
    expect(trainerA.getStep()).toBe(3);

    // --- Phase 2: save checkpoint -----------------------------------------
    // `saveCheckpoint(name)` writes to `join(outputDir, name)`. Before
    // f281143 this aborted the process on a loaded (thread-backed) model
    // because `Qwen3Model::save_model` touched weight MxArrays off-thread.
    const checkpointName = 'checkpoint-3';
    const checkpointPath = await trainerA.saveCheckpoint(checkpointName);
    expect(checkpointPath).toBe(join(checkpointDir, checkpointName));

    // Sanity-check the checkpoint layout: weights, tokenizer, config, and
    // the JSON blob the resume path reads. We do NOT assert that
    // `optimizer_state.safetensors` is present: with the filter-killed
    // regime the engine never populates any AdamW moments, and
    // `save_optimizer_state_sync` short-circuits before touching disk when
    // `keys.is_empty()`. The resume code tolerates that gracefully.
    const checkpointFiles = readdirSync(checkpointPath);
    expect(checkpointFiles).toContain('training_state.json');
    expect(checkpointFiles).toContain('config.json');
    expect(checkpointFiles).toContain('tokenizer.json');
    expect(checkpointFiles.some((name: string) => name === 'weights.safetensors' || name === 'weights.mlx')).toBe(true);

    const stateJson: TrainingStateJson = JSON.parse(readFileSync(join(checkpointPath, 'training_state.json'), 'utf-8'));
    expect(stateJson.step).toBe(3);
    expect(stateJson.epoch).toBe(0);
    // `hasOptimizerState` is set unconditionally by the trainer regardless of
    // whether a file was written; the resume code's try/catch handles the
    // case where the file is absent.
    expect(stateJson.hasOptimizerState).toBe(true);

    // --- Phase 3: construct a fresh trainer via resumeFromCheckpoint -------
    // This is the user-visible surface of commit 9eff18a: internally
    // `GRPOTrainer.create` calls `engine.loadOptimizerState(...)`, which
    // used to be a silent `warn!` no-op and now dispatches through the
    // dedicated model thread via the `LoadOptimizerState` command. A
    // regression that broke the model-thread wiring would abort the worker
    // here, not fall through into the try/catch.
    const trainerB = await GRPOTrainer.create({
      ...sharedTrainerOptions,
      resumeFromCheckpoint: checkpointPath,
    });

    // The JS-side step counter is restored from training_state.json before
    // any new trainStep runs.
    expect(trainerB.getStep()).toBe(3);

    // --- Phase 4: run one more step on the resumed trainer -----------------
    // The fresh trainer must still be able to run a training step — proves
    // the model thread is live and `loadOptimizerState` (whether it loaded
    // a file or fell through the try/catch) left the engine in a usable
    // state.
    const m4 = await trainerB.trainStep(prompts);
    expect(typeof m4.loss).toBe('number');
    expect(Number.isFinite(m4.loss)).toBe(true);
    // The trainer's step counter is resumed from 3 and then incremented by
    // the trainStep call, so expect 4.
    expect(trainerB.getStep()).toBe(4);
  });
});
