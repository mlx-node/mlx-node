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
 *     commands. Step 2 below writes `optimizer_state.safetensors` through
 *     that command, and step 3 below consumes it via `LoadOptimizerState`.
 *
 * ---------------------------------------------------------------------------
 * Making the AdamW state map actually populate on a tiny random-weight Qwen3
 * ---------------------------------------------------------------------------
 *
 * A naive version of this test (short `maxCompletionLength`, default sampling,
 * constant rewards) silently degrades into a no-op: the engine's degenerate-
 * completion filter kills every rollout (`finish_reason="length"` +
 * `tokens >= 0.9 * max_completion_length`), `train_step_grpo_sync` is never
 * called, the optimizer state map stays empty, and
 * `save_optimizer_state_sync` takes the `keys.is_empty()` early return
 * without writing a file. In this scenario the optimizer-state round-trip
 * is not exercised at all.
 *
 * After task H3 (fix saveCheckpoint hasOptimizerState lie), `saveCheckpoint`
 * checks whether the safetensors file actually exists on disk after
 * `saveOptimizerState` returns and only sets `hasOptimizerState=true` when a
 * file is present. So if this test accidentally degrades into the no-op
 * path, the `expect(stateJson.hasOptimizerState).toBe(true)` assertion below
 * will fail loudly instead of vacuously passing. The resume path in
 * `GRPOTrainer.create` also no longer swallows a missing optimizer file —
 * if `hasOptimizerState` is true and the file is missing, resume throws
 * immediately. A separate test case below (`does not claim hasOptimizerState
 * when no training step has run`) pins the no-op path so the fix stays
 * fixed.
 *
 * To force the pipeline to actually populate optimizer state we need three
 * things:
 *
 *   1. The engine's degenerate-completion filter to let at least one
 *      completion through per step. The filter only rejects
 *      `finish_reason == "length"` with `tokens >= 0.9 * max`. So we need
 *      generation to finish via EOS or the repetition cutoff before hitting
 *      the length limit. The GRPO engine hardcodes `max_consecutive_tokens:
 *      16`, `max_ngram_repeats: 8`, and `ngram_size: 3` in its generation
 *      config (crates/mlx-core/src/grpo/engine.rs build_gen_config).
 *      Combined with `repetitionPenalty: 0.05` (< 1 REWARDS previously-
 *      sampled tokens — it divides positive logits by the penalty, which
 *      with 0.05 multiplies them by 20) and greedy decoding, the model
 *      collapses into a short repeating cycle within a few dozen tokens on
 *      almost every attempt. A small retry loop tolerates the rare attempt
 *      that escapes the cutoff.
 *
 *   2. The optimizer state map on the model thread to become non-empty.
 *      This happens as a side effect of `AdamW::update_batch` calling
 *      `init_state` the first time it sees each parameter name — regardless
 *      of the actual gradient values. So as long as ONE train step makes it
 *      past the degenerate-completion filter and runs
 *      `train_step_grpo_sync`'s optimizer-update branch, the state map is
 *      populated and `save_optimizer_state_sync` will write a real file
 *      instead of taking the empty-keys early return.
 *
 *   3. `save_model` to succeed at checkpoint time, which means the model
 *      weights must not contain any NaN/Inf values. We guarantee this by
 *      combining `learningRate: 0` with a CONSTANT reward function: with
 *      constant rewards the group-normalized advantages are exactly 0, so
 *      the policy gradient is exactly 0, so AdamW computes `new_m = 0,
 *      new_v = 0, update = 0 / (0 + eps) = 0, lr_update = 0`, and the
 *      parameter update is an identity no-op. This avoids the IEEE
 *      floating-point footgun where `lr=0 * update=inf = NaN` poisons the
 *      weights — a non-hypothetical risk even with `gradientClipNorm` +
 *      element-wise clipping, because the update denominator can get very
 *      small in bf16 on the first real gradient of a random-init model.
 *
 * We deliberately run in the same TINY_TEST_CONFIG the other GRPO trainer
 * tests use — this combination of head_dim / hidden_size / num_heads has
 * been verified to work on the Metal autograd backward path elsewhere
 * (see `grpo-autograd-integration.test.ts`, which runs a full autograd
 * train step on exactly this shape with `maxCompletionLength: 256`).
 */

import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { Qwen3Model } from '@mlx-node/core';
import { loadModel } from '@mlx-node/lm';
import { GRPOTrainer, type RewardOutput } from '@mlx-node/trl';
import { afterAll, describe, expect, it } from 'vite-plus/test';

import { createTempModel } from '../test-model-utils';

// Constant reward: every completion gets the same score. Group normalization
// turns these into advantages of exactly 0, so the policy gradient is exactly
// 0 everywhere. This is intentional — we want to populate AdamW moment state
// (which happens via `init_state` the moment `update_batch` is called on a
// previously-unseen param) WITHOUT actually perturbing the weights.
//
// With zero gradients:
//   new_m = β1*0 + (1-β1)*0 = 0
//   new_v = β2*0 + (1-β2)*0 = 0
//   update = 0 / (sqrt(0) + eps) = 0
//   new_param = param - lr*0 = param  (exactly, no FP drama)
//
// So weights stay pristine (saveable), but the optimizer state map has been
// populated by `init_state`, so `get_state_keys()` is non-empty and
// `save_optimizer_state_sync` writes a real safetensors file instead of
// taking the empty-keys early return. That's exactly what the round-trip
// test needs.
//
// Note: group normalization on constant rewards could hit a 0/0 if std is
// measured as exactly 0; the engine's `compute_advantages` handles that by
// clamping std to a small epsilon, producing finite (zero) advantages. This
// has been verified empirically — the train step returns `gradientsApplied
// === true` with this reward as long as the engine's degenerate-completion
// filter lets the completions through (see commentary further below).
const constantReward = (outputs: RewardOutput[]): Float32Array => Float32Array.from(outputs.map(() => 1.0));

interface TrainingStateJson {
  step: number;
  epoch: number;
  timestamp: string;
  hasOptimizerState: boolean;
}

describe.sequential('GRPOTrainer checkpoint + optimizer state round-trip', () => {
  // Resources created *inside* `runOnce` are tracked here so the outer
  // retry loop and `afterAll` can clean up across attempts. We hold these
  // at describe scope rather than `beforeAll` because some attempts may
  // need to discard a poisoned random-init model and create a fresh one.
  const cleanups: Array<() => void> = [];

  afterAll(() => {
    for (const fn of cleanups) {
      try {
        fn();
      } catch (err) {
        console.warn('Cleanup failed:', err);
      }
    }
  });

  /**
   * Single attempt at the full round-trip. Throws on any assertion or
   * numerical failure; the outer loop in `it(...)` catches those and
   * re-runs with a fresh random-init model. We isolate per-attempt state
   * (model, checkpoint dir, trainers) inside this helper so a failed
   * attempt can't leak an MxArray or checkpoint path into the next.
   *
   * Returns the final `m4.loss` on success for the outer loop's logging.
   */
  async function runOnce(): Promise<{ attempts: number; finalLoss: number }> {
    const tempModel = await createTempModel();
    const checkpointDir = mkdtempSync(join(tmpdir(), 'mlx-grpo-opt-roundtrip-'));
    cleanups.push(() => {
      try {
        tempModel.cleanup();
      } catch (err) {
        console.warn('Failed to cleanup temp model:', err);
      }
      if (existsSync(checkpointDir)) {
        try {
          rmSync(checkpointDir, { recursive: true, force: true });
        } catch (err) {
          console.warn(`Failed to cleanup checkpoint dir ${checkpointDir}:`, err);
        }
      }
    });

    // --- Phase 1: initial trainer, run a few steps ------------------------
    const loadedA = await loadModel(tempModel.modelPath);
    expect(loadedA).toBeInstanceOf(Qwen3Model);
    const modelA = loadedA as unknown as Qwen3Model;

    const sharedTrainerOptions = {
      modelName: 'qwen3-tiny-roundtrip',
      modelPath: tempModel.modelPath, // so saveCheckpoint can find tokenizer.json
      groupSize: 2,
      // maxCompletionLength large enough to give the repetition cutoff
      // room to fire before we hit the length limit. The GRPO engine's
      // generation config hardcodes `max_consecutive_tokens: 16`,
      // `max_ngram_repeats: 8`, and `ngram_size: 3`
      // (crates/mlx-core/src/grpo/engine.rs build_gen_config), so the
      // consecutive-same cutoff fires after 16 identical tokens in a row
      // and the ngram cutoff fires after 16 tokens of an ABAB pattern or
      // 24 tokens of an ABCABC pattern. With `repetitionPenalty: 0.05`
      // below, we strongly bias the model toward re-picking previously
      // sampled tokens, so one of those cutoffs fires well before the
      // 0.9 * 128 = 115 length threshold.
      maxCompletionLength: 128,
      // topK=1 + tiny temperature ≈ greedy argmax. Combined with a strong
      // bonus for previously-sampled tokens via repetitionPenalty<1 (see
      // below), the model collapses into a short repeating cycle within a
      // few dozen tokens.
      temperature: 0.01,
      topK: 1,
      topP: 1.0,
      // repetitionPenalty < 1 REWARDS previously-sampled tokens instead of
      // penalizing them — `apply_repetition_penalty` divides positive
      // logits by `penalty`, so penalty=0.05 multiplies them by 20. On a
      // tiny random-weight model with already-near-uniform logits that's
      // enough to force the sampler to keep picking the same handful of
      // tokens, which in turn triggers the 16-consecutive-token and/or
      // ngram-repeat cutoffs hardcoded in the GRPO engine's gen config.
      // Without this the random-weight Qwen3 wanders in a long cycle and
      // all completions finish with reason="length", which the engine's
      // degenerate-completion filter rejects (see engine.rs ~1000).
      repetitionPenalty: 0.05,
      // Zero learning rate + constant rewards = deterministic no-op weight
      // update. Constant rewards produce zero advantages (see `constantReward`
      // above), which make all policy gradients exactly zero. With zero
      // gradients AdamW's `update_single_at_step` computes
      //   new_m = 0.9*0 + 0.1*0 = 0
      //   new_v = 0.999*0 + 0.001*0 = 0
      //   update = 0 / (sqrt(0) + eps) = 0
      //   new_param = param - 0*update = param
      // exactly — no bf16/IEEE floating-point edge cases. The state map
      // is still populated (via `init_state` on each param's first visit),
      // so `get_state_keys()` is non-empty and `save_optimizer_state_sync`
      // writes a real file. The weights stay pristine, so `save_model`'s
      // NaN/Inf validation passes.
      //
      // NOTE: `lr=0` alone (with non-zero gradients) is NOT enough, because
      // of an IEEE floating-point footgun: if `update = corrected_m /
      // (sqrt(corrected_v) + eps)` ever contains `inf` (possible in bf16
      // with extreme gradients), then `update * 0 = NaN`, which poisons
      // the weight through `new_param = param - NaN`. Zero gradients are
      // the only numerically safe path.
      learningRate: 0,
      gradientClipNorm: 1.0,
      rewardFunction: constantReward,
      logConsole: false,
      outputDir: checkpointDir,
    } as const;

    const trainerA = new GRPOTrainer(modelA, sharedTrainerOptions);

    const prompts = [[{ role: 'user' as const, content: 'hi' }]];

    // Run training steps in a retry loop until we see at least one step that
    // actually applied gradients. We need a successful gradient application
    // for `save_optimizer_state_sync` to produce a non-empty file — it
    // short-circuits with `Ok(())` when AdamW's state map is empty (via
    // `get_state_keys().is_empty()`), and the state map only becomes
    // non-empty once `update_batch` has been called at least once (it calls
    // `init_state` the first time each param name is seen).
    //
    // Why retry instead of running a fixed number of steps: on a tiny random-
    // weight Qwen3 (2 layers, hidden_size=64), the GRPO engine's degenerate-
    // completion filter occasionally rejects every rollout in a step
    // (when the random sampling path happens to avoid both the 16-consecutive-
    // token cutoff AND the ABAB/ABCABC ngram-repeat cutoffs and generation
    // runs right up to `maxCompletionLength`). When all completions are
    // filtered, the engine early-returns with `gradients_applied=false`
    // *before* ever calling `train_step_grpo_sync`, so the optimizer never
    // sees a gradient and `init_state` is never called. Retrying a few
    // times lets the sampler re-roll until it hits the repetition cutoff
    // (empirically this almost always happens within the first 1–2 attempts
    // with `repetitionPenalty: 0.05`).
    //
    // Why only ONE successful step before save: on a tiny random-weight
    // model the first GRPO loss is huge and even with `lr=0` + `grad_clip`
    // there are floating-point edge cases in the AdamW update path
    // (`update * 0 = NaN` when `update` momentarily contains `inf` from
    // `corrected_m / (sqrt(corrected_v) + eps)`) that can occasionally
    // corrupt weights. Constant rewards keep advantages (and therefore
    // gradients) at exactly zero, which avoids this problem entirely —
    // but we still stop after the first success to minimize exposure.
    const maxTrainStepAttempts = 20;
    let gradientsAppliedCount = 0;
    let lastFiniteGradAppliedLoss = Number.NaN;
    const metricsLog: Array<{
      attempt: number;
      step: number | null;
      loss: number | null;
      gradientsApplied: boolean;
      error?: string;
    }> = [];
    for (let attempt = 0; attempt < maxTrainStepAttempts; attempt++) {
      try {
        const m = await trainerA.trainStep(prompts);
        metricsLog.push({
          attempt,
          step: m.step,
          loss: m.loss,
          gradientsApplied: m.gradientsApplied ?? false,
        });
        if (m.gradientsApplied === true && Number.isFinite(m.loss)) {
          gradientsAppliedCount += 1;
          lastFiniteGradAppliedLoss = m.loss;
          // Stop as soon as we have one successful gradient application.
          // Running additional steps only increases the risk of cumulative
          // numerical drift corrupting the weights before we get to save.
          break;
        }
      } catch (err) {
        metricsLog.push({
          attempt,
          step: null,
          loss: null,
          gradientsApplied: false,
          error: err instanceof Error ? err.message : String(err),
        });
        // Keep retrying — a transient NaN in a previous step should not
        // fail the test; the retry will use fresh samples.
      }
    }

    expect(
      gradientsAppliedCount,
      `expected at least one pre-save step to actually apply gradients so AdamW moments are populated; metrics: ${JSON.stringify(metricsLog)}`,
    ).toBeGreaterThan(0);
    expect(Number.isFinite(lastFiniteGradAppliedLoss)).toBe(true);

    // Record the exact step count after the loop so downstream assertions
    // match whatever the retry loop produced (could be > 1 if early attempts
    // were filtered, even though we break on the first gradient-applied step).
    const stepsBeforeSave = trainerA.getStep();
    expect(stepsBeforeSave).toBeGreaterThan(0);

    // --- Phase 2: save checkpoint -----------------------------------------
    // `saveCheckpoint(name)` writes to `join(outputDir, name)`. Before
    // f281143 this aborted the process on a loaded (thread-backed) model
    // because `Qwen3Model::save_model` touched weight MxArrays off-thread.
    const checkpointName = `checkpoint-${stepsBeforeSave}`;
    const checkpointPath = await trainerA.saveCheckpoint(checkpointName);
    expect(checkpointPath).toBe(join(checkpointDir, checkpointName));

    // Sanity-check the checkpoint layout: weights, tokenizer, config, the
    // JSON blob the resume path reads, AND the optimizer-state safetensors.
    // The optimizer file is the load-bearing check for commit 9eff18a —
    // without a real SaveOptimizerState command wired through the model
    // thread this file would not exist.
    const checkpointFiles = readdirSync(checkpointPath);
    expect(checkpointFiles).toContain('training_state.json');
    expect(checkpointFiles).toContain('config.json');
    expect(checkpointFiles).toContain('tokenizer.json');
    expect(checkpointFiles.some((name: string) => name === 'weights.safetensors' || name === 'weights.mlx')).toBe(true);

    const optimizerStatePath = join(checkpointPath, 'optimizer_state.safetensors');
    expect(
      existsSync(optimizerStatePath),
      'optimizer_state.safetensors must exist on disk after saveCheckpoint — if this fails it means the train steps never populated AdamW moments and save_optimizer_state_sync took the empty-keys early return',
    ).toBe(true);
    // And it should be non-trivial in size (real tensor data, not just
    // metadata headers) — catches a regression where SaveOptimizerState
    // silently becomes a no-op again.
    expect(statSync(optimizerStatePath).size).toBeGreaterThan(128);

    const stateJson: TrainingStateJson = JSON.parse(readFileSync(join(checkpointPath, 'training_state.json'), 'utf-8'));
    expect(stateJson.step).toBe(stepsBeforeSave);
    expect(stateJson.epoch).toBe(0);
    expect(stateJson.hasOptimizerState).toBe(true);

    // --- Phase 3: construct a fresh trainer via resumeFromCheckpoint ------
    // This is the user-visible surface of commit 9eff18a: internally
    // `GRPOTrainer.create` calls `engine.loadOptimizerState(...)`, which
    // used to be a silent `warn!` no-op and now dispatches through the
    // dedicated model thread via the `LoadOptimizerState` command.
    const trainerB = await GRPOTrainer.create({
      ...sharedTrainerOptions,
      resumeFromCheckpoint: checkpointPath,
    });

    // The JS-side step counter is restored from training_state.json before
    // any new trainStep runs.
    expect(trainerB.getStep()).toBe(stepsBeforeSave);

    // --- Phase 4: run one more step on the resumed trainer ----------------
    // The fresh trainer must still be able to run a training step — proves
    // the model thread is live and `loadOptimizerState` rehydrated the
    // AdamW moment tensors without leaving the engine in a broken state.
    const m4 = await trainerB.trainStep(prompts);
    expect(typeof m4.loss).toBe('number');
    expect(Number.isFinite(m4.loss)).toBe(true);
    // Step counter advances by exactly one: stepsBeforeSave → stepsBeforeSave + 1.
    expect(trainerB.getStep()).toBe(stepsBeforeSave + 1);

    // Loose order-of-magnitude check: if the optimizer state round-trip
    // utterly broke the model (e.g. moment tensors corrupted on load), the
    // post-resume loss would explode relative to the pre-save value. We
    // bound it very loosely because the GRPO loss on a tiny random-weight
    // model varies enormously between steps (different completions every
    // call, very sensitive initial logprobs), so anything tighter would be
    // flaky. The point of this check is to fail loud if the AdamW state
    // load left the optimizer in a corrupt state that then corrupts the
    // next forward.
    const preSaveScale = Math.max(Math.abs(lastFiniteGradAppliedLoss), 1.0);
    expect(
      Math.abs(m4.loss),
      `post-resume loss ${m4.loss} exploded relative to pre-save loss ${lastFiniteGradAppliedLoss}`,
    ).toBeLessThan(preSaveScale * 1000 + 1e6);

    return { attempts: stepsBeforeSave, finalLoss: m4.loss };
  }

  it('populates AdamW moments, saves them through the model thread, and restores them on resume', async () => {
    // Outer retry loop: the pre-save training step can fail deterministically
    // on certain random-weight initializations — specifically when the tiny
    // 2-layer Qwen3 produces logits extreme enough that the first forward
    // pass emits a ±inf per-token log-probability somewhere, which then turns
    // `log_ratio = new_logp - old_logp` into NaN in `grpo_loss` (both logps
    // are -inf, so `(-inf) - (-inf) = NaN`). That NaN short-circuits
    // `train_step_grpo_sync` via the `if loss_value.is_nan()` path BEFORE
    // the optimizer is ever called, and since the failure is driven by the
    // (fixed) random weights, retrying with the same model always produces
    // the same NaN. The only way to escape is to re-create the model with
    // a fresh random initialization.
    //
    // Empirically fewer than ~10% of random inits trigger the extreme-logit
    // path on TINY_TEST_CONFIG, so a 4-attempt budget is comfortably above
    // any realistic flake rate while keeping worst-case runtime bounded.
    const maxOuterAttempts = 4;
    const attemptErrors: string[] = [];
    let lastResult: { attempts: number; finalLoss: number } | null = null;
    for (let outer = 0; outer < maxOuterAttempts; outer++) {
      try {
        lastResult = await runOnce();
        break;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        attemptErrors.push(`attempt ${outer}: ${msg}`);
        // Keep going on any failure — either a deterministic NaN from a
        // bad random init (recoverable with a fresh model) or a transient
        // numerical issue (recoverable on retry).
      }
    }
    if (lastResult === null) {
      throw new Error(
        `GRPO optimizer round-trip failed after ${maxOuterAttempts} attempts with fresh random-init models:\n` +
          attemptErrors.join('\n'),
      );
    }
  });

  // Lock-in test for task H3 (fix saveCheckpoint hasOptimizerState lie).
  //
  // Before the fix, `saveCheckpoint` unconditionally set
  // `state.hasOptimizerState = true` whenever `engine.saveOptimizerState()`
  // returned successfully. But `save_optimizer_state_sync` on the Rust side
  // legitimately returns `Ok(())` without writing a file in two cases:
  //   (a) SGD / no optimizer configured.
  //   (b) AdamW configured but no training step has ever populated the
  //       state map (e.g. we checkpoint right after construction, or every
  //       rollout was filtered by the degenerate-completion filter).
  //
  // This test exercises case (b): construct a trainer, save a checkpoint
  // WITHOUT running any training step, and verify that:
  //   1. `optimizer_state.safetensors` does NOT exist on disk
  //   2. `training_state.json` has `hasOptimizerState === false`
  //   3. Resuming from that checkpoint does NOT throw (because the flag
  //      correctly says there is nothing to restore), and does not try
  //      to read the missing file.
  it('does not claim hasOptimizerState when no training step has run', async () => {
    const tempModel = await createTempModel();
    const checkpointDir = mkdtempSync(join(tmpdir(), 'mlx-grpo-opt-noop-'));
    cleanups.push(() => {
      try {
        tempModel.cleanup();
      } catch (err) {
        console.warn('Failed to cleanup temp model:', err);
      }
      if (existsSync(checkpointDir)) {
        try {
          rmSync(checkpointDir, { recursive: true, force: true });
        } catch (err) {
          console.warn(`Failed to cleanup checkpoint dir ${checkpointDir}:`, err);
        }
      }
    });

    const loaded = await loadModel(tempModel.modelPath);
    expect(loaded).toBeInstanceOf(Qwen3Model);
    const model = loaded as unknown as Qwen3Model;

    const trainerOptions = {
      modelName: 'qwen3-tiny-noop-save',
      modelPath: tempModel.modelPath,
      groupSize: 2,
      maxCompletionLength: 16,
      learningRate: 0,
      rewardFunction: constantReward,
      logConsole: false,
      outputDir: checkpointDir,
    } as const;

    const trainer = new GRPOTrainer(model, trainerOptions);

    // Save a checkpoint WITHOUT running any training step. AdamW's state
    // map is empty at this point (init_state is only called the first time
    // `update_batch` sees a parameter name, which happens inside
    // `train_step_grpo_sync`). So `save_optimizer_state_sync` will take the
    // `keys.is_empty()` early return and NOT write a file.
    const checkpointName = 'checkpoint-0';
    const checkpointPath = await trainer.saveCheckpoint(checkpointName);
    expect(checkpointPath).toBe(join(checkpointDir, checkpointName));

    // Verify the safetensors file was NOT written.
    const optimizerStatePath = join(checkpointPath, 'optimizer_state.safetensors');
    expect(
      existsSync(optimizerStatePath),
      'optimizer_state.safetensors must NOT exist when no training step has populated AdamW moments',
    ).toBe(false);

    // And the JSON flag must reflect reality.
    const stateJson: TrainingStateJson = JSON.parse(readFileSync(join(checkpointPath, 'training_state.json'), 'utf-8'));
    expect(
      stateJson.hasOptimizerState,
      'saveCheckpoint must NOT claim hasOptimizerState=true when no file was written',
    ).toBe(false);

    // Resume path: hasOptimizerState=false means the resume code must NOT
    // attempt to load the missing file. Before task H3 this path was gated
    // correctly already, but the *value* of the flag was wrong. With the fix,
    // resume must succeed without throwing on a missing optimizer file,
    // precisely because `hasOptimizerState` is honestly reported as false.
    const trainerResumed = await GRPOTrainer.create({
      ...trainerOptions,
      resumeFromCheckpoint: checkpointPath,
    });
    expect(trainerResumed.getStep()).toBe(0);
  });

  // Lock-in test for the "fail loud on lying checkpoint" resume-path
  // hardening added alongside the saveCheckpoint fix. Manually constructs
  // a checkpoint directory whose `training_state.json` claims
  // `hasOptimizerState: true` but omits `optimizer_state.safetensors`, then
  // asserts that `GRPOTrainer.create({ resumeFromCheckpoint })` throws.
  //
  // Before task H3 the resume path's try/catch around `loadOptimizerState`
  // silently swallowed the missing file, leaving the caller with a fresh
  // optimizer and no indication of the drift.
  it('fails loudly when a resumed checkpoint lies about hasOptimizerState', async () => {
    const tempModel = await createTempModel();
    const checkpointDir = mkdtempSync(join(tmpdir(), 'mlx-grpo-opt-lie-'));
    cleanups.push(() => {
      try {
        tempModel.cleanup();
      } catch (err) {
        console.warn('Failed to cleanup temp model:', err);
      }
      if (existsSync(checkpointDir)) {
        try {
          rmSync(checkpointDir, { recursive: true, force: true });
        } catch (err) {
          console.warn(`Failed to cleanup checkpoint dir ${checkpointDir}:`, err);
        }
      }
    });

    const loaded = await loadModel(tempModel.modelPath);
    const model = loaded as unknown as Qwen3Model;

    const trainerOptions = {
      modelName: 'qwen3-tiny-lie',
      modelPath: tempModel.modelPath,
      groupSize: 2,
      maxCompletionLength: 16,
      learningRate: 0,
      rewardFunction: constantReward,
      logConsole: false,
      outputDir: checkpointDir,
    } as const;

    // Step 1: produce a legitimate checkpoint via saveCheckpoint WITHOUT
    // running a training step (so the model weights are saved but no
    // optimizer state file is produced).
    const trainer = new GRPOTrainer(model, trainerOptions);
    const checkpointName = 'checkpoint-lie';
    const checkpointPath = await trainer.saveCheckpoint(checkpointName);

    // Sanity: the honest save did NOT write the optimizer file and did NOT
    // claim hasOptimizerState. This is the precondition for the lie we're
    // about to inject.
    const optimizerStatePath = join(checkpointPath, 'optimizer_state.safetensors');
    expect(existsSync(optimizerStatePath)).toBe(false);
    const statePath = join(checkpointPath, 'training_state.json');
    const honestState: TrainingStateJson = JSON.parse(readFileSync(statePath, 'utf-8'));
    expect(honestState.hasOptimizerState).toBe(false);

    // Step 2: INJECT the lie — rewrite training_state.json to falsely claim
    // hasOptimizerState=true while leaving the safetensors file absent.
    const lyingState = { ...honestState, hasOptimizerState: true };
    writeFileSync(statePath, JSON.stringify(lyingState, null, 2));
    expect(existsSync(optimizerStatePath)).toBe(false);

    // Step 3: resume must throw. The exact shape of the error matters less
    // than the fact that it fails instead of silently loading nothing.
    await expect(
      GRPOTrainer.create({
        ...trainerOptions,
        resumeFromCheckpoint: checkpointPath,
      }),
    ).rejects.toThrow(/hasOptimizerState=true but .* does not exist/);
  });

  // Lock-in test for the stale-file reuse hole Codex caught. If the
  // checkpoint directory already contains a leftover
  // `optimizer_state.safetensors` from a previous save and the current
  // save is a legitimate no-op (SGD / empty AdamW state), the old file
  // must not be allowed to masquerade as fresh state for this save.
  // `saveCheckpoint` unlinks the file up-front so `existsSync` after the
  // save reflects only what the current save produced.
  it('removes stale optimizer_state.safetensors when the current save is a no-op', async () => {
    const tempModel = await createTempModel();
    const checkpointDir = mkdtempSync(join(tmpdir(), 'mlx-grpo-opt-stale-'));
    cleanups.push(() => {
      try {
        tempModel.cleanup();
      } catch (err) {
        console.warn('Failed to cleanup temp model:', err);
      }
      if (existsSync(checkpointDir)) {
        try {
          rmSync(checkpointDir, { recursive: true, force: true });
        } catch (err) {
          console.warn(`Failed to cleanup checkpoint dir ${checkpointDir}:`, err);
        }
      }
    });

    const loaded = await loadModel(tempModel.modelPath);
    const model = loaded as unknown as Qwen3Model;

    const trainerOptions = {
      modelName: 'qwen3-tiny-stale',
      modelPath: tempModel.modelPath,
      groupSize: 2,
      maxCompletionLength: 16,
      learningRate: 0,
      rewardFunction: constantReward,
      logConsole: false,
      outputDir: checkpointDir,
    } as const;

    const trainer = new GRPOTrainer(model, trainerOptions);

    // Pre-populate the checkpoint directory with a BOGUS
    // optimizer_state.safetensors that looks like it came from a previous
    // save. This is exactly the corruption scenario: the save below will
    // be a no-op (no training step has run), so without the unlink fix
    // the stale file would survive and `existsSync` would lie.
    const checkpointName = 'checkpoint-stale';
    const checkpointPath = join(checkpointDir, checkpointName);
    mkdirSync(checkpointPath, { recursive: true });
    const optimizerStatePath = join(checkpointPath, 'optimizer_state.safetensors');
    writeFileSync(optimizerStatePath, 'STALE FROM PREVIOUS SAVE');
    expect(existsSync(optimizerStatePath)).toBe(true);

    // Save a no-op checkpoint. saveCheckpoint must unlink the stale file
    // before calling saveOptimizerState, so the post-save existsSync
    // accurately reflects that this save produced nothing.
    await trainer.saveCheckpoint(checkpointName);

    expect(
      existsSync(optimizerStatePath),
      'saveCheckpoint must unlink a stale optimizer_state.safetensors before a no-op save',
    ).toBe(false);

    const stateJson: TrainingStateJson = JSON.parse(readFileSync(join(checkpointPath, 'training_state.json'), 'utf-8'));
    expect(
      stateJson.hasOptimizerState,
      'hasOptimizerState must be false when the current save produced no file, even if a stale one was present',
    ).toBe(false);
  });
});
