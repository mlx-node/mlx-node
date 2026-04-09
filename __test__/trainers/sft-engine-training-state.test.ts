/**
 * SFT training-state lifecycle tests.
 *
 * Covers three fixes to the training-state lifecycle on the model thread:
 *
 *   - H1 (single authoritative step counter): `ts.step` on the model thread
 *     is the single source of truth; the engine's `getStep()` is a
 *     read-through cache that never drifts.
 *
 *   - H2 (SFT resume plumb step): `restoreState(step, epoch)` must dispatch
 *     `SetTrainingStep` to the model thread so a subsequent `train_step_sft`
 *     increments from the restored value, not from 0.
 *
 *   - H5 (training-state teardown/reset): `reset()` must dispatch
 *     `ResetTraining` so a new `SftTrainingEngine` can be constructed on the
 *     same model without hitting the "Training already initialized" double-
 *     init guard.
 *
 * These tests drive `SftTrainingEngine` directly against a loaded tiny
 * random-init model — no real training step is required, because the
 * engine's state-management surface is what's under test. The model is
 * loaded via `loadModel()` so `model.thread` is populated (the SFT engine
 * requires a thread-backed model).
 */

import { MxArray, Qwen3Model, SftTrainingEngine, type SftEngineConfig } from '@mlx-node/core';
import { loadModel } from '@mlx-node/lm';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import { createTempModel } from '../test-model-utils';
import { int32, shape } from '../test-utils';

const sftConfig: SftEngineConfig = {
  learningRate: 2e-5,
  gradientAccumulationSteps: 1,
  gradientClipNorm: 1.0,
  weightDecay: 0.01,
  labelSmoothing: 0.0,
  maxNanGradients: 100,
  emergencySaveThreshold: 5,
  computeAccuracy: false,
  verboseNanDetection: false,
  gradientCheckpointing: true,
};

describe.sequential('SftTrainingEngine — training state lifecycle', () => {
  // Shared thread-backed model. `loadModel` spins up the dedicated OS thread
  // that `SftTrainingEngine` requires (it pulls `model.thread.cmd_sender()`).
  let model: Qwen3Model;
  let tempModel: { modelPath: string; cleanup: () => void };

  beforeAll(async () => {
    tempModel = await createTempModel();
    const loaded = await loadModel(tempModel.modelPath);
    expect(loaded).toBeInstanceOf(Qwen3Model);
    model = loaded as unknown as Qwen3Model;
  });

  afterAll(() => {
    tempModel?.cleanup();
  });

  it('restoreState() plumbs the step to the model thread (H2)', async () => {
    const engine = new SftTrainingEngine(model, sftConfig);
    try {
      // Fresh engine starts at step 0.
      expect(engine.getStep()).toBe(0);

      // Restore to a non-zero step. This must update BOTH the engine's
      // read-through cache AND the model thread's authoritative ts.step.
      engine.restoreState(42, 2);

      // Engine-side cache reflects the restored value.
      expect(engine.getStep()).toBe(42);
      expect(engine.getEpoch()).toBe(2);

      // End-to-end plumb verification: run an actual SFT train step.
      // `ts.step` on the model thread is incremented from the restored
      // value, and the returned metrics.step is the authoritative count.
      // If `restoreState()` had only updated the local cache (the exact
      // bug H2 fixes), the model thread would have started from 0 and
      // returned step=1 here instead of step=43.
      const seqLen = 8;
      const inputIds = MxArray.fromInt32(int32(1, 2, 3, 4, 5, 6, 7, 8), shape(1, seqLen));
      const labels = MxArray.fromInt32(int32(2, 3, 4, 5, 6, 7, 8, 9), shape(1, seqLen));
      const metrics = await engine.trainStep(inputIds, labels);
      expect(metrics.step).toBe(43);
      // And the engine-side cache is still consistent after the step.
      expect(engine.getStep()).toBe(43);
    } finally {
      // Clean up: drop training state on the model thread so the next
      // test's `new SftTrainingEngine(model, ...)` doesn't trip the
      // double-init guard.
      engine.reset();
    }
  });

  it('reset() drops training state so a new engine can be constructed (H5)', () => {
    // First engine. Its constructor sends InitTraining to the model thread.
    const engineA = new SftTrainingEngine(model, sftConfig);
    expect(engineA.getStep()).toBe(0);

    // A second construction BEFORE reset must fail with the double-init
    // guard introduced in d98598f. This is the contract `reset()` is
    // meant to release.
    expect(() => new SftTrainingEngine(model, sftConfig)).toThrow(/already initialized/i);

    // Reset tears down both engine-side state AND model-thread training
    // state (the new behavior being tested). After reset, a fresh engine
    // must succeed.
    engineA.reset();

    const engineB = new SftTrainingEngine(model, sftConfig);
    try {
      expect(engineB.getStep()).toBe(0);
      // Restoring step on the NEW engine proves the model thread's
      // training state is fresh (otherwise the restore call would apply
      // to stale state or error out).
      engineB.restoreState(7, 1);
      expect(engineB.getStep()).toBe(7);

      // CRITICAL (H5 widening fence): the old engineA handle still holds a
      // live cmd_sender pointing at engineB's model-thread training state.
      // After reset(), that handle is invalidated and any dispatching call
      // must refuse to touch engineB's state.
      expect(() => engineA.restoreState(999, 999)).toThrow(/invalidated/i);
      // engineB's step is unchanged — the old handle's restore was rejected.
      expect(engineB.getStep()).toBe(7);
    } finally {
      engineB.reset();
    }
  });

  it('restoreState() can be called multiple times; reset() zeros the step', () => {
    const engine = new SftTrainingEngine(model, sftConfig);
    try {
      engine.restoreState(10, 0);
      expect(engine.getStep()).toBe(10);

      // Subsequent restore overwrites.
      engine.restoreState(100, 0);
      expect(engine.getStep()).toBe(100);

      // Reset zeros it back out (engine-side cache), and also drops the
      // model-thread training state (verified in the next test).
      engine.reset();
      expect(engine.getStep()).toBe(0);
    } finally {
      // engine.reset() above already dropped training state on the
      // model thread; no further cleanup needed.
    }
  });
});
