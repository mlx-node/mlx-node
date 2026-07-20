/**
 * GRPO checkpoint + optimizer state round-trip integration tests.
 *
 * The first test seeds a tiny checkpoint with deterministic AdamW moments,
 * then exercises load -> save -> load -> save through the model thread. This
 * covers both optimizer dispatch directions and thread-backed model saving
 * without depending on random generation to reach an optimizer update.
 */

import {
  cpSync,
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

import { createTempModel, TINY_TEST_CONFIG } from '../test-model-utils';

// The remaining no-op checkpoint tests never train, but keep a reward function
// configured so their trainer setup mirrors normal GRPO construction.
const constantReward = (outputs: RewardOutput[]): Float32Array => Float32Array.from(outputs.map(() => 1.0));

interface TrainingStateJson {
  step: number;
  epoch: number;
  timestamp: string;
  hasOptimizerState: boolean;
}

interface SafeTensorEntry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

const OPTIMIZER_PARAM = 'final_norm.weight';
const OPTIMIZER_STEP = 7;
const MOMENT_LENGTH = TINY_TEST_CONFIG.hiddenSize;
const FIRST_MOMENT = 0.125;
const SECOND_MOMENT = 0.25;

/** Write a minimal, standards-compliant AdamW SafeTensors fixture. */
function writeDeterministicOptimizerState(path: string): void {
  const tensorBytes = MOMENT_LENGTH * Float32Array.BYTES_PER_ELEMENT;
  const header = {
    __metadata__: {
      step: String(OPTIMIZER_STEP),
      format: 'adamw_optimizer_state',
    },
    [`${OPTIMIZER_PARAM}.m`]: {
      dtype: 'F32',
      shape: [MOMENT_LENGTH],
      data_offsets: [0, tensorBytes],
    },
    [`${OPTIMIZER_PARAM}.v`]: {
      dtype: 'F32',
      shape: [MOMENT_LENGTH],
      data_offsets: [tensorBytes, tensorBytes * 2],
    },
  };

  const rawHeader = Buffer.from(JSON.stringify(header), 'utf8');
  const headerLength = Math.ceil(rawHeader.length / 8) * 8;
  const file = Buffer.alloc(8 + headerLength + tensorBytes * 2, 0x20);
  file.writeBigUInt64LE(BigInt(headerLength), 0);
  rawHeader.copy(file, 8);

  const dataOffset = 8 + headerLength;
  for (let i = 0; i < MOMENT_LENGTH; i++) {
    file.writeFloatLE(FIRST_MOMENT, dataOffset + i * Float32Array.BYTES_PER_ELEMENT);
    file.writeFloatLE(SECOND_MOMENT, dataOffset + tensorBytes + i * Float32Array.BYTES_PER_ELEMENT);
  }
  writeFileSync(path, file);
}

/** Read enough of an AdamW SafeTensors file to verify exact round-trip data. */
function readOptimizerState(path: string): {
  metadata: Record<string, string>;
  firstMoment: number[];
  secondMoment: number[];
} {
  const file = readFileSync(path);
  const headerLength = Number(file.readBigUInt64LE(0));
  const header = JSON.parse(
    file
      .subarray(8, 8 + headerLength)
      .toString('utf8')
      .trimEnd(),
  ) as Record<string, SafeTensorEntry | Record<string, string>>;
  const dataOffset = 8 + headerLength;

  const readTensor = (name: string): number[] => {
    const entry = header[name] as SafeTensorEntry;
    expect(entry.dtype).toBe('F32');
    expect(entry.shape).toEqual([MOMENT_LENGTH]);
    const [start, end] = entry.data_offsets;
    expect(end - start).toBe(MOMENT_LENGTH * Float32Array.BYTES_PER_ELEMENT);
    return Array.from({ length: MOMENT_LENGTH }, (_, index) =>
      file.readFloatLE(dataOffset + start + index * Float32Array.BYTES_PER_ELEMENT),
    );
  };

  return {
    metadata: header.__metadata__ as Record<string, string>,
    firstMoment: readTensor(`${OPTIMIZER_PARAM}.m`),
    secondMoment: readTensor(`${OPTIMIZER_PARAM}.v`),
  };
}

function expectDeterministicOptimizerState(path: string): void {
  const state = readOptimizerState(path);
  expect(state.metadata).toMatchObject({
    step: String(OPTIMIZER_STEP),
    format: 'adamw_optimizer_state',
  });
  expect(state.firstMoment).toEqual(Array(MOMENT_LENGTH).fill(FIRST_MOMENT));
  expect(state.secondMoment).toEqual(Array(MOMENT_LENGTH).fill(SECOND_MOMENT));
}

describe.sequential('GRPOTrainer checkpoint + optimizer state round-trip', () => {
  // Track every temp model/checkpoint directory at describe scope so cleanup
  // still runs when an assertion or checkpoint load fails partway through.
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

  it('round-trips AdamW moments through model-thread checkpoint save and resume', async () => {
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

    // Build a real resumable checkpoint without running generation. The moment
    // tensors use final_norm.weight's actual shape in TINY_TEST_CONFIG.
    const seedCheckpointPath = join(checkpointDir, 'seed-checkpoint');
    cpSync(tempModel.modelPath, seedCheckpointPath, { recursive: true });
    const seedState: TrainingStateJson = {
      step: OPTIMIZER_STEP,
      epoch: 0,
      timestamp: new Date(0).toISOString(),
      hasOptimizerState: true,
    };
    writeFileSync(join(seedCheckpointPath, 'training_state.json'), JSON.stringify(seedState, null, 2));
    writeDeterministicOptimizerState(join(seedCheckpointPath, 'optimizer_state.safetensors'));
    expectDeterministicOptimizerState(join(seedCheckpointPath, 'optimizer_state.safetensors'));

    const trainerOptions = {
      modelName: 'qwen3-tiny-roundtrip',
      modelPath: tempModel.modelPath,
      groupSize: 2,
      learningRate: 0,
      rewardFunction: constantReward,
      logConsole: false,
      outputDir: checkpointDir,
    } as const;

    let trainerA: GRPOTrainer | undefined;
    let trainerB: GRPOTrainer | undefined;
    try {
      // First load proves LoadOptimizerState accepts and installs the seeded
      // moments. The following save can only write a file if that load was
      // real rather than the historical silent no-op.
      trainerA = await GRPOTrainer.create({
        ...trainerOptions,
        resumeFromCheckpoint: seedCheckpointPath,
      });
      expect(trainerA.getStep()).toBe(OPTIMIZER_STEP);

      const checkpointA = await trainerA.saveCheckpoint('roundtrip-a');
      expect(checkpointA).toBe(join(checkpointDir, 'roundtrip-a'));
      const stateA: TrainingStateJson = JSON.parse(readFileSync(join(checkpointA, 'training_state.json'), 'utf8'));
      expect(stateA).toMatchObject({
        step: OPTIMIZER_STEP,
        epoch: 0,
        hasOptimizerState: true,
      });
      expect(readdirSync(checkpointA)).toContain('config.json');
      expect(
        readdirSync(checkpointA).some((name: string) => name === 'weights.safetensors' || name === 'weights.mlx'),
      ).toBe(true);
      const optimizerA = join(checkpointA, 'optimizer_state.safetensors');
      expect(existsSync(optimizerA)).toBe(true);
      expect(statSync(optimizerA).size).toBeGreaterThan(128);
      expectDeterministicOptimizerState(optimizerA);

      // Load the freshly-saved checkpoint, then save once more. The second
      // output proves the resume path restored the exact moments into a fresh
      // model thread instead of merely validating the first file on disk.
      trainerB = await GRPOTrainer.create({
        ...trainerOptions,
        resumeFromCheckpoint: checkpointA,
      });
      expect(trainerB.getStep()).toBe(OPTIMIZER_STEP);

      const checkpointB = await trainerB.saveCheckpoint('roundtrip-b');
      const stateB: TrainingStateJson = JSON.parse(readFileSync(join(checkpointB, 'training_state.json'), 'utf8'));
      expect(stateB).toMatchObject({
        step: OPTIMIZER_STEP,
        epoch: 0,
        hasOptimizerState: true,
      });
      expectDeterministicOptimizerState(join(checkpointB, 'optimizer_state.safetensors'));
    } finally {
      // Drop model-thread optimizer tensors promptly; the temp directories are
      // removed by afterAll once every assertion has finished reading them.
      trainerB?.reset();
      trainerA?.reset();
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
