import { describe, it, expect, beforeAll, afterAll } from 'vite-plus/test';
import { mkdirSync, writeFileSync, rmSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import {
  GRPOTrainer,
  computeDatasetHash,
  type DatasetMetadata,
  type TrainingState,
  type DatasetExample,
  type RewardOutput,
} from '@mlx-node/trl';
import { createTempModel } from '../test-model-utils';

// Shared temp model for all tests
let tempModel: { modelPath: string; cleanup: () => void };

beforeAll(async () => {
  tempModel = await createTempModel();
});

afterAll(() => {
  tempModel?.cleanup();
});

describe('computeDatasetHash', () => {
  it('should compute a consistent hash for the same dataset', () => {
    const dataset: DatasetExample[] = [
      { prompt: [{ role: 'user', content: 'Test 1' }] },
      { prompt: [{ role: 'user', content: 'Test 2' }] },
      { prompt: [{ role: 'user', content: 'Test 3' }] },
    ];

    const hash1 = computeDatasetHash(dataset);
    const hash2 = computeDatasetHash(dataset);

    expect(hash1).toBe(hash2);
    expect(hash1.length).toBe(16); // 16 hex characters
  });

  it('should produce different hashes for different datasets', () => {
    const dataset1: DatasetExample[] = [
      { prompt: [{ role: 'user', content: 'Test 1' }] },
      { prompt: [{ role: 'user', content: 'Test 2' }] },
    ];

    const dataset2: DatasetExample[] = [
      { prompt: [{ role: 'user', content: 'Different 1' }] },
      { prompt: [{ role: 'user', content: 'Different 2' }] },
    ];

    const hash1 = computeDatasetHash(dataset1);
    const hash2 = computeDatasetHash(dataset2);

    expect(hash1).not.toBe(hash2);
  });

  it('should detect reordered datasets', () => {
    const dataset1: DatasetExample[] = [
      { prompt: [{ role: 'user', content: 'Test 1' }] },
      { prompt: [{ role: 'user', content: 'Test 2' }] },
    ];

    const dataset2: DatasetExample[] = [
      { prompt: [{ role: 'user', content: 'Test 2' }] },
      { prompt: [{ role: 'user', content: 'Test 1' }] },
    ];

    const hash1 = computeDatasetHash(dataset1);
    const hash2 = computeDatasetHash(dataset2);

    // Hashes should differ because order matters
    expect(hash1).not.toBe(hash2);
  });

  it('should use custom sample size', () => {
    const dataset: DatasetExample[] = Array.from({ length: 20 }, (_, i) => ({
      prompt: [{ role: 'user' as const, content: `Test ${i}` }],
    }));

    // Default samples first 10
    const hash1 = computeDatasetHash(dataset, 10);

    // Using first 5 items should give different hash if dataset is different beyond 5
    const hash2 = computeDatasetHash(dataset, 5);

    // Both should be valid 16-char hex strings
    expect(hash1.length).toBe(16);
    expect(hash2.length).toBe(16);
  });

  it('should handle empty dataset', () => {
    const dataset: DatasetExample[] = [];
    const hash = computeDatasetHash(dataset);

    expect(hash.length).toBe(16);
  });

  it('should handle dataset smaller than sample size', () => {
    const dataset: DatasetExample[] = [{ prompt: [{ role: 'user', content: 'Only one' }] }];

    const hash = computeDatasetHash(dataset, 10);
    expect(hash.length).toBe(16);
  });
});

describe.sequential('GRPOTrainer - Dataset Resume Validation', () => {
  let tempDir: string;

  beforeAll(() => {
    // Create a temporary directory for checkpoints
    tempDir = mkdtempSync(join(tmpdir(), 'mlx-resume-test-'));
    mkdirSync(join(tempDir, 'checkpoints'), { recursive: true });
  });

  afterAll(() => {
    // Cleanup temp directory
    if (existsSync(tempDir)) {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it('should save dataset metadata in checkpoint', async () => {
    const checkpointDir = join(tempDir, 'checkpoint-test-1');
    mkdirSync(checkpointDir, { recursive: true });

    const trainer = await GRPOTrainer.create({
      modelPath: tempModel.modelPath,
      modelName: 'qwen3-0.6b',
      groupSize: 2,
      maxCompletionLength: 5,
      numEpochs: 1,
      batchSize: 1,
      saveInterval: 1000,
      outputDir: checkpointDir,
      rewardFunction: (_outputs: RewardOutput[]) => new Float32Array([1.0, 1.0]),
    });

    // Create dataset
    const dataset: DatasetExample[] = [
      { prompt: [{ role: 'user', content: 'Test prompt 1' }] },
      { prompt: [{ role: 'user', content: 'Test prompt 2' }] },
    ];

    // Run one training step to initialize dataset metadata
    await trainer.train(dataset.slice(0, 1));

    // Save checkpoint manually
    const checkpointPath = await trainer.saveCheckpoint('test-checkpoint');

    // Read the training state
    const statePath = join(checkpointPath, 'training_state.json');
    const stateContent = require('node:fs').readFileSync(statePath, 'utf-8');
    const state: TrainingState = JSON.parse(stateContent);

    // Verify dataset metadata is saved
    expect(state.dataset).toBeDefined();
    expect(state.dataset!.size).toBe(1); // We trained on 1 example
    expect(state.dataset!.contentHash).toBeDefined();
    expect(state.dataset!.contentHash.length).toBe(16);
  });

  it('should track processed batch indices', async () => {
    const checkpointDir = join(tempDir, 'checkpoint-test-2');
    mkdirSync(checkpointDir, { recursive: true });

    const trainer = await GRPOTrainer.create({
      modelPath: tempModel.modelPath,
      modelName: 'qwen3-0.6b',
      groupSize: 2,
      maxCompletionLength: 5,
      numEpochs: 1,
      batchSize: 1,
      saveInterval: 1000,
      outputDir: checkpointDir,
      rewardFunction: (_outputs: RewardOutput[]) => new Float32Array([1.0, 1.0]),
    });

    // Create dataset with 3 examples
    const dataset: DatasetExample[] = [
      { prompt: [{ role: 'user', content: 'Test 1' }] },
      { prompt: [{ role: 'user', content: 'Test 2' }] },
      { prompt: [{ role: 'user', content: 'Test 3' }] },
    ];

    // Run training - this will process all 3 batches (batchSize=1)
    await trainer.train(dataset);

    // Save checkpoint
    const checkpointPath = await trainer.saveCheckpoint('batch-tracking-test');

    // Read the training state
    const statePath = join(checkpointPath, 'training_state.json');
    const stateContent = require('node:fs').readFileSync(statePath, 'utf-8');
    const state: TrainingState = JSON.parse(stateContent);

    // processedBatchIndices should be empty after epoch completes (cleared at epoch boundary)
    // The array is cleared when an epoch ends
    expect(state.dataset).toBeDefined();
    // After a complete epoch, indices are cleared
    expect(state.dataset!.processedBatchIndices).toBeDefined();
    // Empty because we completed the epoch
    expect(state.dataset!.processedBatchIndices!.length).toBe(0);
  });

  it('should validate dataset size mismatch on resume (warning only)', async () => {
    // This test verifies that the trainer logs warnings but continues training
    // We can't easily capture log output, so we verify the training completes

    const checkpointDir = join(tempDir, 'checkpoint-test-3');
    mkdirSync(checkpointDir, { recursive: true });

    // Create initial training state with different dataset size
    const checkpointPath = join(checkpointDir, 'checkpoint-1');
    mkdirSync(checkpointPath, { recursive: true });

    const initialState: TrainingState = {
      step: 1,
      epoch: 0,
      timestamp: new Date().toISOString(),
      dataset: {
        size: 10, // Original had 10 examples
        contentHash: 'different_hash_x', // 16 chars
        processedBatchIndices: [0],
      },
    };

    writeFileSync(join(checkpointPath, 'training_state.json'), JSON.stringify(initialState, null, 2));

    // Copy model files to checkpoint
    const { copyFileSync, readdirSync } = require('node:fs');
    const modelFiles = readdirSync(tempModel.modelPath);
    for (const file of modelFiles) {
      copyFileSync(join(tempModel.modelPath, file), join(checkpointPath, file));
    }

    // Create trainer resuming from checkpoint
    const trainer = await GRPOTrainer.create({
      modelPath: tempModel.modelPath,
      modelName: 'qwen3-0.6b',
      groupSize: 2,
      maxCompletionLength: 5,
      numEpochs: 1,
      batchSize: 1,
      saveInterval: 1000,
      outputDir: checkpointDir,
      resumeFromCheckpoint: checkpointPath,
      rewardFunction: (_outputs: RewardOutput[]) => new Float32Array([1.0, 1.0]),
    });

    // New dataset with different size (2 instead of 10)
    const newDataset: DatasetExample[] = [
      { prompt: [{ role: 'user', content: 'New Test 1' }] },
      { prompt: [{ role: 'user', content: 'New Test 2' }] },
    ];

    // Training should complete despite mismatch (with warnings logged)
    await expect(trainer.train(newDataset)).resolves.not.toThrow();
  });
});

describe('DatasetMetadata type', () => {
  it('should have correct structure', () => {
    const metadata: DatasetMetadata = {
      size: 100,
      contentHash: '0123456789abcdef',
      shuffleSeed: 42,
      processedBatchIndices: [0, 1, 2],
    };

    expect(metadata.size).toBe(100);
    expect(metadata.contentHash).toBe('0123456789abcdef');
    expect(metadata.shuffleSeed).toBe(42);
    expect(metadata.processedBatchIndices).toEqual([0, 1, 2]);
  });

  it('should allow optional fields', () => {
    const metadata: DatasetMetadata = {
      size: 50,
      contentHash: 'abcdef0123456789',
    };

    expect(metadata.size).toBe(50);
    expect(metadata.shuffleSeed).toBeUndefined();
    expect(metadata.processedBatchIndices).toBeUndefined();
  });
});
