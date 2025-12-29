import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { existsSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { OutputStore } from '@mlx-node/core';

describe('OutputStore', () => {
  let testDbPath: string;
  let store: OutputStore;

  beforeEach(async () => {
    // Create a unique temp file for each test
    testDbPath = join(tmpdir(), `output-store-test-${Date.now()}.db`);
    store = await OutputStore.local(testDbPath);
  });

  afterEach(async () => {
    // Clean up test database
    if (existsSync(testDbPath)) {
      try {
        unlinkSync(testDbPath);
      } catch {
        // Ignore errors
      }
    }
    // Also clean up WAL files if they exist
    if (existsSync(`${testDbPath}-wal`)) {
      try {
        unlinkSync(`${testDbPath}-wal`);
      } catch {
        // Ignore errors
      }
    }
    if (existsSync(`${testDbPath}-shm`)) {
      try {
        unlinkSync(`${testDbPath}-shm`);
      } catch {
        // Ignore errors
      }
    }
  });

  describe('local store creation', () => {
    it('should create a local database file', async () => {
      expect(existsSync(testDbPath)).toBe(true);
    });
  });

  describe('training run lifecycle', () => {
    it('should start and end a training run', async () => {
      const runId = await store.startRun('test-model', './models/test', '{"test": true}');
      expect(runId).toBeTruthy();
      expect(typeof runId).toBe('string');
      expect(runId.length).toBeGreaterThan(0);

      const currentRunId = await store.currentRunId();
      expect(currentRunId).toBe(runId);

      await store.endRun('completed');
      const afterEnd = await store.currentRunId();
      expect(afterEnd).toBeNull();
    });

    it('should list training runs', async () => {
      // Start a run
      const runId = await store.startRun('test-model-2', undefined, '{}');
      await store.endRun('completed');

      // List runs
      const runs = await store.listRuns(10, undefined);
      expect(runs.length).toBeGreaterThanOrEqual(1);

      const run = runs.find((r) => r.id === runId);
      expect(run).toBeTruthy();
      expect(run?.modelName).toBe('test-model-2');
      expect(run?.status).toBe('completed');
    });

    it('should get a specific run', async () => {
      const runId = await store.startRun('test-model-3', './path', '{"lr": 0.001}');
      await store.endRun('completed');

      const run = await store.getRun(runId);
      expect(run).toBeTruthy();
      expect(run?.id).toBe(runId);
      expect(run?.modelName).toBe('test-model-3');
      expect(run?.modelPath).toBe('./path');
      expect(run?.config).toBe('{"lr": 0.001}');
    });
  });

  describe('step recording', () => {
    it('should record a training step with generations', async () => {
      const runId = await store.startRun('test-model', undefined, '{}');

      const stepRecord = {
        runId,
        step: 1,
        epoch: undefined,
        loss: 0.5,
        meanReward: 0.7,
        stdReward: 0.1,
        meanAdvantage: 0.3,
        totalTokens: 100,
        generationTimeMs: 500.0,
        trainingTimeMs: 200.0,
        gradientsApplied: true,
      };

      const generations = [
        {
          batchIndex: 0,
          groupIndex: 0,
          prompt: 'Test prompt 1',
          expectedAnswer: '42',
          completionText: 'The answer is 42',
          completionRaw: 'The answer is 42',
          thinking: undefined,
          numTokens: 10,
          finishReason: 'eos',
          reward: 1.0,
        },
        {
          batchIndex: 0,
          groupIndex: 1,
          prompt: 'Test prompt 1',
          expectedAnswer: '42',
          completionText: 'I think it is 41',
          completionRaw: 'I think it is 41',
          thinking: undefined,
          numTokens: 8,
          finishReason: 'eos',
          reward: 0.0,
        },
      ];

      const stepId = await store.recordStep(stepRecord, generations, []);
      expect(stepId).toBeGreaterThan(0);

      await store.endRun('completed');
    });
  });

  describe('querying', () => {
    it('should query step summaries', async () => {
      const runId = await store.startRun('query-test', undefined, '{}');

      // Record a step
      const stepRecord = {
        runId,
        step: 1,
        epoch: undefined,
        loss: 0.5,
        meanReward: 0.8,
        stdReward: 0.1,
        meanAdvantage: 0.2,
        totalTokens: 50,
        generationTimeMs: 100.0,
        trainingTimeMs: 50.0,
        gradientsApplied: true,
      };

      await store.recordStep(
        stepRecord,
        [
          {
            batchIndex: 0,
            groupIndex: 0,
            prompt: 'Q',
            expectedAnswer: 'A',
            completionText: 'A',
            completionRaw: 'A',
            thinking: undefined,
            numTokens: 5,
            finishReason: 'eos',
            reward: 0.8,
          },
        ],
        [],
      );

      await store.endRun('completed');

      // Query summaries
      const summaries = await store.getStepSummaries(runId, undefined, undefined);
      expect(summaries.length).toBe(1);
      expect(summaries[0].step).toBe(1);
      expect(summaries[0].loss).toBeCloseTo(0.5);
      expect(summaries[0].meanReward).toBeCloseTo(0.8);
    });

    it('should get generations for a step', async () => {
      const runId = await store.startRun('gen-test', undefined, '{}');

      const stepRecord = {
        runId,
        step: 1,
        epoch: undefined,
        loss: 0.5,
        meanReward: 0.5,
        stdReward: 0.1,
        meanAdvantage: 0.0,
        totalTokens: 20,
        generationTimeMs: 100.0,
        trainingTimeMs: 50.0,
        gradientsApplied: true,
      };

      await store.recordStep(
        stepRecord,
        [
          {
            batchIndex: 0,
            groupIndex: 0,
            prompt: 'Hello',
            expectedAnswer: 'World',
            completionText: 'World',
            completionRaw: 'World',
            thinking: 'Let me think...',
            numTokens: 10,
            finishReason: 'eos',
            reward: 1.0,
          },
        ],
        [],
      );

      await store.endRun('completed');

      const gens = await store.getGenerations(runId, 1);
      expect(gens.length).toBe(1);
      expect(gens[0].generation.prompt).toBe('Hello');
      expect(gens[0].generation.thinking).toBe('Let me think...');
      expect(gens[0].generation.reward).toBe(1.0);
    });

    it('should get reward statistics', async () => {
      const runId = await store.startRun('stats-test', undefined, '{}');

      const stepRecord = {
        runId,
        step: 1,
        epoch: undefined,
        loss: 0.5,
        meanReward: 0.5,
        stdReward: 0.3,
        meanAdvantage: 0.0,
        totalTokens: 30,
        generationTimeMs: 100.0,
        trainingTimeMs: 50.0,
        gradientsApplied: true,
      };

      await store.recordStep(
        stepRecord,
        [
          {
            batchIndex: 0,
            groupIndex: 0,
            prompt: 'P1',
            expectedAnswer: undefined,
            completionText: 'C1',
            completionRaw: 'C1',
            thinking: undefined,
            numTokens: 10,
            finishReason: 'eos',
            reward: 0.0,
          },
          {
            batchIndex: 0,
            groupIndex: 1,
            prompt: 'P1',
            expectedAnswer: undefined,
            completionText: 'C2',
            completionRaw: 'C2',
            thinking: undefined,
            numTokens: 10,
            finishReason: 'eos',
            reward: 0.5,
          },
          {
            batchIndex: 0,
            groupIndex: 2,
            prompt: 'P1',
            expectedAnswer: undefined,
            completionText: 'C3',
            completionRaw: 'C3',
            thinking: undefined,
            numTokens: 10,
            finishReason: 'eos',
            reward: 1.0,
          },
        ],
        [],
      );

      await store.endRun('completed');

      const stats = await store.getRewardStats(runId, undefined);
      expect(stats.count).toBe(3);
      expect(stats.min).toBe(0.0);
      expect(stats.max).toBe(1.0);
      expect(stats.mean).toBeCloseTo(0.5);
    });
  });

  describe('SQL parameter binding edge cases', () => {
    it('getStepSummaries with only endStep should work', async () => {
      const runId = await store.startRun('binding-test', undefined, '{}');

      // Record a step
      const stepRecord = {
        runId,
        step: 5,
        epoch: undefined,
        loss: 0.5,
        meanReward: 0.7,
        stdReward: 0.1,
        meanAdvantage: 0.2,
        totalTokens: 50,
        generationTimeMs: 100.0,
        trainingTimeMs: 50.0,
        gradientsApplied: true,
      };

      await store.recordStep(
        stepRecord,
        [
          {
            batchIndex: 0,
            groupIndex: 0,
            prompt: 'Q',
            expectedAnswer: 'A',
            completionText: 'A',
            completionRaw: 'A',
            thinking: undefined,
            numTokens: 5,
            finishReason: 'eos',
            reward: 0.8,
          },
        ],
        [],
      );

      await store.endRun('completed');

      // Query with only endStep (start_step is undefined)
      // This tests the SQL binding fix for dynamic placeholder numbering
      const summaries = await store.getStepSummaries(runId, undefined, 10);
      expect(summaries.length).toBe(1);
      expect(summaries[0].step).toBe(5);
    });

    it('getGenerationsWithToolCalls with only status should work', async () => {
      const runId = await store.startRun('status-filter-test', undefined, '{}');

      // Record a step with tool calls
      const stepRecord = {
        runId,
        step: 1,
        epoch: undefined,
        loss: 0.5,
        meanReward: 0.7,
        stdReward: 0.1,
        meanAdvantage: 0.2,
        totalTokens: 50,
        generationTimeMs: 100.0,
        trainingTimeMs: 50.0,
        gradientsApplied: true,
      };

      await store.recordStep(
        stepRecord,
        [
          {
            batchIndex: 0,
            groupIndex: 0,
            prompt: 'Use a tool',
            expectedAnswer: undefined,
            completionText: 'I used a tool',
            completionRaw: '<tool_call>{"name": "test"}</tool_call>',
            thinking: undefined,
            numTokens: 5,
            finishReason: 'eos',
            reward: 1.0,
          },
        ],
        [
          [
            {
              callIndex: 0,
              status: 'ok',
              toolName: 'test_tool',
              arguments: '{}',
              rawContent: '<tool_call>{"name": "test"}</tool_call>',
              errorMessage: undefined,
            },
          ],
        ],
      );

      await store.endRun('completed');

      // Query with only status (tool_name is undefined)
      // This tests the SQL binding fix for dynamic placeholder numbering
      const results = await store.getGenerationsWithToolCalls(runId, undefined, 'ok', 10);
      expect(results.length).toBe(1);
    });
  });

  describe('input validation', () => {
    it('recordStepFromOutputs should reject group_size=0', async () => {
      await store.startRun('validation-test', undefined, '{}');

      const metrics = {
        step: 1,
        loss: 0.5,
        meanReward: 0.7,
        stdReward: 0.1,
        meanAdvantage: 0.2,
        totalTokens: 50,
        generationTimeMs: 100.0,
        trainingTimeMs: 50.0,
        gradientsApplied: true,
      };

      const outputsJson = JSON.stringify([
        {
          prompt: 'test',
          completion: {
            text: 'response',
            raw_text: 'response',
            tool_calls: [],
            thinking: null,
            num_tokens: 5,
            finish_reason: 'eos',
          },
          expected_answer: null,
        },
      ]);

      // This should throw because group_size=0 would cause division by zero
      await expect(store.recordStepFromOutputs(1, metrics, outputsJson, [0.5], 0)).rejects.toThrow(
        /group_size must be positive/,
      );

      await store.endRun('failed');
    });
  });

  describe('search', () => {
    it('should search generations by text', async () => {
      const runId = await store.startRun('search-test', undefined, '{}');

      const stepRecord = {
        runId,
        step: 1,
        epoch: undefined,
        loss: 0.5,
        meanReward: 0.5,
        stdReward: 0.1,
        meanAdvantage: 0.0,
        totalTokens: 20,
        generationTimeMs: 100.0,
        trainingTimeMs: 50.0,
        gradientsApplied: true,
      };

      await store.recordStep(
        stepRecord,
        [
          {
            batchIndex: 0,
            groupIndex: 0,
            prompt: 'apple banana',
            expectedAnswer: undefined,
            completionText: 'fruit salad',
            completionRaw: 'fruit salad',
            thinking: undefined,
            numTokens: 5,
            finishReason: 'eos',
            reward: 0.5,
          },
          {
            batchIndex: 0,
            groupIndex: 1,
            prompt: 'orange grape',
            expectedAnswer: undefined,
            completionText: 'citrus blend',
            completionRaw: 'citrus blend',
            thinking: undefined,
            numTokens: 5,
            finishReason: 'eos',
            reward: 0.5,
          },
        ],
        [],
      );

      await store.endRun('completed');

      // Search for "banana" in prompts
      const results = await store.searchGenerations(runId, 'banana', 'prompt', 10);
      expect(results.length).toBe(1);
      expect(results[0].generation.prompt).toContain('banana');

      // Search for "citrus" in completions
      const citrusResults = await store.searchGenerations(runId, 'citrus', 'completion', 10);
      expect(citrusResults.length).toBe(1);
      expect(citrusResults[0].generation.completionText).toContain('citrus');
    });
  });
});
