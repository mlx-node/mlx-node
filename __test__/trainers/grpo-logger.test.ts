/**
 * Tests for GRPO Logger
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { GRPOLogger, createLogger, MetricsAggregator, type LoggerConfig, type TrainingMetrics } from '@mlx-node/trl';
import { readFileSync, rmSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const TEST_OUTPUT_DIR = './test-logs';

describe('MetricsAggregator', () => {
  it('should compute mean correctly', () => {
    const agg = new MetricsAggregator();
    agg.add(1);
    agg.add(2);
    agg.add(3);

    expect(agg.mean()).toBe(2);
  });

  it('should compute sum correctly', () => {
    const agg = new MetricsAggregator();
    agg.add(10);
    agg.add(20);
    agg.add(30);

    expect(agg.sum()).toBe(60);
  });

  it('should track count correctly', () => {
    const agg = new MetricsAggregator();
    agg.add(1);
    agg.add(2);

    expect(agg.count()).toBe(2);
  });

  it('should reset correctly', () => {
    const agg = new MetricsAggregator();
    agg.add(1);
    agg.add(2);

    agg.reset();

    expect(agg.count()).toBe(0);
    expect(agg.mean()).toBe(0);
    expect(agg.sum()).toBe(0);
  });

  it('should handle empty aggregator', () => {
    const agg = new MetricsAggregator();

    expect(agg.mean()).toBe(0);
    expect(agg.sum()).toBe(0);
    expect(agg.count()).toBe(0);
  });
});

describe('GRPOLogger', () => {
  beforeEach(() => {
    // Clean up test logs before each test
    if (existsSync(TEST_OUTPUT_DIR)) {
      rmSync(TEST_OUTPUT_DIR, { recursive: true, force: true });
    }
  });

  afterEach(() => {
    // Clean up test logs after each test
    if (existsSync(TEST_OUTPUT_DIR)) {
      rmSync(TEST_OUTPUT_DIR, { recursive: true, force: true });
    }
  });

  describe('Logger Creation', () => {
    it('should create logger with default config', () => {
      const logger = createLogger();
      expect(logger).toBeInstanceOf(GRPOLogger);
    });

    it('should create logger with custom config', () => {
      const config: Partial<LoggerConfig> = {
        outputDir: TEST_OUTPUT_DIR,
        logJsonl: true,
        logConsole: false,
        logInterval: 5,
        runName: 'test-run',
      };

      const logger = createLogger(config);
      expect(logger).toBeInstanceOf(GRPOLogger);
    });

    it('should create output directory', () => {
      createLogger({
        outputDir: TEST_OUTPUT_DIR,
        logJsonl: true,
      });

      expect(existsSync(TEST_OUTPUT_DIR)).toBe(true);
    });
  });

  describe('JSONL Logging', () => {
    it('should write JSONL log file', () => {
      const logger = createLogger({
        outputDir: TEST_OUTPUT_DIR,
        logJsonl: true,
        logConsole: false,
        runName: 'test-run',
      });

      const metrics: TrainingMetrics = {
        loss: 0.5,
        meanReward: 1.0,
        stdReward: 0.1,
        meanAdvantage: 0.0,
        totalTokens: 100,
        step: 1,
      };

      logger.logStep(metrics);

      const logPath = join(TEST_OUTPUT_DIR, 'test-run.jsonl');
      expect(existsSync(logPath)).toBe(true);
    });

    it('should log training start', () => {
      const logger = createLogger({
        outputDir: TEST_OUTPUT_DIR,
        logJsonl: true,
        logConsole: false,
        runName: 'test-run',
      });

      logger.logTrainingStart(100, {
        numEpochs: 3,
        batchSize: 4,
        groupSize: 8,
        learningRate: 0.001,
      });

      const logPath = join(TEST_OUTPUT_DIR, 'test-run.jsonl');
      const content = readFileSync(logPath, 'utf8');
      const lines = content.trim().split('\n');

      expect(lines.length).toBeGreaterThanOrEqual(2); // Header + training_config
    });

    it('should log step metrics', () => {
      const logger = createLogger({
        outputDir: TEST_OUTPUT_DIR,
        logJsonl: true,
        logConsole: false,
        logInterval: 1,
        runName: 'test-run',
      });

      const metrics: TrainingMetrics = {
        loss: 0.5,
        meanReward: 1.0,
        stdReward: 0.1,
        meanAdvantage: 0.0,
        totalTokens: 100,
        step: 1,
      };

      logger.logStep(metrics);

      const logPath = join(TEST_OUTPUT_DIR, 'test-run.jsonl');
      const content = readFileSync(logPath, 'utf8');
      const lines = content.trim().split('\n');
      const lastLine = JSON.parse(lines[lines.length - 1]);

      expect(lastLine.event).toBe('step');
      expect(lastLine.loss).toBe(0.5);
      expect(lastLine.mean_reward).toBe(1.0);
    });

    it('should log epoch summary', () => {
      const logger = createLogger({
        outputDir: TEST_OUTPUT_DIR,
        logJsonl: true,
        logConsole: false,
        logInterval: 1,
        runName: 'test-run',
      });

      const metrics: TrainingMetrics = {
        loss: 0.5,
        meanReward: 1.0,
        stdReward: 0.1,
        meanAdvantage: 0.0,
        totalTokens: 100,
        step: 1,
      };

      logger.logStep(metrics);
      logger.logEpoch(0, 3);

      const logPath = join(TEST_OUTPUT_DIR, 'test-run.jsonl');
      const content = readFileSync(logPath, 'utf8');
      const lines = content.trim().split('\n');
      const lastLine = JSON.parse(lines[lines.length - 1]);

      expect(lastLine.event).toBe('epoch');
      expect(lastLine.epoch).toBe(1);
      expect(lastLine.avg_loss).toBe(0.5);
      expect(lastLine.avg_reward).toBe(1.0);
    });

    it('should log checkpoint save', () => {
      const logger = createLogger({
        outputDir: TEST_OUTPUT_DIR,
        logJsonl: true,
        logConsole: false,
        runName: 'test-run',
      });

      logger.logCheckpoint(100, '/path/to/checkpoint');

      const logPath = join(TEST_OUTPUT_DIR, 'test-run.jsonl');
      const content = readFileSync(logPath, 'utf8');
      const lines = content.trim().split('\n');
      const lastLine = JSON.parse(lines[lines.length - 1]);

      expect(lastLine.event).toBe('checkpoint');
      expect(lastLine.step).toBe(100);
      expect(lastLine.path).toBe('/path/to/checkpoint');
    });

    it('should log training completion', () => {
      const logger = createLogger({
        outputDir: TEST_OUTPUT_DIR,
        logJsonl: true,
        logConsole: false,
        runName: 'test-run',
      });

      logger.logComplete(100);

      const logPath = join(TEST_OUTPUT_DIR, 'test-run.jsonl');
      const content = readFileSync(logPath, 'utf8');
      const lines = content.trim().split('\n');
      const lastLine = JSON.parse(lines[lines.length - 1]);

      expect(lastLine.event).toBe('training_complete');
      expect(lastLine.final_step).toBe(100);
      expect(lastLine.total_time_ms).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Metrics Aggregation', () => {
    it('should aggregate metrics across steps', () => {
      const logger = createLogger({
        outputDir: TEST_OUTPUT_DIR,
        logJsonl: true,
        logConsole: false,
        logInterval: 1,
        runName: 'test-run',
      });

      // Log 3 steps
      for (let i = 1; i <= 3; i++) {
        const metrics: TrainingMetrics = {
          loss: i * 0.1,
          meanReward: i * 1.0,
          stdReward: 0.1,
          meanAdvantage: 0.0,
          totalTokens: 100,
          step: i,
        };
        logger.logStep(metrics);
      }

      logger.logEpoch(0, 1);

      const logPath = join(TEST_OUTPUT_DIR, 'test-run.jsonl');
      const content = readFileSync(logPath, 'utf8');
      const lines = content.trim().split('\n');
      const lastLine = JSON.parse(lines[lines.length - 1]);

      // Average loss should be (0.1 + 0.2 + 0.3) / 3 = 0.2
      expect(lastLine.avg_loss).toBeCloseTo(0.2, 5);
      // Average reward should be (1.0 + 2.0 + 3.0) / 3 = 2.0
      expect(lastLine.avg_reward).toBeCloseTo(2.0, 5);
    });

    it('should reset aggregators after epoch', () => {
      const logger = createLogger({
        outputDir: TEST_OUTPUT_DIR,
        logJsonl: true,
        logConsole: false,
        logInterval: 1,
        runName: 'test-run',
      });

      // Epoch 1
      const metrics1: TrainingMetrics = {
        loss: 1.0,
        meanReward: 10.0,
        stdReward: 0.1,
        meanAdvantage: 0.0,
        totalTokens: 100,
        step: 1,
      };
      logger.logStep(metrics1);
      logger.logEpoch(0, 2);

      // Epoch 2
      const metrics2: TrainingMetrics = {
        loss: 0.5,
        meanReward: 5.0,
        stdReward: 0.1,
        meanAdvantage: 0.0,
        totalTokens: 100,
        step: 2,
      };
      logger.logStep(metrics2);
      logger.logEpoch(1, 2);

      const logPath = join(TEST_OUTPUT_DIR, 'test-run.jsonl');
      const content = readFileSync(logPath, 'utf8');
      const lines = content.trim().split('\n');

      // Find the two epoch summary lines
      const epochLines = lines.map((line) => JSON.parse(line)).filter((obj) => obj.event === 'epoch');

      expect(epochLines[0].avg_loss).toBeCloseTo(1.0, 5);
      expect(epochLines[0].avg_reward).toBeCloseTo(10.0, 5);

      expect(epochLines[1].avg_loss).toBeCloseTo(0.5, 5);
      expect(epochLines[1].avg_reward).toBeCloseTo(5.0, 5);
    });
  });

  describe('Console Logging', () => {
    it('should respect logConsole flag', () => {
      // This test just ensures the logger doesn't crash with console logging
      const logger = createLogger({
        outputDir: TEST_OUTPUT_DIR,
        logJsonl: false,
        logConsole: true,
        logInterval: 1,
        runName: 'test-run',
      });

      const metrics: TrainingMetrics = {
        loss: 0.5,
        meanReward: 1.0,
        stdReward: 0.1,
        meanAdvantage: 0.0,
        totalTokens: 100,
        step: 1,
      };

      expect(() => logger.logStep(metrics)).not.toThrow();
    });

    it('should respect logInterval', () => {
      const logger = createLogger({
        outputDir: TEST_OUTPUT_DIR,
        logJsonl: true,
        logConsole: false,
        logInterval: 5, // Only log every 5 steps
        runName: 'test-run',
      });

      // Log steps 1-10
      for (let i = 1; i <= 10; i++) {
        const metrics: TrainingMetrics = {
          loss: 0.5,
          meanReward: 1.0,
          stdReward: 0.1,
          meanAdvantage: 0.0,
          totalTokens: 100,
          step: i,
        };
        logger.logStep(metrics);
      }

      const logPath = join(TEST_OUTPUT_DIR, 'test-run.jsonl');
      const content = readFileSync(logPath, 'utf8');
      const lines = content.trim().split('\n');
      const stepLines = lines.map((line) => JSON.parse(line)).filter((obj) => obj.event === 'step');

      // Should only have 2 step logs (step 5 and step 10)
      expect(stepLines.length).toBe(2);
      expect(stepLines[0].step).toBe(5);
      expect(stepLines[1].step).toBe(10);
    });
  });
});
