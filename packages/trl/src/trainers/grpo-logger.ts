/**
 * GRPO Training Logger
 *
 * Provides structured logging and metrics tracking for GRPO training
 */

import { appendFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import type { TrainingMetrics } from './grpo-trainer';

/**
 * Required fields for training config logging
 */
export interface TrainingConfigFields {
  numEpochs: number;
  batchSize: number;
  groupSize: number;
  learningRate: number;
}

/**
 * Union type for all log events written to JSONL
 */
export type LogEvent =
  | { event: 'training_start'; timestamp: string; config: LoggerConfig }
  | { event: 'training_config'; num_examples: number; config: Record<string, unknown>; timestamp: string }
  | {
      event: 'step';
      step: number;
      loss: number;
      mean_reward: number;
      std_reward: number;
      mean_advantage: number;
      total_tokens: number;
      timestamp: string;
    }
  | {
      event: 'epoch';
      epoch: number;
      avg_loss: number;
      avg_reward: number;
      avg_advantage: number;
      total_tokens: number;
      timestamp: string;
    }
  | { event: 'epoch_start'; epoch: number; num_batches: number; timestamp: string }
  | { event: 'training_complete'; final_step: number; total_time_ms: number; timestamp: string }
  | { event: 'checkpoint'; step: number; path: string; timestamp: string };

/**
 * Configuration for the training logger
 */
export interface LoggerConfig {
  /** Output directory for logs */
  outputDir: string;

  /** Whether to write JSONL log file */
  logJsonl: boolean;

  /** Whether to log to console */
  logConsole: boolean;

  /** Logging frequency (every N steps) */
  logInterval: number;

  /** Run name for this training session */
  runName?: string;

  /** Enable TUI mode - outputs structured JSONL to stdout for TUI consumption */
  tuiMode?: boolean;
}

/**
 * TUI message types for structured stdout output
 */
export type TuiMessage =
  | { type: 'init'; model: string; config: Record<string, unknown> }
  | { type: 'epoch_start'; epoch: number; totalEpochs: number; numBatches: number }
  | {
      type: 'step';
      step: number;
      loss: number;
      meanReward: number;
      stdReward: number;
      meanAdvantage: number;
      totalTokens: number;
      generationTimeMs: number;
      trainingTimeMs: number;
    }
  | { type: 'generation'; index: number; prompt: string; completion: string; reward: number; tokens: number }
  | { type: 'checkpoint'; path: string; step: number }
  | { type: 'epoch_end'; epoch: number; avgLoss: number; avgReward: number; epochTimeSecs: number }
  | { type: 'complete'; totalSteps: number; totalTimeSecs: number }
  | { type: 'log'; level: 'info' | 'warn' | 'error' | 'debug'; message: string }
  | { type: 'paused'; step: number }
  | { type: 'resumed'; step: number }
  | { type: 'status'; phase: string; message: string };

/**
 * Metrics aggregator for computing running statistics
 */
export class MetricsAggregator {
  private values: number[] = [];

  add(value: number): void {
    this.values.push(value);
  }

  mean(): number {
    if (this.values.length === 0) return 0;
    return this.values.reduce((a, b) => a + b, 0) / this.values.length;
  }

  sum(): number {
    return this.values.reduce((a, b) => a + b, 0);
  }

  count(): number {
    return this.values.length;
  }

  reset(): void {
    this.values = [];
  }
}

/**
 * Training logger for GRPO
 */
export class GRPOLogger {
  private config: LoggerConfig;
  private jsonlPath?: string;
  private startTime: number;
  private lastLogTime: number;

  // Aggregators for epoch-level metrics
  private epochLoss = new MetricsAggregator();
  private epochReward = new MetricsAggregator();
  private epochAdvantage = new MetricsAggregator();
  private epochTokens = new MetricsAggregator();

  constructor(config: LoggerConfig) {
    this.config = config;
    this.startTime = Date.now();
    this.lastLogTime = this.startTime;

    // Create output directory (skip if TUI mode and no JSONL logging)
    if (this.config.logJsonl && !this.config.tuiMode) {
      try {
        mkdirSync(this.config.outputDir, { recursive: true });
        const runName = this.config.runName || 'grpo-training';
        this.jsonlPath = join(this.config.outputDir, `${runName}.jsonl`);

        // Write header
        this.writeJsonl({
          event: 'training_start',
          timestamp: new Date().toISOString(),
          config: this.config,
        });
      } catch (error) {
        console.warn('Failed to create log file:', error);
      }
    }
  }

  /**
   * Write TUI message to stdout (for TUI mode)
   */
  writeTui(msg: TuiMessage): void {
    if (!this.config.tuiMode) return;
    process.stdout.write(JSON.stringify(msg) + '\n');
  }

  /**
   * Check if TUI mode is enabled
   */
  get isTuiMode(): boolean {
    return this.config.tuiMode === true;
  }

  /**
   * Log training step metrics
   */
  logStep(metrics: TrainingMetrics, batchIdx?: number, numBatches?: number): void {
    // Add to epoch aggregators
    this.epochLoss.add(metrics.loss);
    this.epochReward.add(metrics.meanReward);
    this.epochAdvantage.add(metrics.meanAdvantage);
    this.epochTokens.add(metrics.totalTokens);

    // TUI mode: send structured message
    if (this.config.tuiMode) {
      this.writeTui({
        type: 'step',
        step: metrics.step,
        loss: metrics.loss,
        meanReward: metrics.meanReward,
        stdReward: metrics.stdReward,
        meanAdvantage: metrics.meanAdvantage,
        totalTokens: metrics.totalTokens,
        generationTimeMs: metrics.generationTimeMs ?? 0,
        trainingTimeMs: metrics.trainingTimeMs ?? 0,
      });
      return; // Skip console output in TUI mode
    }

    // Log to console if needed
    if (this.config.logConsole && metrics.step % this.config.logInterval === 0) {
      const now = Date.now();
      const stepTime = (now - this.lastLogTime) / this.config.logInterval;
      this.lastLogTime = now;

      const batchInfo =
        batchIdx !== undefined && numBatches !== undefined ? ` | Batch ${batchIdx + 1}/${numBatches}` : '';

      console.log(
        `Step ${metrics.step}${batchInfo} | ` +
          `Loss: ${metrics.loss.toFixed(4)} | ` +
          `Reward: ${metrics.meanReward.toFixed(4)} | ` +
          `Adv: ${metrics.meanAdvantage.toFixed(4)} | ` +
          `Tokens: ${metrics.totalTokens} | ` +
          `Time: ${stepTime.toFixed(0)}ms/step`,
      );
    }

    // Log to JSONL if needed
    if (this.jsonlPath && metrics.step % this.config.logInterval === 0) {
      this.writeJsonl({
        event: 'step',
        step: metrics.step,
        loss: metrics.loss,
        mean_reward: metrics.meanReward,
        std_reward: metrics.stdReward,
        mean_advantage: metrics.meanAdvantage,
        total_tokens: metrics.totalTokens,
        timestamp: new Date().toISOString(),
      });
    }
  }

  /**
   * Log epoch summary
   */
  logEpoch(epoch: number, numEpochs: number, epochTimeSecs?: number): void {
    const avgLoss = this.epochLoss.mean();
    const avgReward = this.epochReward.mean();
    const avgAdvantage = this.epochAdvantage.mean();
    const totalTokens = this.epochTokens.sum();

    // TUI mode: send structured message
    if (this.config.tuiMode) {
      this.writeTui({
        type: 'epoch_end',
        epoch: epoch + 1,
        avgLoss,
        avgReward,
        epochTimeSecs: epochTimeSecs ?? 0,
      });
    } else if (this.config.logConsole) {
      console.log(
        `\nEpoch ${epoch + 1}/${numEpochs} Summary | ` +
          `Avg Loss: ${avgLoss.toFixed(4)} | ` +
          `Avg Reward: ${avgReward.toFixed(4)} | ` +
          `Avg Advantage: ${avgAdvantage.toFixed(4)} | ` +
          `Total Tokens: ${totalTokens.toFixed(0)}`,
      );
    }

    if (this.jsonlPath) {
      this.writeJsonl({
        event: 'epoch',
        epoch: epoch + 1,
        avg_loss: avgLoss,
        avg_reward: avgReward,
        avg_advantage: avgAdvantage,
        total_tokens: totalTokens,
        timestamp: new Date().toISOString(),
      });
    }

    // Reset aggregators for next epoch
    this.epochLoss.reset();
    this.epochReward.reset();
    this.epochAdvantage.reset();
    this.epochTokens.reset();
  }

  /**
   * Log training completion
   */
  logComplete(finalStep: number): void {
    const totalTime = Date.now() - this.startTime;
    const totalMinutes = totalTime / 60000;
    const totalTimeSecs = totalTime / 1000;

    // TUI mode: send structured message
    if (this.config.tuiMode) {
      this.writeTui({
        type: 'complete',
        totalSteps: finalStep,
        totalTimeSecs,
      });
      return;
    }

    if (this.config.logConsole) {
      console.log(
        `\n✓ Training complete! ` + `Final step: ${finalStep} | ` + `Total time: ${totalMinutes.toFixed(2)} minutes`,
      );
    }

    if (this.jsonlPath) {
      this.writeJsonl({
        event: 'training_complete',
        final_step: finalStep,
        total_time_ms: totalTime,
        timestamp: new Date().toISOString(),
      });
    }
  }

  /**
   * Log checkpoint save
   */
  logCheckpoint(step: number, path: string): void {
    // TUI mode: send structured message
    if (this.config.tuiMode) {
      this.writeTui({
        type: 'checkpoint',
        step,
        path,
      });
      return;
    }

    if (this.config.logConsole) {
      console.log(`💾 Checkpoint saved: ${path}`);
    }

    if (this.jsonlPath) {
      this.writeJsonl({
        event: 'checkpoint',
        step,
        path,
        timestamp: new Date().toISOString(),
      });
    }
  }

  /**
   * Log epoch start
   */
  logEpochStart(epoch: number, numEpochs: number, numBatches: number): void {
    // TUI mode: send structured message
    if (this.config.tuiMode) {
      this.writeTui({
        type: 'epoch_start',
        epoch: epoch + 1,
        totalEpochs: numEpochs,
        numBatches,
      });
      return;
    }

    if (this.config.logConsole) {
      console.log(`\n=== Epoch ${epoch + 1}/${numEpochs} (${numBatches} batches) ===`);
    }

    if (this.jsonlPath) {
      this.writeJsonl({
        event: 'epoch_start',
        epoch: epoch + 1,
        num_batches: numBatches,
        timestamp: new Date().toISOString(),
      });
    }
  }

  /**
   * Log training start
   */
  logTrainingStart<T extends TrainingConfigFields>(numExamples: number, config: T, modelName?: string): void {
    // TUI mode: send init message
    if (this.config.tuiMode) {
      this.writeTui({
        type: 'init',
        model: modelName ?? 'Unknown',
        config: config as Record<string, unknown>,
      });
      return;
    }

    if (this.config.logConsole) {
      console.log(`\n🚀 Starting GRPO training`);
      console.log(`   Examples: ${numExamples}`);
      console.log(`   Epochs: ${config.numEpochs}`);
      console.log(`   Batch size: ${config.batchSize}`);
      console.log(`   Group size: ${config.groupSize}`);
      console.log(`   Learning rate: ${config.learningRate}`);
    }

    if (this.jsonlPath) {
      this.writeJsonl({
        event: 'training_config',
        num_examples: numExamples,
        config: config as Record<string, unknown>,
        timestamp: new Date().toISOString(),
      });
    }
  }

  /**
   * Log a generation sample (for TUI mode)
   */
  logGeneration(index: number, prompt: string, completion: string, reward: number, tokens: number): void {
    if (!this.config.tuiMode) return;
    this.writeTui({
      type: 'generation',
      index,
      prompt: prompt.slice(0, 200), // Truncate for TUI
      completion: completion.slice(0, 500),
      reward,
      tokens,
    });
  }

  /**
   * Log pause event (for TUI mode)
   */
  logPaused(step: number): void {
    if (!this.config.tuiMode) return;
    this.writeTui({ type: 'paused', step });
  }

  /**
   * Log resume event (for TUI mode)
   */
  logResumed(step: number): void {
    if (!this.config.tuiMode) return;
    this.writeTui({ type: 'resumed', step });
  }

  /**
   * Write a line to JSONL log file
   */
  private writeJsonl(data: LogEvent): void {
    if (!this.jsonlPath) return;

    try {
      const line = JSON.stringify(data) + '\n';
      appendFileSync(this.jsonlPath, line, 'utf8');
    } catch (error) {
      console.warn('Failed to write to log file:', error);
    }
  }
}

/**
 * Create a logger instance
 */
export function createLogger(config: Partial<LoggerConfig> = {}): GRPOLogger {
  const defaultConfig: LoggerConfig = {
    outputDir: './outputs/logs',
    logJsonl: true,
    logConsole: true,
    logInterval: 10,
    runName: `grpo-${Date.now()}`,
  };

  return new GRPOLogger({ ...defaultConfig, ...config });
}
