/**
 * Unified Training Logger
 *
 * A high-level logger abstraction for console and JSONL training logs.
 *
 * @example
 * ```typescript
 * const logger = createTrainingLogger();
 *
 * logger.info('Loading model...');           // Console log
 * logger.status('loading', 'Loading...');    // Status update
 * logger.step(metrics);                      // Training step
 * logger.checkpoint(path, step);             // Checkpoint saved
 * ```
 */

import { appendFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

// ============================================================================
// Types
// ============================================================================

/**
 * Training metrics from a training step
 *
 * Supports both SFT and GRPO training:
 * - SFT: loss, perplexity, tokenAccuracy (no reward/advantage)
 * - GRPO: loss, meanReward, stdReward, stdAdvantage
 */
export interface TrainingMetrics {
  step: number;
  loss: number;
  totalTokens: number;
  // GRPO-specific (optional for SFT)
  meanReward?: number;
  stdReward?: number;
  meanAdvantage?: number;
  /** Std of advantages - indicates reward variance within groups */
  stdAdvantage?: number;
  // SFT-specific (optional for GRPO)
  perplexity?: number;
  tokenAccuracy?: number;
  // Timing (optional)
  generationTimeMs?: number;
  trainingTimeMs?: number;
  // Memory (optional)
  peakMemoryMb?: number;
  activeMemoryMb?: number;
}

/**
 * Training configuration fields for logging
 */
export interface TrainingConfigFields {
  numEpochs: number;
  batchSize: number;
  groupSize: number;
  learningRate: number;
  /** Training type: "sft" or "grpo" (default: "grpo") */
  trainingType?: 'sft' | 'grpo';
  [key: string]: unknown;
}

/**
 * Configuration for the training logger
 */
export interface TrainingLoggerConfig {
  /** Enable console logging */
  logConsole: boolean;
  /** Enable JSONL file logging */
  logJsonl: boolean;
  /** Output directory for JSONL files */
  outputDir?: string;
  /** Run name for log files */
  runName?: string;
  /** Logging frequency (every N steps) */
  logInterval: number;
}

/**
 * JSONL log event types
 */
export type LogEvent =
  | { event: 'training_start'; timestamp: string; config: TrainingLoggerConfig }
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

// ============================================================================
// Metrics Aggregator
// ============================================================================

/**
 * Aggregator for computing running statistics over an epoch
 */
class MetricsAggregator {
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

  reset(): void {
    this.values = [];
  }
}

// ============================================================================
// Training Logger
// ============================================================================

/**
 * Training logger for console and JSONL output
 */
export class TrainingLogger {
  private config: TrainingLoggerConfig;
  private jsonlPath?: string;
  private startTime: number;
  private lastLogTime: number;

  // Epoch-level aggregators
  private epochLoss = new MetricsAggregator();
  private epochReward = new MetricsAggregator();
  private epochAdvantage = new MetricsAggregator();
  private epochTokens = new MetricsAggregator();

  constructor(config: Partial<TrainingLoggerConfig> = {}) {
    this.config = {
      logConsole: config.logConsole ?? true,
      logJsonl: config.logJsonl ?? false,
      logInterval: config.logInterval ?? 1,
      outputDir: config.outputDir,
      runName: config.runName,
    };
    this.startTime = Date.now();
    this.lastLogTime = this.startTime;

    this.setupJsonl();
  }

  private setupJsonl(): void {
    if (!this.config.logJsonl) return;
    if (!this.config.outputDir) return;

    try {
      mkdirSync(this.config.outputDir, { recursive: true });
      const runName = this.config.runName || 'grpo-training';
      this.jsonlPath = join(this.config.outputDir, `${runName}.jsonl`);

      this.writeJsonl({
        event: 'training_start',
        timestamp: new Date().toISOString(),
        config: this.config,
      });
    } catch {
      // Silently ignore - logging shouldn't crash training
    }
  }

  // ==========================================================================
  // Core Logging Methods
  // ==========================================================================

  /** Log info message */
  info(message: string): void {
    if (this.config.logConsole) {
      console.log(message);
    }
  }

  /** Log warning message */
  warn(message: string): void {
    console.warn(message);
  }

  /** Log error message */
  error(message: string): void {
    console.error(message);
  }

  /** Log debug message (only in verbose mode) */
  debug(message: string): void {
    if (this.config.logConsole && process.env.DEBUG) {
      console.log(`[DEBUG] ${message}`);
    }
  }

  /** Log a status update */
  status(_phase: string, message: string): void {
    if (this.config.logConsole) {
      console.log(message);
    }
  }

  /** Print decorative banner */
  banner(...lines: string[]): void {
    if (!this.config.logConsole) return;
    for (const line of lines) {
      console.log(line);
    }
  }

  // ==========================================================================
  // Training Event Methods
  // ==========================================================================

  /** Log training initialization */
  init(_model: string, config: TrainingConfigFields, numExamples?: number): void {
    const trainingType = config.trainingType ?? 'grpo';

    if (this.config.logConsole) {
      const trainingLabel = trainingType === 'sft' ? 'SFT' : 'GRPO';
      console.log(`\n Starting ${trainingLabel} training`);
      if (numExamples) console.log(`   Examples: ${numExamples}`);
      console.log(`   Epochs: ${config.numEpochs}`);
      console.log(`   Batch size: ${config.batchSize}`);
      if (trainingType !== 'sft') {
        console.log(`   Group size: ${config.groupSize}`);
      }
      console.log(`   Learning rate: ${config.learningRate}`);
    }

    if (this.jsonlPath && numExamples !== undefined) {
      this.writeJsonl({
        event: 'training_config',
        num_examples: numExamples,
        config: { ...config, trainingType },
        timestamp: new Date().toISOString(),
      });
    }
  }

  /** Log epoch start */
  epochStart(epoch: number, totalEpochs: number, numBatches: number): void {
    if (this.config.logConsole) {
      console.log(`\n=== Epoch ${epoch + 1}/${totalEpochs} (${numBatches} batches) ===`);
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

  /** Log training step */
  step(metrics: TrainingMetrics, batchIdx?: number, numBatches?: number): void {
    // Aggregate for epoch (use available metrics)
    this.epochLoss.add(metrics.loss);
    if (metrics.meanReward !== undefined) {
      this.epochReward.add(metrics.meanReward);
    }
    if (metrics.meanAdvantage !== undefined) {
      this.epochAdvantage.add(metrics.meanAdvantage);
    }
    this.epochTokens.add(metrics.totalTokens);

    if (this.config.logConsole && metrics.step % this.config.logInterval === 0) {
      const now = Date.now();
      const stepTime = (now - this.lastLogTime) / this.config.logInterval;
      this.lastLogTime = now;

      const batchInfo =
        batchIdx !== undefined && numBatches !== undefined ? ` | Batch ${batchIdx + 1}/${numBatches}` : '';

      // Build log message based on available metrics (SFT vs GRPO)
      let logMsg = `Step ${metrics.step}${batchInfo} | Loss: ${metrics.loss.toFixed(4)}`;

      if (metrics.perplexity !== undefined) {
        // SFT format
        logMsg += ` | Perplexity: ${metrics.perplexity.toFixed(2)}`;
      }
      if (metrics.tokenAccuracy !== undefined) {
        logMsg += ` | Acc: ${(metrics.tokenAccuracy * 100).toFixed(1)}%`;
      }
      if (metrics.meanReward !== undefined) {
        // GRPO format
        logMsg += ` | Reward: ${metrics.meanReward.toFixed(4)}`;
      }
      if (metrics.meanAdvantage !== undefined) {
        logMsg += ` | Adv: ${metrics.meanAdvantage.toFixed(4)}`;
      }

      logMsg += ` | Tokens: ${metrics.totalTokens} | Time: ${stepTime.toFixed(0)}ms/step`;
      console.log(logMsg);
    }

    if (this.jsonlPath && metrics.step % this.config.logInterval === 0) {
      this.writeJsonl({
        event: 'step',
        step: metrics.step,
        loss: metrics.loss,
        mean_reward: metrics.meanReward ?? 0,
        std_reward: metrics.stdReward ?? 0,
        mean_advantage: metrics.meanAdvantage ?? 0,
        total_tokens: metrics.totalTokens,
        timestamp: new Date().toISOString(),
      });
    }
  }

  /** Log epoch end/summary */
  epochEnd(epoch: number, totalEpochs: number, _epochTimeSecs?: number): void {
    const avgLoss = this.epochLoss.mean();
    const avgReward = this.epochReward.mean();
    const avgAdvantage = this.epochAdvantage.mean();
    const totalTokens = this.epochTokens.sum();

    if (this.config.logConsole) {
      console.log(
        `\nEpoch ${epoch + 1}/${totalEpochs} Summary | ` +
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

    // Reset aggregators
    this.epochLoss.reset();
    this.epochReward.reset();
    this.epochAdvantage.reset();
    this.epochTokens.reset();
  }

  /** Log checkpoint saved */
  checkpoint(path: string, step: number): void {
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

  /** Log training completion */
  complete(totalSteps: number): void {
    const totalTime = Date.now() - this.startTime;
    const totalMinutes = totalTime / 60000;
    if (this.config.logConsole) {
      console.log(`\n✓ Training complete! Final step: ${totalSteps} | Total time: ${totalMinutes.toFixed(2)} minutes`);
    }

    if (this.jsonlPath) {
      this.writeJsonl({
        event: 'training_complete',
        final_step: totalSteps,
        total_time_ms: totalTime,
        timestamp: new Date().toISOString(),
      });
    }
  }

  // ==========================================================================
  // Accessors
  // ==========================================================================

  /** Get the log interval */
  get logInterval(): number {
    return this.config.logInterval;
  }

  // ==========================================================================
  // Internal Methods
  // ==========================================================================

  private writeJsonl(data: LogEvent): void {
    if (!this.jsonlPath) return;

    try {
      const line = JSON.stringify(data) + '\n';
      appendFileSync(this.jsonlPath, line, 'utf8');
    } catch {
      // Silently ignore - logging shouldn't crash training
    }
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a training logger instance
 *
 * @example
 * ```typescript
 * const logger = createTrainingLogger();
 *
 * // Explicit configuration
 * const logger = createTrainingLogger({
 *   logConsole: true,
 *   logJsonl: true,
 *   outputDir: './outputs',
 *   logInterval: 10,
 * });
 * ```
 */
export function createTrainingLogger(config?: Partial<TrainingLoggerConfig>): TrainingLogger {
  return new TrainingLogger(config);
}
