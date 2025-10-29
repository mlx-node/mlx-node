/**
 * Unified Training Logger
 *
 * A high-level logger abstraction that handles TUI/console mode automatically.
 * Eliminates verbose conditional checks throughout the codebase.
 *
 * @example
 * ```typescript
 * const logger = createTrainingLogger();
 *
 * logger.info('Loading model...');           // Console OR TUI log
 * logger.status('loading', 'Loading...');    // TUI header update
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
 */
export interface TrainingMetrics {
  step: number;
  loss: number;
  meanReward: number;
  stdReward: number;
  meanAdvantage: number;
  totalTokens: number;
  generationTimeMs?: number;
  trainingTimeMs?: number;
}

/**
 * Generation sample for TUI display
 */
export interface GenerationSample {
  index: number;
  prompt: string;
  completion: string;
  reward: number;
  tokens: number;
}

/**
 * Training configuration fields for logging
 */
export interface TrainingConfigFields {
  numEpochs: number;
  batchSize: number;
  groupSize: number;
  learningRate: number;
  [key: string]: unknown;
}

/**
 * Configuration for the training logger
 */
export interface TrainingLoggerConfig {
  /** Enable TUI mode (JSONL to stdout) */
  tuiMode: boolean;
  /** Enable console logging (non-TUI) */
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
 * Unified training logger that handles TUI/console mode automatically
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
    // TUI mode is ONLY enabled via environment variable (set by mlx-tui)
    // This ensures users don't accidentally miss output in CLI mode
    const tuiMode = process.env.MLX_TUI_MODE === '1';

    this.config = {
      tuiMode,
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
    if (!this.config.logJsonl || this.config.tuiMode) return;
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
    if (this.config.tuiMode) {
      this.writeTui({ type: 'log', level: 'info', message });
    } else if (this.config.logConsole) {
      console.log(message);
    }
  }

  /** Log warning message */
  warn(message: string): void {
    if (this.config.tuiMode) {
      this.writeTui({ type: 'log', level: 'warn', message });
    } else {
      console.warn(message);
    }
  }

  /** Log error message */
  error(message: string): void {
    if (this.config.tuiMode) {
      this.writeTui({ type: 'log', level: 'error', message });
    } else {
      console.error(message);
    }
  }

  /** Log debug message (only in verbose mode) */
  debug(message: string): void {
    if (this.config.tuiMode) {
      this.writeTui({ type: 'log', level: 'debug', message });
    } else if (this.config.logConsole && process.env.DEBUG) {
      console.log(`[DEBUG] ${message}`);
    }
  }

  /** Update status (updates TUI header, shows in console) */
  status(phase: string, message: string): void {
    if (this.config.tuiMode) {
      this.writeTui({ type: 'status', phase, message });
    } else if (this.config.logConsole) {
      console.log(message);
    }
  }

  /** Print decorative banner (console only, suppressed in TUI mode) */
  banner(...lines: string[]): void {
    if (this.config.tuiMode) return;
    if (!this.config.logConsole) return;
    for (const line of lines) {
      console.log(line);
    }
  }

  // ==========================================================================
  // Training Event Methods
  // ==========================================================================

  /** Log training initialization */
  init(model: string, config: TrainingConfigFields, numExamples?: number): void {
    if (this.config.tuiMode) {
      this.writeTui({ type: 'init', model, config: config as Record<string, unknown> });
    } else if (this.config.logConsole) {
      console.log(`\n🚀 Starting GRPO training`);
      if (numExamples) console.log(`   Examples: ${numExamples}`);
      console.log(`   Epochs: ${config.numEpochs}`);
      console.log(`   Batch size: ${config.batchSize}`);
      console.log(`   Group size: ${config.groupSize}`);
      console.log(`   Learning rate: ${config.learningRate}`);
    }

    if (this.jsonlPath && numExamples !== undefined) {
      this.writeJsonl({
        event: 'training_config',
        num_examples: numExamples,
        config: config as Record<string, unknown>,
        timestamp: new Date().toISOString(),
      });
    }
  }

  /** Log epoch start */
  epochStart(epoch: number, totalEpochs: number, numBatches: number): void {
    if (this.config.tuiMode) {
      this.writeTui({
        type: 'epoch_start',
        epoch: epoch + 1,
        totalEpochs,
        numBatches,
      });
    } else if (this.config.logConsole) {
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
    // Aggregate for epoch
    this.epochLoss.add(metrics.loss);
    this.epochReward.add(metrics.meanReward);
    this.epochAdvantage.add(metrics.meanAdvantage);
    this.epochTokens.add(metrics.totalTokens);

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
    } else if (this.config.logConsole && metrics.step % this.config.logInterval === 0) {
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

  /** Log epoch end/summary */
  epochEnd(epoch: number, totalEpochs: number, epochTimeSecs?: number): void {
    const avgLoss = this.epochLoss.mean();
    const avgReward = this.epochReward.mean();
    const avgAdvantage = this.epochAdvantage.mean();
    const totalTokens = this.epochTokens.sum();

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
    if (this.config.tuiMode) {
      this.writeTui({ type: 'checkpoint', path, step });
    } else if (this.config.logConsole) {
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
    const totalTimeSecs = totalTime / 1000;

    if (this.config.tuiMode) {
      this.writeTui({ type: 'complete', totalSteps, totalTimeSecs });
    } else if (this.config.logConsole) {
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

  /** Log generation sample (TUI only) */
  generation(sample: GenerationSample): void {
    if (!this.config.tuiMode) return;
    this.writeTui({
      type: 'generation',
      index: sample.index,
      prompt: sample.prompt,
      completion: sample.completion,
      reward: sample.reward,
      tokens: sample.tokens,
    });
  }

  /** Log training paused (TUI only) */
  paused(step: number): void {
    if (!this.config.tuiMode) return;
    this.writeTui({ type: 'paused', step });
  }

  /** Log training resumed (TUI only) */
  resumed(step: number): void {
    if (!this.config.tuiMode) return;
    this.writeTui({ type: 'resumed', step });
  }

  // ==========================================================================
  // Accessors
  // ==========================================================================

  /** Check if TUI mode is enabled */
  get isTuiMode(): boolean {
    return this.config.tuiMode;
  }

  /** Get the log interval */
  get logInterval(): number {
    return this.config.logInterval;
  }

  // ==========================================================================
  // Internal Methods
  // ==========================================================================

  private writeTui(msg: TuiMessage): void {
    if (!this.config.tuiMode) return;
    process.stdout.write(JSON.stringify(msg) + '\n');
  }

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
 * // Auto-detect TUI mode from environment
 * const logger = createTrainingLogger();
 *
 * // Explicit configuration
 * const logger = createTrainingLogger({
 *   tuiMode: false,
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
