/**
 * GRPO Training Engine - Rust-Native Training
 *
 * This module provides a Rust-native GRPO training engine that minimizes
 * FFI overhead by keeping the training loop entirely in Rust.
 *
 * ## Key Features
 * - Training loop runs in Rust (eliminates FFI overhead)
 * - Built-in reward functions (tool use, XML format, length, JSON schema)
 * - Custom JS rewards via callback pattern
 * - Gradient accumulation and memory management in Rust
 * - High-level train() method for full training runs
 * - Low-level trainStep() for custom training loops
 *
 * ## High-Level Usage (train with dataset)
 * ```typescript
 * const trainer = await GRPOTrainer.create({
 *   modelPath: './model',
 *   modelName: 'qwen3-0.6b',
 *   rewardFunction: (prompts, completions) => [...scores],
 * });
 * await trainer.train(dataset);
 * ```
 *
 * ## Low-Level Usage (step-by-step)
 * ```typescript
 * const model = await Qwen3Model.loadPretrained(modelPath);
 * const trainer = new GRPOTrainer(model, config);
 *
 * trainer.registerBuiltinReward({
 *   rewardType: 'ToolUse',
 *   allowedTools: ['search', 'calculate'],
 * });
 *
 * for (const batch of dataset) {
 *   const completions = await trainer.generateBatch(batch.prompts);
 *   const rewards = await myRewardFunction(batch.prompts, completions);
 *   const metrics = await trainer.trainStep(batch.prompts, rewards);
 * }
 * ```
 */

import {
  existsSync,
  mkdirSync,
  writeFileSync,
  readFileSync,
  readdirSync,
  copyFileSync,
  rmSync,
  statSync,
} from 'node:fs';
import { dirname, join } from 'node:path';
import * as readline from 'node:readline';

import {
  GrpoTrainingEngine,
  NativeRewardRegistry,
  Qwen3Model,
  OutputStore,
  buildRewardOutputs,
  type GrpoEngineConfig,
  type EngineEpochMetrics,
  type BuiltinRewardConfig,
  type GenerateBatchResult as NativeGenerateBatchResult,
  type TrainStepResultWithOutputs,
  type RewardOutput,
  type ToolDefinition,
} from '@mlx-node/core';

import type { ChatMessage, DatasetExample, RewardFunction } from '../types';
import { createTrainingLogger, type TrainingLogger } from './training-logger';

// Re-export native types
export { GrpoTrainingEngine, NativeRewardRegistry, OutputStore } from '@mlx-node/core';
export type {
  GrpoEngineConfig,
  EngineStepMetrics,
  EngineEpochMetrics,
  BuiltinRewardConfig,
  TrainStepResult,
  TrainStepResultWithOutputs,
  RewardOutput,
  OutputStoreConfig,
} from '@mlx-node/core';

/**
 * Configuration for GRPOTrainer
 */
export interface GRPOTrainerConfig<T = unknown> {
  // Model loading (for create() factory)
  modelPath?: string;
  modelName?: string;

  // Training hyperparameters
  learningRate?: number;
  gradientAccumulationSteps?: number;
  gradientClipNorm?: number;
  weightDecay?: number;

  // Training loop settings
  numEpochs?: number;
  batchSize?: number;

  // GRPO hyperparameters
  groupSize?: number;
  clipEpsilon?: number;
  klCoef?: number;
  lossType?: 'grpo' | 'dapo' | 'dr_grpo' | 'bnpo';
  advantageNormalization?: boolean;

  // Generation parameters
  /** Maximum completion length for both generation and training (default: 256).
   * Matches Python TRL's max_completion_length config. */
  maxCompletionLength?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;

  // Tool calling (for tool-use training)
  /**
   * Tool definitions for function calling.
   * When provided, tools are included in the chat template so the model
   * can generate tool calls. Essential for tool-use training.
   *
   * @example
   * ```typescript
   * import { createToolDefinition } from '@mlx-node/lm';
   *
   * const config: GRPOTrainerConfig = {
   *   tools: [
   *     createToolDefinition('lsp', 'Query API docs', { method: { type: 'string' } }, ['method']),
   *     createToolDefinition('run_js', 'Execute code', { code: { type: 'string' } }, ['code']),
   *   ],
   * };
   * ```
   */
  tools?: ToolDefinition[];

  /** Enable thinking mode for Qwen3 models (default: true).
   * When false, adds empty <think></think> tags to disable model thinking.
   * This is useful for tool-use training where you want direct outputs. */
  enableThinking?: boolean;

  // Reward configuration
  rewardType?: 'function' | 'builtin' | 'model';
  rewardFunction?: RewardFunction<T>;
  rewardModelPath?: string;

  // Optimization
  gradientClipValue?: number;

  // Logging and checkpointing
  logInterval?: number;
  saveInterval?: number;
  evalInterval?: number;
  outputDir?: string;
  logConsole?: boolean;
  logJsonl?: boolean;
  runName?: string;
  /** Maximum number of checkpoints to keep (default: 3). Set to 0 for unlimited. */
  maxCheckpoints?: number;

  // Device
  device?: string;

  // Checkpoint resumption
  /** Resume training from a checkpoint directory, or 'latest' to auto-find */
  resumeFromCheckpoint?: string | 'latest';

  // TUI mode
  /** Enable TUI mode - outputs structured JSONL to stdout and listens for commands on stdin */
  tuiMode?: boolean;

  // Memory optimization
  /**
   * Batch chunk size for LM head computation (memory optimization).
   * When set, the LM head (hidden_states -> logits) is computed in chunks
   * of this size to reduce peak memory usage.
   * Default: undefined (no chunking, full batch at once)
   * Recommended: 2 for batch_size >= 4 with large vocabularies (e.g., Qwen3 with 151936 vocab)
   * This reduces peak memory from ~1.2GB to ~300MB for Qwen3.
   */
  lmHeadChunkSize?: number;

  /**
   * Batch chunk size for transformer forward pass (memory optimization).
   * When set, the transformer layers process the batch in chunks of this size,
   * reducing peak memory from O(batch × heads × seq²) for attention.
   * Default: undefined (no chunking, full batch at once)
   * Recommended: 4 for batch_size >= 4 with groupSize >= 4
   * Memory savings: ~70-80% for batch=4, groupSize=4 (16 sequences → 4 at a time)
   */
  forwardChunkSize?: number;

  /**
   * Enable true parallel batch generation (default: false).
   * When true, all N*G sequences are processed in parallel using batched FFI
   * with per-sequence RoPE offsets. This provides 2-4x speedup for GRPO training.
   * When false, uses sequential generation (process one prompt at a time,
   * then expand KV cache for G completions).
   */
  useParallelBatchGeneration?: boolean;

  /**
   * Chunk size for vocabulary dimension in cross-entropy computation.
   * When computing logsumexp over large vocabularies (e.g., Qwen3's 151,936 tokens),
   * the computation is split into chunks of this size to reduce peak memory usage.
   *
   * Default: 65536 (2^16)
   *
   * Memory impact for Qwen3 (vocab=151936):
   * - Standard: Full [B, T, 151936] intermediate tensor
   * - Chunked (65536): 3 chunks, ~2.3x lower peak memory
   *
   * Set to a larger value (e.g., 262144) to reduce chunking overhead,
   * or smaller value (e.g., 32768) for tighter memory constraints.
   */
  vocabChunkSize?: number;

  // Output recording
  /** Output recording configuration (records all generations for debugging/research) */
  outputStore?: {
    /** Enable output recording (default: false) */
    enabled: boolean;
    /** Local database path (default: "{outputDir}/outputs.db") */
    localPath?: string;
    /** Remote Turso URL for cloud sync (optional) */
    remoteUrl?: string;
    /** Turso auth token (required if remoteUrl is set) */
    authToken?: string;
    /** Sync interval in seconds (default: 60, only for embedded replica mode) */
    syncInterval?: number;
  };
}

/**
 * Training state saved with checkpoints for resumption
 */
export interface TrainingState {
  step: number;
  epoch: number;
  timestamp: string;
}

/**
 * Result from generateBatch with detailed information
 */
export interface GenerateBatchResult {
  /** Generated completion texts */
  completionTexts: string[];
  /** Native generation result for passing to trainStepWithGenerations */
  nativeResult: NativeGenerateBatchResult;
  /** Completion token counts (derived from nativeResult) */
  tokenCounts: number[];
  /** Finish reasons for each completion ("eos", "length", or "repetition") */
  finishReasons: string[];
}

/**
 * Default configuration
 */
export const DEFAULT_GRPO_CONFIG: GRPOTrainerConfig = {
  learningRate: 1e-6,
  gradientAccumulationSteps: 1,
  gradientClipNorm: 1.0,
  weightDecay: 0.01,
  numEpochs: 1,
  batchSize: 1,
  groupSize: 4,
  clipEpsilon: 0.2,
  klCoef: 0.0,
  lossType: 'grpo',
  advantageNormalization: true,
  maxCompletionLength: 256,
  temperature: 0.8,
  topP: 0.95,
  repetitionPenalty: 1.1,
  logInterval: 1,
  saveInterval: 100,
  evalInterval: 100,
  logConsole: true,
  logJsonl: true,
  maxCheckpoints: 3,
};

/**
 * Training step metrics (compatible with both old and new APIs)
 */
export interface TrainStepMetrics {
  /** Current step number */
  step: number;
  /** GRPO loss value */
  loss: number;
  /** Mean reward across completions */
  meanReward: number;
  /** Standard deviation of rewards */
  stdReward: number;
  /** Mean advantage value */
  meanAdvantage: number;
  /** Std of advantages - indicates reward variance within groups */
  stdAdvantage: number;
  /** Total tokens generated this step */
  totalTokens: number;
  /** Whether gradients were applied */
  gradientsApplied?: boolean;
  /** Time for generation (ms) */
  generationTimeMs?: number;
  /** Time for training (ms) */
  trainingTimeMs?: number;
  /** Current epoch (for high-level API) */
  epoch?: number;
}

/**
 * Legacy type alias for backward compatibility
 */
export type TrainingMetrics = TrainStepMetrics;

// Note: RewardOutput is now built using the Rust buildRewardOutputs function
// which handles tool call parsing and thinking extraction natively.

/**
 * GRPO Trainer - Rust-Native Training Engine
 *
 * Provides a TypeScript-friendly interface to the Rust training engine.
 * Supports both high-level training (train()) and low-level step-by-step (trainStep()).
 */
export class GRPOTrainer<T = unknown> {
  private engine: GrpoTrainingEngine;
  private model: Qwen3Model;
  private config: GRPOTrainerConfig<T>;
  private rewardFn?: RewardFunction<T>;
  private currentEpoch: number = 0;
  private currentStep: number = 0;
  /** Original model path (for tokenizer files when saving checkpoints) */
  private originalModelPath?: string;

  // TUI state
  private paused: boolean = false;
  private stopRequested: boolean = false;
  private stdinInterface?: import('readline').Interface;
  private logger: TrainingLogger;
  private sampleDisplayMode: 'all' | 'best_worst' | 'random' = 'all';

  // Output recording
  private outputStore?: OutputStore;
  private outputStoreInitPromise?: Promise<void>;
  private outputStoreRunId?: string;
  private outputStorePath?: string;

  // Crash recovery
  private lastCheckpointStep: number = 0;
  private signalHandlersInstalled: boolean = false;

  /**
   * Create a new GRPO trainer from a model
   *
   * @param model - Pre-loaded Qwen3 model
   * @param config - Training configuration
   */
  constructor(model: Qwen3Model, config: Partial<GRPOTrainerConfig<T>> = {}, logger?: TrainingLogger) {
    // Auto-detect TUI mode from environment variable (set by mlx-train TUI)
    const tuiModeFromEnv = process.env.MLX_TUI_MODE === '1';
    if (tuiModeFromEnv && config.tuiMode === undefined) {
      config.tuiMode = true;
    }

    // Auto-enable database persistence in TUI mode (enables Database tab)
    if (tuiModeFromEnv && config.outputStore === undefined) {
      config.outputStore = { enabled: true };
    }

    this.config = { ...DEFAULT_GRPO_CONFIG, ...config };
    this.model = model;

    // Create or use provided logger (TUI mode auto-detected from MLX_TUI_MODE env var)
    this.logger =
      logger ??
      createTrainingLogger({
        logConsole: this.config.logConsole,
        logJsonl: this.config.logJsonl,
        outputDir: this.config.outputDir,
        runName: this.config.runName,
        logInterval: this.config.logInterval ?? 1,
      });

    // Set reward function if provided
    if (this.config.rewardFunction) {
      this.rewardFn = this.config.rewardFunction;
    }

    // Convert to native config
    const engineConfig: GrpoEngineConfig = {
      learningRate: this.config.learningRate,
      gradientAccumulationSteps: this.config.gradientAccumulationSteps,
      gradientClipNorm: this.config.gradientClipNorm,
      groupSize: this.config.groupSize,
      clipEpsilon: this.config.clipEpsilon,
      klCoef: this.config.klCoef,
      lossType: this.config.lossType,
      maxCompletionLength: this.config.maxCompletionLength,
      temperature: this.config.temperature,
      topP: this.config.topP,
      topK: this.config.topK,
      repetitionPenalty: this.config.repetitionPenalty,
      // Tool calling support
      tools: this.config.tools,
      enableThinking: this.config.enableThinking,
      // Memory optimization
      lmHeadChunkSize: this.config.lmHeadChunkSize,
      forwardChunkSize: this.config.forwardChunkSize,
      vocabChunkSize: this.config.vocabChunkSize,
      // Parallel batch generation
      useParallelBatchGeneration: this.config.useParallelBatchGeneration,
    };

    this.engine = new GrpoTrainingEngine(model, engineConfig);

    // Setup stdin handler if TUI mode
    if (this.config.tuiMode) {
      this.setupStdinHandler();
    }

    // Always setup signal handlers for crash recovery
    this.setupSignalHandlers();
  }

  /**
   * Setup stdin handler for TUI control commands
   */
  private setupStdinHandler(): void {
    if (!this.config.tuiMode) return;

    this.stdinInterface = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false,
    });

    this.stdinInterface.on('line', (line: string) => {
      const cmd = line.trim();
      this.handleStdinCommand(cmd);
    });
  }

  /**
   * Setup OS signal handlers for graceful shutdown on crash/interrupt
   *
   * Catches SIGTERM, SIGINT, and uncaught exceptions to:
   * - Save emergency checkpoint (if > 10 steps since last)
   * - Finalize OutputStore with 'crashed' status
   * - Exit cleanly
   */
  private setupSignalHandlers(): void {
    if (this.signalHandlersInstalled) return;
    this.signalHandlersInstalled = true;

    const gracefulShutdown = async (signal: string) => {
      this.logger.warn(`Received ${signal}, initiating graceful shutdown...`);
      this.stopRequested = true;

      try {
        // Skip checkpoint if recent one exists (within 10 steps)
        const stepsSinceCheckpoint = this.currentStep - this.lastCheckpointStep;
        if (this.config.outputDir && stepsSinceCheckpoint > 10) {
          this.logger.info(`Saving emergency checkpoint (${stepsSinceCheckpoint} steps since last)...`);
          await this.saveCheckpoint(`emergency-${this.currentStep}`);
        } else if (stepsSinceCheckpoint <= 10) {
          this.logger.info(`Skipping checkpoint (only ${stepsSinceCheckpoint} steps since last)`);
        }

        // Finalize OutputStore with crashed status
        if (this.outputStore) {
          await this.outputStore.endRun('crashed');
          await this.outputStore.flush();
          this.logger.info('OutputStore finalized with crashed status');
        }
      } catch (e) {
        console.error('Emergency save failed:', e);
      }

      // Cleanup stdin interface
      if (this.stdinInterface) {
        this.stdinInterface.close();
      }

      process.exit(0);
    };

    process.on('SIGTERM', () => {
      gracefulShutdown('SIGTERM').catch(console.error);
    });

    process.on('SIGINT', () => {
      gracefulShutdown('SIGINT').catch(console.error);
    });

    process.on('uncaughtException', (err) => {
      console.error('Uncaught exception:', err);
      gracefulShutdown('uncaughtException').catch(console.error);
    });

    process.on('unhandledRejection', (reason) => {
      console.error('Unhandled rejection:', reason);
      gracefulShutdown('unhandledRejection').catch(console.error);
    });
  }

  /**
   * Initialize the output store for recording training outputs
   */
  private async initOutputStore(stepsPerEpoch?: number): Promise<void> {
    // Guard against re-initialization (e.g., train() called after trainStepAuto())
    if (this.outputStore) return;

    const cfg = this.config.outputStore;
    if (!cfg?.enabled) {
      // Even without outputStore, send UI resume state if resuming from checkpoint
      if (this.config.resumeFromCheckpoint && this.currentStep > 0 && stepsPerEpoch) {
        this.sendResumeStateUiOnly(stepsPerEpoch);
      }
      return;
    }

    const localPath = cfg.localPath ?? join(this.config.outputDir ?? '.', 'outputs.db');
    this.outputStorePath = localPath;

    // Ensure parent directory exists (for lazy init via trainStepAuto)
    const parentDir = dirname(localPath);
    if (parentDir !== '.' && !existsSync(parentDir)) {
      mkdirSync(parentDir, { recursive: true });
    }
    this.outputStore = await OutputStore.local(localPath);

    // If resuming from checkpoint AND we have a run name AND checkpoint was actually loaded,
    // try to resume the existing database run. currentStep > 0 means checkpoint was loaded.
    if (this.config.resumeFromCheckpoint && this.config.runName && this.currentStep > 0) {
      const existingRun = await this.outputStore.findRunByName(this.config.runName);
      if (existingRun) {
        this.logger.info(`Resuming database run: ${this.config.runName} (${existingRun.id})`);
        await this.outputStore.resumeRun(existingRun.id);
        this.outputStoreRunId = existingRun.id;

        // Clean up any database records that are ahead of the checkpoint step
        // This prevents UNIQUE constraint errors when re-recording steps
        // Uses cascade delete to also clean up orphaned generations, tool_calls, and logs
        if (this.currentStep > 0) {
          const cleanupStats = await this.outputStore.deleteAllAfterStep(existingRun.id, this.currentStep);
          if (cleanupStats.stepsDeleted > 0) {
            this.logger.info(
              `Cleaned up stale records after step ${this.currentStep}: ` +
                `${cleanupStats.stepsDeleted} steps, ${cleanupStats.generationsDeleted} generations, ` +
                `${cleanupStats.logsDeleted} logs`,
            );
          }
        }

        this.logger.databasePath(localPath, this.outputStoreRunId, this.config.runName ?? undefined);

        // Send resume state to TUI for sparkline and aggregate restoration
        await this.sendResumeState(existingRun.id, stepsPerEpoch);
        return;
      } else {
        // Run name specified but not found - warn and create new
        this.logger.warn(`No existing run found with name: ${this.config.runName}. Starting new run.`);
      }
    }

    // Start a new run with sanitized config (no auth token)
    const modelName = this.config.modelName ?? 'qwen3';
    const modelPath = this.originalModelPath ?? this.config.modelPath ?? undefined;
    const sanitizedConfig = {
      ...this.config,
      outputStore: this.config.outputStore ? { ...this.config.outputStore, authToken: undefined } : undefined,
    };

    // Use startRunWithName if a run name is provided, otherwise use startRun
    if (this.config.runName) {
      this.outputStoreRunId = await this.outputStore.startRunWithName(
        this.config.runName,
        modelName,
        modelPath,
        JSON.stringify(sanitizedConfig),
      );
    } else {
      this.outputStoreRunId = await this.outputStore.startRun(modelName, modelPath, JSON.stringify(sanitizedConfig));
    }

    // Emit database path to TUI for the Database tab
    this.logger.databasePath(localPath, this.outputStoreRunId, this.config.runName ?? undefined);

    // If resuming from checkpoint but didn't find existing DB run, still send UI state
    // This ensures TUI displays correct batch progress even without historical data
    if (this.config.resumeFromCheckpoint && this.currentStep > 0 && stepsPerEpoch) {
      await this.sendResumeStateUiOnly(stepsPerEpoch);
    }
  }

  /**
   * Send minimal resume state to TUI for UI display only (no historical data)
   *
   * Used when resuming from checkpoint without a matching database run.
   * Ensures TUI shows correct epoch/batch progress.
   */
  private sendResumeStateUiOnly(stepsPerEpoch: number): void {
    if (!this.logger.isTuiMode) return;

    const totalEpochs = this.config.numEpochs ?? 1;
    const stepInEpoch = this.currentStep > 0 ? ((this.currentStep - 1) % stepsPerEpoch) + 1 : 0;

    this.logger.resumeState({
      step: this.currentStep,
      epoch: this.currentEpoch + 1, // 1-indexed
      totalEpochs,
      stepInEpoch,
      totalStepsInEpoch: stepsPerEpoch,
      metricsHistory: [], // No historical data
      aggregates: {
        bestReward: 0,
        avgReward: 0,
        rewardCount: 0,
        bestLoss: Infinity,
        avgLoss: 0,
        lossCount: 0,
        totalTokens: 0,
        avgGenerationTimeMs: 0,
        avgTrainingTimeMs: 0,
      },
    });

    this.logger.info(`Sent UI resume state to TUI (step ${this.currentStep}, no historical data)`);
  }

  /**
   * Send resume state to TUI for restoring sparklines and aggregates
   *
   * Queries the database for historical metrics and aggregates, then sends
   * to TUI via the resumeState message.
   *
   * @param runId - Database run ID
   * @param actualStepsPerEpoch - Actual steps per epoch from dataset (if known)
   */
  private async sendResumeState(runId: string, actualStepsPerEpoch?: number): Promise<void> {
    if (!this.outputStore || !this.logger.isTuiMode) return;

    try {
      // Query historical metrics (last 60 for sparklines)
      const metricsHistory = await this.outputStore.getRecentStepMetrics(runId, 60);

      // Query aggregate statistics
      const aggregates = await this.outputStore.getRunAggregates(runId);

      // Use actual steps per epoch if provided, otherwise use a reasonable default
      const totalEpochs = this.config.numEpochs ?? 1;
      const stepsPerEpoch = actualStepsPerEpoch ?? 50;

      // Calculate step within current epoch
      const stepInEpoch = this.currentStep > 0 ? ((this.currentStep - 1) % stepsPerEpoch) + 1 : 0;

      // Send resume state to TUI
      // Note: epoch is 1-indexed to match epochStart() convention
      this.logger.resumeState({
        step: this.currentStep,
        epoch: this.currentEpoch + 1,
        totalEpochs,
        stepInEpoch,
        totalStepsInEpoch: stepsPerEpoch,
        metricsHistory: metricsHistory.map((m) => ({
          step: Number(m.step),
          loss: m.loss,
          meanReward: m.meanReward,
          stdAdvantage: m.stdAdvantage,
          perplexity: m.perplexity ?? undefined,
          tokenAccuracy: m.tokenAccuracy ?? undefined,
          generationTimeMs: m.generationTimeMs ?? undefined,
          trainingTimeMs: m.trainingTimeMs ?? undefined,
        })),
        aggregates: {
          bestReward: aggregates.bestReward,
          avgReward: aggregates.avgReward,
          rewardCount: Number(aggregates.rewardCount),
          bestLoss: aggregates.bestLoss,
          avgLoss: aggregates.avgLoss,
          lossCount: Number(aggregates.lossCount),
          totalTokens: Number(aggregates.totalTokens),
          avgGenerationTimeMs: aggregates.avgGenerationTimeMs,
          avgTrainingTimeMs: aggregates.avgTrainingTimeMs,
        },
      });

      this.logger.info(`Sent ${metricsHistory.length} historical metrics to TUI`);
    } catch (err) {
      this.logger.warn(`Failed to send resume state to TUI: ${err}`);
    }
  }

  /**
   * Ensure output store is initialized (lazy initialization for low-level API users)
   * Uses promise mutex to prevent race conditions from concurrent calls.
   *
   * Call this method from custom training loops before starting training
   * to enable database recording and TUI database tab.
   */
  async ensureOutputStoreInitialized(): Promise<void> {
    if (this.outputStore) return; // Already initialized
    if (!this.config.outputStore?.enabled) return; // Not enabled

    // Use promise mutex to prevent concurrent initialization
    if (this.outputStoreInitPromise) {
      await this.outputStoreInitPromise;
      return;
    }

    this.outputStoreInitPromise = this.initOutputStore();
    try {
      await this.outputStoreInitPromise;
    } catch (err) {
      // Clear promise on failure to allow retry
      this.outputStoreInitPromise = undefined;
      throw err;
    }
  }

  /**
   * Get the output store (for querying recorded data)
   */
  getOutputStore(): OutputStore | undefined {
    return this.outputStore;
  }

  /**
   * Handle a command received from stdin
   */
  private handleStdinCommand(cmd: string): void {
    switch (cmd) {
      case 'PAUSE':
        this.paused = true;
        this.logger.paused(this.currentStep);
        break;
      case 'RESUME':
        this.paused = false;
        this.logger.resumed(this.currentStep);
        break;
      case 'SAVE_CHECKPOINT':
        // Will be handled in the training loop
        this.saveCheckpoint().catch(() => {});
        break;
      case 'STOP':
        this.stopRequested = true;
        break;
      default:
        // Handle SET commands (e.g., SET sample_display=best_worst)
        if (cmd.startsWith('SET ')) {
          const keyValue = cmd.slice(4); // Remove 'SET ' prefix
          const eqIdx = keyValue.indexOf('=');
          if (eqIdx > 0) {
            const key = keyValue.slice(0, eqIdx);
            const value = keyValue.slice(eqIdx + 1);
            if (key === 'sample_display') {
              if (value === 'all' || value === 'best_worst' || value === 'random') {
                this.sampleDisplayMode = value;
              }
            }
          }
        }
        break;
    }
  }

  /**
   * Wait for resume if paused, with polling
   */
  private async waitForResume(): Promise<void> {
    while (this.paused && !this.stopRequested) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  /**
   * Create a trainer by loading a model from disk
   *
   * This is the recommended way to create a trainer for training runs.
   * If resumeFromCheckpoint is set, loads from checkpoint instead of modelPath.
   *
   * @param config - Configuration including modelPath
   * @returns Promise<GRPOTrainer>
   */
  static async create<U>(config: GRPOTrainerConfig<U>): Promise<GRPOTrainer<U>> {
    if (!config.modelPath) {
      throw new Error('modelPath is required when using GRPOTrainer.create()');
    }

    // Validate unsupported config options (fail fast)
    if (config.advantageNormalization === false) {
      throw new Error('advantageNormalization=false is not yet supported. Remove this option or set to true.');
    }
    if (config.weightDecay !== undefined && config.weightDecay !== 0.01) {
      throw new Error(
        'Custom weightDecay is not yet implemented. Optimizer uses simple SGD. Remove weightDecay from config or use default (0.01).',
      );
    }
    if (config.rewardType === 'model') {
      throw new Error(
        'rewardType="model" is not implemented. Use rewardType="function" with a custom reward function.',
      );
    }
    if (config.rewardModelPath) {
      throw new Error('rewardModelPath is not implemented. Use rewardType="function" with a custom reward function.');
    }
    if (config.device && config.device !== 'metal') {
      throw new Error(
        `device="${config.device}" is not supported. MLX only supports Metal GPU. Remove device from config.`,
      );
    }

    // Create logger early (before model loading)
    // TUI mode is auto-detected from MLX_TUI_MODE env var (set by mlx-tui)
    const logger = createTrainingLogger({
      logConsole: config.logConsole,
      logJsonl: config.logJsonl,
      outputDir: config.outputDir,
      runName: config.runName,
      logInterval: config.logInterval ?? 1,
    });

    let modelPath = config.modelPath;
    let resumedState: TrainingState | null = null;

    // Handle checkpoint resumption
    if (config.resumeFromCheckpoint) {
      const checkpointPath =
        config.resumeFromCheckpoint === 'latest'
          ? GRPOTrainer.findLatestCheckpoint(config.outputDir)
          : config.resumeFromCheckpoint;

      if (checkpointPath) {
        const statePath = join(checkpointPath, 'training_state.json');
        if (existsSync(statePath)) {
          resumedState = JSON.parse(readFileSync(statePath, 'utf-8'));

          // Fallback: If training_state.json has step 0 but checkpoint name suggests otherwise,
          // derive step from checkpoint name (e.g., checkpoint-8 → step 8)
          // This handles cases where training_state.json was corrupted or overwritten
          if (resumedState && resumedState.step === 0) {
            const checkpointName = checkpointPath.split('/').pop() ?? '';
            const match = checkpointName.match(/^checkpoint-(\d+)$/);
            if (match) {
              const derivedStep = parseInt(match[1], 10);
              if (derivedStep > 0) {
                logger.warn(
                  `Checkpoint ${checkpointName} has step 0 in training_state.json but name suggests step ${derivedStep}. Using ${derivedStep}.`,
                );
                resumedState.step = derivedStep;
                // Estimate epoch from step (will be refined by actual training data)
              }
            }
          }

          logger.info(
            `Resuming from checkpoint: ${checkpointPath} (step ${resumedState?.step}, epoch ${resumedState?.epoch})`,
          );
        }
        // Load model weights from checkpoint
        modelPath = checkpointPath;
      } else if (config.resumeFromCheckpoint === 'latest') {
        logger.info('No checkpoint found, starting fresh training');
      }
    }

    // Get model name for display
    const modelName = modelPath.split('/').pop() ?? 'Unknown';
    logger.status('loading', `Loading ${modelName}...`);

    // Load model from disk (checkpoint or original)
    const model = await Qwen3Model.loadPretrained(modelPath);

    logger.status('loading', `${modelName} loaded`);

    // Create trainer with the pre-created logger
    const trainer = new GRPOTrainer(model, config, logger);

    // Always store the original model path (for tokenizer files when saving checkpoints)
    trainer.originalModelPath = config.modelPath;

    // Restore training state if resuming
    if (resumedState) {
      trainer.currentStep = resumedState.step;
      trainer.currentEpoch = resumedState.epoch;
    }

    return trainer;
  }

  /**
   * Find the latest checkpoint in the output directory
   */
  static findLatestCheckpoint(outputDir?: string): string | null {
    if (!outputDir || !existsSync(outputDir)) {
      return null;
    }

    const entries = readdirSync(outputDir, { withFileTypes: true });
    const checkpoints = entries
      .filter((e) => e.isDirectory() && e.name.startsWith('checkpoint-'))
      .map((e) => ({
        name: e.name,
        step: parseInt(e.name.replace('checkpoint-', ''), 10),
        path: join(outputDir, e.name),
      }))
      .filter((c) => !isNaN(c.step))
      .sort((a, b) => b.step - a.step);

    return checkpoints.length > 0 ? checkpoints[0].path : null;
  }

  /**
   * Register a built-in reward function
   *
   * Built-in rewards run entirely in Rust with no FFI overhead.
   *
   * @example
   * ```typescript
   * // Tool use validation
   * trainer.registerBuiltinReward({
   *   rewardType: 'ToolUse',
   *   allowedTools: ['search', 'calculate'],
   *   required: true,
   *   weight: 1.0,
   * });
   *
   * // XML format validation
   * trainer.registerBuiltinReward({
   *   rewardType: 'XmlFormat',
   *   requiredTags: ['thinking', 'answer'],
   *   weight: 0.5,
   * });
   *
   * // Length-based reward
   * trainer.registerBuiltinReward({
   *   rewardType: 'Length',
   *   minLength: 100,
   *   maxLength: 500,
   *   useChars: true,
   * });
   * ```
   */
  registerBuiltinReward(config: BuiltinRewardConfig): void {
    this.engine.registerBuiltinReward(config);
  }

  /**
   * Set a custom JavaScript reward function
   *
   * The function will be called after generation to compute rewards.
   *
   * @param fn - Reward function that takes prompts and completions
   */
  setRewardFunction(fn: RewardFunction<T>): void {
    this.rewardFn = fn;
  }

  /**
   * Generate completions for prompts
   *
   * Generates `groupSize` completions per prompt.
   * Returns all data needed for training, including tokens and log probabilities.
   *
   * @param prompts - Array of chat conversations
   * @returns GenerateBatchResult with completion texts and native generation data
   */
  async generateBatch(prompts: ChatMessage[][]): Promise<GenerateBatchResult> {
    if (prompts.length === 0) {
      return {
        completionTexts: [],
        nativeResult: {
          completionTexts: [],
          completionTokens: [],
          completionLogprobs: [],
          completionLengths: [],
          finishReasons: [],
        },
        tokenCounts: [],
        finishReasons: [],
      };
    }

    // Call the native engine to generate completions with full data
    const nativeResult = await this.engine.generateBatchForTraining(prompts);

    return {
      completionTexts: nativeResult.completionTexts,
      nativeResult,
      tokenCounts: nativeResult.completionLengths,
      finishReasons: nativeResult.finishReasons,
    };
  }

  /**
   * Score completions using built-in rewards
   *
   * @param prompts - Prompt texts (one per completion)
   * @param completions - Completion texts
   * @returns Array of reward scores
   */
  scoreCompletions(prompts: string[], completions: string[]): number[] {
    return this.engine.scoreCompletions(prompts, completions);
  }

  /**
   * Score generations using the configured reward function.
   *
   * Builds RewardOutput array with structured completion data and passes to reward function.
   *
   * @param prompts - Array of chat conversations
   * @param completions - Generated completion texts
   * @param context - Context for the reward function
   * @param groupSize - Number of completions per prompt (optional, defaults to config.groupSize)
   * @param tokenCounts - Token counts for each completion (optional, defaults to 0s)
   * @param finishReasons - Finish reasons from generation (optional, e.g. "eos", "length", "repetition")
   * @returns Promise<Float32Array> of reward scores
   */
  async scoreGenerations(
    prompts: ChatMessage[][],
    completions: string[],
    context: T,
    groupSize?: number,
    tokenCounts?: number[],
    finishReasons?: string[],
  ): Promise<Float32Array> {
    const effectiveGroupSize = groupSize ?? this.config.groupSize ?? 4;
    const expectedCompletions = prompts.length * effectiveGroupSize;

    if (completions.length !== expectedCompletions) {
      throw new Error(
        `Expected ${expectedCompletions} completions (${prompts.length} prompts × ${effectiveGroupSize} groupSize) but got ${completions.length}`,
      );
    }

    if (!this.rewardFn && !this.engine.hasBuiltinRewards) {
      throw new Error('No reward function configured. Set rewardFunction in config or call setRewardFunction()');
    }

    // Convert ChatMessage[][] to string[] for Rust function
    const promptTexts = prompts.map((msgs) => msgs.map((m) => `${m.role}: ${m.content}`).join('\n'));

    // Use provided token counts or default to 0
    const effectiveTokenCounts = tokenCounts ?? completions.map(() => 0);

    // Use provided finish reasons or default to empty (triggers inference fallback in Rust)
    const effectiveFinishReasons = finishReasons ?? [];

    // Build structured reward outputs using Rust function
    const rewardOutputs = buildRewardOutputs(
      promptTexts,
      completions,
      effectiveTokenCounts,
      effectiveFinishReasons,
      effectiveGroupSize,
    );

    let rewards: number[] | Float32Array;

    if (this.rewardFn) {
      rewards = await this.rewardFn(rewardOutputs, context);
    } else {
      // For built-in rewards, extract prompts and completions for legacy API
      const promptStrings = rewardOutputs.map((o) => o.prompt);
      const completionTexts = rewardOutputs.map((o) => o.completion.rawText);
      rewards = this.scoreCompletions(promptStrings, completionTexts);
    }

    const rewardsArray = rewards instanceof Float32Array ? rewards : Float32Array.from(rewards);

    if (rewardsArray.length !== expectedCompletions) {
      throw new Error(`Reward function returned ${rewardsArray.length} rewards but expected ${expectedCompletions}`);
    }

    return rewardsArray;
  }

  /**
   * Run a training step
   *
   * This method:
   * 1. Generates completions with tokens and log probabilities
   * 2. Computes rewards using the configured reward function
   * 3. Trains using the SAME completions that were scored (no double-generation)
   *
   * @param prompts - Array of chat conversations
   * @returns Training step metrics
   */
  async trainStep(prompts: ChatMessage[][], context?: T): Promise<TrainStepMetrics> {
    const { metrics } = await this.trainStepAuto(prompts, context);
    return metrics;
  }

  /**
   * Run a complete training step with automatic reward computation
   *
   * This method combines generation, reward scoring, and training into a single
   * Rust call, eliminating FFI overhead by keeping token data in Rust memory.
   *
   * 1. Generates completions with full token/logprob data (stays in Rust)
   * 2. Calls JS reward function with RewardOutput[]
   * 3. Performs training update using the in-memory data
   *
   * @param prompts - Array of chat conversations
   * @param context - Context for the reward function
   * @returns Training metrics and generated completions
   */
  async trainStepAuto(
    prompts: ChatMessage[][],
    context?: T,
  ): Promise<{ metrics: TrainStepMetrics; completions: string[]; rewards: number[]; completionLengths: number[] }> {
    // Lazy initialize output store for low-level API users
    await this.ensureOutputStoreInitialized();

    if (!this.rewardFn && !this.engine.hasBuiltinRewards) {
      throw new Error('No reward function configured. Set rewardFunction in config or call setRewardFunction()');
    }

    // Create reward callback that parses JSON and converts output to number[]
    // The Rust side serializes Vec<RewardOutput> to JSON because complex nested types
    // don't convert properly through ThreadsafeFunction
    // Note: With CalleeHandled=true (default), callback receives (err, value) format
    // Using ThreadsafeFunction<T, Promise<R>> pattern so Rust can await the Promise
    const rewardCallback = async (err: Error | null, outputsJson: string): Promise<number[]> => {
      const rewardStart = Date.now();
      if (err) {
        throw new Error(`Reward callback error from Rust: ${err.message}`);
      }

      if (!outputsJson || outputsJson === 'null') {
        throw new Error(`Invalid JSON received from Rust: ${outputsJson}`);
      }
      // Parse JSON and convert snake_case to camelCase for TypeScript compatibility
      // Rust's serde serializes as snake_case but TypeScript expects camelCase
      const rawOutputs = JSON.parse(outputsJson) as Array<{
        prompt: string;
        completion: {
          text: string;
          raw_text: string;
          tool_calls: unknown[];
          thinking: string | null;
          num_tokens: number;
          finish_reason: string;
        };
        expected_answer: string | null;
      }>;

      // Convert to RewardOutput format with proper camelCase properties
      const outputs: RewardOutput[] = rawOutputs.map((o) => ({
        prompt: o.prompt,
        completion: {
          text: o.completion.text,
          rawText: o.completion.raw_text,
          toolCalls: (
            o.completion.tool_calls as Array<{
              id: string;
              name: string;
              arguments: Record<string, unknown>;
              status: string;
              error?: string;
              raw_content: string;
            }>
          ).map((tc) => ({
            id: tc.id,
            name: tc.name,
            arguments: tc.arguments,
            status: tc.status, // 'ok' | 'invalid_json' | 'missing_name'
            error: tc.error,
            rawContent: tc.raw_content,
          })),
          thinking: o.completion.thinking ?? undefined,
          numTokens: o.completion.num_tokens,
          finishReason: o.completion.finish_reason,
        },
        expectedAnswer: o.expected_answer ?? undefined,
      }));

      this.logger.info(`  → Computing rewards for ${outputs.length} completions...`);

      let rewards: number[] | Float32Array;
      if (this.rewardFn) {
        // @ts-expect-error context is optional
        rewards = await this.rewardFn(outputs, context);
      } else {
        // Use built-in rewards
        const promptStrings = outputs.map((o) => o.prompt);
        const completionTexts = outputs.map((o) => o.completion.rawText);
        rewards = this.scoreCompletions(promptStrings, completionTexts);
      }

      // Convert Float32Array to plain number[] for NAPI compatibility
      let result: number[];
      if (rewards instanceof Float32Array) {
        result = Array.from(rewards, (v) => Number(v));
      } else {
        result = rewards.map((v) => Number(v));
      }

      const rewardDuration = Date.now() - rewardStart;
      const avgReward = result.reduce((a, b) => a + b, 0) / result.length;
      this.logger.info(`  → Rewards computed in ${rewardDuration}ms (avg=${avgReward.toFixed(2)})`);

      return result;
    };

    // Call unified Rust method - generation, scoring, and training in one FFI call
    // Use recording method if output store is enabled
    const recordOutputs = !!this.outputStore;
    const result: TrainStepResultWithOutputs = await this.engine.trainStepAuto(prompts, rewardCallback, recordOutputs);

    this.currentStep++;

    // Record outputs to database if enabled
    if (this.outputStore && result.outputsJson) {
      try {
        await this.outputStore.recordStepFromOutputs(
          this.currentStep,
          result.metrics,
          result.outputsJson,
          result.rewards,
          this.config.groupSize ?? 4,
        );
      } catch (err) {
        // Log error but don't fail training
        console.error('[OutputStore] Failed to record step:', err);
      }
    }

    return {
      metrics: { ...result.metrics, epoch: this.currentEpoch },
      completions: result.completions,
      rewards: result.rewards,
      completionLengths: result.completionLengths,
    };
  }

  /**
   * Increment the step counter (for custom training loops)
   *
   * Call this after each training step when using low-level APIs like
   * engine.trainStepWithGenerations() instead of trainer.trainStepAuto().
   */
  incrementStep(): void {
    this.currentStep++;
  }

  /**
   * Get the current step number
   */
  getStep(): number {
    return this.currentStep;
  }

  /**
   * Get the current epoch number
   */
  getEpoch(): number {
    return this.currentEpoch;
  }

  /**
   * Record a training step to the output store database (for custom training loops)
   *
   * Use this when building custom training loops with engine.trainStepWithGenerations().
   * The step number should be the value after incrementStep() was called.
   *
   * @param step - Step number
   * @param metrics - Step metrics from the engine
   * @param completions - Generated completion texts
   * @param rewards - Reward values for each completion
   * @param prompts - Prompt messages for each completion
   */
  async recordStepToDatabase(
    step: number,
    metrics: {
      loss: number;
      meanReward: number;
      stdReward: number;
      meanAdvantage: number;
      stdAdvantage: number;
      totalTokens: number;
    },
    completions: string[],
    rewards: number[],
    prompts: string[],
  ): Promise<void> {
    if (!this.outputStore) return;

    const groupSize = this.config.groupSize ?? 4;

    // Build outputs JSON in the format expected by recordStepFromOutputs
    const outputs = completions.map((text, i) => ({
      prompt: prompts[Math.floor(i / groupSize)] ?? '',
      completion: {
        text,
        raw_text: text,
        tool_calls: [],
        thinking: null,
        num_tokens: text.length, // Approximate
        finish_reason: 'eos',
      },
      expected_answer: null,
    }));

    const outputsJson = JSON.stringify(outputs);

    try {
      await this.outputStore.recordStepFromOutputs(
        step,
        {
          step,
          loss: metrics.loss,
          totalTokens: metrics.totalTokens,
          meanReward: metrics.meanReward,
          stdReward: metrics.stdReward,
          meanAdvantage: metrics.meanAdvantage,
          stdAdvantage: metrics.stdAdvantage,
          generationTimeMs: 0,
          trainingTimeMs: 0,
          peakMemoryMb: 0,
          activeMemoryMb: 0,
          gradientsApplied: true,
        },
        outputsJson,
        rewards,
        groupSize,
      );
    } catch (err) {
      console.error('[OutputStore] Failed to record step:', err);
    }
  }

  /**
   * Run a full training loop over a dataset
   *
   * This is the high-level training API that handles:
   * - Epoch iteration
   * - Batching
   * - Generation and reward computation
   * - Logging (if configured)
   * - Checkpoint saving and resumption
   * - TUI mode support (pause/resume, sample reporting)
   *
   * @param dataset - Array of DatasetExample items
   */
  async train(dataset: DatasetExample[]): Promise<void> {
    if (dataset.length === 0) {
      return;
    }

    const numEpochs = this.config.numEpochs ?? 1;
    const batchSize = this.config.batchSize ?? 1;
    const saveInterval = this.config.saveInterval ?? 100;

    // Create output directory if needed
    if (this.config.outputDir && !existsSync(this.config.outputDir)) {
      mkdirSync(this.config.outputDir, { recursive: true });
    }

    // Calculate total steps per epoch BEFORE initOutputStore (needed for accurate resume state)
    const stepsPerEpoch = Math.ceil(dataset.length / batchSize);

    // Initialize output store if enabled (pass stepsPerEpoch for accurate batch display on resume)
    await this.initOutputStore(stepsPerEpoch);

    // Determine starting point based on resumed state
    const startEpoch = this.currentEpoch;
    const startStep = this.currentStep;
    const startBatchIdx = startStep > 0 ? startStep % stepsPerEpoch : 0;

    // Get model name from path
    const modelName = this.originalModelPath?.split('/').pop() ?? this.config.modelPath?.split('/').pop() ?? 'Unknown';

    // Log training start
    this.logger.init(
      modelName,
      {
        trainingType: 'grpo',
        numEpochs,
        batchSize,
        groupSize: this.config.groupSize ?? 4,
        learningRate: this.config.learningRate ?? 1e-6,
      },
      dataset.length,
    );

    if (startStep > 0) {
      this.logger.info(
        `Resuming from step ${startStep} (epoch ${startEpoch + 1}, batch ${startBatchIdx + 1}/${stepsPerEpoch})`,
      );
    }

    for (let epoch = startEpoch; epoch < numEpochs; epoch++) {
      // Check for stop request
      if (this.stopRequested) break;

      this.currentEpoch = epoch;
      this.startEpoch();
      const epochStartTime = Date.now();

      // Log epoch start
      this.logger.epochStart(epoch, numEpochs, stepsPerEpoch);

      // Calculate starting batch index for this epoch
      const epochStartBatch = epoch === startEpoch && startStep > 0 ? startBatchIdx * batchSize : 0;

      // Iterate through batches
      for (let i = epochStartBatch; i < dataset.length; i += batchSize) {
        // Check for stop request
        if (this.stopRequested) break;

        // Wait if paused
        if (this.paused) {
          await this.waitForResume();
          if (this.stopRequested) break;
        }

        const batch = dataset.slice(i, Math.min(i + batchSize, dataset.length));

        // Extract prompts and answers from batch
        const prompts = batch.map((ex) => ex.prompt);

        // Verbose logging for debugging stuck batches
        const batchNum = Math.floor(i / batchSize) + 1;
        this.logger.info(
          `Batch ${batchNum}/${stepsPerEpoch} starting (${prompts.length} prompts × ${this.config.groupSize ?? 4} groups)`,
        );

        // Run training step with auto reward computation
        const stepStartTime = Date.now();
        const { metrics, completions, rewards, completionLengths } = await this.trainStepAuto(prompts);
        const stepDuration = Date.now() - stepStartTime;
        this.logger.info(
          `Batch ${batchNum}/${stepsPerEpoch} done in ${(stepDuration / 1000).toFixed(1)}s ` +
            `(gen=${metrics.generationTimeMs?.toFixed(0) ?? '?'}ms, train=${metrics.trainingTimeMs?.toFixed(0) ?? '?'}ms, loss=${metrics.loss.toFixed(4)})`,
        );

        // Log step metrics (logger handles TUI/console mode internally)
        this.logger.step(metrics, Math.floor(i / batchSize), stepsPerEpoch);

        // Report generation samples to TUI based on display mode
        // In console mode, logger.generation() is a no-op
        const groupSize = this.config.groupSize ?? 4;

        // Determine which sample indices to report based on display mode
        let indicesToReport: number[];
        if (this.sampleDisplayMode === 'all') {
          // Report all samples
          indicesToReport = Array.from({ length: completions.length }, (_, i) => i);
        } else if (this.sampleDisplayMode === 'best_worst') {
          // Find indices of best (max reward) and worst (min reward) samples
          let bestIdx = 0;
          let worstIdx = 0;
          let bestReward = rewards[0];
          let worstReward = rewards[0];
          for (let j = 1; j < rewards.length; j++) {
            if (rewards[j] > bestReward) {
              bestReward = rewards[j];
              bestIdx = j;
            }
            if (rewards[j] < worstReward) {
              worstReward = rewards[j];
              worstIdx = j;
            }
          }
          // Avoid duplicates if best and worst are the same
          indicesToReport = bestIdx === worstIdx ? [bestIdx] : [bestIdx, worstIdx];
        } else {
          // random: pick 2 random samples (or fewer if completions.length < 2)
          const numSamples = Math.min(2, completions.length);
          const shuffled = Array.from({ length: completions.length }, (_, i) => i);
          // Fisher-Yates partial shuffle for first numSamples
          for (let k = 0; k < numSamples; k++) {
            const randIdx = k + Math.floor(Math.random() * (shuffled.length - k));
            [shuffled[k], shuffled[randIdx]] = [shuffled[randIdx], shuffled[k]];
          }
          indicesToReport = shuffled.slice(0, numSamples);
        }

        for (const j of indicesToReport) {
          // Get the prompt for this completion (each prompt has groupSize completions)
          const promptIdx = Math.floor(j / groupSize);
          const promptMessages = prompts[promptIdx] ?? [];
          // Format prompt as text (last user message is most relevant)
          const lastUserMsg = promptMessages.filter((m) => m.role === 'user').pop();
          const promptText = lastUserMsg?.content ?? '';

          this.logger.generation({
            index: j,
            prompt: promptText,
            completion: completions[j],
            reward: rewards[j],
            tokens: completionLengths[j] ?? this.config.maxCompletionLength ?? 256,
          });
        }

        // Save checkpoint periodically
        if (this.config.outputDir && this.currentStep > 0 && this.currentStep % saveInterval === 0) {
          const path = await this.saveCheckpoint();
          if (path) {
            this.logger.checkpoint(path, this.currentStep);
          }
        }

        // Check for emergency checkpoint (triggered by consecutive NaN gradients)
        if (this.config.outputDir && this.engine.needsEmergencySave) {
          this.logger.warn(
            `[EMERGENCY] Saving emergency checkpoint at step ${this.currentStep} due to consecutive NaN gradients. ` +
              `Total NaN gradient count: ${this.engine.nanGradientCount}`,
          );
          await this.saveCheckpoint(`emergency-checkpoint-${this.currentStep}`);
          this.engine.clearEmergencySaveFlag();
          this.logger.warn(
            `[EMERGENCY] Emergency checkpoint saved. Consider reducing learning rate or checking training data. ` +
              `Training will continue but model quality may be degraded.`,
          );
        }
      }

      const epochEndTime = Date.now();
      const epochTimeSecs = (epochEndTime - epochStartTime) / 1000;
      this.endEpoch(epochTimeSecs);

      this.logger.epochEnd(epoch, numEpochs, epochTimeSecs);
    }

    // Save final checkpoint
    if (this.config.outputDir && !this.stopRequested) {
      const path = await this.saveCheckpoint('final');
      if (path) {
        this.logger.checkpoint(path, this.currentStep);
      }
    }

    // Log completion
    this.logger.complete(this.currentStep);

    // End output store run if active
    if (this.outputStore) {
      const status = this.stopRequested ? 'stopped' : 'completed';
      await this.outputStore.endRun(status);
      await this.outputStore.flush();
    }

    // Cleanup stdin interface
    if (this.stdinInterface) {
      this.stdinInterface.close();
    }
  }

  /**
   * Save a checkpoint with model weights and training state
   *
   * Validates model health before saving to prevent corrupted checkpoints.
   * Emergency checkpoints (name starts with 'emergency-') skip validation.
   *
   * @param name - Checkpoint name (default: "checkpoint-{step}")
   * @returns Path to saved checkpoint, or empty string if save was skipped due to corruption
   */
  async saveCheckpoint(name?: string): Promise<string> {
    const checkpointName = name ?? `checkpoint-${this.currentStep}`;
    const outputDir = this.config.outputDir ?? './outputs';
    const checkpointPath = join(outputDir, checkpointName);

    // Create checkpoint directory
    if (!existsSync(checkpointPath)) {
      mkdirSync(checkpointPath, { recursive: true });
    }

    // Save training state
    const state: TrainingState = {
      step: this.currentStep,
      epoch: this.currentEpoch,
      timestamp: new Date().toISOString(),
    };
    const statePath = join(checkpointPath, 'training_state.json');
    writeFileSync(statePath, JSON.stringify(state, null, 2));

    // Save model weights
    await this.model.saveModel(checkpointPath);

    // Copy tokenizer files from original model path (required for loading checkpoints)
    const tokenizerSource = this.originalModelPath ?? this.config.modelPath;
    if (tokenizerSource) {
      const tokenizerFiles = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'];
      for (const file of tokenizerFiles) {
        const srcPath = join(tokenizerSource, file);
        const destPath = join(checkpointPath, file);
        if (existsSync(srcPath) && !existsSync(destPath)) {
          copyFileSync(srcPath, destPath);
        }
      }
    }

    this.logger.info(`Checkpoint saved: ${checkpointPath}`);

    // Track last checkpoint step for emergency save throttling
    this.lastCheckpointStep = this.currentStep;

    // Clean up old checkpoints to save disk space
    const maxCheckpoints = this.config.maxCheckpoints ?? 3;
    if (maxCheckpoints > 0) {
      this.cleanupOldCheckpoints(outputDir, maxCheckpoints);
    }

    return checkpointPath;
  }

  /**
   * Remove old checkpoints, keeping only the most recent ones
   * Preserves 'final' and 'emergency-*' checkpoints
   */
  private cleanupOldCheckpoints(outputDir: string, maxToKeep: number): void {
    try {
      const entries = readdirSync(outputDir, { withFileTypes: true });

      // Find regular checkpoint directories (checkpoint-N pattern)
      const checkpoints: { name: string; step: number; mtime: Date }[] = [];
      for (const entry of entries) {
        if (!entry.isDirectory()) continue;

        // Skip 'final' and 'emergency-*' checkpoints
        if (entry.name === 'final' || entry.name.startsWith('emergency-')) continue;

        // Match checkpoint-N pattern
        const match = entry.name.match(/^checkpoint-(\d+)$/);
        if (match) {
          const checkpointPath = join(outputDir, entry.name);
          const stat = statSync(checkpointPath);
          checkpoints.push({
            name: entry.name,
            step: parseInt(match[1], 10),
            mtime: stat.mtime,
          });
        }
      }

      // Sort by step number descending (newest first)
      checkpoints.sort((a, b) => b.step - a.step);

      // Remove old checkpoints beyond maxToKeep
      if (checkpoints.length > maxToKeep) {
        const toRemove = checkpoints.slice(maxToKeep);
        for (const checkpoint of toRemove) {
          const checkpointPath = join(outputDir, checkpoint.name);
          rmSync(checkpointPath, { recursive: true, force: true });
          this.logger.debug(`Removed old checkpoint: ${checkpoint.name}`);
        }
      }
    } catch (error) {
      // Don't fail training if cleanup fails
      this.logger.warn(`Failed to cleanup old checkpoints: ${error}`);
    }
  }

  /**
   * Start a new training epoch
   */
  startEpoch(): void {
    this.engine.startEpoch();
  }

  /**
   * End the current epoch and get metrics
   *
   * @param epochTimeSecs - Duration of the epoch in seconds
   */
  endEpoch(epochTimeSecs: number): EngineEpochMetrics {
    return this.engine.endEpoch(epochTimeSecs);
  }

  /**
   * Reset the trainer for a new training run
   */
  reset(): void {
    this.engine.reset();
  }

  /**
   * Get current training step
   */
  get step(): number {
    return Number(this.engine.step);
  }

  /**
   * Get current epoch
   */
  get epoch(): number {
    return this.engine.epoch;
  }

  /**
   * Get current micro-step within gradient accumulation
   */
  get microStep(): number {
    return this.engine.microStep;
  }

  /**
   * Check if built-in rewards are configured
   */
  get hasBuiltinRewards(): boolean {
    return this.engine.hasBuiltinRewards;
  }

  /**
   * Get names of registered reward functions
   */
  get rewardNames(): string[] {
    return this.engine.rewardNames;
  }

  /**
   * Get the underlying native engine
   *
   * For advanced use cases that need direct access.
   */
  getNativeEngine(): GrpoTrainingEngine {
    return this.engine;
  }
}

/**
 * Create a standalone reward registry for testing rewards
 *
 * @example
 * ```typescript
 * const registry = createRewardRegistry();
 * registry.register({
 *   rewardType: 'ToolUse',
 *   allowedTools: ['search'],
 * });
 *
 * const score = registry.score('prompt', 'completion with <tool_call>...</tool_call>');
 * ```
 */
export function createRewardRegistry(): NativeRewardRegistry {
  return new NativeRewardRegistry();
}
