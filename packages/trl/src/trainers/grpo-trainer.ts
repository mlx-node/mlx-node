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
 *   modelConfig: 'qwen3-0.6b',
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
import { join } from 'node:path';
import * as readline from 'node:readline';

import {
  GrpoTrainingEngine,
  NativeRewardRegistry,
  Qwen3Model,
  type GrpoEngineConfig,
  type EngineEpochMetrics,
  type BuiltinRewardConfig,
  type GenerateBatchResult as NativeGenerateBatchResult,
  type EngineStepMetrics,
} from '@mlx-node/core';

import type { ChatMessage, DatasetExample } from '../types';
import { createTrainingLogger, type TrainingLogger } from './training-logger';

// Re-export native types
export { GrpoTrainingEngine, NativeRewardRegistry } from '@mlx-node/core';
export type { GrpoEngineConfig, EngineStepMetrics, EngineEpochMetrics, BuiltinRewardConfig } from '@mlx-node/core';

/**
 * Reward function type for custom rewards
 *
 * @param prompts - Prompt texts (one per completion)
 * @param completions - Generated completion texts
 * @param answers - Expected answers (for accuracy-based rewards), can be ignored
 */
export type RewardFn = (
  prompts: string[],
  completions: string[],
  answers: (string | null)[],
) => number[] | Float32Array | Promise<number[] | Float32Array>;

/**
 * Configuration for GRPOTrainer
 */
export interface GRPOTrainerConfig {
  // Model loading (for create() factory)
  modelPath?: string;
  modelConfig?: string;

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
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;

  // Reward configuration
  rewardType?: 'function' | 'builtin' | 'model';
  rewardFunction?: RewardFn;
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
  maxNewTokens: 256,
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

/**
 * GRPO Trainer - Rust-Native Training Engine
 *
 * Provides a TypeScript-friendly interface to the Rust training engine.
 * Supports both high-level training (train()) and low-level step-by-step (trainStep()).
 */
export class GRPOTrainer {
  private engine: GrpoTrainingEngine;
  private model: Qwen3Model;
  private config: GRPOTrainerConfig;
  private rewardFn?: RewardFn;
  private currentEpoch: number = 0;
  private currentStep: number = 0;
  /** Original model path (for tokenizer files when saving checkpoints) */
  private originalModelPath?: string;

  // TUI state
  private paused: boolean = false;
  private stopRequested: boolean = false;
  private stdinInterface?: import('readline').Interface;
  private logger: TrainingLogger;

  /**
   * Create a new GRPO trainer from a model
   *
   * @param model - Pre-loaded Qwen3 model
   * @param config - Training configuration
   */
  constructor(model: Qwen3Model, config: Partial<GRPOTrainerConfig> = {}, logger?: TrainingLogger) {
    // Auto-detect TUI mode from environment variable (set by mlx-train TUI)
    const tuiModeFromEnv = process.env.MLX_TUI_MODE === '1';
    if (tuiModeFromEnv && config.tuiMode === undefined) {
      config.tuiMode = true;
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
      maxNewTokens: this.config.maxNewTokens,
      temperature: this.config.temperature,
      topP: this.config.topP,
      topK: this.config.topK,
      repetitionPenalty: this.config.repetitionPenalty,
    };

    this.engine = new GrpoTrainingEngine(model, engineConfig);

    // Setup stdin handler if TUI mode
    if (this.config.tuiMode) {
      this.setupStdinHandler();
    }
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
        // Unknown command, ignore
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
  static async create(config: GRPOTrainerConfig): Promise<GRPOTrainer> {
    if (!config.modelPath) {
      throw new Error('modelPath is required when using GRPOTrainer.create()');
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

    // Validate checkpoint weights if resuming from checkpoint
    if (resumedState && modelPath !== config.modelPath) {
      logger.info('Validating checkpoint weights...');
      const isHealthy = await GRPOTrainer.validateModelHealth(model);
      if (!isHealthy) {
        throw new Error(
          `Checkpoint ${modelPath} appears corrupted (model generates only whitespace/newlines). ` +
            `This usually happens when NaN gradients accumulate during training. ` +
            `Please use an earlier checkpoint or start fresh training.`,
        );
      }
      logger.info('Checkpoint weights validated');
    }

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
   * Validate model health by checking if it generates meaningful output
   *
   * A corrupted model (from NaN gradient accumulation) typically generates
   * only whitespace/newlines.
   *
   * @param model - The model to validate
   * @returns true if model appears healthy, false if corrupted
   */
  static async validateModelHealth(model: Qwen3Model): Promise<boolean> {
    try {
      // Generate a short completion with a simple prompt
      const testPrompt: ChatMessage = { role: 'user', content: 'Hello, please respond with a single word:' };
      const result = await model.generate([testPrompt], {
        maxNewTokens: 20,
        temperature: 0.7,
      });

      if (!result || !result.text) {
        return false;
      }

      const output = result.text;

      // Check if output is only whitespace/newlines (sign of corruption)
      const trimmed = output.trim();
      if (trimmed.length === 0) {
        return false;
      }

      // Check if output is mostly newlines (another sign of corruption)
      const newlineCount = (output.match(/\n/g) || []).length;
      const nonNewlineChars = output.replace(/\n/g, '').length;
      if (newlineCount > 10 && nonNewlineChars < 5) {
        return false;
      }

      return true;
    } catch {
      // If generation fails, model might be corrupted
      return false;
    }
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
  setRewardFunction(fn: RewardFn): void {
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
        },
        tokenCounts: [],
      };
    }

    // Call the native engine to generate completions with full data
    const nativeResult = await this.engine.generateBatchForTraining(prompts);

    return {
      completionTexts: nativeResult.completionTexts,
      nativeResult,
      tokenCounts: nativeResult.completionLengths,
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
   * Score generations using the configured reward function (legacy API)
   *
   * @param prompts - Array of chat conversations
   * @param completions - Generated completion texts
   * @param answers - Expected answers (for accuracy rewards)
   * @param groupSize - Number of completions per prompt
   * @returns Promise<Float32Array> of reward scores
   */
  async scoreGenerations(
    prompts: ChatMessage[][],
    completions: string[],
    answers: (string | null)[],
    groupSize?: number,
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

    // Convert prompts to text format
    const promptTexts = prompts.flatMap((msgs) =>
      Array(effectiveGroupSize).fill(msgs.map((m) => `${m.role}: ${m.content}`).join('\n')),
    );

    // Expand answers to match completions
    const expandedAnswers =
      answers.length > 0
        ? prompts.flatMap((_, i) => Array(effectiveGroupSize).fill(answers[i] ?? null))
        : completions.map(() => null);

    let rewards: number[] | Float32Array;

    if (this.rewardFn) {
      rewards = await this.rewardFn(promptTexts, completions, expandedAnswers);
    } else {
      rewards = this.scoreCompletions(promptTexts, completions);
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
   * @param answers - Expected answers (for legacy reward functions)
   * @returns Training step metrics
   */
  async trainStep(prompts: ChatMessage[][], answers: (string | null)[]): Promise<TrainStepMetrics> {
    // Generate completions with full data (tokens, logprobs)
    const result = await this.generateBatch(prompts);
    const completions = result.completionTexts;

    // Compute rewards on the generated completions
    const rewards = await this.scoreGenerations(prompts, completions, answers);

    // Train using the SAME completions that were scored
    // This fixes the double-generation bug where rewards were computed on
    // different completions than what was used for training
    const metrics = await this.engine.trainStepWithGenerations(prompts, Array.from(rewards), result.nativeResult);

    this.currentStep++;

    return {
      ...metrics,
      epoch: this.currentEpoch,
    };
  }

  /**
   * Run a complete training step with automatic reward computation
   *
   * 1. Generates completions with full token/logprob data
   * 2. Computes rewards (using built-in or custom function)
   * 3. Performs training update using the SAME completions
   *
   * @param prompts - Array of chat conversations
   * @param answers - Expected answers (for legacy reward functions)
   * @returns Training metrics and generated completions
   */
  async trainStepAuto(
    prompts: ChatMessage[][],
    answers: (string | null)[] = [],
  ): Promise<{ metrics: TrainStepMetrics; completions: string[]; rewards: number[] }> {
    // Generate completions with full data
    const result = await this.generateBatch(prompts);
    const completions = result.completionTexts;

    // Compute rewards
    const rewardsArray = await this.scoreGenerations(prompts, completions, answers);
    const rewards = Array.from(rewardsArray);

    // Train using the SAME completions that were scored
    const metrics = await this.engine.trainStepWithGenerations(prompts, rewards, result.nativeResult);

    this.currentStep++;

    return {
      metrics: { ...metrics, epoch: this.currentEpoch },
      completions,
      rewards,
    };
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

    // Calculate total steps per epoch for resumption
    const stepsPerEpoch = Math.ceil(dataset.length / batchSize);

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
        const answers = batch.map((ex) => ex.answer ?? null);

        // Run training step with auto reward computation
        const { metrics, completions, rewards } = await this.trainStepAuto(prompts, answers);

        // Log step metrics (logger handles TUI/console mode internally)
        this.logger.step(metrics, Math.floor(i / batchSize), stepsPerEpoch);

        // Report all generation samples to TUI (TUI handles display filtering)
        // In console mode, logger.generation() is a no-op
        const groupSize = this.config.groupSize ?? 4;
        for (let j = 0; j < completions.length; j++) {
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
            tokens: this.config.maxNewTokens ?? 256,
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

    // Validate model health before saving (skip for emergency saves)
    const isEmergency = name?.startsWith('emergency-');
    if (!isEmergency) {
      const isHealthy = await GRPOTrainer.validateModelHealth(this.model);
      if (!isHealthy) {
        this.logger.error(
          `[CHECKPOINT] Model health check FAILED at step ${this.currentStep}. ` +
            `Skipping checkpoint save to prevent corruption. ` +
            `Consider reducing learning rate or checking training data.`,
        );
        return ''; // Return empty string to indicate save was skipped
      }
    } else {
      this.logger.warn(`[CHECKPOINT] Saving emergency checkpoint without health validation (step ${this.currentStep})`);
    }

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
 * Legacy type alias for backward compatibility
 */
export type GRPOConfig = GRPOTrainerConfig;

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
