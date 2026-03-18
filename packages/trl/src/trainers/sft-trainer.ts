/**
 * SFT (Supervised Fine-Tuning) Trainer
 *
 * This module provides a Rust-native SFT training engine for training
 * models on fixed prompt-completion pairs using cross-entropy loss.
 *
 * ## Key Features
 * - Training loop runs in Rust (eliminates FFI overhead)
 * - Cross-entropy loss with completion masking (ignore_index=-100)
 * - Label smoothing support
 * - Gradient accumulation and clipping
 * - High-level train() method for full training runs
 * - Low-level trainStep() for custom training loops
 *
 * ## Usage
 * ```typescript
 * const trainer = await SFTTrainer.create({
 *   modelPath: './model',
 *   learningRate: 2e-5,
 *   numEpochs: 3,
 * });
 * await trainer.train(dataset);
 * ```
 */

import { existsSync, mkdirSync, writeFileSync, readFileSync, readdirSync, copyFileSync, rmSync } from 'node:fs';
import { join, parse } from 'node:path';
import * as readline from 'node:readline';

import {
  SftTrainingEngine,
  Qwen3Model,
  Qwen35Model,
  Qwen35MoeModel,
  Qwen3Tokenizer,
  MxArray,
  type SftEngineConfig,
  type SftStepMetrics,
  type SftEpochMetrics,
} from '@mlx-node/core';
import { loadModel, type TrainableModel } from '@mlx-node/lm';

import type { SFTTrainerConfig } from './sft-config';
import { getDefaultSFTConfig, mergeSFTConfig } from './sft-config';
import { SFTDataset, loadSFTDataset, type SFTBatch } from '../data/sft-dataset';
import { createTrainingLogger, type TrainingLogger } from './training-logger';

// Re-export types
export { SftTrainingEngine } from '@mlx-node/core';
export type { SftEngineConfig, SftStepMetrics, SftEpochMetrics } from '@mlx-node/core';

/**
 * Training state saved with checkpoints for resumption
 */
export interface SFTTrainingState {
  step: number;
  epoch: number;
  timestamp: string;
  trainerType: 'sft';
}

/**
 * Training step result
 */
export interface SFTTrainStepResult {
  /** Step metrics */
  metrics: SftStepMetrics;
  /** Current epoch */
  epoch: number;
}

/**
 * SFT Trainer - Rust-Native Training Engine
 *
 * Provides a TypeScript-friendly interface to the Rust SFT training engine.
 */
export class SFTTrainer {
  private engine: SftTrainingEngine;
  private model: TrainableModel;
  private tokenizer: Qwen3Tokenizer;
  private config: SFTTrainerConfig;
  private currentEpoch: number = 0;
  private currentStep: number = 0;
  /** Original model path (for tokenizer files when saving checkpoints) */
  private originalModelPath?: string;

  // TUI state
  private paused: boolean = false;
  private stopRequested: boolean = false;
  private stdinInterface?: readline.Interface;
  private logger: TrainingLogger;
  private sampleDisplayMode: 'all' | 'best_worst' | 'random' = 'all';
  private signalHandlersInstalled: boolean = false;

  /**
   * Create a new SFT trainer from a model
   *
   * @param model - Pre-loaded Qwen3 model
   * @param tokenizer - Pre-loaded tokenizer
   * @param config - Training configuration
   * @param logger - Optional custom logger
   */
  constructor(
    model: TrainableModel,
    tokenizer: Qwen3Tokenizer,
    config: Partial<SFTTrainerConfig> = {},
    logger?: TrainingLogger,
  ) {
    // Auto-detect TUI mode from environment variable
    const tuiModeFromEnv = process.env.MLX_TUI_MODE === '1';
    if (tuiModeFromEnv && config.tuiMode === undefined) {
      config.tuiMode = true;
    }

    this.config = mergeSFTConfig(getDefaultSFTConfig(), config);
    this.model = model;
    this.tokenizer = tokenizer;

    // Create or use provided logger
    this.logger =
      logger ??
      createTrainingLogger({
        logConsole: !this.config.tuiMode,
        logJsonl: this.config.logJsonl,
        outputDir: this.config.outputDir,
        runName: this.config.runName,
        logInterval: this.config.loggingSteps,
      });

    // Convert to native config
    const engineConfig: SftEngineConfig = {
      learningRate: this.config.learningRate,
      gradientAccumulationSteps: this.config.gradientAccumulationSteps,
      gradientClipNorm: this.config.maxGradNorm,
      weightDecay: this.config.weightDecay,
      labelSmoothing: this.config.labelSmoothing,
      gradientCheckpointing: this.config.gradientCheckpointing,
    };

    if (model instanceof Qwen35Model) {
      this.engine = SftTrainingEngine.fromQwen35(model, engineConfig);
    } else if (model instanceof Qwen35MoeModel) {
      this.engine = SftTrainingEngine.fromQwen35Moe(model, engineConfig);
    } else if (model instanceof Qwen3Model) {
      this.engine = new SftTrainingEngine(model, engineConfig);
    } else {
      throw new Error(`Unsupported model type: ${(model as object).constructor?.name ?? typeof model}`);
    }

    // Setup stdin handler if TUI mode
    if (this.config.tuiMode) {
      this.setupStdinHandler();
    }
  }

  /**
   * Setup OS signal handlers for graceful shutdown on interrupt.
   * Saves an emergency checkpoint before exiting to prevent progress loss.
   */
  private setupSignalHandlers(): void {
    if (this.signalHandlersInstalled) return;
    this.signalHandlersInstalled = true;

    const gracefulShutdown = async (signal: string) => {
      this.logger.warn(`Received ${signal}, initiating graceful shutdown...`);
      this.stopRequested = true;

      try {
        if (this.config.outputDir && this.currentStep > 0) {
          this.logger.info(`Saving emergency checkpoint at step ${this.currentStep}...`);
          await this.saveCheckpoint(`emergency-checkpoint-${this.currentStep}`);
          this.logger.info('Emergency checkpoint saved.');
        }
      } catch (e) {
        console.error('Failed to save emergency checkpoint:', e);
      }

      process.exit(0);
    };

    process.on('SIGTERM', () => {
      gracefulShutdown('SIGTERM').catch(console.error);
    });

    process.on('SIGINT', () => {
      gracefulShutdown('SIGINT').catch(console.error);
    });
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
   * Wait for resume if paused
   */
  private async waitForResume(): Promise<void> {
    while (this.paused && !this.stopRequested) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  /**
   * Create a trainer by loading a model from disk
   *
   * @param config - Configuration including modelPath
   * @returns Promise<SFTTrainer>
   */
  static async create(config: Partial<SFTTrainerConfig>): Promise<SFTTrainer> {
    if (!config.modelName) {
      throw new Error('modelName is required when using SFTTrainer.create()');
    }

    // Create logger early
    const logger = createTrainingLogger({
      logConsole: !config.tuiMode,
      logJsonl: config.logJsonl ?? true,
      outputDir: config.outputDir,
      runName: config.runName,
      logInterval: config.loggingSteps ?? 10,
    });

    let modelPath = config.modelName;
    let resumedState: SFTTrainingState | null = null;

    // Handle checkpoint resumption
    if (config.resumeFromCheckpoint) {
      const checkpointPath =
        config.resumeFromCheckpoint === 'latest'
          ? SFTTrainer.findLatestCheckpoint(config.outputDir)
          : config.resumeFromCheckpoint;

      if (checkpointPath) {
        const statePath = join(checkpointPath, 'training_state.json');
        if (existsSync(statePath)) {
          resumedState = JSON.parse(readFileSync(statePath, 'utf-8'));
          logger.info(
            `Resuming from checkpoint: ${checkpointPath} (step ${resumedState?.step}, epoch ${resumedState?.epoch})`,
          );
        }
        modelPath = checkpointPath;
      } else if (config.resumeFromCheckpoint === 'latest') {
        logger.info('No checkpoint found, starting fresh training');
      }
    }

    // Get model name for display
    const modelName = parse(modelPath).base || 'Unknown';
    logger.status('loading', `Loading ${modelName}...`);

    // Load model (auto-detects architecture from config.json)
    const model = await loadModel(modelPath);

    const tokenizer = await Qwen3Tokenizer.fromPretrained(join(modelPath, 'tokenizer.json'));

    logger.status('loading', `${modelName} loaded (${model.constructor.name})`);

    // Create trainer
    const trainer = new SFTTrainer(model, tokenizer, config, logger);
    trainer.originalModelPath = config.modelName;

    // Restore training state if resuming
    if (resumedState) {
      trainer.currentStep = resumedState.step;
      trainer.currentEpoch = resumedState.epoch;
      // Also restore engine state to sync step/epoch accounting
      trainer.engine.restoreState(resumedState.step, resumedState.epoch);
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
   * Run a single training step
   *
   * @param batch - Tokenized batch with input_ids and labels
   * @returns Training step metrics
   */
  async trainStep(batch: SFTBatch): Promise<SFTTrainStepResult> {
    // Convert Int32Array to MxArray
    const inputIds = MxArray.fromInt32(batch.inputIds, BigInt64Array.from(batch.shape.map(BigInt)));
    const labels = MxArray.fromInt32(batch.labels, BigInt64Array.from(batch.shape.map(BigInt)));

    // Call native engine
    const metrics = await this.engine.trainStep(inputIds, labels);

    // Sync step with engine when gradients are applied (fixes gradient accumulation accounting)
    // Note: metrics.step is i64 from Rust; JS number may lose precision beyond 2^53-1,
    // but such step counts are unrealistic for any practical training run.
    if (metrics.gradientsApplied) {
      this.currentStep = Number(metrics.step);
    }

    return {
      metrics,
      epoch: this.currentEpoch,
    };
  }

  /**
   * Run a full training loop over a dataset
   *
   * @param dataset - SFT dataset or path to JSONL file
   */
  async train(dataset: SFTDataset | string): Promise<void> {
    // Load dataset if path provided
    let sftDataset: SFTDataset;
    if (typeof dataset === 'string') {
      sftDataset = await loadSFTDataset(dataset, this.tokenizer, {
        maxSeqLength: this.config.maxSeqLength,
        completionOnly: this.config.completionOnly,
        seed: this.config.seed,
        limit: this.config.maxTrainSamples > 0 ? this.config.maxTrainSamples : undefined,
      });
    } else {
      sftDataset = dataset;
    }

    if (sftDataset.length === 0) {
      return;
    }

    // Setup signal handlers for crash recovery
    this.setupSignalHandlers();

    const numEpochs = this.config.numEpochs;
    const batchSize = this.config.batchSize;
    const saveInterval = this.config.saveSteps;

    // Create output directory
    if (this.config.outputDir && !existsSync(this.config.outputDir)) {
      mkdirSync(this.config.outputDir, { recursive: true });
    }

    // Calculate steps per epoch (in batches)
    const stepsPerEpoch = sftDataset.numBatches(batchSize);

    // Compute resume position (all logic centralized in Rust)
    const resumePos = this.engine.computeResumePosition(stepsPerEpoch);
    const effectiveStartEpoch = resumePos.startEpoch;
    const effectiveStartBatchIdx = resumePos.startBatchIdx;

    // Get model name
    const modelName =
      (this.originalModelPath ? parse(this.originalModelPath).base : null) ??
      (this.config.modelName ? parse(this.config.modelName).base : null) ??
      'Unknown';

    // Log training start
    this.logger.init(
      modelName,
      {
        trainingType: 'sft',
        numEpochs,
        batchSize,
        groupSize: 1, // SFT doesn't use groups
        learningRate: this.config.learningRate,
      },
      sftDataset.length,
    );

    if (this.currentStep > 0) {
      if (resumePos.isEpochBoundary) {
        this.logger.info(`Resuming at epoch boundary, advancing to epoch ${effectiveStartEpoch + 1}`);
      } else {
        this.logger.info(
          `Resuming from step ${this.currentStep} (epoch ${effectiveStartEpoch + 1}, batch ${effectiveStartBatchIdx + 1}/${stepsPerEpoch})`,
        );
      }
    }

    for (let epoch = effectiveStartEpoch; epoch < numEpochs; epoch++) {
      if (this.stopRequested) break;

      this.currentEpoch = epoch;
      this.engine.startEpoch(epoch);
      const epochStartTime = Date.now();

      // Use epoch-based shuffle (deterministic, reproducible via seed + epoch)
      sftDataset.shuffleForEpoch(epoch);

      // Log epoch start
      this.logger.epochStart(epoch, numEpochs, stepsPerEpoch);

      // Determine batch start position for this epoch
      const batchStart = epoch === effectiveStartEpoch ? effectiveStartBatchIdx : 0;

      // Iterate through batches
      let batchIdx = 0;
      for await (const batch of sftDataset.batches(batchSize)) {
        if (this.stopRequested) break;

        // Skip batches if resuming mid-epoch
        if (batchIdx < batchStart) {
          batchIdx++;
          continue;
        }

        // Wait if paused
        if (this.paused) {
          await this.waitForResume();
          if (this.stopRequested) break;
        }

        // Run training step
        const { metrics } = await this.trainStep(batch);

        // Log step metrics (only when gradients are applied to avoid duplicate logs during accumulation)
        if (metrics.gradientsApplied) {
          this.logger.step(
            {
              step: this.currentStep,
              loss: metrics.loss,
              totalTokens: metrics.totalTokens,
              // SFT-specific metrics (no reward/advantage!)
              perplexity: Math.exp(metrics.loss),
              // Token accuracy is not currently tracked in the SFT engine
              // Could be added later if the Rust engine exposes it
              trainingTimeMs: metrics.trainingTimeMs,
            },
            batchIdx,
            stepsPerEpoch,
          );

          // Save checkpoint periodically
          if (this.config.outputDir && this.currentStep > 0 && this.currentStep % saveInterval === 0) {
            const path = await this.saveCheckpoint();
            if (path) {
              this.logger.checkpoint(path, this.currentStep);
            }
          }
        }

        // Check for emergency checkpoint
        if (this.config.outputDir && this.engine.needsEmergencySave()) {
          this.logger.warn(`[EMERGENCY] Saving emergency checkpoint at step ${this.currentStep} due to NaN gradients`);
          await this.saveCheckpoint(`emergency-checkpoint-${this.currentStep}`);
          this.engine.clearEmergencySave();
        }

        batchIdx++;
      }

      // Flush any remaining accumulated gradients (TRL parity)
      const flushed = this.engine.flushGradients();
      if (flushed) {
        this.currentStep = this.engine.getStep();

        // Check if flush step aligns with save interval
        if (this.config.outputDir && this.currentStep > 0 && this.currentStep % saveInterval === 0) {
          const path = await this.saveCheckpoint();
          if (path) {
            this.logger.checkpoint(path, this.currentStep);
          }
        }
      }

      const epochEndTime = Date.now();
      const epochTimeSecs = (epochEndTime - epochStartTime) / 1000;
      this.engine.endEpoch(epochTimeSecs);

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

    // Cleanup
    if (this.stdinInterface) {
      this.stdinInterface.close();
    }
  }

  /**
   * Save a checkpoint with model weights and training state
   *
   * @param name - Checkpoint name (default: "checkpoint-{step}")
   * @returns Path to saved checkpoint
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
    const state: SFTTrainingState = {
      step: this.currentStep,
      epoch: this.currentEpoch,
      timestamp: new Date().toISOString(),
      trainerType: 'sft',
    };
    const statePath = join(checkpointPath, 'training_state.json');
    writeFileSync(statePath, JSON.stringify(state, null, 2));

    // Save model weights
    await this.model.saveModel(checkpointPath);

    // Copy tokenizer files
    const tokenizerSource = this.originalModelPath ?? this.config.modelName;
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

    // Clean up old checkpoints
    const maxCheckpoints = this.config.maxCheckpoints;
    if (maxCheckpoints > 0) {
      this.cleanupOldCheckpoints(outputDir, maxCheckpoints);
    }

    return checkpointPath;
  }

  /**
   * Remove old checkpoints, keeping only the most recent ones
   */
  private cleanupOldCheckpoints(outputDir: string, maxToKeep: number): void {
    try {
      const entries = readdirSync(outputDir, { withFileTypes: true });

      const checkpoints: { name: string; step: number }[] = [];
      for (const entry of entries) {
        if (!entry.isDirectory()) continue;
        if (entry.name === 'final' || entry.name.startsWith('emergency-')) continue;

        const match = entry.name.match(/^checkpoint-(\d+)$/);
        if (match) {
          checkpoints.push({
            name: entry.name,
            step: parseInt(match[1], 10),
          });
        }
      }

      checkpoints.sort((a, b) => b.step - a.step);

      if (checkpoints.length > maxToKeep) {
        const toRemove = checkpoints.slice(maxToKeep);
        for (const checkpoint of toRemove) {
          const checkpointPath = join(outputDir, checkpoint.name);
          rmSync(checkpointPath, { recursive: true, force: true });
          this.logger.debug(`Removed old checkpoint: ${checkpoint.name}`);
        }
      }
    } catch (error) {
      this.logger.warn(`Failed to cleanup old checkpoints: ${error as Error}`);
    }
  }

  /**
   * Get current training step
   */
  get step(): number {
    return this.engine.getStep();
  }

  /**
   * Get current epoch
   */
  get epoch(): number {
    return this.engine.getEpoch();
  }

  /**
   * Get the underlying model for inference
   */
  getModel(): TrainableModel {
    if (this.model instanceof Qwen35MoeModel) {
      return this.engine.getQwen35MoeModel();
    }
    if (this.model instanceof Qwen35Model) {
      return this.engine.getQwen35Model();
    }
    return this.engine.getModel();
  }

  /**
   * Get the tokenizer
   */
  getTokenizer(): Qwen3Tokenizer {
    return this.tokenizer;
  }
}
