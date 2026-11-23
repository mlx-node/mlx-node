import { Qwen3Model, Qwen35Model, Qwen35MoeModel } from '@mlx-node/core';
import type { TrainableModel } from '@mlx-node/lm';

import type { TrainingLogger } from './training-logger.js';

export interface TrainingEngineFactory<TConfig, TEngine> {
  new (model: Qwen3Model, config: TConfig): TEngine;
  fromQwen35(model: Qwen35Model, config: TConfig): TEngine;
  fromQwen35Moe(model: Qwen35MoeModel, config: TConfig): TEngine;
}

/**
 * Instantiate a native training engine (GRPO or SFT) by dispatching on the
 * concrete trainable model class.
 */
export function createTrainingEngine<TConfig, TEngine>(
  EngineClass: TrainingEngineFactory<TConfig, TEngine>,
  model: TrainableModel,
  config: TConfig,
): TEngine {
  if (model instanceof Qwen35Model) {
    return EngineClass.fromQwen35(model, config);
  }
  if (model instanceof Qwen35MoeModel) {
    return EngineClass.fromQwen35Moe(model, config);
  }
  if (model instanceof Qwen3Model) {
    return new EngineClass(model, config);
  }
  throw new Error(`Unsupported model type: ${(model as object).constructor?.name ?? typeof model}`);
}

export interface SignalHandlerOptions {
  logger: Pick<TrainingLogger, 'warn'>;
  /** Invoked synchronously before shutdown work (e.g. to set `stopRequested = true`). */
  onStop: () => void;
  /** Async cleanup or emergency-checkpoint work to run before exiting. */
  onShutdown: () => Promise<void>;
  /** Error message logged via `console.error` when `onShutdown` throws. */
  errorMessage: string;
  /** Also catch `uncaughtException` and `unhandledRejection` (default: false). */
  handleUncaught?: boolean;
}

/**
 * Install OS signal handlers for graceful trainer shutdown.
 */
export function installTrainerSignalHandlers(options: SignalHandlerOptions): void {
  const { logger, onStop, onShutdown, errorMessage, handleUncaught = false } = options;

  const gracefulShutdown = async (signal: string) => {
    logger.warn(`Received ${signal}, initiating graceful shutdown...`);
    onStop();

    try {
      await onShutdown();
    } catch (e) {
      console.error(errorMessage, e);
    }

    process.exit(0);
  };

  process.on('SIGTERM', () => {
    gracefulShutdown('SIGTERM').catch(console.error);
  });

  process.on('SIGINT', () => {
    gracefulShutdown('SIGINT').catch(console.error);
  });

  if (handleUncaught) {
    process.on('uncaughtException', (err) => {
      console.error('Uncaught exception:', err);
      gracefulShutdown('uncaughtException').catch(console.error);
    });

    process.on('unhandledRejection', (reason) => {
      console.error('Unhandled rejection:', reason);
      gracefulShutdown('unhandledRejection').catch(console.error);
    });
  }
}
