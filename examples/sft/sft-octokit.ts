/**
 * SFT Training Script for Octokit.js API Knowledge
 *
 * This script fine-tunes a Qwen3 model on Octokit API documentation
 * to prepare it for GRPO training on GitHub tool use tasks.
 *
 * The training uses:
 * - Cross-entropy loss on assistant responses
 * - Completion-only training (only train on completions, not prompts)
 * - Label smoothing for better generalization
 *
 * Usage:
 *   yarn oxnode examples/sft/sft-octokit.ts [options]
 *
 * Options:
 *   --model-path <path>     Path to base model (default: .cache/models/qwen3-0.6b-mlx-bf16)
 *   --dataset <path>        Path to JSONL dataset (default: examples/sft/sft-octokit-dataset.jsonl)
 *   --output-dir <path>     Output directory (default: outputs/sft-octokit)
 *   --num-epochs <N>        Training epochs (default: 5)
 *   --learning-rate <N>     Learning rate (default: 2e-5)
 *   --batch-size <N>        Batch size (default: 4)
 *   --resume                Resume from latest checkpoint
 *   --dry-run               Show config without training
 *
 * Examples:
 *   # Basic training with defaults
 *   yarn oxnode examples/sft/sft-octokit.ts
 *
 *   # Custom configuration
 *   yarn oxnode examples/sft/sft-octokit.ts --num-epochs 5 --learning-rate 3e-5
 *
 *   # Resume interrupted training
 *   yarn oxnode examples/sft/sft-octokit.ts --resume
 *
 *   # Preview without training
 *   yarn oxnode examples/sft/sft-octokit.ts --dry-run
 */

import { parseArgs } from 'node:util';
import { resolve, join, parse as parsePath } from 'node:path';
import { existsSync, readFileSync } from 'node:fs';
import { SFTTrainer, type SFTTrainerConfig, createTrainingLogger } from '@mlx-node/trl';

// =============================================================================
// Configuration
// =============================================================================

const DEFAULT_MODEL_PATH = '.cache/models/qwen3-0.6b-mlx-bf16';
const DEFAULT_DATASET_PATH = 'examples/sft/github-get-pr.jsonl';
const DEFAULT_OUTPUT_DIR = 'outputs/sft-octokit';

/**
 * Default SFT hyperparameters for Octokit API training
 *
 * These are tuned for:
 * - Small model (1.7B parameters)
 * - API documentation learning (structured Q&A)
 * - Preparing for GRPO training
 */
function getDefaultConfig(): Partial<SFTTrainerConfig> {
  return {
    // Lower LR for stable factual knowledge learning
    learning_rate: 2e-5,

    // Batch size that fits Metal GPU memory
    batch_size: 4,

    // Gradient accumulation for effective batch size of 32
    gradient_accumulation_steps: 8,

    // More epochs for better knowledge internalization
    num_epochs: 5,

    // Regularization
    weight_decay: 0.01,

    // Gradient clipping for stability
    max_grad_norm: 1.0,

    // Label smoothing reduces overconfidence
    label_smoothing: 0.1,

    // Only train on assistant responses (not system/user prompts)
    completion_only: true,

    // Room for prompts + detailed answers
    max_seq_length: 2048,

    // Logging frequency
    logging_steps: 10,

    // Checkpoint frequency
    save_steps: 50,

    // Keep last N checkpoints
    max_checkpoints: 3,

    // Reproducibility
    seed: 42,

    // Enable JSONL logging
    log_jsonl: true,
  };
}

// =============================================================================
// CLI Parsing
// =============================================================================

interface CliArgs {
  modelPath: string;
  dataset: string;
  outputDir: string;
  numEpochs: number;
  learningRate: number;
  batchSize: number;
  resume: boolean;
  dryRun: boolean;
  runName: string;
}

function parseCliArgs(): CliArgs {
  const { values } = parseArgs({
    options: {
      'model-path': {
        type: 'string',
        short: 'm',
        default: DEFAULT_MODEL_PATH,
      },
      dataset: {
        type: 'string',
        short: 'd',
        default: DEFAULT_DATASET_PATH,
      },
      'output-dir': {
        type: 'string',
        short: 'o',
        default: DEFAULT_OUTPUT_DIR,
      },
      'num-epochs': {
        type: 'string',
        short: 'e',
        default: '5',
      },
      'learning-rate': {
        type: 'string',
        short: 'l',
        default: '2e-5',
      },
      'batch-size': {
        type: 'string',
        short: 'b',
        default: '4',
      },
      resume: {
        type: 'boolean',
        short: 'r',
        default: false,
      },
      'dry-run': {
        type: 'boolean',
        default: false,
      },
      'run-name': {
        type: 'string',
        short: 'n',
        default: 'sft-octokit',
      },
    },
  });

  const cwd = process.cwd();

  return {
    modelPath: resolve(cwd, values['model-path']!),
    dataset: resolve(cwd, values['dataset']!),
    outputDir: resolve(cwd, values['output-dir']!),
    numEpochs: parseInt(values['num-epochs']!, 10),
    learningRate: parseFloat(values['learning-rate']!),
    batchSize: parseInt(values['batch-size']!, 10),
    resume: values['resume'] ?? false,
    dryRun: values['dry-run'] ?? false,
    runName: values['run-name']!,
  };
}

// =============================================================================
// Validation
// =============================================================================

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  datasetLineCount?: number;
  modelSize?: string;
}

function validateInputs(args: CliArgs): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  let datasetLineCount: number | undefined;
  let modelSize: string | undefined;

  // Check model exists
  if (!existsSync(args.modelPath)) {
    errors.push(`Model not found at: ${args.modelPath}`);
    errors.push('  Download with: yarn download:qwen3');
  } else {
    // Get model size if exists
    try {
      const modelConfig = join(args.modelPath, 'config.json');
      if (existsSync(modelConfig)) {
        modelSize = parsePath(args.modelPath).base;
      }
    } catch {
      // Ignore
    }
  }

  // Check dataset exists
  if (!existsSync(args.dataset)) {
    errors.push(`Dataset not found at: ${args.dataset}`);
    errors.push('  Generate with: yarn oxnode examples/grpo/generate-sft-dataset.ts');
  } else {
    // Count lines for info
    try {
      const content = readFileSync(args.dataset, 'utf8');
      datasetLineCount = content.split('\n').filter((line: string) => line.trim()).length;
    } catch {
      warnings.push('Could not read dataset for line count');
    }
  }

  // Validate hyperparameters
  if (args.numEpochs < 1) {
    errors.push('num-epochs must be >= 1');
  }
  if (args.learningRate <= 0 || args.learningRate > 1) {
    errors.push('learning-rate must be between 0 and 1');
  }
  if (args.batchSize < 1) {
    errors.push('batch-size must be >= 1');
  }

  // Warnings
  if (args.numEpochs > 10) {
    warnings.push(`High epoch count (${args.numEpochs}) may cause overfitting`);
  }
  if (args.learningRate > 1e-4) {
    warnings.push(`High learning rate (${args.learningRate}) may cause instability`);
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    datasetLineCount,
    modelSize,
  };
}

const args = parseCliArgs();

// Create logger (auto-detects TUI mode from MLX_TUI_MODE env var)
const logger = createTrainingLogger({
  outputDir: args.outputDir,
  runName: args.runName,
  logConsole: process.env.MLX_TUI_MODE !== '1',
  logJsonl: true,
});

// Banner
logger.banner(
  '',
  '======================================================',
  '   SFT Training: Octokit.js API Knowledge             ',
  '   Teaching the model GitHub API documentation        ',
  '======================================================',
  '',
);

// Validate inputs
const validation = validateInputs(args);

// Print errors
if (validation.errors.length > 0) {
  for (const error of validation.errors) {
    logger.error(error);
  }
  process.exit(1);
}

// Print warnings
for (const warning of validation.warnings) {
  logger.warn(warning);
}

// Build configuration
const config: Partial<SFTTrainerConfig> = {
  ...getDefaultConfig(),
  model_name: args.modelPath,
  output_dir: args.outputDir,
  run_name: args.runName,
  num_epochs: args.numEpochs,
  learning_rate: args.learningRate,
  batch_size: args.batchSize,
  resume_from_checkpoint: args.resume ? 'latest' : undefined,
};

// Calculate effective batch size
const effectiveBatchSize = args.batchSize * (config.gradient_accumulation_steps ?? 1);

// Print configuration
logger.banner(
  'Configuration:',
  '-----------------------------------------------------',
  `Model: ${validation.modelSize ?? args.modelPath}`,
  `Dataset: ${args.dataset}`,
  `  Examples: ${validation.datasetLineCount ?? 'unknown'}`,
  `Output: ${args.outputDir}`,
  '',
  'Hyperparameters:',
  `  Epochs: ${args.numEpochs}`,
  `  Learning rate: ${args.learningRate}`,
  `  Batch size: ${args.batchSize} (effective: ${effectiveBatchSize})`,
  `  Gradient accumulation: ${config.gradient_accumulation_steps}`,
  `  Weight decay: ${config.weight_decay}`,
  `  Label smoothing: ${config.label_smoothing}`,
  `  Max sequence length: ${config.max_seq_length}`,
  '',
  'Training mode:',
  `  Completion only: ${config.completion_only ? 'yes (train on assistant responses only)' : 'no'}`,
  `  Resume: ${args.resume ? 'yes' : 'no'}`,
  '-----------------------------------------------------',
  '',
);

// Dry run - just show config
if (args.dryRun) {
  logger.info('Dry run mode - showing configuration without training');
  logger.info('');
  logger.info('Full config object:');
  console.log(JSON.stringify(config, null, 2));
  process.exit(0);
}

// Estimate training time
if (validation.datasetLineCount) {
  const stepsPerEpoch = Math.ceil(validation.datasetLineCount / args.batchSize);
  const totalSteps = stepsPerEpoch * args.numEpochs;
  const estimatedSecsPerStep = 0.5; // Conservative estimate
  const estimatedMins = Math.ceil((totalSteps * estimatedSecsPerStep) / 60);
  logger.info(`Estimated training time: ~${estimatedMins} minutes (${totalSteps} steps)`);
  logger.info('');
}

// Create trainer
logger.status('loading', 'Creating SFT trainer...');
const trainer = await SFTTrainer.create(config);

// Start training
logger.info('Starting SFT training...');
logger.info('');

try {
  await trainer.train(args.dataset);

  // Success banner
  const finalModelPath = join(args.outputDir, 'final');
  logger.banner(
    '',
    '======================================================',
    '   SFT Training Complete!                             ',
    '======================================================',
    '',
    `Final model saved to: ${finalModelPath}`,
    '',
    'Next steps:',
    `  1. Test the model: yarn oxnode examples/sft/test-sft-model.ts --model ${finalModelPath}`,
    `  2. Run GRPO training: yarn oxnode examples/grpo/train-github-tool.ts --model-path ${finalModelPath}`,
    `  3. Check logs: cat ${join(args.outputDir, `${args.runName}.jsonl`)}`,
    '',
    '======================================================',
    '',
  );
} catch (error) {
  logger.error(`Training failed: ${error}`);
  if (error instanceof Error) {
    logger.error(`Details: ${error.message}`);
    if (error.stack) {
      logger.error(`Stack: ${error.stack}`);
    }
  }
  process.exitCode = 1;
}
