/**
 * Simple GRPO Training Demo with Qwen3-0.6B
 *
 * This is a minimal training example that:
 * - Uses a pretrained Qwen3-0.6B model
 * - Trains on 50 GSM8K math problems
 * - Uses 2 generations per prompt (fast)
 * - Saves checkpoints every 25 steps
 * - Completes in ~15-20 minutes
 *
 * Perfect for:
 * - First-time users
 * - Quick testing
 * - Verifying setup
 *
 * Usage:
 *   node examples/grpo/train-simple.ts [options]
 *
 * Options:
 *   --model-path <path>     Path to model directory (default: .cache/models/qwen3-0.6b-mlx)
 *   --num-examples <n>      Number of examples to train on (default: 50)
 *   --output-dir <path>     Output directory (default: outputs/grpo-simple)
 *
 * Examples:
 *   node examples/grpo/train-simple.ts --num-examples 10
 *   node examples/grpo/train-simple.ts --model-path ./my-model --output-dir ./my-output
 */

import { parseArgs } from 'node:util';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { GRPOTrainer, type GRPOConfig, loadLocalGsm8kDataset, ALL_REWARD_FUNCTIONS } from '@mlx-node/trl';

const DEFAULT_MODEL_PATH = resolve(process.cwd(), '.cache', 'models', 'qwen3-0.6b-mlx-bf16');
const DEFAULT_NUM_EXAMPLES = 50;
const DEFAULT_OUTPUT_DIR = resolve(process.cwd(), 'outputs', 'grpo-simple');

// Combined reward function (accuracy + format)
// Uses the unified reward function signature: (prompts, completions, answers) => number[] | Float32Array
async function combinedReward(
  prompts: string[],
  completions: string[],
  answers: (string | null)[],
): Promise<Float32Array> {
  // Apply all reward functions and sum their scores
  const allScores = await Promise.all(ALL_REWARD_FUNCTIONS.map((fn) => fn(prompts, completions, answers)));

  // Sum scores for each completion
  const numCompletions = completions.length;
  const combinedScores = new Float32Array(numCompletions);

  for (const scores of allScores) {
    const scoresArray = scores instanceof Float32Array ? Array.from(scores) : scores;
    for (let i = 0; i < numCompletions; i++) {
      combinedScores[i] += scoresArray[i];
    }
  }

  return combinedScores;
}

async function main() {
  // Parse command-line arguments
  const { values } = parseArgs({
    options: {
      'model-path': {
        type: 'string',
        short: 'm',
        default: DEFAULT_MODEL_PATH,
      },
      'num-examples': {
        type: 'string',
        short: 'n',
        default: String(DEFAULT_NUM_EXAMPLES),
      },
      'output-dir': {
        type: 'string',
        short: 'o',
        default: DEFAULT_OUTPUT_DIR,
      },
    },
  });

  const modelPath = resolve(values['model-path']!);
  const numExamples = Number.parseInt(values['num-examples']!, 10);
  const outputDir = resolve(values['output-dir']!);

  console.log('╔════════════════════════════════════════════════════════╗');
  console.log('║     Simple GRPO Training Demo with Qwen3-0.6B         ║');
  console.log('╚════════════════════════════════════════════════════════╝\n');

  // Check if model exists
  if (!existsSync(modelPath)) {
    console.error(`❌ Model not found at: ${modelPath}\n`);
    console.error('Please download the model first:');
    console.error('   yarn download:qwen3\n');
    process.exitCode = 1;
    return;
  }

  console.log('Configuration:');
  console.log('─────────────────────────────────────────────────────');
  console.log(`Model: ${modelPath}`);
  console.log(`Training examples: ${numExamples}`);
  console.log(`Output directory: ${outputDir}`);
  console.log(`Group size: 2 generations per prompt`);
  console.log(`Learning rate: 1e-6`);
  console.log('─────────────────────────────────────────────────────\n');

  // Load dataset
  console.log('📚 Loading GSM8K dataset...\n');
  const examples = await loadLocalGsm8kDataset('train', {
    limit: numExamples,
    includeOneShot: true, // Include one-shot example in prompts
  });

  console.log(`✓ Loaded ${examples.length} training examples\n`);

  // Configure GRPO trainer
  const config: GRPOConfig = {
    // Model
    modelConfig: 'qwen3-0.6b',
    modelPath, // Load pretrained weights from disk

    // Training hyperparameters
    learningRate: 1e-6,
    numEpochs: 1, // Single pass through dataset
    batchSize: 1, // Process one prompt at a time
    gradientAccumulationSteps: 2, // Accumulate gradients over 2 batches

    // GRPO parameters
    groupSize: 2, // Generate 2 completions per prompt
    clipEpsilon: 0.2, // PPO-style clipping
    klCoef: 0.0, // No KL penalty
    advantageNormalization: true,

    // Generation parameters
    maxNewTokens: 256, // Max tokens per generation
    temperature: 0.7, // Moderate randomness
    topP: 0.95, // Nucleus sampling
    topK: 50, // Top-k sampling

    // Reward configuration
    rewardType: 'function',
    rewardFunction: combinedReward,

    // Loss configuration
    lossType: 'grpo',

    // Optimization
    weightDecay: 0.01,
    gradientClipNorm: 1.0,

    // Logging and checkpointing
    logInterval: 1, // Log every step
    saveInterval: 25, // Save checkpoint every 25 steps
    evalInterval: 25, // Evaluate every 25 steps
    outputDir,
    logConsole: true,
    logJsonl: true,
    runName: 'simple-demo',

    // Device
    device: 'metal',
  };

  console.log('🚀 Starting GRPO training...\n');
  console.log('This will take approximately 15-20 minutes.');
  console.log('Watch for checkpoints in the output directory.\n');
  console.log('═══════════════════════════════════════════════════════\n');

  // Create trainer and start training
  const trainer = await GRPOTrainer.create(config);

  try {
    await trainer.train(examples);

    console.log('\n═══════════════════════════════════════════════════════');
    console.log('✅ Training complete!\n');
    console.log('Results saved to:', outputDir);
    console.log('\nNext steps:');
    console.log('  1. Check training logs: cat', resolve(outputDir, 'log.jsonl'));
    console.log('  2. Inspect final checkpoint:', resolve(outputDir, 'final'));
    console.log('  3. Run full training demo: node examples/grpo/train-full.ts');
    console.log('═══════════════════════════════════════════════════════\n');
  } catch (error) {
    console.error('\n❌ Training failed:', error);
    if (error instanceof Error) {
      console.error('\nError details:', error.message);
      console.error('\nStack trace:', error.stack);
    }
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error('[train-simple] Fatal error:', error);
  process.exitCode = 1;
});
