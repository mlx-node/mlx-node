/**
 * Full GRPO Training Demo with Qwen3-0.6B
 *
 * This is a production-ready training example that:
 * - Loads configuration from TOML file
 * - Supports full GSM8K dataset
 * - Uses 8 generations per prompt (production setting)
 * - Saves checkpoints at regular intervals
 * - Provides detailed logging and metrics
 *
 * Perfect for:
 * - Production training runs
 * - Hyperparameter tuning
 * - Full dataset training
 *
 * Usage:
 *   node examples/grpo/train-full.ts [--config path/to/config.toml]
 *
 * Environment variables:
 *   CONFIG_PATH - Path to TOML config (overrides --config flag)
 *   MODEL_PATH - Path to model directory (overrides config)
 *   OUTPUT_DIR - Output directory (overrides config)
 */

import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { parseArgs } from 'node:util';

import { GRPOTrainer, type GRPOConfig, loadLocalGsm8kDataset } from '@mlx-node/trl';

import { loadConfigFromToml, printConfigSummary, type DatasetConfig } from './utils/load-config.js';

const DEFAULT_CONFIG_PATH = resolve(process.cwd(), 'examples', 'grpo', 'config.toml');

function printUsage(): void {
  console.log('Usage: node examples/grpo/train-full.ts [options]\n');
  console.log('Options:');
  console.log('  --config PATH    Path to TOML configuration file');
  console.log('  --help           Show this help message\n');
  console.log('Environment variables:');
  console.log('  CONFIG_PATH      Override config file path');
  console.log('  MODEL_PATH       Override model path from config');
  console.log('  OUTPUT_DIR       Override output directory from config\n');
  console.log('Examples:');
  console.log('  node examples/grpo/train-full.ts');
  console.log('  node examples/grpo/train-full.ts --config my-config.toml');
  console.log('  CONFIG_PATH=custom.toml node examples/grpo/train-full.ts\n');
}

async function main() {
  // Parse command-line arguments
  const { values: args } = parseArgs({
    options: {
      config: { type: 'string' },
      help: { type: 'boolean' },
    },
    allowPositionals: false,
  });

  // Show help if requested
  if (args.help) {
    printUsage();
    return;
  }

  // Determine config path
  const configPath = process.env.CONFIG_PATH ?? (args.config as string | undefined) ?? DEFAULT_CONFIG_PATH;

  console.log('╔════════════════════════════════════════════════════════╗');
  console.log('║       Full GRPO Training Demo with Qwen3-0.6B          ║');
  console.log('╚════════════════════════════════════════════════════════╝\n');

  // Check if config file exists
  if (!existsSync(configPath)) {
    console.error(`❌ Configuration file not found: ${configPath}\n`);
    console.error('Please create a configuration file or use the default:');
    console.error(`   cp examples/grpo/config.toml ${configPath}\n`);
    printUsage();
    process.exitCode = 1;
    return;
  }

  console.log(`📝 Loading configuration from: ${configPath}\n`);

  let grpoConfig: Partial<GRPOConfig>;
  let datasetConfig: DatasetConfig;

  try {
    ({ grpoConfig, datasetConfig } = loadConfigFromToml(configPath));
  } catch (error) {
    console.error('❌ Failed to load configuration:', error);
    if (error instanceof Error) {
      console.error('\nError details:', error.message);
    }
    process.exitCode = 1;
    return;
  }

  // Apply environment variable overrides
  if (process.env.MODEL_PATH) {
    console.log(`🔧 Overriding model path from env: ${process.env.MODEL_PATH}`);
    grpoConfig.modelConfig = process.env.MODEL_PATH;
  }

  if (process.env.OUTPUT_DIR) {
    console.log(`🔧 Overriding output directory from env: ${process.env.OUTPUT_DIR}`);
    grpoConfig.outputDir = process.env.OUTPUT_DIR;
  }

  // Print configuration summary
  printConfigSummary(grpoConfig, datasetConfig);

  // Check if model exists (if it's a path)
  if (
    typeof grpoConfig.modelConfig === 'string' &&
    grpoConfig.modelConfig.includes('/') &&
    !existsSync(grpoConfig.modelConfig)
  ) {
    console.error(`❌ Model not found at: ${grpoConfig.modelConfig}\n`);
    console.error('Please download the model first:');
    console.error('   yarn download:qwen3\n');
    process.exitCode = 1;
    return;
  }

  // Load dataset
  console.log('📚 Loading GSM8K dataset...\n');

  let examples;
  try {
    examples = await loadLocalGsm8kDataset(datasetConfig.split as 'train' | 'test', {
      limit: datasetConfig.maxTrainSamples,
      includeOneShot: datasetConfig.includeOneShot,
    });
  } catch (error) {
    console.error('❌ Failed to load dataset:', error);
    if (error instanceof Error) {
      console.error('\nError details:', error.message);
      console.error('\nMake sure you have downloaded the GSM8K dataset:');
      console.error('   yarn download:gsm8k\n');
    }
    process.exitCode = 1;
    return;
  }

  console.log(`✓ Loaded ${examples.length} training examples\n`);

  // Calculate training metrics
  const stepsPerEpoch = Math.ceil(
    examples.length / ((grpoConfig.batchSize ?? 1) * (grpoConfig.gradientAccumulationSteps ?? 1)),
  );
  const totalSteps = stepsPerEpoch * (grpoConfig.numEpochs ?? 1);
  const estimatedMinutes = Math.ceil((totalSteps * (grpoConfig.groupSize ?? 8) * 0.5) / 60);

  console.log('Training Plan:');
  console.log('─────────────────────────────────────────────────────');
  console.log(`Steps per epoch: ${stepsPerEpoch}`);
  console.log(`Total steps: ${totalSteps}`);
  console.log(`Estimated time: ~${estimatedMinutes} minutes`);
  console.log(`Checkpoints: every ${grpoConfig.saveInterval} steps`);
  console.log('─────────────────────────────────────────────────────\n');

  console.log('🚀 Starting GRPO training...\n');
  console.log('═══════════════════════════════════════════════════════\n');

  // Create trainer and start training
  const trainer = await GRPOTrainer.create(grpoConfig);

  const startTime = Date.now();

  try {
    await trainer.train(examples);

    const endTime = Date.now();
    const elapsedMinutes = ((endTime - startTime) / 1000 / 60).toFixed(1);

    console.log('\n═══════════════════════════════════════════════════════');
    console.log(`✅ Training complete! (${elapsedMinutes} minutes)\n`);
    console.log('Results saved to:', grpoConfig.outputDir);
    console.log('\nNext steps:');
    console.log('  1. Review training logs:');
    console.log(`     cat ${resolve(grpoConfig.outputDir!, 'log.jsonl')}`);
    console.log('  2. Inspect final checkpoint:');
    console.log(`     ls ${resolve(grpoConfig.outputDir!, 'final')}`);
    console.log('  3. Evaluate the trained model:');
    console.log('     node examples/grpo/utils/evaluate.ts');
    console.log('  4. Visualize training metrics (TODO: add plotting script)');
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
  console.error('[train-full] Fatal error:', error);
  process.exitCode = 1;
});
