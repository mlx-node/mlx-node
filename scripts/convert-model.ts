/**
 * Convert HuggingFace SafeTensors Model to MLX Format
 *
 * This script converts a HuggingFace model from any dtype (F16, BF16, etc.)
 * to MLX float32 format for GRPO training.
 *
 * Usage:
 *   node scripts/convert-model.ts --input <input-dir> --output <output-dir> [--dtype float32] [--verbose]
 *   yarn convert:model --input .cache/models/qwen3-0.6b --output .cache/models/qwen3-0.6b-mlx
 *
 * Arguments:
 *   --input, -i     Input directory containing HuggingFace model (required)
 *   --output, -o    Output directory for converted MLX model (required)
 *   --dtype, -d     Target dtype (default: float32, options: float32, float16, bfloat16)
 *   --verbose, -v   Enable verbose logging
 *
 * Example:
 *   # Convert Qwen3-0.6B from BF16 to Float32
 *   yarn convert:model -i .cache/models/qwen3-0.6b -o .cache/models/qwen3-0.6b-mlx
 *
 *   # Convert to Float16
 *   yarn convert:model -i .cache/models/qwen3-0.6b -o .cache/models/qwen3-0.6b-f16 -d float16
 */

import { resolve } from 'node:path';
import { convertModel } from '@mlx-node/core';

interface Args {
  input?: string;
  output?: string;
  dtype?: string;
  verbose?: boolean;
  help?: boolean;
}

function parseArgs(): Args {
  const args: Args = {};
  const argv = process.argv.slice(2);

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];

    switch (arg) {
      case '--input':
      case '-i':
        args.input = argv[++i];
        break;

      case '--output':
      case '-o':
        args.output = argv[++i];
        break;

      case '--dtype':
      case '-d':
        args.dtype = argv[++i];
        break;

      case '--verbose':
      case '-v':
        args.verbose = true;
        break;

      case '--help':
      case '-h':
        args.help = true;
        break;

      default:
        console.error(`Unknown argument: ${arg}`);
        console.error('Use --help for usage information');
        process.exit(1);
    }
  }

  return args;
}

function printHelp() {
  console.log(`
╔════════════════════════════════════════════════════════╗
║   Convert HuggingFace Model to MLX Format              ║
╚════════════════════════════════════════════════════════╝

Usage:
  node scripts/convert-model.ts --input <dir> --output <dir> [options]

Required Arguments:
  --input, -i <dir>     Input directory with HuggingFace model
  --output, -o <dir>    Output directory for MLX model

Optional Arguments:
  --dtype, -d <type>    Target dtype (default: bfloat16)
                        Options: float32, float16, bfloat16
  --verbose, -v         Enable verbose logging
  --help, -h            Show this help message

Examples:
  # Convert Qwen3-0.6B from BF16 to Float32 for GRPO training
  node scripts/convert-model.ts \\
    --input .cache/models/qwen3-0.6b \\
    --output .cache/models/qwen3-0.6b-mlx \\
    --verbose

  # Convert to Float16
  node scripts/convert-model.ts \\
    -i .cache/models/qwen3-0.6b \\
    -o .cache/models/qwen3-0.6b-f16 \\
    -d float16

Why Convert?
  - Float32 provides better gradient stability for GRPO training
  - BFloat16/Float16 models have reduced precision unsuitable for RL
  - Conversion ensures numerical accuracy during training

Note:
  The script will copy all necessary files (config, tokenizer, etc.)
  to the output directory.
`);
}

async function main() {
  const args = parseArgs();

  if (args.help) {
    printHelp();
    process.exit(0);
  }

  // Validate required arguments
  if (!args.input || !args.output) {
    console.error('Error: Both --input and --output are required\n');
    console.error('Use --help for usage information');
    process.exit(1);
  }

  // Resolve paths
  const inputDir = resolve(args.input);
  const outputDir = resolve(args.output);
  const dtype = args.dtype || 'bfloat16';
  const verbose = args.verbose || false;

  console.log('╔════════════════════════════════════════════════════════╗');
  console.log('║   Converting HuggingFace Model to MLX Format           ║');
  console.log('╚════════════════════════════════════════════════════════╝\n');

  console.log(`Input:  ${inputDir}`);
  console.log(`Output: ${outputDir}`);
  console.log(`Dtype:  ${dtype}\n`);

  try {
    const startTime = Date.now();

    const result = await convertModel({
      inputDir,
      outputDir,
      dtype,
      verbose,
    });

    const duration = ((Date.now() - startTime) / 1000).toFixed(2);

    console.log('\n╔════════════════════════════════════════════════════════╗');
    console.log('║   Conversion Complete!                                 ║');
    console.log('╚════════════════════════════════════════════════════════╝\n');

    console.log(`✓ Converted ${result.numTensors} tensors`);
    console.log(`✓ Total parameters: ${result.numParameters.toLocaleString()}`);
    console.log(`✓ Output directory: ${result.outputPath}`);
    console.log(`✓ Duration: ${duration}s\n`);

    if (verbose) {
      console.log('Converted tensors:');
      for (const name of result.tensorNames) {
        console.log(`  - ${name}`);
      }
      console.log('');
    }

    console.log('You can now use this model for GRPO training:');
    console.log(`  yarn oxnode examples/grpo/train-simple.ts --model ${outputDir}\n`);
  } catch (error: any) {
    console.error('\n❌ Conversion failed!\n');
    console.error('Error:', error.message);

    if (error.stack && verbose) {
      console.error('\nStack trace:');
      console.error(error.stack);
    }

    process.exit(1);
  }
}

main().catch((error) => {
  console.error('Unexpected error:', error);
  process.exit(1);
});
