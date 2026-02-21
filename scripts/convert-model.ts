/**
 * Convert Model Weights to MLX Format
 *
 * Supports multiple conversion paths:
 * 1. HuggingFace SafeTensors (dtype conversion: F16/BF16 -> F32)
 * 2. PP-LCNet orientation classifier (Paddle .pdparams -> SafeTensors)
 * 3. UVDoc unwarping model (Paddle .pdiparams or PyTorch .pkl -> SafeTensors)
 *
 * Usage:
 *   # SafeTensors dtype conversion
 *   oxnode scripts/convert-model.ts --input <dir> --output <dir> [--dtype float32]
 *
 *   # Paddle orientation model conversion
 *   oxnode scripts/convert-model.ts -m pp-lcnet-ori --input model.pdparams --output ./models/PP-LCNet_x1_0_doc_ori/
 *
 *   # PyTorch UVDoc conversion
 *   oxnode scripts/convert-model.ts -m uvdoc --input best_model.pkl --output ./models/UVDoc/
 */

import { resolve } from 'node:path';
import { readFileSync, existsSync } from 'node:fs';
import { convertModel, convertForeignWeights } from '@mlx-node/core';

interface Args {
  input?: string;
  output?: string;
  dtype?: string;
  modelType?: string;
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

      case '--model-type':
      case '-m':
        args.modelType = argv[++i];
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
║   Convert Model Weights to MLX Format                  ║
╚════════════════════════════════════════════════════════╝

Usage:
  oxnode scripts/convert-model.ts --input <path> --output <dir> [options]

Required Arguments:
  --input, -i <path>    Input model directory or weights file
  --output, -o <dir>    Output directory for converted model

Optional Arguments:
  --dtype, -d <type>    Target dtype (default: bfloat16)
                        Options: float32, float16, bfloat16
  --model-type, -m      Model type (auto-detected if not specified)
                        Options: paddleocr-vl, pp-lcnet-ori, uvdoc
  --verbose, -v         Enable verbose logging
  --help, -h            Show this help message

Model Types:
  (default)             SafeTensors dtype conversion (HuggingFace models)
  paddleocr-vl          PaddleOCR-VL weight sanitization
  pp-lcnet-ori          PP-LCNet orientation classifier (Paddle .pdparams/.pdiparams -> SafeTensors)
  uvdoc                 UVDoc unwarping model (Paddle .pdiparams or PyTorch .pkl -> SafeTensors)

Examples:
  # Convert Qwen3-0.6B from BF16 to Float32 for GRPO training
  oxnode scripts/convert-model.ts \\
    -i .cache/models/qwen3-0.6b \\
    -o .cache/models/qwen3-0.6b-mlx

  # Convert PP-LCNet orientation model from Paddle format (directory or file)
  oxnode scripts/convert-model.ts \\
    -m pp-lcnet-ori \\
    -i .cache/models/PP-LCNet_x1_0_doc_ori \\
    -o ./models/PP-LCNet_x1_0_doc_ori/

  # Convert UVDoc unwarping model from HuggingFace download
  oxnode scripts/convert-model.ts \\
    -m uvdoc \\
    -i .cache/models/UVDoc \\
    -o .cache/models/UVDoc-mlx
`);
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const args = parseArgs();

  if (args.help) {
    printHelp();
    process.exit(0);
  }

  if (!args.input || !args.output) {
    console.error('Error: Both --input and --output are required\n');
    console.error('Use --help for usage information');
    process.exit(1);
  }

  const inputPath = resolve(args.input);
  const outputDir = resolve(args.output);
  const verbose = args.verbose || false;

  // Auto-detect model type from config.json if not specified
  let modelType = args.modelType;
  if (!modelType) {
    try {
      const configPath = resolve(inputPath, 'config.json');
      const config = JSON.parse(readFileSync(configPath, 'utf-8'));
      if (config.model_type === 'paddleocr_vl') {
        modelType = 'paddleocr-vl';
        console.log(`Auto-detected model type: ${modelType} (from config.json)`);
      }
    } catch {
      // config.json not found or invalid
    }
  }

  const startTime = Date.now();

  // Foreign weight formats (Paddle .pdparams/.pdiparams, PyTorch .pkl)
  if (modelType === 'pp-lcnet-ori' || modelType === 'uvdoc') {
    if (!existsSync(inputPath)) {
      console.error(`Error: Input path not found: ${inputPath}`);
      process.exit(1);
    }

    const label =
      modelType === 'pp-lcnet-ori'
        ? 'PP-LCNet Orientation Classifier (Paddle -> SafeTensors)'
        : 'UVDoc Unwarping Model (-> SafeTensors)';

    console.log(`Converting: ${label}`);
    console.log(`Input:  ${inputPath}`);
    console.log(`Output: ${outputDir}\n`);

    const result = convertForeignWeights({
      inputPath,
      outputDir,
      modelType,
      verbose,
    });

    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`\n✓ Converted ${result.numTensors} tensors`);
    console.log(`✓ Output directory: ${result.outputPath}`);
    console.log(`✓ Duration: ${duration}s`);
    return;
  }

  // Default: SafeTensors dtype conversion
  const dtype = args.dtype || 'bfloat16';

  console.log(`Input:      ${inputPath}`);
  console.log(`Output:     ${outputDir}`);
  console.log(`Dtype:      ${dtype}`);
  if (modelType) {
    console.log(`Model Type: ${modelType}`);
  }
  console.log('');

  try {
    const result = await convertModel({
      inputDir: inputPath,
      outputDir,
      dtype,
      verbose,
      modelType,
    });

    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`\n✓ Converted ${result.numTensors} tensors`);
    console.log(`✓ Total parameters: ${result.numParameters.toLocaleString()}`);
    console.log(`✓ Output directory: ${result.outputPath}`);
    console.log(`✓ Duration: ${duration}s`);

    if (verbose) {
      console.log('\nConverted tensors:');
      for (const name of result.tensorNames) {
        console.log(`  - ${name}`);
      }
    }
  } catch (error: any) {
    console.error('\nConversion failed:', error.message);
    if (error.stack && verbose) {
      console.error('\nStack trace:', error.stack);
    }
    process.exit(1);
  }
}

main().catch((error) => {
  console.error('Unexpected error:', error);
  process.exit(1);
});
