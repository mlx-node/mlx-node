import { parseArgs } from 'node:util';
import { resolve } from 'node:path';
import { readFileSync, existsSync } from 'node:fs';
import { convertModel, convertForeignWeights } from '@mlx-node/core';

function printHelp() {
  console.log(`
Convert Model Weights to MLX Format

Usage:
  mlx convert --input <path> --output <dir> [options]

Required Arguments:
  --input, -i <path>    Input model directory or weights file
  --output, -o <dir>    Output directory for converted model

Optional Arguments:
  --dtype, -d <type>    Target dtype (default: bfloat16)
                        Options: float32, float16, bfloat16
  --model-type, -m      Model type (auto-detected if not specified)
                        Options: paddleocr-vl, pp-lcnet-ori, uvdoc, qwen3_5, qwen3_5_moe
  --verbose, -v         Enable verbose logging
  --help, -h            Show this help message

Quantization Arguments:
  --quantize, -q        Enable quantization of converted weights
  --q-bits <int>        Quantization bits (default: 4 for affine, 8 for mxfp8)
  --q-group-size <int>  Group size (default: 64 for affine, 32 for mxfp8)
  --q-mode <string>     Mode: "affine" (default) or "mxfp8"

Model Types:
  (default)             SafeTensors dtype conversion (HuggingFace models)
  paddleocr-vl          PaddleOCR-VL weight sanitization
  qwen3_5               Qwen3.5 dense model (FP8 dequant, key remapping)
  qwen3_5_moe           Qwen3.5 MoE model (FP8 dequant, expert stacking)
  pp-lcnet-ori          PP-LCNet orientation classifier (Paddle -> SafeTensors)
  uvdoc                 UVDoc unwarping model (Paddle/PyTorch -> SafeTensors)

Examples:
  mlx convert -i .cache/models/qwen3-0.6b -o .cache/models/qwen3-0.6b-mlx
  mlx convert -i .cache/models/Qwen3.5-35B-A3B-FP8 -o .cache/models/Qwen3.5-35B-A3B-4bit -m qwen3_5_moe -q --q-bits 4
  mlx convert -m pp-lcnet-ori -i .cache/models/PP-LCNet -o ./models/PP-LCNet_x1_0_doc_ori/
`);
}

export async function run(argv: string[]) {
  const { values: args } = parseArgs({
    args: argv,
    options: {
      input: { type: 'string', short: 'i' },
      output: { type: 'string', short: 'o' },
      dtype: { type: 'string', short: 'd' },
      'model-type': { type: 'string', short: 'm' },
      verbose: { type: 'boolean', short: 'v', default: false },
      quantize: { type: 'boolean', short: 'q', default: false },
      'q-bits': { type: 'string' },
      'q-group-size': { type: 'string' },
      'q-mode': { type: 'string' },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });

  if (args.help) {
    printHelp();
    return;
  }

  if (!args.input || !args.output) {
    console.error('Error: Both --input and --output are required\n');
    console.error('Use --help for usage information');
    process.exit(1);
  }

  const inputPath = resolve(args.input);
  const outputDir = resolve(args.output);
  const verbose = args.verbose!;

  const parsePositiveInt = (flag: string, raw?: string): number | undefined => {
    if (raw === undefined) return undefined;
    if (!/^[1-9]\d*$/.test(raw)) {
      console.error(`Error: ${flag} requires a positive integer value`);
      process.exit(1);
    }
    return Number(raw);
  };

  const quantBits = parsePositiveInt('--q-bits', args['q-bits']);
  const quantGroupSize = parsePositiveInt('--q-group-size', args['q-group-size']);
  const quantMode = args['q-mode'];

  if (quantMode !== undefined && quantMode !== 'affine' && quantMode !== 'mxfp8') {
    console.error('Error: --q-mode must be "affine" or "mxfp8"');
    process.exit(1);
  }

  // Auto-detect model type from config.json if not specified
  let modelType = args['model-type'];
  if (!modelType) {
    try {
      const configPath = resolve(inputPath, 'config.json');
      const config = JSON.parse(readFileSync(configPath, 'utf-8'));
      if (config.model_type === 'paddleocr_vl') {
        modelType = 'paddleocr-vl';
        console.log(`Auto-detected model type: ${modelType} (from config.json)`);
      } else if (config.model_type === 'qwen3_5_moe' || config.model_type === 'qwen3_5') {
        modelType = config.model_type;
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
  if (args.quantize) {
    const qMode = quantMode || 'affine';
    const qBits = quantBits || (qMode === 'mxfp8' ? 8 : 4);
    const qGs = quantGroupSize || (qMode === 'mxfp8' ? 32 : 64);
    console.log(`Quantize:   ${qBits}-bit ${qMode} (group_size=${qGs})`);
  }
  console.log('');

  try {
    const result = await convertModel({
      inputDir: inputPath,
      outputDir,
      dtype,
      verbose,
      modelType,
      quantize: args.quantize,
      quantBits,
      quantGroupSize,
      quantMode,
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
