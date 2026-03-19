import { parseArgs } from 'node:util';
import { resolve } from 'node:path';
import { readFileSync, existsSync } from 'node:fs';
import { convertModel, convertForeignWeights, convertGgufToSafetensors } from '@mlx-node/core';

function printHelp() {
  console.log(`
Convert Model Weights to MLX Format

Usage:
  mlx convert --input <path> --output <dir> [options]

Required Arguments:
  --input, -i <path>    Input model directory or .gguf file
  --output, -o <dir>    Output directory for converted model

Optional Arguments:
  --dtype, -d <type>    Target dtype (default: bfloat16)
                        Options: float32, float16, bfloat16
  --model-type, -m      Model type (auto-detected if not specified)
                        Options: paddleocr-vl, pp-lcnet-ori, uvdoc, qwen3_5, qwen3_5_moe
  --verbose, -v         Enable verbose logging
  --help, -h            Show this help message

Vision Arguments:
  --mmproj <path>       Path to mmproj GGUF file (vision encoder weights)
                        Converts and merges vision weights into output directory

Quantization Arguments:
  --quantize, -q        Enable quantization of converted weights
  --q-bits <int>        Quantization bits (default: 4 for affine, 8 for mxfp8)
  --q-group-size <int>  Group size (default: 64 for affine, 32 for mxfp8)
  --q-mode <string>     Mode: "affine" (default) or "mxfp8"
  --q-recipe <string>   Per-layer mixed-bit quantization recipe
                        Options: mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6, qwen3_5, unsloth
                        "unsloth" defaults to 3-bit base (gate/up=3b, down=4b,
                        embed=5b, lm_head=6b, attn/SSM=bf16)
  --imatrix-path <path> imatrix GGUF file for AWQ-style pre-scaling
                        Improves quantization quality using calibration data

Model Types:
  (default)             SafeTensors dtype conversion (HuggingFace models)
  paddleocr-vl          PaddleOCR-VL weight sanitization
  qwen3_5               Qwen3.5 dense model (FP8 dequant, key remapping)
  qwen3_5_moe           Qwen3.5 MoE model (FP8 dequant, expert stacking)
  pp-lcnet-ori          PP-LCNet orientation classifier (Paddle -> SafeTensors)
  uvdoc                 UVDoc unwarping model (Paddle/PyTorch -> SafeTensors)

GGUF Support:
  When --input points to a .gguf file, the converter automatically parses the
  GGUF binary format and converts tensors to SafeTensors. Supports BF16, F16,
  F32, Q4_0, Q4_1, and Q8_0 tensor types. Tokenizer files are copied from
  alongside the GGUF file if present.

Examples:
  mlx convert -i .cache/models/qwen3-0.6b -o .cache/models/qwen3-0.6b-mlx
  mlx convert -i .cache/models/Qwen3.5-35B-A3B-FP8 -o .cache/models/Qwen3.5-35B-A3B-4bit -m qwen3_5_moe -q --q-bits 4
  mlx convert -m pp-lcnet-ori -i .cache/models/PP-LCNet -o ./models/PP-LCNet_x1_0_doc_ori/
  mlx convert -i model.gguf -o ./models/converted-mlx
  mlx convert -i model-BF16.gguf -o ./models/converted-4bit -q --q-bits 4
  mlx convert -i model-BF16.gguf -o ./models/mixed-4-6 -q --q-recipe mixed_4_6
  mlx convert -i .cache/models/qwen3.5-9b -o ./models/qwen35-recipe -q --q-recipe qwen3_5 -m qwen3_5
  mlx convert -i model-BF16.gguf -o ./models/awq-4bit -q --q-recipe unsloth --imatrix-path imatrix.gguf
  mlx convert -i .cache/models/Qwen3.5-27B -o ./models/qwen3.5-unsloth -q --q-recipe unsloth --mmproj mmproj-BF16.gguf
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
      'q-recipe': { type: 'string' },
      'imatrix-path': { type: 'string' },
      mmproj: { type: 'string' },
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

  const quantRecipe = args['q-recipe'];
  const validRecipes = ['mixed_2_6', 'mixed_3_4', 'mixed_3_6', 'mixed_4_6', 'qwen3_5', 'unsloth'];
  if (quantRecipe !== undefined) {
    if (!args.quantize) {
      console.error('Error: --q-recipe requires --quantize (-q) to be enabled');
      process.exit(1);
    }
    if (quantMode === 'mxfp8') {
      console.error('Error: --q-recipe is incompatible with --q-mode mxfp8');
      process.exit(1);
    }
    if (!validRecipes.includes(quantRecipe)) {
      console.error(`Error: Unknown recipe "${quantRecipe}". Available: ${validRecipes.join(', ')}`);
      process.exit(1);
    }
    // Unsloth recipe defaults to 3-bit base (MLP gate/up at 3-bit, down at 4-bit,
    // embed_tokens at 5-bit, lm_head at 6-bit, attention/SSM kept bf16).
    // Based on Unsloth's per-tensor KLD analysis showing ffn_up/gate are
    // "generally ok to quantize to 3-bit" and IQ3_XXS is the "best compromise".
    if (quantRecipe === 'unsloth' && !args['q-bits']) {
      console.log('Note: unsloth recipe defaults to 3-bit base (override with --q-bits)');
    }
  }

  // Apply recipe-specific defaults for bits when not explicitly set.
  // Unsloth recipe: 3-bit base → MLP gate/up=3b, down=4b, embed=5b, lm_head=6b
  const effectiveQuantBits = quantBits ?? (quantRecipe === 'unsloth' ? 3 : undefined);

  const mmprojPath = args.mmproj ? resolve(args.mmproj) : undefined;
  if (mmprojPath !== undefined) {
    if (!existsSync(mmprojPath)) {
      console.error(`Error: mmproj file not found: ${mmprojPath}`);
      process.exit(1);
    }
    if (!mmprojPath.endsWith('.gguf')) {
      console.error('Error: --mmproj must point to a .gguf file');
      process.exit(1);
    }
  }

  const imatrixPath = args['imatrix-path'] ? resolve(args['imatrix-path']) : undefined;
  if (imatrixPath !== undefined) {
    if (!existsSync(imatrixPath)) {
      console.error(`Error: imatrix file not found: ${imatrixPath}`);
      process.exit(1);
    }
    if (!imatrixPath.endsWith('.gguf')) {
      console.error('Error: --imatrix-path must point to a .gguf file');
      process.exit(1);
    }
  }

  const startTime = Date.now();

  // GGUF file detection
  if (inputPath.endsWith('.gguf')) {
    if (!existsSync(inputPath)) {
      console.error(`Error: GGUF file not found: ${inputPath}`);
      process.exit(1);
    }

    const dtype = args.dtype || 'bfloat16';
    console.log(`Converting GGUF to SafeTensors`);
    console.log(`Input:      ${inputPath}`);
    console.log(`Output:     ${outputDir}`);
    console.log(`Dtype:      ${dtype}`);
    if (args.quantize) {
      const qMode = quantMode || 'affine';
      const qBits = effectiveQuantBits || (qMode === 'mxfp8' ? 8 : 4);
      const qGs = quantGroupSize || (qMode === 'mxfp8' ? 32 : 64);
      console.log(
        `Quantize:   ${qBits}-bit ${qMode} (group_size=${qGs})${quantRecipe ? `, recipe=${quantRecipe}` : ''}`,
      );
    }
    if (imatrixPath) {
      console.log(`imatrix:    ${imatrixPath}`);
    }
    if (mmprojPath) {
      console.log(`mmproj:     ${mmprojPath}`);
    }
    console.log('');

    try {
      const result = await convertGgufToSafetensors({
        inputPath,
        outputDir,
        dtype,
        verbose,
        quantize: args.quantize,
        quantBits: effectiveQuantBits,
        quantGroupSize,
        quantMode,
        quantRecipe,
        imatrixPath,
        vlmKeyPrefix: !!mmprojPath,
      });

      const duration = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log(`\n✓ Converted ${result.numTensors} tensors (source: ${result.sourceFormat})`);
      console.log(`✓ Total parameters: ${result.numParameters.toLocaleString()}`);
      console.log(`✓ Output directory: ${result.outputPath}`);
      console.log(`✓ Duration: ${duration}s`);

      if (verbose) {
        console.log('\nConverted tensors:');
        for (const name of result.tensorNames) {
          console.log(`  - ${name}`);
        }
      }

      // Convert mmproj (vision encoder) if provided
      if (mmprojPath) {
        console.log('\nConverting mmproj (vision encoder)...');
        const visionResult = await convertGgufToSafetensors({
          inputPath: mmprojPath,
          outputDir,
          dtype: 'bfloat16',
          verbose,
          quantize: false,
          outputFilename: 'vision.safetensors',
        });
        console.log(`✓ Converted ${visionResult.numTensors} vision tensors`);
      }
    } catch (error: any) {
      console.error('\nGGUF conversion failed:', error.message);
      if (error.stack && verbose) {
        console.error('\nStack trace:', error.stack);
      }
      process.exit(1);
    }
    return;
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
    const qBits = effectiveQuantBits || (qMode === 'mxfp8' ? 8 : 4);
    const qGs = quantGroupSize || (qMode === 'mxfp8' ? 32 : 64);
    console.log(`Quantize:   ${qBits}-bit ${qMode} (group_size=${qGs})${quantRecipe ? `, recipe=${quantRecipe}` : ''}`);
  }
  if (imatrixPath) {
    console.log(`imatrix:    ${imatrixPath}`);
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
      quantBits: effectiveQuantBits,
      quantGroupSize,
      quantMode,
      quantRecipe,
      imatrixPath,
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
