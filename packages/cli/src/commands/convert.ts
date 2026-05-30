import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { convertModel, convertForeignWeights, convertGgufToSafetensors } from '@mlx-node/core';

// Canonical per-mode defaults for quantization bits/group_size.
// Mirrors crates/mlx-core/src/convert.rs and crates/mlx-core/src/utils/gguf.rs.
const QUANT_MODE_DEFAULTS: Record<string, [number, number]> = {
  affine: [4, 64],
  mxfp4: [4, 32],
  mxfp8: [8, 32],
  nvfp4: [4, 16],
};

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
                        Options: paddleocr-vl, pp-lcnet-ori, uvdoc, qwen3_5, qwen3_5_moe, lfm2_moe, lfm2, qianfan-ocr, privacy-filter
  --verbose, -v         Enable verbose logging
  --help, -h            Show this help message

Vision Arguments:
  --mmproj <path>       Path to mmproj GGUF file (vision encoder weights)
                        Converts and merges vision weights into output directory

Quantization Arguments:
  --quantize, -q        Enable quantization of converted weights
  --q-bits <int>        Quantization bits (default per --q-mode: affine=4, mxfp4=4, mxfp8=8, nvfp4=4)
  --q-group-size <int>  Group size (default per --q-mode: affine=64, mxfp4=32, mxfp8=32, nvfp4=16)
  --q-mode <string>     Mode: "affine" (default), "mxfp4", "mxfp8", or "nvfp4"
  --q-mxfp              Upgrade quantization to micro-scaling FP (mxfp4 / mxfp8).
                        Applies after the recipe predicate: any 8-bit affine
                        decision becomes mxfp8, any 4-bit becomes mxfp4. Requires
                        --quantize and --q-mode affine (default). Forces
                        group_size=32 for upgraded layers. Skipped keys (kept at
                        8-bit affine for accuracy or loader compatibility):
                        lm_head, router projections, embed_tokens (incl. Gemma4
                        embed_tokens_per_layer), embedding_projection, and MoE
                        router gates (mlp.gate / shared_expert_gate). MXFP8's
                        coarse E8M0 scales destroy top-K expert routing on the
                        gates, matching the Python mlx-lm quant_predicate.

                        Recommended combo: --q-recipe unsloth --q-bits 4 --q-mxfp
                        promotes mlp.gate_proj/up_proj to mxfp4 (q/k/v at 6-bit
                        affine, down_proj at 5-bit affine, router gates at 8-bit
                        affine, lm_head at 8-bit affine, out_proj stays bf16).
                        At default --q-bits 3 only down_proj (4b -> mxfp4) gets
                        the upgrade.
  --q-recipe <string>   Per-layer mixed-bit quantization recipe
                        Options: mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6, qwen3_5, unsloth
                        "unsloth" defaults to 3-bit base (gate/up=3b, down=4b,
                        embed=5b, lm_head=6b, attn q/k/v=5b+AWQ, out_proj=bf16)
                        "unsloth" requires --imatrix-path for quality
  --imatrix-path <path> imatrix GGUF file for AWQ-style pre-scaling
                        Improves quantization quality using calibration data
                        Required for "unsloth" recipe

Model Types:
  (default)             SafeTensors dtype conversion (HuggingFace models)
  paddleocr-vl          PaddleOCR-VL weight sanitization
  qwen3_5               Qwen3.5 dense model (FP8 dequant, key remapping)
  qwen3_5_moe           Qwen3.5 MoE model (FP8 dequant, expert stacking)
  lfm2_moe              LFM2 MoE model (MLP rename, conv transpose, expert stacking; affine quant only)
  lfm2                  LFM2 dense model (MLP rename, conv transpose; affine quant only)
  pp-lcnet-ori          PP-LCNet orientation classifier (Paddle -> SafeTensors)
  uvdoc                 UVDoc unwarping model (Paddle/PyTorch -> SafeTensors)
  qianfan-ocr           Qianfan-OCR InternVL model (key renaming, conv2d transposition)

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
  mlx convert -i .cache/models/Qwen3.5-27B -o ./models/qwen3.5-unsloth -q --q-recipe unsloth --imatrix-path imatrix.gguf
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
      'q-mxfp': { type: 'boolean', default: false },
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

  const validQuantModes = ['affine', 'mxfp4', 'mxfp8', 'nvfp4'];
  if (quantMode !== undefined && !validQuantModes.includes(quantMode)) {
    console.error(`Error: --q-mode must be one of ${validQuantModes.join(', ')}`);
    process.exit(1);
  }

  if (args['q-mxfp'] && !args.quantize) {
    console.error('Error: --q-mxfp requires --quantize');
    process.exit(1);
  }

  if (args['q-mxfp'] && quantMode !== undefined && quantMode !== 'affine') {
    console.error(
      `Error: --q-mxfp requires --q-mode affine (got '${quantMode}'). --q-mxfp orthogonally upgrades affine decisions to mxfp4/mxfp8.`,
    );
    process.exit(1);
  }

  const quantRecipe = args['q-recipe'];
  const validRecipes = ['mixed_2_6', 'mixed_3_4', 'mixed_3_6', 'mixed_4_6', 'qwen3_5', 'unsloth'];
  if (quantRecipe !== undefined) {
    if (!args.quantize) {
      console.error('Error: --q-recipe requires --quantize (-q) to be enabled');
      process.exit(1);
    }
    if (quantMode !== undefined && quantMode !== 'affine' && quantMode !== 'nvfp4') {
      console.error(
        `Error: --q-recipe is compatible with --q-mode affine or nvfp4 only; for mxfp4/mxfp8 use --q-mxfp instead. Got '${quantMode}'.`,
      );
      process.exit(1);
    }
    if (!validRecipes.includes(quantRecipe)) {
      console.error(`Error: Unknown recipe "${quantRecipe}". Available: ${validRecipes.join(', ')}`);
      process.exit(1);
    }
    // Restrict --q-mode nvfp4 + --q-recipe to recipes that have model-aware
    // tensor-class exclusions for NVFP4-sensitive layers (e.g.
    // linear_attn.out_proj — KLD ~6.0 in hybrid Qwen3.5/3.6 models). The
    // generic `mixed_*` recipes lack these skip lists, so they would silently
    // promote highly-sensitive layers to NVFP4 and corrupt the model.
    if (quantMode === 'nvfp4' && quantRecipe !== 'unsloth' && quantRecipe !== 'qwen3_5') {
      console.error(
        `Error: --q-mode nvfp4 + --q-recipe is currently supported only for 'unsloth' and 'qwen3_5' recipes (got '${quantRecipe}'). Other recipes lack tensor-class exclusions for NVFP4-sensitive layers (e.g. linear_attn.out_proj).`,
      );
      process.exit(1);
    }
    // Unsloth recipe defaults to 3-bit base (MLP gate/up at 3-bit, down at 4-bit,
    // embed_tokens at 5-bit, lm_head at 6-bit, attn q/k/v + SSM in_proj at 5-bit
    // with AWQ pre-scaling via input_layernorm, out_proj/o_proj kept at bf16).
    // Based on Unsloth's per-tensor KLD analysis. Requires imatrix for AWQ correction
    // on the attention/SSM projections.
    if (quantRecipe === 'unsloth' && !args['q-bits'] && quantMode !== 'nvfp4') {
      console.log('Note: unsloth recipe defaults to 3-bit base (override with --q-bits)');
    }
    if (quantRecipe === 'unsloth' && !args['imatrix-path']) {
      console.error('Error: --q-recipe unsloth requires --imatrix-path for AWQ pre-scaling');
      console.error('       imatrix calibration data is needed for near-lossless attention/SSM quantization');
      console.error('       Generate with: llama-imatrix -m model.gguf -f calibration.txt -o imatrix.gguf');
      process.exit(1);
    }
  }

  // Apply recipe-specific defaults for bits when not explicitly set.
  // Unsloth recipe: 3-bit base → MLP gate/up=3b, down=4b, embed=5b, lm_head=6b, attn q/k/v=5b+AWQ, out_proj=bf16
  // Exception: --q-mode nvfp4 forces bits=4 regardless of recipe, because
  // NVFP4 invariant requires bits=4 (the unsloth 3-bit default would otherwise
  // produce an inconsistent checkpoint: top-level bits=3 but per-layer
  // overrides at bits=4 from apply_nvfp4_upgrade, with no failure surface).
  const effectiveQuantBits = quantBits ?? (quantMode === 'nvfp4' ? 4 : quantRecipe === 'unsloth' ? 3 : undefined);

  // MXFP modes have strict bits/group_size invariants enforced by the MLX
  // backend. Surface the failure here rather than letting it bubble up as a
  // confusing FFI error mid-conversion. Use the effective bits (post-recipe)
  // and the effective group_size that would be sent to the native side.
  if (args.quantize && quantMode === 'mxfp4') {
    const effBits = effectiveQuantBits ?? 4;
    const effGs = quantGroupSize ?? 32;
    if (effBits !== 4 || effGs !== 32) {
      console.error(`Error: mxfp4 requires bits=4 and group_size=32 (got bits=${effBits}, group_size=${effGs})`);
      process.exit(1);
    }
  }
  if (args.quantize && quantMode === 'mxfp8') {
    const effBits = effectiveQuantBits ?? 8;
    const effGs = quantGroupSize ?? 32;
    if (effBits !== 8 || effGs !== 32) {
      console.error(`Error: mxfp8 requires bits=8 and group_size=32 (got bits=${effBits}, group_size=${effGs})`);
      process.exit(1);
    }
  }
  if (args.quantize && quantMode === 'nvfp4') {
    const effBits = effectiveQuantBits ?? 4;
    const effGs = quantGroupSize ?? 16;
    if (effBits !== 4 || effGs !== 16) {
      console.error(`Error: nvfp4 requires bits=4 and group_size=16 (got bits=${effBits}, group_size=${effGs})`);
      process.exit(1);
    }
  }

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
    // HuggingFace Hub names some GGUF files with .gguf_file extension
    // (e.g. Unsloth's imatrix_unsloth.gguf_file). Accept both extensions.
    if (!imatrixPath.endsWith('.gguf') && !imatrixPath.endsWith('.gguf_file')) {
      console.error('Error: --imatrix-path must point to a .gguf or .gguf_file');
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
      const [defaultBits, defaultGs] = QUANT_MODE_DEFAULTS[qMode] ?? [4, 64];
      const qBits = effectiveQuantBits || defaultBits;
      const qGs = quantGroupSize || defaultGs;
      const qMxfpSuffix = args['q-mxfp'] ? ', --q-mxfp: 8b->mxfp8, 4b->mxfp4' : '';
      console.log(
        `Quantize:   ${qBits}-bit ${qMode} (group_size=${qGs})${quantRecipe ? `, recipe=${quantRecipe}` : ''}${qMxfpSuffix}`,
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
        quantMxfp: args['q-mxfp'],
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
      } else if (config.model_type === 'gemma4' || config.model_type === 'gemma4_text') {
        modelType = 'gemma4';
        console.log(`Auto-detected model type: ${modelType} (from config.json)`);
      } else if (config.model_type === 'lfm2_moe' || config.model_type === 'lfm2') {
        modelType = config.model_type;
        console.log(`Auto-detected model type: ${modelType} (from config.json)`);
      } else if (config.model_type === 'openai_privacy_filter') {
        modelType = 'privacy-filter';
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
    const [defaultBits, defaultGs] = QUANT_MODE_DEFAULTS[qMode] ?? [4, 64];
    const qBits = effectiveQuantBits || defaultBits;
    const qGs = quantGroupSize || defaultGs;
    const qMxfpSuffix = args['q-mxfp'] ? ', --q-mxfp: 8b->mxfp8, 4b->mxfp4' : '';
    console.log(
      `Quantize:   ${qBits}-bit ${qMode} (group_size=${qGs})${quantRecipe ? `, recipe=${quantRecipe}` : ''}${qMxfpSuffix}`,
    );
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
      quantMxfp: args['q-mxfp'],
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
