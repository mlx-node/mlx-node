/**
 * Download Qwen3-0.6B base model from HuggingFace
 *
 * This script downloads the base Qwen3-0.6B model (float16/bfloat16) from HuggingFace Hub.
 * The base model needs to be converted to MLX float32 format for GRPO training.
 *
 * The model will be downloaded to: .cache/models/qwen3-0.6b/
 *
 * Usage:
 *   node scripts/download-qwen3.ts
 *   yarn download:qwen3
 *
 * Environment variables:
 *   MODEL_NAME - HuggingFace model name (default: Qwen/Qwen3-0.6B)
 *   MODEL_OUTPUT_DIR - Output directory (default: .cache/models/qwen3-0.6b)
 *
 * After downloading, convert to MLX float32 format:
 *   cd mlx-lm
 *   python -m mlx_lm.convert --hf-path ../.cache/models/qwen3-0.6b \
 *     --mlx-path ../.cache/models/qwen3-0.6b-mlx --dtype float32
 */

import { mkdir, readdir, stat, copyFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { snapshotDownload } from '@huggingface/hub';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Use base Qwen3-0.6B model (will be converted to MLX float32)
const DEFAULT_MODEL = 'Qwen/Qwen3-0.6B';
const DEFAULT_OUTPUT_DIR = resolve(__dirname, '..', '.cache', 'models', 'qwen3-0.6b');

// Files we need to download from base model
// Note: Base models may have model.safetensors or pytorch_model.bin
// The conversion script will handle the actual weights
const REQUIRED_FILES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'vocab.json',
  'merges.txt',
  'model.safetensors', // Base model weights (will be converted to MLX format)
];

async function ensureDir(path: string): Promise<void> {
  if (!existsSync(path)) {
    await mkdir(path, { recursive: true });
  }
}

async function formatBytes(bytes: number): Promise<string> {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  return `${size.toFixed(2)} ${units[unitIndex]}`;
}

async function findFileInSnapshot(snapshotPath: string, filename: string): Promise<string | null> {
  // HuggingFace snapshots have a flat structure with symlinks to blobs
  // Check if file exists directly in snapshot directory
  const filePath = join(snapshotPath, filename);

  if (existsSync(filePath)) {
    return filePath;
  }

  return null;
}

async function copyModelFile(snapshotPath: string, filename: string, outputDir: string): Promise<void> {
  const outputPath = join(outputDir, filename);

  console.log(`  Copying ${filename}...`);

  try {
    const sourcePath = await findFileInSnapshot(snapshotPath, filename);

    if (!sourcePath) {
      throw new Error(`File ${filename} not found in snapshot`);
    }

    await copyFile(sourcePath, outputPath);

    // Get file size
    const stats = await stat(outputPath);
    const sizeStr = await formatBytes(stats.size);
    console.log(`  ✓ ${filename} (${sizeStr})`);
  } catch (error) {
    console.error(`  ✗ Failed to copy ${filename}:`, error);
    throw error;
  }
}

async function renameWeightsFile(outputDir: string): Promise<void> {
  // For base models, keep the original name (model.safetensors)
  // The MLX conversion script will handle the format
  const sourcePath = join(outputDir, 'model.safetensors');

  if (existsSync(sourcePath)) {
    console.log('\n✓ Weights file present: model.safetensors');
    console.log('  (Will be converted to MLX format in next step)');
  }
}

async function verifyDownload(outputDir: string): Promise<boolean> {
  console.log('\nVerifying download...');

  // For base model, we just need config and weights
  const requiredForLoading = ['config.json', 'model.safetensors'];
  let allPresent = true;

  for (const file of requiredForLoading) {
    const path = join(outputDir, file);
    if (!existsSync(path)) {
      console.error(`  ✗ Missing required file: ${file}`);
      allPresent = false;
    } else {
      console.log(`  ✓ ${file}`);
    }
  }

  return allPresent;
}

async function main() {
  const modelName = process.env.MODEL_NAME ?? DEFAULT_MODEL;
  const outputDir = process.env.MODEL_OUTPUT_DIR ? resolve(process.env.MODEL_OUTPUT_DIR) : DEFAULT_OUTPUT_DIR;

  console.log('╔════════════════════════════════════════════════════════╗');
  console.log('║   Qwen3-0.6B Base Model Download from HuggingFace      ║');
  console.log('╚════════════════════════════════════════════════════════╝\n');

  console.log(`Model: ${modelName}`);
  console.log(`Format: Base model (needs MLX conversion)`);
  console.log(`Output: ${outputDir}\n`);

  console.log('⚠️  Note: After download, convert to MLX float16:');
  console.log(
    '    yarn oxnode ./scripts/convert-model.ts --input .cache/models/qwen3-0.6b --output .cache/models/qwen3-0.6b-mlx-bf16',
  );

  // Check if already downloaded
  if (existsSync(outputDir)) {
    const files = await readdir(outputDir);
    if (files.includes('config.json') && files.includes('model.safetensors')) {
      console.log('✅ Model already downloaded!\n');
      console.log('To re-download, delete the output directory first:');
      console.log(`   rm -rf ${outputDir}\n`);
      return;
    }
  }

  // Create output directory
  await ensureDir(outputDir);

  console.log('📦 Downloading base model from HuggingFace...\n');
  console.log('This may take a while (model is ~1.1 GB)...\n');

  // Download entire model snapshot
  const snapshotPath = await snapshotDownload({
    repo: { type: 'model', name: modelName },
    cacheDir: join(__dirname, '..', '.cache', 'huggingface'),
  });

  console.log(`\nSnapshot downloaded to: ${snapshotPath}\n`);
  console.log('Copying required files...\n');

  // Copy required files from snapshot to output directory
  for (const file of REQUIRED_FILES) {
    await copyModelFile(snapshotPath, file, outputDir);
  }

  // Rename weights file to match loader expectations
  await renameWeightsFile(outputDir);

  // Verify download
  const success = await verifyDownload(outputDir);

  if (success) {
    console.log('\n✅ Model downloaded successfully!\n');
    console.log('You can now run the training demo:');
    console.log('   node examples/grpo/train-simple.ts\n');
  } else {
    console.error('\n❌ Download incomplete. Please try again.\n');
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error('\n[download-model] Error:', error.message);
  console.error('\nStack trace:', error.stack);
  process.exitCode = 1;
});
