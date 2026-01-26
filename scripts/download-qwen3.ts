/**
 * Download Qwen3-0.6B base model from HuggingFace
 *
 * This script downloads the base Qwen3-0.6B model (float16/bfloat16) from HuggingFace Hub.
 * The base model needs to be converted to MLX float32 format for GRPO training.
 *
 * The model will be downloaded to: .cache/models/qwen3-0.6b/
 *
 * Usage:
 *   node scripts/download-qwen3.ts [options]
 *   yarn download:qwen3
 *
 * Options:
 *   -m, --model <name>    HuggingFace model name (default: Qwen/Qwen3-0.6B)
 *   -o, --output <dir>    Output directory (default: .cache/models/qwen3-0.6b)
 *   -h, --help            Show this help message
 *
 * After downloading, convert to MLX float32 format:
 *   cd mlx-lm
 *   python -m mlx_lm.convert --hf-path ../.cache/models/qwen3-0.6b \
 *     --mlx-path ../.cache/models/qwen3-0.6b-mlx --dtype float32
 */

import { mkdir, readdir, stat, copyFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { existsSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { listFiles, whoAmI, downloadFileToCacheDir, type ListFileEntry } from '@huggingface/hub';
import { AsyncEntry } from '@napi-rs/keyring';
import { input } from '@inquirer/prompts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Use base Qwen3-0.6B model (will be converted to MLX float32)
const DEFAULT_MODEL = 'Qwen/Qwen3-0.6B';
const DEFAULT_OUTPUT_DIR = resolve(__dirname, '..', '.cache', 'models', 'qwen3-0.6b');

const keyringEntry = new AsyncEntry('mlx-node', 'huggingface-token');

const HUGGINGFACE_TOKEN = await keyringEntry.getPassword();

if (!HUGGINGFACE_TOKEN) {
  console.warn('No HuggingFace token found, the model will download with anonymous access');
}

// Parse command-line arguments
const { values: args } = parseArgs({
  options: {
    model: {
      type: 'string',
      short: 'm',
      default: DEFAULT_MODEL,
    },
    output: {
      type: 'string',
      short: 'o',
      default: DEFAULT_OUTPUT_DIR,
    },
    help: {
      type: 'boolean',
      short: 'h',
      default: false,
    },
    ['set-token']: {
      type: 'boolean',
      default: false,
    },
  },
});

function printHelp(): void {
  console.log(`
Download ${args.model} base model from HuggingFace

Usage:
  node scripts/download-qwen3.ts [options]
  yarn download:qwen3

Options:
  -m, --model <name>    HuggingFace model name (default: ${DEFAULT_MODEL})
  -o, --output <dir>    Output directory (default: .cache/models/qwen3-0.6b)
  -h, --help            Show this help message
  --set-token           Set HuggingFace token

Examples:
  node scripts/download-qwen3.ts
  node scripts/download-qwen3.ts --model Qwen/Qwen3-1.7B --output .cache/models/qwen3-1.7b
`);
}

async function setToken() {
  const token = await input({
    message: 'Enter your HuggingFace token:',
    required: true,
    theme: {
      validationFailureMode: 'clear',
    },
    validate: async (value) => {
      if (!value) {
        return 'Token is required';
      }
      if (!value.startsWith('hf_')) {
        return 'HuggingFace token must start with "hf_"';
      }
      try {
        const { auth } = await whoAmI({ accessToken: value });
        if (!auth) {
          return 'Invalid token';
        }
        return true;
      } catch {
        return 'Invalid token';
      }
    },
  });
  if (token) {
    keyringEntry.setPassword(token);
  }
}

// Core files we need to download from base model (weights handled separately)
const CORE_FILES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'vocab.json',
  'merges.txt',
];

async function ensureDir(path: string): Promise<void> {
  if (!existsSync(path)) {
    await mkdir(path, { recursive: true });
  }
}

async function getModelFiles(modelName: string) {
  let totalSize = 0;
  const filesToDownload: ListFileEntry[] = [];
  for await (const file of listFiles({ repo: { type: 'model', name: modelName } })) {
    if (CORE_FILES.includes(file.path) || file.path.endsWith('.safetensors') || file.path.endsWith('.json')) {
      filesToDownload.push(file);
      if (file.size) {
        totalSize += file.size;
      }
    }
  }
  return { totalSize, filesToDownload };
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

async function verifyDownload(outputDir: string, weightFiles: string[]): Promise<boolean> {
  console.log('\nVerifying download...');

  let allPresent = true;

  // Check config
  const configPath = join(outputDir, 'config.json');
  if (!existsSync(configPath)) {
    console.error('  ✗ Missing required file: config.json');
    allPresent = false;
  } else {
    console.log('  ✓ config.json');
  }

  // Check weight files
  for (const file of weightFiles) {
    const path = join(outputDir, file);
    if (!existsSync(path)) {
      console.error(`  ✗ Missing weight file: ${file}`);
      allPresent = false;
    } else {
      console.log(`  ✓ ${file}`);
    }
  }

  return allPresent;
}

if (args.help) {
  printHelp();
  process.exit(0);
}

if (args['set-token']) {
  await setToken();
  process.exit(0);
}

const modelName = args.model!;
const outputDir = resolve(args.output!);

const title = `${modelName} Base Model Download from HuggingFace`;
const boxWidth = Math.max(title.length + 6, 58);
const padding = Math.floor((boxWidth - title.length - 2) / 2);
const rightPadding = boxWidth - title.length - padding;
console.log('╔' + '═'.repeat(boxWidth) + '╗');
console.log('║' + ' '.repeat(padding) + title + ' '.repeat(rightPadding) + '║');
console.log('╚' + '═'.repeat(boxWidth) + '╝\n');

console.log(`Model: ${modelName}`);
console.log(`Format: Base model (needs MLX conversion)`);
console.log(`Output: ${outputDir}\n`);

console.log('⚠️  Note: After download, convert to MLX float16:');
console.log(`    yarn oxnode ./scripts/convert-model.ts --input ${args.output} --output ${args.output}-mlx-bf16`);

// Check if already downloaded (single or sharded model)
if (existsSync(outputDir)) {
  const files = await readdir(outputDir);
  const hasConfig = files.includes('config.json');
  const hasSingleModel = files.includes('model.safetensors');
  const hasShardedModel = files.includes('model.safetensors.index.json');

  if (hasConfig && (hasSingleModel || hasShardedModel)) {
    console.log('\n✅ Model already downloaded!\n');
    console.log('To re-download, delete the output directory first:');
    console.log(`   rm -rf ${outputDir}\n`);
    process.exit(0);
  }
}

// Create output directory
await ensureDir(outputDir);

console.log('📦 Downloading base model from HuggingFace...\n');

// Fetch model size from HuggingFace
const { totalSize, filesToDownload } = await getModelFiles(modelName);
const sizeStr = await formatBytes(totalSize);
console.log(`This may take a while (model is ~${sizeStr})...\n`);

const weightFiles: string[] = [];

for (const file of filesToDownload) {
  const snapshotPath = await downloadFileToCacheDir({
    repo: { type: 'model', name: modelName },
    path: file.path,
    cacheDir: join(__dirname, '..', '.cache', 'huggingface'),
    accessToken: HUGGINGFACE_TOKEN,
  });
  // Get file size
  const stats = await stat(snapshotPath);
  const sizeStr = await formatBytes(stats.size);
  await copyFile(snapshotPath, join(outputDir, file.path));
  if (file.path.endsWith('.safetensors')) {
    weightFiles.push(file.path);
  }
  console.log(`  ✓ ${file.path} (${sizeStr})`);
}

const success = await verifyDownload(outputDir, weightFiles);

if (success) {
  console.log('\n✅ Model downloaded successfully!\n');
} else {
  console.error('\n❌ Download incomplete. Please try again.\n');
  process.exit(1);
}
