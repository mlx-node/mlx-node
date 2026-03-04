import { readdir, stat, copyFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { existsSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { listFiles, whoAmI, downloadFileToCacheDir, type ListFileEntry } from '@huggingface/hub';
import { AsyncEntry } from '@napi-rs/keyring';
import { input } from '@inquirer/prompts';

import { ensureDir, formatBytes } from '../utils.js';

const DEFAULT_MODEL = 'Qwen/Qwen3-0.6B';

const keyringEntry = new AsyncEntry('mlx-node', 'huggingface-token');

function printHelp(): void {
  console.log(`
Download a model from HuggingFace

Usage:
  mlx download model [options]

Options:
  -m, --model <name>    HuggingFace model name (default: ${DEFAULT_MODEL})
  -o, --output <dir>    Output directory (default: .cache/models/<model-slug>)
  -h, --help            Show this help message
  --set-token           Set HuggingFace token

Examples:
  mlx download model
  mlx download model --model Qwen/Qwen3-1.7B --output .cache/models/qwen3-1.7b
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
    await keyringEntry.setPassword(token);
  }
}

const CORE_FILES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'vocab.json',
  'merges.txt',
];

async function getModelFiles(modelName: string, accessToken?: string) {
  let totalSize = 0;
  const filesToDownload: ListFileEntry[] = [];
  for await (const file of listFiles({ repo: { type: 'model', name: modelName }, accessToken })) {
    if (
      CORE_FILES.includes(file.path) ||
      file.path.endsWith('.safetensors') ||
      file.path.endsWith('.json') ||
      file.path.endsWith('.pdiparams') ||
      file.path.endsWith('.yml')
    ) {
      filesToDownload.push(file);
      if (file.size) {
        totalSize += file.size;
      }
    }
  }
  return { totalSize, filesToDownload };
}

async function verifyDownload(outputDir: string, weightFiles: string[]): Promise<boolean> {
  console.log('\nVerifying download...');

  let allPresent = true;

  const configPath = join(outputDir, 'config.json');
  if (!existsSync(configPath)) {
    console.error('  ✗ Missing required file: config.json');
    allPresent = false;
  } else {
    console.log('  ✓ config.json');
  }

  if (weightFiles.length === 0) {
    console.error('  ✗ No weight files found');
    allPresent = false;
  }

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

export async function run(argv: string[]) {
  const { values: args } = parseArgs({
    args: argv,
    options: {
      model: {
        type: 'string',
        short: 'm',
        default: DEFAULT_MODEL,
      },
      output: {
        type: 'string',
        short: 'o',
      },
      help: {
        type: 'boolean',
        short: 'h',
        default: false,
      },
      'set-token': {
        type: 'boolean',
        default: false,
      },
    },
  });

  if (args.help) {
    printHelp();
    return;
  }

  if (args['set-token']) {
    await setToken();
    return;
  }

  const modelName = args.model!;
  const modelSlug = modelName.split('/').pop()!.toLowerCase();
  const outputDir = resolve(args.output ?? join('.cache', 'models', modelSlug));

  const HUGGINGFACE_TOKEN = (await keyringEntry.getPassword()) ?? undefined;

  if (!HUGGINGFACE_TOKEN) {
    console.warn('No HuggingFace token found, the model will download with anonymous access');
  }

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

  console.log('Note: After download, convert to MLX format:');
  console.log(`    mlx convert --input ${outputDir} --output ${outputDir}-mlx-bf16\n`);

  // Check if already downloaded
  if (existsSync(outputDir)) {
    const files = await readdir(outputDir);
    const hasConfig = files.includes('config.json');
    const hasSingleModel = files.includes('model.safetensors');
    const hasShardedModel = files.includes('model.safetensors.index.json');
    const hasPaddleModel = files.includes('inference.pdiparams');
    if (hasConfig && (hasSingleModel || hasShardedModel || hasPaddleModel)) {
      console.log('Model already downloaded!\n');
      console.log('To re-download, delete the output directory first:');
      console.log(`   rm -rf ${outputDir}\n`);
      return;
    }
  }

  await ensureDir(outputDir);

  console.log('Downloading base model from HuggingFace...\n');

  const { totalSize, filesToDownload } = await getModelFiles(modelName, HUGGINGFACE_TOKEN);
  const sizeStr = formatBytes(totalSize);
  console.log(`This may take a while (model is ~${sizeStr})...\n`);

  const cacheDir = join(process.cwd(), '.cache', 'huggingface');
  const weightFiles: string[] = [];

  for (const file of filesToDownload) {
    const snapshotPath = await downloadFileToCacheDir({
      repo: { type: 'model', name: modelName },
      path: file.path,
      cacheDir,
      accessToken: HUGGINGFACE_TOKEN,
    });
    const stats = await stat(snapshotPath);
    const fileSizeStr = formatBytes(stats.size);
    const destPath = join(outputDir, file.path);
    await ensureDir(dirname(destPath));
    await copyFile(snapshotPath, destPath);
    if (file.path.endsWith('.safetensors') || file.path.endsWith('.pdiparams')) {
      weightFiles.push(file.path);
    }
    console.log(`  ✓ ${file.path} (${fileSizeStr})`);
  }

  const success = await verifyDownload(outputDir, weightFiles);

  if (success) {
    console.log('\nModel downloaded successfully!\n');
  } else {
    console.error('\nDownload incomplete. Please try again.\n');
    process.exit(1);
  }
}
