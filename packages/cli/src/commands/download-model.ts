import { existsSync } from 'node:fs';
import { readdir, copyFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { parseArgs } from 'node:util';

import { listFiles, whoAmI, downloadFileToCacheDir, type ListFileEntry } from '@huggingface/hub';
import { input } from '@inquirer/prompts';
import { AsyncEntry } from '@napi-rs/keyring';

import { ensureDir, formatBytes } from '../utils.js';

const DEFAULT_CACHE_DIR = join(homedir(), '.cache', 'huggingface');

const DEFAULT_MODEL = 'Qwen/Qwen3-0.6B';

const keyringEntry = new AsyncEntry('mlx-node', 'huggingface-token');

function printHelp(): void {
  console.log(`
Download a model from HuggingFace

Usage:
  mlx download model [options]

Options:
  -m, --model <name>      HuggingFace model name (default: ${DEFAULT_MODEL})
  -o, --output <dir>      Output directory (default: .cache/models/<model-slug>)
  -g, --glob <pattern>    Filter files by glob pattern (can be repeated)
  --cache-dir <dir>       HuggingFace cache directory (default: ~/.cache/huggingface)
  -h, --help              Show this help message
  --set-token             Set HuggingFace token

Glob Filtering:
  Use --glob to download only specific files from a repo. This is especially
  useful for GGUF repos that contain many quantization variants. Patterns use
  simple wildcard matching (* matches any characters).

  Multiple --glob flags can be combined; a file is included if it matches ANY
  of the patterns.

Examples:
  mlx download model
  mlx download model --model Qwen/Qwen3-1.7B --output .cache/models/qwen3-1.7b

  # Download only the BF16 GGUF variant
  mlx download model -m unsloth/Qwen3.5-9B-GGUF -g "*BF16*"

  # Download only Q4_K_M and Q8_0 variants
  mlx download model -m unsloth/Qwen3.5-9B-GGUF -g "*Q4_K_M*" -g "*Q8_0*"

  # Download all .gguf files (skip everything else)
  mlx download model -m unsloth/Qwen3.5-9B-GGUF -g "*.gguf"
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

/** Convert a simple glob pattern (with * wildcards) to a RegExp */
function globToRegex(pattern: string): RegExp {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
  return new RegExp(`^${escaped}$`, 'i');
}

/** Check if a filename matches any of the glob patterns */
function matchesAnyGlob(filename: string, patterns: RegExp[]): boolean {
  return patterns.some((re) => re.test(filename));
}

async function getModelFiles(modelName: string, accessToken?: string, globPatterns?: string[]) {
  let totalSize = 0;
  const filesToDownload: ListFileEntry[] = [];
  const allFiles: ListFileEntry[] = [];

  // Compile glob patterns if provided
  const globs = globPatterns?.map(globToRegex);

  for await (const file of listFiles({ repo: { type: 'model', name: modelName }, accessToken })) {
    allFiles.push(file);

    if (globs) {
      // When glob patterns are active, include files that match the pattern
      // OR are essential metadata files (config, tokenizer)
      const basename = file.path.split('/').pop() || file.path;
      if (matchesAnyGlob(basename, globs) || matchesAnyGlob(file.path, globs)) {
        filesToDownload.push(file);
        if (file.size) totalSize += file.size;
      } else if (CORE_FILES.includes(basename)) {
        // Always include core config/tokenizer files
        filesToDownload.push(file);
        if (file.size) totalSize += file.size;
      }
    } else {
      // Default behavior: download model files
      if (
        CORE_FILES.includes(file.path) ||
        file.path.endsWith('.safetensors') ||
        file.path.endsWith('.json') ||
        file.path.endsWith('.pdiparams') ||
        file.path.endsWith('.yml') ||
        file.path.endsWith('.gguf') ||
        file.path.endsWith('.jinja')
      ) {
        filesToDownload.push(file);
        if (file.size) {
          totalSize += file.size;
        }
      }
    }
  }

  return { totalSize, filesToDownload, allFiles };
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
      glob: {
        type: 'string',
        short: 'g',
        multiple: true,
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
      'cache-dir': {
        type: 'string',
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
  const globPatterns = args.glob;
  const modelSlug = modelName.split('/').pop()!.toLowerCase();
  const outputDir = resolve(args.output ?? join('.cache', 'models', modelSlug));

  const HUGGINGFACE_TOKEN = (await keyringEntry.getPassword()) ?? undefined;

  if (!HUGGINGFACE_TOKEN) {
    console.warn('No HuggingFace token found, the model will download with anonymous access');
  }

  const title = `${modelName} Model Download from HuggingFace`;
  const boxWidth = Math.max(title.length + 6, 58);
  const padding = Math.floor((boxWidth - title.length - 2) / 2);
  const rightPadding = boxWidth - title.length - padding;
  console.log('╔' + '═'.repeat(boxWidth) + '╗');
  console.log('║' + ' '.repeat(padding) + title + ' '.repeat(rightPadding) + '║');
  console.log('╚' + '═'.repeat(boxWidth) + '╝\n');

  console.log(`Model: ${modelName}`);
  if (globPatterns?.length) {
    console.log(`Filter: ${globPatterns.join(', ')}`);
  }
  console.log(`Output: ${outputDir}\n`);

  // Check if already downloaded
  if (existsSync(outputDir)) {
    const files = await readdir(outputDir);
    const hasConfig = files.includes('config.json');
    const hasSingleModel = files.includes('model.safetensors');
    const hasShardedModel = files.includes('model.safetensors.index.json');
    const hasPaddleModel = files.includes('inference.pdiparams');
    const hasGguf = files.some((f) => f.endsWith('.gguf'));
    if (hasConfig && (hasSingleModel || hasShardedModel || hasPaddleModel)) {
      console.log('Model already downloaded!\n');
      console.log('To re-download, delete the output directory first:');
      console.log(`   rm -rf ${outputDir}\n`);
      return;
    }
    if (hasGguf && !globPatterns?.length) {
      console.log('GGUF file(s) already downloaded!\n');
      console.log('To re-download, delete the output directory first:');
      console.log(`   rm -rf ${outputDir}\n`);
      return;
    }
    // For glob downloads, check if all glob-matched files are present
    if (hasGguf && globPatterns?.length) {
      const globs = globPatterns.map(globToRegex);
      const matchedExisting = files.filter((f) => matchesAnyGlob(f, globs) || CORE_FILES.includes(f));
      if (matchedExisting.length > 1) {
        console.log('Matched files already downloaded!\n');
        console.log('To re-download, delete the output directory first:');
        console.log(`   rm -rf ${outputDir}\n`);
        return;
      }
    }
  }

  await ensureDir(outputDir);

  console.log('Fetching file list from HuggingFace...\n');

  const { totalSize, filesToDownload, allFiles } = await getModelFiles(modelName, HUGGINGFACE_TOKEN, globPatterns);

  if (filesToDownload.length === 0) {
    console.error('No files matched the given criteria.\n');
    if (globPatterns?.length) {
      const ggufFiles = allFiles.filter((f) => f.path.endsWith('.gguf'));
      if (ggufFiles.length > 0) {
        console.log('Available GGUF files in this repo:');
        for (const f of ggufFiles) {
          console.log(`  ${f.path} (${formatBytes(f.size)})`);
        }
        console.log(`\nTry: mlx download model -m ${modelName} -g "<pattern>"`);
      }
    }
    process.exit(1);
  }

  // Show what will be downloaded
  if (globPatterns?.length) {
    console.log(`Matched ${filesToDownload.length} file(s):`);
    for (const f of filesToDownload) {
      console.log(`  ${f.path} (${formatBytes(f.size)})`);
    }
    console.log('');
  }

  const sizeStr = formatBytes(totalSize);
  console.log(`Downloading ${filesToDownload.length} file(s) (~${sizeStr})...\n`);

  const cacheDir = args['cache-dir'] ? resolve(args['cache-dir']) : DEFAULT_CACHE_DIR;
  const weightFiles: string[] = [];

  const total = filesToDownload.length;
  for (let i = 0; i < total; i++) {
    const file = filesToDownload[i];
    const fileSizeStr = file.size ? formatBytes(file.size) : '';
    console.log(`  [${i + 1}/${total}] ${file.path}${fileSizeStr ? ` (${fileSizeStr})` : ''}...`);
    const snapshotPath = await downloadFileToCacheDir({
      repo: { type: 'model', name: modelName },
      path: file.path,
      cacheDir,
      accessToken: HUGGINGFACE_TOKEN,
    });
    const destPath = join(outputDir, file.path);
    await ensureDir(dirname(destPath));
    await copyFile(snapshotPath, destPath);
    if (file.path.endsWith('.safetensors') || file.path.endsWith('.pdiparams') || file.path.endsWith('.gguf')) {
      weightFiles.push(file.path);
    }
  }

  // For GGUF downloads, skip strict verification (no config.json required in GGUF repos)
  const hasGgufFiles = weightFiles.some((f) => f.endsWith('.gguf'));
  if (hasGgufFiles) {
    console.log(`\nDownload complete! ${weightFiles.length} file(s) saved to ${outputDir}\n`);
    console.log('To convert GGUF to MLX SafeTensors format:');
    for (const wf of weightFiles) {
      const ggufPath = join(outputDir, wf);
      console.log(`  mlx convert -i ${ggufPath} -o ${outputDir}-mlx`);
    }
    console.log('');
  } else if (weightFiles.length === 0 && globPatterns?.length) {
    if (filesToDownload.length === 0) {
      console.error(`\nNo files matched the glob pattern(s): ${globPatterns.join(', ')}`);
      console.error('Check the pattern and available files in the repository.');
      process.exit(1);
    }
    // Glob filter matched non-weight files (e.g. imatrix, calibration data).
    // Skip model verification — user is downloading auxiliary files.
    console.log(`\nDownload complete! ${filesToDownload.length} non-weight file(s) saved to ${outputDir}\n`);
  } else {
    console.log(`Format: Base model (needs MLX conversion)`);
    console.log('Note: After download, convert to MLX format:');
    console.log(`    mlx convert --input ${outputDir} --output ${outputDir}-mlx-bf16\n`);

    const success = await verifyDownload(outputDir, weightFiles);
    if (success) {
      console.log('\nModel downloaded successfully!\n');
    } else {
      console.error('\nDownload incomplete. Please try again.\n');
      process.exit(1);
    }
  }
}
