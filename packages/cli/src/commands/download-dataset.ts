import { readdir, stat, copyFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { join, dirname, resolve } from 'node:path';
import { snapshotDownload } from '@huggingface/hub';

import { convertParquetToJsonl } from '@mlx-node/core';

import { ensureDir } from '../utils.js';

const DEFAULT_DATASET = 'openai/gsm8k';
const DEFAULT_REVISION = 'main';
const DEFAULT_OUTPUT_DIR = resolve(process.cwd(), 'data', 'gsm8k');
const FILE_SPECS = [
  { output: 'train.jsonl', parquetPrefix: 'train-' },
  { output: 'test.jsonl', parquetPrefix: 'test-' },
];

function printHelp(): void {
  console.log(`
Download a dataset from HuggingFace

Usage:
  mlx download dataset [options]

Options:
  -d, --dataset <name>    HuggingFace dataset name (default: ${DEFAULT_DATASET})
  -r, --revision <rev>    Dataset revision (default: ${DEFAULT_REVISION})
  -o, --output <dir>      Output directory (default: data/gsm8k)
  -h, --help              Show this help message

Examples:
  mlx download dataset
  mlx download dataset --dataset openai/gsm8k --output data/gsm8k
`);
}

async function findFirstMatch(
  root: string,
  predicate: (name: string, fullPath: string) => boolean,
): Promise<string | null> {
  const entries = await readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = join(root, entry.name);
    if ((entry.isFile() || entry.isSymbolicLink()) && predicate(entry.name, fullPath)) {
      return fullPath;
    }
  }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const fullPath = join(root, entry.name);
    const found: string | null = await findFirstMatch(fullPath, predicate);
    if (found) return found;
  }
  return null;
}

export async function run(argv: string[]) {
  const { values: args } = parseArgs({
    args: argv,
    options: {
      dataset: {
        type: 'string',
        short: 'd',
        default: process.env.GSM8K_DATASET ?? DEFAULT_DATASET,
      },
      revision: {
        type: 'string',
        short: 'r',
        default: process.env.GSM8K_REVISION ?? DEFAULT_REVISION,
      },
      output: {
        type: 'string',
        short: 'o',
        default: process.env.GSM8K_OUTPUT_DIR ?? DEFAULT_OUTPUT_DIR,
      },
      help: {
        type: 'boolean',
        short: 'h',
        default: false,
      },
    },
  });

  if (args.help) {
    printHelp();
    return;
  }

  const dataset = args.dataset!;
  const revision = args.revision!;
  const outputDir = resolve(args.output!);

  console.log(`Downloading ${dataset}@${revision} snapshot from Hugging Face…`);

  const cacheDir = join(process.cwd(), '.cache', 'huggingface');
  const snapshotPath = await snapshotDownload({
    repo: { type: 'dataset', name: dataset },
    revision,
    cacheDir,
  });

  console.log(`Snapshot available at ${snapshotPath}`);

  await ensureDir(outputDir);

  for (const spec of FILE_SPECS) {
    const destinationPath = join(outputDir, spec.output);
    await ensureDir(dirname(destinationPath));

    const original = await findFirstMatch(snapshotPath, (name) => name === spec.output);
    if (original) {
      await copyFile(original, destinationPath);
      const stats = await stat(destinationPath);
      console.log(`Copied ${spec.output} (${Math.round(stats.size / 1024)} KiB) → ${destinationPath}`);
      continue;
    }

    const parquetSource = await findFirstMatch(
      snapshotPath,
      (name) => name.endsWith('.parquet') && name.startsWith(spec.parquetPrefix),
    );

    if (!parquetSource) {
      throw new Error(
        `Could not locate ${spec.output} or matching Parquet file (prefix ${spec.parquetPrefix}) inside snapshot ${snapshotPath}`,
      );
    }

    console.log(`Converting ${parquetSource} → ${destinationPath}`);
    convertParquetToJsonl(parquetSource, destinationPath);

    const stats = await stat(destinationPath);
    console.log(`Saved ${spec.output} (${Math.round(stats.size / 1024)} KiB) → ${destinationPath}`);
  }

  console.log('Done.');
  console.log(`Dataset files stored under ${outputDir}`);
}
