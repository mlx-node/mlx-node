import { mkdir, readdir, stat, copyFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { snapshotDownload } from '@huggingface/hub';

import { convertParquetToJsonl } from '@mlx-node/core';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DEFAULT_DATASET = 'openai/gsm8k';
const DEFAULT_REVISION = 'main';
const DEFAULT_OUTPUT_DIR = resolve(process.cwd(), 'data', 'gsm8k');
const FILE_SPECS = [
  { output: 'train.jsonl', parquetPrefix: 'train-' },
  { output: 'test.jsonl', parquetPrefix: 'test-' },
];

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

async function ensureDir(path: string): Promise<void> {
  if (!existsSync(path)) {
    await mkdir(path, { recursive: true });
  }
}

async function main() {
  const dataset = process.env.GSM8K_DATASET ?? DEFAULT_DATASET;
  const revision = process.env.GSM8K_REVISION ?? DEFAULT_REVISION;
  const outputDir = process.env.GSM8K_OUTPUT_DIR ? resolve(process.env.GSM8K_OUTPUT_DIR) : DEFAULT_OUTPUT_DIR;

  console.log(`Downloading ${dataset}@${revision} snapshot from Hugging Face…`);

  const snapshotPath = await snapshotDownload({
    repo: { type: 'dataset', name: dataset },
    revision,
    cacheDir: join(__dirname, '..', '.cache', 'huggingface'),
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
    await convertParquetToJsonl(parquetSource, destinationPath);

    const stats = await stat(destinationPath);
    console.log(`Saved ${spec.output} (${Math.round(stats.size / 1024)} KiB) → ${destinationPath}`);
  }

  console.log('Done.');
  console.log(`Dataset files stored under ${outputDir}`);
}

main().catch((error) => {
  console.error('[download-gsm8k] failed:', error);
  process.exitCode = 1;
});
