#!/usr/bin/env oxnode

/// <reference types="node" />

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { gunzipSync } from 'node:zlib';

interface FixtureCase {
  name: string;
  promptTokens: number;
  sha256: string;
  tokenIds: number[];
  messages: unknown[];
  tools: unknown[];
  rendered: string;
}

interface Manifest {
  id: string;
  storage: {
    accountId: string;
    bucket: string;
    key: string;
    bytes: number;
    sha256: string;
  };
  payload: { path: string; bytes: number; sha256: string };
  cases: { name: string; promptTokens: number; tokenIdsSha256: string }[];
}

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const hash = (data: Buffer | string) => createHash('sha256').update(data).digest('hex');

function verifyBytes(data: Buffer, expected: { bytes: number; sha256: string }, label: string) {
  if (data.length !== expected.bytes || hash(data) !== expected.sha256) {
    throw new Error(`${label}: pinned byte length or SHA-256 mismatch`);
  }
}

function verifyPayload(data: Buffer, manifest: Manifest): FixtureCase[] {
  verifyBytes(data, manifest.payload, 'Fixture');
  const cases: FixtureCase[] = JSON.parse(data.toString('utf8'));
  if (cases.length !== manifest.cases.length) {
    throw new Error('Fixture case count mismatch');
  }
  for (const [index, expected] of manifest.cases.entries()) {
    const actual = cases[index];
    if (
      actual.name !== expected.name ||
      actual.promptTokens !== expected.promptTokens ||
      actual.tokenIds.length !== expected.promptTokens ||
      actual.sha256 !== expected.tokenIdsSha256 ||
      hash(JSON.stringify(actual.tokenIds)) !== expected.tokenIdsSha256
    ) {
      throw new Error(`${expected.name}: pinned token IDs mismatch`);
    }
  }
  return cases;
}

async function fetchPayload(manifest: Manifest): Promise<Buffer> {
  const temporary = await mkdtemp(join(tmpdir(), 'mlx-benchmark-fixture-'));
  const compressed = join(temporary, 'inputs.json.gz');
  try {
    await new Promise<void>((done, reject) => {
      const child = spawn(
        'wrangler',
        ['r2', 'object', 'get', `${manifest.storage.bucket}/${manifest.storage.key}`, '--remote', '--file', compressed],
        {
          cwd: temporary,
          env: {
            ...process.env,
            CLOUDFLARE_ACCOUNT_ID: manifest.storage.accountId,
          },
          stdio: ['ignore', 'inherit', 'inherit'],
        },
      );
      child.once('error', reject);
      child.once('exit', (code, signal) => {
        if (code === 0) done();
        else reject(new Error(`Wrangler download failed: ${signal ?? code}`));
      });
    });
    const archive = await readFile(compressed);
    verifyBytes(archive, manifest.storage, 'Downloaded archive');
    const data = gunzipSync(archive, {
      maxOutputLength: manifest.payload.bytes,
    });
    verifyPayload(data, manifest);
    return data;
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

async function main() {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      output: { type: 'string' },
      tokenizer: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) {
    console.log(
      'Usage: oxnode scripts/benchmark-fixture.ts fetch|verify [--output inputs.json] [--tokenizer tokenizer.json]\n' +
        'fetch: authenticated R2 download, or verify an existing copy; never overwrite a different fixture.\n' +
        'verify: offline hashes; optionally check the current native chat template against every pinned token ID.',
    );
    return;
  }
  const [mode] = positionals;
  if (positionals.length !== 1 || !['fetch', 'verify'].includes(mode)) {
    throw new Error('Expected fetch or verify; use --help for usage');
  }
  const manifest: Manifest = JSON.parse(
    await readFile(resolve(root, 'scripts/fixtures/gemma4-oxc-review-v1.json'), 'utf8'),
  );
  const output = values.output ? resolve(values.output) : resolve(root, manifest.payload.path);
  let data: Buffer;
  try {
    data = await readFile(output);
  } catch (error) {
    if (mode !== 'fetch' || (error as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw error;
    }
    data = await fetchPayload(manifest);
    await mkdir(dirname(output), { recursive: true });
    await writeFile(output, data, { flag: 'wx', mode: 0o600 });
  }
  const cases = verifyPayload(data, manifest);
  if (values.tokenizer) {
    const coreUrl = pathToFileURL(resolve(root, 'packages/core/index.cjs'));
    const { Qwen3Tokenizer } = await import(coreUrl.href);
    const tokenizer = await Qwen3Tokenizer.fromPretrained(resolve(values.tokenizer));
    for (const item of cases) {
      const ids = await tokenizer.applyChatTemplate(item.messages, true, item.tools, true);
      if (
        hash(JSON.stringify(Array.from(ids))) !== item.sha256 ||
        (await tokenizer.decode(ids, false)) !== item.rendered
      ) {
        throw new Error(`${item.name}: current tokenizer/template drifted`);
      }
    }
    console.log('Current native tokenizer/template matches all pinned inputs.');
  }
  console.log(
    `Verified ${manifest.id}: ${cases.map((item) => item.promptTokens.toLocaleString('en-US')).join(' / ')} tokens.\n${output}`,
  );
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : 'Fixture check failed');
  process.exitCode = 1;
});
