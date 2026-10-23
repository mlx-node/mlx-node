#!/usr/bin/env oxnode

/// <reference types="node" />

import { readFile, mkdir, stat, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';

// Capture every position with bounded top-K storage. The paired Splash run
// consumes the exact uint32 IDs saved in this cache, never re-tokenized text.
const [binding, target, textFile, cacheDirectory, sequenceLength = '2048', topK = '256'] = process.argv.slice(2);
if (!binding || !target || !textFile || !cacheDirectory || process.argv.length > 8) {
  throw new Error(
    'Usage: oxnode capture-teacher.ts addon.node native-model-directory text.txt cache-dir [sequence-length=2048] [top-k=256]',
  );
}
for (const value of [sequenceLength, topK]) {
  if (!/^\d+$/.test(value) || Number(value) < 2) throw new Error('Lengths must be positive integers >= 2');
}
if (Number(sequenceLength) > 4096)
  throw new Error('Capture is bounded to 4096 tokens; use separate windows for longer-context work');
if (!(await stat(target)).isDirectory())
  throw new Error(
    'The quality API takes a native model directory. For GGUF use modelAssetsPath recorded by benchmark.ts after loading the exact file.',
  );
const core: typeof import('../../../packages/core/index.cjs') = createRequire(import.meta.url)(resolve(binding));
const text = await readFile(textFile, 'utf8');
await mkdir(cacheDirectory, { recursive: true });
const started = performance.now();
const rows = await core.captureTeacherLogits(
  resolve(target),
  [text],
  Number(sequenceLength),
  Number(topK),
  16,
  resolve(cacheDirectory),
);
await writeFile(
  `${cacheDirectory}/capture-run.json`,
  JSON.stringify(
    {
      binding: resolve(binding),
      target: resolve(target),
      textFile: resolve(textFile),
      sequenceLength: Number(sequenceLength),
      topK: Number(topK),
      logitChunk: 16,
      rows,
      wallMs: performance.now() - started,
      memory: core.memoryStats(),
    },
    null,
    2,
  ) + '\n',
);
console.log(`Captured ${rows} teacher row(s) into ${cacheDirectory}`);
process.exit(0);
