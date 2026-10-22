#!/usr/bin/env oxnode
/// <reference types="node" />

import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, open, rename, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { assertPin, readJson, sha256 } from '../common.js';
import { assertInternalSsd, split } from './checks.js';
import { DEFAULT_MTP_AUX_DIR, DEFAULT_MTP_SOURCE_DIR, MTP_AUX } from './suite.js';

interface Descriptor {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

interface Header {
  dataStart: number;
  tensors: Record<string, Descriptor>;
}

async function readHeader(path: string): Promise<Header> {
  const file = await open(path, 'r');
  try {
    const prefix = Buffer.alloc(8);
    if ((await file.read(prefix, 0, 8, 0)).bytesRead !== 8) throw new Error(`${path}: truncated header`);
    const length = Number(prefix.readBigUInt64LE());
    const json = Buffer.alloc(length);
    if ((await file.read(json, 0, length, 8)).bytesRead !== length) throw new Error(`${path}: truncated header`);
    return { dataStart: 8 + length, tensors: JSON.parse(json.toString('utf8')) as Record<string, Descriptor> };
  } finally {
    await file.close();
  }
}

const { values } = parseArgs({
  options: {
    source: { type: 'string' },
    out: { type: 'string' },
    help: { type: 'boolean', short: 'h' },
  },
});
if (values.help) {
  console.log(
    'Usage: oxnode scripts/perf-tracker/flash-next/mtp-aux.ts [--source <HF checkpoint folder>] [--out <internal SSD folder>]\n' +
      `Copies the ${MTP_AUX.tensors} mtp.* tensors of ${MTP_AUX.repo}@${MTP_AUX.revision.slice(0, 8)} byte for byte into\n` +
      '<out>/model.safetensors, writes <out>/config.json without vision_config, and checks both against the pins.\n' +
      `Defaults: --source ${DEFAULT_MTP_SOURCE_DIR}, --out ${DEFAULT_MTP_AUX_DIR}.`,
  );
  process.exit(0);
}
const source = resolve(values.source ?? DEFAULT_MTP_SOURCE_DIR);
const out = resolve(values.out ?? DEFAULT_MTP_AUX_DIR);

// The pins on the two output files are the source check; only the shards that hold mtp.* are read.
const index = await readJson<{ weight_map: Record<string, string> }>(join(source, 'model.safetensors.index.json'));
const names = Object.keys(index.weight_map)
  .filter((name) => name.startsWith('mtp.'))
  .sort();
if (names.length !== MTP_AUX.tensors)
  throw new Error(`${source}: ${names.length} mtp.* tensors, expected ${MTP_AUX.tensors}`);

const headers = new Map<string, Header>();
const copies: { shard: string; start: number; bytes: number }[] = [];
const tensors: Record<string, Descriptor> = {};
let offset = 0;
for (const name of names) {
  const shard = join(source, index.weight_map[name]);
  if (!headers.has(shard)) headers.set(shard, await readHeader(shard));
  const { dataStart, tensors: shardTensors } = headers.get(shard)!;
  const descriptor = shardTensors[name];
  if (!descriptor) throw new Error(`${shard}: no ${name}`);
  const bytes = descriptor.data_offsets[1] - descriptor.data_offsets[0];
  copies.push({ shard, start: dataStart + descriptor.data_offsets[0], bytes });
  tensors[name] = { dtype: descriptor.dtype, shape: descriptor.shape, data_offsets: [offset, offset + bytes] };
  offset += bytes;
}

await mkdir(out, { recursive: true });
assertInternalSsd([out]);
const header = Buffer.from(
  JSON.stringify({ __metadata__: { format: 'pt', source: `${MTP_AUX.repo}@${MTP_AUX.revision}` }, ...tensors }),
);
// The safetensors format pads the header with spaces to 8 bytes.
const paddedHeader = Buffer.concat([header, Buffer.alloc((8 - (header.length % 8)) % 8, ' ')]);
const prefix = Buffer.alloc(8);
prefix.writeBigUInt64LE(BigInt(paddedHeader.length));

const weightsPath = join(out, 'model.safetensors');
const partial = `${weightsPath}.partial`;
const hash = createHash('sha256');
const file = await open(partial, 'w');
try {
  for (const chunk of [prefix, paddedHeader]) {
    hash.update(chunk);
    await file.write(chunk);
  }
  for (const { shard, start, bytes } of copies) {
    for await (const chunk of createReadStream(shard, { start, end: start + bytes - 1 })) {
      hash.update(chunk as Buffer);
      await file.write(chunk as Buffer);
    }
  }
} finally {
  await file.close();
}
const config = await readJson<Record<string, unknown>>(join(source, 'config.json'));
delete config.vision_config;
const configText = `${JSON.stringify(config, null, 2)}\n`;

const configPath = join(out, MTP_AUX.files[0].file);
const built = [
  { path: configPath, size: Buffer.byteLength(configText), sha256: sha256(configText) },
  { path: weightsPath, size: 8 + paddedHeader.length + offset, sha256: hash.digest('hex') },
];
try {
  for (const [index, pin] of MTP_AUX.files.entries())
    assertPin(built[index].path, built[index].size, built[index].sha256, pin);
} catch (error) {
  await rm(partial, { force: true });
  console.error(JSON.stringify({ event: 'mtp_aux_mismatch', built }));
  throw error;
}
await writeFile(configPath, configText);
await rename(partial, weightsPath);
const files = [await split(configPath, true), await split(weightsPath, true)];
for (const [index, pin] of MTP_AUX.files.entries())
  assertPin(files[index].path, files[index].size, files[index].sha256, pin);
console.log(
  JSON.stringify({
    event: 'mtp_aux_ready',
    source: `${MTP_AUX.repo}@${MTP_AUX.revision}`,
    tensors: names.length,
    files,
  }),
);
