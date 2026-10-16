/// <reference types="node" />

import { appendFileSync } from 'node:fs';
import { rename, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { MODELS_DIR, ROOT, readJson, type FixtureManifest } from '../common.js';

export const HERE = dirname(fileURLToPath(import.meta.url));
export const FIXTURE = 'qwen38-oxc-review-v1';
export const MODEL_PATH = join(MODELS_DIR, 'qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf');
export const PORT = 18789;
export const LOAD_COOLDOWN_SECONDS = 10;

export const RUNTIMES = ['mlx-node', 'llama.cpp'] as const;
export type Runtime = (typeof RUNTIMES)[number];
// Labels match the 2026-09-09 record, so its raw files report unchanged.
export const KIND = { 'mlx-node': 'Mlx', 'llama.cpp': 'Llama' } as const satisfies Record<Runtime, string>;

export type Policy = 'gpu-idle' | 'no-other-local-llm';
// The 32,488-token row admits background desktop graphics but no other local LLM.
export const POLICY: Readonly<Record<number, Policy>> = {
  6219: 'gpu-idle',
  18754: 'gpu-idle',
  32488: 'no-other-local-llm',
};

export interface Protocol {
  repetitions: number;
  outputTokens: number;
  warmupTokens: number;
  cooldownSeconds: number;
  llamaServerArgs: string[];
  mlxEnv: Record<string, string>;
  admissionPolicyByPromptTokens: Record<string, string>;
}

export interface Manifest extends FixtureManifest {
  model: { file: string; bytes: number; sha256: string };
  cases: (FixtureManifest['cases'][number] & { sourceLine: number })[];
  protocol: Protocol;
  measuredRevisions: Record<Runtime, string>;
}

export async function readManifest(): Promise<Manifest> {
  const manifest = await readJson<Manifest>(join(ROOT, 'scripts/fixtures', `${FIXTURE}.json`));
  const pinned = manifest.protocol.admissionPolicyByPromptTokens;
  const agrees = (tokens: number) =>
    (POLICY[tokens] === 'gpu-idle') === pinned[tokens]?.includes('<=5% GPU utilization') &&
    (POLICY[tokens] === 'no-other-local-llm') === pinned[tokens]?.includes('no other active local LLM');
  if (
    Object.keys(pinned).length !== manifest.cases.length ||
    Object.keys(POLICY).length !== manifest.cases.length ||
    !manifest.cases.every((item) => agrees(item.promptTokens))
  ) {
    throw new Error(`${FIXTURE}: the pinned admission policy no longer matches this runner`);
  }
  return manifest;
}

export interface Performance {
  ttftMs: number;
  prefillTokensPerSecond: number;
  decodeTokensPerSecond: number;
  mtpCycles?: number;
}

export interface Sample {
  event: 'result';
  runtime: Runtime;
  name: string;
  run: number;
  pid: number;
  wallMs: number;
  firstEventMs: number | null;
  numTokens: number;
  promptTokens: number;
  cachedTokens: number;
  performance: Performance;
  [field: string]: unknown;
}

export interface Provenance {
  host: { os: string; chip: string; gpuCores: number; memoryBytes: number };
  tree: { commit: string; mlx: string; dirty: boolean };
  llama: { commit: string; sha256: string; dylibs: Record<string, string> };
  addon: Record<string, string>;
  model: { path: string; bytes: number; sha256: string };
  tokenizer: { verified: boolean };
  matrix?: { startedAt: string; finishedAt?: string };
  [field: string]: unknown;
}

export const paths = (out: string) => ({
  provenance: join(out, 'provenance.json'),
  progress: join(out, 'progress.ndjson'),
  samples: join(out, 'samples.json'),
  gpu: join(out, 'gpu.ndjson'),
  raw: join(out, 'raw'),
  system: join(out, 'system'),
});

export async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await writeFile(`${path}.tmp`, `${JSON.stringify(value, null, 2)}\n`);
  await rename(`${path}.tmp`, path);
}

// One NDJSON line per event; `last` feeds the GPU log's phase column.
export class Progress {
  last: Record<string, unknown> = {};

  constructor(private readonly path: string) {}

  emit(event: Record<string, unknown>): void {
    const line = JSON.stringify({ ...event, timestamp: new Date().toISOString() });
    appendFileSync(this.path, `${line}\n`);
    console.log(line);
    const { event: name, label, runtime, run } = event;
    this.last = { event: name, label, runtime, run, name: event.name };
  }
}
