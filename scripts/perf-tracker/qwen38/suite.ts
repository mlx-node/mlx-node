/// <reference types="node" />

import { appendFileSync, existsSync } from 'node:fs';
import { mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  LLAMA_SINGLE_REQUEST_ARGS,
  MODELS_DIR,
  ROOT,
  fileHashes,
  llamaAcceptance,
  mlxAcceptance,
  readJson,
  sh,
  sha256,
  writeJson,
  type Acceptance,
  type DraftPins,
  type FixtureCase,
  type FixtureManifest,
  type LlamaIdentity,
  type MlxSpeculationPerformance,
  type MlxSpeculationRequest,
  type SpeculationRule,
  type TreeIdentity,
} from '../common.js';

export const HERE = dirname(fileURLToPath(import.meta.url));
export const FIXTURE = 'qwen38-oxc-review-v2';
export const MODEL_PATH = join(MODELS_DIR, 'qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf');
// The fixture pins the draft files; these are their local folders.
export const DRAFT_DIRS: Readonly<Record<Runtime, string>> = {
  'mlx-node': join(MODELS_DIR, 'qwen3.8-27b-dflash2'),
  'llama.cpp': join(MODELS_DIR, 'qwen3.8-27b-dflash2-gguf'),
};
// The target GGUF also carries an inline MTP head, which drafts one token per cycle.
export const SPECULATION: SpeculationRule = { llamaType: 'draft-dflash', mlxMinMeanDepth: 1 };
export const PORT = 18789;
export const LOAD_COOLDOWN_SECONDS = 10;
export const SIDECARS = ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'];

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

// --desktop-load applies the 32,488-token rule to every row, for a desktop that never reaches GPU idle.
export function admissionPolicy(desktopLoad: boolean): Readonly<Record<number, Policy>> {
  if (!desktopLoad) return POLICY;
  return Object.fromEntries(Object.keys(POLICY).map((tokens) => [tokens, 'no-other-local-llm' as const]));
}

export interface Protocol {
  repetitions: number;
  outputTokens: number;
  warmupTokens: number;
  cooldownSeconds: number;
  llamaServerArgs: string[];
  mlxEnv: Record<string, string>;
  mlxLoadOptions: { kvFormat: 'bf16' };
  mlxRequest: MlxSpeculationRequest;
  admissionPolicyByPromptTokens: Record<string, string>;
}

export interface Manifest extends FixtureManifest {
  model: { file: string; bytes: number; sha256: string };
  draft: DraftPins;
  cases: (FixtureManifest['cases'][number] & { sourceLine: number })[];
  protocol: Protocol;
}

export const draftPaths = (manifest: Manifest, runtime: Runtime): string[] =>
  manifest.draft[runtime].files.map(({ file }) => join(DRAFT_DIRS[runtime], file));

export const llamaServerArgs = (manifest: Manifest): string[] => [
  ...manifest.protocol.llamaServerArgs,
  '-md',
  ...draftPaths(manifest, 'llama.cpp'),
  ...LLAMA_SINGLE_REQUEST_ARGS,
];

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
  if (manifest.draft['llama.cpp'].files.length !== 1) throw new Error(`${FIXTURE}: llama.cpp takes one draft GGUF`);
  return manifest;
}

export interface Performance extends MlxSpeculationPerformance {
  ttftMs: number;
  prefillTokensPerSecond: number;
  decodeTokensPerSecond: number;
}

export interface Speculated {
  runtime: Runtime;
  name: string;
  run: number | string;
  performance: Performance;
  timings?: { draft_n?: number; draft_n_accepted?: number };
  speculativeTypes?: string;
}

// Throws for a sample that ran without speculation.
export function sampleAcceptance(sample: Speculated, rule: SpeculationRule): Acceptance {
  const label = `${sample.runtime} ${sample.name} run ${sample.run}`;
  return sample.runtime === 'mlx-node'
    ? mlxAcceptance(sample.performance, rule, label)
    : llamaAcceptance(sample.timings ?? {}, sample.speculativeTypes ?? '', rule, label);
}

export interface Sample extends Speculated {
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
  block?: string;
  [field: string]: unknown;
}

export interface FileIdentity {
  path: string;
  realpath: string;
  size: number;
  mtimeMs: number;
  ino: number;
  sha256?: string;
}

// Everything a fresh runtime process reads; file hashes are absent from the per-block check.
export interface Identity {
  tree: TreeIdentity;
  addon: Record<string, string>;
  llama: LlamaIdentity;
  model: FileIdentity;
  drafts: Record<Runtime, FileIdentity[]>;
  sidecars: Record<string, string>;
  scripts: Record<string, string>;
}

export const IDENTITY_KEYS = ['tree', 'addon', 'llama', 'model', 'drafts', 'sidecars', 'scripts'] as const;

export interface Provenance extends Identity {
  host: { os: string; chip: string; gpuCores: number; memoryBytes: number };
  tokenizer: { verified: boolean };
  protocol: { admission: Record<string, Policy>; desktopLoad: boolean; llamaServerArgs: string[] };
  matrix?: { startedAt: string; finishedAt?: string; postRunVerified?: boolean };
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

export async function verifyTokenizer(tokenizer: string) {
  const output = sh('oxnode', [
    join(ROOT, 'scripts/benchmark-fixture.ts'),
    'verify',
    '--fixture',
    'qwen38',
    '--tokenizer',
    tokenizer,
  ]);
  if (!output.includes('Current native tokenizer/template matches all pinned inputs.')) {
    throw new Error(`Tokenizer verification did not run:\n${output}`);
  }
  return { tokenizer, files: await fileHashes(dirname(tokenizer), SIDECARS), verified: true, output };
}

export interface IdCheck {
  name: string;
  promptTokens: number;
  identical: boolean;
  trace: string;
}

// The admitted-prompt trace hashes each u32 token ID little-endian.
export async function checkAdmittedIds(trace: string, cases: readonly FixtureCase[], out: string): Promise<IdCheck[]> {
  const admitted = (await readFile(trace, 'utf8')).split('\n').filter((line) => line.includes('admitted_prompt'));
  if (admitted.length !== cases.length)
    throw new Error(`Traced ${admitted.length} admitted prompts for ${cases.length}`);
  const checks = cases.map((item, index) => {
    const ids = Buffer.alloc(item.tokenIds.length * 4);
    item.tokenIds.forEach((id, offset) => ids.writeUInt32LE(id, offset * 4));
    const identical = admitted[index].includes(`is_delta=false tokens=${item.tokenIds.length} sha256=${sha256(ids)}`);
    return { name: item.name, promptTokens: item.tokenIds.length, identical, trace: admitted[index] };
  });
  await writeJson(join(out, 'check-ids.json'), checks);
  for (const check of checks) {
    if (!check.identical)
      throw new Error(`${check.name}: native session admitted different prompt IDs: ${check.trace}`);
  }
  return checks;
}

export async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await writeFile(`${path}.tmp`, `${JSON.stringify(value, null, 2)}\n`);
  await rename(`${path}.tmp`, path);
}

// A cell is one runtime on one case: one process, its warmup and every repetition. A resume moves
// a partly done cell to discarded/<stamp>/, so the cell reruns in full. Its files are found on disk,
// not only through samples.json: a crash after a result file is written but before its row is
// added would otherwise leave a file that the rerun overwrites.
export async function discardPartialCells(
  out: string,
  samples: readonly Sample[],
  repetitions: number,
  stamp: string,
): Promise<Sample[]> {
  const cell = (runtime: Runtime, name: string) => `${runtime}/${name}`;
  const counts = new Map<string, number>();
  for (const { runtime, name } of samples) {
    counts.set(cell(runtime, name), (counts.get(cell(runtime, name)) ?? 0) + 1);
  }
  const incomplete = (runtime: Runtime, name: string) => (counts.get(cell(runtime, name)) ?? 0) < repetitions;
  const prefixes = [
    ['raw', { mlx: 'mlx-node', llama: 'llama.cpp' }],
    ['system', { [KIND['mlx-node']]: 'mlx-node', [KIND['llama.cpp']]: 'llama.cpp' }],
  ] as const satisfies readonly (readonly [string, Record<string, Runtime>])[];
  const moves: [string, string][] = [];
  for (const [folder, runtimes] of prefixes) {
    if (!existsSync(join(out, folder))) continue;
    for (const file of await readdir(join(out, folder))) {
      const match = /^([A-Za-z]+)-(\d+)-(.+)\.json$/.exec(file);
      const runtime = match ? (runtimes as Record<string, Runtime>)[match[1]] : undefined;
      if (match && runtime && incomplete(runtime, match[3])) moves.push([folder, file]);
    }
  }
  const partial = samples.filter(({ runtime, name }) => incomplete(runtime, name));
  if (!partial.length && !moves.length) return [...samples];
  const dir = join(out, 'discarded', stamp);
  await writeJson(join(dir, 'samples.json'), partial);
  for (const [folder, file] of moves) {
    await mkdir(join(dir, folder), { recursive: true });
    await rename(join(out, folder, file), join(dir, folder, file));
  }
  const kept = samples.filter((sample) => !partial.includes(sample));
  await writeJsonAtomic(paths(out).samples, kept);
  return kept;
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
