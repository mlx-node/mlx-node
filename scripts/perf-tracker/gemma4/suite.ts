/// <reference types="node" />

import { join } from 'node:path';

import {
  LLAMA_SINGLE_REQUEST_ARGS,
  MODELS_DIR,
  ROOT,
  loadFixture,
  readJson,
  type DraftPins,
  type FixtureCase,
  type FixtureManifest,
  type MlxSpeculationRequest,
  type SpeculationRule,
} from '../common.js';

export const FIXTURE = 'gemma4-oxc-review-v2';
export const MODEL_DIR = join(MODELS_DIR, 'gemma-4-12b-it-qat-q4_0-gguf');
export const MODEL_FILE = join(MODEL_DIR, 'gemma-4-12b-it-qat-q4_0.gguf');
// No pin exists for the media companion; provenance records its hash.
export const MMPROJ_FILE = join(MODEL_DIR, 'mmproj-gemma-4-12b-it-qat-q4_0.gguf');
export const REPETITIONS = [1, 2, 3].map(String);
export const PILOT_RUN = 'pilot';
export const PILOT_SETTLE_MS = 60_000;

export const RUNTIMES = ['mlx', 'llama'] as const;
export type Runtime = (typeof RUNTIMES)[number];
export const LABEL = { mlx: 'mlx-node', llama: 'llama.cpp' } as const satisfies Record<Runtime, string>;

// The fixture pins the draft files; these are their local folders. The target folder keeps
// its own download marker, so the llama.cpp draft has a folder of its own.
export const DRAFT_DIRS: Readonly<Record<Runtime, string>> = {
  mlx: join(MODELS_DIR, 'dspark_gemma4_12b_block7'),
  llama: join(MODELS_DIR, 'gemma-4-12b-it-mtp-gguf'),
};
// mlx-node DSpark against llama.cpp assistant MTP: llama.cpp cannot run the DSpark draft.
export const SPECULATION: SpeculationRule = { llamaType: 'draft-mtp', mlxDraftsPerCycle: 7 };

export interface Protocol {
  repetitions: number;
  outputTokens: number;
  warmupTokens: number;
  warmupCase: string;
  cooldownSeconds: number;
  llamaServerArgs: string[];
  mlxEnv: Record<string, string>;
  mlxRequest: MlxSpeculationRequest;
}

export interface Manifest extends FixtureManifest {
  source: Record<string, string>;
  model: { file: string; sha256: string; tokenizerSha256: string; configSha256: string };
  draft: DraftPins;
  cases: (FixtureManifest['cases'][number] & { historicalMessages: number; toolResults: number })[];
  protocol: Protocol;
}

export const draftPaths = (manifest: Manifest, runtime: Runtime): string[] =>
  manifest.draft[LABEL[runtime]].files.map(({ file }) => join(DRAFT_DIRS[runtime], file));

export const llamaServerArgs = (manifest: Manifest): string[] => [
  ...manifest.protocol.llamaServerArgs,
  '-md',
  ...draftPaths(manifest, 'llama'),
  ...LLAMA_SINGLE_REQUEST_ARGS,
];

export const manifestPath = join(ROOT, 'scripts/fixtures', `${FIXTURE}.json`);

export async function readManifest(): Promise<Manifest> {
  const manifest = await readJson<Manifest>(manifestPath);
  const { protocol } = manifest;
  if (
    protocol.repetitions !== REPETITIONS.length ||
    !manifest.cases.some((item) => item.name === protocol.warmupCase) ||
    !Number.isInteger(protocol.outputTokens) ||
    !Number.isInteger(protocol.warmupTokens) ||
    protocol.mlxRequest.mtpDepth !== SPECULATION.mlxDraftsPerCycle ||
    manifest.draft['llama.cpp'].files.length !== 1
  ) {
    throw new Error(`${manifestPath}: protocol does not match this runner`);
  }
  return manifest;
}

// loadFixture checks the payload and token-ID hashes; the per-case counts and
// recorded hashes are checked here.
export async function readFixture(): Promise<{ manifest: Manifest; path: string; cases: FixtureCase[] }> {
  const manifest = await readManifest();
  const { path, cases } = await loadFixture(FIXTURE);
  if (cases.length !== manifest.cases.length) throw new Error(`${path}: case count mismatch`);
  for (const [index, expected] of manifest.cases.entries()) {
    const actual = cases[index];
    if (
      actual.promptTokens !== expected.promptTokens ||
      actual.tokenIds.length !== expected.promptTokens ||
      actual.sha256 !== expected.tokenIdsSha256
    ) {
      throw new Error(`${expected.name}: pinned prompt length or token-ID hash mismatch`);
    }
  }
  return { manifest, path, cases };
}

export const sampleId = (runtime: Runtime, name: string, run: string) => `${runtime}-${name}-${run}`;
export const samplePath = (out: string, id: string) => join(out, 'raw', `${id}.json`);
