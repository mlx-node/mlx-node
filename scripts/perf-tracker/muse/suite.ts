/// <reference types="node" />

import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { MODELS_DIR, ROOT } from '../common.js';

export const HERE = dirname(fileURLToPath(import.meta.url));
export const BOOTSTRAP = join(HERE, 'bootstrap.ts');
export const BASELINE = join(HERE, 'baseline.csv');
// The September measurement ran this exact file; the protocol lives in it.
export const RUNNER = join(ROOT, 'scripts/benchmark-muse-gguf.ts');
export const RUNNER_SHA256 = 'dc82e0f1ea65dd623408c68f51574e3ba5e1c180f8e3a2b9a7032e800ee089ee';
export const SCRIPTS = ['run.ts', 'suite.ts', 'guard.ts', 'bootstrap.ts', 'report.ts', 'baseline.csv', '../common.ts'];

export const TARGET = {
  path: join(MODELS_DIR, 'muse-glimmer-30b-gguf/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf'),
  sha256: 'ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38',
};
// Both engines pick this companion by name: the runner for llama.cpp and
// muse_glimmer_native_draft for MLX.
export const DRAFT = {
  path: join(MODELS_DIR, 'muse-glimmer-30b-gguf/dflash-kquant.gguf'),
  sha256: '27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc',
};
export const FIXTURE = 'gemma4-oxc-review-v1';
// The Muse-rendered inputs.json the runner writes from the fixture.
export const INPUTS_SHA256 = 'e2e5918410c78e37e2b7836557817793760181f244668bbaf3c8a57b6ef367e1';
export const SEPTEMBER = {
  commit: 'c67f354efca723fa050e188d3c39ee90241c2fe1',
  setupSha256: 'f6de18d088d59d074457d61c7aa7c62d5c8b7722ed6bab93e560525e4f77f43e',
  // From its raw samples (not in baseline.csv): every worker hit the native GGUF cache.
  maxMlxLoadMs: 470,
};

export interface Case {
  name: string;
  promptTokens: number;
  sha256: string;
}

export const CASES: Case[] = [
  {
    name: 'diff-review',
    promptTokens: 7254,
    sha256: '37e0ea613674337e3be5a3ba75848943f554b8e5aeebfe91c877507956b9dfb3',
  },
  {
    name: 'test-review',
    promptTokens: 36325,
    sha256: 'a2df6ef3c8ce3e590f057da5d02cd7bd0d57ddd17295fc22f382d0c8eb56f047',
  },
  {
    name: 'final-review',
    promptTokens: 60547,
    sha256: 'd1596ec6be60d02e13767c5d41351966825a9d414fe3277cba84a6bd98495c91',
  },
];

export const PROTOCOL = {
  outputTokens: 96,
  warmup: { case: 'shortest', maxTokens: 32 },
  repetitions: 3,
  cooldownSeconds: 20,
  draftTokens: 15,
  contextCapacity: 61_440,
  jobMinutes: 15,
};

export type Runtime = 'mlx' | 'llama';
export type Speculation = 'off' | 'dflash';

// Series order of the tables and the chart.
export const VARIANTS: { runtime: Runtime; speculation: Speculation; label: string }[] = [
  { runtime: 'mlx', speculation: 'off', label: 'MLX, off' },
  { runtime: 'llama', speculation: 'off', label: 'llama.cpp, off' },
  { runtime: 'mlx', speculation: 'dflash', label: 'MLX, DFlash' },
  { runtime: 'llama', speculation: 'dflash', label: 'llama.cpp, DFlash' },
];

export const sampleId = (runtime: Runtime, speculation: Speculation, name: string, run: number | string) =>
  `${runtime}-${speculation === 'dflash' ? 'dflash' : 'ar'}-${name}-${run}`;

export interface Job {
  id: string;
  runtime: Runtime;
  speculation: Speculation;
  case: Case;
  run: number;
}

// Forward/reverse alternation per repetition and history, as measured.
export function matrixJobs(cases: Case[]): Job[] {
  const jobs: Job[] = [];
  for (let run = 1; run <= PROTOCOL.repetitions; run++) {
    for (const [index, item] of cases.entries()) {
      const order = (run + index) % 2 ? VARIANTS : [...VARIANTS].reverse();
      for (const { runtime, speculation } of order) {
        jobs.push({ id: sampleId(runtime, speculation, item.name, run), runtime, speculation, case: item, run });
      }
    }
  }
  return jobs;
}

export interface Identity {
  path: string;
  bytes: number;
  mtimeMs: number;
  ctimeMs: number;
  sha256: string;
}

// environment.json: written by the runner's prepare, completed by finalize.
// Workers hash it into every sample's setupSha256.
export interface Setup {
  modelFile: string;
  draft: string;
  prepared: string;
  draftTokens: number;
  contextCapacity: number;
  outputTokens: number;
  warmup: { case: string; maxTokens: number };
  repetitions: number;
  cooldown: number;
  fixture: { id: string; sha256: string };
  cases: Case[];
  identities: Identity[];
  mlxCommit: string;
  mlxPatchSha256: string;
  llamaVersion: string;
  hardware: string;
  os: string;
  installedNativeUnmeasured?: Identity;
  frozenNativePath?: string;
  frozenPagedMetalPath?: string;
  llamaServer?: string;
  nativeOverride?: string;
}

export interface Sample {
  id: string;
  runtime: Runtime;
  speculation: Speculation;
  case: string;
  run: string;
  promptTokens: number;
  inputSha256: string;
  completedAt: string;
  setupSha256: string;
  thermalBefore: string;
  thermalAfter: string;
  outputTokensCap: number;
  warmup: { case: string; maxTokens: number; generatedTokens: number; wallMs: number; completedAt: string };
  measuredAt: string;
  loadMs: number;
  wallMs: number;
  prefillMs: number;
  prefillTps: number;
  decodeMs: number;
  decodeTps: number;
  generatedTokens: number;
  cachedTokens: number;
  outputSha256: string;
  result: {
    rawText?: string;
    content?: string;
    performance?: {
      mtpCycles?: number;
      mtpMeanDepth?: number;
      mtpMeanAcceptedTokens?: number;
      mtpAcceptanceByPosition?: number[];
    };
    timings?: { draft_n?: number; draft_n_accepted?: number };
  };
}

export interface Provenance {
  createdAt: string;
  source: { commit: string; mlx: string; dirty: boolean; status?: string };
  oxnode: { path: string; version: string };
  llama: { path: string; sha256: string; version: string; commit: string; dylibs: Record<string, string> };
  models: { target: string; draft: string };
  fixture: { id: string; payloadSha256: string };
  addon: Record<string, string>;
}
