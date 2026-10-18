/// <reference types="node" />

import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { LLAMA_SINGLE_REQUEST_ARGS, MODELS_DIR, type LlamaIdentity, type TreeIdentity } from '../common.js';
import type { IdCheck, Policy } from '../qwen38/suite.js';

export const HERE = dirname(fileURLToPath(import.meta.url));
export const WORKER = join(HERE, 'mlx-worker.ts');
export const DEFAULT_MODEL_DIR = join(MODELS_DIR, 'qwen3.8-flash-next-gguf/UD-Q4_K_XL');

export const SOURCE = {
  repo: 'unsloth/Qwen3.8-Flash-Next-GGUF',
  folder: 'UD-Q4_K_XL',
  revision: '766911a6b7369840a91dbcd95f9f997acaab6cd6',
} as const;

// SHA-256 = the Hugging Face LFS oid of each split at SOURCE.revision.
export const SPLITS = [
  {
    file: 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
    bytes: 10_946_624,
    sha256: '4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082',
  },
  {
    file: 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf',
    bytes: 49_859_583_136,
    sha256: '3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9',
  },
  {
    file: 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf',
    bytes: 49_376_141_504,
    sha256: '56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3',
  },
  {
    file: 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf',
    bytes: 12_087_983_520,
    sha256: '753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a',
  },
] as const;

// Files mlx-node extracts from the GGUF metadata at load; they decide its prompt IDs.
export const ASSET_FILES = ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json'];

// The first load fills the packed weight cache, so the untimed prewarm may run for a long time.
export const PREWARM_TIMEOUT_MS = 2 * 3_600_000;
// mlx-node prefill runs at 15–75 tok/s on this model, so one 32,488-token sample can exceed the shared 1,200 s limit.
export const SAMPLE_TIMEOUT_MS = 3_600_000;

// Repetitions, warmup tokens and cooldowns come from the qwen38 fixture's protocol.
export const PROTOCOL = {
  outputTokens: 256,
  outputTokensReason:
    'qwen4_exp caps the context at 32,768 tokens (prefill and decode refuse beyond it), so 32,488 + 512 cannot run; every row uses 256',
  llamaServerArgs: [
    '-ngl',
    'all',
    '-fit',
    'off',
    '-c',
    '32768',
    '-np',
    '1',
    '-b',
    '1024',
    '-ub',
    '1024',
    '-fa',
    'on',
    '-ctk',
    'f16',
    '-ctv',
    'f16',
    '-t',
    '8',
    '-tb',
    '8',
    '-lm',
    'mmap',
    '--spec-type',
    'none',
    ...LLAMA_SINGLE_REQUEST_ARGS,
    '--no-warmup',
    '--perf',
  ],
  // Untimed prewarm only: info-level libllama logs carry the buffer sizes and backend notes.
  llamaPrewarmArgs: ['-lv', '4'],
  // llama.cpp keeps every weight resident; the automatic budget keeps a reserve and streams experts from SSD.
  mlxEnv: { MLX_QWEN4_RESIDENCY: 'full' } as Record<string, string>,
  mlxLoadOptions: { autoLoadDraft: false },
  weightBudget:
    'mlx-node full hot-path residency (MLX_QWEN4_RESIDENCY=full); the load fails instead of streaming experts',
  kvCache: { 'mlx-node': 'BF16 block-paged', 'llama.cpp': 'F16' },
  sampling: 'Greedy, high thinking, retain historical reasoning; no MTP or other speculation.',
};

export interface SplitIdentity {
  path: string;
  size: number;
  mtimeMs: number;
  ctimeMs: number;
  ino: number;
  dev: number;
  sha256?: string;
}

// mlx-node keys its asset and packed-weight caches on path, inode, ctime and device;
// `sha256` is absent from the per-block check.
export interface Identity {
  tree: TreeIdentity;
  addon: Record<string, string>;
  llama: LlamaIdentity;
  model: { dir: string; splits: SplitIdentity[] };
  scripts: Record<string, string>;
}

export const IDENTITY_KEYS = ['tree', 'addon', 'llama', 'model', 'scripts'] as const;

export interface Volume {
  path: string;
  device: string;
  volumeName: string;
  volumeUuid: string;
  mountPoint: string;
  bus: string;
  internal: boolean;
  solidState: boolean;
}

export interface Residency {
  policy: string;
  weightBudgetBytes: number;
  hotWeightBytes: number;
  fullHotResidency: boolean;
  prefillChunkTokens: number;
  availableMemoryBytes?: number;
  physicalMemoryBytes?: number;
  residentBankBytes: number;
  residentExpertLayers: number;
  partialExpertSlots: number;
}

export interface Loaded {
  event: 'loaded';
  pid: number;
  loadMs: number;
  assets: string;
  context: { effectiveWindowTokens: number };
  residency: Residency;
  [field: string]: unknown;
}

export interface Provenance extends Identity {
  host: { os: string; chip: string; gpuCores: number; memoryBytes: number };
  volumes: Volume[];
  protocol: { admission: Record<string, Policy>; desktopLoad: boolean; [field: string]: unknown };
  tokenizer?: { verified: boolean; [field: string]: unknown };
  prewarm?: {
    at: string;
    assets: string;
    assetFiles: Record<string, string>;
    checkIds: IdCheck[];
    residency: Residency;
  };
  matrix?: { startedAt: string; finishedAt?: string; postRunVerified?: boolean };
  [field: string]: unknown;
}
