#!/usr/bin/env oxnode
/// <reference types="node" />

import { spawn } from 'node:child_process';
import { appendFileSync, closeSync, openSync } from 'node:fs';
import { mkdir, readdir, stat } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import {
  ROOT,
  addonIdentity,
  assertCleanEnv,
  assertLatestMain,
  fetchLatestLlama,
  assertCleanTree,
  hostInfo,
  hostSnapshot,
  llamaIdentity,
  outDir,
  readJson,
  sh,
  sha256File,
  sleep,
  writeJson,
} from '../common.js';
import { report } from './report.js';
import {
  MMPROJ_FILE,
  MODEL_DIR,
  MODEL_FILE,
  PILOT_RUN,
  PILOT_SETTLE_MS,
  REPETITIONS,
  manifestPath,
  readFixture,
  readManifest,
  sampleId,
  samplePath,
  type Manifest,
  type Runtime,
} from './suite.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const WORKER = join(HERE, 'worker.ts');
const RENDER = join(HERE, 'render.ts');
const SCRIPTS = ['run.ts', 'worker.ts', 'render.ts', 'report.ts', 'suite.ts', 'baseline.csv', '../common.ts'];
const STEPS = ['preflight', 'prewarm', 'matrix', 'report', 'all'];

const exists = (path: string) =>
  stat(path).then(
    () => true,
    () => false,
  );

async function modelFiles(manifest: Manifest) {
  if (manifest.model.file !== basename(MODEL_FILE)) throw new Error(`${manifestPath}: unexpected model file`);
  const pinned = async (path: string, expected: string) => {
    const sha256 = await sha256File(path);
    if (sha256 !== expected) throw new Error(`${path}: SHA-256 ${sha256}, expected ${expected}`);
    return { path, bytes: (await stat(path)).size, sha256 };
  };
  // Unpinned, but they change the mlx-node prompt IDs; render.ts checks them.
  const recorded = async (path: string) =>
    (await exists(path)) ? { path, bytes: (await stat(path)).size, sha256: await sha256File(path) } : null;
  const marker = join(MODEL_DIR, '.mlx-download-complete.json');
  return {
    gguf: await pinned(MODEL_FILE, manifest.model.sha256),
    tokenizer: await pinned(join(MODEL_DIR, 'tokenizer.json'), manifest.model.tokenizerSha256),
    config: await pinned(join(MODEL_DIR, 'config.json'), manifest.model.configSha256),
    tokenizerConfig: await recorded(join(MODEL_DIR, 'tokenizer_config.json')),
    chatTemplate: await recorded(join(MODEL_DIR, 'chat_template.jinja')),
    mmproj: { path: MMPROJ_FILE, bytes: (await stat(MMPROJ_FILE)).size, sha256: await sha256File(MMPROJ_FILE) },
    download: (await exists(marker)) ? await readJson<unknown>(marker) : null,
  };
}

function checkout(allowDirty: boolean) {
  if (!allowDirty) return { ...assertCleanTree(), dirty: false };
  const status = sh('git', ['status', '--porcelain', '--ignore-submodules=none'], { cwd: ROOT });
  return {
    commit: sh('git', ['rev-parse', 'HEAD'], { cwd: ROOT }),
    mlx: sh('git', ['rev-parse', 'HEAD'], { cwd: join(ROOT, 'crates/mlx-sys/mlx') }),
    dirty: status !== '',
    status: status ? status.split('\n') : [],
  };
}

async function identity(manifest: Manifest, allowDirty: boolean) {
  const scripts: Record<string, string> = {};
  for (const file of SCRIPTS) scripts[file] = await sha256File(join(HERE, file));
  return {
    checkout: checkout(allowDirty),
    addon: await addonIdentity(),
    llama: await llamaIdentity(),
    model: await modelFiles(manifest),
    scripts,
  };
}

type Identity = Awaited<ReturnType<typeof identity>>;
type Provenance = Identity & { dirty: boolean; createdAt: string; completedAt?: string };

async function readProvenance(out: string, allowDirty: boolean): Promise<Provenance> {
  const path = join(out, 'provenance.json');
  if (!(await exists(path))) throw new Error(`${path} is missing; run the preflight step first`);
  const provenance = await readJson<Provenance>(path);
  if (provenance.dirty && !allowDirty) throw new Error('This run was prepared from a dirty tree; pass --allow-dirty');
  return provenance;
}

async function assertUnchanged(manifest: Manifest, provenance: Provenance) {
  const current = await identity(manifest, provenance.dirty);
  for (const key of ['addon', 'llama', 'model', 'scripts'] as const) {
    if (JSON.stringify(current[key]) !== JSON.stringify(provenance[key])) {
      throw new Error(`${key} changed since preflight`);
    }
  }
  if (current.checkout.commit !== provenance.checkout.commit || current.checkout.mlx !== provenance.checkout.mlx) {
    throw new Error('checkout changed since preflight');
  }
}

async function preflight(out: string, allowDirty: boolean) {
  assertCleanEnv();
  const latestMain = assertLatestMain();
  await fetchLatestLlama();
  if ((await exists(join(out, 'raw'))) && (await readdir(join(out, 'raw'))).length) {
    throw new Error(`${out}/raw already holds samples; use a new --out`);
  }
  const { manifest, path, cases } = await readFixture();
  const current = await identity(manifest, allowDirty);
  const render = JSON.parse(sh('oxnode', [RENDER], { cwd: ROOT })) as unknown;
  const provenance = {
    suite: 'gemma4',
    createdAt: new Date().toISOString(),
    latestMain,
    dirty: current.checkout.dirty,
    host: hostInfo(),
    ...current,
    fixture: {
      id: manifest.id,
      path,
      payloadSha256: manifest.payload.sha256,
      cases: cases.map(({ name, promptTokens, sha256 }) => ({ name, promptTokens, tokenIdsSha256: sha256 })),
      render,
    },
    protocol: manifest.protocol,
    childEnv: { prewarm: {}, matrix: manifest.protocol.mlxEnv },
    before: hostSnapshot(),
  };
  await writeJson(join(out, 'provenance.json'), provenance);
  console.log(
    JSON.stringify({
      event: 'preflight',
      out,
      commit: current.checkout.commit,
      dirty: current.checkout.dirty,
      llama: current.llama.commit,
      model: current.model.gguf.sha256,
    }),
  );
}

let caffeinate: ReturnType<typeof spawn> | undefined;

// Exits with this process; keeps idle sleep away from long unattended runs.
function stayAwake() {
  caffeinate ??= spawn('caffeinate', ['-i', '-w', String(process.pid)], { stdio: 'ignore' });
  caffeinate.unref();
}

async function sample(out: string, runtime: Runtime, name: string, run: string, env: NodeJS.ProcessEnv) {
  const id = sampleId(runtime, name, run);
  console.log(JSON.stringify({ event: 'start', at: new Date().toISOString(), id }));
  await mkdir(join(out, 'raw'), { recursive: true });
  const log = openSync(join(out, 'raw', `${id}.worker.log`), 'w');
  const child = spawn('oxnode', [WORKER, '--out', out, '--runtime', runtime, '--case', name, '--run', run], {
    cwd: ROOT,
    env,
    stdio: ['ignore', log, log],
  });
  closeSync(log);
  const code = await new Promise<number | string | null>((done, fail) => {
    child.once('error', fail);
    child.once('exit', (exitCode, signal) => done(exitCode ?? signal));
  });
  if (code !== 0) throw new Error(`${id}: worker exited with ${code}; see ${out}/raw/${id}.worker.log`);
  const result = await readJson<{ prefillTps: number; decodeTps: number; wallMs: number }>(samplePath(out, id));
  console.log(
    JSON.stringify({
      event: 'complete',
      id,
      prefillTps: result.prefillTps,
      decodeTps: result.decodeTps,
      wallMs: result.wallMs,
    }),
  );
}

function childEnv(manifest: Manifest, matrix: boolean): NodeJS.ProcessEnv {
  assertCleanEnv();
  return { ...process.env, ...(matrix ? manifest.protocol.mlxEnv : {}) };
}

// The first mlx-node load converts the GGUF into the native weight cache; these
// samples absorb that and the cold page cache, and never enter the results.
async function prewarm(out: string, allowDirty: boolean) {
  const manifest = await readManifest();
  await assertUnchanged(manifest, await readProvenance(out, allowDirty));
  stayAwake();
  for (const runtime of ['mlx', 'llama'] as const) {
    const id = sampleId(runtime, manifest.protocol.warmupCase, PILOT_RUN);
    if (await exists(samplePath(out, id))) {
      console.log(JSON.stringify({ event: 'skip', id }));
      continue;
    }
    await sample(out, runtime, manifest.protocol.warmupCase, PILOT_RUN, childEnv(manifest, false));
  }
}

// The cooldown runs before each job, so it also holds after a resume. A kept
// sample's completedAt stands in for its exit; a worker log without a sample is
// an attempt that ended at an unknown time, so the wait restarts from now.
async function lastFinish(out: string): Promise<number> {
  const raw = join(out, 'raw');
  const files = await readdir(raw);
  let latest = 0;
  for (const file of files.filter((name) => name.endsWith('.json'))) {
    latest = Math.max(latest, Date.parse((await readJson<{ completedAt: string }>(join(raw, file))).completedAt));
  }
  const unfinished = files.some(
    (name) => name.endsWith('.worker.log') && !files.includes(name.replace(/\.worker\.log$/, '.json')),
  );
  return unfinished ? Date.now() : latest;
}

async function matrix(out: string, allowDirty: boolean) {
  const manifest = await readManifest();
  const provenance = await readProvenance(out, allowDirty);
  await assertUnchanged(manifest, provenance);
  let pilotsDone = 0;
  for (const runtime of ['mlx', 'llama'] as const) {
    const path = samplePath(out, sampleId(runtime, manifest.protocol.warmupCase, PILOT_RUN));
    if (!(await exists(path))) throw new Error(`${path} is missing; run the prewarm step first`);
    pilotsDone = Math.max(pilotsDone, Date.parse((await readJson<{ completedAt: string }>(path)).completedAt));
  }
  stayAwake();
  const cooldown = manifest.protocol.cooldownSeconds * 1000;
  let finished = await lastFinish(out);
  const settle = PILOT_SETTLE_MS - (Date.now() - pilotsDone);
  if (settle > 0) {
    console.log(JSON.stringify({ event: 'settle', seconds: Math.ceil(settle / 1000) }));
    await sleep(settle);
  }
  const loadLog = join(out, 'load.ndjson');
  // A failed probe is recorded, never allowed to abort the matrix.
  const recordLoad = (phase: string) => {
    let snapshot: object;
    try {
      snapshot = hostSnapshot();
    } catch (error) {
      snapshot = { at: new Date().toISOString(), error: String(error) };
    }
    appendFileSync(loadLog, `${JSON.stringify({ phase, ...snapshot })}\n`);
  };
  recordLoad('matrix');
  const sampler = setInterval(() => recordLoad('matrix'), 5000);
  const env = childEnv(manifest, true);
  try {
    for (const run of REPETITIONS) {
      for (const { name } of manifest.cases) {
        const order: Runtime[] = Number(run) % 2 ? ['mlx', 'llama'] : ['llama', 'mlx'];
        for (const runtime of order) {
          const id = sampleId(runtime, name, run);
          if (await exists(samplePath(out, id))) {
            console.log(JSON.stringify({ event: 'skip', id }));
            continue;
          }
          const wait = cooldown - (Date.now() - finished);
          if (wait > 0) await sleep(wait);
          await sample(out, runtime, name, run, env);
          finished = Date.now();
        }
      }
    }
  } finally {
    clearInterval(sampler);
  }
  await assertUnchanged(manifest, provenance);
  recordLoad('after');
  await writeJson(join(out, 'provenance.json'), { ...provenance, completedAt: new Date().toISOString() });
}

async function main() {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      out: { type: 'string' },
      'allow-dirty': { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h' },
    },
  });
  const [step] = positionals;
  if (values.help || positionals.length !== 1 || !STEPS.includes(step)) {
    console.log(
      'Usage: oxnode scripts/perf-tracker/gemma4/run.ts <step> [--out <dir>]\n' +
        'Steps: preflight | prewarm | matrix | report | all (preflight, prewarm, matrix, report).\n' +
        'Without --out a new run directory is created; with --out the step continues or reports that run.\n' +
        'Preflight requires HEAD to contain origin/main and installs the newest llama.cpp macOS arm64 release.\n' +
        '--allow-dirty: development only; provenance records dirty: true.',
    );
    process.exitCode = values.help ? 0 : 1;
    return;
  }
  if (step === 'report' && !values.out) throw new Error('report needs --out <run directory>');
  const out = await outDir('gemma4', values.out);
  const allowDirty = values['allow-dirty'];
  const resume = step === 'all' && (await exists(join(out, 'provenance.json')));
  if (step === 'preflight' || (step === 'all' && !resume)) await preflight(out, allowDirty);
  if (step === 'prewarm' || step === 'all') await prewarm(out, allowDirty);
  if (step === 'matrix' || step === 'all') await matrix(out, allowDirty);
  if (step === 'report' || step === 'all') {
    const provenance = await readJson<Provenance>(join(out, 'provenance.json'));
    if (provenance.dirty) console.warn('warning: this run was prepared from a dirty tree');
    await report(out);
  }
  if (step !== 'report') console.log(out);
}

await main();
