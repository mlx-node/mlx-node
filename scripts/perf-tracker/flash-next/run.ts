#!/usr/bin/env oxnode
/// <reference types="node" />

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { lstat, mkdir, readdir, readFile, realpath, rm } from 'node:fs/promises';
import { join, relative } from 'node:path';
import { isDeepStrictEqual, parseArgs } from 'node:util';

import {
  ROOT,
  addonIdentity,
  assertCleanEnv,
  assertHostReady,
  assertLatestMain,
  fetchLatestLlama,
  hostInfo,
  hostSnapshot,
  llamaIdentity,
  loadFixture,
  outDir,
  readJson,
  sha256File,
  treeIdentity,
  utcStamp,
  writeJson,
} from '../common.js';
import { admit, recordGpu, snapshot } from '../qwen38/admission.js';
import { printSummary } from '../qwen38/report.js';
import { LlamaServer, MlxWorker, type Result, type Runner } from '../qwen38/runtimes.js';
import {
  FIXTURE,
  HERE as QWEN38_DIR,
  KIND,
  LOAD_COOLDOWN_SECONDS,
  PORT,
  Progress,
  RUNTIMES,
  admissionPolicy,
  checkAdmittedIds,
  discardPartialCells,
  paths,
  readManifest,
  verifyTokenizer,
  writeJsonAtomic,
  type Runtime,
  type Sample,
} from '../qwen38/suite.js';
import { assertInternalSsd, assertNoRuntimeAlive, checkLoaded, checkResult, split } from './checks.js';
import { report } from './report.js';
import {
  DEFAULT_MODEL_DIR,
  HERE,
  IDENTITY_KEYS,
  PREWARM_TIMEOUT_MS,
  PROTOCOL,
  SOURCE,
  SPLITS,
  WORKER,
  type Identity,
  type Loaded,
  type Provenance,
  type SplitIdentity,
} from './suite.js';

const STEPS = ['preflight', 'prewarm', 'matrix', 'report', 'all'] as const;
type Step = (typeof STEPS)[number];

// Workers spawned per block import these, so an edit mid-run would change later blocks.
async function scriptHashes(): Promise<Record<string, string>> {
  const paths = [
    ...(await readdir(HERE)).map((name) => join(HERE, name)),
    ...(await readdir(QWEN38_DIR)).map((name) => join(QWEN38_DIR, name)),
    join(ROOT, 'scripts/perf-tracker/common.ts'),
    join(ROOT, 'scripts/fixtures', `${FIXTURE}.json`),
  ].sort();
  const hashes: Record<string, string> = {};
  for (const path of paths) {
    if ((await lstat(path)).isFile()) hashes[relative(ROOT, path)] = await sha256File(path);
  }
  return hashes;
}

// The 111 GB of splits take minutes to hash, so blocks compare size, times, inode and device.
async function identity(allowDirty: boolean, dir: string, hashModel: boolean): Promise<Identity> {
  if (hashModel) console.log(`Hashing ${dir}`);
  const splits: SplitIdentity[] = [];
  for (const { file } of SPLITS) splits.push(await split(join(dir, file), hashModel));
  return {
    tree: await treeIdentity(allowDirty),
    addon: await addonIdentity(),
    llama: await llamaIdentity(),
    model: { dir, splits },
    scripts: await scriptHashes(),
  };
}

async function verifyIdentity(recorded: Identity, allowDirty: boolean, hashModel: boolean, when: string) {
  const current = await identity(allowDirty, recorded.model.dir, hashModel);
  const splits = recorded.model.splits.map(({ sha256: _sha256, ...rest }) => rest);
  const expected = hashModel ? recorded : { ...recorded, model: { ...recorded.model, splits } };
  const changed = IDENTITY_KEYS.filter((key) => !isDeepStrictEqual(current[key], expected[key]));
  if (changed.length) throw new Error(`${when}: ${changed.join(', ')} changed since preflight`);
}

async function readProvenance(out: string): Promise<Provenance> {
  const file = paths(out).provenance;
  if (!existsSync(file)) throw new Error(`${file} is missing; run preflight first`);
  return readJson<Provenance>(file);
}

function stayAwake() {
  return spawn('caffeinate', ['-i', '-w', String(process.pid)], { stdio: 'ignore' });
}

async function preflight(out: string, allowDirty: boolean, desktopLoad: boolean, modelDir: string): Promise<void> {
  assertCleanEnv();
  const latestMain = assertLatestMain();
  await fetchLatestLlama();
  const files = paths(out);
  if (existsSync(files.samples)) throw new Error(`${out} already holds samples; preflight a new --out directory`);
  const manifest = await readManifest();
  assertHostReady(PORT);
  const fixture = await loadFixture(FIXTURE);
  const dir = await realpath(modelDir);
  for (const { file } of SPLITS) await split(join(dir, file), false);
  const volumes = assertInternalSsd(SPLITS.map(({ file }) => join(dir, file)));
  const recorded = await identity(allowDirty, dir, true);
  for (const [index, pinned] of SPLITS.entries()) {
    const { path, size, sha256 } = recorded.model.splits[index];
    if (size !== pinned.bytes || sha256 !== pinned.sha256) {
      throw new Error(`${path}: ${size} bytes, SHA-256 ${sha256}; not the pinned ${SOURCE.repo} split`);
    }
  }
  const { repetitions, warmupTokens, cooldownSeconds } = manifest.protocol;
  await writeJson(files.provenance, {
    suite: 'flash-next',
    preflightAt: new Date().toISOString(),
    latestMain,
    host: hostInfo(),
    ...recorded,
    source: { ...SOURCE, splits: SPLITS },
    volumes,
    fixture: {
      id: manifest.id,
      payload: fixture.manifest.payload,
      cases: fixture.cases.map(({ name, promptTokens, sha256: idsSha256 }) => ({ name, promptTokens, idsSha256 })),
    },
    protocol: {
      fixture: manifest.id,
      repetitions,
      warmupTokens,
      cooldownSeconds,
      ...PROTOCOL,
      port: PORT,
      loadCooldownSeconds: LOAD_COOLDOWN_SECONDS,
      admission: admissionPolicy(desktopLoad),
      desktopLoad,
    },
    snapshot: hostSnapshot(),
  });
}

// Untimed. The first mlx-node load fills the packed weight cache; one process
// runs every case and traces the prompt IDs its native session admits.
async function prewarm(out: string, allowDirty: boolean): Promise<void> {
  assertCleanEnv();
  const files = paths(out);
  const provenance = await readProvenance(out);
  assertHostReady(PORT);
  assertInternalSsd(provenance.model.splits.map(({ path }) => path));
  await verifyIdentity(provenance, allowDirty, false, 'prewarm');
  const { cases } = await loadFixture(FIXTURE);
  await mkdir(files.raw, { recursive: true });
  const caffeinate = stayAwake();
  try {
    await assertNoRuntimeAlive('prewarm mlx-node');
    const trace = join(out, 'session-id-trace.txt');
    await rm(trace, { force: true });
    const env = { ...PROTOCOL.mlxEnv, MLX_INFERENCE_TRACE: '1', MLX_INFERENCE_TRACE_FILE: trace };
    const worker = await MlxWorker.start('prewarm', out, env, { worker: WORKER, timeoutMs: PREWARM_TIMEOUT_MS });
    const loaded = worker.loaded as Loaded;
    let assetFiles: Record<string, string>;
    const mlx: Result[] = [];
    try {
      console.log(JSON.stringify({ event: 'prewarm_loaded', runtime: 'mlx-node', loadMs: loaded.loadMs }));
      assetFiles = await checkLoaded(loaded, cases);
      for (const item of cases) {
        const result = await worker.sample(item, 'prewarm', PROTOCOL.outputTokens);
        checkResult(result, item);
        mlx.push(result);
        console.log(JSON.stringify({ event: 'prewarm_sample', runtime: 'mlx-node', name: item.name }));
      }
    } finally {
      await worker.close();
    }
    const checkIds = await checkAdmittedIds(trace, cases, out);
    const tokenizer = await verifyTokenizer(join(loaded.assets, 'tokenizer.json'));

    await assertNoRuntimeAlive('prewarm llama.cpp');
    const args = [...PROTOCOL.llamaServerArgs, ...PROTOCOL.llamaPrewarmArgs];
    const model = provenance.model.splits[0].path;
    const server = await LlamaServer.start('prewarm', out, args, { model, timeoutMs: PREWARM_TIMEOUT_MS });
    const llama: Result[] = [];
    try {
      console.log(JSON.stringify({ event: 'prewarm_loaded', runtime: 'llama.cpp', loadMs: server.loadMs }));
      for (const item of cases) {
        const result = await server.sample(item, 'prewarm', PROTOCOL.outputTokens);
        checkResult(result, item);
        llama.push(result);
        console.log(JSON.stringify({ event: 'prewarm_sample', runtime: 'llama.cpp', name: item.name }));
      }
    } finally {
      await server.close();
    }
    const llamaLog = (await readFile(join(files.raw, 'llama-prewarm.log'), 'utf8'))
      .split('\n')
      .filter((line) =>
        /buffer size|n_ctx|kv_unified|n_slots|has_bfloat|fallback|not supported|checkpoint| W |CPU/.test(line),
      );

    const at = new Date().toISOString();
    const timings = (results: Result[]) => results.map(({ name, performance }) => ({ name, performance }));
    await writeJson(join(out, 'prewarm.json'), {
      at,
      loaded,
      tokenizer,
      checkIds,
      mlx: timings(mlx),
      llama: { loadMs: server.loadMs, args: server.args, log: llamaLog, samples: timings(llama) },
    });
    provenance.tokenizer = tokenizer;
    provenance.prewarm = { at, assets: loaded.assets, assetFiles, checkIds, residency: loaded.residency };
    await writeJson(files.provenance, provenance);
    console.log('Native session prompt IDs match the pinned IDs for all cases.');
  } finally {
    caffeinate.kill();
  }
}

async function matrix(out: string, allowDirty: boolean): Promise<void> {
  assertCleanEnv();
  const manifest = await readManifest();
  const { protocol } = manifest;
  const files = paths(out);
  const provenance = await readProvenance(out);
  const proof = provenance.prewarm;
  if (provenance.tokenizer?.verified !== true || !proof?.checkIds.every((check) => check.identical)) {
    throw new Error(`${out}: no verified prewarm; run the prewarm step first`);
  }
  assertHostReady(PORT);
  assertInternalSsd(provenance.model.splits.map(({ path }) => path));
  await verifyIdentity(provenance, allowDirty, true, 'matrix start');
  const { cases } = await loadFixture(FIXTURE);

  const repetitions = Array.from({ length: protocol.repetitions }, (_, index) => index + 1);
  const key = (runtime: Runtime, name: string, run: number) => `${runtime}/${name}/${run}`;
  const expected = new Set(
    RUNTIMES.flatMap((runtime) => cases.flatMap((item) => repetitions.map((run) => key(runtime, item.name, run)))),
  );
  const saved = existsSync(files.samples) ? await readJson<Sample[]>(files.samples) : [];
  const savedIds = new Set(saved.map((sample) => key(sample.runtime, sample.name, sample.run)));
  if (savedIds.size !== saved.length || ![...savedIds].every((id) => expected.has(id))) {
    throw new Error(`${files.samples} does not belong to this matrix`);
  }
  for (const sample of saved) {
    if (sample.numTokens !== PROTOCOL.outputTokens || sample.cachedTokens !== 0) {
      throw new Error(`${files.samples} holds a sample outside the protocol`);
    }
  }

  await mkdir(files.raw, { recursive: true });
  await mkdir(files.system, { recursive: true });
  const progress = new Progress(files.progress);
  const stamp = utcStamp();
  const samples = await discardPartialCells(out, saved, protocol.repetitions, stamp);
  const completed = new Set(samples.map((sample) => key(sample.runtime, sample.name, sample.run)));
  provenance.matrix = { startedAt: provenance.matrix?.startedAt ?? new Date().toISOString() };
  await writeJson(files.provenance, provenance);
  const caffeinate = stayAwake();
  const stopGpu = recordGpu(files.gpu, () => progress.last);
  try {
    if (saved.length) {
      progress.emit({
        event: 'resumed',
        savedSamples: samples.length,
        discardedSamples: saved.length - samples.length,
        remaining: expected.size - samples.length,
      });
    }
    for (const [index, item] of cases.entries()) {
      const policy = provenance.protocol.admission[item.promptTokens];
      for (const runtime of index % 2 === 0 ? RUNTIMES : [...RUNTIMES].reverse()) {
        if (repetitions.every((run) => completed.has(key(runtime, item.name, run)))) continue;
        const kind = KIND[runtime];
        const block = `block-${item.name}-${stamp}`;
        await verifyIdentity(provenance, allowDirty, false, `${kind}-${block}`);
        progress.emit({
          event: 'identity_verified',
          label: `${kind}-${block}`,
          gguf: 'size, mtime, ctime, inode, device',
        });
        const load = await admit(policy, progress, `${kind}-${block}-load`, LOAD_COOLDOWN_SECONDS);
        await assertNoRuntimeAlive(`${kind}-${block}`);
        await writeJson(join(files.system, `${kind}-${block}-load.json`), { before: load });
        let runner: Runner;
        if (runtime === 'mlx-node') {
          const worker = await MlxWorker.start(block, out, PROTOCOL.mlxEnv, { worker: WORKER });
          progress.emit({ event: 'loaded', runtime, run: block, metadata: worker.loaded });
          try {
            await checkLoaded(worker.loaded as Loaded, cases, proof.assetFiles);
          } catch (error) {
            await worker.close();
            throw error;
          }
          runner = worker;
        } else {
          const model = provenance.model.splits[0].path;
          const server = await LlamaServer.start(block, out, PROTOCOL.llamaServerArgs, { model });
          progress.emit({ event: 'loaded', runtime, run: block, loadMs: server.loadMs, args: server.args });
          runner = server;
        }
        try {
          progress.emit({ event: 'warmup_start', runtime: kind, run: block, name: item.name });
          const warm = await runner.sample(item, `warmup-${block}`, protocol.warmupTokens);
          progress.emit({
            event: 'warmup_done',
            runtime: kind,
            run: block,
            name: item.name,
            performance: warm.performance,
          });
          for (const run of repetitions) {
            const before = await admit(policy, progress, `${kind}-${run}-${item.name}`, protocol.cooldownSeconds);
            progress.emit({
              event: 'sample_start',
              runtime: kind,
              run,
              name: item.name,
              promptTokens: item.tokenIds.length,
            });
            const result = await runner.sample(item, run, PROTOCOL.outputTokens);
            const after = snapshot();
            await writeJson(join(files.system, `${kind}-${run}-${item.name}.json`), { before, after });
            checkResult(result, item);
            const {
              text: _t,
              thinking: _k,
              rawText: _r,
              publicRawText: _p,
              final: _f,
              tokens: _i,
              ...compact
            } = result;
            const sample = { ...compact, block } as Sample;
            samples.push(sample);
            completed.add(key(runtime, item.name, run));
            await writeJsonAtomic(files.samples, samples);
            progress.emit(sample);
          }
        } finally {
          await runner.close();
        }
      }
    }
    if (completed.size !== expected.size) throw new Error('Incomplete measurement matrix');
    provenance.postRun = hostSnapshot();
    await verifyIdentity(provenance, allowDirty, true, 'post-run');
    progress.emit({ event: 'identity_verified', label: 'post-run', gguf: 'sha256' });
    progress.emit({ event: 'complete', samples: samples.length });
  } finally {
    await stopGpu();
    caffeinate.kill();
  }
  provenance.matrix = { ...provenance.matrix, finishedAt: new Date().toISOString(), postRunVerified: true };
  await writeJson(files.provenance, provenance);
}

async function printReport(out: string, allowDirty: boolean): Promise<void> {
  printSummary(await report(out, allowDirty));
  console.log(['samples.csv', 'summary.json', 'chart.svg', 'compare.md'].map((name) => join(out, name)).join('\n'));
}

async function main(): Promise<void> {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      out: { type: 'string' },
      'model-dir': { type: 'string' },
      'allow-dirty': { type: 'boolean', default: false },
      'desktop-load': { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h' },
    },
  });
  const step = positionals[0] as Step;
  if (values.help || positionals.length !== 1 || !STEPS.includes(step)) {
    console.log(
      `Usage: oxnode scripts/perf-tracker/flash-next/run.ts <${STEPS.join('|')}> [--out <dir>] [--model-dir <dir>]\n` +
        'Preflight requires HEAD to contain origin/main and installs the newest llama.cpp macOS arm64 release.\n' +
        'Without --out a new run directory is created; with --out the step continues or reports that run.\n' +
        'all = preflight, prewarm, matrix, report. prewarm also proves the native session admits the pinned IDs.\n' +
        `--model-dir (preflight/all): the folder of the four splits on the internal SSD; default ${DEFAULT_MODEL_DIR}.\n` +
        '  Later steps read it from provenance.json.\n' +
        '--allow-dirty measures or reports an uncommitted checkout instead of refusing it (development only).\n' +
        '--desktop-load (preflight/all): admit every row with no other local LLM instead of waiting for GPU idle;\n' +
        '  the run records it and its 6k/16k rows are not comparable with GPU-idle entries.',
    );
    if (!values.help) process.exitCode = 1;
    return;
  }
  if (values['model-dir'] !== undefined && step !== 'preflight' && step !== 'all') {
    throw new Error('--model-dir applies to preflight and all; later steps read the model folder from provenance.json');
  }
  if (step === 'report' && !values.out && !process.env.OUT_DIR) throw new Error('report needs --out <run dir>');
  const out = await outDir('flash-next', values.out);
  const allowDirty = values['allow-dirty'];
  console.log(`flash-next ${step}: ${out}`);
  if (step === 'preflight' || step === 'all') {
    await preflight(out, allowDirty, values['desktop-load'], values['model-dir'] ?? DEFAULT_MODEL_DIR);
  }
  if (step === 'prewarm' || step === 'all') await prewarm(out, allowDirty);
  if (step === 'matrix' || step === 'all') await matrix(out, allowDirty);
  if (step === 'report' || step === 'all') await printReport(out, allowDirty);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? (error.stack ?? error.message) : error);
  process.exitCode = 1;
});
