#!/usr/bin/env oxnode
/// <reference types="node" />

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { lstat, mkdir, readdir, realpath, rm, stat } from 'node:fs/promises';
import { dirname, join, relative } from 'node:path';
import { isDeepStrictEqual, parseArgs } from 'node:util';

import {
  ROOT,
  addonIdentity,
  assertCleanEnv,
  assertPin,
  assertHostReady,
  assertLatestMain,
  fetchLatestLlama,
  fileHashes,
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
import { admit, recordGpu, snapshot } from './admission.js';
import { printSummary, report } from './report.js';
import { LlamaServer, MlxWorker, type Runner } from './runtimes.js';
import {
  FIXTURE,
  HERE,
  IDENTITY_KEYS,
  KIND,
  LOAD_COOLDOWN_SECONDS,
  MODEL_PATH,
  SIDECARS,
  SPECULATION,
  admissionPolicy,
  checkAdmittedIds,
  discardPartialCells,
  draftPaths,
  llamaServerArgs,
  PORT,
  Progress,
  RUNTIMES,
  paths,
  readManifest,
  sampleAcceptance,
  verifyTokenizer,
  writeJsonAtomic,
  type FileIdentity,
  type Identity,
  type Manifest,
  type Provenance,
  type Runtime,
  type Sample,
} from './suite.js';

const STEPS = ['preflight', 'prewarm', 'check-ids', 'matrix', 'report', 'all'] as const;
type Step = (typeof STEPS)[number];

// Workers spawned per block import these, so an edit mid-run would change later blocks.
async function scriptHashes(): Promise<Record<string, string>> {
  const paths = [
    ...(await readdir(HERE)).map((name) => join(HERE, name)),
    join(ROOT, 'scripts/perf-tracker/common.ts'),
    join(ROOT, 'scripts/fixtures', `${FIXTURE}.json`),
  ].sort();
  const hashes: Record<string, string> = {};
  for (const path of paths) {
    if ((await lstat(path)).isFile()) hashes[relative(ROOT, path)] = await sha256File(path);
  }
  return hashes;
}

async function fileIdentity(path: string, hash: boolean): Promise<FileIdentity> {
  const { size, mtimeMs, ino } = await stat(path);
  if (hash) console.log(`Hashing ${path}`);
  return {
    path,
    realpath: await realpath(path),
    size,
    mtimeMs,
    ino,
    ...(hash ? { sha256: await sha256File(path) } : {}),
  };
}

// The 17.5 GB GGUF hash takes about a minute, so blocks compare size, mtime and inode of the model files.
async function identity(manifest: Manifest, allowDirty: boolean, hashModel: boolean): Promise<Identity> {
  const drafts = async (runtime: Runtime) =>
    Promise.all(draftPaths(manifest, runtime).map((path) => fileIdentity(path, hashModel)));
  return {
    tree: await treeIdentity(allowDirty),
    addon: await addonIdentity(),
    llama: await llamaIdentity(),
    model: await fileIdentity(MODEL_PATH, hashModel),
    drafts: { 'mlx-node': await drafts('mlx-node'), 'llama.cpp': await drafts('llama.cpp') },
    sidecars: await fileHashes(dirname(MODEL_PATH), SIDECARS),
    scripts: await scriptHashes(),
  };
}

async function verifyIdentity(
  manifest: Manifest,
  recorded: Identity,
  allowDirty: boolean,
  hashModel: boolean,
  when: string,
) {
  const current = await identity(manifest, allowDirty, hashModel);
  const unhashed = ({ sha256: _sha256, ...file }: FileIdentity) => file;
  const expected = hashModel
    ? recorded
    : {
        ...recorded,
        model: unhashed(recorded.model),
        drafts: {
          'mlx-node': recorded.drafts['mlx-node'].map(unhashed),
          'llama.cpp': recorded.drafts['llama.cpp'].map(unhashed),
        },
      };
  const changed = IDENTITY_KEYS.filter((key) => !isDeepStrictEqual(current[key], expected[key]));
  if (changed.length) throw new Error(`${when}: ${changed.join(', ')} changed since preflight`);
}

async function preflight(out: string, allowDirty: boolean, desktopLoad: boolean): Promise<void> {
  assertCleanEnv();
  const latestMain = assertLatestMain();
  await fetchLatestLlama();
  const files = paths(out);
  if (existsSync(files.samples)) throw new Error(`${out} already holds samples; preflight a new --out directory`);
  const manifest = await readManifest();
  assertHostReady(PORT);
  const fixture = await loadFixture(FIXTURE);
  const recorded = await identity(manifest, allowDirty, true);
  assertPin(MODEL_PATH, recorded.model.size, recorded.model.sha256, manifest.model);
  for (const runtime of RUNTIMES) {
    for (const [index, pin] of manifest.draft[runtime].files.entries()) {
      const { path, size, sha256 } = recorded.drafts[runtime][index];
      assertPin(path, size, sha256, pin);
    }
  }
  const tokenizer = await verifyTokenizer(join(dirname(MODEL_PATH), 'tokenizer.json'));
  await writeJson(files.provenance, {
    preflightAt: new Date().toISOString(),
    latestMain,
    host: hostInfo(),
    ...recorded,
    fixture: {
      id: manifest.id,
      payload: fixture.manifest.payload,
      cases: fixture.cases.map(({ name, promptTokens, sha256: idsSha256 }) => ({ name, promptTokens, idsSha256 })),
    },
    tokenizer,
    protocol: {
      ...manifest.protocol,
      llamaServerArgs: llamaServerArgs(manifest),
      port: PORT,
      loadCooldownSeconds: LOAD_COOLDOWN_SECONDS,
      admission: admissionPolicy(desktopLoad),
      desktopLoad,
    },
    snapshot: hostSnapshot(),
  });
}

async function prewarm(out: string): Promise<void> {
  assertCleanEnv();
  const { protocol } = await readManifest();
  await mkdir(paths(out).raw, { recursive: true });
  const worker = await MlxWorker.start('prewarm', out, protocol.mlxEnv);
  await worker.close();
  const { loaded } = worker;
  const tokenizer = await verifyTokenizer(join(loaded.assets as string, 'tokenizer.json'));
  await writeJson(join(out, 'prewarm.json'), { loaded, tokenizer });
}

async function checkIds(out: string): Promise<void> {
  assertCleanEnv();
  const { protocol } = await readManifest();
  const { cases } = await loadFixture(FIXTURE);
  await mkdir(paths(out).raw, { recursive: true });
  const trace = join(out, 'session-id-trace.txt');
  await rm(trace, { force: true });
  const worker = await MlxWorker.start('idcheck', out, {
    ...protocol.mlxEnv,
    MLX_INFERENCE_TRACE: '1',
    MLX_INFERENCE_TRACE_FILE: trace,
  });
  try {
    for (const item of cases) await worker.sample(item, 'idcheck', 1);
  } finally {
    await worker.close();
  }
  await checkAdmittedIds(trace, cases, out);
  console.log('Native session prompt IDs match the pinned IDs for all cases.');
}

async function matrix(out: string, allowDirty: boolean): Promise<void> {
  assertCleanEnv();
  const manifest = await readManifest();
  const { protocol } = manifest;
  const files = paths(out);
  if (!existsSync(files.provenance)) throw new Error(`${files.provenance} is missing; run preflight first`);
  const provenance = await readJson<Provenance>(files.provenance);
  assertHostReady(PORT);
  await verifyIdentity(manifest, provenance, allowDirty, true, 'matrix start');
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
    if (sample.numTokens !== protocol.outputTokens || sample.cachedTokens !== 0) {
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
  const caffeinate = spawn('caffeinate', ['-i', '-w', String(process.pid)], { stdio: 'ignore' });
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
        await verifyIdentity(manifest, provenance, allowDirty, false, `${kind}-${block}`);
        progress.emit({ event: 'identity_verified', label: `${kind}-${block}`, gguf: 'size, mtime, inode' });
        await admit(policy, progress, `${kind}-${block}-load`, LOAD_COOLDOWN_SECONDS);
        let runner: Runner;
        if (runtime === 'mlx-node') {
          const worker = await MlxWorker.start(block, out, protocol.mlxEnv);
          progress.emit({ event: 'loaded', runtime, run: block, metadata: worker.loaded });
          runner = worker;
        } else {
          const server = await LlamaServer.start(block, out, llamaServerArgs(manifest));
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
            const result = await runner.sample(item, run, protocol.outputTokens);
            const after = snapshot();
            await writeJson(join(files.system, `${kind}-${run}-${item.name}.json`), { before, after });
            if (
              result.numTokens !== protocol.outputTokens ||
              result.cachedTokens !== 0 ||
              result.promptTokens !== item.tokenIds.length
            ) {
              throw new Error(
                `${runtime} ${item.name}: ${result.numTokens} tokens, ${result.cachedTokens} cached, ${result.promptTokens} prompt`,
              );
            }
            sampleAcceptance(result, SPECULATION);
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
    await verifyIdentity(manifest, provenance, allowDirty, true, 'post-run');
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
      'allow-dirty': { type: 'boolean', default: false },
      'desktop-load': { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h' },
    },
  });
  const step = positionals[0] as Step;
  if (values.help || positionals.length !== 1 || !STEPS.includes(step)) {
    console.log(
      `Usage: oxnode scripts/perf-tracker/qwen38/run.ts <${STEPS.join('|')}> [--out <dir>]\n` +
        'Preflight requires HEAD to contain origin/main and installs the newest llama.cpp macOS arm64 release.\n' +
        'Without --out a new run directory is created; with --out the step continues or reports that run.\n' +
        'all = preflight, prewarm, matrix, report. check-ids is an optional untimed native prompt-ID check.\n' +
        '--allow-dirty measures or reports an uncommitted checkout instead of refusing it (development only).\n' +
        '--desktop-load (preflight/all): admit every row with no other local LLM instead of waiting for GPU idle;\n' +
        '  the run records it and its 6k/16k rows are not comparable with GPU-idle entries.',
    );
    if (!values.help) process.exitCode = 1;
    return;
  }
  if (step === 'report' && !values.out && !process.env.OUT_DIR) throw new Error('report needs --out <run dir>');
  const out = await outDir('qwen38', values.out);
  const allowDirty = values['allow-dirty'];
  console.log(`qwen38 ${step}: ${out}`);
  if (step === 'preflight' || step === 'all') await preflight(out, allowDirty, values['desktop-load']);
  if (step === 'prewarm' || step === 'all') await prewarm(out);
  if (step === 'check-ids') await checkIds(out);
  if (step === 'matrix' || step === 'all') await matrix(out, allowDirty);
  if (step === 'report' || step === 'all') await printReport(out, allowDirty);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? (error.stack ?? error.message) : error);
  process.exitCode = 1;
});
