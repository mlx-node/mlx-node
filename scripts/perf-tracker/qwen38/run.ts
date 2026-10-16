#!/usr/bin/env oxnode
/// <reference types="node" />

import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, readFile, realpath, rm, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { isDeepStrictEqual, parseArgs } from 'node:util';

import {
  ROOT,
  addonIdentity,
  assertCleanEnv,
  assertCleanTree,
  assertFileSha256,
  hostInfo,
  hostSnapshot,
  llamaIdentity,
  loadFixture,
  outDir,
  readJson,
  sh,
  sha256,
  sha256File,
  utcStamp,
  writeJson,
} from '../common.js';
import { admit, recordGpu, snapshot } from './admission.js';
import { report } from './report.js';
import { LlamaServer, MlxWorker, type Runner } from './runtimes.js';
import {
  FIXTURE,
  KIND,
  LOAD_COOLDOWN_SECONDS,
  MODEL_PATH,
  POLICY,
  PORT,
  Progress,
  RUNTIMES,
  paths,
  readManifest,
  writeJsonAtomic,
  type Provenance,
  type Runtime,
  type Sample,
} from './suite.js';

const STEPS = ['preflight', 'prewarm', 'check-ids', 'matrix', 'report', 'all'] as const;
type Step = (typeof STEPS)[number];

function checkout(allowDirty: boolean) {
  try {
    return { ...assertCleanTree(), dirty: false };
  } catch (error) {
    if (!allowDirty || !(error instanceof Error) || !error.message.startsWith('The measured checkout must be clean')) {
      throw error;
    }
    return {
      commit: sh('git', ['rev-parse', 'HEAD'], { cwd: ROOT }),
      mlx: sh('git', ['rev-parse', 'HEAD'], { cwd: join(ROOT, 'crates/mlx-sys/mlx') }),
      dirty: true,
      status: sh('git', ['status', '--porcelain', '--ignore-submodules=none'], { cwd: ROOT }),
    };
  }
}

async function assertModel(expected: { bytes: number; sha256: string }): Promise<void> {
  const { size } = await stat(MODEL_PATH);
  if (size !== expected.bytes) throw new Error(`${MODEL_PATH}: ${size} bytes, expected ${expected.bytes}`);
  console.log(`Hashing ${MODEL_PATH}`);
  await assertFileSha256(MODEL_PATH, expected.sha256);
}

function assertHostReady(): void {
  const power = sh('pmset', ['-g', 'batt']).split('\n')[0];
  if (!power.includes("'AC Power'")) throw new Error(`Measure on AC power: ${power}`);
  if (spawnSync('lsof', ['-nP', `-iTCP:${PORT}`, '-sTCP:LISTEN']).status === 0) throw new Error(`Port ${PORT} is busy`);
}

async function verifyTokenizer(tokenizer: string) {
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
  const files: Record<string, string> = {};
  for (const name of ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']) {
    const path = join(dirname(tokenizer), name);
    if (existsSync(path)) files[name] = await sha256File(path);
  }
  return { tokenizer, files, verified: true, output };
}

async function preflight(out: string, allowDirty: boolean): Promise<void> {
  assertCleanEnv();
  const files = paths(out);
  if (existsSync(files.samples)) throw new Error(`${out} already holds samples; preflight a new --out directory`);
  const manifest = await readManifest();
  assertHostReady();
  const tree = checkout(allowDirty);
  const llama = await llamaIdentity();
  const addon = await addonIdentity();
  const fixture = await loadFixture(FIXTURE);
  await assertModel(manifest.model);
  const tokenizer = await verifyTokenizer(join(dirname(MODEL_PATH), 'tokenizer.json'));
  await writeJson(files.provenance, {
    preflightAt: new Date().toISOString(),
    host: hostInfo(),
    tree,
    addon,
    llama,
    model: {
      path: MODEL_PATH,
      realpath: await realpath(MODEL_PATH),
      bytes: manifest.model.bytes,
      sha256: manifest.model.sha256,
    },
    fixture: {
      id: manifest.id,
      payload: fixture.manifest.payload,
      cases: fixture.cases.map(({ name, promptTokens, sha256: idsSha256 }) => ({ name, promptTokens, idsSha256 })),
    },
    tokenizer,
    protocol: { ...manifest.protocol, port: PORT, loadCooldownSeconds: LOAD_COOLDOWN_SECONDS, admission: POLICY },
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
  if (loaded.paged !== true) throw new Error('Block-paged cache inactive; not the recorded protocol');
  const tokenizer = await verifyTokenizer(join(loaded.assets as string, 'tokenizer.json'));
  await writeJson(join(out, 'prewarm.json'), { loaded, tokenizer });
}

// The admitted-prompt trace hashes each u32 token ID little-endian.
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
  console.log('Native session prompt IDs match the pinned IDs for all cases.');
}

async function matrix(out: string, allowDirty: boolean): Promise<void> {
  assertCleanEnv();
  const manifest = await readManifest();
  const { protocol } = manifest;
  const files = paths(out);
  if (!existsSync(files.provenance)) throw new Error(`${files.provenance} is missing; run preflight first`);
  const provenance = await readJson<Provenance>(files.provenance);
  assertHostReady();
  const tree = checkout(allowDirty);
  if (
    tree.commit !== provenance.tree.commit ||
    tree.mlx !== provenance.tree.mlx ||
    tree.dirty !== provenance.tree.dirty
  ) {
    throw new Error('The checkout changed since preflight');
  }
  const llama = await llamaIdentity();
  if (llama.sha256 !== provenance.llama.sha256 || !isDeepStrictEqual(llama.dylibs, provenance.llama.dylibs)) {
    throw new Error('llama.cpp binaries changed since preflight');
  }
  if (!isDeepStrictEqual(await addonIdentity(), provenance.addon)) {
    throw new Error('The mlx-node addon changed since preflight');
  }
  await assertModel(provenance.model);
  const { cases } = await loadFixture(FIXTURE);

  const repetitions = Array.from({ length: protocol.repetitions }, (_, index) => index + 1);
  const key = (runtime: Runtime, name: string, run: number) => `${runtime}/${name}/${run}`;
  const expected = new Set(
    RUNTIMES.flatMap((runtime) => cases.flatMap((item) => repetitions.map((run) => key(runtime, item.name, run)))),
  );
  const samples = existsSync(files.samples) ? await readJson<Sample[]>(files.samples) : [];
  const completed = new Set(samples.map((sample) => key(sample.runtime, sample.name, sample.run)));
  if (completed.size !== samples.length || ![...completed].every((id) => expected.has(id))) {
    throw new Error(`${files.samples} does not belong to this matrix`);
  }
  for (const sample of samples) {
    if (sample.numTokens !== protocol.outputTokens || sample.cachedTokens !== 0) {
      throw new Error(`${files.samples} holds a sample outside the protocol`);
    }
  }

  await mkdir(files.raw, { recursive: true });
  await mkdir(files.system, { recursive: true });
  const progress = new Progress(files.progress);
  const stamp = utcStamp();
  provenance.matrix = { startedAt: provenance.matrix?.startedAt ?? new Date().toISOString() };
  await writeJson(files.provenance, provenance);
  const caffeinate = spawn('caffeinate', ['-i', '-w', String(process.pid)], { stdio: 'ignore' });
  const stopGpu = recordGpu(files.gpu, () => progress.last);
  try {
    if (samples.length) {
      progress.emit({ event: 'resumed', savedSamples: samples.length, remaining: expected.size - samples.length });
    }
    for (const [index, item] of cases.entries()) {
      const policy = POLICY[item.promptTokens];
      for (const runtime of index % 2 === 0 ? RUNTIMES : [...RUNTIMES].reverse()) {
        const missing = repetitions.filter((run) => !completed.has(key(runtime, item.name, run)));
        if (!missing.length) continue;
        const kind = KIND[runtime];
        const block = `block-${item.name}-${stamp}`;
        await admit(policy, progress, `${kind}-${block}-load`, LOAD_COOLDOWN_SECONDS);
        let runner: Runner;
        if (runtime === 'mlx-node') {
          const worker = await MlxWorker.start(block, out, protocol.mlxEnv);
          progress.emit({ event: 'loaded', runtime, run: block, metadata: worker.loaded });
          if (worker.loaded.paged !== true) {
            await worker.close();
            throw new Error('Block-paged cache inactive; not the recorded protocol');
          }
          runner = worker;
        } else {
          const server = await LlamaServer.start(block, out, protocol.llamaServerArgs);
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
          for (const run of missing) {
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
            const {
              text: _t,
              thinking: _k,
              rawText: _r,
              publicRawText: _p,
              final: _f,
              tokens: _i,
              ...compact
            } = result;
            samples.push(compact as Sample);
            completed.add(key(runtime, item.name, run));
            await writeJsonAtomic(files.samples, samples);
            progress.emit(compact);
          }
        } finally {
          await runner.close();
        }
      }
    }
    if (completed.size !== expected.size) throw new Error('Incomplete measurement matrix');
    progress.emit({ event: 'complete', samples: samples.length });
  } finally {
    await stopGpu();
    caffeinate.kill();
  }
  provenance.matrix.finishedAt = new Date().toISOString();
  provenance.postRun = hostSnapshot();
  await writeJson(files.provenance, provenance);
}

async function printReport(out: string): Promise<void> {
  const rows = await report(out);
  console.log(
    '| Prompt tokens | mlx-node prefill | llama.cpp prefill | Ratio | mlx-node decode | llama.cpp decode | Ratio |',
  );
  console.log('|---:|---:|---:|---:|---:|---:|---:|');
  for (const row of rows) {
    const [mlx, llama] = [row['mlx-node'], row['llama.cpp']];
    console.log(
      `| ${row.promptTokens.toLocaleString('en-US')} | ${mlx.prefillTokensPerSecond.median.toFixed(1)} | ${llama.prefillTokensPerSecond.median.toFixed(1)} | ${row.prefillRatio.toFixed(3)}× | ${mlx.decodeTokensPerSecond.median.toFixed(2)} | ${llama.decodeTokensPerSecond.median.toFixed(2)} | ${row.decodeRatio.toFixed(3)}× |`,
    );
  }
  console.log(['samples.csv', 'summary.json', 'chart.svg', 'compare.md'].map((name) => join(out, name)).join('\n'));
}

async function main(): Promise<void> {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      out: { type: 'string' },
      'allow-dirty': { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h' },
    },
  });
  const step = positionals[0] as Step;
  if (values.help || positionals.length !== 1 || !STEPS.includes(step)) {
    console.log(
      `Usage: LLAMA_SERVER=/path/to/llama-server oxnode scripts/perf-tracker/qwen38/run.ts <${STEPS.join('|')}> [--out <dir>]\n` +
        'Without --out a new run directory is created; with --out the step continues or reports that run.\n' +
        'all = preflight, prewarm, matrix, report. check-ids is an optional untimed native prompt-ID check.\n' +
        '--allow-dirty records an uncommitted checkout instead of refusing it (development only).',
    );
    if (!values.help) process.exitCode = 1;
    return;
  }
  if (step === 'report' && !values.out && !process.env.OUT_DIR) throw new Error('report needs --out <run dir>');
  const out = await outDir('qwen38', values.out);
  const allowDirty = values['allow-dirty'];
  console.log(`qwen38 ${step}: ${out}`);
  if (step === 'preflight' || step === 'all') await preflight(out, allowDirty);
  if (step === 'prewarm' || step === 'all') await prewarm(out);
  if (step === 'check-ids') await checkIds(out);
  if (step === 'matrix' || step === 'all') await matrix(out, allowDirty);
  if (step === 'report' || step === 'all') await printReport(out);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? (error.stack ?? error.message) : error);
  process.exitCode = 1;
});
