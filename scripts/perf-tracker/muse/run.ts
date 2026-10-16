#!/usr/bin/env oxnode
/// <reference types="node" />

import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { constants, existsSync } from 'node:fs';
import { appendFile, copyFile, readdir, realpath, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { parseArgs } from 'node:util';

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
  sleep,
  writeJson,
} from '../common.js';
import { guarded, jobPassed, type JobRecord } from './guard.js';
import { report } from './report.js';
import {
  BOOTSTRAP,
  CASES,
  DRAFT,
  FIXTURE,
  HERE,
  INPUTS_SHA256,
  PROTOCOL,
  RUNNER,
  RUNNER_SHA256,
  SCRIPTS,
  SEPTEMBER,
  TARGET,
  matrixJobs,
  type Identity,
  type Provenance,
  type Sample,
  type Setup,
} from './suite.js';

const STEPS = ['preflight', 'freeze', 'prewarm', 'prepare', 'finalize', 'matrix', 'report'] as const;
type Step = (typeof STEPS)[number];

const USAGE = `Usage: oxnode scripts/perf-tracker/muse/run.ts <step> [--out <dir>] [--allow-dirty] [--archived]

Steps, in order: ${STEPS.join(', ')}; "all" runs them and skips the finished ones.
Without --out, preflight and all start a new run directory; other steps continue --out.
LLAMA_SERVER names a llama-server built at llama.cpp a14dba686aaa; MODELS_DIR defaults to <repo>/.cache/models.
  --allow-dirty  development only: preflight a dirty checkout and record dirty: true
  --archived     report only: re-report a run whose inputs and checkout have since moved
                 (skips rehashing and the source check)`;

// The frozen copies and their packages/core sources.
const FROZEN: Record<string, string> = {
  'current.node': 'mlx-core.darwin-arm64.node',
  'mlx.metallib': 'mlx.metallib',
  'paged_attn.metallib': 'paged_attn.metallib',
};
// A cold native GGUF build writes a ~22 GB packed copy; the matrix limit does not fit it.
const PREWARM_MINUTES = 120;

const exists = (path: string) =>
  stat(path).then(
    () => true,
    () => false,
  );

async function passed(path: string): Promise<boolean> {
  return (await exists(path)) && jobPassed(await readJson<JobRecord>(path));
}

function sourceState(allowDirty: boolean): Provenance['source'] {
  if (!allowDirty) return { ...assertCleanTree(), dirty: false };
  const status = sh('git', ['status', '--porcelain', '--ignore-submodules=none'], { cwd: ROOT });
  return {
    commit: sh('git', ['rev-parse', 'HEAD'], { cwd: ROOT }),
    mlx: sh('git', ['rev-parse', 'HEAD'], { cwd: join(ROOT, 'crates/mlx-sys/mlx') }),
    dirty: status !== '',
    status,
  };
}

function checkSource(provenance: Provenance) {
  const head = sh('git', ['rev-parse', 'HEAD'], { cwd: ROOT });
  if (head !== provenance.source.commit) throw new Error(`HEAD moved from ${provenance.source.commit} to ${head}`);
  if (!provenance.source.dirty) assertCleanTree();
}

async function assertNoSamples(out: string) {
  const raw = join(out, 'raw');
  if (existsSync(raw) && (await readdir(raw)).some((name) => /-\d+\.json$/.test(name))) {
    throw new Error(`${raw} already holds measured samples`);
  }
}

async function identity(path: string): Promise<Identity> {
  const real = await realpath(path);
  const before = await stat(real);
  const digest = await sha256File(real);
  const after = await stat(real);
  if (after.mtimeMs !== before.mtimeMs || after.ctimeMs !== before.ctimeMs) {
    throw new Error(`File changed during hashing: ${real}`);
  }
  return { path: real, bytes: before.size, mtimeMs: before.mtimeMs, ctimeMs: before.ctimeMs, sha256: digest };
}

async function validateIdentities(setup: Setup) {
  for (const item of setup.identities) {
    const current = await stat(item.path);
    if (current.size !== item.bytes || current.mtimeMs !== item.mtimeMs || current.ctimeMs !== item.ctimeMs) {
      throw new Error(`Benchmark input changed: ${item.path}`);
    }
  }
}

function workerEnv(out: string, extra: Record<string, string> = {}): NodeJS.ProcessEnv {
  return {
    ...process.env,
    MUSE_NATIVE_VARIANT: join(out, 'current.node'),
    MLX_PAGED_ATTN_METALLIB: join(out, 'paged_attn.metallib'),
    ...extra,
  };
}

async function preflight(out: string, allowDirty: boolean) {
  const source = sourceState(allowDirty);
  await assertFileSha256(RUNNER, RUNNER_SHA256);
  for (const file of ['packages/lm/dist/index.js', 'packages/lm/dist/chat-session.js']) {
    if (!existsSync(join(ROOT, file))) throw new Error(`Missing ${file}; build the TypeScript packages first`);
  }
  const addon = await addonIdentity();
  const llama = await llamaIdentity();
  const target = await realpath(TARGET.path);
  const draft = await realpath(DRAFT.path);
  for (const asset of ['config.json', 'tokenizer.json']) {
    if (!existsSync(join(dirname(target), asset))) throw new Error(`Muse needs ${asset} beside ${target}`);
  }
  await assertFileSha256(target, TARGET.sha256);
  await assertFileSha256(draft, DRAFT.sha256);
  const fixture = await loadFixture(FIXTURE);
  const oxnode = sh('/usr/bin/which', ['oxnode']);
  const provenance = {
    createdAt: new Date().toISOString(),
    host: hostInfo(),
    source,
    oxnode: { path: oxnode, version: sh(oxnode, ['--version']) },
    nodeOnPath: sh('node', ['--version']),
    addon,
    runner: { path: RUNNER, sha256: RUNNER_SHA256 },
    llama,
    models: { target, targetSha256: TARGET.sha256, draft, draftSha256: DRAFT.sha256 },
    // Muse renders these frozen messages with its own template in prepare.
    fixture: {
      id: fixture.manifest.id,
      payloadSha256: fixture.manifest.payload.sha256,
      path: fixture.path,
      cases: fixture.manifest.cases,
    },
    hostSnapshot: hostSnapshot(),
  };
  await writeJson(join(out, 'provenance.json'), provenance);
  console.log(JSON.stringify({ ...provenance, hostSnapshot: undefined }, null, 2));
}

async function freeze(out: string, provenance: Provenance) {
  checkSource(provenance);
  await assertNoSamples(out);
  const frozen: Record<string, string> = {};
  for (const [name, source] of Object.entries(FROZEN)) {
    await copyFile(join(ROOT, 'packages/core', source), join(out, name), constants.COPYFILE_EXCL);
    frozen[name] = await sha256File(join(out, name));
    if (frozen[name] !== provenance.addon[source]) throw new Error(`packages/core/${source} changed after preflight`);
  }
  const protocol = {
    createdAt: new Date().toISOString(),
    sourceCommit: provenance.source.commit,
    dirty: provenance.source.dirty,
    suite: '3 real histories x 2 engines x DFlash off/on x 3 repetitions = 36 fresh samples',
    warmup: PROTOCOL.warmup,
    outputTokens: PROTOCOL.outputTokens,
    cooldownSeconds: PROTOCOL.cooldownSeconds,
    retention:
      'Every successful accounting-valid sample is kept; host activity is recorded and never used to exclude samples.',
    draftPolicy:
      'Same validated five-layer DFlash GGUF in both engines; model-derived 15-proposal cap, MLX adaptive depth/fallback disabled for matched width. No machine-specific thread or attention-plan overrides.',
    frozenNativeSha256: frozen['current.node'],
    frozenPagedMetalSha256: frozen['paged_attn.metallib'],
    frozenMlxMetalSha256: frozen['mlx.metallib'],
    comparedWith: `September 2026 full suite, source ${SEPTEMBER.commit.slice(0, 12)}, setup ${SEPTEMBER.setupSha256}: scripts/perf-tracker/muse/baseline.csv`,
    hostLoad: hostSnapshot(),
  };
  await writeJson(join(out, 'protocol.json'), protocol);
  console.log(JSON.stringify(protocol, null, 2));
}

// Builds the native GGUF cache entry before the guarded prepare, which in
// September was a cache hit. Loads no model weights onto the GPU.
async function prewarm(out: string, provenance: Provenance) {
  if (!existsSync(join(out, 'protocol.json'))) throw new Error('Run freeze first');
  checkSource(provenance);
  const record = await guarded({
    dir: out,
    label: 'prewarm',
    command: provenance.oxnode.path,
    args: [BOOTSTRAP, 'prewarm', provenance.models.target],
    env: workerEnv(out),
    maxMinutes: PREWARM_MINUTES,
  });
  if (!jobPassed(record)) throw new Error(`prewarm failed (${record.reason ?? record.code}); inspect prewarm.log`);
}

async function prepare(out: string, provenance: Provenance) {
  if (!(await passed(join(out, 'prewarm.job.json')))) throw new Error('Run prewarm first');
  checkSource(provenance);
  const record = await guarded({
    dir: out,
    label: 'prepare',
    command: provenance.oxnode.path,
    args: [
      BOOTSTRAP,
      'prepare',
      '--model',
      provenance.models.target,
      '--llama-server',
      provenance.llama.path,
      '--output',
      out,
      '--repetitions',
      String(PROTOCOL.repetitions),
      '--tokens',
      String(PROTOCOL.outputTokens),
      '--warmup-tokens',
      String(PROTOCOL.warmup.maxTokens),
      '--warmup-case',
      PROTOCOL.warmup.case,
      '--cooldown',
      String(PROTOCOL.cooldownSeconds),
    ],
    env: workerEnv(out),
    maxMinutes: PROTOCOL.jobMinutes,
  });
  if (!jobPassed(record)) throw new Error(`prepare failed (${record.reason ?? record.code}); inspect prepare.log`);
}

async function finalize(out: string, provenance: Provenance) {
  if (!(await done('prepare', out))) throw new Error('Run prepare first');
  checkSource(provenance);
  const file = join(out, 'environment.json');
  const setup = await readJson<Setup>(file);
  assert.equal(await sha256File(join(out, 'inputs.json')), INPUTS_SHA256, 'Muse-rendered inputs differ from September');
  assert.deepEqual(setup.cases, CASES, 'Prepared cases differ from September');
  assert.equal(setup.draftTokens, PROTOCOL.draftTokens);
  assert.equal(setup.contextCapacity, PROTOCOL.contextCapacity);
  assert.equal(setup.outputTokens, PROTOCOL.outputTokens);
  assert.deepEqual(setup.warmup, PROTOCOL.warmup);
  assert.equal(setup.repetitions, PROTOCOL.repetitions);
  assert.equal(setup.cooldown, PROTOCOL.cooldownSeconds);
  assert.equal(setup.fixture.sha256, provenance.fixture.payloadSha256);
  assert.equal(setup.modelFile, provenance.models.target);
  assert.equal(setup.draft, provenance.models.draft);
  assert.equal(setup.mlxCommit, provenance.source.commit);
  const installed = join(ROOT, 'packages/core/mlx-core.darwin-arm64.node');
  setup.installedNativeUnmeasured = setup.identities.find((item) => item.path === installed);
  setup.identities = setup.identities.filter((item) => item.path !== installed);
  setup.frozenNativePath = join(out, 'current.node');
  setup.frozenPagedMetalPath = join(out, 'paged_attn.metallib');
  setup.llamaServer = await realpath(provenance.llama.path);
  const add = async (path: string, expected?: string) => {
    const item = await identity(path);
    if (expected && item.sha256 !== expected) throw new Error(`${item.path} changed after preflight`);
    if (!setup.identities.some((known) => known.path === item.path)) setup.identities.push(item);
  };
  for (const [name, source] of Object.entries(FROZEN)) await add(join(out, name), provenance.addon[source]);
  for (const script of SCRIPTS) await add(join(HERE, script));
  await add(join(ROOT, 'packages/core/index.cjs'), provenance.addon['index.cjs']);
  await add(join(ROOT, 'yarn.lock'));
  const dist = join(ROOT, 'packages/lm/dist');
  const scripts = (await readdir(dist, { recursive: true })).filter((name) => name.endsWith('.js')).sort();
  for (const name of scripts) await add(join(dist, name));
  const llamaDir = dirname(setup.llamaServer);
  for (const [name, digest] of Object.entries(provenance.llama.dylibs)) await add(join(llamaDir, name), digest);
  setup.nativeOverride =
    'Scoped core import through bootstrap.ts; current.node is the measured binding. Installed native is recorded separately and is not used.';
  await writeJson(file, setup);
  console.log(
    JSON.stringify({
      nativeSha256: provenance.addon['mlx-core.darwin-arm64.node'],
      draftTokens: setup.draftTokens,
      contextCapacity: setup.contextCapacity,
      cases: setup.cases,
      identities: setup.identities.length,
    }),
  );
}

async function matrix(out: string, provenance: Provenance) {
  if (!(await done('finalize', out))) throw new Error('Run finalize first');
  const setup = await readJson<Setup>(join(out, 'environment.json'));
  if (!setup.frozenNativePath || !setup.frozenPagedMetalPath || !setup.llamaServer) {
    throw new Error('Run finalize first');
  }
  checkSource(provenance);
  const setupSha256 = sha256(JSON.stringify(setup));
  const jobs = matrixJobs(setup.cases);
  assert.equal(jobs.length, 36);
  const progress = async (event: Record<string, unknown>) => {
    console.log(JSON.stringify(event));
    await appendFile(join(out, 'progress.jsonl'), `${JSON.stringify(event)}\n`);
    await writeJson(join(out, 'state.json'), event);
  };
  const at = () => new Date().toISOString();
  for (const [index, job] of jobs.entries()) {
    await validateIdentities(setup);
    const path = join(out, 'raw', `${job.id}.json`);
    if (existsSync(path)) {
      const previous = await readJson<Sample>(path);
      assert.equal(previous.setupSha256, setupSha256, `Cannot resume stale sample ${job.id}`);
      assert.equal(previous.generatedTokens, PROTOCOL.outputTokens);
      assert.equal(previous.cachedTokens, 0);
      assert(await passed(join(out, `${job.id}.job.json`)), `${job.id}: sample exists but its job failed`);
      await progress({ event: 'resume-skip', completed: index + 1, total: jobs.length, id: job.id, at: at() });
      continue;
    }
    const env = workerEnv(
      out,
      job.runtime === 'mlx'
        ? { MLX_NODE_LOG: 'mlx_core::decode_tuning=info', MLX_NODE_LOG_FILE: join(out, `${job.id}.choices.jsonl`) }
        : {},
    );
    const args = [
      job.runtime === 'mlx' ? BOOTSTRAP : RUNNER,
      'worker',
      '--model',
      setup.modelFile,
      '--llama-server',
      setup.llamaServer,
      '--output',
      out,
      '--tokens',
      String(PROTOCOL.outputTokens),
      '--warmup-tokens',
      String(PROTOCOL.warmup.maxTokens),
      '--warmup-case',
      PROTOCOL.warmup.case,
      '--runtime',
      job.runtime,
      '--case',
      job.case.name,
      '--run',
      String(job.run),
      ...(job.speculation === 'dflash' ? ['--spec'] : []),
    ];
    await progress({ event: 'start', completed: index, total: jobs.length, id: job.id, at: at() });
    const record = await guarded({
      dir: out,
      label: job.id,
      command: provenance.oxnode.path,
      args,
      env,
      maxMinutes: PROTOCOL.jobMinutes,
      extraFiles: [join(out, 'raw', `${job.id}.server.log`)],
      hostBefore: hostSnapshot(),
    });
    if (!jobPassed(record)) {
      throw new Error(
        `Failed ${job.id} (${record.reason ?? record.code}); preserve ${job.id}.log and inspect before resuming`,
      );
    }
    const sample = await readJson<Sample>(path);
    assert.equal(sample.setupSha256, setupSha256);
    assert.equal(sample.generatedTokens, PROTOCOL.outputTokens);
    assert.equal(sample.cachedTokens, 0);
    assert.equal(sample.promptTokens, job.case.promptTokens);
    assert.equal(sample.inputSha256, job.case.sha256);
    if (job.speculation === 'dflash') {
      const drafted = job.runtime === 'mlx' ? sample.result.performance?.mtpCycles : sample.result.timings?.draft_n;
      assert((drafted ?? 0) > 0, `${job.id}: DFlash did not draft`);
    }
    await validateIdentities(setup);
    await progress({
      event: 'complete',
      completed: index + 1,
      total: jobs.length,
      id: job.id,
      at: at(),
      prefillTps: sample.prefillTps,
      decodeTps: sample.decodeTps,
      wallMs: sample.wallMs,
      outputSha256: sample.outputSha256,
    });
    if (index < jobs.length - 1) await sleep(PROTOCOL.cooldownSeconds * 1000);
  }
  await progress({ event: 'matrix-complete', completed: jobs.length, total: jobs.length, at: at() });
}

async function done(step: Step, out: string): Promise<boolean> {
  switch (step) {
    case 'preflight':
      return exists(join(out, 'provenance.json'));
    case 'freeze':
      return exists(join(out, 'protocol.json'));
    case 'prewarm':
      return passed(join(out, 'prewarm.job.json'));
    case 'prepare':
      return (await passed(join(out, 'prepare.job.json'))) && (await exists(join(out, 'environment.json')));
    case 'finalize':
      return (
        (await exists(join(out, 'environment.json'))) &&
        Boolean((await readJson<Setup>(join(out, 'environment.json'))).frozenNativePath)
      );
    default:
      return false;
  }
}

// Idle sleep would stall a guarded job and fail its time limit.
function keepAwake(): () => void {
  const child = spawn('caffeinate', ['-i', '-w', String(process.pid)], { stdio: 'ignore' });
  child.unref();
  return () => child.kill();
}

async function main() {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      out: { type: 'string' },
      'allow-dirty': { type: 'boolean', default: false },
      archived: { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) {
    console.log(USAGE);
    return;
  }
  const [step] = positionals;
  if (positionals.length !== 1 || (step !== 'all' && !(STEPS as readonly string[]).includes(step))) {
    throw new Error(`Expected one step\n\n${USAGE}`);
  }
  const fresh = step === 'preflight' || step === 'all';
  if (values['allow-dirty'] && !fresh) throw new Error('--allow-dirty applies to preflight and all');
  if (values.archived && step !== 'report') throw new Error('--archived applies to report');
  if (!values.out && !fresh) throw new Error(`${step} continues a run: pass --out <run dir>`);
  const out = await realpath(await outDir('muse', values.out));
  if (step === 'report') {
    await report(out, { archived: values.archived });
    return;
  }
  assertCleanEnv();
  const release = keepAwake();
  try {
    for (const name of step === 'all' ? STEPS : [step as Step]) {
      if (await done(name, out)) {
        if (step !== 'all') throw new Error(`${name} already ran in ${out}`);
        console.log(`== ${name}: already done`);
        continue;
      }
      console.log(`== ${name} ${out}`);
      if (name === 'preflight') {
        await preflight(out, values['allow-dirty']);
        continue;
      }
      if (name === 'report') {
        await report(out, { archived: false });
        continue;
      }
      if (!existsSync(join(out, 'provenance.json'))) throw new Error('Run preflight first');
      const provenance = await readJson<Provenance>(join(out, 'provenance.json'));
      await { freeze, prewarm, prepare, finalize, matrix }[name](out, provenance);
    }
  } finally {
    release();
  }
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
