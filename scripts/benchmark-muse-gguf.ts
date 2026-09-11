#!/usr/bin/env oxnode
/// <reference types="node" />

import assert from 'node:assert/strict';
import { spawn, execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, openSync, closeSync } from 'node:fs';
import { readFile, writeFile, mkdir, realpath, readdir, stat } from 'node:fs/promises';
import { createServer } from 'node:net';
import { homedir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

// Recorded conversations are data: this runner never executes their tool calls.
const script = fileURLToPath(import.meta.url);
const root = resolve(dirname(script), '..');
const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    model: {
      type: 'string',
      default: join(homedir(), '.mlx-node/models/muse-glimmer-30b-gguf/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf'),
    },
    'llama-server': { type: 'string' },
    output: { type: 'string', default: '.cache/benchmarks/muse-gguf-2026-09-11' },
    repetitions: { type: 'string', default: '3' },
    tokens: { type: 'string', default: '96' },
    cooldown: { type: 'string', default: '20' },
    runtime: { type: 'string' },
    spec: { type: 'boolean', default: false },
    case: { type: 'string', default: 'diff-review' },
    run: { type: 'string', default: 'pilot' },
  },
});
const mode = positionals[0];
assert(
  ['prepare', 'run', 'worker'].includes(mode!),
  'Usage: oxnode scripts/benchmark-muse-gguf.ts prepare|run|worker --llama-server /path/to/llama-server',
);
assert(values['llama-server'], '--llama-server is required');
const modelFile = await realpath(resolve(values.model));
const llamaServer = await realpath(resolve(values['llama-server']));
const dataDir = resolve(values.output);
const outputTokens = Number(values.tokens);
const repetitions = Number(values.repetitions);
const cooldown = Number(values.cooldown);
assert(Number.isInteger(outputTokens) && outputTokens > 1);
assert(Number.isInteger(repetitions) && repetitions > 0);
assert(Number.isFinite(cooldown) && cooldown >= 0);
await mkdir(join(dataDir, 'raw'), { recursive: true });
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const save = (path: string, value: unknown) => writeFile(path, JSON.stringify(value, null, 2) + '\n');
const command = (file: string, args: string[], cwd = root) =>
  execFileSync(file, args, { cwd, encoding: 'utf8' }).trim();
const env = {
  ...process.env,
  MLX_AGENT_METRICS: '0',
  MLX_PERSIST_PAGED_CACHE: '0',
  MLX_PAGED_PREFILL_CHUNK_SIZE: '512',
};
Object.assign(process.env, env);
const core = () => import(pathToFileURL(join(root, 'packages/core/index.cjs')).href);
async function digestFile(path: string) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}
type Input = {
  name: string;
  messages: any[];
  tools: any[];
  promptTokens: number;
  tokenIds: number[];
  rendered: string;
  sha256: string;
};
async function readInputs(): Promise<Input[]> {
  return JSON.parse(await readFile(join(dataDir, 'inputs.json'), 'utf8'));
}
async function draftPath() {
  const parent = dirname(modelFile);
  for (const name of [`dflash-${basename(modelFile)}`, 'dflash-kquant.gguf']) {
    const path = join(parent, name);
    if (
      await stat(path).then(
        (x) => x.isFile(),
        () => false,
      )
    )
      return realpath(path);
  }
  const { ggufArchitecture } = await core();
  const paths = (await readdir(parent))
    .filter((n) => /^dflash-.*\.gguf$/i.test(n))
    .map((n) => join(parent, n))
    .filter((p) => ggufArchitecture(p) === 'dflash');
  assert.equal(paths.length, 1, 'An unambiguous DFlash companion is required');
  return realpath(paths[0]!);
}
async function prepare() {
  assert(
    !(await readdir(join(dataDir, 'raw'))).some((n) => /-\d+\.json$/.test(n)),
    'Measured samples already exist; use a new --output directory to prepare another benchmark',
  );
  const manifest = JSON.parse(await readFile(join(root, 'scripts/fixtures/gemma4-oxc-review-v1.json'), 'utf8'));
  const original = await readFile(resolve(root, manifest.payload.path));
  assert.equal(sha(original), manifest.payload.sha256, 'Fetch the pinned fixture with scripts/benchmark-fixture.ts');
  const { Qwen3Tokenizer, prepareMuseGlimmerGguf } = await core();
  const prepared = await prepareMuseGlimmerGguf(modelFile);
  const tok = await Qwen3Tokenizer.fromPretrained(join(prepared, 'tokenizer.json'));
  const config = JSON.parse(await readFile(join(prepared, 'config.json'), 'utf8'));
  const inputs: Input[] = [];
  for (const source of JSON.parse(original.toString())) {
    const tokenIds = Array.from(await tok.applyChatTemplate(source.messages, true, source.tools, true)) as number[];
    inputs.push({
      ...source,
      sourcePromptTokens: source.promptTokens,
      sourceTokenIdsSha256: source.sha256,
      promptTokens: tokenIds.length,
      tokenIds,
      rendered: await tok.decode(new Uint32Array(tokenIds), false),
      sha256: sha(JSON.stringify(tokenIds)),
    });
  }
  await save(join(dataDir, 'inputs.json'), inputs);
  const draft = await draftPath();
  // llama.cpp's DFlash block includes the anchor; use the common supported
  // width in both engines, derived from model metadata rather than hardware.
  const draftTokens = config.dflash_config.block_size - 1;
  assert(Number.isInteger(draftTokens) && draftTokens > 0);
  const contextCapacity =
    Math.ceil((Math.max(...inputs.map((x) => x.promptTokens)) + outputTokens + draftTokens + 512) / 512) * 512;
  const patch = command('git', ['diff', '--binary']);
  await writeFile(join(dataDir, 'mlx-runtime.patch'), patch);
  const files = [
    modelFile,
    draft,
    join(prepared, 'config.json'),
    join(prepared, 'tokenizer.json'),
    join(root, 'packages/core/mlx-core.darwin-arm64.node'),
    join(root, 'packages/lm/dist/chat-session.js'),
    join(dataDir, 'inputs.json'),
    llamaServer,
    script,
  ];
  const prior = await readFile(join(dataDir, 'environment.json'), 'utf8').then(JSON.parse, () => ({}));
  const identities = [];
  for (const path of files) {
    const before = await stat(path);
    const unchanged = prior.identities?.find(
      (x: any) =>
        x.path === path && x.bytes === before.size && x.mtimeMs === before.mtimeMs && x.ctimeMs === before.ctimeMs,
    );
    const digest = unchanged?.sha256 ?? (await digestFile(path));
    const after = await stat(path);
    assert.equal(after.mtimeMs, before.mtimeMs, `File changed during hashing: ${path}`);
    assert.equal(after.ctimeMs, before.ctimeMs, `File changed during hashing: ${path}`);
    identities.push({ path, bytes: before.size, mtimeMs: before.mtimeMs, ctimeMs: before.ctimeMs, sha256: digest });
  }
  const llamaVersion = spawnSync(llamaServer, ['--version'], { encoding: 'utf8' });
  assert.equal(llamaVersion.status, 0);
  await save(join(dataDir, 'environment.json'), {
    createdAt: new Date().toISOString(),
    modelFile,
    draft,
    prepared,
    draftTokens,
    contextCapacity,
    outputTokens,
    repetitions,
    cooldown,
    fixture: { id: manifest.id, sha256: manifest.payload.sha256, source: manifest.source },
    cases: inputs.map((x) => ({ name: x.name, promptTokens: x.promptTokens, sha256: x.sha256 })),
    identities,
    mlxCommit: command('git', ['rev-parse', 'HEAD']),
    mlxPatchSha256: sha(patch),
    llamaVersion: (llamaVersion.stdout + llamaVersion.stderr).trim(),
    hardware: command('system_profiler', ['SPHardwareDataType', 'SPDisplaysDataType'])
      .split('\n')
      .filter((l) => !/Serial Number|Hardware UUID|Provisioning UDID/.test(l))
      .join('\n'),
    os: command('sw_vers', []),
    thermal: command('pmset', ['-g', 'therm']),
    protocol: `Three real review boundaries; identical Muse input token IDs; greedy, high thinking; exactly ${outputTokens} generated tokens, natural EOS retained and short completions rejected. Fresh process per sample; 32-token shortest-case warmup, reset prompt cache, zero cache hits; serial runs with cooldown. Production mlx-node LM ChatSession with owner-scoped cache lifecycle versus llama.cpp completion server. BF16 target/draft KV, 512-token physical prefill; llama.cpp chooses CPU thread count. DFlash fixed common width, MLX adaptive fallback disabled. Loading/warmup excluded. Native prefill through first token; decode excludes first token; request wall time separately. Historical tool calls are never executed.`,
  });
  console.log(
    JSON.stringify({
      event: 'prepared',
      draftTokens,
      contextCapacity,
      cases: inputs.map((x) => ({ name: x.name, n: x.promptTokens })),
    }),
  );
}
async function post(base: string, path: string, body: unknown): Promise<any> {
  const response = await fetch(`${base}/${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(900_000),
  });
  const text = await response.text();
  assert(response.ok, `${path}: ${response.status}: ${text}`);
  return JSON.parse(text);
}
async function freePort(): Promise<number> {
  const server = createServer();
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  const address = server.address();
  assert(address && typeof address === 'object');
  await new Promise<void>((r, reject) => server.close((e) => (e ? reject(e) : r())));
  return address.port;
}
async function worker() {
  const inputs = await readInputs();
  const data = inputs.find((x) => x.name === values.case);
  assert(data);
  const setup = JSON.parse(await readFile(join(dataDir, 'environment.json'), 'utf8'));
  assert.equal(setup.modelFile, modelFile);
  assert.equal(setup.outputTokens, outputTokens);
  const id = `${values.runtime}-${values.spec ? 'dflash' : 'ar'}-${data.name}-${values.run}`;
  const thermalBefore = command('pmset', ['-g', 'therm']);
  const start = performance.now();
  let sample: any;
  if (values.runtime === 'mlx') {
    const { getMemorySnapshot, resetPeakMemory } = await core();
    const { loadModel, ChatSession } = await import(pathToFileURL(join(root, 'packages/lm/dist/index.js')).href);
    const model = await loadModel(modelFile);
    const session = new ChatSession(model);
    assert(model.hasMtpWeights() && model.hasBlockPagedCache());
    const loadMs = performance.now() - start;
    const config = {
      temperature: 0,
      topK: 1,
      topP: 1,
      minP: 0,
      repetitionPenalty: 1,
      presencePenalty: 0,
      frequencyPenalty: 0,
      reasoningEffort: 'high',
      enableMtp: values.spec,
      mtpDepth: setup.draftTokens,
      mtpAdaptiveDepth: false,
      reuseCache: true,
      reportPerformance: true,
    };
    session.primeHistory(inputs[0]!.messages);
    await session.startFromHistory({ ...config, tools: inputs[0]!.tools, maxNewTokens: 32 });
    await session.reset();
    session.primeHistory(data.messages);
    resetPeakMemory();
    const measured = performance.now();
    const result = await session.startFromHistory({
      ...config,
      tools: data.tools,
      maxNewTokens: outputTokens,
    });
    const wallMs = performance.now() - measured;
    assert.equal(result.promptTokens, data.promptTokens);
    assert.equal(result.cachedTokens, 0);
    assert.equal(result.numTokens, outputTokens, 'Natural completion ended before the fixed output length');
    if (values.spec) assert(result.performance.mtpCycles > 0, 'DFlash must actually run');
    const p = result.performance;
    sample = {
      loadMs,
      wallMs,
      prefillMs: p.ttftMs,
      prefillTps: p.prefillTokensPerSecond,
      decodeMs: ((result.numTokens - 1) / p.decodeTokensPerSecond) * 1000,
      decodeTps: p.decodeTokensPerSecond,
      generatedTokens: result.numTokens,
      cachedTokens: result.cachedTokens,
      outputSha256: sha(result.rawText),
      memory: getMemorySnapshot(),
      result,
    };
    await session.dispose();
  } else {
    assert.equal(values.runtime, 'llama');
    const port = await freePort();
    const base = `http://127.0.0.1:${port}`;
    const args = [
      '-m',
      modelFile,
      '-ngl',
      '999',
      '-fa',
      'on',
      '-ctk',
      'bf16',
      '-ctv',
      'bf16',
      '-c',
      String(setup.contextCapacity),
      '-b',
      '2048',
      '-ub',
      '512',
      '-np',
      '1',
      '--host',
      '127.0.0.1',
      '--port',
      String(port),
      '--no-context-shift',
    ];
    if (values.spec)
      args.push(
        '--spec-type',
        'draft-dflash',
        '-md',
        setup.draft,
        '-ngld',
        '999',
        '-ctkd',
        'bf16',
        '-ctvd',
        'bf16',
        '--spec-draft-n-max',
        String(setup.draftTokens),
      );
    else args.push('--spec-type', 'none');
    const fd = openSync(join(dataDir, 'raw', `${id}.server.log`), 'w');
    const server = spawn(llamaServer, args, { env, stdio: ['ignore', fd, fd] });
    closeSync(fd);
    const exited = new Promise<void>((r, reject) => {
      server.once('error', reject);
      server.once('exit', () => r());
    });
    try {
      let ready = false;
      for (let i = 0; i < 600; i++) {
        assert(server.exitCode === null && server.signalCode === null, 'llama-server exited; inspect its log');
        try {
          const r = await fetch(`${base}/health`, { signal: AbortSignal.timeout(1000) });
          if (r.ok) {
            ready = true;
            break;
          }
        } catch {}
        await sleep(250);
      }
      assert(ready, 'llama-server startup timeout');
      const loadMs = performance.now() - start;
      const tokenized = await post(base, 'tokenize', {
        content: data.rendered,
        add_special: false,
        parse_special: true,
      });
      assert.deepEqual(tokenized.tokens, data.tokenIds, 'Tokenizer mismatch');
      const props = await fetch(`${base}/props`).then((r) => r.json());
      const request = {
        temperature: 0,
        top_k: 1,
        top_p: 1,
        min_p: 0,
        repeat_penalty: 1,
        presence_penalty: 0,
        frequency_penalty: 0,
        seed: 1234,
        cache_prompt: false,
        stream: false,
        return_tokens: true,
        ignore_eos: false,
      };
      await post(base, 'completion', { ...request, prompt: inputs[0]!.tokenIds, n_predict: 32 });
      const measured = performance.now();
      const result = await post(base, 'completion', { ...request, prompt: data.tokenIds, n_predict: outputTokens });
      const wallMs = performance.now() - measured;
      const t = result.timings;
      assert.equal(t.prompt_n, data.promptTokens);
      assert.equal(t.cache_n, 0);
      assert(!result.truncated);
      assert.equal(t.predicted_n, outputTokens, 'Natural completion ended before the fixed output length');
      if (values.spec) assert(t.draft_n > 0, 'DFlash must actually run');
      sample = {
        loadMs,
        wallMs,
        args,
        props,
        prefillMs: t.prompt_ms,
        prefillTps: t.prompt_per_second,
        decodeMs: t.predicted_ms,
        decodeTps: t.predicted_per_second,
        generatedTokens: t.predicted_n,
        cachedTokens: t.cache_n,
        outputSha256: sha(result.content),
        result,
      };
    } finally {
      server.kill('SIGTERM');
      await Promise.race([exited, sleep(5000)]);
      if (server.exitCode === null && server.signalCode === null) {
        server.kill('SIGKILL');
        await exited;
      }
    }
  }
  sample = {
    id,
    runtime: values.runtime,
    speculation: values.spec ? 'dflash' : 'off',
    case: data.name,
    run: values.run,
    promptTokens: data.promptTokens,
    inputSha256: data.sha256,
    completedAt: new Date().toISOString(),
    setupSha256: sha(JSON.stringify(setup)),
    thermalBefore,
    thermalAfter: command('pmset', ['-g', 'therm']),
    outputTokensCap: outputTokens,
    ...sample,
  };
  await save(join(dataDir, 'raw', `${id}.json`), sample);
  console.log(JSON.stringify({ ...sample, result: undefined, props: undefined, args: undefined, memory: undefined }));
}
async function run() {
  const inputs = await readInputs();
  const setup = JSON.parse(await readFile(join(dataDir, 'environment.json'), 'utf8'));
  assert.equal(setup.modelFile, modelFile);
  assert.equal(setup.outputTokens, outputTokens);
  assert.equal(setup.repetitions, repetitions);
  assert.equal(setup.cooldown, cooldown);
  const setupSha256 = sha(JSON.stringify(setup));
  for (const identity of setup.identities) {
    const current = await stat(identity.path);
    assert.equal(current.size, identity.bytes, `Benchmark input changed: ${identity.path}`);
    assert.equal(current.mtimeMs, identity.mtimeMs, `Benchmark input changed: ${identity.path}`);
    assert.equal(current.ctimeMs, identity.ctimeMs, `Benchmark input changed: ${identity.path}`);
  }
  const variants = [
    { runtime: 'mlx', spec: false },
    { runtime: 'llama', spec: false },
    { runtime: 'mlx', spec: true },
    { runtime: 'llama', spec: true },
  ];
  for (let repetition = 1; repetition <= repetitions; repetition++)
    for (const [index, input] of inputs.entries()) {
      const order = (repetition + index) % 2 ? variants : [...variants].reverse();
      for (const variant of order) {
        const id = `${variant.runtime}-${variant.spec ? 'dflash' : 'ar'}-${input.name}-${repetition}`;
        if (
          await stat(join(dataDir, 'raw', `${id}.json`)).then(
            () => true,
            () => false,
          )
        ) {
          const previous = JSON.parse(await readFile(join(dataDir, 'raw', `${id}.json`), 'utf8'));
          assert.equal(previous.setupSha256, setupSha256, `Cannot resume stale sample ${id}`);
          console.log(JSON.stringify({ event: 'resume-skip', id }));
          continue;
        }
        console.log(JSON.stringify({ event: 'start', at: new Date().toISOString(), id }));
        const fd = openSync(join(dataDir, 'raw', `${id}.worker.log`), 'w');
        const args = [
          script,
          'worker',
          '--model',
          modelFile,
          '--llama-server',
          llamaServer,
          '--output',
          dataDir,
          '--tokens',
          String(outputTokens),
          '--runtime',
          variant.runtime,
          '--case',
          input.name,
          '--run',
          String(repetition),
        ];
        if (variant.spec) args.push('--spec');
        const child = spawn('oxnode', args, { env, stdio: ['ignore', fd, fd] });
        closeSync(fd);
        const code = await new Promise<number | null>((r, reject) => {
          child.once('error', reject);
          child.once('exit', r);
        });
        assert.equal(code, 0, `Failed ${id}; inspect raw/${id}.worker.log`);
        const sample = JSON.parse(await readFile(join(dataDir, 'raw', `${id}.json`), 'utf8'));
        console.log(
          JSON.stringify({
            event: 'complete',
            id,
            prefillTps: sample.prefillTps,
            decodeTps: sample.decodeTps,
            generatedTokens: sample.generatedTokens,
            wallMs: sample.wallMs,
          }),
        );
        await sleep(cooldown * 1000);
      }
    }
}
if (mode === 'prepare') await prepare();
else if (mode === 'worker') await worker();
else await run();
