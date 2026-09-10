import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { closeSync, mkdirSync, openSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../../..');
const modelDir = resolve(root, '.cache/models/gemma-4-12b-it-qat-q4_0-gguf');
const modelFile = resolve(modelDir, 'gemma-4-12b-it-qat-q4_0.gguf');
const llamaDir = '/Users/brooklyn/workspace/github/llama.cpp';
const llamaServer = resolve(llamaDir, 'build-codex-bench/bin/llama-server');
const dataDir = resolve(root, '.cache/benchmarks/gemma4-agent-2026-09-10');
const cases = ['diff-review', 'test-review', 'final-review'];
const outputTokens = 256;
const cooldownSeconds = 20;
const env = {
  ...process.env,
  MLX_AGENT_METRICS: '0',
  MLX_PERSIST_PAGED_CACHE: '0',
  MLX_PAGED_PREFILL_CHUNK_SIZE: '512',
};
const sha = (s: string) => createHash('sha256').update(s).digest('hex');
const core = () => import(resolve(root, 'packages/core/index.cjs'));
const save = (path: string, value: unknown) => writeFileSync(path, JSON.stringify(value, null, 2) + '\n');
const input = (name: string) =>
  JSON.parse(readFileSync(resolve(dataDir, 'inputs.json'), 'utf8')).find((x: any) => x.name === name);

async function prepare() {
  mkdirSync(resolve(dataDir, 'raw'), { recursive: true });
  const source =
    '/Users/brooklyn/.mlx-node/agent/sessions/--Users-brooklyn-workspace-github-oxc-node--/2026-09-03T15-07-41-314Z_01a067cf-b782-7b58-855e-8158dcb283ab.jsonl';
  const { buildSessionContext, convertToLlm, createCodingTools, parseSessionEntries } =
    await import('@earendil-works/pi-coding-agent');
  const { buildSystemPrompt } = await import(
    new URL('./core/system-prompt.js', import.meta.resolve('@earendil-works/pi-coding-agent')).href
  );
  const { contextToChatMessages, toolsToDefinitions } = await import(
    resolve(root, 'packages/agent/src/provider/convert-messages.ts')
  );
  const sourceText = readFileSync(source, 'utf8');
  const all = parseSessionEntries(sourceText);
  const header = all.find((entry: any) => entry.type === 'session');
  if (!header || header.type !== 'session') throw new Error('Missing session header');
  const entries = all.filter((entry) => entry.type !== 'session');
  const codingTools = createCodingTools(header.cwd);
  const tools = codingTools.map(({ name, description, parameters }: any) => ({ name, description, parameters }));
  const definitions = toolsToDefinitions(tools);
  // JSONL records messages but not the original system prompt. Reconstruct
  // the production Pi wrapper; historical user and tool content stays intact.
  const systemPrompt = buildSystemPrompt({
    cwd: header.cwd,
    selectedTools: codingTools.map((tool: any) => tool.name),
    toolSnippets: Object.fromEntries(codingTools.map((tool: any) => [tool.name, tool.description.split('\n')[0]])),
  });
  const { Qwen3Tokenizer } = await core();
  const tok = await Qwen3Tokenizer.fromPretrained(resolve(modelDir, 'tokenizer.json'));
  const selectedIds = ['3de6f891', '48096389', '86bd44d6'];
  const inputs = [];
  for (const [index, id] of selectedIds.entries()) {
    const selected = entries.find((entry: any) => entry.id === id);
    if (!selected || selected.type !== 'message' || selected.message.role !== 'assistant')
      throw new Error('Invalid assistant boundary');
    const context = buildSessionContext(entries, selected.parentId);
    const piMessages = convertToLlm(context.messages);
    const messages = contextToChatMessages({ systemPrompt, tools, messages: piMessages });
    if (messages.length !== piMessages.length + 1) throw new Error('Conversion added or removed messages');
    if (messages.some((m: any) => m.role === 'tool' && m.content === 'No result provided'))
      throw new Error('Orphan tool result');
    const ids = Array.from((await tok.applyChatTemplate(messages, true, definitions, true)) as Uint32Array);
    inputs.push({
      name: cases[index],
      assistantEntryId: id,
      parentEntryId: selected.parentId,
      historicalMessages: piMessages.length,
      toolResults: piMessages.filter((m: any) => m.role === 'toolResult').length,
      recordedThinkingLevel: context.thinkingLevel,
      messages,
      tools: definitions,
      promptTokens: ids.length,
      tokenIds: ids,
      rendered: await tok.decode(new Uint32Array(ids), false),
      sha256: sha(JSON.stringify(ids)),
    });
  }
  save(resolve(dataDir, 'inputs.json'), inputs);
  save(resolve(dataDir, 'fixture.json'), {
    source,
    sourceSha256: sha(sourceText),
    sessionId: header.id,
    recordedAt: header.timestamp,
    cwd: header.cwd,
    originalRequest: 'Deepreview https://github.com/oxc-project/oxc-node/pull/745',
    systemPromptSource:
      'Reconstructed with the installed Pi buildSystemPrompt and createCodingTools; original system prompt is absent from session JSONL.',
    historicalContent:
      'Pi parent-chain and compaction reconstruction, then production contextToChatMessages. No padding, repetition, truncation inside a message, or invented user/tool content. Tools are never executed.',
    cases: inputs.map(
      ({ messages: _messages, tools: _tools, tokenIds: _tokenIds, rendered: _rendered, ...metadata }) => metadata,
    ),
  });
  const command = (file: string, args: string[], cwd = root) => execFileSync(file, args, { cwd, encoding: 'utf8' });
  const patch = command('git', ['diff', '--binary']);
  writeFileSync(resolve(dataDir, 'mlx-runtime.patch'), patch);
  const version = spawnSync(llamaServer, ['--version'], { encoding: 'utf8' });
  if (version.status !== 0) throw new Error('Cannot read llama.cpp version');
  const environment = {
    measuredAt: new Date().toISOString(),
    modelFile,
    modelSha256: command('shasum', ['-a', '256', modelFile]).split(/\s/)[0],
    mlxCommit: command('git', ['rev-parse', 'HEAD']).trim(),
    mlxWorkingDiffSha256: sha(patch),
    mlxAddonSha256: command('shasum', ['-a', '256', resolve(root, 'packages/core/mlx-core.darwin-arm64.node')]).split(
      /\s/,
    )[0],
    mlxLibraryCommit: command('git', ['rev-parse', 'HEAD'], resolve(root, 'crates/mlx-sys/mlx')).trim(),
    llamaCommit: command('git', ['rev-parse', 'HEAD'], llamaDir).trim(),
    llamaVersion: (version.stdout + version.stderr).trim(),
    piVersion: JSON.parse(
      readFileSync(resolve(root, 'node_modules/@earendil-works/pi-coding-agent/package.json'), 'utf8'),
    ).version,
    hardware: command('system_profiler', ['SPHardwareDataType', 'SPDisplaysDataType'])
      .split('\n')
      .filter((line) => !/Serial Number|Hardware UUID|Provisioning UDID/.test(line))
      .join('\n'),
    os: command('sw_vers', []).trim(),
    thermal: command('pmset', ['-g', 'therm']).trim(),
    cases: inputs.map((x) => ({ name: x.name, promptTokens: x.promptTokens })),
    outputTokens,
    measuredRepeats: 3,
    cooldownSeconds,
    llamaContextCapacity: 73728,
    reasoningEffort: 'high',
    envOverrides: Object.fromEntries(Object.entries(env).filter(([key]) => key.startsWith('MLX_'))),
    mlxCachePolicy:
      'reuseCache=true is required by the session API; resetCaches() before measurement; assert cachedTokens == 0.',
    kvCache: 'BF16 in both runtimes; MLX paged cache, llama.cpp standard KV cache.',
    prefillBatching: 'MLX chunk 512; llama.cpp logical batch 2048, physical ubatch 512.',
    loadTiming:
      'Excluded. Prepared MLX weight cache and source GGUF are already available. MLX also loads the media companion; llama.cpp is text-only.',
    workstation: 'Interactive desktop session; serial inference. Normal desktop applications remain active.',
    scope:
      'Recorded agent-context replay, high thinking, offline tool definitions and historical results, greedy AR without speculation; cold prompt cache.',
    warmup:
      '32 generated tokens using the first real diff-review fixture before every sample, followed by a full cache reset.',
    inputPolicy:
      'Whole recorded turn boundaries. No synthetic input. Actual generated token count is retained if the model stops before the cap.',
  };
  save(resolve(dataDir, 'environment.json'), environment);
  console.log(
    JSON.stringify(
      inputs.map((x) => ({
        name: x.name,
        tokens: x.promptTokens,
        messages: x.historicalMessages,
        tools: x.toolResults,
      })),
    ),
  );
}

async function mlx(name: string, run: string) {
  const data = input(name);
  const n = data.promptTokens;
  const { Gemma4Model } = await core();
  const started = performance.now();
  const model = await Gemma4Model.load(modelFile);
  const loadMs = performance.now() - started;
  const config = {
    temperature: 0,
    topK: 1,
    topP: 1,
    minP: 0,
    repetitionPenalty: 1,
    presencePenalty: 0,
    frequencyPenalty: 0,
    reasoningEffort: 'high',
    tools: data.tools,
    enableMtp: false,
    reuseCache: true,
    reportPerformance: true,
  };
  await model.chatSessionStart(input(cases[0]).messages, { ...config, maxNewTokens: 32 });
  await model.resetCaches();
  const start = performance.now();
  const result = await model.chatSessionStart(data.messages, { ...config, maxNewTokens: outputTokens });
  const wallMs = performance.now() - start;
  const p = result.performance;
  if (
    result.promptTokens !== n ||
    result.cachedTokens !== 0 ||
    result.numTokens < 2 ||
    result.numTokens > outputTokens
  ) {
    throw new Error(`Workload mismatch: ${JSON.stringify(result)}`);
  }
  const decodeMs = ((result.numTokens - 1) / p.decodeTokensPerSecond) * 1000;
  const sample = {
    runtime: 'mlx',
    name,
    n,
    run,
    loadMs,
    wallMs,
    prefillMs: p.ttftMs,
    prefillTps: p.prefillTokensPerSecond,
    decodeMs,
    decodeTps: p.decodeTokensPerSecond,
    accountedMs: p.ttftMs + decodeMs,
    promptTokens: result.promptTokens,
    generatedTokens: result.numTokens,
    cachedTokens: result.cachedTokens,
    textSha256: sha(result.rawText),
    inputSha256: data.sha256,
    result,
  };
  save(resolve(dataDir, `raw/mlx-${name}-${run}.json`), sample);
  console.log(JSON.stringify({ ...sample, result: undefined }));
  await model.resetCaches();
}

async function llama(name: string, run: string) {
  const data = input(name);
  const n = data.promptTokens;
  const port = 19090;
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
    '73728',
    '-b',
    '2048',
    '-ub',
    '512',
    '-t',
    '6',
    '-np',
    '1',
    '--host',
    '127.0.0.1',
    '--port',
    String(port),
    '--no-webui',
    '--no-context-shift',
  ];
  const fd = openSync(resolve(dataDir, `raw/llama-${name}-${run}.stderr.txt`), 'w');
  const started = performance.now();
  const server = spawn(llamaServer, args, { stdio: ['ignore', fd, fd], env });
  closeSync(fd);
  const exit = new Promise<void>((res) => server.once('exit', () => res()));
  let ready = false;
  try {
    for (let i = 0; i < 240; i++) {
      if (server.exitCode !== null) throw new Error(`llama-server exited: ${server.exitCode}`);
      try {
        if ((await fetch(`http://127.0.0.1:${port}/health`)).ok) {
          ready = true;
          break;
        }
      } catch {}
      await sleep(250);
    }
    if (!ready) throw new Error('llama-server readiness timeout');
    const loadMs = performance.now() - started;
    async function post(path: string, body: unknown) {
      const r = await fetch(`http://127.0.0.1:${port}/${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`${path}: ${await r.text()}`);
      return r.json();
    }
    const tokenized = await post('tokenize', { content: data.rendered, add_special: false, parse_special: true });
    if (JSON.stringify(tokenized.tokens) !== JSON.stringify(data.tokenIds)) throw new Error('Tokenizer mismatch');
    const request = {
      prompt: data.tokenIds,
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
    await post('completion', { ...request, prompt: input(cases[0]).tokenIds, n_predict: 32 });
    const start = performance.now();
    const result = await post('completion', { ...request, n_predict: outputTokens });
    const wallMs = performance.now() - start;
    const t = result.timings;
    if (t.prompt_n !== n || t.predicted_n < 2 || t.predicted_n > outputTokens || t.cache_n !== 0 || result.truncated) {
      throw new Error(`Workload mismatch: ${JSON.stringify(result)}`);
    }
    const sample = {
      runtime: 'llama',
      name,
      n,
      run,
      args,
      loadMs,
      wallMs,
      prefillMs: t.prompt_ms,
      prefillTps: t.prompt_per_second,
      decodeMs: t.predicted_ms,
      decodeTps: t.predicted_per_second,
      accountedMs: t.prompt_ms + t.predicted_ms,
      promptTokens: t.prompt_n,
      generatedTokens: t.predicted_n,
      cachedTokens: t.cache_n,
      textSha256: sha(result.content),
      inputSha256: data.sha256,
      result,
    };
    save(resolve(dataDir, `raw/llama-${name}-${run}.json`), sample);
    console.log(JSON.stringify({ ...sample, args: undefined, result: undefined }));
  } finally {
    server.kill('SIGTERM');
    await Promise.race([exit, sleep(5000)]);
    if (server.exitCode === null) {
      server.kill('SIGKILL');
      await exit;
    }
  }
}

async function runAll() {
  for (let run = 1; run <= 3; run++) {
    for (const name of cases) {
      const order = run % 2 ? ['mlx', 'llama'] : ['llama', 'mlx'];
      for (const runtime of order) {
        console.log(new Date().toISOString(), 'START', runtime, name, run);
        const child = spawn('oxnode', [fileURLToPath(import.meta.url), runtime, name, String(run)], {
          env,
          stdio: 'inherit',
        });
        const code = await new Promise<number | null>((res) => child.once('exit', res));
        if (code !== 0) throw new Error(`${runtime} ${name} run ${run}: exit ${code}`);
        await sleep(cooldownSeconds * 1000);
      }
    }
  }
}

const [mode, n, run] = process.argv.slice(2);
if (mode === 'prepare') await prepare();
else if (mode === 'mlx') await mlx(n!, run!);
else if (mode === 'llama') await llama(n!, run!);
else if (mode === 'run') await runAll();
else throw new Error('Usage: oxnode benchmark.ts prepare | run | mlx|llama <case-name> <run-id>');
