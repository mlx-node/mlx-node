import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { openSync, closeSync, readFileSync, writeFileSync, mkdtempSync, symlinkSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve, join } from 'node:path';

// Run under the artifact directory's resource guard, one fresh server per case.
const [splashPath, outputDirectory, label, caseName = 'short'] = process.argv.slice(2);
if (!splashPath || !outputDirectory || !/^[a-z0-9-]+$/.test(label ?? '')) {
  throw new Error('Usage: node splash-local.mjs <splash-checkout> <artifact-dir> <label> [short|6k|16k|32k]');
}
const root = resolve(splashPath);
const output = resolve(outputDirectory);
const fixtures = JSON.parse(readFileSync('.cache/benchmarks/fixtures-public-2026-09-11/qwen38-inputs.json', 'utf8'));
const item =
  caseName === 'short'
    ? {
        name: 'short',
        promptTokens: 87,
        messages: [
          {
            role: 'user',
            content:
              'Implement a TypeScript LRU cache with a fixed capacity. Include get, set, delete, and tests for eviction and updating existing keys. Explain the design and complexity in detail.',
          },
        ],
      }
    : fixtures.find((entry) => entry.name === caseName);
if (!item) throw new Error(`Unknown case ${caseName}`);
const hash = (value) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const port = 18938;
const base = `http://127.0.0.1:${port}`;
const model = 'incoai/Qwen3.8-27B-Splash';
const packageRoot = join(root, 'install/models', model);
const messages = item.messages.map(({ role, content, reasoningContent, toolCalls, toolCallId, isError }) => {
  if (isError) throw new Error('Error tool responses need an explicit API translation');
  return {
    role,
    content,
    ...(reasoningContent !== undefined ? { reasoning_content: reasoningContent } : {}),
    ...(toolCalls
      ? {
          tool_calls: toolCalls.map(({ id, name, arguments: args }) => ({
            id,
            type: 'function',
            function: { name, arguments: args },
          })),
        }
      : {}),
    ...(toolCallId ? { tool_call_id: toolCallId } : {}),
  };
});
const config = {
  model,
  messages,
  max_completion_tokens: 1024,
  temperature: 0,
  top_k: 20,
  top_p: 1,
  min_p: 0,
  presence_penalty: 0,
  frequency_penalty: 0,
  seed: 0,
  reasoning_effort: 'high',
  stream: false,
};
const command = join(root, '.venv/bin/python');
const args = [
  '-u',
  join(root, 'server/server.py'),
  join(packageRoot, 'target'),
  join(packageRoot, 'draft'),
  '--tokenizer',
  join(packageRoot, 'tokenizer'),
  '--model',
  model,
  '--binary',
  join(root, 'build/splash'),
  '--host',
  '127.0.0.1',
  '--port',
  String(port),
  '--max-memory',
  String(40 * 1024 ** 3),
  '--max-context',
  '40960',
  '--no-webui',
];
const fd = openSync(join(output, `${label}.server.log`), 'w');
const server = spawn(command, args, {
  cwd: root,
  stdio: ['ignore', fd, fd],
  env: { ...process.env, TRANSFORMERS_VERBOSITY: 'error' },
});
closeSync(fd);
let ended = false;
let spawnError;
const finished = new Promise((done) => {
  server.once('error', (error) => {
    spawnError = error;
    ended = true;
    done();
  });
  server.once('exit', () => {
    ended = true;
    done();
  });
});
const delay = (ms) => new Promise((done) => setTimeout(done, ms));
async function request(path, body) {
  const response = await fetch(base + path, {
    method: body ? 'POST' : 'GET',
    headers: body ? { 'content-type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(300000),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(`${path} ${response.status}: ${JSON.stringify(result)}`);
  return result;
}
try {
  const start = performance.now();
  let ready = false;
  while (performance.now() - start < 180000) {
    if (ended) throw spawnError ?? new Error('Splash exited before readiness');
    try {
      ready = (await request('/ready')).status === 'ready';
    } catch {}
    if (ready) break;
    await delay(500);
  }
  if (!ready) throw new Error('Splash readiness timed out');
  const status = await request('/status');
  if (status.instance.pid !== server.pid || status.instance.model !== model)
    throw new Error('Readiness belongs to a different server');
  const loadMs = performance.now() - start;
  // A different initial role/content avoids populating the measured prefix.
  const warmup = await request('/v1/chat/completions', {
    ...config,
    messages: [
      { role: 'system', content: 'Warmup: answer briefly.' },
      { role: 'user', content: 'Count integers from one to twenty.' },
    ],
    reasoning_effort: 'none',
    max_completion_tokens: 16,
  });
  if (warmup.usage.completion_tokens !== 16 || warmup.usage.prompt_tokens_details.cached_tokens !== 0)
    throw new Error('Unexpected warmup result');
  const high = await request('/apply-template', config);
  const xhigh = await request('/apply-template', { ...config, reasoning_effort: 'xhigh' });
  if (high.prompt !== xhigh.prompt) throw new Error('High and xhigh prompts differ');
  const tokenized = await request('/tokenize', { content: high.prompt, add_special: false });
  const nativeTokenizerBinding = resolve('packages/core/mlx-core.darwin-arm64.node');
  const nativeTokenizerSha256 = createHash('sha256').update(readFileSync(nativeTokenizerBinding)).digest('hex');
  const core = createRequire(import.meta.url)(nativeTokenizerBinding);
  // The standalone tokenizer defaults to preserve_thinking=false. Stateful
  // chatSessionStart uses true; emulate that render flag in an artifact-local
  // template copy, leaving the checkpoint untouched.
  const auditDirectory = mkdtempSync(join(output, `${label}.tokenizer-`));
  const tokenizerConfig = JSON.parse(readFileSync('.cache/models/qwen3.8-27b-gguf/tokenizer_config.json', 'utf8'));
  const template =
    tokenizerConfig.chat_template ?? readFileSync('.cache/models/qwen3.8-27b-gguf/chat_template.jinja', 'utf8');
  tokenizerConfig.chat_template = '{%- set preserve_thinking = true %}' + template;
  writeFileSync(join(auditDirectory, 'tokenizer_config.json'), JSON.stringify(tokenizerConfig));
  symlinkSync(resolve('.cache/models/qwen3.8-27b-gguf/tokenizer.json'), join(auditDirectory, 'tokenizer.json'));
  const tokenizer = await core.Qwen3Tokenizer.fromPretrained(join(auditDirectory, 'tokenizer.json'));
  const nativeIds = [
    ...(await tokenizer.applyChatTemplate(item.messages, true, undefined, true, undefined, undefined, 'high')),
  ];
  const equal = JSON.stringify(nativeIds) === JSON.stringify(tokenized.tokens);
  writeFileSync(
    join(output, `${label}.prompt-audit.json`),
    JSON.stringify(
      {
        equal,
        nativeTokenizerBinding,
        nativeTokenizerSha256,
        nativeIds,
        splashIds: tokenized.tokens,
        nativePrompt: await tokenizer.decode(Uint32Array.from(nativeIds), false),
        splashPrompt: high.prompt,
      },
      null,
      2,
    ),
  );
  if (!equal) throw new Error('Rendered prompt tokens differ between engines');
  if (tokenized.tokens.length !== item.promptTokens)
    throw new Error(`Prompt count differs: ${tokenized.tokens.length} vs ${item.promptTokens}`);
  const before = await request('/status');
  const measuredStart = performance.now();
  const response = await request('/v1/chat/completions', config);
  const wallMs = performance.now() - measuredStart;
  const after = await request('/status');
  if (
    after.metrics.metal_failures !== before.metrics.metal_failures ||
    after.metrics.capacity_failures !== before.metrics.capacity_failures
  )
    throw new Error('Splash runtime failure counter changed');
  const record = {
    label,
    caseName,
    command,
    args,
    nativeTokenizerBinding,
    nativeTokenizerSha256,
    config,
    loadMs,
    warmup,
    before,
    after,
    wallMs,
    inputHash: hash(item.messages),
    prompt: high.prompt,
    promptTokenIds: tokenized.tokens,
    promptTokenHash: hash(tokenized.tokens),
    response,
    outputHash: hash(response.choices?.[0]?.message),
  };
  const { usage, metrics } = response;
  if (usage.prompt_tokens !== tokenized.tokens.length || usage.completion_tokens !== 1024)
    throw new Error('Unexpected token counts');
  if (response.choices[0].finish_reason !== 'length') throw new Error('Unexpected finish reason');
  if (
    usage.prompt_tokens_details.cached_tokens !== 0 ||
    metrics.cache.matched_tokens !== 0 ||
    metrics.prefill.tokens !== usage.prompt_tokens
  )
    throw new Error('Measured request reused cached tokens');
  writeFileSync(join(output, `${label}.json`), JSON.stringify(record, null, 2) + '\n');
  console.log(JSON.stringify({ label, loadMs, wallMs, usage, metrics, outputHash: record.outputHash }));
} finally {
  if (!ended) server.kill('SIGTERM');
  let timer;
  await Promise.race([
    finished,
    new Promise((done) => {
      timer = setTimeout(done, 10000);
    }),
  ]);
  clearTimeout(timer);
  if (!ended) {
    server.kill('SIGKILL');
    await finished;
  }
}
