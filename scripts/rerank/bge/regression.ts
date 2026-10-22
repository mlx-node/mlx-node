/** Frozen chat/Jina control and chat with idle/concurrent BGE, through the real host. */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdtemp, readFile, readdir, realpath, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { cpus, totalmem, platform, release } from 'node:os';
import { tmpdir } from 'node:os';
import { dirname, join, resolve, relative } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repositoryRoot = fileURLToPath(new URL('../../../', import.meta.url));
const sourceRoot = resolve(process.env.MLX_BGE_SOURCE_ROOT ?? repositoryRoot);
const sourceRevision =
  sourceRoot === resolve(repositoryRoot)
    ? execFileSync('git', ['rev-parse', 'HEAD'], { cwd: sourceRoot, encoding: 'utf8' }).trim()
    : (await readFile(join(sourceRoot, 'source-revision.txt'), 'utf8')).trim();
assert.match(sourceRevision, /^[a-f0-9]{40}$/);
const workspacePackages = ['core', 'asr', 'tts', 'lm', 'rerank', 'server'];
const modulePaths: Record<string, string> = JSON.parse(
  execFileSync(
    process.execPath,
    [
      '--input-type=module',
      '-e',
      `console.log(JSON.stringify(Object.fromEntries(${JSON.stringify(workspacePackages)}.map(name => [name, import.meta.resolve('@mlx-node/' + name)]))))`,
    ],
    { cwd: sourceRoot, encoding: 'utf8' },
  ),
);
for (const [name, url] of Object.entries(modulePaths)) {
  const packageRoot = await realpath(join(sourceRoot, 'packages', name));
  assert.ok(
    (await realpath(fileURLToPath(url))).startsWith(packageRoot + '/'),
    `Workspace dependency ${name} escaped source root`,
  );
}
// Select MLX only after the TypeScript loader's own N-API addon initialized.
// Exporting the generic N-API override to oxnode would load MLX as its compiler.
const addon = resolve(process.env.MLX_BGE_ADDON_PATH ?? join(sourceRoot, 'packages/core/mlx-core.darwin-arm64.node'));
process.env.NAPI_RS_NATIVE_LIBRARY_PATH = addon;
const { getMemorySnapshot, resetPeakMemory } = (await import(
  pathToFileURL(join(sourceRoot, 'packages/core/index.cjs')).href
)) as typeof import('@mlx-node/core');
const { createInferenceHost } = (await import(
  pathToFileURL(join(sourceRoot, 'packages/server/dist/host/index.js')).href
)) as typeof import('../../../packages/server/src/host/index.js');
const hashFile = async (path: string | URL) => {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
};
async function treeHash(root: string, include: (path: string) => boolean): Promise<string> {
  const files: Array<[string, string]> = [];
  async function visit(folder: string, ancestors = new Set<string>()) {
    const canonical = await realpath(folder);
    assert.ok(!ancestors.has(canonical), `Symlink cycle in fingerprint: ${folder}`);
    const next = new Set([...ancestors, canonical]);
    for (const entry of (await readdir(folder, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      if (['node_modules', '.git', '.cache'].includes(entry.name)) continue;
      const path = join(folder, entry.name);
      const info = entry.isSymbolicLink() ? await stat(path) : entry;
      if (info.isDirectory()) await visit(path, next);
      else if (info.isFile() && include(relative(root, path))) files.push([relative(root, path), await hashFile(path)]);
    }
  }
  await visit(root);
  return createHash('sha256').update(JSON.stringify(files)).digest('hex');
}

const [chatPath, jinaPath, output, bgePath] = process.argv.slice(2);
if (!output) throw Error('Usage: oxnode regression.ts CHAT JINA REPORT [BGE]');
const manifest = JSON.parse(
  await readFile(new URL('../../../fixtures/rerank/bge/manifest.json', import.meta.url), 'utf8'),
);
const identity = {
  node: process.version,
  execArgv: process.execArgv,
  executionEnvironmentSha256: createHash('sha256')
    .update(
      JSON.stringify(
        Object.entries(process.env)
          .filter(
            ([key]) =>
              /^(MLX_|MTL_|METAL_|DYLD_|OMP_|VECLIB_|OPENBLAS_|ACCELERATE_|NODE_OPTIONS$)/.test(key) &&
              !['MLX_BGE_SOURCE_ROOT', 'MLX_BGE_ADDON_PATH'].includes(key),
          )
          .sort(([a], [b]) => a.localeCompare(b)),
      ),
    )
    .digest('hex'),
  sourceRoot,
  sourceRevision,
  modulePaths,
  sourceSha256: await treeHash(
    join(sourceRoot, 'packages'),
    (path) =>
      workspacePackages.includes(path.split('/')[0]) &&
      /\.(?:ts|js|cjs|json)$/.test(path) &&
      !path.includes('/__test__/'),
  ),
  lockSha256: await hashFile(join(sourceRoot, 'yarn.lock')),
  scriptSha256: await hashFile(new URL(import.meta.url)),
  manifestSha256: createHash('sha256').update(JSON.stringify(manifest)).digest('hex'),
  chatSha256: await treeHash(chatPath, (path) => /\.(?:json|safetensors|gguf|model|txt)$/.test(path)),
  jinaSha256: await treeHash(jinaPath, (path) => /\.(?:json|safetensors|gguf|model|txt)$/.test(path)),
  metalSha256: await hashFile(join(dirname(addon), 'mlx.metallib')),
};
const root = await mkdtemp(join(tmpdir(), 'mlx-bge-regression-'));
for (const [name, path] of [['chat', chatPath], ['jina', jinaPath], ...(bgePath ? [['bge', bgePath]] : [])]) {
  await symlink(resolve(path), join(root, name));
}
const memory = () =>
  Object.fromEntries(Object.entries(getMemorySnapshot()).map(([k, v]) => [k, typeof v === 'bigint' ? Number(v) : v]));
const host = await createInferenceHost({
  modelsDir: root,
  model: 'chat',
  port: 0,
  disableStore: true,
  sweepOrphanTempRoots: false,
  logDir: join(root, 'logs'),
});
const report: Record<string, unknown> = {
  recordedAt: new Date().toISOString(),
  identity,
  addon,
  addonSha256: await hashFile(addon),
  hardware: { cpu: cpus()[0]?.model, totalMemory: totalmem(), platform: platform(), release: release() },
  workload: 'unified-memory-chat8-rerank50-v1',
  chatPath: resolve(chatPath),
  jinaPath: resolve(jinaPath),
  bgePath,
  rounds: [],
};
async function post(path: string, body: unknown) {
  const at = performance.now();
  const response = await fetch(host.url + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const value = await response.json();
  assert.equal(response.status, 200, JSON.stringify(value));
  return { ms: performance.now() - at, value };
}
const chat = async () => {
  const result = await post('/v1/messages', {
    model: 'chat',
    messages: [{ role: 'user', content: 'Write one sentence explaining unified memory. /no_think' }],
    temperature: 0,
    max_tokens: 8,
  });
  const outputTokens = (result.value as { usage: { output_tokens: number } }).usage.output_tokens;
  assert.ok(Number.isInteger(outputTokens) && outputTokens > 0);
  return { ...result, outputTokens };
};
const rank = (model: string) =>
  post('/v1/rerank', {
    model,
    query: 'What is unified memory?',
    documents: Array.from(
      { length: 50 },
      (_, i) => `Passage ${i}. ` + 'Apple Silicon uses unified memory for local machine learning inference. '.repeat(4),
    ),
  });
async function sample(run: () => Promise<{ ms: number; value: unknown; outputTokens?: number }>) {
  const values: number[] = [];
  const outputTokens: number[] = [];
  resetPeakMemory();
  for (let i = 0; i < manifest.performance.warmup + manifest.performance.iterations; i++) {
    const result = await run();
    if (i >= manifest.performance.warmup) {
      values.push(result.ms);
      if (result.outputTokens !== undefined) outputTokens.push(result.outputTokens);
    }
  }
  const sorted = [...values].sort((a, b) => a - b);
  return { values, outputTokens, medianMs: (sorted[4] + sorted[5]) / 2, p95Ms: sorted[9], memory: memory() };
}
try {
  const rounds = [];
  for (let i = 0; i < 3; i++) {
    const row = { chat: await sample(chat), jina: await sample(() => rank('jina')) };
    rounds.push(row);
    report.rounds = rounds;
    await writeFile(output, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ round: i, ...row }));
  }
  if (bgePath) {
    const at = performance.now();
    await host.loadModel('bge');
    report.bgeLoadMs = performance.now() - at;
    assert.ok(host.server.registry.getAny('chat'));
    assert.ok(host.server.registry.rerank.get('bge'));
    assert.equal(host.server.registry.rerank.get('jina'), undefined);
    report.chatWithIdleBge = await sample(chat);
    report.bgeAlone = await sample(() => rank('bge'));
    const contention: number[] = [];
    report.chatWithConcurrentBge = await sample(async () => {
      const [c, b] = await Promise.all([chat(), rank('bge')]);
      contention.push(b.ms);
      return c;
    });
    report.concurrentBgeMs = contention.slice(manifest.performance.warmup);
  }
} finally {
  await host.close();
  report.afterClose = memory();
  await rm(root, { recursive: true, force: true });
}
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
