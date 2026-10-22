/** Run alone: warming/measurement must not overlap builds or other GPU jobs. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, writeFile, rename, readdir } from 'node:fs/promises';
import { cpus, totalmem, platform, release } from 'node:os';
import { join, resolve, dirname } from 'node:path';

import { XlmRobertaRerankerModel, getMemorySnapshot, resetPeakMemory } from '../../../packages/core/index.cjs';
const [modelPath, output, selection = 'all'] = process.argv.slice(2);
if (!output) throw Error('Usage: node benchmark.mjs MODEL REPORT [all|512]');
assert.ok(['all', '512'].includes(selection), 'selection must be all or 512');
assert.ok(!Object.hasOwn(process.env, 'MLX_DISABLE_COMPILE'), 'Performance acceptance requires MLX compilation');
const manifest = JSON.parse(
  await readFile(new URL('../../../fixtures/rerank/bge/manifest.json', import.meta.url), 'utf8'),
);
const memory = () =>
  Object.fromEntries(Object.entries(getMemorySnapshot()).map(([k, v]) => [k, typeof v === 'bigint' ? Number(v) : v]));
const stats = (values) => {
  const sorted = values.toSorted((a, b) => a - b);
  return { medianMs: (sorted[4] + sorted[5]) / 2, p95Ms: sorted[Math.ceil(sorted.length * 0.95) - 1] };
};
async function rerank(model, request) {
  const task = model.startRerank(request);
  try {
    return JSON.parse(await task.result());
  } finally {
    task.cancel();
  }
}
const config = JSON.parse(await readFile(join(modelPath, 'config.json'), 'utf8'));
const hardware = { cpu: cpus()[0]?.model, totalMemory: totalmem(), platform: platform(), release: release() };
const hashFile = async (path) => {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
};
const addonPath = process.env.NAPI_RS_NATIVE_LIBRARY_PATH ?? resolve('packages/core/mlx-core.darwin-arm64.node');
const checksums = {};
for (const name of (await readdir(modelPath)).sort()) {
  if (name.endsWith('.json') || name.endsWith('.safetensors')) checksums[name] = await hashFile(join(modelPath, name));
}
const identity = {
  schemaVersion: 1,
  modelPath: resolve(modelPath),
  selection,
  hardware,
  node: process.version,
  execArgv: process.execArgv,
  // Hash execution knobs without copying environment values into reports.
  executionEnvironmentSha256: createHash('sha256')
    .update(
      JSON.stringify(
        Object.entries(process.env)
          .filter(([key]) => /^(MLX_|MTL_|METAL_|DYLD_|OMP_|VECLIB_|OPENBLAS_|ACCELERATE_|NODE_OPTIONS$)/.test(key))
          .sort(([a], [b]) => a.localeCompare(b)),
      ),
    )
    .digest('hex'),
  checksums,
  scriptSha256: await hashFile(new URL(import.meta.url)),
  manifestSha256: createHash('sha256').update(JSON.stringify(manifest)).digest('hex'),
  addonSha256: await hashFile(addonPath),
  metalSha256: await hashFile(join(dirname(addonPath), 'mlx.metallib')),
};
const scenarios = (selection === '512' ? [512] : [64, 512, 2048]).flatMap((pairTokens) =>
  [50, 125, 200].map((documents) => ({ pairTokens, documents })),
);
if (selection === 'all') scenarios.push({ pairTokens: 8192, documents: 1 });
const before = memory();
let report;
try {
  report = JSON.parse(await readFile(output, 'utf8'));
  assert.deepEqual(report.identity, identity, 'Benchmark identity differs; use a new output path');
  const seen = new Set();
  for (const row of report.cases) {
    const key = `${row.maxBatchSize}:${row.pairTokens}:${row.documents}`;
    assert.ok(!seen.has(key), 'Duplicate benchmark case');
    seen.add(key);
    assert.ok([1, 8].includes(row.maxBatchSize));
    assert.ok(scenarios.some((s) => s.pairTokens === row.pairTokens && s.documents === row.documents));
    assert.equal(row.times.length, manifest.performance.iterations);
    assert.ok(row.times.every((v) => Number.isFinite(v) && v > 0));
    assert.deepEqual({ medianMs: row.medianMs, p95Ms: row.p95Ms }, stats(row.times));
  }
  assert.ok(report.completedBatchSizes.every((v) => [1, 8].includes(v)));
  assert.equal(new Set(report.completedBatchSizes).size, report.completedBatchSizes.length);
  for (const batch of report.completedBatchSizes) {
    assert.equal(report.cases.filter((r) => r.maxBatchSize === batch).length, scenarios.length);
    assert.equal(report.cancellation.filter((r) => r.maxBatchSize === batch).length, 8);
    assert.equal(report.activeCycles.filter((r) => r.maxBatchSize === batch).length, 8);
  }
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
  report = {
    identity,
    recordedAt: new Date().toISOString(),
    hardware,
    modelPath,
    computeDtype: config.reranker?.computeDtype ?? config.torch_dtype,
    before,
    cases: [],
    cancellation: [],
    activeCycles: [],
    loads: [],
    completedBatchSizes: [],
    complete: false,
  };
}
const save = async () => {
  const temporary = `${output}.${process.pid}.tmp`;
  await writeFile(temporary, JSON.stringify(report, null, 2) + '\n');
  await rename(temporary, output);
};
function batchingPrecheck() {
  const gains = [125, 200].map((documents) => {
    const rows = [1, 8].map((batch) =>
      report.cases.find((r) => r.maxBatchSize === batch && r.pairTokens === 512 && r.documents === documents),
    );
    return rows.every(Boolean) ? { documents, throughputGain: rows[0].medianMs / rows[1].medianMs - 1 } : undefined;
  });
  if (!gains.every(Boolean)) return;
  const applicable = report.computeDtype === 'float16' && !report.modelInfo.quantized;
  report.batchingPrecheck = {
    applicable,
    gains,
    passed: applicable ? gains.every((r) => r.throughputGain >= manifest.thresholds.batchingMinThroughputGain) : null,
  };
}
batchingPrecheck();
if (report.batchingPrecheck?.passed === false)
  throw Error('Batching target failed; use a new artifact/report for optimization');
// Establish the required batching gain before the longer matrix; every case
// still receives its own three warmups and ten samples after each model load.
for (const [maxBatchSize, phase] of [
  [1, 'acceptance'],
  [8, 'acceptance'],
  [1, 'remaining'],
  [8, 'remaining'],
]) {
  if (report.completedBatchSizes.includes(maxBatchSize)) continue;
  const phaseScenarios = scenarios.filter((s) =>
    phase === 'acceptance' ? s.pairTokens === 512 : s.pairTokens !== 512,
  );
  if (
    phase === 'acceptance' &&
    phaseScenarios.every((s) =>
      report.cases.some(
        (r) => r.maxBatchSize === maxBatchSize && r.pairTokens === s.pairTokens && r.documents === s.documents,
      ),
    )
  )
    continue;
  // Completed measurements survive interruption. Cancellation cycles restart as one set.
  report.cancellation = report.cancellation.filter((r) => r.maxBatchSize !== maxBatchSize);
  report.activeCycles = report.activeCycles.filter((r) => r.maxBatchSize !== maxBatchSize);
  const loadAt = performance.now();
  const model = await XlmRobertaRerankerModel.load(modelPath, JSON.stringify({ maxBatchSize }));
  const loadedMs = performance.now() - loadAt;
  report.modelInfo ??= JSON.parse(model.infoJson);
  const load = { maxBatchSize, loadedMs, startedAt: new Date().toISOString(), before };
  report.loads.push(load);
  try {
    for (const scenario of phaseScenarios) {
      if (
        report.cases.some(
          (r) =>
            r.maxBatchSize === maxBatchSize &&
            r.pairTokens === scenario.pairTokens &&
            r.documents === scenario.documents,
        )
      )
        continue;
      const measuredAt = new Date().toISOString();
      // Frozen official tokenizer: "query" = 2 tokens, "cat" = 1, pair overhead = 4.
      const request = JSON.stringify({
        query: 'query',
        documents: Array(scenario.documents).fill('cat '.repeat(scenario.pairTokens - 6)),
      });
      const times = [];
      let response;
      resetPeakMemory();
      for (let i = 0; i < manifest.performance.warmup + manifest.performance.iterations; i++) {
        const at = performance.now();
        response = await rerank(model, request);
        const elapsed = performance.now() - at;
        load.firstRequestMs ??= elapsed;
        load.coldModelTotalMs ??= loadedMs + elapsed;
        if (i >= manifest.performance.warmup) times.push(elapsed);
        assert.equal(response.usage.promptTokens, scenario.pairTokens * scenario.documents);
      }
      const row = {
        ...scenario,
        maxBatchSize,
        measuredAt,
        loadedMs,
        ...stats(times),
        times,
        memory: memory(),
        meta: response.meta,
      };
      row.documentsPerSecond = scenario.documents / (row.medianMs / 1000);
      report.cases.push(row);
      await save();
      console.log(JSON.stringify({ ...row, times: undefined }));
    }
    for (let i = 0; i < 8; i++) {
      const task = model.startRerank(
        JSON.stringify({ query: 'query', documents: Array(200).fill('cat '.repeat(2042)) }),
      );
      const result = task.result().then(
        () => false,
        (error) => {
          assert.match(error.message, /cancelled/);
          return true;
        },
      );
      await new Promise((r) => setTimeout(r, 100));
      const at = performance.now();
      task.cancel();
      assert.equal(await result, true);
      report.cancellation.push({ maxBatchSize, ms: performance.now() - at, activeBytes: memory().activeBytes });
      await rerank(model, JSON.stringify({ query: 'query', documents: ['cat', 'cat cat'] }));
      report.activeCycles.push({ maxBatchSize, activeBytes: memory().activeBytes });
    }
  } finally {
    await model.dispose();
  }
  load.afterDispose = memory();
  assert.equal(load.afterDispose.activeBytes, before.activeBytes, 'model-owned resources must be released');
  if (report.cases.filter((r) => r.maxBatchSize === maxBatchSize).length === scenarios.length)
    report.completedBatchSizes.push(maxBatchSize);
  batchingPrecheck();
  await save();
  if (report.batchingPrecheck?.passed === false)
    throw Error('Batching target failed before the remaining performance matrix');
}
report.afterDispose = memory();
report.batching = report.cases
  .filter((r) => r.maxBatchSize === 8)
  .map((r) => {
    const baseline = report.cases.find(
      (b) => b.maxBatchSize === 1 && b.documents === r.documents && b.pairTokens === r.pairTokens,
    );
    return { documents: r.documents, pairTokens: r.pairTokens, throughputGain: baseline.medianMs / r.medianMs - 1 };
  });
for (const batch of [1, 8]) {
  const active = report.activeCycles.filter((r) => r.maxBatchSize === batch).map((r) => r.activeBytes);
  assert.ok(Math.max(...active) - Math.min(...active) < 1024 * 1024, 'active resources keep growing');
}
const acceptance = report.batching.filter((r) => r.pairTokens === 512 && r.documents >= 125);
assert.deepEqual(
  acceptance.map((r) => r.documents).sort((a, b) => a - b),
  [125, 200],
);
report.targetApplicable = report.computeDtype === 'float16' && !report.modelInfo.quantized;
report.targetPassed = report.targetApplicable
  ? acceptance.every((r) => r.throughputGain >= manifest.thresholds.batchingMinThroughputGain)
  : null;
report.complete = true;
await save();
console.log(JSON.stringify({ batching: report.batching, targetPassed: report.targetPassed }));

if (report.targetPassed === false) process.exitCode = 1;
