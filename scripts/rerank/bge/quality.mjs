import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, writeFile, appendFile, readdir } from 'node:fs/promises';
import { join, resolve, dirname } from 'node:path';

import { XlmRobertaRerankerModel } from '../../../packages/core/index.cjs';
import {
  rankingMetrics,
  validateQualityInput,
  validateQualityDatasets,
  validateQualityBaseline,
  validateQualityFormat,
  validateQualityPolicy,
} from './quality-contract.mjs';
const [modelPath, inputPath, outputPath, baselinePath] = process.argv.slice(2);
if (!outputPath) throw Error('Usage: node quality.mjs MODEL INPUT OUTPUT [BASELINE]');
const manifest = JSON.parse(
  await readFile(new URL('../../../fixtures/rerank/bge/manifest.json', import.meta.url), 'utf8'),
);
manifest.model.fp32Checksums = JSON.parse(
  await readFile(new URL('../../../fixtures/rerank/bge/fp32-checksums.json', import.meta.url), 'utf8'),
);
const hashFile = async (path) => {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
};
const input = JSON.parse(await readFile(inputPath, 'utf8'));
validateQualityInput(input, manifest);
assert.ok(!Object.hasOwn(process.env, 'MLX_DISABLE_COMPILE'), 'Quality acceptance requires MLX compilation');
const baseline = baselinePath ? JSON.parse(await readFile(baselinePath, 'utf8')) : undefined;
const config = JSON.parse(await readFile(join(modelPath, 'config.json'), 'utf8'));
const checksums = {};
for (const name of (await readdir(modelPath)).sort()) {
  if (name.endsWith('.json') || name.endsWith('.safetensors')) checksums[name] = await hashFile(join(modelPath, name));
}
const addonPath = process.env.NAPI_RS_NATIVE_LIBRARY_PATH ?? resolve('packages/core/mlx-core.darwin-arm64.node');
const identity = {
  schemaVersion: 3,
  node: process.version,
  execArgv: process.execArgv,
  scriptSha256: await hashFile(new URL(import.meta.url)),
  contractSha256: await hashFile(new URL('./quality-contract.mjs', import.meta.url)),
  manifestSha256: createHash('sha256').update(JSON.stringify(manifest)).digest('hex'),
  model: resolve(modelPath),
  checksums,
  inputSha256: await hashFile(inputPath),
  addonPath: resolve(addonPath),
  addonSha256: await hashFile(addonPath),
  metalSha256: await hashFile(join(dirname(addonPath), 'mlx.metallib')),
  executionEnvironmentSha256: createHash('sha256')
    .update(
      JSON.stringify(
        Object.entries(process.env)
          .filter(([key]) => /^(MLX_|MTL_|METAL_|DYLD_|OMP_|VECLIB_|OPENBLAS_|ACCELERATE_|NODE_OPTIONS$)/.test(key))
          .sort(([a], [b]) => a.localeCompare(b)),
      ),
    )
    .digest('hex'),
};
if (baseline) {
  validateQualityBaseline(baseline, identity, input, manifest);
  assert.equal(config.reranker?.sourceRevision, manifest.model.revision, 'Quantized checkpoint source differs');
}
const progressPath = outputPath + '.progress.jsonl';
let completed = [];
try {
  const [savedIdentity, ...rows] = (await readFile(progressPath, 'utf8')).trimEnd().split('\n').map(JSON.parse);
  if (JSON.stringify(savedIdentity) !== JSON.stringify(identity))
    throw Error('Progress identity differs; use a new output path');
  completed = rows;
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
  await writeFile(progressPath, JSON.stringify(identity) + '\n');
}
const model = await XlmRobertaRerankerModel.load(modelPath);
async function rerank(request) {
  const task = model.startRerank(JSON.stringify(request));
  try {
    return JSON.parse(await task.result());
  } finally {
    task.cancel();
  } // Retain task ownership until completion, including GC during long runs.
}
const report = {
  ...identity,
  computeDtype: config.reranker?.computeDtype ?? config.torch_dtype,
  baselineSha256: baselinePath ? await hashFile(baselinePath) : null,
  info: JSON.parse(model.infoJson),
  complete: false,
  datasets: [],
};
try {
  validateQualityPolicy(report.info.policy);
  if (baseline) {
    validateQualityFormat(report.info, config);
    assert.deepEqual(report.info.policy, baseline.info.policy, 'Quality policies differ');
  } else {
    assert.equal(report.computeDtype, 'float32', 'A non-FP32 evaluation needs a pinned FP32 baseline');
    assert.equal(report.info.quantized, false);
    assert.deepEqual(checksums, manifest.model.fp32Checksums);
  }
  for (const dataset of input.datasets) {
    const rows = [];
    for (const [queryIndex, query] of dataset.queries.entries()) {
      const saved = completed.find((r) => r.repository === dataset.repository && r.queryIndex === queryIndex);
      if (saved && saved.sha256 !== query.sha256) throw Error('Progress query identity differs');
      if (saved) {
        const { repository: _repository, ...row } = saved;
        rows.push(row);
        continue;
      }
      const started = performance.now();
      const r = await rerank({ query: query.query, documents: query.documents, truncate: true });
      const ranking = r.results.map((v) => v.index);
      const { ndcg10, mrr10 } = rankingMetrics(ranking, query.labels);
      const row = {
        queryIndex,
        sha256: query.sha256,
        ranking,
        ndcg10,
        mrr10,
        milliseconds: performance.now() - started,
        usage: r.usage,
        meta: r.meta,
      };
      rows.push(row);
      await appendFile(progressPath, JSON.stringify({ repository: dataset.repository, ...row }) + '\n');
      if (rows.length % 32 === 0) console.log(JSON.stringify({ dataset: dataset.repository, completed: rows.length }));
    }
    const result = {
      repository: dataset.repository,
      revision: dataset.revision,
      fileSha256: dataset.fileSha256,
      skipped: dataset.skipped,
      queries: rows,
      ndcg10: rows.reduce((n, r) => n + r.ndcg10, 0) / rows.length,
      mrr10: rows.reduce((n, r) => n + r.mrr10, 0) / rows.length,
    };
    if (baseline) {
      const ref = baseline.datasets.find((d) => d.repository === dataset.repository);
      if (
        !ref ||
        ref.queries.length !== rows.length ||
        ref.queries.some((r, i) => r.sha256 !== rows[i].sha256 || r.queryIndex !== i)
      )
        throw Error('Quality subset mismatch');
      result.ndcg10Drop = ref.ndcg10 - result.ndcg10;
      result.mrr10Change = result.mrr10 - ref.mrr10;
      result.changedRankings = rows.filter(
        (r, i) => JSON.stringify(r.ranking) !== JSON.stringify(ref.queries[i].ranking),
      ).length;
      result.passed = result.ndcg10Drop <= manifest.thresholds.quantizationNdcg10MaxDrop;
    }
    report.datasets.push(result);
    await writeFile(outputPath, JSON.stringify(report, null, 2) + '\n');
    console.log(
      JSON.stringify({
        repository: result.repository,
        ndcg10: result.ndcg10,
        mrr10: result.mrr10,
        ndcg10Drop: result.ndcg10Drop,
        passed: result.passed,
      }),
    );
  }
  validateQualityDatasets(report, input, manifest);
  report.complete = true;
  await writeFile(outputPath, JSON.stringify(report, null, 2) + '\n');
} finally {
  await model.dispose();
}
if (report.datasets.some((d) => d.passed === false)) process.exitCode = 1;
