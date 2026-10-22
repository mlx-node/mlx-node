/** Offline evidence verification; imports no native addon and performs no inference. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, readdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

import {
  validateQualityBaseline,
  validateQualityDatasets,
  validateQualityFormat,
  validateQualityPolicy,
} from './quality-contract.mjs';

const [root] = process.argv.slice(2);
if (!root) throw Error('Usage: node quality-summary.mjs ROOT');
const read = async (path) => JSON.parse(await readFile(path, 'utf8'));
const hash = async (path) => {
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest('hex');
};
const manifest = await read(new URL('../../../fixtures/rerank/bge/manifest.json', import.meta.url));
manifest.model.fp32Checksums = await read(new URL('../../../fixtures/rerank/bge/fp32-checksums.json', import.meta.url));
const output = {};
let baseline;
for (const dtype of ['fp32', 'affine', 'mxfp4']) {
  try {
    const path = join(root, `quality-${dtype}.json`);
    const report = await read(path);
    const inputPath = join(root, 'quality-inputs.json');
    const input = await read(inputPath);
    assert.equal(report.schemaVersion, 3);
    assert.equal(report.complete, true);
    assert.equal(report.scriptSha256, await hash(new URL('./quality.mjs', import.meta.url)));
    assert.equal(report.contractSha256, await hash(new URL('./quality-contract.mjs', import.meta.url)));
    assert.equal(report.manifestSha256, createHash('sha256').update(JSON.stringify(manifest)).digest('hex'));
    assert.equal(report.inputSha256, await hash(inputPath));
    assert.equal(report.model, resolve(root, dtype));
    const checksums = {};
    for (const name of (await readdir(report.model)).sort()) {
      if (name.endsWith('.json') || name.endsWith('.safetensors'))
        checksums[name] = await hash(join(report.model, name));
    }
    assert.deepEqual(report.checksums, checksums, 'Checkpoint files differ from the measured artifact');
    assert.equal(report.addonSha256, await hash(report.addonPath));
    assert.equal(report.metalSha256, await hash(join(dirname(report.addonPath), 'mlx.metallib')));
    const config = await read(join(report.model, 'config.json'));
    assert.equal(report.computeDtype, config.reranker?.computeDtype ?? config.torch_dtype);
    validateQualityDatasets(report, input, manifest);
    validateQualityPolicy(report.info.policy);
    if (dtype === 'fp32') {
      validateQualityBaseline(report, report, input, manifest);
      baseline = report;
    } else {
      assert.ok(baseline, 'A verified FP32 baseline is required');
      validateQualityFormat(report.info, config, dtype);
      validateQualityBaseline(baseline, report, input, manifest);
      assert.equal(report.baselineSha256, await hash(join(root, 'quality-fp32.json')));
      assert.equal(config.reranker?.sourceRevision, manifest.model.revision);
      assert.deepEqual(report.info.policy, baseline.info.policy);
      for (const [i, dataset] of report.datasets.entries()) {
        const drop = baseline.datasets[i].ndcg10 - dataset.ndcg10;
        assert.equal(dataset.ndcg10Drop, drop);
        assert.equal(dataset.passed, drop <= manifest.thresholds.quantizationNdcg10MaxDrop);
      }
    }
    output[dtype] = {
      complete: true,
      accepted: dtype === 'fp32' || report.datasets.every((d) => d.passed === true),
      reportSha256: await hash(path),
      inputSha256: report.inputSha256,
      addonSha256: report.addonSha256,
      metalSha256: report.metalSha256,
      executionEnvironmentSha256: report.executionEnvironmentSha256,
      baselineSha256: report.baselineSha256,
      datasets: report.datasets.map(({ queries, ...dataset }) => ({ ...dataset, queryCount: queries.length })),
    };
  } catch (error) {
    output[dtype] = { complete: false, accepted: false, reason: error.message };
  }
}
console.log(JSON.stringify(output));
