import assert from 'node:assert/strict';
import test from 'node:test';

import {
  rankingMetrics,
  validateQualityBaseline,
  validateQualityDatasets,
  validateQualityFormat,
} from './quality-contract.mjs';

function fixture() {
  const hash = 'a'.repeat(64);
  const manifest = {
    model: { fp32Checksums: { 'model.safetensors': hash } },
    datasets: [
      { repository: 'first', revision: 'one' },
      { repository: 'second', revision: 'two' },
    ],
    qualitySelection: { queriesPerDataset: 2 },
  };
  const input = {
    datasets: manifest.datasets.map((dataset) => ({
      ...dataset,
      fileSha256: hash,
      queries: [
        { sha256: hash, documents: ['good', 'bad'], labels: [1, 0] },
        { sha256: hash, documents: ['bad', 'good'], labels: [0, 1] },
      ],
    })),
  };
  const baseline = {
    schemaVersion: 3,
    complete: true,
    node: process.version,
    execArgv: [],
    scriptSha256: hash,
    contractSha256: hash,
    manifestSha256: hash,
    inputSha256: hash,
    addonSha256: hash,
    metalSha256: hash,
    executionEnvironmentSha256: hash,
    computeDtype: 'float32',
    checksums: structuredClone(manifest.model.fp32Checksums),
    info: {
      family: 'xlm_roberta_reranker',
      profile: 'xlmr-cross-encoder-v1',
      scoreType: 'sigmoid',
      quantized: false,
      policy: {
        maxPairTokens: 8192,
        maxQueryTokens: 512,
        maxDocumentTokens: 8192,
        maxBatchSize: 8,
        maxBatchTokens: 8192,
        maxDocuments: 1024,
        maxInputTokens: 1000000,
      },
    },
    datasets: input.datasets.map((dataset) => ({
      ...dataset,
      queries: dataset.queries.map((query, queryIndex) => ({
        queryIndex,
        sha256: query.sha256,
        ranking: queryIndex === 0 ? [0, 1] : [1, 0],
        ndcg10: 1,
        mrr10: 1,
      })),
      ndcg10: 1,
      mrr10: 1,
    })),
  };
  return { baseline, identity: structuredClone(baseline), input, manifest };
}

void test('duplicate query text retains distinct candidate rankings by sample index', () => {
  const { baseline, identity, input, manifest } = fixture();
  validateQualityBaseline(baseline, identity, input, manifest);
  baseline.datasets[0].queries[1] = baseline.datasets[0].queries[0];
  assert.throws(() => validateQualityDatasets(baseline, input, manifest));
});

void test('quantized self-reference and changed native execution cannot pass as FP32', () => {
  for (const mutate of [
    (b) => {
      b.info.quantized = true;
    },
    (b) => {
      b.computeDtype = 'float16';
    },
    (b) => {
      b.checksums['model.safetensors'] = 'b'.repeat(64);
    },
    (b) => {
      b.addonSha256 = 'b'.repeat(64);
    },
    (b) => {
      b.metalSha256 = 'b'.repeat(64);
    },
    (b) => {
      b.executionEnvironmentSha256 = 'b'.repeat(64);
    },
    (b) => {
      b.scriptSha256 = 'b'.repeat(64);
    },
    (b) => {
      b.node = 'changed';
    },
    (b) => {
      delete b.executionEnvironmentSha256;
    },
    (b) => {
      b.schemaVersion = 2;
    },
    (b) => {
      b.complete = false;
    },
    (b) => {
      delete b.info.policy;
    },
    (b) => {
      delete b.info.policy.maxBatchTokens;
    },
    (b) => {
      b.info.policy.maxBatchTokens = NaN;
    },
  ]) {
    const { baseline, identity, input, manifest } = fixture();
    mutate(baseline);
    assert.throws(() => validateQualityBaseline(baseline, identity, input, manifest));
  }
});

void test('dense checkpoints and other formats cannot stand in for a quantization target', () => {
  const affine = { quantization: { mode: 'affine', bits: 4, group_size: 64 } };
  const info = { quantized: true, weightFormats: ['dense', 'affine-4bit'] };
  assert.equal(validateQualityFormat(info, affine, 'affine'), 'affine');
  assert.throws(() => validateQualityFormat({ quantized: false, weightFormats: ['dense'] }, affine, 'affine'));
  assert.throws(() => validateQualityFormat(info, affine, 'mxfp4'));
  assert.throws(() =>
    validateQualityFormat(info, { quantization: { mode: 'mxfp4', bits: 4, group_size: 32 } }, 'mxfp4'),
  );
  assert.throws(() =>
    validateQualityFormat(info, { quantization: { mode: 'affine', bits: 8, group_size: 64 } }, 'affine'),
  );
});

void test('matching counts do not hide missing datasets, revisions, or changed metrics', () => {
  for (const mutate of [
    (b) => {
      b.datasets[1] = b.datasets[0];
    },
    (b) => {
      b.datasets[1].revision = 'different';
    },
    (b) => {
      b.datasets[0].ndcg10 = 0;
    },
    (b) => {
      b.datasets[0].queries[0].ranking = [1, 0];
    },
    (b) => {
      b.datasets[0].queries[0].mrr10 = NaN;
    },
  ]) {
    const { baseline, input, manifest } = fixture();
    mutate(baseline);
    assert.throws(() => validateQualityDatasets(baseline, input, manifest));
  }
});

void test('ranking validation rejects omitted and repeated candidates', () => {
  assert.throws(() => rankingMetrics([0], [1, 0]));
  assert.throws(() => rankingMetrics([0, 0], [1, 0]));
  assert.throws(() => rankingMetrics([0, 1], [0, 0]));
  assert.deepEqual(rankingMetrics([1, 0], [1, 0]), { ndcg10: 1 / Math.log2(3), mrr10: 0.5 });
});
