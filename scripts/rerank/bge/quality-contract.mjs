import assert from 'node:assert/strict';

export function validateQualityPolicy(policy) {
  const keys = [
    'maxPairTokens',
    'maxQueryTokens',
    'maxDocumentTokens',
    'maxBatchSize',
    'maxBatchTokens',
    'maxDocuments',
    'maxInputTokens',
  ];
  assert.ok(policy && typeof policy === 'object');
  assert.deepEqual(Object.keys(policy).sort(), keys.sort());
  for (const value of Object.values(policy)) assert.ok(Number.isSafeInteger(value) && value > 0 && value <= 2147483647);
  assert.ok(policy.maxQueryTokens + 4 <= policy.maxPairTokens);
  assert.ok(policy.maxBatchTokens >= policy.maxPairTokens);
}

export function validateQualityFormat(info, config, expected) {
  assert.equal(info.quantized, true, 'Quantization quality needs a packed checkpoint');
  const quantization = config.quantization ?? config.quantization_config;
  assert.ok(quantization && ['affine', 'mxfp4'].includes(quantization.mode));
  if (expected !== undefined) assert.equal(quantization.mode, expected, 'Measured quantization format differs');
  assert.equal(quantization.bits, 4);
  assert.equal(quantization.group_size, quantization.mode === 'affine' ? 64 : 32);
  const format = quantization.mode === 'affine' ? 'affine-4bit' : 'mxfp4';
  assert.ok(info.weightFormats.includes(format));
  assert.ok(info.weightFormats.every((value) => value === format || value === 'dense'));
  return quantization.mode;
}

export function rankingMetrics(ranking, labels) {
  assert.equal(ranking.length, labels.length, 'Every candidate must be ranked');
  assert.deepEqual(
    ranking.toSorted((a, b) => a - b),
    labels.map((_, i) => i),
  );
  assert.ok(labels.some((v) => v > 0));
  assert.ok(labels.every((v) => Number.isFinite(v) && v >= 0));
  const dcg = (values) => values.reduce((sum, rel, i) => sum + rel / Math.log2(i + 2), 0);
  const first = ranking.slice(0, 10).findIndex((i) => labels[i] > 0);
  return {
    ndcg10: dcg(ranking.slice(0, 10).map((i) => labels[i])) / dcg(labels.toSorted((a, b) => b - a).slice(0, 10)),
    mrr10: first < 0 ? 0 : 1 / (first + 1),
  };
}

export function validateQualityInput(input, manifest) {
  assert.deepEqual(
    input.datasets.map((d) => [d.repository, d.revision]),
    manifest.datasets.map((d) => [d.repository, d.revision]),
    'Quality datasets must match the pinned manifest in order',
  );
  for (const dataset of input.datasets) {
    assert.equal(dataset.queries.length, manifest.qualitySelection.queriesPerDataset);
    assert.match(dataset.fileSha256, /^[a-f0-9]{64}$/);
    for (const query of dataset.queries) {
      assert.match(query.sha256, /^[a-f0-9]{64}$/);
      assert.equal(query.documents.length, query.labels.length);
      rankingMetrics(
        query.labels.map((_, i) => i),
        query.labels,
      );
    }
  }
}

export function validateQualityDatasets(report, input, manifest) {
  validateQualityInput(input, manifest);
  assert.equal(report.datasets.length, input.datasets.length, 'Incomplete quality report');
  for (const [i, dataset] of input.datasets.entries()) {
    const measured = report.datasets[i];
    for (const key of ['repository', 'revision', 'fileSha256']) assert.equal(measured[key], dataset[key]);
    assert.equal(measured.queries.length, dataset.queries.length);
    for (const [queryIndex, query] of dataset.queries.entries()) {
      const row = measured.queries[queryIndex];
      assert.equal(row.queryIndex, queryIndex);
      assert.equal(row.sha256, query.sha256);
      const metrics = rankingMetrics(row.ranking, query.labels);
      assert.equal(row.ndcg10, metrics.ndcg10);
      assert.equal(row.mrr10, metrics.mrr10);
    }
    for (const key of ['ndcg10', 'mrr10'])
      assert.equal(measured[key], measured.queries.reduce((sum, row) => sum + row[key], 0) / measured.queries.length);
  }
}

export function validateQualityBaseline(baseline, identity, input, manifest) {
  assert.equal(baseline.schemaVersion, 3, 'Quality baseline needs complete execution identity');
  assert.equal(baseline.complete, true, 'Quality baseline must be complete');
  for (const key of [
    'inputSha256',
    'addonSha256',
    'metalSha256',
    'executionEnvironmentSha256',
    'scriptSha256',
    'contractSha256',
    'manifestSha256',
  ]) {
    assert.match(identity[key], /^[a-f0-9]{64}$/);
    assert.equal(baseline[key], identity[key], `Quality baseline ${key} differs`);
  }
  assert.equal(typeof identity.node, 'string');
  assert.equal(baseline.node, identity.node);
  assert.ok(Array.isArray(identity.execArgv));
  assert.deepEqual(baseline.execArgv, identity.execArgv);
  assert.equal(baseline.computeDtype, 'float32');
  assert.equal(baseline.info.family, 'xlm_roberta_reranker');
  assert.equal(baseline.info.profile, 'xlmr-cross-encoder-v1');
  assert.equal(baseline.info.scoreType, 'sigmoid');
  assert.equal(baseline.info.quantized, false, 'Quality baseline must be unquantized FP32');
  validateQualityPolicy(baseline.info.policy);
  assert.deepEqual(baseline.checksums, manifest.model.fp32Checksums, 'Quality baseline must use the pinned FP32 files');
  validateQualityDatasets(baseline, input, manifest);
}
