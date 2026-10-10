/// <reference types="node" />

/** End-to-end check against an externally generated tiny-checkpoint oracle.
 * oxnode scripts/clef/verify.ts /tmp/clef-tiny [/path/to/sdk/dist/index.mjs]
 */
import assert from 'node:assert/strict';
import { readFile, mkdtemp, rm, cp } from 'node:fs/promises';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

import { ClefModel, ClefCancellation, convertModel } from '@mlx-node/core';
import { ClefModel as Model } from '@mlx-node/lm';
import { createServer } from '@mlx-node/server';
import { createInferenceHost } from '@mlx-node/server/host';

const path = process.argv[2];
if (!path) throw new Error('Supply the generated tiny checkpoint directory');
const oracle = JSON.parse(await readFile(join(path, 'oracle.json'), 'utf8'));
const model = await ClefModel.load(path);
const raw = JSON.stringify(oracle.request);
const first = JSON.parse(await model.decideJson(raw));
assert.equal(first.input_tokens, oracle.input_tokens);
assert.equal(first.questions.length, oracle.probabilities.length);
let maxError = 0;
for (const [q, row] of first.questions.entries()) {
  assert.equal(row.probabilities.length, oracle.probabilities[q].length);
  for (const [i, p] of row.probabilities.entries())
    maxError = Math.max(maxError, Math.abs(p - oracle.probabilities[q][i]));
}
assert.ok(maxError < 0.001, `Hybrid backbone/head probability drift: ${maxError}`);
const cancellation = new ClefCancellation();
cancellation.cancel();
await assert.rejects(model.decideJson(raw, cancellation), /cancelled/);
assert.deepEqual(JSON.parse(await model.decideJson(raw)), first, 'Requests must not reuse prior state');
const server = await createServer({ port: 0, host: '127.0.0.1', disableStore: true, authToken: 'local-clef-test' });
server.registry.registerDecision('clef-tiny', new Model(model));
server.registry.registerDecision('alias', server.registry.decisions.get('clef-tiny')!);
try {
  const baseURL = `http://127.0.0.1:${(server.server.address() as AddressInfo).port}`;
  assert.equal((await fetch(`${baseURL}/v1/systemone`, { method: 'POST', body: raw })).status, 401);
  if (process.argv[3]) {
    const { TypeSafeClient } = await import(pathToFileURL(process.argv[3]).href);
    const client = new TypeSafeClient({ baseURL, apiKey: 'local-clef-test', defaultModel: 'alias' });
    const response = await client.systemOne(oracle.request).withResponse();
    assert.equal(response.data.model, 'clef-tiny');
    assert.equal(response.data.usage.input_tokens, oracle.input_tokens);
    assert.equal(response.data.usage.output_tokens, 0);
    assert.ok(response.response.headers.get('x-typesafe-request-id'));
    assert.equal((await client.models.list()).length, 2);
  }
} finally {
  await server.close();
}
const modelsDir = await mkdtemp(join(tmpdir(), 'clef-host-'));
try {
  await cp(path, join(modelsDir, 'clef-tiny'), { recursive: true });
  const host = await createInferenceHost({ modelsDir, port: 0, disableStore: true, sweepOrphanTempRoots: false });
  try {
    assert.equal(host.boundModel, 'clef-tiny');
    assert.equal(host.models[0].modelType, 'clef');
    const response = await fetch(`${host.url}/v1/systemone`, {
      method: 'POST',
      body: JSON.stringify({ model: 'clef-tiny', ...oracle.request }),
    });
    assert.equal(response.status, 200, await response.clone().text());
    const data = await response.json();
    assert.equal(data.model, 'clef-tiny');
    assert.equal(data.usage.input_tokens, oracle.input_tokens);
    assert.equal(host.server.registry.listSessionRegistries().length, 0);
  } finally {
    await host.close();
  }
} finally {
  await rm(modelsDir, { recursive: true, force: true });
}
const converted = await mkdtemp(join(tmpdir(), 'clef-conversion-'));
try {
  await convertModel({ inputDir: path, outputDir: converted, dtype: 'float32', quantize: false });
  for (const name of ['joint_head.safetensors', 'joint_head_config.json']) {
    assert.deepEqual(await readFile(join(converted, name)), await readFile(join(path, name)));
  }
  const roundTrip = await ClefModel.load(converted);
  const result = JSON.parse(await roundTrip.decideJson(raw));
  for (const [q, row] of result.questions.entries()) {
    for (const [i, p] of row.probabilities.entries())
      assert.ok(Math.abs(Number(p) - first.questions[q].probabilities[i]) < 1e-6);
  }
  const quantized = join(converted, 'q8');
  await convertModel({
    inputDir: path,
    outputDir: quantized,
    dtype: 'float32',
    quantize: true,
    quantBits: 8,
    quantGroupSize: 32,
  });
  const q8 = await ClefModel.load(quantized);
  const qResult = JSON.parse(await q8.decideJson(raw));
  assert.equal(qResult.input_tokens, first.input_tokens);
  assert.equal(qResult.questions.length, first.questions.length);
  for (const row of qResult.questions) {
    assert.ok(row.probabilities.every((p: number) => Number.isFinite(p) && p >= 0 && p <= 1));
    assert.ok(Math.abs(row.probabilities.reduce((sum: number, p: number) => sum + p, 0) - 1) < 1e-5);
  }
  await rm(join(converted, 'joint_head.safetensors'));
  await assert.rejects(ClefModel.load(converted), /Missing CLEF checkpoint asset/);
} finally {
  await rm(converted, { recursive: true, force: true });
}
console.log(
  JSON.stringify({
    inputTokens: first.input_tokens,
    maxProbabilityError: maxError,
    repeatAndCancel: 'passed',
    conversion: 'passed',
    quantizedBackbone: 'passed',
    discoveredHost: 'passed',
    incompleteCheckpoint: 'rejected',
    sdk: process.argv[3] ? 'passed' : 'not requested',
  }),
);
