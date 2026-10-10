/// <reference types="node" />

/** Real-checkpoint acceptance, one resident model per process.
 * oxnode scripts/clef/verify-release.ts /path/to/checkpoint /path/to/results.json [sdk.mjs]
 */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { memoryStats, resetPeakMemory } from '@mlx-node/core';
import type { ClefRequest, ClefResult } from '@mlx-node/lm';
import { createInferenceHost } from '@mlx-node/server/host';

const path = resolve(process.argv[2]);
const destination = process.argv[3];
assert.ok(destination, 'Supply checkpoint and result paths');
const cases: { id: string; request: ClefRequest; expected: Record<string, string | boolean> }[] = JSON.parse(
  await readFile(fileURLToPath(new URL('../../__test__/fixtures/clef/acceptance.json', import.meta.url)), 'utf8'),
);
const start = performance.now();
resetPeakMemory();
const host = await createInferenceHost({
  modelsDir: dirname(path),
  model: basename(path),
  port: 0,
  disableStore: true,
  authToken: 'clef-acceptance',
  sweepOrphanTempRoots: false,
});
try {
  await host.loadModel(basename(path));
  const loadMs = performance.now() - start;
  const model = host.server.registry.decisions.get(basename(path));
  assert.ok(model, 'Host must discover and load the decision family');
  assert.equal(host.server.registry.listSessionRegistries().length, 0);
  const results: { id: string; milliseconds: number; result: ClefResult; expectedPassed: boolean }[] = [];
  for (const test of cases) {
    const began = performance.now();
    const result = await model.decideRaw(JSON.stringify(test.request));
    const expectedPassed = Object.entries(test.expected).every(([id, expected]) => {
      const answer = result.answers[id];
      return answer.type === 'noul'
        ? answer.noul >= 0.5 === expected
        : answer.type === 'choice' && answer.choice === expected;
    });
    results.push({ id: test.id, milliseconds: performance.now() - began, result, expectedPassed });
    console.log(JSON.stringify(results.at(-1)));
  }
  assert.deepEqual(await model.decideRaw(JSON.stringify(cases[0].request)), results[0].result);
  const controller = new AbortController();
  const cancelled = model.decideRaw(JSON.stringify(cases.at(-1)!.request), { signal: controller.signal });
  const timer = setTimeout(() => controller.abort(), 10);
  try {
    await assert.rejects(cancelled);
  } finally {
    clearTimeout(timer);
  }
  assert.deepEqual(await model.decideRaw(JSON.stringify(cases[0].request)), results[0].result);
  const headers = { Authorization: 'Bearer clef-acceptance', 'Content-Type': 'application/json' };
  const response = await fetch(`${host.url}/v1/systemone`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ model: basename(path), ...cases[0].request }),
  });
  assert.equal(response.status, 200, await response.clone().text());
  assert.ok(response.headers.get('x-typesafe-request-id'));
  assert.deepEqual(await response.json(), JSON.parse(JSON.stringify({ model: basename(path), ...results[0].result })));
  const media = await fetch(`${host.url}/v1/systemone`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ model: basename(path), ...cases[0].request, images: ['unsupported'] }),
  });
  assert.equal(media.status, 400);
  if (process.argv[4]) {
    const { TypeSafeClient } = await import(pathToFileURL(resolve(process.argv[4])).href);
    const client = new TypeSafeClient({ baseURL: host.url, apiKey: 'clef-acceptance', defaultModel: basename(path) });
    const sdk = await client.systemOne(cases[1].request).withResponse();
    assert.deepEqual(sdk.data, JSON.parse(JSON.stringify({ model: basename(path), ...results[1].result })));
    assert.ok(sdk.response.headers.get('x-typesafe-request-id'));
    assert.ok((await client.models.list()).some((m: { name: string }) => m.name === basename(path)));
  }
  const report = {
    checkpoint: path,
    loadMs,
    memory: memoryStats(),
    repeat: 'passed',
    midInferenceCancellation: 'passed',
    http: 'passed',
    media: 'rejected',
    sdk: process.argv[4] ? 'passed' : 'not requested',
    cases: results,
  };
  await writeFile(destination, JSON.stringify(report, null, 2) + '\n');
  assert.ok(
    results.every((r) => r.expectedPassed),
    'A semantic acceptance case failed; inspect the saved results',
  );
} finally {
  await host.close();
}
