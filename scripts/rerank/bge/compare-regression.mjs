/** Compare separately collected, quiet-machine control runs. */
import assert from 'node:assert/strict';
import { readFile, rm, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const [baselinePath, currentPath, output] = process.argv.slice(2);
if (!output) throw Error('Usage: node compare-regression.mjs BASELINE CURRENT REPORT');
assert.ok(
  ![baselinePath, currentPath].some((path) => resolve(path) === resolve(output)),
  'Output must not overwrite an input',
);
// A failed rerun must not leave an earlier passing result at the requested path.
await rm(output, { force: true });
const read = async (path) => JSON.parse(await readFile(path, 'utf8'));
const [baseline, current, manifest] = await Promise.all([
  read(baselinePath),
  read(currentPath),
  read(new URL('../../../fixtures/rerank/bge/manifest.json', import.meta.url)),
]);
assert.equal(baseline.workload, 'unified-memory-chat8-rerank50-v1');
assert.equal(current.workload, baseline.workload);
assert.equal(current.chatPath, baseline.chatPath);
assert.equal(current.jinaPath, baseline.jinaPath);
assert.deepEqual(current.hardware, baseline.hardware);
for (const report of [baseline, current]) {
  assert.ok(typeof report.identity.node === 'string' && report.identity.node.length > 0);
  assert.ok(Array.isArray(report.identity.execArgv) && report.identity.execArgv.every((v) => typeof v === 'string'));
  for (const key of ['executionEnvironmentSha256', 'scriptSha256', 'manifestSha256', 'chatSha256', 'jinaSha256'])
    assert.match(report.identity[key], /^[a-f0-9]{64}$/);
}
assert.equal(baseline.identity.sourceRevision, manifest.baselineRevision);
for (const key of [
  'node',
  'execArgv',
  'executionEnvironmentSha256',
  'scriptSha256',
  'manifestSha256',
  'chatSha256',
  'jinaSha256',
]) {
  assert.deepEqual(current.identity[key], baseline.identity[key], `Regression identity mismatch: ${key}`);
}
for (const key of ['sourceRevision', 'sourceSha256', 'lockSha256', 'metalSha256']) {
  for (const report of [baseline, current])
    assert.match(report.identity[key], key === 'sourceRevision' ? /^[a-f0-9]{40}$/ : /^[a-f0-9]{64}$/);
}
for (let i = 0; i < 3; i++) {
  assert.equal(baseline.rounds[i].chat.outputTokens.length, manifest.performance.iterations);
  assert.ok(baseline.rounds[i].chat.outputTokens.every((v) => Number.isInteger(v) && v > 0));
  assert.deepEqual(
    current.rounds[i].chat.outputTokens,
    baseline.rounds[i].chat.outputTokens,
    'Chat output workload differs',
  );
}
for (const report of [baseline, current]) {
  assert.match(report.addonSha256, /^[a-f0-9]{64}$/);
  assert.equal(report.rounds.length, 3);
  for (const round of report.rounds) {
    for (const family of ['chat', 'jina']) {
      const values = round[family].values;
      assert.equal(values.length, manifest.performance.iterations);
      assert.ok(values.every((v) => Number.isFinite(v) && v > 0));
      const sorted = values.toSorted((a, b) => a - b);
      assert.equal(round[family].medianMs, (sorted[4] + sorted[5]) / 2);
    }
  }
}
for (const run of [baseline, current]) {
  assert.ok(Number.isSafeInteger(run.afterClose?.activeBytes) && run.afterClose.activeBytes >= 0);
}
assert.equal(
  current.afterClose.activeBytes,
  baseline.afterClose.activeBytes,
  'BGE coexistence must return to the existing chat/Jina active-memory baseline after close',
);
const threshold = manifest.thresholds.baselineMaxMedianLatencyRegression;
const families = ['chat', 'jina'].map((family) => {
  const medianLatencyChanges = current.rounds.map(
    (r, i) => r[family].medianMs / baseline.rounds[i][family].medianMs - 1,
  );
  return { family, medianLatencyChanges, passed: !medianLatencyChanges.every((v) => v > threshold) };
});
const report = {
  baselinePath,
  currentPath,
  baselineSource: baseline.identity,
  currentSource: current.identity,
  baselineAddonSha256: baseline.addonSha256,
  currentAddonSha256: current.addonSha256,
  threshold,
  afterCloseActiveBytes: { baseline: baseline.afterClose.activeBytes, current: current.afterClose.activeBytes },
  families,
  passed: families.every((f) => f.passed),
};
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));
if (!report.passed) process.exitCode = 1;
