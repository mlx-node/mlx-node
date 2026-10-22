/** Real hardware benchmark; no downloads. One process per checkpoint. */
import assert from 'node:assert/strict';
import { rm, writeFile } from 'node:fs/promises';

import { JinaRerankerModel, getMemorySnapshot, resetPeakMemory } from '../../packages/core/index.cjs';
const [modelPath, output, mode] = process.argv.slice(2);
if (!output) throw new Error('Usage: node scripts/rerank/benchmark.mjs MODEL REPORT [long]');
// A failed fresh run must not leave an earlier successful measurement at this path.
await rm(output, { force: true });
const memory = () =>
  Object.fromEntries(Object.entries(getMemorySnapshot()).map(([k, v]) => [k, typeof v === 'bigint' ? Number(v) : v]));
const loadAt = performance.now();
const model = await JinaRerankerModel.load(modelPath);
const report = {
  recordedAt: new Date().toISOString(),
  modelPath,
  info: JSON.parse(model.infoJson),
  loadMs: performance.now() - loadAt,
  resident: memory(),
  cases: [],
};
const request = (count, repeat) => ({
  query: 'How does unified memory help local machine learning on Apple Silicon?',
  documents: Array.from(
    { length: count },
    (_, i) =>
      `Passage ${i}. ` +
      (i % 3 === 0
        ? 'Apple Silicon uses unified memory for local machine learning inference. '
        : 'The city park has flowers and a pond for birds. '
      ).repeat(repeat),
  ),
});
try {
  for (const repeat of mode === 'long' ? [24] : [4])
    for (const count of [50, 125, 200]) {
      resetPeakMemory();
      const started = performance.now();
      const task = model.startRerank(JSON.stringify(request(count, repeat)));
      let response;
      try {
        response = JSON.parse(await task.result());
      } finally {
        // Keep the native task owner alive until its worker has finished.
        task.cancel();
      }
      assert.equal(response.results.length, count);
      const row = {
        count,
        repeat,
        latencyMs: performance.now() - started,
        ...response.usage,
        blocks: response.meta.blocks,
        memory: memory(),
      };
      report.cases.push(row);
      console.log(JSON.stringify(row));
    }
  const before = memory();
  const task = model.startRerank(JSON.stringify(request(200, 16)));
  const result = task.result();
  const stopped = result.then(
    () => false,
    (e) => {
      assert.match(e.message, /cancelled/);
      return true;
    },
  );
  await new Promise((r) => setTimeout(r, 1000));
  const cancelAt = performance.now();
  task.cancel();
  assert.equal(await stopped, true);
  report.cancellation = { stopMs: performance.now() - cancelAt, before, after: memory() };
  const active = [];
  for (let i = 0; i < 6; i++) {
    const t = model.startRerank(JSON.stringify(request(3, 1)));
    try {
      await t.result();
    } finally {
      t.cancel();
    }
    active.push(memory().activeBytes);
  }
  assert.ok(Math.max(...active.slice(1)) - Math.min(...active.slice(1)) < 1024 * 1024, 'active resources keep growing');
  report.repeatedActiveBytes = active;
} finally {
  await model.dispose();
  report.afterDispose = memory();
}
assert.ok(report.afterDispose.activeBytes < 1024 * 1024, 'model resources not released');
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
