import assert from 'node:assert/strict';
import { readFile, rm, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { JinaRerankerModel, getMemorySnapshot } from '../../packages/core/index.cjs';
const [modelPath, referencePath, reportPath] = process.argv.slice(2);
if (!reportPath) throw new Error('Usage: node scripts/rerank/compare.mjs MODEL REFERENCE REPORT');
assert.notEqual(resolve(reportPath), resolve(referencePath), 'Report must not overwrite the reference');
// This script does not resume: a failed rerun must not retain an old passing report.
await rm(reportPath, { force: true });
const cases = JSON.parse(await readFile(new URL('./cases.json', import.meta.url), 'utf8'));
const references = JSON.parse(await readFile(referencePath, 'utf8'));
assert.equal(references.length, cases.length, 'reference case count');
const { tolerance } = JSON.parse(await readFile(new URL('./manifest.json', import.meta.url), 'utf8'));
const report = [];
for (const [i, test] of cases.entries()) {
  const loadedAt = performance.now();
  const model = await JinaRerankerModel.load(modelPath, JSON.stringify(test.policy ?? {}));
  const loadMs = performance.now() - loadedAt;
  try {
    const started = performance.now();
    const task = model.startRerank(JSON.stringify({ query: test.query, documents: test.documents }));
    let result;
    try {
      result = JSON.parse(await task.result());
    } finally {
      // Keep the native task owner alive until its worker has finished.
      task.cancel();
    }
    const scores = result.results.toSorted((a, b) => a.index - b.index).map((r) => r.relevanceScore);
    const expected = references[i];
    assert.equal(expected.name, test.name, 'reference case identity');
    assert.equal(scores.length, expected.scores.length, 'score count');
    assert.equal(scores.length, test.documents.length, 'candidate preservation');
    assert.ok([...scores, ...expected.scores].every(Number.isFinite), 'finite scores');
    const errors = scores.map((v, i) => Math.abs(v - expected.scores[i]));
    const maxError = Math.max(...errors),
      meanError = errors.reduce((a, b) => a + b, 0) / errors.length;
    let inversions = 0;
    for (let a = 0; a < scores.length; a++)
      for (let b = a + 1; b < scores.length; b++) {
        const margin = expected.scores[a] - expected.scores[b];
        if (Math.abs(margin) > tolerance.separatedPairMargin && Math.sign(margin) !== Math.sign(scores[a] - scores[b]))
          inversions++;
      }
    const expectedTokens = expected.blocks.reduce((n, b) => n + b.tokens.length, 0);
    const row = {
      name: test.name,
      loadMs,
      computeMs: performance.now() - started,
      maxError,
      meanError,
      inversions,
      tokens: result.usage.promptTokens,
      blocks: result.meta.blocks,
      peakMemoryBytes: Number(getMemorySnapshot().peakBytes),
      activeMemoryBytes: Number(getMemorySnapshot().activeBytes),
    };
    report.push(row);
    console.log(JSON.stringify(row));
    if (
      maxError > tolerance.maxAbsoluteScoreError ||
      meanError > tolerance.meanAbsoluteScoreError ||
      inversions ||
      result.usage.promptTokens !== expectedTokens ||
      result.meta.blocks !== expected.blocks.length
    )
      throw new Error(`Golden mismatch: ${test.name}`);
  } finally {
    await model.dispose();
  }
}
await writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
