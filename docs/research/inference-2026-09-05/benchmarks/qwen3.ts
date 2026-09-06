import { writeFile } from 'node:fs/promises';
// Run with oxnode; see ../validation.md for arguments and measurement protocol.
import { createRequire } from 'node:module';
import { performance } from 'node:perf_hooks';
const [binding, modelPath, output, revision] = process.argv.slice(2);
if (!binding || !modelPath || !output || !revision)
  throw new Error('Expected binding, checkpoint, output and revision arguments');
const core: typeof import('../../../../packages/core/index.cjs') = createRequire(import.meta.url)(binding);
const begin = performance.now();
const model = await core.Qwen3Model.load(modelPath);
const loadMs = performance.now() - begin;
const runs = [];
for (const rows of [1, 2, 4, 8]) {
  for (let round = -1; round < 3; round++) {
    await model.resetCaches();
    const start = performance.now();
    const results = await Promise.all(
      Array.from({ length: rows }, (_, i) =>
        model.chatSessionStart(
          [
            {
              role: 'user',
              content: 'Explain how a computer runs a program, using numbered steps and short examples.',
            },
          ],
          {
            cacheOwnerId: `bench-${round}-${i}`,
            cacheRootOwnerId: `bench-${round}-${i}`,
            maxNewTokens: 128,
            temperature: 0,

            repetitionPenalty: i % 2 === 0 ? 1.15 : 1.0,
            presencePenalty: i % 2 === 0 ? 0.2 : 0,
            frequencyPenalty: i % 2 === 0 ? 0.1 : 0,
            maxConsecutiveTokens: 0,
            maxNgramRepeats: 0,
            reportPerformance: true,
          },
        ),
      ),
    );
    const ms = performance.now() - start;
    const stats = await model.schedulerStats();
    if (round >= 0)
      runs.push({
        rows,
        round,
        ms,
        tokens: results.reduce((a, r) => a + r.numTokens, 0),
        raw: results.map((r) => r.rawText),
        finish: results.map((r) => r.finishReason),
        performance: results.map((r) => r.performance),
        maxOccupancy: stats.maxBatchOccupancy,
      });
  }
}
await model.resetCaches();
await writeFile(output, JSON.stringify({ revision, modelPath, loadMs, runs }, null, 2));
console.log(
  JSON.stringify({ revision, loadMs, summary: runs.map(({ raw: _raw, performance: _performance, ...r }) => r) }),
);
