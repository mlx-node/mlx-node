import { writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { performance } from 'node:perf_hooks';
const [binding, modelPath, draftPath, output, revision, widthsArg = '1,2,4', roundsArg = '3', family = 'Gemma4'] =
  process.argv.slice(2);
const core: typeof import('../../../../packages/core/index.cjs') = createRequire(import.meta.url)(binding);
const model =
  family === 'MuseGlimmer'
    ? await core.MuseGlimmerModel.load(modelPath)
    : await core.Gemma4Model.load(modelPath, { draftModelPath: draftPath });
if (!model.hasMtpWeights()) throw new Error('Missing DSpark draft');
const runs = [];
for (const rows of widthsArg.split(',').map(Number)) {
  for (let round = -1; round < Number(roundsArg); round++) {
    await model.resetCaches();
    const start = performance.now();
    const results = await Promise.all(
      Array.from({ length: rows }, (_, i) =>
        model.chatSessionStart(
          [
            {
              role: 'user',
              content:
                i % 2 === 0
                  ? 'Give a simple recipe for pancakes with numbered steps.'
                  : 'Explain how a computer runs a program, using numbered steps and short examples.',
            },
          ],
          {
            cacheOwnerId: `spec-${i}`,
            cacheRootOwnerId: `spec-${i}`,
            maxNewTokens: 128,
            temperature: 0,
            reasoningEffort: 'none',
            repetitionPenalty: 1,
            presencePenalty: 0,
            frequencyPenalty: 0,
            maxConsecutiveTokens: 0,
            maxNgramRepeats: 0,
            enableMtp: true,
            mtpDepth: 7,
            mtpAdaptiveDepth: false,
            reportPerformance: true,
          },
        ),
      ),
    );
    const ms = performance.now() - start;
    const stats = await model.schedulerStats();
    if (results.some((result) => (result.performance?.mtpCycles ?? 0) === 0))
      throw new Error('A request did not speculate');
    if (round >= 0)
      runs.push({ rows, round, ms, tokens: results.reduce((sum, r) => sum + r.numTokens, 0), results, stats });
  }
}
await model.resetCaches();
await writeFile(output, JSON.stringify({ revision, modelPath, draftPath, runs }, null, 2));
console.log(
  JSON.stringify(
    runs.map(({ rows, round, ms, tokens, stats }) => ({
      rows,
      round,
      ms,
      tokens,
      maxOccupancy: stats.maxBatchOccupancy,
    })),
  ),
);
