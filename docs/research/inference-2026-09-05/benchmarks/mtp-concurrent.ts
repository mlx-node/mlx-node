import { writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { performance } from 'node:perf_hooks';
const [
  binding,
  modelPath,
  output,
  revision,
  family = 'Qwen35',
  widthsArg = '1,2',
  roundsArg = '2',
  temperatureArg = '0',
] = process.argv.slice(2);
const core: typeof import('../../../../packages/core/index.cjs') = createRequire(import.meta.url)(binding);
const model =
  family === 'Qwen35Moe' ? await core.Qwen35MoeModel.load(modelPath) : await core.Qwen35Model.load(modelPath);
if (!model.hasMtpWeights()) throw new Error('Missing native MTP weights');
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
            temperature: Number(temperatureArg),
            reasoningEffort: 'none',
            repetitionPenalty: 1,
            presencePenalty: 0,
            frequencyPenalty: 0,
            maxConsecutiveTokens: 0,
            maxNgramRepeats: 0,
            enableMtp: true,
            mtpDepth: 3,
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
await writeFile(
  output,
  JSON.stringify({ revision, modelPath, family, temperature: Number(temperatureArg), runs }, null, 2),
);
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
