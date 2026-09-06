import { writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { performance } from 'node:perf_hooks';

const [binding, modelPath, output, revision, family = 'Qwen35', depthArg = '3', roundsArg = '3'] =
  process.argv.slice(2);
const core: typeof import('../../../../packages/core/index.cjs') = createRequire(import.meta.url)(binding);
const model =
  family === 'Qwen35Moe' ? await core.Qwen35MoeModel.load(modelPath) : await core.Qwen35Model.load(modelPath);
if (!model.hasMtpWeights()) throw new Error('Checkpoint has no MTP weights');
core.setProfilingEnabled(true);
const runs = [];
for (let round = -1; round < Number(roundsArg); round++) {
  await model.resetCaches();
  const start = performance.now();
  const result = await model.chatSessionStart(
    [{ role: 'user', content: 'Explain how a computer runs a program, using numbered steps and short examples.' }],
    {
      maxNewTokens: 200,
      temperature: 0,
      reasoningEffort: 'none',
      enableMtp: true,
      mtpDepth: Number(depthArg),
      mtpAdaptiveDepth: false,
      maxConsecutiveTokens: 0,
      maxNgramRepeats: 0,
      reportPerformance: true,
    },
  );
  if ((result.performance?.mtpCycles ?? 0) === 0) throw new Error('Turn did not speculate');
  if (round >= 0) runs.push({ round, ms: performance.now() - start, result });
}
await model.resetCaches();
await writeFile(
  output,
  JSON.stringify(
    { revision, modelPath, family, depth: Number(depthArg), runs, profile: core.getProfilingData() },
    null,
    2,
  ),
);
console.log(
  JSON.stringify(runs.map(({ ms, result }) => ({ ms, tokens: result.numTokens, performance: result.performance }))),
);
