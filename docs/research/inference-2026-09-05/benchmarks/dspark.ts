import { writeFile } from 'node:fs/promises';
// Run with oxnode; see ../validation.md for arguments and measurement protocol.
import { createRequire } from 'node:module';
import { performance } from 'node:perf_hooks';
const [binding, modelPath, draftPath, output, revision, temperatureArg = '0', adaptiveArg = 'false'] =
  process.argv.slice(2);
const temperature = Number(temperatureArg);
if (!Number.isFinite(temperature) || temperature < 0) throw new Error('Invalid temperature');
if (!binding || !modelPath || !draftPath || !output || !revision)
  throw new Error('Expected binding, checkpoint, output and revision arguments');
const core: typeof import('../../../../packages/core/index.cjs') = createRequire(import.meta.url)(binding);
const started = performance.now();
const model = await core.Gemma4Model.load(modelPath, { draftModelPath: draftPath });
if (!model.hasMtpWeights()) throw new Error('Draft not attached');
const loadMs = performance.now() - started;
const runs = [];
for (let round = -1; round < 3; round++) {
  await model.resetCaches();
  const start = performance.now();
  const result = await model.chatSessionStart(
    [{ role: 'user', content: 'Give a simple recipe for pancakes with numbered steps.' }],
    {
      maxNewTokens: 200,
      temperature,
      reasoningEffort: 'none',
      repetitionPenalty: 1.1,
      presencePenalty: 0.1,
      frequencyPenalty: 0.1,
      maxConsecutiveTokens: 0,
      maxNgramRepeats: 0,
      enableMtp: true,
      mtpDepth: 7,
      mtpAdaptiveDepth: adaptiveArg === 'true',
      reportPerformance: true,
    },
  );
  const ms = performance.now() - start;
  if (!((result.performance?.mtpCycles ?? 0) > 0)) throw new Error('No speculative cycles');
  if (round >= 0)
    runs.push({
      round,
      ms,
      tokens: result.numTokens,
      raw: result.rawText,
      finish: result.finishReason,
      performance: result.performance,
    });
}
await model.resetCaches();
await writeFile(output, JSON.stringify({ revision, modelPath, draftPath, temperature, loadMs, runs }, null, 2));
console.log(JSON.stringify({ revision, loadMs, summary: runs.map(({ raw: _raw, ...r }) => r) }));
