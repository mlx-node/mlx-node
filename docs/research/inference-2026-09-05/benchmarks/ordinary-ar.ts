import { writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { performance } from 'node:perf_hooks';

const [binding, modelPath, output, revision, mode = 'sync', roundsArg = '5'] = process.argv.slice(2);
if (!binding || !modelPath || !output || !revision || !['sync', 'stream'].includes(mode))
  throw new Error('Expected binding, model, output, revision, sync|stream, rounds');
const core: typeof import('../../../../packages/core/index.cjs') = createRequire(import.meta.url)(binding);
const model = await core.Qwen35Model.load(modelPath);
core.setProfilingEnabled(true);
const messages: import('../../../../packages/core/index.cjs').ChatMessage[] = [
  { role: 'user', content: 'Explain how a computer runs a program, using numbered steps and short examples.' },
];
const config = {
  maxNewTokens: 256,
  temperature: 0,
  enableMtp: false,
  reportPerformance: true,
  maxConsecutiveTokens: 0,
  maxNgramRepeats: 0,
  cacheOwnerId: 'ordinary-bench',
  cacheRootOwnerId: 'ordinary-bench',
};
const runs = [];
for (let round = -1; round < Number(roundsArg); round++) {
  await model.resetCaches();
  const start = performance.now();
  const result =
    mode === 'sync'
      ? await model.chatSessionStart(messages, config)
      : await new Promise<import('../../../../packages/core/index.cjs').ChatStreamChunk>((resolve, reject) => {
          void model
            .chatStreamSessionStart(messages, config, (err, chunk) => {
              if (err) reject(err);
              else if (chunk.done) resolve(chunk);
            })
            .catch(reject);
        });
  if (result.numTokens !== 256) throw new Error(`Unexpected token count: ${result.numTokens}`);
  if (round >= 0) runs.push({ round, ms: performance.now() - start, result });
}
await model.resetCaches();
await writeFile(output, JSON.stringify({ revision, modelPath, mode, runs, profile: core.getProfilingData() }, null, 2));
console.log(
  JSON.stringify(runs.map(({ ms, result }) => ({ ms, tokens: result.numTokens, performance: result.performance }))),
);
