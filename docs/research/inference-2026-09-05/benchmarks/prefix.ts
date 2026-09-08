import { writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { performance } from 'node:perf_hooks';
const [binding, checkpoint, output, revision] = process.argv.slice(2);
if (!binding || !checkpoint || !output || !revision) throw new Error('binding checkpoint output revision required');
const core: typeof import('../../../../packages/core/index.cjs') = createRequire(import.meta.url)(binding);
const model = await core.Qwen3Model.load(checkpoint);
const runs = [];
const prompt = 'A cache stores reusable data. A processor executes instructions. '.repeat(128);
for (let round = -1; round < 4; round++) {
  await model.resetCaches();
  const config = {
    maxNewTokens: 8,
    temperature: 0,
    reportPerformance: true,
    cacheOwnerId: 'prefix-bench',
    cacheRootOwnerId: 'prefix-bench',
  };
  const first = await model.chatSessionStart([{ role: 'user', content: prompt + 'Summarize this briefly.' }], config);
  const start = performance.now();
  const next = await model.chatSessionContinue(
    [
      { role: 'user', content: prompt + 'Summarize this briefly.' },
      { role: 'assistant', content: first.rawText },
      { role: 'user', content: 'Explain why the cached result can be reused. '.repeat(16) },
    ],
    config,
  );
  if (round >= 0)
    runs.push({
      round,
      ms: performance.now() - start,
      tokens: next.numTokens,
      raw: next.rawText,
      performance: next.performance,
    });
}
await model.resetCaches();
await writeFile(output, JSON.stringify({ revision, runs }, null, 2));
console.log(JSON.stringify({ revision, ms: runs.map((r) => r.ms) }));
