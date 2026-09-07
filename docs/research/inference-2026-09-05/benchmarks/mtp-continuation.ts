import { writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
const [binding, modelPath, output, family = 'Qwen35'] = process.argv.slice(2);
const core: typeof import('../../../../packages/core/index.cjs') = createRequire(import.meta.url)(binding);
const model =
  family === 'Qwen35Moe' ? await core.Qwen35MoeModel.load(modelPath) : await core.Qwen35Model.load(modelPath);
const messages: import('../../../../packages/core/index.cjs').ChatMessage[][] = [
  [{ role: 'user', content: 'Explain how a computer runs a program with numbered steps.' }],
  [{ role: 'user', content: 'Give a pancake recipe with numbered steps.' }],
];
const config = (i: number) => ({
  cacheOwnerId: `mtp-cont-${i}`,
  cacheRootOwnerId: `mtp-cont-${i}`,
  maxNewTokens: 96,
  temperature: 0,
  reasoningEffort: 'none' as const,
  enableMtp: true,
  mtpDepth: 3,
  mtpAdaptiveDepth: false,
  maxConsecutiveTokens: 0,
  maxNgramRepeats: 0,
  reportPerformance: true,
});
const first = await Promise.all(messages.map((history, i) => model.chatSessionStart(history, config(i))));
const histories = messages.map((history, i) => [
  ...history,
  { role: 'assistant', content: first[i].text, thinking: first[i].thinking, thinkingEnabled: first[i].thinkingEnabled },
  { role: 'user', content: 'Give three practical examples, with details.' },
]);
const continued = await Promise.all(histories.map((history, i) => model.chatSessionContinue(history, config(i))));
if (continued.some((result) => result.cachedTokens <= 0 || (result.performance?.mtpCycles ?? 0) !== 0))
  throw new Error('Cached continuation must reuse its target prefix without unseeded MTP');
await model.resetCaches();
const sampled = await Promise.all(
  messages.map((history, i) => model.chatSessionStart(history, { ...config(i), temperature: 0.7 })),
);
if (sampled.some((result) => result.numTokens <= 0 || (result.performance?.mtpCycles ?? 0) === 0))
  throw new Error('Sampled request did not speculate');
await writeFile(
  output,
  JSON.stringify({ binding, family, first, continued, sampled, stats: await model.schedulerStats() }, null, 2),
);
await model.resetCaches();
console.log(
  JSON.stringify({
    cachedTokens: continued.map((result) => result.cachedTokens),
    cycles: continued.map((result) => result.performance?.mtpCycles),
  }),
);
