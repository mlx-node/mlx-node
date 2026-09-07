import { writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';

// Run capture and restore in separate processes with the same isolated
// MLX_COLD_CACHE_DIR. This checks durable target/GDN reuse with scheduled MTP.
const [binding, modelPath, output, family = 'Qwen35', phase = 'capture'] = process.argv.slice(2);
if (phase !== 'capture' && phase !== 'restore') throw new Error('Expected capture or restore');
if (!process.env.MLX_COLD_CACHE_DIR) throw new Error('Set an isolated MLX_COLD_CACHE_DIR');
const core: typeof import('../../../../packages/core/index.cjs') = createRequire(import.meta.url)(binding);
const model =
  family === 'Qwen35Moe' ? await core.Qwen35MoeModel.load(modelPath) : await core.Qwen35Model.load(modelPath);
if (!model.hasMtpWeights()) throw new Error('Missing native MTP weights');
const results = await Promise.all(
  ['computer programs', 'pancake recipes'].map((topic, i) =>
    model.chatSessionStart(
      [
        {
          role: 'user',
          content: `The topic is ${topic}.\n${Array.from(
            { length: 24 },
            (_, row) => `Note ${row + 1}: explain each action clearly and give a practical example.`,
          ).join('\n')}\nSummarize the topic in numbered steps.`,
        },
      ],
      {
        cacheOwnerId: `mtp-ssd-${i}`,
        cacheRootOwnerId: `mtp-ssd-${i}`,
        maxNewTokens: 64,
        temperature: 0,
        reasoningEffort: 'none',
        enableMtp: true,
        mtpDepth: 3,
        mtpAdaptiveDepth: false,
        maxConsecutiveTokens: 0,
        maxNgramRepeats: 0,
        reportPerformance: true,
      },
    ),
  ),
);
const drained = core.coldCacheDrain(10_000);
const cold = core.coldCacheStats();
const sidecars = core.coldSidecarStats();
const stats = await model.schedulerStats();
await writeFile(output, JSON.stringify({ binding, family, phase, results, drained, cold, sidecars, stats }, null, 2));
if (!drained || !cold.enabled || cold.writeErrors !== 0 || cold.queueDrops !== 0)
  throw new Error('SSD persistence failed');
if (stats.maxBatchOccupancy < 2) throw new Error('Requests did not exercise concurrent scheduling');
if (
  results.some((result) =>
    phase === 'capture' ? (result.performance?.mtpCycles ?? 0) === 0 : (result.performance?.mtpCycles ?? 0) !== 0,
  )
)
  throw new Error('Cold capture must speculate; cached restore must use AR without draft history');
if (phase === 'capture' && (cold.bytesWritten <= 0 || sidecars.enqueued < 2))
  throw new Error('Capture did not persist target blocks and both recurrent owners');
if (
  phase === 'restore' &&
  (results.some((result) => result.cachedTokens <= 0) || cold.bytesRestored <= 0 || sidecars.installed < 2)
)
  throw new Error('Restart did not reuse SSD target blocks and both recurrent owners');
console.log(JSON.stringify({ phase, cachedTokens: results.map((result) => result.cachedTokens), cold, sidecars }));
await model.resetCaches();
