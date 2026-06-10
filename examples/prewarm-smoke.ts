#!/usr/bin/env node
// Warm smoke test for the cold-mmap prewarm extension.
// Confirms: the "Pre-warmed N checkpoint shard(s)" log fires (run with
// MLX_NODE_LOG=mlx_core=info), load completes, and output is coherent.
// qwen3.5-0.8b exercises the dense `_with` sidecar-candidate call path.
import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const MODELS: Array<{ family: string; path: string }> = [
  { family: 'qwen3.5-dense', path: '/Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16' },
  { family: 'qwen3', path: '/Volumes/P4510/models/qwen3-0.6b-mlx-bf16' },
];

for (const { family, path } of MODELS) {
  process.stdout.write(`\n=== ${family} :: ${path} ===\n`);
  const t0 = Date.now();
  const loaded = await loadModel(path);
  const loadMs = Date.now() - t0;
  const session = new ChatSession(loaded as unknown as SessionCapableModel, {
    system: 'You are a helpful assistant. Answer directly and briefly.',
  });
  const res = await session.send('What is the capital of France? Reply with just the city name.', {
    config: { maxNewTokens: 96, temperature: 0, reportPerformance: true, reuseCache: false },
  });
  process.stdout.write(`loadMs=${loadMs} promptTok=${res.promptTokens} text=${JSON.stringify(res.text)}\n`);
}
process.stdout.write('\nALL DONE\n');
