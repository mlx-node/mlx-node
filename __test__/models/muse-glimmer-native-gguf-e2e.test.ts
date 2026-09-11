import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { ChatSession, MuseGlimmerModel, type ChatMessage, type ToolDefinition } from '@mlx-node/lm';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

// Opt in with an original GGUF file and its config/tokenizer assets. Restore
// the real review fixture with `oxnode scripts/benchmark-fixture.ts fetch`.
// Historical tool results are replayed as messages; no tools are executed.
const modelPath = process.env.MUSE_GLIMMER_GGUF_MODEL_PATH;
const fixturePath = '.cache/benchmarks/gemma4-agent-2026-09-10/inputs.json';

describe.skipIf(!modelPath)('Muse-Glimmer native GGUF loading and inference', () => {
  let model: MuseGlimmerModel;
  let session: ChatSession;

  beforeAll(async () => {
    model = await MuseGlimmerModel.load(modelPath!);
    session = new ChatSession(model);
  }, 300_000);

  afterAll(async () => {
    await session?.dispose();
  });

  it('renders the real review fixture from cached assets and generates with packed weights', async () => {
    expect(JSON.parse(readFileSync(join(model.modelAssetsPath(), 'config.json'), 'utf8'))).toHaveProperty(
      'quantization',
    );
    const [fixture] = JSON.parse(readFileSync(fixturePath, 'utf8')) as Array<{
      messages: ChatMessage[];
      tools: ToolDefinition[];
    }>;
    for (const enableMtp of model.hasMtpWeights() ? [false, true] : [false]) {
      await session.reset();
      session.primeHistory(fixture!.messages);
      const result = await session.startFromHistory({
        tools: fixture!.tools,
        maxNewTokens: 32,
        temperature: 0,
        reasoningEffort: 'high',
        enableMtp,
        reportPerformance: true,
      });
      expect(result.promptTokens).toBeGreaterThan(1000);
      expect(result.numTokens).toBeGreaterThan(0);
      expect(result.rawText.length).toBeGreaterThan(0);
      if (enableMtp) expect(result.performance?.mtpCycles).toBeGreaterThan(0);
    }
  }, 300_000);
});
