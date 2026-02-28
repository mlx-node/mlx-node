import { resolve } from 'node:path';
import { Qwen35Model } from '@mlx-node/core';

const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', 'qwen3.5-27b-mlx-bf16');

async function main() {
  console.log(`Loading model from: ${MODEL_PATH}`);
  const model = await Qwen35Model.loadPretrained(MODEL_PATH);
  console.log('Model loaded\n');

  const prompt = 'Count from 1 to 100, one number per line.';

  // Warmup — compiles Metal shaders
  console.log('Warming up...');
  await model.chat([{ role: 'user', content: 'Hi' }], { maxNewTokens: 5, temperature: 0.7, topP: 0.9 });

  // Benchmark: 100 tokens
  console.log('\nGenerating 100 tokens...');
  const start = Date.now();
  const result = await model.chat(
    [{ role: 'user', content: prompt }],
    { maxNewTokens: 100, temperature: 0.7, topP: 0.9 }
  );
  const elapsed = (Date.now() - start) / 1000;

  const tps = result.numTokens / elapsed;
  console.log(`Tokens: ${result.numTokens}`);
  console.log(`Time: ${elapsed.toFixed(2)}s`);
  console.log(`Speed: ${tps.toFixed(1)} tok/s`);
  console.log(`Text: ${(result.text || '').slice(0, 200)}...`);
}

main().catch(e => {
  console.error('Error:', e.message);
  process.exit(1);
});
