import { resolve } from 'node:path';
import { Qwen35Model } from '@mlx-node/core';

const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', 'qwen3.5-27b-mlx-bf16');

async function main() {
  console.log('Loading Qwen3.5 model...');
  const startLoad = Date.now();
  const model = await Qwen35Model.loadPretrained(MODEL_PATH);
  console.log(`✓ Model loaded in ${Date.now() - startLoad}ms`);

  // Warmup — compiles Metal shaders
  console.log('\n--- Warmup ---');
  let start = Date.now();
  let result = await model.chat([{ role: 'user', content: 'Hi' }], { maxNewTokens: 1 });
  console.log(`  Time: ${Date.now() - start}ms (includes shader compilation)`);

  // Test 1: 1 token — pure prefill (warmed up)
  console.log('\n--- Test 1: 1 token (warmed up) ---');
  start = Date.now();
  result = await model.chat([{ role: 'user', content: 'Hi' }], { maxNewTokens: 1 });
  console.log(`  Prefill: ${Date.now() - start}ms`);

  // Test 2: 20 tokens
  console.log('\n--- Test 2: 20 tokens ---');
  start = Date.now();
  result = await model.chat([{ role: 'user', content: 'Hi' }], { maxNewTokens: 20 });
  const elapsed = Date.now() - start;
  const decodeTime = elapsed - 4000; // subtract ~prefill estimate
  const tps = result.numTokens > 1 ? ((result.numTokens - 1) / (decodeTime / 1000)) : 0;
  console.log(`  Total: ${elapsed}ms, tokens: ${result.numTokens}`);
  console.log(`  Text: "${result.text}"`);

  // Test 3: 50 tokens — better decode measurement
  console.log('\n--- Test 3: 50 tokens ---');
  start = Date.now();
  result = await model.chat([{ role: 'user', content: 'Count from 1 to 50' }], { maxNewTokens: 50 });
  const elapsed3 = Date.now() - start;
  console.log(`  Total: ${elapsed3}ms, tokens: ${result.numTokens}`);
  console.log(`  Overall: ${((result.numTokens / elapsed3) * 1000).toFixed(1)} tok/s`);
  console.log(`  Text: "${result.text.slice(0, 100)}..."`);
}

main().catch(e => {
  console.error('Error:', e.message);
  process.exit(1);
});
