#!/usr/bin/env node
/**
 * Test Converted MLX Model
 *
 * Simple script to test generation quality with the converted float32 model.
 * Uses the new text-to-text generate() API for simplified usage.
 */

import { resolve } from 'node:path';
import { ModelLoader } from '@mlx-node/lm';

const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', 'qwen3-0.6b-mlx-bf16');

async function main() {
  console.log('╔════════════════════════════════════════════════════════╗');
  console.log('║   Testing Converted MLX Model (bf16)                   ║');
  console.log('╚════════════════════════════════════════════════════════╝\n');

  console.log(`Loading model from: ${MODEL_PATH}`);
  console.log('(Tokenizer will be loaded automatically)\n');

  // Load model (tokenizer is loaded automatically)
  const model = await ModelLoader.loadPretrained(MODEL_PATH);

  console.log('✓ Model and tokenizer loaded');
  console.log(`Config: tie_word_embeddings=${model.getConfig().tieWordEmbeddings}\n`);

  // Test prompts
  const prompts = [
    'Hello! How are you today?',
    'What is the capital of France?',
    'Write a haiku about coding:',
    'Explain what machine learning is in one sentence:',
  ];

  for (const prompt of prompts) {
    console.log('─'.repeat(60));
    console.log(`Prompt: "${prompt}"`);
    console.log('─'.repeat(60));

    // Generate using the simple message-based API
    const messages = [{ role: 'user', content: prompt }];
    const startTime = Date.now();
    const result = await model.generate(messages, {
      maxNewTokens: 2048,
      temperature: 0.7,
      topP: 0.9,
      returnLogprobs: false,
    });
    const duration = Date.now() - startTime;
    const tokensPerSecond = (result.numTokens / duration) * 1000;

    console.log(`\nGenerated (${result.numTokens} tokens, ${duration}ms, ${tokensPerSecond.toFixed(2)} tokens/s):`);
    console.log(result.text); // Text is already decoded!
    console.log('');
  }

  console.log('╔════════════════════════════════════════════════════════╗');
  console.log('║   Test Complete                                        ║');
  console.log('╚════════════════════════════════════════════════════════╝\n');
}

main().catch((error) => {
  console.error('\n❌ Test failed!');
  console.error('Error:', error.message);
  console.error('\nStack trace:', error.stack);
  process.exit(1);
});
