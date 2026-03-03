#!/usr/bin/env node
/**
 * Test Converted MLX Model
 *
 * Simple script to test generation quality with the converted float32 model.
 * Auto-detects model type (Qwen3 vs Qwen3.5) from config.json.
 */

import { resolve } from 'node:path';
import { ModelLoader } from '@mlx-node/lm';
import { Qwen3Model } from '@mlx-node/core';

const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', process.argv[2] || 'qwen3.5-27b-mxfp8');

async function main() {
  console.log('╔════════════════════════════════════════════════════════╗');
  console.log('║   Testing Converted MLX Model (bf16)                   ║');
  console.log('╚════════════════════════════════════════════════════════╝\n');

  console.log(`Loading model from: ${MODEL_PATH}`);
  console.log('(Tokenizer will be loaded automatically)\n');

  // Load model — auto-detects Qwen3 vs Qwen3.5 from config.json
  const model = await ModelLoader.loadPretrained(MODEL_PATH);
  const isQwen3 = model instanceof Qwen3Model;

  console.log(`✓ Model loaded (${isQwen3 ? 'Qwen3' : 'Qwen3.5'})`);
  if (isQwen3) {
    console.log(`Config: tie_word_embeddings=${model.getConfig().tieWordEmbeddings}\n`);
  }

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

    const messages = [{ role: 'user', content: prompt }];
    const startTime = Date.now();

    let text: string;
    let numTokens: number;

    if (isQwen3) {
      const result = await model.generate(messages, {
        maxNewTokens: 2048,
        temperature: 0.7,
        topP: 0.9,
        returnLogprobs: false,
      });
      text = result.text;
      numTokens = result.numTokens;
    } else {
      // Qwen3.5 uses chat() API
      const result = await model.chat(messages, {
        maxNewTokens: 2048,
        temperature: 0.7,
        topP: 0.9,
      });
      text = result.text;
      numTokens = result.numTokens;
    }

    const duration = Date.now() - startTime;
    const tokensPerSecond = (numTokens / duration) * 1000;

    console.log(`\nGenerated (${numTokens} tokens, ${duration}ms, ${tokensPerSecond.toFixed(2)} tokens/s):`);
    console.log(text);
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
