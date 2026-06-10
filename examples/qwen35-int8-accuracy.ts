#!/usr/bin/env node
/**
 * Accuracy smoke for the int8 W8A8 prefill paths: stream 120 greedy tokens
 * (T=0) and print the RAW accumulated text (including any reasoning chunks),
 * so we can eyeball coherence. The int8 ARM is selected by env at load time
 * (MLX_INT8_PREFILL / MLX_INT8_PREFILL_QKVZ), same as the perf harness.
 *
 * Usage:
 *   [MLX_INT8_PREFILL=1 MLX_INT8_PREFILL_QKVZ=1] [MLX_NO_COMPILE=1] \
 *     PATH=/usr/bin:$PATH oxnode examples/qwen35-int8-accuracy.ts \
 *     --model <path> --max-new 120
 *
 * Prints `ACC_JSON:{...}` with the full streamed text + a hash.
 */
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const { values } = parseArgs({
  args: process.argv.slice(2),
  options: {
    model: { type: 'string', default: '/Volumes/P4510/.cache/models/qwen3.5-4b' },
    'max-new': { type: 'string', default: '120' },
    'prompt-tokens': { type: 'string', default: '512' },
  },
});

const modelPath = values.model!;
const maxNew = Number.parseInt(values['max-new']!, 10);
const promptTokens = Number.parseInt(values['prompt-tokens']!, 10);

// A deterministic, self-contained instruction prompt (NOT from shell history).
const SENT =
  'The quick brown fox jumps over the lazy dog beside the quiet river as the evening sun slowly sets. ';
const copies = Math.max(1, Math.ceil(promptTokens / 16));
const prompt =
  `Read the following text, then write a clear factual paragraph explaining what a transformer neural network is and why attention matters.\n` +
  `${SENT.repeat(copies)}\n` +
  `Now write the explanation.`;

const toggles: Record<string, string> = {};
for (const [k, v] of Object.entries(process.env)) {
  if (k.startsWith('MLX_INT8_') || k === 'MLX_NO_COMPILE') toggles[k] = v ?? '';
}

const loaded = await loadModel(modelPath);
const session = new ChatSession(loaded as unknown as SessionCapableModel, {
  system: 'You are a helpful assistant.',
});

let acc = '';
for await (const event of session.sendStream(prompt, {
  config: { maxNewTokens: maxNew, temperature: 0, reuseCache: false },
})) {
  if (event.text) acc += event.text;
}

const hash = createHash('sha256').update(acc).digest('hex');
console.log(
  `ACC_JSON:${JSON.stringify({ model: modelPath, maxNew, toggles, len: acc.length, hash, text: acc.slice(0, 600) })}`,
);
