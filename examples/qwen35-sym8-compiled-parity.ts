#!/usr/bin/env node
/**
 * sym8 compiled-vs-eager T=0 byte-parity probe (single-arm primitive).
 *
 * One invocation = one process = one dispatch arm:
 *   compiled arm : default env          (model_id registered -> C++ flat decode)
 *   eager arm    : MLX_QWEN35_FORCE_EAGER=1 (registration skipped -> Rust forward)
 *
 * Runs N fixed prompts at temperature 0 and prints one line
 * `PARITY_JSON:{...}` with the full generated text + sha256 per prompt.
 * A driver runs both arms and diffs the JSON byte-for-byte.
 *
 * Usage:
 *   PATH=/usr/bin:$PATH oxnode examples/qwen35-sym8-compiled-parity.ts \
 *     --model /tmp/qwen35-0.8b-sym8-mlx --max-new 120
 */
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const { values } = parseArgs({
  args: process.argv.slice(2),
  options: {
    model: { type: 'string', default: '/tmp/qwen35-0.8b-sym8-mlx' },
    'max-new': { type: 'string', default: '120' },
  },
});

const modelPath = values.model!;
const maxNew = Number.parseInt(values['max-new']!, 10);

const PROMPTS = [
  'Explain how a transformer language model generates text, step by step.',
  'Write a short story about a lighthouse keeper who discovers a mysterious map.',
  'List the planets of the solar system and give one interesting fact about each.',
];

const loaded = await loadModel(modelPath);

const results: { prompt: string; sha256: string; text: string }[] = [];
for (const prompt of PROMPTS) {
  // Fresh session per prompt — no cross-prompt cache reuse confound.
  const session = new ChatSession(loaded as unknown as SessionCapableModel, {
    system: 'You are a helpful assistant.',
  });
  const res = await session.send(prompt, {
    config: { maxNewTokens: maxNew, temperature: 0, reuseCache: false },
  });
  // rawText = the FULL raw generation incl. <think> content — the strongest
  // byte-equivalence surface (res.text can legitimately be empty when the
  // whole budget lands inside the reasoning block).
  const text = res.rawText ?? res.text ?? '';
  results.push({
    prompt,
    sha256: createHash('sha256').update(text, 'utf8').digest('hex'),
    text,
  });
}

console.log(
  `PARITY_JSON:${JSON.stringify({
    model: modelPath,
    maxNew,
    forceEager: process.env.MLX_QWEN35_FORCE_EAGER ?? '',
    results,
  })}`,
);
