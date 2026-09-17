#!/usr/bin/env oxnode

// K2-Horizon perf harness (mlx-node native path, mxfp8 converted checkpoint).
// Prints one JSON line per run: prefill TPS, decode TPS, ttft.
// Run: oxnode scripts/k2-perf-node.ts   (K2_MXFP8 env overrides the path)

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const MODEL = process.env.K2_MXFP8 ?? '/tmp/k2-out/k2-horizon-7b-mxfp8';
const MAX_TOKENS = Number(process.env.K2_MAX_TOKENS ?? '64');

const SHORT = 'What is 17 * 23? Give the final number.';
const LONG =
  'You are a careful math tutor. Solve step by step, then give the final answer. ' +
  'Here is the problem context: ' +
  Array.from({ length: 120 }, (_, i) => `term${i}=${(i * 7) % 13}`).join(' ') +
  ' Compute the sum of all terms modulo 97, then multiply by 11.';

async function run(session: ChatSession<SessionCapableModel>, prompt: string, label: string) {
  const result = await session.send(prompt, {
    config: {
      maxNewTokens: MAX_TOKENS,
      temperature: 0,
      reasoningEffort: 'low',
      reportPerformance: true,
    },
  });
  console.log(
    JSON.stringify({
      label,
      numTokens: result.numTokens,
      reasoningTokens: result.reasoningTokens,
      ttftMs: result.performance?.ttftMs ?? null,
      prefillTps: result.performance?.prefillTokensPerSecond ?? null,
      decodeTps: result.performance?.decodeTokensPerSecond ?? null,
      text: (result.text ?? '').slice(0, 60),
    }),
  );
}

const model = await loadModel(MODEL);
console.log(JSON.stringify({ loaded: MODEL }));
const session = new ChatSession(model as unknown as SessionCapableModel);

// Warmup — first turn pays template + compile overhead, same as mlx side.
await run(session, 'hi', 'warmup');
await run(session, SHORT, 'short');
await run(session, LONG, 'long');
await run(session, SHORT, 'short2');
