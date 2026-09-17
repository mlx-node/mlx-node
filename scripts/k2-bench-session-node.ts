#!/usr/bin/env oxnode

// K2-Horizon coding-agent workload benchmark (mlx-node native path).
// Replays a real Devin session shape: large system prompt + 8 turns of
// user/tool-result-sized content, multi-turn with delta prefill.
// Normal user path: ChatSession.send (no flags, no path tricks).
// Run: oxnode scripts/k2-bench-session-node.ts  (K2_MXFP8 overrides path)

import { readFileSync } from 'node:fs';
import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const MODEL = process.env.K2_MXFP8 ?? '/tmp/k2-out/k2-horizon-7b-mxfp8';
const WORKLOAD = process.env.K2_WORKLOAD ?? '/tmp/k2-workload.json';
const MAX_TOKENS = Number(process.env.K2_MAX_TOKENS ?? '200');

const workload = JSON.parse(readFileSync(WORKLOAD, 'utf8')) as {
  system: string;
  turns: string[];
};

const model = await loadModel(MODEL);
console.log(JSON.stringify({ loaded: MODEL }));
const session = new ChatSession(model as unknown as SessionCapableModel, {
  system: workload.system,
});

for (const [i, turn] of workload.turns.entries()) {
  const t0 = performance.now();
  const result = await session.send(turn, {
    config: {
      maxNewTokens: MAX_TOKENS,
      temperature: 0,
      reasoningEffort: 'low',
      reportPerformance: true,
    },
  });
  const wallMs = performance.now() - t0;
  console.log(
    JSON.stringify({
      turn: i,
      promptChars: turn.length,
      numTokens: result.numTokens,
      ttftMs: result.performance?.ttftMs ?? null,
      prefillTps: result.performance?.prefillTokensPerSecond ?? null,
      decodeTps: result.performance?.decodeTokensPerSecond ?? null,
      wallMs: Math.round(wallMs),
      text: (result.text ?? '').slice(0, 50),
    }),
  );
}
