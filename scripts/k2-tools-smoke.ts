#!/usr/bin/env oxnode

import { ChatSession, loadModel, type ChatStreamEvent, type SessionCapableModel } from '@mlx-node/lm';

const SRC = process.env.K2_SRC ?? '/Users/brooklyn/workspace/github/mlx-node/.cache/models/k2-horizon-7b-fp8';

const model = await loadModel(SRC);
const session = new ChatSession(model as unknown as SessionCapableModel);

// --- multi-turn continuation (history replay w/ reasoningContent) ---
const r1 = await session.send('What is 6 * 7? Final number only.', {
  config: { maxNewTokens: 128, temperature: 0, reasoningEffort: 'low' },
});
console.log(JSON.stringify({ turn1: r1.text.slice(0, 120), finish: r1.finishReason }));
const r2 = await session.send('And what is that number plus 5?', {
  config: { maxNewTokens: 128, temperature: 0, reasoningEffort: 'low' },
});
console.log(JSON.stringify({ turn2: r2.text.slice(0, 120), has47: (r2.text + (r2.rawText ?? '')).includes('47') }));

// --- tool call ---
const session2 = new ChatSession(model as unknown as SessionCapableModel, {
  defaultConfig: {
    tools: [
      {
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get the weather for a city',
          parameters: {
            type: 'object',
            properties: JSON.stringify({ city: { type: 'string' } }),
            required: ['city'],
          },
        },
      },
    ],
  },
});
const r3 = await session2.send('What is the weather in Paris? Use the tool.', {
  config: { maxNewTokens: 256, temperature: 0, reasoningEffort: 'low' },
});
console.log(
  JSON.stringify({
    toolTurn: r3.text.slice(0, 200),
    finish: r3.finishReason,
    toolCalls: r3.toolCalls,
    raw: (r3.rawText ?? '').slice(0, 400),
  }),
);

// --- streaming ---
const session3 = new ChatSession(model as unknown as SessionCapableModel);
const events: ChatStreamEvent[] = [];
for await (const ev of session3.sendStream('What is 9 * 9? Final number only.', {
  config: { maxNewTokens: 128, temperature: 0, reasoningEffort: 'low' },
})) {
  events.push(ev);
}
const deltas = events.filter((e) => e.done === false);
const finalEv = events.find((e) => e.done === true);
const streamedText = deltas.map((d) => String(d.text ?? '')).join('');
console.log(
  JSON.stringify({
    streamEvents: events.length,
    deltaKinds: deltas.slice(0, 30).map((d) => ({ r: d.isReasoning, t: String(d.text ?? '').slice(0, 24) })),
    finalText: String(finalEv?.text ?? '').slice(0, 200),
    has81: (streamedText + String(finalEv?.text ?? '')).includes('81'),
  }),
);
