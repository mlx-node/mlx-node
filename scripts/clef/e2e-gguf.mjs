// Plain-node E2E: ClefModel.load(<checkpoint>) then decideJson.
// Usage: node scripts/clef/e2e-gguf.mjs <checkpoint-path> <out.json>
import { createRequire } from 'node:module';
import { writeFileSync } from 'node:fs';

const require = createRequire(new URL('../../packages/core/index.cjs', import.meta.url));
const { ClefModel } = require('../../packages/core/index.cjs');

const [ckpt, out] = process.argv.slice(2);
if (!ckpt || !out) throw new Error('usage: e2e-gguf.mjs <checkpoint> <out.json>');

const request = JSON.stringify({
  state: 'The customer received a damaged parcel two weeks late and requests a replacement and a refund of shipping fees.',
  questions: {
    replace: { type: 'noul', instructions: 'Does the customer want a replacement?' },
    refund: { type: 'noul' },
    priority: {
      type: 'choice',
      instructions: 'Choose the support priority.',
      criteria: { normal: 'Routine support request', urgent: 'Immediate safety concern', vip: 'High-value account' },
    },
    sentiment: {
      type: 'score',
      instructions: 'Rate the customer sentiment.',
      criteria: ['Negative', 'Neutral', 'Positive', 'Very positive'],
    },
  },
});

const t0 = performance.now();
const model = await ClefModel.load(ckpt);
const loadMs = performance.now() - t0;

const d0 = performance.now();
const result = JSON.parse(await model.decideJson(request));
const decideMs = performance.now() - d0;

writeFileSync(out, JSON.stringify({ ckpt, loadMs, decideMs, result }, null, 2));
console.log(JSON.stringify({ ckpt, load_s: +(loadMs / 1000).toFixed(1), decide_s: +(decideMs / 1000).toFixed(3), input_tokens: result.input_tokens, questions: result.questions.length }));
