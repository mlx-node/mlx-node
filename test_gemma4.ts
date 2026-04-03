import { Gemma4Model } from '@mlx-node/core';

const modelPath = process.argv[2] || '.cache/models/gemma-4-e2b';
const isMoE = modelPath.includes('26b');

async function main() {
  console.log(`Loading ${isMoE ? 'MoE 26B-A4B' : 'Dense E2B'} from ${modelPath}...`);
  const t0 = Date.now();
  const model = await Gemma4Model.load(modelPath);
  console.log(`Model loaded in ${Date.now() - t0}ms\n`);

  const tests = [
    { prompt: 'Hi', expected: 'Hi' },
    { prompt: 'What is 2+2?', expected: '4' },
    { prompt: 'The capital of France is', expected: 'Paris' },
  ];

  let passed = 0;
  for (const { prompt, expected } of tests) {
    const t1 = Date.now();
    const r = await model.chat(
      [{ role: 'user', content: prompt }],
      { maxNewTokens: 64, temperature: 0.0 },
    );
    const elapsed = Date.now() - t1;
    const text = r.text.replace(/\n/g, '\\n').slice(0, 120);
    const ok = expected ? r.text.includes(expected) : true;
    const status = ok ? '✓' : '✗';
    console.log(`${status} "${prompt}" → (${r.numTokens}t, ${elapsed}ms) ${text}`);
    if (ok) passed++;
  }

  console.log(`\n${passed}/${tests.length} tests passed`);
  process.exit(passed === tests.length ? 0 : 1);
}
main().catch((e) => { console.error(e); process.exit(1); });
