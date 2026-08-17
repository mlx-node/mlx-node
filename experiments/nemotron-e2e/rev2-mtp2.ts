import { ChatSession, loadModel } from '@mlx-node/lm';

const MODEL_PATH = '/Volumes/P4510/.cache/models/nemotron-3.5-lightning-30b-a3b-nvfp4-mlx';
async function mtp(model: never, prompt: string) {
  const s = new ChatSession(model);
  const r = await s.send(prompt, { config: { temperature: 0, maxNewTokens: 256, reportPerformance: true } });
  const p = r.performance ?? {};
  console.log('MTP', JSON.stringify(prompt), '->', JSON.stringify(r.text), '| toks:', r.numTokens, r.finishReason, '| cycles:', (p as any).mtpCycles, '| drafts/cyc:', (p as any).mtpMeanAcceptedTokens, '| commit/cyc:', (p as any).mtpMeanAcceptedTokensTotal, '| tok/s:', (p as any).decodeTokensPerSecond);
}
async function ar(model: never, prompt: string) {
  const s = new ChatSession(model);
  const r = await s.send(prompt, { config: { temperature: 0, maxNewTokens: 256, reportPerformance: true, enableMtp: false } });
  const p = r.performance ?? {};
  console.log('AR ', JSON.stringify(prompt), '->', JSON.stringify(r.text), '| toks:', r.numTokens, r.finishReason, '| tok/s:', (p as any).decodeTokensPerSecond);
}
async function main() {
  const t0 = Date.now();
  const model = (await loadModel(MODEL_PATH)) as never;
  console.log('LOAD_TIME_S =', ((Date.now() - t0) / 1000).toFixed(1));
  await mtp(model, 'What is 2+2? Answer in one sentence.');
  await mtp(model, 'What is 2+2? Answer in one sentence.');
  await mtp(model, 'How many sides does a triangle have?');
  await ar(model, 'What is 2+2? Answer in one sentence.');
}
main().catch((e) => { console.error('FAILED:', e); process.exit(1); });