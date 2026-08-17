import { ChatSession, loadModel } from '@mlx-node/lm';

const MODEL_PATH = '/Volumes/P4510/.cache/models/nemotron-3.5-lightning-30b-a3b-nvfp4-mlx';
async function main() {
  const model = (await loadModel(MODEL_PATH)) as never;
  const sM = new ChatSession(model);
  const m = await sM.send('What is 2+2? Answer in one sentence.', { config: { temperature: 0, maxNewTokens: 256, reportPerformance: true } });
  const p = m.performance ?? {};
  console.log('MTP text    =', JSON.stringify(m.text));
  console.log('MTP finish  =', m.finishReason, '| toks =', m.numTokens);
  console.log('MTP cycles  =', (p as any).mtpCycles, '| drafts/cyc =', (p as any).mtpMeanAcceptedTokens, '| commit/cyc =', (p as any).mtpMeanAcceptedTokensTotal, '| tok/s =', (p as any).decodeTokensPerSecond);
  const sA = new ChatSession(model);
  const a = await sA.send('What is 2+2? Answer in one sentence.', { config: { temperature: 0, maxNewTokens: 256, reportPerformance: true, enableMtp: false } });
  console.log('AR  text    =', JSON.stringify(a.text), '| finish =', a.finishReason, '| toks =', a.numTokens);
  console.log('LOSSLESS (MTP==AR) =', m.text === a.text);
}
main().catch((e) => { console.error('FAILED:', e); process.exit(1); });