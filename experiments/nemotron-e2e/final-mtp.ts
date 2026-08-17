import { ChatSession, loadModel } from '@mlx-node/lm';

const MODEL_PATH = '/Volumes/P4510/.cache/models/nemotron-3.5-lightning-30b-a3b-nvfp4-mlx';
async function send(model: never, prompt: string, cfg: Record<string, unknown>) {
  const s = new ChatSession(model);
  const r = await s.send(prompt, { config: { temperature: 0, maxNewTokens: 256, reportPerformance: true, ...cfg } });
  const p = r.performance ?? {};
  return { r, p };
}
async function main() {
  const t0 = Date.now();
  const model = (await loadModel(MODEL_PATH)) as never;
  console.log('LOAD_TIME_S =', ((Date.now() - t0) / 1000).toFixed(1));
  for (const prompt of ['What is 2+2? Answer in one sentence.', 'What is 7+8? Answer in one sentence.']) {
    const m = await send(model, prompt, {});
    console.log('### MTP ' + JSON.stringify(prompt));
    console.log('  text  =', JSON.stringify(m.r.text));
    console.log('  toks  =', m.r.numTokens, '| finish =', m.r.finishReason);
    console.log('  cycles =', (m.p as any).mtpCycles, '| drafts/cyc =', (m.p as any).mtpMeanAcceptedTokens, '| commit/cyc =', (m.p as any).mtpMeanAcceptedTokensTotal, '| tok/s =', (m.p as any).decodeTokensPerSecond, '| ttftMs =', (m.p as any).ttftMs);
    if (prompt.startsWith('What is 2+2')) {
      const a = await send(model, prompt, { enableMtp: false });
      console.log('### AR  ' + JSON.stringify(prompt));
      console.log('  text  =', JSON.stringify(a.r.text), '| toks =', a.r.numTokens, '| finish =', a.r.finishReason, '| tok/s =', (a.p as any).decodeTokensPerSecond);
      const eq = m.r.text === a.r.text;
      console.log('  LOSSLESS_AT_T0 (MTP text == AR text) =', eq);
    }
  }
}
main().catch((e) => { console.error('FAILED:', e); process.exit(1); });