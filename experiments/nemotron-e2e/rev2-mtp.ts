import { ChatSession, loadModel } from '@mlx-node/lm';

const MODEL_PATH = '/Volumes/P4510/.cache/models/nemotron-3.5-lightning-30b-a3b-nvfp4-mlx';

async function main() {
  const t0 = Date.now();
  const model = (await loadModel(MODEL_PATH)) as never;
  console.log('LOAD_TIME_S =', ((Date.now() - t0) / 1000).toFixed(1));

  for (const prompt of ['What is 2+2? Answer in one sentence.', 'What is 7+8? Answer in one sentence.']) {
    const s = new ChatSession(model);
    const r = await s.send(prompt, { config: { temperature: 0, maxNewTokens: 256, reportPerformance: true } });
    const p = r.performance ?? {};
    console.log('### ' + JSON.stringify(prompt));
    console.log('  text       =', JSON.stringify(r.text));
    console.log('  toks       =', r.numTokens, '| finish =', r.finishReason);
    console.log('  mtpCycles  =', (p as any).mtpCycles);
    console.log('  mtpMeanAcceptedTokens     =', (p as any).mtpMeanAcceptedTokens);
    console.log('  mtpMeanAcceptedTokensTotal =', (p as any).mtpMeanAcceptedTokensTotal);
    console.log('  mtpAcceptanceByPosition =', JSON.stringify((p as any).mtpAcceptanceByPosition));
    console.log('  mtpMeanDepth =', (p as any).mtpMeanDepth);
    console.log('  decodeTok/s =', (p as any).decodeTokensPerSecond);
    console.log('  ttftMs      =', (p as any).ttftMs);
  }
}
main().catch((e) => { console.error('FAILED:', e); process.exit(1); });
