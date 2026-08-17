import { ChatSession, loadModel } from '@mlx-node/lm';

async function main() {
  const m = (await loadModel('/Volumes/P4510/.cache/models/nemotron-3.5-lightning-30b-a3b-nvfp4-mlx')) as never;
  const sM = new ChatSession(m);
  const rM = await sM.send('What is 2+2? Answer in one sentence.', { config: { temperature: 0, maxNewTokens: 256, reportPerformance: true } });
  console.log('PAGED-MTP text    =', JSON.stringify(rM.text));
  console.log('PAGED-MTP rawHead =', JSON.stringify(rM.rawText.slice(0, 160)));
  const f = (await loadModel('/Volumes/P4510/.cache/models/nemotron-3.5-lightning-30b-a3b-nvfp4-mlx-flat')) as never;
  const sF = new ChatSession(f);
  const rF = await sF.send('What is 2+2? Answer in one sentence.', { config: { temperature: 0, maxNewTokens: 256, reportPerformance: true, enableMtp: false } });
  console.log('FLAT-AR   text    =', JSON.stringify(rF.text));
  console.log('FLAT-AR   rawHead =', JSON.stringify(rF.rawText.slice(0, 160)));
  console.log('MTP == FLAT-AR rawText:', rM.rawText === rF.rawText);
}
main().catch((e) => { console.error('FAILED:', e); process.exit(1); });