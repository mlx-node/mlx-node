import { loadModel, HarrierModel } from '@mlx-node/lm';
for (const c of process.argv.slice(2)) {
  try {
    const m = await loadModel(c);
    if (m instanceof HarrierModel) { console.log(`${c.split('/').pop()}: harrier`); continue; }
    const has = typeof (m as any).hasMtpWeights === 'function' && (m as any).hasMtpWeights();
    console.log(`PROBE ${c.split('/').pop()}: hasMtpWeights=${has}`);
  } catch (e) { console.log(`PROBE ${c.split('/').pop()}: ERR ${(e as Error).message.slice(0,90)}`); }
}
