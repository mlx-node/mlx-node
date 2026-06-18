import { parseArgs } from 'node:util';
import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const { values } = parseArgs({
  args: process.argv.slice(2),
  options: {
    'model-path': { type: 'string' },
    temperature: { type: 'string', default: '0' },
    'max-new': { type: 'string', default: '48' },
  },
});
const loaded = await loadModel(values['model-path']!);
const session = new ChatSession(loaded as unknown as SessionCapableModel, {
  system: 'You are a helpful assistant.',
});
const res = await session.send('Write one sentence about the ocean.', {
  config: {
    maxNewTokens: Number.parseInt(values['max-new']!, 10),
    temperature: Number.parseFloat(values.temperature!),
  },
});
const r = res as unknown as { thinking?: string; rawText?: string };
const out = (r.thinking || '') + '||' + (res.text || '') || r.rawText || '';
console.log(`TEMP=${values.temperature} TEXT:${JSON.stringify(out.slice(0, 220))}`);
