import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';
const loaded = await loadModel(process.argv[2] ?? '/tmp/qwen35-0.8b-sym8-mlx');
const session = new ChatSession(loaded as unknown as SessionCapableModel, { system: 'You are a helpful assistant.' });
const res = await session.send('Count from one to ten in words.', {
  config: { maxNewTokens: 24, temperature: 0, reuseCache: false },
});
console.log('TEXT:' + JSON.stringify(res.text));
