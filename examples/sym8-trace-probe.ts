import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';
const loaded = await loadModel(process.argv[2] ?? '/tmp/qwen35-0.8b-sym8-mlx');
const session = new ChatSession(loaded as unknown as SessionCapableModel, { system: 'You are a helpful assistant.' });
const res = await session.send('Explain how a transformer language model generates text, step by step.', {
  config: { maxNewTokens: 120, temperature: 0, reuseCache: false },
});
console.log('RAW:' + JSON.stringify((res as any).rawText));
