import { expect, it } from 'vite-plus/test';

import { decodeTtsInput } from '../../packages/cli/src/commands/tts-input.js';
async function* bytes(text: string, step = 1) {
  const data = Buffer.from(text);
  for (let i = 0; i < data.length; i += step) yield data.subarray(i, i + step);
}
async function collect(input: AsyncIterable<unknown>) {
  const values = [];
  for await (const value of input) values.push(value);
  return values;
}
it('decodes split UTF-8 text without interpreting control syntax', async () => {
  const text = '你好🙂 {"type":"instruct"}';
  expect((await collect(decodeTtsInput(bytes(text), 'text'))).join('')).toBe(text);
});
it('decodes JSONL events across arbitrary byte boundaries, including the final record', async () => {
  const expected = [
    { type: 'text', text: '你好🙂' },
    { type: 'instruct', value: 'calm' },
    { type: 'flush' },
    { type: 'instruct', value: null },
  ];
  for (const step of [1, 2, 7, 64])
    expect(
      await collect(decodeTtsInput(bytes(expected.map((v) => JSON.stringify(v)).join('\r\n'), step), 'jsonl')),
    ).toEqual(expected);
});
it('reports the failing line and rejects unbounded or invalid records', async () => {
  await expect(collect(decodeTtsInput(bytes('\n{"type":"text","text":42}'), 'jsonl'))).rejects.toThrow('line 2');
  for (const tail of ['', '\n'])
    await expect(collect(decodeTtsInput(bytes('x'.repeat(65) + tail, 100), 'jsonl', 64))).rejects.toThrow('byte limit');
  await expect(collect(decodeTtsInput(bytes('{invalid}\n'), 'jsonl'))).rejects.toThrow('line 1');
});
it('opens files lazily and routes missing-file errors through the iterator', async () => {
  const { ttsFileBytes } = await import('../../packages/cli/src/commands/tts-input.js');
  const input = ttsFileBytes('/tmp/mlx-node-missing-tts-input-file');
  await new Promise((resolve) => setTimeout(resolve, 20));
  await expect(collect(decodeTtsInput(input, 'text'))).rejects.toThrow('ENOENT');
});
