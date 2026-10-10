/** Objective content check; ASR agreement complements, but cannot replace, listening. */
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { parseArgs } from 'node:util';

import { loadTtsModel, readWav, WavWriter } from '@mlx-node/tts';

const { values } = parseArgs({
  options: {
    custom: { type: 'string' },
    base: { type: 'string' },
    output: { type: 'string' },
  },
});
if (!values.custom || !values.base || !values.output) throw new Error('--custom --base --output are required');
await mkdir(values.output, { recursive: true });
const cases = [
  {
    id: 'zh-vivian',
    voice: 'vivian',
    language: 'chinese',
    text: '你好，欢迎使用语音合成。我们正在验证实时播放和声音质量。',
  },
  {
    id: 'zh-numbers',
    voice: 'serena',
    language: 'chinese',
    text: '今天的温度是25度，下午3点15分开会。请带上两本书和一支笔。',
  },
  {
    id: 'en-ryan',
    voice: 'ryan',
    language: 'english',
    text: 'Welcome to the streaming speech test. Every word should be clear, and each sentence should finish naturally.',
  },
  {
    id: 'en-numbers',
    voice: 'aiden',
    language: 'english',
    text: 'The meeting starts at three fifteen. Please bring two books and one blue pen.',
  },
];
const records: {
  id: string;
  text: string;
  path: string;
  language: string;
  reference?: string;
  seconds: number;
  truncated: boolean;
}[] = [];
const custom = await loadTtsModel(values.custom);
try {
  for (const item of cases) {
    const path = join(values.output, `${item.id}.wav`);
    const writer = await WavWriter.open(path, custom.capabilities.sampleRate);
    try {
      const stream = custom.synthesizeStream(item.text, { voice: item.voice, language: item.language, seed: 534 });
      for await (const chunk of stream) await writer.write(chunk);
      const stats = await stream.completed;
      records.push({ ...item, path, seconds: stats.audioSeconds, truncated: stats.finishReason !== 'eos' });
      console.error(`Synthesized ${item.id}: ${stats.audioSeconds.toFixed(2)} s`);
    } finally {
      await writer.close();
    }
  }
} finally {
  await custom.dispose();
}
const base = await loadTtsModel(values.base);
try {
  for (const reference of [records[0], records[2]]) {
    const voice = await base.prepareVoice({ audio: await readWav(reference.path), transcript: reference.text });
    const text =
      reference.language === 'chinese'
        ? '这是使用参考声音生成的新句子。请确认每个字都清晰完整。'
        : 'This is a new sentence spoken with the reference voice. Please check that every word is clear and complete.';
    const id = `clone-${reference.id}`;
    const path = join(values.output, `${id}.wav`);
    const writer = await WavWriter.open(path, base.capabilities.sampleRate);
    try {
      const stream = base.synthesizeStream(text, { voice, language: reference.language, seed: 534 });
      for await (const chunk of stream) await writer.write(chunk);
      const stats = await stream.completed;
      records.push({
        id,
        text,
        path,
        reference: reference.id,
        language: reference.language,
        seconds: stats.audioSeconds,
        truncated: stats.finishReason !== 'eos',
      });
      console.error(`Synthesized ${id}: ${stats.audioSeconds.toFixed(2)} s`);
    } finally {
      await writer.close();
    }
  }
} finally {
  await base.dispose();
}
await writeFile(
  join(values.output, 'samples.json'),
  JSON.stringify({ custom: values.custom, base: values.base, cases: records }, null, 2) + '\n',
);
