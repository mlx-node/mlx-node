/** Native instruction samples and isolated cold/hot prefix-cache measurements. */
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { loadTtsModel, WavWriter } from '@mlx-node/tts';
import type { TtsVoice } from '@mlx-node/tts';

import { readModelProvenance } from './model-provenance.js';
const { values } = parseArgs({
  options: { model: { type: 'string' }, output: { type: 'string' }, cache: { type: 'boolean' } },
});
if (!values.model || !values.output) throw new Error('--model and --output required');
const output = resolve(values.output);
await mkdir(output, { recursive: true });
const metadata = JSON.parse(await readFile(join(values.model, 'config.json'), 'utf8'));
const revision = await readModelProvenance(values.model);
const provenance = {
  revision,
  modelPath: resolve(values.model),
  precision: { dtype: metadata.dtype ?? metadata.torch_dtype, quantization: metadata.quantization ?? null },
};
const cases: unknown[] = [];
const conditions = [
  {
    id: 'zh',
    language: 'chinese',
    text: '现在是下午三点，请到车站等候。',
    preset: 'vivian',
    description: '成年女性，音色清晰、沉稳，普通话发音自然。',
  },
  {
    id: 'en',
    language: 'english',
    text: 'The train arrives at three. Please wait at the station.',
    preset: 'ryan',
    description: 'An adult male with a clear, warm voice and a natural American English accent.',
  },
];
const styles = [
  undefined,
  'Speak calmly and evenly.',
  'Speak with strong excitement and enthusiasm.',
  'Speak very softly, in a whisper.',
];
if (!values.cache) {
  const start = performance.now();
  const model = await loadTtsModel(values.model);
  const loadMs = performance.now() - start;
  try {
    for (const condition of conditions)
      for (let style = 0; style < styles.length; style++)
        for (const seed of [534, 535]) {
          const voice: TtsVoice =
            metadata.tts_model_type === 'voice_design'
              ? { type: 'description', description: condition.description }
              : condition.preset;
          const id = `${condition.id}-${style}-${seed}`;
          const path = join(output, `${id}.wav`);
          const writer = await WavWriter.open(path, model.capabilities.sampleRate);
          try {
            const stream = model.synthesizeStream(condition.text, {
              voice,
              instruct: styles[style],
              language: condition.language,
              seed,
              maxDurationSeconds: 30,
            });
            for await (const chunk of stream) await writer.write(chunk);
            const stats = await stream.completed;
            cases.push({
              id,
              path,
              text: condition.text,
              language: condition.language,
              voice,
              instruct: styles[style] ?? null,
              seed,
              stats,
            });
            await writeFile(join(output, 'samples.json'), JSON.stringify({ ...provenance, loadMs, cases }, null, 2));
            console.error(id, stats.audioSeconds.toFixed(2), stats.realTimeFactor?.toFixed(3));
          } finally {
            await writer.close();
          }
        }
  } finally {
    await model.dispose();
  }
} else {
  for (const [round, enabled] of [false, true, false, true].entries()) {
    const model = await loadTtsModel(values.model, { instructionCache: { enabled } });
    try {
      const voice: TtsVoice =
        metadata.tts_model_type === 'voice_design'
          ? { type: 'description', description: conditions[0].description }
          : 'vivian';
      for (let run = 0; run < 8; run++) {
        const stream = model.synthesizeStream('你好，欢迎回来。', {
          voice,
          instruct: 'Use a clear, calm and steady delivery. Speak naturally, with a small pause at punctuation.',
          language: 'chinese',
          temperature: 0,
          seed: 534,
          maxDurationSeconds: 15,
        });
        for await (const _ of stream) {
          /* Fully materialized PCM, no I/O in the timing path. */
        }
        const stats = await stream.completed;
        cases.push({ round, enabled, run, stats });
        console.error(enabled, run, stats.firstPcmMs, stats.realTimeFactor);
      }
    } finally {
      await model.dispose();
    }
  }
  await writeFile(join(output, 'cache.json'), JSON.stringify({ ...provenance, cases }, null, 2));
}
