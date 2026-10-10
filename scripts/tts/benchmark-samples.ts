/** Extract exact segment WAVs and expected texts for the development ASR oracle. */
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { readWav, WavWriter } from '@mlx-node/tts';
import type { TtsInputEvent } from '@mlx-node/tts';

import { TextSegmenter } from '../../packages/tts/src/segmenter.js';
import { paragraphInstructions } from './instruction-input.js';

const { values } = parseArgs({
  options: { report: { type: 'string' }, audio: { type: 'string' }, output: { type: 'string' } },
});
if (!values.report || !values.audio || !values.output) throw new Error('--report, --audio and --output required');
const report = JSON.parse(await readFile(values.report, 'utf8'));
const audio = await readWav(values.audio);
const channels = audio.channels ?? 1;
const output = resolve(values.output);
await mkdir(output, { recursive: true });
const expected: { text: string; instruct?: string }[] = [];
const segmenter = new TextSegmenter(report.parameters.maxSegmentGraphemes);
let current: string | undefined = report.parameters.instruct;
function submit(text: string, flush = false) {
  for (const part of segmenter.push(text, flush)) if (part.trim()) expected.push({ text: part, instruct: current });
}
async function events(source: AsyncIterable<TtsInputEvent>) {
  for await (const event of source) {
    if (typeof event === 'string') submit(event);
    else if (event.type === 'text') submit(event.text);
    else if (event.type === 'flush') submit('', true);
    else {
      const next = event.value?.trim() ? event.value : undefined;
      if (next !== current) {
        submit('', true);
        current = next;
      }
    }
  }
  submit('', true);
}
if (report.input.mode === 'simulated-llm-clock') {
  assert.ok(Number.isSafeInteger(report.input.chunkGraphemes) && report.input.chunkGraphemes > 0);
  const parts = [...new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(report.input.text)].map(
    (s) => s.segment,
  );
  async function* source() {
    for (let i = 0; i < parts.length; i += report.input.chunkGraphemes)
      yield parts.slice(i, i + report.input.chunkGraphemes).join('');
  }
  await events(paragraphInstructions(source(), report.input.text, report.parameters.instructions));
} else {
  assert.ok(['complete-text-once', 'repeated-text'].includes(report.input.mode));
  assert.ok(report.input.text.trim().length > 0);
  async function* source(): AsyncGenerator<TtsInputEvent> {
    let iteration = 0;
    do {
      if (report.parameters.instructions.length)
        yield {
          type: 'instruct',
          value: report.parameters.instructions[iteration++ % report.parameters.instructions.length],
        };
      yield report.input.text;
      yield { type: 'flush' };
    } while (report.input.mode === 'repeated-text' && expected.length < report.segments.length);
  }
  await events(source());
}
assert.equal(expected.length, report.segments.length, 'Expected text and recorded audio segment counts differ');
const cases = [];
let offset = 0;
for (const [index, segment] of report.segments.entries()) {
  assert.equal(segment.segmentIndex, index);
  assert.equal(segment.startSample, offset, 'Audio gap or overlap');
  offset = segment.endSample;
  const path = join(output, `${index}.wav`);
  const writer = await WavWriter.open(path, audio.sampleRate, channels);
  try {
    await writer.write({
      samples: audio.samples.slice(segment.startSample * channels, segment.endSample * channels),
      sampleRate: audio.sampleRate,
      channels,
      startSample: 0,
      segmentIndex: 0,
    });
  } finally {
    await writer.close();
  }
  cases.push({ id: String(index), path, language: report.parameters.language, ...expected[index] });
}
assert.equal(offset * channels, audio.samples.length, 'Unaccounted WAV samples');
await writeFile(
  join(output, 'samples.json'),
  JSON.stringify({ report: resolve(values.report), checkpoint: report.checkpoint, cases }, null, 2),
);
console.log(`Extracted ${cases.length} contiguous segments`);
