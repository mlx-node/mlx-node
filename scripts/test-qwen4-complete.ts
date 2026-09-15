/** Full-checkpoint validation. Always invoke through guard-model-memory.py. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { deflateSync } from 'node:zlib';

import { memoryStats } from '@mlx-node/core';

import { ChatSession } from '../packages/lm/src/chat-session.js';
import { loadModel } from '../packages/lm/src/models/model-loader.js';
import { Qwen4ExpModel } from '../packages/lm/src/stream.js';

const [target, auxiliary] = process.argv.slice(2);
assert(target, 'Pass the first GGUF split and optionally the matching original HF directory.');
const output = process.env.QWEN4_SMOKE_OUTPUT ?? '.cache/qwen4-complete-e2e.json';
const mode = process.env.QWEN4_SMOKE_MODE ?? 'complete';
assert(['basic', 'complete', 'media'].includes(mode), 'Unknown smoke mode.');
const records: unknown[] = [];
const started = performance.now();
async function record(value: object) {
  records.push(value);
  console.log(JSON.stringify(value));
  await writeFile(output, JSON.stringify({ target, auxiliary, mode, records }, null, 2));
}
const model = await loadModel(target, {
  autoLoadDraft: false,
  auxiliaryModelPath: auxiliary,
});
assert(model instanceof Qwen4ExpModel);
assert(model.hasBlockPagedCache());
assert.equal(model.hasMtpWeights(), Boolean(auxiliary));
assert.equal(model.mtpAutoEnabled(), false);
assert(model.contextLimits().pagedBlockCapacity > 0);
const residency = model.residencyInfo();
// Bootstrap may admit a different cache on every machine/run. Check MLX's
// separately bounded working allowance against that actual weight budget;
// the outer guard independently enforces system pressure and process RSS.
const peakMemoryLimit = residency.weightBudgetBytes + 4 * 1024 ** 3;
await record({
  event: 'loaded',
  seconds: (performance.now() - started) / 1000,
  memory: memoryStats(),
  limits: model.contextLimits(),
  hasMtp: model.hasMtpWeights(),
  residency,
});
const makeSession = (owner: string, mtp = false, length = 8) =>
  new ChatSession(model, {
    defaultConfig: {
      cacheOwnerId: owner,
      temperature: 0,
      maxNewTokens: length,
      reasoningEffort: 'none',
      enableMtp: mtp,
      mtpDepth: 3,
      mtpAdaptiveDepth: false,
      reuseCache: true,
      reportPerformance: true,
    },
  });
async function timed(name: string, fn: () => Promise<unknown>) {
  const start = performance.now();
  const result = await fn();
  await record({
    event: 'turn',
    name,
    seconds: (performance.now() - start) / 1000,
    result,
    outputSha256: createHash('sha256')
      .update(JSON.stringify(Array.isArray(result) ? result.map((r) => r.text) : (result as { text?: string }).text))
      .digest('hex'),
    memory: memoryStats(),
  });
  assert(memoryStats().peak < peakMemoryLimit);
  return result;
}
if (mode !== 'basic') {
  assert(auxiliary, 'MTP/media validation requires the matching auxiliary checkpoint.');
}
if (mode !== 'media') {
  const capitals = makeSession('qwen4-capitals');
  const paris = await timed('Paris AR', () =>
    capitals.send('What is the capital of France? Reply with only the city name.'),
  );
  assert.match((paris as { text: string }).text, /Paris/i);
  let streamedText = '';
  const tokyo = await timed('Tokyo streaming continuation', async () => {
    let final;
    for await (const event of capitals.sendStream('And Japan? Reply with only the city name.')) {
      if (event.done) final = event;
      else streamedText += event.text;
    }
    assert(final);
    return final;
  });
  assert.equal(streamedText, (tokyo as { text: string }).text);
  assert.match(streamedText, /Tokyo/i);
  assert(((tokyo as { cachedTokens?: number }).cachedTokens ?? 0) > 0);
  if (mode === 'basic') {
    await record({
      event: 'passed',
      seconds: (performance.now() - started) / 1000,
      stats: await model.schedulerStats(),
    });
    process.exit(0);
  }
  await model.resetCaches();
  const numbers = 'Write the integers from 1 to 12, separated by commas. Output nothing else.';
  const ar = await timed('32-token AR control', () => makeSession('qwen4-numbers-ar', false, 32).send(numbers));
  const mtp = await timed('32-token native MTP', () => makeSession('qwen4-numbers-mtp', true, 32).send(numbers));
  assert.equal((ar as { text: string }).text, (mtp as { text: string }).text, 'Greedy verified output must match AR.');
  const mtpPerf = (mtp as { performance?: { mtpCycles?: number } }).performance;
  assert((mtpPerf?.mtpCycles ?? 0) > 0, 'Must execute actual draft/verify cycles.');
  const adaptiveSession = makeSession('qwen4-numbers-adaptive', true, 32);
  let adaptiveText = '';
  const adaptive = await timed('32-token adaptive native MTP stream', async () => {
    let final;
    for await (const event of adaptiveSession.sendStream(numbers, {
      config: { mtpAdaptiveDepth: true },
    })) {
      if (event.done) final = event;
      else adaptiveText += event.text;
    }
    assert(final);
    return final;
  });
  assert.equal(adaptiveText, (adaptive as { text: string }).text);
  assert.equal(adaptiveText, (ar as { text: string }).text);
  assert(((adaptive as { performance?: { mtpCycles?: number } }).performance?.mtpCycles ?? 0) > 0);
  await model.resetCaches();
  const a = makeSession('qwen4-owner-maple');
  const b = makeSession('qwen4-owner-orchid');
  await timed('concurrent owner prefill', () =>
    Promise.all([
      a.send('My secret word is maple. Reply only with my word.'),
      b.send('My secret word is orchid. Reply only with my word.'),
    ]),
  );
  const answers = await timed('concurrent owner continuation', () =>
    Promise.all([
      a.send('What is my secret word? Reply only with the word.'),
      b.send('What is my secret word? Reply only with the word.'),
    ]),
  );
  assert.match((answers as Array<{ text: string }>)[0].text, /maple/i);
  assert.match((answers as Array<{ text: string }>)[1].text, /orchid/i);
  await model.resetCaches();
  const cancelled = makeSession('qwen4-cancel');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 150);
  try {
    await assert.rejects(
      cancelled.send('Explain this sequence: ' + Array.from({ length: 100 }, (_, i) => String(i)).join(' '), {
        signal: controller.signal,
      }),
      /cancel|abort/i,
    );
  } finally {
    clearTimeout(timer);
  }
  await record({ event: 'cancelled safely', memory: memoryStats() });
  const recovered = await timed('after cancellation', () =>
    makeSession('qwen4-recovered').send('What is the capital of France? Reply only with the city.'),
  );
  assert.match((recovered as { text: string }).text, /Paris/i);
  await model.resetCaches();
}
function solid(r: number, g: number, b: number) {
  const pixels = Buffer.alloc(64 * (1 + 64 * 3));
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      const i = y * 193 + 1 + x * 3;
      pixels[i] = r;
      pixels[i + 1] = g;
      pixels[i + 2] = b;
    }
  }
  function chunk(type: string, data: Buffer) {
    const body = Buffer.concat([Buffer.from(type), data]);
    let crc = 0xffffffff;
    for (const byte of body) {
      crc ^= byte;
      for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
    const length = Buffer.alloc(4);
    length.writeUInt32BE(data.length);
    const checksum = Buffer.alloc(4);
    checksum.writeUInt32BE((crc ^ 0xffffffff) >>> 0);
    return Buffer.concat([length, body, checksum]);
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(64, 0);
  header.writeUInt32BE(64, 4);
  header[8] = 8;
  header[9] = 2;
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(pixels)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}
const vision = makeSession('qwen4-images');
const red = await timed('red image', () =>
  vision.send('What single color fills this image? Reply with just the color name.', { images: [solid(255, 0, 0)] }),
);
assert.match((red as { text: string }).text, /red/i);
const redAgain = await timed('image text continuation', () => vision.send('Name that color again using one word.'));
assert.match((redAgain as { text: string }).text, /red/i);
assert(
  ((redAgain as { cachedTokens?: number }).cachedTokens ?? 0) > 0,
  'Image continuation should reuse its matching live state.',
);
const blue = await timed('changed image', () =>
  vision.send('What single color fills this new image? Reply with just the color name.', {
    images: [solid(0, 0, 255)],
  }),
);
assert.match((blue as { text: string }).text, /blue/i);
await model.resetCaches();
const stats = await model.schedulerStats();
assert.equal(stats.allocatedBlocks, 0, 'Reset must release all request pages.');
assert.equal(stats.reservedStateBytes, 0, 'Reset must release scheduler state reservations.');
await record({
  event: 'passed',
  seconds: (performance.now() - started) / 1000,
  stats,
});
