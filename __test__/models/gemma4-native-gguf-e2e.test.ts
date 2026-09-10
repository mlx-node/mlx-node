import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { type ChatSession, loadSession } from '@mlx-node/lm';
import { beforeAll, describe, expect, it } from 'vite-plus/test';

// Opt in with the original GGUF file or repository directory, including its
// config/tokenizer assets and media companion. Exercises public loading and
// real packed inference, without requiring this checkpoint on CI.
const modelPath = process.env.GEMMA4_GGUF_MODEL_PATH;
const imagePath = resolve('examples/ocr.png');
const audioPath = resolve('examples/audio-ask-16k.wav');

describe.skipIf(!modelPath)('Gemma4 native GGUF loading and inference', () => {
  let session: ChatSession;

  beforeAll(async () => {
    session = await loadSession(modelPath!);
  }, 300_000);

  async function generate(prompt: string, media: { images?: Uint8Array[]; audio?: Uint8Array[] } = {}) {
    let text = '';
    for await (const event of session.sendStream(prompt, {
      ...media,
      config: { maxNewTokens: 96, temperature: 0, reasoningEffort: 'none', enableMtp: false },
    })) {
      if (!event.done) text += event.text;
    }
    await session.reset();
    return text;
  }

  it('generates a correct text answer from the original packed checkpoint', async () => {
    expect(await generate('What is the capital of France? Answer with just the city name.')).toMatch(/Paris/i);
  });

  it.skipIf(!existsSync(imagePath))('loads the media companion and reads the image fixture', async () => {
    const text = await generate('Read the heading in this image.', {
      images: [new Uint8Array(readFileSync(imagePath))],
    });
    expect(text).toMatch(/Trunch|reconciliation/i);
  });

  it.skipIf(!existsSync(audioPath))('loads the audio projection and transcribes the fixture', async () => {
    const text = await generate('Transcribe this audio.', {
      audio: [new Uint8Array(readFileSync(audioPath))],
    });
    expect(text).toMatch(/what can I do for you/i);
    expect(text).toMatch(/start listening/i);
  });
});
