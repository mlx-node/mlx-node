import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { WavWriter } from '@mlx-node/tts';
import { expect, it } from 'vite-plus/test';

it.each(['excited', '   '])(
  'replays the first instruction for complete-text-once: %j',
  async (instruct) => {
    const root = await mkdtemp(join(tmpdir(), 'tts-benchmark-samples-'));
    try {
      const audio = join(root, 'audio.wav');
      const report = join(root, 'report.json');
      const output = join(root, 'samples');
      const writer = await WavWriter.open(audio, 24000);
      await writer.write({
        samples: new Float32Array(16),
        sampleRate: 24000,
        channels: 1,
        startSample: 0,
        segmentIndex: 0,
      });
      await writer.close();
      await writeFile(
        report,
        JSON.stringify({
          parameters: { instruct: 'calm', instructions: [instruct], language: 'chinese' },
          input: { mode: 'complete-text-once', text: '你好。' },
          segments: [{ segmentIndex: 0, startSample: 0, endSample: 16 }],
        }),
      );
      await promisify(execFile)('vp', [
        'exec',
        'oxnode',
        fileURLToPath(new URL('../../scripts/tts/benchmark-samples.ts', import.meta.url)),
        '--report',
        report,
        '--audio',
        audio,
        '--output',
        output,
      ]);
      const samples = JSON.parse(await readFile(join(output, 'samples.json'), 'utf8'));
      expect(samples.cases).toHaveLength(1);
      expect(samples.cases[0].text).toBe('你好。');
      expect(samples.cases[0].instruct).toBe(instruct.trim() ? instruct : undefined);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  },
  10000,
);
