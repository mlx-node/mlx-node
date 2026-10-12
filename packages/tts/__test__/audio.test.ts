import { mkdtemp, open, rm, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { it, expect, vi } from 'vite-plus/test';

import { WavWriter, readWav } from '../src/audio.js';

vi.mock('node:fs/promises', async (importOriginal) => {
  const fs = await importOriginal<typeof import('node:fs/promises')>();
  return { ...fs, open: vi.fn(fs.open) };
});

it('writes a bounded stream with a correct RIFF header, interleaved channels and clipped PCM16', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-tts-'));
  const path = join(dir, 'audio.wav');
  try {
    const writer = await WavWriter.open(path, 24000, 2);
    await writer.write(new Float32Array([-2, 2, -0.5, 0.5]));
    await writer.write(new Float32Array([0, 0]));
    await writer.close();
    const bytes = await readFile(path);
    expect(bytes.readUInt32LE(4)).toBe(bytes.length - 8);
    expect(bytes.readUInt32LE(40)).toBe(12);
    const audio = await readWav(path);
    expect(audio.sampleRate).toBe(24000);
    expect(audio.channels).toBe(2);
    expect([...audio.samples]).toEqual([-1, 32767 / 32768, -0.5, 0.5, 0, 0]);
    await expect(writer.write(new Float32Array([0, 0]))).rejects.toThrow('closed');
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

it('rejects floating WAV samples that overflow the public Float32 PCM format', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-tts-'));
  const path = join(dir, 'float64.wav');
  try {
    const bytes = Buffer.alloc(52);
    bytes.write('RIFF');
    bytes.writeUInt32LE(44, 4);
    bytes.write('WAVEfmt ', 8);
    bytes.writeUInt32LE(16, 16);
    bytes.writeUInt16LE(3, 20);
    bytes.writeUInt16LE(1, 22);
    bytes.writeUInt32LE(24000, 24);
    bytes.writeUInt32LE(24000 * 8, 28);
    bytes.writeUInt16LE(8, 32);
    bytes.writeUInt16LE(64, 34);
    bytes.write('data', 36);
    bytes.writeUInt32LE(8, 40);
    for (const sample of [1e100, -1e100, Infinity, NaN]) {
      bytes.writeDoubleLE(sample, 44);
      await writeFile(path, bytes);
      await expect(readWav(path)).rejects.toThrow('non-finite');
    }
    bytes.writeDoubleLE(0.25, 44);
    await writeFile(path, bytes);
    expect([...(await readWav(path)).samples]).toEqual([0.25]);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

it('validates all WAV header fields before truncating an existing output file', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-tts-'));
  const path = join(dir, 'existing.wav');
  try {
    await writeFile(path, 'preserve existing audio');
    for (const [sampleRate, channels] of [
      [2 ** 32, 1],
      [2 ** 31, 1],
      [2 ** 26, 32],
    ]) {
      await expect(WavWriter.open(path, sampleRate, channels)).rejects.toThrow('Invalid WAV format');
      expect(await readFile(path, 'utf8')).toBe('preserve existing audio');
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

it('makes every concurrent close await file closure and the final header', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-tts-'));
  const path = join(dir, 'audio.wav');
  let release!: () => void;
  const ready = new Promise<void>((resolve) => {
    release = resolve;
  });
  try {
    const fs = await vi.importActual<typeof import('node:fs/promises')>('node:fs/promises');
    let closeCalls = 0;
    vi.mocked(open).mockImplementationOnce(async (...args) => {
      const file = await fs.open(...args);
      const closeFile = file.close.bind(file);
      vi.spyOn(file, 'close').mockImplementation(async () => {
        closeCalls++;
        await ready;
        await closeFile();
      });
      return file;
    });
    const writer = await WavWriter.open(path, 24000);
    await writer.write(new Float32Array([0.25]));
    const first = writer.close();
    await vi.waitFor(() => expect(closeCalls).toBe(1));
    let secondClosed = false;
    const second = writer.close().then(() => {
      secondClosed = true;
    });
    await Promise.resolve();
    expect(secondClosed).toBe(false);
    release();
    await Promise.all([first, second]);
    expect(secondClosed).toBe(true);
    expect([...(await readWav(path)).samples]).toEqual([0.25]);
  } finally {
    release();
    await rm(dir, { recursive: true, force: true });
  }
});

it('preserves a write failure on repeated close calls', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-tts-'));
  const path = join(dir, 'audio.wav');
  try {
    const fs = await vi.importActual<typeof import('node:fs/promises')>('node:fs/promises');
    const error = new Error('disk write failed');
    vi.mocked(open).mockImplementationOnce(async (...args) => {
      const file = await fs.open(...args);
      const write = file.write.bind(file);
      vi.spyOn(file, 'write').mockImplementationOnce(write).mockRejectedValueOnce(error);
      return file;
    });
    const writer = await WavWriter.open(path, 24000);
    await expect(writer.write(new Float32Array([0.25]))).rejects.toBe(error);
    await expect(writer.close()).rejects.toBe(error);
    await expect(writer.close()).rejects.toBe(error);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
