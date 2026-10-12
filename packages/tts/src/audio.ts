import { open, readFile } from 'node:fs/promises';
import type { FileHandle } from 'node:fs/promises';

import type { AudioChunk, AudioData } from './types.js';

/** Read uncompressed RIFF/WAVE audio, preserving source sample rate and channels. */
export async function readWav(path: string): Promise<AudioData> {
  const data = await readFile(path);
  if (data.length < 12 || data.toString('ascii', 0, 4) !== 'RIFF' || data.toString('ascii', 8, 12) !== 'WAVE')
    throw new Error('Expected a RIFF/WAVE file');
  const end = data.readUInt32LE(4) + 8;
  if (end > data.length) throw new Error('Truncated WAV file');
  let format: { encoding: number; channels: number; rate: number; bits: number; align: number } | undefined;
  const blocks: Buffer[] = [];
  for (let at = 12; at + 8 <= end;) {
    const size = data.readUInt32LE(at + 4);
    const body = at + 8;
    if (body + size > end) throw new Error('Truncated WAV chunk');
    const name = data.toString('ascii', at, body - 4);
    if (name === 'fmt ') {
      if (size < 16) throw new Error('Invalid WAV format');
      let encoding = data.readUInt16LE(body);
      if (encoding === 0xfffe) {
        if (
          size < 40 ||
          data.readUInt16LE(body + 16) < 22 ||
          !data.subarray(body + 28, body + 40).equals(Buffer.from('00001000800000aa00389b71', 'hex'))
        )
          throw new Error('Unsupported extensible WAV format');
        encoding = data.readUInt32LE(body + 24);
      }
      format = {
        encoding,
        channels: data.readUInt16LE(body + 2),
        rate: data.readUInt32LE(body + 4),
        align: data.readUInt16LE(body + 12),
        bits: data.readUInt16LE(body + 14),
      };
    } else if (name === 'data') blocks.push(data.subarray(body, body + size));
    at = body + size + (size & 1);
  }
  if (!format || !blocks.length) throw new Error('WAV is missing format or samples');
  const { encoding, channels, rate, bits, align } = format;
  if (
    !channels ||
    !rate ||
    align !== (channels * bits) / 8 ||
    !((encoding === 1 && [8, 16, 24, 32].includes(bits)) || (encoding === 3 && [32, 64].includes(bits)))
  )
    throw new Error('Unsupported WAV sample format');
  const bytes = Buffer.concat(blocks);
  if (bytes.length % align) throw new Error('Incomplete WAV sample frame');
  const samples = new Float32Array(bytes.length / (bits / 8));
  for (let i = 0, at = 0; i < samples.length; i++, at += bits / 8) {
    const value =
      encoding === 3
        ? bits === 32
          ? bytes.readFloatLE(at)
          : bytes.readDoubleLE(at)
        : bits === 8
          ? (bytes[at] - 128) / 128
          : bytes.readIntLE(at, bits / 8) / 2 ** (bits - 1);
    samples[i] = value;
    if (!Number.isFinite(samples[i])) throw new Error('WAV contains non-finite samples');
  }
  return { samples, sampleRate: rate, channels };
}

/** Streaming PCM16 WAV writer. Memory use is bounded by the caller's chunk. */
export class WavWriter {
  readonly #file: FileHandle;
  #bytes = 0;
  #closed = false;
  #closing: Promise<void> | undefined;
  #pending = Promise.resolve();
  private constructor(
    file: FileHandle,
    readonly sampleRate: number,
    readonly channels: number,
  ) {
    this.#file = file;
  }
  static async open(path: string, sampleRate: number, channels = 1): Promise<WavWriter> {
    if (
      !Number.isSafeInteger(sampleRate) ||
      sampleRate < 1 ||
      !Number.isSafeInteger(channels) ||
      channels < 1 ||
      channels > 32 ||
      sampleRate * channels * 2 > 0xffffffff
    )
      throw new Error('Invalid WAV format');
    const file = await open(path, 'w');
    const writer = new WavWriter(file, sampleRate, channels);
    try {
      await writer.#header();
      return writer;
    } catch (error) {
      await file.close();
      throw error;
    }
  }
  async #header() {
    const header = Buffer.alloc(44);
    header.write('RIFF');
    header.writeUInt32LE(this.#bytes + 36, 4);
    header.write('WAVEfmt ', 8);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
    header.writeUInt16LE(this.channels, 22);
    header.writeUInt32LE(this.sampleRate, 24);
    header.writeUInt32LE(this.sampleRate * this.channels * 2, 28);
    header.writeUInt16LE(this.channels * 2, 32);
    header.writeUInt16LE(16, 34);
    header.write('data', 36);
    header.writeUInt32LE(this.#bytes, 40);
    await this.#writeAll(header, 0);
  }
  async #writeAll(buffer: Buffer, position: number) {
    let offset = 0;
    while (offset < buffer.length) {
      const { bytesWritten } = await this.#file.write(buffer, offset, buffer.length - offset, position + offset);
      if (!bytesWritten) throw new Error('WAV write made no progress');
      offset += bytesWritten;
    }
  }
  write(chunk: AudioChunk | Float32Array): Promise<void> {
    if (this.#closed) return Promise.reject(new Error('WAV writer closed'));
    if (!(chunk instanceof Float32Array) && (chunk.sampleRate !== this.sampleRate || chunk.channels !== this.channels))
      return Promise.reject(new Error('WAV chunk format changed'));
    const samples = chunk instanceof Float32Array ? chunk : chunk.samples;
    if (samples.length % this.channels) return Promise.reject(new Error('Incomplete PCM frame'));
    const buffer = Buffer.allocUnsafe(samples.length * 2);
    for (let i = 0; i < samples.length; i++) {
      if (!Number.isFinite(samples[i])) return Promise.reject(new Error('Non-finite PCM sample'));
      buffer.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32768))), i * 2);
    }
    this.#pending = this.#pending.then(async () => {
      if (this.#bytes + buffer.length > 0xffffffff - 36) throw new Error('WAV exceeds the RIFF 4 GiB limit');
      await this.#writeAll(buffer, 44 + this.#bytes);
      this.#bytes += buffer.length;
    });
    return this.#pending;
  }
  close(): Promise<void> {
    if (this.#closing) return this.#closing;
    this.#closed = true;
    this.#closing = (async () => {
      try {
        await this.#pending;
        await this.#header();
      } finally {
        await this.#file.close();
      }
    })();
    return this.#closing;
  }
}

export interface PlaybackStats {
  playedSeconds: number;
  underruns: number;
  firstPlaybackMs?: number | null;
}
export interface AudioPlayer {
  write(samples: Float32Array): Promise<void>;
  finish(): Promise<PlaybackStats>;
  cancel(): void;
}
/** CoreAudio performs source-rate conversion. The device callback only reads PCM. */
export async function createAudioPlayer(
  sampleRate: number,
  channels = 1,
  options: { bufferSeconds?: number; prebufferSeconds?: number } = {},
): Promise<AudioPlayer> {
  if (process.platform !== 'darwin') throw new Error('PCM device playback currently requires macOS');
  const { PcmPlayer } = await import('@mlx-node/core');
  return PcmPlayer.open(sampleRate, channels, options.bufferSeconds, options.prebufferSeconds);
}
