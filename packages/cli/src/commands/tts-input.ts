import { createReadStream } from 'node:fs';
import { StringDecoder } from 'node:string_decoder';

import type { TtsInputEvent } from '@mlx-node/tts';

/** Opening the file belongs to iterator consumption, after model/sink validation. */
export async function* ttsFileBytes(path: string): AsyncGenerator<Uint8Array> {
  const file = createReadStream(path);
  try {
    yield* file;
  } finally {
    file.destroy();
  }
}

/** Bounded UTF-8 records; JSON syntax never becomes spoken text implicitly. */
export async function* decodeTtsInput(
  source: AsyncIterable<Uint8Array>,
  format: 'text' | 'jsonl',
  maxRecordBytes = 64 * 1024,
): AsyncGenerator<TtsInputEvent> {
  if (!Number.isSafeInteger(maxRecordBytes) || maxRecordBytes < 1)
    throw new RangeError('Invalid TTS input record limit');
  const decoder = new StringDecoder('utf8');
  let pending = '';
  let line = 0;
  function parse(record: string): TtsInputEvent | undefined {
    line++;
    try {
      if (Buffer.byteLength(record) > maxRecordBytes) throw new Error('record exceeds byte limit');
      if (!record.trim()) return;
      const event: unknown = JSON.parse(record);
      if (event && typeof event === 'object' && 'type' in event) {
        if (event.type === 'flush') return { type: 'flush' };
        if (event.type === 'text' && 'text' in event && typeof event.text === 'string')
          return { type: 'text', text: event.text };
        if (event.type === 'instruct' && 'value' in event && (typeof event.value === 'string' || event.value === null))
          return { type: 'instruct', value: event.value };
      }
      throw new Error('expected a text, instruct, or flush event');
    } catch (error) {
      throw new Error(`TTS JSONL line ${line}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  function* records(text: string): Generator<TtsInputEvent> {
    pending += text;
    let end: number;
    while ((end = pending.indexOf('\n')) >= 0) {
      const event = parse(pending.slice(0, end));
      pending = pending.slice(end + 1);
      if (event) yield event;
    }
    if (Buffer.byteLength(pending) > maxRecordBytes)
      throw new Error(`TTS JSONL line ${line + 1}: record exceeds byte limit`);
  }
  for await (const chunk of source) {
    const text = decoder.write(Buffer.from(chunk));
    if (format === 'text') {
      if (text) yield text;
    } else yield* records(text);
  }
  const tail = decoder.end();
  if (format === 'text') {
    if (tail) yield tail;
  } else {
    yield* records(tail);
    if (pending) {
      const event = parse(pending);
      if (event) yield event;
    }
  }
}
