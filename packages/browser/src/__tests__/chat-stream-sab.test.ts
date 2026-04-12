/**
 * Unit tests for the SAB chat-stream reader (chat-stream-sab.ts).
 *
 * No Web Worker is used: the test-producer helpers write directly into the SAB
 * from the same thread. This is safe because Atomics.waitAsync is always async
 * and the reader loop yields to the microtask queue between records.
 */

import type { ChatStreamChunk } from '@mlx-node/core';
import { describe, test, expect } from 'vitest';

import {
  createSabRing,
  SAB_HEADER_BYTES,
  SEQ_IDX,
  WRITE_CUR_IDX,
  CANCELLED_IDX,
  RECORD_HEADER_BYTES,
  KIND_TEXT,
  KIND_JSON,
  KIND_ERROR,
  FLAG_IS_REASONING,
  MIN_SAB_BYTES,
} from '../chat-stream-sab';

// ---------------------------------------------------------------------------
// Test-producer helpers
// ---------------------------------------------------------------------------

/**
 * Encode one record into a Uint8Array.
 * Layout: [len_lo, len_hi, kind, flags, ...payload] — matches write_record in Rust.
 */
function encodeRecord(kind: number, flags: number, payload: Uint8Array): Uint8Array {
  const total = RECORD_HEADER_BYTES + payload.length;
  const buf = new Uint8Array(total);
  buf[0] = payload.length & 0xff;
  buf[1] = (payload.length >>> 8) & 0xff;
  buf[2] = kind;
  buf[3] = flags;
  buf.set(payload, RECORD_HEADER_BYTES);
  return buf;
}

/**
 * Write a pre-encoded record into the ring body with wrap-around support.
 * Bumps write_cur and increments seq + notifies — mirrors SabSink::send.
 */
function writeRecord(headerI32: Int32Array, bodyU8: Uint8Array, record: Uint8Array): void {
  const bodyLen = bodyU8.length;
  const writeCur = Atomics.load(headerI32, WRITE_CUR_IDX);
  const total = record.length;

  if (writeCur + total <= bodyLen) {
    bodyU8.set(record, writeCur);
  } else {
    const firstLen = bodyLen - writeCur;
    bodyU8.set(record.subarray(0, firstLen), writeCur);
    bodyU8.set(record.subarray(firstLen), 0);
  }

  const newWriteCur = (writeCur + total) % bodyLen;
  Atomics.store(headerI32, WRITE_CUR_IDX, newWriteCur);
  // Bump seq and wake any reader waiting on it.
  Atomics.add(headerI32, SEQ_IDX, 1);
  Atomics.notify(headerI32, SEQ_IDX);
}

const enc = new TextEncoder();

/** Write a KIND_TEXT record. */
function writeTextRecord(headerI32: Int32Array, bodyU8: Uint8Array, text: string, isReasoning: boolean): void {
  const payload = enc.encode(text);
  const flags = isReasoning ? FLAG_IS_REASONING : 0;
  writeRecord(headerI32, bodyU8, encodeRecord(KIND_TEXT, flags, payload));
}

/** Write a KIND_JSON record (Rust-style snake_case object). */
function writeJsonRecord(headerI32: Int32Array, bodyU8: Uint8Array, obj: Record<string, unknown>): void {
  const payload = enc.encode(JSON.stringify(obj));
  writeRecord(headerI32, bodyU8, encodeRecord(KIND_JSON, 0, payload));
}

/** Write a KIND_ERROR record. */
function writeErrorRecord(headerI32: Int32Array, bodyU8: Uint8Array, msg: string): void {
  const payload = enc.encode(msg);
  writeRecord(headerI32, bodyU8, encodeRecord(KIND_ERROR, 0, payload));
}

/**
 * Create a SabRing, let setupProducer write records into it, and collect
 * expectedCount chunks from the reader. Fails on first error or timeout.
 */
async function collectFromSabRing(
  sizeBytes: number,
  expectedCount: number,
  setupProducer: (headerI32: Int32Array, bodyU8: Uint8Array) => void,
  timeoutMs = 3000,
): Promise<ChatStreamChunk[]> {
  const ring = createSabRing(sizeBytes);
  const headerI32 = new Int32Array(ring.sab, 0, SAB_HEADER_BYTES / 4);
  const bodyU8 = new Uint8Array(ring.sab, SAB_HEADER_BYTES);

  return new Promise<ChatStreamChunk[]>((resolve, reject) => {
    const collected: ChatStreamChunk[] = [];

    const timer = setTimeout(() => {
      abort.abort();
      reject(
        new Error(`collectFromSabRing timeout after ${timeoutMs}ms — got ${collected.length}/${expectedCount} chunks`),
      );
    }, timeoutMs);

    const abort = ring.reader(
      (chunk) => {
        collected.push(chunk);
        if (collected.length >= expectedCount) {
          clearTimeout(timer);
          abort.abort();
          resolve(collected);
        }
      },
      (err) => {
        clearTimeout(timer);
        abort.abort();
        reject(err);
      },
    );

    // Yield to the event loop so the reader can issue its first
    // Atomics.waitAsync before we write records.
    queueMicrotask(() => setupProducer(headerI32, bodyU8));
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('chat-stream-sab', () => {
  // -------------------------------------------------------------------------
  // 1. Single KIND_TEXT chunk — normal text, no reasoning flag
  // -------------------------------------------------------------------------
  test('decodes single text chunk', async () => {
    const chunks = await collectFromSabRing(MIN_SAB_BYTES + 128, 1, (h, b) =>
      writeTextRecord(h, b, 'hello world', false),
    );

    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toMatchObject({
      text: 'hello world',
      done: false,
      isReasoning: false,
    });
  });

  // -------------------------------------------------------------------------
  // 2. KIND_TEXT with FLAG_IS_REASONING — should report isReasoning: true
  // -------------------------------------------------------------------------
  test('reasoning flag round-trip', async () => {
    const chunks = await collectFromSabRing(MIN_SAB_BYTES + 128, 1, (h, b) =>
      writeTextRecord(h, b, 'thinking...', true),
    );

    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toMatchObject({
      text: 'thinking...',
      done: false,
      isReasoning: true,
    });
  });

  // -------------------------------------------------------------------------
  // 3. KIND_JSON — snake_case fields must be camelCase in output
  // -------------------------------------------------------------------------
  test('KIND_JSON snake_case → camelCase conversion', async () => {
    // Rust WireChunk serialises with snake_case field names.
    const wireChunk = {
      text: '',
      done: true,
      finish_reason: 'stop',
      num_tokens: 42,
      is_reasoning: false,
    };

    const chunks = await collectFromSabRing(MIN_SAB_BYTES + 256, 1, (h, b) => writeJsonRecord(h, b, wireChunk));

    expect(chunks).toHaveLength(1);
    const c = chunks[0]!;
    expect(c.done).toBe(true);
    expect(c.finishReason).toBe('stop');
    expect(c.numTokens).toBe(42);
    expect(c.isReasoning).toBe(false);
    // Snake_case keys must NOT be present on the result.
    expect((c as Record<string, unknown>)['finish_reason']).toBeUndefined();
    expect((c as Record<string, unknown>)['num_tokens']).toBeUndefined();
    expect((c as Record<string, unknown>)['is_reasoning']).toBeUndefined();
  });

  // -------------------------------------------------------------------------
  // 4. KIND_JSON with nested performance object (snake_case → camelCase)
  // -------------------------------------------------------------------------
  test('KIND_JSON performance metrics camelCase conversion', async () => {
    const wireChunk = {
      text: '',
      done: true,
      finish_reason: 'stop',
      num_tokens: 10,
      performance: {
        ttft_ms: 12.5,
        prefill_tokens_per_second: 100.0,
        decode_tokens_per_second: 80.0,
      },
    };

    const chunks = await collectFromSabRing(MIN_SAB_BYTES + 512, 1, (h, b) => writeJsonRecord(h, b, wireChunk));

    expect(chunks).toHaveLength(1);
    const c = chunks[0]!;
    expect(c.performance).toBeDefined();
    expect(c.performance!.ttftMs).toBe(12.5);
    expect(c.performance!.prefillTokensPerSecond).toBe(100.0);
    expect(c.performance!.decodeTokensPerSecond).toBe(80.0);
  });

  // -------------------------------------------------------------------------
  // 5. KIND_ERROR — should call onError, not onChunk
  // -------------------------------------------------------------------------
  test('KIND_ERROR calls onError', async () => {
    const ring = createSabRing(MIN_SAB_BYTES + 128);
    const headerI32 = new Int32Array(ring.sab, 0, SAB_HEADER_BYTES / 4);
    const bodyU8 = new Uint8Array(ring.sab, SAB_HEADER_BYTES);

    const errors: Error[] = [];
    const chunks: ChatStreamChunk[] = [];

    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        abort.abort();
        reject(new Error('timeout'));
      }, 3000);

      const abort = ring.reader(
        (chunk) => chunks.push(chunk),
        (err) => {
          errors.push(err);
          clearTimeout(timer);
          abort.abort();
          resolve();
        },
      );

      queueMicrotask(() => writeErrorRecord(headerI32, bodyU8, 'something went wrong'));
    });

    expect(chunks).toHaveLength(0);
    expect(errors).toHaveLength(1);
    expect(errors[0]!.message).toBe('something went wrong');
  });

  // -------------------------------------------------------------------------
  // 6. Multiple sequential chunks in order
  // -------------------------------------------------------------------------
  test('decodes multiple sequential text chunks in order', async () => {
    const texts = ['alpha', 'beta', 'gamma', 'delta'];

    const chunks = await collectFromSabRing(MIN_SAB_BYTES + 512, texts.length, (h, b) => {
      for (const t of texts) writeTextRecord(h, b, t, false);
    });

    expect(chunks.map((c) => c.text)).toEqual(texts);
  });

  // -------------------------------------------------------------------------
  // 7. Ring wrap: force a record to span the body boundary
  //
  // Strategy: pre-position write_cur and read_cur near the end of the body
  // so the first record written crosses the wrap boundary.
  //
  // body_len = 100 bytes. Record size = 4 header + 5 payload ("textN") = 9 bytes.
  // Pre-position cursors at offset 95 (both read_cur and write_cur = 95).
  // Record 1 at 95: 95+9=104 > 100 → wraps.
  //   body[95..100] = 5 bytes (header + first byte of payload)
  //   body[0..4] = 4 bytes (rest of payload)
  //   write_cur = 4
  // Records 2-5 all fit without further wrapping.
  // Total = 5×9=45 bytes; usable = 99. No overflow.
  // -------------------------------------------------------------------------
  test('ring wrap — records spanning the body boundary decode correctly', async () => {
    const B = 100; // body_len
    const SAB_SIZE = SAB_HEADER_BYTES + B;
    const startPos = B - 5; // 95: first record (9 bytes) crosses boundary

    const ring = createSabRing(SAB_SIZE);
    const headerI32 = new Int32Array(ring.sab, 0, SAB_HEADER_BYTES / 4);
    const bodyU8 = new Uint8Array(ring.sab, SAB_HEADER_BYTES);

    // Pre-position both cursors: ring appears empty at offset startPos.
    Atomics.store(headerI32, WRITE_CUR_IDX, startPos);
    Atomics.store(headerI32, 2 /* READ_CUR_IDX */, startPos);

    const count = 5;
    const texts = Array.from({ length: count }, (_, i) => `text${i}`); // 5 bytes each

    const chunks = await new Promise<ChatStreamChunk[]>((resolve, reject) => {
      const collected: ChatStreamChunk[] = [];
      const timer = setTimeout(() => {
        abort.abort();
        reject(new Error(`wrap test timeout: got ${collected.length}/${count} chunks`));
      }, 3000);

      const abort = ring.reader(
        (chunk) => {
          collected.push(chunk);
          if (collected.length >= count) {
            clearTimeout(timer);
            abort.abort();
            resolve(collected);
          }
        },
        (err) => {
          clearTimeout(timer);
          abort.abort();
          reject(err);
        },
      );

      queueMicrotask(() => {
        for (const t of texts) writeTextRecord(headerI32, bodyU8, t, false);
      });
    });

    expect(chunks).toHaveLength(count);
    for (let i = 0; i < count; i++) {
      expect(chunks[i]!.text).toBe(texts[i]);
    }
  });

  // -------------------------------------------------------------------------
  // 8. Abort signal stops the loop and sets CANCELLED_IDX = 1
  // -------------------------------------------------------------------------
  test('abort signal stops the loop quickly and sets cancelled flag', async () => {
    const ring = createSabRing(MIN_SAB_BYTES + 128);
    const headerI32 = new Int32Array(ring.sab, 0, SAB_HEADER_BYTES / 4);

    const chunks: ChatStreamChunk[] = [];
    const abort = ring.reader(
      (chunk) => chunks.push(chunk),
      () => {},
    );

    // Give the loop a microtask tick to start.
    await new Promise<void>((resolve) => queueMicrotask(resolve));

    const t0 = Date.now();
    abort.abort();

    // CANCELLED_IDX must be set synchronously by the abort event listener.
    expect(Atomics.load(headerI32, CANCELLED_IDX)).toBe(1);

    // The loop should exit within a generous budget. The abort listener fires
    // synchronously and signal.aborted is checked at the top of the loop.
    await new Promise<void>((resolve) => setTimeout(resolve, 150));
    const elapsed = Date.now() - t0;
    expect(elapsed).toBeLessThan(200);
    expect(chunks).toHaveLength(0);
  });

  // -------------------------------------------------------------------------
  // 9. Mixed chunk types: text token then JSON done chunk
  // -------------------------------------------------------------------------
  test('mixed text and JSON chunks', async () => {
    const wireJsonDone = {
      text: '',
      done: true,
      finish_reason: 'stop',
      num_tokens: 5,
    };

    const chunks = await collectFromSabRing(MIN_SAB_BYTES + 512, 2, (h, b) => {
      writeTextRecord(h, b, 'token', false);
      writeJsonRecord(h, b, wireJsonDone);
    });

    expect(chunks).toHaveLength(2);
    expect(chunks[0]!.text).toBe('token');
    expect(chunks[0]!.done).toBe(false);
    expect(chunks[1]!.done).toBe(true);
    expect(chunks[1]!.finishReason).toBe('stop');
    expect(chunks[1]!.numTokens).toBe(5);
  });

  // -------------------------------------------------------------------------
  // 10. createSabRing rejects SAB smaller than MIN_SAB_BYTES
  // -------------------------------------------------------------------------
  test('createSabRing throws RangeError for undersized SAB', () => {
    expect(() => createSabRing(MIN_SAB_BYTES - 1)).toThrow(RangeError);
    expect(() => createSabRing(0)).toThrow(RangeError);
  });

  // -------------------------------------------------------------------------
  // 11. read_cur equals write_cur after all records consumed
  // -------------------------------------------------------------------------
  test('read_cur advances to write_cur after consuming all records', async () => {
    const ring = createSabRing(MIN_SAB_BYTES + 256);
    const headerI32 = new Int32Array(ring.sab, 0, SAB_HEADER_BYTES / 4);
    const bodyU8 = new Uint8Array(ring.sab, SAB_HEADER_BYTES);

    const count = 3;
    const text = 'abcde'; // 5 bytes; record = 9 bytes

    await new Promise<ChatStreamChunk[]>((resolve, reject) => {
      const collected: ChatStreamChunk[] = [];
      const timer = setTimeout(() => {
        abort.abort();
        reject(new Error('timeout'));
      }, 3000);

      const abort = ring.reader(
        (chunk) => {
          collected.push(chunk);
          if (collected.length >= count) {
            clearTimeout(timer);
            abort.abort();
            resolve(collected);
          }
        },
        (err) => {
          clearTimeout(timer);
          abort.abort();
          reject(err);
        },
      );

      queueMicrotask(() => {
        for (let i = 0; i < count; i++) writeTextRecord(headerI32, bodyU8, text, false);
      });
    });

    // After the reader has consumed all records, read_cur should equal write_cur.
    // Allow a small delay for the Atomics.store(READ_CUR_IDX) to have run.
    await new Promise<void>((resolve) => setTimeout(resolve, 50));

    const readCur = Atomics.load(headerI32, 2); // READ_CUR_IDX = 2
    const writeCur = Atomics.load(headerI32, WRITE_CUR_IDX);
    expect(readCur).toBe(writeCur);
  });

  // -------------------------------------------------------------------------
  // 12. KIND_JSON with tool_calls (camelCase)
  // -------------------------------------------------------------------------
  test('KIND_JSON tool_calls field renamed to toolCalls', async () => {
    const wireChunk = {
      text: '',
      done: true,
      finish_reason: 'tool_calls',
      tool_calls: [
        {
          name: 'my_tool',
          arguments: { key: 'val' },
          raw: 'raw',
          result: null,
          error: null,
        },
      ],
    };

    const chunks = await collectFromSabRing(MIN_SAB_BYTES + 512, 1, (h, b) => writeJsonRecord(h, b, wireChunk));

    expect(chunks).toHaveLength(1);
    const c = chunks[0]!;
    expect(c.finishReason).toBe('tool_calls');
    expect(Array.isArray(c.toolCalls)).toBe(true);
    expect(c.toolCalls!).toHaveLength(1);
    // snake_case key must be absent
    expect((c as Record<string, unknown>)['tool_calls']).toBeUndefined();
  });

  // -------------------------------------------------------------------------
  // 13. KIND_JSON raw_text field renamed to rawText
  // -------------------------------------------------------------------------
  test('KIND_JSON raw_text field renamed to rawText', async () => {
    const wireChunk = {
      text: 'output',
      done: true,
      raw_text: 'raw output before parsing',
    };

    const chunks = await collectFromSabRing(MIN_SAB_BYTES + 256, 1, (h, b) => writeJsonRecord(h, b, wireChunk));

    expect(chunks).toHaveLength(1);
    const c = chunks[0]!;
    expect(c.rawText).toBe('raw output before parsing');
    expect((c as Record<string, unknown>)['raw_text']).toBeUndefined();
  });
});
