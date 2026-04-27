/**
 * Browser-side SharedArrayBuffer ring-buffer reader for the SAB chat streaming
 * path. The WASM producer (SabSink in Rust) writes records into the ring; this
 * module's reader decodes them on the main thread via Atomics.waitAsync,
 * eliminating postMessage overhead for the hot decode path.
 *
 * Wire format defined in crates/mlx-core/src/chat_stream/wire.rs.
 */

import type { ChatStreamChunk, PerformanceMetrics, ToolCallResult } from '@mlx-node/core';
import { sanitizeAssistantText } from './generated-text.js';

// ---------------------------------------------------------------------------
// Header layout — matches SAB_HEADER_OFFSET_* constants in wire.rs
// ---------------------------------------------------------------------------

/** Fixed header size in bytes (matches SAB_HEADER_BYTES in Rust). */
export const SAB_HEADER_BYTES = 32;

/** Int32Array index for the `seq` counter (producer bumps on each write). */
export const SEQ_IDX = 0; // byte offset 0

/** Int32Array index for `write_cur` (producer advances after writing). */
export const WRITE_CUR_IDX = 1; // byte offset 4

/** Int32Array index for `read_cur` (reader advances after consuming). */
export const READ_CUR_IDX = 2; // byte offset 8

/** Int32Array index for `cancelled` (reader sets to 1 on abort). */
export const CANCELLED_IDX = 3; // byte offset 12

// ---------------------------------------------------------------------------
// Record constants — matches RECORD_* / KIND_* / FLAG_* in wire.rs
// ---------------------------------------------------------------------------

/** Fixed record header size: [len u16 LE, kind u8, flags u8]. */
export const RECORD_HEADER_BYTES = 4;

/** KIND_TEXT (0): raw UTF-8 text token, fast path. */
export const KIND_TEXT = 0;

/** KIND_JSON (1): full JSON-encoded WireChunk, used for structured fields. */
export const KIND_JSON = 1;

/** KIND_ERROR (2): UTF-8 error message; decoded as Error. */
export const KIND_ERROR = 2;

/** FLAG_IS_REASONING (bit 0): set in KIND_TEXT flags when is_reasoning=true. */
export const FLAG_IS_REASONING = 0x01;

/** Maximum payload bytes per record (u16::MAX = 65535). */
export const MAX_PAYLOAD_BYTES = 0xffff;

/**
 * Minimum usable SAB size: header + record header + 1 payload byte + 1 slack
 * (matches MIN_SAB_LEN in sab_sink.rs).
 */
export const MIN_SAB_BYTES = SAB_HEADER_BYTES + RECORD_HEADER_BYTES + 1 + 1;

/**
 * Reader wake-up poll interval in ms. Producer `SabSink::bump_seq_and_notify`
 * (sab_sink.rs) calls `memory.atomic.notify(seq_addr, u32::MAX)` on every
 * record, which is supposed to wake this reader's `Atomics.waitAsync(seq)`
 * via microtask — under 1 ms when it works. In practice, cross-worker notify
 * to a main-thread `waitAsync` is unreliable in current browsers: a non-zero
 * fraction of wakes are lost, and the reader sits parked on the stale `seq`
 * until the timeout fires. With the original 1000 ms timeout the missed-wake
 * path produced a visible "lag burst every ~1 s" artifact in streaming
 * output. 100 ms hid the artifact but still added up to 100 ms of latency
 * per lost notify. 16 ms (~1 frame) hides the artifact just as well while
 * bounding worst-case latency to a single frame. Idle cost is zero: the loop
 * exits the moment the stream's done-chunk lands. The real fix is a reliable
 * notify path; revisit once browsers improve cross-thread waitAsync delivery.
 */
export const READER_WAIT_TIMEOUT_MS = 16;

// ---------------------------------------------------------------------------
// Shared TextDecoder (module-level — avoid per-call construction cost)
// ---------------------------------------------------------------------------

const utf8 = new TextDecoder();

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface SabRing {
  /** The SharedArrayBuffer shared with the WASM producer. */
  sab: SharedArrayBuffer;
  /**
   * Start the async reader loop. Returns an AbortController whose `.abort()`
   * stops the loop and notifies the WASM producer to stop writing.
   *
   * @param onChunk - Called for each decoded ChatStreamChunk.
   * @param onError - Called on KIND_ERROR records or decode failures.
   */
  reader: (onChunk: (chunk: ChatStreamChunk) => void, onError: (e: Error) => void) => AbortController;
}

export interface SabRingOverHeap {
  /**
   * Start the async reader loop. Returns an AbortController whose `.abort()`
   * stops the loop and notifies the WASM producer to stop writing.
   *
   * @param onChunk - Called for each decoded ChatStreamChunk.
   * @param onError - Called on KIND_ERROR records or decode failures.
   */
  reader: (onChunk: (chunk: ChatStreamChunk) => void, onError: (e: Error) => void) => AbortController;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Allocate a new SharedArrayBuffer ring and return both the SAB (to hand to
 * the WASM producer) and a reader factory function.
 *
 * @param sizeBytes - Total SAB size (default 256 KiB). Must be >= MIN_SAB_BYTES.
 */
export function createSabRing(sizeBytes: number = 262_144): SabRing {
  if (sizeBytes < MIN_SAB_BYTES) {
    throw new RangeError(`createSabRing: sizeBytes (${sizeBytes}) < MIN_SAB_BYTES (${MIN_SAB_BYTES})`);
  }

  const sab = new SharedArrayBuffer(sizeBytes);
  // Int32Array view over the header region (8 i32s = 32 bytes).
  const headerI32 = new Int32Array(sab, 0, SAB_HEADER_BYTES / 4);
  // Uint8Array view over the ring body (everything after the 32-byte header).
  const bodyU8 = new Uint8Array(sab, SAB_HEADER_BYTES);
  const bodyLen = bodyU8.length;

  function reader(onChunk: (chunk: ChatStreamChunk) => void, onError: (e: Error) => void): AbortController {
    const abort = new AbortController();

    // On abort: set cancelled flag and wake any producer waiting for space.
    abort.signal.addEventListener('abort', () => {
      Atomics.store(headerI32, CANCELLED_IDX, 1);
      // Wake the WASM producer if it is sleeping in wait_for_space (waits on read_cur).
      Atomics.notify(headerI32, READ_CUR_IDX);
    });

    void runLoop(headerI32, bodyU8, bodyLen, onChunk, onError, abort.signal);
    return abort;
  }

  return { sab, reader };
}

/**
 * Build a ring reader over an existing memory region inside the WASM heap.
 *
 * Use this when the SAB backing must live inside the WASM instance's own
 * memory so napi-rs can zero-copy it through to the Rust SabSink. The caller
 * is responsible for `malloc`ing `byteLength` bytes at `heapOffset` before
 * calling, and for `free`ing them after the stream finishes (typically after
 * the reader's AbortController has fired on the done chunk).
 *
 * `memory.buffer` is re-fetched on every drain iteration so memory.grow does
 * not leave us holding stale views.
 */
export function createSabRingOverHeap(
  memory: WebAssembly.Memory,
  heapOffset: number,
  byteLength: number,
): SabRingOverHeap {
  if (byteLength < MIN_SAB_BYTES) {
    throw new RangeError(`createSabRingOverHeap: byteLength (${byteLength}) < MIN_SAB_BYTES (${MIN_SAB_BYTES})`);
  }
  if (heapOffset % 4 !== 0) {
    throw new RangeError(`createSabRingOverHeap: heapOffset must be 4-byte aligned (got ${heapOffset})`);
  }

  const bodyOffset = heapOffset + SAB_HEADER_BYTES;
  const bodyLen = byteLength - SAB_HEADER_BYTES;

  // Zero the 32-byte header so seq/write_cur/read_cur/cancelled start at 0.
  new Uint8Array(memory.buffer, heapOffset, SAB_HEADER_BYTES).fill(0);

  const headerView = (): Int32Array => new Int32Array(memory.buffer, heapOffset, SAB_HEADER_BYTES / 4);
  const bodyView = (): Uint8Array => new Uint8Array(memory.buffer, bodyOffset, bodyLen);

  function reader(onChunk: (chunk: ChatStreamChunk) => void, onError: (e: Error) => void): AbortController {
    const abort = new AbortController();

    abort.signal.addEventListener('abort', () => {
      const h = headerView();
      Atomics.store(h, CANCELLED_IDX, 1);
      Atomics.notify(h, READ_CUR_IDX);
      // Also wake the reader's waitAsync if it's sleeping on SEQ_IDX.
      Atomics.notify(h, SEQ_IDX);
    });

    void runLoopHeap(headerView, bodyView, bodyLen, onChunk, onError, abort.signal);
    return abort;
  }

  return { reader };
}

async function runLoopHeap(
  headerView: () => Int32Array,
  bodyView: () => Uint8Array,
  bodyLen: number,
  onChunk: (chunk: ChatStreamChunk) => void,
  onError: (e: Error) => void,
  signal: AbortSignal,
): Promise<void> {
  const scratch = new Uint8Array(RECORD_HEADER_BYTES + MAX_PAYLOAD_BYTES);
  let lastSeq = Atomics.load(headerView(), SEQ_IDX);

  while (!signal.aborted) {
    while (!signal.aborted) {
      const headerI32 = headerView();
      const bodyU8 = bodyView();
      const readCur = Atomics.load(headerI32, READ_CUR_IDX);
      const writeCur = Atomics.load(headerI32, WRITE_CUR_IDX);
      if (readCur === writeCur) break;

      copyFromRing(bodyU8, readCur, bodyLen, scratch, 0, RECORD_HEADER_BYTES);
      const payloadLen = scratch[0]! | (scratch[1]! << 8);
      const kind = scratch[2]!;
      const flags = scratch[3]!;
      const totalLen = RECORD_HEADER_BYTES + payloadLen;

      if (payloadLen > 0) {
        const payloadStart = (readCur + RECORD_HEADER_BYTES) % bodyLen;
        copyFromRing(bodyU8, payloadStart, bodyLen, scratch, RECORD_HEADER_BYTES, payloadLen);
      }

      try {
        dispatchRecord(kind, flags, scratch.subarray(RECORD_HEADER_BYTES, totalLen), onChunk, onError);
      } catch (e) {
        onError(e instanceof Error ? e : new Error(String(e)));
      }

      const newReadCur = (readCur + totalLen) % bodyLen;
      Atomics.store(headerI32, READ_CUR_IDX, newReadCur);
      Atomics.notify(headerI32, READ_CUR_IDX);
    }

    if (signal.aborted) break;

    const headerI32 = headerView();
    const currentSeq = Atomics.load(headerI32, SEQ_IDX);
    if (currentSeq === lastSeq) {
      // Timeout-polled wait — see READER_WAIT_TIMEOUT_MS for rationale.
      const { async, value } = Atomics.waitAsync(headerI32, SEQ_IDX, lastSeq, READER_WAIT_TIMEOUT_MS);
      const _result = async ? await value : (value as string);
    }
    lastSeq = Atomics.load(headerView(), SEQ_IDX);
  }
}

// ---------------------------------------------------------------------------
// Main read loop
// ---------------------------------------------------------------------------

async function runLoop(
  headerI32: Int32Array,
  bodyU8: Uint8Array,
  bodyLen: number,
  onChunk: (chunk: ChatStreamChunk) => void,
  onError: (e: Error) => void,
  signal: AbortSignal,
): Promise<void> {
  // Scratch buffer large enough for any single record.
  const scratch = new Uint8Array(RECORD_HEADER_BYTES + MAX_PAYLOAD_BYTES);
  let lastSeq = Atomics.load(headerI32, SEQ_IDX);

  while (!signal.aborted) {
    // -----------------------------------------------------------------------
    // Drain phase: read all records currently available (read_cur → write_cur).
    // -----------------------------------------------------------------------
    while (!signal.aborted) {
      const readCur = Atomics.load(headerI32, READ_CUR_IDX);
      const writeCur = Atomics.load(headerI32, WRITE_CUR_IDX);
      if (readCur === writeCur) break; // ring empty

      // Copy the 4-byte record header into scratch (handles wrap).
      copyFromRing(bodyU8, readCur, bodyLen, scratch, 0, RECORD_HEADER_BYTES);

      // Parse header fields. Length field is u16 LE at bytes [0..2).
      const payloadLen = scratch[0]! | (scratch[1]! << 8);
      const kind = scratch[2]!;
      const flags = scratch[3]!;
      const totalLen = RECORD_HEADER_BYTES + payloadLen;

      // Copy payload (if any) into scratch immediately after the header.
      if (payloadLen > 0) {
        const payloadStart = (readCur + RECORD_HEADER_BYTES) % bodyLen;
        copyFromRing(bodyU8, payloadStart, bodyLen, scratch, RECORD_HEADER_BYTES, payloadLen);
      }

      // Dispatch the record to the appropriate callback.
      try {
        dispatchRecord(kind, flags, scratch.subarray(RECORD_HEADER_BYTES, totalLen), onChunk, onError);
      } catch (e) {
        onError(e instanceof Error ? e : new Error(String(e)));
      }

      // Advance read_cur (wrapping).
      const newReadCur = (readCur + totalLen) % bodyLen;
      Atomics.store(headerI32, READ_CUR_IDX, newReadCur);

      // Wake the WASM producer if it is sleeping in wait_for_space.
      // The producer waits on read_cur, not seq.
      Atomics.notify(headerI32, READ_CUR_IDX);
    }

    if (signal.aborted) break;

    // -----------------------------------------------------------------------
    // Wait phase: sleep until the producer increments seq.
    // The timeout is READER_WAIT_TIMEOUT_MS (see that constant for rationale) —
    // short enough to hide lost cross-worker notifies, long enough that the
    // idle-polling cost is negligible. A reliable notify would wake us much
    // faster than the timeout; the timeout only dominates when the wake
    // was dropped by the browser's waitAsync delivery path.
    // -----------------------------------------------------------------------
    const currentSeq = Atomics.load(headerI32, SEQ_IDX);
    if (currentSeq === lastSeq) {
      const { async, value } = Atomics.waitAsync(headerI32, SEQ_IDX, lastSeq, READER_WAIT_TIMEOUT_MS);
      // value is a Promise<string> when async=true, or a string when async=false
      // ("not-equal" means seq already changed before the wait).
      const _result = async ? await value : (value as string);
      // Any result ("ok", "not-equal", "timed-out") → loop back and re-drain.
    }
    lastSeq = Atomics.load(headerI32, SEQ_IDX);
  }
}

// ---------------------------------------------------------------------------
// Ring copy helper
// ---------------------------------------------------------------------------

/**
 * Copy `count` bytes from the ring body starting at body offset `srcStart`
 * into `dst` at `dstStart`, handling a single wrap at `bodyLen`.
 */
function copyFromRing(
  body: Uint8Array,
  srcStart: number,
  bodyLen: number,
  dst: Uint8Array,
  dstStart: number,
  count: number,
): void {
  if (srcStart + count <= bodyLen) {
    // Contiguous: the range fits without wrapping.
    dst.set(body.subarray(srcStart, srcStart + count), dstStart);
  } else {
    // Wrapped: two segments.
    const firstLen = bodyLen - srcStart; // bytes until the end of the body
    dst.set(body.subarray(srcStart, bodyLen), dstStart);
    dst.set(body.subarray(0, count - firstLen), dstStart + firstLen);
  }
}

// ---------------------------------------------------------------------------
// Record dispatch
// ---------------------------------------------------------------------------

function dispatchRecord(
  kind: number,
  flags: number,
  payload: Uint8Array,
  onChunk: (chunk: ChatStreamChunk) => void,
  onError: (e: Error) => void,
): void {
  if (kind === KIND_TEXT) {
    // Fast path: text + optional is_reasoning flag. All other fields omitted.
    const text = utf8.decode(payload);
    onChunk({
      text,
      done: false,
      isReasoning: (flags & FLAG_IS_REASONING) !== 0,
    });
  } else if (kind === KIND_JSON) {
    // Structured chunk: parse JSON and remap snake_case → camelCase.
    const raw = JSON.parse(utf8.decode(payload)) as Record<string, unknown>;
    onChunk(jsonToChunk(raw));
  } else if (kind === KIND_ERROR) {
    // Error record: payload is a UTF-8 message.
    const msg = utf8.decode(payload);
    onError(new Error(msg));
  } else {
    onError(new Error(`chat-stream-sab: unknown record kind ${kind}`));
  }
}

// ---------------------------------------------------------------------------
// JSON → ChatStreamChunk conversion (snake_case → camelCase)
// ---------------------------------------------------------------------------

/**
 * Convert a raw WireChunk JSON object (snake_case Rust serde) to the TS
 * ChatStreamChunk interface (camelCase).
 *
 * Only the fields that Rust snake_case-encodes need explicit remapping:
 * finish_reason, tool_calls, num_tokens, raw_text, is_reasoning.
 * The performance sub-object also needs remapping: ttft_ms,
 * prefill_tokens_per_second, decode_tokens_per_second.
 *
 * Fields text, done, thinking are unchanged.
 */
function jsonToChunk(j: Record<string, unknown>): ChatStreamChunk {
  let performance: PerformanceMetrics | undefined;
  if (j['performance'] != null) {
    const p = j['performance'] as Record<string, unknown>;
    performance = {
      ttftMs: p['ttft_ms'] as number,
      prefillTokensPerSecond: p['prefill_tokens_per_second'] as number,
      decodeTokensPerSecond: p['decode_tokens_per_second'] as number,
    };
  }

  const chunk: ChatStreamChunk = {
    text: (j['text'] as string) ?? '',
    done: (j['done'] as boolean) ?? false,
  };

  if (j['finish_reason'] != null) chunk.finishReason = j['finish_reason'] as string;
  if (j['tool_calls'] != null) chunk.toolCalls = j['tool_calls'] as ToolCallResult[];
  if (j['thinking'] != null) chunk.thinking = j['thinking'] as string;
  if (j['num_tokens'] != null) chunk.numTokens = j['num_tokens'] as number;
  if (j['raw_text'] != null) chunk.rawText = j['raw_text'] as string;
  if (performance != null) chunk.performance = performance;
  if (j['is_reasoning'] != null) chunk.isReasoning = j['is_reasoning'] as boolean;

  if (chunk.done) {
    chunk.text = sanitizeAssistantText(chunk.text);
    if (chunk.rawText != null) {
      chunk.rawText = sanitizeAssistantText(chunk.rawText);
    }
  }

  return chunk;
}
