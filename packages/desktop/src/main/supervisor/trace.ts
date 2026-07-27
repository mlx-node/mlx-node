/**
 * Detection of class-(b) failures: a C++ exception caught inside the FFI shim,
 * swallowed, and returned from as if nothing happened.
 *
 * This is the third state's whole reason for existing. A wrapped error (class a)
 * becomes a JS `Error` and an HTTP 500; an unwrapped one (class c) unwinds
 * across `extern "C"` and aborts the process, which the exit code shows. Class
 * (b) does neither: no JS error, no HTTP error, `/health` says `ok`, and the
 * arrays the caller reads hold whatever was in the buffer. The user sees
 * plausible, wrong output.
 *
 * There are TWO swallow channels in the tree, not one, and they need different
 * detectors:
 *
 *   A. `mlx_trace_native_error` (crates/mlx-sys/src/mlx_common.h) — appends one
 *      line to `MLX_INFERENCE_TRACE_FILE`, and ONLY when both that and
 *      `MLX_INFERENCE_TRACE` are set. Covers the eval paths: `array_eval`,
 *      `async_eval`, `eval`, `ensure_readable`'s scalar reads,
 *      `fast_scaled_dot_product_attention`, `paged_attention_*`.
 *
 *   B. `std::cerr << "[MLX] Exception in ..."` (mlx_misc_ops.cpp,
 *      mlx_stream.cpp, mlx_nn_ops.cpp) — always on, no env var needed, and
 *      therefore the only class-(b) signal that survives a misconfigured fork.
 *      Covers `synchronize`, `stream_synchronize`, `clear_cache` and the array
 *      destructor.
 *
 * Both feed the same `lying` state.
 */

import { open, stat } from 'node:fs/promises';
import { StringDecoder } from 'node:string_decoder';

/** A swallowed native error, from either channel. */
export interface SwallowedError {
  /** Which detector fired. */
  channel: 'trace' | 'stderr';
  /** The shim that swallowed it, e.g. `array_eval` or `synchronize`. */
  context: string;
  /** `e.what()`, or a fixed string when the `catch (...)` arm ran. */
  detail: string;
}

/**
 * Parse one line of the trace file.
 *
 * The writer emits exactly `native_error context=<ctx> detail="<what>"`, having
 * already replaced every `\n`, `\r` and `"` in the detail with a space — so
 * there is no escaping to undo and the closing quote is unambiguous.
 *
 * Anchored at both ends deliberately. The same file carries the `[MLX_TRACE]`
 * firehose from the Rust side, which includes lines such as
 * `[MLX_TRACE] gemma4 attention_paged_kv_write_fallback ... error=<msg>`. A
 * detector that looked for the substring `error` would flag a routine paged
 * fallback as silent corruption, and `lying` would stop meaning anything.
 */
export function parseNativeErrorLine(line: string): SwallowedError | null {
  const match = /^native_error context=(\S+) detail="([^"]*)"$/.exec(line);
  if (match === null) return null;
  return { channel: 'trace', context: match[1]!, detail: match[2]! };
}

/**
 * Shims whose `catch` arm provably returns nothing to the caller — `void`
 * functions and one destructor. An exception here is invisible to JS by
 * construction.
 *
 * The `[MLX]` stderr family is larger than this, but the rest of it
 * (`mlx_compile_clear_cache`, the memory getters, `mlx_safetensor_read_raw`,
 * ...) returns a status or a sentinel the Rust caller checks, so those are
 * reported failures rather than silent ones. Flagging them would make `lying`
 * fire on conditions the user would already have seen as an error, and a state
 * that cries wolf is a state people learn to ignore.
 *
 * Extend this only after checking the shim's return type in
 * crates/mlx-sys/src/.
 */
const SILENT_STDERR_CONTEXTS: ReadonlySet<string> = new Set([
  'synchronize',
  'stream_synchronize',
  'clear_cache',
  'array delete',
]);

/**
 * Parse one line of the child's stderr.
 *
 * Two shapes, from the `catch (const std::exception&)` and `catch (...)` arms:
 *
 *     [MLX] Exception in synchronize: <what>
 *     [MLX] Unknown exception in clear_cache
 *     [MLX] Exception during array delete: <what>
 *     [MLX] Unknown exception during array delete
 */
export function parseSwallowedStderrLine(line: string): SwallowedError | null {
  const match = /^\[MLX\] (Unknown exception|Exception) (?:in|during) ([^:]+?)(?::\s*(.*))?$/.exec(line.trimEnd());
  if (match === null) return null;
  const context = match[2]!;
  if (!SILENT_STDERR_CONTEXTS.has(context)) return null;
  return { channel: 'stderr', context, detail: match[3] ?? match[1]! };
}

export interface TraceWatcherOptions {
  /** How often to scan. The supervisor's default is on the order of a second. */
  intervalMs: number;
  onError(error: SwallowedError): void;
  /**
   * Bytes to read per scan. The trace file is a firehose — a Phase 0 spike run
   * produced ~150k lines in 240 s — so a scan reads a bounded window and lets
   * the next tick continue from where it stopped rather than pulling an
   * arbitrarily large tail into memory at once.
   */
  maxBytesPerScan?: number;
}

export interface TraceWatcher {
  /**
   * Run one scan now.
   *
   * Public so a caller can force a read instead of sleeping past an interval.
   * Re-entrant calls collapse onto the in-flight scan, so the offset can never
   * be advanced twice over the same bytes.
   */
  poll(): Promise<void>;
  close(): void;
}

const DEFAULT_MAX_BYTES_PER_SCAN = 1 << 20;

/**
 * Follow an append-only trace file from wherever it is now.
 *
 * Polls rather than using `fs.watch`. The file is appended by two independent
 * writers in another process — Rust's `OpenOptions(append)` and C++'s
 * `std::ofstream(ios::app)`, each reopening per write — and macOS's watcher
 * coalesces events, so "did anything arrive" is answered more reliably by the
 * size than by a notification. A fixed cadence also bounds how much work this
 * can do on the main process's event loop, which must never block.
 */
export function watchTraceFile(path: string, opts: TraceWatcherOptions): TraceWatcher {
  const maxBytes = opts.maxBytesPerScan ?? DEFAULT_MAX_BYTES_PER_SCAN;
  let offset = 0;
  let closed = false;
  let inFlight: Promise<void> | null = null;
  // Whatever followed the last newline. Two writers appending independently
  // means a scan can land mid-line; parsing the tail as if it were complete
  // would drop the error it was in the middle of reporting.
  let pending = '';
  const decoder = new StringDecoder('utf8');

  async function scan(): Promise<void> {
    let size: number;
    try {
      size = (await stat(path)).size;
    } catch {
      // The child creates the file lazily, on its first trace write. Missing is
      // the normal state for a healthy run.
      return;
    }
    // Truncated or replaced underneath us: start over rather than read from an
    // offset that now points into the middle of different content.
    if (size < offset) {
      offset = 0;
      pending = '';
    }
    if (size === offset) return;

    const length = Math.min(size - offset, maxBytes);
    const buffer = Buffer.allocUnsafe(length);
    const handle = await open(path, 'r');
    let read: number;
    try {
      ({ bytesRead: read } = await handle.read(buffer, 0, length, offset));
    } finally {
      await handle.close();
    }
    offset += read;

    // `StringDecoder`, not `toString`: a read window can split a multi-byte
    // sequence, and `e.what()` carries model paths that are not always ASCII.
    pending += decoder.write(buffer.subarray(0, read));
    const lines = pending.split('\n');
    pending = lines.pop() ?? '';
    for (const line of lines) {
      const error = parseNativeErrorLine(line);
      if (error !== null) opts.onError(error);
    }
  }

  async function poll(): Promise<void> {
    if (closed) return;
    inFlight ??= scan().finally(() => {
      inFlight = null;
    });
    await inFlight;
  }

  const timer = setInterval(() => {
    void poll().catch(() => {
      /* a transient stat/read failure is not worth escalating; the next tick retries */
    });
  }, opts.intervalMs);
  timer.unref();

  return {
    poll,
    close(): void {
      closed = true;
      clearInterval(timer);
    },
  };
}
