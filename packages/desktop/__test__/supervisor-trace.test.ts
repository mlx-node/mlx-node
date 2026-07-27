import { appendFileSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  parseNativeErrorLine,
  parseSwallowedStderrLine,
  watchTraceFile,
  type SwallowedError,
  type TraceWatcher,
} from '../src/main/supervisor/trace.js';

describe('parseNativeErrorLine', () => {
  // The exact line `mlx_trace_native_error` writes
  // (crates/mlx-sys/src/mlx_common.h): the detail has already had every \n, \r
  // and " flattened to a space by the writer, so there is nothing to unescape.
  it('reads the line the C++ writer emits', () => {
    expect(parseNativeErrorLine('native_error context=array_eval detail="[metal::malloc] out of memory"')).toEqual({
      channel: 'trace',
      context: 'array_eval',
      detail: '[metal::malloc] out of memory',
    });
    expect(parseNativeErrorLine('native_error context=async_eval detail="unknown exception"')).toMatchObject({
      context: 'async_eval',
    });
    expect(parseNativeErrorLine('native_error context=paged_attention_forward detail=""')).toMatchObject({
      context: 'paged_attention_forward',
      detail: '',
    });
  });

  // The SAME file carries the Rust-side [MLX_TRACE] firehose, and some of those
  // lines end in `error=<msg>` for an ordinary, handled paged fallback. A
  // detector that looked for the substring "error" would call a routine
  // fallback silent corruption, and `lying` would stop meaning anything.
  it('ignores the [MLX_TRACE] firehose sharing the same file', () => {
    for (const line of [
      '[MLX_TRACE] weight_materialize_start arrays=284 total_gb=1.40 budget_mb=860',
      '[MLX_TRACE] gemma4 attention_paged_kv_write_fallback paged_idx=0 first_position=0 seq_len=8 error=shape mismatch',
      '[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=fresh cached_prefix_tokens=0 elapsed_ms=1.2',
    ]) {
      expect(parseNativeErrorLine(line)).toBeNull();
    }
  });

  it('refuses a line that merely contains the marker', () => {
    expect(parseNativeErrorLine('[MLX_TRACE] note native_error context=eval detail="x"')).toBeNull();
    expect(parseNativeErrorLine('native_error context=eval detail="x" trailing')).toBeNull();
    expect(parseNativeErrorLine('native_error context=eval')).toBeNull();
    expect(parseNativeErrorLine('')).toBeNull();
  });
});

describe('parseSwallowedStderrLine', () => {
  // The second channel, and the only one that survives a misconfigured fork:
  // mlx_synchronize / mlx_stream_synchronize / mlx_clear_cache and the array
  // destructor write to stderr unconditionally, with no env var involved.
  it('reads both catch arms of the void shims', () => {
    expect(parseSwallowedStderrLine('[MLX] Exception in synchronize: [metal] command buffer failed')).toEqual({
      channel: 'stderr',
      context: 'synchronize',
      detail: '[metal] command buffer failed',
    });
    expect(parseSwallowedStderrLine('[MLX] Unknown exception in clear_cache')).toMatchObject({
      context: 'clear_cache',
      detail: 'Unknown exception',
    });
    expect(parseSwallowedStderrLine('[MLX] Exception during array delete: bad free')).toMatchObject({
      context: 'array delete',
      detail: 'bad free',
    });
    expect(parseSwallowedStderrLine('[MLX] Unknown exception in stream_synchronize')).toMatchObject({
      context: 'stream_synchronize',
    });
  });

  // The [MLX] stderr family is much larger than the swallow set. The rest of
  // it returns a status or a sentinel the Rust caller checks, so the user has
  // already seen an error — flagging those would make `lying` cry wolf.
  it('does not flag shims that report their failure to the caller', () => {
    for (const line of [
      '[MLX] Exception in compile_clear_cache: nope',
      '[MLX] Exception in get_active_memory: nope',
      '[MLX] mlx_safetensor_read_raw: cannot open /models/x.safetensors',
      '[MLX] quantize returned unexpected number of arrays: 2',
      '[MLX] Exception in mlx_quantize: bad shape',
      'some unrelated log line',
      '',
    ]) {
      expect(parseSwallowedStderrLine(line)).toBeNull();
    }
  });
});

describe('watchTraceFile', () => {
  let dir: string;
  let file: string;
  let seen: SwallowedError[];
  let watcher: TraceWatcher | null;

  const NATIVE_ERROR = 'native_error context=array_eval detail="out of memory"';

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-trace-watch-'));
    file = join(dir, 'inference-1.trace.log');
    seen = [];
    watcher = null;
  });

  afterEach(() => {
    watcher?.close();
    rmSync(dir, { recursive: true, force: true });
  });

  function start(): TraceWatcher {
    watcher = watchTraceFile(file, { intervalMs: 60_000, onError: (error) => seen.push(error) });
    return watcher;
  }

  // The child creates the file lazily, on its first trace write. A watcher that
  // threw on the missing file would take the supervisor down on every healthy
  // start-up.
  it('tolerates a file that does not exist yet', async () => {
    const w = start();
    await w.poll();
    expect(seen).toEqual([]);
    writeFileSync(file, `${NATIVE_ERROR}\n`);
    await w.poll();
    expect(seen).toHaveLength(1);
  });

  it('reports only what is new since the last scan', async () => {
    writeFileSync(file, `${NATIVE_ERROR}\n`);
    const w = start();
    await w.poll();
    await w.poll();
    expect(seen).toHaveLength(1);
    appendFileSync(file, `${NATIVE_ERROR}\n`);
    await w.poll();
    expect(seen).toHaveLength(2);
  });

  // Two independent appenders in another process — Rust's OpenOptions(append)
  // and C++'s ofstream(ios::app), each reopening per write — so a scan lands
  // mid-line routinely. Parsing the tail as though it were complete drops the
  // error it was in the middle of reporting AND corrupts the next one.
  it('holds a partial trailing line until its newline arrives', async () => {
    const w = start();
    writeFileSync(file, '[MLX_TRACE] weight_materialize_start arrays=1\nnative_error context=array_ev');
    await w.poll();
    expect(seen).toEqual([]);
    appendFileSync(file, 'al detail="out of memory"\n');
    await w.poll();
    expect(seen).toEqual([{ channel: 'trace', context: 'array_eval', detail: 'out of memory' }]);
  });

  // e.what() carries model paths, which are not always ASCII, and a read
  // window can split a multi-byte sequence down the middle.
  //
  // 35 is not arbitrary: the line is 48 bytes and `模` occupies bytes 34-36, so
  // a 35-byte window cuts it after its FIRST byte. A round number like 20
  // happens to land on a character boundary, and the test would then pass with
  // the decoder replaced by a plain `toString('utf8')` — verified, it does.
  it('survives a read boundary inside a multi-byte character', async () => {
    writeFileSync(file, `native_error context=eval detail="模型 failed"\n`);
    watcher = watchTraceFile(file, {
      intervalMs: 60_000,
      maxBytesPerScan: 35,
      onError: (error) => seen.push(error),
    });
    for (let i = 0; i < 10; i += 1) await watcher.poll();
    expect(seen).toEqual([{ channel: 'trace', context: 'eval', detail: '模型 failed' }]);
  });

  // A file replaced underneath us (a stale path reused, an editor rewriting it)
  // leaves the offset pointing into the middle of different content.
  it('restarts from the beginning when the file shrinks', async () => {
    writeFileSync(file, `${NATIVE_ERROR}\n${NATIVE_ERROR}\n`);
    const w = start();
    await w.poll();
    expect(seen).toHaveLength(2);
    writeFileSync(file, `${NATIVE_ERROR}\n`);
    await w.poll();
    expect(seen).toHaveLength(3);
  });

  it('reads a burst larger than one scan window across successive scans', async () => {
    writeFileSync(file, `${NATIVE_ERROR}\n`.repeat(50));
    watcher = watchTraceFile(file, {
      intervalMs: 60_000,
      maxBytesPerScan: 64,
      onError: (error) => seen.push(error),
    });
    for (let i = 0; i < 100; i += 1) await watcher.poll();
    expect(seen).toHaveLength(50);
  });

  it('stops reporting once closed', async () => {
    const w = start();
    w.close();
    writeFileSync(file, `${NATIVE_ERROR}\n`);
    await w.poll();
    expect(seen).toEqual([]);
  });
});
