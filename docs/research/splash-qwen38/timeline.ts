/// <reference types="node" />

import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Per-cycle timeline of the DSpark/DFlash2 decode loop (splash-qwen38.md §6).
// Joins the `[dspark-span]` host phases (MLX_PROFILE_DECODE=1) with the
// `[metal-command]` GPU intervals (MLX_METAL_COMMAND_TRACE=1) from one stderr
// log. Both stamp std::chrono::steady_clock seconds; Metal's GPUStartTime is
// the same host clock on macOS (gpuStart - submitCpu is a few hundred µs).
const usage =
  'Usage: oxnode docs/research/splash-qwen38/timeline.ts <stderr.log> [--turn N] [--cycles A-B] [--gap-us 50] [--chrome out.json]';

export interface Command {
  id: number;
  stream: number;
  dispatches: number;
  primitives: number;
  resourceBytes: number;
  encodeStartCpu: number;
  submitCpu: number;
  gpuStart: number;
  gpuEnd: number;
}
export interface Span {
  phase: string;
  cycle: number;
  start: number;
  end: number;
}
export interface Turn {
  label: string;
  spans: Span[];
  marks: Map<number, { depth: number; keep: number }>;
}
export interface Trace {
  commands: Command[];
  turns: Turn[];
}

export function parseLog(text: string): Trace {
  const commands: Command[] = [];
  const turns: Turn[] = [];
  for (const line of text.split('\n')) {
    if (line.startsWith('[metal-command] ')) {
      const c = JSON.parse(line.slice(16)) as Command;
      if (c.gpuEnd > c.gpuStart) commands.push(c);
    } else if (line.startsWith('[dspark-span] ')) {
      const rec = JSON.parse(line.slice(14));
      if (rec.kind === 'turn') turns.push({ label: rec.label, spans: [], marks: new Map() });
      else if (rec.kind === 'span') turns.at(-1)?.spans.push(rec);
      else if (rec.kind === 'cycle') turns.at(-1)?.marks.set(rec.cycle, { depth: rec.depth, keep: rec.keep });
    }
  }
  commands.sort((a, b) => a.gpuStart - b.gpuStart);
  return { commands, turns };
}

const CYCLE = 'dspark_cycle';
const overlap = (a0: number, a1: number, b0: number, b1: number) => Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
const ms = (s: number) => s * 1000;

/** Union length of `[start,end]` intervals clipped to `[lo,hi]`, plus the idle gaps inside the window. */
function union(intervals: [number, number][], lo: number, hi: number) {
  const sorted = intervals
    .map(([s, e]) => [Math.max(s, lo), Math.min(e, hi)] as [number, number])
    .filter(([s, e]) => e > s)
    .sort((a, b) => a[0] - b[0]);
  let busy = 0;
  let cursor = lo;
  const gaps: [number, number][] = [];
  for (const [s, e] of sorted) {
    if (s > cursor) gaps.push([cursor, s]);
    busy += Math.max(0, e - Math.max(s, cursor));
    cursor = Math.max(cursor, e);
  }
  if (hi > cursor) gaps.push([cursor, hi]);
  return { busy, gaps };
}

export interface PhaseRow {
  hostMs: number;
  gpuEncodedMs: number;
  gpuOverlapMs: number;
  dispatches: number;
  commands: number;
}
export interface CycleRow {
  cycle: number;
  keep: number;
  depth: number;
  totalMs: number;
  gpuBusyMs: number;
  gpuEncodedMs: number;
  hostOnlyMs: number;
  gaps: { startMs: number; ms: number; phase: string }[];
  phases: Map<string, PhaseRow>;
}

export function analyzeCycle(turn: Turn, commands: Command[], cycle: number, gapUs: number): CycleRow | undefined {
  const outer = turn.spans.find((s) => s.cycle === cycle && s.phase === CYCLE);
  if (!outer) return undefined;
  const inner = turn.spans.filter((s) => s.cycle === cycle && s.phase !== CYCLE);
  const inWindow = commands.filter((c) => c.gpuEnd > outer.start && c.gpuStart < outer.end);
  const encoded = commands.filter((c) => c.encodeStartCpu >= outer.start && c.encodeStartCpu < outer.end);
  const phases = new Map<string, PhaseRow>();
  const row = (phase: string) => {
    let r = phases.get(phase);
    if (!r) phases.set(phase, (r = { hostMs: 0, gpuEncodedMs: 0, gpuOverlapMs: 0, dispatches: 0, commands: 0 }));
    return r;
  };
  for (const s of inner) {
    const r = row(s.phase);
    r.hostMs += ms(s.end - s.start);
    for (const c of inWindow) r.gpuOverlapMs += ms(overlap(c.gpuStart, c.gpuEnd, s.start, s.end));
  }
  const phaseAt = (t: number) => inner.find((s) => t >= s.start && t < s.end)?.phase ?? '(between phases)';
  for (const c of encoded) {
    const r = row(phaseAt(c.encodeStartCpu));
    r.gpuEncodedMs += ms(c.gpuEnd - c.gpuStart);
    r.dispatches += c.dispatches;
    r.commands += 1;
  }
  const gpu = (cs: Command[]): [number, number][] => cs.map((c) => [c.gpuStart, c.gpuEnd]);
  const { busy, gaps } = union(gpu(inWindow), outer.start, outer.end);
  const mark = turn.marks.get(cycle) ?? { depth: 0, keep: 1 };
  return {
    cycle,
    ...mark,
    totalMs: ms(outer.end - outer.start),
    gpuBusyMs: ms(busy),
    gpuEncodedMs: ms(union(gpu(encoded), -Infinity, Infinity).busy),
    hostOnlyMs: ms(outer.end - outer.start - busy),
    gaps: gaps
      .filter(([s, e]) => e - s > gapUs / 1e6)
      .map(([s, e]) => ({ startMs: ms(s - outer.start), ms: ms(e - s), phase: phaseAt((s + e) / 2) })),
    phases,
  };
}

const quantile = (xs: number[], q: number) => {
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] : 0;
};
const fmt = (x: number, w = 8) => x.toFixed(2).padStart(w);

export function report(turn: Turn, rows: CycleRow[]) {
  const names = [...new Set(rows.flatMap((r) => [...r.phases.keys()]))];
  const pick = (f: (r: CycleRow) => number) => rows.map(f);
  const out: string[] = [];
  out.push(`turn "${turn.label}": cycles ${rows[0].cycle}-${rows.at(-1)!.cycle} (${rows.length})`);
  out.push('');
  out.push('per cycle (ms):');
  out.push('  cycle keep depth    total  gpuBusy  gpuEnc hostOnly  gaps>thr: n/sum, largest 3 (ms@start phase)');
  for (const r of rows) {
    const top = [...r.gaps].sort((x, y) => y.ms - x.ms).slice(0, 3);
    const sum = r.gaps.reduce((a, g) => a + g.ms, 0);
    const gaps = `${r.gaps.length}/${sum.toFixed(2)}  ${top.map((g) => `${g.ms.toFixed(2)}@${g.startMs.toFixed(1)} ${g.phase}`).join(', ')}`;
    out.push(
      `  ${String(r.cycle).padStart(5)} ${String(r.keep).padStart(4)} ${String(r.depth).padStart(5)} ${fmt(r.totalMs)} ${fmt(r.gpuBusyMs)} ${fmt(r.gpuEncodedMs, 7)} ${fmt(r.hostOnlyMs)}  ${gaps}`,
    );
  }
  out.push('');
  out.push('per phase, median [p90] over the range (ms unless noted):');
  out.push('  phase                     host             gpu(encoded)     gpu(overlap)     dispatches  cmdbufs');
  const sums = { hostMs: 0, gpuEncodedMs: 0, gpuOverlapMs: 0, dispatches: 0, commands: 0 };
  for (const name of names) {
    const col = (k: keyof PhaseRow) => pick((r) => r.phases.get(name)?.[k] ?? 0);
    const med = (k: keyof PhaseRow) => quantile(col(k), 0.5);
    const cell = (k: keyof PhaseRow) => `${fmt(med(k))} [${fmt(quantile(col(k), 0.9), 7)}]`;
    for (const k of Object.keys(sums) as (keyof PhaseRow)[]) sums[k] += med(k);
    out.push(
      `  ${name.padEnd(24)} ${cell('hostMs')} ${cell('gpuEncodedMs')} ${cell('gpuOverlapMs')} ${fmt(med('dispatches'), 10)} ${fmt(med('commands'), 8)}`,
    );
  }
  out.push(
    `  ${'sum of medians'.padEnd(24)} ${fmt(sums.hostMs)}           ${fmt(sums.gpuEncodedMs)}           ${fmt(sums.gpuOverlapMs)}           ${fmt(sums.dispatches, 10)} ${fmt(sums.commands, 8)}`,
  );
  out.push('');
  out.push('summary over the range (ms): median [p90]');
  const cells: [string, (r: CycleRow) => number][] = [
    ['cycle total', (r) => r.totalMs],
    ['gpu busy (window union)', (r) => r.gpuBusyMs],
    ['gpu busy (encoded union)', (r) => r.gpuEncodedMs],
    ['host only (gpu idle)', (r) => r.hostOnlyMs],
    ['idle gaps > thr', (r) => r.gaps.reduce((a, g) => a + g.ms, 0)],
    ['keep (tokens)', (r) => r.keep],
  ];
  for (const [name, f] of cells)
    out.push(`  ${name.padEnd(26)} ${fmt(quantile(pick(f), 0.5))} [${fmt(quantile(pick(f), 0.9), 7)}]`);
  return out.join('\n');
}

export function chromeTrace(turn: Turn, commands: Command[], rows: CycleRow[]) {
  const lo = turn.spans.find((s) => s.cycle === rows[0].cycle && s.phase === CYCLE)!.start;
  const hi = turn.spans.find((s) => s.cycle === rows.at(-1)!.cycle && s.phase === CYCLE)!.end;
  const slice = (tid: number, name: string, start: number, end: number, args: object) => ({
    ph: 'X',
    pid: 1,
    tid,
    name,
    ts: (start - lo) * 1e6,
    dur: (end - start) * 1e6,
    args,
  });
  const events: object[] = [
    { ph: 'M', pid: 1, tid: 1, name: 'thread_name', args: { name: 'host phases' } },
    { ph: 'M', pid: 1, tid: 2, name: 'thread_name', args: { name: 'gpu command buffers' } },
  ];
  for (const s of turn.spans) {
    if (s.start < lo || s.end > hi) continue;
    const mark = s.phase === CYCLE ? turn.marks.get(s.cycle) : undefined;
    const name = s.phase === CYCLE ? `cycle ${s.cycle}` : s.phase;
    events.push(slice(1, name, s.start, s.end, { cycle: s.cycle, ...mark }));
  }
  for (const c of commands) {
    if (c.gpuEnd < lo || c.gpuStart > hi) continue;
    const { id, stream, dispatches, primitives, resourceBytes } = c;
    const args = {
      id,
      stream,
      dispatches,
      primitives,
      resourceBytes,
      encodeToGpuStartMs: ms(c.gpuStart - c.encodeStartCpu),
    };
    events.push(slice(2, `cb ${id} (${dispatches} disp)`, c.gpuStart, c.gpuEnd, args));
  }
  return { traceEvents: events, displayTimeUnit: 'ms' };
}

function main(argv: string[]) {
  const [file, ...rest] = argv;
  if (!file) throw new Error(usage);
  const opt = (name: string) => {
    const i = rest.indexOf(name);
    return i >= 0 ? rest[i + 1] : undefined;
  };
  const trace = parseLog(readFileSync(resolve(file), 'utf8'));
  if (!trace.turns.length) throw new Error('no [dspark-span] lines: run with MLX_PROFILE_DECODE=1');
  if (!trace.commands.length) throw new Error('no [metal-command] lines: run with MLX_METAL_COMMAND_TRACE=1');
  const turn = trace.turns[opt('--turn') === undefined ? trace.turns.length - 1 : Number(opt('--turn'))];
  if (!turn) throw new Error(`turn out of range (0-${trace.turns.length - 1})`);
  const cycles = [...new Set(turn.spans.filter((s) => s.phase === CYCLE).map((s) => s.cycle))];
  // Default: drop the first and last cycle (prefill drain, stop handling).
  const range = opt('--cycles')?.split('-').map(Number);
  const a = range?.[0] ?? cycles[1] ?? 0;
  const b = range?.[1] ?? cycles.at(-2) ?? 0;
  const gapUs = Number(opt('--gap-us') ?? 50);
  const rows = cycles
    .filter((c) => c >= a && c <= b)
    .flatMap((c) => analyzeCycle(turn, trace.commands, c, gapUs) ?? []);
  if (!rows.length) throw new Error(`no cycles in ${a}-${b} (turn has ${cycles[0]}-${cycles.at(-1)})`);
  // Same clock when the smallest queue lag is a few µs and never negative.
  const lag = trace.commands.map((c) => ms(c.gpuStart - c.submitCpu));
  console.log(
    `clock check: gpuStart - submitCpu min ${Math.min(...lag).toFixed(3)} ms, median ${quantile(lag, 0.5).toFixed(3)} ms (queue depth)`,
  );
  console.log(report(turn, rows));
  const chrome = opt('--chrome');
  if (chrome) {
    writeFileSync(resolve(chrome), JSON.stringify(chromeTrace(turn, trace.commands, rows)));
    console.log(`wrote ${chrome} (open in https://ui.perfetto.dev)`);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main(process.argv.slice(2));
