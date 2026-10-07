import { describe, expect, it } from 'vite-plus/test';

import { analyzeCycle, chromeTrace, parseLog, report } from '../../docs/research/splash-qwen38/timeline';

// Synthetic log: one turn, two cycles on a steady clock starting at t=100 s.
// Cycle 1: propose 100.000-100.002 encodes cb0 (GPU 100.0015-100.0040),
//          verify 100.002-100.003 (no commands), accept 100.003-100.010
//          encodes cb1 (GPU 100.0045-100.0090) so the GPU idles 0.5 ms between
//          cb0 and cb1 inside accept; eval_boundary 100.0095-100.011; cycle ends
//          100.011 → tail idle gap 2 ms.
// Cycle 2: a plain copy 1 s later with one command only. cb9 has no
//          encoder (encodeStartCpu 0, gpuEnd == gpuStart) and is dropped.
const cb = (id: number, enc: number, gs: number, ge: number, dispatches = 10) =>
  `[metal-command] {"id":${id},"stream":1,"reason":"finalize","dispatches":${dispatches},"primitives":${dispatches + 1},"resources":1,"resourceBytes":64,"sizeUnits":1,"barriers":0,"encoders":1,"waits":0,"signals":0,"encodeStartCpu":${enc.toFixed(9)},"submitCpu":${(enc + 0.0002).toFixed(9)},"gpuStart":${gs.toFixed(9)},"gpuEnd":${ge.toFixed(9)},"status":4}`;
const span = (phase: string, cycle: number, start: number, end: number) =>
  `[dspark-span] {"kind":"span","phase":"${phase}","cycle":${cycle},"start":${start.toFixed(9)},"end":${end.toFixed(9)}}`;
const cycle = (base: number, n: number, withVerifyCommand: boolean) =>
  [
    cb(n * 10, base + 0.0005, base + 0.0015, base + 0.004, 4),
    ...(withVerifyCommand ? [cb(n * 10 + 1, base + 0.0035, base + 0.0045, base + 0.009, 20)] : []),
    span('dspark_propose', n, base, base + 0.002),
    span('dspark_verify', n, base + 0.002, base + 0.003),
    span('dspark_accept', n, base + 0.003, base + 0.0095),
    span('dspark_eval_boundary', n, base + 0.0095, base + 0.011),
    span('dspark_cycle', n, base, base + 0.011),
    `[dspark-span] {"kind":"cycle","cycle":${n},"depth":7,"keep":${n + 1}}`,
  ].join('\n');
const log = [
  '[dspark-span] {"kind":"turn","label":"warmup","model":"qwen3_5","cycles":0,"spans":0}',
  '[PROFILE] noise line',
  '[dspark-span] {"kind":"turn","label":"chat","model":"qwen3_5","cycles":2,"spans":10}',
  cycle(100, 1, true),
  cycle(101, 2, false),
  cb(9, 0, 101.5, 101.5, 0),
  '[mlx-evaluation] {"stream":1,"async":true}',
].join('\n');

describe('dspark timeline', () => {
  const trace = parseLog(log);
  const turn = trace.turns[1];

  it('parses commands and spans into turns', () => {
    expect(trace.turns.map((t) => t.label)).toEqual(['warmup', 'chat']);
    expect(trace.commands.map((c) => c.id)).toEqual([10, 11, 20]);
    expect(turn.spans).toHaveLength(10);
    expect(turn.marks.get(2)).toEqual({ depth: 7, keep: 3 });
  });

  it('attributes GPU time by encode phase and by overlap, and finds idle gaps', () => {
    const row = analyzeCycle(turn, trace.commands, 1, 50)!;
    expect(row.keep).toBe(2);
    expect(row.totalMs).toBeCloseTo(11, 6);
    // cb0 2.5 ms + cb1 4.5 ms, no overlap between them.
    expect(row.gpuBusyMs).toBeCloseTo(7, 6);
    expect(row.gpuEncodedMs).toBeCloseTo(7, 6);
    expect(row.hostOnlyMs).toBeCloseTo(4, 6);
    const propose = row.phases.get('dspark_propose')!;
    expect(propose.gpuEncodedMs).toBeCloseTo(2.5, 6);
    expect(propose.dispatches).toBe(4);
    expect(propose.commands).toBe(1);
    // GPU execution of cb0 overlaps propose only for 100.0015-100.002.
    expect(propose.gpuOverlapMs).toBeCloseTo(0.5, 6);
    expect(row.phases.get('dspark_verify')!.gpuOverlapMs).toBeCloseTo(1, 6);
    expect(row.phases.get('dspark_verify')!.gpuEncodedMs).toBe(0);
    const accept = row.phases.get('dspark_accept')!;
    expect(accept.gpuEncodedMs).toBeCloseTo(4.5, 6);
    expect(accept.gpuOverlapMs).toBeCloseTo(1 + 4.5, 6);
    expect(row.gaps.map((g) => [g.phase, +g.ms.toFixed(3), +g.startMs.toFixed(3)])).toEqual([
      ['dspark_propose', 1.5, 0],
      ['dspark_accept', 0.5, 4],
      ['dspark_eval_boundary', 2, 9],
    ]);
    expect(analyzeCycle(turn, trace.commands, 3, 50)).toBeUndefined();
  });

  it('reports medians and exports a two-track Chrome trace', () => {
    const rows = [1, 2].map((c) => analyzeCycle(turn, trace.commands, c, 50)!);
    const text = report(turn, rows);
    expect(text).toContain('turn "chat": cycles 1-2 (2)');
    expect(text).toMatch(/dspark_propose\s+2\.00 \[\s+2\.00\]\s+2\.50 \[\s+2\.50\]/);
    expect(text).toMatch(/cycle total\s+11\.00 \[\s+11\.00\]/);
    const chrome = chromeTrace(turn, trace.commands, rows);
    const events = chrome.traceEvents as { ph: string; tid: number; name: string; ts?: number; dur?: number }[];
    expect(events.filter((e) => e.ph === 'M')).toHaveLength(2);
    expect(events.filter((e) => e.tid === 1 && e.ph === 'X')).toHaveLength(10);
    expect(events.filter((e) => e.tid === 2 && e.ph === 'X')).toHaveLength(3);
    const first = events.find((e) => e.name === 'cycle 1')!;
    expect(first.ts).toBeCloseTo(0, 3);
    expect(first.dur).toBeCloseTo(11000, 3);
  });
});
