// Analyze command-buffer intervals without mixing CPU and GPU clock domains.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

export function intervalSummary(intervals) {
  const sorted = intervals.map((pair) => [...pair]).sort((a, b) => a[0] - b[0]);
  if (!sorted.length) return { spanMs: 0, unionMs: 0, gapMs: 0 };
  let [start, end] = sorted[0];
  const first = start;
  let union = 0;
  for (const [nextStart, nextEnd] of sorted.slice(1)) {
    assert(nextEnd >= nextStart);
    if (nextStart > end) {
      union += end - start;
      start = nextStart;
      end = nextEnd;
    } else end = Math.max(end, nextEnd);
  }
  union += end - start;
  const span = end - first;
  return { spanMs: span * 1000, unionMs: union * 1000, gapMs: Math.max(0, span - union) * 1000 };
}

assert.deepEqual(
  intervalSummary([
    [1, 2],
    [1.5, 3],
    [4, 5],
  ]),
  { spanMs: 4000, unionMs: 3000, gapMs: 1000 },
);
assert.deepEqual(intervalSummary([]), { spanMs: 0, unionMs: 0, gapMs: 0 });

const [input, output] = process.argv.slice(2);
assert(input && output, 'usage: node analyze-command-trace.mjs TRACE_LOG OUTPUT_JSON');
const commands = [];
const evaluations = [];
const compiled = [];
const windows = [];
let current = null;
for (const line of readFileSync(input, 'utf8').split('\n')) {
  const match = /^\[(metal-command|mlx-evaluation|mlx-compiled)\] (\{.*\})$/.exec(line);
  if (match) {
    const row = JSON.parse(match[2]);
    if (match[1] === 'metal-command') {
      assert.equal(row.status, 4, 'GPU command did not complete successfully');
      commands.push(row);
      current?.commands.push(row);
    } else if (match[1] === 'mlx-evaluation') evaluations.push(row);
    else compiled.push(row);
    continue;
  }
  if (!line.startsWith('{')) continue;
  let row;
  try {
    row = JSON.parse(line);
  } catch {
    continue;
  }
  if (row.event === 'benchmark-start') {
    current = { name: row.name, run: row.run, commands: [] };
    windows.push(current);
  } else if (row.name && row.performance) {
    if (current) current.result = row;
    current = null;
  }
}
assert(commands.length, 'no command trace records');
assert.equal(new Set(commands.map((row) => row.id)).size, commands.length, 'duplicate command IDs');
function summarize(rows) {
  const timed = rows.filter((row) => row.gpuStart > 0 && row.gpuEnd >= row.gpuStart);
  const reasons = {};
  for (const row of rows) reasons[row.reason] = (reasons[row.reason] ?? 0) + 1;
  return {
    commands: rows.length,
    dispatches: rows.reduce((sum, row) => sum + row.dispatches, 0),
    barriers: rows.reduce((sum, row) => sum + row.barriers, 0),
    noDispatchCommands: rows.filter((row) => row.dispatches === 0).length,
    reasons,
    gpuIntervals: intervalSummary(timed.map((row) => [row.gpuStart, row.gpuEnd])),
  };
}
const result = {
  input,
  caveats: [
    'Opt-in diagnostic adds CPU logging/callback work; this is not an uninstrumented throughput cohort.',
    'GPU interval union measures command envelopes, not isolated shader execution. Gaps include legitimate CPU work between requests.',
    'Workload windows use log markers and callback arrival order; deferred or late completion records may fall outside them.',
    'Host evaluation spans include scheduler backpressure; they are not pure CPU time. Compiled spans include first tracing/compilation.',
    'CPU and GPU absolute timestamps are never subtracted from one another.',
  ],
  all: summarize(commands),
  evaluations,
  compiled,
  windows: windows.map(({ commands: rows, ...rest }) => ({ ...rest, summary: summarize(rows) })),
};
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(
  JSON.stringify(
    { all: result.all, windows: result.windows.map(({ name, run, summary }) => ({ name, run, summary })) },
    null,
    2,
  ),
);
