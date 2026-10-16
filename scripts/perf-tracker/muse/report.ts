/// <reference types="node" />

import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import {
  ROOT,
  compareTable,
  median,
  parseCsv,
  readJson,
  renderChart,
  sh,
  sha256,
  sha256File,
  toCsv,
  writeJson,
  type CompareRow,
} from '../common.js';
import type { JobRecord } from './guard.js';
import {
  BASELINE,
  PROTOCOL,
  SEPTEMBER,
  VARIANTS,
  type Case,
  type Provenance,
  type Runtime,
  type Sample,
  type Setup,
  type Speculation,
} from './suite.js';

// The column set of the issue #142 CSV block.
const COLUMNS = [
  'id',
  'runtime',
  'speculation',
  'case',
  'run',
  'promptTokens',
  'generatedTokens',
  'cachedTokens',
  'prefillTps',
  'decodeTps',
  'wallMs',
  'prefillMs',
  'decodeMs',
  'draftAcceptance',
  'outputSha256',
];
const METRICS = ['prefillTps', 'decodeTps', 'wallMs', 'prefillMs', 'decodeMs'] as const;
const CHART_LABELS = ['mlx-node', 'llama.cpp', 'mlx-node DFlash', 'llama.cpp DFlash'];
const TUNING_FIELDS = ['timestamp', 'stage', 'context_bucket', 'early_layers', 'grouped_stripes'];

interface Distribution {
  median: number;
  min: number;
  max: number;
  samples: number[];
}

type Metric = (typeof METRICS)[number];

type Cell = {
  case: string;
  promptTokens: number;
  runtime: Runtime;
  speculation: Speculation;
  outputVariants: number;
  acceptance: Distribution | null;
} & Record<Metric, Distribution>;

const distribution = (samples: number[]): Distribution => ({
  median: median(samples),
  min: Math.min(...samples),
  max: Math.max(...samples),
  samples,
});

const tokens = (count: number) => count.toLocaleString('en-US');
const percent = (value: number) => `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
const lines = async (path: string) => (await readFile(path, 'utf8')).split('\n').filter((line) => line.length > 0);

async function loadSamples(out: string, setup: Setup): Promise<Sample[]> {
  const expected = setup.repetitions * setup.cases.length * VARIANTS.length;
  const names = (await readdir(join(out, 'raw'))).filter((name) => /-[1-3]\.json$/.test(name)).sort();
  assert.equal(names.length, expected, `Only ${names.length}/${expected} completed samples`);
  return Promise.all(names.map((name) => readJson<Sample>(join(out, 'raw', name))));
}

// Inputs and source unchanged since prepare; every sample from this setup.
async function verify(setup: Setup, setupSha256: string, rows: Sample[], archived: boolean) {
  for (const row of rows) assert.equal(row.setupSha256, setupSha256, `${row.id}: sample from another setup`);
  if (archived) return { identities: [], sourceUnchanged: null };
  for (const identity of setup.identities) {
    const current = await stat(identity.path);
    assert(
      current.size === identity.bytes && current.mtimeMs === identity.mtimeMs && current.ctimeMs === identity.ctimeMs,
      `Benchmark input changed: ${identity.path}`,
    );
  }
  assert.equal(sh('git', ['rev-parse', 'HEAD'], { cwd: ROOT }), setup.mlxCommit, 'Source HEAD moved');
  assert.equal(sha256(sh('git', ['diff', '--binary'], { cwd: ROOT })), setup.mlxPatchSha256, 'Source patch changed');
  const identities = [];
  for (const identity of setup.identities) {
    const digest = await sha256File(identity.path);
    assert.equal(digest, identity.sha256, identity.path);
    identities.push({ path: identity.path, sha256: digest });
  }
  return { identities, sourceUnchanged: true };
}

function summarize(setup: Setup, rows: Sample[]) {
  const acceptance = new Map<string, number>();
  const text = new Map<string, string>();
  let thermalClear = true;
  for (const row of rows) {
    const item = setup.cases.find((candidate) => candidate.name === row.case);
    assert(item && row.promptTokens === item.promptTokens && row.inputSha256 === item.sha256, `${row.id}: input`);
    assert(
      row.generatedTokens === row.outputTokensCap && row.outputTokensCap === setup.outputTokens,
      `${row.id}: length`,
    );
    assert.equal(row.cachedTokens, 0, `${row.id}: prompt-cache hit`);
    thermalClear &&= [row.thermalBefore, row.thermalAfter].every(
      (note) => note.includes('No thermal warning') && note.includes('No performance warning'),
    );
    assert(row.prefillMs > 0 && row.decodeTps > 0 && row.wallMs > row.prefillMs, `${row.id}: timings`);
    // Both engines' native timings count the first output token in prefill.
    assert(
      Math.abs((row.prefillTps * row.prefillMs) / 1000 - row.promptTokens) < 0.01,
      `${row.id}: prefill accounting`,
    );
    assert(
      Math.abs((row.decodeTps * row.decodeMs) / 1000 - (setup.outputTokens - 1)) < 0.01,
      `${row.id}: decode accounting`,
    );
    const { performance, timings } = row.result;
    if (row.speculation === 'dflash') {
      if (row.runtime === 'mlx') {
        const depth = performance?.mtpMeanDepth ?? 0;
        assert((performance?.mtpCycles ?? 0) > 0 && depth > 0 && depth <= setup.draftTokens, `${row.id}: drafting`);
        assert.equal(performance?.mtpAcceptanceByPosition?.length, setup.draftTokens, `${row.id}: draft width`);
        acceptance.set(row.id, (performance?.mtpMeanAcceptedTokens ?? 0) / depth);
      } else {
        const drafted = timings?.draft_n ?? 0;
        assert(drafted > 0, `${row.id}: drafting`);
        acceptance.set(row.id, (timings?.draft_n_accepted ?? 0) / drafted);
      }
    } else if (row.runtime === 'mlx') {
      assert(!performance?.mtpCycles, `${row.id}: drafted with DFlash off`);
    } else {
      assert(!timings?.draft_n, `${row.id}: drafted with DFlash off`);
    }
    const output = row.runtime === 'mlx' ? row.result.rawText : row.result.content;
    assert(typeof output === 'string', `${row.id}: no output text`);
    text.set(row.id, output);
  }
  const summary: Cell[] = [];
  for (const item of setup.cases) {
    for (const speculation of ['off', 'dflash'] as const) {
      for (const runtime of ['mlx', 'llama'] as const) {
        const samples = rows.filter(
          (row) => row.case === item.name && row.speculation === speculation && row.runtime === runtime,
        );
        assert.deepEqual(
          samples.map((row) => row.run).sort(),
          Array.from({ length: setup.repetitions }, (_, index) => String(index + 1)),
          `${item.name} ${runtime} ${speculation}: repetitions`,
        );
        const metrics = Object.fromEntries(
          METRICS.map((metric) => [metric, distribution(samples.map((row) => row[metric]))]),
        ) as Record<Metric, Distribution>;
        summary.push({
          case: item.name,
          promptTokens: item.promptTokens,
          runtime,
          speculation,
          ...metrics,
          outputVariants: new Set(samples.map((row) => row.outputSha256)).size,
          acceptance:
            speculation === 'dflash' ? distribution(samples.map((row) => acceptance.get(row.id) ?? NaN)) : null,
        });
      }
    }
  }
  const parity = setup.cases.flatMap((item) =>
    (['mlx', 'llama'] as const).map((runtime) => {
      const mine = rows.filter((row) => row.case === item.name && row.runtime === runtime);
      const hashes = (speculation: Speculation) =>
        [...new Set(mine.filter((row) => row.speculation === speculation).map((row) => row.outputSha256))].sort();
      return {
        case: item.name,
        runtime,
        sameTextAcrossAllModesAndRuns: new Set(mine.map((row) => text.get(row.id))).size === 1,
        arHashes: hashes('off'),
        dflashHashes: hashes('dflash'),
      };
    }),
  );
  const csv = toCsv(
    COLUMNS,
    rows.map((row) => ({ ...row, result: undefined, warmup: undefined, draftAcceptance: acceptance.get(row.id) })),
  );
  return { summary, parity, thermalClear, csv };
}

async function audit(out: string, setup: Setup, rows: Sample[]) {
  const shortest = setup.cases.reduce((a, b) => (a.promptTokens <= b.promptTokens ? a : b)).name;
  const activity = new Map<string, number>();
  const rss: number[] = [];
  const available: number[] = [];
  const gpuBefore: number[] = [];
  const sampleChecks = [];
  const tuning = [];
  let resourceSamples = 0;
  let resourceSamplesWithOtherActivity = 0;
  for (const row of rows) {
    const { warmup } = row;
    assert.equal(warmup.case, shortest, `${row.id}: warmup history`);
    assert(
      warmup.maxTokens === setup.warmup.maxTokens && warmup.generatedTokens === warmup.maxTokens,
      `${row.id}: warmup`,
    );
    assert(warmup.completedAt <= row.measuredAt && row.measuredAt < row.completedAt, `${row.id}: warmup order`);
    const job = await readJson<JobRecord>(join(out, `${row.id}.job.json`));
    assert(job.code === 0 && job.reason === null, `${row.id}: guarded job did not pass`);
    rss.push(job.maxRssBytes);
    const before = (job.hostBefore as { gpuUtilization?: number } | undefined)?.gpuUtilization;
    if (before !== undefined) gpuBefore.push(before);
    for (const line of await lines(join(out, `${row.id}.resources.jsonl`))) {
      const record = JSON.parse(line) as { availablePercent: number; otherActivity: { command: string }[] };
      resourceSamples += 1;
      available.push(record.availablePercent);
      if (record.otherActivity.length) resourceSamplesWithOtherActivity += 1;
      for (const entry of record.otherActivity) {
        const name = entry.command.slice(entry.command.lastIndexOf('/') + 1);
        activity.set(name, (activity.get(name) ?? 0) + 1);
      }
    }
    sampleChecks.push({ id: row.id, warmupTokens: warmup.generatedTokens, guardExitCode: job.code });
    if (row.runtime === 'mlx' && row.speculation === 'off') {
      // grouped_stripes left the decode_tuned event when partitions became a device rule.
      const events = (await lines(join(out, `${row.id}.choices.jsonl`)))
        .map((line) => JSON.parse(line) as Record<string, unknown>)
        .filter((event) => event.event === 'decode_tuned');
      tuning.push({
        id: row.id,
        measuredAt: row.measuredAt,
        completedAt: row.completedAt,
        events: events.map((event) => Object.fromEntries(TUNING_FIELDS.map((field) => [field, event[field] ?? null]))),
      });
    }
  }
  const attempts = existsSync(join(out, 'attempts')) ? (await readdir(join(out, 'attempts'))).sort() : [];
  return {
    sampleChecks,
    maxProcessTreeRssBytes: Math.max(...rss),
    minAvailablePercent: Math.min(...available),
    resourceSamples,
    resourceSamplesWithOtherActivity,
    // Stable sort keeps first-seen order among equal counts.
    activityObservations: [...activity.entries()].sort((a, b) => b[1] - a[1]),
    gpuUtilizationBeforeSamples: gpuBefore.length
      ? { min: Math.min(...gpuBefore), median: median(gpuBefore), max: Math.max(...gpuBefore) }
      : null,
    retriedAttempts: attempts,
    tuning,
  };
}

function host(setup: Setup) {
  const chip = /Chip: (.+)/.exec(setup.hardware)?.[1] ?? 'unknown chip';
  const memory = /Memory: (.+)/.exec(setup.hardware)?.[1] ?? '';
  // The last "Total Number of Cores" is the GPU's, under Graphics/Displays.
  const gpu = [...setup.hardware.matchAll(/Total Number of Cores: (\d+)$/gm)].at(-1)?.[1];
  const os = /ProductVersion:\s*(\S+)/.exec(setup.os)?.[1] ?? '';
  return `${chip}, ${gpu ?? '?'} GPU cores, ${memory}, macOS ${os}`;
}

function chart(setup: Setup, summary: Cell[]): string {
  const cell = (item: Case, runtime: Runtime, speculation: Speculation) =>
    summary.find((row) => row.case === item.name && row.runtime === runtime && row.speculation === speculation)!;
  const llama = /commit ([0-9a-f]+)/.exec(setup.llamaVersion)?.[1] ?? '?';
  return renderChart({
    title: 'Muse-Glimmer 30B Dynamic Q4_K_XL · mlx-node vs llama.cpp',
    subtitle: `mlx-node ${setup.mlxCommit.slice(0, 12)} · llama.cpp ${llama} · ${host(setup)}`,
    groups: setup.cases.map((item) => tokens(item.promptTokens)),
    panels: (
      [
        ['prefillTps', 'Prefill'],
        ['decodeTps', 'Decode'],
      ] as const
    ).map(([metric, title]) => ({
      title,
      unit: 'tok/s',
      series: VARIANTS.map((variant, index) => ({
        label: CHART_LABELS[index],
        values: setup.cases.map((item) => {
          const { median: value, min, max } = cell(item, variant.runtime, variant.speculation)[metric];
          return { median: value, min, max };
        }),
      })),
    })),
    note: `Median of ${setup.repetitions} fresh processes, min–max whiskers · ${setup.outputTokens} output tokens · DFlash up to ${setup.draftTokens} drafts · BF16 KV · no prompt-cache hits · load and warmup excluded`,
  });
}

async function compare(setup: Setup, setupSha256: string, summary: Cell[], rows: Sample[], dirty: boolean) {
  const baseline = parseCsv(await readFile(BASELINE, 'utf8'));
  assert.equal(baseline.length, 36, 'baseline.csv must hold the 36 September samples');
  const september = (item: Case, runtime: Runtime, speculation: Speculation, column: string) => {
    const samples = baseline.filter(
      (row) => row.case === item.name && row.runtime === runtime && row.speculation === speculation,
    );
    assert.equal(samples.length, PROTOCOL.repetitions, `baseline.csv: ${item.name} ${runtime} ${speculation}`);
    assert(
      samples.every((row) => Number(row.promptTokens) === item.promptTokens),
      `baseline.csv: ${item.name} tokens`,
    );
    return median(samples.map((row) => Number(row[column])));
  };
  const current = (item: Case, runtime: Runtime, speculation: Speculation) =>
    summary.find((row) => row.case === item.name && row.runtime === runtime && row.speculation === speculation)!;
  const table = (title: string, metric: Metric, scale = 1) => {
    const cells: CompareRow[] = [];
    for (const item of setup.cases) {
      for (const variant of VARIANTS) {
        cells.push({
          label: `${tokens(item.promptTokens)} · ${variant.label}`,
          before: september(item, variant.runtime, variant.speculation, metric) / scale,
          after: current(item, variant.runtime, variant.speculation)[metric].median / scale,
        });
      }
    }
    return compareTable(title, cells);
  };
  const effect = [
    '### DFlash effect (DFlash median / off median − 1)',
    '',
    '| Input tokens | Engine | Decode, Sep | Decode, now | Request time, Sep | Request time, now | Acceptance, Sep | Acceptance, now |',
    '|---:|---|---:|---:|---:|---:|---:|---:|',
  ];
  for (const item of setup.cases) {
    for (const [runtime, label] of [
      ['mlx', 'MLX'],
      ['llama', 'llama.cpp'],
    ] as const) {
      const change = (metric: 'decodeTps' | 'wallMs') => [
        september(item, runtime, 'dflash', metric) / september(item, runtime, 'off', metric) - 1,
        current(item, runtime, 'dflash')[metric].median / current(item, runtime, 'off')[metric].median - 1,
      ];
      const accepted = [
        september(item, runtime, 'dflash', 'draftAcceptance'),
        current(item, runtime, 'dflash').acceptance!.median,
      ];
      effect.push(
        `| ${tokens(item.promptTokens)} | ${label} | ${change('decodeTps').map(percent).join(' | ')} | ${change('wallMs').map(percent).join(' | ')} | ${accepted.map((value) => `${(value * 100).toFixed(1)}%`).join(' | ')} |`,
      );
    }
  }
  const loads = rows.filter((row) => row.runtime === 'mlx');
  const slowest = loads.reduce((a, b) => (a.loadMs >= b.loadMs ? a : b));
  return [
    '## Muse-Glimmer 30B Dynamic Q4_K_XL: this run vs September 2026',
    '',
    ...(dirty ? ['**Development run: the checkout was dirty, so this measures no commit.**', ''] : []),
    `This run: mlx-node \`${setup.mlxCommit.slice(0, 12)}\`, setup \`${setupSha256.slice(0, 12)}\`. September: \`${SEPTEMBER.commit.slice(0, 12)}\`, setup \`${SEPTEMBER.setupSha256.slice(0, 12)}\`, from \`scripts/perf-tracker/muse/baseline.csv\` (issue #142). Before = September median, After = this run's median, of ${setup.repetitions} fresh processes each.`,
    '',
    table('Decode, tokens/second', 'decodeTps'),
    table('Prefill, tokens/second', 'prefillTps'),
    table('Request time, seconds', 'wallMs', 1000),
    effect.join('\n'),
    '',
    '### Model load',
    '',
    `Max MLX \`loadMs\`: **${Math.round(slowest.loadMs)} ms** (\`${slowest.id}\`); September max ${SEPTEMBER.maxMlxLoadMs} ms. Seconds or more mean a worker rebuilt the native GGUF cache (outside the timed region).`,
    '',
  ].join('\n');
}

export async function report(out: string, options: { archived: boolean }): Promise<void> {
  if (!existsSync(join(out, 'environment.json'))) throw new Error(`${out} holds no prepared run`);
  const setup = await readJson<Setup>(join(out, 'environment.json'));
  const setupSha256 = sha256(JSON.stringify(setup));
  const provenance = existsSync(join(out, 'provenance.json'))
    ? await readJson<Provenance>(join(out, 'provenance.json'))
    : undefined;
  const dirty = provenance?.source.dirty ?? false;
  const rows = await loadSamples(out, setup);
  const verified = await verify(setup, setupSha256, rows, options.archived);
  const { summary, parity, thermalClear, csv } = summarize(setup, rows);
  const audited = await audit(out, setup, rows);
  await writeFile(join(out, 'samples.csv'), csv);
  await writeJson(join(out, 'summary.json'), {
    samples: rows.length,
    setupSha256,
    dirty,
    summary,
    parity,
    allCachedTokensZero: true,
    allOutputs96Tokens: true,
    allThermalChecksClear: thermalClear,
    retriedAttempts: audited.retriedAttempts,
  });
  await writeJson(join(out, 'audit.json'), {
    verifiedAt: new Date().toISOString(),
    samples: rows.length,
    setupSha256,
    identitiesVerified: !options.archived,
    sourceUnchanged: verified.sourceUnchanged,
    rehashCount: verified.identities.length,
    identities: verified.identities,
    ...audited,
  });
  await writeFile(join(out, 'chart.svg'), chart(setup, summary));
  const comparison = await compare(setup, setupSha256, summary, rows, dirty);
  await writeFile(join(out, 'compare.md'), comparison);
  console.log(comparison);
  console.log(
    ['samples.csv', 'summary.json', 'audit.json', 'chart.svg', 'compare.md'].map((name) => join(out, name)).join('\n'),
  );
}
