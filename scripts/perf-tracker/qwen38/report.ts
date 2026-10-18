/// <reference types="node" />

import { existsSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { join, relative } from 'node:path';

import {
  ACCEPTANCE_CSV_COLUMNS,
  ROOT,
  acceptanceCsv,
  acceptanceStats,
  acceptanceTable,
  compareTable,
  median,
  parseCsv,
  range,
  readJson,
  renderChart,
  toCsv,
  writeJson,
  type AcceptanceStats,
  type ChartSeries,
  type CompareRow,
  type SpeculationRule,
} from '../common.js';
import {
  FIXTURE,
  HERE,
  KIND,
  RUNTIMES,
  SPECULATION,
  paths,
  readManifest,
  sampleAcceptance,
  type Manifest,
  type Provenance,
  type Runtime,
  type Sample,
} from './suite.js';

const BASELINE = join(HERE, 'baseline.csv');

export const CSV_COLUMNS = [
  'runtime',
  'prompt_tokens',
  'repetition',
  'output_tokens',
  'cached_tokens',
  'prefill_tokens_per_second',
  'decode_tokens_per_second',
  'ttft_ms',
  'wall_ms',
  ...ACCEPTANCE_CSV_COLUMNS,
] as const;

const METRICS = ['prefillTokensPerSecond', 'decodeTokensPerSecond', 'ttftMs'] as const;
type Metric = (typeof METRICS)[number];

interface Cell {
  runs: number;
  prefillTokensPerSecond: Stat;
  decodeTokensPerSecond: Stat;
  ttftMs: Stat;
  acceptance: AcceptanceStats;
}

interface Stat {
  median: number;
  min: number;
  max: number;
  samples: number[];
}

export interface SummaryRow {
  name: string;
  promptTokens: number;
  sourceLine: number;
  'mlx-node': Cell;
  'llama.cpp': Cell;
  prefillRatio: number;
  decodeRatio: number;
}

const tokens = (count: number) => count.toLocaleString('en-US');

export interface Summary {
  manifest: Manifest;
  provenance: Provenance;
  samples: Sample[];
  summary: SummaryRow[];
}

export interface SummarizeOptions {
  outputTokens?: number;
  speculation: SpeculationRule;
}

// Refuses an incomplete or off-protocol run, or a sample that ran without speculation.
export async function summarize(
  out: string,
  allowDirty: boolean,
  { outputTokens: outputLength, speculation }: SummarizeOptions,
): Promise<Summary> {
  const manifest = await readManifest();
  const { repetitions } = manifest.protocol;
  const outputTokens = outputLength ?? manifest.protocol.outputTokens;
  const files = paths(out);
  const provenance = await readJson<Provenance>(files.provenance);
  if (provenance.matrix?.postRunVerified !== true) {
    throw new Error(`${out}: the matrix has not passed its post-run identity check`);
  }
  if (provenance.tree.dirty && !allowDirty) {
    throw new Error(`${out} measured an uncommitted checkout; pass --allow-dirty to report it`);
  }
  const events = (await readFile(files.progress, 'utf8'))
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line) as { event?: string; label?: string });
  const order = (sample: Sample) => {
    const index = manifest.cases.findIndex((item) => item.name === sample.name);
    const first = RUNTIMES[index % 2];
    return index * 2 * repetitions + (sample.runtime === first ? 0 : repetitions) + sample.run;
  };
  const samples = (await readJson<Sample[]>(files.samples)).sort((a, b) => order(a) - order(b));

  const key = (runtime: string, name: string, run: number) => `${runtime}/${name}/${run}`;
  const expected = new Set(
    RUNTIMES.flatMap((runtime) =>
      manifest.cases.flatMap((item) => Array.from({ length: repetitions }, (_, i) => key(runtime, item.name, i + 1))),
    ),
  );
  const seen = new Set(samples.map((sample) => key(sample.runtime, sample.name, sample.run)));
  if (seen.size !== samples.length || seen.size !== expected.size || ![...seen].every((id) => expected.has(id))) {
    throw new Error('Incomplete measurement matrix');
  }
  for (const sample of samples) {
    const item = manifest.cases.find((candidate) => candidate.name === sample.name)!;
    const label = `${KIND[sample.runtime]}-${sample.run}-${sample.name}`;
    if (sample.numTokens !== outputTokens || sample.cachedTokens !== 0 || sample.promptTokens !== item.promptTokens) {
      throw new Error(`${label}: token accounting differs from the protocol`);
    }
    sampleAcceptance(sample, speculation);
    if (provenance.protocol.admission[item.promptTokens] === 'no-other-local-llm') {
      if (!events.some((event) => event.event === 'llm_isolation_confirmed' && event.label === label)) {
        throw new Error(`${label}: no recorded no-other-local-LLM admission`);
      }
    } else {
      const system = await readJson<{ before: { gpuUtilization: number } }>(join(files.system, `${label}.json`));
      if (!(system.before.gpuUtilization <= 5)) throw new Error(`${label}: admitted above 5% GPU utilization`);
    }
  }
  if (provenance.tokenizer?.verified !== true) throw new Error('Tokenizer verification is missing from provenance');

  const summary = manifest.cases.map((item) => {
    const cell = (runtime: Runtime): Cell => {
      const runs = samples.filter((sample) => sample.runtime === runtime && sample.name === item.name);
      if (new Set(runs.map(({ pid }) => pid)).size !== 1 || new Set(runs.map(({ block }) => block)).size !== 1) {
        throw new Error(`${runtime} ${item.name}: the samples come from more than one process`);
      }
      const stat = (metric: Metric): Stat => {
        const values = runs.map((sample) => sample.performance[metric]);
        return { ...range(values), samples: values };
      };
      return {
        runs: runs.length,
        prefillTokensPerSecond: stat('prefillTokensPerSecond'),
        decodeTokensPerSecond: stat('decodeTokensPerSecond'),
        ttftMs: stat('ttftMs'),
        acceptance: acceptanceStats(runs.map((sample) => sampleAcceptance(sample, speculation))),
      };
    };
    const mlx = cell('mlx-node');
    const llama = cell('llama.cpp');
    return {
      name: item.name,
      promptTokens: item.promptTokens,
      sourceLine: item.sourceLine,
      'mlx-node': mlx,
      'llama.cpp': llama,
      prefillRatio: mlx.prefillTokensPerSecond.median / llama.prefillTokensPerSecond.median,
      decodeRatio: mlx.decodeTokensPerSecond.median / llama.decodeTokensPerSecond.median,
    };
  });
  return { manifest, provenance, samples, summary };
}

export function samplesCsv(samples: readonly Sample[], speculation: SpeculationRule): string {
  return toCsv(
    CSV_COLUMNS,
    samples.map((sample) => ({
      ...acceptanceCsv(sampleAcceptance(sample, speculation)),
      runtime: sample.runtime,
      prompt_tokens: sample.promptTokens,
      repetition: sample.run,
      output_tokens: sample.numTokens,
      cached_tokens: sample.cachedTokens,
      prefill_tokens_per_second: sample.performance.prefillTokensPerSecond,
      decode_tokens_per_second: sample.performance.decodeTokensPerSecond,
      ttft_ms: sample.performance.ttftMs,
      wall_ms: sample.wallMs,
    })),
  );
}

export function printSummary(rows: readonly SummaryRow[]): void {
  console.log(
    '| Prompt tokens | mlx-node prefill | llama.cpp prefill | Ratio | mlx-node decode | llama.cpp decode | Ratio |',
  );
  console.log('|---:|---:|---:|---:|---:|---:|---:|');
  for (const row of rows) {
    const [mlx, llama] = [row['mlx-node'], row['llama.cpp']];
    console.log(
      `| ${row.promptTokens.toLocaleString('en-US')} | ${mlx.prefillTokensPerSecond.median.toFixed(1)} | ${llama.prefillTokensPerSecond.median.toFixed(1)} | ${row.prefillRatio.toFixed(3)}× | ${mlx.decodeTokensPerSecond.median.toFixed(2)} | ${llama.decodeTokensPerSecond.median.toFixed(2)} | ${row.decodeRatio.toFixed(3)}× |`,
    );
  }
}

export const speculationTable = (rows: readonly SummaryRow[]): string =>
  acceptanceTable(
    rows.flatMap((row) =>
      RUNTIMES.map((runtime) => ({
        label: `${runtime} · ${tokens(row.promptTokens)}`,
        stats: row[runtime].acceptance,
      })),
    ),
  );

// The baseline is the previous entry of the same fixture version.
export async function baselineHistory(
  baseline: string,
  rows: readonly SummaryRow[],
  firstEntry: string,
): Promise<string[]> {
  if (!existsSync(baseline)) return [firstEntry, ''];
  const lines = parseCsv(await readFile(baseline, 'utf8'));
  const compare = (metric: 'prefill' | 'decode'): CompareRow[] =>
    rows.flatMap((row) =>
      RUNTIMES.map((runtime) => ({
        label: `${runtime} · ${tokens(row.promptTokens)}`,
        before: median(
          lines
            .filter((line) => line.runtime === runtime && Number(line.prompt_tokens) === row.promptTokens)
            .map((line) => Number(line[`${metric}_tokens_per_second`])),
        ),
        after: row[runtime][`${metric}TokensPerSecond`].median,
      })),
    );
  return [
    `Before = medians of the previous entry in \`${relative(ROOT, baseline)}\`; after = this run.`,
    '',
    compareTable('Prefill', compare('prefill')),
    compareTable('Decode', compare('decode')),
  ];
}

// Refuses an incomplete or off-protocol run, then writes samples.csv,
// summary.json, chart.svg and compare.md.
export async function report(out: string, allowDirty: boolean): Promise<SummaryRow[]> {
  const { manifest, provenance, samples, summary } = await summarize(out, allowDirty, { speculation: SPECULATION });
  const { outputTokens, repetitions } = manifest.protocol;
  await writeFile(join(out, 'samples.csv'), samplesCsv(samples, SPECULATION));
  await writeJson(join(out, 'summary.json'), summary);

  const measuredAt = provenance.matrix?.startedAt.slice(0, 10) ?? 'unknown date';
  const commit = `${provenance.tree.commit.slice(0, 12)}${provenance.tree.dirty ? ' (dirty tree)' : ''}`;
  const host = `${provenance.host.chip} · ${provenance.host.gpuCores} GPU cores · ${Math.round(provenance.host.memoryBytes / 2 ** 30)} GiB`;
  const series = (metric: Metric): ChartSeries[] =>
    RUNTIMES.map((runtime) => ({ label: runtime, values: summary.map((row) => row[runtime][metric]) }));
  await writeFile(
    join(out, 'chart.svg'),
    renderChart({
      title: 'Qwen3.8 27B UD-Q4_K_XL · mlx-node vs llama.cpp',
      subtitle: `${measuredAt} · ${host} · mlx-node ${commit} · llama.cpp ${provenance.llama.commit.slice(0, 12)}`,
      groups: summary.map(
        (row) =>
          `${tokens(row.promptTokens)}${provenance.protocol.admission[row.promptTokens] === 'gpu-idle' ? '' : '*'}`,
      ),
      panels: [
        { title: 'Prefill', unit: 'tok/s', series: series('prefillTokensPerSecond') },
        { title: 'Decode', unit: 'tok/s', series: series('decodeTokensPerSecond') },
      ],
      note: `Median of ${repetitions} runs, ${outputTokens} output tokens; whiskers are min–max; DFlash2 on, 7 drafts per step. *Background graphics allowed, no other local LLM${provenance.protocol.desktopLoad ? '' : '; other rows start at GPU idle'}.`,
    }),
  );

  const history = await baselineHistory(
    BASELINE,
    summary,
    `First entry of the \`${FIXTURE}\` protocol: speculative decoding is on in both runtimes, and it changes prefill as well as decode (mlx-node's DFlash2 turn runs on its flat cache, and both runtimes build the draft context inside prefill). Entries before 2026-10-18 ran without speculation, so there is no before value.`,
  );
  await writeFile(
    join(out, 'compare.md'),
    [
      `## Qwen3.8 27B UD-Q4_K_XL: ${measuredAt}`,
      '',
      `This run: mlx-node ${commit}, llama.cpp ${provenance.llama.commit.slice(0, 12)}; ${host}. Medians of ${repetitions} runs, tokens/second.`,
      ...(provenance.protocol.desktopLoad
        ? [
            '',
            'Admission: --desktop-load. Every row ran with desktop graphics and no other local LLM; the 6,219 and 18,754 rows did not wait for GPU idle.',
          ]
        : []),
      '',
      ...history,
      '### Speculation (DFlash2, 7 drafts per step)',
      '',
      'Medians per cell. mlx-node reports per-step means, so its counts are mean × steps; llama.cpp reports no step count.',
      '',
      speculationTable(summary),
    ].join('\n'),
  );
  return summary;
}
