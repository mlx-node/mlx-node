/// <reference types="node" />

import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import {
  compareTable,
  median,
  parseCsv,
  range,
  readJson,
  renderChart,
  toCsv,
  writeJson,
  type ChartSeries,
  type CompareRow,
} from '../common.js';
import {
  HERE,
  KIND,
  POLICY,
  RUNTIMES,
  paths,
  readManifest,
  type Provenance,
  type Runtime,
  type Sample,
} from './suite.js';

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
] as const;

const METRICS = ['prefillTokensPerSecond', 'decodeTokensPerSecond', 'ttftMs'] as const;
type Metric = (typeof METRICS)[number];

interface Cell {
  runs: number;
  prefillTokensPerSecond: Stat;
  decodeTokensPerSecond: Stat;
  ttftMs: Stat;
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

// Refuses an incomplete or off-protocol run, then writes samples.csv,
// summary.json, chart.svg and compare.md.
export async function report(out: string): Promise<SummaryRow[]> {
  const manifest = await readManifest();
  const { outputTokens, repetitions } = manifest.protocol;
  const files = paths(out);
  const provenance = await readJson<Provenance>(files.provenance);
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
    if (
      sample.numTokens !== outputTokens ||
      sample.cachedTokens !== 0 ||
      sample.promptTokens !== item.promptTokens ||
      sample.performance.mtpCycles
    ) {
      throw new Error(`${label}: token accounting or speculation differs from the protocol`);
    }
    if (POLICY[item.promptTokens] === 'no-other-local-llm') {
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
      const stat = (metric: Metric): Stat => {
        const values = runs.map((sample) => sample.performance[metric]);
        return { ...range(values), samples: values };
      };
      return {
        runs: runs.length,
        prefillTokensPerSecond: stat('prefillTokensPerSecond'),
        decodeTokensPerSecond: stat('decodeTokensPerSecond'),
        ttftMs: stat('ttftMs'),
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

  await writeFile(
    join(out, 'samples.csv'),
    toCsv(
      CSV_COLUMNS,
      samples.map((sample) => ({
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
    ),
  );
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
      groups: summary.map((row) => `${tokens(row.promptTokens)}${POLICY[row.promptTokens] === 'gpu-idle' ? '' : '*'}`),
      panels: [
        { title: 'Prefill', unit: 'tok/s', series: series('prefillTokensPerSecond') },
        { title: 'Decode', unit: 'tok/s', series: series('decodeTokensPerSecond') },
      ],
      note: `Median of ${repetitions} runs, ${outputTokens} output tokens; whiskers are min–max. *Background graphics allowed, no other local LLM; other rows start at GPU idle.`,
    }),
  );

  const baseline = parseCsv(await readFile(join(HERE, 'baseline.csv'), 'utf8'));
  const compare = (metric: 'prefill' | 'decode'): CompareRow[] =>
    summary.flatMap((row) =>
      RUNTIMES.map((runtime) => ({
        label: `${runtime} · ${tokens(row.promptTokens)}`,
        before: median(
          baseline
            .filter((line) => line.runtime === runtime && Number(line.prompt_tokens) === row.promptTokens)
            .map((line) => Number(line[`${metric}_tokens_per_second`])),
        ),
        after: row[runtime][`${metric}TokensPerSecond`].median,
      })),
    );
  await writeFile(
    join(out, 'compare.md'),
    [
      `## Qwen3.8 27B UD-Q4_K_XL: ${measuredAt} vs 2026-09-09`,
      '',
      `Before: issue #142 initial benchmark (mlx-node ${manifest.measuredRevisions['mlx-node'].slice(0, 12)}, llama.cpp ${manifest.measuredRevisions['llama.cpp'].slice(0, 12)}; Apple M5 Max, 40 GPU cores, 128 GiB).`,
      `After: mlx-node ${commit}, llama.cpp ${provenance.llama.commit.slice(0, 12)}; ${host}. Medians of ${repetitions} runs, tokens/second.`,
      '',
      compareTable('Prefill', compare('prefill')),
      compareTable('Decode', compare('decode')),
    ].join('\n'),
  );
  return summary;
}
