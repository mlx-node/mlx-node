/// <reference types="node" />

import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { compareTable, parseCsv, range, readJson, renderChart, toCsv, writeJson, type CompareRow } from '../common.js';
import { LABEL, REPETITIONS, RUNTIMES, readManifest, sampleId, samplePath, type Runtime } from './suite.js';

const BASELINE = join(dirname(fileURLToPath(import.meta.url)), 'baseline.csv');
const BASELINE_DATE = '2026-09-10';
const METRICS = ['prefillMs', 'prefillTps', 'decodeTps', 'wallMs', 'generatedTokens'] as const;
const CSV_COLUMNS = [
  'runtime',
  'prompt_tokens',
  'repetition',
  'output_tokens',
  'cached_tokens',
  'prefill_tokens_per_second',
  'decode_tokens_per_second',
  'ttft_ms',
  'wall_ms',
];

type Metric = (typeof METRICS)[number];
type Stats = ReturnType<typeof range>;

interface Sample extends Record<Metric, number> {
  runtime: string;
  name: string;
  run: string;
  accountedMs: number;
  promptTokens: number;
  cachedTokens: number;
  inputSha256: string;
  textSha256: string;
  addonSha256?: string;
  args?: unknown;
  result?: { finishReason?: string; stop_type?: string };
  finishReason?: string;
  timingOverheadMs?: number;
}

type RuntimeRow = Record<Metric, Stats> & { distinctOutputHashes: number; finishReasons: string[] };

interface Row extends Record<Runtime, RuntimeRow> {
  name: string;
  promptTokens: number;
  llamaOverMlxPrefill: number;
  llamaOverMlxDecode: number;
  mlxOverLlamaLatency: number;
  matchingOutputPairs: number;
}

interface Provenance {
  createdAt: string;
  completedAt?: string;
  host: { os: string; chip: string; gpuCores: number; memoryBytes: number };
  addon: Record<string, string>;
}

export async function report(out: string): Promise<void> {
  const manifest = await readManifest();
  const { protocol } = manifest;
  const environment = await readJson<Provenance>(join(out, 'provenance.json'));
  if (!environment.completedAt) throw new Error(`${out}: the matrix has not passed its post-run identity check`);
  const samples: Sample[] = [];
  for (const expected of manifest.cases) {
    for (const runtime of RUNTIMES) {
      for (const run of REPETITIONS) {
        const path = samplePath(out, sampleId(runtime, expected.name, run));
        const sample = await readJson<Sample>(path);
        const check = (ok: boolean, what: string) => {
          if (!ok) throw new Error(`${path}: ${what}`);
        };
        check(sample.name === expected.name && sample.run === run, 'case or repetition mismatch');
        check(sample.cachedTokens === 0, 'cached tokens');
        check(sample.promptTokens === expected.promptTokens, 'prompt tokens');
        check(sample.inputSha256 === expected.tokenIdsSha256, 'input token IDs');
        check(sample.generatedTokens === protocol.outputTokens, 'generated tokens');
        if (runtime === 'mlx') {
          check(sample.addonSha256 === environment.addon['mlx-core.darwin-arm64.node'], 'native addon changed');
        }
        check(Math.abs(sample.wallMs - sample.accountedMs) < 250, 'wall time does not match native timings');
        const { result, ...kept } = sample;
        delete kept.args;
        samples.push({
          ...kept,
          runtime,
          finishReason: result?.finishReason ?? result?.stop_type,
          timingOverheadMs: sample.wallMs - sample.accountedMs,
        });
      }
    }
  }

  const pick = (name: string, runtime: Runtime, run?: string) =>
    samples.filter((s) => s.name === name && s.runtime === runtime && (run === undefined || s.run === run));
  const rows: Row[] = manifest.cases.map(({ name, promptTokens }) => {
    const side = (runtime: Runtime): RuntimeRow => {
      const group = pick(name, runtime);
      const stats = Object.fromEntries(METRICS.map((metric) => [metric, range(group.map((s) => s[metric]))]));
      return {
        ...(stats as Record<Metric, Stats>),
        distinctOutputHashes: new Set(group.map((s) => s.textSha256)).size,
        finishReasons: [...new Set(group.map((s) => String(s.finishReason)))].sort(),
      };
    };
    const mlx = side('mlx');
    const llama = side('llama');
    return {
      name,
      promptTokens,
      mlx,
      llama,
      llamaOverMlxPrefill: llama.prefillTps.median / mlx.prefillTps.median,
      llamaOverMlxDecode: llama.decodeTps.median / mlx.decodeTps.median,
      mlxOverLlamaLatency: mlx.wallMs.median / llama.wallMs.median,
      matchingOutputPairs: REPETITIONS.filter(
        (run) => pick(name, 'mlx', run)[0].textSha256 === pick(name, 'llama', run)[0].textSha256,
      ).length,
    };
  });

  // Provenance and measurements only: prompts and generated text stay in raw/.
  await writeJson(join(out, 'results.json'), {
    environment,
    fixture: {
      id: manifest.id,
      payload: manifest.payload,
      source: manifest.source,
      cases: manifest.cases,
    },
    rows,
    samples,
  });

  const csvRows = rows.flatMap((row) =>
    RUNTIMES.flatMap((runtime) =>
      REPETITIONS.map((run) => {
        const s = pick(row.name, runtime, run)[0];
        return {
          runtime: LABEL[runtime],
          prompt_tokens: s.promptTokens,
          repetition: run,
          output_tokens: s.generatedTokens,
          cached_tokens: s.cachedTokens,
          prefill_tokens_per_second: s.prefillTps,
          decode_tokens_per_second: s.decodeTps,
          ttft_ms: s.prefillMs,
          wall_ms: s.wallMs,
        };
      }),
    ),
  );
  await writeFile(join(out, 'samples.csv'), toCsv(CSV_COLUMNS, csvRows));

  await writeJson(
    join(out, 'summary.json'),
    rows.map((row) =>
      Object.fromEntries(
        RUNTIMES.map((runtime) => [
          LABEL[runtime],
          { prefillTokensPerSecond: row[runtime].prefillTps, decodeTokensPerSecond: row[runtime].decodeTps },
        ]),
      ),
    ),
  );

  const { host } = environment;
  const groups = rows.map((row) => row.promptTokens.toLocaleString('en-US'));
  const panel = (title: string, metric: 'prefillTps' | 'decodeTps') => ({
    title,
    unit: 'tokens/s',
    series: RUNTIMES.map((runtime) => ({ label: LABEL[runtime], values: rows.map((row) => row[runtime][metric]) })),
  });
  await writeFile(
    join(out, 'chart.svg'),
    renderChart({
      title: 'Gemma 4 12B IT · QAT Q4_0 GGUF · mlx-node vs llama.cpp',
      subtitle: `${host.chip} · ${host.gpuCores} GPU cores · ${Math.round(host.memoryBytes / 2 ** 30)} GB · ${host.os} · ${environment.createdAt.slice(0, 10)}`,
      groups,
      panels: [panel('Prefill', 'prefillTps'), panel('Decode', 'decodeTps')],
      note: `Bars: median of ${REPETITIONS.length} fresh processes; whiskers: min–max. Same token IDs, ${protocol.outputTokens} output tokens, BF16 KV, 0 cached tokens. Higher is faster.`,
    }),
  );

  const baseline = parseCsv(await readFile(BASELINE, 'utf8'));
  const before = (runtime: Runtime, promptTokens: number, column: string) =>
    range(
      baseline
        .filter((r) => r.runtime === LABEL[runtime] && Number(r.prompt_tokens) === promptTokens)
        .map((r) => Number(r[column])),
    ).median;
  const compare = (column: string, metric: Metric, scale = 1): CompareRow[] =>
    rows.flatMap((row) =>
      RUNTIMES.map((runtime) => ({
        label: `${row.promptTokens.toLocaleString('en-US')} · ${LABEL[runtime]}`,
        before: before(runtime, row.promptTokens, column) / scale,
        after: row[runtime][metric].median / scale,
      })),
    );
  await writeFile(
    join(out, 'compare.md'),
    [
      `## Gemma 4 12B IT QAT Q4_0: this run vs ${BASELINE_DATE}`,
      '',
      `Before = medians of the ${BASELINE_DATE} samples in \`scripts/perf-tracker/gemma4/baseline.csv\`; after = medians of this run. Prompt tokens · runtime.`,
      '',
      compareTable('Prefill, tokens/second', compare('prefill_tokens_per_second', 'prefillTps')),
      compareTable('Decode, tokens/second', compare('decode_tokens_per_second', 'decodeTps')),
      compareTable('Request wall time, seconds', compare('wall_ms', 'wallMs', 1000)),
    ].join('\n'),
  );

  const fmt = (value: number, digits: number) =>
    value.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
  console.log(
    '| Prompt tokens | mlx-node prefill | llama.cpp prefill | Ratio | mlx-node decode | llama.cpp decode | Ratio |',
  );
  console.log('|---:|---:|---:|---:|---:|---:|---:|');
  for (const { promptTokens, mlx, llama } of rows) {
    const [mp, lp, md, ld] = [mlx.prefillTps, llama.prefillTps, mlx.decodeTps, llama.decodeTps].map((s) => s.median);
    console.log(
      `| ${fmt(promptTokens, 0)} | ${fmt(mp, 1)} | ${fmt(lp, 1)} | ${(mp / lp).toFixed(3)}× | ${md.toFixed(2)} | ${ld.toFixed(2)} | ${(md / ld).toFixed(3)}× |`,
    );
  }
  console.log('\n| Prompt tokens | mlx-node seconds | llama.cpp seconds | mlx-node latency change |');
  console.log('|---:|---:|---:|---:|');
  for (const { promptTokens, mlx, llama, mlxOverLlamaLatency } of rows) {
    const change = ((mlxOverLlamaLatency - 1) * 100).toFixed(1);
    console.log(
      `| ${fmt(promptTokens, 0)} | ${(mlx.wallMs.median / 1000).toFixed(2)} | ${(llama.wallMs.median / 1000).toFixed(2)} | ${Number(change) > 0 ? '+' : ''}${change}% |`,
    );
  }
  console.log('\n| Prompt tokens | mlx-node prefill | llama.cpp prefill | mlx-node decode | llama.cpp decode |');
  console.log('|---:|---:|---:|---:|---:|');
  for (const { promptTokens, mlx, llama } of rows) {
    const span = (s: Stats) => `${fmt(s.min, 2)}–${fmt(s.max, 2)}`;
    console.log(
      `| ${fmt(promptTokens, 0)} | ${span(mlx.prefillTps)} | ${span(llama.prefillTps)} | ${span(mlx.decodeTps)} | ${span(llama.decodeTps)} |`,
    );
  }
  console.log(
    `\n${['results.json', 'samples.csv', 'summary.json', 'chart.svg', 'compare.md'].map((f) => join(out, f)).join('\n')}`,
  );
}
