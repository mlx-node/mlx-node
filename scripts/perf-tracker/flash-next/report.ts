/// <reference types="node" />

import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { readJson, renderChart, writeJson, type ChartSeries } from '../common.js';
import { baselineHistory, samplesCsv, speculationTable, summarize, type SummaryRow } from '../qwen38/report.js';
import { RUNTIMES, paths } from '../qwen38/suite.js';
import { HERE, PROTOCOL, SPECULATION, type Loaded, type Provenance, type Residency } from './suite.js';

const BASELINE = join(HERE, 'baseline.csv');
const GIB = 2 ** 30;
const tokens = (count: number) => count.toLocaleString('en-US');
const gib = (bytes: number | undefined) => (bytes === undefined ? 'n/a' : (bytes / GIB).toFixed(1));

interface Agreement {
  commonPrefixChars: number[];
  mlxChars: number[];
  llamaChars: number[];
  mlxStart: string;
  llamaStart: string;
}

export interface FlashSummaryRow extends SummaryRow {
  mlxResidency: (Residency & { pid: number })[];
  agreement: Agreement;
}

function commonPrefix(a: string, b: string): number {
  const end = Math.min(a.length, b.length);
  let length = 0;
  while (length < end && a[length] === b[length]) length++;
  return length;
}

const start = (text: string) =>
  text
    .slice(0, 100)
    .replace(/\s+/g, ' ')
    .replace(/[|`]/g, (char) => `\\${char}`)
    .replace(/</g, '&lt;');

function range(values: readonly number[], digits: number): string {
  const [low, high] = [Math.min(...values), Math.max(...values)].map((value) => value.toFixed(digits));
  return low === high ? low : `${low}–${high}`;
}

// Refuses what summarize() refuses or a run without the native prompt-ID proof, then
// writes samples.csv, summary.json, chart.svg and compare.md.
export async function report(out: string, allowDirty: boolean): Promise<FlashSummaryRow[]> {
  const { manifest, samples, summary } = await summarize(out, allowDirty, {
    outputTokens: PROTOCOL.outputTokens,
    speculation: SPECULATION,
  });
  const { repetitions } = manifest.protocol;
  const files = paths(out);
  const provenance = await readJson<Provenance>(files.provenance);
  if (provenance.prewarm?.checkIds.every((check) => check.identical) !== true) {
    throw new Error(`${out}: the native prompt-ID proof is missing from provenance`);
  }

  // The automatic budget is chosen per process; samples name their process.
  const loads = new Map<number, Residency>();
  for (const line of (await readFile(files.progress, 'utf8')).split('\n').filter(Boolean)) {
    const event = JSON.parse(line) as { event?: string; runtime?: string; metadata?: Loaded };
    if (event.event === 'loaded' && event.runtime === 'mlx-node' && event.metadata) {
      loads.set(event.metadata.pid, event.metadata.residency);
    }
  }
  const rows: FlashSummaryRow[] = [];
  for (const row of summary) {
    const pids = [
      ...new Set(
        samples.filter((sample) => sample.runtime === 'mlx-node' && sample.name === row.name).map(({ pid }) => pid),
      ),
    ];
    const mlxResidency = pids.map((pid) => {
      const residency = loads.get(pid);
      if (!residency) throw new Error(`${row.name}: no recorded mlx-node load for process ${pid}`);
      return { pid, ...residency };
    });
    const texts = await Promise.all(
      Array.from({ length: repetitions }, async (_, index) => {
        const raw = (runtime: string) => join(files.raw, `${runtime}-${index + 1}-${row.name}.json`);
        const mlx = (await readJson<{ rawText?: string }>(raw('mlx'))).rawText ?? '';
        const llama = (await readJson<{ text?: string }>(raw('llama'))).text ?? '';
        return { mlx, llama };
      }),
    );
    rows.push({
      ...row,
      mlxResidency,
      agreement: {
        commonPrefixChars: texts.map(({ mlx, llama }) => commonPrefix(mlx, llama)),
        mlxChars: texts.map(({ mlx }) => mlx.length),
        llamaChars: texts.map(({ llama }) => llama.length),
        mlxStart: texts[0].mlx.slice(0, 200),
        llamaStart: texts[0].llama.slice(0, 200),
      },
    });
  }

  await writeFile(join(out, 'samples.csv'), samplesCsv(samples, SPECULATION));
  await writeJson(join(out, 'summary.json'), rows);

  const measuredAt = provenance.matrix?.startedAt.slice(0, 10) ?? 'unknown date';
  const commit = `${provenance.tree.commit.slice(0, 12)}${provenance.tree.dirty ? ' (dirty tree)' : ''}`;
  const host = `${provenance.host.chip} · ${provenance.host.gpuCores} GPU cores · ${Math.round(provenance.host.memoryBytes / GIB)} GiB`;
  const budgets = rows.flatMap((row) => row.mlxResidency.map(({ weightBudgetBytes }) => weightBudgetBytes / GIB));
  const budget = `${range(budgets, 0)} GiB`;
  const series = (metric: 'prefillTokensPerSecond' | 'decodeTokensPerSecond'): ChartSeries[] =>
    RUNTIMES.map((runtime) => ({ label: runtime, values: rows.map((row) => row[runtime][metric]) }));
  await writeFile(
    join(out, 'chart.svg'),
    renderChart({
      title: 'Qwen3.8 Flash Next UD-Q4_K_XL · mlx-node vs llama.cpp',
      subtitle: `${measuredAt} · ${host} · mlx-node ${commit} · llama.cpp ${provenance.llama.commit.slice(0, 12)}`,
      groups: rows.map(
        (row) =>
          `${tokens(row.promptTokens)}${provenance.protocol.admission[row.promptTokens] === 'gpu-idle' ? '' : '*'}`,
      ),
      panels: [
        { title: 'Prefill', unit: 'tok/s', series: series('prefillTokensPerSecond') },
        { title: 'Decode', unit: 'tok/s', series: series('decodeTokensPerSecond') },
      ],
      note: `Median of ${repetitions} runs, ${PROTOCOL.outputTokens} output tokens; whiskers min–max; MTP on, depth 2; mlx-node SSD streaming (auto budget ${budget}). *No other local LLM${provenance.protocol.desktopLoad ? '' : '; other rows start at GPU idle'}.`,
    }),
  );

  const ratios = [
    '| Prompt tokens | mlx-node prefill | llama.cpp prefill | Ratio | mlx-node decode | llama.cpp decode | Ratio |',
    '|---:|---:|---:|---:|---:|---:|---:|',
    ...rows.map((row) => {
      const [mlx, llama] = [row['mlx-node'], row['llama.cpp']];
      return `| ${tokens(row.promptTokens)} | ${mlx.prefillTokensPerSecond.median.toFixed(1)} | ${llama.prefillTokensPerSecond.median.toFixed(1)} | ${row.prefillRatio.toFixed(3)}× | ${mlx.decodeTokensPerSecond.median.toFixed(2)} | ${llama.decodeTokensPerSecond.median.toFixed(2)} | ${row.decodeRatio.toFixed(3)}× |`;
    }),
  ];
  const history = await baselineHistory(
    BASELINE,
    rows,
    'First entry for this model, so there is no before value. MTP is on in both runtimes and changes prefill as well as decode: both runtimes run the MTP head over the prompt inside prefill.',
  );
  await writeFile(
    join(out, 'compare.md'),
    [
      `## Qwen3.8 Flash Next UD-Q4_K_XL: ${measuredAt}`,
      '',
      `This run: mlx-node ${commit}, llama.cpp ${provenance.llama.commit.slice(0, 12)}; ${host}. Medians of ${repetitions} runs, ${PROTOCOL.outputTokens} output tokens, tokens/second.`,
      ...(provenance.protocol.desktopLoad
        ? [
            '',
            'Admission: --desktop-load. Every row ran with desktop graphics and no other local LLM; the 6,219 and 18,754 rows did not wait for GPU idle.',
          ]
        : []),
      '',
      ...ratios,
      '',
      ...history,
      '### Speculation (native MTP, depth 2)',
      '',
      "Medians per cell. mlx-node runs the BF16 `mtp.*` block from the original checkpoint with the target GGUF's embedding and output; llama.cpp runs unsloth's self-contained Q8_0 MTP head. mlx-node reports per-step means, so its counts are mean × steps; llama.cpp reports no step count.",
      '',
      speculationTable(rows),
      '### mlx-node weight budget per load (information)',
      '',
      "mlx-node's default policy: an automatic budget sized from free memory at each load, with routed experts streamed from the internal SSD.",
      '',
      '| Prompt tokens | Process | Budget GiB | Hot weights GiB | Full hot residency | Prefill window | Free at load GiB |',
      '|---:|---:|---:|---:|:---:|---:|---:|',
      ...rows.flatMap((row) =>
        row.mlxResidency.map(
          (load) =>
            `| ${tokens(row.promptTokens)} | ${load.pid} | ${gib(load.weightBudgetBytes)} | ${gib(load.hotWeightBytes)} | ${load.fullHotResidency ? 'yes' : 'no'} | ${load.prefillChunkTokens} | ${gib(load.availableMemoryBytes)} |`,
        ),
      ),
      '',
      '### Output agreement (informational, not a gate)',
      '',
      'Characters both runtimes generated identically from the start. Correct greedy runs can part early; an unreadable start is the warning sign.',
      '',
      `| Prompt tokens | Common prefix, runs 1–${repetitions} | mlx-node chars | llama.cpp chars |`,
      '|---:|---:|---:|---:|',
      ...rows.map(
        ({ promptTokens, agreement }) =>
          `| ${tokens(promptTokens)} | ${agreement.commonPrefixChars.join(', ')} | ${agreement.mlxChars.join(', ')} | ${agreement.llamaChars.join(', ')} |`,
      ),
      '',
      '| Prompt tokens | Runtime | Start of run 1 |',
      '|---:|---|---|',
      ...rows.flatMap(({ promptTokens, agreement }) => [
        `| ${tokens(promptTokens)} | mlx-node | ${start(agreement.mlxStart)} |`,
        `| ${tokens(promptTokens)} | llama.cpp | ${start(agreement.llamaStart)} |`,
      ]),
      '',
    ].join('\n'),
  );
  return rows;
}
