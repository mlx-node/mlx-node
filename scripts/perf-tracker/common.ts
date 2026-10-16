/// <reference types="node" />

import { execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, readdir, readFile, realpath, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
export const LLAMA_COMMIT = 'a14dba686aaafba3a2d6b5eb8820b0df5c5d2d92';
export const MODELS_DIR = resolve(process.env.MODELS_DIR ?? join(ROOT, '.cache/models'));

export type Suite = 'qwen38' | 'gemma4' | 'muse';

export const sleep = (ms: number) => new Promise((done) => setTimeout(done, ms));

export function sh(command: string, args: string[], options: { cwd?: string } = {}): string {
  return execFileSync(command, args, { cwd: options.cwd, encoding: 'utf8', maxBuffer: 64 << 20 }).trim();
}

export function sha256(data: Buffer | string): string {
  return createHash('sha256').update(data).digest('hex');
}

export async function sha256File(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest('hex');
}

export async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(path, 'utf8')) as T;
}

export async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}

export function utcStamp(date = new Date()): string {
  return date
    .toISOString()
    .replace(/[-:]/g, '')
    .replace(/\.\d+Z$/, 'Z');
}

// A new directory per run, so a rerun never mixes samples with an older one.
export async function outDir(suite: Suite, given?: string): Promise<string> {
  const dir = resolve(
    given ?? process.env.OUT_DIR ?? join(ROOT, '.cache/benchmarks/perf-tracker', `${suite}-${utcStamp()}`),
  );
  await mkdir(dir, { recursive: true });
  return dir;
}

export function llamaServerPath(): string {
  const path = process.env.LLAMA_SERVER;
  if (!path) {
    throw new Error(`Set LLAMA_SERVER to a llama-server built at llama.cpp ${LLAMA_COMMIT.slice(0, 12)}`);
  }
  return resolve(path);
}

export interface LlamaIdentity {
  path: string;
  sha256: string;
  version: string;
  commit: string;
  dylibs: Record<string, string>;
}

// A different llama.cpp build is a different comparison; PERF_TRACKER_ANY_LLAMA=1
// records it instead of refusing.
export async function llamaIdentity(path = llamaServerPath()): Promise<LlamaIdentity> {
  // llama-server prints its version to stderr.
  const result = spawnSync(path, ['--version'], { encoding: 'utf8' });
  if (result.error) throw result.error;
  const version = `${result.stdout}${result.stderr}`;
  const commit = /commit ([0-9a-f]+)/.exec(version)?.[1] ?? '';
  if (!commit || (!LLAMA_COMMIT.startsWith(commit) && process.env.PERF_TRACKER_ANY_LLAMA !== '1')) {
    throw new Error(
      `llama-server reports commit "${commit}", expected ${LLAMA_COMMIT.slice(0, 12)}; set PERF_TRACKER_ANY_LLAMA=1 to measure another build`,
    );
  }
  // llama-server is a small launcher; the measured code is in the dylibs beside it.
  const dir = dirname(await realpath(path));
  const dylibs: Record<string, string> = {};
  for (const entry of (await readdir(dir, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    if (entry.isFile() && entry.name.endsWith('.dylib')) dylibs[entry.name] = await sha256File(join(dir, entry.name));
  }
  return { path, sha256: await sha256File(path), version: version.trim(), commit, dylibs };
}

// Any of these changes the measured path; MTL_/METAL_ also switch on Metal validation layers.
const ENV_PREFIXES = ['MLX_', 'NAPI_RS_', 'MUSE_', 'INT8_QMV_', 'GGML_', 'LLAMA_ARG_', 'MTL_', 'METAL_'];

export function assertCleanEnv(allowed: readonly string[] = []): void {
  const inherited = Object.keys(process.env).filter(
    (name) => ENV_PREFIXES.some((prefix) => name.startsWith(prefix)) && !allowed.includes(name),
  );
  if (inherited.length) throw new Error(`Unset inherited runtime variables first: ${inherited.join(', ')}`);
}

export function assertCleanTree(): { commit: string; mlx: string } {
  const status = sh('git', ['status', '--porcelain', '--ignore-submodules=none'], { cwd: ROOT });
  if (status) throw new Error(`The measured checkout must be clean:\n${status}`);
  return {
    commit: sh('git', ['rev-parse', 'HEAD'], { cwd: ROOT }),
    mlx: sh('git', ['rev-parse', 'HEAD'], { cwd: join(ROOT, 'crates/mlx-sys/mlx') }),
  };
}

export async function addonIdentity(): Promise<Record<string, string>> {
  const core = join(ROOT, 'packages/core');
  const files = ['index.cjs', 'mlx-core.darwin-arm64.node', 'mlx.metallib', 'paged_attn.metallib'];
  const identity: Record<string, string> = {};
  for (const file of files) identity[file] = await sha256File(join(core, file));
  return identity;
}

export function gpuUtilization(): number {
  const text = sh('ioreg', ['-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator']);
  const match = /"Device Utilization %"=(\d+)/.exec(text);
  if (!match) throw new Error('ioreg reports no IOAccelerator utilization');
  return Number(match[1]);
}

export interface ProcessRow {
  pid: number;
  cpu: number;
  rssMb: number;
  command: string;
}

export function busyProcesses(minCpu = 5): ProcessRow[] {
  return sh('ps', ['-Ao', 'pid=,%cpu=,rss=,command='])
    .split('\n')
    .map((line) => /^\s*(\d+)\s+([\d.]+)\s+(\d+)\s+(.*)$/.exec(line))
    .filter((match): match is RegExpExecArray => match !== null)
    .map((match) => ({
      pid: Number(match[1]),
      cpu: Number(match[2]),
      rssMb: Math.round(Number(match[3]) / 1024),
      command: match[4].slice(0, 200),
    }))
    .filter((row) => row.cpu >= minCpu && row.pid !== process.pid)
    .sort((a, b) => b.cpu - a.cpu);
}

export function hostSnapshot() {
  return {
    at: new Date().toISOString(),
    gpuUtilization: gpuUtilization(),
    loadAverage: sh('sysctl', ['-n', 'vm.loadavg']),
    power: sh('pmset', ['-g', 'batt']).split('\n')[0],
    thermal: sh('pmset', ['-g', 'therm']),
    busy: busyProcesses(),
  };
}

export function hostInfo() {
  const display = sh('system_profiler', ['SPDisplaysDataType']);
  return {
    os: `${sh('sw_vers', ['-productName'])} ${sh('sw_vers', ['-productVersion'])} (${sh('sw_vers', ['-buildVersion'])})`,
    chip: sh('sysctl', ['-n', 'machdep.cpu.brand_string']),
    gpuCores: Number(/Total Number of Cores: (\d+)/.exec(display)?.[1] ?? 0),
    memoryBytes: Number(sh('sysctl', ['-n', 'hw.memsize'])),
    node: process.version,
  };
}

export interface FixtureCase {
  name: string;
  promptTokens: number;
  sha256: string;
  tokenIds: number[];
  messages: unknown[];
  tools: unknown[] | null;
  rendered: string;
}

export interface FixtureManifest {
  id: string;
  payload: { path: string; bytes: number; sha256: string };
  cases: { name: string; promptTokens: number; tokenIdsSha256: string }[];
}

export async function loadFixture(
  name: 'gemma4-oxc-review-v1' | 'qwen38-oxc-review-v1',
): Promise<{ manifest: FixtureManifest; path: string; cases: FixtureCase[] }> {
  const manifest = await readJson<FixtureManifest>(join(ROOT, 'scripts/fixtures', `${name}.json`));
  const path = join(ROOT, manifest.payload.path);
  let data: Buffer;
  try {
    data = await readFile(path);
  } catch {
    throw new Error(`Missing ${path}; run: oxnode scripts/benchmark-fixture.ts fetch --fixture ${name.split('-')[0]}`);
  }
  if (data.length !== manifest.payload.bytes || sha256(data) !== manifest.payload.sha256) {
    throw new Error(`${path}: pinned byte length or SHA-256 mismatch`);
  }
  const cases = JSON.parse(data.toString('utf8')) as FixtureCase[];
  for (const [index, expected] of manifest.cases.entries()) {
    const actual = cases[index];
    if (actual?.name !== expected.name || sha256(JSON.stringify(actual.tokenIds)) !== expected.tokenIdsSha256) {
      throw new Error(`${expected.name}: pinned token IDs mismatch`);
    }
  }
  return { manifest, path, cases };
}

export async function assertFileSha256(path: string, expected: string): Promise<void> {
  const actual = await sha256File(path);
  if (actual !== expected) throw new Error(`${path}: SHA-256 ${actual}, expected ${expected}`);
}

export function median(values: readonly number[]): number {
  if (!values.length) throw new Error('median of no values');
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function range(values: readonly number[]): { median: number; min: number; max: number } {
  return { median: median(values), min: Math.min(...values), max: Math.max(...values) };
}

export type CsvValue = string | number | boolean | null | undefined;

export function toCsv(columns: readonly string[], rows: readonly Record<string, CsvValue>[]): string {
  const cell = (value: CsvValue) => {
    const text = value === undefined || value === null ? '' : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  return `${[columns.join(','), ...rows.map((row) => columns.map((column) => cell(row[column])).join(','))].join('\n')}\n`;
}

export function parseCsv(text: string): Record<string, string>[] {
  const [header, ...lines] = text.trim().split(/\r?\n/);
  const columns = header.split(',');
  return lines.map((line) => Object.fromEntries(line.split(',').map((value, index) => [columns[index], value])));
}

export interface ChartSeries {
  label: string;
  // One entry per group; null leaves a gap.
  values: ({ median: number; min: number; max: number } | null)[];
}

export interface ChartPanel {
  title: string;
  unit: string;
  series: ChartSeries[];
}

export interface ChartSpec {
  title: string;
  subtitle: string;
  groups: string[];
  panels: ChartPanel[];
  note: string;
}

const SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'];

function niceMax(value: number): number {
  const step = 10 ** Math.floor(Math.log10(value));
  for (const factor of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) {
    if (factor * step >= value) return factor * step;
  }
  return 10 * step;
}

const escapeXml = (text: string) => text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function formatValue(value: number): string {
  if (value === 0) return '0';
  return value >= 100 ? value.toFixed(0) : value.toFixed(1);
}

// Static SVG with an explicit light surface: GitHub embeds it as an image in
// both themes, where scripts and page CSS do not reach it.
export function renderChart(spec: ChartSpec): string {
  const seriesCount = Math.max(...spec.panels.map((panel) => panel.series.length));
  if (seriesCount > SERIES_COLORS.length) throw new Error('renderChart supports at most four series');
  const panelWidth = 460;
  const plotHeight = 260;
  const top = 116;
  const left = 64;
  const gap = 56;
  const width = left + spec.panels.length * panelWidth + (spec.panels.length - 1) * gap + 24;
  const height = top + plotHeight + 92;
  const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family="-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif">`,
    `<rect width="${width}" height="${height}" fill="#fcfcfb"/>`,
    `<text x="${left}" y="34" font-size="18" font-weight="600" fill="#0b0b0b">${escapeXml(spec.title)}</text>`,
    `<text x="${left}" y="56" font-size="13" fill="#52514e">${escapeXml(spec.subtitle)}</text>`,
  ];
  const legendSeries = spec.panels[0].series;
  let legendX = left;
  for (const [index, series] of legendSeries.entries()) {
    parts.push(
      `<rect x="${legendX}" y="72" width="12" height="12" rx="2" fill="${SERIES_COLORS[index]}"/>`,
      `<text x="${legendX + 18}" y="82" font-size="12" fill="#0b0b0b">${escapeXml(series.label)}</text>`,
    );
    legendX += 30 + series.label.length * 7;
  }
  for (const [panelIndex, panel] of spec.panels.entries()) {
    const x0 = left + panelIndex * (panelWidth + gap);
    const peak = Math.max(...panel.series.flatMap((series) => series.values.map((value) => value?.max ?? 0)));
    const yMax = niceMax(peak * 1.12);
    const y = (value: number) => top + plotHeight - (value / yMax) * plotHeight;
    parts.push(
      `<text x="${x0}" y="${top - 12}" font-size="13" font-weight="600" fill="#0b0b0b">${escapeXml(`${panel.title} · ${panel.unit}`)}</text>`,
    );
    for (let tick = 0; tick <= 4; tick++) {
      const value = (yMax * tick) / 4;
      parts.push(
        `<line x1="${x0}" x2="${x0 + panelWidth}" y1="${y(value)}" y2="${y(value)}" stroke="#e4e3df" stroke-width="1"/>`,
        `<text x="${x0 - 8}" y="${y(value) + 4}" font-size="11" text-anchor="end" fill="#52514e">${formatValue(value)}</text>`,
      );
    }
    const groupWidth = panelWidth / spec.groups.length;
    const barWidth = Math.min(30, (groupWidth - 28) / panel.series.length - 2);
    for (const [groupIndex, group] of spec.groups.entries()) {
      const groupLeft = x0 + groupIndex * groupWidth + (groupWidth - panel.series.length * (barWidth + 2)) / 2;
      for (const [seriesIndex, series] of panel.series.entries()) {
        const value = series.values[groupIndex];
        if (!value) continue;
        const bx = groupLeft + seriesIndex * (barWidth + 2);
        const cx = bx + barWidth / 2;
        const barTop = y(value.median);
        const barHeight = top + plotHeight - barTop;
        const radius = Math.min(4, barHeight / 2);
        parts.push(
          `<g><title>${escapeXml(`${series.label}, ${group}: median ${value.median.toFixed(2)} ${panel.unit} (min ${value.min.toFixed(2)}, max ${value.max.toFixed(2)})`)}</title>`,
          `<path d="M${bx},${top + plotHeight} V${barTop + radius} Q${bx},${barTop} ${bx + radius},${barTop} H${bx + barWidth - radius} Q${bx + barWidth},${barTop} ${bx + barWidth},${barTop + radius} V${top + plotHeight} Z" fill="${SERIES_COLORS[seriesIndex]}"/>`,
          `<line x1="${cx}" x2="${cx}" y1="${y(value.min)}" y2="${y(value.max)}" stroke="#0b0b0b" stroke-width="1.5"/>`,
          `<line x1="${cx - 4}" x2="${cx + 4}" y1="${y(value.min)}" y2="${y(value.min)}" stroke="#0b0b0b" stroke-width="1.5"/>`,
          `<line x1="${cx - 4}" x2="${cx + 4}" y1="${y(value.max)}" y2="${y(value.max)}" stroke="#0b0b0b" stroke-width="1.5"/>`,
          `<text x="${cx}" y="${y(value.max) - 6}" font-size="10" text-anchor="middle" fill="#0b0b0b">${formatValue(value.median)}</text></g>`,
        );
      }
      parts.push(
        `<text x="${x0 + groupIndex * groupWidth + groupWidth / 2}" y="${top + plotHeight + 20}" font-size="12" text-anchor="middle" fill="#0b0b0b">${escapeXml(group)}</text>`,
      );
    }
    parts.push(
      `<line x1="${x0}" x2="${x0 + panelWidth}" y1="${top + plotHeight}" y2="${top + plotHeight}" stroke="#52514e" stroke-width="1"/>`,
      `<text x="${x0 + panelWidth / 2}" y="${top + plotHeight + 40}" font-size="11" text-anchor="middle" fill="#52514e">Prompt tokens</text>`,
    );
  }
  parts.push(
    `<text x="${left}" y="${height - 18}" font-size="11" fill="#52514e">${escapeXml(spec.note)}</text>`,
    '</svg>',
  );
  return `${parts.join('\n')}\n`;
}

export interface CompareRow {
  label: string;
  before: number;
  after: number;
}

export function compareTable(title: string, rows: readonly CompareRow[]): string {
  const lines = [`### ${title}`, '', '| Cell | Before | After | Change |', '|---|---:|---:|---:|'];
  for (const row of rows) {
    const change = ((row.after / row.before - 1) * 100).toFixed(1);
    lines.push(
      `| ${row.label} | ${row.before.toFixed(2)} | ${row.after.toFixed(2)} | ${Number(change) > 0 ? '+' : ''}${change}% |`,
    );
  }
  return `${lines.join('\n')}\n`;
}
