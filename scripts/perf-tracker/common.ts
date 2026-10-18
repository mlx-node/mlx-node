/// <reference types="node" />

import { execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync, readdirSync, realpathSync } from 'node:fs';
import { lstat, mkdir, readdir, readFile, readlink, realpath, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
export const LLAMA_RELEASES = join(ROOT, '.cache/llama.cpp');
export const MODELS_DIR = resolve(process.env.MODELS_DIR ?? join(ROOT, '.cache/models'));

export type Suite = 'qwen38' | 'gemma4' | 'flash-next';

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

// The tracker compares the latest mlx-node main with the latest llama.cpp release.
export function assertLatestMain(): string {
  sh('git', ['fetch', '--quiet', 'origin', 'main'], { cwd: ROOT });
  const main = sh('git', ['rev-parse', 'refs/remotes/origin/main'], { cwd: ROOT });
  if (spawnSync('git', ['merge-base', '--is-ancestor', main, 'HEAD'], { cwd: ROOT }).status !== 0) {
    throw new Error(`HEAD does not contain origin/main ${main.slice(0, 12)}; rebase onto the latest main first`);
  }
  return main;
}

interface ReleaseAsset {
  name: string;
  browser_download_url: string;
  digest?: string | null;
}

// Installs the newest llama.cpp release that ships a macOS arm64 build; a release
// whose assets are still uploading is skipped for the one before it.
export async function fetchLatestLlama(): Promise<string> {
  const token = process.env.GITHUB_TOKEN;
  const response = await fetch('https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20', {
    headers: { accept: 'application/vnd.github+json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
    signal: AbortSignal.timeout(60_000),
  });
  if (!response.ok) throw new Error(`GitHub llama.cpp releases: HTTP ${response.status}`);
  const releases = (await response.json()) as { tag_name: string; draft: boolean; assets: ReleaseAsset[] }[];
  for (const release of releases) {
    const tag = release.tag_name;
    if (release.draft || !/^b\d+$/.test(tag)) continue;
    const asset = release.assets.find((item) => item.name === `llama-${tag}-bin-macos-arm64.tar.gz`);
    if (asset) return installLlama(tag, asset);
  }
  throw new Error('None of the latest 20 llama.cpp releases has a macOS arm64 build');
}

async function installLlama(tag: string, asset: ReleaseAsset): Promise<string> {
  const dir = join(LLAMA_RELEASES, tag);
  const server = join(dir, `llama-${tag}`, 'llama-server');
  if (existsSync(server)) return server;
  const expected = asset.digest?.replace(/^sha256:/, '');
  if (!expected) throw new Error(`${asset.name}: the release lists no SHA-256 digest`);
  const response = await fetch(asset.browser_download_url, { signal: AbortSignal.timeout(600_000) });
  if (!response.ok) throw new Error(`${asset.name}: HTTP ${response.status}`);
  const data = Buffer.from(await response.arrayBuffer());
  if (sha256(data) !== expected) throw new Error(`${asset.name}: SHA-256 does not match the release digest`);
  const staging = `${dir}.partial`;
  await rm(staging, { recursive: true, force: true });
  await mkdir(staging, { recursive: true });
  const archive = join(staging, asset.name);
  await writeFile(archive, data);
  const entries = sh('tar', ['-tzf', archive]).split('\n');
  if (entries.some((entry) => !entry.startsWith(`llama-${tag}/`) || entry.split('/').includes('..'))) {
    throw new Error(`${asset.name}: unexpected archive layout`);
  }
  sh('tar', ['-xzf', archive, '-C', staging]);
  await rm(archive);
  await rename(staging, dir);
  return server;
}

// The newest release fetchLatestLlama() installed; every suite's preflight fetches it first.
export function llamaServerPath(): string {
  const tags = existsSync(LLAMA_RELEASES) ? readdirSync(LLAMA_RELEASES).filter((name) => /^b\d+$/.test(name)) : [];
  const tag = tags.sort((a, b) => Number(b.slice(1)) - Number(a.slice(1)))[0];
  if (!tag) throw new Error(`No llama.cpp release in ${LLAMA_RELEASES}; run a suite's preflight step first`);
  return join(LLAMA_RELEASES, tag, `llama-${tag}`, 'llama-server');
}

// Every measured request is a single cold request. Context checkpoints and the RAM prompt cache
// only help a later request reuse the prompt, yet the checkpoints copy state twice in each timed prefill.
export const LLAMA_SINGLE_REQUEST_ARGS = ['-ctxcp', '0', '-cram', '0'] as const;

export interface PinnedFile {
  file: string;
  bytes: number;
  sha256: string;
}

export interface DraftPin {
  repo: string;
  revision: string;
  files: PinnedFile[];
}

export type DraftPins = Record<'mlx-node' | 'llama.cpp', DraftPin>;

export function assertPin(path: string, size: number, sha256: string | undefined, pin: PinnedFile): void {
  if (size !== pin.bytes || sha256 !== pin.sha256) {
    throw new Error(`${path}: ${size} bytes, SHA-256 ${sha256}; pinned ${pin.bytes} bytes, SHA-256 ${pin.sha256}`);
  }
}

export interface SpeculationRule {
  // The type llama.cpp lists in generation_settings["speculative.types"].
  llamaType: string;
  // Tokens d every mlx-node cycle drafts. Only the length tail drafts fewer: a cycle there drafts
  // remaining − 1, a distinct value below d each time, so the run falls short by at most d(d − 1)/2.
  mlxDraftsPerCycle: number;
}

export interface MlxSpeculationRequest {
  enableMtp: true;
  mtpDepth?: number;
  mtpAdaptiveDepth: false;
}

export interface MlxSpeculationPerformance {
  mtpCycles?: number;
  mtpMeanDepth?: number;
  mtpMeanAcceptedTokens?: number;
  mtpMeanAcceptedTokensTotal?: number;
}

export interface Acceptance {
  // llama.cpp's /completion does not report its verify steps.
  cycles: number | null;
  drafted: number;
  accepted: number;
  // Tokens committed per verify step, the verified token included.
  committedPerCycle: number | null;
}

const isCount = (value: unknown): value is number => Number.isSafeInteger(value) && (value as number) >= 0;

// Both runtimes fall back to plain decoding when a draft is missing or fails to start.
// mlx-node reports per-cycle means of integer totals, so its counts are mean × cycles.
export function mlxAcceptance(
  performance: MlxSpeculationPerformance,
  rule: SpeculationRule,
  label: string,
): Acceptance {
  const {
    mtpCycles: cycles,
    mtpMeanDepth: depth,
    mtpMeanAcceptedTokens: mean,
    mtpMeanAcceptedTokensTotal,
  } = performance;
  if (!isCount(cycles) || cycles === 0) {
    throw new Error(`${label}: mlx-node did not speculate (mtpCycles ${cycles})`);
  }
  const total = (name: string, value: number | undefined) => {
    const product = (typeof value === 'number' ? value : Number.NaN) * cycles;
    const count = Math.round(product);
    if (!isCount(count) || Math.abs(product - count) > 1e-6) {
      throw new Error(`${label}: mlx-node ${name} ${value} over ${cycles} cycles is no token count`);
    }
    return count;
  };
  const drafted = total('mtpMeanDepth', depth);
  const accepted = total('mtpMeanAcceptedTokens', mean);
  const most = rule.mlxDraftsPerCycle * cycles;
  const least = most - (rule.mlxDraftsPerCycle * (rule.mlxDraftsPerCycle - 1)) / 2;
  if (drafted > most || drafted < least) {
    throw new Error(
      `${label}: mlx-node drafted ${drafted} tokens in ${cycles} cycles; ${rule.mlxDraftsPerCycle} per cycle drafts ${least}–${most}`,
    );
  }
  if (accepted > drafted) {
    throw new Error(`${label}: mlx-node accepted ${accepted} of ${drafted} drafted tokens`);
  }
  // Native adds 1.0 to the same f64 ratio, so the two agree exactly.
  if (mtpMeanAcceptedTokensTotal !== (mean as number) + 1) {
    throw new Error(
      `${label}: mlx-node mtpMeanAcceptedTokensTotal ${mtpMeanAcceptedTokensTotal} is not mtpMeanAcceptedTokens ${mean} + 1`,
    );
  }
  return { cycles, drafted, accepted, committedPerCycle: mtpMeanAcceptedTokensTotal };
}

export function llamaAcceptance(
  timings: { draft_n?: number; draft_n_accepted?: number },
  speculativeTypes: string,
  rule: SpeculationRule,
  label: string,
): Acceptance {
  const { draft_n: drafted, draft_n_accepted: accepted } = timings;
  if (!speculativeTypes.split(',').includes(rule.llamaType) || !isCount(drafted) || drafted === 0) {
    throw new Error(
      `${label}: llama.cpp did not speculate (speculative.types "${speculativeTypes}", draft_n ${drafted})`,
    );
  }
  if (!isCount(accepted) || accepted > drafted) {
    throw new Error(`${label}: llama.cpp draft_n_accepted ${accepted} of draft_n ${drafted}`);
  }
  return { cycles: null, drafted, accepted, committedPerCycle: null };
}

type Range = { median: number; min: number; max: number };

export interface AcceptanceStats {
  drafted: Range;
  accepted: Range;
  acceptanceRate: Range;
  cycles: Range | null;
  committedPerCycle: Range | null;
}

export function acceptanceStats(items: readonly Acceptance[]): AcceptanceStats {
  const known = (values: (number | null)[]) =>
    values.every((value): value is number => value !== null) ? range(values) : null;
  return {
    drafted: range(items.map(({ drafted }) => drafted)),
    accepted: range(items.map(({ accepted }) => accepted)),
    acceptanceRate: range(items.map(({ drafted, accepted }) => accepted / drafted)),
    cycles: known(items.map(({ cycles }) => cycles)),
    committedPerCycle: known(items.map(({ committedPerCycle }) => committedPerCycle)),
  };
}

export const ACCEPTANCE_CSV_COLUMNS = ['draft_cycles', 'drafted_tokens', 'accepted_drafts'] as const;

export const acceptanceCsv = ({ cycles, drafted, accepted }: Acceptance) => ({
  draft_cycles: cycles,
  drafted_tokens: drafted,
  accepted_drafts: accepted,
});

export function acceptanceTable(rows: readonly { label: string; stats: AcceptanceStats }[]): string {
  const lines = [
    '| Cell | Drafted | Accepted | Acceptance | Tokens per verify step |',
    '|---|---:|---:|---:|---:|',
    ...rows.map(
      ({ label, stats }) =>
        `| ${label} | ${stats.drafted.median.toFixed(0)} | ${stats.accepted.median.toFixed(0)} | ${(stats.acceptanceRate.median * 100).toFixed(1)}% | ${stats.committedPerCycle?.median.toFixed(2) ?? 'not reported'} |`,
    ),
  ];
  return `${lines.join('\n')}\n`;
}

export interface LlamaIdentity {
  path: string;
  sha256: string;
  version: string;
  build: number;
  commit: string;
  dylibs: Record<string, string>;
}

export async function llamaIdentity(path = llamaServerPath()): Promise<LlamaIdentity> {
  // llama-server prints its version to stderr.
  const result = spawnSync(path, ['--version'], { encoding: 'utf8' });
  if (result.error) throw result.error;
  const version = `${result.stdout}${result.stderr}`;
  const match = /build (\d+), commit ([0-9a-f]+)/.exec(version);
  if (!match) throw new Error(`${path}: unexpected --version output: ${version.trim()}`);
  // llama-server is a small launcher; the measured code is in the dylibs beside it.
  const dir = dirname(await realpath(path));
  const dylibs: Record<string, string> = {};
  for (const entry of (await readdir(dir, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    if (entry.isFile() && entry.name.endsWith('.dylib')) dylibs[entry.name] = await sha256File(join(dir, entry.name));
  }
  return {
    path,
    sha256: await sha256File(path),
    version: version.trim(),
    build: Number(match[1]),
    commit: match[2],
    dylibs,
  };
}

// Any of these changes the measured path; MTL_/METAL_ switch on Metal validation layers and
// DYLD_ redirects the libraries of both runtimes.
const ENV_PREFIXES = ['MLX_', 'NAPI_RS_', 'INT8_QMV_', 'GGML_', 'LLAMA_ARG_', 'MTL_', 'METAL_', 'DYLD_'];

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

export interface TreeIdentity {
  commit: string;
  mlx: string;
  dirty: boolean;
  status?: string;
  patch?: string;
  untracked?: Record<string, string>;
}

// A resumed dirty run must still measure the same patch, so hash content, not the file list.
export async function treeIdentity(allowDirty: boolean): Promise<TreeIdentity> {
  try {
    return { ...assertCleanTree(), dirty: false };
  } catch (error) {
    if (!allowDirty || !(error instanceof Error) || !error.message.startsWith('The measured checkout must be clean')) {
      throw error;
    }
    const untracked: Record<string, string> = {};
    const listed = sh('git', ['ls-files', '--others', '--exclude-standard', '-z'], { cwd: ROOT });
    for (const path of listed.split('\0').filter(Boolean).sort()) {
      const full = join(ROOT, path);
      untracked[path] = (await lstat(full)).isSymbolicLink() ? `-> ${await readlink(full)}` : await sha256File(full);
    }
    const diff = ['diff', 'HEAD', '--binary', '--no-ext-diff', '--no-textconv', '--no-color'];
    return {
      commit: sh('git', ['rev-parse', 'HEAD'], { cwd: ROOT }),
      mlx: sh('git', ['rev-parse', 'HEAD'], { cwd: join(ROOT, 'crates/mlx-sys/mlx') }),
      dirty: true,
      status: sh('git', ['status', '--porcelain', '--ignore-submodules=none'], { cwd: ROOT }),
      patch: sha256(sh('git', diff, { cwd: ROOT })),
      untracked,
    };
  }
}

export async function fileHashes(dir: string, names: readonly string[]): Promise<Record<string, string>> {
  const hashes: Record<string, string> = {};
  for (const name of names) {
    if (existsSync(join(dir, name))) hashes[name] = await sha256File(join(dir, name));
  }
  return hashes;
}

export async function addonIdentity(): Promise<Record<string, string>> {
  const core = join(ROOT, 'packages/core');
  const files = ['index.cjs', 'mlx-core.darwin-arm64.node', 'mlx.metallib', 'paged_attn.metallib'];
  const identity: Record<string, string> = {};
  for (const file of files) identity[file] = await sha256File(join(core, file));
  return identity;
}

export const ADDON_PATH = join(ROOT, 'packages/core/mlx-core.darwin-arm64.node');

// index.cjs tries darwin-universal bindings before the arm64 file, so a stale
// universal binary would load while the provenance hashes the arm64 one.
export function assertLoadedAddon(expected = ADDON_PATH): void {
  const report = process.report.getReport() as { sharedObjects?: string[] };
  const addons = (report.sharedObjects ?? []).filter((path) => /mlx-core\.[^/]*\.node$/.test(path));
  if (addons.length !== 1 || realpathSync(addons[0]) !== realpathSync(expected)) {
    throw new Error(`Expected the addon ${expected}, loaded: ${addons.join(', ') || 'none'}`);
  }
}

export function assertHostReady(port: number): void {
  const power = sh('pmset', ['-g', 'batt']).split('\n')[0];
  if (!power.includes("'AC Power'")) throw new Error(`Measure on AC power: ${power}`);
  if (spawnSync('lsof', ['-nP', `-iTCP:${port}`, '-sTCP:LISTEN']).status === 0) throw new Error(`Port ${port} is busy`);
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
  name: 'gemma4-oxc-review-v2' | 'qwen38-oxc-review-v2',
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
